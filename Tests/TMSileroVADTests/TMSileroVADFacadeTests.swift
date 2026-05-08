import XCTest
import AVFoundation
@testable import TMSileroVAD

final class TMSileroVADFacadeTests: XCTestCase {
    func test_processPCM_returns_events_for_each_chunk() throws {
        let mock = MockModelRunner()
        mock.probabilities = [0.9, 0.9, 0.1, 0.1]
        let vad = try TMSileroVAD.makeForTesting(
            config: .realtime,
            runner: mock,
            spec: TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        )
        let chunkLen = 512
        let samples = [Float](repeating: 0.05, count: chunkLen * 4)
        let events = try vad.processPCM(samples)
        XCTAssertEqual(events.count, 4)
    }

    func test_processPCM_buffers_partial_input() throws {
        let mock = MockModelRunner()
        mock.probabilities = [0.1, 0.1]
        let vad = try TMSileroVAD.makeForTesting(
            config: .realtime,
            runner: mock,
            spec: TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        )
        let events1 = try vad.processPCM([Float](repeating: 0, count: 300))
        XCTAssertEqual(events1.count, 0, "less than one chunk → no events")
        let events2 = try vad.processPCM([Float](repeating: 0, count: 300))
        XCTAssertEqual(events2.count, 1, "300+300 = 600 → one 512-chunk")
    }

    func test_processAudioBuffer_resamples_48k_to_16k_and_processes() throws {
        let mock = MockModelRunner()
        mock.probabilities = Array(repeating: 0.1, count: 100)
        let vad = try TMSileroVAD.makeForTesting(
            config: .realtime,
            runner: mock,
            spec: TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        )
        // 1 second of silence at 48k → ~16000 samples → ~31 chunks of 512
        let buffer = PCMSynth.silence(seconds: 1.0, sampleRate: 48_000)
        let events = try vad.processAudioBuffer(buffer)
        XCTAssertGreaterThanOrEqual(events.count, 30)
        XCTAssertLessThanOrEqual(events.count, 32)
    }

    func test_delegate_called_with_same_events_on_configured_queue() throws {
        let mock = MockModelRunner()
        mock.probabilities = [0.1, 0.1, 0.1]
        let queue = DispatchQueue(label: "test.callback")
        let queueKey = DispatchSpecificKey<Int>()
        queue.setSpecific(key: queueKey, value: 42)

        let baseConfig = TMSileroVADConfig.realtime
        let config = TMSileroVADConfig(
            variant: baseConfig.variant,
            startThreshold: baseConfig.startThreshold,
            endThreshold: baseConfig.endThreshold,
            minSpeechDurationMs: baseConfig.minSpeechDurationMs,
            minSilenceDurationMs: baseConfig.minSilenceDurationMs,
            computeUnits: baseConfig.computeUnits,
            callbackQueue: queue
        )

        let vad = try TMSileroVAD.makeForTesting(
            config: config,
            runner: mock,
            spec: TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        )
        let spy = DelegateSpy(queueKey: queueKey)
        vad.delegate = spy
        let exp = expectation(description: "delegate fires")
        spy.onEvent = { _ in
            if spy.events.count == 3 {
                exp.fulfill()
            }
        }
        _ = try vad.processPCM([Float](repeating: 0, count: 512 * 3))
        wait(for: [exp], timeout: 1.0)
        XCTAssertEqual(spy.events.count, 3)
        XCTAssertTrue(spy.allOnExpectedQueue, "delegate must run on configured queue")
    }

    func test_reset_clears_buffer_and_runner_and_state_machine() throws {
        let mock = MockModelRunner()
        mock.probabilities = Array(repeating: 0.1, count: 100)
        let vad = try TMSileroVAD.makeForTesting(
            config: .realtime,
            runner: mock,
            spec: TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        )
        _ = try vad.processPCM([Float](repeating: 0, count: 300))
        vad.reset()
        XCTAssertEqual(mock.resetCalls, 1)
        // After reset the framebuffer should be empty: feeding 300 again should yield zero events.
        let events = try vad.processPCM([Float](repeating: 0, count: 300))
        XCTAssertEqual(events.count, 0)
    }
}

private final class DelegateSpy: TMSileroVADDelegate {
    let queueKey: DispatchSpecificKey<Int>
    var events: [TMSileroVADEvent] = []
    var onEvent: ((TMSileroVADEvent) -> Void)?
    var allOnExpectedQueue = true

    init(queueKey: DispatchSpecificKey<Int>) {
        self.queueKey = queueKey
    }

    func sileroVAD(_ vad: TMSileroVAD, didEmit event: TMSileroVADEvent) {
        if DispatchQueue.getSpecific(key: queueKey) != 42 {
            allOnExpectedQueue = false
        }
        events.append(event)
        onEvent?(event)
    }
}
