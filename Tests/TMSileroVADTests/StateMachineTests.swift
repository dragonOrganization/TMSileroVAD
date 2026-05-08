import XCTest
@testable import TMSileroVAD

final class StateMachineTests: XCTestCase {
    private func make(
        frameMs: Int = 32,
        start: Float = 0.5,
        end: Float = 0.35,
        minSpeech: Int = 96,
        minSilence: Int = 320
    ) -> TMSileroVADStateMachine {
        TMSileroVADStateMachine(
            frameDurationMs: frameMs,
            startThreshold: start,
            endThreshold: end,
            minSpeechDurationMs: minSpeech,
            minSilenceDurationMs: minSilence
        )
    }

    func test_below_start_threshold_stays_silent() {
        let sm = make()
        let event = sm.process(probability: 0.1)
        XCTAssertEqual(event, .silence(probability: 0.1))
    }

    func test_speech_started_only_after_min_duration() {
        let sm = make(frameMs: 32, minSpeech: 96) // 3 frames
        XCTAssertEqual(sm.process(probability: 0.9), .silence(probability: 0.9), "frame 1 still candidate")
        XCTAssertEqual(sm.process(probability: 0.9), .silence(probability: 0.9), "frame 2 still candidate")
        XCTAssertEqual(sm.process(probability: 0.9), .speechStarted(probability: 0.9), "frame 3 fires")
    }

    func test_speech_candidate_drops_back_to_silence_on_low_prob() {
        let sm = make(frameMs: 32, minSpeech: 96)
        _ = sm.process(probability: 0.9)
        XCTAssertEqual(sm.process(probability: 0.1), .silence(probability: 0.1))
        XCTAssertEqual(sm.process(probability: 0.9), .silence(probability: 0.9))
    }

    func test_speaking_emits_continuing_above_end_threshold() {
        let sm = make(frameMs: 32, minSpeech: 32)
        XCTAssertEqual(sm.process(probability: 0.9), .speechStarted(probability: 0.9))
        XCTAssertEqual(sm.process(probability: 0.6), .speechContinuing(probability: 0.6))
        XCTAssertEqual(sm.process(probability: 0.4), .speechContinuing(probability: 0.4),
                       "above end threshold (0.35) so still continuing")
    }

    func test_speech_ended_only_after_min_silence() {
        let sm = make(frameMs: 32, minSpeech: 32, minSilence: 96)
        _ = sm.process(probability: 0.9) // speechStarted
        XCTAssertEqual(sm.process(probability: 0.1), .speechContinuing(probability: 0.1), "frame 1 below end")
        XCTAssertEqual(sm.process(probability: 0.1), .speechContinuing(probability: 0.1), "frame 2")
        XCTAssertEqual(sm.process(probability: 0.1), .speechEnded(probability: 0.1), "frame 3 fires end")
    }

    func test_silence_candidate_recovers_to_speaking_on_high_prob() {
        let sm = make(frameMs: 32, minSpeech: 32, minSilence: 96)
        _ = sm.process(probability: 0.9)
        _ = sm.process(probability: 0.1)
        XCTAssertEqual(sm.process(probability: 0.9), .speechContinuing(probability: 0.9))
        _ = sm.process(probability: 0.1)
        _ = sm.process(probability: 0.1)
        XCTAssertEqual(sm.process(probability: 0.1), .speechEnded(probability: 0.1))
    }

    func test_reset_returns_to_silence() {
        let sm = make(frameMs: 32, minSpeech: 32)
        _ = sm.process(probability: 0.9) // speechStarted
        sm.reset()
        XCTAssertEqual(sm.process(probability: 0.4), .silence(probability: 0.4),
                       "after reset, sub-threshold prob is silence")
    }
}
