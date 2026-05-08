import XCTest
import CoreML
@testable import TMSileroVAD

final class CoreMLRunnerIntegrationTests: XCTestCase {
    func test_balanced_runner_loads_and_predicts_silence() throws {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .balanced256ms)
        let runner = try TMSileroVADCoreMLRunner(spec: spec, computeUnits: .cpuOnly)
        let silence = Array(repeating: Float(0), count: spec.chunkLength)
        let prob = try runner.predict(chunk: silence)
        XCTAssertGreaterThanOrEqual(prob, 0)
        XCTAssertLessThanOrEqual(prob, 1)
        XCTAssertLessThan(prob, 0.5, "silence should produce low speech probability")
    }

    func test_realtime_runner_loads_and_predicts() throws {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        let runner = try TMSileroVADCoreMLRunner(spec: spec, computeUnits: .cpuOnly)
        let silence = Array(repeating: Float(0), count: spec.chunkLength)
        let prob = try runner.predict(chunk: silence)
        XCTAssertGreaterThanOrEqual(prob, 0)
        XCTAssertLessThanOrEqual(prob, 1)
    }

    func test_state_persists_across_calls_and_reset_clears_it() throws {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        let runner = try TMSileroVADCoreMLRunner(spec: spec, computeUnits: .cpuOnly)

        var noise = [Float](repeating: 0, count: spec.chunkLength)
        for i in 0..<noise.count {
            noise[i] = Float.random(in: -0.3...0.3)
        }
        _ = try runner.predict(chunk: noise)
        _ = try runner.predict(chunk: noise)
        let probWithState = try runner.predict(chunk: Array(repeating: 0, count: spec.chunkLength))

        runner.reset()
        let probAfterReset = try runner.predict(chunk: Array(repeating: 0, count: spec.chunkLength))

        // Hidden/cell state being non-zero vs zero typically yields a different output for
        // an identical input. Exact values are model-dependent; assert non-equality.
        XCTAssertGreaterThan(abs(probWithState - probAfterReset), 1e-7,
                             "state must influence output: \(probWithState) vs \(probAfterReset)")
    }

    func test_invalid_chunk_length_throws() throws {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        let runner = try TMSileroVADCoreMLRunner(spec: spec, computeUnits: .cpuOnly)
        XCTAssertThrowsError(try runner.predict(chunk: [Float](repeating: 0, count: 100))) { error in
            guard case TMSileroVADError.invalidFrameLength(let expected, let got) = error else {
                XCTFail("wrong error: \(error)")
                return
            }
            XCTAssertEqual(expected, spec.chunkLength)
            XCTAssertEqual(got, 100)
        }
    }
}
