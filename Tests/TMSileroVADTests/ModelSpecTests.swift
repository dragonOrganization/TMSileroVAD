import XCTest
@testable import TMSileroVAD

final class ModelSpecTests: XCTestCase {
    func test_balanced256ms_spec_has_4096_chunk_and_4160_input() {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .balanced256ms)
        XCTAssertEqual(spec.modelName, "silero-vad-unified-256ms-v6.0.0")
        XCTAssertEqual(spec.contextLength, 64)
        XCTAssertEqual(spec.chunkLength, 4096)
        XCTAssertEqual(spec.inputLength, 4160)
        XCTAssertEqual(spec.frameDurationMs, 256)
        XCTAssertEqual(spec.stateLength, 128)
        XCTAssertEqual(spec.sampleRate, 16_000)
    }

    func test_realtime32ms_spec_has_512_chunk_and_576_input() {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        XCTAssertEqual(spec.modelName, "silero-vad-unified-v6.0.0")
        XCTAssertEqual(spec.contextLength, 64)
        XCTAssertEqual(spec.chunkLength, 512)
        XCTAssertEqual(spec.inputLength, 576)
        XCTAssertEqual(spec.frameDurationMs, 32)
        XCTAssertEqual(spec.stateLength, 128)
        XCTAssertEqual(spec.sampleRate, 16_000)
    }
}
