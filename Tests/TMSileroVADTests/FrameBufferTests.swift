import XCTest
@testable import TMSileroVAD

final class FrameBufferTests: XCTestCase {
    func test_under_one_frame_yields_no_chunk() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 512)
        let chunks = buffer.append(Array(repeating: 0.1, count: 100))
        XCTAssertEqual(chunks.count, 0)
    }

    func test_exactly_one_frame_yields_one_chunk() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 512)
        let chunks = buffer.append(Array(repeating: 0.1, count: 512))
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].count, 512)
    }

    func test_two_and_a_half_frames_yields_two_chunks_with_remainder() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 512)
        let chunks = buffer.append(Array(repeating: 0.1, count: 512 * 2 + 256))
        XCTAssertEqual(chunks.count, 2)
        let next = buffer.append(Array(repeating: 0.1, count: 256))
        XCTAssertEqual(next.count, 1)
    }

    func test_multiple_appends_concatenate() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 512)
        XCTAssertEqual(buffer.append(Array(repeating: 0.1, count: 300)).count, 0)
        let chunks = buffer.append(Array(repeating: 0.2, count: 300))
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0][0], 0.1, accuracy: 1e-6)
        XCTAssertEqual(chunks[0][299], 0.1, accuracy: 1e-6)
        XCTAssertEqual(chunks[0][300], 0.2, accuracy: 1e-6)
    }

    func test_reset_drops_pending() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 512)
        _ = buffer.append(Array(repeating: 0.1, count: 300))
        buffer.reset()
        let chunks = buffer.append(Array(repeating: 0.2, count: 512))
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0][0], 0.2, accuracy: 1e-6, "reset must drop pending samples")
    }

    func test_supports_4096_chunk_for_balanced_variant() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 4096)
        let chunks = buffer.append(Array(repeating: 0.1, count: 4096 * 3))
        XCTAssertEqual(chunks.count, 3)
        XCTAssertEqual(chunks[0].count, 4096)
    }
}
