import Foundation

final class TMSileroVADFrameBuffer {
    private let frameLength: Int
    private var pendingSamples: [Float] = []

    init(frameLength: Int) {
        self.frameLength = frameLength
        self.pendingSamples.reserveCapacity(frameLength * 2)
    }

    func reset() {
        pendingSamples.removeAll(keepingCapacity: true)
    }

    func append(_ samples: [Float]) -> [[Float]] {
        pendingSamples.append(contentsOf: samples)

        var chunks: [[Float]] = []
        while pendingSamples.count >= frameLength {
            chunks.append(Array(pendingSamples.prefix(frameLength)))
            pendingSamples.removeFirst(frameLength)
        }
        return chunks
    }
}
