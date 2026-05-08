import Foundation

protocol TMSileroVADModelRunning: AnyObject {
    /// Run one inference. `chunk.count` MUST equal `spec.chunkLength`.
    /// Returns the speech probability in [0, 1].
    func predict(chunk: [Float]) throws -> Float

    /// Reset LSTM hidden/cell state and 64-sample audio context.
    func reset()
}
