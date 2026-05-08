import Foundation

struct TMSileroVADModelSpec {
    let modelName: String
    let contextLength: Int
    let chunkLength: Int
    let inputLength: Int
    let frameDurationMs: Int
    let stateLength: Int
    let sampleRate: Int

    static func makeSpec(variant: TMSileroVADVariant) -> TMSileroVADModelSpec {
        switch variant {
        case .balanced256ms:
            return TMSileroVADModelSpec(
                modelName: "silero-vad-unified-256ms-v6.0.0",
                contextLength: 64,
                chunkLength: 4096,
                inputLength: 4160,
                frameDurationMs: 256,
                stateLength: 128,
                sampleRate: 16_000
            )
        case .realtime32ms:
            return TMSileroVADModelSpec(
                modelName: "silero-vad-unified-v6.0.0",
                contextLength: 64,
                chunkLength: 512,
                inputLength: 576,
                frameDurationMs: 32,
                stateLength: 128,
                sampleRate: 16_000
            )
        }
    }
}
