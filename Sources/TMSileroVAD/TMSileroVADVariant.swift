import Foundation

public enum TMSileroVADVariant: Sendable, Equatable {
    /// 256 ms / 4096-sample model with internal noisy-OR aggregation.
    /// Use for endpointing, ASR pre-processing, push-to-talk replacement.
    case balanced256ms

    /// 32 ms / 512-sample model. Use for AI barge-in, low-latency speech start detection.
    case realtime32ms
}
