import CoreML
import Dispatch
import Foundation

extension TMSileroVAD {
    public struct SileroVADConfig {
        public let variant: TMSileroVADVariant
        public let startThreshold: Float
        public let endThreshold: Float
        public let minSpeechDurationMs: Int
        public let minSilenceDurationMs: Int
        public let computeUnits: MLComputeUnits
        public let callbackQueue: DispatchQueue

        public init(
            variant: TMSileroVADVariant,
            startThreshold: Float,
            endThreshold: Float,
            minSpeechDurationMs: Int,
            minSilenceDurationMs: Int,
            computeUnits: MLComputeUnits = .all,
            callbackQueue: DispatchQueue = .main
        ) {
            self.variant = variant
            self.startThreshold = startThreshold
            self.endThreshold = endThreshold
            self.minSpeechDurationMs = minSpeechDurationMs
            self.minSilenceDurationMs = minSilenceDurationMs
            self.computeUnits = computeUnits
            self.callbackQueue = callbackQueue
        }

        /// 256 ms variant. Endpointing / ASR pre-processing default.
        public static let balanced = SileroVADConfig(
            variant: .balanced256ms,
            startThreshold: 0.5,
            endThreshold: 0.35,
            minSpeechDurationMs: 256,
            minSilenceDurationMs: 768
        )

        /// 32 ms variant. Barge-in / interruption detection default.
        public static let realtime = SileroVADConfig(
            variant: .realtime32ms,
            startThreshold: 0.5,
            endThreshold: 0.35,
            minSpeechDurationMs: 160,
            minSilenceDurationMs: 640
        )
    }
}
