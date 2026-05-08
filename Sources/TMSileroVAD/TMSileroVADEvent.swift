import Foundation

extension TMSileroVAD {
    public enum SileroVADEvent: Equatable, Sendable {
        case silence(probability: Float)
        case speechStarted(probability: Float)
        case speechContinuing(probability: Float)
        case speechEnded(probability: Float)

        public var probability: Float {
            switch self {
            case .silence(let probability),
                 .speechStarted(let probability),
                 .speechContinuing(let probability),
                 .speechEnded(let probability):
                return probability
            }
        }
    }
}
