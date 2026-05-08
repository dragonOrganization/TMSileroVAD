import Foundation

public protocol TMSileroVADDelegate: AnyObject {
    /// Called for every chunk processed, on the queue configured in `TMSileroVADConfig.callbackQueue`.
    func sileroVAD(_ vad: TMSileroVAD, didEmit event: TMSileroVADEvent)
}
