import Foundation

public enum TMSileroVADError: Error, Equatable {
    case modelNotFound(name: String)
    case modelLoadFailed(underlying: String)
    case invalidAudioFormat(reason: String)
    case invalidFrameLength(expected: Int, got: Int)
    case invalidModelOutput(reason: String)
    case resamplerFailed(reason: String)
}
