import Foundation

final class TMSileroVADStateMachine {
    private enum State {
        case silence
        case speechCandidate
        case speaking
        case silenceCandidate
    }

    private let frameDurationMs: Int
    private let startThreshold: Float
    private let endThreshold: Float
    private let minSpeechDurationMs: Int
    private let minSilenceDurationMs: Int

    private var state: State = .silence
    private var speechDurationMs = 0
    private var silenceDurationMs = 0

    init(
        frameDurationMs: Int,
        startThreshold: Float,
        endThreshold: Float,
        minSpeechDurationMs: Int,
        minSilenceDurationMs: Int
    ) {
        self.frameDurationMs = frameDurationMs
        self.startThreshold = startThreshold
        self.endThreshold = endThreshold
        self.minSpeechDurationMs = minSpeechDurationMs
        self.minSilenceDurationMs = minSilenceDurationMs
    }

    func reset() {
        state = .silence
        speechDurationMs = 0
        silenceDurationMs = 0
    }

    func process(probability: Float) -> TMSileroVADEvent {
        switch state {
        case .silence:
            guard probability >= startThreshold else {
                return .silence(probability: probability)
            }
            speechDurationMs = frameDurationMs
            if speechDurationMs >= minSpeechDurationMs {
                silenceDurationMs = 0
                state = .speaking
                return .speechStarted(probability: probability)
            }
            state = .speechCandidate
            return .silence(probability: probability)

        case .speechCandidate:
            guard probability >= startThreshold else {
                speechDurationMs = 0
                state = .silence
                return .silence(probability: probability)
            }
            speechDurationMs += frameDurationMs
            guard speechDurationMs >= minSpeechDurationMs else {
                return .silence(probability: probability)
            }
            silenceDurationMs = 0
            state = .speaking
            return .speechStarted(probability: probability)

        case .speaking:
            guard probability < endThreshold else {
                silenceDurationMs = 0
                return .speechContinuing(probability: probability)
            }
            silenceDurationMs = frameDurationMs
            if silenceDurationMs >= minSilenceDurationMs {
                speechDurationMs = 0
                silenceDurationMs = 0
                state = .silence
                return .speechEnded(probability: probability)
            }
            state = .silenceCandidate
            return .speechContinuing(probability: probability)

        case .silenceCandidate:
            if probability >= endThreshold {
                silenceDurationMs = 0
                state = .speaking
                return .speechContinuing(probability: probability)
            }
            silenceDurationMs += frameDurationMs
            guard silenceDurationMs >= minSilenceDurationMs else {
                return .speechContinuing(probability: probability)
            }
            speechDurationMs = 0
            silenceDurationMs = 0
            state = .silence
            return .speechEnded(probability: probability)
        }
    }
}
