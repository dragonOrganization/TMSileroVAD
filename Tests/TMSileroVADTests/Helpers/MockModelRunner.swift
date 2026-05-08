import Foundation
@testable import TMSileroVAD

final class MockModelRunner: TMSileroVADModelRunning {
    var probabilities: [Float] = []
    private(set) var predictCalls = 0
    private(set) var resetCalls = 0

    func predict(chunk: [Float]) throws -> Float {
        defer { predictCalls += 1 }
        guard predictCalls < probabilities.count else {
            return 0
        }
        return probabilities[predictCalls]
    }

    func reset() {
        resetCalls += 1
    }
}
