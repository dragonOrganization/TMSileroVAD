import Accelerate
import CoreML
import Foundation

final class TMSileroVADCoreMLRunner: TMSileroVADModelRunning {
    private enum CoreMLLayout {
        static let audioInput = "audio_input"
        static let hiddenInput = "hidden_state"
        static let cellInput = "cell_state"
        static let vadOutput = "vad_output"
        static let newHiddenOutput = "new_hidden_state"
        static let newCellOutput = "new_cell_state"
    }

    private let spec: TMSileroVADModelSpec
    private let model: MLModel

    private let audioInputArray: MLMultiArray
    private let hiddenStateArray: MLMultiArray
    private let cellStateArray: MLMultiArray
    private var contextSamples: [Float]

    init(spec: TMSileroVADModelSpec, computeUnits: MLComputeUnits) throws {
        self.spec = spec

        let url: URL
        do {
            url = try TMSileroVADResources.modelURL(forName: spec.modelName)
        } catch {
            throw TMSileroVADError.modelNotFound(name: spec.modelName)
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        do {
            self.model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw TMSileroVADError.modelLoadFailed(underlying: error.localizedDescription)
        }

        do {
            self.audioInputArray = try MLMultiArray(
                shape: [1, NSNumber(value: spec.inputLength)],
                dataType: .float32
            )
            self.hiddenStateArray = try MLMultiArray(
                shape: [1, NSNumber(value: spec.stateLength)],
                dataType: .float32
            )
            self.cellStateArray = try MLMultiArray(
                shape: [1, NSNumber(value: spec.stateLength)],
                dataType: .float32
            )
        } catch {
            throw TMSileroVADError.modelLoadFailed(underlying: "MLMultiArray alloc failed: \(error.localizedDescription)")
        }

        self.contextSamples = [Float](repeating: 0, count: spec.contextLength)
        Self.zero(array: hiddenStateArray, count: spec.stateLength)
        Self.zero(array: cellStateArray, count: spec.stateLength)
    }

    func reset() {
        Self.zero(array: hiddenStateArray, count: spec.stateLength)
        Self.zero(array: cellStateArray, count: spec.stateLength)
        for i in 0..<contextSamples.count {
            contextSamples[i] = 0
        }
    }

    func predict(chunk: [Float]) throws -> Float {
        guard chunk.count == spec.chunkLength else {
            throw TMSileroVADError.invalidFrameLength(expected: spec.chunkLength, got: chunk.count)
        }

        // Fill audio_input: [context_64 | chunk_N]
        let audioPtr = audioInputArray.dataPointer.assumingMemoryBound(to: Float.self)
        contextSamples.withUnsafeBufferPointer { ctxBuf in
            audioPtr.update(from: ctxBuf.baseAddress!, count: spec.contextLength)
        }
        chunk.withUnsafeBufferPointer { chBuf in
            (audioPtr + spec.contextLength).update(from: chBuf.baseAddress!, count: spec.chunkLength)
        }

        let provider: MLDictionaryFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(dictionary: [
                CoreMLLayout.audioInput: MLFeatureValue(multiArray: audioInputArray),
                CoreMLLayout.hiddenInput: MLFeatureValue(multiArray: hiddenStateArray),
                CoreMLLayout.cellInput: MLFeatureValue(multiArray: cellStateArray)
            ])
        } catch {
            throw TMSileroVADError.invalidModelOutput(reason: "feature provider build: \(error.localizedDescription)")
        }

        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: provider)
        } catch {
            throw TMSileroVADError.invalidModelOutput(reason: "prediction: \(error.localizedDescription)")
        }

        guard let vadArray = Self.featureArray(in: output, matching: CoreMLLayout.vadOutput) else {
            throw TMSileroVADError.invalidModelOutput(reason: "missing vad_output")
        }
        guard let newHidden = Self.featureArray(in: output, matching: CoreMLLayout.newHiddenOutput) else {
            throw TMSileroVADError.invalidModelOutput(reason: "missing new_hidden_state")
        }
        guard let newCell = Self.featureArray(in: output, matching: CoreMLLayout.newCellOutput) else {
            throw TMSileroVADError.invalidModelOutput(reason: "missing new_cell_state")
        }

        let probability = vadArray.dataPointer.assumingMemoryBound(to: Float.self)[0]
        Self.copy(from: newHidden, to: hiddenStateArray, count: spec.stateLength)
        Self.copy(from: newCell, to: cellStateArray, count: spec.stateLength)

        // Update 64-sample context to last samples of this chunk.
        let chunkTail = chunk.suffix(spec.contextLength)
        for (i, v) in chunkTail.enumerated() {
            contextSamples[i] = v
        }

        return probability
    }

    // MARK: helpers

    /// Substring match handles CoreML's tendency to suffix output names.
    private static func featureArray(in provider: MLFeatureProvider, matching name: String) -> MLMultiArray? {
        if let v = provider.featureValue(for: name)?.multiArrayValue {
            return v
        }
        let lower = name.lowercased()
        for candidate in provider.featureNames where candidate.lowercased().contains(lower) {
            if let v = provider.featureValue(for: candidate)?.multiArrayValue {
                return v
            }
        }
        return nil
    }

    private static func zero(array: MLMultiArray, count: Int) {
        let p = array.dataPointer.assumingMemoryBound(to: Float.self)
        vDSP_vclr(p, 1, vDSP_Length(count))
    }

    private static func copy(from src: MLMultiArray, to dst: MLMultiArray, count: Int) {
        let s = src.dataPointer.assumingMemoryBound(to: Float.self)
        let d = dst.dataPointer.assumingMemoryBound(to: Float.self)
        d.update(from: s, count: count)
    }
}
