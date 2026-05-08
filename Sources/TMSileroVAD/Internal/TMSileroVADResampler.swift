import AVFoundation
import Foundation

final class TMSileroVADResampler {
    private let targetFormat: AVAudioFormat
    private var converter: AVAudioConverter?
    private var converterSourceFormat: AVAudioFormat?

    init(targetSampleRate: Double) throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw TMSileroVADError.invalidAudioFormat(reason: "cannot build target 16k mono float32 format")
        }
        self.targetFormat = format
    }

    func resample(_ input: AVAudioPCMBuffer) throws -> [Float] {
        let inputFormat = input.format

        if inputFormat.sampleRate == targetFormat.sampleRate
            && inputFormat.channelCount == 1
            && inputFormat.commonFormat == .pcmFormatFloat32 {
            guard let channelData = input.floatChannelData?[0] else {
                throw TMSileroVADError.invalidAudioFormat(reason: "missing float channel data")
            }
            return Array(UnsafeBufferPointer(start: channelData, count: Int(input.frameLength)))
        }

        if converter == nil || converterSourceFormat != inputFormat {
            guard let newConverter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
                throw TMSileroVADError.resamplerFailed(reason: "AVAudioConverter init failed")
            }
            converter = newConverter
            converterSourceFormat = inputFormat
        }
        guard let converter = converter else {
            throw TMSileroVADError.resamplerFailed(reason: "converter unavailable")
        }

        let ratio = targetFormat.sampleRate / inputFormat.sampleRate
        let outputCapacity = AVAudioFrameCount(Double(input.frameLength) * ratio + 32)
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputCapacity) else {
            throw TMSileroVADError.resamplerFailed(reason: "cannot allocate output buffer")
        }

        var fed = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, outStatus in
            if fed {
                outStatus.pointee = .endOfStream
                return nil
            }
            fed = true
            outStatus.pointee = .haveData
            return input
        }

        if let error = error {
            throw TMSileroVADError.resamplerFailed(reason: error.localizedDescription)
        }
        if status == .error {
            throw TMSileroVADError.resamplerFailed(reason: "convert returned .error")
        }

        guard let outChannel = output.floatChannelData?[0] else {
            throw TMSileroVADError.invalidAudioFormat(reason: "missing output channel data")
        }
        return Array(UnsafeBufferPointer(start: outChannel, count: Int(output.frameLength)))
    }
}
