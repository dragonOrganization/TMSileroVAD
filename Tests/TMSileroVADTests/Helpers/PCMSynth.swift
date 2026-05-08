import AVFoundation

enum PCMSynth {
    static func silence(seconds: Double, sampleRate: Double, channels: AVAudioChannelCount = 1) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: sampleRate,
                                   channels: channels,
                                   interleaved: false)!
        let frames = AVAudioFrameCount(seconds * sampleRate)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        return buffer
    }

    static func sine(
        seconds: Double,
        sampleRate: Double,
        frequency: Double,
        amplitude: Float = 0.3,
        channels: AVAudioChannelCount = 1
    ) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: sampleRate,
                                   channels: channels,
                                   interleaved: false)!
        let frames = AVAudioFrameCount(seconds * sampleRate)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        for ch in 0..<Int(channels) {
            let p = buffer.floatChannelData![ch]
            for i in 0..<Int(frames) {
                p[i] = amplitude * Float(sin(2.0 * .pi * frequency * Double(i) / sampleRate))
            }
        }
        return buffer
    }
}
