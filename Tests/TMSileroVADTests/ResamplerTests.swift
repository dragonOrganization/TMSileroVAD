import XCTest
import AVFoundation
@testable import TMSileroVAD

final class ResamplerTests: XCTestCase {
    func test_passthrough_when_already_16k_mono() throws {
        let resampler = try TMSileroVADResampler(targetSampleRate: 16_000)
        let input = PCMSynth.sine(seconds: 0.1, sampleRate: 16_000, frequency: 440)
        let samples = try resampler.resample(input)
        XCTAssertEqual(samples.count, 1600, "0.1s * 16kHz = 1600 samples")
    }

    func test_downsamples_48k_to_16k() throws {
        let resampler = try TMSileroVADResampler(targetSampleRate: 16_000)
        let input = PCMSynth.silence(seconds: 0.1, sampleRate: 48_000)
        let samples = try resampler.resample(input)
        XCTAssertTrue(abs(samples.count - 1600) <= 2,
                      "expected ~1600 samples, got \(samples.count)")
    }

    func test_downsamples_44100_to_16k() throws {
        let resampler = try TMSileroVADResampler(targetSampleRate: 16_000)
        let input = PCMSynth.silence(seconds: 0.1, sampleRate: 44_100)
        let samples = try resampler.resample(input)
        XCTAssertTrue(abs(samples.count - 1600) <= 2,
                      "expected ~1600 samples, got \(samples.count)")
    }

    func test_stereo_input_is_mixed_to_mono() throws {
        let resampler = try TMSileroVADResampler(targetSampleRate: 16_000)
        let input = PCMSynth.silence(seconds: 0.1, sampleRate: 48_000, channels: 2)
        let samples = try resampler.resample(input)
        XCTAssertTrue(abs(samples.count - 1600) <= 2,
                      "expected ~1600 samples, got \(samples.count)")
    }
}
