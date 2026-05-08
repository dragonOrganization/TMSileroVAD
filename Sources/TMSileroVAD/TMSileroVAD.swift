import AVFoundation
import CoreML
import Foundation
import os

public final class TMSileroVAD {
    public weak var delegate: TMSileroVADDelegate?
    public let config: TMSileroVADConfig

    private let spec: TMSileroVADModelSpec
    private let runner: TMSileroVADModelRunning
    private let frameBuffer: TMSileroVADFrameBuffer
    private let stateMachine: TMSileroVADStateMachine
    private let resampler: TMSileroVADResampler
    private var lock = os_unfair_lock_s()

    public convenience init(config: TMSileroVADConfig = .balanced) throws {
        let spec = TMSileroVADModelSpec.makeSpec(variant: config.variant)
        let runner = try TMSileroVADCoreMLRunner(spec: spec, computeUnits: config.computeUnits)
        try self.init(config: config, runner: runner, spec: spec)
    }

    init(
        config: TMSileroVADConfig,
        runner: TMSileroVADModelRunning,
        spec: TMSileroVADModelSpec
    ) throws {
        self.config = config
        self.spec = spec
        self.runner = runner
        self.frameBuffer = TMSileroVADFrameBuffer(frameLength: spec.chunkLength)
        self.stateMachine = TMSileroVADStateMachine(
            frameDurationMs: spec.frameDurationMs,
            startThreshold: config.startThreshold,
            endThreshold: config.endThreshold,
            minSpeechDurationMs: config.minSpeechDurationMs,
            minSilenceDurationMs: config.minSilenceDurationMs
        )
        self.resampler = try TMSileroVADResampler(targetSampleRate: Double(spec.sampleRate))
    }

    public func reset() {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        frameBuffer.reset()
        runner.reset()
        stateMachine.reset()
    }

    @discardableResult
    public func processPCM(_ samples: [Float]) throws -> [TMSileroVADEvent] {
        let events = try runLocked { try runProcess(samples: samples) }
        dispatchToDelegate(events)
        return events
    }

    @discardableResult
    public func processAudioBuffer(_ buffer: AVAudioPCMBuffer) throws -> [TMSileroVADEvent] {
        let events = try runLocked {
            let resampled = try resampler.resample(buffer)
            return try runProcess(samples: resampled)
        }
        dispatchToDelegate(events)
        return events
    }

    // MARK: private

    /// Pumps PCM through the framebuffer → runner → state machine pipeline.
    ///
    /// **Failure semantics:** if `runner.predict` throws on chunk N, events accumulated from
    /// chunks 0..N-1 are discarded along with the error. Callers cannot recover partial output;
    /// resync the input audio (call `reset()`) before continuing.
    private func runProcess(samples: [Float]) throws -> [TMSileroVADEvent] {
        let chunks = frameBuffer.append(samples)
        var events: [TMSileroVADEvent] = []
        events.reserveCapacity(chunks.count)
        for chunk in chunks {
            let probability = try runner.predict(chunk: chunk)
            events.append(stateMachine.process(probability: probability))
        }
        return events
    }

    private func runLocked<T>(_ body: () throws -> T) throws -> T {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return try body()
    }

    private func dispatchToDelegate(_ events: [TMSileroVADEvent]) {
        guard !events.isEmpty else { return }
        let queue = config.callbackQueue
        queue.async { [weak self] in
            guard let self = self, let delegate = self.delegate else { return }
            for event in events {
                delegate.sileroVAD(self, didEmit: event)
            }
        }
    }

    /// Test-only escape hatch. `internal` so unit tests can inject a mock runner via
    /// `@testable import`. Not part of the supported public API; do not invoke from
    /// production code paths inside the module either.
    static func makeForTesting(
        config: TMSileroVADConfig,
        runner: TMSileroVADModelRunning,
        spec: TMSileroVADModelSpec
    ) throws -> TMSileroVAD {
        return try TMSileroVAD(config: config, runner: runner, spec: spec)
    }
}
