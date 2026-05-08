# TMSileroVAD

Silero VAD CoreML wrapper for iOS 14+. Two model variants, one Swift API.

## Variants

| Variant | Per-frame latency | Use case |
|---|---|---|
| `.balanced256ms` | 256 ms | Endpointing, ASR pre-processing, push-to-talk replacement |
| `.realtime32ms` | 32 ms | Barge-in / interruption detection, low-latency speech start |

Default is `.balanced256ms`. You can run two instances with different variants in parallel.

## Install

### Setup models

The two CoreML model variants live in `Sources/TMSileroVAD/Resources/`. They are vendored
in the repo (~3 MB total). If you cloned the repo without LFS or you want to refresh from
HuggingFace, run:

```bash
./scripts/download-models.sh
```

Requires `git-lfs` (`brew install git-lfs && git lfs install`). The script pins to a
specific HuggingFace revision for reproducibility.

### Swift Package Manager

```swift
.package(path: "../TMSileroVAD")
```

### CocoaPods

```ruby
pod 'TMSileroVAD', :path => '../TMSileroVAD'
```

## Usage

```swift
import TMSileroVAD

final class VoiceController: TMSileroVADDelegate {
    private var vad: TMSileroVAD!

    func setup() throws {
        vad = try TMSileroVAD(config: .balanced)
        vad.delegate = self
    }

    // 1) When you have raw 16 kHz mono Float32 PCM:
    func feedPCM(_ samples: [Float]) throws {
        try vad.processPCM(samples)
    }

    // 2) When you have any-format AVAudioPCMBuffer (auto-resampled to 16 kHz mono):
    func feedBuffer(_ buffer: AVAudioPCMBuffer) throws {
        try vad.processAudioBuffer(buffer)
    }

    func sileroVAD(_ vad: TMSileroVAD, didEmit event: TMSileroVADEvent) {
        switch event {
        case .speechStarted: handleStart()
        case .speechEnded:   handleEnd()
        case .speechContinuing, .silence: break
        }
    }

    private func handleStart() { /* user started speaking */ }
    private func handleEnd()   { /* user finished — submit to ASR */ }
}
```

`processPCM` and `processAudioBuffer` also return `[TMSileroVADEvent]` synchronously, so
you can use either delegate or sync-return (or both — they fire in parallel).

## Configuration

`TMSileroVADConfig.balanced` and `TMSileroVADConfig.realtime` are sensible defaults. You
can build a custom config by passing all fields explicitly:

```swift
let config = TMSileroVADConfig(
    variant: .balanced256ms,
    startThreshold: 0.5,
    endThreshold: 0.35,
    minSpeechDurationMs: 256,
    minSilenceDurationMs: 1000,
    computeUnits: .all,
    callbackQueue: .main
)
let vad = try TMSileroVAD(config: config)
```

Threshold tuning hints:
- `startThreshold ≥ endThreshold` always (hysteresis)
- Raise `startThreshold` for noisier environments to reduce false starts
- Raise `minSilenceDurationMs` if natural pauses are getting clipped

## Threading model

- `processPCM`, `processAudioBuffer`, `reset` are thread-safe (internal `os_unfair_lock`).
- The lock is held for the **entire duration** of the chunk-pump in a single call,
  including all CoreML inferences. For typical real-time audio (10–20 ms buffers from an
  AVAudioEngine tap), this is microseconds. For "process this 1-second buffer all at
  once" calls, the lock is held until inference completes — concurrent callers will block.
  Plan accordingly.
- Delegate callbacks happen `async` on `config.callbackQueue` (default `.main`), so
  the audio-feeding thread is never blocked by delegate work.
- The mock-runner test path uses the same lock, so unit tests verify thread safety
  end-to-end.

Calling pattern recommendation: feed one source thread (e.g. the AVAudioEngine tap or
your RTC audio callback) into one VAD instance. Don't share a single VAD across multiple
producers.

## What's NOT included

- Audio recording / `AVAudioEngine` setup
- ASR
- RTM / RTC signaling
- Push-to-talk state machine

This Pod is a pure VAD engine. Wire your audio source and downstream consumers yourself.

## Known limitations (v0.1.0)

These are deliberate v0.1.0 trade-offs, scheduled for v0.2.0:

- **Lock scope:** the internal `os_unfair_lock` is held across all CoreML inferences in a
  single call. For long buffers + concurrent callers this can cause perceptible
  contention. A future version may reduce the lock window to in/out state copies and run
  inference unlocked.
- **`FrameBuffer.removeFirst`:** O(n) per chunk. At 256 ms cadence this is negligible
  (~4 calls/sec, ~64 floats shifted). At 32 ms cadence with long pending buffers it can
  show up under profiling. A `Deque`-backed implementation is planned.
- **Partial events on error:** if `predict` throws mid-batch, events accumulated before
  the throw are discarded. Documented behavior; future versions may surface a
  "partial-result-with-error" type.

## License

MIT. See [LICENSE](LICENSE).
