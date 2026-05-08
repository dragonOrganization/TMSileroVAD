# TMSileroVAD Pod Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CocoaPods-distributable iOS Swift Pod (`TMSileroVAD`) that wraps the FluidInference Silero VAD unified CoreML models. The Pod exposes a single delegate + sync-return API, supports both 32ms (low-latency / barge-in) and 256ms (balanced / endpointing) variants, and accepts either strict 16 kHz mono Float32 PCM or arbitrary `AVAudioPCMBuffer` via built-in `AVAudioConverter` resampling.

**Architecture:** One public facade `TMSileroVAD` backed by six internal components: `ModelSpec` (variant → constants lookup), `ModelRunning` (protocol for mock-ability), `CoreMLRunner` (loads `.mlmodelc`, owns LSTM hidden/cell state and 64-sample context, runs inference), `FrameBuffer` (slices arbitrary-length PCM into model-fixed chunks), `StateMachine` (hysteresis + min-duration debouncing into `speechStarted` / `speechEnded` events), `Resampler` (`AVAudioConverter` wrapper to 16 kHz mono Float32). Two `.mlmodelc` files ship in `Resources/`; user picks via enum at init time. State is guarded by `os_unfair_lock`; delegate dispatch is `async` to a user-configured `DispatchQueue` (default `.main`).

**Tech Stack:** Swift 5.9+, iOS 14.0 deployment target, CoreML, AVFoundation (`AVAudioConverter`), Accelerate (`vDSP_mmov` for fast PCM → MLMultiArray copy), XCTest via Swift Package Manager (for fast TDD), CocoaPods for downstream distribution.

**Real model signatures (verified from `metadata.json` + FluidAudio source):**

| Variant file | audio_input shape | chunk samples | frame ms | h/c shape |
|---|---|---|---|---|
| `silero-vad-unified-v6.0.0.mlmodelc` | `[1, 576]` | 64 ctx + **512** chunk | 32 ms | `[1, 128]` |
| `silero-vad-unified-256ms-v6.0.0.mlmodelc` | `[1, 4160]` | 64 ctx + **4096** chunk | 256 ms | `[1, 128]` |

Both share input names `audio_input / hidden_state / cell_state` and output names `vad_output / new_hidden_state / new_cell_state`. CoreML may suffix output names; runner must do substring match (FluidAudio convention).

---

## File Structure

```
SileroVAD/
├── Package.swift                                # SPM manifest (testing)
├── TMSileroVAD.podspec                          # CocoaPods distribution
├── README.md                                    # usage docs
├── .gitignore
├── scripts/
│   └── download-models.sh                       # fetches models from HF
├── Sources/TMSileroVAD/
│   ├── TMSileroVAD.swift                        # public facade
│   ├── TMSileroVADConfig.swift                  # public config struct
│   ├── TMSileroVADVariant.swift                 # public enum
│   ├── TMSileroVADEvent.swift                   # public event enum
│   ├── TMSileroVADDelegate.swift                # public delegate protocol
│   ├── TMSileroVADError.swift                   # public errors
│   ├── Resources/                               # SPM resource path
│   │   ├── silero-vad-unified-v6.0.0.mlmodelc/        (downloaded)
│   │   └── silero-vad-unified-256ms-v6.0.0.mlmodelc/  (downloaded)
│   └── Internal/
│       ├── TMSileroVADModelSpec.swift           # variant → constants
│       ├── TMSileroVADModelRunning.swift        # protocol
│       ├── TMSileroVADCoreMLRunner.swift        # real runner
│       ├── TMSileroVADFrameBuffer.swift         # PCM chunker
│       ├── TMSileroVADStateMachine.swift        # event hysteresis
│       ├── TMSileroVADResampler.swift           # AVAudioConverter
│       └── TMSileroVADResources.swift           # bundle resolution
└── Tests/TMSileroVADTests/
    ├── ModelSpecTests.swift
    ├── FrameBufferTests.swift
    ├── StateMachineTests.swift
    ├── ResamplerTests.swift
    ├── CoreMLRunnerIntegrationTests.swift
    ├── TMSileroVADFacadeTests.swift             # mock-runner-based
    └── Helpers/
        ├── MockModelRunner.swift
        └── PCMSynth.swift                       # silence/sine generators
```

**Why both Package.swift and podspec?** SPM gives fast `swift test` for TDD; podspec is for downstream consumption. Both reference the same `Sources/TMSileroVAD/` tree, so there is one source of truth.

---

## Task 0: Project scaffold

**Files:**
- Create: `Package.swift`
- Create: `TMSileroVAD.podspec`
- Create: `.gitignore`
- Create: `README.md` (placeholder, finalized in Task 11)

- [ ] **Step 1: Create `.gitignore`**

```gitignore
.DS_Store
.build/
.swiftpm/
DerivedData/
*.xcodeproj
*.xcworkspace
Pods/
xcuserdata/
*.hmap
```

- [ ] **Step 2: Create `Package.swift`**

```swift
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "TMSileroVAD",
    platforms: [
        .iOS(.v14),
        .macOS(.v11)
    ],
    products: [
        .library(name: "TMSileroVAD", targets: ["TMSileroVAD"])
    ],
    targets: [
        .target(
            name: "TMSileroVAD",
            path: "Sources/TMSileroVAD",
            resources: [
                .copy("Resources/silero-vad-unified-v6.0.0.mlmodelc"),
                .copy("Resources/silero-vad-unified-256ms-v6.0.0.mlmodelc")
            ]
        ),
        .testTarget(
            name: "TMSileroVADTests",
            dependencies: ["TMSileroVAD"],
            path: "Tests/TMSileroVADTests"
        )
    ]
)
```

- [ ] **Step 3: Create `TMSileroVAD.podspec`**

```ruby
Pod::Spec.new do |s|
  s.name             = 'TMSileroVAD'
  s.version          = '0.1.0'
  s.summary          = 'Silero VAD CoreML wrapper for iOS real-time voice activity detection.'
  s.description      = <<-DESC
    A Swift Pod wrapping the FluidInference Silero VAD unified CoreML model.
    Supports two variants: 256 ms balanced (endpointing/ASR) and 32 ms realtime
    (low-latency barge-in). Built-in AVAudioConverter resampling. iOS 14+.
  DESC
  s.homepage         = 'https://example.com/TMSileroVAD'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'TalkMe' => 'ios@example.com' }
  s.source           = { :git => 'https://example.com/TMSileroVAD.git', :tag => s.version.to_s }
  s.ios.deployment_target = '14.0'
  s.swift_versions = ['5.9']
  s.source_files = 'Sources/TMSileroVAD/**/*.swift'
  s.resource_bundles = {
    'TMSileroVADResources' => [
      'Sources/TMSileroVAD/Resources/silero-vad-unified-v6.0.0.mlmodelc',
      'Sources/TMSileroVAD/Resources/silero-vad-unified-256ms-v6.0.0.mlmodelc'
    ]
  }
  s.frameworks = 'Foundation', 'AVFoundation', 'CoreML', 'Accelerate'
end
```

- [ ] **Step 4: Create directory tree**

```bash
mkdir -p Sources/TMSileroVAD/Internal
mkdir -p Sources/TMSileroVAD/Resources
mkdir -p Tests/TMSileroVADTests/Helpers
mkdir -p scripts
mkdir -p docs/superpowers/plans
```

- [ ] **Step 5: Verify SPM scaffold compiles**

```bash
swift build 2>&1 | tail -5
```

Expected: build succeeds with "Build complete!" (zero source files yet, that's fine).

- [ ] **Step 6: Commit**

```bash
git add Package.swift TMSileroVAD.podspec .gitignore Sources Tests scripts
git commit -m "scaffold: TMSileroVAD Pod skeleton (SPM + Podspec)"
```

---

## Task 1: Download models from HuggingFace

**Files:**
- Create: `scripts/download-models.sh`

- [ ] **Step 1: Write the download script**

```bash
#!/usr/bin/env bash
# Downloads the two required Silero VAD CoreML model variants from HuggingFace
# into Sources/TMSileroVAD/Resources/.
#
# Requires: git, git-lfs (HuggingFace stores .mlmodelc binary blobs via LFS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RES_DIR="$ROOT_DIR/Sources/TMSileroVAD/Resources"

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "git-lfs is required. Install with: brew install git-lfs && git lfs install" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Cloning silero-vad-coreml from HuggingFace..."
GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 \
    https://huggingface.co/FluidInference/silero-vad-coreml \
    "$TMP_DIR/silero"

mkdir -p "$RES_DIR"

for variant in silero-vad-unified-v6.0.0 silero-vad-unified-256ms-v6.0.0; do
    src="$TMP_DIR/silero/${variant}.mlmodelc"
    dst="$RES_DIR/${variant}.mlmodelc"
    if [[ ! -d "$src" ]]; then
        echo "Missing $src in HF clone" >&2
        exit 1
    fi
    rm -rf "$dst"
    cp -R "$src" "$dst"
    echo "Installed: $dst"
done

echo "Done."
```

- [ ] **Step 2: Make executable and run it**

```bash
chmod +x scripts/download-models.sh
./scripts/download-models.sh
```

Expected output ends with `Done.`. Both `.mlmodelc` directories present in `Sources/TMSileroVAD/Resources/`.

- [ ] **Step 3: Sanity-check the download**

```bash
ls -la Sources/TMSileroVAD/Resources/silero-vad-unified-256ms-v6.0.0.mlmodelc/
cat Sources/TMSileroVAD/Resources/silero-vad-unified-256ms-v6.0.0.mlmodelc/metadata.json | head -30
```

Expected: directory contains `coremldata.bin`, `metadata.json`, `model.espresso.{net,shape,weights}`, `model.mil`, etc. `metadata.json` shows `"shape": "[1, 4160]"` for `audio_input`.

- [ ] **Step 4: Commit**

```bash
git add scripts/download-models.sh Sources/TMSileroVAD/Resources/
git commit -m "models: vendor Silero VAD unified v6.0.0 (32ms + 256ms variants)"
```

> If `.mlmodelc` files exceed 100 MB combined or you want to keep them out of git, add `Sources/TMSileroVAD/Resources/*.mlmodelc/` to `.gitignore` and have CI run the script. Default plan: vendor them in (~3 MB total).

---

## Task 2: Public types — Variant, Event, Error, Delegate, Config

**Files:**
- Create: `Sources/TMSileroVAD/TMSileroVADVariant.swift`
- Create: `Sources/TMSileroVAD/TMSileroVADEvent.swift`
- Create: `Sources/TMSileroVAD/TMSileroVADError.swift`
- Create: `Sources/TMSileroVAD/TMSileroVADDelegate.swift`
- Create: `Sources/TMSileroVAD/TMSileroVADConfig.swift`

> Public types are small and tightly coupled (Config references Variant, Event, etc.). Ship them as one commit.

- [ ] **Step 1: Write `TMSileroVADVariant.swift`**

```swift
import Foundation

public enum TMSileroVADVariant: Sendable, Equatable {
    /// 256 ms / 4096-sample model with internal noisy-OR aggregation.
    /// Use for endpointing, ASR pre-processing, push-to-talk replacement.
    case balanced256ms
    
    /// 32 ms / 512-sample model. Use for AI barge-in, low-latency speech start detection.
    case realtime32ms
}
```

- [ ] **Step 2: Write `TMSileroVADEvent.swift`**

```swift
import Foundation

extension TMSileroVAD {
    public enum SileroVADEvent: Equatable, Sendable {
        case silence(probability: Float)
        case speechStarted(probability: Float)
        case speechContinuing(probability: Float)
        case speechEnded(probability: Float)
        
        public var probability: Float {
            switch self {
            case .silence(let probability),
                 .speechStarted(let probability),
                 .speechContinuing(let probability),
                 .speechEnded(let probability):
                return probability
            }
        }
    }
}
```

- [ ] **Step 3: Write `TMSileroVADError.swift`**

```swift
import Foundation

public enum TMSileroVADError: Error, Equatable {
    case modelNotFound(name: String)
    case modelLoadFailed(underlying: String)
    case invalidAudioFormat(reason: String)
    case invalidFrameLength(expected: Int, got: Int)
    case invalidModelOutput(reason: String)
    case resamplerFailed(reason: String)
}
```

- [ ] **Step 4: Write `TMSileroVADDelegate.swift`**

```swift
import Foundation

public protocol TMSileroVADDelegate: AnyObject {
    /// Called for every chunk processed, on the queue configured in `SileroVADConfig.callbackQueue`.
    func sileroVAD(_ vad: TMSileroVAD, didEmit event: TMSileroVAD.SileroVADEvent)
}
```

- [ ] **Step 5: Write `TMSileroVADConfig.swift`**

```swift
import CoreML
import Dispatch
import Foundation

extension TMSileroVAD {
    public struct SileroVADConfig {
        public let variant: TMSileroVADVariant
        public let startThreshold: Float
        public let endThreshold: Float
        public let minSpeechDurationMs: Int
        public let minSilenceDurationMs: Int
        public let computeUnits: MLComputeUnits
        public let callbackQueue: DispatchQueue
        
        public init(
            variant: TMSileroVADVariant,
            startThreshold: Float,
            endThreshold: Float,
            minSpeechDurationMs: Int,
            minSilenceDurationMs: Int,
            computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
            callbackQueue: DispatchQueue = .main
        ) {
            self.variant = variant
            self.startThreshold = startThreshold
            self.endThreshold = endThreshold
            self.minSpeechDurationMs = minSpeechDurationMs
            self.minSilenceDurationMs = minSilenceDurationMs
            self.computeUnits = computeUnits
            self.callbackQueue = callbackQueue
        }
        
        /// 256 ms variant. Endpointing / ASR pre-processing default.
        public static let balanced = SileroVADConfig(
            variant: .balanced256ms,
            startThreshold: 0.5,
            endThreshold: 0.35,
            minSpeechDurationMs: 256,
            minSilenceDurationMs: 768
        )
        
        /// 32 ms variant. Barge-in / interruption detection default.
        public static let realtime = SileroVADConfig(
            variant: .realtime32ms,
            startThreshold: 0.5,
            endThreshold: 0.35,
            minSpeechDurationMs: 160,
            minSilenceDurationMs: 640
        )
    }
}
```

- [ ] **Step 6: Stub `TMSileroVAD.swift` so the extensions compile**

Create `Sources/TMSileroVAD/TMSileroVAD.swift` with an empty class shell. Real implementation comes in Task 9.

```swift
import Foundation

public final class TMSileroVAD {
    // Real implementation in Task 9.
    fileprivate init() {}
}
```

- [ ] **Step 7: Run build**

```bash
swift build 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Sources/TMSileroVAD/TMSileroVADVariant.swift \
        Sources/TMSileroVAD/TMSileroVADEvent.swift \
        Sources/TMSileroVAD/TMSileroVADError.swift \
        Sources/TMSileroVAD/TMSileroVADDelegate.swift \
        Sources/TMSileroVAD/TMSileroVADConfig.swift \
        Sources/TMSileroVAD/TMSileroVAD.swift
git commit -m "types: public Variant, Event, Error, Delegate, Config"
```

---

## Task 3: ModelSpec — variant → CoreML constants lookup

**Files:**
- Create: `Sources/TMSileroVAD/Internal/TMSileroVADModelSpec.swift`
- Create: `Tests/TMSileroVADTests/ModelSpecTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
import XCTest
@testable import TMSileroVAD

final class ModelSpecTests: XCTestCase {
    func test_balanced256ms_spec_has_4096_chunk_and_4160_input() {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .balanced256ms)
        XCTAssertEqual(spec.modelName, "silero-vad-unified-256ms-v6.0.0")
        XCTAssertEqual(spec.contextLength, 64)
        XCTAssertEqual(spec.chunkLength, 4096)
        XCTAssertEqual(spec.inputLength, 4160)
        XCTAssertEqual(spec.frameDurationMs, 256)
        XCTAssertEqual(spec.stateLength, 128)
        XCTAssertEqual(spec.sampleRate, 16_000)
    }
    
    func test_realtime32ms_spec_has_512_chunk_and_576_input() {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        XCTAssertEqual(spec.modelName, "silero-vad-unified-v6.0.0")
        XCTAssertEqual(spec.contextLength, 64)
        XCTAssertEqual(spec.chunkLength, 512)
        XCTAssertEqual(spec.inputLength, 576)
        XCTAssertEqual(spec.frameDurationMs, 32)
        XCTAssertEqual(spec.stateLength, 128)
        XCTAssertEqual(spec.sampleRate, 16_000)
    }
}
```

- [ ] **Step 2: Run the test (should fail to compile)**

```bash
swift test --filter ModelSpecTests 2>&1 | tail -20
```

Expected: FAIL — `cannot find 'TMSileroVADModelSpec' in scope`.

- [ ] **Step 3: Implement `TMSileroVADModelSpec`**

```swift
import Foundation

struct TMSileroVADModelSpec {
    let modelName: String
    let contextLength: Int
    let chunkLength: Int
    let inputLength: Int
    let frameDurationMs: Int
    let stateLength: Int
    let sampleRate: Int
    
    static func makeSpec(variant: TMSileroVADVariant) -> TMSileroVADModelSpec {
        switch variant {
        case .balanced256ms:
            return TMSileroVADModelSpec(
                modelName: "silero-vad-unified-256ms-v6.0.0",
                contextLength: 64,
                chunkLength: 4096,
                inputLength: 4160,
                frameDurationMs: 256,
                stateLength: 128,
                sampleRate: 16_000
            )
        case .realtime32ms:
            return TMSileroVADModelSpec(
                modelName: "silero-vad-unified-v6.0.0",
                contextLength: 64,
                chunkLength: 512,
                inputLength: 576,
                frameDurationMs: 32,
                stateLength: 128,
                sampleRate: 16_000
            )
        }
    }
}
```

- [ ] **Step 4: Run tests**

```bash
swift test --filter ModelSpecTests 2>&1 | tail -10
```

Expected: PASS, 2/2.

- [ ] **Step 5: Commit**

```bash
git add Sources/TMSileroVAD/Internal/TMSileroVADModelSpec.swift \
        Tests/TMSileroVADTests/ModelSpecTests.swift
git commit -m "spec: TMSileroVADModelSpec maps variant to CoreML constants"
```

---

## Task 4: FrameBuffer — slice arbitrary-length PCM into model chunks

**Files:**
- Create: `Sources/TMSileroVAD/Internal/TMSileroVADFrameBuffer.swift`
- Create: `Tests/TMSileroVADTests/FrameBufferTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
@testable import TMSileroVAD

final class FrameBufferTests: XCTestCase {
    func test_under_one_frame_yields_no_chunk() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 512)
        let chunks = buffer.append(Array(repeating: 0.1, count: 100))
        XCTAssertEqual(chunks.count, 0)
    }
    
    func test_exactly_one_frame_yields_one_chunk() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 512)
        let chunks = buffer.append(Array(repeating: 0.1, count: 512))
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].count, 512)
    }
    
    func test_two_and_a_half_frames_yields_two_chunks_with_remainder() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 512)
        let chunks = buffer.append(Array(repeating: 0.1, count: 512 * 2 + 256))
        XCTAssertEqual(chunks.count, 2)
        // Next append should fill remainder.
        let next = buffer.append(Array(repeating: 0.1, count: 256))
        XCTAssertEqual(next.count, 1)
    }
    
    func test_multiple_appends_concatenate() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 512)
        XCTAssertEqual(buffer.append(Array(repeating: 0.1, count: 300)).count, 0)
        let chunks = buffer.append(Array(repeating: 0.2, count: 300))
        XCTAssertEqual(chunks.count, 1)
        // First 300 samples are 0.1, next 212 are 0.2.
        XCTAssertEqual(chunks[0][0], 0.1, accuracy: 1e-6)
        XCTAssertEqual(chunks[0][299], 0.1, accuracy: 1e-6)
        XCTAssertEqual(chunks[0][300], 0.2, accuracy: 1e-6)
    }
    
    func test_reset_drops_pending() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 512)
        _ = buffer.append(Array(repeating: 0.1, count: 300))
        buffer.reset()
        let chunks = buffer.append(Array(repeating: 0.2, count: 512))
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0][0], 0.2, accuracy: 1e-6, "reset must drop pending samples")
    }
    
    func test_supports_4096_chunk_for_balanced_variant() {
        let buffer = TMSileroVADFrameBuffer(frameLength: 4096)
        let chunks = buffer.append(Array(repeating: 0.1, count: 4096 * 3))
        XCTAssertEqual(chunks.count, 3)
        XCTAssertEqual(chunks[0].count, 4096)
    }
}
```

- [ ] **Step 2: Run tests (should fail)**

```bash
swift test --filter FrameBufferTests 2>&1 | tail -10
```

Expected: FAIL — type doesn't exist.

- [ ] **Step 3: Implement `TMSileroVADFrameBuffer`**

```swift
import Foundation

final class TMSileroVADFrameBuffer {
    private let frameLength: Int
    private var pendingSamples: [Float] = []
    
    init(frameLength: Int) {
        self.frameLength = frameLength
        self.pendingSamples.reserveCapacity(frameLength * 2)
    }
    
    func reset() {
        pendingSamples.removeAll(keepingCapacity: true)
    }
    
    func append(_ samples: [Float]) -> [[Float]] {
        pendingSamples.append(contentsOf: samples)
        
        var chunks: [[Float]] = []
        while pendingSamples.count >= frameLength {
            chunks.append(Array(pendingSamples.prefix(frameLength)))
            pendingSamples.removeFirst(frameLength)
        }
        return chunks
    }
}
```

- [ ] **Step 4: Run tests**

```bash
swift test --filter FrameBufferTests 2>&1 | tail -10
```

Expected: PASS, 6/6.

- [ ] **Step 5: Commit**

```bash
git add Sources/TMSileroVAD/Internal/TMSileroVADFrameBuffer.swift \
        Tests/TMSileroVADTests/FrameBufferTests.swift
git commit -m "framebuffer: slice PCM into fixed-length chunks"
```

---

## Task 5: StateMachine — hysteresis + min-duration → events

**Files:**
- Create: `Sources/TMSileroVAD/Internal/TMSileroVADStateMachine.swift`
- Create: `Tests/TMSileroVADTests/StateMachineTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
@testable import TMSileroVAD

final class StateMachineTests: XCTestCase {
    private func make(
        frameMs: Int = 32,
        start: Float = 0.5,
        end: Float = 0.35,
        minSpeech: Int = 96,
        minSilence: Int = 320
    ) -> TMSileroVADStateMachine {
        TMSileroVADStateMachine(
            frameDurationMs: frameMs,
            startThreshold: start,
            endThreshold: end,
            minSpeechDurationMs: minSpeech,
            minSilenceDurationMs: minSilence
        )
    }
    
    func test_below_start_threshold_stays_silent() {
        let sm = make()
        let event = sm.process(probability: 0.1)
        XCTAssertEqual(event, .silence(probability: 0.1))
    }
    
    func test_speech_started_only_after_min_duration() {
        let sm = make(frameMs: 32, minSpeech: 96) // 3 frames
        XCTAssertEqual(sm.process(probability: 0.9), .silence(probability: 0.9), "frame 1 still candidate")
        XCTAssertEqual(sm.process(probability: 0.9), .silence(probability: 0.9), "frame 2 still candidate")
        XCTAssertEqual(sm.process(probability: 0.9), .speechStarted(probability: 0.9), "frame 3 fires")
    }
    
    func test_speech_candidate_drops_back_to_silence_on_low_prob() {
        let sm = make(frameMs: 32, minSpeech: 96)
        _ = sm.process(probability: 0.9)
        XCTAssertEqual(sm.process(probability: 0.1), .silence(probability: 0.1))
        // Now we're back at silence, and a single high frame again starts a fresh candidate run.
        XCTAssertEqual(sm.process(probability: 0.9), .silence(probability: 0.9))
    }
    
    func test_speaking_emits_continuing_above_end_threshold() {
        let sm = make(frameMs: 32, minSpeech: 32)
        XCTAssertEqual(sm.process(probability: 0.9), .speechStarted(probability: 0.9))
        XCTAssertEqual(sm.process(probability: 0.6), .speechContinuing(probability: 0.6))
        XCTAssertEqual(sm.process(probability: 0.4), .speechContinuing(probability: 0.4),
                       "above end threshold (0.35) so still continuing")
    }
    
    func test_speech_ended_only_after_min_silence() {
        let sm = make(frameMs: 32, minSpeech: 32, minSilence: 96)
        _ = sm.process(probability: 0.9) // speechStarted
        XCTAssertEqual(sm.process(probability: 0.1), .speechContinuing(probability: 0.1), "frame 1 below end")
        XCTAssertEqual(sm.process(probability: 0.1), .speechContinuing(probability: 0.1), "frame 2")
        XCTAssertEqual(sm.process(probability: 0.1), .speechEnded(probability: 0.1), "frame 3 fires end")
    }
    
    func test_silence_candidate_recovers_to_speaking_on_high_prob() {
        let sm = make(frameMs: 32, minSpeech: 32, minSilence: 96)
        _ = sm.process(probability: 0.9)
        _ = sm.process(probability: 0.1)
        XCTAssertEqual(sm.process(probability: 0.9), .speechContinuing(probability: 0.9))
        // Should still need 3 sub-threshold frames in a row to end.
        _ = sm.process(probability: 0.1)
        _ = sm.process(probability: 0.1)
        XCTAssertEqual(sm.process(probability: 0.1), .speechEnded(probability: 0.1))
    }
    
    func test_reset_returns_to_silence() {
        let sm = make(frameMs: 32, minSpeech: 32)
        _ = sm.process(probability: 0.9) // speechStarted
        sm.reset()
        XCTAssertEqual(sm.process(probability: 0.4), .speechContinuing(probability: 0.4),
                       "without reset would remain in speaking")
        // Wait: after reset we're in silence. 0.4 < startThreshold 0.5, should be silence.
    }
}
```

> Note on the last test: the comment is intentionally misleading — read carefully. After `reset()` state is `.silence`. Probability 0.4 is below the 0.5 start threshold so the expected event is `.silence(probability: 0.4)`, not `.speechContinuing`. Fix the assertion before running.

- [ ] **Step 2: Fix the deliberate trap in `test_reset_returns_to_silence`**

Replace:
```swift
XCTAssertEqual(sm.process(probability: 0.4), .speechContinuing(probability: 0.4), ...)
```
with:
```swift
XCTAssertEqual(sm.process(probability: 0.4), .silence(probability: 0.4),
               "after reset, sub-threshold prob is silence")
```

- [ ] **Step 3: Run tests (should fail to compile)**

```bash
swift test --filter StateMachineTests 2>&1 | tail -10
```

Expected: FAIL.

- [ ] **Step 4: Implement `TMSileroVADStateMachine`**

```swift
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
    
    func process(probability: Float) -> TMSileroVAD.SileroVADEvent {
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
```

> Implementation note: `silence` state can also fire `speechStarted` directly when `minSpeechDurationMs <= frameDurationMs` (the `realtime` config has minSpeech=160ms with 32ms frames so this won't fire on first frame, but the `balanced` config has minSpeech=256ms with 256ms frames, so it does — the user wants instant start when min duration equals one frame). The same applies to `speaking → silence` transition.

- [ ] **Step 5: Run tests**

```bash
swift test --filter StateMachineTests 2>&1 | tail -10
```

Expected: PASS, 7/7.

- [ ] **Step 6: Commit**

```bash
git add Sources/TMSileroVAD/Internal/TMSileroVADStateMachine.swift \
        Tests/TMSileroVADTests/StateMachineTests.swift
git commit -m "stateMachine: hysteresis + min-duration debouncing of VAD probability"
```

---

## Task 6: Resampler — `AVAudioConverter` to 16 kHz mono Float32

**Files:**
- Create: `Sources/TMSileroVAD/Internal/TMSileroVADResampler.swift`
- Create: `Tests/TMSileroVADTests/ResamplerTests.swift`
- Create: `Tests/TMSileroVADTests/Helpers/PCMSynth.swift`

- [ ] **Step 1: Create test helper for synthesizing PCM buffers**

```swift
// Tests/TMSileroVADTests/Helpers/PCMSynth.swift
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
        // Already zero-initialized.
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
```

- [ ] **Step 2: Write the failing tests**

```swift
// Tests/TMSileroVADTests/ResamplerTests.swift
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
        // AVAudioConverter rounds; allow ±2 samples.
        XCTAssertEqual(samples.count, 1600, accuracy: 2)
    }
    
    func test_downsamples_44100_to_16k() throws {
        let resampler = try TMSileroVADResampler(targetSampleRate: 16_000)
        let input = PCMSynth.silence(seconds: 0.1, sampleRate: 44_100)
        let samples = try resampler.resample(input)
        XCTAssertEqual(samples.count, 1600, accuracy: 2)
    }
    
    func test_stereo_input_is_mixed_to_mono() throws {
        let resampler = try TMSileroVADResampler(targetSampleRate: 16_000)
        let input = PCMSynth.silence(seconds: 0.1, sampleRate: 48_000, channels: 2)
        let samples = try resampler.resample(input)
        XCTAssertEqual(samples.count, 1600, accuracy: 2)
    }
}

extension XCTestCase {
    func XCTAssertEqual(_ a: Int, _ b: Int, accuracy: Int, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(abs(a - b) <= accuracy, "\(a) not within \(accuracy) of \(b)", file: file, line: line)
    }
}
```

- [ ] **Step 3: Run tests (should fail)**

Expected: FAIL — `TMSileroVADResampler` doesn't exist.

- [ ] **Step 4: Implement `TMSileroVADResampler`**

```swift
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
    
    /// Convert an arbitrary buffer to 16 kHz mono Float32 [Float] samples.
    func resample(_ input: AVAudioPCMBuffer) throws -> [Float] {
        let inputFormat = input.format
        
        // Fast path: already in target format.
        if inputFormat.sampleRate == targetFormat.sampleRate
            && inputFormat.channelCount == 1
            && inputFormat.commonFormat == .pcmFormatFloat32 {
            guard let channelData = input.floatChannelData?[0] else {
                throw TMSileroVADError.invalidAudioFormat(reason: "missing float channel data")
            }
            return Array(UnsafeBufferPointer(start: channelData, count: Int(input.frameLength)))
        }
        
        // Lazily build/cache converter when input format changes.
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
```

- [ ] **Step 5: Run tests**

```bash
swift test --filter ResamplerTests 2>&1 | tail -10
```

Expected: PASS, 4/4.

- [ ] **Step 6: Commit**

```bash
git add Sources/TMSileroVAD/Internal/TMSileroVADResampler.swift \
        Tests/TMSileroVADTests/ResamplerTests.swift \
        Tests/TMSileroVADTests/Helpers/PCMSynth.swift
git commit -m "resampler: AVAudioConverter to 16kHz mono Float32"
```

---

## Task 7: ModelRunning protocol + Resources bundle helper

**Files:**
- Create: `Sources/TMSileroVAD/Internal/TMSileroVADModelRunning.swift`
- Create: `Sources/TMSileroVAD/Internal/TMSileroVADResources.swift`

- [ ] **Step 1: Write `TMSileroVADModelRunning`**

```swift
import Foundation

protocol TMSileroVADModelRunning: AnyObject {
    /// Run one inference. `chunk.count` MUST equal `spec.chunkLength`.
    /// Returns the speech probability in [0, 1].
    func predict(chunk: [Float]) throws -> Float
    
    /// Reset LSTM hidden/cell state and 64-sample audio context.
    func reset()
}
```

- [ ] **Step 2: Write `TMSileroVADResources`**

```swift
import Foundation

enum TMSileroVADResources {
    /// Resolves the `.mlmodelc` URL by trying:
    ///  1. SPM-generated `Bundle.module` (when consumed via SwiftPM)
    ///  2. CocoaPods `TMSileroVADResources.bundle` next to the framework binary
    ///  3. The framework bundle itself (when the .mlmodelc is direct-included)
    static func modelURL(forName name: String) throws -> URL {
        let candidates: [Bundle?] = [
            cocoapodsResourceBundle(),
            spmModuleBundle(),
            Bundle(for: TMSileroVAD.self)
        ]
        
        for candidate in candidates.compactMap({ $0 }) {
            if let url = candidate.url(forResource: name, withExtension: "mlmodelc") {
                return url
            }
        }
        throw TMSileroVADError.modelNotFound(name: name)
    }
    
    private static func cocoapodsResourceBundle() -> Bundle? {
        let candidates: [URL] = [
            Bundle(for: TMSileroVAD.self).resourceURL,
            Bundle.main.resourceURL
        ].compactMap { $0 }
        for url in candidates {
            let bundleURL = url.appendingPathComponent("TMSileroVADResources.bundle")
            if let bundle = Bundle(url: bundleURL) {
                return bundle
            }
        }
        return nil
    }
    
    private static func spmModuleBundle() -> Bundle? {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return nil
        #endif
    }
}
```

- [ ] **Step 3: Build to verify**

```bash
swift build 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add Sources/TMSileroVAD/Internal/TMSileroVADModelRunning.swift \
        Sources/TMSileroVAD/Internal/TMSileroVADResources.swift
git commit -m "internal: ModelRunning protocol + Resources bundle helper"
```

---

## Task 8: CoreMLRunner — real inference with state

**Files:**
- Create: `Sources/TMSileroVAD/Internal/TMSileroVADCoreMLRunner.swift`
- Create: `Tests/TMSileroVADTests/CoreMLRunnerIntegrationTests.swift`

- [ ] **Step 1: Write the failing integration tests**

```swift
import XCTest
import CoreML
@testable import TMSileroVAD

final class CoreMLRunnerIntegrationTests: XCTestCase {
    func test_balanced_runner_loads_and_predicts_silence() throws {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .balanced256ms)
        let runner = try TMSileroVADCoreMLRunner(spec: spec, computeUnits: .cpuOnly)
        let silence = Array(repeating: Float(0), count: spec.chunkLength)
        let prob = try runner.predict(chunk: silence)
        XCTAssertGreaterThanOrEqual(prob, 0)
        XCTAssertLessThanOrEqual(prob, 1)
        XCTAssertLessThan(prob, 0.5, "silence should produce low speech probability")
    }
    
    func test_realtime_runner_loads_and_predicts() throws {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        let runner = try TMSileroVADCoreMLRunner(spec: spec, computeUnits: .cpuOnly)
        let silence = Array(repeating: Float(0), count: spec.chunkLength)
        let prob = try runner.predict(chunk: silence)
        XCTAssertGreaterThanOrEqual(prob, 0)
        XCTAssertLessThanOrEqual(prob, 1)
    }
    
    func test_state_persists_across_calls_and_reset_clears_it() throws {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        let runner = try TMSileroVADCoreMLRunner(spec: spec, computeUnits: .cpuOnly)
        
        // Feed a sine-ish chunk a few times to drive state away from zero.
        var noise = [Float](repeating: 0, count: spec.chunkLength)
        for i in 0..<noise.count {
            noise[i] = Float.random(in: -0.3...0.3)
        }
        _ = try runner.predict(chunk: noise)
        _ = try runner.predict(chunk: noise)
        let probWithState = try runner.predict(chunk: Array(repeating: 0, count: spec.chunkLength))
        
        runner.reset()
        let probAfterReset = try runner.predict(chunk: Array(repeating: 0, count: spec.chunkLength))
        
        // Hidden/cell state being non-zero vs zero typically yields a different output for
        // an identical input. Exact values are brittle; assert non-equality with tolerance.
        XCTAssertNotEqual(probWithState, probAfterReset, accuracy: 1e-7)
    }
    
    func test_invalid_chunk_length_throws() throws {
        let spec = TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        let runner = try TMSileroVADCoreMLRunner(spec: spec, computeUnits: .cpuOnly)
        XCTAssertThrowsError(try runner.predict(chunk: [Float](repeating: 0, count: 100))) { error in
            guard case TMSileroVADError.invalidFrameLength(let expected, let got) = error else {
                XCTFail("wrong error: \(error)")
                return
            }
            XCTAssertEqual(expected, spec.chunkLength)
            XCTAssertEqual(got, 100)
        }
    }
}

extension XCTestCase {
    func XCTAssertNotEqual(_ a: Float, _ b: Float, accuracy: Float, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertGreaterThan(abs(a - b), accuracy, file: file, line: line)
    }
}
```

- [ ] **Step 2: Run tests (should fail)**

Expected: FAIL — type does not exist.

- [ ] **Step 3: Implement `TMSileroVADCoreMLRunner`**

```swift
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
```

- [ ] **Step 4: Run integration tests**

```bash
swift test --filter CoreMLRunnerIntegrationTests 2>&1 | tail -20
```

Expected: PASS, 4/4. If this fails because the host can't run CoreML on macOS (rare), gate the tests with `#if !targetEnvironment(simulator)` — but on macOS dev machines they should pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/TMSileroVAD/Internal/TMSileroVADCoreMLRunner.swift \
        Tests/TMSileroVADTests/CoreMLRunnerIntegrationTests.swift
git commit -m "runner: CoreMLRunner with stateful LSTM h/c + 64-sample context"
```

---

## Task 9: Mock runner + facade unit tests

**Files:**
- Create: `Tests/TMSileroVADTests/Helpers/MockModelRunner.swift`

- [ ] **Step 1: Write `MockModelRunner`**

```swift
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
```

- [ ] **Step 2: Commit (no separate test yet — used in next task)**

```bash
git add Tests/TMSileroVADTests/Helpers/MockModelRunner.swift
git commit -m "test: MockModelRunner for facade tests"
```

---

## Task 10: Facade `TMSileroVAD` — delegate, lock, queue, both APIs

**Files:**
- Modify: `Sources/TMSileroVAD/TMSileroVAD.swift`
- Create: `Tests/TMSileroVADTests/TMSileroVADFacadeTests.swift`

- [ ] **Step 1: Write the facade tests**

```swift
import XCTest
import AVFoundation
@testable import TMSileroVAD

final class TMSileroVADFacadeTests: XCTestCase {
    func test_processPCM_returns_events_for_each_chunk() throws {
        let mock = MockModelRunner()
        mock.probabilities = [0.9, 0.9, 0.1, 0.1]
        let vad = try TMSileroVAD.makeForTesting(
            config: .realtime,
            runner: mock,
            spec: TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        )
        let chunkLen = 512
        let samples = [Float](repeating: 0.05, count: chunkLen * 4)
        let events = try vad.processPCM(samples)
        XCTAssertEqual(events.count, 4)
    }
    
    func test_processPCM_buffers_partial_input() throws {
        let mock = MockModelRunner()
        mock.probabilities = [0.1, 0.1]
        let vad = try TMSileroVAD.makeForTesting(
            config: .realtime,
            runner: mock,
            spec: TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        )
        let events1 = try vad.processPCM([Float](repeating: 0, count: 300))
        XCTAssertEqual(events1.count, 0, "less than one chunk → no events")
        let events2 = try vad.processPCM([Float](repeating: 0, count: 300))
        XCTAssertEqual(events2.count, 1, "300+300 = 600 → one 512-chunk")
    }
    
    func test_processAudioBuffer_resamples_48k_to_16k_and_processes() throws {
        let mock = MockModelRunner()
        mock.probabilities = Array(repeating: 0.1, count: 100)
        let vad = try TMSileroVAD.makeForTesting(
            config: .realtime,
            runner: mock,
            spec: TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        )
        // 1 second of silence at 48k → 16000 samples → 16000/512 ≈ 31 chunks
        let buffer = PCMSynth.silence(seconds: 1.0, sampleRate: 48_000)
        let events = try vad.processAudioBuffer(buffer)
        XCTAssertGreaterThanOrEqual(events.count, 30)
        XCTAssertLessThanOrEqual(events.count, 32)
    }
    
    func test_delegate_called_with_same_events_on_configured_queue() throws {
        let mock = MockModelRunner()
        mock.probabilities = [0.1, 0.1, 0.1]
        let queue = DispatchQueue(label: "test.callback")
        let queueKey = DispatchSpecificKey<Int>()
        queue.setSpecific(key: queueKey, value: 42)
        
        var config = TMSileroVAD.SileroVADConfig.realtime
        config = TMSileroVAD.SileroVADConfig(
            variant: config.variant,
            startThreshold: config.startThreshold,
            endThreshold: config.endThreshold,
            minSpeechDurationMs: config.minSpeechDurationMs,
            minSilenceDurationMs: config.minSilenceDurationMs,
            computeUnits: config.computeUnits,
            callbackQueue: queue
        )
        
        let vad = try TMSileroVAD.makeForTesting(
            config: config,
            runner: mock,
            spec: TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        )
        let spy = DelegateSpy(queueKey: queueKey)
        vad.delegate = spy
        let exp = expectation(description: "delegate fires")
        spy.onEvent = { _ in
            if spy.events.count == 3 {
                exp.fulfill()
            }
        }
        _ = try vad.processPCM([Float](repeating: 0, count: 512 * 3))
        wait(for: [exp], timeout: 1.0)
        XCTAssertEqual(spy.events.count, 3)
        XCTAssertTrue(spy.allOnExpectedQueue, "delegate must run on configured queue")
    }
    
    func test_reset_clears_buffer_and_runner_and_state_machine() throws {
        let mock = MockModelRunner()
        mock.probabilities = Array(repeating: 0.1, count: 100)
        let vad = try TMSileroVAD.makeForTesting(
            config: .realtime,
            runner: mock,
            spec: TMSileroVADModelSpec.makeSpec(variant: .realtime32ms)
        )
        _ = try vad.processPCM([Float](repeating: 0, count: 300))
        vad.reset()
        XCTAssertEqual(mock.resetCalls, 1)
        // After reset the framebuffer should be empty: feeding 300 again should yield zero events.
        let events = try vad.processPCM([Float](repeating: 0, count: 300))
        XCTAssertEqual(events.count, 0)
    }
}

private final class DelegateSpy: TMSileroVADDelegate {
    let queueKey: DispatchSpecificKey<Int>
    var events: [TMSileroVAD.SileroVADEvent] = []
    var onEvent: ((TMSileroVAD.SileroVADEvent) -> Void)?
    var allOnExpectedQueue = true
    
    init(queueKey: DispatchSpecificKey<Int>) {
        self.queueKey = queueKey
    }
    
    func sileroVAD(_ vad: TMSileroVAD, didEmit event: TMSileroVAD.SileroVADEvent) {
        if DispatchQueue.getSpecific(key: queueKey) != 42 {
            allOnExpectedQueue = false
        }
        events.append(event)
        onEvent?(event)
    }
}
```

- [ ] **Step 2: Run tests (should fail to compile)**

Expected: FAIL — `makeForTesting`, `processPCM`, `processAudioBuffer`, `delegate`, `reset` don't exist.

- [ ] **Step 3: Implement the facade**

Replace `Sources/TMSileroVAD/TMSileroVAD.swift` with:

```swift
import AVFoundation
import CoreML
import Foundation
import os

public final class TMSileroVAD {
    public weak var delegate: TMSileroVADDelegate?
    public let config: SileroVADConfig
    
    private let spec: TMSileroVADModelSpec
    private let runner: TMSileroVADModelRunning
    private let frameBuffer: TMSileroVADFrameBuffer
    private let stateMachine: TMSileroVADStateMachine
    private let resampler: TMSileroVADResampler
    private var lock = os_unfair_lock_s()
    
    public convenience init(config: SileroVADConfig = .balanced) throws {
        let spec = TMSileroVADModelSpec.makeSpec(variant: config.variant)
        let runner = try TMSileroVADCoreMLRunner(spec: spec, computeUnits: config.computeUnits)
        try self.init(config: config, runner: runner, spec: spec)
    }
    
    init(
        config: SileroVADConfig,
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
    public func processPCM(_ samples: [Float]) throws -> [SileroVADEvent] {
        let events = try runLocked { try runProcess(samples: samples) }
        dispatchToDelegate(events)
        return events
    }
    
    @discardableResult
    public func processAudioBuffer(_ buffer: AVAudioPCMBuffer) throws -> [SileroVADEvent] {
        let resampled = try resampler.resample(buffer)
        return try processPCM(resampled)
    }
    
    // MARK: private
    
    private func runProcess(samples: [Float]) throws -> [SileroVADEvent] {
        let chunks = frameBuffer.append(samples)
        var events: [SileroVADEvent] = []
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
    
    private func dispatchToDelegate(_ events: [SileroVADEvent]) {
        guard !events.isEmpty else { return }
        let queue = config.callbackQueue
        let weakDelegate = { [weak self] in self?.delegate }
        queue.async { [weak self] in
            guard let self = self, let delegate = weakDelegate() else { return }
            for event in events {
                delegate.sileroVAD(self, didEmit: event)
            }
        }
    }
    
    /// Test-only escape hatch — DO NOT export from the module.
    static func makeForTesting(
        config: SileroVADConfig,
        runner: TMSileroVADModelRunning,
        spec: TMSileroVADModelSpec
    ) throws -> TMSileroVAD {
        return try TMSileroVAD(config: config, runner: runner, spec: spec)
    }
}
```

- [ ] **Step 4: Run all tests**

```bash
swift test 2>&1 | tail -25
```

Expected: PASS, all suites green. Watch for the facade tests in particular.

- [ ] **Step 5: Commit**

```bash
git add Sources/TMSileroVAD/TMSileroVAD.swift \
        Tests/TMSileroVADTests/TMSileroVADFacadeTests.swift
git commit -m "facade: TMSileroVAD with delegate + sync return + lock + resampling"
```

---

## Task 11: README and finalization

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write README**

```markdown
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

```bash
./scripts/download-models.sh
```

This vendors `silero-vad-unified-v6.0.0.mlmodelc` and `silero-vad-unified-256ms-v6.0.0.mlmodelc`
into `Sources/TMSileroVAD/Resources/`. Requires `git-lfs`
(`brew install git-lfs && git lfs install`).

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

    func sileroVAD(_ vad: TMSileroVAD, didEmit event: TMSileroVAD.SileroVADEvent) {
        switch event {
        case .speechStarted: handleStart()
        case .speechEnded:   handleEnd()
        case .speechContinuing, .silence: break
        }
    }
}
```

`processPCM` and `processAudioBuffer` also return `[SileroVADEvent]` synchronously, so you can use either delegate or sync-return (or both).

## Threading

- `processPCM`, `processAudioBuffer`, `reset` are thread-safe (internal `os_unfair_lock`).
- Delegate callbacks happen `async` on `config.callbackQueue` (default `.main`).
- For audio-thread safety: call from the AVAudioEngine tap directly is fine; the lock contention is microseconds.

## What's NOT included

- Audio recording / `AVAudioEngine` setup
- ASR
- RTM signaling
- Push-to-talk state machine

This Pod is a pure VAD engine. Wire your audio source and downstream consumers yourself.

## License

MIT.
```

- [ ] **Step 2: Verify build + test once more**

```bash
swift build && swift test 2>&1 | tail -25
```

Expected: all green.

- [ ] **Step 3: Lint the podspec**

```bash
pod spec lint TMSileroVAD.podspec --allow-warnings 2>&1 | tail -30
```

Expected: passes. (May warn about license / homepage if not real; that's fine for v0.1.0.)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, variants, usage, threading"
```

---

## Self-Review

**Spec coverage:**
- [x] CocoaPods Pod (`TMSileroVAD.podspec`) — Task 0
- [x] Two variant support (`.balanced256ms`, `.realtime32ms`) — Tasks 2, 3, 8
- [x] Delegate API — Task 10
- [x] Sync return API — Task 10
- [x] AVAudioConverter resampling — Task 6, integrated in Task 10
- [x] Strict 16 kHz path (`processPCM`) — Task 10
- [x] iOS 14 deployment — Task 0
- [x] curl-based model download — Task 1
- [x] Real CoreML inputs (`audio_input`/`hidden_state`/`cell_state` with no `_input` suffix) — Task 8
- [x] LSTM state persistence + reset — Task 8
- [x] 64-sample audio context carry — Task 8
- [x] Substring-match output names — Task 8
- [x] vDSP / dataPointer for fast PCM → MLMultiArray copy — Task 8
- [x] computeUnits exposed — Task 2 config + Task 8 runner
- [x] Thread safety via `os_unfair_lock` — Task 10
- [x] Configurable callback queue — Task 2 config + Task 10 dispatch

**Placeholder scan:** No "TBD", "implement later", "similar to". All steps include code.

**Type consistency:** `TMSileroVADModelSpec` properties referenced from runner + facade match (Task 3 → Tasks 8, 10). `TMSileroVAD.SileroVADEvent` shape matches all consumers. `TMSileroVADModelRunning.predict(chunk:)` matches both real and mock implementations.

**Known caveats:**
1. Task 8 integration tests require the .mlmodelc files to be downloaded first (Task 1). Running tests before Task 1 will fail with `modelNotFound` — that's correct behavior.
2. Resampler test counts use `accuracy: 2` because `AVAudioConverter` rounds output frame count.
3. The state-persistence runner test asserts inequality, not exact values — exact LSTM outputs are model/compute-unit dependent.
4. CocoaPods consumers will get `.mlmodelc` via `TMSileroVADResources.bundle`; SPM consumers get `Bundle.module`. The resolver tries both.
