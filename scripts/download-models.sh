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
