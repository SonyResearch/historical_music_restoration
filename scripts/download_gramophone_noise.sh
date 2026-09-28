#!/usr/bin/env bash
# Download, verify, and unpack the stage-5 Gramophone Record Noise Dataset.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-$ROOT_DIR/data/gramophone_record_noise}"
URL="${GRAMOPHONE_NOISE_URL:-http://research.spa.aalto.fi/publications/papers/icassp22-denoising/media/datasets/Gramophone_Record_Noise_Dataset.zip}"
EXPECTED_SHA256="${GRAMOPHONE_NOISE_SHA256:-b97a78a4da7ed7e05a7b10db31cbcab37ffe927d14e1c1c6948f55da0b720a2c}"
ARCHIVE="${DESTINATION}.zip.part"
SUCCESS="$DESTINATION/_SUCCESS"

if [[ -f "$SUCCESS" ]]; then
    count="$(find "$DESTINATION" -type f -iname '*.wav' | wc -l)"
    echo "Gramophone noise dataset already ready: $DESTINATION ($count WAV files)"
    exit 0
fi
if [[ -e "$DESTINATION" ]] && [[ -n "$(find "$DESTINATION" -mindepth 1 -print -quit)" ]]; then
    echo "Destination exists but is incomplete: $DESTINATION" >&2
    echo "Move it aside or remove it, then rerun this script." >&2
    exit 1
fi

mkdir -p "$(dirname "$DESTINATION")"
actual=""
if [[ -f "$ARCHIVE" ]]; then
    actual="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
fi
if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
    curl --fail --location --retry 3 --continue-at - --output "$ARCHIVE" "$URL"
    actual="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
fi
if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
    echo "Dataset SHA-256 verification failed." >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    echo "Actual:   $actual" >&2
    exit 1
fi

mkdir -p "$DESTINATION"
unzip -q "$ARCHIVE" -d "$DESTINATION"
count="$(find "$DESTINATION" -type f -iname '*.wav' | wc -l)"
if [[ "$count" -eq 0 ]]; then
    echo "Archive extraction produced no WAV files." >&2
    exit 1
fi
touch "$SUCCESS"
rm -f "$ARCHIVE"
echo "Downloaded and verified: $DESTINATION ($count WAV files)"
