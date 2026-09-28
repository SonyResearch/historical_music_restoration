#!/usr/bin/env bash
# Download the paper checkpoint from the GitHub Release, as in BEHM-GAN.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESTINATION="${CHECKPOINT:-$ROOT_DIR/checkpoints/samecfm_40m_fos.pt}"
URL="${CHECKPOINT_URL:-https://github.com/stevencho24/End-to-End_historical_music_restoration/releases/download/v1.0.0/samecfm_40m_fos.pt}"
EXPECTED_SHA256="${CHECKPOINT_SHA256:-dcf0100ed1268201bc5e0db134d1d9677e32118b75a10e2d1d211d3d12dad4ca}"

mkdir -p "$(dirname "$DESTINATION")"
if [[ -f "$DESTINATION" ]]; then
    actual="$(sha256sum "$DESTINATION" | awk '{print $1}')"
    if [[ "$actual" == "$EXPECTED_SHA256" ]]; then
        echo "Checkpoint already present and verified: $DESTINATION"
        exit 0
    fi
    echo "Existing checkpoint has the wrong SHA-256: $DESTINATION" >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    echo "Actual:   $actual" >&2
    exit 1
fi

temporary="${DESTINATION}.part"
trap 'rm -f "$temporary"' EXIT
curl --fail --location --retry 3 --continue-at - --output "$temporary" "$URL"
actual="$(sha256sum "$temporary" | awk '{print $1}')"
if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
    echo "Checkpoint SHA-256 verification failed." >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    echo "Actual:   $actual" >&2
    exit 1
fi
mv "$temporary" "$DESTINATION"
trap - EXIT
echo "Downloaded and verified: $DESTINATION"
