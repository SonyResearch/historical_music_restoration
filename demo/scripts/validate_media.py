#!/usr/bin/env python3
"""Validate the checked-in demo bundle against its generated provenance manifest."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/generated/audio-manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    records = json.loads(MANIFEST.read_text())
    if len(records) != 36:
        raise RuntimeError(f"Expected 36 audio assets, found {len(records)}")

    failures: list[str] = []
    for record in records:
        output = ROOT / record["output"]
        if not output.is_file() or digest(output) != record["sha256"]:
            failures.append(f"missing or hash mismatch: {output}")
            continue
        if record["sample_rate_hz"] != 44100 or record["channels"] != 2:
            failures.append(f"format mismatch: {output}")
        if not 5.0 <= record["duration_seconds"] <= 5.1:
            failures.append(f"duration mismatch: {output} ({record['duration_seconds']})")
        process = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(output), "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        if process.returncode:
            failures.append(f"decode failure: {output}")
            continue
        mean = re.search(r"mean_volume:\s+(-?[0-9.]+) dB", process.stderr)
        peak = re.search(r"max_volume:\s+(-?[0-9.]+) dB", process.stderr)
        if not mean or not peak:
            failures.append(f"missing level statistics: {output}")
            continue
        if float(mean.group(1)) < -80:
            failures.append(f"silent asset: {output}")
        if float(peak.group(1)) > 0:
            failures.append(f"clipped asset: {output}")

    if failures:
        raise RuntimeError("Media validation failed:\n" + "\n".join(failures))
    print("Validated 36 MP3 assets: hashes, decode, 44.1 kHz, stereo, duration, non-silence, and peak safety.")


if __name__ == "__main__":
    main()
