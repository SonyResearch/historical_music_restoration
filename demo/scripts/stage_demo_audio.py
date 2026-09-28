#!/usr/bin/env python3
"""Build the small, precomputed web-audio bundle from audited evaluation outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESTOR = Path("/data/steven/restor")
STAGED = RESTOR / "reports/listening_test_table5/golisten_babe2_25/staged_audio"
MANIFEST = RESTOR / "reports/listening_test_table5/golisten_babe2_25/audio_manifest.csv"
BEHM = RESTOR / "IA_test_set_inference_outputs/behm"

HISTORICAL = [
    ("ballet-egyptian", "orchestra_ballet_egyptian_110", "78_ballet-egyptian-nos-1-and-2_american-symphony-orchestra-alexandre-luigini_gbia0078132a.flac", 110.0),
    ("rigoletto", "orchestra_rigoletto_75", "78_rigoletto-selection-part-1_american-symphony-orchestra-g-verdi_gbia0298845a.flac", 75.0),
    ("hungarian-dance", "orchestra_hungarian_dance_5_95", "78_hungarian-dance-no-5_philadelphia-symphony-orchestra-brahms-leopold-stokowski_gbia0183029a.flac", 95.0),
    ("minuetto", "light_minuetto_80", "78_minuetto_american-symphony-orchestra-g-bolzoni_gbia0078138a.flac", 80.0),
    ("scented-violets", "light_scented_violets_50", "78_scented-violets_peerless-orchestra-jules-reynard_gbia0178673b.flac", 50.0),
    ("lady-bird", "light_lady_bird_95", "78_lady-bird-tango_the-peerless-orchestra-p-s-robinson_gbia3038426b.flac", 95.0),
]

def run(*args: str) -> None:
    subprocess.run(args, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(source: Path, output: Path, offset: float | None = None, gain: float = 0.891250938) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if offset is not None:
        command += ["-ss", f"{offset:.3f}"]
    command += ["-i", str(source), "-t", "5.000", "-af", f"volume={gain:.9f}", "-ar", "44100", "-ac", "2", "-codec:a", "libmp3lame", "-b:a", "192k", str(output)]
    run(*command)


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {"sample_rate_hz": int(stream["sample_rate"]), "channels": int(stream["channels"]), "duration_seconds": float(stream["duration"])}


def common_gains() -> dict[str, float]:
    gains: dict[str, float] = {}
    with MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == "INPUT":
                gains[row["window_id"]] = float(row["common_gain"])
    return gains


def main() -> None:
    records: list[dict[str, object]] = []
    gains = common_gains()
    web_safety_gain = 0.891250938  # common -1 dB applied after the study's shared gain

    for example_id, window_id, filename, offset in HISTORICAL:
        destinations = {
            "input": (STAGED / window_id / "INPUT.wav", None, web_safety_gain),
            "babe2-p": (STAGED / window_id / "BABE2_PRETRAINED.wav", None, web_safety_gain),
            "babe2-fos": (STAGED / window_id / "BABE2_FMS.wav", None, web_safety_gain),
            "samecfm": (STAGED / window_id / "CFM40.wav", None, web_safety_gain),
            "behm-p": (BEHM / "denoiser_only_44k_compat" / filename, offset, gains[window_id] * web_safety_gain),
            "behm-fos": (BEHM / "fms_leakless" / filename, offset, gains[window_id] * web_safety_gain),
        }
        for condition, (source, source_offset, gain) in destinations.items():
            output = ROOT / "public/audio/historical" / example_id / f"{condition}.mp3"
            encode(source, output, source_offset, gain)
            records.append({"example": example_id, "condition": condition, "source": str(source), "source_offset_seconds": source_offset or 0, "applied_gain": gain, "output": str(output.relative_to(ROOT)), "sha256": sha256(output), **probe(output)})

    manifest_dir = ROOT / "data/generated"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "audio-manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    with (manifest_dir / "audio-manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Staged {len(records)} aligned audio assets.")


if __name__ == "__main__":
    main()
