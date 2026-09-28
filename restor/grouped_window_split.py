"""Manifest parsing and leakless aligned-view window splitting."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_MANIFEST_COLUMNS = {
    "recording_id",
    "kind",
    "dataset",
    "song",
    "family",
    "output_file",
}


def load_grouped_source_manifest(
    manifest_path: str | Path,
    wav_names: Iterable[str],
) -> dict[str, dict[str, object]]:
    """Load and validate one explicit row per WAV source view."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Grouped source manifest not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        columns = set(reader.fieldnames or ())
        missing_columns = REQUIRED_MANIFEST_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                "Grouped source manifest is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        rows = list(reader)

    by_output: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        output_file = row["output_file"].strip()
        recording_id = row["recording_id"].strip()
        kind = row["kind"].strip()
        family = row["family"].strip()
        if not output_file or Path(output_file).name != output_file:
            raise ValueError(f"Invalid manifest output_file: {output_file!r}")
        if not recording_id:
            raise ValueError(f"Empty recording_id for {output_file}")
        if kind not in {"full_mix", "section_mix"}:
            raise ValueError(f"Invalid kind {kind!r} for {output_file}")
        if kind == "full_mix" and family != "full_mix":
            raise ValueError(f"Full mix has invalid family {family!r}: {output_file}")
        if output_file in by_output:
            raise ValueError(f"Duplicate manifest output_file: {output_file}")
        normalized = dict(row)
        normalized.update(
            output_file=output_file,
            recording_id=recording_id,
            kind=kind,
            family=family,
        )
        by_output[output_file] = normalized
        grouped[recording_id].append(normalized)

    actual_wavs = set(wav_names)
    manifest_wavs = set(by_output)
    if actual_wavs != manifest_wavs:
        missing = sorted(manifest_wavs - actual_wavs)
        unexpected = sorted(actual_wavs - manifest_wavs)
        raise ValueError(
            "Grouped manifest/WAV inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    group_indices = {
        recording_id: index
        for index, recording_id in enumerate(sorted(grouped))
    }
    for recording_id, views in grouped.items():
        full_mix_count = sum(view["kind"] == "full_mix" for view in views)
        if full_mix_count != 1:
            raise ValueError(
                f"Recording {recording_id!r} must have exactly one full mix; "
                f"found {full_mix_count}"
            )
        view_keys = [(view["kind"], view["family"]) for view in views]
        if len(view_keys) != len(set(view_keys)):
            raise ValueError(f"Duplicate views for recording {recording_id!r}")
        for view in views:
            view["recording_group_idx"] = group_indices[recording_id]
    return by_output


def split_grouped_aligned_windows(
    window_records: list[dict[str, object]],
    seed: int,
    target_ratio: float,
    overlap_window_radius: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Randomly sample per-view windows and promote every aligned sibling.

    Candidates remain weighted per source WAV. A candidate inside the current
    overlap guard may later be promoted to validation. Non-validation windows
    overlapping any validation time in the same recording group are discarded.
    """
    if not window_records:
        raise ValueError("Cannot split zero grouped windows")
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(
            f"target_ratio must be between 0 and 1, got {target_ratio}"
        )
    if overlap_window_radius < 0:
        raise ValueError("overlap_window_radius cannot be negative")

    by_global: dict[int, dict[str, object]] = {}
    by_group_start: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_group_window: dict[tuple[int, int], list[int]] = defaultdict(list)
    for record in window_records:
        global_idx = int(record["global_window_idx"])
        source_idx = int(record["source_idx"])
        local_idx = int(record["source_window_idx"])
        start_sample = int(record["start_sample"])
        group_idx = int(record["recording_group_idx"])
        if global_idx in by_global:
            raise ValueError(f"Duplicate global window index: {global_idx}")
        by_global[global_idx] = record
        by_group_start[(group_idx, start_sample)].append(global_idx)
        by_group_window[(group_idx, local_idx)].append(global_idx)
        if source_idx < 0 or local_idx < 0 or start_sample < 0 or group_idx < 0:
            raise ValueError(f"Negative grouped-window provenance: {record}")

    all_global = set(by_global)
    candidates = sorted(all_global)
    random.Random(seed).shuffle(candidates)
    validate: set[int] = set()
    discarded: set[int] = set()

    for candidate in candidates:
        record = by_global[candidate]
        group_idx = int(record["recording_group_idx"])
        local_idx = int(record["source_window_idx"])
        start_sample = int(record["start_sample"])
        aligned = set(by_group_start[(group_idx, start_sample)])
        if aligned <= validate:
            continue

        # Guarded windows are allowed to become validation later. Promotion is
        # atomic across every full-mix/section view at this recording time.
        validate.update(aligned)
        discarded.difference_update(aligned)
        for neighbor_idx in range(
            local_idx - overlap_window_radius,
            local_idx + overlap_window_radius + 1,
        ):
            for neighbor_global in by_group_window.get(
                (group_idx, neighbor_idx), ()
            ):
                if neighbor_global not in validate:
                    discarded.add(neighbor_global)

        train_count = len(all_global) - len(validate) - len(discarded)
        usable_count = train_count + len(validate)
        if usable_count and len(validate) / usable_count >= target_ratio:
            break
    else:
        raise RuntimeError("Unable to reach grouped validation ratio")

    # Recompute the guard from the final validation set. This is both simpler
    # to audit and protects against future changes to incremental bookkeeping.
    discarded = _grouped_overlap_guard(
        validate,
        by_global,
        by_group_window,
        overlap_window_radius,
    )
    train = all_global - validate - discarded
    audit_grouped_split(
        window_records,
        train,
        validate,
        discarded,
        overlap_window_radius,
    )
    to_records = lambda indices: [by_global[index] for index in sorted(indices)]
    return to_records(train), to_records(validate), to_records(discarded)


def _grouped_overlap_guard(
    validate: set[int],
    by_global: dict[int, dict[str, object]],
    by_group_window: dict[tuple[int, int], list[int]],
    overlap_window_radius: int,
) -> set[int]:
    discarded: set[int] = set()
    for global_idx in validate:
        record = by_global[global_idx]
        group_idx = int(record["recording_group_idx"])
        local_idx = int(record["source_window_idx"])
        for neighbor_idx in range(
            local_idx - overlap_window_radius,
            local_idx + overlap_window_radius + 1,
        ):
            for neighbor_global in by_group_window.get(
                (group_idx, neighbor_idx), ()
            ):
                if neighbor_global not in validate:
                    discarded.add(neighbor_global)
    return discarded


def audit_grouped_split(
    window_records: list[dict[str, object]],
    train: set[int],
    validate: set[int],
    discarded: set[int],
    overlap_window_radius: int,
) -> None:
    """Raise if a grouped split violates partition or leakage invariants."""
    by_global = {
        int(record["global_window_idx"]): record for record in window_records
    }
    all_global = set(by_global)
    if train & validate or train & discarded or validate & discarded:
        raise RuntimeError("Grouped split sets are not disjoint")
    if train | validate | discarded != all_global:
        raise RuntimeError("Grouped split sets do not partition all windows")

    by_group_start: dict[tuple[int, int], set[int]] = defaultdict(set)
    train_positions: dict[int, set[int]] = defaultdict(set)
    for global_idx, record in by_global.items():
        group_idx = int(record["recording_group_idx"])
        local_idx = int(record["source_window_idx"])
        start_sample = int(record["start_sample"])
        by_group_start[(group_idx, start_sample)].add(global_idx)
        if global_idx in train:
            train_positions[group_idx].add(local_idx)

    for global_idx in validate:
        record = by_global[global_idx]
        group_idx = int(record["recording_group_idx"])
        local_idx = int(record["source_window_idx"])
        start_sample = int(record["start_sample"])
        if not by_group_start[(group_idx, start_sample)] <= validate:
            raise RuntimeError(
                "Aligned recording views were split across validation and "
                f"another set at group={group_idx}, window={local_idx}"
            )
        if any(
            abs(local_idx - train_idx) <= overlap_window_radius
            for train_idx in train_positions[group_idx]
        ):
            raise RuntimeError(
                "Train/validation overlap in recording group "
                f"{group_idx} around window {local_idx}"
            )
