import math
import os
import random
import re
import shutil
from collections import Counter

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .corruption import AudioCorruptor
from .dataset import StemMixDataset
from .diffusion_utils import (
    ensure_channel_first,
    normalize_latent,
)
from .grouped_window_split import (
    load_grouped_source_manifest,
    split_grouped_aligned_windows,
)
from .models import build_codec


class data_precompute:
    def __init__(self, cfg):
        self.cfg = cfg
        self._validate_cfg()
        self.corruptor = AudioCorruptor(self.cfg["corruption"])
        if not torch.cuda.is_available():
            raise RuntimeError("data_precompute requires CUDA for GPU precompute.")
        # torchrun gives each worker a LOCAL_RANK. In the single-GPU launch
        # this variable is absent and the original cuda:0 behavior is kept.
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if self.local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK={self.local_rank} but only "
                f"{torch.cuda.device_count()} CUDA devices are visible"
            )
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 0
        )
        self.world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 1
        )
        self.waveform_flac_only = bool(
            self.cfg.get("precompute", {}).get("waveform_flac_only", False)
        )
        if self.waveform_flac_only:
            self.codec = None
            self.sample_rate = int(
                self.cfg["dataset"].get("sample_rate", 44_100)
            )
        else:
            self.codec = build_codec(self.cfg["codec"]["name"], self.device)
            self.codec.eval()
            self.sample_rate = self.codec.sample_rate
        self.segment_samples = int(
            self.cfg["dataset"]["segment_duration"] * self.sample_rate
        )
        self.hop_samples = self._resolve_hop_samples()
        self.overlap_window_radius = math.ceil(
            self.segment_samples / self.hop_samples
        ) - 1
        self.target_lufs = self.cfg.get("precompute", {}).get("target_lufs", -23.0)
        self.source_loudness_scope = self.cfg.get("precompute", {}).get(
            "source_loudness_scope", "window"
        )
        if self.source_loudness_scope not in {
            "window",
            "whole_song_pre_normalized",
        }:
            raise ValueError(
                "precompute.source_loudness_scope must be 'window' or "
                f"'whole_song_pre_normalized', got {self.source_loudness_scope!r}"
            )
        pcfg = self.cfg.get("precompute", {})
        self.related_view_split = pcfg.get(
            "related_view_split", "independent_sources"
        )
        if self.related_view_split not in {
            "independent_sources",
            "aligned_recording_views",
        }:
            raise ValueError(
                "precompute.related_view_split must be 'independent_sources' "
                f"or 'aligned_recording_views', got {self.related_view_split!r}"
            )
        self.latent_stats_scope = pcfg.get("latent_stats_scope", "all_windows")
        if self.latent_stats_scope not in {"all_windows", "train_windows"}:
            raise ValueError(
                "precompute.latent_stats_scope must be 'all_windows' or "
                f"'train_windows', got {self.latent_stats_scope!r}"
            )
        if (
            self.related_view_split == "aligned_recording_views"
            and not pcfg.get("multi_song_manifest")
        ):
            raise ValueError(
                "aligned_recording_views requires precompute.multi_song_manifest"
            )
        self.latent_mean = None
        self.latent_std = None
        self.default_test_files = []

    def _validate_cfg(self):
        required = {
            "codec": ["name"],
            "corruption": [],
            "dataset": ["root", "segment_duration"],
            "training": ["batch_size", "num_workers", "seed"],
        }
        missing = []
        for section, keys in required.items():
            if section not in self.cfg:
                missing.append(section)
                continue
            for key in keys:
                if key not in self.cfg[section]:
                    missing.append(f"{section}.{key}")
        if missing:
            raise ValueError(
                "Precompute config is missing required keys: "
                + ", ".join(missing)
            )

    def _assert_on_device(self, name, tensor):
        if tensor.device != self.device:
            raise RuntimeError(
                f"{name} is on {tensor.device}, expected {self.device}."
            )

    def _resolve_hop_samples(self):
        """Resolve hop size from config, defaulting this variant to 1/10 segment."""
        pcfg = self.cfg.get("precompute", {})
        if pcfg.get("hop_samples") is not None:
            hop_samples = int(pcfg["hop_samples"])
        elif pcfg.get("hop_duration") is not None:
            hop_samples = int(round(float(pcfg["hop_duration"]) * self.sample_rate))
        elif pcfg.get("hop_fraction") is not None:
            hop_samples = int(round(float(pcfg["hop_fraction"]) * self.segment_samples))
        else:
            hop_divisor = float(pcfg.get("hop_divisor", 10))
            hop_samples = int(round(self.segment_samples / hop_divisor))

        if hop_samples <= 0:
            raise ValueError(f"hop_samples must be positive, got {hop_samples}")
        if hop_samples > self.segment_samples:
            raise ValueError(
                "hop_samples cannot exceed segment_samples for this precompute split: "
                f"hop_samples={hop_samples}, segment_samples={self.segment_samples}"
            )
        return hop_samples
    
    def _build_data_one_song(self): 
        """TEMPORARY SINGLE-SONG TRAINING MODE."""
        dcfg = self.cfg["dataset"]
        dcfg["mode"] = "precompute"  # "precompute" or "real_time"
        root = dcfg["root"]

        #get all song directories in root (for this just one)
        song_dirs = sorted(
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
        print(f"Found {song_dirs[0]} in {root}")

        sr = self.codec.sample_rate

        # train_dirs = song_dirs
        # val_dirs = song_dirs
        # self.train_set = StemMixDataset(train_dirs, dcfg, sr)
        # self.val_set = StemMixDataset(val_dirs, dcfg, sr)
        # print(f"Train : {len(self.train_set)} samples from {len(train_dirs)} songs")
        # print(f"Val   : {len(self.val_set)} samples from {len(val_dirs)} songs")

        one_song_dir = song_dirs[0]
        self.one_song_set = StemMixDataset([one_song_dir], dcfg, sr)
        print(f"one_song_set : {self.one_song_set.name()}")

        song = self.one_song_set.songs[0]
        all_window_indices = list(range(self._num_full_windows(song)))
        self._compute_lufs_window_latent_stats(song, all_window_indices)

        #num of degradation copies per one clean 10s file
        degrad_num = self.cfg.get("precompute", {}).get("degradations_per_clean", 10)
        output_root = self.cfg.get("precompute", {}).get("output_root", "data/precomputed")
        self._save_precompute_pairs(output_root, degrad_num, song, all_window_indices)

    def build_data_mult_song(self):
        """Precompute one globally split dataset from multiple full-mix WAV files.

        Window IDs are global and collision-free. With an aligned-view manifest,
        full-mix and section views of one recording are split as one time group.
        """
        pcfg = self.cfg.get("precompute", {})
        source_dir, songs, window_records = self._discover_multi_song_windows(
            log_sources=True
        )
        split_records = self._stochastic_split_multi_song_windows(
            window_records,
            self.cfg["training"]["seed"],
        )
        stats_records = (
            split_records[0]
            if self.latent_stats_scope == "train_windows"
            else window_records
        )
        self._compute_lufs_multi_song_latent_stats(
            songs, stats_records
        )

        degrad_num = int(pcfg.get("degradations_per_clean", 20))
        output_root = pcfg.get("output_root", "data/precomputed")
        output_name = pcfg.get(
            "multi_song_output_name", os.path.basename(source_dir)
        )
        self._save_precompute_pairs_multi_song(
            output_root,
            output_name,
            degrad_num,
            source_dir,
            songs,
            window_records,
            split_records=split_records,
        )

    def build_data_mult_song_waveform_flac(self):
        """Save clean/corrupted waveforms using the latent job's exact split."""
        if not self.waveform_flac_only:
            raise ValueError(
                "Waveform-FLAC precompute requires "
                "precompute.waveform_flac_only=true"
            )
        if self.world_size != 1:
            raise ValueError("Waveform-FLAC precompute currently requires one GPU")

        pcfg = self.cfg.get("precompute", {})
        source_dir, songs, window_records = self._discover_multi_song_windows(
            log_sources=True
        )
        split_records = self._stochastic_split_multi_song_windows(
            window_records,
            self.cfg["training"]["seed"],
        )
        self._save_waveform_flac_pairs_multi_song(
            output_root=pcfg.get("output_root", "data/precomputed"),
            output_name=pcfg.get(
                "multi_song_output_name", os.path.basename(source_dir)
            ),
            degrad_num=int(pcfg.get("degradations_per_clean", 20)),
            source_dir=source_dir,
            songs=songs,
            window_records=window_records,
            split_records=split_records,
        )

    def build_data_mult_song_distributed(self):
        """Run exact multi-song precompute with one independent worker per GPU."""
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "build_data_mult_song_distributed requires a torchrun process group"
            )
        if self.world_size < 2:
            raise ValueError("Distributed precompute requires at least two workers")

        pcfg = self.cfg.get("precompute", {})
        # Every worker discovers the same sorted records locally. This avoids
        # broadcasting a large Python manifest and keeps global IDs identical.
        source_dir, songs, window_records = self._discover_multi_song_windows(
            log_sources=self.rank == 0
        )
        split_records = self._stochastic_split_multi_song_windows(
            window_records,
            self.cfg["training"]["seed"],
        )
        stats_records = (
            split_records[0]
            if self.latent_stats_scope == "train_windows"
            else window_records
        )
        if self.rank == 0:
            print(
                f"Distributed precompute: {self.world_size} GPU workers; "
                f"batch_size={self.cfg['training']['batch_size']} per GPU",
                flush=True,
            )

        # Phase 1: each GPU encodes a different quarter of all clean windows.
        # Three tiny all-reduces combine sum/sum-of-squares/count into the same
        # exact fixed latent mean/std on every rank.
        self._compute_lufs_multi_song_latent_stats_distributed(
            songs, stats_records
        )

        degrad_num = int(pcfg.get("degradations_per_clean", 20))
        output_root = pcfg.get("output_root", "data/precomputed")
        output_name = pcfg.get(
            "multi_song_output_name", os.path.basename(source_dir)
        )
        # Phase 2 is the largest saving: retained windows are divided across
        # ranks before creating 20 corruptions and 20 pair files per window.
        self._save_precompute_pairs_multi_song(
            output_root,
            output_name,
            degrad_num,
            source_dir,
            songs,
            window_records,
            split_records=split_records,
        )

    def _discover_multi_song_windows(self, log_sources):
        """Discover full-mix WAVs and assign deterministic global window IDs."""
        pcfg = self.cfg.get("precompute", {})
        source_dir = pcfg.get("multi_song_dir")
        if not source_dir:
            folder_name = pcfg.get("multi_song_folder", "beethoven_straussII")
            source_dir = os.path.join(self.cfg["dataset"]["root"], folder_name)
        source_dir = os.path.abspath(source_dir)
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Multi-song folder not found: {source_dir}")

        wav_names = sorted(
            name
            for name in os.listdir(source_dir)
            if name.lower().endswith(".wav")
            and os.path.isfile(os.path.join(source_dir, name))
        )
        if not wav_names:
            raise ValueError(f"No WAV files found in multi-song folder: {source_dir}")

        manifest_by_output = None
        manifest_path = pcfg.get("multi_song_manifest")
        if manifest_path:
            if not os.path.isabs(manifest_path):
                manifest_path = os.path.join(source_dir, manifest_path)
            manifest_by_output = load_grouped_source_manifest(
                manifest_path,
                wav_names,
            )
        elif self.related_view_split == "aligned_recording_views":
            raise ValueError(
                "aligned_recording_views requires a grouped source manifest"
            )

        songs = []
        window_records = []
        global_window_idx = 0
        for source_idx, wav_name in enumerate(wav_names):
            path = os.path.join(source_dir, wav_name)
            manifest_row = (
                manifest_by_output[wav_name]
                if manifest_by_output is not None
                else {
                    "recording_id": wav_name,
                    "recording_group_idx": source_idx,
                    "kind": "full_mix",
                    "family": "full_mix",
                    "dataset": "unmanifested",
                    "song": os.path.splitext(wav_name)[0],
                }
            )
            info = torchaudio.info(path)
            if info.sample_rate != self.sample_rate:
                raise ValueError(
                    f"Multi-song precompute requires {self.sample_rate} Hz audio; "
                    f"got {info.sample_rate} Hz for {path}"
                )
            song = {
                "dir": source_dir,
                "stems": [wav_name],
                "orig_sr": info.sample_rate,
                "orig_frames": info.num_frames,
                "recording_id": manifest_row["recording_id"],
                "recording_group_idx": int(
                    manifest_row["recording_group_idx"]
                ),
                "view_kind": manifest_row["kind"],
                "family": manifest_row["family"],
                "dataset": manifest_row["dataset"],
                "song": manifest_row["song"],
            }
            num_windows = self._num_full_windows(song)
            if num_windows <= 0:
                raise ValueError(f"Song is shorter than one segment: {path}")
            songs.append(song)
            for local_window_idx in range(num_windows):
                window_records.append(
                    {
                        "global_window_idx": global_window_idx,
                        "source_idx": source_idx,
                        "source_window_idx": local_window_idx,
                        "start_sample": local_window_idx * self.hop_samples,
                        "recording_group_idx": song["recording_group_idx"],
                        "view_kind": song["view_kind"],
                        "family": song["family"],
                    }
                )
                global_window_idx += 1
            if log_sources:
                print(
                    f"Source {source_idx:02d}: {wav_name} ({num_windows} windows; "
                    f"recording={song['recording_id']}; view={song['family']})",
                    flush=True,
                )

        if log_sources:
            print(
                f"Found {len(songs)} source views and {len(window_records)} "
                f"total windows in {source_dir}",
                flush=True,
            )
        return source_dir, songs, window_records

    def _build_data_test(self, test_files=None, output_root=None, degrad_num=None):
        """Precompute held-out test latent pairs from explicit audio file paths."""
        test_files = list(test_files or self.default_test_files)
        if not test_files:
            raise ValueError("No test files were provided for precompute test data")
        missing = [path for path in test_files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "Missing test audio files:\n" + "\n".join(missing)
            )

        self._ensure_latent_stats()
        output_root = output_root or self.cfg.get("precompute", {}).get(
            "output_root", "data/precomputed"
        )
        degrad_num = degrad_num or self.cfg.get("precompute", {}).get(
            "degradations_per_clean", 20
        )
        self._save_precompute_test_pairs(output_root, degrad_num, test_files)

    def _ensure_latent_stats(self):
        """Compute the same clean latent stats used by train/validation precompute."""
        if self.latent_mean is not None and self.latent_std is not None:
            return

        dcfg = dict(self.cfg["dataset"])
        dcfg["mode"] = "precompute"
        root = dcfg["root"]
        song_dirs = sorted(
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
        if not song_dirs:
            raise ValueError(f"No song directories found under dataset root: {root}")

        stats_set = StemMixDataset([song_dirs[0]], dcfg, self.codec.sample_rate)
        song = stats_set.songs[0]
        all_window_indices = list(range(self._num_full_windows(song)))
        self._compute_lufs_window_latent_stats(song, all_window_indices)

    @torch.no_grad()
    def _save_precompute_test_pairs(self, output_root, degrad_num, test_files):
        """Save held-out test latent pairs and matching ground-truth WAV clips."""
        test_dir = os.path.join(output_root, "ia_classical_FINAL_precompute_test")
        ground_truth_dir = os.path.join(test_dir, "ground_truth")
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)

        metadata = {
            "format": "ddpm_latent_pair_v1",
            "split": "test",
            "sample_rate": self.codec.sample_rate,
            "segment_samples": self.segment_samples,
            "segment_duration": self.segment_samples / self.codec.sample_rate,
            "hop_samples": self.hop_samples,
            "hop_duration": self.hop_samples / self.codec.sample_rate,
            "degradations_per_clean": degrad_num,
            "pair_keys": ["z_cond", "z_clean"],
            "ground_truth_dir": ground_truth_dir,
            "source_files": test_files,
            "target_lufs": self.target_lufs,
            "waveform_normalization": self._waveform_normalization_description(),
            "latent_normalization": "SAME latents normalized with LUFS-normalized train/precompute clean-latent mean/std",
            "latent_mean": self.latent_mean.detach().cpu(),
            "latent_std": self.latent_std.detach().cpu(),
            "corruption": self.cfg["corruption"],
        }
        torch.save(metadata, os.path.join(test_dir, "metadata.pt"))
        torch.save(metadata, os.path.join(ground_truth_dir, "metadata.pt"))
        self._write_test_methods_txt(test_dir, metadata)

        pair_count = 0
        window_idx = 0
        batch = []
        tcfg = self.cfg["training"]
        batch_size = tcfg["batch_size"]

        for source_idx, path in enumerate(test_files):
            info = torchaudio.info(path)
            if info.sample_rate != self.codec.sample_rate:
                raise ValueError(
                    "Test precompute currently requires matching sample rate: "
                    f"{path} has {info.sample_rate}, expected {self.codec.sample_rate}"
                )

            hop_samples = self.hop_samples
            num_windows = 1 + (info.num_frames - self.segment_samples) // hop_samples
            if num_windows <= 0:
                raise ValueError(
                    f"Test file is shorter than one segment: {path}"
                )

            for local_window_idx in range(num_windows):
                start = local_window_idx * hop_samples
                audio, sr = torchaudio.load(
                    path,
                    frame_offset=start,
                    num_frames=self.segment_samples,
                )
                if sr != self.codec.sample_rate:
                    raise ValueError(f"Unexpected sample rate {sr} while loading {path}")
                if audio.shape[0] > 1:
                    audio = audio.mean(0, keepdim=True)
                if audio.shape[-1] != self.segment_samples:
                    raise ValueError(
                        f"Loaded incomplete test window from {path} at frame {start}"
                    )
                audio = self._normalize_lufs(audio)

                batch.append((window_idx, source_idx, local_window_idx, audio))
                if len(batch) == batch_size:
                    pair_count += self._flush_test_precompute_batch(
                        batch, test_dir, ground_truth_dir, degrad_num
                    )
                    batch = []
                window_idx += 1

        if batch:
            pair_count += self._flush_test_precompute_batch(
                batch, test_dir, ground_truth_dir, degrad_num
            )

        print(
            f"Precompute test saved {pair_count} test pairs "
            f"from {window_idx} clean windows."
        )

    def _flush_test_precompute_batch(self, batch, test_dir, ground_truth_dir, degrad_num):
        """Encode and save one in-memory batch of test windows."""
        clean = torch.stack([item[3] for item in batch], dim=0)
        z_conds, z_clean = self._encode_ddpm_pair_precompute(
            clean,
            degrads=degrad_num,
        )

        pair_count = 0
        for batch_idx, (window_idx, source_idx, local_window_idx, audio) in enumerate(batch):
            z_clean_item = z_clean[batch_idx].detach().cpu()
            torchaudio.save(
                os.path.join(ground_truth_dir, f"window_{window_idx:06d}.wav"),
                audio.detach().cpu().clamp(-1.0, 1.0),
                self.codec.sample_rate,
            )

            for degradation_idx, z_cond in enumerate(z_conds):
                pair = {
                    "z_cond": z_cond[batch_idx].detach().cpu(),
                    "z_clean": z_clean_item.clone(),
                    "source_idx": source_idx,
                    "source_window_idx": local_window_idx,
                }
                file_name = (
                    f"window_{window_idx:06d}_"
                    f"degradation_{degradation_idx:02d}.pt"
                )
                torch.save(pair, os.path.join(test_dir, file_name))
                pair_count += 1
        return pair_count

    def _write_test_methods_txt(self, test_dir, metadata):
        """Write a readable provenance note for the generated test set."""
        methods_path = os.path.join(test_dir, "METHODS.txt")
        lines = [
            "DDPM SAME latent precompute test set",
            "",
            "Source files:",
            *[f"- {path}" for path in metadata["source_files"]],
            "",
            "Windowing:",
            f"- sample_rate: {metadata['sample_rate']}",
            f"- segment_samples: {metadata['segment_samples']}",
            f"- segment_duration_sec: {metadata['segment_duration']}",
            f"- hop_samples: {metadata['hop_samples']}",
            f"- hop_duration_sec: {metadata['hop_duration']}",
            "- partial final windows are dropped",
            "",
            "Normalization:",
            f"- waveform: {metadata['waveform_normalization']}",
            f"- latent: {metadata['latent_normalization']}",
            "",
            "Degradation:",
            f"- corruptions_per_clean_window: {metadata['degradations_per_clean']}",
            f"- corruption_config: {metadata['corruption']}",
            "",
            "Output:",
            f"- pair_keys: {metadata['pair_keys']}",
            f"- ground_truth_dir: {metadata['ground_truth_dir']}",
        ]
        with open(methods_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    @torch.no_grad()
    def _save_waveform_flac_pairs_multi_song(
        self,
        output_root,
        output_name,
        degrad_num,
        source_dir,
        songs,
        window_records,
        split_records,
    ):
        """Save PCM16 waveform corruptions and shared clean FLAC targets."""
        output_name = os.path.basename(str(output_name).strip())
        if not output_name:
            raise ValueError("precompute.multi_song_output_name cannot be empty")
        if degrad_num <= 0:
            raise ValueError("degradations_per_clean must be positive")

        prefix = self.cfg.get("precompute", {}).get(
            "waveform_flac_output_prefix", "ia_classical_FINAL_waveform"
        )
        train_dir = os.path.join(
            output_root,
            f"{prefix}_train_blind_valid_hopTenth_{output_name}",
        )
        validate_dir = os.path.join(
            output_root,
            f"{prefix}_validate_blind_valid_hopTenth_{output_name}",
        )
        ground_truth_dir = os.path.join(
            output_root,
            f"{prefix}_ground_truth_blind_valid_hopTenth_{output_name}",
        )
        output_dirs = (train_dir, validate_dir, ground_truth_dir)
        for directory in output_dirs:
            os.makedirs(directory, exist_ok=True)
            if os.listdir(directory):
                raise ValueError(f"Waveform-FLAC output directory is not empty: {directory}")

        train_records, validate_records, discarded_records = split_records
        self._validate_waveform_split_partition(
            window_records,
            train_records,
            validate_records,
            discarded_records,
        )

        train_save = self._save_waveform_flac_split(
            songs,
            train_records,
            train_dir,
            ground_truth_dir,
            degrad_num,
            "train",
        )
        validate_save = self._save_waveform_flac_split(
            songs,
            validate_records,
            validate_dir,
            ground_truth_dir,
            degrad_num,
            "validate",
        )

        source_files = [
            os.path.join(song["dir"], song["stems"][0]) for song in songs
        ]
        usable_windows = len(train_records) + len(validate_records)
        metadata = {
            "format": "waveform_flac_pair_v1",
            "sample_rate": self.sample_rate,
            "segment_samples": self.segment_samples,
            "segment_duration": self.segment_samples / self.sample_rate,
            "hop_samples": self.hop_samples,
            "hop_duration": self.hop_samples / self.sample_rate,
            "degradations_per_clean": degrad_num,
            "split_method": (
                "global_stochastic_multi_song_window_split_"
                "discard_within_song_train_overlap_windows"
            ),
            "split_seed": self.cfg["training"]["seed"],
            "target_usable_val_ratio": self.cfg.get("precompute", {}).get(
                "val_ratio", 0.2
            ),
            "overlap_window_radius": self.overlap_window_radius,
            "train_windows": [
                record["global_window_idx"] for record in train_records
            ],
            "validate_windows": [
                record["global_window_idx"] for record in validate_records
            ],
            "discarded_overlap_windows": [
                record["global_window_idx"] for record in discarded_records
            ],
            "train_window_records": self._serialize_window_records(train_records),
            "validate_window_records": self._serialize_window_records(
                validate_records
            ),
            "discarded_window_records": self._serialize_window_records(
                discarded_records
            ),
            "window_record_fields": [
                "global_window_idx",
                "source_idx",
                "source_window_idx",
            ],
            "split_audit": {
                "total_windows": len(window_records),
                "train_windows": len(train_records),
                "validation_windows": len(validate_records),
                "discarded_overlap_windows": len(discarded_records),
                "usable_validation_ratio": (
                    len(validate_records) / usable_windows
                    if usable_windows
                    else 0.0
                ),
                "partition_all_windows": True,
                "no_train_validation_interval_overlap": True,
            },
            "source_name": output_name,
            "source_path": source_dir,
            "source_files": source_files,
            "source_window_counts": [
                self._num_full_windows(song) for song in songs
            ],
            "source_recording_ids": [song["recording_id"] for song in songs],
            "source_view_kinds": [song["view_kind"] for song in songs],
            "source_families": [song["family"] for song in songs],
            "target_lufs": self.target_lufs,
            "source_loudness_scope": self.source_loudness_scope,
            "waveform_normalization": self._waveform_normalization_description(),
            "corruption": self.cfg["corruption"],
            "latent_encoding": "none; processing stops after waveform corruption",
            "audio_storage": {
                "container": "FLAC",
                "subtype": "PCM_16",
                "channels": 1,
                "sample_rate": self.sample_rate,
                "segment_samples": self.segment_samples,
                "serialization": "clamp to [-1, 1], then PCM16 quantization",
            },
            "degraded_file_pattern": (
                "window_NNNNNN_degradation_DD.flac"
            ),
            "ground_truth_file_pattern": "window_NNNNNN.flac",
            "clean_target_mapping": (
                "remove _degradation_DD from a degraded basename and resolve "
                "that window_NNNNNN.flac in ground_truth_dir"
            ),
            "ground_truth_dir": ground_truth_dir,
            "output_counts": {
                "train_degraded_flacs": train_save["pair_count"],
                "validation_degraded_flacs": validate_save["pair_count"],
                "ground_truth_flacs": usable_windows,
            },
            "pcm_clipping_audit": {
                "train_files_with_prequantization_peak_over_one": train_save[
                    "clipped_files"
                ],
                "train_samples_clamped": train_save["clipped_samples"],
                "validation_files_with_prequantization_peak_over_one": (
                    validate_save["clipped_files"]
                ),
                "validation_samples_clamped": validate_save["clipped_samples"],
            },
        }

        source_metadata = os.path.join(source_dir, "METADATA.txt")
        if not os.path.isfile(source_metadata):
            raise FileNotFoundError(f"Missing source metadata: {source_metadata}")
        for directory in output_dirs:
            self._atomic_torch_save(metadata, os.path.join(directory, "metadata.pt"))
            self._write_waveform_flac_methods_txt(directory, metadata)
            shutil.copy2(source_metadata, os.path.join(directory, "METADATA.txt"))

        self._validate_waveform_flac_outputs(
            train_dir,
            validate_dir,
            ground_truth_dir,
            train_records,
            validate_records,
            discarded_records,
            degrad_num,
        )
        for directory in output_dirs:
            success_tmp = os.path.join(directory, "_SUCCESS.tmp")
            success_path = os.path.join(directory, "_SUCCESS")
            with open(success_tmp, "w", encoding="utf-8") as file:
                file.write("complete\n")
            os.replace(success_tmp, success_path)

        print(
            "Waveform-FLAC precompute saved "
            f"{train_save['pair_count']} train corruptions, "
            f"{validate_save['pair_count']} validation corruptions, and "
            f"{usable_windows} clean targets; "
            f"discarded {len(discarded_records)} overlapping windows.",
            flush=True,
        )
        print(f"Train output: {train_dir}", flush=True)
        print(f"Validation output: {validate_dir}", flush=True)
        print(f"Ground truth output: {ground_truth_dir}", flush=True)

    @torch.no_grad()
    def _save_waveform_flac_split(
        self,
        songs,
        window_records,
        split_dir,
        ground_truth_dir,
        degrad_num,
        split_name,
    ):
        pair_count = 0
        clipped_files = 0
        clipped_samples = 0
        batch_size = int(self.cfg["training"]["batch_size"])
        total_batches = math.ceil(len(window_records) / batch_size)
        for ordinal, start in enumerate(
            range(0, len(window_records), batch_size), start=1
        ):
            chunk = window_records[start:start + batch_size]
            clean = torch.stack(
                [
                    self._load_lufs_window(
                        songs[record["source_idx"]],
                        record["source_window_idx"],
                    )
                    for record in chunk
                ],
                dim=0,
            )
            clean_device = clean.to(self.device, non_blocking=True)
            for item_idx, record in enumerate(chunk):
                self._write_pcm16_flac_atomic(
                    os.path.join(
                        ground_truth_dir,
                        f"window_{record['global_window_idx']:06d}.flac",
                    ),
                    clean[item_idx],
                )

            for degradation_idx in range(degrad_num):
                corrupted = self._corrupt_batch(clean_device)
                if tuple(corrupted.shape) != tuple(clean_device.shape):
                    raise ValueError(
                        f"Corruptor changed shape {tuple(clean_device.shape)} -> "
                        f"{tuple(corrupted.shape)}"
                    )
                for item_idx, record in enumerate(chunk):
                    audio = corrupted[item_idx]
                    over = audio.abs() > 1.0
                    num_clipped = int(over.sum().item())
                    if num_clipped:
                        clipped_files += 1
                        clipped_samples += num_clipped
                    self._write_pcm16_flac_atomic(
                        os.path.join(
                            split_dir,
                            f"window_{record['global_window_idx']:06d}_"
                            f"degradation_{degradation_idx:02d}.flac",
                        ),
                        audio,
                    )
                    pair_count += 1
                del corrupted

            if ordinal == 1 or ordinal % 50 == 0 or ordinal == total_batches:
                print(
                    f"{split_name} waveform FLACs: {ordinal}/{total_batches} batches",
                    flush=True,
                )
        return {
            "pair_count": pair_count,
            "clipped_files": clipped_files,
            "clipped_samples": clipped_samples,
        }

    def _write_pcm16_flac_atomic(self, path, audio):
        audio = audio.detach().float().cpu()
        if tuple(audio.shape) != (1, self.segment_samples):
            raise ValueError(
                f"Expected mono waveform (1, {self.segment_samples}), "
                f"got {tuple(audio.shape)}"
            )
        if not torch.isfinite(audio).all():
            raise ValueError(f"Refusing to save non-finite waveform: {path}")
        path = os.path.abspath(path)
        if os.path.exists(path):
            raise FileExistsError(path)
        tmp_path = path + ".tmp"
        sf.write(
            tmp_path,
            audio.clamp(-1.0, 1.0).squeeze(0).numpy(),
            self.sample_rate,
            format="FLAC",
            subtype="PCM_16",
        )
        os.replace(tmp_path, path)

    @staticmethod
    def _atomic_torch_save(payload, path):
        tmp_path = path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _validate_waveform_split_partition(
        self,
        all_records,
        train_records,
        validate_records,
        discarded_records,
    ):
        def ids(records):
            return {record["global_window_idx"] for record in records}

        all_ids = ids(all_records)
        train_ids = ids(train_records)
        validate_ids = ids(validate_records)
        discarded_ids = ids(discarded_records)
        if train_ids & validate_ids or train_ids & discarded_ids or validate_ids & discarded_ids:
            raise ValueError("Train, validation, and discarded window sets overlap")
        if train_ids | validate_ids | discarded_ids != all_ids:
            raise ValueError("Window split does not partition the complete inventory")

        train_positions = {
            (record["source_idx"], record["source_window_idx"])
            for record in train_records
        }
        for record in validate_records:
            source_idx = record["source_idx"]
            local_idx = record["source_window_idx"]
            for neighbor in range(
                local_idx - self.overlap_window_radius,
                local_idx + self.overlap_window_radius + 1,
            ):
                if (source_idx, neighbor) in train_positions:
                    raise ValueError(
                        "Train/validation audio intervals overlap: "
                        f"source={source_idx}, train_local={neighbor}, "
                        f"validation_local={local_idx}"
                    )

    def _validate_pcm16_flac(self, path):
        info = sf.info(path)
        if (
            info.format != "FLAC"
            or info.subtype != "PCM_16"
            or info.channels != 1
            or info.samplerate != self.sample_rate
            or info.frames != self.segment_samples
        ):
            raise ValueError(f"Invalid waveform-FLAC properties: {path}: {info}")
        with sf.SoundFile(path) as handle:
            while True:
                block = handle.read(65_536, dtype="float32", always_2d=True)
                if not len(block):
                    break
                if not np.isfinite(block).all():
                    raise ValueError(f"Non-finite decoded samples: {path}")

    def _validate_waveform_flac_outputs(
        self,
        train_dir,
        validate_dir,
        ground_truth_dir,
        train_records,
        validate_records,
        discarded_records,
        degrad_num,
    ):
        degraded_pattern = re.compile(
            r"window_(\d{6})_degradation_(\d{2})\.flac"
        )
        clean_pattern = re.compile(r"window_(\d{6})\.flac")
        auxiliary = {"metadata.pt", "METHODS.txt", "METADATA.txt"}
        discarded_ids = {
            record["global_window_idx"] for record in discarded_records
        }

        for directory, records in (
            (train_dir, train_records),
            (validate_dir, validate_records),
        ):
            expected_ids = {record["global_window_idx"] for record in records}
            counts = Counter()
            pair_count = 0
            for entry in os.scandir(directory):
                if entry.name in auxiliary and entry.is_file():
                    continue
                if not entry.is_file():
                    raise ValueError(f"Unexpected output entry: {entry.path}")
                match = degraded_pattern.fullmatch(entry.name)
                if match is None:
                    raise ValueError(f"Unexpected output file: {entry.path}")
                global_idx, degradation_idx = map(int, match.groups())
                if global_idx not in expected_ids or global_idx in discarded_ids:
                    raise ValueError(f"Out-of-split degraded FLAC: {entry.path}")
                if not 0 <= degradation_idx < degrad_num:
                    raise ValueError(f"Unexpected degradation index: {entry.path}")
                counts[global_idx] += 1
                pair_count += 1
                self._validate_pcm16_flac(entry.path)
            if pair_count != len(expected_ids) * degrad_num:
                raise ValueError(
                    f"{directory}: expected {len(expected_ids) * degrad_num} "
                    f"degraded FLACs, found {pair_count}"
                )
            if set(counts) != expected_ids or any(
                count != degrad_num for count in counts.values()
            ):
                raise ValueError(f"Incomplete degradation inventory: {directory}")

        retained_ids = {
            record["global_window_idx"]
            for record in list(train_records) + list(validate_records)
        }
        actual_clean_ids = set()
        for entry in os.scandir(ground_truth_dir):
            if entry.name in auxiliary and entry.is_file():
                continue
            if not entry.is_file():
                raise ValueError(f"Unexpected ground-truth entry: {entry.path}")
            match = clean_pattern.fullmatch(entry.name)
            if match is None:
                raise ValueError(f"Unexpected ground-truth file: {entry.path}")
            global_idx = int(match.group(1))
            if global_idx in discarded_ids:
                raise ValueError(f"Discarded window has ground truth: {entry.path}")
            actual_clean_ids.add(global_idx)
            self._validate_pcm16_flac(entry.path)
        if actual_clean_ids != retained_ids:
            raise ValueError("Ground-truth FLAC inventory does not match retained windows")

        for directory in (train_dir, validate_dir, ground_truth_dir):
            metadata_path = os.path.join(directory, "metadata.pt")
            methods_path = os.path.join(directory, "METHODS.txt")
            source_metadata_path = os.path.join(directory, "METADATA.txt")
            if not all(
                os.path.isfile(path)
                for path in (metadata_path, methods_path, source_metadata_path)
            ):
                raise ValueError(f"Missing waveform-FLAC metadata in {directory}")
            metadata = torch.load(
                metadata_path, map_location="cpu", weights_only=True
            )
            if metadata.get("format") != "waveform_flac_pair_v1":
                raise ValueError(f"Unexpected metadata format in {metadata_path}")

    @staticmethod
    def _write_waveform_flac_methods_txt(directory, metadata):
        lines = [
            "Waveform-FLAC fully-blind public full-mix precompute",
            "",
            "Windowing and split:",
            f"- sample_rate: {metadata['sample_rate']}",
            f"- segment_samples: {metadata['segment_samples']}",
            f"- hop_samples: {metadata['hop_samples']}",
            f"- split_method: {metadata['split_method']}",
            f"- split_seed: {metadata['split_seed']}",
            f"- overlap_window_radius: {metadata['overlap_window_radius']}",
            f"- split_audit: {metadata['split_audit']}",
            "",
            "Waveforms:",
            f"- normalization: {metadata['waveform_normalization']}",
            f"- storage: {metadata['audio_storage']}",
            f"- corruptions_per_clean_window: {metadata['degradations_per_clean']}",
            f"- corruption_config: {metadata['corruption']}",
            "- latent_encoding: none",
            "",
            "Pairing:",
            f"- degraded_file_pattern: {metadata['degraded_file_pattern']}",
            f"- ground_truth_file_pattern: {metadata['ground_truth_file_pattern']}",
            f"- ground_truth_dir: {metadata['ground_truth_dir']}",
            f"- clean_target_mapping: {metadata['clean_target_mapping']}",
        ]
        path = os.path.join(directory, "METHODS.txt")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")
        os.replace(tmp_path, path)

    @torch.no_grad()
    def _save_precompute_pairs(self, output_root, degrad_num, song, all_window_indices):
        """Save train/validation latent pairs as independently loadable records."""
        train_dir = os.path.join(output_root, "ia_classical_FINAL_precompute_train_blind_valid_hopTenth")
        validate_dir = os.path.join(output_root, "ia_classical_FINAL_precompute_validate_blind_valid_hopTenth")
        ground_truth_dir = os.path.join(
            output_root,
            "ia_classical_FINAL_precompute_ground_truth_blind_valid_hopTenth",
        )
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(validate_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)

        train_indices, validate_indices, discarded_indices = self._stochastic_split_windows(
            len(all_window_indices),
            self.cfg["training"]["seed"],
        )
        metadata = {
            "format": "ddpm_latent_pair_v1",
            "sample_rate": self.codec.sample_rate,
            "segment_samples": self.segment_samples,
            "segment_duration": self.segment_samples / self.codec.sample_rate,
            "hop_samples": self.hop_samples,
            "hop_duration": self.hop_samples / self.codec.sample_rate,
            "degradations_per_clean": degrad_num,
            "split_method": "stochastic_window_split_discard_train_overlap_windows",
            "split_seed": self.cfg["training"]["seed"],
            "target_usable_val_ratio": self.cfg.get("precompute", {}).get("val_ratio", 0.2),
            "overlap_window_radius": self.overlap_window_radius,
            "train_windows": train_indices,
            "validate_windows": validate_indices,
            "discarded_overlap_windows": discarded_indices,
            "pair_keys": ["z_cond", "z_clean"],
            "ground_truth_dir": ground_truth_dir,
            "source_name": self.one_song_set.name(),
            "source_path": os.path.join(song["dir"], song["stems"][0]),
            "target_lufs": self.target_lufs,
            "waveform_normalization": self._waveform_normalization_description(),
            "latent_normalization": "SAME latents normalized with LUFS-normalized clean-latent mean/std",
            "latent_mean": self.latent_mean.detach().cpu(),
            "latent_std": self.latent_std.detach().cpu(),
            "corruption": self.cfg["corruption"],
        }
        torch.save(metadata, os.path.join(train_dir, "metadata.pt"))
        torch.save(metadata, os.path.join(validate_dir, "metadata.pt"))
        torch.save(metadata, os.path.join(ground_truth_dir, "metadata.pt"))
        self._write_split_methods_txt(train_dir, metadata, "train")
        self._write_split_methods_txt(validate_dir, metadata, "validate")

        train_count = self._save_window_index_pairs(
            song,
            train_indices,
            train_dir,
            ground_truth_dir,
            degrad_num,
        )
        validate_count = self._save_window_index_pairs(
            song,
            validate_indices,
            validate_dir,
            ground_truth_dir,
            degrad_num,
        )

        print(
            "Precompute saved "
            f"{train_count} train pairs and {validate_count} validation pairs "
            f"from {len(train_indices)} train windows, {len(validate_indices)} "
            f"validation windows, and {len(discarded_indices)} discarded windows."
        )

    @torch.no_grad()
    def _save_precompute_pairs_multi_song(
        self,
        output_root,
        output_name,
        degrad_num,
        source_dir,
        songs,
        window_records,
        split_records=None,
    ):
        """Save globally indexed multi-song train/validation latent pairs."""
        distributed = self.world_size > 1
        output_name = os.path.basename(str(output_name).strip())
        if not output_name:
            raise ValueError("precompute.multi_song_output_name cannot be empty")
        if self.cfg.get("precompute", {}).get("simple_output_layout", False):
            train_dir = os.path.join(output_root, "train")
            validate_dir = os.path.join(output_root, "validate")
            ground_truth_dir = os.path.join(output_root, "ground_truth")
        else:
            prefix = "ia_classical_FINAL_precompute"
            train_dir = os.path.join(
                output_root,
                f"{prefix}_train_blind_valid_hopTenth_{output_name}",
            )
            validate_dir = os.path.join(
                output_root,
                f"{prefix}_validate_blind_valid_hopTenth_{output_name}",
            )
            ground_truth_dir = os.path.join(
                output_root,
                f"{prefix}_ground_truth_blind_valid_hopTenth_{output_name}",
            )
        # Only rank 0 creates shared directories. The barrier ensures they are
        # visible before other ranks begin writing their disjoint pair files.
        if not distributed or self.rank == 0:
            nonempty = [
                directory for directory in (train_dir, validate_dir, ground_truth_dir)
                if os.path.isdir(directory) and os.listdir(directory)
            ]
            if nonempty:
                raise ValueError(
                    "Precompute output directories must be empty before starting: "
                    + ", ".join(nonempty)
                )
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(validate_dir, exist_ok=True)
            os.makedirs(ground_truth_dir, exist_ok=True)
        if distributed:
            torch.distributed.barrier()

        if split_records is None:
            split_records = self._stochastic_split_multi_song_windows(
                window_records,
                self.cfg["training"]["seed"],
            )
        train_records, validate_records, discarded_records = split_records
        source_files = [
            os.path.join(song["dir"], song["stems"][0]) for song in songs
        ]
        source_window_counts = [self._num_full_windows(song) for song in songs]
        grouped_split = self.related_view_split == "aligned_recording_views"
        usable_windows = len(train_records) + len(validate_records)
        split_method = (
            "global_stochastic_per_wav_window_split_promote_aligned_"
            "recording_views_discard_cross_view_train_overlaps"
            if grouped_split
            else (
                "global_stochastic_multi_song_window_split_"
                "discard_within_song_train_overlap_windows"
            )
        )
        metadata = {
            "format": "ddpm_latent_pair_v1",
            "sample_rate": self.codec.sample_rate,
            "segment_samples": self.segment_samples,
            "segment_duration": self.segment_samples / self.codec.sample_rate,
            "hop_samples": self.hop_samples,
            "hop_duration": self.hop_samples / self.codec.sample_rate,
            "degradations_per_clean": degrad_num,
            "split_method": split_method,
            "split_candidate_unit": "per_wav_window",
            "related_view_split": self.related_view_split,
            "paired_validation_views": (
                "all_available_views_at_same_recording_start"
                if grouped_split
                else "none"
            ),
            "guarded_windows_may_later_become_validation": True,
            "split_seed": self.cfg["training"]["seed"],
            "target_usable_val_ratio": self.cfg.get("precompute", {}).get(
                "val_ratio", 0.2
            ),
            "overlap_window_radius": self.overlap_window_radius,
            "train_windows": [
                record["global_window_idx"] for record in train_records
            ],
            "validate_windows": [
                record["global_window_idx"] for record in validate_records
            ],
            "discarded_overlap_windows": [
                record["global_window_idx"] for record in discarded_records
            ],
            "train_window_records": self._serialize_window_records(train_records),
            "validate_window_records": self._serialize_window_records(
                validate_records
            ),
            "discarded_window_records": self._serialize_window_records(
                discarded_records
            ),
            "split_audit": {
                "total_windows": len(window_records),
                "train_windows": len(train_records),
                "validation_windows": len(validate_records),
                "discarded_overlap_windows": len(discarded_records),
                "usable_validation_ratio": (
                    len(validate_records) / usable_windows
                    if usable_windows
                    else 0.0
                ),
                "recording_groups": len(
                    {song["recording_group_idx"] for song in songs}
                ),
                "partition_all_windows": True,
                "aligned_siblings_all_validation": (
                    True if grouped_split else None
                ),
                "no_train_validation_interval_overlap": True,
            },
            "pair_keys": [
                "z_cond",
                "z_clean",
                "source_idx",
                "source_window_idx",
            ],
            "ground_truth_dir": ground_truth_dir,
            "source_name": output_name,
            "source_path": source_dir,
            "source_files": source_files,
            "source_window_counts": source_window_counts,
            "source_recording_ids": [song["recording_id"] for song in songs],
            "source_recording_group_indices": [
                song["recording_group_idx"] for song in songs
            ],
            "source_view_kinds": [song["view_kind"] for song in songs],
            "source_families": [song["family"] for song in songs],
            "source_datasets": [song["dataset"] for song in songs],
            "source_songs": [song["song"] for song in songs],
            "latent_stats_scope": self.latent_stats_scope,
            "target_lufs": self.target_lufs,
            "waveform_normalization": self._waveform_normalization_description(),
            "latent_normalization": (
                "SAME latents normalized with clean-latent mean/std computed "
                f"from {self.latent_stats_scope.replace('_', ' ')}"
            ),
            "latent_mean": self.latent_mean.detach().cpu(),
            "latent_std": self.latent_std.detach().cpu(),
            "corruption": self.cfg["corruption"],
        }
        # Shared metadata must have a single writer. All ranks already hold
        # identical reduced latent stats and deterministically computed splits.
        if not distributed or self.rank == 0:
            for directory in (train_dir, validate_dir, ground_truth_dir):
                torch.save(metadata, os.path.join(directory, "metadata.pt"))
            self._write_multi_song_methods_txt(train_dir, metadata, "train")
            self._write_multi_song_methods_txt(validate_dir, metadata, "validate")
            self._write_multi_song_methods_txt(
                ground_truth_dir, metadata, "ground_truth"
            )
        if distributed:
            torch.distributed.barrier()

        train_records_to_save = train_records
        validate_records_to_save = validate_records
        if distributed:
            train_records_to_save = self._contiguous_rank_shard(
                train_records, self.rank, self.world_size
            )
            validate_records_to_save = self._contiguous_rank_shard(
                validate_records, self.rank, self.world_size
            )
            print(
                f"[rank {self.rank}] pair shards: "
                f"train={len(train_records_to_save)}, "
                f"validate={len(validate_records_to_save)}",
                flush=True,
            )

        train_count = self._save_multi_song_window_pairs(
            songs,
            train_records_to_save,
            train_dir,
            ground_truth_dir,
            degrad_num,
            progress_label="train" if distributed else None,
        )
        validate_count = self._save_multi_song_window_pairs(
            songs,
            validate_records_to_save,
            validate_dir,
            ground_truth_dir,
            degrad_num,
            progress_label="validate" if distributed else None,
        )
        if distributed:
            pair_counts = torch.tensor(
                [train_count, validate_count],
                device=self.device,
                dtype=torch.int64,
            )
            torch.distributed.all_reduce(
                pair_counts, op=torch.distributed.ReduceOp.SUM
            )
            train_count, validate_count = pair_counts.tolist()
            torch.distributed.barrier()

        if not distributed or self.rank == 0:
            # This marker is written last. Its presence means all four ranks
            # finished both splits; training should not start before it exists.
            with open(os.path.join(train_dir, "_SUCCESS"), "w") as file:
                file.write("complete\n")
            print(
                "Multi-song precompute saved "
                f"{train_count} train pairs and {validate_count} validation pairs "
                f"from {len(train_records)} train windows, "
                f"{len(validate_records)} validation windows, and "
                f"{len(discarded_records)} discarded overlap windows.",
                flush=True,
            )
            print(f"Train output: {train_dir}", flush=True)
            print(f"Validation output: {validate_dir}", flush=True)
            print(f"Ground truth output: {ground_truth_dir}", flush=True)
        if distributed:
            torch.distributed.barrier()

    def _stochastic_split_multi_song_windows(self, window_records, seed):
        """Globally select validation windows; discard overlap within each song."""
        if not window_records:
            raise ValueError("Cannot split zero multi-song windows")
        target_ratio = float(
            self.cfg.get("precompute", {}).get("val_ratio", 0.2)
        )
        if not 0.0 < target_ratio < 1.0:
            raise ValueError(
                f"precompute.val_ratio must be between 0 and 1, got {target_ratio}"
            )
        if self.related_view_split == "aligned_recording_views":
            return split_grouped_aligned_windows(
                window_records,
                seed=seed,
                target_ratio=target_ratio,
                overlap_window_radius=self.overlap_window_radius,
            )

        by_global = {
            record["global_window_idx"]: record for record in window_records
        }
        by_source_window = {
            (record["source_idx"], record["source_window_idx"]): record[
                "global_window_idx"
            ]
            for record in window_records
        }
        all_global = set(by_global)
        candidates = sorted(all_global)
        random.Random(seed).shuffle(candidates)

        validate = set()
        for global_idx in candidates:
            validate.add(global_idx)
            discarded = self._discarded_multi_song_overlaps(
                validate, by_global, by_source_window
            )
            train = all_global - validate - discarded
            usable = len(train) + len(validate)
            if usable and len(validate) / usable >= target_ratio:
                break

        discarded = self._discarded_multi_song_overlaps(
            validate, by_global, by_source_window
        )
        train = all_global - validate - discarded
        to_records = lambda indices: [by_global[idx] for idx in sorted(indices)]
        return to_records(train), to_records(validate), to_records(discarded)

    def _discarded_multi_song_overlaps(
        self,
        validate_global_indices,
        by_global,
        by_source_window,
    ):
        """Return non-validation windows overlapping validation in the same song."""
        discarded = set()
        validate_global_indices = set(validate_global_indices)
        for global_idx in validate_global_indices:
            record = by_global[global_idx]
            source_idx = record["source_idx"]
            local_idx = record["source_window_idx"]
            for neighbor_idx in range(
                local_idx - self.overlap_window_radius,
                local_idx + self.overlap_window_radius + 1,
            ):
                neighbor_global = by_source_window.get((source_idx, neighbor_idx))
                if (
                    neighbor_global is not None
                    and neighbor_global not in validate_global_indices
                ):
                    discarded.add(neighbor_global)
        return discarded

    @staticmethod
    def _serialize_window_records(records):
        """Store compact [global, source, local] provenance rows in metadata."""
        return [
            [
                record["global_window_idx"],
                record["source_idx"],
                record["source_window_idx"],
            ]
            for record in records
        ]

    def _stochastic_split_windows(self, num_windows, seed):
        """Pick validation windows, then discard train windows that overlap them."""
        if num_windows <= 0:
            raise ValueError("Cannot split zero windows")
        target_ratio = self.cfg.get("precompute", {}).get("val_ratio", 0.2)
        all_indices = set(range(num_windows))
        rng = random.Random(seed)
        candidates = list(range(num_windows))
        rng.shuffle(candidates)

        validate = set()
        for idx in candidates:
            validate.add(idx)
            discarded = self._discarded_overlaps(validate, num_windows)
            train = all_indices - validate - discarded
            usable = len(train) + len(validate)
            if usable > 0 and len(validate) / usable >= target_ratio:
                break

        discarded = self._discarded_overlaps(validate, num_windows)
        train = all_indices - validate - discarded
        return sorted(train), sorted(validate), sorted(discarded)

    def _discarded_overlaps(self, validate_indices, num_windows):
        """Discard windows that overlap validation, except validation windows."""
        discarded = set()
        validate_indices = set(validate_indices)
        for idx in validate_indices:
            lo = max(0, idx - self.overlap_window_radius)
            hi = min(num_windows - 1, idx + self.overlap_window_radius)
            for neighbor in range(lo, hi + 1):
                if neighbor not in validate_indices:
                    discarded.add(neighbor)
        return discarded

    def _num_full_windows(self, song):
        return 1 + (song["orig_frames"] - self.segment_samples) // self.hop_samples

    def _load_lufs_window(self, song, window_idx):
        orig_sr = song["orig_sr"]
        if orig_sr != self.sample_rate:
            raise ValueError(
                f"Precompute requires source sample rate {self.sample_rate}; "
                f"got {orig_sr} for {song['dir']}"
            )
        path = os.path.join(song["dir"], song["stems"][0])
        start = window_idx * self.hop_samples
        audio, sr = torchaudio.load(
            path,
            frame_offset=start,
            num_frames=self.segment_samples,
        )
        if sr != self.sample_rate:
            raise ValueError(f"Unexpected sample rate {sr} while loading {path}")
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if audio.shape[-1] != self.segment_samples:
            raise ValueError(f"Window {window_idx} is shorter than segment duration")
        if self.source_loudness_scope == "whole_song_pre_normalized":
            return audio
        return self._normalize_lufs(audio)

    def _waveform_normalization_description(self):
        if self.source_loudness_scope == "whole_song_pre_normalized":
            return (
                "each complete source recording was normalized once toward "
                f"{self.target_lufs} LUFS before windowing, with peak safety; "
                "five-second windows preserve the resulting song dynamics and "
                "receive no independent window-level loudness normalization"
            )
        return (
            "integrated loudness normalization with torchaudio.transforms.Loudness "
            f"to {self.target_lufs} LUFS per window, then peak safety if abs peak > 1.0"
        )

    def _normalize_lufs(self, audio):
        loudness = torchaudio.transforms.Loudness(self.sample_rate)(audio)
        if torch.isfinite(loudness):
            gain_db = self.target_lufs - float(loudness.item())
            audio = audio * (10.0 ** (gain_db / 20.0))
        peak = audio.abs().max()
        if peak > 1.0:
            audio = audio / peak
        return audio

    @torch.no_grad()
    def _compute_lufs_window_latent_stats(self, song, window_indices):
        total = None
        total_sq = None
        count = 0
        batch_size = self.cfg["training"]["batch_size"]

        for start in range(0, len(window_indices), batch_size):
            chunk = window_indices[start:start + batch_size]
            clean = torch.stack([
                self._load_lufs_window(song, idx) for idx in chunk
            ], dim=0).to(self.device, non_blocking=True)
            z_clean = ensure_channel_first(
                self.codec.encode(clean),
                channels=self.codec.latent_dim,
            ).float()
            batch_sum = z_clean.sum(dim=(0, 2))
            batch_sum_sq = (z_clean * z_clean).sum(dim=(0, 2))
            batch_count = z_clean.shape[0] * z_clean.shape[2]
            total = batch_sum if total is None else total + batch_sum
            total_sq = batch_sum_sq if total_sq is None else total_sq + batch_sum_sq
            count += batch_count

        if count == 0:
            raise ValueError("Cannot compute latent stats from empty window list")
        mean = total / count
        variance = (total_sq / count) - mean.square()
        std = torch.sqrt(torch.clamp(variance, min=1e-12))
        self.latent_mean = mean.reshape(1, self.codec.latent_dim, 1).to(self.device)
        self.latent_std = std.reshape(1, self.codec.latent_dim, 1).to(self.device)

    @torch.no_grad()
    def _compute_lufs_multi_song_latent_stats(self, songs, window_records):
        """Compute fixed per-channel SAME statistics over every source song."""
        total = None
        total_sq = None
        count = 0
        batch_size = self.cfg["training"]["batch_size"]

        for start in range(0, len(window_records), batch_size):
            chunk = window_records[start:start + batch_size]
            clean = torch.stack(
                [
                    self._load_lufs_window(
                        songs[record["source_idx"]],
                        record["source_window_idx"],
                    )
                    for record in chunk
                ],
                dim=0,
            ).to(self.device, non_blocking=True)
            z_clean = ensure_channel_first(
                self.codec.encode(clean),
                channels=self.codec.latent_dim,
            ).float()
            batch_sum = z_clean.sum(dim=(0, 2))
            batch_sum_sq = (z_clean * z_clean).sum(dim=(0, 2))
            batch_count = z_clean.shape[0] * z_clean.shape[2]
            total = batch_sum if total is None else total + batch_sum
            total_sq = batch_sum_sq if total_sq is None else total_sq + batch_sum_sq
            count += batch_count

        if count == 0:
            raise ValueError("Cannot compute latent stats from empty multi-song windows")
        mean = total / count
        variance = (total_sq / count) - mean.square()
        std = torch.sqrt(torch.clamp(variance, min=1e-12))
        self.latent_mean = mean.reshape(1, self.codec.latent_dim, 1).to(self.device)
        self.latent_std = std.reshape(1, self.codec.latent_dim, 1).to(self.device)

    @staticmethod
    def _contiguous_rank_shard(records, rank, world_size):
        """Return a balanced contiguous shard to reduce shared-disk seeking."""
        start = len(records) * rank // world_size
        end = len(records) * (rank + 1) // world_size
        return records[start:end]

    @torch.no_grad()
    def _compute_lufs_multi_song_latent_stats_distributed(
        self, songs, window_records
    ):
        """Compute exact global latent statistics from disjoint GPU shards."""
        local_records = self._contiguous_rank_shard(
            window_records, self.rank, self.world_size
        )
        channels = self.codec.latent_dim
        # Float64 accumulation reduces cancellation when thousands of window
        # contributions are summed in different orders across ranks.
        local_sum = torch.zeros(channels, device=self.device, dtype=torch.float64)
        local_sum_sq = torch.zeros_like(local_sum)
        local_count = torch.zeros(1, device=self.device, dtype=torch.float64)
        batch_size = self.cfg["training"]["batch_size"]
        total_batches = math.ceil(len(local_records) / batch_size)

        print(
            f"[rank {self.rank}] latent stats shard: "
            f"{len(local_records)} windows in {total_batches} batches",
            flush=True,
        )
        for batch_idx, start in enumerate(
            range(0, len(local_records), batch_size), start=1
        ):
            chunk = local_records[start:start + batch_size]
            clean = torch.stack(
                [
                    self._load_lufs_window(
                        songs[record["source_idx"]],
                        record["source_window_idx"],
                    )
                    for record in chunk
                ],
                dim=0,
            ).to(self.device, non_blocking=True)
            z_clean = ensure_channel_first(
                self.codec.encode(clean),
                channels=channels,
            ).to(torch.float64)
            local_sum += z_clean.sum(dim=(0, 2))
            local_sum_sq += z_clean.square().sum(dim=(0, 2))
            local_count += z_clean.shape[0] * z_clean.shape[2]
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == total_batches:
                print(
                    f"[rank {self.rank}] latent stats "
                    f"{batch_idx}/{total_batches} batches",
                    flush=True,
                )

        # NCCL performs these reductions directly between GPUs. Only 513
        # float64 values are communicated, so synchronization cost is tiny
        # compared with encoding thousands of five-second waveforms.
        torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            local_sum_sq, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            local_count, op=torch.distributed.ReduceOp.SUM
        )
        if local_count.item() <= 0:
            raise ValueError("Cannot compute latent stats from empty windows")

        mean = local_sum / local_count
        variance = (local_sum_sq / local_count) - mean.square()
        std = torch.sqrt(torch.clamp(variance, min=1e-12))
        # Dataset metadata and training use float32 statistics, matching the
        # original single-GPU output representation.
        self.latent_mean = mean.float().reshape(1, channels, 1)
        self.latent_std = std.float().reshape(1, channels, 1)
        if self.rank == 0:
            print(
                "Global latent mean/std reduced across all GPU shards.",
                flush=True,
            )

    def _save_multi_song_window_pairs(
        self,
        songs,
        window_records,
        split_dir,
        ground_truth_dir,
        degrad_num,
        progress_label=None,
    ):
        """Encode and save globally indexed windows from multiple source songs."""
        pair_count = 0
        batch_size = self.cfg["training"]["batch_size"]
        total_batches = math.ceil(len(window_records) / batch_size)
        for batch_idx, start in enumerate(
            range(0, len(window_records), batch_size), start=1
        ):
            chunk = window_records[start:start + batch_size]
            clean = torch.stack(
                [
                    self._load_lufs_window(
                        songs[record["source_idx"]],
                        record["source_window_idx"],
                    )
                    for record in chunk
                ],
                dim=0,
            )
            z_conds, z_clean = self._encode_ddpm_pair_precompute(
                clean,
                degrads=degrad_num,
            )
            for batch_idx, record in enumerate(chunk):
                global_idx = record["global_window_idx"]
                z_clean_item = z_clean[batch_idx].detach().cpu()
                torchaudio.save(
                    os.path.join(
                        ground_truth_dir,
                        f"window_{global_idx:06d}.wav",
                    ),
                    clean[batch_idx].detach().cpu().clamp(-1.0, 1.0),
                    self.codec.sample_rate,
                )
                for degradation_idx, z_cond in enumerate(z_conds):
                    file_name = (
                        f"window_{global_idx:06d}_"
                        f"degradation_{degradation_idx:02d}.pt"
                    )
                    pair = {
                        "z_cond": z_cond[batch_idx].detach().cpu(),
                        "z_clean": z_clean_item.clone(),
                        "source_idx": record["source_idx"],
                        "source_window_idx": record["source_window_idx"],
                    }
                    torch.save(pair, os.path.join(split_dir, file_name))
                    pair_count += 1
            if (
                progress_label is not None
                and (
                    batch_idx == 1
                    or batch_idx % 50 == 0
                    or batch_idx == total_batches
                )
            ):
                print(
                    f"[rank {self.rank}] {progress_label} pairs "
                    f"{batch_idx}/{total_batches} batches",
                    flush=True,
                )
        return pair_count

    def _save_window_index_pairs(self, song, window_indices, split_dir, ground_truth_dir, degrad_num):
        pair_count = 0
        batch_size = self.cfg["training"]["batch_size"]
        for start in range(0, len(window_indices), batch_size):
            chunk = window_indices[start:start + batch_size]
            clean = torch.stack([
                self._load_lufs_window(song, idx) for idx in chunk
            ], dim=0)
            z_conds, z_clean = self._encode_ddpm_pair_precompute(
                clean,
                degrads=degrad_num,
            )
            for batch_idx, window_idx in enumerate(chunk):
                z_clean_item = z_clean[batch_idx].detach().cpu()
                torchaudio.save(
                    os.path.join(ground_truth_dir, f"window_{window_idx:06d}.wav"),
                    clean[batch_idx].detach().cpu().clamp(-1.0, 1.0),
                    self.codec.sample_rate,
                )
                for degradation_idx, z_cond in enumerate(z_conds):
                    file_name = (
                        f"window_{window_idx:06d}_"
                        f"degradation_{degradation_idx:02d}.pt"
                    )
                    pair = {
                        "z_cond": z_cond[batch_idx].detach().cpu(),
                        "z_clean": z_clean_item.clone(),
                    }
                    torch.save(pair, os.path.join(split_dir, file_name))
                    pair_count += 1
        return pair_count

    def _write_split_methods_txt(self, split_dir, metadata, split_name):
        methods_path = os.path.join(split_dir, "METHODS.txt")
        lines = [
            f"DDPM SAME latent precompute {split_name} set",
            "",
            "Source:",
            f"- name: {metadata['source_name']}",
            f"- path: {metadata['source_path']}",
            "",
            "Windowing:",
            f"- sample_rate: {metadata['sample_rate']}",
            f"- segment_samples: {metadata['segment_samples']}",
            f"- segment_duration_sec: {metadata['segment_duration']}",
            f"- hop_samples: {metadata['hop_samples']}",
            f"- hop_duration_sec: {metadata['hop_duration']}",
            "- partial final windows are dropped",
            f"- overlap_window_radius: {metadata['overlap_window_radius']}",
            "",
            "Split:",
            f"- method: {metadata['split_method']}",
            f"- seed: {metadata['split_seed']}",
            f"- target_usable_val_ratio: {metadata['target_usable_val_ratio']}",
            f"- train_window_count: {len(metadata['train_windows'])}",
            f"- validate_window_count: {len(metadata['validate_windows'])}",
            f"- discarded_overlap_window_count: {len(metadata['discarded_overlap_windows'])}",
            "- validation windows may be adjacent",
            "- validation windows are never moved into the discarded set",
            "- train windows within overlap_window_radius of validation windows are discarded",
            "",
            "Normalization:",
            f"- waveform: {metadata['waveform_normalization']}",
            f"- latent: {metadata['latent_normalization']}",
            "",
            "Degradation:",
            f"- corruptions_per_clean_window: {metadata['degradations_per_clean']}",
            f"- corruption_config: {metadata['corruption']}",
            "",
            "Output:",
            f"- pair_keys: {metadata['pair_keys']}",
            f"- ground_truth_dir: {metadata['ground_truth_dir']}",
        ]
        with open(methods_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _write_multi_song_methods_txt(self, split_dir, metadata, split_name):
        """Write split rules and per-song provenance for a multi-song dataset."""
        methods_path = os.path.join(split_dir, "METHODS.txt")
        lines = [
            f"DDPM SAME latent multi-song precompute {split_name} set",
            "",
            "Sources:",
            *[
                f"- source_{idx:02d}: {path} ({window_count} windows)"
                for idx, (path, window_count) in enumerate(
                    zip(
                        metadata["source_files"],
                        metadata["source_window_counts"],
                    )
                )
            ],
            "",
            "Windowing:",
            f"- sample_rate: {metadata['sample_rate']}",
            f"- segment_samples: {metadata['segment_samples']}",
            f"- segment_duration_sec: {metadata['segment_duration']}",
            f"- hop_samples: {metadata['hop_samples']}",
            f"- hop_duration_sec: {metadata['hop_duration']}",
            "- partial final windows are dropped independently per song",
            f"- overlap_window_radius: {metadata['overlap_window_radius']}",
            "",
            "Split:",
            f"- method: {metadata['split_method']}",
            f"- seed: {metadata['split_seed']}",
            f"- target_usable_val_ratio: {metadata['target_usable_val_ratio']}",
            f"- train_window_count: {len(metadata['train_windows'])}",
            f"- validate_window_count: {len(metadata['validate_windows'])}",
            (
                "- discarded_overlap_window_count: "
                f"{len(metadata['discarded_overlap_windows'])}"
            ),
            "- validation candidates are shuffled globally across all songs",
            f"- candidate_unit: {metadata['split_candidate_unit']}",
            f"- related_view_split: {metadata['related_view_split']}",
            f"- paired_validation_views: {metadata['paired_validation_views']}",
            (
                "- guarded_windows_may_later_become_validation: "
                f"{metadata['guarded_windows_may_later_become_validation']}"
            ),
            "- validation windows are never moved into the discarded set",
            (
                "- train windows overlapping validation are discarded across "
                "all aligned views in the same recording group"
                if metadata["related_view_split"] == "aligned_recording_views"
                else "- overlap exclusion is applied only within each source song"
            ),
            "",
            "Normalization:",
            f"- waveform: {metadata['waveform_normalization']}",
            f"- latent: {metadata['latent_normalization']}",
            f"- latent_stats_scope: {metadata['latent_stats_scope']}",
            "",
            "Degradation:",
            f"- corruptions_per_clean_window: {metadata['degradations_per_clean']}",
            f"- corruption_config: {metadata['corruption']}",
            "",
            "Output:",
            f"- pair_keys: {metadata['pair_keys']}",
            f"- ground_truth_dir: {metadata['ground_truth_dir']}",
        ]
        with open(methods_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    @torch.no_grad()
    def _encode_ddpm_pair_precompute(self, clean, degrads, profile=None):
        """Corrupt audio, encode clean/corrupt SAME latents, align and normalize them."""
        # Build the paired SAME latents used by DDPM:
        # z_clean is the normalized clean target latent.
        # z_cond is the normalized corrupted conditioning latent.
        clean = clean.to(self.device, non_blocking=True)
        self._assert_on_device("clean", clean)

        B = clean.shape[0]
        corrupts = []
        #multiple times
        for num in range(degrads):
            corrupt = self._corrupt_batch(clean)
            self._assert_on_device(f"corrupt_{num}", corrupt)
            corrupts.append(corrupt)

        encode_input = torch.cat([clean] + corrupts, dim=0)
        self._assert_on_device("encode_input", encode_input)

        z_both_raw = ensure_channel_first(
            self.codec.encode(encode_input),
            channels=self.codec.latent_dim,
        )
        self._assert_on_device("z_both_raw", z_both_raw)

        z_clean_raw = z_both_raw[:B]
        z_cond_raws = []

        for i in range(degrads):
            start = B * (i + 1)
            end = B * (i + 2)
            z_cond_raws.append(z_both_raw[start:end])

        min_t = min(z_clean_raw.shape[-1], *(z.shape[-1] for z in z_cond_raws))
        z_clean_raw = z_clean_raw[..., :min_t]
        z_clean = normalize_latent(z_clean_raw, self.latent_mean, self.latent_std)

        z_conds = []
        for z_cond_raw in z_cond_raws:
            z_cond_raw = z_cond_raw[..., :min_t]
            z_cond = normalize_latent(z_cond_raw, self.latent_mean, self.latent_std)
            z_conds.append(z_cond)

        return z_conds, z_clean
    
        
    def _corrupt_batch(self, audio):
        """Apply configured waveform degradations to each item in a batch."""
        # AudioCorruptor.corrupt_batch is the optimized path: batched segment
        # FFT filters on audio.device, with per-item random filter parameters.
        # Older corruptors can still fall back to the per-sample __call__ loop.
        sr = self.sample_rate
        if hasattr(self.corruptor, "corrupt_batch"):
            return self.corruptor.corrupt_batch(audio, sr)
        return torch.stack([self.corruptor(a, sr) for a in audio])
    

    
