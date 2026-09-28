import json
import itertools
import math
import os
import random
import re
import signal
import time

import torch
import torch.distributed as dist
import torchaudio
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

try:
    from audiobox_aesthetics.infer import initialize_predictor as _initialize_aes_predictor
    _AES_AVAILABLE = True
except ImportError:
    _AES_AVAILABLE = False

from .corruption import AudioCorruptor
from .experiment import Experiment
from .diffusion_utils import (
    align_latent_pair,
    compute_dataset_latent_stats,
    ensure_channel_first,
    extract,
    make_cfm_euler_time_grid,
    move_ddpm_buffers,
    normalize_latent,
    precompute_ddpm_buffers,
    q_sample,
    unnormalize_latent,
)
from .losses import (
    latent_mse_loss,
    multiscale_spectral_loss,
    multiscale_spectral_loss_DDSP,
)
from .models import DIT_MSE_DENOISER_TYPES, build_codec, build_denoiser
from .spectral_losses import (
    auralossMRSTFT,
    audio_to_stft_channels,
    complex_stft_mse_loss,
    compute_spectral_mse_values,
    compute_stft_channel_stats,
    default_spectral_config,
    multi_resolution_stft_mse_loss,
    normalize_stft,
    prepare_audio_pair,
    stft_channels_to_audio,
    unnormalize_stft,
    waveform_mse_loss,
)


def _worker_init(worker_id):
    """Seed Python random inside each DataLoader worker."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)


def _ema_avg_fn(decay):
    """Build the parameter averaging function used by AveragedModel EMA."""
    def avg_fn(averaged_param, current_param, num_averaged):
        return decay * averaged_param + (1.0 - decay) * current_param

    return avg_fn


def _env_flag(name, default):
    """Parse an optional environment variable as a boolean feature flag."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class PrecomputedLatentPairDataset(Dataset):
    """Load precomputed normalized SAME latent pairs from .pt files."""

    _NAME_RE = re.compile(r"window_(\d+)_degradation_(\d+)\.pt$")

    def __init__(self, root):
        self.root = root
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Precomputed latent folder not found: {root}")
        metadata_path = os.path.join(root, "metadata.pt")
        self.metadata = (
            torch.load(metadata_path, map_location="cpu")
            if os.path.exists(metadata_path)
            else {}
        )
        self.files = sorted(
            os.path.join(root, f) for f in os.listdir(root)
            if f.endswith(".pt") and f != "metadata.pt"
        )
        if not self.files:
            raise ValueError(f"No precomputed latent pair files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        pair = torch.load(path, map_location="cpu")
        match = self._NAME_RE.match(os.path.basename(path))
        if match is None:
            raise ValueError(f"Unexpected precomputed pair filename: {path}")
        return {
            "z_cond": pair["z_cond"].float(),
            "z_clean": pair["z_clean"].float(),
            "window_idx": torch.tensor(int(match.group(1)), dtype=torch.long),
            "degradation_idx": torch.tensor(int(match.group(2)), dtype=torch.long),
        }


class DisjointDistributedEvalSampler(Sampler):
    """Deterministically shard evaluation data without padding or duplication.

    PyTorch's ``DistributedSampler(drop_last=False)`` pads non-divisible
    datasets, which repeats validation examples and changes the aggregate
    metric.  This sampler assigns every dataset index to exactly one rank.
    Shards are equal when the dataset length is divisible by the world size
    and otherwise differ by at most one item.
    """

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}"
            )

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


class PrecomputedSpectrogramPairDataset(Dataset):
    """Load precomputed normalized complex-STFT pairs from .pt files."""

    _NAME_RE = re.compile(r"window_(\d+)_degradation_(\d+)\.pt$")

    def __init__(self, root):
        self.root = root
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Precomputed STFT folder not found: {root}")
        metadata_path = os.path.join(root, "metadata.pt")
        self.metadata = (
            torch.load(metadata_path, map_location="cpu")
            if os.path.exists(metadata_path)
            else {}
        )
        self.files = sorted(
            os.path.join(root, f) for f in os.listdir(root)
            if f.endswith(".pt") and f != "metadata.pt"
        )
        if not self.files:
            raise ValueError(f"No precomputed STFT pair files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        pair = torch.load(path, map_location="cpu")
        match = self._NAME_RE.match(os.path.basename(path))
        if match is None:
            raise ValueError(f"Unexpected precomputed STFT pair filename: {path}")
        return {
            "x_stft": pair["x_stft"].float(),
            "y_stft": pair["y_stft"].float(),
            "num_samples": torch.tensor(int(pair["num_samples"]), dtype=torch.long),
            "window_idx": torch.tensor(int(match.group(1)), dtype=torch.long),
            "degradation_idx": torch.tensor(int(match.group(2)), dtype=torch.long),
        }


class PrecomputedWaveformPairDataset(Dataset):
    """Load precomputed corrupted/clean waveform pairs from .pt files."""

    _NAME_RE = re.compile(r"window_(\d+)_degradation_(\d+)\.pt$")

    def __init__(self, root):
        self.root = root
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Precomputed waveform folder not found: {root}")
        metadata_path = os.path.join(root, "metadata.pt")
        self.metadata = (
            torch.load(metadata_path, map_location="cpu")
            if os.path.exists(metadata_path)
            else {}
        )
        self.files = sorted(
            os.path.join(root, f) for f in os.listdir(root)
            if f.endswith(".pt") and f != "metadata.pt"
        )
        if not self.files:
            raise ValueError(f"No precomputed waveform pair files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        pair = torch.load(path, map_location="cpu")
        match = self._NAME_RE.match(os.path.basename(path))
        if match is None:
            raise ValueError(f"Unexpected precomputed waveform pair filename: {path}")
        return {
            "x_audio": pair["x_audio"].float(),
            "y_audio": pair["y_audio"].float(),
            "window_idx": torch.tensor(int(match.group(1)), dtype=torch.long),
            "degradation_idx": torch.tensor(int(match.group(2)), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Per-dataset canonical validation audio window indices.
#
# Maps the *basename* of ``precompute.train_dir`` to the fixed list of 10
# validation window indices that should be logged to TensorBoard for that
# dataset.  When a new precomputed dataset is introduced, add an entry here
# so all experiments using that dataset automatically pick up the same
# listening examples without having to repeat the list in every config.yaml
# or slurm script.
#
# These are the *validation*-split window indices (the sparse source-window
# IDs baked into the filenames, not contiguous 0-N positions).  They must
# exist in the corresponding ``precompute.validate_dir``.
#
# Precedence: if ``logging.precomputed_audio_window_indices`` is already
# explicit in a config.yaml, it wins and this registry is not consulted.
#
# Note for online (non-precomputed) training: this registry has no effect
# because ``precompute.enabled`` will be False.
# ---------------------------------------------------------------------------
_DATASET_VALIDATION_AUDIO_WINDOWS: dict[str, list[int]] = {
    # Beethoven / Strauss II IA hifi subset — 16,000+ sparse validation
    # windows, selected with seed=42 to span the full range.
    "ia_classical_FINAL_precompute_train_blind_valid_hopTenth_beethoven_straussII": [
        7, 1332, 2998, 4452, 6245, 7832, 9538, 11254, 13337, 15563
    ],
    # Public classical fullmixes only (without sections) — hopTenth blind-val.
    "ia_classical_FINAL_precompute_train_blind_valid_hopTenth_public_classical_fullmixes": [
        964, 4214, 4687, 5949, 9278, 10482, 11877, 28578, 32627, 32842
    ],
}


class Trainer:
    """Owns the full training lifecycle for deterministic, DDPM, and CFM denoisers.

    The code intentionally keeps one Trainer class and branches at the narrow
    points where the objectives differ:

    - mlp denoisers learn codec z_corrupt -> z_clean directly.
    - tcn denoisers learn corrupt waveform -> clean waveform directly.
    - conformer/unet denoisers learn normalized complex STFT restoration.
    - ddpm denoisers learn epsilon prediction in normalized SAME latent space.
    - cfm denoisers learn velocity prediction in normalized SAME latent space.

    For online DDPM/CFM, the expensive shared path is: waveform segment ->
    waveform corruption -> SAME encode clean/corrupt. DDPM then predicts
    diffusion epsilon; CFM predicts straight-path velocity.
    """

    def __init__(self, experiment: Experiment):
        """Initialize experiment state, models, loaders, resume state, and DDPM stats."""
        self.exp = experiment
        self.cfg = experiment.cfg
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_primary = self.rank == 0
        if torch.cuda.is_available():
            if self.distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.device.type == "cuda":
            # Use A100 TensorFloat-32 matrix kernels for the float32 DiT and
            # frozen SAME decoder while retaining float32 model/storage types.
            torch.set_float32_matmul_precision("high")
        self.sample_rate = int(self.cfg.get("dataset", {}).get("sample_rate", 44100))
        self.spectral_cfg = dict(default_spectral_config())
        self.spectral_cfg.update(self.cfg.get("spectral", {}))
        self.global_step = 0

        # Generative-latent state. DDPM and CFM use EMA for validation/audio;
        # DDPM additionally uses schedule buffers. Deterministic denoisers
        # leave these as None and use the direct latent regression path.
        self.codec = None
        self.ema_denoiser = None
        self.ddpm_buffers = None
        self.latent_mean = None
        self.latent_std = None
        self.stft_mean = None
        self.stft_std = None
        self._dit_ground_truth_audio = None
        self._dit_ground_truth_index_lookup = None
        self._logged_precomputed_ground_truth = set()
        self._precomputed_val_audio_examples = None
        self._precomputed_val_fixed_audio_logged = False
        self._latest_val_audio_batch = None
        self._best_total_val_audio_loss = None
        self._best_total_val_audio_step = None
        self._best_total_val_audio_epoch = None
        self.train_sampler = None
        self.val_sampler = None
        self._pending_stop = False
        self._benchmark_step_seconds = []

        # Lightweight profiling. Enable with training.profile_steps=true or
        # RESTOR_PROFILE_STEPS=1. CUDA synchronization makes timings accurate,
        # but it also slows training, so turn it off after finding bottlenecks.
        tcfg = self.cfg.get("training", {})
        default_profile_steps = self.cfg.get("denoiser", {}).get("type") in {
            "ddpm", "ddpm_8m", "cfm", "cfm_4m", "cfm_8m", "cfm_16m", "cfm_40m", "cfm_80m"
        }
        self.profile_steps = self.is_primary and _env_flag(
            "RESTOR_PROFILE_STEPS",
            bool(tcfg.get("profile_steps", default_profile_steps)),
        )
        self.profile_every = max(
            1, int(os.environ.get("RESTOR_PROFILE_EVERY", tcfg.get("profile_every", 1)))
        )
        self.profile_cuda_sync = _env_flag(
            "RESTOR_PROFILE_CUDA_SYNC", bool(tcfg.get("profile_cuda_sync", True))
                )
        # Cheap end-to-end throughput timing is independent of the detailed
        # component profiler above. _step() ends in loss.item(), which waits for
        # the queued GPU work, so this does not add another CUDA synchronization.
        self.log_step_timing = _env_flag("RESTOR_LOG_STEP_TIMING", False)

        self._inject_canonical_validation_windows()
        self._validate_denoiser_mode()
        self._build_models()
        self._build_data()
        self._load_dit_ground_truth_audio_bank()
        self._try_resume()
        self._ensure_ddpm_latent_stats()
        self._ensure_stft_stats()

        if self.distributed:
            self.denoiser = DistributedDataParallel(
                self.denoiser,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )

        # Audiobox aesthetics predictor — loaded once at init, runs on GPU.
        # Resamples any input to 16 kHz internally before scoring.
        if _AES_AVAILABLE and self.is_primary and not self._benchmark_only():
            self.aes_predictor = _initialize_aes_predictor()
            print("Audiobox aesthetics predictor loaded.")
        else:
            self.aes_predictor = None
            if self.is_primary:
                print("Warning: audiobox_aesthetics not installed; PQ scoring disabled.")

    # ---------------------------------------------------------------- setup --

    def _build_models(self):
        """Construct codec, denoiser, generative state, optimizer, and corruptor."""
        # DDPM and CFM are intentionally SAME-L-only in this repo. Their DiT
        # denoisers expect 256-channel SAME latents and latent conditioning.
        if (
            self._is_same_latent_generative() or self._is_dit_mse()
        ) and self.cfg["codec"]["name"] != "same_l":
            raise ValueError(
                "denoiser.type=ddpm/cfm/dit_mse is supported only with codec.name=same_l"
            )

        # Codec-backed models train in frozen latent space. Waveform/STFT
        # deterministic models do not instantiate SAME/DAC/EnCodec.
        if self._uses_codec():
            self.codec = build_codec(self.cfg["codec"]["name"], self.device)
            self.sample_rate = self.codec.sample_rate

        # Denoiser factory keeps latent, waveform, STFT, and generative models
        # swappable from config. The print below is the source of truth for
        # parameter count after architecture/domain changes.
        denoiser_cfg = dict(self.cfg["denoiser"])
        if self._is_ddpm_clean_cond():
            denoiser_cfg["type"] = "ddpm"
        latent_dim = self.codec.latent_dim if self.codec is not None else None
        self.denoiser = build_denoiser(
            denoiser_cfg,
            latent_dim,
            spectral_cfg=self.spectral_cfg,
        ).to(self.device) #gpu or CPU?
        if self._is_ddpm() or self._is_ddpm_clean_cond():
            dcfg = self._ddpm_cfg()

            # DDPM schedule tensors are precomputed once and gathered by
            # timestep during q_sample() and reverse sampling.
            self.ddpm_buffers = precompute_ddpm_buffers(
                num_train_timesteps=dcfg["num_train_timesteps"],
                beta_start=dcfg["beta_start"],
                beta_end=dcfg["beta_end"],
                device=self.device,  #CPU or GPU
            )

            # EMA weights are used for validation and TensorBoard DDPM samples.
            # This usually gives smoother restoration samples than raw weights.
            self.ema_denoiser = AveragedModel(
                self.denoiser,
                avg_fn=_ema_avg_fn(dcfg.get("ema_decay", 0.9999)),
            ).to(self.device)  #CPU or GPU
        elif self._is_cfm():
            cfcfg = self._cfm_cfg()
            # CFM validation/audio uses an EMA velocity field, matching the
            # DDPM generative path while keeping the objective deterministic.
            self.ema_denoiser = AveragedModel(
                self.denoiser,
                avg_fn=_ema_avg_fn(cfcfg.get("ema_decay", 0.9999)),
            ).to(self.device)
        lr = self.cfg["training"]["lr"]
        weight_decay = self.cfg["training"]["weight_decay"]
        betas = (0.9, 0.999)
        if self._is_cfm():
            cfcfg = self._cfm_cfg()
            lr = cfcfg.get("learning_rate", lr)
            weight_decay = cfcfg.get("weight_decay", weight_decay)
            betas = tuple(cfcfg.get("betas", betas))
        elif self._is_dit_mse():
            ditcfg = self._dit_mse_cfg()
            lr = ditcfg.get("learning_rate", lr)
            weight_decay = ditcfg.get("weight_decay", weight_decay)
            betas = tuple(ditcfg.get("betas", betas))
        self.optimizer = torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
        self.corruptor = None
        if not self._precomp_fetch():
            self.corruptor = AudioCorruptor(self.cfg["corruption"])
            if _env_flag("RESTOR_PRELOAD_NOISE_GPU", False):
                self.corruptor.preload_noise(self.device, self.sample_rate)

        n_params = sum(p.numel() for p in self.denoiser.parameters())
        if self.codec is None:
            print(f"Codec : none (waveform/STFT domain, sr={self.sample_rate})")
        else:
            print(f"Codec : {self.cfg['codec']['name']} "
                  f"(latent_dim={self.codec.latent_dim}, sr={self.codec.sample_rate})")
        print(f"Denoiser: {self.cfg['denoiser']['type']} ({n_params:,} params)")

    def _build_data(self):
        """Discover song folders and create train/validation DataLoaders."""
        tcfg = self.cfg["training"]
        loader_kw = dict(
            # num_workers: how many background processes load and pre-process
            # audio chunks in parallel while the GPU is busy training. 0 means
            # the main process does all loading (slow). Higher values overlap
            # disk/CPU work with GPU work so training is not bottlenecked on data.
            num_workers=tcfg["num_workers"],
            # pin_memory: allocates DataLoader output tensors in page-locked
            # (pinned) CPU memory, which allows the GPU to DMA-copy them
            # directly without an extra buffer copy. This speeds up .to(device).
            pin_memory=True,
            # worker_init_fn: called once per worker process at startup. Here
            # it seeds Python's random module independently per worker so that
            # data augmentation/sampling is different across workers and across
            # runs when a global seed is set.
            worker_init_fn=_worker_init,
        )

        if self._precomp_fetch():
            pcfg = self._precompute_cfg()
            print(f"Precompute train dir       : {pcfg['train_dir']}")
            print(f"Precompute validate dir    : {pcfg['validate_dir']}")
            print(f"Precompute ground truth dir: {pcfg['ground_truth_dir']}")
            if self._is_stft_deterministic():
                self.train_set = PrecomputedSpectrogramPairDataset(pcfg["train_dir"])
                self.val_set = PrecomputedSpectrogramPairDataset(pcfg["validate_dir"])
                self._load_precomputed_stft_stats()
            elif self._is_waveform_deterministic():
                self.train_set = PrecomputedWaveformPairDataset(pcfg["train_dir"])
                self.val_set = PrecomputedWaveformPairDataset(pcfg["validate_dir"])
                self._load_precomputed_waveform_metadata()
            else:
                self.train_set = PrecomputedLatentPairDataset(pcfg["train_dir"])
                self.val_set = PrecomputedLatentPairDataset(pcfg["validate_dir"])
                self._load_precomputed_latent_stats()
            if self.distributed:
                self.train_sampler = DistributedSampler(
                    self.train_set,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    seed=int(tcfg["seed"]),
                    drop_last=True,
                )
                self.val_sampler = DisjointDistributedEvalSampler(
                    self.val_set,
                    num_replicas=self.world_size,
                    rank=self.rank,
                )
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=tcfg["batch_size"],
                shuffle=self.train_sampler is None,
                sampler=self.train_sampler,
                drop_last=True,
                **loader_kw,
            )
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=tcfg["batch_size"],
                shuffle=False,
                sampler=self.val_sampler,
                **loader_kw,
            )
            print(f"Train : {len(self.train_set)} precomputed pairs")
            print(f"Val   : {len(self.val_set)} precomputed pairs")
            return

        # The final public path accepts only the leak-free, song-level split
        # of precomputed Full-Orchestra + Section pairs. StemMixDataset, which
        # created arbitrary stem combinations online, belonged to early
        # experiments and is intentionally not part of this release.
        raise ValueError(
            "SAMECFM-FOS requires precompute.enabled=true and explicit "
            "precompute.train_dir/validate_dir/ground_truth_dir paths."
        )

    def _build_logging(self):
        """Create/log experiment metadata before training or resume."""
        # Snapshot config and create checkpoint/TensorBoard directories inside
        # the experiment. Resume reloads this snapshot instead of the current
        # config file, which prevents accidental config drift mid-run.
        if self.is_primary:
            self.exp.log_config()
            os.makedirs(self.cfg["training"]["checkpoint_dir"], exist_ok=True)
        if self.distributed:
            dist.barrier()

    def _try_resume(self):
        """Load the latest checkpoint if the experiment already has one."""
        self._build_logging()
        ckpt_path = self.exp.find_latest_checkpoint()
        if ckpt_path is None:
            self._try_initialize_from_release()
            return

        # Checkpoints contain model/optimizer state for every denoiser. DDPM
        # adds EMA weights, latent normalization stats, and schedule buffers.
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.denoiser.load_state_dict(ckpt["denoiser"])
        if ckpt.get("latent_mean") is not None:
            self.latent_mean = ckpt["latent_mean"].to(self.device)
        if ckpt.get("latent_std") is not None:
            self.latent_std = ckpt["latent_std"].to(self.device)
        if ckpt.get("stft_mean") is not None:
            self.stft_mean = ckpt["stft_mean"].to(self.device)
        if ckpt.get("stft_std") is not None:
            self.stft_std = ckpt["stft_std"].to(self.device)
        if ckpt.get("spectral_cfg") is not None:
            self.spectral_cfg.update(ckpt["spectral_cfg"])
        if self._is_same_latent_generative():
            if "ema_denoiser" in ckpt:
                self.ema_denoiser.load_state_dict(ckpt["ema_denoiser"])
            else:
                self.ema_denoiser.module.load_state_dict(ckpt["denoiser"])
        if self._is_ddpm() or self._is_ddpm_clean_cond():
            if "diffusion_schedule_buffers" in ckpt:
                self.ddpm_buffers = move_ddpm_buffers(
                    ckpt["diffusion_schedule_buffers"], self.device
                )
        cached_audio_examples = ckpt.get("precomputed_val_audio_examples")
        if cached_audio_examples is not None:
            self._precomputed_val_audio_examples = {
                key: value.detach().cpu()
                for key, value in cached_audio_examples.items()
            }
        self._precomputed_val_fixed_audio_logged = bool(
            ckpt.get("precomputed_val_fixed_audio_logged", False)
        )
        self._best_total_val_audio_loss = ckpt.get("best_total_val_audio_loss")
        self._best_total_val_audio_step = ckpt.get("best_total_val_audio_step")
        self._best_total_val_audio_epoch = ckpt.get("best_total_val_audio_epoch")
        logged_gt = ckpt.get("logged_precomputed_ground_truth", [])
        self._logged_precomputed_ground_truth = {
            (str(tag), int(window_idx)) for tag, window_idx in logged_gt
        }
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        print(f"Resumed from {ckpt_path} at step {self.global_step}")

    def _try_initialize_from_release(self):
        """Initialize a new optimization run from a public EMA checkpoint."""
        checkpoint_path = self.cfg.get("training", {}).get("init_checkpoint")
        if not checkpoint_path:
            return
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Training initialization checkpoint not found: {checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        released_type = checkpoint.get("config", {}).get("denoiser", {}).get("type")
        active_type = self.cfg.get("denoiser", {}).get("type")
        if released_type and released_type != active_type:
            raise ValueError(
                "Initialization checkpoint/config denoiser mismatch: "
                f"{released_type!r} != {active_type!r}"
            )

        if "ema_denoiser" in checkpoint and self.ema_denoiser is not None:
            ema_state = checkpoint["ema_denoiser"]
            model_state = {
                key.removeprefix("module."): value
                for key, value in ema_state.items()
                if key.startswith("module.")
            }
            self.denoiser.load_state_dict(model_state)
            self.ema_denoiser.load_state_dict(ema_state)
        elif "denoiser" in checkpoint:
            self.denoiser.load_state_dict(checkpoint["denoiser"])
            if self.ema_denoiser is not None:
                self.ema_denoiser.module.load_state_dict(checkpoint["denoiser"])
        else:
            raise ValueError(
                "Initialization checkpoint has neither EMA nor raw denoiser weights"
            )

        # This is initialization, not an optimizer resume: the new experiment
        # starts at step zero with a fresh optimizer. Latent statistics remain
        # those of the supplied precomputed cache so decoding/logging matches
        # that cache's normalization.
        self.global_step = 0
        print(
            f"Initialized denoiser and EMA from {checkpoint_path}; "
            "optimizer and global step start fresh."
        )

    def _ensure_ddpm_latent_stats(self):
        """Compute clean SAME latent stats for online latent-model normalization."""
        if self._precomp_fetch():
            if self._is_stft_deterministic():
                print("Using precomputed normalized STFTs; skipping SAME latent stats.")
            elif self._is_waveform_deterministic():
                print("Using precomputed waveforms; skipping SAME latent stats.")
            else:
                print("Using precomputed normalized latents; skipping SAME latent stats.")
            return
        if (
            not (self._is_same_latent_generative() or self._is_dit_mse())
            or self.latent_mean is not None
        ):
            return

        # DDPM and CFM train in normalized SAME latent space. These clean-latent
        # stats are fixed for the run and saved in checkpoints so online
        # training, validation, audio sampling, and inference use one scale.
        print("Computing SAME clean-latent mean/std from training set ...")
        tcfg = self.cfg["training"]
        stats_loader = DataLoader(
            self.train_set,
            batch_size=tcfg["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=tcfg["num_workers"],
            pin_memory=True,
            worker_init_fn=_worker_init,
        )
        self.latent_mean, self.latent_std = compute_dataset_latent_stats(
            self.codec,
            stats_loader,
            self.device,
            channels=self.codec.latent_dim,
        )
        self.latent_mean = self.latent_mean.to(self.device)
        self.latent_std = self.latent_std.to(self.device)
        print("SAME latent stats ready.")

    def _objective_loss_name(self):
        """Name the active training/validation objective for TensorBoard tags."""
        if self._is_stft_deterministic():
            names = {
                "complex_stft_mse": "STFT_complex_mse_loss",
                "multires_stft_mse": "multi_resolution_STFT_mse_loss",
                "auralossMRSTFT": "auraloss_MultiResolutionSTFT_loss",
                "multiscale_spectral_loss_DDSP": "DDSP_multiscale_spectral_loss",
            }
            return names[self._stft_objective()]
        if self._is_waveform_deterministic():
            return "waveform_mse_loss"
        if self._is_ddpm() or self._is_ddpm_clean_cond():
            return "DDPM_epsilon_mse_loss"
        if self._is_cfm():
            return "CFM_velocity_mse_loss"
        if self._is_dit_mse():
            names = {
                "latent_mse": "SAME_latent_mse_loss",
                "auralossMRSTFT": "DiT_auraloss_MultiResolutionSTFT_loss",
                "multiscale_spectral_loss_DDSP": "DiT_DDSP_multiscale_spectral_loss",
            }
            return names[self._dit_mse_objective()]
        return "SAME_latent_mse_loss"

    def _train_loss_tag(self):
        """Per-step training loss tag grouped by comparable objective."""
        objective = self._objective_loss_name()
        if objective == "SAME_latent_mse_loss":
            return f"train/{objective}_train"
        return f"train/{objective}"

    def _epoch_train_loss_tag(self):
        """Per-epoch training loss tag grouped by comparable objective."""
        return f"epoch/{self._objective_loss_name()}_train"

    def _epoch_val_loss_tag(self, suffix=None):
        """Per-epoch validation loss tag grouped by comparable objective."""
        suffix = "" if suffix is None else f"_{suffix}"
        return f"epoch/{self._objective_loss_name()}_val{suffix}"

    def _step_val_loss_tag(self, suffix=None):
        """Step-axis validation loss tag grouped by comparable objective."""
        suffix = "" if suffix is None else f"_{suffix}"
        return f"step/{self._objective_loss_name()}_val{suffix}"

    # --------------------------------------------------------- training loop --

    def train(self):
        """Run all epochs; step-based triggers inside _train_epoch handle all logging.

        Cadence (configurable via training.*_every_steps and keep_last_checkpoints):
          - every step          : per-step train loss scalar
          - step 1 + every 15k  : instantaneous loss on step axis (scalars)
          - step 1 + every 30k  : full validation pass, val-loss scalars
          - every 150k (not 1)  : validation WITH audio, best-val-audio check
          - every checkpoint_every steps : checkpoint (keep_last_checkpoints)
        """
        self._ensure_runtime_defaults()
        self._pending_checkpoint = False
        self._pending_audio_log = False
        self._pending_stop = False
        max_steps = self._configured_max_steps()

        if max_steps is not None and self.global_step >= max_steps:
            print(
                f"Training target already reached: global_step={self.global_step}, "
                f"max_steps={max_steps}. No optimizer step was run.",
                flush=True,
            )
            self.exp.close()
            return

        def _sigusr1_handler(signum, frame):
            self._pending_checkpoint = True
            self._pending_stop = _env_flag("QUEUE_CHECKPOINT_AND_STOP", False)
            print(
                f"[signal] SIGUSR1 received at step {self.global_step}: "
                "checkpoint will save at next step boundary"
                + (" and training will stop." if self._pending_stop else "."),
                flush=True,
            )

        def _sigusr2_handler(signum, frame):
            self._pending_audio_log = True
            print(
                f"[signal] SIGUSR2 received at step {self.global_step}: "
                "audio will log at next validation step.",
                flush=True,
            )

        signal.signal(signal.SIGUSR1, _sigusr1_handler)
        signal.signal(signal.SIGUSR2, _sigusr2_handler)
        if self.is_primary:
            self._write_queue_pid()

        for epoch in range(self.cfg["training"]["num_epochs"]):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            if self.is_primary and not self._benchmark_only():
                print(f"\n--- Epoch {epoch + 1}/{self.cfg['training']['num_epochs']} ---")
            mean_loss = self._train_epoch()
            if self.is_primary and not self._benchmark_only():
                print(
                f"  Epoch {epoch + 1} complete.  "
                f"mean_train_loss={mean_loss:.6f}  "
                f"global_step={self.global_step}",
                flush=True,
                )
            if self._pending_stop:
                if self.is_primary:
                    print("[signal] Checkpoint complete; stopping cleanly for queue advance.", flush=True)
                break
            if max_steps is not None and self.global_step >= max_steps:
                if self.is_primary and not self._benchmark_only():
                    print(
                    f"Training complete at configured max_steps={max_steps}.",
                    flush=True,
                    )
                break
        if self._benchmark_only():
            self._print_benchmark_result()
        self.exp.close()

    def _configured_max_steps(self):
        """Return an optional positive optimizer-step limit.

        ``None`` preserves the historical epoch-only stopping behavior. A
        boolean is rejected explicitly because ``bool`` is a subclass of
        ``int`` in Python and would otherwise silently become a one-step run.
        """
        value = self.cfg.get("training", {}).get("max_steps")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("training.max_steps must be null or a positive integer")
        return value

    def _benchmark_only(self):
        return bool(self.cfg.get("training", {}).get("benchmark_only", False))

    def _ensure_runtime_defaults(self):
        """Keep legacy/lightweight single-process Trainer construction valid."""
        if not hasattr(self, "distributed"):
            self.distributed = False
        if not hasattr(self, "world_size"):
            self.world_size = 1
        if not hasattr(self, "rank"):
            self.rank = 0
        if not hasattr(self, "is_primary"):
            self.is_primary = True
        if not hasattr(self, "train_sampler"):
            self.train_sampler = None
        if not hasattr(self, "_pending_stop"):
            self._pending_stop = False

    def _raw_denoiser(self):
        return self.denoiser.module if isinstance(
            self.denoiser, DistributedDataParallel
        ) else self.denoiser

    def _write_queue_pid(self):
        path = os.environ.get("QUEUE_RANK0_PID_FILE")
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
        os.replace(temporary, path)

    def _distributed_mean(self, value):
        if not getattr(self, "distributed", False):
            return float(value)
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float((tensor / self.world_size).item())

    def _distributed_max(self, value):
        if not getattr(self, "distributed", False):
            return float(value)
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def _distributed_validation_totals(self, values):
        """Sum validation numerators/counts across disjoint rank shards."""
        totals = torch.tensor(
            [float(value) for value in values],
            device=self.device,
            dtype=torch.float64,
        )
        if getattr(self, "distributed", False):
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return [float(value) for value in totals.tolist()]

    def _print_benchmark_result(self):
        measured = self._benchmark_step_seconds
        if not measured:
            raise RuntimeError("Batch benchmark completed without measured steps")
        step_seconds = sum(measured) / len(measured)
        per_gpu_batch = int(self.cfg["training"]["batch_size"])
        peak = torch.tensor(
            torch.cuda.max_memory_reserved(self.device),
            device=self.device,
            dtype=torch.int64,
        )
        total = torch.cuda.get_device_properties(self.device).total_memory
        checksum = sum(
            parameter.detach().sum(dtype=torch.float64)
            for parameter in self._raw_denoiser().parameters()
        )
        if self.distributed:
            rank_peaks = [torch.zeros_like(peak) for _ in range(self.world_size)]
            rank_checksums = [torch.zeros_like(checksum) for _ in range(self.world_size)]
            dist.all_gather(rank_peaks, peak)
            dist.all_gather(rank_checksums, checksum)
            sample_ids = list(itertools.islice(iter(self.train_sampler), 16))
            sample_tensor = torch.tensor(sample_ids, device=self.device, dtype=torch.int64)
            rank_samples = [torch.zeros_like(sample_tensor) for _ in range(self.world_size)]
            dist.all_gather(rank_samples, sample_tensor)
        else:
            rank_peaks = [peak]
            rank_checksums = [checksum]
            rank_samples = []
        if not self.is_primary:
            return
        peak_values = [int(value.item()) for value in rank_peaks]
        checksum_values = [float(value.item()) for value in rank_checksums]
        sample_values = [value.tolist() for value in rank_samples]
        flat_samples = [item for shard in sample_values for item in shard]
        payload = {
            "per_gpu_batch": per_gpu_batch,
            "global_batch": per_gpu_batch * self.world_size,
            "world_size": self.world_size,
            "measured_steps": len(measured),
            "step_seconds": step_seconds,
            "global_samples_per_second": (
                per_gpu_batch * self.world_size / max(step_seconds, 1e-12)
            ),
            "peak_memory_bytes": max(peak_values),
            "peak_memory_bytes_per_rank": peak_values,
            "total_memory_bytes": int(total),
            "peak_memory_fraction": float(max(peak_values) / total),
            "parameter_checksums_per_rank": checksum_values,
            "parameter_checksum_max_delta": max(checksum_values) - min(checksum_values),
            "shard_probe_disjoint": len(flat_samples) == len(set(flat_samples)),
            "shard_sample_ids_per_rank": sample_values,
        }
        print("[batch_probe] " + json.dumps(payload, sort_keys=True), flush=True)

    def _train_epoch(self):
        """Run one full pass over the training DataLoader.

        All TensorBoard / validation / audio triggers are step-based and fired
        from inside this loop so they never miss a boundary regardless of epoch
        size.  ``global_step`` is incremented at the *top* of each iteration
        so all logging uses 1-based step indices.
        """
        self._ensure_runtime_defaults()
        self.denoiser.train()
        tcfg = self.cfg["training"]
        max_steps = self._configured_max_steps()
        total, n = 0.0, 0
        benchmark_warmup = int(tcfg.get("benchmark_warmup_steps", 2))
        if self._benchmark_only() and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        iter_end = time.perf_counter()
        for clean in self.train_loader:
            # This guard is also needed when an epoch resumes exactly at the
            # target. It prevents fetching/optimizing one extra batch.
            if max_steps is not None and self.global_step >= max_steps:
                break
            data_wait_s = time.perf_counter() - iter_end
            compute_start = time.perf_counter()
            loss = self._step(clean, data_wait_s=data_wait_s)
            compute_s = time.perf_counter() - compute_start
            step_s = data_wait_s + compute_s
            loss = self._distributed_mean(loss)
            step_s = self._distributed_max(step_s)
            total += loss
            n += 1

            # Increment first so every log uses 1-based step numbers.
            self.global_step += 1

            if self._benchmark_only():
                if self.global_step > benchmark_warmup:
                    self._benchmark_step_seconds.append(step_s)
                iter_end = time.perf_counter()
                continue

            # ----------------------------------------------------------------
            # Throughput diagnostics (unchanged from previous behaviour).
            # ----------------------------------------------------------------
            if self.log_step_timing and self.is_primary:
                if torch.is_tensor(clean):
                    batch_size = int(clean.shape[0])
                elif isinstance(clean, dict):
                    batch_size = next(
                        (
                            int(value.shape[0])
                            for value in clean.values()
                            if torch.is_tensor(value) and value.ndim > 0
                        ),
                        int(tcfg["batch_size"]),
                    )
                else:
                    batch_size = int(tcfg["batch_size"])
                global_batch_size = batch_size * self.world_size
                samples_per_s = global_batch_size / max(step_s, 1e-12)
                print(
                    "[throughput] "
                    f"step={self.global_step} per_gpu_batch={batch_size} "
                    f"global_batch={global_batch_size} "
                    f"step_time={step_s * 1000.0:.2f}ms "
                    f"compute={compute_s * 1000.0:.2f}ms "
                    f"data_wait={data_wait_s * 1000.0:.2f}ms "
                    f"samples_per_s={samples_per_s:.2f}",
                    flush=True,
                )
                self.exp.log_scalar(
                    "throughput/train_step_time_ms",
                    step_s * 1000.0,
                    self.global_step,
                )
                self.exp.log_scalar(
                    "throughput/train_compute_time_ms",
                    compute_s * 1000.0,
                    self.global_step,
                )
                self.exp.log_scalar(
                    "throughput/train_data_wait_time_ms",
                    data_wait_s * 1000.0,
                    self.global_step,
                )
                self.exp.log_scalar(
                    "throughput/train_samples_per_second",
                    samples_per_s,
                    self.global_step,
                )

            # ----------------------------------------------------------------
            # Per-step train loss (always written at every step).
            # ----------------------------------------------------------------
            if self.is_primary:
                self.exp.log_scalar(self._train_loss_tag(), loss, self.global_step)

            # ----------------------------------------------------------------
            # Validation + audio step boundaries.
            # val_every_steps (default 30k): run full validation, log scalars.
            # log_audio_every_steps (default 150k, never step 1): also log
            # audio/spectrograms and check for new best-val-loss audio.
            # SIGUSR2 forces audio at the next validation step.
            # ----------------------------------------------------------------
            force_audio = getattr(self, "_pending_audio_log", False)
            run_val = self._should_run_val_step() or force_audio
            log_audio = self._should_log_audio_step() or force_audio
            if self.distributed:
                validation_flags = torch.tensor(
                    [int(run_val), int(log_audio)],
                    device=self.device,
                    dtype=torch.int32,
                )
                dist.broadcast(validation_flags, src=0)
                run_val, log_audio = map(bool, validation_flags.tolist())

            if run_val:
                if force_audio and self.is_primary:
                    self._pending_audio_log = False

                # Validation computation is sharded across every rank. Keep
                # preview audio and all persistent side effects on rank zero.
                distributed_model = self.denoiser
                self.denoiser = self._raw_denoiser()
                if log_audio and self.is_primary:
                    if self._precomp_fetch():
                        self._log_precomputed_audio("train", clean)
                    else:
                        self._log_audio("train", clean)

                if self._is_same_latent_generative():
                    val_loss_raw, val_loss_ema = self._validate(
                        log_audio=log_audio and self.is_primary
                    )
                    if self.is_primary:
                        self.exp.log_scalar(
                            self._step_val_loss_tag("raw"), val_loss_raw, self.global_step
                        )
                        self.exp.log_scalar(
                            self._step_val_loss_tag("ema"), val_loss_ema, self.global_step
                        )
                        print(
                            f"  [step {self.global_step}] "
                            f"val_loss_raw={val_loss_raw:.6f}  "
                            f"val_loss_ema={val_loss_ema:.6f}",
                            flush=True,
                        )
                        if log_audio:
                            self._maybe_log_best_total_validation_audio(
                                val_loss_raw, log_audio=True
                            )
                else:
                    val_loss = self._validate(
                        log_audio=log_audio and self.is_primary
                    )
                    if self.is_primary:
                        self.exp.log_scalar(
                            self._step_val_loss_tag(), val_loss, self.global_step
                        )
                        print(
                            f"  [step {self.global_step}] val_loss={val_loss:.6f}",
                            flush=True,
                        )
                        if log_audio:
                            self._maybe_log_best_total_validation_audio(
                                val_loss, log_audio=True
                            )

                # Restore training mode; _validate() sets eval mode and audio
                # loggers call train() internally, but we restore here for safety.
                self.denoiser = distributed_model
                self.denoiser.train()
                if self.ema_denoiser is not None:
                    self.ema_denoiser.train()
            if run_val and self.distributed:
                dist.barrier()

            # ----------------------------------------------------------------
            # Checkpoint.
            # ----------------------------------------------------------------
            manual_checkpoint = bool(
                self.is_primary and getattr(self, "_pending_checkpoint", False)
            )
            stop_requested = bool(
                self.is_primary and getattr(self, "_pending_stop", False)
            )
            if self.distributed:
                flags = torch.tensor(
                    [int(manual_checkpoint), int(stop_requested)],
                    device=self.device,
                    dtype=torch.int32,
                )
                dist.broadcast(flags, src=0)
                manual_checkpoint, stop_requested = map(bool, flags.tolist())
                self._pending_stop = stop_requested
            periodic_checkpoint = self.global_step % tcfg["checkpoint_every"] == 0
            if (periodic_checkpoint or manual_checkpoint) and self.is_primary:
                if manual_checkpoint:
                    print(
                        f"[signal] Saving on-demand checkpoint at step {self.global_step}.",
                        flush=True,
                    )
                self._save_checkpoint()
                self._pending_checkpoint = False
            if (periodic_checkpoint or manual_checkpoint) and self.distributed:
                dist.barrier()
            if manual_checkpoint and not self.is_primary:
                self._pending_checkpoint = False
            if stop_requested:
                break

            if max_steps is not None and self.global_step >= max_steps:
                # Periodic checkpoints may not divide a user-selected target.
                # Avoid a duplicate write when the ordinary cadence already
                # saved this exact step.
                if self.global_step % tcfg["checkpoint_every"] != 0 and self.is_primary:
                    self._save_checkpoint()
                if self.distributed:
                    dist.barrier()
                break

            iter_end = time.perf_counter()
        return total / max(n, 1)

    def _step(self, clean, data_wait_s=None):
        """Dispatch one training batch to the objective for the active denoiser."""
        if self._precomp_fetch() and self._is_stft_deterministic():
            if self._uses_multires_stft_objective():
                return self._step_stft_multires_precomputed(
                    clean, data_wait_s=data_wait_s
                )
            return self._step_stft_precomputed(clean, data_wait_s=data_wait_s)
        elif self._precomp_fetch() and self._is_waveform_deterministic():
            return self._step_waveform_precomputed(clean, data_wait_s=data_wait_s)
        elif self._is_stft_deterministic():
            if self._uses_multires_stft_objective():
                return self._step_stft_multires(clean, data_wait_s=data_wait_s)
            return self._step_stft(clean, data_wait_s=data_wait_s)
        elif self._is_waveform_deterministic():
            return self._step_waveform(clean, data_wait_s=data_wait_s)
        elif self._precomp_fetch() and (self._is_ddpm() or self._is_ddpm_clean_cond()):
            return self._step_ddpm_precomputed(clean, data_wait_s=data_wait_s)
        elif self._precomp_fetch() and self._is_cfm():
            return self._step_cfm_precomputed(clean, data_wait_s=data_wait_s)
        elif self._precomp_fetch():
            return self._step_precomputed(clean, data_wait_s=data_wait_s)
        elif self._is_ddpm():
            return self._step_ddpm(clean, data_wait_s=data_wait_s)
        elif self._is_ddpm_clean_cond():
            return self._step_ddpm_clean_cond(clean, data_wait_s=data_wait_s)
        elif self._is_cfm():
            return self._step_cfm(clean, data_wait_s=data_wait_s)
        elif self._is_dit_mse():
            return self._step_dit_mse(clean, data_wait_s=data_wait_s)
        return self._step_one_shot(clean, data_wait_s=data_wait_s)

    def _step_waveform(self, clean, data_wait_s=None):
        """Train TCN as corrupted waveform -> clean waveform."""
        profile = self._new_profile("waveform_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        clean = clean.to(self.device, non_blocking=True)
        corrupt = self._corrupt_batch(clean)
        self._profile_mark(profile, "waveform_corruption")

        pred = self.denoiser(corrupt)
        self._profile_mark(profile, "denoiser_forward")
        loss = waveform_mse_loss(pred, clean)
        self._profile_mark(profile, "loss_compute")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self._profile_print("waveform_train_step", profile, loss.item())
        return loss.item()

    def _step_stft(self, clean, data_wait_s=None):
        """Train Conformer/U-Net as normalized complex-STFT restoration."""
        profile = self._new_profile("stft_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        clean = clean.to(self.device, non_blocking=True)
        corrupt = self._corrupt_batch(clean)
        self._profile_mark(profile, "waveform_corruption")

        x_stft = normalize_stft(
            audio_to_stft_channels(corrupt, self.spectral_cfg),
            self.stft_mean,
            self.stft_std,
        )
        y_stft = normalize_stft(
            audio_to_stft_channels(clean, self.spectral_cfg),
            self.stft_mean,
            self.stft_std,
        )
        self._profile_mark(profile, "stft_transform_normalize")

        pred_stft = self.denoiser(x_stft)
        self._profile_mark(profile, "denoiser_forward")
        loss = complex_stft_mse_loss(pred_stft, y_stft)
        self._profile_mark(profile, "loss_compute")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self._profile_print("stft_train_step", profile, loss.item())
        return loss.item()

    def _step_stft_multires(self, clean, data_wait_s=None):
        """Train an STFT model through an opt-in waveform spectral objective.

        This is an opt-in path. The existing normalized complex-STFT MSE step
        above remains unchanged. Gradients flow from the selected spectral
        loss through ISTFT and STFT denormalization into the spectrogram model.
        """
        profile = self._new_profile("stft_multires_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        clean = clean.to(self.device, non_blocking=True)
        corrupt = self._corrupt_batch(clean)
        self._profile_mark(profile, "waveform_corruption")

        x_stft = normalize_stft(
            audio_to_stft_channels(corrupt, self.spectral_cfg),
            self.stft_mean,
            self.stft_std,
        )
        self._profile_mark(profile, "stft_transform_normalize")

        pred_stft_norm = self.denoiser(x_stft)
        self._profile_mark(profile, "denoiser_forward")

        # The model predicts normalized real/imaginary STFT channels. Undo that
        # affine transform, then use differentiable ISTFT to obtain waveform.
        pred_stft = unnormalize_stft(
            pred_stft_norm, self.stft_mean, self.stft_std
        )
        pred_audio = stft_channels_to_audio(
            pred_stft,
            length=clean.shape[-1],
            spectral_cfg=self.spectral_cfg,
        )
        self._profile_mark(profile, "stft_denormalize_istft")

        loss = self._stft_waveform_objective_loss(pred_audio, clean)
        self._profile_mark(profile, "waveform_spectral_loss")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self._profile_print("stft_multires_train_step", profile, loss.item())
        return loss.item()

    def _step_one_shot(self, clean, data_wait_s=None):
        """Train mlp/conformer models by direct corrupt-latent to clean-latent regression."""
        profile = self._new_profile("train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        # Original denoiser path: corrupt waveform, encode clean/corrupt
        # latents, and directly regress denoised latent toward clean latent.
        corrupt = self._corrupt_batch(clean)
        self._profile_mark(profile, "waveform_corruption")
        clean = self.codec.preprocess(clean.to(self.device))
        corrupt = self.codec.preprocess(corrupt.to(self.device))
        self._profile_mark(profile, "preprocess_to_device")

        with torch.no_grad():
            z_clean = self.codec.encode(clean)
            z_corrupt = self.codec.encode(corrupt)
        self._profile_mark(profile, "codec_encode_clean_corrupt")

        z_denoised = self.denoiser(z_corrupt)
        self._profile_mark(profile, "denoiser_forward")

        lcfg = self.cfg.get("loss", {})
        loss = lcfg.get("latent_weight", 1.0) * latent_mse_loss(z_denoised, z_clean)

        spec_w = lcfg.get("spectral_weight", 0.0)
        if spec_w > 0:
            audio_denoised = self.codec.decode(z_denoised)
            audio_clean = self.codec.decode(z_clean)
            loss = loss + spec_w * multiscale_spectral_loss(
                audio_denoised, audio_clean,
                fft_sizes=lcfg.get("spectral_fft_sizes", [512, 1024, 2048]),
            )
        self._profile_mark(profile, "loss_compute")

        self.optimizer.zero_grad()
        loss.backward()
        self._profile_mark(profile, "backward")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self._profile_print("train_step", profile, loss.item())
        return loss.item()

    def _step_dit_mse(self, clean, data_wait_s=None):
        """Train deterministic DiT from real-time waveform corruption/SAME encoding."""
        profile = self._new_profile("dit_mse_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        # SAME is frozen. Encode clean/corrupt together once, then train only
        # through DiT and (for waveform objectives) one differentiable decode.
        z_cond, z_clean = self._encode_ddpm_pair(clean, profile=profile)
        z_denoised = self.denoiser(z_cond)
        self._profile_mark(profile, "denoiser_forward")

        if self._uses_dit_waveform_spectral_objective():
            target_audio = clean.to(self.device, non_blocking=True)
            pred_audio = self._decode_dit_prediction_for_loss(
                z_denoised, profile=profile
            )
            loss = self._dit_waveform_objective_loss(pred_audio, target_audio)
            self._profile_mark(profile, "waveform_spectral_loss")
        else:
            loss = latent_mse_loss(z_denoised, z_clean)
            self._profile_mark(profile, "latent_mse_loss")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        clip_norm = self._dit_mse_cfg().get("gradient_clip_norm")
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), clip_norm)
            self._profile_mark(profile, "gradient_clip")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self._profile_print("dit_mse_train_step", profile, loss.item())
        return loss.item()

    def _step_ddpm(self, clean, data_wait_s=None):
        """Train DDPM by predicting injected Gaussian epsilon in SAME latent space."""
        profile = self._new_profile("ddpm_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        # DDPM conditioning pair:
        #   clean waveform   -> z_clean, the latent we diffuse/noise
        #   corrupt waveform -> z_cond, the restoration condition
        z_cond, z_clean = self._encode_ddpm_pair(clean, profile=profile)
        B = z_clean.shape[0]
        dcfg = self._ddpm_cfg()

        # Each item receives an independent random diffusion timestep and
        # Gaussian epsilon. The target is epsilon, not z_clean.
        # num_train_timesteps is the size of the diffusion chain (T), i.e.
        # valid timestep indices are [0, T-1]. It is not the count of
        # optimizer/backprop updates.
        t = torch.randint(
            0,
            dcfg["num_train_timesteps"],
            (B,),
            device=z_clean.device,
            dtype=torch.long,
        )
        epsilon = torch.randn_like(z_clean)
        z_t = q_sample(z_clean, t, epsilon, self.ddpm_buffers)
        self._profile_mark(profile, "sample_t_epsilon_qsample")

        # DiT predicts the noise component in z_t, conditioned on z_cond.
        eps_pred = self.denoiser(z_t, t, z_cond)
        self._assert_ddpm_shapes(z_cond, z_clean, z_t, epsilon, eps_pred, t)
        self._profile_mark(profile, "dit_forward")

        loss = latent_mse_loss(eps_pred, epsilon)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        clip_norm = dcfg.get("gradient_clip_norm")
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), clip_norm)
            self._profile_mark(profile, "gradient_clip")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self.ema_denoiser.update_parameters(self._raw_denoiser())
        self._profile_mark(profile, "ema_update")
        self._profile_print("ddpm_train_step", profile, loss.item())
        return loss.item()

    def _step_ddpm_clean_cond(self, clean, data_wait_s=None):
        """Train DDPM by predicting injected Gaussian epsilon in SAME latent space."""
        profile = self._new_profile("ddpm_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        # DDPM conditioning pair:
        #   clean waveform   -> z_clean, the latent we diffuse/noise, and restoration condition
        #   corrupt waveform -> z_cond, not used here
        _ , z_clean = self._encode_ddpm_pair(clean, profile=profile)
        B = z_clean.shape[0]
        dcfg = self._ddpm_cfg()

        # Each item receives an independent random diffusion timestep and
        # Gaussian epsilon. The target is epsilon, not z_clean.
        # num_train_timesteps is the size of the diffusion chain (T), i.e.
        # valid timestep indices are [0, T-1]. It is not the count of
        # optimizer/backprop updates.
        t = torch.randint(
            0,
            dcfg["num_train_timesteps"],
            (B,),
            device=z_clean.device,
            dtype=torch.long,
        )
        epsilon = torch.randn_like(z_clean)
        z_t = q_sample(z_clean, t, epsilon, self.ddpm_buffers)
        self._profile_mark(profile, "sample_t_epsilon_qsample")

        # DiT predicts the noise component in z_t, conditioned on z_clean.
        eps_pred = self.denoiser(z_t, t, z_clean)
        self._assert_ddpm_shapes(z_clean, z_clean, z_t, epsilon, eps_pred, t)
        self._profile_mark(profile, "dit_forward")

        loss = latent_mse_loss(eps_pred, epsilon)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        clip_norm = dcfg.get("gradient_clip_norm")
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), clip_norm)
            self._profile_mark(profile, "gradient_clip")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self.ema_denoiser.update_parameters(self._raw_denoiser())
        self._profile_mark(profile, "ema_update")
        self._profile_print("ddpm_train_step", profile, loss.item())
        return loss.item()

    def _step_ddpm_precomputed(self, batch, data_wait_s=None):
        """Train DDPM from precomputed normalized z_cond/z_clean latent pairs."""
        profile = self._new_profile("ddpm_precomputed_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        z_cond, z_clean = self._precomputed_latents_to_device(batch, profile=profile)
        condition = z_clean if self._is_ddpm_clean_cond() else z_cond
        B = z_clean.shape[0]
        dcfg = self._ddpm_cfg()

        t = torch.randint(
            0,
            dcfg["num_train_timesteps"],
            (B,),
            device=z_clean.device,
            dtype=torch.long,
        )
        epsilon = torch.randn_like(z_clean)
        z_t = q_sample(z_clean, t, epsilon, self.ddpm_buffers)
        self._profile_mark(profile, "sample_t_epsilon_qsample")

        eps_pred = self.denoiser(z_t, t, condition)
        self._assert_ddpm_shapes(condition, z_clean, z_t, epsilon, eps_pred, t)
        self._profile_mark(profile, "dit_forward")

        loss = latent_mse_loss(eps_pred, epsilon)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        clip_norm = dcfg.get("gradient_clip_norm")
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), clip_norm)
            self._profile_mark(profile, "gradient_clip")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self.ema_denoiser.update_parameters(self._raw_denoiser())
        self._profile_mark(profile, "ema_update")
        self._profile_print("ddpm_precomputed_train_step", profile, loss.item())
        return loss.item()

    def _step_cfm(self, clean, data_wait_s=None):
        """Train CFM with online corruption and batched SAME encoding.

        This mirrors the online DDPM data path. The only objective-specific
        difference comes after encoding: CFM samples a continuous straight
        path from Gaussian noise to z_clean and predicts its velocity.
        """
        profile = self._new_profile("cfm_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        z_cond, z_clean = self._encode_ddpm_pair(clean, profile=profile)
        z_t, t, velocity_target = self._sample_cfm_training_tuple(z_clean)
        z_cond_train = self._apply_cfm_cfg_dropout(z_cond)
        self._profile_mark(profile, "sample_t_noise_velocity")

        velocity_pred = self.denoiser(z_t, t, z_cond_train)
        self._assert_cfm_shapes(
            z_cond_train, z_clean, z_t, velocity_target, velocity_pred, t
        )
        self._profile_mark(profile, "dit_forward")

        loss = latent_mse_loss(velocity_pred, velocity_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        clip_norm = self._cfm_cfg().get("gradient_clip_norm")
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), clip_norm)
            self._profile_mark(profile, "gradient_clip")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self.ema_denoiser.update_parameters(self._raw_denoiser())
        self._profile_mark(profile, "ema_update")
        self._profile_print("cfm_train_step", profile, loss.item())
        return loss.item()

    def _step_cfm_precomputed(self, batch, data_wait_s=None):
        """Train CFM by predicting straight-path velocity in normalized SAME latents."""
        profile = self._new_profile("cfm_precomputed_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        z_cond, z_clean = self._precomputed_latents_to_device(batch, profile=profile)
        z_t, t, velocity_target = self._sample_cfm_training_tuple(z_clean)
        z_cond_train = self._apply_cfm_cfg_dropout(z_cond)
        self._profile_mark(profile, "sample_t_noise_velocity")

        velocity_pred = self.denoiser(z_t, t, z_cond_train)
        self._assert_cfm_shapes(z_cond_train, z_clean, z_t, velocity_target, velocity_pred, t)
        self._profile_mark(profile, "dit_forward")

        loss = latent_mse_loss(velocity_pred, velocity_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        clip_norm = self._cfm_cfg().get("gradient_clip_norm")
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), clip_norm)
            self._profile_mark(profile, "gradient_clip")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self.ema_denoiser.update_parameters(self._raw_denoiser())
        self._profile_mark(profile, "ema_update")
        self._profile_print("cfm_precomputed_train_step", profile, loss.item())
        return loss.item()

    def _step_precomputed(self, batch, data_wait_s=None):
        """Train deterministic denoisers from precomputed normalized latent pairs."""
        profile = self._new_profile("precomputed_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        if self._is_dit_mse() and self._uses_dit_waveform_spectral_objective():
            # Precomputed z_cond avoids SAME encoding. The clean waveform target
            # is already resident on the GPU, so only the prediction is decoded.
            z_cond = self._precomputed_condition_to_device(batch, profile=profile)
            z_denoised = self.denoiser(z_cond)
            self._profile_mark(profile, "denoiser_forward")
            target_audio = self._dit_ground_truth_for_batch(batch, profile=profile)
            pred_audio = self._decode_dit_prediction_for_loss(
                z_denoised, profile=profile
            )
            loss = self._dit_waveform_objective_loss(
                pred_audio, target_audio
            )
            self._profile_mark(profile, "waveform_spectral_loss")
        else:
            z_cond, z_clean = self._precomputed_latents_to_device(
                batch, profile=profile
            )
            z_denoised = self.denoiser(z_cond)
            self._profile_mark(profile, "denoiser_forward")
            loss = latent_mse_loss(z_denoised, z_clean)
            self._profile_mark(profile, "latent_mse_loss")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        if self._is_dit_mse():
            clip_norm = self._dit_mse_cfg().get("gradient_clip_norm")
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), clip_norm)
                self._profile_mark(profile, "gradient_clip")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self._profile_print("precomputed_train_step", profile, loss.item())
        return loss.item()

    def _step_waveform_precomputed(self, batch, data_wait_s=None):
        """Train TCN from precomputed corrupted/clean waveform pairs."""
        profile = self._new_profile("waveform_precomputed_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        x_audio, y_audio = self._precomputed_waveforms_to_device(batch, profile=profile)
        pred = self.denoiser(x_audio)
        self._profile_mark(profile, "denoiser_forward")
        loss = waveform_mse_loss(pred, y_audio)
        self._profile_mark(profile, "loss_compute")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self._profile_print("waveform_precomputed_train_step", profile, loss.item())
        return loss.item()

    def _step_stft_precomputed(self, batch, data_wait_s=None):
        """Train Conformer/U-Net from precomputed normalized STFT pairs."""
        profile = self._new_profile("stft_precomputed_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        x_stft, y_stft = self._precomputed_stfts_to_device(batch, profile=profile)
        pred_stft = self.denoiser(x_stft)
        self._profile_mark(profile, "denoiser_forward")
        loss = complex_stft_mse_loss(pred_stft, y_stft)
        self._profile_mark(profile, "loss_compute")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self._profile_print("stft_precomputed_train_step", profile, loss.item())
        return loss.item()

    def _step_stft_multires_precomputed(self, batch, data_wait_s=None):
        """Train precomputed STFT pairs with an opt-in waveform spectral loss."""
        profile = self._new_profile("stft_multires_precomputed_train_step")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        x_stft, y_stft = self._precomputed_stfts_to_device(batch, profile=profile)
        pred_stft_norm = self.denoiser(x_stft)
        self._profile_mark(profile, "denoiser_forward")

        length = self._precomputed_stft_num_samples(batch)
        pred_stft = unnormalize_stft(
            pred_stft_norm, self.stft_mean, self.stft_std
        )
        pred_audio = stft_channels_to_audio(
            pred_stft, length=length, spectral_cfg=self.spectral_cfg
        )

        # The clean target does not need a graph. Keeping its denormalization
        # and ISTFT under no_grad saves activation memory while the prediction
        # remains fully differentiable through its own ISTFT path.
        with torch.no_grad():
            target_stft = unnormalize_stft(
                y_stft, self.stft_mean, self.stft_std
            )
            target_audio = stft_channels_to_audio(
                target_stft, length=length, spectral_cfg=self.spectral_cfg
            )
        self._profile_mark(profile, "stft_denormalize_istft")

        loss = self._stft_waveform_objective_loss(pred_audio, target_audio)
        self._profile_mark(profile, "waveform_spectral_loss")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._profile_mark(profile, "backward")
        self.optimizer.step()
        self._profile_mark(profile, "optimizer_step")
        self._profile_print(
            "stft_multires_precomputed_train_step", profile, loss.item()
        )
        return loss.item()

    # ----------------------------------------------------------- validation --

    @torch.no_grad()
    def _validate(self, log_audio=False): #inside validate_ddpm_batch splits for clean or dirty cond!
        """Evaluate the active objective on the validation loader and log audio if due."""
        epoch_profile = self._new_profile("validation_epoch")
        # Validation mirrors the active objective:
        # - one-shot models are measured by latent reconstruction MSE.
        # - DDPM is measured by epsilon-prediction MSE using both raw and EMA weights.
        self.denoiser.eval()
        if self.ema_denoiser is not None:
            self.ema_denoiser.eval()
        total_sse, total_numel = 0.0, 0
        raw_sse, raw_numel = 0.0, 0
        ema_sse, ema_numel = 0.0, 0
        n = 0
        first_batch = None
        validation_started = time.perf_counter()
        local_batches = len(self.val_loader)
        progress_every = max(1, local_batches // 10)
        if self.is_primary:
            global_items = len(self.val_set)
            val_sampler = getattr(self, "val_sampler", None)
            local_items = len(val_sampler) if val_sampler is not None else global_items
            print(
                f"[validation] step={self.global_step} start "
                f"global_items={global_items} shards={self.world_size} "
                f"rank0_items={local_items} rank0_batches={local_batches}",
                flush=True,
            )
        iter_end = time.perf_counter()
        for clean in self.val_loader:
            data_wait_s = time.perf_counter() - iter_end
            if first_batch is None:
                first_batch = clean
            if self._precomp_fetch():
                max_audio_windows = self._precomputed_validation_audio_windows()
                cached_windows = (
                    0
                    if self._precomputed_val_audio_examples is None
                    else self._precomputed_val_audio_examples["window_idx"].shape[0]
                )
            if self._precomp_fetch() and cached_windows < max_audio_windows:
                precomputed_audio_batch = self._collect_precomputed_audio_windows(
                    self._precomputed_val_audio_examples,
                    clean,
                    max_audio_windows,
                )
                if precomputed_audio_batch is not None:
                    self._precomputed_val_audio_examples = precomputed_audio_batch
            if self._is_stft_deterministic():
                if self._uses_multires_stft_objective():
                    batch_stats = (
                        self._validate_stft_multires_precomputed_batch(
                            clean, data_wait_s=data_wait_s
                        )
                        if self._precomp_fetch()
                        else self._validate_stft_multires_batch(
                            clean, data_wait_s=data_wait_s
                        )
                    )
                else:
                    batch_stats = (
                        self._validate_stft_precomputed_batch(
                            clean, data_wait_s=data_wait_s
                        )
                        if self._precomp_fetch()
                        else self._validate_stft_batch(
                            clean, data_wait_s=data_wait_s
                        )
                    )
                total_sse += batch_stats["sse"]
                total_numel += batch_stats["numel"]

            elif self._is_waveform_deterministic():
                batch_stats = (
                    self._validate_waveform_precomputed_batch(clean, data_wait_s=data_wait_s)
                    if self._precomp_fetch()
                    else self._validate_waveform_batch(clean, data_wait_s=data_wait_s)
                )
                total_sse += batch_stats["sse"]
                total_numel += batch_stats["numel"]

            elif self._is_ddpm() or self._is_ddpm_clean_cond(): #inside validate_ddpm_batch splits for clean or dirty cond!
                # Use the same noised latent target for raw and EMA validation
                # inside _validate_ddpm_batch, so differences reflect weights only.
                batch_losses = self._validate_ddpm_batch(clean, data_wait_s=data_wait_s)
                raw_sse += batch_losses["raw_sse"]
                raw_numel += batch_losses["raw_numel"]
                ema_sse += batch_losses["ema_sse"]
                ema_numel += batch_losses["ema_numel"]

            elif self._is_cfm():
                # Same sampled flow target for raw and EMA validation, so the
                # two curves differ only by weights, not by sampled noise/time.
                batch_losses = self._validate_cfm_batch(clean, data_wait_s=data_wait_s)
                raw_sse += batch_losses["raw_sse"]
                raw_numel += batch_losses["raw_numel"]
                ema_sse += batch_losses["ema_sse"]
                ema_numel += batch_losses["ema_numel"]

            elif self._is_dit_mse():
                batch_stats = (
                    self._validate_precomputed_batch(clean, data_wait_s=data_wait_s)
                    if self._precomp_fetch()
                    else self._validate_dit_mse_batch(clean, data_wait_s=data_wait_s)
                )
                total_sse += batch_stats["sse"]
                total_numel += batch_stats["numel"]

            elif self._precomp_fetch():
                batch_stats = self._validate_precomputed_batch(
                    clean, data_wait_s=data_wait_s
                )
                total_sse += batch_stats["sse"]
                total_numel += batch_stats["numel"]
            else:
                batch_profile = self._new_profile("val_batch")
                self._profile_add_external(batch_profile, "data_loader_wait", data_wait_s)
                corrupt = self._corrupt_batch(clean)
                self._profile_mark(batch_profile, "waveform_corruption")
                clean_d = self.codec.preprocess(clean.to(self.device, non_blocking=True))
                corrupt_d = self.codec.preprocess(corrupt.to(self.device, non_blocking=True))
                self._profile_mark(batch_profile, "preprocess_to_device")

                z_clean = self.codec.encode(clean_d)
                z_corrupt = self.codec.encode(corrupt_d)
                self._profile_mark(batch_profile, "codec_encode_clean_corrupt")
                z_denoised = self.denoiser(z_corrupt)
                self._profile_mark(batch_profile, "denoiser_forward")

                batch_stats = self._mse_stats(z_denoised, z_clean)
                batch_loss = batch_stats["loss"]
                self._profile_mark(batch_profile, "loss_compute")
                self._profile_print("val_batch", batch_profile, batch_loss)
                total_sse += batch_stats["sse"]
                total_numel += batch_stats["numel"]
            n += 1
            if self.is_primary and (
                n == local_batches or n % progress_every == 0
            ):
                elapsed = time.perf_counter() - validation_started
                batches_per_second = n / max(elapsed, 1e-12)
                eta = (local_batches - n) / max(batches_per_second, 1e-12)
                print(
                    f"[validation] step={self.global_step} "
                    f"rank0_batches={n}/{local_batches} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
            iter_end = time.perf_counter()

        self._gather_precomputed_validation_audio_examples()
        (
            total_sse,
            total_numel,
            raw_sse,
            raw_numel,
            ema_sse,
            ema_numel,
        ) = self._distributed_validation_totals(
            [
                total_sse,
                total_numel,
                raw_sse,
                raw_numel,
                ema_sse,
                ema_numel,
            ]
        )

        requested_audio_windows = (
            self._precomputed_validation_audio_window_indices()
            if self._precomp_fetch()
            else None
        )
        if self.is_primary and requested_audio_windows is not None:
            selected_values = (
                []
                if self._precomputed_val_audio_examples is None
                else [
                    int(value.item())
                    for value in self._precomputed_val_audio_examples["window_idx"]
                ]
            )
            selected_audio_windows = set(selected_values)
            missing = [
                value
                for value in requested_audio_windows
                if value not in selected_audio_windows
            ]
            if missing:
                raise ValueError(
                    "Configured validation audio windows were not found in "
                    f"the validation set: {missing}"
                )
            # Canonicalize TensorBoard slot order to the explicit list,
            # independent of validation batch size or filesystem traversal.
            position = {
                window_idx: row
                for row, window_idx in enumerate(selected_values)
            }
            order = torch.tensor(
                [position[value] for value in requested_audio_windows],
                dtype=torch.long,
            )
            self._precomputed_val_audio_examples = {
                key: value.index_select(0, order)
                for key, value in self._precomputed_val_audio_examples.items()
            }

        if self.is_primary and first_batch is not None:
            self._latest_val_audio_batch = (
                (self._precomputed_val_audio_examples or first_batch)
                if self._precomp_fetch()
                else first_batch
            )
        if self.is_primary and first_batch is not None and log_audio:
            if self._precomp_fetch():
                self._log_precomputed_audio(
                    "val",
                    self._latest_val_audio_batch,
                    log_audio_mse=True,
                )
            else:
                self._log_audio("val", first_batch, log_audio_mse=True)
        self._profile_mark(epoch_profile, "validation_batches_and_optional_audio")
        if self.is_primary:
            print(
                f"[validation] step={self.global_step} complete "
                f"elapsed={time.perf_counter() - validation_started:.1f}s",
                flush=True,
            )
        if self._is_same_latent_generative(): #inside validate_ddpm_batch splits for clean or dirty cond!
            raw_avg = raw_sse / max(raw_numel, 1)
            ema_avg = ema_sse / max(ema_numel, 1)
            self._profile_print("validation_epoch", epoch_profile, raw_avg)
            return raw_avg, ema_avg
        val_avg = total_sse / max(total_numel, 1)
        self._profile_print("validation_epoch", epoch_profile, val_avg)
        return val_avg

    @torch.no_grad()
    def _validate_precomputed_batch(self, batch, data_wait_s=None):
        """Validate deterministic denoisers on normalized precomputed latent pairs."""
        profile = self._new_profile("precomputed_val_batch")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        if self._is_dit_mse() and self._uses_dit_waveform_spectral_objective():
            z_cond = self._precomputed_condition_to_device(batch, profile=profile)
            z_denoised = self.denoiser(z_cond)
            self._profile_mark(profile, "denoiser_forward")
            target_audio = self._dit_ground_truth_for_batch(batch, profile=profile)
            pred_audio = self._decode_dit_prediction_for_loss(
                z_denoised, profile=profile
            )
            loss = self._dit_waveform_objective_loss(
                pred_audio, target_audio
            )
            stats = self._batch_mean_loss_stats(loss, z_cond.shape[0])
            self._profile_mark(profile, "waveform_spectral_loss")
        else:
            z_cond, z_clean = self._precomputed_latents_to_device(
                batch, profile=profile
            )
            z_denoised = self.denoiser(z_cond)
            self._profile_mark(profile, "denoiser_forward")
            stats = self._mse_stats(z_denoised, z_clean)
            self._profile_mark(profile, "latent_mse_loss")
        self._profile_print("precomputed_val_batch", profile, stats["loss"])
        return stats

    @torch.no_grad()
    def _validate_dit_mse_batch(self, clean, data_wait_s=None):
        """Validate real-time deterministic DiT with its selected objective."""
        profile = self._new_profile("dit_mse_val_batch")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        z_cond, z_clean = self._encode_ddpm_pair(clean, profile=profile)
        z_denoised = self.denoiser(z_cond)
        self._profile_mark(profile, "denoiser_forward")
        if self._uses_dit_waveform_spectral_objective():
            target_audio = clean.to(self.device, non_blocking=True)
            pred_audio = self._decode_dit_prediction_for_loss(
                z_denoised, profile=profile
            )
            loss = self._dit_waveform_objective_loss(pred_audio, target_audio)
            stats = self._batch_mean_loss_stats(loss, z_cond.shape[0])
            self._profile_mark(profile, "waveform_spectral_loss")
        else:
            stats = self._mse_stats(z_denoised, z_clean)
            self._profile_mark(profile, "latent_mse_loss")
        self._profile_print("dit_mse_val_batch", profile, stats["loss"])
        return stats

    @torch.no_grad()
    def _validate_waveform_batch(self, clean, data_wait_s=None):
        """Validate TCN waveform restoration."""
        profile = self._new_profile("waveform_val_batch")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        clean = clean.to(self.device, non_blocking=True)
        corrupt = self._corrupt_batch(clean)
        self._profile_mark(profile, "waveform_corruption")
        pred = self.denoiser(corrupt)
        self._profile_mark(profile, "denoiser_forward")
        loss = waveform_mse_loss(pred, clean)
        stats = self._batch_mean_loss_stats(loss, clean.shape[0])
        self._profile_mark(profile, "loss_compute")
        self._profile_print("waveform_val_batch", profile, stats["loss"])
        return stats

    @torch.no_grad()
    def _validate_waveform_precomputed_batch(self, batch, data_wait_s=None):
        """Validate TCN on precomputed waveform pairs."""
        profile = self._new_profile("waveform_precomputed_val_batch")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        x_audio, y_audio = self._precomputed_waveforms_to_device(batch, profile=profile)
        pred = self.denoiser(x_audio)
        self._profile_mark(profile, "denoiser_forward")
        loss = waveform_mse_loss(pred, y_audio)
        stats = self._batch_mean_loss_stats(loss, x_audio.shape[0])
        self._profile_mark(profile, "loss_compute")
        self._profile_print("waveform_precomputed_val_batch", profile, stats["loss"])
        return stats

    @torch.no_grad()
    def _validate_stft_batch(self, clean, data_wait_s=None):
        """Validate Conformer/U-Net normalized complex-STFT restoration."""
        profile = self._new_profile("stft_val_batch")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        clean = clean.to(self.device, non_blocking=True)
        corrupt = self._corrupt_batch(clean)
        self._profile_mark(profile, "waveform_corruption")
        x_stft = normalize_stft(
            audio_to_stft_channels(corrupt, self.spectral_cfg),
            self.stft_mean,
            self.stft_std,
        )
        y_stft = normalize_stft(
            audio_to_stft_channels(clean, self.spectral_cfg),
            self.stft_mean,
            self.stft_std,
        )
        self._profile_mark(profile, "stft_transform_normalize")
        pred_stft = self.denoiser(x_stft)
        self._profile_mark(profile, "denoiser_forward")
        loss = complex_stft_mse_loss(pred_stft, y_stft)
        stats = self._batch_mean_loss_stats(loss, clean.shape[0])
        self._profile_mark(profile, "loss_compute")
        self._profile_print("stft_val_batch", profile, stats["loss"])
        return stats

    @torch.no_grad()
    def _validate_stft_precomputed_batch(self, batch, data_wait_s=None):
        """Validate Conformer/U-Net on precomputed normalized STFT pairs."""
        profile = self._new_profile("stft_precomputed_val_batch")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        x_stft, y_stft = self._precomputed_stfts_to_device(batch, profile=profile)
        pred_stft = self.denoiser(x_stft)
        self._profile_mark(profile, "denoiser_forward")
        loss = complex_stft_mse_loss(pred_stft, y_stft)
        stats = self._batch_mean_loss_stats(loss, x_stft.shape[0])
        self._profile_mark(profile, "loss_compute")
        self._profile_print("stft_precomputed_val_batch", profile, stats["loss"])
        return stats

    @torch.no_grad()
    def _validate_stft_multires_batch(self, clean, data_wait_s=None):
        """Validate an online STFT model with its waveform spectral objective."""
        profile = self._new_profile("stft_multires_val_batch")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        clean = clean.to(self.device, non_blocking=True)
        corrupt = self._corrupt_batch(clean)
        self._profile_mark(profile, "waveform_corruption")
        x_stft = normalize_stft(
            audio_to_stft_channels(corrupt, self.spectral_cfg),
            self.stft_mean,
            self.stft_std,
        )
        self._profile_mark(profile, "stft_transform_normalize")
        pred_stft_norm = self.denoiser(x_stft)
        self._profile_mark(profile, "denoiser_forward")
        pred_stft = unnormalize_stft(
            pred_stft_norm, self.stft_mean, self.stft_std
        )
        pred_audio = stft_channels_to_audio(
            pred_stft,
            length=clean.shape[-1],
            spectral_cfg=self.spectral_cfg,
        )
        self._profile_mark(profile, "stft_denormalize_istft")
        loss = self._stft_waveform_objective_loss(pred_audio, clean)
        stats = self._batch_mean_loss_stats(loss, clean.shape[0])
        self._profile_mark(profile, "waveform_spectral_loss")
        self._profile_print("stft_multires_val_batch", profile, stats["loss"])
        return stats

    @torch.no_grad()
    def _validate_stft_multires_precomputed_batch(self, batch, data_wait_s=None):
        """Validate precomputed STFT pairs with the waveform spectral objective."""
        profile = self._new_profile("stft_multires_precomputed_val_batch")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)
        x_stft, y_stft = self._precomputed_stfts_to_device(batch, profile=profile)
        pred_stft_norm = self.denoiser(x_stft)
        self._profile_mark(profile, "denoiser_forward")
        length = self._precomputed_stft_num_samples(batch)
        pred_audio = stft_channels_to_audio(
            unnormalize_stft(pred_stft_norm, self.stft_mean, self.stft_std),
            length=length,
            spectral_cfg=self.spectral_cfg,
        )
        target_audio = stft_channels_to_audio(
            unnormalize_stft(y_stft, self.stft_mean, self.stft_std),
            length=length,
            spectral_cfg=self.spectral_cfg,
        )
        self._profile_mark(profile, "stft_denormalize_istft")
        loss = self._stft_waveform_objective_loss(pred_audio, target_audio)
        stats = self._batch_mean_loss_stats(loss, x_stft.shape[0])
        self._profile_mark(profile, "waveform_spectral_loss")
        self._profile_print(
            "stft_multires_precomputed_val_batch", profile, stats["loss"]
        )
        return stats

    @torch.no_grad()
    def _validate_ddpm_batch(self, clean, data_wait_s=None):
        """Validate DDPM epsilon prediction with raw and EMA weights.

        raw loss uses self.denoiser, the current epsilon function being
        backpropagated during training. EMA loss uses the smoothed denoiser that
        is normally preferred for DDPM audio sampling. Both losses share the
        same timestep, epsilon target, and noised latent z_t for a fair comparison.
        """
        profile = self._new_profile("ddpm_val_batch")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        # Same objective as training, but no gradient. Build one validation
        # target and evaluate both denoisers against it.
        if self._precomp_fetch():
            z_cond, z_clean = self._precomputed_latents_to_device(clean, profile=profile)
        else:
            z_cond, z_clean = self._encode_ddpm_pair(clean, profile=profile)
        B = z_clean.shape[0]
        dcfg = self._ddpm_cfg()
        # Sample one t per item from the configured diffusion horizon.
        # Same meaning as in training: this picks where we are on the
        # forward noising chain, not how many times training has backpropped.
        t = torch.randint(
            0,
            dcfg["num_train_timesteps"],
            (B,),
            device=z_clean.device,
            dtype=torch.long,
        )
        epsilon = torch.randn_like(z_clean)
        z_t = q_sample(z_clean, t, epsilon, self.ddpm_buffers)
        self._profile_mark(profile, "sample_t_epsilon_qsample")

        # EMA denoiser: this smoothed copy is used for DDPM sampling quality.
        # It may lag behind raw weights early in training, especially with a
        # high ema_decay, so it is logged separately instead of replacing raw.
        ema_model = self.ema_denoiser.module if self.ema_denoiser is not None else self.denoiser

        # Raw/current denoiser: this is the validation curve that should be
        # compared directly against the DDPM epsilon train objective. Split
        # with whether it is conditioned on clean or corrupt latents.
        condition = z_clean if self._is_ddpm_clean_cond() else z_cond
        eps_pred_raw = self.denoiser(z_t, t, condition)
        self._assert_ddpm_shapes(condition, z_clean, z_t, epsilon, eps_pred_raw, t)
        eps_pred_ema = ema_model(z_t, t, condition)
        self._assert_ddpm_shapes(condition, z_clean, z_t, epsilon, eps_pred_ema, t)

        self._profile_mark(profile, "raw_dit_forward")
        raw_stats = self._mse_stats(eps_pred_raw, epsilon)
        self._profile_mark(profile, "raw_loss_compute")


        self._profile_mark(profile, "ema_dit_forward")
        ema_stats = self._mse_stats(eps_pred_ema, epsilon)
        self._profile_mark(profile, "ema_loss_compute")

        self._profile_print("ddpm_val_batch", profile, raw_stats["loss"])
        return {
            "raw": raw_stats["loss"],
            "raw_sse": raw_stats["sse"],
            "raw_numel": raw_stats["numel"],
            "ema": ema_stats["loss"],
            "ema_sse": ema_stats["sse"],
            "ema_numel": ema_stats["numel"],
        }

    @torch.no_grad()
    def _validate_cfm_batch(self, batch, data_wait_s=None):
        """Validate CFM velocity prediction with raw and EMA weights."""
        profile = self._new_profile("cfm_val_batch")
        self._profile_add_external(profile, "data_loader_wait", data_wait_s)

        if self._precomp_fetch():
            z_cond, z_clean = self._precomputed_latents_to_device(
                batch, profile=profile
            )
        else:
            z_cond, z_clean = self._encode_ddpm_pair(batch, profile=profile)
        z_t, t, velocity_target = self._sample_cfm_training_tuple(
            z_clean, timestep_sampler="uniform"
        )
        self._profile_mark(profile, "sample_t_noise_velocity")

        ema_model = self.ema_denoiser.module if self.ema_denoiser is not None else self.denoiser

        velocity_pred_raw = self.denoiser(z_t, t, z_cond)
        self._assert_cfm_shapes(z_cond, z_clean, z_t, velocity_target, velocity_pred_raw, t)
        self._profile_mark(profile, "raw_dit_forward")
        raw_stats = self._mse_stats(velocity_pred_raw, velocity_target)
        self._profile_mark(profile, "raw_loss_compute")

        velocity_pred_ema = ema_model(z_t, t, z_cond)
        self._assert_cfm_shapes(z_cond, z_clean, z_t, velocity_target, velocity_pred_ema, t)
        self._profile_mark(profile, "ema_dit_forward")
        ema_stats = self._mse_stats(velocity_pred_ema, velocity_target)
        self._profile_mark(profile, "ema_loss_compute")

        self._profile_print("cfm_val_batch", profile, raw_stats["loss"])
        return {
            "raw": raw_stats["loss"],
            "raw_sse": raw_stats["sse"],
            "raw_numel": raw_stats["numel"],
            "ema": ema_stats["loss"],
            "ema_sse": ema_stats["sse"],
            "ema_numel": ema_stats["numel"],
        }

    # --------------------------------------------------------------- utils --

    def _corrupt_batch(self, audio):
        """Apply configured waveform degradations to each item in a batch."""
        if self.corruptor is None:
            raise RuntimeError(
                "Waveform corruption is unavailable in precomputed-data mode"
            )
        # AudioCorruptor.corrupt_batch is the optimized path: batched segment
        # FFT filters on audio.device, with per-item random filter parameters.
        # Older corruptors can still fall back to the per-sample __call__ loop.
        sr = self.sample_rate
        if hasattr(self.corruptor, "corrupt_batch"):
            return self.corruptor.corrupt_batch(audio, sr)
        return torch.stack([self.corruptor(a, sr) for a in audio])

    def _is_ddpm(self):
        """Return True when the active denoiser is the DDPM/DiT path."""
        return self.cfg["denoiser"]["type"] in {"ddpm", "ddpm_8m"}

    def _is_ddpm_clean_cond(self):
        """Return True for DDPM trained with clean latent conditioning."""
        return self.cfg["denoiser"]["type"] == "ddpm_clean_cond"

    def _is_cfm(self):
        """Return True when the active denoiser is the CFM/DiT velocity path."""
        return self.cfg["denoiser"]["type"] in {
            "cfm", "cfm_4m", "cfm_8m", "cfm_16m", "cfm_40m", "cfm_80m"
        }

    def _is_dit_mse(self):
        """Return True for deterministic DiT latent-MSE restoration."""
        return self.cfg["denoiser"]["type"] in DIT_MSE_DENOISER_TYPES

    def _is_stft_deterministic(self):
        """Return True for deterministic complex-STFT predictors."""
        return self.cfg["denoiser"]["type"] in {
            "conformer", "conformer_8m", "unet", "unet1d"
        }

    def _is_waveform_deterministic(self):
        """Return True for deterministic raw-waveform predictors."""
        return self.cfg["denoiser"]["type"] in {"tcn", "tcn_8m"}

    def _uses_codec(self):
        """Return True for denoisers that operate in frozen codec latent space."""
        return not (self._is_stft_deterministic() or self._is_waveform_deterministic())

    def _is_same_latent_generative(self):
        """Return True for SAME-L generative samplers with EMA validation/audio."""
        return self._is_ddpm() or self._is_ddpm_clean_cond() or self._is_cfm()

    def _precomp_fetch(self):
        """Return True when training should use precomputed latent pairs."""
        return bool(self.cfg.get("precompute", {}).get("enabled", False))

    def _precompute_cfg(self):
        """Resolve precomputed train/validation latent-pair folders."""
        pcfg = dict(self.cfg.get("precompute", {}))
        required = ("train_dir", "validate_dir", "ground_truth_dir")
        missing = [key for key in required if not pcfg.get(key)]
        if self._precomp_fetch() and missing:
            raise ValueError(
                "precompute.enabled=true requires explicit config paths for: "
                + ", ".join(f"precompute.{key}" for key in missing)
            )
        return pcfg

    def _load_precomputed_latent_stats(self):
        """Load latent normalization stats saved beside precomputed pairs."""
        metadata = getattr(self.train_set, "metadata", {})
        if metadata.get("latent_mean") is None or metadata.get("latent_std") is None:
            print(
                "Warning: precomputed metadata has no latent_mean/latent_std; "
                "audio logging cannot correctly decode normalized latents."
            )
            return

        mean = torch.as_tensor(metadata["latent_mean"], dtype=torch.float32)
        std = torch.as_tensor(metadata["latent_std"], dtype=torch.float32)
        if mean.ndim == 1:
            mean = mean.reshape(1, -1, 1)
        if std.ndim == 1:
            std = std.reshape(1, -1, 1)
        if mean.shape[1] != self.codec.latent_dim or std.shape[1] != self.codec.latent_dim:
            raise ValueError(
                "Precomputed latent stats do not match codec latent dimension: "
                f"mean={tuple(mean.shape)}, std={tuple(std.shape)}, "
                f"codec_latent_dim={self.codec.latent_dim}"
            )

        self.latent_mean = mean.to(self.device)
        self.latent_std = std.to(self.device)

    @torch.no_grad()
    def _load_dit_ground_truth_audio_bank(self):
        """Load clean DiT spectral targets once and keep them on the GPU.

        The latent precompute already stores each clean window as a WAV in the
        ground-truth directory. Keeping this small unique-window bank on the
        training device avoids a second frozen SAME decode on every optimizer
        step and avoids repeated disk reads for the 20 degraded variants of
        each clean window.
        """
        if not (
            self._is_dit_mse()
            and self._uses_dit_waveform_spectral_objective()
        ):
            return
        # Online training already has the clean waveform batch in memory, so
        # it needs no preloaded ground-truth bank. It still requires the
        # differentiable SAME decoder for the predicted latent.
        if not self._precomp_fetch():
            if not hasattr(self.codec, "decode_for_loss"):
                raise TypeError(
                    "The configured codec does not provide differentiable decode_for_loss"
                )
            return
        if self.latent_mean is None or self.latent_std is None:
            raise ValueError(
                "Decoded-waveform DiT objectives require latent_mean/latent_std "
                "in the precomputed metadata"
            )
        if not hasattr(self.codec, "decode_for_loss"):
            raise TypeError(
                "The configured codec does not provide differentiable decode_for_loss"
            )

        ground_truth_dir = self._precompute_cfg()["ground_truth_dir"]
        indexed_paths = []
        for filename in os.listdir(ground_truth_dir):
            match = re.fullmatch(r"window_(\d+)\.wav", filename)
            if match is not None:
                indexed_paths.append(
                    (int(match.group(1)), os.path.join(ground_truth_dir, filename))
                )
        if not indexed_paths:
            raise ValueError(
                "DiT waveform spectral objectives require ground-truth WAVs in "
                f"{ground_truth_dir}"
            )

        indexed_paths.sort()

        waveforms = []
        expected_length = None
        for window_idx, path in indexed_paths:
            audio, sample_rate = torchaudio.load(path)
            if sample_rate != self.sample_rate:
                raise ValueError(
                    f"Ground-truth {path} uses {sample_rate} Hz; "
                    f"expected {self.sample_rate} Hz"
                )
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if expected_length is None:
                expected_length = audio.shape[-1]
            elif audio.shape[-1] != expected_length:
                raise ValueError(
                    "All DiT ground-truth windows must share one length; "
                    f"window {window_idx} has {audio.shape[-1]}, "
                    f"expected {expected_length}"
                )
            waveforms.append(audio.float())

        self._dit_ground_truth_audio = torch.stack(waveforms, dim=0).to(
            self.device, non_blocking=True
        )
        # hopTenth validation uses sparse source-window IDs. Map those IDs to
        # compact bank rows on the GPU instead of allocating empty waveforms up
        # to the largest source index.
        lookup = torch.full(
            (indexed_paths[-1][0] + 1,), -1, dtype=torch.long
        )
        for bank_row, (window_idx, _) in enumerate(indexed_paths):
            lookup[window_idx] = bank_row
        self._dit_ground_truth_index_lookup = lookup.to(
            self.device, non_blocking=True
        )
        bank_mib = (
            self._dit_ground_truth_audio.numel()
            * self._dit_ground_truth_audio.element_size()
            / (1024 ** 2)
        )
        print(
            "DiT spectral target bank: "
            f"{len(waveforms)} mono windows on {self.device} "
            f"({bank_mib:.1f} MiB)"
        )

    def _load_precomputed_stft_stats(self):
        """Load STFT normalization stats saved beside precomputed STFT pairs."""
        metadata = getattr(self.train_set, "metadata", {})
        if metadata.get("stft_mean") is None or metadata.get("stft_std") is None:
            raise ValueError(
                "Precomputed STFT metadata must include stft_mean/stft_std. "
                "Rerun STFT precompute for conformer/unet training."
            )
        self.stft_mean = torch.as_tensor(metadata["stft_mean"], dtype=torch.float32).to(
            self.device
        )
        self.stft_std = torch.as_tensor(metadata["stft_std"], dtype=torch.float32).to(
            self.device
        )
        if metadata.get("sample_rate") is not None:
            self.sample_rate = int(metadata["sample_rate"])
        self.spectral_cfg.update(metadata.get("spectral", {}))

    def _load_precomputed_waveform_metadata(self):
        """Load sample-rate metadata saved beside precomputed waveform pairs."""
        metadata = getattr(self.train_set, "metadata", {})
        if metadata.get("sample_rate") is not None:
            self.sample_rate = int(metadata["sample_rate"])

    @torch.no_grad()
    def _ensure_stft_stats(self):
        """Compute clean-only STFT mean/std for online STFT-domain training."""
        if not self._is_stft_deterministic() or self.stft_mean is not None:
            return
        if self._precomp_fetch():
            raise ValueError("STFT precompute did not provide stft_mean/stft_std")

        print("Computing clean STFT mean/std from training set ...")
        self.stft_mean, self.stft_std = compute_stft_channel_stats(
            self.train_loader,
            self.spectral_cfg,
            self.device,
        )
        self.stft_mean = self.stft_mean.to(self.device)
        self.stft_std = self.stft_std.to(self.device)
        print("STFT stats ready.")

    def _precomputed_latents_to_device(self, batch, profile=None):
        """Move a precomputed latent-pair batch to the trainer device/dtype."""
        model_dtype = next(self.denoiser.parameters()).dtype
        z_cond = batch["z_cond"].to(
            self.device, dtype=model_dtype, non_blocking=True
        )
        z_clean = batch["z_clean"].to(
            self.device, dtype=model_dtype, non_blocking=True
        )
        self._profile_add_device(profile, "z_cond", z_cond)
        self._profile_add_device(profile, "z_clean", z_clean)
        self._profile_mark(profile, "precomputed_latents_to_device")
        return z_cond, z_clean

    def _precomputed_condition_to_device(self, batch, profile=None):
        """Move only z_cond for decoded-waveform DiT objectives.

        z_clean remains on CPU because the target waveform is retrieved from
        the GPU-resident ground-truth bank.
        """
        model_dtype = next(self.denoiser.parameters()).dtype
        z_cond = batch["z_cond"].to(
            self.device, dtype=model_dtype, non_blocking=True
        )
        self._profile_add_device(profile, "z_cond", z_cond)
        self._profile_mark(profile, "precomputed_condition_to_device")
        return z_cond

    def _dit_ground_truth_for_batch(self, batch, profile=None):
        """Index clean waveform targets directly on the training device."""
        if self._dit_ground_truth_audio is None:
            raise RuntimeError("DiT ground-truth waveform bank is not loaded")
        window_indices = batch["window_idx"].to(
            self.device, dtype=torch.long, non_blocking=True
        )
        # Filenames are validated by PrecomputedLatentPairDataset, and
        # the compact lookup contains every train/validation window. Avoiding
        # an explicit GPU min/max check prevents a per-step host sync.
        bank_rows = self._dit_ground_truth_index_lookup.index_select(
            0, window_indices
        )
        target_audio = self._dit_ground_truth_audio.index_select(
            0, bank_rows
        )
        self._profile_add_device(profile, "target_audio", target_audio)
        self._profile_mark(profile, "ground_truth_gpu_lookup")
        return target_audio

    def _decode_dit_prediction_for_loss(self, z_denoised_normalized, profile=None):
        """Unnormalize and decode one DiT prediction with latent gradients."""
        if self.latent_mean is None or self.latent_std is None:
            raise ValueError(
                "Decoded-waveform DiT objectives require latent_mean/latent_std "
                "in precomputed metadata"
            )
        z_denoised = unnormalize_latent(
            z_denoised_normalized, self.latent_mean, self.latent_std
        )
        audio = self.codec.decode_for_loss(z_denoised).float()
        self._profile_add_device(profile, "pred_audio", audio)
        self._profile_mark(profile, "same_decode_prediction")
        return audio

    def _precomputed_stfts_to_device(self, batch, profile=None):
        """Move a precomputed normalized STFT-pair batch to the trainer device."""
        model_dtype = next(self.denoiser.parameters()).dtype
        x_stft = batch["x_stft"].to(self.device, dtype=model_dtype, non_blocking=True)
        y_stft = batch["y_stft"].to(self.device, dtype=model_dtype, non_blocking=True)
        self._profile_add_device(profile, "x_stft", x_stft)
        self._profile_add_device(profile, "y_stft", y_stft)
        self._profile_mark(profile, "precomputed_stfts_to_device")
        return x_stft, y_stft

    def _precomputed_stft_num_samples(self, batch):
        """Return one valid waveform length for a fixed-length STFT batch."""
        if "num_samples" not in batch:
            raise ValueError(
                "Multi-resolution STFT training requires num_samples in each "
                "precomputed STFT pair."
            )
        lengths = torch.as_tensor(batch["num_samples"]).reshape(-1)
        if lengths.numel() == 0:
            raise ValueError("Precomputed STFT batch has no waveform lengths")
        length = int(lengths[0].item())
        if length <= 0:
            raise ValueError(f"Expected a positive waveform length, got {length}")
        if not torch.all(lengths == length):
            raise ValueError(
                "Multi-resolution STFT batches must have one shared waveform "
                f"length; got {lengths.tolist()}"
            )
        return length

    def _precomputed_waveforms_to_device(self, batch, profile=None):
        """Move a precomputed waveform-pair batch to the trainer device."""
        model_dtype = next(self.denoiser.parameters()).dtype
        x_audio = batch["x_audio"].to(self.device, dtype=model_dtype, non_blocking=True)
        y_audio = batch["y_audio"].to(self.device, dtype=model_dtype, non_blocking=True)
        self._profile_add_device(profile, "x_audio", x_audio)
        self._profile_add_device(profile, "y_audio", y_audio)
        self._profile_mark(profile, "precomputed_waveforms_to_device")
        return x_audio, y_audio

    def _precomputed_validation_audio_windows(self):
        """Number of distinct validation windows to log from precomputed data."""
        requested = self._precomputed_validation_audio_window_indices()
        if requested is not None:
            return len(requested)
        return max(
            1,
            int(self.cfg.get("logging", {}).get("num_precomputed_audio_windows", 3)),
        )

    def _precomputed_validation_audio_window_indices(self):
        """Return an optional explicit, ordered validation-window selection."""
        raw = self.cfg.get("logging", {}).get(
            "precomputed_audio_window_indices"
        )
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError(
                "logging.precomputed_audio_window_indices must be a non-empty list"
            )
        indices = [int(value) for value in raw]
        if len(indices) != len(set(indices)):
            raise ValueError(
                "logging.precomputed_audio_window_indices contains duplicates"
            )
        configured_count = int(
            self.cfg.get("logging", {}).get(
                "num_precomputed_audio_windows", len(indices)
            )
        )
        if configured_count != len(indices):
            raise ValueError(
                "logging.num_precomputed_audio_windows must match the number "
                "of explicit precomputed_audio_window_indices"
            )
        return indices

    def _collect_precomputed_audio_windows(self, selected, batch, max_windows):
        """Collect one item for each distinct window_idx from a precomputed batch."""
        if batch is None or max_windows <= 0:
            return selected

        requested = self._precomputed_validation_audio_window_indices()
        requested_set = None if requested is None else set(requested)

        value_keys = [
            key for key in ("z_cond", "z_clean", "x_stft", "y_stft", "x_audio", "y_audio", "num_samples")
            if key in batch
        ]
        buffers = {key: [] for key in value_keys}
        buffers.update({"window_idx": [], "degradation_idx": []})
        seen = set()
        if selected is not None:
            keep = [
                i
                for i, idx in enumerate(selected["window_idx"])
                if requested_set is None or int(idx.item()) in requested_set
            ]
            for key in buffers:
                buffers[key].extend([selected[key][i] for i in keep])
            seen.update(
                int(selected["window_idx"][i].item()) for i in keep
            )
            if len(seen) >= max_windows:
                return {
                    key: torch.stack(value, dim=0)
                    for key, value in buffers.items()
                }

        for i in range(batch["window_idx"].shape[0]):
            window_idx = int(batch["window_idx"][i].item())
            if requested_set is not None and window_idx not in requested_set:
                continue
            if window_idx in seen:
                continue
            seen.add(window_idx)
            for key in value_keys:
                buffers[key].append(batch[key][i].detach().cpu())
            buffers["window_idx"].append(batch["window_idx"][i].detach().cpu())
            buffers["degradation_idx"].append(batch["degradation_idx"][i].detach().cpu())
            if len(seen) >= max_windows:
                break

        if not buffers["window_idx"]:
            return None if requested_set is not None else selected
        return {key: torch.stack(value, dim=0) for key, value in buffers.items()}

    def _merge_precomputed_validation_audio_examples(self, rank_examples):
        """Merge small per-rank preview caches into one deterministic rank-zero cache."""
        requested = self._precomputed_validation_audio_window_indices()
        requested_order = (
            {} if requested is None else {value: i for i, value in enumerate(requested)}
        )
        rows = []
        for examples in rank_examples:
            if not examples or "window_idx" not in examples:
                continue
            for row in range(int(examples["window_idx"].shape[0])):
                window_idx = int(examples["window_idx"][row].item())
                degradation_idx = int(examples["degradation_idx"][row].item())
                rows.append((window_idx, degradation_idx, examples, row))
        if not rows:
            return None

        if requested is None:
            rows.sort(key=lambda item: (item[0], item[1]))
        else:
            rows.sort(
                key=lambda item: (
                    requested_order.get(item[0], len(requested_order)),
                    item[1],
                )
            )

        selected_rows = []
        seen = set()
        for window_idx, _, examples, row in rows:
            if requested is not None and window_idx not in requested_order:
                continue
            if window_idx in seen:
                continue
            seen.add(window_idx)
            selected_rows.append((examples, row))
            if len(selected_rows) >= self._precomputed_validation_audio_windows():
                break

        if not selected_rows:
            return None
        keys = selected_rows[0][0].keys()
        return {
            key: torch.stack([examples[key][row] for examples, row in selected_rows])
            for key in keys
        }

    def _gather_precomputed_validation_audio_examples(self):
        """Gather only the tiny fixed preview cache; validation tensors stay sharded."""
        if not self._precomp_fetch() or not getattr(self, "distributed", False):
            return
        gathered = [None] * self.world_size if self.is_primary else None
        dist.gather_object(
            self._precomputed_val_audio_examples,
            object_gather_list=gathered,
            dst=0,
        )
        if self.is_primary:
            self._precomputed_val_audio_examples = (
                self._merge_precomputed_validation_audio_examples(gathered)
            )

    def _mse_stats(self, pred, target):
        """Return MSE plus GPU-computed SSE/element count for weighted averaging."""
        diff = pred - target.to(pred.device, dtype=pred.dtype)
        sse = diff.square().sum()
        numel = diff.numel()
        loss = sse / max(numel, 1)
        return {
            "loss": float(loss.detach().cpu()),
            "sse": float(sse.detach().cpu()),
            "numel": numel,
        }

    def _batch_mean_loss_stats(self, loss, batch_size):
        """Represent a per-file averaged batch loss for epoch aggregation."""
        weighted = loss.detach() * max(int(batch_size), 1)
        return {
            "loss": float(loss.detach().cpu()),
            "sse": float(weighted.cpu()),
            "numel": max(int(batch_size), 1),
        }

    def _maybe_log_best_total_validation_audio(self, val_loss, log_audio=False):
        """Log fixed-tag audio when the full validation loss reaches a new low.

        Only called at audio-logging steps (log_audio_every_steps boundaries)
        so the best-val check never triggers more audio logs than intended.
        """
        if not log_audio or self._latest_val_audio_batch is None:
            return
        val_loss = float(val_loss)
        if val_loss != val_loss:
            return
        if (
            self._best_total_val_audio_loss is not None
            and val_loss >= self._best_total_val_audio_loss
        ):
            return

        self._best_total_val_audio_loss = val_loss
        self._best_total_val_audio_step = self.global_step
        tag = "val_best_total"
        self.exp.log_scalar(f"{tag}/loss", val_loss, self.global_step)
        self.exp.log_scalar(f"{tag}/global_step", self.global_step, self.global_step)
        self._log_best_total_audio_metadata(tag, self._latest_val_audio_batch, val_loss)

        if self._precomp_fetch():
            self._log_precomputed_audio(
                tag,
                self._latest_val_audio_batch,
                fixed_audio_tags=True,
            )
        else:
            self._log_audio(tag, self._latest_val_audio_batch, fixed_audio_tags=True)

    def _log_best_total_audio_metadata(self, tag, batch, val_loss):
        """Write enough metadata to identify fixed-tag best validation audio."""
        lines = [
            f"global_step: {self.global_step}",
            f"validation_loss: {val_loss:.10g}",
        ]
        if isinstance(batch, dict) and "window_idx" in batch:
            n = int(batch["window_idx"].shape[0])
            for i in range(n):
                window_idx = int(batch["window_idx"][i].item())
                lines.append(f"slot_{i}_window_idx: {window_idx}")
                self.exp.log_scalar(f"{tag}/slot_{i}_window_idx", window_idx, self.global_step)
                if "degradation_idx" in batch:
                    degradation_idx = int(batch["degradation_idx"][i].item())
                    lines.append(f"slot_{i}_degradation_idx: {degradation_idx}")
                    self.exp.log_scalar(
                        f"{tag}/slot_{i}_degradation_idx",
                        degradation_idx,
                        self.global_step,
                    )
        self.exp.writer.add_text(tag + "/metadata", "\n".join(lines), self.global_step)

    def _inject_canonical_validation_windows(self) -> None:
        """Populate validation-audio IDs from selection metadata or the registry.

        If the active config already has an explicit window list, it wins and
        this method does nothing.  Otherwise, the basename of
        ``precompute.train_dir`` is first checked for a provenance-bound
        ``VALIDATION_AUDIO_WINDOWS.json``. Legacy datasets then fall back to
        ``_DATASET_VALIDATION_AUDIO_WINDOWS``. The resolved IDs are injected
        before the config snapshot is written by ``_build_logging()``.

        This is a no-op for online (non-precomputed) training.
        """
        if not self.cfg.get("precompute", {}).get("enabled", False):
            return
        logging_cfg = self.cfg.setdefault("logging", {})
        if logging_cfg.get("precomputed_audio_window_indices") is not None:
            return  # explicit config wins
        train_dir = self.cfg.get("precompute", {}).get("train_dir", "")
        selection_path = os.path.join(train_dir, "VALIDATION_AUDIO_WINDOWS.json")
        if os.path.isfile(selection_path):
            with open(selection_path, encoding="utf-8") as file:
                selection = json.load(file)
            selected = selection.get("global_window_indices")
            if (
                not isinstance(selected, list)
                or not selected
                or any(not isinstance(value, int) or isinstance(value, bool) for value in selected)
                or len(selected) != len(set(selected))
            ):
                raise ValueError(
                    "Invalid global_window_indices in validation-audio "
                    f"selection: {selection_path}"
                )
            logging_cfg["precomputed_audio_window_indices"] = list(selected)
            logging_cfg["num_precomputed_audio_windows"] = len(selected)
            print(
                f"[trainer] Loaded {len(selected)} provenance-bound validation "
                f"audio windows from '{selection_path}'.",
                flush=True,
            )
            return
        basename = os.path.basename(train_dir.rstrip("/"))
        canonical = _DATASET_VALIDATION_AUDIO_WINDOWS.get(basename)
        if canonical is None:
            return
        logging_cfg["precomputed_audio_window_indices"] = list(canonical)
        n = len(canonical)
        if logging_cfg.get("num_precomputed_audio_windows", n) != n:
            logging_cfg["num_precomputed_audio_windows"] = n
        print(
            f"[trainer] Injected {n} canonical validation audio windows for "
            f"dataset '{basename}'.",
            flush=True,
        )

    def _validate_denoiser_mode(self):
        """Keep generative denoiser modes explicit and on supported data paths."""
        if self._is_ddpm() and self._is_ddpm_clean_cond():
            raise ValueError(
                "denoiser.type cannot be both 'ddpm' and 'ddpm_clean_cond'."
            )
        if self._is_cfm():
            self._cfm_cfg_dropout_prob()
            cfg_scale = float(self._cfm_cfg().get("cfg_scale", 1.0))
            if not math.isfinite(cfg_scale):
                raise ValueError(
                    f"CFM cfg_scale must be finite, got {cfg_scale!r}"
                )
            # Validate the optional TensorBoard guidance sweep during startup,
            # rather than failing for the first time at an audio-log boundary.
            self._cfm_audio_cfg_scales("val")
        if self._is_dit_mse():
            objective = self._dit_mse_objective()
            allowed = {
                "latent_mse",
                "auralossMRSTFT",
                "multiscale_spectral_loss_DDSP",
            }
            if objective not in allowed:
                raise ValueError(
                    "loss.dit_mse_objective must be one of "
                    f"{sorted(allowed)}, got {objective!r}"
                )
            print(f"Using DiT-MSE with {objective} objective.")
        if self._is_stft_deterministic():
            objective = self._stft_objective()
            allowed = {
                "complex_stft_mse",
                "multires_stft_mse",
                "auralossMRSTFT",
                "multiscale_spectral_loss_DDSP",
            }
            if objective not in allowed:
                raise ValueError(
                    "loss.stft_objective must be one of "
                    f"{sorted(allowed)}, got {objective!r}"
                )
            print(
                "Using complex-STFT domain for conformer/unet "
                f"with {objective} objective."
            )
        if self._is_waveform_deterministic():
            print("Using raw waveform domain for TCN.")
        if self._is_dit_mse():
            print("Using deterministic SAME latent DiT-MSE path.")

    def _ddpm_cfg(self):
        """Return the config block for the active DDPM capacity."""
        key = "ddpm_8m" if self.cfg["denoiser"]["type"] == "ddpm_8m" else "ddpm"
        return self.cfg["denoiser"][key]

    def _cfm_cfg(self):
        """Return the config block for the active CFM capacity."""
        return self.cfg["denoiser"][self.cfg["denoiser"]["type"]]

    def _cfm_cfg_dropout_prob(self):
        """Return the validated probability of null conditioning in CFM training."""
        value = self._cfm_cfg().get("cfg_dropout_prob", 0.0)
        if isinstance(value, bool):
            raise ValueError("CFM cfg_dropout_prob must be a probability, not bool")
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"CFM cfg_dropout_prob must be numeric, got {value!r}"
            ) from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                "CFM cfg_dropout_prob must be finite and within [0, 1], "
                f"got {probability!r}"
            )
        return probability

    def _cfm_audio_cfg_scales(self, tag):
        """Return an opt-in CFG scale sweep for validation audio, else ``None``."""
        if not tag.startswith("val"):
            return None
        raw_scales = self._cfm_cfg().get("cfg_scales_eval")
        if raw_scales is None:
            return None
        if not isinstance(raw_scales, (list, tuple)) or not raw_scales:
            raise ValueError("CFM cfg_scales_eval must be a non-empty list")

        scales = []
        for value in raw_scales:
            if isinstance(value, bool):
                raise ValueError("CFM cfg_scales_eval values must be numeric, not bool")
            try:
                scale = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CFM cfg_scales_eval values must be numeric, got {value!r}"
                ) from exc
            if not math.isfinite(scale):
                raise ValueError(
                    f"CFM cfg_scales_eval values must be finite, got {scale!r}"
                )
            if not any(abs(scale - existing) < 1e-8 for existing in scales):
                scales.append(scale)
        return scales

    def _dit_mse_cfg(self):
        """Return the config block for the active deterministic DiT capacity."""
        denoiser_type = self.cfg["denoiser"]["type"]
        if denoiser_type not in DIT_MSE_DENOISER_TYPES:
            raise ValueError(
                f"Active denoiser {denoiser_type!r} is not a deterministic DiT"
            )
        return self.cfg["denoiser"][denoiser_type]

    def _dit_mse_objective(self):
        """Select the deterministic DiT objective; preserve latent MSE by default."""
        return self.cfg.get("loss", {}).get(
            "dit_mse_objective", "latent_mse"
        )

    def _uses_dit_waveform_spectral_objective(self):
        """Return True when DiT predictions must be decoded before loss."""
        return self._dit_mse_objective() in {
            "auralossMRSTFT",
            "multiscale_spectral_loss_DDSP",
        }

    def _dit_waveform_objective_loss(self, pred_audio, target_audio):
        """Apply the selected decoded-waveform loss for deterministic DiT."""
        pred_audio, target_audio = prepare_audio_pair(pred_audio, target_audio)
        objective = self._dit_mse_objective()
        if objective == "auralossMRSTFT":
            return auralossMRSTFT(
                pred_audio, target_audio, self.spectral_cfg
            )
        if objective == "multiscale_spectral_loss_DDSP":
            return multiscale_spectral_loss_DDSP(pred_audio, target_audio)
        raise ValueError(
            f"{objective!r} is not a decoded-waveform DiT objective"
        )

    def _stft_objective(self):
        """Select the STFT-model objective without changing legacy MSE behavior."""
        return self.cfg.get("loss", {}).get(
            "stft_objective", "complex_stft_mse"
        )

    def _uses_multires_stft_objective(self):
        """Return True for objectives computed after differentiable ISTFT."""
        return self._stft_objective() in {
            "multires_stft_mse",
            "auralossMRSTFT",
            "multiscale_spectral_loss_DDSP",
        }

    def _stft_waveform_objective_loss(self, pred_audio, target_audio):
        """Dispatch the selected waveform spectral loss for STFT denoisers."""
        objective = self._stft_objective()
        if objective == "multires_stft_mse":
            return multi_resolution_stft_mse_loss(
                pred_audio, target_audio, self.spectral_cfg
            )
        if objective == "auralossMRSTFT":
            return auralossMRSTFT(
                pred_audio, target_audio, self.spectral_cfg
            )
        if objective == "multiscale_spectral_loss_DDSP":
            return multiscale_spectral_loss_DDSP(pred_audio, target_audio)
        raise ValueError(
            f"{objective!r} is not a waveform spectral STFT objective"
        )

    def _sample_cfm_training_tuple(self, z_clean, timestep_sampler=None):
        """Sample CFM straight-path state and velocity target.

        CFM starts from Gaussian noise z0, not from the degraded latent. The
        degraded latent is only the conditioning signal. The linear path is
        z_t = (1 - t) * z0 + t * z_clean with target velocity z_clean - z0.

        Training uses the active CFM config's ``timestep_sampler``. ``uniform``
        preserves the original behavior; ``logit_normal`` follows Stable
        Audio 3 exactly with t = sigmoid(N(0, 1)). Callers may explicitly
        select a sampler, which validation uses to retain the original uniform
        comparison distribution across CFM training-sampler ablations.
        """
        B = z_clean.shape[0]
        z0 = torch.randn_like(z_clean)
        sampler = (
            self._cfm_cfg().get("timestep_sampler", "uniform")
            if timestep_sampler is None
            else timestep_sampler
        )
        if sampler == "uniform":
            t = torch.rand((B,), device=z_clean.device, dtype=z_clean.dtype)
        elif sampler == "logit_normal":
            t = torch.sigmoid(
                torch.randn((B,), device=z_clean.device, dtype=z_clean.dtype)
            )
        else:
            raise ValueError(
                "CFM timestep_sampler must be 'uniform' or 'logit_normal', "
                f"got {sampler!r}"
            )
        t_view = t.view(B, 1, 1)
        z_t = (1.0 - t_view) * z0 + t_view * z_clean
        velocity_target = z_clean - z0
        return z_t, t, velocity_target

    def _apply_cfm_cfg_dropout(self, z_cond):
        """Drop conditioning latents to zeros for CFM classifier-free guidance."""
        p = self._cfm_cfg_dropout_prob()
        if p <= 0.0:
            return z_cond
        keep = (
            torch.rand((z_cond.shape[0],), device=z_cond.device, dtype=z_cond.dtype)
            >= p
        ).view(-1, 1, 1)
        return z_cond * keep.to(z_cond.dtype)

    def _profile_should_run(self):
        """Return True when this global step should emit timing diagnostics."""
        return self.profile_steps and self.global_step % self.profile_every == 0

    def _profile_sync_device(self):
        """Synchronize CUDA before taking a timestamp when accurate GPU timing matters."""
        if self.profile_cuda_sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _new_profile(self, label):
        """Start a timing record for the current step/logging event.

        The label is accepted for readability at call sites; printing receives
        its own label too, so this record only needs the timestamp list.
        """
        if not self._profile_should_run():
            return None
        self._profile_sync_device()
        return {
            "last": time.perf_counter(),
            "items": [],
            "devices": [],
        }

    def _profile_add_external(self, profile, name, elapsed_s):
        """Add a duration measured outside the profiled block, e.g. DataLoader wait."""
        if profile is None or elapsed_s is None:
            return
        profile["items"].append((name, elapsed_s))

    def _profile_add_device(self, profile, name, value):
        """Attach tensor/device placement metadata to a profile line."""
        if profile is None:
            return
        device = getattr(value, "device", value)
        profile["devices"].append(f"{name}={device}")

    def _profile_mark(self, profile, name):
        """Record elapsed time since the previous profile mark.

        Example: _profile_mark(profile, "sample_t_epsilon_qsample") means
        "store how long the previous stage took, then start timing the next
        stage labeled sample_t_epsilon_qsample".
        """
        if profile is None:
            return
        self._profile_sync_device()
        now = time.perf_counter()
        profile["items"].append((name, now - profile["last"]))
        profile["last"] = now

    def _profile_print(self, label, profile, loss=None):
        """Print one compact timing line with the slowest measured stage called out."""
        if profile is None or not profile["items"]:
            return
        total_s = sum(duration for _, duration in profile["items"])
        slowest_name, slowest_s = max(profile["items"], key=lambda item: item[1])
        parts = " ".join(
            f"{name}={duration * 1000.0:.1f}ms"
            for name, duration in profile["items"]
        )
        loss_part = "" if loss is None else f" loss={loss:.6f}"
        devices_part = (
            "" if not profile.get("devices")
            else " devices=" + ",".join(profile["devices"])
        )
        print(
            f"[profile:{label}] step={self.global_step}{loss_part} "
            f"total={total_s:.3f}s slowest={slowest_name}:{slowest_s:.3f}s "
            f"| {parts}{devices_part}",
            flush=True,
        )

    # -------------------------------------------------------- step triggers --
    # These step-based methods replace the old epoch-based equivalents and
    # are the primary logging cadence control.  The epoch-based methods below
    # are kept for backward compatibility but are no longer called by default.



    def _val_step_interval(self) -> int:
        """How often (in steps) to run a full validation pass.  Default 30 000."""
        return max(1, int(
            self.cfg.get("training", {}).get("val_every_steps", 30_000)
        ))

    def _audio_step_interval(self) -> int:
        """How often (in steps) to log audio and spectrograms.  Default 150 000."""
        return max(1, int(
            self.cfg.get("training", {}).get("log_audio_every_steps", 150_000)
        ))

    def _should_run_val_step(self) -> bool:
        """True at step 1 (initial baseline) and every val_every_steps."""
        every = self._val_step_interval()
        return self.global_step == 1 or self.global_step % every == 0

    def _should_log_audio_step(self) -> bool:
        """True every log_audio_every_steps steps, never at step 1."""
        every = self._audio_step_interval()
        return self.global_step > 1 and self.global_step % every == 0

    # ------------------------------------------------- epoch-based (legacy) --
    # Kept for backward compat; not invoked by the default step-based flow.

    def _should_log_epoch_audio(self, epoch):
        """(Legacy) Return True when this epoch should emit audio/spectrograms."""
        every = int(self.cfg["training"].get("log_audio_every_epochs", 50))
        return every > 0 and (epoch + 1) % every == 0

    def _tensorboard_epoch_interval(self):
        """(Legacy) Epoch interval for aggregate metrics."""
        return max(
            1,
            int(self.cfg["training"].get("tensorboard_every_epochs", 1)),
        )

    def _should_log_epoch_tensorboard(self, epoch):
        """(Legacy) Return True when aggregate train/validation metrics are due."""
        return (epoch + 1) % self._tensorboard_epoch_interval() == 0

    @torch.no_grad()
    def _encode_ddpm_pair(self, clean, profile=None):
        """Corrupt audio, encode clean/corrupt SAME latents, align and normalize them."""
        # Build the paired SAME latents used by DDPM:
        # z_clean is the normalized clean target latent.
        # z_cond is the normalized corrupted conditioning latent.
        clean = clean.to(self.device, non_blocking=True)
        self._profile_add_device(profile, "clean", clean)
        self._profile_mark(profile, "clean_to_device")
        corrupt = self._corrupt_batch(clean)
        self._profile_add_device(profile, "corrupt", corrupt)
        self._profile_mark(profile, "waveform_corruption")

        # Encode clean and corrupt in a single SAME call. This keeps the model
        # boundary identical, but gives SA3 one larger batch to process.
        B = clean.shape[0]
        encode_input = torch.cat([clean, corrupt], dim=0)
        self._profile_add_device(profile, "same_encode_input", encode_input)
        self._profile_mark(profile, "same_encode_input_ready")
        z_both_raw = ensure_channel_first(
            self.codec.encode(encode_input),
            channels=self.codec.latent_dim,
        )

        self._profile_add_device(profile, "same_encode_output", z_both_raw)
        z_clean_raw, z_cond_raw = z_both_raw[:B], z_both_raw[B:]
        self._profile_mark(profile, "same_encode_clean_corrupt_batched_done")

        # SAME may produce slight length differences after preprocessing; crop
        # both latents to the shared valid time length before conditioning.
        z_cond_raw, z_clean_raw, _ = align_latent_pair(z_cond_raw, z_clean_raw)
        z_clean = normalize_latent(z_clean_raw, self.latent_mean, self.latent_std)
        z_cond = normalize_latent(z_cond_raw, self.latent_mean, self.latent_std)
        model_dtype = next(self.denoiser.parameters()).dtype
        self._profile_add_device(profile, "z_cond", z_cond)
        self._profile_add_device(profile, "z_clean", z_clean)
        self._profile_mark(profile, "align_normalize_latents")
        return z_cond.to(model_dtype), z_clean.to(model_dtype)

    def _assert_ddpm_shapes(self, z_cond, z_clean, z_t, epsilon, eps_pred, t):
        """Assert DDPM tensors follow the expected SAME-L [B, 256, T] contract."""
        # Keep shape assumptions loud: DDPM tensors at the denoiser boundary
        # should always be [B, 256, T] for SAME-L.
        assert z_cond.ndim == 3
        assert z_clean.ndim == 3
        assert z_cond.shape[1] == 256
        assert z_clean.shape[1] == 256
        assert z_cond.shape == z_clean.shape
        assert z_t.shape == z_clean.shape
        assert epsilon.shape == z_clean.shape
        assert eps_pred.shape == z_clean.shape
        assert torch.all(t >= 0)
        # t must be a legal diffusion index for this schedule.
        assert torch.all(t < self._ddpm_cfg()["num_train_timesteps"])

    def _assert_cfm_shapes(self, z_cond, z_clean, z_t, velocity_target, velocity_pred, t):
        """Assert CFM tensors follow the expected SAME-L [B, 256, T] contract."""
        assert z_cond.ndim == 3
        assert z_clean.ndim == 3
        assert z_cond.shape[1] == 256
        assert z_clean.shape[1] == 256
        assert z_cond.shape == z_clean.shape
        assert z_t.shape == z_clean.shape
        assert velocity_target.shape == z_clean.shape
        assert velocity_pred.shape == z_clean.shape
        assert t.shape == (z_clean.shape[0],)
        assert torch.all(t >= 0)
        assert torch.all(t <= 1)

    @torch.no_grad()
    def _validation_spectral_audio_values(
        self,
        pred,
        target,
        sample_rate,
        pred_stft=None,
        target_stft=None,
    ):
        """Compute the four validation-audio spectral metrics logged together."""
        values = compute_spectral_mse_values(
            pred,
            target,
            sample_rate,
            spectral_cfg=self.spectral_cfg,
            pred_stft=pred_stft,
            target_stft=target_stft,
        )
        pred_audio, target_audio = prepare_audio_pair(pred, target)
        values["auralossMRSTFT"] = float(
            auralossMRSTFT(
                pred_audio, target_audio, self.spectral_cfg
            ).detach().cpu()
        )
        values["multiscale_spectral_loss_DDSP"] = float(
            multiscale_spectral_loss_DDSP(
                pred_audio, target_audio
            ).detach().cpu()
        )
        return {
            name: values[name]
            for name in (
                "mel_stft_mse",
                "multi_resolution_stft_mse",
                "auralossMRSTFT",
                "multiscale_spectral_loss_DDSP",
            )
        }

    @torch.no_grad()
    def _log_spectral_audio_losses(self, tag, pred, target, sample_rate,
                                   pred_stft=None, target_stft=None):
        """Log validation-only spectral metrics for the audio examples."""
        if tag != "val":
            return
        values = self._validation_spectral_audio_values(
            pred,
            target,
            sample_rate,
            pred_stft=pred_stft,
            target_stft=target_stft,
        )
        for name, value in values.items():
            self.exp.log_scalar(f"spectral_loss/{tag}/{name}", value, self.global_step)

    @torch.no_grad()
    def _log_spectral_audio_loss_pairs(self, tag, pairs, sample_rate):
        """Log validation-only per-file average spectral metrics over audio pairs."""
        if tag != "val" or not pairs:
            return
        totals = None
        for pair in pairs:
            if len(pair) == 2:
                pred, target = pair
                pred_stft = target_stft = None
            else:
                pred, target, pred_stft, target_stft = pair
            values = self._validation_spectral_audio_values(
                pred,
                target,
                sample_rate,
                pred_stft=pred_stft,
                target_stft=target_stft,
            )
            if totals is None:
                totals = dict(values)
            else:
                for name, value in values.items():
                    totals[name] += value
        for name, value in totals.items():
            self.exp.log_scalar(
                f"spectral_loss/{tag}/{name}",
                value / len(pairs),
                self.global_step,
            )

    @torch.no_grad()
    def _log_waveform_audio(self, tag, clean_cpu, fixed_audio_tags=False, log_audio_mse=False):
        """Log waveform-domain TCN audio and spectral metrics."""
        profile = self._new_profile(f"{tag}_waveform_audio_log")
        self.denoiser.eval()
        n = min(self.cfg["logging"]["num_audio_samples"], clean_cpu.shape[0])
        clean_cpu = clean_cpu[:n]
        clean_d = clean_cpu.to(self.device, non_blocking=True)
        corrupt_d = self._corrupt_batch(clean_d)
        self._profile_mark(profile, "waveform_corruption")
        audio_denoised = self.denoiser(corrupt_d)
        self._profile_mark(profile, "denoiser_forward")

        self._log_spectral_audio_losses(tag, audio_denoised, clean_d, self.sample_rate)
        self._profile_mark(profile, "spectral_mse_metrics")

        T = clean_cpu.shape[-1]
        examples_by_name = {
            "clean": clean_cpu,
            "corrupt": corrupt_d.detach().cpu(),
            "denoised": audio_denoised[:, :, :T].detach().cpu(),
        }
        for i in range(n):
            for name, batch_audio in examples_by_name.items():
                audio_tag = f"{tag}/{name}_{i}" if fixed_audio_tags else f"{tag}/{name}_{i}"
                audio = batch_audio[i, :, :T]
                self.exp.log_audio(audio_tag, audio, self.global_step, self.sample_rate)
                self.exp.log_spectrogram(f"spectrogram/{audio_tag}", audio, self.global_step)
        self._profile_mark(profile, "tensorboard_audio_spectrogram_write")
        self._profile_print(f"{tag}_waveform_audio_log", profile)
        self.denoiser.train()

    @torch.no_grad()
    def _log_stft_audio(self, tag, clean_cpu, fixed_audio_tags=False, log_audio_mse=False):
        """Log STFT-domain Conformer/U-Net audio and spectral metrics."""
        profile = self._new_profile(f"{tag}_stft_audio_log")
        self.denoiser.eval()
        n = min(self.cfg["logging"]["num_audio_samples"], clean_cpu.shape[0])
        clean_cpu = clean_cpu[:n]
        clean_d = clean_cpu.to(self.device, non_blocking=True)
        corrupt_d = self._corrupt_batch(clean_d)
        self._profile_mark(profile, "waveform_corruption")

        x_raw = audio_to_stft_channels(corrupt_d, self.spectral_cfg)
        y_raw = audio_to_stft_channels(clean_d, self.spectral_cfg)
        x_norm = normalize_stft(x_raw, self.stft_mean, self.stft_std)
        pred_norm = self.denoiser(x_norm)
        pred_raw = unnormalize_stft(pred_norm, self.stft_mean, self.stft_std)
        self._profile_mark(profile, "stft_forward_unnormalize")

        length = clean_d.shape[-1]
        audio_denoised = stft_channels_to_audio(
            pred_raw,
            length=length,
            spectral_cfg=self.spectral_cfg,
        )
        audio_corrupt = stft_channels_to_audio(
            x_raw,
            length=length,
            spectral_cfg=self.spectral_cfg,
        )
        self._profile_mark(profile, "istft_reconstruct")

        self._log_spectral_audio_losses(
            tag,
            audio_denoised,
            clean_d,
            self.sample_rate,
            pred_stft=pred_raw,
            target_stft=y_raw,
        )
        self._profile_mark(profile, "spectral_mse_metrics")

        T = clean_cpu.shape[-1]
        examples_by_name = {
            "clean": clean_cpu,
            "corrupt": audio_corrupt.detach().cpu(),
            "denoised": audio_denoised[:, :, :T].detach().cpu(),
        }
        for i in range(n):
            for name, batch_audio in examples_by_name.items():
                audio_tag = f"{tag}/{name}_{i}" if fixed_audio_tags else f"{tag}/{name}_{i}"
                audio = batch_audio[i, :, :T]
                self.exp.log_audio(audio_tag, audio, self.global_step, self.sample_rate)
                self.exp.log_spectrogram(f"spectrogram/{audio_tag}", audio, self.global_step)
        self._profile_mark(profile, "tensorboard_audio_spectrogram_write")
        self._profile_print(f"{tag}_stft_audio_log", profile)
        self.denoiser.train()

    @torch.no_grad()
    def _log_audio(self, tag, clean_cpu, fixed_audio_tags=False, log_audio_mse=False):
        """Write clean/corrupt/recon/denoised audio and spectrograms to TensorBoard."""
        if self._is_stft_deterministic():
            return self._log_stft_audio(
                tag,
                clean_cpu,
                fixed_audio_tags=fixed_audio_tags,
                log_audio_mse=log_audio_mse,
            )
        if self._is_waveform_deterministic():
            return self._log_waveform_audio(
                tag,
                clean_cpu,
                fixed_audio_tags=fixed_audio_tags,
                log_audio_mse=log_audio_mse,
            )
        profile = self._new_profile(f"{tag}_audio_log")

        # TensorBoard examples are diagnostic, not part of the objective. For
        # DDPM they are expensive because "denoised" requires a full reverse
        # diffusion chain before SAME decoding.
        self.denoiser.eval()
        if self.ema_denoiser is not None:
            self.ema_denoiser.eval()
        n = min(self.cfg["logging"]["num_audio_samples"], clean_cpu.shape[0])
        clean_cpu = clean_cpu[:n]
        clean_d = clean_cpu.to(self.device, non_blocking=True)
        self._profile_add_device(profile, "clean", clean_d)
        self._profile_mark(profile, "clean_to_device")
        corrupt_d = self._corrupt_batch(clean_d)
        self._profile_add_device(profile, "corrupt", corrupt_d)
        self._profile_mark(profile, "waveform_corruption")
        corrupt_cpu = corrupt_d.detach().cpu()

        B = clean_d.shape[0]
        encode_input = torch.cat([clean_d, corrupt_d], dim=0)
        self._profile_add_device(profile, "same_encode_input", encode_input)
        self._profile_mark(profile, "same_encode_input_ready")
        z_both = self.codec.encode(encode_input)
        self._profile_add_device(profile, "same_encode_output", z_both)
        z_clean, z_corrupt = z_both[:B], z_both[B:]
        self._profile_mark(profile, "same_encode_clean_corrupt_batched_done")
        if self._is_ddpm():
            z_cond = normalize_latent(
                ensure_channel_first(z_corrupt, channels=self.codec.latent_dim),
                self.latent_mean,
                self.latent_std,
            ).to(next(self.denoiser.parameters()).dtype)
            self._profile_add_device(profile, "z_cond", z_cond)
            z_denoised_norm = self._sample_ddpm_latent(
                z_cond,
                num_sampling_steps=self._ddpm_cfg()["num_train_timesteps"],
            )
            self._profile_mark(profile, "ddpm_reverse_sample")
            z_denoised = unnormalize_latent(
                z_denoised_norm, self.latent_mean, self.latent_std
            )
        elif self._is_ddpm_clean_cond():
            z_cond = normalize_latent(
                ensure_channel_first(z_clean, channels=self.codec.latent_dim),
                self.latent_mean,
                self.latent_std,
            ).to(next(self.denoiser.parameters()).dtype)
            self._profile_add_device(profile, "z_cond", z_cond)
            z_denoised_norm = self._sample_ddpm_latent(
                z_cond,
                num_sampling_steps=self._ddpm_cfg()["num_train_timesteps"],
            )
            self._profile_mark(profile, "ddpm_reverse_sample")
            z_denoised = unnormalize_latent(
                z_denoised_norm, self.latent_mean, self.latent_std
            )
        elif self._is_cfm():
            z_cond = normalize_latent(
                ensure_channel_first(z_corrupt, channels=self.codec.latent_dim),
                self.latent_mean,
                self.latent_std,
            ).to(next(self.denoiser.parameters()).dtype)
            self._profile_add_device(profile, "z_cond", z_cond)
            z_denoised_norm = self._sample_cfm_latent(
                z_cond,
                inference_steps=self._cfm_default_inference_steps(),
                cfg_scale=self._cfm_cfg().get("cfg_scale", 1.0),
            )
            self._profile_mark(profile, "cfm_euler_sample")
            z_denoised = unnormalize_latent(
                z_denoised_norm, self.latent_mean, self.latent_std
            )
        elif self._is_dit_mse():
            z_cond = normalize_latent(
                ensure_channel_first(z_corrupt, channels=self.codec.latent_dim),
                self.latent_mean,
                self.latent_std,
            ).to(next(self.denoiser.parameters()).dtype)
            self._profile_add_device(profile, "z_cond", z_cond)
            z_denoised_norm = self.denoiser(z_cond)
            self._profile_mark(profile, "denoiser_forward")
            z_denoised = unnormalize_latent(
                z_denoised_norm, self.latent_mean, self.latent_std
            )
        else:
            z_denoised = self.denoiser(z_corrupt)
            self._profile_mark(profile, "denoiser_forward")

        audio_recon = self.codec.decode(z_clean)
        audio_denoised = self.codec.decode(z_denoised)
        self._profile_mark(profile, "same_decode_recon_denoised")

        sr = self.codec.sample_rate
        self._log_spectral_audio_losses(tag, audio_denoised, clean_d, sr)
        self._profile_mark(profile, "auraloss_spectral_metrics")
        T = clean_cpu.shape[-1]
        for i in range(n):
            examples = {
                "clean": clean_cpu[i],
                "corrupt": corrupt_cpu[i],
                "recon": audio_recon[i, :, :T].cpu(),
                "denoised": audio_denoised[i, :, :T].cpu(),
            }
            for name, audio in examples.items():
                audio_tag = f"{tag}/{name}_{i}" if fixed_audio_tags else f"{tag}/{name}_{i}"
                self.exp.log_audio(audio_tag, audio, self.global_step, sr)
                self.exp.log_spectrogram(
                    f"spectrogram/{audio_tag}", audio, self.global_step
                )
        self._profile_mark(profile, "tensorboard_audio_spectrogram_write")
        self._profile_print(f"{tag}_audio_log", profile)
        self.denoiser.train()

    @torch.no_grad()
    def _log_precomputed_waveform_audio(
        self,
        tag,
        batch,
        fixed_audio_tags=False,
        log_audio_mse=False,
    ):
        """Log precomputed waveform-domain TCN examples."""
        profile = self._new_profile(f"{tag}_precomputed_waveform_audio_log")
        self.denoiser.eval()
        n = min(self.cfg["logging"]["num_audio_samples"], batch["x_audio"].shape[0])
        sliced = {"x_audio": batch["x_audio"][:n], "y_audio": batch["y_audio"][:n]}
        x_audio, y_audio = self._precomputed_waveforms_to_device(sliced, profile=profile)
        audio_denoised = self.denoiser(x_audio)
        self._profile_mark(profile, "denoiser_forward")
        self._log_spectral_audio_losses(tag, audio_denoised, y_audio, self.sample_rate)
        self._profile_mark(profile, "spectral_mse_metrics")

        for i in range(n):
            window_idx = int(batch["window_idx"][i].item())
            T = min(y_audio.shape[-1], audio_denoised.shape[-1])
            examples = {
                "ground_truth": y_audio[i, :, :T].detach().cpu(),
                "model_input": x_audio[i, :, :T].detach().cpu(),
                "model_output": audio_denoised[i, :, :T].detach().cpu(),
            }
            for name, audio in examples.items():
                audio_tag = (
                    f"{tag}/{name}_{i}"
                    if fixed_audio_tags
                    else f"{tag}/{name}_window_{window_idx:06d}"
                )
                self.exp.log_audio(audio_tag, audio, self.global_step, self.sample_rate)
                self.exp.log_spectrogram(f"spectrogram/{audio_tag}", audio, self.global_step)
            self._log_audiobox_pq(tag, examples, f"window_{window_idx:06d}", self.sample_rate)
        self._profile_mark(profile, "tensorboard_precomputed_audio_spectrogram_write")
        self._profile_print(f"{tag}_precomputed_waveform_audio_log", profile)
        self.denoiser.train()

    @torch.no_grad()
    def _log_precomputed_stft_audio(
        self,
        tag,
        batch,
        fixed_audio_tags=False,
        log_audio_mse=False,
    ):
        """Log precomputed STFT-domain Conformer/U-Net examples."""
        profile = self._new_profile(f"{tag}_precomputed_stft_audio_log")
        self.denoiser.eval()
        n = min(self.cfg["logging"]["num_audio_samples"], batch["x_stft"].shape[0])
        sliced = {
            "x_stft": batch["x_stft"][:n],
            "y_stft": batch["y_stft"][:n],
        }
        x_stft, y_stft = self._precomputed_stfts_to_device(sliced, profile=profile)
        pred_stft = self.denoiser(x_stft)
        self._profile_mark(profile, "denoiser_forward")

        x_raw = unnormalize_stft(x_stft, self.stft_mean, self.stft_std)
        y_raw = unnormalize_stft(y_stft, self.stft_mean, self.stft_std)
        pred_raw = unnormalize_stft(pred_stft, self.stft_mean, self.stft_std)
        length = int(batch["num_samples"][0].item()) if "num_samples" in batch else None
        audio_input = stft_channels_to_audio(x_raw, length=length, spectral_cfg=self.spectral_cfg)
        audio_target = stft_channels_to_audio(y_raw, length=length, spectral_cfg=self.spectral_cfg)
        audio_output = stft_channels_to_audio(pred_raw, length=length, spectral_cfg=self.spectral_cfg)
        self._profile_mark(profile, "istft_reconstruct")

        self._log_spectral_audio_losses(
            tag,
            audio_output,
            audio_target,
            self.sample_rate,
            pred_stft=pred_raw,
            target_stft=y_raw,
        )
        self._profile_mark(profile, "spectral_mse_metrics")

        for i in range(n):
            window_idx = int(batch["window_idx"][i].item())
            T = min(audio_target.shape[-1], audio_output.shape[-1])
            examples = {
                "ground_truth": audio_target[i, :, :T].detach().cpu(),
                "model_input": audio_input[i, :, :T].detach().cpu(),
                "model_output": audio_output[i, :, :T].detach().cpu(),
            }
            for name, audio in examples.items():
                audio_tag = (
                    f"{tag}/{name}_{i}"
                    if fixed_audio_tags
                    else f"{tag}/{name}_window_{window_idx:06d}"
                )
                self.exp.log_audio(audio_tag, audio, self.global_step, self.sample_rate)
                self.exp.log_spectrogram(f"spectrogram/{audio_tag}", audio, self.global_step)
            self._log_audiobox_pq(tag, examples, f"window_{window_idx:06d}", self.sample_rate)
        self._profile_mark(profile, "tensorboard_precomputed_audio_spectrogram_write")
        self._profile_print(f"{tag}_precomputed_stft_audio_log", profile)
        self.denoiser.train()

    @torch.no_grad()
    def _log_precomputed_audio(
        self,
        tag,
        batch,
        fixed_audio_tags=False,
        log_audio_mse=False,
    ):
        """Log ground-truth audio plus decoded precomputed model input/output."""
        if self._is_stft_deterministic():
            return self._log_precomputed_stft_audio(
                tag,
                batch,
                fixed_audio_tags=fixed_audio_tags,
                log_audio_mse=log_audio_mse,
            )
        if self._is_waveform_deterministic():
            return self._log_precomputed_waveform_audio(
                tag,
                batch,
                fixed_audio_tags=fixed_audio_tags,
                log_audio_mse=log_audio_mse,
            )
        profile = self._new_profile(f"{tag}_precomputed_audio_log")

        self.denoiser.eval()
        if self.ema_denoiser is not None:
            self.ema_denoiser.eval()

        n = min(self.cfg["logging"]["num_audio_samples"], batch["z_clean"].shape[0])
        sliced = {
            "z_cond": batch["z_cond"][:n],
            "z_clean": batch["z_clean"][:n],
        }
        z_cond, z_clean = self._precomputed_latents_to_device(sliced, profile=profile)
        z_input = z_clean if self._is_ddpm_clean_cond() else z_cond
        primary_output_name = "model_output"

        if self._is_ddpm() or self._is_ddpm_clean_cond():
            z_outputs = {
                "model_output": self._sample_ddpm_latent(
                    z_input,
                    num_sampling_steps=self._ddpm_cfg()["num_train_timesteps"],
                )
            }
            self._profile_mark(profile, "ddpm_reverse_sample")
        elif self._is_cfm():
            z_outputs, primary_output_name = self._sample_cfm_audio_latents(
                z_input, tag
            )
            self._profile_mark(profile, "cfm_euler_sample")
        else:
            z_outputs = {"model_output": self.denoiser(z_input)}
            self._profile_mark(profile, "denoiser_forward")

        if self.latent_mean is None or self.latent_std is None:
            raise ValueError(
                "Precomputed audio logging requires latent_mean/latent_std in "
                "metadata.pt. Rerun precompute so normalized latents can be "
                "unnormalized before SAME decoding."
            )
        log_fixed_refs = tag != "val" or not self._precomputed_val_fixed_audio_logged
        audio_input = None
        if log_fixed_refs:
            z_input_raw = unnormalize_latent(z_input, self.latent_mean, self.latent_std)
            audio_input = self.codec.decode(z_input_raw)
        audio_outputs = {
            name: self.codec.decode(
                unnormalize_latent(z_output, self.latent_mean, self.latent_std)
            )
            for name, z_output in z_outputs.items()
        }
        primary_audio_output = audio_outputs[primary_output_name]
        self._profile_mark(profile, "same_decode_input_output")

        gt_dir = self._precompute_cfg()["ground_truth_dir"]
        sr = self.codec.sample_rate
        logged_fixed_refs = False
        spectral_pairs = []
        for i in range(n):
            window_idx = int(batch["window_idx"][i].item())
            examples = {}
            ground_truth = None
            if log_fixed_refs or log_audio_mse:
                gt_path = os.path.join(gt_dir, f"window_{window_idx:06d}.wav")
                if not os.path.exists(gt_path):
                    raise FileNotFoundError(f"Ground-truth audio not found: {gt_path}")
                ground_truth, gt_sr = torchaudio.load(gt_path)
                if gt_sr != sr:
                    ground_truth = torchaudio.functional.resample(ground_truth, gt_sr, sr)
                if ground_truth.shape[0] > 1:
                    ground_truth = ground_truth.mean(0, keepdim=True)
                T = min(ground_truth.shape[-1], primary_audio_output.shape[-1])
                spectral_pairs.append((primary_audio_output[i, :, :T], ground_truth[:, :T]))

            if log_fixed_refs and ground_truth is not None:
                gt_key = (tag, window_idx)
                if gt_key not in self._logged_precomputed_ground_truth:
                    examples["ground_truth"] = ground_truth[:, :T]
                    self._logged_precomputed_ground_truth.add(gt_key)
                examples["model_input"] = audio_input[i, :, :T].detach().cpu()
                logged_fixed_refs = True
            else:
                T = primary_audio_output.shape[-1]
            for name, audio_output in audio_outputs.items():
                examples[name] = audio_output[i, :, :T].detach().cpu()
            for name, audio in examples.items():
                audio_tag = (
                    f"{tag}/{name}_{i}"
                    if fixed_audio_tags
                    else f"{tag}/{name}_window_{window_idx:06d}"
                )
                self.exp.log_audio(audio_tag, audio, self.global_step, sr)
                self.exp.log_spectrogram(
                    f"spectrogram/{audio_tag}", audio, self.global_step
                )
            # Audiobox PQ: score model output vs ground truth only (no extra decode).
            if ground_truth is not None:
                pq_examples = {
                    "ground_truth": ground_truth[:, :T],
                    "model_output": primary_audio_output[i, :, :T].detach().cpu(),
                }
                self._log_audiobox_pq(tag, pq_examples, f"window_{window_idx:06d}", sr)
        if spectral_pairs:
            self._log_spectral_audio_loss_pairs(tag, spectral_pairs, sr)
            self._profile_mark(profile, "auraloss_spectral_metrics")

        if tag == "val" and logged_fixed_refs:
            self._precomputed_val_fixed_audio_logged = True
        self._profile_mark(profile, "tensorboard_precomputed_audio_spectrogram_write")
        self._profile_print(f"{tag}_precomputed_audio_log", profile)
        self.denoiser.train()

    @torch.no_grad()
    def _log_audiobox_pq(self, tag, audio_examples, window_key, sr):
        """Score audio examples with the Audiobox PQ predictor and log to TensorBoard.

        audio_examples: dict mapping name -> [1, T] float CPU tensor.
        window_key: tag suffix string, e.g. 'window_000001'.
        sr: sample rate of the supplied audio (predictor resamples to 16 kHz internally).
        """
        if self.aes_predictor is None:
            return
        try:
            names = list(audio_examples.keys())
            batch = []
            for name, audio in audio_examples.items():
                # Audiobox accepts an in-memory waveform shaped [channels, T].
                # Do not squeeze mono [1, T] to [T]: its resampling path later
                # indexes both channel and time dimensions.
                waveform = audio.detach().float().cpu()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                elif waveform.ndim == 3 and waveform.shape[0] == 1:
                    waveform = waveform[0]
                if waveform.ndim != 2 or waveform.shape[-1] == 0:
                    raise ValueError(
                        f"Audiobox example {name!r} must have shape [channels, T], "
                        f"got {tuple(waveform.shape)}"
                    )
                batch.append({"path": waveform, "sample_rate": sr})

            results = self.aes_predictor.forward(batch)
            for name, result in zip(names, results):
                self.exp.log_scalar(
                    f"audiobox_pq/{tag}/{name}_{window_key}",
                    float(result["PQ"]),
                    self.global_step,
                )
        except Exception as exc:
            print(f"[audiobox_pq] Scoring failed for {window_key}: {exc}", flush=True)

    def _cfm_default_inference_steps(self):
        """Default Euler step count used for CFM scalar audio MSE comparison."""
        return max(1, int(self._cfm_cfg().get("inference_steps_default", 1)))

    def _cfm_audio_step_counts(self, tag):
        """Return CFM Euler step counts to log as separate TensorBoard audio tags."""
        default_steps = self._cfm_default_inference_steps()
        if tag.startswith("val"):
            raw_steps = self._cfm_cfg().get("inference_steps_eval", [default_steps])
        else:
            raw_steps = [default_steps]

        # An explicit guidance sweep is a self-contained validation protocol:
        # log exactly its requested evaluation steps. Legacy configurations
        # retain the historical behavior of always including the default.
        steps = [] if self._cfm_audio_cfg_scales(tag) is not None else [default_steps]
        for value in raw_steps:
            step_count = max(1, int(value))
            if step_count not in steps:
                steps.append(step_count)
        if not steps:
            raise ValueError("CFM inference_steps_eval must contain at least one step count")
        return steps

    def _sample_cfm_audio_latents(self, z_input, tag):
        """Sample CFM TensorBoard outputs and identify the primary metric output."""
        outputs = {}
        step_counts = self._cfm_audio_step_counts(tag)
        cfg_scales = self._cfm_audio_cfg_scales(tag)
        if cfg_scales is None:
            for steps in step_counts:
                name = f"model_output_steps_{steps}"
                outputs[name] = self._sample_cfm_latent(
                    z_input,
                    inference_steps=steps,
                    cfg_scale=self._cfm_cfg().get("cfg_scale", 1.0),
                )
            primary_name = (
                f"model_output_steps_{self._cfm_default_inference_steps()}"
            )
            return outputs, primary_name

        default_scale = float(self._cfm_cfg().get("cfg_scale", 1.0))
        primary_scale = next(
            (
                scale
                for scale in cfg_scales
                if abs(scale - default_scale) < 1e-8
            ),
            cfg_scales[0],
        )
        primary_steps = step_counts[0]
        for steps in step_counts:
            # Every scale at this step count starts from exactly the same
            # Gaussian latent, isolating the effect of guidance.
            initial_noise = torch.randn_like(z_input)
            for scale in cfg_scales:
                suffix = self._format_cfm_cfg_scale_tag(scale)
                name = f"model_output_steps_{steps}_cfg_{suffix}"
                outputs[name] = self._sample_cfm_latent(
                    z_input,
                    inference_steps=steps,
                    cfg_scale=scale,
                    initial_noise=initial_noise,
                )
        primary_name = (
            f"model_output_steps_{primary_steps}_cfg_"
            f"{self._format_cfm_cfg_scale_tag(primary_scale)}"
        )
        return outputs, primary_name

    @staticmethod
    def _format_cfm_cfg_scale_tag(scale):
        """Format a finite CFG scale as a filesystem/TensorBoard-safe suffix."""
        scale = float(scale)
        if not math.isfinite(scale):
            raise ValueError(f"CFM cfg scale must be finite, got {scale!r}")
        text = format(scale, ".12g")
        if "." not in text and "e" not in text.lower():
            text += ".0"
        return text.replace("-", "m").replace("+", "").replace(".", "p")

    @torch.no_grad()
    def _sample_cfm_latent(
        self,
        z_cond,
        inference_steps=None,
        cfg_scale=None,
        generator=None,
        initial_noise=None,
    ):
        """Euler-sample a restored normalized latent from the CFM velocity field."""
        steps = max(
            1,
            int(
                self._cfm_default_inference_steps()
                if inference_steps is None
                else inference_steps
            ),
        )
        cfg_scale = (
            float(self._cfm_cfg().get("cfg_scale", 1.0))
            if cfg_scale is None
            else float(cfg_scale)
        )
        if initial_noise is not None and generator is not None:
            raise ValueError("Pass either initial_noise or generator, not both")
        if initial_noise is None:
            z = torch.randn(
                z_cond.shape,
                device=z_cond.device,
                dtype=z_cond.dtype,
                generator=generator,
            )
        else:
            if initial_noise.shape != z_cond.shape:
                raise ValueError(
                    "CFM initial_noise shape must match z_cond, got "
                    f"{tuple(initial_noise.shape)} and {tuple(z_cond.shape)}"
                )
            if initial_noise.device != z_cond.device:
                raise ValueError(
                    "CFM initial_noise and z_cond must be on the same device"
                )
            if initial_noise.dtype != z_cond.dtype:
                raise ValueError(
                    "CFM initial_noise and z_cond must have the same dtype"
                )
            z = initial_noise.clone()
        model = self.ema_denoiser.module if self.ema_denoiser is not None else self.denoiser
        z_uncond = None if abs(cfg_scale - 1.0) < 1e-8 else torch.zeros_like(z_cond)
        time_grid = make_cfm_euler_time_grid(
            steps,
            self._cfm_cfg(),
            device=z.device,
            dtype=z.dtype,
        )

        for step_idx in range(steps):
            t_value = time_grid[step_idx]
            dt = time_grid[step_idx + 1] - t_value
            t = t_value.expand(z.shape[0])
            if z_uncond is None:
                velocity = model(z, t, z_cond)
            else:
                velocity_uncond = model(z, t, z_uncond)
                velocity_cond = model(z, t, z_cond)
                velocity = velocity_uncond + cfg_scale * (
                    velocity_cond - velocity_uncond
                )
            self._assert_cfm_shapes(z_cond, z_cond, z, velocity, velocity, t)
            z = z + dt * velocity
        return z

    @torch.no_grad()
    def _sample_ddpm_latent(self, z_cond, num_sampling_steps=None, generator=None):
        """Run the full DDPM reverse chain to sample a restored normalized latent."""
        # Full ancestral DDPM reverse chain. This is used for TensorBoard audio
        # during training and mirrors inference.py, but inference.py itself is
        # not called from the training loop.
        dcfg = self._ddpm_cfg()
        total_steps = dcfg["num_train_timesteps"]
        if num_sampling_steps is None:
            num_sampling_steps = total_steps
        if num_sampling_steps != total_steps:
            raise ValueError("DDPM sampling uses the full configured reverse chain")
        z = torch.randn(
            z_cond.shape,
            device=z_cond.device,
            dtype=z_cond.dtype,
            generator=generator,
        )
        model = self.ema_denoiser.module if self.ema_denoiser is not None else self.denoiser
        for t_int in range(total_steps - 1, -1, -1):
            t = torch.full((z.shape[0],), t_int, device=z.device, dtype=torch.long)
            eps_pred = model(z, t, z_cond)
            beta_t = extract(self.ddpm_buffers["betas"], t, z.shape).to(z.dtype)
            sqrt_recip_alpha_t = extract(
                self.ddpm_buffers["sqrt_recip_alphas"], t, z.shape
            ).to(z.dtype)
            sqrt_one_minus_alpha_bar_t = extract(
                self.ddpm_buffers["sqrt_one_minus_alphas_cumprod"], t, z.shape
            ).to(z.dtype)
            model_mean = sqrt_recip_alpha_t * (
                z - beta_t / sqrt_one_minus_alpha_bar_t * eps_pred
            )
            if t_int > 0:
                posterior_var_t = extract(
                    self.ddpm_buffers["posterior_variance"], t, z.shape
                ).to(z.dtype)
                noise = torch.randn(
                    z.shape,
                    device=z.device,
                    dtype=z.dtype,
                    generator=generator,
                )
                z = model_mean + torch.sqrt(posterior_var_t) * noise
            else:
                z = model_mean
        return z

    def _save_checkpoint(self):
        """Save model, optimizer, config, and DDPM-specific inference state."""
        # Save enough state to resume training and enough DDPM-specific state
        # for inference: EMA denoiser, latent stats, and schedule buffers.
        state = {
            "global_step": self.global_step,
            "denoiser": self._raw_denoiser().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
            "distributed_world_size": self.world_size,
            "per_gpu_batch_size": int(self.cfg["training"]["batch_size"]),
            "global_batch_size": int(
                self.cfg["training"]["batch_size"] * self.world_size
            ),
            "workers_per_rank": int(self.cfg["training"]["num_workers"]),
            "precomputed_val_fixed_audio_logged": self._precomputed_val_fixed_audio_logged,
            "best_total_val_audio_loss": self._best_total_val_audio_loss,
            "best_total_val_audio_step": self._best_total_val_audio_step,
            "best_total_val_audio_epoch": self._best_total_val_audio_epoch,
            "logged_precomputed_ground_truth": [
                [tag, window_idx]
                for tag, window_idx in sorted(self._logged_precomputed_ground_truth)
            ],
        }
        if self._precomputed_val_audio_examples is not None:
            state["precomputed_val_audio_examples"] = {
                key: value.detach().cpu()
                for key, value in self._precomputed_val_audio_examples.items()
            }
        if self._is_same_latent_generative():
            state["ema_denoiser"] = self.ema_denoiser.state_dict()
        if self._is_ddpm() or self._is_ddpm_clean_cond():
            state["diffusion_schedule_buffers"] = {
                k: v.detach().cpu() for k, v in self.ddpm_buffers.items()
            }
        if self.latent_mean is not None and self.latent_std is not None:
            state.update({
                "latent_mean": self.latent_mean.detach().cpu(),
                "latent_std": self.latent_std.detach().cpu(),
            })
        if self.stft_mean is not None and self.stft_std is not None:
            state.update({
                "stft_mean": self.stft_mean.detach().cpu(),
                "stft_std": self.stft_std.detach().cpu(),
                "spectral_cfg": dict(self.spectral_cfg),
            })
        keep_last = int(
            self.cfg.get("training", {}).get("keep_last_checkpoints", 5)
        )
        self.exp.save_checkpoint(state, self.global_step, keep_last=keep_last)
