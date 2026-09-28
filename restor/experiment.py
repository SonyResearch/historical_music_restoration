import os
import re

import yaml
from torch.utils.tensorboard import SummaryWriter


class Experiment:
    """Manages experiment directories, config snapshots, and TensorBoard writers.

    Directory layout:
        experiments/
        └── my_experiment/
            ├── config.yaml          # frozen config snapshot
            ├── tensorboard/         # TB logs
            └── checkpoints/         # model checkpoints

    Safety features:
        - Refuses to overwrite an existing experiment (use --resume)
        - Snapshots the full config at creation so you always know what ran
        - Validates config before training starts
        - Resume loads the snapshotted config to prevent drift
    """

    REQUIRED_KEYS = ["codec", "denoiser", "dataset", "training"]

    def __init__(self, name, cfg, root="experiments", resume=False):
        self.name = name
        self.root = root
        self.dir = os.path.join(root, name)
        self.tb_dir = os.path.join(self.dir, "tensorboard")
        self.ckpt_dir = os.path.join(self.dir, "checkpoints")
        self.config_path = os.path.join(self.dir, "config.yaml")

        if resume:
            cfg = self._resume(cfg)
        else:
            self._create(cfg)

        self.cfg = cfg
        # Point trainer config to experiment paths
        self.cfg["training"]["checkpoint_dir"] = self.ckpt_dir
        self.cfg["logging"]["log_dir"] = self.tb_dir

        self._writer = None

    # --------------------------------------------------------- lifecycle --

    def _create(self, cfg):
        if os.path.exists(self.dir):
            raise FileExistsError(
                f"Experiment '{self.name}' already exists at {self.dir}.\n"
                f"  Use --resume to continue, or pick a different --name."
            )
        self._validate(cfg)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Created experiment: {self.dir}")

    def _resume(self, cli_cfg):
        if not os.path.exists(self.dir):
            raise FileNotFoundError(
                f"Cannot resume: experiment '{self.name}' not found at {self.dir}."
            )
        with open(self.config_path) as f:
            saved_cfg = yaml.safe_load(f)
        print(f"Resumed experiment: {self.dir} (using saved config)")
        return saved_cfg

    @classmethod
    def _validate(cls, cfg):
        missing = [k for k in cls.REQUIRED_KEYS if k not in cfg]
        if missing:
            raise ValueError(f"Config missing required sections: {missing}")
        precompute = cfg.get("precompute", {})
        if precompute.get("enabled", False):
            for key in ("train_dir", "validate_dir", "ground_truth_dir"):
                path = precompute.get(key)
                if not path or not os.path.isdir(path):
                    raise FileNotFoundError(
                        f"Precomputed dataset directory is unavailable: {key}={path!r}"
                    )
        else:
            root = cfg["dataset"]["root"]
            if not os.path.isdir(root):
                raise FileNotFoundError(
                    f"Dataset root '{root}' does not exist. "
                    f"Create it and add song folders before training."
                )

    # -------------------------------------------------------- tensorboard --

    @property
    def writer(self):
        if self._writer is None:
            self._writer = SummaryWriter(self.tb_dir)
        return self._writer

    def close(self):
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_audio(self, tag, audio, step, sample_rate):
        """Log audio, clamping to [-1, 1] to prevent TB errors."""
        audio = audio.detach().cpu().clamp(-1, 1)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        # TB expects (1, T) or (C, T) — ensure no batch dim
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        self.writer.add_audio(tag, audio, step, sample_rate)

    def log_spectrogram(self, tag, audio, step, n_fft=2048, hop_length=512, eps=1e-8):
        """Log a normalized log-magnitude STFT image."""
        import torch

        audio = audio.detach().cpu()
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        if audio.dim() == 2:
            audio = audio.mean(0)
        window = torch.hann_window(n_fft)
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        spec = torch.log1p(spec + eps)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + eps)
        self.writer.add_image(tag, spec, step, dataformats="HW")

    def log_config(self):
        """Log the config as text to TensorBoard for easy reference."""
        cfg_str = yaml.dump(self.cfg, default_flow_style=False, sort_keys=False)
        self.writer.add_text("config", f"```yaml\n{cfg_str}```", 0)

    # --------------------------------------------------------- checkpoint --

    def save_checkpoint(self, state, step, keep_last=5):
        path = os.path.join(self.ckpt_dir, f"step_{step}.pt")
        import torch
        temporary = f"{path}.tmp.{os.getpid()}"
        torch.save(state, temporary)
        os.replace(temporary, path)
        # Keep a "latest" symlink, replacing it atomically so readers never
        # observe a half-written checkpoint or a missing handoff target.
        latest = os.path.join(self.ckpt_dir, "latest.pt")
        latest_temporary = f"{latest}.tmp.{os.getpid()}"
        if os.path.lexists(latest_temporary):
            os.remove(latest_temporary)
        os.symlink(os.path.abspath(path), latest_temporary)
        os.replace(latest_temporary, latest)
        # prune old checkpoints, keeping only the most recent keep_last
        if keep_last is not None and keep_last > 0:
            ckpts = sorted(
                [
                    f for f in os.listdir(self.ckpt_dir)
                    if re.match(r"step_\d+\.pt$", f)
                ],
                key=lambda f: int(re.search(r"\d+", f).group()),
            )
            for old in ckpts[:-keep_last]:
                old_path = os.path.join(self.ckpt_dir, old)
                os.remove(old_path)
                print(f"  Removed old checkpoint: {old_path}")
        print(f"  Checkpoint: {path}")
        return path

    def find_latest_checkpoint(self):
        latest = os.path.join(self.ckpt_dir, "latest.pt")
        if os.path.exists(latest):
            return latest
        # fallback: find highest step
        ckpts = [
            f for f in os.listdir(self.ckpt_dir)
            if re.match(r"step_\d+\.pt$", f)
        ]
        if not ckpts:
            return None
        ckpts.sort(key=lambda f: int(re.search(r"\d+", f).group()))
        return os.path.join(self.ckpt_dir, ckpts[-1])

    # ------------------------------------------------------------ listing --

    @classmethod
    def list_experiments(cls, root="experiments"):
        if not os.path.isdir(root):
            print("No experiments directory found.")
            return
        exps = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)))
        if not exps:
            print("No experiments found.")
            return
        print(f"{'Name':<30} {'Checkpoints':>12}  Config")
        print("-" * 70)
        for name in exps:
            ckpt_dir = os.path.join(root, name, "checkpoints")
            n_ckpts = len([f for f in os.listdir(ckpt_dir)
                           if f.endswith(".pt") and f != "latest.pt"]
                          ) if os.path.isdir(ckpt_dir) else 0
            cfg_exists = "✓" if os.path.isfile(
                os.path.join(root, name, "config.yaml")) else "✗"
            print(f"{name:<30} {n_ckpts:>12}  {cfg_exists}")

    def __repr__(self):
        return f"Experiment(name={self.name!r}, dir={self.dir!r})"
