"""Standalone single-file inference used by ``main.py infer``.

The training code in ``trainer.py`` contains its own matching sampler for
validation audio and TensorBoard logging; it does not call this module.
Research-scale dataset evaluations also used dedicated batching scripts.
This module is nevertheless the supported public path for checkpoint-based,
arbitrary-length audio restoration and mirrors the Trainer's CFM sampler.
"""

import math

import torch
import torchaudio
import yaml

from .diffusion_utils import (
    ensure_channel_first,
    extract,
    make_cfm_euler_time_grid,
    move_ddpm_buffers,
    normalize_latent,
    precompute_ddpm_buffers,
    unnormalize_latent,
)
from .models import DIT_MSE_DENOISER_TYPES, build_codec, build_denoiser
from .spectral_losses import (
    audio_to_stft_channels,
    default_spectral_config,
    normalize_stft,
    stft_channels_to_audio,
    unnormalize_stft,
)


class Inferencer:
    """End-to-end inference: load audio -> encode -> denoise -> decode -> save."""

    def __init__(self, checkpoint_path, config_path=None, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        cfg = ckpt.get("config")
        if cfg is None:
            if config_path is None:
                raise ValueError(
                    "Checkpoint has no embedded config; pass config_path explicitly."
                )
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

        self.cfg = cfg
        codec_name = cfg["codec"]["name"]
        if (self._is_ddpm() or self._is_cfm()) and codec_name != "same_l":
            raise ValueError(
                "denoiser.type=ddpm/cfm is supported only with codec.name=same_l"
            )
        if self._is_dit_mse() and codec_name != "same_l":
            raise ValueError("denoiser.type=dit_mse is supported only with codec.name=same_l")
        self.codec = None
        self.sample_rate = int(cfg.get("dataset", {}).get("sample_rate", 44100))
        self.spectral_cfg = dict(default_spectral_config())
        self.spectral_cfg.update(cfg.get("spectral", {}))
        if self._uses_codec():
            self.codec = build_codec(codec_name, self.device)
            self.sample_rate = self.codec.sample_rate
        latent_dim = self.codec.latent_dim if self.codec is not None else None
        self.denoiser = build_denoiser(
            cfg["denoiser"],
            latent_dim,
            spectral_cfg=self.spectral_cfg,
        )
        # Generative DDPM/CFM checkpoints sample with their EMA weights, just
        # like Trainer's TensorBoard audio path. AveragedModel stores them
        # beneath the ``module.`` prefix plus a non-model n_averaged buffer.
        if (self._is_ddpm() or self._is_cfm()) and "ema_denoiser" in ckpt:
            denoiser_state = {
                key.removeprefix("module."): value
                for key, value in ckpt["ema_denoiser"].items()
                if key.startswith("module.")
            }
        else:
            denoiser_state = ckpt["denoiser"]
        self.denoiser.load_state_dict(denoiser_state)
        self.denoiser.to(self.device).eval()
        self.ddpm_buffers = None
        self.latent_mean = None
        self.latent_std = None
        self.stft_mean = None
        self.stft_std = None
        if self._is_ddpm() or self._is_cfm() or self._is_dit_mse():
            self.latent_mean = ckpt["latent_mean"].to(self.device)
            self.latent_std = ckpt["latent_std"].to(self.device)
        if self._is_ddpm():
            if "diffusion_schedule_buffers" in ckpt:
                self.ddpm_buffers = move_ddpm_buffers(
                    ckpt["diffusion_schedule_buffers"], self.device
                )
            else:
                dcfg = self._ddpm_cfg()
                self.ddpm_buffers = precompute_ddpm_buffers(
                    num_train_timesteps=dcfg["num_train_timesteps"],
                    beta_start=dcfg["beta_start"],
                    beta_end=dcfg["beta_end"],
                    device=self.device,
                )
        if self._is_stft_deterministic():
            self.stft_mean = ckpt["stft_mean"].to(self.device)
            self.stft_std = ckpt["stft_std"].to(self.device)
            self.spectral_cfg.update(ckpt.get("spectral_cfg", {}))

    @torch.no_grad()
    def denoise_file(self, input_path, output_path, overlap=0.5, chunk_sec=30.0):
        """Denoise an audio file with overlap-add for arbitrary length.

        Args:
            input_path:  path to input wav/flac/mp3
            output_path: path to write denoised wav
            overlap:     fraction of overlap between chunks (0–1)
            chunk_sec:   processing chunk length in seconds
        """
        audio, sr = torchaudio.load(input_path)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)

        denoised = self._overlap_add(audio.unsqueeze(0), overlap, chunk_sec)
        denoised = denoised.squeeze(0).cpu()
        torchaudio.save(output_path, denoised, self.sample_rate)

    @torch.no_grad()
    def denoise_tensor(self, audio):
        """Denoise a (1, T) or (B, 1, T) tensor already at the correct sample rate.

        Returns tensor of same shape.
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)
        if self._is_waveform_deterministic():
            return self.denoiser(audio)
        if self._is_stft_deterministic():
            stft = audio_to_stft_channels(audio, self.spectral_cfg)
            stft_norm = normalize_stft(stft, self.stft_mean, self.stft_std)
            pred_norm = self.denoiser(stft_norm)
            pred = unnormalize_stft(pred_norm, self.stft_mean, self.stft_std)
            return stft_channels_to_audio(
                pred,
                length=audio.shape[-1],
                spectral_cfg=self.spectral_cfg,
            )
        audio = self.codec.preprocess(audio)
        z = self.codec.encode(audio)
        if self._is_ddpm():
            z = self._restore_ddpm_latent(z)
        elif self._is_cfm():
            z = self._restore_cfm_latent(z)
        elif self._is_dit_mse():
            z = ensure_channel_first(z, channels=self.codec.latent_dim)
            z_norm = normalize_latent(z, self.latent_mean, self.latent_std)
            z_norm = z_norm.to(next(self.denoiser.parameters()).dtype)
            z = unnormalize_latent(
                self.denoiser(z_norm),
                self.latent_mean,
                self.latent_std,
            )
        else:
            z = self.denoiser(z)
        return self.codec.decode(z)

    def _is_ddpm(self):
        return self.cfg["denoiser"]["type"] in {"ddpm", "ddpm_8m"}

    def _is_cfm(self):
        return self.cfg["denoiser"]["type"] in {
            "cfm", "cfm_4m", "cfm_8m", "cfm_16m", "cfm_40m", "cfm_80m"
        }

    def _is_dit_mse(self):
        return self.cfg["denoiser"]["type"] in DIT_MSE_DENOISER_TYPES

    def _is_stft_deterministic(self):
        return self.cfg["denoiser"]["type"] in {
            "conformer", "conformer_8m", "unet", "unet1d"
        }

    def _is_waveform_deterministic(self):
        return self.cfg["denoiser"]["type"] in {"tcn", "tcn_8m"}

    def _uses_codec(self):
        return not (self._is_stft_deterministic() or self._is_waveform_deterministic())

    def _ddpm_cfg(self):
        key = "ddpm_8m" if self.cfg["denoiser"]["type"] == "ddpm_8m" else "ddpm"
        return self.cfg["denoiser"][key]

    def _cfm_cfg(self):
        return self.cfg["denoiser"][self.cfg["denoiser"]["type"]]

    @torch.no_grad()
    def _restore_ddpm_latent(self, z_cond_raw):
        z_cond_raw = ensure_channel_first(
            z_cond_raw, channels=self.codec.latent_dim
        )
        valid_t = z_cond_raw.shape[-1]
        z_cond = normalize_latent(z_cond_raw, self.latent_mean, self.latent_std)
        z_cond = z_cond.to(next(self.denoiser.parameters()).dtype)

        dcfg = self._ddpm_cfg()
        total_steps = dcfg["num_train_timesteps"]
        z = torch.randn(z_cond.shape, device=z_cond.device, dtype=z_cond.dtype)

        for t_int in range(total_steps - 1, -1, -1):
            t = torch.full((z.shape[0],), t_int, device=z.device, dtype=torch.long)
            eps_pred = self.denoiser(z, t, z_cond)
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
                z = model_mean + torch.sqrt(posterior_var_t) * torch.randn_like(z)
            else:
                z = model_mean

        z = z[..., :valid_t]
        return unnormalize_latent(z, self.latent_mean, self.latent_std)

    @torch.no_grad()
    def _restore_cfm_latent(self, z_cond_raw, cfg_scale=None):
        """Euler-sample a restored latent using the trained CFM velocity field.

        This mirrors Trainer._sample_cfm_latent: start at Gaussian noise at
        t=0, integrate to t=1, optionally apply classifier-free guidance, then
        undo the fixed SAME latent normalization before decoding.
        """
        z_cond_raw = ensure_channel_first(
            z_cond_raw, channels=self.codec.latent_dim
        )
        valid_t = z_cond_raw.shape[-1]
        z_cond = normalize_latent(
            z_cond_raw, self.latent_mean, self.latent_std
        ).to(next(self.denoiser.parameters()).dtype)

        cfcfg = self._cfm_cfg()
        steps = max(1, int(cfcfg.get("inference_steps_default", 1)))
        cfg_scale = float(
            cfcfg.get("cfg_scale", 1.0) if cfg_scale is None else cfg_scale
        )
        if not math.isfinite(cfg_scale):
            raise ValueError(f"CFM cfg_scale must be finite, got {cfg_scale!r}")
        z = torch.randn_like(z_cond)
        z_uncond = None if abs(cfg_scale - 1.0) < 1e-8 else torch.zeros_like(z_cond)
        time_grid = make_cfm_euler_time_grid(
            steps,
            cfcfg,
            device=z.device,
            dtype=z.dtype,
        )

        for step_idx in range(steps):
            t_value = time_grid[step_idx]
            dt = time_grid[step_idx + 1] - t_value
            t = t_value.expand(z.shape[0])
            if z_uncond is None:
                velocity = self.denoiser(z, t, z_cond)
            else:
                velocity_uncond = self.denoiser(z, t, z_uncond)
                velocity_cond = self.denoiser(z, t, z_cond)
                velocity = velocity_uncond + cfg_scale * (
                    velocity_cond - velocity_uncond
                )
            z = z + dt * velocity

        z = z[..., :valid_t]
        return unnormalize_latent(z, self.latent_mean, self.latent_std)

    # -------------------------------------------------------- overlap-add --

    def _overlap_add(self, audio, overlap_ratio, chunk_sec):
        """Process long audio in overlapping chunks with Hann crossfade.

        Args:
            audio: (1, 1, T)
            overlap_ratio: float in [0, 1)
            chunk_sec: chunk duration in seconds
        Returns:
            (1, 1, T)
        """
        T = audio.shape[-1]
        chunk_samples = int(chunk_sec * self.sample_rate)
        # align chunks to the active representation hop.
        hop = self.codec.hop_length if self.codec is not None else int(
            self.spectral_cfg.get("hop_size", 256)
        )
        chunk_samples = math.ceil(chunk_samples / hop) * hop
        overlap_samples = int(chunk_samples * overlap_ratio)
        overlap_samples = (overlap_samples // hop) * hop
        step = chunk_samples - overlap_samples

        if T <= chunk_samples:
            return self.denoise_tensor(audio)

        # build Hann fade windows
        fade = torch.hann_window(2 * overlap_samples, device=self.device)
        fade_in = fade[:overlap_samples]   # 0 → 1
        fade_out = fade[overlap_samples:]  # 1 → 0

        output = torch.zeros(1, 1, T, device=self.device)
        starts = list(range(0, T, step))

        for i, s in enumerate(starts):
            e = min(s + chunk_samples, T)
            chunk = audio[:, :, s:e].to(self.device)

            # pad short final chunk
            if chunk.shape[-1] < chunk_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, chunk_samples - chunk.shape[-1])
                )

            denoised = self.denoise_tensor(chunk)
            denoised = denoised[:, :, : (e - s)]

            # apply crossfade
            L = denoised.shape[-1]
            if i > 0 and overlap_samples > 0:
                ol = min(overlap_samples, L)
                denoised[:, :, :ol] *= fade_in[:ol]
            if i < len(starts) - 1 and overlap_samples > 0:
                ol = min(overlap_samples, L)
                denoised[:, :, L - ol :] *= fade_out[-ol:]

            output[:, :, s:e] += denoised

        return output[:, :, :T]
