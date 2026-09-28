"""78-rpm degradation simulator for the restor project.

The degradation model is a **paired-data simulator** that applies a chain of
physically motivated corruptions to clean audio *before* it is encoded by the
neural codec.  The model learns to map latent codes of degraded audio back to
latent codes of the clean source.

The chain has exactly FIVE stages, always applied in order (no probability
gating — every training pair is a clean↔degraded pair):

    Stage 1 – Random zero-phase EQ₁
        Pre-nonlinearity shaping: heavy bass loss, mild mid attenuation, and
        strong high-frequency decay.  Implemented as a frequency-domain
        magnitude mask (STFT → multiply → ISTFT) so there is NO phase shift
        that would mis-align the degraded input and clean target in latent
        space. The fast path uses full-segment RFFT/IRFFT; the legacy path
        uses STFT → multiply → ISTFT.

    Stage 2 – Scaled tanh nonlinearity
        y = (1 − wet)·x + wet·tanh(a·x)/a
        Models the smooth amplitude-dependent compression and mild harmonic
        coloration of cutter, groove, and playback transfer in 78-rpm records.
        The /a factor keeps DC gain ≈ 1: loud peaks are compressed, quiet
        passages stay almost linear.

    Stage 3 – Random zero-phase EQ₂
        A second, *independently* sampled EQ that shapes the harmonics created
        by the nonlinearity.  Together stages 1–3 implement the
        Wiener–Hammerstein model:  EQ₁ → static nonlinearity → EQ₂.
        Reference: UnNAFx / WH.py (Moliner & Svento) and arXiv 2504.04751.

    Stage 4 – Final bandpass
        Hard bandwidth enforcement *outside* the range of the EQ control nodes.
        Low cut (~120 Hz) removes subsonic rumble; the gramophone high cut is
        sampled from N(3000 Hz, 300 Hz).
        Roll-off is smooth (dB/octave) rather than brick-wall to avoid ringing
        artefacts in paired training data. Because EQ₂ and bandpass are
        adjacent linear stages, their masks are multiplied and applied in one
        filtering pass.

    Stage 5 – Gramophone record noise injection
        Adds real 78-rpm surface noise from the Gramophone Record Noise Dataset
        (Moliner / eloimoliner) at SNR ~ N(11, 4.5) dB, clipped to [2, 20] dB,
        matching Moliner's historical-denoising training setup.
        Requires `corruption.gramophone_noise.noise_dir` in the config.

White-noise augmentation is intentionally excluded: it was part of Eloi's
waveform-domain diffusion setup and is not needed when working in latent space.

References:
  - UnNAFx / WH.py:
      https://github.com/michalsvento/UnNAFx/blob/main/src/operators/WH.py
  - Unsupervised Estimation of Nonlinear Audio Effects (arXiv 2504.04751):
      https://arxiv.org/pdf/2504.04751
  - Gramophone Record Noise Dataset:
      https://github.com/eloimoliner/denoising-historical-recordings
"""
import os
import random
import time
from pathlib import Path

import torch
import torchaudio

# ──────────────────────── STFT constants for legacy filtering ─────────────────
# n_fft=4096 → ~11 Hz/bin at 44.1 kHz.  Much finer than the octave-spaced EQ
# nodes, so the interpolated magnitude mask is very smooth.
# win_length=2048, hop_length=512: standard 50 % overlap Hann window.
_N_FFT = 4096
_WIN_LENGTH = 2048
_HOP_LENGTH = 512


# ─────────────────────────────── module-level helpers ────────────────────────

def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag using the trainer's conventions."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sample_gaussian(
    mean: float,
    std: float,
    lo: float,
    hi: float,
    generator: torch.Generator | None = None,
) -> float:
    """Draw one scalar from N(mean, std) and clip to [lo, hi].

    Using torch.empty(1).normal_() rather than random.gauss() keeps RNG
    behaviour consistent when the caller sets torch.manual_seed().
    """
    return float(
        torch.empty(1).normal_(mean, std, generator=generator).clamp_(lo, hi)
    )


def _make_eq_mask(
    cfg: dict,
    sample_rate: float,
    device: torch.device,
    n_bins: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a random octave-band EQ curve and return a linear magnitude mask.

    Construction:
      1. Draw per-node gains (dB) from independent Gaussians (one per octave
         control node).  All gains are capped at 0 dB — the EQ can only
         attenuate, never boost above the clean source.
      2. Linearly interpolate the sampled gains between nodes in log-frequency
         space.  Log-frequency interpolation gives perceptually uniform curves
         that look natural on a mel-scale plot.
      3. Convert interpolated dB gains to linear amplitude multipliers.

    Config keys (each a list of length num_nodes):
        freq_nodes_hz : centre frequencies of the octave control nodes [Hz]
        mean_db       : mean gain at each node [dB, ≤ 0]
        std_db        : standard deviation at each node [dB]
        min_db        : minimum (hardest) allowed gain [dB]
        max_db        : maximum allowed gain [dB, ≤ 0]

    Returns:
        mask : (n_fft//2 + 1,) float32 tensor, values in (0, 1]
    """
    n_bins = n_bins or (_N_FFT // 2 + 1)

    freq_nodes = torch.tensor(
        cfg["freq_nodes_hz"], dtype=torch.float32, device=device
    )

    # Sample an independent dB gain at each octave control node.
    sampled_db = torch.tensor(
        [
            _sample_gaussian(m, s, lo, hi, generator=generator)
            for m, s, lo, hi in zip(
                cfg["mean_db"], cfg["std_db"], cfg["min_db"], cfg["max_db"]
            )
        ],
        dtype=torch.float32,
        device=device,
    )  # shape: (num_nodes,)

    # Frequency axis for all FFT bins: 0 Hz … Nyquist
    bin_freqs = torch.linspace(0.0, sample_rate / 2.0, n_bins, device=device)

    # Convert to log₂-frequency space.  Clamp to 1 Hz so log₂ is defined even
    # for the DC bin (0 Hz).
    log_bins  = torch.log2(bin_freqs.clamp(min=1.0))   # (n_bins,)
    log_nodes = torch.log2(freq_nodes.clamp(min=1.0))  # (num_nodes,)

    # Vectorised linear interpolation in log-frequency space.
    # torch.searchsorted(sorted, values) returns idx such that
    #   sorted[idx-1] <= values < sorted[idx]
    # so the interpolation interval is [idx-1, idx].
    idx = torch.searchsorted(log_nodes.contiguous(), log_bins.contiguous())
    # Clamp so both lo_idx=idx-1 and hi_idx=idx are valid node indices.
    # Values outside the node range will be clamped to the edge gain (constant
    # extrapolation) because t is subsequently clipped to [0, 1].
    idx    = idx.clamp(1, len(log_nodes) - 1)
    lo_idx = idx - 1
    hi_idx = idx

    denom    = (log_nodes[hi_idx] - log_nodes[lo_idx]).clamp(min=1e-10)
    t        = ((log_bins - log_nodes[lo_idx]) / denom).clamp(0.0, 1.0)
    gains_db = sampled_db[lo_idx] + t * (sampled_db[hi_idx] - sampled_db[lo_idx])

    # dB → linear amplitude.  All gains ≤ 0 dB, so all multipliers ≤ 1.0.
    return torch.pow(10.0, gains_db / 20.0)  # (n_bins,)


def _make_bandpass_mask(
    cfg: dict,
    sample_rate: float,
    device: torch.device,
    n_bins: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a random bandpass mask that enforces the 78-rpm bandwidth envelope.

    Unlike the EQ stages (octave control nodes), this bandpass is defined by
    explicit low/high cutoff frequencies and smooth roll-off slopes in dB/oct.

    The three regions are:
        f < low_cutoff  : gain = +low_slope  * log₂(f / low_cutoff)   [dB]  (negative)
        passband        : gain = 0 dB
        f > high_cutoff : gain = −high_slope * log₂(f / high_cutoff)  [dB]  (negative)

    Smooth slopes (not brick-wall) avoid ringing artefacts in training pairs.
    The high-cut prior is N(3000 Hz, 300 Hz).

    Config keys (Gaussian: mean, std, min, max for each parameter):
        low_cutoff_{mean,std,min,max}_hz
        high_cutoff_{mean,std,min,max}_hz
        low_slope_{mean,std,min,max}_dboct
        high_slope_{mean,std,min,max}_dboct

    Returns:
        mask : (n_fft//2 + 1,) float32 tensor, values in (0, 1]
    """
    n_bins = n_bins or (_N_FFT // 2 + 1)

    low_hz = _sample_gaussian(
        cfg["low_cutoff_mean_hz"],  cfg["low_cutoff_std_hz"],
        cfg["low_cutoff_min_hz"],   cfg["low_cutoff_max_hz"], generator,
    )
    high_hz = _sample_gaussian(
        cfg["high_cutoff_mean_hz"], cfg["high_cutoff_std_hz"],
        cfg["high_cutoff_min_hz"],  cfg["high_cutoff_max_hz"], generator,
    )
    # Guarantee low < high even if the two Gaussians happen to cross.
    low_hz, high_hz = min(low_hz, high_hz), max(low_hz, high_hz)

    low_slope = _sample_gaussian(
        cfg["low_slope_mean_dboct"],  cfg["low_slope_std_dboct"],
        cfg["low_slope_min_dboct"],   cfg["low_slope_max_dboct"], generator,
    )
    high_slope = _sample_gaussian(
        cfg["high_slope_mean_dboct"], cfg["high_slope_std_dboct"],
        cfg["high_slope_min_dboct"],  cfg["high_slope_max_dboct"], generator,
    )

    bin_freqs = torch.linspace(0.0, sample_rate / 2.0, n_bins, device=device)
    gains_db  = torch.zeros(n_bins, device=device)

    # ── Low-frequency roll-off ──────────────────────────────────────────────
    # gain(f) = low_slope * log₂(f / low_hz)  for f < low_hz
    # Since f < low_hz, log₂(f/low_hz) < 0, so the result is an attenuation.
    # Clamp bin frequency to 0.5 Hz to keep log₂ defined near DC.
    below = bin_freqs < low_hz
    gains_db[below] = low_slope * torch.log2(
        (bin_freqs[below].clamp(min=0.5) / low_hz).clamp(min=1e-12)
    )

    # ── High-frequency roll-off ─────────────────────────────────────────────
    # gain(f) = −high_slope * log₂(f / high_hz)  for f > high_hz
    above = bin_freqs > high_hz
    gains_db[above] = -high_slope * torch.log2(
        (bin_freqs[above] / high_hz).clamp(min=1e-12)
    )

    # Floor at −80 dB to avoid numerical issues from extreme attenuation.
    gains_db = gains_db.clamp(min=-80.0, max=0.0)

    return torch.pow(10.0, gains_db / 20.0)  # (n_bins,)


def _apply_fft_mask(audio: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a zero-phase EQ to audio via STFT → magnitude mask → ISTFT.

    Multiplying the complex STFT by a *real-valued* mask scales the magnitude
    of every bin while leaving the phase angle unchanged — that is exactly
    zero-phase filtering.  This is critical for paired training: the degraded
    input and the clean target remain perfectly time-aligned, so their codec
    latent codes can be compared directly as an input/target pair.

    Implementation follows UnNAFx / WH.py (Moliner & Svento):
        n_fft=4096, win_length=2048, hop_length=512, Hann window, center=True.

    Multi-channel audio is processed channel-by-channel with the *same* mask
    (mono-compatible: the same EQ curve is applied to all channels of a clip).

    Args:
        audio : (C, T) float32 tensor — one audio clip, possibly multi-channel
        mask  : (n_fft//2 + 1,) float32 tensor — real-valued linear gain per bin

    Returns:
        (C, T) float32 tensor — filtered audio, same shape as input
    """
    C, T = audio.shape
    # Re-create the window on the correct device for each call.
    # torch.hann_window is cached internally so this is cheap.
    window = torch.hann_window(_WIN_LENGTH, device=audio.device)

    filtered_channels = []
    for c in range(C):
        # Forward STFT → complex spectrogram: (n_bins, time_frames)
        X = torch.stft(
            audio[c],
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            win_length=_WIN_LENGTH,
            window=window,
            center=True,
            onesided=True,
            return_complex=True,
            normalized=False,
            pad_mode="constant",
        )

        # Multiply by the real-valued magnitude mask.
        # mask : (n_bins,) → unsqueeze(-1) → (n_bins, 1) → broadcast over time.
        # Multiplying complex by real scales magnitude, preserves phase exactly.
        X_eq = X * mask.unsqueeze(-1)

        # Inverse STFT → time domain, trimmed/padded to the original length T.
        x_eq = torch.istft(
            X_eq,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            win_length=_WIN_LENGTH,
            window=window,
            center=True,
            onesided=True,
            normalized=False,
            return_complex=False,
            length=T,
        )
        filtered_channels.append(x_eq)

    return torch.stack(filtered_channels, dim=0)  # (C, T)


def _apply_segment_fft_mask(audio: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply one static zero-phase filter with a full-segment RFFT/IRFFT.

    This is the fast training path. It is appropriate for the current EQ and
    bandpass stages because each sampled filter response is static over the
    whole 10-second clip. Compared with STFT/ISTFT, this avoids thousands of
    overlapping window transforms and runs efficiently on CUDA.

    Tradeoff: a full-segment FFT assumes periodic boundaries. The masks here
    are smooth, attenuating curves, so edge artifacts are usually minor for
    random training excerpts. Set corruption.filter_mode=stft_legacy to use
    the older overlap-add path if exact STFT behavior is needed.
    """
    if audio.ndim != 2:
        raise ValueError(f"Expected audio shape (C, T), got {tuple(audio.shape)}")
    T = audio.shape[-1]
    expected_bins = T // 2 + 1
    if mask.shape[-1] != expected_bins:
        raise ValueError(
            f"Mask has {mask.shape[-1]} bins, expected {expected_bins} for T={T}"
        )
    orig_dtype = audio.dtype
    X = torch.fft.rfft(audio.float(), n=T, dim=-1)
    y = torch.fft.irfft(X * mask.to(X.device, X.real.dtype).unsqueeze(0), n=T, dim=-1)
    return y.to(orig_dtype)


def _apply_segment_fft_mask_batch(audio: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Apply per-item static zero-phase filters to a batch via RFFT/IRFFT.

    Args:
        audio: (B, C, T)
        masks: (B, T//2 + 1), one frequency response per batch item
    """
    if audio.ndim != 3:
        raise ValueError(f"Expected audio shape (B, C, T), got {tuple(audio.shape)}")
    B, _, T = audio.shape
    expected_bins = T // 2 + 1
    if masks.shape != (B, expected_bins):
        raise ValueError(
            f"Expected masks shape {(B, expected_bins)}, got {tuple(masks.shape)}"
        )
    orig_dtype = audio.dtype
    X = torch.fft.rfft(audio.float(), n=T, dim=-1)
    y = torch.fft.irfft(
        X * masks.to(X.device, X.real.dtype).unsqueeze(1),
        n=T,
        dim=-1,
    )
    return y.to(orig_dtype)


# ────────────────────────────── main public class ────────────────────────────

class AudioCorruptor:
    """Five-stage 78-rpm degradation chain for paired training data simulation.

    The chain is a physically motivated simulator of the recording, transfer,
    and playback degradation found in 78-rpm gramophone records.  All five
    stages always fire on every call — every training pair is clean↔degraded.

    Stage summary:
        1. Zero-phase EQ₁       — broadband attenuation, pre-nonlinearity
        2. Scaled tanh           — smooth amplitude saturation (WH middle block)
        3. Zero-phase EQ₂       — post-nonlinearity coloration (independently sampled)
        4. Final bandpass        — hard bandwidth enforcement (outside EQ range)
        5. Gramophone noise      — additive 78-rpm surface noise at random SNR

    See module docstring for full design rationale and literature references.

    Args:
        cfg : the `corruption` sub-dict from config/default.yaml
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.filter_mode = cfg.get("filter_mode", "segment_fft")
        if self.filter_mode not in {"segment_fft", "stft_legacy"}:
            raise ValueError(
                "corruption.filter_mode must be 'segment_fft' or 'stft_legacy', "
                f"got {self.filter_mode!r}"
            )

        # ── Pre-scan the gramophone noise directory ────────────────────────────
        # Scanning once at construction avoids filesystem overhead during training.
        # Stage 5 will raise NotImplementedError on the first __call__ if the
        # noise directory is not yet configured.
        noise_dir = cfg.get("gramophone_noise", {}).get("noise_dir")
        if noise_dir:
            noise_path = Path(noise_dir)
            self.noise_files: list[Path] = sorted(noise_path.rglob("*.wav"))
            if not self.noise_files:
                raise FileNotFoundError(
                    f"No .wav files found in gramophone noise_dir: {noise_dir}\n"
                    "Download the Gramophone Record Noise Dataset from:\n"
                    "  scripts/download_gramophone_noise.sh"
                )
        else:
            # TODO: set `corruption.gramophone_noise.noise_dir` in
            # config/default.yaml before running training.  Stage 5 will raise
            # NotImplementedError on the first forward pass until this is set.
            self.noise_files = []

        # Optional runtime cache populated by preload_noise(). Each entry keeps
        # the decoded/resampled waveform resident on the training GPU so the
        # hot path performs no file open, WAV decode, resample, or H2D copy.
        self.noise_cache = None
        self.noise_cache_names = None
        self.noise_cache_sample_rate = None

    # ──────────────────────────────────────── public interface ────────────────

    @torch.no_grad()
    def preload_noise(self, device: torch.device, sample_rate: int) -> None:
        """Decode/resample the gramophone library once and retain it on GPU."""
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Gramophone GPU preload requires a CUDA device")
        if not self.noise_files:
            raise ValueError("Cannot preload an empty gramophone noise library")
        if (
            self.noise_cache is not None
            and self.noise_cache_sample_rate == int(sample_rate)
            and self.noise_cache[0].device == device
        ):
            return

        started = time.perf_counter()
        cached = []
        total_bytes = 0
        for noise_path in self.noise_files:
            noise_wav, noise_sr = torchaudio.load(str(noise_path))
            if noise_sr != sample_rate:
                noise_wav = torchaudio.functional.resample(
                    noise_wav, noise_sr, sample_rate
                )
            noise_wav = noise_wav.contiguous().to(device)
            cached.append(noise_wav)
            total_bytes += noise_wav.numel() * noise_wav.element_size()

        torch.cuda.synchronize(device)
        self.noise_cache = cached
        self.noise_cache_names = [path.name for path in self.noise_files]
        self.noise_cache_sample_rate = int(sample_rate)
        elapsed = time.perf_counter() - started
        print(
            "Gramophone GPU preload ready: "
            f"{len(cached):,} files, {total_bytes / (1024 ** 3):.2f} GiB, "
            f"{elapsed:.2f}s on {device}.",
            flush=True,
        )

    def __call__(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply all five degradation stages to a single audio clip.

        Args:
            audio       : (C, T) float32 tensor — single clip, NOT a batch
            sample_rate : sample rate of the audio in Hz (e.g. 44100)

        Returns:
            degraded : (C, T) float32 tensor — same shape as input
        """
        audio = audio.clone()  # never modify the original clean tensor in-place
        n_bins = audio.shape[-1] // 2 + 1

        # ── Stage 1: Random zero-phase EQ₁ ────────────────────────────────────
        # Pre-nonlinearity shaping using the Wiener–Hammerstein EQ₁ block.
        # Mean gain profile at octave nodes [40…20480 Hz]:
        #   [−22, −14, −7, −3, −1, −1, −2, −8, −22, −40] dB
        # Heavy bass loss → modest midband → strong treble decay.
        if self.filter_mode == "segment_fft":
            eq1_mask = _make_eq_mask(
                self.cfg["eq1"], sample_rate, audio.device, n_bins=n_bins
            )
            audio = _apply_segment_fft_mask(audio, eq1_mask)
        else:
            eq1_mask = _make_eq_mask(self.cfg["eq1"], sample_rate, audio.device)
            audio = _apply_fft_mask(audio, eq1_mask)

        # ── Stage 2: Scaled tanh nonlinearity ─────────────────────────────────
        # Wiener–Hammerstein static nonlinearity between the two EQ blocks.
        # Loud signal peaks are compressed; quiet passages stay nearly linear.
        # Parameters: a ~ N(2.0, 0.7) ∈ [0.75, 4.0],  wet ~ N(0.18, 0.07) ∈ [0.05, 0.35]
        audio = self._apply_nonlinearity(audio, self.cfg["nonlinearity"])

        # ── Stage 3: Random zero-phase EQ₂ ────────────────────────────────────
        # Post-nonlinearity coloration: independently sampled from EQ₁ but
        # slightly stronger (wider std, lower mean) to shape the harmonics and
        # overtone structure introduced by the saturator.
        # Mean gain profile: [−26, −17, −8, −4, −1, −1, −3, −12, −30, −55] dB
        #
        # ── Stage 4: Final bandpass ────────────────────────────────────────────
        # Hard bandwidth enforcement *outside* the EQ-node range, after the full
        # WH coloration chain.  Low cut removes subsonic rumble (< ~120 Hz);
        # high cut enforces the gramophone bandwidth limit (N(3000 Hz, 300 Hz)).
        #
        # EQ₂ and bandpass are adjacent linear filters, so their magnitude
        # responses can be multiplied and applied in one FFT pass.
        if self.filter_mode == "segment_fft":
            eq2_mask = _make_eq_mask(
                self.cfg["eq2"], sample_rate, audio.device, n_bins=n_bins
            )
            bp_mask = _make_bandpass_mask(
                self.cfg["bandpass"], sample_rate, audio.device, n_bins=n_bins
            )
            audio = _apply_segment_fft_mask(audio, eq2_mask * bp_mask)
        else:
            eq2_mask = _make_eq_mask(self.cfg["eq2"], sample_rate, audio.device)
            bp_mask = _make_bandpass_mask(self.cfg["bandpass"], sample_rate, audio.device)
            audio = _apply_fft_mask(audio, eq2_mask * bp_mask)

        # ── Stage 5: Gramophone record noise ──────────────────────────────────
        # Adds real 78-rpm surface noise at SNR ~ N(11, 4.5) dB ∈ [2, 20] dB,
        # following Moliner's historical-denoising training setup.
        # Requires `corruption.gramophone_noise.noise_dir` to be set in config.
        audio = self._apply_gramophone_noise(
            audio, sample_rate, self.cfg["gramophone_noise"]
        )

        return audio

    def corrupt_batch(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply the degradation chain to a whole batch.

        This is the optimized training path. With filter_mode=segment_fft it
        applies EQ₁ and combined EQ₂*bandpass as batched full-segment FFT
        filters on audio.device. If the trainer moves audio to CUDA before
        calling this method, the expensive filter stages run on the GPU.

        Gramophone noise still loads/crops one random file per item; that is
        intentionally left as-is until a separate preload/cache change.
        """
        if audio.ndim != 3:
            raise ValueError(f"Expected batch audio shape (B, C, T), got {tuple(audio.shape)}")
        if self.filter_mode != "segment_fft":
            return torch.stack([self(a, sample_rate) for a in audio])

        audio = audio.clone()
        B, _, T = audio.shape
        n_bins = T // 2 + 1
        device = audio.device
        eq1_masks = torch.stack([
            _make_eq_mask(self.cfg["eq1"], sample_rate, device, n_bins=n_bins)
            for _ in range(B)
        ])
        audio = _apply_segment_fft_mask_batch(audio, eq1_masks)

        audio = self._apply_nonlinearity_batch(audio, self.cfg["nonlinearity"])

        post_masks = torch.stack([
            _make_eq_mask(self.cfg["eq2"], sample_rate, device, n_bins=n_bins)
            * _make_bandpass_mask(self.cfg["bandpass"], sample_rate, device, n_bins=n_bins)
            for _ in range(B)
        ])
        audio = _apply_segment_fft_mask_batch(audio, post_masks)

        return torch.stack([
            self._apply_gramophone_noise(a, sample_rate, self.cfg["gramophone_noise"])
            for a in audio
        ])

    def corrupt_batch_seeded(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        pair_seeds: list[int],
    ) -> torch.Tensor:
        """Apply the fast five-stage chain with one independent seed per item.

        Unlike resetting the process RNG once for a whole batch, this makes an
        output invariant to rank assignment, batching, restart position, and
        the neighboring items processed alongside it.
        """
        if audio.ndim != 3:
            raise ValueError(
                f"Expected batch audio shape (B, C, T), got {tuple(audio.shape)}"
            )
        if len(pair_seeds) != audio.shape[0]:
            raise ValueError(
                f"Expected {audio.shape[0]} pair seeds, got {len(pair_seeds)}"
            )
        if self.filter_mode != "segment_fft":
            outputs = []
            for item, seed in zip(audio, pair_seeds):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(seed))
                    python_rng = random.Random(int(seed))
                    outputs.append(
                        self._corrupt_single_with_rng(
                            item, sample_rate, torch.default_generator, python_rng
                        )
                    )
            return torch.stack(outputs)

        audio = audio.clone()
        batch_size, _, samples = audio.shape
        n_bins = samples // 2 + 1
        device = audio.device
        generators = []
        python_generators = []
        for seed in pair_seeds:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            generators.append(generator)
            python_generators.append(random.Random(int(seed)))

        eq1_masks = torch.stack(
            [
                _make_eq_mask(
                    self.cfg["eq1"], sample_rate, device,
                    n_bins=n_bins, generator=generator,
                )
                for generator in generators
            ]
        )
        audio = _apply_segment_fft_mask_batch(audio, eq1_masks)
        audio = self._apply_nonlinearity_batch_seeded(
            audio, self.cfg["nonlinearity"], generators
        )
        post_masks = torch.stack(
            [
                _make_eq_mask(
                    self.cfg["eq2"], sample_rate, device,
                    n_bins=n_bins, generator=generator,
                )
                * _make_bandpass_mask(
                    self.cfg["bandpass"], sample_rate, device,
                    n_bins=n_bins, generator=generator,
                )
                for generator in generators
            ]
        )
        audio = _apply_segment_fft_mask_batch(audio, post_masks)
        return torch.stack(
            [
                self._apply_gramophone_noise(
                    item,
                    sample_rate,
                    self.cfg["gramophone_noise"],
                    torch_generator=generator,
                    python_rng=python_rng,
                )
                for item, generator, python_rng in zip(
                    audio, generators, python_generators
                )
            ]
        )

    def _corrupt_single_with_rng(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        torch_generator: torch.Generator,
        python_rng: random.Random,
    ) -> torch.Tensor:
        """Seedable legacy-filter implementation used only outside the fast path."""
        audio = audio.clone()
        eq1_mask = _make_eq_mask(
            self.cfg["eq1"], sample_rate, audio.device,
            generator=torch_generator,
        )
        audio = _apply_fft_mask(audio, eq1_mask)
        audio = self._apply_nonlinearity_seeded(
            audio, self.cfg["nonlinearity"], torch_generator
        )
        eq2_mask = _make_eq_mask(
            self.cfg["eq2"], sample_rate, audio.device,
            generator=torch_generator,
        )
        bandpass_mask = _make_bandpass_mask(
            self.cfg["bandpass"], sample_rate, audio.device,
            generator=torch_generator,
        )
        audio = _apply_fft_mask(audio, eq2_mask * bandpass_mask)
        return self._apply_gramophone_noise(
            audio,
            sample_rate,
            self.cfg["gramophone_noise"],
            torch_generator=torch_generator,
            python_rng=python_rng,
        )

    @torch.no_grad()
    def corrupt_batch_with_stages(
        self, audio: torch.Tensor, sample_rate: int
    ) -> dict[int, torch.Tensor]:
        """Apply one batched draw and retain every cumulative stage output.

        The random draws and Stage-5 result intentionally match
        :meth:`corrupt_batch` when both methods start from identical Python and
        Torch RNG states.  Stage 4 is evaluated using the production fast path
        (the combined EQ2/bandpass mask), while Stage 3 is an additional
        diagnostic application of the already-sampled EQ2 mask.  Consequently,
        observing Stage 3 does not perturb Stage 4 or Stage 5.

        Returns a mapping whose integer keys 1 through 5 mean ``EQ1``,
        ``EQ1+tanh``, ``EQ1+tanh+EQ2``, ``EQ1+tanh+EQ2+bandpass``, and the full
        chain including gramophone noise, respectively.
        """
        if audio.ndim != 3:
            raise ValueError(
                f"Expected batch audio shape (B, C, T), got {tuple(audio.shape)}"
            )
        if self.filter_mode != "segment_fft":
            raise ValueError(
                "Cumulative batched stage capture requires "
                "corruption.filter_mode=segment_fft"
            )

        clean = audio.clone()
        batch_size, _, samples = clean.shape
        n_bins = samples // 2 + 1
        device = clean.device

        eq1_masks = torch.stack(
            [
                _make_eq_mask(
                    self.cfg["eq1"], sample_rate, device, n_bins=n_bins
                )
                for _ in range(batch_size)
            ]
        )
        stage1 = _apply_segment_fft_mask_batch(clean, eq1_masks)
        stage2 = self._apply_nonlinearity_batch(
            stage1, self.cfg["nonlinearity"]
        )

        eq2_masks = []
        bandpass_masks = []
        for _ in range(batch_size):
            eq2_masks.append(
                _make_eq_mask(
                    self.cfg["eq2"], sample_rate, device, n_bins=n_bins
                )
            )
            bandpass_masks.append(
                _make_bandpass_mask(
                    self.cfg["bandpass"], sample_rate, device, n_bins=n_bins
                )
            )
        eq2_masks = torch.stack(eq2_masks)
        bandpass_masks = torch.stack(bandpass_masks)

        # Stage 3 is retained for analysis. Stage 4 starts from Stage 2 and
        # applies the mathematically equivalent combined production mask so
        # Stage 5 remains exactly aligned with corrupt_batch().
        stage3 = _apply_segment_fft_mask_batch(stage2, eq2_masks)
        stage4 = _apply_segment_fft_mask_batch(
            stage2, eq2_masks * bandpass_masks
        )
        stage5 = torch.stack(
            [
                self._apply_gramophone_noise(
                    item, sample_rate, self.cfg["gramophone_noise"]
                )
                for item in stage4
            ]
        )
        return {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5}

    # ───────────────────────────────────── private stage implementations ──────

    @staticmethod
    def _apply_nonlinearity(audio: torch.Tensor, cfg: dict) -> torch.Tensor:
        """Scaled tanh soft-saturation:  y = (1 − wet)·x + wet·tanh(a·x) / a

        Why divide by a?
            tanh(a·x)/a ≈ x  for small |x|  (Taylor: tanh(z)/z → 1 as z → 0)
        So the DC gain is ≈ 1 at low amplitudes — the nonlinearity is a pure
        compressor/saturator that does not drop the overall level.  Without /a,
        the saturated signal would be noticeably quieter than the dry signal.

        The dry/wet mix  (1 − wet)·dry + wet·saturated  lets the coder control
        the saturation intensity continuously from almost-linear (wet ≈ 0) to
        strong saturation (wet ≈ 0.35).

        Eloi Moliner recommended this scaled tanh over a spline nonlinearity
        because splines can create excessive distortion and aliasing when their
        control points are randomised freely.

        Parameters drawn fresh each call:
            a   ~ N(2.0,  0.7)  clipped to [0.75, 4.0]   saturation strength
            wet ~ N(0.18, 0.07) clipped to [0.05, 0.35]  dry/wet mix ratio

        Args:
            audio : (C, T) float32 tensor
            cfg   : dict with a_mean, a_std, a_min, a_max,
                         wet_mean, wet_std, wet_min, wet_max

        Returns:
            (C, T) float32 tensor — saturated audio
        """
        a   = _sample_gaussian(cfg["a_mean"],   cfg["a_std"],   cfg["a_min"],   cfg["a_max"])
        wet = _sample_gaussian(cfg["wet_mean"], cfg["wet_std"], cfg["wet_min"], cfg["wet_max"])

        saturated = torch.tanh(a * audio) / a
        return (1.0 - wet) * audio + wet * saturated

    @staticmethod
    def _apply_nonlinearity_batch(audio: torch.Tensor, cfg: dict) -> torch.Tensor:
        """Vectorized scaled tanh soft-saturation for (B, C, T) audio."""
        B = audio.shape[0]
        a = torch.empty(B, 1, 1, device=audio.device, dtype=audio.dtype).normal_(
            cfg["a_mean"], cfg["a_std"]
        ).clamp_(cfg["a_min"], cfg["a_max"])
        wet = torch.empty(B, 1, 1, device=audio.device, dtype=audio.dtype).normal_(
            cfg["wet_mean"], cfg["wet_std"]
        ).clamp_(cfg["wet_min"], cfg["wet_max"])
        saturated = torch.tanh(a * audio) / a
        return (1.0 - wet) * audio + wet * saturated

    @staticmethod
    def _apply_nonlinearity_seeded(
        audio: torch.Tensor,
        cfg: dict,
        generator: torch.Generator,
    ) -> torch.Tensor:
        a = _sample_gaussian(
            cfg["a_mean"], cfg["a_std"], cfg["a_min"], cfg["a_max"], generator
        )
        wet = _sample_gaussian(
            cfg["wet_mean"], cfg["wet_std"], cfg["wet_min"], cfg["wet_max"], generator
        )
        saturated = torch.tanh(a * audio) / a
        return (1.0 - wet) * audio + wet * saturated

    @classmethod
    def _apply_nonlinearity_batch_seeded(
        cls,
        audio: torch.Tensor,
        cfg: dict,
        generators: list[torch.Generator],
    ) -> torch.Tensor:
        a = torch.tensor(
            [
                _sample_gaussian(
                    cfg["a_mean"], cfg["a_std"], cfg["a_min"], cfg["a_max"], generator
                )
                for generator in generators
            ],
            device=audio.device,
            dtype=audio.dtype,
        ).view(-1, 1, 1)
        wet = torch.tensor(
            [
                _sample_gaussian(
                    cfg["wet_mean"], cfg["wet_std"], cfg["wet_min"], cfg["wet_max"], generator
                )
                for generator in generators
            ],
            device=audio.device,
            dtype=audio.dtype,
        ).view(-1, 1, 1)
        saturated = torch.tanh(a * audio) / a
        return (1.0 - wet) * audio + wet * saturated


    def _apply_gramophone_noise(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        cfg: dict,
        torch_generator: torch.Generator | None = None,
        python_rng: random.Random | None = None,
    ) -> torch.Tensor:
        """Mix in a real gramophone / 78-rpm surface noise clip.

        A random WAV file is chosen uniformly from `noise_dir`, resampled to
        match the audio sample rate if needed, then tiled or randomly cropped
        to the exact clip length.  It is scaled so the signal-to-noise ratio
        equals the randomly drawn target SNR.

        SNR ~ N(11.0, 4.5) dB, clipped to [2.0, 20.0] dB.
        This matches Moliner's historical-denoising training pairs, where SNR
        was sampled uniformly from 2–20 dB with random gain scaling −6 to +4 dB.

        ⚠  PLACEHOLDER — requires the noise dataset to be downloaded:
           1. Download from:
                scripts/download_gramophone_noise.sh
           2. Set `corruption.gramophone_noise.noise_dir` in config/default.yaml
              to the local path of the directory containing the .wav files.

        Args:
            audio       : (C, T) float32 tensor
            sample_rate : sample rate of `audio` in Hz
            cfg         : the `gramophone_noise` config sub-dict

        Returns:
            (C, T) float32 tensor — audio + scaled surface noise

        Raises:
            NotImplementedError : if no noise files were found at init time
        """
        if not self.noise_files:
            raise NotImplementedError(
                "Stage 5 (gramophone noise) requires\n"
                "  corruption.gramophone_noise.noise_dir\n"
                "to be set in config/default.yaml.\n"
                "Download the dataset from:\n"
                "  scripts/download_gramophone_noise.sh"
            )

        # One flag controls both Trainer stage timing and this fine-grained
        # corruption timing. With profiling off, do not synchronize or print;
        # the corruption itself remains unchanged.
        profile_enabled = _env_flag("RESTOR_PROFILE_STEPS", False)
        profile_device = audio.device
        if profile_enabled and profile_device.type == "cuda":
            torch.cuda.synchronize(profile_device)
        t0 = time.perf_counter()
        last_t = t0

        def mark(stage: str, tensor: torch.Tensor | None = None, extra: str = ""):
            nonlocal last_t
            if not profile_enabled:
                return
            if profile_device.type == "cuda":
                torch.cuda.synchronize(profile_device)
            now = time.perf_counter()
            tensor_device = ""
            if tensor is not None:
                tensor_device = f" tensor_device={tensor.device}"
            extra_part = f" {extra}" if extra else ""
            print(
                f"[profile:gramophone_noise] {stage}="
                f"{(now - last_t) * 1000.0:.1f}ms"
                f" total={(now - t0) * 1000.0:.1f}ms"
                f" audio_device={audio.device}{tensor_device}{extra_part}",
                flush=True,
            )
            last_t = now

        python_rng = python_rng or random
        snr_db = _sample_gaussian(
            cfg["snr_mean_db"], cfg["snr_std_db"],
            cfg["snr_min_db"],  cfg["snr_max_db"],
            generator=torch_generator,
        )
        mark("sample_snr", extra=f"snr_db={snr_db:.2f}")

        # Select a resident GPU tensor when preloading is enabled. Otherwise,
        # preserve the original per-sample file-load path.
        if self.noise_cache is not None:
            noise_idx = python_rng.randrange(len(self.noise_cache))
            noise_wav = self.noise_cache[noise_idx]
            noise_sr = self.noise_cache_sample_rate
            noise_name = self.noise_cache_names[noise_idx]
            mark("choose_cached_noise", noise_wav, extra=f"path={noise_name}")
        else:
            noise_path = python_rng.choice(self.noise_files)
            mark("choose_noise_file", extra=f"path={noise_path.name}")
            noise_wav, noise_sr = torchaudio.load(str(noise_path))  # (C_n, T_n)
            noise_wav = noise_wav.to(audio.device, non_blocking=True)
            mark(
                "torchaudio_load",
                noise_wav,
                extra=f"noise_sr={noise_sr} shape={tuple(noise_wav.shape)}",
            )

        # Resample if the noise file has a different sample rate than the audio.
        if noise_sr != sample_rate:
            noise_wav = torchaudio.functional.resample(noise_wav, noise_sr, sample_rate)
            mark(
                "resample",
                noise_wav,
                extra=f"{noise_sr}->{sample_rate} shape={tuple(noise_wav.shape)}",
            )
        else:
            mark("resample_skipped", noise_wav, extra=f"noise_sr={noise_sr}")

        C, T = audio.shape

        # Match channel count.
        # Mono noise → broadcast to all audio channels (most common case).
        # Multi-channel noise → trim to the required number of channels.
        if noise_wav.shape[0] == 1:
            noise_wav = noise_wav.expand(C, -1)
        else:
            noise_wav = noise_wav[:C]
        mark("match_channels", noise_wav, extra=f"shape={tuple(noise_wav.shape)}")

        # Tile noise if it is shorter than the audio clip.
        if noise_wav.shape[1] < T:
            repeats = (T // noise_wav.shape[1]) + 1
            noise_wav = noise_wav.repeat(1, repeats)
            mark("tile_short_noise", noise_wav, extra=f"repeats={repeats}")
        else:
            mark("tile_skipped", noise_wav)

        # Randomly crop so the same noise file does not always start at time 0.
        start = python_rng.randint(0, noise_wav.shape[1] - T)
        mark("choose_crop", noise_wav, extra=f"start={start} T={T}")

        noise_wav = noise_wav[:, start : start + T].to(audio.device)
        mark("crop_to_device", noise_wav, extra=f"shape={tuple(noise_wav.shape)}")

        # Scale noise to achieve the target SNR.
        # SNR = 10·log₁₀(P_signal / P_noise)  →  P_noise = P_signal / 10^(SNR/10)
        sig_power         = audio.pow(2).mean().clamp(min=1e-10)
        mark("signal_power", sig_power)
        noise_power       = noise_wav.pow(2).mean().clamp(min=1e-10)
        mark("noise_power", noise_power)
        target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        scale             = (target_noise_power / noise_power).sqrt()
        mark("scale", scale)

        out = audio + scale * noise_wav
        mark("mix", out)
        return out
