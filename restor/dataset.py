import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


class StemMixDataset(Dataset):
    """Loads song folders containing individual stem files"""

    def __init__(self, song_dirs, cfg, sample_rate):
        self.sample_rate = sample_rate
        self.segment_samples = int(cfg["segment_duration"] * sample_rate)
        self.min_stems = cfg.get("min_stems", 1)
        self.max_stems = cfg.get("max_stems")
        self.samples_per_song = cfg.get("samples_per_song", 1)
        self.mode = cfg.get("mode", "real_time")  # "precompute" or "real_time"

        self.songs = []
        for d in song_dirs:
            stems = sorted(
                f for f in os.listdir(d)
                if os.path.splitext(f)[1].lower() in AUDIO_EXTS
            )
            if len(stems) < self.min_stems:
                continue
            info = torchaudio.info(os.path.join(d, stems[0]))
            dur = int(info.num_frames * sample_rate / info.sample_rate)
            if dur >= self.segment_samples:
                self.songs.append({
                    "dir": d,
                    "stems": stems,
                    "orig_sr": info.sample_rate,
                    "orig_frames": info.num_frames,
                })

    def __len__(self):
        if self.mode == "precompute":
            song = self.songs[0]
            hop_samples = self.segment_samples // 2
            return 1 + (song["orig_frames"] - self.segment_samples) // hop_samples

        return len(self.songs) * self.samples_per_song

    def name(self):
        #return name 
        return self.songs[0]["stems"][0]

    def __getitem__(self, idx):
        if self.mode == "real_time":
            song = self.songs[idx % len(self.songs)]

            # random stem count
            n_avail = len(song["stems"])
            hi = min(self.max_stems or n_avail, n_avail)
            n_stems = random.randint(self.min_stems, hi)
            selected = random.sample(song["stems"], n_stems)

            # random offset (in original sample rate)
            orig_sr = song["orig_sr"]
            if orig_sr == self.sample_rate:
                max_start = song["orig_frames"] - self.segment_samples
                start = random.randint(0, max(0, max_start))
                num_frames = self.segment_samples
            else:
                orig_seg = int(self.segment_samples * orig_sr / self.sample_rate) + 1
                max_start = song["orig_frames"] - orig_seg
                start = random.randint(0, max(0, max_start))
                num_frames = orig_seg

            # load & sum
            mix = torch.zeros(1, self.segment_samples)
            for stem in selected:
                path = os.path.join(song["dir"], stem)
                audio, sr = torchaudio.load(path, frame_offset=start, num_frames=num_frames)
                if sr != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
                if audio.shape[0] > 1:
                    audio = audio.mean(0, keepdim=True)
                if audio.shape[-1] > self.segment_samples:
                    audio = audio[..., :self.segment_samples]
                elif audio.shape[-1] < self.segment_samples:
                    raise ValueError("Precompute window is shorter than segment duration")
                mix = mix + audio

            # peak-normalise to [-1, 1]
            peak = mix.abs().max()
            if peak > 1.0:
                mix = mix / peak

            return mix  # (1, T)
        
        elif self.mode == "precompute":
            song = self.songs[0]
            selected = song["stems"][0]  

            # random offset (in original sample rate)
            orig_sr = song["orig_sr"]
            if orig_sr == self.sample_rate: 
                start = idx * (self.segment_samples // 2)  # 50% overlap
                num_frames = self.segment_samples

                # load
                path = os.path.join(song["dir"], selected)
                audio, sr = torchaudio.load(path, frame_offset=start, num_frames=num_frames)
                if sr != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
                if audio.shape[0] > 1:
                    audio = audio.mean(0, keepdim=True)
                if audio.shape[-1] > self.segment_samples:
                    audio = audio[..., :self.segment_samples]
                elif audio.shape[-1] < self.segment_samples:
                    raise ValueError("Precompute window is shorter than segment duration")

                # peak-normalise to [-1, 1]
                peak = audio.abs().max()
                if peak > 1.0:
                    audio = audio / peak
                return audio  # (1, T)

            else:
                raise ValueError("Precompute mode only supports songs with the same sample rate as 44.1 kHz.")

            
