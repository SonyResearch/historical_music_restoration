# Degradation and latent-pair precompute

The training launcher reads normalized SAME-L pairs from `train/` and
`validate/`, plus clean WAV windows from `ground_truth/`. Generate all three
with the public `precompute` command; the same five-stage corruption class and
configuration used by training are called directly.

## Inputs

Prepare a flat directory of mono or stereo WAV files at 44.1 kHz. Normalize
each complete source recording once toward -23 LUFS with peak safety before
placing it there. Windows are intentionally not loudness-normalized
individually, so within-song dynamics are preserved.

Download the Gramophone Record Noise Dataset separately and pass the folder
containing its WAV files with `--noise-dir`.

For FOS data, provide a tab-separated manifest with one row per WAV and these
columns:

```text
recording_id\tkind\tdataset\tsong\tfamily\toutput_file
piece_001\tfull_mix\tmy_dataset\tpiece_001\tfull_mix\tpiece_001_full.wav
piece_001\tsection_mix\tmy_dataset\tpiece_001\tstrings\tpiece_001_strings.wav
```

The manifest keeps the full mix and every aligned section view in the same
split. Validation windows are promoted together across views, and every
overlapping non-validation window is discarded. Without a manifest, every WAV
is treated as an independent recording.

## Run

```bash
python main.py precompute \
  --source-dir data/public_classical_orchestral_plus_sections \
  --manifest data/public_classical_orchestral_plus_sections/MANIFEST.tsv \
  --noise-dir data/gramophone_record_noise \
  --output-root data/fos_precomputed
```

Defaults reproduce the paper cache: five-second windows, 0.5-second hop,
20 independently sampled degradations per clean window, train-window SAME-L
channel statistics, seed 42, and the full five-stage degradation in
`config/samecfm40_fos.yaml`. The command requires CUDA because SAME-L and the
batched corruption pipeline run on the GPU.

The output is directly compatible with:

```bash
PRECOMPUTED_ROOT=data/fos_precomputed \
FOS_CLEAN_ROOT=data/public_classical_orchestral_plus_sections \
scripts/train_samecfm40_fos_4gpu.sh
```

Precompute refuses nonempty output directories. Move or remove an incomplete
cache before restarting. A successful run writes `_SUCCESS` last.
