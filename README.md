# End-to-End Historical Music Restoration in Latent Space

Official implementation and evaluation resources for **End-to-End Historical Music Restoration in Latent Space**.

This repository studies historical music restoration as conditional flow matching in the continuous latent space of the frozen [SAME-L](https://huggingface.co/stabilityai/SAME-L) audio autoencoder. The proposed 40M-parameter model, **SAMECFM**, maps degraded historical-audio latents toward clean musical-audio latents and decodes the restored representation at 44.1 kHz.

> **Release status:** implementation, current paper PDF, interactive demo,
> aggregate subjective results, published test set, and SAMECFM-40M checkpoint
> are included. The arXiv identifier is forthcoming.

**[Interactive demo](https://full-mix-historical-music-restorati.vercel.app)** · **[Paper PDF](paper/full_mix_historical_music_restoration.pdf)** · **[Published dataset](https://doi.org/10.5281/zenodo.22737610)**

## Method

```text
historical audio -> frozen SAME-L encoder -> degraded latent
                                                |
                                      conditional flow matching
                                      1-D DiT velocity network
                                                |
clean estimate  <- frozen SAME-L decoder <- restored latent
```

The principal training configuration uses:

- Full-Orchestra recordings plus instrumental sections, denoted Full-Orchestra + Section (FOS);
- a leak-free song-level train/validation split;
- five-second, 44.1-kHz training windows;
- whole-song loudness normalization before windowing;
- a five-stage historical-recording degradation model;
- a 40M-parameter conditional flow-matching DiT in SAME-L space.

The CFM follows the straight path from unit Gaussian noise to the clean
normalized latent and minimizes velocity MSE in latent space. Per-channel
SAME-L statistics are essential: they prevent high-variance channels from
dominating the objective and keep the target scale compatible with the
Gaussian source. Latents must be unnormalized before the frozen SAME-L
decoder; omitting that step causes a large quality loss. Mel-spectrogram MSE
is used for analysis, not to train SAMECFM-40M.

### Synthetic historical degradation

Every clean training window passes through the same ordered five-stage chain;
there is no probability gating. Gaussian draws are clipped to the stated
ranges. The main five-stage contribution is implemented in
[`restor/corruption.py`](restor/corruption.py); all sampled parameter
distributions are in
[`config/samecfm40_fos.yaml`](config/samecfm40_fos.yaml).

| Stage | Operation | Sampling parameters |
|---|---|---|
| 1 | Zero-phase EQ 1 | Nodes: 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480 Hz; mean gains: −9.0, −4.9, −4.9, 0.0, 0.0, −3.1, −1.1, −8.6, −2.1, 0.0 dB; independent standard deviation 14.7 dB; clipped to [−80, 0] dB |
| 2 | Scaled-tanh nonlinearity | Drive `a ~ N(2.2, 0.9)`, clipped to [0.75, 4.5]; wet mix `w ~ N(0.18, 0.09)`, clipped to [0.03, 0.40]; `y=(1−w)x+w tanh(ax)/a` |
| 3 | Zero-phase EQ 2 | Same nodes, standard deviation, and clipping as EQ 1; mean gains: −3.0, −4.9, −4.9, 0.0, 0.0, −3.1, −1.1, −8.6, −2.1, 0.0 dB |
| 4 | Smooth band-pass | Low cutoff `N(100,50)` Hz clipped to [40,250]; high cutoff `N(3200,850)` Hz clipped to [2000,5500]; low slope `N(18,8)` dB/oct clipped to [6,48]; high slope `N(30,10)` dB/oct clipped to [12,60] |
| 5 | Real gramophone surface noise | Random segment from the Gramophone Record Noise Dataset; SNR `N(11,4.5)` dB clipped to [2,20] dB |

The two EQ curves are independently sampled and use log-frequency
interpolation. They form a Wiener–Hammerstein sequence around the static
nonlinearity. Filtering is zero phase to retain temporal alignment between
each degraded input and its clean target. White-noise augmentation is not used.

## Repository layout

```text
.
├── assets/          # Paper-ready qualitative waveform/spectrogram examples
├── checkpoints/     # Downloaded model checkpoint destination
├── config/          # Public SAMECFM-40M configuration
├── demo/            # Static Next.js paper site and synchronized audio examples
├── paper/           # Paper PDF
├── restor/          # Model, corruption, training, and inference implementation
├── results/         # Aggregate, anonymous evaluation results
├── scripts/         # Dataset download, four-GPU training, and inference launchers
├── main.py          # Training and inference entry point
└── pyproject.toml
```

No private listening-test responses, credentials, training data, inference corpora, or model checkpoints are stored in this repository.

## Interactive demo

The static site under [`demo/`](demo/) contains six synchronized historical comparisons, the method diagram, and aggregate subjective results. Run it locally with Node.js 22:

```bash
cd demo
npm ci
npm run dev
```

For Vercel, import this repository and set the project **Root Directory** to `demo`. No server, environment variables, or runtime inference are required.

## Installation

Python 3.11 and a CUDA-capable PyTorch environment are recommended.

```bash
git clone https://github.com/stevencho24/End-to-End_historical_music_restoration.git
cd End-to-End_historical_music_restoration
pip install -e .
```

SAME-L is distributed under the Stability AI Community License. Review and accept its terms before use.

## Checkpoint setup

Checkpoint binaries are kept outside Git and distributed as a GitHub Release
asset, following the approach used by BEHM-GAN. The asset is also available on
the [v1.0.0 release page](https://github.com/stevencho24/End-to-End_historical_music_restoration/releases/tag/v1.0.0).
Download and SHA-256 verify it with:

```bash
bash prepare_data.sh
```

This creates:

```text
checkpoints/samecfm_40m_fos.pt
```

See [`checkpoints/README.md`](checkpoints/README.md) for the release convention.
Its SHA-256 is
`2b13d250a66e3c640a52d3b5969951fd6d7a1336b5b5bd97770b28c5707f7ae3`.
The checkpoint contains the architecture configuration, EMA denoiser,
training-set latent mean/std, and all states needed for inference.

## Inference

The paper evaluation uses ten uniform Euler steps, CFG scale 1.0, and seed 42.
For one arbitrary audio file:

```bash
scripts/infer_samecfm40_fos.sh path/to/historical_input.wav output/restored
```

For a directory, the same launcher recursively restores every supported audio
file while preserving the input subdirectories:

```bash
scripts/infer_samecfm40_fos.sh path/to/input_dataset output/samecfm40_fos
```

Set `CHECKPOINT`, `DEVICE`, `CFM_STEPS`, `SEED`, `CHUNK_SEC`, or `OVERLAP` to
override defaults. Arbitrary-length input is processed with overlap-add and is
converted to 44.1-kHz mono before SAME-L encoding.

[`restor/inference.py`](restor/inference.py) is the implementation behind the
public `main.py infer` command. `trainer.py` has a matching internal sampler
for validation/TensorBoard audio, which is why research runs may appear to
perform inference from the Trainer without importing `inference.py`.

## Training

The paper's Full-Orchestra + Section configuration is
[`config/samecfm40_fos.yaml`](config/samecfm40_fos.yaml). It is the template
for the codec, CFM/DiT, optimizer, inference schedule, and five-stage
degradation. The exact final launch used four DDP ranks, a per-GPU batch of 24
(global batch 96), and precomputed five-second latent pairs:

```bash
python main.py precompute \
  --source-dir data/public_classical_orchestral_plus_sections \
  --manifest data/public_classical_orchestral_plus_sections/MANIFEST.tsv \
  --noise-dir data/gramophone_record_noise \
  --output-root data/fos_precomputed
```

See [`docs/precompute.md`](docs/precompute.md) for source preparation, manifest
schema, defaults, and split guarantees. Then run:

```bash
PRECOMPUTED_ROOT=data/fos_precomputed \
FOS_CLEAN_ROOT=data/public_classical_orchestral_plus_sections \
scripts/train_samecfm40_fos_4gpu.sh
```

`PRECOMPUTED_ROOT` must contain `train/`, `validate/`, and `ground_truth/`
directories from the leak-free song-level FOS split. The removed
`StemMixDataset` path belonged to early arbitrary-stem-combination experiments;
the final submission trains only from the fixed full-orchestra and section
mixtures. Use `RESUME=1` to resume the same experiment.

The published Zenodo archive below is the unpaired historical **evaluation**
set, not the clean FOS training corpus and not the paired latent cache.

## Evaluation

The paper evaluates restoration on:

- an unpaired historical Internet Archive test set split into Full-Orchestra and Light Orchestra;
- a synthetic paired test set with aligned clean references;
- objective perceptual, spectral, embedding-distribution, embedding-similarity, and fidelity metrics;
- MOS-Quality and reference-based MOS-Preservation listening tests.

Aggregate subjective results are provided under [`results/`](results/). The sensitivity-analysis folder retains listeners whose mean score over four clean ground-truth quality items is at least 4. Individual listener data are intentionally excluded.

## Dataset

The published historical unpaired test set contains 149 full-length recordings totaling **9.30 hours**: 70 Full-Orchestra and 79 Light Orchestra items. It is available from Zenodo at DOI [`10.5281/zenodo.22737610`](https://doi.org/10.5281/zenodo.22737610).

Download, verify, and unpack it under the directory name expected by the
public inference launcher:

```bash
scripts/download_zenodo_test_set.sh data/historical_unpaired_test
scripts/infer_samecfm40_fos.sh \
  data/historical_unpaired_test \
  output/zenodo_samecfm40_fos
```

The downloader fetches the official `audio_orchestra_70.zip`,
`audio_light_orchestra_79.zip`, metadata, rights, and checksum files directly
from Zenodo record 22737610 and checks the published SHA-256 values before
unpacking. A different input dataset may be substituted in the second command.

## Checkpoints and audio demos

The small public listening examples are already bundled in [`demo/`](demo/),
so a separate `examples/` directory is unnecessary. The model weight is a
versioned GitHub Release asset rather than part of ordinary Git history.

## Limitations

- The system is designed for instrumental historical classical recordings represented by the paper's training degradations.
- Restoration quality may decline for degradations or musical domains outside the training distribution.
- SAME-L licensing is separate from this repository's MIT-licensed code.
- Generative restoration can alter fine musical details; restored audio should not be treated as an archival ground-truth reconstruction.

## Citation

If you use this work, please cite the paper and the accompanying dataset. Final bibliographic metadata will replace the placeholder below when the arXiv record is available.

```bibtex
@article{cho2026endtoend,
  title   = {End-to-End Historical Music Restoration in Latent Space},
  author  = {Cho, Steven and Koo, Junghyun and Lafargue, Raphael and Dhyani, Tushar and Moliner, Eloi and Mitsufuji, Yuki},
  year    = {2026},
  note    = {arXiv preprint; identifier forthcoming}
}
```

## License

The original code in this repository is released under the [MIT License](LICENSE). Third-party models, datasets, and evaluation packages retain their own licenses.

## Contact

Steven Cho — [ORCID 0009-0008-0040-9312](https://orcid.org/0009-0008-0040-9312)
