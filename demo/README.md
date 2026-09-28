# Historical Music Restoration Demo

Companion audio-demonstration website for **End-to-End Historical Music Restoration in Latent Space**. It lives inside the official paper repository and contains precomputed five-second comparisons that Vercel can host without a backend or GPU.

## Run locally

Node.js 20.9 or newer is required.

The repository includes `.nvmrc` for Node.js 22 LTS. On the current research workstation, a compatible project-local runtime is already available at `/data/steven/.cache/node-v22.23.2/bin`.

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Before publication, run:

```bash
npm run lint
npm run typecheck
npm test
npm run validate:media
npm run build
```

## Project structure

```text
app/                  Page, metadata, and global visual system
components/audio/     Synchronized comparison player and cards
components/results/   Subjective listening-result figure
config/site.ts        Paper, code, dataset, and publication links
data/                 Audio examples and reported results
public/audio/         Small compressed demonstration excerpts
public/figures/       Paper figures added for the site
scripts/              Reproducible local audio staging
types/                Public content interfaces
```

## Add or replace audio examples

Historical conditions use this convention:

```text
public/audio/historical/example-id/input.mp3
public/audio/historical/example-id/babe2-p.mp3
public/audio/historical/example-id/babe2-fos.mp3
public/audio/historical/example-id/samecfm.mp3
```

Edit [`data/audioExamples.ts`](data/audioExamples.ts) to add examples or point a condition at an absolute object-storage, Hugging Face, or GitHub Releases URL. The browser uses `preload="metadata"`; it does not eagerly download every file. Keep only a small curated set in the Vercel repository and externally host any large collection.

On the original research workstation, the checked-in media bundle can be regenerated from audited outputs with:

```bash
python3 scripts/stage_demo_audio.py
```

The script records audio source paths, offsets, gains, hashes, channels, sample rates, and durations in `data/generated/`. Those absolute source paths are provenance records and are not used by the deployed site.
Provenance-only source paths may retain the legacy `fms` experiment identifier; all public terminology uses FOS (Full-Orchestra + Section).

The Method section renders the paper pipeline figure from `public/figures/finalICASSPgood_qual.png` at the full content width.

## Add the paper

Replace `public/paper.pdf`. The paper link is controlled by `paperUrl` in [`config/site.ts`](config/site.ts); an empty URL hides the control.

## Configure external links

Edit [`config/site.ts`](config/site.ts):

```ts
paperUrl: "/paper.pdf",
githubUrl: "https://github.com/stevencho24/End-to-End_historical_music_restoration",
datasetUrl: "https://doi.org/10.5281/zenodo.22737610",
arxivUrl: "",
```

Unknown paper and code links are intentionally hidden. The published dataset uses its permanent Zenodo DOI.

## Deploy to Vercel

1. Push the official repository to GitHub.
2. In Vercel, choose **Add New → Project** and import it.
3. Set **Root Directory** to `demo` and accept the detected Next.js settings.
4. Deploy. No environment variables are required.

`next.config.ts` uses static export, so `npm run build` writes the deployable site to `out/`. Vercel can build it directly from GitHub, or another static host can serve that directory.

## Audio provenance and rights

The historical inputs are short excerpts of public-domain 78-rpm recordings obtained through the Internet Archive. Each public example links to its source record. Restoration outputs are precomputed research artifacts derived from the same excerpt. Verify publication rights and institutional policy before substituting any other material.

## License

Website source code is released under the MIT License. Research audio, paper content, and third-party source recordings retain their respective rights and attribution requirements.
