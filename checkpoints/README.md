# Checkpoints

This directory is the stable local destination for released model weights.
Download the final paper model using the BEHM-GAN-style setup script:

```bash
bash prepare_data.sh
```

The script downloads the `v1.0.0` GitHub Release asset, resumes interrupted
downloads, and verifies SHA-256 before installing it at:

```text
checkpoints/samecfm_40m_fos.pt
```

Expected SHA-256:

```text
2b13d250a66e3c640a52d3b5969951fd6d7a1336b5b5bd97770b28c5707f7ae3
```

Override `CHECKPOINT`, `CHECKPOINT_URL`, or `CHECKPOINT_SHA256` when
mirroring the release. The file is intentionally excluded from ordinary Git
history.
