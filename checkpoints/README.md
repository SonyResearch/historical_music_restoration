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
dcf0100ed1268201bc5e0db134d1d9677e32118b75a10e2d1d211d3d12dad4ca
```

Override `CHECKPOINT`, `CHECKPOINT_URL`, or `CHECKPOINT_SHA256` when
mirroring the release. The file is intentionally excluded from ordinary Git
history.
