# Hugging Face publishing

Target repository: `RobinY99/MR-IQA-2`.

Create a private source map from `checkpoints/hf_sources.example.json`. Keep
local checkpoint paths outside the Git worktree.

Validate the source exports:

```bash
python scripts/hf_export_checkpoints.py \
  --manifest ../huggingface/checkpoint_manifest.json \
  --sources /secure/local/hf_sources.json
```

Materialize a fresh staging directory:

```bash
python scripts/hf_export_checkpoints.py \
  --manifest ../huggingface/checkpoint_manifest.json \
  --sources /secure/local/hf_sources.json \
  --hf-template ../huggingface \
  --output /secure/staging/MR-IQA-2-hf-release \
  --materialize
```

Run the upload preflight:

```bash
python scripts/hf_upload_release.py \
  --folder /secure/staging/MR-IQA-2-hf-release
```

After authenticating with Hugging Face, upload explicitly:

```bash
python scripts/hf_upload_release.py \
  --folder /secure/staging/MR-IQA-2-hf-release \
  --revision main \
  --commit-upload
```

Pin the returned Hub commit and run the released evaluation against a clean
download before tagging the release.
