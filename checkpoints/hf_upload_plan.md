# Hugging Face publishing

Target: `RobinY99/MR-IQA-2`.

The published tree is fixed:

```text
.gitattributes
README.md
LICENSE
actor/   # mask E5 Actor, step 1455
judge/   # E5 Judge, step 725
editor/  # FLUX.2-klein-4B, pinned revision
```

Create a private source map from `checkpoints/hf_sources.example.json`. Keep
that file and all local checkpoint paths outside the Git worktree.

Validate the three source trees without writing:

```bash
python scripts/hf_export_checkpoints.py \
  --sources /secure/local/hf_sources.json
```

Materialize a new staging tree. The Editor's verified upstream `LICENSE.md`
is published once as the root `LICENSE`:

```bash
python scripts/hf_export_checkpoints.py \
  --sources /secure/local/hf_sources.json \
  --hf-template ../huggingface \
  --output /secure/staging/MR-IQA-2-hf-release \
  --materialize
```

Run the local preflight. This is a dry run and does not contact the Hub:

```bash
python scripts/hf_upload_release.py \
  --folder /secure/staging/MR-IQA-2-hf-release
```

After authentication, atomically replace the remote tree in one parent-pinned
commit:

```bash
python scripts/hf_upload_release.py \
  --folder /secure/staging/MR-IQA-2-hf-release \
  --revision main \
  --commit-upload \
  --replace-remote
```

Both mutation flags are required. Every old remote path absent from the local
staging tree is explicitly deleted, including prior completion checkpoints,
caches, and root-level model weights.
