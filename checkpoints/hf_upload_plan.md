# Hugging Face release plan

Target: `RobinY99/MR-IQA-2` (model repository).

The release is intentionally two-phase. Validation and staging are local;
upload is a separate, explicit command. Neither tool deletes a source
checkpoint or a remote Hub file.

## 1. Prepare an uncommitted source map

Copy `checkpoints/hf_sources.example.json` outside the Git worktree and replace
each `SET_LOCAL_CHECKPOINT_DIRECTORY` with the corresponding local inference
export. Never commit this map: local paths are environment-specific and may
contain private information.

## 2. Validate without writing

```bash
python scripts/hf_export_checkpoints.py \
  --manifest ../huggingface/checkpoint_manifest.json \
  --sources /secure/local/hf_sources.json
```

This validates the public inference files from every source listed in the
manifest. Source paths may point directly at full training checkpoints:
optimizer/trainer files and directories are ignored and are never copied.
Every required file must be a non-symlink regular file. Metadata is also
checked for private path, IP, and credential patterns.

## 3. Materialize a fresh staging directory

```bash
python scripts/hf_export_checkpoints.py \
  --manifest ../huggingface/checkpoint_manifest.json \
  --sources /secure/local/hf_sources.json \
  --hf-template ../huggingface \
  --output /secure/staging/MR-IQA-2-hf-release \
  --materialize
```

The output path must not already exist. The exporter copies the metadata-only
Hub template, installs the verified snapshots at their manifest locations,
writes sanitized provenance records, changes only the staged manifest to
`ready_for_upload`, and generates `export_report.json`.

`--transfer-mode hardlink` may be used on one trusted local filesystem to avoid
a second copy. Do not move or modify source checkpoints while a hard-linked
staging tree exists.

## 4. Run the upload preflight

```bash
python scripts/hf_upload_release.py \
  --folder /secure/staging/MR-IQA-2-hf-release
```

Dry-run is the default. The preflight validates every staged model artifact and
the portable cache, checks the SQLite/schema/logical-ID contract, validates
release state, scans public text, and confirms LFS attributes.

## 5. Upload explicitly

Install `huggingface_hub`, authenticate through `HF_TOKEN` or the standard Hub
credential store, and then run:

```bash
python scripts/hf_upload_release.py \
  --folder /secure/staging/MR-IQA-2-hf-release \
  --revision main \
  --commit-upload
```

The command uses `upload_folder`; it has no delete operation or delete pattern.
After upload, pin the returned Hub commit, download a clean snapshot, rerun the
upload preflight, and execute the GitHub smoke evaluation against that pinned
revision before tagging the release.

## Portable J0 cache

The portable J0 cache is a separate training asset and is not sourced from the
private experiment database. The ready artifact is
`training_assets/original_score_cache.sqlite`: 10,073 rows and 15,003,648
bytes. The export/upload preflight validates its artifact identity, schema,
row count, logical Judge/Actor IDs, rating statistics, and relative-path
contract.
