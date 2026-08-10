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

This reads and hashes the 10-file allowlist from all four sources. Source paths
may point directly at full training checkpoints: optimizer/trainer files and
directories are reported as ignored and are never traversed or copied. Every
required file must be a non-symlink regular file. The selected export must be
9,098,689,558 bytes per asset and match its export-tree and independently
recorded weight-shard SHA-256 values. Metadata is also checked for private
path/IP/key patterns.

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
`ready_for_upload`, and generates `SHA256SUMS.full` plus `export_report.json`.

`--transfer-mode hardlink` may be used on one trusted local filesystem to avoid
a second 36.4 GB copy. Do not move or modify source checkpoints while a
hard-linked staging tree exists.

## 4. Run the upload preflight

```bash
python scripts/hf_upload_release.py \
  --folder /secure/staging/MR-IQA-2-hf-release
```

Dry-run is the default. The preflight rehashes all 40 model files plus the
portable cache, checks exact byte counts, validates the SQLite/schema/logical-ID
contract, validates release state, scans public text, and confirms LFS
attributes.

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
After upload, pin the returned Hub commit, download a clean snapshot, run both
checksum files, and execute the GitHub smoke evaluation against that pinned
revision before tagging the release.

## Portable J0 cache

The portable J0 cache is a separate training asset and is not sourced from the
private experiment database. The ready artifact is
`training_assets/original_score_cache.sqlite`: 10,073 rows, 15,003,648 bytes,
and SHA-256
`7d5410f57f17ff1957e7cbeef951ac01973c0bce97da6f700d61bb222bdd5532`.
The export/upload preflight validates its digest, schema, row count, logical
Judge/Actor IDs, rating statistics, and relative-path contract.
