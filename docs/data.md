# Data manifests

The repository distributes metadata manifests, not dataset images. All image
locations are relative paths so one release can be used with independently
obtained datasets on different machines.

## Inventory

| File | Split | Rows |
| --- | --- | ---: |
| `train.jsonl` | Training mixture | 7,000 |
| `validation.jsonl` | Fixed validation | 200 |
| `koniq.jsonl` | KonIQ-10K test | 2,010 |
| `spaq_full.jsonl` | SPAQ test | 11,125 |
| `livew.jsonl` | LIVE-W test | 1,162 |
| `kadid_full.jsonl` | KADID-10K test | 10,125 |
| `agiqa3k.jsonl` | AGIQA-3K test | 2,982 |
| `csiq.jsonl` | CSIQ test | 866 |

The six test manifests total exactly 28,270 rows. Validation is separate, so
`scripts/evaluate.sh all` processes 28,470 rows.

## Schemas

Training rows contain the minimal fields needed to reconstruct the public
Actor prompt:

```text
images, target_mean, target_std, sample_id, dataset_name, source_image
```

`actor/scripts/prepare_phase_a_dataset.py` resolves `images` against
`TRAIN_IMAGE_ROOT` and creates the structured evidence/solution/rating request.

Validation rows contain:

```text
id, image, gt_score, normalized_score, std, std_norm
```

Test rows share:

```text
id, dataset, image, source_image, normalized_score, gt_score_norm,
source_score, gt_score, std, std_norm
```

Some score aliases are retained so dataset-specific source values and the
normalized comparison value remain auditable. Evaluation uses the normalized
quality target declared by its parser contract.

## Image layout

Set `TRAIN_IMAGE_ROOT` and `EVAL_IMAGE_ROOT` in `.env`. The path stored in a row
is joined to the corresponding root. Release checks reject absolute image paths
and parent-directory traversal.

Datasets do not necessarily share the same folder layout. Construct a single
evaluation root whose subpaths match the committed `image` fields, or create a
read-only view with the same relative layout. Do not edit a manifest merely to
encode a private machine path.

## Redistribution and ethics

The JSONL manifests do not grant rights to the source images. Obtain each
dataset from its official distributor, review its research/commercial-use
terms, and comply with attribution, privacy, and deletion requirements. Do not
publish edited images or raw model outputs without considering whether the
source content includes people, private places, or other sensitive material.

When creating a new manifest:

1. confirm that redistribution of its metadata is permitted;
2. use stable, non-identifying sample IDs where possible;
3. keep image paths relative;
4. record score normalization and split construction;
5. run `bash scripts/test_release.sh --static`.
