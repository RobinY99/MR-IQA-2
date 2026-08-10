# Data manifests

The repository distributes relative-path metadata manifests, not images.

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

## Image layout

Set `TRAIN_IMAGE_ROOT` and `EVAL_IMAGE_ROOT` in `.env`. The path stored in a row
is joined to the corresponding root. Release checks reject absolute image paths
and parent-directory traversal.

The directory under each root must match the committed relative paths.

## Redistribution and ethics

Obtain source images from official distributors and follow their licenses,
attribution, privacy, and deletion requirements.

When creating a new manifest:

1. confirm metadata redistribution rights;
2. use stable, non-identifying IDs and relative paths;
3. record score normalization and split construction;
4. run `bash scripts/test_release.sh --static`.
