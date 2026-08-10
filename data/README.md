# Manifests

This directory contains one training JSONL file, one validation JSONL file,
and six test JSONL files. Each record is a single JSON object. Image paths are
relative, and image files are not included.

Training records contain:

```text
images, target_mean, target_std, sample_id, dataset_name, source_image
```

The training launcher calls `actor/scripts/prepare_phase_a_dataset.py` to build
the current `reasoning/evidence/solution/rating` prompt from these fields and
resolve `images` against `TRAIN_IMAGE_ROOT`.

Validation records contain:

```text
id, image, gt_score, normalized_score, std, std_norm
```

Test records share these fields:

```text
id, dataset, image, source_image, normalized_score, gt_score_norm,
source_score, gt_score, std, std_norm
```

Internal-only metadata paths were removed from the source manifests. Image
paths are checked for absolute paths and parent-directory traversal. Image
datasets and their licenses are not distributed with this repository and must
be obtained separately.
