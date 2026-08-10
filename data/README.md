# Manifests

One training, one validation, and six test JSONL manifests are included. Image
paths are relative; images are not included.

Training records contain:

```text
images, target_mean, target_std, sample_id, dataset_name, source_image
```

Validation records contain:

```text
id, image, gt_score, normalized_score, std, std_norm
```

Test records share these fields:

```text
id, dataset, image, source_image, normalized_score, gt_score_norm,
source_score, gt_score, std, std_norm
```

Obtain the image datasets separately under their licenses.
