# Manifests

本目录包含 1 个训练 JSONL、1 个 validation JSONL 和 6 个 test JSONL。所有记录均为单行 JSON，图片路径为相对路径，不包含图片文件。

训练记录字段：

```text
images, target_mean, target_std, sample_id, dataset_name, source_image
```

训练入口会调用 `actor/scripts/prepare_phase_a_dataset.py`，从这些必要字段重建当前 `reasoning/evidence/solution/rating` prompt，并使用 `TRAIN_IMAGE_ROOT` 解析 `images`。

validation 记录字段：

```text
id, image, gt_score, normalized_score, std, std_norm
```

test 记录的公共字段：

```text
id, dataset, image, source_image, normalized_score, gt_score_norm,
source_score, gt_score, std, std_norm
```

原 manifests 中只用于内部索引的元数据路径已经移除。所有图片路径均经过绝对路径与父目录穿越检查。图片数据集及其许可不随本包分发，使用者需自行获得对应数据集。

`checksums.sha256` 记录本目录所有 JSONL 的发布版 SHA256。
