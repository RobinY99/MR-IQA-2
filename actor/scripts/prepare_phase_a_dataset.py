#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

from prompt_contract import (
    ACTOR_SCHEMA,
    PROMPT_HASH,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TRAINING_USER_PROMPT as USER_PROMPT,
    build_training_messages,
    prompt_metadata,
)
from actor_contract import REASONS_RATING_ACTOR_SCHEMA

LOW_TARGET_WARMSTART_INSTRUCTION = (
    "Improve visible perceptual quality while preserving the original scene content."
)


def load_rows(path: str) -> list[dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("data", "records", "train_records", "test_records"):
        rows = data.get(key)
        if isinstance(rows, list):
            return rows
    raise ValueError(f"Unsupported data structure in {path}")


def resolve_image_path(image: str, image_root: str | None) -> str | None:
    image = str(image).strip()
    candidates = [image]
    if image_root:
        candidates.append(os.path.join(image_root, image))
        candidates.append(os.path.join(image_root, os.path.basename(image)))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def row_image_reference(row: dict[str, Any]) -> str | None:
    image: Any = row.get("image") or row.get("image_path") or row.get("img_path")
    if image is None:
        images = row.get("images")
        if isinstance(images, list) and len(images) == 1:
            image = images[0]
    while isinstance(image, list) and len(image) == 1:
        image = image[0]
    if isinstance(image, dict):
        image = image.get("path") or image.get("image")
    if image is None:
        return None
    value = str(image).strip()
    return value or None


def as_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def warmstart_target_payload(score: float, low_quality_threshold: float) -> dict[str, Any]:
    suggestion = ""
    if score <= low_quality_threshold:
        suggestion = LOW_TARGET_WARMSTART_INSTRUCTION
    if ACTOR_SCHEMA == REASONS_RATING_ACTOR_SCHEMA:
        reasons = (
            "The visible image quality is degraded; improve it globally while preserving the original scene content."
            if suggestion
            else "The image shows no meaningful visible quality issue; no correction is necessary."
        )
        return {
            "reasons": reasons,
            "rating": f"{score:.2f}",
        }
    reason = (
        "Low-quality image with visible defects that need a faithful global improvement."
        if suggestion
        else "Image quality is high enough that no faithful edit is required."
    )
    return {
        "reason": reason,
        "rating": f"{score:.2f}",
        "suggestion": suggestion,
    }


def assistant_target_text(score: float, low_quality_threshold: float) -> str:
    return json.dumps(
        warmstart_target_payload(score, low_quality_threshold),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_rows(args.input)
    if not bool(getattr(args, "preserve_order", False)):
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    records: list[dict[str, Any]] = []
    low_quality_repeat = max(1, int(getattr(args, "low_quality_repeat", 1) or 1))
    low_quality_threshold = float(getattr(args, "low_quality_threshold", 3.0))
    low_target_region_warmstart = bool(getattr(args, "low_target_region_warmstart", False))
    sft_assistant_targets = bool(getattr(args, "sft_assistant_targets", False))
    low_target_only = bool(getattr(args, "low_target_only", False))
    for row in rows:
        image = row_image_reference(row)
        if not image:
            continue
        image_path = resolve_image_path(str(image), args.image_root)
        if not image_path:
            continue
        score = as_float(
            row,
            (
                "score_norm",
                "gt_score_norm",
                "normalized_score",
                "source_score",
                "score",
                "human_score",
                "mos",
                "rating",
                "quality_score",
                "target_mean",
            ),
        )
        if score is None:
            continue
        score = float(score)
        if not math.isfinite(score) or not 1.0 <= score <= 5.0:
            continue
        is_low_target = score <= low_quality_threshold
        if low_target_only and not is_low_target:
            continue
        std = as_float(row, ("std_norm", "source_std", "std", "score_std", "mos_std", "target_std"))
        if std is None or std < 0:
            std = 1.0

        base_sample_id = row.get("sample_id") or row.get("id") or f"sample_{len(records):07d}"
        repeat = low_quality_repeat if is_low_target else 1
        for repeat_idx in range(repeat):
            sample_id = str(base_sample_id)
            if repeat > 1:
                sample_id = f"{sample_id}__lowrep{repeat_idx}"
            record = {
                "messages": build_training_messages([image_path]),
                "images": [image_path],
                "solution": json.dumps({"rating": f"{score:.2f}"}, ensure_ascii=False),
                "target_mean": float(score),
                "target_std": max(float(std), 1e-4),
                "sample_id": sample_id,
                "dataset_name": row.get("dataset_name", "iqa"),
                "source_image": str(image),
                **prompt_metadata(),
            }
            if low_target_region_warmstart:
                record["warmstart_target"] = json.dumps(
                    warmstart_target_payload(score, low_quality_threshold),
                    ensure_ascii=False,
                )
            if sft_assistant_targets:
                target_text = assistant_target_text(score, low_quality_threshold)
                record["messages"].append({"role": "assistant", "content": target_text})
                record["solution"] = target_text
            records.append(record)
            if args.max_samples and len(records) >= args.max_samples:
                break
        if args.max_samples and len(records) >= args.max_samples:
            break
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preserve-order", action="store_true")
    parser.add_argument("--low-quality-threshold", type=float, default=3.0)
    parser.add_argument("--low-quality-repeat", type=int, default=1)
    parser.add_argument("--low-target-region-warmstart", action="store_true")
    parser.add_argument("--sft-assistant-targets", action="store_true")
    parser.add_argument("--low-target-only", action="store_true")
    args = parser.parse_args()

    records = build_records(args)
    if not records:
        raise SystemExit("No usable records found.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "num_records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
