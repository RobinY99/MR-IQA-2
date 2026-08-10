#!/usr/bin/env python3
"""Export path-free result tables from the internal analysis bundle.

This script intentionally keeps numeric scientific evidence while dropping
machine-specific source/checkpoint columns. It does not copy raw images,
credentials, W&B metadata, service logs, or absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


STEP_TABLES = (
    "step_metrics_fieldmask_kl002_e5.csv",
    "step_metrics_completioncredit_globalkl002_e5.csv",
)
COPY_TABLES = (
    "generalization_exact_summary.csv",
    "collapse_milestones.csv",
    "training_epoch_summary.csv",
)
DROP_COLUMNS = {
    "checkpoint",
    "data_file",
    "image_root",
    "model_name_or_path",
    "processor_name_or_path",
    "source_log",
}


def export_csv(source: Path, destination: Path) -> None:
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name for name in (reader.fieldnames or []) if name not in DROP_COLUMNS]
        rows = [{name: row.get(name, "") for name in fieldnames} for row in reader]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def assert_path_free(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    forbidden = ("/mnt/", "/home/", "/Users/", "ssh-rsa", "BEGIN OPENSSH")
    hits = [token for token in forbidden if token in text]
    if hits:
        raise RuntimeError(f"private path/token in {path}: {hits}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, object]] = []
    for name in (*STEP_TABLES, *COPY_TABLES, "validation_checkpoint_metrics.csv"):
        source = args.source / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = args.output / name
        export_csv(source, destination)
        assert_path_free(destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        with destination.open(encoding="utf-8", newline="") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
        exported.append(
            {
                "file": name,
                "rows": row_count,
                "sha256": digest,
                "source_kind": "derived_numeric_or_generated_text",
            }
        )

    manifest = {
        "schema_version": "mr_iqa_2_public_results_v1",
        "privacy_policy": "absolute machine paths and raw runtime metadata removed",
        "files": exported,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
