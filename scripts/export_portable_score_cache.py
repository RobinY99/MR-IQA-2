#!/usr/bin/env python3
"""Convert the private-path J0 cache into a minimal portable release asset.

The exported SQLite database keeps only fields required by the training loss:
the frozen source-Judge rating, source dimensions/hash, and immutable Judge
provenance. It deliberately drops ground truth, raw Judge completions, generated
reasoning, image byte counts, and every machine-local path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PORTABLE_SCHEMA = "vf_original_score_cache_e5_judge_e5prompt_portable_v1"
DEFAULT_ACTOR_ID = "source-e5-judge-step725-original-score"
DEFAULT_JUDGE_MODEL_ID = "source-e5-judge-step725"
DEFAULT_JUDGE_MODEL_URI = "hf://RobinY99/MR-IQA-2/judge"
DEFAULT_JUDGE_TREE_SHA256 = (
    "e25415173aacf515e97d5d561c6647a7a84f586061f3a9b2ab3fc079fe21be0a"
)
DEFAULT_PROMPT_HASH = (
    "fa78a4ccfd2194a2026ff0b6b722bf22b28f8fa060389c57c4adb1618ac280f6"
)
DEFAULT_SOURCE_ACTOR_ID = "source_e5_checkpoint725_e5prompt_original_score"
DEFAULT_SOURCE_SCHEMA = "vf_original_score_cache_e5_judge_e5prompt_v2"
FORBIDDEN_FRAGMENTS = ("/mnt/", "/home/", "/Users/", "10.232.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_rating(value: Any) -> float:
    try:
        rating = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"cache rating is not numeric: {value!r}") from exc
    if not math.isfinite(rating) or not 0.0 <= rating <= 5.0:
        raise RuntimeError(f"cache rating is outside [0, 5]: {value!r}")
    return rating


def relative_image_path(prefix: str, source_path: str) -> str:
    prefix_path = PurePosixPath(prefix)
    if prefix_path.is_absolute() or ".." in prefix_path.parts:
        raise RuntimeError("dataset-relative-prefix must be a safe relative path")
    image_name = Path(source_path).name
    if not image_name:
        raise RuntimeError(f"source path has no image name: {source_path!r}")
    return (prefix_path / image_name).as_posix()


def create_database(
    output: Path,
    records: Iterable[tuple[str, str, str, float, str]],
) -> None:
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA page_size = 4096")
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute(
            """
            CREATE TABLE records (
                sample_id TEXT PRIMARY KEY,
                actor_id TEXT NOT NULL,
                source_image_path TEXT NOT NULL UNIQUE,
                source_judge_rating REAL NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
            records,
        )
        connection.execute("CREATE INDEX records_actor_id ON records(actor_id)")
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"portable SQLite integrity check failed: {integrity}")
        connection.execute("VACUUM")
    finally:
        connection.close()
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def scan_export(output: Path) -> None:
    connection = sqlite3.connect(f"file:{output}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        for row in connection.execute(
            "SELECT sample_id, actor_id, source_image_path, payload_json FROM records"
        ):
            text = "\n".join(str(value) for value in row)
            matches = [value for value in FORBIDDEN_FRAGMENTS if value in text]
            if matches:
                raise RuntimeError(f"private fragment survived cache export: {matches}")
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=7000)
    parser.add_argument("--dataset-relative-prefix", default="koniq-10k/512x384")
    parser.add_argument("--source-actor-id", default=DEFAULT_SOURCE_ACTOR_ID)
    parser.add_argument("--source-schema", default=DEFAULT_SOURCE_SCHEMA)
    parser.add_argument("--actor-id", default=DEFAULT_ACTOR_ID)
    parser.add_argument("--judge-model-id", default=DEFAULT_JUDGE_MODEL_ID)
    parser.add_argument("--judge-model-uri", default=DEFAULT_JUDGE_MODEL_URI)
    parser.add_argument(
        "--judge-model-tree-sha256", default=DEFAULT_JUDGE_TREE_SHA256
    )
    parser.add_argument("--judge-prompt-hash", default=DEFAULT_PROMPT_HASH)
    args = parser.parse_args()

    if not args.input_sqlite.is_file():
        raise FileNotFoundError(args.input_sqlite)
    actual_input_sha256 = sha256_file(args.input_sqlite)
    if actual_input_sha256 != args.input_sha256:
        raise RuntimeError(
            "input cache SHA256 mismatch: "
            f"expected={args.input_sha256}, actual={actual_input_sha256}"
        )
    if not args.judge_model_uri.startswith("hf://"):
        raise RuntimeError("judge-model-uri must use the hf:// scheme")

    source = sqlite3.connect(f"file:{args.input_sqlite}?mode=ro", uri=True)
    records: list[tuple[str, str, str, float, str]] = []
    ratings: list[float] = []
    source_manifest_hashes: set[str] = set()
    judge_scores_hashes: set[str] = set()
    seen_paths: set[str] = set()
    try:
        source.execute("PRAGMA query_only = ON")
        rows = source.execute(
            "SELECT sample_id, actor_id, source_image_path, "
            "source_judge_rating, payload_json FROM records ORDER BY sample_id"
        )
        for sample_id_raw, actor_id_raw, path_raw, rating_raw, payload_raw in rows:
            sample_id = str(sample_id_raw)
            if str(actor_id_raw) != args.source_actor_id:
                raise RuntimeError(f"unexpected source actor ID for {sample_id}")
            try:
                payload = json.loads(str(payload_raw))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid source payload for {sample_id}") from exc
            if payload.get("schema_version") != args.source_schema:
                raise RuntimeError(f"unexpected source schema for {sample_id}")
            original_source = payload.get("source")
            original_judge = payload.get("source_judge")
            provenance = payload.get("cache_provenance")
            if not all(
                isinstance(value, dict)
                for value in (original_source, original_judge, provenance)
            ):
                raise RuntimeError(f"malformed source payload for {sample_id}")

            rating = finite_rating(rating_raw)
            if not math.isclose(
                finite_rating(original_judge.get("rating")),
                rating,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"source rating mismatch for {sample_id}")
            if original_judge.get("model_tree_sha256") != args.judge_model_tree_sha256:
                raise RuntimeError(f"source Judge tree mismatch for {sample_id}")
            if original_judge.get("prompt_hash") != args.judge_prompt_hash:
                raise RuntimeError(f"source Judge prompt mismatch for {sample_id}")

            image_path = relative_image_path(
                args.dataset_relative_prefix,
                str(path_raw),
            )
            if image_path in seen_paths:
                raise RuntimeError(f"duplicate portable image path: {image_path}")
            seen_paths.add(image_path)
            image_sha256 = str(original_source.get("image_sha256") or "")
            if len(image_sha256) != 64:
                raise RuntimeError(f"invalid image SHA256 for {sample_id}")
            width = int(original_source.get("width") or 0)
            height = int(original_source.get("height") or 0)
            if width <= 0 or height <= 0:
                raise RuntimeError(f"invalid image dimensions for {sample_id}")

            source_manifest_sha256 = str(
                provenance.get("source_manifest_sha256") or ""
            )
            judge_scores_sha256 = str(
                provenance.get("judge_scores_sha256") or ""
            )
            if len(source_manifest_sha256) != 64 or len(judge_scores_sha256) != 64:
                raise RuntimeError(f"invalid upstream provenance for {sample_id}")
            source_manifest_hashes.add(source_manifest_sha256)
            judge_scores_hashes.add(judge_scores_sha256)

            portable_payload = {
                "schema_version": PORTABLE_SCHEMA,
                "sample_id": sample_id,
                "actor_id": args.actor_id,
                "dataset": "koniq10k",
                "split": str(payload.get("split") or "train"),
                "source": {
                    "image_path": image_path,
                    "image_sha256": image_sha256,
                    "width": width,
                    "height": height,
                },
                "source_judge": {
                    "model_id": args.judge_model_id,
                    "model_uri": args.judge_model_uri,
                    "model_tree_sha256": args.judge_model_tree_sha256,
                    "prompt_schema": str(original_judge.get("prompt_schema") or ""),
                    "prompt_version": str(original_judge.get("prompt_version") or ""),
                    "prompt_hash": args.judge_prompt_hash,
                    "rating": rating,
                    "deterministic": True,
                },
                "cache_provenance": {
                    "source_cache_sha256": actual_input_sha256,
                    "source_manifest_sha256": source_manifest_sha256,
                    "judge_scores_sha256": judge_scores_sha256,
                    "one_row_per_source_image": True,
                    "edited_image_scores_included": False,
                    "raw_judge_text_included": False,
                    "ground_truth_included": False,
                },
            }
            records.append(
                (
                    sample_id,
                    args.actor_id,
                    image_path,
                    rating,
                    json.dumps(
                        portable_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
            ratings.append(rating)
    finally:
        source.close()

    if len(records) != args.expected_rows:
        raise RuntimeError(
            f"row-count mismatch: expected={args.expected_rows}, actual={len(records)}"
        )
    if len(source_manifest_hashes) != 1 or len(judge_scores_hashes) != 1:
        raise RuntimeError("source provenance hashes are not uniform")

    create_database(args.output_sqlite, records)
    scan_export(args.output_sqlite)
    output_sha256 = sha256_file(args.output_sqlite)
    summary = {
        "schema_version": "mriqa2_portable_original_score_cache_manifest_v1",
        "status": "complete",
        "file": args.output_sqlite.name,
        "sha256": output_sha256,
        "bytes": args.output_sqlite.stat().st_size,
        "row_count": len(records),
        "sample_count": len(records),
        "actor_ids": [args.actor_id],
        "payload_schema": PORTABLE_SCHEMA,
        "dataset_relative_prefix": args.dataset_relative_prefix,
        "judge": {
            "model_id": args.judge_model_id,
            "model_uri": args.judge_model_uri,
            "model_tree_sha256": args.judge_model_tree_sha256,
            "prompt_hash": args.judge_prompt_hash,
            "rating_acceptance_range": [0.0, 5.0],
        },
        "rating": {
            "min": min(ratings),
            "max": max(ratings),
            "mean": sum(ratings) / len(ratings),
        },
        "source": {
            "private_cache_sha256": actual_input_sha256,
            "source_manifest_sha256": next(iter(source_manifest_hashes)),
            "judge_scores_sha256": next(iter(judge_scores_hashes)),
        },
        "omitted_fields": [
            "absolute_paths",
            "ground_truth",
            "image_bytes",
            "raw_completion",
            "reasoning_evidence",
            "reasoning_solution",
        ],
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
