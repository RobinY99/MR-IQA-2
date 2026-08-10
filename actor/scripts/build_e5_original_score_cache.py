#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PAYLOAD_SCHEMA = "vf_original_score_cache_e5_judge_v1"
JUDGE_PROMPT_RATING_RANGE = (1.0, 5.0)
E5_JUDGE_SCORE_ACCEPTANCE_RANGE = (0.0, 5.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(payload)
    return rows


def unique_by_sample(rows: Iterable[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise RuntimeError(f"{label} row has no sample_id")
        if sample_id in result:
            raise RuntimeError(f"duplicate {label} sample_id: {sample_id}")
        result[sample_id] = row
    return result


def finite_rating(value: Any) -> float:
    try:
        rating = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Judge rating is not numeric: {value!r}") from exc
    minimum, maximum = E5_JUDGE_SCORE_ACCEPTANCE_RANGE
    if not math.isfinite(rating) or not minimum <= rating <= maximum:
        raise RuntimeError(
            f"Judge rating is outside [{minimum:g}, {maximum:g}]: {rating!r}"
        )
    return rating


def validate_score(
    score: dict[str, Any],
    source: dict[str, Any],
    args: argparse.Namespace,
) -> float:
    expected = {
        "model_id": args.judge_model_id,
        "model_path": args.judge_model_path,
        "model_tree_sha256": args.judge_model_tree_sha256,
        "prompt_schema": args.judge_prompt_schema,
        "prompt_hash": args.judge_prompt_hash,
        "prompt_mode": "judge",
    }
    mismatches = {
        key: (value, score.get(key))
        for key, value in expected.items()
        if score.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Judge provenance mismatch for {source['sample_id']}: {mismatches}"
        )
    if score.get("inference_status") != "success" or score.get("parse_ok") is not True:
        raise RuntimeError(
            f"invalid Judge output for {source['sample_id']}: "
            f"status={score.get('inference_status')}, errors={score.get('parse_errors')}"
        )
    if score.get("source_image_path") != source.get("source_image_path"):
        raise RuntimeError(f"source path mismatch for {source['sample_id']}")
    if score.get("input_image_sha256") != source.get("source_image_sha256"):
        raise RuntimeError(f"source image SHA256 mismatch for {source['sample_id']}")
    return finite_rating(score.get("model_rating"))


def build_payload(
    source: dict[str, Any],
    score: dict[str, Any],
    rating: float,
    args: argparse.Namespace,
    *,
    manifest_sha256: str,
    scores_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": args.payload_schema,
        "sample_id": source["sample_id"],
        "actor_id": args.cache_actor_id,
        "dataset": source.get("dataset"),
        "split": source.get("split"),
        "source_index": source.get("source_index"),
        "source": {
            "image_name": source.get("image_name"),
            "image_path": source["source_image_path"],
            "image_sha256": source["source_image_sha256"],
            "image_bytes": source.get("source_image_bytes"),
            "width": int(source["source_width"]),
            "height": int(source["source_height"]),
            "ground_truth_raw": source.get("ground_truth_raw"),
            "ground_truth_normalized": source.get("ground_truth_normalized"),
        },
        "source_judge": {
            "model_id": args.judge_model_id,
            "model_path": args.judge_model_path,
            "model_tree_sha256": args.judge_model_tree_sha256,
            "prompt_schema": args.judge_prompt_schema,
            "prompt_version": score.get("prompt_version"),
            "prompt_hash": args.judge_prompt_hash,
            "system_prompt_hash": score.get("system_prompt_hash"),
            "user_prompt_hash": score.get("user_prompt_hash"),
            "rating": rating,
            "rating_text": score.get("rating_text"),
            "rating_format_ok": score.get("rating_format_ok"),
            "rating_representation": score.get("rating_representation"),
            "rating_format_warning": score.get("rating_format_warning"),
            "rating_prompt_range_ok": score.get("rating_prompt_range_ok"),
            "rating_range_warning": score.get("rating_range_warning"),
            "reasons": score.get("judge_reasons"),
            "reasoning": {
                "evidence": score.get("reasoning_evidence"),
                "solution": score.get("reasoning_solution"),
            },
            "raw_completion": score.get("raw_completion"),
            "finish_reason": score.get("finish_reason"),
            "prompt_token_count": score.get("prompt_token_count"),
            "completion_token_count": score.get("completion_token_count"),
            "sampling_profile": score.get("sampling_profile"),
            "deterministic": True,
        },
        "cache_provenance": {
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": manifest_sha256,
            "judge_scores_jsonl": str(args.judge_scores.resolve()),
            "judge_scores_sha256": scores_sha256,
            "one_row_per_source_image": True,
            "edited_image_scores_included": False,
            "legacy_dapo_cache_reused": False,
            "judge_prompt_matches_e5_training": (
                args.judge_prompt_schema == "e5_training_reasoning_v5"
            ),
            "prompt_rating_range": list(JUDGE_PROMPT_RATING_RANGE),
            "score_acceptance_range": list(E5_JUDGE_SCORE_ACCEPTANCE_RANGE),
        },
    }


def create_database(
    path: Path,
    records: list[tuple[str, str, str, float, str]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
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
            """
            INSERT INTO records (
                sample_id,
                actor_id,
                source_image_path,
                source_judge_rating,
                payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.execute("CREATE INDEX records_actor_id ON records(actor_id)")
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        connection.execute("VACUUM")
    finally:
        connection.close()
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--judge-scores", type=Path, required=True)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--judge-model-id", required=True)
    parser.add_argument("--judge-model-path", required=True)
    parser.add_argument("--judge-model-tree-sha256", required=True)
    parser.add_argument("--judge-prompt-schema", required=True)
    parser.add_argument("--judge-prompt-hash", required=True)
    parser.add_argument(
        "--payload-schema",
        default=DEFAULT_PAYLOAD_SCHEMA,
    )
    parser.add_argument(
        "--cache-actor-id",
        default="source_actor_original_score",
    )
    parser.add_argument("--expected-samples", type=int, default=10073)
    args = parser.parse_args()

    sources = unique_by_sample(
        read_jsonl(args.source_manifest),
        label="source manifest",
    )
    scores = unique_by_sample(
        read_jsonl(args.judge_scores),
        label="Judge score",
    )
    if len(sources) != args.expected_samples or set(sources) != set(scores):
        raise RuntimeError(
            "source/Judge coverage mismatch: "
            f"sources={len(sources)}, scores={len(scores)}, "
            f"missing_scores={sorted(set(sources) - set(scores))[:8]}, "
            f"extra_scores={sorted(set(scores) - set(sources))[:8]}"
        )

    manifest_sha256 = sha256_file(args.source_manifest)
    scores_sha256 = sha256_file(args.judge_scores)
    records: list[tuple[str, str, str, float, str]] = []
    ratings: list[float] = []
    for sample_id in sorted(sources):
        source = sources[sample_id]
        score = scores[sample_id]
        rating = validate_score(score, source, args)
        payload = build_payload(
            source,
            score,
            rating,
            args,
            manifest_sha256=manifest_sha256,
            scores_sha256=scores_sha256,
        )
        records.append(
            (
                sample_id,
                args.cache_actor_id,
                str(Path(source["source_image_path"]).resolve()),
                rating,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
        ratings.append(rating)

    create_database(args.output_sqlite, records)
    cache_sha256 = sha256_file(args.output_sqlite)
    summary = {
        "schema_version": "vf_original_score_cache_build_v1",
        "status": "complete",
        "payload_schema": args.payload_schema,
        "sqlite": str(args.output_sqlite.resolve()),
        "sqlite_sha256": cache_sha256,
        "row_count": len(records),
        "sample_count": len(records),
        "actor_ids": [args.cache_actor_id],
        "judge": {
            "model_id": args.judge_model_id,
            "model_path": args.judge_model_path,
            "model_tree_sha256": args.judge_model_tree_sha256,
            "prompt_schema": args.judge_prompt_schema,
            "prompt_hash": args.judge_prompt_hash,
            "prompt_version": next(iter(scores.values())).get("prompt_version"),
            "deterministic": True,
            "prompt_rating_range": list(JUDGE_PROMPT_RATING_RANGE),
            "score_acceptance_range": list(E5_JUDGE_SCORE_ACCEPTANCE_RANGE),
        },
        "source_manifest_sha256": manifest_sha256,
        "judge_scores_sha256": scores_sha256,
        "rating_min": min(ratings),
        "rating_max": max(ratings),
        "rating_mean": sum(ratings) / len(ratings),
        "accepted_below_prompt_floor_count": sum(
            rating < JUDGE_PROMPT_RATING_RANGE[0] for rating in ratings
        ),
        "legacy_dapo_cache_reused": False,
        "judge_prompt_matches_e5_training": (
            args.judge_prompt_schema == "e5_training_reasoning_v5"
        ),
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
