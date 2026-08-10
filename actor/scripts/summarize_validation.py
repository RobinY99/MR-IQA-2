#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import secrets
import sqlite3
import statistics
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite(values: list[Any]) -> list[float]:
    output = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            output.append(parsed)
    return output


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int = 5000,
    seed: int = 42,
) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "mean": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
    generator = random.Random(seed)
    count = len(values)
    means = [
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(iterations)
    ]
    return {
        "n": count,
        "mean": statistics.fmean(values),
        "ci95_low": quantile(means, 0.025),
        "ci95_high": quantile(means, 0.975),
    }


def reasoning_reward(delta: float, tau_s: float = 1.0) -> float:
    if delta == 0:
        return 0.0
    return math.copysign(
        1.0 - math.exp(-(delta * delta) / (2.0 * tau_s)),
        delta,
    )


def read_target(label: str, eval_root: Path) -> dict[str, Any]:
    contract = json.loads(
        (eval_root / "contract.json").read_text(encoding="utf-8")
    )
    actor_payload = json.loads(
        (eval_root / "actor_outputs/validation/merged.json").read_text(
            encoding="utf-8"
        )
    )
    actor = actor_payload["summary"]
    actor_rows = actor_payload["results"]
    editor = json.loads(
        (eval_root / "editor_judge/editor_summary.json").read_text(
            encoding="utf-8"
        )
    )
    judge = json.loads(
        (eval_root / "editor_judge/summary.json").read_text(encoding="utf-8")
    )
    audit_path = eval_root / "state/final_audit.json"
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.is_file()
        else {
            "status": judge.get("status"),
            "source": "editor_judge_summary",
        }
    )
    connection = sqlite3.connect(eval_root / "index/evaluation.sqlite")
    raw_results = [
        json.loads(raw)
        for (raw,) in connection.execute(
            """
            SELECT result_json
            FROM editor_judge_results
            ORDER BY dataset, sample_index
            """
        )
    ]
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()
    if integrity != "ok":
        raise RuntimeError(f"{label}: SQLite integrity check failed: {integrity}")
    if len(actor_rows) != 200 or len(raw_results) != 200:
        raise RuntimeError(
            f"{label}: validation row mismatch actor={len(actor_rows)} "
            f"judge={len(raw_results)}"
        )
    success = [row for row in raw_results if row["status"] == "success"]
    service_errors = [
        row for row in raw_results if row["status"] == "service_error"
    ]
    if service_errors:
        raise RuntimeError(f"{label}: service errors present: {len(service_errors)}")
    cache_hit_rows = sum(
        row.get("original_score_cache_hit") is True for row in success
    )
    if cache_hit_rows != len(success):
        raise RuntimeError(
            f"{label}: fixed J0 cache coverage is incomplete: "
            f"{cache_hit_rows}/{len(success)}"
        )
    deltas = finite([row.get("judge_delta") for row in success])
    original = finite([row.get("original_judge_score") for row in success])
    edited = finite([row.get("edited_judge_score") for row in success])
    if not deltas or not (len(deltas) == len(original) == len(edited)):
        raise RuntimeError(f"{label}: incomplete finite Judge results")
    completion_lengths = finite(
        [row.get("completion_token_count") for row in actor_rows]
    )
    clipped = sum(row.get("finish_reason") == "length" for row in actor_rows)
    reasoning_rewards_all_rows = [
        (
            reasoning_reward(float(row["judge_delta"]))
            if row["status"] == "success"
            else 0.0
        )
        for row in raw_results
    ]
    by_sample = {
        (str(row["dataset"]), int(row["index"])): row for row in raw_results
    }
    return {
        "label": label,
        "eval_root": str(eval_root),
        "checkpoint": contract.get(
            "checkpoint", contract.get("actor_model_path")
        ),
        "checkpoint_export_tree_sha256": contract.get(
            "checkpoint_export_tree_sha256",
            (contract.get("checkpoint_digest") or {}).get("sha256", ""),
        ),
        "checkpoint_digest_semantics": (
            contract.get("checkpoint_digest") or {}
        ).get("semantics", ""),
        "validation_manifest_sha256": contract.get(
            "validation_manifest_sha256", ""
        ),
        "actor": {
            key: actor.get(key)
            for key in (
                "json_parse_success_rate",
                "actor_format_success_rate",
                "rating_parse_success_rate",
                "num_valid_rating",
                "plcc",
                "srcc",
                "mae",
                "edit_request_rate",
                "no_edit_rate",
                "low_target_edit_request_rate",
                "unique_completion_ratio",
                "batch_generate_exception_count",
                "singleton_generate_exception_count",
            )
        }
        | {
            "completion_length_mean": statistics.fmean(completion_lengths),
            "completion_length_median": statistics.median(completion_lengths),
            "completion_length_p95": quantile(completion_lengths, 0.95),
            "length_finish_count": clipped,
            "length_finish_rate": clipped / len(actor_rows),
        },
        "editor": {
            "edited_rows": int(editor["edited_rows"]),
            "actor_ineligible_rows": int(editor["actor_ineligible_rows"]),
            "service_error_rows": int(editor["service_error_rows"]),
            "edited_images_retained": int(editor["edited_images_retained"]),
        },
        "e5_judge": {
            "success_rows": len(success),
            "actor_ineligible_rows": sum(
                row["status"] == "actor_ineligible" for row in raw_results
            ),
            "service_error_rows": int(judge["service_error_rows"]),
            "original_score_cache_hit_rows": cache_hit_rows,
            "original_score_cache_hit_rate": cache_hit_rows / len(success),
            "original_score_mean": statistics.fmean(original),
            "edited_score_mean": statistics.fmean(edited),
            "judge_delta_mean": statistics.fmean(deltas),
            "judge_delta_median": statistics.median(deltas),
            "judge_delta_p05": quantile(deltas, 0.05),
            "judge_delta_p95": quantile(deltas, 0.95),
            "positive_delta_rate": sum(value > 0 for value in deltas)
            / len(deltas),
            "nonnegative_delta_rate": sum(value >= 0 for value in deltas)
            / len(deltas),
            "mean_reasoning_raw_reward_success_rows": statistics.fmean(
                reasoning_reward(value) for value in deltas
            ),
            "mean_reasoning_raw_reward_all_rows_zero_filled": statistics.fmean(
                reasoning_rewards_all_rows
            ),
        },
        "audit": audit,
        "_by_sample": by_sample,
    }


def paired_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_rows = baseline["_by_sample"]
    candidate_rows = candidate["_by_sample"]
    keys = sorted(set(baseline_rows) & set(candidate_rows))
    pairs = [
        (baseline_rows[key], candidate_rows[key])
        for key in keys
        if baseline_rows[key]["status"] == "success"
        and candidate_rows[key]["status"] == "success"
    ]
    delta_differences = [
        float(candidate_row["judge_delta"]) - float(baseline_row["judge_delta"])
        for baseline_row, candidate_row in pairs
    ]
    edited_differences = [
        float(candidate_row["edited_judge_score"])
        - float(baseline_row["edited_judge_score"])
        for baseline_row, candidate_row in pairs
    ]
    reasoning_reward_differences = [
        reasoning_reward(float(candidate_row["judge_delta"]))
        - reasoning_reward(float(baseline_row["judge_delta"]))
        for baseline_row, candidate_row in pairs
    ]
    original_differences = [
        float(candidate_row["original_judge_score"])
        - float(baseline_row["original_judge_score"])
        for baseline_row, candidate_row in pairs
    ]
    if any(value != 0 for value in original_differences):
        raise RuntimeError(
            "paired validation targets do not share identical cached J0 scores"
        )
    all_keys = sorted(set(baseline_rows) | set(candidate_rows))
    if len(all_keys) != 200:
        raise RuntimeError(
            f"paired validation union must contain 200 rows, got {len(all_keys)}"
        )
    all_row_reasoning_differences = []
    for key in all_keys:
        baseline_row = baseline_rows.get(key)
        candidate_row = candidate_rows.get(key)
        baseline_reward = (
            reasoning_reward(float(baseline_row["judge_delta"]))
            if baseline_row is not None and baseline_row["status"] == "success"
            else 0.0
        )
        candidate_reward = (
            reasoning_reward(float(candidate_row["judge_delta"]))
            if candidate_row is not None and candidate_row["status"] == "success"
            else 0.0
        )
        all_row_reasoning_differences.append(candidate_reward - baseline_reward)
    return {
        "candidate_minus_baseline": (
            f"{candidate['label']}_minus_{baseline['label']}"
        ),
        "common_success_rows": len(pairs),
        "judge_delta_difference": bootstrap_mean_ci(delta_differences),
        "edited_score_difference": bootstrap_mean_ci(edited_differences),
        "common_success_reasoning_raw_reward_difference": bootstrap_mean_ci(
            reasoning_reward_differences
        ),
        "max_absolute_original_score_difference": max(
            (abs(value) for value in original_differences),
            default=0.0,
        ),
        "candidate_delta_win_rate": (
            sum(value > 0 for value in delta_differences)
            / len(delta_differences)
            if delta_differences
            else math.nan
        ),
        "candidate_delta_tie_rate": (
            sum(value == 0 for value in delta_differences)
            / len(delta_differences)
            if delta_differences
            else math.nan
        ),
        "all_200_zero_filled_reasoning_reward_difference": bootstrap_mean_ci(
            all_row_reasoning_differences
        ),
        "candidate_zero_filled_reasoning_win_rate": sum(
            value > 0 for value in all_row_reasoning_differences
        )
        / len(all_row_reasoning_differences),
        "candidate_zero_filled_reasoning_tie_rate": sum(
            value == 0 for value in all_row_reasoning_differences
        )
        / len(all_row_reasoning_differences),
    }


def numeric_metrics(prefix: str, payload: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, bool):
            output[name] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            output[name] = float(value)
        elif isinstance(value, dict):
            output.update(numeric_metrics(name, value))
    return output


def log_wandb(
    stage_root: Path,
    targets: list[dict[str, Any]],
    *,
    entity: str | None,
    project: str,
) -> dict[str, str]:
    import wandb

    urls: dict[str, str] = {}
    state = stage_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    group = stage_root.name
    for target in targets:
        label = target["label"]
        run_id_path = state / f"wandb_{label}_run_id"
        if run_id_path.is_file():
            run_id = run_id_path.read_text(encoding="utf-8").strip()
        else:
            run_id = secrets.token_hex(8)
            run_id_path.write_text(run_id + "\n", encoding="utf-8")
        run = wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            resume="allow",
            name=f"{group}_{label}",
            group=group,
            job_type="validation200_editor_e5judge",
            config={
                "label": label,
                "checkpoint": target["checkpoint"],
                "checkpoint_export_tree_sha256": target[
                    "checkpoint_export_tree_sha256"
                ],
                "checkpoint_digest_semantics": target[
                    "checkpoint_digest_semantics"
                ],
                "validation_manifest_sha256": target[
                    "validation_manifest_sha256"
                ],
                "evaluation_protocol": "vf_training_aligned_vllm_v1_20260718",
                "editor": os.environ.get(
                    "DIFFUSERS_MODEL_PATH", "external_editor_model"
                ),
                "judge": os.environ.get("JUDGER_MODEL_ID", "e5_judge"),
                "num_validation_rows": 200,
            },
            reinit=True,
        )
        run.log(
            numeric_metrics(
                "",
                {
                    "actor": target["actor"],
                    "editor": target["editor"],
                    "e5_judge": target["e5_judge"],
                },
            ),
            step=0,
        )
        run.summary.update(
            numeric_metrics(
                "",
                {
                    "actor": target["actor"],
                    "editor": target["editor"],
                    "e5_judge": target["e5_judge"],
                },
            )
        )
        urls[label] = run.url
        run.finish()
    return urls


def parse_target(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("--target must be LABEL=/absolute/path")
    return label, Path(path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "mr-iqa-grpo-editor-judge"),
    )
    args = parser.parse_args()
    stage_root = args.stage_root.resolve()
    targets = [read_target(label, path) for label, path in args.target]
    public_targets = []
    for target in targets:
        public = dict(target)
        public.pop("_by_sample")
        public_targets.append(public)
    payload: dict[str, Any] = {
        "schema_version": "vf_two_checkpoint_validation200_e5judge_v1",
        "status": "complete",
        "stage_root": str(stage_root),
        "targets": public_targets,
    }
    if len(targets) >= 2:
        payload["paired_comparisons"] = [
            paired_comparison(targets[0], candidate)
            for candidate in targets[1:]
        ]
        if len(targets) == 2:
            payload["paired_comparison"] = payload["paired_comparisons"][0]
    atomic_json(args.output.resolve(), payload)
    if args.log_wandb:
        payload["wandb_urls"] = log_wandb(
            stage_root,
            targets,
            entity=args.wandb_entity,
            project=args.wandb_project,
        )
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
