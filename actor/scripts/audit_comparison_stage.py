#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OOM_PATTERNS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "outofmemoryerror: cuda",
    "cuda failure 2 'out of memory'",
    "cublas_status_alloc_failed",
)


def finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def original_score_cache_contract_valid(
    cache: dict[str, Any],
    expected_cache: dict[str, Any],
    expected_judge: dict[str, Any],
) -> bool:
    try:
        expected_sha256 = str(expected_cache["sha256"])
        expected_row_count = int(expected_cache["expected_row_count"])
        expected_sample_count = int(expected_cache["expected_sample_count"])
        actual_row_count = int(cache["row_count"])
        actual_sample_count = int(cache["sample_count"])
        expected_rating_range = [
            float(value) for value in expected_cache["rating_acceptance_range"]
        ]
        actual_rating_range = [
            float(value) for value in cache["rating_acceptance_range"]
        ]
        expected_model_id = str(expected_judge["model_id"])
        expected_model_path = str(expected_judge["model_path"])
        expected_model_tree_sha256 = str(expected_judge["model_tree_sha256"])
        expected_prompt_hash = str(expected_judge["prompt_hash"])
    except (KeyError, TypeError, ValueError):
        return False

    return all(
        (
            bool(re.fullmatch(r"[0-9a-f]{64}", expected_sha256)),
            bool(expected_model_id),
            bool(expected_model_path),
            bool(re.fullmatch(r"[0-9a-f]{64}", expected_model_tree_sha256)),
            bool(re.fullmatch(r"[0-9a-f]{64}", expected_prompt_hash)),
            expected_row_count > 0,
            expected_sample_count > 0,
            isinstance(expected_cache.get("expected_actor_ids"), list),
            bool(expected_cache.get("expected_actor_ids")),
            isinstance(expected_cache.get("payload_schema"), str),
            bool(expected_cache.get("payload_schema")),
            len(expected_rating_range) == 2,
            len(actual_rating_range) == 2,
            expected_cache.get("read_only") is True,
            cache.get("schema_version") == "vf_original_score_cache_v1",
            cache.get("expected_sha256") == expected_sha256,
            cache.get("read_only") is True,
            cache.get("db_path") == expected_cache.get("path"),
            actual_row_count == expected_row_count,
            actual_sample_count == expected_sample_count,
            cache.get("actor_ids") == expected_cache.get("expected_actor_ids"),
            cache.get("payload_schema") == expected_cache.get("payload_schema"),
            actual_rating_range == expected_rating_range,
            cache.get("judge_model_id") == expected_model_id,
            cache.get("judge_model_path") == expected_model_path,
            cache.get("judge_model_tree_sha256") == expected_model_tree_sha256,
            cache.get("judge_prompt_hash") == expected_prompt_hash,
        )
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row in {path}:{line_number}")
            rows.append(value)
    return rows


def metric_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def logging_rows(run_dir: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    candidates = sorted((run_dir / "train").glob("**/logging.jsonl"), key=lambda path: path.stat().st_mtime)
    path = candidates[-1] if candidates else None
    return path, load_jsonl(path) if path else []


def extract_step(row: dict[str, Any]) -> int | None:
    marker = row.get("global_step/max_steps")
    if isinstance(marker, str) and "/" in marker:
        try:
            return int(marker.split("/", 1)[0])
        except ValueError:
            return None
    value = row.get("global_step")
    return int(value) if isinstance(value, (int, float)) else None


def schema_valid(
    row: dict[str, Any],
    *,
    editor_judge_component_grpo: bool = False,
) -> bool:
    if editor_judge_component_grpo:
        status = row.get("editor_judge_status")
        success = status == "success"
        common = all(
            (
                row.get("actor_only") is True,
                row.get("actor_schema")
                == "reasoning_evidence_solution_rating",
                row.get("a1_text") == "",
                row.get("a1_eligible") is False,
                row.get("editor_backend") == "diffusers",
                status in {"success", "actor_ineligible", "service_error"},
                isinstance(row.get("j0"), (int, float)),
                isinstance(row.get("original_score_cache"), dict),
                row.get("rating_processed_before_editor_judge") is True,
                row.get("editor_judge_reasoning_reward") is True,
                isinstance(row.get("editor_attempts"), list),
                isinstance(row.get("judge_attempts"), list),
            )
        )
        if not common:
            return False
        if success:
            component_failure_owner_valid = (
                row.get("failure_owner") in {"none", "actor"}
                and row.get("service_failure_owner", "none") == "none"
                and row.get("reasoning_reward_eligible") is True
            )
            return all(
                (
                    isinstance(row.get("edited_image_path"), str),
                    bool(row.get("edited_image_path")),
                    isinstance(row.get("j1"), (int, float)),
                    isinstance(row.get("judge_delta"), (int, float)),
                    row.get("size_preserved") is True,
                    row.get("semantic_guardrail_applied") is False,
                    row.get("solution_only_applied") is True,
                    row.get("editor_prompt")
                    == str(row.get("solution") or "").strip(),
                    component_failure_owner_valid,
                )
            )
        return (
            row.get("j1") is None
            and row.get("judge_delta") is None
            and row.get("reasoning_reward_eligible") is False
        )
    return (
        row.get("actor_only") is True
        and row.get("actor_schema") == "reasons_rating"
        and row.get("suggestion") == ""
        and row.get("a1_text") == ""
        and row.get("j0") is None
        and row.get("j1") is None
        and row.get("edited_image_path") is None
        and row.get("editor_backend") == "disabled"
    )


def credit_valid(
    row: dict[str, Any],
    algorithm: str,
    *,
    editor_judge_component_grpo: bool = False,
) -> bool:
    credit = row.get("credit_assignment") or {}
    components = credit.get("components") if isinstance(credit, dict) else None
    if editor_judge_component_grpo:
        expected = {"format_a0", "rating0", "reasoning", "soft_overlong"}
        metadata_key = "editor_judge"
    else:
        expected = {"dapo_policy" if algorithm == "dapo" else "grpo_policy"}
        metadata_key = "dapo" if algorithm == "dapo" else "grpo"
    if not isinstance(components, dict) or set(components) != expected:
        return False
    if credit.get("trajectory_id") != row.get("trajectory_id"):
        return False
    if not isinstance(credit.get(metadata_key), dict):
        return False
    if editor_judge_component_grpo:
        metadata = credit[metadata_key]
        if not all(
            (
                metadata.get("schema_version")
                == "vf_editor_judge_reasoning_reward_v1",
                metadata.get("rating_reward") == "local_six_l2_margin",
                metadata.get("reasoning_reward")
                == "signed_l2_judge_delta",
                metadata.get("tau_s") == 1.0,
                metadata.get("division_by_four") is False,
                metadata.get("reward_population")
                == "same_image_six_completions",
                metadata.get("margin_reward_scope")
                == "local_six_images",
                metadata.get("margin_reward_population")
                == "complete_rank_local_six_image_cohort",
                metadata.get("reward_gather_order")
                == "local_reward_then_global_gather",
                metadata.get("reward_computed_before_global_gather")
                is True,
            )
        ):
            return False
        targets = {
            "format_a0": ["a0.all"],
            "rating0": ["a0.rating_content"],
            "reasoning": [
                "a0.reasoning.evidence_content",
                "a0.reasoning.solution_content",
            ],
            "soft_overlong": ["a0.completion_non_padding"],
        }
        if any(
            components[name].get("targets") != expected_targets
            for name, expected_targets in targets.items()
        ):
            return False
    for component in components.values():
        if not isinstance(component, dict):
            return False
        for key in ("raw_reward", "group_advantage", "weight", "weighted_advantage"):
            value = component.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return False
    return True


def padding_credit_valid(row: dict[str, Any]) -> bool:
    credit = row.get("credit_assignment") or {}
    component = (credit.get("components") or {}).get("dapo_policy")
    padding = credit.get("shape_padding") or {}
    if not isinstance(component, dict) or not isinstance(padding, dict):
        return False
    try:
        zeros = (
            float(component.get("group_advantage")) == 0.0
            and float(component.get("weight")) == 0.0
            and float(component.get("weighted_advantage")) == 0.0
            and float(padding.get("loss_weight")) == 0.0
        )
    except (TypeError, ValueError):
        return False
    return all(
        (
            component.get("eligible") is False,
            zeros,
            padding.get("enabled") is True,
            padding.get("token_denominator_eligible") is False,
        )
    )


def checkpoint_state(checkpoint: Path | None) -> dict[str, Any]:
    if checkpoint is None:
        return {"valid": True, "path": None}
    trainer_state = checkpoint / "trainer_state.json"
    scheduler = checkpoint / "scheduler.pt"
    rng_files = sorted(checkpoint.glob("rng_state*.pth"))
    optimizer_files = sorted(checkpoint.rglob("*optim_states.pt")) + sorted(checkpoint.glob("optimizer.pt"))
    model_files = sorted(checkpoint.glob("*.safetensors")) + sorted(checkpoint.glob("pytorch_model*.bin"))
    state: dict[str, Any] = {}
    if trainer_state.is_file():
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
    valid = all(
        (
            trainer_state.is_file(),
            scheduler.is_file(),
            bool(rng_files),
            bool(optimizer_files),
            bool(model_files),
        )
    )
    return {
        "valid": valid,
        "path": str(checkpoint),
        "trainer_global_step": state.get("global_step"),
        "trainer_epoch": state.get("epoch"),
        "rng_files": len(rng_files),
        "optimizer_state_files": len(optimizer_files),
        "model_weight_files": len(model_files),
    }


def telemetry_summary(path: Path) -> dict[str, float | None]:
    max_used: float | None = None
    max_util: float | None = None
    if not path.is_file():
        return {"max_memory_used_gib": None, "max_gpu_utilization_pct": None}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                used = float(row["memory_used_mib"]) / 1024.0
                util = float(row["utilization_gpu_pct"])
            except (KeyError, TypeError, ValueError):
                continue
            max_used = used if max_used is None else max(max_used, used)
            max_util = util if max_util is None else max(max_util, util)
    return {"max_memory_used_gib": max_used, "max_gpu_utilization_pct": max_util}


def origin_rank_coverage_valid(
    rank_row_counts: dict[str, int],
    rank_selected_counts: dict[str, int],
    *,
    algorithm: str,
    expected_calls: int,
    batch: int,
    expected_selected: int,
) -> bool:
    if not rank_row_counts or set(rank_row_counts) != set(rank_selected_counts):
        return False
    expected_per_rank = expected_calls * batch
    if algorithm == "grpo":
        return all(
            rank_row_counts[name] == expected_per_rank
            and rank_selected_counts[name] == expected_per_rank
            for name in rank_row_counts
        )

    # DAPO selects effective groups globally, then evenly redistributes the
    # selected batch to learner ranks. Selection counts by generation-origin
    # rank may therefore differ after a legal resampling round.
    generated = list(rank_row_counts.values())
    return all(
        (
            len(set(generated)) == 1,
            all(count >= expected_per_rank and count % batch == 0 for count in generated),
            sum(rank_selected_counts.values()) == expected_selected,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trainer-exit-code", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--expected-start-step", type=int, required=True)
    parser.add_argument("--expected-end-step", type=int, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "artifacts" / "config.json").read_text(encoding="utf-8"))
    algorithm = str(config["algorithm"])
    editor_judge_component_grpo = bool(
        config.get("editor_judge_reasoning_reward", False)
    )
    component_kl = config.get("component_kl") or {}
    component_kl_mode = str(component_kl.get("mode") or "off")
    global_completion_kl = config.get("global_completion_kl") or {}
    global_completion_kl_enabled = bool(
        global_completion_kl.get(
            "enabled",
            config.get("global_completion_kl_applied", False),
        )
    )
    world_size = int(config["world_size"])
    batch = int(config["per_device_batch_size"])
    global_batch = int(config["global_generation_batch_size"])
    iterations = int(config["num_iterations"])
    stage_steps = args.expected_end_step - args.expected_start_step
    expected_calls = math.ceil(stage_steps / iterations)

    train_log = run_dir / "logs" / "train.log"
    train_text = train_log.read_text(encoding="utf-8", errors="replace") if train_log.is_file() else ""
    lower = train_text.lower()
    package_preflight_log = run_dir / "logs" / "package_preflight.log"
    package_preflight_text = (
        package_preflight_log.read_text(encoding="utf-8", errors="replace")
        if package_preflight_log.is_file()
        else ""
    )
    package_preflight_lower = package_preflight_text.lower()
    launch_command_path = run_dir / "artifacts" / "launch_command.txt"
    launch_command = (
        launch_command_path.read_text(encoding="utf-8", errors="replace")
        if launch_command_path.is_file()
        else ""
    )
    explicit_oom = any(pattern in lower for pattern in OOM_PATTERNS)
    fast_path_fallback = "the fast path is not available" in lower

    log_path, all_logging = logging_rows(run_dir)
    step_by_number: dict[int, dict[str, Any]] = {}
    for row in all_logging:
        step = extract_step(row)
        if step is not None and "loss" in row and args.expected_start_step < step <= args.expected_end_step:
            item = dict(row)
            item["_step"] = step
            step_by_number[step] = item
    steps = [step_by_number[key] for key in sorted(step_by_number)]
    expected_step_numbers = list(range(args.expected_start_step + 1, args.expected_end_step + 1))
    completed_steps = sorted(step_by_number) == expected_step_numbers

    trajectory_paths = sorted((run_dir / "artifacts").glob("trajectory.rank*.jsonl"))
    rank_rows = {path.name: load_jsonl(path) for path in trajectory_paths}
    trajectories = [row for rows in rank_rows.values() for row in rows]
    selected = (
        [row for row in trajectories if row.get("dapo_selected_for_update") is True]
        if algorithm == "dapo"
        else list(trajectories)
    )
    physical = (
        [row for row in trajectories if row.get("dapo_physical_selected") is True]
        if algorithm == "dapo"
        else list(trajectories)
    )
    padding = (
        [row for row in trajectories if row.get("dapo_padding_for_shape") is True]
        if algorithm == "dapo"
        else []
    )
    generated_by_call = Counter(int(row.get("rollout_call") or 0) for row in trajectories)
    selected_by_call = Counter(int(row.get("rollout_call") or 0) for row in selected)
    physical_by_call = Counter(int(row.get("rollout_call") or 0) for row in physical)
    padding_by_call = Counter(int(row.get("rollout_call") or 0) for row in padding)
    min_effective_rows = int(config.get("min_effective_rows", global_batch))
    expected_calls_set = set(range(1, expected_calls + 1))
    trajectories_by_call: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        trajectories_by_call[int(row.get("rollout_call") or 0)].append(row)
    skipped_calls = {
        call
        for call, rows in trajectories_by_call.items()
        if rows and all(row.get("dapo_low_effective_batch_skipped") is True for row in rows)
    }
    mixed_skip_marker_calls = {
        call
        for call, rows in trajectories_by_call.items()
        if any(row.get("dapo_low_effective_batch_skipped") is True for row in rows)
        and call not in skipped_calls
    }
    update_calls = expected_calls_set - skipped_calls
    expected_physical = len(update_calls) * global_batch
    skipped_step_numbers: set[int] = set()
    for call in skipped_calls:
        first = args.expected_start_step + (call - 1) * iterations + 1
        skipped_step_numbers.update(
            range(first, min(first + iterations, args.expected_end_step + 1))
        )
    expected_optimizer_updates = stage_steps - len(skipped_step_numbers)
    ids = [str(row.get("trajectory_id") or "") for row in trajectories]

    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    group_field = "dapo_group_key" if algorithm == "dapo" else "group_id"
    for row in trajectories:
        groups[(int(row.get("rollout_call") or 0), str(row.get(group_field) or ""))].append(row)
    if algorithm == "dapo":
        group_integrity = all(
            call > 0
            and key
            and len(rows) == int(config["num_generations"])
            and len({bool(row.get("dapo_effective_group")) for row in rows}) == 1
            and sum(bool(row.get("dapo_selected_for_update")) for row in rows) in {0, len(rows)}
            and sum(bool(row.get("dapo_padding_for_shape")) for row in rows) in {0, len(rows)}
            and sum(bool(row.get("dapo_physical_selected")) for row in rows) in {0, len(rows)}
            and all(
                bool(row.get("dapo_physical_selected"))
                == (
                    bool(row.get("dapo_selected_for_update"))
                    or bool(row.get("dapo_padding_for_shape"))
                )
                for row in rows
            )
            and all(
                not (
                    bool(row.get("dapo_selected_for_update"))
                    and bool(row.get("dapo_padding_for_shape"))
                )
                for row in rows
            )
            and all(
                not bool(row.get("dapo_selected_for_update"))
                or bool(row.get("dapo_effective_group"))
                for row in rows
            )
            and all(
                not bool(row.get("dapo_padding_for_shape"))
                or not bool(row.get("dapo_effective_group"))
                for row in rows
            )
            for (call, key), rows in groups.items()
        )
    else:
        group_integrity = all(
            call > 0 and key and len(rows) == int(config["num_generations"])
            for (call, key), rows in groups.items()
        )

    rank_selected_counts = {
        name: sum(
            row.get("dapo_selected_for_update") is True if algorithm == "dapo" else True
            for row in rows
        )
        for name, rows in rank_rows.items()
    }
    rank_physical_counts = {
        name: sum(
            row.get("dapo_physical_selected") is True if algorithm == "dapo" else True
            for row in rows
        )
        for name, rows in rank_rows.items()
    }
    learner_rank_physical_counts = Counter(
        int(row.get("dapo_learner_rank"))
        for row in physical
        if row.get("dapo_learner_rank") is not None
    )
    learner_rank_active_counts = Counter(
        int(row.get("dapo_learner_rank"))
        for row in selected
        if row.get("dapo_learner_rank") is not None
    )
    learner_rank_physical_by_call = Counter(
        (int(row.get("rollout_call") or 0), int(row.get("dapo_learner_rank")))
        for row in physical
        if row.get("dapo_learner_rank") is not None
    )
    learner_rank_active_by_call = Counter(
        (int(row.get("rollout_call") or 0), int(row.get("dapo_learner_rank")))
        for row in selected
        if row.get("dapo_learner_rank") is not None
    )
    rank_row_counts = {name: len(rows) for name, rows in rank_rows.items()}
    origin_rank_coverage_ok = origin_rank_coverage_valid(
        rank_row_counts,
        rank_physical_counts,
        algorithm=algorithm,
        expected_calls=expected_calls,
        batch=batch,
        expected_selected=expected_physical,
    )
    effective_by_call = Counter(
        int(row.get("rollout_call") or 0)
        for row in trajectories
        if row.get("dapo_effective_group") is True
    )
    partial_calls_retain_all_effective = all(
        padding_by_call[call] == 0
        or selected_by_call[call] == effective_by_call[call]
        for call in update_calls
    )
    low_effective_skip_config_ok = algorithm != "dapo" or all(
        (
            config.get("low_effective_action") in {"error", "skip_batch"},
            config.get("low_effective_action") != "skip_batch"
            or config.get("low_effective_skip_contract")
            == {
                "consume_data_batch": True,
                "backward": False,
                "optimizer_update": False,
                "scheduler_update": False,
                "global_step_advances": True,
            },
        )
    )
    low_effective_skip_contract_ok = algorithm != "dapo" or all(
        (
            not mixed_skip_marker_calls,
            skipped_calls <= expected_calls_set,
            config.get("low_effective_action") == "skip_batch" or not skipped_calls,
            all(
                0 <= effective_by_call[call] < min_effective_rows
                and selected_by_call[call] == 0
                and physical_by_call[call] == 0
                and padding_by_call[call] == 0
                and all(
                    row.get("dapo_selected_for_update") is False
                    and row.get("dapo_physical_selected") is False
                    and row.get("dapo_padding_for_shape") is False
                    and row.get("dapo_update_executed") is False
                    and row.get("dapo_low_effective_action") == "skip_batch"
                    and float(row.get("dapo_policy_loss_weight", float("nan"))) == 0.0
                    and row.get("dapo_token_denominator_eligible") is False
                    and int(row.get("dapo_min_effective_rows") or 0)
                    == min_effective_rows
                    for row in trajectories_by_call[call]
                )
                for call in skipped_calls
            ),
        )
    )
    padding_credit_ok = all(padding_credit_valid(row) for row in padding)
    reward_population_ok = algorithm != "dapo" or all(
        (
            config.get("reward_population")
            == "all_complete_groups_in_sampling_round",
            config.get("reward_computed_before_effective_filter") is True,
            config.get("ineffective_groups_participate_in_reward") is True,
            all(
                row.get("dapo_reward_population")
                == "all_complete_groups_in_sampling_round"
                and row.get("dapo_reward_computed_before_effective_filter") is True
                and row.get("dapo_ineffective_groups_participate_in_reward") is True
                and ((row.get("credit_assignment") or {}).get("dapo") or {}).get(
                    "reward_population"
                )
                == "all_complete_groups_in_sampling_round"
                and ((row.get("credit_assignment") or {}).get("dapo") or {}).get(
                    "reward_computed_before_effective_filter"
                )
                is True
                and ((row.get("credit_assignment") or {}).get("dapo") or {}).get(
                    "ineffective_groups_participate_in_reward"
                )
                is True
                for row in trajectories
            ),
        )
    )
    dapo_shape_contract_ok = algorithm != "dapo" or all(
        (
            config.get("partial_effective_batch_padding") is True,
            int(config.get("padding_target_rows", 0)) == global_batch,
            int(config.get("padding_group_size", 0)) == int(config["num_generations"]),
            float(config.get("padding_loss_weight", float("nan"))) == 0.0,
            config.get("padding_token_denominator_eligible") is False,
            set(physical_by_call) == update_calls,
            all(physical_by_call[call] == global_batch for call in update_calls),
            set(selected_by_call) == update_calls,
            all(
                min_effective_rows <= selected_by_call[call] <= global_batch
                and selected_by_call[call] % int(config["num_generations"]) == 0
                for call in update_calls
            ),
            all(
                selected_by_call[call] + padding_by_call[call] == global_batch
                for call in update_calls
            ),
            partial_calls_retain_all_effective,
            padding_credit_ok,
            reward_population_ok,
            low_effective_skip_config_ok,
            low_effective_skip_contract_ok,
            set(learner_rank_physical_counts) == set(range(world_size)),
            all(
                learner_rank_physical_counts[rank] == len(update_calls) * batch
                for rank in range(world_size)
            ),
            all(
                learner_rank_physical_by_call[(call, rank)] == batch
                and learner_rank_active_by_call[(call, rank)]
                >= int(config["num_generations"])
                for call in update_calls
                for rank in range(world_size)
            ),
        )
    )
    coverage_valid = all(
        (
            len(trajectory_paths) == world_size,
            bool(trajectories),
            len(physical) == expected_physical,
            set(generated_by_call) == expected_calls_set,
            origin_rank_coverage_ok,
            len(set(ids)) == len(ids),
            all(ids),
            group_integrity,
            dapo_shape_contract_ok,
        )
    )
    schema_ok = bool(trajectories) and all(
        schema_valid(
            row,
            editor_judge_component_grpo=editor_judge_component_grpo,
        )
        for row in trajectories
    )
    credit_ok = bool(trajectories) and all(
        credit_valid(
            row,
            algorithm,
            editor_judge_component_grpo=editor_judge_component_grpo,
        )
        for row in trajectories
    )
    selected_ok = bool(selected) and all(int(row.get("a0_token_length") or 0) > 0 for row in selected)
    finite_ok = all(finite(row) for row in trajectories) and all(finite(row) for row in steps)
    editor_judge_success_rows = [
        row
        for row in trajectories
        if row.get("editor_judge_status") == "success"
    ]
    editor_judge_ineligible_rows = [
        row
        for row in trajectories
        if row.get("editor_judge_status") != "success"
    ]
    editor_judge_group_stats_ok = True
    editor_judge_service_contract_ok = True
    if editor_judge_component_grpo:
        expected_cache = config.get("original_score_cache") or {}
        expected_judge = config.get("judge") or {}
        expected_overlong = config.get("soft_overlong") or {}
        for row in trajectories:
            cache = row.get("original_score_cache") or {}
            credit = row.get("credit_assignment") or {}
            components = credit.get("components") or {}
            reasoning_component = components.get("reasoning") or {}
            overlong_component = components.get("soft_overlong") or {}
            delta = row.get("judge_delta")
            reasoning_eligible = row.get("reasoning_reward_eligible") is True
            expected_reasoning_reward = 0.0
            if reasoning_eligible and isinstance(delta, (int, float)):
                magnitude = -math.expm1(-(float(delta) ** 2) / 2.0)
                expected_reasoning_reward = math.copysign(magnitude, float(delta))
            max_length = int(expected_overlong.get("max_length", 160))
            cache_length = int(expected_overlong.get("cache_length", 16))
            max_penalty = float(expected_overlong.get("max_penalty", 1.0))
            expected_overlong_reward = -min(
                max(
                    int(row.get("a0_token_length") or 0)
                    - (max_length - cache_length),
                    0,
                )
                / cache_length,
                1.0,
            ) * max_penalty
            row_ok = all(
                (
                    original_score_cache_contract_valid(
                        cache,
                        expected_cache,
                        expected_judge,
                    ),
                    math.isclose(
                        float(row.get("reasoning_raw_reward") or 0.0),
                        expected_reasoning_reward,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    reasoning_component.get("eligible") is reasoning_eligible,
                    math.isclose(
                        float(reasoning_component.get("raw_reward") or 0.0),
                        expected_reasoning_reward,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    overlong_component.get("eligible") is True,
                    math.isclose(
                        float(overlong_component.get("raw_reward") or 0.0),
                        expected_overlong_reward,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                )
            )
            if row.get("editor_judge_status") == "success":
                row_ok = row_ok and all(
                    (
                        Path(str(row.get("edited_image_path") or "")).is_file(),
                        math.isclose(
                            float(row["judge_delta"]),
                            float(row["j1"]) - float(row["j0"]),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        ),
                        row.get("semantic_guardrail_applied") is False,
                        row.get("solution_only_applied") is True,
                        row.get("editor_prompt")
                        == str(row.get("solution") or "").strip(),
                        row.get("size_preserved") is True,
                        any(
                            attempt.get("success") is True
                            and int(attempt.get("gpu_index", -1))
                            in {4, 5, 6, 7}
                            for attempt in row.get("editor_attempts", [])
                        ),
                        any(
                            attempt.get("success") is True
                            and int(attempt.get("gpu_index", -1))
                            in {4, 5, 6, 7}
                            for attempt in row.get("judge_attempts", [])
                        ),
                    )
                )
            else:
                row_ok = row_ok and all(
                    (
                        reasoning_component.get("eligible") is False,
                        float(reasoning_component.get("raw_reward") or 0.0)
                        == 0.0,
                        float(reasoning_component.get("group_advantage") or 0.0)
                        == 0.0,
                    )
                )
            editor_judge_service_contract_ok = (
                editor_judge_service_contract_ok and row_ok
            )

        editor_groups: dict[tuple[int, int, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for row in trajectories:
            editor_groups[
                (
                    int(row.get("rollout_call") or 0),
                    int(row.get("rank") or 0),
                    str(row.get("group_id") or ""),
                )
            ].append(row)
        for rows in editor_groups.values():
            eligible = [
                row
                for row in rows
                if row.get("reasoning_reward_eligible") is True
            ]
            deltas = [float(row["judge_delta"]) for row in eligible]
            rewards = [float(row["reasoning_raw_reward"]) for row in eligible]

            def summary(values: list[float]) -> tuple[float | None, float | None, float | None]:
                if not values:
                    return None, None, None
                mean = math.fsum(values) / len(values)
                variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
                return mean, variance, math.sqrt(variance)

            delta_mean, delta_variance, delta_std = summary(deltas)
            reward_mean, reward_variance, reward_std = summary(rewards)
            expected_stats = {
                "group_size": 6,
                "reasoning_eligible_count": len(eligible),
                "judge_delta_mean": delta_mean,
                "judge_delta_variance": delta_variance,
                "judge_delta_std": delta_std,
                "signed_reward_mean": reward_mean,
                "signed_reward_variance": reward_variance,
                "signed_reward_std": reward_std,
                "tau_s": 1.0,
            }
            editor_judge_group_stats_ok = (
                editor_judge_group_stats_ok
                and len(rows) == 6
                and all(
                    row.get("editor_judge_group_stats") == expected_stats
                    for row in rows
                )
            )

    margin_scope = str(config.get("margin_reward_scope") or "global_batch")
    local_margin_cohorts: dict[tuple[int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    local_margin_rounds: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    local_margin_metadata_ok = True
    if margin_scope == "local_six_images":
        for row in trajectories:
            call = int(row.get("rollout_call") or 0)
            rank = int(row.get("rank") if row.get("rank") is not None else -1)
            sampling_round = int(row.get("sampling_round") or 0)
            cohort_id = str(row.get("margin_cohort_id") or "")
            algorithm_metadata = (row.get("credit_assignment") or {}).get(
                "editor_judge"
                if editor_judge_component_grpo
                else ("dapo" if algorithm == "dapo" else "grpo")
            ) or {}
            local_margin_metadata_ok = local_margin_metadata_ok and all(
                (
                    call > 0,
                    0 <= rank < world_size,
                    bool(cohort_id),
                    row.get("margin_reward_scope") == "local_six_images",
                    int(row.get("margin_cohort_image_count") or 0) == 6,
                    int(row.get("margin_cohort_completion_count") or 0) == 36,
                    row.get("reward_gather_order") == "local_reward_then_global_gather",
                    row.get("reward_computed_before_global_gather") is True,
                    algorithm_metadata.get("margin_reward_scope")
                    == "local_six_images",
                    algorithm_metadata.get("margin_cohort_id") == cohort_id,
                    int(
                        algorithm_metadata.get("margin_cohort_image_count") or 0
                    )
                    == 6,
                    algorithm_metadata.get("reward_gather_order")
                    == "local_reward_then_global_gather",
                    algorithm_metadata.get("reward_computed_before_global_gather")
                    is True,
                    algorithm_metadata.get("reward_population")
                    == config.get("reward_population"),
                )
            )
            local_margin_cohorts[(call, rank, sampling_round, cohort_id)].append(row)
            local_margin_rounds[(call, rank, sampling_round)].add(cohort_id)
        local_margin_cohort_integrity = bool(local_margin_cohorts) and all(
            len(rows) == 36
            and len({str(row.get("dapo_group_key") or "") for row in rows}) == 6
            and all(
                sum(
                    str(candidate.get("dapo_group_key") or "") == group_key
                    for candidate in rows
                )
                == int(config["num_generations"])
                for group_key in {str(row.get("dapo_group_key") or "") for row in rows}
            )
            for rows in local_margin_cohorts.values()
        )
        local_margin_round_integrity = bool(local_margin_rounds) and all(
            len(cohort_ids) == int(config.get("margin_cohorts_per_rank", 0))
            for cohort_ids in local_margin_rounds.values()
        )
        local_margin_config_ok = all(
            (
                algorithm in {"dapo", "grpo"},
                world_size == (4 if editor_judge_component_grpo else 8),
                int(config.get("num_generations", 0)) == 6,
                int(config.get("margin_images_per_cohort", 0)) == 6,
                int(config.get("margin_local_images_per_rank", 0)) > 0,
                batch
                == int(config.get("margin_local_images_per_rank", 0))
                * int(config.get("num_generations", 0)),
                int(config.get("margin_local_images_per_rank", 0))
                % int(config.get("margin_images_per_cohort", 1))
                == 0,
                int(config.get("margin_cohorts_per_rank", 0))
                == int(config.get("margin_local_images_per_rank", 0))
                // int(config.get("margin_images_per_cohort", 1)),
                config.get("reward_gather_order") == "local_reward_then_global_gather",
                config.get("reward_computed_before_global_gather") is True,
            )
        )
        local_margin_contract_ok = all(
            (
                local_margin_metadata_ok,
                local_margin_cohort_integrity,
                local_margin_round_integrity,
                local_margin_config_ok,
            )
        )
    else:
        local_margin_cohort_integrity = True
        local_margin_round_integrity = True
        local_margin_config_ok = True
        local_margin_contract_ok = True

    def metric(key: str) -> list[float]:
        return [float(row[key]) for row in steps if isinstance(row.get(key), (int, float))]

    reasoning_kl_metrics = metric("vf/a0_reasoning_kl_loss")
    rating_kl_metrics = metric("vf/a0_rating0_kl_loss")
    global_kl_mean_metrics = metric("vf/global_completion_kl_mean")
    global_kl_loss_metrics = metric("vf/global_completion_kl_loss")
    global_kl_apply_metrics = metric("vf/global_completion_kl_apply_count")
    component_kl_apply_metrics = metric("vf/component_kl_apply_count")
    if component_kl_mode == "field":
        segments = component_kl.get("segments") or {}
        reasoning_segment = segments.get("reasoning") or {}
        rating_segment = segments.get("rating0") or {}
        reference_path = str(component_kl.get("reference_model_path") or "")
        reference_tree = str(
            component_kl.get("reference_model_tree_sha256") or ""
        )
        expected_activation_beta = float(
            component_kl.get(
                "expected_reference_activation_beta",
                config.get("reference_activation_beta", -1.0),
            )
        )
        expected_reasoning_beta = float(
            component_kl.get(
                "expected_reasoning_beta",
                reasoning_segment.get("beta", -1.0),
            )
        )
        expected_rating_beta = float(
            component_kl.get(
                "expected_rating_beta",
                rating_segment.get("beta", -1.0),
            )
        )
        expected_reference_tree = str(
            config.get("initial_actor_tree_sha256") or reference_tree
        )
        component_kl_contract_ok = all(
            (
                config.get("component_credit_mask_mode") == "field",
                config.get("credit_mask_disabled") is False,
                config.get("global_completion_kl_applied") is False,
                component_kl.get("expected_mode", component_kl_mode)
                == component_kl_mode,
                component_kl.get("estimator") == "sampled_k3",
                component_kl.get("normalization")
                == "per_sequence_segment_token_mean_then_active_sequence_mean",
                component_kl.get("loss_sign") == "positive_regularization",
                math.isclose(
                    float(config.get("reference_activation_beta", 0.0)),
                    expected_activation_beta,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                math.isclose(
                    float(reasoning_segment.get("beta", -1.0)),
                    expected_reasoning_beta,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                math.isclose(
                    float(rating_segment.get("beta", -1.0)),
                    expected_rating_beta,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                reasoning_segment.get("token_targets")
                == [
                    "a0.reasoning.evidence_content",
                    "a0.reasoning.solution_content",
                ],
                rating_segment.get("token_targets") == ["a0.rating_content"],
                bool(reference_path),
                reference_path == str(config.get("model_path") or ""),
                bool(re.fullmatch(r"[0-9a-f]{64}", reference_tree)),
                reference_tree == expected_reference_tree,
            )
        )
        component_kl_metrics_ok = all(
            (
                len(reasoning_kl_metrics) == expected_optimizer_updates,
                len(rating_kl_metrics) == expected_optimizer_updates,
                all(
                    math.isfinite(value) and value >= 0.0
                    for value in reasoning_kl_metrics
                ),
                all(
                    math.isfinite(value) and value >= 0.0
                    for value in rating_kl_metrics
                ),
            )
        )
    else:
        component_kl_contract_ok = all(
            (
                component_kl_mode == "off",
                component_kl.get("expected_mode", component_kl_mode) == "off",
                not (component_kl.get("segments") or {}),
                math.isclose(
                    float(component_kl.get("expected_reasoning_beta", 0.0)),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                math.isclose(
                    float(component_kl.get("expected_rating_beta", 0.0)),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
            )
        )
        component_kl_metrics_ok = (
            not reasoning_kl_metrics and not rating_kl_metrics
        )

    if global_completion_kl_enabled:
        global_kl_beta = float(global_completion_kl.get("beta", -1.0))
        reference_path = str(component_kl.get("reference_model_path") or "")
        reference_tree = str(component_kl.get("reference_model_tree_sha256") or "")
        expected_reference_tree = str(config.get("initial_actor_tree_sha256") or "")
        global_completion_kl_contract_ok = all(
            (
                component_kl_mode == "off",
                config.get("component_credit_mask_mode") == "completion",
                config.get("credit_mask_disabled") is True,
                config.get("global_completion_kl_applied") is True,
                component_kl.get("global_completion_kl_applied") is True,
                config.get("kl_in_reward") is False,
                global_completion_kl.get("kl_in_reward") is False,
                global_completion_kl.get("estimator") == "sampled_k3",
                global_completion_kl.get("normalization")
                == "per_sequence_completion_token_mean_then_active_sequence_mean",
                global_completion_kl.get("loss_sign") == "positive_regularization",
                global_completion_kl.get("token_targets")
                == ["a0.active_eligible_completion_non_padding"],
                global_kl_beta > 0.0,
                math.isclose(
                    global_kl_beta,
                    float(config.get("reference_activation_beta", -1.0)),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                reference_path == str(config.get("model_path") or ""),
                bool(re.fullmatch(r"[0-9a-f]{64}", reference_tree)),
                reference_tree == expected_reference_tree,
            )
        )
        global_completion_kl_metrics_ok = all(
            (
                len(global_kl_mean_metrics) == expected_optimizer_updates,
                len(global_kl_loss_metrics) == expected_optimizer_updates,
                len(global_kl_apply_metrics) == expected_optimizer_updates,
                len(component_kl_apply_metrics) == expected_optimizer_updates,
                all(
                    math.isfinite(value) and value >= 0.0
                    for value in global_kl_mean_metrics + global_kl_loss_metrics
                ),
                all(
                    math.isclose(loss, global_kl_beta * mean, rel_tol=1e-6, abs_tol=1e-8)
                    for mean, loss in zip(global_kl_mean_metrics, global_kl_loss_metrics)
                ),
                all(
                    math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
                    for value in global_kl_apply_metrics
                ),
                all(
                    math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12)
                    for value in component_kl_apply_metrics
                ),
                not reasoning_kl_metrics,
                not rating_kl_metrics,
            )
        )
    else:
        global_completion_kl_contract_ok = all(
            (
                config.get("global_completion_kl_applied") is False,
                global_completion_kl.get("enabled", False) is False,
            )
        )
        global_completion_kl_metrics_ok = not any(
            (
                global_kl_mean_metrics,
                global_kl_loss_metrics,
                global_kl_apply_metrics,
            )
        )

    chunks = metric("vf/backward_chunk_count")
    expected_backward_chunks = int(config.get("backward_chunks_per_rank", 1))
    backward_ok = len(chunks) == stage_steps and all(
        int(row["vf/backward_chunk_count"])
        == (0 if int(row["_step"]) in skipped_step_numbers else expected_backward_chunks)
        for row in steps
    )
    finite_step_metrics = len(steps) == stage_steps and all(
        math.isfinite(float(row[key]))
        for row in steps
        for key in ("loss", "grad_norm", "vf/backward_seconds")
        if isinstance(row.get(key), (int, float))
    )
    dapo_generation_metric_keys = (
        "vf/dapo_generation_rounds",
        "vf/dapo_generated_rows",
        "vf/dapo_effective_rows",
        "vf/dapo_acceptance_rate",
        "vf/dapo_selected_rows",
        "vf/dapo_local_selected_rows",
        "vf/dapo_effective_selected_rows",
        "vf/dapo_padding_rows",
        "vf/dapo_local_effective_selected_rows",
        "vf/dapo_local_padding_rows",
        "vf/dapo_partial_batch_fallback",
        "vf/dapo_min_effective_rows",
        "vf/dapo_low_effective_batch_skipped",
    )
    dapo_training_metric_keys = (
        "vf/dapo_global_policy_tokens",
        "vf/dapo_token_mean_normalizer",
        "vf/dapo_physical_policy_tokens",
        "vf/dapo_training_padding_rows",
    )
    dapo_metrics_ok = True
    dapo_selected_batch_ok = True
    token_normalization_ok = True
    if algorithm == "dapo":
        global_selected_metrics = metric("vf/dapo_selected_rows")
        local_selected_metrics = metric("vf/dapo_local_selected_rows")
        effective_selected_metrics = metric("vf/dapo_effective_selected_rows")
        padding_metrics = metric("vf/dapo_padding_rows")
        local_effective_metrics = metric("vf/dapo_local_effective_selected_rows")
        local_padding_metrics = metric("vf/dapo_local_padding_rows")
        fallback_metrics = metric("vf/dapo_partial_batch_fallback")
        low_effective_skip_metrics = metric("vf/dapo_low_effective_batch_skipped")
        dapo_metrics_ok = all(
            len(metric(key)) == expected_calls for key in dapo_generation_metric_keys
        ) and all(
            len(metric(key)) == expected_optimizer_updates
            for key in dapo_training_metric_keys
        ) and all(
            len(metric(key)) == stage_steps
            for key in (
                "vf/optimizer_update_executed",
                "vf/scheduler_update_executed",
                "vf/dapo_low_effective_training_skip",
            )
        ) and all(
            math.isclose(
                float(row["vf/optimizer_update_executed"]),
                float(int(row["_step"]) not in skipped_step_numbers),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                float(row["vf/scheduler_update_executed"]),
                float(int(row["_step"]) not in skipped_step_numbers),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                float(row["vf/dapo_low_effective_training_skip"]),
                float(int(row["_step"]) in skipped_step_numbers),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for row in steps
        ) and all(
            math.isclose(
                value,
                float(call in skipped_calls),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for call, value in enumerate(low_effective_skip_metrics, 1)
        )
        dapo_selected_batch_ok = (
            len(global_selected_metrics) == expected_calls
            and len(local_selected_metrics) == expected_calls
            and all(
                (
                    math.isclose(global_selected, 0.0, rel_tol=0.0, abs_tol=1e-6)
                    and math.isclose(local_selected, 0.0, rel_tol=0.0, abs_tol=1e-6)
                    and math.isclose(effective, 0.0, rel_tol=0.0, abs_tol=1e-6)
                    and math.isclose(pad, 0.0, rel_tol=0.0, abs_tol=1e-6)
                    and math.isclose(local_effective, 0.0, rel_tol=0.0, abs_tol=1e-6)
                    and math.isclose(local_pad, 0.0, rel_tol=0.0, abs_tol=1e-6)
                    and math.isclose(fallback, 0.0, rel_tol=0.0, abs_tol=1e-6)
                    if call in skipped_calls
                    else math.isclose(
                        global_selected, global_batch, rel_tol=0.0, abs_tol=1e-6
                    )
                    and math.isclose(local_selected, batch, rel_tol=0.0, abs_tol=1e-6)
                    and min_effective_rows <= effective <= global_batch
                    and int(effective) % int(config["num_generations"]) == 0
                    and math.isclose(
                        effective + pad, global_batch, rel_tol=0.0, abs_tol=1e-6
                    )
                    and math.isclose(
                        fallback, float(pad > 0), rel_tol=0.0, abs_tol=1e-6
                    )
                    and local_effective > 0
                    and int(local_effective) % int(config["num_generations"]) == 0
                    and math.isclose(
                        local_effective + local_pad,
                        batch,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                )
                for call, (
                    global_selected,
                    local_selected,
                    effective,
                    pad,
                    local_effective,
                    local_pad,
                    fallback,
                ) in enumerate(zip(
                    global_selected_metrics,
                    local_selected_metrics,
                    effective_selected_metrics,
                    padding_metrics,
                    local_effective_metrics,
                    local_padding_metrics,
                    fallback_metrics,
                ), 1)
            )
        )
        token_normalization_ok = len(metric("vf/dapo_global_policy_tokens")) == expected_optimizer_updates and all(
            math.isclose(global_tokens / world_size, normalizer, rel_tol=0.0, abs_tol=1e-6)
            for global_tokens, normalizer in zip(
                metric("vf/dapo_global_policy_tokens"),
                metric("vf/dapo_token_mean_normalizer"),
            )
        ) and all(
            physical_tokens >= global_tokens > 0
            for physical_tokens, global_tokens in zip(
                metric("vf/dapo_physical_policy_tokens"),
                metric("vf/dapo_global_policy_tokens"),
            )
        )

    checkpoints = sorted(path for path in (run_dir / "train").rglob("checkpoint-*") if path.is_dir())
    checkpoint = checkpoints[0] if len(checkpoints) == 1 else None
    checkpoint_required = config["mode"] in {"formal", "steps"}
    checkpoint_evidence = checkpoint_state(checkpoint) if checkpoint_required else checkpoint_state(None)
    if checkpoint_required and checkpoint is not None:
        checkpoint_evidence["global_step_matches"] = int(checkpoint_evidence.get("trainer_global_step") or -1) == args.expected_end_step
        checkpoint_evidence["epoch_matches"] = (
            config["mode"] == "steps"
            or math.isclose(
                float(checkpoint_evidence.get("trainer_epoch") or -1.0),
                float(config["target_epoch"]),
                rel_tol=0.0,
                abs_tol=1e-4,
            )
        )
        checkpoint_ok = bool(
            checkpoint_evidence["valid"]
            and checkpoint_evidence["global_step_matches"]
            and checkpoint_evidence["epoch_matches"]
        )
    else:
        checkpoint_ok = not checkpoint_required

    qwen35_fast_path_evidence = {
        "package_preflight": "qwen3.5 fused fast path true" in package_preflight_lower,
        "flash_attention_configured": "--attn_impl flash_attention_2" in lower,
        "vllm_fla_runtime": "using triton/fla gdn prefill kernel" in lower,
        "vllm_fa2_runtime": "using flashattention version 2" in lower,
        "fallback_detected": fast_path_fallback,
    }
    qwen35_fast_path_ok = config["model_family"] != "qwen35" or all(
        (
            qwen35_fast_path_evidence["package_preflight"],
            qwen35_fast_path_evidence["flash_attention_configured"],
            qwen35_fast_path_evidence["vllm_fla_runtime"],
            qwen35_fast_path_evidence["vllm_fa2_runtime"],
            not qwen35_fast_path_evidence["fallback_detected"],
        )
    )
    full_visual_no_grad_requested = bool(
        config.get("require_full_visual_no_grad_runtime", False)
    )
    full_visual_no_grad_ranks = sorted(
        {
            int(rank)
            for rank in re.findall(
                r"\[vf-visual-freeze\] rank=(\d+) "
                r"parameters=\d+ trainable=0 "
                r"(?:gc_configured=[01] gc_runtime_effective=0 )?"
                r"gc_modules=0 input_grad_hooks=0",
                train_text,
            )
        }
    )
    expected_visual_freeze_ranks = list(range(world_size))
    full_visual_no_grad_ok = not full_visual_no_grad_requested or (
        full_visual_no_grad_ranks == expected_visual_freeze_ranks
    )
    full_visual_no_grad_evidence = {
        "requested": full_visual_no_grad_requested,
        "confirmed_ranks": full_visual_no_grad_ranks,
        "expected_ranks": expected_visual_freeze_ranks,
        "trainable_visual_parameters": 0 if full_visual_no_grad_ok else None,
        "visual_gc_modules": 0 if full_visual_no_grad_ok else None,
        "visual_input_grad_hooks": 0 if full_visual_no_grad_ok else None,
    }
    selective_gc_requested = (
        config.get("gradient_checkpointing_scope") == "language_only"
    )
    selective_gc_rows = [
        (int(rank), int(language_gc), int(other_gc))
        for rank, language_gc, other_gc in re.findall(
            r"\[vf-selective-gc\] rank=(\d+) "
            r"vision_gc_configured=\d+ "
            r"vision_gc_runtime_effective=\d+ "
            r"language_gc=(\d+) "
            r"visual_gc=0 other_gc=(\d+) visual_trainable=0 "
            r"visual_input_grad_hooks=0 exact_restore=1",
            train_text,
        )
    ]
    selective_gc_ranks = sorted({rank for rank, _, _ in selective_gc_rows})
    selective_gc_ok = not selective_gc_requested or (
        selective_gc_ranks == expected_visual_freeze_ranks
        and all(language_gc > 0 for _, language_gc, _ in selective_gc_rows)
    )
    selective_gc_evidence = {
        "requested": selective_gc_requested,
        "confirmed_ranks": selective_gc_ranks,
        "expected_ranks": expected_visual_freeze_ranks,
        "language_gc_counts": {
            str(rank): language_gc
            for rank, language_gc, _ in selective_gc_rows
        },
        "visual_gc_modules": 0 if selective_gc_ok else None,
        "visual_trainable_parameters": 0 if selective_gc_ok else None,
        "visual_input_grad_hooks": 0 if selective_gc_ok else None,
        "scope_restore": config.get("gradient_checkpointing_scope_restore"),
    }
    frozen_vision_gc_rows = [
        (
            int(rank),
            int(vision_configured),
            int(vision_effective),
            int(language_gc),
            int(visual_gc),
            int(other_gc),
        )
        for (
            rank,
            vision_configured,
            vision_effective,
            language_gc,
            visual_gc,
            other_gc,
        ) in re.findall(
            r"\[vf-selective-gc\] rank=(\d+) "
            r"vision_gc_configured=(\d+) "
            r"vision_gc_runtime_effective=(\d+) "
            r"language_gc=(\d+) visual_gc=(\d+) other_gc=(\d+) "
            r"visual_trainable=0 visual_input_grad_hooks=0 exact_restore=1",
            train_text,
        )
    ]
    frozen_vision_gc_requested = editor_judge_component_grpo
    expected_language_gc = bool(config.get("language_gc_configured", False))
    frozen_vision_gc_ok = not frozen_vision_gc_requested or all(
        (
            sorted({row[0] for row in frozen_vision_gc_rows})
            == expected_visual_freeze_ranks,
            all(
                configured == 1
                and effective == 0
                and ((language_gc > 0) == expected_language_gc)
                and visual_gc == 0
                for (
                    _,
                    configured,
                    effective,
                    language_gc,
                    visual_gc,
                    _,
                ) in frozen_vision_gc_rows
            ),
        )
    )
    frozen_vision_gc_evidence = {
        "requested": frozen_vision_gc_requested,
        "confirmed_ranks": sorted({row[0] for row in frozen_vision_gc_rows}),
        "expected_ranks": expected_visual_freeze_ranks,
        "configured": True if frozen_vision_gc_ok and frozen_vision_gc_requested else None,
        "runtime_effective": False if frozen_vision_gc_ok and frozen_vision_gc_requested else None,
        "language_gc_active": (
            expected_language_gc
            if frozen_vision_gc_ok and frozen_vision_gc_requested
            else None
        ),
    }
    activation_offload_requested = bool(
        config.get("learner_activation_offload", False)
    )
    activation_offload_rows = [
        {
            "rank": int(rank),
            "budget_bytes": int(budget),
            "offloaded_bytes": int(offloaded),
            "offloaded_tensors": int(tensors),
            "seen_cuda_bytes": int(seen),
        }
        for rank, budget, offloaded, tensors, seen in re.findall(
            r"\[vf-activation-offload\] rank=(\d+) enabled=1 "
            r"backend=torch\.autograd\.graph\.saved_tensors_hooks\.selective_cpu_v1 "
            r"budget_bytes=(\d+) min_tensor_mib=16 "
            r"offloaded_bytes=(\d+) offloaded_tensors=(\d+) "
            r"seen_cuda_bytes=(\d+) pin_memory=0 exact_autograd=1",
            train_text,
        )
    ]
    activation_offload_ranks = sorted({row["rank"] for row in activation_offload_rows})
    activation_budget_bytes = 12 * 1024**3
    activation_offload_ok = not activation_offload_requested or (
        activation_offload_ranks == expected_visual_freeze_ranks
        and config.get("learner_activation_offload_backend")
        == "torch.autograd.graph.saved_tensors_hooks.selective_cpu_v1"
        and config.get("learner_activation_offload_budget_gib_per_rank") == 12
        and config.get("learner_activation_offload_min_tensor_mib") == 16
        and config.get("learner_activation_offload_pin_memory") is False
        and config.get("learner_activation_offload_exact_autograd") is True
        and len(activation_offload_rows) == len(expected_visual_freeze_ranks)
        and all(
            row["budget_bytes"] == activation_budget_bytes
            and 0 < row["offloaded_bytes"] <= activation_budget_bytes
            and row["offloaded_tensors"] > 0
            and row["seen_cuda_bytes"] >= row["offloaded_bytes"]
            for row in activation_offload_rows
        )
    )
    activation_offload_evidence = {
        "requested": activation_offload_requested,
        "confirmed_ranks": activation_offload_ranks,
        "expected_ranks": expected_visual_freeze_ranks,
        "backend": config.get("learner_activation_offload_backend"),
        "budget_gib_per_rank": config.get(
            "learner_activation_offload_budget_gib_per_rank"
        ),
        "min_tensor_mib": config.get("learner_activation_offload_min_tensor_mib"),
        "runtime_rows": activation_offload_rows,
        "pin_memory": config.get("learner_activation_offload_pin_memory"),
        "exact_autograd": config.get("learner_activation_offload_exact_autograd"),
    }
    zero3_cpu_offload_ok = all(
        (
            config.get("deepspeed") == "zero3_cpu_offload",
            config.get("optimizer") == "adamw_torch",
            config.get("cpu_optimizer_implementation") == "pytorch_adamw",
            "'offload_optimizer': {'device': 'cpu'" in lower,
            "'offload_param': {'device': 'cpu'" in lower,
            "deepspeedzerooffload initialize [end]" in lower,
            "deepspeedcpuadam" not in lower,
        )
    )
    vllm_memory_contract_ok = all(
        (
            math.isclose(
                float(config.get("vllm_gpu_memory_utilization", -1.0)),
                0.22,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "--vllm_gpu_memory_utilization 0.22" in launch_command,
            int(config.get("vllm_sleep_level", -1)) in {0, 1},
            f'--sleep_level {int(config.get("vllm_sleep_level", -1))}' in launch_command,
        )
    )
    wandb_urls = re.findall(r"https://wandb\.ai/[^\s]+/runs/[A-Za-z0-9_-]+", train_text)
    wandb_ids = re.findall(r"(?:setting up run\s+|/runs/)([a-z0-9]{8})(?:\s|$)", train_text, flags=re.IGNORECASE)
    runtime_rows = [row for row in all_logging if isinstance(row.get("train_runtime"), (int, float))]
    train_runtime = float(runtime_rows[-1]["train_runtime"]) if runtime_rows else None

    failures: list[str] = []
    checks = {
        "trainer_exit_zero": args.trainer_exit_code == 0,
        "completed_steps": completed_steps,
        "coverage": coverage_valid,
        "schema": schema_ok,
        "credit": credit_ok,
        "selected_policy_tokens": selected_ok,
        "finite": finite_ok,
        "finite_step_metrics": finite_step_metrics,
        "backward_chunks": backward_ok,
        "algorithm_metrics": dapo_metrics_ok,
        "dapo_selected_batch": dapo_selected_batch_ok,
        "dapo_shape_padding": dapo_shape_contract_ok,
        "dapo_padding_credit": padding_credit_ok,
        "dapo_reward_population": reward_population_ok,
        "dapo_low_effective_skip_config": low_effective_skip_config_ok,
        "dapo_low_effective_skip": low_effective_skip_contract_ok,
        "token_normalization": token_normalization_ok,
        "checkpoint_full_state": checkpoint_ok,
        "no_oom": not explicit_oom,
        "qwen35_fast_path": qwen35_fast_path_ok,
        "full_visual_no_grad_runtime": full_visual_no_grad_ok,
        "selective_gradient_checkpointing_runtime": selective_gc_ok,
        "frozen_vision_gc_configuration_runtime": frozen_vision_gc_ok,
        "learner_activation_offload_runtime": activation_offload_ok,
        "zero3_cpu_offload": zero3_cpu_offload_ok,
        "vllm_memory_contract": vllm_memory_contract_ok,
        "local_margin_reward_contract": local_margin_contract_ok,
        "editor_judge_service_contract": editor_judge_service_contract_ok,
        "editor_judge_group_statistics": editor_judge_group_stats_ok,
        "component_kl_contract": component_kl_contract_ok,
        "component_kl_metrics": component_kl_metrics_ok,
        "global_completion_kl_contract": global_completion_kl_contract_ok,
        "global_completion_kl_metrics": global_completion_kl_metrics_ok,
    }
    failures.extend(name for name, passed in checks.items() if not passed)

    trajectory_summary = {
        "rollout_mode": "actor_only",
        "algorithm": algorithm,
        "num_rows": len(trajectories),
        "num_rank_shards": len(trajectory_paths),
        "unique_trajectory_ids": len(set(ids)) if all(ids) else 0,
        "credit_integrity_rate": sum(
            credit_valid(
                row,
                algorithm,
                editor_judge_component_grpo=editor_judge_component_grpo,
            )
            for row in trajectories
        ) / len(trajectories) if trajectories else 0.0,
        "non_finite_count": 0 if all(finite(row) for row in trajectories) else 1,
        "selected_rows": len(selected),
        "minimum_selected_rows": len(update_calls) * min_effective_rows,
        "maximum_selected_rows": expected_physical,
        "physical_selected_rows": len(physical),
        "expected_physical_selected_rows": expected_physical,
        "padding_rows": len(padding),
        "low_effective_skipped_calls": sorted(skipped_calls),
        "num_low_effective_skipped_calls": len(skipped_calls),
        "num_optimizer_updates": expected_optimizer_updates,
        "reward_population_valid": reward_population_ok,
        "rank_rows": {name: len(rows) for name, rows in rank_rows.items()},
        "rank_selected_rows": rank_selected_counts,
        "rank_physical_selected_rows": rank_physical_counts,
        "learner_rank_selected_rows": dict(sorted(learner_rank_active_counts.items())),
        "learner_rank_physical_rows": dict(sorted(learner_rank_physical_counts.items())),
        "margin_reward_scope": margin_scope,
        "local_margin_cohort_count": len(local_margin_cohorts),
        "local_margin_round_count": len(local_margin_rounds),
        "local_margin_metadata_valid": local_margin_metadata_ok,
        "local_margin_cohort_integrity": local_margin_cohort_integrity,
        "local_margin_round_integrity": local_margin_round_integrity,
        "local_margin_config_valid": local_margin_config_ok,
        "editor_judge_success_rows": len(editor_judge_success_rows),
        "editor_judge_ineligible_rows": len(editor_judge_ineligible_rows),
        "editor_judge_service_contract_valid": editor_judge_service_contract_ok,
        "editor_judge_group_statistics_valid": editor_judge_group_stats_ok,
        "component_kl_mode": component_kl_mode,
        "component_kl_contract_valid": component_kl_contract_ok,
        "component_kl_metrics_valid": component_kl_metrics_ok,
        "global_completion_kl_enabled": global_completion_kl_enabled,
        "global_completion_kl_contract_valid": global_completion_kl_contract_ok,
        "global_completion_kl_metrics_valid": global_completion_kl_metrics_ok,
    }
    (run_dir / "artifacts" / "trajectory_summary.json").write_text(
        json.dumps(trajectory_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        **config,
        "passed": not failures,
        "failures": failures,
        "trainer_exit_code": args.trainer_exit_code,
        "expected_start_step": args.expected_start_step,
        "expected_end_step": args.expected_end_step,
        "completed_data_steps_this_stage": len(steps),
        "completed_optimizer_updates_this_stage": expected_optimizer_updates,
        "completed_optimizer_steps_this_stage": expected_optimizer_updates,
        "low_effective_skipped_calls": sorted(skipped_calls),
        "low_effective_skipped_global_steps": sorted(skipped_step_numbers),
        "expected_rollout_calls_this_stage": expected_calls,
        "generated_trajectory_rows": len(trajectories),
        "selected_trajectory_rows": len(selected),
        "minimum_selected_trajectory_rows": len(update_calls) * min_effective_rows,
        "maximum_selected_trajectory_rows": expected_physical,
        "physical_selected_trajectory_rows": len(physical),
        "expected_physical_selected_trajectory_rows": expected_physical,
        "padding_trajectory_rows": len(padding),
        "generated_rows_by_rollout_call": dict(sorted(generated_by_call.items())),
        "selected_rows_by_rollout_call": dict(sorted(selected_by_call.items())),
        "physical_rows_by_rollout_call": dict(sorted(physical_by_call.items())),
        "padding_rows_by_rollout_call": dict(sorted(padding_by_call.items())),
        "rank_trajectory_rows": {name: len(rows) for name, rows in rank_rows.items()},
        "rank_selected_rows": rank_selected_counts,
        "rank_physical_selected_rows": rank_physical_counts,
        "learner_rank_selected_rows": dict(sorted(learner_rank_active_counts.items())),
        "learner_rank_physical_rows": dict(sorted(learner_rank_physical_counts.items())),
        "origin_rank_coverage_valid": origin_rank_coverage_ok,
        "trajectory_ids_unique": len(set(ids)) == len(ids) and all(ids),
        "group_integrity": group_integrity,
        "partial_calls_retain_all_effective": partial_calls_retain_all_effective,
        "padding_credit_valid": padding_credit_ok,
        "reward_population_valid": reward_population_ok,
        "checks": checks,
        "explicit_oom": explicit_oom,
        "fast_path_fallback_detected": fast_path_fallback,
        "qwen35_fast_path_evidence": qwen35_fast_path_evidence,
        "full_visual_no_grad_evidence": full_visual_no_grad_evidence,
        "selective_gradient_checkpointing_evidence": selective_gc_evidence,
        "frozen_vision_gc_evidence": frozen_vision_gc_evidence,
        "learner_activation_offload_evidence": activation_offload_evidence,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_evidence": checkpoint_evidence,
        "wall_seconds_including_initialization": args.wall_seconds,
        "train_runtime_seconds": train_runtime,
        "selected_trajectories_per_second": len(selected) / train_runtime if train_runtime else None,
        "optimizer_step_seconds": metric_stats(metric("step_time")),
        "backward_seconds": metric_stats(metric("vf/backward_seconds")),
        "reasoning_component_kl_loss": metric_stats(reasoning_kl_metrics),
        "rating_component_kl_loss": metric_stats(rating_kl_metrics),
        "global_completion_kl_mean": metric_stats(global_kl_mean_metrics),
        "global_completion_kl_loss": metric_stats(global_kl_loss_metrics),
        "dapo_generation_rounds": metric_stats(metric("vf/dapo_generation_rounds")),
        "dapo_acceptance_rate": metric_stats(metric("vf/dapo_acceptance_rate")),
        "max_backward_peak_allocated_gib": max(metric("vf/backward_peak_allocated_gib"), default=None),
        "max_backward_peak_reserved_gib": max(metric("vf/backward_peak_reserved_gib"), default=None),
        "logging_jsonl": str(log_path) if log_path else None,
        "wandb_url": wandb_urls[-1].rstrip(".,") if wandb_urls else None,
        "wandb_run_id": wandb_ids[-1] if wandb_ids else None,
        "telemetry": telemetry_summary(run_dir / "logs" / "gpu_telemetry.csv"),
    }
    output = run_dir / "artifacts" / "run_result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
