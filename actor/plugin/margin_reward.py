from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Sequence

from actor_contract import actor_rating_number, parse_actor_json
from reward_scale import l2_laplace_reward


def _value_at(values: Sequence[Any] | None, index: int, default: Any = None) -> Any:
    if values is None or index >= len(values):
        return default
    return values[index]


def _rating(text: str) -> float | None:
    payload = parse_actor_json(text)
    if payload is None:
        return None
    return actor_rating_number(payload.get("rating"))


def compute_local_margin_rewards(
    completions: Sequence[str],
    target_mean: Sequence[Any],
    target_std: Sequence[Any] | None = None,
    sample_ids: Sequence[Any] | None = None,
    comparison_cohort_ids: Sequence[Any] | None = None,
    expected_groups_per_cohort: int | None = None,
) -> tuple[list[float], dict[str, Any]]:
    count = len(completions)
    if comparison_cohort_ids is not None and len(comparison_cohort_ids) != count:
        raise ValueError(
            "comparison_cohort_ids must have one value per completion: "
            f"cohorts={len(comparison_cohort_ids)}, completions={count}"
        )
    if expected_groups_per_cohort is not None:
        if expected_groups_per_cohort <= 0:
            raise ValueError("expected_groups_per_cohort must be positive")
        if comparison_cohort_ids is None:
            raise ValueError(
                "expected_groups_per_cohort requires explicit comparison_cohort_ids"
            )
    predictions = [_rating(str(text or "")) for text in completions]
    rewards = [0.0] * count
    groups: OrderedDict[Any, list[int]] = OrderedDict()
    group_cohorts: dict[Any, Any] = {}
    for index in range(count):
        key = _value_at(sample_ids, index)
        if key is None:
            key = (_value_at(target_mean, index), _value_at(target_std, index, 1.0))
        groups.setdefault(key, []).append(index)
        if comparison_cohort_ids is not None:
            cohort_id = comparison_cohort_ids[index]
            if cohort_id is None:
                raise ValueError(f"missing comparison cohort for completion {index}")
            previous = group_cohorts.setdefault(key, cohort_id)
            if previous != cohort_id:
                raise ValueError(
                    f"sample group {key!r} spans multiple comparison cohorts: "
                    f"{previous!r} and {cohort_id!r}"
                )

    group_stats: list[dict[str, Any]] = []
    cohorts: OrderedDict[Any, list[dict[str, Any]]] = OrderedDict()
    global_cohort = "__all_groups__"
    for key, indices in groups.items():
        values = [predictions[index] for index in indices if predictions[index] is not None]
        valid_values = [float(value) for value in values if value is not None]
        predicted_mean = sum(valid_values) / len(valid_values) if valid_values else None
        label = _value_at(target_mean, indices[0]) if indices else None
        try:
            label_number = float(label)
        except (TypeError, ValueError):
            label_number = None
        if label_number is not None and not math.isfinite(label_number):
            label_number = None
        cohort_id = group_cohorts[key] if comparison_cohort_ids is not None else global_cohort
        item = {
            "key": key,
            "indices": indices,
            "predicted_mean": predicted_mean,
            "valid_values": valid_values,
            "label": label_number,
            "cohort_id": cohort_id,
        }
        group_stats.append(item)
        cohorts.setdefault(cohort_id, []).append(item)

    cohort_group_sizes = [len(items) for items in cohorts.values()]
    if expected_groups_per_cohort is not None:
        invalid = [
            (cohort_id, len(items))
            for cohort_id, items in cohorts.items()
            if len(items) != expected_groups_per_cohort
        ]
        if invalid:
            raise ValueError(
                "comparison cohort group-count mismatch: "
                f"expected={expected_groups_per_cohort}, actual={invalid}"
            )

    eligible_cohorts = {
        cohort_id
        for cohort_id, items in cohorts.items()
        if sum(
            item["predicted_mean"] is not None and item["label"] is not None
            for item in items
        )
        >= 2
    }
    eligible = bool(eligible_cohorts)
    pair_count = 0
    if eligible:
        for item in group_stats:
            indices = item["indices"]
            label = item["label"]
            if label is None:
                continue
            for index in indices:
                prediction = predictions[index]
                if prediction is None:
                    continue
                pair_rewards: list[float] = []
                for other in cohorts[item["cohort_id"]]:
                    other_mean = other["predicted_mean"]
                    other_label = other["label"]
                    if other is item or other_mean is None or other_label is None:
                        continue
                    error = (float(prediction) - other_mean) - (label - other_label)
                    pair_rewards.append(l2_laplace_reward(error))
                if pair_rewards:
                    rewards[index] = sum(pair_rewards) / len(pair_rewards)
                    pair_count += len(pair_rewards)

    stats = {
        "eligible": eligible,
        "num_completions": count,
        "num_groups": len(groups),
        "num_valid_ratings": sum(value is not None for value in predictions),
        "pair_count": pair_count,
        "comparison_scope": (
            "explicit_cohort" if comparison_cohort_ids is not None else "all_groups"
        ),
        "num_cohorts": len(cohorts),
        "num_eligible_cohorts": len(eligible_cohorts),
        "cohort_group_sizes": cohort_group_sizes,
        "expected_groups_per_cohort": expected_groups_per_cohort,
    }
    return rewards, stats
