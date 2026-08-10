from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from statistics import stdev
from typing import Any


@dataclass(frozen=True)
class DapoShapePaddedSelection:
    physical_indices: tuple[int, ...]
    active_indices: tuple[int, ...]
    padding_indices: tuple[int, ...]
    active_rows_per_rank: tuple[int, ...]
    padding_rows_per_rank: tuple[int, ...]


def soft_overlong_reward(
    token_length: int,
    *,
    max_length: int,
    cache_length: int,
    max_penalty: float = 1.0,
) -> float:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if cache_length <= 0 or cache_length >= max_length:
        raise ValueError("cache_length must be in (0, max_length)")
    if max_penalty < 0.0 or not math.isfinite(max_penalty):
        raise ValueError("max_penalty must be finite and non-negative")
    length = max(0, int(token_length))
    expected_length = max_length - cache_length
    exceed = max(0, length - expected_length)
    return -min(exceed / cache_length, 1.0) * max_penalty


def compute_dapo_group_advantages(
    rows: Sequence[Mapping[str, Any]],
    *,
    reward_key: str = "dapo_total_reward",
    group_key: str = "dapo_group_key",
    epsilon: float = 1e-6,
    expected_group_size: int | None = None,
) -> list[dict[str, float | int | bool]]:
    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")
    if expected_group_size is not None and expected_group_size <= 1:
        raise ValueError("expected_group_size must be greater than one")
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, row in enumerate(rows):
        key = str(row.get(group_key, ""))
        if not key:
            raise ValueError(f"missing {group_key}")
        groups.setdefault(key, []).append(index)

    result: list[dict[str, float | int | bool]] = [dict() for _ in rows]
    for key, indices in groups.items():
        values = []
        for index in indices:
            value = float(rows[index].get(reward_key, float("nan")))
            if not math.isfinite(value):
                raise ValueError(f"non-finite {reward_key} for group {key}")
            values.append(value)
        complete = expected_group_size is None or len(indices) == expected_group_size
        scale = stdev(values) if len(values) > 1 else 0.0
        effective = bool(complete and math.isfinite(scale) and scale > epsilon)
        mean = math.fsum(values) / len(values)
        for index, value in zip(indices, values):
            advantage = 0.0
            if effective:
                advantage = (value - mean) / (scale + epsilon)
            result[index] = {
                "advantage": advantage,
                "effective_group": effective,
                "group_std": scale,
                "group_size": len(indices),
            }
    return result


def select_effective_group_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_rows: int,
    group_size: int,
    group_key: str = "dapo_group_key",
) -> list[int]:
    if target_rows <= 0 or group_size <= 1 or target_rows % group_size:
        raise ValueError("target_rows must be positive and divisible by group_size")
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, row in enumerate(rows):
        key = str(row.get(group_key, ""))
        if not key:
            raise ValueError(f"missing {group_key}")
        groups.setdefault(key, []).append(index)

    selected: list[int] = []
    for key, indices in groups.items():
        if len(indices) != group_size:
            raise ValueError(f"incomplete DAPO group {key}: {len(indices)} != {group_size}")
        effective_values = {bool(rows[index].get("dapo_effective_group", False)) for index in indices}
        if len(effective_values) != 1:
            raise ValueError(f"inconsistent effective-group flag for {key}")
        if not effective_values.pop():
            continue
        selected.extend(indices)
        if len(selected) == target_rows:
            return selected
        if len(selected) > target_rows:
            raise RuntimeError("DAPO group selection crossed the target boundary")
    return selected


def select_effective_groups_with_shape_padding(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_rows: int,
    group_size: int,
    world_size: int,
    min_effective_rows: int,
    group_key: str = "dapo_group_key",
) -> DapoShapePaddedSelection:
    if target_rows <= 0 or group_size <= 1 or world_size <= 0:
        raise ValueError("target rows, group size, and world size must be positive")
    if target_rows % (group_size * world_size):
        raise ValueError(
            "target_rows must be divisible by complete groups on every rank"
        )
    if (
        min_effective_rows <= 0
        or min_effective_rows > target_rows
        or min_effective_rows % group_size
    ):
        raise ValueError(
            "min_effective_rows must be positive, no larger than target_rows, "
            "and divisible by group_size"
        )

    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, row in enumerate(rows):
        key = str(row.get(group_key, ""))
        if not key:
            raise ValueError(f"missing {group_key}")
        groups.setdefault(key, []).append(index)

    effective_groups: list[list[int]] = []
    padding_candidates: list[list[int]] = []
    for key, indices in groups.items():
        if len(indices) != group_size:
            raise ValueError(f"incomplete DAPO group {key}: {len(indices)} != {group_size}")
        flags = {bool(rows[index].get("dapo_effective_group", False)) for index in indices}
        if len(flags) != 1:
            raise ValueError(f"inconsistent effective-group flag for {key}")
        if flags.pop():
            effective_groups.append(indices)
        else:
            padding_candidates.append(indices)

    target_groups = target_rows // group_size
    if len(effective_groups) >= target_groups:
        active_groups = effective_groups[:target_groups]
        padding_groups: list[list[int]] = []
    else:
        active_groups = effective_groups
        active_rows = len(active_groups) * group_size
        if active_rows < min_effective_rows:
            raise RuntimeError(
                "DAPO effective trajectory floor was not met after dynamic sampling: "
                f"effective={active_rows}, minimum={min_effective_rows}, "
                f"target={target_rows}"
            )
        padding_group_count = target_groups - len(active_groups)
        if len(padding_candidates) < padding_group_count:
            raise RuntimeError(
                "DAPO cannot build complete-group shape padding: "
                f"needed={padding_group_count}, available={len(padding_candidates)}"
            )
        padding_groups = padding_candidates[:padding_group_count]

    groups_per_rank = target_groups // world_size
    rank_active: list[list[list[int]]] = [[] for _ in range(world_size)]
    for group_index, indices in enumerate(active_groups):
        rank_active[group_index % world_size].append(indices)
    if any(len(bucket) > groups_per_rank for bucket in rank_active):
        raise RuntimeError("DAPO active-group redistribution exceeded rank capacity")

    padding_iterator = iter(padding_groups)
    rank_physical: list[list[tuple[list[int], bool]]] = []
    for active_bucket in rank_active:
        bucket = [(indices, True) for indices in active_bucket]
        while len(bucket) < groups_per_rank:
            try:
                bucket.append((next(padding_iterator), False))
            except StopIteration as exc:
                raise RuntimeError("DAPO shape-padding groups were exhausted") from exc
        rank_physical.append(bucket)
    try:
        next(padding_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("DAPO shape-padding selection left unused padding groups")

    physical_indices: list[int] = []
    active_indices: list[int] = []
    padding_indices: list[int] = []
    active_rows_per_rank: list[int] = []
    padding_rows_per_rank: list[int] = []
    for bucket in rank_physical:
        rank_active_rows = 0
        rank_padding_rows = 0
        for indices, is_active in bucket:
            physical_indices.extend(indices)
            if is_active:
                active_indices.extend(indices)
                rank_active_rows += len(indices)
            else:
                padding_indices.extend(indices)
                rank_padding_rows += len(indices)
        active_rows_per_rank.append(rank_active_rows)
        padding_rows_per_rank.append(rank_padding_rows)

    if len(physical_indices) != target_rows:
        raise RuntimeError("DAPO shape-padded selection has the wrong physical size")
    if len(set(physical_indices)) != target_rows:
        raise RuntimeError("DAPO shape-padded selection contains duplicate rows")
    if max(active_rows_per_rank) - min(active_rows_per_rank) > group_size:
        raise RuntimeError("DAPO active groups are not balanced across learner ranks")
    return DapoShapePaddedSelection(
        physical_indices=tuple(physical_indices),
        active_indices=tuple(active_indices),
        padding_indices=tuple(padding_indices),
        active_rows_per_rank=tuple(active_rows_per_rank),
        padding_rows_per_rank=tuple(padding_rows_per_rank),
    )


def zero_weight_padding_credit(credit: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(credit))
    components = result.get("components")
    if not isinstance(components, dict) or not components:
        raise ValueError("padding credit requires token-credit components")
    for component in components.values():
        if not isinstance(component, dict):
            raise ValueError("padding credit component must be an object")
        component["eligible"] = False
        component["group_advantage"] = 0.0
        component["weight"] = 0.0
        component["weighted_advantage"] = 0.0
    result["shape_padding"] = {
        "enabled": True,
        "loss_weight": 0.0,
        "token_denominator_eligible": False,
    }
    dapo = result.get("dapo")
    if isinstance(dapo, dict):
        dapo["shape_padding"] = True
    return result
