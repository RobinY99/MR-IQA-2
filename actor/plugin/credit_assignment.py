from __future__ import annotations

import math
from typing import Any, Mapping


FAILURE_OWNERS = {"actor", "service", "data", "none"}
TOKEN_COMPONENT_TARGETS = {
    "dapo_policy": ["a0.all"],
    "grpo_policy": ["a0.all"],
    "format_a0": ["a0.all"],
    "format_a1": ["a1.all"],
    "rating0": ["a0.rating_content"],
    "reasoning": ["a0.reasoning.evidence_content", "a0.reasoning.solution_content"],
    "soft_overlong": ["a0.completion_non_padding"],
    "edit_gain": ["a0.editing_content"],
    "edit_gate": ["a0.editing_decision"],
    "delta_margin": ["a1.rating_content"],
    "rating1_anchor": ["a1.rating_content"],
}


def build_credit_assignment(
    *,
    trajectory_id: str,
    training_phase: str,
    segments: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
    eligibility: Mapping[str, bool],
    failure_owner: str,
) -> dict[str, Any]:
    if failure_owner not in FAILURE_OWNERS:
        raise ValueError(f"unsupported failure owner: {failure_owner}")
    segment_payload: dict[str, Any] = {}
    total = 0.0
    for segment_name, components in segments.items():
        eligible = bool(eligibility.get(segment_name, False))
        raw_components = {name: float(value) for name, value in components.items()}
        weighted_components = {
            name: (raw * float(weights.get(name, 1.0)) if eligible else 0.0)
            for name, raw in raw_components.items()
        }
        if not all(math.isfinite(value) for value in weighted_components.values()):
            raise ValueError(f"non-finite credited reward in {segment_name}")
        credited = sum(weighted_components.values())
        total += credited
        segment_payload[segment_name] = {
            "eligible": eligible,
            "raw_components": raw_components,
            "weights": {name: float(weights.get(name, 1.0)) for name in raw_components},
            "weighted_components": weighted_components,
            "credited_reward": credited,
        }
    return {
        "schema_version": "vf_credit_assignment_v1",
        "trajectory_id": str(trajectory_id),
        "training_phase": str(training_phase),
        "segments": segment_payload,
        "total_reward": total,
        "failure_owner": failure_owner,
    }


def build_token_credit_assignment(
    *,
    trajectory_id: str,
    rewards: Mapping[str, float],
    eligibility: Mapping[str, bool],
    advantages: Mapping[str, float],
    weights: Mapping[str, float],
    failure_owner: str,
) -> dict[str, Any]:
    if failure_owner not in FAILURE_OWNERS:
        raise ValueError(f"unsupported failure owner: {failure_owner}")
    components: dict[str, Any] = {}
    names = sorted(set(rewards) | set(eligibility) | set(advantages) | set(weights))
    for name in names:
        if name not in TOKEN_COMPONENT_TARGETS:
            raise ValueError(f"unsupported token credit component: {name}")
        reward = float(rewards.get(name, 0.0))
        advantage = float(advantages.get(name, 0.0))
        weight = float(weights.get(name, 1.0))
        if not all(math.isfinite(value) for value in (reward, advantage, weight)):
            raise ValueError(f"non-finite token credit component: {name}")
        components[name] = {
            "raw_reward": reward,
            "eligible": bool(eligibility.get(name, False)),
            "group_advantage": advantage,
            "weight": weight,
            "weighted_advantage": advantage * weight,
            "targets": list(TOKEN_COMPONENT_TARGETS[name]),
        }
    return {
        "schema_version": "vf_token_credit_v3",
        "trajectory_id": str(trajectory_id),
        "components": components,
        "failure_owner": failure_owner,
    }
