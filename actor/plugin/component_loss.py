from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


def compute_component_policy_loss(
    *,
    ratio: torch.Tensor,
    clipped_ratio: torch.Tensor,
    rollout_is_weights: torch.Tensor,
    completion_mask: torch.Tensor,
    component_masks: Mapping[str, torch.Tensor],
    component_advantages: Mapping[str, torch.Tensor],
    component_weights: Mapping[str, float],
    component_denominators: Mapping[str, float | int] | None = None,
    loss_scale: float = 1.0,
    normalization: str = "sequence_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    if ratio.shape != clipped_ratio.shape or ratio.shape != completion_mask.shape:
        raise ValueError("ratio, clipped_ratio, and completion_mask shapes must match")
    if rollout_is_weights.shape != ratio.shape:
        raise ValueError("rollout importance weights must match ratio shape")
    if normalization not in {"sequence_mean", "token_mean"}:
        raise ValueError(f"unsupported normalization: {normalization}")
    finite_ratio = torch.where(torch.isfinite(ratio), ratio, torch.zeros_like(ratio))
    zero = finite_ratio.sum() * 0.0
    total = zero
    component_losses: dict[str, torch.Tensor] = {}
    active_union = torch.zeros(ratio.shape[0], dtype=torch.bool, device=ratio.device)

    for name, mask in component_masks.items():
        if name not in component_advantages:
            raise ValueError(f"missing component advantage: {name}")
        if mask.shape != ratio.shape:
            raise ValueError(f"component mask shape mismatch: {name}")
        advantages = component_advantages[name].to(device=ratio.device, dtype=ratio.dtype)
        if advantages.shape != (ratio.shape[0],):
            raise ValueError(f"component advantage shape mismatch: {name}")

        effective_mask = mask.bool() & completion_mask.bool()
        active_rows = effective_mask.sum(-1) > 0
        active_union |= active_rows
        if not bool(active_rows.any()):
            component_losses[name] = zero
            continue

        safe_ratio = torch.where(effective_mask, ratio, torch.ones_like(ratio))
        safe_clipped_ratio = torch.where(effective_mask, clipped_ratio, torch.ones_like(clipped_ratio))
        safe_rollout_is_weights = torch.where(
            effective_mask,
            rollout_is_weights,
            torch.ones_like(rollout_is_weights),
        )
        token_advantages = advantages.unsqueeze(-1)
        token_loss = -torch.minimum(
            safe_ratio * token_advantages,
            safe_clipped_ratio * token_advantages,
        )
        token_loss = token_loss * safe_rollout_is_weights
        masked_token_loss = torch.where(effective_mask, token_loss, torch.zeros_like(token_loss))
        denominator = float(
            active_rows.sum().item()
            if normalization == "sequence_mean"
            else effective_mask.sum().item()
        )
        if component_denominators is not None:
            if name not in component_denominators:
                raise ValueError(f"missing component denominator: {name}")
            denominator = float(component_denominators[name])
            if denominator <= 0:
                raise ValueError(f"component denominator must be positive: {name}={denominator}")
        if normalization == "sequence_mean":
            token_counts = effective_mask.sum(-1).clamp(min=1)
            numerator = (masked_token_loss.sum(-1) / token_counts)[active_rows].sum()
        else:
            numerator = masked_token_loss.sum()
        weighted_loss = (
            numerator / denominator
            * float(component_weights.get(name, 1.0))
            * float(loss_scale)
        )
        component_losses[name] = weighted_loss
        total = total + weighted_loss

    return total, component_losses, int(active_union.sum().item())


def compute_component_kl_loss(
    *,
    per_token_kl: torch.Tensor,
    completion_mask: torch.Tensor,
    component_masks: Mapping[str, torch.Tensor],
    component_betas: Mapping[str, float],
    component_denominators: Mapping[str, float | int] | None = None,
    loss_scale: float = 1.0,
    normalization: str = "sequence_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    if per_token_kl.shape != completion_mask.shape:
        raise ValueError("per_token_kl and completion_mask shapes must match")
    if normalization not in {"sequence_mean", "token_mean"}:
        raise ValueError(f"unsupported normalization: {normalization}")

    finite_kl = torch.where(
        torch.isfinite(per_token_kl),
        per_token_kl,
        torch.zeros_like(per_token_kl),
    )
    zero = finite_kl.sum() * 0.0
    total = zero
    component_losses: dict[str, torch.Tensor] = {}
    active_union = torch.zeros(
        per_token_kl.shape[0],
        dtype=torch.bool,
        device=per_token_kl.device,
    )

    for name, beta_value in component_betas.items():
        beta = float(beta_value)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError(f"component KL beta must be finite and non-negative: {name}={beta}")
        if name not in component_masks:
            raise ValueError(f"missing component KL mask: {name}")
        mask = component_masks[name]
        if mask.shape != per_token_kl.shape:
            raise ValueError(f"component KL mask shape mismatch: {name}")

        effective_mask = mask.bool() & completion_mask.bool()
        active_rows = effective_mask.sum(-1) > 0
        active_union |= active_rows
        if beta == 0.0 or not bool(active_rows.any()):
            component_losses[name] = zero
            continue
        if not bool(torch.isfinite(per_token_kl[effective_mask]).all()):
            raise ValueError(f"non-finite active component KL values: {name}")

        denominator = float(
            active_rows.sum().item()
            if normalization == "sequence_mean"
            else effective_mask.sum().item()
        )
        if component_denominators is not None:
            if name not in component_denominators:
                raise ValueError(f"missing component KL denominator: {name}")
            denominator = float(component_denominators[name])
            if denominator <= 0:
                raise ValueError(
                    f"component KL denominator must be positive: {name}={denominator}"
                )

        masked_kl = torch.where(
            effective_mask,
            per_token_kl,
            torch.zeros_like(per_token_kl),
        )
        if normalization == "sequence_mean":
            token_counts = effective_mask.sum(-1).clamp(min=1)
            numerator = (masked_kl.sum(-1) / token_counts)[active_rows].sum()
        else:
            numerator = masked_kl.sum()
        weighted_loss = numerator / denominator * beta * float(loss_scale)
        component_losses[name] = weighted_loss
        total = total + weighted_loss

    return total, component_losses, int(active_union.sum().item())


def combine_active_branch_losses(
    branches: Sequence[tuple[torch.Tensor, int]],
) -> torch.Tensor:
    active = [(loss, int(count)) for loss, count in branches if int(count) > 0]
    if not active:
        raise ValueError("no active branch losses")
    denominator = sum(count for _, count in active)
    return sum(loss * count for loss, count in active) / denominator
