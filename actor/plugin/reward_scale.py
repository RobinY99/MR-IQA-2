from __future__ import annotations

import math


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def l2_laplace_reward(error: float, tau: float = 1.0) -> float:
    error = _finite(error, "error")
    tau = _finite(tau, "tau")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    return math.exp(-0.5 * error * error / tau)


def signed_l2_improvement_reward(delta: float, tau_s: float = 1.0) -> float:
    delta = _finite(delta, "delta")
    tau_s = _finite(tau_s, "tau_s")
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive")
    if delta == 0.0:
        return 0.0
    magnitude = -math.expm1(-0.5 * delta * delta / tau_s)
    return math.copysign(magnitude, delta)


def rating_anchor_counterfactual_penalty(
    prediction: float,
    target: float,
    *,
    tau: float = 1.0,
) -> float:
    prediction = _finite(prediction, "prediction")
    target = _finite(target, "target")
    if not 1.0 <= target <= 5.0:
        raise ValueError("target must be in [1, 5]")
    actual = l2_laplace_reward(prediction - target, tau=tau)
    perfect = l2_laplace_reward(0.0, tau=tau)
    return min(0.0, actual - perfect)


def gaussian_edit_probability(
    rating: float,
    *,
    mean: float = 3.0,
    std: float = 1.0,
) -> float:
    rating = _finite(rating, "rating")
    mean = _finite(mean, "mean")
    std = _finite(std, "std")
    if not 1.0 <= rating <= 5.0:
        raise ValueError("rating must be in [1, 5]")
    if std <= 0.0:
        raise ValueError("std must be positive")

    def survival(value: float) -> float:
        z = (value - mean) / (std * math.sqrt(2.0))
        return 0.5 * math.erfc(z)

    low = survival(1.0)
    high = survival(5.0)
    denominator = low - high
    if denominator <= 0.0:
        raise ValueError("invalid Gaussian edit-probability normalization")
    if rating == 1.0:
        return 1.0
    if rating == 5.0:
        return 0.0
    return min(1.0, max(0.0, (survival(rating) - high) / denominator))


def edit_gate_reward(
    target_rating: float,
    editing_nonempty: bool,
    *,
    tau: float = 1.0,
    gaussian_mean: float = 3.0,
    gaussian_std: float = 1.0,
) -> float:
    target = gaussian_edit_probability(
        target_rating,
        mean=gaussian_mean,
        std=gaussian_std,
    )
    decision = 1.0 if editing_nonempty else 0.0
    return l2_laplace_reward(decision - target, tau=tau)


def edit_gain_reward(target_rating: float, judger_delta: float, *, tau: float = 1.0) -> float:
    target_rating = _finite(target_rating, "target_rating")
    judger_delta = _finite(judger_delta, "judger_delta")
    if not 1.0 <= target_rating <= 5.0:
        raise ValueError("target_rating must be in [1, 5]")
    expected_gain = (5.0 - target_rating) / 4.0
    observed_gain = judger_delta / 4.0
    missed_gain = max(0.0, expected_gain - observed_gain)
    return l2_laplace_reward(missed_gain, tau=tau)


def delta_margin_reward(
    actor_rating0: float,
    actor_rating1: float,
    judger_rating0: float,
    judger_rating1: float,
    *,
    tau: float = 1.0,
) -> float:
    actor_delta = _finite(actor_rating1, "actor_rating1") - _finite(actor_rating0, "actor_rating0")
    judger_delta = _finite(judger_rating1, "judger_rating1") - _finite(judger_rating0, "judger_rating0")
    return l2_laplace_reward(actor_delta - judger_delta, tau=tau)
