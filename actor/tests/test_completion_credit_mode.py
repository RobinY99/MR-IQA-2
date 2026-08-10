from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

from component_loss import compute_component_policy_loss  # noqa: E402
from token_credit import (  # noqa: E402
    build_rollout_component_credit,
    compose_rollout_token_advantages,
)


class CompletionCreditModeTests(unittest.TestCase):
    def test_soft_overlong_uses_all_completion_tokens_in_field_mode(self) -> None:
        masks = {
            "format": [True, True, True, True],
            "rating_content": [False, False, True, False],
            "reasoning_content": [True, True, False, False],
            "editing_decision": [False] * 4,
            "editing_content": [False] * 4,
        }
        with patch.dict(
            os.environ,
            {"VF_COMPONENT_CREDIT_MASK_MODE": "field"},
            clear=False,
        ):
            credit = build_rollout_component_credit(
                "a0",
                masks,
                {"soft_overlong": -1.0},
                {"soft_overlong": 1.0},
                {"soft_overlong": True},
            )
        self.assertEqual(credit["soft_overlong"]["mask"], [True] * 4)
        self.assertEqual(credit["soft_overlong"]["mask_mode"], "field")

    def test_every_eligible_component_uses_the_full_completion(self) -> None:
        masks = {
            "format": [True, True, True, True],
            "rating_content": [False, False, True, False],
            "reasoning_content": [True, True, False, False],
            "editing_decision": [False] * 4,
            "editing_content": [False] * 4,
        }
        advantages = {
            "format_a0": 0.5,
            "rating0": -0.2,
            "reasoning": 1.0,
        }
        eligibility = {
            "format_a0": True,
            "rating0": True,
            "reasoning": True,
        }
        with patch.dict(
            os.environ,
            {"VF_COMPONENT_CREDIT_MASK_MODE": "completion"},
            clear=False,
        ):
            credit = build_rollout_component_credit(
                "a0",
                masks,
                advantages,
                {"format_a0": 1.0, "rating0": 1.0, "reasoning": 1.0},
                eligibility,
                format_mask_override=[False, True, False, False],
            )
        for component in ("format_a0", "rating0", "reasoning"):
            self.assertEqual(credit[component]["mask"], [True] * 4)
            self.assertEqual(credit[component]["mask_mode"], "completion")

    def test_ineligible_component_remains_removed(self) -> None:
        masks = {
            "format": [True, True],
            "rating_content": [False, True],
            "reasoning_content": [True, False],
            "editing_decision": [False, False],
            "editing_content": [False, False],
        }
        with patch.dict(
            os.environ,
            {"VF_COMPONENT_CREDIT_MASK_MODE": "completion"},
            clear=False,
        ):
            credit = build_rollout_component_credit(
                "a0",
                masks,
                {"format_a0": 1.0, "rating0": 4.0, "reasoning": -1.0},
                {"format_a0": 1.0, "rating0": 1.0, "reasoning": 1.0},
                {"format_a0": True, "rating0": False, "reasoning": True},
            )
        self.assertEqual(credit["format_a0"]["mask"], [True, True])
        self.assertEqual(credit["rating0"]["mask"], [False, False])
        self.assertEqual(credit["reasoning"]["mask"], [True, True])

    def test_component_sum_is_uniform_across_completion_tokens(self) -> None:
        masks = {
            "format": [True, True, True],
            "rating_content": [False, True, False],
            "reasoning_content": [True, False, False],
            "editing_decision": [False] * 3,
            "editing_content": [False] * 3,
        }
        with patch.dict(
            os.environ,
            {"VF_COMPONENT_CREDIT_MASK_MODE": "completion"},
            clear=False,
        ):
            token_advantages = compose_rollout_token_advantages(
                "a0",
                masks,
                {"format_a0": 0.5, "rating0": -0.2, "reasoning": 1.0},
                {"format_a0": 1.0, "rating0": 1.0, "reasoning": 1.0},
            )
        self.assertEqual(token_advantages, [1.3, 1.3, 1.3])

    def test_padding_positions_have_zero_gradient(self) -> None:
        ratio = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        total, _, active = compute_component_policy_loss(
            ratio=ratio,
            clipped_ratio=ratio,
            rollout_is_weights=torch.ones_like(ratio),
            completion_mask=torch.tensor([[True, True, False, False]]),
            component_masks={
                "format_a0": torch.tensor([[True, True, True, True]]),
                "rating0": torch.tensor([[True, True, True, True]]),
                "reasoning": torch.tensor([[True, True, True, True]]),
            },
            component_advantages={
                "format_a0": torch.tensor([0.5], dtype=torch.float64),
                "rating0": torch.tensor([-0.2], dtype=torch.float64),
                "reasoning": torch.tensor([1.0], dtype=torch.float64),
            },
            component_weights={
                "format_a0": 1.0,
                "rating0": 1.0,
                "reasoning": 1.0,
            },
            normalization="sequence_mean",
        )
        total.backward()
        self.assertEqual(active, 1)
        assert ratio.grad is not None
        self.assertNotEqual(float(ratio.grad[0, 0]), 0.0)
        self.assertNotEqual(float(ratio.grad[0, 1]), 0.0)
        self.assertEqual(float(ratio.grad[0, 2]), 0.0)
        self.assertEqual(float(ratio.grad[0, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
