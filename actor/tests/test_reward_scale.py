from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

import reward_scale as reward_scale_module

from reward_scale import (  # noqa: E402
    delta_margin_reward,
    edit_gain_reward,
    edit_gate_reward,
    gaussian_edit_probability,
    l2_laplace_reward,
)


class RewardScaleTests(unittest.TestCase):
    def test_gaussian_edit_probability_is_endpoint_normalized_and_monotonic(self) -> None:
        values = [gaussian_edit_probability(float(rating)) for rating in range(1, 6)]
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[-1], 0.0)
        self.assertTrue(all(left > right for left, right in zip(values, values[1:])))
        self.assertAlmostEqual(values[1], 0.8576, places=3)
        self.assertAlmostEqual(values[2], 0.5, places=6)
        self.assertAlmostEqual(values[3], 0.1424, places=3)

    def test_l2_laplace_reward_stays_in_zero_one_range(self) -> None:
        self.assertEqual(l2_laplace_reward(0.0), 1.0)
        self.assertAlmostEqual(l2_laplace_reward(1.0), math.exp(-0.5))
        self.assertGreater(l2_laplace_reward(4.0), 0.0)
        self.assertLessEqual(l2_laplace_reward(4.0), 1.0)

    def test_edit_gate_prefers_edit_for_low_rating_and_empty_for_high_rating(self) -> None:
        self.assertGreater(edit_gate_reward(1.0, True), edit_gate_reward(1.0, False))
        self.assertGreater(edit_gate_reward(5.0, False), edit_gate_reward(5.0, True))
        self.assertAlmostEqual(edit_gate_reward(3.0, False), edit_gate_reward(3.0, True))

    def test_rating_anchor_fallback_is_zero_only_at_the_absolute_target(self) -> None:
        function = getattr(reward_scale_module, "rating_anchor_counterfactual_penalty", None)
        self.assertIsNotNone(function)
        self.assertEqual(function(3.0, 3.0), 0.0)
        self.assertLess(function(4.0, 3.0), 0.0)
        self.assertLess(function(8.0, 3.0), function(4.0, 3.0))

    def test_edit_gain_compares_normalized_expected_and_observed_gain(self) -> None:
        self.assertEqual(edit_gain_reward(1.0, 4.0), 1.0)
        self.assertAlmostEqual(edit_gain_reward(1.0, 0.0), math.exp(-0.5))
        self.assertAlmostEqual(edit_gain_reward(1.0, -4.0), math.exp(-2.0))

    def test_delta_margin_uses_actor_judger_change_error(self) -> None:
        self.assertEqual(delta_margin_reward(2.0, 3.0, 2.5, 3.5), 1.0)
        self.assertAlmostEqual(delta_margin_reward(2.0, 4.0, 2.5, 3.5), math.exp(-0.5))


if __name__ == "__main__":
    unittest.main()
