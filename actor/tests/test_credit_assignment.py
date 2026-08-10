from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

from credit_assignment import build_credit_assignment, build_token_credit_assignment


class CreditAssignmentTests(unittest.TestCase):
    def test_dapo_policy_credit_targets_all_a0_tokens(self) -> None:
        assignment = build_token_credit_assignment(
            trajectory_id="dapo-row",
            rewards={"dapo_policy": 1.25},
            eligibility={"dapo_policy": True},
            advantages={"dapo_policy": 0.75},
            weights={"dapo_policy": 1.0},
            failure_owner="none",
        )
        component = assignment["components"]["dapo_policy"]
        self.assertEqual(component["targets"], ["a0.all"])
        self.assertEqual(component["weighted_advantage"], 0.75)

    def test_scalar_grpo_policy_credit_targets_all_a0_tokens(self) -> None:
        assignment = build_token_credit_assignment(
            trajectory_id="grpo-row",
            rewards={"grpo_policy": 1.5},
            eligibility={"grpo_policy": True},
            advantages={"grpo_policy": -0.25},
            weights={"grpo_policy": 1.0},
            failure_owner="none",
        )
        component = assignment["components"]["grpo_policy"]
        self.assertEqual(component["targets"], ["a0.all"])
        self.assertEqual(component["weighted_advantage"], -0.25)

    def test_segment_credits_sum_to_total_reward(self) -> None:
        assignment = build_credit_assignment(
            trajectory_id="traj-1",
            training_phase="phase_a",
            segments={
                "S0_original_understanding": {"R_S0_format": 1.0, "R_S0_margin": 0.5},
                "S1_edit_decision_instruction": {"R_S1_edit_gate": 0.25},
                "S2_external_outcome": {"R_S2_judger_gain": 0.8},
            },
            weights={
                "R_S0_format": 1.0,
                "R_S0_margin": 2.0,
                "R_S1_edit_gate": 1.0,
                "R_S2_judger_gain": 0.5,
            },
            eligibility={
                "S0_original_understanding": True,
                "S1_edit_decision_instruction": True,
                "S2_external_outcome": True,
            },
            failure_owner="none",
        )
        self.assertAlmostEqual(assignment["total_reward"], 2.65)
        credited = sum(segment["credited_reward"] for segment in assignment["segments"].values())
        self.assertAlmostEqual(credited, assignment["total_reward"])

    def test_ineligible_segment_receives_zero_credit(self) -> None:
        assignment = build_credit_assignment(
            trajectory_id="traj-2",
            training_phase="phase_a",
            segments={"S2_external_outcome": {"R_S2_judger_gain": 1.0}},
            weights={"R_S2_judger_gain": 1.0},
            eligibility={"S2_external_outcome": False},
            failure_owner="service",
        )
        segment = assignment["segments"]["S2_external_outcome"]
        self.assertEqual(segment["credited_reward"], 0.0)
        self.assertEqual(assignment["failure_owner"], "service")

    def test_token_credit_records_reward_advantage_weight_and_target_fields(self) -> None:
        assignment = build_token_credit_assignment(
            trajectory_id="traj-token-1",
            rewards={"rating0": 0.8, "edit_gain": 0.5, "delta_margin": 0.7, "rating1_anchor": 0.0},
            eligibility={
                "rating0": True,
                "edit_gain": False,
                "delta_margin": True,
                "rating1_anchor": True,
            },
            advantages={
                "rating0": 1.2,
                "edit_gain": 0.0,
                "delta_margin": -0.4,
                "rating1_anchor": -0.7,
            },
            weights={"rating0": 1.0, "edit_gain": 2.0, "delta_margin": 0.5, "rating1_anchor": 1.0},
            failure_owner="service",
        )
        self.assertEqual(assignment["schema_version"], "vf_token_credit_v3")
        self.assertEqual(assignment["components"]["rating0"]["targets"], ["a0.rating_content"])
        self.assertEqual(assignment["components"]["edit_gain"]["targets"], ["a0.editing_content"])
        self.assertEqual(
            assignment["components"]["delta_margin"]["targets"],
            ["a1.rating_content"],
        )
        self.assertEqual(
            assignment["components"]["rating1_anchor"]["targets"],
            ["a1.rating_content"],
        )
        self.assertFalse(assignment["components"]["edit_gain"]["eligible"])
        self.assertEqual(assignment["failure_owner"], "service")


if __name__ == "__main__":
    unittest.main()
