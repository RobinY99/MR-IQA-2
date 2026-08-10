from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

from margin_reward import compute_local_margin_rewards


def completion(rating) -> str:
    return json.dumps({"reason": "Evidence", "rating": rating, "suggestion": ""})


class MarginRewardTests(unittest.TestCase):
    def test_pairwise_margin_preserves_shape_and_rewards_close_deltas(self) -> None:
        completions = [completion("2.0"), completion("2.5"), completion("4.0"), completion("4.5")]
        rewards, stats = compute_local_margin_rewards(
            completions,
            target_mean=[2.0, 2.0, 4.0, 4.0],
            target_std=[1.0, 1.0, 1.0, 1.0],
            sample_ids=["low", "low", "high", "high"],
        )
        self.assertEqual(len(rewards), len(completions))
        self.assertTrue(stats["eligible"])
        self.assertEqual(stats["num_groups"], 2)
        self.assertTrue(all(0.9 < reward <= 1.0 for reward in rewards))

    def test_malformed_rating_gets_zero_without_breaking_shape(self) -> None:
        completions = [completion("3.75.00"), completion("2.0"), completion("4.0"), completion("4.2")]
        rewards, _ = compute_local_margin_rewards(
            completions,
            target_mean=[2.0, 2.0, 4.0, 4.0],
            target_std=[1.0] * 4,
            sample_ids=["low", "low", "high", "high"],
        )
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in rewards))

    def test_single_group_is_ineligible(self) -> None:
        rewards, stats = compute_local_margin_rewards(
            [completion("2.0"), completion("2.2")],
            target_mean=[2.0, 2.0],
            target_std=[1.0, 1.0],
            sample_ids=["same", "same"],
        )
        self.assertEqual(rewards, [0.0, 0.0])
        self.assertFalse(stats["eligible"])

    def test_explicit_six_image_cohorts_do_not_leak_pairwise_comparisons(self) -> None:
        completions: list[str] = []
        targets: list[float] = []
        sample_ids: list[str] = []
        cohort_ids: list[str] = []
        for cohort_index in range(2):
            for image_index in range(6):
                target = 1.5 + 0.5 * image_index
                prediction = target if cohort_index == 0 else 5.5 - target
                for _ in range(2):
                    completions.append(completion(str(prediction)))
                    targets.append(target)
                    sample_ids.append(f"c{cohort_index}-image{image_index}")
                    cohort_ids.append(f"cohort-{cohort_index}")

        rewards, stats = compute_local_margin_rewards(
            completions,
            target_mean=targets,
            target_std=[1.0] * len(completions),
            sample_ids=sample_ids,
            comparison_cohort_ids=cohort_ids,
            expected_groups_per_cohort=6,
        )
        expected: list[float] = []
        for cohort_index in range(2):
            start = cohort_index * 12
            stop = start + 12
            cohort_rewards, _ = compute_local_margin_rewards(
                completions[start:stop],
                target_mean=targets[start:stop],
                target_std=[1.0] * 12,
                sample_ids=sample_ids[start:stop],
            )
            expected.extend(cohort_rewards)

        self.assertEqual(len(rewards), 24)
        for actual, wanted in zip(rewards, expected):
            self.assertAlmostEqual(actual, wanted, places=12)
        self.assertEqual(stats["comparison_scope"], "explicit_cohort")
        self.assertEqual(stats["num_cohorts"], 2)
        self.assertEqual(stats["cohort_group_sizes"], [6, 6])
        self.assertEqual(stats["pair_count"], 24 * 5)

    def test_sample_group_cannot_span_margin_cohorts(self) -> None:
        with self.assertRaisesRegex(ValueError, "spans multiple comparison cohorts"):
            compute_local_margin_rewards(
                [completion("2.0"), completion("2.1")],
                target_mean=[2.0, 2.0],
                sample_ids=["same", "same"],
                comparison_cohort_ids=["a", "b"],
            )

    def test_expected_margin_cohort_size_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "group-count mismatch"):
            compute_local_margin_rewards(
                [completion("2.0"), completion("4.0")],
                target_mean=[2.0, 4.0],
                sample_ids=["low", "high"],
                comparison_cohort_ids=["a", "a"],
                expected_groups_per_cohort=6,
            )


if __name__ == "__main__":
    unittest.main()
