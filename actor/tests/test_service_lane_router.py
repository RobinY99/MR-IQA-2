from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

from service_lane_router import PairedServiceLaneRouter, ServiceLaneRouter  # noqa: E402


EDITORS = [f"http://127.0.0.1:{8212 + index}" for index in range(4)]
JUDGES = [f"http://127.0.0.1:{8204 + index}" for index in range(4)]


class ServiceLaneRouterTests(unittest.TestCase):
    def test_full_machine_router_balances_across_eight_lanes(self) -> None:
        editors = [f"http://127.0.0.1:{8212 + index}" for index in range(8)]
        judges = [f"http://127.0.0.1:{8204 + index}" for index in range(8)]
        router = ServiceLaneRouter(
            editors,
            judges,
            gpu_indices=tuple(range(8)),
        )
        leases = [router.reserve_editor() for _ in range(8)]
        self.assertEqual({lease.gpu_index for lease in leases}, set(range(8)))
        self.assertEqual(
            router.snapshot()["schema_version"],
            "vf_service_lane_router_v2",
        )

    def test_full_machine_router_allows_two_lanes_per_gpu_when_explicit(self) -> None:
        editors = [f"http://127.0.0.1:{8212 + index}" for index in range(16)]
        judges = [f"http://127.0.0.1:{8240 + index}" for index in range(16)]
        gpu_indices = tuple(range(8)) + tuple(range(8))
        router = ServiceLaneRouter(
            editors,
            judges,
            gpu_indices=gpu_indices,
            max_lanes_per_gpu=2,
        )
        leases = [router.reserve_editor() for _ in range(16)]
        self.assertEqual(
            [lease.gpu_index for lease in leases].count(0),
            2,
        )
        self.assertEqual(
            router.snapshot()["max_lanes_per_gpu"],
            2,
        )

    def test_duplicate_gpu_lane_requires_explicit_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_lanes_per_gpu=1"):
            ServiceLaneRouter(
                ["http://127.0.0.1:8212", "http://127.0.0.1:8213"],
                ["http://127.0.0.1:8204", "http://127.0.0.1:8205"],
                gpu_indices=(0, 0),
            )

    def test_gpu_lane_count_cannot_exceed_explicit_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_lanes_per_gpu=2"):
            ServiceLaneRouter(
                [
                    "http://127.0.0.1:8212",
                    "http://127.0.0.1:8213",
                    "http://127.0.0.1:8214",
                ],
                [
                    "http://127.0.0.1:8204",
                    "http://127.0.0.1:8205",
                    "http://127.0.0.1:8206",
                ],
                gpu_indices=(0, 0, 0),
                max_lanes_per_gpu=2,
            )

    def test_paired_router_remains_fixed_to_training_gpus(self) -> None:
        editors = [f"http://127.0.0.1:{8212 + index}" for index in range(8)]
        judges = [f"http://127.0.0.1:{8204 + index}" for index in range(8)]
        with self.assertRaisesRegex(ValueError, "exactly four paired lanes"):
            PairedServiceLaneRouter(
                editors,
                judges,
                gpu_indices=tuple(range(8)),
            )

    def test_first_four_editor_requests_balance_across_all_lanes(self) -> None:
        router = PairedServiceLaneRouter(EDITORS, JUDGES, process_rank=1)
        leases = [router.reserve_editor() for _ in range(4)]
        self.assertEqual({lease.lane_index for lease in leases}, {0, 1, 2, 3})
        for lease in leases:
            router.complete(lease, elapsed_seconds=0.8, success=True)
        snapshot = router.snapshot()
        self.assertTrue(all(lane["editor_completed"] == 1 for lane in snapshot["lanes"]))

    def test_judge_prefers_same_lane_when_cost_is_comparable(self) -> None:
        router = PairedServiceLaneRouter(EDITORS, JUDGES)
        lease = router.reserve_judge(preferred_lane_index=2)
        self.assertEqual(lease.lane_index, 2)
        self.assertFalse(lease.work_stolen)
        router.complete(lease, elapsed_seconds=0.4, success=True)

    def test_judge_steals_work_when_preferred_lane_is_materially_congested(self) -> None:
        router = PairedServiceLaneRouter(
            EDITORS,
            JUDGES,
            judge_steal_ratio=1.0,
        )
        held = [
            router.reserve_judge(preferred_lane_index=0)
            for _ in range(8)
        ]
        lease = router.reserve_judge(preferred_lane_index=0)
        self.assertNotEqual(lease.lane_index, 0)
        self.assertTrue(lease.work_stolen)
        router.complete(lease, elapsed_seconds=0.2, success=True)
        for item in held:
            router.complete(item, elapsed_seconds=0.5, success=True)

    def test_failure_counts_and_pending_underflow_are_audited(self) -> None:
        router = PairedServiceLaneRouter(EDITORS, JUDGES)
        lease = router.reserve_editor()
        router.complete(lease, elapsed_seconds=1.0, success=False)
        lane = router.snapshot()["lanes"][lease.lane_index]
        self.assertEqual(lane["editor_failures"], 1)
        with self.assertRaisesRegex(RuntimeError, "underflow"):
            router.complete(lease, elapsed_seconds=1.0, success=True)


if __name__ == "__main__":
    unittest.main()
