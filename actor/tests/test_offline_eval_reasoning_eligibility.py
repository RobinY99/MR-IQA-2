from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path


os.environ["VF_ACTOR_SCHEMA"] = "reasoning_evidence_solution_rating"
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_actor_outputs_editor_judge.py"
)
SPEC = importlib.util.spec_from_file_location(
    "offline_eval_reasoning_eligibility",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source(completion: str) -> dict[str, object]:
    return {
        "index": 0,
        "image_path": "/tmp/image.png",
        "completion": completion,
        "gold_score": 3.0,
    }


class OfflineEvalReasoningEligibilityTest(unittest.TestCase):
    def test_valid_reasoning_is_eligible(self) -> None:
        record = MODULE.actor_record(
            "validation",
            source(
                '{"reasoning":{"evidence":"visible blur",'
                '"solution":"apply mild sharpening"},"rating":"3.00"}'
            ),
        )
        self.assertTrue(record["component_eligibility"]["reasoning"])
        self.assertTrue(record["component_eligibility"]["rating"])

    def test_invalid_rating_does_not_invalidate_reasoning(self) -> None:
        record = MODULE.actor_record(
            "validation",
            source(
                '{"reasoning":{"evidence":"visible blur",'
                '"solution":"apply mild sharpening"},"rating":"invalid"}'
            ),
        )
        self.assertTrue(record["component_eligibility"]["reasoning"])
        self.assertFalse(record["component_eligibility"]["rating"])

    def test_schema_error_invalidates_nonempty_reasoning(self) -> None:
        record = MODULE.actor_record(
            "validation",
            source(
                '{"reasoning":{"evidence":"visible blur",'
                '"solution":"apply mild sharpening","extra":"not allowed"},'
                '"rating":"3.00"}'
            ),
        )
        self.assertFalse(record["component_eligibility"]["reasoning"])


if __name__ == "__main__":
    unittest.main()
