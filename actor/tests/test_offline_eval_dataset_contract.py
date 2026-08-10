from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_actor_outputs_editor_judge.py"
)
SPEC = importlib.util.spec_from_file_location("offline_eval_dataset_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OfflineEvalDatasetContractTest(unittest.TestCase):
    def test_default_contract_is_unchanged(self) -> None:
        self.assertEqual(
            MODULE.parse_dataset_contract(None),
            MODULE.DEFAULT_DATASETS,
        )

    def test_validation_only_contract(self) -> None:
        self.assertEqual(
            MODULE.parse_dataset_contract('{"validation": 200}'),
            (("validation", 200),),
        )
        self.assertEqual(
            MODULE.parse_dataset_contract('[["validation", 200]]'),
            (("validation", 200),),
        )

    def test_rejects_invalid_or_duplicate_entries(self) -> None:
        for payload in (
            "[]",
            '[["validation", 0]]',
            '[["validation", 200], ["validation", 200]]',
            '[["validation/path", 200]]',
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(RuntimeError):
                    MODULE.parse_dataset_contract(payload)


if __name__ == "__main__":
    unittest.main()
