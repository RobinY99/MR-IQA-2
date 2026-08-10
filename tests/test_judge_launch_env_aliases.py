from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class JudgeLaunchEnvironmentAliasTests(unittest.TestCase):
    def test_print_plan_accepts_public_judge_environment_names(self) -> None:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("JUDGER_")
        }
        environment.update(
            {
                "JUDGER_PYTHON": sys.executable,
                "JUDGE_MODEL_ID": "source-e5-judge-step725",
                "JUDGE_MODEL_PATH": "/models/judge",
                "JUDGE_MANIFEST_PATH": "/models/judge/provenance.json",
                "JUDGE_MODEL_TREE_SHA256": "a" * 64,
                "JUDGE_MODEL_EXPORT_TREE_SHA256": "b" * 64,
                "JUDGE_PROMPT_HASH": "c" * 64,
            }
        )

        result = subprocess.run(
            ["bash", "judge/launch.sh", "--print-plan"],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["model_id"], "source-e5-judge-step725")
        self.assertEqual(plan["model_path"], "/models/judge")
        self.assertEqual(plan["model_tree_sha256"], "a" * 64)
        self.assertEqual(plan["model_export_tree_sha256"], "b" * 64)
        self.assertEqual(plan["prompt_hash"], "c" * 64)


if __name__ == "__main__":
    unittest.main()
