from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["VF_JUDGE_PROMPT_SCHEMA"] = "e5_training_reasoning_v5"
os.environ["VF_JUDGE_MODEL_ID"] = "e5_judge"
os.environ["VF_JUDGE_MODEL_PATH"] = "models/judge"
os.environ["VF_JUDGE_MODEL_TREE_SHA256"] = "test-tree-sha256"
sys.path.insert(0, str(ROOT))

from contract import (  # noqa: E402
    JUDGER_GENERATION,
    JUDGER_MODEL_ID,
    JUDGER_MODEL_PATH,
    JUDGER_MODEL_TREE_SHA256,
    JUDGER_PORTS,
    JUDGER_PROMPT,
    JUDGER_PROMPT_HASH,
    JUDGER_PROMPT_SHA256,
    JUDGER_PYTHON,
    JUDGER_SYSTEM_PROMPT,
    JUDGER_SYSTEM_PROMPT_SHA256,
    JUDGER_USER_PROMPT,
    JUDGER_USER_PROMPT_SHA256,
    judger_metadata,
    parse_score_completion,
)


class FrozenJudgerContractTests(unittest.TestCase):
    def test_accepts_only_complete_ordered_reasoning_rating_json(self) -> None:
        for text, expected in [
            (
                '{"reasoning":{"evidence":"Visible blur.",'
                '"solution":"Apply mild sharpening."},"rating":"3.42"}',
                3.42,
            ),
            (
                '  {"reasoning":{"evidence":"Visible noise.",'
                '"solution":"Apply mild denoising."},"rating":"1.00"}\n',
                1.0,
            ),
            (
                '{"reasoning":{"evidence":"Clean detail.",'
                '"solution":"Preserve the current rendering."},"rating":"5.00"}',
                5.0,
            ),
        ]:
            score, errors = parse_score_completion(text)
            self.assertEqual(score, expected)
            self.assertEqual(errors, [])

    def test_rejects_recovery_clamping_and_extra_text(self) -> None:
        invalid = [
            "3.42",
            (
                '{"rating":"3.42","reasoning":{"evidence":"Visible blur.",'
                '"solution":"Apply mild sharpening."}}'
            ),
            (
                '{"reasoning":{"evidence":"Visible blur.",'
                '"solution":"Apply mild sharpening."},"rating":"3.42"} trailing'
            ),
            (
                '{"reasoning":{"evidence":"","solution":"Apply mild sharpening."},'
                '"rating":"3.42"}'
            ),
            (
                '{"reasoning":{"evidence":"Visible blur.","solution":""},'
                '"rating":"3.42"}'
            ),
            (
                '{"reasoning":{"evidence":"Visible blur.",'
                '"solution":"Apply mild sharpening."},"rating":"5.01"}'
            ),
            (
                '{"reasoning":{"evidence":"Visible blur.",'
                '"solution":"Apply mild sharpening."},"rating":"nan"}'
            ),
            (
                '{"reasoning":{"evidence":"Visible blur.",'
                '"solution":"Apply mild sharpening."},"rating":"3.4.2"}'
            ),
            (
                '{"reasoning":{"evidence":"Visible blur.",'
                '"solution":"Apply mild sharpening."},"rating":"3.42","extra":1}'
            ),
            "",
        ]
        for text in invalid:
            with self.subTest(text=text):
                score, errors = parse_score_completion(text)
                self.assertIsNone(score)
                self.assertTrue(errors)

    def test_identity_uses_the_configured_e5_judge(self) -> None:
        self.assertEqual(JUDGER_MODEL_ID, "e5_judge")
        self.assertEqual(JUDGER_MODEL_PATH, "models/judge")
        self.assertEqual(JUDGER_MODEL_TREE_SHA256, "test-tree-sha256")
        self.assertTrue(JUDGER_PYTHON)
        self.assertEqual(JUDGER_PORTS, (8204, 8205, 8206, 8207))
        self.assertEqual(hashlib.sha256(JUDGER_PROMPT.encode()).hexdigest(), JUDGER_PROMPT_SHA256)
        self.assertEqual(
            hashlib.sha256(JUDGER_SYSTEM_PROMPT.encode()).hexdigest(),
            JUDGER_SYSTEM_PROMPT_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(JUDGER_USER_PROMPT.encode()).hexdigest(),
            JUDGER_USER_PROMPT_SHA256,
        )
        self.assertEqual(
            JUDGER_PROMPT_HASH,
            "fa78a4ccfd2194a2026ff0b6b722bf22b28f8fa060389c57c4adb1618ac280f6",
        )
        metadata = judger_metadata()
        self.assertEqual(metadata["backend"], "e5_qwen35_4b_vllm_judge")
        self.assertTrue(metadata["deterministic"])
        self.assertTrue(metadata["cache_compatible"])
        self.assertEqual(metadata["generation"], JUDGER_GENERATION)
        self.assertEqual(metadata["generation"]["max_tokens"], 256)
        self.assertEqual(metadata["generation"]["temperature"], 0.0)
        self.assertEqual(metadata["generation"]["enable_thinking"], False)
        self.assertEqual(metadata["generation"]["max_pixels"], 196608)
        self.assertEqual(metadata["generation"]["min_pixels"], 3136)


if __name__ == "__main__":
    unittest.main()
