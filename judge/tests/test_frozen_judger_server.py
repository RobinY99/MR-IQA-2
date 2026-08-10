from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
os.environ["VF_JUDGE_PROMPT_SCHEMA"] = "e5_training_reasoning_v5"
os.environ["VF_JUDGE_MODEL_ID"] = "e5_judge"
os.environ["VF_JUDGE_MODEL_PATH"] = "models/judge"
os.environ["VF_JUDGE_MODEL_TREE_SHA256"] = "test-tree-sha256"
sys.path.insert(0, str(ROOT))


def load_server_module():
    path = ROOT / "server.py"
    spec = importlib.util.spec_from_file_location("frozen_judger_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FrozenJudgerServerTests(unittest.TestCase):
    def test_request_exactly_matches_precompute_message_and_image_contract(self) -> None:
        module = load_server_module()
        payload = module.build_infer_request_payload("/tmp/image.png")
        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["system", "user"],
        )
        self.assertEqual(payload["images"], ["/tmp/image.png"])
        self.assertTrue(payload["messages"][1]["content"].startswith("<image>"))
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_request_config_exactly_matches_precompute_deterministic_decode(self) -> None:
        module = load_server_module()
        kwargs = module.build_request_config_kwargs()
        self.assertEqual(
            kwargs,
            {
                "max_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
                "repetition_penalty": 1.0,
                "presence_penalty": 1.5,
                "seed": 42,
                "return_details": True,
            },
        )

    def test_completion_summary_preserves_raw_outputs_and_strict_invalids(self) -> None:
        module = load_server_module()
        result = module.summarize_completions(
            [
                '{"reasoning":{"evidence":"Visible evidence.",'
                '"solution":"Apply a mild faithful refinement."},"rating":"2.50"}',
                "2.80",
                '{"reasons":"Visible evidence.","rating":"5.20"}',
            ]
        )
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["valid_count"], 1)
        self.assertEqual(result["requested_count"], 3)
        self.assertEqual([item["completion"] for item in result["outputs"]], [
            '{"reasoning":{"evidence":"Visible evidence.",'
            '"solution":"Apply a mild faithful refinement."},"rating":"2.50"}',
            "2.80",
            '{"reasons":"Visible evidence.","rating":"5.20"}',
        ])
        self.assertIsNone(result["outputs"][1]["score"])
        self.assertTrue(result["outputs"][1]["errors"])

    def test_import_has_no_heavy_framework_dependency(self) -> None:
        source = (ROOT / "server.py").read_text(encoding="utf-8")
        prefix = source.split("class FrozenJudger", 1)[0]
        self.assertNotIn("import torch", prefix)
        self.assertNotIn("from transformers", prefix)
        self.assertNotIn("from swift", prefix)

    def test_non_thinking_prefix_is_removed_only_when_exact(self) -> None:
        module = load_server_module()
        self.assertEqual(
            module.strip_non_thinking_prefix(
                '<think>\n\n</think>\n\n{"reasons":"x","rating":"3.00"}'
            ),
            '{"reasons":"x","rating":"3.00"}',
        )
        self.assertEqual(module.strip_non_thinking_prefix("prefix"), "prefix")

    def test_dynamic_batcher_groups_concurrent_jobs_without_changing_payloads(
        self,
    ) -> None:
        module = load_server_module()

        class DummyInferRequest:
            def __init__(self, **payload):
                self.payload = payload

        class DummyEngine:
            def __init__(self) -> None:
                self.batch_sizes = []

            def infer(self, requests, *, request_config, use_tqdm):
                self.batch_sizes.append(len(requests))
                return [
                    SimpleNamespace(
                        prompt_token_ids=[1, 2],
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content=(
                                        '{"reasoning":{"evidence":"Visible blur.",'
                                        '"solution":"Apply restrained sharpening."},'
                                        '"rating":"3.00"}'
                                    )
                                ),
                                finish_reason="stop",
                                token_ids=[3, 4],
                            )
                        ],
                    )
                    for _ in requests
                ]

        judger = module.FrozenJudger.__new__(module.FrozenJudger)
        judger._InferRequest = DummyInferRequest
        judger._request_config = object()
        judger._engine = DummyEngine()
        judger._max_batch_size = 4
        judger._batch_wait_ms = 50.0
        judger._queue = module.queue.Queue()
        judger._batch_index = 0

        futures = []
        submitted_at = time.perf_counter()
        for index in range(4):
            future = module.Future()
            futures.append(future)
            judger._queue.put(
                module._ScoreJob(
                    image_path=f"/tmp/image-{index}.png",
                    repeats=1,
                    submitted_at=submitted_at,
                    future=future,
                )
            )
        worker = threading.Thread(target=judger._batch_loop, daemon=True)
        worker.start()

        results = [future.result(timeout=2.0) for future in futures]
        self.assertEqual(judger._engine.batch_sizes, [4])
        self.assertTrue(all(result["batch_request_count"] == 4 for result in results))
        self.assertTrue(all(result["batch_size"] == 4 for result in results))
        self.assertTrue(all(result["mean"] == 3.0 for result in results))
        self.assertTrue(
            all(
                result["outputs"][0]["completion_token_count"] == 2
                for result in results
            )
        )


if __name__ == "__main__":
    unittest.main()
