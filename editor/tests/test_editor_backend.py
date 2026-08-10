from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "actor" / "plugin"))

from editor_backend import (  # noqa: E402
    editor_backend,
    editor_urls,
    request_image_edit,
    select_editor_url,
    trajectory_request_index,
)


class EditorBackendTests(unittest.TestCase):
    def test_diffusers_is_default_and_comfy_requires_explicit_selection(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(editor_backend(), "diffusers")
        with patch.dict(os.environ, {"IMAGE_EDIT_BACKEND": "comfy"}, clear=True):
            self.assertEqual(editor_backend(), "comfy")
        with patch.dict(os.environ, {"IMAGE_EDIT_BACKEND": "automatic"}, clear=True):
            with self.assertRaisesRegex(ValueError, "IMAGE_EDIT_BACKEND"):
                editor_backend()

    def test_default_diffusers_urls_and_rank_plus_index_routing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                editor_urls("diffusers"),
                [
                    "http://127.0.0.1:8212",
                    "http://127.0.0.1:8213",
                    "http://127.0.0.1:8214",
                    "http://127.0.0.1:8215",
                ],
            )
            self.assertEqual(select_editor_url(0, rank=0), "http://127.0.0.1:8212")
            self.assertEqual(select_editor_url(0, rank=3), "http://127.0.0.1:8215")
            self.assertEqual(select_editor_url(1, rank=3), "http://127.0.0.1:8212")

    def test_request_index_is_stable_and_unique_across_rank_step_and_completion(self) -> None:
        values = {
            trajectory_request_index(rollout_call=step, rank=rank, completion_index=index)
            for step in (1, 2)
            for rank in (0, 3)
            for index in (0, 7)
        }
        self.assertEqual(len(values), 8)
        self.assertEqual(
            trajectory_request_index(rollout_call=2, rank=3, completion_index=7),
            trajectory_request_index(rollout_call=2, rank=3, completion_index=7),
        )

    def test_diffusers_receives_exact_actor_editing_and_never_falls_back(self) -> None:
        calls: list[dict] = []
        comfy_calls: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            edited = Path(tmp) / "edited.png"
            source.write_bytes(b"source")
            edited.write_bytes(b"edited")

            def diffusers_request(**kwargs):
                calls.append(kwargs)
                return {
                    "status": "success",
                    "backend": "diffusers_flux2",
                    "edited_path": str(edited),
                    "seed": 123,
                    "profile_name": "eager_4b_bf16",
                }

            def comfy_request(*args, **kwargs):
                comfy_calls.append({"args": args, "kwargs": kwargs})
                raise AssertionError("ComfyUI must not be called")

            result = request_image_edit(
                image_path=str(source),
                editing="Increase local clarity without changing composition.",
                request_index=42,
                completion_index=2,
                backend="diffusers",
                editor_url="http://127.0.0.1:8212",
                diffusers_request=diffusers_request,
                comfy_request=comfy_request,
            )
            self.assertEqual(result["edited_path"], str(edited))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["positive_prompt"], "Increase local clarity without changing composition.")
        self.assertEqual(calls[0]["negative_prompt"], "")
        self.assertEqual(calls[0]["region_prompt"], "")
        self.assertEqual(calls[0]["edit_plan"], {})
        self.assertEqual(calls[0]["request_index"], 42)
        self.assertEqual(comfy_calls, [])

    def test_diffusers_exception_is_not_retried_with_comfy(self) -> None:
        comfy_called = False

        def fail_diffusers(**kwargs):
            raise RuntimeError("diffusers failed")

        def comfy_request(*args, **kwargs):
            nonlocal comfy_called
            comfy_called = True

        with self.assertRaisesRegex(RuntimeError, "diffusers failed"):
            request_image_edit(
                image_path="/tmp/source.png",
                editing="fix blur",
                request_index=1,
                completion_index=0,
                backend="diffusers",
                editor_url="http://127.0.0.1:8212",
                diffusers_request=fail_diffusers,
                comfy_request=comfy_request,
            )
        self.assertFalse(comfy_called)

    def test_comfy_backup_is_used_only_when_explicit(self) -> None:
        calls = []

        def comfy_request(image_path, regions, idx, adapter_url=None):
            calls.append((image_path, regions, idx, adapter_url))
            return {"status": "success", "path": "/tmp/edited.png"}

        result = request_image_edit(
            image_path="/tmp/source.png",
            editing="fix blur",
            request_index=99,
            completion_index=3,
            backend="comfy",
            editor_url="http://127.0.0.1:8214",
            comfy_request=comfy_request,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(calls[0][1], [{"bbox": [0.0, 0.0, 1.0, 1.0], "instruction": "fix blur"}])


if __name__ == "__main__":
    unittest.main()
