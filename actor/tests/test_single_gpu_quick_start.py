from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.quick_start import _editor_size, _quality_delta, parse_args, run_sequential


class SingleGpuQuickStartTests(unittest.TestCase):
    def test_editor_size_preserves_small_images_and_caps_large_images(self) -> None:
        self.assertEqual(_editor_size(512, 336, 196608), (512, 336))
        width, height = _editor_size(4000, 3000, 196608)
        self.assertEqual((width, height), (512, 384))
        self.assertEqual(_editor_size(1000, 1000, 196608), (432, 432))
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)

    def test_orchestrator_reuses_one_visible_gpu_in_validated_environments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "input.png"
            output = root / "output"
            image.write_bytes(b"image")
            args = parse_args(
                [
                    str(image),
                    "--output-dir",
                    str(output),
                    "--gpu",
                    "3",
                ]
            )
            calls = []

            def fake_runner(command, *, check, env):
                calls.append((command, check, env["CUDA_VISIBLE_DEVICES"]))
                stage = command[command.index("--stage") + 1]
                if stage == "judge":
                    (output / "result.json").write_text(
                        json.dumps({"status": "success"}),
                        encoding="utf-8",
                    )

            with patch("examples.quick_start.shutil.which", return_value="/conda"):
                result = run_sequential(args, command_runner=fake_runner)

        self.assertEqual(result, {"status": "success"})
        self.assertEqual(len(calls), 3)
        self.assertIn("mr_iqa_actor_judge", calls[0][0])
        self.assertIn("mr_iqa_editor", calls[1][0])
        self.assertIn("mr_iqa_actor_judge", calls[2][0])
        self.assertEqual([call[2] for call in calls], ["3", "3", "3"])
        self.assertTrue(all(call[1] for call in calls))

    def test_explicit_interpreters_do_not_require_conda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "input.png"
            output = root / "output"
            image.write_bytes(b"image")
            args = parse_args(
                [
                    str(image),
                    "--output-dir",
                    str(output),
                    "--actor-python",
                    "/envs/actor/bin/python",
                    "--editor-python",
                    "/envs/editor/bin/python",
                    "--actor-model",
                    "/models/actor",
                    "--editor-model",
                    "/models/editor",
                ]
            )
            calls = []

            def fake_runner(command, *, check, env):
                calls.append(command)
                stage = command[command.index("--stage") + 1]
                if stage == "judge":
                    (output / "result.json").write_text(
                        json.dumps({"status": "success"}),
                        encoding="utf-8",
                    )

            with patch("examples.quick_start.shutil.which", return_value=None):
                run_sequential(args, command_runner=fake_runner)

        self.assertEqual(calls[0][0], "/envs/actor/bin/python")
        self.assertEqual(calls[1][0], "/envs/editor/bin/python")
        self.assertEqual(calls[2][0], "/envs/actor/bin/python")
        self.assertIn("/models/actor", calls[0])
        self.assertIn("/models/editor", calls[1])

    def test_quality_delta_requires_two_valid_scores(self) -> None:
        self.assertAlmostEqual(_quality_delta({"mean": 2.75}, {"mean": 3.5}), 0.75)
        with self.assertRaisesRegex(ValueError, "input image"):
            _quality_delta({"mean": None}, {"mean": 3.5})
        with self.assertRaisesRegex(ValueError, "edited image"):
            _quality_delta({"mean": 2.75}, {"mean": None})


if __name__ == "__main__":
    unittest.main()
