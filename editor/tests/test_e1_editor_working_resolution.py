from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "actor"
    / "scripts"
    / "run_actor_outputs_editor_judge.py"
)
SPEC = importlib.util.spec_from_file_location("run_actor_outputs_editor_judge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EditorWorkingResolutionTest(unittest.TestCase):
    def test_existing_training_resolution_is_unchanged(self) -> None:
        self.assertEqual(
            MODULE.bounded_editor_working_size((512, 384)),
            (512, 384),
        )

    def test_large_landscape_and_portrait_are_bounded(self) -> None:
        self.assertEqual(
            MODULE.bounded_editor_working_size((4160, 3120)),
            (512, 384),
        )
        self.assertEqual(
            MODULE.bounded_editor_working_size((3120, 4160)),
            (384, 512),
        )
        width, height = MODULE.bounded_editor_working_size((5312, 2988))
        self.assertLessEqual(width * height, MODULE.EDITOR_WORKING_MAX_PIXELS)
        self.assertEqual(width % MODULE.EDITOR_WORKING_SIZE_MULTIPLE, 0)
        self.assertEqual(height % MODULE.EDITOR_WORKING_SIZE_MULTIPLE, 0)

    def test_preprocess_and_retention_preserve_final_source_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (832, 624), (20, 40, 60)).save(source)
            record = {"dataset": "spaq_full", "index": 3}

            editor_input, metadata = MODULE.prepare_editor_input(
                root,
                record,
                source,
                (832, 624),
            )
            with Image.open(editor_input) as opened:
                self.assertEqual(opened.size, (512, 384))
            self.assertTrue(metadata["resize_applied"])
            self.assertEqual(metadata["max_pixels"], 196608)

            generated = root / "generated.png"
            Image.new("RGB", (512, 384), (80, 100, 120)).save(generated)
            destination = root / "retained" / "edited.png"
            native, final, resized = MODULE.retain_edited_image(
                generated,
                destination,
                (832, 624),
            )
            self.assertEqual(native, (512, 384))
            self.assertEqual(final, (832, 624))
            self.assertTrue(resized)
            self.assertTrue(destination.is_file())


if __name__ == "__main__":
    unittest.main()
