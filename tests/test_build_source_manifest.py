from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_source_manifest.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceManifestBuilderCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_root = self.root / "images"
        self.image_root.mkdir()
        self.train_manifest = self.root / "train.jsonl"
        self.output = self.root / "source-manifest.jsonl"

        self.first_image = self.image_root / "nested" / "first.png"
        self.first_image.parent.mkdir()
        Image.new("RGB", (7, 5), color=(11, 23, 47)).save(self.first_image)

        self.second_image = self.image_root / "second.png"
        Image.new("RGB", (3, 9), color=(97, 53, 19)).save(self.second_image)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_train_manifest(self, rows: list[dict[str, object]]) -> None:
        self.train_manifest.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    def run_builder(
        self,
        *,
        expected_samples: int | None = 2,
        output: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(SCRIPT),
            "--train-manifest",
            str(self.train_manifest),
            "--image-root",
            str(self.image_root),
            "--output",
            str(output or self.output),
        ]
        if expected_samples is not None:
            command.extend(["--expected-samples", str(expected_samples)])
        return subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def valid_rows(self) -> list[dict[str, object]]:
        return [
            {
                "sample_id": "sample-one",
                "source_image": "nested/first.png",
                "images": ["nested/first.png"],
            },
            {
                "sample_id": "sample-two",
                "source_image": "second.png",
                "images": ["second.png"],
            },
        ]

    def assert_failed_without_output(
        self, result: subprocess.CompletedProcess[str]
    ) -> None:
        self.assertNotEqual(result.returncode, 0, msg=result.stdout)
        self.assertFalse(self.output.exists(), msg=result.stdout + result.stderr)

    def test_builds_atomic_jsonl_with_complete_image_provenance(self) -> None:
        self.write_train_manifest(self.valid_rows())

        result = self.run_builder()

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        records = [
            json.loads(line)
            for line in self.output.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            records,
            [
                {
                    "sample_id": "sample-one",
                    "source_image_path": str(self.first_image.resolve()),
                    "source_image_sha256": sha256(self.first_image),
                    "source_width": 7,
                    "source_height": 5,
                },
                {
                    "sample_id": "sample-two",
                    "source_image_path": str(self.second_image.resolve()),
                    "source_image_sha256": sha256(self.second_image),
                    "source_width": 3,
                    "source_height": 9,
                },
            ],
        )
        self.assertTrue(self.output.read_bytes().endswith(b"\n"))
        self.assertEqual(list(self.root.glob(f".{self.output.name}.*.tmp")), [])

    def test_rejects_duplicate_sample_id_without_partial_output(self) -> None:
        rows = self.valid_rows()
        rows[1]["sample_id"] = rows[0]["sample_id"]
        self.write_train_manifest(rows)

        self.assert_failed_without_output(self.run_builder())

    def test_rejects_duplicate_source_image_without_partial_output(self) -> None:
        rows = self.valid_rows()
        rows[1]["source_image"] = rows[0]["source_image"]
        rows[1]["images"] = rows[0]["images"]
        self.write_train_manifest(rows)

        self.assert_failed_without_output(self.run_builder())

    def test_rejects_missing_image_without_partial_output(self) -> None:
        rows = self.valid_rows()
        rows[1]["source_image"] = "missing.png"
        rows[1]["images"] = ["missing.png"]
        self.write_train_manifest(rows)

        self.assert_failed_without_output(self.run_builder())

    def test_rejects_existing_output_without_overwriting_it(self) -> None:
        self.write_train_manifest(self.valid_rows())
        sentinel = b"keep-existing-output-byte-for-byte\n"
        self.output.write_bytes(sentinel)

        result = self.run_builder()

        self.assertNotEqual(result.returncode, 0, msg=result.stdout)
        self.assertEqual(self.output.read_bytes(), sentinel)

    def test_default_expected_sample_count_is_7000(self) -> None:
        self.write_train_manifest(self.valid_rows())

        result = self.run_builder(expected_samples=None)

        self.assert_failed_without_output(result)
        self.assertIn("7000", result.stdout + result.stderr)

    def test_rejects_explicit_expected_sample_mismatch_atomically(self) -> None:
        self.write_train_manifest(self.valid_rows())

        self.assert_failed_without_output(self.run_builder(expected_samples=3))


if __name__ == "__main__":
    unittest.main()
