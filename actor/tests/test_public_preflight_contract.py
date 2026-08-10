from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formal_preflight import (
    SYSTEM_PROMPT,
    TRAINING_USER_PROMPT,
    inspect_dataset,
    prompt_metadata,
    validate_wandb_mode,
)


class PublicPreflightContractTests(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _retained_row(image: str, sample_id: str = "sample-0") -> dict[str, object]:
        return {
            "images": [image],
            "sample_id": sample_id,
            "target_mean": 2.5,
            "target_std": 0.75,
        }

    @staticmethod
    def _run_row(image: str, sample_id: str = "sample-0") -> dict[str, object]:
        return {
            **PublicPreflightContractTests._retained_row(image, sample_id),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": TRAINING_USER_PROMPT},
            ],
            **prompt_metadata(),
        }

    def test_training_accepts_offline_and_online_logging(self) -> None:
        self.assertEqual(validate_wandb_mode(smoke=False, mode="offline"), "offline")
        self.assertEqual(validate_wandb_mode(smoke=False, mode="online"), "online")

    def test_training_rejects_disabled_or_missing_logging_mode(self) -> None:
        for mode in (None, "disabled", "dryrun"):
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                validate_wandb_mode(smoke=False, mode=mode)

    def test_smoke_requires_disabled_logging(self) -> None:
        self.assertEqual(validate_wandb_mode(smoke=True, mode="disabled"), "disabled")
        for mode in (None, "offline", "online"):
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                validate_wandb_mode(smoke=True, mode=mode)

    def test_relative_public_manifest_matches_run_scoped_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "images"
            image = root / "koniq-10k" / "sample.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            retained = Path(temporary) / "retained.jsonl"
            rewritten = Path(temporary) / "rewritten.jsonl"
            self._write_jsonl(
                retained,
                [self._retained_row("koniq-10k/sample.jpg")],
            )
            self._write_jsonl(rewritten, [self._run_row(str(image.resolve()))])

            report = inspect_dataset(
                rewritten,
                retained_source=retained,
                image_root=root,
                expected_rows=1,
            )

            self.assertEqual(report["num_rows"], 1)

    def test_retained_image_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "images"
            root.mkdir()
            outside = base / "outside.jpg"
            outside.write_bytes(b"image")
            rewritten = base / "rewritten.jsonl"
            self._write_jsonl(rewritten, [self._run_row(str(outside))])
            for name, image in (
                ("traversal", "../outside.jpg"),
                ("absolute", str(outside)),
            ):
                retained = base / f"retained-{name}.jsonl"
                self._write_jsonl(retained, [self._retained_row(image)])
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError,
                    "forbidden '\\.\\.'|must be relative|escapes TRAIN_IMAGE_ROOT",
                ):
                    inspect_dataset(
                        rewritten,
                        retained_source=retained,
                        image_root=root,
                        expected_rows=1,
                    )

    def test_retained_absolute_image_inside_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "images"
            root.mkdir()
            image = root / "sample.jpg"
            image.write_bytes(b"image")
            retained = Path(temporary) / "retained.jsonl"
            rewritten = Path(temporary) / "rewritten.jsonl"
            self._write_jsonl(retained, [self._retained_row(str(image))])
            self._write_jsonl(rewritten, [self._run_row(str(image))])

            with self.assertRaisesRegex(ValueError, "must be relative"):
                inspect_dataset(
                    rewritten,
                    retained_source=retained,
                    image_root=root,
                    expected_rows=1,
                )

    def test_retained_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "images"
            root.mkdir()
            outside = base / "outside.jpg"
            outside.write_bytes(b"image")
            (root / "escape.jpg").symlink_to(outside)
            retained = base / "retained.jsonl"
            rewritten = base / "rewritten.jsonl"
            self._write_jsonl(retained, [self._retained_row("escape.jpg")])
            self._write_jsonl(rewritten, [self._run_row(str(outside))])

            with self.assertRaisesRegex(ValueError, "escapes TRAIN_IMAGE_ROOT"):
                inspect_dataset(
                    rewritten,
                    retained_source=retained,
                    image_root=root,
                    expected_rows=1,
                )

    def test_sample_identity_still_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "images"
            root.mkdir()
            image = root / "sample.jpg"
            image.write_bytes(b"image")
            retained = Path(temporary) / "retained.jsonl"
            rewritten = Path(temporary) / "rewritten.jsonl"
            self._write_jsonl(
                retained,
                [self._retained_row("sample.jpg", "original")],
            )
            self._write_jsonl(rewritten, [self._run_row(str(image), "changed")])

            with self.assertRaisesRegex(
                ValueError,
                "changed retained sample identity, target, or order",
            ):
                inspect_dataset(
                    rewritten,
                    retained_source=retained,
                    image_root=root,
                    expected_rows=1,
                )


if __name__ == "__main__":
    unittest.main()
