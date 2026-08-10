from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

from original_score_cache import (  # noqa: E402
    PORTABLE_PAYLOAD_SCHEMA,
    OriginalScoreCache,
)


MODEL_ID = "judge"
MODEL_PATH = "/models/judge"
PROMPT_HASH = "a" * 64
ACTORS = {"native", "dapo"}
LEGACY_PAYLOAD_SCHEMA = "vf_original_score_cache_e5_judge_v1"


def payload(
    *,
    sample_id: str,
    actor_id: str,
    image_path: str,
    rating: float,
) -> str:
    return json.dumps(
        {
            "schema_version": LEGACY_PAYLOAD_SCHEMA,
            "sample_id": sample_id,
            "actor_id": actor_id,
            "source": {
                "image_path": image_path,
                "width": 512,
                "height": 384,
                "image_sha256": "b" * 64,
            },
            "source_judge": {
                "model_id": MODEL_ID,
                "model_path": MODEL_PATH,
                "prompt_hash": PROMPT_HASH,
                "rating": rating,
            },
        },
        separators=(",", ":"),
    )


class OriginalScoreCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "cache.sqlite"
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            "CREATE TABLE records ("
            "sample_id TEXT, actor_id TEXT, source_image_path TEXT, "
            "source_judge_rating REAL, payload_json TEXT)"
        )
        for sample_index in range(2):
            sample_id = f"koniq10k:image{sample_index}.jpg"
            image_path = f"/images/image{sample_index}.jpg"
            rating = 3.0 + sample_index
            for actor_id in sorted(ACTORS):
                connection.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                    (
                        sample_id,
                        actor_id,
                        image_path,
                        rating,
                        payload(
                            sample_id=sample_id,
                            actor_id=actor_id,
                            image_path=image_path,
                            rating=rating,
                        ),
                    ),
                )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def load(self) -> OriginalScoreCache:
        digest = hashlib.sha256(self.db_path.read_bytes()).hexdigest()
        return OriginalScoreCache(
            self.db_path,
            expected_sha256=digest,
            expected_row_count=4,
            expected_sample_count=2,
            expected_actor_ids=ACTORS,
            expected_payload_schema=LEGACY_PAYLOAD_SCHEMA,
            expected_judge_model_id=MODEL_ID,
            expected_judge_model_path=MODEL_PATH,
            expected_judge_prompt_hash=PROMPT_HASH,
            expected_rating_min=1.0,
            expected_rating_max=5.0,
            verify_file_sha256=True,
        )

    def test_read_only_cache_resolves_path_sample_and_unique_basename(self) -> None:
        cache = self.load()
        self.assertEqual(cache.lookup("/images/image0.jpg").rating, 3.0)
        self.assertEqual(
            cache.lookup(
                "/images/image1.jpg",
                sample_id="koniq10k:image1.jpg",
            ).rating,
            4.0,
        )
        self.assertEqual(cache.lookup("/other/image1.jpg").sample_id, "koniq10k:image1.jpg")

    def test_disagreement_across_actor_rows_fails_closed(self) -> None:
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            "UPDATE records SET source_judge_rating = 2.0 "
            "WHERE sample_id = ? AND actor_id = ?",
            ("koniq10k:image0.jpg", "native"),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(RuntimeError, "Judge rating mismatch"):
            self.load()

    def test_sample_and_path_mismatch_fails_closed(self) -> None:
        cache = self.load()
        with self.assertRaises(KeyError):
            cache.lookup(
                "/images/image1.jpg",
                sample_id="koniq10k:image0.jpg",
            )

    def test_nonnegative_judge_range_requires_explicit_opt_in(self) -> None:
        connection = sqlite3.connect(self.db_path)
        for actor_id in sorted(ACTORS):
            connection.execute(
                "UPDATE records SET source_judge_rating = ?, payload_json = ? "
                "WHERE sample_id = ? AND actor_id = ?",
                (
                    0.83,
                    payload(
                        sample_id="koniq10k:image0.jpg",
                        actor_id=actor_id,
                        image_path="/images/image0.jpg",
                        rating=0.83,
                    ),
                    "koniq10k:image0.jpg",
                    actor_id,
                ),
            )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(ValueError, r"\[1, 5\]"):
            self.load()

        digest = hashlib.sha256(self.db_path.read_bytes()).hexdigest()
        cache = OriginalScoreCache(
            self.db_path,
            expected_sha256=digest,
            expected_row_count=4,
            expected_sample_count=2,
            expected_actor_ids=ACTORS,
            expected_payload_schema=LEGACY_PAYLOAD_SCHEMA,
            expected_judge_model_id=MODEL_ID,
            expected_judge_model_path=MODEL_PATH,
            expected_judge_prompt_hash=PROMPT_HASH,
            expected_rating_min=0.0,
            expected_rating_max=5.0,
            verify_file_sha256=True,
        )
        self.assertEqual(cache.lookup("/images/image0.jpg").rating, 0.83)
        self.assertEqual(
            cache.audit_metadata()["rating_acceptance_range"],
            [0.0, 5.0],
        )

    def test_portable_cache_resolves_relocated_dataset_path(self) -> None:
        portable_db = Path(self.tempdir.name) / "portable.sqlite"
        portable_actor = "source-e5-judge-step725-original-score"
        portable_model = "source-e5-judge-step725"
        relative_path = "koniq-10k/512x384/image0.jpg"
        portable_payload = json.dumps(
            {
                "schema_version": PORTABLE_PAYLOAD_SCHEMA,
                "sample_id": "koniq10k:image0.jpg",
                "actor_id": portable_actor,
                "source": {
                    "image_path": relative_path,
                    "width": 512,
                    "height": 384,
                    "image_sha256": "b" * 64,
                },
                "source_judge": {
                    "model_id": portable_model,
                    "model_uri": "hf://RobinY99/MR-IQA-2/judge/source-e5",
                    "model_tree_sha256": "c" * 64,
                    "prompt_hash": PROMPT_HASH,
                    "rating": 3.25,
                },
            },
            separators=(",", ":"),
        )
        connection = sqlite3.connect(portable_db)
        connection.execute(
            "CREATE TABLE records ("
            "sample_id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, "
            "source_image_path TEXT NOT NULL UNIQUE, "
            "source_judge_rating REAL NOT NULL, payload_json TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
            (
                "koniq10k:image0.jpg",
                portable_actor,
                relative_path,
                3.25,
                portable_payload,
            ),
        )
        connection.commit()
        connection.close()

        cache = OriginalScoreCache(
            portable_db,
            expected_row_count=1,
            expected_sample_count=1,
            expected_actor_ids={portable_actor},
            expected_payload_schema=PORTABLE_PAYLOAD_SCHEMA,
            expected_judge_model_id=portable_model,
            expected_judge_model_tree_sha256="c" * 64,
            expected_judge_prompt_hash=PROMPT_HASH,
            expected_rating_min=0.0,
            expected_rating_max=5.0,
        )
        relocated = "/datasets/koniq-10k/512x384/image0.jpg"
        self.assertEqual(cache.lookup(relocated).rating, 3.25)
        self.assertEqual(
            cache.lookup(relocated, sample_id="koniq10k:image0.jpg").rating,
            3.25,
        )
        self.assertTrue(cache.audit_metadata()["portable"])

    def test_portable_cache_rejects_absolute_stored_paths(self) -> None:
        portable_db = Path(self.tempdir.name) / "bad-portable.sqlite"
        connection = sqlite3.connect(portable_db)
        connection.execute(
            "CREATE TABLE records (sample_id TEXT, actor_id TEXT, "
            "source_image_path TEXT, source_judge_rating REAL, payload_json TEXT)"
        )
        connection.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
            (
                "koniq10k:image0.jpg",
                "portable",
                "/private/images/image0.jpg",
                3.0,
                "{}",
            ),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(RuntimeError, "dataset-relative"):
            OriginalScoreCache(
                portable_db,
                expected_row_count=1,
                expected_sample_count=1,
                expected_actor_ids={"portable"},
                expected_payload_schema=PORTABLE_PAYLOAD_SCHEMA,
            )


if __name__ == "__main__":
    unittest.main()
