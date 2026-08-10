from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_CACHE_SHA256 = ""
PORTABLE_PAYLOAD_SCHEMA = "vf_original_score_cache_e5_judge_e5prompt_portable_v1"
_DEFAULT_ACTOR_IDS = "source-e5-judge-step725-original-score"
_DEFAULT_JUDGE_MODEL_ID = "source-e5-judge-step725"
_DEFAULT_JUDGE_MODEL_PATH = ""
_DEFAULT_JUDGE_PROMPT_HASH = (
    "fa78a4ccfd2194a2026ff0b6b722bf22b28f8fa060389c57c4adb1618ac280f6"
)
EXPECTED_CACHE_SHA256 = os.environ.get(
    "VF_ORIGINAL_SCORE_CACHE_SHA256",
    _DEFAULT_CACHE_SHA256,
)
EXPECTED_ROW_COUNT = int(
    os.environ.get("VF_ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT", "10073")
)
EXPECTED_SAMPLE_COUNT = int(
    os.environ.get("VF_ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT", "10073")
)
EXPECTED_ACTOR_IDS = {
    actor_id.strip()
    for actor_id in os.environ.get(
        "VF_ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS",
        _DEFAULT_ACTOR_IDS,
    ).split(",")
    if actor_id.strip()
}
EXPECTED_PAYLOAD_SCHEMA = os.environ.get(
    "VF_ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA",
    PORTABLE_PAYLOAD_SCHEMA,
)
EXPECTED_RATING_MIN = float(
    os.environ.get("VF_ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN", "0.0")
)
EXPECTED_RATING_MAX = float(
    os.environ.get("VF_ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX", "5.0")
)
EXPECTED_JUDGE_MODEL_ID = os.environ.get(
    "VF_JUDGE_MODEL_ID",
    _DEFAULT_JUDGE_MODEL_ID,
)
EXPECTED_JUDGE_MODEL_PATH = os.environ.get(
    "VF_JUDGE_MODEL_PATH",
    _DEFAULT_JUDGE_MODEL_PATH,
)
EXPECTED_JUDGE_MODEL_TREE_SHA256 = os.environ.get(
    "VF_JUDGE_MODEL_TREE_SHA256",
    "",
)
EXPECTED_JUDGE_PROMPT_HASH = os.environ.get(
    "VF_JUDGE_PROMPT_HASH",
    _DEFAULT_JUDGE_PROMPT_HASH,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_path(path: str | Path) -> str:
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def _portable_path(path: str | Path) -> str:
    """Normalize a dataset-relative path without binding it to this machine."""

    raw = os.fspath(path).replace("\\", "/")
    candidate = Path(raw)
    if candidate.is_absolute() or not raw or ".." in candidate.parts:
        raise RuntimeError(f"portable cache path must be dataset-relative: {raw!r}")
    return candidate.as_posix()


def _finite_rating(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    try:
        rating = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(rating) or not minimum <= rating <= maximum:
        raise ValueError(
            f"{name} must be finite and in [{minimum:g}, {maximum:g}]"
        )
    return rating


@dataclass(frozen=True)
class OriginalScoreRecord:
    sample_id: str
    image_path: str
    rating: float
    width: int
    height: int
    image_sha256: str


class OriginalScoreCache:
    """Read-only, provenance-checked access to precomputed original scores."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        expected_sha256: str = EXPECTED_CACHE_SHA256,
        expected_row_count: int = EXPECTED_ROW_COUNT,
        expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
        expected_actor_ids: set[str] | frozenset[str] = frozenset(EXPECTED_ACTOR_IDS),
        expected_payload_schema: str = EXPECTED_PAYLOAD_SCHEMA,
        expected_judge_model_id: str = EXPECTED_JUDGE_MODEL_ID,
        expected_judge_model_path: str = EXPECTED_JUDGE_MODEL_PATH,
        expected_judge_model_tree_sha256: str = EXPECTED_JUDGE_MODEL_TREE_SHA256,
        expected_judge_prompt_hash: str = EXPECTED_JUDGE_PROMPT_HASH,
        expected_rating_min: float = EXPECTED_RATING_MIN,
        expected_rating_max: float = EXPECTED_RATING_MAX,
        verify_file_sha256: bool = False,
    ):
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(f"original-score cache does not exist: {self.db_path}")
        if verify_file_sha256:
            actual_sha256 = sha256_file(self.db_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "original-score cache SHA256 mismatch: "
                    f"expected={expected_sha256}, actual={actual_sha256}"
                )
        self.expected_sha256 = str(expected_sha256)
        self.expected_payload_schema = str(expected_payload_schema)
        self.portable = self.expected_payload_schema == PORTABLE_PAYLOAD_SCHEMA
        self.expected_judge_model_id = str(expected_judge_model_id)
        self.expected_judge_model_path = str(expected_judge_model_path)
        self.expected_judge_model_tree_sha256 = str(
            expected_judge_model_tree_sha256
        )
        self.expected_judge_prompt_hash = str(expected_judge_prompt_hash)
        self.expected_rating_min = float(expected_rating_min)
        self.expected_rating_max = float(expected_rating_max)
        if (
            not math.isfinite(self.expected_rating_min)
            or not math.isfinite(self.expected_rating_max)
            or self.expected_rating_min > self.expected_rating_max
        ):
            raise ValueError("invalid original-score cache rating range")
        self._records_by_sample: dict[str, OriginalScoreRecord] = {}
        self._records_by_path: dict[str, OriginalScoreRecord] = {}
        self._records_by_basename: dict[str, OriginalScoreRecord | None] = {}
        self._row_count = 0
        self._actor_ids: set[str] = set()
        self._load_and_audit(
            expected_row_count=int(expected_row_count),
            expected_sample_count=int(expected_sample_count),
            expected_actor_ids=set(expected_actor_ids),
        )

    def _load_and_audit(
        self,
        *,
        expected_row_count: int,
        expected_sample_count: int,
        expected_actor_ids: set[str],
    ) -> None:
        connection = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
        )
        try:
            connection.execute("PRAGMA query_only = ON")
            table_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(records)")
            }
            required_columns = {
                "sample_id",
                "actor_id",
                "source_image_path",
                "source_judge_rating",
                "payload_json",
            }
            missing = sorted(required_columns - table_columns)
            if missing:
                raise RuntimeError(f"original-score cache is missing columns: {missing}")
            rows = connection.execute(
                "SELECT sample_id, actor_id, source_image_path, "
                "source_judge_rating, payload_json FROM records "
                "ORDER BY sample_id, actor_id"
            )
            seen_actors: set[str] = set()
            sample_actors: dict[str, set[str]] = {}
            row_count = 0
            for sample_id_raw, actor_id_raw, image_path_raw, rating_raw, payload_raw in rows:
                row_count += 1
                sample_id = str(sample_id_raw)
                actor_id = str(actor_id_raw)
                image_path = self._cache_path(str(image_path_raw))
                rating = _finite_rating(
                    rating_raw,
                    name="source_judge_rating",
                    minimum=self.expected_rating_min,
                    maximum=self.expected_rating_max,
                )
                try:
                    payload = json.loads(str(payload_raw))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid cache payload_json for {sample_id}/{actor_id}"
                    ) from exc
                self._validate_payload(
                    payload,
                    sample_id=sample_id,
                    actor_id=actor_id,
                    image_path=image_path,
                    rating=rating,
                )
                source = payload["source"]
                record = OriginalScoreRecord(
                    sample_id=sample_id,
                    image_path=image_path,
                    rating=rating,
                    width=int(source["width"]),
                    height=int(source["height"]),
                    image_sha256=str(source["image_sha256"]),
                )
                prior = self._records_by_sample.get(sample_id)
                if prior is not None and prior != record:
                    raise RuntimeError(
                        f"original-score cache disagrees across actor rows for {sample_id}"
                    )
                self._records_by_sample[sample_id] = record
                path_prior = self._records_by_path.get(image_path)
                if path_prior is not None and path_prior != record:
                    raise RuntimeError(
                        f"original-score cache path maps to multiple samples: {image_path}"
                    )
                self._records_by_path[image_path] = record
                basename = Path(image_path).name
                basename_prior = self._records_by_basename.get(basename)
                if basename not in self._records_by_basename:
                    self._records_by_basename[basename] = record
                elif basename_prior != record:
                    self._records_by_basename[basename] = None
                seen_actors.add(actor_id)
                sample_actors.setdefault(sample_id, set()).add(actor_id)
        finally:
            connection.close()

        if row_count != expected_row_count:
            raise RuntimeError(
                "original-score cache row count mismatch: "
                f"expected={expected_row_count}, actual={row_count}"
            )
        if len(self._records_by_sample) != expected_sample_count:
            raise RuntimeError(
                "original-score cache sample count mismatch: "
                f"expected={expected_sample_count}, actual={len(self._records_by_sample)}"
            )
        if seen_actors != expected_actor_ids:
            raise RuntimeError(
                "original-score cache actor IDs mismatch: "
                f"expected={sorted(expected_actor_ids)}, actual={sorted(seen_actors)}"
            )
        bad_actor_sets = [
            sample_id
            for sample_id, actor_ids in sample_actors.items()
            if actor_ids != expected_actor_ids
        ]
        if bad_actor_sets:
            raise RuntimeError(
                "original-score cache does not have one row per expected actor: "
                f"{bad_actor_sets[:8]}"
            )
        self._row_count = row_count
        self._actor_ids = seen_actors

    def _validate_payload(
        self,
        payload: Any,
        *,
        sample_id: str,
        actor_id: str,
        image_path: str,
        rating: float,
    ) -> None:
        if not isinstance(payload, dict):
            raise RuntimeError(f"cache payload is not an object for {sample_id}/{actor_id}")
        required = {"sample_id", "actor_id", "source", "source_judge"}
        if not required.issubset(payload):
            raise RuntimeError(f"cache payload is incomplete for {sample_id}/{actor_id}")
        if payload.get("schema_version") != self.expected_payload_schema:
            raise RuntimeError(f"unexpected cache schema for {sample_id}/{actor_id}")
        if str(payload["sample_id"]) != sample_id or str(payload["actor_id"]) != actor_id:
            raise RuntimeError(f"cache payload identity mismatch for {sample_id}/{actor_id}")
        source = payload["source"]
        judge = payload["source_judge"]
        if not isinstance(source, dict) or not isinstance(judge, dict):
            raise RuntimeError(f"cache provenance is malformed for {sample_id}/{actor_id}")
        if self._cache_path(str(source.get("image_path") or "")) != image_path:
            raise RuntimeError(f"cache source path mismatch for {sample_id}/{actor_id}")
        if int(source.get("width", 0)) <= 0 or int(source.get("height", 0)) <= 0:
            raise RuntimeError(f"cache source dimensions are invalid for {sample_id}/{actor_id}")
        image_sha256 = str(source.get("image_sha256") or "")
        if len(image_sha256) != 64:
            raise RuntimeError(f"cache source SHA256 is invalid for {sample_id}/{actor_id}")
        if str(judge.get("model_id")) != self.expected_judge_model_id:
            raise RuntimeError(f"cache Judge model ID mismatch for {sample_id}/{actor_id}")
        if self.portable:
            model_uri = str(judge.get("model_uri") or "")
            if not model_uri.startswith("hf://"):
                raise RuntimeError(
                    f"portable cache Judge model URI is invalid for {sample_id}/{actor_id}"
                )
        elif str(judge.get("model_path")) != self.expected_judge_model_path:
            raise RuntimeError(f"cache Judge model path mismatch for {sample_id}/{actor_id}")
        if (
            self.expected_judge_model_tree_sha256
            and str(judge.get("model_tree_sha256"))
            != self.expected_judge_model_tree_sha256
        ):
            raise RuntimeError(
                f"cache Judge model tree mismatch for {sample_id}/{actor_id}"
            )
        if str(judge.get("prompt_hash")) != self.expected_judge_prompt_hash:
            raise RuntimeError(f"cache Judge prompt hash mismatch for {sample_id}/{actor_id}")
        judge_rating = _finite_rating(
            judge.get("rating"),
            name="payload source Judge rating",
            minimum=self.expected_rating_min,
            maximum=self.expected_rating_max,
        )
        if not math.isclose(judge_rating, rating, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"cache Judge rating mismatch for {sample_id}/{actor_id}")

    def lookup(
        self,
        image_path: str | Path,
        *,
        sample_id: str | None = None,
    ) -> OriginalScoreRecord:
        if sample_id:
            by_sample = self._records_by_sample.get(str(sample_id))
            if by_sample is not None:
                requested_path = _normalized_path(image_path)
                path_matches = requested_path == by_sample.image_path
                if self.portable:
                    path_matches = Path(requested_path).name == Path(
                        by_sample.image_path
                    ).name
                if not path_matches:
                    raise KeyError(
                        f"sample/path mismatch in original-score cache: {sample_id}"
                    )
                return by_sample
        normalized = _normalized_path(image_path)
        lookup_path = normalized
        if self.portable and not Path(os.fspath(image_path)).is_absolute():
            lookup_path = _portable_path(image_path)
        by_path = self._records_by_path.get(lookup_path)
        if by_path is not None:
            return by_path
        by_basename = self._records_by_basename.get(Path(normalized).name)
        if by_basename is not None:
            return by_basename
        inferred_sample = f"koniq10k:{Path(normalized).name}"
        if inferred_sample in self._records_by_sample:
            return self._records_by_sample[inferred_sample]
        raise KeyError(f"image is absent from original-score cache: {image_path}")

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "vf_original_score_cache_v1",
            "db_path": str(self.db_path),
            "expected_sha256": self.expected_sha256,
            "row_count": self._row_count,
            "sample_count": len(self._records_by_sample),
            "actor_ids": sorted(self._actor_ids),
            "payload_schema": self.expected_payload_schema,
            "portable": self.portable,
            "judge_model_id": self.expected_judge_model_id,
            "judge_model_path": self.expected_judge_model_path,
            "judge_model_tree_sha256": self.expected_judge_model_tree_sha256,
            "judge_prompt_hash": self.expected_judge_prompt_hash,
            "rating_acceptance_range": [
                self.expected_rating_min,
                self.expected_rating_max,
            ],
            "read_only": True,
        }

    def _cache_path(self, path: str | Path) -> str:
        if self.portable:
            return _portable_path(path)
        return _normalized_path(path)
