#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugin"
if str(PLUGIN) not in sys.path:
    sys.path.insert(0, str(PLUGIN))

from actor_contract import (  # noqa: E402
    actor_payload_errors,
    parse_actor_json,
    parse_valid_reasoning_component_json,
)
from editor_backend import request_image_edit  # noqa: E402
from editor_judge_contract import (  # noqa: E402
    EDITOR_PROMPT_TEMPLATE_HASH,
    EDITOR_SEMANTIC_GUARDRAIL,
    build_editor_prompt,
)
from frozen_judger_contract import (  # noqa: E402
    JUDGER_MODEL_ID,
    JUDGER_MODEL_PATH,
    JUDGER_MODEL_TREE_SHA256,
    JUDGER_PROMPT_HASH,
)
from service_lane_router import ServiceLaneRouter  # noqa: E402


DEFAULT_DATASETS = (
    ("validation", 200),
    ("koniq", 2010),
    ("spaq_full", 11125),
    ("livew", 1162),
    ("kadid_full", 10125),
    ("agiqa3k", 2982),
    ("csiq", 866),
)


def parse_dataset_contract(raw: str | None) -> tuple[tuple[str, int], ...]:
    if raw is None or not raw.strip():
        return DEFAULT_DATASETS
    payload = json.loads(raw)
    if isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = payload
    else:
        raise RuntimeError(
            "VF_OFFLINE_DATASETS_JSON must encode an object or a list of pairs"
        )
    datasets: list[tuple[str, int]] = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError(
                "VF_OFFLINE_DATASETS_JSON list entries must be [name, count] pairs"
            )
        name = str(item[0]).strip()
        count = item[1]
        if (
            not name
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for character in name
            )
        ):
            raise RuntimeError(f"invalid offline evaluation dataset name: {name!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise RuntimeError(
                f"offline evaluation dataset count must be a positive integer: {item!r}"
            )
        datasets.append((name, count))
    names = [name for name, _ in datasets]
    if not datasets or len(set(names)) != len(names):
        raise RuntimeError(
            "VF_OFFLINE_DATASETS_JSON must contain unique dataset names"
        )
    return tuple(datasets)


DATASETS = parse_dataset_contract(os.environ.get("VF_OFFLINE_DATASETS_JSON"))
EXPECTED_TOTAL_ROWS = sum(expected for _, expected in DATASETS)


def endpoint_urls(name: str, default_ports: tuple[int, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    values = (
        tuple(value.strip().rstrip("/") for value in raw.split(",") if value.strip())
        if raw
        else tuple(f"http://127.0.0.1:{port}" for port in default_ports)
    )
    if not 1 <= len(values) <= 16 or len(set(values)) != len(values):
        raise RuntimeError(f"{name} must contain one to sixteen unique URLs")
    if any(not value.startswith("http://127.0.0.1:") for value in values):
        raise RuntimeError(f"{name} must contain loopback HTTP URLs")
    return values


def endpoint_gpu_indices(name: str, endpoint_count: int) -> tuple[int, ...]:
    raw = os.environ.get(name)
    values = (
        tuple(int(value.strip()) for value in raw.split(",") if value.strip())
        if raw
        else tuple(range(endpoint_count))
    )
    if len(values) != endpoint_count:
        raise RuntimeError(
            f"{name} must contain one physical GPU index per Judge URL"
        )
    if any(value < 0 or value > 7 for value in values):
        raise RuntimeError(f"{name} GPU indices must be in [0, 7]")
    return values


EDITOR_URLS = endpoint_urls(
    "VF_OFFLINE_EDITOR_URLS",
    (8212, 8213, 8214, 8215),
)
JUDGE_URLS = endpoint_urls(
    "VF_OFFLINE_JUDGE_URLS",
    (8204, 8205, 8206, 8207),
)
if len(EDITOR_URLS) != len(JUDGE_URLS):
    raise RuntimeError("Editor and Judge endpoint counts differ")
JUDGE_GPU_INDICES = endpoint_gpu_indices(
    "VF_OFFLINE_JUDGE_GPU_INDICES",
    len(JUDGE_URLS),
)
MAX_JUDGE_INSTANCES_PER_GPU = int(
    os.environ.get("VF_OFFLINE_MAX_JUDGE_INSTANCES_PER_GPU", "1")
)
if MAX_JUDGE_INSTANCES_PER_GPU not in (1, 2):
    raise RuntimeError(
        "VF_OFFLINE_MAX_JUDGE_INSTANCES_PER_GPU must be 1 or 2"
    )
EDITOR_WORKING_MAX_PIXELS = 196608
EDITOR_WORKING_SIZE_MULTIPLE = 16
LANCZOS = (
    Image.Resampling.LANCZOS
    if hasattr(Image, "Resampling")
    else Image.LANCZOS
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


class GpuTelemetry:
    def __init__(self, eval_root: Path, phase: str, interval_seconds: float = 2.0):
        self.eval_root = eval_root
        self.phase = phase
        self.interval_seconds = interval_seconds
        self.started = time.perf_counter()
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "GpuTelemetry":
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_seconds * 2))
        self._write_summary()

    def _run(self) -> None:
        output_path = self.eval_root / "telemetry" / f"{self.phase}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    capture_output=True,
                    check=True,
                )
                sampled_at = now()
                batch = []
                for line in result.stdout.splitlines():
                    gpu, utilization, memory, power = [
                        value.strip() for value in line.split(",")
                    ]
                    row = {
                        "sampled_at": sampled_at,
                        "gpu": int(gpu),
                        "utilization_gpu_percent": float(utilization),
                        "memory_used_mib": float(memory),
                        "power_draw_watts": float(power),
                    }
                    self.samples.append(row)
                    batch.append(row)
                with output_path.open("a", encoding="utf-8") as output:
                    for row in batch:
                        output.write(
                            json.dumps(row, sort_keys=True) + "\n"
                        )
            except Exception as error:
                with output_path.open("a", encoding="utf-8") as output:
                    output.write(
                        json.dumps(
                            {
                                "sampled_at": now(),
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            self.stop_event.wait(self.interval_seconds)

    def _write_summary(self) -> None:
        by_gpu: dict[int, list[dict[str, Any]]] = {}
        for row in self.samples:
            by_gpu.setdefault(int(row["gpu"]), []).append(row)
        summaries = []
        for gpu, rows in sorted(by_gpu.items()):
            utilization = [row["utilization_gpu_percent"] for row in rows]
            summaries.append(
                {
                    "gpu": gpu,
                    "samples": len(rows),
                    "mean_utilization_gpu_percent": statistics.fmean(utilization),
                    "median_utilization_gpu_percent": statistics.median(utilization),
                    "zero_utilization_fraction": (
                        sum(value == 0 for value in utilization) / len(utilization)
                    ),
                    "max_memory_used_mib": max(
                        row["memory_used_mib"] for row in rows
                    ),
                    "mean_power_draw_watts": statistics.fmean(
                        row["power_draw_watts"] for row in rows
                    ),
                }
            )
        atomic_json(
            self.eval_root / "telemetry" / f"{self.phase}_summary.json",
            {
                "schema_version": "vf_gpu_phase_telemetry_v1",
                "phase": self.phase,
                "updated_at": now(),
                "elapsed_seconds": time.perf_counter() - self.started,
                "sampling_interval_seconds": self.interval_seconds,
                "gpu_count": len(summaries),
                "gpus": summaries,
            },
        )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_id(dataset: str, index: int, row: dict[str, Any], image_path: str) -> str:
    source = row.get("row") if isinstance(row.get("row"), dict) else {}
    for key in ("sample_id", "id", "image_id", "uid"):
        value = source.get(key)
        if value is not None and str(value).strip():
            return f"{dataset}:{str(value).strip()}"
    return f"{dataset}:{index:06d}:{Path(image_path).name}"


def actor_paths(eval_root: Path) -> list[tuple[str, int, Path]]:
    paths: list[tuple[str, int, Path]] = []
    for dataset, expected in DATASETS:
        path = eval_root / "actor_outputs" / dataset / "merged.json"
        paths.append((dataset, expected, path))
    return paths


def actor_record(dataset: str, source: dict[str, Any]) -> dict[str, Any]:
    index = int(source["index"])
    image_path = str(source["image_path"])
    completion = str(source.get("completion") or "")
    payload = source.get("parsed_payload")
    if not isinstance(payload, dict):
        payload = parse_actor_json(completion)
    errors = source.get("format_errors")
    if not isinstance(errors, list):
        errors = actor_payload_errors(payload)
    reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
    evidence = (
        str(reasoning.get("evidence") or "").strip()
        if isinstance(reasoning, dict)
        else ""
    )
    solution = (
        str(reasoning.get("solution") or "").strip()
        if isinstance(reasoning, dict)
        else ""
    )
    rating_raw = payload.get("rating") if isinstance(payload, dict) else None
    try:
        rating = float(rating_raw)
    except (TypeError, ValueError):
        rating = None
    rating_eligible = (
        rating is not None and math.isfinite(rating) and 1.0 <= rating <= 5.0
    )
    reasoning_payload, reasoning_errors = parse_valid_reasoning_component_json(
        completion
    )
    reasoning_eligible = bool(
        reasoning_payload is not None and not reasoning_errors
    )
    return {
        "dataset": dataset,
        "index": index,
        "sample_id": sample_id(dataset, index, source, image_path),
        "image_path": image_path,
        "gold_score": source.get("gold_score"),
        "raw_completion": completion,
        "finish_reason": source.get("finish_reason"),
        "prompt_token_count": source.get("prompt_token_count"),
        "completion_token_count": source.get("completion_token_count"),
        "parsed_payload": payload,
        "evidence": evidence,
        "solution": solution,
        "rating": rating,
        "json_parse_success": isinstance(payload, dict),
        "actor_format_success": not errors,
        "format_errors": list(errors),
        "component_eligibility": {
            "reasoning": reasoning_eligible,
            "rating": rating_eligible,
        },
        "source_row": source.get("row"),
    }


def connect_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS actor_outputs (
            dataset TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            gold_score REAL,
            raw_completion TEXT NOT NULL,
            evidence TEXT NOT NULL,
            solution TEXT NOT NULL,
            rating REAL,
            json_parse_success INTEGER NOT NULL,
            actor_format_success INTEGER NOT NULL,
            reasoning_eligible INTEGER NOT NULL,
            rating_eligible INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (dataset, sample_index)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS editor_results (
            dataset TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            status TEXT NOT NULL,
            edited_image_path TEXT,
            size_preserved INTEGER,
            result_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (dataset, sample_index)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS editor_judge_results (
            dataset TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            status TEXT NOT NULL,
            edited_image_path TEXT,
            original_judge_score REAL,
            edited_judge_score REAL,
            judge_delta REAL,
            size_preserved INTEGER,
            result_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (dataset, sample_index)
        )
        """
    )
    connection.commit()
    return connection


def prepare_actor_index(eval_root: Path) -> dict[str, Any]:
    output_jsonl = eval_root / "actor_outputs" / "all_samples.jsonl"
    temporary = output_jsonl.with_suffix(".jsonl.tmp")
    index_path = eval_root / "index" / "evaluation.sqlite"
    connection = connect_index(index_path)
    summaries: dict[str, Any] = {}
    total = 0
    with temporary.open("w", encoding="utf-8") as output:
        for dataset, expected, path in actor_paths(eval_root):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("results") or []
            if len(rows) != expected:
                raise RuntimeError(f"{dataset}: {len(rows)} actor rows != {expected}")
            records = [actor_record(dataset, row) for row in rows]
            if len({record["index"] for record in records}) != expected:
                raise RuntimeError(f"{dataset}: actor indices are not unique")
            for record in records:
                output.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO actor_outputs VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["dataset"],
                        record["index"],
                        record["sample_id"],
                        record["image_path"],
                        record["gold_score"],
                        record["raw_completion"],
                        record["evidence"],
                        record["solution"],
                        record["rating"],
                        int(record["json_parse_success"]),
                        int(record["actor_format_success"]),
                        int(record["component_eligibility"]["reasoning"]),
                        int(record["component_eligibility"]["rating"]),
                        json.dumps(record, ensure_ascii=False, sort_keys=True),
                    ),
                )
            summary = dict(payload.get("summary") or {})
            summary.update(
                {
                    "expected_rows": expected,
                    "reasoning_eligible_rows": sum(
                        record["component_eligibility"]["reasoning"]
                        for record in records
                    ),
                    "rating_eligible_rows": sum(
                        record["component_eligibility"]["rating"]
                        for record in records
                    ),
                    "actor_format_success_rows": sum(
                        record["actor_format_success"] for record in records
                    ),
                }
            )
            summaries[dataset] = summary
            total += expected
            connection.commit()
    os.replace(temporary, output_jsonl)
    connection.close()
    payload = {
        "status": "complete",
        "updated_at": now(),
        "total_rows": total,
        "expected_total_rows": EXPECTED_TOTAL_ROWS,
        "all_samples_jsonl": str(output_jsonl),
        "sqlite_index": str(index_path),
        "datasets": summaries,
    }
    if total != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"actor output total {total} != {EXPECTED_TOTAL_ROWS}"
        )
    atomic_json(eval_root / "actor_outputs" / "summary.json", payload)
    return payload


def load_actor_records(eval_root: Path) -> list[dict[str, Any]]:
    path = eval_root / "actor_outputs" / "all_samples.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_original_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    rows = connection.execute(
        """
        SELECT sample_id, source_image_path, source_judge_rating, payload_json
        FROM records
        GROUP BY sample_id
        """
    ).fetchall()
    connection.close()
    cache: dict[str, dict[str, Any]] = {}
    expected_payload_schema = os.environ.get(
        "VF_ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA"
    )
    for sample, source_path, score, raw_payload in rows:
        payload = json.loads(raw_payload)
        source = payload.get("source") or {}
        source_judge = payload.get("source_judge") or {}
        if (
            expected_payload_schema
            and payload.get("schema_version") != expected_payload_schema
        ):
            raise RuntimeError(
                f"cache payload schema mismatch for {sample}: "
                f"{payload.get('schema_version')!r} != {expected_payload_schema!r}"
            )
        expected_judge = {
            "model_id": JUDGER_MODEL_ID,
            "model_path": JUDGER_MODEL_PATH,
            "model_tree_sha256": JUDGER_MODEL_TREE_SHA256,
            "prompt_hash": JUDGER_PROMPT_HASH,
        }
        mismatches = {
            key: (expected, source_judge.get(key))
            for key, expected in expected_judge.items()
            if source_judge.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"cache Judge provenance mismatch for {sample}: {mismatches}"
            )
        cache[Path(source_path).name] = {
            "sample_id": sample,
            "source_image_path": source_path,
            "source_image_sha256": source.get("image_sha256"),
            "score": float(score),
            "source_judge": source_judge,
        }
    return cache


def judge_request(
    router: ServiceLaneRouter,
    image_path: Path,
    *,
    preferred_lane: int,
    retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    excluded: list[int] = []
    for attempt in range(1, retries + 1):
        lease = router.reserve_judge(
            preferred_lane_index=preferred_lane,
            excluded_lane_indices=excluded,
        )
        started = time.perf_counter()
        success = False
        try:
            response = requests.post(
                f"{lease.url}/score_image",
                json={"image_path": str(image_path), "repeats": 1},
                timeout=360,
            )
            response.raise_for_status()
            payload = response.json()
            score = payload.get("mean")
            metadata = payload.get("judger") or {}
            if (
                payload.get("status") != "success"
                or not isinstance(score, (int, float))
                or metadata.get("prompt_hash") != JUDGER_PROMPT_HASH
                or metadata.get("model_id") != JUDGER_MODEL_ID
                or metadata.get("model_path") != JUDGER_MODEL_PATH
                or metadata.get("model_tree_sha256")
                != JUDGER_MODEL_TREE_SHA256
            ):
                raise RuntimeError(f"invalid Judge response: {payload}")
            success = True
            return payload, attempts + [
                {
                    "attempt": attempt,
                    "success": True,
                    "lane_index": lease.lane_index,
                    "gpu_index": lease.gpu_index,
                    "url": lease.url,
                    "preferred_lane_index": preferred_lane,
                    "work_stolen": lease.work_stolen,
                    "predicted_wait_seconds": lease.predicted_wait_seconds,
                    "request_seconds": time.perf_counter() - started,
                }
            ]
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "success": False,
                    "lane_index": lease.lane_index,
                    "gpu_index": lease.gpu_index,
                    "url": lease.url,
                    "preferred_lane_index": preferred_lane,
                    "work_stolen": lease.work_stolen,
                    "predicted_wait_seconds": lease.predicted_wait_seconds,
                    "request_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            excluded.append(lease.lane_index)
            if len(excluded) == len(JUDGE_URLS):
                excluded.clear()
        finally:
            router.complete(
                lease,
                elapsed_seconds=time.perf_counter() - started,
                success=success,
            )
    raise RuntimeError(json.dumps(attempts, ensure_ascii=False))


def editor_request(
    router: ServiceLaneRouter,
    record: dict[str, Any],
    *,
    editor_image_path: Path,
    global_index: int,
    retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    attempts: list[dict[str, Any]] = []
    excluded: list[int] = []
    prompt = build_editor_prompt(record["evidence"], record["solution"])
    if prompt != str(record["solution"]).strip():
        raise RuntimeError("Editor prompt must equal the stripped solution")
    for attempt in range(1, retries + 1):
        lease = router.reserve_editor(excluded_lane_indices=excluded)
        started = time.perf_counter()
        success = False
        try:
            payload = request_image_edit(
                image_path=str(editor_image_path),
                editing=prompt,
                request_index=global_index,
                completion_index=0,
                backend="diffusers",
                editor_url=lease.url,
            )
            success = True
            return payload, attempts + [
                {
                    "attempt": attempt,
                    "success": True,
                    "lane_index": lease.lane_index,
                    "gpu_index": lease.gpu_index,
                    "url": lease.url,
                    "predicted_wait_seconds": lease.predicted_wait_seconds,
                    "request_seconds": time.perf_counter() - started,
                }
            ], lease.lane_index
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "success": False,
                    "lane_index": lease.lane_index,
                    "gpu_index": lease.gpu_index,
                    "url": lease.url,
                    "predicted_wait_seconds": lease.predicted_wait_seconds,
                    "request_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            excluded.append(lease.lane_index)
            if len(excluded) == len(EDITOR_URLS):
                excluded.clear()
        finally:
            router.complete(
                lease,
                elapsed_seconds=time.perf_counter() - started,
                success=success,
            )
    raise RuntimeError(json.dumps(attempts, ensure_ascii=False))


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def bounded_editor_working_size(
    source_size: tuple[int, int],
    *,
    max_pixels: int = EDITOR_WORKING_MAX_PIXELS,
    multiple: int = EDITOR_WORKING_SIZE_MULTIPLE,
) -> tuple[int, int]:
    width, height = map(int, source_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid source size: {source_size}")
    if max_pixels <= 0 or multiple <= 0:
        raise ValueError("max_pixels and multiple must be positive")
    if width * height <= max_pixels:
        return width, height

    scale = math.sqrt(max_pixels / float(width * height))
    working_width = max(
        multiple,
        int(round(width * scale / multiple)) * multiple,
    )
    working_height = max(
        multiple,
        int(round(height * scale / multiple)) * multiple,
    )
    while working_width * working_height > max_pixels:
        if working_width / width >= working_height / height:
            working_width = max(multiple, working_width - multiple)
        else:
            working_height = max(multiple, working_height - multiple)
    return working_width, working_height


def prepare_editor_input(
    eval_root: Path,
    record: dict[str, Any],
    source: Path,
    source_size: tuple[int, int],
) -> tuple[Path, dict[str, Any]]:
    working_size = bounded_editor_working_size(source_size)
    resize_applied = working_size != source_size
    if not resize_applied:
        editor_input = source
    else:
        editor_input = (
            eval_root
            / "editor_judge"
            / "editor_inputs"
            / record["dataset"]
            / (
                f"{record['index']:06d}_{source.stem}_"
                f"{working_size[0]}x{working_size[1]}.png"
            )
        )
        editor_input.parent.mkdir(parents=True, exist_ok=True)
        if not editor_input.is_file() or image_size(editor_input) != working_size:
            temporary = editor_input.with_name(
                f".{editor_input.name}.{threading.get_ident()}.tmp.png"
            )
            with Image.open(source) as opened:
                resized = opened.convert("RGB").resize(
                    working_size,
                    LANCZOS,
                )
                resized.save(temporary)
            os.replace(temporary, editor_input)

    return editor_input, {
        "policy": "bounded_area_lanczos_v1",
        "max_pixels": EDITOR_WORKING_MAX_PIXELS,
        "size_multiple": EDITOR_WORKING_SIZE_MULTIPLE,
        "source_size": list(source_size),
        "working_size": list(working_size),
        "resize_applied": resize_applied,
        "path": str(editor_input),
        "sha256": sha256_file(editor_input),
    }


def retain_edited_image(
    generated: Path,
    destination: Path,
    source_size: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], bool]:
    native_size = image_size(generated)
    resize_applied = native_size != source_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if resize_applied:
        temporary = destination.with_name(
            f".{destination.name}.{threading.get_ident()}.tmp.png"
        )
        with Image.open(generated) as opened:
            restored = opened.convert("RGB").resize(
                source_size,
                LANCZOS,
            )
            restored.save(temporary)
        os.replace(temporary, destination)
        if generated != destination:
            generated.unlink(missing_ok=True)
    elif generated != destination:
        if destination.exists():
            destination.unlink()
        try:
            os.replace(generated, destination)
        except OSError:
            shutil.copy2(generated, destination)
            generated.unlink()
    return native_size, image_size(destination), resize_applied


def process_record(
    record: dict[str, Any],
    *,
    global_index: int,
    eval_root: Path,
    router: ServiceLaneRouter,
    original_cache: dict[str, dict[str, Any]],
    retries: int,
) -> dict[str, Any]:
    base = {
        "schema_version": "vf_e1_checkpoint_editor_judge_result_v1",
        "dataset": record["dataset"],
        "index": record["index"],
        "sample_id": record["sample_id"],
        "actor_output": record,
        "updated_at": now(),
        "editor_prompt_template_hash": EDITOR_PROMPT_TEMPLATE_HASH,
        "semantic_guardrail": EDITOR_SEMANTIC_GUARDRAIL,
        "semantic_guardrail_applied": False,
    }
    if not record["component_eligibility"]["reasoning"]:
        return {
            **base,
            "status": "actor_ineligible",
            "failure_owner": "actor",
            "failure_stage": "reasoning_parse",
            "edited_image_path": None,
            "original_judge_score": None,
            "edited_judge_score": None,
            "judge_delta": None,
        }

    source = Path(record["image_path"]).resolve()
    source_hash = sha256_file(source)
    source_size = image_size(source)
    cache_entry = (
        original_cache.get(source.name)
        if record["dataset"] in {"validation", "koniq"}
        else None
    )
    cache_hit = bool(
        cache_entry
        and cache_entry.get("source_image_sha256")
        and cache_entry["source_image_sha256"] == source_hash
    )
    if cache_hit:
        original_judge = {
            "status": "cache_hit",
            "mean": cache_entry["score"],
            "cache_entry": cache_entry,
            "prompt_hash": JUDGER_PROMPT_HASH,
        }
        original_attempts: list[dict[str, Any]] = []
    else:
        original_judge, original_attempts = judge_request(
            router,
            source,
            preferred_lane=global_index % len(JUDGE_URLS),
            retries=retries,
        )

    editor_input, editor_input_metadata = prepare_editor_input(
        eval_root,
        record,
        source,
        source_size,
    )
    editor_payload, editor_attempts, editor_lane = editor_request(
        router,
        record,
        editor_image_path=editor_input,
        global_index=global_index,
        retries=retries,
    )
    generated = Path(str(editor_payload["edited_path"])).resolve()
    destination = (
        eval_root
        / "editor_judge"
        / "edited_images"
        / record["dataset"]
        / f"{record['index']:06d}_{source.stem}.png"
    )
    native_edited_size, edited_size, output_resize_applied = retain_edited_image(
        generated,
        destination,
        source_size,
    )
    if source_size != edited_size:
        raise RuntimeError(
            f"Editor size mismatch for {record['sample_id']}: "
            f"{source_size} != {edited_size}"
        )
    edited_judge, edited_attempts = judge_request(
        router,
        destination,
        preferred_lane=editor_lane,
        retries=retries,
    )
    original_score = float(original_judge["mean"])
    edited_score = float(edited_judge["mean"])
    return {
        **base,
        "status": "success",
        "failure_owner": "none",
        "failure_stage": None,
        "source_image_path": str(source),
        "source_image_sha256": source_hash,
        "source_size": list(source_size),
        "editor_prompt": build_editor_prompt(
            record["evidence"], record["solution"]
        ),
        "editor_input_preprocess": editor_input_metadata,
        "editor": editor_payload,
        "editor_attempts": editor_attempts,
        "editor_native_output_size": list(native_edited_size),
        "editor_output_resize_to_source_applied": output_resize_applied,
        "edited_image_path": str(destination),
        "edited_image_sha256": sha256_file(destination),
        "edited_size": list(edited_size),
        "size_preserved": True,
        "semantic_guardrail_applied": False,
        "solution_only_applied": True,
        "original_score_cache_hit": cache_hit,
        "original_judge": original_judge,
        "original_judge_attempts": original_attempts,
        "edited_judge": edited_judge,
        "edited_judge_attempts": edited_attempts,
        "original_judge_score": original_score,
        "edited_judge_score": edited_score,
        "judge_delta": edited_score - original_score,
        "route_snapshot": router.snapshot(),
    }


def process_editor_record(
    record: dict[str, Any],
    *,
    global_index: int,
    eval_root: Path,
    router: ServiceLaneRouter,
    retries: int,
) -> dict[str, Any]:
    base = {
        "schema_version": "vf_offline_editor_result_v2",
        "dataset": record["dataset"],
        "index": record["index"],
        "sample_id": record["sample_id"],
        "actor_output": record,
        "updated_at": now(),
        "editor_prompt_template_hash": EDITOR_PROMPT_TEMPLATE_HASH,
        "semantic_guardrail": EDITOR_SEMANTIC_GUARDRAIL,
        "semantic_guardrail_applied": False,
        "judge_deferred_until_editor_barrier": True,
    }
    if not record["component_eligibility"]["reasoning"]:
        return {
            **base,
            "status": "actor_ineligible",
            "failure_owner": "actor",
            "failure_stage": "reasoning_parse",
            "edited_image_path": None,
            "size_preserved": None,
        }

    source = Path(record["image_path"]).resolve()
    source_hash = sha256_file(source)
    source_size = image_size(source)
    editor_input, editor_input_metadata = prepare_editor_input(
        eval_root,
        record,
        source,
        source_size,
    )
    editor_payload, editor_attempts, editor_lane = editor_request(
        router,
        record,
        editor_image_path=editor_input,
        global_index=global_index,
        retries=retries,
    )
    generated = Path(str(editor_payload["edited_path"])).resolve()
    destination = (
        eval_root
        / "editor_judge"
        / "edited_images"
        / record["dataset"]
        / f"{record['index']:06d}_{source.stem}.png"
    )
    native_edited_size, edited_size, output_resize_applied = retain_edited_image(
        generated,
        destination,
        source_size,
    )
    if source_size != edited_size:
        raise RuntimeError(
            f"Editor size mismatch for {record['sample_id']}: "
            f"{source_size} != {edited_size}"
        )
    return {
        **base,
        "status": "edited",
        "failure_owner": "none",
        "failure_stage": None,
        "source_image_path": str(source),
        "source_image_sha256": source_hash,
        "source_size": list(source_size),
        "editor_prompt": build_editor_prompt(
            record["evidence"], record["solution"]
        ),
        "editor_input_preprocess": editor_input_metadata,
        "editor": editor_payload,
        "editor_attempts": editor_attempts,
        "editor_native_output_size": list(native_edited_size),
        "editor_output_resize_to_source_applied": output_resize_applied,
        "edited_image_path": str(destination),
        "edited_image_sha256": sha256_file(destination),
        "edited_size": list(edited_size),
        "size_preserved": True,
        "semantic_guardrail_applied": False,
        "solution_only_applied": True,
        "editor_lane_index": editor_lane,
        "route_snapshot": router.snapshot(),
    }


def process_judge_record(
    editor_result: dict[str, Any],
    *,
    global_index: int,
    router: ServiceLaneRouter,
    original_cache: dict[str, dict[str, Any]],
    retries: int,
) -> dict[str, Any]:
    if editor_result["status"] == "actor_ineligible":
        return {
            **editor_result,
            "schema_version": "vf_offline_editor_e5judge_result_v2",
            "status": "actor_ineligible",
            "judge_model_id": JUDGER_MODEL_ID,
            "judge_model_path": JUDGER_MODEL_PATH,
            "judge_model_tree_sha256": JUDGER_MODEL_TREE_SHA256,
            "judge_prompt_hash": JUDGER_PROMPT_HASH,
            "original_judge_score": None,
            "edited_judge_score": None,
            "judge_delta": None,
            "updated_at": now(),
        }
    if editor_result["status"] != "edited":
        raise RuntimeError(
            f"Judge received non-terminal Editor row: {editor_result['status']}"
        )

    source = Path(editor_result["source_image_path"]).resolve()
    destination = Path(editor_result["edited_image_path"]).resolve()
    if not source.is_file() or not destination.is_file():
        raise RuntimeError(
            f"Judge input missing for {editor_result['sample_id']}: "
            f"source={source.is_file()} edited={destination.is_file()}"
        )
    source_hash = sha256_file(source)
    edited_hash = sha256_file(destination)
    if source_hash != editor_result["source_image_sha256"]:
        raise RuntimeError(
            f"source hash changed before Judge for {editor_result['sample_id']}"
        )
    if edited_hash != editor_result["edited_image_sha256"]:
        raise RuntimeError(
            f"edited hash changed before Judge for {editor_result['sample_id']}"
        )
    if image_size(destination) != tuple(editor_result["source_size"]):
        raise RuntimeError(
            f"edited size changed before Judge for {editor_result['sample_id']}"
        )

    cache_entry = (
        original_cache.get(source.name)
        if editor_result["dataset"] in {"validation", "koniq"}
        else None
    )
    cache_hit = bool(
        cache_entry
        and cache_entry.get("source_image_sha256")
        and cache_entry["source_image_sha256"] == source_hash
    )
    preferred_lane = global_index % len(JUDGE_URLS)
    if cache_hit:
        original_judge = {
            "status": "cache_hit",
            "mean": cache_entry["score"],
            "cache_entry": cache_entry,
            "prompt_hash": JUDGER_PROMPT_HASH,
        }
        original_attempts: list[dict[str, Any]] = []
    else:
        original_judge, original_attempts = judge_request(
            router,
            source,
            preferred_lane=preferred_lane,
            retries=retries,
        )
    edited_judge, edited_attempts = judge_request(
        router,
        destination,
        preferred_lane=preferred_lane,
        retries=retries,
    )
    original_score = float(original_judge["mean"])
    edited_score = float(edited_judge["mean"])
    return {
        **editor_result,
        "schema_version": "vf_offline_editor_e5judge_result_v2",
        "status": "success",
        "failure_owner": "none",
        "failure_stage": None,
        "judge_started_after_editor_barrier": True,
        "judge_model_id": JUDGER_MODEL_ID,
        "judge_model_path": JUDGER_MODEL_PATH,
        "judge_model_tree_sha256": JUDGER_MODEL_TREE_SHA256,
        "judge_prompt_hash": JUDGER_PROMPT_HASH,
        "original_score_cache_hit": cache_hit,
        "original_judge": original_judge,
        "original_judge_attempts": original_attempts,
        "edited_judge": edited_judge,
        "edited_judge_attempts": edited_attempts,
        "original_judge_score": original_score,
        "edited_judge_score": edited_score,
        "judge_delta": edited_score - original_score,
        "judge_route_snapshot": router.snapshot(),
        "updated_at": now(),
    }


def persist_editor_result(
    connection: sqlite3.Connection,
    output: Any,
    result: dict[str, Any],
) -> None:
    output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    output.flush()
    connection.execute(
        """
        INSERT OR REPLACE INTO editor_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["dataset"],
            result["index"],
            result["sample_id"],
            result["status"],
            result.get("edited_image_path"),
            (
                int(result["size_preserved"])
                if result.get("size_preserved") is not None
                else None
            ),
            json.dumps(result, ensure_ascii=False, sort_keys=True),
            result["updated_at"],
        ),
    )
    connection.commit()


def persist_result(
    connection: sqlite3.Connection,
    output: Any,
    result: dict[str, Any],
) -> None:
    output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    output.flush()
    connection.execute(
        """
        INSERT OR REPLACE INTO editor_judge_results VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["dataset"],
            result["index"],
            result["sample_id"],
            result["status"],
            result.get("edited_image_path"),
            result.get("original_judge_score"),
            result.get("edited_judge_score"),
            result.get("judge_delta"),
            (
                int(result["size_preserved"])
                if result.get("size_preserved") is not None
                else None
            ),
            json.dumps(result, ensure_ascii=False, sort_keys=True),
            result["updated_at"],
        ),
    )
    connection.commit()


def completed_keys(
    connection: sqlite3.Connection,
    *,
    table: str = "editor_judge_results",
    statuses: tuple[str, ...] = ("success", "actor_ineligible"),
) -> set[tuple[str, int]]:
    if table not in {"editor_results", "editor_judge_results"}:
        raise ValueError(f"unsupported results table: {table}")
    placeholders = ",".join("?" for _ in statuses)
    return {
        (str(dataset), int(index))
        for dataset, index in connection.execute(
            f"""
            SELECT dataset, sample_index
            FROM {table}
            WHERE status IN ({placeholders})
            """,
            statuses,
        )
    }


def canonicalize_results_jsonl(
    connection: sqlite3.Connection,
    output_path: Path,
    *,
    table: str = "editor_judge_results",
) -> None:
    if table not in {"editor_results", "editor_judge_results"}:
        raise ValueError(f"unsupported results table: {table}")
    order = " ".join(
        f"WHEN '{dataset}' THEN {index}"
        for index, (dataset, _) in enumerate(DATASETS)
    )
    rows = connection.execute(
        f"""
        SELECT result_json
        FROM {table}
        ORDER BY CASE dataset {order} ELSE {len(DATASETS)} END, sample_index
        """
    )
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for (result_json,) in rows:
            output.write(str(result_json).rstrip("\n") + "\n")
    os.replace(temporary, output_path)


def write_progress(
    eval_root: Path,
    *,
    phase: str,
    total: int,
    completed: int,
    success: int,
    actor_ineligible: int,
    service_error: int,
    router: ServiceLaneRouter,
) -> None:
    atomic_json(
        eval_root / "state" / f"{phase}_state.json",
        {
            "status": "running",
            "phase": phase,
            "updated_at": now(),
            "total": total,
            "completed": completed,
            "success": success,
            "actor_ineligible": actor_ineligible,
            "service_error": service_error,
            "router": router.snapshot(),
        },
    )


def run_editor(
    eval_root: Path,
    *,
    workers: int,
    retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    records = load_actor_records(eval_root)
    if len(records) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"actor record count {len(records)} != {EXPECTED_TOTAL_ROWS}"
        )
    router = ServiceLaneRouter(
        EDITOR_URLS,
        JUDGE_URLS,
        process_rank=0,
        gpu_indices=tuple(range(len(EDITOR_URLS))),
    )
    index_path = eval_root / "index" / "evaluation.sqlite"
    connection = connect_index(index_path)
    done = completed_keys(
        connection,
        table="editor_results",
        statuses=("edited", "actor_ineligible"),
    )
    pending = [
        (global_index, record)
        for global_index, record in enumerate(records)
        if (record["dataset"], int(record["index"])) not in done
    ]
    counts = {
        status: int(count)
        for status, count in connection.execute(
            """
            SELECT status, COUNT(*)
            FROM editor_results
            WHERE status IN ('edited', 'actor_ineligible')
            GROUP BY status
            """
        )
    }
    output_path = eval_root / "editor_judge" / "editor_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = len(done)
    with output_path.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_record = {
                executor.submit(
                    process_editor_record,
                    record,
                    global_index=global_index,
                    eval_root=eval_root,
                    router=router,
                    retries=retries,
                ): record
                for global_index, record in pending
            }
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "schema_version": "vf_offline_editor_result_v2",
                        "dataset": record["dataset"],
                        "index": record["index"],
                        "sample_id": record["sample_id"],
                        "actor_output": record,
                        "status": "service_error",
                        "failure_owner": "service",
                        "failure_stage": "editor",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "edited_image_path": None,
                        "size_preserved": None,
                        "judge_deferred_until_editor_barrier": True,
                        "updated_at": now(),
                    }
                persist_editor_result(connection, output, result)
                completed += 1
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                if completed % 20 == 0 or completed == len(records):
                    write_progress(
                        eval_root,
                        phase="editor",
                        total=len(records),
                        completed=completed,
                        success=counts.get("edited", 0),
                        actor_ineligible=counts.get("actor_ineligible", 0),
                        service_error=counts.get("service_error", 0),
                        router=router,
                    )
    canonicalize_results_jsonl(
        connection,
        output_path,
        table="editor_results",
    )
    summary_rows = connection.execute(
        """
        SELECT dataset, status, COUNT(*)
        FROM editor_results
        GROUP BY dataset, status
        ORDER BY dataset, status
        """
    ).fetchall()
    image_count = connection.execute(
        """
        SELECT COUNT(*) FROM editor_results
        WHERE status='edited' AND edited_image_path IS NOT NULL
        """
    ).fetchone()[0]
    edited_rows = connection.execute(
        "SELECT COUNT(*) FROM editor_results WHERE status='edited'"
    ).fetchone()[0]
    actor_ineligible = connection.execute(
        "SELECT COUNT(*) FROM editor_results WHERE status='actor_ineligible'"
    ).fetchone()[0]
    service_errors = connection.execute(
        "SELECT COUNT(*) FROM editor_results WHERE status='service_error'"
    ).fetchone()[0]
    total_rows = connection.execute(
        "SELECT COUNT(*) FROM editor_results"
    ).fetchone()[0]
    missing_images = [
        path
        for (path,) in connection.execute(
            """
            SELECT edited_image_path FROM editor_results
            WHERE status='edited' AND edited_image_path IS NOT NULL
            """
        )
        if not Path(path).is_file()
    ]
    connection.close()
    payload = {
        "status": (
            "complete"
            if (
                service_errors == 0
                and total_rows == len(records)
                and image_count == edited_rows
                and not missing_images
            )
            else "incomplete"
        ),
        "phase": "editor",
        "updated_at": now(),
        "total_actor_rows": len(records),
        "total_editor_rows": int(total_rows),
        "edited_rows": int(edited_rows),
        "actor_ineligible_rows": int(actor_ineligible),
        "edited_images_retained": int(image_count),
        "service_error_rows": int(service_errors),
        "missing_edited_images": missing_images[:100],
        "editor_results_jsonl": str(output_path),
        "sqlite_index": str(index_path),
        "router": router.snapshot(),
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
        "eligible_rows_per_second": (
            edited_rows / max(time.perf_counter() - started, 1e-9)
        ),
        "datasets": [
            {
                "dataset": dataset,
                "status": status,
                "count": count,
            }
            for dataset, status, count in summary_rows
        ],
    }
    atomic_json(eval_root / "editor_judge" / "editor_summary.json", payload)
    atomic_json(
        eval_root / "state" / "editor_state.json",
        payload,
    )
    if payload["status"] != "complete":
        raise RuntimeError(
            "Editor phase did not satisfy its global barrier: "
            f"rows={total_rows}/{len(records)} edited={edited_rows} "
            f"images={image_count} service_errors={service_errors} "
            f"missing_images={len(missing_images)}"
        )
    barrier = {
        "schema_version": "vf_offline_editor_barrier_v1",
        "status": "passed",
        "updated_at": now(),
        "total_actor_rows": len(records),
        "total_editor_rows": int(total_rows),
        "edited_rows": int(edited_rows),
        "actor_ineligible_rows": int(actor_ineligible),
        "edited_images_retained": int(image_count),
        "service_error_rows": 0,
        "judge_started": False,
        "all_edits_finished_before_any_judge_request": True,
        "editor_results_jsonl": str(output_path),
        "sqlite_index": str(index_path),
    }
    atomic_json(eval_root / "state" / "editor_barrier.json", barrier)
    return payload


def load_editor_results(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    order = " ".join(
        f"WHEN '{dataset}' THEN {index}"
        for index, (dataset, _) in enumerate(DATASETS)
    )
    return [
        json.loads(result_json)
        for (result_json,) in connection.execute(
            f"""
            SELECT result_json
            FROM editor_results
            ORDER BY CASE dataset {order} ELSE {len(DATASETS)} END, sample_index
            """
        )
    ]


def require_editor_barrier(
    eval_root: Path,
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    barrier_path = eval_root / "state" / "editor_barrier.json"
    if not barrier_path.is_file():
        raise RuntimeError("Judge cannot start before editor_barrier.json exists")
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    if barrier.get("status") != "passed":
        raise RuntimeError(f"Editor barrier is not passed: {barrier}")
    rows = connection.execute(
        "SELECT status, COUNT(*) FROM editor_results GROUP BY status"
    ).fetchall()
    counts = {str(status): int(count) for status, count in rows}
    total = sum(counts.values())
    if total != EXPECTED_TOTAL_ROWS or counts.get("service_error", 0):
        raise RuntimeError(
            f"Editor barrier database mismatch: total={total}, counts={counts}"
        )
    if counts.get("edited", 0) != int(barrier["edited_rows"]):
        raise RuntimeError(f"Editor barrier edited-row mismatch: {counts}, {barrier}")
    missing = [
        path
        for (path,) in connection.execute(
            """
            SELECT edited_image_path FROM editor_results
            WHERE status='edited'
            """
        )
        if not path or not Path(path).is_file()
    ]
    if missing:
        raise RuntimeError(f"Editor barrier has missing images: {missing[:8]}")
    return barrier


def run_judge(
    eval_root: Path,
    *,
    cache_path: Path,
    workers: int,
    retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    original_cache = load_original_cache(cache_path)
    router = ServiceLaneRouter(
        EDITOR_URLS,
        JUDGE_URLS,
        process_rank=0,
        gpu_indices=JUDGE_GPU_INDICES,
        max_lanes_per_gpu=MAX_JUDGE_INSTANCES_PER_GPU,
    )
    index_path = eval_root / "index" / "evaluation.sqlite"
    connection = connect_index(index_path)
    barrier = require_editor_barrier(eval_root, connection)
    editor_results = load_editor_results(connection)
    if len(editor_results) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Editor result count {len(editor_results)} != {EXPECTED_TOTAL_ROWS}"
        )
    barrier["judge_started"] = True
    barrier["judge_started_at"] = now()
    barrier["judge_model_id"] = JUDGER_MODEL_ID
    barrier["judge_model_path"] = JUDGER_MODEL_PATH
    barrier["judge_model_tree_sha256"] = JUDGER_MODEL_TREE_SHA256
    barrier["judge_prompt_hash"] = JUDGER_PROMPT_HASH
    atomic_json(eval_root / "state" / "editor_barrier.json", barrier)

    done = completed_keys(connection)
    pending = [
        (global_index, result)
        for global_index, result in enumerate(editor_results)
        if (result["dataset"], int(result["index"])) not in done
    ]
    counts = {
        status: int(count)
        for status, count in connection.execute(
            """
            SELECT status, COUNT(*)
            FROM editor_judge_results
            WHERE status IN ('success', 'actor_ineligible')
            GROUP BY status
            """
        )
    }
    output_path = eval_root / "editor_judge" / "results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = len(done)
    with output_path.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_result = {
                executor.submit(
                    process_judge_record,
                    editor_result,
                    global_index=global_index,
                    router=router,
                    original_cache=original_cache,
                    retries=retries,
                ): editor_result
                for global_index, editor_result in pending
            }
            for future in as_completed(future_to_result):
                editor_result = future_to_result[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        **editor_result,
                        "schema_version": "vf_offline_editor_e5judge_result_v2",
                        "status": "service_error",
                        "failure_owner": "service",
                        "failure_stage": "e5_judge",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "original_judge_score": None,
                        "edited_judge_score": None,
                        "judge_delta": None,
                        "judge_model_id": JUDGER_MODEL_ID,
                        "judge_model_path": JUDGER_MODEL_PATH,
                        "judge_model_tree_sha256": JUDGER_MODEL_TREE_SHA256,
                        "judge_prompt_hash": JUDGER_PROMPT_HASH,
                        "updated_at": now(),
                    }
                persist_result(connection, output, result)
                completed += 1
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                if completed % 20 == 0 or completed == len(editor_results):
                    write_progress(
                        eval_root,
                        phase="e5_judge",
                        total=len(editor_results),
                        completed=completed,
                        success=counts.get("success", 0),
                        actor_ineligible=counts.get("actor_ineligible", 0),
                        service_error=counts.get("service_error", 0),
                        router=router,
                    )
    canonicalize_results_jsonl(connection, output_path)
    summary_rows = connection.execute(
        """
        SELECT dataset, status, COUNT(*), AVG(judge_delta)
        FROM editor_judge_results
        GROUP BY dataset, status
        ORDER BY dataset, status
        """
    ).fetchall()
    image_count = connection.execute(
        """
        SELECT COUNT(*) FROM editor_judge_results
        WHERE status='success' AND edited_image_path IS NOT NULL
        """
    ).fetchone()[0]
    total_rows = connection.execute(
        "SELECT COUNT(*) FROM editor_judge_results"
    ).fetchone()[0]
    service_errors = connection.execute(
        "SELECT COUNT(*) FROM editor_judge_results WHERE status='service_error'"
    ).fetchone()[0]
    connection.close()
    payload = {
        "status": (
            "complete"
            if service_errors == 0 and total_rows == len(editor_results)
            else "incomplete_service_errors"
        ),
        "phase": "e5_judge",
        "updated_at": now(),
        "total_actor_rows": len(editor_results),
        "total_judge_rows": int(total_rows),
        "edited_images_retained": int(image_count),
        "service_error_rows": int(service_errors),
        "results_jsonl": str(output_path),
        "sqlite_index": str(index_path),
        "judge_model_id": JUDGER_MODEL_ID,
        "judge_model_path": JUDGER_MODEL_PATH,
        "judge_model_tree_sha256": JUDGER_MODEL_TREE_SHA256,
        "judge_prompt_hash": JUDGER_PROMPT_HASH,
        "editor_barrier": str(eval_root / "state" / "editor_barrier.json"),
        "all_edits_finished_before_any_judge_request": True,
        "router": router.snapshot(),
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
        "successful_rows_per_second": (
            image_count / max(time.perf_counter() - started, 1e-9)
        ),
        "datasets": [
            {
                "dataset": dataset,
                "status": status,
                "count": count,
                "mean_judge_delta": mean_delta,
            }
            for dataset, status, count, mean_delta in summary_rows
        ],
    }
    atomic_json(eval_root / "editor_judge" / "summary.json", payload)
    atomic_json(eval_root / "state" / "e5_judge_state.json", payload)
    if service_errors or total_rows != len(editor_results):
        raise RuntimeError(
            "E5 Judge finished incompletely: "
            f"rows={total_rows}/{len(editor_results)} "
            f"service_errors={service_errors}"
        )
    return payload


def run_editor_judge(
    eval_root: Path,
    *,
    cache_path: Path,
    workers: int,
    retries: int,
) -> dict[str, Any]:
    stack = ROOT / "scripts" / "resident_service_stack.sh"
    if not stack.is_file():
        raise RuntimeError(f"resident service stack is missing: {stack}")

    def service(action: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["bash", str(stack), action, str(eval_root)],
            cwd=ROOT,
            env=os.environ,
            text=True,
            capture_output=True,
            check=False,
        )
        with (
            eval_root / "logs" / "three_phase_service_handoff.log"
        ).open("a", encoding="utf-8") as output:
            output.write(
                f"[{now()}] action={action} returncode={result.returncode}\n"
            )
            output.write(result.stdout)
            output.write(result.stderr)
        if check and result.returncode:
            raise RuntimeError(
                f"service action failed: {action}; "
                f"stdout={result.stdout[-2000:]!r}; stderr={result.stderr[-2000:]!r}"
            )
        return result

    barrier_path = eval_root / "state" / "editor_barrier.json"
    editor_already_complete = False
    if barrier_path.is_file():
        barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
        editor_already_complete = barrier.get("status") == "passed"

    editor_workers = max(workers, len(EDITOR_URLS) * 8)
    judge_workers = max(workers, len(JUDGE_URLS) * 8)
    editor: dict[str, Any]
    if editor_already_complete:
        editor = json.loads(
            (eval_root / "editor_judge" / "editor_summary.json").read_text(
                encoding="utf-8"
            )
        )
        service("stop-editor", check=False)
        service("stop-judge", check=False)
    else:
        editor_running = service("status-editor", check=False).returncode == 0
        service("stop-judge", check=False)
        if not editor_running:
            service("start-editor")
        atomic_json(
            eval_root / "state" / "service_handoff_state.json",
            {
                "status": "running",
                "phase": "editor",
                "updated_at": now(),
                "gpus": list(range(len(EDITOR_URLS))),
                "lane_count": len(EDITOR_URLS),
                "judge_services_running": False,
            },
        )
        try:
            with GpuTelemetry(eval_root, "editor"):
                editor = run_editor(
                    eval_root,
                    workers=editor_workers,
                    retries=retries,
                )
        finally:
            service("stop-editor", check=False)

    service("stop-editor", check=False)
    if service("status-judge", check=False).returncode != 0:
        service("start-judge")
    atomic_json(
        eval_root / "state" / "service_handoff_state.json",
        {
            "status": "running",
            "phase": "e5_judge",
            "updated_at": now(),
            "gpus": list(range(len(JUDGE_URLS))),
            "lane_count": len(JUDGE_URLS),
            "editor_services_running": False,
            "judge_model_id": JUDGER_MODEL_ID,
            "judge_model_path": JUDGER_MODEL_PATH,
            "judge_model_tree_sha256": JUDGER_MODEL_TREE_SHA256,
            "judge_prompt_hash": JUDGER_PROMPT_HASH,
        },
    )
    try:
        with GpuTelemetry(eval_root, "e5_judge"):
            judge = run_judge(
                eval_root,
                cache_path=cache_path,
                workers=judge_workers,
                retries=retries,
            )
    finally:
        service("stop-judge", check=False)
    atomic_json(
        eval_root / "state" / "service_handoff_state.json",
        {
            "status": "complete",
            "phase": "complete",
            "updated_at": now(),
            "editor_services_running": False,
            "judge_services_running": False,
        },
    )
    return {
        "status": "complete",
        "sequence": [
            "actor_outputs_complete",
            "editor_complete_barrier",
            "e5_judge_complete",
        ],
        "editor_workers": editor_workers,
        "e5_judge_workers": judge_workers,
        "editor": editor,
        "e5_judge": judge,
    }


def validate_config(eval_root: Path, cache_path: Path) -> dict[str, Any]:
    contract = json.loads((eval_root / "contract.json").read_text(encoding="utf-8"))
    if contract.get("status") != "active":
        raise RuntimeError("evaluation contract is not active")
    if contract["editor_judge"]["judge_prompt_hash"] != JUDGER_PROMPT_HASH:
        raise RuntimeError("Judge prompt hash mismatch")
    for key, actual in (
        ("judge_model_id", JUDGER_MODEL_ID),
        ("judge_model_path", JUDGER_MODEL_PATH),
        ("judge_model_tree_sha256", JUDGER_MODEL_TREE_SHA256),
    ):
        if contract["editor_judge"].get(key) != actual:
            raise RuntimeError(f"Judge {key} mismatch")
    if not cache_path.is_file():
        raise RuntimeError(f"original-score cache is missing: {cache_path}")
    return {
        "status": "passed",
        "eval_root": str(eval_root),
        "cache_path": str(cache_path),
        "judge_model_id": JUDGER_MODEL_ID,
        "judge_model_path": JUDGER_MODEL_PATH,
        "judge_model_tree_sha256": JUDGER_MODEL_TREE_SHA256,
        "judge_prompt_hash": JUDGER_PROMPT_HASH,
        "editor_prompt_template_hash": EDITOR_PROMPT_TEMPLATE_HASH,
        "editor_urls": list(EDITOR_URLS),
        "judge_urls": list(JUDGE_URLS),
        "judge_gpu_indices": list(JUDGE_GPU_INDICES),
        "max_judge_instances_per_gpu": MAX_JUDGE_INSTANCES_PER_GPU,
        "service_lane_count": len(EDITOR_URLS),
        "expected_actor_rows": EXPECTED_TOTAL_ROWS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "validate-config",
            "prepare-actor-index",
            "run-editor",
            "run-judge",
            "run-editor-judge",
        ),
    )
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root).resolve()
    cache_path = Path(args.cache_path).resolve()
    validated = validate_config(eval_root, cache_path)
    if args.action == "validate-config":
        print(json.dumps(validated, indent=2, sort_keys=True))
        return 0
    if args.action == "prepare-actor-index":
        print(
            json.dumps(
                prepare_actor_index(eval_root),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.workers < 1 or args.retries < 1:
        raise SystemExit("workers and retries must be positive")
    if args.action == "run-editor":
        with GpuTelemetry(eval_root, "editor"):
            payload = run_editor(
                eval_root,
                workers=args.workers,
                retries=args.retries,
            )
    elif args.action == "run-judge":
        with GpuTelemetry(eval_root, "e5_judge"):
            payload = run_judge(
                eval_root,
                cache_path=cache_path,
                workers=args.workers,
                retries=args.retries,
            )
    else:
        payload = run_editor_judge(
            eval_root,
            cache_path=cache_path,
            workers=args.workers,
            retries=args.retries,
        )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
