#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_PROMPT_SCHEMA = "e5_training_reasoning_v5"
EXPECTED_GENERATION = {
    "max_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "seed": 42,
    "return_details": True,
    "enable_thinking": False,
    "max_pixels": 196608,
    "min_pixels": 3136,
}
IDENTITY_FIELDS = (
    "backend",
    "model_id",
    "model_path",
    "model_tree_sha256",
    "prompt_schema",
    "prompt_version",
    "prompt_hash",
    "system_prompt_sha256",
    "user_prompt_sha256",
)
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"non-object JSON row at {path}:{line_number}")
            rows.append(payload)
    return rows


def validate_sources(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        sample_id = str(row.get("sample_id") or "")
        image_path = str(row.get("source_image_path") or "")
        image_sha256 = str(row.get("source_image_sha256") or "")
        if not sample_id:
            raise RuntimeError(f"source row {index} has no sample_id")
        if sample_id in seen:
            raise RuntimeError(f"duplicate source sample_id: {sample_id}")
        if not image_path:
            raise RuntimeError(f"source row {sample_id} has no source_image_path")
        if not SHA256_RE.fullmatch(image_sha256):
            raise RuntimeError(
                f"source row {sample_id} has an invalid source_image_sha256"
            )
        seen.add(sample_id)
        row["sample_id"] = sample_id
        row["source_image_path"] = image_path
        row["source_image_sha256"] = image_sha256
        result.append(row)
    if not result:
        raise RuntimeError("source manifest is empty")
    return result


def parse_ports(value: str) -> tuple[int, ...]:
    try:
        ports = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ports must be integers: {value!r}") from exc
    if not ports or len(set(ports)) != len(ports):
        raise argparse.ArgumentTypeError("ports must be a non-empty unique list")
    if any(port < 1 or port > 65535 for port in ports):
        raise argparse.ArgumentTypeError("ports must be in [1, 65535]")
    return ports


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        ).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"request failed for {url}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"non-object response from {url}")
    return result


def contract_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in IDENTITY_FIELDS if not metadata.get(field)]
    if missing:
        raise RuntimeError(f"Judge metadata is missing fields: {missing}")
    if metadata.get("prompt_schema") != EXPECTED_PROMPT_SCHEMA:
        raise RuntimeError(
            "Judge prompt schema is not cache compatible: "
            f"{metadata.get('prompt_schema')!r}"
        )
    if metadata.get("deterministic") is not True:
        raise RuntimeError("Judge metadata does not declare deterministic inference")
    if metadata.get("cache_compatible") is not True:
        raise RuntimeError("Judge metadata does not declare cache compatibility")
    generation = metadata.get("generation")
    if not isinstance(generation, dict):
        raise RuntimeError("Judge metadata has no generation contract")
    mismatches = {
        key: (expected, generation.get(key))
        for key, expected in EXPECTED_GENERATION.items()
        if generation.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Judge deterministic generation mismatch: {mismatches}")
    return {field: metadata[field] for field in IDENTITY_FIELDS}


def discover_services(
    host: str,
    ports: Sequence[int],
    *,
    timeout: float,
) -> tuple[list[str], dict[str, Any]]:
    urls = [f"http://{host}:{port}" for port in ports]
    selected_metadata: dict[str, Any] | None = None
    selected_identity: dict[str, Any] | None = None
    for url in urls:
        health = request_json(f"{url}/health", timeout=timeout)
        if health.get("ready") is not True:
            raise RuntimeError(f"Judge is not ready at {url}")
        metadata = health.get("judger")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Judge health response has no contract at {url}")
        identity = contract_identity(metadata)
        top_level = {
            "backend": health.get("backend"),
            "model_id": health.get("model_id"),
            "model_path": health.get("model_path"),
            "model_tree_sha256": health.get("model_tree_sha256"),
            "prompt_hash": health.get("prompt_hash"),
        }
        for field, value in top_level.items():
            if value != metadata.get(field):
                raise RuntimeError(
                    f"Judge health identity mismatch at {url}: {field}={value!r}"
                )
        if selected_identity is not None and identity != selected_identity:
            raise RuntimeError(f"Judge lanes have different contracts at {url}")
        selected_metadata = dict(metadata)
        selected_identity = identity
    assert selected_metadata is not None
    return urls, selected_metadata


def base_record(source: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": source["sample_id"],
        "source_image_path": source["source_image_path"],
        "input_image_sha256": source["source_image_sha256"],
        "backend": metadata["backend"],
        "model_id": metadata["model_id"],
        "model_path": metadata["model_path"],
        "model_tree_sha256": metadata["model_tree_sha256"],
        "prompt_schema": metadata["prompt_schema"],
        "prompt_version": metadata["prompt_version"],
        "prompt_hash": metadata["prompt_hash"],
        "system_prompt_sha256": metadata["system_prompt_sha256"],
        "user_prompt_sha256": metadata["user_prompt_sha256"],
        "system_prompt_hash": metadata["system_prompt_sha256"],
        "user_prompt_hash": metadata["user_prompt_sha256"],
        "prompt_mode": "judge",
        "deterministic": True,
    }


def failure_record(
    source: dict[str, Any],
    metadata: dict[str, Any],
    *,
    status: str,
    errors: list[str],
    service_url: str | None,
) -> dict[str, Any]:
    record = base_record(source, metadata)
    record.update(
        {
            "inference_status": status,
            "parse_ok": False,
            "parse_errors": errors,
            "model_rating": None,
            "service_url": service_url,
        }
    )
    return record


def successful_record(
    source: dict[str, Any],
    metadata: dict[str, Any],
    response: dict[str, Any],
    *,
    service_url: str,
) -> dict[str, Any]:
    validate_response_envelope(source, metadata, response)
    outputs = response.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError("Judge response must contain exactly one output")
    output = outputs[0]
    if not isinstance(output, dict):
        raise RuntimeError("Judge output is not an object")
    errors = output.get("errors")
    if not isinstance(errors, list):
        raise RuntimeError("Judge output parse errors are not a list")
    rating = output.get("score")
    try:
        rating_value = float(rating)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Judge rating is not numeric: {rating!r}") from exc
    accepted_range = metadata.get("score_acceptance_range")
    if not (
        isinstance(accepted_range, list)
        and len(accepted_range) == 2
        and math.isfinite(rating_value)
        and float(accepted_range[0]) <= rating_value <= float(accepted_range[1])
    ):
        raise RuntimeError(f"Judge rating is outside its contract: {rating_value!r}")
    if (
        response.get("status") != "success"
        or response.get("valid_count") != 1
        or response.get("requested_count") != 1
        or errors
    ):
        raise RuntimeError(
            "Judge response is not a single parsed deterministic score: "
            f"status={response.get('status')!r}, errors={errors!r}"
        )
    mean = response.get("mean")
    if mean is None or not math.isclose(float(mean), rating_value, abs_tol=1e-12):
        raise RuntimeError("Judge response mean does not match its parsed score")

    record = base_record(source, metadata)
    record.update(
        {
            "inference_status": "success",
            "parse_ok": True,
            "parse_errors": [],
            "model_rating": rating_value,
            "rating_text": output.get("rating_text"),
            "rating_format_ok": output.get("rating_format_ok"),
            "rating_representation": output.get("rating_representation"),
            "rating_format_warning": output.get("rating_format_warning"),
            "rating_prompt_range_ok": output.get("rating_prompt_range_ok"),
            "rating_range_warning": output.get("rating_range_warning"),
            "judge_reasons": output.get("judge_reasons"),
            "reasoning_evidence": output.get("reasoning_evidence"),
            "reasoning_solution": output.get("reasoning_solution"),
            "raw_completion": output.get("completion"),
            "finish_reason": output.get("finish_reason"),
            "prompt_token_count": output.get("prompt_token_count"),
            "completion_token_count": output.get("completion_token_count"),
            "sampling_profile": metadata.get("generation"),
            "service_url": service_url,
            "runtime_sec": response.get("runtime_sec"),
            "queue_wait_sec": response.get("queue_wait_sec"),
        }
    )
    return record


def validate_response_envelope(
    source: dict[str, Any],
    metadata: dict[str, Any],
    response: dict[str, Any],
) -> None:
    response_metadata = response.get("judger")
    if not isinstance(response_metadata, dict):
        raise RuntimeError("Judge response has no contract metadata")
    if contract_identity(response_metadata) != contract_identity(metadata):
        raise RuntimeError("Judge response contract differs from health contract")
    response_path = str(response.get("image_path") or "")
    if Path(response_path).resolve() != Path(source["source_image_path"]).resolve():
        raise RuntimeError("Judge response image path differs from the source manifest")


def score_one(
    source: dict[str, Any],
    metadata: dict[str, Any],
    urls: Sequence[str],
    *,
    primary_index: int,
    timeout: float,
) -> dict[str, Any]:
    errors: list[str] = []
    ordered_urls = [urls[(primary_index + offset) % len(urls)] for offset in range(len(urls))]
    for url in ordered_urls:
        try:
            response = request_json(
                f"{url}/score_image",
                payload={"image_path": source["source_image_path"], "repeats": 1},
                timeout=timeout,
            )
            if response.get("status") != "success":
                validate_response_envelope(source, metadata, response)
                output_errors: list[str] = []
                outputs = response.get("outputs")
                if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
                    raw_errors = outputs[0].get("errors")
                    if isinstance(raw_errors, list):
                        output_errors = [str(item) for item in raw_errors]
                return failure_record(
                    source,
                    metadata,
                    status=str(response.get("status") or "unparsed"),
                    errors=output_errors or ["unparsed_judge_output"],
                    service_url=url,
                )
            return successful_record(
                source,
                metadata,
                response,
                service_url=url,
            )
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    return failure_record(
        source,
        metadata,
        status="service_error",
        errors=errors,
        service_url=None,
    )


def load_resume_rows(
    output: Path,
    sources: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not output.exists():
        return {}
    source_by_id = {row["sample_id"]: row for row in sources}
    result: dict[str, dict[str, Any]] = {}
    expected_identity = contract_identity(metadata)
    for row in read_jsonl(output):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in result:
            raise RuntimeError(f"invalid or duplicate resume sample_id: {sample_id!r}")
        source = source_by_id.get(sample_id)
        if source is None:
            raise RuntimeError(f"resume output contains an unknown sample: {sample_id}")
        if row.get("source_image_path") != source["source_image_path"]:
            raise RuntimeError(f"resume source path mismatch for {sample_id}")
        if row.get("input_image_sha256") != source["source_image_sha256"]:
            raise RuntimeError(f"resume image identity mismatch for {sample_id}")
        if row.get("inference_status") == "success" and row.get("parse_ok") is True:
            row_identity = {field: row.get(field) for field in IDENTITY_FIELDS}
            if row_identity != expected_identity:
                raise RuntimeError(f"resume Judge contract mismatch for {sample_id}")
            if row.get("prompt_mode") != "judge" or row.get("deterministic") is not True:
                raise RuntimeError(f"resume inference contract mismatch for {sample_id}")
        result[sample_id] = row
    return result


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> int:
    sources = validate_sources(read_jsonl(args.source_manifest))
    urls, metadata = discover_services(
        args.host,
        args.ports,
        timeout=args.health_timeout,
    )
    if args.output.exists() and not args.resume:
        raise RuntimeError(f"output already exists; use --resume: {args.output}")
    existing = (
        load_resume_rows(args.output, sources, metadata)
        if args.resume
        else {}
    )
    results: dict[str, dict[str, Any]] = {
        sample_id: row
        for sample_id, row in existing.items()
        if row.get("inference_status") == "success" and row.get("parse_ok") is True
    }
    pending = [row for row in sources if row["sample_id"] not in results]

    buckets: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in urls]
    source_positions = {row["sample_id"]: index for index, row in enumerate(sources)}
    for row in pending:
        index = source_positions[row["sample_id"]]
        buckets[index % len(urls)].append((index, row))

    def score_bucket(
        bucket: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[str, dict[str, Any]]]:
        return [
            (
                row["sample_id"],
                score_one(
                    row,
                    metadata,
                    urls,
                    primary_index=index % len(urls),
                    timeout=args.request_timeout,
                ),
            )
            for index, row in bucket
        ]

    if pending:
        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            for scored in executor.map(score_bucket, buckets):
                results.update(scored)

    ordered = [results[row["sample_id"]] for row in sources]
    atomic_write_jsonl(args.output, ordered)
    failures = [row for row in ordered if row.get("inference_status") != "success" or row.get("parse_ok") is not True]
    summary = {
        "status": "complete" if not failures else "completed_with_failures",
        "source_count": len(sources),
        "success_count": len(sources) - len(failures),
        "failure_count": len(failures),
        "resumed_success_count": len(sources) - len(pending),
        "output": str(args.output.resolve()),
        "judge": contract_identity(metadata),
        "services": urls,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a source-image JSONL manifest with running frozen Judge services."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--ports",
        type=parse_ports,
        default=parse_ports("8204,8205,8206,8207"),
        help="comma-separated Judge service ports",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.health_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except Exception as exc:
        print(f"score_manifest: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
