#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "vf_checkpoint_manifest_v2"
QUARANTINED = "quarantined"
TECHNICALLY_VALID = "technically_valid"
PROMOTED = "promoted"

DEFAULT_VALIDATION_THRESHOLDS: dict[str, float] = {
    "num_total": 200,
    "num_shards": 4,
    "actor_format_success_rate": 0.95,
    "rating_parse_success_rate": 0.95,
    "plcc": 0.65,
    "srcc": 0.65,
    "low_target_edit_request_rate": 0.05,
    "num_low_target_edits": 1,
    "unique_completion_count": 20,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    root = Path(path).resolve()
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"checkpoint tree contains no files: {root}")
    for item in files:
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _contained_checkpoint(run_dir: Path, checkpoint: Path) -> tuple[Path, Path]:
    run_root = Path(run_dir).resolve()
    train_root = (run_root / "train").resolve()
    candidate = Path(checkpoint).resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {candidate}")
    try:
        candidate.relative_to(train_root)
    except ValueError as exc:
        raise ValueError(f"checkpoint escapes current run train directory: {candidate}") from exc
    return run_root, candidate


def discover_single_checkpoint(run_dir: Path) -> Path:
    run_root = Path(run_dir).resolve()
    train_root = (run_root / "train").resolve()
    if not train_root.is_dir():
        raise FileNotFoundError(f"run train directory does not exist: {train_root}")
    candidates = sorted(
        item.resolve()
        for item in train_root.rglob("checkpoint-*")
        if item.is_dir()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one checkpoint inside {train_root}, found {len(candidates)}: "
            + ", ".join(str(path) for path in candidates)
        )
    _contained_checkpoint(run_root, candidates[0])
    return candidates[0]


def _indexed_weight_files(checkpoint: Path) -> list[Path]:
    indexed: list[Path] = []
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint / index_name
        if not index_path.is_file():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            names = sorted(set((payload.get("weight_map") or {}).values()))
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise RuntimeError(f"invalid checkpoint weight index {index_path}: {exc}") from exc
        indexed.extend(checkpoint / str(name) for name in names)
    return indexed


def validate_checkpoint_artifacts(checkpoint: Path) -> dict[str, Any]:
    root = Path(checkpoint).resolve()
    errors: list[str] = []
    required = ["config.json", "tokenizer_config.json"]
    for name in required:
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing_or_empty:{name}")

    processor_candidates = [root / "preprocessor_config.json", root / "processor_config.json"]
    if not any(path.is_file() and path.stat().st_size > 0 for path in processor_candidates):
        errors.append("missing_or_empty:processor_config")

    direct_weights = sorted(root.glob("*.safetensors")) + sorted(root.glob("pytorch_model*.bin"))
    indexed_weights = _indexed_weight_files(root)
    weight_files = sorted(set(direct_weights + indexed_weights))
    if not weight_files:
        errors.append("missing:model_weights")
    for path in weight_files:
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing_or_empty:{path.name}")

    return {
        "valid": not errors,
        "errors": errors,
        "weight_files": [path.name for path in weight_files if path.is_file()],
        "total_weight_bytes": sum(path.stat().st_size for path in weight_files if path.is_file()),
    }


def build_checkpoint_manifest(
    run_dir: Path,
    checkpoint: Path,
    *,
    run_id: str | None = None,
    parent_checkpoint: Path | None = None,
    parent_kind: str = "approved_initial",
    parent_manifest: Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_root, candidate = _contained_checkpoint(run_dir, checkpoint)
    artifacts = validate_checkpoint_artifacts(candidate)
    try:
        tree_sha256 = sha256_tree(candidate)
    except RuntimeError:
        tree_sha256 = None
    identity = {
        "tree_sha256": tree_sha256,
        "artifact_validation": artifacts,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or run_root.name,
        "run_dir": str(run_root),
        "checkpoint": str(candidate),
        "parent": {
            "kind": str(parent_kind),
            "checkpoint": str(Path(parent_checkpoint).resolve()) if parent_checkpoint is not None else None,
            "manifest": str(Path(parent_manifest).resolve()) if parent_manifest is not None else None,
        },
        "provenance": dict(provenance or {}),
        "checkpoint_identity": identity,
        "status": QUARANTINED,
        "usable": False,
        "quarantine_reason": "trainer exit, artifact, trajectory, and validation evidence are incomplete",
        "created_at": _now(),
        "history": [{"status": QUARANTINED, "timestamp": _now()}],
    }


def _validate_trajectory_summary(
    summary: dict[str, Any], *, expected_rank_shards: int = 4
) -> list[str]:
    errors: list[str] = []
    num_rows = int(summary.get("num_rows") or 0)
    if num_rows <= 0:
        errors.append("num_rows")
    if int(summary.get("num_rank_shards") or 0) != int(expected_rank_shards):
        errors.append("num_rank_shards")
    if int(summary.get("unique_trajectory_ids") or 0) != num_rows:
        errors.append("unique_trajectory_ids")
    if float(summary.get("credit_integrity_rate") or 0.0) != 1.0:
        errors.append("credit_integrity_rate")
    if int(summary.get("non_finite_count") or 0) != 0:
        errors.append("non_finite_count")
    return errors


def transition_to_technically_valid(
    manifest: dict[str, Any],
    *,
    trainer_exit_code: int,
    trajectory_summary: dict[str, Any],
    wandb_url: str,
    expected_rank_shards: int = 4,
) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint manifest schema")
    if manifest.get("status") != QUARANTINED:
        raise RuntimeError(f"technical validation requires quarantined status, got {manifest.get('status')}")
    if int(trainer_exit_code) != 0:
        raise RuntimeError(f"trainer exit code is not zero: {trainer_exit_code}")

    run_root, checkpoint = _contained_checkpoint(Path(manifest["run_dir"]), Path(manifest["checkpoint"]))
    artifacts = validate_checkpoint_artifacts(checkpoint)
    if not artifacts["valid"]:
        raise RuntimeError("checkpoint artifacts are incomplete: " + ", ".join(artifacts["errors"]))
    trajectory_errors = _validate_trajectory_summary(
        trajectory_summary,
        expected_rank_shards=expected_rank_shards,
    )
    if trajectory_errors:
        raise RuntimeError("trajectory evidence is incomplete: " + ", ".join(trajectory_errors))
    provenance = manifest.get("provenance") or {}
    missing_provenance = [
        name for name in ("data_sha256", "prompt_hash", "code_sha256") if not provenance.get(name)
    ]
    if missing_provenance:
        raise RuntimeError("checkpoint provenance is incomplete: " + ", ".join(missing_provenance))
    if not str(wandb_url).startswith(("https://", "http://")):
        raise RuntimeError("W&B run URL is missing or invalid")

    result = copy.deepcopy(manifest)
    result["run_dir"] = str(run_root)
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_identity"] = {
        "tree_sha256": sha256_tree(checkpoint),
        "artifact_validation": artifacts,
    }
    result["technical_validation"] = {
        "trainer_exit_code": 0,
        "trajectory_summary": copy.deepcopy(trajectory_summary),
        "wandb_url": str(wandb_url),
        "validated_at": _now(),
    }
    result["status"] = TECHNICALLY_VALID
    result["usable"] = False
    result["quarantine_reason"] = "validation and explicit promotion are incomplete"
    result.setdefault("history", []).append({"status": TECHNICALLY_VALID, "timestamp": _now()})
    return result


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate_validation_gate(
    summary: dict[str, Any], thresholds: dict[str, float] | None = None
) -> dict[str, Any]:
    limits = dict(DEFAULT_VALIDATION_THRESHOLDS)
    if summary.get("actor_schema") == "reasons_rating":
        limits.pop("low_target_edit_request_rate", None)
        limits.pop("num_low_target_edits", None)
        limits["reasons_nonempty_rate"] = 0.95
    if thresholds:
        limits.update({str(key): float(value) for key, value in thresholds.items()})
    failures: list[str] = []
    exact = {"num_shards"}
    for name, threshold in limits.items():
        value = _finite_number(summary.get(name))
        if value is None:
            failures.append(f"{name}:non_finite_or_missing")
        elif name in exact and value != threshold:
            failures.append(f"{name}:{value}!={threshold}")
        elif name not in exact and value < threshold:
            failures.append(f"{name}:{value}<{threshold}")
    return {"passed": not failures, "failures": failures, "thresholds": limits}


def evaluate_observational_validation_gate(
    summary: dict[str, Any],
    *,
    expected_shards: int = 8,
    expected_total: int = 200,
    expected_actor_schema: str = "reasons_rating",
) -> dict[str, Any]:
    failures: list[str] = []
    exact = {
        "num_total": expected_total,
        "num_shards": expected_shards,
        "num_missing_or_bad_gold": 0,
        "batch_generate_exception_count": 0,
        "singleton_generate_exception_count": 0,
    }
    for name, expected in exact.items():
        value = summary.get(name)
        try:
            number = int(value)
        except (TypeError, ValueError):
            failures.append(f"{name}:missing_or_invalid")
            continue
        if number != expected:
            failures.append(f"{name}:{number}!={expected}")
    actor_schema = summary.get("actor_schema")
    if actor_schema != expected_actor_schema:
        failures.append(
            f"actor_schema:{actor_schema!r}!={expected_actor_schema!r}"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "policy": "comparison_observational",
        "requirements": {**exact, "actor_schema": expected_actor_schema},
    }


def transition_to_promoted(
    manifest: dict[str, Any],
    *,
    validation_summary: dict[str, Any],
    approval: str,
    thresholds: dict[str, float] | None = None,
    validation_policy: str = "quality_gate",
    expected_actor_schema: str = "reasons_rating",
) -> dict[str, Any]:
    if manifest.get("status") != TECHNICALLY_VALID:
        raise RuntimeError(f"promotion requires technically_valid status, got {manifest.get('status')}")
    if not str(approval).strip():
        raise RuntimeError("promotion requires an explicit approval identity")
    if validation_policy == "comparison_observational":
        expected_shards = int((thresholds or {}).get("num_shards", 8))
        gate = evaluate_observational_validation_gate(
            validation_summary,
            expected_shards=expected_shards,
            expected_actor_schema=expected_actor_schema,
        )
    elif validation_policy == "quality_gate":
        gate = evaluate_validation_gate(validation_summary, thresholds=thresholds)
        gate["policy"] = "quality_gate"
    else:
        raise ValueError(f"unsupported validation policy: {validation_policy}")
    if not gate["passed"]:
        raise RuntimeError("validation gate failed: " + ", ".join(gate["failures"]))

    result = copy.deepcopy(manifest)
    result["validation"] = {"summary": copy.deepcopy(validation_summary), **gate}
    result["promotion"] = {"approval": str(approval), "promoted_at": _now()}
    result["status"] = PROMOTED
    result["usable"] = True
    result["quarantine_reason"] = None
    result.setdefault("history", []).append({"status": PROMOTED, "timestamp": _now()})
    return result


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)


def load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint manifest")
    return payload


def write_checkpoint_manifest(run_dir: Path, checkpoint: Path, output: Path, **kwargs: Any) -> None:
    write_manifest(output, build_checkpoint_manifest(run_dir, checkpoint, **kwargs))


def resolve_promoted_checkpoint(manifest_path: Path) -> Path:
    manifest = load_checkpoint_manifest(manifest_path)
    if manifest.get("status") != PROMOTED or manifest.get("usable") is not True:
        raise RuntimeError(f"checkpoint is not promoted: {manifest.get('status')}")
    _, checkpoint = _contained_checkpoint(Path(manifest["run_dir"]), Path(manifest["checkpoint"]))
    artifacts = validate_checkpoint_artifacts(checkpoint)
    if not artifacts["valid"]:
        raise RuntimeError("promoted checkpoint artifacts changed: " + ", ".join(artifacts["errors"]))
    expected = str((manifest.get("checkpoint_identity") or {}).get("tree_sha256") or "")
    actual = sha256_tree(checkpoint)
    if not expected or actual != expected:
        raise RuntimeError("promoted checkpoint identity changed")
    return checkpoint


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover")
    discover.add_argument("--run-dir", type=Path, required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--run-dir", type=Path, required=True)
    create.add_argument("--checkpoint", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--run-id", required=True)
    create.add_argument("--parent-checkpoint", type=Path, required=True)
    create.add_argument("--parent-kind", default="approved_initial")
    create.add_argument("--parent-manifest", type=Path)
    create.add_argument("--data-sha256", required=True)
    create.add_argument("--prompt-hash", required=True)
    create.add_argument("--code-sha256", required=True)

    technical = subparsers.add_parser("technical-validate")
    technical.add_argument("--manifest", type=Path, required=True)
    technical.add_argument("--trainer-exit-code", type=int, required=True)
    technical.add_argument("--trajectory-summary", type=Path, required=True)
    technical.add_argument("--wandb-url", required=True)
    technical.add_argument("--expected-rank-shards", type=int, default=4)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--manifest", type=Path, required=True)
    promote.add_argument("--validation", type=Path, required=True)
    promote.add_argument("--approval", required=True)
    promote.add_argument("--expected-validation-shards", type=int, default=4)
    promote.add_argument("--expected-actor-schema", default="reasons_rating")
    promote.add_argument(
        "--validation-policy",
        choices=("quality_gate", "comparison_observational"),
        default="quality_gate",
    )

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "discover":
        print(discover_single_checkpoint(args.run_dir))
        return 0
    if args.command == "create":
        manifest = build_checkpoint_manifest(
            args.run_dir,
            args.checkpoint,
            run_id=args.run_id,
            parent_checkpoint=args.parent_checkpoint,
            parent_kind=args.parent_kind,
            parent_manifest=args.parent_manifest,
            provenance={
                "data_sha256": args.data_sha256,
                "prompt_hash": args.prompt_hash,
                "code_sha256": args.code_sha256,
            },
        )
        write_manifest(args.output, manifest)
        print(json.dumps({"status": manifest["status"], "manifest": str(args.output)}, ensure_ascii=False))
        return 0
    if args.command == "technical-validate":
        manifest = load_checkpoint_manifest(args.manifest)
        updated = transition_to_technically_valid(
            manifest,
            trainer_exit_code=args.trainer_exit_code,
            trajectory_summary=_load_json(args.trajectory_summary),
            wandb_url=args.wandb_url,
            expected_rank_shards=args.expected_rank_shards,
        )
        write_manifest(args.manifest, updated)
        print(json.dumps({"status": updated["status"], "manifest": str(args.manifest)}, ensure_ascii=False))
        return 0
    if args.command == "promote":
        manifest = load_checkpoint_manifest(args.manifest)
        validation = _load_json(args.validation)
        summary = validation.get("summary") if isinstance(validation.get("summary"), dict) else validation
        updated = transition_to_promoted(
            manifest,
            validation_summary=summary,
            approval=args.approval,
            thresholds={"num_shards": args.expected_validation_shards},
            validation_policy=args.validation_policy,
            expected_actor_schema=args.expected_actor_schema,
        )
        write_manifest(args.manifest, updated)
        print(json.dumps({"status": updated["status"], "checkpoint": updated["checkpoint"]}, ensure_ascii=False))
        return 0
    if args.command == "resolve":
        print(resolve_promoted_checkpoint(args.manifest))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
