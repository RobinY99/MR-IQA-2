#!/usr/bin/env python3
"""Validate and stage the MR-IQA-2 Hugging Face checkpoint release.

The committed manifest contains logical source IDs only. A maintainer supplies
the local path mapping out-of-band with ``--sources``. The default operation is
read-only; ``--materialize`` is required before any staging directory is made.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


EXPORTER_VERSION = "mriqa2_hf_export_v1"
MANIFEST_SCHEMA = "mriqa2_hf_checkpoint_manifest_v1"
PRIVATE_TEXT_PATTERNS = (
    re.compile(r"/(?:home|Users|mnt|scratch|private)/"),
    re.compile(r"\b10(?:\.\d{1,3}){3}\b"),
    re.compile(r"\b192\.168(?:\.\d{1,3}){2}\b"),
    re.compile(r"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_subfolder(raw: object) -> str:
    value = str(raw)
    if value == ".":
        return value
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe Hub subfolder: {value!r}")
    return path.as_posix()


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    required = manifest.get("required_files")
    if not isinstance(required, list) or len(required) != len(set(required)) or not required:
        raise ValueError("required_files must be a non-empty unique list")
    for name in required:
        if not isinstance(name, str) or Path(name).name != name or name.startswith("."):
            raise ValueError(f"required file must be a basename: {name!r}")

    assets = manifest.get("assets")
    if not isinstance(assets, list) or len(assets) != manifest.get("expected_asset_count"):
        raise ValueError("asset count does not match expected_asset_count")
    ids: set[str] = set()
    folders: set[str] = set()
    total_files = 0
    total_bytes = 0
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("every asset must be an object")
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", asset_id):
            raise ValueError(f"invalid asset_id: {asset_id!r}")
        if asset_id in ids:
            raise ValueError(f"duplicate asset_id: {asset_id}")
        ids.add(asset_id)
        folder = _safe_subfolder(asset.get("hub_subfolder"))
        if folder in folders:
            raise ValueError(f"duplicate Hub subfolder: {folder}")
        folders.add(folder)
        for key in ("source_checkpoint_tree_sha256", "expected_export_tree_sha256"):
            digest = str(asset.get(key, ""))
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid {key} for {asset_id}")
        if asset.get("expected_file_count") != len(required):
            raise ValueError(f"file count contract mismatch for {asset_id}")
        weight_hashes = asset.get("verified_weight_sha256")
        if not isinstance(weight_hashes, dict) or not weight_hashes:
            raise ValueError(f"missing verified weight hashes for {asset_id}")
        for name, value in weight_hashes.items():
            if name not in required or not re.fullmatch(r"[0-9a-f]{64}", str(value)):
                raise ValueError(f"invalid verified weight hash for {asset_id}/{name}")
        total_files += int(asset["expected_file_count"])
        total_bytes += int(asset["expected_total_bytes"])

    if manifest.get("default_asset_id") not in ids:
        raise ValueError("default_asset_id does not name an asset")
    if manifest.get("expected_total_files") != total_files:
        raise ValueError("expected_total_files mismatch")
    if manifest.get("expected_total_bytes") != total_bytes:
        raise ValueError("expected_total_bytes mismatch")

    training_assets = manifest.get("training_assets", [])
    if not isinstance(training_assets, list):
        raise ValueError("training_assets must be a list")
    training_bytes = 0
    for asset in training_assets:
        if not isinstance(asset, dict) or asset.get("release_state") != "ready":
            raise ValueError("every published training asset must be ready")
        _safe_subfolder(asset.get("hub_path"))
        _safe_subfolder(asset.get("manifest_path"))
        if not re.fullmatch(r"[0-9a-f]{64}", str(asset.get("sha256", ""))):
            raise ValueError(f"invalid training-asset SHA-256: {asset.get('asset_id')}")
        for key in ("bytes", "row_count", "sample_count"):
            if not isinstance(asset.get(key), int) or asset[key] <= 0:
                raise ValueError(f"invalid {key} for training asset {asset.get('asset_id')}")
        training_bytes += asset["bytes"]
    if training_assets:
        if manifest.get("expected_release_payload_files") != total_files + len(training_assets):
            raise ValueError("expected_release_payload_files mismatch")
        if manifest.get("expected_release_payload_bytes") != total_bytes + training_bytes:
            raise ValueError("expected_release_payload_bytes mismatch")
    return assets


def inspect_training_asset(root: Path, asset: dict[str, Any]) -> dict[str, Any]:
    relative = _safe_subfolder(asset["hub_path"])
    manifest_relative = _safe_subfolder(asset["manifest_path"])
    path = root / relative
    metadata_path = root / manifest_relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked training asset: {relative}")
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError(f"missing or symlinked training-asset manifest: {manifest_relative}")
    if path.stat().st_size != asset["bytes"]:
        raise ValueError(f"training-asset byte mismatch: {relative}")
    actual_sha = _sha256(path)
    if actual_sha != asset["sha256"]:
        raise ValueError(f"training-asset SHA-256 mismatch: {relative}")
    metadata = _json(metadata_path)
    for key, expected in (
        ("bytes", asset["bytes"]),
        ("row_count", asset["row_count"]),
        ("sample_count", asset["sample_count"]),
        ("sha256", asset["sha256"]),
        ("payload_schema", asset["schema_version"]),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"training-asset manifest mismatch for {key}")

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("portable cache failed SQLite quick_check")
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        if tables != [("records",)]:
            raise ValueError(f"unexpected portable-cache tables: {tables}")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(records)")]
        expected_columns = [
            "sample_id",
            "actor_id",
            "source_image_path",
            "source_judge_rating",
            "payload_json",
        ]
        if columns != expected_columns:
            raise ValueError(f"unexpected portable-cache columns: {columns}")
        stats = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT sample_id), MIN(source_judge_rating), "
            "MAX(source_judge_rating), AVG(source_judge_rating) FROM records"
        ).fetchone()
        if stats[0] != asset["row_count"] or stats[1] != asset["sample_count"]:
            raise ValueError("portable-cache row/sample count mismatch")
        rating = asset["rating"]
        for actual, key in zip(stats[2:], ("min", "max", "mean")):
            if not math.isclose(float(actual), float(rating[key]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"portable-cache rating {key} mismatch")
        actor_ids = [row[0] for row in connection.execute("SELECT DISTINCT actor_id FROM records")]
        if actor_ids != [asset["actor_id"]]:
            raise ValueError(f"portable-cache actor IDs mismatch: {actor_ids}")
        for sample_id, actor_id, source_path, payload_raw in connection.execute(
            "SELECT sample_id, actor_id, source_image_path, payload_json FROM records"
        ):
            source = PurePosixPath(source_path)
            if source.is_absolute() or ".." in source.parts:
                raise ValueError(f"non-portable image path for sample {sample_id}")
            for pattern in PRIVATE_TEXT_PATTERNS:
                if pattern.search(source_path) or pattern.search(payload_raw):
                    raise ValueError(f"private text in portable cache for sample {sample_id}")
            payload = json.loads(payload_raw)
            judge = payload.get("source_judge") or {}
            if (
                actor_id != asset["actor_id"]
                or payload.get("schema_version") != asset["schema_version"]
                or judge.get("model_id") != asset["judge_id"]
                or judge.get("model_uri") != asset["logical_model_uri"]
            ):
                raise ValueError(f"portable-cache logical contract mismatch for sample {sample_id}")
    finally:
        connection.close()
    return {
        "asset_id": asset["asset_id"],
        "path": relative,
        "sha256": actual_sha,
        "bytes": asset["bytes"],
        "row_count": asset["row_count"],
        "sample_count": asset["sample_count"],
    }


def _scan_public_text(path: Path) -> None:
    # Tokenizer vocabularies contain arbitrary corpus fragments (including
    # path-like tokens), so scanning the vocabulary creates false positives.
    # Privacy-bearing tokenizer metadata remains covered by tokenizer_config.
    if path.suffix == ".safetensors" or path.name == "tokenizer.json":
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in PRIVATE_TEXT_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"private or secret-like text in {path.name}: {pattern.pattern}")


def _tree_digest(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["name"]):
        digest.update(item["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def inspect_source(source: Path, asset: dict[str, Any], required: list[str]) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"source for {asset['asset_id']} is not a directory")
    children = list(source.iterdir())
    by_name = {child.name: child for child in children}
    missing = sorted(set(required) - set(by_name))
    if missing:
        raise ValueError(f"missing required files for {asset['asset_id']}: {missing}")
    for name in required:
        path = by_name[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required export entry is not a regular non-symlink file: {asset['asset_id']}/{name}")
    ignored_entries = [
        {
            "name": child.name,
            "kind": "symlink" if child.is_symlink() else "directory" if child.is_dir() else "file",
        }
        for child in sorted(children, key=lambda value: value.name)
        if child.name not in set(required)
    ]

    files: list[dict[str, Any]] = []
    for name in required:
        path = source / name
        if path.stat().st_size <= 0:
            raise ValueError(f"empty source file: {asset['asset_id']}/{name}")
        _scan_public_text(path)
        files.append({"name": name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})

    total_bytes = sum(item["size_bytes"] for item in files)
    if total_bytes != asset["expected_total_bytes"]:
        raise ValueError(
            f"byte count mismatch for {asset['asset_id']}: {total_bytes} != {asset['expected_total_bytes']}"
        )
    digest = _tree_digest(files)
    if digest != asset["expected_export_tree_sha256"]:
        raise ValueError(f"export-tree SHA-256 mismatch for {asset['asset_id']}: {digest}")
    files_by_name = {item["name"]: item for item in files}
    for name, expected in asset["verified_weight_sha256"].items():
        if files_by_name[name]["sha256"] != expected:
            raise ValueError(f"weight SHA-256 mismatch for {asset['asset_id']}/{name}")
    return {
        "asset_id": asset["asset_id"],
        "source": source,
        "source_checkpoint_tree_sha256_recorded": asset["source_checkpoint_tree_sha256"],
        "export_tree_sha256": digest,
        "total_bytes": total_bytes,
        "files": files,
        "ignored_source_entries": ignored_entries,
    }


def _copy_template(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"HF template directory does not exist: {source}")
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        if ".git" in relative.parts or item.is_symlink():
            continue
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:  # guarded by argparse
        raise ValueError(f"unsupported transfer mode: {mode}")


def materialize(
    *,
    template: Path,
    output: Path,
    manifest: dict[str, Any],
    inspections: list[dict[str, Any]],
    transfer_mode: str,
) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new staging path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _copy_template(template.resolve(), temp)
        manifest_out = copy.deepcopy(manifest)
        manifest_out["release_state"] = "ready_for_upload"
        exported_at = datetime.now(timezone.utc).isoformat()
        checksum_lines: list[str] = []
        training_inspections = [
            inspect_training_asset(temp, asset) for asset in manifest_out.get("training_assets", [])
        ]
        checksum_lines.extend(
            f"{item['sha256']}  {item['path']}" for item in training_inspections
        )
        inspection_by_id = {item["asset_id"]: item for item in inspections}
        for asset in manifest_out["assets"]:
            inspection = inspection_by_id[asset["asset_id"]]
            folder = _safe_subfolder(asset["hub_subfolder"])
            destination = temp if folder == "." else temp / folder
            destination.mkdir(parents=True, exist_ok=True)
            for file_info in inspection["files"]:
                name = file_info["name"]
                _transfer(inspection["source"] / name, destination / name, transfer_mode)
                relative = name if folder == "." else f"{folder}/{name}"
                checksum_lines.append(f"{file_info['sha256']}  {relative}")
            provenance = {
                "schema_version": "mriqa2_checkpoint_provenance_v1",
                "asset_id": asset["asset_id"],
                "source_logical_id": asset["source_logical_id"],
                "hub_subfolder": folder,
                "exported_at_utc": exported_at,
                "exporter_version": EXPORTER_VERSION,
                "source_identity": {
                    "source_checkpoint_tree_sha256_recorded": inspection[
                        "source_checkpoint_tree_sha256_recorded"
                    ],
                    "selected_export_tree_sha256": inspection["export_tree_sha256"],
                    "file_count": len(inspection["files"]),
                    "total_bytes": inspection["total_bytes"],
                    "ignored_source_entries": inspection["ignored_source_entries"],
                },
                "training": {
                    "mode": asset["training_mode"],
                    "epoch": asset["epoch"],
                    "global_step": asset["global_step"],
                    "base_model": "Qwen/Qwen3.5-4B",
                },
                "selection": {
                    "status": asset["selection"],
                    "recommended": asset["recommended"],
                    "research_status": asset["research_status"],
                },
                "validation": asset.get("validation", {}),
                "files": inspection["files"],
            }
            provenance_name = "default_checkpoint_provenance.json" if folder == "." else "provenance.json"
            (destination / provenance_name).write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            asset["release_verification"] = {
                "exported_at_utc": exported_at,
                "export_tree_sha256_verified": True,
                "file_count_verified": True,
                "byte_count_verified": True,
                "known_weight_sha256_verified": True,
            }

        (temp / "checkpoint_manifest.json").write_text(
            json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temp / "SHA256SUMS.full").write_text("\n".join(sorted(checksum_lines)) + "\n", encoding="utf-8")
        report = {
            "schema_version": "mriqa2_hf_export_report_v1",
            "exporter_version": EXPORTER_VERSION,
            "exported_at_utc": exported_at,
            "repository_id": manifest["repository_id"],
            "ready_for_upload": True,
            "asset_count": len(inspections),
            "model_file_count": sum(len(item["files"]) for item in inspections),
            "model_total_bytes": sum(item["total_bytes"] for item in inspections),
            "training_asset_count": len(training_inspections),
            "release_payload_file_count": sum(len(item["files"]) for item in inspections)
            + len(training_inspections),
            "release_payload_total_bytes": sum(item["total_bytes"] for item in inspections)
            + sum(item["bytes"] for item in training_inspections),
            "assets": [
                {
                    "asset_id": item["asset_id"],
                    "source_checkpoint_tree_sha256_recorded": item[
                        "source_checkpoint_tree_sha256_recorded"
                    ],
                    "export_tree_sha256": item["export_tree_sha256"],
                    "file_count": len(item["files"]),
                    "total_bytes": item["total_bytes"],
                    "ignored_source_entries": item["ignored_source_entries"],
                }
                for item in inspections
            ],
            "training_assets": training_inspections,
        }
        (temp / "export_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp.replace(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Canonical checkpoint_manifest.json")
    parser.add_argument("--sources", type=Path, help="Uncommitted JSON mapping logical IDs to local directories")
    parser.add_argument("--hf-template", type=Path, help="Metadata-only Hugging Face repository template")
    parser.add_argument("--output", type=Path, help="New staging directory; existing paths are rejected")
    parser.add_argument("--transfer-mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--materialize", action="store_true", help="Create the staging directory after validation")
    parser.add_argument(
        "--validate-manifest-only",
        action="store_true",
        help="Validate public metadata without requiring local checkpoints",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _json(args.manifest.resolve())
    assets = validate_manifest(manifest)
    if args.validate_manifest_only:
        print(json.dumps({"valid": True, "assets": [item["asset_id"] for item in assets]}, indent=2))
        return 0
    if args.sources is None:
        raise ValueError("--sources is required unless --validate-manifest-only is used")
    sources = _json(args.sources.resolve())
    required_ids = {item["source_logical_id"] for item in assets}
    if set(sources) != required_ids:
        raise ValueError(f"source mapping IDs must be exactly: {sorted(required_ids)}")
    inspections = [
        inspect_source(Path(str(sources[asset["source_logical_id"]])), asset, manifest["required_files"])
        for asset in assets
    ]
    plan = {
        "valid": True,
        "materialize": bool(args.materialize),
        "asset_count": len(inspections),
        "model_file_count": sum(len(item["files"]) for item in inspections),
        "model_total_bytes": sum(item["total_bytes"] for item in inspections),
        "export_tree_sha256": {
            item["asset_id"]: item["export_tree_sha256"] for item in inspections
        },
        "ignored_source_entries": {
            item["asset_id"]: item["ignored_source_entries"] for item in inspections
        },
    }
    if not args.materialize:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.hf_template is None or args.output is None:
        raise ValueError("--hf-template and --output are required with --materialize")
    materialize(
        template=args.hf_template,
        output=args.output,
        manifest=manifest,
        inspections=inspections,
        transfer_mode=args.transfer_mode,
    )
    plan["output"] = str(args.output.resolve())
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
