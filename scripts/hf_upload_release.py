#!/usr/bin/env python3
"""Validate an exported MR-IQA-2 model repository and optionally upload it.

This command is dry-run by default. It never deletes remote files and requires
the explicit ``--commit-upload`` flag before contacting the Hugging Face API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from pathlib import Path, PurePosixPath
from typing import Any


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
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _checksum_entries(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})\s{2}(.+)", line)
        if not match:
            raise ValueError(f"invalid checksum line {path}:{line_number}")
        relative = match.group(2)
        if relative in seen_paths:
            raise ValueError(f"duplicate checksum path {path}:{line_number}: {relative}")
        seen_paths.add(relative)
        entries.append((match.group(1), relative))
    if not entries:
        raise ValueError(f"checksum file has no entries: {path}")
    return entries


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


def _safe_relative(raw: object) -> str:
    value = str(raw)
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe release path: {value!r}")
    return path.as_posix()


def _validate_training_asset(root: Path, asset: dict[str, Any]) -> dict[str, Any]:
    relative = _safe_relative(asset.get("hub_path"))
    manifest_relative = _safe_relative(asset.get("manifest_path"))
    path = root / relative
    metadata_path = root / manifest_relative
    if path.is_symlink() or not path.is_file() or metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError(f"missing or symlinked portable training asset: {relative}")
    if path.stat().st_size != asset.get("bytes") or _sha256(path) != asset.get("sha256"):
        raise ValueError(f"portable training-asset size/hash mismatch: {relative}")
    metadata = _json(metadata_path)
    expected_metadata = {
        "bytes": asset["bytes"],
        "row_count": asset["row_count"],
        "sample_count": asset["sample_count"],
        "sha256": asset["sha256"],
        "payload_schema": asset["schema_version"],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"portable training-asset manifest mismatch for {key}")

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
        if columns != [
            "sample_id",
            "actor_id",
            "source_image_path",
            "source_judge_rating",
            "payload_json",
        ]:
            raise ValueError(f"unexpected portable-cache columns: {columns}")
        stats = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT sample_id), MIN(source_judge_rating), "
            "MAX(source_judge_rating), AVG(source_judge_rating) FROM records"
        ).fetchone()
        if stats[0] != asset["row_count"] or stats[1] != asset["sample_count"]:
            raise ValueError("portable-cache row/sample count mismatch")
        for actual, key in zip(stats[2:], ("min", "max", "mean")):
            if not math.isclose(
                float(actual), float(asset["rating"][key]), rel_tol=0.0, abs_tol=1e-12
            ):
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
    return {"path": relative, "bytes": asset["bytes"], "sha256": asset["sha256"]}


def validate_release(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("release directory contains a symlink")
    manifest = _json(root / "checkpoint_manifest.json")
    report = _json(root / "export_report.json")
    if manifest.get("release_state") != "ready_for_upload":
        raise ValueError("checkpoint manifest is not ready_for_upload")
    if report.get("ready_for_upload") is not True:
        raise ValueError("export report is not ready_for_upload")
    if manifest.get("repository_id") != report.get("repository_id"):
        raise ValueError("repository ID differs between manifest and export report")
    training_assets = manifest.get("training_assets", [])
    if not isinstance(training_assets, list):
        raise ValueError("training_assets must be a list")
    training_checks = [_validate_training_asset(root, asset) for asset in training_assets]
    entries = _checksum_entries(root / "SHA256SUMS.full")
    expected_payload_files = manifest.get(
        "expected_release_payload_files",
        manifest.get("expected_total_files"),
    )
    if len(entries) != expected_payload_files:
        raise ValueError("SHA256SUMS.full entry count does not match manifest")
    verified_bytes = 0
    verified_files: dict[str, dict[str, Any]] = {}
    for expected, relative in entries:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe checksum path: {relative}")
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"missing release file: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch: {relative}")
        verified_bytes += path.stat().st_size
        verified_files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": actual,
        }
    expected_payload_bytes = manifest.get(
        "expected_release_payload_bytes",
        manifest.get("expected_total_bytes"),
    )
    if verified_bytes != expected_payload_bytes:
        raise ValueError("verified payload byte count does not match manifest")

    report_assets = {item["asset_id"]: item for item in report.get("assets", [])}
    for asset in manifest.get("assets", []):
        folder = str(asset["hub_subfolder"])
        if folder != ".":
            folder = _safe_relative(folder)
        selected: list[dict[str, Any]] = []
        for name in manifest["required_files"]:
            relative = name if folder == "." else f"{folder}/{name}"
            if relative not in verified_files:
                raise ValueError(f"required export file missing from complete checksums: {relative}")
            selected.append({"name": name, **verified_files[relative]})
        digest = _tree_digest(selected)
        if digest != asset["expected_export_tree_sha256"]:
            raise ValueError(f"selected export-tree mismatch for {asset['asset_id']}: {digest}")
        report_asset = report_assets.get(asset["asset_id"], {})
        if report_asset.get("export_tree_sha256") != digest:
            raise ValueError(f"export report tree mismatch for {asset['asset_id']}")
        selected_by_name = {item["name"]: item for item in selected}
        for name, expected in asset["verified_weight_sha256"].items():
            if selected_by_name[name]["sha256"] != expected:
                raise ValueError(f"known weight-shard mismatch for {asset['asset_id']}/{name}")

    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix == ".safetensors"
            or path.suffix == ".sqlite"
            or path.name == "tokenizer.json"
            or ".git" in path.parts
        ):
            continue
        if path.stat().st_size > 64 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in PRIVATE_TEXT_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"private or secret-like text in {path.relative_to(root)}")
    if not (root / ".gitattributes").is_file():
        raise ValueError("missing .gitattributes for LFS payloads")
    return {
        "valid": True,
        "repository_id": manifest["repository_id"],
        "asset_count": manifest["expected_asset_count"],
        "training_asset_count": len(training_checks),
        "payload_file_count": len(entries),
        "payload_total_bytes": verified_bytes,
        "export_trees_verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True, help="Staging directory produced by hf_export_checkpoints.py")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", default="Publish verified MR-IQA-2 checkpoint snapshots")
    parser.add_argument("--commit-upload", action="store_true", help="Perform the upload after all checks pass")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = validate_release(args.folder)
    plan["revision"] = args.revision
    plan["will_upload"] = bool(args.commit_upload)
    if not args.commit_upload:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("Dry run complete. Re-run with --commit-upload to upload without deleting remote files.")
        return 0
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError("huggingface_hub is required for --commit-upload") from exc
    api = HfApi()
    result = api.upload_folder(
        folder_path=str(args.folder.resolve()),
        repo_id=plan["repository_id"],
        repo_type="model",
        revision=args.revision,
        commit_message=args.commit_message,
        ignore_patterns=[".git/**", "**/__pycache__/**"],
    )
    print(json.dumps({**plan, "commit": str(result)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
