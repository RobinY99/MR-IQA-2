#!/usr/bin/env python3
"""Validate promoted and portable Judge checkpoint manifests offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


LEGACY_SCHEMA = "vf_checkpoint_manifest_v2"
PORTABLE_SCHEMA = "mriqa2_checkpoint_provenance_v1"
LEGACY_WEIGHT_BYTES = 9_078_620_432
REQUIRED_RELEASE_FILES = (
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
LEGACY_WEIGHT_FILES = REQUIRED_RELEASE_FILES[:2]
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ManifestValidationError(ValueError):
    """The manifest or selected checkpoint payload violated its contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_export_tree_sha256(files: Iterable[dict[str, Any]]) -> str:
    """Use the exact selected-tree algorithm from the Hub exporter."""

    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["name"]):
        digest.update(str(item["name"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def full_tree_sha256(root: Path) -> str:
    """Use the exact full-tree algorithm from checkpoint promotion."""

    root = root.resolve()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ManifestValidationError(f"checkpoint tree contains no files: {root}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise ManifestValidationError(f"checkpoint file must not be a symlink: {item}")
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise ManifestValidationError(message)


def _digest(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{label} must be a positive integer")
    return value


def _safe_hub_subfolder(value: object) -> str:
    raw = str(value or "")
    path = PurePosixPath(raw)
    if (
        not raw
        or raw == "."
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(f"unsafe hub_subfolder: {raw!r}")
    return path.as_posix()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"cannot read Judge manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail("Judge manifest must be a JSON object")
    return value


def _regular_file(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.is_file():
        _fail(f"required checkpoint entry is not a regular file: {name}")
    return path


def _validate_legacy(
    manifest: dict[str, Any],
    *,
    model_path: Path,
    expected_source_tree: str,
    expected_legacy_weight_bytes: int,
    verify_files: bool,
) -> dict[str, Any]:
    identity = manifest.get("checkpoint_identity")
    artifacts = identity.get("artifact_validation") if isinstance(identity, dict) else None
    if not isinstance(identity, dict) or not isinstance(artifacts, dict):
        _fail("legacy Judge checkpoint identity is malformed")
    failures: list[str] = []
    if manifest.get("status") != "promoted" or manifest.get("usable") is not True:
        failures.append("not_promoted")
    if str(Path(str(manifest.get("checkpoint", ""))).resolve()) != str(model_path):
        failures.append("checkpoint_path")
    if identity.get("tree_sha256") != expected_source_tree:
        failures.append("source_tree_sha256")
    if artifacts.get("valid") is not True or artifacts.get("errors") not in ([], None):
        failures.append("artifact_validation")
    if artifacts.get("weight_files") != list(LEGACY_WEIGHT_FILES):
        failures.append("weight_files")
    if artifacts.get("total_weight_bytes") != expected_legacy_weight_bytes:
        failures.append("weight_bytes")
    if failures:
        _fail("legacy Judge checkpoint manifest mismatch: " + ", ".join(failures))

    if verify_files:
        actual_weight_bytes = sum(
            _regular_file(model_path, name).stat().st_size for name in LEGACY_WEIGHT_FILES
        )
        if actual_weight_bytes != expected_legacy_weight_bytes:
            _fail(
                "legacy Judge checkpoint weight bytes mismatch: "
                f"{actual_weight_bytes} != {expected_legacy_weight_bytes}"
            )
        actual_tree = full_tree_sha256(model_path)
        if actual_tree != expected_source_tree:
            _fail(f"legacy Judge checkpoint full-tree mismatch: {actual_tree}")
    return {
        "valid": True,
        "manifest_kind": "promoted_training_checkpoint",
        "schema_version": LEGACY_SCHEMA,
        "model_path": str(model_path),
        "source_checkpoint_tree_sha256": expected_source_tree,
        "selected_export_tree_sha256": None,
        "files_verified": verify_files,
    }


def _validate_portable(
    manifest: dict[str, Any],
    *,
    model_path: Path,
    expected_asset_id: str,
    expected_hub_subfolder: str,
    expected_source_tree: str,
    expected_export_tree: str,
    expected_file_count: int,
    expected_total_bytes: int | None,
    expected_required_files: Sequence[str],
    verify_files: bool,
) -> dict[str, Any]:
    if manifest.get("asset_id") != expected_asset_id:
        _fail("portable Judge asset_id mismatch")
    if manifest.get("source_logical_id") != expected_asset_id:
        _fail("portable Judge source_logical_id mismatch")
    expected_folder = _safe_hub_subfolder(expected_hub_subfolder)
    if _safe_hub_subfolder(manifest.get("hub_subfolder")) != expected_folder:
        _fail("portable Judge hub_subfolder mismatch")

    required = tuple(expected_required_files)
    if (
        expected_file_count != 10
        or len(required) != 10
        or len(set(required)) != 10
        or any(not isinstance(name, str) or Path(name).name != name for name in required)
    ):
        _fail("portable Judge release contract must contain exactly ten unique basenames")
    identity = manifest.get("source_identity")
    if not isinstance(identity, dict):
        _fail("portable Judge source_identity must be an object")
    source_tree = _digest(
        "source checkpoint tree SHA-256",
        identity.get("source_checkpoint_tree_sha256_recorded"),
    )
    selected_tree = _digest(
        "selected export-tree SHA-256", identity.get("selected_export_tree_sha256")
    )
    if source_tree != expected_source_tree:
        _fail("portable Judge source checkpoint full-tree identity mismatch")
    if selected_tree != expected_export_tree:
        _fail("portable Judge selected export-tree identity mismatch")
    if identity.get("file_count") != expected_file_count:
        _fail("portable Judge source_identity.file_count mismatch")
    declared_total = _positive_int(
        "portable Judge source_identity.total_bytes", identity.get("total_bytes")
    )
    if expected_total_bytes is not None:
        if _positive_int("expected portable Judge total bytes", expected_total_bytes) != declared_total:
            _fail("portable Judge expected total byte mismatch")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != expected_file_count:
        _fail("portable Judge files must contain exactly ten records")
    by_name: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(raw_files):
        if not isinstance(record, dict):
            _fail(f"portable Judge file record {index} must be an object")
        name = record.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            _fail(f"portable Judge file record {index} has an unsafe name")
        if name in by_name:
            _fail(f"portable Judge file record is duplicated: {name}")
        by_name[name] = record
    if set(by_name) != set(required):
        _fail("portable Judge selected filenames mismatch")

    declared: list[dict[str, Any]] = []
    for name in required:
        record = by_name[name]
        declared.append(
            {
                "name": name,
                "size_bytes": _positive_int(
                    f"portable Judge file size for {name}", record.get("size_bytes")
                ),
                "sha256": _digest(
                    f"portable Judge file SHA-256 for {name}", record.get("sha256")
                ),
            }
        )
    if sum(item["size_bytes"] for item in declared) != declared_total:
        _fail("portable Judge declared file bytes do not match source_identity.total_bytes")
    if selected_export_tree_sha256(declared) != expected_export_tree:
        _fail("portable Judge declared selected export-tree mismatch")

    if verify_files:
        actual: list[dict[str, Any]] = []
        for item in declared:
            path = _regular_file(model_path, item["name"])
            size = path.stat().st_size
            if size != item["size_bytes"]:
                _fail(f"portable Judge file size mismatch for {item['name']}")
            digest = sha256_file(path)
            if digest != item["sha256"]:
                _fail(f"portable Judge file SHA-256 mismatch for {item['name']}")
            actual.append({"name": item["name"], "size_bytes": size, "sha256": digest})
        if selected_export_tree_sha256(actual) != expected_export_tree:
            _fail("portable Judge actual selected export-tree mismatch")
    return {
        "valid": True,
        "manifest_kind": "portable_hub_export",
        "schema_version": PORTABLE_SCHEMA,
        "asset_id": expected_asset_id,
        "hub_subfolder": expected_folder,
        "model_path": str(model_path),
        "source_checkpoint_tree_sha256": source_tree,
        "selected_export_tree_sha256": expected_export_tree,
        "files_verified": verify_files,
        "file_count": expected_file_count,
        "total_bytes": declared_total,
    }


def validate_checkpoint_manifest(
    *,
    manifest_path: Path,
    model_path: Path,
    expected_asset_id: str,
    expected_hub_subfolder: str,
    expected_source_tree: str,
    expected_export_tree: str,
    expected_file_count: int = 10,
    expected_total_bytes: int | None = None,
    verify_files: bool = True,
    expected_required_files: Sequence[str] = REQUIRED_RELEASE_FILES,
    expected_legacy_weight_bytes: int = LEGACY_WEIGHT_BYTES,
) -> dict[str, Any]:
    """Validate one of the two supported manifest schemas."""

    expected_source_tree = _digest("expected source checkpoint tree SHA-256", expected_source_tree)
    expected_export_tree = _digest("expected export-tree SHA-256", expected_export_tree)
    raw_model_path = Path(model_path)
    if raw_model_path.is_symlink() or not raw_model_path.is_dir():
        _fail(f"Judge model path must be a regular directory: {raw_model_path}")
    resolved_model = raw_model_path.resolve()
    manifest = _load(Path(manifest_path))
    schema = manifest.get("schema_version")
    if schema == LEGACY_SCHEMA:
        return _validate_legacy(
            manifest,
            model_path=resolved_model,
            expected_source_tree=expected_source_tree,
            expected_legacy_weight_bytes=expected_legacy_weight_bytes,
            verify_files=verify_files,
        )
    if schema == PORTABLE_SCHEMA:
        return _validate_portable(
            manifest,
            model_path=resolved_model,
            expected_asset_id=expected_asset_id,
            expected_hub_subfolder=expected_hub_subfolder,
            expected_source_tree=expected_source_tree,
            expected_export_tree=expected_export_tree,
            expected_file_count=expected_file_count,
            expected_total_bytes=expected_total_bytes,
            expected_required_files=expected_required_files,
            verify_files=verify_files,
        )
    _fail(f"unsupported Judge checkpoint manifest schema: {schema!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-asset-id", required=True)
    parser.add_argument("--expected-hub-subfolder", default="judge/source-e5")
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-export-tree", required=True)
    parser.add_argument("--expected-file-count", type=int, default=10)
    parser.add_argument("--expected-total-bytes", type=int)
    parser.add_argument("--verify-files", choices=("0", "1"), default="1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_checkpoint_manifest(
        manifest_path=args.manifest,
        model_path=args.model_path,
        expected_asset_id=args.expected_asset_id,
        expected_hub_subfolder=args.expected_hub_subfolder,
        expected_source_tree=args.expected_source_tree,
        expected_export_tree=args.expected_export_tree,
        expected_file_count=args.expected_file_count,
        expected_total_bytes=args.expected_total_bytes,
        verify_files=args.verify_files == "1",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestValidationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
