#!/usr/bin/env python3
"""Validate and stage the three-model MR-IQA-2 Hugging Face release.

Local checkpoint paths are supplied out-of-band with ``--sources``. The
default operation is read-only. ``--materialize`` creates a new staging tree
containing only root metadata plus ``actor/``, ``judge/``, and ``editor/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


EXPORTER_VERSION = "mriqa2_hf_export_v2"
MANIFEST_SCHEMA = "mriqa2_hf_checkpoint_manifest_v2"
PROVENANCE_SCHEMA = "mriqa2_checkpoint_provenance_v1"
PRODUCTION_MANIFEST_CONTENT_SHA256 = "a54f3465b30a0f63e1ced19d7a8470e24318f74a24ca93a697d96a584486ab8b"
ROOT_TEMPLATE_FILES = (".gitattributes", "README.md")
EXPECTED_ROOT_FILES = (".gitattributes", "README.md", "LICENSE")
EXPECTED_LAYOUT = {
    "actor": ("actor", "transformers"),
    "judge": ("judge", "transformers"),
    "editor": ("editor", "diffusers"),
}
EDITOR_IDENTITY = {
    "upstream_repository_id": "black-forest-labs/FLUX.2-klein-4B",
    "upstream_revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "license": "Apache-2.0",
}
TRANSFORMERS_REQUIRED_FILES = (
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
EDITOR_REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
EXPECTED_ASSET_CONTRACTS = {
    "actor": {
        "asset_id": "actor-field-e5-step1455",
        "source_logical_id": "actor-field-e5-step1455",
        "hub_subfolder": "actor",
        "artifact_format": "transformers",
        "training_mode": "field_credit_component_kl_0.02",
        "epoch": 5,
        "global_step": 1455,
        "required_files": TRANSFORMERS_REQUIRED_FILES,
    },
    "judge": {
        "asset_id": "source-e5-judge-step725",
        "source_logical_id": "source-e5-judge-step725",
        "hub_subfolder": "judge",
        "artifact_format": "transformers",
        "training_mode": "source_e5_judge",
        "epoch": 5,
        "global_step": 725,
        "required_files": TRANSFORMERS_REQUIRED_FILES,
    },
    "editor": {
        "asset_id": "flux2-klein-4b-editor",
        "source_logical_id": "flux2-klein-4b-editor",
        "hub_subfolder": "editor",
        "artifact_format": "diffusers",
        **EDITOR_IDENTITY,
        "required_files": EDITOR_REQUIRED_FILES,
    },
}
ROOT_LICENSE_CONTRACT = {
    "source_logical_id": "flux2-klein-4b-editor",
    "source_path": "LICENSE.md",
    "published_path": "LICENSE",
    "size_bytes": 9584,
    "sha256": "ca02bc51900ab07789d1b70283329e7137f5af98f5161c23a1c81fc38a4af1fe",
}
PRIVATE_TEXT_PATTERNS = (
    re.compile(r"/(?:home|Users|mnt|scratch|private)/"),
    re.compile(r"\b10(?:\.\d{1,3}){3}\b"),
    re.compile(r"\b192\.168(?:\.\d{1,3}){2}\b"),
    re.compile(r"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"\b(?:hf|github)_[-A-Za-z0-9]{20,}\b"),
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


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


def _manifest_content_sha256(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(raw: object) -> str:
    value = str(raw)
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("./")
        or "\\" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def _positive_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _tree_digest(files: list[dict[str, Any]]) -> str:
    """Digest a selected file tree; retained for Judge/test compatibility."""

    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["name"]):
        digest.update(item["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verified_records(asset: dict[str, Any]) -> list[dict[str, Any]]:
    required = asset["required_files"]
    verified = asset["verified_files"]
    return [
        {
            "name": name,
            "size_bytes": verified[name]["size_bytes"],
            "sha256": verified[name]["sha256"],
        }
        for name in required
    ]


def validate_manifest(
    manifest: dict[str, Any], *, allow_test_payloads: bool = False
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("repository_id") != "RobinY99/MR-IQA-2":
        raise ValueError("unexpected Hugging Face repository_id")
    if manifest.get("allowed_root_files") != list(EXPECTED_ROOT_FILES):
        raise ValueError("allowed_root_files must be exactly .gitattributes, README.md, LICENSE")

    assets = manifest.get("assets")
    if not isinstance(assets, list) or len(assets) != 3:
        raise ValueError("manifest must contain exactly three assets")
    if manifest.get("expected_asset_count") != len(assets):
        raise ValueError("expected_asset_count mismatch")
    if [asset.get("role") for asset in assets if isinstance(asset, dict)] != [
        "actor",
        "judge",
        "editor",
    ]:
        raise ValueError("asset order and roles must be exactly actor, judge, editor")

    seen_ids: set[str] = set()
    seen_roles: set[str] = set()
    total_files = 0
    total_bytes = 0
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("every asset must be a JSON object")
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]*", asset_id) is None:
            raise ValueError(f"invalid asset_id: {asset_id!r}")
        if asset_id in seen_ids or asset.get("source_logical_id") != asset_id:
            raise ValueError(f"duplicate or mismatched logical asset ID: {asset_id!r}")
        seen_ids.add(asset_id)

        role = str(asset.get("role", ""))
        if role not in EXPECTED_LAYOUT or role in seen_roles:
            raise ValueError(f"unexpected or duplicate role: {role!r}")
        seen_roles.add(role)
        expected_folder, expected_format = EXPECTED_LAYOUT[role]
        if _safe_relative(asset.get("hub_subfolder")) != expected_folder:
            raise ValueError(f"{role} must publish to {expected_folder}/")
        if asset.get("artifact_format") != expected_format:
            raise ValueError(f"unexpected artifact_format for {role}")
        contract = EXPECTED_ASSET_CONTRACTS[role]
        for key, expected in contract.items():
            if key == "required_files":
                continue
            if asset.get(key) != expected:
                raise ValueError(f"production {role} {key} mismatch")

        required = asset.get("required_files")
        if not isinstance(required, list) or not required or len(required) != len(set(required)):
            raise ValueError(f"required_files must be a non-empty unique list for {asset_id}")
        normalized = [_safe_relative(name) for name in required]
        if normalized != required:
            raise ValueError(f"required_files are not normalized for {asset_id}")
        if required != list(contract["required_files"]):
            raise ValueError(f"production required_files mismatch for {role}")
        if role in {"actor", "judge"} and any("/" in name for name in required):
            raise ValueError(f"{role} runtime files must be top-level basenames")

        verified = asset.get("verified_files")
        if not isinstance(verified, dict) or set(verified) != set(required):
            raise ValueError(f"verified_files must exactly cover required_files for {asset_id}")
        for name in required:
            record = verified[name]
            if not isinstance(record, dict):
                raise ValueError(f"invalid verified file record for {asset_id}/{name}")
            _positive_int(f"{asset_id}/{name} size_bytes", record.get("size_bytes"))
            if _SHA256.fullmatch(str(record.get("sha256", ""))) is None:
                raise ValueError(f"invalid SHA-256 for {asset_id}/{name}")

        records = _verified_records(asset)
        file_count = len(records)
        byte_count = sum(item["size_bytes"] for item in records)
        if asset.get("expected_file_count") != file_count:
            raise ValueError(f"expected_file_count mismatch for {asset_id}")
        if asset.get("expected_total_bytes") != byte_count:
            raise ValueError(f"expected_total_bytes mismatch for {asset_id}")
        if _tree_digest(records) != asset.get("expected_export_tree_sha256"):
            raise ValueError(f"expected_export_tree_sha256 mismatch for {asset_id}")

        if role in {"actor", "judge"}:
            if _SHA256.fullmatch(str(asset.get("source_checkpoint_tree_sha256", ""))) is None:
                raise ValueError(f"invalid source checkpoint tree for {asset_id}")
        total_files += file_count
        total_bytes += byte_count

    if seen_roles != set(EXPECTED_LAYOUT):
        raise ValueError("roles must be exactly actor, judge, and editor")
    if manifest.get("expected_runtime_file_count") != total_files or (
        not allow_test_payloads and total_files != 38
    ):
        raise ValueError("expected_runtime_file_count mismatch")
    if manifest.get("expected_checkpoint_bytes") != total_bytes or (
        not allow_test_payloads and total_bytes != 34_177_510_861
    ):
        raise ValueError("expected_checkpoint_bytes mismatch")

    license_record = manifest.get("root_license")
    if not isinstance(license_record, dict):
        raise ValueError("missing root_license contract")
    for key in ("source_logical_id", "source_path", "published_path"):
        if license_record.get(key) != ROOT_LICENSE_CONTRACT[key]:
            raise ValueError("root LICENSE must be copied from the Editor's upstream LICENSE.md")
    if not allow_test_payloads and license_record != ROOT_LICENSE_CONTRACT:
        raise ValueError("root LICENSE must be copied from the Editor's upstream LICENSE.md")
    _positive_int("root LICENSE size_bytes", license_record.get("size_bytes"))
    if _SHA256.fullmatch(str(license_record.get("sha256", ""))) is None:
        raise ValueError("invalid root LICENSE SHA-256")
    if (
        not allow_test_payloads
        and _manifest_content_sha256(manifest) != PRODUCTION_MANIFEST_CONTENT_SHA256
    ):
        raise ValueError("production manifest content digest mismatch")
    return assets


def _scan_public_text(path: Path, relative: str) -> None:
    if path.suffix == ".safetensors" or relative in {
        "tokenizer.json",
        "tokenizer/added_tokens.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer.json",
        "tokenizer/vocab.json",
    }:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in PRIVATE_TEXT_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"private or secret-like text in {relative}: {pattern.pattern}")


def _source_file(root: Path, relative: str) -> Path:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"source entry must not be a symlink: {relative}")
    if not current.is_file():
        raise ValueError(f"missing required source file: {relative}")
    return current


def inspect_source(source: Path, asset: dict[str, Any]) -> dict[str, Any]:
    expanded = source.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise ValueError(f"source for {asset['asset_id']} must be a non-symlink directory")
    root = expanded.resolve()
    files: list[dict[str, Any]] = []
    for name in asset["required_files"]:
        path = _source_file(root, name)
        expected = asset["verified_files"][name]
        size = path.stat().st_size
        if size != expected["size_bytes"]:
            raise ValueError(f"size mismatch for {asset['asset_id']}/{name}: {size}")
        actual_sha = _sha256(path)
        if actual_sha != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {asset['asset_id']}/{name}")
        _scan_public_text(path, name)
        files.append({"name": name, "size_bytes": size, "sha256": actual_sha})
    digest = _tree_digest(files)
    if digest != asset["expected_export_tree_sha256"]:
        raise ValueError(f"selected export tree mismatch for {asset['asset_id']}")
    return {
        "asset_id": asset["asset_id"],
        "source": root,
        "files": files,
        "export_tree_sha256": digest,
        "total_bytes": sum(item["size_bytes"] for item in files),
    }


def inspect_root_license(
    sources: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    contract = manifest["root_license"]
    source_root = Path(str(sources[contract["source_logical_id"]])).expanduser()
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("Editor source for root LICENSE must be a non-symlink directory")
    path = _source_file(source_root.resolve(), contract["source_path"])
    if path.stat().st_size != contract["size_bytes"] or _sha256(path) != contract["sha256"]:
        raise ValueError("upstream Editor LICENSE.md size/hash mismatch")
    _scan_public_text(path, "LICENSE")
    return {"source": path, "size_bytes": path.stat().st_size, "sha256": contract["sha256"]}


def _copy_root_template(template: Path, destination: Path) -> None:
    if template.is_symlink() or not template.is_dir():
        raise ValueError(f"HF template directory does not exist or is a symlink: {template}")
    for name in ROOT_TEMPLATE_FILES:
        source = template / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"missing regular HF template file: {name}")
        _scan_public_text(source, name)
        shutil.copy2(source, destination / name)


def _transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        raise ValueError(f"unsupported transfer mode: {mode}")


def _expected_public_paths(manifest: dict[str, Any]) -> set[str]:
    paths = set(EXPECTED_ROOT_FILES)
    for asset in manifest["assets"]:
        folder = asset["hub_subfolder"]
        paths.update(f"{folder}/{name}" for name in asset["required_files"])
        if asset["role"] in {"actor", "judge"}:
            paths.add(f"{folder}/provenance.json")
    return paths


def _verify_materialized_paths(root: Path, manifest: dict[str, Any]) -> None:
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = _expected_public_paths(manifest)
    if actual != expected:
        raise ValueError(
            f"materialized public paths differ from the strict release contract; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def materialize(
    *,
    template: Path,
    output: Path,
    manifest: dict[str, Any],
    inspections: list[dict[str, Any]],
    license_inspection: dict[str, Any],
    transfer_mode: str,
) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new staging path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _copy_root_template(template.resolve(), temp)
        _transfer(license_inspection["source"], temp / "LICENSE", transfer_mode)
        inspection_by_id = {item["asset_id"]: item for item in inspections}
        exported_at = datetime.now(timezone.utc).isoformat()
        for asset in manifest["assets"]:
            inspection = inspection_by_id[asset["asset_id"]]
            destination = temp / asset["hub_subfolder"]
            destination.mkdir(parents=True, exist_ok=False)
            for file_info in inspection["files"]:
                name = file_info["name"]
                _transfer(inspection["source"] / name, destination / name, transfer_mode)
            if asset["role"] in {"actor", "judge"}:
                provenance = {
                    "schema_version": PROVENANCE_SCHEMA,
                    "asset_id": asset["asset_id"],
                    "source_logical_id": asset["source_logical_id"],
                    "hub_subfolder": asset["hub_subfolder"],
                    "exported_at_utc": exported_at,
                    "exporter_version": EXPORTER_VERSION,
                    "source_identity": {
                        "source_checkpoint_tree_sha256_recorded": asset[
                            "source_checkpoint_tree_sha256"
                        ],
                        "selected_export_tree_sha256": inspection["export_tree_sha256"],
                        "file_count": len(inspection["files"]),
                        "total_bytes": inspection["total_bytes"],
                    },
                    "training": {
                        "mode": asset["training_mode"],
                        "epoch": asset["epoch"],
                        "global_step": asset["global_step"],
                    },
                    "validation": asset.get("validation", {}),
                    "files": inspection["files"],
                }
                (destination / "provenance.json").write_text(
                    json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        _verify_materialized_paths(temp, manifest)
        temp.replace(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, help="Uncommitted JSON mapping logical IDs to local directories")
    parser.add_argument("--hf-template", type=Path, help="Metadata-only Hugging Face repository template")
    parser.add_argument("--output", type=Path, help="New staging directory; existing paths are rejected")
    parser.add_argument("--transfer-mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--validate-manifest-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(__file__).resolve().parents[1] / "checkpoints" / "hf_checkpoint_manifest.json"
    manifest = _json(manifest_path)
    assets = validate_manifest(manifest)
    if args.validate_manifest_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema_version": MANIFEST_SCHEMA,
                    "assets": [item["asset_id"] for item in assets],
                    "public_layout": ["actor/", "judge/", "editor/"],
                },
                indent=2,
            )
        )
        return 0
    if args.sources is None:
        raise ValueError("--sources is required unless --validate-manifest-only is used")
    sources = _json(args.sources.resolve())
    required_ids = {item["source_logical_id"] for item in assets}
    if set(sources) != required_ids:
        raise ValueError(f"source mapping IDs must be exactly: {sorted(required_ids)}")
    inspections = [
        inspect_source(Path(str(sources[asset["source_logical_id"]])), asset)
        for asset in assets
    ]
    license_inspection = inspect_root_license(sources, manifest)
    plan = {
        "valid": True,
        "materialize": bool(args.materialize),
        "asset_count": len(inspections),
        "runtime_file_count": sum(len(item["files"]) for item in inspections),
        "checkpoint_bytes": sum(item["total_bytes"] for item in inspections),
        "export_tree_sha256": {
            item["asset_id"]: item["export_tree_sha256"] for item in inspections
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
        license_inspection=license_inspection,
        transfer_mode=args.transfer_mode,
    )
    plan["output"] = str(args.output.resolve())
    plan["public_file_count"] = len(_expected_public_paths(manifest))
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
