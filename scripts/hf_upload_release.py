#!/usr/bin/env python3
"""Validate and atomically replace the MR-IQA-2 Hugging Face model tree.

The command is dry-run by default and does not contact Hugging Face. Publishing
requires both ``--commit-upload`` and ``--replace-remote``. The remote update is
one parent-pinned commit containing explicit additions and deletions, so stale
completion checkpoints, caches, and root-level weights cannot survive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_SCHEMA = "mriqa2_hf_checkpoint_manifest_v2"
PROVENANCE_SCHEMA = "mriqa2_checkpoint_provenance_v1"
PRODUCTION_MANIFEST_CONTENT_SHA256 = "a54f3465b30a0f63e1ced19d7a8470e24318f74a24ca93a697d96a584486ab8b"
COMMIT_MESSAGE = "Publish Actor, Judge, and Editor"
EXPECTED_ROOT_FILES = (".gitattributes", "README.md", "LICENSE")
EXPECTED_LAYOUT = {"actor": "actor", "judge": "judge", "editor": "editor"}
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
        "upstream_repository_id": "black-forest-labs/FLUX.2-klein-4B",
        "upstream_revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "license": "Apache-2.0",
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
        raise ValueError(f"expected JSON object: {path}")
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
    if (
        not value
        or value.startswith("./")
        or "\\" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe release path: {value!r}")
    return path.as_posix()


def _scan_public_text(path: Path, relative: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in PRIVATE_TEXT_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"private or secret-like text in {relative}: {pattern.pattern}")


def _manifest_assets(
    manifest: dict[str, Any], *, allow_test_payloads: bool = False
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("repository_id") != "RobinY99/MR-IQA-2":
        raise ValueError("unexpected repository_id")
    if manifest.get("allowed_root_files") != list(EXPECTED_ROOT_FILES):
        raise ValueError("manifest root-file contract mismatch")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or len(assets) != 3:
        raise ValueError("manifest must contain exactly three assets")
    if manifest.get("expected_asset_count") != 3:
        raise ValueError("expected_asset_count must be exactly three")
    if [asset.get("role") for asset in assets if isinstance(asset, dict)] != [
        "actor",
        "judge",
        "editor",
    ]:
        raise ValueError("asset order and roles must be exactly actor, judge, editor")
    roles: set[str] = set()
    runtime_count = 0
    checkpoint_bytes = 0
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("every manifest asset must be an object")
        role = str(asset.get("role", ""))
        if role not in EXPECTED_LAYOUT or role in roles:
            raise ValueError(f"unexpected or duplicate role: {role!r}")
        roles.add(role)
        if asset.get("hub_subfolder") != EXPECTED_LAYOUT[role]:
            raise ValueError(f"manifest folder mismatch for {role}")
        contract = EXPECTED_ASSET_CONTRACTS[role]
        for key, expected in contract.items():
            if key == "required_files":
                continue
            if asset.get(key) != expected:
                raise ValueError(f"production {role} {key} mismatch")
        required = asset.get("required_files")
        verified = asset.get("verified_files")
        if (
            not isinstance(required, list)
            or not required
            or len(required) != len(set(required))
            or not isinstance(verified, dict)
            or set(required) != set(verified)
        ):
            raise ValueError(f"invalid exact-file contract for {role}")
        if required != list(contract["required_files"]):
            raise ValueError(f"production required_files mismatch for {role}")
        records: list[dict[str, Any]] = []
        for name in required:
            if _safe_relative(name) != name:
                raise ValueError(f"non-normalized required path for {role}: {name!r}")
            record = verified[name]
            if (
                not isinstance(record, dict)
                or isinstance(record.get("size_bytes"), bool)
                or not isinstance(record.get("size_bytes"), int)
                or record["size_bytes"] <= 0
                or _SHA256.fullmatch(str(record.get("sha256", ""))) is None
            ):
                raise ValueError(f"invalid verified record for {role}/{name}")
            records.append({"name": name, **record})
        if len(records) != asset.get("expected_file_count"):
            raise ValueError(f"file count mismatch for {role}")
        total = sum(item["size_bytes"] for item in records)
        if total != asset.get("expected_total_bytes"):
            raise ValueError(f"byte count mismatch for {role}")
        if _tree_digest(records) != asset.get("expected_export_tree_sha256"):
            raise ValueError(f"tree digest mismatch for {role}")
        runtime_count += len(records)
        checkpoint_bytes += total
    if roles != set(EXPECTED_LAYOUT):
        raise ValueError("roles must be exactly actor, judge, and editor")
    if runtime_count != manifest.get("expected_runtime_file_count") or (
        not allow_test_payloads and runtime_count != 38
    ):
        raise ValueError("aggregate runtime-file count mismatch")
    if checkpoint_bytes != manifest.get("expected_checkpoint_bytes") or (
        not allow_test_payloads and checkpoint_bytes != 34_177_510_861
    ):
        raise ValueError("aggregate checkpoint-byte count mismatch")
    root_license = manifest.get("root_license")
    if not isinstance(root_license, dict):
        raise ValueError("missing root LICENSE contract")
    for key in ("source_logical_id", "source_path", "published_path"):
        if root_license.get(key) != ROOT_LICENSE_CONTRACT[key]:
            raise ValueError("root LICENSE source contract mismatch")
    if not allow_test_payloads and root_license != ROOT_LICENSE_CONTRACT:
        raise ValueError("root LICENSE integrity contract mismatch")
    if (
        not allow_test_payloads
        and _manifest_content_sha256(manifest) != PRODUCTION_MANIFEST_CONTENT_SHA256
    ):
        raise ValueError("production manifest content digest mismatch")
    return assets


def _expected_paths(manifest: dict[str, Any], assets: list[dict[str, Any]]) -> set[str]:
    expected = set(EXPECTED_ROOT_FILES)
    for asset in assets:
        folder = asset["hub_subfolder"]
        expected.update(f"{folder}/{name}" for name in asset["required_files"])
        if asset["role"] in {"actor", "judge"}:
            expected.add(f"{folder}/provenance.json")
    return expected


def _validate_provenance(
    path: Path,
    asset: dict[str, Any],
    actual_records: list[dict[str, Any]],
) -> None:
    value = _json(path)
    if (
        value.get("schema_version") != PROVENANCE_SCHEMA
        or value.get("asset_id") != asset.get("asset_id")
        or value.get("source_logical_id") != asset.get("source_logical_id")
        or value.get("hub_subfolder") != asset.get("hub_subfolder")
    ):
        raise ValueError(f"provenance identity mismatch: {asset['role']}/provenance.json")
    identity = value.get("source_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"missing source_identity in {asset['role']}/provenance.json")
    expected_identity = {
        "source_checkpoint_tree_sha256_recorded": asset["source_checkpoint_tree_sha256"],
        "selected_export_tree_sha256": asset["expected_export_tree_sha256"],
        "file_count": asset["expected_file_count"],
        "total_bytes": asset["expected_total_bytes"],
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"provenance {key} mismatch for {asset['role']}")
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != len(actual_records):
        raise ValueError(f"provenance files mismatch for {asset['role']}")
    try:
        provenance_records = {str(item["name"]): item for item in raw_files}
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed provenance files for {asset['role']}") from exc
    if len(provenance_records) != len(raw_files):
        raise ValueError(f"duplicate provenance file records for {asset['role']}")
    for record in actual_records:
        published = provenance_records.get(record["name"])
        if not isinstance(published, dict) or any(
            published.get(key) != record[key] for key in ("name", "size_bytes", "sha256")
        ):
            raise ValueError(f"provenance content mismatch for {asset['role']}/{record['name']}")
    _scan_public_text(path, f"{asset['role']}/provenance.json")


def validate_release(
    root: Path,
    manifest: dict[str, Any] | None = None,
    *,
    allow_test_payloads: bool = False,
) -> dict[str, Any]:
    candidate = root.expanduser()
    if candidate.is_symlink():
        raise ValueError("release directory must not be a symlink")
    root = candidate.resolve()
    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("release directory contains a symlink")
    if manifest is None:
        manifest_file = Path(__file__).resolve().parents[1] / "checkpoints" / "hf_checkpoint_manifest.json"
        manifest = _json(manifest_file)
    assets = _manifest_assets(manifest, allow_test_payloads=allow_test_payloads)

    root_entries = {item.name for item in root.iterdir()}
    expected_root_entries = set(EXPECTED_ROOT_FILES) | set(EXPECTED_LAYOUT.values())
    if root_entries != expected_root_entries:
        raise ValueError(
            f"root entries must be exactly {sorted(expected_root_entries)}; "
            f"missing={sorted(expected_root_entries - root_entries)}, "
            f"extra={sorted(root_entries - expected_root_entries)}"
        )
    for name in EXPECTED_ROOT_FILES:
        if not (root / name).is_file():
            raise ValueError(f"missing root file: {name}")
    for name in EXPECTED_LAYOUT.values():
        if not (root / name).is_dir():
            raise ValueError(f"missing model directory: {name}/")

    expected_paths = _expected_paths(manifest, assets)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError(
            "release file set differs from the exact three-model contract; "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )

    license_contract = manifest.get("root_license")
    if not isinstance(license_contract, dict):
        raise ValueError("missing root LICENSE contract")
    license_path = root / "LICENSE"
    if (
        license_path.stat().st_size != license_contract.get("size_bytes")
        or _sha256(license_path) != license_contract.get("sha256")
    ):
        raise ValueError("root LICENSE is not the verified FLUX.2 Klein Apache-2.0 text")
    for name in EXPECTED_ROOT_FILES:
        _scan_public_text(root / name, name)

    verified_count = 0
    verified_bytes = 0
    export_trees: dict[str, str] = {}
    for asset in assets:
        records: list[dict[str, Any]] = []
        folder = root / asset["hub_subfolder"]
        for name in asset["required_files"]:
            path = folder / name
            expected = asset["verified_files"][name]
            size = path.stat().st_size
            if size != expected["size_bytes"]:
                raise ValueError(f"size mismatch: {asset['hub_subfolder']}/{name}")
            actual_sha = _sha256(path)
            if actual_sha != expected["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {asset['hub_subfolder']}/{name}")
            records.append({"name": name, "size_bytes": size, "sha256": actual_sha})
        digest = _tree_digest(records)
        if digest != asset["expected_export_tree_sha256"]:
            raise ValueError(f"selected export-tree mismatch for {asset['role']}")
        if asset["role"] in {"actor", "judge"}:
            _validate_provenance(folder / "provenance.json", asset, records)
        export_trees[asset["role"]] = digest
        verified_count += len(records)
        verified_bytes += sum(item["size_bytes"] for item in records)

    return {
        "valid": True,
        "repository_id": manifest["repository_id"],
        "asset_count": len(assets),
        "runtime_file_count": verified_count,
        "public_file_count": len(actual_paths),
        "checkpoint_bytes": verified_bytes,
        "export_tree_sha256": export_trees,
        "local_paths": sorted(actual_paths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-upload", action="store_true")
    parser.add_argument(
        "--replace-remote",
        action="store_true",
        help="Delete every remote path not present in the verified local tree",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.commit_upload != args.replace_remote:
        raise ValueError("publishing requires both --commit-upload and --replace-remote")
    plan = validate_release(args.folder)
    local_paths = plan.pop("local_paths")
    plan.update(
        {
            "revision": args.revision,
            "will_upload": bool(args.commit_upload),
            "will_replace_remote": bool(args.replace_remote),
        }
    )
    if not args.commit_upload:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("Dry run complete; Hugging Face was not contacted.")
        return 0

    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    except ImportError as exc:
        raise ValueError("huggingface_hub is required for publishing") from exc

    api = HfApi()
    info = api.repo_info(
        repo_id=plan["repository_id"],
        repo_type="model",
        revision=args.revision,
    )
    parent_commit = str(info.sha)
    remote_paths = {
        _safe_relative(path)
        for path in api.list_repo_files(
            repo_id=plan["repository_id"],
            repo_type="model",
            revision=parent_commit,
        )
    }
    local_set = set(local_paths)
    delete_paths = sorted(remote_paths - local_set)
    operations = [CommitOperationDelete(path_in_repo=path) for path in delete_paths]
    operations.extend(
        CommitOperationAdd(
            path_in_repo=path,
            path_or_fileobj=str((args.folder.resolve() / path)),
        )
        for path in sorted(local_set)
    )
    result = api.create_commit(
        repo_id=plan["repository_id"],
        repo_type="model",
        revision=args.revision,
        parent_commit=parent_commit,
        operations=operations,
        commit_message=COMMIT_MESSAGE,
    )
    output = {
        **plan,
        "remote_parent_commit": parent_commit,
        "deleted_remote_file_count": len(delete_paths),
        "added_or_replaced_file_count": len(local_set),
        "commit_oid": str(getattr(result, "oid", "")),
        "commit_url": str(getattr(result, "commit_url", result)),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
