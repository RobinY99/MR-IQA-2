from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("migration manifest must be a JSON object")
    if payload.get("schema_version") != "vf_migration_manifest_v1":
        raise ValueError("unsupported migration manifest schema")
    if not isinstance(payload.get("components"), dict):
        raise ValueError("migration manifest components must be an object")
    return payload


def unresolved_blockers(manifest: dict[str, Any]) -> list[str]:
    components = manifest.get("components")
    if not isinstance(components, dict):
        return ["manifest.components"]
    blockers: list[str] = []
    for name, component in components.items():
        if not isinstance(component, dict):
            blockers.append(str(name))
            continue
        status = component.get("status")
        if component.get("training_blocker", True) or status != "verified":
            blockers.append(str(name))
    return blockers


def require_training_ready(manifest: dict[str, Any]) -> None:
    blockers = unresolved_blockers(manifest)
    if blockers:
        raise RuntimeError("training blocked by: " + ", ".join(blockers))
