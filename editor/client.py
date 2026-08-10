from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests


def get_distributed_rank() -> int:
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def select_diffusers_server(servers: list[str], routing_strategy: str, request_index: int) -> str:
    normalized = [server.strip().rstrip("/") for server in servers if server.strip()]
    if not normalized:
        raise ValueError("No Diffusers servers configured")
    rank = get_distributed_rank()
    strategy = (routing_strategy or "rank_affine").lower()
    if strategy == "rank_affine":
        return normalized[rank % len(normalized)]
    if strategy == "rank_round_robin":
        return normalized[(rank + int(request_index)) % len(normalized)]
    if strategy == "round_robin":
        return normalized[int(request_index) % len(normalized)]
    raise ValueError(f"unsupported Diffusers routing strategy: {routing_strategy!r}")


def request_feedback_image_via_diffusers(
    *,
    servers: list[str],
    image_path: str,
    positive_prompt: str,
    negative_prompt: str,
    region_prompt: str,
    edit_plan: dict[str, Any] | None,
    request_index: int,
    timeout_sec: float,
    routing_strategy: str = "rank_affine",
    seed: int | None = None,
) -> dict[str, Any]:
    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"image_path does not exist: {image_path}")
    server = select_diffusers_server(servers, routing_strategy, request_index)
    response = requests.post(
        f"{server}/edit",
        json={
            "image_path": str(source),
            "positive_prompt": str(positive_prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
            "region_prompt": str(region_prompt or ""),
            "edit_plan": edit_plan or {},
            "request_index": int(request_index),
            "seed": seed,
        },
        timeout=float(timeout_sec),
    )
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"Diffusers service returned invalid JSON: HTTP {response.status_code}") from exc
    if response.status_code >= 400:
        message = payload.get("error") or payload.get("detail") or f"HTTP {response.status_code}"
        raise RuntimeError(f"Diffusers service error: {message}")
    if payload.get("status") != "success":
        raise RuntimeError(f"Diffusers service error: {payload.get('error') or payload.get('status') or 'unknown'}")
    edited_path = Path(str(payload.get("edited_path") or ""))
    if not edited_path.is_file():
        raise RuntimeError(f"Diffusers service edited_path does not exist: {edited_path}")
    if payload.get("backend") != "diffusers_flux2":
        raise RuntimeError(f"unexpected Diffusers backend: {payload.get('backend')!r}")
    return payload
