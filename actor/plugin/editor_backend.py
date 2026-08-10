from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Callable


DIFFUSERS_URLS = (
    "http://127.0.0.1:8212",
    "http://127.0.0.1:8213",
    "http://127.0.0.1:8214",
    "http://127.0.0.1:8215",
)
COMFY_URLS = (
    "http://127.0.0.1:8214",
    "http://127.0.0.1:8215",
    "http://127.0.0.1:8216",
    "http://127.0.0.1:8217",
)


def editor_backend() -> str:
    backend = os.environ.get("IMAGE_EDIT_BACKEND", "diffusers").strip().lower()
    if backend not in {"diffusers", "comfy"}:
        raise ValueError("IMAGE_EDIT_BACKEND must be 'diffusers' or 'comfy'")
    return backend


def _parse_urls(value: str, fallback: tuple[str, ...]) -> list[str]:
    urls = [item.strip().rstrip("/") for item in re.split(r"[\s,]+", value.strip()) if item.strip()]
    return urls or list(fallback)


def editor_urls(backend: str | None = None) -> list[str]:
    selected = backend or editor_backend()
    if selected == "diffusers":
        return _parse_urls(os.environ.get("DIFFUSERS_SERVERS", ""), DIFFUSERS_URLS)
    if selected == "comfy":
        raw = os.environ.get("VF_LOOP_COMFY_ADAPTER_URLS", "")
        if not raw:
            raw = os.environ.get("VF_LOOP_COMFY_ADAPTER_URL", "")
        return _parse_urls(raw, COMFY_URLS)
    raise ValueError(f"unsupported editor backend: {selected}")


def process_rank() -> int:
    for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def select_editor_url(completion_index: int, *, rank: int | None = None, backend: str | None = None) -> str:
    urls = editor_urls(backend)
    selected_rank = process_rank() if rank is None else int(rank)
    return urls[(selected_rank + int(completion_index)) % len(urls)]


def trajectory_request_index(*, rollout_call: int, rank: int, completion_index: int) -> int:
    rollout_call = int(rollout_call)
    rank = int(rank)
    completion_index = int(completion_index)
    if rollout_call < 1 or rank < 0 or completion_index < 0 or completion_index >= 1_000_000:
        raise ValueError("invalid trajectory request-index coordinates")
    return rank * 1_000_000_000_000 + rollout_call * 1_000_000 + completion_index


def _project_diffusers_request(**kwargs):
    package_root = Path(__file__).resolve().parents[2]
    project_root = Path(os.environ.get("VF_PROJECT_ROOT", package_root)).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from editor.client import request_feedback_image_via_diffusers

    return request_feedback_image_via_diffusers(**kwargs)


def request_image_edit(
    *,
    image_path: str,
    editing: str,
    request_index: int,
    completion_index: int,
    backend: str | None = None,
    editor_url: str | None = None,
    diffusers_request: Callable[..., dict[str, Any]] | None = None,
    comfy_request: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = backend or editor_backend()
    instruction = str(editing or "").strip()
    if not instruction:
        return {"status": "skipped", "reason": "empty_editing", "backend": selected}
    url = (editor_url or select_editor_url(completion_index, backend=selected)).rstrip("/")

    if selected == "diffusers":
        request_fn = diffusers_request or _project_diffusers_request
        result = request_fn(
            servers=[url],
            image_path=str(image_path),
            positive_prompt=instruction,
            negative_prompt="",
            region_prompt="",
            edit_plan={},
            request_index=int(request_index),
            timeout_sec=float(os.environ.get("VF_LOOP_SERVICE_TIMEOUT", "300")),
            routing_strategy="round_robin",
            seed=None,
        )
        if not isinstance(result, dict):
            raise RuntimeError("Diffusers service returned a non-object response")
        if result.get("status") != "success" or result.get("backend") != "diffusers_flux2":
            raise RuntimeError("Diffusers service returned an invalid success identity")
        edited_path = Path(str(result.get("edited_path") or ""))
        if not edited_path.is_file():
            raise RuntimeError(f"Diffusers edited_path does not exist: {edited_path}")
        return result

    if selected == "comfy":
        if comfy_request is None:
            raise RuntimeError("ComfyUI backup requires an explicit request adapter")
        regions = [{"bbox": [0.0, 0.0, 1.0, 1.0], "instruction": instruction}]
        return comfy_request(str(image_path), regions, int(completion_index), adapter_url=url)

    raise ValueError(f"unsupported editor backend: {selected}")
