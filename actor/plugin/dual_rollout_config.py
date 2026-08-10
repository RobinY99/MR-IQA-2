from __future__ import annotations

import math
import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def apply_qwen35_sampling_config(
    request_config: Any,
    *,
    presence_penalty: float,
    repetition_penalty: float,
) -> Any:
    presence = float(presence_penalty)
    repetition = float(repetition_penalty)
    if not math.isfinite(presence) or not -2.0 <= presence <= 2.0:
        raise ValueError(f"presence_penalty must be finite and in [-2, 2], got {presence}")
    if not math.isfinite(repetition) or repetition <= 0.0:
        raise ValueError(f"repetition_penalty must be finite and positive, got {repetition}")
    request_config.presence_penalty = presence
    request_config.repetition_penalty = repetition
    return request_config


def image_reference_path(value: Any) -> str:
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("image path sequence is empty")
        value = value[0]
    if isinstance(value, Mapping):
        for key in ("path", "image_path", "source_image"):
            if value.get(key):
                return image_reference_path(value[key])
        raise ValueError("image path mapping has no path field")
    if isinstance(value, (str, os.PathLike)):
        path = os.fspath(value).strip()
        if path:
            return path
    raise ValueError(f"unsupported image path value: {type(value).__name__}")


def source_image_path(record: dict[str, Any]) -> str:
    images = record.get("images")
    if isinstance(images, (list, tuple)) and images:
        return image_reference_path(images)
    for key in ("image_path", "source_image"):
        value = record.get(key)
        if value:
            return image_reference_path(value)
    raise ValueError("rollout record has no source image")


def prepare_a1_record(a0_record: Mapping[str, Any], image: Any) -> dict[str, Any]:
    data = deepcopy(dict(a0_record))
    messages = deepcopy(data.get("messages") or [])
    if messages and messages[-1].get("role") == "assistant":
        messages.pop()
    data["messages"] = messages
    data["images"] = [image_reference_path(image)]
    for key in (
        "prompt_id",
        "request_id",
        "response_token_ids",
        "response_loss_mask",
        "rollout_logprobs",
        "rollout_infos",
        "finish_reason",
        "is_truncated",
        "add_eos",
    ):
        data.pop(key, None)
    return data
