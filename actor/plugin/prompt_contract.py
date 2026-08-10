from __future__ import annotations

import hashlib
import json
from typing import Any

from actor_contract import (
    LEGACY_ACTOR_SCHEMA,
    REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA,
    REASONS_RATING_ACTOR_SCHEMA,
    REASONING_FIELDS,
    active_top_level_fields,
    actor_schema,
)

ACTOR_SCHEMA = actor_schema()
ACTOR_TOP_LEVEL_FIELDS = active_top_level_fields()
ENABLE_THINKING = False
ADD_NON_THINKING_PREFIX = False

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, respond with exactly one valid JSON object and no other text."
)

if ACTOR_SCHEMA == REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA:
    PROMPT_VERSION = "vf_reasoning_evidence_solution_rating_v5_20260724"
    USER_PROMPT_TEXT = (
        "Assess the overall perceptual quality of this image.\n\n"
        "Respond with exactly one JSON object containing these keys in this order: \"reasoning\" and \"rating\". "
        "\"reasoning\" must be one JSON object containing these keys in this order: \"evidence\" and \"solution\".\n\n"
        "\"evidence\" must be one concise string grounded in the visible image evidence that determines its current "
        "overall perceptual quality, and must indicate where that evidence appears in the image.\n"
        "\"solution\" must be one concise string containing a coherent image-edit plan that causally addresses the "
        "evidence. The edited result must retain the same semantic meaning as the input image. If the image is already "
        "high quality, request only a minimal preservation-first refinement without inventing a defect.\n"
        "\"rating\" must be a numeric string from 1.00 to 5.00 with exactly two decimal places. \"1.00\" represents "
        "the worst quality and \"5.00\" represents excellent quality."
    )
elif ACTOR_SCHEMA == REASONS_RATING_ACTOR_SCHEMA:
    PROMPT_VERSION = "vf_reasons_rating_qwen3vl_v1_20260716"
    USER_PROMPT_TEXT = (
        "Please assess the overall perceptual quality of this image and provide a quality rating written with "
        "exactly two decimal places.\n\n"
        "Respond with exactly one JSON object containing these keys in this order: \"reasons\" and \"rating\".\n\n"
        "\"reasons\" must be one concise string that first describes the visible evidence affecting image quality "
        "and then, when meaningful, gives one specific action for improving the image. If no meaningful improvement "
        "is needed, state that no correction is necessary.\n"
        "\"rating\" must be a finite number or numeric string from 1.00 to 5.00. \"1.00\" represents the worst "
        "quality and \"5.00\" represents excellent quality."
    )
else:
    PROMPT_VERSION = "vf_reason_rating_suggestion_qwen3vl_v1_20260716"
    USER_PROMPT_TEXT = (
        "Please assess the overall perceptual quality of this image. Explain the visible reasons affecting its quality, "
        "provide a quality rating written with exactly two decimal places, and give one specific suggestion for reducing "
        "the negative factors affecting the image.\n\n"
        "Respond with exactly one JSON object containing these keys in this order: \"reason\", \"rating\", and \"suggestion\".\n\n"
        "\"reason\" must be a string describing the visible reasons that affect the image quality.\n"
        "\"rating\" must be a finite number or numeric string from 1.00 to 5.00. \"1.00\" represents the worst quality "
        "and \"5.00\" represents excellent quality.\n"
        "\"suggestion\" must be a string containing a specific image-improvement instruction. If the image has no "
        "meaningful room for improvement, use an empty string."
    )
TRAINING_USER_PROMPT = f"<image>{USER_PROMPT_TEXT}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contract_payload() -> dict[str, Any]:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_text": USER_PROMPT_TEXT,
        "enable_thinking": ENABLE_THINKING,
        "add_non_thinking_prefix": ADD_NON_THINKING_PREFIX,
    }
    if ACTOR_SCHEMA != LEGACY_ACTOR_SCHEMA:
        payload["actor_schema"] = ACTOR_SCHEMA
        payload["top_level_fields"] = list(ACTOR_TOP_LEVEL_FIELDS)
    if ACTOR_SCHEMA == REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA:
        payload["reasoning_fields"] = list(REASONING_FIELDS)
        payload["editor_contract_version"] = "same_semantics_same_size_v2_20260724"
    return payload


PROMPT_HASH = _sha256(json.dumps(_contract_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
SYSTEM_PROMPT_HASH = _sha256(SYSTEM_PROMPT)
USER_PROMPT_HASH = _sha256(USER_PROMPT_TEXT)


def prompt_metadata() -> dict[str, Any]:
    metadata = {
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "system_prompt_hash": SYSTEM_PROMPT_HASH,
        "user_prompt_hash": USER_PROMPT_HASH,
        "enable_thinking": ENABLE_THINKING,
        "add_non_thinking_prefix": ADD_NON_THINKING_PREFIX,
    }
    if ACTOR_SCHEMA != LEGACY_ACTOR_SCHEMA:
        metadata["actor_schema"] = ACTOR_SCHEMA
        metadata["top_level_fields"] = list(ACTOR_TOP_LEVEL_FIELDS)
    if ACTOR_SCHEMA == REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA:
        metadata["reasoning_fields"] = list(REASONING_FIELDS)
        metadata["editor_contract_version"] = "same_semantics_same_size_v2_20260724"
    return metadata


def validate_image_binding(user_prompt: str, images: list[Any]) -> None:
    tag_count = str(user_prompt).count("<image>")
    image_count = len(images)
    if tag_count != image_count:
        raise ValueError(f"image placeholder mismatch: tags={tag_count}, images={image_count}")


def build_training_messages(images: list[Any]) -> list[dict[str, str]]:
    validate_image_binding(TRAINING_USER_PROMPT, images)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TRAINING_USER_PROMPT},
    ]


def build_structured_validation_messages(image_path: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_PROMPT_TEXT},
            ],
        },
    ]
