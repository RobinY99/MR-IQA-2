from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from typing import Any


_DEFAULT_JUDGER_MODEL_ID = "e5_judge"
_DEFAULT_JUDGER_MODEL_PATH = ""
_DEFAULT_JUDGER_MODEL_TREE_SHA256 = ""
JUDGER_MODEL_ID = os.environ.get(
    "VF_JUDGE_MODEL_ID",
    os.environ.get("JUDGER_MODEL_ID", _DEFAULT_JUDGER_MODEL_ID),
)
JUDGER_MODEL_PATH = os.environ.get(
    "VF_JUDGE_MODEL_PATH",
    os.environ.get("JUDGER_MODEL_PATH", _DEFAULT_JUDGER_MODEL_PATH),
)
JUDGER_MODEL_TREE_SHA256 = os.environ.get(
    "VF_JUDGE_MODEL_TREE_SHA256",
    os.environ.get(
        "JUDGER_MODEL_TREE_SHA256",
        _DEFAULT_JUDGER_MODEL_TREE_SHA256,
    ),
)
JUDGER_PYTHON = os.environ.get("JUDGER_PYTHON", sys.executable)


def _csv_int_tuple(name: str, default: str) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in os.environ.get(name, default).split(",")
        if value.strip()
    )


JUDGER_GPUS = _csv_int_tuple("JUDGER_GPUS", "4,5,6,7")
JUDGER_PORTS = _csv_int_tuple("JUDGER_PORTS", "8204,8205,8206,8207")
JUDGER_SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, respond with "
    "exactly one valid JSON object and no other text."
)

LEGACY_JUDGER_PROMPT_SCHEMA = "legacy_reasons_rating_v1"
E5_TRAINING_JUDGER_PROMPT_SCHEMA = "e5_training_reasoning_v5"
JUDGER_PROMPT_SCHEMA = os.environ.get(
    "VF_JUDGE_PROMPT_SCHEMA",
    os.environ.get("JUDGER_PROMPT_SCHEMA", E5_TRAINING_JUDGER_PROMPT_SCHEMA),
).strip()

_LEGACY_JUDGER_USER_PROMPT = (
    "Please assess the overall perceptual quality of this image and provide a quality rating written with "
    "exactly two decimal places.\n\n"
    "Respond with exactly one JSON object containing these keys in this order: \"reasons\" and \"rating\".\n\n"
    "\"reasons\" must be one concise string that first describes the visible evidence affecting image quality "
    "and then, when meaningful, gives one specific action for improving the image. If no meaningful improvement "
    "is needed, state that no correction is necessary.\n"
    "\"rating\" must be a finite number or numeric string from 1.00 to 5.00. \"1.00\" represents the worst "
    "quality and \"5.00\" represents excellent quality."
)

_E5_TRAINING_JUDGER_USER_PROMPT = (
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

if JUDGER_PROMPT_SCHEMA == LEGACY_JUDGER_PROMPT_SCHEMA:
    JUDGER_PROMPT_VERSION = "vf_reasons_rating_qwen3vl_v1_20260716"
    JUDGER_USER_PROMPT = _LEGACY_JUDGER_USER_PROMPT
    JUDGER_ACTOR_SCHEMA = "reasons_rating"
    JUDGER_TOP_LEVEL_FIELDS = ("reasons", "rating")
    JUDGER_REASONING_FIELDS: tuple[str, ...] = ()
    JUDGER_PROMPT_HASH = (
        "ba13643b5a1dd4bc7c5def1c8ec46a2d7af5cfb5e128d815e77fa7b6bfc7dc3e"
    )
elif JUDGER_PROMPT_SCHEMA == E5_TRAINING_JUDGER_PROMPT_SCHEMA:
    JUDGER_PROMPT_VERSION = "vf_reasoning_evidence_solution_rating_v5_20260724"
    JUDGER_USER_PROMPT = _E5_TRAINING_JUDGER_USER_PROMPT
    JUDGER_ACTOR_SCHEMA = "reasoning_evidence_solution_rating"
    JUDGER_TOP_LEVEL_FIELDS = ("reasoning", "rating")
    JUDGER_REASONING_FIELDS = ("evidence", "solution")
    JUDGER_PROMPT_HASH = (
        "fa78a4ccfd2194a2026ff0b6b722bf22b28f8fa060389c57c4adb1618ac280f6"
    )
else:
    raise RuntimeError(f"unsupported Judge prompt schema: {JUDGER_PROMPT_SCHEMA!r}")

JUDGER_PROMPT_RATING_RANGE = (1.0, 5.0)
JUDGER_SCORE_ACCEPTANCE_RANGE = (
    (0.0, 5.0)
    if JUDGER_PROMPT_SCHEMA == E5_TRAINING_JUDGER_PROMPT_SCHEMA
    else JUDGER_PROMPT_RATING_RANGE
)
JUDGER_PROMPT = JUDGER_USER_PROMPT
JUDGER_SYSTEM_PROMPT_SHA256 = (
    "ca179438060193895f1a9282de1ed3dd8edfd871cc5c1aa7e43635d9f783c9a5"
)
JUDGER_USER_PROMPT_SHA256 = hashlib.sha256(JUDGER_USER_PROMPT.encode()).hexdigest()
JUDGER_PROMPT_SHA256 = hashlib.sha256(JUDGER_PROMPT.encode()).hexdigest()
RATING_TEXT_RE = re.compile(r"(?:[1-4]\.\d{2}|5\.00)")

_prompt_contract = {
    "prompt_version": JUDGER_PROMPT_VERSION,
    "system_prompt": JUDGER_SYSTEM_PROMPT,
    "user_prompt_text": JUDGER_USER_PROMPT,
    "enable_thinking": False,
    "add_non_thinking_prefix": False,
    "actor_schema": JUDGER_ACTOR_SCHEMA,
    "top_level_fields": list(JUDGER_TOP_LEVEL_FIELDS),
}
if JUDGER_REASONING_FIELDS:
    _prompt_contract["reasoning_fields"] = list(JUDGER_REASONING_FIELDS)
    _prompt_contract["editor_contract_version"] = (
        "same_semantics_same_size_v2_20260724"
    )
_computed_prompt_hash = hashlib.sha256(
    json.dumps(
        _prompt_contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
if _computed_prompt_hash != JUDGER_PROMPT_HASH:
    raise RuntimeError(
        "Judge prompt contract hash mismatch: "
        f"computed={_computed_prompt_hash}, expected={JUDGER_PROMPT_HASH}"
    )
_requested_prompt_hash = os.environ.get(
    "VF_JUDGE_PROMPT_HASH",
    os.environ.get("JUDGER_PROMPT_HASH", ""),
).strip()
if _requested_prompt_hash and _requested_prompt_hash != JUDGER_PROMPT_HASH:
    raise RuntimeError(
        "requested Judge prompt hash does not match selected schema: "
        f"requested={_requested_prompt_hash}, selected={JUDGER_PROMPT_HASH}"
    )
JUDGER_GENERATION = {
    "max_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "seed": 42,
    "return_details": True,
    "enable_thinking": False,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.24,
    "max_model_len": 2048,
    "max_num_seqs": int(os.environ.get("VF_JUDGER_MAX_NUM_SEQS", "1")),
    "enforce_eager": True,
    "limit_mm_per_prompt": {"image": 1},
    "max_pixels": 196608,
    "min_pixels": 3136,
}


def _rating_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    minimum, maximum = JUDGER_SCORE_ACCEPTANCE_RANGE
    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return number


def parse_score_payload(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    try:
        payload, end = json.JSONDecoder().raw_decode(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {
            "score": None,
            "errors": ["json"],
            "payload": None,
            "judge_reasons": None,
            "reasoning_evidence": None,
            "reasoning_solution": None,
            "rating_text": None,
            "rating_format_ok": False,
            "rating_representation": "invalid",
            "rating_format_warning": None,
            "rating_prompt_range_ok": False,
            "rating_range_warning": None,
        }
    if raw[end:].strip() or not isinstance(payload, dict):
        return {
            "score": None,
            "errors": ["payload"],
            "payload": payload if isinstance(payload, dict) else None,
            "judge_reasons": None,
            "reasoning_evidence": None,
            "reasoning_solution": None,
            "rating_text": None,
            "rating_format_ok": False,
            "rating_representation": "invalid",
            "rating_format_warning": None,
            "rating_prompt_range_ok": False,
            "rating_range_warning": None,
        }

    errors: list[str] = []
    if tuple(payload) != JUDGER_TOP_LEVEL_FIELDS:
        errors.append("top_level")

    evidence = None
    solution = None
    judge_reasons = None
    if JUDGER_PROMPT_SCHEMA == LEGACY_JUDGER_PROMPT_SCHEMA:
        reasons = payload.get("reasons")
        if not isinstance(reasons, str) or not reasons.strip():
            errors.append("reasons")
        else:
            judge_reasons = reasons.strip()
    else:
        reasoning = payload.get("reasoning")
        if not isinstance(reasoning, dict):
            errors.append("reasoning")
        else:
            if tuple(reasoning) != JUDGER_REASONING_FIELDS:
                errors.append("reasoning_fields")
            raw_evidence = reasoning.get("evidence")
            raw_solution = reasoning.get("solution")
            if not isinstance(raw_evidence, str) or not raw_evidence.strip():
                errors.append("evidence")
            else:
                evidence = raw_evidence.strip()
            if not isinstance(raw_solution, str) or not raw_solution.strip():
                errors.append("solution")
            else:
                solution = raw_solution.strip()
            if evidence and solution:
                judge_reasons = f"Evidence: {evidence} Solution: {solution}"

    rating_value = payload.get("rating")
    rating = _rating_number(rating_value)
    if rating is None:
        errors.append("rating")
    rating_format_ok = (
        isinstance(rating_value, str)
        and bool(RATING_TEXT_RE.fullmatch(rating_value.strip()))
    )
    if rating_format_ok:
        rating_representation = "numeric_string_two_decimals"
    elif isinstance(rating_value, bool):
        rating_representation = "invalid"
    elif isinstance(rating_value, (int, float)):
        rating_representation = "json_number"
    elif isinstance(rating_value, str) and rating is not None:
        rating_representation = "numeric_string_noncanonical"
    else:
        rating_representation = "invalid"
    rating_format_warning = (
        "e5_prompt_expected_numeric_string_two_decimals"
        if (
        JUDGER_PROMPT_SCHEMA == E5_TRAINING_JUDGER_PROMPT_SCHEMA
        and not rating_format_ok
        and rating is not None
        )
        else None
    )
    prompt_minimum, prompt_maximum = JUDGER_PROMPT_RATING_RANGE
    rating_prompt_range_ok = (
        rating is not None and prompt_minimum <= rating <= prompt_maximum
    )
    rating_range_warning = (
        "accepted_nonnegative_e5_judge_score_below_prompt_floor"
        if (
            JUDGER_PROMPT_SCHEMA == E5_TRAINING_JUDGER_PROMPT_SCHEMA
            and rating is not None
            and rating < prompt_minimum
        )
        else None
    )
    valid_rating = rating if not errors else None
    return {
        "score": valid_rating,
        "errors": errors,
        "payload": payload,
        "judge_reasons": judge_reasons,
        "reasoning_evidence": evidence,
        "reasoning_solution": solution,
        "rating_text": f"{valid_rating:.2f}" if valid_rating is not None else None,
        "rating_format_ok": rating_format_ok,
        "rating_representation": rating_representation,
        "rating_format_warning": rating_format_warning,
        "rating_prompt_range_ok": rating_prompt_range_ok,
        "rating_range_warning": rating_range_warning,
    }


def parse_score_completion(text: str) -> tuple[float | None, list[str]]:
    result = parse_score_payload(text)
    return result["score"], result["errors"]


def judger_metadata() -> dict[str, Any]:
    metadata = {
        "backend": (
            "dapo_qwen35_4b_vllm_judge"
            if JUDGER_PROMPT_SCHEMA == LEGACY_JUDGER_PROMPT_SCHEMA
            else "e5_qwen35_4b_vllm_judge"
        ),
        "model_id": JUDGER_MODEL_ID,
        "model_path": JUDGER_MODEL_PATH,
        "model_tree_sha256": JUDGER_MODEL_TREE_SHA256,
        "python": JUDGER_PYTHON,
        "prompt_schema": JUDGER_PROMPT_SCHEMA,
        "prompt_version": JUDGER_PROMPT_VERSION,
        "prompt_hash": JUDGER_PROMPT_HASH,
        "system_prompt_sha256": JUDGER_SYSTEM_PROMPT_SHA256,
        "user_prompt_sha256": JUDGER_USER_PROMPT_SHA256,
        "actor_schema": JUDGER_ACTOR_SCHEMA,
        "top_level_fields": list(JUDGER_TOP_LEVEL_FIELDS),
        "generation": dict(JUDGER_GENERATION),
        "gpus": list(JUDGER_GPUS),
        "ports": list(JUDGER_PORTS),
        "deterministic": True,
        "cache_compatible": True,
        "prompt_rating_range": list(JUDGER_PROMPT_RATING_RANGE),
        "score_acceptance_range": list(JUDGER_SCORE_ACCEPTANCE_RANGE),
    }
    if JUDGER_REASONING_FIELDS:
        metadata["reasoning_fields"] = list(JUDGER_REASONING_FIELDS)
    return metadata
