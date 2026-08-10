from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Mapping


LEGACY_ACTOR_SCHEMA = "reason_rating_suggestion"
REASONS_RATING_ACTOR_SCHEMA = "reasons_rating"
REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA = (
    "reasoning_evidence_solution_rating"
)
SUPPORTED_ACTOR_SCHEMAS = {
    LEGACY_ACTOR_SCHEMA,
    REASONS_RATING_ACTOR_SCHEMA,
    REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA,
}
TOP_LEVEL_FIELDS = ("reason", "rating", "suggestion")
REASONS_RATING_FIELDS = ("reasons", "rating")
REASONING_RATING_FIELDS = ("reasoning", "rating")
REASONING_FIELDS = ("evidence", "solution")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
QWEN35_NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def actor_schema() -> str:
    schema = os.environ.get("VF_ACTOR_SCHEMA", LEGACY_ACTOR_SCHEMA).strip()
    if schema not in SUPPORTED_ACTOR_SCHEMAS:
        raise ValueError(f"unsupported VF_ACTOR_SCHEMA: {schema}")
    return schema


def active_top_level_fields() -> tuple[str, ...]:
    if actor_schema() == REASONS_RATING_ACTOR_SCHEMA:
        return REASONS_RATING_FIELDS
    if actor_schema() == REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA:
        return REASONING_RATING_FIELDS
    return TOP_LEVEL_FIELDS


def strip_qwen35_non_thinking_prefix(text: str) -> str:
    raw = str(text or "")
    if raw.startswith(QWEN35_NON_THINKING_PREFIX):
        return raw[len(QWEN35_NON_THINKING_PREFIX):]
    return raw


def actor_rating_number(value: object) -> float | None:
    number = unbounded_rating_number(value)
    if number is None or not 1.0 <= number <= 5.0:
        return None
    return number


def unbounded_rating_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str) and NUMBER_RE.fullmatch(value.strip()):
        number = float(value.strip())
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def score_number(value: object) -> float | None:
    """Parse a bounded external score without repairing invalid values."""
    return actor_rating_number(value)


def parse_actor_json(text: str) -> dict[str, Any] | None:
    raw = strip_qwen35_non_thinking_prefix(text).strip()
    if not raw:
        return None
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if raw[end:].strip() or not isinstance(payload, dict):
        return None
    return payload


def parse_tokenizable_actor_json(text: str) -> dict[str, Any] | None:
    """Return ordered field values when semantic rating credit can be located."""
    payload = parse_actor_json(text)
    fields = active_top_level_fields()
    if payload is None or tuple(payload) != fields:
        return None
    schema = actor_schema()
    if schema == REASONS_RATING_ACTOR_SCHEMA:
        if not isinstance(payload.get("reasons"), str) or not payload["reasons"].strip():
            return None
    elif schema == REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA:
        # Rating credit remains locatable when the nested reasoning payload is
        # malformed. Editor/Judge eligibility is validated separately.
        pass
    elif not isinstance(payload.get("reason"), str) or not isinstance(payload.get("suggestion"), str):
        return None
    if unbounded_rating_number(payload.get("rating")) is None:
        return None
    return payload


def actor_payload_errors(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["payload:not_object"]
    fields = active_top_level_fields()
    keys = set(payload)
    expected = set(fields)
    errors = [f"top_level:missing:{key}" for key in fields if key not in keys]
    errors.extend(f"top_level:unexpected:{key}" for key in sorted(keys - expected))
    if not errors and tuple(payload) != fields:
        errors.append("top_level:order")
    if errors:
        return errors
    schema = actor_schema()
    if schema == REASONS_RATING_ACTOR_SCHEMA:
        if not isinstance(payload.get("reasons"), str):
            errors.append("reasons:not_string")
        elif not payload["reasons"].strip():
            errors.append("reasons:empty")
    elif schema == REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA:
        reasoning = payload.get("reasoning")
        if not isinstance(reasoning, dict):
            errors.append("reasoning:not_object")
        else:
            reasoning_keys = set(reasoning)
            expected_reasoning = set(REASONING_FIELDS)
            errors.extend(
                f"reasoning:missing:{key}"
                for key in REASONING_FIELDS
                if key not in reasoning_keys
            )
            errors.extend(
                f"reasoning:unexpected:{key}"
                for key in sorted(reasoning_keys - expected_reasoning)
            )
            if not any(error.startswith("reasoning:") for error in errors):
                if tuple(reasoning) != REASONING_FIELDS:
                    errors.append("reasoning:order")
            evidence = reasoning.get("evidence")
            solution = reasoning.get("solution")
            if not isinstance(evidence, str):
                errors.append("evidence:not_string")
            elif not evidence.strip():
                errors.append("evidence:empty")
            if not isinstance(solution, str):
                errors.append("solution:not_string")
            elif not solution.strip():
                errors.append("solution:empty")
    else:
        if not isinstance(payload.get("reason"), str):
            errors.append("reason:not_string")
    if actor_rating_number(payload.get("rating")) is None:
        errors.append("rating:invalid")
    if actor_schema() == LEGACY_ACTOR_SCHEMA and not isinstance(payload.get("suggestion"), str):
        errors.append("suggestion:not_string")
    return errors


def to_internal_actor_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Map the strict actor-visible schema onto stable reward/editor names."""
    schema = actor_schema()
    if schema == REASONS_RATING_ACTOR_SCHEMA:
        return {
            "think": payload["reasons"],
            "rating": payload["rating"],
            "editing": "",
        }
    if schema == REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA:
        reasoning = payload["reasoning"]
        evidence = reasoning["evidence"]
        solution = reasoning["solution"]
        return {
            "think": f"{evidence}\n{solution}",
            "rating": payload["rating"],
            "editing": solution,
            "evidence": evidence,
            "solution": solution,
        }
    return {
        "think": payload["reason"],
        "rating": payload["rating"],
        "editing": payload["suggestion"],
    }


def parse_valid_actor_json(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    payload = parse_actor_json(text)
    errors = actor_payload_errors(payload)
    return (payload if not errors else None), errors


def parse_valid_reasoning_component_json(
    text: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate nested reasoning independently from rating value semantics."""
    payload = parse_actor_json(text)
    errors = actor_payload_errors(payload)
    if actor_schema() != REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA:
        return (payload if not errors else None), errors
    reasoning_errors = [
        error for error in errors if error != "rating:invalid"
    ]
    return (
        payload if payload is not None and not reasoning_errors else None,
        reasoning_errors,
    )
