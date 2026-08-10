from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from statistics import stdev
from typing import Any, Mapping, Sequence

from actor_contract import (
    QWEN35_NON_THINKING_PREFIX,
    REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA,
    REASONS_RATING_ACTOR_SCHEMA,
    active_top_level_fields,
    actor_schema,
    parse_actor_json,
    parse_tokenizable_actor_json,
    parse_valid_actor_json,
)


class _JsonSpanParser:
    """Parse JSON while retaining exact source spans for nested values."""

    def __init__(self, text: str, start: int = 0):
        self.text = text
        self.position = int(start)
        self.decoder = json.JSONDecoder()
        self.value_spans: dict[tuple[str, ...], tuple[int, int]] = {}

    def _skip_whitespace(self) -> None:
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1

    def _expect(self, character: str) -> None:
        self._skip_whitespace()
        if self.position >= len(self.text) or self.text[self.position] != character:
            raise ValueError(f"expected {character!r} at offset {self.position}")
        self.position += 1

    def _decode_primitive(self) -> tuple[Any, int, int]:
        self._skip_whitespace()
        start = self.position
        value, end = self.decoder.raw_decode(self.text, start)
        if isinstance(value, (dict, list)):
            raise ValueError("container must be parsed structurally")
        self.position = end
        return value, start, end

    def _parse_value(self, path: tuple[str, ...]) -> Any:
        self._skip_whitespace()
        start = self.position
        if self.position >= len(self.text):
            raise ValueError("unexpected end of JSON")
        character = self.text[self.position]
        if character == "{":
            value = self._parse_object(path)
        elif character == "[":
            value = self._parse_array(path)
        else:
            value, _, _ = self._decode_primitive()
        end = self.position
        if path:
            self.value_spans[path] = (start, end)
        return value

    def _parse_object(self, path: tuple[str, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self._expect("{")
        self._skip_whitespace()
        if self.position < len(self.text) and self.text[self.position] == "}":
            self.position += 1
            return result
        while True:
            key, _, _ = self._decode_primitive()
            if not isinstance(key, str):
                raise ValueError("JSON object key must be a string")
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            self._expect(":")
            result[key] = self._parse_value(path + (key,))
            self._skip_whitespace()
            if self.position < len(self.text) and self.text[self.position] == ",":
                self.position += 1
                continue
            self._expect("}")
            return result

    def _parse_array(self, path: tuple[str, ...]) -> list[Any]:
        result: list[Any] = []
        self._expect("[")
        self._skip_whitespace()
        if self.position < len(self.text) and self.text[self.position] == "]":
            self.position += 1
            return result
        while True:
            result.append(self._parse_value(path + (str(len(result)),)))
            self._skip_whitespace()
            if self.position < len(self.text) and self.text[self.position] == ",":
                self.position += 1
                continue
            self._expect("]")
            return result

    def parse_document(self) -> Any:
        value = self._parse_value(())
        self._skip_whitespace()
        if self.position != len(self.text):
            raise ValueError("trailing content after JSON document")
        return value


def _first_non_whitespace(text: str, start: int, end: int) -> int | None:
    for index in range(max(0, start), min(len(text), end)):
        if not text[index].isspace():
            return index
    return None


def _first_json_object(text: str) -> tuple[int, int] | None:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return start, end
    return None


def _incomplete_json_prefix_at_eof(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    if not raw.startswith("{"):
        return False
    try:
        json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError as exc:
        if exc.msg.startswith("Unterminated string"):
            return True
        return not raw[exc.pos:].strip()
    return False


def classify_format_boundary(
    text: str,
    *,
    finish_reason: str | None,
    is_truncated: bool,
) -> tuple[str, int | None]:
    raw = str(text or "")
    if bool(is_truncated) or finish_reason == "length":
        return "length_truncated", None

    located = _first_json_object(raw)
    if located is not None:
        object_start, object_end = located
        prefix = _first_non_whitespace(raw, 0, object_start)
        if prefix is not None:
            return "boundary_overrun", prefix
        suffix = _first_non_whitespace(raw, object_end, len(raw))
        if suffix is not None:
            return "boundary_overrun", suffix

    payload, errors = parse_valid_actor_json(raw)
    if payload is not None and not errors and finish_reason == "stop":
        return "correct_stop", None
    if finish_reason == "stop" and _incomplete_json_prefix_at_eof(raw):
        return "premature_stop", None
    return "unlocalized_invalid", None


def build_format_boundary_mask(
    *,
    text: str,
    offsets: Sequence[tuple[int, int]],
    base_format_mask: Sequence[bool],
    advantage: float,
    finish_reason: str | None,
    is_truncated: bool,
    stop_token_mask: Sequence[bool],
) -> tuple[list[bool], str]:
    base = list(base_format_mask)
    stop = list(stop_token_mask)
    if len(offsets) != len(base) or len(stop) != len(base):
        raise ValueError("format boundary mask length mismatch")
    state, _ = classify_format_boundary(
        text,
        finish_reason=finish_reason,
        is_truncated=is_truncated,
    )
    return [visible or terminal for visible, terminal in zip(base, stop)], state


def align_visible_token_offsets(
    *,
    response_ids: Sequence[int],
    visible_response_ids: Sequence[int],
    decoded_token_pieces: Sequence[str],
    decoded_visible_text: str,
    content_text: str,
) -> list[tuple[int, int]]:
    response = list(response_ids)
    visible = list(visible_response_ids)
    if response[:len(visible)] != visible:
        raise ValueError("visible completion tokens are not a response prefix")
    pieces = [str(piece) for piece in decoded_token_pieces]
    if len(pieces) != len(visible):
        raise ValueError("decoded token piece count does not match visible token count")
    if "".join(pieces) != decoded_visible_text:
        raise ValueError("decoded token pieces are not compositional")

    boundaries = [0]
    for piece in pieces:
        boundaries.append(boundaries[-1] + len(piece))
    return _project_decoded_boundaries(
        response_length=len(response),
        visible_length=len(visible),
        boundaries=boundaries,
        decoded_visible_text=decoded_visible_text,
        content_text=content_text,
    )


def align_prefix_decoded_token_offsets(
    *,
    response_ids: Sequence[int],
    visible_response_ids: Sequence[int],
    decoded_prefixes: Sequence[str],
    decoded_visible_text: str,
    content_text: str,
) -> list[tuple[int, int]]:
    response = list(response_ids)
    visible = list(visible_response_ids)
    if response[:len(visible)] != visible:
        raise ValueError("visible completion tokens are not a response prefix")
    prefixes = [str(prefix) for prefix in decoded_prefixes]
    if len(prefixes) != len(visible) + 1:
        raise ValueError("decoded prefix count does not match visible token count")
    if prefixes[0] or prefixes[-1] != decoded_visible_text:
        raise ValueError("decoded prefixes do not terminate at the visible completion")

    boundaries: list[int] = []
    previous = 0
    for prefix in prefixes:
        common = 0
        for left, right in zip(prefix, decoded_visible_text):
            if left != right:
                break
            common += 1
        previous = max(previous, common)
        boundaries.append(previous)
    boundaries[-1] = len(decoded_visible_text)
    return _project_decoded_boundaries(
        response_length=len(response),
        visible_length=len(visible),
        boundaries=boundaries,
        decoded_visible_text=decoded_visible_text,
        content_text=content_text,
    )


def _project_decoded_boundaries(
    *,
    response_length: int,
    visible_length: int,
    boundaries: Sequence[int],
    decoded_visible_text: str,
    content_text: str,
) -> list[tuple[int, int]]:
    if len(boundaries) != visible_length + 1:
        raise ValueError("decoded boundary count does not match visible token count")
    if decoded_visible_text == content_text:
        content_start = 0
    elif decoded_visible_text.endswith(content_text):
        content_start = len(decoded_visible_text) - len(content_text)
    else:
        raise ValueError("decoded visible completion does not end with trainer completion text")
    content_length = len(content_text)

    def project(position: int) -> int:
        return min(content_length, max(0, int(position) - content_start))

    offsets = [
        (project(boundaries[index]), project(boundaries[index + 1]))
        for index in range(visible_length)
    ]
    offsets.extend([(content_length, content_length)] * (response_length - visible_length))
    return offsets


def actor_field_char_spans(text: str) -> dict[str, tuple[int, int]] | None:
    raw = str(text or "")
    schema = actor_schema()
    if schema == REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA:
        payload = parse_actor_json(raw)
        if payload is None or tuple(payload) != active_top_level_fields():
            return None
    elif parse_tokenizable_actor_json(raw) is None:
        return None
    start = len(QWEN35_NON_THINKING_PREFIX) if raw.startswith(QWEN35_NON_THINKING_PREFIX) else 0
    parser = _JsonSpanParser(raw, start=start)
    try:
        parser.parse_document()
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if schema == REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA:
        paths = {
            "reasoning": ("reasoning",),
            "evidence": ("reasoning", "evidence"),
            "solution": ("reasoning", "solution"),
            "rating": ("rating",),
        }
    else:
        paths = {field: (field,) for field in active_top_level_fields()}
    return {
        field: parser.value_spans[path]
        for field, path in paths.items()
        if path in parser.value_spans
    }


def _string_content_span(text: str, span: tuple[int, int]) -> tuple[int, int]:
    start, end = span
    if end - start >= 2 and text[start] == '"' and text[end - 1] == '"':
        return start + 1, end - 1
    return start, end


def _span_mask(
    offsets: Sequence[tuple[int, int]],
    span: tuple[int, int],
) -> list[bool]:
    field_start, field_end = span
    return [
        token_end > field_start and token_start < field_end and token_end > token_start
        for token_start, token_end in offsets
    ]


def build_field_token_masks(
    text: str,
    offsets: Sequence[tuple[int, int]],
) -> dict[str, list[bool]]:
    spans = actor_field_char_spans(text)
    masks = {
        "format": [end > start for start, end in offsets],
        "reasons": [False] * len(offsets),
        "reason": [False] * len(offsets),
        "think": [False] * len(offsets),
        "reasoning": [False] * len(offsets),
        "reasoning_content": [False] * len(offsets),
        "evidence": [False] * len(offsets),
        "evidence_content": [False] * len(offsets),
        "solution": [False] * len(offsets),
        "solution_content": [False] * len(offsets),
        "rating": [False] * len(offsets),
        "rating_content": [False] * len(offsets),
        "suggestion": [False] * len(offsets),
        "editing": [False] * len(offsets),
        "editing_decision": [False] * len(offsets),
        "editing_content": [False] * len(offsets),
    }
    if offsets and not any(masks["format"]):
        masks["format"] = [True] * len(offsets)
    if spans is None:
        return masks
    for field, span in spans.items():
        masks[field] = _span_mask(offsets, span)
    if "evidence" in spans and "solution" in spans:
        evidence_content = _span_mask(offsets, _string_content_span(text, spans["evidence"]))
        solution_content = _span_mask(offsets, _string_content_span(text, spans["solution"]))
        masks["evidence_content"] = evidence_content
        masks["solution_content"] = solution_content
        masks["reasoning_content"] = [
            evidence or solution
            for evidence, solution in zip(evidence_content, solution_content)
        ]
        masks["think"] = list(masks["reasoning_content"])
        masks["editing"] = list(masks["solution"])
        masks["editing_content"] = list(solution_content)
    elif "reasons" in spans:
        masks["think"] = list(masks["reasons"])
    else:
        masks["think"] = list(masks["reason"])
        masks["editing"] = list(masks["suggestion"])
    masks["rating_content"] = _span_mask(
        offsets,
        _string_content_span(text, spans["rating"]),
    )
    if "suggestion" in spans:
        editing_start, editing_end = spans["suggestion"]
        content_start = editing_start + 1
        content_end = editing_end - 1
        decision_start = editing_start
        decision_end = content_start
        masks["editing_decision"] = [
            token_end > decision_start and token_start < decision_end and token_end > token_start
            for token_start, token_end in offsets
        ]
        if content_start < content_end:
            masks["editing_content"] = [
                token_end > content_start and token_start < content_end and token_end > token_start
                for token_start, token_end in offsets
            ]
    return masks


def compute_component_group_advantages(
    rows: Sequence[Mapping[str, Any]],
    *,
    epsilon: float = 1e-6,
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = [defaultdict(float) for _ in rows]  # type: ignore[list-item]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get("group_id", ""))].append(index)

    for indices in groups.values():
        component_names = sorted(
            {
                str(name)
                for index in indices
                for name in (rows[index].get("rewards") or {}).keys()
            }
        )
        for component in component_names:
            for index in indices:
                result[index][component] = 0.0
            eligible: list[tuple[int, float]] = []
            for index in indices:
                eligibility = rows[index].get("eligibility") or {}
                if not bool(eligibility.get(component, False)):
                    continue
                value = float((rows[index].get("rewards") or {}).get(component, 0.0))
                if not math.isfinite(value):
                    raise ValueError(f"non-finite reward for {component}")
                eligible.append((index, value))
            if len(eligible) < 2:
                for index, _ in eligible:
                    fallback = float(
                        ((rows[index].get("fallback_advantages") or {}).get(component, 0.0))
                    )
                    if not math.isfinite(fallback):
                        raise ValueError(f"non-finite fallback advantage for {component}")
                    if component in {"format_a0", "format_a1", "edit_gate"}:
                        fallback = 0.0
                    result[index][component] = fallback
                continue
            values = [value for _, value in eligible]
            scale = stdev(values)
            if not math.isfinite(scale) or scale <= epsilon:
                for index, _ in eligible:
                    fallback = float(
                        ((rows[index].get("fallback_advantages") or {}).get(component, 0.0))
                    )
                    if not math.isfinite(fallback):
                        raise ValueError(f"non-finite fallback advantage for {component}")
                    if component in {"format_a0", "format_a1", "edit_gate"}:
                        fallback = 0.0
                    result[index][component] = fallback
                continue
            mean = math.fsum(values) / len(values)
            for index, value in eligible:
                result[index][component] = (value - mean) / (scale + epsilon)
    return [dict(row) for row in result]


_ROLLOUT_COMPONENT_MASKS = {
    "a0": {
        "dapo_policy": "format",
        "grpo_policy": "format",
        "format_a0": "format",
        "rating0": "rating_content",
        "reasoning": "reasoning_content",
        "soft_overlong": "format",
        "edit_gate": "editing_decision",
        "edit_gain": "editing_content",
    },
    "a1": {
        "format_a1": "format",
        "delta_margin": "rating_content",
        "rating1_anchor": "rating_content",
    },
}


COMPONENT_CREDIT_MASK_MODES = {"field", "completion"}


def component_credit_mask_mode() -> str:
    mode = os.environ.get("VF_COMPONENT_CREDIT_MASK_MODE", "field").strip().lower()
    if mode not in COMPONENT_CREDIT_MASK_MODES:
        raise ValueError(
            "VF_COMPONENT_CREDIT_MASK_MODE must be one of "
            f"{sorted(COMPONENT_CREDIT_MASK_MODES)}, got {mode!r}"
        )
    return mode


def build_rollout_component_credit(
    rollout: str,
    masks: Mapping[str, Sequence[bool]],
    component_advantages: Mapping[str, float],
    weights: Mapping[str, float],
    eligibility: Mapping[str, bool] | None = None,
    *,
    format_mask_override: Sequence[bool] | None = None,
) -> dict[str, dict[str, Any]]:
    if rollout not in _ROLLOUT_COMPONENT_MASKS:
        raise ValueError(f"unsupported rollout: {rollout}")
    length = len(masks["format"])
    mask_mode = component_credit_mask_mode()
    result: dict[str, dict[str, Any]] = {}
    for component, mask_name in _ROLLOUT_COMPONENT_MASKS[rollout].items():
        if component == "reasoning" and component not in {
            *component_advantages,
            *weights,
            *(eligibility or {}),
        }:
            continue
        # DAPO assigns one scalar group advantage to every policy token in the
        # completion, including a trailing stop token with a zero-width text
        # offset. Field-level rewards keep their narrower semantic masks.
        mask = (
            [True] * length
            if mask_mode == "completion"
            or component in {"dapo_policy", "grpo_policy", "soft_overlong"}
            else list(masks[mask_name])
        )
        if len(mask) != length:
            raise ValueError(f"mask length mismatch for {mask_name}")
        eligible = True if eligibility is None else bool(eligibility.get(component, False))
        advantage = float(component_advantages.get(component, 0.0))
        if not eligible:
            mask = [False] * length
        elif (
            mask_mode == "field"
            and component in {"format_a0", "format_a1"}
            and format_mask_override is not None
        ):
            mask = list(format_mask_override)
            if len(mask) != length:
                raise ValueError("format mask override length mismatch")
        result[component] = {
            "mask": mask,
            "advantage": advantage,
            "weight": float(weights.get(component, 1.0)),
            "eligible": eligible,
            "mask_mode": mask_mode,
        }
    return result


def compose_rollout_token_advantages(
    rollout: str,
    masks: Mapping[str, Sequence[bool]],
    component_advantages: Mapping[str, float],
    weights: Mapping[str, float],
) -> list[float]:
    length = len(masks["format"])
    result = [0.0] * length

    for payload in build_rollout_component_credit(rollout, masks, component_advantages, weights).values():
        value = float(payload["advantage"]) * float(payload["weight"])
        if value == 0.0:
            continue
        mask = payload["mask"]
        for index, enabled in enumerate(mask):
            if enabled:
                result[index] += value
    if not all(math.isfinite(value) for value in result):
        raise ValueError("non-finite token advantage")
    return result
