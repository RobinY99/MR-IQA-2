from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from token_credit import actor_field_char_spans


def _empty_suggestion_context(
    text: str,
    token_offsets: Sequence[tuple[int, int]],
) -> tuple[int, int, int, str] | None:
    spans = actor_field_char_spans(text)
    if spans is None:
        return None
    suggestion_start, suggestion_end = spans["suggestion"]
    content_start = suggestion_start + 1
    content_end = suggestion_end - 1
    if content_start >= content_end:
        return None

    boundary_tokens = [
        (index, int(start))
        for index, (start, end) in enumerate(token_offsets)
        if end > start and end == content_start
    ]
    if boundary_tokens:
        token_index, token_start = boundary_tokens[-1]
    else:
        containing = [
            (index, int(start))
            for index, (start, end) in enumerate(token_offsets)
            if end > start and start <= content_start < end
        ]
        if not containing:
            return None
        token_index, token_start = containing[0]

    empty_text = text[:content_start] + '"' + text[suggestion_end:]
    return token_index, token_start, content_start - token_start, empty_text[token_start:]


def _top_entries(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [entry for entry in value.values() if isinstance(entry, Mapping)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [entry for entry in value if isinstance(entry, Mapping)]
    return []


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def empty_suggestion_topk_support(
    text: str,
    token_logprobs: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    token_offsets: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    raw = str(text or "")
    spans = actor_field_char_spans(raw)
    base_result = {
        "status": "unparseable",
        "empty_action_in_top_k": False,
        "empty_action_logprob": None,
        "topk_sampling_probability": 0.0,
        "empty_candidate_tokens": [],
        "decision_token_index": None,
        "sampled_token": None,
        "sampled_token_logprob": None,
    }
    if spans is None:
        return base_result

    items = list(token_logprobs)
    pieces = [str(item.get("token") or "") for item in items]
    if token_offsets is None:
        joined = "".join(pieces)
        if not joined.endswith(raw):
            return {**base_result, "status": "token_alignment_error"}
        text_base = len(joined) - len(raw)
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for piece in pieces:
            next_cursor = cursor + len(piece)
            offsets.append((max(0, cursor - text_base), max(0, next_cursor - text_base)))
            cursor = next_cursor
    else:
        offsets = [(int(start), int(end)) for start, end in token_offsets]
        if len(offsets) != len(items):
            return {**base_result, "status": "token_alignment_error"}

    context = _empty_suggestion_context(raw, offsets)
    if context is None:
        return {**base_result, "status": "decision_token_missing"}
    decision_index, _, boundary_width, target_suffix = context
    item = items[decision_index]
    candidates: list[tuple[int, str, float]] = []
    seen: set[str] = set()
    for rank, entry in enumerate(_top_entries(item.get("top_logprobs"))[:top_k], start=1):
        token = str(entry.get("token") or "")
        try:
            logprob = float(entry.get("logprob"))
        except (TypeError, ValueError):
            continue
        if not token or token in seen or not math.isfinite(logprob):
            continue
        if len(token) > boundary_width and target_suffix.startswith(token):
            seen.add(token)
            candidates.append((rank, token, logprob))

    candidate_logprob = _logsumexp([entry[2] for entry in candidates]) if candidates else None
    return {
        "status": "ok",
        "empty_action_in_top_k": bool(candidates),
        "empty_action_logprob": candidate_logprob,
        "topk_sampling_probability": math.exp(candidate_logprob) if candidate_logprob is not None else 0.0,
        "empty_candidate_tokens": [entry[1] for entry in candidates],
        "empty_candidate_ranks": [entry[0] for entry in candidates],
        "decision_token_index": decision_index,
        "sampled_token": str(item.get("token") or ""),
        "sampled_token_logprob": float(item.get("logprob")),
        "target_suffix": target_suffix,
    }
