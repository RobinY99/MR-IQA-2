#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

from actor_contract import (
    REASONS_RATING_ACTOR_SCHEMA,
    actor_schema,
    actor_payload_errors,
    actor_rating_number,
    parse_actor_json,
    to_internal_actor_payload,
)
from prompt_contract import (
    ADD_NON_THINKING_PREFIX,
    ENABLE_THINKING,
    PROMPT_HASH,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT_TEXT,
    build_structured_validation_messages,
    prompt_metadata,
)


def load_rows(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "train_records", "test_records"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
    raise ValueError(f"Unsupported data structure in {path}")


def resolve_image_path(row: dict[str, Any], image_root: str | None) -> str | None:
    for key in ("image", "image_path", "img_path", "source_image"):
        value = row.get(key)
        if value is None:
            continue
        image = str(value).strip()
        candidates = [image]
        if image_root:
            candidates.extend([os.path.join(image_root, image), os.path.join(image_root, os.path.basename(image))])
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return None


def gold_score(row: dict[str, Any]) -> float | None:
    candidates = [
        row.get("score_norm"),
        row.get("normalized_score"),
        row.get("gt_score_norm"),
        row.get("score"),
        row.get("human_score"),
        row.get("mos"),
        row.get("rating"),
        row.get("quality_score"),
        row.get("source_score"),
    ]
    annotation = row.get("human_annotation")
    if isinstance(annotation, dict):
        candidates.append(annotation.get("normalized_score"))
    target = row.get("target")
    if isinstance(target, dict):
        candidates.append(target.get("score"))
    for value in candidates:
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and 1.0 <= score <= 5.0:
            return score
    return None


def build_validation_messages(image_path: str) -> list[dict[str, Any]]:
    return build_structured_validation_messages(image_path)


NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def apply_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    enable_thinking: bool = ENABLE_THINKING,
    add_non_thinking_prefix: bool = ADD_NON_THINKING_PREFIX,
) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if not enable_thinking:
        rendered = processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
        if not add_non_thinking_prefix and rendered.endswith(NON_THINKING_PREFIX):
            rendered = rendered[: -len(NON_THINKING_PREFIX)]
        return rendered
    try:
        return processor.apply_chat_template(messages, enable_thinking=True, **kwargs)
    except TypeError:
        try:
            return processor.apply_chat_template(messages, chat_template_kwargs={"enable_thinking": True}, **kwargs)
        except TypeError:
            return processor.apply_chat_template(messages, **kwargs)


def rendered_image_placeholder_count(rendered: str) -> int:
    markers = ("<|image_pad|>", "<|image|>", "<image>")
    return sum(str(rendered).count(marker) for marker in markers)


def audit_prompt_rendering(processor: Any, image_path: str = "image-placeholder.png") -> dict[str, Any]:
    messages = build_validation_messages(image_path)
    rendered = apply_chat_template(
        processor,
        messages,
        enable_thinking=ENABLE_THINKING,
        add_non_thinking_prefix=ADD_NON_THINKING_PREFIX,
    )
    count = rendered_image_placeholder_count(rendered)
    if count != 1:
        raise RuntimeError(f"validation prompt rendered {count} image placeholders, expected exactly 1")
    return {**prompt_metadata(), "rendered_image_placeholder_count": count, "rendered_prompt": rendered}


def load_image(path: str) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def generate_batch(model: Any, processor: Any, image_paths: list[str], args: Any) -> list[str]:
    import torch

    texts = []
    images = []
    for image_path in image_paths:
        messages = build_validation_messages(image_path)
        rendered = apply_chat_template(
            processor,
            messages,
            enable_thinking=args.enable_thinking,
            add_non_thinking_prefix=args.add_non_thinking_prefix,
        )
        placeholder_count = rendered_image_placeholder_count(rendered)
        if placeholder_count != 1:
            raise RuntimeError(
                f"validation prompt rendered {placeholder_count} image placeholders for {image_path}, expected exactly 1"
            )
        texts.append(rendered)
        images.append(load_image(image_path))
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "use_cache": True,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    for key in ("top_p", "top_k", "min_p", "repetition_penalty"):
        value = getattr(args, key, None)
        if value is not None:
            generation_kwargs[key] = value
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    completion_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return math.nan
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator else math.nan


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def correlations(gold: list[float], pred: list[float]) -> tuple[float, float]:
    return _pearson(gold, pred), _pearson(_ranks(gold), _ranks(pred))


def summarize_results(
    results: list[dict[str, Any]], missing: int = 0, extra_summary: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    for row in results:
        item = dict(row)
        completion = str(item.get("completion") or "")
        payload = parse_actor_json(completion)
        errors = actor_payload_errors(payload)
        rating = actor_rating_number(payload.get("rating")) if isinstance(payload, dict) else None
        internal_payload = (
            to_internal_actor_payload(payload)
            if isinstance(payload, dict) and not errors
            else None
        )
        editing = str(internal_payload.get("editing") or "").strip() if internal_payload else ""
        reasons = ""
        if isinstance(payload, dict) and not errors:
            reasons = str(payload.get("reasons") or payload.get("reason") or "").strip()
        item.update(
            {
                "parsed_payload": payload,
                "format_errors": errors,
                "json_parse_success": payload is not None,
                "actor_format_success": not errors,
                "pred_score": rating,
                "reasons": reasons,
                "edit_requested": bool(editing),
            }
        )
        enriched.append(item)

    valid = [row for row in enriched if row["pred_score"] is not None and row.get("gold_score") is not None]
    gold = [float(row["gold_score"]) for row in valid]
    pred = [float(row["pred_score"]) for row in valid]
    plcc, srcc = correlations(gold, pred)
    n = len(enriched)
    active_schema = actor_schema()
    instructions = {
        str(row["parsed_payload"].get("suggestion") or "").strip()
        for row in enriched
        if isinstance(row.get("parsed_payload"), dict)
        and str(row["parsed_payload"].get("suggestion") or "").strip()
    }
    low_target = [
        row for row in enriched
        if row.get("gold_score") is not None and float(row["gold_score"]) <= 3.0
    ]
    low_target_edits = sum(bool(row["edit_requested"]) for row in low_target)
    unique_completion_count = len({str(row.get("completion") or "").strip() for row in enriched})
    summary: dict[str, Any] = {
        **prompt_metadata(),
        "num_total": n,
        "num_missing_or_bad_gold": int(missing),
        "json_parse_success_rate": sum(row["json_parse_success"] for row in enriched) / n if n else math.nan,
        "actor_format_success_rate": sum(row["actor_format_success"] for row in enriched) / n if n else math.nan,
        "rating_parse_success_rate": len(valid) / n if n else math.nan,
        "num_valid_rating": len(valid),
        "plcc": plcc,
        "srcc": srcc,
        "mae": sum(abs(x - y) for x, y in zip(gold, pred)) / len(valid) if valid else math.nan,
        "unique_completion_count": unique_completion_count,
        "unique_completion_ratio": unique_completion_count / n if n else math.nan,
    }
    if active_schema == REASONS_RATING_ACTOR_SCHEMA:
        nonempty_reasons = [row["reasons"] for row in enriched if row["reasons"]]
        summary.update(
            {
                "reasons_nonempty_rate": len(nonempty_reasons) / n if n else math.nan,
                "unique_reasons_count": len(set(nonempty_reasons)),
                "no_correction_statement_rate": (
                    sum("no correction is necessary" in value.lower() for value in nonempty_reasons) / n
                    if n
                    else math.nan
                ),
            }
        )
    else:
        summary.update(
            {
                "edit_request_rate": sum(row["edit_requested"] for row in enriched) / n if n else math.nan,
                "no_edit_rate": 1.0 - sum(row["edit_requested"] for row in enriched) / n if n else math.nan,
                "low_target_threshold": 3.0,
                "low_target_count": len(low_target),
                "num_low_target_edits": low_target_edits,
                "low_target_edit_request_rate": low_target_edits / len(low_target) if low_target else math.nan,
                "unique_instruction_count": len(instructions),
            }
        )
    if extra_summary:
        summary.update(extra_summary)
    return summary, enriched


def model_class(model_path: str) -> Any:
    from transformers import AutoConfig, AutoModelForImageTextToText

    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError:
        AutoModelForMultimodalLM = None
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        identity = " ".join(
            [model_path, config.__class__.__name__, str(getattr(config, "model_type", ""))]
        ).lower()
    except Exception:
        identity = model_path.lower()
    if any(token in identity for token in ("qwen3.5", "qwen3_5", "qwen35")) and AutoModelForMultimodalLM:
        return AutoModelForMultimodalLM
    return AutoModelForImageTextToText


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch

    processor = load_processor(
        args.processor_name_or_path or args.model_name_or_path,
        max_pixels=args.max_pixels,
    )
    model = model_class(args.model_name_or_path).from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).eval()
    return model, processor


def load_processor(path: str, max_pixels: int | None = None) -> Any:
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if max_pixels is not None:
        kwargs["max_pixels"] = int(max_pixels)
    return AutoProcessor.from_pretrained(path, **kwargs)


def merge_shards(paths: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing = 0
    shard_summaries: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.extend(payload.get("results") or [])
        shard_summary = payload.get("summary") or {}
        shard_summaries.append(shard_summary)
        missing += int(shard_summary.get("num_missing_or_bad_gold", 0))
    rows.sort(key=lambda row: int(row.get("index", 0)))
    requested_batches = {
        value for value in (item.get("requested_batch_size") for item in shard_summaries)
        if value is not None
    }
    return summarize_results(
        rows,
        missing=missing,
        extra_summary={
            "num_shards": len(paths),
            "requested_batch_size": next(iter(requested_batches)) if len(requested_batches) == 1 else None,
            "batch_generate_exception_count": sum(
                int(item.get("batch_generate_exception_count", 0)) for item in shard_summaries
            ),
            "singleton_generate_exception_count": sum(
                int(item.get("singleton_generate_exception_count", 0)) for item in shard_summaries
            ),
        },
    )


def write_output(path: str, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--processor_name_or_path")
    parser.add_argument("--data_file")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--merge_inputs", nargs="+")
    parser.add_argument("--audit_prompt_rendering", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--min_p", type=float)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--add_non_thinking_prefix", dest="add_non_thinking_prefix", action="store_true")
    parser.add_argument("--no_add_non_thinking_prefix", dest="add_non_thinking_prefix", action="store_false")
    parser.set_defaults(add_non_thinking_prefix=ADD_NON_THINKING_PREFIX)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_pixels", type=int)
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge_inputs:
        summary, results = merge_shards(args.merge_inputs)
        write_output(args.output_json, summary, results)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.model_name_or_path:
        raise SystemExit("--model_name_or_path is required")
    if args.audit_prompt_rendering:
        processor = load_processor(args.processor_name_or_path or args.model_name_or_path)
        audit = audit_prompt_rendering(processor)
        write_output(args.output_json, audit, [])
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    if not args.data_file:
        raise SystemExit("--data_file is required")
    model, processor = load_model_and_processor(args)

    source_rows = load_rows(args.data_file)
    selected = [(index, row) for index, row in enumerate(source_rows) if index % args.num_shards == args.shard_id]
    pending: list[tuple[int, dict[str, Any], str, float]] = []
    missing = 0
    for index, row in selected:
        image_path = resolve_image_path(row, args.image_root)
        gold = gold_score(row)
        if image_path is None or gold is None:
            missing += 1
            continue
        pending.append((index, row, image_path, gold))

    import torch

    results: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    batch_generate_exception_count = 0
    singleton_generate_exception_count = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        try:
            completions = generate_batch(model, processor, [item[2] for item in batch], args)
        except Exception as batch_error:
            batch_generate_exception_count += 1
            print(
                f"BATCH_GENERATE_EXCEPTION items={len(batch)} type={type(batch_error).__name__}: {batch_error}",
                file=sys.stderr,
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            completions = []
            for _, _, image_path, _ in batch:
                try:
                    completions.append(generate_batch(model, processor, [image_path], args)[0])
                except Exception as inner_error:
                    singleton_generate_exception_count += 1
                    completions.append(f"ERROR: {type(inner_error).__name__}: {inner_error}")
        for (index, row, image_path, gold), completion in zip(batch, completions):
            results.append(
                {
                    "index": index,
                    "image_path": image_path,
                    "gold_score": gold,
                    "completion": completion,
                    "row": row,
                }
            )

    summary, enriched = summarize_results(
        results,
        missing=missing,
        extra_summary={
            "model_name_or_path": args.model_name_or_path,
            "processor_name_or_path": args.processor_name_or_path or args.model_name_or_path,
            "data_file": args.data_file,
            "image_root": args.image_root,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "requested_batch_size": batch_size,
            "batch_generate_exception_count": batch_generate_exception_count,
            "singleton_generate_exception_count": singleton_generate_exception_count,
        },
    )
    write_output(args.output_json, summary, enriched)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
