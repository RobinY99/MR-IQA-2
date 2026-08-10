#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))
sys.path.insert(0, str(ROOT / "scripts"))

from actor_contract import strip_qwen35_non_thinking_prefix
from eval_threefield_actor import gold_score, load_rows, resolve_image_path, summarize_results, write_output
from prompt_contract import ENABLE_THINKING, TRAINING_USER_PROMPT, build_training_messages, prompt_metadata


EVALUATION_PROTOCOL = "vf_training_aligned_vllm_v1_20260718"
INFERENCE_BACKEND = "swift_vllm"
MESSAGE_FORMAT = "training_jsonl_messages"
CONTEXT_BOUND_OUTPUT_POLICY = "eos_or_context_window_minus_prompt_tokens"


def parse_optional_max_tokens(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "null", "unlimited", "context"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max_new_tokens must be a positive integer or one of: none, null, "
            "unlimited, context"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max_new_tokens must be positive")
    return parsed


def training_aligned_request_payload(image_path: str) -> dict[str, Any]:
    messages = build_training_messages([image_path])
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise RuntimeError("training-aligned evaluation requires string message content")
    if messages[-1] != {"role": "user", "content": TRAINING_USER_PROMPT}:
        raise RuntimeError("training-aligned user prompt contract mismatch")
    return {
        "messages": messages,
        "images": [image_path],
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }


def request_contract_summary(image_path: str = "image-placeholder.png") -> dict[str, Any]:
    payload = training_aligned_request_payload(image_path)
    return {
        **prompt_metadata(),
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "inference_backend": INFERENCE_BACKEND,
        "message_format": MESSAGE_FORMAT,
        "training_prompt_aligned": True,
        "structured_transformers_messages": False,
        "request_payload": payload,
    }


def make_infer_request(image_path: str) -> Any:
    from swift.infer_engine import InferRequest

    return InferRequest(**training_aligned_request_payload(image_path))


def load_engine(args: argparse.Namespace) -> Any:
    os.environ["MAX_PIXELS"] = str(args.max_pixels)
    os.environ["MIN_PIXELS"] = str(args.min_pixels)
    from swift.infer_engine import VllmEngine

    return VllmEngine(
        args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=max(1, args.batch_size),
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
    )


def request_config(args: argparse.Namespace) -> Any:
    from swift.infer_engine import RequestConfig

    return RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
        return_details=True,
    )


def response_record(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    content = choice.message.content
    text = strip_qwen35_non_thinking_prefix(content if isinstance(content, str) else str(content or "")).strip()
    return {
        "completion": text,
        "finish_reason": choice.finish_reason,
        "prompt_token_count": len(response.prompt_token_ids or []),
        "completion_token_count": len(choice.token_ids or []),
    }


def common_value(summaries: list[dict[str, Any]], key: str) -> Any:
    values = [summary.get(key) for summary in summaries]
    first = values[0] if values else None
    if any(value != first for value in values[1:]):
        raise RuntimeError(f"shard summary mismatch for {key}: {values}")
    return first


def merge_shards(paths: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing = 0
    shard_summaries: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.extend(payload.get("results") or [])
        summary = payload.get("summary") or {}
        shard_summaries.append(summary)
        missing += int(summary.get("num_missing_or_bad_gold", 0))
    for summary in shard_summaries:
        if summary.get("evaluation_protocol") != EVALUATION_PROTOCOL:
            raise RuntimeError(f"refusing to merge non-aligned shard: {summary.get('evaluation_protocol')!r}")
    rows.sort(key=lambda row: int(row.get("index", 0)))
    summary, enriched = summarize_results(
        rows,
        missing=missing,
        extra_summary={
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "inference_backend": common_value(shard_summaries, "inference_backend"),
            "message_format": common_value(shard_summaries, "message_format"),
            "training_prompt_aligned": common_value(shard_summaries, "training_prompt_aligned"),
            "structured_transformers_messages": common_value(
                shard_summaries, "structured_transformers_messages"
            ),
            "chat_template_kwargs": common_value(shard_summaries, "chat_template_kwargs"),
            "sampling_profile": common_value(shard_summaries, "sampling_profile"),
            "model_name_or_path": common_value(shard_summaries, "model_name_or_path"),
            "processor_name_or_path": common_value(shard_summaries, "processor_name_or_path"),
            "data_file": common_value(shard_summaries, "data_file"),
            "image_root": common_value(shard_summaries, "image_root"),
            "num_shards": len(paths),
            "requested_batch_size": common_value(shard_summaries, "requested_batch_size"),
            "batch_generate_exception_count": sum(
                int(summary.get("batch_generate_exception_count", 0)) for summary in shard_summaries
            ),
            "singleton_generate_exception_count": sum(
                int(summary.get("singleton_generate_exception_count", 0)) for summary in shard_summaries
            ),
        },
    )
    return summary, enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--processor_name_or_path")
    parser.add_argument("--data_file")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--merge_inputs", nargs="+")
    parser.add_argument("--audit_request_contract", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=parse_optional_max_tokens, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_pixels", type=int, default=196608)
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.22)
    parser.add_argument("--max_model_len", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge_inputs:
        summary, results = merge_shards(args.merge_inputs)
        write_output(args.output_json, summary, results)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.audit_request_contract:
        summary = request_contract_summary()
        write_output(args.output_json, summary, [])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.model_name_or_path:
        raise SystemExit("--model_name_or_path is required")
    if not args.data_file:
        raise SystemExit("--data_file is required")
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("invalid shard configuration")

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

    engine = load_engine(args)
    config = request_config(args)
    results: list[dict[str, Any]] = []
    batch_generate_exception_count = 0
    singleton_generate_exception_count = 0
    for start in range(0, len(pending), max(1, args.batch_size)):
        batch = pending[start : start + max(1, args.batch_size)]
        requests = [make_infer_request(item[2]) for item in batch]
        try:
            responses = engine.infer(requests, request_config=config, use_tqdm=False)
            generated = [response_record(response) for response in responses]
            if len(generated) != len(batch):
                raise RuntimeError(f"response count mismatch: {len(generated)} != {len(batch)}")
        except Exception as batch_error:
            batch_generate_exception_count += 1
            print(
                f"BATCH_GENERATE_EXCEPTION items={len(batch)} type={type(batch_error).__name__}: {batch_error}",
                file=sys.stderr,
                flush=True,
            )
            generated = []
            for request in requests:
                try:
                    response = engine.infer([request], request_config=config, use_tqdm=False)[0]
                    generated.append(response_record(response))
                except Exception as inner_error:
                    singleton_generate_exception_count += 1
                    generated.append(
                        {
                            "completion": f"ERROR: {type(inner_error).__name__}: {inner_error}",
                            "finish_reason": "error",
                            "prompt_token_count": None,
                            "completion_token_count": None,
                        }
                    )
        for (index, row, image_path, gold), generated_row in zip(batch, generated):
            results.append(
                {
                    "index": index,
                    "image_path": image_path,
                    "gold_score": gold,
                    "row": row,
                    **generated_row,
                }
            )

    sampling_profile = {
        "kind": "deterministic_validation",
        "max_tokens": args.max_new_tokens,
        "output_length_policy": (
            CONTEXT_BOUND_OUTPUT_POLICY
            if args.max_new_tokens is None
            else "fixed_max_tokens"
        ),
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "seed": args.seed,
    }
    summary, enriched = summarize_results(
        results,
        missing=missing,
        extra_summary={
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "inference_backend": INFERENCE_BACKEND,
            "message_format": MESSAGE_FORMAT,
            "training_prompt_aligned": True,
            "structured_transformers_messages": False,
            "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
            "sampling_profile": sampling_profile,
            "model_name_or_path": args.model_name_or_path,
            "processor_name_or_path": args.processor_name_or_path or args.model_name_or_path,
            "data_file": args.data_file,
            "image_root": args.image_root,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "requested_batch_size": max(1, args.batch_size),
            "batch_generate_exception_count": batch_generate_exception_count,
            "singleton_generate_exception_count": singleton_generate_exception_count,
        },
    )
    write_output(args.output_json, summary, enriched)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
