#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


JUDGE_ROOT = Path(__file__).resolve().parent
if str(JUDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(JUDGE_ROOT))

from contract import (  # noqa: E402
    JUDGER_GENERATION,
    JUDGER_MODEL_PATH,
    JUDGER_PROMPT_HASH,
    JUDGER_SYSTEM_PROMPT,
    JUDGER_USER_PROMPT,
    judger_metadata,
    parse_score_payload,
)


NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def strip_non_thinking_prefix(text: str) -> str:
    raw = str(text or "")
    if raw.startswith(NON_THINKING_PREFIX):
        return raw[len(NON_THINKING_PREFIX) :]
    return raw


def build_infer_request_payload(image_path: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": JUDGER_SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>{JUDGER_USER_PROMPT}"},
        ],
        "images": [image_path],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def build_request_config_kwargs() -> dict[str, Any]:
    return {
        key: JUDGER_GENERATION[key]
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "seed",
            "return_details",
        )
    }


def summarize_completions(outputs: list[dict[str, Any] | str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    valid_scores: list[float] = []
    for output in outputs:
        if isinstance(output, dict):
            completion = str(output.get("completion") or "")
            record = dict(output)
        else:
            completion = str(output)
            record = {}
        parsed = parse_score_payload(completion)
        score = parsed["score"]
        errors = parsed["errors"]
        if score is not None:
            valid_scores.append(float(score))
        record.update(
            {
                "completion": completion,
                "score": score,
                "errors": errors,
                "rating_text": parsed["rating_text"],
                "rating_format_ok": parsed["rating_format_ok"],
                "rating_representation": parsed["rating_representation"],
                "rating_format_warning": parsed["rating_format_warning"],
                "rating_prompt_range_ok": parsed["rating_prompt_range_ok"],
                "rating_range_warning": parsed["rating_range_warning"],
                "reasoning_evidence": parsed["reasoning_evidence"],
                "reasoning_solution": parsed["reasoning_solution"],
            }
        )
        records.append(record)
    return {
        "status": "success" if valid_scores else "unparsed",
        "mean": (
            sum(valid_scores) / len(valid_scores)
            if valid_scores
            else None
        ),
        "valid_count": len(valid_scores),
        "requested_count": len(outputs),
        "outputs": records,
    }


class FrozenJudger:
    def __init__(self, model_path: str) -> None:
        if model_path != JUDGER_MODEL_PATH:
            raise RuntimeError(
                "Judge model path does not match the cache-compatible contract: "
                f"{model_path!r} != {JUDGER_MODEL_PATH!r}"
            )

        os.environ["MAX_PIXELS"] = str(JUDGER_GENERATION["max_pixels"])
        os.environ["MIN_PIXELS"] = str(JUDGER_GENERATION["min_pixels"])
        os.environ["IMAGE_MAX_TOKEN_NUM"] = str(
            int(JUDGER_GENERATION["max_pixels"]) // 1024
        )
        os.environ["IMAGE_MIN_TOKEN_NUM"] = str(
            max(1, int(JUDGER_GENERATION["min_pixels"]) // 1024)
        )

        from swift.infer_engine import InferRequest, RequestConfig, VllmEngine

        self._InferRequest = InferRequest
        self._request_config = RequestConfig(**build_request_config_kwargs())
        self._engine = VllmEngine(
            model_path,
            tensor_parallel_size=int(
                JUDGER_GENERATION["tensor_parallel_size"]
            ),
            gpu_memory_utilization=float(
                JUDGER_GENERATION["gpu_memory_utilization"]
            ),
            max_model_len=int(JUDGER_GENERATION["max_model_len"]),
            max_num_seqs=int(JUDGER_GENERATION["max_num_seqs"]),
            enforce_eager=bool(JUDGER_GENERATION["enforce_eager"]),
            limit_mm_per_prompt=dict(
                JUDGER_GENERATION["limit_mm_per_prompt"]
            ),
            seed=int(JUDGER_GENERATION["seed"]),
        )
        self._max_batch_size = int(os.environ.get("VF_JUDGER_MAX_BATCH_SIZE", "1"))
        self._batch_wait_ms = float(os.environ.get("VF_JUDGER_BATCH_WAIT_MS", "0"))
        if not 1 <= self._max_batch_size <= int(JUDGER_GENERATION["max_num_seqs"]):
            raise RuntimeError(
                "Judge max batch size must be in [1, max_num_seqs]: "
                f"batch={self._max_batch_size}, "
                f"max_num_seqs={JUDGER_GENERATION['max_num_seqs']}"
            )
        if not 0 <= self._batch_wait_ms <= 100:
            raise RuntimeError("Judge batch wait must be in [0, 100] milliseconds")
        self._queue: queue.Queue[_ScoreJob] = queue.Queue()
        self._batch_index = 0
        self._worker = threading.Thread(
            target=self._batch_loop,
            name="vf-frozen-judger-batcher",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _completion_record(response: Any) -> dict[str, Any]:
        choice = response.choices[0]
        content = choice.message.content
        completion = strip_non_thinking_prefix(
            content if isinstance(content, str) else str(content or "")
        ).strip()
        return {
            "completion": completion,
            "finish_reason": choice.finish_reason,
            "prompt_token_count": len(response.prompt_token_ids or []),
            "completion_token_count": len(choice.token_ids or []),
        }

    def _batch_loop(self) -> None:
        while True:
            first = self._queue.get()
            jobs = [first]
            deadline = time.perf_counter() + self._batch_wait_ms / 1000.0
            while len(jobs) < self._max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    jobs.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            batch_started = time.perf_counter()
            self._batch_index += 1
            batch_index = self._batch_index
            requests: list[Any] = []
            owners: list[int] = []
            try:
                for owner, job in enumerate(jobs):
                    for _ in range(job.repeats):
                        requests.append(
                            self._InferRequest(
                                **build_infer_request_payload(job.image_path)
                            )
                        )
                        owners.append(owner)
                responses = self._engine.infer(
                    requests,
                    request_config=self._request_config,
                    use_tqdm=False,
                )
                if len(responses) != len(owners):
                    raise RuntimeError(
                        "Judge batched inference response count mismatch: "
                        f"responses={len(responses)}, requests={len(owners)}"
                    )
                grouped: list[list[dict[str, Any]]] = [[] for _ in jobs]
                for owner, response in zip(owners, responses):
                    grouped[owner].append(self._completion_record(response))
                batch_runtime = time.perf_counter() - batch_started
                for job, outputs in zip(jobs, grouped):
                    result = summarize_completions(outputs)
                    result.update(
                        {
                            "image_path": job.image_path,
                            "runtime_sec": time.perf_counter() - job.submitted_at,
                            "queue_wait_sec": batch_started - job.submitted_at,
                            "batch_runtime_sec": batch_runtime,
                            "batch_size": len(requests),
                            "batch_request_count": len(jobs),
                            "batch_index": batch_index,
                            "judger": judger_metadata(),
                        }
                    )
                    job.future.set_result(result)
            except Exception as exc:
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)
            finally:
                for _ in jobs:
                    self._queue.task_done()

    def score_image(self, image_path: str, repeats: int) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Judge input image does not exist: {path}")
        if not 1 <= repeats <= 4:
            raise ValueError("repeats must be in [1, 4]")
        future: Future[dict[str, Any]] = Future()
        self._queue.put(
            _ScoreJob(
                image_path=str(path.resolve()),
                repeats=repeats,
                submitted_at=time.perf_counter(),
                future=future,
            )
        )
        timeout = float(os.environ.get("VF_JUDGER_REQUEST_TIMEOUT_SEC", "900"))
        return future.result(timeout=timeout)

    def batching_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "vf_frozen_judger_dynamic_batch_v1",
            "max_batch_size": self._max_batch_size,
            "batch_wait_ms": self._batch_wait_ms,
            "max_num_seqs": int(JUDGER_GENERATION["max_num_seqs"]),
            "queue_depth": self._queue.qsize(),
        }


class _ScoreJob:
    def __init__(
        self,
        *,
        image_path: str,
        repeats: int,
        submitted_at: float,
        future: Future[dict[str, Any]],
    ) -> None:
        self.image_path = image_path
        self.repeats = repeats
        self.submitted_at = submitted_at
        self.future = future


class JudgerHandler(BaseHTTPRequestHandler):
    server_version = "VFFrozenJudger/2"

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        metadata = judger_metadata()
        self._write_json(
            HTTPStatus.OK,
            {
                "ready": True,
                "backend": metadata["backend"],
                "model_id": metadata["model_id"],
                "model_path": metadata["model_path"],
                "model_tree_sha256": metadata["model_tree_sha256"],
                "prompt_hash": metadata["prompt_hash"],
                "generation": metadata["generation"],
                "batching": self.server.judger.batching_metadata(),  # type: ignore[attr-defined]
                "judger": metadata,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/score_image":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            image_path = payload["image_path"]
            repeats = int(payload.get("repeats", 1))
            result = self.server.judger.score_image(image_path, repeats)  # type: ignore[attr-defined]
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        except Exception as exc:
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        self._write_json(HTTPStatus.OK, result)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (
                self.address_string(),
                self.log_date_time_string(),
                format % args,
            )
        )
        sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", default=JUDGER_MODEL_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    judger = FrozenJudger(args.model_path)
    server = ThreadingHTTPServer((args.host, args.port), JudgerHandler)
    server.judger = judger  # type: ignore[attr-defined]
    metadata = judger_metadata()
    print(
        json.dumps(
            {
                "event": "judger_ready",
                "host": args.host,
                "port": args.port,
                "model_id": metadata["model_id"],
                "model_path": metadata["model_path"],
                "prompt_hash": JUDGER_PROMPT_HASH,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
