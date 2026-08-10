from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from judge import score_manifest


GENERATION = dict(score_manifest.EXPECTED_GENERATION)
METADATA = {
    "backend": "e5_qwen35_4b_vllm_judge",
    "model_id": "source-e5-judge-step725",
    "model_path": "/models/judge",
    "model_tree_sha256": "a" * 64,
    "prompt_schema": "e5_training_reasoning_v5",
    "prompt_version": "vf_reasoning_evidence_solution_rating_v5_20260724",
    "prompt_hash": "b" * 64,
    "system_prompt_sha256": "c" * 64,
    "user_prompt_sha256": "d" * 64,
    "generation": GENERATION,
    "deterministic": True,
    "cache_compatible": True,
    "score_acceptance_range": [0.0, 5.0],
}


class MockJudgeHandler(BaseHTTPRequestHandler):
    post_count = 0
    unparsed_paths: set[str] = set()

    def write_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        assert self.path == "/health"
        self.write_json(
            HTTPStatus.OK,
            {
                "ready": True,
                "backend": METADATA["backend"],
                "model_id": METADATA["model_id"],
                "model_path": METADATA["model_path"],
                "model_tree_sha256": METADATA["model_tree_sha256"],
                "prompt_hash": METADATA["prompt_hash"],
                "generation": GENERATION,
                "judger": METADATA,
            },
        )

    def do_POST(self) -> None:
        assert self.path == "/score_image"
        type(self).post_count += 1
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        assert request["repeats"] == 1
        image_path = request["image_path"]
        if image_path in type(self).unparsed_paths:
            status = "unparsed"
            score = None
            errors = ["json"]
            mean = None
            valid_count = 0
            completion = "not-json"
        else:
            status = "success"
            score = 3.25
            errors = []
            mean = 3.25
            valid_count = 1
            completion = (
                '{"reasoning":{"evidence":"Visible evidence.",'
                '"solution":"Apply a restrained correction."},"rating":"3.25"}'
            )
        self.write_json(
            HTTPStatus.OK,
            {
                "status": status,
                "mean": mean,
                "valid_count": valid_count,
                "requested_count": 1,
                "outputs": [
                    {
                        "completion": completion,
                        "score": score,
                        "errors": errors,
                        "rating_text": None if score is None else "3.25",
                        "rating_format_ok": score is not None,
                        "rating_representation": (
                            "invalid"
                            if score is None
                            else "numeric_string_two_decimals"
                        ),
                        "rating_format_warning": None,
                        "rating_prompt_range_ok": score is not None,
                        "rating_range_warning": None,
                        "judge_reasons": "Visible evidence.",
                        "reasoning_evidence": "Visible evidence.",
                        "reasoning_solution": "Apply a restrained correction.",
                        "finish_reason": "stop",
                        "prompt_token_count": 32,
                        "completion_token_count": 18,
                    }
                ],
                "image_path": str(Path(image_path).resolve()),
                "runtime_sec": 0.1,
                "queue_wait_sec": 0.01,
                "judger": METADATA,
            },
        )

    def log_message(self, format: str, *args) -> None:
        return


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def start_server() -> tuple[ThreadingHTTPServer, threading.Thread, int]:
    MockJudgeHandler.post_count = 0
    MockJudgeHandler.unparsed_paths = set()
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockJudgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, int(server.server_address[1])


def stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_scores_manifest_with_contract_fields_and_atomic_output(tmp_path: Path) -> None:
    server, thread, port = start_server()
    try:
        source = tmp_path / "source.jsonl"
        output = tmp_path / "scores.jsonl"
        rows = [
            {
                "sample_id": f"sample-{index}",
                "source_image_path": str(tmp_path / f"image-{index}.png"),
                "source_image_sha256": str(index + 1) * 64,
            }
            for index in range(2)
        ]
        write_manifest(source, rows)
        status = score_manifest.main(
            [
                "--source-manifest",
                str(source),
                "--output",
                str(output),
                "--ports",
                str(port),
            ]
        )
        assert status == 0
        scores = score_manifest.read_jsonl(output)
        assert [row["sample_id"] for row in scores] == ["sample-0", "sample-1"]
        assert MockJudgeHandler.post_count == 2
        for source_row, score in zip(rows, scores):
            assert score["source_image_path"] == source_row["source_image_path"]
            assert score["input_image_sha256"] == source_row["source_image_sha256"]
            assert score["model_id"] == METADATA["model_id"]
            assert score["model_path"] == METADATA["model_path"]
            assert score["model_tree_sha256"] == METADATA["model_tree_sha256"]
            assert score["prompt_schema"] == METADATA["prompt_schema"]
            assert score["prompt_hash"] == METADATA["prompt_hash"]
            assert score["prompt_mode"] == "judge"
            assert score["inference_status"] == "success"
            assert score["parse_ok"] is True
            assert score["model_rating"] == 3.25
            assert score["reasoning_evidence"] == "Visible evidence."
        assert not list(tmp_path.glob(".scores.jsonl.tmp.*"))
    finally:
        stop_server(server, thread)


def test_resume_reuses_successful_rows_without_rescoring(tmp_path: Path) -> None:
    server, thread, port = start_server()
    try:
        source = tmp_path / "source.jsonl"
        output = tmp_path / "scores.jsonl"
        write_manifest(
            source,
            [
                {
                    "sample_id": "sample-0",
                    "source_image_path": str(tmp_path / "image.png"),
                    "source_image_sha256": "e" * 64,
                }
            ],
        )
        command = [
            "--source-manifest",
            str(source),
            "--output",
            str(output),
            "--ports",
            str(port),
            "--resume",
        ]
        assert score_manifest.main(command) == 0
        assert MockJudgeHandler.post_count == 1
        assert score_manifest.main(command) == 0
        assert MockJudgeHandler.post_count == 1
    finally:
        stop_server(server, thread)


def test_unparsed_judge_output_is_retained_as_explicit_failure(tmp_path: Path) -> None:
    server, thread, port = start_server()
    try:
        image_path = str(tmp_path / "unparsed.png")
        MockJudgeHandler.unparsed_paths = {image_path}
        source = tmp_path / "source.jsonl"
        output = tmp_path / "scores.jsonl"
        write_manifest(
            source,
            [
                {
                    "sample_id": "sample-failed",
                    "source_image_path": image_path,
                    "source_image_sha256": "f" * 64,
                }
            ],
        )
        status = score_manifest.main(
            [
                "--source-manifest",
                str(source),
                "--output",
                str(output),
                "--ports",
                str(port),
            ]
        )
        assert status == 1
        [score] = score_manifest.read_jsonl(output)
        assert score["sample_id"] == "sample-failed"
        assert score["inference_status"] == "unparsed"
        assert score["parse_ok"] is False
        assert score["parse_errors"] == ["json"]
        assert score["model_rating"] is None
    finally:
        stop_server(server, thread)
