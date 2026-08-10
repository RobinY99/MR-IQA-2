#!/usr/bin/env python3
"""Run one MR-IQA-2 Actor -> Editor example.

The Editor service must run on the same machine as this script because its
public API accepts a local ``image_path`` and returns a local ``edited_path``.
No Judge is used in this example.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
ACTOR_PLUGIN_ROOT = ROOT / "actor" / "plugin"

# The schema must be selected before prompt_contract is imported because that
# module materializes the public prompt at import time.
os.environ["VF_ACTOR_SCHEMA"] = "reasoning_evidence_solution_rating"
if str(ACTOR_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ACTOR_PLUGIN_ROOT))

from actor_contract import actor_payload_errors, parse_actor_json  # noqa: E402
from prompt_contract import (  # noqa: E402
    ACTOR_SCHEMA,
    ADD_NON_THINKING_PREFIX,
    ENABLE_THINKING,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT_TEXT,
    build_structured_validation_messages,
)


EXPECTED_ACTOR_SCHEMA = "reasoning_evidence_solution_rating"
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
DEFAULT_ACTOR_MODEL = "RobinY99/MR-IQA-2"
DEFAULT_EDITOR_URL = "http://127.0.0.1:8212"
DEFAULT_SEED = 764952063587760

if ACTOR_SCHEMA != EXPECTED_ACTOR_SCHEMA:
    raise RuntimeError(
        f"prompt contract loaded schema {ACTOR_SCHEMA!r}; "
        f"expected {EXPECTED_ACTOR_SCHEMA!r}"
    )


class ActorOutputError(ValueError):
    """Raised when the Actor completion does not satisfy the public schema."""


class EditorServiceError(RuntimeError):
    """Raised when the Editor service does not return a usable image."""


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` with UTF-8 text on the same filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    atomic_write_text(path, serialized + "\n")


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file and expose it at ``destination`` only after copy succeeds."""

    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and source.samefile(destination):
        return
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_valid_actor_output(raw_completion: str) -> dict[str, Any]:
    """Parse and strictly validate the public evidence/solution/rating JSON."""

    payload = parse_actor_json(raw_completion)
    errors = actor_payload_errors(payload)
    if errors:
        raise ActorOutputError("invalid Actor output: " + ", ".join(errors))
    assert isinstance(payload, dict)
    return payload


def build_editor_request(
    *,
    image_path: Path,
    actor_payload: Mapping[str, Any],
    seed: int,
    request_index: int = 0,
) -> dict[str, Any]:
    """Build an Editor request with the Actor solution copied verbatim."""

    reasoning = actor_payload["reasoning"]
    if not isinstance(reasoning, Mapping):
        raise ActorOutputError("invalid Actor output: reasoning:not_object")
    solution = reasoning["solution"]
    if not isinstance(solution, str) or not solution.strip():
        raise ActorOutputError("invalid Actor output: solution:not_string_or_empty")
    return {
        "image_path": str(image_path.resolve(strict=True)),
        "positive_prompt": solution,
        "negative_prompt": "",
        "region_prompt": "",
        "edit_plan": {},
        "request_index": int(request_index),
        "seed": int(seed),
    }


def post_json(url: str, payload: Mapping[str, Any], timeout_sec: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_sec)) as response:
            status_code = int(getattr(response, "status", response.getcode()))
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(body)
        except json.JSONDecodeError:
            error_payload = {}
        detail = (
            error_payload.get("detail")
            or error_payload.get("error")
            or body.strip()
            or exc.reason
        )
        raise EditorServiceError(f"Editor HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise EditorServiceError(f"Editor request failed: {exc.reason}") from exc

    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EditorServiceError(
            f"Editor returned invalid JSON with HTTP status {status_code}"
        ) from exc
    if not isinstance(decoded, dict):
        raise EditorServiceError("Editor response must be one JSON object")
    return decoded


def validate_editor_response(payload: Mapping[str, Any]) -> Path:
    if payload.get("status") != "success":
        detail = payload.get("error") or payload.get("detail") or payload.get("status")
        raise EditorServiceError(f"Editor did not succeed: {detail or 'unknown error'}")
    if payload.get("backend") != "diffusers_flux2":
        raise EditorServiceError(
            f"unexpected Editor backend: {payload.get('backend')!r}"
        )
    edited_value = payload.get("edited_path")
    if not isinstance(edited_value, str) or not edited_value.strip():
        raise EditorServiceError("Editor response has no edited_path")
    edited_path = Path(edited_value).expanduser()
    if not edited_path.is_file():
        raise EditorServiceError(f"Editor output does not exist: {edited_path}")
    return edited_path.resolve()


def _render_actor_prompt(processor: Any, image_path: Path) -> str:
    messages = build_structured_validation_messages(str(image_path))
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    if not ADD_NON_THINKING_PREFIX and rendered.endswith(NON_THINKING_PREFIX):
        rendered = rendered[: -len(NON_THINKING_PREFIX)]
    placeholder_count = sum(
        rendered.count(marker) for marker in ("<|image_pad|>", "<|image|>", "<image>")
    )
    if placeholder_count != 1:
        raise RuntimeError(
            f"Actor prompt rendered {placeholder_count} image placeholders; expected 1"
        )
    return rendered


def _actor_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    model_path = Path(args.actor_model).expanduser()
    subfolder = str(args.actor_subfolder or "").strip()
    if model_path.is_dir() and (model_path / "config.json").is_file():
        subfolder = ""
    if subfolder:
        kwargs["subfolder"] = subfolder
    if args.actor_revision:
        kwargs["revision"] = args.actor_revision
    if args.local_files_only:
        kwargs["local_files_only"] = True
    return kwargs


def generate_actor_completion(image_path: Path, args: argparse.Namespace) -> str:
    """Load the released Actor and generate one deterministic completion."""

    # Heavy ML dependencies stay inside the inference function so contract
    # tests and ``--help`` do not require PyTorch or Transformers.
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_kwargs = _actor_load_kwargs(args)
    processor_kwargs = dict(load_kwargs)
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = int(args.max_pixels)
    processor = AutoProcessor.from_pretrained(args.actor_model, **processor_kwargs)

    dtype: Any
    if args.dtype == "auto":
        dtype = "auto"
    else:
        dtype = getattr(torch, args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.actor_model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        **load_kwargs,
    ).to(args.device).eval()

    rendered = _render_actor_prompt(processor, image_path)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        inputs = processor(
            text=[rendered],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
    inputs = inputs.to(args.device)
    torch.manual_seed(int(args.seed))
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(args.seed))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            use_cache=True,
        )
    input_length = int(inputs["input_ids"].shape[1])
    completion_ids = generated[:, input_length:]
    return processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def run_example(
    args: argparse.Namespace,
    *,
    actor_generator: Callable[[Path, argparse.Namespace], str] = generate_actor_completion,
    editor_post: Callable[[str, Mapping[str, Any], float], dict[str, Any]] = post_json,
) -> dict[str, Any]:
    image_path = Path(args.image).expanduser().resolve(strict=True)
    if not image_path.is_file():
        raise FileNotFoundError(f"input image is not a file: {image_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_suffix = image_path.suffix.lower() or ".png"
    source_copy = output_dir / f"source_image{source_suffix}"
    atomic_copy(image_path, source_copy)

    raw_completion = actor_generator(image_path, args)
    atomic_write_text(output_dir / "actor_raw.txt", raw_completion)
    actor_payload = parse_valid_actor_output(raw_completion)
    atomic_write_json(output_dir / "actor_output.json", actor_payload)

    editor_request = build_editor_request(
        image_path=image_path,
        actor_payload=actor_payload,
        seed=args.seed,
        request_index=args.request_index,
    )
    atomic_write_json(output_dir / "editor_request.json", editor_request)
    edit_url = args.editor_url.rstrip("/") + "/edit"
    editor_response = editor_post(edit_url, editor_request, args.timeout_sec)
    atomic_write_json(output_dir / "editor_response.json", editor_response)
    remote_edited_path = validate_editor_response(editor_response)

    edited_suffix = remote_edited_path.suffix.lower() or ".png"
    edited_copy = output_dir / f"edited_image{edited_suffix}"
    atomic_copy(remote_edited_path, edited_copy)

    provenance = {
        "schema_version": "mr_iqa_2_actor_editor_example_v1",
        "status": "success",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "source_image": source_copy.name,
            "actor_raw": "actor_raw.txt",
            "actor_output": "actor_output.json",
            "editor_request": "editor_request.json",
            "editor_response": "editor_response.json",
            "edited_image": edited_copy.name,
        },
        "actor": {
            "model": args.actor_model,
            "subfolder": args.actor_subfolder,
            "revision": args.actor_revision or None,
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": int(args.max_new_tokens),
            "max_pixels": args.max_pixels,
            "seed": int(args.seed),
            "schema": EXPECTED_ACTOR_SCHEMA,
            "prompt_version": PROMPT_VERSION,
            "enable_thinking": ENABLE_THINKING,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT_TEXT,
        },
        "editor": {
            "url": args.editor_url.rstrip("/"),
            "request_index": int(args.request_index),
            "seed": int(args.seed),
            "solution_forwarded_verbatim": (
                editor_request["positive_prompt"]
                == actor_payload["reasoning"]["solution"]
            ),
            "backend": editor_response.get("backend"),
            "profile_name": editor_response.get("profile_name"),
        },
        "judge": None,
    }
    atomic_write_json(output_dir / "provenance.json", provenance)
    return provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one MR-IQA-2 Actor -> FLUX.2 Klein Editor example (no Judge)."
    )
    parser.add_argument("--image", required=True, help="Source image visible to both processes")
    parser.add_argument("--actor-model", default=DEFAULT_ACTOR_MODEL)
    parser.add_argument(
        "--actor-subfolder",
        default="actor",
        help="Hugging Face model subfolder; ignored for a local model directory containing config.json",
    )
    parser.add_argument("--actor-revision", default="")
    parser.add_argument("--editor-url", default=DEFAULT_EDITOR_URL)
    parser.add_argument("--output-dir", default="outputs/actor_to_editor")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-pixels", type=int, default=196608)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_pixels is not None and args.max_pixels <= 0:
        parser.error("--max-pixels must be positive")
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    provenance = run_example(args)
    print(
        json.dumps(
            {
                "status": provenance["status"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "actor_output": provenance["files"]["actor_output"],
                "edited_image": provenance["files"]["edited_image"],
                "provenance": "provenance.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
