#!/usr/bin/env python3
"""Assess, edit, and evaluate one image with the released MR-IQA-2 models.

Actor, Editor, and Judge run sequentially, so all three stages can reuse one
physical GPU without sharing model memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ID = "RobinY99/MR-IQA-2"
SEED = 764952063587760
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MR-IQA-2 Actor, Editor, and Judge on one GPU."
    )
    parser.add_argument("image", help="Input image")
    parser.add_argument("--output-dir", default="outputs/quick_start")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index")
    parser.add_argument(
        "--cuda-home",
        default="",
        help="CUDA toolkit root used for runtime extension compilation",
    )
    parser.add_argument("--actor-env", default="mr_iqa_actor_judge")
    parser.add_argument("--editor-env", default="mr_iqa_editor")
    parser.add_argument("--actor-python", default="")
    parser.add_argument("--editor-python", default="")
    parser.add_argument("--actor-model", default=REPO_ID)
    parser.add_argument("--actor-subfolder", default="actor")
    parser.add_argument("--judge-model", default=REPO_ID)
    parser.add_argument("--judge-subfolder", default="judge")
    parser.add_argument(
        "--editor-model",
        default="",
        help="Local Editor directory; omitted means REPO_ID/editor",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Optional Hugging Face revision; defaults to the current model release",
    )
    parser.add_argument("--max-pixels", type=int, default=196608)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("actor", "editor", "judge"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if args.max_pixels < 256:
        parser.error("--max-pixels must be at least 256")
    return args


def _actor_stage(args: argparse.Namespace, image_path: Path, output_dir: Path) -> None:
    from actor_to_editor import (
        atomic_write_json,
        atomic_write_text,
        generate_actor_completion,
        parse_valid_actor_output,
    )

    actor_args = argparse.Namespace(
        actor_model=args.actor_model,
        actor_subfolder=args.actor_subfolder,
        actor_revision=(
            ""
            if Path(args.actor_model).expanduser().is_dir()
            else args.revision
        ),
        device="cuda:0",
        dtype="bfloat16",
        attn_implementation="sdpa",
        max_new_tokens=512,
        max_pixels=args.max_pixels,
        seed=SEED,
        local_files_only=args.local_files_only,
    )
    completion = generate_actor_completion(image_path, actor_args)
    payload = parse_valid_actor_output(completion)
    atomic_write_text(output_dir / "actor_raw.txt", completion)
    atomic_write_json(output_dir / "assessment.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _editor_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    scale = min(1.0, math.sqrt(float(max_pixels) / float(width * height)))
    resized_width = max(16, int(width * scale) // 16 * 16)
    resized_height = max(16, int(height * scale) // 16 * 16)
    return resized_width, resized_height


def _editor_stage(args: argparse.Namespace, image_path: Path, output_dir: Path) -> None:
    import torch
    from diffusers import Flux2KleinPipeline
    from huggingface_hub import snapshot_download
    from PIL import Image

    from actor_to_editor import atomic_write_json

    assessment_path = output_dir / "assessment.json"
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    reasoning = assessment.get("reasoning")
    if not isinstance(reasoning, dict):
        raise ValueError("assessment.json has no reasoning object")
    solution = reasoning.get("solution")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("assessment.json has no usable solution")

    if args.editor_model:
        editor_path = Path(args.editor_model).expanduser().resolve(strict=True)
    else:
        download_kwargs: dict[str, Any] = {
            "repo_id": REPO_ID,
            "allow_patterns": ["editor/**"],
            "local_files_only": args.local_files_only,
        }
        if args.revision:
            download_kwargs["revision"] = args.revision
        snapshot = Path(
            snapshot_download(**download_kwargs)
        )
        editor_path = snapshot / "editor"
    editor = Flux2KleinPipeline.from_pretrained(
        editor_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    edit_width, edit_height = _editor_size(
        source.width,
        source.height,
        args.max_pixels,
    )
    editor_input = source.resize(
        (edit_width, edit_height),
        Image.Resampling.LANCZOS,
    )
    with torch.inference_mode():
        edited = editor(
            prompt=solution,
            image=editor_input,
            width=edit_width,
            height=edit_height,
            num_inference_steps=4,
            sigmas=[1.0, 0.75, 0.5, 0.25],
            guidance_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        ).images[0].convert("RGB")
    if edited.size != source.size:
        edited = edited.resize(source.size, Image.Resampling.LANCZOS)

    edited_path = output_dir / "edited.png"
    edited.save(edited_path)
    result = {
        "input_image": str(image_path),
        "assessment": assessment,
        "edited_image": str(edited_path),
        "actor_model": (
            f"{args.actor_model}/{args.actor_subfolder}"
            if args.actor_subfolder and not Path(args.actor_model).expanduser().is_dir()
            else args.actor_model
        ),
        "editor_model": args.editor_model or f"{REPO_ID}/editor",
        "revision": args.revision or None,
        "seed": SEED,
        "original_size": [source.width, source.height],
        "inference_size": [edit_width, edit_height],
        "solution_forwarded_verbatim": (
            solution == assessment["reasoning"]["solution"]
        ),
        "judge": None,
    }
    atomic_write_json(output_dir / "result.json", result)


def _quality_delta(original: dict[str, Any], edited: dict[str, Any]) -> float:
    original_score = original.get("mean")
    edited_score = edited.get("mean")
    if not isinstance(original_score, (int, float)):
        raise ValueError("Judge did not return a valid score for the input image")
    if not isinstance(edited_score, (int, float)):
        raise ValueError("Judge did not return a valid score for the edited image")
    return float(edited_score) - float(original_score)


def _resolve_model_subfolder(
    model: str,
    subfolder: str,
    revision: str,
    local_files_only: bool,
) -> Path:
    from huggingface_hub import snapshot_download

    local_path = Path(model).expanduser()
    if local_path.is_dir():
        candidate = (
            local_path
            if (local_path / "config.json").is_file()
            else local_path / subfolder
        )
        return candidate.resolve(strict=True)
    download_kwargs: dict[str, Any] = {
        "repo_id": model,
        "allow_patterns": [f"{subfolder}/**"],
        "local_files_only": local_files_only,
    }
    if revision:
        download_kwargs["revision"] = revision
    snapshot = Path(snapshot_download(**download_kwargs))
    return (snapshot / subfolder).resolve(strict=True)


def _judge_stage(args: argparse.Namespace, image_path: Path, output_dir: Path) -> None:
    from actor_to_editor import atomic_write_json

    judge_path = _resolve_model_subfolder(
        args.judge_model,
        args.judge_subfolder,
        args.revision,
        args.local_files_only,
    )
    os.environ["VF_JUDGE_MODEL_PATH"] = str(judge_path)
    os.environ["VF_JUDGE_MODEL_ID"] = "mr-iqa-2-e5-judge"
    os.environ["VF_JUDGE_PROMPT_SCHEMA"] = "e5_training_reasoning_v5"
    os.environ["VF_JUDGER_MAX_BATCH_SIZE"] = "1"

    from judge.server import FrozenJudger

    edited_path = (output_dir / "edited.png").resolve(strict=True)
    judger = FrozenJudger(str(judge_path))
    original = judger.score_image(str(image_path), repeats=1)
    edited = judger.score_image(str(edited_path), repeats=1)
    delta = _quality_delta(original, edited)
    evaluation = {
        "j0": original["mean"],
        "j1": edited["mean"],
        "j1_minus_j0": delta,
        "original": original,
        "edited": edited,
        "judge_model": str(judge_path),
        "revision": args.revision or None,
    }
    atomic_write_json(output_dir / "evaluation.json", evaluation)

    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["judge"] = evaluation
    atomic_write_json(result_path, result)


def _stage_command(
    *,
    conda: str,
    environment: str,
    python_executable: str,
    stage: str,
    args: argparse.Namespace,
    image_path: Path,
    output_dir: Path,
) -> list[str]:
    command = (
        [python_executable]
        if python_executable
        else [conda, "run", "--no-capture-output", "-n", environment, "python"]
    )
    command.extend(
        [
            str(Path(__file__).resolve()),
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--gpu",
            "0",
            "--max-pixels",
            str(args.max_pixels),
            "--actor-model",
            args.actor_model,
            "--actor-subfolder",
            args.actor_subfolder,
            "--judge-model",
            args.judge_model,
            "--judge-subfolder",
            args.judge_subfolder,
            "--editor-model",
            args.editor_model,
            "--stage",
            stage,
        ]
    )
    if args.local_files_only:
        command.append("--local-files-only")
    if args.revision:
        command.extend(["--revision", args.revision])
    return command


def run_sequential(
    args: argparse.Namespace,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    image_path = Path(args.image).expanduser().resolve(strict=True)
    if not image_path.is_file():
        raise FileNotFoundError(f"input image is not a file: {image_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conda = shutil.which("conda") or ""
    if not conda and (not args.actor_python or not args.editor_python):
        raise RuntimeError("conda is required to run the two validated environments")

    child_environment = os.environ.copy()
    child_environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    child_environment["PYTHONUNBUFFERED"] = "1"
    cuda_bin: Path | None = None
    effective_cuda_home = args.cuda_home or child_environment.get("CUDA_HOME", "")
    if effective_cuda_home:
        cuda_home = Path(effective_cuda_home).expanduser().resolve(strict=True)
        nvcc = cuda_home / "bin" / "nvcc"
        if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
            raise RuntimeError(f"CUDA toolkit has no executable nvcc: {nvcc}")
        child_environment["CUDA_HOME"] = str(cuda_home)
        cuda_bin = nvcc.parent

    for stage, environment, python_executable in (
        ("actor", args.actor_env, args.actor_python),
        ("editor", args.editor_env, args.editor_python),
        ("judge", args.actor_env, args.actor_python),
    ):
        stage_environment = child_environment.copy()
        resolved_python = python_executable
        if python_executable:
            python_path = Path(
                os.path.abspath(os.path.expanduser(python_executable))
            )
            if not python_path.is_file() or not os.access(python_path, os.X_OK):
                raise RuntimeError(f"Python interpreter is not executable: {python_path}")
            resolved_python = str(python_path)
            stage_environment["PATH"] = os.pathsep.join(
                [str(python_path.parent), stage_environment.get("PATH", "")]
            )
        if cuda_bin is not None:
            stage_environment["PATH"] = os.pathsep.join(
                [str(cuda_bin), stage_environment.get("PATH", "")]
            )
        command_runner(
            _stage_command(
                conda=conda,
                environment=environment,
                python_executable=resolved_python,
                stage=stage,
                args=args,
                image_path=image_path,
                output_dir=output_dir,
            ),
            check=True,
            env=stage_environment,
        )
    return json.loads((output_dir / "result.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image_path = Path(args.image).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "actor":
        _actor_stage(args, image_path, output_dir)
        return 0
    if args.stage == "editor":
        _editor_stage(args, image_path, output_dir)
        return 0
    if args.stage == "judge":
        _judge_stage(args, image_path, output_dir)
        return 0

    result = run_sequential(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
