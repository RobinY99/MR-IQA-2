#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))
sys.path.insert(0, str(ROOT / "scripts"))

from actor_contract import (
    LEGACY_ACTOR_SCHEMA,
    REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA,
)
from editor_judge_contract import (
    EDITOR_PROMPT_TEMPLATE_HASH,
    EDITOR_PROMPT_VERSION,
    EDITOR_SEMANTIC_GUARDRAIL,
)
from frozen_judger_contract import (
    JUDGER_MODEL_ID,
    JUDGER_MODEL_PATH,
    JUDGER_MODEL_TREE_SHA256,
    JUDGER_PROMPT_HASH,
)
from original_score_cache import EXPECTED_CACHE_SHA256, OriginalScoreCache
from token_credit import component_credit_mask_mode
from prompt_contract import (
    ACTOR_SCHEMA,
    PROMPT_HASH,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TRAINING_USER_PROMPT,
    prompt_metadata,
)
from checkpoint_manifest import resolve_promoted_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_sha256(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for directory in (root / "plugin", root / "scripts")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_wandb_mode(*, smoke: bool, mode: str | None) -> str:
    """Validate logging without requiring network access for reproducibility."""
    if smoke:
        if mode != "disabled":
            raise RuntimeError("WANDB_MODE must be disabled for smoke preflight")
        return mode
    if mode not in {"offline", "online"}:
        raise RuntimeError("WANDB_MODE must be offline or online for training preflight")
    return mode


def validate_batch_contract(
    *,
    per_device_batch_size: int,
    num_generations: int,
    world_size: int,
    require_global_margin_gather: bool,
    num_iterations: int = 1,
    learner_microbatch_size: int = 4,
) -> dict[str, int | bool]:
    batch = int(per_device_batch_size)
    generations = int(num_generations)
    world = int(world_size)
    iterations = int(num_iterations)
    microbatch = int(learner_microbatch_size)
    allowed_batches = {36, 24, 12}
    if os.environ.get("VF_ALLOW_BATCH_PROBE") == "1" or os.environ.get("VF_ALLOW_FORMAL_BATCH48") == "1":
        allowed_batches.add(48)
    if batch not in allowed_batches:
        raise ValueError(
            f"per-device batch must follow 36/24/12 policy"
            f" (48 requires explicit probe or formal authorization), got {batch}"
        )
    if generations != 6:
        raise ValueError(f"num_generations must be 6, got {generations}")
    if world not in {4, 8}:
        raise ValueError(f"learner world size must be 4 or 8, got {world}")
    if iterations not in {1, 4}:
        raise ValueError(f"num_iterations must be 1 or 4, got {iterations}")
    if microbatch <= 0 or batch % microbatch:
        raise ValueError(
            f"per-device batch must be divisible by learner microbatch: "
            f"batch={batch}, microbatch={microbatch}"
        )
    global_batch = batch * world
    if global_batch % generations:
        raise ValueError("global completion batch must be divisible by num_generations")
    if not require_global_margin_gather:
        raise ValueError(f"batch {batch} requires global margin gather before reward computation")
    return {
        "per_device_batch_size": batch,
        "num_generations": generations,
        "world_size": world,
        "num_iterations": iterations,
        "learner_microbatch_size": microbatch,
        "global_completion_batch": global_batch,
        "logical_prompts_per_step": global_batch // generations,
        "require_global_margin_gather": bool(require_global_margin_gather),
    }


def validate_reward_contract(edit_gate_weight: float) -> dict[str, float]:
    weight = float(edit_gate_weight)
    if not math.isfinite(weight) or weight != 0.0:
        raise ValueError(f"edit gate must be disabled, got {weight}")
    return {"edit_gate_weight": weight}


def validate_actor_only_reward_contract(environment: Mapping[str, str]) -> dict[str, Any]:
    expected = {
        "VF_WEIGHT_FORMAT_A0": 1.0,
        "VF_WEIGHT_RATING0": 1.0,
        "VF_WEIGHT_FORMAT_A1": 0.0,
        "VF_WEIGHT_RATING1_ANCHOR": 0.0,
        "VF_WEIGHT_EDIT_GATE": 0.0,
        "VF_WEIGHT_EDIT_GAIN": 0.0,
        "VF_WEIGHT_DELTA_MARGIN": 0.0,
    }
    observed: dict[str, float] = {}
    for name, required in expected.items():
        try:
            value = float(environment.get(name, "nan"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(value) or value != required:
            raise ValueError(f"actor-only reward contract requires {name}={required}, got {value}")
        observed[name] = value
    if environment.get("VF_ACTOR_ONLY") != "1":
        raise ValueError("actor-only preflight requires VF_ACTOR_ONLY=1")
    if environment.get("VF_LOOP_ENABLE_JUDGER") != "0":
        raise ValueError("actor-only preflight requires VF_LOOP_ENABLE_JUDGER=0")
    if environment.get("IMAGE_EDIT_BACKEND", "disabled") != "disabled":
        raise ValueError("actor-only preflight requires IMAGE_EDIT_BACKEND=disabled")
    return {
        "mode": "actor_only",
        "actor_schema": ACTOR_SCHEMA,
        "active_components": ["format_a0", "rating0"],
        "disabled_components": [
            "format_a1", "rating1_anchor", "edit_gate", "edit_gain", "delta_margin"
        ],
        "weights": observed,
        "editing_enabled": False,
        "judger_enabled": False,
        "num_actor_rollouts": 1,
    }


def validate_editor_judge_component_grpo_contract(
    environment: Mapping[str, str],
) -> dict[str, Any]:
    expected_weights = {
        "VF_WEIGHT_FORMAT_A0": 1.0,
        "VF_WEIGHT_RATING0": 1.0,
        "VF_WEIGHT_REASONING": 1.0,
        "VF_SOFT_OVERLONG_WEIGHT": 1.0,
        "VF_WEIGHT_FORMAT_A1": 0.0,
        "VF_WEIGHT_RATING1_ANCHOR": 0.0,
        "VF_WEIGHT_EDIT_GATE": 0.0,
        "VF_WEIGHT_EDIT_GAIN": 0.0,
        "VF_WEIGHT_DELTA_MARGIN": 0.0,
    }
    observed: dict[str, float] = {}
    for name, expected in expected_weights.items():
        value = float(environment.get(name, "nan"))
        if not math.isfinite(value) or value != expected:
            raise ValueError(
                f"Editor+Judge component GRPO requires {name}={expected}, got {value}"
            )
        observed[name] = value
    flags = {
        "VF_ACTOR_ONLY": "1",
        "VF_EDITOR_JUDGE_REASONING_REWARD": "1",
        "VF_DAPO_ENABLED": "0",
        "VF_SCALAR_GRPO_ENABLED": "0",
        "VF_LOOP_ENABLE_JUDGER": "1",
        "IMAGE_EDIT_BACKEND": "diffusers",
        "VF_ACTOR_SCHEMA": REASONING_EVIDENCE_SOLUTION_RATING_ACTOR_SCHEMA,
        "VF_VISION_GC_CONFIGURED": "1",
    }
    mismatches = {
        name: (expected, environment.get(name))
        for name, expected in flags.items()
        if environment.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Editor+Judge component GRPO flag mismatch: {mismatches}")
    expect_language_gc = environment.get("VF_EXPECT_LANGUAGE_GC")
    allow_language_gc_fallback = environment.get("VF_ALLOW_LLM_GC_FALLBACK")
    gradient_checkpointing = environment.get("GRADIENT_CHECKPOINTING")
    use_reentrant = environment.get("GRADIENT_CHECKPOINTING_USE_REENTRANT")
    activation_offload = environment.get("VF_LEARNER_ACTIVATION_OFFLOAD")
    if expect_language_gc == "0":
        language_gc_contract = {
            "VF_ALLOW_LLM_GC_FALLBACK": "0",
            "GRADIENT_CHECKPOINTING": "false",
        }
    elif expect_language_gc == "1":
        language_gc_contract = {
            "VF_ALLOW_LLM_GC_FALLBACK": "1",
            "GRADIENT_CHECKPOINTING": "true",
            "GRADIENT_CHECKPOINTING_USE_REENTRANT": "false",
            "VF_LEARNER_ACTIVATION_OFFLOAD": "1",
        }
    else:
        raise ValueError(
            "VF_EXPECT_LANGUAGE_GC must be 0 or 1 for Editor+Judge GRPO, "
            f"got {expect_language_gc!r}"
        )
    observed_language_gc_contract = {
        "VF_ALLOW_LLM_GC_FALLBACK": allow_language_gc_fallback,
        "GRADIENT_CHECKPOINTING": gradient_checkpointing,
        "GRADIENT_CHECKPOINTING_USE_REENTRANT": use_reentrant,
        "VF_LEARNER_ACTIVATION_OFFLOAD": activation_offload,
    }
    language_gc_mismatches = {
        name: (expected, observed_language_gc_contract.get(name))
        for name, expected in language_gc_contract.items()
        if observed_language_gc_contract.get(name) != expected
    }
    if language_gc_mismatches:
        raise ValueError(
            "Editor+Judge language-GC fallback mismatch: "
            f"{language_gc_mismatches}"
        )
    tau_s = float(environment.get("VF_REASONING_REWARD_TAU_S", "nan"))
    if not math.isclose(tau_s, 1.0):
        raise ValueError(f"reasoning reward tau_s must be 1.0, got {tau_s}")
    mask_mode = component_credit_mask_mode()
    expected_mask_mode = environment.get(
        "VF_EXPECT_COMPONENT_CREDIT_MASK_MODE",
        "field",
    )
    if mask_mode != expected_mask_mode:
        raise ValueError(
            "component credit mask mode mismatch: "
            f"expected={expected_mask_mode}, actual={mask_mode}"
        )
    component_kl_mode = environment.get("VF_COMPONENT_KL_MODE", "off")
    if component_kl_mode not in {"off", "field"}:
        raise ValueError(f"unsupported component KL mode: {component_kl_mode!r}")
    expected_component_kl_mode = environment.get(
        "VF_EXPECT_COMPONENT_KL_MODE",
        component_kl_mode,
    )
    if component_kl_mode != expected_component_kl_mode:
        raise ValueError(
            "component KL mode mismatch: "
            f"expected={expected_component_kl_mode}, actual={component_kl_mode}"
        )
    reasoning_kl_beta = float(environment.get("VF_BETA_KL_REASONING", "0"))
    rating_kl_beta = float(environment.get("VF_BETA_KL_RATING", "0"))
    reference_activation_beta = float(
        environment.get("REFERENCE_ACTIVATION_BETA", "0")
    )
    component_kl_values = {
        "reasoning": reasoning_kl_beta,
        "rating0": rating_kl_beta,
    }
    if not all(
        math.isfinite(value) and value >= 0
        for value in (*component_kl_values.values(), reference_activation_beta)
    ):
        raise ValueError(
            "KL values must be finite and non-negative: "
            f"components={component_kl_values}, activation={reference_activation_beta}"
        )
    if component_kl_mode == "off":
        if any(value != 0 for value in component_kl_values.values()):
            raise ValueError("component KL off requires zero reasoning/rating betas")
    else:
        if mask_mode != "field":
            raise ValueError("field component KL requires field component credit")
        if not all(value > 0 for value in component_kl_values.values()) or not (
            reference_activation_beta > 0
        ):
            raise ValueError("field component KL requires positive KL values")
    if reference_activation_beta > 0:
        reference_model = Path(environment.get("VF_REFERENCE_MODEL_PATH", ""))
        if not reference_model.is_dir():
            raise FileNotFoundError(
                f"KL reference model is missing: {reference_model}"
            )
        if reference_model.resolve() != Path(environment["MODEL_PATH"]).resolve():
            raise ValueError(
                "KL reference must match the fixed initial Actor"
            )
        reference_tree = environment.get("VF_REFERENCE_MODEL_TREE_SHA256", "")
        if len(reference_tree) != 64:
            raise ValueError("KL reference tree hash is missing")
    cache = Path(environment.get("VF_ORIGINAL_SCORE_CACHE_PATH", ""))
    if not cache.is_file():
        raise FileNotFoundError(f"original-score cache is missing: {cache}")
    cache_sha = environment.get("VF_ORIGINAL_SCORE_CACHE_SHA256")
    if cache_sha != EXPECTED_CACHE_SHA256 or sha256_file(cache) != EXPECTED_CACHE_SHA256:
        raise ValueError("original-score cache SHA256 mismatch")
    expected_judge = {
        "VF_JUDGE_MODEL_ID": JUDGER_MODEL_ID,
        "VF_JUDGE_MODEL_PATH": JUDGER_MODEL_PATH,
        "VF_JUDGE_MODEL_TREE_SHA256": JUDGER_MODEL_TREE_SHA256,
        "VF_JUDGE_PROMPT_HASH": JUDGER_PROMPT_HASH,
    }
    judge_mismatches = {
        name: (expected, environment.get(name))
        for name, expected in expected_judge.items()
        if environment.get(name) != expected
    }
    if judge_mismatches:
        raise ValueError(f"Judge provenance mismatch: {judge_mismatches}")

    def urls(name: str) -> list[str]:
        return [
            value.strip().rstrip("/")
            for value in environment.get(name, "").split(",")
            if value.strip()
        ]

    editor_urls = urls("DIFFUSERS_SERVERS")
    judge_urls = urls("VF_LOOP_JUDGER_URLS")
    if editor_urls != [f"http://127.0.0.1:{port}" for port in range(8212, 8216)]:
        raise ValueError(f"Editor lane topology mismatch: {editor_urls}")
    if judge_urls != [f"http://127.0.0.1:{port}" for port in range(8204, 8208)]:
        raise ValueError(f"Judge lane topology mismatch: {judge_urls}")
    return {
        "mode": "actor_only_editor_judge_component_grpo",
        "actor_schema": ACTOR_SCHEMA,
        "active_components": [
            "format_a0",
            "rating0",
            "reasoning",
            "soft_overlong",
        ],
        "component_token_targets": {
            "format_a0": (
                ["a0.completion_non_padding"]
                if mask_mode == "completion"
                else ["a0.format"]
            ),
            "rating0": (
                ["a0.completion_non_padding"]
                if mask_mode == "completion"
                else ["a0.rating_content"]
            ),
            "reasoning": (
                ["a0.completion_non_padding"]
                if mask_mode == "completion"
                else [
                    "a0.reasoning.evidence_content",
                    "a0.reasoning.solution_content",
                ]
            ),
            "soft_overlong": ["a0.completion_non_padding"],
        },
        "component_credit_mask_mode": mask_mode,
        "credit_mask_disabled": mask_mode == "completion",
        "global_completion_kl_applied": (
            component_kl_mode == "off" and reference_activation_beta > 0
        ),
        "kl_in_reward": False,
        "global_completion_kl": {
            "enabled": (
                component_kl_mode == "off" and reference_activation_beta > 0
            ),
            "beta": reference_activation_beta,
            "token_targets": ["a0.active_eligible_completion_non_padding"],
            "estimator": "sampled_k3",
            "normalization": "per_sequence_completion_token_mean_then_active_sequence_mean",
            "loss_sign": "positive_regularization",
            "kl_in_reward": False,
        },
        "component_kl": {
            "mode": component_kl_mode,
            "estimator": "sampled_k3",
            "global_completion_kl_applied": (
                component_kl_mode == "off" and reference_activation_beta > 0
            ),
            "reference_activation_beta": reference_activation_beta,
            "reference_model_path": environment.get("VF_REFERENCE_MODEL_PATH"),
            "reference_model_tree_sha256": environment.get(
                "VF_REFERENCE_MODEL_TREE_SHA256"
            ),
            "segments": (
                {
                    "reasoning": {
                        "beta": reasoning_kl_beta,
                        "token_targets": [
                            "a0.reasoning.evidence_content",
                            "a0.reasoning.solution_content",
                        ],
                    },
                    "rating0": {
                        "beta": rating_kl_beta,
                        "token_targets": ["a0.rating_content"],
                    },
                }
                if component_kl_mode == "field"
                else {}
            ),
        },
        "weights": observed,
        "rating_reward": "local_six_l2_margin",
        "reasoning_reward": {
            "formula": "sign(delta)*(1-exp(-delta^2/(2*tau_s)))",
            "tau_s": tau_s,
            "division_by_four": False,
        },
        "soft_overlong": {
            "max_length": int(
                environment.get("VF_SOFT_OVERLONG_MAX_LENGTH", "0")
            ),
            "cache_length": int(
                environment.get("VF_SOFT_OVERLONG_CACHE_LENGTH", "0")
            ),
            "max_penalty": float(
                environment.get("VF_SOFT_OVERLONG_MAX_PENALTY", "nan")
            ),
            "weight": float(
                environment.get("VF_SOFT_OVERLONG_WEIGHT", "nan")
            ),
            "hard_overlong_filter": False,
        },
        "original_score_cache": {
            "path": str(cache.resolve()),
            "sha256": cache_sha,
            "read_only": True,
        },
        "editor": {
            "model": "FLUX.2-klein-4B",
            "urls": editor_urls,
            "prompt_version": EDITOR_PROMPT_VERSION,
            "prompt_template_hash": EDITOR_PROMPT_TEMPLATE_HASH,
            "semantic_guardrail": EDITOR_SEMANTIC_GUARDRAIL,
            "semantic_guardrail_applied": False,
            "input_fields": ["solution"],
            "positive_prompt_equals_solution": True,
        },
        "judge": {
            "model_id": JUDGER_MODEL_ID,
            "model_path": JUDGER_MODEL_PATH,
            "model_tree_sha256": JUDGER_MODEL_TREE_SHA256,
            "prompt_hash": JUDGER_PROMPT_HASH,
            "urls": judge_urls,
            "deterministic": True,
            "cache_compatible": True,
        },
        "num_actor_rollouts": 1,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"dataset row {line_number} is not an object")
            rows.append(payload)
    return rows


def _image_path(row: Mapping[str, Any]) -> str | None:
    image: Any = row.get("image") or row.get("image_path") or row.get("img_path")
    if image is None:
        images = row.get("images")
        if isinstance(images, list) and len(images) == 1:
            image = images[0]
    while isinstance(image, list) and len(image) == 1:
        image = image[0]
    if isinstance(image, dict):
        image = image.get("path") or image.get("image")
    return str(image).strip() if image is not None else None


def canonical_dataset_image(
    image: str,
    *,
    image_root: Path,
    source: str,
    require_relative: bool = False,
) -> str:
    """Resolve a manifest image without allowing it to escape the public root."""

    root = Path(image_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"TRAIN_IMAGE_ROOT is not a directory: {root}")
    candidate = Path(image).expanduser()
    if require_relative and candidate.is_absolute():
        raise ValueError(f"{source} image must be relative: {image}")
    if not candidate.is_absolute() and ".." in candidate.parts:
        raise ValueError(f"{source} image contains forbidden '..': {image}")
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{source} image cannot be resolved: {image}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{source} image escapes TRAIN_IMAGE_ROOT: {image} -> {resolved}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"{source} image is not a file: {resolved}")
    return resolved.as_posix()


def retained_row_signature(
    row: Mapping[str, Any],
    *,
    image_root: Path,
    source: str,
    require_relative_image: bool = False,
) -> tuple[str, str, float, float]:
    image = _image_path(row)
    sample_id = str(row.get("sample_id") or row.get("id") or "")
    try:
        target = float(row.get("target_mean"))
        std = float(row.get("target_std"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"retained row has invalid target: {sample_id}") from exc
    if (
        not image
        or not sample_id
        or not all(math.isfinite(value) for value in (target, std))
    ):
        raise ValueError(f"retained row is incomplete: {sample_id or '<missing-id>'}")
    canonical_image = canonical_dataset_image(
        image,
        image_root=image_root,
        source=source,
        require_relative=require_relative_image,
    )
    return sample_id, canonical_image, target, std


def inspect_dataset(
    dataset_file: Path,
    *,
    retained_source: Path,
    image_root: Path,
    expected_rows: int = 7000,
    require_images: bool = True,
) -> dict[str, Any]:
    rows = _load_jsonl(dataset_file)
    source_rows = _load_jsonl(retained_source)
    if len(rows) != expected_rows or len(source_rows) != expected_rows:
        raise ValueError(
            f"retained dataset row count mismatch: rewritten={len(rows)}, source={len(source_rows)}, "
            f"expected={expected_rows}"
        )
    source_signatures = [
        retained_row_signature(
            row,
            image_root=image_root,
            source="retained manifest",
            require_relative_image=True,
        )
        for row in source_rows
    ]
    rewritten_signatures = [
        retained_row_signature(
            row,
            image_root=image_root,
            source="run-scoped manifest",
        )
        for row in rows
    ]
    if rewritten_signatures != source_signatures:
        raise ValueError("run-scoped dataset changed retained sample identity, target, or order")
    if len({signature[0] for signature in rewritten_signatures}) != expected_rows:
        raise ValueError("run-scoped dataset contains duplicate sample_id values")

    missing_images: list[str] = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"dataset row {index} does not contain exactly system/user messages")
        if messages[0] != {"role": "system", "content": SYSTEM_PROMPT}:
            raise ValueError(f"dataset row {index} uses a stale system prompt")
        if messages[1] != {"role": "user", "content": TRAINING_USER_PROMPT}:
            raise ValueError(f"dataset row {index} uses a stale user prompt")
        if row.get("prompt_version") != PROMPT_VERSION or row.get("prompt_hash") != PROMPT_HASH:
            raise ValueError(f"dataset row {index} has stale prompt provenance")
        if ACTOR_SCHEMA != LEGACY_ACTOR_SCHEMA:
            expected_fields = prompt_metadata()["top_level_fields"]
            if (
                row.get("actor_schema") != ACTOR_SCHEMA
                or row.get("top_level_fields") != expected_fields
            ):
                raise ValueError(f"dataset row {index} has stale actor schema provenance")
        image = _image_path(row)
        if require_images and (not image or not Path(image).is_file()):
            missing_images.append(str(image or "<missing>"))
            if len(missing_images) >= 10:
                break
    if missing_images:
        raise ValueError("dataset images are missing: " + ", ".join(missing_images))
    return {
        "dataset_file": str(Path(dataset_file).resolve()),
        "retained_source": str(Path(retained_source).resolve()),
        "num_rows": len(rows),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "data_sha256": sha256_file(Path(dataset_file)),
        "retained_source_sha256": sha256_file(Path(retained_source)),
        "sample_order_sha256": hashlib.sha256(
            json.dumps(rewritten_signatures, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def inspect_original_score_cache_coverage(
    dataset_file: Path,
    cache_path: Path,
) -> dict[str, Any]:
    rows = _load_jsonl(dataset_file)
    cache = OriginalScoreCache(cache_path, verify_file_sha256=False)
    digest = hashlib.sha256()
    ratings: list[float] = []
    for index, row in enumerate(rows):
        image = _image_path(row)
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        if not image or not sample_id:
            raise ValueError(f"cache coverage row {index} has no sample identity")
        try:
            record = cache.lookup(image, sample_id=sample_id)
        except KeyError as exc:
            raise KeyError(
                f"training sample is absent or mismatched in original-score cache: "
                f"index={index}, sample_id={sample_id}, image={image}"
            ) from exc
        ratings.append(record.rating)
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.image_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(f"{record.rating:.17g}".encode("ascii"))
        digest.update(b"\n")
    if len(ratings) != len(rows) or len(ratings) != len(set(
        str(row.get("sample_id") or row.get("id") or "") for row in rows
    )):
        raise RuntimeError("original-score cache coverage is not one-to-one")
    return {
        **cache.audit_metadata(),
        "training_rows": len(rows),
        "training_rows_covered": len(ratings),
        "coverage": 1.0,
        "rating_min": min(ratings),
        "rating_max": max(ratings),
        "rating_mean": sum(ratings) / len(ratings),
        "coverage_sha256": digest.hexdigest(),
    }


def inspect_initial_model(
    model_path: Path,
    approved_initials: Iterable[Path],
    *,
    promoted_parent_manifest: Path | None = None,
) -> dict[str, Any]:
    model = Path(model_path).resolve()
    allowed = {Path(path).resolve() for path in approved_initials}
    parent_kind = "approved_initial"
    parent_manifest = None
    if promoted_parent_manifest is not None:
        manifest_path = Path(promoted_parent_manifest).resolve()
        promoted = resolve_promoted_checkpoint(manifest_path)
        if model != promoted:
            raise ValueError(
                f"initial model {model} does not match promoted manifest checkpoint {promoted}"
            )
        allowed.add(promoted)
        parent_kind = "promoted_checkpoint"
        parent_manifest = str(manifest_path)
    if model not in allowed:
        raise ValueError(f"initial model is not approved for this run: {model}")
    config_path = model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = " ".join(
        [str(config.get("model_type") or ""), str(config.get("architectures") or ""), str(model)]
    ).lower()
    supported_identities = ("qwen3_5", "qwen3.5", "qwen35", "qwen3_vl", "qwen3-vl")
    if not any(token in identity for token in supported_identities):
        raise ValueError(f"initial model is not an approved Qwen3.5/Qwen3-VL model: {model}")
    weights = sorted(model.glob("*.safetensors"))
    if not weights or any(path.stat().st_size <= 0 for path in weights):
        raise ValueError(f"initial model weights are missing or empty: {model}")
    return {
        "path": str(model),
        "model_type": config.get("model_type"),
        "weight_files": [path.name for path in weights],
        "total_weight_bytes": sum(path.stat().st_size for path in weights),
        "parent_kind": parent_kind,
        "parent_manifest": parent_manifest,
    }


def inspect_gpus(
    *, require_resident_services: bool = True, training_world_size: int = 4
) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    rows: list[dict[str, int]] = []
    for line in output.splitlines():
        values = [int(item.strip()) for item in line.split(",")]
        if len(values) != 5:
            raise ValueError(f"unexpected nvidia-smi row: {line}")
        rows.append(dict(zip(("index", "total_mib", "used_mib", "free_mib", "utilization"), values)))
    if len(rows) < 8:
        raise ValueError(f"expected at least eight GPUs, found {len(rows)}")
    world = int(training_world_size)
    if world not in {4, 8}:
        raise ValueError(f"unsupported training world size: {world}")
    training = rows[:world]
    services = rows[world:8]
    if any(row["used_mib"] > 1024 or row["free_mib"] < 45000 for row in training):
        raise RuntimeError(f"learner GPUs are not idle enough for launch: {training}")
    if require_resident_services:
        if world != 4:
            raise RuntimeError("resident-service training requires the four-GPU learner topology")
        if any(row["used_mib"] < 10000 for row in services):
            raise RuntimeError(f"resident service GPUs are not populated: {services}")
    return {
        "training": training,
        "services": services,
        "resident_services_required": require_resident_services,
    }


def runtime_identity() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "fla_tilelang": os.environ.get("FLA_TILELANG"),
    }
    for name in ("torch", "vllm", "swift", "trl", "transformers"):
        try:
            module = __import__(name)
            result[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            result[name] = f"unavailable:{type(exc).__name__}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--approved-initial", type=Path, action="append", required=True)
    parser.add_argument("--promoted-parent-manifest", type=Path)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--retained-source", type=Path, required=True)
    parser.add_argument("--service-run-dir", type=Path)
    parser.add_argument("--actor-only", action="store_true")
    parser.add_argument("--editor-judge-component-grpo", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--per-device-batch-size", type=int, required=True)
    parser.add_argument("--num-generations", type=int, default=6)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--learner-microbatch-size", type=int, default=4)
    parser.add_argument("--require-global-margin-gather", action="store_true")
    parser.add_argument("--learner-backward-mode", required=True)
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-train-epochs", type=int)
    parser.add_argument("--freeze-vit", action="store_true")
    parser.add_argument("--freeze-aligner", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.actor_only and args.editor_judge_component_grpo:
        raise ValueError("actor-only baseline and Editor+Judge modes are mutually exclusive")
    services_required = args.editor_judge_component_grpo or not args.actor_only
    if services_required and (
        args.service_run_dir is None or not args.service_run_dir.is_dir()
    ):
        raise FileNotFoundError(f"SERVICE_RUN_DIR does not exist: {args.service_run_dir}")
    validate_wandb_mode(smoke=args.smoke, mode=os.environ.get("WANDB_MODE"))
    if os.environ.get("FLA_TILELANG") != "0":
        raise RuntimeError("FLA_TILELANG must be 0 on the RTX A6000 formal runtime")
    if args.learner_backward_mode != "branch":
        raise RuntimeError(
            f"formal learner backward mode must be branch, got {args.learner_backward_mode}"
        )
    if args.editor_judge_component_grpo:
        reward_contract = validate_editor_judge_component_grpo_contract(os.environ)
        expected_activation_beta = float(
            os.environ.get("REFERENCE_ACTIVATION_BETA", "0")
        )
        if args.beta is None or not math.isclose(
            args.beta,
            expected_activation_beta,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "reference activation beta differs between CLI and component KL contract"
            )
        rollout_mode = "actor_only_editor_judge_component_grpo"
    elif args.actor_only:
        reward_contract = validate_actor_only_reward_contract(os.environ)
        rollout_mode = "actor_only"
    else:
        reward_contract = validate_reward_contract(
            os.environ.get("VF_WEIGHT_EDIT_GATE", "0.0")
        )
        rollout_mode = "dual"
    storage_root = Path(
        os.environ.get("VF_STORAGE_ROOT", args.output.parent)
    ).resolve()
    minimum_free_gib = int(os.environ.get("VF_MIN_FREE_GIB", "500"))
    disk = shutil.disk_usage(storage_root)
    if disk.free < minimum_free_gib * 1024**3:
        raise RuntimeError(
            f"{storage_root} free space is below {minimum_free_gib} GiB: {disk.free}"
        )
    dataset_report = inspect_dataset(
        args.dataset_file,
        retained_source=args.retained_source,
        image_root=Path(os.environ["TRAIN_IMAGE_ROOT"]),
    )
    cache_coverage = None
    if args.editor_judge_component_grpo:
        cache_coverage = inspect_original_score_cache_coverage(
            args.dataset_file,
            Path(os.environ["VF_ORIGINAL_SCORE_CACHE_PATH"]),
        )
    payload = {
        "schema_version": "vf_formal_preflight_v2",
        "rollout_mode": rollout_mode,
        "batch": validate_batch_contract(
            per_device_batch_size=args.per_device_batch_size,
            num_generations=args.num_generations,
            world_size=args.world_size,
            require_global_margin_gather=args.require_global_margin_gather,
            num_iterations=args.num_iterations,
            learner_microbatch_size=args.learner_microbatch_size,
        ),
        "model": inspect_initial_model(
            args.model,
            args.approved_initial,
            promoted_parent_manifest=args.promoted_parent_manifest,
        ),
        "dataset": dataset_report,
        "original_score_cache_coverage": cache_coverage,
        "services": {
            "used": services_required,
            "run_dir": str(args.service_run_dir.resolve()) if args.service_run_dir else None,
        },
        "gpus": inspect_gpus(
            require_resident_services=services_required,
            training_world_size=args.world_size,
        ),
        "disk": {"path": str(storage_root), "free_bytes": disk.free},
        "runtime": runtime_identity(),
        "learner": {
            "backward_mode": args.learner_backward_mode,
            "microbatch_size": args.learner_microbatch_size,
            "num_iterations": args.num_iterations,
        },
        "training_config": {
            "max_completion_length": args.max_completion_length,
            "max_length": args.max_length,
            "max_pixels": args.max_pixels,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "seed": args.seed,
            "num_train_epochs": args.num_train_epochs,
            "freeze_vit": args.freeze_vit,
            "freeze_aligner": args.freeze_aligner,
        },
        "prompt_contract": prompt_metadata(),
        "reward_contract": reward_contract,
        "code_sha256": code_sha256(),
        "wandb_mode": os.environ["WANDB_MODE"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
