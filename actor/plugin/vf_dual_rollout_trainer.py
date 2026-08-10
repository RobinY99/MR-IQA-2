from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from accelerate.utils import DistributedType, gather_object
from swift.rewards import ORM, orms
from swift.rlhf_trainers import GRPOTrainer
from swift.rlhf_trainers import grpo_trainer as swift_grpo_trainer_module
from swift.trainers import TrainerFactory
from transformers import Trainer as HfTrainer, TrainerCallback


PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from actor_contract import (
    actor_schema,
    actor_rating_number,
    parse_tokenizable_actor_json,
    parse_valid_actor_json,
    parse_valid_reasoning_component_json,
    strip_qwen35_non_thinking_prefix,
    to_internal_actor_payload,
    unbounded_rating_number,
)
from activation_offload import (
    MIN_TENSOR_MIB,
    activation_offload_budget_bytes,
    activation_offload_enabled,
    saved_tensor_cpu_offload,
)
from component_loss import (
    combine_active_branch_losses,
    compute_component_kl_loss,
    compute_component_policy_loss,
)
from credit_assignment import build_token_credit_assignment
from dual_rollout_config import apply_qwen35_sampling_config, prepare_a1_record, source_image_path
from dapo_iqa import (
    compute_dapo_group_advantages,
    select_effective_group_indices,
    select_effective_groups_with_shape_padding,
    soft_overlong_reward,
    zero_weight_padding_credit,
)
from editor_judge_contract import (
    EDITOR_PROMPT_TEMPLATE_HASH,
    EDITOR_PROMPT_VERSION,
    EDITOR_SEMANTIC_GUARDRAIL,
    build_editor_prompt,
)
from editor_backend import (
    editor_backend,
    editor_urls,
    request_image_edit,
    select_editor_url,
    trajectory_request_index,
)
from margin_reward import compute_local_margin_rewards
from ms_swift_vf_loop_plugin import (
    judger_urls,
    request_comfy_edit,
    request_judger_score,
    score_payload_mean,
    select_judger_url,
)
from original_score_cache import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_JUDGE_MODEL_ID,
    EXPECTED_JUDGE_MODEL_PATH,
    EXPECTED_JUDGE_PROMPT_HASH,
    OriginalScoreCache,
)
from frozen_judger_contract import (
    JUDGER_GENERATION as EXPECTED_JUDGE_GENERATION,
    JUDGER_MODEL_TREE_SHA256 as EXPECTED_JUDGE_MODEL_TREE_SHA256,
)
from frozen_visual_gc import deactivate_frozen_visual_checkpointing
from reward_scale import (
    delta_margin_reward,
    edit_gain_reward,
    edit_gate_reward,
    rating_anchor_counterfactual_penalty,
    signed_l2_improvement_reward,
)
from service_lane_router import PairedServiceLaneRouter
from streaming_backward import BACKWARD_MODES, LearnerChunk, build_backward_schedule
from token_credit import (
    align_prefix_decoded_token_offsets,
    align_visible_token_offsets,
    build_field_token_masks,
    build_format_boundary_mask,
    build_rollout_component_credit,
    classify_format_boundary,
    component_credit_mask_mode,
    compute_component_group_advantages,
)
from trajectory_io import append_jsonl, rank_sharded_path, stable_trajectory_id


@contextmanager
def _vf_disable_gradient_checkpointing_exact(model, gradient_checkpointing_kwargs=None):
    """Temporarily disable GC while preserving its per-module scope exactly."""
    del gradient_checkpointing_kwargs
    states = [
        (module, bool(module.gradient_checkpointing))
        for module in model.modules()
        if hasattr(module, "gradient_checkpointing")
    ]
    for module, _ in states:
        module.gradient_checkpointing = False
    try:
        yield
    finally:
        for module, enabled in states:
            module.gradient_checkpointing = enabled


# Swift's stock context restores GC through the top-level model, which broadens
# a vision-only or language-only setting to every checkpoint-capable module.
swift_grpo_trainer_module.disable_gradient_checkpointing = (
    _vf_disable_gradient_checkpointing_exact
)


class _DualRolloutPlaceholderReward(ORM):
    def __call__(self, completions, **kwargs) -> list[float]:
        return [0.0] * len(completions)


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _component_kl_betas() -> dict[str, float]:
    mode = os.environ.get("VF_COMPONENT_KL_MODE", "off").strip().lower()
    if mode not in {"off", "field"}:
        raise RuntimeError(f"unsupported VF_COMPONENT_KL_MODE: {mode!r}")
    betas = {
        "reasoning": float(os.environ.get("VF_BETA_KL_REASONING", "0")),
        "rating0": float(os.environ.get("VF_BETA_KL_RATING", "0")),
    }
    invalid = {
        name: value
        for name, value in betas.items()
        if not math.isfinite(value) or value < 0
    }
    if invalid:
        raise RuntimeError(f"component KL betas must be finite and non-negative: {invalid}")
    if mode == "off":
        if any(value != 0.0 for value in betas.values()):
            raise RuntimeError("component KL betas require VF_COMPONENT_KL_MODE=field")
        return {}
    if not all(value > 0.0 for value in betas.values()):
        raise RuntimeError("field component KL requires positive reasoning and rating betas")
    return betas


def _future_result(future: Future | None, kind: str) -> dict[str, Any]:
    if future is None:
        return {"status": "not_requested", "kind": kind}
    try:
        result = future.result()
    except Exception as exc:
        return {"status": "error", "kind": kind, "error": repr(exc)}
    if not isinstance(result, dict):
        return {"status": "error", "kind": kind, "error": "non_object_response"}
    return result


def _flatten_gathered_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        result: list[dict[str, Any]] = []
        for item in value:
            result.extend(_flatten_gathered_rows(item))
        return result
    raise TypeError(f"unexpected gathered trajectory payload: {type(value).__name__}")


class _JudgeProvenanceError(RuntimeError):
    pass


class _EpochBoundaryStopCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        raw_target = os.environ.get("VF_STOP_AFTER_EPOCH", "")
        if not raw_target:
            return control
        try:
            target = int(raw_target)
        except ValueError as exc:
            raise RuntimeError("VF_STOP_AFTER_EPOCH must be an integer") from exc
        try:
            total_epochs = int(os.environ.get("VF_TOTAL_TRAIN_EPOCHS", "3"))
        except ValueError as exc:
            raise RuntimeError("VF_TOTAL_TRAIN_EPOCHS must be an integer") from exc
        if total_epochs <= 0:
            raise RuntimeError("VF_TOTAL_TRAIN_EPOCHS must be positive")
        if not 1 <= target <= total_epochs:
            raise RuntimeError(
                "VF_STOP_AFTER_EPOCH must be between 1 and "
                f"VF_TOTAL_TRAIN_EPOCHS={total_epochs}"
            )
        current = float(state.epoch or 0.0)
        if current + 1e-8 >= target:
            control.should_save = True
            control.should_training_stop = True
        return control


class _StepBoundaryStopCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        raw_target = os.environ.get("VF_STOP_AFTER_STEP", "")
        if not raw_target:
            return control
        try:
            target = int(raw_target)
        except ValueError as exc:
            raise RuntimeError("VF_STOP_AFTER_STEP must be an integer") from exc
        if target <= 0:
            raise RuntimeError("VF_STOP_AFTER_STEP must be positive")
        if int(args.max_steps) <= 0 or target > int(args.max_steps):
            raise RuntimeError(
                "VF_STOP_AFTER_STEP must not exceed the positive max_steps="
                f"{args.max_steps}"
            )
        if int(state.global_step) >= target:
            control.should_save = True
            control.should_training_stop = True
        return control


class VFDualRolloutGRPOTrainer(GRPOTrainer):
    """Token-credit GRPO with either A0/A1 feedback or an explicit A0-only ablation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vf_component_kl_betas = _component_kl_betas()
        self.add_callback(_EpochBoundaryStopCallback())
        self.add_callback(_StepBoundaryStopCallback())
        safe_backward_mode = os.environ.get("VF_SAFE_BACKWARD_MODE", "off")
        if safe_backward_mode not in {"off", "anomaly"}:
            raise RuntimeError(f"unsupported VF_SAFE_BACKWARD_MODE: {safe_backward_mode}")
        if safe_backward_mode == "anomaly" or os.environ.get("VF_DEBUG_DETECT_ANOMALY", "0") == "1":
            torch.autograd.set_detect_anomaly(True, check_nan=True)
        self._vf_empty_cache_before_backward = self._empty_cache_before_backward_enabled()
        self._vf_rollout_call = 0
        self._vf_skip_next_optimizer_step = False
        self._vf_skip_next_scheduler_step = False
        self._vf_guarded_optimizer_id: int | None = None
        self._vf_guarded_scheduler_id: int | None = None
        self._vf_optimizer_guard_executions = 0
        self._vf_scheduler_guard_executions = 0
        self._vf_selective_gc_audited = False
        self._vf_activation_offload = activation_offload_enabled(
            os.environ.get("VF_LEARNER_ACTIVATION_OFFLOAD", "0")
        )
        self._vf_activation_offload_budget_bytes = activation_offload_budget_bytes(
            os.environ.get("VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB", "12")
        )
        self._vf_activation_offload_audited = False
        self._vf_original_score_cache_instance: OriginalScoreCache | None = None
        self._vf_service_lane_router_instance: PairedServiceLaneRouter | None = None
        self._vf_validate_runtime_contract()

    def _prepare_gradient_checkpointing(self, model) -> None:
        super()._prepare_gradient_checkpointing(model)
        if os.environ.get("VF_REQUIRE_FULL_VISUAL_FROZEN", "0") != "1":
            return
        report = deactivate_frozen_visual_checkpointing(model)
        print(
            "[vf-frozen-visual-gc-deactivate] "
            f"rank={os.environ.get('RANK', 'unknown')} "
            f"vision_gc_configured={os.environ.get('VF_VISION_GC_CONFIGURED', '0')} "
            f"visual_modules={len(report['visual_modules'])} "
            f"disabled_gc={len(report['disabled_gc'])} "
            f"removed_input_grad_hooks={len(report['removed_input_grad_hooks'])} "
            "runtime_effective=0",
            flush=True,
        )

    def _vf_validate_selective_gc_runtime(self, model) -> None:
        if self._vf_selective_gc_audited:
            return
        require_frozen_visual = os.environ.get("VF_REQUIRE_FULL_VISUAL_FROZEN", "0")
        expect_language_gc = os.environ.get("VF_EXPECT_LANGUAGE_GC", "0")
        if require_frozen_visual not in {"0", "1"}:
            raise RuntimeError(
                "VF_REQUIRE_FULL_VISUAL_FROZEN must be 0 or 1, "
                f"got {require_frozen_visual!r}"
            )
        if expect_language_gc not in {"0", "1"}:
            raise RuntimeError(
                f"VF_EXPECT_LANGUAGE_GC must be 0 or 1, got {expect_language_gc!r}"
            )
        if require_frozen_visual == "0" and expect_language_gc == "0":
            self._vf_selective_gc_audited = True
            return

        active_language_gc: list[str] = []
        active_visual_gc: list[str] = []
        active_other_gc: list[str] = []
        visual_input_grad_hooks: list[str] = []
        for name, module in model.named_modules():
            dotted = f".{name}."
            if bool(getattr(module, "gradient_checkpointing", False)):
                if ".visual." in dotted:
                    active_visual_gc.append(name)
                elif ".language_model." in dotted:
                    active_language_gc.append(name)
                else:
                    active_other_gc.append(name)
            if (
                ".visual." in dotted
                and getattr(module, "_require_grads_hook", None) is not None
            ):
                visual_input_grad_hooks.append(name)

        trainable_visual = [
            name
            for name, parameter in model.named_parameters()
            if ".visual." in f".{name}." and parameter.requires_grad
        ]
        failed: list[str] = []
        if require_frozen_visual == "1":
            if trainable_visual:
                failed.append("visual_parameters_trainable")
            if active_visual_gc:
                failed.append("visual_gc_active")
            if visual_input_grad_hooks:
                failed.append("visual_input_grad_hooks_active")
        if expect_language_gc == "1" and not active_language_gc:
            failed.append("language_gc_inactive")
        if expect_language_gc == "0" and active_language_gc:
            failed.append("language_gc_unexpected")
        if failed:
            raise RuntimeError(
                "selective GC runtime contract failed: "
                f"{', '.join(failed)}; "
                f"language_gc={active_language_gc[:16]}; "
                f"visual_gc={active_visual_gc[:16]}; "
                f"other_gc={active_other_gc[:16]}; "
                f"trainable_visual={trainable_visual[:16]}; "
                f"visual_input_grad_hooks={visual_input_grad_hooks[:16]}"
            )
        print(
            "[vf-selective-gc] "
            f"rank={os.environ.get('RANK', 'unknown')} "
            f"vision_gc_configured={os.environ.get('VF_VISION_GC_CONFIGURED', '0')} "
            f"vision_gc_runtime_effective={int(bool(active_visual_gc))} "
            f"language_gc={len(active_language_gc)} "
            f"visual_gc={len(active_visual_gc)} "
            f"other_gc={len(active_other_gc)} "
            "visual_trainable=0 visual_input_grad_hooks=0 exact_restore=1",
            flush=True,
        )
        self._vf_selective_gc_audited = True

    def _prepare_rollout_params(self):
        super()._prepare_rollout_params()
        apply_qwen35_sampling_config(
            self.request_config,
            presence_penalty=float(os.environ.get("VF_PRESENCE_PENALTY", "1.5")),
            repetition_penalty=float(os.environ.get("VF_REPETITION_PENALTY", "1.0")),
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        backward_mode, _ = self._backward_mode()
        if backward_mode in {"branch", "microbatch"}:
            time_before = time.perf_counter()
            bundle = self._prepare_inputs(inputs)
            if not isinstance(bundle, dict) or bundle.get("vf_backward_mode") != backward_mode:
                raise RuntimeError(
                    f"streaming preparation returned an invalid bundle: expected={backward_mode}"
                )
            if bundle.get("vf_skip_optimizer_step"):
                loss = self._skip_streaming_training_step(model, bundle)
            else:
                empty_cache_before_backward = getattr(
                    self,
                    "_vf_empty_cache_before_backward",
                    self._empty_cache_before_backward_enabled(),
                )
                if empty_cache_before_backward and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                loss = self._streaming_training_step(model, bundle, num_items_in_batch)
            self._step += 1
            self._current_train_step_time += time.perf_counter() - time_before
            if self._step % self.current_gradient_accumulation_steps == 0:
                self._metrics["train"]["step_time"].append(self._current_train_step_time)
                self._current_train_step_time = 0.0
        else:
            loss = super().training_step(model, inputs, num_items_in_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return loss

    def _install_step_skip_guards(self) -> None:
        optimizer = self.optimizer
        scheduler = self.lr_scheduler
        if optimizer is None or scheduler is None:
            raise RuntimeError("low-effective skip requires initialized optimizer and scheduler")

        if self._vf_guarded_optimizer_id != id(optimizer):
            original_optimizer_step = optimizer.step

            def guarded_optimizer_step(*args, **kwargs):
                if self._vf_skip_next_optimizer_step:
                    self._vf_skip_next_optimizer_step = False
                    self._vf_skip_next_scheduler_step = True
                    self._vf_optimizer_guard_executions += 1
                    return None
                # If an external overflow path suppressed the scheduler call, do
                # not carry a stale skip into the next real update.
                if self._vf_skip_next_scheduler_step:
                    self._vf_skip_next_scheduler_step = False
                return original_optimizer_step(*args, **kwargs)

            optimizer.step = guarded_optimizer_step
            self._vf_guarded_optimizer_id = id(optimizer)

        if self._vf_guarded_scheduler_id != id(scheduler):
            original_scheduler_step = scheduler.step

            def guarded_scheduler_step(*args, **kwargs):
                if self._vf_skip_next_scheduler_step:
                    self._vf_skip_next_scheduler_step = False
                    self._vf_scheduler_guard_executions += 1
                    return None
                return original_scheduler_step(*args, **kwargs)

            scheduler.step = guarded_scheduler_step
            self._vf_guarded_scheduler_id = id(scheduler)

    def _arm_low_effective_update_skip(self) -> None:
        if self._vf_skip_next_optimizer_step or self._vf_skip_next_scheduler_step:
            raise RuntimeError("a prior low-effective optimizer/scheduler skip is still pending")
        self._install_step_skip_guards()
        self._vf_skip_next_optimizer_step = True

    def _skip_streaming_training_step(
        self,
        model,
        bundle: Mapping[str, Any],
    ) -> torch.Tensor:
        if not self._dapo_enabled() or self._dapo_low_effective_action() != "skip_batch":
            raise RuntimeError("low-effective skip bundle is not enabled by the DAPO contract")
        if bundle.get("vf_backward_chunks"):
            raise RuntimeError("low-effective skip bundle must not contain backward chunks")
        effective_rows = int(bundle.get("vf_dapo_effective_rows", -1))
        min_effective_rows = int(bundle.get("vf_dapo_min_effective_rows", -1))
        if not 0 <= effective_rows < min_effective_rows:
            raise RuntimeError(
                "invalid low-effective skip bundle: "
                f"effective={effective_rows}, minimum={min_effective_rows}"
            )

        model.train()
        model.zero_grad()
        self._arm_low_effective_update_skip()
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["vf/backward_chunk_count"].append(0.0)
        self._metrics[mode]["vf/backward_scheduled_chunk_count"].append(0.0)
        self._metrics[mode]["vf/backward_seconds"].append(0.0)
        self._metrics[mode]["vf/optimizer_update_executed"].append(0.0)
        self._metrics[mode]["vf/scheduler_update_executed"].append(0.0)
        self._metrics[mode]["vf/dapo_low_effective_training_skip"].append(1.0)
        return torch.zeros((), dtype=torch.float32, device=self.accelerator.device)

    @staticmethod
    def _empty_cache_before_backward_enabled() -> bool:
        value = os.environ.get("VF_EMPTY_CACHE_BEFORE_BACKWARD", "0")
        if value not in {"0", "1"}:
            raise RuntimeError(
                f"VF_EMPTY_CACHE_BEFORE_BACKWARD must be 0 or 1, got {value!r}"
            )
        return value == "1"

    @staticmethod
    def _backward_mode() -> tuple[str, int]:
        mode = os.environ.get("VF_LEARNER_BACKWARD_MODE", "combined")
        if mode not in BACKWARD_MODES:
            raise RuntimeError(f"unsupported VF_LEARNER_BACKWARD_MODE: {mode}")
        try:
            microbatch_size = int(os.environ.get("VF_LEARNER_MICROBATCH_SIZE", "4"))
        except ValueError as exc:
            raise RuntimeError("VF_LEARNER_MICROBATCH_SIZE must be an integer") from exc
        if microbatch_size <= 0:
            raise RuntimeError("VF_LEARNER_MICROBATCH_SIZE must be positive")
        return mode, microbatch_size

    @staticmethod
    def _actor_only_enabled() -> bool:
        value = os.environ.get("VF_ACTOR_ONLY", "0")
        if value not in {"0", "1"}:
            raise RuntimeError(f"VF_ACTOR_ONLY must be 0 or 1, got {value!r}")
        return value == "1"

    @staticmethod
    def _dapo_enabled() -> bool:
        value = os.environ.get("VF_DAPO_ENABLED", "0")
        if value not in {"0", "1"}:
            raise RuntimeError(f"VF_DAPO_ENABLED must be 0 or 1, got {value!r}")
        return value == "1"

    @staticmethod
    def _dapo_low_effective_action() -> str:
        value = os.environ.get("VF_DAPO_LOW_EFFECTIVE_ACTION", "error")
        if value not in {"error", "skip_batch"}:
            raise RuntimeError(
                "VF_DAPO_LOW_EFFECTIVE_ACTION must be error or skip_batch, "
                f"got {value!r}"
            )
        return value

    @staticmethod
    def _scalar_grpo_enabled() -> bool:
        value = os.environ.get("VF_SCALAR_GRPO_ENABLED", "0")
        if value not in {"0", "1"}:
            raise RuntimeError(f"VF_SCALAR_GRPO_ENABLED must be 0 or 1, got {value!r}")
        return value == "1"

    @staticmethod
    def _editor_judge_reasoning_enabled() -> bool:
        value = os.environ.get("VF_EDITOR_JUDGE_REASONING_REWARD", "0")
        if value not in {"0", "1"}:
            raise RuntimeError(
                "VF_EDITOR_JUDGE_REASONING_REWARD must be 0 or 1, "
                f"got {value!r}"
            )
        return value == "1"

    @staticmethod
    def _margin_reward_scope() -> str:
        value = os.environ.get("VF_MARGIN_REWARD_SCOPE", "global_batch")
        if value not in {"global_batch", "local_six_images"}:
            raise RuntimeError(f"unsupported VF_MARGIN_REWARD_SCOPE: {value!r}")
        return value

    @classmethod
    def _local_six_margin_enabled(cls) -> bool:
        return cls._margin_reward_scope() == "local_six_images"

    def _vf_validate_runtime_contract(self) -> None:
        args = self.args
        expected_iterations = int(os.environ.get("VF_EXPECTED_NUM_ITERATIONS", "1"))
        expected_world_size = int(os.environ.get("VF_EXPECTED_WORLD_SIZE", "4"))
        if expected_iterations not in {1, 4}:
            raise RuntimeError(
                f"VF_EXPECTED_NUM_ITERATIONS must be 1 or 4, got {expected_iterations}"
            )
        if expected_world_size not in {4, 8}:
            raise RuntimeError(
                f"VF_EXPECTED_WORLD_SIZE must be 4 or 8, got {expected_world_size}"
            )
        required = {
            "use_vllm": bool(getattr(args, "use_vllm", False)),
            "vllm_colocate": getattr(args, "vllm_mode", None) == "colocate",
            "num_iterations_expected": (
                int(getattr(args, "num_iterations", 0)) == expected_iterations
            ),
            "steps_per_generation_one": int(getattr(args, "steps_per_generation", 0)) == 1,
            "gradient_accumulation_one": int(getattr(args, "gradient_accumulation_steps", 0)) == 1,
            "loss_expected": getattr(args, "loss_type", None) == (
                "dapo" if self._dapo_enabled() else "grpo"
            ),
            "importance_sampling_token": getattr(args, "importance_sampling_level", None) == "token",
            "not_async": not bool(getattr(args, "async_generate", False)),
            "not_liger": not bool(getattr(args, "use_liger_kernel", False)),
        }
        failed = sorted(name for name, passed in required.items() if not passed)
        if failed:
            raise RuntimeError("dual rollout trainer contract failed: " + ", ".join(failed))
        require_frozen_visual = os.environ.get("VF_REQUIRE_FULL_VISUAL_FROZEN", "0")
        if require_frozen_visual not in {"0", "1"}:
            raise RuntimeError(
                "VF_REQUIRE_FULL_VISUAL_FROZEN must be 0 or 1, "
                f"got {require_frozen_visual!r}"
            )
        if require_frozen_visual == "1":
            visual_parameters = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if ".visual." in f".{name}"
            ]
            trainable_visual = [
                name for name, parameter in visual_parameters if parameter.requires_grad
            ]
            visual_gc_modules = [
                name
                for name, module in self.model.named_modules()
                if ".visual" in f".{name}"
                and bool(getattr(module, "gradient_checkpointing", False))
            ]
            visual_input_grad_hooks = [
                name
                for name, module in self.model.named_modules()
                if ".visual" in f".{name}"
                and getattr(module, "_require_grads_hook", None) is not None
            ]
            vision_gc_configured = (
                os.environ.get("VF_VISION_GC_CONFIGURED", "0") == "1"
            )
            frozen_visual_required = {
                "visual_parameters_present": bool(visual_parameters),
                "visual_parameters_frozen": not trainable_visual,
                "vit_gradient_checkpointing_configuration_matches": bool(
                    getattr(args, "vit_gradient_checkpointing", False)
                )
                == vision_gc_configured,
                "vision_gc_configured_for_editor_judge": (
                    vision_gc_configured
                    if self._editor_judge_reasoning_enabled()
                    else True
                ),
                "visual_gc_modules_disabled": not visual_gc_modules,
                "visual_input_grad_hooks_disabled": not visual_input_grad_hooks,
            }
            frozen_visual_failed = sorted(
                name for name, passed in frozen_visual_required.items() if not passed
            )
            if frozen_visual_failed:
                raise RuntimeError(
                    "full visual freeze runtime contract failed: "
                    f"{', '.join(frozen_visual_failed)}; "
                    f"trainable={trainable_visual[:16]}; "
                    f"gc_modules={visual_gc_modules[:16]}; "
                    f"input_grad_hooks={visual_input_grad_hooks[:16]}"
                )
            print(
                "[vf-visual-freeze] "
                f"rank={os.environ.get('RANK', 'unknown')} "
                f"parameters={len(visual_parameters)} trainable=0 "
                f"gc_configured={int(vision_gc_configured)} "
                "gc_runtime_effective=0 gc_modules=0 input_grad_hooks=0",
                flush=True,
            )
        tp = int(getattr(args, "vllm_tensor_parallel_size", 0))
        world_size = int(self.accelerator.num_processes)
        if world_size != expected_world_size:
            raise RuntimeError(
                f"learner world size does not match the explicit contract: "
                f"world={world_size}, expected={expected_world_size}"
            )
        if tp != world_size:
            raise RuntimeError(
                f"vLLM tensor parallel size must equal learner world size: tp={tp}, world={self.accelerator.num_processes}"
            )
        margin_scope = self._margin_reward_scope()
        if margin_scope == "local_six_images":
            editor_judge_reasoning = self._editor_judge_reasoning_enabled()
            cohort_images = int(os.environ.get("VF_MARGIN_IMAGES_PER_COHORT", "0"))
            local_images = int(os.environ.get("VF_MARGIN_LOCAL_IMAGES_PER_RANK", "0"))
            gather_order = os.environ.get("VF_REWARD_GATHER_ORDER", "")
            generation_batch = int(getattr(args, "generation_batch_size", 0))
            num_generations = int(getattr(args, "num_generations", self.num_generations))
            local_rows = generation_batch // world_size if world_size else 0
            expected_local_rows = local_images * num_generations
            local_margin_required = {
                "dapo_or_scalar_grpo_enabled": (
                    editor_judge_reasoning
                    or self._dapo_enabled() != self._scalar_grpo_enabled()
                ),
                "actor_only": self._actor_only_enabled(),
                "world_size_matches_contract": world_size == expected_world_size,
                "num_generations_6": num_generations == 6,
                "local_completion_rows_match": local_rows == expected_local_rows,
                "positive_local_images": local_images > 0,
                "six_images_per_cohort": cohort_images == 6,
                "whole_cohorts_per_rank": (
                    local_images > 0 and local_images % cohort_images == 0
                ),
                "reward_before_gather": gather_order == "local_reward_then_global_gather",
            }
            local_margin_failed = sorted(
                name for name, passed in local_margin_required.items() if not passed
            )
            if local_margin_failed:
                raise RuntimeError(
                    "local-six margin runtime contract failed: "
                    + ", ".join(local_margin_failed)
                )
        backward_mode, _ = self._backward_mode()
        if self._actor_only_enabled() and backward_mode != "branch":
            raise RuntimeError("actor-only training requires branch backward mode")
        if self._dapo_enabled():
            if self._scalar_grpo_enabled():
                raise RuntimeError("DAPO and scalar GRPO modes are mutually exclusive")
            self._dapo_low_effective_action()
            dapo_required = {
                "actor_only": self._actor_only_enabled(),
                "dynamic_sample": bool(getattr(args, "dynamic_sample", False)),
                "beta_zero": float(getattr(args, "beta", float("nan"))) == 0.0,
                "epsilon_low_0_2": math.isclose(float(self.epsilon_low), 0.2),
                "epsilon_high_0_28": math.isclose(float(self.epsilon_high), 0.28),
                "overlong_filter_disabled": not bool(getattr(args, "overlong_filter", False)),
            }
            dapo_failed = sorted(name for name, passed in dapo_required.items() if not passed)
            if dapo_failed:
                raise RuntimeError("DAPO runtime contract failed: " + ", ".join(dapo_failed))
        if self._scalar_grpo_enabled():
            grpo_required = {
                "actor_only": self._actor_only_enabled(),
                "dapo_disabled": not self._dapo_enabled(),
                "dynamic_sample_disabled": not bool(getattr(args, "dynamic_sample", False)),
                "beta_zero": float(getattr(args, "beta", float("nan"))) == 0.0,
                "epsilon_low_0_2": math.isclose(float(self.epsilon_low), 0.2),
                "epsilon_high_0_2": math.isclose(float(self.epsilon_high), 0.2),
            }
            grpo_failed = sorted(name for name, passed in grpo_required.items() if not passed)
            if grpo_failed:
                raise RuntimeError("scalar GRPO runtime contract failed: " + ", ".join(grpo_failed))
        if self._editor_judge_reasoning_enabled():
            if self._dapo_enabled() or self._scalar_grpo_enabled():
                raise RuntimeError(
                    "Editor+Judge component GRPO cannot enable DAPO or scalar GRPO"
                )
            component_kl_enabled = bool(self._vf_component_kl_betas)
            component_kl_mode = os.environ.get("VF_COMPONENT_KL_MODE", "off")
            expected_component_kl_mode = os.environ.get(
                "VF_EXPECT_COMPONENT_KL_MODE",
                component_kl_mode,
            )
            global_completion_kl_enabled = (
                component_kl_mode == "off" and float(self.beta) > 0.0
            )
            expected_reasoning_beta = float(
                os.environ.get(
                    "VF_EXPECT_BETA_KL_REASONING",
                    str(self._vf_component_kl_betas.get("reasoning", 0.0)),
                )
            )
            expected_rating_beta = float(
                os.environ.get(
                    "VF_EXPECT_BETA_KL_RATING",
                    str(self._vf_component_kl_betas.get("rating0", 0.0)),
                )
            )
            editor_judge_required = {
                "actor_only": self._actor_only_enabled(),
                "world_size_4": world_size == 4,
                "local_six_margin": self._local_six_margin_enabled(),
                "dynamic_sample_disabled": not bool(
                    getattr(args, "dynamic_sample", False)
                ),
                "actor_schema": actor_schema()
                == "reasoning_evidence_solution_rating",
                "editor_enabled": os.environ.get("IMAGE_EDIT_BACKEND") == "diffusers",
                "judge_enabled": os.environ.get("VF_LOOP_ENABLE_JUDGER") == "1",
                "cache_sha": os.environ.get("VF_ORIGINAL_SCORE_CACHE_SHA256")
                == EXPECTED_CACHE_SHA256,
                "judge_model_id": os.environ.get("VF_JUDGE_MODEL_ID")
                == EXPECTED_JUDGE_MODEL_ID,
                "judge_model_path": os.environ.get("VF_JUDGE_MODEL_PATH")
                == EXPECTED_JUDGE_MODEL_PATH,
                "judge_prompt_hash": os.environ.get("VF_JUDGE_PROMPT_HASH")
                == EXPECTED_JUDGE_PROMPT_HASH,
                "component_credit_mask_mode": component_credit_mask_mode()
                == os.environ.get(
                    "VF_EXPECT_COMPONENT_CREDIT_MASK_MODE",
                    "field",
                ),
                "component_kl_field_mode": (
                    not component_kl_enabled
                    or (
                        component_kl_mode == "field"
                        and expected_component_kl_mode == "field"
                    )
                ),
                "component_kl_uses_field_credit": (
                    not component_kl_enabled
                    or component_credit_mask_mode() == "field"
                ),
                "component_kl_reference_enabled": (
                    not component_kl_enabled
                    or (
                        float(self.beta) > 0.0
                        and not self.kl_in_reward
                        and getattr(self, "ref_model", None) is not None
                    )
                ),
                "kl_mode_matches_locked_contract": (
                    component_kl_mode == expected_component_kl_mode
                ),
                "kl_route_is_single_and_loss_side": (
                    not (component_kl_enabled and global_completion_kl_enabled)
                    and (
                        float(self.beta) == 0.0
                        or (
                            not self.kl_in_reward
                            and getattr(self, "ref_model", None) is not None
                            and (component_kl_enabled or global_completion_kl_enabled)
                        )
                    )
                ),
                "component_kl_reasoning_beta": (
                    not component_kl_enabled
                    or math.isclose(
                        self._vf_component_kl_betas["reasoning"],
                        expected_reasoning_beta,
                    )
                ),
                "component_kl_rating_beta": (
                    not component_kl_enabled
                    or math.isclose(
                        self._vf_component_kl_betas["rating0"],
                        expected_rating_beta,
                    )
                ),
                "tau_s_one": math.isclose(
                    float(os.environ.get("VF_REASONING_REWARD_TAU_S", "nan")),
                    1.0,
                ),
                "four_editor_lanes": len(editor_urls("diffusers")) == 4,
                "four_judge_lanes": len(judger_urls()) == 4,
            }
            editor_judge_failed = sorted(
                name
                for name, passed in editor_judge_required.items()
                if not passed
            )
            if editor_judge_failed:
                raise RuntimeError(
                    "Editor+Judge component GRPO runtime contract failed: "
                    + ", ".join(editor_judge_failed)
                )

    def _prepare_streaming_batches(
        self,
        combined_generated: list[dict[str, Any]],
        schedule: Sequence[LearnerChunk],
    ) -> list[dict[str, Any]]:
        original_steps = int(self.args.steps_per_generation)
        self.args.steps_per_generation = len(schedule)
        try:
            prepared_batches = self._prepare_batch_inputs(combined_generated)
        finally:
            self.args.steps_per_generation = original_steps
        if len(prepared_batches) != len(schedule):
            raise RuntimeError(
                f"streaming preparation mismatch: expected={len(schedule)}, actual={len(prepared_batches)}"
            )
        for chunk, batch in zip(schedule, prepared_batches):
            if int(batch["completion_mask"].shape[0]) != chunk.size:
                raise RuntimeError(
                    f"streaming chunk size mismatch: rollout={chunk.rollout}, "
                    f"expected={chunk.size}, actual={batch['completion_mask'].shape[0]}"
                )
        return prepared_batches

    @staticmethod
    def _raw_completion_text(data: Mapping[str, Any]) -> str:
        messages = data.get("messages") or []
        if not messages or messages[-1].get("role") != "assistant":
            return ""
        content = messages[-1].get("content")
        return content if isinstance(content, str) else ""

    @classmethod
    def _completion_text(cls, data: Mapping[str, Any]) -> str:
        return strip_qwen35_non_thinking_prefix(cls._raw_completion_text(data))

    @staticmethod
    def _response_ids(data: Mapping[str, Any]) -> list[int]:
        token_ids = data.get("response_token_ids") or []
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise RuntimeError("multi-turn response_token_ids are forbidden in dual rollout")
            token_ids = token_ids[0]
        if not token_ids or not all(isinstance(token, int) for token in token_ids):
            raise RuntimeError("missing flat response_token_ids")
        return list(token_ids)

    def _token_offsets(self, data: Mapping[str, Any]) -> list[tuple[int, int]]:
        raw_text = self._raw_completion_text(data)
        text = self._completion_text(data)
        response_ids = self._response_ids(data)
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        visible_response_ids = self.template.skip_stop_tokens(response_ids, is_finished=True)
        decode_kwargs = {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        decoded_visible_text = tokenizer.decode(visible_response_ids, **decode_kwargs)
        decoded_token_pieces = [
            tokenizer.decode([token_id], **decode_kwargs)
            for token_id in visible_response_ids
        ]
        try:
            return align_visible_token_offsets(
                response_ids=response_ids,
                visible_response_ids=visible_response_ids,
                decoded_token_pieces=decoded_token_pieces,
                decoded_visible_text=decoded_visible_text,
                content_text=text,
            )
        except ValueError as fast_path_error:
            decoded_prefixes = [""] + [
                tokenizer.decode(visible_response_ids[:index], **decode_kwargs)
                for index in range(1, len(visible_response_ids) + 1)
            ]
            try:
                return align_prefix_decoded_token_offsets(
                    response_ids=response_ids,
                    visible_response_ids=visible_response_ids,
                    decoded_prefixes=decoded_prefixes,
                    decoded_visible_text=decoded_visible_text,
                    content_text=text,
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Qwen3.5 completion token alignment failed: "
                    f"fast_path={fast_path_error}; fallback={exc}; "
                    f"decoded_matches_message={decoded_visible_text == raw_text}; text={text[:240]!r}"
                ) from exc

    def _stop_token_mask(self, data: Mapping[str, Any]) -> list[bool]:
        response_ids = self._response_ids(data)
        result = [False] * len(response_ids)
        if bool(data.get("is_truncated")) or data.get("finish_reason") != "stop":
            return result
        visible_response_ids = self.template.skip_stop_tokens(response_ids, is_finished=True)
        if len(visible_response_ids) < len(response_ids):
            result[-1] = True
        return result

    @staticmethod
    def _weights() -> dict[str, float]:
        if VFDualRolloutGRPOTrainer._dapo_enabled():
            return {"dapo_policy": 1.0}
        if VFDualRolloutGRPOTrainer._scalar_grpo_enabled():
            return {"grpo_policy": 1.0}
        if VFDualRolloutGRPOTrainer._actor_only_enabled():
            weights = {
                "format_a0": float(os.environ.get("VF_WEIGHT_FORMAT_A0", "1.0")),
                "rating0": float(os.environ.get("VF_WEIGHT_RATING0", "1.0")),
            }
            if VFDualRolloutGRPOTrainer._editor_judge_reasoning_enabled():
                weights["reasoning"] = float(
                    os.environ.get("VF_WEIGHT_REASONING", "1.0")
                )
                weights["soft_overlong"] = float(
                    os.environ.get("VF_SOFT_OVERLONG_WEIGHT", "1.0")
                )
            return weights
        return {
            "format_a0": float(os.environ.get("VF_WEIGHT_FORMAT_A0", "1.0")),
            "format_a1": float(os.environ.get("VF_WEIGHT_FORMAT_A1", "1.0")),
            "rating0": float(os.environ.get("VF_WEIGHT_RATING0", "1.0")),
            "rating1_anchor": float(os.environ.get("VF_WEIGHT_RATING1_ANCHOR", "1.0")),
            "edit_gain": float(os.environ.get("VF_WEIGHT_EDIT_GAIN", "1.0")),
            "delta_margin": float(os.environ.get("VF_WEIGHT_DELTA_MARGIN", "1.0")),
            "edit_gate": float(os.environ.get("VF_WEIGHT_EDIT_GATE", "0.0")),
        }

    @staticmethod
    def _edit_gain_reward(target: float | None, delta: float | None) -> float:
        if target is None or delta is None:
            return 0.0
        return edit_gain_reward(
            target,
            delta,
            tau=float(os.environ.get("VF_EDIT_GAIN_TAU", "1.0")),
        )

    def _load_original_score_cache(self) -> OriginalScoreCache:
        if self._vf_original_score_cache_instance is not None:
            return self._vf_original_score_cache_instance
        cache_path = os.environ.get("VF_ORIGINAL_SCORE_CACHE_PATH", "").strip()
        if not cache_path:
            raise RuntimeError("VF_ORIGINAL_SCORE_CACHE_PATH is required")
        verify_sha = os.environ.get("VF_ORIGINAL_SCORE_CACHE_VERIFY_SHA256", "0")
        if verify_sha not in {"0", "1"}:
            raise RuntimeError(
                "VF_ORIGINAL_SCORE_CACHE_VERIFY_SHA256 must be 0 or 1"
            )
        self._vf_original_score_cache_instance = OriginalScoreCache(
            cache_path,
            expected_sha256=os.environ.get(
                "VF_ORIGINAL_SCORE_CACHE_SHA256",
                EXPECTED_CACHE_SHA256,
            ),
            verify_file_sha256=verify_sha == "1",
        )
        return self._vf_original_score_cache_instance

    def _service_lane_router(self) -> PairedServiceLaneRouter:
        if self._vf_service_lane_router_instance is not None:
            return self._vf_service_lane_router_instance
        self._vf_service_lane_router_instance = PairedServiceLaneRouter(
            editor_urls("diffusers"),
            judger_urls(),
            process_rank=int(self.accelerator.process_index),
            ewma_alpha=float(os.environ.get("VF_SERVICE_EWMA_ALPHA", "0.2")),
            judge_steal_ratio=float(
                os.environ.get("VF_JUDGE_WORK_STEAL_RATIO", "1.25")
            ),
        )
        return self._vf_service_lane_router_instance

    @staticmethod
    def _service_attempt_count() -> int:
        attempts = int(os.environ.get("VF_SERVICE_MAX_ATTEMPTS", "3"))
        if not 1 <= attempts <= 4:
            raise RuntimeError("VF_SERVICE_MAX_ATTEMPTS must be in [1, 4]")
        return attempts

    @staticmethod
    def _request_queue_wait(
        elapsed_seconds: float,
        payload: Mapping[str, Any],
    ) -> float:
        reported_queue_wait = _float(payload.get("queue_wait_sec"))
        if reported_queue_wait is not None:
            return max(0.0, reported_queue_wait)
        batch_runtime = _float(payload.get("batch_runtime_sec"))
        if batch_runtime is not None:
            return max(0.0, float(elapsed_seconds) - batch_runtime)
        runtime = _float(payload.get("runtime_sec"))
        if runtime is None:
            return max(0.0, float(elapsed_seconds))
        return max(0.0, float(elapsed_seconds) - runtime)

    @staticmethod
    def _validate_edited_image_size(
        edited_path: str | Path,
        *,
        expected_width: int,
        expected_height: int,
    ) -> tuple[int, int]:
        from PIL import Image

        path = Path(edited_path)
        if not path.is_file():
            raise RuntimeError(f"Editor output does not exist: {path}")
        with Image.open(path) as image:
            actual = (int(image.width), int(image.height))
        expected = (int(expected_width), int(expected_height))
        if actual != expected:
            raise RuntimeError(
                "Editor output size contract failed: "
                f"path={path}, expected={expected}, actual={actual}"
            )
        return actual

    @staticmethod
    def _validate_judge_response_provenance(payload: Mapping[str, Any]) -> None:
        metadata = payload.get("judger")
        if not isinstance(metadata, Mapping):
            raise _JudgeProvenanceError(
                "Judge response is missing provenance metadata"
            )
        required = {
            "model_id": EXPECTED_JUDGE_MODEL_ID,
            "model_path": EXPECTED_JUDGE_MODEL_PATH,
            "model_tree_sha256": EXPECTED_JUDGE_MODEL_TREE_SHA256,
            "prompt_hash": EXPECTED_JUDGE_PROMPT_HASH,
        }
        mismatches = {
            key: (value, metadata.get(key))
            for key, value in required.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise _JudgeProvenanceError(
                f"Judge response provenance mismatch: {mismatches}"
            )
        if (
            metadata.get("deterministic") is not True
            or metadata.get("cache_compatible") is not True
            or metadata.get("generation") != EXPECTED_JUDGE_GENERATION
        ):
            raise _JudgeProvenanceError(
                "Judge response deterministic generation provenance mismatch"
            )

    def _run_single_editor_judge_request(
        self,
        row: Mapping[str, Any],
        *,
        completion_index: int,
    ) -> dict[str, Any]:
        payload, reasoning_errors = parse_valid_reasoning_component_json(
            str(row["a0_text"])
        )
        if reasoning_errors or payload is None:
            return {
                "status": "actor_ineligible",
                "actor_errors": list(reasoning_errors),
                "edit_result": {
                    "status": "not_requested",
                    "reason": "actor_payload_invalid",
                },
                "judger_edited": {
                    "status": "not_requested",
                    "reason": "actor_payload_invalid",
                },
                "editor_attempts": [],
                "judge_attempts": [],
            }
        internal = to_internal_actor_payload(payload)
        evidence = str(internal.get("evidence") or "").strip()
        solution = str(internal.get("solution") or "").strip()
        editor_prompt = build_editor_prompt(evidence, solution)
        if editor_prompt != solution:
            raise RuntimeError("Editor prompt must equal the stripped solution")
        request_index = trajectory_request_index(
            rollout_call=int(row["rollout_call"]),
            rank=int(row["rank"]),
            completion_index=int(completion_index),
        )
        router = self._service_lane_router()
        maximum_attempts = self._service_attempt_count()
        editor_attempts: list[dict[str, Any]] = []
        excluded_editors: list[int] = []
        edit_result: dict[str, Any] | None = None
        editor_lane_index: int | None = None
        edited_width: int | None = None
        edited_height: int | None = None
        for attempt_index in range(maximum_attempts):
            lease = router.reserve_editor(
                excluded_lane_indices=excluded_editors,
            )
            started = time.perf_counter()
            success = False
            try:
                candidate = request_image_edit(
                    image_path=str(row["source_image_path"]),
                    editing=editor_prompt,
                    request_index=request_index,
                    completion_index=completion_index,
                    backend="diffusers",
                    editor_url=lease.url,
                )
                success = (
                    isinstance(candidate, dict)
                    and candidate.get("status") == "success"
                )
                if success:
                    candidate_edited_path = str(
                        candidate.get("edited_path")
                        or candidate.get("edited_image_path")
                        or ""
                    )
                    try:
                        edited_width, edited_height = (
                            self._validate_edited_image_size(
                                candidate_edited_path,
                                expected_width=int(row["source_width"]),
                                expected_height=int(row["source_height"]),
                            )
                        )
                    except Exception as exc:
                        success = False
                        candidate = {
                            **dict(candidate),
                            "status": "error",
                            "error_type": "EditorSizeMismatch",
                            "error": str(exc),
                        }
                    else:
                        edit_result = dict(candidate)
                        editor_lane_index = lease.lane_index
            except Exception as exc:
                candidate = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            elapsed = time.perf_counter() - started
            router.complete(
                lease,
                elapsed_seconds=elapsed,
                success=success,
            )
            editor_attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "lane_index": lease.lane_index,
                    "gpu_index": lease.gpu_index,
                    "url": lease.url,
                    "predicted_wait_seconds": lease.predicted_wait_seconds,
                    "queue_wait_seconds": self._request_queue_wait(
                        elapsed,
                        candidate,
                    ),
                    "request_seconds": elapsed,
                    "success": success,
                    "error_type": candidate.get("error_type"),
                    "error": candidate.get("error"),
                }
            )
            if success:
                break
            excluded_editors.append(lease.lane_index)
        if edit_result is None or editor_lane_index is None:
            return {
                "status": "service_error",
                "failure_stage": "editor",
                "editor_prompt": editor_prompt,
                "editor_prompt_version": EDITOR_PROMPT_VERSION,
                "editor_prompt_template_hash": EDITOR_PROMPT_TEMPLATE_HASH,
                "semantic_guardrail": EDITOR_SEMANTIC_GUARDRAIL,
                "semantic_guardrail_applied": False,
                "solution_only_applied": True,
                "request_index": request_index,
                "edit_result": editor_attempts[-1] if editor_attempts else {
                    "status": "error",
                },
                "judger_edited": {
                    "status": "not_requested",
                    "reason": "editor_failed",
                },
                "editor_attempts": editor_attempts,
                "judge_attempts": [],
            }

        edited_path = str(
            edit_result.get("edited_path")
            or edit_result.get("edited_image_path")
            or ""
        )
        if edited_width is None or edited_height is None:
            raise RuntimeError("successful Editor response has no audited dimensions")

        judge_attempts: list[dict[str, Any]] = []
        excluded_judges: list[int] = []
        judge_result: dict[str, Any] | None = None
        edited_score: float | None = None
        for attempt_index in range(maximum_attempts):
            lease = router.reserve_judge(
                preferred_lane_index=editor_lane_index,
                excluded_lane_indices=excluded_judges,
            )
            started = time.perf_counter()
            success = False
            try:
                candidate = request_judger_score(
                    edited_path,
                    judger_url=lease.url,
                )
                self._validate_judge_response_provenance(candidate)
                score = score_payload_mean(candidate)
                success = score is not None
                if success:
                    judge_result = dict(candidate)
                    edited_score = float(score)
            except Exception as exc:
                if isinstance(exc, _JudgeProvenanceError):
                    router.complete(
                        lease,
                        elapsed_seconds=time.perf_counter() - started,
                        success=False,
                    )
                    raise
                candidate = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            elapsed = time.perf_counter() - started
            router.complete(
                lease,
                elapsed_seconds=elapsed,
                success=success,
            )
            judge_attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "lane_index": lease.lane_index,
                    "gpu_index": lease.gpu_index,
                    "url": lease.url,
                    "preferred_lane_index": lease.preferred_lane_index,
                    "work_stolen": lease.work_stolen,
                    "predicted_wait_seconds": lease.predicted_wait_seconds,
                    "queue_wait_seconds": self._request_queue_wait(
                        elapsed,
                        candidate,
                    ),
                    "request_seconds": elapsed,
                    "success": success,
                    "error_type": candidate.get("error_type"),
                    "error": candidate.get("error"),
                }
            )
            if success:
                break
            excluded_judges.append(lease.lane_index)
        common = {
            "editor_prompt": editor_prompt,
            "editor_prompt_version": EDITOR_PROMPT_VERSION,
            "editor_prompt_template_hash": EDITOR_PROMPT_TEMPLATE_HASH,
            "semantic_guardrail": EDITOR_SEMANTIC_GUARDRAIL,
            "semantic_guardrail_applied": False,
            "solution_only_applied": True,
            "request_index": request_index,
            "edited_image_path": edited_path,
            "edited_width": edited_width,
            "edited_height": edited_height,
            "size_preserved": True,
            "edit_result": edit_result,
            "editor_attempts": editor_attempts,
            "judge_attempts": judge_attempts,
        }
        if judge_result is None or edited_score is None:
            return {
                **common,
                "status": "service_error",
                "failure_stage": "judge",
                "judger_edited": (
                    judge_attempts[-1]
                    if judge_attempts
                    else {"status": "error"}
                ),
                "j1": None,
            }
        return {
            **common,
            "status": "success",
            "failure_stage": None,
            "judger_edited": judge_result,
            "j1": edited_score,
            "judge_delta": edited_score - float(row["j0"]),
        }

    def _run_actor_only_editor_judge_services(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        cache = self._load_original_score_cache()
        for row in rows:
            cached = cache.lookup(
                str(row["source_image_path"]),
                sample_id=str(row["group_id"]),
            )
            row["original_score_cache"] = {
                **cache.audit_metadata(),
                "sample_id": cached.sample_id,
                "image_sha256": cached.image_sha256,
            }
            row["j0"] = cached.rating
            row["source_width"] = cached.width
            row["source_height"] = cached.height

        max_workers = max(
            1,
            int(os.environ.get("VF_EDITOR_JUDGE_SERVICE_WORKERS", "12")),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._run_single_editor_judge_request,
                    row,
                    completion_index=index,
                )
                for index, row in enumerate(rows)
            ]
            results = [future.result() for future in futures]
        for row, result in zip(rows, results):
            row["editor_judge_status"] = result["status"]
            row["editor_backend"] = "diffusers"
            row["editor_prompt"] = result.get("editor_prompt")
            row["editor_prompt_version"] = result.get("editor_prompt_version")
            row["editor_prompt_template_hash"] = result.get(
                "editor_prompt_template_hash"
            )
            row["semantic_guardrail"] = result.get("semantic_guardrail")
            row["semantic_guardrail_applied"] = bool(
                result.get("semantic_guardrail_applied", False)
            )
            row["solution_only_applied"] = bool(
                result.get("solution_only_applied", False)
            )
            row["edited_image_path"] = result.get("edited_image_path")
            row["edited_width"] = result.get("edited_width")
            row["edited_height"] = result.get("edited_height")
            row["size_preserved"] = bool(result.get("size_preserved", False))
            row["editor_request_index"] = result.get("request_index")
            row["edit_result"] = result.get("edit_result")
            row["judger_edited"] = result.get("judger_edited")
            row["editor_attempts"] = result.get("editor_attempts", [])
            row["judge_attempts"] = result.get("judge_attempts", [])
            successful_editor_attempt = next(
                (
                    attempt
                    for attempt in row["editor_attempts"]
                    if attempt.get("success") is True
                ),
                None,
            )
            row["editor_url"] = (
                successful_editor_attempt.get("url")
                if successful_editor_attempt
                else None
            )
            row["j1"] = _float(result.get("j1"))
            row["judge_delta"] = _float(result.get("judge_delta"))
            row["failure_stage"] = result.get("failure_stage")
            row["service_failure_owner"] = (
                "none" if result["status"] == "success" else "service"
            )
            row["service_router_schema_version"] = (
                "vf_paired_service_lane_router_v1"
            )
            row["failure_owner"] = self._failure_owner(row)
        actor_eligible_count = sum(
            result.get("status") != "actor_ineligible" for result in results
        )
        size_violation_count = sum(
            any(
                attempt.get("error_type") == "EditorSizeMismatch"
                for attempt in result.get("editor_attempts", [])
            )
            for result in results
        )
        systemic_threshold = max(
            6,
            math.ceil(max(1, actor_eligible_count) * 0.25),
        )
        if size_violation_count >= systemic_threshold:
            raise RuntimeError(
                "systemic Editor size-preservation failure: "
                f"affected_rows={size_violation_count}, "
                f"actor_eligible_rows={actor_eligible_count}, "
                f"threshold={systemic_threshold}"
            )

    def _gather_global_rows(self, local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        global_rows = _flatten_gathered_rows(gather_object(local_rows))
        VFDualRolloutGRPOTrainer._validate_global_gather(
            self, len(local_rows), len(global_rows)
        )
        return global_rows

    def _validate_global_gather(self, local_count: int, global_count: int) -> None:
        if os.environ.get("VF_REQUIRE_GLOBAL_MARGIN_GATHER", "0") == "1":
            world_size = int(getattr(self.accelerator, "num_processes", 0))
            expected_world_size = int(os.environ.get("VF_EXPECTED_WORLD_SIZE", "4"))
            expected_count = local_count * world_size
            if world_size != expected_world_size or global_count != expected_count:
                raise RuntimeError(
                    "global margin gather contract failed: "
                    f"world={world_size}, local_rows={local_count}, global_rows={global_count}"
                )

    def _build_actor_only_dapo_credit(
        self,
        global_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        margin_scope = self._margin_reward_scope()
        comparison_cohort_ids = None
        expected_groups_per_cohort = None
        if margin_scope == "local_six_images":
            comparison_cohort_ids = [row.get("margin_cohort_id") for row in global_rows]
            expected_groups_per_cohort = int(
                os.environ.get("VF_MARGIN_IMAGES_PER_COHORT", "0")
            )
            if not all(comparison_cohort_ids):
                raise RuntimeError("local-six margin rows require margin_cohort_id")
        rating_rewards, rating_stats = compute_local_margin_rewards(
            [row["a0_text"] for row in global_rows],
            [row["target_mean"] for row in global_rows],
            [row["target_std"] for row in global_rows],
            [row["dapo_group_key"] for row in global_rows],
            comparison_cohort_ids=comparison_cohort_ids,
            expected_groups_per_cohort=expected_groups_per_cohort,
        )
        format_weight = float(os.environ.get("VF_WEIGHT_FORMAT_A0", "1.0"))
        rating_weight = float(os.environ.get("VF_WEIGHT_RATING0", "1.0"))
        overlong_weight = float(os.environ.get("VF_DAPO_OVERLONG_WEIGHT", "1.0"))
        soft_max_length = int(os.environ.get("VF_DAPO_SOFT_MAX_LENGTH", "256"))
        soft_cache_length = int(os.environ.get("VF_DAPO_SOFT_CACHE_LENGTH", "64"))
        epsilon = float(os.environ.get("VF_DAPO_GROUP_EPSILON", "1e-6"))

        advantage_rows: list[dict[str, Any]] = []
        reward_breakdowns: list[dict[str, float]] = []
        for row, rating_reward in zip(global_rows, rating_rewards):
            _, format_errors = parse_valid_actor_json(row["a0_text"])
            format_reward = 0.0 if format_errors else 1.0
            overlong_reward = soft_overlong_reward(
                int(row["a0_token_length"]),
                max_length=soft_max_length,
                cache_length=soft_cache_length,
                max_penalty=1.0,
            )
            total_reward = (
                format_weight * format_reward
                + rating_weight * float(rating_reward)
                + overlong_weight * overlong_reward
            )
            breakdown = {
                "format_a0": format_reward,
                "rating0": float(rating_reward),
                "soft_overlong": overlong_reward,
                "total": total_reward,
            }
            reward_breakdowns.append(breakdown)
            advantage_rows.append(
                {
                    "dapo_group_key": row["dapo_group_key"],
                    "dapo_total_reward": total_reward,
                }
            )

        group_stats = compute_dapo_group_advantages(
            advantage_rows,
            epsilon=epsilon,
            expected_group_size=int(self.num_generations),
        )
        credits: dict[str, dict[str, Any]] = {}
        for row, breakdown, stats in zip(global_rows, reward_breakdowns, group_stats):
            credit = build_token_credit_assignment(
                trajectory_id=row["trajectory_id"],
                rewards={"dapo_policy": breakdown["total"]},
                eligibility={"dapo_policy": True},
                advantages={"dapo_policy": float(stats["advantage"])},
                weights={"dapo_policy": 1.0},
                failure_owner=str(row["failure_owner"]),
            )
            credit["dapo"] = {
                "reward_breakdown": breakdown,
                "effective_group": bool(stats["effective_group"]),
                "group_std": float(stats["group_std"]),
                "group_size": int(stats["group_size"]),
                "rating_stats": dict(rating_stats),
                "token_length": int(row["a0_token_length"]),
                "reward_population": "all_complete_groups_in_sampling_round",
                "reward_computed_before_effective_filter": True,
                "ineffective_groups_participate_in_reward": True,
                "margin_reward_scope": margin_scope,
                "margin_cohort_id": row.get("margin_cohort_id"),
                "margin_cohort_image_count": row.get("margin_cohort_image_count"),
                "reward_gather_order": os.environ.get(
                    "VF_REWARD_GATHER_ORDER", "global_gather_then_reward"
                ),
                "reward_computed_before_global_gather": bool(
                    row.get("reward_computed_before_global_gather", False)
                ),
            }
            credits[row["row_id"]] = credit
        return credits

    def _build_actor_only_scalar_grpo_credit(
        self,
        global_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        margin_scope = self._margin_reward_scope()
        comparison_cohort_ids = None
        expected_groups_per_cohort = None
        if margin_scope == "local_six_images":
            comparison_cohort_ids = [
                row.get("margin_cohort_id") for row in global_rows
            ]
            expected_groups_per_cohort = int(
                os.environ.get("VF_MARGIN_IMAGES_PER_COHORT", "0")
            )
            if not all(comparison_cohort_ids):
                raise RuntimeError("local-six GRPO rows require margin_cohort_id")
        rating_rewards, rating_stats = compute_local_margin_rewards(
            [row["a0_text"] for row in global_rows],
            [row["target_mean"] for row in global_rows],
            [row["target_std"] for row in global_rows],
            [row["group_id"] for row in global_rows],
            comparison_cohort_ids=comparison_cohort_ids,
            expected_groups_per_cohort=expected_groups_per_cohort,
        )
        format_weight = float(os.environ.get("VF_WEIGHT_FORMAT_A0", "1.0"))
        rating_weight = float(os.environ.get("VF_WEIGHT_RATING0", "1.0"))
        epsilon = float(os.environ.get("VF_GRPO_GROUP_EPSILON", "1e-6"))

        advantage_rows: list[dict[str, Any]] = []
        reward_breakdowns: list[dict[str, float]] = []
        for row, rating_reward in zip(global_rows, rating_rewards):
            _, format_errors = parse_valid_actor_json(row["a0_text"])
            format_reward = 0.0 if format_errors else 1.0
            total_reward = format_weight * format_reward + rating_weight * float(rating_reward)
            breakdown = {
                "format_a0": format_reward,
                "rating0": float(rating_reward),
                "total": total_reward,
            }
            reward_breakdowns.append(breakdown)
            advantage_rows.append(
                {
                    "group_id": row["group_id"],
                    "grpo_total_reward": total_reward,
                }
            )

        group_stats = compute_dapo_group_advantages(
            advantage_rows,
            reward_key="grpo_total_reward",
            group_key="group_id",
            epsilon=epsilon,
            expected_group_size=int(self.num_generations),
        )
        credits: dict[str, dict[str, Any]] = {}
        for row, breakdown, stats in zip(global_rows, reward_breakdowns, group_stats):
            credit = build_token_credit_assignment(
                trajectory_id=row["trajectory_id"],
                rewards={"grpo_policy": breakdown["total"]},
                eligibility={"grpo_policy": True},
                advantages={"grpo_policy": float(stats["advantage"])},
                weights={"grpo_policy": 1.0},
                failure_owner=str(row["failure_owner"]),
            )
            credit["grpo"] = {
                "reward_breakdown": breakdown,
                "group_std": float(stats["group_std"]),
                "group_size": int(stats["group_size"]),
                "rating_stats": dict(rating_stats),
                "reward_population": (
                    "complete_rank_local_six_image_cohort"
                    if margin_scope == "local_six_images"
                    else "full_grpo_batch"
                ),
                "margin_reward_scope": margin_scope,
                "margin_cohort_id": row.get("margin_cohort_id"),
                "margin_cohort_image_count": row.get("margin_cohort_image_count"),
                "reward_gather_order": os.environ.get(
                    "VF_REWARD_GATHER_ORDER", "global_gather_then_reward"
                ),
                "reward_computed_before_global_gather": bool(
                    row.get("reward_computed_before_global_gather", False)
                ),
            }
            credits[row["row_id"]] = credit
        return credits

    def _build_actor_only_editor_judge_credit(
        self,
        local_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not self._editor_judge_reasoning_enabled():
            raise RuntimeError("Editor+Judge reasoning reward is not enabled")
        if not local_rows or any(
            "pre_service_rating_reward" not in row for row in local_rows
        ):
            raise RuntimeError(
                "Editor+Judge rating rewards must be computed before service calls"
            )
        group_counts: dict[str, int] = {}
        for row in local_rows:
            group_id = str(row["group_id"])
            group_counts[group_id] = group_counts.get(group_id, 0) + 1
        invalid_groups = {
            group_id: count
            for group_id, count in group_counts.items()
            if count != int(self.num_generations)
        }
        if invalid_groups:
            raise RuntimeError(
                "Editor+Judge rewards require complete six-completion groups: "
                f"{invalid_groups}"
            )
        rating_stats = dict(local_rows[0]["pre_service_rating_stats"])
        rating_rewards = [
            float(row["pre_service_rating_reward"]) for row in local_rows
        ]
        tau_s = float(os.environ.get("VF_REASONING_REWARD_TAU_S", "1.0"))
        if not math.isclose(tau_s, 1.0):
            raise RuntimeError("formal Editor+Judge training fixes tau_s=1.0")
        soft_max_length = int(
            os.environ.get("VF_SOFT_OVERLONG_MAX_LENGTH", "160")
        )
        soft_cache_length = int(
            os.environ.get("VF_SOFT_OVERLONG_CACHE_LENGTH", "16")
        )
        soft_max_penalty = float(
            os.environ.get("VF_SOFT_OVERLONG_MAX_PENALTY", "1.0")
        )
        weights = self._weights()
        component_rows: list[dict[str, Any]] = []
        group_values: dict[str, list[tuple[float, float]]] = {}
        for row, rating_reward in zip(local_rows, rating_rewards):
            _, a0_errors = parse_valid_actor_json(row["a0_text"])
            reasoning_payload, reasoning_errors = (
                parse_valid_reasoning_component_json(row["a0_text"])
            )
            a0_tokenizable = parse_tokenizable_actor_json(row["a0_text"])
            a0_unbounded = (
                unbounded_rating_number(a0_tokenizable.get("rating"))
                if a0_tokenizable
                else None
            )
            target = _float(row["target_mean"])
            target_eligible = bool(target is not None and 1.0 <= target <= 5.0)
            rating_eligible = bool(target_eligible and a0_unbounded is not None)
            delta = _float(row.get("judge_delta"))
            reasoning_eligible = bool(
                reasoning_payload is not None
                and not reasoning_errors
                and row.get("editor_judge_status") == "success"
                and delta is not None
            )
            reasoning_reward = (
                signed_l2_improvement_reward(delta, tau_s=tau_s)
                if reasoning_eligible and delta is not None
                else 0.0
            )
            overlong_reward = soft_overlong_reward(
                int(row["a0_token_length"]),
                max_length=soft_max_length,
                cache_length=soft_cache_length,
                max_penalty=soft_max_penalty,
            )
            row["rating_reward"] = float(rating_reward)
            row["reasoning_raw_reward"] = reasoning_reward
            row["reasoning_reward_eligible"] = reasoning_eligible
            row["soft_overlong_reward"] = overlong_reward
            row["soft_overlong_penalized"] = overlong_reward < 0.0
            row["soft_overlong_saturated"] = math.isclose(
                overlong_reward,
                -soft_max_penalty,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            if reasoning_eligible and delta is not None:
                group_values.setdefault(str(row["group_id"]), []).append(
                    (delta, reasoning_reward)
                )
            component_rows.append(
                {
                    "group_id": row["group_id"],
                    "rewards": {
                        "format_a0": 0.0 if a0_errors else 1.0,
                        "rating0": float(rating_reward),
                        "reasoning": reasoning_reward,
                        "soft_overlong": overlong_reward,
                    },
                    "fallback_advantages": {
                        "format_a0": 0.0,
                        "rating0": (
                            rating_anchor_counterfactual_penalty(
                                a0_unbounded,
                                target,
                            )
                            if rating_eligible
                            else 0.0
                        ),
                        "reasoning": 0.0,
                        "soft_overlong": overlong_reward,
                    },
                    "eligibility": {
                        "format_a0": True,
                        "rating0": rating_eligible,
                        "reasoning": reasoning_eligible,
                        "soft_overlong": True,
                    },
                }
            )

        mode = "train" if self.model.training else "eval"
        row_count = len(local_rows)
        self._metrics[mode]["vf/reasoning_reward_mean"].append(
            math.fsum(float(row["reasoning_raw_reward"]) for row in local_rows)
            / row_count
        )
        self._metrics[mode]["vf/rating_reward_mean"].append(
            math.fsum(float(row["rating_reward"]) for row in local_rows)
            / row_count
        )
        self._metrics[mode]["vf/soft_overlong_reward_mean"].append(
            math.fsum(float(row["soft_overlong_reward"]) for row in local_rows)
            / row_count
        )
        self._metrics[mode]["vf/soft_overlong_penalized_rate"].append(
            math.fsum(bool(row["soft_overlong_penalized"]) for row in local_rows)
            / row_count
        )
        self._metrics[mode]["vf/soft_overlong_saturated_rate"].append(
            math.fsum(bool(row["soft_overlong_saturated"]) for row in local_rows)
            / row_count
        )

        group_stats: dict[str, dict[str, Any]] = {}
        all_groups = {str(row["group_id"]) for row in local_rows}
        for group_id in all_groups:
            values = group_values.get(group_id, [])
            deltas = [item[0] for item in values]
            rewards = [item[1] for item in values]

            def summarize(items: list[float]) -> tuple[float | None, float | None, float | None]:
                if not items:
                    return None, None, None
                mean = math.fsum(items) / len(items)
                variance = (
                    math.fsum((value - mean) ** 2 for value in items)
                    / len(items)
                )
                return mean, variance, math.sqrt(variance)

            delta_mean, delta_variance, delta_std = summarize(deltas)
            reward_mean, reward_variance, reward_std = summarize(rewards)
            group_stats[group_id] = {
                "group_size": int(self.num_generations),
                "reasoning_eligible_count": len(values),
                "judge_delta_mean": delta_mean,
                "judge_delta_variance": delta_variance,
                "judge_delta_std": delta_std,
                "signed_reward_mean": reward_mean,
                "signed_reward_variance": reward_variance,
                "signed_reward_std": reward_std,
                "tau_s": tau_s,
            }

        advantages = compute_component_group_advantages(component_rows)
        credits: dict[str, dict[str, Any]] = {}
        for row, component_row, advantage in zip(
            local_rows,
            component_rows,
            advantages,
        ):
            stats = group_stats[str(row["group_id"])]
            row["editor_judge_group_stats"] = dict(stats)
            row["component_advantages"] = {
                name: float(value) for name, value in advantage.items()
            }
            total_advantage = math.fsum(
                float(advantage.get(name, 0.0))
                * float(weights.get(name, 1.0))
                for name, eligible in component_row["eligibility"].items()
                if bool(eligible)
            )
            mask_mode = component_credit_mask_mode()
            row["component_credit_audit"] = {
                "mask_mode": mask_mode,
                "credit_mask_disabled": mask_mode == "completion",
                "completion_token_count": int(row["a0_token_length"]),
                "prompt_tokens_excluded": True,
                "padding_tokens_excluded_by_completion_mask": True,
                "uniform_non_padding_completion_token_advantage": (
                    mask_mode == "completion"
                ),
                "total_advantage": total_advantage,
            }
            credit = build_token_credit_assignment(
                trajectory_id=row["trajectory_id"],
                rewards=component_row["rewards"],
                eligibility=component_row["eligibility"],
                advantages=advantage,
                weights=weights,
                failure_owner=str(row["failure_owner"]),
            )
            credit["editor_judge"] = {
                "schema_version": "vf_editor_judge_reasoning_reward_v1",
                "rating_reward": "local_six_l2_margin",
                "reasoning_reward": "signed_l2_judge_delta",
                "reasoning_reward_formula": (
                    "sign(delta)*(1-exp(-delta^2/(2*tau_s)))"
                ),
                "tau_s": tau_s,
                "division_by_four": False,
                "cached_original_score": float(row["j0"]),
                "edited_judge_score": _float(row.get("j1")),
                "judge_delta": _float(row.get("judge_delta")),
                "rating_stats": dict(rating_stats),
                "group_stats": dict(stats),
                "reward_population": "same_image_six_completions",
                "margin_reward_scope": "local_six_images",
                "margin_reward_population": (
                    "complete_rank_local_six_image_cohort"
                ),
                "margin_cohort_id": row.get("margin_cohort_id"),
                "margin_cohort_image_count": row.get(
                    "margin_cohort_image_count"
                ),
                "reward_gather_order": "local_reward_then_global_gather",
                "reward_computed_before_global_gather": True,
                "component_credit_mask_mode": mask_mode,
                "credit_mask_disabled": mask_mode == "completion",
                "total_advantage": total_advantage,
                "uniform_non_padding_completion_token_advantage": (
                    mask_mode == "completion"
                ),
                "prompt_tokens_excluded": True,
                "padding_tokens_excluded_by_completion_mask": True,
                "component_token_targets": {
                    "format_a0": (
                        ["a0.completion_non_padding"]
                        if mask_mode == "completion"
                        else ["a0.format"]
                    ),
                    "reasoning": (
                        ["a0.completion_non_padding"]
                        if mask_mode == "completion"
                        else [
                            "a0.reasoning.evidence_content",
                            "a0.reasoning.solution_content",
                        ]
                    ),
                    "rating0": (
                        ["a0.completion_non_padding"]
                        if mask_mode == "completion"
                        else ["a0.rating_content"]
                    ),
                    "soft_overlong": ["a0.completion_non_padding"],
                },
                "soft_overlong": {
                    "max_length": soft_max_length,
                    "cache_length": soft_cache_length,
                    "penalty_start": soft_max_length - soft_cache_length,
                    "max_penalty": soft_max_penalty,
                    "raw_reward": float(row["soft_overlong_reward"]),
                },
            }
            credits[row["row_id"]] = credit
        return credits

    def _prepare_actor_only_editor_judge_rating_rewards(
        self,
        local_rows: list[dict[str, Any]],
    ) -> None:
        comparison_cohort_ids = [
            row.get("margin_cohort_id") for row in local_rows
        ]
        if not all(comparison_cohort_ids):
            raise RuntimeError(
                "Editor+Judge local-six rows require margin_cohort_id"
            )
        expected_groups_per_cohort = int(
            os.environ.get("VF_MARGIN_IMAGES_PER_COHORT", "0")
        )
        rating_rewards, rating_stats = compute_local_margin_rewards(
            [row["a0_text"] for row in local_rows],
            [row["target_mean"] for row in local_rows],
            [row["target_std"] for row in local_rows],
            [row["group_id"] for row in local_rows],
            comparison_cohort_ids=comparison_cohort_ids,
            expected_groups_per_cohort=expected_groups_per_cohort,
        )
        for row, rating_reward in zip(local_rows, rating_rewards):
            row["pre_service_rating_reward"] = float(rating_reward)
            row["pre_service_rating_stats"] = dict(rating_stats)
            row["rating_processed_before_editor_judge"] = True

    def _build_actor_only_global_credit(
        self,
        local_rows: list[dict[str, Any]],
        *,
        already_global: bool = False,
    ) -> dict[str, dict[str, Any]]:
        global_rows = (
            local_rows
            if already_global
            else VFDualRolloutGRPOTrainer._gather_global_rows(self, local_rows)
        )
        if self._dapo_enabled():
            return self._build_actor_only_dapo_credit(global_rows)
        if self._scalar_grpo_enabled():
            return self._build_actor_only_scalar_grpo_credit(global_rows)
        rating_rewards, rating_stats = compute_local_margin_rewards(
            [row["a0_text"] for row in global_rows],
            [row["target_mean"] for row in global_rows],
            [row["target_std"] for row in global_rows],
            [row["group_id"] for row in global_rows],
        )
        weights = self._weights()
        component_rows: list[dict[str, Any]] = []
        for row, rating_reward in zip(global_rows, rating_rewards):
            a0_payload, a0_errors = parse_valid_actor_json(row["a0_text"])
            a0_tokenizable = parse_tokenizable_actor_json(row["a0_text"])
            a0_unbounded = (
                unbounded_rating_number(a0_tokenizable.get("rating"))
                if a0_tokenizable
                else None
            )
            target = _float(row["target_mean"])
            target_eligible = bool(target is not None and 1.0 <= target <= 5.0)
            rating_eligible = bool(target_eligible and a0_unbounded is not None)
            component_rows.append(
                {
                    "group_id": row["group_id"],
                    "rewards": {
                        "format_a0": 0.0 if a0_errors else 1.0,
                        "rating0": float(rating_reward),
                    },
                    "fallback_advantages": {
                        "format_a0": 0.0,
                        "rating0": (
                            rating_anchor_counterfactual_penalty(a0_unbounded, target)
                            if rating_eligible
                            else 0.0
                        ),
                    },
                    "eligibility": {
                        "format_a0": True,
                        "rating0": rating_eligible,
                    },
                }
            )
        advantages = compute_component_group_advantages(component_rows)
        return {
            row["row_id"]: build_token_credit_assignment(
                trajectory_id=row["trajectory_id"],
                rewards=component_row["rewards"],
                eligibility=component_row["eligibility"],
                advantages=advantage,
                weights=weights,
                failure_owner=str(row["failure_owner"]),
            )
            for row, component_row, advantage in zip(global_rows, component_rows, advantages)
        }

    def _build_global_credit(self, local_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if VFDualRolloutGRPOTrainer._actor_only_enabled():
            return VFDualRolloutGRPOTrainer._build_actor_only_global_credit(self, local_rows)
        global_rows = VFDualRolloutGRPOTrainer._gather_global_rows(self, local_rows)
        rating_rewards, rating_stats = compute_local_margin_rewards(
            [row["a0_text"] for row in global_rows],
            [row["target_mean"] for row in global_rows],
            [row["target_std"] for row in global_rows],
            [row["group_id"] for row in global_rows],
        )
        weights = self._weights()
        component_rows: list[dict[str, Any]] = []
        for row, rating_reward in zip(global_rows, rating_rewards):
            a0_payload, a0_errors = parse_valid_actor_json(row["a0_text"])
            a1_payload, a1_errors = parse_valid_actor_json(row["a1_text"])
            a0_tokenizable = parse_tokenizable_actor_json(row["a0_text"])
            a1_tokenizable = parse_tokenizable_actor_json(row["a1_text"])
            a0 = actor_rating_number(a0_payload.get("rating")) if a0_payload else None
            a1 = actor_rating_number(a1_payload.get("rating")) if a1_payload else None
            a0_unbounded = (
                unbounded_rating_number(a0_tokenizable.get("rating"))
                if a0_tokenizable
                else None
            )
            a1_unbounded = (
                unbounded_rating_number(a1_tokenizable.get("rating"))
                if a1_tokenizable
                else None
            )
            j0 = _float(row.get("j0"))
            j1 = _float(row.get("j1"))
            jd = None if j0 is None or j1 is None else j1 - j0
            delta_error = None if a0 is None or a1 is None or jd is None else (a1 - a0) - jd
            target = _float(row["target_mean"])
            target_eligible = bool(target is not None and 1.0 <= target <= 5.0)
            a0_internal = to_internal_actor_payload(a0_tokenizable) if a0_tokenizable else None
            editing = str(a0_internal.get("editing") or "").strip() if a0_internal else ""
            rating_eligible = bool(target_eligible and a0_unbounded is not None)
            rating1_anchor_eligible = bool(
                row["a1_eligible"]
                and a1_unbounded is not None
                and a1 is None
                and j1 is not None
                and 1.0 <= j1 <= 5.0
            )
            edit_gain_eligible = bool(row["a1_eligible"] and jd is not None and a0_tokenizable)
            delta_eligible = bool(edit_gain_eligible and not a1_errors and delta_error is not None)
            edit_gate_eligible = bool(
                weights["edit_gate"] != 0.0 and target_eligible and a0_tokenizable
            )
            edit_gate_kwargs = {
                "tau": float(os.environ.get("VF_EDIT_GATE_TAU", "1.0")),
                "gaussian_mean": float(os.environ.get("VF_EDIT_GAUSSIAN_MEAN", "3.0")),
                "gaussian_std": float(os.environ.get("VF_EDIT_GAUSSIAN_STD", "1.0")),
            }
            gate_reward = (
                edit_gate_reward(target, bool(editing), **edit_gate_kwargs)
                if edit_gate_eligible
                else 0.0
            )
            rating_fallback = (
                rating_anchor_counterfactual_penalty(a0_unbounded, target)
                if rating_eligible
                else 0.0
            )
            rating1_fallback = (
                rating_anchor_counterfactual_penalty(a1_unbounded, j1)
                if rating1_anchor_eligible
                else 0.0
            )
            component_rows.append(
                {
                    "group_id": row["group_id"],
                    "rewards": {
                        "format_a0": 0.0 if a0_errors else 1.0,
                        "format_a1": 0.0 if a1_errors else 1.0,
                        "rating0": float(rating_reward),
                        "rating1_anchor": 0.0,
                        "edit_gain": self._edit_gain_reward(target, jd),
                        "delta_margin": delta_margin_reward(
                            a0,
                            a1,
                            j0,
                            j1,
                            tau=float(os.environ.get("VF_DELTA_MARGIN_TAU", "1.0")),
                        ) if delta_eligible else 0.0,
                        "edit_gate": gate_reward,
                    },
                    "fallback_advantages": {
                        "format_a0": 0.0,
                        "format_a1": 0.0,
                        "rating0": rating_fallback,
                        "rating1_anchor": rating1_fallback,
                        "edit_gate": 0.0,
                    },
                    "eligibility": {
                        "format_a0": True,
                        "format_a1": bool(row["a1_eligible"]),
                        "rating0": rating_eligible,
                        "rating1_anchor": rating1_anchor_eligible,
                        "edit_gain": edit_gain_eligible,
                        "delta_margin": delta_eligible,
                        "edit_gate": edit_gate_eligible,
                    },
                }
            )
        advantages = compute_component_group_advantages(component_rows)
        return {
            row["row_id"]: build_token_credit_assignment(
                trajectory_id=row["trajectory_id"],
                rewards=component_row["rewards"],
                eligibility=component_row["eligibility"],
                advantages=advantage,
                weights=weights,
                failure_owner=str(row["failure_owner"]),
            )
            for row, component_row, advantage in zip(global_rows, component_rows, advantages)
        }

    @staticmethod
    def _failure_owner(row: Mapping[str, Any]) -> str:
        a0_payload, a0_errors = parse_valid_actor_json(str(row.get("a0_text") or ""))
        if a0_errors:
            return "actor"
        if bool(row.get("actor_only")):
            if bool(row.get("editor_judge_reasoning_reward")):
                return (
                    "none"
                    if row.get("editor_judge_status") == "success"
                    else "service"
                )
            return "none"
        a0_internal = to_internal_actor_payload(a0_payload) if a0_payload else None
        editing = str(a0_internal.get("editing") or "").strip() if a0_internal else ""
        if editing and (row.get("edit_result") or {}).get("status") != "success":
            return "service"
        if row.get("a1_eligible"):
            _, a1_errors = parse_valid_actor_json(str(row.get("a1_text") or ""))
            if a1_errors:
                return "actor"
        if row.get("j0") is None or (row.get("a1_eligible") and row.get("j1") is None):
            return "service"
        return "none"

    def _attach_component_credit(
        self,
        batch: dict[str, Any],
        generated: Sequence[Mapping[str, Any]],
        advantages: Sequence[Mapping[str, float]],
        eligibility: Sequence[Mapping[str, bool]],
        *,
        rollout: str,
        active: Sequence[bool],
        row_offset: int = 0,
    ) -> None:
        completion_mask = batch["completion_mask"]
        if row_offset < 0 or row_offset + len(generated) > completion_mask.shape[0]:
            raise RuntimeError(
                f"{rollout} row range exceeds combined batch: "
                f"offset={row_offset}, rows={len(generated)}, batch={completion_mask.shape[0]}"
            )
        weights = self._weights()
        component_masks: dict[str, torch.Tensor] = batch.get("vf_component_masks", {})
        advantage_tensors: dict[str, torch.Tensor] = batch.get("vf_component_advantages", {})
        component_weights: dict[str, float] = batch.get("vf_component_weights", {})
        for index, (data, row_advantages, row_eligibility, is_active) in enumerate(
            zip(generated, advantages, eligibility, active)
        ):
            row_index = row_offset + index
            positions = completion_mask[row_index].nonzero(as_tuple=True)[0]
            offsets = self._token_offsets(data)
            if len(offsets) < len(positions):
                offsets.extend([(len(self._completion_text(data)), len(self._completion_text(data)))] *
                               (len(positions) - len(offsets)))
            if len(positions) != len(offsets):
                raise RuntimeError(
                    f"{rollout} completion mask/token mismatch: mask={len(positions)}, offsets={len(offsets)}"
                )
            if not is_active:
                completion_mask[row_index] = False
                continue
            field_masks = build_field_token_masks(self._completion_text(data), offsets)
            format_component = "format_a0" if rollout == "a0" else "format_a1"
            format_mask, _ = build_format_boundary_mask(
                text=self._completion_text(data),
                offsets=offsets,
                base_format_mask=field_masks["format"],
                advantage=float(row_advantages.get(format_component, 0.0)),
                finish_reason=data.get("finish_reason"),
                is_truncated=bool(data.get("is_truncated")),
                stop_token_mask=self._stop_token_mask(data),
            )
            credit = build_rollout_component_credit(
                rollout,
                field_masks,
                row_advantages,
                weights,
                row_eligibility,
                format_mask_override=format_mask,
            )
            if component_credit_mask_mode() == "completion":
                for name, payload in credit.items():
                    expected_mask = (
                        [True] * len(offsets)
                        if bool(payload["eligible"])
                        else [False] * len(offsets)
                    )
                    if list(payload["mask"]) != expected_mask:
                        raise RuntimeError(
                            "completion-wide component credit mask mismatch: "
                            f"component={name}, row={row_index}"
                        )
            for name, payload in credit.items():
                if name not in component_masks:
                    component_masks[name] = torch.zeros_like(completion_mask, dtype=torch.bool)
                    advantage_tensors[name] = torch.zeros(
                        completion_mask.shape[0],
                        dtype=torch.float32,
                        device=completion_mask.device,
                    )
                    component_weights[name] = float(payload["weight"])
                component_masks[name][row_index, positions] = torch.tensor(
                    payload["mask"],
                    dtype=torch.bool,
                    device=completion_mask.device,
                )
                advantage_tensors[name][row_index] = float(payload["advantage"])
        batch["vf_component_masks"] = component_masks
        batch["vf_component_advantages"] = advantage_tensors
        batch["vf_component_weights"] = component_weights
        batch["advantages"] = torch.zeros(completion_mask.shape[0], device=completion_mask.device)

    @staticmethod
    def _effective_completion_mask(
        completion_mask: torch.Tensor,
        truncated_mask: torch.Tensor | None,
        *,
        overlong_filter: bool,
    ) -> torch.Tensor:
        effective = completion_mask.bool()
        if not overlong_filter or truncated_mask is None:
            return effective
        truncated = truncated_mask.bool()
        if truncated.ndim != 1 or truncated.shape[0] != effective.shape[0]:
            raise RuntimeError("truncated mask does not match completion rows")
        return effective & (~truncated.unsqueeze(-1))

    @staticmethod
    def _streaming_policy_accounting(
        prepared_batches: Sequence[dict[str, Any]],
        *,
        overlong_filter: bool,
    ) -> dict[str, Any]:
        branch_counts = {"a0": 0, "a1": 0}
        component_denominators: dict[str, int] = {}
        covered_rows = 0
        total_policy_tokens = 0
        token_mean = VFDualRolloutGRPOTrainer._dapo_enabled()
        for batch in prepared_batches:
            effective = VFDualRolloutGRPOTrainer._effective_completion_mask(
                batch["completion_mask"],
                batch.get("truncated_mask"),
                overlong_filter=overlong_filter,
            )
            component_union = torch.zeros(effective.shape[0], dtype=torch.bool, device=effective.device)
            for name, mask in batch["vf_component_masks"].items():
                component_active = (mask.bool() & effective).sum(-1) > 0
                component_union |= component_active
                denominator_increment = (
                    int((mask.bool() & effective).sum().item())
                    if token_mean
                    else int(component_active.sum().item())
                )
                component_denominators[name] = (
                    component_denominators.get(name, 0) + denominator_increment
                )
            policy_completion_mask = effective & component_union.unsqueeze(-1)
            batch["vf_policy_completion_mask"] = policy_completion_mask
            policy_rows = policy_completion_mask.sum(-1) > 0
            rollout = str(batch["vf_rollout"])
            branch_counts[rollout] += int(policy_rows.sum().item())
            covered_rows += int(policy_rows.sum().item())
            total_policy_tokens += int(policy_completion_mask.sum().item())
        return {
            "branch_counts": branch_counts,
            "component_denominators": component_denominators,
            "covered_rows": covered_rows,
            "total_active": branch_counts["a0"] + branch_counts["a1"],
            "total_policy_tokens": total_policy_tokens,
        }

    def _apply_dapo_token_normalization(
        self,
        prepared_batches: Sequence[dict[str, Any]],
        accounting: dict[str, Any],
    ) -> None:
        if not self._dapo_enabled():
            return
        local_policy_tokens = int(accounting["total_policy_tokens"])
        token_tensor = torch.tensor(
            local_policy_tokens,
            dtype=torch.long,
            device=self.accelerator.device,
        )
        global_policy_tokens = int(
            self.accelerator.reduce(token_tensor, reduction="sum").item()
        )
        world_size = int(self.accelerator.num_processes)
        if global_policy_tokens <= 0 or world_size <= 0:
            raise RuntimeError(
                "DAPO token normalization requires positive global tokens and world size"
            )

        declared_totals = {
            int(batch["num_items_in_batch"])
            for batch in prepared_batches
            if batch.get("num_items_in_batch") is not None
        }
        local_padding_rows = sum(
            int(batch.get("vf_dapo_padding_rows", 0)) for batch in prepared_batches
        )
        padding_tensor = torch.tensor(
            local_padding_rows,
            dtype=torch.long,
            device=self.accelerator.device,
        )
        global_padding_rows = int(
            self.accelerator.reduce(padding_tensor, reduction="sum").item()
        )
        if global_padding_rows < local_padding_rows:
            raise RuntimeError(
                "DAPO global padding count is smaller than the local count: "
                f"global={global_padding_rows}, local={local_padding_rows}"
            )
        physical_declared_tokens = None
        if global_padding_rows:
            if len(declared_totals) != 1:
                raise RuntimeError(
                    "DAPO shape padding requires one unambiguous physical token count: "
                    f"declared={sorted(declared_totals)}"
                )
            physical_declared_tokens = next(iter(declared_totals))
            if physical_declared_tokens < global_policy_tokens:
                raise RuntimeError(
                    "DAPO physical token count is smaller than the effective-token count: "
                    f"physical={physical_declared_tokens}, effective={global_policy_tokens}"
                )
            for batch in prepared_batches:
                if batch.get("num_items_in_batch") is not None:
                    batch["vf_physical_num_items_in_batch"] = int(
                        batch["num_items_in_batch"]
                    )
                    batch["num_items_in_batch"] = global_policy_tokens
        elif declared_totals and declared_totals != {global_policy_tokens}:
            raise RuntimeError(
                "DAPO global token count disagrees with ms-swift: "
                f"policy={global_policy_tokens}, declared={sorted(declared_totals)}"
            )

        normalizer = global_policy_tokens / world_size
        accounting["component_denominators"] = {
            name: normalizer for name in accounting["component_denominators"]
        }
        accounting["global_policy_tokens"] = global_policy_tokens
        accounting["token_mean_normalizer"] = normalizer
        accounting["physical_declared_tokens"] = (
            physical_declared_tokens
            if physical_declared_tokens is not None
            else global_policy_tokens
        )
        accounting["padding_rows"] = global_padding_rows
        accounting["local_padding_rows"] = local_padding_rows

    def _append_trajectories(self, rows: Sequence[Mapping[str, Any]], credits: Sequence[Mapping[str, Any]]) -> None:
        base = os.environ.get("VF_LOOP_TRAJECTORY_LOG", "")
        if not base:
            return
        rank = int(self.accelerator.process_index)
        output = rank_sharded_path(Path(base), rank)
        for row, credit in zip(rows, credits):
            record = dict(row)
            record["credit_assignment"] = dict(credit)
            append_jsonl(output, record)

    def _build_actor_only_rows(
        self,
        generated: Sequence[Mapping[str, Any]],
        original_paths: Sequence[str | Path],
        *,
        sampling_round: int,
    ) -> list[dict[str, Any]]:
        rank = int(self.accelerator.process_index)
        if len(generated) != len(original_paths):
            raise RuntimeError("actor-only generated/path length mismatch")
        local_rows: list[dict[str, Any]] = []
        for index, (a0_data, original_path) in enumerate(zip(generated, original_paths)):
            group_id = str(a0_data.get("sample_id") or a0_data.get("prompt_id") or "")
            if not group_id:
                raise RuntimeError("actor-only rollout requires sample_id or prompt_id")
            trajectory_id = stable_trajectory_id(
                run_id=os.environ.get("VF_RUN_ID", "unset-run"),
                phase=f"actor_only_rollout_round_{sampling_round}",
                rank=rank,
                reward_call=self._vf_rollout_call,
                sample_id=group_id,
                completion_index=index,
            )
            a0_payload = parse_tokenizable_actor_json(self._completion_text(a0_data))
            reasoning = (
                a0_payload.get("reasoning")
                if a0_payload and isinstance(a0_payload.get("reasoning"), dict)
                else {}
            )
            reasons = str(
                a0_payload.get("reasons")
                or a0_payload.get("reason")
                or reasoning.get("evidence")
                or ""
            ) if a0_payload else ""
            suggestion = str(
                a0_payload.get("suggestion")
                or reasoning.get("solution")
                or ""
            ) if a0_payload else ""
            row = {
                "schema_version": "vf_actor_only_rollout_v1",
                "actor_schema": actor_schema(),
                "actor_only": True,
                "row_id": f"{rank}:{self._vf_rollout_call}:{sampling_round}:{index}",
                "trajectory_id": trajectory_id,
                "group_id": group_id,
                "dapo_group_key": f"{self._vf_rollout_call}:{sampling_round}:{group_id}",
                "rank": rank,
                "rollout_call": self._vf_rollout_call,
                "sampling_round": sampling_round,
                "completion_index": index,
                "source_image_path": str(original_path),
                "edited_image_path": None,
                "editor_backend": "disabled",
                "editor_url": None,
                "editor_request_index": None,
                "editor_seed": None,
                "editor_profile": None,
                "a1_eligible": False,
                "a0_text": self._completion_text(a0_data),
                "a1_text": "",
                "a0_finish_reason": a0_data.get("finish_reason"),
                "a1_finish_reason": None,
                "a0_is_truncated": bool(a0_data.get("is_truncated")),
                "a1_is_truncated": False,
                "a0_token_length": len(self._response_ids(a0_data)),
                "reasons": reasons,
                "suggestion": suggestion,
                "evidence": str(reasoning.get("evidence") or ""),
                "solution": str(reasoning.get("solution") or ""),
                "editing": None,
                "target_mean": _float(a0_data.get("target_mean")),
                "target_std": _float(a0_data.get("target_std")) or 1.0,
                "j0": None,
                "j1": None,
                "edit_result": {"status": "not_requested", "kind": "actor_only"},
                "judger_original": {"status": "not_requested", "kind": "actor_only"},
                "judger_edited": {"status": "not_requested", "kind": "actor_only"},
                "sampling": {
                    "presence_penalty": self.request_config.presence_penalty,
                    "repetition_penalty": self.request_config.repetition_penalty,
                },
            }
            row["a0_format_boundary"] = classify_format_boundary(
                row["a0_text"],
                finish_reason=row["a0_finish_reason"],
                is_truncated=row["a0_is_truncated"],
            )[0]
            row["a1_format_boundary"] = "not_applicable"
            row["failure_owner"] = self._failure_owner(row)
            local_rows.append(row)
        return local_rows

    def _assign_local_margin_cohorts(
        self,
        local_rows: list[dict[str, Any]],
    ) -> None:
        if not self._local_six_margin_enabled():
            return
        images_per_cohort = int(os.environ.get("VF_MARGIN_IMAGES_PER_COHORT", "0"))
        expected_local_images = int(
            os.environ.get("VF_MARGIN_LOCAL_IMAGES_PER_RANK", "0")
        )
        if images_per_cohort <= 0 or expected_local_images <= 0:
            raise RuntimeError("local-six margin cohort sizes must be positive")

        groups: dict[str, list[dict[str, Any]]] = {}
        for row in local_rows:
            key = str(row.get("dapo_group_key") or "")
            if not key:
                raise RuntimeError("local-six margin row is missing dapo_group_key")
            groups.setdefault(key, []).append(row)
        if len(groups) != expected_local_images:
            raise RuntimeError(
                "local-six margin image-group count does not match the configured rank-local batch: "
                f"expected={expected_local_images}, actual={len(groups)}"
            )
        group_size = int(self.num_generations)
        invalid = {key: len(rows) for key, rows in groups.items() if len(rows) != group_size}
        if invalid:
            raise RuntimeError(
                "local-six margin image groups must contain all six completions: "
                f"{invalid}"
            )
        if expected_local_images % images_per_cohort:
            raise RuntimeError("local image count must divide evenly into margin cohorts")

        group_keys = list(groups)
        for cohort_index, start in enumerate(
            range(0, expected_local_images, images_per_cohort)
        ):
            cohort_keys = group_keys[start : start + images_per_cohort]
            first = groups[cohort_keys[0]][0]
            cohort_id = (
                f"rank{int(first['rank'])}:call{int(first['rollout_call'])}:"
                f"round{int(first['sampling_round'])}:cohort{cohort_index}"
            )
            for key in cohort_keys:
                for row in groups[key]:
                    row["margin_reward_scope"] = "local_six_images"
                    row["margin_cohort_id"] = cohort_id
                    row["margin_cohort_image_count"] = images_per_cohort
                    row["margin_cohort_completion_count"] = images_per_cohort * group_size
                    row["reward_gather_order"] = "local_reward_then_global_gather"
                    row["reward_computed_before_global_gather"] = True

    @staticmethod
    def _attach_dapo_row_metadata(
        row: dict[str, Any],
        credit: Mapping[str, Any],
    ) -> None:
        dapo = credit.get("dapo") or {}
        breakdown = dapo.get("reward_breakdown") or {}
        row["dapo_total_reward"] = float(breakdown.get("total", 0.0))
        row["dapo_reward_breakdown"] = dict(breakdown)
        row["dapo_effective_group"] = bool(dapo.get("effective_group", False))
        row["dapo_group_std"] = float(dapo.get("group_std", 0.0))
        row["dapo_reward_population"] = dapo.get("reward_population")
        row["dapo_reward_computed_before_effective_filter"] = bool(
            dapo.get("reward_computed_before_effective_filter", False)
        )
        row["dapo_ineffective_groups_participate_in_reward"] = bool(
            dapo.get("ineffective_groups_participate_in_reward", False)
        )

    def _generate_actor_only_dapo_completions(
        self,
        inputs,
    ) -> (
        tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
        | dict[str, Any]
    ):
        target_rows = int(self.args.generation_batch_size)
        group_size = int(self.num_generations)
        world_size = int(self.accelerator.num_processes)
        if target_rows % world_size or target_rows % group_size:
            raise RuntimeError("DAPO generation batch must divide world size and group size")
        local_target = target_rows // world_size
        max_rounds = int(os.environ.get("VF_DAPO_MAX_GENERATION_ROUNDS", "4"))
        min_effective_rows = int(
            os.environ.get("VF_DAPO_MIN_EFFECTIVE_ROWS", "96")
        )
        if max_rounds <= 0:
            raise RuntimeError("VF_DAPO_MAX_GENERATION_ROUNDS must be positive")
        if (
            min_effective_rows <= 0
            or min_effective_rows > target_rows
            or min_effective_rows % group_size
        ):
            raise RuntimeError(
                "VF_DAPO_MIN_EFFECTIVE_ROWS must be positive, no larger than the "
                "generation batch, and divisible by the DAPO group size"
            )

        accumulated_generated: list[dict[str, Any]] = []
        accumulated_rows: list[dict[str, Any]] = []
        accumulated_credits: list[dict[str, Any]] = []
        local_evidence: list[tuple[dict[str, Any], dict[str, Any]]] = []
        current_inputs = inputs
        selected_indices: list[int] = []

        for sampling_round in range(max_rounds):
            original_paths = [source_image_path(dict(data)) for data in current_inputs]
            local_generated = super()._generate_completions(current_inputs)
            local_rows = self._build_actor_only_rows(
                local_generated,
                original_paths,
                sampling_round=sampling_round,
            )
            for data, row in zip(local_generated, local_rows):
                data["vf_dapo_row_id"] = row["row_id"]

            local_credit_by_id: dict[str, dict[str, Any]] | None = None
            if self._local_six_margin_enabled():
                self._assign_local_margin_cohorts(local_rows)
                local_credit_by_id = self._build_actor_only_dapo_credit(local_rows)
                for row in local_rows:
                    self._attach_dapo_row_metadata(row, local_credit_by_id[row["row_id"]])
                local_rewarded = [
                    {"row": row, "credit": local_credit_by_id[row["row_id"]]}
                    for row in local_rows
                ]
                global_rewarded = _flatten_gathered_rows(gather_object(local_rewarded))
                self._validate_global_gather(len(local_rewarded), len(global_rewarded))
                global_rows = [dict(item["row"]) for item in global_rewarded]
                credit_by_id = {
                    str(item["row"]["row_id"]): dict(item["credit"])
                    for item in global_rewarded
                }
                if len(credit_by_id) != len(global_rows):
                    raise RuntimeError("gathered local-six credits have duplicate row ids")
                global_generated = _flatten_gathered_rows(gather_object(local_generated))
            else:
                global_generated = _flatten_gathered_rows(gather_object(local_generated))
                global_rows = self._gather_global_rows(local_rows)
                credit_by_id = self._build_actor_only_global_credit(
                    global_rows,
                    already_global=True,
                )
            generated_by_id = {
                str(data.get("vf_dapo_row_id", "")): data for data in global_generated
            }
            if len(generated_by_id) != len(global_rows):
                raise RuntimeError(
                    "DAPO gathered generated rows are not one-to-one with trajectory rows: "
                    f"generated={len(generated_by_id)}, rows={len(global_rows)}"
                )
            for row in global_rows:
                credit = credit_by_id[row["row_id"]]
                self._attach_dapo_row_metadata(row, credit)
                accumulated_generated.append(generated_by_id[row["row_id"]])
                accumulated_rows.append(row)
                accumulated_credits.append(credit)
            for row in local_rows:
                credit = (
                    local_credit_by_id[row["row_id"]]
                    if local_credit_by_id is not None
                    else credit_by_id[row["row_id"]]
                )
                self._attach_dapo_row_metadata(row, credit)
                local_evidence.append((row, credit))

            selected_indices = select_effective_group_indices(
                accumulated_rows,
                target_rows=target_rows,
                group_size=group_size,
            )
            if len(selected_indices) == target_rows:
                break
            if sampling_round + 1 >= max_rounds:
                continue
            resampled = next(self.dynamic_resample_iterator)
            if self.template.truncation_strategy == "raise":
                resampled = self.resample_encode_failed_inputs(resampled)
            current_inputs = HfTrainer._prepare_inputs(self, resampled)

        generated_effective_rows = sum(
            bool(row.get("dapo_effective_group")) for row in accumulated_rows
        )
        if generated_effective_rows < min_effective_rows:
            action = self._dapo_low_effective_action()
            for row, _ in local_evidence:
                row["dapo_selected_for_update"] = False
                row["dapo_padding_for_shape"] = False
                row["dapo_physical_selected"] = False
                row["dapo_policy_loss_weight"] = 0.0
                row["dapo_token_denominator_eligible"] = False
                row["dapo_min_effective_stop"] = action == "error"
                row["dapo_low_effective_batch_skipped"] = action == "skip_batch"
                row["dapo_update_executed"] = False
                row["dapo_low_effective_action"] = action
                row["dapo_min_effective_rows"] = min_effective_rows
            if local_evidence:
                self._append_trajectories(
                    [row for row, _ in local_evidence],
                    [credit for _, credit in local_evidence],
                )
            mode = "train" if self.model.training else "eval"
            rounds = 1 + max(int(row["sampling_round"]) for row in accumulated_rows)
            self._metrics[mode]["vf/dapo_generation_rounds"].append(float(rounds))
            self._metrics[mode]["vf/dapo_generated_rows"].append(
                float(len(accumulated_rows))
            )
            self._metrics[mode]["vf/dapo_effective_rows"].append(
                float(generated_effective_rows)
            )
            for key in (
                "vf/dapo_selected_rows",
                "vf/dapo_local_selected_rows",
                "vf/dapo_effective_selected_rows",
                "vf/dapo_padding_rows",
                "vf/dapo_local_effective_selected_rows",
                "vf/dapo_local_padding_rows",
                "vf/dapo_partial_batch_fallback",
            ):
                self._metrics[mode][key].append(0.0)
            self._metrics[mode]["vf/dapo_min_effective_rows"].append(
                float(min_effective_rows)
            )
            self._metrics[mode]["vf/dapo_acceptance_rate"].append(
                generated_effective_rows / len(accumulated_rows)
            )
            self._metrics[mode]["vf/dapo_low_effective_batch_skipped"].append(
                float(action == "skip_batch")
            )
            if action == "error":
                raise RuntimeError(
                    "DAPO effective trajectory floor was not met after dynamic sampling: "
                    f"effective={generated_effective_rows}, minimum={min_effective_rows}, "
                    f"target={target_rows}, generated={len(accumulated_rows)}, "
                    f"rounds={rounds}"
                )
            return {
                "vf_dapo_skip_batch": True,
                "vf_dapo_skip_reason": "effective_rows_below_minimum",
                "vf_dapo_generation_rounds": rounds,
                "vf_dapo_generated_rows": len(accumulated_rows),
                "vf_dapo_effective_rows": generated_effective_rows,
                "vf_dapo_min_effective_rows": min_effective_rows,
                "vf_dapo_target_rows": target_rows,
            }

        selection = select_effective_groups_with_shape_padding(
            accumulated_rows,
            target_rows=target_rows,
            group_size=group_size,
            world_size=world_size,
            min_effective_rows=min_effective_rows,
        )
        selected_indices = list(selection.physical_indices)
        active_index_set = set(selection.active_indices)
        padding_index_set = set(selection.padding_indices)
        active_ids = {
            accumulated_rows[index]["row_id"] for index in active_index_set
        }
        padding_ids = {
            accumulated_rows[index]["row_id"] for index in padding_index_set
        }
        physical_ids = active_ids | padding_ids
        learner_assignment = {
            accumulated_rows[index]["row_id"]: (
                position // local_target,
                position % local_target,
            )
            for position, index in enumerate(selected_indices)
        }

        def mark_selection(row: dict[str, Any]) -> None:
            row_id = row["row_id"]
            is_active = row_id in active_ids
            is_padding = row_id in padding_ids
            row["dapo_selected_for_update"] = is_active
            row["dapo_padding_for_shape"] = is_padding
            row["dapo_physical_selected"] = row_id in physical_ids
            row["dapo_policy_loss_weight"] = 1.0 if is_active else 0.0
            row["dapo_token_denominator_eligible"] = is_active
            assignment = learner_assignment.get(row_id)
            row["dapo_learner_rank"] = assignment[0] if assignment else None
            row["dapo_learner_rank_slot"] = assignment[1] if assignment else None

        for row in accumulated_rows:
            mark_selection(row)
        evidence_rows: list[dict[str, Any]] = []
        evidence_credits: list[dict[str, Any]] = []
        for row, credit in local_evidence:
            mark_selection(row)
            evidence_rows.append(row)
            evidence_credits.append(
                zero_weight_padding_credit(credit)
                if row["dapo_padding_for_shape"]
                else credit
            )
        if local_evidence:
            self._append_trajectories(evidence_rows, evidence_credits)

        selected_generated = [accumulated_generated[index] for index in selected_indices]
        selected_rows = [accumulated_rows[index] for index in selected_indices]
        selected_credits = [
            zero_weight_padding_credit(accumulated_credits[index])
            if index in padding_index_set
            else accumulated_credits[index]
            for index in selected_indices
        ]
        process_slice = slice(
            int(self.accelerator.process_index) * local_target,
            (int(self.accelerator.process_index) + 1) * local_target,
        )
        local_generated = selected_generated[process_slice]
        local_rows = selected_rows[process_slice]
        local_credits = selected_credits[process_slice]
        if len(local_generated) != local_target or len(local_rows) != local_target:
            raise RuntimeError("DAPO selected local batch has the wrong size")

        mode = "train" if self.model.training else "eval"
        effective_rows = sum(bool(row.get("dapo_effective_group")) for row in accumulated_rows)
        effective_selected_rows = len(selection.active_indices)
        padding_rows = len(selection.padding_indices)
        local_effective_rows = sum(
            bool(row.get("dapo_selected_for_update")) for row in local_rows
        )
        local_padding_rows = sum(
            bool(row.get("dapo_padding_for_shape")) for row in local_rows
        )
        if local_effective_rows <= 0 or local_effective_rows + local_padding_rows != local_target:
            raise RuntimeError(
                "DAPO rank-local active/padding allocation is invalid: "
                f"active={local_effective_rows}, padding={local_padding_rows}, "
                f"target={local_target}"
            )
        rounds = 1 + max(int(row["sampling_round"]) for row in accumulated_rows)
        self._metrics[mode]["vf/dapo_generation_rounds"].append(float(rounds))
        self._metrics[mode]["vf/dapo_generated_rows"].append(float(len(accumulated_rows)))
        self._metrics[mode]["vf/dapo_effective_rows"].append(float(effective_rows))
        self._metrics[mode]["vf/dapo_selected_rows"].append(float(len(selected_rows)))
        self._metrics[mode]["vf/dapo_local_selected_rows"].append(float(len(local_rows)))
        self._metrics[mode]["vf/dapo_effective_selected_rows"].append(
            float(effective_selected_rows)
        )
        self._metrics[mode]["vf/dapo_padding_rows"].append(float(padding_rows))
        self._metrics[mode]["vf/dapo_local_effective_selected_rows"].append(
            float(local_effective_rows)
        )
        self._metrics[mode]["vf/dapo_local_padding_rows"].append(
            float(local_padding_rows)
        )
        self._metrics[mode]["vf/dapo_partial_batch_fallback"].append(
            float(padding_rows > 0)
        )
        self._metrics[mode]["vf/dapo_min_effective_rows"].append(
            float(min_effective_rows)
        )
        self._metrics[mode]["vf/dapo_low_effective_batch_skipped"].append(0.0)
        self._metrics[mode]["vf/dapo_acceptance_rate"].append(
            effective_rows / len(accumulated_rows)
        )
        active_selected_rows = [
            row for row in selected_rows if row["dapo_selected_for_update"]
        ]
        selected_breakdowns = [
            row["dapo_reward_breakdown"] for row in active_selected_rows
        ]
        self._metrics[mode]["vf/dapo_reward_mean"].append(
            math.fsum(float(item["total"]) for item in selected_breakdowns)
            / effective_selected_rows
        )
        self._metrics[mode]["vf/dapo_format_reward_mean"].append(
            math.fsum(float(item["format_a0"]) for item in selected_breakdowns)
            / effective_selected_rows
        )
        self._metrics[mode]["vf/dapo_rating_reward_mean"].append(
            math.fsum(float(item["rating0"]) for item in selected_breakdowns)
            / effective_selected_rows
        )
        self._metrics[mode]["vf/dapo_soft_overlong_mean"].append(
            math.fsum(float(item["soft_overlong"]) for item in selected_breakdowns)
            / effective_selected_rows
        )
        return local_generated, local_rows, local_credits

    def _generate_actor_only_completions(self, inputs):
        if self._dapo_enabled():
            dapo_result = self._generate_actor_only_dapo_completions(inputs)
            if isinstance(dapo_result, dict):
                if not dapo_result.get("vf_dapo_skip_batch"):
                    raise RuntimeError("unexpected DAPO generation result")
                return [{
                    "vf_dual_rollout": True,
                    "vf_actor_only": True,
                    "vf_backward_mode": "branch",
                    "vf_skip_optimizer_step": True,
                    **dapo_result,
                }]
            a0_generated, local_rows, local_credits = dapo_result
        else:
            original_paths = [source_image_path(dict(data)) for data in inputs]
            a0_generated = super()._generate_completions(inputs)
            local_rows = self._build_actor_only_rows(
                a0_generated,
                original_paths,
                sampling_round=0,
            )
            if self._editor_judge_reasoning_enabled():
                for row in local_rows:
                    row["editor_judge_reasoning_reward"] = True
                    row["editor_judge_status"] = "pending"
                self._assign_local_margin_cohorts(local_rows)
                self._prepare_actor_only_editor_judge_rating_rewards(
                    local_rows
                )
                self._run_actor_only_editor_judge_services(local_rows)
                credit_by_id = self._build_actor_only_editor_judge_credit(
                    local_rows
                )
                editor_judge_local_rewarded = [
                    {"row": row, "credit": credit_by_id[row["row_id"]]}
                    for row in local_rows
                ]
                editor_judge_global_rewarded = _flatten_gathered_rows(
                    gather_object(editor_judge_local_rewarded)
                )
                self._validate_global_gather(
                    len(editor_judge_local_rewarded),
                    len(editor_judge_global_rewarded),
                )
                gathered_ids = {
                    str(item["row"]["row_id"])
                    for item in editor_judge_global_rewarded
                }
                if len(gathered_ids) != len(editor_judge_global_rewarded):
                    raise RuntimeError(
                        "gathered Editor+Judge GRPO credits have duplicate row ids"
                    )
            elif self._scalar_grpo_enabled() and self._local_six_margin_enabled():
                self._assign_local_margin_cohorts(local_rows)
                credit_by_id = self._build_actor_only_scalar_grpo_credit(local_rows)
                local_rewarded = [
                    {"row": row, "credit": credit_by_id[row["row_id"]]}
                    for row in local_rows
                ]
                global_rewarded = _flatten_gathered_rows(
                    gather_object(local_rewarded)
                )
                self._validate_global_gather(
                    len(local_rewarded), len(global_rewarded)
                )
                gathered_ids = {
                    str(item["row"]["row_id"]) for item in global_rewarded
                }
                if len(gathered_ids) != len(global_rewarded):
                    raise RuntimeError(
                        "gathered local-six GRPO credits have duplicate row ids"
                    )
            else:
                credit_by_id = self._build_actor_only_global_credit(local_rows)
            local_credits = [credit_by_id[row["row_id"]] for row in local_rows]

        local_advantages = [
            {
                name: float(component["group_advantage"])
                for name, component in credit["components"].items()
            }
            for credit in local_credits
        ]
        local_eligibility = [
            {
                name: bool(component["eligible"])
                for name, component in credit["components"].items()
            }
            for credit in local_credits
        ]
        local_active = [
            bool(row.get("dapo_selected_for_update", False))
            if self._dapo_enabled()
            else True
            for row in local_rows
        ]

        a0_size = len(a0_generated)
        _, microbatch_size = self._backward_mode()
        if a0_size % microbatch_size:
            raise RuntimeError(
                "actor-only A0 size must be divisible by learner microbatch size: "
                f"a0_size={a0_size}, microbatch_size={microbatch_size}"
            )
        schedule = [
            LearnerChunk("a0", start, start + microbatch_size)
            for start in range(0, a0_size, microbatch_size)
        ]
        prepared_batches = self._prepare_streaming_batches(deepcopy(a0_generated), schedule)
        for batch, chunk in zip(prepared_batches, schedule):
            start, end = chunk.start, chunk.end
            self._attach_component_credit(
                batch,
                a0_generated[start:end],
                local_advantages[start:end],
                local_eligibility[start:end],
                rollout="a0",
                active=local_active[start:end],
            )
            if self._dapo_enabled() and "dapo_policy" not in batch["vf_component_masks"]:
                completion_mask = batch["completion_mask"]
                batch["vf_component_masks"]["dapo_policy"] = torch.zeros_like(
                    completion_mask, dtype=torch.bool
                )
                batch["vf_component_advantages"]["dapo_policy"] = torch.zeros(
                    completion_mask.shape[0],
                    dtype=torch.float32,
                    device=completion_mask.device,
                )
                batch["vf_component_weights"]["dapo_policy"] = 1.0
            batch["vf_dapo_effective_rows"] = sum(local_active[start:end])
            batch["vf_dapo_padding_rows"] = chunk.size - sum(
                local_active[start:end]
            )
            batch["vf_rollout"] = "a0"
        accounting = self._streaming_policy_accounting(
            prepared_batches,
            overlong_filter=self.overlong_filter,
        )
        self._apply_dapo_token_normalization(prepared_batches, accounting)
        branch_counts = accounting["branch_counts"]
        component_denominators = accounting["component_denominators"]
        covered_rows = int(accounting["covered_rows"])
        total_active = int(accounting["total_active"])
        total_policy_tokens = int(accounting["total_policy_tokens"])
        global_policy_tokens = int(accounting.get("global_policy_tokens", total_policy_tokens))
        token_mean_normalizer = float(
            accounting.get("token_mean_normalizer", total_policy_tokens)
        )
        physical_declared_tokens = int(
            accounting.get("physical_declared_tokens", global_policy_tokens)
        )
        padding_rows = int(accounting.get("padding_rows", 0))
        if total_active == 0 or covered_rows != total_active:
            raise RuntimeError(
                f"actor-only active-row coverage mismatch: active={total_active}, covered={covered_rows}"
            )
        for batch in prepared_batches:
            batch["vf_branch_active_counts"] = dict(branch_counts)
            batch["vf_component_denominators"] = dict(component_denominators)
            batch["vf_total_active_count"] = total_active
            batch["vf_total_policy_tokens"] = total_policy_tokens
            batch["vf_global_policy_tokens"] = global_policy_tokens
            batch["vf_token_mean_normalizer"] = token_mean_normalizer
            batch["vf_physical_declared_tokens"] = physical_declared_tokens
            batch["vf_total_padding_rows"] = padding_rows
        bundle = {
            "vf_dual_rollout": True,
            "vf_actor_only": True,
            "vf_backward_mode": "branch",
            "vf_backward_chunks": prepared_batches,
            "vf_a0_size": a0_size,
            "vf_a1_size": 0,
            "vf_total_policy_tokens": total_policy_tokens,
            "vf_global_policy_tokens": global_policy_tokens,
            "vf_token_mean_normalizer": token_mean_normalizer,
            "vf_physical_declared_tokens": physical_declared_tokens,
            "vf_total_padding_rows": padding_rows,
        }
        if not self._dapo_enabled():
            self._append_trajectories(local_rows, local_credits)
        return [bundle]

    def _generate_and_score_completions(self, inputs):
        self._vf_rollout_call += 1
        if self._actor_only_enabled():
            return self._generate_actor_only_completions(inputs)
        rank = int(self.accelerator.process_index)
        max_workers = max(1, int(os.environ.get("VF_SERVICE_WORKERS", "8")))
        backend = editor_backend()

        original_paths = [source_image_path(dict(data)) for data in inputs]
        editor_routes = [select_editor_url(index, rank=rank, backend=backend) for index in range(len(inputs))]
        judger_routes = [select_judger_url(index) for index in range(len(inputs))]
        editor_request_indices = [
            trajectory_request_index(
                rollout_call=self._vf_rollout_call,
                rank=rank,
                completion_index=index,
            )
            for index in range(len(inputs))
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            j0_futures = [
                executor.submit(request_judger_score, str(path), judger_url=judger_routes[index])
                for index, path in enumerate(original_paths)
            ]

            a0_generated = super()._generate_completions(inputs)
            edit_futures: list[Future | None] = []
            for index, data in enumerate(a0_generated):
                payload = parse_tokenizable_actor_json(self._completion_text(data))
                internal_payload = to_internal_actor_payload(payload) if payload else None
                editing = str(internal_payload.get("editing") or "").strip() if internal_payload else ""
                if not editing:
                    edit_futures.append(None)
                    continue
                edit_futures.append(
                    executor.submit(
                        request_image_edit,
                        image_path=str(original_paths[index]),
                        editing=editing,
                        request_index=editor_request_indices[index],
                        completion_index=index,
                        backend=backend,
                        editor_url=editor_routes[index],
                        comfy_request=request_comfy_edit,
                    )
                )

            edit_results = [_future_result(future, "editor") for future in edit_futures]
            edited_paths: list[str | None] = []
            a1_eligible: list[bool] = []
            a1_prompts: list[dict[str, Any]] = []
            for index, result in enumerate(edit_results):
                edited_path = result.get("edited_path") or result.get("path")
                eligible = bool(result.get("status") == "success" and edited_path)
                edited_paths.append(str(edited_path) if eligible else None)
                a1_eligible.append(eligible)
                a1_prompts.append(prepare_a1_record(a0_generated[index], edited_path or original_paths[index]))

            j1_futures = [
                executor.submit(request_judger_score, str(path), judger_url=judger_routes[index])
                if eligible and path
                else None
                for index, (eligible, path) in enumerate(zip(a1_eligible, edited_paths))
            ]
            a1_generated = super()._generate_completions(a1_prompts)
            j0_results = [_future_result(future, "judger_original") for future in j0_futures]
            j1_results = [_future_result(future, "judger_edited") for future in j1_futures]

        local_rows: list[dict[str, Any]] = []
        for index, (a0_data, a1_data) in enumerate(zip(a0_generated, a1_generated)):
            group_id = str(a0_data.get("sample_id") or a0_data.get("prompt_id") or "")
            if not group_id:
                raise RuntimeError("dual rollout requires sample_id or prompt_id")
            trajectory_id = stable_trajectory_id(
                run_id=os.environ.get("VF_RUN_ID", "unset-run"),
                phase="dual_rollout",
                rank=rank,
                reward_call=self._vf_rollout_call,
                sample_id=group_id,
                completion_index=index,
            )
            a0_payload = parse_tokenizable_actor_json(self._completion_text(a0_data))
            a0_internal = to_internal_actor_payload(a0_payload) if a0_payload else None
            reasons = str(
                a0_payload.get("reasons") or a0_payload.get("reason") or ""
            ) if a0_payload else ""
            suggestion = str(a0_payload.get("suggestion") or "") if a0_payload else ""
            editing = str(a0_internal.get("editing") or "") if a0_internal else ""
            row = {
                "schema_version": "vf_dual_rollout_v2",
                "actor_schema": actor_schema(),
                "row_id": f"{rank}:{self._vf_rollout_call}:{index}",
                "trajectory_id": trajectory_id,
                "group_id": group_id,
                "rank": rank,
                "rollout_call": self._vf_rollout_call,
                "completion_index": index,
                "source_image_path": str(original_paths[index]),
                "edited_image_path": edited_paths[index],
                "editor_backend": backend,
                "editor_url": editor_routes[index],
                "editor_request_index": editor_request_indices[index],
                "editor_seed": edit_results[index].get("seed"),
                "editor_profile": edit_results[index].get("profile_name"),
                "a1_eligible": a1_eligible[index],
                "a0_text": self._completion_text(a0_data),
                "a1_text": self._completion_text(a1_data),
                "a0_finish_reason": a0_data.get("finish_reason"),
                "a1_finish_reason": a1_data.get("finish_reason"),
                "a0_is_truncated": bool(a0_data.get("is_truncated")),
                "a1_is_truncated": bool(a1_data.get("is_truncated")),
                "reasons": reasons,
                "suggestion": suggestion,
                "editing": editing,
                "target_mean": _float(a0_data.get("target_mean")),
                "target_std": _float(a0_data.get("target_std")) or 1.0,
                "j0": score_payload_mean(j0_results[index]),
                "j1": score_payload_mean(j1_results[index]),
                "edit_result": edit_results[index],
                "judger_original": j0_results[index],
                "judger_edited": j1_results[index],
                "sampling": {
                    "presence_penalty": self.request_config.presence_penalty,
                    "repetition_penalty": self.request_config.repetition_penalty,
                },
            }
            row["a0_format_boundary"] = classify_format_boundary(
                row["a0_text"],
                finish_reason=row["a0_finish_reason"],
                is_truncated=row["a0_is_truncated"],
            )[0]
            row["a1_format_boundary"] = classify_format_boundary(
                row["a1_text"],
                finish_reason=row["a1_finish_reason"],
                is_truncated=row["a1_is_truncated"],
            )[0]
            row["failure_owner"] = self._failure_owner(row)
            local_rows.append(row)

        credit_by_id = self._build_global_credit(local_rows)
        local_credits = [credit_by_id[row["row_id"]] for row in local_rows]
        local_advantages = [
            {
                name: float(component["group_advantage"])
                for name, component in credit["components"].items()
            }
            for credit in local_credits
        ]
        local_eligibility = [
            {
                name: bool(component["eligible"])
                for name, component in credit["components"].items()
            }
            for credit in local_credits
        ]

        a0_size = len(a0_generated)
        combined_generated = deepcopy(a0_generated) + deepcopy(a1_generated)
        mode, microbatch_size = self._backward_mode()
        schedule = build_backward_schedule(
            mode,
            a0_size=a0_size,
            a1_size=len(a1_generated),
            microbatch_size=microbatch_size,
        )
        if mode == "combined":
            combined_batches = self._prepare_batch_inputs(combined_generated)
            if len(combined_batches) != 1:
                raise RuntimeError("dual rollout requires one combined prepared batch per optimizer step")
            combined_batch = combined_batches[0]
            self._attach_component_credit(
                combined_batch,
                a0_generated,
                local_advantages,
                local_eligibility,
                rollout="a0",
                active=[True] * len(a0_generated),
                row_offset=0,
            )
            self._attach_component_credit(
                combined_batch,
                a1_generated,
                local_advantages,
                local_eligibility,
                rollout="a1",
                active=a1_eligible,
                row_offset=a0_size,
            )
            bundle = {"vf_dual_rollout": True, "combined": combined_batch, "a0_size": a0_size}
        else:
            prepared_batches = self._prepare_streaming_batches(combined_generated, schedule)
            for batch, chunk in zip(prepared_batches, schedule):
                if chunk.rollout == "a0":
                    start, end = chunk.start, chunk.end
                    self._attach_component_credit(
                        batch,
                        a0_generated[start:end],
                        local_advantages[start:end],
                        local_eligibility[start:end],
                        rollout="a0",
                        active=[True] * chunk.size,
                    )
                else:
                    start, end = chunk.start - a0_size, chunk.end - a0_size
                    self._attach_component_credit(
                        batch,
                        a1_generated[start:end],
                        local_advantages[start:end],
                        local_eligibility[start:end],
                        rollout="a1",
                        active=a1_eligible[start:end],
                    )
                batch["vf_rollout"] = chunk.rollout

            accounting = self._streaming_policy_accounting(
                prepared_batches,
                overlong_filter=self.overlong_filter,
            )
            branch_counts = accounting["branch_counts"]
            component_denominators = accounting["component_denominators"]
            covered_rows = int(accounting["covered_rows"])
            total_active = int(accounting["total_active"])
            if total_active == 0 or covered_rows != total_active:
                raise RuntimeError(
                    f"streaming active-row coverage mismatch: active={total_active}, covered={covered_rows}"
                )
            for batch in prepared_batches:
                batch["vf_branch_active_counts"] = dict(branch_counts)
                batch["vf_component_denominators"] = dict(component_denominators)
                batch["vf_total_active_count"] = total_active
            bundle = {
                "vf_dual_rollout": True,
                "vf_backward_mode": mode,
                "vf_backward_chunks": prepared_batches,
                "vf_a0_size": a0_size,
                "vf_a1_size": len(a1_generated),
            }
        self._append_trajectories(local_rows, local_credits)
        return [bundle]

    def _component_reference_kl_loss(
        self,
        *,
        per_token_logps: torch.Tensor,
        inputs: Mapping[str, Any],
        completion_mask: torch.Tensor,
        component_denominators: Mapping[str, float | int] | None = None,
        loss_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
        if not self._vf_component_kl_betas:
            zero = per_token_logps.sum() * 0.0
            return zero, {}, 0
        if self.beta == 0.0 or self.kl_in_reward:
            raise RuntimeError(
                "field component KL requires a loaded reference model and loss-side KL"
            )
        ref_per_token_logps = inputs.get("ref_per_token_logps")
        if ref_per_token_logps is None:
            raise RuntimeError("field component KL requires ref_per_token_logps")
        missing_masks = sorted(
            name
            for name in self._vf_component_kl_betas
            if name not in inputs["vf_component_masks"]
        )
        if missing_masks:
            if inputs.get("vf_rollout") == "a1":
                zero = per_token_logps.sum() * 0.0
                return zero, {}, 0
            raise RuntimeError(
                "field component KL masks are missing: " + ", ".join(missing_masks)
            )
        ref_minus_current = torch.clamp(
            ref_per_token_logps - per_token_logps,
            min=-20,
            max=20,
        )
        per_token_kl = torch.clamp(
            torch.exp(ref_minus_current) - ref_minus_current - 1,
            min=0,
            max=10,
        )
        return compute_component_kl_loss(
            per_token_kl=per_token_kl,
            completion_mask=completion_mask,
            component_masks=inputs["vf_component_masks"],
            component_betas=self._vf_component_kl_betas,
            component_denominators=component_denominators,
            loss_scale=loss_scale,
            normalization="sequence_mean",
        )

    def _combined_grpo_loss(self, model, inputs: dict[str, Any], a0_size: int) -> torch.Tensor:
        completion_mask = self._effective_completion_mask(
            inputs["completion_mask"],
            inputs.get("truncated_mask"),
            overlong_filter=self.overlong_filter,
        )
        component_union = torch.zeros(completion_mask.shape[0], dtype=torch.bool, device=completion_mask.device)
        for mask in inputs["vf_component_masks"].values():
            component_union |= (mask.bool() & completion_mask).sum(-1) > 0
        completion_mask = completion_mask & component_union.unsqueeze(-1)
        active_rows = completion_mask.sum(-1) > 0
        batch_size = completion_mask.shape[0]
        if not 0 < a0_size < batch_size:
            raise RuntimeError(f"invalid combined rollout split: a0={a0_size}, batch={batch_size}")
        a0_active_rows = active_rows[:a0_size]
        a1_active_rows = active_rows[a0_size:]
        a0_active_count = int(a0_active_rows.sum().item())
        a1_active_count = int(a1_active_rows.sum().item())
        if a0_active_count + a1_active_count == 0:
            raise RuntimeError("dual rollout produced no active policy tokens")

        # Qwen3.5 FLA backward is not reentrant-safe when two independent learner
        # forward graphs are retained until one combined backward. A0 and A1 are
        # therefore encoded together and evaluated in one model forward.
        with saved_tensor_cpu_offload(
            self._vf_activation_offload,
            budget_bytes=self._vf_activation_offload_budget_bytes,
        ) as offload_stats:
            per_token_logps, _ = self._get_per_token_logps_and_entropies_single(
                model,
                inputs,
                compute_entropy=False,
            )
        if self._vf_activation_offload and not self._vf_activation_offload_audited:
            print(
                "[vf-activation-offload] "
                f"rank={os.environ.get('RANK', 'unknown')} "
                "enabled=1 "
                "backend=torch.autograd.graph.saved_tensors_hooks.selective_cpu_v1 "
                f"budget_bytes={offload_stats.budget_bytes} "
                f"min_tensor_mib={MIN_TENSOR_MIB} "
                f"offloaded_bytes={offload_stats.offloaded_bytes} "
                f"offloaded_tensors={offload_stats.offloaded_tensors} "
                f"seen_cuda_bytes={offload_stats.seen_cuda_bytes} "
                "pin_memory=0 exact_autograd=1",
                flush=True,
            )
            self._vf_activation_offload_audited = True
        old_per_token_logps = inputs["old_per_token_logps"]
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        log_ratio = per_token_logps - old_per_token_logps
        ratio = torch.exp(torch.clamp(log_ratio, min=-20, max=20))
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)

        rollout_per_token_logps = inputs.get("rollout_per_token_logps")
        if rollout_per_token_logps is None:
            raise RuntimeError("processed vLLM rollout logprobs are required for presence-penalty correction")
        _, rollout_is_weights = self._get_rollout_is_correction(
            old_per_token_logps,
            rollout_per_token_logps,
            completion_mask,
        )
        if rollout_is_weights is None:
            raise RuntimeError("rollout importance-sampling correction must be enabled")
        _, component_losses, component_active_count = compute_component_policy_loss(
            ratio=ratio,
            clipped_ratio=clipped_ratio,
            rollout_is_weights=rollout_is_weights,
            completion_mask=completion_mask,
            component_masks=inputs["vf_component_masks"],
            component_advantages=inputs["vf_component_advantages"],
            component_weights=inputs["vf_component_weights"],
            normalization="token_mean" if self._dapo_enabled() else "sequence_mean",
        )

        zero = torch.where(completion_mask, per_token_logps, torch.zeros_like(per_token_logps)).sum() * 0.0
        a0_components = (
            "dapo_policy",
            "grpo_policy",
            "format_a0",
            "rating0",
            "reasoning",
            "soft_overlong",
            "edit_gate",
            "edit_gain",
        )
        a1_components = ("format_a1", "delta_margin", "rating1_anchor")
        a0_loss = sum((component_losses[name] for name in a0_components if name in component_losses), zero)
        a1_loss = sum((component_losses[name] for name in a1_components if name in component_losses), zero)

        component_kl_losses: dict[str, torch.Tensor] = {}
        global_completion_kl_mean: torch.Tensor | None = None
        global_completion_kl_loss: torch.Tensor | None = None
        if self._vf_component_kl_betas:
            component_kl_total, component_kl_losses, _ = self._component_reference_kl_loss(
                per_token_logps=per_token_logps,
                inputs=inputs,
                completion_mask=completion_mask,
            )
            a0_loss = a0_loss + component_kl_total
        elif self.beta != 0.0 and not self.kl_in_reward:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            safe_ratio = torch.clamp(ref_per_token_logps - per_token_logps, min=-20, max=20)
            per_token_kl = torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-10, max=10)
            token_counts = completion_mask.sum(-1).clamp(min=1)
            masked_kl = torch.where(completion_mask, per_token_kl, torch.zeros_like(per_token_kl))
            sequence_kl = masked_kl.sum(-1) / token_counts
            global_completion_kl_mean = sequence_kl[active_rows].mean()
            global_completion_kl_loss = self.beta * global_completion_kl_mean
            if a0_active_count:
                a0_loss = a0_loss + self.beta * sequence_kl[:a0_size][a0_active_rows].mean()
            if a1_active_count:
                a1_loss = a1_loss + self.beta * sequence_kl[a0_size:][a1_active_rows].mean()

        try:
            loss = combine_active_branch_losses(
                [(a0_loss, a0_active_count), (a1_loss, a1_active_count)]
            )
        except ValueError as exc:
            raise RuntimeError("dual rollout produced no active policy tokens") from exc
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("dual rollout produced a non-finite combined loss")

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["vf/a0_loss"].append(float(a0_loss.detach().item()))
        self._metrics[mode]["vf/a1_loss"].append(float(a1_loss.detach().item()))
        self._metrics[mode]["vf/a0_active_rate"].append(float(a0_active_rows.float().mean().item()))
        self._metrics[mode]["vf/a1_active_rate"].append(float(a1_active_rows.float().mean().item()))
        for name, component_loss in component_losses.items():
            rollout = "a0" if name in a0_components else "a1"
            self._metrics[mode][f"vf/{rollout}_{name}_loss"].append(
                float(component_loss.detach().item())
            )
        for name, component_kl_loss in component_kl_losses.items():
            self._metrics[mode][f"vf/a0_{name}_kl_loss"].append(
                float(component_kl_loss.detach().item())
            )
        if global_completion_kl_mean is not None and global_completion_kl_loss is not None:
            self._metrics[mode]["vf/global_completion_kl_mean"].append(
                float(global_completion_kl_mean.detach().item())
            )
            self._metrics[mode]["vf/global_completion_kl_loss"].append(
                float(global_completion_kl_loss.detach().item())
            )
            self._metrics[mode]["vf/global_completion_kl_apply_count"].append(1.0)
            self._metrics[mode]["vf/component_kl_apply_count"].append(0.0)
        if component_active_count != int(active_rows.sum().item()):
            raise RuntimeError("combined component masks do not cover every active completion")
        return loss

    def _streaming_chunk_loss(
        self,
        model,
        inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        completion_mask = inputs.get("vf_policy_completion_mask")
        if completion_mask is None:
            completion_mask = self._effective_completion_mask(
                inputs["completion_mask"],
                inputs.get("truncated_mask"),
                overlong_filter=self.overlong_filter,
            )
            component_union = torch.zeros(
                completion_mask.shape[0], dtype=torch.bool, device=completion_mask.device
            )
            for mask in inputs["vf_component_masks"].values():
                component_union |= (mask.bool() & completion_mask).sum(-1) > 0
            completion_mask = completion_mask & component_union.unsqueeze(-1)
        else:
            completion_mask = completion_mask.bool()
        active_rows = completion_mask.sum(-1) > 0
        rollout = str(inputs["vf_rollout"])
        branch_counts = inputs["vf_branch_active_counts"]
        branch_count = int(branch_counts[rollout])
        total_active = int(inputs["vf_total_active_count"])
        if total_active <= 0:
            raise RuntimeError("streaming bundle has no active policy tokens")

        with saved_tensor_cpu_offload(
            self._vf_activation_offload,
            budget_bytes=self._vf_activation_offload_budget_bytes,
        ) as offload_stats:
            per_token_logps, _ = self._get_per_token_logps_and_entropies_single(
                model,
                inputs,
                compute_entropy=False,
            )
        if self._vf_activation_offload and not self._vf_activation_offload_audited:
            print(
                "[vf-activation-offload] "
                f"rank={os.environ.get('RANK', 'unknown')} "
                "enabled=1 "
                "backend=torch.autograd.graph.saved_tensors_hooks.selective_cpu_v1 "
                f"budget_bytes={offload_stats.budget_bytes} "
                f"min_tensor_mib={MIN_TENSOR_MIB} "
                f"offloaded_bytes={offload_stats.offloaded_bytes} "
                f"offloaded_tensors={offload_stats.offloaded_tensors} "
                f"seen_cuda_bytes={offload_stats.seen_cuda_bytes} "
                "pin_memory=0 exact_autograd=1",
                flush=True,
            )
            self._vf_activation_offload_audited = True
        old_per_token_logps = inputs["old_per_token_logps"]
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        log_ratio = per_token_logps - old_per_token_logps
        ratio = torch.exp(torch.clamp(log_ratio, min=-20, max=20))
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        rollout_per_token_logps = inputs.get("rollout_per_token_logps")
        if rollout_per_token_logps is None:
            raise RuntimeError("processed vLLM rollout logprobs are required for presence-penalty correction")
        _, rollout_is_weights = self._get_rollout_is_correction(
            old_per_token_logps,
            rollout_per_token_logps,
            completion_mask,
        )
        if rollout_is_weights is None:
            raise RuntimeError("rollout importance-sampling correction must be enabled")

        branch_weight = branch_count / total_active
        loss, component_contributions, component_active_count = compute_component_policy_loss(
            ratio=ratio,
            clipped_ratio=clipped_ratio,
            rollout_is_weights=rollout_is_weights,
            completion_mask=completion_mask,
            component_masks=inputs["vf_component_masks"],
            component_advantages=inputs["vf_component_advantages"],
            component_weights=inputs["vf_component_weights"],
            component_denominators=inputs["vf_component_denominators"],
            loss_scale=branch_weight,
            normalization="token_mean" if self._dapo_enabled() else "sequence_mean",
        )
        raw_component_partials = {
            name: value / branch_weight if branch_count else value.detach() * 0.0
            for name, value in component_contributions.items()
        }
        raw_branch_partial = sum(
            raw_component_partials.values(),
            per_token_logps.sum() * 0.0,
        )

        global_completion_kl_sum = per_token_logps.sum().detach() * 0.0
        global_completion_kl_active_count = 0
        if self._vf_component_kl_betas:
            component_kl_total, component_kl_contributions, _ = (
                self._component_reference_kl_loss(
                    per_token_logps=per_token_logps,
                    inputs=inputs,
                    completion_mask=completion_mask,
                    component_denominators=inputs["vf_component_denominators"],
                    loss_scale=branch_weight,
                )
            )
            loss = loss + component_kl_total
            for name, value in component_kl_contributions.items():
                metric_name = f"{name}_kl"
                raw_value = (
                    value / branch_weight
                    if branch_count
                    else value.detach() * 0.0
                )
                raw_component_partials[metric_name] = raw_value
                raw_branch_partial = raw_branch_partial + raw_value
        elif self.beta != 0.0 and not self.kl_in_reward:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            safe_ratio = torch.clamp(ref_per_token_logps - per_token_logps, min=-20, max=20)
            per_token_kl = torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-10, max=10)
            token_counts = completion_mask.sum(-1).clamp(min=1)
            masked_kl = torch.where(completion_mask, per_token_kl, torch.zeros_like(per_token_kl))
            sequence_kl = masked_kl.sum(-1) / token_counts
            kl_sum = sequence_kl[active_rows].sum()
            loss = loss + self.beta * kl_sum / total_active
            global_completion_kl_sum = kl_sum.detach()
            global_completion_kl_active_count = int(active_rows.sum().item())
            if branch_count:
                raw_branch_partial = raw_branch_partial + self.beta * kl_sum / branch_count

        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite streaming chunk loss: rollout={rollout}")
        dapo_policy_token_count = 0
        dapo_clip_high_count = 0
        dapo_clip_low_count = 0
        if self._dapo_enabled():
            dapo_mask = inputs["vf_component_masks"]["dapo_policy"].bool() & completion_mask
            dapo_advantages = inputs["vf_component_advantages"]["dapo_policy"].to(
                device=ratio.device,
                dtype=ratio.dtype,
            ).unsqueeze(-1)
            dapo_policy_token_count = int(dapo_mask.sum().item())
            dapo_clip_high_count = int(
                ((ratio > 1 + self.epsilon_high) & (dapo_advantages > 0) & dapo_mask).sum().item()
            )
            dapo_clip_low_count = int(
                ((ratio < 1 - self.epsilon_low) & (dapo_advantages < 0) & dapo_mask).sum().item()
            )
        return loss, {
            "rollout": rollout,
            "active_count": int(active_rows.sum().item()),
            "component_active_count": component_active_count,
            "raw_branch_partial": raw_branch_partial.detach(),
            "raw_component_partials": {
                name: value.detach() for name, value in raw_component_partials.items()
            },
            "global_completion_kl_sum": global_completion_kl_sum,
            "global_completion_kl_active_count": global_completion_kl_active_count,
            "dapo_policy_token_count": dapo_policy_token_count,
            "dapo_clip_high_count": dapo_clip_high_count,
            "dapo_clip_low_count": dapo_clip_low_count,
        }

    def _backward_streaming_loss(self, loss: torch.Tensor, *, is_last: bool) -> None:
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.deepspeed_engine_wrapped.backward(
                loss,
                sync_gradients=is_last,
                scale_wrt_gas=False,
            )
        else:
            self.accelerator.backward(loss)

    def _streaming_training_step(self, model, bundle: dict[str, Any], num_items_in_batch=None) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        self._vf_validate_selective_gc_runtime(model)
        chunks = bundle["vf_backward_chunks"]
        if not chunks:
            raise RuntimeError("streaming backward requires at least one chunk")
        mode = "train" if self.model.training else "eval"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        total_loss = None
        branch_losses = {"a0": 0.0, "a1": 0.0}
        component_losses: dict[str, float] = {}
        covered = 0
        executed_chunks = 0
        dapo_policy_tokens = 0
        dapo_clip_high = 0
        dapo_clip_low = 0
        global_completion_kl_sum = 0.0
        global_completion_kl_active_count = 0
        sync_each_chunk = os.environ.get("VF_SYNC_EACH_BACKWARD_CHUNK", "0")
        if sync_each_chunk not in {"0", "1"}:
            raise RuntimeError(
                f"VF_SYNC_EACH_BACKWARD_CHUNK must be 0 or 1, got {sync_each_chunk!r}"
            )

        for chunk_index, chunk in enumerate(chunks):
            with self.compute_loss_context_manager():
                loss, stats = self._streaming_chunk_loss(model, chunk)
            self._backward_streaming_loss(loss, is_last=chunk_index == len(chunks) - 1)
            if sync_each_chunk == "1" and torch.cuda.is_available():
                # ZeRO-3 and repeated checkpointed forwards can otherwise reuse
                # asynchronous buffers before a prior chunk has fully retired.
                torch.cuda.synchronize()
            detached = loss.detach()
            total_loss = detached if total_loss is None else total_loss + detached
            rollout = stats["rollout"]
            branch_losses[rollout] += float(stats["raw_branch_partial"].item())
            for name, value in stats["raw_component_partials"].items():
                component_losses[name] = component_losses.get(name, 0.0) + float(value.item())
            covered += int(stats["component_active_count"])
            dapo_policy_tokens += int(stats["dapo_policy_token_count"])
            dapo_clip_high += int(stats["dapo_clip_high_count"])
            dapo_clip_low += int(stats["dapo_clip_low_count"])
            global_completion_kl_sum += float(
                stats["global_completion_kl_sum"].item()
            )
            global_completion_kl_active_count += int(
                stats["global_completion_kl_active_count"]
            )
            executed_chunks += 1
            del loss, stats

        expected_covered = int(chunks[0]["vf_total_active_count"])
        if covered != expected_covered:
            raise RuntimeError(
                f"streaming policy-token coverage mismatch: covered={covered}, expected={expected_covered}"
            )
        if total_loss is None or not bool(torch.isfinite(total_loss)):
            raise RuntimeError("streaming backward produced no finite loss")
        if global_completion_kl_active_count:
            if global_completion_kl_active_count != expected_covered:
                raise RuntimeError(
                    "global completion KL active-row coverage mismatch: "
                    f"covered={global_completion_kl_active_count}, "
                    f"expected={expected_covered}"
                )
            global_completion_kl_mean = (
                global_completion_kl_sum / global_completion_kl_active_count
            )
            self._metrics[mode]["vf/global_completion_kl_mean"].append(
                global_completion_kl_mean
            )
            self._metrics[mode]["vf/global_completion_kl_loss"].append(
                float(self.beta) * global_completion_kl_mean
            )
            self._metrics[mode]["vf/global_completion_kl_apply_count"].append(1.0)
            self._metrics[mode]["vf/component_kl_apply_count"].append(0.0)
        elapsed = time.perf_counter() - started
        a0_count = int(chunks[0]["vf_branch_active_counts"]["a0"])
        a1_count = int(chunks[0]["vf_branch_active_counts"]["a1"])
        a0_size = int(bundle["vf_a0_size"])
        a1_size = int(bundle["vf_a1_size"])
        self._metrics[mode]["vf/a0_loss"].append(branch_losses["a0"])
        self._metrics[mode]["vf/a1_loss"].append(branch_losses["a1"])
        self._metrics[mode]["vf/a0_active_rate"].append(a0_count / a0_size)
        self._metrics[mode]["vf/a1_active_rate"].append(a1_count / a1_size if a1_size else 0.0)
        for name, value in component_losses.items():
            rollout = "a0" if name in {
                "dapo_policy",
                "grpo_policy",
                "format_a0",
                "rating0",
                "reasoning",
                "soft_overlong",
                "rating0_kl",
                "reasoning_kl",
                "edit_gate",
                "edit_gain",
            } else "a1"
            self._metrics[mode][f"vf/{rollout}_{name}_loss"].append(value)
        self._metrics[mode]["vf/backward_chunk_count"].append(float(executed_chunks))
        self._metrics[mode]["vf/backward_scheduled_chunk_count"].append(float(len(chunks)))
        self._metrics[mode]["vf/backward_seconds"].append(float(elapsed))
        self._metrics[mode]["vf/optimizer_update_executed"].append(1.0)
        self._metrics[mode]["vf/scheduler_update_executed"].append(1.0)
        if self._dapo_enabled():
            self._metrics[mode]["vf/dapo_low_effective_training_skip"].append(0.0)
            expected_policy_tokens = int(bundle["vf_total_policy_tokens"])
            if dapo_policy_tokens != expected_policy_tokens:
                raise RuntimeError(
                    "DAPO policy-token denominator mismatch: "
                    f"observed={dapo_policy_tokens}, expected={expected_policy_tokens}"
                )
            denominator = max(dapo_policy_tokens, 1)
            self._metrics[mode]["vf/dapo_policy_tokens"].append(float(dapo_policy_tokens))
            self._metrics[mode]["vf/dapo_global_policy_tokens"].append(
                float(bundle["vf_global_policy_tokens"])
            )
            self._metrics[mode]["vf/dapo_token_mean_normalizer"].append(
                float(bundle["vf_token_mean_normalizer"])
            )
            self._metrics[mode]["vf/dapo_physical_policy_tokens"].append(
                float(bundle["vf_physical_declared_tokens"])
            )
            self._metrics[mode]["vf/dapo_training_padding_rows"].append(
                float(bundle["vf_total_padding_rows"])
            )
            self._metrics[mode]["vf/dapo_clip_high_frac"].append(dapo_clip_high / denominator)
            self._metrics[mode]["vf/dapo_clip_low_frac"].append(dapo_clip_low / denominator)
        if torch.cuda.is_available():
            self._metrics[mode]["vf/backward_peak_allocated_gib"].append(
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )
            self._metrics[mode]["vf/backward_peak_reserved_gib"].append(
                torch.cuda.max_memory_reserved() / (1024 ** 3)
            )
        return total_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise RuntimeError("dual rollout trainer does not support return_outputs=True")
        if isinstance(inputs, list):
            if len(inputs) != 1:
                raise RuntimeError("dual rollout expects exactly one optimizer bundle")
            inputs = inputs[0]
        if not isinstance(inputs, dict) or not inputs.get("vf_dual_rollout"):
            return super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)

        if inputs.get("vf_backward_mode") in {"branch", "microbatch"}:
            raise RuntimeError("streaming bundles must be handled by _streaming_training_step")

        self._vf_validate_selective_gc_runtime(model)
        return self._combined_grpo_loss(model, inputs["combined"], int(inputs["a0_size"]))


orms["vf_dual_rollout_placeholder"] = _DualRolloutPlaceholderReward
TrainerFactory.TRAINER_MAPPING["grpo"] = "vf_dual_rollout_trainer.VFDualRolloutGRPOTrainer"
