from __future__ import annotations

import json
import math
import os
import time
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

try:
    import requests
except Exception:  # pragma: no cover - tests can monkeypatch service calls.
    requests = None

try:
    from swift.rewards import ORM, orms
except Exception:  # pragma: no cover - local unit tests stub this path.
    class ORM:  # type: ignore[no-redef]
        pass

    orms = {}


PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from actor_contract import (
    actor_schema,
    actor_payload_errors,
    actor_rating_number,
    parse_actor_json,
    score_number,
    to_internal_actor_payload,
)
from prompt_contract import PROMPT_HASH, PROMPT_VERSION
from readiness import load_manifest, require_training_ready


EXECUTION_PROMPT_VERSION = "vf_whole_image_executor_v1_20260710"

def strip_generation_artifacts(text: str) -> str:
    text = str(text or "").strip()
    text = text.removesuffix("<|im_end|>").strip()
    if text.startswith("```"):
        import re

        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
        text = re.sub(r"\s*```\s*$", "", text).strip()
    return text


def completion_text(text: str) -> str:
    return str(text or "").strip()


def suffix_closure_text(text: str) -> tuple[str, bool]:
    """Legacy closure is intentionally unreachable in the canonical path."""
    return text, False


def parse_json_object(text: str) -> Optional[dict[str, Any]]:
    return parse_actor_json(text)


def actor_payload_field_errors(payload: Any) -> list[str]:
    return actor_payload_errors(payload)


def rating_number(value: Any) -> Optional[float]:
    return actor_rating_number(value)


def l2_reward(error: float, tau: float = 1.0) -> float:
    tau = max(float(tau), 1e-6)
    return math.exp(-0.5 * (float(error) * float(error)) / tau)


def target_at(values: Any, idx: int, default: Any = None) -> Any:
    if isinstance(values, (list, tuple)):
        return values[idx] if idx < len(values) else default
    return default


def image_path_at(kwargs: dict[str, Any], idx: int) -> Optional[str]:
    def resolve(path: Any) -> Optional[str]:
        if path is None:
            return None
        text = str(path)
        if not text:
            return None
        if os.path.isabs(text) or not os.environ.get("VF_LOOP_IMAGE_ROOT"):
            return text
        return str(Path(os.environ["VF_LOOP_IMAGE_ROOT"]) / text)

    for key in ("image_path", "image_paths", "source_image", "source_images"):
        value = kwargs.get(key)
        if isinstance(value, (list, tuple)) and idx < len(value):
            return resolve(value[idx])
        if isinstance(value, str) and idx == 0:
            return resolve(value)
    images = kwargs.get("images")
    if isinstance(images, (list, tuple)) and idx < len(images):
        item = images[idx]
        if isinstance(item, (list, tuple)) and item:
            return resolve(item[0])
        if isinstance(item, str):
            return resolve(item)
    return None


def region_prompt(regions: list[dict[str, Any]]) -> str:
    instructions = []
    for region in regions:
        instruction = str(region.get("instruction") or "").strip()
        if instruction:
            instructions.append(instruction)
    if not instructions:
        return ""
    return (
        "Apply this faithful whole-image quality improvement: " + " ".join(instructions) + "\n"
        "Preserve scene content, identity, geometry, composition, text, and unrelated details."
    )


def editing_to_regions(editing: str) -> list[dict[str, Any]]:
    instruction = str(editing or "").strip()
    if not instruction:
        return []
    return [{"bbox": [0.0, 0.0, 1.0, 1.0], "instruction": instruction}]


def smoke_synthetic_region(target: Optional[float], has_actor_editing: bool, image_path: Optional[str]) -> Optional[dict[str, Any]]:
    if os.environ.get("VF_LOOP_SMOKE_INJECT_LOW_TARGET_REGION", "0") != "1":
        return None
    if has_actor_editing or target is None or not image_path:
        return None
    low = float(os.environ.get("VF_LOOP_LOW_THRESHOLD", "3.0"))
    if target > low:
        return None
    instruction = os.environ.get(
        "VF_LOOP_SMOKE_SYNTHETIC_INSTRUCTION",
        "Improve overall perceptual image quality while preserving the original scene content.",
    ).strip()
    if not instruction:
        return None
    return {"bbox": [0.0, 0.0, 1.0, 1.0], "instruction": instruction}


def maybe_logged_completion(text: str) -> Optional[str]:
    if os.environ.get("VF_LOOP_LOG_COMPLETION_TEXT", "0") != "1":
        return None
    limit = max(0, int(os.environ.get("VF_LOOP_COMPLETION_LOG_CHARS", "1200")))
    if limit <= 0:
        return ""
    return str(text or "")[:limit]


def score_payload_mean(payload: dict[str, Any] | None) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    for key in ("mean", "score", "after_mean", "before_mean"):
        if key in payload:
            return score_number(payload.get(key))
    return None


def actor_pass2_payload_score(payload: dict[str, Any] | None) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    for key in ("mean", "score"):
        if key in payload:
            return score_number(payload.get(key))
    if "rating" in payload:
        return actor_rating_number(payload.get("rating"))
    nested = payload.get("json")
    if isinstance(nested, dict) and "rating" in nested:
        return actor_rating_number(nested.get("rating"))
    outputs = payload.get("outputs")
    if isinstance(outputs, list):
        scores: list[float] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            score = actor_pass2_payload_score(item)
            if score is not None:
                scores.append(score)
        if scores:
            return sum(scores) / len(scores)
    return None


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is unavailable")
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("service returned non-object JSON")
    return data


def _service_urls(many_key: str, one_key: str, fallback: str) -> list[str]:
    raw_urls = os.environ.get(many_key, "").strip()
    if raw_urls:
        urls = [item.strip().rstrip("/") for item in re.split(r"[\s,]+", raw_urls) if item.strip()]
    else:
        single = os.environ.get(one_key, "").strip()
        urls = [single.rstrip("/")] if single else []
    if not urls:
        urls = [fallback.rstrip("/")]
    return urls


def judger_urls() -> list[str]:
    return _service_urls("VF_LOOP_JUDGER_URLS", "VF_LOOP_JUDGER_URL", "http://127.0.0.1:8207")


def comfy_adapter_urls() -> list[str]:
    return _service_urls(
        "VF_LOOP_COMFY_ADAPTER_URLS",
        "VF_LOOP_COMFY_ADAPTER_URL",
        "http://127.0.0.1:8211",
    )


def actor_pass2_urls() -> list[str]:
    return _service_urls(
        "VF_LOOP_ACTOR_PASS2_URLS",
        "VF_LOOP_ACTOR_PASS2_URL",
        "http://127.0.0.1:8221",
    )


def process_rank() -> int:
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return 0


def select_service_slot(idx: int) -> int:
    return process_rank() + int(idx)


def select_judger_url(idx: int) -> str:
    urls = judger_urls()
    slot = select_service_slot(idx) % len(urls)
    return urls[slot]


def select_comfy_adapter_url(idx: int) -> str:
    urls = comfy_adapter_urls()
    slot = select_service_slot(idx) % len(urls)
    return urls[slot]


def select_actor_pass2_url(idx: int) -> str:
    urls = actor_pass2_urls()
    slot = select_service_slot(idx) % len(urls)
    return urls[slot]


def select_service_urls(idx: int) -> tuple[int, str, str, str]:
    slot = select_service_slot(idx)
    adapters = comfy_adapter_urls()
    judges = judger_urls()
    actors = actor_pass2_urls()
    adapter = adapters[slot % len(adapters)]
    judger = judges[slot % len(judges)]
    actor = actors[slot % len(actors)]
    return slot, adapter, judger, actor


def request_comfy_edit(
    image_path: str,
    regions: list[dict[str, Any]],
    idx: int,
    adapter_url: str | None = None,
) -> dict[str, Any]:
    if os.environ.get("VF_LOOP_ENABLE_COMFY", "0") != "1":
        return {"status": "skipped", "reason": "comfy_disabled"}
    url = (adapter_url or select_comfy_adapter_url(idx)).rstrip("/") + "/edit"
    return post_json(
        url,
        {
            "image_path": image_path,
            "regions": regions,
            "positive_prompt": region_prompt(regions),
            "request_index": idx,
        },
        float(os.environ.get("VF_LOOP_SERVICE_TIMEOUT", "300")),
    )


def request_judger_score(image_path: str, judger_url: str | None = None) -> dict[str, Any]:
    if os.environ.get("VF_LOOP_ENABLE_JUDGER", "0") != "1":
        return {"mean": None, "status": "skipped", "reason": "judger_disabled"}
    url = (judger_url or select_judger_url(0)).rstrip("/") + "/score_image"
    return post_json(
        url,
        {"image_path": image_path, "repeats": int(os.environ.get("VF_LOOP_JUDGER_REPEATS", "1"))},
        float(os.environ.get("VF_LOOP_SERVICE_TIMEOUT", "300")),
    )


def request_actor_pass2_score(image_path: str, actor_url: str | None = None) -> dict[str, Any]:
    if os.environ.get("VF_LOOP_ENABLE_ACTOR_PASS2", "0") != "1":
        return {"mean": None, "status": "skipped", "reason": "actor_pass2_disabled"}
    url = (actor_url or select_actor_pass2_url(0)).rstrip("/") + "/score_image"
    return post_json(
        url,
        {"image_path": image_path, "repeats": int(os.environ.get("VF_LOOP_ACTOR_PASS2_REPEATS", "1"))},
        float(os.environ.get("VF_LOOP_SERVICE_TIMEOUT", "300")),
    )


def append_trajectory(record: dict[str, Any]) -> None:
    path = os.environ.get("VF_LOOP_TRAJECTORY_LOG", "")
    if not path:
        return
    try:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(json.dumps({"vf_loop_trajectory_error": repr(exc), "path": path}, ensure_ascii=False), flush=True)


def append_pass2_buffer(record: dict[str, Any]) -> None:
    path = os.environ.get("VF_LOOP_PASS2_BUFFER_LOG", "")
    if not path:
        return
    try:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(json.dumps({"vf_loop_pass2_buffer_error": repr(exc), "path": path}, ensure_ascii=False), flush=True)


def float_context_at(kwargs: dict[str, Any], idx: int, keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = kwargs.get(key)
        if isinstance(value, (list, tuple)):
            if idx >= len(value):
                continue
            value = value[idx]
        elif idx != 0:
            continue
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            return value
    return None


def string_context_at(kwargs: dict[str, Any], idx: int, keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = kwargs.get(key)
        if isinstance(value, (list, tuple)):
            if idx >= len(value):
                continue
            value = value[idx]
        elif idx != 0:
            continue
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return None


def pass2_context_at(kwargs: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "a0": float_context_at(kwargs, idx, ("a0", "pass1_rating", "actor_original_rating")),
        "j1": float_context_at(kwargs, idx, ("j1", "judger_edited", "judger_edited_mean")),
        "jd": float_context_at(kwargs, idx, ("jd", "judger_delta", "judger_margin")),
        "j0": float_context_at(kwargs, idx, ("j0", "judger_original", "judger_original_mean")),
        "source_trajectory_id": string_context_at(
            kwargs, idx, ("source_trajectory_id", "trajectory_id", "phase_a_trajectory_id")
        ),
    }


def segment_masks() -> dict[str, bool]:
    return {
        "pass1_valid": False,
        "edit_requested": False,
        "edit_executed": False,
        "judger_original_success": False,
        "judger_edited_success": False,
        "pass2_valid": False,
        "actor_caused_failure": False,
        "infrastructure_failure": False,
    }


def edit_probability_from_target(target: Optional[float]) -> Optional[float]:
    if target is None:
        return None
    return max(0.0, min(1.0, (4.5 - float(target)) / 3.5))


class VisualFeedbackSegmentPhaseAReward(ORM):
    """Phase A: reward original-image analysis, edit decision, and external edit outcome."""

    call_count = 0

    def __call__(self, completions, target_mean=None, **kwargs) -> list[float]:
        self.__class__.call_count += 1
        rewards: list[float] = []
        for idx, text in enumerate(completions):
            reward, record = self.score_one(idx, str(text or ""), target_mean, kwargs)
            rewards.append(reward)
            append_trajectory(record)
        return rewards

    def score_one(self, idx: int, text: str, target_mean: Any, kwargs: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        full_text = completion_text(text)
        raw_payload = parse_json_object(full_text)
        raw_errors = actor_payload_field_errors(raw_payload)
        closed_text, closure_applied = suffix_closure_text(full_text) if raw_errors else (full_text, False)
        payload = raw_payload if not closure_applied else parse_json_object(closed_text)
        errors = actor_payload_field_errors(payload)
        target = score_number(target_at(target_mean, idx, None))
        image_path = image_path_at(kwargs, idx)
        record: dict[str, Any] = {
            "time": time.time(),
            "reward_name": "vf_loop_segment_phase_a",
            "reward_schema_version": "vf_segment_reward_v1",
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": PROMPT_HASH,
            "executor_prompt_version": EXECUTION_PROMPT_VERSION,
            "training_phase": "phase_a",
            "idx": idx,
            "image_path": image_path,
            "target_mean": target,
            "raw_format_success": not raw_errors,
            "raw_format_errors": raw_errors,
            "closure_applied": closure_applied,
            "closed_format_success": not errors,
            "closed_format_errors": errors,
            "format_success": not errors,
            "format_errors": errors,
            "edit_requested": False,
            "edit_status": "not_started",
            "judger_status": "not_started",
            "pass2_status": "not_started",
            "failure_labels": [],
            "components": {},
            "segments": {
                "S0_original_understanding": {},
                "S1_edit_policy": {},
                "S2_external_outcome": {},
            },
            "credit_assignment": {},
            "masks": segment_masks(),
        }
        if os.environ.get("REWARD_COMPLETION_PREFIX", ""):
            record["completion_prefix_used"] = True
        logged_completion = maybe_logged_completion(full_text)
        if logged_completion is not None:
            record["completion_text"] = logged_completion
            record["completion_text_truncated"] = len(str(full_text or "")) > len(logged_completion)
            if closure_applied:
                logged_closed = maybe_logged_completion(closed_text)
                if logged_closed is not None:
                    record["closed_completion_text"] = logged_closed
                    record["closed_completion_text_truncated"] = len(str(closed_text or "")) > len(logged_closed)
        if errors or not isinstance(payload, dict):
            record["failure_labels"].append("format_parse_error")
            record["masks"]["actor_caused_failure"] = True
            return 0.0, record

        internal_payload = to_internal_actor_payload(payload)
        actor_editing = str(internal_payload.get("editing") or "").strip()
        a0 = rating_number(payload.get("rating"))
        actor_regions = editing_to_regions(actor_editing)
        valid_actor_regions = bool(actor_regions)
        synthetic_region = smoke_synthetic_region(target, bool(actor_editing), image_path)
        regions = [synthetic_region] if synthetic_region else actor_regions
        valid_regions = bool(regions)
        record["a0"] = a0
        record["actor_schema"] = actor_schema()
        record["reasons"] = str(payload.get("reasons") or payload.get("reason") or "")
        record["suggestion"] = str(payload.get("suggestion") or "")
        record["editing"] = actor_editing
        record["edit_requested"] = bool(actor_editing)
        record["actor_region_count"] = len(actor_regions)
        record["execution_region_count"] = len(regions)
        record["synthetic_region_injected"] = synthetic_region is not None
        record["masks"]["pass1_valid"] = True
        record["masks"]["edit_requested"] = bool(actor_editing)
        if synthetic_region is not None:
            record["synthetic_region"] = synthetic_region
            record["failure_labels"].append("missing_tool_call")
        record["components"]["R_S0_format"] = 1.0
        record["components"]["R_format"] = 1.0
        edit_gate = self.edit_decision_reward(target, bool(actor_editing), valid_actor_regions)
        record["components"]["R_S1_edit_gate"] = edit_gate
        record["components"]["R_edit_decision"] = edit_gate
        record["segments"]["S0_original_understanding"] = {
            "rating": a0,
            "format_reward": 1.0,
        }
        record["segments"]["S1_edit_policy"] = {
            "editing": actor_editing,
            "target_edit_probability": edit_probability_from_target(target),
            "edit_gate_reward": edit_gate,
        }

        if not regions:
            record["edit_status"] = "no_edit"
            record["pass2_status"] = "not_run_phase_a"
            no_edit_exec = 1.0 if self.no_edit_is_reasonable(target) else 0.0
            record["components"]["R_S1_instruction_exec"] = no_edit_exec
            record["components"]["R_S2_judger_gain"] = 0.0
            record["components"]["R_S2_no_degrade"] = 0.0
            record["components"]["R_judger_gain"] = 0.0
            record["components"]["R_alignment_l2"] = 0.0
            return self.total_reward(record), record

        if not valid_regions:
            record["edit_status"] = "invalid_regions"
            record["pass2_status"] = "not_run_phase_a"
            record["failure_labels"].append("region_selection_failure")
            record["masks"]["actor_caused_failure"] = True
            record["components"]["R_S1_instruction_exec"] = 0.0
            record["components"]["R_S2_judger_gain"] = 0.0
            record["components"]["R_S2_no_degrade"] = 0.0
            record["components"]["R_judger_gain"] = 0.0
            record["components"]["R_alignment_l2"] = 0.0
            return self.total_reward(record), record

        if not image_path:
            record["edit_status"] = "missing_image_path"
            record["pass2_status"] = "not_run_phase_a"
            record["failure_labels"].append("data_issue")
            record["masks"]["infrastructure_failure"] = True
            record["components"]["R_S1_instruction_exec"] = 0.0
            record["components"]["R_S2_judger_gain"] = 0.0
            record["components"]["R_S2_no_degrade"] = 0.0
            record["components"]["R_judger_gain"] = 0.0
            record["components"]["R_alignment_l2"] = 0.0
            return self.total_reward(record), record

        service_slot, adapter_url, judger_url, actor_pass2_url = select_service_urls(idx)
        record["service_slot"] = service_slot
        record["comfy_adapter_url"] = adapter_url
        record["judger_url"] = judger_url
        record["actor_pass2_url"] = actor_pass2_url
        before: dict[str, Any] | None = None
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                before_future = executor.submit(request_judger_score, image_path, judger_url=judger_url)
                edit_future = executor.submit(request_comfy_edit, image_path, regions, idx, adapter_url=adapter_url)
                edit_result = edit_future.result()
                try:
                    before = before_future.result()
                except Exception as exc:
                    record["judger_status"] = "error"
                    record["failure_labels"].append("judge_failure")
                    record["judger_error"] = repr(exc)
                    record["pass2_status"] = "not_run_phase_a"
                    record["masks"]["infrastructure_failure"] = True
                    record["components"]["R_S1_instruction_exec"] = 0.0
                    record["components"]["R_S2_judger_gain"] = 0.0
                    record["components"]["R_S2_no_degrade"] = 0.0
                    record["components"]["R_judger_gain"] = 0.0
                    record["components"]["R_alignment_l2"] = 0.0
                    return self.total_reward(record), record
        except Exception as exc:
            record["edit_status"] = "error"
            record["pass2_status"] = "not_run_phase_a"
            record["failure_labels"].append("generation_failure")
            record["edit_error"] = repr(exc)
            record["masks"]["infrastructure_failure"] = True
            record["components"]["R_S1_instruction_exec"] = 0.0
            record["components"]["R_S2_judger_gain"] = 0.0
            record["components"]["R_S2_no_degrade"] = 0.0
            record["components"]["R_judger_gain"] = 0.0
            record["components"]["R_alignment_l2"] = 0.0
            return self.total_reward(record), record
        record["judger_original_concurrent_with_edit"] = True

        record["edit_result"] = edit_result
        record["edit_status"] = str(edit_result.get("status") or "unknown")
        edited_path = edit_result.get("edited_path") or edit_result.get("path")
        if record["edit_status"] != "success" or not edited_path:
            record["failure_labels"].append("generation_failure" if record["edit_status"] != "skipped" else "tool_skipped")
            record["pass2_status"] = "not_run_phase_a"
            record["components"]["R_S1_instruction_exec"] = 0.0
            record["components"]["R_S2_judger_gain"] = 0.0
            record["components"]["R_S2_no_degrade"] = 0.0
            record["components"]["R_judger_gain"] = 0.0
            record["components"]["R_alignment_l2"] = 0.0
            return self.total_reward(record), record
        record["masks"]["edit_executed"] = True
        record["components"]["R_S1_instruction_exec"] = 1.0
        record["edited_path"] = str(edited_path)
        record["edited_image_path"] = str(edited_path)

        try:
            after = request_judger_score(str(edited_path), judger_url=judger_url)
        except Exception as exc:
            record["judger_status"] = "error"
            record["failure_labels"].append("judge_failure")
            record["judger_error"] = repr(exc)
            record["pass2_status"] = "not_run_phase_a"
            record["masks"]["infrastructure_failure"] = True
            record["components"]["R_S2_judger_gain"] = 0.0
            record["components"]["R_S2_no_degrade"] = 0.0
            record["components"]["R_judger_gain"] = 0.0
            record["components"]["R_alignment_l2"] = 0.0
            return self.total_reward(record), record

        j0 = score_payload_mean(before)
        j1 = score_payload_mean(after)
        record["judger_original"] = before
        record["judger_edited"] = after
        record["j0"] = j0
        record["j1"] = j1
        record["jd"] = None if j0 is None or j1 is None else j1 - j0
        record["judger_margin"] = record["jd"]
        record["judger_margin_abs"] = None if record["jd"] is None else abs(record["jd"])
        record["judger_status"] = "success" if record["jd"] is not None else "unparsed"
        record["masks"]["judger_original_success"] = j0 is not None
        record["masks"]["judger_edited_success"] = j1 is not None
        record["components"]["R_S2_judger_gain"] = self.judger_gain_reward(target, record["jd"])
        record["components"]["R_S2_no_degrade"] = self.no_degrade_reward(record["jd"])
        record["components"]["R_judger_gain"] = record["components"]["R_S2_judger_gain"]
        record["components"]["R_alignment_l2"] = 0.0
        record["pass2_status"] = "not_run_phase_a"
        record["segments"]["S2_external_outcome"] = {
            "j0": j0,
            "j1": j1,
            "jd": record["jd"],
            "judger_gain_reward": record["components"]["R_S2_judger_gain"],
            "no_degrade_reward": record["components"]["R_S2_no_degrade"],
        }
        if record["jd"] is None:
            record["failure_labels"].append("judge_failure")
            record["masks"]["infrastructure_failure"] = True
        else:
            append_pass2_buffer(
                {
                    "schema_version": "vf_pass2_buffer_v1",
                    "time": record["time"],
                    "source_reward_name": record["reward_name"],
                    "source_idx": idx,
                    "source_image_path": image_path,
                    "edited_image_path": str(edited_path),
                    "target_mean": target,
                    "a0": a0,
                    "j0": j0,
                    "j1": j1,
                    "jd": record["jd"],
                    "editing": actor_editing,
                    "edit_result": edit_result,
                    "service_slot": service_slot,
                    "comfy_adapter_url": adapter_url,
                    "judger_url": judger_url,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_hash": PROMPT_HASH,
                    "executor_prompt_version": EXECUTION_PROMPT_VERSION,
                }
            )
        return self.total_reward(record), record

    @staticmethod
    def edit_decision_reward(target: Optional[float], has_regions: bool, valid_regions: bool) -> float:
        p_edit = edit_probability_from_target(target)
        if p_edit is None:
            return 0.0
        edit_value = 1.0 if has_regions and valid_regions else 0.0
        return l2_reward(edit_value - p_edit, tau=float(os.environ.get("VF_LOOP_EDIT_TAU", "0.5")))

    @staticmethod
    def no_edit_is_reasonable(target: Optional[float]) -> bool:
        p_edit = edit_probability_from_target(target)
        if p_edit is None:
            return False
        return p_edit <= float(os.environ.get("VF_LOOP_NO_EDIT_PROB_THRESHOLD", "0.25"))

    @staticmethod
    def judger_gain_reward(target: Optional[float], jd: Optional[float]) -> float:
        if target is None or jd is None:
            return 0.0
        expected_gain = max(0.0, (5.0 - target) / 4.0)
        missed_gain = max(0.0, expected_gain - float(jd))
        return l2_reward(missed_gain, tau=float(os.environ.get("VF_LOOP_JUDGER_TAU", "1.0")))

    @staticmethod
    def no_degrade_reward(jd: Optional[float]) -> float:
        if jd is None:
            return 0.0
        degrade = max(0.0, -float(jd))
        return l2_reward(degrade, tau=float(os.environ.get("VF_LOOP_DEGRADE_TAU", "1.0")))

    @staticmethod
    def actor_judger_alignment_reward(delta_error: Optional[float]) -> float:
        if delta_error is None:
            return 0.0
        return l2_reward(float(delta_error), tau=float(os.environ.get("VF_LOOP_ALIGNMENT_TAU", "1.0")))

    @staticmethod
    def total_reward(record: dict[str, Any]) -> float:
        weights = {
            "R_S0_format": float(os.environ.get("VF_LOOP_WEIGHT_FORMAT", "1.0")),
            "R_S1_edit_gate": float(os.environ.get("VF_LOOP_WEIGHT_EDIT_DECISION", "1.0")),
            "R_S1_instruction_exec": float(os.environ.get("VF_LOOP_WEIGHT_INSTRUCTION_EXEC", "1.0")),
            "R_S2_judger_gain": float(os.environ.get("VF_LOOP_WEIGHT_JUDGER_GAIN", "1.0")),
            "R_S2_no_degrade": float(os.environ.get("VF_LOOP_WEIGHT_NO_DEGRADE", "1.0")),
        }
        return sum(weights[key] * float(record["components"].get(key, 0.0)) for key in weights)


class VisualFeedbackPass2Reward(ORM):
    """Phase B: reward current actor pass2 completions on edited-image samples."""

    call_count = 0

    def __call__(self, completions, target_mean=None, **kwargs) -> list[float]:
        self.__class__.call_count += 1
        rewards: list[float] = []
        for idx, text in enumerate(completions):
            reward, record = self.score_one(idx, str(text or ""), target_mean, kwargs)
            rewards.append(reward)
            append_trajectory(record)
        return rewards

    def score_one(self, idx: int, text: str, target_mean: Any, kwargs: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        full_text = completion_text(text)
        payload = parse_json_object(full_text)
        errors = actor_payload_field_errors(payload)
        context = pass2_context_at(kwargs, idx)
        target = score_number(target_at(target_mean, idx, None))
        image_path = image_path_at(kwargs, idx)
        record: dict[str, Any] = {
            "time": time.time(),
            "reward_name": "vf_loop_segment_phase_b",
            "reward_schema_version": "vf_segment_reward_v1",
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": PROMPT_HASH,
            "executor_prompt_version": EXECUTION_PROMPT_VERSION,
            "training_phase": "phase_b",
            "idx": idx,
            "image_path": image_path,
            "target_mean": target,
            "format_success": not errors,
            "format_errors": errors,
            "pass2_status": "not_started",
            "failure_labels": [],
            "components": {},
            "segments": {"S3_edited_understanding": {}},
            "credit_assignment": {},
            "masks": segment_masks(),
            "context": context,
        }
        logged_completion = maybe_logged_completion(full_text)
        if logged_completion is not None:
            record["completion_text"] = logged_completion
            record["completion_text_truncated"] = len(str(full_text or "")) > len(logged_completion)
        if errors or not isinstance(payload, dict):
            record["failure_labels"].append("format_parse_error")
            record["masks"]["actor_caused_failure"] = True
            record["pass2_status"] = "format_error"
            return 0.0, record

        internal_payload = to_internal_actor_payload(payload)
        editing = str(internal_payload.get("editing") or "").strip()
        record["actor_schema"] = actor_schema()
        record["reasons"] = str(payload.get("reasons") or payload.get("reason") or "")
        record["suggestion"] = str(payload.get("suggestion") or "")
        record["editing"] = editing

        a1 = rating_number(payload.get("rating"))
        a0 = context["a0"]
        j1 = context["j1"]
        jd = context["jd"]
        if a1 is None or a0 is None or j1 is None or jd is None:
            record["a1"] = a1
            record["failure_labels"].append("missing_pass2_context")
            record["pass2_status"] = "missing_context"
            return 0.0, record

        ad = float(a1) - float(a0)
        delta_error = ad - float(jd)
        abs_error = float(a1) - float(j1)
        record["a1"] = a1
        record["ad"] = ad
        record["actor_margin"] = ad
        record["actor_margin_abs"] = abs(ad)
        record["actor_judger_delta_error"] = delta_error
        record["actor_judger_delta_error_abs"] = abs(delta_error)
        record["actor_judger_abs_error"] = abs_error
        record["actor_judger_abs_error_abs"] = abs(abs_error)
        record["pass2_status"] = "success"
        record["masks"]["pass2_valid"] = True
        record["components"]["R_S3_format"] = 1.0
        record["components"]["R_S3_abs_judger"] = l2_reward(
            abs_error, tau=float(os.environ.get("VF_LOOP_PASS2_ABS_TAU", "1.0"))
        )
        record["components"]["R_S3_delta_align"] = self.actor_judger_alignment_reward(delta_error)
        record["components"]["R_alignment_l2"] = record["components"]["R_S3_delta_align"]
        record["segments"]["S3_edited_understanding"] = {
            "a0": a0,
            "a1": a1,
            "j1": j1,
            "jd": jd,
            "ad": ad,
            "abs_judger_reward": record["components"]["R_S3_abs_judger"],
            "delta_align_reward": record["components"]["R_S3_delta_align"],
            "editing": editing,
        }
        return self.total_reward(record), record

    @staticmethod
    def actor_judger_alignment_reward(delta_error: Optional[float]) -> float:
        if delta_error is None:
            return 0.0
        return l2_reward(float(delta_error), tau=float(os.environ.get("VF_LOOP_ALIGNMENT_TAU", "1.0")))

    @staticmethod
    def total_reward(record: dict[str, Any]) -> float:
        weights = {
            "R_S3_format": float(os.environ.get("VF_LOOP_WEIGHT_FORMAT", "1.0")),
            "R_S3_abs_judger": float(os.environ.get("VF_LOOP_WEIGHT_PASS2_ABS", "1.0")),
            "R_S3_delta_align": float(os.environ.get("VF_LOOP_WEIGHT_ALIGNMENT", "1.0")),
        }
        return sum(weights[key] * float(record["components"].get(key, 0.0)) for key in weights)


VisualFeedbackLoopReward = VisualFeedbackSegmentPhaseAReward

def registration_enabled() -> bool:
    if os.environ.get("VF_CANONICAL_REGISTER_ORMS", "0") != "1":
        return False
    try:
        manifest = load_manifest(PLUGIN_DIR.parent / "migration_manifest.json")
        require_training_ready(manifest)
    except Exception:
        return False
    return True


if registration_enabled():
    orms["vf_loop_reward"] = VisualFeedbackSegmentPhaseAReward
    orms["vf_loop_segment_phase_a"] = VisualFeedbackSegmentPhaseAReward
    orms["vf_loop_pass2_reward"] = VisualFeedbackPass2Reward
    orms["vf_loop_segment_phase_b"] = VisualFeedbackPass2Reward
