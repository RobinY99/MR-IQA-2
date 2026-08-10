#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE_NAME="completion_global_kl002"
ACTION="run"
EPOCH_LIMIT=""
RUN_VALIDATION="${RUN_VALIDATION_EACH_EPOCH:-1}"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"

usage() {
  cat <<'EOF'
Usage: scripts/train.sh [options]

Options:
  --mode NAME          Training preset in configs/training (default: completion_global_kl002)
  --smoke              Run one optimizer-step smoke with the selected reward/KL contract
  --epochs N           Run only the first N formal epochs (1-5)
  --skip-validation    Leave a single formal epoch technically valid but unpromoted
  --print-plan         Print the resolved public training contract; no models are loaded
  --validate-config    Validate paths, hashes, mode invariants, and service topology
  -h, --help           Show this help
EOF
}

while (($#)); do
  case "$1" in
    --mode) MODE_NAME="${2:?--mode requires a value}"; shift 2 ;;
    --smoke) ACTION=smoke; shift ;;
    --epochs) EPOCH_LIMIT="${2:?--epochs requires a value}"; shift 2 ;;
    --skip-validation) RUN_VALIDATION=0; shift ;;
    --print-plan) ACTION=print-plan; shift ;;
    --validate-config) ACTION=validate-config; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${MODE_NAME}" =~ ^[a-z0-9_]+$ ]] || {
  echo "invalid mode name: ${MODE_NAME}" >&2
  exit 2
}
PROFILE="${ROOT}/configs/training/${MODE_NAME}.env"
[[ -f "${PROFILE}" ]] || {
  echo "unknown mode ${MODE_NAME}; see configs/training/mode_matrix.json" >&2
  exit 2
}

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
set -a
# shellcheck disable=SC1090
source "${PROFILE}"
set +a

python3 - "${ROOT}/configs/training/mode_matrix.json" "${MODE_NAME}" <<'PY'
import json
import pathlib
import sys

matrix = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if sys.argv[2] not in matrix["modes"]:
    raise SystemExit(f"mode is absent from mode_matrix.json: {sys.argv[2]}")
PY

print_plan() {
  python3 - \
    "${MODE_NAME}" "${MR_IQA_MODE_LABEL}" "${MR_IQA_TRAIN_KIND}" \
    "${COMPONENT_CREDIT_MASK_MODE}" "${COMPONENT_KL_MODE}" \
    "${BETA_KL_REASONING}" "${BETA_KL_RATING}" \
    "${REFERENCE_ACTIVATION_BETA}" "${KL_IN_REWARD}" \
    "${FREEZE_VIT}" "${FREEZE_ALIGNER}" \
    "${TOTAL_TRAIN_EPOCHS}" "${OPTIMIZER_STEPS_PER_EPOCH:-0}" \
    "${TRAIN_MAX_STEPS:-}" <<'PY'
import json
import sys

(
    mode, label, kind, credit, component_kl, reasoning_beta, rating_beta,
    activation_beta, kl_in_reward, freeze_vit, freeze_aligner, epochs,
    steps_per_epoch, max_steps,
) = sys.argv[1:]
activation_beta = float(activation_beta)
payload = {
    "mode": mode,
    "label": label,
    "kind": kind,
    "actor_gpus": [0, 1, 2, 3],
    "editor_judge_gpus": [4, 5, 6, 7],
    "credit_mask": credit,
    "component_kl": {
        "mode": component_kl,
        "reasoning_beta": float(reasoning_beta),
        "rating_beta": float(rating_beta),
    },
    "global_completion_kl": {
        "enabled": component_kl == "off" and activation_beta > 0,
        "beta": activation_beta if component_kl == "off" else 0.0,
        "kl_in_reward": kl_in_reward == "true",
        "expected_apply_count_per_step": (
            1 if component_kl == "off" and activation_beta > 0 else 0
        ),
    },
    "vision": {"vit_frozen": freeze_vit == "true", "aligner_frozen": freeze_aligner == "true"},
    "trajectory_rows_per_optimizer_step": 144,
    "epochs": int(epochs),
    "optimizer_steps_per_epoch": int(steps_per_epoch),
    "max_steps": int(max_steps) if max_steps else None,
    "validation_between_epochs": kind == "formal",
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

if [[ "${ACTION}" == "print-plan" ]]; then
  print_plan
  exit 0
fi

required_env=(
  CONDA_SH CONDA_ENV_NAME ACTOR_MODEL_PATH ACTOR_MODEL_TREE_SHA256
  TRAIN_IMAGE_ROOT DIFFUSERS_VENV DIFFUSERS_MODEL_PATH JUDGER_PYTHON
  JUDGE_MODEL_PATH JUDGE_MANIFEST_PATH JUDGE_MODEL_TREE_SHA256
  JUDGE_MODEL_EXPORT_TREE_SHA256
  JUDGE_PROMPT_HASH ORIGINAL_SCORE_CACHE_PATH ORIGINAL_SCORE_CACHE_SHA256
  ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT
  ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA
  FLASH_ATTN_WHEEL FLASH_ATTN_WHEEL_SHA256
)
for name in "${required_env[@]}"; do
  [[ -n "${!name:-}" ]] || {
    echo "missing required environment variable: ${name}" >&2
    exit 2
  }
done

export VF_PROJECT_ROOT="${ROOT}"
export VF_STORAGE_ROOT="${VF_STORAGE_ROOT:-${OUTPUT_ROOT:-${ROOT}/outputs}}"
export IMAGE_EDIT_BACKEND=diffusers
export SERVICE_GPUS=4,5,6,7
export SERVICE_EDITOR_PORTS=8212,8213,8214,8215
export SERVICE_JUDGER_PORTS=8204,8205,8206,8207
export JUDGER_MODEL_ID="${JUDGE_MODEL_ID:-source-e5-judge-step725}"
export JUDGER_MODEL_PATH="${JUDGE_MODEL_PATH}"
export JUDGER_MANIFEST_PATH="${JUDGE_MANIFEST_PATH}"
export JUDGER_MODEL_TREE_SHA256="${JUDGE_MODEL_TREE_SHA256}"
export JUDGER_MODEL_EXPORT_TREE_SHA256="${JUDGE_MODEL_EXPORT_TREE_SHA256}"
export JUDGER_PROMPT_SCHEMA="${JUDGER_PROMPT_SCHEMA:-e5_training_reasoning_v5}"
export JUDGER_PROMPT_HASH="${JUDGE_PROMPT_HASH}"
export JUDGER_BACKEND="${JUDGER_BACKEND:-e5_qwen35_4b_vllm_judge}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-mr-iqa-2}"

CODE_ROOT="${ROOT}/actor"
STAGE_RUNNER="${CODE_ROOT}/scripts/run_editor_judge_grpo_stage.sh"
SERVICE_STACK="${CODE_ROOT}/scripts/resident_service_stack.sh"
CHECKPOINT_TOOL="${CODE_ROOT}/scripts/checkpoint_manifest.py"
TRAIN_DATASET="${TRAIN_DATASET:-${ROOT}/data/train.jsonl}"

validate_config() {
  [[ -x "${STAGE_RUNNER}" && -x "${SERVICE_STACK}" ]] || {
    echo "training scripts are not executable" >&2
    return 1
  }
  [[ -d "${ACTOR_MODEL_PATH}" ]] || { echo "missing Actor model" >&2; return 1; }
  [[ -d "${TRAIN_IMAGE_ROOT}" ]] || { echo "missing training image root" >&2; return 1; }
  [[ -f "${TRAIN_DATASET}" ]] || { echo "missing training manifest" >&2; return 1; }
  [[ -f "${ORIGINAL_SCORE_CACHE_PATH}" ]] || { echo "missing score cache" >&2; return 1; }
  [[ -f "${FLASH_ATTN_WHEEL}" ]] || { echo "missing FlashAttention wheel" >&2; return 1; }
  [[ "$(wc -l <"${TRAIN_DATASET}" | tr -d ' ')" == 7000 ]] || {
    echo "training manifest must contain exactly 7000 rows" >&2
    return 1
  }
  python3 - "${PROFILE}" "${MODE_NAME}" <<'PY'
import os
import sys

mode = sys.argv[2]
credit = os.environ["COMPONENT_CREDIT_MASK_MODE"]
component = os.environ["COMPONENT_KL_MODE"]
reasoning = float(os.environ["BETA_KL_REASONING"])
rating = float(os.environ["BETA_KL_RATING"])
activation = float(os.environ["REFERENCE_ACTIVATION_BETA"])
if os.environ["FREEZE_VIT"] != "true" or os.environ["FREEZE_ALIGNER"] != "true":
    raise SystemExit("published MR-IQA-2 modes require frozen ViT and aligner")
if component == "field" and not (credit == "field" and reasoning > 0 and rating > 0 and activation > 0):
    raise SystemExit("invalid field component-KL contract")
if component == "off" and (reasoning != 0 or rating != 0):
    raise SystemExit("component KL off requires zero component betas")
if "global_kl" in mode and not (credit == "completion" and component == "off" and activation > 0):
    raise SystemExit("invalid global completion-KL contract")
PY
  [[ "$(sha256sum "${ORIGINAL_SCORE_CACHE_PATH}" | awk '{print $1}')" == "${ORIGINAL_SCORE_CACHE_SHA256}" ]] || {
    echo "score cache SHA256 mismatch" >&2
    return 1
  }
  [[ "$(sha256sum "${FLASH_ATTN_WHEEL}" | awk '{print $1}')" == "${FLASH_ATTN_WHEEL_SHA256}" ]] || {
    echo "FlashAttention wheel SHA256 mismatch" >&2
    return 1
  }
  bash "${SERVICE_STACK}" --validate-config
}

if [[ "${ACTION}" == "validate-config" ]]; then
  validate_config
  echo "configuration is valid"
  exit 0
fi

validate_config
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import secrets; print(secrets.token_hex(4))')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/training/${MODE_NAME}/${RUN_ID}}"
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "refusing to reuse an existing OUTPUT_ROOT: ${OUTPUT_ROOT}" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}/runs" "${OUTPUT_ROOT}/state" "${OUTPUT_ROOT}/logs"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
mkdir -p "${WANDB_DIR}"

ACTIVE_SERVICE_RUN=""
LAST_WANDB_ID=""
stop_services() {
  if [[ -n "${ACTIVE_SERVICE_RUN}" ]]; then
    bash "${SERVICE_STACK}" stop "${ACTIVE_SERVICE_RUN}" || true
    ACTIVE_SERVICE_RUN=""
  fi
}
trap stop_services EXIT INT TERM

new_run_id() {
  python3 -c 'import secrets; print(secrets.token_hex(8))'
}

run_stage() {
  local run_dir="$1" target_epoch="$2" step_start="$3" step_end="$4" train_max="$5" resume="$6" parent_manifest="$7" stage_mode="$8"
  local wandb_id
  wandb_id="$(new_run_id)"
  LAST_WANDB_ID="${wandb_id}"
  [[ ! -e "${run_dir}" ]] || {
    echo "refusing to reuse an existing stage run directory: ${run_dir}" >&2
    return 2
  }
  mkdir -p "${run_dir}"
  ACTIVE_SERVICE_RUN="${run_dir}"
  bash "${SERVICE_STACK}" start "${run_dir}"
  bash "${SERVICE_STACK}" status "${run_dir}"
  env \
    MODE="${stage_mode}" RUN_DIR="${run_dir}" CODE_ROOT="${CODE_ROOT}" PACKAGE_ROOT="${ROOT}" \
    SERVICE_RUN_DIR="${run_dir}" EXPERIMENT_ID="${MODE_NAME}_epoch${target_epoch}" \
    MODEL_TAG=qwen35_4b MODEL_FAMILY=qwen35 MODEL_PATH="${ACTOR_MODEL_PATH}" \
    ALGORITHM=grpo NUM_ITERATIONS=1 PER_DEVICE_BATCH_SIZE=36 LEARNER_MICROBATCH_SIZE=36 \
    ALLOW_LEARNER_MICROBATCH_SPLIT=0 MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH}" \
    SOFT_OVERLONG_MAX_LENGTH="${SOFT_OVERLONG_MAX_LENGTH}" \
    SOFT_OVERLONG_CACHE_LENGTH="${SOFT_OVERLONG_CACHE_LENGTH}" \
    SOFT_OVERLONG_MAX_PENALTY="${SOFT_OVERLONG_MAX_PENALTY}" \
    SOFT_OVERLONG_WEIGHT="${SOFT_OVERLONG_WEIGHT}" \
    FREEZE_VIT="${FREEZE_VIT}" FREEZE_ALIGNER="${FREEZE_ALIGNER}" \
    GRADIENT_CHECKPOINTING=true VIT_GRADIENT_CHECKPOINTING=true \
    GRADIENT_CHECKPOINTING_USE_REENTRANT=false VF_ALLOW_LLM_GC_FALLBACK=1 \
    VF_LEARNER_ACTIVATION_OFFLOAD=1 VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB=12 \
    BATCH_DECAY_STEP=0 MARGIN_REWARD_SCOPE=local_six_images MARGIN_IMAGES_PER_COHORT=6 \
    MARGIN_LOCAL_IMAGES_PER_RANK=6 REWARD_GATHER_ORDER=local_reward_then_global_gather \
    TOTAL_TRAIN_EPOCHS="${TOTAL_TRAIN_EPOCHS}" EXPECTED_TOTAL_TRAIN_EPOCHS="${EXPECTED_TOTAL_TRAIN_EPOCHS}" \
    TARGET_EPOCH="${target_epoch}" CONDA_SH="${CONDA_SH}" CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    TRAIN_IMAGE_ROOT="${TRAIN_IMAGE_ROOT}" VLLM_SLEEP_LEVEL=1 \
    ORIGINAL_SCORE_CACHE_PATH="${ORIGINAL_SCORE_CACHE_PATH}" \
    ORIGINAL_SCORE_CACHE_SHA256="${ORIGINAL_SCORE_CACHE_SHA256}" \
    ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT="${ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT}" \
    ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT="${ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT}" \
    ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS="${ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS}" \
    ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA="${ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA}" \
    ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN="${ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN:-0.0}" \
    ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX="${ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX:-5.0}" \
    JUDGE_MODEL_ID="${JUDGER_MODEL_ID}" JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH}" \
    JUDGE_MODEL_TREE_SHA256="${JUDGE_MODEL_TREE_SHA256}" \
    JUDGE_MODEL_EXPORT_TREE_SHA256="${JUDGE_MODEL_EXPORT_TREE_SHA256}" \
    JUDGE_PROMPT_HASH="${JUDGE_PROMPT_HASH}" \
    COMPONENT_CREDIT_MASK_MODE="${COMPONENT_CREDIT_MASK_MODE}" \
    EXPECTED_COMPONENT_CREDIT_MASK_MODE="${EXPECTED_COMPONENT_CREDIT_MASK_MODE}" \
    COMPONENT_KL_MODE="${COMPONENT_KL_MODE}" EXPECTED_COMPONENT_KL_MODE="${EXPECTED_COMPONENT_KL_MODE}" \
    BETA_KL_REASONING="${BETA_KL_REASONING}" BETA_KL_RATING="${BETA_KL_RATING}" \
    EXPECTED_BETA_KL_REASONING="${EXPECTED_BETA_KL_REASONING}" \
    EXPECTED_BETA_KL_RATING="${EXPECTED_BETA_KL_RATING}" \
    REFERENCE_ACTIVATION_BETA="${REFERENCE_ACTIVATION_BETA}" \
    REFERENCE_MODEL_PATH="${ACTOR_MODEL_PATH}" REFERENCE_MODEL_TREE_SHA256="${ACTOR_MODEL_TREE_SHA256}" \
    JUDGER_MAX_NUM_SEQS=1 JUDGER_MAX_BATCH_SIZE=1 JUDGER_BATCH_WAIT_MS=0 \
    EDITOR_JUDGE_SERVICE_WORKERS=12 RETAINED_DATASET_SOURCE="${TRAIN_DATASET}" \
    TRAIN_MAX_STEPS="${train_max}" STOP_AFTER_STEP="${step_end}" STEP_START="${step_start}" \
    SMOKE_MAX_STEPS="${train_max}" \
    SAVE_STEPS="$((step_end - step_start))" SAVE_TOTAL_LIMIT=1 RESUME_CHECKPOINT="${resume}" \
    PARENT_MANIFEST="${parent_manifest}" \
    WANDB_RUN_ID="${wandb_id}" WANDB_MODE="${WANDB_MODE}" \
    FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL}" FLASH_ATTN_WHEEL_SHA256="${FLASH_ATTN_WHEEL_SHA256}" \
    "${STAGE_RUNNER}"
  stop_services
}

if [[ "${ACTION}" == "smoke" ]]; then
  smoke_dir="${OUTPUT_ROOT}/runs/smoke"
  run_stage "${smoke_dir}" 0 0 1 1 "" "" smoke
  echo "smoke completed: ${smoke_dir}"
  exit 0
fi

if [[ "${MR_IQA_TRAIN_KIND}" == "steps" ]]; then
  run_dir="${OUTPUT_ROOT}/runs/step30"
  run_stage "${run_dir}" 0 "${STEP_START}" "${STOP_AFTER_STEP}" "${TRAIN_MAX_STEPS}" "" "" steps
  echo "training completed: ${run_dir}"
  exit 0
fi

[[ "${MR_IQA_TRAIN_KIND}" == "formal" ]] || {
  echo "unsupported MR_IQA_TRAIN_KIND=${MR_IQA_TRAIN_KIND}" >&2
  exit 2
}
epochs="${EPOCH_LIMIT:-${TOTAL_TRAIN_EPOCHS}}"
[[ "${epochs}" =~ ^[1-5]$ ]] || { echo "--epochs must be in 1..5" >&2; exit 2; }
if [[ "${RUN_VALIDATION}" != "1" && "${epochs}" != "1" ]]; then
  echo "--skip-validation may only be used with --epochs 1: an unpromoted checkpoint cannot seed another epoch" >&2
  exit 2
fi
total_steps="$((TOTAL_TRAIN_EPOCHS * OPTIMIZER_STEPS_PER_EPOCH))"
resume=""
parent_manifest=""
data_sha256="$(sha256sum "${TRAIN_DATASET}" | awk '{print $1}')"
prompt_hash="$(VF_ACTOR_SCHEMA=reasoning_evidence_solution_rating python3 - "${CODE_ROOT}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/plugin")
from prompt_contract import prompt_metadata
print(prompt_metadata()["prompt_hash"])
PY
)"
code_sha256="$(python3 - "${ROOT}" "${PROFILE}" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
profile = pathlib.Path(sys.argv[2]).resolve()
paths = [root / "scripts/train.sh", profile]
for subtree in (root / "actor/plugin", root / "actor/scripts"):
    paths.extend(
        path for path in subtree.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh", ".json"}
    )
digest = hashlib.sha256()
for path in sorted(set(paths)):
    relative = path.relative_to(root).as_posix()
    payload = path.read_bytes()
    digest.update(relative.encode())
    digest.update(b"\0")
    digest.update(str(len(payload)).encode())
    digest.update(b"\0")
    digest.update(hashlib.sha256(payload).hexdigest().encode())
    digest.update(b"\n")
print(digest.hexdigest())
PY
)"
for ((epoch=1; epoch<=epochs; epoch++)); do
  start="$(((epoch - 1) * OPTIMIZER_STEPS_PER_EPOCH))"
  end="$((epoch * OPTIMIZER_STEPS_PER_EPOCH))"
  run_dir="${OUTPUT_ROOT}/runs/epoch${epoch}"
  run_stage "${run_dir}" "${epoch}" "${start}" "${end}" "${total_steps}" "${resume}" "${parent_manifest}" formal
  checkpoint="$(python3 "${CHECKPOINT_TOOL}" discover --run-dir "${run_dir}")"
  [[ "$(basename "${checkpoint}")" == "checkpoint-${end}" ]] || {
    echo "missing checkpoint-${end} after epoch ${epoch}" >&2
    exit 1
  }
  manifest="${OUTPUT_ROOT}/state/checkpoints/epoch${epoch}.json"
  parent_kind=approved_initial
  if [[ -n "${parent_manifest}" ]]; then
    parent_kind=promoted_checkpoint
  fi
  create_args=(
    create --run-dir "${run_dir}" --checkpoint "${checkpoint}" --output "${manifest}"
    --run-id "${MODE_NAME}_epoch${epoch}" --parent-checkpoint "${resume:-${ACTOR_MODEL_PATH}}"
    --parent-kind "${parent_kind}"
    --data-sha256 "${data_sha256}" --prompt-hash "${prompt_hash}" --code-sha256 "${code_sha256}"
  )
  if [[ -n "${parent_manifest}" ]]; then
    create_args+=(--parent-manifest "${parent_manifest}")
  fi
  python3 "${CHECKPOINT_TOOL}" "${create_args[@]}"
  python3 "${CHECKPOINT_TOOL}" technical-validate \
    --manifest "${manifest}" \
    --trainer-exit-code 0 \
    --trajectory-summary "${run_dir}/artifacts/trajectory_summary.json" \
    --wandb-url "wandb://${WANDB_PROJECT}/${LAST_WANDB_ID}" \
    --expected-rank-shards 4
  checkpoint_digest="$(python3 - "${manifest}" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["checkpoint_identity"]["selected_export_tree_sha256"])
PY
)"
  validation_root=""
  validation_summary=""
  if [[ "${RUN_VALIDATION}" == "1" ]]; then
    validation_root="${OUTPUT_ROOT}/evaluation/${MODE_NAME}_epoch${epoch}/validation"
    env \
      ENV_FILE="${ENV_FILE}" \
      ACTOR_PROCESSOR_PATH="${ACTOR_MODEL_PATH}" \
      ACTOR_MODEL_PATH="${checkpoint}" \
      ACTOR_MODEL_EXPORT_TREE_SHA256="${checkpoint_digest}" \
      EVAL_NAME="${MODE_NAME}_epoch${epoch}" \
      EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}/evaluation" \
      EVAL_RESUME=1 \
      "${ROOT}/scripts/evaluate.sh" validation
    validation_summary="${OUTPUT_ROOT}/state/checkpoints/epoch${epoch}.validation.json"
    python3 "${CHECKPOINT_TOOL}" validation-summary \
      --eval-root "${validation_root}" --output "${validation_summary}"
    python3 "${CHECKPOINT_TOOL}" promote \
      --manifest "${manifest}" --validation "${validation_summary}" \
      --approval "public_pipeline:${MODE_NAME}:epoch${epoch}:validation200" \
      --expected-validation-shards 8 \
      --expected-actor-schema reasoning_evidence_solution_rating \
      --validation-policy comparison_observational
    # This is the only assignment that can seed the next epoch.  ``resolve``
    # re-hashes the ten-file inference identity and rejects non-promoted state.
    resume="$(python3 "${CHECKPOINT_TOOL}" resolve --manifest "${manifest}")"
    parent_manifest="${manifest}"
  fi
  python3 - \
    "${OUTPUT_ROOT}/state/epoch${epoch}.json" "${MODE_NAME}" "${epoch}" \
    "${start}" "${end}" "${checkpoint}" "${manifest}" \
    "${validation_root}" "${validation_summary}" <<'PY'
import json
import pathlib
import sys

output, mode, epoch, start, end, checkpoint, manifest_path, validation_root, validation_summary = sys.argv[1:]
manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
identity = manifest["checkpoint_identity"]
payload = {
    "schema_version": "mr_iqa_2_epoch_chain_v2",
    "mode": mode,
    "epoch": int(epoch),
    "start_step": int(start),
    "end_step": int(end),
    "checkpoint": checkpoint,
    "checkpoint_manifest": manifest_path,
    "checkpoint_status": manifest["status"],
    "checkpoint_digest": {
        "semantics": identity["digest_semantics"],
        "algorithm": identity["digest_algorithm"],
        "sha256": identity["selected_export_tree_sha256"],
        "file_count": identity["file_count"],
    },
    "validation_root": validation_root or None,
    "validation_summary": validation_summary or None,
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
done

echo "training completed: ${OUTPUT_ROOT}"
if [[ -n "${resume}" ]]; then
  echo "last promoted checkpoint: ${resume}"
else
  echo "last checkpoint remains technically valid but unpromoted: ${checkpoint}"
fi
