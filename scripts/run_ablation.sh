#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-run}"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"

if [[ "${ACTION}" == "--print-plan" ]]; then
  printf '%s\n' \
    "GPU 0-3: native Qwen3.5-4B Actor learner" \
    "GPU 4-7: unchanged FLUX Editor and E5 Judge services" \
    "phase1_nomask: native start, completion-wide credit, KL off, steps 1-30" \
    "phase2_mask: native start, field credit mask, KL off, steps 1-30"
  exit 0
fi

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

CODE_ROOT="${ROOT}/actor"
RUNNER="${CODE_ROOT}/scripts/run_editor_judge_grpo_stage.sh"
SERVICE_STACK="${CODE_ROOT}/scripts/resident_service_stack.sh"
TRAIN_DATASET="${TRAIN_DATASET:-${ROOT}/data/train.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/training/${RUN_ID}}"
STATE_DIR="${OUTPUT_ROOT}/state"
LOG_DIR="${OUTPUT_ROOT}/logs"
RUNS_DIR="${OUTPUT_ROOT}/runs"
STATE_FILE="${STATE_DIR}/run_state.json"
SUMMARY_FILE="${STATE_DIR}/summary.json"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-qwen35_4b_native_independent_nomask30_mask30_nokl}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-100}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
CURRENT_PHASE="preparation"
CURRENT_RUN_DIR=""
ACTIVE_SERVICE_RUN_DIR=""

required_env=(
  CONDA_SH
  CONDA_ENV_NAME
  ACTOR_MODEL_PATH
  TRAIN_IMAGE_ROOT
  DIFFUSERS_VENV
  DIFFUSERS_MODEL_PATH
  JUDGER_PYTHON
  JUDGE_MODEL_PATH
  JUDGE_MANIFEST_PATH
  JUDGE_MODEL_TREE_SHA256
  JUDGE_MODEL_EXPORT_TREE_SHA256
  JUDGE_PROMPT_HASH
  ORIGINAL_SCORE_CACHE_PATH
  ORIGINAL_SCORE_CACHE_SHA256
  ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT
  ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT
  ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS
  ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA
  FLASH_ATTN_WHEEL
  FLASH_ATTN_WHEEL_SHA256
)
for name in "${required_env[@]}"; do
  [[ -n "${!name:-}" ]] || {
    echo "missing required environment variable: ${name}" >&2
    exit 2
  }
done

export VF_PROJECT_ROOT="${ROOT}"
export IMAGE_EDIT_BACKEND=diffusers
export SERVICE_GPUS=4,5,6,7
export SERVICE_EDITOR_PORTS=8212,8213,8214,8215
export SERVICE_JUDGER_PORTS=8204,8205,8206,8207
export JUDGER_MODEL_ID="${JUDGE_MODEL_ID:-source-e5-judge-step725}"
export JUDGER_MODEL_PATH="${JUDGE_MODEL_PATH}"
export JUDGER_MANIFEST_PATH="${JUDGE_MANIFEST_PATH}"
export JUDGER_MODEL_TREE_SHA256="${JUDGE_MODEL_TREE_SHA256}"
export JUDGER_MODEL_EXPORT_TREE_SHA256="${JUDGE_MODEL_EXPORT_TREE_SHA256}"
export JUDGER_BACKEND="${JUDGER_BACKEND:-e5_qwen35_4b_vllm_judge}"
export JUDGER_PROMPT_SCHEMA="${JUDGER_PROMPT_SCHEMA:-e5_training_reasoning_v5}"
export JUDGER_PROMPT_HASH="${JUDGE_PROMPT_HASH}"
export VF_JUDGE_MODEL_ID="${JUDGER_MODEL_ID}"
export VF_JUDGE_MODEL_PATH="${JUDGER_MODEL_PATH}"
export VF_JUDGE_MODEL_TREE_SHA256="${JUDGER_MODEL_TREE_SHA256}"
export VF_JUDGE_PROMPT_SCHEMA="${JUDGER_PROMPT_SCHEMA}"
export VF_JUDGE_PROMPT_HASH="${JUDGER_PROMPT_HASH}"
export VF_JUDGER_MAX_NUM_SEQS=1
export VF_JUDGER_MAX_BATCH_SIZE=1
export VF_JUDGER_BATCH_WAIT_MS=0
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-mr-iqa}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
export VF_STORAGE_ROOT="${VF_STORAGE_ROOT:-${OUTPUT_ROOT}}"

validate_config() {
  [[ -x "${RUNNER}" && -x "${SERVICE_STACK}" ]] || {
    echo "training runner or service stack is not executable" >&2
    exit 2
  }
  [[ -d "${ACTOR_MODEL_PATH}" ]] || { echo "missing Actor model directory" >&2; exit 2; }
  [[ -d "${DIFFUSERS_MODEL_PATH}" ]] || { echo "missing Editor model directory" >&2; exit 2; }
  [[ -d "${JUDGE_MODEL_PATH}" ]] || { echo "missing Judge model directory" >&2; exit 2; }
  [[ -d "${TRAIN_IMAGE_ROOT}" ]] || { echo "missing training image root" >&2; exit 2; }
  [[ -f "${TRAIN_DATASET}" ]] || { echo "missing training dataset" >&2; exit 2; }
  [[ -f "${JUDGE_MANIFEST_PATH}" ]] || { echo "missing Judge checkpoint manifest" >&2; exit 2; }
  [[ -f "${ORIGINAL_SCORE_CACHE_PATH}" ]] || { echo "missing original-score cache" >&2; exit 2; }
  [[ -f "${FLASH_ATTN_WHEEL}" ]] || { echo "missing FlashAttention wheel" >&2; exit 2; }
  [[ "$(wc -l <"${TRAIN_DATASET}" | tr -d ' ')" == "7000" ]] || {
    echo "training dataset must contain exactly 7000 JSONL rows" >&2
    exit 2
  }
  [[ "$(sha256sum "${ORIGINAL_SCORE_CACHE_PATH}" | awk '{print $1}')" == "${ORIGINAL_SCORE_CACHE_SHA256}" ]] || {
    echo "original-score cache hash mismatch" >&2
    exit 2
  }
  [[ "$(sha256sum "${FLASH_ATTN_WHEEL}" | awk '{print $1}')" == "${FLASH_ATTN_WHEEL_SHA256}" ]] || {
    echo "FlashAttention wheel hash mismatch" >&2
    exit 2
  }
  bash "${SERVICE_STACK}" --validate-config
}

if [[ "${ACTION}" == "--validate-config" ]]; then
  validate_config
  echo "configuration is valid"
  exit 0
fi
[[ "${ACTION}" == "run" ]] || { echo "usage: $0 [run|--print-plan|--validate-config]" >&2; exit 2; }

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${RUNS_DIR}" "${WANDB_DIR}"
exec > >(tee -a "${LOG_DIR}/driver.log") 2>&1

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

write_state() {
  local status="$1" phase="$2" detail="$3"
  python3 - "${STATE_FILE}" "${status}" "${phase}" "${CURRENT_RUN_DIR}" "${detail}" "$$" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "status": sys.argv[2],
    "phase": sys.argv[3],
    "current_run_dir": sys.argv[4] or None,
    "detail": sys.argv[5],
    "driver_pid": int(sys.argv[6]),
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

wait_for_all_gpus() {
  local snapshot
  while true; do
    snapshot="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
    if python3 - "${snapshot}" "${GPU_IDLE_MEMORY_MIB}" <<'PY'
import subprocess
import sys

rows = []
for line in sys.argv[1].splitlines():
    index, used, utilization = (int(value.strip()) for value in line.split(","))
    rows.append((index, used, utilization))
if [row[0] for row in rows] != list(range(8)):
    raise SystemExit(1)
if any(used > int(sys.argv[2]) or utilization != 0 for _, used, utilization in rows):
    raise SystemExit(1)
processes = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    text=True,
).strip()
raise SystemExit(1 if processes else 0)
PY
    then
      echo "[$(timestamp)] all eight GPUs are idle"
      return
    fi
    echo "[$(timestamp)] waiting for all eight GPUs to become idle"
    printf '%s\n' "${snapshot}"
    sleep "${GPU_POLL_SECONDS}"
  done
}

stop_services() {
  if [[ -n "${ACTIVE_SERVICE_RUN_DIR}" ]]; then
    bash "${SERVICE_STACK}" stop "${ACTIVE_SERVICE_RUN_DIR}" || true
    ACTIVE_SERVICE_RUN_DIR=""
  fi
}

handle_exit() {
  local status=$?
  trap - EXIT ERR INT TERM
  stop_services
  if [[ "${status}" -ne 0 ]]; then
    write_state failed "${CURRENT_PHASE}" "driver exited with status ${status}"
  fi
  exit "${status}"
}
trap handle_exit EXIT ERR INT TERM

run_phase() {
  local phase="$1" mask_mode="$2"
  local run_dir="${RUNS_DIR}/${phase}"
  local wandb_id_file="${STATE_DIR}/wandb_${phase}_run_id"
  local wandb_run_id
  [[ ! -e "${run_dir}" ]] || { echo "run directory already exists: ${run_dir}" >&2; exit 2; }
  python3 -c 'import secrets; print(secrets.token_hex(8))' >"${wandb_id_file}"
  wandb_run_id="$(tr -d '[:space:]' <"${wandb_id_file}")"
  CURRENT_PHASE="${phase}"
  CURRENT_RUN_DIR="${run_dir}"
  mkdir -p "${run_dir}"
  write_state running "${phase}" "waiting for all eight GPUs"
  wait_for_all_gpus
  write_state running "${phase}" "starting Editor and Judge services"
  ACTIVE_SERVICE_RUN_DIR="${run_dir}"
  bash "${SERVICE_STACK}" start "${run_dir}"
  bash "${SERVICE_STACK}" status "${run_dir}"

  # Both phases start independently from the same native Actor; neither resumes the other.
  write_state running "${phase}" "native start, steps 1-30, mask=${mask_mode}, KL=off"
  env \
    MODE=steps \
    RUN_DIR="${run_dir}" \
    CODE_ROOT="${CODE_ROOT}" \
    PACKAGE_ROOT="${ROOT}" \
    SERVICE_RUN_DIR="${run_dir}" \
    EXPERIMENT_ID="${EXPERIMENT_GROUP}_${phase}" \
    MODEL_TAG=qwen35_4b \
    MODEL_FAMILY=qwen35 \
    MODEL_PATH="${ACTOR_MODEL_PATH}" \
    ALGORITHM=grpo \
    NUM_ITERATIONS=1 \
    PER_DEVICE_BATCH_SIZE=36 \
    LEARNER_MICROBATCH_SIZE=36 \
    ALLOW_LEARNER_MICROBATCH_SPLIT=0 \
    MAX_COMPLETION_LENGTH=160 \
    FREEZE_VIT=true \
    FREEZE_ALIGNER=true \
    GRADIENT_CHECKPOINTING=true \
    VIT_GRADIENT_CHECKPOINTING=true \
    GRADIENT_CHECKPOINTING_USE_REENTRANT=false \
    VF_ALLOW_LLM_GC_FALLBACK=1 \
    BATCH_DECAY_STEP=0 \
    MARGIN_REWARD_SCOPE=local_six_images \
    MARGIN_IMAGES_PER_COHORT=6 \
    MARGIN_LOCAL_IMAGES_PER_RANK=6 \
    REWARD_GATHER_ORDER=local_reward_then_global_gather \
    TOTAL_TRAIN_EPOCHS=5 \
    EXPECTED_TOTAL_TRAIN_EPOCHS=5 \
    TARGET_EPOCH=0 \
    CONDA_SH="${CONDA_SH}" \
    CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    TRAIN_IMAGE_ROOT="${TRAIN_IMAGE_ROOT}" \
    VLLM_SLEEP_LEVEL=1 \
    VF_LEARNER_ACTIVATION_OFFLOAD=1 \
    VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB=12 \
    ORIGINAL_SCORE_CACHE_PATH="${ORIGINAL_SCORE_CACHE_PATH}" \
    ORIGINAL_SCORE_CACHE_SHA256="${ORIGINAL_SCORE_CACHE_SHA256}" \
    ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT="${ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT}" \
    ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT="${ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT}" \
    ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS="${ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS}" \
    ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA="${ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA}" \
    ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN="${ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN:-0.0}" \
    ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX="${ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX:-5.0}" \
    JUDGE_MODEL_ID="${JUDGER_MODEL_ID}" \
    JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH}" \
    JUDGE_MODEL_TREE_SHA256="${JUDGE_MODEL_TREE_SHA256}" \
    JUDGE_MODEL_EXPORT_TREE_SHA256="${JUDGE_MODEL_EXPORT_TREE_SHA256}" \
    JUDGE_PROMPT_HASH="${JUDGE_PROMPT_HASH}" \
    COMPONENT_CREDIT_MASK_MODE="${mask_mode}" \
    EXPECTED_COMPONENT_CREDIT_MASK_MODE="${mask_mode}" \
    COMPONENT_KL_MODE=off \
    EXPECTED_COMPONENT_KL_MODE=off \
    BETA_KL_REASONING=0 \
    BETA_KL_RATING=0 \
    EXPECTED_BETA_KL_REASONING=0 \
    EXPECTED_BETA_KL_RATING=0 \
    REFERENCE_ACTIVATION_BETA=0 \
    REFERENCE_MODEL_PATH="${ACTOR_MODEL_PATH}" \
    REFERENCE_MODEL_TREE_SHA256= \
    JUDGER_MAX_NUM_SEQS=1 \
    JUDGER_MAX_BATCH_SIZE=1 \
    JUDGER_BATCH_WAIT_MS=0 \
    EDITOR_JUDGE_SERVICE_WORKERS=12 \
    RETAINED_DATASET_SOURCE="${TRAIN_DATASET}" \
    TRAIN_MAX_STEPS=30 \
    STOP_AFTER_STEP=30 \
    STEP_START=0 \
    SAVE_STEPS=30 \
    SAVE_TOTAL_LIMIT=2 \
    WANDB_RUN_ID="${wandb_run_id}" \
    FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL}" \
    FLASH_ATTN_WHEEL_SHA256="${FLASH_ATTN_WHEEL_SHA256}" \
    "${RUNNER}"

  stop_services
  write_state passed "${phase}" "checkpoint-30 completed and audited"
}

checkpoint_30() {
  local run_dir="$1"
  local matches count
  matches="$(find "${run_dir}/train" -type d -name checkpoint-30 -print)"
  count="$(printf '%s\n' "${matches}" | sed '/^$/d' | wc -l | tr -d ' ')"
  [[ "${count}" == "1" ]] || {
    echo "expected one checkpoint-30 in ${run_dir}, found ${count}" >&2
    exit 2
  }
  printf '%s\n' "${matches}"
}

validate_config
write_state running preparation "two independent native Actor runs"
run_phase phase1_nomask completion
PHASE1_CHECKPOINT="$(checkpoint_30 "${RUNS_DIR}/phase1_nomask")"
run_phase phase2_mask field
PHASE2_CHECKPOINT="$(checkpoint_30 "${RUNS_DIR}/phase2_mask")"

python3 - \
  "${SUMMARY_FILE}" \
  "${ACTOR_MODEL_PATH}" \
  "${DIFFUSERS_MODEL_PATH}" \
  "${JUDGE_MODEL_PATH}" \
  "${RUNS_DIR}/phase1_nomask" \
  "${RUNS_DIR}/phase2_mask" \
  "${PHASE1_CHECKPOINT}" \
  "${PHASE2_CHECKPOINT}" <<'PY'
import datetime
import json
import pathlib
import sys

(
    output_path,
    actor_model,
    editor_model,
    judge_model,
    no_mask_run,
    mask_run,
    no_mask_checkpoint,
    mask_checkpoint,
) = sys.argv[1:]

payload = {
    "schema_version": "mr_iqa_independent_native_runs_v1",
    "completed_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "actor_initial_model": actor_model,
    "editor_model": editor_model,
    "judge_model": judge_model,
    "independent_runs": True,
    "no_mask": {
        "run_dir": no_mask_run,
        "checkpoint_30": no_mask_checkpoint,
        "credit_mask_mode": "completion",
        "component_kl_mode": "off",
    },
    "mask": {
        "run_dir": mask_run,
        "checkpoint_30": mask_checkpoint,
        "credit_mask_mode": "field",
        "component_kl_mode": "off",
    },
}
path = pathlib.Path(output_path)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

CURRENT_PHASE="complete"
CURRENT_RUN_DIR=""
write_state passed complete "both independent 30-step runs completed"
echo "summary: ${SUMMARY_FILE}"
