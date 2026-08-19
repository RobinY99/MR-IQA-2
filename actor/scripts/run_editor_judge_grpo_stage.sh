#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:?MODE is required: smoke or formal}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
CODE_ROOT="${CODE_ROOT:?CODE_ROOT is required}"
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "${CODE_ROOT}/.." && pwd)}"
SERVICE_RUN_DIR="${SERVICE_RUN_DIR:?SERVICE_RUN_DIR is required}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID is required}"
MODEL_TAG="${MODEL_TAG:?MODEL_TAG is required}"
MODEL_FAMILY="${MODEL_FAMILY:?MODEL_FAMILY is required: qwen3vl or qwen35}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
ALGORITHM="${ALGORITHM:?ALGORITHM is required: dapo or grpo}"
NUM_ITERATIONS="${NUM_ITERATIONS:?NUM_ITERATIONS is required}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:?PER_DEVICE_BATCH_SIZE is required}"
TARGET_EPOCH="${TARGET_EPOCH:-0}"
TOTAL_TRAIN_EPOCHS="${TOTAL_TRAIN_EPOCHS:-5}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:?CONDA_ENV_NAME is required}"
CONDA_SH="${CONDA_SH:?CONDA_SH must point to conda.sh}"
TRAIN_IMAGE_ROOT="${TRAIN_IMAGE_ROOT:?TRAIN_IMAGE_ROOT is required}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PARENT_MANIFEST="${PARENT_MANIFEST:-}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-5}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:-}"
STEP_START="${STEP_START:-0}"
SAVE_STEPS="${SAVE_STEPS:-${STOP_AFTER_STEP:-1}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
FREEZE_VIT="${FREEZE_VIT:-true}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-true}"
BATCH_DECAY_STEP="${BATCH_DECAY_STEP:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
VIT_GRADIENT_CHECKPOINTING="${VIT_GRADIENT_CHECKPOINTING:-true}"
GRADIENT_CHECKPOINTING_USE_REENTRANT="${GRADIENT_CHECKPOINTING_USE_REENTRANT:-false}"
VF_ALLOW_LLM_GC_FALLBACK="${VF_ALLOW_LLM_GC_FALLBACK:-0}"
MARGIN_REWARD_SCOPE="${MARGIN_REWARD_SCOPE:-global_batch}"
MARGIN_IMAGES_PER_COHORT="${MARGIN_IMAGES_PER_COHORT:-0}"
MARGIN_LOCAL_IMAGES_PER_RANK="${MARGIN_LOCAL_IMAGES_PER_RANK:-0}"
REWARD_GATHER_ORDER="${REWARD_GATHER_ORDER:-global_gather_then_reward}"

EXPERIMENT_PROFILE="qwen35_4b_grpo_editor_judge_reasoning_v1"
ACTOR_SCHEMA="reasoning_evidence_solution_rating"
WORLD_SIZE=4
NUM_GENERATIONS=6
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-192}"
SOFT_OVERLONG_MAX_LENGTH="${SOFT_OVERLONG_MAX_LENGTH:-${MAX_COMPLETION_LENGTH}}"
SOFT_OVERLONG_CACHE_LENGTH="${SOFT_OVERLONG_CACHE_LENGTH:-16}"
SOFT_OVERLONG_MAX_PENALTY="${SOFT_OVERLONG_MAX_PENALTY:-1.0}"
SOFT_OVERLONG_WEIGHT="${SOFT_OVERLONG_WEIGHT:-1.0}"
MAX_LENGTH=2048
MAX_PIXELS=196608
MIN_PIXELS=3136
LEARNER_MICROBATCH_SIZE="${LEARNER_MICROBATCH_SIZE:-${PER_DEVICE_BATCH_SIZE}}"
ALLOW_LEARNER_MICROBATCH_SPLIT="${ALLOW_LEARNER_MICROBATCH_SPLIT:-0}"
GENERATION_BATCH_SIZE="$((PER_DEVICE_BATCH_SIZE * WORLD_SIZE))"
PROMPTS_PER_ROLLOUT="$((GENERATION_BATCH_SIZE / NUM_GENERATIONS))"
DATASET_ROWS=7000
ROLLOUTS_PER_EPOCH="$((DATASET_ROWS / PROMPTS_PER_ROLLOUT))"
STEPS_PER_EPOCH="$((ROLLOUTS_PER_EPOCH * NUM_ITERATIONS))"
EXPECTED_START_STEP="$(((TARGET_EPOCH - 1) * STEPS_PER_EPOCH))"
EXPECTED_END_STEP="$((TARGET_EPOCH * STEPS_PER_EPOCH))"
EXPECTED_MESSAGES_HASH="daa804f5a03e74ccb08abdb5a7f92b4bafdcbb42d17e5d5521ddf09972f8b748"
EXPECTED_PROMPT_CONTRACT_HASH="fa78a4ccfd2194a2026ff0b6b722bf22b28f8fa060389c57c4adb1618ac280f6"
RETAINED_DATASET_SOURCE="${RETAINED_DATASET_SOURCE:?RETAINED_DATASET_SOURCE is required}"
ORIGINAL_SCORE_CACHE_PATH="${ORIGINAL_SCORE_CACHE_PATH:?ORIGINAL_SCORE_CACHE_PATH is required}"
ORIGINAL_SCORE_CACHE_SHA256="${ORIGINAL_SCORE_CACHE_SHA256:?ORIGINAL_SCORE_CACHE_SHA256 is required}"
ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT="${ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT:-7000}"
ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT="${ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT:-7000}"
ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS="${ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS:-source-e5-judge-step725-original-score}"
ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA="${ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA:-vf_original_score_cache_e5_judge_v1}"
ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN="${ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN:-0.0}"
ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX="${ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX:-5.0}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-source-e5-judge-step725}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:?JUDGE_MODEL_PATH is required}"
JUDGE_MODEL_TREE_SHA256="${JUDGE_MODEL_TREE_SHA256:?JUDGE_MODEL_TREE_SHA256 is required}"
JUDGE_PROMPT_HASH="${JUDGE_PROMPT_HASH:-fa78a4ccfd2194a2026ff0b6b722bf22b28f8fa060389c57c4adb1618ac280f6}"
COMPONENT_CREDIT_MASK_MODE="${COMPONENT_CREDIT_MASK_MODE:-field}"
EXPECTED_COMPONENT_CREDIT_MASK_MODE="${EXPECTED_COMPONENT_CREDIT_MASK_MODE:-field}"
COMPONENT_KL_MODE="${COMPONENT_KL_MODE:-off}"
EXPECTED_COMPONENT_KL_MODE="${EXPECTED_COMPONENT_KL_MODE:-${COMPONENT_KL_MODE}}"
BETA_KL_REASONING="${BETA_KL_REASONING:-0}"
BETA_KL_RATING="${BETA_KL_RATING:-0}"
EXPECTED_BETA_KL_REASONING="${EXPECTED_BETA_KL_REASONING:-${BETA_KL_REASONING}}"
EXPECTED_BETA_KL_RATING="${EXPECTED_BETA_KL_RATING:-${BETA_KL_RATING}}"
REFERENCE_ACTIVATION_BETA="${REFERENCE_ACTIVATION_BETA:-0.0}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-${MODEL_PATH}}"
REFERENCE_MODEL_TREE_SHA256="${REFERENCE_MODEL_TREE_SHA256:-}"
JUDGER_MAX_NUM_SEQS="${JUDGER_MAX_NUM_SEQS:-1}"
JUDGER_MAX_BATCH_SIZE="${JUDGER_MAX_BATCH_SIZE:-1}"
JUDGER_BATCH_WAIT_MS="${JUDGER_BATCH_WAIT_MS:-0}"
EDITOR_JUDGE_SERVICE_WORKERS="${EDITOR_JUDGE_SERVICE_WORKERS:-12}"
EXPECTED_TOTAL_TRAIN_EPOCHS="${EXPECTED_TOTAL_TRAIN_EPOCHS:-5}"
EDITOR_URLS="http://127.0.0.1:8212,http://127.0.0.1:8213,http://127.0.0.1:8214,http://127.0.0.1:8215"
JUDGE_URLS="http://127.0.0.1:8204,http://127.0.0.1:8205,http://127.0.0.1:8206,http://127.0.0.1:8207"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:?FLASH_ATTN_WHEEL is required}"
FLASH_ATTN_WHEEL_SHA256="${FLASH_ATTN_WHEEL_SHA256:?FLASH_ATTN_WHEEL_SHA256 is required}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${CODE_ROOT}/configs/zero3_cpu_offload.json}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.22}"
VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-0}"
VF_EMPTY_CACHE_BEFORE_BACKWARD="${VF_EMPTY_CACHE_BEFORE_BACKWARD:-1}"
VF_LEARNER_ACTIVATION_OFFLOAD="${VF_LEARNER_ACTIVATION_OFFLOAD:-0}"
VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB="${VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB:-12}"
export VF_DAPO_MAX_GENERATION_ROUNDS="${VF_DAPO_MAX_GENERATION_ROUNDS:-4}"
export VF_DAPO_MIN_EFFECTIVE_ROWS="${VF_DAPO_MIN_EFFECTIVE_ROWS:-96}"
export VF_DAPO_LOW_EFFECTIVE_ACTION="${VF_DAPO_LOW_EFFECTIVE_ACTION:-error}"

if [[ "${MODE}" == "smoke" ]]; then
  EXPECTED_START_STEP=0
  EXPECTED_END_STEP="${SMOKE_MAX_STEPS}"
elif [[ "${MODE}" == "steps" ]]; then
  EXPECTED_START_STEP="${STEP_START}"
  EXPECTED_END_STEP="${STOP_AFTER_STEP}"
fi

[[ "${MODE}" == "smoke" || "${MODE}" == "formal" || "${MODE}" == "steps" ]] || { echo "invalid MODE=${MODE}" >&2; exit 2; }
[[ "${MODEL_TAG}" == "qwen35_4b" && "${MODEL_FAMILY}" == "qwen35" ]] || { echo "Editor+Judge profile requires Qwen3.5-4B" >&2; exit 2; }
[[ "${ALGORITHM}" == "grpo" ]] || { echo "Editor+Judge profile requires GRPO" >&2; exit 2; }
[[ "${NUM_ITERATIONS}" == "1" ]] || { echo "Editor+Judge profile requires one GRPO iteration" >&2; exit 2; }
[[ "${TOTAL_TRAIN_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || { echo "total train epochs must be positive" >&2; exit 2; }
[[ "${TARGET_EPOCH}" =~ ^[0-9]+$ ]] || { echo "target epoch must be a non-negative integer" >&2; exit 2; }
(( TARGET_EPOCH <= TOTAL_TRAIN_EPOCHS )) || { echo "target epoch exceeds total train epochs" >&2; exit 2; }
if [[ "${MODE}" == "steps" ]]; then
  [[ "${TRAIN_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "steps mode requires positive TRAIN_MAX_STEPS" >&2; exit 2; }
  [[ "${STOP_AFTER_STEP}" =~ ^[1-9][0-9]*$ ]] || { echo "steps mode requires positive STOP_AFTER_STEP" >&2; exit 2; }
  [[ "${STEP_START}" =~ ^[0-9]+$ ]] || { echo "steps mode requires non-negative STEP_START" >&2; exit 2; }
  [[ "${SAVE_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "steps mode requires positive SAVE_STEPS" >&2; exit 2; }
  [[ "${SAVE_TOTAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || { echo "steps mode requires positive SAVE_TOTAL_LIMIT" >&2; exit 2; }
  (( STEP_START < STOP_AFTER_STEP )) || { echo "STEP_START must be below STOP_AFTER_STEP" >&2; exit 2; }
  (( STOP_AFTER_STEP <= TRAIN_MAX_STEPS )) || { echo "STOP_AFTER_STEP exceeds TRAIN_MAX_STEPS" >&2; exit 2; }
fi
[[ "${PER_DEVICE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid batch" >&2; exit 2; }
[[ "${LEARNER_MICROBATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid learner microbatch" >&2; exit 2; }
[[ "${MAX_COMPLETION_LENGTH}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid max completion length" >&2; exit 2; }
(( MAX_COMPLETION_LENGTH <= 256 )) || { echo "max completion length exceeds the 256-token contract cap" >&2; exit 2; }
python3 - \
  "${MAX_COMPLETION_LENGTH}" \
  "${SOFT_OVERLONG_MAX_LENGTH}" \
  "${SOFT_OVERLONG_CACHE_LENGTH}" \
  "${SOFT_OVERLONG_MAX_PENALTY}" \
  "${SOFT_OVERLONG_WEIGHT}" <<'PY'
import math
import sys

completion, maximum, cache = map(int, sys.argv[1:4])
max_penalty, weight = map(float, sys.argv[4:])
if maximum != completion:
    raise SystemExit("soft-overlong max length must equal max completion length")
if not 0 < cache < maximum:
    raise SystemExit("soft-overlong cache length must be in (0, max length)")
if not all(math.isfinite(value) and value >= 0 for value in (max_penalty, weight)):
    raise SystemExit("soft-overlong penalty and weight must be finite and non-negative")
PY
[[ "${ALLOW_LEARNER_MICROBATCH_SPLIT}" == "0" || "${ALLOW_LEARNER_MICROBATCH_SPLIT}" == "1" ]] || { echo "invalid learner microbatch split flag" >&2; exit 2; }
[[ "${FREEZE_VIT}" == "true" || "${FREEZE_VIT}" == "false" ]] || { echo "freeze_vit must be true or false" >&2; exit 2; }
[[ "${FREEZE_ALIGNER}" == "true" || "${FREEZE_ALIGNER}" == "false" ]] || { echo "freeze_aligner must be true or false" >&2; exit 2; }
[[ "${GRADIENT_CHECKPOINTING}" == "true" || "${GRADIENT_CHECKPOINTING}" == "false" ]] || { echo "gradient_checkpointing must be true or false" >&2; exit 2; }
[[ "${VIT_GRADIENT_CHECKPOINTING}" == "true" || "${VIT_GRADIENT_CHECKPOINTING}" == "false" ]] || { echo "vit_gradient_checkpointing must be true or false" >&2; exit 2; }
[[ "${GRADIENT_CHECKPOINTING_USE_REENTRANT}" == "true" || "${GRADIENT_CHECKPOINTING_USE_REENTRANT}" == "false" ]] || { echo "gradient checkpointing use_reentrant must be true or false" >&2; exit 2; }
[[ "${VF_ALLOW_LLM_GC_FALLBACK}" == "0" || "${VF_ALLOW_LLM_GC_FALLBACK}" == "1" ]] || { echo "invalid LLM-GC fallback authorization" >&2; exit 2; }
[[ "${BATCH_DECAY_STEP}" =~ ^[0-9]+$ ]] || { echo "invalid batch decay step" >&2; exit 2; }
[[ "${MARGIN_REWARD_SCOPE}" == "global_batch" || "${MARGIN_REWARD_SCOPE}" == "local_six_images" ]] || { echo "invalid margin reward scope" >&2; exit 2; }
[[ "${MARGIN_IMAGES_PER_COHORT}" =~ ^[0-9]+$ ]] || { echo "invalid margin cohort image count" >&2; exit 2; }
[[ "${MARGIN_LOCAL_IMAGES_PER_RANK}" =~ ^[0-9]+$ ]] || { echo "invalid local margin image count" >&2; exit 2; }
[[ "${VLLM_GPU_MEMORY_UTILIZATION}" == "0.22" ]] || { echo "vLLM GPU memory utilization is locked to 0.22" >&2; exit 2; }
[[ "${VLLM_SLEEP_LEVEL}" == "0" || "${VLLM_SLEEP_LEVEL}" == "1" ]] || { echo "vLLM sleep level must be 0 or 1" >&2; exit 2; }
[[ "${VF_EMPTY_CACHE_BEFORE_BACKWARD}" == "1" ]] || { echo "pre-backward cache release is required" >&2; exit 2; }
[[ "${VF_LEARNER_ACTIVATION_OFFLOAD}" == "0" || "${VF_LEARNER_ACTIVATION_OFFLOAD}" == "1" ]] || { echo "learner activation offload must be 0 or 1" >&2; exit 2; }
[[ "${VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB}" == "12" ]] || { echo "learner activation offload budget is locked to 12 GiB per rank" >&2; exit 2; }
[[ "${VF_DAPO_MAX_GENERATION_ROUNDS}" =~ ^[1-9][0-9]*$ ]] || { echo "DAPO generation rounds must be positive" >&2; exit 2; }
(( VF_DAPO_MAX_GENERATION_ROUNDS <= 10 )) || { echo "DAPO generation rounds exceed the safety cap of ten" >&2; exit 2; }
[[ "${VF_DAPO_MIN_EFFECTIVE_ROWS}" =~ ^[1-9][0-9]*$ ]] || { echo "DAPO minimum effective rows must be positive" >&2; exit 2; }
(( VF_DAPO_MIN_EFFECTIVE_ROWS <= GENERATION_BATCH_SIZE )) || { echo "DAPO minimum effective rows exceed the generation batch" >&2; exit 2; }
(( VF_DAPO_MIN_EFFECTIVE_ROWS % NUM_GENERATIONS == 0 )) || { echo "DAPO minimum effective rows must preserve complete generation groups" >&2; exit 2; }
[[ "${VF_DAPO_LOW_EFFECTIVE_ACTION}" == "error" || "${VF_DAPO_LOW_EFFECTIVE_ACTION}" == "skip_batch" ]] || { echo "invalid DAPO low-effective action" >&2; exit 2; }
[[ "${COMPONENT_CREDIT_MASK_MODE}" == "field" || "${COMPONENT_CREDIT_MASK_MODE}" == "completion" ]] || { echo "invalid component credit mask mode" >&2; exit 2; }
[[ "${COMPONENT_CREDIT_MASK_MODE}" == "${EXPECTED_COMPONENT_CREDIT_MASK_MODE}" ]] || { echo "component credit mask mode differs from the locked contract" >&2; exit 2; }
[[ "${COMPONENT_KL_MODE}" == "off" || "${COMPONENT_KL_MODE}" == "field" ]] || { echo "invalid component KL mode" >&2; exit 2; }
[[ "${COMPONENT_KL_MODE}" == "${EXPECTED_COMPONENT_KL_MODE}" ]] || {
  echo "component KL mode differs from the locked contract" >&2
  exit 2
}
python3 - \
  "${COMPONENT_KL_MODE}" \
  "${BETA_KL_REASONING}" \
  "${BETA_KL_RATING}" \
  "${EXPECTED_BETA_KL_REASONING}" \
  "${EXPECTED_BETA_KL_RATING}" \
  "${REFERENCE_ACTIVATION_BETA}" <<'PY'
import math
import sys

mode = sys.argv[1]
reasoning, rating, expected_reasoning, expected_rating, activation = map(
    float, sys.argv[2:]
)
values = {
    "reasoning": reasoning,
    "rating": rating,
    "expected_reasoning": expected_reasoning,
    "expected_rating": expected_rating,
    "reference_activation": activation,
}
if not all(math.isfinite(value) and value >= 0 for value in values.values()):
    raise SystemExit(f"component KL values must be finite and non-negative: {values}")
if not math.isclose(reasoning, expected_reasoning, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("reasoning KL beta differs from the locked contract")
if not math.isclose(rating, expected_rating, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("rating KL beta differs from the locked contract")
if mode == "off" and any(value != 0 for value in (reasoning, rating)):
    raise SystemExit("component KL off requires zero reasoning/rating component betas")
if mode == "field" and not (
    reasoning > 0 and rating > 0 and activation > 0
):
    raise SystemExit("field component KL requires positive component and activation betas")
PY
if [[ "${COMPONENT_KL_MODE}" == "field" ]]; then
  [[ "${COMPONENT_CREDIT_MASK_MODE}" == "field" ]] || { echo "field component KL requires field credit masks" >&2; exit 2; }
fi
if python3 - "${REFERENCE_ACTIVATION_BETA}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)
PY
then
  [[ "$(realpath "${REFERENCE_MODEL_PATH}")" == "$(realpath "${MODEL_PATH}")" ]] || {
    echo "KL reference model must be the fixed initial Actor" >&2
    exit 2
  }
  [[ -n "${REFERENCE_MODEL_TREE_SHA256}" ]] || { echo "KL reference model tree hash is required" >&2; exit 2; }
fi
python3 - \
  "${JUDGER_MAX_NUM_SEQS}" \
  "${JUDGER_MAX_BATCH_SIZE}" \
  "${JUDGER_BATCH_WAIT_MS}" \
  "${EDITOR_JUDGE_SERVICE_WORKERS}" <<'PY'
import math
import sys

max_num_seqs, max_batch_size, service_workers = map(
    int, (sys.argv[1], sys.argv[2], sys.argv[4])
)
batch_wait_ms = float(sys.argv[3])
if not 1 <= max_batch_size <= max_num_seqs <= 32:
    raise SystemExit("Judge batching requires 1 <= batch <= max_num_seqs <= 32")
if not math.isfinite(batch_wait_ms) or not 0 <= batch_wait_ms <= 100:
    raise SystemExit("Judge batch wait must be in [0, 100] ms")
if not 1 <= service_workers <= 72:
    raise SystemExit("Editor/Judge service workers must be in [1, 72]")
PY
(( GENERATION_BATCH_SIZE % NUM_GENERATIONS == 0 )) || { echo "global batch must divide by generations" >&2; exit 2; }
(( LEARNER_MICROBATCH_SIZE <= PER_DEVICE_BATCH_SIZE )) || { echo "learner microbatch exceeds per-device batch" >&2; exit 2; }
(( PER_DEVICE_BATCH_SIZE % LEARNER_MICROBATCH_SIZE == 0 )) || { echo "per-device batch must divide by learner microbatch" >&2; exit 2; }
if [[ "${ALLOW_LEARNER_MICROBATCH_SPLIT}" == "0" ]]; then
  (( LEARNER_MICROBATCH_SIZE == PER_DEVICE_BATCH_SIZE )) || { echo "learner microbatch split was not explicitly enabled" >&2; exit 2; }
fi
if [[ "${MARGIN_REWARD_SCOPE}" == "local_six_images" ]]; then
  [[ "${WORLD_SIZE}" == "4" && "${NUM_GENERATIONS}" == "6" ]] || { echo "Editor+Judge local-six margin requires 4 actor GPUs and g6" >&2; exit 2; }
  (( PER_DEVICE_BATCH_SIZE == MARGIN_LOCAL_IMAGES_PER_RANK * NUM_GENERATIONS )) || { echo "local-six batch must match rank-local image count times g6" >&2; exit 2; }
  [[ "${MARGIN_IMAGES_PER_COHORT}" == "6" ]] || { echo "local-six margin requires 6 images per cohort" >&2; exit 2; }
  (( MARGIN_LOCAL_IMAGES_PER_RANK > 0 && MARGIN_LOCAL_IMAGES_PER_RANK % MARGIN_IMAGES_PER_COHORT == 0 )) || { echo "local-six rank-local images must form complete six-image cohorts" >&2; exit 2; }
  [[ "${REWARD_GATHER_ORDER}" == "local_reward_then_global_gather" ]] || { echo "local-six margin requires reward-before-gather order" >&2; exit 2; }
  [[ "${BATCH_DECAY_STEP}" == "0" ]] || { echo "local-six effective batch is fixed; use explicit learner-microbatch fallback" >&2; exit 2; }
else
  [[ "${REWARD_GATHER_ORDER}" == "global_gather_then_reward" ]] || { echo "global margin requires gather-before-reward order" >&2; exit 2; }
fi
[[ "${TOTAL_TRAIN_EPOCHS}" == "${EXPECTED_TOTAL_TRAIN_EPOCHS}" ]] || { echo "Editor+Judge GRPO epoch count differs from the locked contract" >&2; exit 2; }
[[ "${FREEZE_VIT}" == "true" && "${FREEZE_ALIGNER}" == "true" ]] || { echo "Editor+Judge GRPO profile requires frozen ViT and aligner" >&2; exit 2; }
[[ "${VIT_GRADIENT_CHECKPOINTING}" == "true" ]] || { echo "Editor+Judge GRPO profile requires vision_gc=true configured" >&2; exit 2; }
if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
  [[ "${VF_ALLOW_LLM_GC_FALLBACK}" == "1" ]] || { echo "LLM GC requires an archived-evidence fallback authorization" >&2; exit 2; }
  [[ "${MAX_COMPLETION_LENGTH}" == "160" ]] || { echo "LLM GC fallback is only allowed after c160 OOM" >&2; exit 2; }
  [[ "${GRADIENT_CHECKPOINTING_USE_REENTRANT}" == "false" ]] || { echo "LLM GC fallback must be non-reentrant" >&2; exit 2; }
  [[ "${VF_LEARNER_ACTIVATION_OFFLOAD}" == "1" ]] || { echo "LLM GC fallback must retain bounded activation offload" >&2; exit 2; }
else
  [[ "${VF_ALLOW_LLM_GC_FALLBACK}" == "0" ]] || { echo "LLM-GC fallback authorization cannot be set while LLM GC is disabled" >&2; exit 2; }
fi
[[ "${PER_DEVICE_BATCH_SIZE}" == "36" && "${LEARNER_MICROBATCH_SIZE}" == "36" ]] || { echo "Editor+Judge GRPO profile requires b36/mb36" >&2; exit 2; }
[[ "${MARGIN_REWARD_SCOPE}" == "local_six_images" && "${MARGIN_IMAGES_PER_COHORT}" == "6" && "${MARGIN_LOCAL_IMAGES_PER_RANK}" == "6" ]] || { echo "Editor+Judge GRPO profile requires local-six reward" >&2; exit 2; }
[[ "${MAX_COMPLETION_LENGTH}" == "192" || "${MAX_COMPLETION_LENGTH}" == "160" ]] || { echo "Editor+Judge GRPO completion length must be 192 or 160" >&2; exit 2; }
[[ -f "${ORIGINAL_SCORE_CACHE_PATH}" ]] || { echo "missing original-score cache" >&2; exit 2; }
[[ "$(sha256sum "${ORIGINAL_SCORE_CACHE_PATH}" | awk '{print $1}')" == "${ORIGINAL_SCORE_CACHE_SHA256}" ]] || { echo "original-score cache hash mismatch" >&2; exit 2; }
if [[ "${MODE}" == "formal" ]]; then
  (( TARGET_EPOCH >= 1 )) || { echo "formal mode requires target epoch 1..TOTAL_TRAIN_EPOCHS" >&2; exit 2; }
  [[ -n "${WANDB_RUN_ID}" ]] || { echo "formal mode requires stable WANDB_RUN_ID" >&2; exit 2; }
  if (( TARGET_EPOCH == 1 )); then
    [[ -z "${RESUME_CHECKPOINT}" && -z "${PARENT_MANIFEST}" ]] || { echo "epoch1 must start from native" >&2; exit 2; }
  else
    [[ -d "${RESUME_CHECKPOINT}" && -f "${PARENT_MANIFEST}" ]] || { echo "epoch2+ requires promoted full-state parent" >&2; exit 2; }
    RESOLVED_PARENT="$(python3 "${CODE_ROOT}/scripts/checkpoint_manifest.py" resolve --manifest "${PARENT_MANIFEST}")"
    [[ "$(realpath "${RESOLVED_PARENT}")" == "$(realpath "${RESUME_CHECKPOINT}")" ]] || {
      echo "resume checkpoint does not match promoted parent manifest" >&2
      exit 2
    }
    python3 - "${RESUME_CHECKPOINT}" "${EXPECTED_START_STEP}" <<'PY'
import json, pathlib, sys
checkpoint = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
assert int(state["global_step"]) == expected, (state.get("global_step"), expected)
assert (checkpoint / "scheduler.pt").is_file()
assert list(checkpoint.glob("rng_state*.pth"))
assert list(checkpoint.rglob("*optim_states.pt")) or (checkpoint / "optimizer.pt").is_file()
PY
  fi
fi
if [[ "${MODE}" == "steps" ]]; then
  [[ -n "${WANDB_RUN_ID}" ]] || { echo "steps mode requires stable WANDB_RUN_ID" >&2; exit 2; }
  if (( STEP_START == 0 )); then
    [[ -z "${RESUME_CHECKPOINT}" && -z "${PARENT_MANIFEST}" ]] || {
      echo "step-zero phase must start from the approved native model" >&2
      exit 2
    }
  else
    [[ -d "${RESUME_CHECKPOINT}" ]] || { echo "continued steps phase requires a full-state checkpoint" >&2; exit 2; }
    python3 - "${RESUME_CHECKPOINT}" "${STEP_START}" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
assert int(state["global_step"]) == expected, (state.get("global_step"), expected)
assert (checkpoint / "scheduler.pt").is_file()
assert list(checkpoint.glob("rng_state*.pth"))
assert list(checkpoint.rglob("*optim_states.pt")) or (checkpoint / "optimizer.pt").is_file()
PY
  fi
fi

STAGE_ROOT_FROM_RUN="$(dirname "$(dirname "$(dirname "${RUN_DIR}")")")"
PAUSE_MARKER="${STAGE_ROOT_FROM_RUN}/state/pause_before_epoch_${TARGET_EPOCH}"
if [[ "${MODE}" == "formal" && "${TARGET_EPOCH}" -ge 2 && -f "${PAUSE_MARKER}" ]]; then
  python3 - "${STAGE_ROOT_FROM_RUN}" "${EXPERIMENT_ID}" "${TARGET_EPOCH}" "${PPID}" <<'PY'
import datetime
import json
import os
import pathlib
import sys

stage = pathlib.Path(sys.argv[1])
experiment = sys.argv[2]
target_epoch = int(sys.argv[3])
wrapper_pid = int(sys.argv[4])
previous_epoch = target_epoch - 1
promoted = []
for run_dir in sorted(
    (stage / "runs" / experiment).glob(f"epoch{previous_epoch}_attempt*")
):
    run_state = run_dir / "artifacts" / "run_state.json"
    if not run_state.is_file():
        continue
    payload = json.loads(run_state.read_text(encoding="utf-8"))
    if payload.get("status") == "promoted":
        promoted.append(run_dir)
if len(promoted) != 1:
    raise RuntimeError(
        f"pause gate requires one promoted E{previous_epoch} run, found {promoted}"
    )

now = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
family_state = {
    "updated_at": now,
    "status": "paused",
    "phase": "between_epochs",
    "current_experiment": experiment,
    "current_epoch": previous_epoch,
    "current_run_dir": str(promoted[0]),
    "detail": (
        f"E{previous_epoch} training, tensor audit, validation and ranking complete; "
        f"E{target_epoch} launch is paused by user request"
    ),
    "wrapper_pid": wrapper_pid,
}
state_path = stage / "state" / "family_state.json"
tmp = state_path.with_suffix(".tmp")
tmp.write_text(
    json.dumps(family_state, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(tmp, state_path)

events_path = stage / "state" / "events.jsonl"
existing = []
if events_path.is_file():
    existing = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
event_name = "paused_after_promoted_epoch"
if not any(
    event.get("event") == event_name
    and event.get("experiment") == experiment
    and int(event.get("epoch") or 0) == previous_epoch
    for event in existing
):
    event = {
        "time": now,
        "event": event_name,
        "experiment": experiment,
        "epoch": previous_epoch,
        "run_dir": str(promoted[0]),
        "detail": (
            f"E{previous_epoch} validated and ranked; E{target_epoch} not launched"
        ),
    }
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
PY
  while [[ -f "${PAUSE_MARKER}" ]]; do
    sleep 30
  done
  python3 - "${STAGE_ROOT_FROM_RUN}" "${EXPERIMENT_ID}" "${TARGET_EPOCH}" "${RUN_DIR}" "${PPID}" <<'PY'
import datetime
import json
import os
import pathlib
import sys

stage = pathlib.Path(sys.argv[1])
experiment = sys.argv[2]
target_epoch = int(sys.argv[3])
run_dir = sys.argv[4]
wrapper_pid = int(sys.argv[5])
now = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
state = {
    "updated_at": now,
    "status": "running",
    "phase": "training",
    "current_experiment": experiment,
    "current_epoch": target_epoch,
    "current_run_dir": run_dir,
    "detail": f"pause released; preparing E{target_epoch}",
    "wrapper_pid": wrapper_pid,
}
state_path = stage / "state" / "family_state.json"
tmp = state_path.with_suffix(".tmp")
tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, state_path)
event = {
    "time": now,
    "event": "pause_released",
    "experiment": experiment,
    "epoch": target_epoch - 1,
    "run_dir": run_dir,
    "detail": f"pause marker removed; E{target_epoch} may launch when GPUs are idle",
}
with (stage / "state" / "events.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(event, sort_keys=True) + "\n")
PY
fi

[[ -d "${MODEL_PATH}" ]] || { echo "missing model: ${MODEL_PATH}" >&2; exit 2; }
[[ -d "${CODE_ROOT}" ]] || { echo "missing code root: ${CODE_ROOT}" >&2; exit 2; }
[[ -f "${CODE_ROOT}/plugin/vf_dual_rollout_trainer.py" ]] || { echo "missing trainer plugin" >&2; exit 2; }
[[ -f "${RETAINED_DATASET_SOURCE}" ]] || { echo "missing training dataset" >&2; exit 2; }
[[ -f "${DEEPSPEED_CONFIG}" ]] || { echo "missing DeepSpeed config: ${DEEPSPEED_CONFIG}" >&2; exit 2; }
python3 - "${DEEPSPEED_CONFIG}" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
zero = config["zero_optimization"]
assert int(zero["stage"]) == 3
assert zero["offload_optimizer"]["device"] == "cpu"
assert zero["offload_param"]["device"] == "cpu"
assert zero["overlap_comm"] is True
assert int(zero["reduce_bucket_size"]) == 25_000_000
assert int(zero["allgather_bucket_size"]) == 25_000_000
assert int(zero["stage3_prefetch_bucket_size"]) == 25_000_000
assert config["zero_force_ds_cpu_optimizer"] is False
assert config["bf16"]["enabled"] is True
assert config["fp16"]["enabled"] is False
PY
if [[ -e "${RUN_DIR}" ]]; then
  [[ -d "${RUN_DIR}" ]] || { echo "run path exists but is not a directory: ${RUN_DIR}" >&2; exit 2; }
  [[ "$(realpath "${RUN_DIR}")" == "$(realpath "${SERVICE_RUN_DIR}")" ]] || {
    echo "an existing run directory is allowed only when owned by its service stack" >&2
    exit 2
  }
  [[ -d "${RUN_DIR}/services" ]] || {
    echo "existing run directory has no owned service stack: ${RUN_DIR}" >&2
    exit 2
  }
  [[ ! -e "${RUN_DIR}/train" && ! -e "${RUN_DIR}/input" && ! -e "${RUN_DIR}/logs" ]] || {
    echo "existing run directory already contains training state: ${RUN_DIR}" >&2
    exit 2
  }
  [[ ! -e "${RUN_DIR}/artifacts/config.json" && ! -e "${RUN_DIR}/artifacts/run_result.json" ]] || {
    echo "existing run directory already contains a training attempt: ${RUN_DIR}" >&2
    exit 2
  }
fi

TRAIN_DIR="${RUN_DIR}/train"
ARTIFACT_DIR="${RUN_DIR}/artifacts"
LOG_DIR="${RUN_DIR}/logs"
INPUT_DIR="${RUN_DIR}/input"
DATASET_FILE="${INPUT_DIR}/koniq7k_seed42_reasoning_evidence_solution_rating.jsonl"
mkdir -p "${TRAIN_DIR}" "${ARTIFACT_DIR}" "${LOG_DIR}" "${INPUT_DIR}"

source "${CONDA_SH}"
set +u
conda activate "${CONDA_ENV_NAME}"
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
source "${CODE_ROOT}/scripts/vllm_cuda13_runtime.sh"
vllm_prepare_cuda13_runtime

[[ -f "${FLASH_ATTN_WHEEL}" ]] || { echo "missing FlashAttention wheel" >&2; exit 2; }
[[ "$(sha256sum "${FLASH_ATTN_WHEEL}" | awk '{print $1}')" == "${FLASH_ATTN_WHEEL_SHA256}" ]] || {
  echo "FlashAttention wheel hash mismatch" >&2
  exit 2
}
MODEL_FAMILY="${MODEL_FAMILY}" python3 - <<'PY' | tee "${LOG_DIR}/package_preflight.log"
import importlib.metadata as metadata
import os
import flash_attn
from transformers.utils import is_flash_attn_2_available

assert is_flash_attn_2_available()
print("flash-attn", flash_attn.__version__)
if os.environ["MODEL_FAMILY"] == "qwen35":
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
    assert is_flash_linear_attention_available()
    assert is_causal_conv1d_available()
    assert modeling_qwen3_5.is_fast_path_available
    assert modeling_qwen3_5.causal_conv1d_fn is not None
    assert modeling_qwen3_5.chunk_gated_delta_rule is not None
    print("flash-linear-attention", metadata.version("flash-linear-attention"))
    print("causal-conv1d", metadata.version("causal-conv1d"))
    print("qwen3.5 fused fast path true")
PY

VF_ACTOR_SCHEMA="${ACTOR_SCHEMA}" python3 "${CODE_ROOT}/scripts/prepare_phase_a_dataset.py" \
  --input "${RETAINED_DATASET_SOURCE}" \
  --output "${DATASET_FILE}" \
  --image-root "${TRAIN_IMAGE_ROOT}" \
  --seed 42 \
  --preserve-order

bash "${CODE_ROOT}/scripts/resident_service_stack.sh" status "${SERVICE_RUN_DIR}"
python3 - <<'PY'
import subprocess

gpu_rows = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).splitlines()
uuid_to_index = {
    row.split(",", 1)[1].strip(): int(row.split(",", 1)[0].strip())
    for row in gpu_rows
}
process_rows = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).splitlines()
actor_processes = []
for row in process_rows:
    if not row.strip():
        continue
    pid, gpu_uuid = [item.strip() for item in row.split(",", 1)]
    gpu = uuid_to_index.get(gpu_uuid)
    if gpu in {0, 1, 2, 3}:
        actor_processes.append((int(pid), gpu))
if actor_processes:
    raise SystemExit(f"actor GPUs are not idle: {actor_processes}")
print("[editor-judge-preflight] actor GPUs 0-3 idle; resident services 4-7 healthy")
PY

DATA_SHA256="$(sha256sum "${DATASET_FILE}" | awk '{print $1}')"
MODEL_CONFIG_SHA256="$(sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')"
DEEPSPEED_CONFIG_SHA256="$(sha256sum "${DEEPSPEED_CONFIG}" | awk '{print $1}')"
read -r PROMPT_HASH PROMPT_CONTRACT_HASH < <(python3 - "${DATASET_FILE}" <<'PY'
import hashlib, json, sys
row = json.loads(open(sys.argv[1], encoding="utf-8").readline())
messages = json.dumps(row.get("messages"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(messages.encode()).hexdigest(), row.get("prompt_hash") or "")
PY
)
read -r EDITOR_PROMPT_VERSION EDITOR_PROMPT_TEMPLATE_HASH < <(
  PYTHONPATH="${CODE_ROOT}/plugin" python3 - <<'PY'
from editor_judge_contract import EDITOR_PROMPT_TEMPLATE_HASH, EDITOR_PROMPT_VERSION
print(EDITOR_PROMPT_VERSION, EDITOR_PROMPT_TEMPLATE_HASH)
PY
)
[[ "${PROMPT_HASH}" == "${EXPECTED_MESSAGES_HASH}" ]] || {
  echo "messages hash mismatch: ${PROMPT_HASH} != ${EXPECTED_MESSAGES_HASH}" >&2
  exit 4
}
[[ "${PROMPT_CONTRACT_HASH}" == "${EXPECTED_PROMPT_CONTRACT_HASH}" ]] || {
  echo "prompt contract hash mismatch: ${PROMPT_CONTRACT_HASH} != ${EXPECTED_PROMPT_CONTRACT_HASH}" >&2
  exit 4
}
CODE_SHA256="$(python3 - "${CODE_ROOT}" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
h = hashlib.sha256()
paths = sorted((root / "plugin").glob("*.py")) + sorted(
    path for path in (root / "scripts").iterdir()
    if path.is_file() and path.suffix in {".py", ".sh"}
)
for path in paths:
    h.update(path.relative_to(root).as_posix().encode())
    h.update(b"\0")
    h.update(path.read_bytes())
    h.update(b"\n")
print(h.hexdigest())
PY
)"
GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')"

export MODE RUN_DIR CODE_ROOT EXPERIMENT_ID MODEL_TAG MODEL_FAMILY MODEL_PATH ALGORITHM
export NUM_ITERATIONS PER_DEVICE_BATCH_SIZE TARGET_EPOCH TOTAL_TRAIN_EPOCHS RESUME_CHECKPOINT PARENT_MANIFEST
export TRAIN_MAX_STEPS STOP_AFTER_STEP STEP_START SAVE_STEPS SAVE_TOTAL_LIMIT
export WORLD_SIZE NUM_GENERATIONS MAX_COMPLETION_LENGTH MAX_LENGTH MAX_PIXELS MIN_PIXELS
export SOFT_OVERLONG_MAX_LENGTH SOFT_OVERLONG_CACHE_LENGTH
export SOFT_OVERLONG_MAX_PENALTY SOFT_OVERLONG_WEIGHT
export LEARNER_MICROBATCH_SIZE ALLOW_LEARNER_MICROBATCH_SPLIT GENERATION_BATCH_SIZE PROMPTS_PER_ROLLOUT DATASET_ROWS
export ROLLOUTS_PER_EPOCH STEPS_PER_EPOCH EXPECTED_START_STEP EXPECTED_END_STEP
export DATASET_FILE RETAINED_DATASET_SOURCE DATA_SHA256 MODEL_CONFIG_SHA256 PROMPT_HASH PROMPT_CONTRACT_HASH CODE_SHA256 GPU_TOTAL_MIB
export DEEPSPEED_CONFIG DEEPSPEED_CONFIG_SHA256 VLLM_GPU_MEMORY_UTILIZATION VLLM_SLEEP_LEVEL
export VF_EMPTY_CACHE_BEFORE_BACKWARD VF_LEARNER_ACTIVATION_OFFLOAD
export VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB
export FREEZE_VIT FREEZE_ALIGNER BATCH_DECAY_STEP GRADIENT_CHECKPOINTING VIT_GRADIENT_CHECKPOINTING
export GRADIENT_CHECKPOINTING_USE_REENTRANT
export MARGIN_REWARD_SCOPE MARGIN_IMAGES_PER_COHORT MARGIN_LOCAL_IMAGES_PER_RANK REWARD_GATHER_ORDER
export EXPERIMENT_PROFILE ACTOR_SCHEMA SERVICE_RUN_DIR
export ORIGINAL_SCORE_CACHE_PATH ORIGINAL_SCORE_CACHE_SHA256
export ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT
export ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA
export ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX
export JUDGE_MODEL_ID JUDGE_MODEL_PATH JUDGE_MODEL_TREE_SHA256 JUDGE_PROMPT_HASH
export COMPONENT_CREDIT_MASK_MODE EXPECTED_COMPONENT_CREDIT_MASK_MODE
export COMPONENT_KL_MODE EXPECTED_COMPONENT_KL_MODE BETA_KL_REASONING BETA_KL_RATING
export EXPECTED_BETA_KL_REASONING EXPECTED_BETA_KL_RATING
export REFERENCE_ACTIVATION_BETA REFERENCE_MODEL_PATH REFERENCE_MODEL_TREE_SHA256
export JUDGER_MAX_NUM_SEQS JUDGER_MAX_BATCH_SIZE JUDGER_BATCH_WAIT_MS
export EDITOR_JUDGE_SERVICE_WORKERS
export EDITOR_PROMPT_VERSION EDITOR_PROMPT_TEMPLATE_HASH
export EDITOR_URLS JUDGE_URLS
python3 - "${ARTIFACT_DIR}/config.json" <<'PY'
import importlib.metadata as metadata
import json, os, pathlib, platform
import flash_attn, torch, transformers

def integer(name): return int(os.environ[name])
def version(name):
    try: return metadata.version(name)
    except metadata.PackageNotFoundError: return None

deepspeed_config = json.loads(pathlib.Path(os.environ["DEEPSPEED_CONFIG"]).read_text())
zero3 = deepspeed_config["zero_optimization"]
margin_scope = os.environ["MARGIN_REWARD_SCOPE"]

config = {
    "contract": (
        "vf_iqa_qwen35_4b_grpo_editor_judge_completion_credit_e5_judge_v1"
        if os.environ["COMPONENT_CREDIT_MASK_MODE"] == "completion"
        else "vf_iqa_qwen35_4b_grpo_editor_judge_component_credit_e5_v1"
    ),
    "experiment_profile": os.environ["EXPERIMENT_PROFILE"],
    "mode": os.environ["MODE"],
    "experiment_id": os.environ["EXPERIMENT_ID"],
    "model_tag": os.environ["MODEL_TAG"],
    "model_family": os.environ["MODEL_FAMILY"],
    "model_path": os.environ["MODEL_PATH"],
    "model_config_sha256": os.environ["MODEL_CONFIG_SHA256"],
    "initial_actor_tree_sha256": (
        os.environ["REFERENCE_MODEL_TREE_SHA256"]
        if float(os.environ["REFERENCE_ACTIVATION_BETA"]) > 0
        else None
    ),
    "algorithm": os.environ["ALGORITHM"],
    "rlhf_type": "grpo",
    "loss_type": os.environ["ALGORITHM"],
    "target_epoch": integer("TARGET_EPOCH"),
    "total_train_epochs": integer("TOTAL_TRAIN_EPOCHS"),
    "resume_checkpoint": os.environ.get("RESUME_CHECKPOINT") or None,
    "parent_manifest": os.environ.get("PARENT_MANIFEST") or None,
    "world_size": integer("WORLD_SIZE"),
    "per_device_batch_size": integer("PER_DEVICE_BATCH_SIZE"),
    "learner_microbatch_size": integer("LEARNER_MICROBATCH_SIZE"),
    "backward_chunks_per_rank": integer("PER_DEVICE_BATCH_SIZE") // integer("LEARNER_MICROBATCH_SIZE"),
    "learner_microbatch_split_enabled": os.environ["ALLOW_LEARNER_MICROBATCH_SPLIT"] == "1",
    "global_generation_batch_size": integer("GENERATION_BATCH_SIZE"),
    "num_generations": integer("NUM_GENERATIONS"),
    "num_iterations": integer("NUM_ITERATIONS"),
    "dataset_rows": integer("DATASET_ROWS"),
    "prompts_per_rollout": integer("PROMPTS_PER_ROLLOUT"),
    "rollouts_per_epoch": integer("ROLLOUTS_PER_EPOCH"),
    "optimizer_steps_per_epoch": integer("STEPS_PER_EPOCH"),
    "data_steps_per_epoch": integer("STEPS_PER_EPOCH"),
    "optimizer_steps_per_epoch_upper_bound": integer("STEPS_PER_EPOCH"),
    "expected_start_step": integer("EXPECTED_START_STEP"),
    "expected_end_step": integer("EXPECTED_END_STEP"),
    "train_max_steps": (
        integer("TRAIN_MAX_STEPS")
        if os.environ.get("TRAIN_MAX_STEPS")
        else None
    ),
    "stop_after_step": (
        integer("STOP_AFTER_STEP")
        if os.environ.get("STOP_AFTER_STEP")
        else None
    ),
    "max_completion_length": integer("MAX_COMPLETION_LENGTH"),
    "max_length": integer("MAX_LENGTH"),
    "max_pixels": integer("MAX_PIXELS"),
    "min_pixels": integer("MIN_PIXELS"),
    "min_pixels_transport": "template_environment",
    "dataset_file": os.environ["DATASET_FILE"],
    "dataset_source": os.environ["RETAINED_DATASET_SOURCE"],
    "data_sha256": os.environ["DATA_SHA256"],
    "prompt_hash": os.environ["PROMPT_HASH"],
    "prompt_contract_hash": os.environ["PROMPT_CONTRACT_HASH"],
    "code_sha256": os.environ["CODE_SHA256"],
    "actor_only": True,
    "actor_schema": os.environ["ACTOR_SCHEMA"],
    "editing_enabled": True,
    "judger_enabled": True,
    "a1_enabled": False,
    "editor_judge_reasoning_reward": True,
    "reward_weights": {
        "format_a0": 1.0,
        "rating0": 1.0,
        "reasoning": 1.0,
        "soft_overlong": float(os.environ["SOFT_OVERLONG_WEIGHT"]),
    },
    "soft_overlong": {
        "enabled": True,
        "max_length": integer("SOFT_OVERLONG_MAX_LENGTH"),
        "cache_length": integer("SOFT_OVERLONG_CACHE_LENGTH"),
        "penalty_start": (
            integer("SOFT_OVERLONG_MAX_LENGTH")
            - integer("SOFT_OVERLONG_CACHE_LENGTH")
        ),
        "max_penalty": float(os.environ["SOFT_OVERLONG_MAX_PENALTY"]),
        "weight": float(os.environ["SOFT_OVERLONG_WEIGHT"]),
        "hard_overlong_filter": False,
        "token_targets": ["a0.completion_non_padding"],
    },
    "policy_component": "component_grpo",
    "advantage_std": "sample_std_ddof_1",
    "advantage_epsilon": 1e-6,
    "advantage_hard_clip": None,
    "dynamic_sample": False,
    "max_generation_rounds": 1,
    "min_effective_rows": integer("GENERATION_BATCH_SIZE"),
    "low_effective_action": "not_applicable",
    "low_effective_skip_contract": None,
    "partial_effective_batch_padding": False,
    "padding_target_rows": integer("GENERATION_BATCH_SIZE"),
    "padding_group_size": integer("NUM_GENERATIONS"),
    "padding_loss_weight": 0.0,
    "padding_token_denominator_eligible": False,
    "reward_population": "same_image_six_completions",
    "rating_margin_population": "complete_rank_local_six_image_cohort",
    "reward_computed_before_effective_filter": False,
    "ineffective_groups_participate_in_reward": False,
    "epsilon_low": 0.2,
    "epsilon_high": 0.28 if os.environ["ALGORITHM"] == "dapo" else 0.2,
    "loss_normalization": "token_mean" if os.environ["ALGORITHM"] == "dapo" else "sequence_mean",
    "beta": float(os.environ["REFERENCE_ACTIVATION_BETA"]),
    "reference_activation_beta": float(os.environ["REFERENCE_ACTIVATION_BETA"]),
    "global_completion_kl_applied": (
        os.environ["COMPONENT_KL_MODE"] == "off"
        and float(os.environ["REFERENCE_ACTIVATION_BETA"]) > 0
    ),
    "kl_in_reward": False,
    "global_completion_kl": {
        "enabled": (
            os.environ["COMPONENT_KL_MODE"] == "off"
            and float(os.environ["REFERENCE_ACTIVATION_BETA"]) > 0
        ),
        "beta": float(os.environ["REFERENCE_ACTIVATION_BETA"]),
        "token_targets": ["a0.active_eligible_completion_non_padding"],
        "estimator": "sampled_k3",
        "normalization": "per_sequence_completion_token_mean_then_active_sequence_mean",
        "loss_sign": "positive_regularization",
        "kl_in_reward": False,
    },
    "component_kl": {
        "mode": os.environ["COMPONENT_KL_MODE"],
        "expected_mode": os.environ["EXPECTED_COMPONENT_KL_MODE"],
        "estimator": "sampled_k3",
        "global_completion_kl_applied": (
            os.environ["COMPONENT_KL_MODE"] == "off"
            and float(os.environ["REFERENCE_ACTIVATION_BETA"]) > 0
        ),
        "normalization": (
            "per_sequence_segment_token_mean_then_active_sequence_mean"
            if os.environ["COMPONENT_KL_MODE"] == "field"
            else None
        ),
        "loss_sign": "positive_regularization",
        "expected_reference_activation_beta": float(
            os.environ["REFERENCE_ACTIVATION_BETA"]
        ),
        "expected_reasoning_beta": float(
            os.environ["EXPECTED_BETA_KL_REASONING"]
        ),
        "expected_rating_beta": float(os.environ["EXPECTED_BETA_KL_RATING"]),
        "reference_model_path": os.environ["REFERENCE_MODEL_PATH"],
        "reference_model_tree_sha256": (
            os.environ["REFERENCE_MODEL_TREE_SHA256"] or None
        ),
        "segments": (
            {
                "reasoning": {
                    "beta": float(os.environ["BETA_KL_REASONING"]),
                    "token_targets": [
                        "a0.reasoning.evidence_content",
                        "a0.reasoning.solution_content",
                    ],
                },
                "rating0": {
                    "beta": float(os.environ["BETA_KL_RATING"]),
                    "token_targets": ["a0.rating_content"],
                },
            }
            if os.environ["COMPONENT_KL_MODE"] == "field"
            else {}
        ),
    },
    "vllm_gpu_memory_utilization": float(os.environ["VLLM_GPU_MEMORY_UTILIZATION"]),
    "vllm_sleep_level": integer("VLLM_SLEEP_LEVEL"),
    "empty_cache_before_backward": os.environ["VF_EMPTY_CACHE_BEFORE_BACKWARD"] == "1",
    "learner_activation_offload": os.environ["VF_LEARNER_ACTIVATION_OFFLOAD"] == "1",
    "learner_activation_offload_backend": (
        "torch.autograd.graph.saved_tensors_hooks.selective_cpu_v1"
        if os.environ["VF_LEARNER_ACTIVATION_OFFLOAD"] == "1"
        else None
    ),
    "learner_activation_offload_budget_gib_per_rank": integer(
        "VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB"
    ),
    "learner_activation_offload_min_tensor_mib": 16,
    "learner_activation_offload_pin_memory": False,
    "learner_activation_offload_exact_autograd": True,
    "attention_implementation": "flash_attention_2",
    "flash_attn": flash_attn.__version__,
    "flash_linear_attention": version("flash-linear-attention"),
    "causal_conv1d": version("causal-conv1d"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "python": platform.python_version(),
    "deepspeed": "zero3_cpu_offload",
    "deepspeed_config": os.environ["DEEPSPEED_CONFIG"],
    "deepspeed_config_sha256": os.environ["DEEPSPEED_CONFIG_SHA256"],
    "zero3_overlap_comm": zero3["overlap_comm"],
    "zero3_reduce_bucket_size": zero3["reduce_bucket_size"],
    "zero3_allgather_bucket_size": zero3["allgather_bucket_size"],
    "zero3_prefetch_bucket_size": zero3["stage3_prefetch_bucket_size"],
    "offload_model": True,
    "offload_optimizer": True,
    "gradient_checkpointing": os.environ["GRADIENT_CHECKPOINTING"] == "true",
    "vit_gradient_checkpointing": os.environ["VIT_GRADIENT_CHECKPOINTING"] == "true",
    "gradient_checkpointing_kwargs": {
        "use_reentrant": os.environ["GRADIENT_CHECKPOINTING_USE_REENTRANT"] == "true"
    } if os.environ["GRADIENT_CHECKPOINTING"] == "true" else None,
    "freeze_vit": os.environ["FREEZE_VIT"] == "true",
    "freeze_aligner": os.environ["FREEZE_ALIGNER"] == "true",
    "visual_trainable": os.environ["FREEZE_VIT"] == "false",
    "aligner_trainable": os.environ["FREEZE_ALIGNER"] == "false",
    "require_full_visual_no_grad_runtime": True,
    "gradient_checkpointing_scope": (
        "language_only"
        if os.environ["GRADIENT_CHECKPOINTING"] == "true"
        else "vision_configured_frozen_runtime_effective_false_language_disabled"
    ),
    "vision_gc_configured": True,
    "vision_gc_runtime_effective": False,
    "language_gc_configured": os.environ["GRADIENT_CHECKPOINTING"] == "true",
    "language_gc_runtime_effective": os.environ["GRADIENT_CHECKPOINTING"] == "true",
    "language_gc_fallback_authorized": os.environ["VF_ALLOW_LLM_GC_FALLBACK"] == "1",
    "gradient_checkpointing_scope_restore": "per_module_exact",
    "stock_swift_gc_restore_broadening_patched": True,
    "batch_decay_step": integer("BATCH_DECAY_STEP"),
    "initial_per_device_batch_size": 36,
    "batch_ladder": [36],
    "learner_microbatch_ladder": [36],
    "completion_length_ladder": [192, 160],
    "batch_decay_trigger": "forbidden",
    "oom_priority": [
        "archive_evidence",
        "exclude_service_leak_and_configuration_error",
        "completion_length_192_to_160_only",
        "bounded_activation_offload_after_archived_c160_oom",
        "language_only_non_reentrant_gc_after_archived_c160_language_oom",
    ],
    "margin_reward_scope": os.environ["MARGIN_REWARD_SCOPE"],
    "margin_images_per_cohort": integer("MARGIN_IMAGES_PER_COHORT"),
    "margin_local_images_per_rank": integer("MARGIN_LOCAL_IMAGES_PER_RANK"),
    "margin_cohorts_per_rank": (
        integer("MARGIN_LOCAL_IMAGES_PER_RANK") // integer("MARGIN_IMAGES_PER_COHORT")
        if integer("MARGIN_IMAGES_PER_COHORT") else 0
    ),
    "reward_gather_order": os.environ["REWARD_GATHER_ORDER"],
    "reward_computed_before_global_gather": os.environ["MARGIN_REWARD_SCOPE"] == "local_six_images",
    "rating_reward": "local_six_l2_margin",
    "reasoning_reward": "signed_l2_judge_delta",
    "reasoning_reward_formula": "sign(delta)*(1-exp(-delta^2/(2*tau_s)))",
    "reasoning_reward_tau_s": 1.0,
    "reasoning_reward_division_by_four": False,
    "component_credit_mask_mode": os.environ["COMPONENT_CREDIT_MASK_MODE"],
    "credit_mask_disabled": os.environ["COMPONENT_CREDIT_MASK_MODE"] == "completion",
    "component_token_targets": (
        {
            "format_a0": ["a0.completion_non_padding"],
            "rating0": ["a0.completion_non_padding"],
            "reasoning": ["a0.completion_non_padding"],
            "soft_overlong": ["a0.completion_non_padding"],
        }
        if os.environ["COMPONENT_CREDIT_MASK_MODE"] == "completion"
        else {
            "format_a0": ["a0.format"],
            "reasoning": [
                "a0.reasoning.evidence_content",
                "a0.reasoning.solution_content",
            ],
            "rating0": ["a0.rating_content"],
            "soft_overlong": ["a0.completion_non_padding"],
        }
    ),
    "original_score_cache": {
        "path": os.environ["ORIGINAL_SCORE_CACHE_PATH"],
        "sha256": os.environ["ORIGINAL_SCORE_CACHE_SHA256"],
        "read_only": True,
        "expected_row_count": integer("ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT"),
        "expected_sample_count": integer("ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT"),
        "expected_actor_ids": [
            value
            for value in os.environ["ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS"].split(",")
            if value
        ],
        "payload_schema": os.environ["ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA"],
        "rating_acceptance_range": [
            float(os.environ["ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN"]),
            float(os.environ["ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX"]),
        ],
    },
    "editor": {
        "model": "FLUX.2-klein-4B",
        "urls": os.environ["EDITOR_URLS"].split(","),
        "prompt_version": os.environ["EDITOR_PROMPT_VERSION"],
        "prompt_template_hash": os.environ["EDITOR_PROMPT_TEMPLATE_HASH"],
        "input_fields": ["solution"],
        "positive_prompt_equals_solution": True,
        "semantic_guardrail": "",
        "pairs_per_gpu": 1,
    },
    "judge": {
        "model_id": os.environ["JUDGE_MODEL_ID"],
        "model_path": os.environ["JUDGE_MODEL_PATH"],
        "model_tree_sha256": os.environ["JUDGE_MODEL_TREE_SHA256"],
        "prompt_hash": os.environ["JUDGE_PROMPT_HASH"],
        "urls": os.environ["JUDGE_URLS"].split(","),
        "deterministic": True,
        "cache_compatible": True,
        "execution_batching": {
            "max_num_seqs": integer("JUDGER_MAX_NUM_SEQS"),
            "max_batch_size": integer("JUDGER_MAX_BATCH_SIZE"),
            "batch_wait_ms": float(os.environ["JUDGER_BATCH_WAIT_MS"]),
            "service_workers": integer("EDITOR_JUDGE_SERVICE_WORKERS"),
            "scientific_contract_unchanged": True,
        },
    },
    "service_run_dir": os.environ["SERVICE_RUN_DIR"],
    "service_routing": "paired_lane_ewma_with_judge_work_stealing",
    "optimizer": "adamw_torch",
    "cpu_optimizer_implementation": "pytorch_adamw",
    "learning_rate": 1e-6,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "seed": 42,
    "gpu_total_gib": integer("GPU_TOTAL_MIB") / 1024.0,
}
path = pathlib.Path(__import__("sys").argv[1])
path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2, sort_keys=True))
PY

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export min_pixels="${MIN_PIXELS}"
export FLA_TILELANG=0
export CUDA_LAUNCH_BLOCKING=1
export VF_SAFE_BACKWARD_MODE=off
export VF_SYNC_EACH_BACKWARD_CHUNK=1
export VF_EMPTY_CACHE_BEFORE_BACKWARD
export VF_LEARNER_ACTIVATION_OFFLOAD
export VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB
export VF_LEARNER_BACKWARD_MODE=branch
export VF_LEARNER_MICROBATCH_SIZE="${LEARNER_MICROBATCH_SIZE}"
export VF_EXPECTED_NUM_ITERATIONS="${NUM_ITERATIONS}"
export VF_EXPECTED_WORLD_SIZE="${WORLD_SIZE}"
export VF_TOTAL_TRAIN_EPOCHS="${TOTAL_TRAIN_EPOCHS}"
export VF_REQUIRE_FULL_VISUAL_FROZEN=1
if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
  export VF_EXPECT_LANGUAGE_GC=1
else
  export VF_EXPECT_LANGUAGE_GC=0
fi
export VF_VISION_GC_CONFIGURED=1
export VF_ACTOR_ONLY=1
export VF_DAPO_ENABLED=0
export VF_SCALAR_GRPO_ENABLED=0
export VF_EDITOR_JUDGE_REASONING_REWARD=1
export VF_DAPO_MAX_GENERATION_ROUNDS
export VF_DAPO_MIN_EFFECTIVE_ROWS
export VF_DAPO_GROUP_EPSILON=1e-6
export VF_GRPO_GROUP_EPSILON=1e-6
export VF_DAPO_SOFT_MAX_LENGTH=256
export VF_DAPO_SOFT_CACHE_LENGTH=64
export VF_DAPO_OVERLONG_WEIGHT="$([[ "${ALGORITHM}" == "dapo" ]] && echo 1.0 || echo 0.0)"
export VF_SOFT_OVERLONG_MAX_LENGTH="${SOFT_OVERLONG_MAX_LENGTH}"
export VF_SOFT_OVERLONG_CACHE_LENGTH="${SOFT_OVERLONG_CACHE_LENGTH}"
export VF_SOFT_OVERLONG_MAX_PENALTY="${SOFT_OVERLONG_MAX_PENALTY}"
export VF_SOFT_OVERLONG_WEIGHT="${SOFT_OVERLONG_WEIGHT}"
export VF_ACTOR_SCHEMA="${ACTOR_SCHEMA}"
export VF_REQUIRE_GLOBAL_MARGIN_GATHER=1
export VF_MARGIN_REWARD_SCOPE="${MARGIN_REWARD_SCOPE}"
export VF_MARGIN_IMAGES_PER_COHORT="${MARGIN_IMAGES_PER_COHORT}"
export VF_MARGIN_LOCAL_IMAGES_PER_RANK="${MARGIN_LOCAL_IMAGES_PER_RANK}"
export VF_REWARD_GATHER_ORDER="${REWARD_GATHER_ORDER}"
export VF_ALLOW_BATCH_PROBE=0
export IMAGE_EDIT_BACKEND=diffusers
export VF_LOOP_ENABLE_COMFY=0
export VF_LOOP_ENABLE_JUDGER=1
export DIFFUSERS_SERVERS="${EDITOR_URLS}"
export VF_LOOP_JUDGER_URLS="${JUDGE_URLS}"
export VF_PROJECT_ROOT="${PACKAGE_ROOT}"
export VF_LOOP_SERVICE_TIMEOUT=900
export VF_EDITOR_JUDGE_SERVICE_WORKERS="${EDITOR_JUDGE_SERVICE_WORKERS}"
export VF_JUDGER_MAX_NUM_SEQS="${JUDGER_MAX_NUM_SEQS}"
export VF_JUDGER_MAX_BATCH_SIZE="${JUDGER_MAX_BATCH_SIZE}"
export VF_JUDGER_BATCH_WAIT_MS="${JUDGER_BATCH_WAIT_MS}"
export VF_SERVICE_MAX_ATTEMPTS=3
export VF_SERVICE_EWMA_ALPHA=0.2
export VF_JUDGE_WORK_STEAL_RATIO=1.25
export VF_ORIGINAL_SCORE_CACHE_PATH="${ORIGINAL_SCORE_CACHE_PATH}"
export VF_ORIGINAL_SCORE_CACHE_SHA256="${ORIGINAL_SCORE_CACHE_SHA256}"
export VF_ORIGINAL_SCORE_CACHE_VERIFY_SHA256=1
export VF_ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT="${ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT}"
export VF_ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT="${ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT}"
export VF_ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS="${ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS}"
export VF_ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA="${ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA}"
export VF_ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN="${ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN}"
export VF_ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX="${ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX}"
export VF_JUDGE_MODEL_ID="${JUDGE_MODEL_ID}"
export VF_JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH}"
export VF_JUDGE_MODEL_TREE_SHA256="${JUDGE_MODEL_TREE_SHA256}"
export VF_JUDGE_PROMPT_HASH="${JUDGE_PROMPT_HASH}"
export VF_COMPONENT_CREDIT_MASK_MODE="${COMPONENT_CREDIT_MASK_MODE}"
export VF_EXPECT_COMPONENT_CREDIT_MASK_MODE="${EXPECTED_COMPONENT_CREDIT_MASK_MODE}"
export VF_COMPONENT_KL_MODE="${COMPONENT_KL_MODE}"
export VF_EXPECT_COMPONENT_KL_MODE="${EXPECTED_COMPONENT_KL_MODE}"
export VF_BETA_KL_REASONING="${BETA_KL_REASONING}"
export VF_BETA_KL_RATING="${BETA_KL_RATING}"
export VF_EXPECT_BETA_KL_REASONING="${EXPECTED_BETA_KL_REASONING}"
export VF_EXPECT_BETA_KL_RATING="${EXPECTED_BETA_KL_RATING}"
export VF_REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH}"
export VF_REFERENCE_MODEL_TREE_SHA256="${REFERENCE_MODEL_TREE_SHA256}"
export VF_REASONING_REWARD_TAU_S=1.0
export VF_PRESENCE_PENALTY=1.5
export VF_REPETITION_PENALTY=1.0
export VF_WEIGHT_FORMAT_A0=1.0
export VF_WEIGHT_RATING0=1.0
export VF_WEIGHT_REASONING=1.0
export VF_WEIGHT_FORMAT_A1=0.0
export VF_WEIGHT_RATING1_ANCHOR=0.0
export VF_WEIGHT_EDIT_GATE=0.0
export VF_WEIGHT_EDIT_GAIN=0.0
export VF_WEIGHT_DELTA_MARGIN=0.0
export VF_RUN_ID="${EXPERIMENT_ID}_epoch${TARGET_EPOCH}"
export VF_LOOP_TRAJECTORY_LOG="${ARTIFACT_DIR}/trajectory.jsonl"
export VLLM_USE_FLASHINFER_SAMPLER=0
export WANDB_PROJECT="${WANDB_PROJECT:-mr-iqa-grpo-editor-judge}"
export WANDB_NAME="${EXPERIMENT_ID}"
export WANDB_RUN_GROUP="${EXPERIMENT_ID}"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
mkdir -p "${WANDB_DIR}"

TRAIN_CONTROL_ARGS=()
REPORT_TO=none
if [[ "${MODE}" == "formal" ]]; then
  export VF_STOP_AFTER_EPOCH="${TARGET_EPOCH}"
  unset VF_STOP_AFTER_STEP
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_RUN_ID
  export WANDB_RESUME=allow
  REPORT_TO=wandb
  TRAIN_CONTROL_ARGS+=(--num_train_epochs "${TOTAL_TRAIN_EPOCHS}" --save_strategy epoch --save_only_model false --save_total_limit 1)
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    TRAIN_CONTROL_ARGS+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
  fi
elif [[ "${MODE}" == "steps" ]]; then
  unset VF_STOP_AFTER_EPOCH
  export VF_STOP_AFTER_STEP="${STOP_AFTER_STEP}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_RUN_ID
  export WANDB_RESUME=allow
  REPORT_TO=wandb
  TRAIN_CONTROL_ARGS+=(
    --max_steps "${TRAIN_MAX_STEPS}"
    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
    --save_only_model false
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
  )
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    TRAIN_CONTROL_ARGS+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
  fi
else
  unset VF_STOP_AFTER_EPOCH
  unset VF_STOP_AFTER_STEP
  export WANDB_MODE=disabled
  TRAIN_CONTROL_ARGS+=(--max_steps "${SMOKE_MAX_STEPS}" --save_strategy no)
  EXPECTED_START_STEP=0
  EXPECTED_END_STEP="${SMOKE_MAX_STEPS}"
fi

PREFLIGHT_MODEL="${MODEL_PATH}"
PREFLIGHT_PARENT_ARGS=()
PREFLIGHT_MODE_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" && "${MODE}" != "steps" ]]; then
  PREFLIGHT_MODEL="${RESUME_CHECKPOINT}"
  PREFLIGHT_PARENT_ARGS+=(--promoted-parent-manifest "${PARENT_MANIFEST}")
fi
if [[ "${MODE}" == "smoke" ]]; then
  PREFLIGHT_MODE_ARGS+=(--smoke)
fi
python3 "${CODE_ROOT}/scripts/formal_preflight.py" \
  --model "${PREFLIGHT_MODEL}" \
  --approved-initial "${MODEL_PATH}" \
  "${PREFLIGHT_PARENT_ARGS[@]}" \
  --dataset-file "${DATASET_FILE}" \
  --retained-source "${RETAINED_DATASET_SOURCE}" \
  --service-run-dir "${SERVICE_RUN_DIR}" \
  --editor-judge-component-grpo \
  "${PREFLIGHT_MODE_ARGS[@]}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --num-generations "${NUM_GENERATIONS}" \
  --world-size "${WORLD_SIZE}" \
  --num-iterations "${NUM_ITERATIONS}" \
  --learner-microbatch-size "${LEARNER_MICROBATCH_SIZE}" \
  --require-global-margin-gather \
  --learner-backward-mode branch \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --max-length "${MAX_LENGTH}" \
  --max-pixels "${MAX_PIXELS}" \
  --learning-rate 1e-6 \
  --beta "${REFERENCE_ACTIVATION_BETA}" \
  --temperature 0.7 \
  --top-p 1.0 \
  --top-k 20 \
  --repetition-penalty 1.0 \
  --presence-penalty 1.5 \
  --seed 42 \
  --num-train-epochs "${TOTAL_TRAIN_EPOCHS}" \
  --freeze-vit \
  --freeze-aligner \
  --output "${ARTIFACT_DIR}/formal_preflight.json" \
  | tee "${LOG_DIR}/formal_preflight.log"

ALGORITHM_ARGS=()
if [[ "${ALGORITHM}" == "dapo" ]]; then
  ALGORITHM_ARGS+=(
    --loss_type dapo --epsilon 0.2 --epsilon_high 0.28
    --dynamic_sample true --max_resample_times 3
    --soft_max_length 256 --soft_cache_length 64
  )
else
  ALGORITHM_ARGS+=(
    --loss_type grpo --epsilon 0.2 --epsilon_high 0.2
    --dynamic_sample false
  )
fi

GRADIENT_CHECKPOINTING_ARGS=(
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --vit_gradient_checkpointing "${VIT_GRADIENT_CHECKPOINTING}"
)
if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
  GRADIENT_CHECKPOINTING_ARGS+=(
    --gradient_checkpointing_kwargs "{\"use_reentrant\": ${GRADIENT_CHECKPOINTING_USE_REENTRANT}}"
  )
fi

ARGS=(
  --rlhf_type grpo
  --model "${MODEL_PATH}"
  --tuner_type full
  --torch_dtype bfloat16
  --attn_impl flash_attention_2
  --freeze_vit "${FREEZE_VIT}"
  --freeze_aligner "${FREEZE_ALIGNER}"
  --dataset "${DATASET_FILE}"
  --external_plugins "${CODE_ROOT}/plugin/vf_dual_rollout_trainer.py"
  --reward_funcs vf_dual_rollout_placeholder
  --num_generations "${NUM_GENERATIONS}"
  --num_iterations "${NUM_ITERATIONS}"
  --generation_batch_size "${GENERATION_BATCH_SIZE}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --gradient_accumulation_steps 1
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --max_length "${MAX_LENGTH}"
  --max_pixels "${MAX_PIXELS}"
  --learning_rate 1e-6
  --lr_scheduler_type cosine
  --optim adamw_torch
  --weight_decay 0.1
  --max_grad_norm 1.0
  --beta "${REFERENCE_ACTIVATION_BETA}"
  --kl_in_reward false
  --scale_rewards group
  --importance_sampling_level token
  --rollout_importance_sampling_mode token_truncate
  --rollout_importance_sampling_threshold 2.0
  --log_rollout_offpolicy_metrics true
  --use_vllm true
  --vllm_mode colocate
  --vllm_tensor_parallel_size "${WORLD_SIZE}"
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --vllm_max_model_len "${MAX_LENGTH}"
  --vllm_enforce_eager true
  --sleep_level "${VLLM_SLEEP_LEVEL}"
  --offload_model true
  --offload_optimizer true
  --move_model_batches 4
  --deepspeed "${DEEPSPEED_CONFIG}"
  "${GRADIENT_CHECKPOINTING_ARGS[@]}"
  --overlong_filter false
  --truncation_strategy delete
  --enable_thinking false
  --add_non_thinking_prefix false
  --temperature 0.7
  --top_p 1.0
  --top_k 20
  --repetition_penalty 1.0
  --seed 42
  --data_seed 42
  --dataloader_drop_last true
  --ignore_data_skip false
  --eval_strategy no
  --logging_steps 1
  --log_completions true
  --report_to "${REPORT_TO}"
  --output_dir "${TRAIN_DIR}"
  "${ALGORITHM_ARGS[@]}"
  "${TRAIN_CONTROL_ARGS[@]}"
)

printf '%q ' swift rlhf "${ARGS[@]}" >"${ARTIFACT_DIR}/launch_command.txt"
printf '\n' >>"${ARTIFACT_DIR}/launch_command.txt"

TELEMETRY_FILE="${LOG_DIR}/gpu_telemetry.csv"
(
  echo "timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu_pct"
  while true; do
    timestamp="$(date --iso-8601=seconds)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | sed "s/^/${timestamp},/"
    sleep 2
  done
) >"${TELEMETRY_FILE}" &
TELEMETRY_PID=$!
trap 'kill "${TELEMETRY_PID}" 2>/dev/null || true' EXIT

echo "[editor-judge-grpo] mode=${MODE} experiment=${EXPERIMENT_ID} epoch=${TARGET_EPOCH} b/mb=${PER_DEVICE_BATCH_SIZE}/${LEARNER_MICROBATCH_SIZE} world=${WORLD_SIZE} global=${GENERATION_BATCH_SIZE} i${NUM_ITERATIONS}/g${NUM_GENERATIONS} c${MAX_COMPLETION_LENGTH} llm_gc=${GRADIENT_CHECKPOINTING} vision_gc_configured=${VIT_GRADIENT_CHECKPOINTING} activation_offload=${VF_LEARNER_ACTIVATION_OFFLOAD} activation_offload_budget_gib=${VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB}"
START_NS="$(date +%s%N)"
set +e
swift rlhf "${ARGS[@]}" 2>&1 | tee "${LOG_DIR}/train.log"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e
END_NS="$(date +%s%N)"
WALL_SECONDS="$(python3 -c "print((${END_NS} - ${START_NS}) / 1e9)")"
kill "${TELEMETRY_PID}" 2>/dev/null || true
wait "${TELEMETRY_PID}" 2>/dev/null || true
trap - EXIT
printf '%s\n' "${TRAIN_STATUS}" >"${ARTIFACT_DIR}/trainer_exit_code.txt"
printf '%s\n' "${WALL_SECONDS}" >"${ARTIFACT_DIR}/wall_seconds.txt"

set +e
python3 "${CODE_ROOT}/scripts/audit_comparison_stage.py" \
  --run-dir "${RUN_DIR}" \
  --trainer-exit-code "${TRAIN_STATUS}" \
  --wall-seconds "${WALL_SECONDS}" \
  --expected-start-step "${EXPECTED_START_STEP}" \
  --expected-end-step "${EXPECTED_END_STEP}" | tee "${LOG_DIR}/audit.log"
AUDIT_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  exit "${TRAIN_STATUS}"
fi
exit "${AUDIT_STATUS}"
