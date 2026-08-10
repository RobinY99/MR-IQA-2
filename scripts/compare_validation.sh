#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

: "${NATIVE_ACTOR_MODEL_PATH:?NATIVE_ACTOR_MODEL_PATH is required}"
: "${NO_MASK_ACTOR_MODEL_PATH:?NO_MASK_ACTOR_MODEL_PATH is required}"
: "${MASK_ACTOR_MODEL_PATH:?MASK_ACTOR_MODEL_PATH is required}"

for specification in \
  "native:${NATIVE_ACTOR_MODEL_PATH}" \
  "no_mask_step30:${NO_MASK_ACTOR_MODEL_PATH}" \
  "mask_step30:${MASK_ACTOR_MODEL_PATH}"; do
  name="${specification%%:*}"
  model="${specification#*:}"
  env \
    ENV_FILE=/dev/null \
    EVAL_NAME="${name}" \
    ACTOR_MODEL_PATH="${model}" \
    ACTOR_PROCESSOR_PATH="${NATIVE_ACTOR_MODEL_PATH}" \
    "${ROOT}/scripts/evaluate.sh" validation
done

if [[ "${EVAL_ACTOR_ONLY:-0}" == "0" ]]; then
  summary="${EVAL_OUTPUT_ROOT:-${ROOT}/outputs/evaluation}/validation_suite_summary.json"
  arguments=(
    --stage-root "${EVAL_OUTPUT_ROOT:-${ROOT}/outputs/evaluation}"
    --target "native=${EVAL_OUTPUT_ROOT:-${ROOT}/outputs/evaluation}/native/validation"
    --target "no_mask_step30=${EVAL_OUTPUT_ROOT:-${ROOT}/outputs/evaluation}/no_mask_step30/validation"
    --target "mask_step30=${EVAL_OUTPUT_ROOT:-${ROOT}/outputs/evaluation}/mask_step30/validation"
    --output "${summary}"
  )
  if [[ "${VALIDATION_LOG_WANDB:-0}" == "1" ]]; then
    arguments+=(--log-wandb)
  fi
  "${ACTOR_PYTHON:-python}" "${ROOT}/actor/scripts/summarize_validation.py" "${arguments[@]}"
  echo "validation comparison: ${summary}"
fi
