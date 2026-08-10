#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-validation}"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"

if [[ "${MODE}" == "--print-plan" ]]; then
  printf '%s\n' \
    "validation: 200 rows" \
    "test: koniq, spaq_full, livew, kadid_full, agiqa3k, csiq" \
    "all: validation plus all six test datasets" \
    "sequence: 8-GPU Actor -> 8-GPU Editor barrier -> 8-GPU E5 Judge"
  exit 0
fi

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

case "${MODE}" in
  validation)
    labels=(validation)
    counts=(200)
    dataset_contract='{"validation":200}'
    ;;
  test)
    labels=(koniq spaq_full livew kadid_full agiqa3k csiq)
    counts=(2010 11125 1162 10125 2982 866)
    dataset_contract='{"koniq":2010,"spaq_full":11125,"livew":1162,"kadid_full":10125,"agiqa3k":2982,"csiq":866}'
    ;;
  all)
    labels=(validation koniq spaq_full livew kadid_full agiqa3k csiq)
    counts=(200 2010 11125 1162 10125 2982 866)
    dataset_contract='{"validation":200,"koniq":2010,"spaq_full":11125,"livew":1162,"kadid_full":10125,"agiqa3k":2982,"csiq":866}'
    ;;
  *)
    echo "usage: $0 [validation|test|all|--print-plan]" >&2
    exit 2
    ;;
esac

: "${ACTOR_MODEL_PATH:?ACTOR_MODEL_PATH is required}"
: "${EVAL_IMAGE_ROOT:?EVAL_IMAGE_ROOT is required}"

ACTOR_PYTHON="${ACTOR_PYTHON:-python}"
ACTOR_PROCESSOR_PATH="${ACTOR_PROCESSOR_PATH:-${ACTOR_MODEL_PATH}}"
EVAL_NAME="${EVAL_NAME:-actor}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${ROOT}/outputs/evaluation}"
EVAL_ROOT="${EVAL_ROOT:-${EVAL_OUTPUT_ROOT}/${EVAL_NAME}/${MODE}}"
EVAL_ACTOR_ONLY="${EVAL_ACTOR_ONLY:-0}"
EVAL_RESUME="${EVAL_RESUME:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-64}"
EVAL_RETRIES="${EVAL_RETRIES:-3}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-100}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
ACTOR_LAUNCHER="${ROOT}/actor/scripts/eval_reasons_rating_actor_8shard.sh"
OFFLINE_RUNNER="${ROOT}/actor/scripts/run_actor_outputs_editor_judge.py"
SERVICE_STACK="${ROOT}/actor/scripts/resident_service_stack.sh"
CHECKPOINT_TOOL="${ROOT}/actor/scripts/checkpoint_manifest.py"

[[ -x "${ACTOR_LAUNCHER}" && -x "${SERVICE_STACK}" ]] || {
  echo "evaluation launcher or service stack is not executable" >&2
  exit 2
}
[[ -d "${ACTOR_MODEL_PATH}" ]] || { echo "missing Actor model directory" >&2; exit 2; }
[[ -d "${EVAL_IMAGE_ROOT}" ]] || { echo "missing evaluation image root" >&2; exit 2; }
command -v "${ACTOR_PYTHON}" >/dev/null 2>&1 || {
  echo "Actor Python is not executable: ${ACTOR_PYTHON}" >&2
  exit 2
}

# The validation identity belongs to the checkpoint being evaluated.  It is
# deliberately computed over the ten public inference-export files and never
# inherited from ACTOR_MODEL_TREE_SHA256 (which identifies the initial model in
# training profiles and may include mutable trainer state).
actor_export_identity="$(${ACTOR_PYTHON} "${CHECKPOINT_TOOL}" inspect-export --checkpoint "${ACTOR_MODEL_PATH}")"
actor_export_digest="$(${ACTOR_PYTHON} - "${actor_export_identity}" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["selected_export_tree_sha256"])
PY
)"
if [[ -n "${ACTOR_MODEL_EXPORT_TREE_SHA256:-}" && "${ACTOR_MODEL_EXPORT_TREE_SHA256}" != "${actor_export_digest}" ]]; then
  echo "Actor inference-export SHA256 mismatch" >&2
  exit 2
fi
export ACTOR_MODEL_EXPORT_TREE_SHA256="${actor_export_digest}"

mkdir -p "${EVAL_ROOT}/actor_outputs" "${EVAL_ROOT}/logs" "${EVAL_ROOT}/state"
exec > >(tee -a "${EVAL_ROOT}/logs/driver.log") 2>&1

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
      return
    fi
    printf '%s\n' "waiting for all eight GPUs" "${snapshot}"
    sleep "${GPU_POLL_SECONDS}"
  done
}

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  expected="${counts[$index]}"
  data_file="${ROOT}/data/${label}.jsonl"
  [[ -f "${data_file}" ]] || { echo "missing data file: ${data_file}" >&2; exit 2; }
  actual="$(wc -l <"${data_file}" | tr -d ' ')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label}: expected ${expected} rows, found ${actual}" >&2
    exit 2
  }
done

export VF_ACTOR_SCHEMA=reasoning_evidence_solution_rating
export VALIDATION_GPUS=0,1,2,3,4,5,6,7
export VALIDATION_MAX_NEW_TOKENS=160
export VALIDATION_BATCH_SIZE="${EVAL_BATCH_SIZE}"
export VALIDATION_MAX_PIXELS=196608
export VALIDATION_MIN_PIXELS=3136
export VALIDATION_MAX_MODEL_LEN=2048
export VALIDATION_VLLM_GPU_MEMORY_UTILIZATION=0.24
export VALIDATION_TOP_K=20
export VALIDATION_PRESENCE_PENALTY=1.5
export VALIDATION_SEED=42
export PYTHON_BIN="${ACTOR_PYTHON}"

wait_for_all_gpus
for label in "${labels[@]}"; do
  output_dir="${EVAL_ROOT}/actor_outputs/${label}"
  if [[ "${EVAL_RESUME}" == "1" && -f "${output_dir}/merged.json" ]]; then
    echo "reusing Actor output: ${output_dir}/merged.json"
    continue
  fi
  bash "${ACTOR_LAUNCHER}" \
    "${ACTOR_MODEL_PATH}" \
    "${ROOT}/data/${label}.jsonl" \
    "${EVAL_IMAGE_ROOT}" \
    "${output_dir}" \
    "${ACTOR_PROCESSOR_PATH}"
done

if [[ "${EVAL_ACTOR_ONLY}" == "1" ]]; then
  echo "Actor-only evaluation complete: ${EVAL_ROOT}"
  exit 0
fi
[[ "${EVAL_ACTOR_ONLY}" == "0" ]] || { echo "EVAL_ACTOR_ONLY must be 0 or 1" >&2; exit 2; }

required_full_env=(
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
)
for name in "${required_full_env[@]}"; do
  [[ -n "${!name:-}" ]] || {
    echo "full evaluation requires environment variable: ${name}" >&2
    exit 2
  }
done

[[ -d "${DIFFUSERS_MODEL_PATH}" ]] || { echo "missing Editor model directory" >&2; exit 2; }
[[ -d "${JUDGE_MODEL_PATH}" ]] || { echo "missing Judge model directory" >&2; exit 2; }
[[ -f "${JUDGE_MANIFEST_PATH}" ]] || { echo "missing Judge manifest" >&2; exit 2; }
[[ -f "${ORIGINAL_SCORE_CACHE_PATH}" ]] || { echo "missing original-score cache" >&2; exit 2; }
[[ "$(sha256sum "${ORIGINAL_SCORE_CACHE_PATH}" | awk '{print $1}')" == "${ORIGINAL_SCORE_CACHE_SHA256}" ]] || {
  echo "original-score cache hash mismatch" >&2
  exit 2
}

export VF_PROJECT_ROOT="${ROOT}"
export IMAGE_EDIT_BACKEND=diffusers
export SERVICE_GPUS=0,1,2,3,4,5,6,7
export SERVICE_EDITOR_PORTS=8212,8213,8214,8215,8216,8217,8218,8219
export SERVICE_JUDGER_PORTS=8204,8205,8206,8207,8208,8209,8210,8211
export VF_OFFLINE_EDITOR_URLS=http://127.0.0.1:8212,http://127.0.0.1:8213,http://127.0.0.1:8214,http://127.0.0.1:8215,http://127.0.0.1:8216,http://127.0.0.1:8217,http://127.0.0.1:8218,http://127.0.0.1:8219
export VF_OFFLINE_JUDGE_URLS=http://127.0.0.1:8204,http://127.0.0.1:8205,http://127.0.0.1:8206,http://127.0.0.1:8207,http://127.0.0.1:8208,http://127.0.0.1:8209,http://127.0.0.1:8210,http://127.0.0.1:8211
export VF_OFFLINE_JUDGE_GPU_INDICES=0,1,2,3,4,5,6,7
export VF_OFFLINE_MAX_JUDGE_INSTANCES_PER_GPU=1
export VF_OFFLINE_DATASETS_JSON="${dataset_contract}"
export JUDGER_MODEL_ID="${JUDGE_MODEL_ID:-source-e5-judge-step725}"
export JUDGER_MODEL_PATH="${JUDGE_MODEL_PATH}"
export JUDGER_MANIFEST_PATH="${JUDGE_MANIFEST_PATH}"
export JUDGER_MODEL_TREE_SHA256="${JUDGE_MODEL_TREE_SHA256}"
export JUDGER_MODEL_EXPORT_TREE_SHA256="${JUDGE_MODEL_EXPORT_TREE_SHA256}"
export JUDGER_PROMPT_SCHEMA="${JUDGER_PROMPT_SCHEMA:-e5_training_reasoning_v5}"
export JUDGER_PROMPT_HASH="${JUDGE_PROMPT_HASH}"
export JUDGER_BACKEND="${JUDGER_BACKEND:-e5_qwen35_4b_vllm_judge}"
export VF_JUDGE_MODEL_ID="${JUDGER_MODEL_ID}"
export VF_JUDGE_MODEL_PATH="${JUDGER_MODEL_PATH}"
export VF_JUDGE_MODEL_TREE_SHA256="${JUDGER_MODEL_TREE_SHA256}"
export VF_JUDGE_PROMPT_SCHEMA="${JUDGER_PROMPT_SCHEMA}"
export VF_JUDGE_PROMPT_HASH="${JUDGER_PROMPT_HASH}"
export VF_JUDGER_MAX_NUM_SEQS=1
export VF_JUDGER_MAX_BATCH_SIZE=1
export VF_JUDGER_BATCH_WAIT_MS=0
export DIFFUSERS_OUTPUT_ROOT="${EVAL_ROOT}/editor_outputs"
if [[ -n "${ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA:-}" ]]; then
  export VF_ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA="${ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA}"
fi

if [[ -z "${DIFFUSERS_WARMUP_IMAGE:-}" ]]; then
  first_image="$("${ACTOR_PYTHON}" - "${ROOT}/data/${labels[0]}.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.loads(next(line for line in handle if line.strip()))["image"])
PY
)"
  if [[ "${first_image}" = /* ]]; then
    export DIFFUSERS_WARMUP_IMAGE="${first_image}"
  else
    export DIFFUSERS_WARMUP_IMAGE="${EVAL_IMAGE_ROOT}/${first_image}"
  fi
fi
[[ -f "${DIFFUSERS_WARMUP_IMAGE}" ]] || {
  echo "Editor warmup image is missing: ${DIFFUSERS_WARMUP_IMAGE}" >&2
  exit 2
}

"${ACTOR_PYTHON}" - \
  "${EVAL_ROOT}/contract.json" \
  "${ACTOR_MODEL_PATH}" \
  "${JUDGER_MODEL_ID}" \
  "${JUDGER_MODEL_PATH}" \
  "${JUDGER_MODEL_TREE_SHA256}" \
  "${JUDGER_PROMPT_HASH}" \
  "${dataset_contract}" \
  "${ACTOR_MODEL_EXPORT_TREE_SHA256}" \
  "$(sha256sum "${ROOT}/data/validation.jsonl" | awk '{print $1}')" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "mr_iqa_evaluation_contract_v1",
    "status": "active",
    "epoch2_release_allowed": False,
    "actor_model_path": sys.argv[2],
    "checkpoint": sys.argv[2],
    "checkpoint_export_tree_sha256": sys.argv[8],
    "checkpoint_digest": {
        "semantics": "selected_inference_export",
        "algorithm": (
            "sha256(sorted(relative_path + NUL + decimal_size + NUL + "
            "file_sha256 + LF)) over the 10 allowlisted inference-export files"
        ),
        "sha256": sys.argv[8],
        "file_count": 10,
    },
    "validation_manifest_sha256": sys.argv[9],
    "datasets": json.loads(sys.argv[7]),
    "editor_judge": {
        "judge_model_id": sys.argv[3],
        "judge_model_path": sys.argv[4],
        "judge_model_tree_sha256": sys.argv[5],
        "judge_prompt_hash": sys.argv[6],
        "sequence": ["actor", "editor_barrier", "judge"],
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

wait_for_all_gpus
"${ACTOR_PYTHON}" "${OFFLINE_RUNNER}" validate-config \
  --eval-root "${EVAL_ROOT}" \
  --cache-path "${ORIGINAL_SCORE_CACHE_PATH}"
"${ACTOR_PYTHON}" "${OFFLINE_RUNNER}" prepare-actor-index \
  --eval-root "${EVAL_ROOT}" \
  --cache-path "${ORIGINAL_SCORE_CACHE_PATH}"
bash "${SERVICE_STACK}" --validate-config

# The runner completes every edit before unloading Editor and starting Judge.
"${ACTOR_PYTHON}" "${OFFLINE_RUNNER}" run-editor-judge \
  --eval-root "${EVAL_ROOT}" \
  --cache-path "${ORIGINAL_SCORE_CACHE_PATH}" \
  --workers "${EVAL_WORKERS}" \
  --retries "${EVAL_RETRIES}"

echo "evaluation complete: ${EVAL_ROOT}"
