#!/usr/bin/env bash
set -euo pipefail

NUM_SHARDS=8
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VALIDATION_GPUS="${VALIDATION_GPUS:-0,1,2,3,4,5,6,7}"
MAX_NEW_TOKENS="${VALIDATION_MAX_NEW_TOKENS:-256}"
BATCH_SIZE="${VALIDATION_BATCH_SIZE:-8}"
MAX_PIXELS="${VALIDATION_MAX_PIXELS:-196608}"
MIN_PIXELS="${VALIDATION_MIN_PIXELS:-3136}"
MAX_MODEL_LEN="${VALIDATION_MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${VALIDATION_VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_PORT_BASE="${VALIDATION_VLLM_PORT_BASE:-40000}"
VLLM_PORT_STRIDE="${VALIDATION_VLLM_PORT_STRIDE:-1024}"
TOP_K="${VALIDATION_TOP_K:-20}"
PRESENCE_PENALTY="${VALIDATION_PRESENCE_PENALTY:-1.5}"
SEED="${VALIDATION_SEED:-42}"

if [[ -z "${VF_ACTOR_SCHEMA:-}" ]]; then
  export VF_ACTOR_SCHEMA="reasons_rating"
else
  export VF_ACTOR_SCHEMA
fi
IFS=',' read -r -a GPU_IDS <<< "${VALIDATION_GPUS}"
[[ "${VLLM_PORT_BASE}" =~ ^[0-9]+$ && "${VLLM_PORT_STRIDE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "validation vLLM port base and stride must be positive integers" >&2
  exit 2
}
((VLLM_PORT_BASE >= 1024 && VLLM_PORT_BASE + (NUM_SHARDS - 1) * VLLM_PORT_STRIDE <= 65535)) || {
  echo "validation vLLM shard port ranges exceed the usable TCP port range" >&2
  exit 2
}

if [[ "${1:-}" == "--validate-config" ]]; then
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.22}"
  [[ "${#GPU_IDS[@]}" -eq "${NUM_SHARDS}" ]] || {
    echo "validation requires exactly ${NUM_SHARDS} GPU ids" >&2
    exit 2
  }
  printf 'validation config accepted: protocol=%s shards=%s gpus=%s schema=%s max_new_tokens=%s batch=%s max_pixels=%s vllm_util=%s vllm_port_base=%s vllm_port_stride=%s\n' \
    "vf_training_aligned_vllm_v1_20260718" \
    "${NUM_SHARDS}" "${VALIDATION_GPUS}" "${VF_ACTOR_SCHEMA}" \
    "${MAX_NEW_TOKENS}" "${BATCH_SIZE}" "${MAX_PIXELS}" "${GPU_MEMORY_UTILIZATION}" \
    "${VLLM_PORT_BASE}" "${VLLM_PORT_STRIDE}"
  exit 0
fi

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "usage: $0 MODEL DATA_FILE IMAGE_ROOT OUTPUT_DIR [PROCESSOR]" >&2
  exit 2
fi

MODEL="$1"
DATA_FILE="$2"
IMAGE_ROOT="$3"
OUTPUT_DIR="$4"
PROCESSOR="${5:-$MODEL}"
if [[ -z "${GPU_MEMORY_UTILIZATION}" ]]; then
  GPU_MEMORY_UTILIZATION=0.22
  if [[ "${PROCESSOR,,}" == *"qwen3.5-4b"* ]]; then
    GPU_MEMORY_UTILIZATION=0.24
  fi
fi
mkdir -p "${OUTPUT_DIR}"

source "${ROOT_DIR}/scripts/vllm_cuda13_runtime.sh"
vllm_prepare_cuda13_runtime
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PIDS=()
SHARD_OUTPUTS=()
for ((SHARD_ID=0; SHARD_ID<NUM_SHARDS; SHARD_ID++)); do
  SHARD_OUTPUT="${OUTPUT_DIR}/shard_${SHARD_ID}.json"
  SHARD_VLLM_PORT="$((VLLM_PORT_BASE + SHARD_ID * VLLM_PORT_STRIDE))"
  SHARD_OUTPUTS+=("${SHARD_OUTPUT}")
  VLLM_PORT="${SHARD_VLLM_PORT}" \
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$SHARD_ID]}" "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/eval_training_aligned_actor.py" \
    --model_name_or_path "${MODEL}" \
    --processor_name_or_path "${PROCESSOR}" \
    --data_file "${DATA_FILE}" \
    --image_root "${IMAGE_ROOT}" \
    --output_json "${SHARD_OUTPUT}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_id "${SHARD_ID}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.0 \
    --top_p 1.0 \
    --top_k "${TOP_K}" \
    --repetition_penalty 1.0 \
    --presence_penalty "${PRESENCE_PENALTY}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --max_pixels "${MAX_PIXELS}" \
    --min_pixels "${MIN_PIXELS}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    STATUS=1
  fi
done
[[ "${STATUS}" -eq 0 ]] || exit "${STATUS}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_training_aligned_actor.py" \
  --merge_inputs "${SHARD_OUTPUTS[@]}" \
  --output_json "${OUTPUT_DIR}/merged.json"
