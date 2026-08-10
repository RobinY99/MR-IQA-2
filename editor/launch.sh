#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${DIFFUSERS_VENV:?DIFFUSERS_VENV is required}"
MODEL_PATH="${DIFFUSERS_MODEL_PATH:?DIFFUSERS_MODEL_PATH is required}"
OUTPUT_ROOT="${DIFFUSERS_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/editor/images}"
LOG_ROOT="${DIFFUSERS_LOG_ROOT:-${PROJECT_ROOT}/outputs/editor/logs}"
PID_MANIFEST="${DIFFUSERS_PID_MANIFEST:-${LOG_ROOT}/pids.tsv}"
GPUS="${DIFFUSERS_GPUS:-4,5,6,7}"
PORTS="${DIFFUSERS_PORTS:-8212,8213,8214,8215}"
PROFILE_NAME="${DIFFUSERS_PROFILE_NAME:-eager_4b_bf16}"
BASE_SEED="${DIFFUSERS_BASE_SEED:-764952063587760}"
ATTENTION_BACKEND="${DIFFUSERS_ATTENTION_BACKEND:-}"
CACHE_MODE="${DIFFUSERS_CACHE_MODE:-none}"
CACHE_THRESHOLD="${DIFFUSERS_CACHE_THRESHOLD:-0.05}"
MAG_RATIOS="${DIFFUSERS_MAG_RATIOS:-}"
COMPILE="${DIFFUSERS_COMPILE:-0}"
WARMUP_IMAGE="${DIFFUSERS_WARMUP_IMAGE:-}"
ACTION="${1:-start}"

stop_services() {
  if [[ ! -f "${PID_MANIFEST}" ]]; then
    echo "[diffusers] no pid manifest: ${PID_MANIFEST}"
    return
  fi
  while IFS=$'\t' read -r pid gpu port; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}"
    fi
  done < "${PID_MANIFEST}"
  for _ in $(seq 1 30); do
    alive=0
    while IFS=$'\t' read -r pid gpu port; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        alive=1
      fi
    done < "${PID_MANIFEST}"
    [[ "${alive}" == "0" ]] && break
    sleep 1
  done
  rm -f "${PID_MANIFEST}"
  echo "[diffusers] stopped owned services"
}

if [[ "${ACTION}" == "stop" ]]; then
  stop_services
  exit 0
fi
if [[ "${ACTION}" != "start" ]]; then
  echo "usage: $0 [start|stop]" >&2
  exit 2
fi

test -x "${VENV}/bin/python" || { echo "missing Diffusers venv: ${VENV}" >&2; exit 1; }
test -d "${MODEL_PATH}" || { echo "missing model: ${MODEL_PATH}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
IFS=',' read -r -a PORT_ARRAY <<< "${PORTS}"
[[ "${#GPU_ARRAY[@]}" == "${#PORT_ARRAY[@]}" ]] || { echo "GPU and port counts differ" >&2; exit 1; }
[[ ! -e "${PID_MANIFEST}" ]] || { echo "pid manifest already exists: ${PID_MANIFEST}" >&2; exit 1; }

for port in "${PORT_ARRAY[@]}"; do
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "port already serves HTTP: ${port}" >&2
    exit 1
  fi
done

tmp_manifest="${PID_MANIFEST}.starting.$$"
: > "${tmp_manifest}"
cleanup_failed_start() {
  if [[ -f "${tmp_manifest}" ]]; then
    while IFS=$'\t' read -r pid gpu port; do
      kill "${pid}" 2>/dev/null || true
    done < "${tmp_manifest}"
    rm -f "${tmp_manifest}"
  fi
}
trap cleanup_failed_start ERR INT TERM

for idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$idx]}"
  port="${PORT_ARRAY[$idx]}"
  args=(
    -m editor.server
    --host 127.0.0.1
    --port "${port}"
    --model-path "${MODEL_PATH}"
    --output-dir "${OUTPUT_ROOT}/gpu${gpu}"
    --profile-name "${PROFILE_NAME}"
    --base-seed "${BASE_SEED}"
    --attention-backend "${ATTENTION_BACKEND}"
    --cache-mode "${CACHE_MODE}"
    --cache-threshold "${CACHE_THRESHOLD}"
    --mag-ratios "${MAG_RATIOS}"
  )
  [[ "${COMPILE}" == "1" ]] && args+=(--compile)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    nohup "${VENV}/bin/python" "${args[@]}" >"${LOG_ROOT}/gpu${gpu}.log" 2>&1 &
  pid=$!
  printf '%s\t%s\t%s\n' "${pid}" "${gpu}" "${port}" >> "${tmp_manifest}"
done

while IFS=$'\t' read -r pid gpu port; do
  ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -100 "${LOG_ROOT}/gpu${gpu}.log" >&2 || true
      exit 1
    fi
    if "${VENV}/bin/python" - "${port}" 2>/dev/null <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5) as response:
    payload = json.load(response)
raise SystemExit(0 if payload.get("ready") is True else 1)
PY
    then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" == "1" ]] || { echo "service not ready on port ${port}" >&2; exit 1; }

  actual_uuid="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits | awk -F', ' -v pid="${pid}" '$1 == pid {print $2; exit}')"
  expected_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' -v gpu="${gpu}" '$1 == gpu {print $2; exit}')"
  [[ -n "${actual_uuid}" && "${actual_uuid}" == "${expected_uuid}" ]] || {
    echo "GPU placement mismatch pid=${pid} expected_gpu=${gpu}" >&2
    exit 1
  }
done < "${tmp_manifest}"

if [[ -z "${WARMUP_IMAGE}" ]]; then
  WARMUP_IMAGE="$(find "${TRAIN_IMAGE_ROOT:?TRAIN_IMAGE_ROOT is required}" -type f -print -quit)"
fi
test -f "${WARMUP_IMAGE}" || { echo "missing warmup image: ${WARMUP_IMAGE}" >&2; exit 1; }
while IFS=$'\t' read -r pid gpu port; do
  "${VENV}/bin/python" - "${port}" "${WARMUP_IMAGE}" <<'PY'
import json, sys, urllib.request
payload = json.dumps({
    "image_path": sys.argv[2],
    "positive_prompt": "Improve perceptual quality while preserving content, identity, geometry, and composition.",
    "request_index": -1,
}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[1]}/edit",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=900) as response:
    result = json.load(response)
assert result["status"] == "success", result
PY
done < "${tmp_manifest}"

mv "${tmp_manifest}" "${PID_MANIFEST}"
trap - ERR INT TERM
echo "[diffusers] ready profile=${PROFILE_NAME} pids=${PID_MANIFEST}"
