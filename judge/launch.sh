#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${ROOT_DIR}/server.py"
RUNTIME_ENV="${ROOT_DIR}/vllm_runtime.sh"
CHECKPOINT_TOOL="${ROOT_DIR}/checkpoint_manifest.py"
RELEASE_CHECKPOINT_TOOL="${ROOT_DIR}/release_manifest.py"
JUDGER_PYTHON="${JUDGER_PYTHON:?JUDGER_PYTHON is required}"
JUDGER_ENV_ROOT=""
JUDGER_SITE_PACKAGES=""
JUDGER_TORCH_LIB=""
JUDGER_CUDA_TOOLKIT_ROOT=""
JUDGER_DRIVER_STUB_DIR=""
FLASHINFER_NVCC=""
JUDGER_MODEL_ID="${JUDGER_MODEL_ID:-source-e5-judge-step725}"
JUDGER_MODEL_PATH="${JUDGER_MODEL_PATH:?JUDGER_MODEL_PATH is required}"
JUDGER_MANIFEST_PATH="${JUDGER_MANIFEST_PATH:?JUDGER_MANIFEST_PATH is required}"
JUDGER_MODEL_TREE_SHA256="${JUDGER_MODEL_TREE_SHA256:?JUDGER_MODEL_TREE_SHA256 is required}"
JUDGER_MODEL_EXPORT_TREE_SHA256="${JUDGER_MODEL_EXPORT_TREE_SHA256:?JUDGER_MODEL_EXPORT_TREE_SHA256 is required}"
JUDGER_PROMPT_HASH="${JUDGER_PROMPT_HASH:?JUDGER_PROMPT_HASH is required}"
JUDGER_PROMPT_SCHEMA="${JUDGER_PROMPT_SCHEMA:-e5_training_reasoning_v5}"
JUDGER_BACKEND="${JUDGER_BACKEND:-e5_qwen35_4b_vllm_judge}"
JUDGER_GPUS="${JUDGER_GPUS:-4,5,6,7}"
JUDGER_PORTS="${JUDGER_PORTS:-8204,8205,8206,8207}"
JUDGER_MIN_FREE_MIB="${JUDGER_MIN_FREE_MIB:-14336}"
JUDGER_START_TIMEOUT_SEC="${JUDGER_START_TIMEOUT_SEC:-900}"
JUDGER_VERIFY_TREE="${JUDGER_VERIFY_TREE:-1}"
ACTION="${1:---print-plan}"
RUN_DIR="${2:-}"
export JUDGER_MODEL_ID JUDGER_MODEL_PATH JUDGER_MODEL_TREE_SHA256 JUDGER_MODEL_EXPORT_TREE_SHA256
export VF_JUDGE_MODEL_ID="${JUDGER_MODEL_ID}"
export VF_JUDGE_MODEL_PATH="${JUDGER_MODEL_PATH}"
export VF_JUDGE_MODEL_TREE_SHA256="${JUDGER_MODEL_TREE_SHA256}"
export VF_JUDGE_PROMPT_SCHEMA="${JUDGER_PROMPT_SCHEMA}"
export VF_JUDGE_PROMPT_HASH="${JUDGER_PROMPT_HASH}"

initialize_judger_runtime() {
  [[ -z "${JUDGER_ENV_ROOT}" ]] || return 0
  JUDGER_ENV_ROOT="$(dirname "$(dirname "$(readlink -f "${JUDGER_PYTHON}")")")"
  export PATH="${JUDGER_ENV_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${JUDGER_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
  JUDGER_SITE_PACKAGES="$("${JUDGER_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
  JUDGER_TORCH_LIB="$("${JUDGER_PYTHON}" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
  JUDGER_CUDA_TOOLKIT_ROOT="${JUDGER_SITE_PACKAGES}/nvidia/cu13"
  JUDGER_DRIVER_STUB_DIR="${JUDGER_ENV_ROOT}/targets/x86_64-linux/lib/stubs"
  export LD_LIBRARY_PATH="${JUDGER_TORCH_LIB}:${LD_LIBRARY_PATH}"
  export LIBRARY_PATH="${JUDGER_DRIVER_STUB_DIR}:${JUDGER_CUDA_TOOLKIT_ROOT}/lib:${LIBRARY_PATH:-}"
  FLASHINFER_NVCC="${JUDGER_ENV_ROOT}/bin/nvcc"
  export FLASHINFER_NVCC
  export CUDACXX="${FLASHINFER_NVCC}"
}

print_plan() {
  python3 - "${JUDGER_GPUS}" "${JUDGER_PORTS}" <<PY
import json
import sys

gpus = [int(value) for value in sys.argv[1].split(",") if value]
ports = [int(value) for value in sys.argv[2].split(",") if value]
print(json.dumps({
  "service": "e5_qwen35_4b_vllm_judge",
  "model_id": "${JUDGER_MODEL_ID}",
	  "model_path": "${JUDGER_MODEL_PATH}",
	  "model_tree_sha256": "${JUDGER_MODEL_TREE_SHA256}",
	  "model_export_tree_sha256": "${JUDGER_MODEL_EXPORT_TREE_SHA256}",
	  "prompt_hash": "${JUDGER_PROMPT_HASH}",
	  "backend": "${JUDGER_BACKEND}",
  "python": "${JUDGER_PYTHON}",
  "gpus": gpus,
  "ports": ports,
  "gpu_memory_utilization": 0.24,
  "min_free_mib_after_editor": int("${JUDGER_MIN_FREE_MIB}"),
  "one_editor_and_one_judge_per_gpu": True,
}, indent=2))
PY
}

require_run_dir() {
  [[ -n "${RUN_DIR}" ]] || { echo "RUN_DIR is required" >&2; exit 2; }
  RUN_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${RUN_DIR}")"
  SERVICE_DIR="${RUN_DIR}/services/frozen_judger"
  PID_MANIFEST="${SERVICE_DIR}/pids.tsv"
}

port_is_free() {
  local port="$1"
  ! ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .
}

gpu_free_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', *' -v target="${gpu}" '$1 == target {print $2; exit}'
}

validate_checkpoint_manifest() {
  "${JUDGER_PYTHON}" "${RELEASE_CHECKPOINT_TOOL}" \
    --manifest "${JUDGER_MANIFEST_PATH}" \
    --model-path "${JUDGER_MODEL_PATH}" \
    --expected-asset-id "${JUDGER_MODEL_ID}" \
    --expected-hub-subfolder judge/source-e5 \
    --expected-source-tree "${JUDGER_MODEL_TREE_SHA256}" \
    --expected-export-tree "${JUDGER_MODEL_EXPORT_TREE_SHA256}" \
    --expected-file-count 10 \
    --expected-total-bytes 9098689558 \
    --verify-files "${JUDGER_VERIFY_TREE}"
}

validate_config() {
  [[ -x "${JUDGER_PYTHON}" ]] || { echo "missing Judge Python: ${JUDGER_PYTHON}" >&2; return 1; }
  initialize_judger_runtime
  [[ -f "${SERVER}" ]] || { echo "missing Judge server: ${SERVER}" >&2; return 1; }
  [[ -f "${RUNTIME_ENV}" ]] || { echo "missing vLLM runtime environment: ${RUNTIME_ENV}" >&2; return 1; }
  [[ -f "${CHECKPOINT_TOOL}" ]] || { echo "missing checkpoint tool: ${CHECKPOINT_TOOL}" >&2; return 1; }
  [[ -f "${RELEASE_CHECKPOINT_TOOL}" ]] || { echo "missing release checkpoint tool: ${RELEASE_CHECKPOINT_TOOL}" >&2; return 1; }
  [[ -d "${JUDGER_MODEL_PATH}" ]] || { echo "missing Judge model: ${JUDGER_MODEL_PATH}" >&2; return 1; }
  [[ -f "${JUDGER_MANIFEST_PATH}" ]] || { echo "missing Judge manifest: ${JUDGER_MANIFEST_PATH}" >&2; return 1; }
  [[ -f "${JUDGER_MODEL_PATH}/model.safetensors.index.json" ]] || { echo "missing model index" >&2; return 1; }
  [[ -f "${JUDGER_MODEL_PATH}/model-00001-of-00002.safetensors" ]] || { echo "missing model shard 1" >&2; return 1; }
  [[ -f "${JUDGER_MODEL_PATH}/model-00002-of-00002.safetensors" ]] || { echo "missing model shard 2" >&2; return 1; }
  [[ -x "${JUDGER_ENV_ROOT}/bin/ninja" ]] || { echo "missing Judge JIT builder: ${JUDGER_ENV_ROOT}/bin/ninja" >&2; return 1; }
  [[ -x "${FLASHINFER_NVCC}" ]] || { echo "missing matching FlashInfer compiler: ${FLASHINFER_NVCC}" >&2; return 1; }
  [[ -f "${JUDGER_CUDA_TOOLKIT_ROOT}/include/cuda_runtime_api.h" ]] || { echo "missing CUDA 13 headers" >&2; return 1; }
  [[ -f "${JUDGER_DRIVER_STUB_DIR}/libcuda.so" ]] || { echo "missing CUDA driver link stub" >&2; return 1; }
  "${JUDGER_ENV_ROOT}/bin/ninja" --version
  "${JUDGER_PYTHON}" - "${FLASHINFER_NVCC}" \
    "${JUDGER_CUDA_TOOLKIT_ROOT}/include/cuda_runtime_api.h" <<'PY'
import re
import subprocess
import sys

import torch
import flash_attn_2_cuda

nvcc_text = subprocess.check_output([sys.argv[1], "--version"], text=True)
nvcc_match = re.search(r"release\s+(\d+\.\d+)", nvcc_text)
header_text = open(sys.argv[2], encoding="utf-8").read()
header_match = re.search(r"#define\s+CUDART_VERSION\s+(\d+)", header_text)
if not nvcc_match or not header_match:
    raise RuntimeError("could not resolve FlashInfer compiler/header versions")
header_version = int(header_match.group(1))
header_release = f"{header_version // 1000}.{(header_version % 1000) // 10}"
if nvcc_match.group(1) != header_release:
    raise RuntimeError(
        f"FlashInfer compiler/header mismatch: {nvcc_match.group(1)} != {header_release}"
    )
print(
    f"[judger-preflight] torch={torch.__version__} "
    f"flash_attn_2_cuda={flash_attn_2_cuda.__file__} "
    f"flashinfer_nvcc={nvcc_match.group(1)} headers={header_release}"
)
PY
  checkpoint_validation="$(validate_checkpoint_manifest)"
  echo "${checkpoint_validation}"

  IFS=',' read -r -a gpu_array <<< "${JUDGER_GPUS}"
  IFS=',' read -r -a port_array <<< "${JUDGER_PORTS}"
  python3 - "${JUDGER_GPUS}" "${JUDGER_PORTS}" <<'PY'
import sys

try:
    gpus = [int(value) for value in sys.argv[1].split(",") if value]
    ports = [int(value) for value in sys.argv[2].split(",") if value]
except ValueError as exc:
    raise SystemExit(f"Judge topology must contain integers: {exc}")
if not 1 <= len(gpus) <= 8 or len(gpus) != len(ports):
    raise SystemExit("Judge topology must contain one to eight paired slots")
if len(set(gpus)) != len(gpus):
    raise SystemExit("Judge GPU indices must be unique")
if len(set(ports)) != len(ports):
    raise SystemExit("Judge ports must be unique")
if any(gpu < 0 for gpu in gpus):
    raise SystemExit("Judge GPU indices must be non-negative")
if any(port < 1024 or port > 65535 for port in ports):
    raise SystemExit("Judge ports must be in [1024, 65535]")
PY
  for idx in "${!gpu_array[@]}"; do
    gpu="${gpu_array[$idx]}"
    port="${port_array[$idx]}"
    port_is_free "${port}" || { echo "Judge port in use: ${port}" >&2; return 1; }
    free_mib="$(gpu_free_mib "${gpu}")"
    [[ -n "${free_mib}" && "${free_mib}" -ge "${JUDGER_MIN_FREE_MIB}" ]] || {
      echo "insufficient free memory for Judge gpu=${gpu} free_mib=${free_mib:-unknown}" >&2
      return 1
    }
  done
}

prewarm_flashinfer_sampler() {
  local gpu="$1"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    "${JUDGER_PYTHON}" - <<'PY'
import torch
from flashinfer.sampling import top_k_mask_logits

logits = torch.randn(2, 128, device="cuda", dtype=torch.float32)
masked = top_k_mask_logits(logits, 20)
torch.cuda.synchronize()
if masked.shape != logits.shape or not torch.isfinite(masked).any():
    raise RuntimeError("FlashInfer sampler prewarm produced an invalid result")
print(
    f"[judger-preflight] flashinfer_sampler=ready "
    f"device={torch.cuda.get_device_name(0)} shape={tuple(masked.shape)}"
)
PY
}

owned_process() {
  local pid="$1"
  local port="$2"
  local cmdline
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  [[ "${cmdline}" == *"${SERVER}"* && "${cmdline}" == *"--port ${port}"* ]]
}

process_tree_pids() {
  "${JUDGER_PYTHON}" - "$1" <<'PY'
import pathlib
import sys

root = int(sys.argv[1])
parents = {}
for status in pathlib.Path("/proc").glob("[0-9]*/status"):
    try:
        text = status.read_text(encoding="utf-8")
        pid = int(status.parent.name)
        ppid = int(text.split("PPid:\t", 1)[1].splitlines()[0])
    except (OSError, ValueError, IndexError):
        continue
    parents[pid] = ppid

owned = {root}
changed = True
while changed:
    changed = False
    for pid, ppid in parents.items():
        if ppid in owned and pid not in owned:
            owned.add(pid)
            changed = True
for pid in sorted(owned):
    if pathlib.Path(f"/proc/{pid}").is_dir():
        print(pid)
PY
}

gpu_tree_matches() {
  local root_pid="$1"
  local expected_gpu="$2"
  "${JUDGER_PYTHON}" - "${root_pid}" "${expected_gpu}" <<'PY'
import pathlib
import subprocess
import sys

root = int(sys.argv[1])
expected_gpu = int(sys.argv[2])
parents = {}
for status in pathlib.Path("/proc").glob("[0-9]*/status"):
    try:
        text = status.read_text(encoding="utf-8")
        pid = int(status.parent.name)
        ppid = int(text.split("PPid:\t", 1)[1].splitlines()[0])
    except (OSError, ValueError, IndexError):
        continue
    parents[pid] = ppid

def descends_from(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == root:
            return True
        seen.add(pid)
        pid = parents.get(pid, 0)
    return False

gpu_rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
    text=True,
).splitlines()
uuid_to_gpu = {
    row.split(",", 1)[1].strip(): int(row.split(",", 1)[0].strip())
    for row in gpu_rows
}
process_rows = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
    text=True,
).splitlines()
owned = []
for row in process_rows:
    if not row.strip():
        continue
    pid_text, uuid = [item.strip() for item in row.split(",", 1)]
    pid = int(pid_text)
    if descends_from(pid):
        owned.append((pid, uuid_to_gpu.get(uuid)))
if not owned:
    raise SystemExit(f"Judge process tree {root} owns no GPU compute process")
wrong = [(pid, gpu) for pid, gpu in owned if gpu != expected_gpu]
if wrong:
    raise SystemExit(
        f"Judge process tree {root} expected GPU {expected_gpu}, got {wrong}"
    )
print(
    f"[judger] placement root_pid={root} gpu={expected_gpu} "
    f"compute_pids={','.join(str(pid) for pid, _ in owned)}"
)
PY
}

stop_manifest() {
  local manifest="$1"
  local pid gpu port tree_pid
  local -a owned_pids=()
  local -a alive=()
  [[ -f "${manifest}" ]] || return 0
  while IFS=$'\t' read -r pid gpu port; do
    if kill -0 "${pid}" 2>/dev/null && owned_process "${pid}" "${port}"; then
      while read -r tree_pid; do
        [[ -n "${tree_pid}" ]] && owned_pids+=("${tree_pid}")
      done < <(process_tree_pids "${pid}")
    fi
  done < "${manifest}"
  if ((${#owned_pids[@]})); then
    mapfile -t owned_pids < <(printf '%s\n' "${owned_pids[@]}" | sort -nu)
    kill -TERM "${owned_pids[@]}" 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    alive=()
    for pid in "${owned_pids[@]}"; do
      kill -0 "${pid}" 2>/dev/null && alive+=("${pid}")
    done
    ((${#alive[@]} == 0)) && break
    sleep 1
  done
  ((${#alive[@]} == 0)) || kill -KILL "${alive[@]}" 2>/dev/null || true
  rm -f "${manifest}"
}

health_matches() {
  local port="$1"
  "${JUDGER_PYTHON}" - \
    "${port}" \
    "${JUDGER_MODEL_ID}" \
	    "${JUDGER_MODEL_PATH}" \
	    "${JUDGER_MODEL_TREE_SHA256}" \
	    "${JUDGER_PROMPT_HASH}" \
	    "${JUDGER_BACKEND}" <<'PY'
import json
import sys
import urllib.request

port, model_id, model_path, tree_sha, prompt_hash, expected_backend = sys.argv[1:]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
    data = json.load(response)
generation = data.get("generation") or {}
metadata = data.get("judger") or {}
ok = (
    data.get("ready") is True
    and data.get("backend") == expected_backend
    and data.get("model_id") == model_id
    and data.get("model_path") == model_path
    and data.get("model_tree_sha256") == tree_sha
    and data.get("prompt_hash") == prompt_hash
    and metadata.get("deterministic") is True
    and metadata.get("cache_compatible") is True
    and metadata.get("backend") == expected_backend
    and generation.get("max_tokens") == 256
    and generation.get("temperature") == 0.0
    and generation.get("seed") == 42
    and generation.get("enable_thinking") is False
    and generation.get("max_pixels") == 196608
    and generation.get("min_pixels") == 3136
)
raise SystemExit(0 if ok else 1)
PY
}

start_services() {
  require_run_dir
  validate_config
  source "${RUNTIME_ENV}"
  vllm_prepare_cuda13_runtime "${JUDGER_PYTHON}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  mkdir -p "${SERVICE_DIR}"
  [[ ! -e "${PID_MANIFEST}" ]] || { echo "Judge PID manifest already exists" >&2; exit 1; }
  tmp_manifest="${PID_MANIFEST}.starting.$$"
  : > "${tmp_manifest}"
  cleanup_partial() { stop_manifest "${tmp_manifest}"; }
  trap cleanup_partial ERR INT TERM

  IFS=',' read -r -a gpu_array <<< "${JUDGER_GPUS}"
  IFS=',' read -r -a port_array <<< "${JUDGER_PORTS}"
  prewarm_flashinfer_sampler "${gpu_array[0]}"
  for idx in "${!gpu_array[@]}"; do
    gpu="${gpu_array[$idx]}"
    port="${port_array[$idx]}"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
      nohup "${JUDGER_PYTHON}" "${SERVER}" \
        --port "${port}" \
        --model-path "${JUDGER_MODEL_PATH}" \
        >"${SERVICE_DIR}/gpu${gpu}.log" 2>&1 &
    pid=$!
    printf '%s\t%s\t%s\n' "${pid}" "${gpu}" "${port}" >> "${tmp_manifest}"
  done

  deadline=$((SECONDS + JUDGER_START_TIMEOUT_SEC))
  while IFS=$'\t' read -r pid gpu port; do
    until health_matches "${port}" 2>/dev/null; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        tail -100 "${SERVICE_DIR}/gpu${gpu}.log" >&2 || true
        cleanup_partial
        trap - ERR INT TERM
        return 1
      fi
      if (( SECONDS >= deadline )); then
        echo "Judge startup timeout port=${port}" >&2
        cleanup_partial
        trap - ERR INT TERM
        return 1
      fi
      sleep 2
    done
    if ! gpu_tree_matches "${pid}" "${gpu}"; then
      cleanup_partial
      trap - ERR INT TERM
      return 1
    fi
  done < "${tmp_manifest}"

  mv "${tmp_manifest}" "${PID_MANIFEST}"
  trap - ERR INT TERM
  urls=()
  for port in "${port_array[@]}"; do
    urls+=("http://127.0.0.1:${port}")
  done
  joined_urls="$(IFS=,; echo "${urls[*]}")"
  echo "VF_LOOP_JUDGER_URLS=${joined_urls}"
}

status_services() {
  require_run_dir
  [[ -f "${PID_MANIFEST}" ]] || { echo "Judge stack not owned by ${RUN_DIR}" >&2; return 1; }
  while IFS=$'\t' read -r pid gpu port; do
    kill -0 "${pid}" 2>/dev/null && owned_process "${pid}" "${port}" && health_matches "${port}" || return 1
  done < "${PID_MANIFEST}"
  echo "[judger] healthy ${PID_MANIFEST}"
}

stop_services() {
  local partial
  require_run_dir
  stop_manifest "${PID_MANIFEST}"
  shopt -s nullglob
  for partial in "${PID_MANIFEST}.starting."*; do
    stop_manifest "${partial}"
  done
  shopt -u nullglob
  echo "[judger] stopped owned services"
}

case "${ACTION}" in
  --print-plan) print_plan ;;
  --validate-config) validate_config ;;
  start) start_services ;;
  status) status_services ;;
  stop) stop_services ;;
  *) echo "usage: $0 [--print-plan|--validate-config|start RUN_DIR|status RUN_DIR|stop RUN_DIR]" >&2; exit 2 ;;
esac
