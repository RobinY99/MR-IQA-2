#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${VF_PROJECT_ROOT:-$(cd "${ROOT_DIR}/.." && pwd)}"
DIFFUSERS_SCRIPT="${PROJECT_ROOT}/editor/launch.sh"
JUDGER_SCRIPT="${PROJECT_ROOT}/judge/launch.sh"
DIFFUSERS_VENV="${DIFFUSERS_VENV:?DIFFUSERS_VENV is required}"
DIFFUSERS_MODEL_PATH="${DIFFUSERS_MODEL_PATH:?DIFFUSERS_MODEL_PATH is required}"
IMAGE_EDIT_BACKEND="${IMAGE_EDIT_BACKEND:-diffusers}"
SERVICE_GPUS="${SERVICE_GPUS:-4,5,6,7}"
SERVICE_EDITOR_PORTS="${SERVICE_EDITOR_PORTS:-8212,8213,8214,8215}"
SERVICE_JUDGER_PORTS="${SERVICE_JUDGER_PORTS:-8204,8205,8206,8207}"
export JUDGER_GPUS="${SERVICE_GPUS}"
export JUDGER_PORTS="${SERVICE_JUDGER_PORTS}"
ACTION="${1:---print-plan}"
RUN_DIR="${2:-}"

print_plan() {
  python3 - \
    "${SERVICE_GPUS}" \
    "${SERVICE_EDITOR_PORTS}" \
    "${SERVICE_JUDGER_PORTS}" <<PY
import json
import sys

gpus = [int(value) for value in sys.argv[1].split(",") if value]
editor_ports = [int(value) for value in sys.argv[2].split(",") if value]
judger_ports = [int(value) for value in sys.argv[3].split(",") if value]
print(json.dumps({
  "stack": "flux_editor_e5_judge_v1",
  "backend": "${IMAGE_EDIT_BACKEND}",
  "gpus": gpus,
  "diffusers_ports": editor_ports,
  "judger_ports": judger_ports,
  "diffusers_profile": "eager_4b_bf16",
  "pairs_per_gpu": 1,
  "routing": "paired_lane_ewma_with_judge_work_stealing",
}, indent=2))
PY
}

require_diffusers_backend() {
  [[ "${IMAGE_EDIT_BACKEND}" == "diffusers" ]] || {
    echo "resident stack starts only the explicitly selected diffusers backend" >&2
    return 1
  }
}

require_run_dir() {
  [[ -n "${RUN_DIR}" ]] || { echo "RUN_DIR is required" >&2; exit 2; }
  RUN_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${RUN_DIR}")"
  DIFFUSERS_DIR="${RUN_DIR}/services/diffusers_flux2"
  export DIFFUSERS_OUTPUT_ROOT="${DIFFUSERS_DIR}/outputs"
  export DIFFUSERS_LOG_ROOT="${DIFFUSERS_DIR}/logs"
  export DIFFUSERS_PID_MANIFEST="${DIFFUSERS_DIR}/pids.tsv"
  export DIFFUSERS_GPUS="${SERVICE_GPUS}"
  export DIFFUSERS_PORTS="${SERVICE_EDITOR_PORTS}"
  export DIFFUSERS_PROFILE_NAME="eager_4b_bf16"
  export DIFFUSERS_MODEL_PATH
  export DIFFUSERS_CACHE_MODE="none"
  export DIFFUSERS_COMPILE="0"
  STACK_MANIFEST="${RUN_DIR}/services/service_stack.json"
}

validate_config() {
  require_diffusers_backend
  python3 - \
    "${SERVICE_GPUS}" \
    "${SERVICE_EDITOR_PORTS}" \
    "${SERVICE_JUDGER_PORTS}" <<'PY'
import sys

try:
    gpus = [int(value) for value in sys.argv[1].split(",") if value]
    editor_ports = [int(value) for value in sys.argv[2].split(",") if value]
    judger_ports = [int(value) for value in sys.argv[3].split(",") if value]
except ValueError as exc:
    raise SystemExit(f"service topology must contain integers: {exc}")
if not 1 <= len(gpus) <= 8:
    raise SystemExit("service topology must contain one to eight lanes")
if len(gpus) != len(editor_ports) or len(gpus) != len(judger_ports):
    raise SystemExit("service GPU, Editor-port, and Judge-port counts differ")
if len(set(gpus)) != len(gpus):
    raise SystemExit("service GPU indices must be unique")
if len(set(editor_ports + judger_ports)) != len(editor_ports) + len(judger_ports):
    raise SystemExit("service ports must be unique across Editor and Judge")
if any(gpu < 0 for gpu in gpus):
    raise SystemExit("service GPU indices must be non-negative")
if any(port < 1024 or port > 65535 for port in editor_ports + judger_ports):
    raise SystemExit("service ports must be in [1024, 65535]")
PY
  [[ -x "${DIFFUSERS_SCRIPT}" ]] || { echo "missing Diffusers launcher: ${DIFFUSERS_SCRIPT}" >&2; return 1; }
  [[ -x "${DIFFUSERS_VENV}/bin/python" ]] || { echo "missing Diffusers venv" >&2; return 1; }
  [[ -d "${DIFFUSERS_MODEL_PATH}" ]] || { echo "missing Diffusers model" >&2; return 1; }
  [[ -f "${PROJECT_ROOT}/editor/client.py" ]] || { echo "missing Diffusers client" >&2; return 1; }
  bash -n "${DIFFUSERS_SCRIPT}"
  "${JUDGER_SCRIPT}" --validate-config
}

validate_editor_only_config() {
  require_diffusers_backend
  python3 - "${SERVICE_GPUS}" "${SERVICE_EDITOR_PORTS}" <<'PY'
import sys

gpus = [int(value) for value in sys.argv[1].split(",") if value]
ports = [int(value) for value in sys.argv[2].split(",") if value]
if not 1 <= len(gpus) <= 8 or len(gpus) != len(ports):
    raise SystemExit("Editor topology must contain one to eight GPU/port lanes")
if len(set(gpus)) != len(gpus) or len(set(ports)) != len(ports):
    raise SystemExit("Editor GPU indices and ports must be unique")
PY
  [[ -x "${DIFFUSERS_SCRIPT}" ]] || { echo "missing Diffusers launcher: ${DIFFUSERS_SCRIPT}" >&2; return 1; }
  [[ -x "${DIFFUSERS_VENV}/bin/python" ]] || { echo "missing Diffusers venv" >&2; return 1; }
  [[ -d "${DIFFUSERS_MODEL_PATH}" ]] || { echo "missing Diffusers model" >&2; return 1; }
  bash -n "${DIFFUSERS_SCRIPT}"
}

validate_editor_pid_topology() {
  [[ -f "${DIFFUSERS_PID_MANIFEST}" ]] || {
    echo "missing Editor PID manifest: ${DIFFUSERS_PID_MANIFEST}" >&2
    return 1
  }
  python3 - \
    "${DIFFUSERS_PID_MANIFEST}" \
    "${SERVICE_GPUS}" \
    "${SERVICE_EDITOR_PORTS}" <<'PY'
import os
import sys
from pathlib import Path

rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    pid, gpu, port = line.split("\t")
    rows.append((int(pid), int(gpu), int(port)))
expected_gpus = [int(value) for value in sys.argv[2].split(",") if value]
expected_ports = [int(value) for value in sys.argv[3].split(",") if value]
if len(rows) != len(expected_gpus):
    raise SystemExit("Editor process count does not match topology")
if [row[1] for row in rows] != expected_gpus:
    raise SystemExit("Editor GPU topology mismatch")
if [row[2] for row in rows] != expected_ports:
    raise SystemExit("Editor port topology mismatch")
dead = [pid for pid, _, _ in rows if not Path(f"/proc/{pid}").is_dir()]
if dead:
    raise SystemExit(f"Editor processes are dead: {dead}")
PY
}

editor_health() {
  python3 - "${SERVICE_EDITOR_PORTS}" "${DIFFUSERS_MODEL_PATH}" <<'PY'
import json
import sys
import urllib.request

for port in (int(value) for value in sys.argv[1].split(",") if value):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
        data = json.load(response)
    assert data.get("ready") is True, data
    assert data.get("backend") == "diffusers_flux2", data
    assert data.get("model_path") == sys.argv[2], data
    assert data.get("profile_name") == "eager_4b_bf16", data
    assert data.get("dtype") == "bf16", data
    assert data.get("steps") == 4, data
    assert data.get("cfg") == 1.0, data
    assert data.get("compile") is False, data
print("[diffusers] healthy")
PY
}

write_component_manifest() {
  local component="$1"
  mkdir -p "${RUN_DIR}/services"
  python3 - \
    "${RUN_DIR}/services/${component}_stack.json" \
    "${component}" \
    "${SERVICE_GPUS}" \
    "${SERVICE_EDITOR_PORTS}" \
    "${SERVICE_JUDGER_PORTS}" <<'PY'
import datetime
import json
import subprocess
import sys

path, component, gpu_text, editor_text, judge_text = sys.argv[1:]
payload = {
    "schema_version": "vf_sequential_offline_service_component_v1",
    "component": component,
    "status": "ready",
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "gpus": [int(value) for value in gpu_text.split(",") if value],
    "editor_ports": (
        [int(value) for value in editor_text.split(",") if value]
        if component == "editor"
        else []
    ),
    "judge_ports": (
        [int(value) for value in judge_text.split(",") if value]
        if component == "e5_judge"
        else []
    ),
    "gpu_memory": subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines(),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

start_editor_only() {
  require_run_dir
  validate_editor_only_config
  mkdir -p "${DIFFUSERS_DIR}"
  "${DIFFUSERS_SCRIPT}" start
  validate_editor_pid_topology
  editor_health
  write_component_manifest editor
  echo "[resident-stack] Editor-only eight-lane service ready"
}

status_editor_only() {
  require_run_dir
  validate_editor_pid_topology
  editor_health
}

stop_editor_only() {
  require_run_dir
  if [[ -f "${DIFFUSERS_PID_MANIFEST}" ]]; then
    "${DIFFUSERS_SCRIPT}" stop
  fi
  echo "[resident-stack] Editor-only services stopped"
}

start_judge_only() {
  require_run_dir
  "${JUDGER_SCRIPT}" --validate-config
  "${JUDGER_SCRIPT}" start "${RUN_DIR}"
  "${JUDGER_SCRIPT}" status "${RUN_DIR}"
  write_component_manifest e5_judge
  echo "[resident-stack] E5-Judge-only eight-lane service ready"
}

status_judge_only() {
  require_run_dir
  "${JUDGER_SCRIPT}" status "${RUN_DIR}"
}

stop_judge_only() {
  require_run_dir
  if [[ -f "${RUN_DIR}/services/frozen_judger/pids.tsv" ]]; then
    "${JUDGER_SCRIPT}" stop "${RUN_DIR}"
  fi
  echo "[resident-stack] E5-Judge-only services stopped"
}

validate_gpu_pairing() {
  [[ -f "${DIFFUSERS_PID_MANIFEST}" ]] || { echo "missing Editor PID manifest" >&2; return 1; }
  JUDGER_PID_MANIFEST="${RUN_DIR}/services/frozen_judger/pids.tsv"
  [[ -f "${JUDGER_PID_MANIFEST}" ]] || { echo "missing Judge PID manifest" >&2; return 1; }
  python3 - \
    "${DIFFUSERS_PID_MANIFEST}" \
    "${JUDGER_PID_MANIFEST}" \
    "${SERVICE_GPUS}" \
    "${SERVICE_EDITOR_PORTS}" \
    "${SERVICE_JUDGER_PORTS}" <<'PY'
import sys
from pathlib import Path

def rows(path):
    parsed = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        pid, gpu, port = line.split("\t")
        parsed.append((int(pid), int(gpu), int(port)))
    return parsed

editors = rows(sys.argv[1])
judgers = rows(sys.argv[2])
expected_gpus = [int(value) for value in sys.argv[3].split(",") if value]
expected_editor_ports = [int(value) for value in sys.argv[4].split(",") if value]
expected_judger_ports = [int(value) for value in sys.argv[5].split(",") if value]
if len(editors) != len(expected_gpus) or len(judgers) != len(expected_gpus):
    raise SystemExit("Editor/Judge process count does not match service topology")
if [row[1] for row in editors] != expected_gpus:
    raise SystemExit("Editor GPU topology mismatch")
if [row[1] for row in judgers] != expected_gpus:
    raise SystemExit("Judge GPU topology mismatch")
if [row[2] for row in editors] != expected_editor_ports:
    raise SystemExit("Editor port topology mismatch")
if [row[2] for row in judgers] != expected_judger_ports:
    raise SystemExit("Judge port topology mismatch")
PY
}

write_stack_manifest() {
  mkdir -p "$(dirname "${STACK_MANIFEST}")"
  python3 - \
    "${STACK_MANIFEST}" \
    "${SERVICE_GPUS}" \
    "${SERVICE_EDITOR_PORTS}" \
    "${SERVICE_JUDGER_PORTS}" <<'PY'
import json, subprocess, sys, urllib.request

def health(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
        return json.load(response)

gpus = [int(value) for value in sys.argv[2].split(",") if value]
editor_ports = [int(value) for value in sys.argv[3].split(",") if value]
judger_ports = [int(value) for value in sys.argv[4].split(",") if value]
payload = {
    "schema_version": "vf_resident_service_stack_v2",
    "editor_backend": "diffusers",
    "diffusers": [health(port) for port in editor_ports],
    "judgers": [health(port) for port in judger_ports],
    "lanes": [
        {
            "lane": index,
            "gpu": gpu,
            "editor_url": f"http://127.0.0.1:{editor_ports[index]}",
            "judger_url": f"http://127.0.0.1:{judger_ports[index]}",
        }
        for index, gpu in enumerate(gpus)
    ],
    "lane_count": len(gpus),
    "pairs_per_gpu": 1,
    "routing": "paired_lane_ewma_with_judge_work_stealing",
    "gpu_memory": subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True).splitlines(),
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

start_stack() {
  require_run_dir
  require_diffusers_backend
  mkdir -p "${DIFFUSERS_DIR}"
  diffusers_started=0
  judger_started=0
  cleanup_failed_start() {
    if [[ "${judger_started}" == "1" ]]; then "${JUDGER_SCRIPT}" stop "${RUN_DIR}" || true; fi
    if [[ "${diffusers_started}" == "1" ]]; then "${DIFFUSERS_SCRIPT}" stop || true; fi
  }
  trap cleanup_failed_start ERR INT TERM

  "${DIFFUSERS_SCRIPT}" start
  diffusers_started=1
  "${JUDGER_SCRIPT}" start "${RUN_DIR}"
  judger_started=1
  validate_gpu_pairing
  write_stack_manifest
  trap - ERR INT TERM
  echo "[resident-stack] ready manifest=${STACK_MANIFEST}"
}

status_stack() {
  require_run_dir
  "${JUDGER_SCRIPT}" status "${RUN_DIR}"
  validate_gpu_pairing
  editor_health
}

stop_stack() {
  require_run_dir
  stop_judge_only
  stop_editor_only
  echo "[resident-stack] stopped owned services"
}

case "${ACTION}" in
  --print-plan) print_plan ;;
  --validate-config) validate_config ;;
  --validate-editor-config) validate_editor_only_config ;;
  start) start_stack ;;
  status) status_stack ;;
  stop) stop_stack ;;
  start-editor) start_editor_only ;;
  status-editor) status_editor_only ;;
  stop-editor) stop_editor_only ;;
  start-judge) start_judge_only ;;
  status-judge) status_judge_only ;;
  stop-judge) stop_judge_only ;;
  *) echo "usage: $0 [--print-plan|--validate-config|--validate-editor-config|start|status|stop|start-editor|status-editor|stop-editor|start-judge|status-judge|stop-judge] [RUN_DIR]" >&2; exit 2 ;;
esac
