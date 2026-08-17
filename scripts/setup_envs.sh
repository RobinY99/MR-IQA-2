#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="all"
CONDA_BIN="${CONDA_EXE:-}"
FLASH_WHEEL="${FLASH_ATTN_WHEEL:-}"
DRY_RUN=0
VERIFY=1

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_envs.sh [options]

Create the pinned MR-IQA-2 environments without activating them.

Options:
  --profile NAME             inference, training, test, or all (default: all)
  --flash-attn-wheel PATH    validated wheel required by training/all
  --conda PATH               conda executable (default: CONDA_EXE or PATH)
  --no-verify                install without import/tests verification
  --dry-run                  print commands without changing the machine
  -h, --help                 show this help

Examples:
  bash scripts/setup_envs.sh --profile inference
  bash scripts/setup_envs.sh --profile test
  FLASH_ATTN_WHEEL=/abs/flash_attn.whl bash scripts/setup_envs.sh --profile all
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --profile)
      (($# >= 2)) || die "--profile requires a value"
      PROFILE="$2"
      shift 2
      ;;
    --flash-attn-wheel)
      (($# >= 2)) || die "--flash-attn-wheel requires a path"
      FLASH_WHEEL="$2"
      shift 2
      ;;
    --conda)
      (($# >= 2)) || die "--conda requires a path"
      CONDA_BIN="$2"
      shift 2
      ;;
    --no-verify)
      VERIFY=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "${PROFILE}" in
  inference|training|test|all) ;;
  *) die "unsupported profile: ${PROFILE}" ;;
esac

if [[ -z "${CONDA_BIN}" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "${CONDA_BIN}" ]]; then
  if ((DRY_RUN)); then
    CONDA_BIN="conda"
  else
    die "conda was not found; install Miniconda/Conda or pass --conda"
  fi
fi
if ((!DRY_RUN)) && [[ ! -x "${CONDA_BIN}" ]]; then
  die "conda executable is not runnable: ${CONDA_BIN}"
fi

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_command "$@"
  if ((!DRY_RUN)); then
    "$@"
  fi
}

setup_conda_environment() {
  local definition="$1"
  local name="$2"
  local requirements="$3"
  run "${CONDA_BIN}" env update --file "${ROOT}/${definition}" --prune
  run "${CONDA_BIN}" run --no-capture-output -n "${name}" \
    python -m pip install --upgrade pip
  run "${CONDA_BIN}" run --no-capture-output -n "${name}" \
    python -m pip install -r "${ROOT}/${requirements}"
}

setup_gpu_environments() {
  setup_conda_environment \
    environment/actor-judge.yml \
    mr_iqa_actor_judge \
    requirements/actor-judge.txt
  setup_conda_environment \
    environment/editor.yml \
    mr_iqa_editor \
    requirements/editor.txt
}

verify_inference_environments() {
  run "${CONDA_BIN}" run --no-capture-output -n mr_iqa_actor_judge \
    python -c 'import torch, transformers, swift, vllm; print("Actor/Judge imports OK")'
  run "${CONDA_BIN}" run --no-capture-output -n mr_iqa_editor \
    python -c 'import diffusers, torch, transformers; print("Editor imports OK")'
}

require_flash_attention_wheel() {
  [[ -n "${FLASH_WHEEL}" ]] || die \
    "training requires --flash-attn-wheel or FLASH_ATTN_WHEEL"
  if ((!DRY_RUN)); then
    [[ -f "${FLASH_WHEEL}" ]] || die \
      "FlashAttention wheel does not exist: ${FLASH_WHEEL}"
  fi
}

install_flash_attention() {
  if [[ -n "${FLASH_ATTN_WHEEL_SHA256:-}" ]] && ((!DRY_RUN)); then
    local actual_sha
    actual_sha="$(python3 - "${FLASH_WHEEL}" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
)"
    [[ "${actual_sha}" == "${FLASH_ATTN_WHEEL_SHA256}" ]] || die \
      "FlashAttention wheel SHA256 does not match FLASH_ATTN_WHEEL_SHA256"
  fi
  run "${CONDA_BIN}" run --no-capture-output -n mr_iqa_actor_judge \
    python -m pip install "${FLASH_WHEEL}"
  if ((VERIFY)); then
    run "${CONDA_BIN}" run --no-capture-output -n mr_iqa_actor_judge \
      python -c 'import flash_attn; print("FlashAttention import OK")'
  fi
}

ensure_environment_template() {
  if [[ ! -e "${ROOT}/.env" ]]; then
    run cp "${ROOT}/.env.example" "${ROOT}/.env"
  fi
}

setup_test_environment() {
  run "${CONDA_BIN}" env update \
    --file "${ROOT}/environment/test.yml" --prune
  run "${CONDA_BIN}" run --no-capture-output -n mr_iqa_release_test \
    python -m pip install --upgrade pip
  run "${CONDA_BIN}" run --no-capture-output -n mr_iqa_release_test \
    python -m pip install 'torch==2.11.0' \
    --index-url https://download.pytorch.org/whl/cpu
  run "${CONDA_BIN}" run --no-capture-output -n mr_iqa_release_test \
    python -m pip install -r "${ROOT}/requirements/test.txt"
  if ((VERIFY)); then
    run "${CONDA_BIN}" run --no-capture-output -n mr_iqa_release_test \
      python -c 'import PIL, numpy, pytest, requests, torch; print("Test imports OK")'
    if [[ -f "${ROOT}/scripts/test_release.sh" && \
          -f "${ROOT}/scripts/check_release.py" ]]; then
      run "${CONDA_BIN}" run --no-capture-output -n mr_iqa_release_test \
        bash "${ROOT}/scripts/test_release.sh"
    fi
  fi
}

case "${PROFILE}" in
  inference)
    setup_gpu_environments
    ((VERIFY)) && verify_inference_environments
    ;;
  training)
    require_flash_attention_wheel
    setup_gpu_environments
    install_flash_attention
    ((VERIFY)) && verify_inference_environments
    ensure_environment_template
    ;;
  test)
    setup_test_environment
    ;;
  all)
    require_flash_attention_wheel
    setup_gpu_environments
    install_flash_attention
    ((VERIFY)) && verify_inference_environments
    ensure_environment_template
    setup_test_environment
    ;;
esac

echo "Environment setup complete for profile: ${PROFILE}"
