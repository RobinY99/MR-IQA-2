#!/usr/bin/env bash

vllm_prepare_cuda13_runtime() {
  local python_bin="${1:-python}"
  local site_packages
  local cuda_toolkit_root
  local cuda_runtime_lib
  local torch_runtime_lib
  local nvcc_version

  site_packages="$("${python_bin}" -c 'import site; print(site.getsitepackages()[0])')"
  cuda_toolkit_root="${site_packages}/nvidia/cu13"
  cuda_runtime_lib="${cuda_toolkit_root}/lib"
  torch_runtime_lib="${site_packages}/torch/lib"

  [[ -x "${cuda_toolkit_root}/bin/nvcc" ]] || {
    echo "missing CUDA 13 compiler: ${cuda_toolkit_root}/bin/nvcc" >&2
    return 1
  }
  [[ -f "${cuda_runtime_lib}/libnvrtc.so.13" ]] || {
    echo "missing CUDA 13 runtime library: ${cuda_runtime_lib}/libnvrtc.so.13" >&2
    return 1
  }
  [[ -d "${torch_runtime_lib}" ]] || {
    echo "missing Torch runtime library directory: ${torch_runtime_lib}" >&2
    return 1
  }

  export CUDA_HOME="${cuda_toolkit_root}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${cuda_runtime_lib}:${torch_runtime_lib}:${LD_LIBRARY_PATH:-}"
  export FLA_TILELANG="${FLA_TILELANG:-0}"
  [[ "${FLA_TILELANG}" == "0" ]] || {
    echo "FLA_TILELANG must be 0 on the RTX A6000 runtime" >&2
    return 1
  }
  nvcc_version="$("${CUDA_HOME}/bin/nvcc" --version)"
  [[ "${nvcc_version}" == *"release 13."* ]] || {
    echo "CUDA compiler is not major version 13" >&2
    return 1
  }

  "${python_bin}" - <<'PY'
import importlib.metadata
import os
import re
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME

if torch.version.cuda is None or torch.version.cuda.split(".", 1)[0] != "13":
    raise RuntimeError(f"vLLM runtime requires Torch CUDA 13, got {torch.version.cuda!r}")
if CUDA_HOME != os.environ["CUDA_HOME"]:
    raise RuntimeError(f"Torch resolved CUDA_HOME={CUDA_HOME!r}, expected {os.environ['CUDA_HOME']!r}")
vllm_version = importlib.metadata.version("vllm")
if vllm_version != "0.24.0":
    raise RuntimeError(f"validated vLLM version is 0.24.0, got {vllm_version}")
nvcc_text = subprocess.check_output([str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"), "--version"], text=True)
nvcc_match = re.search(r"release\s+(\d+\.\d+)", nvcc_text)
if not nvcc_match:
    raise RuntimeError("could not parse CUDA compiler release")
nvcc_release = nvcc_match.group(1)
header = (Path(os.environ["CUDA_HOME"]) / "include" / "cuda_runtime_api.h").read_text(encoding="utf-8")
header_match = re.search(r"#define\s+CUDART_VERSION\s+(\d+)", header)
if not header_match:
    raise RuntimeError("could not parse CUDART_VERSION from CUDA headers")
header_version = int(header_match.group(1))
header_release = f"{header_version // 1000}.{(header_version % 1000) // 10}"
print(
    f"[vllm-runtime] torch_cuda={torch.version.cuda} vllm={vllm_version} "
    f"nvcc={nvcc_release} headers={header_release} fla_tilelang={os.environ['FLA_TILELANG']} sleep_mode=false"
)
PY
}
