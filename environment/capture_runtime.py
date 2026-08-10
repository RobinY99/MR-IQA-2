#!/usr/bin/env python3
"""Emit a sanitized runtime manifest without hostnames, users, or local paths."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from typing import Any


ROLE_PACKAGES = {
    "actor-judge": (
        "accelerate",
        "causal-conv1d",
        "datasets",
        "deepspeed",
        "fastapi",
        "flash-attn",
        "flash-linear-attention",
        "ms-swift",
        "numpy",
        "Pillow",
        "safetensors",
        "scipy",
        "torch",
        "transformers",
        "uvicorn",
        "vllm",
        "wandb",
    ),
    "editor": (
        "accelerate",
        "diffusers",
        "fastapi",
        "Pillow",
        "protobuf",
        "safetensors",
        "sentencepiece",
        "torch",
        "transformers",
        "uvicorn",
    ),
    "test": ("numpy", "Pillow", "pytest", "requests", "torch"),
}


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        return {"installed": False}
    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": devices,
    }


def nvidia_runtime() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    devices = []
    for raw in result.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 4:
            continue
        index, name, memory_mib, driver = fields
        devices.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory_mib),
                "driver_version": driver,
            }
        )
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=sorted(ROLE_PACKAGES), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "schema_version": "mr_iqa_2_runtime_manifest_v1",
        "role": args.role,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "packages": package_versions(ROLE_PACKAGES[args.role]),
        "torch": torch_runtime(),
        "nvidia_gpus": nvidia_runtime(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
