from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch


GIB = 1024**3
MIB = 1024**2
DEFAULT_BUDGET_GIB = 12
MIN_TENSOR_MIB = 16


def activation_offload_enabled(value: str) -> bool:
    if value not in {"0", "1"}:
        raise RuntimeError(
            f"VF_LEARNER_ACTIVATION_OFFLOAD must be 0 or 1, got {value!r}"
        )
    return value == "1"


def activation_offload_budget_bytes(value: str) -> int:
    try:
        budget_gib = int(value)
    except ValueError as exc:
        raise RuntimeError(
            "VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB must be an integer"
        ) from exc
    if budget_gib != DEFAULT_BUDGET_GIB:
        raise RuntimeError(
            "VF_LEARNER_ACTIVATION_OFFLOAD_BUDGET_GIB is locked to "
            f"{DEFAULT_BUDGET_GIB}, got {budget_gib}"
        )
    return budget_gib * GIB


@dataclass
class SavedTensorOffloadStats:
    budget_bytes: int
    min_tensor_bytes: int
    seen_cuda_tensors: int = 0
    seen_cuda_bytes: int = 0
    offloaded_tensors: int = 0
    offloaded_bytes: int = 0


@dataclass
class _OffloadedTensor:
    device: torch.device
    value: torch.Tensor


class SelectiveSavedTensorCpuOffload:
    """Move a bounded amount of large saved tensors to CPU without approximation."""

    def __init__(self, *, budget_bytes: int, min_tensor_bytes: int):
        if budget_bytes <= 0:
            raise RuntimeError("saved-tensor offload budget must be positive")
        if min_tensor_bytes <= 0:
            raise RuntimeError("saved-tensor offload threshold must be positive")
        self.stats = SavedTensorOffloadStats(
            budget_bytes=budget_bytes,
            min_tensor_bytes=min_tensor_bytes,
        )

    def pack(self, tensor: torch.Tensor) -> Any:
        if not tensor.is_cuda:
            return tensor
        tensor_bytes = tensor.numel() * tensor.element_size()
        self.stats.seen_cuda_tensors += 1
        self.stats.seen_cuda_bytes += tensor_bytes
        if tensor_bytes < self.stats.min_tensor_bytes:
            return tensor
        if self.stats.offloaded_bytes + tensor_bytes > self.stats.budget_bytes:
            return tensor
        value = tensor.detach().to(device="cpu", non_blocking=False)
        self.stats.offloaded_tensors += 1
        self.stats.offloaded_bytes += tensor_bytes
        return _OffloadedTensor(device=tensor.device, value=value)

    @staticmethod
    def unpack(packed: Any) -> torch.Tensor:
        if isinstance(packed, _OffloadedTensor):
            return packed.value.to(device=packed.device, non_blocking=False)
        return packed


@contextmanager
def saved_tensor_cpu_offload(
    enabled: bool,
    *,
    budget_bytes: int,
) -> Iterator[SavedTensorOffloadStats]:
    offloader = SelectiveSavedTensorCpuOffload(
        budget_bytes=budget_bytes,
        min_tensor_bytes=MIN_TENSOR_MIB * MIB,
    )
    if not enabled:
        yield offloader.stats
        return
    with torch.autograd.graph.saved_tensors_hooks(
        offloader.pack,
        offloader.unpack,
    ):
        yield offloader.stats
