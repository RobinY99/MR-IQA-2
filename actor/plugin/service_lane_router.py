from __future__ import annotations

import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ServiceLease:
    stage: str
    lane_index: int
    gpu_index: int
    url: str
    reserved_at: float
    predicted_wait_seconds: float
    preferred_lane_index: int | None
    work_stolen: bool


@dataclass
class _LaneState:
    lane_index: int
    gpu_index: int
    editor_url: str
    judge_url: str
    editor_pending: int = 0
    judge_pending: int = 0
    editor_ewma_seconds: float = 1.0
    judge_ewma_seconds: float = 1.0
    editor_completed: int = 0
    judge_completed: int = 0
    editor_failures: int = 0
    judge_failures: int = 0


class ServiceLaneRouter:
    """EWMA routing across an explicit set of service lanes."""

    def __init__(
        self,
        editor_urls: Sequence[str],
        judge_urls: Sequence[str],
        *,
        process_rank: int = 0,
        gpu_indices: Sequence[int] | None = None,
        max_lanes_per_gpu: int = 1,
        ewma_alpha: float = 0.2,
        judge_steal_ratio: float = 1.25,
        schema_version: str = "vf_service_lane_router_v2",
    ):
        lane_count = len(editor_urls)
        if lane_count < 1 or len(judge_urls) != lane_count:
            raise ValueError("service topology must contain equal nonempty URL lists")
        if gpu_indices is None:
            gpu_indices = tuple(range(lane_count))
        normalized_gpu_indices = list(map(int, gpu_indices))
        if len(normalized_gpu_indices) != lane_count:
            raise ValueError("service topology must contain one GPU index per lane")
        if any(index < 0 for index in normalized_gpu_indices):
            raise ValueError("service GPU indices must be nonnegative")
        max_lanes_per_gpu = int(max_lanes_per_gpu)
        if max_lanes_per_gpu < 1:
            raise ValueError("max_lanes_per_gpu must be positive")
        gpu_lane_counts = Counter(normalized_gpu_indices)
        if any(count > max_lanes_per_gpu for count in gpu_lane_counts.values()):
            raise ValueError(
                "service GPU lane count exceeds max_lanes_per_gpu="
                f"{max_lanes_per_gpu}"
            )
        if not 0.0 < float(ewma_alpha) <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if float(judge_steal_ratio) < 1.0:
            raise ValueError("judge_steal_ratio must be at least 1")
        self.process_rank = int(process_rank)
        self.max_lanes_per_gpu = max_lanes_per_gpu
        self.ewma_alpha = float(ewma_alpha)
        self.judge_steal_ratio = float(judge_steal_ratio)
        self.schema_version = str(schema_version)
        self._lock = threading.Lock()
        self._sequence = 0
        self._lanes = [
            _LaneState(
                lane_index=index,
                gpu_index=int(gpu_indices[index]),
                editor_url=str(editor_urls[index]).rstrip("/"),
                judge_url=str(judge_urls[index]).rstrip("/"),
            )
            for index in range(lane_count)
        ]

    @staticmethod
    def _predicted_wait(pending: int, ewma_seconds: float) -> float:
        return max(0, int(pending)) * max(1e-6, float(ewma_seconds))

    def _rotated_tie_order(self) -> list[int]:
        start = (self.process_rank + self._sequence) % len(self._lanes)
        return [(start + offset) % len(self._lanes) for offset in range(len(self._lanes))]

    def reserve_editor(
        self,
        *,
        excluded_lane_indices: Sequence[int] = (),
    ) -> ServiceLease:
        excluded = {int(index) for index in excluded_lane_indices}
        with self._lock:
            candidates = [
                lane for lane in self._lanes if lane.lane_index not in excluded
            ]
            if not candidates:
                raise RuntimeError("all Editor lanes are excluded")
            tie_order = self._rotated_tie_order()
            tie_rank = {lane_index: rank for rank, lane_index in enumerate(tie_order)}
            lane = min(
                candidates,
                key=lambda item: (
                    self._predicted_wait(item.editor_pending, item.editor_ewma_seconds),
                    tie_rank[item.lane_index],
                ),
            )
            predicted = self._predicted_wait(
                lane.editor_pending,
                lane.editor_ewma_seconds,
            )
            lane.editor_pending += 1
            self._sequence += 1
            return ServiceLease(
                stage="editor",
                lane_index=lane.lane_index,
                gpu_index=lane.gpu_index,
                url=lane.editor_url,
                reserved_at=time.monotonic(),
                predicted_wait_seconds=predicted,
                preferred_lane_index=None,
                work_stolen=False,
            )

    def reserve_judge(
        self,
        *,
        preferred_lane_index: int,
        excluded_lane_indices: Sequence[int] = (),
    ) -> ServiceLease:
        excluded = {int(index) for index in excluded_lane_indices}
        preferred_lane_index = int(preferred_lane_index)
        with self._lock:
            candidates = [
                lane for lane in self._lanes if lane.lane_index not in excluded
            ]
            if not candidates:
                raise RuntimeError("all Judge lanes are excluded")
            if not 0 <= preferred_lane_index < len(self._lanes):
                raise ValueError("preferred Judge lane is out of range")
            preferred = self._lanes[preferred_lane_index]
            if preferred in candidates:
                preferred_cost = self._predicted_wait(
                    preferred.judge_pending,
                    preferred.judge_ewma_seconds,
                )
            else:
                preferred_cost = math.inf
            tie_order = self._rotated_tie_order()
            tie_rank = {lane_index: rank for rank, lane_index in enumerate(tie_order)}
            best = min(
                candidates,
                key=lambda item: (
                    self._predicted_wait(item.judge_pending, item.judge_ewma_seconds),
                    tie_rank[item.lane_index],
                ),
            )
            best_cost = self._predicted_wait(
                best.judge_pending,
                best.judge_ewma_seconds,
            )
            same_lane_budget = (
                best_cost * self.judge_steal_ratio
                + preferred.judge_ewma_seconds
            )
            lane = (
                preferred
                if preferred in candidates and preferred_cost <= same_lane_budget
                else best
            )
            predicted = self._predicted_wait(
                lane.judge_pending,
                lane.judge_ewma_seconds,
            )
            lane.judge_pending += 1
            self._sequence += 1
            return ServiceLease(
                stage="judge",
                lane_index=lane.lane_index,
                gpu_index=lane.gpu_index,
                url=lane.judge_url,
                reserved_at=time.monotonic(),
                predicted_wait_seconds=predicted,
                preferred_lane_index=preferred_lane_index,
                work_stolen=lane.lane_index != preferred_lane_index,
            )

    def complete(
        self,
        lease: ServiceLease,
        *,
        elapsed_seconds: float,
        success: bool,
    ) -> None:
        elapsed = float(elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("service elapsed time must be finite and nonnegative")
        with self._lock:
            lane = self._lanes[int(lease.lane_index)]
            if lease.stage == "editor":
                if lane.editor_pending <= 0:
                    raise RuntimeError("Editor lease completion underflow")
                lane.editor_pending -= 1
                lane.editor_ewma_seconds = (
                    self.ewma_alpha * elapsed
                    + (1.0 - self.ewma_alpha) * lane.editor_ewma_seconds
                )
                lane.editor_completed += 1
                lane.editor_failures += int(not success)
            elif lease.stage == "judge":
                if lane.judge_pending <= 0:
                    raise RuntimeError("Judge lease completion underflow")
                lane.judge_pending -= 1
                lane.judge_ewma_seconds = (
                    self.ewma_alpha * elapsed
                    + (1.0 - self.ewma_alpha) * lane.judge_ewma_seconds
                )
                lane.judge_completed += 1
                lane.judge_failures += int(not success)
            else:
                raise ValueError(f"unsupported service stage: {lease.stage}")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": self.schema_version,
                "process_rank": self.process_rank,
                "max_lanes_per_gpu": self.max_lanes_per_gpu,
                "ewma_alpha": self.ewma_alpha,
                "judge_steal_ratio": self.judge_steal_ratio,
                "lanes": [
                    {
                        "lane_index": lane.lane_index,
                        "gpu_index": lane.gpu_index,
                        "editor_url": lane.editor_url,
                        "judge_url": lane.judge_url,
                        "editor_pending": lane.editor_pending,
                        "judge_pending": lane.judge_pending,
                        "editor_ewma_seconds": lane.editor_ewma_seconds,
                        "judge_ewma_seconds": lane.judge_ewma_seconds,
                        "editor_completed": lane.editor_completed,
                        "judge_completed": lane.judge_completed,
                        "editor_failures": lane.editor_failures,
                        "judge_failures": lane.judge_failures,
                    }
                    for lane in self._lanes
                ],
            }


class PairedServiceLaneRouter(ServiceLaneRouter):
    """Training-time router fixed to the four paired GPU4-7 lanes."""

    def __init__(
        self,
        editor_urls: Sequence[str],
        judge_urls: Sequence[str],
        *,
        process_rank: int = 0,
        gpu_indices: Sequence[int] = (4, 5, 6, 7),
        ewma_alpha: float = 0.2,
        judge_steal_ratio: float = 1.25,
    ):
        if len(editor_urls) != 4 or len(judge_urls) != 4 or len(gpu_indices) != 4:
            raise ValueError("service topology must contain exactly four paired lanes")
        if list(map(int, gpu_indices)) != [4, 5, 6, 7]:
            raise ValueError("service lanes must map exactly to GPU4, GPU5, GPU6, GPU7")
        super().__init__(
            editor_urls,
            judge_urls,
            process_rank=process_rank,
            gpu_indices=gpu_indices,
            ewma_alpha=ewma_alpha,
            judge_steal_ratio=judge_steal_ratio,
            schema_version="vf_paired_service_lane_router_v1",
        )
