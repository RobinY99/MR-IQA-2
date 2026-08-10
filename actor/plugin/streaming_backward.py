from __future__ import annotations

from dataclasses import dataclass


BACKWARD_MODES = ("combined", "branch", "microbatch")


@dataclass(frozen=True)
class LearnerChunk:
    rollout: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def build_backward_schedule(
    mode: str,
    *,
    a0_size: int,
    a1_size: int,
    microbatch_size: int,
) -> list[LearnerChunk]:
    if mode not in BACKWARD_MODES:
        raise ValueError(f"unsupported learner backward mode: {mode}")
    if a0_size <= 0 or a1_size <= 0:
        raise ValueError("A0 and A1 sizes must be positive")
    total = a0_size + a1_size
    if mode == "combined":
        return [LearnerChunk("combined", 0, total)]
    if mode == "branch":
        return [LearnerChunk("a0", 0, a0_size), LearnerChunk("a1", a0_size, total)]
    if microbatch_size <= 0:
        raise ValueError("microbatch size must be positive")
    if a0_size % microbatch_size or a1_size % microbatch_size:
        raise ValueError("A0 and A1 sizes must be divisible by microbatch size")

    chunks: list[LearnerChunk] = []
    for rollout, start, end in (("a0", 0, a0_size), ("a1", a0_size, total)):
        for chunk_start in range(start, end, microbatch_size):
            chunks.append(LearnerChunk(rollout, chunk_start, chunk_start + microbatch_size))
    return chunks
