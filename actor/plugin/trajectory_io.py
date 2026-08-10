from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_trajectory_id(
    *,
    run_id: str,
    phase: str,
    rank: int,
    reward_call: int,
    sample_id: str,
    completion_index: int,
) -> str:
    identity = "|".join(
        [str(run_id), str(phase), str(rank), str(reward_call), str(sample_id), str(completion_index)]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"vf-{digest}"


def rank_sharded_path(base: Path, rank: int) -> Path:
    path = Path(base)
    suffix = path.suffix or ".jsonl"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}.rank{int(rank):03d}{suffix}")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
