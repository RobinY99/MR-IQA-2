#!/usr/bin/env python3
"""Build the deterministic source-image manifest used by the local J0 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from PIL import Image, UnidentifiedImageError


DEFAULT_EXPECTED_SAMPLES = 7000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSON row at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise RuntimeError(f"training manifest is empty: {path}")
    return rows


def source_reference(row: dict[str, Any], index: int) -> str:
    source = row.get("source_image")
    images = row.get("images")
    image = images[0] if isinstance(images, list) and len(images) == 1 else None
    if source is None:
        source = image
    if not isinstance(source, str) or not source.strip():
        raise RuntimeError(f"training row {index} has no source image")
    source = source.strip().replace("\\", "/")
    if image is not None and str(image).strip().replace("\\", "/") != source:
        raise RuntimeError(f"training row {index} has conflicting source_image/images")
    relative = PurePosixPath(source)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"training row {index} has an unsafe source image path: {source!r}")
    return relative.as_posix()


def inspect_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"cannot read source image {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"source image has invalid dimensions: {path}")
    return int(width), int(height)


def build_rows(
    train_manifest: Path,
    image_root: Path,
    *,
    expected_samples: int,
) -> list[dict[str, Any]]:
    if expected_samples <= 0:
        raise RuntimeError("expected sample count must be positive")
    root = image_root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"image root is not a directory: {root}")

    output: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    image_paths: set[Path] = set()
    for index, row in enumerate(read_jsonl(train_manifest)):
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise RuntimeError(f"training row {index} has no sample_id")
        if sample_id in sample_ids:
            raise RuntimeError(f"duplicate sample_id: {sample_id}")

        relative = source_reference(row, index)
        image_path = (root / relative).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"source image escapes image root: {relative}") from exc
        if not image_path.is_file():
            raise RuntimeError(f"source image is missing: {image_path}")
        if image_path in image_paths:
            raise RuntimeError(f"duplicate source image: {image_path}")

        width, height = inspect_image(image_path)
        output.append(
            {
                "sample_id": sample_id,
                "source_image_path": str(image_path),
                "source_image_sha256": sha256_file(image_path),
                "source_width": width,
                "source_height": height,
            }
        )
        sample_ids.add(sample_id)
        image_paths.add(image_path)

    if len(output) != expected_samples:
        raise RuntimeError(
            f"source sample count mismatch: expected={expected_samples}, actual={len(output)}"
        )
    return output


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    output = path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=DEFAULT_EXPECTED_SAMPLES,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        rows = build_rows(
            args.train_manifest,
            args.image_root,
            expected_samples=args.expected_samples,
        )
        atomic_write_jsonl(args.output, rows)
        print(
            json.dumps(
                {
                    "output": str(args.output.expanduser().resolve()),
                    "sample_count": len(rows),
                    "status": "complete",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"build_source_manifest: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
