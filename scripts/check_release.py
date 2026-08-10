#!/usr/bin/env python3
"""Deterministic, dependency-free checks for the public release tree."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {
    "",
    ".cff",
    ".cfg",
    ".env",
    ".example",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
MODEL_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".db",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".whl",
}
REQUIRED_FILES = (
    ".env.example",
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "actor/plugin/vf_dual_rollout_trainer.py",
    "actor/scripts/run_editor_judge_grpo_stage.sh",
    "editor/server.py",
    "judge/server.py",
    "data/checksums.sha256",
    "data/train.jsonl",
    "data/validation.jsonl",
    "environment/actor-judge.yml",
    "environment/editor.yml",
    "requirements/actor-judge.txt",
    "requirements/editor.txt",
    "requirements/test.txt",
)
REQUIRED_ENV_KEYS = (
    "CONDA_SH",
    "CONDA_ENV_NAME",
    "ACTOR_MODEL_PATH",
    "TRAIN_IMAGE_ROOT",
    "DIFFUSERS_VENV",
    "DIFFUSERS_MODEL_PATH",
    "JUDGE_MODEL_PATH",
    "JUDGE_MODEL_TREE_SHA256",
    "JUDGE_MODEL_EXPORT_TREE_SHA256",
    "ORIGINAL_SCORE_CACHE_PATH",
    "ORIGINAL_SCORE_CACHE_SHA256",
    "FLASH_ATTN_WHEEL",
    "FLASH_ATTN_WHEEL_SHA256",
)


class Checks:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.errors: list[str] = []
        self.check_count = 0

    def error(self, message: str) -> None:
        self.errors.append(message)

    def pass_check(self, message: str) -> None:
        self.check_count += 1
        print(f"[OK] {message}")


def release_files(root: Path) -> Iterable[Path]:
    for current_root, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in {".git", ".pytest_cache", "__pycache__"}
        )
        current = Path(current_root)
        for name in sorted(file_names):
            yield current / name


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def check_required_files(checks: Checks) -> None:
    for item in REQUIRED_FILES:
        if not (checks.root / item).is_file():
            checks.error(f"missing required release file: {item}")
    if not checks.errors:
        checks.pass_check(f"required release surface ({len(REQUIRED_FILES)} files)")


def check_symlinks_and_blobs(checks: Checks) -> None:
    symlink_count = 0
    file_count = 0
    for path in release_files(checks.root):
        file_count += 1
        rel = relative(path, checks.root)
        if path.is_symlink():
            symlink_count += 1
            target = path.resolve(strict=False)
            try:
                target.relative_to(checks.root.resolve())
            except ValueError:
                checks.error(f"symlink escapes the release tree: {rel} -> {os.readlink(path)}")
        if path.suffix.lower() in MODEL_SUFFIXES:
            checks.error(
                f"checkpoint/runtime blob belongs on Hugging Face, not GitHub: {rel}"
            )
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        if size > 25 * 1024 * 1024:
            checks.error(f"GitHub release file exceeds 25 MiB: {rel} ({size} bytes)")
        if path.name == ".env":
            checks.error("private .env file is present; publish only .env.example")
    if not any("symlink" in item or "blob" in item or "25 MiB" in item for item in checks.errors):
        checks.pass_check(f"file/symlink policy ({file_count} files, {symlink_count} symlinks)")


def is_text_candidate(path: Path) -> bool:
    if path.name in {"Dockerfile", "Makefile"}:
        return True
    if path.name.endswith(".env.example"):
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def check_privacy(checks: Checks) -> None:
    private_patterns = (
        ("user home path", re.compile("/" + "(?:home|Users)" + r"/[A-Za-z0-9_.-]+/")),
        ("mounted data path", re.compile("/" + "mnt" + r"/[A-Za-z0-9_.-]+/")),
        ("private IPv4 address", re.compile(r"(?<![0-9])10(?:\.[0-9]{1,3}){3}(?![0-9])")),
        ("AWS access key", re.compile("AK" + r"IA[0-9A-Z]{16}")),
        ("Hugging Face token", re.compile("h" + r"f_[A-Za-z0-9]{20,}")),
        ("GitHub token", re.compile("github" + r"_pat_[A-Za-z0-9_]{20,}")),
        ("OpenAI-style key", re.compile("s" + r"k-[A-Za-z0-9_-]{20,}")),
        ("private key block", re.compile("BEGIN " + r"(?:RSA |OPENSSH )?PRIVATE KEY")),
    )
    scanned = 0
    for path in release_files(checks.root):
        if not is_text_candidate(path) or path.is_symlink():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        rel = relative(path, checks.root)
        for description, pattern in private_patterns:
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                checks.error(f"{description} in {rel}:{line}")
    if not any(
        label in item
        for item in checks.errors
        for label in (
            "user home path",
            "mounted data path",
            "private IPv4",
            "access key",
            "token",
            "OpenAI-style",
            "private key",
        )
    ):
        checks.pass_check(f"privacy/credential scan ({scanned} text files)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_data(checks: Checks) -> None:
    data_root = checks.root / "data"
    checksum_file = data_root / "checksums.sha256"
    if not checksum_file.is_file():
        return
    expected: dict[str, str] = {}
    for line_number, raw in enumerate(
        checksum_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", raw)
        if match is None:
            checks.error(f"invalid checksum line data/checksums.sha256:{line_number}")
            continue
        digest, name = match.groups()
        if name in expected:
            checks.error(f"duplicate checksum entry: data/{name}")
        expected[name] = digest

    jsonl_names = {path.name for path in data_root.glob("*.jsonl")}
    if set(expected) != jsonl_names:
        missing = sorted(jsonl_names - set(expected))
        stale = sorted(set(expected) - jsonl_names)
        if missing:
            checks.error(f"JSONL files missing checksums: {', '.join(missing)}")
        if stale:
            checks.error(f"checksum entries without files: {', '.join(stale)}")

    total_rows = 0
    for name, digest in sorted(expected.items()):
        path = data_root / name
        if not path.is_file():
            continue
        actual = sha256(path)
        if actual != digest:
            checks.error(f"checksum mismatch: data/{name}: {actual} != {digest}")
        row_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    checks.error(f"blank JSONL row: data/{name}:{line_number}")
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    checks.error(f"invalid JSON: data/{name}:{line_number}: {exc.msg}")
                    continue
                if not isinstance(payload, dict):
                    checks.error(f"JSONL row is not an object: data/{name}:{line_number}")
                    continue
                image_values: list[object] = []
                if "image" in payload:
                    image_values.append(payload["image"])
                if "source_image" in payload:
                    image_values.append(payload["source_image"])
                images = payload.get("images")
                if isinstance(images, list):
                    image_values.extend(images)
                for value in image_values:
                    value_text = str(value)
                    value_path = Path(value_text)
                    if value_path.is_absolute() or ".." in value_path.parts:
                        checks.error(
                            f"non-portable image path: data/{name}:{line_number}: {value_text}"
                        )
                row_count += 1
        if row_count == 0:
            checks.error(f"empty dataset: data/{name}")
        total_rows += row_count

    if not any(
        item.startswith(("invalid checksum", "duplicate checksum", "JSONL", "checksum", "blank JSONL", "invalid JSON", "JSONL row", "non-portable", "empty dataset"))
        for item in checks.errors
    ):
        checks.pass_check(f"dataset integrity ({len(expected)} files, {total_rows} rows)")


def check_python_syntax(checks: Checks) -> None:
    count = 0
    for path in release_files(checks.root):
        if path.suffix != ".py":
            continue
        count += 1
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=relative(path, checks.root))
        except (SyntaxError, UnicodeDecodeError) as exc:
            checks.error(f"Python syntax error in {relative(path, checks.root)}: {exc}")
    if not any(item.startswith("Python syntax") for item in checks.errors):
        checks.pass_check(f"Python syntax ({count} files)")


def check_shell_syntax(checks: Checks) -> None:
    count = 0
    for path in release_files(checks.root):
        if path.suffix != ".sh":
            continue
        count += 1
        result = subprocess.run(
            ["bash", "-n", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            checks.error(f"shell syntax error in {relative(path, checks.root)}: {detail}")
    if not any(item.startswith("shell syntax") for item in checks.errors):
        checks.pass_check(f"shell syntax ({count} files)")


def check_requirements(checks: Checks) -> None:
    count = 0
    for path in sorted((checks.root / "requirements").glob("*.txt")):
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("--"):
                continue
            count += 1
            is_exact = "==" in line
            is_pinned_git = bool(re.search(r" @ git\+https://[^ ]+@[0-9a-f]{40}$", line))
            if not (is_exact or is_pinned_git):
                checks.error(
                    f"unpinned requirement: {relative(path, checks.root)}:{line_number}: {line}"
                )
            if " @ file:" in line or line.startswith(("-e /", "/")):
                checks.error(
                    f"non-portable local requirement: {relative(path, checks.root)}:{line_number}"
                )
    if not any("requirement" in item for item in checks.errors):
        checks.pass_check(f"pinned requirements ({count} entries)")


def check_env_example(checks: Checks) -> None:
    path = checks.root / ".env.example"
    if not path.is_file():
        return
    keys: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _ = line.split("=", 1)
        keys.add(key.strip())
    missing = sorted(set(REQUIRED_ENV_KEYS) - keys)
    if missing:
        checks.error(f".env.example is missing keys: {', '.join(missing)}")
    else:
        checks.pass_check(f"environment template ({len(keys)} variables)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="release repository root (default: inferred from this script)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not (root / ".git").exists():
        print(f"error: not a release repository: {root}", file=sys.stderr)
        return 2
    checks = Checks(root)
    check_required_files(checks)
    check_symlinks_and_blobs(checks)
    check_privacy(checks)
    check_data(checks)
    check_python_syntax(checks)
    check_shell_syntax(checks)
    check_requirements(checks)
    check_env_example(checks)
    if checks.errors:
        print("\nRelease checks failed:", file=sys.stderr)
        for item in checks.errors:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print(f"\nRelease checks passed ({checks.check_count} groups).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
