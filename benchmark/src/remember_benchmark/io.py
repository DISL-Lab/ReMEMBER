"""Strict, dependency-free JSONL helpers for release artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def fsync_parent_directory(path: Path) -> None:
    """Persist a completed rename in the filesystem's parent directory."""
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield object rows from a JSONL file, rejecting malformed input."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            yield row


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write object rows atomically as UTF-8 JSONL with one newline per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("JSONL output rows must be objects")
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)
    fsync_parent_directory(path)
