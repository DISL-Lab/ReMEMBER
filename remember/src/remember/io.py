"""Strict, line-aware JSONL input and atomic JSONL output."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .schemas import Turn, Window


class InputError(ValueError):
    """An input JSONL line is malformed or does not meet the window schema."""


def _fsync_parent(path: Path) -> None:
    """Persist a completed rename in the containing directory."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(Path(path).parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _error(path: Path, line_number: int, message: str) -> InputError:
    return InputError(f"{path}:{line_number}: {message}")


def _parse_turn(value: Any, location: str, path: Path, line_number: int) -> Turn:
    if not isinstance(value, Mapping):
        raise _error(path, line_number, f"{location} must be an object")
    fields: dict[str, str] = {}
    for field in ("time", "speaker", "dialogue"):
        item = value.get(field)
        if not isinstance(item, str):
            raise _error(path, line_number, f"{location}.{field} must be a string")
        fields[field] = item
    return Turn(**fields)


def _parse_window(value: Any, path: Path, line_number: int) -> Window:
    if not isinstance(value, Mapping):
        raise _error(path, line_number, "row must be an object")
    window_id = value.get("window_id")
    if not isinstance(window_id, str) or not window_id.strip():
        raise _error(path, line_number, "window_id must be a non-empty string")
    parsed: dict[str, tuple[Turn, ...]] = {}
    for field in ("history", "turns"):
        rows = value.get(field)
        if not isinstance(rows, list):
            raise _error(path, line_number, f"{field} must be an array")
        parsed[field] = tuple(
            _parse_turn(turn, f"{field}[{index}]", path, line_number)
            for index, turn in enumerate(rows)
        )
    return Window(window_id=window_id, **parsed)


def load_windows(path: Path) -> Iterator[Window]:
    """Yield validated windows, preserving physical JSONL line numbers in errors."""
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise InputError(f"{source}: unable to read input: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _error(source, line_number, "invalid JSON") from exc
            yield _parse_window(row, source, line_number)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSONL to a sibling temporary file, then publish with atomic rename."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        _fsync_parent(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
