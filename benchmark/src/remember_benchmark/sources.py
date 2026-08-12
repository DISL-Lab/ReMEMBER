"""Read user-acquired upstream corpora into one deterministic schema.

These adapters intentionally contain no network or downloader code.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .schemas import NormalizedDialogue, NormalizedTurn, SCHEMA_VERSION, dialogue_sort_key


class SourceInputError(ValueError):
    """A user-provided source path or record is unavailable or malformed."""


_GUIDANCE = {
    "EverMemBench": "Acquire EverMemBench from its upstream distributor, then pass the user-acquired file with --evermembench PATH.",
    "LoCoMo": "Acquire LoCoMo from its upstream distributor, then pass the user-acquired file with --locomo PATH.",
    "RealTalk": "Acquire RealTalk from its upstream distributor, then pass the user-acquired directory with --realtalk PATH.",
}


def _require(path: Path, source: str, *, directory: bool = False) -> Path:
    path = Path(path)
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise SourceInputError(f"{source} input not found at {path}. {_GUIDANCE[source]} No download is performed.")
    return path


def _json_error(path: Path, location: str, message: str, line: int | None = None) -> SourceInputError:
    where = f"{path}:{line}" if line is not None else str(path)
    return SourceInputError(f"{where}: {location}: {message}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _json_error(path, "JSON", exc.msg, exc.lineno) from exc


def _turn(path: Path, location: str, raw: Any, *, time: str | None = None, dialogue_key: str = "dialogue") -> NormalizedTurn:
    if not isinstance(raw, dict):
        raise _json_error(path, location, "must be an object")
    for key in ("speaker", dialogue_key):
        if key not in raw:
            raise _json_error(path, location, f"missing {key}", 1 if path.suffix == ".jsonl" else None)
    actual_time = time if time is not None else raw.get("time")
    if actual_time is None:
        raise _json_error(path, location, "missing time", 1 if path.suffix == ".jsonl" else None)
    try:
        return NormalizedTurn(str(actual_time), raw["speaker"], raw[dialogue_key])
    except ValueError as exc:
        raise _json_error(path, location, str(exc), 1 if path.suffix == ".jsonl" else None) from exc


def load_evermembench(path: Path) -> list[NormalizedDialogue]:
    path = _require(path, "EverMemBench")
    groups: dict[tuple[int, int], list[NormalizedTurn]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _json_error(path, "JSON", exc.msg, line_no) from exc
            if not isinstance(row, dict):
                raise _json_error(path, "record", "must be an object", line_no)
            try:
                topic = int(row["topic_id"])
                dialogues = row["dialogues"]
            except (KeyError, TypeError, ValueError) as exc:
                raise _json_error(path, "record", "requires integer topic_id and dialogues object", line_no) from exc
            if not isinstance(dialogues, dict):
                raise _json_error(path, "dialogues", "must be an object", line_no)
            for group_name, raw_turns in dialogues.items():
                match = re.fullmatch(r"Group (\d+)", str(group_name))
                if not match or not isinstance(raw_turns, list):
                    raise _json_error(path, f"dialogues[{group_name}]", "expected Group N mapped to a list", line_no)
                key = (topic, int(match.group(1)))
                bucket = groups.setdefault(key, [])
                for index, raw in enumerate(raw_turns):
                    location = f"dialogues[{group_name}][{index}]"
                    try:
                        bucket.append(_turn(path, location, raw))
                    except SourceInputError as exc:
                        # JSONL adapters always identify the actual record line.
                        raise SourceInputError(str(exc).replace(f"{path}:1", f"{path}:{line_no}", 1)) from exc
    return _dialogues("E", "evermembench", groups)


def load_locomo(path: Path) -> list[NormalizedDialogue]:
    path = _require(path, "LoCoMo")
    data = _read_json(path)
    if not isinstance(data, list):
        raise _json_error(path, "root", "must be a list")
    groups: dict[tuple[int, int], list[NormalizedTurn]] = {}
    for topic, sample in enumerate(data, 1):
        if not isinstance(sample, dict) or not isinstance(sample.get("conversation"), dict):
            raise _json_error(path, f"[{topic - 1}].conversation", "must be an object")
        conversation = sample["conversation"]
        session_numbers = sorted(
            int(match.group(1))
            for key in conversation
            if (match := re.fullmatch(r"session_(\d+)_date_time", key))
        )
        bucket = groups.setdefault((topic, 1), [])
        for session in session_numbers:
            turns = conversation.get(f"session_{session}")
            if not isinstance(turns, list):
                raise _json_error(path, f"[{topic - 1}].conversation.session_{session}", "must be a list")
            try:
                base = datetime.strptime(conversation[f"session_{session}_date_time"].strip(), "%I:%M %p on %d %B, %Y")
            except (KeyError, AttributeError, ValueError) as exc:
                raise _json_error(path, f"[{topic - 1}].conversation.session_{session}_date_time", "invalid timestamp") from exc
            for index, raw in enumerate(turns):
                stamp = (base + timedelta(seconds=index)).strftime("%Y-%m-%d %H:%M:%S")
                bucket.append(_turn(path, f"[{topic - 1}].conversation.session_{session}[{index}]", raw, time=stamp, dialogue_key="text"))
    return _dialogues("L", "locomo", groups)


def load_realtalk(path: Path) -> list[NormalizedDialogue]:
    path = _require(path, "RealTalk", directory=True)
    files = sorted(path.glob("Chat_*_*.json"), key=lambda item: item.name)
    if not files:
        raise SourceInputError(f"RealTalk directory {path} has no Chat_N_*.json files. {_GUIDANCE['RealTalk']}")
    groups: dict[tuple[int, int], list[NormalizedTurn]] = {}
    for topic, file_path in enumerate(files, 1):
        data = _read_json(file_path)
        if not isinstance(data, dict):
            raise _json_error(file_path, "root", "must be an object")
        session_numbers = sorted(
            int(match.group(1)) for key in data if (match := re.fullmatch(r"session_(\d+)_date_time", key))
        )
        bucket = groups.setdefault((topic, 1), [])
        for session in session_numbers:
            turns = data.get(f"session_{session}")
            if not isinstance(turns, list):
                raise _json_error(file_path, f"session_{session}", "must be a list")
            for index, raw in enumerate(turns):
                location = f"session_{session}[{index}]"
                if not isinstance(raw, dict) or "date_time" not in raw:
                    raise _json_error(file_path, location, "missing date_time")
                try:
                    stamp = datetime.strptime(raw["date_time"].strip(), "%d.%m.%Y, %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
                except (AttributeError, ValueError) as exc:
                    raise _json_error(file_path, location, "invalid date_time") from exc
                bucket.append(_turn(file_path, location, raw, time=stamp, dialogue_key="clean_text"))
    return _dialogues("R", "realtalk", groups)


def _dialogues(prefix: str, source: str, groups: dict[tuple[int, int], list[NormalizedTurn]]) -> list[NormalizedDialogue]:
    result: list[NormalizedDialogue] = []
    for (topic, group), turns in sorted(groups.items()):
        if not turns:
            continue
        # Python's stable sort preserves the upstream ordinal when timestamps tie.
        ordered = tuple(sorted(turns, key=lambda turn: turn.time))
        result.append(NormalizedDialogue(SCHEMA_VERSION, f"{prefix}_topic-{topic}_group-{group}", source, ordered))
    return result


def normalize_sources(
    *, evermembench: Path | None = None, locomo: Path | None = None, realtalk: Path | None = None
) -> list[NormalizedDialogue]:
    result: list[NormalizedDialogue] = []
    if evermembench is not None:
        result.extend(load_evermembench(evermembench))
    if locomo is not None:
        result.extend(load_locomo(locomo))
    if realtalk is not None:
        result.extend(load_realtalk(realtalk))
    result.sort(key=dialogue_sort_key)
    ids = [item.dialogue_id for item in result]
    if len(ids) != len(set(ids)):
        raise SourceInputError("normalized dialogue_id values must be unique")
    return result
