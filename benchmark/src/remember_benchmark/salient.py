"""Strict salient/gap candidate stage."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .providers import StageProvider


class StageOutputError(ValueError):
    pass


def _parse_response(raw: str, stage: str, window_id: str, field: str) -> list[Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageOutputError(f"{stage} {window_id}: invalid JSON response: {exc.msg}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise StageOutputError(f"{stage} {window_id}: response requires schema_version 1.0")
    value = payload.get(field)
    if not isinstance(value, list):
        raise StageOutputError(f"{stage} {window_id}: {field} must be a list")
    return value


def _sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def expand_window_sentences(turns: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    expanded: list[dict[str, str]] = []
    for turn in turns:
        for sentence in _sentences(str(turn.get("dialogue", ""))):
            expanded.append({"time": str(turn.get("time", "")), "speaker": str(turn.get("speaker", "")), "sentence": sentence})
    return expanded


def format_numbered_window(expanded: Sequence[dict[str, str]]) -> str:
    return "\n".join(
        f"[{index}]. [{item['time']}] {item['speaker']}: {item['sentence']}"
        for index, item in enumerate(expanded)
    )


def run_salient(records: Sequence[dict[str, Any]], provider: StageProvider, prompt: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        window_id = _window_id(record, seen)
        expanded = expand_window_sentences(_turn_list(record, "turns", window_id))
        rendered = prompt.format(window=format_numbered_window(expanded))
        candidates = _parse_response(provider.complete(rendered), "salient", window_id, "candidates")
        gaps: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            location = f"salient {window_id} candidate {candidate_index}"
            if not isinstance(candidate, dict):
                raise StageOutputError(f"{location}: must be an object")
            numbers = candidate.get("sentence_numbers")
            sentences = candidate.get("sentences")
            gap_type = candidate.get("gap_type")
            reason = candidate.get("reason")
            if not isinstance(numbers, list) or not numbers or not all(isinstance(item, int) for item in numbers):
                raise StageOutputError(f"{location}: sentence_numbers must be a non-empty integer list")
            if numbers != sorted(set(numbers)) or any(item < 0 or item >= len(expanded) for item in numbers):
                raise StageOutputError(f"{location}: sentence_numbers are duplicate, unordered, or out of range")
            expected = [expanded[item]["sentence"] for item in numbers]
            if sentences != expected:
                raise StageOutputError(f"{location}: sentences must exactly match numbered window coordinates")
            if gap_type not in {"N", "C", "A", "L"}:
                raise StageOutputError(f"{location}: gap_type must be N, C, A, or L")
            if not isinstance(reason, str):
                raise StageOutputError(f"{location}: reason must be a string")
            gaps.append({
                "gap_idx": len(gaps),
                "target_sentence_numbers": numbers,
                "target_sentence": sentences,
                "gap_reason": reason,
                "gap_type": gap_type,
            })
        rows.append({
            "schema_version": "1.0",
            "window_id": window_id,
            "source": record.get("source"),
            "range": record.get("range", "other"),
            "gaps": gaps,
        })
    return rows


def _window_id(record: dict[str, Any], seen: set[str] | None = None) -> str:
    value = record.get("window_id")
    if not isinstance(value, str) or not re.fullmatch(r"[ELR]_U\d{4,}", value):
        raise ValueError(f"invalid composed window_id: {value!r}")
    if seen is not None:
        if value in seen:
            raise ValueError(f"duplicate window_id: {value}")
        seen.add(value)
    return value


def _turn_list(record: dict[str, Any], field: str, window_id: str) -> list[dict[str, Any]]:
    turns = record.get(field)
    if not isinstance(turns, list) or not all(isinstance(item, dict) for item in turns):
        raise ValueError(f"{window_id}: {field} must be a list of turn objects")
    return turns
