"""Sentence-coordinate evidence resolution stage."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .providers import StageProvider
from .salient import StageOutputError, _parse_response, _sentences, _window_id


def _index(rows: Sequence[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        window_id = _window_id(row)
        if window_id in result:
            raise ValueError(f"duplicate {label} window_id: {window_id}")
        result[window_id] = row
    return result


def _history_index(history: Sequence[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for turn_index, turn in enumerate(history):
        for sentence_index, sentence in enumerate(_sentences(str(turn.get("dialogue", ""))), 1):
            result[f"T{turn_index}-S{sentence_index}"] = sentence
    return result


def run_evidence(
    windows: Sequence[dict[str, Any]],
    salient_rows: Sequence[dict[str, Any]],
    filtered_rows: Sequence[dict[str, Any]],
    provider: StageProvider,
    prompt: str,
) -> list[dict[str, Any]]:
    salient = _index(salient_rows, "salient")
    filtered = _index(filtered_rows, "chunk filter")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for window in windows:
        window_id = _window_id(window, seen)
        if window_id not in salient or window_id not in filtered:
            raise ValueError(f"{window_id}: missing salient or chunk-filter stage row")
        history = window.get("history")
        if not isinstance(history, list):
            raise ValueError(f"{window_id}: history must be a list")
        sentence_index = _history_index(history)
        allowed_turns: set[int] = set()
        for chunk in filtered[window_id].get("relevant_chunks", []):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("start_turn"), int) or not isinstance(chunk.get("end_turn"), int):
                raise ValueError(f"{window_id}: invalid relevant chunk boundary")
            allowed_turns.update(range(chunk["start_turn"], chunk["end_turn"] + 1))
        allowed_index = {
            source_id: sentence for source_id, sentence in sentence_index.items()
            if int(source_id[1:source_id.index("-")]) in allowed_turns
        }
        gaps: list[dict[str, Any]] = []
        for gap in salient[window_id].get("gaps", []):
            if not isinstance(gap, dict):
                raise ValueError(f"{window_id}: salient gap must be an object")
            copied = dict(gap)
            copied["evidence"] = []
            if gap.get("gap_type") != "N":
                rendered = prompt.format(
                    history=json.dumps(allowed_index, ensure_ascii=False, sort_keys=True),
                    gap=json.dumps(gap, ensure_ascii=False, sort_keys=True),
                    window=json.dumps(window.get("turns", []), ensure_ascii=False),
                )
                evidence = _parse_response(provider.complete(rendered), "evidence", window_id, "evidence")
                for item_index, item in enumerate(evidence):
                    location = f"evidence {window_id} item {item_index}"
                    if not isinstance(item, dict):
                        raise StageOutputError(f"{location}: must be an object")
                    source_id = item.get("source_id")
                    if not isinstance(source_id, str) or not re.fullmatch(r"T\d+-S\d+", source_id) or source_id not in allowed_index:
                        raise StageOutputError(f"{location}: invalid or filtered-out source_id")
                    if item.get("sentence") != allowed_index[source_id]:
                        raise StageOutputError(f"{location}: sentence does not match source_id coordinate")
                    if item.get("status") not in {"resolve", "not_resolve"} or not isinstance(item.get("reasoning"), str):
                        raise StageOutputError(f"{location}: invalid status or reasoning")
                    copied["evidence"].append({
                        "source_id": source_id,
                        "sentence": item["sentence"],
                        "reasoning": item["reasoning"],
                        "status": item["status"],
                    })
            gaps.append(copied)
        output.append({
            "schema_version": "1.0", "window_id": window_id,
            "source": window.get("source"), "range": window.get("range", "other"), "gaps": gaps,
        })
    return output
