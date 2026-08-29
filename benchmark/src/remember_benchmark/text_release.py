"""Construct and semantically validate the text-bearing v2 benchmark."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from .evidence import _history_index
from .release import RANGE_VALUES, SOURCE_BY_PREFIX
from .salient import expand_window_sentences


BENCHMARK_SCHEMA_VERSION = "2.0"
BENCHMARK_SHARD_COUNT = 10
BENCHMARK_ROWS_PER_SHARD = 90

SOURCE_VALUES = ("evermembench", "locomo", "realtalk")
BENCHMARK_RANGE_VALUES = tuple(value for value in RANGE_VALUES if value != "other")
VOTE_KEYS = frozenset({"gemma", "GLM", "Qwen"})
WINDOW_ID_PATTERN = re.compile(r"[ELR]_U\d{4,}")

WINDOW_KEYS = frozenset({
    "schema_version", "window_id", "source", "domain", "range", "tok_start", "tok_end",
    "history", "turns", "salient_utterances", "gaps",
})
TURN_KEYS = frozenset({"turn_index", "time", "speaker", "dialogue"})
SALIENT_KEYS = frozenset({"salient_idx", "sentence_numbers", "sentences", "gap_type", "vote_count"})
GAP_KEYS = frozenset({
    "gap_idx", "target_sentence_numbers", "target_sentence", "gap_type", "gap_reason", "evidence",
})
EVIDENCE_KEYS = frozenset({
    "source_id", "sentence", "status", "validation", "vote_count", "votes",
})


def _source_for_window(window_id: Any) -> str:
    if not isinstance(window_id, str) or WINDOW_ID_PATTERN.fullmatch(window_id) is None:
        raise ValueError(f"unknown source prefix for window ID: {window_id!r}")
    return SOURCE_BY_PREFIX[window_id[0]]


def _duplicate(values: Sequence[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _normalize_turns(value: Any, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    rows = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{location}[{index}] must be an object")
        clean = {"turn_index": index}
        for field in ("time", "speaker", "dialogue"):
            field_value = item.get(field)
            if not isinstance(field_value, str):
                raise ValueError(f"{location}[{index}].{field} must be a string")
            if not field_value:
                raise ValueError(f"{location}[{index}].{field} must be non-empty")
            clean[field] = field_value
        rows.append(clean)
    return rows


def _coordinates(value: Any, location: str, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or not all(type(number) is int and number >= 0 for number in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{location}: {label} coordinates must be ordered and unique")
    return value


def _resolve_target(turns: list[dict[str, Any]], numbers: Any, location: str) -> list[str]:
    coordinates = _coordinates(numbers, location, "target")
    expanded = expand_window_sentences(turns)
    try:
        return [expanded[number]["sentence"] for number in coordinates]
    except IndexError as exc:
        raise ValueError(f"{location}: target coordinate is out of range") from exc


def _resolve_evidence(history: list[dict[str, Any]], source_id: Any, location: str) -> str:
    if not isinstance(source_id, str):
        raise ValueError(f"{location}: evidence coordinate is invalid")
    value = _history_index(history).get(source_id)
    if value is None:
        raise ValueError(f"{location}: evidence coordinate is invalid")
    return value


def _selected_index(
    rows: Iterable[dict[str, Any]], split_set: set[str], label: str
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} row must be an object")
        window_id = row.get("window_id")
        if window_id not in split_set:
            continue
        if window_id in selected:
            raise ValueError(f"duplicate {label} window_id: {window_id}")
        selected[window_id] = row
    return selected


def _selected_rcv(
    rows: Iterable[dict[str, Any]], split_set: set[str]
) -> dict[str, list[dict[str, Any]]]:
    selected = {window_id: [] for window_id in split_set}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("RCV row must be an object")
        window_id = row.get("window_id")
        if window_id in split_set:
            selected[window_id].append(row)
    return selected


def _require_selected(indexed: dict[str, Any], split_ids: list[str], label: str) -> None:
    for window_id in split_ids:
        if window_id not in indexed or not indexed[window_id]:
            raise ValueError(f"{window_id}: missing {label} row")


def _window_range(window_id: str, source: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    value: str | None = None
    for row_index, row in enumerate(rows):
        row_range = row.get("range")
        if row_range not in BENCHMARK_RANGE_VALUES:
            raise ValueError(f"{window_id} RCV[{row_index}]: invalid range")
        if value is not None and row_range != value:
            raise ValueError(f"{window_id}: inconsistent range")
        value = row_range
    if value is None:
        raise ValueError(f"{window_id}: missing RCV row")
    if "range" in source and source["range"] != value:
        raise ValueError(f"{window_id}: pipeline range does not match RCV range")
    return value


def _build_evidence(
    history: list[dict[str, Any]], raw_items: Any, location: str
) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        raise ValueError(f"{location}: validated must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item_index, item in enumerate(raw_items):
        item_location = f"{location}.validated[{item_index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_location} must be an object")
        if item.get("validation") != 1:
            continue
        source_id = item.get("source_id")
        sentence = _resolve_evidence(history, source_id, item_location)
        if source_id in seen:
            raise ValueError(f"{item_location}: duplicate evidence source_id")
        seen.add(source_id)
        if "sentence" in item and item["sentence"] != sentence:
            raise ValueError(f"{item_location}: evidence text mismatch")
        result.append({
            "source_id": source_id,
            "sentence": sentence,
            "status": item.get("status"),
            "validation": item.get("validation"),
            "vote_count": item.get("vote_count"),
            "votes": item.get("votes"),
        })
    return result


def _build_gaps(
    history: list[dict[str, Any]], turns: list[dict[str, Any]], rows: list[dict[str, Any]], window_id: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for row_index, row in enumerate(rows):
        raw_target = row.get("target_sentence")
        if isinstance(raw_target, str) and not raw_target.strip():
            continue
        location = f"{window_id} RCV[{row_index}]"
        gap_idx = row.get("gap_idx")
        if type(gap_idx) is not int or gap_idx < 0:
            raise ValueError(f"{location}: gap_idx must be a non-negative integer")
        if gap_idx in seen_indices:
            raise ValueError(f"{location}: duplicate gap_idx {gap_idx}")
        seen_indices.add(gap_idx)
        numbers = row.get("sentence_numbers")
        resolved = _resolve_target(turns, numbers, location)
        expected_target = " ".join(resolved)
        if isinstance(raw_target, str):
            if raw_target != expected_target:
                raise ValueError(f"{location}: target text mismatch")
        elif isinstance(raw_target, list):
            if raw_target != resolved:
                raise ValueError(f"{location}: target text mismatch")
        else:
            raise ValueError(f"{location}: target_sentence must be a string or list of strings")
        result.append({
            "gap_idx": gap_idx,
            "target_sentence_numbers": list(numbers),
            "target_sentence": expected_target,
            "gap_type": row.get("gap_type"),
            "gap_reason": row.get("gap_reason"),
            "evidence": _build_evidence(history, row.get("validated"), location),
        })
    return result


def _build_salient(
    raw: dict[str, Any], turns: list[dict[str, Any]], window_id: str
) -> list[dict[str, Any]]:
    pipeline_shape = "gap_analysis" not in raw
    groups = raw.get("gaps" if pipeline_shape else "gap_analysis")
    if not isinstance(groups, list):
        raise ValueError(f"{window_id}: salient gaps must be a list")
    result: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        location = f"{window_id} salient[{group_index}]"
        if not isinstance(group, dict):
            raise ValueError(f"{location} must be an object")
        numbers = group.get("target_sentence_numbers" if pipeline_shape else "sentence_numbers")
        sentences = group.get("target_sentence" if pipeline_shape else "sentences")
        resolved = _resolve_target(turns, numbers, location)
        if not isinstance(sentences, list) or sentences != resolved:
            raise ValueError(f"{location}: salient text mismatch")
        for number, sentence in zip(numbers, resolved):
            result.append({
                "salient_idx": len(result),
                "sentence_numbers": [number],
                "sentences": [sentence],
                "gap_type": group.get("gap_type"),
                "vote_count": group.get("vote_count"),
            })
    return result


def build_benchmark_records(
    source_rows: Iterable[dict[str, Any]],
    rcv_rows: Iterable[dict[str, Any]],
    salient_rows: Iterable[dict[str, Any]],
    ordered_window_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Join selected source, RCV, and salient rows into ordered v2 records."""
    split_ids = list(ordered_window_ids)
    duplicate = _duplicate(split_ids)
    if duplicate is not None:
        raise ValueError(f"duplicate split ID: {duplicate}")
    for window_id in split_ids:
        _source_for_window(window_id)
    split_set = set(split_ids)
    sources = _selected_index(source_rows, split_set, "source")
    rcv = _selected_rcv(rcv_rows, split_set)
    salient = _selected_index(salient_rows, split_set, "salient")
    _require_selected(sources, split_ids, "source")
    _require_selected(rcv, split_ids, "RCV")
    _require_selected(salient, split_ids, "salient")

    result: list[dict[str, Any]] = []
    for window_id in split_ids:
        raw_source = sources[window_id]
        expected_source = _source_for_window(window_id)
        domain = raw_source.get("domain", raw_source.get("source"))
        if domain != expected_source:
            raise ValueError(f"{window_id}: domain/source does not match window ID")
        tok_start = raw_source.get("tok_start")
        tok_end = raw_source.get("tok_end")
        if (
            type(tok_start) is not int or type(tok_end) is not int
            or tok_start < 0 or tok_start >= tok_end
        ):
            raise ValueError(f"{window_id}: invalid token bounds")
        history = _normalize_turns(raw_source.get("history"), f"{window_id}.history")
        turns = _normalize_turns(raw_source.get("turns"), f"{window_id}.turns")
        result.append({
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "window_id": window_id,
            "source": expected_source,
            "domain": expected_source,
            "range": _window_range(window_id, raw_source, rcv[window_id]),
            "tok_start": tok_start,
            "tok_end": tok_end,
            "history": history,
            "turns": turns,
            "salient_utterances": _build_salient(salient[window_id], turns, window_id),
            "gaps": _build_gaps(history, turns, rcv[window_id], window_id),
        })
    validate_benchmark_records(result, split_ids)
    return result


def build_benchmark_records_from_pipeline(
    windows: Iterable[dict[str, Any]],
    private_rcv: Iterable[dict[str, Any]],
    merged_salient: Iterable[dict[str, Any]],
    ordered_window_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Normalize pipeline-shaped inputs through the canonical record builder."""
    return build_benchmark_records(windows, private_rcv, merged_salient, ordered_window_ids)


def _require_keys(value: Any, expected: frozenset[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: must be an object")
    if set(value) != expected:
        raise ValueError(
            f"invalid keys at {location}: "
            f"unexpected={sorted(set(value) - expected)}, missing={sorted(expected - set(value))}"
        )
    return value


def _validate_turns(value: Any, location: str) -> list[dict[str, Any]]:
    normalized = _normalize_turns(value, location)
    for index, item in enumerate(value):
        _require_keys(item, TURN_KEYS, f"{location}[{index}]")
        if item["turn_index"] != index:
            raise ValueError(f"{location}[{index}].turn_index does not match position")
    return normalized


def _nonempty_string(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")


def validate_benchmark_records(
    rows: Sequence[dict[str, Any]],
    ordered_window_ids: Sequence[str],
    expected_counts: tuple[int, int, int] | None = None,
    expected_salient_count: int | None = None,
) -> dict[str, Any]:
    """Validate exact structure plus every in-record coordinate-to-text binding."""
    split_ids = list(ordered_window_ids)
    duplicate = _duplicate(split_ids)
    if duplicate is not None:
        raise ValueError(f"duplicate split ID: {duplicate}")
    for window_id in split_ids:
        _source_for_window(window_id)

    row_ids: list[str] = []
    gap_count = 0
    evidence_count = 0
    salient_count = 0
    by_source: dict[str, int] = {}
    by_range: dict[str, int] = {}
    for row_index, raw_row in enumerate(rows):
        location = f"window[{row_index}]"
        row = _require_keys(raw_row, WINDOW_KEYS, location)
        if row["schema_version"] != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(f"{location}: unsupported schema version")
        window_id = row["window_id"]
        expected_source = _source_for_window(window_id)
        if row["source"] != expected_source or row["domain"] != expected_source:
            raise ValueError(f"{location}: source/domain does not match window ID")
        if row["range"] not in BENCHMARK_RANGE_VALUES:
            raise ValueError(f"{location}: invalid range")
        if (
            type(row["tok_start"]) is not int or type(row["tok_end"]) is not int
            or row["tok_start"] < 0 or row["tok_start"] >= row["tok_end"]
        ):
            raise ValueError(f"{location}: invalid token bounds")
        history = _validate_turns(row["history"], f"{location}.history")
        turns = _validate_turns(row["turns"], f"{location}.turns")
        row_ids.append(window_id)
        by_source[expected_source] = by_source.get(expected_source, 0) + 1
        by_range[row["range"]] = by_range.get(row["range"], 0) + 1

        if not isinstance(row["salient_utterances"], list):
            raise ValueError(f"{location}.salient_utterances must be a list")
        for salient_index, raw_salient in enumerate(row["salient_utterances"]):
            salient_location = f"{location}.salient_utterances[{salient_index}]"
            item = _require_keys(raw_salient, SALIENT_KEYS, salient_location)
            if item["salient_idx"] != salient_index:
                raise ValueError(f"{salient_location}.salient_idx does not match position")
            numbers = _coordinates(item["sentence_numbers"], salient_location, "salient")
            if len(numbers) != 1:
                raise ValueError(f"{salient_location}: salient coordinate must identify one sentence")
            resolved = _resolve_target(turns, numbers, salient_location)
            if item["sentences"] != resolved:
                raise ValueError(f"{salient_location}: salient text mismatch")
            if item["gap_type"] not in {"N", "C", "A", "L"}:
                raise ValueError(f"{salient_location}: invalid gap_type")
            if type(item["vote_count"]) is not int or not 1 <= item["vote_count"] <= 3:
                raise ValueError(f"{salient_location}: invalid salient vote_count")
            salient_count += 1

        if not isinstance(row["gaps"], list):
            raise ValueError(f"{location}.gaps must be a list")
        seen_gap_indices: set[int] = set()
        for gap_index, raw_gap in enumerate(row["gaps"]):
            gap_location = f"{location}.gaps[{gap_index}]"
            gap = _require_keys(raw_gap, GAP_KEYS, gap_location)
            gap_idx = gap["gap_idx"]
            if type(gap_idx) is not int or gap_idx < 0:
                raise ValueError(f"{gap_location}: invalid gap_idx")
            if gap_idx in seen_gap_indices:
                raise ValueError(f"{gap_location}: duplicate gap_idx")
            seen_gap_indices.add(gap_idx)
            numbers = _coordinates(gap["target_sentence_numbers"], gap_location, "target")
            expected_target = " ".join(_resolve_target(turns, numbers, gap_location))
            if gap["target_sentence"] != expected_target:
                raise ValueError(f"{gap_location}: target text mismatch")
            if gap["gap_type"] not in {"A", "C", "L"}:
                raise ValueError(f"{gap_location}: invalid gap_type")
            _nonempty_string(gap["gap_reason"], f"{gap_location}.gap_reason")
            if not isinstance(gap["evidence"], list):
                raise ValueError(f"{gap_location}.evidence must be a list")
            seen_source_ids: set[str] = set()
            for evidence_index, raw_evidence in enumerate(gap["evidence"]):
                evidence_location = f"{gap_location}.evidence[{evidence_index}]"
                evidence = _require_keys(raw_evidence, EVIDENCE_KEYS, evidence_location)
                source_id = evidence["source_id"]
                sentence = _resolve_evidence(history, source_id, evidence_location)
                if source_id in seen_source_ids:
                    raise ValueError(f"{evidence_location}: duplicate evidence source_id")
                seen_source_ids.add(source_id)
                if evidence["sentence"] != sentence:
                    raise ValueError(f"{evidence_location}: evidence text mismatch")
                votes = _require_keys(evidence["votes"], VOTE_KEYS, f"{evidence_location}.votes")
                if any(type(vote) is not int or vote not in {0, 1} for vote in votes.values()):
                    raise ValueError(f"{evidence_location}: invalid evidence votes")
                vote_count = sum(votes.values())
                if (
                    evidence["status"] != "resolve"
                    or evidence["validation"] != 1
                    or type(evidence["validation"]) is not int
                    or type(evidence["vote_count"]) is not int
                    or evidence["vote_count"] != vote_count
                    or vote_count < 2
                ):
                    raise ValueError(f"{evidence_location}: invalid evidence fields")
                evidence_count += 1
            gap_count += 1

    if row_ids != split_ids:
        raise ValueError("emitted windows do not exactly match the ordered split")
    observed_counts = (len(rows), gap_count, evidence_count)
    if expected_counts is not None and observed_counts != expected_counts:
        raise ValueError(f"count mismatch: expected {expected_counts}, got {observed_counts}")
    if expected_salient_count is not None and salient_count != expected_salient_count:
        raise ValueError(
            f"salient count mismatch: expected {expected_salient_count}, got {salient_count}"
        )
    return {
        "windows": len(rows),
        "gaps": gap_count,
        "evidence": evidence_count,
        "salient_utterances": salient_count,
        "by_source": by_source,
        "by_range": by_range,
    }


def _string_schema() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(properties),
        "properties": properties,
    }


def benchmark_schema() -> dict[str, Any]:
    """Return the closed JSON Schema for a single v2 benchmark record."""
    turn = _object_schema({
        "turn_index": {"type": "integer", "minimum": 0},
        "time": _string_schema(),
        "speaker": _string_schema(),
        "dialogue": _string_schema(),
    })
    votes = _object_schema({
        name: {"type": "integer", "enum": [0, 1]} for name in ("gemma", "GLM", "Qwen")
    })
    evidence = _object_schema({
        "source_id": {"type": "string", "pattern": r"^T(0|[1-9]\d*)-S([1-9]\d*)$"},
        "sentence": _string_schema(),
        "status": {"const": "resolve"},
        "validation": {"const": 1},
        "vote_count": {"type": "integer", "minimum": 2, "maximum": 3},
        "votes": votes,
    })
    coordinate = {"type": "integer", "minimum": 0}
    salient = _object_schema({
        "salient_idx": {"type": "integer", "minimum": 0},
        "sentence_numbers": {
            "type": "array", "minItems": 1, "maxItems": 1, "uniqueItems": True,
            "items": coordinate,
        },
        "sentences": {
            "type": "array", "minItems": 1, "maxItems": 1,
            "items": _string_schema(),
        },
        "gap_type": {"enum": ["N", "C", "A", "L"]},
        "vote_count": {"type": "integer", "minimum": 1, "maximum": 3},
    })
    gap = _object_schema({
        "gap_idx": {"type": "integer", "minimum": 0},
        "target_sentence_numbers": {
            "type": "array", "minItems": 1, "uniqueItems": True, "items": coordinate,
        },
        "target_sentence": _string_schema(),
        "gap_type": {"enum": ["A", "C", "L"]},
        "gap_reason": _string_schema(),
        "evidence": {"type": "array", "items": evidence},
    })
    properties = {
        "schema_version": {"const": BENCHMARK_SCHEMA_VERSION},
        "window_id": {"type": "string", "pattern": r"^[ELR]_U\d{4,}$"},
        "source": {"enum": list(SOURCE_VALUES)},
        "domain": {"enum": list(SOURCE_VALUES)},
        "range": {"enum": list(BENCHMARK_RANGE_VALUES)},
        "tok_start": {"type": "integer", "minimum": 0},
        "tok_end": {"type": "integer", "minimum": 1},
        "history": {"type": "array", "items": turn},
        "turns": {"type": "array", "items": turn},
        "salient_utterances": {"type": "array", "items": salient},
        "gaps": {"type": "array", "items": gap},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ReMEMber text-bearing benchmark record",
        **_object_schema(properties),
    }
