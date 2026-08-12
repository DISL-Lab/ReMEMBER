"""Create the text-free ReMEMber annotation release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .io import atomic_write_jsonl, fsync_parent_directory, iter_jsonl


SCHEMA_VERSION = "1.0"
SOURCE_BY_PREFIX = {"E": "evermembench", "L": "locomo", "R": "realtalk"}
FORBIDDEN_KEYS = frozenset({"target_sentence", "sentence", "gap_reason", "reasoning"})
WINDOW_KEYS = frozenset({"schema_version", "window_id", "source", "range", "gaps"})
GAP_KEYS = frozenset({"gap_idx", "target_sentence_numbers", "gap_type", "evidence"})
EVIDENCE_KEYS = frozenset({"source_id", "status", "vote_count", "votes", "validation"})
VOTE_KEYS = frozenset({"gemma", "GLM", "Qwen"})


def _read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        ids = [line.strip() for line in handle]
    if any(not window_id for window_id in ids):
        raise ValueError("split contains an empty window ID")
    return ids


def _source_for_window(window_id: Any) -> str:
    if not isinstance(window_id, str) or "_" not in window_id:
        raise ValueError(f"unknown source prefix for window ID: {window_id!r}")
    prefix = window_id.split("_", 1)[0]
    try:
        return SOURCE_BY_PREFIX[prefix]
    except KeyError as error:
        raise ValueError(f"unknown source prefix for window ID: {window_id!r}") from error


def _duplicate(values: Sequence[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def sanitize_annotations(
    rcv_rows: Iterable[dict[str, Any]], ordered_window_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Project canonical RCV records into the text-free public annotation schema."""
    split_ids = list(ordered_window_ids)
    duplicate = _duplicate(split_ids)
    if duplicate is not None:
        raise ValueError(f"duplicate split ID: {duplicate}")
    for window_id in split_ids:
        _source_for_window(window_id)
    split_set = set(split_ids)
    grouped: dict[str, list[dict[str, Any]]] = {window_id: [] for window_id in split_ids}
    ranges: dict[str, str] = {}

    for row in rcv_rows:
        window_id = row.get("window_id")
        if window_id not in split_set or not row.get("target_sentence"):
            continue
        _source_for_window(window_id)
        row_range = row.get("range")
        if not isinstance(row_range, str):
            raise ValueError(f"missing or invalid range for window ID: {window_id}")
        if window_id in ranges and ranges[window_id] != row_range:
            raise ValueError(f"inconsistent range for window ID: {window_id}")
        ranges[window_id] = row_range
        evidence = [
            {
                "source_id": item["source_id"],
                "status": item["status"],
                "vote_count": item["vote_count"],
                "votes": item["votes"],
                "validation": item["validation"],
            }
            for item in row.get("validated", [])
            if item.get("validation") == 1
        ]
        grouped[window_id].append(
            {
                "gap_idx": row["gap_idx"],
                "target_sentence_numbers": row["sentence_numbers"],
                "gap_type": row["gap_type"],
                "evidence": evidence,
            }
        )

    missing = [window_id for window_id in split_ids if window_id not in ranges]
    if missing:
        raise ValueError(f"split window has no non-empty RCV gap: {missing[0]}")
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "window_id": window_id,
            "source": _source_for_window(window_id),
            "range": ranges[window_id],
            "gaps": grouped[window_id],
        }
        for window_id in split_ids
    ]


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def _require_keys(value: dict[str, Any], expected: frozenset[str], location: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"invalid object at {location}")
    keys = set(value)
    if keys != expected:
        unexpected = sorted(keys - expected)
        missing = sorted(expected - keys)
        raise ValueError(f"invalid keys at {location}: unexpected={unexpected}, missing={missing}")


def validate_annotations(
    rows: Sequence[dict[str, Any]],
    ordered_window_ids: Sequence[str],
    expected_counts: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Verify structure, text exclusion, ordered membership, and optional counts."""
    split_ids = list(ordered_window_ids)
    duplicate = _duplicate(split_ids)
    if duplicate is not None:
        raise ValueError(f"duplicate split ID: {duplicate}")
    for key in _walk_keys(rows):
        if key in FORBIDDEN_KEYS:
            raise ValueError(f"forbidden key in public annotations: {key}")

    row_ids: list[str] = []
    gap_count = 0
    evidence_count = 0
    for row_index, row in enumerate(rows):
        _require_keys(row, WINDOW_KEYS, f"window[{row_index}]")
        window_id = row["window_id"]
        if not isinstance(window_id, str):
            raise ValueError(f"window[{row_index}] has a non-string window_id")
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"window[{row_index}] has unsupported schema version")
        if row["source"] != _source_for_window(window_id):
            raise ValueError(f"window[{row_index}] source does not match window ID")
        if not isinstance(row["range"], str):
            raise ValueError(f"window[{row_index}] has a non-string range")
        if not isinstance(row["gaps"], list):
            raise ValueError(f"invalid gaps at window[{row_index}]")
        row_ids.append(window_id)
        for gap_index, gap in enumerate(row["gaps"]):
            _require_keys(gap, GAP_KEYS, f"window[{row_index}].gaps[{gap_index}]")
            if type(gap["gap_idx"]) is not int or not isinstance(gap["gap_type"], str):
                raise ValueError(f"invalid gap fields at window[{row_index}].gaps[{gap_index}]")
            if not isinstance(gap["target_sentence_numbers"], list) or not all(
                type(number) is int for number in gap["target_sentence_numbers"]
            ):
                raise ValueError(f"invalid target sentence numbers at window[{row_index}].gaps[{gap_index}]")
            if not isinstance(gap["evidence"], list):
                raise ValueError(f"invalid evidence at window[{row_index}].gaps[{gap_index}]")
            gap_count += 1
            for evidence_index, evidence in enumerate(gap["evidence"]):
                _require_keys(
                    evidence, EVIDENCE_KEYS, f"window[{row_index}].gaps[{gap_index}].evidence[{evidence_index}]"
                )
                _require_keys(evidence["votes"], VOTE_KEYS, "evidence votes")
                if (
                    not isinstance(evidence["source_id"], str)
                    or not isinstance(evidence["status"], str)
                    or type(evidence["vote_count"]) is not int
                    or type(evidence["validation"]) is not int
                    or evidence["validation"] != 1
                    or not all(type(vote) is int for vote in evidence["votes"].values())
                ):
                    raise ValueError("invalid evidence fields")
                evidence_count += 1

    duplicate = _duplicate(row_ids)
    if duplicate is not None:
        raise ValueError(f"duplicate emitted window: {duplicate}")
    if row_ids != split_ids:
        raise ValueError("emitted windows do not exactly match the ordered split")
    stats = {"windows": len(rows), "gaps": gap_count, "evidence": evidence_count}
    if expected_counts is not None and tuple(stats.values()) != expected_counts:
        raise ValueError(f"count mismatch: expected {expected_counts}, got {tuple(stats.values())}")
    return stats


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)
    fsync_parent_directory(path)


def _schema() -> dict[str, Any]:
    evidence = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(EVIDENCE_KEYS),
        "properties": {
            "source_id": {"type": "string"},
            "status": {"type": "string"},
            "vote_count": {"type": "integer"},
            "votes": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(VOTE_KEYS),
                "properties": {name: {"type": "integer"} for name in sorted(VOTE_KEYS)},
            },
            "validation": {"const": 1},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ReMEMber text-free annotations",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(WINDOW_KEYS),
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "window_id": {"type": "string"},
            "source": {"enum": sorted(SOURCE_BY_PREFIX.values())},
            "range": {"type": "string"},
            "gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(GAP_KEYS),
                    "properties": {
                        "gap_idx": {"type": "integer"},
                        "target_sentence_numbers": {"type": "array", "items": {"type": "integer"}},
                        "gap_type": {"type": "string"},
                        "evidence": {"type": "array", "items": evidence},
                    },
                },
            },
        },
    }


def _prepare_annotations(rcv_path: Path, split_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_ids = _read_ids(split_path)
    rows = sanitize_annotations(iter_jsonl(rcv_path), split_ids)
    stats = validate_annotations(rows, split_ids)
    return rows, stats


def _write_export(
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    split_path: Path,
    output_path: Path,
    metadata_path: Path,
    schema_path: Path,
) -> None:
    atomic_write_jsonl(output_path, rows)
    _atomic_write_json(schema_path, _schema())
    _atomic_write_json(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "counts": stats,
            "artifacts": {
                "annotations": {
                    "path": "annotations/remember_900_annotations.jsonl",
                    "sha256": _sha256(output_path),
                },
                "split": {
                    "path": "splits/main_900.txt",
                    "sha256": _sha256(split_path),
                },
            },
        },
    )


def export_annotations(
    rcv_path: Path, split_path: Path, output_path: Path, metadata_path: Path, schema_path: Path
) -> dict[str, Any]:
    """Write annotations, metadata, and schema from canonical RCV and ordered IDs."""
    rows, stats = _prepare_annotations(rcv_path, split_path)
    _write_export(rows, stats, split_path, output_path, metadata_path, schema_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rcv", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--expected-windows", type=int)
    parser.add_argument("--expected-gaps", type=int)
    parser.add_argument("--expected-evidence", type=int)
    args = parser.parse_args()
    rows, stats = _prepare_annotations(args.rcv, args.split)
    expected = (args.expected_windows, args.expected_gaps, args.expected_evidence)
    if any(value is not None for value in expected):
        if any(value is None for value in expected):
            parser.error("expected counts must be supplied together")
        if tuple(stats.values()) != expected:
            raise ValueError(f"count mismatch: expected {expected}, got {tuple(stats.values())}")
    _write_export(rows, stats, args.split, args.output, args.metadata, args.schema)
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
