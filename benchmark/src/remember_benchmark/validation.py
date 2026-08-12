"""Exact-source-ID evidence validation stage."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .evidence import _index
from .providers import StageProvider
from .salient import StageOutputError, _parse_response, _window_id


def run_validation(
    windows: Sequence[dict[str, Any]],
    evidence_rows: Sequence[dict[str, Any]],
    provider: StageProvider,
    prompt: str,
) -> list[dict[str, Any]]:
    evidence_by_id = _index(evidence_rows, "evidence")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for window in windows:
        window_id = _window_id(window, seen)
        if window_id not in evidence_by_id:
            raise ValueError(f"{window_id}: missing evidence stage row")
        gaps: list[dict[str, Any]] = []
        for gap in evidence_by_id[window_id].get("gaps", []):
            if not isinstance(gap, dict):
                raise ValueError(f"{window_id}: evidence gap must be an object")
            candidates = [item for item in gap.get("evidence", []) if isinstance(item, dict) and item.get("status") == "resolve"]
            valid_ids: list[Any] = []
            if candidates:
                rendered = prompt.format(
                    candidates=json.dumps(candidates, ensure_ascii=False, sort_keys=True),
                    gap=json.dumps({key: value for key, value in gap.items() if key != "evidence"}, ensure_ascii=False, sort_keys=True),
                    window=json.dumps(window.get("turns", []), ensure_ascii=False),
                )
                valid_ids = _parse_response(provider.complete(rendered), "validation", window_id, "valid_source_ids")
                if not all(isinstance(value, str) for value in valid_ids) or len(valid_ids) != len(set(valid_ids)):
                    raise StageOutputError(f"validation {window_id}: valid_source_ids must be unique strings")
                candidate_ids = {item["source_id"] for item in candidates}
                if not set(valid_ids) <= candidate_ids:
                    raise StageOutputError(f"validation {window_id}: response contains an unknown source_id")
            copied = {key: value for key, value in gap.items() if key != "evidence"}
            copied["validated"] = [
                {**item, "validation": int(item["source_id"] in valid_ids)} for item in candidates
            ]
            gaps.append(copied)
        source_row = evidence_by_id[window_id]
        output.append({
            "schema_version": "1.0", "window_id": window_id,
            "source": source_row.get("source"), "range": source_row.get("range", "other"), "gaps": gaps,
        })
    return output
