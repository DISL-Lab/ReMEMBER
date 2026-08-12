"""Deterministic paper-threshold ensembles and text-free final rows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def _rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        window_id = row.get("window_id")
        if not isinstance(window_id, str) or window_id in result:
            raise ValueError("each model must have unique string window_id values")
        result[window_id] = row
    return result


def _common(rows_by_model: Mapping[str, Sequence[dict[str, Any]]]) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    if not rows_by_model:
        return [], {}
    if any(not isinstance(model, str) or not model.strip() for model in rows_by_model):
        raise ValueError("model names must be non-empty strings")
    model_priority = {"gemma": 0, "GLM": 1, "Qwen": 2}
    ordered = sorted(rows_by_model, key=lambda model: (model_priority.get(model, 99), model))
    indexed = {model: _rows(rows_by_model[model]) for model in ordered}
    common = set.intersection(*(set(rows) for rows in indexed.values()))
    return sorted(common), indexed


def ensemble_salient_votes(
    rows_by_model: Mapping[str, Sequence[dict[str, Any]]], threshold: int | None = None
) -> list[dict[str, Any]]:
    windows, indexed = _common(rows_by_model)
    if threshold is None and len(indexed) != 3:
        raise ValueError("paper salient ensemble requires exactly three models")
    required = len(indexed) if threshold is None else threshold
    if required <= 0 or required > len(indexed):
        raise ValueError("salient threshold must be between one and number of models")
    output: list[dict[str, Any]] = []
    priority = {"C": 0, "A": 1, "L": 2, "N": 3}

    def majority_gap_type(types: list[str]) -> str:
        counts = Counter(types)
        for gap_type in priority:
            if counts.get(gap_type, 0) >= 2:
                return gap_type
        return counts.most_common(1)[0][0]

    for window_id in windows:
        sentence_votes: dict[int, list[str]] = {}
        groups: set[tuple[int, ...]] = set()
        for model in indexed:
            voted_by_model: set[int] = set()
            gaps = indexed[model][window_id].get("gaps", [])
            if not isinstance(gaps, list):
                raise ValueError(f"{window_id}: gaps must be a list")
            for gap in gaps:
                if not isinstance(gap, dict):
                    raise ValueError(f"{window_id}: salient gaps must be objects")
                coordinates = gap.get("target_sentence_numbers")
                gap_type = gap.get("gap_type")
                if (
                    not isinstance(coordinates, list)
                    or not coordinates
                    or not all(isinstance(value, int) and value >= 0 for value in coordinates)
                    or coordinates != sorted(set(coordinates))
                    or gap_type not in priority
                ):
                    raise ValueError(f"{window_id}: invalid salient gap coordinates or type")
                duplicate = voted_by_model.intersection(coordinates)
                if duplicate:
                    raise ValueError(f"{window_id}: duplicate sentence vote within model {model}: {min(duplicate)}")
                voted_by_model.update(coordinates)
                for coordinate in coordinates:
                    sentence_votes.setdefault(coordinate, []).append(gap_type)
                groups.add(tuple(coordinates))
        surviving = {
            coordinate for coordinate, types in sentence_votes.items() if len(types) >= required
        }
        accepted: list[dict[str, Any]] = []
        for coordinates in sorted(groups, key=lambda item: (min(item), item)):
            if not set(coordinates) <= surviving:
                continue
            types = sentence_votes[coordinates[0]]
            gap_type = majority_gap_type(types)
            accepted.append({"gap_idx": len(accepted), "target_sentence_numbers": list(coordinates), "gap_type": gap_type})
        output.append({"schema_version": "1.0", "window_id": window_id, "gaps": accepted})
    return output


def ensemble_chunk_votes(
    rows_by_model: Mapping[str, Sequence[dict[str, Any]]], threshold: int | None = None
) -> list[dict[str, Any]]:
    windows, indexed = _common(rows_by_model)
    if threshold is None and len(indexed) != 3:
        raise ValueError("paper chunk ensemble requires exactly three models")
    required = len(indexed) if threshold is None else threshold
    if required <= 0 or required > len(indexed):
        raise ValueError("chunk threshold must be between one and number of models")
    output: list[dict[str, Any]] = []
    for window_id in windows:
        counts: Counter[tuple[int, int]] = Counter()
        for model in indexed:
            unique = {
                (chunk.get("start_turn"), chunk.get("end_turn"))
                for chunk in indexed[model][window_id].get("relevant_chunks", []) if isinstance(chunk, dict)
            }
            counts.update(unique)
        relevant = [
            {"start_turn": start, "end_turn": end}
            for (start, end), count in sorted(counts.items()) if count >= required
        ]
        output.append({"schema_version": "1.0", "window_id": window_id, "relevant_chunks": relevant})
    return output


def ensemble_votes(
    rows_by_model: Mapping[str, Sequence[dict[str, Any]]], threshold: int
) -> list[dict[str, Any]]:
    windows, indexed = _common(rows_by_model)
    if threshold <= 0 or threshold > len(indexed):
        raise ValueError("validation threshold must be between one and number of models")
    output: list[dict[str, Any]] = []
    for window_id in windows:
        first_model = next(iter(indexed))
        first_row = indexed[first_model][window_id]
        gap_metadata: dict[int, dict[str, Any]] = {}
        candidate_votes: dict[tuple[int, str], dict[str, int]] = {}
        for model in indexed:
            seen_candidates: set[tuple[int, str]] = set()
            for gap in indexed[model][window_id].get("gaps", []):
                gap_idx = gap.get("gap_idx")
                if not isinstance(gap_idx, int):
                    raise ValueError(f"{window_id}: gap_idx must be an integer")
                clean = {
                    "gap_idx": gap_idx,
                    "target_sentence_numbers": list(gap.get("target_sentence_numbers", [])),
                    "gap_type": gap.get("gap_type"),
                }
                if gap_idx in gap_metadata and gap_metadata[gap_idx] != clean:
                    raise ValueError(f"{window_id}: inconsistent gap metadata for gap_idx {gap_idx}")
                gap_metadata[gap_idx] = clean
                for item in gap.get("validated", []):
                    if not isinstance(item, dict) or not isinstance(item.get("source_id"), str):
                        raise ValueError(f"{window_id}: invalid validated evidence item")
                    candidate_key = (gap_idx, item["source_id"])
                    if candidate_key in seen_candidates:
                        raise ValueError(
                            f"{window_id}: duplicate candidate judgment for {item['source_id']} in model {model}"
                        )
                    seen_candidates.add(candidate_key)
                    candidate_votes.setdefault((gap_idx, item["source_id"]), {})[model] = int(item.get("validation") == 1)
        gaps: list[dict[str, Any]] = []
        for gap_idx in sorted(gap_metadata):
            evidence: list[dict[str, Any]] = []
            for (candidate_gap, source_id), votes in sorted(candidate_votes.items()):
                if candidate_gap != gap_idx:
                    continue
                complete_votes = {model: votes.get(model, 0) for model in indexed}
                vote_count = sum(complete_votes.values())
                if vote_count >= threshold:
                    evidence.append({
                        "source_id": source_id,
                        "status": "resolve",
                        "vote_count": vote_count,
                        "votes": dict(sorted(complete_votes.items())),
                        "validation": 1,
                    })
            gaps.append({**gap_metadata[gap_idx], "evidence": evidence})
        output.append({
            "schema_version": "1.0",
            "window_id": window_id,
            "source": first_row.get("source"),
            "range": first_row.get("range", "other"),
            "gaps": gaps,
        })
    return output
