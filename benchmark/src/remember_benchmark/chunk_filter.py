"""Whole-turn 128/32 chunk filtering stage."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from .providers import StageProvider
from .salient import StageOutputError, _parse_response, _turn_list, _window_id


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class Cl100kCounter:
    def __init__(self) -> None:
        self._encoding = None

    def count(self, text: str) -> int:
        if self._encoding is None:
            try:
                import tiktoken
            except ImportError as exc:
                raise RuntimeError("install tiktoken to use the paper token counter") from exc
            self._encoding = tiktoken.get_encoding("cl100k_base")
        return len(self._encoding.encode(text))


def _format_turn(turn: dict[str, Any]) -> str:
    return f"[{turn.get('time', '')}] {turn.get('speaker', '')}: {turn.get('dialogue', '')}"


def _next_chunk_start(
    *, start: int, end: int, counts: Sequence[int], overlap_tokens: int
) -> int:
    """Return the trailing whole-turn overlap start for an emitted [start,end).

    A caller that has not yet appended the new turn uses this retained suffix
    exactly once; the canonical builder then appends the new turn regardless
    of whether overlap plus that turn exceeds ``max_tokens``.
    """
    overlap_start = end
    overlap_total = 0
    for index in range(end - 1, start - 1, -1):
        if overlap_total + counts[index] > overlap_tokens:
            break
        overlap_start = index
        overlap_total += counts[index]
    return overlap_start


def build_history_chunks(
    history: Sequence[dict[str, Any]],
    counter: TokenCounter,
    max_tokens: int = 128,
    overlap_tokens: int = 32,
) -> list[dict[str, Any]]:
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("require max_tokens > overlap_tokens >= 0")
    counts = [counter.count(_format_turn(turn)) for turn in history]
    if any(not isinstance(value, int) or value <= 0 for value in counts):
        raise ValueError("token counter must return positive integers")
    raw_chunks: list[tuple[int, int, int]] = []
    current_start = 0
    current_indices: list[int] = []
    current_total = 0
    for index, count in enumerate(counts):
        if current_indices and current_total + count > max_tokens:
            end = current_indices[-1] + 1
            raw_chunks.append((current_start, end, current_total))
            overlap_start = _next_chunk_start(
                start=current_start, end=end, counts=counts, overlap_tokens=overlap_tokens
            )
            current_indices = list(range(overlap_start, end))
            current_start = overlap_start if current_indices else index
            current_total = sum(counts[item] for item in current_indices)
        # Match the paper builder: after overflow, retain fitting overlap and
        # append the new turn once, even if their combined size exceeds budget.
        current_indices.append(index)
        current_total += count
        if len(current_indices) == 1:
            current_start = index
    if current_indices:
        raw_chunks.append((current_start, current_indices[-1] + 1, current_total))

    chunks: list[dict[str, Any]] = []
    for start, end, total in raw_chunks:
        chunks.append({
            "chunk_id": len(chunks),
            "start_turn": start,
            "end_turn": end - 1,
            "token_count": total,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "turns": [dict(turn) for turn in history[start:end]],
        })
    return chunks


def run_chunk_filter(
    records: Sequence[dict[str, Any]],
    provider: StageProvider,
    prompt: str,
    *,
    counter: TokenCounter | None = None,
    chunk_tokens: int = 128,
    overlap_tokens: int = 32,
    chunks_per_group: int = 16,
) -> list[dict[str, Any]]:
    if chunks_per_group <= 0 or chunks_per_group > 16:
        raise ValueError("chunks_per_group must be between 1 and 16")
    actual_counter = counter or Cl100kCounter()
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        window_id = _window_id(record, seen)
        chunks = build_history_chunks(_turn_list(record, "history", window_id), actual_counter, chunk_tokens, overlap_tokens)
        labels: list[Any] = []
        for group_start in range(0, len(chunks), chunks_per_group):
            group = chunks[group_start:group_start + chunks_per_group]
            rendered_chunks = [
                {"chunk_id": chunk["chunk_id"], "start_turn": chunk["start_turn"], "end_turn": chunk["end_turn"], "turns": chunk["turns"]}
                for chunk in group
            ]
            group_labels = _parse_response(
                provider.complete(prompt.format(chunks=json.dumps(rendered_chunks, ensure_ascii=False), window=json.dumps(record.get("turns", []), ensure_ascii=False))),
                "chunk_filter",
                window_id,
                "labels",
            )
            if len(group_labels) != len(group):
                raise StageOutputError(
                    f"chunk_filter {window_id}: label cardinality {len(group_labels)} does not match {len(group)} chunks"
                )
            if any(type(label) is not int or label not in (0, 1) for label in group_labels):
                raise StageOutputError(f"chunk_filter {window_id}: labels must be binary integers")
            labels.extend(group_labels)
        relevant = [
            {"start_turn": chunk["start_turn"], "end_turn": chunk["end_turn"]}
            for label, chunk in zip(labels, chunks) if label == 1
        ]
        result.append({"schema_version": "1.0", "window_id": window_id, "relevant_chunks": relevant})
    return result
