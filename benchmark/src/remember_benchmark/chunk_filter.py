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
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(history):
        end = start
        total = 0
        while end < len(history) and (total + counts[end] <= max_tokens or end == start):
            total += counts[end]
            end += 1
        chunks.append({
            "chunk_id": len(chunks),
            "start_turn": start,
            "end_turn": end - 1,
            "token_count": total,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "turns": [dict(turn) for turn in history[start:end]],
        })
        if end >= len(history):
            break
        overlap_start = end
        overlap_total = 0
        for index in range(end - 1, start - 1, -1):
            if overlap_total + counts[index] > overlap_tokens:
                break
            overlap_start = index
            overlap_total += counts[index]
        start = overlap_start if overlap_start < end else end
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
