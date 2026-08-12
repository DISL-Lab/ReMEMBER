"""Deterministic construction of the paper's fixed token windows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .config import PAPER_SOURCE_CONFIG, RANGES, WindowConfig
from .schemas import NormalizedDialogue, SOURCE_PREFIX, dialogue_sort_key


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


def assign_range(tok_end: int) -> str:
    for label, low, high in RANGES:
        if low <= tok_end < high:
            return label
    return "other"


def _format_turn(turn: object) -> str:
    return f"[{turn.time}] {turn.speaker}: {turn.dialogue}"  # type: ignore[attr-defined]


def build_windows(
    dialogues: Sequence[NormalizedDialogue],
    counter: TokenCounter,
    *,
    configs: Mapping[str, WindowConfig] | None = None,
) -> list[dict[str, object]]:
    source_configs = dict(PAPER_SOURCE_CONFIG if configs is None else configs)
    counters = {source: 0 for source in SOURCE_PREFIX}
    rows: list[dict[str, object]] = []
    for dialogue in sorted(dialogues, key=dialogue_sort_key):
        try:
            config = source_configs[dialogue.source]
        except KeyError as exc:
            raise ValueError(f"missing window configuration for {dialogue.source}") from exc
        spans: list[tuple[int, int]] = []
        cumulative = 0
        for turn in dialogue.turns:
            start = cumulative
            count = counter.count(_format_turn(turn))
            if not isinstance(count, int) or count <= 0:
                raise ValueError("token counter must return a positive integer for every turn")
            cumulative += count
            spans.append((start, cumulative))
        win_start = config.start_tokens
        while win_start + config.size_tokens <= cumulative:
            win_end = win_start + config.size_tokens
            history = [turn.to_dict() for turn, (_, end) in zip(dialogue.turns, spans) if end <= win_start]
            selected = [
                turn.to_dict()
                for turn, (start, end) in zip(dialogue.turns, spans)
                if end > win_start and start < win_end
            ]
            prefix = SOURCE_PREFIX[dialogue.source]
            window_id = f"{prefix}_U{counters[dialogue.source]:04d}"
            counters[dialogue.source] += 1
            rows.append({
                "schema_version": "1.0",
                "window_id": window_id,
                "dialogue_id": dialogue.dialogue_id,
                "source": dialogue.source,
                "range": assign_range(win_end),
                "tok_start": win_start,
                "tok_end": win_end,
                "n_history": len(history),
                "n_turns": len(selected),
                "history": history,
                "turns": selected,
            })
            win_start += config.stride_tokens
    return rows
