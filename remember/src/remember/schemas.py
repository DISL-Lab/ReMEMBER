"""Stable immutable data contracts shared by the public pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Turn:
    time: str
    speaker: str
    dialogue: str


@dataclass(frozen=True)
class Window:
    window_id: str
    history: tuple[Turn, ...]
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class PredictedGap:
    index: int
    utterance_index: int | None
    quote: str
    gap_reason: str
    evidence_need_query: str
    keywords: tuple[str, ...]
    is_fallback: bool = False


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    start_turn_index: int
    end_turn_index: int
    turns: tuple[Turn, ...]
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int | str
    score: float
    source_start_index: int = 0
    source_end_index: int = 0
    text: str = ""
    sparse_rank: int | None = None
    dense_rank: int | None = None


class TokenCounter(Protocol):
    """Injectable token accounting used for chunk and memory budgets."""

    def count(self, text: str) -> int: ...


class Embedder(Protocol):
    """Injectable dense encoder; implementations stay outside this core."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...
