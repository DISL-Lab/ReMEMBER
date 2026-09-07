"""Canonical round-robin facet-turn memory packing used by the paper method."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .config import DENSE_QUERY_PREFIX
from .gap_detection import MODEL_TAG_RE
from .schemas import Embedder, PredictedGap, RetrievedChunk, TokenCounter, Turn, Window


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class MemoryItem:
    text: str
    chunk_id: int | str
    source_turn_index: int
    unit_index: int
    gap_indices: tuple[int, ...]
    facet_types: tuple[str, ...]
    score: float
    rank: int
    pack_unit: str
    token_count: int


@dataclass(frozen=True)
class MemoryResult:
    text: str
    items: tuple[MemoryItem, ...]
    token_count: int
    policy: str
    layout: str
    candidate_count: int
    facet_count: int
    sentence_fallback_count: int


@dataclass(frozen=True)
class _Candidate:
    chunk_id: int | str
    chunk_rank: int
    source_turn_index: int
    unit_index: int
    turn_text: str
    unit_text: str


@dataclass(frozen=True)
class _Facet:
    gap_index: int
    facet_index: int
    facet_type: str
    text: str


@dataclass(frozen=True)
class _Event:
    candidate_index: int
    gap_index: int
    facet_index: int
    facet_type: str
    score: float
    chunk_rank: int
    facet_rank: int


class _PrefixedEmbedder:
    def __init__(self, embedder: Embedder, prefix: str) -> None:
        self.embedder = embedder
        self.prefix = prefix

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self.embedder.encode([self.prefix + text for text in texts])


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _render_turn(turn: Turn) -> str:
    return f"[{turn.time}] {turn.speaker}: {turn.dialogue}"


def _split_turn(turn: Turn) -> list[str]:
    dialogue = _normalize_space(turn.dialogue)
    if not dialogue:
        return []
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(dialogue) if piece.strip()]
    units: list[str] = []
    prefix = f"[{turn.time}] {turn.speaker}:".strip()
    for piece in pieces or [dialogue]:
        subparts = (
            [part.strip() for part in re.split(r"(?<=[,;:，；：])\s+", piece) if part.strip()]
            if len(piece) > 360
            else [piece]
        )
        for part in subparts:
            units.append(f"{prefix} {part}".strip())
    return units


def _clean_gap_reason(value: str, max_chars: int = 520) -> str:
    """gap_reason facets get the stronger cleanup applied when gaps are
    parsed elsewhere (strip bracketed model tags and "|" separators, cap
    length), since a gap_reason reaching facet construction here may not
    have passed through that path (e.g. cached/pre-parsed gaps)."""
    text = MODEL_TAG_RE.sub(" ", _normalize_space(value)).replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0]
    return text


def _facets(gaps: Sequence[PredictedGap]) -> list[_Facet]:
    facets: list[_Facet] = []
    for gap in gaps:
        raw = [
            ("evidence_need_query", gap.evidence_need_query),
            *(("gap_reason", part) for part in re.split(r"\s*\|\s*", gap.gap_reason)),
            ("gap_utterance", gap.quote),
        ]
        seen: set[str] = set()
        facet_index = 0
        for facet_type, value in raw:
            text = _clean_gap_reason(value) if facet_type == "gap_reason" else _normalize_space(value)
            key = text.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            facets.append(_Facet(gap.index, facet_index, facet_type, text))
            facet_index += 1
    return facets


def _score_matrix(rows: Sequence[Sequence[float]], columns: Sequence[Sequence[float]]) -> np.ndarray:
    """All-pairs dot product as a single matrix product, matching the paper's
    `cand_embs @ facet_embs.T`: a per-pair Python-loop dot product is not
    bit-identical to a batched matmul in float32, so near-tied candidates can
    rank differently. This is a plain dot product, not a re-normalized
    cosine similarity: the paper's embedders already L2-normalize their
    output, and re-dividing by each vector's norm here would only
    reintroduce float16-rounding noise from that normalization (norms end up
    near-but-not-exactly 1.0), which can flip near-tied rankings that the
    paper's own un-normalized dot product does not."""
    row_matrix = np.asarray(rows, dtype=np.float32)
    column_matrix = np.asarray(columns, dtype=np.float32)
    return row_matrix @ column_matrix.T


def _stable_chunk_id(value: int | str) -> tuple[int, int | str]:
    """Sort key that orders chunk_id numerically when it is an int (matching
    the paper's plain `sorted(candidates_by_chunk)` over integer chunk ids),
    falling back to string order only across mixed types. Stringifying an
    int chunk_id here (e.g. via `str(value)`) would sort it lexicographically
    ("111" before "9") and reorder candidates into different embedding
    batches than the paper's run, which can shift near-tied facet scores."""
    return (0, value) if isinstance(value, int) else (1, str(value))


def _candidates(
    window: Window,
    gaps: Sequence[PredictedGap],
    retrieved: Mapping[int, Sequence[RetrievedChunk]],
) -> tuple[list[_Candidate], dict[int, dict[int | str, int]]]:
    candidate_by_key: dict[tuple[int | str, int, int], _Candidate] = {}
    ranks_by_gap: dict[int, dict[int | str, int]] = {}
    for gap in gaps:
        ranks: dict[int | str, int] = {}
        for chunk_rank, chunk in enumerate(retrieved.get(gap.index, ()), start=1):
            ranks.setdefault(chunk.chunk_id, chunk_rank)
            for source_index in range(chunk.source_start_index, chunk.source_end_index + 1):
                if not 0 <= source_index < len(window.history):
                    continue
                turn = window.history[source_index]
                for unit_index, unit_text in enumerate(_split_turn(turn)):
                    key = (chunk.chunk_id, source_index, unit_index)
                    candidate_by_key.setdefault(
                        key,
                        _Candidate(
                            chunk_id=chunk.chunk_id,
                            chunk_rank=chunk_rank,
                            source_turn_index=source_index,
                            unit_index=unit_index,
                            turn_text=_render_turn(turn),
                            unit_text=unit_text,
                        ),
                    )
        ranks_by_gap[gap.index] = ranks
    candidates = sorted(
        candidate_by_key.values(),
        key=lambda item: (
            _stable_chunk_id(item.chunk_id),
            item.source_turn_index,
            item.unit_index,
        ),
    )
    return candidates, ranks_by_gap


def facet_turn_pack(
    window: Window,
    gaps: Sequence[PredictedGap],
    retrieved: Mapping[int, Sequence[RetrievedChunk]],
    embedder: Embedder,
    counter: TokenCounter,
    budget_tokens: int = 1024,
) -> MemoryResult:
    """Pack with the canonical paper prefix and 1,024-token default."""
    return _facet_turn_pack(
        window,
        gaps,
        retrieved,
        embedder,
        _PrefixedEmbedder(embedder, DENSE_QUERY_PREFIX),
        counter,
        budget_tokens,
    )


def _facet_turn_pack(
    window: Window,
    gaps: Sequence[PredictedGap],
    retrieved: Mapping[int, Sequence[RetrievedChunk]],
    embedder: Embedder,
    facet_embedder: Embedder,
    counter: TokenCounter,
    budget_tokens: int,
) -> MemoryResult:
    """Pack retrieved history using the canonical gap-facet round robin.

    Candidate sentences are ranked separately for query, reason, and utterance
    facets. Selection proceeds one rank-depth at a time across facets. A chosen
    sentence promotes its complete raw source turn; only a turn that does not
    fit in the remaining budget falls back to the selected sentence.
    """
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")

    candidates, ranks_by_gap = _candidates(window, gaps, retrieved)
    facets = _facets(gaps)
    candidate_vectors = list(embedder.encode([item.unit_text for item in candidates]))
    facet_vectors = list(facet_embedder.encode([item.text for item in facets]))
    if len(candidate_vectors) != len(candidates):
        raise ValueError("embedder must return one vector per candidate")
    if len(facet_vectors) != len(facets):
        raise ValueError("embedder must return one vector per facet")
    score_matrix = (
        _score_matrix(candidate_vectors, facet_vectors)
        if candidates and facets
        else np.zeros((len(candidates), len(facets)), dtype=np.float32)
    )

    ranked_by_facet: list[list[_Event]] = []
    for facet_index, facet in enumerate(facets):
        ranked: list[_Event] = []
        gap_ranks = ranks_by_gap.get(facet.gap_index, {})
        for candidate_index, candidate in enumerate(candidates):
            if candidate.chunk_id not in gap_ranks:
                continue
            ranked.append(_Event(
                candidate_index=candidate_index,
                gap_index=facet.gap_index,
                facet_index=facet.facet_index,
                facet_type=facet.facet_type,
                score=float(score_matrix[candidate_index, facet_index]),
                chunk_rank=gap_ranks[candidate.chunk_id],
                facet_rank=0,
            ))
        ranked.sort(key=lambda event: (
            -event.score,
            event.chunk_rank,
            _stable_chunk_id(candidates[event.candidate_index].chunk_id),
            candidates[event.candidate_index].source_turn_index,
            candidates[event.candidate_index].unit_index,
        ))
        ranked_by_facet.append([
            replace(event, facet_rank=rank)
            for rank, event in enumerate(ranked, start=1)
        ])

    rows: list[MemoryItem] = []
    row_by_turn: dict[tuple[int, str], int] = {}
    seen_text: set[str] = set()
    fallbacks = 0
    used = 0
    sep_tokens = counter.count("\n\n")
    max_depth = max((len(ranked) for ranked in ranked_by_facet), default=0)
    for depth in range(max_depth):
        for ranked in ranked_by_facet:
            if depth >= len(ranked):
                continue
            event = ranked[depth]
            candidate = candidates[event.candidate_index]
            turn_key = (candidate.source_turn_index, _normalize_space(candidate.turn_text).lower())
            if turn_key in row_by_turn:
                row_index = row_by_turn[turn_key]
                row = rows[row_index]
                rows[row_index] = replace(
                    row,
                    gap_indices=tuple(sorted({*row.gap_indices, event.gap_index})),
                    facet_types=tuple(sorted({*row.facet_types, event.facet_type})),
                    score=max(row.score, event.score),
                    rank=min(row.rank, event.chunk_rank),
                )
                continue

            # Budget is tracked as a running sum of each piece's own token count
            # (paper implementation), not by re-tokenizing the joined text on
            # every step: cl100k_base's BPE merges across a "\n\n" boundary
            # differently than counting pieces separately, so re-tokenizing the
            # whole prospective string can accept a different last item near
            # the cap.
            text = candidate.turn_text
            pack_unit = "source_turn"
            extra = counter.count(text) + (sep_tokens if rows else 0)
            if used + extra > budget_tokens:
                text = candidate.unit_text
                pack_unit = "sentence_fallback"
                extra = counter.count(text) + (sep_tokens if rows else 0)
                if used + extra > budget_tokens:
                    continue
                fallbacks += 1
            text_key = _normalize_space(text).lower()
            if not text_key or text_key in seen_text:
                continue
            seen_text.add(text_key)
            row_by_turn[turn_key] = len(rows)
            rows.append(MemoryItem(
                text=text,
                chunk_id=candidate.chunk_id,
                source_turn_index=candidate.source_turn_index,
                unit_index=candidate.unit_index,
                gap_indices=(event.gap_index,),
                facet_types=(event.facet_type,),
                score=round(event.score, 6),
                rank=event.chunk_rank,
                pack_unit=pack_unit,
                token_count=counter.count(text),
            ))
            used += extra

    text = "\n\n".join(row.text for row in rows)
    return MemoryResult(
        text=text,
        items=tuple(rows),
        token_count=counter.count(text),
        policy="gap_facet_dense_turn_winner_refill_pack",
        layout="raw_source_turns_selected_by_gap_facet_cosine",
        candidate_count=len(candidates),
        facet_count=len(facets),
        sentence_fallback_count=fallbacks,
    )
