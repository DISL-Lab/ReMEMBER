"""Dependency-light chunking and deterministic hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Hashable, Sequence

from .schemas import Chunk, Embedder, PredictedGap, RetrievedChunk, TokenCounter, Turn


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+./:-]*")


def _render_turn(turn: Turn) -> str:
    return f"[{turn.time}] {turn.speaker}: {turn.dialogue}"


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def build_token_chunks(
    history: Sequence[Turn],
    counter: TokenCounter,
    max_tokens: int = 128,
    overlap_tokens: int = 32,
) -> list[Chunk]:
    """Split history on turn boundaries and retain a token-bounded tail overlap."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap_tokens < max_tokens")

    rendered = [_render_turn(turn) for turn in history]
    turn_lengths = [counter.count(text) for text in rendered]
    chunks: list[Chunk] = []
    current_indexes: list[int] = []
    current_tokens = 0

    def publish() -> None:
        if not current_indexes:
            return
        start, end = current_indexes[0], current_indexes[-1]
        chunk_turns = tuple(history[index] for index in current_indexes)
        chunks.append(Chunk(
            chunk_id=len(chunks),
            start_turn_index=start,
            end_turn_index=end,
            turns=chunk_turns,
            text="\n".join(rendered[index] for index in current_indexes),
        ))

    for index, token_count in enumerate(turn_lengths):
        if current_indexes and current_tokens + token_count > max_tokens:
            publish()
            retained: list[int] = []
            retained_tokens = 0
            for previous in reversed(current_indexes):
                previous_tokens = turn_lengths[previous]
                if retained_tokens + previous_tokens > overlap_tokens:
                    break
                retained.insert(0, previous)
                retained_tokens += previous_tokens
            current_indexes = retained
            current_tokens = retained_tokens
        current_indexes.append(index)
        current_tokens += token_count
    publish()
    return chunks


def reciprocal_rank_fuse(
    sparse_ids: Sequence[Hashable], dense_ids: Sequence[Hashable], k: int = 60
) -> list[RetrievedChunk]:
    """Fuse two ranked ID lists; ties retain stable first-seen order."""
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    ranks: dict[Hashable, list[int | None]] = {}
    for source, identifiers in enumerate((sparse_ids, dense_ids)):
        for rank, identifier in enumerate(identifiers, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(identifier, len(first_seen))
            ranks.setdefault(identifier, [None, None])[source] = rank
    ordered = sorted(scores, key=lambda identifier: (-scores[identifier], first_seen[identifier]))
    return [
        RetrievedChunk(
            chunk_id=identifier,
            score=scores[identifier],
            sparse_rank=ranks[identifier][0],
            dense_rank=ranks[identifier][1],
        )
        for identifier in ordered
    ]


def _bm25_scores(query: str, chunks: Sequence[Chunk]) -> list[float]:
    documents = [_tokens(chunk.text) for chunk in chunks]
    query_tokens = _tokens(query)
    if not documents:
        return []
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    average_length = sum(len(document) for document in documents) / len(documents)
    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for term in query_tokens:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            # Standard Robertson/Sparck Jones BM25 IDF; no optional package needed.
            idf = math.log(1 + (len(documents) - document_frequency[term] + 0.5) /
                           (document_frequency[term] + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(document) / average_length)
            score += idf * frequency * 2.5 / denominator
        scores.append(score)
    return scores


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedder returned vectors with inconsistent dimensions")
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(_dot(left, left))
    right_norm = math.sqrt(_dot(right, right))
    if not left_norm or not right_norm:
        return 0.0
    return _dot(left, right) / (left_norm * right_norm)


def _rank(scores: Sequence[float], chunks: Sequence[Chunk], limit: int) -> list[int]:
    return sorted(
        range(len(chunks)),
        key=lambda index: (-scores[index], chunks[index].start_turn_index, chunks[index].chunk_id),
    )[:limit]


def _query_text(gap: PredictedGap) -> str:
    return "\n".join(part for part in (
        gap.quote,
        gap.gap_reason,
        gap.evidence_need_query,
        " ".join(gap.keywords),
    ) if part)


def hybrid_retrieve(
    gaps: Sequence[PredictedGap],
    chunks: Sequence[Chunk],
    embedder: Embedder,
    top_k: int = 8,
    rrf_k: int = 60,
) -> dict[int, list[RetrievedChunk]]:
    """Run lexical and dense retrieval for each gap, then fuse their candidates."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if not chunks:
        return {gap.index: [] for gap in gaps}

    document_vectors = list(embedder.encode([chunk.text for chunk in chunks]))
    if len(document_vectors) != len(chunks):
        raise ValueError("embedder must return one vector per chunk")
    candidate_limit = min(len(chunks), 4 * top_k)
    results: dict[int, list[RetrievedChunk]] = {}
    for gap in gaps:
        query = _query_text(gap)
        sparse_order = _rank(_bm25_scores(query, chunks), chunks, candidate_limit)
        query_vectors = list(embedder.encode([gap.evidence_need_query]))
        if len(query_vectors) != 1:
            raise ValueError("embedder must return one vector per query")
        dense_scores = [_cosine(query_vectors[0], vector) for vector in document_vectors]
        dense_order = _rank(dense_scores, chunks, candidate_limit)
        fused = reciprocal_rank_fuse(
            [chunks[index].chunk_id for index in sparse_order],
            [chunks[index].chunk_id for index in dense_order],
            k=rrf_k,
        )
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        fused = sorted(
            fused,
            key=lambda item: (
                -item.score,
                by_id[item.chunk_id].start_turn_index,
                by_id[item.chunk_id].chunk_id,
            ),
        )
        results[gap.index] = [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                score=item.score,
                source_start_index=by_id[item.chunk_id].start_turn_index,
                source_end_index=by_id[item.chunk_id].end_turn_index,
                text=by_id[item.chunk_id].text,
                sparse_rank=item.sparse_rank,
                dense_rank=item.dense_rank,
            )
            for item in fused[:top_k]
        ]
    return results
