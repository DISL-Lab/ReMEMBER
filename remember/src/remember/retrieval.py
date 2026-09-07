"""Chunking and deterministic hybrid retrieval (BM25 via rank_bm25, as used for the paper)."""

from __future__ import annotations

import re
from typing import Hashable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

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
            # The overlap tail is bounded only by overlap_tokens (paper
            # implementation): it does not also check the incoming turn
            # against max_tokens here, so retained tail size is independent
            # of how large the next turn is.
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


def _bm25_index(chunks: Sequence[Chunk]) -> BM25Okapi:
    """Build the ATIRE-variant BM25 index the paper implementation used."""
    return BM25Okapi([_tokens(chunk.text) for chunk in chunks])


def _bm25_scores(bm25: BM25Okapi, query: str) -> list[float]:
    return list(bm25.get_scores(_tokens(query)))


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
    return _hybrid_retrieve(
        gaps,
        chunks,
        embedder,
        query_texts=[_query_text(gap) for gap in gaps],
        query_embedder=embedder,
        top_k=top_k,
        rrf_k=rrf_k,
    )


def _hybrid_retrieve(
    gaps: Sequence[PredictedGap],
    chunks: Sequence[Chunk],
    embedder: Embedder,
    *,
    query_texts: Sequence[str],
    query_embedder: Embedder,
    top_k: int,
    rrf_k: int,
) -> dict[int, list[RetrievedChunk]]:
    """Internal retrieval path with explicitly bound canonical query controls."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if len(query_texts) != len(gaps):
        raise ValueError("query_texts must contain one query per gap")
    if not chunks:
        return {gap.index: [] for gap in gaps}

    document_vectors = list(embedder.encode([chunk.text for chunk in chunks]))
    if len(document_vectors) != len(chunks):
        raise ValueError("embedder must return one vector per chunk")
    # All gap queries for a window are embedded in a single batch (matching
    # the paper implementation): a transformer encoder's batched matmuls are
    # not bit-identical to encoding the same text one query at a time, so
    # batching queries individually here would silently drift from the paper.
    query_vectors_all = list(query_embedder.encode(list(query_texts)))
    if len(query_vectors_all) != len(query_texts):
        raise ValueError("embedder must return one vector per query")
    # Cosine scores are computed as a single matrix-vector product (matching
    # the paper's `doc_matrix @ query_vector`), not a per-document Python loop:
    # float32 summation order differs enough between the two that near-tied
    # candidates can rank differently.
    # A plain dot product (matching the paper's `doc_matrix @ query_vector`),
    # not a re-normalized cosine similarity: the paper's embedders already
    # L2-normalize their output, and re-dividing by each vector's norm here
    # would only reintroduce float16-rounding noise from that normalization
    # (norms end up near-but-not-exactly 1.0), which can flip near-tied
    # rankings that the paper's own un-normalized dot product does not.
    document_matrix = np.asarray(document_vectors, dtype=np.float32)
    query_matrix = np.asarray(query_vectors_all, dtype=np.float32)
    candidate_limit = min(len(chunks), 4 * top_k)
    bm25 = _bm25_index(chunks)
    results: dict[int, list[RetrievedChunk]] = {}
    for gap_index, (gap, query) in enumerate(zip(gaps, query_texts)):
        sparse_order = _rank(_bm25_scores(bm25, query), chunks, candidate_limit)
        query_vector = query_matrix[gap_index]
        dense_scores = (document_matrix @ query_vector).tolist()
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
