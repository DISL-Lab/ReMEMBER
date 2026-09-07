import pytest

from remember.retrieval import build_token_chunks, hybrid_retrieve, reciprocal_rank_fuse
from remember.schemas import Chunk, PredictedGap, Turn


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class HandWrittenEmbedder:
    """Deterministic semantic vectors with no model import or network access."""

    def encode(self, texts):
        vectors = {
            "cedar": (1.0, 0.0),
            "budget": (0.0, 1.0),
            "query cedar": (1.0, 0.0),
        }
        return [vectors.get(text, (0.0, 0.0)) for text in texts]


def test_rrf_combines_sparse_and_dense_rankings():
    result = reciprocal_rank_fuse(["c1", "c2"], ["c2", "c3"], k=60)
    assert [item.chunk_id for item in result] == ["c2", "c1", "c3"]
    assert result[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_chunking_retains_whole_turn_overlap_without_duplicate_chunk_ids():
    history = (
        Turn("1", "A", "one two"),
        Turn("2", "B", "three four"),
        Turn("3", "A", "five six"),
    )

    chunks = build_token_chunks(history, WordCounter(), max_tokens=8, overlap_tokens=4)

    assert [(chunk.chunk_id, chunk.start_turn_index, chunk.end_turn_index) for chunk in chunks] == [
        (0, 0, 1),
        (1, 1, 2),
    ]
    assert [turn.dialogue for turn in chunks[1].turns] == ["three four", "five six"]
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_overlap_tail_is_bounded_only_by_overlap_tokens():
    """The paper implementation retains an overlap tail up to overlap_tokens
    on its own, without also checking it against max_tokens together with the
    turn that triggered the new chunk. A chunk can therefore end up over
    max_tokens when the retained tail plus that turn together exceed it."""
    history = (
        Turn("1", "A", "one two"),
        Turn("2", "B", "three four"),
        Turn("3", "A", "five six"),
    )

    chunks = build_token_chunks(history, WordCounter(), max_tokens=10, overlap_tokens=8)

    assert [(chunk.start_turn_index, chunk.end_turn_index) for chunk in chunks] == [(0, 1), (0, 2)]
    assert WordCounter().count(chunks[1].text) == 12


def test_hybrid_retrieval_fuses_sparse_and_dense_candidates_per_gap():
    chunks = (
        Chunk(0, 0, 0, (), "cedar"),
        Chunk(1, 4, 4, (), "budget cedar"),
        Chunk(2, 9, 9, (), "budget"),
        # Decoy chunks keep "cedar"/"budget" document frequency below half the
        # corpus so BM25 idf stays positive (a 3-document corpus makes both
        # terms exactly as common as rare, which floors rank_bm25's idf to a
        # degenerate epsilon value and is not representative of a real corpus).
        Chunk(3, 14, 14, (), "weather report today"),
        Chunk(4, 19, 19, (), "team standup notes"),
    )
    gap = PredictedGap(
        index=7,
        utterance_index=0,
        quote="cedar",
        gap_reason="missing plan",
        evidence_need_query="query cedar",
        keywords=("cedar",),
    )

    result = hybrid_retrieve([gap], chunks, HandWrittenEmbedder(), top_k=2)

    assert list(result) == [7]
    assert [item.chunk_id for item in result[7]] == [0, 1]
    assert result[7][0].sparse_rank == 1
    assert result[7][0].dense_rank == 1
    assert result[7][0].score == pytest.approx(1 / 61 + 1 / 61)
