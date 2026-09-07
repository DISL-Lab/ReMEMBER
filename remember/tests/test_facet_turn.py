import json
from pathlib import Path

from remember.facet_turn import facet_turn_pack
from remember.schemas import PredictedGap, RetrievedChunk, Turn, Window


FIXTURES = Path(__file__).parent / "fixtures"


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class FacetEmbedder:
    """Literal semantic vectors for the fabricated golden fixture."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append((
                1.0 if "cedar query" in lowered else 0.0,
                1.0 if "budget reason" in lowered else 0.0,
                1.0 if "launch utterance" in lowered else 0.0,
            ))
        return vectors


class ConstantEmbedder:
    def encode(self, texts):
        return [(1.0,) for _ in texts]


class NonAdditiveCounter:
    """Makes a joined document cost more than separately counted pieces."""

    def count(self, text: str) -> int:
        return len(text.split()) + (2 if text.strip() and "\n\n" in text else 0)


def _turn(value):
    return Turn(value["time"], value["speaker"], value["dialogue"])


def _gap(value):
    return PredictedGap(
        index=value["index"],
        utterance_index=value["utterance_index"],
        quote=value["quote"],
        gap_reason=value["gap_reason"],
        evidence_need_query=value["evidence_need_query"],
        keywords=tuple(value["keywords"]),
    )


def _retrieved(value):
    return RetrievedChunk(**value)


def test_golden_facet_order_round_robin_and_raw_turn_promotion():
    fixture = json.loads((FIXTURES / "facet_turn_golden.json").read_text())
    raw_window = fixture["window"]
    window = Window(
        window_id=raw_window["window_id"],
        history=tuple(_turn(row) for row in raw_window["history"]),
        turns=tuple(_turn(row) for row in raw_window["turns"]),
    )
    gaps = tuple(_gap(row) for row in fixture["gaps"])
    retrieved = {0: tuple(_retrieved(row) for row in fixture["retrieved"])}

    memory = facet_turn_pack(
        window, gaps, retrieved, FacetEmbedder(), WordCounter(), budget_tokens=1024
    )

    assert [item.text for item in memory.items] == fixture["expected"]["texts"]
    assert [list(item.facet_types) for item in memory.items] == fixture["expected"]["facet_types"]
    assert memory.items[0].text.endswith("This complete turn is retained.")
    assert memory.policy == fixture["expected"]["policy"]
    assert memory.layout == fixture["expected"]["layout"]
    assert memory.facet_count == 3


def test_oversized_promoted_turn_falls_back_to_selected_sentence():
    window = Window(
        "fallback",
        history=(
            Turn("1", "A", "query match."),
            Turn("2", "B", "reason match. filler filler filler filler."),
        ),
        turns=(),
    )
    gap = PredictedGap(0, None, "unmatched quote", "reason match", "query match", ())
    retrieved = {0: (RetrievedChunk(0, 1.0, 0, 1, "unused"),)}

    memory = facet_turn_pack(
        window, (gap,), retrieved, FacetEmbedder(), WordCounter(), budget_tokens=8
    )

    assert [item.text for item in memory.items] == [
        "[1] A: query match.",
        "[2] B: reason match.",
    ]
    assert [item.pack_unit for item in memory.items] == ["source_turn", "sentence_fallback"]
    assert memory.sentence_fallback_count == 1
    assert memory.token_count == 8


def test_equal_scores_use_source_turn_then_sentence_as_deterministic_ties():
    window = Window(
        "ties",
        history=(
            Turn("1", "A", "First sentence. Second sentence."),
            Turn("2", "B", "Third sentence."),
        ),
        turns=(),
    )
    gap = PredictedGap(0, None, "quote", "reason", "query", ())
    retrieved = {0: (RetrievedChunk("chunk-a", 1.0, 0, 1, "unused"),)}

    first = facet_turn_pack(window, (gap,), retrieved, ConstantEmbedder(), WordCounter())
    second = facet_turn_pack(window, (gap,), retrieved, ConstantEmbedder(), WordCounter())

    assert [item.source_turn_index for item in first.items] == [0, 1]
    assert [item.text for item in first.items] == [item.text for item in second.items]
    assert first.items[0].unit_index == 0


def test_default_paper_budget_never_exceeds_1024_tokens():
    history = tuple(
        Turn(str(index), "A", "query " + "token " * 697 + f"end{index}.")
        for index in range(3)
    )
    window = Window("cap", history=history, turns=())
    gap = PredictedGap(0, None, "quote", "reason", "query", ())
    retrieved = {0: (RetrievedChunk(0, 1.0, 0, 2, "unused"),)}

    memory = facet_turn_pack(window, (gap,), retrieved, ConstantEmbedder(), WordCounter())

    assert memory.token_count <= 1024
    assert WordCounter().count(memory.text) <= 1024
    assert len(memory.items) == 1


def test_cap_is_checked_against_a_running_sum_of_each_pieces_own_count():
    """The paper implementation tracks the budget as a running sum of each
    selected piece's own token count plus a separator count, rather than by
    re-tokenizing the whole joined memory text on every step. A non-additive
    counter can then let the final joined text exceed the nominal cap, because
    the accept/reject decision never re-tokenizes the join as a whole."""
    window = Window(
        "non-additive-cap",
        history=(
            Turn("1", "A", "query evidence."),
            Turn("2", "B", "reason evidence."),
        ),
        turns=(),
    )
    gap = PredictedGap(0, None, "quote", "reason", "query", ())
    retrieved = {0: (RetrievedChunk(0, 1.0, 0, 1, "unused"),)}

    memory = facet_turn_pack(
        window, (gap,), retrieved, ConstantEmbedder(), NonAdditiveCounter(), budget_tokens=8
    )

    assert [item.text for item in memory.items] == [
        "[1] A: query evidence.",
        "[2] B: reason evidence.",
    ]
    # Each piece counts 4 on its own and the running sum never exceeds the
    # budget (4, then 4+4=8), even though re-tokenizing the final join scores
    # 10 (NonAdditiveCounter adds +2 once real content surrounds a "\n\n").
    assert memory.token_count == 10


def test_each_gap_scores_only_chunks_retrieved_for_that_gap():
    window = Window(
        "gap-scope",
        history=(
            Turn("1", "A", "alpha evidence."),
            Turn("2", "B", "beta evidence."),
        ),
        turns=(),
    )
    gaps = (
        PredictedGap(0, None, "beta", "beta", "beta", ()),
        PredictedGap(1, None, "alpha", "alpha", "alpha", ()),
    )
    retrieved = {
        0: (RetrievedChunk(5, 1.0, 0, 0, "unused"),),
        1: (RetrievedChunk(1, 1.0, 0, 1, "unused"),),
    }

    memory = facet_turn_pack(
        window, gaps, retrieved, FacetEmbedder(), WordCounter(), budget_tokens=1024
    )

    assert memory.items[0].chunk_id == 5
    assert memory.items[0].source_turn_index == 0
    assert memory.items[1].source_turn_index == 1
    assert memory.items[1].gap_indices == (1,)


def test_shared_chunk_uses_its_rank_within_each_gap():
    window = Window(
        "per-gap-rank",
        history=(
            Turn("1", "A", "shared chunk evidence."),
            Turn("2", "B", "gap one first-ranked evidence."),
            Turn("3", "C", "gap zero second-ranked evidence."),
        ),
        turns=(),
    )
    gaps = (
        PredictedGap(0, None, "q0", "r0", "n0", ()),
        PredictedGap(1, None, "q1", "r1", "n1", ()),
    )
    retrieved = {
        0: (
            RetrievedChunk(0, 1.0, 0, 0, "unused"),
            RetrievedChunk(2, 0.5, 2, 2, "unused"),
        ),
        1: (
            RetrievedChunk(1, 1.0, 1, 1, "unused"),
            RetrievedChunk(0, 0.5, 0, 0, "unused"),
        ),
    }

    memory = facet_turn_pack(
        window, gaps, retrieved, ConstantEmbedder(), WordCounter(), budget_tokens=11
    )

    assert [item.source_turn_index for item in memory.items[:2]] == [0, 1]
