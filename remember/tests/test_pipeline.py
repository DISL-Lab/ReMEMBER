import json

from remember.config import RememberConfig
from remember.facet_turn import MemoryResult
from remember.pipeline import METHOD, run_window
from remember.schemas import Turn, Window
from remember.summarization import summarize


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class RecordingGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, prompt, *, model=None, temperature=0.0, max_tokens=768):
        self.calls.append({
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return next(self.outputs)


class KeywordEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append((
                float("cedar" in lowered),
                float("budget" in lowered),
                float("launch" in lowered),
            ))
        return vectors


WINDOW = Window(
    "E_fixture",
    history=(
        Turn("09:00", "Ava", "The cedar plan uses a fixed budget."),
        Turn("09:01", "Bo", "The launch owner is Cy."),
    ),
    turns=(Turn("10:00", "Ava", "Use it for the launch."),),
)


def test_run_window_executes_gap_retrieval_pack_and_summary():
    generator = RecordingGenerator([
        json.dumps({"gaps": [{
            "utterance_index": 0,
            "quote": "Use it for the launch.",
            "gap_reason": "The cedar plan is not identified.",
            "evidence_need_query": "retrieve earlier dialogue identifying the cedar plan",
            "keywords": ["cedar", "launch"],
        }]}),
        '{"summary":"Fixture summary."}',
    ])
    config = RememberConfig(
        gap_model="fixture/gap-model",
        summary_model="fixture/summary-model",
        dense_model="fixture/dense-model",
        chunk_tokens=32,
        overlap_tokens=0,
        memory_budget_tokens=1024,
        top_k=2,
    )

    result = run_window(
        WINDOW, config, generator, KeywordEmbedder(), counter=WordCounter()
    )

    assert result.window_id == "E_fixture"
    assert result.method == "pred_gap_hybrid_facet_turn_pack1024" == METHOD
    assert result.memory.token_count <= 1024
    assert result.summary == "Fixture summary."
    assert [call["model"] for call in generator.calls] == [
        "fixture/gap-model",
        "fixture/summary-model",
    ]
    assert result.effective_config["dense_model"] == "fixture/dense-model"
    assert result.effective_config["token_encoding"] == "cl100k_base"


def test_dense_query_instruction_prefix_is_applied_only_to_queries_and_facets():
    generator = RecordingGenerator(["not json"])
    embedder = KeywordEmbedder()
    config = RememberConfig(dense_query_prefix="CANONICAL PREFIX: ")

    run_window(
        WINDOW,
        config,
        generator,
        embedder,
        counter=WordCounter(),
        memory_only=True,
    )

    assert embedder.calls[0] == [
        "[09:00] Ava: The cedar plan uses a fixed budget.\n[09:01] Bo: The launch owner is Cy."
    ]
    assert embedder.calls[1][0].startswith("CANONICAL PREFIX: ")
    assert embedder.calls[2] == [
        "[09:00] Ava: The cedar plan uses a fixed budget.",
        "[09:01] Bo: The launch owner is Cy.",
    ]
    assert all(text.startswith("CANONICAL PREFIX: ") for text in embedder.calls[3])


def test_run_result_public_dict_has_required_fields_without_source_dialogue():
    generator = RecordingGenerator(["not json", '{"summary":"Fallback summary."}'])
    result = run_window(
        WINDOW, RememberConfig(), generator, KeywordEmbedder(), counter=WordCounter()
    )

    public = result.to_dict()

    assert set(public) == {
        "schema_version",
        "window_id",
        "method",
        "predicted_gaps",
        "retrieval",
        "memory",
        "summary",
        "token_counts",
        "timing_seconds",
        "effective_config",
    }
    assert "history" not in public
    assert "turns" not in public
    assert "The cedar plan uses a fixed budget." not in json.dumps(public["predicted_gaps"])
    assert public["memory"]["text"]


def test_memory_only_skips_summary_generation():
    generator = RecordingGenerator(["not json"])

    result = run_window(
        WINDOW,
        RememberConfig(),
        generator,
        KeywordEmbedder(),
        counter=WordCounter(),
        memory_only=True,
    )

    assert result.summary == ""
    assert len(generator.calls) == 1


def test_summarize_uses_window_only_prompt_for_empty_memory():
    generator = RecordingGenerator(['{"summary":"Window-only summary."}'])
    empty = MemoryResult(
        text="",
        items=(),
        token_count=0,
        policy="gap_facet_dense_turn_winner_refill_pack",
        layout="raw_source_turns_selected_by_gap_facet_cosine",
        candidate_count=0,
        facet_count=0,
        sentence_fallback_count=0,
    )

    result = summarize(
        WINDOW,
        empty,
        generator,
        "WITH {window} MEMORY {memory}",
        "WITHOUT {window}",
    )

    assert result == "Window-only summary."
    assert generator.calls[0]["prompt"].startswith("WITHOUT ")


def test_summary_parser_accepts_plain_text_response():
    generator = RecordingGenerator(["A concise plain-text summary."])
    empty = MemoryResult("", (), 0, "p", "l", 0, 0, 0)

    assert summarize(WINDOW, empty, generator, "{window} {memory}", "{window}") == (
        "A concise plain-text summary."
    )
