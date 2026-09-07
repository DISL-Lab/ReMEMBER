import json
import inspect

from remember.config import RememberConfig
from remember.facet_turn import MemoryResult, facet_turn_pack
from remember.pipeline import _read_prompt, _run_window, run_window, run_windows
from remember.retrieval import hybrid_retrieve
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


class BatchRecordingGenerator:
    """Records the size of each generate_batch call so tests can assert on
    batch grouping; never calls the single-prompt generate()."""

    def __init__(self, gap_output, summary_output):
        self.gap_output = gap_output
        self.summary_output = summary_output
        self.batch_sizes: list[int] = []

    def generate(self, prompt, *, model=None, temperature=0.0, max_tokens=768):
        raise AssertionError("run_windows must use generate_batch, not generate")

    def generate_batch(self, prompts, *, model=None, temperature=0.0, max_tokens=768):
        self.batch_sizes.append(len(prompts))
        # Gap prompts contain the gap-detection instructions; summary prompts
        # contain the summary instructions.
        is_summary = "Reinforced Dialogue Summary" in prompts[0] or "Dialogue Summary" in prompts[0]
        output = self.summary_output if is_summary else self.gap_output
        return [output for _ in prompts]


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


def test_run_window_executes_gap_retrieval_pack_and_summary(monkeypatch):
    monkeypatch.setattr("remember.inference.Cl100kTokenCounter.count", WordCounter.count)
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

    result = run_window(WINDOW, config, generator, KeywordEmbedder())

    assert result.window_id == "E_fixture"
    assert result.schema_version == "1.1"
    assert "method" not in result.to_dict()
    assert result.memory.token_count <= 1024
    assert result.summary == "Fixture summary."
    assert [call["model"] for call in generator.calls] == [
        "fixture/gap-model",
        "fixture/summary-model",
    ]
    assert result.effective_config["dense_model"] == "fixture/dense-model"
    assert result.effective_config["token_encoding"] == "cl100k_base"


def test_run_windows_batches_gap_detection_and_summarization_across_windows(monkeypatch):
    """Matches the paper run: gap detection and summarization are each
    submitted as generation_batch_size-sized batches spanning multiple
    windows, not one generate() call per window."""
    monkeypatch.setattr("remember.inference.Cl100kTokenCounter.count", WordCounter.count)
    windows = [
        Window(
            f"E_fixture_{i}",
            history=(
                Turn("09:00", "Ava", "The cedar plan uses a fixed budget."),
                Turn("09:01", "Bo", "The launch owner is Cy."),
            ),
            turns=(Turn("10:00", "Ava", "Use it for the launch."),),
        )
        for i in range(5)
    ]
    gap_output = json.dumps({"gaps": [{
        "utterance_index": 0,
        "quote": "Use it for the launch.",
        "gap_reason": "The cedar plan is not identified.",
        "evidence_need_query": "retrieve earlier dialogue identifying the cedar plan",
        "keywords": ["cedar", "launch"],
    }]})
    generator = BatchRecordingGenerator(gap_output, '{"summary":"Fixture summary."}')
    config = RememberConfig(
        chunk_tokens=32, overlap_tokens=0, memory_budget_tokens=1024, top_k=2,
        generation_batch_size=2,
    )

    results = run_windows(windows, config, generator, KeywordEmbedder())

    assert [r.window_id for r in results] == [w.window_id for w in windows]
    assert all(r.summary == "Fixture summary." for r in results)
    # 5 windows batched by 2 -> [2, 2, 1] for gap detection, same for summary.
    assert generator.batch_sizes == [2, 2, 1, 2, 2, 1]


def test_dense_query_instruction_prefix_is_applied_only_to_queries_and_facets(monkeypatch):
    monkeypatch.setattr("remember.inference.Cl100kTokenCounter.count", WordCounter.count)
    sparse_queries = []
    original_bm25 = __import__("remember.retrieval", fromlist=["_bm25_scores"])._bm25_scores

    def record_bm25(bm25, query):
        sparse_queries.append(query)
        return original_bm25(bm25, query)

    monkeypatch.setattr("remember.retrieval._bm25_scores", record_bm25)
    generator = RecordingGenerator([json.dumps({"gaps": [{
        "utterance_index": 0,
        "quote": "Use cedar",
        "gap_reason": "Missing plan",
        "evidence_need_query": "retrieve earlier dialogue identifying cedar plan",
        "keywords": ["cedar", "plan"],
    }]}), '{"summary":"Fixture."}'])
    embedder = KeywordEmbedder()
    config = RememberConfig(chunk_tokens=32, overlap_tokens=0)

    result = run_window(
        WINDOW,
        config,
        generator,
        embedder,
    )

    structured = (
        "Find earlier dialogue that directly supports a faithful memory for the current window. "
        "Current utterance: Use cedar. Missing context: Missing plan. "
        "Needed evidence: retrieve earlier dialogue identifying cedar plan. "
        "Anchor terms: cedar plan use missing retrieve earlier dialogue identifying use cedar. "
        "Topic anchors: 10:00 ava: use launch.."
    )
    prefix = (
        "Instruct: Given a current-window utterance with missing dialogue context, "
        "retrieve earlier dialogue that directly supports a faithful memory item.\nQuery: "
    )
    assert embedder.calls[0] == [
        "[09:00] Ava: The cedar plan uses a fixed budget.\n[09:01] Bo: The launch owner is Cy."
    ]
    assert embedder.calls[1] == [prefix + structured]
    assert sparse_queries == [structured]
    assert embedder.calls[2] == [
        "[09:00] Ava: The cedar plan uses a fixed budget.",
        "[09:01] Bo: The launch owner is Cy.",
    ]
    assert embedder.calls[3] == [
        prefix + "retrieve earlier dialogue identifying cedar plan",
        prefix + "Missing plan",
        prefix + "Use cedar",
    ]
    assert result.retrieval


def test_public_pipeline_signatures_match_the_brief_exactly():
    expected = {
        facet_turn_pack: ["window", "gaps", "retrieved", "embedder", "counter", "budget_tokens"],
        summarize: [
            "window", "memory", "generator", "prompt_with_memory", "prompt_without_memory"
        ],
        run_window: ["window", "config", "generator", "embedder"],
        hybrid_retrieve: ["gaps", "chunks", "embedder", "top_k", "rrf_k"],
    }
    for function, names in expected.items():
        assert list(inspect.signature(function).parameters) == names
    assert inspect.signature(facet_turn_pack).parameters["budget_tokens"].default == 1024
    assert inspect.signature(hybrid_retrieve).parameters["top_k"].default == 8
    assert inspect.signature(hybrid_retrieve).parameters["rrf_k"].default == 60


def test_run_result_public_dict_has_required_fields_without_source_dialogue(monkeypatch):
    monkeypatch.setattr("remember.inference.Cl100kTokenCounter.count", WordCounter.count)
    generator = RecordingGenerator(["not json", '{"summary":"Fallback summary."}'])
    result = run_window(WINDOW, RememberConfig(), generator, KeywordEmbedder())

    public = result.to_dict()

    assert set(public) == {
        "schema_version",
        "window_id",
        "predicted_gaps",
        "retrieval",
        "memory",
        "summary",
        "token_counts",
        "timing_seconds",
        "effective_config",
    }
    assert public["schema_version"] == "1.1"
    assert "history" not in public
    assert "turns" not in public
    assert "The cedar plan uses a fixed budget." not in json.dumps(public["predicted_gaps"])
    assert public["memory"]["text"]


def test_memory_only_skips_summary_generation():
    generator = RecordingGenerator(["not json"])

    result = _run_window(
        WINDOW,
        RememberConfig(),
        generator,
        KeywordEmbedder(),
        counter=WordCounter(),
        memory_only=True,
        gap_prompt=_read_prompt("gap_detection.txt"),
        summary_prompt_with_memory=_read_prompt("summary_with_memory.txt"),
        summary_prompt_without_memory=_read_prompt("summary_window_only.txt"),
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
    generator = RecordingGenerator(["A concise plain-text summary of the whole window."])
    empty = MemoryResult("", (), 0, "p", "l", 0, 0, 0)

    assert summarize(WINDOW, empty, generator, "{window} {memory}", "{window}") == (
        "A concise plain-text summary of the whole window."
    )


def test_summary_parser_rejects_short_plain_text_as_unusable():
    """Canonical `parse_summary`'s raw-text fallback requires >=6 tokens, so a
    short non-JSON response (e.g. a truncated or garbled generation) is
    treated as unusable rather than accepted as a real summary."""
    generator = RecordingGenerator(["Too short."])
    empty = MemoryResult("", (), 0, "p", "l", 0, 0, 0)

    assert summarize(WINDOW, empty, generator, "{window} {memory}", "{window}") == ""
