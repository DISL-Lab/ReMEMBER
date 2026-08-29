"""Tests for the one-file ReMEMber evaluator."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "evaluation/evaluate.py"
PROMPT_DIR = ROOT / "evaluation/prompts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
BENCHMARK = FIXTURES / "benchmark.jsonl"
PREDICTIONS = FIXTURES / "predictions.jsonl"

sys.path.insert(0, str(ROOT / "evaluation"))
import evaluate as ev  # noqa: E402


EXPECTED_PROMPT_HASHES = {
    "memory_recall": "1bc6228fa148c6b1411a6c460c13682100e976ac9e9bb9f4da4ce3fcd8085874",
    "mkf": "75aea6ec084a1c9016c914ecca915198487254dbdbe7a0c2f9b3708af397342c",
    "wkf": "c4fc5c3f91e1d13208bd6b4972c9754992e042125814800a62a792cbafe976bd",
    "atomic_facts": "db406d5eafeb0dd51f813cf45d5b22aed57e11bdedf493748b695c2172cc1676",
    "factuality": "f4613f72b2355cb96c53c10b8340e02ffa1e203b9797acb0489283b88df1a1f2",
}


def _args(**overrides):
    namespace = argparse.Namespace(
        benchmark=BENCHMARK, predictions=PREDICTIONS, output_dir=Path("unused-output"),
        provider="fake", model=ev.PAPER_MODEL, metrics="all", prompt_dir=PROMPT_DIR,
        temperature=0.0, mr_max_tokens=2048, coverage_max_tokens=4096, atomic_max_tokens=2048,
        factuality_max_tokens=2048, batch_size=64, gpu_devices=None, max_model_len=25000,
        gpu_memory_utilization=0.8, tensor_parallel_size=1, base_url=None,
        api_key_env="OPENAI_API_KEY", max_retries=2, allow_partial=True,
        resume=False, overwrite=False,
    )
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


# ------------------------------------------------------------------- prompts

def test_shipped_prompts_have_exact_paper_hashes():
    prompts = ev.load_prompts(PROMPT_DIR)
    assert {name: ev.sha256_text(text) for name, text in prompts.items()} == EXPECTED_PROMPT_HASHES


def test_each_prompt_declares_its_approved_placeholder_set():
    prompts = ev.load_prompts(PROMPT_DIR)
    for name, expected in ev.PROMPT_PLACEHOLDERS.items():
        assert ev.extract_live_placeholders(prompts[name]) == expected


def test_escaped_braces_are_not_placeholders():
    template = 'Example {{"gap_id": "x"}} and a real {window} field.'
    assert ev.extract_live_placeholders(template) == {"window"}
    assert ev.fill_prompt(template, window="W") == 'Example {"gap_id": "x"} and a real W field.'


@pytest.mark.parametrize("values", [{}, {"window": "W", "extra": "E"}])
def test_fill_prompt_rejects_both_sides_of_the_call_contract(values):
    with pytest.raises(ev.InputError):
        ev.fill_prompt("only {window}", **values)


def test_placeholder_validation_reports_missing_and_unexpected():
    with pytest.raises(ev.InputError, match="unexpected"):
        ev.validate_placeholder_set("{window} {rogue}", {"window"})


# ----------------------------------------------------------------- benchmark

def test_benchmark_loads_from_plain_and_gzip_jsonl(tmp_path):
    import gzip
    plain = ev.load_benchmark(BENCHMARK)
    gz = tmp_path / "benchmark.jsonl.gz"
    with gzip.open(gz, "wb") as handle:
        handle.write(BENCHMARK.read_bytes())
    assert ev.load_benchmark(gz).ordered_ids == plain.ordered_ids
    assert plain.input_sha256 == ev.sha256_file(BENCHMARK)
    assert plain.ordered_ids == ("E_U0001", "L_U0002")


def test_benchmark_rejects_a_directory(tmp_path):
    with pytest.raises(ev.InputError, match="one .* file"):
        ev.load_benchmark(tmp_path)


@pytest.mark.parametrize("mutation,message", [
    (lambda rows: rows[0].update(schema_version="1.0"), "schema version"),
    (lambda rows: rows.append(dict(rows[0])), "duplicate benchmark window_id"),
    (lambda rows: rows[0].update(history=[]), "history"),
    (lambda rows: rows[0]["gaps"][0].update(target_sentence=[]), "target_sentence"),
    (lambda rows: rows[0]["gaps"][0]["evidence"][0].update(sentence="  "), "sentence"),
    (lambda rows: rows[0]["gaps"].append(dict(rows[0]["gaps"][0])), "duplicate gap_idx"),
])
def test_benchmark_rejects_malformed_records(tmp_path, mutation, message):
    rows = [json.loads(line) for line in BENCHMARK.read_text().splitlines() if line.strip()]
    mutation(rows)
    path = tmp_path / "bad.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ev.InputError, match=message):
        ev.load_benchmark(path)


def test_non_canonical_population_requires_allow_partial():
    benchmark = ev.load_benchmark(BENCHMARK)
    with pytest.raises(ev.InputError, match="--allow-partial"):
        ev.require_canonical_population(benchmark, allow_partial=False)
    ev.require_canonical_population(benchmark, allow_partial=True)


def test_ordered_ids_digest_matches_the_canonical_split_file():
    split = ROOT / "dataset/splits/main_900.txt"
    ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    assert ev.ordered_ids_sha256(ids) == ev.CANONICAL_SPLIT_SHA256 == ev.sha256_file(split)


def test_the_published_payload_is_the_canonical_population():
    benchmark = ev.load_benchmark(ROOT / "dataset/remember_900.jsonl.gz")
    assert ev.is_canonical_benchmark(benchmark)
    assert len(benchmark.ordered_ids) == 900
    ev.require_canonical_population(benchmark, allow_partial=False)


# ---------------------------------------------------------------- predictions

def test_structured_memory_adapters():
    assert ev.render_structured_memory({"text": "packed memory", "items": []}) == "packed memory"
    assert ev.render_structured_memory({
        "text": "", "items": [{"text": "first"}, {"text": "second"}],
    }) == "first\nsecond"


def test_unsupported_structured_memory_is_rejected_not_stringified():
    with pytest.raises(ev.InputError):
        ev.render_structured_memory({"blob": {"nested": 1}})


def test_memory_text_takes_precedence_over_structured_memory(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({
        "window_id": "E_U0001", "summary": "S.",
        "memory_text": "generic wins", "memory": {"text": "structured loses"},
    }) + "\n", encoding="utf-8")
    loaded = ev.load_predictions(path, ["E_U0001"], allow_partial=True)
    assert loaded["E_U0001"]["memory"] == "generic wins"


@pytest.mark.parametrize("row,message", [
    ({"window_id": "E_U0001", "summary": "  ", "memory_text": "m"}, "summary"),
    ({"window_id": "E_U0001", "summary": "S.", "memory_text": " "}, "memory_text"),
    ({"window_id": "ZZZ", "summary": "S.", "memory_text": "m"}, "not in the benchmark"),
    ({"window_id": "E_U0001", "summary": "S."}, "no memory"),
])
def test_prediction_rows_are_validated(tmp_path, row, message):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ev.InputError, match=message):
        ev.load_predictions(path, ["E_U0001"], allow_partial=True)


def test_duplicate_prediction_ids_are_rejected(tmp_path):
    path = tmp_path / "p.jsonl"
    row = {"window_id": "E_U0001", "summary": "S.", "memory_text": "m"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ev.InputError, match="duplicate prediction"):
        ev.load_predictions(path, ["E_U0001"], allow_partial=True)


def test_full_run_requires_every_benchmark_window():
    with pytest.raises(ev.InputError, match="missing 1 benchmark windows"):
        ev.load_predictions(PREDICTIONS, ["E_U0001", "L_U0002", "R_U0003"], allow_partial=False)


# -------------------------------------------------------------------- parsing

def test_mr_treats_only_one_and_covered_as_positive():
    expected = ["gap_0"]
    for value in (1, "1", "covered", "COVERED"):
        assert ev.normalize_mr([{"gap_id": "gap_0", "decision": value}], expected)[0]["decision"] == 1
    for value in (0, "0", "PARTIAL", 0.5, True, None):
        assert ev.normalize_mr([{"gap_id": "gap_0", "decision": value}], expected)[0]["decision"] == 0


def test_mr_accepts_the_canonical_output_shapes():
    body = '{"gap_id": "gap_0", "decision": "1"}'
    assert ev.parse_mr(f"[{body}]") == [json.loads(body)]
    assert ev.parse_mr('{"results": [%s]}' % body) == [json.loads(body)]
    assert ev.parse_mr(f"```json\n{body}\n```") == [json.loads(body)]
    assert ev.parse_mr(f"[{body},]") == [json.loads(body)]
    assert ev.parse_mr(f"[{body}, {{\"gap_id\": \"gap_1\", \"dec") == [json.loads(body)]


def test_mr_strips_a_completed_think_block():
    assert ev.parse_mr('<think>ignore</think>[{"gap_id": "gap_0", "decision": 1}]') == [
        {"gap_id": "gap_0", "decision": 1}
    ]


@pytest.mark.parametrize("items", [
    [{"gap_id": "gap_0", "decision": 1}, {"gap_id": "gap_0", "decision": 0}],   # duplicate
    [{"gap_id": "gap_9", "decision": 1}],                                        # unexpected
    [],                                                                           # missing
])
def test_any_mr_item_defect_invalidates_the_whole_judgment(items):
    assert ev.normalize_mr(items, ["gap_0", "gap_1"]) is None


def test_coverage_uses_the_last_balanced_array_and_no_comma_repair():
    raw = '<think>[{"utterance_num": "U0", "score": 1.0}]</think>\n[{"utterance_num": "U0", "score": 0.0, "line_number": ["S1"]}]'
    parsed = ev.parse_coverage(raw)
    assert parsed == [{"utterance_num": "U0", "score": 0.0, "line_number": ["S1"]}]
    assert ev.parse_coverage('[{"utterance_num": "U0", "score": 1.0},]') is None


def test_coverage_normalization_requires_exact_expected_ids():
    items = [{"utterance_num": "U0", "score": 1.0, "line_number": ["S1"]}]
    assert ev.normalize_coverage(items, ["U0"], id_field="utterance_num") == [
        {"utterance_num": "U0", "score": 1.0, "line_number": ["S1"]}
    ]
    assert ev.normalize_coverage(items, ["U0", "U1"], id_field="utterance_num") is None
    bad = [{"utterance_num": "U0", "score": 0.5, "line_number": []}]
    assert ev.normalize_coverage(bad, ["U0"], id_field="utterance_num") is None


def test_atomic_failure_falls_back_to_one_fact_per_sentence():
    sentences = ["First.", "Second."]
    assert ev.normalize_atomic_facts(None, sentences) == [
        {"sentence_num": "S0", "fact_id": "S0-F0", "text": "First."},
        {"sentence_num": "S1", "fact_id": "S1-F0", "text": "Second."},
    ]
    assert ev.normalize_atomic_facts([], sentences)[0]["fact_id"] == "S0-F0"
    assert len(ev.normalize_atomic_facts([{"sentence_num": "S9", "fact": "x"}], sentences)) == 2


def test_atomic_keeps_a_usable_multi_fact_decomposition():
    facts = ev.normalize_atomic_facts(
        [{"sentence_num": "S0", "fact_id": "S0-F0", "fact": "A."},
         {"sentence_num": "S0", "fact_id": "S0-F1", "fact": "B."}],
        ["A and B."],
    )
    assert [item["fact_id"] for item in facts] == ["S0-F0", "S0-F1"]


def test_factuality_strips_unterminated_think_and_repairs_commas():
    # Everything after an unterminated <think> is dropped, leaving no JSON at all,
    # which is a parse failure rather than an empty judgment.
    assert ev.parse_factuality('<think>cut off [{"sentence": "x"}') is None
    assert ev.parse_factuality('[{"sentence": "x", "category": "no error"},]') == [
        {"sentence": "x", "category": "no error"}
    ]


@pytest.mark.parametrize("alias", ["noerror", "no_error", "factual", "none", "no-error", "NO ERROR"])
def test_factuality_aliases_map_to_no_error(alias):
    assert ev.normalize_category(alias) == "no error"


def test_unknown_factuality_category_becomes_other_error():
    assert ev.normalize_category("wildly novel error") == "other error"
    assert ev.normalize_category(None) == "other error"


def test_factuality_category_order_is_fixed():
    assert ev.FACTUALITY_CATEGORIES == (
        "no error", "out-of-context error", "entity error", "predicate error",
        "circumstantial error", "grammatical error", "coreference error",
        "linking error", "other error",
    )


def test_factuality_maps_each_echo_onto_exactly_one_fact():
    facts = [{"fact_id": "S0-F0", "text": "A."}, {"fact_id": "S1-F0", "text": "B."}]
    assert ev.normalize_factuality(
        [{"sentence": "B.", "category": "entity error"},
         {"sentence": "A.", "category": "no error"}], facts,
    ) == [{"fact_id": "S0-F0", "category": "no error"},
          {"fact_id": "S1-F0", "category": "entity error"}]
    assert ev.normalize_factuality([{"sentence": "A.", "category": "no error"}], facts) == []
    assert ev.normalize_factuality(
        [{"sentence": "A.", "category": "no error"},
         {"sentence": "A.", "category": "no error"}], facts) == []


# ------------------------------------------------------------------ rendering

def test_mr_renderer_maps_the_record_not_only_labels():
    row = ev.load_benchmark(BENCHMARK).rows[0]
    rendered = json.loads(ev.render_mr_gaps(row["gaps"]))
    assert [item["gap_id"] for item in rendered] == ["gap_0", "gap_3"]
    assert rendered[0]["supporting_reasoning"] == row["gaps"][0]["gap_reason"]
    assert [item["source_id"] for item in rendered[0]["supporting_sentences"]] == ["T0-S1", "T1-S1"]
    assert rendered[0]["supporting_sentences"][0]["text"] == "The migration deadline is March 14."


def test_window_lines_use_the_canonical_shape():
    row = ev.load_benchmark(BENCHMARK).rows[1]
    assert ev.render_window(row["turns"]) == "[2024-01-01 09:00:00] Dee: Latte knocked the plant over again."


def test_summary_numbering_bases_differ_by_metric():
    assert ev.render_summary_lines(["A.", "B."], start=1) == "S1: A.\nS2: B."
    assert ev.render_summary_lines(["A.", "B."], start=0) == "S0: A.\nS1: B."


def test_coverage_renderers_number_gaps_and_salients_in_released_order():
    row = ev.load_benchmark(BENCHMARK).rows[0]
    assert ev.render_gap_cases(row["gaps"]).startswith("G0:")
    assert "G1:" in ev.render_gap_cases(row["gaps"])          # G# is positional, not gap_idx
    assert ev.render_salient_utterances(row["salient_utterances"]).startswith("U0: ")


def test_factuality_transcript_is_memory_then_current_window():
    row = ev.load_benchmark(BENCHMARK).rows[1]
    transcript = ev.render_factuality_transcript("MEM", row["turns"])
    assert transcript.startswith("[MEMORY]\nMEM\n\n[CURRENT WINDOW]\n[2024-01-01")


def test_checker_facts_are_numbered_from_one():
    assert ev.render_numbered_facts([{"text": "A."}, {"text": "B."}]) == "1. A.\n2. B."


# ------------------------------------------------------------------ providers

def test_fake_provider_decides_from_literal_containment_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    benchmark = ev.load_benchmark(BENCHMARK)
    predictions = ev.load_predictions(PREDICTIONS, benchmark.ordered_ids, allow_partial=True)
    prompts = ev.load_prompts(PROMPT_DIR)
    per_window, raw = ev.evaluate_predictions(
        list(benchmark.rows), predictions, prompts, ev.FakeProvider(),
        metrics=list(ev.REPORTED_METRICS), metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8,
    )
    first = {row["window_id"]: row for row in per_window}["E_U0001"]
    # gap_0's evidence is verbatim in memory; gap_3's is not.
    assert first["mr"]["decisions"] == {"0": 1, "3": 0}
    assert first["factuality"]["categories"]["out-of-context error"] == 1
    assert all(row["parse_ok"] for row in raw)
    assert "vllm" not in sys.modules and "openai" not in sys.modules
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "sentinel"


def test_openai_provider_sends_the_paper_request(monkeypatch):
    import types
    seen = {}

    class _Message:
        content = '[{"gap_id": "gap_0", "decision": 1}]'

    class _Choice:
        message = _Message()

    class _Usage:
        prompt_tokens, completion_tokens = 12, 3

    class _Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(choices=[_Choice()], usage=_Usage())

    class _OpenAI:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = ev.build_provider(_args(provider="openai", base_url="http://local",
                                       api_key_env="TEST_JUDGE_KEY"))
    [completion] = provider.complete_batch(["prompt"], max_tokens=2048)
    assert seen["model"] == "Qwen/Qwen3.6-27B"
    assert seen["temperature"] == 0.0
    assert seen["messages"] == [{"role": "user", "content": "prompt"}]
    assert seen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert completion.prompt_tokens == 12 and completion.completion_tokens == 3


def test_openai_provider_requires_only_the_named_key(monkeypatch):
    monkeypatch.delenv("TEST_JUDGE_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated")
    with pytest.raises(ev.InputError, match="TEST_JUDGE_KEY"):
        ev.build_provider(_args(provider="openai", base_url="http://local",
                                api_key_env="TEST_JUDGE_KEY"))


def test_openai_provider_retries_then_reports_a_bounded_failure(monkeypatch):
    import types
    attempts = {"n": 0}

    class _Completions:
        def create(self, **kwargs):
            attempts["n"] += 1
            raise RuntimeError("temporary upstream failure")

    class _OpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")
    provider = ev.build_provider(_args(provider="openai", base_url="http://local",
                                       api_key_env="TEST_JUDGE_KEY", max_retries=2))
    [completion] = provider.complete_batch(["p"], max_tokens=16)
    assert attempts["n"] == 3
    assert completion.error and completion.retry_count == 2
    assert len(completion.attempt_errors) == 3


def test_openai_provider_reports_a_thinking_incompatibility_clearly(monkeypatch):
    import types

    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("unknown field: chat_template_kwargs")

    class _OpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")
    provider = ev.build_provider(_args(provider="openai", base_url="http://local",
                                       api_key_env="TEST_JUDGE_KEY"))
    # Reported as an exhausted request rather than raised mid-batch, so the
    # caller can checkpoint everything already paid for before aborting.
    [completion] = provider.complete_batch(["p"], max_tokens=16)
    assert "no-thinking" in completion.error
    assert completion.retry_count == 0 and completion.text == ""


def test_vllm_provider_sets_visibility_before_importing(monkeypatch):
    import types
    order = []

    class _LLM:
        def __init__(self, **kwargs):
            import os
            order.append(("import-time-devices", os.environ.get("CUDA_VISIBLE_DEVICES")))
            self.kwargs = kwargs

        def chat(self, conversations, params, **kwargs):
            order.append(("chat", len(conversations), kwargs))
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="[]")],
                                          prompt_token_ids=[1, 2])
                    for _ in conversations]

    def _sampling(**kwargs):
        order.append(("sampling", kwargs))
        return kwargs

    monkeypatch.setitem(sys.modules, "vllm",
                        types.SimpleNamespace(LLM=_LLM, SamplingParams=_sampling))
    provider = ev.build_provider(_args(provider="vllm", gpu_devices="2,3"))
    provider.complete_batch(["a", "b"], max_tokens=64)
    assert order[0] == ("import-time-devices", "2,3")
    assert ("sampling", {"temperature": 0.0, "max_tokens": 64}) in order
    assert order[-1] == ("chat", 2, {"chat_template_kwargs": {"enable_thinking": False}})


# ------------------------------------------------------------------- scoring

def _window(**kwargs):
    return {"window_id": kwargs.pop("window_id", "W"), "summary_sentences": 2, **kwargs}


def test_micro_and_macro_arithmetic_is_literal():
    per_window = [
        _window(window_id="A", mr={"numerator": 2, "denominator": 2, "parse_failure": False,
                                   "decisions": {}}),
        _window(window_id="B", mr={"numerator": 0, "denominator": 1, "parse_failure": True,
                                   "decisions": {}}),
    ]
    scores = ev.score_run(per_window)
    assert scores["mr"]["micro"] == {
        "numerator": 2, "denominator": 3, "value": pytest.approx(2 / 3),
        "aggregation": "micro", "parse_failure_count": 1,
        "windows_with_items": 2, "windows_without_items": 0,
    }
    assert scores["mr"]["macro"]["value"] == pytest.approx(0.5)


def test_macro_skips_windows_without_items():
    per_window = [
        _window(window_id="A", wkf={"numerator": 1.0, "denominator": 2, "parse_failure": False}),
        _window(window_id="B", wkf={"numerator": 0.0, "denominator": 0, "parse_failure": False}),
    ]
    scores = ev.score_run(per_window)
    assert scores["wkf"]["macro"]["denominator"] == 1
    assert scores["wkf"]["macro"]["value"] == pytest.approx(0.5)
    assert scores["wkf"]["micro"]["windows_without_items"] == 1


def test_conciseness_counts_unique_positive_lines_across_mkf_and_wkf():
    benchmark = ev.load_benchmark(BENCHMARK)
    predictions = ev.load_predictions(PREDICTIONS, benchmark.ordered_ids, allow_partial=True)
    per_window, _ = ev.evaluate_predictions(
        list(benchmark.rows), predictions, ev.load_prompts(PROMPT_DIR), ev.FakeProvider(),
        metrics=["conciseness"], metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8,
    )
    scores = ev.score_run(per_window)
    assert scores["conciseness"]["micro"]["numerator"] == 1
    assert scores["conciseness"]["micro"]["denominator"] == 4


def test_mr_gated_mkf_joins_strictly_on_window_and_gap_idx():
    per_window = [_window(
        window_id="A",
        mr={"numerator": 1, "denominator": 2, "parse_failure": False, "decisions": {"0": 1, "3": 0}},
        mkf={"numerator": 1.0, "denominator": 2, "parse_failure": False,
             "scores": {"0": 1.0, "3": 0.0}},
        mr_gated_mkf={"numerator": 1.0, "denominator": 1, "parse_failure": False},
    )]
    scores = ev.score_run(per_window)
    assert scores["mr_gated_mkf"]["diagnostic"] is True
    assert scores["mr_gated_mkf"]["micro"]["value"] == pytest.approx(1.0)


def test_factuality_reports_all_nine_categories_and_phase_failures():
    per_window = [_window(window_id="A", factuality={
        "numerator": 1, "denominator": 2,
        "categories": {name: (1 if name in ("no error", "entity error") else 0)
                       for name in ev.FACTUALITY_CATEGORIES},
        "atomic_parse_failure": True, "checker_parse_failure": False,
    })]
    scores = ev.score_run(per_window)
    assert set(scores["factuality"]["categories"]) == set(ev.FACTUALITY_CATEGORIES)
    assert scores["factuality"]["parse_failures_by_phase"] == {"atomic": 1, "checker": 0}
    assert scores["factuality"]["micro"]["parse_failure_count"] == 1


def test_metric_dependency_closure():
    assert ev.expand_metric_dependencies(["conciseness"]) == ["mkf", "wkf"]
    assert ev.expand_metric_dependencies(["mr_gated_mkf"]) == ["mr", "mkf"]
    assert ev.expand_metric_dependencies(list(ev.REPORTED_METRICS)) == list(ev.INFERENCE_METRICS)


# ------------------------------------------------------------- CLI validation

@pytest.mark.parametrize("overrides,message", [
    ({"metrics": "all,mr"}, "cannot be combined"),
    ({"metrics": "mr,mr"}, "duplicate"),
    ({"metrics": "nope"}, "unknown metric"),
    ({"metrics": ""}, "at least one"),
    ({"batch_size": 0}, "batch-size"),
    ({"temperature": -0.1}, "temperature"),
    ({"max_retries": -1}, "max-retries"),
    ({"gpu_memory_utilization": 0}, "gpu-memory-utilization"),
    ({"gpu_memory_utilization": 1.5}, "gpu-memory-utilization"),
    ({"provider": "vllm", "gpu_devices": None}, "--gpu-devices is required"),
    ({"provider": "vllm", "gpu_devices": "0,0"}, "unique"),
    ({"provider": "vllm", "gpu_devices": "0,x"}, "unique non-negative"),
    ({"provider": "vllm", "gpu_devices": "0", "tensor_parallel_size": 2}, "cannot exceed"),
    ({"provider": "openai", "base_url": " "}, "--base-url is required"),
    ({"provider": "openai", "base_url": "http://x", "api_key_env": "1BAD"}, "environment variable"),
])
def test_argument_validation_rejects_bad_input(overrides, message):
    with pytest.raises(ev.InputError, match=message):
        ev.validate_args(_args(**overrides))


def test_metrics_resolve_in_reported_order():
    assert ev.resolve_metrics("wkf,mr") == ["mr", "wkf"]
    assert ev.resolve_metrics("all") == list(ev.REPORTED_METRICS)


def test_parser_defaults_match_the_documented_contract():
    args = ev.build_parser().parse_args(
        ["--benchmark", "b", "--predictions", "p", "--output-dir", "o"])
    assert (args.provider, args.model, args.metrics) == ("vllm", "Qwen/Qwen3.6-27B", "all")
    assert (args.temperature, args.batch_size, args.max_model_len) == (0.0, 64, 25000)
    assert (args.gpu_memory_utilization, args.tensor_parallel_size) == (0.8, 1)
    assert (args.api_key_env, args.max_retries) == ("OPENAI_API_KEY", 2)
    assert (args.mr_max_tokens, args.coverage_max_tokens) == (2048, 4096)


def test_resume_and_overwrite_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        ev.build_parser().parse_args(
            ["--benchmark", "b", "--predictions", "p", "--output-dir", "o",
             "--resume", "--overwrite"])


# ----------------------------------------------------------------- end to end

def _run_cli(output_dir, *extra):
    return subprocess.run(
        [sys.executable, str(EVALUATOR), "--benchmark", str(BENCHMARK),
         "--predictions", str(PREDICTIONS), "--output-dir", str(output_dir),
         "--provider", "fake", "--allow-partial", *extra],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_end_to_end_produces_exactly_the_five_artifacts(tmp_path):
    output = tmp_path / "results"
    completed = _run_cli(output)
    assert completed.returncode == 0, completed.stderr
    assert sorted(p.name for p in output.iterdir()) == sorted(ev.FINAL_FILES)
    scores = json.loads((output / "scores.json").read_text())
    assert scores == json.loads((FIXTURES / "golden_scores.json").read_text())
    for metric in ev.REPORTED_METRICS:
        assert metric in completed.stdout
    assert (output / "scores.csv").read_text().startswith(
        "metric,aggregation,value,numerator,denominator,")
    per_window = [json.loads(line) for line in
                  (output / "per_window.jsonl").read_text().splitlines() if line.strip()]
    assert [row["window_id"] for row in per_window] == ["E_U0001", "L_U0002"]
    raw = [json.loads(line) for line in
           (output / "raw_judgments.jsonl").read_text().splitlines() if line.strip()]
    assert raw and all("prompt" not in row for row in raw)
    assert all(row["prompt_sha256"] and row["completed"] for row in raw)
    assert not list(tmp_path.glob(".results.in_progress"))


def test_run_manifest_records_provenance_and_is_not_the_official_preset(tmp_path):
    output = tmp_path / "results"
    assert _run_cli(output).returncode == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["provenance"]["canonical_prompts"] is True
    assert manifest["provenance"]["paper_model_temperature"] is True
    assert manifest["provenance"]["canonical_benchmark"] is False   # two-window fixture
    assert manifest["provenance"]["scientific_provider"] is False   # fake judge
    assert manifest["provenance"]["official_preset"] is False
    assert manifest["immutable"]["evaluated_population"] == 2
    assert set(manifest["output_hashes"]) == {
        "scores.json", "scores.csv", "per_window.jsonl", "raw_judgments.jsonl"}


def test_existing_results_block_a_normal_rerun(tmp_path):
    output = tmp_path / "results"
    assert _run_cli(output).returncode == 0
    again = _run_cli(output)
    assert again.returncode == 2 and "already contains results" in again.stderr


def test_overwrite_replaces_a_complete_result(tmp_path):
    output = tmp_path / "results"
    assert _run_cli(output).returncode == 0
    before = (output / "scores.json").read_text()
    assert _run_cli(output, "--overwrite").returncode == 0
    assert (output / "scores.json").read_text() == before
    assert sorted(p.name for p in output.iterdir()) == sorted(ev.FINAL_FILES)
    assert not list(tmp_path.glob(".results.*"))


class _CountingProvider:
    def __init__(self):
        self.calls = 0
        self._inner = ev.FakeProvider()

    def complete_batch(self, prompts, *, max_tokens):
        self.calls += len(prompts)
        return self._inner.complete_batch(prompts, max_tokens=max_tokens)


def test_resume_reuses_every_completed_request_without_new_calls(tmp_path):
    benchmark = ev.load_benchmark(BENCHMARK)
    predictions = ev.load_predictions(PREDICTIONS, benchmark.ordered_ids, allow_partial=True)
    prompts = ev.load_prompts(PROMPT_DIR)
    first = _CountingProvider()
    _, raw = ev.evaluate_predictions(
        list(benchmark.rows), predictions, prompts, first,
        metrics=list(ev.REPORTED_METRICS), metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8)
    assert first.calls > 0
    second = _CountingProvider()
    _, replayed = ev.evaluate_predictions(
        list(benchmark.rows), predictions, prompts, second,
        metrics=list(ev.REPORTED_METRICS), metric_max_tokens=ev.METRIC_MAX_TOKENS,
        batch_size=8, cached_raw=raw)
    assert second.calls == 0
    assert [row["request_id"] for row in replayed] == [row["request_id"] for row in raw]


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(completed=False),
    lambda row: row.update(error="boom"),
    lambda row: row.update(prompt_sha256="0" * 64),
    lambda row: row.update(normalized_input_sha256="0" * 64),
    lambda row: row.update(expected_item_ids=["gap_9"]),
    lambda row: row.update(expected_items=99),
])
def test_unusable_cached_rows_are_reissued(tmp_path, mutation):
    benchmark = ev.load_benchmark(BENCHMARK)
    predictions = ev.load_predictions(PREDICTIONS, benchmark.ordered_ids, allow_partial=True)
    prompts = ev.load_prompts(PROMPT_DIR)
    _, raw = ev.evaluate_predictions(
        list(benchmark.rows), predictions, prompts, ev.FakeProvider(),
        metrics=["mr"], metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8)
    damaged = [dict(row) for row in raw]
    mutation(damaged[0])
    provider = _CountingProvider()
    ev.evaluate_predictions(
        list(benchmark.rows), predictions, prompts, provider,
        metrics=["mr"], metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8,
        cached_raw=damaged)
    assert provider.calls == 1


@pytest.mark.parametrize("field,value", [
    ("predictions_sha256", "0" * 64),
    ("model", "other/model"),
    ("temperature", 0.7),
    ("provider", "openai"),
    ("requested_metrics", ["mr"]),
    ("allow_partial", False),
    ("evaluated_ids_sha256", "0" * 64),
])
def test_resume_manifest_rejects_any_immutable_change(field, value):
    benchmark = ev.load_benchmark(BENCHMARK)
    current = ev.build_run_manifest(
        benchmark, PREDICTIONS, dict(ev.CANONICAL_PROMPT_HASHES), _args(),
        list(ev.REPORTED_METRICS), list(ev.INFERENCE_METRICS),
        list(benchmark.ordered_ids), status="in_progress")
    stale = json.loads(json.dumps(current))
    stale["immutable"][field] = value
    with pytest.raises(ev.ResumeError, match=field):
        ev.validate_resume_manifest(stale, current)
    ev.validate_resume_manifest(current, current)


def test_finalization_failure_leaves_the_previous_result_intact(tmp_path, monkeypatch):
    output = tmp_path / "results"
    assert _run_cli(output).returncode == 0
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    monkeypatch.setattr(ev, "_reload_and_validate", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("injected finalization failure")))
    with pytest.raises(RuntimeError, match="injected finalization failure"):
        ev.execute(_args(output_dir=output, overwrite=True))
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not list(tmp_path.glob(".results.staged-*"))
    assert not list(tmp_path.glob(".results.backup-*"))


def test_finalization_failure_keeps_the_resume_point(tmp_path, monkeypatch):
    output = tmp_path / "results"
    work = tmp_path / ".results.in_progress"
    monkeypatch.setattr(ev, "_reload_and_validate", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("injected finalization failure")))
    with pytest.raises(RuntimeError, match="injected finalization failure"):
        ev.execute(_args(output_dir=output))
    # Without the checkpoint directory a --resume would have to re-run every request.
    assert work.is_dir()
    assert (work / "raw_judgments.jsonl").is_file()
    assert (work / "run_manifest.json").is_file()
    assert not output.exists()


def test_successful_run_removes_the_resume_point(tmp_path):
    output = tmp_path / "results"
    assert ev.execute(_args(output_dir=output)) == 0
    assert not (tmp_path / ".results.in_progress").exists()


def test_a_failed_request_checkpoints_the_whole_batch_before_aborting(tmp_path):
    benchmark = ev.load_benchmark(BENCHMARK)
    predictions = ev.load_predictions(PREDICTIONS, benchmark.ordered_ids, allow_partial=True)
    saved = []

    class _FailsSecond:
        def __init__(self):
            self._inner = ev.FakeProvider()

        def complete_batch(self, prompts, *, max_tokens):
            done = self._inner.complete_batch(prompts, max_tokens=max_tokens)
            return [done[0]] + [ev.Completion(text="", prompt_tokens=None, completion_tokens=None,
                                             retry_count=2, attempt_errors=("boom",), error="boom")
                                for _ in done[1:]]

    with pytest.raises(RuntimeError, match="failed after retries"):
        ev.evaluate_predictions(
            list(benchmark.rows), predictions, ev.load_prompts(PROMPT_DIR), _FailsSecond(),
            metrics=["mr"], metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8,
            checkpoint=lambda rows: saved.append([dict(row) for row in rows]))
    assert saved, "the batch must be checkpointed before the abort"
    final = {row["request_id"]: row for row in saved[-1]}
    assert len(final) == 2
    assert [row["completed"] for row in final.values()] == [True, False]
    assert any(row["error"] == "boom" for row in final.values())


def test_resume_carries_prior_attempt_errors_into_the_audit_row():
    benchmark = ev.load_benchmark(BENCHMARK)
    predictions = ev.load_predictions(PREDICTIONS, benchmark.ordered_ids, allow_partial=True)
    prompts = ev.load_prompts(PROMPT_DIR)
    _, raw = ev.evaluate_predictions(
        list(benchmark.rows), predictions, prompts, ev.FakeProvider(),
        metrics=["mr"], metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8)
    failed = [dict(row) for row in raw]
    failed[0].update(completed=False, error="earlier failure",
                     attempt_errors=["earlier failure"], raw_output="")
    _, replayed = ev.evaluate_predictions(
        list(benchmark.rows), predictions, prompts, ev.FakeProvider(),
        metrics=["mr"], metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8,
        cached_raw=failed)
    row = {item["request_id"]: item for item in replayed}[failed[0]["request_id"]]
    assert row["completed"] is True and row["error"] is None
    assert "earlier failure" in row["attempt_errors"]


def test_a_prediction_cannot_inject_a_placeholder_or_unescape_its_own_braces():
    template = "{N} lines:\n{summary}\n\n{M} cases:\n{gap_cases}"
    rendered = ev.fill_prompt(template, N=1, summary="S1: uses {{db}} and {gap_cases}",
                              M=1, gap_cases="G0: gold answer")
    assert "gold answer" not in rendered.split("1 cases:")[0]
    assert "{{db}}" in rendered


def test_out_of_range_line_citations_invalidate_the_coverage_judgment():
    items = [{"utterance_num": "U0", "score": 1.0, "line_number": ["S1", "S99"]}]
    assert ev.normalize_coverage(items, ["U0"], id_field="utterance_num", max_line=2) is None
    assert ev.normalize_coverage(items, ["U0"], id_field="utterance_num", max_line=99) is not None
    zero = [{"utterance_num": "U0", "score": 1.0, "line_number": ["S0"]}]
    assert ev.normalize_coverage(zero, ["U0"], id_field="utterance_num", max_line=3) is None


def test_the_fake_provider_refuses_an_ambiguous_prompt():
    with pytest.raises(AssertionError, match="could not identify"):
        ev.FakeProvider().complete_batch(["nothing recognizable here"], max_tokens=16)
    prompts = ev.load_prompts(PROMPT_DIR)
    for name, template in prompts.items():
        matched = [marker for marker, _ in ev.FakeProvider.MARKERS if marker in template]
        assert len(matched) == 1, (name, matched)


def test_only_requested_metrics_are_reported(tmp_path):
    benchmark = ev.load_benchmark(BENCHMARK)
    predictions = ev.load_predictions(PREDICTIONS, benchmark.ordered_ids, allow_partial=True)
    per_window, _ = ev.evaluate_predictions(
        list(benchmark.rows), predictions, ev.load_prompts(PROMPT_DIR), ev.FakeProvider(),
        metrics=["mkf"], metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8)
    scores = ev.score_run(per_window)
    assert "mkf" in scores and "conciseness" not in scores and "mr_gated_mkf" not in scores


def test_results_are_readable_by_other_users(tmp_path):
    import stat
    output = tmp_path / "results"
    assert ev.execute(_args(output_dir=output)) == 0
    assert stat.S_IMODE(output.stat().st_mode) & stat.S_IROTH
    for name in ev.FINAL_FILES:
        assert stat.S_IMODE((output / name).stat().st_mode) & stat.S_IROTH, name


def test_a_cached_row_without_its_output_is_reissued():
    benchmark = ev.load_benchmark(BENCHMARK)
    predictions = ev.load_predictions(PREDICTIONS, benchmark.ordered_ids, allow_partial=True)
    prompts = ev.load_prompts(PROMPT_DIR)
    _, raw = ev.evaluate_predictions(
        list(benchmark.rows), predictions, prompts, ev.FakeProvider(),
        metrics=["mr"], metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8)
    damaged = [dict(row) for row in raw]
    del damaged[0]["raw_output"]
    provider = _CountingProvider()
    ev.evaluate_predictions(
        list(benchmark.rows), predictions, prompts, provider,
        metrics=["mr"], metric_max_tokens=ev.METRIC_MAX_TOKENS, batch_size=8, cached_raw=damaged)
    assert provider.calls == 1


@pytest.mark.parametrize("damage,message", [
    ("missing", "nothing to resume"),
    ("corrupt_manifest", "unreadable"),
    ("checkpoint_hash", "does not match its recorded hash"),
    ("malformed_checkpoint", "malformed"),
])
def test_resume_rejects_unusable_interrupted_state(tmp_path, damage, message):
    output = tmp_path / "results"
    work = tmp_path / ".results.in_progress"
    if damage != "missing":
        work.mkdir(parents=True)
        args = _args(output_dir=output)
        benchmark = ev.load_benchmark(BENCHMARK)
        manifest = ev.build_run_manifest(
            benchmark, PREDICTIONS, {n: ev.sha256_text(t) for n, t in ev.load_prompts(PROMPT_DIR).items()},
            args, list(ev.REPORTED_METRICS), list(ev.INFERENCE_METRICS),
            list(benchmark.ordered_ids), status="in_progress")
        (work / "raw_judgments.jsonl").write_text('{"request_id": "mr:judge:E_U0001"}\n')
        if damage == "corrupt_manifest":
            (work / "run_manifest.json").write_text("not json\n")
        elif damage == "checkpoint_hash":
            manifest["checkpoint_sha256"] = "0" * 64
            (work / "run_manifest.json").write_text(json.dumps(manifest))
        else:
            manifest["checkpoint_sha256"] = ev.sha256_file(work / "raw_judgments.jsonl")
            (work / "run_manifest.json").write_text(json.dumps(manifest))
            (work / "raw_judgments.jsonl").write_text("not json\n")
            manifest["checkpoint_sha256"] = ev.sha256_file(work / "raw_judgments.jsonl")
            (work / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ev.ResumeError, match=message):
        ev.execute(_args(output_dir=output, resume=True))


def test_mr_gap_item_matches_the_canonical_shape():
    """Keys and values as the canonical memory-recall evaluator builds them."""
    row = ev.load_benchmark(BENCHMARK).rows[0]
    [first, second] = json.loads(ev.render_mr_gaps(row["gaps"]))
    assert list(first) == ["gap_id", "salient_utterance", "gap_type", "gap_reason",
                           "supporting_sentences", "supporting_reasoning"]
    assert first["gap_id"] == "gap_0" and second["gap_id"] == "gap_3"
    assert first["salient_utterance"] == "She will finish it before the deadline."
    assert first["gap_reason"] == first["supporting_reasoning"] == row["gaps"][0]["gap_reason"]
    assert first["supporting_sentences"] == [
        {"source_id": "T0-S1", "text": "The migration deadline is March 14."},
        {"source_id": "T1-S1", "text": "Ada owns the billing service."},
    ]


def test_mkf_gap_case_block_matches_the_canonical_shape():
    """Layout as the canonical coverage evaluator builds it."""
    row = ev.load_benchmark(BENCHMARK).rows[0]
    blocks = ev.render_gap_cases(row["gaps"]).split("\n\n")
    assert blocks[0] == (
        "G0:\n"
        "- utterance_num: U0\n"
        "- salient_utterance: She will finish it before the deadline.\n"
        "- gap_type: C\n"
        "- relevant_chunk: The migration deadline is March 14.\n"
        "Ada owns the billing service."
    )
    # G# is positional, and gap_3 becomes G1/U1 - never gap_idx 3.
    assert blocks[1].startswith("G1:\n- utterance_num: U1\n")


def test_wkf_lines_match_the_canonical_shape():
    row = ev.load_benchmark(BENCHMARK).rows[0]
    assert ev.render_salient_utterances(row["salient_utterances"]) == (
        "U0: She will finish it before the deadline.\n"
        "U1: That matches the plan we agreed."
    )


def test_factuality_transcript_and_fact_numbering_match_the_canonical_shape():
    """Memory-then-window transcript, and the checker's one-based fact numbering."""
    row = ev.load_benchmark(BENCHMARK).rows[1]
    assert ev.render_factuality_transcript("MEM", row["turns"]) == (
        "[MEMORY]\nMEM\n\n[CURRENT WINDOW]\n"
        "[2024-01-01 09:00:00] Dee: Latte knocked the plant over again."
    )
    assert ev.render_numbered_facts([{"text": "A."}, {"text": "B."}]) == "1. A.\n2. B."
