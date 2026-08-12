import json

import pytest

from remember_benchmark.chunk_filter import build_history_chunks, run_chunk_filter
from remember_benchmark.ensemble import ensemble_chunk_votes, ensemble_salient_votes, ensemble_votes
from remember_benchmark.evidence import run_evidence
from remember_benchmark.providers import ScriptedProvider
from remember_benchmark.salient import run_salient
from remember_benchmark.validation import run_validation


WINDOW = {
    "schema_version": "1.0",
    "window_id": "E_U0000",
    "source": "evermembench",
    "range": "8k~16k",
    "history": [
        {"time": "2024-01-01 00:00:00", "speaker": "A", "dialogue": "red blue"},
        {"time": "2024-01-01 00:00:01", "speaker": "B", "dialogue": "green yellow"},
        {"time": "2024-01-01 00:00:02", "speaker": "A", "dialogue": "orange purple"},
    ],
    "turns": [
        {"time": "2024-01-01 00:01:00", "speaker": "B", "dialogue": "That plan works. What comes next?"}
    ],
}


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def _response(payload):
    return json.dumps({"schema_version": "1.0", **payload})


def test_salient_parses_versioned_candidates_and_checks_sentence_coordinates():
    provider = ScriptedProvider([_response({"candidates": [{
        "sentence_numbers": [0], "sentences": ["That plan works."], "reason": "depends on an earlier plan", "gap_type": "A"
    }]})])

    rows = run_salient([WINDOW], provider, "Inspect {window}")

    assert rows[0]["gaps"] == [{
        "gap_idx": 0,
        "target_sentence_numbers": [0],
        "target_sentence": ["That plan works."],
        "gap_reason": "depends on an earlier plan",
        "gap_type": "A",
    }]
    assert provider.calls[0].startswith("Inspect [0]. [2024-01-01")


def test_history_chunks_use_128_32_defaults_and_keep_stable_boundaries():
    turns = [
        {"time": f"t{i}", "speaker": "A", "dialogue": " ".join([f"w{i}"] * 50)}
        for i in range(6)
    ]
    chunks = build_history_chunks(turns, WordCounter())
    assert [(item["start_turn"], item["end_turn"]) for item in chunks] == [(0, 1), (2, 3), (4, 5)]
    assert all(item["max_tokens"] == 128 and item["overlap_tokens"] == 32 for item in chunks)


def test_chunk_filter_requires_one_binary_label_per_chunk():
    provider = ScriptedProvider([_response({"labels": [1, 0]})])
    row = {**WINDOW, "history": [
        {"time": "t0", "speaker": "A", "dialogue": "one two"},
        {"time": "t1", "speaker": "B", "dialogue": "three four"},
    ]}
    result = run_chunk_filter([row], provider, "Filter {chunks}", counter=WordCounter(), chunk_tokens=4, overlap_tokens=0)
    assert result[0]["relevant_chunks"] == [{"start_turn": 0, "end_turn": 0}]

    bad = ScriptedProvider([_response({"labels": [1]})])
    with pytest.raises(ValueError, match="label cardinality"):
        run_chunk_filter([row], bad, "Filter {chunks}", counter=WordCounter(), chunk_tokens=4, overlap_tokens=0)


def test_chunk_filter_limits_each_provider_call_to_sixteen_chunks():
    row = {**WINDOW, "history": [
        {"time": f"t{i}", "speaker": "A", "dialogue": f"word{i}"} for i in range(17)
    ]}
    provider = ScriptedProvider([
        _response({"labels": [0] * 16}),
        _response({"labels": [1]}),
    ])
    result = run_chunk_filter(
        [row], provider, "Filter {chunks}", counter=WordCounter(), chunk_tokens=4, overlap_tokens=0
    )
    assert len(provider.calls) == 2
    assert result[0]["relevant_chunks"] == [{"start_turn": 16, "end_turn": 16}]


def test_evidence_emits_sentence_coordinates_and_validation_marks_returned_ids():
    salient = [{
        "schema_version": "1.0", "window_id": "E_U0000", "range": "8k~16k",
        "gaps": [{"gap_idx": 0, "target_sentence_numbers": [0], "target_sentence": ["That plan works."], "gap_reason": "missing", "gap_type": "A"}],
    }]
    filtered = [{"schema_version": "1.0", "window_id": "E_U0000", "relevant_chunks": [{"start_turn": 0, "end_turn": 1}]}]
    evidence_provider = ScriptedProvider([_response({"evidence": [
        {"source_id": "T0-S1", "sentence": "red blue", "reasoning": "explicit", "status": "resolve"},
        {"source_id": "T1-S1", "sentence": "green yellow", "reasoning": "related only", "status": "not_resolve"},
    ]})])
    evidence = run_evidence([WINDOW], salient, filtered, evidence_provider, "Resolve {history} {gap}")
    assert [item["source_id"] for item in evidence[0]["gaps"][0]["evidence"]] == ["T0-S1", "T1-S1"]

    validation_provider = ScriptedProvider([_response({"valid_source_ids": ["T0-S1"]})])
    validated = run_validation([WINDOW], evidence, validation_provider, "Validate {candidates}")
    assert [(item["source_id"], item["validation"]) for item in validated[0]["gaps"][0]["validated"]] == [
        ("T0-S1", 1)
    ]


@pytest.mark.parametrize("stage,call", [
    ("salient", lambda p: run_salient([WINDOW], p, "{window}")),
    ("chunk_filter", lambda p: run_chunk_filter([WINDOW], p, "{chunks}", counter=WordCounter())),
])
def test_malformed_provider_output_is_rejected(stage, call):
    provider = ScriptedProvider(["not-json"])
    with pytest.raises(ValueError, match=rf"{stage}.*E_U0000.*JSON"):
        call(provider)


def test_ensembles_apply_paper_thresholds_and_final_votes_are_text_free():
    salient_by_model = {
        name: [{"window_id": "E_U0000", "gaps": [{"target_sentence_numbers": [0], "gap_type": gap_type}]}]
        for name, gap_type in (("gemma", "A"), ("GLM", "A"), ("Qwen", "C"))
    }
    assert ensemble_salient_votes(salient_by_model)[0]["gaps"] == [
        {"gap_idx": 0, "target_sentence_numbers": [0], "gap_type": "A"}
    ]

    chunks_by_model = {
        name: [{"window_id": "E_U0000", "relevant_chunks": chunks}]
        for name, chunks in (
            ("gemma", [{"start_turn": 0, "end_turn": 2}]),
            ("GLM", [{"start_turn": 0, "end_turn": 2}]),
            ("Qwen", [{"start_turn": 0, "end_turn": 2}, {"start_turn": 3, "end_turn": 4}]),
        )
    }
    assert ensemble_chunk_votes(chunks_by_model)[0]["relevant_chunks"] == [{"start_turn": 0, "end_turn": 2}]

    def model_row(valid):
        return [{
            "window_id": "E_U0000", "source": "evermembench", "range": "8k~16k",
            "gaps": [{"gap_idx": 0, "target_sentence_numbers": [0], "gap_type": "A", "validated": [
                {"source_id": "T0-S1", "sentence": "private text", "reasoning": "private reason", "status": "resolve", "validation": valid}
            ]}],
        }]
    final = ensemble_votes({"gemma": model_row(1), "GLM": model_row(1), "Qwen": model_row(0)}, threshold=2)
    assert final == [{
        "schema_version": "1.0", "window_id": "E_U0000", "source": "evermembench", "range": "8k~16k",
        "gaps": [{"gap_idx": 0, "target_sentence_numbers": [0], "gap_type": "A", "evidence": [{
            "source_id": "T0-S1", "status": "resolve", "vote_count": 2,
            "votes": {"GLM": 1, "Qwen": 0, "gemma": 1}, "validation": 1,
        }]}],
    }]
    serialized = json.dumps(final)
    assert "private text" not in serialized and "private reason" not in serialized


def test_salient_ensemble_votes_by_sentence_before_retaining_reference_groups():
    rows = {
        "gemma": [{"window_id": "E_U0000", "gaps": [
            {"target_sentence_numbers": [1, 2], "gap_type": "A"},
        ]}],
        "GLM": [{"window_id": "E_U0000", "gaps": [
            {"target_sentence_numbers": [1], "gap_type": "C"},
            {"target_sentence_numbers": [2], "gap_type": "A"},
        ]}],
        "Qwen": [{"window_id": "E_U0000", "gaps": [
            {"target_sentence_numbers": [1], "gap_type": "A"},
            {"target_sentence_numbers": [2], "gap_type": "A"},
        ]}],
    }

    assert ensemble_salient_votes(rows) == [{
        "schema_version": "1.0",
        "window_id": "E_U0000",
        "gaps": [{"gap_idx": 0, "target_sentence_numbers": [1, 2], "gap_type": "A"}],
    }]


def test_salient_ensemble_rejects_duplicate_sentence_vote_within_one_model():
    rows = {
        "gemma": [{"window_id": "E_U0000", "gaps": [
            {"target_sentence_numbers": [1], "gap_type": "A"},
            {"target_sentence_numbers": [1, 2], "gap_type": "A"},
        ]}],
        "GLM": [{"window_id": "E_U0000", "gaps": []}],
        "Qwen": [{"window_id": "E_U0000", "gaps": []}],
    }
    with pytest.raises(ValueError, match="duplicate sentence vote"):
        ensemble_salient_votes(rows)
