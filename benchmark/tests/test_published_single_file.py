"""Regression pins for the single self-contained benchmark file.

The published payload is one gzipped JSONL file holding all 900 windows with
their dialogue text, gold gaps, evidence candidates, and salient utterances.
Every released string is exactly what the canonical evaluators read - MR and
MKF each apply their own `validation` filter to the same evidence list rather
than being shipped two different texts.
"""

import gzip
import hashlib
import json
from pathlib import Path

import pytest


DATASET = Path(__file__).parents[2] / "dataset"
PAYLOAD = DATASET / "remember_900.jsonl.gz"
PAYLOAD_SHA256 = "aacc984848440f6a8a5268106c3f84bd8a672235712b24d75de45fac90df62b5"

EXPECTED_COUNTS = {
    "windows": 900, "gaps": 1498,
    "evidence_candidates": 8450, "evidence_validated": 4135,
    "salient_utterances": 7245,
}
RANGE_VALUES = {"8k~16k", "16k~32k", "32k~64k", "64k~128k", "128k~160k"}
SOURCE_BY_PREFIX = {"E": "evermembench", "L": "locomo", "R": "realtalk"}


def read_ids(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def rows():
    with gzip.open(PAYLOAD, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_published_payload_is_a_single_pinned_file():
    assert PAYLOAD.is_file()
    assert hashlib.sha256(PAYLOAD.read_bytes()).hexdigest() == PAYLOAD_SHA256
    assert sorted(p.name for p in DATASET.glob("*.jsonl.gz")) == ["remember_900.jsonl.gz"]


def test_published_payload_matches_the_ordered_split(rows):
    assert [row["window_id"] for row in rows] == read_ids(DATASET / "splits/main_900.txt")


def test_published_payload_has_the_paper_population(rows):
    evidence = [item for row in rows for gap in row["gaps"] for item in gap["evidence"]]
    counts = {
        "windows": len(rows),
        "gaps": sum(len(row["gaps"]) for row in rows),
        "evidence_candidates": len(evidence),
        "evidence_validated": sum(1 for item in evidence if item["validation"] == 1),
        "salient_utterances": sum(len(row["salient_utterances"]) for row in rows),
    }
    assert counts == EXPECTED_COUNTS


def test_every_record_carries_its_own_dialogue_text(rows):
    for row in rows:
        assert row["schema_version"] == "2.0"
        assert row["source"] == row["domain"] == SOURCE_BY_PREFIX[row["window_id"][0]]
        assert row["range"] in RANGE_VALUES
        assert row["history"] and row["turns"]
        for field in ("history", "turns"):
            for index, turn in enumerate(row[field]):
                assert turn["turn_index"] == index
                assert isinstance(turn["dialogue"], str)
                assert isinstance(turn["time"], str) and isinstance(turn["speaker"], str)


def test_every_gap_carries_a_target_sentence_and_at_least_one_validated_candidate(rows):
    for row in rows:
        for gap in row["gaps"]:
            assert isinstance(gap["target_sentence"], str) and gap["target_sentence"].strip()
            assert gap["evidence"], (row["window_id"], gap["gap_idx"])
            assert any(item["validation"] == 1 for item in gap["evidence"]), (
                row["window_id"], gap["gap_idx"])
            for item in gap["evidence"]:
                assert item["validation"] in (0, 1)
                assert item["sentence"].strip()
                assert isinstance(item["source_id"], str) and item["source_id"]


def test_gap_indices_are_unique_and_ordered_within_each_window(rows):
    for row in rows:
        indices = [gap["gap_idx"] for gap in row["gaps"]]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)
        salient_indices = [item["salient_idx"] for item in row["salient_utterances"]]
        assert salient_indices == list(range(len(salient_indices)))


def test_the_payload_relates_to_the_v1_index_exactly_as_documented(rows):
    """Gap set identical; the validation==1 evidence subset matches v1 exactly
    once coordinate corrections are set aside (v1 predates them and is frozen)."""
    path = DATASET / "annotations/remember_900_annotations.jsonl"
    v1 = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    v1_gaps = {(row["window_id"], gap["gap_idx"]) for row in v1 for gap in row["gaps"]}
    v2_gaps = {(row["window_id"], gap["gap_idx"]) for row in rows for gap in row["gaps"]}
    assert v1_gaps == v2_gaps

    v1_evidence_count = sum(len(gap["evidence"]) for row in v1 for gap in row["gaps"])
    v2_validated_count = sum(
        1 for row in rows for gap in row["gaps"] for item in gap["evidence"]
        if item["validation"] == 1
    )
    assert v1_evidence_count == v2_validated_count == 4135


def test_published_v1_index_bytes_are_unchanged():
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256(DATASET / "annotations/remember_900_annotations.jsonl") == (
        "a921ae399bd646d469a6824db4d5421358aee387dd1124ff2f1ee2f8c0a8ab3a"
    )
    assert sha256(DATASET / "schema.json") == (
        "2bea6dcd4446fd7f3c980f8d9c8a50d6ffd7757b097ca6efff4db7912999e5c2"
    )
    assert sha256(DATASET / "metadata.json") == (
        "ddf8387a312743e433a6d4493e064e972b6dc6f93f67ce6251dfb39b591f493c"
    )


def test_provenance_file_matches_the_shipped_artifacts(rows):
    provenance = json.loads((DATASET / "PROVENANCE.json").read_text(encoding="utf-8"))
    payload = provenance["artifacts"]["payload"]
    assert payload["sha256"] == PAYLOAD_SHA256
    assert payload["counts"] == EXPECTED_COUNTS
    assert provenance["artifacts"]["v1_index"]["sha256"] == hashlib.sha256(
        (DATASET / "annotations/remember_900_annotations.jsonl").read_bytes()).hexdigest()
    assert provenance["artifacts"]["split"]["sha256"] == hashlib.sha256(
        (DATASET / "splits/main_900.txt").read_bytes()).hexdigest()
    assert (DATASET.parent / provenance["builder"]).is_file()
