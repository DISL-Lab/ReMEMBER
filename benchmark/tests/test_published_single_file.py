"""Regression pins for the single self-contained benchmark file.

The published payload is one gzipped JSONL file holding all 900 windows with
their dialogue text, gold gaps, resolved evidence sentences, and salient
utterances. Every released sentence was resolved from its annotation coordinate,
so these tests assert the file is internally self-contained: no consumer needs
the research tree or any sentence splitter to use it.
"""

import gzip
import hashlib
import json
from pathlib import Path

import pytest


DATASET = Path(__file__).parents[2] / "dataset"
PAYLOAD = DATASET / "remember_900.jsonl.gz"
PAYLOAD_SHA256 = "9353ec8f71d8d0a92ab4ae7ca95ebfeb152e8cb9be73f02ad982778f9f61278f"

EXPECTED_COUNTS = {"windows": 900, "gaps": 1498, "evidence": 4132, "salient_utterances": 7245}
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
    counts = {
        "windows": len(rows),
        "gaps": sum(len(row["gaps"]) for row in rows),
        "evidence": sum(len(gap["evidence"]) for row in rows for gap in row["gaps"]),
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


def test_every_gold_coordinate_resolves_to_text_inside_the_same_record(rows):
    for row in rows:
        for gap in row["gaps"]:
            assert gap["target_sentence"], row["window_id"]
            assert len(gap["target_sentence"]) == len(gap["target_sentence_numbers"])
            assert all(sentence.strip() for sentence in gap["target_sentence"])
            assert gap["evidence"], (row["window_id"], gap["gap_idx"])
            seen = set()
            for item in gap["evidence"]:
                assert item["source_id"] not in seen
                seen.add(item["source_id"])
                turn_part, sentence_part = item["source_id"].split("-S")
                turn_index = int(turn_part[1:])
                # The coordinate must address a turn this record actually ships.
                assert 0 <= turn_index < len(row["history"])
                assert int(sentence_part) >= 1
                assert item["sentence"].strip()
                assert item["status"] == "resolve" and item["validation"] == 1
        for salient in row["salient_utterances"]:
            assert len(salient["sentences"]) == len(salient["sentence_numbers"]) == 1
            assert salient["sentences"][0].strip()


def test_gap_indices_are_unique_and_ordered_within_each_window(rows):
    for row in rows:
        indices = [gap["gap_idx"] for gap in row["gaps"]]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)
        salient_indices = [item["salient_idx"] for item in row["salient_utterances"]]
        assert salient_indices == list(range(len(salient_indices)))


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
