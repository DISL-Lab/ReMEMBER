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
PAYLOAD_SHA256 = "d5404512111703e03ed5df8a61c69a7094b4a78393ec7b21c01b9841b86a25d3"

EXPECTED_COUNTS = {"windows": 900, "gaps": 1498, "evidence": 4134, "salient_utterances": 7245}
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


def _history_index(history):
    """Reproduce the annotation-time history segmentation from the record alone."""
    import re
    from nltk.tokenize import sent_tokenize

    emoji = re.compile("[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
                       "\U0001F1E0-\U0001F1FF" "\U00002600-\U000027BF" "\U0001F900-\U0001F9FF"
                       "\U0001FA00-\U0001FA6F" "\U0001FA70-\U0001FAFF" "\U00002702-\U000027B0"
                       "\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
    index = {}
    for turn_index, turn in enumerate(history):
        stripped = emoji.sub("", turn["dialogue"]).strip()
        sentences = [s.strip() for s in sent_tokenize(stripped) if s.strip()] or [stripped]
        for sentence_index, sentence in enumerate(sentences, 1):
            index[f"T{turn_index}-S{sentence_index}"] = sentence
    return index


def test_every_evidence_sentence_re_resolves_from_its_own_record(rows):
    """The payload is self-contained: no external file is needed to check it."""
    checked = 0
    for row in rows:
        index = _history_index(row["history"])
        for gap in row["gaps"]:
            for item in gap["evidence"]:
                assert index[item["source_id"]] == item["sentence"], (
                    row["window_id"], gap["gap_idx"], item["source_id"])
                checked += 1
    assert checked == EXPECTED_COUNTS["evidence"]


def test_every_target_and_salient_sentence_re_resolves_from_its_own_record(rows):
    from nltk.tokenize import sent_tokenize

    for row in rows:
        flat = []
        for turn in row["turns"]:
            text = turn["dialogue"]
            flat.extend([s.strip() for s in sent_tokenize(text.strip()) if s.strip()] or [text])
        for gap in row["gaps"]:
            for number, sentence in zip(gap["target_sentence_numbers"], gap["target_sentence"]):
                assert flat[number] == sentence, (row["window_id"], gap["gap_idx"], number)
        for salient in row["salient_utterances"]:
            [number], [sentence] = salient["sentence_numbers"], salient["sentences"]
            assert flat[number] == sentence, (row["window_id"], salient["salient_idx"])


def test_the_payload_relates_to_the_v1_index_exactly_as_documented(rows):
    """Gap set identical; evidence differs only by the one documented drop."""
    path = DATASET / "annotations/remember_900_annotations.jsonl"
    v1 = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    v1_gaps = {(row["window_id"], gap["gap_idx"]) for row in v1 for gap in row["gaps"]}
    v2_gaps = {(row["window_id"], gap["gap_idx"]) for row in rows for gap in row["gaps"]}
    assert v1_gaps == v2_gaps

    v1_links = sum(len(gap["evidence"]) for row in v1 for gap in row["gaps"])
    v2_links = sum(len(gap["evidence"]) for row in rows for gap in row["gaps"])
    assert (v1_links, v2_links) == (4135, 4134)

    # Every v2 link is its v1 counterpart at the same or the next sentence index.
    v1_by_gap = {(row["window_id"], gap["gap_idx"]): [item["source_id"] for item in gap["evidence"]]
                 for row in v1 for gap in row["gaps"]}
    shifted = dropped = 0
    for row in rows:
        for gap in row["gaps"]:
            original = v1_by_gap[(row["window_id"], gap["gap_idx"])]
            released = [item["source_id"] for item in gap["evidence"]]
            dropped += len(original) - len(released)
            offset = 0
            for source_id in released:
                while original[offset] != source_id:
                    turn, _, number = original[offset].rpartition("-S")
                    assert f"{turn}-S{int(number) + 1}" == source_id, (source_id, original[offset])
                    shifted += 1
                    break
                offset += 1
    assert (shifted, dropped) == (132, 1)


def test_provenance_file_matches_the_shipped_artifacts(rows):
    provenance = json.loads((DATASET / "PROVENANCE.json").read_text(encoding="utf-8"))
    payload = provenance["artifacts"]["payload"]
    assert payload["sha256"] == PAYLOAD_SHA256
    assert payload["counts"] == EXPECTED_COUNTS
    assert provenance["artifacts"]["v1_index"]["sha256"] == hashlib.sha256(
        (DATASET / "annotations/remember_900_annotations.jsonl").read_bytes()).hexdigest()
    assert provenance["artifacts"]["split"]["sha256"] == hashlib.sha256(
        (DATASET / "splits/main_900.txt").read_bytes()).hexdigest()
    assert (ROOT_BUILDER := DATASET.parent / provenance["builder"]).is_file(), ROOT_BUILDER
    assert provenance["corrections"]["shifted_evidence_coordinates"] == 132
    assert provenance["corrections"]["dropped_evidence_links"] == ["R_U0774 gap 0 T728-S3"]
