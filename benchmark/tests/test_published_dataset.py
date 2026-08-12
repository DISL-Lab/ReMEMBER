import hashlib
import json
import re
from pathlib import Path

import pytest

from remember_benchmark.io import iter_jsonl
from remember_benchmark.release import _schema, validate_annotations


DATASET = Path(__file__).parents[2] / "dataset"


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_dataset_has_paper_counts_and_no_text_fields():
    rows = list(iter_jsonl(DATASET / "annotations/remember_900_annotations.jsonl"))
    stats = validate_annotations(rows, read_ids(DATASET / "splits/main_900.txt"), (900, 1498, 4135))
    assert stats == {"windows": 900, "gaps": 1498, "evidence": 4135}


def test_published_metadata_hashes_distributed_annotation_and_split_files():
    metadata = json.loads((DATASET / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifacts"] == {
        "annotations": {
            "path": "annotations/remember_900_annotations.jsonl",
            "sha256": sha256(DATASET / "annotations/remember_900_annotations.jsonl"),
        },
        "schema": {
            "path": "schema.json",
            "sha256": sha256(DATASET / "schema.json"),
        },
        "split": {
            "path": "splits/main_900.txt",
            "sha256": sha256(DATASET / "splits/main_900.txt"),
        },
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(window_id="L_bad"), "window ID"),
        (lambda row: row.update(source="private"), "source"),
        (lambda row: row.update(range="private"), "range"),
        (lambda row: row.update(schema_version="2.0"), "schema version"),
        (lambda row: row["gaps"][0].update(gap_idx=-1), "gap_idx"),
        (lambda row: row["gaps"].append(dict(row["gaps"][0])), "duplicate gap_idx"),
        (lambda row: row["gaps"][0].update(target_sentence_numbers=[-1]), "target sentence"),
        (lambda row: row["gaps"][0].update(gap_type="N"), "gap_type"),
        (lambda row: row["gaps"][0]["evidence"][0].update(source_id="PRIVATE DIALOGUE SENTENCE"), "source_id"),
        (lambda row: row["gaps"][0]["evidence"][0].update(status="not_resolve"), "evidence"),
        (lambda row: row["gaps"][0]["evidence"][0].update(vote_count=3), "evidence"),
        (lambda row: row["gaps"][0]["evidence"][0]["votes"].update(gemma=2), "votes"),
        (lambda row: row["gaps"][0]["evidence"][0].update(validation=0), "evidence"),
    ],
)
def test_annotation_validator_rejects_scalar_abuse(mutation, message):
    row = json.loads((DATASET / "annotations/remember_900_annotations.jsonl").read_text().splitlines()[0])
    mutation(row)
    with pytest.raises(ValueError, match=message):
        validate_annotations([row], [row["window_id"]])


def test_public_schema_declares_binding_scalar_constraints():
    schema = _schema()
    window = schema["properties"]
    gap = window["gaps"]["items"]["properties"]
    evidence = gap["evidence"]["items"]["properties"]
    assert window["window_id"]["pattern"] == "^[ELR]_U[0-9]{4}$"
    assert window["range"]["enum"] == ["8k~16k", "16k~32k", "32k~64k", "64k~128k", "128k~160k", "other"]
    assert gap["gap_idx"]["minimum"] == 0
    assert gap["target_sentence_numbers"]["items"]["minimum"] == 0
    assert gap["target_sentence_numbers"]["minItems"] == 1
    assert gap["target_sentence_numbers"]["uniqueItems"] is True
    assert gap["gap_type"]["enum"] == ["C", "A", "L"]
    assert evidence["source_id"]["pattern"] == "^T[0-9]+-S[0-9]+$"
    assert evidence["status"]["const"] == "resolve"
    assert evidence["vote_count"] == {"type": "integer", "minimum": 2, "maximum": 3}
    assert all(spec == {"type": "integer", "enum": [0, 1]}
               for spec in evidence["votes"]["properties"].values())
