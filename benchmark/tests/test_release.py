import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from remember_benchmark.io import atomic_write_jsonl, iter_jsonl
from remember_benchmark.release import export_annotations, sanitize_annotations, validate_annotations


FIXTURES = Path(__file__).parent / "fixtures"
SRC = Path(__file__).parents[1] / "src"


def _keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _keys(nested)


def test_sanitize_annotations_keeps_only_owned_coordinates():
    rows = sanitize_annotations(iter_jsonl(FIXTURES / "rcv.jsonl"), ["E_w1"])
    assert rows == [{
        "schema_version": "1.0",
        "window_id": "E_w1",
        "source": "evermembench",
        "range": "8k~16k",
        "gaps": [{
            "gap_idx": 0,
            "target_sentence_numbers": [2],
            "gap_type": "A",
            "evidence": [{
                "source_id": "T1-S0",
                "status": "resolve",
                "vote_count": 3,
                "votes": {"gemma": 1, "GLM": 1, "Qwen": 1},
                "validation": 1,
            }],
        }],
    }]
    serialized = json.dumps(rows)
    keys = set(_keys(rows))
    for forbidden in ("target_sentence", "sentence", "gap_reason", "reasoning", "Earlier context"):
        assert forbidden not in keys
        assert forbidden not in serialized.replace("target_sentence_numbers", "")


def test_sanitize_annotations_rejects_unknown_source_prefix():
    rows = [{
        "window_id": "X_w1",
        "range": "8k~16k",
        "gap_idx": 0,
        "sentence_numbers": [1],
        "target_sentence": "needs context",
        "gap_type": "A",
        "validated": [],
    }]
    with pytest.raises(ValueError, match="unknown source prefix"):
        sanitize_annotations(rows, ["X_w1"])


def test_sanitize_annotations_rejects_unknown_prefix_in_split_before_reading_rows():
    with pytest.raises(ValueError, match="unknown source prefix"):
        sanitize_annotations([], ["X_w1"])


def test_validate_annotations_rejects_duplicate_split_ids():
    with pytest.raises(ValueError, match="duplicate split"):
        validate_annotations([], ["E_w1", "E_w1"])


def test_validate_annotations_rejects_forbidden_nested_key():
    rows = [{
        "schema_version": "1.0",
        "window_id": "E_w1",
        "source": "evermembench",
        "range": "8k~16k",
        "gaps": [{"gap_idx": 0, "target_sentence_numbers": [1], "gap_type": "A", "evidence": [], "reasoning": "leak"}],
    }]
    with pytest.raises(ValueError, match="forbidden key"):
        validate_annotations(rows, ["E_w1"])


def test_validate_annotations_rejects_count_mismatch():
    rows = sanitize_annotations(iter_jsonl(FIXTURES / "rcv.jsonl"), ["E_w1"])
    with pytest.raises(ValueError, match="count mismatch"):
        validate_annotations(rows, ["E_w1"], (2, 1, 1))


def test_validate_annotations_rejects_malformed_required_field_types():
    rows = [{
        "schema_version": "1.0",
        "window_id": "E_w1",
        "source": "evermembench",
        "range": "8k~16k",
        "gaps": [{
            "gap_idx": 0,
            "target_sentence_numbers": "",
            "gap_type": "A",
            "evidence": [{
                "source_id": 7,
                "status": False,
                "vote_count": "3",
                "votes": ["GLM", "Qwen", "gemma"],
                "validation": 1,
            }],
        }],
    }]
    with pytest.raises(ValueError, match="invalid"):
        validate_annotations(rows, ["E_w1"])


def test_cli_rejects_expected_count_mismatch_before_writing_outputs(tmp_path):
    output = tmp_path / "annotations" / "out.jsonl"
    metadata = tmp_path / "metadata" / "metadata.json"
    schema = tmp_path / "schema" / "schema.json"
    environment = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [
            sys.executable, "-m", "remember_benchmark.release",
            "--rcv", str(FIXTURES / "rcv.jsonl"),
            "--split", str(FIXTURES / "main_ids.txt"),
            "--output", str(output),
            "--metadata", str(metadata),
            "--schema", str(schema),
            "--expected-windows", "2",
            "--expected-gaps", "1",
            "--expected-evidence", "1",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "count mismatch" in result.stderr
    assert not output.exists()
    assert not metadata.exists()
    assert not schema.exists()


def test_write_helpers_fsync_each_parent_directory(tmp_path, monkeypatch):
    real_fsync = os.fsync
    synced_directories = set()

    def observe_fsync(file_descriptor):
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            synced_directories.add(Path(os.readlink(f"/proc/self/fd/{file_descriptor}")))
        real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", observe_fsync)
    annotation_path = tmp_path / "annotations" / "out.jsonl"
    metadata_path = tmp_path / "metadata" / "metadata.json"
    schema_path = tmp_path / "schema" / "schema.json"
    atomic_write_jsonl(annotation_path, [{"row": 1}])
    export_annotations(
        FIXTURES / "rcv.jsonl", FIXTURES / "main_ids.txt", annotation_path, metadata_path, schema_path
    )
    assert {annotation_path.parent, metadata_path.parent, schema_path.parent} <= synced_directories
