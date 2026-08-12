import json
from pathlib import Path

import pytest

from remember_benchmark.manifest import (
    ResumeMismatch,
    publish_stage,
    verify_resume,
    write_stage_manifest,
)


def _files(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"window_id":"E_U0000"}\n', encoding="utf-8")
    output_path = tmp_path / "salient.jsonl"
    output_path.write_text('{"window_id":"E_U0000"}\n', encoding="utf-8")
    return input_path, output_path


def test_resume_rejects_changed_prompt_hash(tmp_path):
    input_path, output_path = _files(tmp_path)
    old = write_stage_manifest(output_path, stage="salient", prompt="v1", inputs=[input_path], config={"seed": 42})
    with pytest.raises(ResumeMismatch, match="prompt hash"):
        verify_resume(old.path, prompt="v2", inputs=[input_path], config={"seed": 42})


def test_resume_rejects_changed_input_and_configuration_hashes(tmp_path):
    input_path, output_path = _files(tmp_path)
    old = write_stage_manifest(output_path, stage="salient", prompt="v1", inputs=[input_path], config={"seed": 42})
    input_path.write_text('{"window_id":"E_U0001"}\n', encoding="utf-8")
    with pytest.raises(ResumeMismatch, match="input hash"):
        verify_resume(old.path, prompt="v1", inputs=[input_path], config={"seed": 42})

    input_path.write_text('{"window_id":"E_U0000"}\n', encoding="utf-8")
    with pytest.raises(ResumeMismatch, match="configuration hash"):
        verify_resume(old.path, prompt="v1", inputs=[input_path], config={"seed": 43})


def test_resume_rejects_modified_or_missing_output_and_incomplete_temp(tmp_path):
    input_path, output_path = _files(tmp_path)
    old = write_stage_manifest(output_path, stage="salient", prompt="v1", inputs=[input_path], config={})
    output_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ResumeMismatch, match="output hash"):
        verify_resume(old.path, prompt="v1", inputs=[input_path], config={})

    output_path.write_text('{"window_id":"E_U0000"}\n', encoding="utf-8")
    (tmp_path / ".salient.jsonl.interrupted.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(ResumeMismatch, match="temporary output"):
        verify_resume(old.path, prompt="v1", inputs=[input_path], config={})


def test_publish_stage_is_canonical_atomic_and_exact_resume_succeeds(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"a":1}\n', encoding="utf-8")
    output_path = tmp_path / "stage.jsonl"
    rows = [{"z": 2, "a": 1}, {"window_id": "E_U0000"}]

    manifest = publish_stage(
        output_path, rows, stage="salient", prompt="fixture prompt", inputs=[input_path],
        config={"nested": {"z": 2, "a": 1}, "seed": 42},
    )

    assert output_path.read_text(encoding="utf-8") == '{"a":1,"z":2}\n{"window_id":"E_U0000"}\n'
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["stage"] == "salient"
    assert len(payload["prompt_sha256"]) == len(payload["configuration_sha256"]) == 64
    assert not list(tmp_path.glob("*.tmp"))
    assert verify_resume(
        manifest.path, prompt="fixture prompt", inputs=[input_path],
        config={"seed": 42, "nested": {"a": 1, "z": 2}},
    ) == manifest


def test_manifest_rejects_duplicate_input_paths(tmp_path):
    input_path, output_path = _files(tmp_path)
    with pytest.raises(ValueError, match="duplicate input"):
        write_stage_manifest(output_path, stage="x", prompt="p", inputs=[input_path, input_path], config={})


def test_resume_strictly_checks_expected_stage_and_output_identity(tmp_path):
    input_path, output_path = _files(tmp_path)
    old = write_stage_manifest(output_path, stage="salient", prompt="v1", inputs=[input_path], config={})
    with pytest.raises(ResumeMismatch, match="stage identity"):
        verify_resume(old.path, prompt="v1", inputs=[input_path], config={}, expected_stage="validation", output_path=output_path)
    with pytest.raises(ResumeMismatch, match="output identity"):
        verify_resume(old.path, prompt="v1", inputs=[input_path], config={}, expected_stage="salient", output_path=tmp_path / "other.jsonl")
    assert verify_resume(
        old.path, prompt="v1", inputs=[input_path], config={}, expected_stage="salient", output_path=output_path
    ) == old


def test_resume_rejects_leftover_manifest_temp(tmp_path):
    input_path, output_path = _files(tmp_path)
    old = write_stage_manifest(output_path, stage="salient", prompt="v1", inputs=[input_path], config={})
    (tmp_path / f".{old.path.name}.interrupted.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(ResumeMismatch, match="temporary manifest"):
        verify_resume(old.path, prompt="v1", inputs=[input_path], config={})


def test_resume_rejects_manifest_output_name_traversal(tmp_path):
    input_path, output_path = _files(tmp_path)
    old = write_stage_manifest(output_path, stage="salient", prompt="v1", inputs=[input_path], config={})
    payload = json.loads(old.path.read_text(encoding="utf-8"))
    payload["output_name"] = "../victim.jsonl"
    old.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResumeMismatch, match="output identity|output_name"):
        verify_resume(old.path, prompt="v1", inputs=[input_path], config={})
