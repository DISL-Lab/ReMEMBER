import json
import os
import stat
import sys
import types
from dataclasses import replace

import pytest

import remember.cli as cli
from remember.cli import ResumeError, main, run_file
from remember.config import RememberConfig
from remember.inference import (
    Cl100kTokenCounter,
    DeterministicFakeGenerator,
    DeterministicHashEmbedder,
    VLLMGenerator,
)


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def _write_windows(path, ids=("w1", "w2")):
    rows = [
        {
            "window_id": window_id,
            "history": [
                {"time": "09:00", "speaker": "A", "dialogue": f"Earlier cedar detail {window_id}."}
            ],
            "turns": [
                {"time": "10:00", "speaker": "B", "dialogue": "Use it now."}
            ],
        }
        for window_id in ids
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_resume_never_duplicates_window_ids_and_rewrites_atomically(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "out.jsonl"
    _write_windows(source)
    config = RememberConfig()

    kwargs = dict(
        input_path=source,
        output_path=output,
        config=config,
        generator=DeterministicFakeGenerator(),
        embedder=DeterministicHashEmbedder(),
        counter=WordCounter(),
    )
    run_file(**kwargs)
    run_file(**kwargs, resume=True)

    rows = _read_rows(output)
    assert [row["window_id"] for row in rows] == ["w1", "w2"]
    assert len({row["window_id"] for row in rows}) == 2
    assert not list(tmp_path.glob(".out.jsonl.*.tmp"))
    assert (tmp_path / "out.jsonl.manifest.json").exists()


def test_resume_rejects_changed_input_or_configuration_hash(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "out.jsonl"
    _write_windows(source)
    config = RememberConfig()
    common = dict(
        input_path=source,
        output_path=output,
        generator=DeterministicFakeGenerator(),
        embedder=DeterministicHashEmbedder(),
        counter=WordCounter(),
    )
    run_file(config=config, **common)

    _write_windows(source, ids=("w1", "w2", "w3"))
    with pytest.raises(ResumeError, match="input hash"):
        run_file(config=config, resume=True, **common)

    _write_windows(source)
    with pytest.raises(ResumeError, match="configuration hash"):
        run_file(config=replace(config, top_k=7), resume=True, **common)


def test_resume_rejects_output_that_no_longer_matches_manifest(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "out.jsonl"
    _write_windows(source, ids=("w1",))
    common = dict(
        input_path=source,
        output_path=output,
        config=RememberConfig(),
        generator=DeterministicFakeGenerator(),
        embedder=DeterministicHashEmbedder(),
        counter=WordCounter(),
    )
    run_file(**common)
    output.write_text(output.read_text() + output.read_text(), encoding="utf-8")

    with pytest.raises(ResumeError, match="output hash"):
        run_file(**common, resume=True)


def test_existing_output_requires_resume_or_overwrite(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "out.jsonl"
    _write_windows(source, ids=("w1",))
    common = dict(
        input_path=source,
        output_path=output,
        config=RememberConfig(),
        generator=DeterministicFakeGenerator(),
        embedder=DeterministicHashEmbedder(),
        counter=WordCounter(),
    )
    run_file(**common)

    with pytest.raises(FileExistsError, match="--resume or --overwrite"):
        run_file(**common)


def test_cli_fake_provider_runs_without_network_and_never_echoes_input_dialogue(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "out.jsonl"
    config = tmp_path / "paper.yaml"
    _write_windows(source, ids=("w1",))
    config.write_text(json.dumps({
        "gap_model": "fixture/gap",
        "summary_model": "fixture/summary",
        "dense_model": "fixture/dense",
        "max_gaps": 8,
        "chunk_tokens": 128,
        "overlap_tokens": 32,
        "memory_budget_tokens": 1024,
        "top_k": 8,
        "rrf_k": 60,
    }))

    original_count = Cl100kTokenCounter.count
    Cl100kTokenCounter.count = WordCounter.count
    try:
        exit_code = main([
        "run",
        "--config", str(config),
        "--input", str(source),
        "--output", str(output),
        "--provider", "fake",
        ])
    finally:
        Cl100kTokenCounter.count = original_count

    assert exit_code == 0
    [row] = _read_rows(output)
    assert row["summary"] == "Deterministic fake summary."
    assert "history" not in row and "turns" not in row
    assert "Earlier cedar detail w1." not in json.dumps(row["predicted_gaps"])
    assert set(row) == {
        "schema_version", "window_id", "predicted_gaps", "retrieval",
        "memory", "summary", "token_counts", "timing_seconds", "effective_config",
    }
    assert row["schema_version"] == "1.1"


def test_resume_rejects_output_schema_version_change(monkeypatch, tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "out.jsonl"
    _write_windows(source, ids=("w1",))
    common = dict(
        input_path=source,
        output_path=output,
        config=RememberConfig(),
        generator=DeterministicFakeGenerator(),
        embedder=DeterministicHashEmbedder(),
        counter=WordCounter(),
    )
    run_file(**common)

    monkeypatch.setattr(cli, "SCHEMA_VERSION", "9.9")
    with pytest.raises(ResumeError, match="configuration hash"):
        run_file(**common, resume=True)


def test_paper_config_uses_canonical_v15_prompt_and_dense_prefix():
    package = __import__("remember.pipeline", fromlist=["PROMPT_ROOT"])
    prompt = (package.PROMPT_ROOT / "gap_detection.txt").read_text(encoding="utf-8")
    config = __import__("remember.cli", fromlist=["_load_config"])._load_config(
        package.PACKAGE_ROOT / "configs" / "paper.yaml"
    )

    assert prompt.startswith("You are building retrieval queries for long-dialogue memory.")
    assert "Select an utterance when:" in prompt
    assert "Output up to {max_gaps} gaps." in prompt
    assert config.dense_query_prefix == (
        "Instruct: Given a current-window utterance with missing dialogue context, "
        "retrieve earlier dialogue that directly supports a faithful memory item.\nQuery: "
    )


def test_fake_provider_uses_exact_cl100k_counter(monkeypatch, tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "out.jsonl"
    config = tmp_path / "paper.yaml"
    _write_windows(source, ids=("w1",))
    config.write_text("{}\n")
    sentinel = object()
    captured = {}
    monkeypatch.setattr(cli, "Cl100kTokenCounter", lambda: sentinel)
    monkeypatch.setattr(cli, "run_file", lambda *args, **kwargs: captured.update(kwargs) or [])

    assert main([
        "run", "--config", str(config), "--input", str(source),
        "--output", str(output), "--provider", "fake",
    ]) == 0

    assert captured["counter"] is sentinel


def test_resume_rejects_provider_change_before_api_key_access(monkeypatch, tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "out.jsonl"
    config = tmp_path / "paper.yaml"
    _write_windows(source, ids=("w1",))
    config.write_text("{}\n")
    monkeypatch.setattr(Cl100kTokenCounter, "count", WordCounter.count)

    main([
        "run", "--config", str(config), "--input", str(source),
        "--output", str(output), "--provider", "fake",
    ])
    monkeypatch.delenv("MISSING_REVIEW_KEY", raising=False)

    with pytest.raises(ResumeError, match="configuration hash"):
        main([
            "run", "--config", str(config), "--input", str(source),
            "--output", str(output), "--provider", "openai", "--resume",
            "--base-url", "https://fixture.invalid/v1",
            "--api-key-env", "MISSING_REVIEW_KEY", "--dense-device", "cpu",
        ])


def test_cli_atomic_manifest_write_fsyncs_parent_directory(monkeypatch, tmp_path):
    fsynced_modes = []
    real_fsync = os.fsync

    def record_fsync(fd):
        fsynced_modes.append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(cli.os, "fsync", record_fsync)
    cli._atomic_write_json(tmp_path / "manifest.json", {"ok": True})

    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)


def test_vllm_model_switch_tears_down_old_engine_before_loading_new(monkeypatch):
    events = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    class FakeEngine:
        def __init__(self, model):
            self.model = model

        def shutdown(self):
            events.append(f"shutdown:{self.model}")

    class FakeLLM:
        def __init__(self, *, model, **kwargs):
            del kwargs
            events.append(f"load:{model}")
            self.model = model
            self.llm_engine = FakeEngine(model)

        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, params, use_tqdm=False):
            del prompts, params, use_tqdm
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="ok")])]

    fake_vllm = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    generator = VLLMGenerator()

    generator.generate("first", model="model-a")
    generator.generate("second", model="model-b")

    assert events == ["load:model-a", "shutdown:model-a", "load:model-b"]
