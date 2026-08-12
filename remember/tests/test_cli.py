import json
from dataclasses import replace

import pytest

from remember.cli import ResumeError, main, run_file
from remember.config import RememberConfig
from remember.inference import DeterministicFakeGenerator, DeterministicHashEmbedder


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

    exit_code = main([
        "run",
        "--config", str(config),
        "--input", str(source),
        "--output", str(output),
        "--provider", "fake",
    ])

    assert exit_code == 0
    [row] = _read_rows(output)
    assert row["summary"] == "Deterministic fake summary."
    assert "history" not in row and "turns" not in row
    assert "Earlier cedar detail w1." not in json.dumps(row["predicted_gaps"])
    assert set(row) == {
        "schema_version", "window_id", "method", "predicted_gaps", "retrieval",
        "memory", "summary", "token_counts", "timing_seconds", "effective_config",
    }


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
        "retrieve earlier dialogue that resolves the missing referent, artifact, "
        "decision, rationale, or prior state.\nQuery: "
    )
