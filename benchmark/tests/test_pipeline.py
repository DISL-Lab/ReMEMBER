import json
import copy
from pathlib import Path

import pytest

from remember_benchmark.pipeline import STAGES, build, flatten_ensemble_to_rcv


class WordCounter:
    def count(self, text):
        return 1


class FixtureProvider:
    def complete(self, prompt):
        if "STAGE: salient" in prompt:
            numbered = prompt.split("WINDOW:\n", 1)[1].splitlines()[0]
            sentence = numbered.split(": ", 1)[-1]
            return json.dumps({"schema_version": "1.0", "candidates": [{
                "sentence_numbers": [0], "sentences": [sentence],
                "reason": "depends on prior context", "gap_type": "A",
            }]})
        if "STAGE: chunk_filter" in prompt:
            chunks = json.loads(prompt.split("CHUNKS:", 1)[1].split("\nWINDOW:", 1)[0])
            return json.dumps({"schema_version": "1.0", "labels": [1] * len(chunks)})
        if "STAGE: evidence" in prompt:
            history = json.loads(prompt.split("HISTORY:", 1)[1].split("\nGAP:", 1)[0])
            return json.dumps({"schema_version": "1.0", "evidence": [
                {"source_id": source_id, "sentence": sentence, "reasoning": "explicit",
                 "status": "resolve"} for source_id, sentence in history.items()
            ]})
        if "STAGE: validation" in prompt:
            candidates = json.loads(prompt.split("CANDIDATES:", 1)[1].split("\nGAP:", 1)[0])
            return json.dumps({"schema_version": "1.0", "valid_source_ids": [
                item["source_id"] for item in candidates
            ]})
        raise AssertionError(prompt)


def _fixture(tmp_path):
    source = tmp_path / "normalized.jsonl"
    turns = [
        {"time": f"2024-01-01 00:00:{i:02d}", "speaker": "A", "dialogue": text}
        for i, text in enumerate(["Earlier fact.", "More context.", "Use the prior fact."])
    ]
    source.write_text(json.dumps({"schema_version": "1.0", "dialogue_id": "E_topic-1_group-1",
                                  "source": "evermembench", "turns": turns}) + "\n")
    split = tmp_path / "split.txt"
    split.write_text("E_U0000\n")
    prompt_dir = Path(__file__).parents[1] / "prompts"
    config = {
        "sources": {"normalized": str(source)},
        "models": {"gemma": "fixture/gemma", "GLM": "fixture/glm", "Qwen": "fixture/qwen"},
        "thresholds": {"salient": 3, "chunk_filter": 3, "validation": 2},
        "windowing": {"start_tokens": 2, "size_tokens": 1,
                      "strides": {"evermembench": 99}},
        "chunking": {"tokens": 32, "overlap_tokens": 0, "chunks_per_group": 16},
        "prompts": {name: str(prompt_dir / filename) for name, filename in {
            "salient": "salient.txt", "chunk_filter": "relevant_chunk_filtering.txt",
            "evidence": "relevant_chunk.txt", "validation": "relevant_chunk_validation.txt"}.items()},
        "split": str(split), "expected_counts": {"windows": 1, "gaps": 1, "evidence": 2},
        "temperature": 0,
    }
    return config, source, split


def test_full_build_runs_eight_stages_and_exports_text_free_payload(tmp_path):
    config, _, _ = _fixture(tmp_path)
    providers = {name: FixtureProvider() for name in config["models"]}
    result = build(config, tmp_path / "work", tmp_path / "out", providers, counter=WordCounter())
    assert result.stage_names == list(STAGES)
    annotation = tmp_path / "out/annotations/remember_900_annotations.jsonl"
    [row] = [json.loads(line) for line in annotation.read_text().splitlines()]
    assert row["gaps"][0]["evidence"][0]["votes"] == {"GLM": 1, "Qwen": 1, "gemma": 1}
    def keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield key
                yield from keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from keys(nested)
    forbidden = {"history", "turns", "target_sentence", "gap_reason", "reasoning", "raw_output"}
    assert not (forbidden & set(keys(row)))
    assert not (tmp_path / "out/private_rcv.jsonl").exists()
    assert (tmp_path / "work/release/private_rcv.jsonl").exists()


def test_flat_bridge_densifies_two_of_three_vote_map_and_uses_real_target():
    nested = [{"window_id": "E_U0000", "source": "evermembench", "range": "other", "gaps": [{
        "gap_idx": 0, "target_sentence_numbers": [0], "gap_type": "A", "evidence": [{
            "source_id": "T0-S1", "status": "resolve", "votes": {"gemma": 1, "GLM": 1},
            "vote_count": 2, "validation": 1}]}]}]
    salient = [{"window_id": "E_U0000", "gaps": [{"gap_idx": 0,
                "target_sentence": ["Real target."], "gap_reason": "private"}]}]
    [row] = flatten_ensemble_to_rcv(nested, salient)
    assert row["target_sentence"] == ["Real target."]
    assert row["sentence_numbers"] == [0]
    assert row["validated"][0]["votes"] == {"gemma": 1, "GLM": 1, "Qwen": 0}
    from remember_benchmark.release import sanitize_annotations
    [public] = sanitize_annotations([row], ["E_U0000"])
    assert public["gaps"][0]["evidence"][0]["votes"] == {"gemma": 1, "GLM": 1, "Qwen": 0}
    assert public["gaps"][0]["evidence"][0]["vote_count"] == 2


def test_resume_overwrite_and_partial_stage_contract(tmp_path):
    config, source, _ = _fixture(tmp_path)
    providers = {name: FixtureProvider() for name in config["models"]}
    build(config, tmp_path / "work", tmp_path / "out", providers, counter=WordCounter())
    resumed = build(config, tmp_path / "work", tmp_path / "out", providers,
                    counter=WordCounter(), resume=True)
    assert resumed.stage_names == list(STAGES)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build(config, tmp_path / "work", tmp_path / "out", providers,
              counter=WordCounter(), resume=True, overwrite=True)
    with pytest.raises(FileExistsError):
        build(config, tmp_path / "work", tmp_path / "out", providers, counter=WordCounter())
    source.write_text(source.read_text().replace("Earlier fact", "Changed fact"))
    with pytest.raises(ValueError, match="hash mismatch"):
        build(config, tmp_path / "work", tmp_path / "out", providers,
              counter=WordCounter(), resume=True)
    with pytest.raises(ValueError, match="stage"):
        build(config, tmp_path / "w2", tmp_path / "o2", providers,
              counter=WordCounter(), from_stage="evidence", resume=True)
    with pytest.raises(ValueError, match="order"):
        build(config, tmp_path / "w3", tmp_path / "o3", providers,
              counter=WordCounter(), from_stage="evidence", to_stage="windows")


def test_early_to_stage_never_emits_release(tmp_path):
    config, _, _ = _fixture(tmp_path)
    result = build(config, tmp_path / "work", tmp_path / "out",
                   {name: FixtureProvider() for name in config["models"]},
                   counter=WordCounter(), to_stage="windows")
    assert result.stage_names == ["sources", "windows"]
    assert not (tmp_path / "out").exists()


def _complete_build(tmp_path):
    config, source, split = _fixture(tmp_path)
    providers = {name: FixtureProvider() for name in config["models"]}
    work, out = tmp_path / "work", tmp_path / "out"
    build(config, work, out, providers, counter=WordCounter())
    return config, source, split, providers, work, out


@pytest.mark.parametrize("change,match", [
    ("prompt", "prompt hash mismatch"),
    ("model", "configuration hash mismatch"),
    ("threshold", "configuration hash mismatch"),
    ("source", "input hash mismatch"),
    ("split", "hash mismatch"),
    ("stage_output", "output hash mismatch"),
    ("missing_manifest", "manifest"),
    ("corrupt_manifest", "manifest"),
    ("orphan_temp", "temporary"),
])
def test_resume_rejects_every_stale_stage_signal(tmp_path, change, match):
    config, source, split, providers, work, out = _complete_build(tmp_path)
    updated = copy.deepcopy(config)
    if change == "prompt":
        prompt = tmp_path / "changed-prompt.txt"
        prompt.write_text(Path(config["prompts"]["salient"]).read_text() + "\nChanged instruction.\n")
        updated["prompts"]["salient"] = str(prompt)
    elif change == "model":
        updated["models"]["gemma"] = "fixture/different-model"
    elif change == "threshold":
        updated["thresholds"]["salient"] = 2
    elif change == "source":
        source.write_text(source.read_text().replace("Earlier fact", "Changed fact"))
    elif change == "split":
        split.write_text("E_U0000\nE_U9999\n")
    elif change == "stage_output":
        path = work / "windows.jsonl"
        path.write_text(path.read_text() + path.read_text())
    elif change == "missing_manifest":
        (work / "windows.jsonl.manifest.json").unlink()
    elif change == "corrupt_manifest":
        (work / "windows.jsonl.manifest.json").write_text("not-json\n")
    elif change == "orphan_temp":
        (work / ".sources.jsonl.interrupted.tmp").write_text("partial")
    with pytest.raises(ValueError, match=match):
        build(updated, work, out, providers, counter=WordCounter(), resume=True)


@pytest.mark.parametrize("relative", [
    "annotations/remember_900_annotations.jsonl", "schema.json", "metadata.json",
])
@pytest.mark.parametrize("damage", ["missing", "corrupt", "valid_tamper"])
def test_resume_rejects_missing_or_corrupt_release_artifacts(tmp_path, relative, damage):
    config, _, _, providers, work, out = _complete_build(tmp_path)
    path = out / relative
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("not-json\n")
    else:
        path.write_text("{}\n")
    with pytest.raises(ValueError, match="release|output hash|stage output|artifact|metadata|schema"):
        build(config, work, out, providers, counter=WordCounter(), resume=True)


def test_valid_partial_overwrite_verifies_prerequisites_and_recomputes_downstream(tmp_path):
    config, _, _, _, work, out = _complete_build(tmp_path)
    source_hash = (work / "sources.jsonl").read_bytes()
    evidence_path = work / "evidence/gemma.jsonl"
    evidence_path.write_text("tampered\n")
    result = build(config, work, out, {name: FixtureProvider() for name in config["models"]},
                   counter=WordCounter(), overwrite=True, from_stage="evidence")
    assert result.stage_names == ["evidence", "validation", "ensemble", "release"]
    assert (work / "sources.jsonl").read_bytes() == source_hash
    assert "tampered" not in evidence_path.read_text()
    assert (out / "metadata.json").is_file()


def test_partial_resume_with_verified_prerequisites_and_early_stop(tmp_path):
    config, _, _, _, work, out = _complete_build(tmp_path)
    result = build(config, work, out, {name: FixtureProvider() for name in config["models"]},
                   counter=WordCounter(), resume=True, from_stage="validation", to_stage="ensemble")
    assert result.stage_names == ["validation", "ensemble"]


def test_resume_rejects_changed_provider_identity(tmp_path):
    config, _, _, providers, work, out = _complete_build(tmp_path)

    class DifferentProvider(FixtureProvider):
        pass

    providers["gemma"] = DifferentProvider()
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        build(config, work, out, providers, counter=WordCounter(), resume=True)


def test_partial_no_flag_cannot_overwrite_selected_existing_stage(tmp_path):
    config, _, _, _, work, out = _complete_build(tmp_path)
    with pytest.raises(FileExistsError, match="--resume or --overwrite"):
        build(config, work, out, {name: FixtureProvider() for name in config["models"]},
              counter=WordCounter(), from_stage="evidence", to_stage="evidence")


def test_fresh_preflight_detects_downstream_sentinel_before_any_upstream_write(tmp_path):
    config, _, _ = _fixture(tmp_path)
    work, out = tmp_path / "work", tmp_path / "out"
    sentinel = work / "salient/gemma.jsonl"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("do-not-touch\n")
    with pytest.raises(FileExistsError, match="salient/gemma"):
        build(config, work, out, {name: FixtureProvider() for name in config["models"]},
              counter=WordCounter())
    assert sentinel.read_text() == "do-not-touch\n"
    assert not (work / "sources.jsonl").exists()
