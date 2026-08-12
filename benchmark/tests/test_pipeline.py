import json
import copy
from pathlib import Path

import pytest
import remember_benchmark.pipeline as pipeline

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


@pytest.mark.parametrize("relationship", ["work_in_output", "output_in_work", "symlink_alias"])
def test_private_and_public_directories_must_be_disjoint_before_any_write(tmp_path, relationship):
    config, _, _ = _fixture(tmp_path)
    if relationship == "work_in_output":
        output, work = tmp_path / "public", tmp_path / "public/private"
    elif relationship == "output_in_work":
        work, output = tmp_path / "private", tmp_path / "private/public"
    else:
        real = tmp_path / "real"
        real.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(real, target_is_directory=True)
        output, work = real, alias / "private"
    with pytest.raises(ValueError, match="disjoint"):
        build(config, work, output, {name: FixtureProvider() for name in config["models"]},
              counter=WordCounter())
    assert not work.exists()
    assert not (output / "annotations").exists()


def test_release_rejects_unrelated_preexisting_public_output(tmp_path):
    config, _, _ = _fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "private.txt"
    sentinel.write_text("never publish\n")
    with pytest.raises(ValueError, match="public output allowlist"):
        build(config, tmp_path / "work", out,
              {name: FixtureProvider() for name in config["models"]}, counter=WordCounter())
    assert sentinel.read_text() == "never publish\n"
    assert not (tmp_path / "work/sources.jsonl").exists()


def test_fresh_build_allows_explicit_empty_output_directory(tmp_path):
    config, _, _ = _fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    build(config, tmp_path / "work", out,
          {name: FixtureProvider() for name in config["models"]}, counter=WordCounter())
    assert {path.relative_to(out) for path in out.rglob("*") if path.is_file()} == pipeline.PUBLIC_FILES


def test_merged_salient_preserves_canonical_private_reasons_and_evidence_receives_them():
    windows = [{"window_id": "E_U0000", "source": "evermembench", "range": "other",
                "turns": [{"time": "t", "speaker": "A", "dialogue": "Use it."}]}]
    voted = [{"window_id": "E_U0000", "gaps": [{"gap_idx": 0,
              "target_sentence_numbers": [0], "gap_type": "A"}]}]
    by_model = {name: [{"window_id": "E_U0000", "gaps": [{"gap_idx": 0,
        "target_sentence_numbers": [0], "gap_type": "A", "gap_reason": reason}]}]
        for name, reason in (("gemma", "gemma reason"), ("GLM", "glm reason"), ("Qwen", "qwen reason"))}
    [row] = pipeline._merge_salient(windows, voted, by_model)
    [gap] = row["gaps"]
    assert gap["gap_reasons"] == ["gemma reason", "glm reason", "qwen reason"]
    assert gap["gap_reason"] == "gemma: gemma reason; GLM: glm reason; Qwen: qwen reason"


@pytest.mark.parametrize("changed", ["temperature", "device"])
def test_resume_rejects_output_affecting_runtime_identity_change(tmp_path, changed):
    config, _, _ = _fixture(tmp_path)

    class RuntimeProvider(FixtureProvider):
        def __init__(self, temperature=0, device="cpu:0"):
            self.temperature = temperature
            self.device = device

    providers = {name: RuntimeProvider() for name in config["models"]}
    work, out = tmp_path / "work", tmp_path / "out"
    build(config, work, out, providers, counter=WordCounter())
    if changed == "temperature":
        config = copy.deepcopy(config)
        config["temperature"] = 0.25
    else:
        providers = {name: RuntimeProvider(device="cpu:1") for name in config["models"]}
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        build(config, work, out, providers, counter=WordCounter(), resume=True)


def test_model_backed_providers_are_released_after_each_stage_and_on_error(tmp_path):
    config, _, _ = _fixture(tmp_path)
    state = {"live": 0, "max": 0, "released": 0}

    class Releasable(FixtureProvider):
        temperature = 0
        device = "fixture"
        loaded = False

        def complete(self, prompt):
            if not self.loaded:
                self.loaded = True
                state["live"] += 1
                state["max"] = max(state["max"], state["live"])
            return super().complete(prompt)

        def release(self):
            if self.loaded:
                self.loaded = False
                state["live"] -= 1
                state["released"] += 1

    build(config, tmp_path / "work", tmp_path / "out",
          {name: Releasable() for name in config["models"]}, counter=WordCounter())
    assert state["max"] == 1
    assert state["live"] == 0
    assert state["released"] >= 12

    class Failing(Releasable):
        def complete(self, prompt):
            super().complete(prompt)
            raise RuntimeError("provider failed after load")

    failing = Failing()
    with pytest.raises(RuntimeError, match="provider failed"):
        build(config, tmp_path / "fail-work", tmp_path / "fail-out",
              {"gemma": failing, "GLM": Releasable(), "Qwen": Releasable()}, counter=WordCounter())
    assert state["live"] == 0


def test_failed_release_generation_leaves_previous_output_and_inventory_unchanged(tmp_path, monkeypatch):
    config, _, _, _, work, out = _complete_build(tmp_path)
    old = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    inventory = (work / "release/release.manifest.json").read_bytes()
    monkeypatch.setattr(pipeline, "export_annotations", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected export failure")))
    with pytest.raises(RuntimeError, match="injected export failure"):
        build(config, work, out, {name: FixtureProvider() for name in config["models"]},
              counter=WordCounter(), overwrite=True, from_stage="release")
    assert {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()} == old
    assert (work / "release/release.manifest.json").read_bytes() == inventory


def test_fresh_export_failure_leaves_no_public_tree_inventory_or_staging_sibling(tmp_path, monkeypatch):
    config, _, _ = _fixture(tmp_path)
    work, out = tmp_path / "work", tmp_path / "out"
    monkeypatch.setattr(pipeline, "export_annotations", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fresh export failure")))
    with pytest.raises(RuntimeError, match="fresh export failure"):
        build(config, work, out, {name: FixtureProvider() for name in config["models"]}, counter=WordCounter())
    assert not out.exists()
    assert not (work / "release/release.manifest.json").exists()
    assert not list(tmp_path.glob(".out.release-*"))


def test_inventory_failure_rolls_back_public_directory_and_old_inventory(tmp_path, monkeypatch):
    config, _, _, _, work, out = _complete_build(tmp_path)
    old = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    inventory_path = work / "release/release.manifest.json"
    inventory = inventory_path.read_bytes()
    monkeypatch.setattr(pipeline, "_write_release_inventory", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("inventory fsync failure")))
    with pytest.raises(OSError, match="inventory fsync failure"):
        build(config, work, out, {name: FixtureProvider() for name in config["models"]},
              counter=WordCounter(), overwrite=True, from_stage="release")
    assert {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()} == old
    assert inventory_path.read_bytes() == inventory
    assert not list(tmp_path.glob(".out.backup-*"))
    assert not list(tmp_path.glob(".out.release-*"))


def test_inventory_parent_fsync_failure_after_replace_restores_old_bytes(tmp_path, monkeypatch):
    config, _, _, _, work, out = _complete_build(tmp_path)
    old_output = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    inventory_path = work / "release/release.manifest.json"
    old_inventory = inventory_path.read_bytes()
    real_fsync_parent = pipeline.fsync_parent_directory
    raised = False

    def one_inventory_failure(path):
        nonlocal raised
        if Path(path) == inventory_path and not raised:
            raised = True
            raise OSError("inventory parent fsync failure")
        return real_fsync_parent(path)

    monkeypatch.setattr(pipeline, "fsync_parent_directory", one_inventory_failure)
    with pytest.raises(OSError, match="inventory parent fsync failure"):
        build(config, work, out, {name: FixtureProvider() for name in config["models"]},
              counter=WordCounter(), overwrite=True, from_stage="release")
    assert {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()} == old_output
    assert inventory_path.read_bytes() == old_inventory
    assert not list(tmp_path.glob(".out.backup-*"))


def test_every_stage_manifest_contains_current_pipeline_version(tmp_path):
    config, _, _, _, work, _ = _complete_build(tmp_path)
    manifest_paths = list(work.rglob("*.manifest.json"))
    assert manifest_paths
    # Every stage config is opaque-hashed, so changing only PIPELINE_VERSION
    # must invalidate every stage family. Direct configs are exercised by the
    # full resume matrix; this asserts the current version is no stale literal.
    assert pipeline.PIPELINE_VERSION == "2.0"
    assert '"pipeline": "1.0"' not in Path(pipeline.__file__).read_text()


def test_failed_directory_swap_rolls_back_complete_previous_release(tmp_path, monkeypatch):
    config, _, _, _, work, out = _complete_build(tmp_path)
    old = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    real_replace = pipeline._replace_directory
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected swap failure")
        return real_replace(source, target)

    monkeypatch.setattr(pipeline, "_replace_directory", fail_second)
    with pytest.raises(OSError, match="injected swap failure"):
        build(config, work, out, {name: FixtureProvider() for name in config["models"]},
              counter=WordCounter(), overwrite=True, from_stage="release")
    assert {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()} == old
    assert not list(tmp_path.glob(".out.backup-*"))
    assert not list(tmp_path.glob(".out.release-*"))


def test_public_parent_fsync_failure_after_successful_rename_rolls_back(tmp_path, monkeypatch):
    config, _, _, _, work, out = _complete_build(tmp_path)
    old = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    real_fsync_parent = pipeline.fsync_parent_directory
    raised = False

    def fail_new_output_once(path):
        nonlocal raised
        if Path(path) == out and not raised:
            raised = True
            raise OSError("public parent fsync failure")
        return real_fsync_parent(path)

    monkeypatch.setattr(pipeline, "fsync_parent_directory", fail_new_output_once)
    with pytest.raises(OSError, match="public parent fsync failure"):
        build(config, work, out, {name: FixtureProvider() for name in config["models"]},
              counter=WordCounter(), overwrite=True, from_stage="release")
    assert {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()} == old
    assert not list(tmp_path.glob(".out.backup-*"))
    assert not list(tmp_path.glob(".out.failed-*"))


def test_public_prompts_preserve_binding_canonical_semantics():
    prompts = Path(__file__).parents[1] / "prompts"
    salient = (prompts / "salient.txt").read_text()
    assert all(term in salient for term in ("decision", "request", "action", "status update", "self-contained"))
    chunk = (prompts / "relevant_chunk_filtering.txt").read_text()
    assert all(term in chunk for term in ("independently", "necessary or directly useful", "one label", "every chunk"))
    evidence = (prompts / "relevant_chunk.txt").read_text()
    assert all(term in evidence for term in ("exact missing", "strict logic gate", "every", "explicit"))
    validation = (prompts / "relevant_chunk_validation.txt").read_text()
    assert all(term in validation for term in ("Sufficiency", "Non-Inference", "Independence", "Distinctness", "Memory-Necessity"))
