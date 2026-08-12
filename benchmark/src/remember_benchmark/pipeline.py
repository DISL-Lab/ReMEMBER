"""Ordered, restart-safe benchmark annotation builder."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import WindowConfig
from .ensemble import ensemble_chunk_votes, ensemble_salient_votes, ensemble_votes
from .evidence import run_evidence
from .io import atomic_write_jsonl, fsync_parent_directory, iter_jsonl
from .manifest import (ResumeMismatch, canonical_json, configuration_hash, manifest_path_for,
                       publish_stage, sha256_file, verify_resume)
from .release import export_annotations, sanitize_annotations, validate_annotations
from .salient import expand_window_sentences, run_salient
from .schemas import NormalizedDialogue, NormalizedTurn
from .sources import normalize_sources
from .chunk_filter import Cl100kCounter, run_chunk_filter
from .validation import run_validation
from .windows import build_windows

STAGES = ("sources", "windows", "salient", "chunk_filter", "evidence", "validation", "ensemble", "release")
MODEL_ORDER = ("gemma", "GLM", "Qwen")


@dataclass(frozen=True)
class BuildResult:
    stage_names: list[str]
    work_dir: Path
    output_dir: Path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _normalized_rows(path: Path) -> list[NormalizedDialogue]:
    result: list[NormalizedDialogue] = []
    for row in _rows(path):
        result.append(NormalizedDialogue(
            row["schema_version"], row["dialogue_id"], row["source"],
            tuple(NormalizedTurn(item["time"], item["speaker"], item["dialogue"]) for item in row["turns"]),
        ))
    return result


def _provider_identity(provider: object, label: str, model_id: str) -> dict[str, Any]:
    return {
        "label": label,
        "model_id": model_id,
        "adapter": f"{type(provider).__module__}.{type(provider).__qualname__}",
        "base_url": getattr(provider, "base_url", None),
        "api_key_env": getattr(provider, "api_key_env", None),
    }


def _mode_for(stage_index: int, start: int, *, resume: bool, overwrite: bool) -> str:
    if stage_index < start:
        return "resume"
    if overwrite:
        return "overwrite"
    return "resume" if resume else "write"


def _materialize(path: Path, rows: Sequence[dict[str, Any]] | None, *, stage: str, prompt: str,
                 inputs: Sequence[Path], config: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    manifest = manifest_path_for(path)
    if mode == "resume":
        verify_resume(manifest, prompt=prompt, inputs=inputs, config=config,
                      expected_stage=stage, output_path=path)
        return _rows(path)
    if mode == "write" and (path.exists() or manifest.exists()):
        raise FileExistsError(f"{path} exists; use --resume or --overwrite")
    if rows is None:
        raise ValueError(f"{stage}: rows are required for materialization")
    publish_stage(path, rows, stage=stage, prompt=prompt, inputs=inputs, config=config)
    return list(rows)


def _stage_targets(work_dir: Path, output_dir: Path, stage_index: int) -> list[Path]:
    outputs: list[Path]
    if stage_index == 0: outputs = [work_dir / "sources.jsonl"]
    elif stage_index == 1: outputs = [work_dir / "windows.jsonl"]
    elif stage_index == 2: outputs = [work_dir / "salient" / f"{name}.jsonl" for name in (*MODEL_ORDER, "ensemble")]
    elif stage_index == 3: outputs = [work_dir / "chunk_filter" / f"{name}.jsonl" for name in (*MODEL_ORDER, "ensemble")]
    elif stage_index == 4: outputs = [work_dir / "evidence" / f"{name}.jsonl" for name in MODEL_ORDER]
    elif stage_index == 5: outputs = [work_dir / "validation" / f"{name}.jsonl" for name in MODEL_ORDER]
    elif stage_index == 6: outputs = [work_dir / "ensemble/annotations.jsonl"]
    else:
        outputs = [work_dir / "release/private_rcv.jsonl", _release_inventory_path(work_dir),
                   output_dir / "annotations/remember_900_annotations.jsonl", output_dir / "schema.json",
                   output_dir / "metadata.json", output_dir / "splits/main_900.txt"]
    targets: list[Path] = []
    for path in outputs:
        targets.append(path)
        if stage_index < 7:
            targets.append(manifest_path_for(path))
    return targets


def _preflight_fresh_range(work_dir: Path, output_dir: Path, start: int, end: int) -> None:
    existing = [path for index in range(start, end + 1)
                for path in _stage_targets(work_dir, output_dir, index) if path.exists()]
    if existing:
        raise FileExistsError(f"{existing[0]} exists; use --resume or --overwrite")


def _source_inputs(source_config: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("normalized", "evermembench", "locomo", "realtalk"):
        if key not in source_config:
            continue
        path = Path(source_config[key])
        if key == "realtalk" and path.is_dir():
            paths.extend(sorted(path.glob("Chat_*_*.json")))
        else:
            paths.append(path)
    if not paths:
        raise ValueError("sources must name user-acquired normalized, evermembench, locomo, or realtalk input")
    return paths


def _normalize(source_config: Mapping[str, Any]) -> list[NormalizedDialogue]:
    if "normalized" in source_config:
        path = Path(source_config["normalized"])
        if not path.is_file():
            raise ValueError(f"normalized source input not found at {path}; acquire the upstream data and provide its path; no download is performed")
        return _normalized_rows(path)
    return normalize_sources(**{key: Path(value) for key, value in source_config.items()})


def _model_map(config: Mapping[str, Any], providers: object) -> tuple[dict[str, str], dict[str, object]]:
    models = config.get("models")
    if not isinstance(models, Mapping) or tuple(models) != MODEL_ORDER:
        raise ValueError(f"models must be ordered exactly as {MODEL_ORDER}")
    model_ids = {str(key): str(value) for key, value in models.items()}
    if isinstance(providers, Mapping):
        if set(providers) != set(MODEL_ORDER):
            raise ValueError("provider mapping must contain gemma, GLM, and Qwen")
        provider_map = {label: providers[label] for label in MODEL_ORDER}
    else:
        provider_map = {label: providers for label in MODEL_ORDER}
    return model_ids, provider_map


def _prompt(config: Mapping[str, Any], name: str) -> tuple[str, Path]:
    prompts = config.get("prompts")
    if not isinstance(prompts, Mapping) or name not in prompts:
        raise ValueError(f"missing prompt path for {name}")
    path = Path(prompts[name])
    try:
        return path.read_text(encoding="utf-8"), path
    except OSError as exc:
        raise ValueError(f"unable to read {name} prompt at {path}") from exc


def _merge_salient(windows: Sequence[dict[str, Any]], voted: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    window_map = {row["window_id"]: row for row in windows}
    output: list[dict[str, Any]] = []
    for row in voted:
        window = window_map[row["window_id"]]
        expanded = expand_window_sentences(window["turns"])
        gaps = []
        for gap in row["gaps"]:
            numbers = gap["target_sentence_numbers"]
            gaps.append({**gap, "target_sentence": [expanded[index]["sentence"] for index in numbers],
                         "gap_reason": ""})
        output.append({**row, "source": window["source"], "range": window["range"], "gaps": gaps})
    return output


def flatten_ensemble_to_rcv(ensemble_rows: Sequence[dict[str, Any]],
                            merged_salient_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create the private flat exporter bridge without inventing target text."""
    salient = {(row["window_id"], gap["gap_idx"]): gap
               for row in merged_salient_rows for gap in row.get("gaps", [])}
    output: list[dict[str, Any]] = []
    for row in ensemble_rows:
        for gap in row.get("gaps", []):
            key = (row["window_id"], gap["gap_idx"])
            source_gap = salient.get(key)
            if source_gap is None or not source_gap.get("target_sentence"):
                raise ValueError(f"private bridge has no real merged-salient target for {key}")
            validated = []
            for item in gap.get("evidence", []):
                votes = {label: int(item.get("votes", {}).get(label, 0)) for label in MODEL_ORDER}
                validated.append({**item, "votes": votes, "vote_count": sum(votes.values())})
            output.append({
                "window_id": row["window_id"], "range": row["range"], "gap_idx": gap["gap_idx"],
                "sentence_numbers": gap["target_sentence_numbers"],
                "target_sentence": source_gap["target_sentence"], "gap_reason": source_gap.get("gap_reason", ""),
                "gap_type": gap["gap_type"], "validated": validated,
            })
    return output


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        fsync_parent_directory(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _release_inventory_path(work_dir: Path) -> Path:
    return work_dir / "release" / "release.manifest.json"


def _write_release_inventory(work_dir: Path, output_dir: Path, split_path: Path,
                             inputs: Sequence[Path], config: Mapping[str, Any]) -> None:
    names = ("annotations/remember_900_annotations.jsonl", "schema.json", "metadata.json", "splits/main_900.txt")
    payload = {
        "schema_version": "1.0", "stage": "release",
        "configuration_sha256": configuration_hash(config),
        "inputs": [{"name": path.name, "sha256": sha256_file(path)} for path in inputs],
        "artifacts": {name: sha256_file(output_dir / name) for name in names},
        "source_split_sha256": sha256_file(split_path),
    }
    _atomic_text(_release_inventory_path(work_dir), canonical_json(payload) + "\n")


def _verify_release(work_dir: Path, output_dir: Path, split_path: Path, expected: tuple[int, int, int],
                    *, inputs: Sequence[Path], config: Mapping[str, Any]) -> None:
    annotation = output_dir / "annotations/remember_900_annotations.jsonl"
    schema = output_dir / "schema.json"
    metadata = output_dir / "metadata.json"
    copied_split = output_dir / "splits/main_900.txt"
    inventory_path = _release_inventory_path(work_dir)
    leftovers = sorted(inventory_path.parent.glob(f".{inventory_path.name}.*.tmp"))
    if leftovers:
        raise ResumeMismatch(f"incomplete temporary release inventory exists: {leftovers[0]}")
    for path in (annotation, schema, metadata, copied_split, inventory_path):
        if not path.is_file():
            raise ResumeMismatch(f"release artifact is missing: {path}")
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        json.loads(schema.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeMismatch("release metadata or schema is corrupt") from exc
    if inventory.get("configuration_sha256") != configuration_hash(config):
        raise ResumeMismatch("release configuration hash mismatch")
    current_inputs = [{"name": path.name, "sha256": sha256_file(path)} for path in inputs]
    if inventory.get("inputs") != current_inputs:
        raise ResumeMismatch("release input hash mismatch")
    artifacts = inventory.get("artifacts", {})
    for name in ("annotations/remember_900_annotations.jsonl", "schema.json", "metadata.json", "splits/main_900.txt"):
        if artifacts.get(name) != sha256_file(output_dir / name):
            raise ResumeMismatch(f"release artifact hash mismatch: {name}")
    if meta.get("counts") != dict(zip(("windows", "gaps", "evidence"), expected)):
        raise ResumeMismatch("release expected-count mismatch")
    if meta.get("artifacts", {}).get("annotations", {}).get("sha256") != sha256_file(annotation):
        raise ResumeMismatch("release annotation metadata hash mismatch")
    if meta.get("artifacts", {}).get("split", {}).get("sha256") != sha256_file(copied_split):
        raise ResumeMismatch("release split metadata hash mismatch")
    if sha256_file(copied_split) != sha256_file(split_path):
        raise ResumeMismatch("release ordered split mismatch")


def build(config: Mapping[str, Any], work_dir: Path, output_dir: Path, provider: object, *,
          counter: object | None = None, resume: bool = False, overwrite: bool = False,
          from_stage: str | None = None, to_stage: str | None = None) -> BuildResult:
    """Run the exact eight-stage builder; private text stays in ``work_dir``."""
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    start_name, end_name = from_stage or STAGES[0], to_stage or STAGES[-1]
    if start_name not in STAGES or end_name not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}")
    start, end = STAGES.index(start_name), STAGES.index(end_name)
    if start > end:
        raise ValueError("from-stage/to-stage order is invalid")
    work_dir, output_dir = Path(work_dir), Path(output_dir)
    if work_dir.resolve() == output_dir.resolve():
        raise ValueError("work-dir and output-dir must be different explicit directories")
    if not resume and not overwrite:
        _preflight_fresh_range(work_dir, output_dir, start, end)
    model_ids, providers = _model_map(config, provider)
    threshold = config.get("thresholds", {})
    source_config = config.get("sources")
    if not isinstance(source_config, Mapping):
        raise ValueError("sources must be a mapping of user-acquired paths")
    source_inputs = _source_inputs(source_config)
    stage_names: list[str] = []
    global_config = _jsonable(config)

    # sources
    mode = _mode_for(0, start, resume=resume, overwrite=overwrite)
    source_path = work_dir / "sources.jsonl"
    source_rows = None if mode == "resume" else [item.to_dict() for item in _normalize(source_config)]
    _materialize(source_path, source_rows, stage="sources", prompt="", inputs=source_inputs,
                                       config={"pipeline": "1.0", "sources": global_config["sources"]}, mode=mode)
    dialogues = _normalized_rows(source_path)
    if end == 0:
        return BuildResult(list(STAGES[start:end + 1]), work_dir, output_dir)

    # windows
    windowing = config.get("windowing", {})
    strides = windowing.get("strides", {}) if isinstance(windowing, Mapping) else {}
    window_configs = {source: WindowConfig(int(windowing.get("start_tokens", 7000)),
                                            int(windowing.get("size_tokens", 1024)), int(stride))
                      for source, stride in strides.items()}
    mode = _mode_for(1, start, resume=resume, overwrite=overwrite)
    windows_path = work_dir / "windows.jsonl"
    actual_counter = counter or Cl100kCounter()
    generated_windows = None if mode == "resume" else build_windows(dialogues, actual_counter, configs=window_configs)
    windows = _materialize(windows_path, generated_windows, stage="windows", prompt="", inputs=[source_path],
                           config={"pipeline": "1.0", "windowing": _jsonable(windowing),
                                   "counter": type(actual_counter).__name__}, mode=mode)
    if end == 1:
        return BuildResult(list(STAGES[start:end + 1]), work_dir, output_dir)

    # salient: explicit per-model outputs, then deterministic merged rows.
    salient_prompt, salient_prompt_path = _prompt(config, "salient")
    salient_by_model: dict[str, list[dict[str, Any]]] = {}
    mode = _mode_for(2, start, resume=resume, overwrite=overwrite)
    salient_paths: list[Path] = []
    for label in MODEL_ORDER:
        path = work_dir / "salient" / f"{label}.jsonl"; salient_paths.append(path)
        rows = None if mode == "resume" else run_salient(windows, providers[label], salient_prompt)
        salient_by_model[label] = _materialize(path, rows, stage=f"salient/{label}", prompt=salient_prompt,
            inputs=[windows_path, salient_prompt_path], config={"pipeline": "1.0", "threshold": threshold.get("salient"),
            "provider": _provider_identity(providers[label], label, model_ids[label])}, mode=mode)
    merged_salient_path = work_dir / "salient" / "ensemble.jsonl"
    merged_rows = None if mode == "resume" else _merge_salient(windows, ensemble_salient_votes(
        salient_by_model, int(threshold.get("salient", 3))))
    merged_salient = _materialize(merged_salient_path, merged_rows, stage="salient/ensemble", prompt="",
        inputs=salient_paths, config={"pipeline": "1.0", "threshold": threshold.get("salient")}, mode=mode)
    if end == 2: return BuildResult(list(STAGES[start:end + 1]), work_dir, output_dir)

    # chunk filtering and ensemble
    chunk_prompt, chunk_prompt_path = _prompt(config, "chunk_filter")
    chunking = config.get("chunking", {})
    mode = _mode_for(3, start, resume=resume, overwrite=overwrite)
    chunks_by_model: dict[str, list[dict[str, Any]]] = {}; chunk_paths: list[Path] = []
    for label in MODEL_ORDER:
        path = work_dir / "chunk_filter" / f"{label}.jsonl"; chunk_paths.append(path)
        rows = None if mode == "resume" else run_chunk_filter(windows, providers[label], chunk_prompt,
            counter=actual_counter, chunk_tokens=int(chunking.get("tokens", 128)),
            overlap_tokens=int(chunking.get("overlap_tokens", 32)),
            chunks_per_group=int(chunking.get("chunks_per_group", 16)))
        chunks_by_model[label] = _materialize(path, rows, stage=f"chunk_filter/{label}", prompt=chunk_prompt,
            inputs=[windows_path, chunk_prompt_path], config={"pipeline": "1.0", "chunking": _jsonable(chunking),
            "provider": _provider_identity(providers[label], label, model_ids[label])}, mode=mode)
    merged_chunks_path = work_dir / "chunk_filter" / "ensemble.jsonl"
    merged_chunk_rows = None if mode == "resume" else ensemble_chunk_votes(chunks_by_model, int(threshold.get("chunk_filter", 3)))
    merged_chunks = _materialize(merged_chunks_path, merged_chunk_rows, stage="chunk_filter/ensemble", prompt="",
        inputs=chunk_paths, config={"pipeline": "1.0", "threshold": threshold.get("chunk_filter")}, mode=mode)
    if end == 3: return BuildResult(list(STAGES[start:end + 1]), work_dir, output_dir)

    evidence_prompt, evidence_prompt_path = _prompt(config, "evidence")
    mode = _mode_for(4, start, resume=resume, overwrite=overwrite)
    evidence_by_model: dict[str, list[dict[str, Any]]] = {}; evidence_paths: list[Path] = []
    for label in MODEL_ORDER:
        path = work_dir / "evidence" / f"{label}.jsonl"; evidence_paths.append(path)
        rows = None if mode == "resume" else run_evidence(windows, merged_salient, merged_chunks,
                                                           providers[label], evidence_prompt)
        evidence_by_model[label] = _materialize(path, rows, stage=f"evidence/{label}", prompt=evidence_prompt,
            inputs=[windows_path, merged_salient_path, merged_chunks_path, evidence_prompt_path],
            config={"pipeline": "1.0", "provider": _provider_identity(providers[label], label, model_ids[label])}, mode=mode)
    if end == 4: return BuildResult(list(STAGES[start:end + 1]), work_dir, output_dir)

    validation_prompt, validation_prompt_path = _prompt(config, "validation")
    mode = _mode_for(5, start, resume=resume, overwrite=overwrite)
    validated_by_model: dict[str, list[dict[str, Any]]] = {}; validation_paths: list[Path] = []
    for label in MODEL_ORDER:
        path = work_dir / "validation" / f"{label}.jsonl"; validation_paths.append(path)
        rows = None if mode == "resume" else run_validation(windows, evidence_by_model[label],
                                                             providers[label], validation_prompt)
        validated_by_model[label] = _materialize(path, rows, stage=f"validation/{label}", prompt=validation_prompt,
            inputs=[windows_path, evidence_paths[MODEL_ORDER.index(label)], validation_prompt_path],
            config={"pipeline": "1.0", "provider": _provider_identity(providers[label], label, model_ids[label])}, mode=mode)
    if end == 5: return BuildResult(list(STAGES[start:end + 1]), work_dir, output_dir)

    mode = _mode_for(6, start, resume=resume, overwrite=overwrite)
    ensemble_path = work_dir / "ensemble" / "annotations.jsonl"
    final_rows = None if mode == "resume" else ensemble_votes(validated_by_model, int(threshold.get("validation", 2)))
    ensemble_rows = _materialize(ensemble_path, final_rows, stage="ensemble", prompt="", inputs=validation_paths,
        config={"pipeline": "1.0", "threshold": threshold.get("validation"), "models": model_ids}, mode=mode)
    if end == 6: return BuildResult(list(STAGES[start:end + 1]), work_dir, output_dir)

    mode = _mode_for(7, start, resume=resume, overwrite=overwrite)
    split_path = Path(config.get("split", ""))
    if not split_path.is_file():
        raise ValueError(f"ordered split file is missing: {split_path}")
    counts_map = config.get("expected_counts", {})
    expected = tuple(int(counts_map[key]) for key in ("windows", "gaps", "evidence"))
    annotation = output_dir / "annotations/remember_900_annotations.jsonl"
    release_config = {"pipeline": "1.0", "expected_counts": list(expected), "models": model_ids,
                      "thresholds": _jsonable(threshold), "split_sha256": sha256_file(split_path)}
    if mode == "resume":
        _verify_release(work_dir, output_dir, split_path, expected, inputs=[ensemble_path, split_path], config=release_config)
    else:
        release_targets = [annotation, output_dir / "schema.json", output_dir / "metadata.json",
                           output_dir / "splits/main_900.txt", _release_inventory_path(work_dir)]
        if mode == "write" and any(path.exists() for path in release_targets):
            raise FileExistsError("release output exists; use --resume or --overwrite")
        private_rcv = flatten_ensemble_to_rcv(ensemble_rows, merged_salient)
        private_path = work_dir / "release" / "private_rcv.jsonl"
        atomic_write_jsonl(private_path, private_rcv)
        split_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()]
        # Count/schema validation is a preflight: a mismatch must not publish a
        # partial or misleading public release.
        validate_annotations(sanitize_annotations(private_rcv, split_ids), split_ids, expected)
        output_dir.mkdir(parents=True, exist_ok=True)
        copied_split = output_dir / "splits/main_900.txt"
        _atomic_text(copied_split, split_path.read_text(encoding="utf-8"))
        stats = export_annotations(private_path, copied_split, annotation, output_dir / "metadata.json", output_dir / "schema.json")
        _write_release_inventory(work_dir, output_dir, split_path, [ensemble_path, split_path], release_config)
        validate_annotations(_rows(annotation), [line.strip() for line in copied_split.read_text().splitlines()], expected)
    stage_names = list(STAGES[start:end + 1])
    return BuildResult(stage_names, work_dir, output_dir)
