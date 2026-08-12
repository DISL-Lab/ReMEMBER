"""Canonical SHA-256 manifests and strict, durable stage publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import atomic_write_jsonl, fsync_parent_directory


class ResumeMismatch(ValueError):
    """An existing stage cannot be safely resumed under current inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ResumeMismatch(f"input or output file is unavailable: {path}") from exc
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("configuration must be canonical JSON data") from exc


def configuration_hash(config: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(dict(config)))


@dataclass(frozen=True)
class StageManifest:
    stage: str
    prompt_sha256: str
    configuration_sha256: str
    inputs: tuple[tuple[str, str], ...]
    output_name: str
    output_sha256: str
    path: Path = field(compare=True)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "prompt_sha256": self.prompt_sha256,
            "configuration_sha256": self.configuration_sha256,
            "inputs": [{"path": path, "sha256": digest} for path, digest in self.inputs],
            "output_name": self.output_name,
            "output_sha256": self.output_sha256,
        }


def manifest_path_for(output_path: Path) -> Path:
    return Path(str(Path(output_path)) + ".manifest.json")


def _input_hashes(inputs: Sequence[Path]) -> tuple[tuple[str, str], ...]:
    paths = [Path(path) for path in inputs]
    normalized = [str(path.resolve()) for path in paths]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate input paths are not allowed")
    return tuple((str(path), sha256_file(path)) for path in paths)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(dict(payload)))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        fsync_parent_directory(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_stage_manifest(
    output_path: Path,
    *,
    stage: str,
    prompt: str,
    inputs: Sequence[Path],
    config: Mapping[str, Any],
) -> StageManifest:
    output_path = Path(output_path)
    if not output_path.is_file():
        raise ValueError(f"stage output does not exist: {output_path}")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    manifest = StageManifest(
        stage=stage.strip(),
        prompt_sha256=sha256_text(prompt),
        configuration_sha256=configuration_hash(config),
        inputs=_input_hashes(inputs),
        output_name=output_path.name,
        output_sha256=sha256_file(output_path),
        path=manifest_path_for(output_path),
    )
    _atomic_write_json(manifest.path, manifest.to_dict())
    return manifest


def publish_stage(
    output_path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    stage: str,
    prompt: str,
    inputs: Sequence[Path],
    config: Mapping[str, Any],
) -> StageManifest:
    materialized = list(rows)
    ids = [row.get("window_id") for row in materialized if "window_id" in row]
    if len(ids) != len(set(ids)):
        raise ValueError("stage output contains duplicate window_id values")
    output_path = Path(output_path)
    atomic_write_jsonl(output_path, materialized)
    return write_stage_manifest(output_path, stage=stage, prompt=prompt, inputs=inputs, config=config)


def _read_manifest(path: Path) -> StageManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
            raise ValueError("unsupported manifest schema")
        inputs = payload["inputs"]
        if not isinstance(inputs, list):
            raise ValueError("inputs must be a list")
        parsed_inputs = tuple((item["path"], item["sha256"]) for item in inputs)
        return StageManifest(
            stage=payload["stage"],
            prompt_sha256=payload["prompt_sha256"],
            configuration_sha256=payload["configuration_sha256"],
            inputs=parsed_inputs,
            output_name=payload["output_name"],
            output_sha256=payload["output_sha256"],
            path=path,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResumeMismatch(f"invalid or missing stage manifest: {path}") from exc


def verify_resume(
    manifest: Path | StageManifest,
    *,
    prompt: str,
    inputs: Sequence[Path],
    config: Mapping[str, Any],
) -> StageManifest:
    recorded = _read_manifest(manifest.path if isinstance(manifest, StageManifest) else Path(manifest))
    output_path = recorded.path.parent / recorded.output_name
    leftovers = sorted(output_path.parent.glob(f".{output_path.name}.*.tmp"))
    if leftovers:
        raise ResumeMismatch(f"incomplete temporary output exists: {leftovers[0]}")
    if sha256_text(prompt) != recorded.prompt_sha256:
        raise ResumeMismatch("prompt hash mismatch")
    try:
        current_inputs = _input_hashes(inputs)
    except ResumeMismatch as exc:
        raise ResumeMismatch("input hash mismatch: an input is unavailable") from exc
    if current_inputs != recorded.inputs:
        raise ResumeMismatch("input hash mismatch")
    if configuration_hash(config) != recorded.configuration_sha256:
        raise ResumeMismatch("configuration hash mismatch")
    if not output_path.is_file():
        raise ResumeMismatch(f"stage output is missing: {output_path}")
    if sha256_file(output_path) != recorded.output_sha256:
        raise ResumeMismatch("output hash mismatch")
    return recorded
