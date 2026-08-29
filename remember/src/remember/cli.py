"""Command-line entry point and resumable atomic file runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ConfigError, RememberConfig
from .inference import (
    Cl100kTokenCounter,
    WhitespaceTokenCounter,
    DeterministicFakeGenerator,
    DeterministicHashEmbedder,
    OpenAICompatibleGenerator,
    SentenceTransformerEmbedder,
    VLLMGenerator,
)
from .io import _fsync_parent, atomic_write_jsonl, load_windows
from .pipeline import SCHEMA_VERSION, _read_prompt, _run_window
from .schemas import Embedder, TokenCounter


class ResumeError(RuntimeError):
    """Raised when an existing run cannot be resumed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_path(output_path: Path) -> Path:
    return Path(str(output_path) + ".manifest.json")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_parent(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _read_output_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResumeError(f"{path}:{line_number}: invalid existing output JSON") from exc
            if not isinstance(row, dict) or not isinstance(row.get("window_id"), str):
                raise ResumeError(f"{path}:{line_number}: existing output lacks window_id")
            rows.append(row)
    ids = [row["window_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ResumeError("existing output contains duplicate window IDs")
    return rows


def run_file(
    input_path: Path,
    output_path: Path,
    config: RememberConfig,
    generator,
    embedder: Embedder,
    *,
    counter: TokenCounter | None = None,
    resume: bool = False,
    overwrite: bool = False,
    memory_only: bool = False,
    runtime_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    input_path, output_path = Path(input_path), Path(output_path)
    manifest_path = _manifest_path(output_path)
    run_config = {
        **asdict(config),
        "output_schema_version": SCHEMA_VERSION,
        "memory_only": memory_only,
        "runtime": dict(runtime_config or {
            "generator_adapter": type(generator).__name__,
            "embedder_adapter": type(embedder).__name__,
            "counter_adapter": type(counter).__name__ if counter is not None else "Cl100kTokenCounter",
        }),
    }
    input_hash = _sha256(input_path)
    config_hash = _canonical_hash(run_config)

    existing: list[dict[str, Any]] = []
    if resume:
        if output_path.exists() != manifest_path.exists():
            raise ResumeError("resume requires both output and adjacent manifest")
        if output_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ResumeError(f"unable to read resume manifest: {manifest_path}") from exc
            if manifest.get("input_sha256") != input_hash:
                raise ResumeError("input hash changed; refusing stale resume")
            if manifest.get("config_sha256") != config_hash:
                raise ResumeError("configuration hash changed; refusing stale resume")
            if manifest.get("output_sha256") != _sha256(output_path):
                raise ResumeError("output hash changed; refusing stale resume")
            existing = _read_output_rows(output_path)
    elif output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; use --resume or --overwrite")

    rows_by_id = {row["window_id"]: row for row in existing}
    ordered_ids: list[str] = []
    for window in load_windows(input_path):
        if window.window_id in ordered_ids:
            raise ValueError(f"input contains duplicate window_id: {window.window_id}")
        ordered_ids.append(window.window_id)
        if window.window_id not in rows_by_id:
            rows_by_id[window.window_id] = _run_window(
                window,
                config,
                generator,
                embedder,
                counter=counter or Cl100kTokenCounter(),
                memory_only=memory_only,
                gap_prompt=_read_prompt("gap_detection.txt"),
                summary_prompt_with_memory=_read_prompt("summary_with_memory.txt"),
                summary_prompt_without_memory=_read_prompt("summary_window_only.txt"),
            ).to_dict()
    rows = [rows_by_id[window_id] for window_id in ordered_ids]
    atomic_write_jsonl(output_path, rows)
    _atomic_write_json(manifest_path, {
        "schema_version": "1.0",
        "input_sha256": input_hash,
        "config_sha256": config_hash,
        "output_sha256": _sha256(output_path),
        "window_count": len(rows),
    })
    return rows


def _load_config(path: Path) -> RememberConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read --config files") from exc
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"unable to read config: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("configuration must be a YAML object")
    allowed = {field.name for field in fields(RememberConfig)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown configuration fields: {', '.join(unknown)}")
    return RememberConfig(**value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m remember")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the canonical ReMEMber pipeline")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--provider", choices=("fake", "openai", "vllm"), default="vllm")
    run.add_argument("--base-url")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--gpu-devices")
    run.add_argument("--dense-device", default="cpu")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--memory-only", action="store_true")
    run.add_argument("--token-counter", choices=("paper", "whitespace"), default="paper",
                     help="paper uses tiktoken cl100k_base; whitespace is an approximate "
                          "dependency-free counter for smoke runs only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _load_config(args.config)
    counter = (Cl100kTokenCounter() if args.token_counter == "paper"
               else WhitespaceTokenCounter())
    if args.provider == "fake":
        generator = DeterministicFakeGenerator()
        embedder = DeterministicHashEmbedder()
    else:
        generator = (
            OpenAICompatibleGenerator(base_url=args.base_url, api_key_env=args.api_key_env)
            if args.provider == "openai"
            else VLLMGenerator(gpu_devices=args.gpu_devices)
        )
        embedder = SentenceTransformerEmbedder(
            config.dense_model,
            device=args.dense_device,
            gpu_devices=args.gpu_devices,
        )
    runtime_config = {
        "provider": args.provider,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "dense_device": args.dense_device,
        "gpu_devices": args.gpu_devices,
        "generator_adapter": type(generator).__name__,
        "embedder_adapter": type(embedder).__name__,
        "counter_adapter": type(counter).__name__,
    }
    run_file(
        args.input,
        args.output,
        config,
        generator,
        embedder,
        counter=counter,
        resume=args.resume,
        overwrite=args.overwrite,
        memory_only=args.memory_only,
        runtime_config=runtime_config,
    )
    return 0
