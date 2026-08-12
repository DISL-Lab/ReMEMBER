"""Public command line for the benchmark builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .pipeline import MODEL_ORDER, build
from .providers import OpenAICompatibleProvider, VLLMProvider


class DeterministicFixtureProvider:
    """Schema-valid CPU fixture used only by tests and release validation."""
    def complete(self, prompt: str) -> str:
        if "STAGE: salient" in prompt:
            numbered = prompt.split("WINDOW:\n", 1)[1].splitlines()[0]
            sentence = numbered.split(": ", 1)[-1]
            return json.dumps({"schema_version": "1.0", "candidates": [{"sentence_numbers": [0],
                "sentences": [sentence], "reason": "fixture", "gap_type": "A"}]})
        if "STAGE: chunk_filter" in prompt:
            chunks = json.loads(prompt.split("CHUNKS:", 1)[1].split("\nWINDOW:", 1)[0])
            return json.dumps({"schema_version": "1.0", "labels": [1] * len(chunks)})
        if "STAGE: evidence" in prompt:
            history = json.loads(prompt.split("HISTORY:", 1)[1].split("\nGAP:", 1)[0])
            return json.dumps({"schema_version": "1.0", "evidence": [{"source_id": key,
                "sentence": value, "reasoning": "fixture", "status": "resolve"} for key, value in history.items()]})
        if "STAGE: validation" in prompt:
            candidates = json.loads(prompt.split("CANDIDATES:", 1)[1].split("\nGAP:", 1)[0])
            return json.dumps({"schema_version": "1.0", "valid_source_ids": [item["source_id"] for item in candidates]})
        raise ValueError("fixture provider received an unknown stage prompt")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read benchmark configs") from exc
    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark config must be a YAML mapping")
    for section in ("sources", "prompts"):
        if isinstance(value.get(section), dict):
            value[section] = {key: str((config_path.parent / item).resolve()) if not Path(item).is_absolute() else item
                              for key, item in value[section].items()}
    if "split" in value and not Path(value["split"]).is_absolute():
        value["split"] = str((config_path.parent / value["split"]).resolve())
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m remember_benchmark")
    subs = parser.add_subparsers(dest="command", required=True)
    command = subs.add_parser("build", help="run the ordered benchmark builder")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--work-dir", required=True, type=Path)
    command.add_argument("--output-dir", required=True, type=Path)
    command.add_argument("--provider", choices=("fake", "openai", "vllm"), default="openai")
    command.add_argument("--base-url")
    command.add_argument("--api-key-env", default="OPENAI_API_KEY")
    command.add_argument("--device")
    command.add_argument("--resume", action="store_true")
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--from-stage")
    command.add_argument("--to-stage")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    models = config.get("models", {})
    if args.provider == "fake":
        providers = {label: DeterministicFixtureProvider() for label in MODEL_ORDER}
    elif args.provider == "openai":
        providers = {label: OpenAICompatibleProvider(models[label], base_url=args.base_url,
                                                      api_key_env=args.api_key_env) for label in MODEL_ORDER}
    else:
        providers = {label: VLLMProvider(models[label], device=args.device) for label in MODEL_ORDER}
    result = build(config=config, work_dir=args.work_dir, output_dir=args.output_dir, provider=providers,
                   resume=args.resume, overwrite=args.overwrite, from_stage=args.from_stage, to_stage=args.to_stage)
    print(json.dumps({"stage_names": result.stage_names}, sort_keys=True))
    return 0
