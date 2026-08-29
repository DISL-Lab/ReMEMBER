#!/usr/bin/env python3
"""Validate that the public release is complete, reproducible, and text-safe."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP = {
    "dataset", "benchmark", "remember", "evaluation", "docs", "scripts", "tests",
    "README.md", "LICENSE", ".gitignore",
}
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache"}
# The local planning workspace of the tool that produced this release. Skipped
# only at the release root, and only ever as untracked scratch state: a tracked
# path of this name is still reported by the layout scan.
ROOT_ONLY_IGNORED = {Path(".superpowers")}
FORBIDDEN_DIRS = {".venv", "venv", "build", "dist", "checkpoints"}
FORBIDDEN_SUFFIXES = {".ipynb", ".log", ".zip", ".tar", ".gz", ".7z", ".rar", ".pyc",
                      ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx"}
FORBIDDEN_DATA_KEYS = {
    "history", "turns", "target_sentence", "salient_turn", "sentence", "gap_reason",
    "reasoning", "raw_output", "speaker", "speakers", "time", "timestamp", "media",
    "media_url", "url", "urls",
}
# The one text-bearing payload. It is the only `.gz` the release may carry.
CANONICAL_PAYLOAD = Path("remember_900.jsonl.gz")
ALLOWED_DATASET_FILES = frozenset({
    Path("README.md"), Path("DATASET_CARD.md"), Path("schema.json"), Path("metadata.json"),
    Path("annotations/remember_900_annotations.jsonl"), Path("splits/main_900.txt"),
    Path("PROVENANCE.json"), CANONICAL_PAYLOAD,
})
# Text keys are legitimate in the v2 payload; these structural keys never are.
FORBIDDEN_PAYLOAD_KEYS = {
    "media", "media_url", "url", "urls", "raw_output", "reasoning", "reasons",
}
EXPECTED_PAYLOAD_COUNTS = {
    "windows": 900, "gaps": 1498, "evidence": 4134, "salient_utterances": 7245,
}


def _is_canonical_payload(relative: Path) -> bool:
    """True only for the exact released payload path, relative to the release root."""
    return relative == Path("dataset") / CANONICAL_PAYLOAD


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk(root: Path):
    root = Path(root)
    for current, directories, files in os.walk(root):
        base = Path(current)
        for name in list(directories):
            candidate = base / name
            if candidate.is_symlink():
                yield candidate
        directories[:] = [
            name for name in directories
            if name not in IGNORED_DIRS
            and (base / name).relative_to(root) not in ROOT_ONLY_IGNORED
        ]
        directories[:] = [name for name in directories if not (base / name).is_symlink()]
        for name in files:
            yield base / name


def _keys(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _keys(nested)


def _tracked(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    result = subprocess.run(["git", "ls-files"], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.splitlines() if result.returncode == 0 else []


def _layout_scan(root: Path) -> list[str]:
    failures: list[str] = []
    for path in root.iterdir():
        if path.name in IGNORED_DIRS or Path(path.name) in ROOT_ONLY_IGNORED:
            continue
        if path.name not in ALLOWED_TOP:
            failures.append(f"disallowed top-level path: {path.relative_to(root)}")
    for relative in _tracked(root):
        parts = Path(relative).parts
        if any(part in IGNORED_DIRS or part in FORBIDDEN_DIRS or part.endswith(".egg-info") for part in parts):
            failures.append(f"tracked forbidden directory/cache: {relative}")
        if (Path(relative).suffix.lower() in FORBIDDEN_SUFFIXES
                and not _is_canonical_payload(Path(relative))):
            failures.append(f"tracked forbidden file type: {relative}")
    for path in _walk(root):
        relative = path.relative_to(root)
        if path.is_symlink():
            failures.append(f"symlink is forbidden: {relative}")
            continue
        if any(part in FORBIDDEN_DIRS or part.endswith(".egg-info") for part in relative.parts):
            failures.append(f"forbidden packaged directory: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and not _is_canonical_payload(relative):
            failures.append(f"forbidden packaged file type: {relative}")
        lowered = path.name.lower()
        if lowered in {"credentials", "credentials.json", "service-account.json", ".env"} or lowered.endswith(".key"):
            failures.append(f"credential/key file: {relative}")
    return sorted(set(failures))


def _payload_scan(root: Path) -> list[str]:
    failures: list[str] = []
    dataset = root / "dataset"
    if not dataset.exists():
        return ["missing dataset directory"]
    payload_files: list[Path] = []
    for path in _walk(dataset):
        if path.is_symlink():
            continue
        relative = path.relative_to(dataset)
        if path.is_file() and relative not in ALLOWED_DATASET_FILES:
            failures.append(f"unknown dataset payload: {relative}")
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            payload_files.append(path)
    for path in sorted(payload_files):
        try:
            values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
                if path.suffix == ".jsonl" else [json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid dataset JSON: {path.relative_to(root)}: {exc}")
            continue
        for value in values:
            for key in _keys(value):
                if key in FORBIDDEN_DATA_KEYS:
                    failures.append(f"forbidden dataset key {key}: {path.relative_to(root)}")
    failures.extend(_payload_integrity(dataset / CANONICAL_PAYLOAD, root))
    return failures


def _payload_integrity(path: Path, root: Path) -> list[str]:
    """Parse the compressed payload and check structure, order, and counts."""
    if not path.is_file():
        return [f"missing benchmark payload: {path.relative_to(root)}"]
    relative = path.relative_to(root)
    split_path = root / "dataset/splits/main_900.txt"
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, gzip.BadGzipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"unreadable benchmark payload: {relative}: {exc}"]

    failures: list[str] = []
    for row in rows:
        for key in _keys(row):
            if key in FORBIDDEN_PAYLOAD_KEYS:
                failures.append(f"forbidden payload key {key}: {relative}")
                break
        if failures:
            break
    try:
        split_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    except OSError as exc:
        return failures + [f"unreadable split: {exc}"]
    if [row.get("window_id") for row in rows] != split_ids:
        failures.append(f"payload window order does not match the ordered split: {relative}")
    counts = {
        "windows": len(rows),
        "gaps": sum(len(row.get("gaps") or []) for row in rows),
        "evidence": sum(len(gap.get("evidence") or [])
                        for row in rows for gap in row.get("gaps") or []),
        "salient_utterances": sum(len(row.get("salient_utterances") or []) for row in rows),
    }
    if len(split_ids) == EXPECTED_PAYLOAD_COUNTS["windows"] and counts != EXPECTED_PAYLOAD_COUNTS:
        failures.append(f"payload counts {counts} != {EXPECTED_PAYLOAD_COUNTS}: {relative}")
    for index, row in enumerate(rows):
        if row.get("schema_version") != "2.0":
            failures.append(f"payload row {index} has an unsupported schema version: {relative}")
            break
        if not (row.get("history") and row.get("turns")):
            failures.append(f"payload row {index} carries no dialogue text: {relative}")
            break
    failures.extend(_payload_content_scan(rows, relative))
    return failures


# Keys whose values are dialogue and therefore legitimately contain anything.
_FREE_TEXT_KEYS = {"dialogue", "sentence", "sentences", "target_sentence", "gap_reason"}


def _payload_content_scan(rows: Sequence[Mapping[str, Any]], relative: Path) -> list[str]:
    """Scan the decoded payload for the leaks the UTF-8 source scan cannot reach.

    Dialogue is exempt from the absolute-path rule (real conversations quote
    paths and URLs), but key names and non-dialogue values are not, and
    credential material and research-tree names are rejected everywhere.
    """
    research_names = ("0-" + "dataset", "1-" + "research", "2-" + "rebuttal")
    credential_literal = re.compile(
        r"\b(?:" + "s" + "k-" + r"[A-Za-z0-9_-]{4,}|" + "h" + "f_" + r"[A-Za-z0-9]{4,})")
    absolute = re.compile(r"(?:(?<=^)|(?<=[\s\"'=]))/[A-Za-z0-9._-]+/|[A-Za-z]:[\\/]")
    failures: list[str] = []

    def walk(value: Any, free_text: bool) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(key, str):
                    if credential_literal.search(key) or absolute.search(key):
                        failures.append(f"suspicious payload key {key!r}: {relative}")
                walk(item, free_text or key in _FREE_TEXT_KEYS)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, free_text)
        elif isinstance(value, str):
            if credential_literal.search(value):
                failures.append(f"credential literal in payload: {relative}")
            if any(name in value for name in research_names):
                failures.append(f"research-tree reference in payload: {relative}")
            if not free_text and absolute.search(value):
                failures.append(f"machine-local absolute path in payload: {relative}")

    for row in rows:
        walk(row, False)
        if failures:
            break
    return sorted(set(failures))


def _is_historical_document(relative: Path) -> bool:
    parts = relative.parts
    return (
        parts[:2] == ("docs", "plans")
        or parts[:3] == ("docs", "superpowers", "specs")
        or parts[:3] == ("docs", "superpowers", "plans")
    )


def _source_scan(root: Path) -> list[str]:
    failures: list[str] = []
    research_names = ("0-" + "dataset", "1-" + "research", "2-" + "rebuttal")
    # Match real key material, not any word containing the prefix: "task-by-task"
    # is not an OpenAI key.
    credential_literal = re.compile(
        r"\b(?:" + "s" + "k-" + r"[A-Za-z0-9_-]{4,}|" + "h" + "f_" + r"[A-Za-z0-9]{4,})"
    )
    fixed_gpu = re.compile(r"CUDA_VISIBLE_DEVICES\s*=\s*[\"']?\d")
    absolute = re.compile(
        r"(?:(?<=^)|(?<=[\s\"'=]))/(?!proc/self/fd(?:/|$)|dev/null(?:$|\s))[A-Za-z0-9._-]+/"
        r"|(?:(?<=^)|(?<=[\s\"'=]))[A-Za-z]:[\\/]",
        re.MULTILINE,
    )
    secret_words = "api[_-]?" + "key|" + "token|" + "secret|" + "password"
    python_secret = re.compile(
        r"(?im)\b(?:" + secret_words + r")\b\s*=\s*([\"'])(.{6,}?)\1"
    )
    yaml_secret = re.compile(
        r"(?im)^\s*(?:" + secret_words + r")\s*:\s*(?:[\"'])?([^\s\"'#]{6,})"
    )
    for path in _walk(root):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            if not _is_canonical_payload(relative):
                failures.append(f"non-UTF8 source asset: {relative}")
            continue
        if relative == Path(".git") and re.fullmatch(
                r"gitdir: (?:/[^\r\n]+|[A-Za-z]:[\\/][^\r\n]+)\r?\n?", text):
            continue
        intentional_tmp_output = "/" + "tmp/remember-smoke.jsonl"
        scan_text = re.sub(
            re.escape(intentional_tmp_output) + r"(?=$|[\s\"'`\\])", "", text
        ).replace("/proc/self/fd", "")
        if absolute.search(scan_text): failures.append(f"machine-local absolute path: {relative}")
        is_historical_document = _is_historical_document(relative)
        if not is_historical_document and fixed_gpu.search(text):
            failures.append(f"fixed GPU assignment: {relative}")
        if not is_historical_document and "no" + "hup" in text:
            failures.append(f"detached runtime command: {relative}")
        safe_env_name = re.compile(r"[A-Z][A-Z0-9_]*_(?:KEY|TOKEN|SECRET|PASSWORD)")
        unsafe_python_secret = any(
            not safe_env_name.fullmatch(match.group(2)) for match in python_secret.finditer(text)
        )
        unsafe_yaml_secret = any(
            not safe_env_name.fullmatch(match.group(1)) for match in yaml_secret.finditer(text)
        )
        if (credential_literal.search(text) or unsafe_python_secret or unsafe_yaml_secret):
            failures.append(f"credential literal/assignment: {relative}")
        if not is_historical_document and any(name in text for name in research_names):
            failures.append(f"research-tree runtime reference: {relative}")
        fallback_pattern = r"(?i)(?:" + "key|" + "secret|" + "token" + r")[_-]?file"
        if re.search(fallback_pattern, text):
            failures.append(f"{'secret' + '-file'} fallback: {relative}")
    return failures


def _dataset_summary(root: Path) -> dict[str, Any]:
    benchmark_src = str(ROOT / "benchmark/src")
    if benchmark_src not in sys.path:
        sys.path.insert(0, benchmark_src)
    from remember_benchmark.release import validate_annotations

    annotation = root / "dataset/annotations/remember_900_annotations.jsonl"
    schema = root / "dataset/schema.json"
    metadata = root / "dataset/metadata.json"
    split = root / "dataset/splits/main_900.txt"
    for path in (annotation, schema, metadata, split, root / "dataset" / CANONICAL_PAYLOAD):
        if path.is_symlink():
            raise ValueError(f"dataset artifact must not be a symlink: {path.relative_to(root)}")
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines() if line.strip()]
    split_ids = [line.strip() for line in split.read_text(encoding="utf-8").splitlines()]
    row_ids = [row["window_id"] for row in rows]
    if len(split_ids) != len(set(split_ids)) or row_ids != split_ids:
        raise ValueError("dataset split IDs must be unique and exactly match annotation order")
    counts = validate_annotations(rows, split_ids)
    metadata_value = json.loads(metadata.read_text(encoding="utf-8"))
    artifact_paths = {"annotations": annotation, "schema": schema, "split": split}
    for name, path in artifact_paths.items():
        recorded = metadata_value.get("artifacts", {}).get(name, {})
        if recorded.get("sha256") != _sha256(path):
            raise ValueError(f"dataset metadata hash mismatch: {name}")
    if metadata_value.get("counts") != counts:
        raise ValueError("dataset metadata counts do not match recomputed counts")
    index_hashes = {"annotations": _sha256(annotation), "schema": _sha256(schema),
                    "metadata": _sha256(metadata), "split": _sha256(split)}
    payload = root / "dataset" / CANONICAL_PAYLOAD
    payload_counts: dict[str, int] = {}
    if payload.is_file():
        with gzip.open(payload, "rt", encoding="utf-8") as handle:
            payload_rows = [json.loads(line) for line in handle if line.strip()]
        payload_counts = {
            "windows": len(payload_rows),
            "gaps": sum(len(row.get("gaps") or []) for row in payload_rows),
            "evidence": sum(len(gap.get("evidence") or [])
                            for row in payload_rows for gap in row.get("gaps") or []),
            "salient_utterances": sum(len(row.get("salient_utterances") or [])
                                      for row in payload_rows),
        }
    return {
        # `counts`/`hashes` stay at the top level for the v1 compatibility contract.
        "counts": counts, "hashes": index_hashes,
        "index": {"counts": counts, "hashes": index_hashes},
        "payload": {"counts": payload_counts,
                    "sha256": _sha256(payload) if payload.is_file() else None,
                    "path": str(CANONICAL_PAYLOAD)},
    }


def scan_release(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    layout = _layout_scan(root)
    leakage = _payload_scan(root) + _source_scan(root)
    dataset: dict[str, Any]
    try:
        dataset = _dataset_summary(root)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        dataset = {"counts": {}, "hashes": {}, "error": str(exc)}
        leakage.append(f"dataset integrity: {exc}")
    return {"layout": {"failures": layout}, "leakage": {"failures": sorted(set(leakage))}, "dataset": dataset}


def _run(command: list[str], *, env: dict[str, str] | None = None,
         remove: Sequence[str] = ()) -> dict[str, Any]:
    additions = dict(env or {})
    if additions.get("PYTHONPATH") and os.environ.get("PYTHONPATH"):
        additions["PYTHONPATH"] = os.pathsep.join((additions["PYTHONPATH"], os.environ["PYTHONPATH"]))
    child_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **additions}
    for name in remove:
        child_env.pop(name, None)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                               env=child_env, check=False)
    return {"ok": completed.returncode == 0, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:], "stderr_tail": completed.stderr[-1000:]}


def _compile() -> dict[str, Any]:
    failures = []
    for base in (ROOT / "benchmark/src", ROOT / "remember/src", ROOT / "scripts",
                 ROOT / "evaluation"):
        for path in base.rglob("*.py"):
            try:
                py_compile.compile(str(path), doraise=True, cfile=str(Path(tempfile.gettempdir()) / (path.name + ".pyc")))
            except py_compile.PyCompileError as exc:
                failures.append(f"{path.relative_to(ROOT)}: {exc.msg}")
    return {"ok": not failures, "failures": failures}


def _benchmark_smoke(temp: Path) -> dict[str, Any]:
    source = temp / "normalized.jsonl"; split = temp / "split.txt"; config = temp / "benchmark.yaml"
    # The v2 schema excludes the "other" range, so the smoke window must land in a
    # real range; with the whitespace counter that needs a turn of 8001 words.
    long_sentence = "Fixture " + "filler " * 7999 + "sentence."
    source.write_text(json.dumps({"schema_version": "1.0", "dialogue_id": "E_topic-1_group-1",
        "source": "evermembench", "turns": [{"time": "2024-01-01", "speaker": "A", "dialogue": long_sentence}]}) + "\n")
    split.write_text("E_U0000\n")
    prompts = ROOT / "benchmark/prompts"
    config.write_text(json.dumps({"sources": {"normalized": str(source)},
        "models": {"gemma": "fixture/gemma", "GLM": "fixture/glm", "Qwen": "fixture/qwen"},
        "thresholds": {"salient": 3, "chunk_filter": 3, "validation": 2},
        "windowing": {"start_tokens": 8000, "size_tokens": 1, "strides": {"evermembench": 99999}},
        "chunking": {"tokens": 128000, "overlap_tokens": 32, "chunks_per_group": 16},
        "temperature": 0, "prompts": {"salient": str(prompts / "salient.txt"),
        "chunk_filter": str(prompts / "relevant_chunk_filtering.txt"), "evidence": str(prompts / "relevant_chunk.txt"),
        "validation": str(prompts / "relevant_chunk_validation.txt")}, "split": str(split),
        "expected_counts": {"windows": 1, "gaps": 1, "evidence": 0}}))
    return _run([sys.executable, "-m", "remember_benchmark", "build", "--config", str(config),
        "--work-dir", str(temp / "benchmark-work"), "--output-dir", str(temp / "benchmark-out"),
        "--provider", "fake", "--token-counter", "whitespace"],
        env={"PYTHONPATH": str(ROOT / "benchmark/src")})


def _remember_smoke(temp: Path) -> dict[str, Any]:
    source = temp / "windows.jsonl"; output = temp / "remember.jsonl"
    source.write_text(json.dumps({"window_id": "fixture-window", "history": [{"time": "09:00", "speaker": "A",
        "dialogue": "Earlier fixture detail."}], "turns": [{"time": "10:00", "speaker": "B", "dialogue": "Use it now."}]}) + "\n")
    return _run([sys.executable, "-m", "remember", "run", "--config", str(ROOT / "remember/configs/paper.yaml"),
        "--input", str(source), "--output", str(output), "--provider", "fake",
        "--token-counter", "whitespace"],
        env={"PYTHONPATH": str(ROOT / "remember/src")})


def _evaluation_smoke(temp: Path) -> dict[str, Any]:
    """CPU-only fake evaluation: no NLTK download, no GPU, no credential."""
    work = temp / "evaluation-smoke"
    work.mkdir(parents=True, exist_ok=True)
    # Any import of a judge backend during a fake run is a defect, so make it fatal.
    for module in ("vllm", "openai"):
        (work / f"{module}.py").write_text(
            f"raise AssertionError('{module} must not be imported by a fake run')\n",
            encoding="utf-8")

    def turn(index: int, speaker: str, text: str) -> dict[str, Any]:
        return {"turn_index": index, "time": "2024-01-01 09:00:00", "speaker": speaker,
                "dialogue": text}

    row = {
        "schema_version": "2.0", "window_id": "E_U0000", "source": "evermembench",
        "domain": "evermembench", "range": "8k~16k", "tok_start": 8000, "tok_end": 8001,
        "history": [turn(0, "A", "The deadline is March 14.")],
        "turns": [turn(0, "B", "She will finish before it.")],
        "salient_utterances": [{"salient_idx": 0, "sentence_numbers": [0],
                                "sentences": ["She will finish before it."],
                                "gap_type": "C", "vote_count": 3}],
        "gaps": [{"gap_idx": 0, "target_sentence_numbers": [0],
                  "target_sentence": ["She will finish before it."], "gap_type": "C",
                  "gap_reason": "Identify the deadline.",
                  "evidence": [{"source_id": "T0-S1", "sentence": "The deadline is March 14.",
                                "status": "resolve", "validation": 1, "vote_count": 3,
                                "votes": {"gemma": 1, "GLM": 1, "Qwen": 1}}]}],
    }
    benchmark = work / "benchmark.jsonl"
    benchmark.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    predictions = work / "predictions.jsonl"
    # Punctuation-free single sentences keep the smoke off the NLTK tokenizer.
    predictions.write_text(json.dumps({
        "window_id": "E_U0000", "summary": "The deadline is March 14",
        "memory_text": "The deadline is March 14",
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    output = work / "results"
    environment = {"PYTHONPATH": str(work)}
    # A fake run must never touch a credential, so do not even offer it one.
    result = _run([sys.executable, str(ROOT / "evaluation/evaluate.py"),
                   "--benchmark", str(benchmark), "--predictions", str(predictions),
                   "--output-dir", str(output), "--provider", "fake", "--allow-partial"],
                  env=environment, remove=("OPENAI_API_KEY",))
    failures: list[str] = []
    if result["ok"]:
        expected = {"scores.json", "scores.csv", "per_window.jsonl", "raw_judgments.jsonl",
                    "run_manifest.json"}
        produced = {item.name for item in output.iterdir()} if output.is_dir() else set()
        if produced != expected:
            failures.append(f"evaluator produced {sorted(produced)}, expected {sorted(expected)}")
        else:
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                failures.append("evaluator run manifest is not complete")
            if manifest.get("provenance", {}).get("scientific_provider") is not False:
                failures.append("fake provider must never be marked scientific")
            scores = json.loads((output / "scores.json").read_text(encoding="utf-8"))
            missing = [name for name in
                       ("mr", "mkf", "wkf", "factuality", "conciseness", "mr_gated_mkf")
                       if name not in scores]
            if missing:
                failures.append(f"evaluator omitted metrics: {missing}")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        failures.append("validation process gained CUDA_VISIBLE_DEVICES")
    result["failures"] = failures
    result["ok"] = bool(result["ok"]) and not failures
    return result


def main() -> int:
    summary = scan_release(ROOT)
    compile_result = _compile()
    tests = {
        "benchmark": _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "benchmark/tests"],
                          env={"PYTHONPATH": str(ROOT / "benchmark/src")}),
        "remember": _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "remember/tests"],
                         env={"PYTHONPATH": str(ROOT / "remember/src")}),
        "evaluation": _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "evaluation/tests"], env={"PYTHONPATH": str(ROOT)}),
    }
    with tempfile.TemporaryDirectory(prefix="remember-release-validation-") as directory:
        temp = Path(directory)
        smokes = {"benchmark": _benchmark_smoke(temp), "remember": _remember_smoke(temp),
                  "evaluation": _evaluation_smoke(temp)}
    summary["compile"] = compile_result
    summary["tests"] = {"ok": all(item["ok"] for item in tests.values()), "commands": tests}
    summary["smokes"] = {"ok": all(item["ok"] for item in smokes.values()), "commands": smokes}
    exact = (summary["dataset"].get("counts") == {"windows": 900, "gaps": 1498, "evidence": 4135}
             and summary["dataset"].get("payload", {}).get("counts") == EXPECTED_PAYLOAD_COUNTS)
    summary["ok"] = (not summary["layout"]["failures"] and not summary["leakage"]["failures"] and exact
                     and compile_result["ok"] and summary["tests"]["ok"] and summary["smokes"]["ok"])
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
