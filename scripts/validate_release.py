#!/usr/bin/env python3
"""Validate that the public release is complete, reproducible, and text-safe."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP = {
    "dataset", "benchmark", "remember", "docs", "scripts", "tests",
    "README.md", "LICENSE", ".gitignore",
}
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache"}
FORBIDDEN_DIRS = {".venv", "venv", "build", "dist", "checkpoints"}
FORBIDDEN_SUFFIXES = {".ipynb", ".log", ".zip", ".tar", ".gz", ".7z", ".rar", ".pyc",
                      ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx"}
FORBIDDEN_DATA_KEYS = {
    "history", "turns", "target_sentence", "salient_turn", "sentence", "gap_reason",
    "reasoning", "raw_output", "speaker", "speakers", "time", "timestamp", "media",
    "media_url", "url", "urls",
}
ALLOWED_DATASET_FILES = frozenset({
    Path("README.md"), Path("DATASET_CARD.md"), Path("schema.json"), Path("metadata.json"),
    Path("annotations/remember_900_annotations.jsonl"), Path("splits/main_900.txt"),
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk(root: Path):
    for current, directories, files in os.walk(root):
        base = Path(current)
        for name in list(directories):
            candidate = base / name
            if candidate.is_symlink():
                yield candidate
        directories[:] = [name for name in directories if name not in IGNORED_DIRS]
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
        if path.name in IGNORED_DIRS:
            continue
        if path.name not in ALLOWED_TOP:
            failures.append(f"disallowed top-level path: {path.relative_to(root)}")
    for relative in _tracked(root):
        parts = Path(relative).parts
        if any(part in IGNORED_DIRS or part in FORBIDDEN_DIRS or part.endswith(".egg-info") for part in parts):
            failures.append(f"tracked forbidden directory/cache: {relative}")
        if Path(relative).suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"tracked forbidden file type: {relative}")
    for path in _walk(root):
        relative = path.relative_to(root)
        if path.is_symlink():
            failures.append(f"symlink is forbidden: {relative}")
            continue
        if any(part in FORBIDDEN_DIRS or part.endswith(".egg-info") for part in relative.parts):
            failures.append(f"forbidden packaged directory: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
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
    return failures


def _source_scan(root: Path) -> list[str]:
    failures: list[str] = []
    research_names = ("0-" + "dataset", "1-" + "research", "2-" + "rebuttal")
    credential_prefixes = ("s" + "k-", "h" + "f_")
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
            failures.append(f"non-UTF8 source asset: {relative}")
            continue
        intentional_tmp_output = "/" + "tmp/remember-smoke.jsonl"
        scan_text = re.sub(
            re.escape(intentional_tmp_output) + r"(?=$|[\s\"'`\\])", "", text
        ).replace("/proc/self/fd", "")
        if absolute.search(scan_text): failures.append(f"machine-local absolute path: {relative}")
        is_plan = relative.parts[:2] == ("docs", "plans")
        if not is_plan and fixed_gpu.search(text): failures.append(f"fixed GPU assignment: {relative}")
        if not is_plan and "no" + "hup" in text: failures.append(f"detached runtime command: {relative}")
        safe_env_name = re.compile(r"[A-Z][A-Z0-9_]*_(?:KEY|TOKEN|SECRET|PASSWORD)")
        unsafe_python_secret = any(
            not safe_env_name.fullmatch(match.group(2)) for match in python_secret.finditer(text)
        )
        unsafe_yaml_secret = any(
            not safe_env_name.fullmatch(match.group(1)) for match in yaml_secret.finditer(text)
        )
        if not is_plan and (any(prefix in text for prefix in credential_prefixes)
                or unsafe_python_secret or unsafe_yaml_secret):
            failures.append(f"credential literal/assignment: {relative}")
        if not is_plan and any(name in text for name in research_names):
            failures.append(f"research-tree runtime reference: {relative}")
        fallback_pattern = r"(?i)(?:" + "key|" + "secret|" + "token" + r")[_-]?file"
        if not is_plan and re.search(fallback_pattern, text):
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
    for path in (annotation, schema, metadata, split):
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
    return {"counts": counts, "hashes": {"annotations": _sha256(annotation), "schema": _sha256(schema),
            "metadata": _sha256(metadata), "split": _sha256(split)}}


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


def _run(command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    additions = dict(env or {})
    if additions.get("PYTHONPATH") and os.environ.get("PYTHONPATH"):
        additions["PYTHONPATH"] = os.pathsep.join((additions["PYTHONPATH"], os.environ["PYTHONPATH"]))
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **additions}, check=False)
    return {"ok": completed.returncode == 0, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:], "stderr_tail": completed.stderr[-1000:]}


def _compile() -> dict[str, Any]:
    failures = []
    for base in (ROOT / "benchmark/src", ROOT / "remember/src", ROOT / "scripts"):
        for path in base.rglob("*.py"):
            try:
                py_compile.compile(str(path), doraise=True, cfile=str(Path(tempfile.gettempdir()) / (path.name + ".pyc")))
            except py_compile.PyCompileError as exc:
                failures.append(f"{path.relative_to(ROOT)}: {exc.msg}")
    return {"ok": not failures, "failures": failures}


def _benchmark_smoke(temp: Path) -> dict[str, Any]:
    source = temp / "normalized.jsonl"; split = temp / "split.txt"; config = temp / "benchmark.yaml"
    source.write_text(json.dumps({"schema_version": "1.0", "dialogue_id": "E_topic-1_group-1",
        "source": "evermembench", "turns": [{"time": "2024-01-01", "speaker": "A", "dialogue": "Fixture sentence."}]}) + "\n")
    split.write_text("E_U0000\n")
    prompts = ROOT / "benchmark/prompts"
    config.write_text(json.dumps({"sources": {"normalized": str(source)},
        "models": {"gemma": "fixture/gemma", "GLM": "fixture/glm", "Qwen": "fixture/qwen"},
        "thresholds": {"salient": 3, "chunk_filter": 3, "validation": 2},
        "windowing": {"start_tokens": 0, "size_tokens": 1, "strides": {"evermembench": 999}},
        "chunking": {"tokens": 128, "overlap_tokens": 32, "chunks_per_group": 16},
        "temperature": 0, "prompts": {"salient": str(prompts / "salient.txt"),
        "chunk_filter": str(prompts / "relevant_chunk_filtering.txt"), "evidence": str(prompts / "relevant_chunk.txt"),
        "validation": str(prompts / "relevant_chunk_validation.txt")}, "split": str(split),
        "expected_counts": {"windows": 1, "gaps": 1, "evidence": 0}}))
    return _run([sys.executable, "-m", "remember_benchmark", "build", "--config", str(config),
        "--work-dir", str(temp / "benchmark-work"), "--output-dir", str(temp / "benchmark-out"), "--provider", "fake"],
        env={"PYTHONPATH": str(ROOT / "benchmark/src")})


def _remember_smoke(temp: Path) -> dict[str, Any]:
    source = temp / "windows.jsonl"; output = temp / "remember.jsonl"
    source.write_text(json.dumps({"window_id": "fixture-window", "history": [{"time": "09:00", "speaker": "A",
        "dialogue": "Earlier fixture detail."}], "turns": [{"time": "10:00", "speaker": "B", "dialogue": "Use it now."}]}) + "\n")
    return _run([sys.executable, "-m", "remember", "run", "--config", str(ROOT / "remember/configs/paper.yaml"),
        "--input", str(source), "--output", str(output), "--provider", "fake"],
        env={"PYTHONPATH": str(ROOT / "remember/src")})


def main() -> int:
    summary = scan_release(ROOT)
    compile_result = _compile()
    tests = {
        "benchmark": _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "benchmark/tests"],
                          env={"PYTHONPATH": str(ROOT / "benchmark/src")}),
        "remember": _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "remember/tests"],
                         env={"PYTHONPATH": str(ROOT / "remember/src")}),
    }
    with tempfile.TemporaryDirectory(prefix="remember-release-validation-") as directory:
        temp = Path(directory)
        smokes = {"benchmark": _benchmark_smoke(temp), "remember": _remember_smoke(temp)}
    summary["compile"] = compile_result
    summary["tests"] = {"ok": all(item["ok"] for item in tests.values()), "commands": tests}
    summary["smokes"] = {"ok": all(item["ok"] for item in smokes.values()), "commands": smokes}
    exact = summary["dataset"].get("counts") == {"windows": 900, "gaps": 1498, "evidence": 4135}
    summary["ok"] = (not summary["layout"]["failures"] and not summary["leakage"]["failures"] and exact
                     and compile_result["ok"] and summary["tests"]["ok"] and summary["smokes"]["ok"])
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
