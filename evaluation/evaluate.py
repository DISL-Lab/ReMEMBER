#!/usr/bin/env python3
"""Self-contained evaluator for the ReMEMber benchmark.

One file. It loads the published benchmark payload, renders the five canonical
judge prompts, runs them through a provider, and reports Memory Recall (MR),
Memory-Knowledge Faithfulness (MKF), Window-Knowledge Faithfulness (WKF),
factuality, conciseness, and the diagnostic MR-gated MKF.

It imports nothing from the research tree and nothing from the benchmark
construction package; only the standard library plus, for canonical runs, NLTK
and an OpenAI-compatible client.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = "2.0"
CANONICAL_POPULATION = 900
CANONICAL_SPLIT_SHA256 = "0eef55c9307dbb8155f664a860503760eb9314ba2594383996a48728bdb961ee"
PAPER_MODEL = "Qwen/Qwen3.6-27B"
PAPER_TEMPERATURE = 0.0

PROMPT_NAMES = ("memory_recall", "mkf", "wkf", "atomic_facts", "factuality")
PROMPT_PLACEHOLDERS = {
    "memory_recall": {"window", "memory", "gaps"},
    "mkf": {"N", "summary", "M", "gap_cases"},
    "wkf": {"N", "summary", "M", "salient_utterances"},
    "atomic_facts": {"N", "summary"},
    "factuality": {"transcript", "N", "summary"},
}
CANONICAL_PROMPT_HASHES = {
    "memory_recall": "1bc6228fa148c6b1411a6c460c13682100e976ac9e9bb9f4da4ce3fcd8085874",
    "mkf": "75aea6ec084a1c9016c914ecca915198487254dbdbe7a0c2f9b3708af397342c",
    "wkf": "c4fc5c3f91e1d13208bd6b4972c9754992e042125814800a62a792cbafe976bd",
    "atomic_facts": "db406d5eafeb0dd51f813cf45d5b22aed57e11bdedf493748b695c2172cc1676",
    "factuality": "f4613f72b2355cb96c53c10b8340e02ffa1e203b9797acb0489283b88df1a1f2",
}

REPORTED_METRICS = ("mr", "mkf", "wkf", "factuality", "conciseness", "mr_gated_mkf")
INFERENCE_METRICS = ("mr", "mkf", "wkf", "factuality")
METRIC_DEPENDENCIES = {
    "mr": ("mr",),
    "mkf": ("mkf",),
    "wkf": ("wkf",),
    "factuality": ("factuality",),
    "conciseness": ("mkf", "wkf"),
    "mr_gated_mkf": ("mr", "mkf"),
}
_ATTEMPT_ERROR_CAP = 20
METRIC_MAX_TOKENS = {"mr": 2048, "mkf": 4096, "wkf": 4096, "atomic": 2048, "factuality": 2048}

FACTUALITY_CATEGORIES = (
    "no error", "out-of-context error", "entity error", "predicate error",
    "circumstantial error", "grammatical error", "coreference error",
    "linking error", "other error",
)
FACTUALITY_ALIASES = {
    "noerror": "no error", "no_error": "no error", "no-error": "no error",
    "factual": "no error", "none": "no error",
}

# Matches analysis_v3's `length_group_for_tokens`: bucket edges in thousands
# of tokens against `tok_end`, not the dataset's finer 5-way `range` field.
LENGTH_BIN_EDGES_K = (32, 64)
LENGTH_BUCKETS = ("8K-32K", "32K-64K", "64K-160K")


class InputError(ValueError):
    """A caller-supplied input is missing, malformed, or inconsistent."""


class ResumeError(RuntimeError):
    """Existing run state cannot be reused for the requested run."""


# ---------------------------------------------------------------- primitives

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ordered_ids_sha256(ids: Sequence[str]) -> str:
    """Digest of the ordered ID list in the same form as the split file."""
    return sha256_text("".join(f"{item}\n" for item in ids))


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_dir(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    buffer = io.StringIO()
    for row in rows:
        buffer.write(canonical_json(row))
        buffer.write("\n")
    atomic_write_text(path, buffer.getvalue())


def _open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return Path(path).open("r", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with _open_maybe_gzip(Path(path)) as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise InputError(f"{path}:{number}: row must be a JSON object")
                rows.append(row)
    except OSError as exc:
        raise InputError(f"{path}: unable to read: {exc}") from exc
    except gzip.BadGzipFile as exc:
        raise InputError(f"{path}: not a valid gzip file") from exc
    return rows


# ------------------------------------------------------------ benchmark input

@dataclass(frozen=True)
class BenchmarkInput:
    rows: tuple[dict[str, Any], ...]
    ordered_ids: tuple[str, ...]
    artifact_hashes: dict[str, str]
    input_sha256: str
    schema_sha256: str | None
    manifest_sha256: str | None
    split_sha256: str | None
    construction: str | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputError(message)


def _validate_turns(value: Any, location: str) -> None:
    _require(isinstance(value, list) and bool(value), f"{location} must be a non-empty array")
    for index, turn in enumerate(value):
        _require(isinstance(turn, dict), f"{location}[{index}] must be an object")
        _require(turn.get("turn_index") == index, f"{location}[{index}].turn_index is wrong")
        for key in ("time", "speaker", "dialogue"):
            _require(isinstance(turn.get(key), str), f"{location}[{index}].{key} must be a string")


def validate_benchmark_row(row: Mapping[str, Any], location: str) -> None:
    _require(row.get("schema_version") == SCHEMA_VERSION, f"{location}: unsupported schema version")
    window_id = row.get("window_id")
    _require(isinstance(window_id, str) and bool(window_id), f"{location}: window_id must be a string")
    for key in ("source", "domain", "range"):
        _require(isinstance(row.get(key), str) and bool(row[key]), f"{location}: {key} must be a string")
    _validate_turns(row.get("history"), f"{location}.history")
    _validate_turns(row.get("turns"), f"{location}.turns")

    salient = row.get("salient_utterances")
    _require(isinstance(salient, list), f"{location}.salient_utterances must be an array")
    for index, item in enumerate(salient):
        where = f"{location}.salient_utterances[{index}]"
        _require(isinstance(item, dict), f"{where} must be an object")
        sentences = item.get("sentences")
        _require(isinstance(sentences, list) and bool(sentences), f"{where}.sentences must be non-empty")
        # The canonical evaluator does not filter blank annotation text either
        # (format_salient_utterances renders it as a real, uncoverable prompt
        # line), so an empty string is allowed here; only the type is enforced.
        _require(all(isinstance(s, str) for s in sentences), f"{where}.sentences must be text")

    gaps = row.get("gaps")
    _require(isinstance(gaps, list), f"{location}.gaps must be an array")
    seen_gap_ids: set[int] = set()
    for index, gap in enumerate(gaps):
        where = f"{location}.gaps[{index}]"
        _require(isinstance(gap, dict), f"{where} must be an object")
        gap_idx = gap.get("gap_idx")
        _require(isinstance(gap_idx, int) and gap_idx >= 0, f"{where}.gap_idx must be a non-negative integer")
        _require(gap_idx not in seen_gap_ids, f"{where}: duplicate gap_idx")
        seen_gap_ids.add(gap_idx)
        target = gap.get("target_sentence")
        _require(isinstance(target, str) and target.strip(), f"{where}.target_sentence must be non-empty text")
        evidence = gap.get("evidence")
        _require(isinstance(evidence, list), f"{where}.evidence must be an array")
        for item_index, item in enumerate(evidence):
            item_where = f"{where}.evidence[{item_index}]"
            _require(isinstance(item, dict), f"{item_where} must be an object")
            _require(isinstance(item.get("source_id"), str), f"{item_where}.source_id must be a string")
            _require(isinstance(item.get("sentence"), str) and item["sentence"].strip(),
                     f"{item_where}.sentence must be text")
            # MR keeps only validation==1 evidence, MKF keeps every candidate
            # (see render_mr_gaps / render_gap_cases), so both values must survive.
            _require(item.get("validation") in (0, 1), f"{item_where}.validation must be 0 or 1")


def _validate_rows(rows: Sequence[dict[str, Any]], location: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        validate_benchmark_row(row, f"{location}[{index}]")
        window_id = row["window_id"]
        _require(window_id not in seen, f"{location}: duplicate benchmark window_id: {window_id}")
        seen.add(window_id)
        ordered.append(window_id)
    return ordered


def load_benchmark(path: Path) -> BenchmarkInput:
    """Load and fully validate the published benchmark payload.

    A single file is the whole benchmark. A directory holds one file per
    domain shard (`*.jsonl` / `*.jsonl.gz`); every file in it is loaded, in
    filename order, and concatenated - this is how the released benchmark's
    chit-chat and workplace shards reconstruct the canonical split order.
    """
    path = Path(path)
    if path.is_dir():
        shard_paths = sorted(p for p in path.iterdir()
                             if p.is_file() and p.suffix in (".jsonl", ".gz"))
        _require(bool(shard_paths), f"{path}: directory has no .jsonl/.jsonl.gz shards")
        rows: list[dict[str, Any]] = []
        ordered: list[str] = []
        seen: set[str] = set()
        artifact_hashes: dict[str, str] = {}
        for shard_path in shard_paths:
            shard_rows = read_jsonl(shard_path)
            _require(bool(shard_rows), f"{shard_path}: benchmark shard is empty")
            shard_ordered = _validate_rows(shard_rows, str(shard_path))
            for window_id in shard_ordered:
                _require(window_id not in seen,
                         f"{path}: duplicate benchmark window_id across shards: {window_id}")
                seen.add(window_id)
            rows.extend(shard_rows)
            ordered.extend(shard_ordered)
            artifact_hashes[shard_path.name] = sha256_file(shard_path)
        input_sha256 = sha256_text(canonical_json(
            [{"name": name, "sha256": artifact_hashes[name]} for name in sorted(artifact_hashes)]
        ))
        return BenchmarkInput(
            rows=tuple(rows), ordered_ids=tuple(ordered), artifact_hashes=artifact_hashes,
            input_sha256=input_sha256, schema_sha256=None, manifest_sha256=None,
            split_sha256=ordered_ids_sha256(ordered), construction="canonical-source-join",
        )

    if not path.is_file():
        raise InputError(f"benchmark path does not exist: {path}")
    rows = read_jsonl(path)
    _require(bool(rows), f"{path}: benchmark file is empty")
    ordered = _validate_rows(rows, str(path))
    digest = sha256_file(path)
    return BenchmarkInput(
        rows=tuple(rows),
        ordered_ids=tuple(ordered),
        artifact_hashes={path.name: digest},
        input_sha256=digest,
        schema_sha256=None,
        manifest_sha256=None,
        split_sha256=ordered_ids_sha256(ordered),
        construction="canonical-source-join",
    )


def require_canonical_population(benchmark: BenchmarkInput, *, allow_partial: bool) -> None:
    """A run is official only over the exact fixed 900-window population."""
    canonical = (
        len(benchmark.ordered_ids) == CANONICAL_POPULATION
        and benchmark.split_sha256 == CANONICAL_SPLIT_SHA256
    )
    if not canonical and not allow_partial:
        raise InputError(
            f"benchmark holds {len(benchmark.ordered_ids)} windows whose ordered-ID digest is "
            f"{benchmark.split_sha256}; a full run requires the canonical {CANONICAL_POPULATION} "
            f"windows ({CANONICAL_SPLIT_SHA256}). Use --allow-partial to evaluate a subset."
        )


def is_canonical_benchmark(benchmark: BenchmarkInput) -> bool:
    return (
        len(benchmark.ordered_ids) == CANONICAL_POPULATION
        and benchmark.split_sha256 == CANONICAL_SPLIT_SHA256
    )


# ----------------------------------------------------------------- predictions

def render_structured_memory(value: Mapping[str, Any]) -> str:
    """Render a system's memory object without ever JSON-stringifying it."""
    if not isinstance(value, Mapping):
        raise InputError("memory must be a string or an object")
    text = value.get("text")
    if isinstance(text, str) and text.strip():
        return text
    items = value.get("items")
    if isinstance(items, list):
        parts = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
                raise InputError(f"memory.items[{index}] must be an object with text")
            parts.append(item["text"])
        if any(part.strip() for part in parts):
            return "\n".join(parts)
    raise InputError("memory object has no usable text")


def _prediction_memory(row: Mapping[str, Any], location: str) -> str:
    if "memory_text" in row:
        value = row["memory_text"]
        if not isinstance(value, str) or not value.strip():
            raise InputError(f"{location}.memory_text must be non-empty text")
        return value
    memory = row.get("memory")
    if isinstance(memory, str):
        if not memory.strip():
            raise InputError(f"{location}.memory must be non-empty text")
        return memory
    if memory is None:
        raise InputError(f"{location}: prediction has no memory or memory_text")
    try:
        return render_structured_memory(memory)
    except InputError as exc:
        raise InputError(f"{location}.memory: {exc}") from exc


def load_predictions(
    path: Path,
    benchmark_ids: Sequence[str],
    *,
    allow_partial: bool,
) -> dict[str, dict[str, str]]:
    rows = read_jsonl(Path(path))
    known = list(benchmark_ids)
    known_set = set(known)
    result: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        location = f"{path}[{index}]"
        window_id = row.get("window_id")
        if not isinstance(window_id, str) or not window_id:
            raise InputError(f"{location}.window_id must be a non-empty string")
        if window_id in result:
            raise InputError(f"{location}: duplicate prediction window_id: {window_id}")
        if window_id not in known_set:
            raise InputError(f"{location}: window_id is not in the benchmark: {window_id}")
        summary = row.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise InputError(f"{location}.summary must be non-empty text")
        result[window_id] = {"summary": summary, "memory": _prediction_memory(row, location)}

    if not result:
        raise InputError(f"{path}: no predictions found")
    missing = [item for item in known if item not in result]
    if missing and not allow_partial:
        raise InputError(
            f"{path}: predictions are missing {len(missing)} benchmark windows "
            f"(e.g. {missing[:3]}); use --allow-partial to score a subset"
        )
    return result


# --------------------------------------------------------------------- prompts

_LIVE_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def extract_live_placeholders(template: str) -> set[str]:
    """Fields the template really substitutes; `{{`/`}}` are escaped literals."""
    masked = template.replace("{{", "\0").replace("}}", "\1")
    return set(_LIVE_PLACEHOLDER.findall(masked))


def validate_placeholder_set(template: str, expected: set[str]) -> None:
    live = extract_live_placeholders(template)
    if live != set(expected):
        raise InputError(
            "prompt placeholders do not match the approved set: "
            f"unexpected={sorted(live - set(expected))}, missing={sorted(set(expected) - live)}"
        )


_PROMPT_TOKEN = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")


def fill_prompt(template: str, **values: object) -> str:
    """Substitute in a single pass so a value is never rescanned.

    Sequential replacement would let a prediction containing `{gap_cases}` pull
    the gold answers into the summary section, and would unescape doubled braces
    inside the caller's own text.
    """
    expected = extract_live_placeholders(template)
    if set(values) != expected:
        raise InputError("prompt values do not match template placeholders")

    def substitute(match: re.Match[str]) -> str:
        token = match.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        return str(values[match.group(1)])

    return _PROMPT_TOKEN.sub(substitute, template)


def load_prompts(prompt_dir: Path) -> dict[str, str]:
    prompt_dir = Path(prompt_dir)
    if not prompt_dir.is_dir():
        raise InputError(f"prompt directory does not exist: {prompt_dir}")
    prompts: dict[str, str] = {}
    for name in PROMPT_NAMES:
        path = prompt_dir / f"{name}.txt"
        if not path.is_file():
            raise InputError(f"missing prompt file: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise InputError(f"{path}: unable to read prompt: {exc}") from exc
        validate_placeholder_set(text, PROMPT_PLACEHOLDERS[name])
        prompts[name] = text
    return prompts


def prompts_are_canonical(prompts: Mapping[str, str]) -> bool:
    return {name: sha256_text(text) for name, text in prompts.items()} == CANONICAL_PROMPT_HASHES


_SENTENCE_PUNCTUATION = re.compile(r"[.!?]")


def split_summary_sentences(summary: str) -> list[str]:
    """Split a summary the way the canonical evaluator does (NLTK punkt).

    A punctuation-free single sentence short-circuits before importing NLTK so a
    dependency-light smoke run stays possible.
    """
    text = (summary or "").strip()
    if not text:
        return []
    if not _SENTENCE_PUNCTUATION.search(text):
        return [text]
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise InputError(
            "NLTK is required to split summaries for a canonical run. Install it with "
            "`pip install -r evaluation/requirements.txt` and fetch the tokenizer with "
            "`python -m nltk.downloader punkt_tab`."
        ) from exc
    try:
        sentences = sent_tokenize(text)
    except LookupError as exc:  # pragma: no cover - environment dependent
        raise InputError(
            "The NLTK sentence tokenizer data is missing. Fetch it with "
            "`python -m nltk.downloader punkt_tab`."
        ) from exc
    return [item.strip() for item in sentences if item.strip()]


# --------------------------------------------------------------------- parsing

_COMPLETED_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_FENCE = re.compile(r"```[a-zA-Z]*\n?")


def strip_completed_think(text: str) -> str:
    return _COMPLETED_THINK.sub("", text or "")


def strip_unterminated_think(text: str) -> str:
    cleaned = strip_completed_think(text)
    marker = cleaned.rfind("<think>")
    return cleaned[:marker] if marker != -1 else cleaned


def repair_trailing_commas(text: str) -> str:
    return _TRAILING_COMMA.sub(r"\1", text or "")


def _balanced_spans(text: str, opener: str, closer: str) -> list[tuple[int, int]]:
    """Every balanced top-level `opener..closer` span, ignoring string contents."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            if depth == 0:
                start = index
            depth += 1
        elif char == closer and depth:
            depth -= 1
            if depth == 0:
                spans.append((start, index + 1))
    return spans


def _load_array(candidate: str) -> list[Any] | None:
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        results = value.get("results")
        if isinstance(results, list):
            return results
    return None


def _salvage_objects(text: str) -> list[dict[str, Any]]:
    """Recover every complete JSON object from a truncated response."""
    recovered: list[dict[str, Any]] = []
    for start, end in _balanced_spans(text, "{", "}"):
        try:
            value = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            recovered.append(value)
    return recovered


def _objects_only(items: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    if items is None:
        return None
    if not all(isinstance(item, dict) for item in items):
        return None
    return [dict(item) for item in items]


def parse_mr(raw: str) -> list[dict[str, Any]] | None:
    """MR accepts JSON, {"results": [...]}, fenced JSONL, and truncated output."""
    text = _FENCE.sub("", strip_completed_think(raw or "")).strip()
    if not text:
        return None
    for candidate in (text, repair_trailing_commas(text)):
        parsed = _objects_only(_load_array(candidate))
        if parsed is not None:
            return parsed
    lines: list[dict[str, Any]] = []
    ok = False
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            ok = False
            break
        if not isinstance(value, dict):
            ok = False
            break
        lines.append(value)
        ok = True
    if ok and lines:
        return lines
    salvaged = _salvage_objects(repair_trailing_commas(text))
    return salvaged or None


def parse_coverage(raw: str) -> list[dict[str, Any]] | None:
    """MKF/WKF: strip finished think blocks, take the last balanced array."""
    text = _FENCE.sub("", strip_completed_think(raw or "")).strip()
    if not text:
        return None
    for start, end in reversed(_balanced_spans(text, "[", "]")):
        parsed = _objects_only(_load_array(text[start:end]))
        if parsed is not None:
            return parsed
    return None


def parse_atomic(raw: str) -> list[dict[str, Any]] | None:
    text = _FENCE.sub("", strip_completed_think(raw or "")).strip()
    if not text:
        return None
    for candidate in (text, repair_trailing_commas(text)):
        parsed = _objects_only(_load_array(candidate))
        if parsed is not None:
            return parsed
    salvaged = _salvage_objects(repair_trailing_commas(text))
    return salvaged or None


def parse_factuality(raw: str) -> list[dict[str, Any]] | None:
    text = _FENCE.sub("", strip_unterminated_think(raw or "")).strip()
    if not text:
        return None
    for candidate in (text, repair_trailing_commas(text)):
        parsed = _objects_only(_load_array(candidate))
        if parsed is not None:
            return parsed
    salvaged = _salvage_objects(repair_trailing_commas(text))
    return salvaged or None


# --------------------------------------------------------------- normalization

def _mr_positive(value: Any) -> bool:
    """Only `1` and `COVERED` count; `PARTIAL` and `0.5` are zero."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == 1
    if isinstance(value, float):
        return value == 1.0
    if isinstance(value, str):
        return value.strip().upper() in {"1", "COVERED"}
    return False


def normalize_mr(
    items: Sequence[Mapping[str, Any]] | None,
    expected_gap_ids: Sequence[str],
) -> list[dict[str, Any]] | None:
    """Per-gap fault tolerance: an unrecognized or duplicate `gap_id` is
    ignored and a `gap_id` the judge never returned defaults to decision=0,
    matching memory_recall.py's `main()` (UNCOVERED-fill on a per-gap basis,
    not a whole-window parse failure)."""
    if items is None:
        return None
    expected = list(expected_gap_ids)
    by_id: dict[str, int] = {}
    for item in items:
        gap_id = item.get("gap_id")
        if not isinstance(gap_id, str) or gap_id not in expected or gap_id in by_id:
            continue
        by_id[gap_id] = 1 if _mr_positive(item.get("decision")) else 0
    return [{"gap_id": gap_id, "decision": by_id.get(gap_id, 0)} for gap_id in expected]


def _coverage_score(value: Any) -> float | None:
    """Any finite number is a usable score, matching analysis_v3's
    `numeric_score` — the MKF/WKF rubric asks for 0/1, but a judge that drifts
    (e.g. 0.5) still contributes a real value rather than invalidating itself."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _line_numbers(value: Any, max_line: int | None) -> list[str] | None:
    """Citations must name a real summary line; MKF/WKF number from S1."""
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    lines: list[str] = []
    for item in value:
        if not isinstance(item, str) or not re.fullmatch(r"S\d+", item.strip()):
            return None
        number = int(item.strip()[1:])
        if number < 1 or (max_line is not None and number > max_line):
            return None
        lines.append(item.strip())
    return lines


def normalize_coverage(
    items: Sequence[Mapping[str, Any]] | None,
    expected_ids: Sequence[str],
    *,
    id_field: str,
    max_line: int | None = None,
) -> list[dict[str, Any]] | None:
    """Per-item fault tolerance: an item with an unrecognized/duplicate id, an
    unusable score, or an unusable line-number citation is dropped rather than
    invalidating the whole window. Matches analysis_v3's `coerce_scores`, which
    sums only over items with a finite score and never zero-fills a window
    that has at least one valid item — a window is excluded only when it has
    none (`skipped_empty_windows`). The returned list may be shorter than
    `expected_ids`; callers must key results by id, not position."""
    if items is None:
        return None
    expected = list(expected_ids)
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item.get(id_field)
        if not isinstance(item_id, str) or item_id not in expected or item_id in by_id:
            continue
        score = _coverage_score(item.get("score"))
        lines = _line_numbers(item.get("line_number"), max_line)
        if score is None or lines is None:
            continue
        by_id[item_id] = {id_field: item_id, "score": score, "line_number": lines}
    return [by_id[item_id] for item_id in expected if item_id in by_id]


def normalize_atomic_facts(
    parsed: Sequence[Mapping[str, Any]] | None,
    summary_sentences: Sequence[str],
) -> list[dict[str, Any]]:
    """Fall back to one fact per summary sentence, as the canonical evaluator
    does. A malformed item is skipped, not treated as invalidating the whole
    array — matching factuality_eval.py's `normalize_atomic_facts`, which
    keeps every well-formed item and only falls back when none survive."""
    fallback = [
        {"sentence_num": f"S{index}", "fact_id": f"S{index}-F0", "text": sentence}
        for index, sentence in enumerate(summary_sentences)
    ]
    if not parsed:
        return fallback
    facts: list[dict[str, Any]] = []
    seen_fact_ids: set[str] = set()
    for item in parsed:
        sentence_num = item.get("sentence_num")
        fact_id = item.get("fact_id")
        text = item.get("fact") if isinstance(item.get("fact"), str) else item.get("text")
        if not isinstance(sentence_num, str) or not re.fullmatch(r"S\d+", sentence_num):
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        if int(sentence_num[1:]) >= len(summary_sentences):
            continue
        if not isinstance(fact_id, str) or not fact_id.strip() or fact_id in seen_fact_ids:
            fact_id = f"{sentence_num}-F{len(facts)}"
        seen_fact_ids.add(fact_id)
        facts.append({"sentence_num": sentence_num, "fact_id": fact_id, "text": text.strip()})
    return facts or fallback


def normalize_category(value: Any) -> str:
    if not isinstance(value, str):
        return "other error"
    key = value.strip().lower()
    key = FACTUALITY_ALIASES.get(key.replace(" ", "_"), FACTUALITY_ALIASES.get(key, key))
    return key if key in FACTUALITY_CATEGORIES else "other error"


def normalize_factuality(
    parsed: Sequence[Mapping[str, Any]] | None,
    facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Positionally align the judge's response to `facts`, padding/truncating
    to len(facts) and defaulting an unusable slot to "other error" — matching
    factuality_eval.py's checker alignment (`aligned = parsed_arr[:len(expected)]`,
    padded with None, each slot run through `normalize_parsed_sentence`), which
    never discards a whole window over one bad or non-verbatim item."""
    aligned: list[Any] = list(parsed[: len(facts)]) if parsed else []
    aligned += [None] * (len(facts) - len(aligned))
    results = []
    for fact, item in zip(facts, aligned):
        category = normalize_category(item.get("category")) if isinstance(item, Mapping) else "other error"
        results.append({"fact_id": fact["fact_id"], "category": category})
    return results


def length_bucket(tok_end: Any) -> str | None:
    """Matches analysis_v3's `length_group_for_tokens`/`length_label_for_tokens`:
    buckets a window's `tok_end` against LENGTH_BIN_EDGES_K = (32, 64) (in
    thousands of tokens), independent of the dataset's own 5-way `range`
    sampling-stratum field. Returns None (analysis_v3's "unknown") when
    `tok_end` is missing or not a real number."""
    if isinstance(tok_end, bool) or not isinstance(tok_end, (int, float)):
        return None
    if isinstance(tok_end, float) and math.isnan(tok_end):
        return None
    for edge, label in zip(LENGTH_BIN_EDGES_K, LENGTH_BUCKETS):
        if tok_end < edge * 1000:
            return label
    return LENGTH_BUCKETS[-1]


# ------------------------------------------------------------------- rendering

def render_window(turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"[{turn.get('time', '')}] {turn.get('speaker', '')}: {turn.get('dialogue', '')}"
        for turn in turns
    )


def mr_gap_id(gap: Mapping[str, Any]) -> str:
    return f"gap_{gap['gap_idx']}"


def render_mr_gaps(gaps: Sequence[Mapping[str, Any]]) -> str:
    """Exactly the GAP item the canonical memory-recall evaluator builds.

    MR keeps only validation==1 evidence (load_pw_from_rcv); MKF's
    render_gap_cases keeps every candidate. The released schema carries both,
    tagged by `validation`, so each renderer applies its own filter.
    """
    payload = [
        {
            "gap_id": mr_gap_id(gap),
            "salient_utterance": gap["target_sentence"],
            "gap_type": gap.get("gap_type", ""),
            "gap_reason": gap.get("gap_reason", ""),
            "supporting_sentences": [
                {"source_id": item["source_id"], "text": item["sentence"]}
                for item in gap.get("evidence", [])
                if item.get("validation") == 1
            ],
            "supporting_reasoning": gap.get("gap_reason", ""),
        }
        for gap in gaps
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_summary_lines(sentences: Sequence[str], *, start: int) -> str:
    return "\n".join(f"S{index}: {sentence}" for index, sentence in enumerate(sentences, start))


def render_gap_cases(gaps: Sequence[Mapping[str, Any]]) -> str:
    """Exactly the G# block the canonical coverage evaluator builds.

    `utterance_num` is the gap's position in the released gap list, which is what
    the canonical builder emits when every row carries validated evidence. Unlike
    MR, this joins every evidence candidate regardless of `validation`
    (format_relevant_chunk_from_validated has no validation filter).
    """
    blocks = []
    for index, gap in enumerate(gaps):
        chunk = "\n".join(item["sentence"] for item in gap.get("evidence", []))
        blocks.append(
            f"G{index}:\n"
            f"- utterance_num: U{index}\n"
            f"- salient_utterance: {gap['target_sentence']}\n"
            f"- gap_type: {gap.get('gap_type', '')}\n"
            f"- relevant_chunk: {chunk}"
        )
    return "\n\n".join(blocks)


def render_salient_utterances(items: Sequence[Mapping[str, Any]]) -> str:
    """One line per flattened salient sentence, as the canonical evaluator does."""
    return "\n".join(
        f"U{index}: {' '.join(item['sentences'])}" for index, item in enumerate(items)
    )


def render_factuality_transcript(memory_text: str, turns: Sequence[Mapping[str, Any]]) -> str:
    return f"[MEMORY]\n{memory_text}\n\n[CURRENT WINDOW]\n{render_window(turns)}"


def render_numbered_facts(facts: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f"{index}. {fact['text']}" for index, fact in enumerate(facts, 1))


# ------------------------------------------------------------------- providers

@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    retry_count: int = 0
    attempt_errors: tuple[str, ...] = ()
    error: str | None = None


class JudgeProvider(Protocol):
    def complete_batch(self, prompts: Sequence[str], *, max_tokens: int) -> list[Completion]: ...


def _summary_lines(prompt: str) -> list[tuple[str, str]]:
    return [
        (match.group(1), match.group(2).strip())
        for match in re.finditer(r"^(S\d+): (.*)$", prompt, flags=re.MULTILINE)
    ]


class FakeProvider:
    """Deterministic substring oracle for integration testing.

    It is NOT a scientific judge: it decides purely by literal containment so an
    end-to-end run can be exercised without a model, a GPU, or a credential.
    """

    name = "fake"
    scientific = False

    def complete_batch(self, prompts: Sequence[str], *, max_tokens: int) -> list[Completion]:
        return [self._respond(prompt) for prompt in prompts]

    # Every marker below occurs in the prompt TEMPLATE, never only in a
    # substituted value, so a prediction cannot steer the dispatch.
    MARKERS = (
        ("[GAP ITEMS]", "_mr"),
        ("Gap Cases (G#)", "_mkf"),
        ("Salient Utterances (U#)", "_wkf"),
        ("Provide your answer in JSON format", "_factuality"),
        ('"fact_id"', "_atomic"),
    )

    def _respond(self, prompt: str) -> Completion:
        matched = [name for marker, name in self.MARKERS if marker in prompt]
        if len(matched) != 1:
            raise AssertionError(
                f"fake provider could not identify the prompt (matched {matched}); "
                "a prompt template changed or a value injected a marker"
            )
        text = getattr(self, matched[0])(prompt)
        return Completion(text=text, prompt_tokens=len(prompt.split()),
                          completion_tokens=len(text.split()))

    @staticmethod
    def _section(prompt: str, start: str, end: str | None) -> str:
        if start not in prompt:
            return ""
        body = prompt.split(start, 1)[1]
        return (body.split(end, 1)[0] if end and end in body else body).strip()

    def _mr(self, prompt: str) -> str:
        memory = self._section(prompt, "[MEMORY]\n", "\n\n[GAP ITEMS]")
        try:
            gaps = json.loads(self._section(prompt, "[GAP ITEMS]\n", None))
        except json.JSONDecodeError:
            gaps = []
        out = []
        for gap in gaps:
            supporting = [item.get("text", "") for item in gap.get("supporting_sentences", [])]
            covered = any(text and text in memory for text in supporting)
            out.append({"gap_id": gap.get("gap_id"), "gap_type": gap.get("gap_type"),
                        "decision": "1" if covered else "0",
                        "reasoning": "fake substring oracle"})
        return "\n".join(json.dumps(item, ensure_ascii=False) for item in out)

    def _score(self, claim: str, lines: Sequence[tuple[str, str]]) -> tuple[float, list[str]]:
        hits = [tag for tag, text in lines if claim and claim in text]
        return (1.0 if hits else 0.0), hits

    def _mkf(self, prompt: str) -> str:
        lines = _summary_lines(prompt)
        cases = re.findall(r"^G(\d+):\n- utterance_num: U\d+\n- salient_utterance: (.*)$",
                           prompt, flags=re.MULTILINE)
        results = []
        for number, claim in cases:
            score, hits = self._score(claim.strip(), lines)
            results.append({"gap_num": f"G{number}", "utterance_num": f"U{number}",
                            "question": "fake", "selected_F": claim.strip(),
                            "score": score, "line_number": hits})
        return json.dumps(results, ensure_ascii=False)

    def _wkf(self, prompt: str) -> str:
        lines = _summary_lines(prompt)
        section = prompt.split("Salient Utterances (U#):", 1)[1] if "Salient Utterances (U#):" in prompt else ""
        results = []
        for number, claim in re.findall(r"^U(\d+): (.*)$", section, flags=re.MULTILINE):
            score, hits = self._score(claim.strip(), lines)
            results.append({"utterance_num": f"U{number}", "score": score, "line_number": hits})
        return json.dumps(results, ensure_ascii=False)

    def _atomic(self, prompt: str) -> str:
        return json.dumps([
            {"sentence_num": tag, "fact_id": f"{tag}-F0", "fact": text}
            for tag, text in _summary_lines(prompt)
        ], ensure_ascii=False)

    def _factuality(self, prompt: str) -> str:
        # The transcript ends where the numbered summary begins; without that
        # bound the facts would trivially appear inside their own prompt.
        transcript = self._section(prompt, "Transcript:\n", "\n\nSummary with ")
        facts_section = prompt.split("\n", 1)[1] if "Summary with " not in prompt else \
            prompt.split("Summary with ", 1)[1]
        facts = [
            match.group(2).strip()
            for match in re.finditer(r"^(\d+)\. (.*)$", facts_section, flags=re.MULTILINE)
        ]
        return json.dumps([
            {"sentence": fact, "reason": "fake substring oracle",
             "category": "no error" if fact and fact in transcript else "out-of-context error"}
            for fact in facts
        ], ensure_ascii=False)


class OpenAICompatibleProvider:
    """Sequential OpenAI-compatible chat completions with bounded retries."""

    name = "openai"
    scientific = True

    def __init__(self, *, model: str, base_url: str, api_key: str, temperature: float,
                 max_retries: int) -> None:
        from openai import OpenAI  # imported lazily, after validation

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries

    def complete_batch(self, prompts: Sequence[str], *, max_tokens: int) -> list[Completion]:
        results: list[Completion] = []
        for prompt in prompts:
            errors: list[str] = []
            for attempt in range(self._max_retries + 1):
                try:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self._temperature,
                        max_tokens=max_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                except Exception as exc:  # provider/transport failure
                    message = f"{type(exc).__name__}: {exc}"
                    errors.append(message)
                    if attempt == 0 and _is_thinking_incompatibility(message):
                        # Record it like any other exhausted request so the caller
                        # can checkpoint the batch before aborting.
                        results.append(Completion(
                            text="", prompt_tokens=None, completion_tokens=None,
                            retry_count=attempt, attempt_errors=tuple(errors),
                            error="the endpoint rejected the no-thinking chat template option "
                                  "(chat_template_kwargs.enable_thinking=false); it is not "
                                  f"compatible with the paper's judge configuration: {message}",
                        ))
                        break
                    continue
                usage = getattr(response, "usage", None)
                results.append(Completion(
                    text=response.choices[0].message.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    retry_count=attempt,
                    attempt_errors=tuple(errors),
                ))
                break
            else:
                results.append(Completion(
                    text="", prompt_tokens=None, completion_tokens=None,
                    retry_count=self._max_retries, attempt_errors=tuple(errors),
                    error=errors[-1] if errors else "request failed",
                ))
        return results


def _is_thinking_incompatibility(message: str) -> bool:
    lowered = message.lower()
    return "chat_template_kwargs" in lowered or "enable_thinking" in lowered


class VLLMProvider:
    """Batched local vLLM generation on explicitly requested devices."""

    name = "vllm"
    scientific = True

    def __init__(self, *, model: str, gpu_devices: str, temperature: float, max_model_len: int,
                 gpu_memory_utilization: float, tensor_parallel_size: int) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        from vllm import LLM, SamplingParams  # imported only after visibility is set

        self._sampling_params = SamplingParams
        self._temperature = temperature
        self._llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )

    def complete_batch(self, prompts: Sequence[str], *, max_tokens: int) -> list[Completion]:
        params = self._sampling_params(temperature=self._temperature, max_tokens=max_tokens)
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        try:
            outputs = self._llm.chat(
                conversations, params, chat_template_kwargs={"enable_thinking": False}
            )
        except TypeError:  # older vLLM without chat_template_kwargs
            outputs = self._llm.chat(conversations, params)
        results = []
        for output in outputs:
            text = output.outputs[0].text if getattr(output, "outputs", None) else ""
            results.append(Completion(
                text=text,
                prompt_tokens=len(getattr(output, "prompt_token_ids", []) or []) or None,
                completion_tokens=None,
            ))
        return results


def build_provider(args: argparse.Namespace) -> JudgeProvider:
    if args.provider == "fake":
        return FakeProvider()
    if args.provider == "openai":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise InputError(
                f"environment variable {args.api_key_env} is unset or empty; "
                "set it or choose a different --api-key-env"
            )
        return OpenAICompatibleProvider(
            model=args.model, base_url=args.base_url, api_key=api_key,
            temperature=args.temperature, max_retries=args.max_retries,
        )
    return VLLMProvider(
        model=args.model, gpu_devices=args.gpu_devices, temperature=args.temperature,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )


# ----------------------------------------------------------------- orchestration

def expand_metric_dependencies(metrics: Sequence[str]) -> list[str]:
    """Reported metrics plus every inference metric they are computed from."""
    needed: set[str] = set()
    for metric in metrics:
        needed.update(METRIC_DEPENDENCIES[metric])
    return [metric for metric in INFERENCE_METRICS if metric in needed]


@dataclass
class _Request:
    metric: str
    phase: str
    window_id: str
    prompt: str
    normalized_input: Any
    expected_item_ids: list[str]
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def request_id(self) -> str:
        return f"{self.metric}:{self.phase}:{self.window_id}"


def _raw_row(request: _Request, completion: Completion, *, parse_ok: bool,
             returned_items: int) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "metric": request.metric,
        "phase": request.phase,
        "window_id": request.window_id,
        "prompt_sha256": sha256_text(request.prompt),
        "normalized_input_sha256": sha256_text(canonical_json(request.normalized_input)),
        "expected_item_ids": list(request.expected_item_ids),
        "raw_output": completion.text,
        "completed": completion.error is None,
        "parse_ok": parse_ok,
        "expected_items": len(request.expected_item_ids),
        "returned_items": returned_items,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "retry_count": completion.retry_count,
        "attempt_errors": list(completion.attempt_errors),
        "error": completion.error,
    }


def _reusable(row: Mapping[str, Any], request: _Request) -> bool:
    return (
        row.get("completed") is True
        and row.get("error") is None
        and isinstance(row.get("raw_output"), str)
        and row.get("metric") == request.metric
        and row.get("phase") == request.phase
        and row.get("window_id") == request.window_id
        and row.get("prompt_sha256") == sha256_text(request.prompt)
        and row.get("normalized_input_sha256") == sha256_text(canonical_json(request.normalized_input))
        and list(row.get("expected_item_ids") or []) == list(request.expected_item_ids)
        and row.get("expected_items") == len(request.expected_item_ids)
    )


def _mr_requests(rows, predictions, prompts) -> list[_Request]:
    requests = []
    for row in rows:
        window_id = row["window_id"]
        if window_id not in predictions or not row["gaps"]:
            continue
        gaps = row["gaps"]
        normalized = {"window": render_window(row["turns"]),
                      "memory": predictions[window_id]["memory"],
                      "gaps": render_mr_gaps(gaps)}
        requests.append(_Request(
            metric="mr", phase="judge", window_id=window_id,
            prompt=fill_prompt(prompts["memory_recall"], **normalized),
            normalized_input=normalized,
            expected_item_ids=[mr_gap_id(gap) for gap in gaps],
            payload={"gap_indices": [gap["gap_idx"] for gap in gaps]},
        ))
    return requests


def _coverage_requests(rows, predictions, prompts, *, metric) -> list[_Request]:
    requests = []
    for row in rows:
        window_id = row["window_id"]
        if window_id not in predictions:
            continue
        sentences = split_summary_sentences(predictions[window_id]["summary"])
        if not sentences:
            continue
        if metric == "mkf":
            items = row["gaps"]
            if not items:
                continue
            normalized = {"N": len(sentences), "summary": render_summary_lines(sentences, start=1),
                          "M": len(items), "gap_cases": render_gap_cases(items)}
            expected = [f"G{index}" for index in range(len(items))]
            payload = {"gap_indices": [gap["gap_idx"] for gap in items],
                       "summary_sentences": len(sentences)}
            template = prompts["mkf"]
        else:
            items = row["salient_utterances"]
            if not items:
                continue
            normalized = {"N": len(sentences), "summary": render_summary_lines(sentences, start=1),
                          "M": len(items), "salient_utterances": render_salient_utterances(items)}
            expected = [f"U{index}" for index in range(len(items))]
            payload = {"summary_sentences": len(sentences)}
            template = prompts["wkf"]
        requests.append(_Request(
            metric=metric, phase="judge", window_id=window_id,
            prompt=fill_prompt(template, **normalized), normalized_input=normalized,
            expected_item_ids=expected, payload=payload,
        ))
    return requests


def _atomic_requests(rows, predictions, prompts) -> list[_Request]:
    requests = []
    for row in rows:
        window_id = row["window_id"]
        if window_id not in predictions:
            continue
        sentences = split_summary_sentences(predictions[window_id]["summary"])
        if not sentences:
            continue
        normalized = {"N": len(sentences), "summary": render_summary_lines(sentences, start=0)}
        requests.append(_Request(
            metric="factuality", phase="atomic", window_id=window_id,
            prompt=fill_prompt(prompts["atomic_facts"], **normalized),
            normalized_input=normalized,
            expected_item_ids=[f"S{index}" for index in range(len(sentences))],
            payload={"summary_sentences": sentences},
        ))
    return requests


def _run_batches(requests, provider, *, max_tokens, batch_size, cache, store, checkpoint):
    """Issue only uncached requests, checkpointing every returned entry."""
    pending = [request for request in requests if request.request_id not in cache]
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        completions = provider.complete_batch([item.prompt for item in chunk],
                                              max_tokens=max_tokens)
        if len(completions) != len(chunk):
            raise RuntimeError("provider returned a different number of completions than prompts")
        failure = None
        for request, completion in zip(chunk, completions):
            store(request, completion)
            if completion.error is not None and failure is None:
                failure = (request, completion)
        if checkpoint is not None:
            checkpoint()
        if failure is not None:
            raise RuntimeError(
                f"request {failure[0].request_id} failed after retries: {failure[1].error}"
            )


def evaluate_predictions(
    benchmark_rows: Sequence[dict[str, Any]],
    predictions: Mapping[str, Mapping[str, str]],
    prompts: Mapping[str, str],
    provider: JudgeProvider,
    *,
    metrics: Sequence[str],
    metric_max_tokens: Mapping[str, int],
    batch_size: int,
    cached_raw: Sequence[Mapping[str, Any]] = (),
    checkpoint: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inference = expand_metric_dependencies(metrics)
    raw_by_id: dict[str, dict[str, Any]] = {}
    for row in cached_raw:
        request_id = row.get("request_id")
        if isinstance(request_id, str) and request_id not in raw_by_id:
            raw_by_id[request_id] = dict(row)

    parsed: dict[str, dict[str, Any]] = {}

    def store_factory(parser):
        def store(request: _Request, completion: Completion) -> None:
            value, parse_ok, returned = parser(request, completion.text)
            previous = raw_by_id.get(request.request_id, {}).get("attempt_errors") or []
            row = _raw_row(request, completion, parse_ok=parse_ok, returned_items=returned)
            # A resumed request keeps the messages from the attempt it replaces.
            row["attempt_errors"] = ([str(item) for item in previous]
                                     + list(completion.attempt_errors))[-_ATTEMPT_ERROR_CAP:]
            raw_by_id[request.request_id] = row
            parsed[request.request_id] = {"request": request, "value": value, "parse_ok": parse_ok}
        return store

    def replay(request: _Request, parser) -> bool:
        """Reuse a checkpointed result, or leave it to be overwritten by a rerun."""
        row = raw_by_id.get(request.request_id)
        if row is None or not _reusable(row, request):
            return False
        value, parse_ok, _ = parser(request, row.get("raw_output", ""))
        parsed[request.request_id] = {"request": request, "value": value, "parse_ok": parse_ok}
        return True

    def emit() -> None:
        if checkpoint is not None:
            checkpoint(_ordered_raw(raw_by_id))

    def run(requests, parser, *, max_tokens):
        cache = {}
        for request in requests:
            if replay(request, parser):
                cache[request.request_id] = True
        _run_batches(requests, provider, max_tokens=max_tokens, batch_size=batch_size,
                     cache=cache, store=store_factory(parser), checkpoint=emit)

    def mr_parser(request, text):
        items = normalize_mr(parse_mr(text), request.expected_item_ids)
        return items, items is not None, len(items or [])

    def coverage_parser(field_name):
        def parser(request, text):
            items = normalize_coverage(parse_coverage(text), request.expected_item_ids,
                                       id_field=field_name,
                                       max_line=request.payload["summary_sentences"])
            return items, items is not None, len(items or [])
        return parser

    def atomic_parser(request, text):
        sentences = request.payload["summary_sentences"]
        raw = parse_atomic(text)
        facts = normalize_atomic_facts(raw, sentences)
        return facts, raw is not None, len(facts)

    if "mr" in inference:
        run(_mr_requests(benchmark_rows, predictions, prompts), mr_parser,
            max_tokens=metric_max_tokens["mr"])
    if "mkf" in inference:
        run(_coverage_requests(benchmark_rows, predictions, prompts, metric="mkf"),
            coverage_parser("gap_num"), max_tokens=metric_max_tokens["mkf"])
    if "wkf" in inference:
        run(_coverage_requests(benchmark_rows, predictions, prompts, metric="wkf"),
            coverage_parser("utterance_num"), max_tokens=metric_max_tokens["wkf"])
    if "factuality" in inference:
        atomic_requests = _atomic_requests(benchmark_rows, predictions, prompts)
        run(atomic_requests, atomic_parser, max_tokens=metric_max_tokens["atomic"])

        checker_requests = []
        rows_by_id = {row["window_id"]: row for row in benchmark_rows}
        for request in atomic_requests:
            facts = parsed[request.request_id]["value"]
            if not facts:
                continue
            row = rows_by_id[request.window_id]
            transcript = render_factuality_transcript(
                predictions[request.window_id]["memory"], row["turns"])
            normalized = {"transcript": transcript, "N": len(facts),
                          "summary": render_numbered_facts(facts)}
            checker_requests.append(_Request(
                metric="factuality", phase="checker", window_id=request.window_id,
                prompt=fill_prompt(prompts["factuality"], **normalized),
                normalized_input=normalized,
                expected_item_ids=[fact["fact_id"] for fact in facts],
                payload={"facts": facts},
            ))

        def checker_parser(request, text):
            raw = parse_factuality(text)
            judged = normalize_factuality(raw, request.payload["facts"])
            return judged, raw is not None, len(judged)

        run(checker_requests, checker_parser, max_tokens=metric_max_tokens["factuality"])

    per_window = _build_per_window(benchmark_rows, predictions, parsed, inference,
                                   set(metrics))
    return per_window, _ordered_raw(raw_by_id)


_PHASE_ORDER = {("mr", "judge"): 0, ("mkf", "judge"): 1, ("wkf", "judge"): 2,
                ("factuality", "atomic"): 3, ("factuality", "checker"): 4}


def _ordered_raw(raw_by_id: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in sorted(
            raw_by_id.values(),
            key=lambda item: (
                _PHASE_ORDER.get((item.get("metric"), item.get("phase")), 99),
                str(item.get("window_id")),
                str(item.get("request_id")),
            ),
        )
    ]


# --------------------------------------------------------------------- scoring

def _positive_lines(items: Sequence[Mapping[str, Any]] | None) -> set[str]:
    if not items:
        return set()
    lines: set[str] = set()
    for item in items:
        if item.get("score") == 1.0:
            lines.update(item.get("line_number") or [])
    return lines


def _build_per_window(benchmark_rows, predictions, parsed, inference,
                      requested) -> list[dict[str, Any]]:
    per_window: list[dict[str, Any]] = []
    for row in benchmark_rows:
        window_id = row["window_id"]
        if window_id not in predictions:
            continue
        record: dict[str, Any] = {"window_id": window_id}
        record["length_bucket"] = length_bucket(row.get("tok_end"))
        sentences = split_summary_sentences(predictions[window_id]["summary"])
        record["summary_sentences"] = len(sentences)

        mr_entry = parsed.get(f"mr:judge:{window_id}")
        if "mr" in inference:
            gaps = row["gaps"]
            decisions = {}
            if mr_entry and mr_entry["value"] is not None:
                indices = mr_entry["request"].payload["gap_indices"]
                decisions = {index: item["decision"]
                             for index, item in zip(indices, mr_entry["value"])}
            record["mr"] = {
                "numerator": sum(decisions.values()),
                "denominator": len(gaps),
                "parse_failure": bool(gaps) and (mr_entry is None or mr_entry["value"] is None),
                "decisions": {str(key): value for key, value in sorted(decisions.items())},
            }

        mkf_entry = parsed.get(f"mkf:judge:{window_id}")
        if "mkf" in inference:
            gaps = row["gaps"]
            scores = {}
            if mkf_entry and mkf_entry["value"] is not None:
                request = mkf_entry["request"]
                id_to_index = dict(zip(request.expected_item_ids, request.payload["gap_indices"]))
                scores = {id_to_index[item["gap_num"]]: item["score"] for item in mkf_entry["value"]}
            record["mkf"] = {
                "numerator": sum(scores.values()),
                "denominator": len(scores),
                "parse_failure": bool(gaps) and not scores,
                "scores": {str(key): value for key, value in sorted(scores.items())},
            }

        wkf_entry = parsed.get(f"wkf:judge:{window_id}")
        if "wkf" in inference:
            items = row["salient_utterances"]
            value = wkf_entry["value"] if wkf_entry else None
            record["wkf"] = {
                "numerator": sum(item["score"] for item in value) if value else 0.0,
                "denominator": len(value) if value else 0,
                "parse_failure": bool(items) and not value,
            }

        if "conciseness" in requested:
            covered = _positive_lines(mkf_entry["value"] if mkf_entry else None)
            covered |= _positive_lines(wkf_entry["value"] if wkf_entry else None)
            record["conciseness"] = {
                "numerator": len(covered),
                "denominator": len(sentences),
                "parse_failure": bool(record.get("mkf", {}).get("parse_failure"))
                or bool(record.get("wkf", {}).get("parse_failure")),
            }

        if "mr_gated_mkf" in requested:
            mr_scores = record.get("mr", {}).get("decisions", {})
            mkf_scores = record.get("mkf", {}).get("scores", {})
            gated = [value if mr_scores.get(key) == 1 else 0.0 for key, value in mkf_scores.items()]
            record["mr_gated_mkf"] = {
                "numerator": sum(gated),
                "denominator": len(gated),
                "parse_failure": bool(record.get("mr", {}).get("parse_failure"))
                or bool(record.get("mkf", {}).get("parse_failure")),
            }

        if "factuality" in inference:
            atomic_entry = parsed.get(f"factuality:atomic:{window_id}")
            checker_entry = parsed.get(f"factuality:checker:{window_id}")
            facts = atomic_entry["value"] if atomic_entry else []
            judged = checker_entry["value"] if checker_entry else []
            categories = {name: 0 for name in FACTUALITY_CATEGORIES}
            for item in judged:
                categories[item["category"]] += 1
            record["factuality"] = {
                "numerator": categories["no error"],
                "denominator": len(facts),
                "categories": categories,
                "atomic_parse_failure": bool(atomic_entry) and not atomic_entry["parse_ok"],
                "checker_parse_failure": bool(facts) and not (checker_entry and checker_entry["parse_ok"]),
            }
        per_window.append(record)
    return per_window


def _aggregate(per_window: Sequence[Mapping[str, Any]], metric: str, *,
               parse_failure_count: int, diagnostic: bool = False) -> dict[str, Any]:
    entries = [row[metric] for row in per_window if metric in row]
    numerator = sum(entry["numerator"] for entry in entries)
    denominator = sum(entry["denominator"] for entry in entries)
    with_items = [entry for entry in entries if entry["denominator"] > 0]
    macro_numerator = sum(entry["numerator"] / entry["denominator"] for entry in with_items)
    result = {
        "micro": {
            "numerator": numerator, "denominator": denominator,
            "value": (numerator / denominator) if denominator else 0.0,
            "aggregation": "micro", "parse_failure_count": parse_failure_count,
            "windows_with_items": len(with_items),
            "windows_without_items": len(entries) - len(with_items),
        },
        "macro": {
            "numerator": macro_numerator, "denominator": len(with_items),
            "value": (macro_numerator / len(with_items)) if with_items else 0.0,
            "aggregation": "macro", "parse_failure_count": parse_failure_count,
            "windows_with_items": len(with_items),
            "windows_without_items": len(entries) - len(with_items),
        },
    }
    if diagnostic:
        result["diagnostic"] = True
    return result


def _score_metrics(per_window: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for metric in ("mr", "mkf", "wkf"):
        if any(metric in row for row in per_window):
            failures = sum(1 for row in per_window if row.get(metric, {}).get("parse_failure"))
            scores[metric] = _aggregate(per_window, metric, parse_failure_count=failures)

    if any("conciseness" in row for row in per_window):
        failures = sum(
            1 for row in per_window
            if row.get("mkf", {}).get("parse_failure") or row.get("wkf", {}).get("parse_failure")
        )
        scores["conciseness"] = _aggregate(per_window, "conciseness", parse_failure_count=failures)

    if any("mr_gated_mkf" in row for row in per_window):
        failures = sum(
            1 for row in per_window
            if row.get("mr", {}).get("parse_failure") or row.get("mkf", {}).get("parse_failure")
        )
        scores["mr_gated_mkf"] = _aggregate(
            per_window, "mr_gated_mkf", parse_failure_count=failures, diagnostic=True)

    if any("factuality" in row for row in per_window):
        atomic_failures = sum(
            1 for row in per_window if row.get("factuality", {}).get("atomic_parse_failure"))
        checker_failures = sum(
            1 for row in per_window if row.get("factuality", {}).get("checker_parse_failure"))
        aggregate = _aggregate(per_window, "factuality",
                               parse_failure_count=atomic_failures + checker_failures)
        categories = {name: 0 for name in FACTUALITY_CATEGORIES}
        for row in per_window:
            for name, count in row.get("factuality", {}).get("categories", {}).items():
                categories[name] += count
        aggregate["categories"] = categories
        aggregate["parse_failures_by_phase"] = {
            "atomic": atomic_failures, "checker": checker_failures}
        scores["factuality"] = aggregate
    return scores


def score_run(per_window: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Overall scores, plus each metric's breakdown across the paper's three
    `length_bucket` groups (analysis_v3's `LENGTH_BIN_EDGES_K` on `tok_end`),
    matching how the paper reports MR/MKF/WKF/conciseness/factuality by
    dialogue length rather than the dataset's finer 5-way `range` field."""
    scores: dict[str, Any] = {
        "metadata": {
            "macro_definition": "numerator is the sum of per-window ratios; "
                                "denominator is windows_with_items",
            "micro_definition": "numerator and denominator are summed across windows",
            "factuality_categories": list(FACTUALITY_CATEGORIES),
            "length_buckets": list(LENGTH_BUCKETS),
        }
    }
    for name, block in _score_metrics(per_window).items():
        by_bucket = {}
        for bucket in LENGTH_BUCKETS:
            subset = [row for row in per_window if row.get("length_bucket") == bucket]
            bucket_scores = _score_metrics(subset)
            if name in bucket_scores:
                by_bucket[bucket] = bucket_scores[name]
        if by_bucket:
            block["by_length_bucket"] = by_bucket
        scores[name] = block
    return scores


def scores_to_csv(scores: Mapping[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([
        "metric", "length_bucket", "aggregation", "value", "numerator", "denominator",
        "windows_with_items", "windows_without_items", "parse_failure_count", "diagnostic",
    ])
    for metric in REPORTED_METRICS:
        block = scores.get(metric)
        if not block:
            continue
        by_bucket = block.get("by_length_bucket", {})
        for length_bucket_label, bucket_block in [("All", block), *by_bucket.items()]:
            for aggregation in ("micro", "macro"):
                entry = bucket_block[aggregation]
                writer.writerow([
                    metric, length_bucket_label, aggregation, f"{entry['value']:.6f}",
                    f"{float(entry['numerator']):.6f}", entry["denominator"],
                    entry["windows_with_items"],
                    entry["windows_without_items"], entry["parse_failure_count"],
                    "true" if block.get("diagnostic") else "false",
                ])
    return buffer.getvalue()


# ------------------------------------------------------------------------- CLI

FINAL_FILES = ("scores.json", "scores.csv", "per_window.jsonl", "raw_judgments.jsonl",
               "run_manifest.json")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py", description="Evaluate summaries against the ReMEMber benchmark.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("fake", "vllm", "openai"), default="vllm")
    parser.add_argument("--model", default=PAPER_MODEL)
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--prompt-dir", type=Path,
                        default=Path(__file__).resolve().parent / "prompts")
    parser.add_argument("--temperature", type=float, default=PAPER_TEMPERATURE)
    parser.add_argument("--mr-max-tokens", type=int, default=METRIC_MAX_TOKENS["mr"])
    parser.add_argument("--coverage-max-tokens", type=int, default=METRIC_MAX_TOKENS["mkf"])
    parser.add_argument("--atomic-max-tokens", type=int, default=METRIC_MAX_TOKENS["atomic"])
    parser.add_argument("--factuality-max-tokens", type=int, default=METRIC_MAX_TOKENS["factuality"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-devices")
    parser.add_argument("--max-model-len", type=int, default=25000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--allow-partial", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--overwrite", action="store_true")
    return parser


def resolve_metrics(value: str) -> list[str]:
    names = [item.strip() for item in (value or "").split(",") if item.strip()]
    if not names:
        raise InputError("--metrics must name at least one metric")
    if "all" in names:
        if len(names) > 1:
            raise InputError("--metrics all cannot be combined with other metric names")
        return list(REPORTED_METRICS)
    if len(set(names)) != len(names):
        raise InputError("--metrics contains duplicate names")
    unknown = [name for name in names if name not in REPORTED_METRICS]
    if unknown:
        raise InputError(f"unknown metric(s): {', '.join(unknown)}; "
                         f"choose from {', '.join(REPORTED_METRICS)}")
    return [name for name in REPORTED_METRICS if name in names]


def validate_args(args: argparse.Namespace) -> list[str]:
    metrics = resolve_metrics(args.metrics)
    for name in ("mr_max_tokens", "coverage_max_tokens", "atomic_max_tokens",
                 "factuality_max_tokens", "batch_size", "max_model_len", "tensor_parallel_size"):
        if getattr(args, name) <= 0:
            raise InputError(f"--{name.replace('_', '-')} must be positive")
    if args.temperature < 0:
        raise InputError("--temperature must be non-negative")
    if args.max_retries < 0:
        raise InputError("--max-retries must be non-negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise InputError("--gpu-memory-utilization must be within (0, 1]")

    if args.provider == "vllm":
        raw = (args.gpu_devices or "").strip()
        if not raw:
            raise InputError("--gpu-devices is required for the vllm provider")
        parts = [item.strip() for item in raw.split(",")]
        if not all(item.isdigit() for item in parts) or len(set(parts)) != len(parts):
            raise InputError("--gpu-devices must be unique non-negative integers, comma separated")
        if args.tensor_parallel_size > len(parts):
            raise InputError("--tensor-parallel-size cannot exceed the number of --gpu-devices")
    if args.provider == "openai":
        if not (args.base_url or "").strip():
            raise InputError("--base-url is required for the openai provider")
        if not _ENV_NAME.fullmatch(args.api_key_env or ""):
            raise InputError("--api-key-env must be a valid environment variable name")
    return metrics


def build_run_manifest(
    benchmark: BenchmarkInput,
    predictions_path: Path,
    prompt_hashes: Mapping[str, str],
    args: argparse.Namespace,
    requested_metrics: Sequence[str],
    inference_metrics: Sequence[str],
    evaluated_ids: Sequence[str],
    *,
    status: str,
    checkpoint_sha256: str | None = None,
    output_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    canonical_benchmark = is_canonical_benchmark(benchmark)
    canonical_prompts = dict(prompt_hashes) == CANONICAL_PROMPT_HASHES
    paper_model_temperature = args.model == PAPER_MODEL and args.temperature == PAPER_TEMPERATURE
    scientific_provider = args.provider in ("vllm", "openai")
    return {
        "schema_version": "1.0",
        "status": status,
        "immutable": {
            "benchmark": {
                "input_sha256": benchmark.input_sha256,
                "artifact_hashes": dict(benchmark.artifact_hashes),
                "schema_sha256": benchmark.schema_sha256,
                "manifest_sha256": benchmark.manifest_sha256,
                "split_sha256": benchmark.split_sha256,
                "construction": benchmark.construction,
                "population": len(benchmark.ordered_ids),
            },
            "predictions_sha256": sha256_file(predictions_path),
            "prompts": dict(prompt_hashes),
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": {
                "mr": args.mr_max_tokens, "mkf": args.coverage_max_tokens,
                "wkf": args.coverage_max_tokens, "atomic": args.atomic_max_tokens,
                "factuality": args.factuality_max_tokens,
            },
            "requested_metrics": list(requested_metrics),
            "inference_metrics": list(inference_metrics),
            "allow_partial": bool(args.allow_partial),
            "evaluated_ids_sha256": ordered_ids_sha256(evaluated_ids),
            "evaluated_population": len(evaluated_ids),
            "sentence_splitter": "nltk.tokenize.sent_tokenize (punkt)",
        },
        "provenance": {
            "canonical_benchmark": canonical_benchmark,
            "canonical_prompts": canonical_prompts,
            "paper_model_temperature": paper_model_temperature,
            "scientific_provider": scientific_provider,
            "official_preset": all((canonical_benchmark, canonical_prompts,
                                    paper_model_temperature, scientific_provider)),
        },
        "checkpoint_sha256": checkpoint_sha256,
        "output_hashes": dict(output_hashes or {}),
    }


def validate_resume_manifest(existing: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    previous = existing.get("immutable")
    if not isinstance(previous, Mapping):
        raise ResumeError("existing run manifest has no immutable section")
    expected = current["immutable"]
    for key, value in expected.items():
        if previous.get(key) != value:
            raise ResumeError(
                f"cannot resume: {key} changed since the interrupted run "
                f"({previous.get(key)!r} -> {value!r})"
            )


def write_outputs_atomically(
    work_dir: Path,
    output_dir: Path,
    *,
    scores: Mapping[str, Any],
    scores_csv: str,
    per_window: Sequence[Mapping[str, Any]],
    raw_judgments: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    """Stage all five artifacts, revalidate them, then swap the directory in."""
    work_dir, output_dir = Path(work_dir), Path(output_dir)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staged-", dir=output_dir.parent))
    backup: Path | None = None
    try:
        atomic_write_text(staged / "scores.json", json.dumps(scores, ensure_ascii=False, indent=2) + "\n")
        atomic_write_text(staged / "scores.csv", scores_csv)
        atomic_write_jsonl(staged / "per_window.jsonl", per_window)
        atomic_write_jsonl(staged / "raw_judgments.jsonl", raw_judgments)
        final_manifest = dict(manifest)
        final_manifest["status"] = "complete"
        final_manifest["output_hashes"] = {
            name: sha256_file(staged / name) for name in FINAL_FILES if name != "run_manifest.json"
        }
        atomic_write_text(staged / "run_manifest.json",
                          json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n")

        _reload_and_validate(staged, per_window)
        umask = os.umask(0); os.umask(umask)
        os.chmod(staged, 0o777 & ~umask)
        for name in FINAL_FILES:
            os.chmod(staged / name, 0o666 & ~umask)

        if output_dir.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-",
                                           dir=output_dir.parent))
            backup.rmdir()
            os.replace(output_dir, backup)
        os.replace(staged, output_dir)
        staged = None  # type: ignore[assignment]
        _fsync_dir(output_dir.parent)
    except BaseException:
        if staged is not None and Path(staged).exists():
            shutil.rmtree(staged, ignore_errors=True)
        if backup is not None and Path(backup).exists() and not output_dir.exists():
            os.replace(backup, output_dir)
            backup = None
        # The in-progress directory is the resume point: keep it on failure.
        raise
    else:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
    finally:
        if backup is not None and Path(backup).exists():
            shutil.rmtree(backup, ignore_errors=True)


def _reload_and_validate(directory: Path, per_window: Sequence[Mapping[str, Any]]) -> None:
    for name in FINAL_FILES:
        if not (directory / name).is_file():
            raise RuntimeError(f"staged result is missing {name}")
    json.loads((directory / "scores.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    reloaded = read_jsonl(directory / "per_window.jsonl")
    if [row["window_id"] for row in reloaded] != [row["window_id"] for row in per_window]:
        raise RuntimeError("staged per_window.jsonl lost the benchmark window order")
    read_jsonl(directory / "raw_judgments.jsonl")
    for name, digest in manifest.get("output_hashes", {}).items():
        if sha256_file(directory / name) != digest:
            raise RuntimeError(f"staged {name} does not match its recorded hash")
    if not (directory / "scores.csv").read_text(encoding="utf-8").startswith("metric,length_bucket,aggregation,"):
        raise RuntimeError("staged scores.csv has an unexpected header")


def _print_scores(scores: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    provenance = manifest["provenance"]
    if not provenance["official_preset"]:
        reasons = [name for name in ("canonical_benchmark", "canonical_prompts",
                                     "paper_model_temperature", "scientific_provider")
                   if not provenance[name]]
        print(f"NOTE: this run is not the official paper preset (failing: {', '.join(reasons)})")
    print(f"{'metric':<16}{'micro':>10}{'macro':>10}{'n':>8}  notes")
    for metric in REPORTED_METRICS:
        block = scores.get(metric)
        if not block:
            continue
        notes = []
        if block.get("diagnostic"):
            notes.append("diagnostic")
        failures = block["micro"]["parse_failure_count"]
        if failures:
            notes.append(f"{failures} parse failure(s)")
        print(f"{metric:<16}{block['micro']['value']:>10.4f}{block['macro']['value']:>10.4f}"
              f"{block['micro']['denominator']:>8}  {', '.join(notes)}")


def _work_dir_for(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.in_progress"


def execute(
    args: argparse.Namespace,
    *,
    provider_factory: Callable[[argparse.Namespace], JudgeProvider] = build_provider,
) -> int:
    metrics = validate_args(args)
    output_dir = Path(args.output_dir)
    work_dir = _work_dir_for(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not (args.resume or args.overwrite):
        raise InputError(
            f"{output_dir} already contains results; pass --overwrite to replace them "
            "or --resume to continue an interrupted run"
        )

    benchmark = load_benchmark(Path(args.benchmark))
    require_canonical_population(benchmark, allow_partial=args.allow_partial)
    predictions = load_predictions(Path(args.predictions), benchmark.ordered_ids,
                                   allow_partial=args.allow_partial)
    prompts = load_prompts(Path(args.prompt_dir))
    prompt_hashes = {name: sha256_text(text) for name, text in prompts.items()}
    inference = expand_metric_dependencies(metrics)
    evaluated_ids = [item for item in benchmark.ordered_ids if item in predictions]

    manifest = build_run_manifest(
        benchmark, Path(args.predictions), prompt_hashes, args, metrics, inference,
        evaluated_ids, status="in_progress",
    )

    cached_raw: list[dict[str, Any]] = []
    if args.resume:
        existing_manifest = work_dir / "run_manifest.json"
        if not existing_manifest.is_file():
            raise ResumeError(f"nothing to resume: {existing_manifest} does not exist")
        try:
            previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeError("interrupted run manifest is unreadable") from exc
        validate_resume_manifest(previous, manifest)
        checkpoint_path = work_dir / "raw_judgments.jsonl"
        if checkpoint_path.is_file():
            if previous.get("checkpoint_sha256") not in (None, sha256_file(checkpoint_path)):
                raise ResumeError("interrupted checkpoint does not match its recorded hash")
            try:
                cached_raw = read_jsonl(checkpoint_path)
            except InputError as exc:
                raise ResumeError(f"interrupted checkpoint is malformed: {exc}") from exc

    work_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(work_dir / "run_manifest.json",
                      json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    def checkpoint(rows: Sequence[Mapping[str, Any]]) -> None:
        path = work_dir / "raw_judgments.jsonl"
        atomic_write_jsonl(path, rows)
        updated = dict(manifest)
        updated["checkpoint_sha256"] = sha256_file(path)
        atomic_write_text(work_dir / "run_manifest.json",
                          json.dumps(updated, ensure_ascii=False, indent=2) + "\n")

    provider = provider_factory(args)
    max_tokens = {
        "mr": args.mr_max_tokens, "mkf": args.coverage_max_tokens,
        "wkf": args.coverage_max_tokens, "atomic": args.atomic_max_tokens,
        "factuality": args.factuality_max_tokens,
    }
    per_window, raw_rows = evaluate_predictions(
        list(benchmark.rows), predictions, prompts, provider,
        metrics=metrics, metric_max_tokens=max_tokens, batch_size=args.batch_size,
        cached_raw=cached_raw, checkpoint=checkpoint,
    )

    scores = score_run(per_window)
    write_outputs_atomically(
        work_dir, output_dir, scores=scores, scores_csv=scores_to_csv(scores),
        per_window=per_window, raw_judgments=raw_rows, manifest=manifest,
    )
    _print_scores(scores, manifest)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except (InputError, ResumeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
