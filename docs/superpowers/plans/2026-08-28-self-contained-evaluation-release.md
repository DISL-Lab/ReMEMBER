# Self-Contained Benchmark and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a text-bearing 900-window benchmark and a single-file evaluator that turns generic summary/memory predictions into reproducible MR, MKF, WKF, factuality, conciseness, and MR-gated MKF scores.

**Architecture:** Keep the existing v1 coordinate-only annotation as a compatibility index, and add a canonical v2 benchmark under `dataset/benchmark/` as ten deterministic gzip JSONL shards. Put all evaluation runtime behavior in `evaluation/evaluate.py`; keep the five exact judge prompts as inspectable text assets, and route fake, vLLM, and OpenAI-compatible inference through lazy adapters in that file. Extend existing release staging, manifests, validation, and documentation so the text-bearing benchmark and evaluator are verified together without starting a GPU model.

**Tech Stack:** Python 3.10+, standard library (`argparse`, `csv`, `gzip`, `hashlib`, `json`, `tempfile`), PyYAML/tiktoken in the existing benchmark package, NLTK 3.9.1 for the canonical summary sentence split, optional OpenAI Python client, optional vLLM, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-evaluation-benchmark-release-design.md`

## Global Constraints

- Modify, generate, stage, and commit files only under `3-release/`; `0-dataset/` and `1-research/` are read-only sources.
- `evaluation/evaluate.py` is the only evaluation runtime Python file. Tests may use separate Python files.
- Preserve `dataset/annotations/remember_900_annotations.jsonl`, root `dataset/schema.json`, and root `dataset/metadata.json` as the v1 lightweight compatibility index unless a failing compatibility test proves a required byte change.
- The canonical v2 population is the exact ordered set in `dataset/splits/main_900.txt`: 900 windows, 1,498 non-empty target gaps, 4,135 validated evidence links, and 7,245 flattened WKF salient utterances.
- Target sentence coordinates are zero-based in the flattened current window. Evidence IDs use zero-based history turns and one-based sentences: `T{turn_index}-S{sentence_index}`.
- Preserve source RCV order for validated evidence inside each gap because MKF renders evidence in that order. Sort windows by the split and gaps by `gap_idx` only.
- Normalize `source` and `domain` to the source implied by the window-ID prefix (`E=evermembench`, `L=locomo`, `R=realtalk`). External rows provide `domain`; pipeline rows provide `source`.
- `gap_reason` is the released concise gap criterion used by the canonical MR prompt. It is not validator chain-of-thought. Generic `reasoning`, per-model `reasons`, validator rationale, and raw model output remain forbidden.
- Every generated gzip member uses UTF-8 JSONL with a real trailing newline per record, `mtime=0`, and an empty embedded filename.
- The canonical split SHA-256 is `0eef55c9307dbb8155f664a860503760eb9314ba2594383996a48728bdb961ee`.
- The five prompt files must have these exact SHA-256 hashes:
  - `memory_recall.txt`: `1bc6228fa148c6b1411a6c460c13682100e976ac9e9bb9f4da4ce3fcd8085874`
  - `mkf.txt`: `75aea6ec084a1c9016c914ecca915198487254dbdbe7a0c2f9b3708af397342c`
  - `wkf.txt`: `c4fc5c3f91e1d13208bd6b4972c9754992e042125814800a62a792cbafe976bd`
  - `atomic_facts.txt`: `db406d5eafeb0dd51f813cf45d5b22aed57e11bdedf493748b695c2172cc1676`
  - `factuality.txt`: `f4613f72b2355cb96c53c10b8340e02ffa1e203b9797acb0489283b88df1a1f2`
- Official judge defaults are `Qwen/Qwen3.6-27B`, temperature `0.0`, and no reasoning mode.
- Local vLLM requires an explicit non-empty `--gpu-devices`. No test, validator, or fake-provider command may initialize CUDA, vLLM, OpenAI, credentials, or network access.
- The user declined external API calls for implementation/verification. Exercise the OpenAI-compatible adapter only through injected fake modules; do not inspect, create, reuse, or send a real API key.
- MR/MKF/WKF and factuality-checker parse failures stay in metric denominators and score zero. Atomic decomposition parse failure follows the canonical exception: fall back to one fact per summary sentence, report the atomic parse failure, then score those fallback facts with the checker. Transient provider failures retry and then abort without marking the request complete.
- Full evaluation requires the exact 900-ID set. `--allow-partial` is the only way to evaluate a non-empty subset.
- Use `apply_patch` for hand-edited source, test, prompt, and documentation files. Use the tested exporter itself for generated shard/schema/manifest artifacts.

---

## File Structure

### New files

- `benchmark/src/remember_benchmark/text_release.py`: v2 row construction, semantic validation, schema/manifest generation, directory validation, and standalone materialization CLI.
- `benchmark/tests/test_text_release.py`: literal coordinate-binding, semantic-validation, deterministic gzip, manifest, and CLI tests.
- `dataset/benchmark/remember_900-00000-of-00010.jsonl.gz` through `remember_900-00009-of-00010.jsonl.gz`: canonical text-bearing payload.
- `dataset/benchmark/schema.json`: v2 record schema.
- `dataset/benchmark/manifest.json`: ordered shard/source/schema/split hashes and denominators.
- `dataset/benchmark/README.md`: payload-specific schema, provenance, and usage.
- `evaluation/evaluate.py`: all evaluation runtime logic and CLI.
- `evaluation/prompts/{memory_recall,mkf,wkf,atomic_facts,factuality}.txt`: exact canonical prompts.
- `evaluation/requirements.txt`: NLTK and OpenAI-compatible API dependencies; vLLM remains optional and environment-specific.
- `evaluation/README.md`: evaluator installation, inputs, providers, metrics, outputs, resume, and examples.
- `evaluation/tests/test_evaluate.py`: pure-function, provider-adapter, aggregation, CLI, and end-to-end tests.
- `evaluation/tests/fixtures/predictions.jsonl`: small generic prediction fixture.
- `evaluation/tests/fixtures/golden_scores.json`: exact fake-provider aggregate result.

### Existing files modified

- `benchmark/src/remember_benchmark/io.py`: deterministic atomic gzip JSONL reader/writer.
- `benchmark/src/remember_benchmark/ensemble.py`: retain exact salient vote counts needed in v2 records.
- `benchmark/src/remember_benchmark/pipeline.py`: stage, publish, inventory, and resume-verify both v1 index and v2 benchmark.
- `benchmark/tests/test_pipeline.py`: transactional publication and corruption/resume cases for shards.
- `benchmark/tests/test_published_dataset.py`: validate the checked-in v1 index and v2 benchmark together.
- `scripts/validate_release.py`: narrowly permit and deeply validate canonical gzip shards and the evaluator; add a fake evaluator smoke.
- `tests/test_release_layout.py`: update product-folder, text-policy, corruption, and smoke assertions.
- `README.md`, `benchmark/README.md`, `dataset/README.md`, `dataset/DATASET_CARD.md`: reframe the canonical payload and document direct evaluation.

---

### Task 1: Restore release-validation support for approved design and plan documents

**Files:**
- Modify: `tests/test_release_layout.py`
- Modify: `scripts/validate_release.py:_source_scan`

**Interfaces:**
- Consumes: repository-relative paths scanned by `scan_release(root)`.
- Produces: `_is_historical_document(relative: Path) -> bool`, used only to exempt historical repository names, fixed-GPU prose, and detached-runner prose in `docs/plans/`, `docs/superpowers/specs/`, and `docs/superpowers/plans/`; machine-local paths, credential literals, and credential-path fallbacks remain rejected everywhere.

- [ ] **Step 1: Add the failing historical-document scope test**

Add this test beside `test_plan_docs_still_scan_machine_paths_but_may_name_historical_research_tree`:

```python
def test_superpowers_docs_may_name_historical_tree_but_still_scan_machine_paths(tmp_path):
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
    path = tmp_path / "docs/superpowers/specs/release.md"
    path.parent.mkdir(parents=True)
    historical = "1-" + "research"
    machine_path = "/" + "mnt/private/source"
    secret_name = "api_" + "key"
    path.write_text(
        f"source: {historical}\nlocal: {machine_path}\n{secret_name}: abcdefgh\n"
    )

    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "research-tree runtime reference" not in failures
    assert "machine-local absolute path: docs/superpowers/specs/release.md" in failures
    assert "credential literal/assignment: docs/superpowers/specs/release.md" in failures
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q -p no:cacheprovider tests/test_release_layout.py::test_superpowers_docs_may_name_historical_tree_but_still_scan_machine_paths
```

Expected: FAIL because `_source_scan()` exempts only `docs/plans`.

- [ ] **Step 3: Implement the narrow document exemption**

Add and use this helper in `scripts/validate_release.py`:

```python
def _is_historical_document(relative: Path) -> bool:
    parts = relative.parts
    return (
        parts[:2] == ("docs", "plans")
        or parts[:3] == ("docs", "superpowers", "specs")
        or parts[:3] == ("docs", "superpowers", "plans")
    )
```

Replace the local `is_plan` checks with `is_historical_document`. Apply the exemption only to research-tree names, fixed GPU examples, and detached runner names. Keep absolute paths, credential literals/assignments, and credential-path fallbacks outside the exemption.

- [ ] **Step 4: Run focused and existing layout tests**

Run:

```bash
pytest -q -p no:cacheprovider tests/test_release_layout.py -k 'superpowers_docs or plan_docs'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_release.py tests/test_release_layout.py
git commit -m "fix(release): scan superpowers docs as historical prose"
```

---

### Task 2: Build and semantically validate v2 text-bearing benchmark records

**Files:**
- Create: `benchmark/src/remember_benchmark/text_release.py`
- Create: `benchmark/tests/test_text_release.py`
- Modify: `benchmark/src/remember_benchmark/ensemble.py`
- Modify: `benchmark/tests/test_stages.py`

**Interfaces:**
- Consumes: window rows shaped like `combined_sampled.jsonl` or pipeline `windows.jsonl`, RCV gap rows, salient rows shaped as either `gap_analysis[]` or pipeline `gaps[]`, and ordered split IDs.
- Produces:

```python
BENCHMARK_SCHEMA_VERSION = "2.0"
BENCHMARK_SHARD_COUNT = 10
BENCHMARK_ROWS_PER_SHARD = 90

def build_benchmark_records(
    source_rows: Iterable[dict[str, Any]],
    rcv_rows: Iterable[dict[str, Any]],
    salient_rows: Iterable[dict[str, Any]],
    ordered_window_ids: Sequence[str],
) -> list[dict[str, Any]]: ...

def build_benchmark_records_from_pipeline(
    windows: Iterable[dict[str, Any]],
    private_rcv: Iterable[dict[str, Any]],
    merged_salient: Iterable[dict[str, Any]],
    ordered_window_ids: Sequence[str],
) -> list[dict[str, Any]]: ...

def validate_benchmark_records(
    rows: Sequence[dict[str, Any]],
    ordered_window_ids: Sequence[str],
    expected_counts: tuple[int, int, int] | None = None,
    expected_salient_count: int | None = None,
) -> dict[str, Any]: ...

def benchmark_schema() -> dict[str, Any]: ...
```

- [ ] **Step 1: Write a failing exact-record construction test**

Create a literal one-window fixture in `test_text_release.py`:

```python
def test_build_benchmark_records_binds_every_coordinate_to_text():
    sources = [{
        "window_id": "E_U0000",
        "domain": "evermembench",
        "tok_start": 7000,
        "tok_end": 8024,
        "history": [{
            "time": "09:00", "speaker": "A",
            "dialogue": "First support. Second support.",
            "topic": 1, "img_url": "not-published",
        }],
        "turns": [{
            "time": "10:00", "speaker": "B",
            "dialogue": "The plan was approved. Work starts now.",
            "message_index": 3,
        }],
    }]
    rcv = [{
        "window_id": "E_U0000", "range": "8k~16k", "gap_idx": 0,
        "sentence_numbers": [0, 1],
        "target_sentence": "The plan was approved. Work starts now.",
        "gap_type": "A", "gap_reason": "The earlier support is missing.",
        "validated": [{
            "source_id": "T0-S2", "sentence": "Second support.",
            "status": "resolve", "validation": 1, "vote_count": 3,
            "votes": {"gemma": 1, "GLM": 1, "Qwen": 1},
        }],
    }]
    salient = [{
        "window_id": "E_U0000",
        "gap_analysis": [{
            "sentence_numbers": [0, 1],
            "sentences": ["The plan was approved.", "Work starts now."],
            "gap_type": "A", "vote_count": 3,
        }],
    }]

    [row] = build_benchmark_records(sources, rcv, salient, ["E_U0000"])

    assert row["schema_version"] == "2.0"
    assert row["source"] == row["domain"] == "evermembench"
    assert row["history"] == [{
        "turn_index": 0, "time": "09:00", "speaker": "A",
        "dialogue": "First support. Second support.",
    }]
    assert [item["sentences"] for item in row["salient_utterances"]] == [
        ["The plan was approved."], ["Work starts now."],
    ]
    assert row["gaps"][0]["target_sentence"] == (
        "The plan was approved. Work starts now."
    )
    assert row["gaps"][0]["evidence"][0]["sentence"] == "Second support."
    assert "img_url" not in json.dumps(row)
```

This fixes the WKF representation: each source salient sentence becomes one `salient_utterances` item with one coordinate and one text, so the array length is the WKF denominator.

- [ ] **Step 2: Add failing coordinate and schema abuse tests**

Parametrize literal mutations and exact error fragments:

```python
@pytest.mark.parametrize(("mutation", "message"), [
    (lambda s, r, a: s.append(dict(s[0])), "duplicate source window_id"),
    (lambda s, r, a: r[0].update(sentence_numbers=[99]), "target coordinate"),
    (lambda s, r, a: r[0].update(target_sentence="wrong"), "target text mismatch"),
    (lambda s, r, a: r[0]["validated"][0].update(source_id="T0-S0"), "evidence coordinate"),
    (lambda s, r, a: r[0]["validated"][0].update(sentence="wrong"), "evidence text mismatch"),
    (lambda s, r, a: a[0]["gap_analysis"][0].update(sentences=["wrong"]), "salient text mismatch"),
    (lambda s, r, a: s[0]["history"][0].update(dialogue=1), "history[0].dialogue"),
])
def test_build_benchmark_records_rejects_unverifiable_text(mutation, message):
    sources, rcv, salient = valid_fixture()
    mutation(sources, rcv, salient)
    with pytest.raises(ValueError, match=re.escape(message)):
        build_benchmark_records(sources, rcv, salient, ["E_U0000"])
```

Also assert missing split source/salient rows, unknown prefixes, inconsistent ranges, duplicate `gap_idx`, duplicate evidence IDs, `validation != 1` evidence filtering, and complete omission of an RCV row whose `target_sentence` is empty/whitespace (including all of that row's evidence).

Add a pipeline-shaped case where `source` exists but `domain` does not, the target is a sentence list, and validated evidence has only `source_id`. Assert the adapter publishes `domain`, a joined target string, and evidence text resolved from `history`.

- [ ] **Step 3: Add a failing pipeline-only salient vote-retention test**

In `benchmark/tests/test_stages.py`, assert `ensemble_salient_votes()` returns the minimum supporting sentence vote count for each retained group:

```python
assert result[0]["gaps"][0]["vote_count"] == 3
```

Expected: FAIL because the current ensemble drops the count.

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=benchmark/src pytest -q -p no:cacheprovider \
  benchmark/tests/test_text_release.py benchmark/tests/test_stages.py -k 'benchmark_records or salient_vote'
```

Expected: FAIL because `text_release.py` and the v2 APIs do not exist, and salient vote counts are absent.

- [ ] **Step 5: Implement exact coordinate resolution and record construction**

Use existing canonical splitters rather than inventing new coordinate semantics:

```python
from .evidence import _history_index
from .salient import expand_window_sentences

def _normalize_turns(value: Any, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    rows = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{location}[{index}] must be an object")
        clean = {"turn_index": index}
        for field in ("time", "speaker", "dialogue"):
            field_value = item.get(field)
            if not isinstance(field_value, str):
                raise ValueError(f"{location}[{index}].{field} must be a string")
            clean[field] = field_value
        rows.append(clean)
    return rows

def _resolve_target(turns: list[dict[str, Any]], numbers: list[int], location: str) -> list[str]:
    expanded = expand_window_sentences(turns)
    if not numbers or numbers != sorted(set(numbers)):
        raise ValueError(f"{location}: target coordinates must be ordered and unique")
    try:
        return [expanded[number]["sentence"] for number in numbers]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"{location}: target coordinate is out of range") from exc

def _resolve_evidence(history: list[dict[str, Any]], source_id: str, location: str) -> str:
    value = _history_index(history).get(source_id)
    if value is None:
        raise ValueError(f"{location}: evidence coordinate is invalid")
    return value
```

Consume each iterable once and retain only rows whose `window_id` is in the split set; never materialize the 5,249-row multi-gigabyte source file. Still detect duplicate selected IDs and reject missing selected rows after the scan.

Resolve the source before constructing the row:

```python
expected_source = _source_for_window(window_id)
domain = raw_source.get("domain", raw_source.get("source"))
if domain != expected_source:
    raise ValueError(f"{window_id}: domain/source does not match window ID")
source = expected_source
```

External source rows have no `range`; derive it from the RCV rows and require every non-empty gap for the window to carry the same valid range. If a pipeline window already has `range`, require it to equal the RCV range. Publish `source == domain == expected_source`, and require integer `tok_start`/`tok_end` with `0 <= tok_start < tok_end`.

Use all RCV rows to establish the window range, then omit every row whose target is an empty/whitespace string before counting or publishing gaps/evidence. This preserves the observed canonical rule: 1,500 selected raw RCV rows become 1,498 published gaps, and the six validated links attached to the two empty targets are excluded so the evidence count is 4,135 rather than 4,141. Preserve the remaining original `gap_idx` values; do not reindex them.

Handle the two target encodings exactly:

```python
resolved = _resolve_target(turns, sentence_numbers, location)
expected_target = " ".join(resolved)
raw_target = rcv_row["target_sentence"]
if isinstance(raw_target, str):
    if raw_target != expected_target:
        raise ValueError(f"{location}: target text mismatch")
elif isinstance(raw_target, list):
    if raw_target != resolved:
        raise ValueError(f"{location}: target text mismatch")
else:
    raise ValueError(f"{location}: target_sentence must be a string or list of strings")
published_target = expected_target
```

For each validated evidence item, resolve `source_id` from normalized history. If the external RCV includes `sentence`, require exact equality. If the pipeline bridge omits `sentence`, publish the resolved text. In both forms, publish only items with `validation == 1`, preserve source order, and reject duplicate `source_id` values.

For salient input, accept external `gap_analysis` or pipeline `gaps`; flatten each `(sentence_number, sentence)` pair in source order into one item. External rows already carry `vote_count`; pipeline rows obtain it from Step 7. Re-resolve every sentence from the current window before publication.

- [ ] **Step 6: Implement strict semantic validation and generated schema**

`validate_benchmark_records()` must require exact key sets and re-resolve all text from the in-record `history` and `turns`. Return:

```python
{
    "windows": len(rows),
    "gaps": gap_count,
    "evidence": evidence_count,
    "salient_utterances": salient_count,
    "by_source": {name: count, ...},
    "by_range": {name: count, ...},
}
```

Require gap types `A/C/L`, salient types `N/C/A/L`, exact vote keys `gemma/GLM/Qwen` with binary integer values, evidence `vote_count == sum(votes) >= 2`, evidence `status == "resolve"`, and `validation == 1`. Salient `vote_count` is an integer in `[1, 3]`. Validate ordered unique target/salient coordinates, unique original `gap_idx` values, and unique evidence IDs within each gap.

`benchmark_schema()` must declare schema `2.0`, exact top-level/gap/evidence/turn/salient properties, `additionalProperties: false` at every level, non-empty text fields, integer bounds, source enum `evermembench/locomo/realtalk`, range enum `8k~16k/16k~32k/32k~64k/64k~128k/128k~160k`, window IDs `^[ELR]_U\d{4,}$`, and evidence IDs `^T(0|[1-9]\d*)-S([1-9]\d*)$`. Permit `gap_reason` only at the gap level; forbid generic reasoning fields by exact-key validation.

- [ ] **Step 7: Retain vote counts in pipeline `ensemble_salient_votes()`**

For every accepted coordinate group, compute:

```python
vote_count = min(len(sentence_votes[coordinate]) for coordinate in coordinates)
accepted.append({
    "gap_idx": len(accepted),
    "target_sentence_numbers": list(coordinates),
    "gap_type": gap_type,
    "vote_count": vote_count,
})
```

This change supports the Task 4 pipeline-derived mirror. The checked-in canonical Task 5 export reads the already-voted `gap_analysis[].vote_count` values from the canonical salient artifact and does not depend on this ensemble change.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run:

```bash
PYTHONPATH=benchmark/src pytest -q -p no:cacheprovider \
  benchmark/tests/test_text_release.py benchmark/tests/test_stages.py
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add benchmark/src/remember_benchmark/text_release.py \
  benchmark/src/remember_benchmark/ensemble.py \
  benchmark/tests/test_text_release.py benchmark/tests/test_stages.py
git commit -m "feat(dataset): bind benchmark coordinates to released text"
```

---

### Task 3: Export deterministic gzip shards, schema, and provenance manifest

**Files:**
- Modify: `benchmark/src/remember_benchmark/io.py`
- Modify: `benchmark/src/remember_benchmark/text_release.py`
- Modify: `benchmark/tests/test_text_release.py`

**Interfaces:**
- Consumes: Task 2 record builders plus input artifact paths.
- Produces:

```python
def iter_jsonl_gzip(path: Path) -> Iterator[dict[str, Any]]: ...
def atomic_write_jsonl_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None: ...

def export_benchmark(
    source_path: Path,
    rcv_path: Path,
    salient_path: Path,
    split_path: Path,
    benchmark_dir: Path,
    *,
    expected_counts: tuple[int, int, int] | None = None,
    expected_salient_count: int | None = None,
    shard_count: int = BENCHMARK_SHARD_COUNT,
    construction: str = "canonical-source-join",
) -> dict[str, Any]: ...

def validate_benchmark_directory(
    benchmark_dir: Path,
    ordered_window_ids: Sequence[str],
    *,
    expected_counts: tuple[int, int, int] | None = None,
    expected_salient_count: int | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]: ...

def main(argv: Sequence[str] | None = None) -> int: ...
```

- [ ] **Step 1: Add failing deterministic-gzip tests**

Test the same rows in two different parents and assert identical compressed bytes:

```python
def test_atomic_gzip_jsonl_is_byte_deterministic(tmp_path):
    rows = [{"window_id": "E_U0000", "text": "한글 text"}]
    left = tmp_path / "left/a.jsonl.gz"
    right = tmp_path / "right/b.jsonl.gz"
    atomic_write_jsonl_gzip(left, rows)
    atomic_write_jsonl_gzip(right, rows)
    assert left.read_bytes() == right.read_bytes()
    assert list(iter_jsonl_gzip(left)) == rows
    with gzip.open(left, "rt", encoding="utf-8") as handle:
        assert handle.read().endswith("\n")
```

- [ ] **Step 2: Add failing shard and manifest tests**

Build 900 fabricated valid rows and assert:

```python
names = [item["path"] for item in manifest["shards"]]
assert names == [
    f"remember_900-{index:05d}-of-00010.jsonl.gz" for index in range(10)
]
assert [item["rows"] for item in manifest["shards"]] == [90] * 10
assert manifest["counts"]["windows"] == 900
assert manifest["authorization"]["redistributable_text"] is True
```

Add mutations for a missing shard, extra shard, corrupt gzip, swapped shard order, changed compressed bytes, changed split hash, schema hash mismatch, invalid source-hash syntax, a source hash that differs from supplied `expected_source_hashes`, and manifest count mismatch. Generic directory validation cannot independently recompute unavailable upstream source files, so it validates source-hash syntax unless the caller supplies expected hashes.

Add an overwrite test proving an existing regular UTF-8 `README.md` survives byte-for-byte, while a symlinked or non-UTF-8 README fails before publication and leaves the old directory intact.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=benchmark/src pytest -q -p no:cacheprovider \
  benchmark/tests/test_text_release.py -k 'gzip or shard or manifest or export_cli'
```

Expected: FAIL because gzip/export/manifest APIs are absent.

- [ ] **Step 4: Implement atomic deterministic gzip I/O**

Use the existing atomic-file/fsync pattern in `io.py`:

```python
with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as raw:
    temporary = Path(raw.name)
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        for row in rows:
            payload = json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            zipped.write(payload + b"\n")
    raw.flush()
    os.fsync(raw.fileno())
os.replace(temporary, path)
fsync_parent_directory(path)
```

Reject blank lines, non-object JSON values, malformed UTF-8/JSON, and trailing non-whitespace bytes in `iter_jsonl_gzip()` with path and physical line number.

- [ ] **Step 5: Implement shard creation and manifest-last publication**

Validate every row before writing and reject `shard_count <= 0`. Divide the ordered row list into exactly `shard_count` balanced contiguous slices: the first `len(rows) % shard_count` slices receive one extra row, and only trailing slices may be empty when `len(rows) < shard_count`. For the canonical 900/10 export, require exactly 90 rows per shard; the one-window pipeline fixture deterministically emits `[1, 0, ..., 0]`. Record this policy in the manifest and enforce it during validation. Write schema and shards first, compute hashes over compressed bytes, then write `manifest.json` last.

The manifest must contain:

```python
{
    "schema_version": "2.0",
    "dataset": "remember_900",
    "sharding": {"count": shard_count, "policy": "balanced-contiguous-v1"},
    "shards": [{"path": name, "rows": count, "sha256": digest}, ...],
    "artifacts": {
        "schema": {"path": "schema.json", "sha256": ...},
        "split": {"path": "../splits/main_900.txt", "sha256": ...},
    },
    "sources": {
        "windows": {"name": source_path.name, "sha256": ...},
        "rcv": {"name": rcv_path.name, "sha256": ...},
        "salient": {"name": salient_path.name, "sha256": ...},
    },
    "provenance": {
        "construction": construction,
        "window_normalization": "time-speaker-dialogue-v1",
        "current_sentence_splitter": "terminal-punctuation-whitespace-v1",
        "history_source_id_splitter": "T-zero-based-S-one-based-v1",
        "target_join": "single-space-v1",
        "record_builder": "remember-benchmark-text-release-v1",
    },
    "coordinate_system": {
        "target_sentence_numbers": "zero-based flattened current-window sentences",
        "source_id": "zero-based T history turn and one-based S sentence",
    },
    "counts": stats,
    "authorization": {
        "redistributable_text": True,
        "scope": ["EverMemBench", "LoCoMo", "RealTalk"],
    },
}
```

Do not record absolute source paths.

- [ ] **Step 6: Implement full-directory validation and CLI**

`validate_benchmark_directory()` must validate the manifest before trusting shard order, require exactly the listed canonical files plus optional `README.md`, hash compressed files before decompression, load all rows, run `validate_benchmark_records()`, and compare every recomputable released-artifact count/hash to the manifest. Require the exact provenance keys and known values above. Validate upstream source digests as lowercase 64-hex strings and compare them when `expected_source_hashes` is supplied.

The CLI arguments are:

```text
--source PATH --rcv PATH --salient PATH --split PATH --benchmark-dir PATH
--expected-windows INT --expected-gaps INT --expected-evidence INT
--expected-salient INT --shard-count INT --overwrite
```

Require the three expected paper counts together; `--expected-salient` is independently optional for fixture builds. Refuse a non-empty output directory without `--overwrite`. Stage in a sibling temporary directory and atomically replace only after validation. On overwrite, preserve an existing `README.md` only after proving it is a regular, non-symlink UTF-8 file; copy it into the staged directory before validation. The pipeline-generated directory has no README.

- [ ] **Step 7: Run the complete text-release suite**

Run:

```bash
PYTHONPATH=benchmark/src pytest -q -p no:cacheprovider benchmark/tests/test_text_release.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add benchmark/src/remember_benchmark/io.py \
  benchmark/src/remember_benchmark/text_release.py \
  benchmark/tests/test_text_release.py
git commit -m "feat(dataset): export deterministic text benchmark shards"
```

---

### Task 4: Publish and resume-verify a schema-compatible pipeline-derived v2 mirror

**Files:**
- Modify: `benchmark/src/remember_benchmark/pipeline.py`
- Modify: `benchmark/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `windows_path`, private flattened RCV, merged salient rows, split, and Task 3 `build_benchmark_records_from_pipeline()` / `export_benchmark()` / `validate_benchmark_directory()`.
- Produces: one atomic release directory containing the unchanged v1 index and a schema-compatible v2 `benchmark/` subtree, with every artifact bound into the private release inventory.
- Provenance boundary: this mirror is marked `construction: pipeline-derived`. It proves that a newly run eight-stage builder emits self-contained records, but it is not claimed byte- or count-identical to the checked-in canonical artifact built from the fixed RCV and salient sources in Task 5.

- [ ] **Step 1: Add a failing complete-publication test**

Extend the fixture pipeline assertion:

```python
build(config, work, out, providers, counter=WordCounter())
assert (out / "annotations/remember_900_annotations.jsonl").is_file()
assert (out / "benchmark/manifest.json").is_file()
assert len(list((out / "benchmark").glob("remember_900-*.jsonl.gz"))) == 10
manifest = json.loads((out / "benchmark/manifest.json").read_text())
assert manifest["counts"]["windows"] == 1
assert manifest["provenance"]["construction"] == "pipeline-derived"
```

The tiny builder fixture may produce empty trailing shards; the checked-in 900-row release must still produce `[90] * 10`.

Also load the single emitted row and assert its actual bridge path has published `domain`, target text, evidence text resolved from `source_id`, and salient text resolved from `target_sentence_numbers`. This is the integration proof for the pipeline-specific source shapes.

- [ ] **Step 2: Add failing resume and rollback tests**

For each mutation below, run `build(..., resume=True)` and expect `ResumeMismatch` without changing a sentinel in the previous public directory:

- delete one shard;
- corrupt gzip bytes;
- alter `manifest.json`;
- add an unexpected `benchmark/private.jsonl`;
- change `windows.jsonl`, private RCV, merged salient, or split hash;
- inject an exporter failure during overwrite and verify the old v1/v2 release and inventory are restored together.

- [ ] **Step 3: Run pipeline tests and verify RED**

Run:

```bash
PYTHONPATH=benchmark/src pytest -q -p no:cacheprovider benchmark/tests/test_pipeline.py
```

Expected: FAIL because the release stage knows only the v1 artifacts.

- [ ] **Step 4: Extend the exact public-tree allowlist**

Generate the shard paths rather than broadly permitting `.gz`:

```python
BENCHMARK_SHARDS = frozenset(
    Path("benchmark") / f"remember_900-{index:05d}-of-00010.jsonl.gz"
    for index in range(10)
)
PUBLIC_FILES = frozenset({
    Path("annotations/remember_900_annotations.jsonl"),
    Path("schema.json"), Path("metadata.json"), Path("splits/main_900.txt"),
    Path("benchmark/schema.json"), Path("benchmark/manifest.json"),
}) | BENCHMARK_SHARDS
PUBLIC_DIRS = frozenset({Path("annotations"), Path("splits"), Path("benchmark")})
```

Allow `benchmark/README.md` only in the checked-in repository validator, not in generated pipeline output.

- [ ] **Step 5: Stage both artifacts through the existing transaction**

Change `_build_staged_release()` to accept source, RCV, and salient paths. Call `export_annotations()` first and the pipeline adapter/exporter second inside the same staging directory with `construction="pipeline-derived"`. Pass the fixture's expected v1 counts to v2 validation; derive the mirror's salient count rather than hard-coding 7,245. The exporter follows the balanced-shard rule, so a one-row fixture has one populated shard followed by nine empty shards.

Extend `_stage_targets()`, `_write_release_inventory()`, and `_verify_release()` to enumerate root v1 files, both benchmark metadata files, and all ten compressed shards. `_verify_release()` must also call `validate_benchmark_directory()` and compare its counts to the inventory.

- [ ] **Step 6: Bind internal release sources to the inventory**

The builder must use its own `windows.jsonl`, private RCV bridge, and `salient/ensemble.jsonl`; it must never refer to historical research-tree paths at runtime. Add their hashes to `release_config` and the release inventory. The standalone Task 3 CLI is the sole canonical publisher for the checked-in artifact from the fixed source/RCV/salient files.

- [ ] **Step 7: Run benchmark pipeline and release tests**

Run:

```bash
PYTHONPATH=benchmark/src pytest -q -p no:cacheprovider \
  benchmark/tests/test_pipeline.py benchmark/tests/test_release.py \
  benchmark/tests/test_text_release.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add benchmark/src/remember_benchmark/pipeline.py benchmark/tests/test_pipeline.py
git commit -m "feat(benchmark): publish text shards transactionally"
```

---

### Task 5: Generate and validate the real 900-window benchmark artifacts

**Files:**
- Create: `dataset/benchmark/schema.json`
- Create: `dataset/benchmark/manifest.json`
- Create: `dataset/benchmark/remember_900-00000-of-00010.jsonl.gz` through `remember_900-00009-of-00010.jsonl.gz`
- Modify: `benchmark/tests/test_published_dataset.py`

**Interfaces:**
- Consumes: the canonical read-only source/RCV/salient artifacts and Task 3 CLI.
- Produces: checked-in v2 payload with fixed hashes/counts, while the v1 index bytes remain unchanged.

- [ ] **Step 1: Add failing published-artifact tests**

Add these assertions without changing existing v1 tests:

```python
def test_published_text_benchmark_has_exact_paper_population():
    split_ids = read_ids(DATASET / "splits/main_900.txt")
    stats = validate_benchmark_directory(
        DATASET / "benchmark", split_ids,
        expected_counts=(900, 1498, 4135),
        expected_salient_count=7245,
    )
    assert stats["windows"] == 900
    assert stats["salient_utterances"] == 7245

def test_published_text_benchmark_is_ten_ordered_90_row_shards():
    manifest = json.loads((DATASET / "benchmark/manifest.json").read_text())
    assert [item["rows"] for item in manifest["shards"]] == [90] * 10
    assert manifest["authorization"]["redistributable_text"] is True
```

Also assert that the current v1 annotation/schema/metadata SHA-256 values do not change.

Use literal regression assertions, not values recomputed at test setup:

```python
assert sha256(DATASET / "annotations/remember_900_annotations.jsonl") == (
    "a921ae399bd646d469a6824db4d5421358aee387dd1124ff2f1ee2f8c0a8ab3a"
)
assert sha256(DATASET / "schema.json") == (
    "2bea6dcd4446fd7f3c980f8d9c8a50d6ffd7757b097ca6efff4db7912999e5c2"
)
assert sha256(DATASET / "metadata.json") == (
    "ddf8387a312743e433a6d4493e064e972b6dc6f93f67ce6251dfb39b591f493c"
)
```

Pin the canonical manifest's source provenance to the observed artifact hashes:

```python
EXPECTED_SOURCE_HASHES = {
    "windows": "867534724b6c73800e815f8e5d8e0c5426a39662c210c143fd7bece655cfe4ec",
    "rcv": "3414a101cf9f99c76a5a2404da6296543d037de1871395334d0bb7a19ee8a41e",
    "salient": "e65ed00720cb8bd6ce7e576f0e8f53c966423625da5ca6a8017158762994ebf2",
}
```

Pass this mapping to `validate_benchmark_directory()` in published-data tests and the release validator.

- [ ] **Step 2: Run the published test and verify RED**

Run:

```bash
PYTHONPATH=benchmark/src pytest -q -p no:cacheprovider \
  benchmark/tests/test_published_dataset.py -k text_benchmark
```

Expected: FAIL because `dataset/benchmark/` does not exist.

- [ ] **Step 3: Materialize the real benchmark with explicit sources**

From `3-release/`, run:

```bash
PYTHONPATH=benchmark/src python -m remember_benchmark.text_release \
  --source ../0-dataset/combined_sampled.jsonl \
  --rcv ../1-research/0/outputs/rcv/ensemble/all/relevant_chunk_validation.filtered.jsonl \
  --salient ../1-research/0/outputs/salient/ensemble/5/salient.jsonl \
  --split dataset/splits/main_900.txt \
  --benchmark-dir dataset/benchmark \
  --expected-windows 900 \
  --expected-gaps 1498 \
  --expected-evidence 4135 \
  --expected-salient 7245 \
  --shard-count 10
```

Do not use `--overwrite` on the first build. On regeneration, resolve and inspect the target directory first, then use the exporter's atomic `--overwrite` path; an existing validated `README.md` is preserved.

- [ ] **Step 4: Verify generated file format and size**

Run:

```bash
python -m json.tool dataset/benchmark/schema.json
python -m json.tool dataset/benchmark/manifest.json
gzip -t dataset/benchmark/remember_900-00000-of-00010.jsonl.gz \
  dataset/benchmark/remember_900-00001-of-00010.jsonl.gz \
  dataset/benchmark/remember_900-00002-of-00010.jsonl.gz \
  dataset/benchmark/remember_900-00003-of-00010.jsonl.gz \
  dataset/benchmark/remember_900-00004-of-00010.jsonl.gz \
  dataset/benchmark/remember_900-00005-of-00010.jsonl.gz \
  dataset/benchmark/remember_900-00006-of-00010.jsonl.gz \
  dataset/benchmark/remember_900-00007-of-00010.jsonl.gz \
  dataset/benchmark/remember_900-00008-of-00010.jsonl.gz \
  dataset/benchmark/remember_900-00009-of-00010.jsonl.gz
find dataset/benchmark -name '*.jsonl.gz' -size +20M -print
```

Expected: both JSON files parse; `gzip -t` exits 0; `find` prints nothing.

- [ ] **Step 5: Run published-data and source-to-export validation**

Run:

```bash
PYTHONPATH=benchmark/src pytest -q -p no:cacheprovider \
  benchmark/tests/test_published_dataset.py benchmark/tests/test_text_release.py
```

Expected: PASS with exact 900/1,498/4,135/7,245 counts and unchanged v1 hashes.

- [ ] **Step 6: Commit**

```bash
git add dataset/benchmark benchmark/tests/test_published_dataset.py
git commit -m "data: publish self-contained ReMEMber benchmark"
```

---

### Task 6: Add exact prompts, strict inputs, renderers, and canonical parsers to one evaluator file

**Files:**
- Create: `evaluation/evaluate.py`
- Create: `evaluation/prompts/memory_recall.txt`
- Create: `evaluation/prompts/mkf.txt`
- Create: `evaluation/prompts/wkf.txt`
- Create: `evaluation/prompts/atomic_facts.txt`
- Create: `evaluation/prompts/factuality.txt`
- Create: `evaluation/requirements.txt`
- Create: `evaluation/tests/test_evaluate.py`
- Create: `evaluation/tests/fixtures/predictions.jsonl`

**Interfaces:**
- Consumes: Task 5 benchmark directory or a standalone `.jsonl` / `.jsonl.gz`, generic predictions, and five prompt files.
- Produces these testable names in `evaluation/evaluate.py`:

```python
class InputError(ValueError): ...
class ResumeError(RuntimeError): ...

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

def load_benchmark(path: Path) -> BenchmarkInput: ...
def load_predictions(
    path: Path,
    benchmark_ids: Sequence[str],
    *,
    allow_partial: bool,
) -> dict[str, dict[str, str]]: ...
def render_structured_memory(value: Mapping[str, Any]) -> str: ...
def load_prompts(prompt_dir: Path) -> dict[str, str]: ...
def validate_placeholder_set(template: str, expected: set[str]) -> None: ...
def fill_prompt(template: str, **values: object) -> str: ...
def split_summary_sentences(summary: str) -> list[str]: ...
def parse_mr(raw: str) -> list[dict[str, Any]] | None: ...
def parse_coverage(raw: str) -> list[dict[str, Any]] | None: ...
def parse_atomic(raw: str) -> list[dict[str, Any]] | None: ...
def parse_factuality(raw: str) -> list[dict[str, Any]] | None: ...
def normalize_mr(
    items: Sequence[Mapping[str, Any]] | None,
    expected_gap_ids: Sequence[str],
) -> list[dict[str, Any]] | None: ...
def normalize_coverage(
    items: Sequence[Mapping[str, Any]] | None,
    expected_ids: Sequence[str],
    *,
    id_field: str,
) -> list[dict[str, Any]] | None: ...
def normalize_atomic_facts(
    parsed: Sequence[Mapping[str, Any]] | None,
    summary_sentences: Sequence[str],
) -> list[dict[str, Any]]: ...
def normalize_factuality(
    parsed: Sequence[Mapping[str, Any]] | None,
    facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]: ...
```

- [ ] **Step 1: Add failing prompt hash and placeholder tests**

Hard-code the five approved hashes in the test; do not import research files at runtime:

```python
EXPECTED_PROMPT_HASHES = {
    "memory_recall": "1bc6228fa148c6b1411a6c460c13682100e976ac9e9bb9f4da4ce3fcd8085874",
    "mkf": "75aea6ec084a1c9016c914ecca915198487254dbdbe7a0c2f9b3708af397342c",
    "wkf": "c4fc5c3f91e1d13208bd6b4972c9754992e042125814800a62a792cbafe976bd",
    "atomic_facts": "db406d5eafeb0dd51f813cf45d5b22aed57e11bdedf493748b695c2172cc1676",
    "factuality": "f4613f72b2355cb96c53c10b8340e02ffa1e203b9797acb0489283b88df1a1f2",
}

def test_shipped_prompts_have_exact_paper_hashes():
    prompts = load_prompts(PROMPT_DIR)
    assert {name: sha256_text(value) for name, value in prompts.items()} == EXPECTED_PROMPT_HASHES
```

Assert exact placeholder sets:

- MR: `window`, `memory`, `gaps`;
- MKF: `N`, `summary`, `M`, `gap_cases`;
- WKF: `N`, `summary`, `M`, `salient_utterances`;
- atomic: `N`, `summary`;
- factuality: `transcript`, `N`, `summary`.

Escaped `{{` / `}}` example JSON must not count as placeholders. Missing or extra live placeholders are errors.

Test both sides of the call contract: omitting a required value and supplying an unexpected value to `fill_prompt()` must fail before rendering.

- [ ] **Step 2: Add failing benchmark and prediction loader tests**

Cover:

- a manifest-ordered shard directory;
- one plain JSONL and one gzip JSONL;
- invalid schema version, duplicate benchmark ID, duplicate prediction ID, unknown ID, blank summary, blank memory, and missing full-run IDs;
- `--allow-partial` accepting only a non-empty unique subset;
- generic `memory_text` taking precedence;
- current ReMEMber output using `memory["text"]`;
- fallback over ordered `memory["items"][].text`;
- rejection of unsupported structured memory rather than JSON-stringifying it.
- a 900-row standalone file whose ordered-ID hash differs from the canonical split hash being rejected without `--allow-partial`;
- the canonical split hash `0eef55c9307dbb8155f664a860503760eb9314ba2594383996a48728bdb961ee` being retained in `BenchmarkInput` and the run manifest.

The expected adapter test is:

```python
assert render_structured_memory({"text": "packed memory", "items": []}) == "packed memory"
assert render_structured_memory({
    "text": "", "items": [{"text": "first"}, {"text": "second"}],
}) == "first\nsecond"
```

- [ ] **Step 3: Add failing renderer and parser compatibility tests**

Assert these canonical quirks with literal raw strings:

- MR uses IDs `gap_<gap_idx>` and treats only `1` / `COVERED` case-insensitively as positive; `PARTIAL` and `0.5` are zero.
- MR accepts full JSON, `{"results": [...]}`, fenced JSONL objects, repaired trailing commas, and complete-object salvage from truncation.
- MKF/WKF strip completed `<think>`, inspect balanced arrays from last to first, and do not apply trailing-comma repair.
- MKF/WKF use `S1...`, gaps `G0...`, and salients `U0...`.
- Atomic/factuality use `S0...`; factuality strips unterminated `<think>`, repairs trailing commas, and salvages complete objects.
- Unknown factuality category becomes `other error`; aliases `noerror`, `no_error`, `factual`, `none`, and `no-error` become `no error`.
- The nine fixed categories, in output order, are `no error`, `out-of-context error`, `entity error`, `predicate error`, `circumstantial error`, `grammatical error`, `coreference error`, `linking error`, and `other error`.
- Atomic parse failure or an empty/unusable atomic array yields one `S{i}-F0` fact per summary sentence, preserving the established evaluator behavior even though the prompt permits `[]`.
- Any duplicate, missing, unexpected, or out-of-range expected item invalidates the whole judgment (`None`), so later scoring assigns zeros to all expected items.

Assert renderer-to-record mappings, not only labels:

- MR current-window lines are `[time] speaker: dialogue`; each gap becomes `gap_<gap_idx>` with target text, type, ordered `{source_id,text}` evidence, and `gap_reason` as `supporting_reasoning`.
- MKF summary lines are `S1...`; prompt gaps are `G0...` in released gap-array order, and each `G#` maps back to that row's original `gap_idx` for per-window output and MR-gated joining. Evidence sentences are newline-joined in released order.
- WKF summary lines are `S1...`; flattened salient items are `U0...` in released order without deduplication.
- Atomic input sentences are `S0...`; checker input facts are numbered `1...N`, and returned checker `sentence` echoes must map exactly once to the ordered atomic-fact text.

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
pytest -q -p no:cacheprovider evaluation/tests/test_evaluate.py \
  -k 'prompt or benchmark or prediction or render or parse or normalize'
```

Expected: FAIL because evaluator assets and functions do not exist.

- [ ] **Step 5: Copy the exact prompt bytes and verify hashes**

Using `apply_patch`, create each prompt with the byte-for-byte content of its read-only canonical source:

```text
../1-research/4/memory_recall_prompt_v5.txt -> evaluation/prompts/memory_recall.txt
../1-research/5/prompt_mkf_v5.txt           -> evaluation/prompts/mkf.txt
../1-research/5/prompt_wkf_v3.txt           -> evaluation/prompts/wkf.txt
../1-research/5/prompt_atomic_facts.txt      -> evaluation/prompts/atomic_facts.txt
../1-research/5/prompt_factuality.txt        -> evaluation/prompts/factuality.txt
```

Then run `sha256sum evaluation/prompts/*.txt` and compare to the Global Constraints.

- [ ] **Step 6: Implement stdlib-first loaders and prompt renderers**

`load_benchmark()` must use only code contained in `evaluate.py` for directory, plain JSONL, and gzip JSONL inputs. It must not import `remember_benchmark` or any research-tree module. For a manifest directory, validate the manifest, schema, sibling split, ordered shard hashes, and every record; return all provenance in `BenchmarkInput`, with `input_sha256` computed from canonical JSON over the ordered relative-path/file-hash pairs. For a standalone file, validate every v2 record, preserve file order, and set `input_sha256` to the exact file-byte hash. A non-partial official run requires both 900 rows and the canonical ordered-ID digest derived from the fixed split; `--allow-partial` is required for every other non-empty population.

`validate_placeholder_set()` extracts only live single-brace fields and requires the exact approved set for that prompt. `fill_prompt()` first requires `set(values)` to equal that validated set, then replaces known placeholders and unescapes doubled braces:

```python
def fill_prompt(template: str, **values: object) -> str:
    expected = extract_live_placeholders(template)
    if set(values) != expected:
        raise InputError("prompt values do not match template placeholders")
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered.replace("{{", "{").replace("}}", "}")
```

`split_summary_sentences()` must use `nltk.tokenize.sent_tokenize` for canonical runs. Import NLTK lazily and raise an actionable error naming `pip install -r evaluation/requirements.txt` and `python -m nltk.downloader punkt_tab` if unavailable. A punctuation-free non-empty one-sentence fast path may return `[summary.strip()]` before importing NLTK so the release fake smoke remains dependency-light.

- [ ] **Step 7: Implement canonical rendering and parsers**

Port only the final behavior described in Step 3. Do not import `1-research` or copy historical method/path logic. Factuality evidence must be exactly:

```text
[MEMORY]
<memory_text>

[CURRENT WINDOW]
[time] speaker: dialogue
...
```

MR `supporting_reasoning` uses the released gap's `gap_reason`, matching `load_pw_from_rcv()`. This field is the approved concise gold gap criterion. Never copy evidence-validator `reasoning`, salient per-model `reasons`, or raw chain-of-thought into prompts or benchmark rows.

- [ ] **Step 8: Add `evaluation/requirements.txt`**

Use:

```text
nltk==3.9.1
openai>=1,<2
```

Document vLLM as an optional environment-specific installation rather than pinning it here.

- [ ] **Step 9: Run focused evaluator tests and verify GREEN**

Run:

```bash
pytest -q -p no:cacheprovider evaluation/tests/test_evaluate.py \
  -k 'prompt or benchmark or prediction or render or parse or normalize'
python -m py_compile evaluation/evaluate.py
```

Expected: PASS; compilation exits 0.

- [ ] **Step 10: Commit**

```bash
git add evaluation/evaluate.py evaluation/prompts evaluation/requirements.txt \
  evaluation/tests/test_evaluate.py evaluation/tests/fixtures/predictions.jsonl
git commit -m "feat(evaluation): add canonical inputs prompts and parsers"
```

---

### Task 7: Execute all metrics through lazy fake, vLLM, and OpenAI-compatible providers

**Files:**
- Modify: `evaluation/evaluate.py`
- Modify: `evaluation/tests/test_evaluate.py`

**Interfaces:**
- Consumes: Task 6 normalized benchmark/predictions/prompts.
- Produces:

```python
@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    retry_count: int = 0
    attempt_errors: tuple[str, ...] = ()
    error: str | None = None

class JudgeProvider(Protocol):
    def complete_batch(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
    ) -> list[Completion]: ...

def build_provider(args: argparse.Namespace) -> JudgeProvider: ...
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...
def score_run(per_window: Sequence[dict[str, Any]]) -> dict[str, Any]: ...
```

- [ ] **Step 1: Add failing fake-provider lifecycle tests**

Use a two-window fabricated benchmark where some supporting text is present verbatim and some is absent. Assert the fake provider:

- makes positive/negative MR decisions from supporting text in memory;
- makes MKF/WKF line attributions from exact fixture text in summary;
- emits one atomic fact per fixture summary line;
- marks facts present in memory/current window as `no error` and absent facts as `out-of-context error`;
- leaves `CUDA_VISIBLE_DEVICES` unchanged;
- does not import `vllm` or `openai` and never reads an API key.

Fake behavior is an integration oracle only and must be labeled non-scientific in outputs/docs.

- [ ] **Step 2: Add failing provider-adapter tests with isolated fake modules**

For OpenAI, inject a minimal fake `openai.OpenAI` module and assert:

```python
assert request.model == "Qwen/Qwen3.6-27B"
assert request.temperature == 0.0
assert request.messages == [{"role": "user", "content": "prompt"}]
assert request.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
assert completion.prompt_tokens == 12
assert completion.completion_tokens == 3
```

Set `--api-key-env TEST_JUDGE_KEY`; assert that missing `TEST_JUDGE_KEY` fails before constructing the client, that no other credential variable is read, and that transient errors retry no more than `max_retries`. The OpenAI-compatible request always sends the no-thinking `extra_body` shown above; an endpoint that rejects it fails with an actionable compatibility error instead of silently enabling reasoning.

For vLLM, inject fake `LLM` and `SamplingParams` objects. Assert explicit `--gpu-devices`, lazy import after visibility is set, batched `llm.generate`, `temperature=0.0`, and chat-template `enable_thinking=False` with a `TypeError` compatibility fallback.

- [ ] **Step 3: Add failing exact metric arithmetic tests**

Use literal per-window values to assert:

```python
assert scores["mr"]["micro"] == {
    "numerator": 2, "denominator": 3, "value": pytest.approx(2 / 3),
    "aggregation": "micro", "parse_failure_count": 1,
}
assert scores["conciseness"]["micro"]["numerator"] == 2
assert scores["conciseness"]["micro"]["denominator"] == 4
assert scores["mr_gated_mkf"]["diagnostic"] is True
```

Cover macro means only over windows with nonzero denominators, `windows_with_items`, `windows_without_items`, all nine factuality category counts, union/deduplication of positive MKF/WKF `S#` lines, and strict MR/MKF join on `(window_id, gap_idx)`.

- [ ] **Step 4: Run provider and metric tests and verify RED**

Run:

```bash
pytest -q -p no:cacheprovider evaluation/tests/test_evaluate.py \
  -k 'provider or fake or metric or aggregate or conciseness or gated'
```

Expected: FAIL because providers, orchestration, and aggregation are absent.

- [ ] **Step 5: Implement lazy batch providers**

Do not initialize providers in module import. `build_provider()` is called only after inputs, prompt hashes/placeholders, metric dependencies, and output policy pass.

Metric max-token defaults in `evaluate.py` are:

```python
METRIC_MAX_TOKENS = {
    "mr": 2048,
    "mkf": 4096,
    "wkf": 4096,
    "atomic": 2048,
    "factuality": 2048,
}
```

For `vllm`, expose `--max-model-len 25000`, `--gpu-memory-utilization 0.8`, `--tensor-parallel-size 1`, and `--batch-size 64` defaults. No accelerator ID has a default.

- [ ] **Step 6: Implement phased evaluation and raw audit rows**

Run phases MR, MKF, WKF, atomic, then factuality. Conciseness is deterministic from MKF/WKF, and MR-gated MKF is deterministic from MR/MKF; requesting either adds its prerequisites to the inference metric dependency closure.

Each provider call produces a raw audit record:

```python
{
    "request_id": f"{metric}:{phase}:{window_id}",
    "metric": metric,
    "phase": phase,
    "window_id": window_id,
    "prompt_sha256": sha256_text(prompt),
    "normalized_input_sha256": normalized_input_sha256,
    "expected_item_ids": expected_item_ids,
    "raw_output": completion.text,
    "completed": True,
    "parse_ok": parse_ok,
    "expected_items": expected_count,
    "returned_items": returned_count,
    "prompt_tokens": completion.prompt_tokens,
    "completion_tokens": completion.completion_tokens,
    "retry_count": completion.retry_count,
    "attempt_errors": completion.attempt_errors,
    "error": None,
}
```

Providers always return exactly one `Completion` per input prompt. Successful entries have `error=None`; an entry that exhausts retries has empty text plus its final error and bounded `attempt_errors`. This lets an OpenAI-compatible sequential batch retain earlier successful responses even when a later request fails, and lets a vLLM batch report every affected request. `request_id` is unique and is the resume cache key. Do not store full prompts. Checkpoint every returned entry; if any entry has an error, mark that request `completed=false` and abort only after the whole returned batch has been checkpointed. Such a row is audit state but is never reusable; resume replaces it when the request succeeds and carries prior messages in `attempt_errors`. A semantic parse failure has `completed=true`, `error=null`, and `parse_ok=false`, so it is reusable and scores zero for every expected item.

- [ ] **Step 7: Implement all aggregate results**

Every macro/micro object contains `numerator`, `denominator`, `value`, `aggregation`, `parse_failure_count`, and population counts. For macro, `numerator` is the sum of included per-window ratios and `denominator` is `windows_with_items`; document this explicitly in `scores.json` metadata. MR/MKF/WKF parse failures count failed per-window judge requests. Factuality reports `parse_failures_by_phase` for atomic and checker requests and uses their sum as `parse_failure_count`. Conciseness uses the sum of unique failed MKF/WKF prerequisite requests; MR-gated MKF uses unique failed MR/MKF prerequisite requests.

- [ ] **Step 8: Run provider and metric tests and verify GREEN**

Run:

```bash
pytest -q -p no:cacheprovider evaluation/tests/test_evaluate.py \
  -k 'provider or fake or metric or aggregate or conciseness or gated'
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add evaluation/evaluate.py evaluation/tests/test_evaluate.py
git commit -m "feat(evaluation): run and aggregate paper metrics"
```

---

### Task 8: Add atomic outputs, provenance-safe resume, and the one-command CLI

**Files:**
- Modify: `evaluation/evaluate.py`
- Modify: `evaluation/tests/test_evaluate.py`
- Create: `evaluation/tests/fixtures/golden_scores.json`

**Interfaces:**
- Consumes: Task 7 per-window/raw results.
- Produces:

```python
def build_parser() -> argparse.ArgumentParser: ...
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
) -> dict[str, Any]: ...
def validate_resume_manifest(
    existing: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None: ...
def write_outputs_atomically(
    work_dir: Path,
    output_dir: Path,
    *,
    scores: Mapping[str, Any],
    scores_csv: str,
    per_window: Sequence[Mapping[str, Any]],
    raw_judgments: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None: ...
def execute(
    args: argparse.Namespace,
    *,
    provider_factory: Callable[[argparse.Namespace], JudgeProvider] = build_provider,
) -> int: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

- [ ] **Step 1: Add a failing end-to-end subprocess test**

Run a two-window fixture through:

```bash
python evaluation/evaluate.py \
  --benchmark <fixture.jsonl> \
  --predictions evaluation/tests/fixtures/predictions.jsonl \
  --output-dir <tmp/results> \
  --provider fake \
  --allow-partial
```

Assert exactly these final files:

```text
scores.json
scores.csv
per_window.jsonl
raw_judgments.jsonl
run_manifest.json
```

Assert `scores.json` equals `golden_scores.json`, CSV headers are stable, per-window rows follow benchmark order, raw rows contain no full prompt, and stdout contains rows for all six reported metrics: MR, MKF, WKF, factuality, conciseness, and diagnostic MR-gated MKF.

- [ ] **Step 2: Add failing output-policy and resume tests**

Cover:

- existing non-empty output rejects a normal run;
- `--resume` and `--overwrite` are mutually exclusive;
- identical resume reuses all completed `request_id = metric:phase:window_id` raw records using a counting provider with zero new calls;
- changed benchmark, prediction, prompt, model, provider, temperature, max tokens, selected metrics, partial/full population, or evaluated ID set raises `ResumeError` before provider construction;
- malformed/incomplete raw state is not reused;
- an injected failure during finalization leaves the previous complete five-file result intact;
- `--overwrite` atomically replaces a complete result.

- [ ] **Step 3: Run CLI/output/resume tests and verify RED**

Run:

```bash
pytest -q -p no:cacheprovider evaluation/tests/test_evaluate.py \
  -k 'end_to_end or cli or output or resume or manifest'
```

Expected: FAIL because CLI/finalization/resume behavior is absent.

- [ ] **Step 4: Implement the exact CLI contract**

Arguments:

```text
--benchmark PATH                 required
--predictions PATH               required
--output-dir PATH                required
--provider {fake,vllm,openai}    default vllm
--model MODEL                    default Qwen/Qwen3.6-27B
--metrics LIST                   default all; comma-separated names from
                                 mr,mkf,wkf,factuality,conciseness,mr_gated_mkf
--prompt-dir PATH                default evaluation/prompts
--temperature FLOAT              default 0.0
--mr-max-tokens INT              default 2048
--coverage-max-tokens INT        default 4096
--atomic-max-tokens INT          default 2048
--factuality-max-tokens INT      default 2048
--batch-size INT                 default 64
--gpu-devices IDS                required only for vllm
--max-model-len INT              default 25000
--gpu-memory-utilization FLOAT   default 0.8
--tensor-parallel-size INT       default 1
--base-url URL                   required only for openai
--api-key-env NAME               default OPENAI_API_KEY
--max-retries INT                default 2
--allow-partial
--resume | --overwrite
```

Validate every scalar range and provider-specific requirement before `provider_factory()`: metric names are unique and `all` cannot be combined; token limits/batch/model length/tensor parallelism are positive; temperature and retries are non-negative; GPU utilization is in `(0, 1]`; GPU IDs are a non-empty comma-separated list of unique non-negative integers for vLLM; tensor parallelism cannot exceed the visible-ID count; and OpenAI requires a non-empty base URL plus a syntactically valid environment-variable name.

- [ ] **Step 5: Implement provenance and resumable state**

At run start, atomically write `run_manifest.json` with `status: in_progress` and immutable hashes from `BenchmarkInput` for benchmark shards/files/schema/manifest/split, plus predictions, each prompt, provider/model/options, sentence splitter, requested/dependency-expanded metrics, population, and ordered evaluated-ID hash. Record booleans `canonical_benchmark`, `canonical_prompts`, `paper_model_temperature`, and `scientific_provider`; derive `official_preset` only when all four are true. Custom prompt directories remain usable but are visibly non-official rather than silently presented as the paper preset; fake is always non-scientific.

Maintain the raw checkpoint as a mapping keyed by unique `request_id`. After each completed or exhausted request, serialize all rows in canonical phase/window order to a sibling temporary JSONL, fsync it, and `os.replace` it; atomically update the in-progress manifest checkpoint hash the same way. Never append directly. On resume, validate the immutable manifest section and reuse only rows with `completed=true`, `error=null`, a unique request ID, the same metric/phase/window, rendered prompt hash, normalized-input hash, exact ordered `expected_item_ids`, and expected count. Malformed, incomplete, or exhausted rows are retained for audit but reissued.

On success, render deterministic `scores.csv` bytes from `scores`, build all five final artifacts in a sibling staging directory, reload and validate them there, and atomically swap the whole directory into place with backup restoration on failure. For a new or resumed run, the current in-progress directory remains recoverable until the staged final directory validates. For `--overwrite`, all work happens in a hidden sibling so the previous complete five-file result remains untouched until the final swap.

- [ ] **Step 6: Validate finalized artifacts before printing scores**

Reload all JSON/JSONL/CSV outputs, verify canonical window order and manifest hashes, then print the concise score table. Do not print a success table before this check passes.

- [ ] **Step 7: Run the complete evaluator suite**

Run:

```bash
pytest -q -p no:cacheprovider evaluation/tests/test_evaluate.py
python -m py_compile evaluation/evaluate.py
```

Expected: PASS; compilation exits 0.

- [ ] **Step 8: Commit**

```bash
git add evaluation/evaluate.py evaluation/tests/test_evaluate.py \
  evaluation/tests/fixtures/golden_scores.json
git commit -m "feat(evaluation): add resumable one-command scoring"
```

---

### Task 9: Integrate release validation, evaluator smoke, and public documentation

**Files:**
- Modify: `scripts/validate_release.py`
- Modify: `tests/test_release_layout.py`
- Modify: `README.md`
- Modify: `benchmark/README.md`
- Modify: `dataset/README.md`
- Modify: `dataset/DATASET_CARD.md`
- Create: `dataset/benchmark/README.md`
- Create: `evaluation/README.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: one `python scripts/validate_release.py` result that validates v1 compatibility, v2 shards, evaluator compilation/tests, existing package tests/smokes, and a CPU-only fake evaluation smoke.

- [ ] **Step 1: Add failing path-scoped release-policy tests**

Update/add tests asserting:

- the allowed product directories are `dataset`, `benchmark`, `remember`, and `evaluation`;
- `evaluation/evaluate.py` is the only non-test Python file anywhere under `evaluation/`;
- only the exact ten `dataset/benchmark/remember_900-*.jsonl.gz` paths may use `.gz`;
- a gzip elsewhere remains forbidden;
- v1 annotations still reject `history`, `turns`, `target_sentence`, evidence `sentence`, and other text keys;
- v2 shards accept required text keys but reject media/path/URL/raw-output/reasoning keys;
- malformed gzip, JSON, schema, order, count, or hash produces a precise dataset-integrity failure;
- an unlisted extra shard or payload is rejected;
- the manifest-fixed salient denominator is 7,245.

Replace repeated `shutil.copytree(ROOT / "dataset", ...)` calls in mutation-only layout tests with a helper that builds a tiny valid v1/v2 dataset through the release exporters. Keep one clean-root test against the checked-in 900-row payload. This prevents every scanner test from copying roughly 77 MB of shards while preserving semantic coverage.

- [ ] **Step 2: Add a failing fake evaluator smoke test**

Make the validator build a tiny temporary v2 plain JSONL and prediction file, then run:

```text
python evaluation/evaluate.py --benchmark ... --predictions ...
  --output-dir ... --provider fake --allow-partial
```

Assert exactly five artifacts, `status=complete`, and all six scores: MR, MKF, WKF, factuality, conciseness, and MR-gated MKF. Use punctuation-free one-sentence summaries so the smoke does not download NLTK resources. Prepend temporary `vllm.py` and `openai.py` sentinel modules that raise if imported, remove the configured API-key variable from the child environment, and assert the process environment does not gain `CUDA_VISIBLE_DEVICES`.

- [ ] **Step 3: Run release tests and verify RED**

Run:

```bash
PYTHONPATH=benchmark/src:remember/src pytest -q -p no:cacheprovider \
  tests/test_release_layout.py benchmark/tests/test_published_dataset.py
```

Expected: FAIL because current validation forbids evaluation, gzip, and text-bearing keys.

- [ ] **Step 4: Implement narrowly scoped layout and payload rules**

Add `evaluation` to `ALLOWED_TOP`. Introduce one `_classify_release_path()` / `_is_canonical_benchmark_shard()` helper shared by layout, source, and payload scans. Keep `.gz` in the globally forbidden suffix set, but bypass that single failure only when the release-relative path is one of the exact ten manifest-named canonical shards under `dataset/benchmark/`. `_source_scan()` must skip raw UTF-8 text scanning only for those classified binary shards; every other non-UTF-8 asset remains a failure.

Split `_payload_scan()` policy:

```python
if relative.parts[:1] == ("annotations",):
    enforce_v1_text_free_keys(value)
elif relative.parts[:1] == ("benchmark",):
    reject_forbidden_v2_keys(value, {
        "media", "media_url", "url", "urls", "raw_output", "reasoning", "reasons"
    })
```

Call `validate_benchmark_directory(dataset / "benchmark", split_ids, expected_source_hashes=...)` once from `_dataset_summary()`, not once per payload file, and let it parse/validate the compressed records. The payload scan handles v1 JSON keys and the benchmark directory's JSON metadata only. Do not scan URL-looking substrings inside legitimate `dialogue` text; reject only forbidden structural keys and extra schema fields. `gap_reason` is permitted only because the v2 exact schema fixes its location; generic `reasoning` and per-model `reasons` remain forbidden.

- [ ] **Step 5: Expand dataset summaries and validation commands**

Return:

```python
{
    "counts": benchmark_counts_without_salient,
    "hashes": index_hashes,
    "index": {"counts": ..., "hashes": ...},
    "benchmark": {"counts": ..., "hashes": ..., "shards": ...},
}
```

Retain top-level `counts` and `hashes` during the compatibility migration because existing tests and `main()` read them. The release's final `ok` calculation must separately require exact v1 and v2 900/1,498/4,135 counts, v2 salient count 7,245, the fixed canonical source hashes from Task 5, all package/evaluator tests, and all three smokes (`benchmark`, `remember`, `evaluation`). Compile `evaluation/evaluate.py` along with existing sources.

- [ ] **Step 6: Write public documentation with exact runnable commands**

Documentation must include:

- text shards are canonical; v1 annotations are a lightweight index;
- redistribution authorization for all three sources;
- prediction schema `window_id`, `summary`, `memory_text` and ReMEMber `memory.text` adaptation;
- direct full-run vLLM and OpenAI-compatible commands;
- explicit GPU IDs only in operator examples, never implicit startup;
- fake partial smoke command and warning that fake scores are not scientific;
- MR/MKF/WKF/factuality/conciseness/MR-gated MKF formulas and denominators;
- parse failures score zero and are reported;
- output file definitions, resume/overwrite behavior, prompt hashes/model preset;
- deterministic benchmark regeneration command using neutral `PATH/TO/windows.jsonl`, `PATH/TO/rcv.jsonl`, and `PATH/TO/salient.jsonl` placeholders; public docs must not depend on historical research-tree directory names;
- NLTK install plus `punkt_tab` resource setup;
- exact 900/1,498/4,135/7,245 counts and coordinate bases.

- [ ] **Step 7: Run focused validation and fix only observed failures**

Run:

```bash
PYTHONPATH=benchmark/src:remember/src pytest -q -p no:cacheprovider \
  tests/test_release_layout.py benchmark/tests/test_published_dataset.py \
  evaluation/tests/test_evaluate.py
```

Expected: PASS.

- [ ] **Step 8: Run the full release suite**

Run:

```bash
PYTHONPATH=benchmark/src:remember/src pytest -q -p no:cacheprovider \
  benchmark/tests remember/tests evaluation/tests tests
```

Expected: all tests pass with zero failures.

- [ ] **Step 9: Run syntax, artifact, and end-to-end release validation**

Run:

```bash
python -m py_compile evaluation/evaluate.py \
  benchmark/src/remember_benchmark/text_release.py \
  benchmark/src/remember_benchmark/pipeline.py \
  scripts/validate_release.py
python scripts/validate_release.py
git status --short
git diff --check
```

Expected: compilation and release validation exit 0; `git diff --check` prints nothing. Inspect `git status` first so untracked artifacts cannot be overlooked, then use the release validator's manifest-driven inventory/hash report to verify every generated shard even though `git diff --check` does not inspect untracked binary files.

- [ ] **Step 10: Re-read the acceptance checklist against generated evidence**

Verify explicitly:

- ten shards, 90 rows each, 900 unique ordered IDs;
- 1,498 gaps, 4,135 validated evidence links, 7,245 salient utterances;
- every target/evidence/salient coordinate resolves to released text;
- one evaluation runtime Python file and five exact prompt hashes;
- one fake command produces every metric and five output artifacts;
- no test or validation log shows network access or model/GPU initialization;
- v1 index hashes remain unchanged;
- README commands match the tested CLI help.

- [ ] **Step 11: Commit**

```bash
git add README.md benchmark/README.md dataset/README.md dataset/DATASET_CARD.md \
  dataset/benchmark/README.md evaluation/README.md scripts/validate_release.py \
  tests/test_release_layout.py
git commit -m "docs(release): document and validate reproducible evaluation"
```

---

## Final Verification

After all task commits, rerun from `3-release/`:

```bash
PYTHONPATH=benchmark/src:remember/src pytest -q -p no:cacheprovider \
  benchmark/tests remember/tests evaluation/tests tests
python scripts/validate_release.py
git log --oneline --decorate -12
git status --short --branch
```

Do not run the real Qwen judge during verification. A scientific full evaluation is a separate, explicitly authorized operator action that must name the provider and, for vLLM, the exact GPU IDs.
