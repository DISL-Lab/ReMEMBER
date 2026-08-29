# ReMEMber Public Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained public release under `3-release/` with an annotation-only 900-window dataset, a single-command dataset-generation pipeline, and an end-to-end implementation of the paper's ReMEMber method.

**Architecture:** `dataset/` is a static, text-free artifact generated through a strict allowlist. `benchmark/` and `remember/` are independent Python packages with injectable inference backends, deterministic stage files, atomic writes, and one primary CLI each. Canonical research code under `0-dataset/`, `1-research/`, and `2-rebuttal/` is read-only reference material and is never imported at runtime by the release.

**Tech Stack:** Python 3.10+, standard library, PyYAML, NumPy, tiktoken, rank-bm25, optional sentence-transformers/OpenAI/vLLM, pytest

## Global Constraints

- Modify, stage, and commit files only under `3-release/`; all other repository folders are read-only references.
- The only dataset population is the paper's ordered 900-window split: exactly 900 unique windows, 1,498 gaps, and 4,135 validated evidence entries.
- Do not distribute upstream dialogue text or derived free-text excerpts. Forbidden release fields include `history`, `turns`, `target_sentence`, `salient_turn`, evidence `sentence`, `gap_reason`, validator `reasoning`, speaker names, timestamps, media paths, URLs, prompts containing dialogue, and raw model responses.
- Do not distribute model summaries, generated memories, factuality atoms, evaluator outputs, MTurk artifacts, logs, notebooks, caches, or model weights.
- Dataset output order must match `main_900.txt`; every evidence entry must have `validation == 1`.
- Release code must not contain machine-local absolute paths, fixed GPU IDs, credential literals, external credential-path fallbacks, `nohup`, or runtime imports from `0-dataset/`, `1-research/`, or `2-rebuttal/`.
- The ReMEMber configuration uses at most 8 gaps, 128-token chunks, 32-token overlap, BM25 plus Qwen3-Embedding-0.6B dense retrieval, reciprocal-rank fusion constant 60, round-robin facet-turn packing, a 1,024-token memory budget, and `cl100k_base` token accounting.
- Default gap and summary model: `Qwen/Qwen3.5-4B`; default dense model: `Qwen/Qwen3-Embedding-0.6B`.
- RCV/gold annotations are never required or read by `remember run`; a window-ID filter may only restrict the inference population.
- Both public CLIs must have CPU-only tests using deterministic fake providers. Default tests must not access the network, download models, or require a GPU.
- All JSONL inputs must report path and line number on malformed JSON or schema failure. All durable writes use a temporary sibling plus atomic rename.
- Resume must never duplicate a `window_id`; changed input/configuration hashes must stop instead of silently reusing stale output.
- Preserve the repository's existing MIT License and clearly distinguish it from upstream corpus and model terms; do not invent or replace license text.

---

### Task 1: Annotation-only 900-window dataset and release exporter

**Files:**
- Create: `3-release/benchmark/src/remember_benchmark/__init__.py`
- Create: `3-release/benchmark/src/remember_benchmark/io.py`
- Create: `3-release/benchmark/src/remember_benchmark/release.py`
- Create: `3-release/benchmark/tests/fixtures/rcv.jsonl`
- Create: `3-release/benchmark/tests/fixtures/main_ids.txt`
- Create: `3-release/benchmark/tests/test_release.py`
- Create: `3-release/benchmark/tests/test_published_dataset.py`
- Create: `3-release/dataset/annotations/remember_900_annotations.jsonl`
- Create: `3-release/dataset/splits/main_900.txt`
- Create: `3-release/dataset/schema.json`
- Create: `3-release/dataset/metadata.json`
- Create: `3-release/dataset/README.md`
- Create: `3-release/dataset/DATASET_CARD.md`

**Interfaces:**
- Consumes: JSONL rows shaped like the canonical RCV artifact and an ordered text file of window IDs.
- Produces: `iter_jsonl(path: Path) -> Iterator[dict[str, Any]]`, `atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None`, `sanitize_annotations(rcv_rows: Iterable[dict[str, Any]], ordered_window_ids: Sequence[str]) -> list[dict[str, Any]]`, `validate_annotations(rows: Sequence[dict[str, Any]], ordered_window_ids: Sequence[str], expected_counts: tuple[int, int, int] | None = None) -> dict[str, Any]`, and `export_annotations(rcv_path: Path, split_path: Path, output_path: Path, metadata_path: Path, schema_path: Path) -> dict[str, Any]`.
- Later benchmark tasks import `export_annotations` as the final pipeline stage.

- [ ] **Step 1: Write fixture-driven failing release tests**

```python
def test_sanitize_annotations_keeps_only_owned_coordinates():
    rows = sanitize_annotations(iter_jsonl(FIXTURES / "rcv.jsonl"), ["E_w1"])
    assert rows == [{
        "schema_version": "1.0",
        "window_id": "E_w1",
        "source": "evermembench",
        "range": "8k~16k",
        "gaps": [{
            "gap_idx": 0,
            "target_sentence_numbers": [2],
            "gap_type": "A",
            "evidence": [{
                "source_id": "T1-S0",
                "status": "resolve",
                "vote_count": 3,
                "votes": {"gemma": 1, "GLM": 1, "Qwen": 1},
                "validation": 1,
            }],
        }],
    }]
    serialized = json.dumps(rows)
    for forbidden in ("target_sentence", "sentence", "gap_reason", "reasoning", "Earlier context"):
        assert forbidden not in serialized
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_release.py`

Expected: FAIL because `remember_benchmark.release` does not exist.

- [ ] **Step 3: Implement strict JSONL I/O, allowlist sanitization, validation, hashing, and atomic export**

Implement source mapping `E -> evermembench`, `L -> locomo`, and `R -> realtalk`. Include only the fields shown in Step 1. Drop RCV rows not listed in the ordered split, rows with empty `target_sentence`, and evidence entries whose `validation` is not `1`. Reject unknown source prefixes, duplicate split IDs, duplicate emitted windows, forbidden keys at any nesting level, and count mismatches.

- [ ] **Step 4: Run fixture tests and verify GREEN**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_release.py`

Expected: all tests pass.

- [ ] **Step 5: Generate the real annotation-only artifact from read-only canonical inputs**

Run:

```bash
cp 1-research/analysis_outputs/analysis_v3/main_900_window_ids.txt \
  3-release/dataset/splits/main_900.txt
PYTHONPATH=3-release/benchmark/src python -m remember_benchmark.release \
  --rcv 1-research/0/outputs/rcv/ensemble/all/relevant_chunk_validation.filtered.jsonl \
  --split 3-release/dataset/splits/main_900.txt \
  --output 3-release/dataset/annotations/remember_900_annotations.jsonl \
  --metadata 3-release/dataset/metadata.json \
  --schema 3-release/dataset/schema.json \
  --expected-windows 900 \
  --expected-gaps 1498 \
  --expected-evidence 4135
```

The copied split contains only ordered IDs, not source data.

- [ ] **Step 6: Add failing published-artifact validation tests**

```python
def test_published_dataset_has_paper_counts_and_no_text_fields():
    rows = list(iter_jsonl(DATASET / "annotations/remember_900_annotations.jsonl"))
    stats = validate_annotations(rows, read_ids(DATASET / "splits/main_900.txt"), (900, 1498, 4135))
    assert stats == {"windows": 900, "gaps": 1498, "evidence": 4135}
```

Also test metadata SHA-256 values against the distributed annotation and split files.

- [ ] **Step 7: Run all dataset tests and verify GREEN**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_release.py 3-release/benchmark/tests/test_published_dataset.py`

Expected: all tests pass with exact paper counts.

- [ ] **Step 8: Write dataset documentation**

Document field meanings, upstream acquisition responsibility, join keys, the removal of all free text, exact denominators, limitations, and the repository MIT License boundary. Do not paste examples from the source corpora.

- [ ] **Step 9: Commit**

```bash
git add 3-release/benchmark/src/remember_benchmark 3-release/benchmark/tests 3-release/dataset
git commit -m "feat(release): publish text-free ReMEMber annotations"
```

---

### Task 2: ReMEMber schemas, gap parsing, chunking, and hybrid retrieval

**Files:**
- Create: `3-release/remember/pyproject.toml`
- Create: `3-release/remember/src/remember/__init__.py`
- Create: `3-release/remember/src/remember/config.py`
- Create: `3-release/remember/src/remember/schemas.py`
- Create: `3-release/remember/src/remember/io.py`
- Create: `3-release/remember/src/remember/gap_detection.py`
- Create: `3-release/remember/src/remember/retrieval.py`
- Create: `3-release/remember/tests/fixtures/windows.jsonl`
- Create: `3-release/remember/tests/test_schemas.py`
- Create: `3-release/remember/tests/test_gap_detection.py`
- Create: `3-release/remember/tests/test_retrieval.py`

**Interfaces:**
- Consumes: normalized window rows with `window_id`, `history`, and `turns`; each turn has string `time`, `speaker`, and `dialogue`.
- Produces: frozen dataclasses `Turn`, `Window`, `PredictedGap`, `Chunk`, and `RetrievedChunk`; `parse_predicted_gaps(raw_output: str, window: Window, max_gaps: int = 8, fallback: bool = True) -> list[PredictedGap]`; `build_token_chunks(history: Sequence[Turn], counter: TokenCounter, max_tokens: int = 128, overlap_tokens: int = 32) -> list[Chunk]`; and `hybrid_retrieve(gaps: Sequence[PredictedGap], chunks: Sequence[Chunk], embedder: Embedder, top_k: int = 8, rrf_k: int = 60) -> dict[int, list[RetrievedChunk]]`.
- Task 3 consumes these exact dataclasses and functions.

- [ ] **Step 1: Write failing schema and JSONL-error tests**

```python
def test_window_requires_complete_turn_fields(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"window_id":"w","history":[{"speaker":"A"}],"turns":[]}\n')
    with pytest.raises(InputError, match=r"bad\.jsonl:1.*history\[0\].*time"):
        list(load_windows(path))
```

- [ ] **Step 2: Run schema tests and verify RED**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests/test_schemas.py`

Expected: FAIL because release schemas and loader do not exist.

- [ ] **Step 3: Implement dataclasses, config validation, and line-aware atomic JSONL I/O**

Validate non-empty model IDs, `max_gaps` in `1..8`, positive chunk/budget values, and `0 <= overlap_tokens < chunk_tokens`. Keep heavyweight dependencies out of module import paths.

- [ ] **Step 4: Run schema tests and verify GREEN**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests/test_schemas.py`

Expected: all tests pass.

- [ ] **Step 5: Write failing lenient gap-parser tests**

Cover valid wrapped JSON, fenced JSON, a truncated outer wrapper with complete inner objects, quote recovery from `utterance_index`, duplicate quote removal, the eight-gap cap, and fallback creation. Expected structures must be literal values derived from fabricated windows.

- [ ] **Step 6: Run gap-parser tests and verify RED**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests/test_gap_detection.py`

Expected: FAIL because parser behavior is not implemented.

- [ ] **Step 7: Extract and simplify the canonical target-free gap normalization path**

Port only the behavior used by the paper prompt from `1-research/3/predicted_evidence_need_memory.py`; do not import that module. Preserve quote canonicalization, keyword construction, complete-object salvage, deduplication, and fallback semantics.

- [ ] **Step 8: Run gap-parser tests and verify GREEN**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests/test_gap_detection.py`

Expected: all tests pass.

- [ ] **Step 9: Write failing chunking and hybrid-retrieval tests**

```python
def test_rrf_combines_sparse_and_dense_rankings():
    result = reciprocal_rank_fuse(["c1", "c2"], ["c2", "c3"], k=60)
    assert [item.chunk_id for item in result] == ["c2", "c1", "c3"]
    assert result[0].score == pytest.approx(1 / 62 + 1 / 61)
```

Use a deterministic hand-written embedder and assert that chunk overlap retains whole turns without duplicating a chunk ID.

- [ ] **Step 10: Run retrieval tests and verify RED**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests/test_retrieval.py`

Expected: FAIL because chunking and hybrid retrieval are not implemented.

- [ ] **Step 11: Implement token chunking, BM25, dense cosine ranking, RRF, and per-gap retrieval**

Candidate lists are deterministic: descending score followed by source start index and chunk ID. Retrieve up to `4 * top_k` candidates from each backend, fuse with `rrf_k=60`, and return the first `top_k` per gap.

- [ ] **Step 12: Run the complete Task 2 suite and verify GREEN**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests/test_schemas.py 3-release/remember/tests/test_gap_detection.py 3-release/remember/tests/test_retrieval.py`

Expected: all tests pass.

- [ ] **Step 13: Commit**

```bash
git add 3-release/remember
git commit -m "feat(remember): add gap detection and hybrid retrieval core"
```

---

### Task 3: Canonical facet-turn packing and end-to-end ReMEMber CLI

**Files:**
- Create: `3-release/remember/src/remember/facet_turn.py`
- Create: `3-release/remember/src/remember/inference.py`
- Create: `3-release/remember/src/remember/summarization.py`
- Create: `3-release/remember/src/remember/pipeline.py`
- Create: `3-release/remember/src/remember/cli.py`
- Create: `3-release/remember/src/remember/__main__.py`
- Create: `3-release/remember/configs/paper.yaml`
- Create: `3-release/remember/prompts/gap_detection.txt`
- Create: `3-release/remember/prompts/summary_with_memory.txt`
- Create: `3-release/remember/prompts/summary_window_only.txt`
- Create: `3-release/remember/tests/fixtures/facet_turn_golden.json`
- Create: `3-release/remember/tests/test_facet_turn.py`
- Create: `3-release/remember/tests/test_pipeline.py`
- Create: `3-release/remember/tests/test_cli.py`
- Create: `3-release/remember/README.md`

**Interfaces:**
- Consumes: Task 2 dataclasses/functions and injectable `TextGenerator`/`Embedder` protocols.
- Produces: `facet_turn_pack(window: Window, gaps: Sequence[PredictedGap], retrieved: Mapping[int, Sequence[RetrievedChunk]], embedder: Embedder, counter: TokenCounter, budget_tokens: int = 1024) -> MemoryResult`; `summarize(window: Window, memory: MemoryResult, generator: TextGenerator, prompt_with_memory: str, prompt_without_memory: str) -> str`; `run_window(window: Window, config: RememberConfig, generator: TextGenerator, embedder: Embedder) -> RunResult`; and CLI `python -m remember run --config PATH --input PATH --output PATH [--resume|--overwrite]`.
- Task 5 invokes this CLI in a cross-package smoke test.

- [ ] **Step 1: Write failing canonical facet-turn behavior tests**

Use fabricated turns and a literal golden fixture to cover query/reason/utterance facets, round-robin facet allocation, promotion from selected sentences to complete raw turns, sentence fallback when one raw turn exceeds the remaining budget, deterministic tie-breaking, and the 1,024-token cap.

- [ ] **Step 2: Run facet-turn tests and verify RED**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests/test_facet_turn.py`

Expected: FAIL because `remember.facet_turn` does not exist.

- [ ] **Step 3: Extract the canonical paper branch into focused pure functions**

Use the current `facet_turn_pack` path only as read-only reference. Keep round-robin allocation and raw-turn promotion semantics. Exclude target-evidence, allocation-policy, bridge, quote-pair, post-processing, and rebuttal variants.

- [ ] **Step 4: Run facet-turn tests and verify GREEN**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests/test_facet_turn.py`

Expected: all tests pass.

- [ ] **Step 5: Write failing end-to-end pipeline and resume tests**

```python
def test_run_window_executes_gap_retrieval_pack_and_summary(fake_generator, fake_embedder):
    result = run_window(WINDOW, PAPER_CONFIG, fake_generator, fake_embedder)
    assert result.window_id == "E_fixture"
    assert result.schema_version == "1.1"
    assert result.memory.token_count <= 1024
    assert result.summary == "Fixture summary."

def test_resume_never_duplicates_window_ids(tmp_path):
    run_file(INPUT_TWO_WINDOWS, tmp_path / "out.jsonl", resume=False)
    run_file(INPUT_TWO_WINDOWS, tmp_path / "out.jsonl", resume=True)
    assert read_ids(tmp_path / "out.jsonl") == ["w1", "w2"]
```

- [ ] **Step 6: Run pipeline tests and verify RED**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests/test_pipeline.py 3-release/remember/tests/test_cli.py`

Expected: FAIL because inference, pipeline, and CLI modules do not exist.

- [ ] **Step 7: Implement lazy inference adapters, summarization, pipeline orchestration, and CLI**

Define protocols before adapters. Provide a deterministic fake adapter for tests, an OpenAI-compatible chat adapter using `OPENAI_API_KEY` or an explicit environment-variable name, a lazy in-process vLLM adapter, and a lazy sentence-transformers embedder. Never inspect cache internals or a YAML key file. Set device visibility before heavyweight imports.

Output each window as a JSON object with `schema_version`, `window_id`, `predicted_gaps`, `retrieval`, `memory`, `summary`, `token_counts`, `timing_seconds`, and `effective_config`. Do not echo `history` or `turns`. Resume validates an adjacent manifest's input and configuration hashes and rewrites atomically without duplicates.

- [ ] **Step 8: Run pipeline and CLI tests and verify GREEN**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests/test_pipeline.py 3-release/remember/tests/test_cli.py`

Expected: all tests pass without network/GPU access.

- [ ] **Step 9: Add paper configuration, exact prompts, and usage documentation**

Copy only the canonical prompt templates, without source dialogue. Document install commands, normalized input schema, fake smoke invocation, local-vLLM/OpenAI-compatible options, model requirements, memory-only versus end-to-end behavior, output schema, resume rules, and the retained `cl100k_base` accounting choice.

- [ ] **Step 10: Run the entire ReMEMber package suite**

Run: `PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests`

Expected: all tests pass.

- [ ] **Step 11: Commit**

```bash
git add 3-release/remember
git commit -m "feat(remember): release end-to-end paper pipeline"
```

---

### Task 4: Deterministic benchmark stages and resumable manifests

**Files:**
- Create: `3-release/benchmark/pyproject.toml`
- Create: `3-release/benchmark/src/remember_benchmark/config.py`
- Create: `3-release/benchmark/src/remember_benchmark/schemas.py`
- Create: `3-release/benchmark/src/remember_benchmark/providers.py`
- Create: `3-release/benchmark/src/remember_benchmark/sources.py`
- Create: `3-release/benchmark/src/remember_benchmark/windows.py`
- Create: `3-release/benchmark/src/remember_benchmark/salient.py`
- Create: `3-release/benchmark/src/remember_benchmark/chunk_filter.py`
- Create: `3-release/benchmark/src/remember_benchmark/evidence.py`
- Create: `3-release/benchmark/src/remember_benchmark/validation.py`
- Create: `3-release/benchmark/src/remember_benchmark/ensemble.py`
- Create: `3-release/benchmark/src/remember_benchmark/manifest.py`
- Create: `3-release/benchmark/tests/fixtures/source_dialogues.jsonl`
- Create: `3-release/benchmark/tests/test_sources.py`
- Create: `3-release/benchmark/tests/test_windows.py`
- Create: `3-release/benchmark/tests/test_stages.py`
- Create: `3-release/benchmark/tests/test_manifest.py`

**Interfaces:**
- Consumes: user-acquired EverMemBench, LoCoMo, and RealTalk paths plus injectable stage providers.
- Produces: normalized dialogue JSONL; stable `E_`, `L_`, and `R_` windows; stage functions `run_salient`, `run_chunk_filter`, `run_evidence`, `run_validation`; `ensemble_votes(rows_by_model: Mapping[str, Sequence[dict[str, Any]]], threshold: int) -> list[dict[str, Any]]`; and `StageManifest` with configuration/input/output/prompt hashes.
- Task 5 composes these stages into the public `build` command.

- [ ] **Step 1: Write failing source-adapter and window tests**

Cover the locally observed upstream schemas with fabricated records, stable prefixes, chronological turn normalization, missing-file errors with acquisition guidance, and the paper window rules. Assert literal window IDs and boundaries. Test that range assignment includes `other` explicitly instead of silently losing those windows.

- [ ] **Step 2: Run source/window tests and verify RED**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_sources.py 3-release/benchmark/tests/test_windows.py`

Expected: FAIL because source and window modules do not exist.

- [ ] **Step 3: Implement source adapters, normalized schemas, composition, and window construction**

Adapters read user paths only and never download implicitly. Keep domain-specific paper stride/start defaults in configuration. Composition is a Python function, not a notebook. Validate IDs and make output deterministic.

- [ ] **Step 4: Run source/window tests and verify GREEN**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_sources.py 3-release/benchmark/tests/test_windows.py`

Expected: all tests pass.

- [ ] **Step 5: Write failing LLM-stage and ensemble tests**

Use a scripted fake provider whose complete responses mirror the documented provider interface. Assert parsed salient candidates, 128/32 history chunks, evidence coordinates, unanimous chunk filtering, two-of-three validation, rejection of malformed provider output, and preservation of model votes without free text in the final release stage.

- [ ] **Step 6: Run stage tests and verify RED**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_stages.py`

Expected: FAIL because stage modules do not exist.

- [ ] **Step 7: Implement provider protocol and four explicit annotation stages**

Each stage accepts records plus a provider and prompt; it does not discover files or global configuration. Provider results are parsed into versioned schemas. Expose OpenAI-compatible and lazy local-vLLM adapters without importing optional dependencies during tests.

- [ ] **Step 8: Run stage tests and verify GREEN**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_stages.py`

Expected: all tests pass.

- [ ] **Step 9: Write failing manifest/resume tests**

```python
def test_resume_rejects_changed_prompt_hash(tmp_path):
    old = write_stage_manifest(tmp_path, prompt="v1", inputs=[INPUT], config={"seed": 42})
    with pytest.raises(ResumeMismatch, match="prompt hash"):
        verify_resume(old.path, prompt="v2", inputs=[INPUT], config={"seed": 42})
```

Also test changed input/config hashes, incomplete temporary output, atomic publication, and a valid exact-match resume.

- [ ] **Step 10: Run manifest tests and verify RED**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_manifest.py`

Expected: FAIL because manifest logic is not implemented.

- [ ] **Step 11: Implement hashes, atomic stage publication, and strict resume verification**

Use canonical sorted JSON for configuration hashes and SHA-256 for files/prompts. A stage is complete only when output and manifest both exist and all recorded hashes match.

- [ ] **Step 12: Run the complete Task 4 suite and verify GREEN**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_sources.py 3-release/benchmark/tests/test_windows.py 3-release/benchmark/tests/test_stages.py 3-release/benchmark/tests/test_manifest.py`

Expected: all tests pass.

- [ ] **Step 13: Commit**

```bash
git add 3-release/benchmark
git commit -m "feat(benchmark): add deterministic annotation stages"
```

---

### Task 5: Single benchmark CLI, documentation, and whole-release validation

**Files:**
- Create: `3-release/benchmark/src/remember_benchmark/pipeline.py`
- Create: `3-release/benchmark/src/remember_benchmark/cli.py`
- Create: `3-release/benchmark/src/remember_benchmark/__main__.py`
- Create: `3-release/benchmark/configs/paper.yaml`
- Create: `3-release/benchmark/prompts/salient.txt`
- Create: `3-release/benchmark/prompts/relevant_chunk_filtering.txt`
- Create: `3-release/benchmark/prompts/relevant_chunk.txt`
- Create: `3-release/benchmark/prompts/relevant_chunk_validation.txt`
- Create: `3-release/benchmark/tests/test_pipeline.py`
- Create: `3-release/benchmark/tests/test_cli.py`
- Create: `3-release/benchmark/README.md`
- Create: `3-release/scripts/validate_release.py`
- Create: `3-release/tests/test_release_layout.py`
- Create: `3-release/README.md`

**Interfaces:**
- Consumes: Task 1 release exporter and Task 4 source/stage/manifest interfaces.
- Produces: CLI `python -m remember_benchmark build --config PATH --work-dir PATH --output-dir PATH [--resume] [--overwrite] [--from-stage NAME] [--to-stage NAME]`; validation command `python 3-release/scripts/validate_release.py`; and root release documentation.

- [ ] **Step 1: Write failing CPU-only full-build tests**

```python
def test_build_runs_all_stages_and_exports_annotations(tmp_path, scripted_provider):
    result = build(config=FIXTURE_CONFIG, work_dir=tmp_path / "work", output_dir=tmp_path / "out", provider=scripted_provider)
    assert result.stage_names == [
        "sources", "windows", "salient", "chunk_filter", "evidence", "validation", "ensemble", "release"
    ]
    assert (tmp_path / "out/annotations/remember_900_annotations.jsonl").exists()
```

For fixtures, configure expected counts to literal synthetic values rather than paper counts. Add separate tests for `--from-stage`, `--to-stage`, resume, overwrite protection, and actionable missing-source errors.

- [ ] **Step 2: Run pipeline/CLI tests and verify RED**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_pipeline.py 3-release/benchmark/tests/test_cli.py`

Expected: FAIL because pipeline and CLI modules do not exist.

- [ ] **Step 3: Implement the ordered stage runner and public CLI**

Use one stage registry with the exact order in Step 1. `build` creates stage directories under `work-dir`, uses Task 4 manifests for skip/restart decisions, and calls Task 1 `export_annotations` last. It must not shell out to historical scripts.

- [ ] **Step 4: Run pipeline/CLI tests and verify GREEN**

Run: `PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests/test_pipeline.py 3-release/benchmark/tests/test_cli.py`

Expected: all tests pass without network/GPU access.

- [ ] **Step 5: Add exact paper config, canonical prompts, and package documentation**

Document that `build` is one CLI with modular resumable stages; list user-supplied upstream data requirements; describe provider/model configuration; explain historical non-byte-identical LLM limitations; give full and partial-run commands; and identify all generated artifacts. Copy only canonical prompt instructions, never source examples containing upstream dialogue.

- [ ] **Step 6: Write failing whole-release layout and leakage tests**

The tests recursively reject files outside `dataset`, `benchmark`, `remember`, `docs`, `scripts`, `tests`, and root `README.md`; reject forbidden extensions/directories; scan release source/config files for local absolute paths, credential assignments, credential-path fallbacks, and research-tree runtime imports; recursively scan only the published dataset JSON/JSONL payloads for forbidden content fields; compile Python files; and invoke both fake-provider CLIs.

- [ ] **Step 7: Run whole-release tests and verify RED**

Run: `pytest -q 3-release/tests/test_release_layout.py`

Expected: FAIL until validation script and root documentation are complete.

- [ ] **Step 8: Implement the validation command and root documentation**

`validate_release.py` prints a machine-readable summary with dataset counts, file hashes, Python compile result, test commands, and leakage-scan result. The root README explains the three folders, quick-start commands, optional dependencies, ownership boundary, known reproduction limitations, and the MIT/upstream-license boundary.

- [ ] **Step 9: Run package and whole-release verification**

Run:

```bash
PYTHONPATH=3-release/remember/src pytest -q 3-release/remember/tests
PYTHONPATH=3-release/benchmark/src pytest -q 3-release/benchmark/tests
pytest -q 3-release/tests
python 3-release/scripts/validate_release.py
python -m compileall -q 3-release/benchmark/src 3-release/remember/src 3-release/scripts
```

Expected: every command exits `0`; dataset summary is exactly `900/1498/4135`; no network/GPU is used.

- [ ] **Step 10: Confirm the modification boundary**

Run:

```bash
git status --short
git diff --name-only --cached
git log --name-only --pretty=format: --all | sort -u
```

Expected: all staged/committed paths begin with `3-release/`; unrelated untracked research files remain untouched.

- [ ] **Step 11: Commit**

```bash
git add 3-release
git commit -m "feat(release): add single benchmark builder and validation"
```
