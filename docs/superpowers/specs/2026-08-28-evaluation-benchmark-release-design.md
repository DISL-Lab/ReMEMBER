# Self-Contained Benchmark and Single-File Evaluation Design

**Date:** 2026-08-28

**Status:** Approved in chat

**Scope:** `3-release/` only

## Goal

Make the public release independently usable for paper-style evaluation. A user must be able to clone the release, supply one prediction JSONL, run one Python evaluator, and receive scores for every published automatic metric.

The released benchmark must be self-contained. Stable identifiers and coordinates remain for provenance, but every evaluation-relevant coordinate must also carry its corresponding text. The project has authorization to redistribute the included EverMemBench, LoCoMo, and RealTalk-derived text.

## Success Criteria

1. The canonical 900-window benchmark contains the full history and current window text, all target-gap text, all validated evidence text, and all salient-utterance text required by the evaluators.
2. Evaluation runtime logic is contained in one file: `evaluation/evaluate.py`.
3. The five canonical prompt templates are readable standalone files and are hashed in every run manifest.
4. One command evaluates MR, MKF, WKF, factuality, conciseness, and the diagnostic MR-gated MKF score.
5. Every aggregate score records its numerator, denominator, aggregation rule, and parse-failure count.
6. A deterministic CPU-only fake-provider test proves the complete input-to-score path without loading a model or using a GPU.
7. The official full-data path requires exactly the ordered 900 windows. Partial evaluation is possible only with explicit `--allow-partial`.

## Non-Goals

- Releasing raw judge generations from the original experiments.
- Releasing validator chain-of-thought or private model reasoning.
- Releasing source media files or media URLs.
- Reproducing historical directory names, method identifiers, background shell runners, or fixed GPU assignments from `1-research/`.
- Automatically starting or loading a GPU model during validation.

## Release Layout

```text
3-release/
├── dataset/
│   ├── benchmark/
│   │   ├── remember_900-00000-of-00010.jsonl.gz
│   │   ├── ...
│   │   ├── remember_900-00009-of-00010.jsonl.gz
│   │   ├── schema.json
│   │   ├── manifest.json
│   │   └── README.md
│   ├── annotations/remember_900_annotations.jsonl
│   ├── splits/main_900.txt
│   ├── DATASET_CARD.md
│   └── README.md
├── evaluation/
│   ├── evaluate.py
│   ├── prompts/
│   │   ├── memory_recall.txt
│   │   ├── mkf.txt
│   │   ├── wkf.txt
│   │   ├── atomic_facts.txt
│   │   └── factuality.txt
│   ├── requirements.txt
│   └── README.md
└── ...
```

`evaluation/evaluate.py` is the only evaluation runtime module. Test files are not runtime modules and may live under the existing release test directories.

The existing coordinate-only annotation file remains as a lightweight derived index for backward compatibility. It is no longer described as the benchmark or as sufficient for evaluation. The text-bearing shards are the canonical benchmark payload.

## Canonical Benchmark Construction

The builder joins these read-only research artifacts by `window_id` and writes only under `3-release/dataset/`:

- ordered population: `3-release/dataset/splits/main_900.txt`;
- full history and current windows: `0-dataset/combined_sampled.jsonl`;
- target gaps and validated evidence: `1-research/0/outputs/rcv/ensemble/all/relevant_chunk_validation.filtered.jsonl`;
- salient utterances for WKF: `1-research/0/outputs/salient/ensemble/5/salient.jsonl`.

Rows follow `main_900.txt` order and are split into exactly ten shards of 90 rows. Each shard is UTF-8 JSONL with real newline-delimited records, then deterministically gzip-compressed. The gzip writer fixes metadata that would otherwise make byte output depend on wall-clock time or local filenames.

The observed 900-window source payload is approximately 312 MB uncompressed and 77 MB gzip-compressed. Ten shards keep individual Git objects comfortably below common hosting limits and make integrity checks and partial downloads practical.

### Benchmark Record Schema

Each shard line is one complete evaluation window:

```json
{
  "schema_version": "2.0",
  "window_id": "E_U0000",
  "source": "evermembench",
  "domain": "...",
  "range": "8k~16k",
  "tok_start": 7000,
  "tok_end": 8024,
  "history": [
    {"turn_index": 0, "time": "...", "speaker": "...", "dialogue": "..."}
  ],
  "turns": [
    {"turn_index": 0, "time": "...", "speaker": "...", "dialogue": "..."}
  ],
  "salient_utterances": [
    {"salient_idx": 0, "sentence_numbers": [34], "sentences": ["..."], "gap_type": "A", "vote_count": 3}
  ],
  "gaps": [
    {
      "gap_idx": 0,
      "target_sentence_numbers": [34],
      "target_sentence": "...",
      "gap_type": "A",
      "gap_reason": "...",
      "evidence": [
        {
          "source_id": "T0-S2",
          "sentence": "...",
          "status": "resolve",
          "validation": 1,
          "vote_count": 3,
          "votes": {"gemma": 1, "GLM": 1, "Qwen": 1}
        }
      ]
    }
  ]
}
```

All `history` and `turns` entries contain normalized `time`, `speaker`, and `dialogue` strings. `turn_index` is zero-based within its container. Source-specific non-text metadata is omitted unless it is needed to prove coordinate alignment. Media paths and URLs are never copied.

Every target coordinate is resolved against the canonical flattened current-window sentence order. Every evidence `source_id` is resolved against canonical history turn and sentence indices. Export fails rather than emitting a key whose text cannot be verified exactly.

### Benchmark Manifest

`dataset/benchmark/manifest.json` records:

- schema version and ordered shard paths;
- SHA-256 and row count for every shard;
- SHA-256 for the schema and ordered split;
- source artifact hashes;
- windowing and sentence-splitting provenance;
- exact counts for windows, gaps, validated evidence links, and salient utterances;
- per-source and per-range counts;
- the authorization statement for redistributed text.

The fixed paper denominators remain 900 windows, 1,498 non-empty target gaps, and 4,135 validated evidence links. The salient-utterance denominator is computed from the canonical salient artifact and then fixed in the generated manifest and tests.

## Prediction Contract

The canonical input is UTF-8 JSONL with one row per benchmark window:

```json
{"window_id":"E_U0000","summary":"...","memory_text":"..."}
```

Rules:

- `window_id`, `summary`, and `memory_text` are required non-empty strings.
- Duplicate or unknown IDs are errors.
- A full official run requires exactly the same 900-ID set as the ordered benchmark split. Input order may differ; the evaluator normalizes output to benchmark order.
- `--allow-partial` permits a unique non-empty subset for smoke tests and development. All denominators then explicitly describe the evaluated subset.
- The evaluator also recognizes the current `remember run` output. If `memory_text` is absent but a structured `memory` object is present, it renders that object with the released canonical formatter. The normalized content must still be non-empty.
- Factuality always uses the actual supplied memory plus the canonical current window as evidence. It does not infer behavior from historical method names.

## Single-File Evaluator

Primary command:

```bash
python evaluation/evaluate.py \
  --benchmark dataset/benchmark \
  --predictions predictions.jsonl \
  --output-dir results \
  --provider vllm \
  --model Qwen/Qwen3.6-27B \
  --gpu-devices 0
```

The evaluator accepts a benchmark directory, one `.jsonl` or `.jsonl.gz` benchmark file, and a prediction JSONL. The default metric selection is `all`.

Supported providers:

- `vllm`: local judge, with an explicit `--gpu-devices` requirement;
- `openai`: OpenAI-compatible chat-completions endpoint, configured with `--base-url` and an environment-variable name such as `OPENAI_API_KEY`;
- `fake`: deterministic CPU-only integration provider.

The paper preset is `Qwen/Qwen3.6-27B` with temperature `0`. Heavy dependencies are imported lazily. No provider is initialized until inputs, schemas, hashes, and output policy have passed validation. The release validator and tests use only `fake` and never set GPU visibility or load a model.

`evaluation/requirements.txt` declares the lightweight parsing, tokenization, and OpenAI-compatible client dependencies needed for CPU/API evaluation. vLLM remains an explicitly documented optional installation because its compatible version depends on the local CUDA and PyTorch environment.

## Canonical Prompts

The released templates are exact, standalone copies of the final research prompts:

- MR: `1-research/4/memory_recall_prompt_v5.txt`;
- MKF: `1-research/5/prompt_mkf_v5.txt`;
- WKF: `1-research/5/prompt_wkf_v3.txt`;
- atomic-fact extraction: `1-research/5/prompt_atomic_facts.txt`;
- factuality checking: `1-research/5/prompt_factuality.txt`.

Prompt files remain external to `evaluate.py` so users can inspect and cite them directly. The evaluator validates required placeholders before inference and stores prompt hashes in `run_manifest.json`.

## Metric Definitions

### Memory Recall (MR)

For each gold gap, the MR prompt judges whether `memory_text` explicitly contains information that resolves the target using the validated evidence definition. Only decision `1` or `COVERED` is covered.

- micro MR: covered gaps divided by evaluated gaps;
- macro MR: mean of per-window covered-gap fractions.

### Missing-Knowledge Faithfulness / Gap-Conditioned Completeness (MKF)

For each gold gap case, the MKF prompt judges whether the summary recovers the minimal gap-resolving fact.

- micro MKF: positive gap scores divided by evaluated gap cases;
- macro MKF: mean of per-window gap-score means.

### Window-Knowledge Faithfulness / Window Completeness (WKF)

For each canonical salient utterance, the WKF prompt judges whether the summary preserves its referent and relation.

- micro WKF: positive utterance scores divided by evaluated salient utterances;
- macro WKF: mean of per-window utterance-score means.

### Factuality

The atomic-fact prompt decomposes each summary. The factuality prompt checks every atomic fact against `memory_text` plus the current window. A fact is factual only when normalized category is `no error`.

- atomic-fact micro factuality: factual atoms divided by evaluated atoms;
- window macro factuality: mean per-window factual fraction;
- category counts are preserved for all nine canonical categories.

### Conciseness

Conciseness reuses MKF and WKF line attribution. The numerator is the number of distinct summary lines receiving at least one positive MKF or WKF judgment. The denominator is the total number of normalized summary lines. Both per-window and corpus micro values are emitted.

### MR-Gated MKF

This diagnostic joins MR and MKF on `(window_id, gap_idx)`. An MKF gap score is forced to zero when the corresponding memory gap was not recalled. Macro and micro values are emitted but are labeled diagnostic rather than a primary paper metric.

## Parsing and Failure Semantics

- Parsers accept canonical JSON and the lenient structures supported by the research evaluators, including fenced JSON and complete-object salvage.
- Every expected item must be present exactly once after normalization. Duplicates, unexpected IDs, and out-of-range indices invalidate that judgment.
- A judge parse failure is scored as zero for its expected items and remains in the denominator.
- Atomic-fact parse failure falls back to one fact per normalized summary sentence, matching the established research behavior.
- Factuality checker parse failure marks the affected facts non-factual.
- Network and transient provider errors may be retried with bounded exponential backoff. Semantic parse failures are not automatically resampled.
- Raw responses, parser status, token counts, and error messages are retained for audit.

## Resume and Output Safety

`--resume` requires an existing run manifest whose benchmark, predictions, prompts, model, provider, and evaluation-option hashes match the current run. Completed `(metric, window_id)` judgments are reused; mismatches fail closed.

Without `--resume` or `--overwrite`, an existing output directory is rejected. Writes use temporary siblings and atomic replacement. Output order always follows the canonical benchmark order.

## Output Contract

```text
results/
├── scores.json
├── scores.csv
├── per_window.jsonl
├── raw_judgments.jsonl
└── run_manifest.json
```

- `scores.json`: nested metric results with macro/micro scores, numerators, denominators, counts, parse failures, and evaluated population.
- `scores.csv`: one row per `(metric, aggregation)` with the same core values.
- `per_window.jsonl`: normalized per-window scores and item-level decisions for every metric.
- `raw_judgments.jsonl`: prompt identity, raw judge response, parse status, token counts, and retry metadata.
- `run_manifest.json`: immutable provenance hashes and effective run configuration.

The command prints a concise metric table only after all output files have been validated and atomically finalized.

## Validation and Testing

Tests are behavior-first and cover:

1. benchmark construction retains exact target/evidence/salient text for fabricated coordinates;
2. unresolved or misaligned coordinates fail export;
3. deterministic sharding, gzip bytes, hashes, order, and exact paper counts;
4. prediction schema validation, ReMEMber-output adaptation, duplicate/missing/unknown IDs, and partial-run labeling;
5. prompt placeholder validation and hashes against the canonical copied templates;
6. strict and lenient parsing for all judge response formats;
7. exact macro/micro, conciseness, and MR-gated MKF calculations on literal fixtures;
8. parse failures remain in denominators and raw audit rows;
9. resume accepts identical provenance and rejects changed inputs or prompts;
10. a fake-provider end-to-end run produces exact golden `scores.json`, `scores.csv`, per-window, raw-judgment, and manifest outputs;
11. release validation checks every text-bearing shard, schema, count, and checksum without attempting a GPU or network call.

Real Qwen evaluation is an explicit operator action requiring a chosen provider and, for local vLLM, explicit GPU IDs. It is not part of automated release validation.

## Documentation Changes

- Root README adds a direct “Evaluate predictions” command and links the prediction schema.
- Dataset README and card identify the text-bearing shards as canonical, document authorization, counts, provenance, and limitations, and reclassify the old annotation file as a lightweight index.
- Benchmark-builder documentation explains how the ten shards are regenerated and validated.
- Evaluation README defines installation, providers, exact prompt/model preset, input and output schemas, full/partial runs, resume, and score formulas.

## Acceptance Checklist

- [ ] Ten ordered text-bearing shards contain exactly 900 unique windows.
- [ ] Every coordinate required by MR, MKF, or WKF has verified corresponding text.
- [ ] Counts are exactly 900 windows, 1,498 gaps, and 4,135 validated evidence links.
- [ ] `evaluation/evaluate.py` is the only evaluation runtime Python file.
- [ ] All five canonical prompts are present and hash-checked.
- [ ] One command emits every requested score and its denominator.
- [ ] Full-run omissions and parse failures cannot silently improve reported scores.
- [ ] Fake-provider end-to-end tests pass without network or GPU use.
- [ ] Existing benchmark, method, and release-validation suites remain green after migration.
