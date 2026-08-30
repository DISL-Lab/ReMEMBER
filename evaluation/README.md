# ReMEMber evaluation

`evaluate.py` is the whole evaluator: one file, standard library plus NLTK and
an OpenAI-compatible client. It imports nothing from `benchmark/`, nothing from
`remember/`, and nothing from the research tree.

## Install

```bash
python -m pip install -r evaluation/requirements.txt
python -m nltk.downloader punkt_tab
```

vLLM is optional and environment-specific; install it separately if you want to
run the judge locally.

## Inputs

**Benchmark** — a v2-shaped file, or a directory of them:

```bash
--benchmark dataset                                        # all three domain shards together (900 windows)
--benchmark dataset/remember_900_workplace_1.jsonl.gz       # one shard only
--benchmark my_predictions.jsonl                            # or a standalone uncompressed .jsonl
```

The released benchmark ships as three files across two conversational domains -
`remember_900_chitchat.jsonl.gz` (LoCoMo + RealTalk, 300 windows) and
`remember_900_workplace_1.jsonl.gz` / `remember_900_workplace_2.jsonl.gz`
(EverMemBench, 300 windows each, split by count only) - so each stays well
under GitHub's file-size warning threshold. Pointing `--benchmark` at `dataset`
(or any directory) loads every `.jsonl`/`.jsonl.gz` file inside it, in filename
order, and treats them as one benchmark; for the shipped shards that
reconstructs the exact canonical 900-window order. A run over anything other
than the exact canonical 900 windows requires `--allow-partial`.

**Predictions** — one JSON object per line:

```json
{"window_id": "E_U0001", "summary": "...", "memory_text": "..."}
```

`summary` is what the system produced for the current window. Memory may be
given in these forms, in precedence order:

| form | meaning |
|---|---|
| `"memory_text": "..."` | generic, wins over everything else |
| `"memory": "..."` | a plain string |
| `"memory": {"text": "..."}` | ReMEMber's own output shape |
| `"memory": {"items": [{"text": "..."}, ...]}` | fallback; joined with newlines in order |

Anything else is rejected rather than JSON-stringified, so a memory object is
never silently turned into `{'k': 'v'}` text inside a judge prompt.

## Running

Local vLLM judge. vLLM is not in `requirements.txt` — install it separately
(`python -m pip install vllm`) for this path. GPU IDs are always explicit;
nothing is chosen for you:

```bash
python evaluation/evaluate.py \
  --benchmark dataset \
  --predictions runs/my_system.jsonl \
  --output-dir runs/my_system_scores \
  --provider vllm --gpu-devices 0,1 --tensor-parallel-size 2
```

OpenAI-compatible endpoint:

```bash
export MY_JUDGE_KEY=...
python evaluation/evaluate.py \
  --benchmark dataset \
  --predictions runs/my_system.jsonl \
  --output-dir runs/my_system_scores \
  --provider openai --base-url https://your-endpoint/v1 --api-key-env MY_JUDGE_KEY
```

Only the variable named by `--api-key-env` is read. No other credential is
inspected, and none is ever written to an output file.

Wiring check with no model, GPU, or credential:

```bash
python evaluation/evaluate.py --benchmark <small.jsonl> --predictions <preds.jsonl> \
  --output-dir runs/smoke --provider fake --allow-partial
```

The `fake` provider decides purely by literal substring containment. It is an
integration oracle, never a scientific result; every run it produces is marked
`scientific_provider: false`.

## Metrics

| metric | numerator / denominator |
|---|---|
| `mr` | gaps whose evidence the memory covers / all gaps |
| `mkf` | gap facts the summary faithfully carries / all gaps |
| `wkf` | salient utterances the summary covers / all salient utterances |
| `factuality` | atomic facts judged `no error` / all atomic facts |
| `conciseness` | summary sentences credited by MKF or WKF / all summary sentences |
| `mr_gated_mkf` | MKF over gaps MR marked covered — **diagnostic**, not a headline number |

Both `micro` (sum numerators / sum denominators) and `macro` (mean of per-window
ratios over windows that have items) are reported. Every metric also reports
`parse_failure_count`; a window whose judge output cannot be parsed scores zero
for all of its expected items rather than being dropped.

Atomic decomposition is the one exception to that rule: if it fails to parse,
the evaluator falls back to one fact per summary sentence and still scores them.

## Outputs

`--output-dir` receives exactly five files, written atomically:

| file | contents |
|---|---|
| `scores.json` | every aggregate, with the macro/micro definitions inline |
| `scores.csv` | the same table, deterministic bytes |
| `per_window.jsonl` | one row per evaluated window, in benchmark order |
| `raw_judgments.jsonl` | one audit row per judge request — never the prompt itself |
| `run_manifest.json` | input hashes, provider/model options, and provenance flags |

`run_manifest.json` records `canonical_benchmark`, `canonical_prompts`,
`paper_model_temperature`, and `scientific_provider`; `official_preset` is true
only when all four are. A custom prompt directory stays usable — it is just
visibly not the paper preset.

Each metric block in `scores.json`/`scores.csv` also carries `by_length_bucket`:
the same micro/macro numbers restricted to windows in one of the paper's three
dialogue-length groups (`8K-32K`, `32K-64K`, `64K-160K`), bucketed on each
window's `tok_end` — this is analysis_v3's `LENGTH_BIN_EDGES_K = (32, 64)`
grouping, not the dataset's finer 5-way `range` sampling stratum. Each
`per_window.jsonl` row carries the same bucket under `length_bucket`.

## Resume and overwrite

An interrupted run leaves a recoverable `.<output-dir>.in_progress` directory
holding the raw checkpoint. `--resume` reuses every completed request keyed by
`metric:phase:window_id`, and refuses if the benchmark, predictions, prompts,
model, provider, temperature, token limits, metric selection, or evaluated ID
set changed. `--overwrite` replaces a complete result; the previous five files
stay untouched until the new ones validate.
