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

**Benchmark** — the published payload, or any file in the same v2 record shape:

```bash
--benchmark dataset/remember_900.jsonl.gz     # or an uncompressed .jsonl
```

A run over anything other than the exact canonical 900 windows requires
`--allow-partial`.

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
  --benchmark dataset/remember_900.jsonl.gz \
  --predictions runs/my_system.jsonl \
  --output-dir runs/my_system_scores \
  --provider vllm --gpu-devices 0,1 --tensor-parallel-size 2
```

OpenAI-compatible endpoint:

```bash
export MY_JUDGE_KEY=...
python evaluation/evaluate.py \
  --benchmark dataset/remember_900.jsonl.gz \
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

## Resume and overwrite

An interrupted run leaves a recoverable `.<output-dir>.in_progress` directory
holding the raw checkpoint. `--resume` reuses every completed request keyed by
`metric:phase:window_id`, and refuses if the benchmark, predictions, prompts,
model, provider, temperature, token limits, metric selection, or evaluated ID
set changed. `--overwrite` replaces a complete result; the previous five files
stay untouched until the new ones validate.
