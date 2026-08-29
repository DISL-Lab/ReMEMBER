# ReMEMber

This package is a self-contained implementation of ReMEMber. It predicts
contextual gaps from only the current window, retrieves earlier dialogue with
BM25 plus dense embeddings, packs retrieved evidence with round-robin
facet-turn allocation, and optionally produces a reinforced summary.

The public runner never reads relevant-chunk-validation (RCV) or other gold
annotations. It does not import research scripts at runtime.

## Install

Python 3.10 or newer is required. Install the base package for the fake smoke
provider and exact `cl100k_base` accounting:

```bash
python -m pip install -e .
```

For an OpenAI-compatible endpoint and sentence-transformers retrieval:

```bash
python -m pip install -e '.[retrieval,inference]'
```

The in-process vLLM adapter is optional and imports vLLM only when first used.
Install `.[retrieval,vllm]` (or a vLLM build compatible with the local
CUDA/PyTorch environment) before selecting `--provider vllm`.

## Input

Input is UTF-8 JSONL with one normalized window per line:

```json
{"window_id":"demo-1","history":[{"time":"09:00","speaker":"A","dialogue":"Earlier dialogue."}],"turns":[{"time":"10:00","speaker":"B","dialogue":"Current dialogue."}]}
```

`window_id` must be a non-empty string. `history` and `turns` must be arrays;
each turn requires string-valued `time`, `speaker`, and `dialogue` fields.
Malformed rows report the input path and physical line number.

## Offline smoke run

From the package directory:

```bash
python -m remember run \
  --config configs/paper.yaml \
  --input tests/fixtures/windows.jsonl \
  --output /tmp/remember-smoke.jsonl \
  --provider fake
```

The fake provider is deterministic, CPU-only, and never accesses the network
or downloads a model. Like every provider, it uses exact `cl100k_base`
accounting.

## Paper providers

The canonical defaults in `configs/paper.yaml` are:

- gap and summary model: `Qwen/Qwen3.5-4B`
- dense model: `Qwen/Qwen3-Embedding-0.6B`
- at most 8 gaps
- 128-token chunks with 32-token whole-turn overlap
- BM25 plus dense retrieval, top 8 after reciprocal-rank fusion with `k=60`
- round-robin query/reason/utterance facet allocation
- complete raw-turn promotion with sentence fallback
- a 1,024-token memory cap using `cl100k_base`

For an OpenAI-compatible chat endpoint, put the key in `OPENAI_API_KEY` (or
name a different environment variable with `--api-key-env`) and run:

```bash
python -m remember run \
  --config configs/paper.yaml \
  --input windows.jsonl \
  --output remember.jsonl \
  --provider openai \
  --base-url https://your-endpoint.example/v1
```

The adapter never reads a YAML key file and does not inspect model-cache
internals. The endpoint must expose OpenAI-compatible chat completions for both
the configured gap and summary model IDs.

For local in-process vLLM:

```bash
python -m remember run \
  --config configs/paper.yaml \
  --input windows.jsonl \
  --output remember.jsonl \
  --provider vllm \
  --gpu-devices 0 \
  --dense-device cuda
```

Device visibility is set before vLLM, PyTorch, or sentence-transformers is
imported. The gap and summary model IDs are independent; when they differ, the
lazy vLLM adapter switches models between stages. Ensure adequate accelerator
memory and a compatible vLLM installation before starting a local run.

Use `--memory-only` to stop after gap detection, retrieval, and packing. The
output retains an empty `summary` field. Without this flag, the runner executes
the complete end-to-end summary path.

## Output

Every output JSON object has these top-level fields:

- `schema_version` and `window_id`
- normalized `predicted_gaps` and per-gap `retrieval`
- `memory`, including selected evidence items, layout, policy, and token count
- `summary`
- stage `token_counts` and `timing_seconds`
- `effective_config`

The input `history` and `turns` containers are not copied into output rows.
Retrieved evidence and packed memory necessarily contain the selected dialogue
text used by the method. Output files may therefore contain sensitive source
dialogue and should be handled under the upstream dataset's terms.

## Resume and overwrite

Writes use a temporary sibling followed by atomic rename. Each output has an
adjacent `<output>.manifest.json` containing SHA-256 hashes for the input,
effective configuration, and published output.

- A new run refuses to replace an existing output unless `--overwrite` is set.
- `--resume` requires both output and manifest, validates input and
  configuration hashes, skips completed IDs, and atomically rewrites the
  ordered result without duplicate `window_id` values.
- `--resume` and `--overwrite` are mutually exclusive.
- Changed input or configuration stops the run instead of reusing stale rows.

## License

The project-authored implementation, prompts, documentation, and released
annotations are provided under the repository's MIT License. This license does
not apply to upstream dialogue data, model weights, or separately generated
outputs governed by other terms.

### `--token-counter`

`--token-counter paper` (the default) is the exact `cl100k_base` accounting the
paper used and needs `tiktoken`. `--token-counter whitespace` is an approximate
dependency-free counter that exists so release smokes can run without
`tiktoken`; **it does not reproduce paper windowing and must not be used for
results.** Whichever is used is recorded as `counter_adapter` in the run
manifest.
