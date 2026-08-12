# ReMEMber public release

This repository publishes only the ReMEMber-owned parts of the project. It has three product folders:

- `dataset/` contains the text-free 900-window annotation release. It includes coordinates, gap labels,
  evidence coordinates, and model votes, but no upstream dialogue or model reasoning.
- `benchmark/` is the single, modular dataset-generation pipeline.
- `remember/` is the end-to-end ReMEMber method: gap prediction, hybrid retrieval, facet-turn memory,
  and final summarization.

The `docs/`, `scripts/`, and `tests/` folders support the release. They are not additional products.

## Install

Use separate editable installs if desired:

```bash
python -m pip install ./benchmark
python -m pip install ./remember
```

The base packages require PyYAML and `tiktoken`. OpenAI-compatible clients, vLLM, sentence-transformer
retrieval, and their model weights are optional extras; installing or running tests does not download
models. Real provider credentials are read from an environment-variable name selected at runtime and
are never stored in configuration.

## Quick start

Run the ReMEMber method with its deterministic CPU fake provider:

```bash
PYTHONPATH=remember/src python -m remember run \
  --config remember/configs/paper.yaml \
  --input INPUT_WINDOWS.jsonl \
  --output OUTPUT.jsonl \
  --provider fake
```

Build benchmark annotations with one command after filling a copy of the paper configuration with
paths to user-acquired source corpora and an ordered split:

```bash
PYTHONPATH=benchmark/src python -m remember_benchmark build \
  --config CONFIG.yaml \
  --work-dir PRIVATE_WORK_DIR \
  --output-dir PUBLIC_OUTPUT_DIR \
  --provider openai
```

The benchmark command contains eight resumable stages in fixed order: `sources`, `windows`, `salient`,
`chunk_filter`, `evidence`, `validation`, `ensemble`, and `release`. See
[`benchmark/README.md`](benchmark/README.md) for partial execution and provider details.

## Inputs, outputs, and privacy boundary

EverMemBench, LoCoMo, and RealTalk are not redistributed here. Acquire them from their respective
upstream distributors and supply their local paths. The caller-selected benchmark `work-dir` contains
normalized dialogue, prompts sent to providers, target sentences, evidence text, and intermediate model
decisions. It is private working data: do not commit, upload, or distribute it.

The public output contains sanitized annotations, schema, metadata, and the ordered split. The checked-in
dataset has exactly 900 windows, 1,498 gaps, and 4,135 validated evidence links. Run
`python scripts/validate_release.py` to verify hashes, counts, Python compilation, CPU-only tests and
smokes, layout, and leakage checks.

## Reproduction and ownership limitations

The deterministic code path, schemas, thresholds, stage order, prompts, and manifests are provided.
Outputs from real language models are not guaranteed to be byte-identical across model revisions,
serving stacks, hardware, or provider behavior. This release was validated with scripted CPU providers;
it does not claim that every optional real-provider or GPU combination was executed.

Project-authored annotations, code, prompts, and documentation in this repository are released under
the repository's MIT License. Upstream dialogue remains governed by its original owners and licenses
and is deliberately absent; the MIT License does not grant rights to those upstream corpora or model
weights.
