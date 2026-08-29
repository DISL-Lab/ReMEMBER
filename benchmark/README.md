# ReMEMber benchmark builder

> The checked-in benchmark in [`../dataset/`](../dataset/README.md) is the
> artifact to evaluate against; this package is the pipeline that produced it
> and is not needed to run or score a system.

`remember_benchmark` exposes one builder command with eight deterministic, resumable stages:

```text
sources -> windows -> salient -> chunk_filter -> evidence -> validation -> ensemble -> release
```

The implementation never downloads source data. Supply user-acquired EverMemBench, LoCoMo, and RealTalk
paths in a copy of `configs/paper.yaml`. The source adapters report acquisition guidance when a path is
missing. The paper configuration fixes model label order, model IDs, voting thresholds, token windows,
chunk sizes, prompt paths, temperature, ordered split, and expected release counts; it intentionally
contains no source paths, credential values, local absolute paths, or fixed accelerator IDs.

## Install and providers

```bash
python -m pip install ./benchmark
```

The default paper token counter is exact `cl100k_base`. The base install includes PyYAML and `tiktoken`.
OpenAI-compatible and vLLM providers are optional extras. Provider imports are lazy, so the test suite and
the documented `fake` provider use CPU only and require no network, credentials, or model download.
The fake provider exists for schema/integration validation; it is not a scientific replacement for the
paper models.

## Full and partial builds

```bash
PYTHONPATH=benchmark/src python -m remember_benchmark build \
  --config CONFIG.yaml \
  --work-dir PRIVATE_WORK_DIR \
  --output-dir PUBLIC_OUTPUT_DIR \
  --provider openai
```

Resume only when all manifests, input hashes, prompts, configuration, provider identities, intermediate
outputs, and release artifacts are unchanged:

```bash
PYTHONPATH=benchmark/src python -m remember_benchmark build \
  --config CONFIG.yaml --work-dir PRIVATE_WORK_DIR --output-dir PUBLIC_OUTPUT_DIR \
  --provider openai --resume
```

Run a verified range or recompute one stage and everything downstream:

```bash
PYTHONPATH=benchmark/src python -m remember_benchmark build \
  --config CONFIG.yaml --work-dir PRIVATE_WORK_DIR --output-dir PUBLIC_OUTPUT_DIR \
  --provider openai --resume --from-stage validation --to-stage ensemble

PYTHONPATH=benchmark/src python -m remember_benchmark build \
  --config CONFIG.yaml --work-dir PRIVATE_WORK_DIR --output-dir PUBLIC_OUTPUT_DIR \
  --provider openai --overwrite --from-stage evidence
```

Stage names and order are validated. A partial run verifies every skipped prerequisite. Ending before
`release` does not create public artifacts. Existing outputs are protected unless `--resume` or
`--overwrite` is explicit; these flags are mutually exclusive. Without `--to-stage`, overwrite runs from
the selected stage through `release`. An explicit `--to-stage` is the bound; later stages remain untouched.

## Artifact boundary

The work directory contains all private upstream-derived text and intermediates, with per-model files and
manifests such as `salient/gemma.jsonl`, `validation/Qwen.jsonl`, and a private flat exporter bridge.
**Do not place the work directory in this checkout, commit it, or distribute it.**

The output directory contains only:

- `annotations/remember_900_annotations.jsonl`
- `schema.json`
- `metadata.json`
- `splits/main_900.txt`

The final annotation rows contain only allowlisted coordinates, types, statuses, counts, and dense vote
maps. Private target text is joined from the merged salient stage only inside the work directory and is
discarded during export.

Real language-model generations may vary across model/provider revisions and therefore may not reproduce
the published annotations byte-for-byte. The paper configuration's exact expected counts are a fail-closed
publication integrity contract: if a real-model run drifts, `release` aborts without replacing the public
directory, while complete private intermediates remain in the work directory for inspection. Only scripted
CPU providers were exercised in this release validation; optional real providers and GPU serving require
environment-specific validation by the user.
