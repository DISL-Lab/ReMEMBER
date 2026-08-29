# ReMEMber dataset

This directory publishes two views of the same 900-window release.

## `remember_900.jsonl.gz` — the self-contained benchmark (canonical)

One gzipped JSONL file, one JSON object per line, in the exact order of
`splits/main_900.txt`. Every record carries everything an evaluation needs, so
no research tree, no upstream corpus download, and no sentence splitter is
required to use it:

| field | meaning |
|---|---|
| `window_id`, `source`, `domain`, `range` | window identity, corpus, and context-length bucket |
| `tok_start`, `tok_end` | the window's token span in its dialogue |
| `history[]` | every prior turn, with `turn_index`, `time`, `speaker`, `dialogue` |
| `turns[]` | the current window's turns, same shape |
| `salient_utterances[]` | flattened WKF units: `sentence_numbers`, resolved `sentences`, `gap_type`, `vote_count` |
| `gaps[]` | `gap_idx`, `target_sentence_numbers`, resolved `target_sentence`, `gap_type`, `gap_reason`, and `evidence[]` |
| `gaps[].evidence[]` | `source_id`, the resolved `sentence`, `status`, `validation`, `vote_count`, per-model `votes` |

Released population: **900 windows, 1,498 gaps, 4,134 validated evidence links,
7,245 flattened salient utterances.**

### Coordinates and how the text was resolved

`target_sentence_numbers` are zero-based indices into the flattened current
window. `source_id` is `T{turn}-S{sentence}` with zero-based history turns and
one-based sentences.

The annotations were produced over two different segmentations, and this
release resolves each coordinate with the segmentation it was annotated under:

- current-window targets and salient utterances: NLTK punkt over the raw turn text;
- history evidence: emoji removal, then NLTK punkt, per turn.

Because the resolved text is baked into every record, consumers never have to
reproduce either splitter.

Two corrections were applied and are worth knowing about:

1. **Off-by-one evidence coordinates.** 132 validated coordinates are shifted by
   one sentence: the text the annotation recorded sits at `S+1`. This is not
   confined to literal `S0` indices — it occurs in whole turn blocks starting at
   any index. Each was corrected to `S+1` **only when the shifted coordinate
   resolves to the recorded text exactly**, so every correction is verified
   rather than inferred. No coordinate needed a shift in any other direction.
2. **One dropped evidence link.** `R_U0774` gap 0 `T728-S3` addresses a sentence
   that does not exist under any offset. It is the entire difference between this
   file's 4,134 evidence links and the v1 index's 4,135. Gaps are identical in
   both views (1,498).

After these corrections 4,103 of the 4,135 links carry text identical to what the
annotation recorded. The remaining 31 differ because the recorded string is a
model echo — markdown stripped, an `@all` prefix dropped, or truncated to `...` —
and the released text is the real dialogue sentence the coordinate points at.

The `sentence` strings recorded alongside coordinates in the upstream research
artifacts are model echoes — sometimes paraphrased, emoji-stripped, or truncated
to `...`. They were **not** used as released text; every released sentence comes
from resolving the coordinate against the dialogue.

## `annotations/remember_900_annotations.jsonl` — the v1 text-free index

The original coordinate-only release, kept byte-for-byte unchanged for
compatibility. It carries no dialogue, target, evidence, or rationale text: only
`window_id`, `range`, `source`, and per-gap coordinates, adjudication status,
vote counts, and votes. `schema.json` and `metadata.json` describe and hash it.
Its denominators remain 900 / 1,498 / 4,135.

Use the v1 index when you want to inspect annotation outcomes without
redistributing corpus content. Use `remember_900.jsonl.gz` when you want to run
or evaluate a system.

## What is redistributed

Be precise about this when reusing the payload: the annotations are derived
work, but the dialogue is not. `history[]` and `turns[]` carry the upstream
dialogue **verbatim** — 823,384 turns and roughly 192 million characters,
byte-identical to the source corpora apart from being cut into windows. What
this release adds on top is the annotation layer: gap and evidence coordinates,
adjudication votes, salient utterances, and the sentences those coordinates
resolve to.

The payload is redistributed under authorization covering EverMemBench, LoCoMo,
and RealTalk. The project-authored annotations are provided under the
repository's MIT License; that license does not itself grant rights to the
upstream corpora. If you redistribute this payload further, you are
redistributing those corpora's text, so review their terms first.

The v1 index in `annotations/` remains the option that carries no corpus text at
all.

## Caveats worth knowing

- `salient_utterances` is a **per-group flattening, not a set.** 188 of the
  7,245 entries repeat a coordinate already present in the same window, and 26
  of those repeats carry a different `gap_type`. A scorer that iterates the list
  weights such a sentence more than once; that is the paper's denominator.
- `gap_reason` is the **concatenated per-model rationale** carried over from
  annotation. It names the annotator models, can contain dissenting judgments,
  and is occasionally truncated. Treat it as provenance, not as a single
  adjudicated criterion.
- Punkt occasionally splits a numbered list into a fragment, so a handful of
  released sentences are as short as `"2."`.

## Provenance

`PROVENANCE.json` records the payload hash and counts, the v1 index and split
hashes, the SHA-256 of the three read-only inputs the payload was built from,
the two segmentations, and the coordinate corrections. `scripts/build_payload.py`
is the join that produced it.

## Reading it

```python
import gzip, json

with gzip.open("dataset/remember_900.jsonl.gz", "rt", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

assert len(rows) == 900
print(rows[0]["window_id"], len(rows[0]["history"]), len(rows[0]["gaps"]))
```
