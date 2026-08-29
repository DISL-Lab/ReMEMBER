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
| `salient_utterances[]` | flattened WKF units: `sentence_numbers`, `sentences`, `gap_type`, `vote_count` |
| `gaps[]` | `gap_idx`, `target_sentence_numbers`, `target_sentence` (a string), `gap_type`, `gap_reason`, and `evidence[]` |
| `gaps[].evidence[]` | `source_id`, `sentence`, `status`, `validation` (0 or 1), `vote_count`, per-model `votes` |

Released population: **900 windows, 1,498 gaps, 8,450 evidence candidates
(4,135 with `validation == 1`), 7,245 flattened salient utterances.**

### Design rule: publish exactly the strings the judges read

Every `sentence` in `gaps[].evidence[]` and every string in
`salient_utterances[].sentences` is the recorded annotation text, byte-identical
to what the paper's canonical evaluators consumed — never a paraphrase, and
never re-derived from a sentence-splitter coordinate. `target_sentence` is
likewise the annotation's own string.

### Evidence carries both accepted and rejected candidates — on purpose

`gaps[].evidence[]` includes every candidate the validators scored, tagged by
its `validation` value (0 = rejected, 1 = accepted), because MR and MKF read
this list differently:

- **MR** (memory recall) uses only `validation == 1` candidates.
- **MKF** (memory-knowledge faithfulness) uses every candidate with non-empty
  text, rejected ones included — its judge prompt shows the full pool the
  validators saw, not just what passed.

Every gap has at least one `validation == 1` candidate, so both metrics are
always computable. Filter by `validation` yourself if you only want the
accepted subset (4,135 candidates, exactly matching the v1 index below).

### `source_id` corrections

`source_id` is `T{turn}-S{sentence}`, zero-based history turns and one-based
sentences, over the segmentation the evidence coordinates were annotated
under (emoji removal, then NLTK punkt, per turn). 424 coordinates across the
full candidate pool were shifted by one sentence — the recorded text sits at
`S+1`, in whole turn blocks starting at any index, not only literal `S0`. A
coordinate was corrected only when the shifted value resolves to the recorded
text **exactly**, so every correction is verified, never inferred. The
correction only ever moves `source_id`; the shipped `sentence` text is always
the original recorded string, unchanged. 99 candidates' coordinates still do
not resolve under either offset — they ship with their original `source_id`
and their recorded text, unchanged; the coordinate is uncertain, the text is
not.

`gap_idx not resolving` and `evidence text not matching its own coordinate` are
therefore two different, unrelated things: consumers who don't care about
`source_id` provenance can ignore this section entirely.

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
- 11 `salient_utterances[].sentences` entries are **empty strings.** The
  canonical evaluator does not filter them either — it renders them as a real,
  necessarily-uncoverable prompt line and counts them in the WKF denominator.
- `gap_reason` is a **concatenation of per-model rationales**
  (`reason A. | reason B. | reason C.`), carried over from annotation and
  occasionally truncated or containing dissenting judgments. The leading
  `[model_name]` tag on each segment is stripped — the reasoning is released,
  the annotator model's identity is not. Treat it as provenance, not as a
  single adjudicated criterion.
- Gaps are filtered to a non-empty `target_sentence` (1,498 of 1,500 split RCV
  rows), matching the v1 index. MKF's own gap filter, read in isolation, is
  looser — it would include those 2 empty-target rows too — but this release
  keeps one gap set across MR, MKF, and v1 rather than a third, MKF-only
  population.

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
