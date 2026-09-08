# ReMEMber dataset

This directory publishes two views of the same 900-window release.

## The self-contained benchmark (canonical)

Three gzipped JSONL files, across two conversational domains, together holding
all 900 windows in the exact order of `splits/main_900.txt`:

| file | domain | windows |
|---|---|---|
| `remember_900_chitchat.jsonl.gz` | LoCoMo + RealTalk (casual conversation) | 300 |
| `remember_900_workplace_1.jsonl.gz` | EverMemBench (professional conversation) | 300 |
| `remember_900_workplace_2.jsonl.gz` | EverMemBench (professional conversation) | 300 |

The domain split follows the underlying source corpora, not an arbitrary cut:
each record's own `source`/`domain` field still names its corpus
(`evermembench`/`locomo`/`realtalk`) unchanged. EverMemBench alone is 600 of
900 windows (~64 MB compressed), so it is further split by count into two
files, in order, purely to keep every file well under GitHub's recommended
file-size warning — that split carries no semantic meaning. `chitchat`'s
windows precede `workplace_1`'s, which precede `workplace_2`'s, in
`splits/main_900.txt`, so loading all three files in filename (alphabetical)
order and concatenating reconstructs the canonical order exactly —
`evaluation/evaluate.py --benchmark dataset` does this automatically.

Every record carries everything an evaluation needs, so no research tree, no
upstream corpus download, and no sentence splitter is required to use it:

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
redistributing corpus content. Use the three `remember_900_*.jsonl.gz` shards
when you want to run or evaluate a system.

## What is redistributed

Be precise about this when reusing the payload: the annotations are derived
work, but the dialogue is not. `history[]` and `turns[]` carry the upstream
dialogue **verbatim** — 823,384 turns and roughly 192 million characters,
byte-identical to the source corpora apart from being cut into windows. What
this release adds on top is the annotation layer: gap and evidence coordinates,
adjudication votes, salient utterances, and the sentences those coordinates
resolve to.

The project-authored annotations (gap/evidence coordinates, adjudication
votes, salient utterances) are provided under the repository's MIT License;
that license does not itself grant rights to the upstream dialogue text,
which each source corpus governs separately:

| source | upstream license | terms |
|---|---|---|
| EverMemBench (`workplace_1`, `workplace_2`) | [Apache-2.0](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic) | permissive; redistribution and commercial use are permitted |
| LoCoMo (part of `chitchat`) | [CC BY-NC 4.0](https://github.com/snap-research/locomo/blob/main/LICENSE.txt) | non-commercial use only, with attribution — cite [Maharana et al., 2024](#citations) |
| RealTalk (part of `chitchat`) | no public license file | redistributed here with the authors' direct permission, conditioned on citation — cite [Lee et al., 2025](#citations) |

RealTalk's own repository (<https://github.com/danny911kr/REALTALK>) ships no
LICENSE file and states no public redistribution terms; the authors' stated
condition for reuse, communicated directly, is citation. It contains
real participants' dialogue (and, in the authors' own repository, photos and
demographic details) collected under a study consent process, not a public
open-data license — treat it accordingly if you redistribute this payload
further, beyond what citation covers.

### Citations

```bibtex
@inproceedings{maharana2024evaluating,
  title     = {Evaluating Very Long-Term Conversational Memory of {LLM} Agents},
  author    = {Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {13851--13870},
  year      = {2024},
  address   = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics}
}

@article{lee2025realtalk,
  title   = {Realtalk: A 21-day real-world dataset for long-term conversation},
  author  = {Lee, Dong-Ho and Maharana, Adyasha and Pujara, Jay and Ren, Xiang and Barbieri, Francesco},
  journal = {arXiv preprint arXiv:2502.13270},
  year    = {2025}
}
```

EverMemBench is Apache-2.0 and does not require citation, but see its
[dataset card](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic)
if you want one.

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
is the join that produced it — an internal build script, not part of this
release.

## Reading it

```python
import gzip, json

def read_shard(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

rows = (read_shard("dataset/remember_900_chitchat.jsonl.gz")
        + read_shard("dataset/remember_900_workplace_1.jsonl.gz")
        + read_shard("dataset/remember_900_workplace_2.jsonl.gz"))

assert len(rows) == 900
print(rows[0]["window_id"], len(rows[0]["history"]), len(rows[0]["gaps"]))
```
