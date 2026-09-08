# Dataset card: ReMEMber 900-window release

## Two views of one release

- **Three `remember_900_*.jsonl.gz` shards (canonical).** Self-contained: every
  record carries its own dialogue history, current window, gold gaps, every
  evidence candidate (accepted and rejected, tagged by `validation`), and
  salient utterances. Split by conversational domain — `chitchat` (LoCoMo +
  RealTalk, 300 windows) and `workplace_1` + `workplace_2` (EverMemBench, 300
  windows each, split by count) — rather than shipped as one file, so each
  stays under GitHub's file-size warning; each record's own `source`/`domain`
  field is unaffected. Population: 900 windows, 1,498 gaps, 8,450 evidence
  candidates (4,135 with `validation == 1`), 7,245 flattened salient
  utterances. Every string is the recorded annotation text, never re-derived
  from a coordinate. This is what the evaluator reads; MR and MKF each apply
  their own filter to the same evidence list.
- **`annotations/remember_900_annotations.jsonl` (v1 compatibility index).** The
  original coordinate-only view, byte-for-byte unchanged, described below.

The gap set is identical between the two. The `validation == 1` subset of the
canonical payload's evidence exactly matches v1's 4,135 evidence links (v1
predates the `source_id` corrections and is frozen, so a handful of individual
coordinates differ while the set of validated candidates is the same). See
[`README.md`](README.md) for the coordinate corrections and the MR/MKF filter
asymmetry.

The canonical payload carries upstream dialogue verbatim (823,384 turns,
~192M characters, byte-identical to the source apart from windowing), not a
paraphrased or otherwise derived form. It is redistributed under authorization
covering EverMemBench, LoCoMo, and RealTalk; redistributing it further means
redistributing those corpora's text.

## Summary (v1 compatibility index)

This is a coordinate-and-label release for the ReMEMber 900-window evaluation
set. It exposes structural annotation outcomes while removing all free text
from the original RCV records and source corpora.

## Contents and fields

Each JSONL record represents one window:

- `schema_version`: release schema identifier (`1.0`).
- `window_id`: stable evaluation-window join key.
- `source`: source-family label derived from the ID (`evermembench`, `locomo`,
  or `realtalk`).
- `range`: original context-length bucket.
- `gaps`: annotation rows for the window.

Each gap contains its index, target sentence number coordinates, label type,
and only adjudicated evidence coordinates. Every evidence object includes a
source coordinate, adjudication status, vote count, per-model votes, and
`validation=1`.

## Data construction and denominators

The exporter consumes the canonical filtered RCV JSONL and the ordered main
split. It retains only split members with a non-empty target sentence and
evidence entries with `validation == 1`, then allowlists the published fields.
The final release contains 900 windows, 1,498 gaps, and 4,135 validated
evidence links. These denominators apply to the final ordered release only;
they do not describe the source corpus, all sampled windows, or all candidate
evidence links.

## Access and joining (v1 compatibility index only)

`annotations/remember_900_annotations.jsonl` embeds no text. Acquiring source
corpora is the downstream user's responsibility when working from that file:
`window_id` joins a line to an independently acquired evaluation window, and
`source_id` identifies an upstream evidence coordinate. Neither key grants
access to source content.

This does not apply to the three `remember_900_*.jsonl.gz` shards, which carry
the dialogue itself and need no external join.

## Limitations and responsible use

The v1 index carries no text, so on its own it cannot support text
reconstruction, direct language-model training, or independent review of
textual adequacy; the canonical payload can, and is redistributed on that
understanding.

Both views retain source-derived labels and model-vote metadata, which may
reflect annotation and model biases. `gap_reason` is a concatenation of
per-model rationales carried over from annotation and can contain dissenting
judgments; the leading `[model_name]` tag on each segment is stripped, so the
reasoning is released but the annotator model's identity is not. Treat it as
provenance, not as a single adjudicated criterion. The canonical payload's
evidence includes rejected
(`validation == 0`) candidates alongside accepted ones, because MKF's judge
prompt reads the full validated pool; see `README.md` for which metric uses
which subset.

## License

The project-authored annotations are provided under the repository's MIT
License. That license does not cover the upstream corpora.

- `annotations/remember_900_annotations.jsonl` redistributes no upstream text.
- `remember_900_chitchat.jsonl.gz`, `remember_900_workplace_1.jsonl.gz`, and
  `remember_900_workplace_2.jsonl.gz` **do** redistribute upstream dialogue
  verbatim. Redistributing any of them further means redistributing those
  corpora's text, subject to each source's own terms:

  | source | terms |
  |---|---|
  | EverMemBench | [Apache-2.0](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic) |
  | LoCoMo | [CC BY-NC 4.0](https://github.com/snap-research/locomo/blob/main/LICENSE.txt) — non-commercial, attribution required |
  | RealTalk | no public license file; redistributed here under the authors' direct permission, conditioned on citation |

  RealTalk in particular ships no LICENSE and no stated public redistribution
  terms in its own repository — its authors' condition for reuse, given
  directly, is citation, not an open license. It contains real participants'
  dialogue collected under a study consent process. See
  [`README.md`](README.md#citations) for the citations both LoCoMo and RealTalk
  ask for.
