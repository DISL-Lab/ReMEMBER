# Dataset card: ReMEMber 900-window release

## Two views of one release

- **`remember_900.jsonl.gz` (canonical).** Self-contained: every record carries
  its own dialogue history, current window, gold gaps, resolved evidence
  sentences, and salient utterances. Population: 900 windows, 1,498 gaps, 4,132
  validated evidence links, 7,245 flattened salient utterances. This is what the
  evaluator reads.
- **`annotations/remember_900_annotations.jsonl` (v1 compatibility index).** The
  original coordinate-only view, byte-for-byte unchanged, described below.

The two differ by exactly three evidence links: one coordinate that addresses a
sentence which does not exist, and two that collided after `S0` evidence
coordinates were normalized to `S1`. Gaps are identical in both. See
[`README.md`](README.md) for how each coordinate was resolved to text.

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

## Access and joining

Acquiring source corpora is the downstream user's responsibility. `window_id`
joins a release line to an independently acquired evaluation window;
`source_id` identifies an upstream evidence coordinate. Neither key grants
access to source content, and this release contains no embedded text for such
a join.

## Limitations and responsible use

The removal of text means this package cannot support text reconstruction,
direct language-model training, or independent review of textual adequacy. It
also retains source-derived labels and model-vote metadata, which may reflect
annotation and model biases. Use it for the stated structural evaluation scope,
and respect each upstream source's terms when obtaining any source material.

## License

The project-authored annotation release is provided under the repository's MIT
License. Upstream corpora are not redistributed or relicensed here and remain
subject to their respective terms.
