# ReMEMber annotation-only dataset

This directory publishes the 900-window ReMEMber annotation release without
any dialogue, target-sentence, evidence-sentence, rationale, or other free
text. It is intended to make annotation outcomes inspectable without
redistributing source-corpus content.

`annotations/remember_900_annotations.jsonl` is the canonical payload. Each
line is one ordered window and has a stable `window_id`, source label,
context-length `range`, and its annotated gaps. A gap stores its source-row
`gap_idx`, target sentence coordinates (`target_sentence_numbers`), and
`gap_type`. Each retained evidence link stores only its source coordinate,
adjudication status, vote count, three model votes, and `validation=1`.

`splits/main_900.txt` is the ordered 900-window split. `window_id` is the join
key between split and annotation lines; `source_id` is a coordinate within the
upstream corpus and is not a public-content lookup service. `schema.json`
defines the allowlisted record structure. `metadata.json` records the exact
denominators and SHA-256 hashes of the distributed annotation, schema, and split files.

The paper denominator is exactly 900 windows, 1,498 non-empty target gaps,
and 4,135 evidence links with `validation=1`. These are release counts, not
counts for the larger candidate pool or for unvalidated evidence proposals.

Users are responsible for independently obtaining, reviewing, and complying
with the upstream corpus terms before joining these coordinates to any source
material. This release deliberately cannot reconstruct those texts. It also
does not convey target wording, evidence wording, annotator rationale, or
unvalidated links, so it is unsuitable for text-generation training or for
reproducing text-level judgments by itself.

License selection for this annotation-only release remains an unresolved
release gate. Do not treat this directory as licensed for redistribution until
the project owners add an explicit license decision.
