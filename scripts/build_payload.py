"""Build the single self-contained ReMEMber benchmark payload.

This is the join that defines the published benchmark. It is kept in the
repository so the artifact is reproducible and its rulings are auditable; it is
not part of the evaluation runtime and nothing imports it.

Inputs are supplied on the command line. The published payload was built from
these exact bytes:

    windows  sha256 867534724b6c73800e815f8e5d8e0c5426a39662c210c143fd7bece655cfe4ec
    rcv      sha256 3414a101cf9f99c76a5a2404da6296543d037de1871395334d0bb7a19ee8a41e
    salient  sha256 e65ed00720cb8bd6ce7e576f0e8f53c966423625da5ca6a8017158762994ebf2

Design rule: publish the strings the judges actually read.

The canonical evaluators consume the annotation's recorded text directly:
MKF's canonical coverage-evaluator function `format_relevant_chunk_from_validated`
joins every `validated[].sentence` with no `validation` filter; WKF's
`format_salient_utterances` reads `sentences` from salient.jsonl verbatim.
Neither re-derives text from a coordinate, so this builder does not either -
every evidence and salient sentence ships exactly as recorded.

The one place a coordinate is trusted over recorded text is `source_id`: a
subset of evidence coordinates is shifted by one sentence (the recorded text
sits at `S+1`, in whole turn blocks starting at any index), and is corrected to
`S+1` only when the shifted coordinate resolves to the recorded text exactly.
This never changes the shipped `sentence`, only which coordinate it is filed
under, so it does not affect what a judge reads.

MR's canonical loader `load_pw_from_rcv` and MKF read the same
`validated` list differently: MR keeps only `validation == 1` items, MKF keeps
every item with non-empty text. Both are shipped - every validated candidate,
tagged with its `validation` value - so the evaluator's two renderers can each
apply their own filter without a second source file.

Gaps are filtered to a non-empty `target_sentence`, matching
`release.sanitize_annotations` and the checked-in v1 index (1,498 of the 1,500
split RCV rows). MKF's own gap filter is looser in isolation (any row with
non-empty `validated`, which literally includes the 2 empty-target rows), but
this release keeps one gap set across MR, MKF, and the v1 index rather than a
third, MKF-only population.
"""
import argparse, gzip, json, re, sys
from pathlib import Path
from nltk.tokenize import sent_tokenize

# Verbatim copy of the emoji pattern the annotation pipeline used.
_EMOJI_RE = re.compile("[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF" "\U00002600-\U000027BF" "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF" "\U00002702-\U000027B0" "\U000024C2-\U0001F251" "]+", flags=re.UNICODE)

SOURCE_BY_PREFIX = {"E": "evermembench", "L": "locomo", "R": "realtalk"}


def remove_emojis(text):
    return _EMOJI_RE.sub("", text).strip()


def history_index(history):
    """Reproduce the segmentation evidence coordinates were assigned under.

    Used only to verify a `source_id` shift, never to replace shipped text.
    """
    index = {}
    for turn_index, turn in enumerate(history):
        stripped = remove_emojis(turn.get("dialogue", ""))
        sentences = [s.strip() for s in sent_tokenize(stripped) if s.strip()] or [stripped]
        for sentence_index, sentence in enumerate(sentences, 1):
            index[f"T{turn_index}-S{sentence_index}"] = sentence
    return index


def shift_sentence(source_id):
    """Return the same coordinate one sentence later, or None if unparsable."""
    if not isinstance(source_id, str) or "-S" not in source_id:
        return None
    turn, _, sentence = source_id.rpartition("-S")
    return f"{turn}-S{int(sentence) + 1}" if sentence.isdigit() else None


def normalized_turns(turns):
    return [
        {"turn_index": i, "time": str(t.get("time", "")), "speaker": str(t.get("speaker", "")),
         "dialogue": str(t.get("dialogue", ""))}
        for i, t in enumerate(turns or [])
    ]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    for name in ("source", "rcv", "salient", "split", "out"):
        ap.add_argument(f"--{name}", required=True)
    args = ap.parse_args()

    split_ids = [l.strip() for l in Path(args.split).read_text(encoding="utf-8").splitlines() if l.strip()]
    split_set = set(split_ids)

    salient_by_id = {r["window_id"]: r for r in read_jsonl(args.salient) if r.get("window_id") in split_set}
    rcv_by_id = {}
    for row in read_jsonl(args.rcv):
        if row.get("window_id") in split_set:
            rcv_by_id.setdefault(row["window_id"], []).append(row)

    windows = {r["window_id"]: r for r in read_jsonl(args.source) if r.get("window_id") in split_set}
    missing = [w for w in split_ids if w not in windows]
    if missing:
        sys.exit(f"source is missing {len(missing)} split windows, e.g. {missing[:3]}")

    stats = {"windows": 0, "gaps": 0, "candidates": 0, "candidates_validated": 0,
             "salient_utterances": 0}
    shifted = unresolved = excluded_empty_target = 0
    records = []
    for window_id in split_ids:
        raw = windows[window_id]
        expected_source = SOURCE_BY_PREFIX[window_id[0]]
        index = history_index(raw.get("history") or [])

        rows = sorted(rcv_by_id.get(window_id, []), key=lambda r: r.get("gap_idx", 0))
        window_ranges = {r.get("range") for r in rows}
        if len(window_ranges) != 1:
            sys.exit(f"{window_id}: inconsistent or missing RCV range: {sorted(window_ranges)}")

        gaps = []
        for row in rows:
            # One gap set across MR, MKF, and the v1 index: exclude rows the
            # annotation never assigned a target sentence to.
            if not row.get("target_sentence"):
                excluded_empty_target += 1
                continue

            candidates = []
            for item in row.get("validated", []):
                source_id = item.get("source_id")
                recorded = item.get("sentence")
                if not recorded:
                    continue
                candidate = shift_sentence(source_id)
                if (candidate is not None and index.get(source_id) != recorded
                        and index.get(candidate) == recorded):
                    source_id = candidate
                    shifted += 1
                elif index.get(source_id) != recorded:
                    unresolved += 1
                candidates.append({
                    "source_id": source_id,
                    "sentence": recorded,
                    "status": item.get("status"),
                    "validation": item.get("validation"),
                    "vote_count": item.get("vote_count"),
                    "votes": item.get("votes"),
                })
            gaps.append({
                "gap_idx": row.get("gap_idx"),
                "gap_type": row.get("gap_type"),
                "target_sentence_numbers": row.get("sentence_numbers")
                                           or row.get("target_sentence_numbers") or [],
                "target_sentence": row.get("target_sentence", ""),
                "gap_reason": row.get("gap_reason", ""),
                "evidence": candidates,
            })
            stats["candidates"] += len(candidates)
            stats["candidates_validated"] += sum(1 for c in candidates if c["validation"] == 1)

        salient = []
        srow = salient_by_id.get(window_id, {})
        for group in (srow.get("gap_analysis") or srow.get("gaps") or []):
            numbers = group.get("sentence_numbers") or group.get("target_sentence_numbers") or []
            sentences = group.get("sentences") or group.get("target_sentence") or []
            if isinstance(sentences, str):
                sentences = [sentences]
            for position, sentence in enumerate(sentences):
                # The canonical coverage evaluator's format_salient_utterances does not
                # filter empty text either; it renders "U{i}: " as a real prompt line
                # and counts it in the denominator, so this ships unfiltered too.
                salient.append({
                    "salient_idx": len(salient),
                    "sentence_numbers": [numbers[position]] if position < len(numbers) else [],
                    "sentences": [sentence],
                    "gap_type": group.get("gap_type"),
                    "vote_count": group.get("vote_count"),
                })

        stats["gaps"] += len(gaps)
        stats["salient_utterances"] += len(salient)
        stats["windows"] += 1
        records.append({
            "schema_version": "2.0", "window_id": window_id,
            "source": expected_source, "domain": expected_source,
            "range": window_ranges.pop() if window_ranges else None,
            "tok_start": raw.get("tok_start"), "tok_end": raw.get("tok_end"),
            "history": normalized_turns(raw.get("history")),
            "turns": normalized_turns(raw.get("turns")),
            "salient_utterances": salient, "gaps": gaps,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename="", mode="wb", fileobj=out.open("wb"), mtime=0, compresslevel=9) as gz:
        for record in records:
            gz.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")).encode("utf-8"))
            gz.write(b"\n")
    print(json.dumps({
        "counts": stats, "gaps_excluded_empty_target": excluded_empty_target,
        "source_ids_shifted": shifted, "source_ids_not_matching_recorded_text": unresolved,
        "bytes": out.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
