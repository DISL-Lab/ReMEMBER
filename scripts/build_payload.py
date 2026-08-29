"""Build the single self-contained ReMEMber benchmark payload.

This is the join that defines the published benchmark. It is kept in the
repository so the artifact is reproducible and its rulings are auditable; it is
not part of the evaluation runtime and nothing imports it.

Inputs are supplied on the command line. The published payload was built from
these exact bytes:

    windows  sha256 867534724b6c73800e815f8e5d8e0c5426a39662c210c143fd7bece655cfe4ec
    rcv      sha256 3414a101cf9f99c76a5a2404da6296543d037de1871395334d0bb7a19ee8a41e
    salient  sha256 e65ed00720cb8bd6ce7e576f0e8f53c966423625da5ca6a8017158762994ebf2

Coordinates are the trustworthy anchor; the recorded `sentence` fields are model
echoes (paraphrased, truncated to '...', markdown or emoji stripped), so every
released sentence is resolved from its coordinate and baked into the record.

Two segmentations are reproduced exactly as the annotations were produced:
  * target/salient  - NLTK punkt over raw dialogue, flattened, zero-based.
  * evidence        - emoji removal then NLTK punkt, per turn, one-based.

A subset of validated evidence coordinates is shifted by one sentence: the
recorded text sits at `S+1`. This is not confined to literal `S0` indices - it
appears in whole turn blocks that happen to start at any index. Such a
coordinate is corrected to `S+1` only when the shifted coordinate resolves to
the recorded text *exactly*, so the correction is verified rather than guessed.
Coordinates that still do not resolve are dropped and reported.

A gap whose annotated `target_sentence` is empty is excluded, matching the rule
the v1 index was built with.
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


def split_raw(dialogue):
    return [s.strip() for s in sent_tokenize(dialogue.strip()) if s.strip()] or [dialogue]


def flat_window_sentences(turns):
    out = []
    for turn in turns:
        out.extend(split_raw(turn.get("dialogue", "")))
    return out


def history_index(history):
    index = {}
    for turn_index, turn in enumerate(history):
        stripped = remove_emojis(turn.get("dialogue", ""))
        for sentence_index, sentence in enumerate(
            [s.strip() for s in sent_tokenize(stripped) if s.strip()] or [stripped], 1
        ):
            index[f"T{turn_index}-S{sentence_index}"] = sentence
    return index


def normalized_turns(turns):
    return [
        {"turn_index": i, "time": str(t.get("time", "")), "speaker": str(t.get("speaker", "")),
         "dialogue": str(t.get("dialogue", ""))}
        for i, t in enumerate(turns or [])
    ]


def _shift_sentence(source_id):
    """Return the same coordinate one sentence later, or None if unparsable."""
    if not isinstance(source_id, str) or "-S" not in source_id:
        return None
    turn, _, sentence = source_id.rpartition("-S")
    if not sentence.isdigit():
        return None
    return f"{turn}-S{int(sentence) + 1}"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--rcv", required=True)
    ap.add_argument("--salient", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split_ids = [l.strip() for l in Path(args.split).read_text(encoding="utf-8").splitlines() if l.strip()]
    split_set = set(split_ids)

    salient_by_id = {r["window_id"]: r for r in read_jsonl(args.salient) if r.get("window_id") in split_set}
    rcv_by_id = {}
    for row in read_jsonl(args.rcv):
        if row.get("window_id") in split_set:
            rcv_by_id.setdefault(row["window_id"], []).append(row)

    windows = {}
    for row in read_jsonl(args.source):
        if row.get("window_id") in split_set:
            windows[row["window_id"]] = row
    missing = [w for w in split_ids if w not in windows]
    if missing:
        sys.exit(f"source is missing {len(missing)} split windows, e.g. {missing[:3]}")

    stats = {"windows": 0, "gaps": 0, "evidence": 0, "salient_utterances": 0}
    dropped = []
    shifted_count = [0]
    records = []
    for window_id in split_ids:
        raw = windows[window_id]
        expected_source = SOURCE_BY_PREFIX[window_id[0]]
        history = normalized_turns(raw.get("history"))
        turns = normalized_turns(raw.get("turns"))
        flat = flat_window_sentences(raw.get("turns") or [])
        index = history_index(raw.get("history") or [])

        salient = []
        srow = salient_by_id.get(window_id, {})
        for group in (srow.get("gap_analysis") or srow.get("gaps") or []):
            numbers = group.get("sentence_numbers") or group.get("target_sentence_numbers") or []
            for number in numbers:
                if not isinstance(number, int) or not 0 <= number < len(flat):
                    dropped.append((window_id, "salient", number)); continue
                salient.append({
                    "salient_idx": len(salient),
                    "sentence_numbers": [number],
                    "sentences": [flat[number]],
                    "gap_type": group.get("gap_type"),
                    "vote_count": group.get("vote_count"),
                })

        # combined_sampled.jsonl carries no `range`; the RCV rows do, and every
        # row for one window must agree (the same rule sanitize_annotations uses).
        window_ranges = {r.get("range") for r in rcv_by_id.get(window_id, [])}
        if len(window_ranges) != 1:
            sys.exit(f"{window_id}: inconsistent or missing RCV range: {sorted(window_ranges)}")
        window_range = window_ranges.pop()

        gaps = []
        for row in sorted(rcv_by_id.get(window_id, []), key=lambda r: r.get("gap_idx", 0)):
            # Same rule the published v1 index used (release.sanitize_annotations):
            # a gap whose annotated target text is empty is not part of the benchmark.
            if not row.get("target_sentence"):
                continue
            numbers = row.get("sentence_numbers") or row.get("target_sentence_numbers") or []
            target = []
            for number in numbers:
                if isinstance(number, int) and 0 <= number < len(flat):
                    target.append(flat[number])
            if not target:
                continue
            evidence = []
            seen = set()
            for item in row.get("validated", []):
                if item.get("validation") != 1:
                    continue
                source_id = item.get("source_id")
                resolved_id = source_id
                recorded = item.get("sentence")
                shifted = _shift_sentence(source_id)
                # Adopt the +1 correction only when it lands on the recorded text
                # exactly; otherwise the original coordinate stands.
                if (shifted is not None and index.get(resolved_id) != recorded
                        and index.get(shifted) == recorded):
                    resolved_id = shifted
                    shifted_count[0] += 1
                sentence = index.get(resolved_id)
                if sentence is None or resolved_id in seen:
                    dropped.append((window_id, "evidence", source_id)); continue
                seen.add(resolved_id)
                evidence.append({
                    "source_id": resolved_id,
                    "sentence": sentence,
                    "status": item.get("status"),
                    "validation": item.get("validation"),
                    "vote_count": item.get("vote_count"),
                    "votes": item.get("votes"),
                })
            gaps.append({
                "gap_idx": row.get("gap_idx"),
                "target_sentence_numbers": numbers,
                "target_sentence": target,
                "gap_type": row.get("gap_type"),
                "gap_reason": row.get("gap_reason", ""),
                "evidence": evidence,
            })
            stats["evidence"] += len(evidence)
        stats["gaps"] += len(gaps)
        stats["salient_utterances"] += len(salient)
        stats["windows"] += 1
        records.append({
            "schema_version": "2.0", "window_id": window_id,
            "source": expected_source, "domain": expected_source,
            "range": window_range, "tok_start": raw.get("tok_start"), "tok_end": raw.get("tok_end"),
            "history": history, "turns": turns,
            "salient_utterances": salient, "gaps": gaps,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename="", mode="wb", fileobj=out.open("wb"), mtime=0, compresslevel=9) as gz:
        for record in records:
            gz.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            gz.write(b"\n")
    print(json.dumps({"counts": stats, "shifted": shifted_count[0], "dropped": len(dropped),
                      "bytes": out.stat().st_size}, indent=2))
    for item in dropped[:20]:
        print("  dropped:", item)


if __name__ == "__main__":
    main()
