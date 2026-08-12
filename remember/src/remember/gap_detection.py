"""Lenient parsing for target-free gap-detector responses."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable

from .schemas import PredictedGap, Window


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+./:-]*")
CODE_FENCE_RE = re.compile(r"```(?:json|jsonl)?\s*|```\s*", re.IGNORECASE)
MODEL_TAG_RE = re.compile(r"\s*\[[^\]]+\]\s*")
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "let", "may", "me", "my", "need", "needed", "needs", "no",
    "not", "now", "of", "on", "or", "our", "please", "regarding", "she",
    "should", "so", "that", "the", "their", "them", "there", "these", "they",
    "this", "those", "to", "we", "what", "when", "where", "which", "who",
    "will", "with", "would", "you", "your",
})


def _norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _token_f1(left: str, right: str) -> float:
    left_tokens, right_tokens = Counter(_tokenize(left)), Counter(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((left_tokens & right_tokens).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(left_tokens.values())
    recall = overlap / sum(right_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _keywords(text: str, max_terms: int = 28) -> list[str]:
    counts: Counter[str] = Counter()
    first_position: dict[str, int] = {}
    for position, token in enumerate(_tokenize(text)):
        if token in STOPWORDS or len(token) <= 2:
            continue
        counts[token] += 1
        first_position.setdefault(token, position)
    return sorted(counts, key=lambda item: (-counts[item], first_position[item], item))[:max_terms]


def _clean_reason(value: Any, max_chars: int = 420) -> str:
    text = _norm_space(value)
    text = MODEL_TAG_RE.sub(" ", text).replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0]
    return text


def _parse_utterance_index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value or ""))
        return int(match.group(0)) if match else None


def _canonicalize_quote(quote: str, utterance_index: int | None, window: Window) -> str:
    quote = _norm_space(quote).strip("\"'")
    if utterance_index is None or not 0 <= utterance_index < len(window.turns):
        return quote
    full_turn = _norm_space(window.turns[utterance_index].dialogue)
    if not full_turn:
        return quote
    if not quote:
        return full_turn
    compact_quote = _norm_space(quote.replace("...", " ").replace("…", " "))
    if ("..." in quote or "…" in quote) and _token_f1(compact_quote, full_turn) >= 0.25:
        return full_turn
    if quote.lower() not in full_turn.lower() and _token_f1(quote, full_turn) >= 0.55:
        return full_turn
    return quote


def _strip_fences(text: str) -> str:
    return CODE_FENCE_RE.sub("", text or "").strip()


def _complete_json_objects(text: str) -> list[str]:
    """Return balanced objects, including nested ones in a truncated wrapper."""
    objects: list[str] = []
    stack: list[int] = []
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
        elif character == '"':
            in_string = not in_string
        elif not in_string and character == "{":
            stack.append(index)
        elif not in_string and character == "}" and stack:
            start = stack.pop()
            objects.append(text[start : index + 1])
    return objects


def _parse_json_lenient(text: str) -> Any:
    text = _strip_fences(text)
    candidates = [text, *_complete_json_objects(text)]
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        for fixed in (candidate, re.sub(r",(\s*[}\]])", r"\1", candidate)):
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    return None


def _salvage_gap_objects(raw_output: str) -> list[dict[str, Any]]:
    salvaged: list[dict[str, Any]] = []
    for raw_object in _complete_json_objects(_strip_fences(raw_output)):
        object_value = _parse_json_lenient(raw_object)
        if not isinstance(object_value, dict):
            continue
        if any(object_value.get(key) for key in (
            "quote", "utterance_quote", "salient_utterance", "evidence_need_query"
        )):
            salvaged.append(object_value)
    return salvaged


def _window_keywords(window: Window) -> list[str]:
    rendered = "\n".join(
        f"[{turn.time}] {turn.speaker}: {turn.dialogue}" for turn in window.turns
    )
    return _keywords(rendered, max_terms=35)


def _merged_keywords(parts: Iterable[Any]) -> tuple[str, ...]:
    values: list[str] = []
    for part in parts:
        term = _norm_space(part).lower()
        if term and term not in STOPWORDS and term not in values:
            values.append(term)
    return tuple(values[:24])


def parse_predicted_gaps(
    raw_output: str,
    window: Window,
    max_gaps: int = 8,
    fallback: bool = True,
) -> list[PredictedGap]:
    """Normalize a gap-model response without consulting gold annotations.

    Complete nested objects are intentionally salvageable when model output has
    a truncated outer JSON wrapper.  Every resulting query uses only the model
    output and the current window.
    """
    if not 1 <= max_gaps <= 8:
        raise ValueError("max_gaps must be in 1..8")
    parsed = _parse_json_lenient(raw_output)
    if isinstance(parsed, dict):
        raw_gaps = parsed.get("gaps", [])
    elif isinstance(parsed, list):
        raw_gaps = parsed
    else:
        raw_gaps = []
    if not isinstance(raw_gaps, list) or not raw_gaps:
        raw_gaps = _salvage_gap_objects(raw_output)

    normalized: list[PredictedGap] = []
    seen_quotes: set[str] = set()
    for item in raw_gaps:
        if not isinstance(item, dict):
            continue
        utterance_index = _parse_utterance_index(item.get("utterance_index"))
        quote = _norm_space(
            item.get("quote") or item.get("utterance_quote") or item.get("salient_utterance")
        )
        if not quote and utterance_index is not None and 0 <= utterance_index < len(window.turns):
            quote = _norm_space(window.turns[utterance_index].dialogue)
        if not quote:
            continue
        quote = _canonicalize_quote(quote, utterance_index, window)
        if len(quote) > 360:
            quote = quote[:360].rsplit(" ", 1)[0]
        quote_key = quote.lower()
        if quote_key in seen_quotes:
            continue
        seen_quotes.add(quote_key)

        reason = _clean_reason(item.get("gap_reason") or item.get("reason") or "")
        query = _clean_reason(
            item.get("evidence_need_query") or item.get("query") or "", max_chars=520
        )
        if not query:
            query = (
                "retrieve earlier dialogue that can find the earlier evidence needed "
                f"to make the utterance self-contained for this utterance: {quote}"
            )
        supplied_keywords = item.get("keywords", [])
        if not isinstance(supplied_keywords, list):
            supplied_keywords = []
        normalized.append(PredictedGap(
            index=len(normalized),
            utterance_index=utterance_index,
            quote=quote,
            gap_reason=reason,
            evidence_need_query=query,
            keywords=_merged_keywords([
                *supplied_keywords, *_keywords(quote, 16), *_keywords(reason, 16), *_keywords(query, 16)
            ]),
            is_fallback=False,
        ))
        if len(normalized) >= max_gaps:
            break

    if normalized or not fallback:
        return normalized
    return [PredictedGap(
        index=0,
        utterance_index=None,
        quote="current window",
        gap_reason=(
            "The model did not identify a specific gap; retrieve earlier context related to the current window."
        ),
        evidence_need_query=(
            "retrieve earlier dialogue that provides background, definitions, decisions, "
            "owners, artifacts, versions, or rationale for the current window"
        ),
        keywords=tuple(_window_keywords(window)[:24]),
        is_fallback=True,
    )]
