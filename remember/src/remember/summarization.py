"""Prompt rendering and robust summary extraction."""

from __future__ import annotations

import ast
import json
import re

from .facet_turn import MemoryResult
from .inference import TextGenerator
from .schemas import Window

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+./:-]*")
_FENCE_RE = re.compile(r"```(?:json)?\s*|```\s*", re.IGNORECASE)


def render_window(window: Window, *, indexed: bool = False) -> str:
    lines = []
    for index, turn in enumerate(window.turns):
        prefix = f"[U{index}] " if indexed else ""
        lines.append(f"{prefix}[{turn.time}] {turn.speaker}: {turn.dialogue}")
    return "\n".join(lines)


def _parse_summary(raw_output: str) -> str:
    """Five-tier lenient extraction, matching `predicted_evidence_need_memory.py`'s
    canonical `parse_summary`: whole-object JSON, embedded-object JSON, Python
    literal, a quoted-string regex, an escaped-quote-continuation regex, and
    finally a raw-text fallback gated on a minimum token count so short
    garbage isn't accepted as a summary."""
    output = raw_output or ""

    cleaned = _FENCE_RE.sub("", output).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str):
            return parsed["summary"].strip()
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*?"summary".*?\}', output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str):
                return parsed["summary"].strip()
        except json.JSONDecodeError:
            pass

    try:
        parsed = ast.literal_eval(output)
        if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str):
            return parsed["summary"].strip()
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass

    match = re.search(r'["\']?summary["\']\s*:\s*"(.*)', output, re.DOTALL)
    if match:
        text = match.group(1)
        if text.endswith('"}'):
            text = text[:-2]
        elif text.endswith('"'):
            text = text[:-1]
        return text.strip()

    match = re.search(r'["\']?summary["\']\s*:\s*\\+"(.*)', output, re.DOTALL)
    if match:
        text = re.split(r'\\+"(?:\s*})?\s*$', match.group(1), maxsplit=1)[0]
        return text.replace('\\"', '"').strip()

    if cleaned and not cleaned.startswith(("{", "[")) and len(_TOKEN_RE.findall(cleaned)) >= 6:
        return cleaned
    return ""


def summarize(
    window: Window,
    memory: MemoryResult,
    generator: TextGenerator,
    prompt_with_memory: str,
    prompt_without_memory: str,
) -> str:
    """Summarize through the exact public interface from the release brief."""
    return _summarize(
        window,
        memory,
        generator,
        prompt_with_memory,
        prompt_without_memory,
        model=None,
        temperature=0.7,
        max_tokens=768,
    )


def _summarize(
    window: Window,
    memory: MemoryResult,
    generator: TextGenerator,
    prompt_with_memory: str,
    prompt_without_memory: str,
    *,
    model: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    template = prompt_with_memory if memory.text.strip() else prompt_without_memory
    values = {"window": render_window(window), "memory": memory.text}
    raw = generator.generate(
        template.format(**values),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _parse_summary(raw)
