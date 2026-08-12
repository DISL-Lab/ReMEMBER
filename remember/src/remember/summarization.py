"""Prompt rendering and robust summary extraction."""

from __future__ import annotations

import json
import re

from .facet_turn import MemoryResult
from .inference import TextGenerator
from .schemas import Window


def render_window(window: Window, *, indexed: bool = False) -> str:
    lines = []
    for index, turn in enumerate(window.turns):
        prefix = f"[U{index}] " if indexed else ""
        lines.append(f"{prefix}[{turn.time}] {turn.speaker}: {turn.dialogue}")
    return "\n".join(lines)


def _parse_summary(raw_output: str) -> str:
    cleaned = re.sub(r"```(?:json)?\s*|```\s*", "", raw_output or "", flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str):
            return parsed["summary"].strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r'["\']?summary["\']?\s*:\s*["\'](.*?)["\']\s*}', cleaned, re.DOTALL)
    if match:
        return match.group(1).replace('\\"', '"').strip()
    return cleaned if cleaned and not cleaned.startswith(("{", "[")) else ""


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
