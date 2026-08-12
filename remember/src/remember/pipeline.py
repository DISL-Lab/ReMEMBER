"""End-to-end orchestration for the canonical public ReMEMber method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import RememberConfig
from .facet_turn import MemoryResult, facet_turn_pack
from .gap_detection import parse_predicted_gaps
from .inference import Cl100kTokenCounter, TextGenerator
from .retrieval import build_token_chunks, hybrid_retrieve
from .schemas import Embedder, PredictedGap, RetrievedChunk, TokenCounter, Window
from .summarization import render_window, summarize


METHOD = "pred_gap_hybrid_facet_turn_pack1024"
SCHEMA_VERSION = "1.0"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = PACKAGE_ROOT / "prompts"


@dataclass(frozen=True)
class RunResult:
    window_id: str
    method: str
    predicted_gaps: tuple[PredictedGap, ...]
    retrieval: dict[int, tuple[RetrievedChunk, ...]]
    memory: MemoryResult
    summary: str
    token_counts: dict[str, int]
    timing_seconds: dict[str, float]
    effective_config: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "window_id": self.window_id,
            "method": self.method,
            "predicted_gaps": [asdict(gap) for gap in self.predicted_gaps],
            "retrieval": {
                str(gap_index): [asdict(item) for item in items]
                for gap_index, items in self.retrieval.items()
            },
            "memory": {
                "text": self.memory.text,
                "items": [asdict(item) for item in self.memory.items],
                "token_count": self.memory.token_count,
                "policy": self.memory.policy,
                "layout": self.memory.layout,
                "candidate_count": self.memory.candidate_count,
                "facet_count": self.memory.facet_count,
                "sentence_fallback_count": self.memory.sentence_fallback_count,
            },
            "summary": self.summary,
            "token_counts": self.token_counts,
            "timing_seconds": self.timing_seconds,
            "effective_config": self.effective_config,
        }


def _read_prompt(name: str) -> str:
    return (PROMPT_ROOT / name).read_text(encoding="utf-8")


def run_window(
    window: Window,
    config: RememberConfig,
    generator: TextGenerator,
    embedder: Embedder,
    *,
    counter: TokenCounter | None = None,
    memory_only: bool = False,
    gap_prompt: str | None = None,
    summary_prompt_with_memory: str | None = None,
    summary_prompt_without_memory: str | None = None,
) -> RunResult:
    counter = counter or Cl100kTokenCounter()
    gap_prompt = gap_prompt if gap_prompt is not None else _read_prompt("gap_detection.txt")
    summary_prompt_with_memory = (
        summary_prompt_with_memory
        if summary_prompt_with_memory is not None
        else _read_prompt("summary_with_memory.txt")
    )
    summary_prompt_without_memory = (
        summary_prompt_without_memory
        if summary_prompt_without_memory is not None
        else _read_prompt("summary_window_only.txt")
    )

    timings: dict[str, float] = {}
    started = perf_counter()
    prompt = gap_prompt.replace("{max_gaps}", str(config.max_gaps)).replace(
        "{window}", render_window(window, indexed=True)
    )
    raw_gaps = generator.generate(
        prompt,
        model=config.gap_model,
        temperature=config.gap_temperature,
        max_tokens=config.gap_max_tokens,
    )
    gaps = tuple(parse_predicted_gaps(raw_gaps, window, config.max_gaps))
    timings["gap_detection"] = perf_counter() - started

    started = perf_counter()
    chunks = build_token_chunks(
        window.history,
        counter,
        max_tokens=config.chunk_tokens,
        overlap_tokens=config.overlap_tokens,
    )
    retrieved_lists = hybrid_retrieve(
        gaps,
        chunks,
        embedder,
        top_k=config.top_k,
        rrf_k=config.rrf_k,
        dense_query_prefix=config.dense_query_prefix,
    )
    retrieval = {index: tuple(items) for index, items in retrieved_lists.items()}
    timings["retrieval"] = perf_counter() - started

    started = perf_counter()
    memory = facet_turn_pack(
        window,
        gaps,
        retrieval,
        embedder,
        counter,
        budget_tokens=config.memory_budget_tokens,
        dense_query_prefix=config.dense_query_prefix,
    )
    timings["packing"] = perf_counter() - started

    started = perf_counter()
    summary = "" if memory_only else summarize(
        window,
        memory,
        generator,
        summary_prompt_with_memory,
        summary_prompt_without_memory,
        model=config.summary_model,
        temperature=config.summary_temperature,
        max_tokens=config.summary_max_tokens,
    )
    timings["summarization"] = perf_counter() - started
    timings["total"] = sum(timings.values())

    return RunResult(
        window_id=window.window_id,
        method=METHOD,
        predicted_gaps=gaps,
        retrieval=retrieval,
        memory=memory,
        summary=summary,
        token_counts={
            "history": counter.count("\n".join(chunk.text for chunk in chunks)),
            "window": counter.count(render_window(window)),
            "memory": memory.token_count,
        },
        timing_seconds={key: round(value, 6) for key, value in timings.items()},
        effective_config=asdict(config),
    )
