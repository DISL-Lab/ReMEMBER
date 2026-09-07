"""End-to-end orchestration for the canonical public ReMEMber method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.resources import files
from time import perf_counter
from typing import Any, Sequence

from .config import RememberConfig
from .facet_turn import MemoryResult, _facet_turn_pack
from .gap_detection import _keywords, parse_predicted_gaps
from .inference import Cl100kTokenCounter, TextGenerator
from .retrieval import _hybrid_retrieve, build_token_chunks
from .schemas import Embedder, PredictedGap, RetrievedChunk, TokenCounter, Window
from .summarization import _parse_summary, _summarize, render_window


SCHEMA_VERSION = "1.1"
PACKAGE_ROOT = files("remember")
PROMPT_ROOT = PACKAGE_ROOT.joinpath("prompts")


@dataclass(frozen=True)
class RunResult:
    window_id: str
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
    return PROMPT_ROOT.joinpath(name).read_text(encoding="utf-8")


class _PrefixedEmbedder:
    def __init__(self, embedder: Embedder, prefix: str) -> None:
        self.embedder = embedder
        self.prefix = prefix

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self.embedder.encode([self.prefix + text for text in texts])


def _faithful_query(window: Window, gap: PredictedGap) -> str:
    window_keywords = " ".join(_keywords(render_window(window), max_terms=35))
    quote_keywords = " ".join(_keywords(gap.quote, max_terms=20))
    model_keywords = " ".join(gap.keywords[:24])
    return (
        "Find earlier dialogue that directly supports a faithful memory for the current window. "
        f"Current utterance: {gap.quote}. "
        f"Missing context: {gap.gap_reason}. "
        f"Needed evidence: {gap.evidence_need_query}. "
        f"Anchor terms: {model_keywords} {quote_keywords}. "
        f"Topic anchors: {window_keywords}."
    )


def run_window(
    window: Window,
    config: RememberConfig,
    generator: TextGenerator,
    embedder: Embedder,
) -> RunResult:
    """Execute the canonical method through the exact public brief interface."""
    return _run_window(
        window,
        config,
        generator,
        embedder,
        counter=Cl100kTokenCounter(),
        memory_only=False,
        gap_prompt=_read_prompt("gap_detection.txt"),
        summary_prompt_with_memory=_read_prompt("summary_with_memory.txt"),
        summary_prompt_without_memory=_read_prompt("summary_window_only.txt"),
    )


def _run_window(
    window: Window,
    config: RememberConfig,
    generator: TextGenerator,
    embedder: Embedder,
    *,
    counter: TokenCounter,
    memory_only: bool,
    gap_prompt: str,
    summary_prompt_with_memory: str,
    summary_prompt_without_memory: str,
) -> RunResult:
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
    query_embedder = _PrefixedEmbedder(embedder, config.dense_query_prefix)
    retrieved_lists = _hybrid_retrieve(
        gaps,
        chunks,
        embedder,
        query_texts=[_faithful_query(window, gap) for gap in gaps],
        query_embedder=query_embedder,
        top_k=config.top_k,
        rrf_k=config.rrf_k,
    )
    retrieval = {index: tuple(items) for index, items in retrieved_lists.items()}
    timings["retrieval"] = perf_counter() - started

    started = perf_counter()
    memory = _facet_turn_pack(
        window,
        gaps,
        retrieval,
        embedder,
        query_embedder,
        counter,
        config.memory_budget_tokens,
    )
    timings["packing"] = perf_counter() - started

    started = perf_counter()
    summary = "" if memory_only else _summarize(
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


def run_windows(
    windows: Sequence[Window],
    config: RememberConfig,
    generator: TextGenerator,
    embedder: Embedder,
    *,
    counter: TokenCounter | None = None,
    memory_only: bool = False,
    batch_size: int | None = None,
    gap_prompt: str | None = None,
    summary_prompt_with_memory: str | None = None,
    summary_prompt_without_memory: str | None = None,
) -> list[RunResult]:
    """Run a whole set of windows with gap detection and summarization each
    submitted as `generation_batch_size`-sized batches (matching the paper
    run), rather than one window at a time.

    `run_window` stays the exact single-window brief interface; this is the
    entry point the CLI uses so that a local vLLM generator sees the same
    batch composition the paper's runs did. Retrieval and packing are
    unaffected by this and stay per-window (batching them together across
    windows is not something the paper implementation does either).
    """
    counter = counter or Cl100kTokenCounter()
    batch_size = max(1, batch_size if batch_size is not None else config.generation_batch_size)
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

    windows = list(windows)
    n = len(windows)
    if n == 0:
        return []

    gap_timings = [0.0] * n
    gap_prompts = [
        gap_prompt.replace("{max_gaps}", str(config.max_gaps)).replace(
            "{window}", render_window(window, indexed=True)
        )
        for window in windows
    ]
    raw_gaps: list[str] = [""] * n
    for start in range(0, n, batch_size):
        batch = gap_prompts[start : start + batch_size]
        started = perf_counter()
        outputs = generator.generate_batch(
            batch, model=config.gap_model, temperature=config.gap_temperature,
            max_tokens=config.gap_max_tokens,
        )
        per_window = (perf_counter() - started) / len(batch)
        for offset, output in enumerate(outputs):
            raw_gaps[start + offset] = output
            gap_timings[start + offset] = per_window

    gaps_per_window = [
        tuple(parse_predicted_gaps(raw, window, config.max_gaps))
        for raw, window in zip(raw_gaps, windows)
    ]

    chunks_per_window = []
    retrieval_per_window = []
    memory_per_window = []
    query_embedder_per_window = []
    retrieval_packing_timings = [(0.0, 0.0)] * n
    for i, (window, gaps) in enumerate(zip(windows, gaps_per_window)):
        started = perf_counter()
        chunks = build_token_chunks(
            window.history, counter, max_tokens=config.chunk_tokens, overlap_tokens=config.overlap_tokens,
        )
        query_embedder = _PrefixedEmbedder(embedder, config.dense_query_prefix)
        retrieved_lists = _hybrid_retrieve(
            gaps, chunks, embedder,
            query_texts=[_faithful_query(window, gap) for gap in gaps],
            query_embedder=query_embedder, top_k=config.top_k, rrf_k=config.rrf_k,
        )
        retrieval = {index: tuple(items) for index, items in retrieved_lists.items()}
        retrieval_elapsed = perf_counter() - started

        started = perf_counter()
        memory = _facet_turn_pack(
            window, gaps, retrieval, embedder, query_embedder, counter, config.memory_budget_tokens,
        )
        packing_elapsed = perf_counter() - started

        chunks_per_window.append(chunks)
        retrieval_per_window.append(retrieval)
        memory_per_window.append(memory)
        query_embedder_per_window.append(query_embedder)
        retrieval_packing_timings[i] = (retrieval_elapsed, packing_elapsed)

    summary_timings = [0.0] * n
    summaries: list[str] = [""] * n
    if not memory_only:
        summary_prompts = []
        for window, memory in zip(windows, memory_per_window):
            template = summary_prompt_with_memory if memory.text.strip() else summary_prompt_without_memory
            values = {"window": render_window(window), "memory": memory.text}
            summary_prompts.append(template.format(**values))
        for start in range(0, n, batch_size):
            batch = summary_prompts[start : start + batch_size]
            started = perf_counter()
            outputs = generator.generate_batch(
                batch, model=config.summary_model, temperature=config.summary_temperature,
                max_tokens=config.summary_max_tokens,
            )
            per_window = (perf_counter() - started) / len(batch)
            for offset, output in enumerate(outputs):
                summaries[start + offset] = _parse_summary(output)
                summary_timings[start + offset] = per_window

    results: list[RunResult] = []
    for i, window in enumerate(windows):
        retrieval_elapsed, packing_elapsed = retrieval_packing_timings[i]
        timings = {
            "gap_detection": gap_timings[i],
            "retrieval": retrieval_elapsed,
            "packing": packing_elapsed,
            "summarization": summary_timings[i],
        }
        timings["total"] = sum(timings.values())
        results.append(RunResult(
            window_id=window.window_id,
            predicted_gaps=gaps_per_window[i],
            retrieval=retrieval_per_window[i],
            memory=memory_per_window[i],
            summary=summaries[i],
            token_counts={
                "history": counter.count("\n".join(chunk.text for chunk in chunks_per_window[i])),
                "window": counter.count(render_window(window)),
                "memory": memory_per_window[i].token_count,
            },
            timing_seconds={key: round(value, 6) for key, value in timings.items()},
            effective_config=asdict(config),
        ))
    return results
