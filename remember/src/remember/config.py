"""Validated, model-agnostic public configuration."""

from __future__ import annotations

from dataclasses import dataclass


DENSE_QUERY_PREFIX = (
    "Instruct: Given a current-window utterance with missing dialogue context, "
    "retrieve earlier dialogue that resolves the missing referent, artifact, "
    "decision, rationale, or prior state.\nQuery: "
)


class ConfigError(ValueError):
    """Raised when a public ReMEMber configuration is internally invalid."""


@dataclass(frozen=True)
class RememberConfig:
    gap_model: str = "Qwen/Qwen3.5-4B"
    summary_model: str = "Qwen/Qwen3.5-4B"
    dense_model: str = "Qwen/Qwen3-Embedding-0.6B"
    max_gaps: int = 8
    chunk_tokens: int = 128
    overlap_tokens: int = 32
    memory_budget_tokens: int = 1024
    top_k: int = 8
    rrf_k: int = 60
    gap_temperature: float = 0.0
    gap_max_tokens: int = 768
    summary_temperature: float = 0.7
    summary_max_tokens: int = 768
    token_encoding: str = "cl100k_base"
    dense_query_prefix: str = DENSE_QUERY_PREFIX

    def __post_init__(self) -> None:
        for name in (
            "gap_model", "summary_model", "dense_model", "token_encoding",
            "dense_query_prefix",
        ):
            if not getattr(self, name).strip():
                raise ConfigError(f"{name} must be a non-empty model ID")
        if not 1 <= self.max_gaps <= 8:
            raise ConfigError("max_gaps must be in 1..8")
        for name in (
            "chunk_tokens", "memory_budget_tokens", "top_k", "rrf_k",
            "gap_max_tokens", "summary_max_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive")
        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ConfigError("overlap_tokens must satisfy 0 <= overlap_tokens < chunk_tokens")
        for name in ("gap_temperature", "summary_temperature"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be non-negative")
