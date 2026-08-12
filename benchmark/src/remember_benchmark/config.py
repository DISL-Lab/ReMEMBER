"""Versioned benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowConfig:
    start_tokens: int = 7_000
    size_tokens: int = 1_024
    stride_tokens: int = 256

    def __post_init__(self) -> None:
        if self.start_tokens < 0:
            raise ValueError("start_tokens must be non-negative")
        if self.size_tokens <= 0:
            raise ValueError("size_tokens must be positive")
        if self.stride_tokens <= 0:
            raise ValueError("stride_tokens must be positive")


PAPER_SOURCE_CONFIG: dict[str, WindowConfig] = {
    "evermembench": WindowConfig(stride_tokens=1_024),
    "locomo": WindowConfig(stride_tokens=256),
    "realtalk": WindowConfig(stride_tokens=256),
}

RANGES: tuple[tuple[str, int, int], ...] = (
    ("8k~16k", 8_000, 16_000),
    ("16k~32k", 16_000, 32_000),
    ("32k~64k", 32_000, 64_000),
    ("64k~128k", 64_000, 128_000),
    ("128k~160k", 128_000, 160_000),
)
