"""Small, versioned schemas shared by benchmark stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "1.0"
SOURCE_PREFIX = {"evermembench": "E", "locomo": "L", "realtalk": "R"}


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class NormalizedTurn:
    time: str
    speaker: str
    dialogue: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _required_text(self.time, "time"))
        object.__setattr__(self, "speaker", _required_text(self.speaker, "speaker"))
        object.__setattr__(self, "dialogue", _required_text(self.dialogue, "dialogue"))

    def to_dict(self) -> dict[str, str]:
        return {"time": self.time, "speaker": self.speaker, "dialogue": self.dialogue}


@dataclass(frozen=True)
class NormalizedDialogue:
    schema_version: str
    dialogue_id: str
    source: str
    turns: tuple[NormalizedTurn, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        prefix = SOURCE_PREFIX.get(self.source)
        if prefix is None:
            raise ValueError(f"unknown source: {self.source!r}")
        if not isinstance(self.dialogue_id, str) or not self.dialogue_id.startswith(prefix + "_"):
            raise ValueError(f"dialogue_id must start with {prefix}_")
        if not self.turns:
            raise ValueError("turns must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dialogue_id": self.dialogue_id,
            "source": self.source,
            "turns": [turn.to_dict() for turn in self.turns],
        }
