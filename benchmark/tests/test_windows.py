from remember_benchmark.config import PAPER_SOURCE_CONFIG, WindowConfig
from remember_benchmark.schemas import NormalizedDialogue, NormalizedTurn
from remember_benchmark.windows import assign_range, build_windows


class WordCounter:
    def count(self, text: str) -> int:
        return 1


def _dialogue(source: str = "evermembench") -> NormalizedDialogue:
    prefix = {"evermembench": "E", "locomo": "L", "realtalk": "R"}[source]
    turns = tuple(
        NormalizedTurn(time=f"2024-01-01 00:00:0{i}", speaker="A", dialogue=f"word{i}")
        for i in range(7)
    )
    return NormalizedDialogue("1.0", f"{prefix}_topic-1_group-1", source, turns)


def test_paper_defaults_encode_source_specific_stride():
    assert PAPER_SOURCE_CONFIG["evermembench"] == WindowConfig(start_tokens=7000, size_tokens=1024, stride_tokens=1024)
    assert PAPER_SOURCE_CONFIG["locomo"] == WindowConfig(start_tokens=7000, size_tokens=1024, stride_tokens=256)
    assert PAPER_SOURCE_CONFIG["realtalk"] == WindowConfig(start_tokens=7000, size_tokens=1024, stride_tokens=256)


def test_window_ids_boundaries_and_history_are_literal_and_stable():
    config = WindowConfig(start_tokens=2, size_tokens=2, stride_tokens=2)
    windows = build_windows([_dialogue()], WordCounter(), configs={"evermembench": config})

    assert [row["window_id"] for row in windows] == ["E_U0000", "E_U0001"]
    assert [(row["tok_start"], row["tok_end"]) for row in windows] == [(2, 4), (4, 6)]
    assert [turn["dialogue"] for turn in windows[0]["history"]] == ["word0", "word1"]
    assert [turn["dialogue"] for turn in windows[0]["turns"]] == ["word2", "word3"]
    assert windows[0]["n_history"] == 2
    assert windows[0]["n_turns"] == 2


def test_ids_are_independent_stable_counters_per_source():
    config = WindowConfig(start_tokens=2, size_tokens=2, stride_tokens=2)
    configs = {name: config for name in ("evermembench", "locomo", "realtalk")}
    windows = build_windows([_dialogue("realtalk"), _dialogue("locomo"), _dialogue("evermembench")], WordCounter(), configs=configs)
    assert [row["window_id"] for row in windows] == [
        "E_U0000", "E_U0001", "L_U0000", "L_U0001", "R_U0000", "R_U0001"
    ]


def test_range_assignment_keeps_other_explicitly():
    assert assign_range(7_999) == "other"
    assert assign_range(8_000) == "8k~16k"
    assert assign_range(15_999) == "8k~16k"
    assert assign_range(16_000) == "16k~32k"
    assert assign_range(159_999) == "128k~160k"
    assert assign_range(160_000) == "other"


def test_windows_keep_other_instead_of_dropping_them():
    config = WindowConfig(start_tokens=2, size_tokens=2, stride_tokens=2)
    windows = build_windows([_dialogue()], WordCounter(), configs={"evermembench": config})
    assert [row["range"] for row in windows] == ["other", "other"]
