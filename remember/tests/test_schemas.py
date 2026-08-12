from pathlib import Path

import pytest

from remember.config import ConfigError, RememberConfig
from remember.io import InputError, atomic_write_jsonl, load_windows
from remember.schemas import Turn, Window


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_windows_returns_frozen_normalized_dataclasses():
    [window] = list(load_windows(FIXTURES / "windows.jsonl"))

    assert window == Window(
        window_id="demo-1",
        history=(
            Turn(time="09:00", speaker="Ava", dialogue="The cedar proposal needs a decision."),
            Turn(time="09:01", speaker="Bo", dialogue="I will prepare the budget."),
        ),
        turns=(
            Turn(time="10:00", speaker="Ava", dialogue="Use the revised plan."),
            Turn(time="10:01", speaker="Bo", dialogue="I will update it today."),
        ),
    )
    with pytest.raises(AttributeError):
        window.window_id = "other"  # type: ignore[misc]


def test_window_requires_complete_turn_fields(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"window_id":"w","history":[{"speaker":"A"}],"turns":[]}\n')
    with pytest.raises(InputError, match=r"bad\.jsonl:1.*history\[0\].*time"):
        list(load_windows(path))


def test_loader_reports_malformed_json_with_path_and_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"window_id":"one","history":[],"turns":[]}\nnot json\n')

    with pytest.raises(InputError, match=r"bad\.jsonl:2.*invalid JSON"):
        list(load_windows(path))


def test_atomic_jsonl_write_replaces_the_complete_file(tmp_path):
    path = tmp_path / "nested" / "windows.jsonl"
    atomic_write_jsonl(path, [{"window_id": "one"}, {"window_id": "two"}])

    assert path.read_text() == '{"window_id":"one"}\n{"window_id":"two"}\n'
    assert not list(path.parent.glob(".windows.jsonl.*.tmp"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gap_model": ""}, "gap_model"),
        ({"max_gaps": 0}, "max_gaps"),
        ({"max_gaps": 9}, "max_gaps"),
        ({"chunk_tokens": 0}, "chunk_tokens"),
        ({"memory_budget_tokens": 0}, "memory_budget_tokens"),
        ({"chunk_tokens": 32, "overlap_tokens": 32}, "overlap_tokens"),
    ],
)
def test_config_rejects_invalid_paper_parameters(kwargs, message):
    with pytest.raises(ConfigError, match=message):
        RememberConfig(**kwargs)
