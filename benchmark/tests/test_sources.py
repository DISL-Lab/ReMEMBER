import json
from pathlib import Path

import pytest

from remember_benchmark.schemas import NormalizedDialogue
from remember_benchmark.sources import (
    SourceInputError,
    load_evermembench,
    load_locomo,
    load_realtalk,
    normalize_sources,
)


def test_evermembench_adapter_observed_schema_is_stable_and_chronological(tmp_path):
    path = tmp_path / "ever.jsonl"
    rows = [
        {
            "topic_id": 7,
            "date": "2024-01-02",
            "dialogues": {
                "Group 2": [
                    {"message_index": 1, "time": "2024-01-02 09:01:00", "speaker": "B", "dialogue": "later"},
                    {"message_index": 0, "time": "2024-01-02 09:00:00", "speaker": "A", "dialogue": "earlier"},
                ]
            },
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = load_evermembench(path)

    assert [item.dialogue_id for item in result] == ["E_topic-7_group-2"]
    assert [turn.dialogue for turn in result[0].turns] == ["earlier", "later"]
    assert result[0].to_dict()["source"] == "evermembench"


def test_locomo_adapter_observed_schema_uses_session_time_order(tmp_path):
    path = tmp_path / "locomo10.json"
    path.write_text(
        json.dumps([
            {"conversation": {
                "speaker_a": "A", "speaker_b": "B",
                "session_2_date_time": "10:00 AM on 02 January, 2024",
                "session_2": [{"speaker": "B", "dia_id": "d2", "text": "second"}],
                "session_1_date_time": "09:00 AM on 01 January, 2024",
                "session_1": [{"speaker": "A", "dia_id": "d1", "text": "first"}],
            }}
        ]),
        encoding="utf-8",
    )

    result = load_locomo(path)

    assert result[0].dialogue_id == "L_topic-1_group-1"
    assert [turn.dialogue for turn in result[0].turns] == ["first", "second"]
    assert [turn.time for turn in result[0].turns] == ["2024-01-01 09:00:00", "2024-01-02 10:00:00"]


def test_realtalk_adapter_observed_schema_sorts_files_and_turns(tmp_path):
    (tmp_path / "Chat_2_B_C.json").write_text(json.dumps({
        "name": {"speaker_1": "B", "speaker_2": "C"},
        "session_1_date_time": "02.01.2024, 09:00:00",
        "session_1": [{"clean_text": "two", "speaker": "B", "date_time": "02.01.2024, 09:00:01", "dia_id": "2"}],
    }), encoding="utf-8")
    (tmp_path / "Chat_1_A_B.json").write_text(json.dumps({
        "name": {"speaker_1": "A", "speaker_2": "B"},
        "session_1_date_time": "01.01.2024, 09:00:00",
        "session_1": [
            {"clean_text": "later", "speaker": "B", "date_time": "01.01.2024, 09:00:02", "dia_id": "2"},
            {"clean_text": "earlier", "speaker": "A", "date_time": "01.01.2024, 09:00:01", "dia_id": "1"},
        ],
    }), encoding="utf-8")

    result = load_realtalk(tmp_path)

    assert [item.dialogue_id for item in result] == ["R_topic-1_group-1", "R_topic-2_group-1"]
    assert [turn.dialogue for turn in result[0].turns] == ["earlier", "later"]


@pytest.mark.parametrize("loader,name", [
    (load_evermembench, "EverMemBench"),
    (load_locomo, "LoCoMo"),
    (load_realtalk, "RealTalk"),
])
def test_missing_source_has_actionable_user_acquisition_guidance(tmp_path, loader, name):
    path = tmp_path / "missing"
    with pytest.raises(SourceInputError, match=rf"{name}.*user-acquired.*--{name.lower()}"):
        loader(path)


def test_normalize_sources_is_deterministic_and_rejects_duplicate_ids(tmp_path):
    ever = tmp_path / "ever.jsonl"
    ever.write_text(json.dumps({
        "topic_id": 1, "date": "2024-01-01",
        "dialogues": {"Group 1": [{"message_index": 0, "time": "2024-01-01 00:00:00", "speaker": "A", "dialogue": "x"}]},
    }) + "\n", encoding="utf-8")
    rows = normalize_sources(evermembench=ever)
    assert rows == sorted(rows, key=lambda item: item.dialogue_id)
    assert isinstance(rows[0], NormalizedDialogue)


def test_schema_error_reports_source_path_and_location(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"topic_id": 1, "date": "2024-01-01", "dialogues": {"Group 1": [{"speaker": "A"}]}}\n', encoding="utf-8")
    with pytest.raises(SourceInputError, match=r"bad\.jsonl:1.*dialogues\[Group 1\]\[0\].*dialogue"):
        load_evermembench(path)


def test_normalize_sources_orders_source_then_numeric_topic_group_across_eleven_topics(tmp_path):
    ever = tmp_path / "ever.jsonl"
    ever_rows = []
    for topic in (10, 2, 11, 1, 9, 3, 8, 4, 7, 5, 6):
        ever_rows.append({
            "topic_id": topic,
            "date": "2024-01-01",
            "dialogues": {"Group 1": [{
                "message_index": 0,
                "time": "2024-01-01 00:00:00",
                "speaker": "A",
                "dialogue": f"topic {topic}",
            }]},
        })
    ever.write_text("".join(json.dumps(row) + "\n" for row in ever_rows), encoding="utf-8")

    locomo = tmp_path / "locomo.json"
    locomo.write_text(json.dumps([{"conversation": {
        "speaker_a": "A", "speaker_b": "B",
        "session_1_date_time": "09:00 AM on 01 January, 2024",
        "session_1": [{"speaker": "A", "dia_id": "d1", "text": "locomo"}],
    }}]), encoding="utf-8")

    rows = normalize_sources(evermembench=ever, locomo=locomo)

    assert [row.dialogue_id for row in rows] == [
        *(f"E_topic-{topic}_group-1" for topic in range(1, 12)),
        "L_topic-1_group-1",
    ]
