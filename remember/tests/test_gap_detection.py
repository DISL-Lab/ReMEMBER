import json

from remember.gap_detection import parse_predicted_gaps
from remember.schemas import PredictedGap, Turn, Window


WINDOW = Window(
    window_id="gap-window",
    history=(),
    turns=(
        Turn("10:00", "Ava", "Please use the revised cedar plan for the launch."),
        Turn("10:01", "Bo", "I will update it today."),
    ),
)


def test_parses_a_valid_wrapped_json_response():
    raw = json.dumps({"gaps": [{
        "utterance_index": 0,
        "quote": "use the revised cedar plan",
        "gap_reason": "The plan is not named.",
        "evidence_need_query": "retrieve the earlier cedar plan decision",
        "keywords": ["Cedar", "launch"],
    }]})

    assert parse_predicted_gaps(raw, WINDOW) == [
        PredictedGap(
            index=0,
            utterance_index=0,
            quote="use the revised cedar plan",
            gap_reason="The plan is not named.",
            evidence_need_query="retrieve the earlier cedar plan decision",
            keywords=("cedar", "launch", "use", "revised", "plan", "named.", "retrieve", "earlier", "decision"),
            is_fallback=False,
        )
    ]


def test_accepts_fenced_json_and_recovers_a_missing_quote_from_index():
    raw = """```json
    {"gaps": [{"utterance_index": "U1", "gap_reason": "The referent is hidden.",
      "evidence_need_query": "retrieve what it refers to"}]}
    ```"""

    [gap] = parse_predicted_gaps(raw, WINDOW)

    assert gap.utterance_index == 1
    assert gap.quote == "I will update it today."
    assert gap.keywords[:3] == ("update", "today.", "referent")


def test_salvages_complete_inner_gap_objects_from_a_truncated_outer_wrapper():
    raw = """{"gaps": [
      {"utterance_index": 0, "quote": "cedar plan", "gap_reason": "missing plan", "evidence_need_query": "retrieve cedar plan"},
      {"utterance_index": 1, "quote": "update it", "gap_reason": "missing referent", "evidence_need_query": "retrieve update referent"}
    """

    gaps = parse_predicted_gaps(raw, WINDOW)

    assert [(gap.index, gap.utterance_index, gap.quote) for gap in gaps] == [
        (0, 0, "cedar plan"),
        (1, 1, "update it"),
    ]


def test_removes_duplicate_quotes_and_applies_the_eight_gap_cap():
    entries = [
        {"utterance_index": None, "quote": f"distinct clue {index}", "gap_reason": "missing", "evidence_need_query": "retrieve clue"}
        for index in range(10)
    ]
    entries.insert(1, {"quote": "distinct clue 0", "gap_reason": "duplicate", "evidence_need_query": "retrieve duplicate"})

    gaps = parse_predicted_gaps(json.dumps({"gaps": entries}), WINDOW, max_gaps=8)

    assert [gap.quote for gap in gaps] == [f"distinct clue {index}" for index in range(8)]
    assert [gap.index for gap in gaps] == list(range(8))


def test_creates_a_target_free_fallback_when_no_gap_can_be_normalized():
    [gap] = parse_predicted_gaps("not JSON", WINDOW, fallback=True)

    assert gap.index == 0
    assert gap.utterance_index is None
    assert gap.quote == "current window"
    assert gap.is_fallback is True
    assert gap.evidence_need_query == (
        "retrieve earlier dialogue that provides background, definitions, decisions, "
        "owners, artifacts, versions, or rationale for the current window"
    )


def test_can_return_no_gaps_when_fallback_is_disabled():
    assert parse_predicted_gaps("not JSON", WINDOW, fallback=False) == []
