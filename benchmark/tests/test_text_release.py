import copy
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import remember_benchmark.text_release as text_release
from remember_benchmark.io import atomic_write_jsonl, atomic_write_jsonl_gzip, iter_jsonl_gzip
from remember_benchmark.text_release import (
    BENCHMARK_SCHEMA_VERSION,
    build_benchmark_records,
    build_benchmark_records_from_pipeline,
    benchmark_schema,
    export_benchmark,
    validate_benchmark_directory,
    validate_benchmark_records,
)
from remember_benchmark.pipeline import _merge_salient, flatten_ensemble_to_rcv


def valid_fixture():
    sources = [{
        "window_id": "E_U0000",
        "domain": "evermembench",
        "tok_start": 7000,
        "tok_end": 8024,
        "history": [{
            "time": "09:00", "speaker": "A",
            "dialogue": "First support. Second support.",
            "topic": 1, "img_url": "not-published",
        }],
        "turns": [{
            "time": "10:00", "speaker": "B",
            "dialogue": "The plan was approved. Work starts now.",
            "message_index": 3,
        }],
    }]
    rcv = [{
        "window_id": "E_U0000", "range": "8k~16k", "gap_idx": 0,
        "sentence_numbers": [0, 1],
        "target_sentence": "The plan was approved. Work starts now.",
        "gap_type": "A", "gap_reason": "The earlier support is missing.",
        "validated": [{
            "source_id": "T0-S2", "sentence": "Second support.",
            "status": "resolve", "validation": 1, "vote_count": 3,
            "votes": {"gemma": 1, "GLM": 1, "Qwen": 1},
        }],
    }]
    salient = [{
        "window_id": "E_U0000",
        "gap_analysis": [{
            "sentence_numbers": [0, 1],
            "sentences": ["The plan was approved.", "Work starts now."],
            "gap_type": "A", "vote_count": 3,
        }],
    }]
    return sources, rcv, salient


def built_row():
    sources, rcv, salient = valid_fixture()
    return build_benchmark_records(sources, rcv, salient, ["E_U0000"])[0]


def test_build_benchmark_records_binds_every_coordinate_to_text():
    sources, rcv, salient = valid_fixture()

    [row] = build_benchmark_records(sources, rcv, salient, ["E_U0000"])

    assert row["schema_version"] == "2.0"
    assert row["source"] == row["domain"] == "evermembench"
    assert row["history"] == [{
        "turn_index": 0, "time": "09:00", "speaker": "A",
        "dialogue": "First support. Second support.",
    }]
    assert row["turns"] == [{
        "turn_index": 0, "time": "10:00", "speaker": "B",
        "dialogue": "The plan was approved. Work starts now.",
    }]
    assert row["salient_utterances"] == [
        {
            "salient_idx": 0, "sentence_numbers": [0],
            "sentences": ["The plan was approved."], "gap_type": "A", "vote_count": 3,
        },
        {
            "salient_idx": 1, "sentence_numbers": [1],
            "sentences": ["Work starts now."], "gap_type": "A", "vote_count": 3,
        },
    ]
    assert row["gaps"] == [{
        "gap_idx": 0,
        "target_sentence_numbers": [0, 1],
        "target_sentence": "The plan was approved. Work starts now.",
        "gap_type": "A",
        "gap_reason": "The earlier support is missing.",
        "evidence": [{
            "source_id": "T0-S2", "sentence": "Second support.",
            "status": "resolve", "validation": 1, "vote_count": 3,
            "votes": {"gemma": 1, "GLM": 1, "Qwen": 1},
        }],
    }]
    assert "img_url" not in json.dumps(row)


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda s, r, a: s.append(dict(s[0])), "duplicate source window_id"),
    (lambda s, r, a: r[0].update(sentence_numbers=[99]), "target coordinate"),
    (lambda s, r, a: r[0].update(target_sentence="wrong"), "target text mismatch"),
    (lambda s, r, a: r[0]["validated"][0].update(source_id="T0-S0"), "evidence coordinate"),
    (lambda s, r, a: r[0]["validated"][0].update(sentence="wrong"), "evidence text mismatch"),
    (lambda s, r, a: a[0]["gap_analysis"][0].update(sentences=["wrong"]), "salient text mismatch"),
    (lambda s, r, a: s[0]["history"][0].update(dialogue=1), "history[0].dialogue"),
])
def test_build_benchmark_records_rejects_unverifiable_text(mutation, message):
    sources, rcv, salient = valid_fixture()
    mutation(sources, rcv, salient)
    with pytest.raises(ValueError, match=re.escape(message)):
        build_benchmark_records(sources, rcv, salient, ["E_U0000"])


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda s, r, a: s.clear(), "missing source row"),
    (lambda s, r, a: a.clear(), "missing salient row"),
    (lambda s, r, a: a.append(dict(a[0])), "duplicate salient window_id"),
    (lambda s, r, a: r[0].update(range="16k~32k"), "pipeline range"),
    (lambda s, r, a: r.append(dict(r[0], range="16k~32k", gap_idx=1)), "inconsistent range"),
    (lambda s, r, a: r.append(dict(r[0])), "duplicate gap_idx"),
    (lambda s, r, a: r[0]["validated"].append(dict(r[0]["validated"][0])), "duplicate evidence"),
])
def test_build_benchmark_records_rejects_incomplete_or_ambiguous_joins(mutation, message):
    sources, rcv, salient = valid_fixture()
    if message == "pipeline range":
        sources[0]["range"] = "8k~16k"
    mutation(sources, rcv, salient)
    with pytest.raises(ValueError, match=re.escape(message)):
        build_benchmark_records(sources, rcv, salient, ["E_U0000"])


@pytest.mark.parametrize(("split_id", "source_value", "message"), [
    ("X_U0000", "evermembench", "unknown source prefix"),
    ("E_U0000", "locomo", "domain/source does not match window ID"),
])
def test_build_benchmark_records_rejects_invalid_source_identity(split_id, source_value, message):
    sources, rcv, salient = valid_fixture()
    sources[0]["window_id"] = rcv[0]["window_id"] = salient[0]["window_id"] = split_id
    sources[0]["domain"] = source_value
    with pytest.raises(ValueError, match=re.escape(message)):
        build_benchmark_records(sources, rcv, salient, [split_id])


def test_build_benchmark_records_filters_unvalidated_evidence_and_omits_empty_targets():
    sources, rcv, salient = valid_fixture()
    rejected = dict(rcv[0]["validated"][0], source_id="T0-S1", sentence="First support.", validation=0)
    rcv[0]["validated"].append(rejected)
    rcv.append({
        **copy.deepcopy(rcv[0]),
        "gap_idx": 7,
        "sentence_numbers": [1],
        "target_sentence": "  ",
        "validated": [dict(rcv[0]["validated"][0])],
    })

    [row] = build_benchmark_records(sources, rcv, salient, ["E_U0000"])

    assert [gap["gap_idx"] for gap in row["gaps"]] == [0]
    assert [item["source_id"] for item in row["gaps"][0]["evidence"]] == ["T0-S2"]


def test_build_benchmark_records_filters_before_validating_unselected_rows():
    sources, rcv, salient = valid_fixture()
    sources.insert(0, {"window_id": "not-selected", "history": "invalid"})
    rcv.insert(0, {"window_id": "not-selected", "target_sentence": 1})
    salient.insert(0, {"window_id": "not-selected", "gap_analysis": "invalid"})

    rows = build_benchmark_records(iter(sources), iter(rcv), iter(salient), ["E_U0000"])

    assert [row["window_id"] for row in rows] == ["E_U0000"]


def test_build_benchmark_records_from_pipeline_resolves_missing_text_and_domain():
    sources, rcv, salient = valid_fixture()
    source_name = sources[0].pop("domain")
    window = {**sources[0], "source": source_name, "range": "8k~16k"}
    rcv[0]["target_sentence"] = ["The plan was approved.", "Work starts now."]
    rcv[0]["validated"][0].pop("sentence")
    salient = [{
        "window_id": "E_U0000", "gaps": [{
            "gap_idx": 0, "target_sentence_numbers": [0, 1],
            "target_sentence": ["The plan was approved.", "Work starts now."],
            "gap_type": "A", "gap_reason": "The earlier support is missing.",
            "vote_count": 3,
        }],
    }]

    [row] = build_benchmark_records_from_pipeline([window], rcv, salient, ["E_U0000"])

    assert row["source"] == row["domain"] == "evermembench"
    assert row["gaps"][0]["target_sentence"] == "The plan was approved. Work starts now."
    assert row["gaps"][0]["evidence"][0]["sentence"] == "Second support."
    assert [item["sentences"] for item in row["salient_utterances"]] == [
        ["The plan was approved."], ["Work starts now."],
    ]
    assert "gap_reasons" not in json.dumps(row)


def test_pipeline_chain_replaces_composed_reasons_and_strips_private_explanations():
    sources, _, _ = valid_fixture()
    source_name = sources[0].pop("domain")
    window = {**sources[0], "source": source_name, "range": "8k~16k"}
    voted = [{
        "schema_version": "1.0", "window_id": "E_U0000", "gaps": [{
            "gap_idx": 0, "target_sentence_numbers": [0, 1],
            "gap_type": "A", "vote_count": 3,
        }],
    }]
    private_reasons = {
        "gemma": "gemma private rationale",
        "GLM": "GLM private rationale",
        "Qwen": "Qwen private rationale",
    }
    rows_by_model = {
        model: [{"window_id": "E_U0000", "gaps": [{
            "target_sentence_numbers": [0, 1], "gap_reason": reason,
        }]}]
        for model, reason in private_reasons.items()
    }
    merged = _merge_salient([window], voted, rows_by_model)
    ensemble = [{
        "window_id": "E_U0000", "range": "8k~16k", "gaps": [{
            "gap_idx": 0, "target_sentence_numbers": [0, 1], "gap_type": "A",
            "evidence": [{
                "source_id": "T0-S2", "status": "resolve", "validation": 1,
                "vote_count": 3, "votes": {"gemma": 1, "GLM": 1, "Qwen": 1},
                "reasoning": "validator chain of thought",
                "private_explanation": "do not publish this explanation",
            }],
        }],
    }]
    private_rcv = flatten_ensemble_to_rcv(ensemble, merged)

    [row] = build_benchmark_records_from_pipeline([window], private_rcv, merged, ["E_U0000"])

    assert row["gaps"][0]["gap_reason"] == (
        "Resolve the target utterance using the validated evidence."
    )
    serialized = json.dumps(row)
    for forbidden in (
        "gemma:", "GLM:", "Qwen:",
        "gemma private rationale", "GLM private rationale", "Qwen private rationale",
        "gap_reasons", "reasoning", "validator chain of thought",
        "private_explanation", "do not publish this explanation",
    ):
        assert forbidden not in serialized


def test_validate_benchmark_records_rechecks_text_and_returns_literal_counts():
    row = built_row()

    stats = validate_benchmark_records(
        [row], ["E_U0000"], expected_counts=(1, 1, 1), expected_salient_count=2
    )

    assert stats == {
        "windows": 1,
        "gaps": 1,
        "evidence": 1,
        "salient_utterances": 2,
        "by_source": {"evermembench": 1},
        "by_range": {"8k~16k": 1},
    }


def test_validate_benchmark_records_rejects_whitespace_only_gap_reason():
    row = built_row()
    row["gaps"][0]["gap_reason"] = "   "

    with pytest.raises(ValueError, match="gap_reason.*non-empty"):
        validate_benchmark_records([row], ["E_U0000"])


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda row: row.update(reasoning="leak"), "invalid keys"),
    (lambda row: row["gaps"][0].update(reasons=["leak"]), "invalid keys"),
    (lambda row: row["gaps"][0].update(target_sentence="wrong"), "target text mismatch"),
    (lambda row: row["gaps"].append(copy.deepcopy(row["gaps"][0])), "duplicate gap_idx"),
    (lambda row: row["gaps"][0]["evidence"][0].update(sentence="wrong"), "evidence text mismatch"),
    (lambda row: row["gaps"][0]["evidence"][0]["votes"].update(gemma=0), "evidence fields"),
    (lambda row: row["salient_utterances"][0].update(sentences=["wrong"]), "salient text mismatch"),
    (lambda row: row["salient_utterances"][0].update(vote_count=0), "salient vote_count"),
])
def test_validate_benchmark_records_rejects_schema_and_semantic_abuse(mutation, message):
    row = built_row()
    mutation(row)
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_benchmark_records([row], ["E_U0000"])


def test_benchmark_schema_declares_closed_text_bearing_contract():
    schema = benchmark_schema()
    window = schema["properties"]
    turn = window["history"]["items"]
    salient = window["salient_utterances"]["items"]
    gap = window["gaps"]["items"]
    evidence = gap["properties"]["evidence"]["items"]

    assert BENCHMARK_SCHEMA_VERSION == "2.0"
    assert schema["additionalProperties"] is False
    assert schema["required"] == sorted(window)
    assert window["window_id"]["pattern"] == r"^[ELR]_U\d{4,}$"
    assert window["source"]["enum"] == ["evermembench", "locomo", "realtalk"]
    assert window["range"]["enum"] == [
        "8k~16k", "16k~32k", "32k~64k", "64k~128k", "128k~160k",
    ]
    votes = evidence["properties"]["votes"]
    assert all(item["additionalProperties"] is False for item in (turn, salient, gap, evidence, votes))
    assert turn["properties"]["dialogue"] == {"type": "string", "minLength": 1}
    assert salient["properties"]["sentences"]["items"] == {"type": "string", "minLength": 1}
    assert gap["properties"]["gap_reason"] == {
        "type": "string", "minLength": 1, "pattern": r"\S",
    }
    assert evidence["properties"]["sentence"] == {"type": "string", "minLength": 1}
    assert evidence["properties"]["source_id"]["pattern"] == r"^T(0|[1-9]\d*)-S([1-9]\d*)$"


def test_atomic_gzip_jsonl_is_byte_deterministic(tmp_path):
    rows = [{"window_id": "E_U0000", "text": "한글 text"}]
    left = tmp_path / "left/a.jsonl.gz"
    right = tmp_path / "right/b.jsonl.gz"

    atomic_write_jsonl_gzip(left, rows)
    atomic_write_jsonl_gzip(right, rows)

    assert left.read_bytes() == right.read_bytes()
    assert list(iter_jsonl_gzip(left)) == rows
    with gzip.open(left, "rt", encoding="utf-8") as handle:
        assert handle.read().endswith("\n")


@pytest.mark.parametrize(("payload", "message"), [
    (b'{"ok":1}\n\n', "blank JSONL line.*:2"),
    (b'[1]\n', "not an object.*:1"),
    (b'{bad}\n', "invalid JSON.*:1"),
    (b'\xff\n', "invalid UTF-8.*:1"),
    (b'{"ok":1}\ntrailing', "invalid JSON.*:2"),
])
def test_iter_jsonl_gzip_rejects_invalid_physical_lines(tmp_path, payload, message):
    path = tmp_path / "invalid.jsonl.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(payload)

    with pytest.raises(ValueError, match=message):
        list(iter_jsonl_gzip(path))


def write_raw_gzip(path: Path, payload: bytes):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(payload)


def test_iter_jsonl_gzip_requires_final_record_newline_but_allows_empty_shard(tmp_path):
    missing_newline = tmp_path / "missing-newline.jsonl.gz"
    empty = tmp_path / "empty.jsonl.gz"
    write_raw_gzip(missing_newline, b'{"ok":1}')
    write_raw_gzip(empty, b"")

    with pytest.raises(ValueError, match=r"trailing newline.*missing-newline.jsonl.gz:1"):
        list(iter_jsonl_gzip(missing_newline))
    assert list(iter_jsonl_gzip(empty)) == []


def write_export_inputs(root: Path, count: int = 1):
    source_path = root / "inputs/windows.jsonl"
    rcv_path = root / "inputs/rcv.jsonl"
    salient_path = root / "inputs/salient.jsonl"
    split_path = root / "splits/main_900.txt"
    source_rows = []
    rcv_rows = []
    salient_rows = []
    for index in range(count):
        window_id = f"E_U{index:04d}"
        sources, rcv, salient = valid_fixture()
        for row in (sources[0], rcv[0], salient[0]):
            row["window_id"] = window_id
        source_rows.append(sources[0])
        rcv_rows.append(rcv[0])
        salient_rows.append(salient[0])
    atomic_write_jsonl(source_path, source_rows)
    atomic_write_jsonl(rcv_path, rcv_rows)
    atomic_write_jsonl(salient_path, salient_rows)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        "".join(f"E_U{index:04d}\n" for index in range(count)), encoding="utf-8"
    )
    return source_path, rcv_path, salient_path, split_path, root / "benchmark"


def read_manifest(benchmark_dir: Path):
    return json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(benchmark_dir: Path, manifest):
    (benchmark_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_benchmark_writes_ten_balanced_canonical_shards_and_manifest(tmp_path):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path, count=900)

    manifest = export_benchmark(
        source, rcv, salient, split, benchmark_dir,
        expected_counts=(900, 900, 900), expected_salient_count=1800,
    )

    assert [item["path"] for item in manifest["shards"]] == [
        f"remember_900-{index:05d}-of-00010.jsonl.gz" for index in range(10)
    ]
    assert [item["rows"] for item in manifest["shards"]] == [90] * 10
    assert manifest["counts"]["windows"] == 900
    assert manifest["authorization"]["redistributable_text"] is True
    assert manifest["sharding"] == {"count": 10, "policy": "balanced-contiguous-v1"}
    assert manifest["artifacts"]["split"] == {
        "path": "../splits/main_900.txt", "sha256": sha256(split),
    }
    assert manifest["sources"] == {
        "windows": {"name": "windows.jsonl", "sha256": sha256(source)},
        "rcv": {"name": "rcv.jsonl", "sha256": sha256(rcv)},
        "salient": {"name": "salient.jsonl", "sha256": sha256(salient)},
    }
    assert not any(str(tmp_path) in json.dumps(value) for value in manifest["sources"].values())
    assert validate_benchmark_directory(
        benchmark_dir,
        [f"E_U{index:04d}" for index in range(900)],
        expected_counts=(900, 900, 900),
        expected_salient_count=1800,
    ) == manifest["counts"]


def test_export_benchmark_balances_small_exports_with_only_trailing_empty_shards(tmp_path):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path)

    manifest = export_benchmark(
        source, rcv, salient, split, benchmark_dir, construction="pipeline-derived"
    )

    assert [item["rows"] for item in manifest["shards"]] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert manifest["provenance"]["construction"] == "pipeline-derived"
    assert [row["window_id"] for item in manifest["shards"]
            for row in iter_jsonl_gzip(benchmark_dir / item["path"])] == ["E_U0000"]


def test_canonical_900_export_requires_exactly_ten_ninety_row_shards(tmp_path):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path, count=900)

    with pytest.raises(ValueError, match="canonical 900-row export requires 10 shards"):
        export_benchmark(
            source, rcv, salient, split, benchmark_dir,
            shard_count=9, construction="canonical-source-join",
        )

    assert not benchmark_dir.exists()


def test_validation_rejects_canonical_provenance_on_variable_900_row_layout(tmp_path):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path, count=900)
    export_benchmark(
        source, rcv, salient, split, benchmark_dir,
        shard_count=9, construction="pipeline-derived",
    )
    manifest = read_manifest(benchmark_dir)
    manifest["provenance"]["construction"] = "canonical-source-join"
    write_manifest(benchmark_dir, manifest)

    with pytest.raises(ValueError, match="canonical 900-row manifest requires 10 shards"):
        validate_benchmark_directory(
            benchmark_dir, [f"E_U{index:04d}" for index in range(900)]
        )


def test_export_benchmark_rejects_nonpositive_shard_count_before_writing(tmp_path):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path)

    with pytest.raises(ValueError, match="shard_count must be positive"):
        export_benchmark(source, rcv, salient, split, benchmark_dir, shard_count=0)

    assert not benchmark_dir.exists()


def exported_fixture(tmp_path):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path)
    export_benchmark(source, rcv, salient, split, benchmark_dir)
    return source, rcv, salient, split, benchmark_dir


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda d, m: (d / m["shards"][0]["path"]).unlink(), "directory contents"),
    (lambda d, m: (d / "extra.jsonl.gz").write_bytes(b"extra"), "directory contents"),
    (lambda d, m: m["shards"].__setitem__(slice(0, 2), [m["shards"][1], m["shards"][0]]),
     "shard order"),
    (lambda d, m: (d / m["shards"][0]["path"]).write_bytes(
        bytes([*(d / m["shards"][0]["path"]).read_bytes()[:4], 1,
               *(d / m["shards"][0]["path"]).read_bytes()[5:]])
     ), "shard hash mismatch"),
    (lambda d, m: (d / "schema.json").write_text("{}\n", encoding="utf-8"),
     "schema hash mismatch"),
    (lambda d, m: m["sources"]["windows"].update(sha256="INVALID"),
     "source hash"),
    (lambda d, m: m["counts"].update(windows=2), "manifest count mismatch"),
])
def test_validate_benchmark_directory_rejects_manifest_and_artifact_mutations(
    tmp_path, mutation, message
):
    _, _, _, _, benchmark_dir = exported_fixture(tmp_path)
    manifest = read_manifest(benchmark_dir)
    mutation(benchmark_dir, manifest)
    write_manifest(benchmark_dir, manifest)

    with pytest.raises((ValueError, OSError), match=message):
        validate_benchmark_directory(benchmark_dir, ["E_U0000"])


def test_validate_benchmark_directory_hashes_before_rejecting_corrupt_gzip(tmp_path):
    _, _, _, _, benchmark_dir = exported_fixture(tmp_path)
    manifest = read_manifest(benchmark_dir)
    shard = benchmark_dir / manifest["shards"][0]["path"]
    shard.write_bytes(b"not gzip")
    manifest["shards"][0]["sha256"] = sha256(shard)
    write_manifest(benchmark_dir, manifest)

    with pytest.raises((ValueError, OSError), match="gzip|compressed|Not a gzipped"):
        validate_benchmark_directory(benchmark_dir, ["E_U0000"])


def test_validate_benchmark_directory_rejects_rehashed_shard_without_final_newline(tmp_path):
    _, _, _, _, benchmark_dir = exported_fixture(tmp_path)
    manifest = read_manifest(benchmark_dir)
    shard = benchmark_dir / manifest["shards"][0]["path"]
    with gzip.open(shard, "rb") as handle:
        payload = handle.read()
    assert payload.endswith(b"\n")
    write_raw_gzip(shard, payload[:-1])
    manifest["shards"][0]["sha256"] = sha256(shard)
    write_manifest(benchmark_dir, manifest)

    with pytest.raises(ValueError, match=r"trailing newline.*:1"):
        validate_benchmark_directory(benchmark_dir, ["E_U0000"])


def test_validate_benchmark_directory_rejects_changed_split_hash(tmp_path):
    _, _, _, split, benchmark_dir = exported_fixture(tmp_path)
    split.write_text("E_U0000\nE_U0001\n", encoding="utf-8")

    with pytest.raises(ValueError, match="split hash mismatch"):
        validate_benchmark_directory(benchmark_dir, ["E_U0000"])


def test_validate_benchmark_directory_compares_supplied_source_hashes(tmp_path):
    _, _, _, _, benchmark_dir = exported_fixture(tmp_path)
    expected = {name: "0" * 64 for name in ("windows", "rcv", "salient")}

    with pytest.raises(ValueError, match="windows source hash mismatch"):
        validate_benchmark_directory(
            benchmark_dir, ["E_U0000"], expected_source_hashes=expected
        )


@pytest.mark.parametrize("construction", [
    "standalone-canonical", "unapproved", "/tmp/pipeline-derived", "../pipeline-derived",
    ["pipeline-derived"],
])
def test_export_benchmark_rejects_unapproved_or_pathlike_construction(tmp_path, construction):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path)

    with pytest.raises(ValueError, match="approved construction"):
        export_benchmark(
            source, rcv, salient, split, benchmark_dir, construction=construction
        )

    assert not benchmark_dir.exists()


@pytest.mark.parametrize("construction", [
    "unapproved", "/tmp/private", "../pipeline-derived", ["pipeline-derived"],
])
def test_validation_rejects_manifest_defined_construction_expectations(tmp_path, construction):
    _, _, _, _, benchmark_dir = exported_fixture(tmp_path)
    manifest = read_manifest(benchmark_dir)
    manifest["provenance"]["construction"] = construction
    write_manifest(benchmark_dir, manifest)

    with pytest.raises(ValueError, match="approved construction"):
        validate_benchmark_directory(benchmark_dir, ["E_U0000"])


@pytest.mark.parametrize("source_name", [
    "/tmp/windows.jsonl", "../windows.jsonl", r"C:\private\windows.jsonl",
    r"..\windows.jsonl",
])
def test_validation_rejects_absolute_or_pathlike_source_names(tmp_path, source_name):
    _, _, _, _, benchmark_dir = exported_fixture(tmp_path)
    manifest = read_manifest(benchmark_dir)
    manifest["sources"]["windows"]["name"] = source_name
    write_manifest(benchmark_dir, manifest)

    with pytest.raises(ValueError, match="invalid windows source name"):
        validate_benchmark_directory(benchmark_dir, ["E_U0000"])


def test_export_rejects_pathlike_source_provenance_name(tmp_path):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path)
    pathlike_source = source.with_name(r"C:\private\windows.jsonl")
    source.rename(pathlike_source)

    with pytest.raises(ValueError, match="invalid windows source name"):
        export_benchmark(pathlike_source, rcv, salient, split, benchmark_dir)

    assert not benchmark_dir.exists()


@pytest.mark.parametrize("fail_at", [1, 2])
def test_publish_staged_directory_restores_old_target_after_fsync_failure(
    tmp_path, monkeypatch, fail_at
):
    target = tmp_path / "public"
    staged = tmp_path / "staged"
    target.mkdir()
    staged.mkdir()
    (target / "inventory.json").write_text("old inventory", encoding="utf-8")
    (staged / "inventory.json").write_text("new inventory", encoding="utf-8")
    real_fsync_parent = text_release.fsync_parent_directory
    calls = 0

    def fail_selected_boundary(path):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(f"injected fsync failure {fail_at}")
        return real_fsync_parent(path)

    monkeypatch.setattr(text_release, "fsync_parent_directory", fail_selected_boundary)

    with pytest.raises(OSError, match=f"injected fsync failure {fail_at}"):
        text_release._publish_staged_directory(staged, target)

    assert (target / "inventory.json").read_text(encoding="utf-8") == "old inventory"
    assert not staged.exists()
    assert list(tmp_path.glob(".public.backup-*")) == []
    assert list(tmp_path.glob(".public.failed-*")) == []


def cli_command(source, rcv, salient, split, benchmark_dir, *extra):
    return [
        sys.executable, "-m", "remember_benchmark.text_release",
        "--source", str(source), "--rcv", str(rcv), "--salient", str(salient),
        "--split", str(split), "--benchmark-dir", str(benchmark_dir), *extra,
    ]


def run_cli(command):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str((Path.cwd() / "benchmark/src").resolve())
    return subprocess.run(command, env=environment, text=True, capture_output=True, check=False)


def test_export_cli_preserves_utf8_readme_and_atomically_replaces_old_directory(tmp_path):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path)
    benchmark_dir.mkdir()
    readme_bytes = "# 안내\nkeep this exactly\n".encode()
    (benchmark_dir / "README.md").write_bytes(readme_bytes)
    (benchmark_dir / "old.txt").write_text("old", encoding="utf-8")

    completed = run_cli(cli_command(
        source, rcv, salient, split, benchmark_dir,
        "--expected-salient", "2", "--shard-count", "10", "--overwrite",
    ))

    assert completed.returncode == 0, completed.stderr
    assert (benchmark_dir / "README.md").read_bytes() == readme_bytes
    assert not (benchmark_dir / "old.txt").exists()
    assert read_manifest(benchmark_dir)["counts"]["windows"] == 1


@pytest.mark.parametrize("readme_kind", ["symlink", "non_utf8"])
def test_export_cli_rejects_unsafe_readme_before_publication(tmp_path, readme_kind):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path)
    benchmark_dir.mkdir()
    marker = benchmark_dir / "old.txt"
    marker.write_text("old", encoding="utf-8")
    readme = benchmark_dir / "README.md"
    if readme_kind == "symlink":
        target = tmp_path / "actual-readme.md"
        target.write_text("safe text", encoding="utf-8")
        readme.symlink_to(target)
    else:
        readme.write_bytes(b"\xff\xfe")

    completed = run_cli(cli_command(source, rcv, salient, split, benchmark_dir, "--overwrite"))

    assert completed.returncode != 0
    assert marker.read_text(encoding="utf-8") == "old"
    if readme_kind == "symlink":
        assert readme.is_symlink()
    else:
        assert readme.read_bytes() == b"\xff\xfe"
    assert not (benchmark_dir / "manifest.json").exists()


def test_export_cli_refuses_nonempty_directory_without_overwrite_and_partial_counts(tmp_path):
    source, rcv, salient, split, benchmark_dir = write_export_inputs(tmp_path)
    benchmark_dir.mkdir()
    marker = benchmark_dir / "old.txt"
    marker.write_text("old", encoding="utf-8")

    occupied = run_cli(cli_command(source, rcv, salient, split, benchmark_dir))
    partial = run_cli(cli_command(
        source, rcv, salient, split, tmp_path / "other-benchmark",
        "--expected-windows", "1",
    ))

    assert occupied.returncode != 0
    assert "non-empty" in occupied.stderr
    assert marker.read_text(encoding="utf-8") == "old"
    assert partial.returncode != 0
    assert "expected paper counts must be supplied together" in partial.stderr
