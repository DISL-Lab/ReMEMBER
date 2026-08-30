import gzip
import importlib.util
import hashlib
import json
import subprocess
import sys

import pytest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("validate_release", ROOT / "scripts/validate_release.py")
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)


def _tiny_dataset(root: Path) -> Path:
    """Build a minimal valid v1 index + v2 payload.

    Mutation-only scans need a structurally valid dataset, not the real 74 MB
    payload; copying that into every temporary tree dominated the suite runtime.
    """
    dataset = root / "dataset"
    (dataset / "annotations").mkdir(parents=True)
    (dataset / "splits").mkdir(parents=True)
    (dataset / "splits/main_900.txt").write_text("E_U0000\nE_U0001\n", encoding="utf-8")

    def annotation(window_id):
        return {
            "schema_version": "1.0", "window_id": window_id, "source": "evermembench",
            "range": "8k~16k",
            "gaps": [{"gap_idx": 0, "target_sentence_numbers": [0], "gap_type": "C",
                      "evidence": [{"source_id": "T0-S1", "status": "resolve", "validation": 1,
                                    "vote_count": 2, "votes": {"gemma": 1, "GLM": 1, "Qwen": 0}}]}],
        }
    (dataset / "annotations/remember_900_annotations.jsonl").write_text(
        "".join(json.dumps(annotation(window_id)) + "\n"
                for window_id in ("E_U0000", "E_U0001")), encoding="utf-8")

    def turn(index, speaker, text):
        return {"turn_index": index, "time": "2024-01-01 09:00:00", "speaker": speaker,
                "dialogue": text}

    def payload(window_id):
        return {
        "schema_version": "2.0", "window_id": window_id, "source": "evermembench",
        "domain": "evermembench", "range": "8k~16k", "tok_start": 8000, "tok_end": 8001,
        "history": [turn(0, "A", "The deadline is March 14.")],
        "turns": [turn(0, "B", "She will finish before it.")],
        "salient_utterances": [{"salient_idx": 0, "sentence_numbers": [0],
                                "sentences": ["She will finish before it."],
                                "gap_type": "C", "vote_count": 2}],
        "gaps": [{"gap_idx": 0, "target_sentence_numbers": [0],
                  "target_sentence": "She will finish before it.", "gap_type": "C",
                  "gap_reason": "Identify the deadline.",
                  "evidence": [{"source_id": "T0-S1", "sentence": "The deadline is March 14.",
                                "status": "resolve", "validation": 1, "vote_count": 2,
                                "votes": {"gemma": 1, "GLM": 1, "Qwen": 0}}]}],
        }
    # Both windows are evermembench (workplace_1); the chit-chat and
    # workplace_2 shards ship empty since these mutation tests never need
    # that content - all three canonical shard names must still exist.
    with gzip.GzipFile(filename="", mode="wb", mtime=0,
                       fileobj=(dataset / "remember_900_workplace_1.jsonl.gz").open("wb")) as handle:
        handle.write("".join(json.dumps(payload(window_id), ensure_ascii=False) + "\n"
                             for window_id in ("E_U0000", "E_U0001")).encode("utf-8"))
    for empty_shard in ("remember_900_chitchat.jsonl.gz", "remember_900_workplace_2.jsonl.gz"):
        with gzip.GzipFile(filename="", mode="wb", mtime=0,
                           fileobj=(dataset / empty_shard).open("wb")) as handle:
            pass

    sys.path.insert(0, str(ROOT / "benchmark/src"))
    from remember_benchmark.release import _schema
    (dataset / "schema.json").write_text(
        json.dumps(_schema(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")

    def digest(relative: str) -> str:
        return hashlib.sha256((dataset / relative).read_bytes()).hexdigest()

    (dataset / "metadata.json").write_text(json.dumps({
        "counts": {"windows": 2, "gaps": 2, "evidence": 2},
        "artifacts": {
            "annotations": {"path": "annotations/remember_900_annotations.jsonl",
                            "sha256": digest("annotations/remember_900_annotations.jsonl")},
            "schema": {"path": "schema.json", "sha256": digest("schema.json")},
            "split": {"path": "splits/main_900.txt", "sha256": digest("splits/main_900.txt")},
        },
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return dataset


def _payload_failures(root: Path) -> str:
    return "\n".join(validator.scan_release(root)["leakage"]["failures"])


def _read_payload(dataset: Path, shard: str = "workplace_1") -> list[dict]:
    with gzip.open(dataset / f"remember_900_{shard}.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_payload(dataset: Path, rows: list[dict], shard: str = "workplace_1") -> None:
    with gzip.open(dataset / f"remember_900_{shard}.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

def test_release_layout_dataset_and_source_scans_are_clean():
    summary = validator.scan_release(ROOT)
    assert summary["layout"]["failures"] == []
    assert summary["leakage"]["failures"] == []
    assert summary["dataset"]["counts"] == {"windows": 900, "gaps": 1498, "evidence": 4135}
    assert summary["dataset"]["hashes"]["annotations"] == "a921ae399bd646d469a6824db4d5421358aee387dd1124ff2f1ee2f8c0a8ab3a"


def test_scans_diagnose_exact_payload_and_executable_leaks(tmp_path):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "dataset/bad.json").write_text(json.dumps({
        "target_sentence_numbers": [1], "target_sentence": "private"}))
    research_name = "1-" + "research"
    credential = "s" + "k-fixture"
    (tmp_path / "benchmark/bad.py").write_text(
        f"import sys\nsys.path.append('{research_name}')\nVALUE='{credential}'\n")
    summary = validator.scan_release(tmp_path)
    failures = "\n".join(summary["leakage"]["failures"])
    assert "target_sentence" in failures
    assert "target_sentence_numbers" not in failures
    assert "research-tree" in failures
    assert "credential" in failures


def test_validator_prints_one_machine_readable_json_summary():
    generated = (
        ROOT / ".pytest_cache",
        ROOT / "benchmark/.pytest_cache",
        ROOT / "remember/.pytest_cache",
        ROOT / "remember/build",
        ROOT / "remember/src/remember.egg-info",
    )
    before = {path for path in generated if path.exists()}
    for _ in range(2):
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/validate_release.py")],
                                   cwd=ROOT, text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr + completed.stdout
        summary = json.loads(completed.stdout)
        assert summary["ok"] is True
        assert summary["compile"]["ok"] is True
        assert summary["tests"]["ok"] is True
        assert summary["smokes"]["ok"] is True
    assert {path for path in generated if path.exists()} == before


def test_documentation_and_every_product_folder_exist():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    products = {"dataset", "benchmark", "remember", "evaluation"}
    assert {path.name for path in ROOT.iterdir() if path.is_dir()} >= products
    for name in products:
        assert (ROOT / name / "README.md").is_file(), name
    benchmark_readme = (ROOT / "benchmark/README.md").read_text(encoding="utf-8")
    assert "fail-closed" in benchmark_readme
    assert "without replacing the public" in benchmark_readme
    assert "private intermediates remain" in benchmark_readme


def test_documented_offline_smoke_uses_a_tracked_existing_input_path():
    readme = (ROOT / "remember/README.md").read_text(encoding="utf-8")
    relative = Path("remember/tests/fixtures/windows.jsonl")
    documented = ROOT / relative
    assert "--input tests/fixtures/windows.jsonl" in readme
    assert documented.is_file()
    tracked = subprocess.run(["git", "ls-files", str(relative)], cwd=ROOT,
                             text=True, capture_output=True, check=True).stdout.splitlines()
    assert str(relative) in tracked


def test_dataset_integrity_reports_metadata_count_tampering(tmp_path):
    _tiny_dataset(tmp_path)
    metadata_path = tmp_path / "dataset/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["counts"]["gaps"] = 999
    metadata_path.write_text(json.dumps(metadata))
    summary = validator.scan_release(tmp_path)
    assert "metadata counts" in "\n".join(summary["leakage"]["failures"])


def test_dataset_integrity_reports_split_order_tampering_without_crashing(tmp_path):
    _tiny_dataset(tmp_path)
    split_path = tmp_path / "dataset/splits/main_900.txt"
    ids = split_path.read_text().splitlines()
    ids[:2] = reversed(ids[:2])
    split_path.write_text("\n".join(ids) + "\n")
    summary = validator.scan_release(tmp_path)
    assert "split IDs" in "\n".join(summary["leakage"]["failures"])


def test_dataset_integrity_rejects_scalar_abuse_even_with_updated_artifact_hash(tmp_path):
    _tiny_dataset(tmp_path)
    annotation = tmp_path / "dataset/annotations/remember_900_annotations.jsonl"
    lines = annotation.read_text().splitlines()
    row = json.loads(lines[0])
    row["gaps"][0]["evidence"][0]["source_id"] = "PRIVATE DIALOGUE SENTENCE"
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    annotation.write_text("\n".join(lines) + "\n")
    metadata_path = tmp_path / "dataset/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["artifacts"]["annotations"]["sha256"] = hashlib.sha256(annotation.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "dataset integrity" in failures
    assert "source_id" in failures


def test_validator_rejects_symlinks_machine_paths_yaml_secrets_and_unknown_dataset_payload(tmp_path):
    _tiny_dataset(tmp_path)
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "benchmark/escape").symlink_to(ROOT, target_is_directory=True)
    local_root = "/" + "mnt/private/model"
    secret_name = "api_" + "key"
    (tmp_path / "benchmark/bad.yaml").write_text(f"path: {local_root}\n{secret_name}: abcdefgh\n")
    (tmp_path / "dataset/hidden_payload.txt").write_text("not allowed\n")
    summary = validator.scan_release(tmp_path)
    failures = "\n".join(summary["layout"]["failures"] + summary["leakage"]["failures"])
    assert "symlink" in failures
    assert "machine-local" in failures
    assert "credential" in failures
    assert "unknown dataset" in failures


def test_secret_scan_allows_environment_lookup_but_rejects_literal_assignments(tmp_path):
    _tiny_dataset(tmp_path)
    (tmp_path / "benchmark").mkdir()
    safe_name = "api_" + "key"
    (tmp_path / "benchmark/safe.py").write_text(
        f"import os\n{safe_name} = os.environ.get('OPENAI_API_KEY')\n"
        f"other_{safe_name} = 'OPENAI_API_KEY'\n")
    (tmp_path / "benchmark/safe.yaml").write_text(f"{safe_name}: OPENAI_API_KEY\n")
    assert "credential" not in "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    (tmp_path / "benchmark/opaque.yaml").write_text(f"{safe_name}: ABCDEFGH\n")
    assert "credential" in "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    (tmp_path / "benchmark/opaque.yaml").unlink()
    (tmp_path / "benchmark/bad.py").write_text(f"{safe_name}='abcdefgh'\n")
    assert "credential" in "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])


def test_source_scan_checks_toml_and_all_markdown_for_paths_and_secrets(tmp_path):
    _tiny_dataset(tmp_path)
    (tmp_path / "benchmark").mkdir()
    secret_name = "api_" + "key"
    (tmp_path / "benchmark/pyproject.toml").write_text(f"{secret_name}='abcdefgh'\n")
    machine_path = "/" + "mnt/private/x"
    (tmp_path / "dataset/DATASET_CARD.md").write_text(f"private source: {machine_path}\n")
    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "credential literal/assignment: benchmark/pyproject.toml" in failures
    assert "machine-local absolute path: dataset/DATASET_CARD.md" in failures


def test_source_scan_cannot_be_bypassed_with_arbitrary_text_extensions(tmp_path):
    _tiny_dataset(tmp_path)
    (tmp_path / "benchmark").mkdir()
    secret_name = "api_" + "key"
    machine_path = "/" + "mnt/private/x"
    (tmp_path / "benchmark/opaque.payload").write_text(
        f"{secret_name}='abcdefgh'\npath={machine_path}\n")
    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "credential literal/assignment: benchmark/opaque.payload" in failures
    assert "machine-local absolute path: benchmark/opaque.payload" in failures


def test_source_scan_rejects_generic_posix_machine_paths_but_allows_intentional_tmp_output(tmp_path):
    _tiny_dataset(tmp_path)
    (tmp_path / "benchmark").mkdir()
    private_model = "/" + "var/lib/private/model"
    system_secret = "/" + "etc/secret"
    (tmp_path / "benchmark/bad.txt").write_text(
        f"MODEL='{private_model}'\nOTHER='{system_secret}'\n")
    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "machine-local absolute path: benchmark/bad.txt" in failures

    (tmp_path / "benchmark/bad.txt").write_text("output: /tmp/remember-smoke.jsonl\n")
    assert "machine-local absolute path" not in "\n".join(
        validator.scan_release(tmp_path)["leakage"]["failures"])

    private_tmp = "/" + "tmp/private/model"
    credential_tmp = "/" + "tmp/credentials.json"
    (tmp_path / "benchmark/bad.txt").write_text(
        f"MODEL='{private_tmp}'\nSECRET_PATH='{credential_tmp}'\n")
    assert "machine-local absolute path: benchmark/bad.txt" in "\n".join(
        validator.scan_release(tmp_path)["leakage"]["failures"])


def test_plan_docs_still_scan_machine_paths_but_may_name_historical_research_tree(tmp_path):
    _tiny_dataset(tmp_path)
    (tmp_path / "docs/plans").mkdir(parents=True)
    research_name = "1-" + "research"
    machine_path = "/" + "mnt/private/x"
    secret_name = "api_" + "key"
    fallback_name = "token_" + "file"
    plan = tmp_path / "docs/plans/release.md"
    plan.write_text(
        f"historical input: {research_name}\nlocal input: {machine_path}\n"
        f"{secret_name}: abcdefgh\n{fallback_name}: credentials.txt\n"
    )
    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "machine-local absolute path: docs/plans/release.md" in failures
    assert "research-tree runtime reference: docs/plans/release.md" not in failures
    assert "credential literal/assignment: docs/plans/release.md" in failures
    secret_failure = "secret" + "-file fallback: docs/plans/release.md"
    assert secret_failure in failures


def test_superpowers_docs_may_name_historical_tree_but_still_scan_machine_paths(tmp_path):
    _tiny_dataset(tmp_path)
    path = tmp_path / "docs/superpowers/specs/release.md"
    path.parent.mkdir(parents=True)
    historical = "1-" + "research"
    machine_path = "/" + "mnt/private/source"
    secret_name = "api_" + "key"
    fixed_gpu = "CUDA_VISIBLE_DEVICES" + "=0"
    detached_runner = "no" + "hup python run.py"
    path.write_text(
        f"source: {historical}\nlocal: {machine_path}\n{secret_name}: abcdefgh\n"
        f"gpu: {fixed_gpu}\nrunner: {detached_runner}\n"
    )

    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "research-tree runtime reference" not in failures
    assert "fixed GPU assignment" not in failures
    assert "detached runtime command" not in failures
    assert "machine-local absolute path: docs/superpowers/specs/release.md" in failures
    assert "credential literal/assignment: docs/superpowers/specs/release.md" in failures


def test_source_scan_ignores_only_root_git_worktree_metadata_file(tmp_path):
    _tiny_dataset(tmp_path)
    (tmp_path / "benchmark").mkdir()
    gitdir_metadata = "gitdir: /" + "mnt/private/release/.git/worktrees/release\n"
    (tmp_path / ".git").write_text(gitdir_metadata)
    (tmp_path / "benchmark/copied-metadata.txt").write_text(gitdir_metadata)

    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "machine-local absolute path: .git" not in failures
    assert "machine-local absolute path: benchmark/copied-metadata.txt" in failures

    (tmp_path / ".git").write_text("notes: /" + "mnt/private/release\n")
    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "machine-local absolute path: .git" in failures


def test_evaluation_is_an_allowed_product_directory():
    assert "evaluation" in validator.ALLOWED_TOP
    assert {"dataset", "benchmark", "remember", "evaluation"} <= validator.ALLOWED_TOP


def test_evaluate_py_is_the_only_non_test_python_under_evaluation():
    modules = [path.relative_to(ROOT) for path in (ROOT / "evaluation").rglob("*.py")
               if "tests" not in path.relative_to(ROOT).parts
               and "__pycache__" not in path.parts]
    assert modules == [Path("evaluation/evaluate.py")]


def test_only_the_canonical_payload_may_be_a_gzip(tmp_path):
    _tiny_dataset(tmp_path)
    assert validator.scan_release(tmp_path)["layout"]["failures"] == []

    (tmp_path / "benchmark").mkdir()
    with gzip.open(tmp_path / "benchmark/extra.jsonl.gz", "wb") as handle:
        handle.write(b"{}\n")
    failures = "\n".join(validator.scan_release(tmp_path)["layout"]["failures"])
    assert "forbidden packaged file type: benchmark/extra.jsonl.gz" in failures


def test_an_unlisted_extra_dataset_payload_is_rejected(tmp_path):
    _tiny_dataset(tmp_path)
    (tmp_path / "dataset/extra.jsonl").write_text("{}\n", encoding="utf-8")
    assert "unknown dataset payload: extra.jsonl" in _payload_failures(tmp_path)


def test_v1_annotations_still_reject_text_bearing_keys(tmp_path):
    _tiny_dataset(tmp_path)
    path = tmp_path / "dataset/annotations/remember_900_annotations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["gaps"][0]["target_sentence"] = "private wording"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert "forbidden dataset key target_sentence" in _payload_failures(tmp_path)


def test_the_payload_may_carry_text_but_not_structural_leak_keys(tmp_path):
    dataset = _tiny_dataset(tmp_path)
    assert "forbidden payload key" not in _payload_failures(tmp_path)

    rows = _read_payload(dataset)
    rows[0]["gaps"][0]["evidence"][0]["reasoning"] = "chain of thought"
    _write_payload(dataset, rows)
    assert "forbidden payload key reasoning" in _payload_failures(tmp_path)


@pytest.mark.parametrize("mutation,message", [
    (lambda row: row.update(window_id="E_U9999"), "window order"),
    (lambda row: row.update(schema_version="1.0"), "unsupported schema version"),
    (lambda row: row.update(history=[]), "carries no dialogue text"),
])
def test_payload_integrity_failures_are_precise(tmp_path, mutation, message):
    dataset = _tiny_dataset(tmp_path)
    rows = _read_payload(dataset)
    mutation(rows[0])
    _write_payload(dataset, rows)
    assert message in _payload_failures(tmp_path)


def test_a_corrupt_or_missing_payload_is_reported(tmp_path):
    dataset = _tiny_dataset(tmp_path)
    workplace_1 = dataset / "remember_900_workplace_1.jsonl.gz"
    workplace_1.write_bytes(b"not gzip at all")
    assert "unreadable benchmark payload" in _payload_failures(tmp_path)
    workplace_1.unlink()
    assert "missing benchmark payload" in _payload_failures(tmp_path)


def test_the_checked_in_payload_has_the_released_population():
    summary = validator.scan_release(ROOT)
    assert summary["dataset"]["payload"]["counts"] == {
        "windows": 900, "gaps": 1498, "evidence_candidates": 8450,
        "evidence_validated": 4135, "salient_utterances": 7245}
    assert set(summary["dataset"]["payload"]["shards"]) == {
        "remember_900_chitchat.jsonl.gz", "remember_900_workplace_1.jsonl.gz",
        "remember_900_workplace_2.jsonl.gz"}
    assert summary["dataset"]["index"]["counts"] == {
        "windows": 900, "gaps": 1498, "evidence": 4135}


def test_domain_shards_split_evermembench_from_locomo_and_realtalk():
    dataset = ROOT / "dataset"

    def read(name):
        with gzip.open(dataset / name, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    workplace_1, workplace_2 = read("remember_900_workplace_1.jsonl.gz"), read(
        "remember_900_workplace_2.jsonl.gz")
    chitchat = read("remember_900_chitchat.jsonl.gz")
    workplace = workplace_1 + workplace_2
    assert len(workplace) == 600 and all(row["source"] == "evermembench" for row in workplace)
    assert len(chitchat) == 300 and all(row["source"] in ("locomo", "realtalk") for row in chitchat)
    for path in (dataset / "remember_900_workplace_1.jsonl.gz",
                dataset / "remember_900_workplace_2.jsonl.gz",
                dataset / "remember_900_chitchat.jsonl.gz"):
        assert path.stat().st_size < 50_000_000, path
    ids = [line.strip() for line in (dataset / "splits/main_900.txt").read_text().splitlines()
          if line.strip()]
    assert [row["window_id"] for row in chitchat] + [row["window_id"] for row in workplace] == ids


def test_payload_counts_are_enforced_for_a_canonical_sized_split(tmp_path):
    dataset = _tiny_dataset(tmp_path)
    ids = [f"E_U{index:04d}" for index in range(900)]
    (dataset / "splits/main_900.txt").write_text(
        "".join(f"{item}\n" for item in ids), encoding="utf-8")
    template = _read_payload(dataset)[0]
    _write_payload(dataset, [dict(template, window_id=item) for item in ids])
    failures = _payload_failures(tmp_path)
    assert "payload counts" in failures
    assert "'gaps': 900" in failures and "'evidence_validated': 4135" in failures


def test_the_payload_is_scanned_for_leaks_despite_the_utf8_exemption(tmp_path):
    dataset = _tiny_dataset(tmp_path)
    rows = _read_payload(dataset)
    research = "1-" + "research"
    credential = "s" + "k-projABCDEFGH"
    rows[0]["history"][0]["dialogue"] = f"see {research} and key {credential}"
    _write_payload(dataset, rows)
    failures = _payload_failures(tmp_path)
    assert "credential literal in payload" in failures
    assert "research-tree reference in payload" in failures


def test_dialogue_may_quote_a_path_but_a_non_dialogue_value_may_not(tmp_path):
    dataset = _tiny_dataset(tmp_path)
    rows = _read_payload(dataset)
    machine_path = "/" + "mnt/private/notes"
    rows[0]["history"][0]["dialogue"] = f"the log is at {machine_path}"
    _write_payload(dataset, rows)
    assert "machine-local absolute path in payload" not in _payload_failures(tmp_path)

    rows[0]["speaker_home"] = machine_path
    _write_payload(dataset, rows)
    assert "machine-local absolute path in payload" in _payload_failures(tmp_path)


def test_a_planning_workspace_below_the_root_is_still_scanned(tmp_path):
    _tiny_dataset(tmp_path)
    hidden = tmp_path / "benchmark/.superpowers"
    hidden.mkdir(parents=True)
    research = "1-" + "research"
    (hidden / "notes.md").write_text(f"input: {research}\n", encoding="utf-8")
    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "research-tree runtime reference: benchmark/.superpowers/notes.md" in failures
