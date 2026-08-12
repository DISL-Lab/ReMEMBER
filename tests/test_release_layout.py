import importlib.util
import hashlib
import json
import subprocess
import sys
import shutil
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("validate_release", ROOT / "scripts/validate_release.py")
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)


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


def test_documentation_and_three_product_folders_exist():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "benchmark/README.md").is_file()
    assert {path.name for path in ROOT.iterdir() if path.is_dir() and path.name in {"dataset", "benchmark", "remember"}} == {
        "dataset", "benchmark", "remember"
    }
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
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
    metadata_path = tmp_path / "dataset/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["counts"]["gaps"] = 999
    metadata_path.write_text(json.dumps(metadata))
    summary = validator.scan_release(tmp_path)
    assert "metadata counts" in "\n".join(summary["leakage"]["failures"])


def test_dataset_integrity_reports_split_order_tampering_without_crashing(tmp_path):
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
    split_path = tmp_path / "dataset/splits/main_900.txt"
    ids = split_path.read_text().splitlines()
    ids[:2] = reversed(ids[:2])
    split_path.write_text("\n".join(ids) + "\n")
    summary = validator.scan_release(tmp_path)
    assert "split IDs" in "\n".join(summary["leakage"]["failures"])


def test_dataset_integrity_rejects_scalar_abuse_even_with_updated_artifact_hash(tmp_path):
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
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
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
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
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
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
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
    (tmp_path / "benchmark").mkdir()
    secret_name = "api_" + "key"
    (tmp_path / "benchmark/pyproject.toml").write_text(f"{secret_name}='abcdefgh'\n")
    machine_path = "/" + "mnt/private/x"
    (tmp_path / "dataset/DATASET_CARD.md").write_text(f"private source: {machine_path}\n")
    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "credential literal/assignment: benchmark/pyproject.toml" in failures
    assert "machine-local absolute path: dataset/DATASET_CARD.md" in failures


def test_source_scan_cannot_be_bypassed_with_arbitrary_text_extensions(tmp_path):
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
    (tmp_path / "benchmark").mkdir()
    secret_name = "api_" + "key"
    machine_path = "/" + "mnt/private/x"
    (tmp_path / "benchmark/opaque.payload").write_text(
        f"{secret_name}='abcdefgh'\npath={machine_path}\n")
    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "credential literal/assignment: benchmark/opaque.payload" in failures
    assert "machine-local absolute path: benchmark/opaque.payload" in failures


def test_source_scan_rejects_generic_posix_machine_paths_but_allows_intentional_tmp_output(tmp_path):
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
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
    shutil.copytree(ROOT / "dataset", tmp_path / "dataset")
    (tmp_path / "docs/plans").mkdir(parents=True)
    research_name = "1-" + "research"
    machine_path = "/" + "mnt/private/x"
    plan = tmp_path / "docs/plans/release.md"
    plan.write_text(f"historical input: {research_name}\nlocal input: {machine_path}\n")
    failures = "\n".join(validator.scan_release(tmp_path)["leakage"]["failures"])
    assert "machine-local absolute path: docs/plans/release.md" in failures
    assert "research-tree runtime reference: docs/plans/release.md" not in failures
