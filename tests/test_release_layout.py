import importlib.util
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
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/validate_release.py")],
                               cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    summary = json.loads(completed.stdout)
    assert summary["ok"] is True
    assert summary["compile"]["ok"] is True
    assert summary["tests"]["ok"] is True
    assert summary["smokes"]["ok"] is True


def test_documentation_and_three_product_folders_exist():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "benchmark/README.md").is_file()
    assert {path.name for path in ROOT.iterdir() if path.is_dir() and path.name in {"dataset", "benchmark", "remember"}} == {
        "dataset", "benchmark", "remember"
    }


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
