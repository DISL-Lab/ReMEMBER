import hashlib
import json
from pathlib import Path

from remember_benchmark.io import iter_jsonl
from remember_benchmark.release import validate_annotations


DATASET = Path(__file__).parents[2] / "dataset"


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_dataset_has_paper_counts_and_no_text_fields():
    rows = list(iter_jsonl(DATASET / "annotations/remember_900_annotations.jsonl"))
    stats = validate_annotations(rows, read_ids(DATASET / "splits/main_900.txt"), (900, 1498, 4135))
    assert stats == {"windows": 900, "gaps": 1498, "evidence": 4135}


def test_published_metadata_hashes_distributed_annotation_and_split_files():
    metadata = json.loads((DATASET / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifacts"] == {
        "annotations": {
            "path": "annotations/remember_900_annotations.jsonl",
            "sha256": sha256(DATASET / "annotations/remember_900_annotations.jsonl"),
        },
        "split": {
            "path": "splits/main_900.txt",
            "sha256": sha256(DATASET / "splits/main_900.txt"),
        },
    }
