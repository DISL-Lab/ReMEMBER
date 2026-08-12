import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


PACKAGE = Path(__file__).parents[1]


def test_wheel_contains_runtime_prompts_and_config_and_reads_them_from_zip(tmp_path):
    for relative in (
        "prompts/gap_detection.txt",
        "prompts/summary_with_memory.txt",
        "prompts/summary_window_only.txt",
        "configs/paper.yaml",
    ):
        assert (PACKAGE / relative).read_bytes() == (PACKAGE / "src/remember" / relative).read_bytes()

    project = tmp_path / "remember-project"
    shutil.copytree(
        PACKAGE,
        project,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"),
    )
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--wheel-dir", str(wheels), str(project)],
        text=True, capture_output=True, check=False,
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    [wheel] = list(wheels.glob("remember-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert {
        "remember/prompts/gap_detection.txt",
        "remember/prompts/summary_with_memory.txt",
        "remember/prompts/summary_window_only.txt",
        "remember/configs/paper.yaml",
    } <= names

    probe = subprocess.run(
        [sys.executable, "-c", "from remember.pipeline import _read_prompt; print(_read_prompt('gap_detection.txt')[:20])"],
        text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONPATH": str(wheel), "PYTHONDONTWRITEBYTECODE": "1"},
        cwd=tmp_path,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert probe.stdout == "You are building ret\n"
