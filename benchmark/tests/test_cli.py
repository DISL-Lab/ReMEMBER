import pytest

import remember_benchmark.cli as cli


def test_cli_requires_explicit_work_and_output_directories():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["build", "--config", "paper.yaml"])


def test_cli_forwards_single_build_command(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(cli, "build", lambda **kwargs: called.update(kwargs) or type("R", (), {"stage_names": []})())
    config = tmp_path / "config.yaml"
    config.write_text("models: {}\n")
    assert cli.main(["build", "--config", str(config), "--work-dir", str(tmp_path / "work"),
                     "--output-dir", str(tmp_path / "out"), "--provider", "fake"]) == 0
    assert called["work_dir"] == tmp_path / "work"
    assert called["output_dir"] == tmp_path / "out"
