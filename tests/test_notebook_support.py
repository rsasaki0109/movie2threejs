import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "notebook_support", Path(__file__).parents[1] / "scripts/notebook_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def test_failed_retry_download_excludes_previous_world_and_keeps_evidence(tmp_path):
    run = support.NotebookRun(tmp_path, tmp_path / "project", {"frames": 24})
    run.world.mkdir()
    (run.world / "world.json").write_text('{"old": true}')
    run.execute([sys.executable, "-c", "print('success')"], "pipeline")
    with zipfile.ZipFile(run.bundle()) as archive:
        assert "world/world.json" in archive.namelist()
    with pytest.raises(Exception):
        run.execute([sys.executable, "-c", "print('failed retry'); raise SystemExit(3)"], "pipeline")
    with zipfile.ZipFile(run.bundle()) as archive:
        assert not any(name.startswith("world/") for name in archive.namelist())
        receipt = json.loads(archive.read("job.json"))
        assert not receipt["completed"]
        assert receipt["stages"]["pipeline"]["status"] == "failed"
        assert b"failed retry" in archive.read("pipeline.log")


def test_setup_failure_can_be_exported_before_input_exists(tmp_path):
    run = support.NotebookRun(tmp_path, tmp_path / "project", {})
    with pytest.raises(FileNotFoundError):
        run.execute([str(tmp_path / "missing-command")], "setup")
    with zipfile.ZipFile(run.bundle()) as archive:
        assert json.loads(archive.read("job.json"))["stages"]["setup"]["status"] == "failed"
        assert "setup.log" in archive.namelist()


def test_new_attempt_is_separate_and_success_requires_world(tmp_path):
    first = support.NotebookRun(tmp_path, tmp_path / "project", {})
    second = support.NotebookRun(tmp_path, tmp_path / "project", {})
    assert first.path != second.path
    with pytest.raises(RuntimeError, match="world.json"):
        second.execute([sys.executable, "-c", "pass"], "pipeline")
    assert not json.loads((second.path / "job.json").read_text())["completed"]


def test_carriage_return_progress_keeps_full_log_without_flooding_notebook(tmp_path, capsys):
    run = support.NotebookRun(tmp_path, tmp_path / "project", {})
    script = "import sys; print('start'); [sys.stdout.write('10%| step '+str(i)+'\\r') for i in range(50)]; print('done')"
    run.execute([sys.executable, "-c", script], "setup")
    display = capsys.readouterr().out
    log = (run.path / "setup.log").read_text(encoding="utf-8")
    assert log.count("10%| step") == 50
    assert display.count("10%| step") <= 2
    assert "start" in display and "done" in display and "step 49" in display
