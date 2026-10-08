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


def completed_run(tmp_path):
    run = support.NotebookRun(tmp_path, tmp_path / 'project', {'train_steps': 7000})
    run.world.mkdir()
    (run.world / 'world.json').write_text('{}')
    run.execute([sys.executable, '-c', 'pass'], 'pipeline')
    checkpoint = run.scene / 'gs' / 'ckpts' / 'ckpt_6999_rank0.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'final synthetic weights')
    return run


def test_checkpoint_is_published_before_zip_is_packaged(tmp_path, monkeypatch):
    run = completed_run(tmp_path)
    original_bundle = run.bundle
    def bundle():
        index = json.loads((run.path / 'backup-index.json').read_text())
        assert [a['kind'] for a in index['artifacts']] == ['final_checkpoint']
        assert not index['complete']
        return original_bundle()
    monkeypatch.setattr(run, 'bundle', bundle)
    index = json.loads(run.prepare_backups().read_text())
    assert index['complete']
    assert [a['kind'] for a in index['artifacts']] == ['final_checkpoint', 'world_zip']
    for artifact in index['artifacts']:
        manifest = json.loads((Path(artifact['directory']) / 'manifest.json').read_text())
        assert sum(p['bytes'] for p in manifest['parts']) == manifest['bytes']


def test_failed_zip_preserves_ready_checkpoint_and_retry_invalidates_index(tmp_path, monkeypatch):
    run = completed_run(tmp_path)
    def fail():
        raise OSError('fixture archive failure')
    monkeypatch.setattr(run, 'bundle', fail)
    with pytest.raises(OSError):
        run.prepare_backups()
    index = json.loads((run.path / 'backup-index.json').read_text())
    assert not index['complete'] and index['error_type'] == 'OSError'
    assert index['artifacts'][0]['kind'] == 'final_checkpoint'
    run.execute([sys.executable, '-c', 'pass'], 'setup')
    assert not (run.path / 'backup-index.json').exists()
    assert 'backups' not in run.report


def test_missing_final_checkpoint_does_not_offer_an_intermediate_backup(tmp_path):
    run = completed_run(tmp_path)
    (run.scene / 'gs' / 'ckpts' / 'ckpt_6999_rank0.pt').rename(
        run.scene / 'gs' / 'ckpts' / 'ckpt_4665_rank0.pt')
    with pytest.raises(FileNotFoundError):
        run.prepare_backups()
    index = json.loads((run.path / 'backup-index.json').read_text())
    assert not index['complete'] and not index['artifacts']
