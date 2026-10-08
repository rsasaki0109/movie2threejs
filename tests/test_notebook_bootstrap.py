import importlib.util
import subprocess
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "notebook_bootstrap", Path(__file__).parents[1] / "scripts/notebook_bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    for name in ("scripts/setup_gpu.sh", "src/playworld/cli.py", "notebooks/test.ipynb",
                 "demo-assets/large.spz", "README.md"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test source\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return repo, revision


def test_sparse_fetch_and_repeat_preserve_existing_results(tmp_path, source_repo):
    repo, revision = source_repo
    content = tmp_path / "content"
    project = bootstrap.load_project(content, revision, str(repo))
    assert (project / "scripts/setup_gpu.sh").is_file()
    assert not (project / "demo-assets").exists()
    result = project / ".gpu/result.pt"
    result.parent.mkdir()
    result.write_bytes(b"saved work")
    assert bootstrap.load_project(content, revision, str(repo)) == project
    assert result.read_bytes() == b"saved work"
    (project / "README.md").write_text("edited")
    with pytest.raises(RuntimeError, match="edited source"):
        bootstrap.load_project(content, revision, str(repo))
    assert (project / "README.md").read_text() == "edited"


def test_mismatched_revision_and_non_git_project_are_preserved(tmp_path, source_repo):
    repo, revision = source_repo
    content = tmp_path / "content"
    project = bootstrap.load_project(content, revision, str(repo))
    with pytest.raises(RuntimeError, match="differs"):
        bootstrap.load_project(content, "f" * 40, str(repo))
    other = tmp_path / "other/playworld"
    other.mkdir(parents=True)
    (other / "result.pt").write_bytes(b"saved")
    with pytest.raises(RuntimeError, match="another source"):
        bootstrap.load_project(other.parent, revision, str(repo))
    assert (other / "result.pt").read_bytes() == b"saved"


def test_invalid_revision_and_failed_checkout_leave_no_partial_project(tmp_path, source_repo):
    repo, _ = source_repo
    content = tmp_path / "content"
    with pytest.raises(ValueError, match="immutable"):
        bootstrap.load_project(content, "master", str(repo))
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.load_project(content, "f" * 40, str(repo))
    assert not (content / "playworld").exists()
    assert list(content.iterdir()) == []
