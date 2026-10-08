"""Fetch only notebook sources at an immutable Git revision, without replacing work."""
import re
import subprocess
import tempfile
from pathlib import Path


PROJECT_REPOSITORY = "https://github.com/rsasaki0109/movie2threejs.git"


def load_project(content, revision, repository=PROJECT_REPOSITORY):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full immutable Git commit SHA")
    content = Path(content).resolve()
    content.mkdir(parents=True, exist_ok=True)
    project = content / "playworld"

    def git(*args, cwd=project):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    if project.exists():
        if not (project / ".git").is_dir():
            raise RuntimeError("Existing project has another source; preserve its results and use a fresh runtime")
        if git("remote", "get-url", "origin") != repository or git("rev-parse", "HEAD") != revision:
            raise RuntimeError("Existing project differs from this notebook's revision; preserve its results and use a fresh runtime")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("Existing project has edited source files; preserve them and use a fresh runtime")
    else:
        # A failed fetch/checkout leaves no half-created project directory. All
        # cleanup stays inside this newly created temporary directory.
        with tempfile.TemporaryDirectory(prefix="playworld-fetch-", dir=content) as temporary:
            checkout = Path(temporary) / "checkout"
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                            "--", repository, str(checkout)], check=True)
            git("sparse-checkout", "init", "--cone", cwd=checkout)
            git("sparse-checkout", "set", "src", "scripts", "notebooks", cwd=checkout)
            git("checkout", "--detach", revision, cwd=checkout)
            if not (checkout / "scripts/setup_gpu.sh").is_file():
                raise RuntimeError("Project revision is missing the GPU setup script")
            checkout.rename(project)
    print("Project revision:", git("rev-parse", "HEAD"))
    return project
