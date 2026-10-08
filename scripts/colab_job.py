"""Colab CLI entry point: check an existing GPU, then explicitly run the pipeline.

Execute with `colab exec -s SESSION -f scripts/colab_job.py`. This script never
allocates a runtime. Upload playworld-colab.zip to /content before ACTION=run.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


CONTENT = Path("/content")


def preflight():
    import torch

    prompts = os.environ.get("PLAYWORLD_PROMPTS", "chair,box,mug,bottle")
    gpu = torch.cuda.is_available()
    properties = torch.cuda.get_device_properties(0) if gpu else None
    token = os.environ.get("HF_TOKEN")
    token_file = CONTENT / ".playworld-hf-token"
    if not token and token_file.is_file():
        token_file.chmod(0o600)
        token = token_file.read_text().strip()
        token_file.unlink()
    if prompts.strip() and not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    # Clipboard/Secrets values can include a trailing newline. Never let an
    # invalid header reach urllib, whose exception includes the credential.
    token = token.strip() if isinstance(token, str) else None
    invalid_token = bool(token) and not re.fullmatch(r"[\x21-\x7e]+", token)
    if invalid_token:
        token = None
    access = "not required" if not prompts.strip() else (
        "HF_TOKEN format invalid" if invalid_token else "HF_TOKEN not available")
    if prompts.strip() and token:
        request = urllib.request.Request(
            "https://huggingface.co/facebook/sam3/resolve/main/config.json",
            headers={"Authorization": "Bearer " + token},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                access = "approved" if response.status == 200 else f"HTTP {response.status}"
        except urllib.error.HTTPError as error:
            access = f"HTTP {error.code}"
        except urllib.error.URLError:
            access = "network check failed"
    ready = gpu and (not prompts.strip() or (properties.major >= 8 and access == "approved"))
    report = {"gpu": properties.name if gpu else None,
              "gpu_memory_mib": properties.total_memory // 2**20 if gpu else None,
              "native_bf16": properties.major >= 8 if gpu else False,
              "sam3_access": access, "ready": ready}
    CONTENT.mkdir(exist_ok=True)
    (CONTENT / "playworld-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if token:
        os.environ["HF_TOKEN"] = token  # never echo the token or pass it as a CLI argument
    return ready


def unpack_project():
    archive_path = CONTENT / "playworld-colab.zip"
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    project = CONTENT / "playworld"
    marker = project / ".playworld-project-sha256"
    if project.exists():
        if not marker.is_file() or marker.read_text().strip() != digest:
            raise RuntimeError("Existing /content/playworld differs from the uploaded ZIP; use a fresh runtime or explicitly upload the intended changed files")
        return project
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if not item.filename.startswith("playworld/") or not (CONTENT / item.filename).resolve().is_relative_to(project):
                raise ValueError("Unsafe project ZIP path")
        archive.extractall(CONTENT)
    if not (project / "scripts/setup_gpu.sh").is_file():
        raise ValueError("Project ZIP is missing scripts/setup_gpu.sh")
    marker.write_text(digest + "\n")
    return project


def setup(project):
    with (CONTENT / "setup.log").open("w") as log:
        process = subprocess.Popen(["bash", str(project / "scripts/setup_gpu.sh")],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, process.args)


def setup_or_reuse(project):
    if os.environ.get("PLAYWORLD_CLI_SKIP_SETUP") != "1":
        setup(project)
        return
    # An explicit reuse request still needs evidence that every setup check passed.
    log = CONTENT / "setup.log"
    gpu_root = Path(os.environ.get("PLAYWORLD_GPU_ROOT", project / ".gpu"))
    required = [gpu_root / f"{stage}-freeze.txt" for stage in ("core", "vggt", "gsplat", "sam3")]
    required += [gpu_root / "vggt-help.txt", gpu_root / "gsplat-help.txt"]
    if not log.is_file() or "Setup complete (no model inference yet)." not in log.read_text():
        raise RuntimeError("Cannot reuse setup: no successful setup log")
    if any(not path.is_file() or not path.stat().st_size for path in required):
        raise RuntimeError("Cannot reuse setup: environment/API check evidence is missing")
    print("Reusing the explicitly requested, checked GPU environments.", flush=True)


def run_streamed(command):
    # Child stdout must be relayed through the kernel's IOPub stream for CLI users.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)


def bundle(completed):
    # A failed retry must not export an earlier successful world as its own result.
    staging = CONTENT / "playworld-cli-result"
    staging.mkdir(exist_ok=False)
    for name in ["setup.log", "playworld-preflight.json"]:
        if (CONTENT / name).is_file():
            shutil.copy(CONTENT / name, staging / name)
    (staging / "job.json").write_text(json.dumps({"completed": completed}) + "\n")
    if completed:
        shutil.copytree(CONTENT / "world", staging / "world")
    for pattern in ["scene/run-*", "scene/gpu-revisions.env", "playworld/.gpu/*-freeze.txt",
                    "public_source/*.json", "public_source/EYEFULTOWER-LICENSE.txt"]:
        for path in CONTENT.glob(pattern):
            if path.is_file():
                shutil.copy(path, staging / path.name)
    result = shutil.make_archive(str(CONTENT / "playworld-result"), "zip", root_dir=staging)
    print("Download result with colab download:", result, flush=True)


def main():
    action = os.environ.get("PLAYWORLD_CLI_ACTION", "check")
    if action not in {"check", "setup", "run"}:
        raise ValueError("PLAYWORLD_CLI_ACTION must be check, setup or run")
    os.environ.setdefault('PLAYWORLD_PROMPTS', 'cardboard box,plastic bottle,folding chair')
    os.environ.setdefault('PLAYWORLD_POSE_CONFIDENCE', '1.5')
    ready = preflight()
    if action == "check":
        return
    if action == "setup":
        # Installation is independent of gated weight approval. Never infer
        # that a successful installation proves the GPU pipeline works.
        if not json.loads((CONTENT / "playworld-preflight.json").read_text())["gpu"]:
            raise RuntimeError("Select a GPU runtime before setup")
        setup(unpack_project())
        return
    if not ready:
        raise RuntimeError("GPU/SAM 3 preflight failed; inspect playworld-preflight.json before running")
    if (CONTENT / "playworld-cli-result").exists():
        raise RuntimeError("Download and preserve the existing result before reusing this runtime")
    project = unpack_project()
    completed = False
    try:
        setup_or_reuse(project)
        core = project / ".gpu/envs/core/bin/python"
        run_streamed([str(core), str(project / "scripts/download_public.py"),
                      "--out", str(CONTENT / "public_source"), '--workshop'])
        run_streamed(["bash", str(project / "scripts/run.sh"),
                      str(CONTENT / "public_source/apartment-workshop.mp4"),
                      str(CONTENT / "scene"), str(CONTENT / "world"),
                      '--ba', '--shared-camera', '--ba-reprojection-error', '32.0'])
        completed = True
    finally:
        bundle(completed)


if __name__ == "__main__":
    main()
