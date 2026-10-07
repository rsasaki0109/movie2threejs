"""Colab CLI entry point: check an existing GPU, then explicitly run the pipeline.

Execute with `colab exec -s SESSION -f scripts/colab_job.py`. This script never
allocates a runtime. Upload playworld-colab.zip to /content before ACTION=run.
"""
import hashlib
import json
import os
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
    if prompts.strip() and not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    access = "not required" if not prompts.strip() else "HF_TOKEN not available"
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
    if action not in {"check", "run"}:
        raise ValueError("PLAYWORLD_CLI_ACTION must be check or run")
    ready = preflight()
    if action == "check":
        return
    if not ready:
        raise RuntimeError("GPU/SAM 3 preflight failed; inspect playworld-preflight.json before running")
    if (CONTENT / "playworld-cli-result").exists():
        raise RuntimeError("Download and preserve the existing result before reusing this runtime")
    project = unpack_project()
    completed = False
    try:
        setup(project)
        core = project / ".gpu/envs/core/bin/python"
        subprocess.run([str(core), str(project / "scripts/download_public.py"),
                        "--out", str(CONTENT / "public_source")], check=True)
        subprocess.run(["bash", str(project / "scripts/run.sh"),
                        str(CONTENT / "public_source/apartment-camera19.mp4"),
                        str(CONTENT / "scene"), str(CONTENT / "world")], check=True)
        completed = True
    finally:
        bundle(completed)


if __name__ == "__main__":
    main()
