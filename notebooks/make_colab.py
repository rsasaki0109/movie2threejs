"""Generate the notebook; GPU setup and execution are shared with Linux scripts."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"id": f"playworld-{len(cells):02d}", "cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"id": f"playworld-{len(cells):02d}", "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# playworld: experimental phone video → physical browser world

This notebook completed the public video/VGGT recipe on a Colab L4 in
12 min 25.7 s, including initial model downloads. Environment setup took another
10 min 1.6 s. Browser preview and result ZIP download remain unverified.
See `docs/colab-ui.md` and `docs/benchmarks.md` for the measured run and its limits.

1. Select a GPU runtime. For movable objects use an Ampere-or-newer GPU (L4/A100 etc.).
   The pinned SAM 3 uses bf16; **T4 is not validated** and the script rejects it for SAM 3.
2. Request access to [facebook/sam3](https://huggingface.co/facebook/sam3), then store your
   token in Colab secrets as `HF_TOKEN` and grant this notebook access.
3. Open **Files** in the left sidebar and upload the current `playworld.zip` to
   session storage. Set `PROJECT_ZIP` below to its path if its filename differs.
   There is no public repository to clone yet.
4. The default input is an MIT-licensed public apartment capture from Eyeful Tower.
   It is a visualization of capture-rig photographs, **not a phone video**.
   To use a phone video instead, set `DATA_SOURCE = "upload"`.

The VGGT default weights have non-commercial restrictions. Read each model's terms before use.
""")

code("""
NUM_FRAMES = 24       # verified for the public workbench input on L4
PROJECT_ZIP = "/content/playworld.zip"  # upload using Files → Upload to session storage
DATA_SOURCE = "public"  # "public" = Eyeful Tower apartment; "upload" = your video
TRAIN_STEPS = 7000
PROMPTS = "cardboard box,plastic bottle,folding chair"
EYE_HEIGHT = 1.5      # actual height of the phone above the floor, meters
POSE_CONFIDENCE = 1.5 if DATA_SOURCE == "public" else 5.0
USE_BA = DATA_SOURCE == "public"
BA_REPROJECTION_ERROR = 32.0  # initial matching tolerance, not final BA error
assert DATA_SOURCE in {"public", "upload"}, "DATA_SOURCE must be public or upload"
assert NUM_FRAMES >= 2 and TRAIN_STEPS >= 1 and EYE_HEIGHT > 0
""")

code("""
# Load the project ZIP uploaded in the Files sidebar (not the room video yet).
from google.colab import files
from pathlib import Path
import zipfile
import hashlib
content = Path("/content").resolve()
project = content / "playworld"
archive_path = Path(PROJECT_ZIP).resolve()
assert archive_path.is_file(), "Upload the project ZIP in the Files sidebar, then set PROJECT_ZIP to its path"
digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
marker = project / ".playworld-project-sha256"
if project.exists():
    assert marker.is_file() and marker.read_text().strip() == digest, "Existing project differs; preserve its results and use a fresh runtime for a different ZIP"
else:
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            assert item.filename.startswith("playworld/") and (content / item.filename).resolve().is_relative_to(project), "Unsafe project ZIP path"
        assert "playworld/scripts/setup_gpu.sh" in archive.namelist(), "Not a playworld project ZIP"
        archive.extractall(content)
    marker.write_text(digest + "\\n", encoding="utf-8")
assert (project / "scripts/setup_gpu.sh").is_file(), "ZIP must contain playworld/scripts/setup_gpu.sh"
import sys
sys.path.insert(0, str(project / "scripts"))
from notebook_support import NotebookRun
run = NotebookRun(content, project, {
    "num_frames": NUM_FRAMES, "data_source": DATA_SOURCE, "train_steps": TRAIN_STEPS,
    "prompts": PROMPTS, "eye_height": EYE_HEIGHT, "pose_confidence": POSE_CONFIDENCE,
    "use_ba": USE_BA, "ba_reprojection_error": BA_REPROJECTION_ERROR,
})
print("This attempt:", run.path)
""")

code("""
import os
os.environ["PLAYWORLD_NUM_FRAMES"] = str(NUM_FRAMES)
os.environ["PLAYWORLD_TRAIN_STEPS"] = str(TRAIN_STEPS)
os.environ["PLAYWORLD_PROMPTS"] = PROMPTS
os.environ["PLAYWORLD_POSE_CONFIDENCE"] = str(POSE_CONFIDENCE)
import json
from colab_job import preflight
# Shared with CLI: accepts Colab Secrets, an existing environment token, or the
# one-use transfer documented in docs/colab-cli.md. Never print a token.
ready = preflight()
run.report["preflight"] = json.loads((content / "playworld-preflight.json").read_text())
run.save()
assert ready, "Check GPU (L4/A100 for SAM 3), HF_TOKEN access, and SAM 3 model approval before setup"
""")

code("""
run.execute(["bash", str(project / "scripts/setup_gpu.sh")], "setup")
""")

code("""
import subprocess
if DATA_SOURCE == "public":
    run.execute([
        "/content/playworld/.gpu/envs/core/bin/python",
        "/content/playworld/scripts/download_public.py", "--out", str(run.source), "--workshop",
    ], "source")
    VIDEO = str(run.source / "apartment-workshop.mp4")
elif DATA_SOURCE == "upload":
    video_upload = files.upload()
    assert len(video_upload) == 1, "Upload one room video"
    VIDEO = str(Path("/content") / next(iter(video_upload)))
else:
    raise ValueError("DATA_SOURCE must be public or upload")
""")

code("""
import subprocess
command = [
    "bash", str(project / "scripts/run.sh"), VIDEO, str(run.scene), str(run.world),
    "--eye-height", str(EYE_HEIGHT),
]
if USE_BA:
    command += ["--ba", "--shared-camera", "--ba-reprojection-error", str(BA_REPROJECTION_ERROR)]
run.execute(command, "pipeline")
""")

md("""
## Inspect the result

Check floor height, object masks, collisions and holes before recording. **WASD** walk,
**Space** jump, **Click** throw, **E** push, **R** reset, **C** show colliders.
Each attempt has its own directory, printed after project upload. Stage times and
failures are saved in `scene/run-*.json`, logs and `job.json` under that directory.
One public-input L4 conversion took 12 min 25.7 s, plus 10 min 1.6 s for setup;
this is not a timing guarantee for other videos.
If a stage fails, run the final download cell anyway to export the logs and available
measurements. A failed run cannot be used as the README hero.
""")

code("""
assert run.report["completed"] and (run.world / "world.json").is_file(), "Finish reconstruction before preview"
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from threading import Thread
if "preview_server" in globals():
    preview_server.shutdown()
    preview_server.server_close()
preview_server = ThreadingHTTPServer(("0.0.0.0", 0), partial(SimpleHTTPRequestHandler, directory=str(run.world)))
Thread(target=preview_server.serve_forever, daemon=True).start()
from google.colab.output import serve_kernel_port_as_window
serve_kernel_port_as_window(preview_server.server_port, path="/index.html")
""")

code("""
# Download measured evidence even if setup or reconstruction failed.
result_zip = run.bundle()
print(result_zip)
files.download(str(result_zip))
""")

nb = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).with_name("playworld_colab.ipynb")
out.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
print(out)
