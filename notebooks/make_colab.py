"""Generate the notebook; GPU setup and execution are shared with Linux scripts."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# playworld: experimental phone video → physical browser world

**This GPU recipe has not yet been validated end to end on a real video.**

1. Select a GPU runtime. For movable objects use an Ampere-or-newer GPU (L4/A100 etc.).
   The pinned SAM 3 uses bf16; **T4 is not validated** and the script rejects it for SAM 3.
2. Request access to [facebook/sam3](https://huggingface.co/facebook/sam3), then store your
   token in Colab secrets as `HF_TOKEN` and grant this notebook access.
3. Upload the current `playworld.zip` below. There is no public repository to clone yet.
4. The default input is an MIT-licensed public apartment capture from Eyeful Tower.
   It is a visualization of capture-rig photographs, **not a phone video**.
   To use a phone video instead, set `DATA_SOURCE = "upload"`.

The VGGT default weights have non-commercial restrictions. Read each model's terms before use.
""")

code("""
NUM_FRAMES = 24       # starting setting, not a measured memory guarantee
DATA_SOURCE = "public"  # "public" = Eyeful Tower apartment; "upload" = your video
TRAIN_STEPS = 7000    # tune after the first real run
PROMPTS = "chair,box,mug,bottle"   # use "" only for an explicitly static world
EYE_HEIGHT = 1.5      # actual height of the phone above the floor, meters
""")

code("""
# Upload the project ZIP (not the room video yet).
from google.colab import files
from pathlib import Path
import zipfile
project_upload = files.upload()
archives = [name for name in project_upload if name.lower().endswith(".zip")]
assert len(archives) == 1, "Upload exactly one playworld project ZIP"
content = Path("/content").resolve()
project = content / "playworld"
assert not project.exists(), "Project already exists; restart the runtime to replace it"
with zipfile.ZipFile(content / archives[0]) as archive:
    for item in archive.infolist():
        assert (content / item.filename).resolve().is_relative_to(content), "Unsafe ZIP path"
    archive.extractall(content)
assert (project / "scripts/setup_gpu.sh").is_file(), "ZIP must contain playworld/scripts/setup_gpu.sh"
""")

code("""
import os
from google.colab import userdata
if PROMPTS.strip():
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["PLAYWORLD_NUM_FRAMES"] = str(NUM_FRAMES)
os.environ["PLAYWORLD_TRAIN_STEPS"] = str(TRAIN_STEPS)
os.environ["PLAYWORLD_PROMPTS"] = PROMPTS
import torch
assert torch.cuda.is_available(), "Select a GPU runtime first"
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**20, "MiB")
if PROMPTS.strip():
    assert torch.cuda.get_device_capability(0)[0] >= 8, "SAM 3 recipe requires Ampere or newer; choose L4/A100. T4 is not validated."
""")

code("""
%%bash
set -euo pipefail
cd /content/playworld
bash scripts/setup_gpu.sh 2>&1 | tee /content/setup.log
""")

code("""
import subprocess
if DATA_SOURCE == "public":
    subprocess.run([
        "/content/playworld/.gpu/envs/core/bin/python",
        "/content/playworld/scripts/download_public.py", "--out", "/content/public_source",
    ], check=True)
    VIDEO = "/content/public_source/apartment-camera19.mp4"
elif DATA_SOURCE == "upload":
    video_upload = files.upload()
    assert len(video_upload) == 1, "Upload one room video"
    VIDEO = str(Path("/content") / next(iter(video_upload)))
else:
    raise ValueError("DATA_SOURCE must be public or upload")
""")

code("""
import subprocess
subprocess.run([
    "bash", "/content/playworld/scripts/run.sh", VIDEO, "/content/scene", "/content/world",
    "--eye-height", str(EYE_HEIGHT),
], check=True)
""")

md("""
## Inspect the result

Check floor height, object masks, collisions and holes before recording. **WASD** walk,
**Space** jump, **Click** throw, **E** push, **R** reset, **C** show colliders.
Stage times and failures are saved under `/content/scene/run-*.json` and `.log`.
The 15 minute conversion target is not a measured result.
If a stage fails, run the final download cell anyway to export the logs and available
measurements. A failed run cannot be used as the README hero.
""")

code("""
server = subprocess.Popen([
    "python", "-m", "http.server", "8000", "--bind", "0.0.0.0", "-d", "/content/world"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
from google.colab.output import serve_kernel_port_as_window
serve_kernel_port_as_window(8000, path="/index.html")
""")

code("""
# Download measured evidence even if setup or reconstruction failed.
import shutil
bundle = Path("/content/playworld-result")
bundle.mkdir(exist_ok=True)
if Path("/content/world/world.json").is_file():
    shutil.copytree("/content/world", bundle / "world", dirs_exist_ok=True)
if Path("/content/setup.log").is_file():
    shutil.copy("/content/setup.log", bundle / "setup.log")
for name in list(Path("/content/scene").glob("run-*")) + [Path("/content/scene/gpu-revisions.env")]:
    if name.is_file(): shutil.copy(name, bundle / name.name)
for name in Path("/content/playworld/.gpu").glob("*-freeze.txt"):
    shutil.copy(name, bundle / name.name)
if DATA_SOURCE == "public":
    for name in Path("/content/public_source").glob("*.json"):
        shutil.copy(name, bundle / name.name)
    notice = Path("/content/public_source/EYEFULTOWER-LICENSE.txt")
    if notice.is_file(): shutil.copy(notice, bundle)
shutil.make_archive("/content/playworld-result", "zip", root_dir=bundle)
files.download("/content/playworld-result.zip")
""")

nb = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).with_name("playworld_colab.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
