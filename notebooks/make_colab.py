"""Generates playworld_colab.ipynb (kept as code so the notebook stays reviewable in diffs)."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# playworld: phone video → walkable, physical 3D world

1. **Runtime → Change runtime type → GPU** (T4 works; L4/A100 is faster).
2. For movable objects you need access to the gated [facebook/sam3](https://huggingface.co/facebook/sam3) weights
   and a Hugging Face token in Colab secrets as `HF_TOKEN`. Without it, set `PROMPTS = ""` (static world only).
3. Run all cells. Upload a 20–40 s video: walk slowly through a room, phone upright, things on the floor/table.
""")

code("""
# --- settings -------------------------------------------------------------
NUM_FRAMES = 48        # VGGT memory grows with this; ~48 fits a 16 GB T4
TRAIN_STEPS = 7000     # gaussian splatting iterations
PROMPTS = "chair,box,cup,mug,bottle,ball,plant pot,lamp,backpack,book,pillow,stool"
EYE_HEIGHT = 1.5       # how high you held the phone, meters (sets the metric scale)
PLAYWORLD_REPO = "https://github.com/rsasaki0109/playworld"  # or upload playworld.zip to /content
import os; os.environ["PLAYWORLD_REPO"] = PLAYWORLD_REPO  # read by the %%bash setup cell
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import time; T0 = time.time()
""")

code("""
%%bash
# Three isolated environments: VGGT, gsplat and SAM 3 pin incompatible numpy/torch versions.
set -e
cd /content
pip -q install uv
[ -d vggt ] || git clone -q https://github.com/facebookresearch/vggt.git
[ -d gsplat ] || git clone -q https://github.com/nerfstudio-project/gsplat.git
if [ -f playworld.zip ] && [ ! -d playworld ]; then unzip -q playworld.zip -d playworld_src && mv playworld_src/* playworld; fi
[ -d playworld ] || git clone -q "$PLAYWORLD_REPO" playworld
pip -q install -e playworld

TORCH="torch==2.9.1 torchvision==0.24.1"
uv venv -q --python 3.12 envs/vggt
uv pip install -q --python envs/vggt/bin/python --torch-backend=auto $TORCH "numpy<2" \\
  pillow huggingface_hub einops safetensors scipy trimesh opencv-python-headless \\
  pycolmap==3.10.0 pyceres==2.3 "lightglue @ git+https://github.com/jytime/LightGlue.git" -e vggt
echo vggt ready

uv venv -q --python 3.12 envs/gsplat
uv pip install -q --python envs/gsplat/bin/python --torch-backend=auto $TORCH
uv pip install -q --python envs/gsplat/bin/python --torch-backend=auto --no-build-isolation gsplat -r gsplat/examples/requirements.txt
echo gsplat ready

uv venv -q --python 3.12 envs/sam3
uv pip install -q --python envs/sam3/bin/python --torch-backend=auto $TORCH "numpy<2" pillow \\
  "sam3 @ git+https://github.com/facebookresearch/sam3.git"
echo sam3 ready
""")

code("""
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception as e:
    print("no HF_TOKEN secret:", e, "-> set PROMPTS = '' for a static world")
print(f"setup: {time.time() - T0:.0f} s")
""")

code("""
from google.colab import files
up = files.upload()
VIDEO = "/content/" + next(iter(up))
print(VIDEO)
""")

code("""
!playworld all "$VIDEO" /content/scene --out /content/world \\
  --num {NUM_FRAMES} --steps {TRAIN_STEPS} --prompts "{PROMPTS}" --eye-height {EYE_HEIGHT} \\
  --vggt-dir /content/vggt --gsplat-dir /content/gsplat \\
  --vggt-python /content/envs/vggt/bin/python \\
  --gsplat-python /content/envs/gsplat/bin/python \\
  --sam3-python /content/envs/sam3/bin/python
""")

md("""
## Play it

The next cell serves the world and opens it in a new browser tab. Click to look around,
**WASD** walk, **Click** throw a ball, **E** push, **R** reset, **C** show colliders.
""")

code("""
import subprocess
subprocess.Popen(["python", "-m", "http.server", "8000", "-d", "/content/world"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
from google.colab.output import serve_kernel_port_as_window
serve_kernel_port_as_window(8000, path="/index.html")
""")

code("""
# Download the world to host it anywhere (GitHub Pages, itch.io, ...)
!cd /content && zip -qr world.zip world && ls -lh world.zip
files.download("/content/world.zip")
""")

nb = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).with_name("playworld_colab.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(out)
