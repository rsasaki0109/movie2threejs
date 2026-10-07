#!/usr/bin/env bash
# Linux (Colab / an already provisioned GPU host); does not create paid resources.
set -euo pipefail
unset PYTHONPATH
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ROOT="${PLAYWORLD_GPU_ROOT:-$ROOT/.gpu}"
source "$ROOT/scripts/gpu-revisions.env"
if [[ "${1:-}" == "--check" ]]; then
  printf 'GPU root: %s\nVGGT: %s\ngsplat: %s\nSAM 3: %s\n' "$GPU_ROOT" "$VGGT_REV" "$GSPLAT_REV" "$SAM3_REV"
  exit 0
fi
[[ "$(uname -s)" == "Linux" ]] || { echo 'Use a Linux GPU host or Colab; native Windows GPU setup is not validated.' >&2; exit 1; }
for cmd in git ffmpeg ffprobe nvidia-smi nvcc; do
  command -v "$cmd" >/dev/null || { echo "Missing $cmd (CUDA toolkit, not only a GPU driver, is required)." >&2; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h "$ROOT"
mkdir -p "$GPU_ROOT/repos" "$GPU_ROOT/envs"
python3 -m pip install uv
clone_revision() {
  local repo="$1" revision="$2" target="$3"
  if [[ ! -d "$target/.git" ]]; then
    git init "$target"
    git -C "$target" remote add origin "https://github.com/$repo.git"
  fi
  if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
    echo "Local changes in $target; refusing to overwrite them." >&2; exit 1
  fi
  git -C "$target" fetch --depth 1 origin "$revision"
  git -C "$target" checkout --detach "$revision"
}
clone_revision facebookresearch/vggt "$VGGT_REV" "$GPU_ROOT/repos/vggt"
clone_revision nerfstudio-project/gsplat "$GSPLAT_REV" "$GPU_ROOT/repos/gsplat"
clone_revision facebookresearch/sam3 "$SAM3_REV" "$GPU_ROOT/repos/sam3"

for stage in core vggt gsplat sam3; do
  [[ -x "$GPU_ROOT/envs/$stage/bin/python" ]] || uv venv --python 3.12 "$GPU_ROOT/envs/$stage"
done
CORE="$GPU_ROOT/envs/core/bin/python"
VGGT="$GPU_ROOT/envs/vggt/bin/python"
GSPLAT="$GPU_ROOT/envs/gsplat/bin/python"
SAM3="$GPU_ROOT/envs/sam3/bin/python"
uv pip install --python "$CORE" -e "$ROOT"
# Use upstream gsplat's stable examples with its matching fork of pycolmap;
# VGGT uses the official bindings in a different environment.
uv pip install --python "$VGGT" --torch-backend cu128 torch==2.9.1 torchvision==0.24.1 'numpy==1.26.4'
uv pip install --python "$VGGT" 'setuptools<81' scipy trimesh pycolmap==3.10.0 pyceres==2.3 \
  'lightglue @ git+https://github.com/jytime/LightGlue.git' -e "$GPU_ROOT/repos/vggt"
uv pip install --python "$GSPLAT" --torch-backend cu128 torch==2.9.1 torchvision==0.24.1 \
  'numpy==1.26.4' setuptools wheel ninja
uv pip install --python "$GSPLAT" --no-build-isolation -e "$GPU_ROOT/repos/gsplat" \
  -r "$GPU_ROOT/repos/gsplat/examples/requirements.txt"
uv pip install --python "$SAM3" --torch-backend cu128 torch==2.10.0 torchvision==0.25.0 'numpy==1.26.4'
uv pip install --python "$SAM3" 'setuptools<81' pillow einops psutil opencv-python-headless scipy \
  -e "$GPU_ROOT/repos/sam3"

for stage in vggt gsplat sam3; do
  "$GPU_ROOT/envs/$stage/bin/python" - "$stage" <<'PY'
import sys, torch
assert torch.cuda.is_available(), 'PyTorch cannot use the GPU'
p = torch.cuda.get_device_properties(0)
print(sys.argv[1], torch.__version__, p.name, p.total_memory // 2**20, 'MiB', 'capability', p.major, p.minor)
PY
done
# Resolve tyro/API issues before uploading a long video or starting training.
( cd "$GPU_ROOT/repos/vggt" && "$VGGT" demo_colmap.py --help ) > "$GPU_ROOT/vggt-help.txt"
( cd "$GPU_ROOT/repos/gsplat/examples" && "$GSPLAT" simple_trainer.py default --help ) > "$GPU_ROOT/gsplat-help.txt"
for flag in --no-normalize-world-space --save-ply --ply-steps --disable-video; do
  grep -q -- "$flag" "$GPU_ROOT/gsplat-help.txt" || { echo "Missing trainer flag $flag" >&2; exit 1; }
done
"$SAM3" -c 'from sam3.model_builder import build_sam3_video_predictor; print("SAM 3 API imports (weights not loaded)")'
for stage in core vggt gsplat sam3; do
  uv pip freeze --python "$GPU_ROOT/envs/$stage/bin/python" > "$GPU_ROOT/$stage-freeze.txt"
done
printf '\nSetup complete (no model inference yet). Run: bash scripts/run.sh /path/to/room.mp4\n'
