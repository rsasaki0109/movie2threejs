#!/usr/bin/env bash
# gsplat checkpoint refinement only. Does not allocate remote/paid resources.
set -euo pipefail
unset PYTHONPATH
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/gpu-revisions.env"
REFINE_ENV="${PLAYWORLD_REFINE_ENV:-$HOME/.local/share/playworld/refine-gpu}"
REFINE_REPO="${PLAYWORLD_REFINE_REPO:-$HOME/.cache/playworld/gsplat-refine}"
UV="${PLAYWORLD_UV:-$(command -v uv || true)}"
[[ -n "$UV" ]] || UV="$HOME/.local/bin/uv"
[[ "$(uname -s)" == Linux ]] || { echo 'Use Linux or WSL with an NVIDIA GPU.' >&2; exit 1; }
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export MAX_JOBS="${MAX_JOBS:-2}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo 'CUDA 12.8 nvcc is required; see docs/gpu.md.' >&2; exit 1; }
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.8' || { echo 'This recipe was tested with CUDA 12.8.' >&2; exit 1; }
if [[ "${1:-}" == --check ]]; then
    "$REFINE_ENV/bin/python" - <<'PY'
import torch, gsplat, gsplat.csrc
assert torch.cuda.is_available()
print(torch.__version__, gsplat.__version__, torch.cuda.get_device_name(0))
PY
    exit 0
fi
[[ -x "$UV" ]] || { echo 'Install uv or set PLAYWORLD_UV to its executable.' >&2; exit 1; }
[[ -x "$REFINE_ENV/bin/python" ]] || "$UV" venv --python 3.12 "$REFINE_ENV"
PYTHON="$REFINE_ENV/bin/python"
"$UV" pip install --python "$PYTHON" --torch-backend cu128 'torch==2.9.1+cu128'
"$UV" pip install --python "$PYTHON" 'numpy==1.26.4' scipy pillow setuptools wheel ninja rich jaxtyping plyfile -e "$ROOT"
if [[ ! -d "$REFINE_REPO/.git" ]]; then
    git init "$REFINE_REPO"
fi
[[ -z "$(git -C "$REFINE_REPO" status --porcelain)" ]] || { echo 'Refusing to overwrite changed gsplat sources.' >&2; exit 1; }
git -C "$REFINE_REPO" fetch --depth 1 https://github.com/nerfstudio-project/gsplat.git "$GSPLAT_REV"
git -C "$REFINE_REPO" checkout --detach "$GSPLAT_REV"
git -C "$REFINE_REPO" submodule update --init --recursive --depth 1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$("$PYTHON" -c 'import torch; a,b=torch.cuda.get_device_capability(0); print(f"{a}.{b}")')}"
if ! "$PYTHON" - "$REFINE_REPO" <<'PY'
import sys
from pathlib import Path
import gsplat, gsplat.csrc
assert Path(gsplat.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
PY
then
    "$UV" pip install --python "$PYTHON" --no-build-isolation -e "$REFINE_REPO"
fi
"$UV" pip freeze --python "$PYTHON" > "$REFINE_ENV/refinement-freeze.txt"
"$PYTHON" -c 'import torch, gsplat.csrc; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
printf 'Ready. Interpreter: %s\nRun: %s %s/scripts/refine_checkpoint.py --help\n' "$PYTHON" "$PYTHON" "$ROOT"
