#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
export MPLBACKEND=Agg
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ROOT="${PLAYWORLD_GPU_ROOT:-$ROOT/.gpu}"
if [[ "${1:-}" == "--check" ]]; then
  printf 'Core: %s\nStages: %s\n' "$GPU_ROOT/envs/core/bin/python" "$GPU_ROOT/envs/{vggt,gsplat,sam3}/bin/python"
  exit 0
fi
[[ $# -ge 1 ]] || { echo 'Usage: bash scripts/run.sh VIDEO [SCENE] [OUT] [playworld all options...]' >&2; exit 2; }
VIDEO="$1"; shift
SCENE="${1:-$ROOT/scenes/room}"; [[ $# -eq 0 ]] || shift
OUT="${1:-$ROOT/world}"; [[ $# -eq 0 ]] || shift
[[ -f "$VIDEO" ]] || { echo "Video not found: $VIDEO" >&2; exit 1; }
for stage in core vggt gsplat sam3; do
  [[ -x "$GPU_ROOT/envs/$stage/bin/python" ]] || { echo 'First run bash scripts/setup_gpu.sh' >&2; exit 1; }
done
PROMPTS="${PLAYWORLD_PROMPTS-chair,box,mug,bottle}"
if [[ -n "$PROMPTS" ]]; then
  "$GPU_ROOT/envs/sam3/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
    raise SystemExit('The pinned SAM 3 uses bf16 autocast. Use an Ampere-or-newer GPU (L4/A100 etc.) for movable objects. T4 is not validated; static-only runs can set PLAYWORLD_PROMPTS="".')
from huggingface_hub import hf_hub_download
# Checks gated access before spending time on reconstruction and training.
hf_hub_download('facebook/sam3', 'config.json')
PY
fi
mkdir -p "$SCENE"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
cp "$ROOT/scripts/gpu-revisions.env" "$SCENE/gpu-revisions.env"
"$GPU_ROOT/envs/core/bin/python" -m playworld.cli all "$VIDEO" "$SCENE" --out "$OUT" \
  --vggt-dir "$GPU_ROOT/repos/vggt" --gsplat-dir "$GPU_ROOT/repos/gsplat" \
  --vggt-python "$GPU_ROOT/envs/vggt/bin/python" --gsplat-python "$GPU_ROOT/envs/gsplat/bin/python" \
  --sam3-python "$GPU_ROOT/envs/sam3/bin/python" --num "${PLAYWORLD_NUM_FRAMES:-24}" \
  --steps "${PLAYWORLD_TRAIN_STEPS:-7000}" --prompts "$PROMPTS" \
  --pose-confidence "${PLAYWORLD_POSE_CONFIDENCE:-5.0}" \
  --report "$SCENE/run-$STAMP.json" "$@" 2>&1 | tee "$SCENE/run-$STAMP.log"
