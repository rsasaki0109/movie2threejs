#!/usr/bin/env bash
# Optional CPU-only SPZ encoder; uses the existing core environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ROOT="${PLAYWORLD_GPU_ROOT:-$ROOT/.gpu}"
CORE="$GPU_ROOT/envs/core/bin/python"
[[ -x "$CORE" ]] || { echo 'First run scripts/setup_gpu.sh' >&2; exit 1; }
# Some Colab images provide a non-PIC static zstd. Build a private PIC copy
# rather than altering the system libraries or the Torch stage environments.
CMAKE_ARGS='-DCMAKE_DISABLE_FIND_PACKAGE_zstd=ON -DSPZ_BUILD_TOOLS=OFF' \
CMAKE_BUILD_PARALLEL_LEVEL=2 uv pip install --python "$CORE" \
  'spz @ git+https://github.com/nianticlabs/spz.git@affd0ecea7fbb4c265ee119475af7ee5b2997482'
