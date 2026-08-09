#!/usr/bin/env bash
# Power Foam requires CUDA and Warp. Run this on a CUDA host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv venv --python 3.11 "$ROOT/.venv-powerfoam"
source "$ROOT/.venv-powerfoam/bin/activate"
uv pip install torch torchvision
uv pip install -r "$ROOT/external/powerfoam/requirements.txt"
uv pip install -e "$ROOT"
# Open3D imports libGL even for headless rendering. Prefer a system libGL; the
# Nsight fallback is useful on managed compute nodes where apt is unavailable.
if ! ldconfig -p 2>/dev/null | grep -q 'libGL.so.1'; then
  MESA_FALLBACK=/opt/nvidia/nsight-compute/2024.3.2/host/linux-desktop-glibc_2_11_3-x64/Mesa
  if [ -f "$MESA_FALLBACK/libGL.so.1" ]; then
    export LD_LIBRARY_PATH="$MESA_FALLBACK:${LD_LIBRARY_PATH:-}"
  else
    echo "Power Foam needs libGL.so.1 (install libgl1 or set LD_LIBRARY_PATH)." >&2
    exit 1
  fi
fi
PYTHONPATH="$ROOT/external/powerfoam" python - <<'PY'
import torch
import warp as wp
import powerfoam.scene
if not torch.cuda.is_available():
    raise RuntimeError("Power Foam bootstrap requires a visible CUDA GPU")
wp.init()
print("Power Foam import OK on", torch.cuda.get_device_name(0))
PY
