#!/usr/bin/env bash
# Install the two upstream projects in isolated environments.  Their NumPy pins
# conflict, so do not install both into the project development environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"
git submodule update --init --recursive

uv venv --python 3.11 "$ROOT/.venv-vggt-omega"
source "$ROOT/.venv-vggt-omega/bin/activate"
uv pip install torch torchvision
uv pip install -r "$ROOT/external/vggt-omega/requirements.txt"
uv pip install -e "$ROOT/external/vggt-omega"
python -c 'import vggt_omega; print("VGGT-Omega import OK")'
deactivate

"$ROOT/scripts/bootstrap_powerfoam_env.sh"
