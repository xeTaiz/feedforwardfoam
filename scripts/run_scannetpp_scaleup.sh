#!/usr/bin/env bash
# Wait for the gated VGGT-Omega checkpoint to materialize on the network share,
# verify it byte-exactly, then run the ScanNet++ arm A scale-up training.
set -uo pipefail

ROOT=/code/feedforwardfoam-scaleup
SHARE=/data
EXPECT_MD5=bc5302eada6222303c5e5f8d7dbce709
EXPECT_SIZE=4576706117
DEST="$ROOT/checkpoints/vggt_omega_1b_512.pt"
MESA=/opt/nvidia/nsight-compute/2024.3.2/host/linux-desktop-glibc_2_11_3-x64/Mesa

mkdir -p "$ROOT/checkpoints"

verified() {
  test -f "$1" || return 1
  test "$(stat -c %s "$1" 2>/dev/null || echo 0)" = "$EXPECT_SIZE" || return 1
  test "$(md5sum "$1" 2>/dev/null | cut -d' ' -f1)" = "$EXPECT_MD5"
}

fetch() {
  local source
  for source in "$SHARE/vggt_omega_1b_512.pt" "$SHARE"/vggt_omega_1b_512.pt.*.partial; do
    test -f "$source" || continue
    test "$(stat -c %s "$source" 2>/dev/null || echo 0)" = "$EXPECT_SIZE" || continue
    cp "$source" "$DEST.tmp" 2>/dev/null || { rm -f "$DEST.tmp"; continue; }
    if verified "$DEST.tmp"; then
      mv "$DEST.tmp" "$DEST"
      echo "CHECKPOINT_SOURCE=$source"
      return 0
    fi
    rm -f "$DEST.tmp"
  done
  return 1
}

if ! verified "$DEST"; then
  echo WAITING_FOR_CHECKPOINT
  until fetch; do sleep 120; done
fi
echo CHECKPOINT_READY

cd "$ROOT" || exit 1
echo "HEAD=$(git rev-parse --short HEAD)"
# shellcheck disable=SC1091
source .venv-powerfoam/bin/activate
export LD_LIBRARY_PATH="$MESA:${LD_LIBRARY_PATH:-}"

resume=()
if test -f runs/scannetpp_scaleup_arm_a_seed17/latest.pt; then
  resume=(--resume runs/scannetpp_scaleup_arm_a_seed17/latest.pt)
fi

CUDA_VISIBLE_DEVICES=0 python -m feedforwardfoam.train \
  --config configs/experiments/scannetpp_scaleup_arm_a.yaml \
  --data-root /data_ibex_c2324/data/scannetpp/data \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  "${resume[@]}"
echo "TRAIN_EXIT=$?"
