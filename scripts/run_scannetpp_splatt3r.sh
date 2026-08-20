#!/usr/bin/env bash
# Complete the exact Splatt3R preprocessing protocol, then launch arm A training.
set -euo pipefail

ROOT=${FFFOAM_ROOT:-/code/feedforwardfoam-scaleup}
DATASET_ROOT=${SCANNETPP_ROOT:-/data_ibex_c2324/data/scannetpp}
CHECKPOINT=${VGGT_OMEGA_CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}
EXPECT_MD5=bc5302eada6222303c5e5f8d7dbce709
EXPECT_SIZE=4576706117
MESA=/opt/nvidia/nsight-compute/2024.3.2/host/linux-desktop-glibc_2_11_3-x64/Mesa

cd "$ROOT"
# shellcheck disable=SC1091
source .venv-powerfoam/bin/activate
export LD_LIBRARY_PATH="$MESA:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -c 'import huggingface_hub, pyrender, trimesh'
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="brandonsmart/splatt3r_v1.0", allow_patterns="scannetpp/coverage/*.json", local_dir="data/splatt3r")'

for split in train val; do
  python scripts/render_scannetpp_depths.py \
    --data-root "$DATASET_ROOT/data" \
    --scene-list "$DATASET_ROOT/splits/nvs_sem_${split}.txt"
done

python scripts/build_splatt3r_scannetpp_manifest.py \
  --scene-root "$DATASET_ROOT/data" \
  --split-root "$DATASET_ROOT/splits" \
  --coverage-root data/splatt3r/scannetpp/coverage \
  --test-assets data/manifests \
  --evaluation-stride 100 \
  --output data/manifests/scannetpp_splatt3r_v1.json

test -f "$CHECKPOINT"
test "$(stat -c %s "$CHECKPOINT")" = "$EXPECT_SIZE"
test "$(md5sum "$CHECKPOINT" | cut -d' ' -f1)" = "$EXPECT_MD5"

resume=()
if test -f runs/scannetpp_splatt3r_256_arm_a_seed17/latest.pt; then
  resume=(--resume runs/scannetpp_splatt3r_256_arm_a_seed17/latest.pt)
fi
python -m feedforwardfoam.train \
  --config configs/experiments/scannetpp_splatt3r_256_arm_a.yaml \
  --data-root "$DATASET_ROOT/data" \
  --checkpoint "$CHECKPOINT" \
  "${resume[@]}"
