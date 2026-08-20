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
GPU_IDS=${FFFOAM_RENDER_GPUS:-0,1,2}

python -c 'import huggingface_hub, pyrender, trimesh'
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="brandonsmart/splatt3r_v1.0", allow_patterns="scannetpp/coverage/*.json", local_dir="data/splatt3r")'

python scripts/build_splatt3r_scannetpp_manifest.py \
  --scene-root "$DATASET_ROOT/data" \
  --split-root "$DATASET_ROOT/splits" \
  --coverage-root data/splatt3r/scannetpp/coverage \
  --test-assets data/manifests \
  --evaluation-stride 100 \
  --output data/manifests/scannetpp_splatt3r_v1.json

scene_list=$(mktemp)
trap 'rm -f "$scene_list"' EXIT
python - "$scene_list" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path("data/manifests/scannetpp_splatt3r_v1.json").read_text())
scene_ids = set(manifest["train"])
scene_ids.update(entry["scene_id"] for entry in manifest["val"])
Path(sys.argv[1]).write_text("".join(f"{scene_id}\n" for scene_id in sorted(scene_ids)))
PY

IFS=, read -r -a render_gpus <<<"$GPU_IDS"
if ((${#render_gpus[@]} == 0)); then
  echo "FFFOAM_RENDER_GPUS must name at least one GPU" >&2
  exit 1
fi
render_pids=()
for shard_index in "${!render_gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${render_gpus[$shard_index]} PYOPENGL_PLATFORM=egl \
    python scripts/render_scannetpp_depths.py \
      --data-root "$DATASET_ROOT/data" \
      --scene-list "$scene_list" \
      --num-shards "${#render_gpus[@]}" \
      --shard-index "$shard_index" &
  render_pids+=("$!")
done
for pid in "${render_pids[@]}"; do
  wait "$pid"
done

test -f "$CHECKPOINT"
test "$(stat -c %s "$CHECKPOINT")" = "$EXPECT_SIZE"
test "$(md5sum "$CHECKPOINT" | cut -d' ' -f1)" = "$EXPECT_MD5"
export CUDA_VISIBLE_DEVICES=${FFFOAM_TRAIN_GPU:-0}

resume=()
if test -f runs/scannetpp_splatt3r_256_arm_a_seed17/latest.pt; then
  resume=(--resume runs/scannetpp_splatt3r_256_arm_a_seed17/latest.pt)
fi
python -m feedforwardfoam.train \
  --config configs/experiments/scannetpp_splatt3r_256_arm_a.yaml \
  --data-root "$DATASET_ROOT/data" \
  --checkpoint "$CHECKPOINT" \
  "${resume[@]}"
