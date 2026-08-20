# ScanNet++ longer-training plan and execution state

Status: **preparing a Splatt3R-comparable rerun**. The earlier 80x80 arm A run
terminated without `final.pt`; its step-1,000 metrics remain diagnostic only.

## Current execution state (2026-08-19)

- Worker: `KW60995`, 3x RTX A6000 48 GB visible; full ScanNet++ root:
  `/data_ibex_c2324/data/scannetpp`. GPU 0 runs this job at about 5.1 GB.
- Published manifest: `data/manifests/scannetpp_scaleup_v1.json`.
  It contains 5,108 episodes from 852 training scenes and 149 episodes from
  50 validation scenes, with no scene overlap. Training bins are balanced:
  1,696 low-angle, 1,712 mid-angle, and 1,700 high-angle episodes. Four official
  training scenes were skipped: two contain no usable DSLR frames, and two use
  non-square pixels that this experiment's centered pinhole model rejects.
  The builder now applies the loader's own camera check and confirms that every
  selected view file exists, so a manifest cannot fail mid-run as the first
  launch did.
- Selected experiment: `configs/experiments/scannetpp_scaleup_arm_a.yaml`;
  arm A concatenates both 80x80 proposal lattices into one 12,800-cell Foam.
- The trainer is single-process/single-GPU. It has no DDP implementation; a
  claimed multi-GPU run would instead be multiple independent models. The
  primary 50,000-step run therefore uses one GPU and keeps the other devices
  available.
- Checkpoint source: the gated `vggt_omega_1b_512.pt` was placed on the shared
  `/data` mount and is copied into the checkout only after its size and MD5
  (`bc5302eada6222303c5e5f8d7dbce709`) match exactly.

Launcher `scripts/run_scannetpp_scaleup.sh` waits for that verified checkpoint,
copies it into the checkout, and then runs:

```bash
cd /code/feedforwardfoam-scaleup
source .venv-powerfoam/bin/activate
export LD_LIBRARY_PATH=/opt/nvidia/nsight-compute/2024.3.2/host/linux-desktop-glibc_2_11_3-x64/Mesa:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=0 python -m feedforwardfoam.train \
  --config configs/experiments/scannetpp_scaleup_arm_a.yaml \
  --data-root /data_ibex_c2324/data/scannetpp/data \
  --checkpoint checkpoints/vggt_omega_1b_512.pt
```

`--data-root` is the scene directory (`<root>/data`), not the dataset root;
split files live one level above it. Re-running the launcher resumes
automatically from `runs/scannetpp_scaleup_arm_a_seed17/latest.pt` when present.

## Observed run state

Job `d48642b1-23a4-4a17-947c-dbd022685dc0` (`wh_fffoam-scaleup-a3`) started at
17:18 on repo commit `07e3b25`, reached step 1,000 at 17:35, and later failed
without writing `final.pt`. It therefore did not complete the planned 50,000
steps and is not resumable from a verified terminal state.
During the observed checkpoints, `active_cells` stayed at 12,800, confirming
that arm A retained both 80x80 proposal lattices.

First scene-disjoint validation (step 1,000, 18 episodes, untrained-scene
targets):

| Bin | val PSNR | support PSNR | coverage | gauge bound hits |
|---|---:|---:|---:|---:|
| all | 17.91 | 18.21 | 0.989 | 0.333 |
| low_angle | 18.94 | 18.97 | 0.995 | 0.500 |
| mid_angle | 17.27 | 17.73 | 0.987 | 0.167 |
| high_angle | 17.69 | 18.01 | 0.985 | 0.333 |

`best_full.pt`, `best_support.pt`, `latest.pt`, per-bin validation renders, and
diagnostic renders are all being written.

**Depth-gauge clipping is the leading open risk.** One third of validation
episodes hit the `[0.25, 4]` clamp at step 1,000, and training logs raw scales
as high as 6.98 and as low as 0.21. Those episodes are mis-scaled before the
head sees them, so treat their metrics as suspect and fix the gauge before
reading this run as a statement about achievable quality.

## 1. Reduction decision

The fixed-triplet reduction study is complete. Arm A remains the measured
quality ceiling; fixed-budget world-space reduction and containment merging do
not justify delaying the scene-disjoint run.

## 2. Compute inventory (measured 2026-08-18)

| Worker | GPUs | Notes |
|---|---|---|
| `KW60996` | 4× RTX A6000 48 GB | idle after the fixed-triplet jobs; holds the only accessible checkpoint copy, but advertises no transferable data paths |
| `KW60995` | 3× RTX A6000 48 GB | full dataset mounted; previous run environment remains |
| `KW60898` | 1× RTX 6000 Ada 48 GB | shared multi-tenant box; full dataset is reachable through the IBEX mount |
| `gpu210-02` | 1× Tesla V100 32 GB | **no data paths bound** |
| `KW61627` | 2× RTX PRO 6000 Blackwell 96 GB | busy with unrelated DRRT work |
| `KW61633` | 1× RTX A2000 12 GB | too small |
| `archdome` | 1× RTX 3090 24 GB | offline at inspection |

There is **no multi-V100 machine with the full dataset**. The only V100 is
`gpu210-02` and it has nothing mounted.

## 3. Dataset location (verified)

`KW60995` mounts the official ScanNet++ tree at
`/data_ibex_c2324/data/scannetpp`. Its NVS split files contain 856 training,
50 validation, and 50 test scenes. The manifest builder loaded native
`dslr/nerfstudio/transforms_undistorted.json` metadata and native resized DSLR
images from this tree. The official test scenes remain excluded from training
and model selection.

## 4. Episode sampling decision

The manifest uses calibrated pose constraints only: target between contexts,
baseline ratio `[0.5, 2.5]`, perpendicular fraction at most `0.20`, and maximum
view-angle bins `[3°, 8°)`, `[8°, 16°)`, and `[16°, 25°)`. It balances those bins
within each scene and across the split.

Target-observable fraction remains depth-derived and is not mislabeled as a
pose metric. Computing it for every candidate before training would require
VGGT inference over the full dataset. Instead, fixed validation computes actual
projected support from decoded depth and reports full-frame and support metrics
for each pose bin. This preserves the measurement without leaking a costly,
uncalibrated pose proxy into candidate selection.

The older `q05…q90` labels remain selector-score quantiles, not overlap
quantiles; they are not used by this manifest.

## 5. Planned long-run shape

- Scene-disjoint official ScanNet++ training and validation splits.
- Pose-constrained episodes balanced across low-, mid-, and high-angle bins;
  projected support is measured, not approximated, during validation.
- Reduction arm: arm A (all-proposal concatenation, no reduction). Arms B–E are
  all worse at a reduced budget, and the two world-space follow-ups (D fps,
  E confidence voxel) were measured and rejected — see
  `docs/experiments/proposal_reduction_arms.md`. Containment merging (arm F) is
  quality-neutral and may be enabled as a structural guarantee. Revisit only if a pixel-space
  reduction arm beats arm B.
- Retain `best_full.pt` / `best_support.pt`; cosine LR.

## 6. Remaining interpretation safeguards

- Depth-gauge clipping remains material. Training now logs both the unclamped
  `depth_alignment_raw_scale` and `depth_alignment_bound_hit`; validation
  reports the bound-hit rate. Treat episodes that hit `[0.25, 4]` as suspect
  rather than silently interpreting them as valid geometry.
- Fixed per-bin validation, support metrics, and render bundles are implemented.
- The registers-only two-context control is still absent from the stratified
  table. It does not block this exploratory scale-up, but it remains required
  before attributing a scene-disjoint gain specifically to pixel-aligned fusion.

## 7. Deferred

- **Matched gsplat baseline.** Deferred by decision; rough reference numbers to
  be taken from published pixel-aligned Gaussian tables (pixelSplat, MVSplat)
  rather than a matched in-house run for now. A matched run is still required
  before any fairness claim is published.

## 8. Splatt3R-comparable protocol

The next run uses
`configs/experiments/scannetpp_splatt3r_256_arm_a.yaml`:

- 256x256 inputs, two contexts, and four independently sampled targets per
  optimizer step;
- effective scene batch 12 through gradient accumulation because the trainer
  remains single-process;
- masked MSE plus LPIPS 0.25, matching Splatt3R's published objective;
- exact official ScanNet++ train/validation splits and the four published
  close/medium/wide/very-wide evaluation tuple files;
- exhaustive fixed evaluation with full-frame and laser-supported PSNR, SSIM,
  and LPIPS reported per overlap bin;
- arm A's full two-view proposal concatenation. At 256x256 this is 131,072
  active Foam cells, not the earlier 12,800-cell 80x80 setting.

`scripts/build_splatt3r_scannetpp_manifest.py` constructs the training split and
maps the published integer tuple indices onto ScanNet++ frame names.
`python -m feedforwardfoam.train --evaluate-checkpoint ...` runs the exhaustive
fixed evaluation independently of training.

### Required preprocessing

The mounted ScanNet++ tree does not contain laser depth PNGs. Splatt3R renders
them from each scene's `scans/mesh_aligned_0.05.ply`; silently substituting VGGT
depth would invalidate the matched mask protocol. Before the run,
`scripts/render_scannetpp_depths.py` must render resumable uint16 millimeter
depths beside `resized_undistorted_images`:

```bash
python scripts/render_scannetpp_depths.py \
  --data-root /data_ibex_c2324/data/scannetpp/data \
  --scene-list /data_ibex_c2324/data/scannetpp/splits/nvs_sem_train.txt
python scripts/render_scannetpp_depths.py \
  --data-root /data_ibex_c2324/data/scannetpp/data \
  --scene-list /data_ibex_c2324/data/scannetpp/splits/nvs_sem_val.txt
```

The rendering pass can be distributed over workers sharing the dataset by
assigning disjoint `--shard-index 0..N-1 --num-shards N` values. All shards
write the same resumable output tree. After the shards finish,
`scripts/run_scannetpp_splatt3r.sh` verifies/skips every completed depth,
rebuilds the manifest, verifies the gated checkpoint byte-for-byte, and resumes
or starts training.

### Deployment state

No A100 worker is registered. The available dedicated nodes are single 32 GB
V100s with no advertised transferable data paths; `KW60995` has three idle
48 GB A6000s and the full dataset mount. Direct inspection found the raw images,
camera metadata, and meshes there, but no rendered depth directory.

No job has been launched yet. Worker Harness `exec` fails immediately on both
`KW60995` and a V100 even after worker-container restarts, and Pi delegation
returns HTTP 500 before creating a session. Starting training without the depth
pass would silently change the requested Splatt3R mask protocol; it is therefore
intentionally blocked rather than replaced with predicted-depth masks.

The renderer uses Nerfstudio OpenGL camera-to-world poses directly, scales
intrinsics to the resized image dimensions, applies the published anonymous
pixel mask convention, rejects empty renders, writes atomically, and skips
completed files on restart. A synthetic EGL smoke rendered a plane at 2.0 m to
an exact 2,000 mm uint16 depth and verified that `ScanNetPPDataset` loads it as
2.0 m.
