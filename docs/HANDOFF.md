# Feed-Forward Power Foam — Agent Handoff

**Date written:** 2026-08-16 (next ~12 h of unattended work queued)
**Repository:** `git@github.com:xeTaiz/feedforwardfoam.git`
**Latest commit:** `4f1f3c0` (docs); `2cf3fab` (code) — `2cf3fab..4f1f3c0` adds doc only
**Branch:** `main` (pushed, clean working tree)

## 1. Project goal

Build and validate a feed-forward Power Foam reconstruction system using frozen VGGT-Ω features. Compare it fairly against a matched canonical-view gsplat Gaussian baseline. Establish reliable self-view and held-out NVS behavior. Use one canonical foam, predict from frozen VGGT-Ω, supervise with held-out ScanNet++ views.

## 2. Hard constraints / invariants

These are **non-negotiable** (see `specs/FF-POWER-FOAM-SPEC-v0.md`):

1. **One global canonical Power Foam per scene.** Multiple views contribute evidence, but **no merging of independently constructed power diagrams**.
2. **Frozen VGGT-Ω as backbone** (initial implementation). Submodule: `external/vggt-omega @ 39a0cb8`.
3. **Full unmodified Power Foam renderer.** Submodule: `external/powerfoam @ 9639225`. Treat the renderer as the ground truth for any cell output.
4. **Bounded physical initialization.** Outputs are bounded residuals around physical depth, normals, footprint radii, and source RGB.
5. **Render into every selected target; average losses.** No per-target backups; no per-target gradient stop.
6. **No leakage between context and held-out target.**
7. **Self-view quality must never be presented as NVS evidence.**
8. **Deterministic scene-disjoint validation; resumable checkpoints.**
9. **Matched Foam/Gaussian budgets** for fairness.
11. **Coordinate system: foam is world-space, anchored to context view 0.** Points, radii, quaternions, and spherical-Voronoi (SV) axes are world-space; texel sites/heights are local surface coordinates; SV RGB is centered (upstream adds +0.5). See Section 7.

## 3. Environment & infrastructure

### Local
- Python venv at `./.venv` (project). Tests: `pytest -q` → **45 passed, 2 skipped** (CUDA-gated).
- Repo layout:
  - `src/feedforwardfoam/` — package (backbone, head, renderer, fusion, train, data/*, gaussian)
  - `configs/experiments/` — experiment configs
  - `external/vggt-omega`, `external/powerfoam` — pinned submodules
  - `scripts/` — bootstrap, visualization, optimization
  - `tests/` — unit + CUDA integration
  - `docs/` — experiments, references, handoff

### Worker (CUDA)
- **Worker:** `KW60996` (SSH `kw60996.hs.d0me.xyz`, user `engeld`)
- 4× NVIDIA RTX A6000 (48 GB each), 32 cores, 1 TB RAM
- Worker project: `/code/feedforwardfoam-project`
- WGGT checkpoint: `/code/feedforwardfoam-project/checkpoints/vggt_omega_1b_512.pt` (SHA-256 `c02da418…0796934`)
- ScanNet++ data: `/code/feedforwardfoam-project/data/staged/scannetpp_p0` (743 MB, 2,558 files, 4 scenes: `00a231a370`, `f9397af4cb`, `fd361ab85f`, `ff17657f71`)
- Power Foam venv: `source /code/feedforwardfoam-upstream-smoke/venv-powerfoam/bin/activate`
- CUDA env:
  ```
  LD_LIBRARY_PATH=/.singularity.d/libs:/opt/nvidia/nsight-compute/2024.3.2/host/linux-desktop-glibc_2_11_3-x64/Mesa:${LD_LIBRARY_PATH:-}
  CUDA_VISIBLE_DEVICES=<0..3>
  PYTHONUNBUFFERED=1
  ```

### Worker harness (this environment)
- `wh_read({action:"list_workers"})` — workers
- `wh_read({action:"list_jobs", worker_id, status})` — jobs
- `wh_read({action:"get_job_logs", job_id, tail|head|follow})` — log inspection
- `wh_dispatch({action:"exec", worker_id, command, sync, sync_timeout})` — launch shell
- `wh_dispatch({action:"stop_job", job_id})` — stop

For backgrounded long jobs use `sync:false` (returns job_id). `tail:1` returns last log line.

## 4. Codebase key files

| File | Role |
|---|---|
| `src/feedforwardfoam/head.py` | `CanonicalPowerFoamHead`, `FoamParameters`, helpers: `concatenate_foam_parameters`, `select_foam_parameters`, `voxel_budget_indices`. Three prediction modes: `residual`, `absolute` (position-only init), `initialization`. Residual flags: `enable_point_residual`, `enable_radius_residual`, `enable_orientation_residual`, `enable_rgb_residual`. Proposal flags: `proposal_views` (`canonical`/`all`), `proposal_reduction` (`none`/`all`/`balanced`/`voxel`), `selection_mode` (`gate`/`uniform`). |
| `src/feedforwardfoam/train.py` | Training loop. Scheduler: `constant` or `cosine`. Best full/support checkpoints: `best_full.pt`/`best_support.pt`. Support metrics: `report_support_metrics`, `support_mask_contexts` (`canonical`/`all`). |
| `src/feedforwardfoam/fusion.py` | Depth gauge alignment (predicted ↔ calibrated baselines), world unprojection, projected context support mask, canonical support build. Depth normalized to forward-z; gauge scale bounded `[0.25, 4]`. |
| `src/feedforwardfoam/renderer.py` | `PowerFoamRendererBridge` (deliberately non-`nn.Parameter`-wrapped for gradient flow to head). `camera_from_view`, `pinhole_ray_map_from_view`, `powerfoam_args`. |
| `src/feedforwardfoam/backbone.py` | `FrozenVGGTOmega` — exposes `depth`, `depth_conf`, `registers`, `patch_tokens`, `predicted_extrinsics`, `predicted_intrinsics`. `register_dim = 2 × camera_token_dim` (2048). Also `FrozenGeometryStub` for tests. |
| `src/feedforwardfoam/data/scannetpp.py` | Native ScanNet++ DSLR loader. Pinhole-only, centered cx/cy, square pixels. Center-crops 1752×1168 → 1168² and resizes. `episode_from_names` for fixed-triplet experiments. |
| `src/feedforwardfoam/data/multiscene.py` | Scene-disjoint splits, stochastic + deterministic fixed episodes. |
| `scripts/run_stratified_triplet_matrix.py` | Sequential sharded launcher: `--manifest`, `--base-config`, `--data-root`, `--checkpoint`, `--output-root`, `--shard-index/--shard-count`, `--steps`, `--modes` (`initialization,full,appearance`). Skips completed runs (`. |
- `scripts/optimize_triplet_foam.py` | Direct nn.Parameter optimization of all eight foam fields (per-triplet oracle). Cosine schedule, support mask, best-state save. |
- `scripts/select_scannetpp_triplet.py` | Ranks geometric triplets by interpolation, perpendicular fraction, baseline, view angles; emits `candidates.json` + `contact_sheet.png`. |
- `scripts/summarize_triplet_overfits.py` | Aggregates runs into CSV + Markdown (used for fixed-triplet reports). |

## 5. Experiment timeline & results

### 5.1 Renderer & same-view sanity (`docs/experiments/p0_overfit_progress.md`)
- **Renderer identity:** 154.92 dB, MSE 3.22e-16.
- **Same-view residual head (Lego):** capped 100 dB.
- **VGGT-geometry head H8:** 32.91 dB (conditioning test, not NVS evidence).

### 5.2 First multi-target ScanNet++ pilot
- Sup 1→8 targets per update: 5.50 / **6.18** / 5.88 / 5.62 dB validation.
- Source-render 46.39 dB; cross-view 5.27 dB → **source copy without transferable geometry**.

### 5.3 Two-context multi-scene training (4×2000 steps, 4 scenes)
| Arm | Best / Final val PSNR |
|---|---|
| One-context | 9.03 / 9.02 dB |
| Two-context registers | **9.42 / 9.42 dB** |
| Two-context projected | 9.40 / 9.23 dB |
| Projected + mask | 9.36 / 9.33 dB |

Two-context gives a small real gain; explicit projection adapter did not beat global registers. Absolute quality remains poor.

### 5.4 Fixed ideal-parallax triplet `f9397af4cb` (contexts `DSC04956,04970`, target `04962`)
| Model | Best target PSNR |
|---|---|
| One-context | 32.02 dB |
| Two-context registers | 29.71 dB |
| Two-context projected | **31.27 dB** |

Interpolation 0.500, perp fraction 0.0064, max angle 3.16°. **Overfit capacity, not generalization.**

### 5.5 Random-triplet overlap study (4 triplets)
Observability matters strongly. random `00a`: 13.93 dB full-frame vs **29.79 dB** on observable pixels. Provisional Pearson r vs support PSNR: observable fraction **+0.95**, max angle **−0.82**.

### 5.6 Initialization/absolute ablations on ideal triplet
| Experiment | Best PSNR |
|---|---|
| Exact initialization (no head) | 23.61 dB |
| Residual full, cosine LR | 29.90 dB |
| Residual observable, cosine LR | 30.95 dB |
| **Absolute (position-only init)** | **30.76 dB** |
| Appearance-only residual | 30.71 dB |
| Radius + appearance residual | 31.05 dB |
| Direct foam optimization, constant LR | 30.18 dB best / 27.33 dB final |
| Direct foam optimization, cosine LR | 31.07 dB best / 27.85 dB final |

**Decoding SV axes/RGB is here called "appearance" (not "RGB-only"); SV axes are still predicted, just point+normal+radius residuals disabled.** Direct optimization has comparable capacity but degrades later — topology/Adam instability, not yet a clean upper bound.

### 5.7 Stratified 12-triplet overlap study — **NEW (3,000 steps each)**
Manifest: `data/manifests/scannetpp_stratified_triplets_v1.json` (4 selector-score quantiles × 3 scenes). Outputs: `/code/feedforwardfoam-project/runs/stratified_triplets_v1`. Modes: `initialization`, `full` (all residual flags), `appearance` (only RGB/axes residual on top of fixed positions).

Full-head vs appearance-only head **support PSNR** differences (full minus appearance):
| Triplet | obs | init | appearance | full |
|---|---:|---:|---:|---:|
| 00a_q05 | 0.88 | 15.16 | 15.95 | **18.47** |
| 00a_q30 | 1.00 | 17.59 | 19.33 | **32.88** |
| 00a_q60 | 0.86 | 13.83 | 14.60 | **32.68** |
| 00a_q90 | 0.90 | 15.77 | 16.00 | **17.15** |
| f939_q05 | 0.99 | 18.85 | 19.93 | **21.13** |
| f939_q30 | 0.95 | 18.87 | 19.53 | **21.96** |
| f939_q60 | 1.00 | 20.84 | 22.16 | **24.52** |
| f939_q90 | 0.80 | 15.93 | 17.83 | **22.59** |
| fd_q05 | 0.85 | 16.91 | 17.20 | **23.13** |
| fd_q30 | 1.00 | 19.47 | 25.92 | **27.50** |
| fd_q60 | 0.86 | 17.06 | 17.40 | **20.86** |
| fd_q90 | 0.93 | 15.66 | 17.46 | **19.26** |

**Full head beats appearance head on every single triplet (support PSNR).** The geometry branch is doing real work, not just memorizing appearance. Differences range from +1 to +18 dB, with the largest gains on harder overlap bins.

### 5.8 Two-target geometry test — **NEW**
One ideal context pair supervises two distinct targets (`04962` at t=0.500, `04965` at t=0.734):
| Mode | best PSNR |
|---|---:|
| initialization | 22.50 dB |
| appearance | 27.03 dB |
| full | **29.13 dB** |

**Full > appearance > init.** Confirms geometry branch helps when there are two distinct targets — not only overfitting one camera.

### 5.9 All-view pixel-aligned proposal study (A/B/C) — **NEW**
Manifest: same 12 triplets. Outputs: `/code/feedforwardfoam-abc/runs/proposals_{a_all,b_balanced,c_voxel}_v1`. Each arm has 12 `full` + 12 `initialization` runs.

| Arm | Strategy | Cells | budget policy |
|---|---|---:|---|
| A | All | 12,800 | All per-context proposals rendered |
| B | Balanced | 6,400 | 3,200 uniform from each context |
| C | Voxel | 6,400 | All 12,800 then world-space voxel selection (deterministic bisection) |

**Best full-frame PSNR (per arm):**
| Triplet | obs | A (12,800) | B (6,400) | C (6,400) |
|---|---:|---:|---:|---:|
| 00a_q05 | 0.90 | 12.26 | 11.90 | 11.86 |
| 00a_q30 | 1.00 | **36.36** | 25.31 | 20.33 |
| 00a_q60 | 0.97 | **19.24** | 17.86 | 12.97 |
| 00a_q90 | 0.92 | **12.32** | 11.75 | 11.78 |
| f939_q05 | 1.00 | **26.52** | 22.27 | 21.57 |
| f939_q30 | 1.00 | **30.88** | 29.21 | 19.61 |
| f939_q60 | 1.00 | **26.31** | 24.57 | 21.97 |
| f939_q90 | 0.99 | **23.65** | 20.00 | 12.72 |
| fd_q05 | 1.00 | **31.42** | 28.09 | 22.28 |
| fd_q30 | 1.00 | **33.63** | 26.87 | 25.27 |
| fd_q60 | 1.00 | **28.78** | 27.09 | 18.83 |
| fd_q90 | 0.96 | **14.78** | 14.57 | 13.43 |

**Comparisons (mean over 12 triplets):**
- A − view0-only: **+7.54 dB** (A is 12,800 cells; view0-only is 6,400 from canonical — but A is also using both contexts)
- B − view0-only (matched 6,400 budget): **+4.48 dB** mean; wins 9/12
- A − B: **+3.06 dB** (extra primitive budget helps)
- B − C: **+3.91 dB** (uniform balanced beats voxel selection by large margin — voxel selection loses depth edges)
- A − C: **+6.96 dB**

**Conclusions:**
- All-view pixel-aligned proposals substantially beat view-0-only at matched budget (B vs stratified, ~+4.5 dB mean).
- Even more capacity is unlocked by rendering all proposals (A, ~+7.5 dB).
- **Voxel selection (C) is harmful vs balanced uniform.** Likely because one representative per voxel loses surface/density structure and depth edges. Needs a different reduction (e.g. confidence-then-voxel-pool, FPS, or soft assignment) to be competitive.
- Canonical-view choice still matters: B beats view0-only by +4.5 dB mean but loses on `00a_q30` (−7.57 dB) where the chosen canonical context happened to be well-positioned for view-0-only decoding.

## 6. Key architectural findings

### 6.1 Coordinate system (audited, see Section 7)
Foam is **world-space**, anchored to context view 0:
- Points, radii, quaternions, SV axes: world-space
- Texel sites/heights: local surface coordinates (upstream multiplies by radius and rotates by quaternion)
- SV RGB: centered; upstream adds +0.5
- Density: applied upstream as `softplus(..., beta=100)`

### 6.2 View-0-only limitation (now removed in A/B/C arms)
Before A/B/C, the head emitted **all cells anchored to context view 0**. View 1 contributed projected features but **no additional cells** in view-2-only regions. This is the architectural reason previous studies showed the head "copies" its source but fails cross-view.

A/B/C fix this by treating both contexts as proposal sources. A `select_foam_parameters` + `concatenate_foam_parameters` + `voxel_budget_indices` API was added to the head for budgeted world-space selection.

### 6.3 Depth gauge
VGGT-Ω predicts cameras and depth in an up-to-scale gauge. For 2+ contexts, the ratio of predicted vs calibrated camera-center baselines supplies a robust scale bounded to `[0.25, 4]`. With one context, depth is retained as-is.

### 6.4 Renderer contract
- `PowerFoamRendererBridge` deliberately avoids `nn.Parameter` wrapping to keep gradients flowing to the head.
- Čech adjacency is rebuilt from detached geometry each render; gradients still flow through positions, radii, quaternions, texels, density, and SV radiance.

### 6.5 Renderer double-softplus subtlety
Upstream applies `F.softplus(..., beta=100)` to raw `scene.density` and `scene.radii`. `inverse_softplus` is provided for converting physical values to the raw domain. `fixed_density=100` in our configs is effective density 100, not raw 100.

## 7. Literature: how pixel-aligned Gaussian methods merge proposals

The user's intuition that "they just render all of them" is correct for the released methods we audited:
- **pixelSplat (CVPR 2024, arXiv 2312.12337)** — 2 calibrated images, ~1 Gaussian per pixel per view, unproject into world, **flatten+render all**. Soft suppression comes from depth-distribution opacity, not hard pruning. No explicit deduplication or budget. Repo: `github.com/dcharatan/pixelsplat`.
- **MVSplat (ECCV 2024, arXiv 2403.14627)** — improves cross-view depth via cost volume, then **still flattens+renders all**.
- **Splatt3R** — similar dense per-pixel aligned maps combined; masks for invalid pixels.

**None of them do geometric deduplication or voxel merging.** That part of our proposal fusion (arm C) is novel for Foam and we now have evidence it currently *underperforms* simple balanced concatenation. Likely needs confidence-weighted voxel pooling, FPS, or learned selection to be useful.

For Foam we must additionally build **one Čech graph** over the combined sites — we cannot merge two independently built foam topologies. The head's new helpers enforce this: proposals are concatenated at the parameter level, then selection happens, then one `PowerFoamRendererBridge.build()` call constructs a single upstream `PowerfoamScene`.

## 8. Discussion decisions (user pushback that changed the plan)

These were explicit user corrections during our conversation:

1. **Empty-mask supervision was deprioritized for now.** A mask that excludes pixels with no cell traversal produces **no gradient** for those pixels (rendered value is locally constant → ∂I/∂θ = 0). Full-frame MSE loss has gradient magnitude proportional to *supported* fraction, not 1. The main reason for masks is **normalization + interpretable validation metrics**, not better gradients. Keep full-frame loss as default, only add masking on a few low/medium/high-overlap episodes if needed.

2. **Direct foam optimization is not a research priority.** We have demonstrated ~31 dB transient capacity with a cosine schedule; later degradation is topology/Adam instability, not architecture. Don't spend time tuning hyperparameters. Just retain best state.

3. **Stratified sampling study is in progress and should drive scaled-training sampling.** Use 12–20 fixed episodes stratified by target observability, angle, normalized baseline, target interpolation, and scene type. Use the result to define sampling bins for scaled training (reject near-empty, balance easy/medium/difficult).

4. **Geometry-vs-appearance control is small but important.** Concluded via the two-target test (Section 5.8) — full beats appearance by +2.1 dB when two distinct targets are supervised. The user was right to push for this even though the appearance-only arm looked competitive on the single ideal triplet.

5. **Stay close to proven pixel-aligned decoding for the proposal-fusion step.** Avoid simultaneously changing representation and decoder. Use pixel-aligned proposals from both contexts + budget selection; defer free learned 3-D queries to later.

## 9. Currently running / queued

**Everything is done at handoff time** (A/B/C completed within the queued window):

- All 36 stratified runs: ✓ complete (`runs/stratified_triplets_v1`)
- All 3 two-target runs: ✓ complete (`runs/two_target_geometry_v1`)
- All 36 A/B/C proposal runs: ✓ complete (`runs/proposals_{a_all,b_balanced,c_voxel}_v1` under `/code/feedforwardfoam-abc`)

**Nothing is actively training when this handoff was written.**

## 10. Agreed next steps (priority order from discussion)

The user agreed to 1–4 done; 4–7 are the next phase:

1. ~~Stratified fixed-triplet overlap study~~ ✓ (Section 5.7)
2. ~~Two-target geometry test~~ ✓ (Section 5.8)
3. ~~All-view pixel-aligned proposal ablation (A/B/C)~~ ✓ (Section 5.9)
4. **Implement two-view pixel-aligned proposal concatenation** (A and B are done; their configs exist; further arms can extend C with FPS or confidence-weighted voxel pooling as a follow-up).
5. **Neighbor-derived radius + learned scale** (see `docs/references/decoder_design_overview.md`). `r_j = r_j⁰ · exp(λ tanh s_j)`, where `r_j⁰ = η · median_KNN_j ||p_j - p_k||`. Inputs: KNN distance, support count, voxel occupancy, covariance eigenvalues.
6. **Scale-up readiness checks** before larger scene-disjoint training:
   - Depth-gauge scale log (hits-bounds count)
   - Fixed validation bins by overlap/angle/coverage
   - best_full.pt / best_support.pt (already implemented)
   - Visualization bundles for best checkpoints
   - Matched Gaussian baseline at same primitive/render/wall-clock budget
7. **Larger scene-disjoint training** with overlap-aware sampling.

**Decision gate before scale-up:** the fused Foam head must consistently beat (a) exact initialization, (b) appearance-only residual, (c) registers-only two-context control, on observable target regions across the stratified triplets — and proposal fusion must demonstrate improved coverage rather than merely improved appearance.

## 11. Open questions / caveats

- **Voxel selection (arm C) is currently harmful.** Probably needs confidence-weighted voxel pooling or FPS to be useful. This is a clear research direction.
- **Canonical view choice matters.** Arm B (balanced) loses to view-0-only on `00a_q30` by −7.57 dB. Future: try averaging predictions across multiple canonical choices, or learn the canonical view.
- **Constant LR 5e-4 did not stably converge all runs.** Some peaks then degrade. Cosine schedule helps; best-checkpoint retention is in place (`best_full.pt`/`best_support.pt`).
- **Texel sites/heights are still zero in all configs.** Surface detail is not represented; deferred.
- **Density is fixed at 10,000** for these ablations. Decoupling density from radius is future work.
- **No Gaussian (gsplat) baseline run yet** at matched primitive budget on this matrix. Important fairness check before scale-up.
- **Direct foam optimization degrades after best** at ~525 steps even with cosine LR. Likely topology/Adam instability, not searched further per Section 8.2.

## 12. How to inspect / reproduce

### Inspect latest results
```bash
# Local: summarize fixed-triplet overfit matrix
python scripts/summarize_triplet_overfits.py \
  runs/stratified_triplets_v1/00a_q05/full \
  runs/proposals_a_all_v1/00a_q05/full \
  runs/proposals_b_balanced_v1/00a_q05/full \
  --output-markdown runs/comparison.md
```

### Run a single fresh overfit
```bash
python -m feedforwardfoam.train \
  --config configs/experiments/overfit_scannetpp_triplet_2ctx_fused.yaml \
  --data-root data/staged/scannetpp_p0 \
  --checkpoint checkpoints/vggt_omega_1b_512.pt
```

### Run the A/B/C matrix on the worker
```bash
# Worker checkout already exists at /code/feedforwardfoam-abc
ssh kw60996.hs.d0me.xyz
cd /code/feedforwardfoam-abc
git fetch origin && git reset --hard origin/main
# re-link submodules if needed:
ln -sf /code/feedforwardfoam-project/external/vggt-omega external/vggt-omega
ln -sf /code/feedforwardfoam-project/external/powerfoam external/powerfoam
source /code/feedforwardfoam-upstream-smoke/venv-powerfoam/bin/activate
export PYTHONPATH=/code/feedforwardfoam-abc/src \
       LD_LIBRARY_PATH=/.singularity.d/libs:/opt/nvidia/nsight-compute/2024.3.2/host/linux-desktop-glibc_2_11_3-x64/Mesa:${LD_LIBRARY_PATH:-} \
       PYTHONUNBUFFERED=1
# Example: arm C, shard 0
CUDA_VISIBLE_DEVICES=0 python scripts/run_stratified_triplet_matrix.py \
  --manifest data/manifests/scannetpp_stratified_triplets_v1.json \
  --base-config configs/experiments/multiview_proposals_c_voxel.yaml \
  --data-root /code/feedforwardfoam-project/data/staged/scannetpp_p0 \
  --checkpoint /code/feedforwardfoam-project/checkpoints/vggt_omega_1b_512.pt \
  --output-root runs/proposals_c_voxel_v1 \
  --shard-index 0 --shard-count 1 \
  --steps 3000 --modes initialization,full
```

### Direct foam oracle on a fixed triplet
```bash
python scripts/optimize_triplet_foam.py \
  --config configs/experiments/overfit_scannetpp_triplet_2ctx_fused.yaml \
  --data-root data/staged/scannetpp_p0 \
  --vggt-checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir runs/direct_oracle \
  --steps 2000 --learning-rate 0.001 --learning-rate-schedule cosine \
  --visibility-mask
```

### Validate geometry of a candidate triplet
```bash
python scripts/select_scannetpp_triplet.py \
  --scene-root data/staged/scannetpp_p0/f9397af4cb \
  --output-dir runs/triplet_selection/f9397af4cb \
  --image-resolution 224 --top-k 12
```

## 13. Critical implementation gotchas (hit during this conversation)

- **Power Foam venv must be active** before any `feedforwardfoam.train` invocation. Bootstrap: `scripts/bootstrap_powerfoam_env.sh`.
- **VGGT-Ω submodule path injection:** `FrozenVGGTOmega.__init__` does `sys.path.insert(0, external/vggt-omega)`; requires the submodule to exist (or be symlinked from the primary project).
- **`register_dim = 2048`** (2 × aggregator camera_token_dim). Older configs used `1024` — the head will accept the config but the upstream projector would error; current code handles it correctly.
- **Native ScanNet++ loader** rejects off-center principal points or non-square pixels. Center-crops 1752×1168 → 1168² before resize. Only `PINHOLE` cameras.
- **`episode_from_names`** expects scene-relative image paths including the `dslr/resized_undistorted_images/` prefix.
- **Mask empty fallback:** an all-zero support mask falls back to full-frame loss rather than dropping the step.
- **`select_foam_parameters`** expects a CPU or GPU `torch.Tensor` of indices. The proposal-selection cache is keyed by `(reduction, scene_id, *context_names, H, W, budget)` so it's deterministic per fixed episode and cannot be shared between reduction arms.
- **Worker checkout `/code/feedforwardfoam-abc`**: a separate clone used during the queued runs to avoid touching `/code/feedforwardfoam-project`. Symlinks the two external submodule directories so packages resolve. **`git reset --hard` destroys those symlinks** and replaces them with empty submodule directories, producing `ModuleNotFoundError: No module named 'vggt_omega'`. Re-link after every reset: `cd external && rmdir vggt-omega powerfoam && ln -sfn /code/feedforwardfoam-project/external/vggt-omega vggt-omega && ln -sfn /code/feedforwardfoam-project/external/powerfoam powerfoam`.

## 14. Documentation index

Concise (start here):
- `docs/experiments/overview.md` — one-page experiment summary
- `docs/experiments/scannetpp_fixed_triplet_overfit.md` — fixed-triplet overfit + observable-region matrix
- `docs/experiments/foam_initialization_and_ablation.md` — initialization/absolute/residual ablations
- `docs/experiments/scannetpp_two_context_progress.md` — stability gate + multi-scene two-context
- `docs/experiments/scannetpp_multiview_progress.md` — 1/2/4/8 target supervision
- `docs/experiments/stratified_and_proposal_matrix.md` — stratified + two-target + A/B/C/D/E queue
- `docs/experiments/proposal_reduction_arms.md` — **A–G reduction/merge results; containment merge validated as safe, fps and confidence-voxel rejected**
- `docs/experiments/scale_up_plan.md` — parked long-run plan, worker/data inventory, obs-proxy calibration prerequisite
- `docs/references/decoder_design_overview.md` — concise decoder/proposal ideas

Detailed logs:
- `docs/references/feedforward_splat_training.md` — released Gaussian prior-work + F0–F3 plan
- `docs/references/future_head_directions.md` — detailed future-head backlog

Specification / contract:
- `specs/FF-POWER-FOAM-SPEC-v0.md`
- `docs/baselines.md`

Data:
- `docs/data.md`

Machine-readable results:
- `docs/experiments/scannetpp_triplet_overfit_table.csv`

## 15. The single most important takeaway

**The architectural limitation "view-2 contributes features but no cells" was real and removing it (A/B/C arms) yields large gains at matched budget (B vs stratified, +4.48 dB mean).** All-view pixel-aligned proposal concatenation is the right architectural step forward. Voxel reduction (arm C) as-implemented is the wrong direction; it loses depth edges and density structure. Next: confidence-weighted voxel pooling, FPS, or learned proposal selection — combined with neighbor-derived radius initialization.

Until **scene-disjoint generalization** is demonstrated with proposal fusion + neighbor-derived radius, do not scale up. The current 4-scene / 4-view ScanNet++ pilot (Section 5.2/5.3) shows that even with multi-context training the model can copy its source well but fails to produce transferable cross-view geometry at scale. Stratified fixed-triplet capacity is encouraging but is *overfit capacity*.

Good luck. The current state is stable, tests pass, runs are documented. The hard work ahead is selection-strategy design, matched Gaussian baseline, and the scale-up safeguards in Section 10 step 6.