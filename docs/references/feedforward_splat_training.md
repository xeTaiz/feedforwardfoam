# Feed-forward splatting training references

Status: source audit, 2026-08-14

See also `future_head_directions.md` for the longer-term experiment and architecture backlog.

This note records the training contracts that matter for the current failure:
the MV2 Power Foam head reconstructs its source camera at **46.39 dB**, but
cross-renders at only **5.27 dB**. That is a geometry/generalization failure,
not a renderer-capacity failure.

Released implementations are pinned under `external/references/`. They are
reference-only and are not imported by `feedforwardfoam`.

| Method | Paper | Pinned code | Context / supervision | Most relevant idea |
|---|---|---|---|---|
| DA3 | [arXiv:2511.10647](https://arxiv.org/abs/2511.10647) | `depth-anything-3@3d835ec` | Multiple context images enter the backbone; target image is render-only ground truth | Frozen backbone; MSE+LPIPS target rendering plus context-depth loss; physical GS adapter |
| MVSplat | [arXiv:2403.14627](https://arxiv.org/abs/2403.14627) | `mvsplat@01f9a28` | 2 context / 1 held-out target | Explicit cost-volume geometry; MSE 1, delayed LPIPS .05, depth smoothness .25 |
| pixelSplat | [arXiv:2312.12337](https://arxiv.org/abs/2312.12337) | `pixelsplat@59d420a` | 2 context / 1 held-out target | Epipolar cross-attention, probabilistic depth, opacity warm-up |
| Splatt3R | [arXiv:2408.13912](https://arxiv.org/abs/2408.13912) | `splatt3r@bda1dd0` | 2 context / 3 targets | Frozen geometry backbone, target correspondence mask, geometry/render curriculum |
| StreamSplat | [arXiv:2506.08862](https://arxiv.org/abs/2506.08862) | `streamsplat@5d686e9` | Source frames are also supervised; 1-frame static then 2-endpoint dynamic window | Two stages, RGB+depth, probabilistic positions, delayed LPIPS |
| FlashMono | [arXiv:2604.03092](https://arxiv.org/abs/2604.03092) | no training code released | Recurrent monocular sequence | Per-frame plus merged-map rendering losses; staged sequence curriculum |
| Anchor3R | [arXiv:2606.05035](https://arxiv.org/abs/2606.05035) | no code released | 10-frame current-centric window; point/pose supervision, not NVS | Local pointmap/pose loss with shared scale; do not treat as a splatting baseline |

## What the released methods actually optimize

### DA3

DA3's public repository contains the model and renderer, but **not its training
loop**. The paper states:

- context images alone enter the cross-view backbone;
- every context view emits pixel-aligned Gaussians and their union is rendered;
- target images are used only for `MSE + LPIPS` on novel-view renders;
- context-view depth receives a scale/shift-invariant depth objective;
- the pretrained backbone is frozen while the GS-DPT head is trained.

The released `GaussianAdapter` is especially actionable. It predicts bounded
sub-pixel XY and depth offsets, scales physical radii by depth and pixel
footprint, normalizes camera-frame quaternions and rotates them into world
space, uses low-amplitude higher-order SH initialization, and maps confidence
to opacity. Relevant files:

- `model/gsdpt.py`
- `model/gs_adapter.py`
- `model/utils/gs_renderer.py`

### MVSplat and pixelSplat

Both use two input views and a separate target. Their successful geometry does
not come from merely averaging more target losses: MVSplat constructs a
plane-sweep cost volume; pixelSplat performs epipolar cross-attention and
samples probabilistic depth hypotheses.

Code-verified common defaults:

- `num_context_views: 2`, `num_target_views: 1`;
- RGB MSE weight `1.0`;
- LPIPS weight `0.05`, enabled only after step `150000`;
- depth smoothness weight `0.25`;
- bounded context baseline sampling.

MVSplat uses Adam with LR `2e-4`, 2k warm-up, cosine scheduling, and gradient
clip `0.5`. pixelSplat additionally warms the PDF-to-opacity mapping to avoid
early committing to brittle opaque geometry.

### Splatt3R

Splatt3R is the closest released architectural precedent: a Gaussian head on a
frozen MASt3R geometry backbone.

Code-verified defaults in `configs/main.yaml`:

- 2 context views, 3 target views;
- 512-square images, batch 12;
- head LR `1e-5`, weight decay `.05`, gradient clip `.5`;
- MSE `1.0`, LPIPS `.25`, optional MASt3R geometry loss `.05`;
- mean offsets enabled, SH degree 1;
- rendering losses are averaged only over a projected correspondence/visibility
  mask (`apply_mask`, `average_over_mask`).

The visibility mask is important scientifically: a one-view representation
cannot explain disoccluded target pixels. Penalizing those pixels forces a
black/blur compromise and gives an impossible training signal. The mask must
come from geometry/correspondence, not from the target RGB.

### StreamSplat

StreamSplat is dynamic and uncalibrated, so it is not our direct protocol. Its
released training code nevertheless provides useful stabilization patterns:

- Stage 1 learns static RGB-D-to-Gaussian prediction.
- Stage 2 freezes the static predictor and trains a bidirectional dynamic
  decoder from the first/last views of a six-frame window.
- AdamW LR `5e-4`, weight decay `.05`, betas `.9/.95`, warm-up then cosine.
- MSE plus scale/shift-invariant depth; VGG LPIPS `.05` starts at epoch 50.
- Gaussian positions use a truncated-Gaussian probabilistic decoder.

Unlike DA3/MVSplat/Splatt3R, its source frames are also reconstruction targets;
there is no clean held-out target split. Do not copy that protocol as NVS
evidence.

### FlashMono and Anchor3R

FlashMono has no released Python training implementation. The paper's useful
lesson is to combine per-frame-map rendering with merged-map rendering: merged
loss alone encourages primitives to shrink to avoid cross-frame conflicts.
Its recurrent SLAM and loop-closure machinery is outside P0.

Anchor3R is **not a Gaussian or NVS method**. It predicts current-centric
relative poses and pointmaps. Its code is unavailable. Its shared scale factor
for pointmap and camera translation, confidence-weighted point/gradient loss,
and local-window gauge are possible future geometry ideas, not immediate Foam
losses.

## Diagnosis of our pilot

Our experiment did apply held-out target losses correctly: one foam was
predicted once and rendered into every selected target, with their losses
averaged. The problem is that this alone is substantially weaker than the
successful recipes above:

1. **Only one context image.** VGGT-Ω therefore has no cross-view evidence in
   the episode. The head can copy source color onto source-depth cells.
2. **Self-view initialization is already nearly exact.** Source RGB plus one
   cell per ray yields 46 dB without demonstrating 3D consistency.
3. **VGGT depth is frozen and unconstrained.** Rendering loss can move cells
   and normals away from the depth prior with no context-depth/point loss.
4. **One opaque layer.** Fixed density 10,000 gives no empty space,
   uncertainty, disocclusion, or multiple depth hypotheses.
5. **Unmasked target RGB.** Pixels not observable from the source are included
   even though a one-layer one-view foam cannot infer them.
6. **Tiny data and schedule.** Three training scenes, 80², 6,400 cells, 1,000
   updates are a renderer smoke, unlike published multi-dataset schedules.

## Proposed training ladder

Do not change everything in one run. Each stage should be compared to the
matched Gaussian head and must report self and cross-view PSNR separately.

### F0 — one-context geometry-locked diagnostic

Purpose: determine whether source-depth geometry itself cross-renders.

- Keep the one-source P0 definition.
- Freeze points to VGGT-Ω depth for an initial phase; train appearance and a
  bounded radius/density head only.
- Supervise one nearby held-out target per step, sampled by both translation
  and viewing-direction overlap, not camera-center distance alone.
- Add confidence-weighted penalties for point/depth residual, normal residual,
  and radius residual. Start at zero residual and anneal their bounds slowly.
- Mask loss to target pixels geometrically visible from source depth; report
  both masked and full-frame metrics.
- Use plain RGB MSE first. Add LPIPS only after cross PSNR begins increasing.

This is still monocular NVS and should have a low ceiling, but it tests whether
our camera/depth/world contracts can generate a transferable surface.

### F1 — two-context, one canonical foam

This is the recommended next real experiment.

- Feed **two calibrated overlapping context images jointly to VGGT-Ω**.
- Retain one canonical output foam, satisfying the project constraint; fuse
  evidence before decoding cells rather than creating and merging two foams.
- Warp/sample view-2 frozen features, RGB, depth, confidence, and ray direction
  into canonical-view anchors using the VGGT point map and cameras. Concatenate
  those with view-1 features plus agreement statistics.
- Predict bounded point/radius/normal/density residuals from fused anchor
  evidence. Density must be learned or confidence/visibility initialized.
- Sample one held-out target per optimizer step, as in MVSplat. More target
  views can be sampled across steps; averaging 8 expensive renders in one step
  is not inherently superior.
- Loss: target MSE + context depth/point consistency + cross-context
  reprojection/feature consistency. Add delayed LPIPS after geometry warm-up.

### F2 — proposal fusion before Foam decoding

If F1 works, allow both contexts to propose points, but fuse them in world
space before creating cells:

- confidence-weighted voxel/cluster fusion of per-view points;
- one anchor token and one Foam cell per fused region;
- track view count, point variance, normal variance, and visibility;
- coverage-aware budget selection rather than source-view top-M;
- one global Foam is decoded and rendered into targets.

This follows the useful part of DA3's multi-view union while avoiding the
invalid operation of merging already-formed power diagrams.

### F3 — uncertainty and disocclusion

- Predict 2–3 depth hypotheses only near low-confidence/depth-edge pixels,
  inspired by pixelSplat's probabilistic depth and StreamSplat's probabilistic
  position head.
- Learn density/occupancy with a warm-up rather than fixing every cell opaque.
- Delay high-order directional appearance; begin with view-independent color
  or heavily damped spherical-Voronoi residuals so appearance cannot hide bad
  geometry.
- Consider a source-visible component loss plus global-foam target loss,
  analogous to FlashMono's component/merged objectives.

## First ablation matrix

At fixed resolution, data, primitive budget, optimizer steps, and target
sampler:

| Arm | Context | Geometry | Target loss | Purpose |
|---|---:|---|---|---|
| A | 1 | frozen VGGT points | masked MSE | camera/depth cross-render gate |
| B | 1 | bounded residual | masked MSE + geometry | does residual improve? |
| C | 2 | canonical anchors, view-1 only | same as B | input-count control |
| D | 2 | pre-decoder fused features | same as B | key fusion test |
| E | 2 | fused proposals | same as B | P1 value |

Run the same A–E matrix with the DA3-style Gaussian adapter. This distinguishes
representation failure from conditioning/training failure.

## Required metrics

- self-camera and held-out cross-camera PSNR/SSIM/LPIPS;
- masked-visible and full-frame metrics;
- context-depth residual before/after the head;
- target rendered-depth consistency where geometry exists;
- alpha/coverage, empty-space fraction, mean/effective radius;
- cells visible in 1 vs 2 contexts;
- point reprojection error and cross-context RGB/feature agreement;
- gradient norms by geometry, density, radius, and appearance parameter groups.

## Bottom line

Our multi-target averaging implementation was not obviously wrong. It was
**insufficient**: successful methods pair held-out rendering with multi-view
geometric reasoning, geometry/depth supervision, careful visibility, and a
curriculum. The highest-value next change is not 8 targets. It is **2 context
views jointly processed, one canonical foam, fused evidence before cell
prediction, one held-out target per step, and explicit geometry preservation**.
