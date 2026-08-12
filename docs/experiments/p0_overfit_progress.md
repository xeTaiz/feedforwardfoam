# P0 single-scene overfit progress

Status: complete for the initial O0/H0–H5 matrix

## Goal

Before held-out NVS, establish that (1) upstream Power Foam can optimize the Lego scene, and (2) the frozen-VGGT-Ω P0 head can overfit one fixed canonical image without collapsing rendered opacity.

## Experiment matrix

| ID | Representation / optimization | Radius initialization | Density / alpha mitigation | Status |
|---|---|---|---|---|
| O0 | Upstream per-scene Power Foam oracle, 5k cells/2k steps | Upstream kNN/camera-footprint initialization | Upstream learned density | complete |
| H0 | P0 head, all 160×160 pixels | Learned absolute radius | Learned density, RGB only | complete |
| H1 | P0 head, all 160×160 pixels | Depth × neighboring-ray pixel footprint × learned scale | Learned density, RGB only | complete |
| H2 | Same as H1 | Geometry-aware | Learned density + foreground alpha loss | complete |
| H3 | Same as H1 | Geometry-aware | Fixed high Beer–Lambert density for every pixel, RGB only | complete |
| H4 | Same as H1 | Geometry-aware | Fixed high density only for source-foreground cells | complete |
| H5 | Same as H4 | Geometry-aware | Foreground-masked fixed density + foreground alpha loss | complete |

All H-runs use one fixed Lego canonical view as both context and target, frozen VGGT-Ω, full unmodified Power Foam renderer, 160×160 images, and one cell per input pixel (25,600 cells). This is an overfit/conditioning test, not held-out NVS evidence.

## Progress log

- 2026-08-09: Real VGGT-Ω adapter corrected to 2048-wide registers and channel-first dense maps. NeRF-Synthetic Lego downloaded on `KW60996`.
- 2026-08-09: Sparse 64-cell held-out tests produced zero opacity/zero gradients. Same-view 512-cell tests produced initial gradients but Foam alpha declined toward zero, motivating this matrix.
- 2026-08-09: Implemented footprint-scaled radii, RGBA foreground supervision, fixed-density mode, source-RGB initialization, and all-pixel overfit configs at `8321432`.
- 2026-08-09: H1 completed: RGB loss 0.1320→0.1280, PSNR 11.40→11.78 dB, alpha 0.00345→0.01481. Geometry-aware radii recovered opacity rather than collapsing, but convergence remained weak.
- 2026-08-09: H2 completed: foreground alpha L1 alone failed to recover missed rays; alpha 0.00344→0.00145 and alpha loss stayed ~0.313. Raster support is discrete, so pixels with no intersecting cell do not provide a useful radius-growth gradient.
- 2026-08-09: H3 completed: fixed density prevented collapse (alpha ~0.71) but made background cells opaque and hurt RGB fitting; RGB loss 0.3203→best 0.2886, ending 0.2904, PSNR 9.84 dB. This motivates H4 foreground-masked fixed density.
- 2026-08-09: H0 completed: learned absolute radii recovered from an initial alpha decline and reached 13.43 dB / RGB loss 0.1201 / alpha 0.0601 at step 30.
- 2026-08-09: H4 completed 30 steps: foreground-masked fixed density reached 20.63 dB and RGB loss 0.04353; alpha stayed nonzero (0.229→0.218). This is the first successful same-view P0 head overfit signal. Extended H4 to 100 steps and added H5 with alpha loss.
- 2026-08-09: First O0 invocation exposed upstream Blender's `eval: false` requirement for `transforms_all.json`; switched oracle config to `eval: true`. The 5k-cell/200-step upstream oracle reached held-out-test PSNR 12.10 dB, SSIM 0.305, LPIPS 0.621. At 2,000 steps it improved to PSNR 14.13 dB, SSIM 0.487, LPIPS 0.489. This confirms optimization is working but is not a saturated upstream ceiling (small 5k budget, only 2k steps, 100-view scene fitting).
- 2026-08-09: H4 at 100 steps ended RGB loss 0.04293 / PSNR 20.69 dB / alpha 0.216; best PSNR was 20.71 dB at step 40. It plateaued rather than reaching a near-perfect image fit.
- 2026-08-09: H5 at 100 steps ended RGB loss 0.04839 / PSNR 19.55 dB / alpha 0.237; best PSNR 19.88 dB. Alpha supervision increased mean alpha/radius but degraded RGB relative to H4, so weight 1.0 is too strong once density is foreground-masked.

## Result summary

| ID | Steps | Final/best PSNR | RGB loss | Mean alpha | Mean radius | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| O0 | 2,000 | 14.13 dB held-out average | upstream composite objective | — | upstream kNN | Working scene optimizer, not saturated |
| H0 | 30 | 13.43 dB | 0.12015 | 0.0601 | 0.0758 | Absolute radius eventually recovers, inefficient |
| H1 | 30 | 11.78 dB | 0.12803 | 0.0148 | 0.00726 | Footprint scale is well-conditioned but density remains too weak |
| H2 | 30 | 11.40 dB | 0.13173 | 0.00145 | 0.00471 | Alpha loss cannot create raster support on missing rays |
| H3 | 30 | 9.84 dB | 0.29040 | 0.7123 | 0.00472 | Opaque background cells are harmful |
| H4 | 100 | **20.71 dB best** | **0.04293 final** | 0.2163 | 0.00472 | Best arm: footprint radius + source RGB + foreground-masked fixed density |
| H5 | 100 | 19.88 dB best | 0.04839 final | 0.2374 | 0.00774 | Alpha weight 1 raises coverage but hurts RGB |

## Decision

Use pixel-footprint radius initialization with a predicted bounded multiplicative scale. For the first P0 stage, do not predict unconstrained density: initialize/fix high density only on source-foreground cells (effectively opaque Beer–Lambert segments) and leave background cells empty. Do not use a full-strength global alpha L1 term; revisit a small foreground-only weight after held-out geometry is stable. H4 is a successful non-collapse same-view overfit signal, but its 20.7 dB plateau means it is not yet a near-perfect oracle fit and must not be presented as held-out NVS evidence.

## Initialization identity audit

The 20.7 dB plateau triggered a stricter renderer-contract audit. H4 was not an identity initialization: the head supplied `[0,1]` sigmoid RGB although upstream spherical-Voronoi values are centered and receive `+0.5`; it supplied positive physical radii although `PowerfoamScene.get_radii()` applies another `softplus(beta=100)`; density 100 is only finite Beer–Lambert attenuation; and the identity quaternion produces a world-`+X` dipole normal rather than a camera-facing normal. Therefore H4 cannot establish whether exact pixel-ray initialization reproduces the image. `scripts/check_same_view_identity.py` adds the required deterministic test, bypassing VGGT and the decoder: one cell on every exact renderer ray, centered source RGB, inverse-softplus physical radius, camera-facing normals, and effectively opaque density. Its CUDA result must be recorded before relying on the P0 initialization.

## Reporting contract

For every run record: command/config, commit, steps, initial/final/best RGB loss and PSNR, alpha mean, gradient norm, mean radius, active cells, runtime/failure, and whether opacity collapsed. Do not interpret same-view metrics as NVS quality.
