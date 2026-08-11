# P0 single-scene overfit progress

Status: in progress

## Goal

Before held-out NVS, establish that (1) upstream Power Foam can optimize the Lego scene, and (2) the frozen-VGGT-Ω P0 head can overfit one fixed canonical image without collapsing rendered opacity.

## Experiment matrix

| ID | Representation / optimization | Radius initialization | Density / alpha mitigation | Status |
|---|---|---|---|---|
| O0 | Upstream per-scene Power Foam oracle | Upstream kNN/camera-footprint initialization | Upstream learned density | pending |
| H0 | P0 head, all 160×160 pixels | Learned absolute radius | Learned density, RGB only | pending |
| H1 | P0 head, all 160×160 pixels | Depth × neighboring-ray pixel footprint × learned scale | Learned density, RGB only | pending |
| H2 | Same as H1 | Geometry-aware | Learned density + foreground alpha loss | pending |
| H3 | Same as H1 | Geometry-aware | Fixed high Beer–Lambert density (“effectively opaque”), RGB only | pending |

All H-runs use one fixed Lego canonical view as both context and target, frozen VGGT-Ω, full unmodified Power Foam renderer, 160×160 images, and one cell per input pixel (25,600 cells). This is an overfit/conditioning test, not held-out NVS evidence.

## Progress log

- 2026-08-09: Real VGGT-Ω adapter corrected to 2048-wide registers and channel-first dense maps. NeRF-Synthetic Lego downloaded on `KW60996`.
- 2026-08-09: Sparse 64-cell held-out tests produced zero opacity/zero gradients. Same-view 512-cell tests produced initial gradients but Foam alpha declined toward zero, motivating this matrix.
- 2026-08-09: Began implementing footprint-scaled radii, RGBA foreground supervision, fixed-density mode, and all-pixel overfit configs.

## Reporting contract

For every run record: command/config, commit, steps, initial/final/best RGB loss and PSNR, alpha mean, gradient norm, mean radius, active cells, runtime/failure, and whether opacity collapsed. Do not interpret same-view metrics as NVS quality.
