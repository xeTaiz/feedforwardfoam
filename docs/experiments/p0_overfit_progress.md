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
| H3 | Same as H1 | Geometry-aware | Fixed high Beer–Lambert density for every pixel, RGB only | complete |
| H4 | Same as H1 | Geometry-aware | Fixed high density only for source-foreground cells | 30-step complete; 100-step running |
| H5 | Same as H4 | Geometry-aware | Foreground-masked fixed density + foreground alpha loss | running |

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
- 2026-08-09: First O0 invocation exposed upstream Blender's `eval: false` requirement for `transforms_all.json`; switched oracle config to `eval: true`. The 5k-cell/200-step upstream oracle trained successfully but only reached held-out-test PSNR 12.10 dB, SSIM 0.305, LPIPS 0.621; extended to 2,000 steps because 200 steps is not a meaningful oracle ceiling.

## Reporting contract

For every run record: command/config, commit, steps, initial/final/best RGB loss and PSNR, alpha mean, gradient norm, mean radius, active cells, runtime/failure, and whether opacity collapsed. Do not interpret same-view metrics as NVS quality.
