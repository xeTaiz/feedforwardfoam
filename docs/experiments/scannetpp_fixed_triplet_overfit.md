# Fixed ScanNet++ parallax-triplet overfit

Status: running

## Purpose

Test whether the current canonical Power Foam head can memorize one controlled
held-out target when given two calibrated context views with useful translational
parallax. This separates per-triplet capacity/optimization from stochastic
multi-scene generalization.

## Selected triplet

Scene: `f9397af4cb`

| Role | Image |
|---|---|
| Context 0 | `4b1cd91b_DSC04956.JPG` |
| Target | `4b1cd91b_DSC04962.JPG` |
| Context 1 | `4b1cd91b_DSC04970.JPG` |

Calibrated camera geometry:

- context baseline: 0.28619 m;
- target interpolation along the context segment: 0.50018;
- target perpendicular distance: 0.00184 m;
- perpendicular distance / baseline: 0.00643;
- context-to-target distances: 0.14316 m and 0.14305 m;
- context-context optical-axis angle: 3.16 degrees;
- context-target optical-axis angles: 0.95 and 2.82 degrees.

Thus, the target is almost exactly halfway between the two context cameras,
with translation-dominated parallax and strongly overlapping viewing axes.
The triplet was selected by `scripts/select_scannetpp_triplet.py`; its geometric
constraints and ranked contact sheet are reproducible.

## Runs

All runs use 80×80 images, 6,400 canonical cells, frozen VGGT-Ω, the full Power
Foam renderer, fixed density 10,000, full-frame target MSE, learning rate 5e-4,
and 5,000 repeated updates on exactly this triplet. Diagnostic renders and
checkpoints are written every 250 steps.

| Config | Purpose | Worker job |
|---|---|---|
| `overfit_scannetpp_triplet_1ctx.yaml` | one-context capacity control | `834dda43` |
| `overfit_scannetpp_triplet_2ctx_registers.yaml` | joint VGGT/register control | `25bdaa06` |
| `overfit_scannetpp_triplet_2ctx_fused.yaml` | projected spatial-evidence experiment | `0f8a3b1c` |

The fixed image names are part of each config, so resume cannot silently select
a different episode. Training validates that the target lies on the context
camera segment before optimization.

## Early signal

The two-context projected model exceeded 25 dB on the fixed held-out target by
approximately step 160 and reached 26.39 dB at step 280. Gradients, radii,
opacity, and the depth-alignment scale remained finite. This already shows that
the two-context path can optimize the intended controlled triplet; final values
and render inspection remain pending.
