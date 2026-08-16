# Fixed ScanNet++ parallax-triplet overfit

Status: first triplet complete; observable-region matrix running

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

## First-triplet results

| Arm | Final PSNR | Best PSNR | Best step | Last-1k slope |
|---|---:|---:|---:|---:|
| one context | 30.64 dB | 32.02 dB | 4,159 | -0.71 dB/1k |
| two-context registers | 29.18 dB | 29.71 dB | 1,709 | +0.67 dB/1k |
| two-context projected fusion | **31.23 dB** | **31.27 dB** | 4,971 | +0.52 dB/1k |

All three completed 5,000 finite updates. They are budget-complete but not all
strictly converged: constant-LR noise makes the one-context final lower than its
best value, while both two-context curves still have positive last-1k slopes.
The projected fusion path nevertheless demonstrates successful fixed-triplet
overfitting and substantially outperforms the two-context registers-only arm.

## Observable-region matrix

A second matrix repeats projected-fusion overfitting on the selected top triplet
and three deterministic random, geometrically valid triplets (seed 1701), one
from each training scene. These runs train and report both full-frame and
observable-region metrics. The observable mask projects the canonical view-1
VGGT depth anchors into the target and dilates them by two target pixels; this
matches the current representation, which creates cells only from view-1
anchors. It intentionally does not credit view-2-only pixels, because view 2
currently contributes features but no additional cells.

| Run | Scene | Contexts → target | Baseline | Interpolation | Perpendicular fraction | Max angle |
|---|---|---|---:|---:|---:|---:|
| top masked | `f9397af4cb` | `04956,04970 → 04962` | 0.286 | 0.500 | 0.006 | 3.16° |
| random 00a | `00a231a370` | `05455,05464 → 05458` | 0.206 | 0.399 | 0.179 | 19.75° |
| random f939 | `f9397af4cb` | `04953,04968 → 04962` | 0.312 | 0.681 | 0.010 | 4.42° |
| random fd | `fd361ab85f` | `04560,04564 → 04562` | 0.221 | 0.548 | 0.112 | 34.31° |

The final table is generated with `scripts/summarize_triplet_overfits.py` and
will include camera geometry, cross-context support, target observable fraction,
full/support PSNR, opacity, depth scale, and convergence slope.
