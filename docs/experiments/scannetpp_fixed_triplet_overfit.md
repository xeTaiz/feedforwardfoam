# Fixed ScanNet++ parallax-triplet overfit

Status: complete

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

The machine-readable full table is
`docs/experiments/scannetpp_triplet_overfit_table.csv` and can be regenerated
with `scripts/summarize_triplet_overfits.py`.

### Geometry and support

`Cross support` is the fraction of canonical view-1 anchors with geometrically
consistent projected evidence from view 2. `Target observable` is the fraction
of target pixels reached by projected canonical anchors after two-pixel
dilation.

| Run | Baseline | Interp. | Perp. frac. | Max angle | Cross support | Target observable |
|---|---:|---:|---:|---:|---:|---:|
| top, registers | 0.286 | 0.500 | 0.006 | 3.16° | — | 100.0% |
| top, fused full loss | 0.286 | 0.500 | 0.006 | 3.16° | — | 100.0% |
| top, fused masked | 0.286 | 0.500 | 0.006 | 3.16° | 38.8% | 100.0% |
| random 00a, fused masked | 0.206 | 0.399 | 0.179 | 19.75° | 1.0% | 91.7% |
| random f939, fused masked | 0.312 | 0.681 | 0.010 | 4.42° | 29.8% | 100.0% |
| random fd, fused masked | 0.221 | 0.548 | 0.112 | 34.31° | 13.5% | 61.6% |

### Overfit results

Best PSNR measures capacity; final PSNR and the slope expose constant-LR
instability. For the old top-triplet runs, the current canonical mask covers
100% of the target, so support and full-frame PSNR are equivalent.

| Run | Best full PSNR | Best support PSNR | Best step | Final full PSNR | Final support PSNR | Final alpha | Last-1k slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| top, registers | 29.71 | 29.71 | 1,709 | 29.18 | 29.18 | 0.999 | +0.67 |
| top, fused full loss | **31.27** | **31.27** | 4,971 | **31.23** | **31.23** | 0.999 | +0.52 |
| top, fused masked | 29.80 | 29.80 | 3,577 | 23.72 | 23.72 | 0.999 | -1.57 |
| random 00a, fused masked | 13.93 | **29.79** | 1,441 | 13.70 | 24.78 | 0.916 | +0.06 |
| random f939, fused masked | 27.94 | 27.94 | 3,609 | 23.79 | 23.79 | 0.998 | -0.16 |
| random fd, fused masked | 8.22 | 17.36 | 4,923 | 8.22 | 17.30 | 0.592 | +0.00 |

### Interpretation

- The original top fused run demonstrates approximately 31.3 dB fixed-triplet
  capacity and beats its registers-only control by 1.57 dB at best.
- Observable-only evaluation changes the conclusion dramatically for partial
  targets: random 00a reaches 29.79 dB on supported pixels despite only 13.93 dB
  full-frame PSNR.
- The random fd triplet has the largest viewing-angle change, lowest target
  observable fraction, low alpha, and by far the worst support PSNR. Masking
  cannot repair poor transferred geometry inside the nominally supported area.
- With only four masked triplets, Pearson correlations with best support PSNR
  are provisional: target observable fraction `r=+0.951`, maximum view angle
  `r=-0.816`, perpendicular fraction `r=-0.209`, and cross-context support
  `r=+0.244`. The sample is too small for a statistical claim, but target
  coverage and angular overlap are the strongest observed predictors.
- Constant learning rate 5e-4 does not yield stable convergence for every
  triplet. Only random fd is clearly plateaued; other runs either retain a
  positive slope or degrade after an earlier best. Use best PSNR as the capacity
  diagnostic and add learning-rate decay plus retained best checkpoints before
  treating final-step PSNR as a convergence result.
