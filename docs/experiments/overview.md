# Experiments so far — concise overview

This page summarizes what we tested, what happened, and what it means. Detailed
protocols remain in `docs/experiments/` and machine-readable triplet results are
in `scannetpp_triplet_overfit_table.csv`.

## 1. Renderer and same-view sanity checks

**Question:** Can the unmodified Power Foam renderer reproduce a foam that is
already correct, and can our optimization path fit one image?

- Exact renderer identity: **154.9 dB** (numerical identity).
- Residual initialization identity: metric-capped **100 dB**.
- Same-view color/head tests: approximately **33–36 dB**.

**Conclusion:** Renderer conventions, camera transforms, and gradients work.
These are conditioning tests, not novel-view evidence.

## 2. First real ScanNet++ multi-target pilot

**Question:** Does supervising more target views fix novel-view synthesis from
one canonical context?

| Targets per update | Final validation PSNR |
|---:|---:|
| 1 | 5.50 dB |
| 2 | **6.18 dB** |
| 4 | 5.88 dB |
| 8 | 5.62 dB |

A separate visualization measured **46.39 dB** when rendering the source view
but only **5.27 dB** across cameras.

**Conclusion:** More losses on the same one-view geometry do not solve the
problem. The model copied its source well but did not construct transferable
3-D geometry. Two targets helped slightly; gains were not monotonic.

## 3. Joint two-context training on multiple scenes

**Question:** Does jointly processing two calibrated context views help?

| Head input | Best / final validation PSNR |
|---|---:|
| One context | 9.03 / 9.02 dB |
| Two-context global registers | **9.42 / 9.42 dB** |
| Two-context projected spatial evidence | 9.40 / 9.23 dB |
| Projected evidence + support-masked loss | 9.36 / 9.33 dB |

**Conclusion:** A second context gives a small real gain, but the explicit
projection adapter did not beat the simpler global-register control in general
training. Absolute quality remains poor. This motivates controlled overfits and
better proposal coverage rather than immediate scale-up.

## 4. Fixed ideal-parallax triplet

**Question:** Can the head overfit when two contexts have strong overlap and the
target lies almost exactly between them?

The selected target is at interpolation **0.500**, only **0.0064 baseline
lengths** off the context segment, with **3.16°** maximum view-angle change.

| Model | Best target PSNR |
|---|---:|
| Two-context registers only | 29.71 dB |
| Two-context projected fusion | **31.27 dB** |
| One-context control | 32.02 dB |

**Conclusion:** The two-context projected head has enough capacity to fit a good
triplet and beats registers-only by 1.57 dB. This is overfit capacity, not proof
of scene-disjoint generalization. One context can also memorize this easy target.

## 5. Random-triplet overlap study

**Question:** Does camera overlap predict how well a fixed triplet can be fit?

| Triplet | Target observable | Max angle | Best full PSNR | Best observable PSNR |
|---|---:|---:|---:|---:|
| Ideal f939 | 100.0% | 3.16° | 29.80 | 29.80 |
| Random 00a | 91.7% | 19.75° | 13.93 | **29.79** |
| Random f939 | 100.0% | 4.42° | 27.94 | 27.94 |
| Random fd | 61.6% | 34.31° | 8.22 | 17.36 |

With only four samples, provisional correlations with observable PSNR were
`+0.95` for target coverage and `-0.82` for maximum view angle.

**Conclusion:** Full-frame PSNR can badly understate performance when much of a
target is impossible from view-0 cells. Coverage and angular overlap are strong
candidate variables for training-time view sampling. Four triplets are not
enough for a statistical claim.

## 6. How much does physical initialization help?

**Question:** Is the head succeeding only because depth normals, footprint
radii, and source color already define most of the answer?

| Experiment on the ideal triplet | Best PSNR |
|---|---:|
| Exact initialization; no learned head | 23.61 dB |
| Residual head, full loss with LR decay | 29.90 dB |
| Residual head, observable loss with LR decay | 30.95 dB |
| Absolute attributes; initialize position only | 30.76 dB |
| Appearance-only residual branch | 30.71 dB |
| Radius + appearance residuals | 31.05 dB |

Here “appearance-only” disables point, radius, and orientation residuals, but
still decodes spherical-Voronoi world directions and RGB. Density is fixed and
texel sites/heights remain zero.

**Conclusion:** Initialization is valuable—it starts at 23.61 dB—but it does
not explain all performance. A position-only initialized head can directly
learn the other attributes. On this easy triplet, most learned improvement is
appearance; it does not prove that learned geometry transfers to harder views.

## 7. Direct foam optimization

**Question:** Can the foam tensors themselves fit better than the feed-forward
head?

Direct optimization transiently reached approximately **31.1 dB**, then
degraded even with LR decay.

**Conclusion:** The representation has at least similar capacity to the head,
but this is not a clean upper bound because direct optimization/topology is
unstable. Further hyperparameter tuning is not required before scale-up; best
state retention is sufficient as a diagnostic.

## What is established

1. The renderer and training graph work.
2. One-view source copying is easy; cross-view geometry is the real failure.
3. Two-context input helps modestly in general training and can fit controlled
   triplets well.
4. Physical initialization helps substantially but is not indispensable beyond
   point initialization.
5. Observable-region metrics are necessary for fair diagnosis.
6. Camera overlap and target coverage should inform sampling.
7. The current foam still creates cells only from view-0 pixels; view 2 changes
   their attributes but cannot add cells in view-2-only regions.
