# ScanNet++ canonical two-context experiment

Status: CUDA stability gate passed; full matrix launching

## Question

Does adding a second calibrated overlapping context view improve cross-camera
Power Foam reconstruction when the model still predicts exactly one canonical
foam from view-1 anchors?

## Arms

| Arm | Contexts | Spatial fusion | Training loss |
|---|---:|---|---|
| 1ctx control | 1 | none | full target MSE |
| 2ctx registers | 2 | global pooled VGGT-Ω registers only | full target MSE |
| 2ctx fused | 2 | view-2 RGB/depth/confidence/final patch tokens projected to view-1 anchors | full target MSE |
| 2ctx fused masked | 2 | same projected fusion | projected-support target MSE |

Every arm uses one held-out target per optimizer update. The one-context control
reserves/discards the same second sampled view so its source and target indices
match the two-context arms exactly.

## Geometry corrections introduced before launch

- VGGT-Ω final per-view patch tokens are exposed without modifying the pinned
  upstream submodule.
- VGGT depth is treated as camera-forward z-depth, not Euclidean ray distance.
- For two contexts, predicted-camera baseline versus calibrated-camera baseline
  supplies one bounded scale applied to every context depth.
- The supporting context is projected into canonical anchors using calibrated
  cameras and depth-consistency validity.
- The optional target mask is built only from projected context geometry, never
  target RGB; empty masks fall back to full RGB supervision.
- Non-finite gradient clipping now raises immediately.

## Fixed pilot protocol

- Same four-scene ScanNet++ subset as the prior pilot: three train, one
  scene-disjoint validation scene.
- 80×80, 6,400 cells, fixed raw density 10,000.
- One overlapping target sampled from a 24-view translation+orientation pool.
- MSE, LR 1e-4, 2,000 steps, seed 17.
- Eight deterministic validation episodes every 200 steps.
- Full-frame validation PSNR for every arm, even when training is masked.

## Stability gate

All four one-step CUDA smokes passed with finite gradients. An initial 300-step
stability run caught and fixed unbounded depth-gauge scale and empty-mask
failure modes. After fixes, every two-context arm completed 300 steps with
finite parameters/gradients.

One-episode full-frame validation PSNR at step 300 (diagnostic only):

| Arm | PSNR |
|---|---:|
| 1ctx matched control | 4.10 dB |
| 2ctx registers | 4.93 dB |
| 2ctx fused | 4.85 dB |
| 2ctx fused masked | **5.09 dB** |

These single-episode values are not results. The full eight-episode 2,000-step
matrix determines whether the ordering persists.
