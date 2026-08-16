# Fixed-triplet initialization, absolute-head, and residual ablations

Status: complete first diagnostic pass

## Coordinate systems

The canonical foam is a **world-space** scene, despite canonical proposals
coming from context view 0.

- Points and Power Foam radii are world-space.
- Quaternions are world-space WXYZ rotations mapping local `+X` to the world
  normal.
- Spherical-Voronoi axes are world-space viewing directions; they must **not**
  be rotated by a primitive quaternion. Upstream compares them directly to the
  world camera-to-texel direction.
- Texel sites are the exception: they are local 2-D tangent/bitangent
  coordinates normalized by radius. Texel heights are local dimensionless
  offsets. The upstream renderer transforms them through each primitive
  quaternion and radius.
- Spherical-Voronoi RGB is centered around zero; upstream adds `+0.5` after
  directional interpolation.

Sources: `src/feedforwardfoam/head.py`, `src/feedforwardfoam/renderer.py`, and
`external/powerfoam/powerfoam/{scene.py,color_fn.py}`.

## Protocol

All results use the fixed ideal ScanNet++ triplet from `f9397af4cb`:
contexts `DSC04956`, `DSC04970`; target `DSC04962`. The target is halfway
between contexts (interpolation 0.500), 0.0064 baseline lengths off their
segment, and has 3.16° maximum viewing-axis difference. Images are 80×80,
there are 6,400 view-0 anchored cells, density is fixed to 10,000, and all
heads use frozen VGGT-Ω.

The baseline is a decoder-free exact physical initialization: calibrated VGGT
world points, pixel-footprint radii, depth normals, fixed density, source RGB
repeated across the directional field, canonical SV axes, zero texel offsets.
It does not select cells with decoder gate logits. The 6,400-cell setting means
all canonical pixels are included.

The `absolute` head retains **only points** from that initialization. It directly
predicts radius, WXYZ orientation, world SV axes/RGB, and leaves texel locations
at the present fixed zero-site restriction. It begins with a neutral 5-cm
radius, identity orientation, canonical axes, and grey RGB; these are decoder
bias/defaults, not depth-normal, footprint, or source-color initialization.

## Results

| Experiment | Training objective | Final PSNR | Best PSNR | Interpretation |
|---|---|---:|---:|---|
| Exact initialization | none | 23.61 | 23.61 | Required no-head baseline |
| Residual full, cosine LR | full target MSE | 29.65 | 29.90 | Stable but below masked arm |
| Residual observable-only, cosine LR | canonical support MSE | **30.95** | **30.95** | Best residual result; target is fully observable here |
| Absolute except position, cosine LR | full target MSE | 30.76 | 30.76 | Direct attributes can overfit nearly as well |
| RGB-only residual | full target MSE | 30.71 | 30.71 | Most improvement is appearance correction given initialized geometry |
| Radius + RGB residual | full target MSE | 30.73 | **31.05** | Radius residual contributes little on this triplet |
| Direct foam optimization, constant LR | support MSE | 27.33 | 30.18 | Optimizer/topology instability after early peak |
| Direct foam optimization, cosine LR | support MSE | 27.85 | 31.07 | Same early capacity, still degrades after step 525 |

The support mask is 100% of this easy target, so full and observable PSNR are
the same. It remains necessary for partial-overlap triplets; their
initialization baselines were:

| Triplet | Init full PSNR | Init support PSNR | Target observable |
|---|---:|---:|---:|
| random 00a | 11.76 | 15.60 | 91.7% |
| random f939 | 23.10 | 23.10 | 100.0% |
| random fd | 8.02 | 15.22 | 61.6% |

## Conclusions

1. Initialization matters materially: it supplies 23.61 dB before learning,
   but it does **not** account for all success—absolute prediction improves by
   7.15 dB while retaining positions only.
2. On this easy overlap, residual RGB alone nearly matches full residual
   optimization. This is evidence that geometry is already adequate for this
   triplet, not evidence that geometry learning is unimportant in general.
3. The current canonical support mask is not selective in the easy triplet:
   projected coverage and rendered-alpha coverage are both essentially 100%.
   It is selective on difficult triplets and must remain reported there.
4. Direct optimization has at least 31.07 dB transient capacity, but cannot yet
   be called an upper bound because topology/Adam updates degrade it. Future
   direct upper-bound runs must retain the best parameter state and/or optimize
   under a topology-rebuild schedule.
5. The cosine decoder runs did stabilize head training; best checkpoints are now
   saved as `best_full.pt` and `best_support.pt` for future runs.

Relevant commits: `0c2a758`, `0ccee4c`, `33292fa`, `a624066`, `b6961cc`.
