# Proposal reduction arms A–E on the stratified 12-triplet matrix

Status: complete. 3,000 steps, cosine LR, `initialization` + `full` modes,
12 stratified triplets per arm, real Power Foam renderer.

Worker: `KW60996`, checkout `/code/feedforwardfoam-abc`.
Outputs: `runs/proposals_{a_all,b_balanced,c_voxel,d_fps,e_confvoxel}_v1`.

## Arms

| Arm | Reduction | Cells | Selection criterion |
|---|---|---:|---|
| A | none (`all`) | 12,800 | every pixel of both contexts is rendered |
| B | `balanced` | 6,400 | uniform pixel stride, 3,200 per context |
| C | `voxel` | 6,400 | world voxel grid, lowest-index member per voxel |
| D | `fps` | 6,400 | **new** — farthest-point sampling in world space |
| E | `confidence_voxel` | 6,400 | **new** — same voxel grid as C, highest depth-confidence member per voxel |

D and E are the two follow-ups named in the previous handoff as the likely fixes
for arm C. Arms A–C were re-aggregated from their existing runs; their mean
deltas reproduce the previously published values exactly (A − B = +3.06,
B − C = +3.91, A − C = +6.96 dB), which validates the aggregation.

E is a strict single-variable change from C: identical voxel grid, identical
bisection, identical trim/fill, only the intra-voxel representative differs.
Without scores that representative is the lowest concatenation index, which
systematically favours whichever context view was concatenated first.

## Results — `full` mode, best full-frame PSNR

| Triplet | A all | B balanced | C voxel | D fps | E conf-voxel |
|---|---:|---:|---:|---:|---:|
| 00a_q05 | **12.26** | 11.90 | 11.86 | 11.87 | 11.62 |
| 00a_q30 | **36.36** | 25.31 | 20.33 | 20.16 | 20.17 |
| 00a_q60 | **19.24** | 17.86 | 12.97 | 11.96 | 13.02 |
| 00a_q90 | **12.32** | 11.75 | 11.78 | 11.68 | 11.69 |
| f939_q05 | **26.52** | 22.27 | 21.57 | 20.98 | 20.87 |
| f939_q30 | **30.88** | 29.21 | 19.61 | 19.76 | 19.70 |
| f939_q60 | **26.31** | 24.57 | 21.97 | 21.39 | 20.56 |
| f939_q90 | **23.65** | 20.00 | 12.72 | 14.28 | 12.93 |
| fd_q05 | **31.42** | 28.09 | 22.28 | 21.70 | 20.03 |
| fd_q30 | **33.63** | 26.87 | 25.27 | 24.08 | 23.60 |
| fd_q60 | **28.78** | 27.09 | 18.83 | 18.65 | 18.60 |
| fd_q90 | **14.78** | 14.57 | 13.43 | 12.98 | 13.12 |
| **mean** | **24.68** | 21.62 | 17.72 | 17.46 | 17.16 |

Best support PSNR means: A 27.46, B 23.44, C 18.96, D 18.62, E 18.28.

## Results — `initialization` mode, best full-frame PSNR

Decoder-free physical baseline; no learned residuals.

| Arm | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| mean | **19.20** | 14.65 | 13.29 | 12.94 | 13.11 |

## Findings

**1. Both new arms fail. The handoff's hypothesis is refuted.**
Farthest-point sampling and confidence-weighted voxel pooling do not repair arm
C. Against C: D wins 3/12 triplets (mean **−0.26 dB**), E wins 3/12
(mean **−0.56 dB**). Neither is a meaningful improvement and both trend slightly
worse. Against the budget-matched arm B they lose by 4.17 and 4.47 dB mean.

**2. The decisive variable is world-space versus pixel-space reduction, not the
world-space criterion.**
Every world-space reduction lands in a tight 17.16–17.72 dB band regardless of
whether it selects by voxel occupancy, by depth confidence, or by farthest-point
coverage. Uniform pixel striding at the identical 6,400 budget reaches 21.62 dB.
The ~4 dB gap is a property of *where* selection happens, not of *how well* the
world-space criterion covers space — FPS maximizes world-space coverage by
construction and still loses.

**3. The damage is to the tessellation, not to learning.**
The same ordering appears in `initialization` mode, which has no trained
decoder at all (A 19.20 > B 14.65 > C 13.29 ≈ E 13.11 ≈ D 12.94). Because that
mode only builds and renders the physical initialization, the penalty must come
from the site distribution fed to the single Čech/power-diagram build. Irregular
world-space site sets tessellate worse than pixel-regular ones, before any
gradient is taken.

**4. Extra primitive budget still dominates everything.**
Arm A wins 12/12 against every other arm on both full-frame and support PSNR.
No 6,400-cell reduction is competitive with simply rendering all 12,800
proposals.

## Consequences for the next reduction attempt

Stop searching world-space criteria. If a reduced-budget arm is wanted, it
should stay in per-view pixel space, where arm B already demonstrates the
regularity that matters — for example confidence-weighted or edge-aware pixel
striding, or a learned per-view pixel gate, each preserving an approximately
regular per-view lattice. Until such an arm beats B, arm A remains the default.

## Incidental diagnostic: depth gauge saturation

`depth_alignment_scale` is identical across arms (alignment runs upstream of
proposal reduction). **4 of 12 triplets saturate the upper bound of the
`[0.25, 4]` gauge range**: `00a_q90`, `f939_q05`, `f939_q60`, `fd_q30` all end at
exactly 4.0000. Two more sit above 2.5. This is the depth-gauge bound-hit check
listed as a scale-up readiness item; a third of the fixed matrix is clipping, so
the bound is doing real work and the predicted/calibrated baseline ratio is
frequently outside the assumed range. Investigate before scale-up.

## Caveats

- Fixed-triplet overfit capacity across 3 scenes, not scene-disjoint
  generalization. Same caveat as every prior entry in this matrix.
- `q05…q90` are selector-score quantiles, not overlap quantiles; measured
  observability is non-monotonic in `q`. Do not read the triplet axis as an
  overlap axis.
