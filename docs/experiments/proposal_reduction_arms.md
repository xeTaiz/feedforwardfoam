# Proposal reduction and merge arms A–G on the stratified 12-triplet matrix

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
| D | `fps` | 6,400 | farthest-point sampling in world space |
| E | `confidence_voxel` | 6,400 | same voxel grid as C, highest depth-confidence member |
| F | `incremental`, power, κ=1.0 | 12,618 | **new** — keep view 0 whole, drop view-1 sites whose centre a kept power cell already claims |
| G | `incremental`, ball, κ=1.5 | 11,190 | **new** — same, plain sphere test at 1.5× the incumbent radius |

F and G are budget-free: they discard only redundancy, so the surviving count
floats with the scene. The containment test against a kept site `i` is

- power: `|p_j - p_i|^2 + r_j^2 <= (κ r_i)^2`, exactly the condition under which
  `p_j` falls outside its own power cell and the new cell degenerates;
- ball: `|p_j - p_i| <= κ r_i`.

κ = 0 reproduces arm A, so the family contains its own control.

D and E are the two follow-ups named in the previous handoff as the likely fixes
for arm C. Arms A–C were re-aggregated from their existing runs; their mean
deltas reproduce the previously published values exactly (A − B = +3.06,
B − C = +3.91, A − C = +6.96 dB), which validates the aggregation.

E is a strict single-variable change from C: identical voxel grid, identical
bisection, identical trim/fill, only the intra-voxel representative differs.
Without scores that representative is the lowest concatenation index, which
systematically favours whichever context view was concatenated first.

## Results — `full` mode, best full-frame PSNR

| Triplet | A all | B balanced | C voxel | D fps | E conf-voxel | F incr power | G incr ball |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00a_q05 | 12.26 | 11.90 | 11.86 | 11.87 | 11.62 | 12.26 | **12.27** |
| 00a_q30 | **36.36** | 25.31 | 20.33 | 20.16 | 20.17 | 33.21 | 32.61 |
| 00a_q60 | **19.24** | 17.86 | 12.97 | 11.96 | 13.02 | 19.05 | 17.25 |
| 00a_q90 | 12.32 | 11.75 | 11.78 | 11.68 | 11.69 | **12.37** | 12.25 |
| f939_q05 | 26.52 | 22.27 | 21.57 | 20.98 | 20.87 | 25.91 | **26.77** |
| f939_q30 | 30.88 | 29.21 | 19.61 | 19.76 | 19.70 | 31.17 | **31.26** |
| f939_q60 | **26.31** | 24.57 | 21.97 | 21.39 | 20.56 | 26.03 | 25.96 |
| f939_q90 | 23.65 | 20.00 | 12.72 | 14.28 | 12.93 | **23.67** | 23.51 |
| fd_q05 | **31.42** | 28.09 | 22.28 | 21.70 | 20.03 | 30.91 | 31.12 |
| fd_q30 | **33.63** | 26.87 | 25.27 | 24.08 | 23.60 | 33.55 | 30.35 |
| fd_q60 | 28.78 | 27.09 | 18.83 | 18.65 | 18.60 | **29.25** | 29.14 |
| fd_q90 | 14.78 | 14.57 | 13.43 | 12.98 | 13.12 | 14.76 | **14.84** |
| **mean** | **24.68** | 21.62 | 17.72 | 17.46 | 17.16 | 24.35 | 23.94 |
| **mean cells** | 12,800 | 6,400 | 6,400 | 6,400 | 6,400 | 12,618 | 11,190 |

Mean delta against arm A, with per-triplet wins:

| Arm | Δ vs A | wins |
|---|---:|---:|
| B balanced | −3.06 dB | 0/12 |
| C voxel | −6.96 dB | 0/12 |
| D fps | −7.22 dB | 0/12 |
| E conf-voxel | −7.52 dB | 0/12 |
| **F incr power** | **−0.34 dB** | **4/12** |
| **G incr ball** | **−0.74 dB** | **5/12** |

Best support PSNR means for the budget arms: A 27.46, B 23.44, C 18.96,
D 18.62, E 18.28.

## Results — `initialization` mode, best full-frame PSNR

Decoder-free physical baseline; no learned residuals.

| Arm | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| mean | **19.20** | 14.65 | 13.29 | 12.94 | 13.11 |

## Findings

**1. The two world-space budget follow-ups fail. The handoff hypothesis is refuted.**
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

**4. Containment merging is the only reduction that preserves arm A's quality.**
F costs **−0.34 dB** mean against A while removing 1.4% of cells, and wins 4/12
triplets outright. Its mean is dominated by a single triplet, `00a_q30`
(−3.15 dB, the strongest result anywhere in the matrix and therefore the most
sensitive); across the other eleven triplets F averages **−0.08 dB**, i.e.
indistinguishable from A. G removes 12.6% of cells for −0.74 dB.

So the merge is *safe*: unlike every budget arm, it does not destroy the scene.
But it is not a *gain*, because there is almost nothing to remove — see the
overlap diagnostic below. Its value is the guarantee, not the quality.

**5. Extra primitive budget still dominates every fixed-budget arm.**
No 6,400-cell reduction is competitive with rendering all 12,800 proposals.

## Why containment merging cannot help: the overlap diagnostic

`scripts/diagnose_proposal_overlap.py` measures nearest-neighbour distances
between the two proposal groups in units of the local physical cell radius.

| Triplet | gauge | co-visible frac | within-view NN/r | cross-view NN/r | co-visible NN/r | power hit | ball hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00a_q05 | 1.94 | 0.122 | 1.12 | 6.58 | **0.99** | 0.0112 | 0.0741 |
| f939_q05 | 4.00 | 0.430 | 1.12 | 2.95 | **1.12** | 0.0030 | 0.1778 |
| f939_q30 | 2.99 | 0.609 | 1.10 | 2.32 | **1.38** | 0.0108 | 0.1834 |
| fd_q30 | 4.00 | 0.195 | 1.02 | 4.05 | **1.30** | 0.0014 | 0.0794 |

(medians; `hit` columns are the fraction of view-1 sites the criterion swallows.)

Two facts settle the design question:

- **The clouds are correctly registered.** Restricted to pixels the fusion stage
  marks co-visible, cross-view nearest-neighbour distance is 0.99–1.38 radii,
  statistically the same as the within-view lattice spacing of 1.02–1.12. The
  large *global* cross-view distance (2.3–6.6 radii) is not misregistration; it
  is simply that only 12–61% of the canonical view is co-visible and the rest of
  view 1 contributes genuinely new surface.
- **Co-visible "duplicates" are adjacent, not coincident.** They sit about one
  radius apart — exactly the within-view spacing. A second view therefore does
  not deposit a copy on top of an existing cell; it interleaves a finer sample
  of the same surface. Containment can only fire on 0.14–1.1% of sites (power)
  or 1.1–18% (ball), and what it removes is lattice densification rather than
  redundancy.

This also explains arm A's dominance mechanically: in co-visible regions the
second view roughly doubles surface sampling density, and elsewhere it supplies
coverage a single canonical view cannot reach. Both effects are destroyed by any
fixed budget.

A retention sweep over the containment knob (3 triplets, fraction of 12,800
proposals kept): power κ=1.0 → 99.4%, κ=2.0 → 88.2%, κ=3.0 → 83.4%;
ball κ=0.5 → 99.4%, κ=1.0 → 95.6%, κ=1.5 → 90.2%. Even the most aggressive
setting keeps 83%.

## Consequences

Keep arm A as the rendering default, and enable `incremental` power containment
at κ=1.0 as a cheap structural guarantee: it provably leaves no site stranded
inside another cell, costs ~0.1 dB outside one sensitive triplet, and removes a
small class of degenerate slivers that would otherwise be handed to the Čech
build. Do not expect quality from it.

Stop searching world-space *budget* criteria. If a reduced-budget arm is ever
needed, it must stay in per-view pixel space, where arm B already demonstrates
the regularity that matters.

## Incidental diagnostic: depth gauge saturation

`depth_alignment_scale` is identical across arms (alignment runs upstream of
proposal reduction). **4 of 12 triplets saturate the upper bound of the
`[0.25, 4]` gauge range**: `00a_q90`, `f939_q05`, `f939_q60`, `fd_q30` all end at
exactly 4.0000. Two more sit above 2.5. Note that saturation does **not** imply
broken registration: `f939_q05` and `fd_q30` both clip at 4.0 yet still show
co-visible nearest-neighbour distances of 1.12 and 1.30 radii. The bound is
nevertheless active on a third of the matrix and should be understood before
scale-up.

## Caveats

- Fixed-triplet overfit capacity across 3 scenes, not scene-disjoint
  generalization. Same caveat as every prior entry in this matrix.
- `q05…q90` are selector-score quantiles, not overlap quantiles; measured
  observability is non-monotonic in `q`. Do not read the triplet axis as an
  overlap axis.
