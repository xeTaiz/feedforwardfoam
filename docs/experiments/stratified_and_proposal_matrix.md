# Stratified overlap, two-target, and all-view proposal matrices

Status: stratified/two-target complete; A/B/C/D/E proposal matrices complete

## Stratified current-head study

Manifest: `data/manifests/scannetpp_stratified_triplets_v1.json`

- 12 deterministic triplets: four selector-score quantiles in each of three
  ScanNet++ training scenes.
- For each triplet: exact initialization, full residual head, and directional
  appearance-only head.
- 3,000 steps with cosine LR for trained modes.
- Full-frame and observable-region metrics, rendered coverage, and best
  full/support checkpoints.

Worker outputs: `/code/feedforwardfoam-project/runs/stratified_triplets_v1`

## Two-target geometry test

Manifest: `data/manifests/scannetpp_two_target_geometry_v1.json`

One ideal context pair supervises two held-out targets at camera-segment
interpolations 0.500 and 0.734. Initialization, full residual, and appearance-
only modes test whether learned geometry transfers beyond one memorized target.

Worker outputs: `/code/feedforwardfoam-project/runs/two_target_geometry_v1`

## All-view pixel-aligned proposal study

Every context pixel proposes a world-space Foam cell using the shared decoder.
All proposals are combined before constructing one global Power Foam topology.
The study repeats all 12 stratified triplets with exact initialization and a
3,000-step full residual head.

| Arm | Proposal rule | Final budget |
|---|---|---:|
| A — all | All 6,400 pixels from each of two contexts | 12,800 |
| B — balanced | Uniform 3,200 pixels from each context | 6,400 |
| C — voxel | Start with all 12,800, deterministic world-voxel selection | 6,400 |
| D — fps | Start with all 12,800, farthest-point selection in world space | 6,400 |
| E — confidence voxel | Arm C's grid, highest depth-confidence member per voxel | 6,400 |

This separates additional view-2 coverage (A) from primitive budget (B) and
world-space deduplication/coverage selection (C). Support masks use both context
proposal sets. Each arm retains pixel-aligned decoding; no independently built
foam diagrams are merged.

Worker checkout/output root: `/code/feedforwardfoam-abc`

- A: `runs/proposals_a_all_v1`
- B: `runs/proposals_b_balanced_v1`
- C: `runs/proposals_c_voxel_v1`
- D: `runs/proposals_d_fps_v1`
- E: `runs/proposals_e_confvoxel_v1`

Results and analysis for all five arms:
`docs/experiments/proposal_reduction_arms.md`. Summary: arm A wins 12/12; both
new world-space arms (D, E) fail to improve on arm C, and every world-space
reduction loses ~4 dB to pixel-space uniform striding at the same budget.

CUDA smokes passed with 12,800 / 6,400 / 6,400 active cells respectively.
Full runs use all four A6000 GPUs and are resumable via per-run checkpoints.
