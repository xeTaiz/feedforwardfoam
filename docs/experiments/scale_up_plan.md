# Parked plan: longer training and full-dataset episode sampling

Status: **parked** — recorded during the merging-strategy phase, not started.
Owner note: dataset staging is being handled outside this document.

## 1. Why this is parked

Merging/reduction strategy for multi-context proposals is being resolved first on
fixed triplets and few-scene tests. The scale-up below should only start once a
reduction arm is selected, because the sampling manifest and the reduction arm
both change what a long run measures.

## 2. Compute inventory (measured 2026-08-18)

| Worker | GPUs | Notes |
|---|---|---|
| `KW60996` | 4× RTX A6000 48 GB | idle at inspection; existing project checkout `/code/feedforwardfoam-project`, plus `/code/feedforwardfoam-abc` |
| `KW60995` | 3× RTX A6000 48 GB | ~15 GB/GPU already in use by other work |
| `KW60898` | 1× RTX 6000 Ada 48 GB | shared multi-tenant box; **holds the ScanNet++ data**; exec calls were failing/timing out at inspection |
| `gpu210-02` | 1× Tesla V100 32 GB | **no data paths bound** |
| `KW61627` | 2× RTX PRO 6000 Blackwell 96 GB | busy with unrelated DRRT work |
| `KW61633` | 1× RTX A2000 12 GB | too small |
| `archdome` | 1× RTX 3090 24 GB | offline at inspection |

There is **no multi-V100 machine with the full dataset**. The only V100 is
`gpu210-02` and it has nothing mounted.

## 3. Dataset locations (partially verified)

On `KW60898`:

- `/data_ibex/scannetpp_pf` — 395 entries; consistent with a full ScanNet++ scene
  set. **Per-scene structure and total size unverified** (exec calls to that
  mount failed repeatedly during inspection).
- `/data_local/scannetpp_preproc` — contains `train/`, `val/`, `test/`.
  Split contents unverified.
- `/data_local/scannetpp` — 3 entries, 4.9 GB. Small subset only.

Current staged pilot data (4 scenes, 743 MB) lives on `KW60996` at
`/code/feedforwardfoam-project/data/staged/scannetpp_p0`.

Transfer path when needed: `wh_dispatch data_copy` from `KW60898` to `KW60996`.

## 4. Episode sampling — required calibration before any manifest is built

The two quantities that best predicted support PSNR in the 4-triplet study were
target-observable fraction (`r = +0.95`) and maximum view angle (`r = -0.82`).

**Constraint that blocks a naive full-dataset scan:** target-observable fraction
is *not* a pose-only quantity. It is defined by projecting VGGT depth anchors
into the target and dilating by two target pixels
(`docs/experiments/scannetpp_fixed_triplet_overfit.md`). It cannot be derived
from camera poses alone, so `select_scannetpp_triplet.py`'s geometry math cannot
produce it.

Two options, both requiring work before a 395-scene scan:

1. **Depth-based obs, run for real.** Requires a VGGT forward pass per candidate
   triple — expensive, but exact and directly comparable to recorded numbers.
2. **Pose-only frustum-overlap proxy.** Cheap, but the existing
   `obs < 0.6` / `angle > 30°` reject thresholds are **not transferable** to it;
   they were derived from the depth-based metric.

**Prerequisite for option 2:** compute the proxy on the 12 stratified triplets
(and the 4 earlier random/top triplets) where depth-based `obs` is already
recorded, check the proxy actually correlates with the recorded `obs`, and derive
the reject cut from that calibration. Only then scan the full dataset.

Also note: the `q05…q90` bins in the stratified manifest are **selector-score**
quantiles, not overlap quantiles. Measured `obs` is non-monotonic in `q`
(`f939`: q05 = 0.99, q90 = 0.80; `fd`: q05 = 0.85, q90 = 0.93). Do not treat the
`q` axis as an overlap axis, and do not derive a sampling policy from it. Bin on
measured quantities directly.

## 5. Planned long-run shape (to be revised after the reduction arm is chosen)

- Scene-disjoint split across the largest verified scene set.
- Overlap-aware sampling from the calibrated manifest: reject near-empty coverage
  and near-degenerate parallax; balance the remaining bins.
- Reduction arm: whichever arm wins the current single-scene/few-scene study;
  arm A (all-proposal concatenation, no reduction) is the fallback default since
  it currently leads.
- Retain `best_full.pt` / `best_support.pt`; cosine LR.

## 6. Scale-up readiness checks still outstanding

- Depth-gauge scale logging, including a count of runs hitting the `[0.25, 4]`
  bounds.
- Fixed validation bins by measured overlap / angle / coverage.
- Visualization bundles for best checkpoints.
- Registers-only two-context control on the stratified set. Section 5.7 tabulates
  only `initialization` / `appearance` / `full`; the registers-only control named
  in the decision gate is not yet in that table.

## 7. Deferred

- **Matched gsplat baseline.** Deferred by decision; rough reference numbers to
  be taken from published pixel-aligned Gaussian tables (pixelSplat, MVSplat)
  rather than a matched in-house run for now. A matched run is still required
  before any fairness claim is published.
