# Feed-Forward Power Foam (FF-PF) — v0 Research Specification

**Status:** draft for discussion  
**Primary goal:** predict a sparse, connected, bounded Power Foam scene directly from a small set of posed, overlapping RGB images, and render held-out overlapping views without per-scene optimization.  
**Initial strategy:** freeze a geometry foundation model; train only a multi-view Power Foam head; selectively fine-tune late backbone layers only after the frozen-head result is established.

---

## 1. Research hypothesis and claim boundary

### Hypothesis
A geometry foundation model can amortize the expensive per-scene optimization of Power Foam by predicting a globally consistent set of bounded power cells with explicit local surfaces. Compared with a conventional pixel-aligned Gaussian head, the result should use fewer primitives and less empty-space support at comparable held-out novel-view quality.

### Core claim to test

> Given 4–12 calibrated, overlapping source views, FF-PF predicts a fixed-budget, bounded Power Foam that has competitive held-out NVS quality and a better quality-versus-live-cell-count (and quality-versus-memory) Pareto frontier than an identical-backbone Gaussian head and a naïve per-pixel foam head.

This is deliberately narrower than claiming that Power Foam is universally better than 3DGS. Power Foam has evidence for **per-scene optimized** reconstruction; this project tests whether its representation is also suitable for **amortized feed-forward** reconstruction.

### Non-goals for v0

- Unposed input images or jointly learned camera estimation.
- Dynamic scenes, editing, relighting, or generative completion.
- Extreme extrapolation outside observed geometry.
- Test-time densification or per-scene gradient optimization in the primary result.
- Full backbone fine-tuning before a frozen-backbone baseline exists.

---

## 2. Terminology and representation correction

Power Foam should not use a Delaunay triangulation as its primary training-time adjacency structure. Delaunay/Voronoi topology is central to **Radiant Foam**. For bounded Power Foam, the intended cheap, exact-for-rendering adjacency superset is the **Čech complex**:

\[
E = \{(i,j) : \|s_i-s_j\| \le r_i+r_j+\epsilon\}.
\]

Here, `s_i` is a power-cell site and `r_i` is its bounding-sphere radius. A weighted α/regular complex is the mathematical dual but is more expensive to construct. The Čech graph includes extra edges whose faces lie outside the cells; these do not alter correct rendering. GPU spatial hashing/collision detection should build it.

The phrase “Czech diagram” in the project discussion is therefore interpreted as **Čech complex**.

### One predicted cell

For a cell \(i\), predict

\[
\Theta_i = \{s_i,r_i,q_i,n_i,\sigma_i,g_i,\{u_{ik},h_{ik},a_{ik}\}_{k=1}^{K_t}\}.
\]

| Symbol | Constraint / shape | Meaning |
|---|---|---|
| \(s_i\) | \(\mathbb{R}^3\) | Power-diagram site in normalized world coordinates. |
| \(r_i\) | `r_min + softplus(raw)` | Bounding-sphere radius / power weight (use a radius internally; convert to squared weight where the renderer requires it). |
| \(q_i\) | `s_i + r_i*tanh(raw_offset)` | Dipole-face center; distinct from the power-cell site. |
| \(n_i\) | normalized \(\mathbb{R}^3\) | Dipole surface orientation. |
| \(\sigma_i\) | positive scalar | Interior density/opacity; exterior half-space is fixed empty. |
| \(g_i\) | \([0,1]\) | Continuous existence/contribution gate; threshold only on export. |
| \(u_{ik}\) | \(\mathbb{R}^2\) | Local 2D soft-Voronoi detail-site coordinate in the dipole tangent frame. |
| \(h_{ik}\) | bounded scalar | Detail-site normal displacement, expressed relative to \(r_i\). |
| \(a_{ik}\) | appearance vector | Per-detail-site diffuse or directional radiance coefficients. |

### Representation schedule: retain the full renderer, constrain the predicted fields

The first experiment should use the **unmodified Power Foam renderer and its full parameter/data contract**, including its spherical-Voronoi appearance function. The staged plan must not mean removing surfaces or replacing the renderer with a simplified one.

`K_t` denotes *tangential detail sites*, whereas spherical-Voronoi appearance uses its own directional axes. `K_t=0` therefore means **zero learned surface displacement/detail**, not “no surface” and not “no directional radiance.” If the repository requires its default `K_t=8` arrays, instantiate all eight sites, fix/tie their UV positions and set all displacement values to zero; do not implement a special K=0 renderer path.

1. **P0 full-foam, low-DoF:** predict sites, radii, dipole centers/normals, density, and spherical-Voronoi appearance from the first training run. Keep all surface-displacement values at zero and tie the 2D detail-site appearance within a cell, so texture sites add no spatial complexity. The renderer remains full Power Foam.
2. **P1 surface detail:** untie the 2D appearance sites and predict bounded displacements, first with four independent sites (or four groups tied across the renderer's eight sites).
3. **P2 full detail:** use eight independent 2D sites, matching the Power Foam ablation-preferred count.
4. **Directional axes:** retain the renderer's spherical-Voronoi axes from P0. Prefer fixed/canonical axes initially and predict their RGB radiance values. Learning the axes themselves is a later ablation, because jointly learning axes, colors, and geometry is poorly conditioned.

The tangent frame should be deterministically built from \(n_i\) in v0, with a robust reference-axis fallback. Do not predict a free quaternion until required by an ablation.

---

## 3. Inputs, coordinate contract, and output budget

### Inputs

- `Nc = 4–12` RGB context images, arbitrary order.
- Calibrated intrinsics and world-to-camera transforms.
- A scene-normalization transform derived from robust predicted point-map/depth percentiles and camera frusta.
- Geometry-backbone outputs: multi-layer image tokens; point/depth/ray maps; confidence; optional normals and camera estimates.

The initial task uses supplied/calibrated cameras even if the backbone is capable of predicting cameras. This prevents pose error from confounding representation evaluation.

### Output

- A set of at most `M` active power cells and their Čech adjacency.
- Initial budgets: `M=8k–16k` for 256–384px development; `M=16k–64k` for 512px indoor scenes. Outdoor budgets are determined only after degree, memory, and quality profiling.
- Rasterize during training. Report ray-tracing quality and runtime separately once the renderer is integrated.

### Canonical scale

Normalize each scene to a bounded world frame before the head. Predict cell radii, dipole offsets, and texel heights in this frame, preferably relative to \(r_i\). Enforce

\[
r_i \in [r_{min},r_{max}], \qquad |h_{ik}| \le \eta r_i.
\]

This avoids a head whose geometry changes meaning with capture scale and stops pathological giant cells.

---

## 4. Backbone policy

The project default is **VGGT-Ω (VGGT-Omega; “VGGT-OHM” is its ASCII transliteration)**, with vanilla VGGT as a fallback. VGGT-Ω is a released successor with a public project page/repository and exposes global register tokens in addition to dense geometry features. Use its checkpoint subject to its gated-weight and research-license terms. The head must remain adapter-isolated so vanilla VGGT or another backbone can be substituted.

### Required adapter interface

The model-specific code must be isolated behind `BackboneAdapter`:

```text
(images, cameras) -> {
  layer_tokens: [L, Nc, Hf, Wf, C],
  point_or_depth: [Nc, H, W, 3 or 1],
  ray_map: optional [Nc, H, W, 6],
  confidence: [Nc, H, W, 1],
  normals: optional [Nc, H, W, 3],
  camera_estimates: optional
}
```

- Learn a small layer-wise fusion/projection over backbone layers; do not assume final-layer tokens are optimal.
- In phase A, all backbone weights and native prediction heads remain frozen.
- Use backbone depth/normal outputs only as confidence-masked auxiliary pseudo-targets unless true labels exist.
- Keep the adapter contract generic enough to compare released VGGT, Pi3X, or another geometry foundation model later.

---

## 5. Head architecture

### 5.1 P0: canonical-view prediction — no fusion required

The easiest valid first experiment is **not** to predict an independent foam from every input view. Choose one canonical source view, use its `1/8` or `1/16` grid as the only set of cell anchors, and decode one scene-level Power Foam from those anchors. VGGT-Ω may still receive multiple context views: its cross-view features and register tokens inform the canonical-view anchors, but no other view emits cells.

```text
posed context views -> frozen VGGT-Ω -> canonical-view token grid + global registers
  -> one cell proposal per selected canonical patch
  -> PowerFoamDecoder -> one global cell set {Theta_i, g_i}
  -> Čech construction -> unmodified Power Foam rasterizer
  -> held-out overlapping target-view loss
```

Each canonical patch predicts a depth/point residual around the detached backbone point map, confidence, then its cell parameters. Use confidence/coverage top-M selection if the grid yields too many cells. This produces **one** power diagram, so it never attempts to merge multiple already-formed foams and does not create cross-view overlapping primitive sets.

P0 should be trained and evaluated source-to-**held-out target** NVS, initially at high overlap. Rendering only the same source view is acceptable as an overfit/renderer smoke test, but it is not a meaningful feed-forward reconstruction result: it does not test geometry or novel-view radiance.

### 5.2 P1: fuse proposals *before* cells exist

Once P0 is stable, allow every context view to contribute geometry proposals, but fuse them before decoding Power Foam primitives:

```text
per-view point proposals -> world-space voxel hash / clusters
  -> one pooled anchor token per 3D region
  -> local set/graph transformer + global VGGT-Ω registers
  -> one decoded Power Foam cell per anchor
```

This is not a merge of cells: it is evidence fusion before a single power diagram is formed. Pool positions, features, ray directions, view count, confidence, and normal statistics. Retain anchors with coverage-aware selection rather than only texture confidence. The decoder then jointly predicts compatible sites/radii/dipoles and an existence gate.

### 5.3 P2: sequential residual fusion (promising later)

The proposed render-and-update idea is a natural online extension. Given a prior foam \(\Theta_{t-1}\) and a new observed posed image \(I_t\), render prior RGB, depth, normal, transmittance, cell ID, and visibility into the new camera. A learned updater consumes these together with VGGT-Ω features and predicts `keep / modify / remove / add` actions:

- low RGB/depth residual and visible support → retain geometry; update directional spherical-Voronoi radiance only if it improves the new-angle appearance;
- depth/visibility mismatch or unexplained image region → propose new local anchors from residual pixels;
- combine prior-cell tokens and new proposals in a local transformer, then emit a **replacement global/local cell set**, not an appended overlapping foam.

At training time, unroll 2–4 updates and replay every prior input view in the rendering loss. Otherwise, updating a directional lobe for the newest camera can destroy the explanation of earlier cameras. This is P2 because state management, cell birth/death, order invariance, and topology churn are meaningful new research problems.

### 5.4 Alternatives to compare

| Design | Value | Risk / decision |
|---|---|---|
| **Canonical-view head** | No fusion or cell merging; valid held-out NVS experiment | P0 recommended starting point; coverage limited by reference view. |
| Proposal fusion before cell decoding | Multi-view ownership without overlapping foams | P1 recommended. |
| Sequential residual updater | Exploits render error/depth and supports progressive acquisition | P2; must replace/revise cells rather than append. |
| Learned 3D query/slot decoder | Very compact global set | May miss thin/occluded surfaces; later comparison. |
| Sparse voxel lifting + sparse 3D network | Natural low-cardinality world-space layout | More systems complexity. |
| Multi-view-track tokens | One token per correspondence track | Excellent if tracks are reliable; matching failure sensitive. |
| Hierarchical split/prune decoder | Variable budget and progressive detail | Strong long-term design. |

The initial comparison is P0 canonical-view Foam versus a same-backbone canonical-view Gaussian head. It should use held-out target views, not only source-view reconstruction.

---

## 6. Rendering and topology implementation requirements

1. Build candidate neighbors via a GPU uniform grid/spatial hash; never use all-pairs sphere checks.
2. Construct Čech edges with the radius-overlap predicate. Edge membership is discrete: treat it as fixed for a forward/backward pass. Refresh each step if affordable; otherwise use a bounded refresh interval and measure sensitivity.
3. Back-propagate through geometry, radii, dipoles, texels, density, and appearance. Do not claim differentiation through topology membership changes.
4. Use a soft overlap margin and bounded radius range during early training to reduce topology churn.
5. Apply Power Foam's connectivity/overlap regularizer locally on candidate pairs, not globally. Global repulsion would destroy surface coverage.
6. Rendering-only Steiner points may regularize traversal in empty regions but must never be predicted cells or carry learned appearance parameters.
7. Require deterministic topology construction for fixed seeds and expose diagnostics: mean degree, isolated fraction, clamped-radii fraction, graph-refresh churn, and renderer failures.

---

## 7. Data and episode protocol

### Dataset phases

| Phase | Data | Purpose |
|---|---|---|
| Synthetic warm-up | Kubric/Blender/Habitat-style static multiview scenes with depth, normals, masks | Make sites/radii/normals/dipoles identifiable and validate parameter recovery. |
| Main real training | DL3DV, ScanNet++-style posed captures, and posed real-estate sequences subject to license/pose-quality review | Broad real photometric NVS supervision. |
| Evaluation only | Mip-NeRF 360 and Deep Blending | Keep scene-disjoint and avoid benchmark contamination. |
| Stress suite | Thin structures, foliage, reflective surfaces, textureless corridors, broad-baseline captures | Diagnose geometry/appearance shortcuts. |

Before committing to any training corpus, validate its license, per-frame pose quality, exposure behavior, static-scene fraction, depth availability, and whether scenes overlap benchmarks.

### Per-episode sampling

For one scene/sequence:

- Sample `Nc=4–12` source views with useful mutual coverage.
- Sample `Nt=1–4` target views not supplied to the backbone or head.
- Require source-union/target surface overlap using depth or point-map visibility estimates.
- Curriculum overlap bands: early `70–95%`, middle `40–80%`, late mixture including `20–60%`; replay short-baseline samples throughout.
- Reject adjacent near-duplicate target frames, invalid camera estimates, large exposure jumps, and dynamic/invalid regions where masks are available.
- Split **by scene**, never by frames, capture trajectory, or near-duplicate assets.

This makes held-out NVS a genuine novel-camera reconstruction objective rather than an image-copying shortcut.

---

## 8. Training objective and curriculum

### Objective

\[
\mathcal{L} =
\lambda_{rgb}\mathcal{L}_{rgb}+
\lambda_{ssim}\mathcal{L}_{ssim}+
\lambda_{perc}\mathcal{L}_{LPIPS}+
\lambda_{depth}\mathcal{L}_{depth}+
\lambda_{normal}\mathcal{L}_{normal}+
\lambda_{mv}\mathcal{L}_{mv}+
\lambda_{sparse}\mathcal{L}_{sparse}+
\lambda_{connect}\mathcal{L}_{connect}+
\lambda_{budget}\mathcal{L}_{budget}+
\lambda_{dedup}\mathcal{L}_{dedup}.
\]

| Term | Definition / use |
|---|---|
| `L_rgb` | Robust L1/Charbonnier rendered-target RGB loss; active from start. |
| `L_ssim` | Image-structure loss; active from start at modest weight. |
| `L_LPIPS` | Add after low-frequency geometry stabilizes; prevent it from dominating topology. |
| `L_depth`, `L_normal` | Measured labels when available; otherwise detached, confidence-masked backbone pseudo-targets. Must be ablated because the backbone can be biased. |
| `L_mv` | Reprojection consistency of rendered depth/normal/color where source and target pixels are mutually visible. |
| `L_sparse` | Contribution-weighted sparsity/opacity regularizer adapted from Power Foam. |
| `L_connect` | Local sphere-overlap/connectivity loss on Čech candidate pairs, with the Power Foam decay schedule adapted after profiling. |
| `L_budget` | Penalize expected count, e.g. `(sum(g_i)-M_target)^2`; use annealing to avoid early collapse. |
| `L_dedup` | Penalize multiple active cells explaining the same confident proposal region; test only after fusion baseline. |

Use a normal-facing regularizer only after confirming its convention against camera rays and observed surfaces. The v0 architecture should delay high-capacity directional appearance because it can hide erroneous geometry photometrically.

### Curriculum

1. **Synthetic / high overlap / full-renderer P0:** frozen backbone; predict sites/radii/dipoles/density and spherical-Voronoi radiance from the outset; fix detail displacement to zero and tie 2D texture sites; RGB+SSIM+measured geometry losses.
2. **Real indoor geometry:** activate learned radii, dipole offset/normal, sparsity/connectivity, and confidence-masked pseudo geometry losses.
3. **Surface detail:** untie 2D sites and activate bounded displacement, first with four independent groups, then eight; retain geometry metrics and exposure controls.
4. **Wider coverage:** broaden baseline/overlap curriculum with short-baseline replay.
5. **Fusion:** compare P1 proposal fusion against P0, then test P2 sequential residual updates only after P1 works.
6. **Selective E2E:** unfreeze layer fusion and only final backbone blocks at 10–100x lower LR than the head. Preserve frozen-backbone auxiliaries. Full fine-tuning is a separately reported final ablation.

---

## 9. Baselines and evaluation

### Required baselines

At matched source/target views, resolution, and active-primitive budget:

1. Identical frozen geometry backbone + canonical-view direct per-pixel/patch Gaussian head.
2. **P0:** identical frozen backbone + canonical-view full-renderer Power Foam head (spherical Voronoi retained; zero displacement/tied texture sites).
3. **P1:** proposal-fusion-before-decoding Power Foam, with the same full renderer.
4. Full independent 2D detail sites/displacements (`K_t=4/8`).
5. **P2:** sequential residual updater, evaluated on progressive input sequences.
6. Learned-query cell head at the same budget (later comparison).
7. Optional Power Foam per-scene optimization upper bound, reported separately from feed-forward runtime.
8. Frozen versus selective end-to-end tuning.

### Metrics

| Category | Metrics |
|---|---|
| NVS quality | PSNR, SSIM, LPIPS on held-out targets. |
| Geometry | Depth AbsRel/RMSE, normal angular error; Chamfer/F-score only where reliable geometry ground truth exists. |
| Efficiency | Active cells, bytes/scene, parameters/cell, cells per visible surface area, empty-space occupancy, duplicate-site rate. |
| Topology | Mean/p95 Čech degree, isolated-cell fraction, radius-bound hits, adjacency churn, graph-build time/failure rate. |
| Runtime | Backbone, proposal fusion, cell decoder, graph build, raster render, ray render individually; peak memory. |
| Robustness | Stratify by input count, source-target overlap, baseline, indoor/outdoor, view ordering, and scene scale. |

Report both equal-cell-count and equal-memory comparisons. The project claim is not credible if it only compares a sparse foam to an unconstrained dense Gaussian model.

### Primary success gate

The first publishable milestone is not peak PSNR. At an equal NVS-quality operating point, show that 3D-fused FF-PF has materially fewer live cells / less empty-space support than direct per-pixel Power Foam and the same-backbone Gaussian baseline **without** worse geometry or topology stability.

---

## 10. Milestones and go/no-go gates

| Milestone | Deliverable | Gate |
|---|---|---|
| M0: renderer oracle | Small-scene per-scene Power Foam fit; raster/ray parity, export/import, Čech diagnostics | Credible optimized-foam upper bound and deterministic graph construction. |
| M1: canonical-view adapter | Frozen VGGT-Ω adapter and calibrated canonical proposal lifting | One predicted foam renders held-out overlapping targets without merging multiple foams. |
| M2: frozen full-foam P0 | Fixed-budget canonical-view head using the unmodified renderer, spherical Voronoi, zero displacement/tied texture sites | Better than same-backbone Gaussian at matched active budget, with no renderer/topology collapse. |
| M3: proposal fusion P1 | Fuse multi-view proposals before one cell set is decoded | Fewer duplicate proposals and better target NVS/coverage than P0 at equal cell budget. |
| M4: surface detail | Independent 2D sites/displacements (`K_t=4/8`) and full directional appearance | Better quality/cell without geometry/topology regression. |
| M5: residual updater and selective E2E | Sequential render-residual updater, then late-block backbone fine-tuning | Improvements justify state/topology complexity and do not forget prior observed views. |

A failed milestone should generate a diagnosis/ablation, not automatically trigger a larger model or full fine-tuning.

---

## 11. Principal risks and mitigations

| Risk | Mitigation |
|---|---|
| VGGT-Ω checkpoint/license availability | Use the released model subject to gated-access and research-license terms; retain a generic adapter plus vanilla-VGGT fallback. |
| NVS-only objective permits photometric geometry cheating | Hold out targets, delay directional appearance, add trusted/confidence-masked geometry and reprojection metrics. |
| Per-pixel emissions duplicate sites | Fuse in 3D before decoding cells; report duplication metrics. |
| Radius/site predictions cause topology churn | Bound radii, local spatial hash, refresh diagnostics, soft margins, connectivity loss. |
| Fixed budget erases thin structures | Stratify stress suite; compare hierarchical splitting only after baseline; use coverage-aware anchor selection. |
| Pseudo depth/normal biases geometry | Confidence mask, decouple pseudo-label gradients, compare with/without them and true-label synthetic tests. |
| Čech degree/memory grows unexpectedly | Set radius bounds, measure degree distribution, use adaptive budgets and local neighbor caps only if correctness remains verified. |
| Foundation model already encodes a strong scene prior but fails in domain | Scene-disjoint evaluation and domain stratification; retain direct 3DGS baseline. |

---

## 12. Decisions required before implementation

1. Confirm the VGGT-Ω checkpoint/license and input/output API for v0; it is the default backbone, with vanilla VGGT retained only as fallback.
2. Is the v0 target **posed static NVS**, as specified, or must unposed input be part of the first result?
3. What compute envelope permits: image resolution, number of views, initial `M`, and rasterizer batch size?
4. Which datasets can be used legally and are locally accessible? Establish scene-disjoint training/eval manifests before model work.
5. Is the benchmark objective fast scene creation, compression/quality per primitive, or final raster/ray FPS? This determines the primary Pareto metric.
6. Does an existing Power Foam renderer already expose batched scenes and differentiable rasterization? M0 must answer this before head development.

---

## References / design inputs

- Power Foam project/paper: bounded power cells, dipoles, soft-Voronoi surface detail, spherical directional appearance, Čech adjacency, connectivity/sparsity losses. https://arxiv.org/abs/2604.24994
- Radiant Foam: Voronoi/Delaunay predecessor and differentiable traversal context. https://arxiv.org/abs/2502.01157
- VGGT: Visual Geometry Grounded Transformer. https://arxiv.org/abs/2503.11651 ; https://github.com/facebookresearch/vggt
- pixelSplat: pixel-aligned feed-forward 3DGS. https://arxiv.org/abs/2312.12337
- MVSplat: geometry-aware feed-forward multi-view splatting. https://arxiv.org/abs/2403.14627
- DepthSplat: depth-prior feed-forward splatting. https://arxiv.org/abs/2410.13862
- NoPoSplat: pose-free feed-forward splatting reference for a later extension. https://arxiv.org/abs/2410.17958
- DL3DV: large-scale posed multi-view data candidate. https://dl3dv-10k.github.io/
- Mip-NeRF 360 evaluation benchmark. https://jonbarron.info/mipnerf360/
