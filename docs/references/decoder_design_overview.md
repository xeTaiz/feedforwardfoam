# Decoder and proposal-design ideas

This is the short design index. Detailed literature notes and older formulations
remain in:

- `feedforward_splat_training.md` — released Gaussian methods and training ladder;
- `future_head_directions.md` — full backlog of geometry, loss, and optimizer ideas;
- `../../specs/FF-POWER-FOAM-SPEC-v0.md` — project constraints.

The invariant is always: **combine evidence before topology, then construct one
global Power Foam**. We must not merge independently constructed power diagrams.

## Short list

| Idea | Simple explanation | When |
|---|---|---|
| Overlap-stratified sampling | Prefer context/target tuples with enough shared view, while retaining difficulty bins | Now |
| Pixel-aligned proposals from both contexts | Let every source pixel propose a world point and attributes | Strong next candidate |
| Budgeted proposal selection | Reduce all source proposals to one fixed cell budget using confidence, FPS, or voxels | With multi-view proposals |
| Neighbor-derived radius + learned scale | Initialize each radius from local 3-D spacing; predict only how strongly it expands/contracts | With proposal selection |
| Projected feature fusion | Sample view-2 RGB/depth/features at view-1 surface points | Already tested; retain as evidence channel |
| Residual physical head | Predict bounded changes to points, normals, footprint radii, and source appearance | Current stable default |
| Absolute-attribute head | Keep depth-derived positions but predict radius/orientation/appearance directly | Proven overfit ablation |
| Layered depth/occupancy | Allow multiple surfaces or suppress unreliable proposals | Later, after two-view proposals |
| Learned latent/non-pixel queries | Decode cells from free 3-D queries rather than pixels | Later; larger architectural change |
| Sequential render-and-correct | Render an intermediate foam and update it from residuals | Later |

## Recommended progression

### A. Preserve the proven pixel-aligned decoder first

Feed-forward Gaussian systems commonly emit one primitive per source pixel or
feature location. The closest Foam adaptation is:

1. predict a point and feature for every pixel in each context;
2. transform every point into the calibrated world frame;
3. combine the proposal sets;
4. reduce them to a fixed budget;
5. decode attributes for the retained proposals;
6. build one Foam adjacency/topology over the final set.

This changes proposal aggregation without simultaneously abandoning the
well-tested pixel-aligned prediction pattern. It is lower risk than immediately
moving to free learned 3-D queries.

### B. Use both contexts as proposal sources

The current two-context head anchors all cells to view 0. View 2 contributes
features but cannot create a cell in a region absent from view 0. This remains
true with two contexts and one target: view-2-only target content has evidence,
but no primitive carrier.

A minimal extension is to concatenate world-space proposals from both contexts.
Unlike Gaussian concatenation, Foam must deduplicate/select them **before** one
Čech graph is built.

### C. Add budgeted world-space selection

Candidate strategies, ordered from simplest to most architectural:

1. **Confidence top-k:** cheap but may cluster proposals and miss coverage.
2. **Farthest-point sampling (FPS):** encourages spatial coverage but can retain
   outliers and ignores uncertainty.
3. **Voxel pooling:** deduplicates nearby surfaces and naturally exposes local
   support count/density; resolution must adapt to scene scale.
4. **Weighted clustering/soft assignment:** fuses features and uncertainty but
   adds complexity and may blur depth edges.
5. **Learned selection:** highest flexibility, but should follow a reliable
   geometric baseline.

A useful hybrid is confidence filtering followed by FPS or voxel pooling.
Always log rejected mass, duplicate rate, support count, and per-view retention.

## Detailed formulation: fused world-space proposals

For context `v` and pixel `i`, unproject depth into a world candidate

`p_vi = C_v + d_vi r_vi`,

with feature

`z_vi = [backbone feature, RGB, confidence, ray direction, depth uncertainty,
view ID]`.

Collect all candidates `P = {(p_vi, z_vi)}`. Select or cluster them into at most
`M` fused proposals. For a cluster `S_j`, use confidence weights `w_vi`:

`p_j = sum(w_vi p_vi) / sum(w_vi)`

`z_j = Pool({z_vi}, {w_vi})`.

Pool should retain more than the mean:

- weighted mean and variance;
- number of supporting views/proposals;
- minimum and median depth confidence;
- view-direction spread;
- local covariance eigenvalues;
- provenance or canonical-view embedding.

The decoder then receives `[z_j, positional_encoding(p_j), local_spacing_j]`
and predicts one cell. Only after all `M` cells exist do we build the global
Power Foam topology.

## Detailed formulation: neighbor-derived radius

An absolute radius is hard to predict because the required support depends on
which neighboring proposals survive selection. Compute the base radius **after**
selection from local spacing:

`r0_j = eta * median_{k in KNN(j)} ||p_j - p_k||`.

Then predict a bounded relative scale:

`r_j = r0_j * exp(lambda * tanh(s_j))`.

Possible decoder inputs are nearest-neighbor distance, median KNN distance,
local point count, voxel occupancy, and covariance eigenvalues. This lets the
model decide how strongly cells compete for space without learning scene-scale
radius from scratch. At depth discontinuities, use same-surface/confidence
neighbors or robust lower quantiles so foreground cells are not inflated toward
background surfaces.

## Detailed formulation: voxel selection

Choose a scene-normalized voxel width `h`. For voxel `q`, aggregate candidates
whose `floor(p/h)=q`. Retain one or several proposals depending on depth/normal
multimodality. A fixed budget can be enforced by ranking occupied voxels using:

`score_q = confidence_q + alpha * support_views_q + beta * novelty_q`.

If occupied voxels exceed `M`, select top-scoring voxels or FPS their centroids.
If fewer than `M`, permit multiple depth/normal modes in high-variance voxels.
This preserves a deterministic one-foam budget while exposing density and
neighbor spacing to the decoder.

## Pixel-aligned versus non-pixel-aligned decoding

### Pixel-aligned advantages

- closest to successful feed-forward Gaussian systems;
- direct RGB/depth/feature correspondence;
- straightforward physical initialization;
- easy per-view confidence and visibility reasoning.

### Pixel-aligned limitations

- duplicates the same surface across contexts;
- allocates budget according to image sampling rather than 3-D need;
- view-0-only anchors cannot cover view-2-only regions;
- radius depends on the post-selection neighborhood.

### Non-pixel-aligned advantages

- budget follows 3-D coverage rather than image pixels;
- natural multi-view deduplication;
- proposals can represent disocclusions and nonuniform density.

### Non-pixel-aligned risks

- changes representation and decoder simultaneously;
- loses direct source-color correspondence;
- requires positional encoding, neighborhood features, selection, and new
  initialization rules;
- harder to compare fairly with proven Gaussian methods.

**Recommendation:** first use pixel-aligned proposals from **both** contexts plus
geometric budget selection. Treat free learned queries as a later ablation.

## Appearance and geometry terminology

In the current experiment, “RGB-only” was shorthand and is better called
**directional appearance-only**:

- fixed: points, footprint radius, orientation, density, texel sites/heights;
- learned: spherical-Voronoi world axes and RGB through the shared decoder.

A broader appearance-only model could additionally learn texel positions,
heights, density/opacity, and directional temperature. Geometry-versus-
appearance tests matter because target RGB can improve through appearance
without producing transferable 3-D geometry. We do not need a large sweep, but
hard-parallax validation should check whether geometry-enabled heads beat this
appearance-only control.

## Sampling study

Before full scale-up, use 12–20 fixed triplets stratified by:

- target observable fraction;
- context-context and context-target angle;
- normalized baseline;
- target offset from the context segment;
- cross-context depth-consistent support;
- scene type.

For each triplet, record initialization, trained full/support PSNR, alpha,
coverage, and gain over initialization. Use the result to define training bins,
not one global nearest-camera rule. A practical sampler should reject near-zero
support, then balance easy, medium, and difficult overlap bins.

## Safeguards for scaled training

- fixed scene-disjoint validation episodes and overlap bins;
- exact initialization baseline for every evaluation episode;
- full-frame and observable-region metrics together;
- projected proposal support, rendered alpha coverage, and their IoU;
- best full/support checkpoints plus resume state;
- finite-gradient and depth-scale-bound alarms;
- periodic context/target/prediction/error/coverage contact sheets;
- primitive count, radius, support-count, duplicate, topology, runtime, and
  memory statistics;
- matched Gaussian control at the same primitive/render/wall-clock budget.
