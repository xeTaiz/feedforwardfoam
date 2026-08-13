# Future directions for the feed-forward Power Foam head

Status: idea backlog; not all items belong in the next experiment

This file preserves potentially useful ideas from DA3, MVSplat, pixelSplat,
Splatt3R, StreamSplat, FlashMono, and Anchor3R. The evidence audit and exact
source links live in `feedforward_splat_training.md`. Items below are hypotheses
to test, not claims that the corresponding Gaussian technique transfers
unchanged to Power Foam.

## Geometry and feature conditioning

### Two-view canonical decoding

Process two calibrated overlapping context images jointly through VGGT-Ω, but
retain one canonical output foam. Reproject or cross-attend view-2 evidence into
view-1 anchors before decoding cells. Candidate fused fields include:

- RGB and dense frozen features from both views;
- predicted depth and confidence from both views;
- canonical and supporting ray directions in world coordinates;
- reprojection validity, depth agreement, and viewing-angle difference;
- number of supporting views and local feature/color variance.

This is the smallest change that introduces actual triangulation evidence
without merging two independently constructed power diagrams.

### Multi-view proposal fusion before cell decoding

Let every context view propose world-space surface samples, then fuse proposals
before they become Power Foam cells:

1. lift each proposal using calibrated rays and aligned depth;
2. cluster with a voxel hash or confidence-aware radius graph;
3. pool features, position, normals, rays, confidence, and visibility;
4. decode exactly one cell per fused anchor;
5. construct one global Čech graph and one global foam.

Track duplicate rate, proposal variance, support-view count, and coverage at a
fixed final cell budget. This borrows DA3's multi-view proposal density while
respecting Power Foam's single-diagram constraint.

### Explicit epipolar or cost-volume evidence

If VGGT-Ω's frozen registers and dense depth do not provide enough local
cross-view evidence, add a small trainable epipolar/cost-volume adapter over
frozen patch features, following MVSplat/pixelSplat. Restrict matching to the
known camera epipolar geometry. Compare against simple reprojection fusion
before adding a large transformer.

### Intermediate backbone features

The current adapter exposes only depth, confidence, and registers. DA3 and
Splatt3R decode from multi-scale dense features. Expose selected frozen
VGGT-Ω aggregator layers or dense-head features, project each to a small common
width, and learn a layer-wise mixture. Keep the backbone frozen initially.
Ablate final-only, four-layer sum, and learned layer weights.

### Depth/camera gauge alignment

VGGT-family depth and pose may occupy a learned scale/gauge while released
ScanNet++ poses use the dataset gauge. Before lifting depth into world space:

- measure scale consistency between predicted depth and calibrated camera
  baselines;
- estimate one robust scene/episode scale from cross-view reprojection or
  available metric depth;
- consider DA3-style Umeyama alignment or Anchor3R/π3-style shared robust scale;
- apply the same scale to depth and camera translation, never depth alone;
- log scale, reprojection error, and failure rate.

A gauge mismatch can produce perfect self-renders and unusable cross-renders,
so this is a hard diagnostic before increasing model capacity.

## Primitive parameterization

### Rendering depth offset around geometry depth

Keep a geometry depth used for auxiliary supervision, then predict a small
separate rendering-depth offset as DA3 does. Bound it by pixel footprint or
local depth uncertainty. This prevents photometric tuning from destroying the
backbone's geometric estimate while allowing the renderer to compensate for
surface/rasterization conventions.

### Sub-pixel ray offsets

Predict bounded XY offsets in normalized pixel coordinates before ray lifting.
DA3 and pixelSplat use this to avoid locking every primitive to a pixel center.
For foam, keep offsets small relative to one canonical pixel and penalize large
or spatially irregular offsets.

### Physical radius initialization

Continue using raw pre-softplus radii, but condition physical radius on:

- depth and calibrated pixel footprint;
- incidence angle and depth gradients;
- confidence and support-view count;
- depth discontinuities, with asymmetric or layered coverage near edges.

Uniform radius inflation already failed. Edge-aware footprint support is more
promising than a global scale.

### Probabilistic or layered depth proposals

Inspired by pixelSplat and StreamSplat, predict 2–3 depth hypotheses only at
low-confidence or discontinuity pixels. Fuse high-confidence single-depth
regions normally. This can represent disocclusion boundaries without
multiplying every source pixel's cell count.

### Learned occupancy and density curriculum

Replace fixed density 10,000 with an occupancy-aware parameterization:

- initialize confident source surfaces opaque and unsupported cells empty;
- warm opacity/density from a conservative prior rather than allowing immediate
  collapse;
- regularize empty-space occupancy and total visible mass;
- supervise density with source/target visibility when geometry permits;
- report alpha, empty fraction, visible-cell count, and density histograms.

pixelSplat's opacity mapping is a useful curriculum pattern, but Foam density
must respect Beer–Lambert opacity and the upstream raw-softplus contract.

### Appearance curriculum

Start with view-independent or heavily damped directional appearance. Only add
full spherical-Voronoi freedom after geometry transfers across cameras:

1. source-centered RGB shared across directions;
2. small directional RGB residuals with canonical axes fixed;
3. learned directional values;
4. learned axes and untied texture sites last.

DA3's damped higher-order SH initialization motivates this. A powerful
view-dependent appearance head can otherwise hide geometry errors.

### Surface detail scheduling

Keep texel positions tied and heights zero during geometry training. Untie
appearance sites first; enable displacement only after held-out geometry and
coverage pass a threshold. Penalize displacement magnitude and local
roughness.

## Objectives and supervision

### Geometry-first curriculum

Use stages rather than one joint objective from initialization:

1. geometry-locked: frozen points/normals, learn stable appearance/radius;
2. bounded geometry: enable small depth/point/normal residuals with context
   depth and reprojection losses;
3. occupancy: enable density with visibility/empty-space supervision;
4. appearance: add delayed LPIPS and directional radiance;
5. optional selective backbone fine-tuning at a very small LR.

This combines DA3's frozen-backbone recipe, Splatt3R's geometry/render staging,
and StreamSplat/FlashMono's head warm-up curricula.

### Context geometry loss

Add confidence-weighted losses that preserve or improve frozen geometry:

- scale/shift- or gauge-aligned depth loss on context views;
- pointmap loss in a shared camera/world frame;
- finite-difference point/normal gradient loss;
- uncertainty/confidence regularization to prevent zero-confidence collapse;
- cross-context depth and reprojection consistency.

Use real depth where trustworthy; otherwise treat frozen VGGT-Ω output as a
regularizing prior, not ground truth.

### Visibility-masked target loss

Following Splatt3R, compute a target mask from projected context geometry and
camera calibration. Optimize a masked visible-region loss, while still report
full-frame metrics. Do not derive the mask from target RGB. Distinguish:

- source-observable target pixels;
- disoccluded/unobserved pixels;
- dynamic or invalid pixels;
- geometric occlusion conflicts.

This prevents impossible disoccluded content from dominating a one- or
two-view reconstruction objective.

### Component and global rendering losses

FlashMono reports that merged-map-only loss encourages primitive shrinkage.
For multi-view proposal fusion, combine:

- a component/source-supported render that verifies each context proposal set;
- the final fused/global foam render into source and target cameras;
- geometry agreement between component predictions and the fused foam.

For P0's single canonical foam there is no merged-map distinction, so do not
add this term until proposal fusion exists.

### Target sampling versus target averaging

Published methods often use one target per optimizer update. Sampling diverse
targets across steps gives similar expectation with lower memory and more
updates. Compare:

- one randomly sampled overlapping target per step;
- two targets per step;
- all available targets only for deterministic validation.

Normalize the objective consistently and compare by renders processed, wall
clock, and optimizer updates—not steps alone.

### Delayed perceptual loss

Use RGB MSE/Charbonnier until cross-camera geometry improves. Introduce LPIPS
later, with correct input normalization, and a small weight. Track whether it
improves perceptual quality without reducing geometric consistency.

### Coverage and topology regularization

Power Foam-specific diagnostics/regularizers should include:

- uncovered target pixels and alpha mass;
- isolated cells and Čech graph degree;
- excessive overlap and radius inflation;
- topology churn across steps;
- visibility count per cell and per context;
- cell support disagreement across views.

Do not differentiate discrete Čech membership; recompute it each forward and
regularize continuous distances/margins.

## Sampling and data

### Overlap-aware episode sampling

Camera-center distance alone is insufficient. Score candidate pairs/triples by:

- relative translation normalized by median scene depth;
- relative optical-axis angle;
- frustum intersection;
- projected depth/point overlap where available;
- static valid-pixel fraction;
- exposure and sharpness consistency.

Train with an overlap curriculum: very high overlap first, then gradually
increase baseline and viewpoint angle. Validation should contain fixed bins of
high, medium, and extrapolative overlap.

### Context ordering and canonical choice

Randomize which context is canonical so the network cannot learn a scene- or
trajectory-position shortcut. Candidate canonical policies:

- first sampled context;
- highest confidence/coverage context;
- medoid camera among contexts;
- random context with symmetric fusion features.

Report sensitivity to canonical choice.

### Data scale and diversity

Three training scenes cannot establish feed-forward generalization. After the
geometry gate passes on a controlled subset, scale toward many scenes with
stable cameras/depth. Keep scene-disjoint validation and a final untouched test
split. Use synthetic data with exact depth/normals for early geometry stages,
then mix real posed data.

### Constant token/render budget

As DA3 varies resolution and view count, hold approximate tokens or rendered
pixels per optimizer update constant. This enables fair 1/2/4-context
comparisons and avoids silently giving low-view arms more updates.

## Architecture and optimization

### Better decoder than a two-layer local CNN

The current head has a shallow `5→256→256` CNN and global register average.
Potential upgrades, after fixing gauge and context evidence:

- multi-scale DPT/FPN decoder over frozen layers;
- local window transformer over canonical anchors;
- cross-attention from canonical anchors to supporting-view tokens;
- graph/set transformer over fused world-space proposals;
- separate geometry, occupancy, and appearance heads with shared trunk.

Do not increase output capacity before the two-view geometry-locked baseline.

### Separate geometry and appearance optimization groups

Use smaller LR and tighter clipping for points/normals/radii, larger LR for
appearance, and staged density activation. Log per-group gradients. Consider
zero weight decay on physically initialized output biases and modest weight
decay on the feature decoder.

### Selective backbone fine-tuning

Only after a frozen-backbone head demonstrates cross-view learning, unfreeze a
small set of late register/cross-view blocks or train low-rank adapters. Use a
much smaller LR than the head and preserve depth/pose performance with auxiliary
losses.

### Sequential residual updater

For future online reconstruction, follow the project's P2 direction:

- render current foam into a new context;
- compare rendered RGB/depth/normal/visibility with the observation;
- predict keep/modify/remove/add actions;
- replay all previous context views to prevent forgetting;
- update directional appearance separately from stable geometry;
- rebuild one replacement foam rather than appending overlapping diagrams.

Anchor3R's transient gauge and image-only cache, StreamSplat's staged static/
dynamic split, and FlashMono's component/global losses are useful references.

## Evaluation gates

Before calling a head successful, require:

1. renderer/camera identity still passes;
2. self-view PSNR does not substitute for NVS;
3. high-overlap cross-view PSNR improves substantially over source-copy and
   black/background baselines;
4. predicted depth/points reproject consistently across contexts;
5. performance persists across scene-disjoint validation;
6. improvements hold at a matched primitive, parameter, render, and wall-clock
   budget against the DA3-style Gaussian baseline;
7. qualitative geometry, alpha, depth, normal, and cell-support visualizations
   agree with the metrics.
