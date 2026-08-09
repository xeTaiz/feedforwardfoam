# Baselines

## Canonical-view 3DGS baseline (`gsplat`)

The frozen-feature, canonical-view Power Foam head has a same-budget
counterpart in `src/feedforwardfoam/gaussian.py`:

| Aspect | Power Foam head (`head.py`) | Gaussian head (`gaussian.py`) |
|---|---|---|
| Backbone | Frozen VGGT-Ω | Frozen VGGT-Ω |
| Canonical anchor grid | 1/8-1/16 patch grid of one source view | identical |
| Top-M budget | deterministic top-M on a head channel | identical |
| Local feature extractor | 5→256→256 ConvNet + GELU | identical |
| Register projection | Linear(1024, 256) | identical |
| Per-patch output channels | 394 (3+1+4+1+1 + 2 × K_t × sv_dof × 3 with K_t=8, sv_dof=8) | 14 (3+3+4+1+3) |
| Renderer | Unmodified Power Foam (`PowerfoamScene` + Warp rasterizer) | `gsplat.rasterization`, `render_mode="RGB+D"`, `rasterize_mode="classic"`, `camera_model="pinhole"`, `sh_degree=None` |
| Camera convention | Blender / OpenGL `TorchCamera` | gsplat OpenCV `viewmats` + `Ks` (helper `view_to_gsplat_camera`) |
| Trainable parameters (`max_cells=1024`) | ~1.03M | ~0.93M |

The parameter-count gap comes entirely from the per-patch output head (foam
emits 8 texel sites × 8 spherical-Voronoi directions × 3 channels × 2 (axis
+ rgb) + per-cell dipole/quat/radius/density/gate, while Gaussian emits 3+3+4+1+3).
At equal primitive budget and identical local CNN, the two heads differ only
in their per-patch decoder and their renderer.

### Why `gsplat == 1.5.3`?

- Pinned PyPI release; the `main` branch of `nerfstudio-project/gsplat` tracks
  v1.6 development (3DGUT, G-SHARP, HiGS, fused bilagrid) -- explicitly out of
  scope for a stable v0 baseline.
- The pre-built wheel index at `docs.gsplat.studio/whl/pt20cu118` targets
  PyTorch 2.0 + CUDA 11.8 only and is incompatible with this project's
  `torch>=2.3`. Use `pip install gsplat` (JIT compile on first import).

### Camera convention notes

The project's `View.c2w` uses Blender / OpenGL conventions (right = +X,
up = +Y, **back = +Z**, camera looks down -Z). gsplat expects OpenCV
conventions (right = +X, down = +Y, **forward = +Z**). The helper
`view_to_gsplat_camera(view, device)` applies the basis change
`flip = diag(1, -1, -1, 1)` to the inverted `c2w` to obtain `w2c` in
gsplat's convention:

```
w2c_opencv = flip @ inv(c2w_blender)
```

`Ks` is built directly from the horizontal FoV:

```
f = 0.5 * width / tan(fov_x / 2)
K = [[f, 0, width/2],
     [0, f, height/2],
     [0, 0, 1]]
```

### Comparison with Depth Anything 3's Gaussian head

DA3 is a useful implementation reference, but it is not this exact baseline.
Its `GSDPT` is a multi-layer DPT decoder over **every input view**, and its
`GaussianAdapter` emits a pixel-aligned set from every view: optional 2D pixel
offset, three scales, a camera-frame quaternion, RGB/SH coefficients, and an
optional depth offset; opacity is predicted separately. It then (i) lifts
points with depth and rays, (ii) rotates quaternions into world coordinates,
and (iii) bounds scales and multiplies them by predicted depth plus an
intrinsics/resolution-derived pixel footprint. It does not fuse duplicates
before emitting Gaussians. Sources: `depth_anything_3/model/{gsdpt.py,gs_adapter.py}`
in the official DA3 repository.

Our P0 baseline intentionally remains canonical-view-only, because P0 Foam is
also canonical-view-only. It already matches DA3's pixel/depth ray-lifting, positive scales,
normalized quaternion, opacity, and RGB appearance at a fixed primitive budget.
Before using it for a publishable comparison, add DA3-style sub-pixel XY
offsets, depth/intrinsics-normalized scale initialization, and explicit
camera-to-world quaternion rotation; this improves conditioning without
changing count or head depth. Use this as a controlled baseline refinement,
not a claim that we reproduced DA3's multi-view DPT architecture.

### What this baseline does **not** do

- It does not implement view-dependent appearance via SH (`sh_degree=None`).
  Adding SH is straightforward but increases the head output dimension.
- It does not implement densification/pruning/adaptive control; the foam and
  Gaussian baselines use the same fixed budget at training time.
- It does not address the dynamic / incremental setting (P2 in the foam
  spec); P2 is shared research for both representations.

### Running the baseline

```bash
uv pip install -e '.[gsplat]'   # CUDA host; JIT compile on first import
fffoam-train \
  --config configs/p0_blender_smoke_gaussian.yaml \
  --data-root data/processed/nerf_synthetic/chair \
  --checkpoint /path/to/VGGT-Omega-1B-512/model.pt \
  # or: --use-stub-backbone for a renderer/head wiring smoke test
```

The integration test `tests/test_gsplat_integration.py` is gated by
`FFFOAM_RUN_CUDA_TESTS=1` and skips otherwise. The CPU-safe unit tests in
`tests/test_gaussian_head.py` verify shape, activation bounds, gradient
propagation, batch-size enforcement, and parameter-count parity on any host.