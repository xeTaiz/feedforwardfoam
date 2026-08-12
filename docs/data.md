# Data policy and loader contracts

## Recommended order

1. **`tests/` generated images:** validates loader, canonical camera conversion, and no context/target leakage without downloading data.
2. **Blender / NeRF-Synthetic format:** first CUDA renderer-gradient smoke run. It is small, calibrated, static, and has native held-out frames.
3. **DTU:** first controlled real training/evaluation corpus. Convert each scan to the same COLMAP or manifest convention. DTU's static calibrated views and known geometry make regressions diagnosable.
4. **ScanNet++:** first real indoor generalization corpus, not the first debugging dataset. Its high-quality DSLR imagery is useful, but data access, sequence selection, exposure, dynamic content, and pose auditing are operationally heavier.
5. **DL3DV:** scale-up only after the frozen P0 path works. Preserve it as an out-of-domain test split until a train/validation manifest is explicitly approved.
6. **Mip-NeRF 360 / Deep Blending:** evaluation-only.

## Scene-disjoint split rule

A scene ID is assigned to exactly one of train, validation, or test. Never partition individual frames or trajectories from a scene across splits. Context and target views must be disjoint inside every episode.

## ScanNet++ processed manifest

`ScanNetPPDataset` intentionally does not infer camera provenance from raw files. After selecting a static, calibrated DSLR sequence, create:

```text
data/processed/scannetpp/<scene>/
  images/...png
  fffoam_views.json
```

```json
{
  "scene_id": "scene-name",
  "train": [
    {"image": "images/0001.png", "c2w": [[...], [...], [...], [...]], "fov_x_radians": 0.9}
  ],
  "test": [
    {"image": "images/0017.png", "c2w": [[...], [...], [...], [...]], "fov_x_radians": 0.9}
  ]
}
```

Audit the pose reprojection, static mask, resolution, and exposure before producing this manifest. The input pose uses camera-to-world matrices in the same normalised coordinate system used by the experiment.

### Native audited DSLR layout

The loader also accepts ScanNet++'s released undistorted DSLR product directly:

```text
<root>/<scene>/dslr/
  nerfstudio/transforms_undistorted.json
  resized_undistorted_images/*.JPG
```

It hard-rejects anything except centered `PINHOLE` calibration with approximately square pixels. The released 1752×1168 images are center-cropped to 1168×1168 and then resized, with the corresponding horizontal FoV recomputed from `fl_x`. This avoids pretending that arbitrary principal points or distorted images fit the project's current `View.fov_x_radians` camera contract.

The first multi-scene P0 manifest is `data/manifests/scannetpp_p0_4scene.json`: three train scenes and one scene-disjoint validation scene. `scripts/prepare_scannetpp_subset.py` copies only the required DSLR metadata and undistorted images into an experiment staging root.
