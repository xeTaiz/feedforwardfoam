# Feed-Forward Power Foam

Research harness for predicting one bounded Power Foam scene from frozen VGGT-Ω features and supervising it with held-out novel views.

## Layout

```text
src/feedforwardfoam/  # package: backbone adapter, canonical Foam head, renderer bridge, loaders
configs/              # reproducible experiment configuration
data/raw/             # downloaded source data (ignored)
data/processed/       # converted data (ignored)
data/manifests/       # scene-disjoint split manifests (ignored)
external/             # pinned upstream git submodules (VGGT-Ω, Power Foam)
tests/                # unit and CUDA integration tests
scripts/              # environment/bootstrap commands
specs/                # research specification
```

Research summaries:

- [`docs/experiments/overview.md`](docs/experiments/overview.md) — concise experiments, results, and conclusions;
- [`docs/references/decoder_design_overview.md`](docs/references/decoder_design_overview.md) — decoder/proposal ideas and recommended progression;
- [`docs/HANDOFF.md`](docs/HANDOFF.md) — full handoff document with all recent experiments, results, decisions, environment, and next steps for another agent.

Initialize upstream dependencies after cloning:

```bash
git submodule update --init --recursive
./scripts/bootstrap_upstreams.sh
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[dev]'
```

`bootstrap_upstreams.sh` creates separate environments because the upstream projects currently pin incompatible NumPy versions. The training environment must install Power Foam's CUDA dependencies; see `scripts/bootstrap_powerfoam_env.sh`.

## First experiment (P0)

`configs/p0_blender_smoke.yaml` uses a canonical source view to emit a **single** Power Foam. It trains against held-out overlapping target views; it never merges independently predicted foams. The head retains Power Foam's spherical-Voronoi appearance and ties 2D surface sites with zero displacement.

```bash
fffoam-train --config configs/p0_blender_smoke.yaml \
  --data-root data/processed/nerf_synthetic/chair \
  --checkpoint /path/to/VGGT-Omega-1B-512/model.pt
```

For the renderer/head wiring smoke test before gated weights are available, add
`--use-stub-backbone`; never report that mode as a VGGT-Ω result. The command
requires CUDA and a Power Foam-capable environment.

## Same-budget 3DGS baseline

`docs/baselines.md` documents the canonical-view gsplat baseline
(`src/feedforwardfoam/gaussian.py`, config
`configs/p0_blender_smoke_gaussian.yaml`). It mirrors the foam head's
canonical-view architecture with a 14-channel per-patch output (vs. 394 for
the foam) and renders via `gsplat.rasterization` in `RGB+D` / `classic`
mode. The source-audited training comparison against DA3, MVSplat, pixelSplat,
Splatt3R, StreamSplat, FlashMono, and Anchor3R is in
`docs/references/feedforward_splat_training.md`; the broader future-head idea
backlog is in `docs/references/future_head_directions.md`. Released reference
code is pinned under `external/references/` and is never imported by the project.
Install on a CUDA host with `uv pip install -e '.[gsplat]'` and run it with:

```bash
fffoam-train --representation gaussian \
  --config configs/p0_blender_smoke_gaussian.yaml \
  --data-root data/processed/nerf_synthetic/chair \
  --checkpoint /path/to/VGGT-Omega-1B-512/model.pt
```

## Data recommendation

- **Smoke / renderer-gradient test:** a tiny generated Blender scene or a NeRF-Synthetic-format scene. Its `transforms_*.json` camera format makes held-out NVS deterministic and cheap.
- **First real, controlled training set:** **DTU**, converted to the Power Foam COLMAP layout. It has calibrated static scenes and ground-truth geometry, making it a better first research dataset than ScanNet++.
- **First real indoor generalization set:** **ScanNet++**, after `fffoam-prepare-scannetpp` converts/selects static DSLR sequences into the same manifest schema. It is valuable but considerably larger and noisier operationally.
- Keep Mip-NeRF 360 and Deep Blending as evaluation-only sets; do not consume them for initial training.

## Validation

```bash
pytest -q
```

CUDA integration tests are skipped unless `FFFOAM_RUN_CUDA_TESTS=1`, the Power Foam dependencies are installed, and a CUDA GPU is visible.
