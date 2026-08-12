# ScanNet++ P0 multi-view supervision matrix

Status: running on four RTX A6000 GPUs

## Question

Given exactly one ScanNet++ source view, predict one complete Power Foam and supervise that same foam by rendering it into `N ∈ {1,2,4,8}` distinct held-out target cameras. Only the feed-forward head is optimized; VGGT-Ω remains frozen.

## Fixed protocol

- Data: four audited native ScanNet++ undistorted DSLR scenes already staged on `KW60898`.
- Scene split: train `{00a231a370, f9397af4cb, fd361ab85f}`; validation `{ff17657f71}`.
- Resolution: center-crop released 1752×1168 pinhole images to 1168², resize to 80² for the overnight pilot.
- Input: one source view.
- Output: one `FoamParameters` object with 6,400 cells.
- Supervision: arithmetic mean photometric loss over all N target renders before one backward pass. Targets are sampled from the 32 nearest camera centers to the source, avoiding arbitrary low-overlap pairs.
- Density: fixed raw density 10,000 for this first RGB-only test; no alpha loss.
- Geometry/appearance: corrected physical residual head at commit recorded below.
- Validation: four deterministic episodes on the scene-disjoint validation scene every 250 steps.
- Runs: 2,000 steps, seed 17, target counts 1/2/4/8.
- Resume: atomic `latest.pt` stores head, optimizer, RNG, scene/view sampler state, history, and config.
- Launch commit: `2c05152`; data subset 743 MB / 2,558 files on `KW60996`; four runs launched concurrently on GPUs 0–3.

## Configs

- `configs/experiments/p0_scannetpp_mv1.yaml`
- `configs/experiments/p0_scannetpp_mv2.yaml`
- `configs/experiments/p0_scannetpp_mv4.yaml`
- `configs/experiments/p0_scannetpp_mv8.yaml`

## Launch record

- MV1 PID 202596, GPU 0.
- MV2 PID 202598, GPU 1.
- MV4 PID 202600, GPU 2.
- MV8 PID 202604, GPU 3.
- Two-target and eight-target one-step CUDA smokes completed with finite gradients and deterministic validation metrics before launch.
- At the first live check all four processes were healthy; approximate progress after two minutes was 310/180/90/40 steps for MV1/MV2/MV4/MV8 respectively.

## Interpretation limits

This is a four-scene pilot, not a ScanNet++ benchmark. It tests whether additional target-view supervision improves scene-disjoint validation for a one-source P0 head. It does not establish broad indoor generalization, and fixed high density is an initialization ablation rather than a final occupancy model.
