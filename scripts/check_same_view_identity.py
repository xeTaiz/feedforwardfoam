#!/usr/bin/env python3
"""Test same-camera Power Foam identity rendering without VGGT or a decoder.

This deliberately constructs one cell on every renderer ray, copies the target
color using Power Foam's centered spherical-Voronoi color convention, and
inverts the upstream raw-radius softplus. It is a renderer/initialization
contract test, not a learned reconstruction experiment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image

from feedforwardfoam.data.blender import BlenderNvsDataset
from feedforwardfoam.head import FoamParameters
from feedforwardfoam.renderer import (
    PowerFoamRendererBridge,
    camera_from_view,
    pinhole_ray_map_from_view,
    powerfoam_args,
)


def inverse_softplus(value: torch.Tensor, beta: float) -> torch.Tensor:
    """Return x such that softplus(x, beta) equals positive ``value``."""
    scaled = beta * value
    return torch.where(
        scaled > 20.0,
        value,
        torch.log(torch.expm1(scaled)) / beta,
    )


def quaternions_from_positive_x(normals: torch.Tensor) -> torch.Tensor:
    """Return wxyz rotations mapping local +X to unit ``normals``."""
    normal = torch.nn.functional.normalize(normals, dim=-1)
    quaternion = torch.stack(
        [
            1.0 + normal[:, 0],
            torch.zeros_like(normal[:, 0]),
            -normal[:, 2],
            normal[:, 1],
        ],
        dim=-1,
    )
    opposite = normal[:, 0] < -1.0 + 1e-6
    fallback = torch.zeros_like(quaternion)
    fallback[:, 2] = 1.0  # 180 degrees around +Y maps +X to -X.
    quaternion = torch.where(opposite[:, None], fallback, quaternion)
    return torch.nn.functional.normalize(quaternion, dim=-1)


def ray_neighbor_pitch(ray_directions: torch.Tensor, depth: float) -> torch.Tensor:
    """Mean 3D distance to four-connected neighboring rays at fixed distance."""
    height, width = ray_directions.shape[:2]
    pitch = torch.zeros((height, width), device=ray_directions.device)
    counts = torch.zeros_like(pitch)
    dx = torch.linalg.vector_norm(ray_directions[:, 1:] - ray_directions[:, :-1], dim=-1)
    dy = torch.linalg.vector_norm(ray_directions[1:] - ray_directions[:-1], dim=-1)
    pitch[:, :-1] += dx
    pitch[:, 1:] += dx
    counts[:, :-1] += 1
    counts[:, 1:] += 1
    pitch[:-1] += dy
    pitch[1:] += dy
    counts[:-1] += 1
    counts[1:] += 1
    return depth * pitch / counts.clamp_min(1)


def save_image(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    Image.fromarray(array).save(path)


def masked_mse(error_sq: torch.Tensor, mask: torch.Tensor) -> float:
    selected = error_sq[mask]
    return float(selected.mean()) if selected.numel() else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-downsample", type=int, default=5)
    parser.add_argument("--depth", type=float, default=2.0)
    parser.add_argument("--radius-scale", type=float, default=0.8)
    parser.add_argument("--density", type=float, default=10_000.0)
    parser.add_argument(
        "--color-init",
        choices=("source", "gray"),
        default="source",
        help="Initialize centered SV colors from the source or from renderer gray.",
    )
    parser.add_argument("--optimize-color-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/same_view_identity"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA GPU")
    if args.depth <= 0 or args.radius_scale <= 0 or args.density <= 0:
        raise ValueError("depth, radius-scale, and density must be positive")
    if args.optimize_color_steps < 0 or args.learning_rate <= 0:
        raise ValueError("optimize-color-steps must be nonnegative and learning-rate positive")

    dataset = BlenderNvsDataset(
        args.data_root,
        split="train",
        context_views=1,
        target_views=1,
        image_downsample=args.image_downsample,
    )
    frame = dataset.frames[args.frame_index]
    view = dataset._load_view(frame)  # deterministic diagnostic, not training data sampling
    device = torch.device("cuda")
    target = view.image.to(device)
    alpha_target = view.alpha.to(device) if view.alpha is not None else None
    rays = pinhole_ray_map_from_view(view, device)
    directions = rays[..., 3:]
    points = rays[..., :3] + args.depth * directions
    physical_radii = args.radius_scale * ray_neighbor_pitch(directions, args.depth)
    raw_radii = inverse_softplus(physical_radii, beta=100.0)

    height, width = target.shape[:2]
    count = height * width
    num_texel_sites = 8
    sv_dof = 8
    source_centered_rgb = target.reshape(count, 3) - 0.5
    if args.color_init == "source":
        initial_rgb = source_centered_rgb
    else:
        initial_rgb = torch.zeros_like(source_centered_rgb)
    # P0 ties directional/site colors, so optimize one RGB triplet per cell and
    # broadcast it through the full upstream tensor contract. The broadcast also
    # sums gradients over the 64 tied values instead of diluting them.
    rgb_parameter = torch.nn.Parameter(initial_rgb.clone())
    canonical_axes = torch.eye(3, device=device).repeat((sv_dof + 2) // 3, 1)[:sv_dof]
    axes = canonical_axes[None, None].expand(count, num_texel_sites, sv_dof, 3)
    flat_directions = directions.reshape(count, 3)
    parameters = FoamParameters(
        points=points.reshape(count, 3),
        radii=raw_radii.reshape(count),
        quaternions=quaternions_from_positive_x(-flat_directions),
        density=torch.full((count,), args.density, device=device),
        texel_sites=torch.zeros(count, num_texel_sites, 2, device=device),
        texel_sv_axis=axes.reshape(count, num_texel_sites, 3 * sv_dof).contiguous(),
        texel_sv_rgb=rgb_parameter[:, None, None, :]
        .expand(count, num_texel_sites, sv_dof, 3)
        .reshape(count, num_texel_sites, 3 * sv_dof),
        texel_height=torch.zeros(count, num_texel_sites, device=device),
    )
    camera = camera_from_view(view, device)
    bridge = PowerFoamRendererBridge(
        powerfoam_args(num_texel_sites=num_texel_sites, sv_dof=sv_dof), camera
    )
    scene = bridge.build(parameters)
    effective_radii = scene.get_radii().detach()
    trajectory = []
    if args.optimize_color_steps:
        optimizer = torch.optim.Adam([rgb_parameter], lr=args.learning_rate, eps=1e-12)
        for step in range(1, args.optimize_color_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            rendered_step = scene.forward(camera)[0]
            loss = (rendered_step - target).square().mean()
            loss.backward()
            optimizer.step()
            if step == 1 or step % 10 == 0 or step == args.optimize_color_steps:
                step_mse = float(loss.detach())
                record = {
                    "step": step,
                    "mse": step_mse,
                    "psnr": -10.0 * math.log10(max(step_mse, 1e-30)),
                }
                trajectory.append(record)
                print(json.dumps(record, sort_keys=True))

    result = scene.forward(camera)
    rendered, rendered_alpha = result[0], result[1]
    error = rendered - target
    error_sq = error.square()
    mse = float(error_sq.detach().mean())
    metrics: dict[str, float | int] = {
        "cell_count": count,
        "color_init": args.color_init,
        "optimize_color_steps": args.optimize_color_steps,
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-30)),
        "mae": float(error.abs().mean()),
        "max_abs_error": float(error.abs().max()),
        "mean_alpha": float(rendered_alpha.mean()),
        "min_alpha": float(rendered_alpha.min()),
        "mean_desired_physical_radius": float(physical_radii.mean()),
        "mean_effective_scene_radius": float(effective_radii.mean()),
        "max_radius_roundtrip_error": float(
            (effective_radii - physical_radii.reshape(-1)).abs().max()
        ),
    }
    if alpha_target is not None:
        foreground = alpha_target > 0.5
        background = ~foreground
        metrics.update(
            foreground_mse=masked_mse(error_sq, foreground[..., None].expand_as(error_sq)),
            background_mse=masked_mse(error_sq, background[..., None].expand_as(error_sq)),
            target_foreground_fraction=float(foreground.float().mean()),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_image(args.output_dir / "target.png", target)
    save_image(args.output_dir / "rendered.png", rendered)
    save_image(args.output_dir / "alpha.png", rendered_alpha[..., None].expand(-1, -1, 3))
    save_image(args.output_dir / "abs_error.png", error.abs())
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
