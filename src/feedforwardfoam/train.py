"""P0 frozen-feature, image-level Power Foam training entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from .backbone import FrozenGeometryStub, FrozenVGGTOmega
from .data.blender import BlenderNvsDataset
from .gaussian import CanonicalGaussianHead, GaussianRendererBridge
from .head import CanonicalPowerFoamHead
from .renderer import (
    PowerFoamRendererBridge,
    camera_from_view,
    pinhole_ray_map_from_view,
    powerfoam_args,
)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _context_tensor(episode, device: torch.device) -> torch.Tensor:
    return torch.stack([v.image.permute(2, 0, 1) for v in episode.context])[None].to(device)


def _charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


def _validate(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(rendered.clamp(0, 1), target).item()
    return {"mse": mse, "psnr": -10.0 * torch.log10(torch.tensor(mse + 1e-10)).item()}


def train(
    config: dict[str, Any],
    data_root: Path,
    checkpoint: Path | None,
    use_stub: bool,
    representation: str,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("P0 training requires CUDA because the Power Foam renderer uses Warp kernels")
    device = torch.device("cuda")
    torch.manual_seed(int(config["seed"]))
    data_cfg = config["data"]
    dataset = BlenderNvsDataset(
        data_root,
        split="train",
        context_views=int(data_cfg["context_views"]),
        target_views=int(data_cfg["target_views"]),
        image_downsample=int(data_cfg["image_downsample"]),
        seed=int(config["seed"]),
    )
    if use_stub:
        backbone = FrozenGeometryStub().to(device)
        register_dim = backbone.register_dim
    else:
        if checkpoint is None:
            raise ValueError("VGGT-Ω checkpoint is required; pass --checkpoint or --use-stub-backbone")
        backbone = FrozenVGGTOmega(checkpoint).to(device)
        register_dim = backbone.register_dim
    backbone.eval()

    head_cfg = config["head"]
    if representation == "foam":
        head = CanonicalPowerFoamHead(
            register_dim=register_dim,
            hidden_dim=int(head_cfg["hidden_dim"]),
            max_cells=int(head_cfg["max_cells"]),
            num_texel_sites=int(head_cfg["num_texel_sites"]),
            spherical_voronoi_dof=int(head_cfg["spherical_voronoi_dof"]),
            radius_mode=str(head_cfg.get("radius_mode", "learned_absolute")),
            radius_scale_init=float(head_cfg.get("radius_scale_init", 1.5)),
            radius_residual_log_scale=float(
                head_cfg.get("radius_residual_log_scale", 0.25)
            ),
            density_mode=str(head_cfg.get("density_mode", "learned")),
            fixed_density=float(head_cfg.get("fixed_density", 100.0)),
            initialize_rgb_from_image=bool(head_cfg.get("initialize_rgb_from_image", False)),
            initialize_normals_from_depth=bool(
                head_cfg.get("initialize_normals_from_depth", True)
            ),
            point_residual_scale=float(head_cfg.get("point_residual_scale", 0.05)),
            normal_residual_radians=float(head_cfg.get("normal_residual_radians", 0.25)),
            rgb_residual_scale=float(head_cfg.get("rgb_residual_scale", 0.5)),
        ).to(device)
    elif representation == "gaussian":
        head = CanonicalGaussianHead(
            register_dim=register_dim,
            hidden_dim=int(head_cfg["hidden_dim"]),
            max_cells=int(head_cfg["max_cells"]),
        ).to(device)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(config["train"]["learning_rate"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))

    initial_episode = dataset[0]
    if representation == "foam":
        reference_camera = camera_from_view(initial_episode.context[0], device)
        args = powerfoam_args(
            num_texel_sites=int(head_cfg["num_texel_sites"]),
            sv_dof=int(head_cfg["spherical_voronoi_dof"]),
            bkgd_color=tuple(config["renderer"]["bkgd_color"]),
            is_pinhole=bool(config["renderer"]["is_pinhole"]),
        )
        bridge = PowerFoamRendererBridge(args, reference_camera)
    else:
        baseline = config["baseline"]
        bridge = GaussianRendererBridge(
            bkgd_color=tuple(baseline["bkgd_color"]),
            eps2d=float(baseline["eps2d"]),
            radius_clip=float(baseline["radius_clip"]),
            tile_size=int(baseline["tile_size"]),
            rasterize_mode=str(baseline["rasterize_mode"]),
        )

    history: list[dict[str, float]] = []
    # A fixed held-out pair is required for the first per-scene convergence
    # experiment. Later multi-pair training can opt into stochastic resampling.
    resample_episodes = bool(config["train"].get("resample_episodes", False))
    fixed_episode = initial_episode
    for step in range(1, int(config["train"]["steps"]) + 1):
        episode = dataset[step % len(dataset)] if resample_episodes else fixed_episode
        images = _context_tensor(episode, device)
        expected_resolution = config["backbone"].get("image_resolution")
        if expected_resolution is not None and images.shape[-2:] != (expected_resolution, expected_resolution):
            raise ValueError(
                "Context image resolution does not match backbone.image_resolution: "
                f"got {tuple(images.shape[-2:])}, expected {(expected_resolution, expected_resolution)}"
            )
        with torch.inference_mode():
            features = backbone(images)
        ray_map = pinhole_ray_map_from_view(episode.context[0], device)
        if representation == "foam":
            params = head(images, features, ray_map, episode.context[0].alpha)
        else:
            params = head(images, features, ray_map)
        target_view = (
            episode.context[0]
            if bool(config["train"].get("target_from_context", False))
            else episode.target[0]
        )
        target = target_view.image.to(device)
        if representation == "foam":
            render_output = bridge.render(params, camera_from_view(target_view, device))
        else:
            render_output = bridge.render(params, target_view)
        rendered = render_output.rgb
        rgb_loss = _charbonnier(rendered, target)
        alpha_loss = torch.zeros((), device=device)
        alpha_weight = float(config["train"].get("alpha_loss_weight", 0.0))
        if alpha_weight > 0:
            if target_view.alpha is None:
                raise ValueError("alpha_loss_weight requires target alpha from the dataset")
            alpha_loss = F.l1_loss(render_output.alpha, target_view.alpha.to(device))
        loss = rgb_loss + alpha_weight * alpha_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        record = {
            "step": float(step),
            "loss": float(loss.detach()),
            "rgb_loss": float(rgb_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "grad_norm": float(grad_norm),
            "active_cells": float(params.points.shape[0] if representation == "foam" else params.means.shape[0]),
            "render_rgb_mean": float(rendered.detach().mean()),
            "render_alpha_mean": float(render_output.alpha.detach().mean()),
            "mean_radius": float(
                params.radii.detach().mean() if representation == "foam" else params.scales.detach().mean()
            ),
        }
        if step % int(config["train"]["validate_every"]) == 0:
            metrics = _validate(rendered.detach(), target)
            record.update(metrics)
            torch.save({"head": head.state_dict(), "step": step}, output_dir / "checkpoint.pt")
        history.append(record)
        if step % int(config["train"]["log_every"]) == 0:
            print(json.dumps(record))
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--use-stub-backbone", action="store_true")
    parser.add_argument("--representation", choices=("foam", "gaussian"), default="foam")
    cli = parser.parse_args()
    train(
        _load_config(cli.config),
        cli.data_root,
        cli.checkpoint,
        cli.use_stub_backbone,
        cli.representation,
    )


if __name__ == "__main__":
    main()
