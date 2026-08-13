"""Frozen-VGGT, head-only Power Foam training entry point."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from .backbone import FrozenGeometryStub, FrozenVGGTOmega
from .data.blender import BlenderNvsDataset
from .data.multiscene import MultiSceneScanNetPP
from .fusion import (
    align_depths_to_calibrated_cameras,
    build_canonical_support,
    projected_context_support_mask,
    world_points_from_z_depth,
)
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
    return torch.stack([view.image.permute(2, 0, 1) for view in episode.context])[None].to(device)


def _charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


def _rgb_loss(prediction: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "charbonnier":
        return _charbonnier(prediction, target)
    if name == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError(f"Unknown RGB loss: {name}")


def _metrics(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(rendered.clamp(0, 1), target).item()
    return {"mse": mse, "psnr": -10.0 * torch.log10(torch.tensor(mse + 1e-10)).item()}


def _build_datasets(config: dict[str, Any], data_root: Path):
    data_cfg = config["data"]
    dataset_name = str(data_cfg.get("dataset", "blender"))
    context_views = int(data_cfg["context_views"])
    target_views = int(data_cfg["target_views"])
    if context_views not in {1, 2}:
        raise ValueError("Canonical experiments require one or two context views")
    if dataset_name == "blender":
        train_dataset = BlenderNvsDataset(
            data_root,
            split="train",
            context_views=context_views,
            target_views=target_views,
            image_downsample=int(data_cfg["image_downsample"]),
            seed=int(config["seed"]),
        )
        return train_dataset, None
    if dataset_name == "scannetpp":
        manifest = Path(data_cfg["scene_manifest"])
        resolution = int(config["backbone"]["image_resolution"])
        train_dataset = MultiSceneScanNetPP(
            data_root,
            manifest,
            split="train",
            context_views=context_views,
            target_views=target_views,
            image_resolution=resolution,
            target_pool_size=int(data_cfg.get("target_pool_size", 32)),
            seed=int(config["seed"]),
        )
        val_dataset = MultiSceneScanNetPP(
            data_root,
            manifest,
            split="val",
            context_views=context_views,
            target_views=target_views,
            image_resolution=resolution,
            target_pool_size=int(data_cfg.get("target_pool_size", 32)),
            seed=int(config["seed"]) + 10_000,
        )
        return train_dataset, val_dataset
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _sample_episode(dataset, step: int, resample: bool, fixed_episode):
    if isinstance(dataset, MultiSceneScanNetPP):
        return dataset.sample_episode() if resample else fixed_episode
    return dataset[step % len(dataset)] if resample else fixed_episode


def _render_targets(params, target_views, bridge, representation: str, device: torch.device):
    outputs = []
    for target_view in target_views:
        if representation == "foam":
            outputs.append(bridge.render(params, camera_from_view(target_view, device)))
        else:
            outputs.append(bridge.render(params, target_view))
    return outputs


def _episode_objective(
    outputs,
    target_views,
    *,
    rgb_loss_name: str,
    alpha_weight: float,
    device: torch.device,
    masks=None,
):
    rgb_losses = []
    alpha_losses = []
    for index, (output, target_view) in enumerate(zip(outputs, target_views, strict=True)):
        target = target_view.image.to(device)
        if masks is None:
            rgb_losses.append(_rgb_loss(output.rgb, target, rgb_loss_name))
        else:
            mask = masks[index].to(device)
            if mask.any():
                rgb_losses.append(_rgb_loss(output.rgb[mask], target[mask], rgb_loss_name))
            else:
                # Predicted context geometry can occasionally have no projected
                # support. Fall back to full RGB rather than dropping the step.
                rgb_losses.append(_rgb_loss(output.rgb, target, rgb_loss_name))
        if alpha_weight > 0:
            if target_view.alpha is None:
                raise ValueError("alpha_loss_weight requires target alpha")
            alpha_losses.append(F.l1_loss(output.alpha, target_view.alpha.to(device)))
    rgb_loss = torch.stack(rgb_losses).mean()
    alpha_loss = (
        torch.stack(alpha_losses).mean() if alpha_losses else torch.zeros((), device=device)
    )
    return rgb_loss + alpha_weight * alpha_loss, rgb_loss, alpha_loss


def _predict(head, backbone, episode, representation: str, device: torch.device):
    images = _context_tensor(episode, device)
    with torch.no_grad():
        features = backbone(images)
    # FrozenVGGTOmega uses inference mode internally. Clone its outputs into
    # ordinary no-grad tensors before feeding trainable projection layers.
    features = {name: value.clone() for name, value in features.items()}
    aligned_depths, alignment = align_depths_to_calibrated_cameras(features, episode.context)
    features["depth"] = aligned_depths
    features["depth_alignment_scale"] = alignment.scale[None]
    ray_map = pinhole_ray_map_from_view(episode.context[0], device)
    support = build_canonical_support(images, features, episode.context, device)
    if representation == "foam":
        canonical_points = world_points_from_z_depth(
            episode.context[0], features["depth"][:, 0], device
        )
        params = head(
            images,
            features,
            ray_map,
            episode.context[0].alpha,
            canonical_support=support,
            canonical_base_points=canonical_points,
        )
    else:
        params = head(images, features, ray_map)
    return params, features


def _validation(
    episodes,
    *,
    head,
    backbone,
    bridge,
    representation: str,
    device: torch.device,
) -> dict[str, float]:
    mse_values = []
    alpha_values = []
    with torch.no_grad():
        for episode in episodes:
            params, _ = _predict(head, backbone, episode, representation, device)
            outputs = _render_targets(params, episode.target, bridge, representation, device)
            for output, target_view in zip(outputs, episode.target, strict=True):
                mse_values.append(
                    F.mse_loss(output.rgb.clamp(0, 1), target_view.image.to(device)).item()
                )
                alpha_values.append(float(output.alpha.mean()))
    mse = sum(mse_values) / len(mse_values)
    return {
        "val_mse": mse,
        "val_psnr": -10.0 * torch.log10(torch.tensor(mse + 1e-10)).item(),
        "val_alpha_mean": sum(alpha_values) / len(alpha_values),
        "val_renders": float(len(mse_values)),
    }


def _checkpoint_state(head, optimizer, step, history, train_dataset, config):
    state = {
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "config": config,
    }
    if isinstance(train_dataset, MultiSceneScanNetPP):
        state["dataset"] = train_dataset.state_dict()
    elif hasattr(train_dataset, "generator"):
        state["dataset_generator"] = train_dataset.generator.get_state()
    return state


def _atomic_save(state: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train(
    config: dict[str, Any],
    data_root: Path,
    checkpoint: Path | None,
    use_stub: bool,
    representation: str,
    resume: Path | None = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA because Power Foam uses Warp kernels")
    device = torch.device("cuda")
    torch.manual_seed(int(config["seed"]))
    train_dataset, val_dataset = _build_datasets(config, data_root)
    if use_stub:
        backbone = FrozenGeometryStub().to(device)
        register_dim = backbone.register_dim
    else:
        if checkpoint is None:
            raise ValueError("VGGT-Ω checkpoint is required")
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
            radius_residual_log_scale=float(head_cfg.get("radius_residual_log_scale", 0.25)),
            density_mode=str(head_cfg.get("density_mode", "learned")),
            fixed_density=float(head_cfg.get("fixed_density", 100.0)),
            initialize_rgb_from_image=bool(head_cfg.get("initialize_rgb_from_image", False)),
            initialize_normals_from_depth=bool(head_cfg.get("initialize_normals_from_depth", True)),
            base_depth_mode=str(head_cfg.get("base_depth_mode", "predicted")),
            constant_base_depth=float(head_cfg.get("constant_base_depth", 2.0)),
            point_residual_scale=float(head_cfg.get("point_residual_scale", 0.05)),
            normal_residual_radians=float(head_cfg.get("normal_residual_radians", 0.25)),
            rgb_residual_scale=float(head_cfg.get("rgb_residual_scale", 0.5)),
            fusion_mode=str(head_cfg.get("fusion_mode", "none")),
            patch_token_dim=register_dim,
        ).to(device)
    elif representation == "gaussian":
        head = CanonicalGaussianHead(
            register_dim=register_dim,
            hidden_dim=int(head_cfg["hidden_dim"]),
            max_cells=int(head_cfg["max_cells"]),
        ).to(device)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.01)),
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))

    initial_episode = _sample_episode(train_dataset, 0, True, None)
    if representation == "foam":
        reference_camera = camera_from_view(initial_episode.context[0], device)
        bridge = PowerFoamRendererBridge(
            powerfoam_args(
                num_texel_sites=int(head_cfg["num_texel_sites"]),
                sv_dof=int(head_cfg["spherical_voronoi_dof"]),
                bkgd_color=tuple(config["renderer"]["bkgd_color"]),
                is_pinhole=bool(config["renderer"]["is_pinhole"]),
            ),
            reference_camera,
        )
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
    start_step = 1
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start_step = int(state["step"]) + 1
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        if isinstance(train_dataset, MultiSceneScanNetPP) and "dataset" in state:
            train_dataset.load_state_dict(state["dataset"])
        elif "dataset_generator" in state:
            train_dataset.generator.set_state(state["dataset_generator"])

    train_cfg = config["train"]
    fixed_episode = initial_episode
    resample = bool(train_cfg.get("resample_episodes", True))
    val_episodes = (
        val_dataset.fixed_episodes(
            int(train_cfg.get("validation_episodes", 4)), int(config["seed"]) + 20_000
        )
        if val_dataset is not None
        else ()
    )
    for step in range(start_step, int(train_cfg["steps"]) + 1):
        episode = _sample_episode(train_dataset, step, resample, fixed_episode)
        params, features = _predict(head, backbone, episode, representation, device)
        target_views = episode.context if bool(train_cfg.get("target_from_context", False)) else episode.target
        outputs = _render_targets(params, target_views, bridge, representation, device)
        use_mask = bool(train_cfg.get("visibility_mask", False))
        masks = (
            [
                projected_context_support_mask(
                    episode.context,
                    features["depth"],
                    target,
                    device,
                    dilation=int(train_cfg.get("visibility_mask_dilation", 2)),
                )
                for target in target_views
            ]
            if use_mask
            else None
        )
        loss, rgb_loss, alpha_loss = _episode_objective(
            outputs,
            target_views,
            rgb_loss_name=str(train_cfg.get("rgb_loss", "charbonnier")),
            alpha_weight=float(train_cfg.get("alpha_loss_weight", 0.0)),
            device=device,
            masks=masks,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            head.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()

        per_target = [
            _metrics(output.rgb.detach(), view.image.to(device))
            for output, view in zip(outputs, target_views, strict=True)
        ]
        record = {
            "step": float(step),
            "loss": float(loss.detach()),
            "rgb_loss": float(rgb_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "grad_norm": float(grad_norm),
            "target_views": float(len(target_views)),
            "train_psnr": sum(metric["psnr"] for metric in per_target) / len(per_target),
            "train_mse": sum(metric["mse"] for metric in per_target) / len(per_target),
            "render_alpha_mean": sum(float(output.alpha.detach().mean()) for output in outputs)
            / len(outputs),
            "active_cells": float(params.points.shape[0] if representation == "foam" else params.means.shape[0]),
            "mean_radius": float(
                F.softplus(params.radii.detach(), beta=100).mean()
                if representation == "foam"
                else params.scales.detach().mean()
            ),
            "depth_alignment_scale": float(features["depth_alignment_scale"].mean()),
            "visibility_mask_fraction": float(
                torch.stack([mask.float().mean() for mask in masks]).mean()
                if masks is not None
                else 1.0
            ),
        }
        if val_episodes and step % int(train_cfg["validate_every"]) == 0:
            record.update(
                _validation(
                    val_episodes,
                    head=head,
                    backbone=backbone,
                    bridge=bridge,
                    representation=representation,
                    device=device,
                )
            )
        history.append(record)
        if step % int(train_cfg["log_every"]) == 0:
            print(json.dumps(record), flush=True)
        if step % int(train_cfg.get("checkpoint_every", train_cfg["validate_every"])) == 0:
            _atomic_save(
                _checkpoint_state(head, optimizer, step, history, train_dataset, config),
                output_dir / "latest.pt",
            )
            (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))

    _atomic_save(
        _checkpoint_state(head, optimizer, int(train_cfg["steps"]), history, train_dataset, config),
        output_dir / "final.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--use-stub-backbone", action="store_true")
    parser.add_argument("--representation", choices=("foam", "gaussian"), default="foam")
    cli = parser.parse_args()
    train(
        _load_config(cli.config),
        cli.data_root,
        cli.checkpoint,
        cli.use_stub_backbone,
        cli.representation,
        cli.resume,
    )


if __name__ == "__main__":
    main()
