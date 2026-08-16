#!/usr/bin/env python3
"""Directly optimize one fixed-triplet Power Foam scene as an upper bound."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn

from feedforwardfoam.backbone import FrozenVGGTOmega
from feedforwardfoam.head import CanonicalPowerFoamHead, FoamParameters
from feedforwardfoam.renderer import PowerFoamRendererBridge, camera_from_view, powerfoam_args
from feedforwardfoam.train import (
    _build_datasets,
    _configured_fixed_episode,
    _predict,
    _triplet_geometry,
)
from feedforwardfoam.fusion import projected_context_support_mask


PARAMETER_NAMES = (
    "points",
    "radii",
    "quaternions",
    "density",
    "texel_sites",
    "texel_sv_axis",
    "texel_sv_rgb",
    "texel_height",
)
REPORT_EVERY = 25


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Experiment config must contain a YAML mapping")
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("Experiment config must contain a data mapping")
    if str(data.get("dataset", "")) != "scannetpp":
        raise ValueError("Direct fixed-triplet optimization requires data.dataset: scannetpp")
    missing = [
        name for name in ("fixed_scene_id", "context_names", "target_names") if not data.get(name)
    ]
    if missing:
        raise ValueError(f"Fixed-triplet config is missing explicit data fields: {missing}")
    return config


def _build_head(
    config: dict[str, Any], register_dim: int, device: torch.device
) -> CanonicalPowerFoamHead:
    head_cfg = config["head"]
    return CanonicalPowerFoamHead(
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
        prediction_mode=str(head_cfg.get("prediction_mode", "residual")),
        enable_point_residual=bool(head_cfg.get("enable_point_residual", True)),
        enable_radius_residual=bool(head_cfg.get("enable_radius_residual", True)),
        enable_orientation_residual=bool(head_cfg.get("enable_orientation_residual", True)),
        enable_rgb_residual=bool(head_cfg.get("enable_rgb_residual", True)),
    ).to(device)


def _load_head_checkpoint(head: nn.Module, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "head" in state:
        state = state["head"]
    if not isinstance(state, dict):
        raise ValueError(
            "Initial checkpoint must be a head state dict or contain a 'head' state dict"
        )
    head.load_state_dict(state, strict=True)


def _make_parameters(initial: FoamParameters) -> FoamParameters:
    values = {
        name: nn.Parameter(getattr(initial, name).detach().clone().contiguous())
        for name in PARAMETER_NAMES
    }
    return FoamParameters(**values)


def _render(parameters: FoamParameters, episode, bridge, device: torch.device):
    return [
        bridge.render(parameters, camera_from_view(target, device)) for target in episode.target
    ]


def _mse(
    outputs, episode, device: torch.device, masks: list[torch.Tensor] | None = None
) -> torch.Tensor:
    losses = []
    for index, (output, target_view) in enumerate(zip(outputs, episode.target, strict=True)):
        target = target_view.image.to(device)
        if masks is not None and masks[index].any():
            losses.append(F.mse_loss(output.rgb[masks[index]], target[masks[index]]))
        else:
            losses.append(F.mse_loss(output.rgb, target))
    return torch.stack(losses).mean()


def _metrics(outputs, episode, device: torch.device, masks: list[torch.Tensor]) -> dict[str, Any]:
    full_errors = []
    support_errors = []
    for output, target_view, mask in zip(outputs, episode.target, masks, strict=True):
        prediction = output.rgb.detach().clamp(0, 1)
        target = target_view.image.to(device)
        full_errors.append((prediction - target).square().reshape(-1))
        if mask.any():
            support_errors.append((prediction[mask] - target[mask]).square().reshape(-1))
    full_mse = float(torch.cat(full_errors).mean())
    support_mse = float(torch.cat(support_errors).mean()) if support_errors else None
    return {
        "full_mse": full_mse,
        "full_psnr": -10.0 * math.log10(max(full_mse, 1e-10)),
        "support_mse": support_mse,
        "support_psnr": (
            -10.0 * math.log10(max(support_mse, 1e-10)) if support_mse is not None else None
        ),
        "support_fraction": float(torch.stack([mask.float().mean() for mask in masks]).mean()),
    }


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(array).save(path)


def _save_render(output_dir: Path, step: int, outputs, episode, device: torch.device) -> None:
    render_dir = output_dir / "renders" / f"step_{step:06d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    multiple = len(outputs) > 1
    for index, (output, target_view) in enumerate(zip(outputs, episode.target, strict=True)):
        suffix = f"_{index}" if multiple else ""
        target = target_view.image.to(device)
        _save_tensor(render_dir / f"target{suffix}.png", target)
        _save_tensor(render_dir / f"pred{suffix}.png", output.rgb)
        _save_tensor(render_dir / f"error{suffix}.png", (output.rgb - target).abs())


def optimize(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Direct Power Foam optimization requires a CUDA GPU")
    if args.steps < 0:
        raise ValueError("--steps must be nonnegative")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.render_every <= 0:
        raise ValueError("--render-every must be positive")

    config = _load_config(args.config)
    device = torch.device("cuda")
    torch.manual_seed(int(config["seed"]))
    dataset, validation_dataset = _build_datasets(config, args.data_root)
    if validation_dataset is not None:
        raise ValueError("Fixed-triplet config unexpectedly constructed a validation dataset")
    episode = _configured_fixed_episode(dataset, config)
    if episode is None:
        raise ValueError("Config did not produce an explicit fixed episode")

    backbone = FrozenVGGTOmega(args.vggt_checkpoint).to(device).eval()
    head = _build_head(config, backbone.register_dim, device).eval()
    if args.init_checkpoint is not None:
        _load_head_checkpoint(head, args.init_checkpoint, device)
    with torch.no_grad():
        initial, features = _predict(head, backbone, episode, "foam", device)
    parameters = _make_parameters(initial)

    head_cfg = config["head"]
    fixed_density = str(head_cfg.get("density_mode", "learned")) in {
        "fixed",
        "source_alpha_fixed",
    }
    optimized = [
        getattr(parameters, name)
        for name in PARAMETER_NAMES
        if not (fixed_density and name == "density")
    ]
    optimizer = torch.optim.AdamW(optimized, lr=args.learning_rate, weight_decay=0.0)

    bridge = PowerFoamRendererBridge(
        powerfoam_args(
            num_texel_sites=int(head_cfg["num_texel_sites"]),
            sv_dof=int(head_cfg["spherical_voronoi_dof"]),
            bkgd_color=tuple(config["renderer"]["bkgd_color"]),
            is_pinhole=bool(config["renderer"]["is_pinhole"]),
        ),
        camera_from_view(episode.context[0], device),
    )
    dilation = int(config.get("train", {}).get("visibility_mask_dilation", 2))
    support_masks = [
        projected_context_support_mask(
            episode.context[:1],
            features["depth"][:, :1],
            target,
            device,
            dilation=dilation,
        )
        for target in episode.target
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    geometry = _triplet_geometry(episode)
    (args.output_dir / "triplet_geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    metadata = {
        "config": str(args.config),
        "data_root": str(args.data_root),
        "vggt_checkpoint": str(args.vggt_checkpoint),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "initialization": "checkpoint" if args.init_checkpoint else "fresh_exact",
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "render_every": args.render_every,
        "visibility_mask": args.visibility_mask,
        "visibility_mask_contexts": "canonical",
        "density_optimized": not fixed_density,
        "parameter_names": list(PARAMETER_NAMES),
    }
    (args.output_dir / "optimization_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    history: list[dict[str, Any]] = []
    for step in range(args.steps + 1):
        if step > 0:
            for parameter in parameters.as_upstream_tensors().values():
                parameter.grad = None
            outputs = _render(parameters, episode, bridge, device)
            loss = _mse(outputs, episode, device, support_masks if args.visibility_mask else None)
            loss.backward()
            optimizer.step()

        report = step % REPORT_EVERY == 0 or step == args.steps
        save_render = step % args.render_every == 0 or step == args.steps
        if report or save_render:
            with torch.no_grad():
                evaluated = _render(parameters, episode, bridge, device)
            if report:
                record = {"step": step, **_metrics(evaluated, episode, device, support_masks)}
                history.append(record)
                (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
            if save_render:
                _save_render(args.output_dir, step, evaluated, episode, device)

    torch.save(
        {name: tensor.detach().cpu() for name, tensor in parameters.as_upstream_tensors().items()},
        args.output_dir / "final.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vggt-checkpoint", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--render-every", type=int, default=250)
    parser.add_argument(
        "--visibility-mask",
        action="store_true",
        help="Restrict optimization loss to projected canonical-context support.",
    )
    optimize(parser.parse_args())


if __name__ == "__main__":
    main()
