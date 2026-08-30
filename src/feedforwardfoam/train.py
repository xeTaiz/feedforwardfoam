"""Frozen-VGGT, head-only Power Foam training entry point."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from .backbone import FrozenGeometryStub, FrozenVGGTOmega
from .data.blender import BlenderNvsDataset
from .data.multiscene import MultiSceneScanNetPP
from .data.types import NvsEpisode
from .data.scannetpp import (
    CorruptDepthMapError,
    MissingDepthMapError,
    ScanNetPPDataset,
)
from .fusion import (
    InvalidDepthGaugeError,
    align_depths_to_calibrated_cameras,
    build_canonical_support,
    laser_context_support_mask,
    projected_context_support_mask,
    world_points_from_z_depth,
)
from .gaussian import CanonicalGaussianHead, GaussianRendererBridge
from .head import (
    CanonicalPowerFoamHead,
    concatenate_foam_parameters,
    farthest_point_indices,
    incremental_containment_indices,
    select_foam_parameters,
    uniform_selection_indices,
    voxel_budget_indices,
)
from .metrics import SpatialLPIPS, masked_lpips, masked_ssim, new_lpips
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


def _batched_backbone_features(
    backbone,
    episodes: list[NvsEpisode],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Run the frozen backbone once for equally shaped scene contexts."""
    if not episodes:
        return []
    images = torch.cat([_context_tensor(episode, device) for episode in episodes])
    with torch.no_grad():
        batched = backbone(images)
    # FrozenVGGTOmega uses inference mode internally. Clone once into ordinary
    # no-grad storage before trainable projection layers consume these tensors.
    batched = {name: value.clone() for name, value in batched.items()}
    return [
        {name: value[index : index + 1] for name, value in batched.items()}
        for index in range(len(episodes))
    ]


def _charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


def _rgb_loss(prediction: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "charbonnier":
        return _charbonnier(prediction, target)
    if name == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError(f"Unknown RGB loss: {name}")


def _metrics(
    rendered: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    check_nonempty: bool = True,
) -> dict[str, torch.Tensor]:
    rendered = rendered.clamp(0, 1)
    if mask is not None:
        if check_nonempty and not mask.any():
            raise ValueError("Cannot compute support metrics for an empty mask")
        rendered = rendered[mask]
        target = target[mask]
    mse = F.mse_loss(rendered, target)
    return {"mse": mse, "psnr": -10.0 * torch.log10(mse + 1e-10)}


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
        resolution = int(config["backbone"]["image_resolution"])
        if "fixed_scene_id" in data_cfg:
            train_dataset = ScanNetPPDataset(
                data_root / str(data_cfg["fixed_scene_id"]),
                split=str(data_cfg.get("fixed_split", "train")),
                context_views=context_views,
                target_views=target_views,
                image_resolution=resolution,
                seed=int(config["seed"]),
            )
            return train_dataset, None
        manifest = Path(data_cfg["scene_manifest"])
        target_pool = data_cfg.get("target_pool_size", 32)
        common: dict[str, Any] = {
            "data_root": data_root,
            "scene_manifest": manifest,
            "context_views": context_views,
            "image_resolution": resolution,
            "target_pool_size": int(target_pool) if target_pool is not None else None,
            "reserve_support_view": bool(data_cfg.get("reserve_support_view", False)),
            "native_image_directory": str(
                data_cfg.get("native_image_directory", "resized_undistorted_images")
            ),
            "resize_mode": str(data_cfg.get("resize_mode", "area")),
            "load_depth": bool(data_cfg.get("load_depth", False)),
            "tensor_cache_root": data_cfg.get("tensor_cache_root"),
        }
        train_dataset = MultiSceneScanNetPP(
            **common,
            split="train",
            target_views=target_views,
            coverage_root=data_cfg.get("coverage_root"),
            context_overlap_threshold=float(data_cfg.get("context_overlap_threshold", 0.5)),
            target_overlap_threshold=float(data_cfg.get("target_overlap_threshold", 0.6)),
            seed=int(config["seed"]),
        )
        val_dataset = MultiSceneScanNetPP(
            **common,
            split="val",
            target_views=int(data_cfg.get("validation_target_views", target_views)),
            seed=int(config["seed"]) + 10_000,
        )
        return train_dataset, val_dataset
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _sample_episode(
    dataset, step: int, resample: bool, fixed_episode: NvsEpisode | None
) -> NvsEpisode:
    episode = (
        dataset.sample_episode()
        if isinstance(dataset, MultiSceneScanNetPP) and resample
        else dataset[step % len(dataset)]
        if resample
        else fixed_episode
    )
    if episode is None:
        raise ValueError("A fixed episode is required when resampling is disabled")
    return episode


def _prefetched_episodes(
    dataset,
    start_index: int,
    count: int,
    resample: bool,
    fixed_episode: NvsEpisode | None,
    workers: int = 1,
):
    """Load selected episodes concurrently while preserving their sampled order."""
    if workers <= 0:
        raise ValueError("Episode prefetch workers must be positive")
    if isinstance(dataset, MultiSceneScanNetPP) and resample and workers > 1:
        requests = [dataset.sample_episode_request() for _ in range(count)]
        with ThreadPoolExecutor(max_workers=min(workers, count)) as pool:
            yield from pool.map(dataset.load_episode_request, requests)
        return

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_sample_episode, dataset, start_index, resample, fixed_episode)
        for offset in range(count):
            episode = future.result()
            if offset + 1 < count:
                future = pool.submit(
                    _sample_episode,
                    dataset,
                    start_index + offset + 1,
                    resample,
                    fixed_episode,
                )
            yield episode


@torch.no_grad()
def _clip_grad_norm_stable(parameters, max_norm: float) -> torch.Tensor:
    """Clip finite float32 gradients without overflowing their aggregate norm."""
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.zeros((), dtype=torch.float64)
    norms = torch.stack(
        [torch.linalg.vector_norm(gradient, dtype=torch.float64) for gradient in gradients]
    )
    total_norm = torch.linalg.vector_norm(norms)
    if not torch.isfinite(total_norm):
        raise RuntimeError("Gradient norm is non-finite")
    coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return total_norm


def _configured_fixed_episode(dataset, config: dict[str, Any]) -> NvsEpisode | None:
    data_cfg = config["data"]
    context_names = data_cfg.get("context_names")
    target_names = data_cfg.get("target_names")
    if context_names is None and target_names is None:
        return None
    if not isinstance(dataset, ScanNetPPDataset):
        raise ValueError("Explicit context_names/target_names require a fixed ScanNet++ scene")
    if context_names is None or target_names is None:
        raise ValueError("Both context_names and target_names must be configured")
    return dataset.episode_from_names(list(context_names), list(target_names))


def _triplet_geometry(episode) -> dict[str, Any]:
    if len(episode.context) != 2 or len(episode.target) != 1:
        return {}
    centers = [view.c2w[:3, 3] for view in (*episode.context, *episode.target)]
    c0, c1, target = centers
    segment = c1 - c0
    baseline = torch.linalg.vector_norm(segment).clamp_min(1e-8)
    interpolation = torch.dot(target - c0, segment) / baseline.square()
    closest = c0 + interpolation * segment
    forwards = [-view.c2w[:3, 2] for view in (*episode.context, *episode.target)]

    def angle_degrees(first: torch.Tensor, second: torch.Tensor) -> float:
        cosine = F.cosine_similarity(first[None], second[None]).clamp(-1, 1)
        return float(torch.rad2deg(torch.acos(cosine))[0])

    return {
        "scene_id": episode.scene_id,
        "context_names": [view.name for view in episode.context],
        "target_names": [view.name for view in episode.target],
        "context_baseline": float(baseline),
        "target_interpolation": float(interpolation),
        "target_perpendicular_distance": float(torch.linalg.vector_norm(target - closest)),
        "target_perpendicular_fraction": float(
            torch.linalg.vector_norm(target - closest) / baseline
        ),
        "context_target_distances": [
            float(torch.linalg.vector_norm(target - c0)),
            float(torch.linalg.vector_norm(target - c1)),
        ],
        "view_angle_degrees": {
            "context_context": angle_degrees(forwards[0], forwards[1]),
            "context0_target": angle_degrees(forwards[0], forwards[2]),
            "context1_target": angle_degrees(forwards[1], forwards[2]),
        },
        "target_between_contexts": bool(0.0 <= interpolation <= 1.0),
    }


def _save_render_bundle(snapshot_dir: Path, episode, outputs) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_tensor(tensor: torch.Tensor, path: Path) -> None:
        array = tensor.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(array).save(path)

    for index, view in enumerate(episode.context):
        save_tensor(view.image, snapshot_dir / f"context_{index}.png")
    for index, (view, output) in enumerate(zip(episode.target, outputs, strict=True)):
        target = view.image.to(output.rgb.device)
        save_tensor(view.image, snapshot_dir / f"target_{index}.png")
        save_tensor(output.rgb, snapshot_dir / f"prediction_{index}.png")
        save_tensor((output.rgb - target).abs(), snapshot_dir / f"error_{index}.png")


def _save_diagnostic_images(output_dir: Path, step: int, episode, outputs) -> None:
    _save_render_bundle(output_dir / "diagnostic_renders" / f"step_{step:06d}", episode, outputs)


def _render_targets(params, target_views, bridge, representation: str, device: torch.device):
    if representation == "foam":
        cameras = [camera_from_view(target_view, device) for target_view in target_views]
        return bridge.render_many(params, cameras)
    return [bridge.render(params, target_view) for target_view in target_views]


class EmptySupportMaskError(ValueError):
    """A sampled episode has no observable pixels for masked supervision."""


def _episode_objective(
    outputs,
    target_views,
    *,
    rgb_loss_name: str,
    alpha_weight: float,
    device: torch.device,
    masks=None,
    lpips_weight: float = 0.0,
    lpips_model: SpatialLPIPS | None = None,
    splatt3r_masked_reduction: bool = False,
):
    predictions = torch.stack([output.rgb for output in outputs])
    targets = torch.stack([view.image.to(device) for view in target_views])
    mask_batch = torch.stack([mask.to(device) for mask in masks]) if masks is not None else None
    if mask_batch is not None and not mask_batch.flatten(1).any(dim=1).all():
        raise EmptySupportMaskError("Masked benchmark supervision cannot use an empty support mask")

    if rgb_loss_name == "charbonnier":
        rgb_error = torch.sqrt((predictions - targets).square() + 1e-6)
    elif rgb_loss_name == "mse":
        rgb_error = (predictions - targets).square()
    else:
        raise ValueError(f"Unknown RGB loss: {rgb_loss_name}")
    if mask_batch is None:
        rgb_loss = rgb_error.flatten(1).mean(dim=1).mean()
    else:
        per_target_sum = (rgb_error * mask_batch[..., None]).flatten(1).sum(dim=1)
        per_target_count = 3 * mask_batch.flatten(1).sum(dim=1)
        rgb_loss = (per_target_sum / per_target_count.clamp_min(1)).mean()
        if splatt3r_masked_reduction:
            if rgb_loss_name != "mse":
                raise ValueError("Splatt3R masked reduction is defined only for MSE")
            rgb_loss = 3.0 * rgb_loss

    if lpips_weight > 0:
        if lpips_model is None:
            raise ValueError("lpips_weight requires an LPIPS model")
        perceptual_loss = masked_lpips(
            lpips_model,
            predictions,
            targets,
            mask_batch,
            check_nonempty=False,
        )
    else:
        perceptual_loss = torch.zeros((), device=device)
    if alpha_weight > 0:
        if any(target_view.alpha is None for target_view in target_views):
            raise ValueError("alpha_loss_weight requires target alpha")
        target_alpha = torch.stack([target_view.alpha.to(device) for target_view in target_views])
        output_alpha = torch.stack([output.alpha for output in outputs])
        alpha_loss = F.l1_loss(output_alpha, target_alpha)
    else:
        alpha_loss = torch.zeros((), device=device)
    loss = rgb_loss + lpips_weight * perceptual_loss + alpha_weight * alpha_loss
    return loss, rgb_loss, perceptual_loss, alpha_loss


def _reorder_context_features(
    features: dict[str, torch.Tensor], order: list[int], view_count: int
) -> dict[str, torch.Tensor]:
    return {
        name: value[:, order] if value.ndim >= 2 and value.shape[1] == view_count else value
        for name, value in features.items()
    }


def _proposal_confidence(
    features: dict[str, torch.Tensor], height: int, width: int, budget: int
) -> torch.Tensor:
    """Frozen depth confidence per proposal, ordered like the head's cells.

    The head decodes canonical-view pixels in raster order and reduces them
    with the same uniform stride, so applying that stride here keeps scores
    aligned with the cells they describe.
    """
    confidence = features["depth_conf"][:, 0]
    if confidence.ndim == 3:
        confidence = confidence[:, None]
    if confidence.shape[-2:] != (height, width):
        confidence = F.interpolate(
            confidence, size=(height, width), mode="bilinear", align_corners=False
        )
    flat = confidence.reshape(-1)
    return flat[uniform_selection_indices(flat.shape[0], budget, flat.device)]


def _predict(
    head,
    backbone,
    episode,
    representation: str,
    device: torch.device,
    frozen_features: dict[str, torch.Tensor] | None = None,
):
    images = _context_tensor(episode, device)
    if frozen_features is None:
        frozen_features = _batched_backbone_features(backbone, [episode], device)[0]
    features = frozen_features
    aligned_depths, alignment = align_depths_to_calibrated_cameras(features, episode.context)
    features["depth"] = aligned_depths
    features["depth_alignment_scale"] = alignment.scale[None]
    features["depth_alignment_raw_scale"] = alignment.raw_scale[None]
    features["depth_alignment_bound_hit"] = torch.tensor(float(alignment.bound_hit), device=device)[
        None
    ]

    proposal_views = getattr(head, "proposal_views", "canonical")
    if representation != "foam" or proposal_views == "canonical":
        ray_map = pinhole_ray_map_from_view(episode.context[0], device)
        support = build_canonical_support(images, features, episode.context, device)
        features["canonical_support_fraction"] = (
            support.maps[:, -1].mean() if support is not None else torch.zeros((), device=device)
        )[None]
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

    view_count = len(episode.context)
    if view_count < 2:
        raise ValueError("All-view proposals require at least two context views")
    reduction = head.proposal_reduction
    if reduction == "none":
        raise ValueError(
            "All-view proposals require all, balanced, voxel, fps, "
            "confidence_voxel, or incremental reduction"
        )
    if reduction == "confidence_voxel" and not (
        head.selection_mode == "uniform" or head.prediction_mode == "initialization"
    ):
        raise ValueError(
            "confidence_voxel reduction requires uniform selection so proposal "
            "scores stay aligned with decoded cells"
        )
    height, width = episode.context[0].image.shape[:2]
    full_budget_reductions = {"all", "voxel", "fps", "confidence_voxel", "incremental"}
    per_view_budgets = []
    for index in range(view_count):
        if reduction in full_budget_reductions:
            per_view_budgets.append(height * width)
        else:
            per_view_budgets.append(
                head.max_cells // view_count + int(index < head.max_cells % view_count)
            )

    proposals = []
    support_fractions = []
    proposal_scores = []
    for canonical_index in range(view_count):
        order = [canonical_index, *[i for i in range(view_count) if i != canonical_index]]
        ordered_images = images[:, order]
        ordered_features = _reorder_context_features(features, order, view_count)
        ordered_contexts = tuple(episode.context[index] for index in order)
        support = build_canonical_support(
            ordered_images, ordered_features, ordered_contexts, device
        )
        support_fractions.append(
            support.maps[:, -1].mean() if support is not None else torch.zeros((), device=device)
        )
        ray_map = pinhole_ray_map_from_view(ordered_contexts[0], device)
        canonical_points = world_points_from_z_depth(
            ordered_contexts[0], ordered_features["depth"][:, 0], device
        )
        proposals.append(
            head(
                ordered_images,
                ordered_features,
                ray_map,
                ordered_contexts[0].alpha,
                canonical_support=support,
                canonical_base_points=canonical_points,
                max_cells_override=per_view_budgets[canonical_index],
            )
        )
        if reduction == "confidence_voxel":
            proposal_scores.append(
                _proposal_confidence(
                    ordered_features, height, width, per_view_budgets[canonical_index]
                )
            )
    params = concatenate_foam_parameters(proposals)
    if reduction in {"voxel", "fps", "confidence_voxel"}:
        cache_key = (
            reduction,
            episode.scene_id,
            *(view.name for view in episode.context),
            height,
            width,
            head.max_cells,
        )
        indices = head._proposal_index_cache.get(cache_key)
        if indices is None:
            if reduction == "voxel":
                indices = voxel_budget_indices(params.points, head.max_cells)
            elif reduction == "fps":
                indices = farthest_point_indices(params.points, head.max_cells)
            else:
                indices = voxel_budget_indices(
                    params.points, head.max_cells, scores=torch.cat(proposal_scores)
                )
            indices = indices.detach().cpu()
            head._proposal_index_cache[cache_key] = indices
        params = select_foam_parameters(params, indices.to(params.points.device))
    if reduction == "incremental":
        indices = incremental_containment_indices(
            params.points,
            params.radii,
            [proposal.points.shape[0] for proposal in proposals],
            criterion=head.proposal_containment,
            tolerance=head.proposal_containment_tolerance,
        )
        params = select_foam_parameters(params, indices)
    features["canonical_support_fraction"] = torch.stack(support_fractions).mean()[None]
    return params, features


def _validation(
    records,
    *,
    head,
    backbone,
    bridge,
    representation: str,
    device: torch.device,
    support_context_mode: str,
    support_dilation: int,
    output_dir: Path | None = None,
    step: int | None = None,
    support_mask_source: str = "predicted",
    lpips_model: SpatialLPIPS | None = None,
    benchmark_metrics: bool = False,
) -> dict[str, float]:
    groups: dict[str, dict[str, list[float]]] = {}

    def group(name: str) -> dict[str, list[float]]:
        return groups.setdefault(
            name,
            {
                "mse": [],
                "support_mse": [],
                "support_fraction": [],
                "alpha": [],
                "coverage": [],
                "gauge_hit": [],
                "ssim": [],
                "lpips": [],
                "support_ssim": [],
                "support_lpips": [],
            },
        )

    with torch.no_grad():
        for index, (label, episode) in enumerate(records):
            params, features = _predict(head, backbone, episode, representation, device)
            outputs = _render_targets(params, episode.target, bridge, representation, device)
            mask_contexts = (
                episode.context[:1] if support_context_mode == "canonical" else episode.context
            )
            if support_mask_source == "laser":
                support_masks = [
                    laser_context_support_mask(mask_contexts, target, device)
                    for target in episode.target
                ]
            elif support_mask_source == "predicted":
                support_masks = [
                    projected_context_support_mask(
                        mask_contexts,
                        features["depth"][:, : len(mask_contexts)],
                        target,
                        device,
                        dilation=support_dilation,
                    )
                    for target in episode.target
                ]
            else:
                raise ValueError("support_mask_source must be 'predicted' or 'laser'")
            label = str(label)
            buckets = (group("all"),) if label == "all" else (group("all"), group(label))
            for output, target_view, support_mask in zip(
                outputs, episode.target, support_masks, strict=True
            ):
                target = target_view.image.to(device)
                mse = F.mse_loss(output.rgb.clamp(0, 1), target).item()
                support_mse = (
                    F.mse_loss(output.rgb.clamp(0, 1)[support_mask], target[support_mask]).item()
                    if support_mask.any()
                    else None
                )
                if benchmark_metrics:
                    if lpips_model is None:
                        raise ValueError("benchmark_metrics require an LPIPS model")
                    ssim = masked_ssim(output.rgb, target)
                    lpips_value = float(masked_lpips(lpips_model, output.rgb, target))
                    support_ssim = (
                        masked_ssim(output.rgb, target, support_mask)
                        if support_mask.any()
                        else None
                    )
                    support_lpips = (
                        float(masked_lpips(lpips_model, output.rgb, target, support_mask))
                        if support_mask.any()
                        else None
                    )
                else:
                    ssim = None
                    lpips_value = None
                    support_ssim = None
                    support_lpips = None
                alpha_mask = output.alpha > 0.01
                for bucket in buckets:
                    bucket["mse"].append(mse)
                    bucket["alpha"].append(float(output.alpha.mean()))
                    bucket["support_fraction"].append(float(support_mask.float().mean()))
                    bucket["coverage"].append(float(alpha_mask.float().mean()))
                    if support_mse is not None:
                        bucket["support_mse"].append(support_mse)
                    if ssim is not None and lpips_value is not None:
                        bucket["ssim"].append(ssim)
                        bucket["lpips"].append(lpips_value)
                    if support_ssim is not None and support_lpips is not None:
                        bucket["support_ssim"].append(support_ssim)
                        bucket["support_lpips"].append(support_lpips)
            gauge_hit = float(features["depth_alignment_bound_hit"].mean())
            for bucket in buckets:
                bucket["gauge_hit"].append(gauge_hit)
            if output_dir is not None and step is not None:
                _save_render_bundle(
                    output_dir
                    / "validation_renders"
                    / f"step_{step:06d}"
                    / f"{index:03d}_{label}_{episode.scene_id}",
                    episode,
                    outputs,
                )

    result: dict[str, float] = {}
    for label, values in groups.items():
        suffix = "" if label == "all" else f"_{label}"
        mse = sum(values["mse"]) / len(values["mse"])
        result[f"val_mse{suffix}"] = mse
        result[f"val_psnr{suffix}"] = -10.0 * torch.log10(torch.tensor(mse + 1e-10)).item()
        result[f"val_alpha_mean{suffix}"] = sum(values["alpha"]) / len(values["alpha"])
        result[f"val_support_fraction{suffix}"] = sum(values["support_fraction"]) / len(
            values["support_fraction"]
        )
        result[f"val_render_coverage_fraction{suffix}"] = sum(values["coverage"]) / len(
            values["coverage"]
        )
        result[f"val_gauge_bound_hit_fraction{suffix}"] = sum(values["gauge_hit"]) / len(
            values["gauge_hit"]
        )
        if values["support_mse"]:
            support_mse = sum(values["support_mse"]) / len(values["support_mse"])
            result[f"val_support_mse{suffix}"] = support_mse
            result[f"val_support_psnr{suffix}"] = (
                -10.0 * torch.log10(torch.tensor(support_mse + 1e-10)).item()
            )
            splatt3r_mse = 3.0 * support_mse
            result[f"val_splatt3r_masked_mse{suffix}"] = splatt3r_mse
            result[f"val_splatt3r_masked_psnr{suffix}"] = (
                -10.0 * torch.log10(torch.tensor(splatt3r_mse + 1e-10)).item()
            )
        if values["ssim"]:
            result[f"val_ssim{suffix}"] = sum(values["ssim"]) / len(values["ssim"])
            result[f"val_lpips{suffix}"] = sum(values["lpips"]) / len(values["lpips"])
        if values["support_ssim"]:
            result[f"val_support_ssim{suffix}"] = sum(values["support_ssim"]) / len(
                values["support_ssim"]
            )
            result[f"val_support_lpips{suffix}"] = sum(values["support_lpips"]) / len(
                values["support_lpips"]
            )
            result[f"val_splatt3r_masked_ssim{suffix}"] = 3.0 * result[f"val_support_ssim{suffix}"]
    result["val_renders"] = float(len(groups["all"]["mse"]))
    return result


def _training_episode(
    head,
    backbone,
    bridge,
    episode: NvsEpisode,
    representation: str,
    device: torch.device,
    train_cfg: dict[str, Any],
    lpips_model: SpatialLPIPS | None,
    loss_scale: float,
    frozen_features: dict[str, torch.Tensor] | None = None,
):
    params, features = _predict(
        head,
        backbone,
        episode,
        representation,
        device,
        frozen_features=frozen_features,
    )
    target_views = (
        episode.context if bool(train_cfg.get("target_from_context", False)) else episode.target
    )
    outputs = _render_targets(params, target_views, bridge, representation, device)
    use_mask = bool(train_cfg.get("visibility_mask", False))
    report_support = bool(train_cfg.get("report_support_metrics", False)) or use_mask
    support_context_mode = str(train_cfg.get("support_mask_contexts", "all"))
    if support_context_mode not in {"all", "canonical"}:
        raise ValueError("support_mask_contexts must be 'all' or 'canonical'")
    support_mask_source = str(train_cfg.get("support_mask_source", "predicted"))
    mask_contexts = episode.context[:1] if support_context_mode == "canonical" else episode.context
    if report_support:
        if support_mask_source == "laser":
            support_masks = [
                laser_context_support_mask(mask_contexts, target, device) for target in target_views
            ]
        elif support_mask_source == "predicted":
            support_masks = [
                projected_context_support_mask(
                    mask_contexts,
                    features["depth"][:, : len(mask_contexts)],
                    target,
                    device,
                    dilation=int(train_cfg.get("visibility_mask_dilation", 2)),
                )
                for target in target_views
            ]
        else:
            raise ValueError("support_mask_source must be 'predicted' or 'laser'")
    else:
        support_masks = None
    masks = support_masks if use_mask else None
    loss, rgb_loss, perceptual_loss, alpha_loss = _episode_objective(
        outputs,
        target_views,
        rgb_loss_name=str(train_cfg.get("rgb_loss", "charbonnier")),
        alpha_weight=float(train_cfg.get("alpha_loss_weight", 0.0)),
        device=device,
        masks=masks,
        lpips_weight=float(train_cfg.get("lpips_loss_weight", 0.0)),
        lpips_model=lpips_model,
        splatt3r_masked_reduction=bool(train_cfg.get("splatt3r_masked_rgb_reduction", False)),
    )
    (loss * loss_scale).backward()

    per_target = [
        _metrics(output.rgb.detach(), view.image.to(device))
        for output, view in zip(outputs, target_views, strict=True)
    ]
    support_nonempty = support_masks is not None and (
        use_mask or bool(torch.stack(support_masks).flatten(1).any(dim=1).all())
    )
    support_per_target = (
        [
            _metrics(
                output.rgb.detach(),
                view.image.to(device),
                mask,
                check_nonempty=False,
            )
            for output, view, mask in zip(outputs, target_views, support_masks, strict=True)
        ]
        if support_nonempty
        else None
    )
    stats = {
        "loss": loss.detach(),
        "rgb_loss": rgb_loss.detach(),
        "lpips_loss": perceptual_loss.detach(),
        "alpha_loss": alpha_loss.detach(),
        "target_views": torch.tensor(float(len(target_views)), device=device),
        "train_psnr": torch.stack([metric["psnr"] for metric in per_target]).mean(),
        "train_mse": torch.stack([metric["mse"] for metric in per_target]).mean(),
        "render_alpha_mean": torch.stack(
            [output.alpha.detach().mean() for output in outputs]
        ).mean(),
        "active_cells": torch.tensor(
            float(params.points.shape[0] if representation == "foam" else params.means.shape[0]),
            device=device,
        ),
        "mean_radius": (
            F.softplus(params.radii.detach(), beta=100).mean()
            if representation == "foam"
            else params.scales.detach().mean()
        ),
        "depth_alignment_scale": features["depth_alignment_scale"].mean(),
        "depth_alignment_raw_scale": features["depth_alignment_raw_scale"].mean(),
        "depth_alignment_bound_hit": features["depth_alignment_bound_hit"].mean(),
        "canonical_support_fraction": features["canonical_support_fraction"].mean(),
        "visibility_mask_fraction": (
            torch.stack([mask.float().mean() for mask in support_masks]).mean()
            if support_masks is not None
            else torch.ones((), device=device)
        ),
    }
    if support_per_target is not None:
        alpha_threshold = float(train_cfg.get("coverage_alpha_threshold", 0.01))
        alpha_masks = [output.alpha.detach() > alpha_threshold for output in outputs]
        intersections = [
            (alpha_mask & support_mask).float().mean()
            for alpha_mask, support_mask in zip(alpha_masks, support_masks, strict=True)
        ]
        unions = [
            (alpha_mask | support_mask).float().mean()
            for alpha_mask, support_mask in zip(alpha_masks, support_masks, strict=True)
        ]
        stats.update(
            {
                "support_mse": torch.stack([metric["mse"] for metric in support_per_target]).mean(),
                "support_psnr": torch.stack(
                    [metric["psnr"] for metric in support_per_target]
                ).mean(),
                "render_coverage_fraction": torch.stack(
                    [alpha_mask.float().mean() for alpha_mask in alpha_masks]
                ).mean(),
                "support_render_iou": torch.stack(intersections).mean()
                / torch.stack(unions).mean().clamp_min(1e-8),
            }
        )
    return stats, outputs


def _checkpoint_state(
    head, optimizer, scheduler, step, history, train_dataset, config
) -> dict[str, object]:
    state: dict[str, object] = {
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "config": config,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if isinstance(train_dataset, MultiSceneScanNetPP):
        state["dataset"] = train_dataset.state_dict()
    elif hasattr(train_dataset, "generator"):
        state["dataset_generator"] = train_dataset.generator.get_state()
    return state


def _atomic_save(state: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def build_head(
    config: dict[str, Any], register_dim: int, representation: str, device: torch.device
):
    """Construct the configured head. Shared by training and diagnostics."""
    head_cfg = config["head"]
    if representation == "gaussian":
        return CanonicalGaussianHead(
            register_dim=register_dim,
            hidden_dim=int(head_cfg["hidden_dim"]),
            max_cells=int(head_cfg["max_cells"]),
        ).to(device)
    if representation != "foam":
        raise ValueError(f"Unknown representation: {representation}")
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
        proposal_views=str(head_cfg.get("proposal_views", "canonical")),
        proposal_reduction=str(head_cfg.get("proposal_reduction", "none")),
        selection_mode=str(head_cfg.get("selection_mode", "gate")),
        proposal_containment=str(head_cfg.get("proposal_containment", "power")),
        proposal_containment_tolerance=float(head_cfg.get("proposal_containment_tolerance", 1.0)),
    ).to(device)


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
    head = build_head(config, register_dim, representation, device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.01)),
    )
    schedule_name = str(config["train"].get("learning_rate_schedule", "constant"))
    if schedule_name == "constant":
        scheduler = None
    elif schedule_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["train"]["steps"]),
            eta_min=float(config["train"].get("min_learning_rate", 1e-6)),
        )
    elif schedule_name == "half_decay":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(config["train"]["steps"]) // 2], gamma=0.1
        )
    else:
        raise ValueError("learning_rate_schedule must be 'constant', 'cosine', or 'half_decay'")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    configured_episode = _configured_fixed_episode(train_dataset, config)
    initial_episode = configured_episode or _sample_episode(train_dataset, 0, True, None)
    geometry = _triplet_geometry(initial_episode)
    if geometry:
        (output_dir / "triplet_geometry.json").write_text(json.dumps(geometry, indent=2))
        print(json.dumps({"triplet_geometry": geometry}), flush=True)
        data_cfg = config["data"]
        if bool(data_cfg.get("require_target_between_contexts", False)):
            max_perpendicular = float(data_cfg.get("max_perpendicular_fraction", 0.25))
            if not geometry["target_between_contexts"]:
                raise ValueError("Configured target camera is not between context cameras")
            if geometry["target_perpendicular_fraction"] > max_perpendicular:
                raise ValueError(
                    "Configured target camera is too far from the context-camera segment"
                )
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
        state = torch.load(resume, map_location="cpu", weights_only=False)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        start_step = int(state["step"]) + 1
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        if isinstance(train_dataset, MultiSceneScanNetPP) and "dataset" in state:
            train_dataset.load_state_dict(state["dataset"])
        elif "dataset_generator" in state:
            train_dataset.generator.set_state(state["dataset_generator"])

    train_cfg = config["train"]
    scene_batch_size = int(train_cfg.get("scene_batch_size", 1))
    if scene_batch_size <= 0:
        raise ValueError("scene_batch_size must be positive")
    backbone_batch_size = int(train_cfg.get("backbone_batch_size", 1))
    if backbone_batch_size <= 0:
        raise ValueError("backbone_batch_size must be positive")
    episode_prefetch_workers = int(train_cfg.get("episode_prefetch_workers", 1))
    if episode_prefetch_workers <= 0:
        raise ValueError("episode_prefetch_workers must be positive")
    support_context_mode = str(train_cfg.get("support_mask_contexts", "all"))
    support_mask_source = str(train_cfg.get("support_mask_source", "predicted"))
    benchmark_metrics = bool(train_cfg.get("report_benchmark_metrics", False))
    needs_lpips = float(train_cfg.get("lpips_loss_weight", 0.0)) > 0 or benchmark_metrics
    lpips_model = new_lpips(device) if needs_lpips else None
    fixed_episode = initial_episode
    resample = bool(train_cfg.get("resample_episodes", True))
    if configured_episode is not None and resample:
        raise ValueError("Explicit fixed episodes require train.resample_episodes: false")
    val_records = (
        val_dataset.fixed_episode_records(
            int(train_cfg.get("validation_episodes", 4)), int(config["seed"]) + 20_000
        )
        if val_dataset is not None
        else ()
    )
    best_full_key = "val_psnr" if val_records else "train_psnr"
    best_support_key = "val_support_psnr" if val_records else "support_psnr"
    best_full_psnr = max(
        (record.get(best_full_key, float("-inf")) for record in history),
        default=float("-inf"),
    )
    best_support_psnr = max(
        (record.get(best_support_key, float("-inf")) for record in history),
        default=float("-inf"),
    )
    for step in range(start_step, int(train_cfg["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_stats: list[dict[str, torch.Tensor]] = []
        outputs = None
        episode = None
        start_index = (step - 1) * scene_batch_size
        episodes = iter(
            _prefetched_episodes(
                train_dataset,
                start_index,
                scene_batch_size,
                resample,
                fixed_episode,
                episode_prefetch_workers,
            )
        )
        rejected_depth_gauge_episodes = 0
        rejected_empty_support_episodes = 0
        rejected_corrupt_depth_episodes = 0
        loaded_episodes: list[NvsEpisode] = []
        while len(loaded_episodes) < scene_batch_size:
            rejected_episodes = (
                rejected_depth_gauge_episodes
                + rejected_empty_support_episodes
                + rejected_corrupt_depth_episodes
            )
            try:
                try:
                    episode = next(episodes)
                except StopIteration:
                    episode = _sample_episode(
                        train_dataset,
                        start_index + scene_batch_size + rejected_episodes,
                        resample,
                        fixed_episode,
                    )
            except (CorruptDepthMapError, MissingDepthMapError) as error:
                if not isinstance(train_dataset, MultiSceneScanNetPP) or not resample:
                    raise
                rejected_corrupt_depth_episodes += 1
                if rejected_episodes + 1 > scene_batch_size:
                    raise RuntimeError("Too many sampled episodes are invalid") from error
                continue
            loaded_episodes.append(episode)

        for offset in range(0, scene_batch_size, backbone_batch_size):
            episode_group = loaded_episodes[offset : offset + backbone_batch_size]
            feature_group = _batched_backbone_features(backbone, episode_group, device)
            for episode, frozen_features in zip(episode_group, feature_group, strict=True):
                while True:
                    rejected_episodes = (
                        rejected_depth_gauge_episodes
                        + rejected_empty_support_episodes
                        + rejected_corrupt_depth_episodes
                    )
                    try:
                        episode_stats, outputs = _training_episode(
                            head=head,
                            backbone=backbone,
                            bridge=bridge,
                            episode=episode,
                            representation=representation,
                            device=device,
                            train_cfg=train_cfg,
                            lpips_model=lpips_model,
                            loss_scale=1.0 / scene_batch_size,
                            frozen_features=frozen_features,
                        )
                    except (InvalidDepthGaugeError, EmptySupportMaskError) as error:
                        if not isinstance(train_dataset, MultiSceneScanNetPP) or not resample:
                            raise
                        if isinstance(error, InvalidDepthGaugeError):
                            rejected_depth_gauge_episodes += 1
                        else:
                            rejected_empty_support_episodes += 1
                        if rejected_episodes + 1 > scene_batch_size:
                            raise RuntimeError("Too many sampled episodes are invalid") from error
                        while True:
                            try:
                                episode = _sample_episode(
                                    train_dataset,
                                    start_index + scene_batch_size + rejected_episodes + 1,
                                    resample,
                                    fixed_episode,
                                )
                                break
                            except (CorruptDepthMapError, MissingDepthMapError) as load_error:
                                rejected_corrupt_depth_episodes += 1
                                rejected_episodes += 1
                                if rejected_episodes + 1 > scene_batch_size:
                                    raise RuntimeError(
                                        "Too many sampled episodes are invalid"
                                    ) from load_error
                        frozen_features = None
                        continue
                    batch_stats.append(episode_stats)
                    break
        grad_norm = _clip_grad_norm_stable(
            head.parameters(),
            float(train_cfg.get("gradient_clip_norm", 1.0)),
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        metric_names = set().union(*(stats.keys() for stats in batch_stats))
        tensor_record = {
            name: torch.stack([stats[name] for stats in batch_stats if name in stats]).mean()
            for name in metric_names
        }
        tensor_record["grad_norm"] = grad_norm
        tensor_record["supervised_target_views"] = torch.stack(
            [stats["target_views"] for stats in batch_stats]
        ).sum()
        ordered_names = sorted(tensor_record)
        ordered_values = torch.stack(
            [
                tensor_record[name].detach().to(device=device, dtype=torch.float64)
                for name in ordered_names
            ]
        )
        record = dict(zip(ordered_names, ordered_values.cpu().tolist(), strict=True))
        record.update(
            {
                "step": float(step),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "scene_batch_size": float(scene_batch_size),
                "rejected_depth_gauge_episodes": float(rejected_depth_gauge_episodes),
                "rejected_empty_support_episodes": float(rejected_empty_support_episodes),
                "rejected_corrupt_depth_episodes": float(rejected_corrupt_depth_episodes),
            }
        )
        if val_records and step % int(train_cfg["validate_every"]) == 0:
            record.update(
                _validation(
                    val_records,
                    head=head,
                    backbone=backbone,
                    bridge=bridge,
                    representation=representation,
                    device=device,
                    support_context_mode=support_context_mode,
                    support_dilation=int(train_cfg.get("visibility_mask_dilation", 2)),
                    support_mask_source=support_mask_source,
                    lpips_model=lpips_model,
                    benchmark_metrics=benchmark_metrics,
                    output_dir=(
                        output_dir
                        if bool(train_cfg.get("validation_render_bundles", False))
                        else None
                    ),
                    step=step,
                )
            )
        history.append(record)
        full_psnr = record.get(best_full_key, float("-inf"))
        if full_psnr > best_full_psnr:
            best_full_psnr = full_psnr
            _atomic_save(
                _checkpoint_state(head, optimizer, scheduler, step, history, train_dataset, config),
                output_dir / "best_full.pt",
            )
        support_psnr = record.get(best_support_key, float("-inf"))
        if support_psnr > best_support_psnr:
            best_support_psnr = support_psnr
            _atomic_save(
                _checkpoint_state(head, optimizer, scheduler, step, history, train_dataset, config),
                output_dir / "best_support.pt",
            )
        diagnostic_every = int(train_cfg.get("diagnostic_render_every", 0))
        if (
            diagnostic_every > 0
            and step % diagnostic_every == 0
            and not bool(train_cfg.get("target_from_context", False))
            and episode is not None
            and outputs is not None
        ):
            _save_diagnostic_images(output_dir, step, episode, outputs)
        if step % int(train_cfg["log_every"]) == 0:
            print(json.dumps(record), flush=True)
        if step % int(train_cfg.get("checkpoint_every", train_cfg["validate_every"])) == 0:
            _atomic_save(
                _checkpoint_state(head, optimizer, scheduler, step, history, train_dataset, config),
                output_dir / "latest.pt",
            )
            (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))

    _atomic_save(
        _checkpoint_state(
            head, optimizer, scheduler, int(train_cfg["steps"]), history, train_dataset, config
        ),
        output_dir / "final.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))


def evaluate(
    config: dict[str, Any],
    data_root: Path,
    backbone_checkpoint: Path | None,
    model_checkpoint: Path,
    use_stub: bool,
    representation: str,
    output: Path,
) -> None:
    """Evaluate one trained head on every fixed manifest episode."""
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA because Power Foam uses Warp kernels")
    device = torch.device("cuda")
    data_cfg = config["data"]
    if str(data_cfg.get("dataset")) != "scannetpp":
        raise ValueError("Fixed benchmark evaluation currently requires ScanNet++")
    resolution = int(config["backbone"]["image_resolution"])
    target_pool = data_cfg.get("target_pool_size", 32)
    validation = MultiSceneScanNetPP(
        data_root,
        Path(data_cfg["scene_manifest"]),
        split="val",
        context_views=int(data_cfg["context_views"]),
        target_views=int(data_cfg.get("validation_target_views", data_cfg["target_views"])),
        image_resolution=resolution,
        target_pool_size=int(target_pool) if target_pool is not None else None,
        reserve_support_view=bool(data_cfg.get("reserve_support_view", False)),
        native_image_directory=str(
            data_cfg.get("native_image_directory", "resized_undistorted_images")
        ),
        resize_mode=str(data_cfg.get("resize_mode", "area")),
        load_depth=bool(data_cfg.get("load_depth", False)),
        seed=int(config["seed"]) + 10_000,
    )
    if validation.episode_entries is None:
        raise ValueError("Benchmark evaluation requires explicit fixed validation episodes")
    if use_stub:
        backbone = FrozenGeometryStub().to(device)
    else:
        if backbone_checkpoint is None:
            raise ValueError("VGGT-Ω checkpoint is required")
        backbone = FrozenVGGTOmega(backbone_checkpoint).to(device)
    backbone.eval()
    head = build_head(config, backbone.register_dim, representation, device)
    state = torch.load(model_checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(state["head"])
    head.eval()
    records = validation.all_episode_records()
    initial_episode = records[0][1]
    head_cfg = config["head"]
    if representation == "foam":
        bridge = PowerFoamRendererBridge(
            powerfoam_args(
                num_texel_sites=int(head_cfg["num_texel_sites"]),
                sv_dof=int(head_cfg["spherical_voronoi_dof"]),
                bkgd_color=tuple(config["renderer"]["bkgd_color"]),
                is_pinhole=bool(config["renderer"]["is_pinhole"]),
            ),
            camera_from_view(initial_episode.context[0], device),
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
    train_cfg = config["train"]
    metrics = _validation(
        records,
        head=head,
        backbone=backbone,
        bridge=bridge,
        representation=representation,
        device=device,
        support_context_mode=str(train_cfg.get("support_mask_contexts", "all")),
        support_dilation=int(train_cfg.get("visibility_mask_dilation", 2)),
        support_mask_source=str(train_cfg.get("support_mask_source", "predicted")),
        lpips_model=new_lpips(device),
        benchmark_metrics=True,
    )
    if int(metrics["val_renders"]) != len(validation.episode_entries):
        raise RuntimeError(
            "Exhaustive evaluation did not render every manifest episode exactly once"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--evaluation-output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--use-stub-backbone", action="store_true")
    parser.add_argument("--representation", choices=("foam", "gaussian"), default="foam")
    cli = parser.parse_args()
    config = _load_config(cli.config)
    if cli.evaluate_checkpoint is not None:
        evaluate(
            config,
            cli.data_root,
            cli.checkpoint,
            cli.evaluate_checkpoint,
            cli.use_stub_backbone,
            cli.representation,
            cli.evaluation_output,
        )
    else:
        train(
            config,
            cli.data_root,
            cli.checkpoint,
            cli.use_stub_backbone,
            cli.representation,
            cli.resume,
        )


if __name__ == "__main__":
    main()
