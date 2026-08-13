"""Calibrated projection helpers for canonical multi-context evidence fusion."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data.types import View
from .renderer import pinhole_ray_map_from_view


@dataclass(frozen=True)
class DepthAlignment:
    """Affine map from predicted VGGT z-depth into the calibrated scene gauge."""

    scale: torch.Tensor
    offset: torch.Tensor
    samples: int


@dataclass(frozen=True)
class CanonicalSupport:
    """One supporting view sampled at every canonical pixel."""

    maps: torch.Tensor
    patch_tokens: torch.Tensor
    grid: torch.Tensor


def _resize_map(values: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if values.ndim == 2:
        values = values[None, None]
    elif values.ndim == 3:
        values = values[None]
    if values.shape[-2:] != (height, width):
        values = F.interpolate(values, (height, width), mode="bilinear", align_corners=True)
    return values


def align_depths_to_calibrated_cameras(
    features: dict[str, torch.Tensor],
    context_views: tuple[View, ...],
) -> tuple[torch.Tensor, DepthAlignment]:
    """Scale VGGT depth into the calibrated scene gauge using camera baselines.

    VGGT-Ω predicts cameras and depth in one shared up-to-scale gauge. For two
    or more contexts, the ratio between calibrated and predicted camera-center
    baselines supplies one robust scale that is applied to every context depth.
    With one context there is no scale evidence, so the input depth is retained.
    """
    depths = features["depth"].float()
    if len(context_views) < 2:
        alignment = DepthAlignment(
            scale=torch.ones((), device=depths.device),
            offset=torch.zeros((), device=depths.device),
            samples=0,
        )
        return depths, alignment
    predicted_w2c = features["predicted_extrinsics"].float()
    homogeneous = torch.eye(4, device=depths.device, dtype=depths.dtype).expand(
        *predicted_w2c.shape[:-2], 4, 4
    ).clone()
    homogeneous[..., :3, :4] = predicted_w2c
    predicted_c2w = torch.linalg.inv(homogeneous)
    predicted_centers = predicted_c2w[0, : len(context_views), :3, 3]
    calibrated_centers = torch.stack(
        [view.c2w[:3, 3].to(depths) for view in context_views]
    )
    predicted_distances = torch.pdist(predicted_centers)
    calibrated_distances = torch.pdist(calibrated_centers)
    valid = (predicted_distances > 1e-6) & torch.isfinite(predicted_distances)
    ratios = calibrated_distances[valid] / predicted_distances[valid]
    scale = ratios.median() if ratios.numel() else torch.ones((), device=depths.device)
    scale = scale.clamp(1e-3, 1e3)
    aligned = (depths * scale).clamp_min(1e-3)
    return aligned, DepthAlignment(
        scale=scale,
        offset=torch.zeros((), device=depths.device),
        samples=int(ratios.numel()),
    )


def world_points_from_z_depth(
    view: View, depth: torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    """Lift camera-forward (pinhole z) depth into world coordinates."""
    height, width = view.image.shape[:2]
    depth = _resize_map(depth, height, width)[0, 0]
    rays = pinhole_ray_map_from_view(view, device)
    c2w = view.c2w.to(device=device, dtype=depth.dtype)
    forward = -c2w[:3, 2]
    ray_forward = (rays[..., 3:] * forward).sum(dim=-1).clamp_min(1e-6)
    distance = depth / ray_forward
    return rays[..., :3] + distance[..., None] * rays[..., 3:]


def project_world_points(
    points: torch.Tensor, view: View, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points into a centered OpenGL camera.

    Returns a grid in ``grid_sample`` coordinates, positive camera-forward
    depth, and an in-frustum validity mask.
    """
    c2w = view.c2w.to(device=device, dtype=points.dtype)
    camera = torch.einsum("ij,...j->...i", c2w[:3, :3].T, points - c2w[:3, 3])
    z_depth = -camera[..., 2]
    height, width = view.image.shape[:2]
    aspect = width / height
    half_width = torch.tan(
        torch.tensor(view.fov_x_radians / 2, device=device, dtype=points.dtype)
    )
    half_height = half_width / aspect
    safe_depth = z_depth.clamp_min(1e-6)
    grid_x = camera[..., 0] / (safe_depth * half_width)
    grid_y = -camera[..., 1] / (safe_depth * half_height)
    grid = torch.stack([grid_x, grid_y], dim=-1)
    epsilon = 1e-5
    valid = (
        (z_depth > 1e-6)
        & (grid_x >= -1 - epsilon)
        & (grid_x <= 1 + epsilon)
        & (grid_y >= -1 - epsilon)
        & (grid_y <= 1 + epsilon)
    )
    return grid, z_depth, valid


def _sample(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(
        values,
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def build_canonical_support(
    images: torch.Tensor,
    features: dict[str, torch.Tensor],
    context_views: tuple[View, ...],
    device: torch.device | str,
    *,
    relative_depth_tolerance: float = 0.05,
    absolute_depth_tolerance: float = 0.02,
) -> CanonicalSupport | None:
    """Project context view 2 onto context view 1's depth surface."""
    if len(context_views) < 2:
        return None
    canonical, support = context_views[:2]
    height, width = canonical.image.shape[:2]
    canonical_depth = _resize_map(features["depth"][:, 0], height, width)
    points = world_points_from_z_depth(canonical, canonical_depth, device)
    grid, projected_depth, in_frustum = project_world_points(points, support, device)

    support_rgb = _sample(images[:, 1], grid)
    support_depth = _sample(features["depth"][:, 1], grid)
    support_conf = _sample(features["depth_conf"][:, 1], grid)
    depth_residual = support_depth[:, 0] - projected_depth[None]
    tolerance = absolute_depth_tolerance + relative_depth_tolerance * support_depth[:, 0].abs()
    depth_consistent = depth_residual.abs() <= tolerance
    valid = in_frustum[None] & (support_depth[:, 0] > 0) & depth_consistent

    canonical_c2w = canonical.c2w.to(device=device, dtype=points.dtype)
    support_c2w = support.c2w.to(device=device, dtype=points.dtype)
    canonical_view = F.normalize(canonical_c2w[:3, 3] - points, dim=-1)
    support_view = F.normalize(support_c2w[:3, 3] - points, dim=-1)
    view_cosine = (canonical_view * support_view).sum(dim=-1)[None]
    normalized_residual = depth_residual / support_depth[:, 0].abs().clamp_min(1e-3)
    maps = torch.cat(
        [
            support_rgb,
            support_conf,
            normalized_residual[:, None],
            view_cosine[:, None],
            valid[:, None].to(support_rgb.dtype),
        ],
        dim=1,
    )
    maps = maps * valid[:, None]
    return CanonicalSupport(
        maps=maps,
        patch_tokens=features["patch_tokens"][:, 1],
        grid=grid[None],
    )


def projected_context_support_mask(
    context_views: tuple[View, ...],
    depths: torch.Tensor,
    target_view: View,
    device: torch.device | str,
    *,
    dilation: int = 2,
) -> torch.Tensor:
    """Approximate target pixels supported by any context depth surface."""
    height, width = target_view.image.shape[:2]
    mask = torch.zeros(height * width, device=device, dtype=torch.float32)
    for index, context in enumerate(context_views):
        points = world_points_from_z_depth(context, depths[:, index], device)
        grid, _, valid = project_world_points(points, target_view, device)
        x = ((grid[..., 0] + 1) * 0.5 * (width - 1)).round().long()
        y = ((grid[..., 1] + 1) * 0.5 * (height - 1)).round().long()
        valid = valid & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        indices = y[valid] * width + x[valid]
        mask.scatter_(0, indices, 1.0)
    mask = mask.reshape(1, 1, height, width)
    if dilation > 0:
        kernel = 2 * dilation + 1
        mask = F.max_pool2d(mask, kernel, stride=1, padding=dilation)
    return mask[0, 0].bool()
