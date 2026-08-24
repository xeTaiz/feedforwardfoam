from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn.functional as F
from torch import nn


def inverse_softplus(value: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    """Convert a positive physical value to Power Foam's raw parameter domain."""
    scaled = beta * value
    return value + torch.log(-torch.expm1(-scaled)) / beta


def quaternions_from_positive_x(normals: torch.Tensor) -> torch.Tensor:
    """Return wxyz rotations mapping Power Foam's local +X normal to ``normals``."""
    normal = F.normalize(normals, dim=-1, eps=1e-6)
    quaternion = torch.stack(
        [
            1.0 + normal[..., 0],
            torch.zeros_like(normal[..., 0]),
            normal[..., 2],
            -normal[..., 1],
        ],
        dim=-1,
    )
    opposite = normal[..., 0] < -1.0 + 1e-6
    fallback = torch.zeros_like(quaternion)
    fallback[..., 2] = 1.0
    quaternion = torch.where(opposite[..., None], fallback, quaternion)
    return F.normalize(quaternion, dim=-1, eps=1e-6)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compose wxyz quaternions, applying ``right`` before ``left``."""
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )


def depth_normals(points: torch.Tensor, ray_directions: torch.Tensor) -> torch.Tensor:
    """Estimate camera-facing world normals from an H×W depth-lifted point map."""
    dx = torch.empty_like(points)
    dy = torch.empty_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dx[:, 0] = points[:, 1] - points[:, 0]
    dx[:, -1] = points[:, -1] - points[:, -2]
    dy[1:-1] = points[2:] - points[:-2]
    dy[0] = points[1] - points[0]
    dy[-1] = points[-1] - points[-2]
    normals = torch.linalg.cross(dy, dx, dim=-1)
    valid = torch.linalg.vector_norm(normals, dim=-1, keepdim=True) > 1e-8
    normals = F.normalize(normals, dim=-1, eps=1e-6)
    fallback = -ray_directions
    normals = torch.where(valid, normals, fallback)
    toward_camera = -ray_directions
    flip = (normals * toward_camera).sum(dim=-1, keepdim=True) < 0
    return F.normalize(torch.where(flip, -normals, normals), dim=-1, eps=1e-6)


@dataclass(frozen=True)
class FoamParameters:
    """Differentiable tensors matching PowerfoamScene's parameter names."""

    points: torch.Tensor
    radii: torch.Tensor
    quaternions: torch.Tensor
    density: torch.Tensor
    texel_sites: torch.Tensor
    texel_sv_axis: torch.Tensor
    texel_sv_rgb: torch.Tensor
    texel_height: torch.Tensor

    def as_upstream_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "points": self.points,
            "radii": self.radii,
            "quaternions": self.quaternions,
            "density": self.density,
            "texel_sites": self.texel_sites,
            "texel_sv_axis": self.texel_sv_axis,
            "texel_sv_rgb": self.texel_sv_rgb,
            "texel_height": self.texel_height,
        }


def concatenate_foam_parameters(parameters: list[FoamParameters]) -> FoamParameters:
    if not parameters:
        raise ValueError("Cannot concatenate an empty Foam parameter list")
    return FoamParameters(
        **{
            name: torch.cat([getattr(item, name) for item in parameters], dim=0)
            for name in parameters[0].as_upstream_tensors()
        }
    )


def select_foam_parameters(parameters: FoamParameters, indices: torch.Tensor) -> FoamParameters:
    return FoamParameters(
        **{name: value[indices] for name, value in parameters.as_upstream_tensors().items()}
    )


def uniform_selection_indices(count: int, budget: int, device: torch.device) -> torch.Tensor:
    """Return a deterministic evenly strided subset of ``count`` items."""
    if budget <= 0:
        raise ValueError("Selection budget must be positive")
    selected = min(budget, count)
    return torch.div(torch.arange(selected, device=device) * count, selected, rounding_mode="floor")


def farthest_point_indices(points: torch.Tensor, budget: int, start_index: int = 0) -> torch.Tensor:
    """Select a deterministic farthest-point subset of world points.

    Voxel selection keeps one member per occupied cell and therefore drops
    isolated proposals such as depth-edge anchors. Farthest-point sampling
    maximizes spatial spread instead, so thin structure survives reduction.
    """
    count = points.shape[0]
    if budget <= 0:
        raise ValueError("Proposal budget must be positive")
    if count <= budget:
        return torch.arange(count, device=points.device)
    detached = points.detach()
    selected = torch.empty(budget, dtype=torch.long, device=points.device)
    squared = torch.full((count,), float("inf"), device=points.device, dtype=detached.dtype)
    current = int(start_index)
    for position in range(budget):
        selected[position] = current
        squared = torch.minimum(squared, (detached - detached[current]).square().sum(dim=-1))
        # Negative marking keeps an already chosen point out of every later argmax.
        squared[current] = -1.0
        current = int(torch.argmax(squared))
    return selected


def incremental_containment_indices(
    points: torch.Tensor,
    raw_radii: torch.Tensor,
    group_sizes: list[int],
    *,
    criterion: str = "power",
    tolerance: float = 1.0,
    beta: float = 100.0,
    chunk: int = 4096,
) -> torch.Tensor:
    """Keep one context's proposals whole, then add only non-redundant others.

    Every site of the first group survives. A later-group site is dropped when
    an already kept site already claims its centre, so the merge removes
    duplicates rather than distinct geometry and the surviving count floats
    with the scene instead of hitting a fixed budget.

    ``criterion`` selects the containment test against a kept site ``i``:

    - ``power``: ``|p_j - p_i|^2 + r_j^2 <= (tolerance * r_i)^2``. This is
      exactly the condition under which ``p_j`` is not inside ``j``'s own power
      cell, so the new cell would collapse into a sliver.
    - ``ball``: ``|p_j - p_i| <= tolerance * r_i``, the plain sphere test.

    ``tolerance`` scales the incumbent radius; ``0`` keeps every proposal and
    reproduces unreduced concatenation, ``1`` applies the exact test.

    ``raw_radii`` are upstream-domain radii as carried by ``FoamParameters``;
    the physical extent used for the test is recovered here with the softplus
    that ``inverse_softplus`` inverts. Sites within one group are never tested
    against each other because a single view's proposals are a regular pixel
    lattice with no duplicates.
    """
    if criterion not in {"power", "ball"}:
        raise ValueError(f"Unknown containment criterion: {criterion}")
    if tolerance < 0.0:
        raise ValueError("Containment tolerance must be non-negative")
    if sum(group_sizes) != points.shape[0]:
        raise ValueError("Group sizes must cover every proposal exactly")
    device = points.device
    detached_points = points.detach()
    physical_radii = F.softplus(raw_radii.detach(), beta=beta).reshape(-1)
    bounds = []
    start = 0
    for size in group_sizes:
        bounds.append((start, start + size))
        start += size

    first_begin, first_end = bounds[0]
    kept = [torch.arange(first_begin, first_end, device=device)]
    kept_points = detached_points[first_begin:first_end]
    kept_radii = physical_radii[first_begin:first_end]
    for begin, end in bounds[1:]:
        if end <= begin:
            continue
        candidates = torch.arange(begin, end, device=device)
        candidate_points = detached_points[begin:end]
        candidate_radii = physical_radii[begin:end]
        limit = (tolerance * kept_radii).square()
        survives = torch.ones(candidates.shape[0], dtype=torch.bool, device=device)
        for offset in range(0, candidates.shape[0], chunk):
            block = slice(offset, min(offset + chunk, candidates.shape[0]))
            squared = torch.cdist(candidate_points[block], kept_points).square()
            if criterion == "power":
                squared = squared + candidate_radii[block, None].square()
            survives[block] = ~(squared <= limit[None, :]).any(dim=1)
        selected = candidates[survives]
        kept.append(selected)
        kept_points = torch.cat([kept_points, detached_points[selected]])
        kept_radii = torch.cat([kept_radii, physical_radii[selected]])
    return torch.cat(kept)


def voxel_budget_indices(
    points: torch.Tensor,
    budget: int,
    iterations: int = 12,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select a deterministic approximately voxel-uniform subset of world points.

    Without ``scores`` an occupied voxel is represented by its lowest-index
    member, which biases selection toward whichever context view was
    concatenated first. With ``scores`` the highest-scoring member represents
    the voxel instead. The voxel grid, the budget bisection, and the trim/fill
    steps are identical either way, so ``scores`` changes only which member of
    a voxel survives.
    """
    count = points.shape[0]
    if budget <= 0:
        raise ValueError("Proposal budget must be positive")
    if count <= budget:
        return torch.arange(count, device=points.device)
    detached = points.detach()
    score_order: torch.Tensor | None = None
    if scores is not None:
        if scores.shape[0] != count:
            raise ValueError("Proposal scores must match the proposal count")
        score_order = torch.argsort(-scores.detach().reshape(-1), stable=True)
    extent = (detached.amax(dim=0) - detached.amin(dim=0)).amax().clamp_min(1e-6)
    low = extent / 4096.0
    high = extent
    best = torch.arange(count, device=points.device)
    best_error = count
    for _ in range(iterations):
        width = (low + high) * 0.5
        coordinates = torch.floor((detached - detached.amin(dim=0)) / width).long()
        _, inverse = torch.unique(coordinates, dim=0, return_inverse=True)
        if score_order is None:
            order = torch.argsort(inverse, stable=True)
        else:
            order = score_order[torch.argsort(inverse[score_order], stable=True)]
        sorted_inverse = inverse[order]
        first = torch.ones_like(sorted_inverse, dtype=torch.bool)
        first[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
        representatives = order[first]
        error = abs(representatives.numel() - budget)
        if error < best_error:
            best = representatives
            best_error = error
        if representatives.numel() > budget:
            low = width
        else:
            high = width
    if best.numel() > budget:
        positions = torch.div(
            torch.arange(budget, device=points.device) * best.numel(), budget, rounding_mode="floor"
        )
        return best[positions]
    if best.numel() < budget:
        retained = torch.zeros(count, dtype=torch.bool, device=points.device)
        retained[best] = True
        remaining = torch.nonzero(~retained, as_tuple=False)[:, 0]
        needed = budget - best.numel()
        positions = torch.div(
            torch.arange(needed, device=points.device) * remaining.numel(),
            needed,
            rounding_mode="floor",
        )
        best = torch.cat([best, remaining[positions]])
    return best


class CanonicalPowerFoamHead(nn.Module):
    """P0 canonical-patch decoder for one full Power Foam scene.

    Geometry anchors are emitted only by the canonical view. Other views affect
    its predictions through frozen VGGT-Ω register features, so this module
    creates one power diagram and never merges independently formed foams.
    """

    def __init__(
        self,
        *,
        register_dim: int,
        hidden_dim: int = 256,
        max_cells: int = 1024,
        num_texel_sites: int = 8,
        spherical_voronoi_dof: int = 8,
        radius_mode: str = "learned_absolute",
        radius_scale_init: float = 1.5,
        radius_residual_log_scale: float = 0.25,
        density_mode: str = "learned",
        fixed_density: float = 100.0,
        initialize_rgb_from_image: bool = False,
        initialize_normals_from_depth: bool = True,
        base_depth_mode: str = "predicted",
        constant_base_depth: float = 2.0,
        point_residual_scale: float = 0.05,
        normal_residual_radians: float = 0.25,
        rgb_residual_scale: float = 0.5,
        fusion_mode: str = "none",
        patch_token_dim: int | None = None,
        prediction_mode: str = "residual",
        enable_point_residual: bool = True,
        enable_radius_residual: bool = True,
        enable_orientation_residual: bool = True,
        enable_rgb_residual: bool = True,
        proposal_views: str = "canonical",
        proposal_reduction: str = "none",
        selection_mode: str = "gate",
        proposal_containment: str = "power",
        proposal_containment_tolerance: float = 1.0,
    ) -> None:
        super().__init__()
        if radius_mode not in {"learned_absolute", "pixel_footprint"}:
            raise ValueError(f"Unknown radius mode: {radius_mode}")
        if density_mode not in {"learned", "fixed", "source_alpha_fixed"}:
            raise ValueError(f"Unknown density mode: {density_mode}")
        if base_depth_mode not in {"predicted", "constant"}:
            raise ValueError(f"Unknown base depth mode: {base_depth_mode}")
        if fusion_mode not in {"none", "projected"}:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        if prediction_mode not in {"residual", "absolute", "initialization"}:
            raise ValueError(f"Unknown prediction mode: {prediction_mode}")
        if proposal_views not in {"canonical", "all"}:
            raise ValueError(f"Unknown proposal views: {proposal_views}")
        if proposal_reduction not in {
            "none",
            "all",
            "balanced",
            "voxel",
            "fps",
            "confidence_voxel",
            "incremental",
        }:
            raise ValueError(f"Unknown proposal reduction: {proposal_reduction}")
        if selection_mode not in {"gate", "uniform"}:
            raise ValueError(f"Unknown selection mode: {selection_mode}")
        if proposal_containment not in {"power", "ball"}:
            raise ValueError(f"Unknown proposal containment: {proposal_containment}")
        if proposal_containment_tolerance < 0.0:
            raise ValueError("Proposal containment tolerance must be non-negative")
        if fusion_mode == "projected" and patch_token_dim is None:
            raise ValueError("Projected fusion requires patch_token_dim")
        self.max_cells = max_cells
        self.num_texel_sites = num_texel_sites
        self.spherical_voronoi_dof = spherical_voronoi_dof
        self.radius_mode = radius_mode
        self.radius_scale_init = radius_scale_init
        self.radius_residual_log_scale = radius_residual_log_scale
        self.density_mode = density_mode
        self.fixed_density = fixed_density
        self.initialize_rgb_from_image = initialize_rgb_from_image
        self.initialize_normals_from_depth = initialize_normals_from_depth
        self.base_depth_mode = base_depth_mode
        self.constant_base_depth = constant_base_depth
        self.point_residual_scale = point_residual_scale
        self.normal_residual_radians = normal_residual_radians
        self.rgb_residual_scale = rgb_residual_scale
        self.fusion_mode = fusion_mode
        self.prediction_mode = prediction_mode
        self.enable_point_residual = enable_point_residual
        self.enable_radius_residual = enable_radius_residual
        self.enable_orientation_residual = enable_orientation_residual
        self.enable_rgb_residual = enable_rgb_residual
        self.proposal_views = proposal_views
        self.proposal_reduction = proposal_reduction
        self.selection_mode = selection_mode
        self.proposal_containment = proposal_containment
        self.proposal_containment_tolerance = float(proposal_containment_tolerance)
        self._proposal_index_cache: dict[tuple, torch.Tensor] = {}
        self.local = nn.Sequential(
            nn.Conv2d(5, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.register_projection = nn.Linear(register_dim, hidden_dim)
        self.support_map_projection = (
            nn.Sequential(
                nn.Conv2d(7, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            )
            if fusion_mode == "projected"
            else None
        )
        self.support_token_projection = (
            nn.Conv2d(int(patch_token_dim), hidden_dim, 1) if fusion_mode == "projected" else None
        )
        # point residual, radius, quaternion, density, confidence/gate,
        # spherical axes, and spherical RGB values.
        self.output_dim = 3 + 1 + 4 + 1 + 1 + 2 * num_texel_sites * spherical_voronoi_dof * 3
        self.decode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self.output_dim)
        )
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        """Start with visible, compact cells rather than an empty renderer."""
        output = self.decode[-1]
        assert isinstance(output, nn.Linear)
        with torch.no_grad():
            # Retain the learned gate row so top-M selects spatially varying anchors.
            output.weight[:9].zero_()
            output.weight[10:].zero_()
            output.bias.zero_()
            # Geometry and appearance rows are residuals around the physical
            # depth/ray/footprint/source-RGB initialization.
            output.bias[3] = 0.0
            output.bias[4] = 1.0  # identity residual quaternion (wxyz)
            # Density is already in Power Foam's raw softplus domain.
            output.bias[8] = 0.1

    @staticmethod
    def _canonical_maps(
        images: torch.Tensor, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inputs: images [B,V,3,H,W], depth/confidence [B,V,1,H,W] or compatible.
        image = images[:, 0]
        depth = features["depth"][:, 0]
        confidence = features["depth_conf"][:, 0]
        if depth.ndim == 3:
            depth = depth[:, None]
        if confidence.ndim == 3:
            confidence = confidence[:, None]
        if depth.shape[-2:] != image.shape[-2:]:
            depth = F.interpolate(
                depth, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
            confidence = F.interpolate(
                confidence, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        return image, depth, confidence

    def forward(
        self,
        images: torch.Tensor,
        frozen_features: dict[str, torch.Tensor],
        canonical_ray_map: torch.Tensor,
        canonical_alpha: torch.Tensor | None = None,
        canonical_support=None,
        canonical_base_points: torch.Tensor | None = None,
        max_cells_override: int | None = None,
    ) -> FoamParameters:
        """Decode a batch-size-one canonical scene.

        `canonical_ray_map` is H×W×6 in the target world coordinate frame.
        Batch size one is deliberate for Power Foam's variable Čech graph.
        """
        if images.shape[0] != 1:
            raise ValueError("P0 renderer currently accepts one scene per batch")
        image, depth, confidence = self._canonical_maps(images, frozen_features)
        registers = frozen_features["registers"]
        register_feature = self.register_projection(registers.mean(dim=(1, 2)))
        local = self.local(torch.cat([image, depth, confidence], dim=1))
        local = local + register_feature[:, :, None, None]
        if self.fusion_mode == "projected":
            if canonical_support is None:
                raise ValueError("Projected fusion requires canonical support evidence")
            assert self.support_map_projection is not None
            assert self.support_token_projection is not None
            support_maps = F.interpolate(
                canonical_support.maps,
                size=local.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            support_tokens = self.support_token_projection(canonical_support.patch_tokens)
            support_tokens = F.grid_sample(
                support_tokens,
                canonical_support.grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            local = local + self.support_map_projection(support_maps) + support_tokens
        h, w = local.shape[-2:]
        tokens = local.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        logits = self.decode(tokens)[0]

        # Deterministic top-M is a P0 budget mechanism; later P1 replaces it
        # with coverage-aware multi-view proposal fusion.
        cell_budget = self.max_cells if max_cells_override is None else max_cells_override
        m = min(cell_budget, logits.shape[0])
        # Initialization must not depend on random decoder gate weights.
        if self.selection_mode == "uniform" or self.prediction_mode == "initialization":
            selected = uniform_selection_indices(logits.shape[0], m, logits.device)
        else:
            selected = logits[:, 9].topk(m, sorted=False).indices
        values = logits[selected]
        # A decoder-free physical baseline. Selection is deterministic and uses
        # every pixel when max_cells equals H*W, as in the overfit protocol.
        if self.prediction_mode == "initialization":
            # Retain a zero-gradient graph so the common training loop can
            # evaluate this baseline without a special backward path.
            values = values * 0.0
            values[:, 4] = values[:, 4] + 1.0
        ray_map = canonical_ray_map.to(device=values.device, dtype=values.dtype).reshape(-1, 6)
        if ray_map.shape[0] != h * w:
            ray_map = (
                F.interpolate(
                    canonical_ray_map.permute(2, 0, 1)[None].to(values.dtype),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                .permute(1, 2, 0)
                .reshape(-1, 6)
            )
        rays = ray_map[selected]
        depth = depth.clamp_min(1e-3)
        if self.base_depth_mode == "constant":
            depth = torch.full_like(depth, self.constant_base_depth)

        ray_directions = ray_map[:, 3:].reshape(h, w, 3)
        if canonical_base_points is None:
            base_points = ray_map[:, :3].reshape(h, w, 3) + depth[0, 0, ..., None] * ray_directions
        else:
            base_points = canonical_base_points.to(values).permute(2, 0, 1)[None]
            if base_points.shape[-2:] != (h, w):
                base_points = F.interpolate(
                    base_points, size=(h, w), mode="bilinear", align_corners=True
                )
            base_points = base_points[0].permute(1, 2, 0)
        base_points_selected = base_points.reshape(-1, 3)[selected]
        point_residual = self.point_residual_scale * torch.tanh(values[:, :3])
        points = (
            base_points_selected + point_residual
            if self.prediction_mode == "residual" and self.enable_point_residual
            else base_points_selected
        )
        if self.radius_mode == "pixel_footprint":
            dx = torch.linalg.vector_norm(ray_directions[:, 1:] - ray_directions[:, :-1], dim=-1)
            dy = torch.linalg.vector_norm(ray_directions[1:] - ray_directions[:-1], dim=-1)
            footprint = torch.zeros(h, w, device=values.device, dtype=values.dtype)
            counts = torch.zeros_like(footprint)
            footprint[:, :-1] += dx
            footprint[:, 1:] += dx
            counts[:, :-1] += 1
            counts[:, 1:] += 1
            footprint[:-1] += dy
            footprint[1:] += dy
            counts[:-1] += 1
            counts[1:] += 1
            footprint = depth[0, 0] * footprint / counts.clamp_min(1)
            scale = self.radius_scale_init * torch.exp(
                self.radius_residual_log_scale
                * torch.tanh(
                    values[:, 3] if self.enable_radius_residual else torch.zeros_like(values[:, 3])
                )
            )
            physical_radii = footprint.reshape(-1)[selected].clamp_min(1e-4) * scale
        else:
            physical_radii = 0.05 * torch.exp(0.5 * torch.tanh(values[:, 3]))
        if self.prediction_mode == "absolute":
            # Direct physical radius prediction: no footprint or source-scale prior.
            physical_radii = 0.05 * torch.exp(0.5 * torch.tanh(values[:, 3]))
        radii = inverse_softplus(physical_radii)

        if self.initialize_normals_from_depth:
            base_normals = depth_normals(base_points, ray_directions).reshape(-1, 3)[selected]
        else:
            base_normals = -rays[:, 3:]
        base_quaternion = quaternions_from_positive_x(base_normals)
        # Bounded residual quaternion with a nonzero first derivative at the
        # identity; axis-angle normalization at a zero vector would be dead.
        residual_xyz = (
            math.tan(0.5 * self.normal_residual_radians)
            / math.sqrt(3.0)
            * torch.tanh(values[:, 5:8])
        )
        residual_quaternion = F.normalize(
            torch.cat([torch.ones_like(values[:, 4:5]), residual_xyz], dim=-1),
            dim=-1,
            eps=1e-6,
        )
        quaternion = F.normalize(
            quaternion_multiply(residual_quaternion, base_quaternion), dim=-1, eps=1e-6
        )
        if not self.enable_orientation_residual:
            quaternion = base_quaternion
        if self.prediction_mode == "absolute":
            quaternion = F.normalize(values[:, 4:8], dim=-1, eps=1e-6)
        if self.density_mode == "fixed":
            density = torch.full_like(values[:, 8], self.fixed_density)
        elif self.density_mode == "source_alpha_fixed":
            if canonical_alpha is None:
                raise ValueError("source_alpha_fixed density requires a canonical alpha mask")
            alpha_selected = canonical_alpha.to(values).reshape(-1)[selected]
            density = torch.where(
                alpha_selected > 0.5,
                torch.full_like(alpha_selected, self.fixed_density),
                torch.full_like(alpha_selected, -1.0),
            )
        else:
            density = 0.1 + F.softplus(values[:, 8], beta=10)

        cursor = 10
        count = self.num_texel_sites * self.spherical_voronoi_dof * 3
        axis = values[:, cursor : cursor + count].reshape(
            m, self.num_texel_sites, self.spherical_voronoi_dof, 3
        )
        cursor += count
        # Non-zero canonical directions make spherical Voronoi valid from step 0.
        canonical_axes = torch.eye(3, device=values.device, dtype=values.dtype).repeat(
            (self.spherical_voronoi_dof + 2) // 3, 1
        )[: self.spherical_voronoi_dof]
        axis = F.normalize(axis + canonical_axes[None, None], dim=-1, eps=1e-6)
        rgb_logits = values[:, cursor : cursor + count].reshape(
            m, self.num_texel_sites, self.spherical_voronoi_dof, 3
        )
        rgb_residual = self.rgb_residual_scale * torch.tanh(rgb_logits)
        if not self.enable_rgb_residual:
            rgb_residual = torch.zeros_like(rgb_residual)
        if self.prediction_mode == "absolute":
            # Upstream adds +0.5 after spherical-Voronoi interpolation.
            rgb = 0.5 * torch.tanh(rgb_logits)
        elif self.initialize_rgb_from_image:
            source_rgb = image[0].permute(1, 2, 0).reshape(-1, 3)[selected]
            rgb = source_rgb[:, None, None] - 0.5 + rgb_residual
        else:
            rgb = rgb_residual

        # Keep full upstream shapes. P0 constrains, rather than removes, 2D
        # surface texture: all sites are tied at the dipole and height is zero.
        texel_sites = torch.zeros(
            m, self.num_texel_sites, 2, device=values.device, dtype=values.dtype
        )
        texel_height = torch.zeros(
            m, self.num_texel_sites, device=values.device, dtype=values.dtype
        )
        return FoamParameters(
            points=points,
            radii=radii,
            quaternions=quaternion,
            density=density,
            texel_sites=texel_sites,
            texel_sv_axis=axis.reshape(m, self.num_texel_sites, -1),
            texel_sv_rgb=rgb.reshape(m, self.num_texel_sites, -1),
            texel_height=texel_height,
        )
