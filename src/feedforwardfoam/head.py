from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn.functional as F
from torch import nn


def inverse_softplus(value: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    """Convert a positive physical value to Power Foam's raw parameter domain."""
    scaled = beta * value
    return torch.where(
        scaled > 20.0,
        value,
        torch.log(torch.expm1(scaled)) / beta,
    )


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
    ) -> None:
        super().__init__()
        if radius_mode not in {"learned_absolute", "pixel_footprint"}:
            raise ValueError(f"Unknown radius mode: {radius_mode}")
        if density_mode not in {"learned", "fixed", "source_alpha_fixed"}:
            raise ValueError(f"Unknown density mode: {density_mode}")
        if base_depth_mode not in {"predicted", "constant"}:
            raise ValueError(f"Unknown base depth mode: {base_depth_mode}")
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
        self.local = nn.Sequential(
            nn.Conv2d(5, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.register_projection = nn.Linear(register_dim, hidden_dim)
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
            depth = F.interpolate(depth, size=image.shape[-2:], mode="bilinear", align_corners=False)
            confidence = F.interpolate(confidence, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return image, depth, confidence

    def forward(
        self,
        images: torch.Tensor,
        frozen_features: dict[str, torch.Tensor],
        canonical_ray_map: torch.Tensor,
        canonical_alpha: torch.Tensor | None = None,
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
        h, w = local.shape[-2:]
        tokens = local.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        logits = self.decode(tokens)[0]

        # Deterministic top-M is a P0 budget mechanism; later P1 replaces it
        # with coverage-aware multi-view proposal fusion.
        m = min(self.max_cells, logits.shape[0])
        selected = logits[:, 9].topk(m, sorted=False).indices
        values = logits[selected]
        ray_map = canonical_ray_map.to(device=values.device, dtype=values.dtype).reshape(-1, 6)
        if ray_map.shape[0] != h * w:
            ray_map = F.interpolate(
                canonical_ray_map.permute(2, 0, 1)[None].to(values.dtype),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0).reshape(-1, 6)
        rays = ray_map[selected]
        depth = depth.clamp_min(1e-3)
        if self.base_depth_mode == "constant":
            depth = torch.full_like(depth, self.constant_base_depth)

        ray_directions = ray_map[:, 3:].reshape(h, w, 3)
        base_points = ray_map[:, :3].reshape(h, w, 3) + depth[0, 0, ..., None] * ray_directions
        base_points_selected = base_points.reshape(-1, 3)[selected]
        point_residual = self.point_residual_scale * torch.tanh(values[:, :3])
        points = base_points_selected + point_residual
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
                self.radius_residual_log_scale * torch.tanh(values[:, 3])
            )
            physical_radii = footprint.reshape(-1)[selected].clamp_min(1e-4) * scale
        else:
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
        if self.initialize_rgb_from_image:
            source_rgb = image[0].permute(1, 2, 0).reshape(-1, 3)[selected]
            rgb = source_rgb[:, None, None] - 0.5 + rgb_residual
        else:
            rgb = rgb_residual

        # Keep full upstream shapes. P0 constrains, rather than removes, 2D
        # surface texture: all sites are tied at the dipole and height is zero.
        texel_sites = torch.zeros(m, self.num_texel_sites, 2, device=values.device, dtype=values.dtype)
        texel_height = torch.zeros(m, self.num_texel_sites, device=values.device, dtype=values.dtype)
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
