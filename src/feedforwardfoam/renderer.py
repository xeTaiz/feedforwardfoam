"""Thin, non-forking bridge to the pinned Power Foam subrepository."""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .data.types import View
    from .head import FoamParameters


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_powerfoam() -> None:
    """Make the pinned subrepository importable without copying upstream code."""
    source = _repo_root() / "external" / "powerfoam"
    if not source.exists():
        raise RuntimeError("Missing external/powerfoam; run git submodule update --init --recursive")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        importlib.import_module("powerfoam.scene")
    except ImportError as error:
        raise RuntimeError(
            "Power Foam dependencies are unavailable. Run scripts/bootstrap_powerfoam_env.sh "
            "on a CUDA machine."
        ) from error


def powerfoam_args(
    *,
    num_texel_sites: int = 8,
    sv_dof: int = 8,
    bkgd_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    is_pinhole: bool = True,
) -> SimpleNamespace:
    """Only renderer fields required by the upstream scene/rasterizer bridge."""
    return SimpleNamespace(
        num_texel_sites=num_texel_sites,
        sv_dof=sv_dof,
        bkgd_color=list(bkgd_color),
        is_pinhole=is_pinhole,
        render_objective="volume",
        disable_coop_prim_load=False,
        disable_coop_adj_load=False,
    )


def pinhole_ray_map_from_view(view: "View", device: torch.device | str) -> torch.Tensor:
    """Return H×W world-space pinhole origins/directions without an upstream dependency."""
    c2w = view.c2w.to(device=device, dtype=torch.float32)
    height, width = view.image.shape[:2]
    aspect = width / height
    half_width = torch.tan(torch.tensor(view.fov_x_radians / 2, device=device))
    half_height = half_width / aspect
    x = torch.linspace(-1.0, 1.0, width, device=device)
    y = torch.linspace(1.0, -1.0, height, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    # Blender/OpenGL's visible forward direction is -Z.
    directions = (
        xx[..., None] * c2w[:3, 0] * half_width
        + yy[..., None] * c2w[:3, 1] * half_height
        - c2w[:3, 2]
    )
    directions = torch.nn.functional.normalize(directions, dim=-1)
    origins = c2w[:3, 3].expand_as(directions)
    return torch.cat([origins, directions], dim=-1)


def camera_from_view(view: "View", device: torch.device | str):
    """Convert a normalised c2w pose and horizontal FoV to Power Foam's camera."""
    require_powerfoam()
    from powerfoam.camera import TorchCamera

    c2w = view.c2w.to(device=device, dtype=torch.float32)
    height, width = view.image.shape[:2]
    aspect = width / height
    half_width = torch.tan(torch.tensor(view.fov_x_radians / 2, device=device))
    half_height = half_width / aspect
    # Blender/OpenGL camera forward is -Z; TorchCamera uses cross(up, right).
    return TorchCamera(
        eye=c2w[:3, 3],
        right=c2w[:3, 0] * half_width,
        up=c2w[:3, 1] * half_height,
        width=width,
        height=height,
    )


@dataclass
class FoamRender:
    rgb: torch.Tensor
    alpha: torch.Tensor
    normal: torch.Tensor
    depth: torch.Tensor


class PowerFoamRendererBridge:
    """Build one transient upstream scene directly from head output tensors.

    The tensors are deliberately *not* wrapped in ``nn.Parameter``: wrapping
    would detach them from the feed-forward head. Čech membership is rebuilt
    from detached geometry, while the upstream rasterizer still propagates
    gradients through positions, radii, dipoles, texels, density, and radiance.
    """

    def __init__(self, args: SimpleNamespace, reference_camera) -> None:
        require_powerfoam()
        import warp as wp
        from powerfoam.bvh import AABBTree
        from powerfoam.color_fn import SphericalVoronoi
        from powerfoam.rasterize import Rasterizer
        from powerfoam.scene import PowerfoamScene

        if not torch.cuda.is_available():
            raise RuntimeError("Power Foam rendering requires a CUDA GPU")
        wp.init()
        self.args = args
        self._aabb_tree_cls = AABBTree
        self._sv_cls = SphericalVoronoi
        self._scene_cls = PowerfoamScene
        self._rasterizer_cls = Rasterizer
        self.reference_camera = reference_camera

    def build(self, parameters: "FoamParameters"):
        scene = self._scene_cls(self.args)
        # Avoid nn.Module parameter registration: preserve the upstream graph
        # back to the decoder rather than making detached optimization variables.
        for name, tensor in parameters.as_upstream_tensors().items():
            object.__setattr__(scene, name, tensor)
        object.__setattr__(scene, "aabb_tree", self._aabb_tree_cls(parameters.points.device))
        scene.rebuild_adjacency()
        object.__setattr__(
            scene,
            "rasterizer",
            self._rasterizer_cls(self.args, parameters.points.device, attr_dtype="float"),
        )
        sv = self._sv_cls(self.args, parameters.points.device, attr_dtype="float")
        sv.fov_cos_cutoff = self._sv_cls.compute_fov_cos_cutoff(self.reference_camera)
        object.__setattr__(scene, "sv", sv)
        return scene

    def render(self, parameters: "FoamParameters", camera) -> FoamRender:
        result = self.build(parameters).forward(camera)
        # Upstream ordering: RGB, alpha/transmittance auxiliaries, normal, depth.
        return FoamRender(rgb=result[0], alpha=result[1], normal=result[3], depth=result[4])
