"""Same-budget canonical-view 3D Gaussian baseline built on gsplat.

The decoder mirrors :class:`feedforwardfoam.head.CanonicalPowerFoamHead` exactly:
identical inputs, identical deterministic top-M budget, identical local feature
extractor, and an identical register projection. Only the per-patch output
parameterization differs (3 scales + 4 quats + 1 opacity + 3 colors + 3 mean
residual = 14 channels vs. the foam's much larger surface-detail output). This
keeps the foam-vs-Gaussian comparison interpretable at matched primitive
budget.

gsplat conventions (verified against gsplat==1.5.3 docs):

* ``viewmats`` is the world-to-camera transform with **OpenCV** conventions
  (``+x`` right, ``+y`` down, ``+z`` forward).
* ``Ks`` is a 3x3 pinhole intrinsic matrix.
* Quaternions use the ``wxyz`` convention.
* Colors are post-activation RGB when ``sh_degree`` is ``None``.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Soft import gate
# ---------------------------------------------------------------------------

def require_gsplat() -> None:
    """Make the gsplat dependency importable and report a clear error."""
    try:
        importlib.import_module("gsplat")
    except ImportError as error:
        raise RuntimeError(
            "gsplat is not installed. Run `uv pip install -e '.[gsplat]'` on a CUDA host."
        ) from error


# ---------------------------------------------------------------------------
# Camera conversion
# ---------------------------------------------------------------------------

# Basis change from the project's Blender/OpenGL camera frame to OpenCV's
# ``+x`` right, ``+y`` down, ``+z`` forward. The matrix is its own inverse and
# preserves rotation norms. Concretely, a point in Blender camera coordinates
# (right=+x, up=+y, back=+z) becomes a point in OpenCV coordinates via
# ``p_cv = flip @ p_blender``.
_BLENDER_TO_OPENCV_FLIP = (1.0, -1.0, -1.0, 1.0)


def _flip_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.diag(torch.tensor(_BLENDER_TO_OPENCV_FLIP, device=device, dtype=dtype))


def view_to_gsplat_camera(view, device: torch.device | str) -> dict[str, torch.Tensor]:
    """Convert a project :class:`View` into gsplat's pinhole camera tensors.

    Returns ``{viewmats, Ks, width, height}`` all on ``device``. ``viewmats`` is
    ``[1, 4, 4]`` world-to-camera in OpenCV conventions and ``Ks`` is
    ``[1, 3, 3]`` with ``cx = width/2``, ``cy = height/2``,
    ``f = 0.5 * width / tan(fov_x / 2)``. The conversion assumes a positive-Z
    forward OpenCV camera frame: ``w2c_cv = flip @ inv(c2w_blender)``.
    """
    c2w = view.c2w.to(device=device, dtype=torch.float32)
    flip = _flip_matrix(device, torch.float32)
    # c2w inverts cleanly for affine transforms; do not rely on `.inverse()` for
    # degenerate poses, the project's loader does not emit them.
    w2c_blender = torch.linalg.inv(c2w)
    w2c_opencv = flip @ w2c_blender

    height, width = view.image.shape[:2]
    f = 0.5 * float(width) / torch.tan(
        torch.tensor(view.fov_x_radians / 2, device=device, dtype=torch.float32)
    )
    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = float(width) / 2.0
    K[1, 2] = float(height) / 2.0
    return {
        "viewmats": w2c_opencv[None],  # [1, 4, 4]
        "Ks": K[None],  # [1, 3, 3]
        "width": int(width),
        "height": int(height),
    }


# ---------------------------------------------------------------------------
# Head parameters and decoder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussianParameters:
    """gsplat-ready per-Gaussian attributes for one scene."""

    means: torch.Tensor  # [N, 3]
    quats: torch.Tensor  # [N, 4] wxyz
    scales: torch.Tensor  # [N, 3] positive
    opacities: torch.Tensor  # [N] in [0, 1]
    colors: torch.Tensor  # [N, 3] post-activation RGB

    def as_gsplat_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "means": self.means,
            "quats": self.quats,
            "scales": self.scales,
            "opacities": self.opacities,
            "colors": self.colors,
        }


class CanonicalGaussianHead(nn.Module):
    """Canonical-patch decoder for one 3DGS scene.

    Architecture is intentionally identical to :class:`CanonicalPowerFoamHead`
    up to the per-patch output layer. The local convolutional feature
    extractor, register projection, top-M anchor selection, and ray lifting
    are shared structurally so the foam-vs-Gaussian delta is solely the
    output head. The output channels are:

    * 3 -- mean residual (added to back-projected depth)
    * 3 -- per-axis anisotropic scale (raw exponent)
    * 4 -- quaternion (wxyz, normalized at decode time)
    * 1 -- opacity (raw sigmoid)
    * 3 -- RGB color (raw sigmoid)
    """

    OUTPUT_DIM = 3 + 3 + 4 + 1 + 3

    def __init__(
        self,
        *,
        register_dim: int,
        hidden_dim: int = 256,
        max_cells: int = 1024,
    ) -> None:
        super().__init__()
        self.max_cells = max_cells
        self.local = nn.Sequential(
            nn.Conv2d(5, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.register_projection = nn.Linear(register_dim, hidden_dim)
        self.decode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.OUTPUT_DIM),
        )
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        """Start with compact, non-empty splats for a stable image-loss signal."""
        output = self.decode[-1]
        assert isinstance(output, nn.Linear)
        with torch.no_grad():
            # Keep the opacity/gate row's random weights for non-degenerate top-M.
            output.weight[:10].zero_()
            output.weight[11:].zero_()
            output.bias.zero_()
            output.bias[3:6] = torch.log(torch.tensor(0.02))
            output.bias[6] = 1.0  # identity quaternion (wxyz)
            output.bias[10] = -2.0

    @staticmethod
    def _canonical_maps(
        images: torch.Tensor, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = images[:, 0]
        depth = features["depth"][:, 0]
        confidence = features["depth_conf"][:, 0]
        if depth.ndim == 3:
            depth = depth[:, None]
        if confidence.ndim == 3:
            confidence = confidence[:, None]
        if depth.shape[-2:] != image.shape[-2:]:
            depth = F.interpolate(depth, size=image.shape[-2:], mode="bilinear", align_corners=False)
            confidence = F.interpolate(
                confidence, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        return image, depth, confidence

    def forward(
        self,
        images: torch.Tensor,
        frozen_features: dict[str, torch.Tensor],
        canonical_ray_map: torch.Tensor,
    ) -> GaussianParameters:
        """Decode a batch-size-one canonical scene into 3DGS attributes."""
        if images.shape[0] != 1:
            raise ValueError("CanonicalGaussianHead currently accepts one scene per batch")
        image, depth, confidence = self._canonical_maps(images, frozen_features)
        registers = frozen_features["registers"]
        register_feature = self.register_projection(registers.mean(dim=(1, 2)))
        local = self.local(torch.cat([image, depth, confidence], dim=1))
        local = local + register_feature[:, :, None, None]
        h, w = local.shape[-2:]
        tokens = local.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        logits = self.decode(tokens)[0]

        # Match the foam head's deterministic budget selector.
        m = min(self.max_cells, logits.shape[0])
        # Gate on the opacity raw channel (index 10) -- the most representative
        # "is this primitive worth keeping" signal.
        selected = logits[:, 10].topk(m, sorted=False).indices
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
        depth_selected = depth[0, 0].reshape(-1)[selected].clamp_min(1e-3)

        # 0..2: mean residual added to the back-projected point.
        mean_residual = 0.05 * torch.tanh(values[:, :3])
        means = rays[:, :3] + (depth_selected[:, None] + mean_residual[:, :1]) * rays[:, 3:]
        # 3..5: per-axis scales, positive via exp().
        scales = 0.02 + torch.exp(values[:, 3:6]).clamp_min(1e-5)
        # 6..9: quaternions, normalized defensively (gsplat says optional).
        quats = F.normalize(values[:, 6:10], dim=-1, eps=1e-6)
        # 10: opacity.
        opacities = 0.1 + 0.9 * torch.sigmoid(values[:, 10])
        # 11..13: RGB post-activation.
        colors = torch.sigmoid(values[:, 11:14])

        return GaussianParameters(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
        )


# ---------------------------------------------------------------------------
# Renderer bridge
# ---------------------------------------------------------------------------

@dataclass
class GaussianRender:
    """gsplat-produced tensors consumed by the same loss path as :class:`FoamRender`."""

    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor


class GaussianRendererBridge:
    """Build and run gsplat's differentiable rasterizer on a head's output.

    The bridge mirrors :class:`feedforwardfoam.renderer.PowerFoamRendererBridge`:
    it owns a CUDA-capable gsplat instance and renders held-out cameras from
    one head output, propagating gradients to the head. It does *not* wrap the
    output tensors in :class:`nn.Parameter`: gradients must flow back to the
    feed-forward decoder.
    """

    def __init__(
        self,
        *,
        bkgd_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        eps2d: float = 0.3,
        radius_clip: float = 0.0,
        tile_size: int = 16,
        rasterize_mode: str = "classic",
        near_plane: float = 0.01,
        far_plane: float = 1e10,
    ) -> None:
        require_gsplat()
        if not torch.cuda.is_available():
            raise RuntimeError("gsplat rasterization requires a CUDA GPU")
        self.bkgd_color = bkgd_color
        self.eps2d = eps2d
        self.radius_clip = radius_clip
        self.tile_size = tile_size
        # gsplat reads "antialiased" vs "classic" differently for opacities; pin
        # to "classic" so this baseline matches the foam on apples-to-apples
        # NVS comparison rather than re-running Mip-Splatting's compensation.
        self.rasterize_mode = rasterize_mode
        self.near_plane = near_plane
        self.far_plane = far_plane

    def render(self, parameters: GaussianParameters, view) -> GaussianRender:
        from gsplat import rasterization

        device = parameters.means.device
        camera = view_to_gsplat_camera(view, device)
        # gsplat 1.5.3's packed RGB+D path turns a [C, RGB] background into
        # [C, RGB+D] internally, but its packed rasterizer subsequently expects
        # no camera batch dimensions. Use its native zero background here; P0's
        # configured background is intentionally black for both representations.
        if any(value != 0.0 for value in self.bkgd_color):
            raise ValueError("The packed gsplat RGB+D P0 baseline currently requires black background")
        render_colors, render_alphas, _ = rasterization(
            means=parameters.means,
            quats=parameters.quats,
            scales=parameters.scales,
            opacities=parameters.opacities,
            colors=parameters.colors,
            viewmats=camera["viewmats"],
            Ks=camera["Ks"],
            width=camera["width"],
            height=camera["height"],
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            radius_clip=self.radius_clip,
            eps2d=self.eps2d,
            sh_degree=None,
            packed=True,
            tile_size=self.tile_size,
            backgrounds=None,
            render_mode="RGB+D",
            rasterize_mode=self.rasterize_mode,
            camera_model="pinhole",
        )
        rgb = render_colors[0, ..., :3]
        depth = render_colors[0, ..., 3]
        alpha = render_alphas[0, ..., 0]
        return GaussianRender(rgb=rgb, alpha=alpha, depth=depth)