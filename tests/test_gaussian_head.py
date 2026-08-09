"""CPU-safe smoke tests for the canonical-view Gaussian baseline.

The gsplat rasterizer is exercised only by ``tests/test_gsplat_integration.py``
under ``FFFOAM_RUN_CUDA_TESTS=1``; this file ensures the head decoder shapes,
budget enforcement, and parameter-count budget parity with the foam head are
correct on any host.
"""

import pytest
import torch

from feedforwardfoam.data.types import View
from feedforwardfoam.gaussian import (
    CanonicalGaussianHead,
    GaussianParameters,
    view_to_gsplat_camera,
)
from feedforwardfoam.head import CanonicalPowerFoamHead


def _canonical_features(batch: int, views: int, height: int, width: int, register_dim: int):
    images = torch.rand(batch, views, 3, height, width)
    features = {
        "depth": torch.ones(batch, views, 1, height, width),
        "depth_conf": torch.ones(batch, views, 1, height, width),
        "registers": torch.randn(batch, views, 3, register_dim),
    }
    rays = torch.zeros(height, width, 6)
    rays[..., 2] = 1.0
    rays[..., 5] = 1.0
    return images, features, rays


def test_blender_view_conversion_places_front_geometry_at_positive_opencv_z():
    c2w = torch.eye(4)
    c2w[2, 3] = 3.0
    view = View(
        image=torch.zeros(20, 40, 3), c2w=c2w, fov_x_radians=torch.pi / 2, name="camera"
    )
    camera = view_to_gsplat_camera(view, "cpu")
    origin = torch.tensor([0.0, 0.0, 0.0, 1.0])
    point_cv = camera["viewmats"][0] @ origin
    assert torch.allclose(point_cv, torch.tensor([0.0, 0.0, 3.0, 1.0]))
    assert torch.allclose(camera["Ks"][0, 0, 0], torch.tensor(20.0))
    assert torch.allclose(camera["Ks"][0, 1, 1], torch.tensor(20.0))


def test_canonical_gaussian_head_shapes_and_positive_scale():
    torch.manual_seed(0)
    head = CanonicalGaussianHead(register_dim=8, hidden_dim=16, max_cells=6)
    images, features, rays = _canonical_features(1, 2, 4, 5, 8)
    params = head(images, features, rays)

    assert params.means.shape == (6, 3)
    assert params.scales.shape == (6, 3)
    assert params.quats.shape == (6, 4)
    assert params.opacities.shape == (6,)
    assert params.colors.shape == (6, 3)

    # Scale activation is exp() -> strictly positive; gsplat rejects non-positive scales.
    assert torch.all(params.scales > 0)
    assert torch.allclose(params.scales.mean(), torch.tensor(0.04), atol=1e-4)
    # Opacity/colors are sigmoid -> bounded in [0, 1]; gsplat rejects out-of-range opacities.
    assert torch.all(params.opacities >= 0) and torch.all(params.opacities <= 1)
    assert torch.all(params.colors >= 0) and torch.all(params.colors <= 1)
    # Quaternions are normalized.
    norms = torch.linalg.norm(params.quats, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_canonical_gaussian_head_propagates_gradients():
    torch.manual_seed(0)
    head = CanonicalGaussianHead(register_dim=8, hidden_dim=16, max_cells=4)
    images, features, rays = _canonical_features(1, 2, 4, 5, 8)
    params = head(images, features, rays)
    loss = (
        params.means.square().mean()
        + params.scales.square().mean()
        + params.quats.square().mean()
        + params.opacities.square().mean()
        + params.colors.square().mean()
    )
    loss.backward()
    grads = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert grads, "head received no gradients"
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_canonical_gaussian_head_rejects_non_canonical_batch():
    head = CanonicalGaussianHead(register_dim=8, hidden_dim=16, max_cells=4)
    images, features, rays = _canonical_features(2, 2, 4, 5, 8)
    with pytest.raises(ValueError, match="one scene per batch"):
        head(images, features, rays)


def test_canonical_gaussian_head_is_smaller_than_foam_head():
    """Budget parity sanity check: the Gaussian head must not be larger.

    The 3DGS head shares its local CNN + register projection with the foam head
    but emits far fewer per-patch output channels (14 vs. 394), so the Gaussian
    head's trainable parameter count must be strictly smaller at the default
    configuration.
    """
    foam = CanonicalPowerFoamHead(register_dim=1024, hidden_dim=256, max_cells=1024)
    gauss = CanonicalGaussianHead(register_dim=1024, hidden_dim=256, max_cells=1024)
    foam_params = sum(parameter.numel() for parameter in foam.parameters() if parameter.requires_grad)
    gauss_params = sum(parameter.numel() for parameter in gauss.parameters() if parameter.requires_grad)
    assert gauss_params < foam_params
    assert gauss_params > 0


def test_gaussian_parameters_dataclass_is_immutable():
    params = GaussianParameters(
        means=torch.zeros(1, 3),
        quats=torch.zeros(1, 4),
        scales=torch.ones(1, 3),
        opacities=torch.ones(1),
        colors=torch.zeros(1, 3),
    )
    with pytest.raises((AttributeError, Exception)):
        params.means = torch.ones(1, 3)