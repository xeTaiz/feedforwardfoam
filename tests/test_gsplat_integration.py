"""CUDA-gated integration test for the canonical-view gsplat baseline.

Requires the ``gsplat`` extra (``uv pip install -e '.[gsplat]'``) and a CUDA
host with the ``FFFOAM_RUN_CUDA_TESTS=1`` env var set, mirroring the gating
of :mod:`tests.test_powerfoam_integration`.
"""

import os

import pytest
import torch

from feedforwardfoam.backbone import FrozenGeometryStub
from feedforwardfoam.data.types import View
from feedforwardfoam.gaussian import CanonicalGaussianHead, GaussianRendererBridge


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("FFFOAM_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set FFFOAM_RUN_CUDA_TESTS=1 in a gsplat CUDA environment",
)
def test_canonical_gaussian_head_renders_and_backpropagates_image_loss():
    device = torch.device("cuda")
    image = torch.full((24, 24, 3), 0.25)
    pose = torch.eye(4)
    target = View(image=image * 0.8, c2w=pose, fov_x_radians=0.8, name="target")
    source = View(image=image, c2w=pose, fov_x_radians=0.8, name="source")
    images = source.image.permute(2, 0, 1)[None, None].to(device)
    frozen = FrozenGeometryStub(register_dim=16).to(device)
    with torch.inference_mode():
        features = frozen(images)
    head = CanonicalGaussianHead(register_dim=16, hidden_dim=32, max_cells=32).to(device)
    bridge = GaussianRendererBridge()
    # Use the source view as the canonical anchor; render from the target view.
    canonical_view = source
    params = head(images, features, _canonical_rays(canonical_view, device))
    rendered = bridge.render(params, target)
    assert rendered.rgb.shape == target.image.shape
    assert rendered.alpha.shape == target.image.shape[:-1]
    assert rendered.depth.shape == target.image.shape[:-1]
    loss = (rendered.rgb - target.image.to(device)).square().mean()
    loss.backward()
    grads = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def _canonical_rays(view: View, device: torch.device) -> torch.Tensor:
    height, width = view.image.shape[:2]
    rays = torch.zeros(height, width, 6)
    rays[..., 2] = 1.0  # z origin
    rays[..., 5] = 1.0  # z direction (toward +z forward in OpenCV camera frame)
    return rays.to(device)