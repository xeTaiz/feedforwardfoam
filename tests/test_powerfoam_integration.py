import os

import pytest
import torch

from feedforwardfoam.backbone import FrozenGeometryStub
from feedforwardfoam.data.types import View
from feedforwardfoam.head import CanonicalPowerFoamHead
from feedforwardfoam.renderer import PowerFoamRendererBridge, camera_from_view, powerfoam_args


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("FFFOAM_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set FFFOAM_RUN_CUDA_TESTS=1 in a Power Foam CUDA environment",
)
def test_frozen_features_to_powerfoam_to_image_loss_backpropagates():
    device = torch.device("cuda")
    image = torch.full((24, 24, 3), 0.25)
    pose = torch.eye(4)
    source = View(image=image, c2w=pose, fov_x_radians=0.8, name="source")
    target = View(image=image * 0.8, c2w=pose, fov_x_radians=0.8, name="target")
    second_target = View(
        image=image * 0.7,
        c2w=pose,
        fov_x_radians=0.8,
        name="second-target",
    )
    camera = camera_from_view(source, device)
    images = source.image.permute(2, 0, 1)[None, None].to(device)
    frozen = FrozenGeometryStub(register_dim=16).to(device)
    with torch.inference_mode():
        features = frozen(images)
    head = CanonicalPowerFoamHead(register_dim=16, hidden_dim=32, max_cells=32).to(device)
    params = head(images, features, camera._build_pinhole_ray_maps())
    bridge = PowerFoamRendererBridge(powerfoam_args(), camera)
    rendered = bridge.render_many(
        params,
        [camera_from_view(view, device) for view in (target, second_target)],
    )
    assert all(output.rgb.shape == target.image.shape for output in rendered)
    loss = sum(
        (output.rgb - view.image.to(device)).square().mean()
        for output, view in zip(rendered, (target, second_target), strict=True)
    )
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )
