import torch

from feedforwardfoam.head import CanonicalPowerFoamHead


def test_canonical_head_produces_full_powerfoam_tensors_and_gradients():
    torch.manual_seed(0)
    head = CanonicalPowerFoamHead(register_dim=8, hidden_dim=16, max_cells=6)
    images = torch.rand(1, 2, 3, 4, 5)
    features = {
        "depth": torch.ones(1, 2, 1, 4, 5),
        "depth_conf": torch.ones(1, 2, 1, 4, 5),
        "registers": torch.randn(1, 2, 3, 8),
    }
    rays = torch.zeros(4, 5, 6)
    rays[..., 2] = 1.0
    rays[..., 5] = 1.0

    params = head(images, features, rays)
    assert params.points.shape == (6, 3)
    assert params.texel_sites.shape == (6, 8, 2)
    assert params.texel_height.shape == (6, 8)
    assert params.texel_sv_axis.shape == (6, 8, 24)
    assert params.texel_sv_rgb.shape == (6, 8, 24)
    assert torch.count_nonzero(params.texel_height) == 0
    # Decoder initialization must produce nonzero Power Foam density; an empty
    # volume has no photometric gradient in the initial P0 smoke experiment.
    assert torch.all(params.density > 0)
    assert torch.allclose(params.radii.mean(), torch.tensor(0.09), atol=1e-3)

    loss = sum(value.square().mean() for value in params.as_upstream_tensors().values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in head.parameters())
