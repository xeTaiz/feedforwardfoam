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


def test_geometry_aware_head_uses_pixel_footprint_and_fixed_density():
    head = CanonicalPowerFoamHead(
        register_dim=8,
        hidden_dim=16,
        max_cells=20,
        radius_mode="pixel_footprint",
        radius_scale_init=2.0,
        density_mode="fixed",
        fixed_density=50.0,
        initialize_rgb_from_image=True,
    )
    images = torch.full((1, 1, 3, 4, 5), 0.25)
    features = {
        "depth": torch.full((1, 1, 1, 4, 5), 2.0),
        "depth_conf": torch.ones(1, 1, 1, 4, 5),
        "registers": torch.zeros(1, 1, 2, 8),
    }
    rays = torch.zeros(4, 5, 6)
    rays[..., 3] = torch.linspace(-0.2, 0.2, 5)[None]
    rays[..., 4] = torch.linspace(0.15, -0.15, 4)[:, None]
    rays[..., 5] = 1.0
    rays[..., 3:] = torch.nn.functional.normalize(rays[..., 3:], dim=-1)

    params = head(images, features, rays)
    assert torch.all(params.radii > 0)
    assert torch.allclose(params.density, torch.full_like(params.density, 50.0))
    assert torch.allclose(params.texel_sv_rgb.mean(), torch.tensor(0.25), atol=1e-3)


def test_source_alpha_fixed_density_makes_background_cells_empty():
    head = CanonicalPowerFoamHead(
        register_dim=8,
        hidden_dim=16,
        max_cells=4,
        density_mode="source_alpha_fixed",
        fixed_density=100.0,
    )
    images = torch.ones(1, 1, 3, 2, 2)
    features = {
        "depth": torch.ones(1, 1, 1, 2, 2),
        "depth_conf": torch.ones(1, 1, 1, 2, 2),
        "registers": torch.zeros(1, 1, 2, 8),
    }
    rays = torch.zeros(2, 2, 6)
    rays[..., 5] = 1
    alpha = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    params = head(images, features, rays, alpha)
    assert torch.count_nonzero(params.density == 100.0) == 2
    assert torch.count_nonzero(params.density == -1.0) == 2
