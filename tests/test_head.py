import torch

from feedforwardfoam.fusion import CanonicalSupport
from feedforwardfoam.head import (
    CanonicalPowerFoamHead,
    depth_normals,
    concatenate_foam_parameters,
    inverse_softplus,
    select_foam_parameters,
    voxel_budget_indices,
)


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
    effective_radii = torch.nn.functional.softplus(params.radii, beta=100)
    assert torch.allclose(effective_radii.mean(), torch.tensor(0.05), atol=1e-4)

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
    effective_radii = torch.nn.functional.softplus(params.radii, beta=100)
    assert torch.all(effective_radii > 0)
    assert torch.allclose(params.density, torch.full_like(params.density, 50.0))
    # Upstream spherical Voronoi adds 0.5, so source 0.25 is encoded as -0.25.
    assert torch.allclose(params.texel_sv_rgb.mean(), torch.tensor(-0.25), atol=1e-3)


def test_raw_radius_round_trip_and_depth_normals_face_camera():
    physical = torch.tensor([1e-4, 0.005, 0.05, 1.0])
    raw = inverse_softplus(physical)
    assert torch.allclose(torch.nn.functional.softplus(raw, beta=100), physical, atol=1e-6)

    ray_directions = torch.zeros(3, 4, 3)
    ray_directions[..., 2] = 1
    x = torch.linspace(-0.2, 0.2, 4)[None].expand(3, -1)
    y = torch.linspace(-0.1, 0.1, 3)[:, None].expand(-1, 4)
    points = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    normals = depth_normals(points, ray_directions)
    assert torch.all((normals * -ray_directions).sum(dim=-1) > 0.99)


def test_zero_residual_head_uses_camera_facing_base_geometry_and_centered_rgb():
    head = CanonicalPowerFoamHead(
        register_dim=8,
        hidden_dim=16,
        max_cells=12,
        radius_mode="pixel_footprint",
        radius_scale_init=0.8,
        radius_residual_log_scale=0.0,
        density_mode="fixed",
        fixed_density=10_000.0,
        initialize_rgb_from_image=True,
        initialize_normals_from_depth=False,
        point_residual_scale=0.0,
        normal_residual_radians=0.0,
        rgb_residual_scale=0.0,
    )
    images = torch.rand(1, 1, 3, 3, 4)
    features = {
        "depth": torch.full((1, 1, 1, 3, 4), 2.0),
        "depth_conf": torch.ones(1, 1, 1, 3, 4),
        "registers": torch.zeros(1, 1, 2, 8),
    }
    rays = torch.zeros(3, 4, 6)
    rays[..., 3] = torch.linspace(-0.2, 0.2, 4)[None]
    rays[..., 4] = torch.linspace(0.15, -0.15, 3)[:, None]
    rays[..., 5] = 1.0
    rays[..., 3:] = torch.nn.functional.normalize(rays[..., 3:], dim=-1)

    params = head(images, features, rays)
    expected_points = 2.0 * rays[..., 3:].reshape(-1, 3)
    distances = torch.cdist(params.points.detach(), expected_points)
    matched = distances.argmin(dim=1)
    assert torch.all(distances[torch.arange(12), matched] < 1e-6)
    normals = torch.stack(
        [
            1 - 2 * (params.quaternions[:, 2].square() + params.quaternions[:, 3].square()),
            2
            * (
                params.quaternions[:, 1] * params.quaternions[:, 2]
                - params.quaternions[:, 3] * params.quaternions[:, 0]
            ),
            2
            * (
                params.quaternions[:, 1] * params.quaternions[:, 3]
                + params.quaternions[:, 2] * params.quaternions[:, 0]
            ),
        ],
        dim=-1,
    )
    expected_directions = rays[..., 3:].reshape(-1, 3)[matched]
    assert torch.allclose(normals, -expected_directions, atol=1e-5)
    expected_rgb = images[0, 0].permute(1, 2, 0).reshape(-1, 3)[matched] - 0.5
    actual_rgb = params.texel_sv_rgb.reshape(12, 8, 8, 3)
    assert torch.allclose(actual_rgb[:, 0, 0], expected_rgb, atol=1e-6)


def test_decoder_free_initialization_is_independent_of_decoder_weights():
    first = CanonicalPowerFoamHead(
        register_dim=8, hidden_dim=16, max_cells=12, prediction_mode="initialization",
        radius_mode="pixel_footprint", density_mode="fixed", fixed_density=100.0,
        initialize_rgb_from_image=True,
    )
    second = CanonicalPowerFoamHead(
        register_dim=8, hidden_dim=16, max_cells=12, prediction_mode="initialization",
        radius_mode="pixel_footprint", density_mode="fixed", fixed_density=100.0,
        initialize_rgb_from_image=True,
    )
    images = torch.rand(1, 1, 3, 3, 4)
    features = {
        "depth": torch.full((1, 1, 1, 3, 4), 2.0),
        "depth_conf": torch.ones(1, 1, 1, 3, 4),
        "registers": torch.randn(1, 1, 2, 8),
    }
    rays = torch.zeros(3, 4, 6)
    rays[..., 5] = 1.0
    output_first = first(images, features, rays)
    output_second = second(images, features, rays)
    for name, tensor in output_first.as_upstream_tensors().items():
        assert torch.allclose(tensor, output_second.as_upstream_tensors()[name])


def test_absolute_mode_keeps_only_position_initialization():
    head = CanonicalPowerFoamHead(
        register_dim=8, hidden_dim=16, max_cells=4, prediction_mode="absolute",
        radius_mode="pixel_footprint", density_mode="fixed", fixed_density=100.0,
        initialize_rgb_from_image=True,
    )
    images = torch.rand(1, 1, 3, 2, 2)
    features = {
        "depth": torch.ones(1, 1, 1, 2, 2),
        "depth_conf": torch.ones(1, 1, 1, 2, 2),
        "registers": torch.zeros(1, 1, 2, 8),
    }
    rays = torch.zeros(2, 2, 6)
    rays[..., 5] = 1.0
    params = head(images, features, rays)
    # Direct RGB starts centered (rendered as grey upstream), not from source RGB.
    assert torch.allclose(params.texel_sv_rgb, torch.zeros_like(params.texel_sv_rgb))
    assert torch.allclose(
        torch.nn.functional.softplus(params.radii, beta=100), torch.full_like(params.radii, 0.05)
    )


def test_projected_support_fusion_still_decodes_one_foam():
    head = CanonicalPowerFoamHead(
        register_dim=8,
        hidden_dim=16,
        max_cells=6,
        fusion_mode="projected",
        patch_token_dim=8,
    )
    images = torch.rand(1, 2, 3, 4, 4)
    features = {
        "depth": torch.ones(1, 2, 1, 4, 4),
        "depth_conf": torch.ones(1, 2, 1, 4, 4),
        "registers": torch.randn(1, 2, 3, 8),
    }
    rays = torch.zeros(4, 4, 6)
    rays[..., 5] = 1
    axis = torch.linspace(-1, 1, 4)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    support = CanonicalSupport(
        maps=torch.randn(1, 7, 4, 4),
        patch_tokens=torch.randn(1, 8, 2, 2),
        grid=torch.stack([xx, yy], dim=-1)[None],
    )
    params = head(images, features, rays, canonical_support=support)
    assert params.points.shape == (6, 3)
    assert params.texel_sv_rgb.shape == (6, 8, 24)


def test_multi_view_parameter_concatenation_and_voxel_budget():
    head = CanonicalPowerFoamHead(register_dim=8, hidden_dim=16, max_cells=6)
    images = torch.rand(1, 1, 3, 2, 3)
    features = {
        "depth": torch.ones(1, 1, 1, 2, 3),
        "depth_conf": torch.ones(1, 1, 1, 2, 3),
        "registers": torch.zeros(1, 1, 2, 8),
    }
    rays = torch.zeros(2, 3, 6)
    rays[..., 3] = torch.linspace(-0.2, 0.2, 3)
    rays[..., 5] = 1.0
    first = head(images, features, rays)
    second = head(images, features, rays)
    combined = concatenate_foam_parameters([first, second])
    assert combined.points.shape[0] == 12
    indices = voxel_budget_indices(combined.points, 6)
    assert indices.shape == (6,)
    assert torch.unique(indices).numel() == 6
    selected = select_foam_parameters(combined, indices)
    assert all(value.shape[0] == 6 for value in selected.as_upstream_tensors().values())


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
