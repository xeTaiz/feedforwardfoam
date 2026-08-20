import torch

from feedforwardfoam.data.types import View
from feedforwardfoam.fusion import (
    laser_context_support_mask,
    project_world_points,
    world_points_from_z_depth,
)


def _view(translation=(0.0, 0.0, 0.0)) -> View:
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor(translation)
    return View(torch.zeros(5, 5, 3), c2w, 1.0, "view")


def test_z_depth_lifting_and_projection_round_trip():
    view = _view()
    depth = torch.full((1, 1, 5, 5), 2.0)
    points = world_points_from_z_depth(view, depth, "cpu")
    grid, z_depth, valid = project_world_points(points, view, "cpu")
    expected = torch.linspace(-1, 1, 5)
    yy, xx = torch.meshgrid(expected, expected, indexing="ij")
    assert torch.allclose(grid[..., 0], xx, atol=1e-5)
    assert torch.allclose(grid[..., 1], yy, atol=1e-5)
    assert torch.allclose(z_depth, torch.full_like(z_depth, 2.0), atol=1e-5)
    assert valid.all()


def test_support_projection_changes_with_camera_translation():
    canonical = _view()
    support = _view((0.2, 0.0, 0.0))
    points = world_points_from_z_depth(canonical, torch.full((1, 1, 5, 5), 2.0), "cpu")
    grid, _, valid = project_world_points(points, support, "cpu")
    assert valid.any()
    assert grid[2, 2, 0] < 0


def test_laser_support_requires_matching_context_depth():
    c2w = torch.eye(4)
    depth = torch.full((5, 5), 2.0)
    context = View(torch.zeros(5, 5, 3), c2w, 1.0, "context", depth=depth)
    target_depth = depth.clone()
    target_depth[2, 2] = 0.0
    target = View(torch.zeros(5, 5, 3), c2w, 1.0, "target", depth=target_depth)
    mask = laser_context_support_mask((context,), target, "cpu")
    assert mask[1, 1]
    assert not mask[2, 2]
