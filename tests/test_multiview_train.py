from types import SimpleNamespace

import torch

from feedforwardfoam.data.types import NvsEpisode, View
from feedforwardfoam.train import _episode_objective, _triplet_geometry


def _view(value: float) -> View:
    return View(
        image=torch.full((2, 2, 3), value),
        c2w=torch.eye(4),
        fov_x_radians=0.7,
        name=str(value),
    )


def test_triplet_geometry_detects_midpoint_target():
    context_0 = _view(0.0)
    context_1 = _view(0.0)
    target = _view(0.0)
    context_0.c2w[0, 3] = -1.0
    context_1.c2w[0, 3] = 1.0
    geometry = _triplet_geometry(
        NvsEpisode(context=(context_0, context_1), target=(target,), scene_id="test")
    )
    assert geometry["target_between_contexts"]
    assert geometry["target_interpolation"] == 0.5
    assert geometry["target_perpendicular_fraction"] == 0.0


def test_multiview_objective_averages_all_target_views():
    targets = (_view(0.0), _view(0.5), _view(1.0), _view(0.25))
    predictions = (
        SimpleNamespace(rgb=torch.full((2, 2, 3), 0.1), alpha=torch.ones(2, 2)),
        SimpleNamespace(rgb=torch.full((2, 2, 3), 0.4), alpha=torch.ones(2, 2)),
        SimpleNamespace(rgb=torch.full((2, 2, 3), 0.7), alpha=torch.ones(2, 2)),
        SimpleNamespace(rgb=torch.full((2, 2, 3), 0.3), alpha=torch.ones(2, 2)),
    )
    loss, rgb_loss, alpha_loss = _episode_objective(
        predictions,
        targets,
        rgb_loss_name="mse",
        alpha_weight=0.0,
        device=torch.device("cpu"),
    )
    expected = torch.tensor((0.1**2 + 0.1**2 + 0.3**2 + 0.05**2) / 4)
    assert torch.allclose(rgb_loss, expected)
    assert torch.allclose(loss, expected)
    assert alpha_loss == 0
