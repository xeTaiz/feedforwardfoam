from types import SimpleNamespace

import pytest
import torch

import feedforwardfoam.train as train_module
from feedforwardfoam.backbone import FrozenGeometryStub
from feedforwardfoam.data.types import NvsEpisode, View
from feedforwardfoam.head import CanonicalPowerFoamHead
from feedforwardfoam.train import _episode_objective, _metrics, _predict, _triplet_geometry


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


def test_support_metrics_ignore_pixels_outside_mask():
    target = torch.zeros(2, 2, 3)
    prediction = target.clone()
    prediction[1, 1] = 1.0
    mask = torch.tensor([[True, True], [True, False]])
    assert _metrics(prediction, target)["mse"] == 0.25
    assert _metrics(prediction, target, mask)["mse"] == 0.0


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


class _StubBackboneWithCameras(torch.nn.Module):
    """Geometry stub plus the predicted cameras depth alignment requires."""

    def __init__(self, register_dim: int) -> None:
        super().__init__()
        self.stub = FrozenGeometryStub(register_dim=register_dim, register_count=2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = dict(self.stub(images))
        view_count = images.shape[1]
        extrinsics = torch.eye(4).expand(1, view_count, 4, 4).clone()
        for index in range(view_count):
            # World-to-camera translation of a camera centred at x = index.
            extrinsics[0, index, 0, 3] = -float(index)
        features["predicted_extrinsics"] = extrinsics
        return features


def _posed_view(name: str, offset: float, seed: int) -> View:
    generator = torch.Generator().manual_seed(seed)
    c2w = torch.eye(4)
    c2w[0, 3] = offset
    return View(
        image=torch.rand(4, 4, 3, generator=generator),
        c2w=c2w,
        fov_x_radians=0.7,
        name=name,
    )


def _proposal_episode() -> NvsEpisode:
    return NvsEpisode(
        context=(_posed_view("c0", 0.0, 1), _posed_view("c1", 0.5, 2)),
        target=(_posed_view("t0", 0.25, 3),),
        scene_id="stub_scene",
    )


def _proposal_head(
    reduction: str, selection_mode: str = "uniform", **kwargs
) -> CanonicalPowerFoamHead:
    return CanonicalPowerFoamHead(
        register_dim=8,
        hidden_dim=16,
        max_cells=8,
        proposal_views="all",
        proposal_reduction=reduction,
        selection_mode=selection_mode,
        **kwargs,
    )


@pytest.mark.parametrize("reduction", ["voxel", "fps", "confidence_voxel"])
def test_proposal_reductions_emit_exactly_the_cell_budget(reduction):
    head = _proposal_head(reduction)
    params, _ = _predict(
        head,
        _StubBackboneWithCameras(register_dim=8),
        _proposal_episode(),
        "foam",
        torch.device("cpu"),
    )
    assert all(value.shape[0] == 8 for value in params.as_upstream_tensors().values())


def test_reduction_arms_do_not_share_cached_selections():
    episode = _proposal_episode()
    backbone = _StubBackboneWithCameras(register_dim=8)
    for reduction in ("voxel", "fps"):
        head = _proposal_head(reduction)
        _predict(head, backbone, episode, "foam", torch.device("cpu"))
        assert all(key[0] == reduction for key in head._proposal_index_cache)


def test_confidence_voxel_rejects_gate_selection_that_misaligns_scores():
    head = _proposal_head("confidence_voxel", selection_mode="gate")
    with pytest.raises(ValueError, match="uniform selection"):
        _predict(
            head,
            _StubBackboneWithCameras(register_dim=8),
            _proposal_episode(),
            "foam",
            torch.device("cpu"),
        )


def _incremental_cells(**kwargs) -> int:
    head = _proposal_head("incremental", **kwargs)
    params, _ = _predict(
        head,
        _StubBackboneWithCameras(register_dim=8),
        _proposal_episode(),
        "foam",
        torch.device("cpu"),
    )
    return params.points.shape[0]


def test_incremental_merge_at_zero_tolerance_keeps_every_proposal():
    # Two 4x4 contexts propose 32 cells; zero tolerance must reduce to nothing.
    assert _incremental_cells(proposal_containment_tolerance=0.0) == 32


def test_incremental_merge_never_drops_the_first_context():
    for criterion in ("power", "ball"):
        cells = _incremental_cells(
            proposal_containment=criterion, proposal_containment_tolerance=1.0
        )
        assert 16 <= cells <= 32


def test_incremental_merge_is_weaker_under_the_power_criterion():
    # The power test subtracts the newcomer's own radius, so it can only keep
    # at least as many sites as the plain sphere test.
    power = _incremental_cells(proposal_containment="power", proposal_containment_tolerance=1.0)
    ball = _incremental_cells(proposal_containment="ball", proposal_containment_tolerance=1.0)
    assert power >= ball


def _validation_episode(prediction: float, scene_id: str) -> NvsEpisode:
    target = View(
        image=torch.zeros(2, 2, 3),
        c2w=torch.eye(4),
        fov_x_radians=0.7,
        name=str(prediction),
    )
    return NvsEpisode(context=(_view(0.0), _view(0.0)), target=(target,), scene_id=scene_id)


def test_validation_aggregates_fixed_bins_and_counts_each_render_once(monkeypatch):
    def fake_predict(*_args, **_kwargs):
        return None, {
            "depth": torch.ones(1, 2, 1, 2, 2),
            "depth_alignment_bound_hit": torch.zeros(1),
        }

    def fake_render(_params, targets, _bridge, _representation, _device):
        return [
            SimpleNamespace(
                rgb=torch.full((2, 2, 3), float(target.name)),
                alpha=torch.ones(2, 2),
            )
            for target in targets
        ]

    monkeypatch.setattr(train_module, "_predict", fake_predict)
    monkeypatch.setattr(train_module, "_render_targets", fake_render)
    monkeypatch.setattr(
        train_module,
        "projected_context_support_mask",
        lambda *_args, **_kwargs: torch.ones(2, 2, dtype=torch.bool),
    )
    records = (
        ("low_angle", _validation_episode(0.1, "low")),
        ("mid_angle", _validation_episode(0.3, "mid")),
    )
    metrics = train_module._validation(
        records,
        head=None,
        backbone=None,
        bridge=None,
        representation="foam",
        device=torch.device("cpu"),
        support_context_mode="all",
        support_dilation=2,
    )
    assert metrics["val_renders"] == 2
    assert metrics["val_mse"] == pytest.approx((0.1**2 + 0.3**2) / 2)
    assert metrics["val_mse_low_angle"] == pytest.approx(0.1**2)
    assert metrics["val_mse_mid_angle"] == pytest.approx(0.3**2)
    assert metrics["val_psnr"] == pytest.approx(
        -10.0 * torch.log10(torch.tensor((0.1**2 + 0.3**2) / 2)).item()
    )
    assert metrics["val_support_psnr"] == pytest.approx(metrics["val_psnr"])

    all_metrics = train_module._validation(
        (("all", _validation_episode(0.2, "all")),),
        head=None,
        backbone=None,
        bridge=None,
        representation="foam",
        device=torch.device("cpu"),
        support_context_mode="all",
        support_dilation=2,
    )
    assert all_metrics["val_renders"] == 1
    assert all_metrics["val_mse"] == pytest.approx(0.2**2)
