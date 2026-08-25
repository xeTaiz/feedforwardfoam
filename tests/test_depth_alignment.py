import torch
import pytest

from feedforwardfoam.data.types import View
from feedforwardfoam.fusion import (
    InvalidDepthGaugeError,
    align_depths_to_calibrated_cameras,
)


def _view(x: float) -> View:
    c2w = torch.eye(4)
    c2w[0, 3] = x
    return View(torch.zeros(2, 2, 3), c2w, 0.8, str(x))


def test_depth_alignment_uses_predicted_and_calibrated_camera_baselines():
    predicted_c2w = torch.eye(4).repeat(1, 2, 1, 1)
    predicted_c2w[0, 1, 0, 3] = 1.0
    features = {
        "depth": torch.full((1, 2, 1, 2, 2), 3.0),
        "predicted_extrinsics": torch.linalg.inv(predicted_c2w),
    }
    aligned, transform = align_depths_to_calibrated_cameras(features, (_view(0.0), _view(2.0)))
    assert torch.allclose(transform.scale, torch.tensor(2.0))
    assert torch.allclose(aligned, torch.full_like(aligned, 6.0))
    assert torch.allclose(transform.raw_scale, torch.tensor(2.0))
    assert not transform.bound_hit


def test_single_context_depth_alignment_is_identity():
    depth = torch.full((1, 1, 1, 2, 2), 3.0)
    aligned, transform = align_depths_to_calibrated_cameras({"depth": depth}, (_view(0.0),))
    assert torch.equal(aligned, depth)
    assert transform.samples == 0


def test_depth_alignment_does_not_clip_valid_scene_scale():
    predicted_c2w = torch.eye(4).repeat(1, 2, 1, 1)
    predicted_c2w[0, 1, 0, 3] = 0.1
    features = {
        "depth": torch.ones(1, 2, 1, 2, 2),
        "predicted_extrinsics": torch.linalg.inv(predicted_c2w),
    }
    aligned, transform = align_depths_to_calibrated_cameras(features, (_view(0.0), _view(2.0)))
    assert torch.allclose(transform.raw_scale, torch.tensor(20.0))
    assert torch.allclose(transform.scale, torch.tensor(20.0))
    assert not transform.bound_hit
    assert torch.allclose(aligned, torch.full_like(aligned, 20.0))


def test_depth_alignment_marks_zero_calibrated_baseline_as_resampleable():
    predicted_c2w = torch.eye(4).repeat(1, 2, 1, 1)
    predicted_c2w[0, 1, 0, 3] = 1.0
    features = {
        "depth": torch.ones(1, 2, 1, 2, 2),
        "predicted_extrinsics": torch.linalg.inv(predicted_c2w),
    }

    with pytest.raises(InvalidDepthGaugeError, match="scale: 0"):
        align_depths_to_calibrated_cameras(features, (_view(0.0), _view(0.0)))
