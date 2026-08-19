import pytest
import torch

from feedforwardfoam.backbone import FrozenGeometryStub, _channel_first_dense_map


def test_vggt_omega_dense_maps_normalize_to_p0_channel_first_contract():
    depth = torch.rand(1, 2, 4, 5, 1)
    confidence = torch.rand(1, 2, 4, 5)

    assert _channel_first_dense_map(depth).shape == (1, 2, 1, 4, 5)
    assert _channel_first_dense_map(confidence).shape == (1, 2, 1, 4, 5)


def test_geometry_stub_preserves_per_view_feature_contract():
    output = FrozenGeometryStub(register_dim=8)(torch.rand(1, 2, 3, 8, 10))
    assert output["patch_tokens"].shape == (1, 2, 8, 4, 5)
    assert output["predicted_extrinsics"].shape == (1, 2, 3, 4)
    assert torch.allclose(
        output["predicted_extrinsics"],
        torch.eye(4)[:3].expand(1, 2, 3, 4),
    )


@pytest.mark.parametrize("shape", [(1, 2, 2, 4, 5), (1, 2, 4, 5, 2), (1, 2, 4)])
def test_vggt_omega_dense_map_rejects_non_scalar_channels(shape):
    with pytest.raises(ValueError, match="single-channel"):
        _channel_first_dense_map(torch.rand(shape))
