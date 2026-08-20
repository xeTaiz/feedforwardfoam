"""Image metrics matching the published Splatt3R ScanNet++ evaluation."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


class SpatialLPIPS(Protocol):
    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, *, normalize: bool
    ) -> torch.Tensor: ...


def new_lpips(device: torch.device) -> SpatialLPIPS:
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("LPIPS metrics require the project benchmark dependencies") from error
    model = lpips.LPIPS(net="vgg", spatial=True).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def masked_lpips(
    model: SpatialLPIPS,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute spatial VGG LPIPS, zeroing and averaging as Splatt3R does."""
    prediction_nchw = prediction.permute(2, 0, 1)[None]
    mask_nchw: torch.Tensor | None = None
    target_nchw = target.permute(2, 0, 1)[None]
    if mask is not None:
        if not mask.any():
            raise ValueError("Cannot compute LPIPS for an empty mask")
        mask_nchw = mask[None, None].to(prediction.dtype)
        prediction_nchw = prediction_nchw * mask_nchw
        target_nchw = target_nchw * mask_nchw
    score = model(prediction_nchw, target_nchw, normalize=True)
    if mask is None:
        return score.mean()
    assert mask_nchw is not None
    score_mask = F.interpolate(mask_nchw, size=score.shape[-2:], mode="nearest")
    return (score * score_mask).sum() / score_mask.sum().clamp_min(1)


@torch.no_grad()
def masked_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Compute the PixelSplat/Splatt3R 11-pixel Gaussian SSIM protocol."""
    prediction_chw = prediction.detach().clamp(0, 1).permute(2, 0, 1).cpu().numpy()
    target_chw = target.detach().clamp(0, 1).permute(2, 0, 1).cpu().numpy()
    mask_array: np.ndarray | None = None
    if mask is not None:
        if not mask.any():
            raise ValueError("Cannot compute SSIM for an empty mask")
        mask_array = mask.detach().cpu().numpy().astype(np.float32)
        prediction_chw = prediction_chw * mask_array[None]
        target_chw = target_chw * mask_array[None]
    _, spatial = structural_similarity(
        target_chw,
        prediction_chw,
        win_size=11,
        gaussian_weights=True,
        channel_axis=0,
        data_range=1.0,
        full=True,
    )
    spatial_array = np.asarray(spatial)
    if mask is None:
        return float(spatial_array.mean())
    assert mask_array is not None
    return float((spatial_array * mask_array[None]).sum() / (3 * mask_array.sum()))
