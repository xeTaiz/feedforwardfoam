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
    *,
    check_nonempty: bool = True,
) -> torch.Tensor:
    """Compute spatial VGG LPIPS for one HWC image or a batch of BHWC images."""
    single = prediction.ndim == 3
    if single:
        prediction = prediction[None]
        target = target[None]
        if mask is not None:
            mask = mask[None]
    if prediction.ndim != 4 or target.shape != prediction.shape:
        raise ValueError("LPIPS prediction and target must have matching HWC or BHWC shapes")
    prediction_nchw = prediction.permute(0, 3, 1, 2)
    target_nchw = target.permute(0, 3, 1, 2)
    mask_nchw: torch.Tensor | None = None
    if mask is not None:
        if mask.shape != prediction.shape[:3]:
            raise ValueError("LPIPS mask must match the image batch and spatial dimensions")
        if check_nonempty and not mask.flatten(1).any(dim=1).all():
            raise ValueError("Cannot compute LPIPS for an empty mask")
        mask_nchw = mask[:, None].to(prediction.dtype)
        prediction_nchw = prediction_nchw * mask_nchw
        target_nchw = target_nchw * mask_nchw
    score = model(prediction_nchw, target_nchw, normalize=True)
    if mask_nchw is None:
        return score.flatten(1).mean(dim=1).mean()
    score_mask = F.interpolate(mask_nchw, size=score.shape[-2:], mode="nearest")
    per_image = (score * score_mask).flatten(1).sum(dim=1) / score_mask.flatten(1).sum(
        dim=1
    ).clamp_min(1)
    return per_image.mean()


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
