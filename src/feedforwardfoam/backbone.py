"""Frozen VGGT-Ω adapter with an explicit test-only geometry stub."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

import torch
from torch import nn


class GeometryFeatures(Protocol):
    depth: torch.Tensor
    depth_conf: torch.Tensor
    registers: torch.Tensor


def _channel_first_dense_map(values: torch.Tensor) -> torch.Tensor:
    """Normalize VGGT-Ω dense outputs to the P0 ``[B,V,1,H,W]`` contract."""
    if values.ndim == 4:  # confidence: [B,V,H,W]
        return values.unsqueeze(2)
    if values.ndim == 5 and values.shape[-1] == 1:  # depth: [B,V,H,W,1]
        return values.movedim(-1, 2)
    if values.ndim == 5 and values.shape[2] == 1:
        return values
    raise ValueError(f"Expected a single-channel dense map, got shape {tuple(values.shape)}")


class FrozenVGGTOmega(nn.Module):
    """Inference-only VGGT-Ω adapter.

    Upstream VGGT-Ω is intentionally not vendored or modified. This adapter
    exposes only dense depth/confidence and register tokens used by P0.
    """

    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        root = Path(__file__).resolve().parents[2]
        upstream = root / "external" / "vggt-omega"
        if not upstream.exists():
            raise RuntimeError("Missing external/vggt-omega submodule")
        sys.path.insert(0, str(upstream))
        from vggt_omega.models import VGGTOmega

        self.model = VGGTOmega()
        # The dense heads consume concatenated intermediate/final token features,
        # so the exposed camera/register stream is 2× the aggregator width.
        self.register_dim = 2 * self.model.aggregator.camera_token.shape[-1]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.model(images)
        camera_and_registers = prediction["camera_and_register_tokens"]
        return {
            "depth": _channel_first_dense_map(prediction["depth"]),
            "depth_conf": _channel_first_dense_map(prediction["depth_conf"]),
            "registers": camera_and_registers[:, :, 1:],
        }


class FrozenGeometryStub(nn.Module):
    """Deterministic test-only stand-in; never use for research results."""

    def __init__(self, register_dim: int = 64, register_count: int = 4) -> None:
        super().__init__()
        self.register_dim = register_dim
        self.register_count = register_count

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # images: [B, V, 3, H, W]
        b, v, _, _h, _w = images.shape
        luminance = images.mean(dim=2, keepdim=True)
        depth = 1.0 + 0.1 * luminance
        confidence = torch.ones_like(depth)
        summary = images.mean(dim=(2, 3, 4), keepdim=False)
        registers = summary[..., None, None].expand(b, v, self.register_count, self.register_dim)
        return {"depth": depth, "depth_conf": confidence, "registers": registers}
