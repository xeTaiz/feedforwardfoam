from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class View:
    """One posed RGB view in the Power Foam camera convention.

    `c2w` maps OpenCV/Blender camera coordinates to the normalized world frame.
    Images are HWC RGB floats in [0, 1].
    """

    image: torch.Tensor
    c2w: torch.Tensor
    fov_x_radians: float
    name: str


@dataclass(frozen=True)
class NvsEpisode:
    """Context-only reconstruction evidence and never-input held-out targets."""

    context: tuple[View, ...]
    target: tuple[View, ...]
    scene_id: str
