from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .types import NvsEpisode, View


class BlenderNvsDataset(Dataset[NvsEpisode]):
    """NeRF-Synthetic/Blender transforms loader with scene-local held-out NVS.

    The loader intentionally returns one scene episode, rather than mixing frames
    across a dataset split.  This prevents the common frame-level leakage bug.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: Literal["train", "test", "val"] = "train",
        context_views: int = 1,
        target_views: int = 1,
        image_downsample: int = 1,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.context_views = context_views
        self.target_views = target_views
        self.image_downsample = image_downsample
        self.generator = torch.Generator().manual_seed(seed)
        metadata_path = self.root / f"transforms_{split}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing Blender metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        self.frames = self.metadata["frames"]
        if len(self.frames) < context_views + target_views:
            raise ValueError("Not enough frames for context plus held-out targets")

    def __len__(self) -> int:
        # Episode sampling is stochastic; this gives each epoch multiple pairings.
        return len(self.frames)

    def _image_path(self, frame: dict) -> Path:
        candidate = self.root / frame["file_path"]
        if candidate.suffix:
            return candidate
        for extension in (".png", ".jpg", ".jpeg"):
            if candidate.with_suffix(extension).exists():
                return candidate.with_suffix(extension)
        raise FileNotFoundError(f"Image for Blender frame not found: {candidate}")

    def _load_view(self, frame: dict) -> View:
        image = Image.open(self._image_path(frame)).convert("RGBA")
        rgba = torch.from_numpy(__import__("numpy").asarray(image).copy()).float() / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:]  # composite onto the configured black background
        alpha = rgba[..., 3]
        if self.image_downsample > 1:
            h, w = rgb.shape[:2]
            rgba_downsampled = F.interpolate(
                torch.cat([rgb, alpha[..., None]], dim=-1).permute(2, 0, 1)[None],
                size=(h // self.image_downsample, w // self.image_downsample),
                mode="area",
            )[0].permute(1, 2, 0)
            rgb = rgba_downsampled[..., :3]
            alpha = rgba_downsampled[..., 3]
        return View(
            image=rgb,
            c2w=torch.tensor(frame["transform_matrix"], dtype=torch.float32),
            fov_x_radians=float(self.metadata["camera_angle_x"]),
            name=frame["file_path"],
            alpha=alpha,
        )

    def __getitem__(self, _: int) -> NvsEpisode:
        indices = torch.randperm(len(self.frames), generator=self.generator).tolist()
        context_idx = indices[: self.context_views]
        target_idx = indices[self.context_views : self.context_views + self.target_views]
        return NvsEpisode(
            context=tuple(self._load_view(self.frames[i]) for i in context_idx),
            target=tuple(self._load_view(self.frames[i]) for i in target_idx),
            scene_id=self.root.name,
        )
