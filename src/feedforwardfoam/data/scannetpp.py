"""ScanNet++ manifest loader.

Raw ScanNet++ releases have several camera/image products.  The research harness
uses an explicit, versioned manifest after a scene has been pose-audited, rather
than silently guessing among them.  A manifest contains `train`/`test` lists of
views, each with a scene-relative image path, a 4x4 camera-to-world matrix, and
horizontal FoV in radians.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .types import NvsEpisode, View


class ScanNetPPDataset(Dataset[NvsEpisode]):
    """Scene-disjoint NVS episodes from an audited ScanNet++ view manifest."""

    def __init__(
        self,
        scene_root: str | Path,
        *,
        split: str,
        context_views: int,
        target_views: int,
        image_downsample: int = 1,
        seed: int = 0,
    ) -> None:
        self.scene_root = Path(scene_root)
        manifest_path = self.scene_root / "fffoam_views.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} is required. Create it only after choosing a static "
                "DSLR sequence and validating COLMAP poses."
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.frames = self.manifest[split]
        self.context_views = context_views
        self.target_views = target_views
        self.image_downsample = image_downsample
        self.generator = torch.Generator().manual_seed(seed)
        if len(self.frames) < context_views + target_views:
            raise ValueError("Not enough audited views for one held-out NVS episode")

    def __len__(self) -> int:
        return len(self.frames)

    def _load_view(self, frame: dict) -> View:
        image_path = self.scene_root / frame["image"]
        rgb = torch.from_numpy(__import__("numpy").asarray(Image.open(image_path).convert("RGB")).copy())
        rgb = rgb.float() / 255.0
        if self.image_downsample > 1:
            h, w = rgb.shape[:2]
            rgb = F.interpolate(
                rgb.permute(2, 0, 1)[None],
                size=(h // self.image_downsample, w // self.image_downsample),
                mode="area",
            )[0].permute(1, 2, 0)
        return View(
            image=rgb,
            c2w=torch.tensor(frame["c2w"], dtype=torch.float32),
            fov_x_radians=float(frame["fov_x_radians"]),
            name=frame["image"],
        )

    def __getitem__(self, _: int) -> NvsEpisode:
        indices = torch.randperm(len(self.frames), generator=self.generator).tolist()
        return NvsEpisode(
            context=tuple(self._load_view(self.frames[i]) for i in indices[: self.context_views]),
            target=tuple(
                self._load_view(self.frames[i])
                for i in indices[self.context_views : self.context_views + self.target_views]
            ),
            scene_id=str(self.manifest.get("scene_id", self.scene_root.name)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an FF-Foam ScanNet++ scene manifest")
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    dataset = ScanNetPPDataset(
        args.scene_root, split=args.split, context_views=1, target_views=1
    )
    episode = dataset[0]
    print(f"Validated {episode.scene_id}: {len(dataset.frames)} {args.split} views")


if __name__ == "__main__":
    main()
