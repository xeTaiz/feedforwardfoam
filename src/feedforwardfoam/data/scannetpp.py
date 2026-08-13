"""Audited ScanNet++ DSLR loader for one-source/held-out-view NVS."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .types import NvsEpisode, View


class ScanNetPPDataset(Dataset[NvsEpisode]):
    """Sample disjoint NVS episodes from one ScanNet++ scene.

    Two layouts are accepted:

    1. ``fffoam_views.json`` with explicit ``train``/``test`` frame lists.
    2. Native ScanNet++ DSLR data with
       ``dslr/nerfstudio/transforms_undistorted.json``. Only centered,
       undistorted ``PINHOLE`` cameras are accepted. Images are center-cropped
       to a square before resizing, preserving a valid centered pinhole model.
    """

    def __init__(
        self,
        scene_root: str | Path,
        *,
        split: str,
        context_views: int,
        target_views: int,
        image_downsample: int = 1,
        image_resolution: int | None = None,
        target_pool_size: int | None = None,
        reserve_support_view: bool = False,
        seed: int = 0,
    ) -> None:
        self.scene_root = Path(scene_root)
        self.context_views = context_views
        self.target_views = target_views
        self.image_downsample = image_downsample
        self.image_resolution = image_resolution
        self.target_pool_size = target_pool_size
        self.reserve_support_view = reserve_support_view
        self.generator = torch.Generator().manual_seed(seed)
        manifest_path = self.scene_root / "fffoam_views.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            self.frames = self.manifest[split]
            self.native = False
            self.scene_id = str(self.manifest.get("scene_id", self.scene_root.name))
            self.camera = None
        else:
            self.native = True
            self.scene_id = self.scene_root.name
            transforms_path = (
                self.scene_root / "dslr" / "nerfstudio" / "transforms_undistorted.json"
            )
            if not transforms_path.exists():
                raise FileNotFoundError(
                    f"Expected {manifest_path} or native metadata at {transforms_path}"
                )
            transforms = json.loads(transforms_path.read_text())
            self.camera = transforms
            self._validate_native_camera(transforms)
            if split == "train":
                self.frames = [frame for frame in transforms["frames"] if not frame.get("is_bad")]
            elif split in {"test", "val"}:
                self.frames = [
                    frame for frame in transforms.get("test_frames", []) if not frame.get("is_bad")
                ]
            else:
                raise ValueError(f"Unknown ScanNet++ split: {split}")
        if len(self.frames) < context_views + target_views:
            raise ValueError(
                f"{self.scene_id}/{split} has {len(self.frames)} views, fewer than "
                f"{context_views + target_views} required"
            )

    @staticmethod
    def _validate_native_camera(camera: dict) -> None:
        if camera.get("camera_model") != "PINHOLE":
            raise ValueError("ScanNet++ experiment requires undistorted PINHOLE imagery")
        width, height = int(camera["w"]), int(camera["h"])
        if abs(float(camera["cx"]) - width / 2) > 1e-3:
            raise ValueError("ScanNet++ principal point must be centered in x")
        if abs(float(camera["cy"]) - height / 2) > 1e-3:
            raise ValueError("ScanNet++ principal point must be centered in y")
        if abs(float(camera["fl_x"]) - float(camera["fl_y"])) / float(camera["fl_x"]) > 0.01:
            raise ValueError("ScanNet++ experiment assumes approximately square pixels")

    def __len__(self) -> int:
        return len(self.frames)

    def _native_image_path(self, frame: dict) -> Path:
        filename = Path(frame["file_path"]).name
        return self.scene_root / "dslr" / "resized_undistorted_images" / filename

    def _load_view(self, frame: dict) -> View:
        if self.native:
            assert self.camera is not None
            image_path = self._native_image_path(frame)
            source_width = int(self.camera["w"])
            source_height = int(self.camera["h"])
            crop_size = min(source_width, source_height)
            fov_x = 2.0 * math.atan(0.5 * crop_size / float(self.camera["fl_x"]))
            rgb = torch.from_numpy(
                __import__("numpy").asarray(Image.open(image_path).convert("RGB")).copy()
            )
            top = (rgb.shape[0] - crop_size) // 2
            left = (rgb.shape[1] - crop_size) // 2
            rgb = rgb[top : top + crop_size, left : left + crop_size]
            c2w = frame["transform_matrix"]
            name = str(image_path.relative_to(self.scene_root))
        else:
            image_path = self.scene_root / frame["image"]
            rgb = torch.from_numpy(
                __import__("numpy").asarray(Image.open(image_path).convert("RGB")).copy()
            )
            fov_x = float(frame["fov_x_radians"])
            c2w = frame["c2w"]
            name = frame["image"]
        rgb = rgb.float() / 255.0
        if self.image_resolution is not None:
            rgb = F.interpolate(
                rgb.permute(2, 0, 1)[None],
                size=(self.image_resolution, self.image_resolution),
                mode="area",
            )[0].permute(1, 2, 0)
        elif self.image_downsample > 1:
            height, width = rgb.shape[:2]
            rgb = F.interpolate(
                rgb.permute(2, 0, 1)[None],
                size=(height // self.image_downsample, width // self.image_downsample),
                mode="area",
            )[0].permute(1, 2, 0)
        return View(
            image=rgb,
            c2w=torch.tensor(c2w, dtype=torch.float32),
            fov_x_radians=fov_x,
            name=name,
        )

    @staticmethod
    def _frame_c2w(frame: dict) -> torch.Tensor:
        return torch.tensor(frame.get("transform_matrix", frame.get("c2w")), dtype=torch.float32)

    def _sample_indices(self, generator: torch.Generator) -> list[int]:
        source = int(torch.randint(len(self.frames), (), generator=generator))
        reserved = int(self.reserve_support_view and self.context_views == 1)
        needed = self.context_views - 1 + self.target_views + reserved
        if self.target_pool_size is None:
            remaining = torch.randperm(len(self.frames), generator=generator).tolist()
            selected = [index for index in remaining if index != source][:needed]
        else:
            source_c2w = self._frame_c2w(self.frames[source])
            centers = torch.stack([self._frame_c2w(frame)[:3, 3] for frame in self.frames])
            distances = torch.linalg.vector_norm(centers - source_c2w[:3, 3], dim=-1)
            directions = torch.stack([-self._frame_c2w(frame)[:3, 2] for frame in self.frames])
            source_direction = -source_c2w[:3, 2]
            angle_penalty = 1.0 - (directions * source_direction).sum(dim=-1).clamp(-1, 1)
            median_distance = distances[distances > 0].median().clamp_min(1e-6)
            overlap_score = distances / median_distance + angle_penalty
            overlap_score[source] = torch.inf
            pool_size = min(max(self.target_pool_size, needed), len(self.frames) - 1)
            pool = overlap_score.topk(pool_size, largest=False).indices
            order = torch.randperm(pool_size, generator=generator)[:needed]
            selected = pool[order].tolist()
        contexts = [source, *selected[: self.context_views - 1]]
        target_start = self.context_views - 1 + reserved
        targets = selected[target_start:]
        return [*contexts, *targets]

    def sample_episode(self) -> NvsEpisode:
        return self.episode_from_indices(self._sample_indices(self.generator))

    def episode_from_indices(self, indices: list[int]) -> NvsEpisode:
        if len(indices) != self.context_views + self.target_views:
            raise ValueError("Episode index count does not match context plus target views")
        return NvsEpisode(
            context=tuple(self._load_view(self.frames[i]) for i in indices[: self.context_views]),
            target=tuple(self._load_view(self.frames[i]) for i in indices[self.context_views :]),
            scene_id=self.scene_id,
        )

    def __getitem__(self, _: int) -> NvsEpisode:
        return self.sample_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an FF-Foam ScanNet++ scene")
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-views", type=int, default=1)
    args = parser.parse_args()
    dataset = ScanNetPPDataset(
        args.scene_root,
        split=args.split,
        context_views=1,
        target_views=args.target_views,
        image_resolution=160,
    )
    episode = dataset[0]
    print(f"Validated {episode.scene_id}: {len(dataset.frames)} {args.split} views")


if __name__ == "__main__":
    main()
