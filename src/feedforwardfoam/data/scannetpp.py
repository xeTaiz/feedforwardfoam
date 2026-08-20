"""Audited ScanNet++ DSLR loader for one-source/held-out-view NVS."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .types import NvsEpisode, View


def validate_native_camera(camera: dict) -> None:
    """Reject native ScanNet++ scenes this experiment's pinhole model cannot use."""
    if camera.get("camera_model") != "PINHOLE":
        raise ValueError("ScanNet++ experiment requires undistorted PINHOLE imagery")
    width, height = int(camera["w"]), int(camera["h"])
    if abs(float(camera["cx"]) - width / 2) > 1e-3:
        raise ValueError("ScanNet++ principal point must be centered in x")
    if abs(float(camera["cy"]) - height / 2) > 1e-3:
        raise ValueError("ScanNet++ principal point must be centered in y")
    if abs(float(camera["fl_x"]) - float(camera["fl_y"])) / float(camera["fl_x"]) > 0.01:
        raise ValueError("ScanNet++ experiment assumes approximately square pixels")


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
        native_image_directory: str = "resized_undistorted_images",
        resize_mode: str = "area",
        load_depth: bool = False,
        overlap_path: str | Path | None = None,
        context_overlap_threshold: float = 0.5,
        target_overlap_threshold: float = 0.6,
        seed: int = 0,
    ) -> None:
        self.scene_root = Path(scene_root)
        self.context_views = context_views
        self.target_views = target_views
        self.image_downsample = image_downsample
        self.image_resolution = image_resolution
        self.target_pool_size = target_pool_size
        self.reserve_support_view = reserve_support_view
        self.native_image_directory = native_image_directory
        if resize_mode not in {"area", "lanczos"}:
            raise ValueError("resize_mode must be 'area' or 'lanczos'")
        self.resize_mode = resize_mode
        self.load_depth = load_depth
        self.context_overlap_threshold = context_overlap_threshold
        self.target_overlap_threshold = target_overlap_threshold
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
            validate_native_camera(transforms)
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
        self.coverage: torch.Tensor | None = None
        self.coverage_frame_indices: list[int] | None = None
        if overlap_path is not None:
            if not self.native:
                raise ValueError("Splatt3R overlap sampling requires native ScanNet++ data")
            if context_views != 2:
                raise ValueError("Splatt3R overlap sampling requires exactly two context views")
            coverage_payload: dict[str, Any] = json.loads(Path(overlap_path).read_text())
            if self.scene_id not in coverage_payload:
                raise ValueError(f"Coverage file does not contain scene {self.scene_id}")
            self.coverage = torch.tensor(coverage_payload[self.scene_id], dtype=torch.float32)
            self.coverage_frame_indices = self._splatt3r_frame_indices()
            expected = len(self.coverage_frame_indices)
            if self.coverage.shape != (expected, expected):
                raise ValueError(
                    f"Coverage matrix is {tuple(self.coverage.shape)}, expected {(expected, expected)}"
                )

    def __len__(self) -> int:
        return len(self.frames)

    def _native_image_path(self, frame: dict) -> Path:
        filename = Path(frame["file_path"]).name
        return self.scene_root / "dslr" / self.native_image_directory / filename

    def _native_depth_path(self, frame: dict) -> Path:
        filename = Path(frame["file_path"]).with_suffix(".png").name
        candidates = (
            self.scene_root / "dslr" / "undistorted_depths" / filename,
            self.scene_root / "dslr" / "resized_undistorted_depths" / filename,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Missing undistorted ScanNet++ depth map: {candidates[0]}")

    def _splatt3r_frame_indices(self) -> list[int]:
        split_path = self.scene_root / "dslr" / "train_test_lists.json"
        transforms_path = self.scene_root / "dslr" / "nerfstudio" / "transforms.json"
        if not split_path.exists() or not transforms_path.exists():
            raise FileNotFoundError(
                "Splatt3R overlap sampling requires dslr/train_test_lists.json and "
                "dslr/nerfstudio/transforms.json"
            )
        train_names = json.loads(split_path.read_text())["train"]
        raw_frames = json.loads(transforms_path.read_text())["frames"]
        raw_by_name = {Path(frame["file_path"]).name: frame for frame in raw_frames}
        native_by_name = {
            Path(frame["file_path"]).name: index for index, frame in enumerate(self.frames)
        }
        ordered: list[int] = []
        for raw_name in train_names:
            name = Path(raw_name).name
            if raw_by_name[name].get("is_bad"):
                continue
            if name not in native_by_name:
                raise ValueError(f"Splatt3R frame is absent from undistorted metadata: {name}")
            ordered.append(native_by_name[name])
        return ordered

    def _load_view(self, frame: dict) -> View:
        depth = None
        if self.native:
            assert self.camera is not None
            image_path = self._native_image_path(frame)
            rgb = torch.from_numpy(
                __import__("numpy").asarray(Image.open(image_path).convert("RGB")).copy()
            )
            image_height, image_width = rgb.shape[:2]
            camera_width = int(self.camera["w"])
            camera_height = int(self.camera["h"])
            if abs(image_width / image_height - camera_width / camera_height) > 1e-3:
                raise ValueError(
                    "Native ScanNet++ image aspect ratio does not match camera metadata"
                )
            crop_size = min(image_height, image_width)
            top = (image_height - crop_size) // 2
            left = (image_width - crop_size) // 2
            rgb = rgb[top : top + crop_size, left : left + crop_size]
            if self.load_depth:
                depth = torch.from_numpy(
                    __import__("numpy").asarray(Image.open(self._native_depth_path(frame))).copy()
                ).float()
                depth_height, depth_width = depth.shape[:2]
                if abs(depth_width / depth_height - camera_width / camera_height) > 1e-3:
                    raise ValueError(
                        "Native ScanNet++ depth aspect ratio does not match camera metadata"
                    )
                depth_crop = min(depth_height, depth_width)
                depth_top = (depth_height - depth_crop) // 2
                depth_left = (depth_width - depth_crop) // 2
                depth = depth[
                    depth_top : depth_top + depth_crop, depth_left : depth_left + depth_crop
                ]
                depth = depth / 1000.0
            camera_crop_size = min(camera_width, camera_height)
            fov_x = 2.0 * math.atan(0.5 * camera_crop_size / float(self.camera["fl_x"]))
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
        if self.image_resolution is not None and self.resize_mode == "lanczos":
            resized = Image.fromarray(rgb.numpy()).resize(
                (self.image_resolution, self.image_resolution), Image.Resampling.LANCZOS
            )
            rgb = torch.from_numpy(__import__("numpy").asarray(resized).copy())
        rgb = rgb.float() / 255.0
        if self.image_resolution is not None and self.resize_mode == "area":
            rgb = F.interpolate(
                rgb.permute(2, 0, 1)[None],
                size=(self.image_resolution, self.image_resolution),
                mode="area",
            )[0].permute(1, 2, 0)
        elif self.image_resolution is None and self.image_downsample > 1:
            height, width = rgb.shape[:2]
            rgb = F.interpolate(
                rgb.permute(2, 0, 1)[None],
                size=(height // self.image_downsample, width // self.image_downsample),
                mode="area",
            )[0].permute(1, 2, 0)
        if depth is not None:
            if self.image_resolution is not None:
                depth = F.interpolate(
                    depth[None, None],
                    size=(self.image_resolution, self.image_resolution),
                    mode="nearest-exact",
                )[0, 0]
            elif self.image_downsample > 1:
                depth = F.interpolate(
                    depth[None, None],
                    size=(rgb.shape[0], rgb.shape[1]),
                    mode="nearest-exact",
                )[0, 0]
        return View(
            image=rgb,
            c2w=torch.tensor(c2w, dtype=torch.float32),
            fov_x_radians=fov_x,
            name=name,
            depth=depth,
        )

    @staticmethod
    def _frame_c2w(frame: dict) -> torch.Tensor:
        return torch.tensor(frame.get("transform_matrix", frame.get("c2w")), dtype=torch.float32)

    def _sample_indices(self, generator: torch.Generator) -> list[int]:
        if self.coverage is not None:
            assert self.coverage_frame_indices is not None
            count = len(self.coverage_frame_indices)
            source = int(torch.randint(count, (), generator=generator))
            row = self.coverage[source]
            valid_contexts = torch.nonzero(
                (row > self.context_overlap_threshold) & (torch.arange(count) != source),
                as_tuple=False,
            ).flatten()
            if valid_contexts.numel():
                selected = int(
                    valid_contexts[torch.randint(valid_contexts.numel(), (), generator=generator)]
                )
            else:
                scores = row.clone()
                scores[source] = -torch.inf
                selected = int(scores.argmax())
            target_scores = torch.maximum(self.coverage[source], self.coverage[selected])
            target_scores[source] = -torch.inf
            target_scores[selected] = -torch.inf
            valid_targets = torch.nonzero(
                target_scores > self.target_overlap_threshold, as_tuple=False
            ).flatten()
            if valid_targets.numel() >= self.target_views:
                order = torch.randperm(valid_targets.numel(), generator=generator)[
                    : self.target_views
                ]
                targets = valid_targets[order]
            else:
                targets = target_scores.topk(self.target_views).indices
            protocol_indices = [source, selected, *targets.tolist()]
            return [self.coverage_frame_indices[index] for index in protocol_indices]

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

    def _frame_name(self, frame: dict) -> str:
        if self.native:
            return str(self._native_image_path(frame).relative_to(self.scene_root))
        return str(frame["image"])

    def episode_from_names(self, context_names: list[str], target_names: list[str]) -> NvsEpisode:
        """Load one explicit, reproducible episode by scene-relative image name."""
        requested = [str(Path(name)) for name in (*context_names, *target_names)]
        if len(context_names) != self.context_views or len(target_names) != self.target_views:
            raise ValueError("Configured names do not match context/target view counts")
        if len(set(requested)) != len(requested):
            raise ValueError("Configured context and target names must be distinct")
        name_to_index = {self._frame_name(frame): index for index, frame in enumerate(self.frames)}
        missing = [name for name in requested if name not in name_to_index]
        if missing:
            raise ValueError(f"Configured ScanNet++ images not found: {missing}")
        return self.episode_from_indices([name_to_index[name] for name in requested])

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
