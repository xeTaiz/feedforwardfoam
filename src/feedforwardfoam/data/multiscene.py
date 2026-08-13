"""Scene-level sampling utilities for head-only feed-forward training."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .scannetpp import ScanNetPPDataset
from .types import NvsEpisode


class MultiSceneScanNetPP:
    """Randomly sample scenes, while preserving scene-disjoint split manifests."""

    def __init__(
        self,
        data_root: str | Path,
        scene_manifest: str | Path,
        *,
        split: str,
        context_views: int,
        target_views: int,
        image_resolution: int,
        target_pool_size: int | None,
        reserve_support_view: bool,
        seed: int,
    ) -> None:
        self.data_root = Path(data_root)
        manifest_path = Path(scene_manifest)
        manifest = json.loads(manifest_path.read_text())
        scene_ids = manifest[split]
        if not scene_ids:
            raise ValueError(f"Scene manifest has no {split} scenes")
        # Validation scenes are scene-disjoint from training. Their native
        # train-view pool supplies source/targets; native test views remain a
        # final within-scene test set rather than leaking into model selection.
        view_split = "train"
        self.datasets = [
            ScanNetPPDataset(
                self.data_root / scene_id,
                split=view_split,
                context_views=context_views,
                target_views=target_views,
                image_resolution=image_resolution,
                target_pool_size=target_pool_size,
                reserve_support_view=reserve_support_view,
                seed=seed + index,
            )
            for index, scene_id in enumerate(scene_ids)
        ]
        self.generator = torch.Generator().manual_seed(seed)

    def sample_episode(self) -> NvsEpisode:
        index = int(torch.randint(len(self.datasets), (), generator=self.generator))
        return self.datasets[index].sample_episode()

    def fixed_episodes(self, count: int, seed: int) -> tuple[NvsEpisode, ...]:
        generator = torch.Generator().manual_seed(seed)
        episodes = []
        for index in range(count):
            dataset = self.datasets[index % len(self.datasets)]
            episodes.append(dataset.episode_from_indices(dataset._sample_indices(generator)))
        return tuple(episodes)

    def state_dict(self) -> dict:
        return {
            "scene_generator": self.generator.get_state(),
            "dataset_generators": [dataset.generator.get_state() for dataset in self.datasets],
        }

    def load_state_dict(self, state: dict) -> None:
        self.generator.set_state(state["scene_generator"])
        for dataset, generator_state in zip(
            self.datasets, state["dataset_generators"], strict=True
        ):
            dataset.generator.set_state(generator_state)
