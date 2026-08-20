"""Scene-level sampling utilities for head-only feed-forward training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import torch

from .scannetpp import ScanNetPPDataset
from .types import NvsEpisode


class EpisodeEntry(TypedDict):
    scene_id: str
    context_names: list[str]
    target_names: list[str]
    bin: str


class MultiSceneScanNetPP:
    """Sample either scenes or preselected episodes from scene-disjoint splits."""

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
        coverage_root: str | Path | None = None,
        context_overlap_threshold: float = 0.5,
        target_overlap_threshold: float = 0.6,
        native_image_directory: str = "resized_undistorted_images",
        resize_mode: str = "area",
        load_depth: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        manifest_path = Path(scene_manifest)
        manifest = cast(dict[str, object], json.loads(manifest_path.read_text()))
        raw_entries = manifest.get(split)
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError(f"Scene manifest has no {split} scenes or episodes")
        entries = cast(list[str | EpisodeEntry], raw_entries)
        explicit_flags = [isinstance(entry, dict) for entry in entries]
        if any(explicit_flags) and not all(explicit_flags):
            raise ValueError(f"Manifest {split} split cannot mix scene IDs and episodes")

        self.context_views = context_views
        self.target_views = target_views
        self.image_resolution = image_resolution
        self.target_pool_size = target_pool_size
        self.reserve_support_view = reserve_support_view
        self.seed = seed
        self.coverage_root = Path(coverage_root) if coverage_root is not None else None
        self.context_overlap_threshold = context_overlap_threshold
        self.target_overlap_threshold = target_overlap_threshold
        self.native_image_directory = native_image_directory
        self.resize_mode = resize_mode
        self.load_depth = load_depth
        self.generator = torch.Generator().manual_seed(seed)
        self.episode_entries: list[EpisodeEntry] | None = (
            cast(list[EpisodeEntry], entries) if all(explicit_flags) else None
        )
        self._dataset_cache: dict[str, ScanNetPPDataset] = {}

        if self.episode_entries is not None:
            for entry in self.episode_entries:
                if len(entry["context_names"]) != context_views:
                    raise ValueError("Manifest episode has the wrong number of context views")
                if len(entry["target_names"]) != target_views:
                    raise ValueError("Manifest episode has the wrong number of target views")
                if not entry["scene_id"]:
                    raise ValueError("Manifest episode is missing scene_id")
            self.datasets: list[ScanNetPPDataset] = []
        else:
            # Validation scenes are scene-disjoint from training. Native test
            # views remain a final within-scene test set, not model selection data.
            scene_ids = cast(list[str], entries)
            self.datasets = [
                self._new_dataset(scene_id, seed + index)
                for index, scene_id in enumerate(scene_ids)
            ]

    def _new_dataset(self, scene_id: str, seed: int) -> ScanNetPPDataset:
        overlap_path = (
            self.coverage_root / f"{scene_id}.json" if self.coverage_root is not None else None
        )
        return ScanNetPPDataset(
            self.data_root / scene_id,
            split="train",
            context_views=self.context_views,
            target_views=self.target_views,
            image_resolution=self.image_resolution,
            target_pool_size=self.target_pool_size,
            reserve_support_view=self.reserve_support_view,
            native_image_directory=self.native_image_directory,
            resize_mode=self.resize_mode,
            load_depth=self.load_depth,
            overlap_path=overlap_path,
            context_overlap_threshold=self.context_overlap_threshold,
            target_overlap_threshold=self.target_overlap_threshold,
            seed=seed,
        )

    def _dataset_for(self, scene_id: str) -> ScanNetPPDataset:
        dataset = self._dataset_cache.get(scene_id)
        if dataset is None:
            dataset = self._new_dataset(scene_id, self.seed + len(self._dataset_cache))
            self._dataset_cache[scene_id] = dataset
        return dataset

    def _episode_from_entry(self, entry: EpisodeEntry) -> NvsEpisode:
        return self._dataset_for(str(entry["scene_id"])).episode_from_names(
            list(entry["context_names"]), list(entry["target_names"])
        )

    def sample_episode(self) -> NvsEpisode:
        if self.episode_entries is not None:
            index = int(torch.randint(len(self.episode_entries), (), generator=self.generator))
            return self._episode_from_entry(self.episode_entries[index])
        index = int(torch.randint(len(self.datasets), (), generator=self.generator))
        return self.datasets[index].sample_episode()

    def __len__(self) -> int:
        return len(self.episode_entries) if self.episode_entries is not None else len(self.datasets)

    def fixed_episode_records(self, count: int, seed: int) -> tuple[tuple[str, NvsEpisode], ...]:
        """Return deterministic validation episodes, balanced over manifest bins."""
        if count <= 0:
            return ()
        if self.episode_entries is None:
            generator = torch.Generator().manual_seed(seed)
            random_records: list[tuple[str, NvsEpisode]] = []
            for index in range(count):
                dataset = self.datasets[index % len(self.datasets)]
                episode = dataset.episode_from_indices(dataset._sample_indices(generator))
                random_records.append(("all", episode))
            return tuple(random_records)

        groups: dict[str, list[EpisodeEntry]] = {}
        for entry in self.episode_entries:
            groups.setdefault(entry["bin"], []).append(entry)
        labels = sorted(groups)
        generators = {
            label: torch.Generator().manual_seed(seed + index) for index, label in enumerate(labels)
        }
        orders = {
            label: cast(
                list[int],
                torch.randperm(len(groups[label]), generator=generators[label]).tolist(),
            )
            for label in labels
        }
        positions = {label: 0 for label in labels}
        explicit_records: list[tuple[str, NvsEpisode]] = []
        while len(explicit_records) < count:
            for label in labels:
                order = orders[label]
                position = positions[label]
                entry = groups[label][order[position % len(order)]]
                positions[label] = position + 1
                explicit_records.append((label, self._episode_from_entry(entry)))
                if len(explicit_records) == count:
                    break
        return tuple(explicit_records)

    def fixed_episodes(self, count: int, seed: int) -> tuple[NvsEpisode, ...]:
        return tuple(episode for _, episode in self.fixed_episode_records(count, seed))

    def all_episode_records(self) -> tuple[tuple[str, NvsEpisode], ...]:
        """Load every explicit manifest episode once, preserving manifest order."""
        if self.episode_entries is None:
            raise ValueError("Exhaustive records require explicit manifest episodes")
        return tuple(
            (entry["bin"], self._episode_from_entry(entry)) for entry in self.episode_entries
        )

    def state_dict(self) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        state: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "scene_generator": self.generator.get_state()
        }
        if self.episode_entries is None:
            state["dataset_generators"] = [
                dataset.generator.get_state() for dataset in self.datasets
            ]
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor | list[torch.Tensor]]) -> None:
        _ = self.generator.set_state(cast(torch.Tensor, state["scene_generator"]))
        if self.episode_entries is None:
            generator_states = cast(list[torch.Tensor], state["dataset_generators"])
            for dataset, generator_state in zip(self.datasets, generator_states, strict=True):
                _ = dataset.generator.set_state(generator_state)
