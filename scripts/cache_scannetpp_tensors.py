"""Materialize the exact ScanNet++ train/validation tensor cache on local storage."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from feedforwardfoam.data.multiscene import MultiSceneScanNetPP
from feedforwardfoam.data.scannetpp import (
    CorruptDepthMapError,
    MissingDepthMapError,
    ScanNetPPDataset,
)
from feedforwardfoam.train import _build_datasets


@dataclass(frozen=True)
class CacheTask:
    dataset: ScanNetPPDataset
    frame: dict[str, Any]
    path: Path


def _dataset_tasks(dataset: MultiSceneScanNetPP) -> Iterable[CacheTask]:
    if dataset.episode_entries is None:
        for scene_dataset in dataset.datasets:
            for frame in scene_dataset.frames:
                path = scene_dataset._tensor_cache_path(frame)
                if path is None:
                    raise ValueError("data.tensor_cache_root is required")
                yield CacheTask(scene_dataset, frame, path)
        return

    for entry in dataset.episode_entries:
        scene_dataset = dataset._dataset_for(str(entry["scene_id"]))
        indices = scene_dataset._indices_from_names(
            list(entry["context_names"]), list(entry["target_names"])
        )
        for index in indices:
            frame = scene_dataset.frames[index]
            path = scene_dataset._tensor_cache_path(frame)
            if path is None:
                raise ValueError("data.tensor_cache_root is required")
            yield CacheTask(scene_dataset, frame, path)


def cache_tasks(
    train_dataset: MultiSceneScanNetPP, validation_dataset: MultiSceneScanNetPP | None
) -> list[CacheTask]:
    """Return the unique tensor-cache entries used by this experiment."""
    unique: dict[Path, CacheTask] = {}
    for dataset in (train_dataset, validation_dataset):
        if dataset is None:
            continue
        for task in _dataset_tasks(dataset):
            unique.setdefault(task.path, task)
    return list(unique.values())


def _materialize(task: CacheTask) -> tuple[str, str | None]:
    if task.path.is_file():
        return "cached", None
    try:
        task.dataset._load_view(task.frame)
    except CorruptDepthMapError as error:
        return "corrupt", str(error)
    except MissingDepthMapError as error:
        return "missing", str(error)
    if not task.path.is_file():
        raise RuntimeError(f"Tensor cache was not written: {task.path}")
    return "written", None


def materialize(
    tasks: list[CacheTask], *, workers: int, min_free_bytes: int, progress_every: int
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    counts = {"cached": 0, "written": 0, "corrupt": 0, "missing": 0}
    corrupt: list[str] = []
    pending_tasks = iter(tasks)
    futures: dict[Future[tuple[str, str | None]], CacheTask] = {}

    def submit_one(pool: ThreadPoolExecutor) -> bool:
        try:
            task = next(pending_tasks)
        except StopIteration:
            return False
        futures[pool.submit(_materialize, task)] = task
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(min(len(tasks), workers * 2)):
            submit_one(pool)
        completed = 0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                status, detail = future.result()
                counts[status] += 1
                if detail is not None:
                    corrupt.append(detail)
                completed += 1
                if completed % progress_every == 0 or completed == len(tasks):
                    free = shutil.disk_usage(task.path.parent).free
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": len(tasks),
                                "counts": counts,
                                "free_gib": free / 2**30,
                            }
                        ),
                        flush=True,
                    )
                    if free < min_free_bytes:
                        raise RuntimeError(
                            f"Tensor-cache materialization stopped below {min_free_bytes / 2**30:.1f} GiB free"
                        )
                submit_one(pool)
    return {"total": len(tasks), "counts": counts, "corrupt": corrupt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=40.0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    train_dataset, validation_dataset = _build_datasets(config, args.data_root)
    if not isinstance(train_dataset, MultiSceneScanNetPP):
        raise ValueError("Cache materialization requires a multi-scene ScanNet++ experiment")
    if validation_dataset is not None and not isinstance(validation_dataset, MultiSceneScanNetPP):
        raise ValueError("Expected a multi-scene ScanNet++ validation dataset")
    tasks = cache_tasks(train_dataset, validation_dataset)
    existing = [task.path.stat().st_size for task in tasks if task.path.is_file()]
    mean_bytes = sum(existing) / len(existing) if existing else 256 * 1024
    summary = {
        "frames": len(tasks),
        "cached": len(existing),
        "missing": len(tasks) - len(existing),
        "existing_gib": sum(existing) / 2**30,
        "projected_gib": (sum(existing) + (len(tasks) - len(existing)) * mean_bytes) / 2**30,
        "workers": args.workers,
    }
    print(json.dumps(summary), flush=True)
    if args.dry_run:
        return

    torch.set_num_threads(1)
    result = materialize(
        tasks,
        workers=args.workers,
        min_free_bytes=int(args.min_free_gib * 2**30),
        progress_every=args.progress_every,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
