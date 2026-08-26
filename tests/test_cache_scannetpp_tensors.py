from pathlib import Path
from types import SimpleNamespace
from typing import cast

from feedforwardfoam.data.multiscene import MultiSceneScanNetPP
from feedforwardfoam.data.scannetpp import MissingDepthMapError, ScanNetPPDataset
from scripts.cache_scannetpp_tensors import CacheTask, cache_tasks, materialize


class _SceneDataset:
    def __init__(self, root: Path, scene_id: str, frame_count: int = 3):
        self.root = root
        self.scene_id = scene_id
        self.frames = [{"file_path": f"frame-{index}.jpg"} for index in range(frame_count)]

    def _tensor_cache_path(self, frame):
        return self.root / self.scene_id / f"{Path(frame['file_path']).stem}.npz"

    def _frame_name(self, frame):
        return frame["file_path"]

    def _indices_from_names(self, context_names, target_names):
        names = [*context_names, *target_names]
        by_name = {self._frame_name(frame): index for index, frame in enumerate(self.frames)}
        return [by_name[name] for name in names]

    def _load_view(self, frame):
        path = self._tensor_cache_path(frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cached")


def test_cache_tasks_cover_all_training_and_exact_validation_frames(tmp_path):
    train_scene = _SceneDataset(tmp_path, "train")
    validation_scene = _SceneDataset(tmp_path, "validation")
    train = SimpleNamespace(episode_entries=None, datasets=[train_scene])
    validation = SimpleNamespace(
        episode_entries=[
            {
                "scene_id": "validation",
                "context_names": ["frame-0.jpg"],
                "target_names": ["frame-1.jpg"],
            },
            {
                "scene_id": "validation",
                "context_names": ["frame-0.jpg"],
                "target_names": ["frame-2.jpg"],
            },
        ],
        _dataset_for=lambda _scene_id: validation_scene,
    )

    tasks = cache_tasks(cast(MultiSceneScanNetPP, train), cast(MultiSceneScanNetPP, validation))

    assert len(tasks) == 6
    assert len({task.path for task in tasks}) == 6


def test_materialize_is_resumable_and_writes_missing_entries(tmp_path):
    scene = _SceneDataset(tmp_path, "scene", frame_count=2)
    first_path = scene._tensor_cache_path(scene.frames[0])
    first_path.parent.mkdir(parents=True)
    first_path.write_bytes(b"existing")
    tasks = [
        CacheTask(cast(ScanNetPPDataset, scene), frame, scene._tensor_cache_path(frame))
        for frame in scene.frames
    ]

    result = materialize(tasks, workers=2, min_free_bytes=0, progress_every=2)

    assert result["counts"] == {
        "cached": 1,
        "written": 1,
        "corrupt": 0,
        "missing": 0,
    }
    assert all(task.path.is_file() for task in tasks)


def test_materialize_records_missing_depth_and_continues(tmp_path):
    scene = _SceneDataset(tmp_path, "scene", frame_count=2)
    original_load_view = scene._load_view

    def load_view(frame):
        if frame is scene.frames[0]:
            raise MissingDepthMapError("missing depth")
        original_load_view(frame)

    scene._load_view = load_view
    tasks = [
        CacheTask(cast(ScanNetPPDataset, scene), frame, scene._tensor_cache_path(frame))
        for frame in scene.frames
    ]

    result = materialize(tasks, workers=2, min_free_bytes=0, progress_every=2)

    assert result["counts"] == {
        "cached": 0,
        "written": 1,
        "corrupt": 0,
        "missing": 1,
    }
    assert result["corrupt"] == ["missing depth"]
