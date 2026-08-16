import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_stratified_triplet_matrix.py"
_SPEC = importlib.util.spec_from_file_location("run_stratified_triplet_matrix", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
LauncherError = _MODULE.LauncherError
build_config = _MODULE.build_config
load_manifest = _MODULE.load_manifest
shard_episodes = _MODULE.shard_episodes


def _episode(identifier="episode-0", *, targets=None):
    return {
        "id": identifier,
        "scene": "scene-a",
        "context_names": ["context-0.jpg", "context-1.jpg"],
        "target_names": targets or ["target.jpg"],
        "unused_metadata": 123,
    }


def _base_config():
    return {
        "output_dir": "old-output",
        "data": {
            "dataset": "scannetpp",
            "context_views": 1,
            "target_views": 2,
            "require_target_between_contexts": False,
        },
        "head": {
            "prediction_mode": "absolute",
            "enable_point_residual": False,
            "enable_radius_residual": False,
            "enable_orientation_residual": False,
            "enable_rgb_residual": False,
            "decode_sv_axes": True,
        },
        "train": {
            "steps": 10,
            "learning_rate": 0.1,
            "visibility_mask": True,
            "resample_episodes": True,
        },
    }


def test_load_manifest_validates_required_episode_fields(tmp_path):
    path = tmp_path / "manifest.json"
    episodes = [_episode()]
    path.write_text(json.dumps({"episodes": episodes}))

    assert load_manifest(path) == episodes

    path.write_text(json.dumps({"episodes": [{**_episode(), "target_names": []}]}))
    with pytest.raises(LauncherError, match="target_names.*non-empty list"):
        load_manifest(path)

    path.write_text(json.dumps({"episodes": [_episode("same"), _episode("same")]}))
    with pytest.raises(LauncherError, match="duplicate episode id"):
        load_manifest(path)

    path.write_text(json.dumps({"episodes": [{**_episode(), "id": "../escape"}]}))
    with pytest.raises(LauncherError, match="unsafe id"):
        load_manifest(path)


@pytest.mark.parametrize("mode", ["initialization", "full", "appearance"])
def test_build_config_sets_common_fixed_triplet_values_without_mutating_base(mode, tmp_path):
    base = _base_config()
    original = _base_config()
    output_dir = tmp_path / "episode-0" / mode

    config = build_config(base, _episode(), mode, output_dir, 3000)

    assert base == original
    assert config["output_dir"] == str(output_dir)
    assert config["data"] == {
        "dataset": "scannetpp",
        "context_views": 2,
        "target_views": 1,
        "require_target_between_contexts": True,
        "max_perpendicular_fraction": 0.25,
        "fixed_scene_id": "scene-a",
        "context_names": ["context-0.jpg", "context-1.jpg"],
        "target_names": ["target.jpg"],
    }
    assert config["train"]["resample_episodes"] is False
    assert config["train"]["report_support_metrics"] is True
    assert config["train"]["support_mask_contexts"] == "canonical"
    assert config["train"]["visibility_mask"] is False
    assert config["train"]["learning_rate_schedule"] == "cosine"
    assert config["train"]["min_learning_rate"] == 1e-6
    assert config["head"]["decode_sv_axes"] is True


def test_build_config_sets_mode_specific_values_and_multiple_target_rule(tmp_path):
    initial = build_config(_base_config(), _episode(), "initialization", tmp_path / "i", 3000)
    assert initial["head"]["prediction_mode"] == "initialization"
    assert initial["train"]["steps"] == 1
    assert initial["train"]["learning_rate"] == 0.0

    full = build_config(_base_config(), _episode(), "full", tmp_path / "f", 3000)
    assert full["head"]["prediction_mode"] == "residual"
    assert full["train"]["steps"] == 3000
    assert full["train"]["learning_rate"] == 5e-4
    assert all(
        full["head"][key]
        for key in (
            "enable_point_residual",
            "enable_radius_residual",
            "enable_orientation_residual",
            "enable_rgb_residual",
        )
    )

    appearance = build_config(
        _base_config(),
        _episode(targets=["target-0.jpg", "target-1.jpg"]),
        "appearance",
        tmp_path / "a",
        99,
    )
    assert appearance["head"]["prediction_mode"] == "residual"
    assert appearance["head"]["enable_point_residual"] is False
    assert appearance["head"]["enable_radius_residual"] is False
    assert appearance["head"]["enable_orientation_residual"] is False
    assert appearance["head"]["enable_rgb_residual"] is True
    assert appearance["data"]["target_views"] == 2
    assert "require_target_between_contexts" not in appearance["data"]


def test_shard_episodes_uses_manifest_list_index_modulo_count():
    episodes = [_episode(f"episode-{index}") for index in range(7)]

    assert [episode["id"] for episode in shard_episodes(episodes, 1, 3)] == [
        "episode-1",
        "episode-4",
    ]
    with pytest.raises(LauncherError, match="shard-count must be positive"):
        shard_episodes(episodes, 0, 0)
    with pytest.raises(LauncherError, match="shard-index"):
        shard_episodes(episodes, 3, 3)
