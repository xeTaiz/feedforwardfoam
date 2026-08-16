#!/usr/bin/env python3
"""Sequentially launch a sharded matrix of fixed-triplet overfit runs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

MODES = ("initialization", "full", "appearance")


class LauncherError(ValueError):
    """The launch inputs cannot produce a safe fixed-triplet matrix."""


def _nonempty_strings(value: Any, field: str, index: int) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise LauncherError(
            f"manifest episode {index} field {field!r} must be a non-empty list of strings"
        )
    return value


def load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LauncherError(f"manifest does not exist: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise LauncherError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), list):
        raise LauncherError("manifest must be a JSON object containing an episodes list")
    episodes = manifest["episodes"]
    seen_ids: set[str] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise LauncherError(f"manifest episode {index} must be a JSON object")
        for field in ("id", "scene"):
            value = episode.get(field)
            if not isinstance(value, str) or not value:
                raise LauncherError(
                    f"manifest episode {index} field {field!r} must be a non-empty string"
                )
        episode_id = episode["id"]
        if Path(episode_id).name != episode_id or episode_id in {".", ".."}:
            raise LauncherError(f"manifest episode {index} has unsafe id {episode_id!r}")
        if episode_id in seen_ids:
            raise LauncherError(f"manifest contains duplicate episode id {episode_id!r}")
        seen_ids.add(episode_id)
        _nonempty_strings(episode.get("context_names"), "context_names", index)
        _nonempty_strings(episode.get("target_names"), "target_names", index)
    return episodes


def load_base_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LauncherError(f"base config does not exist: {path}") from error
    except (OSError, yaml.YAMLError) as error:
        raise LauncherError(f"cannot read base config {path}: {error}") from error
    if not isinstance(config, dict):
        raise LauncherError("base config must contain a YAML mapping")
    for section in ("data", "head", "train"):
        if not isinstance(config.get(section), dict):
            raise LauncherError(f"base config must contain a {section} mapping")
    return config


def parse_modes(value: str) -> list[str]:
    modes = [mode.strip() for mode in value.split(",") if mode.strip()]
    if not modes:
        raise LauncherError("--modes must select at least one mode")
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise LauncherError(f"unknown modes: {', '.join(unknown)}")
    if len(set(modes)) != len(modes):
        raise LauncherError("--modes must not contain duplicates")
    return modes


def shard_episodes(
    episodes: list[dict[str, Any]], shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    if shard_count <= 0:
        raise LauncherError("--shard-count must be positive")
    if not 0 <= shard_index < shard_count:
        raise LauncherError("--shard-index must be between 0 and shard-count - 1")
    return [episode for index, episode in enumerate(episodes) if index % shard_count == shard_index]


def build_config(
    base_config: dict[str, Any],
    episode: dict[str, Any],
    mode: str,
    output_dir: Path,
    steps: int,
) -> dict[str, Any]:
    if steps <= 0:
        raise LauncherError("--steps must be positive")
    if mode not in MODES:
        raise LauncherError(f"unknown mode: {mode}")

    config = copy.deepcopy(base_config)
    data = config["data"]
    head = config["head"]
    train = config["train"]
    context_names = list(episode["context_names"])
    target_names = list(episode["target_names"])

    config["output_dir"] = str(output_dir)
    data.update(
        {
            "fixed_scene_id": episode["scene"],
            "context_names": context_names,
            "target_names": target_names,
            "context_views": len(context_names),
            "target_views": len(target_names),
        }
    )
    if len(target_names) == 1:
        data["require_target_between_contexts"] = True
        data["max_perpendicular_fraction"] = 0.25
    else:
        data.pop("require_target_between_contexts", None)

    train.update(
        {
            "resample_episodes": False,
            "report_support_metrics": True,
            "support_mask_contexts": "canonical",
            "visibility_mask": False,
            "learning_rate_schedule": "cosine",
            "learning_rate": 5e-4,
            "min_learning_rate": 1e-6,
            "steps": steps,
        }
    )

    if mode == "initialization":
        head["prediction_mode"] = "initialization"
        train["steps"] = 1
        train["learning_rate"] = 0.0
    else:
        head.update(
            {
                "prediction_mode": "residual",
                "enable_point_residual": mode == "full",
                "enable_radius_residual": mode == "full",
                "enable_orientation_residual": mode == "full",
                "enable_rgb_residual": True,
            }
        )
    return config


def _write_config(config: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "config.yaml"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", prefix=".config-", dir=output_dir,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(config, stream, sort_keys=True)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def launch(args: argparse.Namespace) -> None:
    episodes = shard_episodes(
        load_manifest(args.manifest), args.shard_index, args.shard_count
    )
    base_config = load_base_config(args.base_config)
    modes = parse_modes(args.modes)
    if args.steps <= 0:
        raise LauncherError("--steps must be positive")

    for episode in episodes:
        for mode in modes:
            output_dir = args.output_root / episode["id"] / mode
            if (output_dir / "final.pt").exists() and not args.overwrite:
                print(f"Skipping completed run: {output_dir}", flush=True)
                continue
            config = build_config(base_config, episode, mode, output_dir, args.steps)
            config_path = _write_config(config, output_dir)
            command = [
                sys.executable,
                "-m",
                "feedforwardfoam.train",
                "--config",
                str(config_path),
                "--data-root",
                str(args.data_root),
                "--checkpoint",
                str(args.checkpoint),
            ]
            print(f"Launching {episode['id']}/{mode}", flush=True)
            subprocess.run(command, check=True, env=os.environ.copy())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        launch(args)
    except LauncherError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
