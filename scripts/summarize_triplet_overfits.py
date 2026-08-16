#!/usr/bin/env python3
"""Aggregate fixed-triplet overfit runs into CSV and Markdown summaries."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


COLUMNS = (
    "run_name",
    "fusion_mode",
    "visibility_mask",
    "support_mask_contexts",
    "scene_name",
    "context_names",
    "target_names",
    "steps",
    "context_baseline",
    "target_interpolation",
    "perpendicular_fraction",
    "max_view_angle",
    "depth_alignment_scale",
    "canonical_support_fraction",
    "visibility_mask_fraction",
    "final_train_psnr",
    "best_train_psnr",
    "best_train_psnr_step",
    "final_support_psnr",
    "best_support_psnr",
    "best_support_psnr_step",
    "final_alpha",
    "psnr_slope_db_per_1000",
)

MARKDOWN_LABELS = {
    "run_name": "Run",
    "fusion_mode": "Fusion",
    "visibility_mask": "Vis. mask",
    "support_mask_contexts": "Support contexts",
    "scene_name": "Scene",
    "context_names": "Contexts",
    "target_names": "Targets",
    "steps": "Steps",
    "context_baseline": "Baseline",
    "target_interpolation": "Interpolation",
    "perpendicular_fraction": "Perp. fraction",
    "max_view_angle": "Max angle (deg)",
    "depth_alignment_scale": "Depth scale",
    "canonical_support_fraction": "Canonical support",
    "visibility_mask_fraction": "Vis. fraction",
    "final_train_psnr": "Final PSNR",
    "best_train_psnr": "Best PSNR",
    "best_train_psnr_step": "Best step",
    "final_support_psnr": "Final support PSNR",
    "best_support_psnr": "Best support PSNR",
    "best_support_psnr_step": "Support best step",
    "final_alpha": "Final alpha",
    "psnr_slope_db_per_1000": "PSNR slope (dB/1000)",
}


class AggregationError(ValueError):
    """An input run cannot be aggregated safely."""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise AggregationError(f"missing required file: {path}") from error
    except OSError as error:
        raise AggregationError(f"cannot read {path}: {error}") from error


def _read_json(path: Path) -> Any:
    text = _read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise AggregationError(
            f"malformed JSON in {path}: line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error


def _yaml_scalars(path: Path) -> dict[tuple[str, str], str]:
    """Read the simple nested scalar keys needed from PyYAML's safe_dump output.

    The utility deliberately does not import PyYAML: training writes plain block-style
    mappings, and all requested configuration values are scalar children of top-level
    mappings.
    """
    result: dict[tuple[str, str], str] = {}
    section: str | None = None
    for line_number, raw_line in enumerate(_read_text(path).splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            raise AggregationError(f"malformed YAML in {path}:{line_number}: tab indentation")
        top_match = re.fullmatch(r"([A-Za-z_][\w-]*):(?:\s*(.*))?", raw_line)
        if top_match:
            section = top_match.group(1) if not (top_match.group(2) or "").strip() else None
            continue
        child_match = re.fullmatch(r"  ([A-Za-z_][\w-]*):(?:\s*(.*))?", raw_line)
        if child_match and section is not None:
            value = (child_match.group(2) or "").strip()
            if value:
                key = (section, child_match.group(1))
                if key in result:
                    raise AggregationError(
                        f"malformed YAML in {path}:{line_number}: duplicate {section}.{key[1]}"
                    )
                result[key] = value
            continue
        # Nested structures and block-list items are irrelevant, but malformed top-level
        # content should not silently turn into defaults.
        if not raw_line.startswith(" "):
            raise AggregationError(f"malformed YAML in {path}:{line_number}: {raw_line!r}")
    return result


def _yaml_string(values: dict[tuple[str, str], str], section: str, key: str, default: str) -> str:
    value = values.get((section, key))
    if value is None:
        return default
    if value.startswith("'"):
        if not value.endswith("'"):
            raise AggregationError(f"invalid YAML string for {section}.{key}: {value!r}")
        return value[1:-1].replace("''", "'")
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise AggregationError(f"invalid YAML scalar for {section}.{key}: {value!r}") from error
        if not isinstance(parsed, str):
            raise AggregationError(f"invalid YAML string for {section}.{key}: {value!r}")
        return parsed
    return value


def _yaml_positive_int(values: dict[tuple[str, str], str], section: str, key: str) -> int:
    value = values.get((section, key))
    try:
        parsed = int(value) if value is not None else None
    except ValueError as error:
        raise AggregationError(f"invalid YAML integer for {section}.{key}: {value!r}") from error
    if parsed is None or parsed <= 0 or str(parsed) != value:
        raise AggregationError(f"{section}.{key} must be a positive integer in config.yaml")
    return parsed


def _yaml_bool(
    values: dict[tuple[str, str], str], section: str, key: str, default: bool
) -> bool:
    value = values.get((section, key))
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"true", "yes", "on"}:
        return True
    if normalized in {"false", "no", "off"}:
        return False
    raise AggregationError(f"invalid YAML boolean for {section}.{key}: {value!r}")


def _mapping(value: Any, path: Path, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AggregationError(f"{path}: {description} must be a JSON object")
    return value


def _number(mapping: dict[str, Any], key: str, path: Path) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AggregationError(f"{path}: {key!r} must be a finite number")
    return float(value)


def _step(mapping: dict[str, Any], path: Path) -> int:
    value = _number(mapping, "step", path)
    if value <= 0 or not value.is_integer():
        raise AggregationError(f"{path}: 'step' must be a positive integer")
    return int(value)


def _name_list(mapping: dict[str, Any], key: str, path: Path) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise AggregationError(f"{path}: {key!r} must be a non-empty list of strings")
    return value


def _linear_slope(records: list[dict[str, Any]], path: Path) -> float | str:
    final_step = _step(records[-1], path)
    points = [
        (_step(record, path), _number(record, "train_psnr", path))
        for record in records
        if _step(record, path) >= final_step - 1000
    ]
    if len(points) < 2:
        return ""
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0.0:
        return ""
    return 1000.0 * sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    ) / denominator


def summarize_run(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise AggregationError(f"run directory does not exist or is not a directory: {run_dir}")

    config_path = run_dir / "config.yaml"
    geometry_path = run_dir / "triplet_geometry.json"
    metrics_path = run_dir / "metrics.json"
    config = _yaml_scalars(config_path)
    geometry = _mapping(_read_json(geometry_path), geometry_path, "triplet geometry")
    raw_metrics = _read_json(metrics_path)
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise AggregationError(f"{metrics_path}: metrics must be a non-empty JSON array")
    records = [_mapping(record, metrics_path, f"metric record {index}") for index, record in enumerate(raw_metrics)]

    previous_step = 0
    for record in records:
        step = _step(record, metrics_path)
        _number(record, "train_psnr", metrics_path)
        if step <= previous_step:
            raise AggregationError(f"{metrics_path}: metric steps must be strictly increasing")
        previous_step = step

    final = records[-1]
    best_train = max(records, key=lambda record: _number(record, "train_psnr", metrics_path))
    support_records = [record for record in records if "support_psnr" in record]
    best_support = (
        max(support_records, key=lambda record: _number(record, "support_psnr", metrics_path))
        if support_records
        else None
    )
    angles = _mapping(geometry.get("view_angle_degrees"), geometry_path, "view_angle_degrees")
    if not angles:
        raise AggregationError(f"{geometry_path}: view_angle_degrees must not be empty")
    max_angle = max(_number(angles, key, geometry_path) for key in angles)

    scene_name = geometry.get("scene_id")
    if not isinstance(scene_name, str) or not scene_name:
        raise AggregationError(f"{geometry_path}: 'scene_id' must be a non-empty string")
    context_names = _name_list(geometry, "context_names", geometry_path)
    target_names = _name_list(geometry, "target_names", geometry_path)

    return {
        "run_name": run_dir.name,
        "fusion_mode": _yaml_string(config, "head", "fusion_mode", "none"),
        "visibility_mask": _yaml_bool(config, "train", "visibility_mask", False),
        "support_mask_contexts": _yaml_string(config, "train", "support_mask_contexts", "all"),
        "scene_name": scene_name,
        "context_names": ";".join(context_names),
        "target_names": ";".join(target_names),
        "steps": _yaml_positive_int(config, "train", "steps"),
        "context_baseline": _number(geometry, "context_baseline", geometry_path),
        "target_interpolation": _number(geometry, "target_interpolation", geometry_path),
        "perpendicular_fraction": _number(geometry, "target_perpendicular_fraction", geometry_path),
        "max_view_angle": max_angle,
        "depth_alignment_scale": _number(final, "depth_alignment_scale", metrics_path),
        "canonical_support_fraction": (
            _number(final, "canonical_support_fraction", metrics_path)
            if "canonical_support_fraction" in final
            else ""
        ),
        "visibility_mask_fraction": (
            _number(final, "visibility_mask_fraction", metrics_path)
            if "visibility_mask_fraction" in final
            else ""
        ),
        "final_train_psnr": _number(final, "train_psnr", metrics_path),
        "best_train_psnr": _number(best_train, "train_psnr", metrics_path),
        "best_train_psnr_step": _step(best_train, metrics_path),
        "final_support_psnr": (
            _number(final, "support_psnr", metrics_path) if "support_psnr" in final else ""
        ),
        "best_support_psnr": (
            _number(best_support, "support_psnr", metrics_path) if best_support else ""
        ),
        "best_support_psnr_step": _step(best_support, metrics_path) if best_support else "",
        "final_alpha": _number(final, "render_alpha_mean", metrics_path),
        "psnr_slope_db_per_1000": _linear_slope(records, metrics_path),
    }


def render_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _display(value: Any, column: str) -> str:
    if value == "":
        return ""
    if column in {"context_names", "target_names"}:
        value = "; ".join(Path(name).name for name in str(value).split(";"))
    elif isinstance(value, bool):
        value = "yes" if value else "no"
    elif isinstance(value, float):
        value = str(int(value)) if value.is_integer() and column.endswith("step") else f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(rows: list[dict[str, Any]]) -> str:
    headings = [MARKDOWN_LABELS[column] for column in COLUMNS]
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_display(row[column], column) for column in COLUMNS) + " |"
        for row in rows
    )
    lines.extend(
        [
            "",
            "_Blank support metrics mean that an older run predated support-metric logging._",
            "",
        ]
    )
    return "\n".join(lines)


def _write_output(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        raise AggregationError(f"cannot write {path}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, metavar="RUN_DIR")
    parser.add_argument("--output-csv", type=Path, help="write the full-filename CSV to PATH")
    parser.add_argument("--output-markdown", type=Path, help="write the readable Markdown table to PATH")
    args = parser.parse_args(argv)

    try:
        rows = [summarize_run(run_dir) for run_dir in args.run_dirs]
        if args.output_csv is not None:
            _write_output(args.output_csv, render_csv(rows))
        if args.output_markdown is not None:
            _write_output(args.output_markdown, render_markdown(rows))
        if args.output_csv is None and args.output_markdown is None:
            sys.stdout.write(render_markdown(rows))
    except AggregationError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
