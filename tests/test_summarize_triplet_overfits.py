import csv
import importlib.util
import json
from io import StringIO
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "summarize_triplet_overfits.py"
_SPEC = importlib.util.spec_from_file_location("summarize_triplet_overfits", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
AggregationError = _MODULE.AggregationError
render_csv = _MODULE.render_csv
render_markdown = _MODULE.render_markdown
summarize_run = _MODULE.summarize_run


def _write_run(tmp_path, *, support=True):
    run_dir = tmp_path / "example_run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "head:\n"
        "  fusion_mode: projected\n"
        "train:\n"
        "  steps: 1001\n"
        "  visibility_mask: true\n"
        "  support_mask_contexts: canonical\n"
    )
    (run_dir / "triplet_geometry.json").write_text(
        json.dumps(
            {
                "scene_id": "scene-a",
                "context_names": ["images/context-0.JPG", "images/context-1.JPG"],
                "target_names": ["images/target.JPG"],
                "context_baseline": 2.0,
                "target_interpolation": 0.4,
                "target_perpendicular_fraction": 0.1,
                "view_angle_degrees": {
                    "context_context": 12.0,
                    "context0_target": 5.0,
                    "context1_target": 7.0,
                },
            }
        )
    )
    records = []
    for step, psnr in [(1, 20.0), (501, 21.0), (1001, 22.0)]:
        record = {
            "step": step,
            "train_psnr": psnr,
            "render_alpha_mean": 0.9,
            "depth_alignment_scale": 1.1,
            "canonical_support_fraction": 0.8,
            "visibility_mask_fraction": 0.7,
        }
        if support:
            record["support_psnr"] = psnr + 1.0
        records.append(record)
    (run_dir / "metrics.json").write_text(json.dumps(records))
    return run_dir


def test_summarize_run_computes_best_final_and_slope(tmp_path):
    row = summarize_run(_write_run(tmp_path))

    assert row["run_name"] == "example_run"
    assert row["fusion_mode"] == "projected"
    assert row["visibility_mask"] is True
    assert row["support_mask_contexts"] == "canonical"
    assert row["max_view_angle"] == 12.0
    assert row["final_train_psnr"] == 22.0
    assert row["best_train_psnr_step"] == 1001.0
    assert row["final_support_psnr"] == 23.0
    assert row["psnr_slope_db_per_1000"] == pytest.approx(2.0)


def test_csv_keeps_paths_and_markdown_shortens_them(tmp_path):
    row = summarize_run(_write_run(tmp_path, support=False))

    csv_row = next(csv.DictReader(StringIO(render_csv([row]))))
    assert csv_row["context_names"] == "images/context-0.JPG;images/context-1.JPG"
    markdown = render_markdown([row])
    assert "images/context-0.JPG" not in markdown
    assert "context-0.JPG; context-1.JPG" in markdown
    assert "predated support-metric logging" in markdown
    assert row["final_support_psnr"] == ""


def test_missing_and_malformed_inputs_have_contextual_errors(tmp_path):
    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    with pytest.raises(AggregationError, match="missing required file.*config.yaml"):
        summarize_run(run_dir)

    (run_dir / "config.yaml").write_text("head:\n  fusion_mode: none\n")
    (run_dir / "triplet_geometry.json").write_text("not JSON")
    with pytest.raises(AggregationError, match="malformed JSON.*triplet_geometry.json"):
        summarize_run(run_dir)
