"""Static guards for code paths unit tests cannot execute.

``train()`` requires CUDA, so a plain undefined name inside it survives the
whole CPU test suite and only fails once a GPU job starts. A targeted
undefined-name check covers those entry points cheaply.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _ruff_available() -> bool:
    if shutil.which("ruff") is not None:
        return True
    probe = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"], capture_output=True
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _ruff_available(), reason="ruff is not installed")
def test_no_undefined_or_duplicate_names():
    # F821 undefined name, F811 redefinition; both survive CPU-only testing of
    # CUDA-gated entry points.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "F821,F811",
            "src",
            "scripts",
            "tests",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
