from __future__ import annotations

import subprocess
import sys

import odmr_bench


def test_package_exposes_version() -> None:
    assert odmr_bench.__version__ == "0.1.0"


def test_cli_reports_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "odmr_bench.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "odmrbench 0.1.0"
