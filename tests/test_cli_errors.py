"""Console-facing expected-failure behavior for CLI inputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("filename", "contents", "arguments", "expected_message"),
    [
        (
            None,
            None,
            ("playback", "--path", "missing.dat"),
            "could not read sweep file",
        ),
        (
            None,
            None,
            ("simulate", "--config", "missing.yaml"),
            "could not read simulation config",
        ),
        (
            "malformed.yaml",
            "baseline: [",
            ("simulate", "--config", "malformed.yaml"),
            "could not parse simulation config",
        ),
        (
            "invalid.yaml",
            "resonances: []\n",
            ("simulate", "--config", "invalid.yaml"),
            "simulation config must contain exactly",
        ),
        (
            "invalid.dat",
            "not a documented sweep\n",
            ("playback", "--path", "invalid.dat"),
            "data cells must be finite floating-point values",
        ),
    ],
)
def test_expected_input_errors_are_concise_at_the_subprocess_boundary(
    tmp_path: Path,
    filename: str | None,
    contents: str | None,
    arguments: tuple[str, ...],
    expected_message: str,
) -> None:
    if filename is not None and contents is not None:
        (tmp_path / filename).write_text(contents, encoding="utf-8")
    executable = Path(sys.executable).with_name("odmrbench")
    completed = subprocess.run(
        [executable, *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith(f"odmrbench: error: {expected_message}")
    assert "Traceback" not in completed.stderr


def test_oversized_yaml_integer_is_a_concise_console_configuration_error(
    tmp_path: Path,
) -> None:
    config = tmp_path / "oversized-integer.yaml"
    source_config = Path("configs/drift.yaml").read_text(encoding="utf-8")
    config.write_text(
        source_config.replace(
            "frequency_overhead_s: 0.001",
            f"frequency_overhead_s: {'9' * 400}",
        ),
        encoding="utf-8",
    )

    executable = Path(sys.executable).with_name("odmrbench")
    completed = subprocess.run(
        [executable, "simulate", "--config", str(config)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith(
        "odmrbench: error: frequency_overhead_s must be finite"
    )
    assert "Traceback" not in completed.stderr
