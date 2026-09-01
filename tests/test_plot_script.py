from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_static_spectrum_plot_is_generated(tmp_path: Path) -> None:
    output_path = tmp_path / "spectrum.png"
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_spectrum.py",
            "--config",
            "configs/static.yaml",
            "--output",
            str(output_path),
        ],
        check=True,
    )
    assert output_path.is_file()
    assert output_path.stat().st_size > 10_000
