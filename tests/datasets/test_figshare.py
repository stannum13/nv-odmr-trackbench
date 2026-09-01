"""Tests for parsing Figshare-format raw ODMR sweep files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from odmr_bench.datasets import load_figshare_28788437, parse_figshare_sweep_file


def _write_sweep_file(
    path: Path,
    *,
    sweep_count: int = 2,
    stop_hz: float = 30.0,
    rows: str = "1.0\t2.0\t3.0\n4.0\t5.0\t6.0\n",
) -> None:
    path.write_text(
        "\n".join(
            [
                "#Saved Data from the class odmr_logic on 24.12.2024 at 16h51m50s.",
                "#",
                "#Parameters:",
                "#===========",
                f"#Number of frequency sweeps (#): {sweep_count}",
                "#Start Frequencies (Hz): [10.0]",
                f"#Stop Frequencies (Hz): [{stop_hz}]",
                "#Step sizes (Hz): [10.0]",
                "#Clock Frequencies (Hz): 200",
                "#Channel: 0: /Dev1/AI0",
                "#",
                "#Data:",
                "#====",
                "#count data (counts/s)",
                rows.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_parser_preserves_figshare_data_order_and_declared_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny.dat"
    _write_sweep_file(path)

    dataset = parse_figshare_sweep_file(path)

    np.testing.assert_array_equal(
        dataset.signal, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    np.testing.assert_array_equal(dataset.frequency_hz, np.array([10.0, 20.0, 30.0]))
    assert dataset.declared_metadata == {
        "declared_signal_label": "count data (counts/s)",
        "detector_channel": "/Dev1/AI0",
        "nominal_clock_hz": "200",
        "number_of_frequency_sweeps": "2",
        "start_frequency_hz": "10.0",
        "stop_frequency_hz": "30.0",
        "step_size_hz": "10.0",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rows": "1.0\t2.0\n4.0\t5.0\t6.0\n"}, "regular matrix"),
        ({"rows": "1.0\tnan\t3.0\n4.0\t5.0\t6.0\n"}, "finite"),
        ({"stop_hz": 40.0}, "inclusive frequency grid"),
        ({"sweep_count": 3}, "sweep count"),
    ],
)
def test_parser_rejects_malformed_figshare_data(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    path = tmp_path / "malformed.dat"
    _write_sweep_file(path, **kwargs)

    with pytest.raises(ValueError, match=message):
        parse_figshare_sweep_file(path)


def test_verified_loader_rejects_a_tiny_fixture_with_wrong_checksum(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny.dat"
    _write_sweep_file(path)

    with pytest.raises(ValueError, match="checksum"):
        load_figshare_28788437(path)
