"""Command-line summaries for the optional recorded-data path."""

from __future__ import annotations

import json
from pathlib import Path

from odmr_bench.cli import main


def _write_sweep_file(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#Saved Data from the class odmr_logic on 24.12.2024 at 16h51m50s.",
                "#",
                "#Parameters:",
                "#===========",
                "#Number of frequency sweeps (#): 2",
                "#Start Frequencies (Hz): [10.0]",
                "#Stop Frequencies (Hz): [30.0]",
                "#Step sizes (Hz): [10.0]",
                "#Clock Frequencies (Hz): 200",
                "#Channel: 0: /Dev1/AI0",
                "#",
                "#Data:",
                "#====",
                "#count data (counts/s)",
                "1.0\t2.0\t3.0",
                "4.0\t5.0\t6.0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_dataset_info_reports_checked_provenance_and_limitations(
    capsys: object,
) -> None:
    assert main(["dataset-info"]) == 0

    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert payload["doi"] == "10.6084/m9.figshare.28788437.v1"
    assert payload["license"] == "CC BY 4.0"
    assert payload["byte_size"] == 18_974_276
    assert payload["checksum"] == {
        "algorithm": "md5",
        "value": "df03ef2385cdd64d2f0e117ecd9d6c7e",
    }
    assert payload["limitations"] == {
        "ground_truth_status": "none",
        "missing_metadata": [
            "per_sweep_timestamps",
            "current",
            "field",
            "field_direction",
            "resonance_trajectory",
        ],
        "timing_status": "nominal_without_timestamps",
        "unit_status": "conflicted_unverified",
    }


def test_playback_summarizes_raw_fixture_without_inferred_timestamps(
    tmp_path: Path, capsys: object
) -> None:
    path = tmp_path / "tiny.dat"
    _write_sweep_file(path)

    assert main(["playback", "--path", str(path), "--max-observations", "4"]) == 0

    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload == {
        "inferred_timestamps": False,
        "mode": "recorded_playback",
        "observation_count": 4,
        "points_per_sweep": 3,
        "signal_max": 4.0,
        "signal_min": 1.0,
        "sweep_count": 2,
        "timing_status": "nominal_without_timestamps",
        "unit_status": "conflicted_unverified",
    }
