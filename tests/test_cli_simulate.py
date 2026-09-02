"""Command-line summaries for deterministic synthetic emulation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from odmr_bench.cli import main


def _write_config(path: Path, *, resonance_count: int = 8) -> None:
    resonances = "\n".join(
        f"  - id: r{index}\n"
        f"    center_hz: {2_750_000_000 + index * 10_000_000}.0\n"
        "    fwhm_hz: 1000000.0\n"
        "    amplitude: 0.01\n"
        "    eta: 0.5"
        for index in range(resonance_count)
    )
    path.write_text(
        "\n".join(
            [
                "baseline:",
                "  intercept: 1.0",
                "  reference_hz: 2785000000.0",
                "  slope_per_hz: 0.0",
                "  quadratic_per_hz2: 0.0",
                "resonances:",
                resonances,
                "center_slew_hz_per_s: 1000.0",
                "noise:",
                "  kind: poisson",
                "nominal_photon_rate_hz: 100000.0",
                "frequency_overhead_s: 0.002",
                "seed: 12",
                "queries:",
                "  - frequency_hz: 2750000000.0",
                "    integration_time_s: 0.01",
                "  - frequency_hz: 2760000000.0",
                "    integration_time_s: 0.02",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_simulate_reports_deterministic_synthetic_resource_summary(
    tmp_path: Path, capsys: object
) -> None:
    config = tmp_path / "drift.yaml"
    _write_config(config)

    assert main(["simulate", "--config", str(config)]) == 0
    first = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert main(["simulate", "--config", str(config)]) == 0
    second = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert first == second
    assert first["mode"] == "synthetic_emulation"
    assert first["seed"] == 12
    assert first["query_count"] == 2
    assert first["final_virtual_time_s"] == pytest.approx(0.034)
    assert first["integration_time_s"] == pytest.approx(0.03)
    assert first["nominal_exposure_photons"] == pytest.approx(3000.0)
    assert 0.0 < first["expected_photons"] <= first["nominal_exposure_photons"]
    assert isinstance(first["realized_photons"], int)
    assert first["first_sample"] == {
        "fluorescence": first["first_sample"]["fluorescence"],
        "frequency_hz": 2_750_000_000.0,
        "sequence_index": 0,
        "timestamp_s": 0.012,
    }
    assert first["last_sample"] == {
        "fluorescence": first["last_sample"]["fluorescence"],
        "frequency_hz": 2_760_000_000.0,
        "sequence_index": 1,
        "timestamp_s": 0.034,
    }


def test_simulate_rejects_configurations_without_eight_resonances(
    tmp_path: Path
) -> None:
    config = tmp_path / "invalid.yaml"
    _write_config(config, resonance_count=7)

    with pytest.raises(ValueError, match="exactly eight"):
        main(["simulate", "--config", str(config)])
