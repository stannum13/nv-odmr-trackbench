"""Command-line summaries for deterministic synthetic emulation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from odmr_bench.cli import _load_drift_scenario, main
from odmr_bench.emulator import ODMRInstrument


def _write_config(
    path: Path,
    *,
    resonance_count: int = 8,
    first_integration_time_s: str = "0.01",
    second_frequency_hz: str = "2760000000.0",
    second_integration_time_s: str = "0.02",
) -> None:
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
                f"    integration_time_s: {first_integration_time_s}",
                f"  - frequency_hz: {second_frequency_hz}",
                f"    integration_time_s: {second_integration_time_s}",
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


def test_simulate_reports_invalid_configuration_at_cli_boundary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "invalid.yaml"
    _write_config(config, resonance_count=7)

    assert main(["simulate", "--config", str(config)]) == 2
    assert "resonances must contain exactly eight entries" in capsys.readouterr().err


def test_simulate_validates_a_later_invalid_query_before_any_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "invalid-later-query.yaml"
    _write_config(config, second_frequency_hz=".nan")
    query_calls = 0

    def record_query(*args: object, **kwargs: object) -> object:
        nonlocal query_calls
        query_calls += 1
        raise AssertionError("a query must not run for an invalid schedule")

    monkeypatch.setattr(ODMRInstrument, "query", record_query)

    assert main(["simulate", "--config", str(config)]) == 2
    assert query_calls == 0
    assert "queries[1].frequency_hz must be finite" in capsys.readouterr().err


def test_simulate_canonicalizes_query_scalars_before_constructing_instrument(
    tmp_path: Path,
) -> None:
    config = tmp_path / "integer-queries.yaml"
    _write_config(config, second_frequency_hz="2760000000")

    instrument, queries, _ = _load_drift_scenario(config)

    assert isinstance(queries[0]["frequency_hz"], float)
    assert isinstance(queries[0]["integration_time_s"], float)
    assert instrument.virtual_time_s == 0.0


def test_simulate_rejects_a_nonfinite_prospective_schedule_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "overflowing-schedule.yaml"
    _write_config(
        config,
        first_integration_time_s="1.0e+308",
        second_integration_time_s="1.0e+308",
    )

    def forbid_construction(*args: object, **kwargs: object) -> object:
        raise AssertionError("instrument must not be constructed")

    monkeypatch.setattr("odmr_bench.cli.ODMRInstrument", forbid_construction)

    assert main(["simulate", "--config", str(config)]) == 2
    assert (
        "query schedule virtual timestamps must remain finite"
        in capsys.readouterr().err
    )
