"""Scientific invariants for the test-only deterministic linewidth drift."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import pytest

from odmr_bench.dynamics import LinearCenterDrift, SpectralDynamics, SpectralSnapshot
from odmr_bench.models import Baseline, Resonance
from tests.evaluation.sparse_linewidth_fixture_dynamics import (
    DeterministicLinewidthDrift,
)


def _snapshot() -> SpectralSnapshot:
    resonances = tuple(
        Resonance(
            resonance_id=f"r{index}",
            center_hz=2.76e9 + index * 34.0e6,
            fwhm_hz=1.25e6 + index * 1.0e4,
            amplitude=0.012 + index * 0.001,
            eta=index / 10.0,
        )
        for index in range(8)
    )
    return SpectralSnapshot(
        baseline=Baseline(
            intercept=1.1,
            reference_hz=2.88e9,
            slope_per_hz=2.0e-12,
            quadratic_per_hz2=-3.0e-23,
        ),
        resonances=(
            resonances[4],
            resonances[1],
            resonances[7],
            resonances[0],
            resonances[2],
            resonances[6],
            resonances[3],
            resonances[5],
        ),
    )


def _reference_widths(snapshot: SpectralSnapshot) -> dict[str, float]:
    return {
        resonance.resonance_id: resonance.fwhm_hz
        for resonance in snapshot.resonances
    }


def _center_drift() -> LinearCenterDrift:
    initial = _snapshot()
    slews = {
        resonance.resonance_id: float(index - 3) * 125.0
        for index, resonance in enumerate(initial.resonances)
    }
    return LinearCenterDrift(initial, slews)


def test_linewidth_drift_composes_without_changing_base_center() -> None:
    center_drift = _center_drift()
    reference_widths = _reference_widths(_snapshot())
    dynamics = DeterministicLinewidthDrift(center_drift, reference_widths, 100.0)

    actual = dynamics.snapshot_at(2.0)
    base = center_drift.snapshot_at(2.0)

    assert actual.resonances[0].center_hz == base.resonances[0].center_hz
    assert actual.resonances[0].fwhm_hz == reference_widths["r4"] + 200.0


def test_scalar_slew_preserves_base_order_ids_and_all_non_width_fields() -> None:
    center_drift = _center_drift()
    reference_widths = _reference_widths(_snapshot())
    dynamics = DeterministicLinewidthDrift(center_drift, reference_widths, -25.0)

    actual = dynamics.snapshot_at(3.5)
    base = center_drift.snapshot_at(3.5)

    assert actual is not base
    assert actual.baseline == base.baseline
    assert [item.resonance_id for item in actual.resonances] == [
        item.resonance_id for item in base.resonances
    ]
    for observed, base_resonance in zip(
        actual.resonances, base.resonances, strict=True
    ):
        assert observed.center_hz == base_resonance.center_hz
        assert observed.amplitude == base_resonance.amplitude
        assert observed.eta == base_resonance.eta
        assert observed.fwhm_hz == (
            reference_widths[observed.resonance_id] - 87.5
        )


def test_exact_id_slews_are_applied_by_id_without_frequency_sorting() -> None:
    center_drift = _center_drift()
    reference_widths = _reference_widths(_snapshot())
    slews = {
        resonance_id: float(index - 4) * 20.0
        for index, resonance_id in enumerate(reference_widths)
    }
    dynamics = DeterministicLinewidthDrift(center_drift, reference_widths, slews)

    actual = dynamics.snapshot_at(2.5)

    assert tuple(item.resonance_id for item in actual.resonances) == tuple(
        item.resonance_id for item in _snapshot().resonances
    )
    for resonance in actual.resonances:
        assert resonance.fwhm_hz == (
            reference_widths[resonance.resonance_id]
            + slews[resonance.resonance_id] * 2.5
        )


def test_configuration_maps_are_canonical_immutable_copies() -> None:
    center_drift = _center_drift()
    reference_widths = _reference_widths(_snapshot())
    slews = {resonance_id: 10 for resonance_id in reference_widths}
    dynamics = DeterministicLinewidthDrift(center_drift, reference_widths, slews)

    reference_widths["r4"] = 9.9e9
    slews["r4"] = 9_999

    assert dynamics.reference_fwhm_hz["r4"] == pytest.approx(1.29e6)
    assert dynamics.fwhm_slew_hz_per_s["r4"] == 10.0
    assert isinstance(dynamics.reference_fwhm_hz["r4"], float)
    assert isinstance(dynamics.fwhm_slew_hz_per_s["r4"], float)
    with pytest.raises(TypeError):
        dynamics.reference_fwhm_hz["r4"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        dynamics.fwhm_slew_hz_per_s["r4"] = 1.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        dynamics.base_dynamics = center_drift  # type: ignore[misc]


@pytest.mark.parametrize("invalid", [None, object(), 1.0, "dynamics"])
def test_base_dynamics_must_implement_spectral_dynamics(invalid: object) -> None:
    with pytest.raises(TypeError, match="base_dynamics"):
        DeterministicLinewidthDrift(
            cast(SpectralDynamics, invalid), _reference_widths(_snapshot()), 1.0
        )


@pytest.mark.parametrize("invalid", [None, 1.0, "widths", [1.0]])
def test_reference_widths_must_be_a_mapping(invalid: object) -> None:
    with pytest.raises(TypeError, match="reference_fwhm_hz"):
        DeterministicLinewidthDrift(
            _center_drift(), cast(dict[str, float], invalid), 1.0
        )


@pytest.mark.parametrize("invalid", [1, "", "   "])
def test_reference_width_ids_must_be_nonempty_strings(invalid: object) -> None:
    widths: dict[object, float] = _reference_widths(_snapshot())
    widths[invalid] = widths.pop("r4")

    with pytest.raises(TypeError, match="reference_fwhm_hz"):
        DeterministicLinewidthDrift(
            _center_drift(), cast(dict[str, float], widths), 1.0
        )


@pytest.mark.parametrize("invalid", [True, np.bool_(False), "1", None, object()])
def test_reference_width_values_reject_non_real_types(invalid: object) -> None:
    widths: dict[str, object] = _reference_widths(_snapshot())
    widths["r4"] = invalid

    with pytest.raises(TypeError, match="reference_fwhm_hz"):
        DeterministicLinewidthDrift(
            _center_drift(), cast(dict[str, float], widths), 1.0
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_reference_width_values_must_be_finite_and_positive(invalid: float) -> None:
    widths = _reference_widths(_snapshot())
    widths["r4"] = invalid

    with pytest.raises(ValueError, match="reference_fwhm_hz"):
        DeterministicLinewidthDrift(_center_drift(), widths, 1.0)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_reference_width_ids_must_match_base_snapshot_exactly(change: str) -> None:
    center_drift = _center_drift()
    widths = _reference_widths(_snapshot())
    if change == "missing":
        del widths["r4"]
    else:
        widths["unexpected"] = 1.0e6
    dynamics = DeterministicLinewidthDrift(center_drift, widths, 1.0)

    with pytest.raises(ValueError, match=r"reference_fwhm_hz.*exactly"):
        dynamics.snapshot_at(0.0)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_slew_mapping_requires_exactly_the_reference_ids(change: str) -> None:
    widths = _reference_widths(_snapshot())
    slews = {resonance_id: 1.0 for resonance_id in widths}
    if change == "missing":
        del slews["r4"]
    else:
        slews["unexpected"] = 1.0

    with pytest.raises(ValueError, match=r"fwhm_slew_hz_per_s.*exactly"):
        DeterministicLinewidthDrift(_center_drift(), widths, slews)


@pytest.mark.parametrize("invalid", [True, np.bool_(False), "1", None, object()])
def test_scalar_and_mapping_slews_reject_non_real_types(invalid: object) -> None:
    widths = _reference_widths(_snapshot())
    with pytest.raises(TypeError, match="fwhm_slew_hz_per_s"):
        DeterministicLinewidthDrift(_center_drift(), widths, invalid)  # type: ignore[arg-type]

    slews: dict[str, object] = {resonance_id: 1.0 for resonance_id in widths}
    slews["r4"] = invalid
    with pytest.raises(TypeError, match="fwhm_slew_hz_per_s"):
        DeterministicLinewidthDrift(
            _center_drift(), widths, cast(dict[str, float], slews)
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_scalar_and_mapping_slews_reject_nonfinite_values(invalid: float) -> None:
    widths = _reference_widths(_snapshot())
    with pytest.raises(ValueError, match="fwhm_slew_hz_per_s"):
        DeterministicLinewidthDrift(_center_drift(), widths, invalid)

    slews = {resonance_id: 1.0 for resonance_id in widths}
    slews["r4"] = invalid
    with pytest.raises(ValueError, match="fwhm_slew_hz_per_s"):
        DeterministicLinewidthDrift(_center_drift(), widths, slews)


def test_repeated_calls_are_deterministic_and_do_not_mutate_base_input() -> None:
    center_drift = _center_drift()
    initial_before = center_drift.initial_snapshot
    widths = _reference_widths(_snapshot())
    dynamics = DeterministicLinewidthDrift(center_drift, widths, 7.5)

    first = dynamics.snapshot_at(4.0)
    second = dynamics.snapshot_at(4.0)

    assert first == second
    assert first is not second
    assert first.resonances is not second.resonances
    assert center_drift.initial_snapshot == initial_before
    assert tuple(item.fwhm_hz for item in initial_before.resonances) == tuple(
        widths[item.resonance_id] for item in initial_before.resonances
    )


@pytest.mark.parametrize(
    "timestamp, error",
    [(-1.0, ValueError), (np.nan, ValueError), (np.inf, ValueError), (True, TypeError)],
)
def test_invalid_timestamp_is_rejected(
    timestamp: object, error: type[Exception]
) -> None:
    dynamics = DeterministicLinewidthDrift(
        _center_drift(), _reference_widths(_snapshot()), 1.0
    )

    with pytest.raises(error, match="timestamp_s"):
        dynamics.snapshot_at(timestamp)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("reference_width", "slew", "timestamp"),
    [(10.0, -5.0, 2.0), (10.0, -6.0, 2.0), (1.0e308, 1.0e308, 2.0)],
)
def test_generated_width_must_remain_finite_and_positive(
    reference_width: float, slew: float, timestamp: float
) -> None:
    widths = {
        resonance.resonance_id: reference_width
        for resonance in _snapshot().resonances
    }
    dynamics = DeterministicLinewidthDrift(_center_drift(), widths, slew)

    with pytest.raises(ValueError, match=r"fwhm_hz.*finite and positive"):
        dynamics.snapshot_at(timestamp)
