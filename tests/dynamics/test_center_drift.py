from __future__ import annotations

import numpy as np
import pytest

from odmr_bench.dynamics import LinearCenterDrift, SpectralSnapshot
from odmr_bench.models import Baseline, Resonance


def _initial_snapshot() -> SpectralSnapshot:
    resonances = tuple(
        Resonance(
            resonance_id=f"nv-{index}",
            center_hz=2.80e9 + index * 20.0e6,
            fwhm_hz=2.0e6 + index,
            amplitude=0.01 + index * 0.001,
            eta=0.1 * index,
        )
        for index in range(8)
    )
    return SpectralSnapshot(
        baseline=Baseline(intercept=1.0, reference_hz=2.87e9),
        resonances=(
            resonances[4],
            resonances[1],
            resonances[7],
            resonances[0],
            *resonances[2:4],
            *resonances[5:7],
        ),
    )


def test_common_slew_moves_all_centers_without_changing_ids_or_other_parameters(
) -> None:
    initial = _initial_snapshot()
    dynamics = LinearCenterDrift(initial, center_slew_hz_per_s=-125.0)

    snapshot = dynamics.snapshot_at(4.0)

    assert snapshot.baseline is initial.baseline
    assert [resonance.resonance_id for resonance in snapshot.resonances] == [
        resonance.resonance_id for resonance in initial.resonances
    ]
    for original, drifted in zip(initial.resonances, snapshot.resonances, strict=True):
        assert drifted.center_hz == original.center_hz - 500.0
        assert drifted.fwhm_hz == original.fwhm_hz
        assert drifted.amplitude == original.amplitude
        assert drifted.eta == original.eta


def test_complete_per_id_slew_map_is_deterministic_and_does_not_sort_by_frequency(
) -> None:
    initial = _initial_snapshot()
    slews = {
        resonance.resonance_id: float(index - 3) * 10.0
        for index, resonance in enumerate(initial.resonances)
    }
    dynamics = LinearCenterDrift(initial, center_slew_hz_per_s=slews)

    first = dynamics.snapshot_at(3.5)
    second = dynamics.snapshot_at(3.5)

    assert first == second
    assert [resonance.resonance_id for resonance in first.resonances] == [
        resonance.resonance_id for resonance in initial.resonances
    ]
    for original, drifted in zip(initial.resonances, first.resonances, strict=True):
        expected_center = original.center_hz + slews[original.resonance_id] * 3.5
        assert drifted.center_hz == expected_center


@pytest.mark.parametrize("configuration", ["missing", "extra"])
def test_per_id_slew_map_requires_exactly_the_snapshot_ids(configuration: str) -> None:
    initial = _initial_snapshot()
    slews = {resonance.resonance_id: 1.0 for resonance in initial.resonances}
    if configuration == "missing":
        del slews[initial.resonances[0].resonance_id]
    else:
        slews["not-a-parent-id"] = 1.0

    with pytest.raises(ValueError, match="exactly"):
        LinearCenterDrift(initial, center_slew_hz_per_s=slews)


@pytest.mark.parametrize("invalid", [True, np.bool_(False), np.nan, np.inf, -np.inf])
def test_linear_drift_rejects_boolean_and_nonfinite_slews(invalid: object) -> None:
    initial = _initial_snapshot()

    with pytest.raises((TypeError, ValueError), match="center_slew_hz_per_s"):
        LinearCenterDrift(initial, center_slew_hz_per_s=invalid)  # type: ignore[arg-type]

    slews = {resonance.resonance_id: 1.0 for resonance in initial.resonances}
    slews[initial.resonances[0].resonance_id] = invalid  # type: ignore[assignment]
    with pytest.raises((TypeError, ValueError), match="center_slew_hz_per_s"):
        LinearCenterDrift(initial, center_slew_hz_per_s=slews)


def test_linear_drift_rejects_a_query_with_nonpositive_center() -> None:
    initial = _initial_snapshot()
    dynamics = LinearCenterDrift(initial, center_slew_hz_per_s=-1.0e9)

    with pytest.raises(ValueError, match="positive"):
        dynamics.snapshot_at(3.0)
