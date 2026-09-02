from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from odmr_bench.dynamics import SpectralDynamics, SpectralSnapshot, StationaryDynamics
from odmr_bench.models import Baseline, Resonance


def _baseline() -> Baseline:
    return Baseline(intercept=1.0, reference_hz=2.87e9)


def _eight_resonances() -> tuple[Resonance, ...]:
    return tuple(
        Resonance(
            resonance_id=f"nv-{index}",
            center_hz=2.80e9 + index * 20.0e6,
            fwhm_hz=2.0e6,
            amplitude=0.01 + index * 0.001,
            eta=0.35,
        )
        for index in range(8)
    )


@pytest.mark.parametrize("count", [7, 9])
def test_snapshot_requires_exactly_eight_resonances(count: int) -> None:
    resonances = _eight_resonances()
    candidate = (
        resonances[:-1] if count == 7 else (*resonances, resonances[-1])
    )

    with pytest.raises(ValueError, match="eight"):
        SpectralSnapshot(_baseline(), candidate)


def test_snapshot_rejects_duplicate_ids_and_nonpositive_centers() -> None:
    resonances = _eight_resonances()
    duplicate_ids = (
        replace(resonances[0], resonance_id=resonances[1].resonance_id),
        *resonances[1:],
    )
    with pytest.raises(ValueError, match="unique"):
        SpectralSnapshot(_baseline(), duplicate_ids)

    nonpositive_center = (
        replace(resonances[0], center_hz=0.0),
        *resonances[1:],
    )
    with pytest.raises(ValueError, match="positive"):
        SpectralSnapshot(_baseline(), nonpositive_center)


def test_snapshot_copies_resonances_to_an_immutable_tuple() -> None:
    input_resonances = _eight_resonances()

    snapshot = SpectralSnapshot(_baseline(), input_resonances)

    assert snapshot.resonances == input_resonances
    assert snapshot.resonances is not input_resonances
    assert isinstance(snapshot.resonances, tuple)
    with pytest.raises(AttributeError):
        snapshot.resonances.append(input_resonances[0])  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "timestamp_s",
    [True, np.bool_(False), -1.0, np.nan, np.inf, -np.inf],
)
def test_stationary_dynamics_rejects_invalid_virtual_timestamps(
    timestamp_s: object,
) -> None:
    dynamics = StationaryDynamics(SpectralSnapshot(_baseline(), _eight_resonances()))

    with pytest.raises((TypeError, ValueError), match="timestamp_s"):
        dynamics.snapshot_at(timestamp_s)  # type: ignore[arg-type]


def test_stationary_dynamics_returns_new_validated_snapshots_without_reordering(
) -> None:
    initial = SpectralSnapshot(_baseline(), _eight_resonances())
    dynamics = StationaryDynamics(initial)

    first = dynamics.snapshot_at(0.0)
    later = dynamics.snapshot_at(12.5)

    assert isinstance(dynamics, SpectralDynamics)
    assert first is not initial
    assert later is not first
    assert first == later == initial
    assert [resonance.resonance_id for resonance in later.resonances] == [
        resonance.resonance_id for resonance in initial.resonances
    ]
