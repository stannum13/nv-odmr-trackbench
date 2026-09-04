"""Tests for estimator-safe two-point resource accounting."""

from __future__ import annotations

import math

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators.two_point_resources import (
    _advance_public_resources,
    _replay_public_resources,
    _zero_public_resources,
)


def make_estimator_observation(
    *,
    sequence_index: int,
    integration_time_s: float,
    realized_photons: int | None,
) -> EstimatorObservation:
    return EstimatorObservation(
        sequence_index=sequence_index,
        timestamp_s=(sequence_index + 1) * integration_time_s,
        frequency_hz=2.80e9 + sequence_index * 1.0e6,
        fluorescence=0.98,
        integration_time_s=integration_time_s,
        nominal_exposure_photons=12_500.0,
        realized_photons=realized_photons,
    )


def test_public_resource_replay_uses_one_arrival_order_atom() -> None:
    observations = tuple(
        make_estimator_observation(
            sequence_index=index,
            integration_time_s=0.005,
            realized_photons=realized,
        )
        for index, realized in enumerate((None, 3, None, 5, 7, None))
    )
    total = _zero_public_resources()
    for observation in observations:
        total = _advance_public_resources(total, observation, 0.001)
    assert total == _replay_public_resources(observations, 0.001)
    assert total.integration_time_s.hex() == "0x1.eb851eb851eb9p-6"
    assert total.integration_time_s != math.fsum([0.005] * 6)
    assert total.virtual_elapsed_time_s.hex() == "0x1.26e978d4fdf3bp-5"
    assert total.realized_photons == 15
    assert total.observations_without_realized_counts == 3
