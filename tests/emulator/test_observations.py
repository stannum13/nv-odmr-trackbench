from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from odmr_bench.emulator.observations import (
    EstimatorObservation,
    InstrumentObservation,
)


def _observation(**overrides: object) -> InstrumentObservation:
    values: dict[str, object] = {
        "sequence_index": np.int64(3),
        "timestamp_s": np.float64(1.5),
        "frequency_hz": np.float64(2.87e9),
        "fluorescence": np.float64(0.96),
        "integration_time_s": np.float64(0.02),
        "nominal_exposure_photons": np.float64(20_000.0),
        "expected_photons": np.float64(19_200.0),
        "realized_photons": np.int64(19_204),
        "sampling_rule": "poisson",
    }
    values.update(overrides)
    return InstrumentObservation(**values)  # type: ignore[arg-type]


def test_full_observation_canonicalizes_fields_and_creates_safe_frozen_view() -> None:
    observation = _observation()

    estimator_observation = observation.estimator_view()

    assert observation.sequence_index == 3
    assert type(observation.timestamp_s) is float
    assert type(observation.realized_photons) is int
    assert estimator_observation == EstimatorObservation(
        sequence_index=3,
        timestamp_s=1.5,
        frequency_hz=2.87e9,
        fluorescence=0.96,
        integration_time_s=0.02,
        nominal_exposure_photons=20_000.0,
        realized_photons=19_204,
    )
    assert estimator_observation is not observation
    with pytest.raises(FrozenInstanceError):
        estimator_observation.fluorescence = 0.0  # type: ignore[misc]


def test_estimator_view_has_no_hidden_truth_or_randomness_fields() -> None:
    field_names = {field.name for field in fields(EstimatorObservation)}

    assert field_names == {
        "sequence_index",
        "timestamp_s",
        "frequency_hz",
        "fluorescence",
        "integration_time_s",
        "nominal_exposure_photons",
        "realized_photons",
    }
    for forbidden in (
        "expected_photons",
        "noiseless_fluorescence",
        "snapshot",
        "dynamics",
        "rng",
        "sampling_rule",
    ):
        assert forbidden not in field_names
        assert not hasattr(_observation().estimator_view(), forbidden)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sequence_index": -1},
        {"sequence_index": True},
        {"timestamp_s": -0.1},
        {"timestamp_s": np.nan},
        {"frequency_hz": np.inf},
        {"fluorescence": np.nan},
        {"integration_time_s": 0.0},
        {"nominal_exposure_photons": -1.0},
        {"expected_photons": -1.0},
        {"realized_photons": -1},
        {"realized_photons": 1.5},
        {"sampling_rule": ""},
    ],
)
def test_full_observation_rejects_invalid_or_nonphysical_fields(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _observation(**overrides)
