"""Atomic construction-failure tests for the calibrated two-point tracker."""

from __future__ import annotations

import pytest

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators import (
    CalibratedTwoPointTracker,
    TwoPointBudgetCeiling,
    TwoPointRunMetadata,
    TwoPointUpdateConstructionError,
    calibrate_two_point,
)
from odmr_bench.estimators import two_point_tracker as tracker_module
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_tracker_configuration,
)


def _tracker_with_pending_first_query():
    configuration = make_legal_tracker_configuration()
    calibration = calibrate_two_point(
        make_legal_caller_asserted_source(),
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    tracker = CalibratedTwoPointTracker(configuration)
    tracker.reset(
        TwoPointRunMetadata(
            "clock",
            None,
            0.010,
            2.5e6,
            0.0,
            "normalized_fluorescence",
        ),
        calibration,
        TwoPointBudgetCeiling(2, None, None, None),
        seed=23,
    )
    query = tracker.choose_next_query()
    assert query is not None
    observation = EstimatorObservation(
        query.expected_sequence_index,
        query.expected_end_timestamp_s,
        query.frequency_hz,
        0.9,
        query.integration_time_s,
        query.expected_nominal_exposure_photons,
        13,
    )
    return tracker, observation


def _tracker_snapshot(tracker: CalibratedTwoPointTracker) -> tuple[object, ...]:
    return (
        tracker.configuration,
        tracker.calibration,
        tracker.pending_query,
        tracker.pair_history,
        tracker.estimate(),
    )


@pytest.mark.parametrize(
    ("constructor_name", "failure_call"),
    (
        pytest.param("TwoPointPartialPair", 1, id="partial-pair"),
        pytest.param("PublicAcquisitionResources", 1, id="tracking-resource"),
        pytest.param(
            "PublicAcquisitionResources", 2, id="charged-resource-second-call"
        ),
        pytest.param("TwoPointEstimate", 1, id="aggregate-estimate"),
    ),
)
def test_first_side_construction_faults_roll_back_every_field(
    monkeypatch: pytest.MonkeyPatch, constructor_name: str, failure_call: int
) -> None:
    tracker, observation = _tracker_with_pending_first_query()
    before = _tracker_snapshot(tracker)
    original_constructor = getattr(tracker_module, constructor_name)
    construction_calls = 0

    def fail_construction(*args, **kwargs):
        nonlocal construction_calls
        construction_calls += 1
        if construction_calls == failure_call:
            raise ValueError(
                f"injected {constructor_name} call {failure_call} failure"
            )
        return original_constructor(*args, **kwargs)

    monkeypatch.setattr(tracker_module, constructor_name, fail_construction)

    with pytest.raises(TwoPointUpdateConstructionError) as caught:
        tracker.update(observation)
    assert caught.value.code == "partial_pair_construction_failed"
    assert caught.value.message
    assert caught.value.__cause__ is not None
    assert str(caught.value.__cause__)
    assert construction_calls == failure_call
    assert _tracker_snapshot(tracker) == before
