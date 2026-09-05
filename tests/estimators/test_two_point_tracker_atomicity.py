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
from odmr_bench.estimators.two_point_calibration import (
    _evaluate_target_only_model,
)
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


def _tracker_with_pending_second_query():
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
    cell = calibration.identities[0]
    first_query = tracker.choose_next_query()
    assert first_query is not None
    first_observation = EstimatorObservation(
        first_query.expected_sequence_index,
        first_query.expected_end_timestamp_s,
        first_query.frequency_hz,
        _evaluate_target_only_model(
            calibration.source.source_fit,
            cell.source_fit_index,
            first_query.frequency_hz,
            first_query.interrogation_center_hz,
        ),
        first_query.integration_time_s,
        first_query.expected_nominal_exposure_photons,
        13,
    )
    tracker.update(first_observation)
    second_query = tracker.choose_next_query()
    assert second_query is not None
    second_observation = EstimatorObservation(
        second_query.expected_sequence_index,
        second_query.expected_end_timestamp_s,
        second_query.frequency_hz,
        _evaluate_target_only_model(
            calibration.source.source_fit,
            cell.source_fit_index,
            second_query.frequency_hz,
            second_query.interrogation_center_hz,
        ),
        second_query.integration_time_s,
        second_query.expected_nominal_exposure_photons,
        17,
    )
    return tracker, second_observation


def _fail_constructor(
    monkeypatch: pytest.MonkeyPatch, constructor_name: str, failure_call: int
) -> None:
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


def _nonfinite_derived_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def overflowing_derivative(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return 1.0e308 if calls == 1 else -1.0e308

    monkeypatch.setattr(
        tracker_module, "_target_center_derivative", overflowing_derivative
    )


def _raise_derived_arithmetic_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_overflow(*args, **kwargs):
        del args, kwargs
        raise OverflowError("injected derived-arithmetic overflow")

    monkeypatch.setattr(tracker_module, "_target_center_derivative", raise_overflow)


def _raise_ordinary_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_model(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected ordinary model exception")

    monkeypatch.setattr(tracker_module, "_evaluate_target_only_model", fail_model)


@pytest.mark.parametrize(
    ("inject_failure", "expected_code"),
    (
        pytest.param(
            lambda monkeypatch: _fail_constructor(
                monkeypatch, "TwoPointPairResult", 1
            ),
            "pair_result_construction_failed",
            id="pair-result",
        ),
        pytest.param(
            lambda monkeypatch: _fail_constructor(
                monkeypatch, "TwoPointIdentityEstimate", 1
            ),
            "identity_estimate_construction_failed",
            id="identity-estimate",
        ),
        pytest.param(
            lambda monkeypatch: _fail_constructor(
                monkeypatch, "PublicAcquisitionResources", 2
            ),
            "resource_construction_failed",
            id="charged-resource",
        ),
        pytest.param(
            lambda monkeypatch: _fail_constructor(
                monkeypatch, "TwoPointEstimate", 1
            ),
            "aggregate_estimate_construction_failed",
            id="aggregate-estimate",
        ),
        pytest.param(
            _raise_derived_arithmetic_overflow,
            "pair_result_construction_failed",
            id="raised-derived-arithmetic-overflow",
        ),
        pytest.param(
            _raise_ordinary_exception,
            "pair_result_construction_failed",
            id="ordinary-exception",
        ),
    ),
)
def test_second_side_construction_faults_roll_back_every_field(
    monkeypatch: pytest.MonkeyPatch, inject_failure, expected_code: str
) -> None:
    tracker, observation = _tracker_with_pending_second_query()
    before = _tracker_snapshot(tracker)
    inject_failure(monkeypatch)

    with pytest.raises(TwoPointUpdateConstructionError) as caught:
        tracker.update(observation)

    assert caught.value.code == expected_code
    assert caught.value.__cause__ is not None
    assert _tracker_snapshot(tracker) == before


def test_second_side_nonfinite_derived_value_commits_scientific_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker, observation = _tracker_with_pending_second_query()
    before = tracker.estimate()
    _nonfinite_derived_arithmetic(monkeypatch)

    update = tracker.update(observation)

    assert update.completed_pair is not None
    assert update.completed_pair.failure_code == "numerical_failure"
    assert update.completed_pair.lock_state == "lost"
    assert update.completed_pair.applied_step_hz == 0.0
    assert tracker.pending_query is None
    assert tracker.estimate().accepted_observations == before.accepted_observations + 1
    assert tracker.estimate().completed_pairs == before.completed_pairs + 1


@pytest.mark.parametrize(
    ("constructor_name", "failure_call", "expected_code"),
    (
        pytest.param(
            "TwoPointPairResult",
            1,
            "pair_result_construction_failed",
            id="pair-result",
        ),
        pytest.param(
            "TwoPointIdentityEstimate",
            1,
            "identity_estimate_construction_failed",
            id="identity-estimate",
        ),
        pytest.param(
            "PublicAcquisitionResources",
            1,
            "resource_construction_failed",
            id="resource-state",
        ),
        pytest.param(
            "TwoPointEstimate",
            1,
            "aggregate_estimate_construction_failed",
            id="aggregate-estimate",
        ),
    ),
)
def test_second_side_construction_exception_codes_follow_stage_order(
    monkeypatch: pytest.MonkeyPatch,
    constructor_name: str,
    failure_call: int,
    expected_code: str,
) -> None:
    tracker, observation = _tracker_with_pending_second_query()
    before = _tracker_snapshot(tracker)
    _fail_constructor(monkeypatch, constructor_name, failure_call)

    with pytest.raises(TwoPointUpdateConstructionError) as caught:
        tracker.update(observation)

    assert caught.value.code == expected_code
    assert caught.value.message
    assert caught.value.__cause__ is not None
    assert str(caught.value.__cause__)
    assert _tracker_snapshot(tracker) == before
