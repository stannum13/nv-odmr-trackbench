"""Reset precedence and rollback tests for the sparse composite tracker."""

from __future__ import annotations

from copy import copy
from dataclasses import replace

import pytest

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators import (
    PublicAcquisitionResources,
    SparseLinewidthCompositeTracker,
    SparseLinewidthConfiguration,
    SparseLinewidthObservationValidationError,
    SparseLinewidthResetError,
    SparseLinewidthUpdateConstructionError,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointRunMetadata,
    calibrate_two_point,
)
from odmr_bench.estimators import sparse_linewidth_tracker as tracker_module
from odmr_bench.estimators.two_point_calibration import (
    _evaluate_target_only_model,
)
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_tracker_configuration,
)


def _calibration(*, included: bool = False) -> TwoPointCalibration:
    source = make_legal_caller_asserted_source()
    if included:
        source = replace(
            source,
            fluorescence_provenance=replace(
                source.fluorescence_provenance,
                normalization_rule=(
                    "odmr_instrument_normalized_fluorescence_v1"
                ),
            ),
        )
        object.__setattr__(source, "provenance", "verified_factory_acquisition")
    return calibrate_two_point(
        source,
        make_legal_tracker_configuration(),
        budget_treatment=(
            "included_same_run" if included else "conditional_free_precalibration"
        ),
    )


def _metadata(
    calibration: TwoPointCalibration, *, included: bool = False
) -> TwoPointRunMetadata:
    return TwoPointRunMetadata(
        tracker_clock_id=calibration.source.clock_mapping.tracker_clock_id,
        current_sequence_index=(
            calibration.source.availability_sequence_index if included else None
        ),
        current_timestamp_s=(
            calibration.source.availability_timestamp_s if included else 0.020
        ),
        nominal_photon_rate_hz=(
            calibration.source.fluorescence_provenance.nominal_photon_rate_hz
        ),
        frequency_overhead_s=(
            calibration.source.source_frequency_overhead_s if included else 0.001
        ),
        fluorescence_quantity="normalized_fluorescence",
    )


def _valid_tracker() -> SparseLinewidthCompositeTracker:
    calibration = _calibration()
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    tracker.reset(
        _metadata(calibration),
        calibration,
        TwoPointBudgetCeiling(100, None, None, None),
        seed=7,
    )
    assert tracker.choose_next_query() is not None
    return tracker


def _snapshot(tracker: SparseLinewidthCompositeTracker) -> tuple[object, ...]:
    return tracker._configuration, tracker._configuration_snapshot, tracker._state


def _corrupt_calibration(calibration: TwoPointCalibration) -> TwoPointCalibration:
    corrupted = copy(calibration)
    object.__setattr__(corrupted, "identities", tuple(reversed(calibration.identities)))
    return corrupted


def _corrupt_nested_source(
    calibration: TwoPointCalibration, corruption: str
) -> TwoPointCalibration:
    source = copy(calibration.source)
    if corruption == "resource_trace":
        object.__setattr__(
            source,
            "safe_resources",
            PublicAcquisitionResources(0, 0.0, 0.0, 0, 0, 0.0),
        )
    elif corruption == "availability_trace":
        object.__setattr__(source, "availability_timestamp_s", 0.009)
    else:  # pragma: no cover - test helper is closed over the parametrization
        raise AssertionError(f"unsupported corruption: {corruption}")
    corrupted = copy(calibration)
    object.__setattr__(corrupted, "source", source)
    return corrupted


def _invalid_geometry_calibration(
    calibration: TwoPointCalibration,
) -> TwoPointCalibration:
    late = calibration.identities[-1]
    invalid = replace(
        late,
        calibration_cell_lower_hz=late.calibration_center_hz - 600_000.0,
        calibration_cell_upper_hz=late.calibration_center_hz + 600_000.0,
    )
    return replace(calibration, identities=(*calibration.identities[:-1], invalid))


def test_invalid_argument_type_precedes_every_later_reset_failure() -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _invalid_geometry_calibration(_calibration())

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(  # type: ignore[arg-type]
            object(),
            calibration,
            TwoPointBudgetCeiling(0, None, None, None),
            seed=-1,
        )

    assert raised.value.code == "invalid_argument_type"
    assert _snapshot(tracker) == before


def test_configuration_integrity_precedes_calibration_and_metadata() -> None:
    tracker = _valid_tracker()
    object.__setattr__(
        tracker,
        "_configuration",
        replace(tracker._configuration, scan_period_fast_pairs=7),
    )
    before = _snapshot(tracker)
    calibration = _corrupt_calibration(_calibration())

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            replace(_metadata(calibration), tracker_clock_id="wrong-clock"),
            calibration,
            TwoPointBudgetCeiling(0, None, None, None),
            seed=11,
        )

    assert raised.value.code == "configuration_mismatch"
    assert _snapshot(tracker) == before


def test_calibration_mismatch_precedes_metadata_geometry_and_budget() -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _corrupt_calibration(_invalid_geometry_calibration(_calibration()))

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            replace(_metadata(calibration), tracker_clock_id="wrong-clock"),
            calibration,
            TwoPointBudgetCeiling(0, None, None, None),
            seed=11,
        )

    assert raised.value.code == "calibration_mismatch"
    assert _snapshot(tracker) == before


@pytest.mark.parametrize("corruption", ("resource_trace", "availability_trace"))
def test_nested_calibration_source_corruption_rolls_back(
    corruption: str,
) -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _corrupt_nested_source(_calibration(), corruption)

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            _metadata(calibration),
            calibration,
            TwoPointBudgetCeiling(100, None, None, None),
            seed=11,
        )

    assert raised.value.code == "calibration_mismatch"
    assert _snapshot(tracker) == before


def test_metadata_mismatch_precedes_geometry_and_budget() -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _invalid_geometry_calibration(_calibration())

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            replace(_metadata(calibration), tracker_clock_id="wrong-clock"),
            calibration,
            TwoPointBudgetCeiling(0, None, None, None),
            seed=11,
        )

    assert raised.value.code == "metadata_mismatch"
    assert _snapshot(tracker) == before


def test_geometry_mismatch_precedes_budget_and_rolls_back_value_equally() -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _invalid_geometry_calibration(_calibration())

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            _metadata(calibration),
            calibration,
            TwoPointBudgetCeiling(0, None, None, None),
            seed=11,
        )

    assert raised.value.code == "invalid_base_sparse_geometry"
    assert _snapshot(tracker) == before


def test_starting_budget_mismatch_rolls_back_value_equally() -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _calibration(included=True)

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            _metadata(calibration, included=True),
            calibration,
            TwoPointBudgetCeiling(1, None, None, None),
            seed=11,
        )

    assert raised.value.code == "budget_mismatch"
    assert _snapshot(tracker) == before


@pytest.mark.parametrize(
    "constructor_name",
    ("CompositeIdentityEstimate", "SparseLinewidthCompositeEstimate"),
)
def test_initial_state_construction_failure_rolls_back_value_equally(
    monkeypatch: pytest.MonkeyPatch, constructor_name: str
) -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _calibration()

    def fail_construction(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(f"injected {constructor_name} failure")

    monkeypatch.setattr(tracker_module, constructor_name, fail_construction)

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            _metadata(calibration),
            calibration,
            TwoPointBudgetCeiling(100, None, None, None),
            seed=11,
        )

    assert raised.value.code == "initial_state_construction_failed"
    assert raised.value.__cause__ is not None
    assert _snapshot(tracker) == before


def test_budget_stop_record_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration()
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    tracker.reset(
        _metadata(calibration),
        calibration,
        TwoPointBudgetCeiling(1, None, None, None),
        seed=13,
    )
    before = _snapshot(tracker)

    def fail_construction(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected stop construction failure")

    monkeypatch.setattr(
        tracker_module, "SparseLinewidthCompositeEstimate", fail_construction
    )

    with pytest.raises(RuntimeError, match="stop construction"):
        tracker.choose_next_query()

    assert _snapshot(tracker) == before


def _pending_fast_observation(
    tracker: SparseLinewidthCompositeTracker,
) -> EstimatorObservation:
    state = tracker._state
    assert state is not None
    query = tracker.choose_next_query()
    assert query is not None
    cell = state.calibration.identities[query.pair_index % 8]
    fluorescence = _evaluate_target_only_model(
        state.calibration.source.source_fit,
        cell.source_fit_index,
        query.frequency_hz,
        query.interrogation_center_hz,
    )
    return EstimatorObservation(
        query.expected_sequence_index,
        query.expected_end_timestamp_s,
        query.frequency_hz,
        fluorescence,
        query.integration_time_s,
        query.expected_nominal_exposure_photons,
        13,
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        (lambda observation: object(), "invalid_observation_type"),
        (
            lambda observation: replace(
                observation,
                sequence_index=observation.sequence_index + 1,
                frequency_hz=observation.frequency_hz + 1.0,
            ),
            "sequence_mismatch",
        ),
        (
            lambda observation: replace(
                observation,
                frequency_hz=observation.frequency_hz + 1.0,
                integration_time_s=observation.integration_time_s * 2.0,
            ),
            "frequency_mismatch",
        ),
        (
            lambda observation: replace(
                observation,
                integration_time_s=observation.integration_time_s * 2.0,
                timestamp_s=observation.timestamp_s + 1.0,
            ),
            "integration_time_mismatch",
        ),
        (
            lambda observation: replace(
                observation,
                timestamp_s=observation.timestamp_s + 1.0,
                nominal_exposure_photons=(
                    observation.nominal_exposure_photons + 1.0
                ),
            ),
            "endpoint_mismatch",
        ),
        (
            lambda observation: replace(
                observation,
                nominal_exposure_photons=(
                    observation.nominal_exposure_photons + 1.0
                ),
            ),
            "nominal_exposure_mismatch",
        ),
    ),
)
def test_fast_validation_precedence_is_exact_and_atomic(
    mutation, expected_code: str
) -> None:
    tracker = _valid_tracker()
    observation = _pending_fast_observation(tracker)
    before = _snapshot(tracker)

    with pytest.raises(SparseLinewidthObservationValidationError) as raised:
        tracker.update(mutation(observation))

    assert raised.value.code == expected_code
    assert _snapshot(tracker) == before


def test_no_pending_precedes_sequence_and_preserves_corrupted_boundary() -> None:
    tracker = _valid_tracker()
    observation = _pending_fast_observation(tracker)
    state = tracker._state
    assert state is not None
    estimate = replace(
        state.estimate,
        pending_mode=None,
        pending_query=None,
    )
    object.__setattr__(tracker, "_state", replace(state, estimate=estimate))
    before = _snapshot(tracker)
    observation = replace(
        observation, sequence_index=observation.sequence_index + 1
    )

    with pytest.raises(SparseLinewidthObservationValidationError) as raised:
        tracker.update(observation)

    assert raised.value.code == "no_pending_query"
    assert _snapshot(tracker) == before


def test_invalid_type_precedes_no_pending_query() -> None:
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    before = _snapshot(tracker)

    with pytest.raises(SparseLinewidthObservationValidationError) as raised:
        tracker.update(object())  # type: ignore[arg-type]

    assert raised.value.code == "invalid_observation_type"
    assert _snapshot(tracker) == before


def test_pending_mode_precedes_fast_echo_and_sequence() -> None:
    tracker = _valid_tracker()
    observation = _pending_fast_observation(tracker)
    state = tracker._state
    assert state is not None
    estimate = copy(state.estimate)
    object.__setattr__(estimate, "pending_mode", "sparse_scan")
    object.__setattr__(tracker, "_state", replace(state, estimate=estimate))
    before = _snapshot(tracker)
    observation = replace(
        observation, sequence_index=observation.sequence_index + 1
    )

    with pytest.raises(SparseLinewidthObservationValidationError) as raised:
        tracker.update(observation)

    assert raised.value.code == "pending_mode_mismatch"
    assert _snapshot(tracker) == before


def test_fast_reserved_query_echo_precedes_sequence() -> None:
    tracker = _valid_tracker()
    observation = _pending_fast_observation(tracker)
    state = tracker._state
    assert state is not None
    cloned_query = replace(state.estimate.pending_query)
    estimate = replace(state.estimate, pending_query=cloned_query)
    object.__setattr__(tracker, "_state", replace(state, estimate=estimate))
    before = _snapshot(tracker)
    observation = replace(
        observation, sequence_index=observation.sequence_index + 1
    )

    with pytest.raises(SparseLinewidthObservationValidationError) as raised:
        tracker.update(observation)

    assert raised.value.code == "fast_query_echo_mismatch"
    assert _snapshot(tracker) == before


def test_invalid_fast_value_is_last_validation_code_and_atomic() -> None:
    tracker = _valid_tracker()
    observation = _pending_fast_observation(tracker)
    object.__setattr__(observation, "fluorescence", float("nan"))
    object.__setattr__(observation, "realized_photons", -1)
    before = _snapshot(tracker)

    with pytest.raises(SparseLinewidthObservationValidationError) as raised:
        tracker.update(observation)

    assert raised.value.code == "invalid_observation_value"
    assert _snapshot(tracker) == before


def test_nominal_exposure_precedes_invalid_fast_value() -> None:
    tracker = _valid_tracker()
    observation = _pending_fast_observation(tracker)
    observation = replace(
        observation,
        nominal_exposure_photons=observation.nominal_exposure_photons + 1.0,
    )
    object.__setattr__(observation, "fluorescence", float("nan"))
    before = _snapshot(tracker)

    with pytest.raises(SparseLinewidthObservationValidationError) as raised:
        tracker.update(observation)

    assert raised.value.code == "nominal_exposure_mismatch"
    assert _snapshot(tracker) == before


@pytest.mark.parametrize(
    ("constructor_name", "failure_call", "expected_code"),
    (
        ("TwoPointPartialPair", 1, "fast_partial_pair_construction_failed"),
        ("PublicAcquisitionResources", 1, "resource_construction_failed"),
        (
            "SparseLinewidthCompositeEstimate",
            1,
            "aggregate_estimate_construction_failed",
        ),
        (
            "SparseLinewidthCompositeUpdate",
            1,
            "update_construction_failed",
        ),
    ),
)
def test_first_fast_side_construction_codes_and_rollback(
    monkeypatch: pytest.MonkeyPatch,
    constructor_name: str,
    failure_call: int,
    expected_code: str,
) -> None:
    tracker = _valid_tracker()
    observation = _pending_fast_observation(tracker)
    before = _snapshot(tracker)
    original = getattr(tracker_module, constructor_name)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError(f"injected {constructor_name}")
        return original(*args, **kwargs)

    monkeypatch.setattr(tracker_module, constructor_name, fail)

    with pytest.raises(SparseLinewidthUpdateConstructionError) as raised:
        tracker.update(observation)

    assert raised.value.code == expected_code
    assert raised.value.__cause__ is not None
    assert _snapshot(tracker) == before


def _pending_second_fast_observation(
    tracker: SparseLinewidthCompositeTracker,
) -> EstimatorObservation:
    first = _pending_fast_observation(tracker)
    tracker.update(first)
    return _pending_fast_observation(tracker)


@pytest.mark.parametrize(
    ("constructor_name", "failure_call", "expected_code"),
    (
        ("TwoPointPairResult", 1, "fast_pair_result_construction_failed"),
        ("CompositeIdentityEstimate", 1, "fast_identity_estimate_construction_failed"),
        ("PublicAcquisitionResources", 1, "resource_construction_failed"),
        (
            "SparseLinewidthCompositeEstimate",
            1,
            "aggregate_estimate_construction_failed",
        ),
        (
            "SparseLinewidthCompositeUpdate",
            1,
            "update_construction_failed",
        ),
    ),
)
def test_second_fast_side_construction_codes_and_rollback(
    monkeypatch: pytest.MonkeyPatch,
    constructor_name: str,
    failure_call: int,
    expected_code: str,
) -> None:
    tracker = _valid_tracker()
    observation = _pending_second_fast_observation(tracker)
    before = _snapshot(tracker)
    original = getattr(tracker_module, constructor_name)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError(f"injected {constructor_name}")
        return original(*args, **kwargs)

    monkeypatch.setattr(tracker_module, constructor_name, fail)

    with pytest.raises(SparseLinewidthUpdateConstructionError) as raised:
        tracker.update(observation)

    assert raised.value.code == expected_code
    assert raised.value.__cause__ is not None
    assert _snapshot(tracker) == before


@pytest.mark.parametrize(
    ("side", "constructor_name"),
    (
        ("first", "TwoPointPartialPair"),
        ("first", "PublicAcquisitionResources"),
        ("first", "SparseLinewidthCompositeEstimate"),
        ("first", "SparseLinewidthCompositeUpdate"),
        ("second", "TwoPointPairResult"),
        ("second", "CompositeIdentityEstimate"),
        ("second", "PublicAcquisitionResources"),
        ("second", "SparseLinewidthCompositeEstimate"),
        ("second", "SparseLinewidthCompositeUpdate"),
    ),
)
def test_every_fast_construction_stage_preserves_identical_base_exception(
    monkeypatch: pytest.MonkeyPatch,
    side: str,
    constructor_name: str,
) -> None:
    tracker = _valid_tracker()
    observation = (
        _pending_fast_observation(tracker)
        if side == "first"
        else _pending_second_fast_observation(tracker)
    )
    before = _snapshot(tracker)
    injected = KeyboardInterrupt(f"injected {side} {constructor_name}")

    def fail(*args, **kwargs):
        del args, kwargs
        raise injected

    monkeypatch.setattr(tracker_module, constructor_name, fail)

    with pytest.raises(KeyboardInterrupt) as raised:
        tracker.update(observation)

    assert raised.value is injected
    assert _snapshot(tracker) == before
