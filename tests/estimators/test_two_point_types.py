"""Tests for calibrated two-point public primitive contracts."""

from concurrent.futures import Future
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

import numpy as np
import pytest

from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_estimate,
    make_legal_fit_configuration,
    make_legal_identity_calibrations,
    make_legal_identity_estimates,
    make_legal_pair_result,
    make_legal_partial_pair,
    make_legal_query,
    make_legal_source_fit,
    make_legal_source_observations,
    make_legal_tracker_configuration,
    make_legal_update,
)


def test_two_point_primitive_names_are_public() -> None:
    from odmr_bench.estimators import (
        CalibrationBudgetTreatment,
        CalibrationIdentityMode,
        CalibrationSourceProvenance,
        ClockMappingKind,
        NormalizedFluorescenceProvenance,
        PairSide,
        PublicAcquisitionResources,
        TwoPointBudgetCeiling,
        TwoPointCalibrationConstructionCode,
        TwoPointCalibrationConstructionError,
        TwoPointClockMapping,
        TwoPointFailureCode,
        TwoPointIdentityBinding,
        TwoPointLockState,
        TwoPointObservationValidationCode,
        TwoPointObservationValidationError,
        TwoPointRunMetadata,
        TwoPointStopReason,
        TwoPointTrackerConfiguration,
        TwoPointUpdateConstructionCode,
        TwoPointUpdateConstructionError,
    )

    assert CalibrationBudgetTreatment
    assert CalibrationIdentityMode
    assert CalibrationSourceProvenance
    assert ClockMappingKind
    assert PublicAcquisitionResources
    assert TwoPointBudgetCeiling
    assert TwoPointIdentityBinding
    assert NormalizedFluorescenceProvenance
    assert PairSide
    assert TwoPointClockMapping
    assert TwoPointTrackerConfiguration
    assert TwoPointRunMetadata
    assert TwoPointCalibrationConstructionCode
    assert TwoPointCalibrationConstructionError
    assert TwoPointFailureCode
    assert TwoPointLockState
    assert TwoPointObservationValidationCode
    assert TwoPointObservationValidationError
    assert TwoPointStopReason
    assert TwoPointUpdateConstructionCode
    assert TwoPointUpdateConstructionError


def test_calibration_record_names_are_public() -> None:
    from odmr_bench.estimators import (
        TwoPointCalibration,
        TwoPointCalibrationSource,
        TwoPointIdentityCalibration,
    )

    assert TwoPointCalibrationSource
    assert TwoPointIdentityCalibration
    assert TwoPointCalibration


def test_two_point_state_record_names_are_public() -> None:
    from odmr_bench.estimators import (
        TwoPointEstimate,
        TwoPointIdentityEstimate,
        TwoPointPairResult,
        TwoPointPartialPair,
        TwoPointQuery,
        TwoPointUpdate,
    )

    assert TwoPointQuery
    assert TwoPointPartialPair
    assert TwoPointPairResult
    assert TwoPointIdentityEstimate
    assert TwoPointEstimate
    assert TwoPointUpdate

    expected_fields = {
        TwoPointQuery: (
            "query_index",
            "pair_index",
            "identity_pair_index",
            "resonance_id",
            "side",
            "interrogation_center_hz",
            "frequency_hz",
            "integration_time_s",
            "expected_sequence_index",
            "expected_end_timestamp_s",
            "expected_nominal_exposure_photons",
        ),
        TwoPointPartialPair: (
            "pair_index",
            "identity_pair_index",
            "resonance_id",
            "interrogation_center_hz",
            "first_side",
            "first_query",
            "first_observation",
        ),
        TwoPointPairResult: (
            "pair_index",
            "identity_pair_index",
            "resonance_id",
            "interrogation_center_hz",
            "first_side",
            "minus_query",
            "plus_query",
            "minus_observation",
            "plus_observation",
            "pair_reference_timestamp_s",
            "release_sequence_index",
            "release_timestamp_s",
            "discriminator",
            "zero_discriminator",
            "discriminator_slope_per_hz",
            "raw_innovation_hz",
            "requested_step_hz",
            "candidate_center_hz",
            "applied_step_hz",
            "common_mode_target_depths",
            "lock_state",
            "failure_code",
        ),
        TwoPointIdentityEstimate: (
            "resonance_id",
            "center_hz",
            "calibration_fwhm_hz",
            "calibration_cell_lower_hz",
            "calibration_cell_upper_hz",
            "allowed_center_min_hz",
            "allowed_center_max_hz",
            "active_source_kind",
            "active_source_pair_index",
            "active_reference_timestamp_s",
            "active_release_sequence_index",
            "active_release_timestamp_s",
            "estimate_age_sequence_indices",
            "estimate_age_s",
            "release_age_s",
            "completed_pairs",
            "lock_state",
            "failure_code",
            "latest_pair",
        ),
        TwoPointEstimate: (
            "identities",
            "calibration_source_id",
            "calibration_source_provenance",
            "calibration_budget_treatment",
            "current_sequence_index",
            "current_timestamp_s",
            "accepted_observations",
            "completed_pairs",
            "incomplete_pair",
            "pending_query",
            "pair_history",
            "tracking_resources",
            "calibration_resources",
            "charged_resources",
            "budget_ceiling",
            "stopped_reason",
            "seed",
        ),
        TwoPointUpdate: ("query", "observation", "completed_pair", "estimate"),
    }
    instances = {
        TwoPointQuery: make_legal_query(),
        TwoPointPartialPair: make_legal_partial_pair(),
        TwoPointPairResult: make_legal_pair_result(),
        TwoPointIdentityEstimate: make_legal_identity_estimates(
            current_timestamp_s=0.0
        )[0],
        TwoPointEstimate: make_legal_estimate(),
        TwoPointUpdate: make_legal_update(),
    }
    for record_type, field_names in expected_fields.items():
        assert tuple(field.name for field in fields(record_type)) == field_names
        assert record_type.__slots__ == field_names
        assert record_type.__dataclass_params__.frozen
        instance = instances[record_type]
        assert not hasattr(instance, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(instance, field_names[0], getattr(instance, field_names[0]))


@pytest.mark.parametrize(
    "case",
    (
        "query_pair_index",
        "query_identity_pair_index",
        "query_side",
        "query_frequency_side",
        "query_expected_endpoint",
        "partial_pair_index",
        "partial_identity_pair_index",
        "partial_resonance_id",
        "partial_center",
        "partial_first_side",
        "partial_observation_echo",
        "pair_pair_index",
        "pair_identity_pair_index",
        "pair_resonance_id",
        "pair_center",
        "pair_first_side",
        "pair_minus_side",
        "pair_plus_side",
        "pair_observation_echo",
        "pair_arrival_order",
        "pair_reference",
        "pair_release_sequence",
        "pair_release_timestamp",
        "tracking_failure",
        "tracking_step",
        "step_limited_step",
        "lost_failure",
        "lost_applied_step",
        "invalid_normalization_diagnostics",
        "common_mode_diagnostics",
    ),
)
def test_query_partial_and_pair_intrinsic_state_matrix(case: str) -> None:
    from odmr_bench.emulator.observations import EstimatorObservation

    query = make_legal_query()
    partial = make_legal_partial_pair()
    pair = make_legal_pair_result()

    assert pair.minus_query.side == "minus"
    assert pair.plus_query.side == "plus"
    second_observation = pair.plus_observation
    assert pair.release_sequence_index == second_observation.sequence_index
    assert pair.release_timestamp_s == second_observation.timestamp_s
    assert make_legal_pair_result(
        pair_index=8, identity_pair_index=1
    ).first_side == "plus"
    assert make_legal_pair_result(lock_state="step_limited")
    assert make_legal_pair_result(
        lock_state="lost", failure_code="invalid_pair_normalization"
    )
    assert make_legal_pair_result(lock_state="lost", failure_code="numerical_failure")
    assert make_legal_pair_result(
        lock_state="lost", failure_code="common_mode_limit_exceeded"
    )
    assert make_legal_pair_result(lock_state="lost", failure_code="capture_exceeded")
    assert make_legal_pair_result(
        lock_state="lost", failure_code="calibration_domain_exceeded"
    )

    wrong_observation = EstimatorObservation(
        2,
        partial.first_observation.timestamp_s,
        partial.first_observation.frequency_hz,
        partial.first_observation.fluorescence,
        partial.first_observation.integration_time_s,
        partial.first_observation.nominal_exposure_photons,
        partial.first_observation.realized_photons,
    )
    invalid_pair = {
        "query_pair_index": (query, {"pair_index": 1}),
        "query_identity_pair_index": (query, {"identity_pair_index": 1}),
        "query_side": (query, {"side": "plus"}),
        "query_frequency_side": (
            query,
            {"frequency_hz": query.interrogation_center_hz},
        ),
        "query_expected_endpoint": (
            query,
            {"expected_end_timestamp_s": 0.001},
        ),
        "partial_pair_index": (partial, {"pair_index": 1}),
        "partial_identity_pair_index": (partial, {"identity_pair_index": 1}),
        "partial_resonance_id": (partial, {"resonance_id": "r1"}),
        "partial_center": (partial, {"interrogation_center_hz": 2.75e9}),
        "partial_first_side": (partial, {"first_side": "plus"}),
        "partial_observation_echo": (
            partial,
            {"first_observation": wrong_observation},
        ),
        "pair_pair_index": (pair, {"pair_index": 1}),
        "pair_identity_pair_index": (pair, {"identity_pair_index": 1}),
        "pair_resonance_id": (pair, {"resonance_id": "r1"}),
        "pair_center": (pair, {"interrogation_center_hz": 2.75e9}),
        "pair_first_side": (pair, {"first_side": "plus"}),
        "pair_minus_side": (pair, {"minus_query": pair.plus_query}),
        "pair_plus_side": (pair, {"plus_query": pair.minus_query}),
        "pair_observation_echo": (pair, {"minus_observation": wrong_observation}),
        "pair_arrival_order": (
            pair,
            {
                "plus_observation": replace(
                    pair.plus_observation,
                    sequence_index=3,
                )
            },
        ),
        "pair_reference": (pair, {"pair_reference_timestamp_s": 0.006}),
        "pair_release_sequence": (pair, {"release_sequence_index": 0}),
        "pair_release_timestamp": (pair, {"release_timestamp_s": 0.006}),
        "tracking_failure": (pair, {"failure_code": "capture_exceeded"}),
        "tracking_step": (pair, {"applied_step_hz": 5_000.0}),
        "step_limited_step": (
            make_legal_pair_result(lock_state="step_limited"),
            {"applied_step_hz": 10_000.0},
        ),
        "lost_failure": (
            make_legal_pair_result(lock_state="lost", failure_code="capture_exceeded"),
            {"failure_code": None},
        ),
        "lost_applied_step": (
            make_legal_pair_result(lock_state="lost", failure_code="capture_exceeded"),
            {"applied_step_hz": 1.0},
        ),
        "invalid_normalization_diagnostics": (
            make_legal_pair_result(
                lock_state="lost", failure_code="invalid_pair_normalization"
            ),
            {"discriminator": 0.0},
        ),
        "common_mode_diagnostics": (
            make_legal_pair_result(
                lock_state="lost", failure_code="common_mode_limit_exceeded"
            ),
            {"common_mode_target_depths": None},
        ),
    }[case]
    with pytest.raises((TypeError, ValueError)):
        replace(invalid_pair[0], **invalid_pair[1])


@pytest.mark.parametrize(
    ("lock_state", "failure_code", "numerical_prefix", "field", "value"),
    (
        ("tracking", None, 0, "common_mode_target_depths", None),
        ("step_limited", None, 0, "common_mode_target_depths", None),
        ("lost", "invalid_pair_normalization", 0, "discriminator", 0.0),
        ("lost", "numerical_failure", 0, "common_mode_target_depths", 0.0),
        ("lost", "numerical_failure", 1, "raw_innovation_hz", 10_000.0),
        ("lost", "numerical_failure", 2, "discriminator", None),
        ("lost", "numerical_failure", 3, "common_mode_target_depths", None),
        ("lost", "numerical_failure", 4, "raw_innovation_hz", None),
        ("lost", "numerical_failure", 5, "requested_step_hz", None),
        ("lost", "common_mode_limit_exceeded", 0, "common_mode_target_depths", None),
        ("lost", "common_mode_limit_exceeded", 0, "raw_innovation_hz", 10_000.0),
        ("lost", "capture_exceeded", 0, "common_mode_target_depths", None),
        ("lost", "capture_exceeded", 0, "requested_step_hz", 10_000.0),
        ("lost", "calibration_domain_exceeded", 0, "common_mode_target_depths", None),
        ("lost", "calibration_domain_exceeded", 0, "candidate_center_hz", None),
    ),
)
def test_pair_diagnostic_prefix_matrix(
    lock_state: str,
    failure_code: str | None,
    numerical_prefix: int,
    field: str,
    value: object,
) -> None:
    pair = make_legal_pair_result(
        lock_state=lock_state,
        failure_code=failure_code,
        numerical_prefix=numerical_prefix,
    )
    with pytest.raises((TypeError, ValueError)):
        replace(pair, **{field: value})


def test_pair_local_geometry_is_optional_only_before_it_is_available() -> None:
    from odmr_bench.estimators import TwoPointPairResult

    assert TwoPointPairResult.__annotations__["zero_discriminator"] == "float | None"
    assert (
        TwoPointPairResult.__annotations__["discriminator_slope_per_hz"]
        == "float | None"
    )

    invalid_normalization = make_legal_pair_result(
        lock_state="lost",
        failure_code="invalid_pair_normalization",
    )
    unavailable_model = make_legal_pair_result(
        lock_state="lost",
        failure_code="numerical_failure",
        numerical_prefix=0,
    )
    assert (
        invalid_normalization.zero_discriminator,
        invalid_normalization.discriminator_slope_per_hz,
    ) == (None, None)
    assert (
        unavailable_model.zero_discriminator,
        unavailable_model.discriminator_slope_per_hz,
    ) == (None, None)

    discriminator_failure = replace(
        unavailable_model,
        zero_discriminator=0.0,
        discriminator_slope_per_hz=1.0e-6,
    )
    assert discriminator_failure.zero_discriminator == 0.0
    assert discriminator_failure.discriminator_slope_per_hz == 1.0e-6

    for changes in (
        {"zero_discriminator": 0.0},
        {"discriminator_slope_per_hz": 1.0e-6},
        {
            "zero_discriminator": 0.0,
            "discriminator_slope_per_hz": 0.0,
        },
        {
            "zero_discriminator": 0.0,
            "discriminator_slope_per_hz": -1.0e-6,
        },
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(unavailable_model, **changes)

    with pytest.raises(ValueError):
        replace(
            invalid_normalization,
            zero_discriminator=0.0,
            discriminator_slope_per_hz=1.0e-6,
        )
    for pair in (
        make_legal_pair_result(),
        make_legal_pair_result(
            lock_state="lost",
            failure_code="common_mode_limit_exceeded",
        ),
        make_legal_pair_result(
            lock_state="lost",
            failure_code="numerical_failure",
            numerical_prefix=1,
        ),
    ):
        with pytest.raises(ValueError):
            replace(
                pair,
                zero_discriminator=None,
                discriminator_slope_per_hz=None,
            )


def test_identity_estimate_aggregate_estimate_and_update_intrinsic_matrix() -> None:
    from odmr_bench.estimators import PublicAcquisitionResources

    boundary = make_legal_estimate()
    first_pending = make_legal_estimate("first")
    partial = make_legal_estimate("partial")
    second_pending = make_legal_estimate("second")
    complete = make_legal_estimate("complete")
    next_pending = make_legal_estimate("next")
    stopped = make_legal_estimate("stopped")
    first_update = make_legal_update("first")
    second_update = make_legal_update("second")

    assert boundary.identities[0].active_reference_timestamp_s == -1.0
    assert boundary.identities[0].estimate_age_s == 1.0
    assert first_pending.pending_query is not None
    assert partial.incomplete_pair is not None
    assert second_pending.incomplete_pair is not None
    assert second_pending.pending_query is not None
    assert complete.pair_history[-1] is complete.identities[0].latest_pair
    assert next_pending.pending_query is not None
    assert stopped.stopped_reason == "budget_exhausted"
    assert first_update.completed_pair is None
    assert first_update.estimate.incomplete_pair is not None
    assert second_update.completed_pair is second_update.estimate.pair_history[-1]

    included_identities = tuple(
        replace(
            identity,
            active_release_sequence_index=1,
            active_release_timestamp_s=0.01,
            estimate_age_sequence_indices=0,
            estimate_age_s=1.01,
            release_age_s=0.0,
        )
        for identity in boundary.identities
    )
    included = replace(
        boundary,
        identities=included_identities,
        calibration_source_provenance="verified_factory_acquisition",
        calibration_budget_treatment="included_same_run",
        current_sequence_index=1,
        current_timestamp_s=0.01,
        charged_resources=boundary.calibration_resources,
    )
    assert included.charged_resources.observations == 2

    identity = boundary.identities[0]
    pair_identity = complete.identities[0]
    invalid_identity_rows = (
        (identity, {"center_hz": identity.allowed_center_max_hz + 1.0}),
        (identity, {"calibration_fwhm_hz": 0.0}),
        (identity, {"active_source_kind": "pair"}),
        (identity, {"active_source_pair_index": 0}),
        (identity, {"active_reference_timestamp_s": float("inf")}),
        (identity, {"active_release_timestamp_s": -1.0}),
        (identity, {"estimate_age_s": -1.0}),
        (identity, {"completed_pairs": 1}),
        (identity, {"lock_state": "lost"}),
        (identity, {"failure_code": "capture_exceeded"}),
        (pair_identity, {"active_reference_timestamp_s": -1.0}),
        (pair_identity, {"active_source_pair_index": None}),
        (pair_identity, {"active_release_sequence_index": None}),
        (pair_identity, {"latest_pair": None}),
    )
    for record, changes in invalid_identity_rows:
        with pytest.raises((TypeError, ValueError)):
            replace(record, **changes)

    wrong_age_identity = replace(
        complete.identities[0],
        estimate_age_s=complete.identities[0].estimate_age_s + 1.0,
    )
    zero_resources = PublicAcquisitionResources(0, 0.0, 0.0, 0, 0, 0.0)
    invalid_estimate_rows = (
        (boundary, {"accepted_observations": 1}),
        (complete, {"completed_pairs": 0}),
        (complete, {"pair_history": ()}),
        (complete, {"identities": (wrong_age_identity, *complete.identities[1:])}),
        (complete, {"identities": (*complete.identities[:-1], complete.identities[0])}),
        (partial, {"incomplete_pair": None}),
        (boundary, {"pending_query": second_pending.pending_query}),
        (stopped, {"pending_query": first_pending.pending_query}),
        (complete, {"tracking_resources": zero_resources}),
        (complete, {"charged_resources": zero_resources}),
        (boundary, {"seed": -1}),
        (boundary, {"calibration_source_id": " "}),
        (
            boundary,
            {"calibration_budget_treatment": "included_same_run"},
        ),
    )
    for record, changes in invalid_estimate_rows:
        with pytest.raises((TypeError, ValueError)):
            replace(record, **changes)

    invalid_update_rows = (
        (first_update, {"query": second_update.query}),
        (first_update, {"observation": second_update.observation}),
        (first_update, {"completed_pair": second_update.completed_pair}),
        (first_update, {"estimate": boundary}),
        (second_update, {"completed_pair": None}),
        (second_update, {"estimate": partial}),
        (second_update, {"query": first_update.query}),
    )
    for record, changes in invalid_update_rows:
        with pytest.raises((TypeError, ValueError)):
            replace(record, **changes)

    assert make_legal_identity_estimates(current_timestamp_s=0.0)


def test_estimate_full_accepted_trace_and_current_endpoint() -> None:
    from odmr_bench.emulator.observations import EstimatorObservation
    from odmr_bench.estimators import (
        PublicAcquisitionResources,
        TwoPointPartialPair,
    )

    complete = make_legal_estimate("complete")

    def shifted_identities(
        current_sequence_index: int,
        current_timestamp_s: float,
    ) -> tuple[object, ...]:
        return tuple(
            replace(
                identity,
                estimate_age_sequence_indices=(
                    None
                    if identity.active_release_sequence_index is None
                    else current_sequence_index
                    - identity.active_release_sequence_index
                ),
                estimate_age_s=(
                    current_timestamp_s - identity.active_reference_timestamp_s
                ),
                release_age_s=(
                    current_timestamp_s - identity.active_release_timestamp_s
                ),
            )
            for identity in complete.identities
        )

    with pytest.raises(ValueError, match="endpoint"):
        replace(
            complete,
            identities=shifted_identities(99, 5.0),
            current_sequence_index=99,
            current_timestamp_s=5.0,
        )

    target = complete.identities[1]

    def estimate_with_partial(
        sequence_index: int,
        timestamp_s: float,
    ) -> object:
        query = make_legal_query(
            query_index=2,
            pair_index=1,
            resonance_id=target.resonance_id,
            interrogation_center_hz=target.center_hz,
            expected_sequence_index=sequence_index,
            expected_end_timestamp_s=timestamp_s,
        )
        observation = EstimatorObservation(
            sequence_index,
            timestamp_s,
            query.frequency_hz,
            0.98,
            query.integration_time_s,
            query.expected_nominal_exposure_photons,
            12_250,
        )
        partial = TwoPointPartialPair(
            pair_index=query.pair_index,
            identity_pair_index=query.identity_pair_index,
            resonance_id=query.resonance_id,
            interrogation_center_hz=query.interrogation_center_hz,
            first_side=query.side,
            first_query=query,
            first_observation=observation,
        )
        resources = PublicAcquisitionResources(
            3,
            0.015,
            37_500.0,
            36_875,
            0,
            timestamp_s,
        )
        return replace(
            complete,
            identities=shifted_identities(sequence_index, timestamp_s),
            current_sequence_index=sequence_index,
            current_timestamp_s=timestamp_s,
            accepted_observations=3,
            incomplete_pair=partial,
            tracking_resources=resources,
            charged_resources=resources,
        )

    assert estimate_with_partial(2, 0.018)
    with pytest.raises(ValueError, match="contiguous"):
        estimate_with_partial(99, 5.0)


def test_pending_and_partial_interrogation_centers_echo_target_identity() -> None:
    from odmr_bench.emulator.observations import EstimatorObservation
    from odmr_bench.estimators import TwoPointPartialPair

    pending_estimate = make_legal_estimate("first")
    target = pending_estimate.identities[0]
    shifted_query = make_legal_query(
        interrogation_center_hz=target.center_hz + 1.0e6,
    )
    with pytest.raises(ValueError, match="center"):
        replace(pending_estimate, pending_query=shifted_query)

    partial_estimate = make_legal_estimate("partial")
    shifted_observation = EstimatorObservation(
        shifted_query.expected_sequence_index,
        shifted_query.expected_end_timestamp_s,
        shifted_query.frequency_hz,
        0.98,
        shifted_query.integration_time_s,
        shifted_query.expected_nominal_exposure_photons,
        12_250,
    )
    shifted_partial = TwoPointPartialPair(
        shifted_query.pair_index,
        shifted_query.identity_pair_index,
        shifted_query.resonance_id,
        shifted_query.interrogation_center_hz,
        shifted_query.side,
        shifted_query,
        shifted_observation,
    )
    with pytest.raises(ValueError, match="center"):
        replace(partial_estimate, incomplete_pair=shifted_partial)


def test_estimator_record_edges_reject_capability_bearing_subclasses() -> None:
    from odmr_bench.estimators import (
        PublicAcquisitionResources,
        TwoPointBudgetCeiling,
        TwoPointEstimate,
        TwoPointIdentityEstimate,
        TwoPointPairResult,
        TwoPointPartialPair,
        TwoPointQuery,
    )

    def callback() -> None:
        return None

    def values(record: object) -> dict[str, object]:
        return {field.name: getattr(record, field.name) for field in fields(record)}

    class QueryWithDict(TwoPointQuery):
        pass

    class QueryWithSlot(TwoPointQuery):
        __slots__ = ("callback",)

    query = make_legal_query()
    query_with_dict = QueryWithDict(**values(query))
    object.__setattr__(query_with_dict, "callback", callback)
    query_with_slot = QueryWithSlot(**values(query))
    object.__setattr__(query_with_slot, "callback", callback)
    assert query_with_dict.__dict__["callback"] is callback
    assert query_with_slot.callback is callback
    partial = make_legal_partial_pair()
    for injected_query in (query_with_dict, query_with_slot):
        with pytest.raises(TypeError, match="exact"):
            replace(partial, first_query=injected_query)

    class PartialWithCallback(TwoPointPartialPair):
        pass

    partial_with_callback = PartialWithCallback(**values(partial))
    object.__setattr__(partial_with_callback, "callback", callback)
    with pytest.raises(TypeError, match="exact"):
        replace(
            make_legal_estimate("partial"),
            incomplete_pair=partial_with_callback,
        )

    class PairWithCallback(TwoPointPairResult):
        pass

    pair = make_legal_pair_result()
    pair_with_callback = PairWithCallback(**values(pair))
    object.__setattr__(pair_with_callback, "callback", callback)
    pair_identity = make_legal_estimate("complete").identities[0]
    with pytest.raises(TypeError, match="exact"):
        replace(pair_identity, latest_pair=pair_with_callback)

    class IdentityWithCallback(TwoPointIdentityEstimate):
        pass

    boundary = make_legal_estimate()
    identity_with_callback = IdentityWithCallback(**values(boundary.identities[0]))
    object.__setattr__(identity_with_callback, "callback", callback)
    with pytest.raises((TypeError, ValueError), match="exact"):
        replace(
            boundary,
            identities=(identity_with_callback, *boundary.identities[1:]),
        )

    class ResourcesWithCallback(PublicAcquisitionResources):
        pass

    resources_with_callback = ResourcesWithCallback(
        **values(boundary.tracking_resources)
    )
    object.__setattr__(resources_with_callback, "callback", callback)
    with pytest.raises(TypeError, match="exact"):
        replace(
            boundary,
            tracking_resources=resources_with_callback,
            charged_resources=resources_with_callback,
        )

    class CeilingWithCallback(TwoPointBudgetCeiling):
        pass

    ceiling_with_callback = CeilingWithCallback(**values(boundary.budget_ceiling))
    object.__setattr__(ceiling_with_callback, "callback", callback)
    with pytest.raises(TypeError, match="exact"):
        replace(boundary, budget_ceiling=ceiling_with_callback)

    class EstimateWithCallback(TwoPointEstimate):
        pass

    first_update = make_legal_update("first")
    estimate_with_callback = EstimateWithCallback(**values(first_update.estimate))
    object.__setattr__(estimate_with_callback, "callback", callback)
    with pytest.raises(TypeError, match="exact"):
        replace(first_update, estimate=estimate_with_callback)


@pytest.mark.parametrize(
    ("case", "text"),
    (
        ("query_resonance_id", "r0"),
        ("query_side", "minus"),
        ("partial_resonance_id", "r0"),
        ("partial_first_side", "minus"),
        ("pair_resonance_id", "r0"),
        ("pair_first_side", "minus"),
        ("pair_lock_state", "tracking"),
        ("pair_failure_code", "capture_exceeded"),
        ("identity_resonance_id", "r0"),
        ("identity_active_source_kind", "calibration"),
        ("identity_lock_state", "calibrated"),
        ("identity_failure_code", "capture_exceeded"),
        ("estimate_calibration_source_id", "caller-asserted-calibration"),
        ("estimate_calibration_source_provenance", "caller_asserted"),
        (
            "estimate_calibration_budget_treatment",
            "conditional_free_precalibration",
        ),
        ("estimate_stopped_reason", "budget_exhausted"),
    ),
)
def test_task3_strings_discard_subclass_capabilities(case: str, text: str) -> None:
    class StringWithCallback(str):
        pass

    injected = StringWithCallback(text)
    injected.callback = lambda: None
    assert callable(injected.callback)

    if case == "query_resonance_id":
        record = replace(make_legal_query(), resonance_id=injected)
        field_name = "resonance_id"
    elif case == "query_side":
        record = replace(make_legal_query(), side=injected)
        field_name = "side"
    elif case == "partial_resonance_id":
        record = replace(make_legal_partial_pair(), resonance_id=injected)
        field_name = "resonance_id"
    elif case == "partial_first_side":
        record = replace(make_legal_partial_pair(), first_side=injected)
        field_name = "first_side"
    elif case == "pair_resonance_id":
        record = replace(make_legal_pair_result(), resonance_id=injected)
        field_name = "resonance_id"
    elif case == "pair_first_side":
        record = replace(make_legal_pair_result(), first_side=injected)
        field_name = "first_side"
    elif case == "pair_lock_state":
        record = replace(make_legal_pair_result(), lock_state=injected)
        field_name = "lock_state"
    elif case == "pair_failure_code":
        record = replace(
            make_legal_pair_result(
                lock_state="lost", failure_code="capture_exceeded"
            ),
            failure_code=injected,
        )
        field_name = "failure_code"
    elif case == "identity_resonance_id":
        identity = make_legal_identity_estimates(current_timestamp_s=0.0)[0]
        record = replace(identity, resonance_id=injected)
        field_name = "resonance_id"
    elif case == "identity_active_source_kind":
        identity = make_legal_identity_estimates(current_timestamp_s=0.0)[0]
        record = replace(identity, active_source_kind=injected)
        field_name = "active_source_kind"
    elif case == "identity_lock_state":
        identity = make_legal_identity_estimates(current_timestamp_s=0.0)[0]
        record = replace(identity, lock_state=injected)
        field_name = "lock_state"
    elif case == "identity_failure_code":
        identity = make_legal_identity_estimates(current_timestamp_s=0.0)[0]
        lost_pair = make_legal_pair_result(
            lock_state="lost", failure_code="capture_exceeded"
        )
        record = replace(
            identity,
            completed_pairs=1,
            lock_state="lost",
            failure_code=injected,
            latest_pair=lost_pair,
        )
        field_name = "failure_code"
    elif case == "estimate_calibration_source_id":
        source = replace(make_legal_caller_asserted_source(), source_id=injected)
        record = replace(
            make_legal_estimate(), calibration_source_id=source.source_id
        )
        field_name = "calibration_source_id"
    elif case == "estimate_calibration_source_provenance":
        source = replace(make_legal_caller_asserted_source(), provenance=injected)
        record = replace(
            make_legal_estimate(),
            calibration_source_provenance=source.provenance,
        )
        field_name = "calibration_source_provenance"
    elif case == "estimate_calibration_budget_treatment":
        record = replace(
            make_legal_estimate(), calibration_budget_treatment=injected
        )
        field_name = "calibration_budget_treatment"
    else:
        record = replace(make_legal_estimate("stopped"), stopped_reason=injected)
        field_name = "stopped_reason"

    stored = getattr(record, field_name)
    assert stored == text
    assert type(stored) is str
    assert not hasattr(stored, "callback")


def test_estimator_record_graph_contains_no_truth_or_full_resource_type() -> None:
    from odmr_bench.dynamics.base import SpectralDynamics, SpectralSnapshot
    from odmr_bench.emulator.instrument import ODMRInstrument
    from odmr_bench.emulator.observations import InstrumentObservation
    from odmr_bench.emulator.resources import ResourceSnapshot

    roots = (
        make_legal_query(),
        make_legal_partial_pair(),
        make_legal_pair_result(),
        *make_legal_identity_estimates(
            current_timestamp_s=0.012,
            latest_pair=make_legal_pair_result(),
        ),
        make_legal_estimate("complete"),
        make_legal_update("second"),
    )
    forbidden_types = {
        ODMRInstrument,
        SpectralSnapshot,
        InstrumentObservation,
        ResourceSnapshot,
        Future,
    }
    forbidden_type_names = {
        "ODMRInstrument",
        "SpectralDynamics",
        "SpectralSnapshot",
        "InstrumentObservation",
        "ResourceSnapshot",
        "Future",
    }
    seen: set[int] = set()

    def inspect(value: object) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        assert type(value) not in forbidden_types
        assert not callable(value)
        if isinstance(value, str):
            assert type(value) is str
            return
        if is_dataclass(value) and not isinstance(value, type):
            dataclass_field_names = {
                dataclass_field.name for dataclass_field in fields(value)
            }
            for dataclass_field in fields(value):
                assert dataclass_field.name != "expected_photons"
                assert not any(
                    token in dataclass_field.name
                    for token in ("truth", "callback", "evaluator", "future")
                )
                annotation = str(dataclass_field.type)
                assert not any(name in annotation for name in forbidden_type_names)
                inspect(getattr(value, dataclass_field.name))
            if hasattr(value, "__dict__"):
                for name, item in vars(value).items():
                    if name not in dataclass_field_names:
                        assert not any(
                            token in name
                            for token in ("truth", "callback", "evaluator", "future")
                        )
                        inspect(item)
            for record_type in type(value).__mro__:
                slots = record_type.__dict__.get("__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                for name in slots:
                    if (
                        name not in dataclass_field_names
                        and name not in {"__dict__", "__weakref__"}
                        and hasattr(value, name)
                    ):
                        assert not any(
                            token in name
                            for token in ("truth", "callback", "evaluator", "future")
                        )
                        inspect(getattr(value, name))
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                inspect(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                inspect(key)
                inspect(item)

    for root in roots:
        inspect(root)

    base_query = make_legal_query()

    class QueryWithDict(type(base_query)):
        pass

    class QueryWithSlot(type(base_query)):
        __slots__ = ("callback",)

    for leaky_type in (QueryWithDict, QueryWithSlot):
        leaky_query = leaky_type(
            **{
                field.name: getattr(base_query, field.name)
                for field in fields(base_query)
            }
        )
        object.__setattr__(leaky_query, "callback", lambda: None)
        with pytest.raises(AssertionError):
            inspect(leaky_query)

    class StringWithCallback(str):
        pass

    leaky_string = StringWithCallback("minus")
    leaky_string.callback = lambda: None
    with pytest.raises(AssertionError):
        inspect(leaky_string)

    assert SpectralDynamics.__name__ in forbidden_type_names


def test_caller_asserted_source_snapshots_values_and_rejects_verified_direct_construction(  # noqa: E501
) -> None:
    mutable_observation_list = list(make_legal_source_observations())
    mutable_resonance_ids = [f"r{index}" for index in range(8)]
    fit_configuration = make_legal_fit_configuration(mutable_resonance_ids)
    source_fit = make_legal_source_fit(fit_configuration)
    source = make_legal_caller_asserted_source(
        source_observations=mutable_observation_list,
        source_fit=source_fit,
        fit_configuration=fit_configuration,
    )

    assert source.provenance == "caller_asserted"
    assert source.source_observations is not mutable_observation_list
    assert source.source_observations == tuple(mutable_observation_list)
    assert source.source_observations[0] is not mutable_observation_list[0]
    assert source.source_fit is not source_fit
    assert source.fit_configuration is not fit_configuration
    with pytest.raises(ValueError, match="verified"):
        replace(source, provenance="verified_factory_acquisition")

    original_frequency_hz = source.source_observations[0].frequency_hz
    original_q_value = source.source_fit.q_values[0]
    mutable_observation_list.clear()
    mutable_resonance_ids[-1] = "mutated"
    object.__setattr__(fit_configuration, "resonance_ids", ("mutated",) * 8)
    source_fit.q_values.setflags(write=True)
    source_fit.q_values[0] = 0.0
    assert source.source_observations[0].frequency_hz == original_frequency_hz
    assert source.fit_configuration.resonance_ids[-1] == "r7"
    assert source.source_fit.q_values[0] == original_q_value


def test_caller_asserted_source_deeply_snapshots_nested_fit_values() -> None:
    from odmr_bench.estimators import FitUncertainty

    uncertainty = FitUncertainty(
        baseline_standard_errors=np.array([0.01, 0.02]),
        center_hz=np.full(8, 10.0),
        fwhm_hz=np.full(8, 20.0),
        amplitude=np.full(8, 0.03),
        eta=np.full(8, 0.04),
    )
    source_fit = replace(
        make_legal_source_fit(),
        uncertainty=uncertainty,
        uncertainty_reason=None,
    )
    source = make_legal_caller_asserted_source(source_fit=source_fit)
    stored_fit = source.source_fit
    stored_diagnostics = stored_fit.diagnostics
    stored_uncertainty = stored_fit.uncertainty
    assert stored_diagnostics is not source_fit.diagnostics
    assert stored_uncertainty is not uncertainty
    assert stored_uncertainty is not None
    expected_arrays = {
        name: np.array(getattr(stored_uncertainty, name), copy=True)
        for name in (
            "baseline_standard_errors",
            "center_hz",
            "fwhm_hz",
            "amplitude",
            "eta",
        )
    }
    expected_q_values = np.array(stored_fit.q_values, copy=True)

    object.__setattr__(source_fit.diagnostics, "messages", ("mutated",))
    object.__setattr__(source_fit.diagnostics, "source", "none")
    object.__setattr__(source_fit, "resonance_estimates", ())
    object.__setattr__(source_fit, "baseline_estimate", None)
    object.__setattr__(source_fit, "initial_guess", None)
    object.__setattr__(source_fit, "nfev", 0)
    source_fit.q_values.setflags(write=True)
    source_fit.q_values[0] = 0.0
    object.__setattr__(uncertainty, "method", "mutated")
    for name in expected_arrays:
        original = getattr(uncertainty, name)
        assert original is not None
        original.setflags(write=True)
        original[...] = 0.0

    assert stored_diagnostics.source == "user"
    assert stored_diagnostics.messages == ()
    assert len(stored_fit.resonance_estimates) == 8
    assert stored_fit.baseline_estimate is not None
    assert stored_fit.initial_guess is not None
    assert stored_fit.nfev == 1
    assert np.array_equal(stored_fit.q_values, expected_q_values)
    assert not stored_fit.q_values.flags.writeable
    assert stored_uncertainty.method == "local_linearized_jacobian_covariance"
    for name, expected in expected_arrays.items():
        stored = getattr(stored_uncertainty, name)
        assert stored is not None
        assert np.array_equal(stored, expected)
        assert not stored.flags.writeable


def test_calibration_records_preserve_source_identity_and_snapshot_configuration(
) -> None:
    from odmr_bench.estimators import TwoPointCalibration

    source = make_legal_caller_asserted_source()
    mutable_configuration = make_legal_tracker_configuration()
    identities = make_legal_identity_calibrations(source)
    treatment = "conditional_free_precalibration"
    calibration = TwoPointCalibration(
        source, mutable_configuration, treatment, identities
    )

    assert calibration.source is source
    assert calibration.configuration == mutable_configuration
    assert calibration.configuration is not mutable_configuration
    assert calibration.identities == tuple(identities)
    assert calibration.identities is not identities

    object.__setattr__(
        mutable_configuration.identity_binding,
        "expected_resonance_ids",
        ("mutated",) * 8,
    )
    assert calibration.configuration.identity_binding.expected_resonance_ids == tuple(
        f"r{index}" for index in range(8)
    )
    with pytest.raises(ValueError, match="conditional_free"):
        TwoPointCalibration(
            source, make_legal_tracker_configuration(), "included_same_run", identities
        )
    with pytest.raises(ValueError, match="unique"):
        TwoPointCalibration(
            source,
            make_legal_tracker_configuration(),
            treatment,
            (*identities[:-1], replace(identities[-1], resonance_id="r0")),
        )
    with pytest.raises(ValueError, match="positive"):
        replace(identities[0], calibration_fwhm_hz=0.0)
    with pytest.raises(ValueError, match="nonempty"):
        replace(
            identities[0],
            allowed_center_min_hz=identities[0].allowed_center_max_hz + 1.0,
        )


def test_public_resources_and_budget_validate_intrinsic_domains() -> None:
    from odmr_bench.estimators import (
        PublicAcquisitionResources,
        TwoPointBudgetCeiling,
    )

    resources = PublicAcquisitionResources(1, 0.005, 2.5e6, 0, 1, 0.006)
    assert resources.observations == 1
    assert type(resources.integration_time_s) is float
    numpy_resources = PublicAcquisitionResources(
        np.int64(1),
        np.float64(0.005),
        np.float64(2.5e6),
        np.int64(0),
        np.int64(1),
        np.float64(0.006),
    )
    assert type(numpy_resources.observations) is int
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(resources, observations_without_realized_counts=2)
    with pytest.raises(ValueError, match="include integration"):
        replace(resources, virtual_elapsed_time_s=0.004)
    with pytest.raises(ValueError, match="at least one"):
        TwoPointBudgetCeiling(None, None, None, None)
    with pytest.raises(TypeError):
        replace(resources, observations=True)


@pytest.mark.parametrize("capture_fwhm_fraction", (0.35, 0.36))
def test_tracker_configuration_requires_capture_fraction_below_offset(
    capture_fwhm_fraction: float,
) -> None:
    from odmr_bench.estimators import TwoPointTrackerConfiguration

    with pytest.raises(ValueError, match="strictly less"):
        TwoPointTrackerConfiguration(capture_fwhm_fraction=capture_fwhm_fraction)


def test_string_tuple_contracts_snapshot_lists_and_reject_non_sequences() -> None:
    from odmr_bench.estimators import (
        NormalizedFluorescenceProvenance,
        TwoPointIdentityBinding,
    )

    expected_ids = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]
    canonical_ids = tuple(expected_ids)
    sampling_rules = ["poisson"]
    binding = TwoPointIdentityBinding("require_expected_ids", expected_ids)
    provenance = NormalizedFluorescenceProvenance(
        "normalized_fluorescence", "instrument_v1", 2.5e6, sampling_rules
    )
    expected_ids[-1] = "mutated"
    sampling_rules.append("mutated")
    assert binding.expected_resonance_ids == canonical_ids
    assert provenance.sampling_rules == ("poisson",)

    for value in (
        set(canonical_ids),
        dict.fromkeys(canonical_ids),
        (resonance_id for resonance_id in canonical_ids),
    ):
        with pytest.raises(TypeError, match="ordered sequence"):
            TwoPointIdentityBinding("require_expected_ids", value)
    for value in (
        {"poisson"},
        {"poisson": None},
        (rule for rule in ("poisson",)),
    ):
        with pytest.raises(TypeError, match="ordered sequence"):
            NormalizedFluorescenceProvenance(
                "normalized_fluorescence", "instrument_v1", 2.5e6, value
            )


@pytest.mark.parametrize(
    ("error_type", "codes"),
    [
        (
            "calibration",
            (
                "invalid_argument_type",
                "invalid_argument_value",
                "invalid_provenance_or_quantity",
                "invalid_source_trace",
                "source_resource_mismatch",
                "fit_input_mismatch",
                "source_fit_failed",
                "source_identity_mismatch",
                "invalid_source_epoch",
                "invalid_availability_or_clock",
                "invalid_calibration_geometry",
                "invalid_budget_treatment",
            ),
        ),
        (
            "observation",
            (
                "invalid_observation_type",
                "no_pending_query",
                "sequence_mismatch",
                "frequency_mismatch",
                "integration_time_mismatch",
                "endpoint_mismatch",
                "nominal_exposure_mismatch",
                "invalid_observation_value",
            ),
        ),
        (
            "update",
            (
                "partial_pair_construction_failed",
                "pair_result_construction_failed",
                "identity_estimate_construction_failed",
                "resource_construction_failed",
                "aggregate_estimate_construction_failed",
            ),
        ),
    ],
)
def test_identity_clock_configuration_and_errors_are_closed(
    error_type: str, codes: tuple[str, ...]
) -> None:
    from odmr_bench.estimators import (
        NormalizedFluorescenceProvenance,
        TwoPointCalibrationConstructionError,
        TwoPointClockMapping,
        TwoPointIdentityBinding,
        TwoPointObservationValidationError,
        TwoPointRunMetadata,
        TwoPointTrackerConfiguration,
        TwoPointUpdateConstructionError,
    )

    expected_ids = ("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7")
    required = TwoPointIdentityBinding("require_expected_ids", expected_ids)
    assert required.expected_resonance_ids == expected_ids
    assert TwoPointIdentityBinding("adopt_fit_ids", None).expected_resonance_ids is None
    with pytest.raises(ValueError):
        TwoPointIdentityBinding("require_expected_ids", expected_ids[:-1])
    with pytest.raises(ValueError):
        TwoPointIdentityBinding("require_expected_ids", (*expected_ids[:-1], "r0"))
    with pytest.raises(ValueError):
        TwoPointIdentityBinding("require_expected_ids", (*expected_ids[:-1], " "))
    with pytest.raises(ValueError):
        TwoPointIdentityBinding("adopt_fit_ids", expected_ids)
    with pytest.raises(ValueError):
        TwoPointIdentityBinding("unknown", None)

    assert TwoPointClockMapping("shared_clock", "clock", "clock", 1.0, 0.0)
    assert TwoPointClockMapping("unit_scale_offset", "source", "tracker", 1.0, 2.0)
    with pytest.raises(ValueError):
        TwoPointClockMapping("shared_clock", "source", "tracker", 1.0, 0.0)
    with pytest.raises(ValueError):
        TwoPointClockMapping("shared_clock", "clock", "clock", 1.0, 0.1)
    with pytest.raises(ValueError):
        TwoPointClockMapping("unit_scale_offset", "clock", "clock", 1.0, 0.0)
    with pytest.raises(ValueError):
        TwoPointClockMapping("shared_clock", "clock", "clock", 1.01, 0.0)
    with pytest.raises(ValueError):
        TwoPointClockMapping("shared_clock", " ", " ", 1.0, 0.0)

    assert NormalizedFluorescenceProvenance(
        "normalized_fluorescence", "instrument_v1", 2.5e6, ("poisson",)
    )
    with pytest.raises(ValueError):
        NormalizedFluorescenceProvenance("raw_counts", "instrument_v1", 2.5e6, ())
    with pytest.raises(ValueError):
        NormalizedFluorescenceProvenance(
            "normalized_fluorescence", "instrument_v1", 0.0, ()
        )

    configuration = TwoPointTrackerConfiguration()
    assert configuration.identity_binding == required
    assert configuration.offset_fwhm_fraction == 0.35
    assert configuration.capture_fwhm_fraction == 0.20
    assert configuration.proportional_gain == 1.0
    assert configuration.max_step_fwhm_fraction == 0.10
    assert configuration.integration_time_s == 0.005
    assert configuration.common_mode_limit_target_depths is None
    with pytest.raises(ValueError):
        TwoPointTrackerConfiguration(integration_time_s=0.0)
    assert TwoPointRunMetadata(
        "clock", None, 0.0, 2.5e6, 0.001, "normalized_fluorescence"
    )
    with pytest.raises(ValueError):
        TwoPointRunMetadata("clock", None, 0.0, 2.5e6, 0.001, "raw_counts")

    error_class = {
        "calibration": TwoPointCalibrationConstructionError,
        "observation": TwoPointObservationValidationError,
        "update": TwoPointUpdateConstructionError,
    }[error_type]
    for code in codes:
        error = error_class(code, "details")
        assert error.code == code
        assert error.message == "details"
    with pytest.raises(ValueError):
        error_class("unknown", "details")
    with pytest.raises(ValueError):
        error_class(codes[0], "")
