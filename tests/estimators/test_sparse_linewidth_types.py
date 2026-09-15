"""Contract tests for sparse-linewidth primitive estimator values."""

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from typing import get_args

import numpy as np
import pytest

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators import (
    CompositeIdentityEstimate,
    CompositeMode,
    CompositeStopReason,
    SparseGeometryFailureCode,
    SparseGeometryUnavailableDiagnostic,
    SparseLinewidthCompositeEstimate,
    SparseLinewidthCompositeUpdate,
    SparseLinewidthConfiguration,
    SparseLinewidthFailureCode,
    SparseLinewidthObservationValidationError,
    SparseLinewidthQuery,
    SparseLinewidthResetError,
    SparseLinewidthScanResult,
    SparseLinewidthSourceKind,
    SparseLinewidthUpdateConstructionError,
    SparseObservationValidationCode,
    SparsePartialScan,
    SparseResetFailureCode,
    SparseUpdateConstructionCode,
)
from odmr_bench.estimators.two_point_types import PublicAcquisitionResources
from tests.sparse_linewidth_helpers import (
    make_composite_estimate,
    make_composite_identity,
    make_partial_scan,
    make_scan_result,
    make_sparse_query,
)
from tests.two_point_helpers import (
    make_legal_pair_result,
    make_legal_partial_pair,
    make_legal_query,
)


def _ordered_mean(values: tuple[float, ...]) -> float:
    mean = values[0]
    for count, value in enumerate(values[1:], start=2):
        mean = mean + (value - mean) / count
    return mean


def _scan_with_midpoints(
    midpoints: tuple[float, float, float, float, float],
) -> SparseLinewidthScanResult:
    offsets = (0.5, -1.0, 0.0, 1.0, -0.5)
    queries = tuple(
        make_sparse_query(
            acquisition_index=index,
            point_index=index,
            offset_multiplier=offsets[index],
            frequency_hz=2.87e9 + offsets[index] * 1.0e6,
            integration_time_s=2.0,
            expected_sequence_index=index,
            expected_end_timestamp_s=midpoint + 1.0,
        )
        for index, midpoint in enumerate(midpoints)
    )
    observations = tuple(
        EstimatorObservation(
            query.expected_sequence_index,
            query.expected_end_timestamp_s,
            query.frequency_hz,
            1.0,
            query.integration_time_s,
            query.expected_nominal_exposure_photons,
        )
        for query in queries
    )
    return make_scan_result(
        queries=queries,
        observations=observations,
        public_reference_timestamp_s=_ordered_mean(midpoints),
        release_sequence_index=queries[-1].expected_sequence_index,
        release_timestamp_s=queries[-1].expected_end_timestamp_s,
    )


def _one_observation_resources() -> PublicAcquisitionResources:
    return PublicAcquisitionResources(1, 0.005, 1.0, 0, 1, 0.005)


def _completed_fast_estimate() -> tuple[object, object]:
    pair = make_legal_pair_result()
    current_timestamp_s = pair.release_timestamp_s
    first = make_composite_identity(
        resonance_id="r0",
        fast_center_hz=pair.candidate_center_hz,
        fast_center_source_kind="pair",
        fast_center_source_pair_index=pair.pair_index,
        fast_center_reference_timestamp_s=pair.pair_reference_timestamp_s,
        fast_center_release_sequence_index=pair.release_sequence_index,
        fast_center_release_timestamp_s=pair.release_timestamp_s,
        active_fwhm_hz=1.0e6,
        live_q=pair.candidate_center_hz / 1.0e6,
        center_age_s=current_timestamp_s - pair.pair_reference_timestamp_s,
        fwhm_age_s=current_timestamp_s,
        center_release_age_s=0.0,
        fwhm_release_age_s=current_timestamp_s,
        completed_fast_pairs=1,
        latest_fast_pair=pair,
    )
    identities = (
        first,
        *(
            make_composite_identity(
                resonance_id=f"r{index}",
                center_age_s=current_timestamp_s,
                fwhm_age_s=current_timestamp_s,
                center_release_age_s=current_timestamp_s,
                fwhm_release_age_s=current_timestamp_s,
            )
            for index in range(1, 8)
        ),
    )
    resources = PublicAcquisitionResources(2, 0.01, 25_000.0, 24_625, 0, 0.01)
    estimate = make_composite_estimate(
        identities=identities,
        fast_pair_history=(pair,),
        accepted_observations=2,
        completed_fast_pairs=1,
        fast_pairs_since_scan=1,
        current_sequence_index=pair.release_sequence_index,
        current_timestamp_s=current_timestamp_s,
        fast_tracking_resources=resources,
        tracking_resources=resources,
        charged_resources=resources,
    )
    return pair, estimate


def test_scan_public_reference_uses_exact_ordered_mean_not_sum() -> None:
    midpoints = (
        float.fromhex("0x1.71ac192603038p+27"),
        float.fromhex("0x1.7b1f2de93f4dcp+28"),
        float.fromhex("0x1.604ff56fc7d1dp+29"),
        float.fromhex("0x1.807d78016510cp+29"),
        float.fromhex("0x1.d3df4c4d16825p+29"),
    )
    expected = _ordered_mean(midpoints)
    regrouped = sum(midpoints) / 5.0
    assert expected != regrouped
    assert _scan_with_midpoints(midpoints).public_reference_timestamp_s == expected


def test_optimizer_failure_accepts_negative_scipy_status() -> None:
    result = make_scan_result(
        status="failure",
        failure_code="optimizer_failed",
        fitted_center_correction_hz=None,
        fitted_local_center_hz=None,
        fitted_fwhm_hz=None,
        fitted_amplitude=None,
        fitted_baseline_offset=None,
        fitted_q=None,
        rmse=None,
        amplitude_normalized_rmse=None,
        scaled_jacobian_rank=None,
        scaled_jacobian_condition=None,
        scipy_status=-1,
    )
    assert result.scipy_status == -1


def test_composite_replays_resources_and_ties_them_to_accepted_count() -> None:
    one = _one_observation_resources()
    with pytest.raises(ValueError):
        make_composite_estimate(
            fast_tracking_resources=one,
            tracking_resources=one,
            charged_resources=one,
        )


def test_composite_rejects_pending_sparse_query_that_repeats_partial_point() -> None:
    partial = make_partial_scan()
    one = _one_observation_resources()
    with pytest.raises(ValueError):
        make_composite_estimate(
            pending_mode="sparse_scan",
            pending_query=partial.queries[0],
            incomplete_sparse_scan=partial,
            accepted_observations=1,
            sparse_tracking_resources=one,
            tracking_resources=one,
            charged_resources=one,
        )


def test_sparse_scan_requires_adjacent_acquisition_indices() -> None:
    partial = make_partial_scan()
    second = make_sparse_query(
        acquisition_index=2,
        point_index=1,
        offset_multiplier=-1.0,
        frequency_hz=2.869e9,
        expected_sequence_index=1,
        expected_end_timestamp_s=0.010,
    )
    observation = EstimatorObservation(1, 0.010, 2.869e9, 1.0, 0.005, 1.0)
    with pytest.raises(ValueError):
        make_partial_scan(
            queries=(partial.queries[0], second),
            observations=(partial.observations[0], observation),
        )


def test_empty_trace_retains_available_calibration_boundary_and_zero_cpu() -> None:
    identities = tuple(
        make_composite_identity(
            resonance_id=f"r{index}",
            center_age_s=0.1,
            fwhm_age_s=0.1,
            center_release_age_s=0.1,
            fwhm_release_age_s=0.1,
        )
        for index in range(8)
    )
    estimate = make_composite_estimate(
        calibration_budget_treatment="included_same_run",
        identities=identities,
        current_sequence_index=7,
        current_timestamp_s=0.1,
    )
    assert estimate.current_sequence_index == 7
    with pytest.raises(ValueError, match="zero"):
        make_composite_estimate(
            fast_update_cpu_time_s=0.001, total_update_cpu_time_s=0.001
        )


def test_nonzero_frequency_overhead_is_not_replayed_from_observations() -> None:
    pair, estimate = _completed_fast_estimate()
    overhead = PublicAcquisitionResources(2, 0.01, 25_000.0, 24_625, 0, 0.25)
    accepted = make_composite_estimate(
        identities=estimate.identities,
        fast_pair_history=(pair,),
        accepted_observations=2,
        completed_fast_pairs=1,
        fast_pairs_since_scan=1,
        current_sequence_index=pair.release_sequence_index,
        current_timestamp_s=pair.release_timestamp_s,
        fast_tracking_resources=overhead,
        tracking_resources=overhead,
        charged_resources=overhead,
    )
    assert accepted.tracking_resources.virtual_elapsed_time_s == 0.25


def test_pending_sparse_query_must_extend_partial_acquisition_sequence() -> None:
    with pytest.raises(ValueError, match="pending sparse"):
        _sparse_partial_at_due_cadence(pending_acquisition_index=5)


def _one_fast_partial_estimate(*, resonance_id: str, pending: bool) -> object:
    partial = make_legal_partial_pair()
    first_query = replace(partial.first_query, resonance_id=resonance_id)
    first_observation = replace(
        partial.first_observation, frequency_hz=first_query.frequency_hz
    )
    partial = replace(
        partial,
        resonance_id=resonance_id,
        first_query=first_query,
        first_observation=first_observation,
    )
    identities = tuple(
        make_composite_identity(
            resonance_id=f"r{index}",
            fast_center_hz=2.76e9 if index == 0 else 2.87e9,
            live_q=2760.0 if index == 0 else 2870.0,
            center_age_s=0.006,
            fwhm_age_s=0.006,
            center_release_age_s=0.006,
            fwhm_release_age_s=0.006,
        )
        for index in range(8)
    )
    resource = PublicAcquisitionResources(1, 0.005, 12_500.0, 12_250, 0, 0.005)
    pending_query = (
        None
        if not pending
        else make_legal_query(
            side="plus",
            query_index=1,
            resonance_id=resonance_id,
            interrogation_center_hz=2.76e9,
            expected_sequence_index=1,
            expected_end_timestamp_s=0.012,
        )
    )
    return make_composite_estimate(
        identities=identities,
        pending_mode=None if pending_query is None else "fast_pair",
        pending_query=pending_query,
        incomplete_fast_pair=partial,
        accepted_observations=1,
        current_sequence_index=0,
        current_timestamp_s=0.006,
        fast_tracking_resources=resource,
        tracking_resources=resource,
        charged_resources=resource,
    )


def _one_sparse_partial_estimate(
    *, resonance_id: str, pending: bool, frozen_prior_fwhm_hz: float = 1.0e6
) -> object:
    query = make_sparse_query(
        resonance_id=resonance_id, frozen_prior_fwhm_hz=frozen_prior_fwhm_hz
    )
    observation = EstimatorObservation(0, 0.005, query.frequency_hz, 1.0, 0.005, 1.0)
    partial = make_partial_scan(
        resonance_id=resonance_id,
        frozen_prior_fwhm_hz=frozen_prior_fwhm_hz,
        queries=(query,),
        observations=(observation,),
    )
    pending_query = (
        None
        if not pending
        else make_sparse_query(
            acquisition_index=1,
            point_index=1,
            resonance_id=resonance_id,
            frozen_prior_fwhm_hz=frozen_prior_fwhm_hz,
            offset_multiplier=-1.0,
            frequency_hz=2.869e9,
            expected_sequence_index=1,
            expected_end_timestamp_s=0.010,
        )
    )
    resource = _one_observation_resources()
    identities = tuple(
        make_composite_identity(
            resonance_id=f"r{index}",
            center_age_s=0.005,
            fwhm_age_s=0.005,
            center_release_age_s=0.005,
            fwhm_release_age_s=0.005,
        )
        for index in range(8)
    )
    return make_composite_estimate(
        identities=identities,
        pending_mode=None if pending_query is None else "sparse_scan",
        pending_query=pending_query,
        incomplete_sparse_scan=partial,
        accepted_observations=1,
        current_sequence_index=0,
        current_timestamp_s=0.005,
        sparse_tracking_resources=resource,
        tracking_resources=resource,
        charged_resources=resource,
    )


def _fast_partial_at_due_cadence() -> object:
    pair, estimate = _completed_fast_estimate()
    first_query = make_legal_query(
        query_index=2,
        pair_index=1,
        identity_pair_index=0,
        resonance_id="r1",
        interrogation_center_hz=2.87e9,
        expected_sequence_index=2,
        expected_end_timestamp_s=0.018,
    )
    partial = replace(
        make_legal_partial_pair(),
        pair_index=1,
        identity_pair_index=0,
        resonance_id="r1",
        interrogation_center_hz=2.87e9,
        first_query=first_query,
        first_observation=EstimatorObservation(
            2, 0.018, first_query.frequency_hz, 0.98, 0.005, 12_500.0, 12_250
        ),
    )
    current_timestamp_s = 0.018
    identities = tuple(
        replace(
            identity,
            center_age_s=current_timestamp_s
            - identity.fast_center_reference_timestamp_s,
            fwhm_age_s=current_timestamp_s - identity.fwhm_reference_timestamp_s,
            center_release_age_s=current_timestamp_s
            - identity.fast_center_release_timestamp_s,
            fwhm_release_age_s=current_timestamp_s
            - identity.fwhm_release_timestamp_s,
        )
        for identity in estimate.identities
    )
    resources = PublicAcquisitionResources(3, 0.015, 37_500.0, 36_875, 0, 0.015)
    return make_composite_estimate(
        configuration=SparseLinewidthConfiguration(scan_period_fast_pairs=1),
        identities=identities,
        incomplete_fast_pair=partial,
        fast_pair_history=(pair,),
        accepted_observations=3,
        completed_fast_pairs=1,
        fast_pairs_since_scan=1,
        current_sequence_index=2,
        current_timestamp_s=current_timestamp_s,
        fast_tracking_resources=resources,
        tracking_resources=resources,
        charged_resources=resources,
    )


def _sparse_partial_at_due_cadence(*, pending_acquisition_index: int) -> object:
    pair, estimate = _completed_fast_estimate()
    identity = estimate.identities[0]
    first_query = make_sparse_query(
        acquisition_index=2,
        frozen_fast_center_hz=identity.fast_center_hz,
        frozen_fast_center_source_kind="pair",
        frozen_fast_center_source_pair_index=0,
        frozen_fast_center_reference_timestamp_s=pair.pair_reference_timestamp_s,
        frozen_fast_center_release_sequence_index=pair.release_sequence_index,
        frozen_fast_center_release_timestamp_s=pair.release_timestamp_s,
        frequency_hz=identity.fast_center_hz + 0.5e6,
        expected_sequence_index=2,
        expected_end_timestamp_s=0.017,
    )
    observation = EstimatorObservation(
        2, 0.017, first_query.frequency_hz, 1.0, 0.005, 1.0
    )
    partial = make_partial_scan(
        frozen_fast_center_hz=identity.fast_center_hz,
        frozen_fast_center_source_kind="pair",
        frozen_fast_center_source_pair_index=0,
        frozen_fast_center_reference_timestamp_s=pair.pair_reference_timestamp_s,
        frozen_fast_center_release_sequence_index=pair.release_sequence_index,
        frozen_fast_center_release_timestamp_s=pair.release_timestamp_s,
        queries=(first_query,),
        observations=(observation,),
    )
    current_timestamp_s = 0.017
    identities = tuple(
        replace(
            item,
            center_age_s=current_timestamp_s - item.fast_center_reference_timestamp_s,
            fwhm_age_s=current_timestamp_s - item.fwhm_reference_timestamp_s,
            center_release_age_s=current_timestamp_s
            - item.fast_center_release_timestamp_s,
            fwhm_release_age_s=current_timestamp_s - item.fwhm_release_timestamp_s,
        )
        for item in estimate.identities
    )
    fast_resources = PublicAcquisitionResources(2, 0.01, 25_000.0, 24_625, 0, 0.01)
    sparse_resources = _one_observation_resources()
    tracking_resources = PublicAcquisitionResources(
        3, 0.015, 25_001.0, 24_625, 1, 0.015
    )
    pending = make_sparse_query(
        acquisition_index=pending_acquisition_index,
        point_index=1,
        offset_multiplier=-1.0,
        frozen_fast_center_hz=identity.fast_center_hz,
        frozen_fast_center_source_kind="pair",
        frozen_fast_center_source_pair_index=0,
        frozen_fast_center_reference_timestamp_s=pair.pair_reference_timestamp_s,
        frozen_fast_center_release_sequence_index=pair.release_sequence_index,
        frozen_fast_center_release_timestamp_s=pair.release_timestamp_s,
        frequency_hz=identity.fast_center_hz - 1.0e6,
        expected_sequence_index=3,
        expected_end_timestamp_s=0.022,
    )
    return make_composite_estimate(
        configuration=SparseLinewidthConfiguration(scan_period_fast_pairs=1),
        identities=identities,
        pending_mode="sparse_scan",
        pending_query=pending,
        incomplete_sparse_scan=partial,
        fast_pair_history=(pair,),
        accepted_observations=3,
        completed_fast_pairs=1,
        fast_pairs_since_scan=1,
        current_sequence_index=2,
        current_timestamp_s=current_timestamp_s,
        fast_tracking_resources=fast_resources,
        sparse_tracking_resources=sparse_resources,
        tracking_resources=tracking_resources,
        charged_resources=tracking_resources,
    )


def test_incomplete_fast_pair_cannot_start_at_due_sparse_cadence() -> None:
    with pytest.raises(ValueError, match="cadence"):
        _fast_partial_at_due_cadence()


def test_incomplete_sparse_scan_cannot_start_before_due_cadence() -> None:
    with pytest.raises(ValueError, match="cadence"):
        _one_sparse_partial_estimate(resonance_id="r0", pending=False)


@pytest.mark.parametrize("pending", (False, True))
def test_incomplete_fast_pair_must_match_its_scheduled_identity(pending: bool) -> None:
    with pytest.raises(ValueError):
        _one_fast_partial_estimate(resonance_id="r1", pending=pending)


@pytest.mark.parametrize("pending", (False, True))
def test_incomplete_sparse_scan_must_match_its_scheduled_identity(
    pending: bool,
) -> None:
    with pytest.raises(ValueError):
        _one_sparse_partial_estimate(resonance_id="r1", pending=pending)


@pytest.mark.parametrize("pending", (False, True))
def test_incomplete_sparse_scan_must_match_its_frozen_source(pending: bool) -> None:
    with pytest.raises(ValueError):
        _one_sparse_partial_estimate(
            resonance_id="r0", pending=pending, frozen_prior_fwhm_hz=2.0e6
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"fast_pairs_since_scan": 1},
        {"fast_update_cpu_time_s": 1.0, "total_update_cpu_time_s": 0.5},
        {"sparse_update_cpu_time_s": 1.0, "total_update_cpu_time_s": 0.5},
    ),
)
def test_composite_rejects_unprovable_endpoint_schedule_and_cpu_state(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        make_composite_estimate(**overrides)


def test_identity_retains_last_successful_fast_source_after_later_failure() -> None:
    successful = make_legal_pair_result()
    failed = make_legal_pair_result(
        pair_index=8,
        identity_pair_index=1,
        resonance_id="r0",
        lock_state="lost",
        failure_code="invalid_pair_normalization",
    )
    identity = make_composite_identity(
        fast_center_hz=successful.candidate_center_hz,
        fast_center_source_kind="pair",
        fast_center_source_pair_index=successful.pair_index,
        fast_center_reference_timestamp_s=successful.pair_reference_timestamp_s,
        fast_center_release_sequence_index=successful.release_sequence_index,
        fast_center_release_timestamp_s=successful.release_timestamp_s,
        center_age_s=1.0,
        center_release_age_s=1.0,
        live_q=successful.candidate_center_hz / 1.0e6,
        completed_fast_pairs=2,
        latest_fast_pair=failed,
    )
    assert identity.fast_center_source_pair_index == successful.pair_index


def test_completed_fast_update_rejects_mismatched_query_or_observation() -> None:
    pair, estimate = _completed_fast_estimate()
    with pytest.raises(ValueError):
        SparseLinewidthCompositeUpdate(
            pair.minus_query,
            pair.minus_observation,
            pair,
            None,
            estimate,
            0.0,
        )
    with pytest.raises(ValueError):
        SparseLinewidthCompositeUpdate(
            pair.plus_query,
            replace(pair.plus_observation, fluorescence=0.0),
            pair,
            None,
            estimate,
            0.0,
        )


def test_scan_q_preserves_repository_signed_convention() -> None:
    result = make_scan_result(
        status="success",
        fitted_local_center_hz=-1.0,
        fitted_fwhm_hz=2.0,
        fitted_q=-0.5,
    )
    assert result.fitted_q == -0.5
    assert type(result.fitted_q) is float


def test_sparse_record_surface_and_tuple_boundaries_are_exact() -> None:
    assert [field.name for field in fields(SparseLinewidthQuery)] == [
        "acquisition_index",
        "scan_index",
        "identity_scan_index",
        "point_index",
        "resonance_id",
        "offset_multiplier",
        "frozen_fast_center_hz",
        "frozen_fast_center_source_kind",
        "frozen_fast_center_source_pair_index",
        "frozen_fast_center_reference_timestamp_s",
        "frozen_fast_center_release_sequence_index",
        "frozen_fast_center_release_timestamp_s",
        "frozen_prior_fwhm_hz",
        "frozen_fwhm_source_kind",
        "frozen_fwhm_source_scan_index",
        "frozen_fwhm_reference_timestamp_s",
        "frozen_fwhm_release_sequence_index",
        "frozen_fwhm_release_timestamp_s",
        "frequency_hz",
        "integration_time_s",
        "expected_sequence_index",
        "expected_end_timestamp_s",
        "expected_nominal_exposure_photons",
    ]
    partial = make_partial_scan(queries=list(make_partial_scan().queries))
    assert type(partial.queries) is tuple


def test_all_task_two_records_have_exact_documented_field_surfaces() -> None:
    assert [field.name for field in fields(SparsePartialScan)] == [
        "scan_index",
        "identity_scan_index",
        "resonance_id",
        "frozen_fast_center_hz",
        "frozen_fast_center_source_kind",
        "frozen_fast_center_source_pair_index",
        "frozen_fast_center_reference_timestamp_s",
        "frozen_fast_center_release_sequence_index",
        "frozen_fast_center_release_timestamp_s",
        "frozen_prior_fwhm_hz",
        "frozen_fwhm_source_kind",
        "frozen_fwhm_source_scan_index",
        "frozen_fwhm_reference_timestamp_s",
        "frozen_fwhm_release_sequence_index",
        "frozen_fwhm_release_timestamp_s",
        "queries",
        "observations",
    ]
    assert [field.name for field in fields(SparseLinewidthScanResult)] == [
        *[field.name for field in fields(SparsePartialScan)[:-2]],
        "queries",
        "observations",
        "public_reference_timestamp_s",
        "release_sequence_index",
        "release_timestamp_s",
        "status",
        "failure_code",
        "fitted_center_correction_hz",
        "fitted_local_center_hz",
        "fitted_fwhm_hz",
        "fitted_amplitude",
        "fitted_baseline_offset",
        "fitted_q",
        "rmse",
        "amplitude_normalized_rmse",
        "scaled_jacobian_rank",
        "scaled_jacobian_condition",
        "scipy_status",
        "scipy_message",
        "nfev",
        "fit_cpu_time_s",
    ]
    assert [field.name for field in fields(CompositeIdentityEstimate)] == [
        "resonance_id",
        "fast_center_hz",
        "fast_center_source_kind",
        "fast_center_source_pair_index",
        "fast_center_reference_timestamp_s",
        "fast_center_release_sequence_index",
        "fast_center_release_timestamp_s",
        "active_fwhm_hz",
        "fwhm_source_kind",
        "fwhm_source_scan_index",
        "fwhm_reference_timestamp_s",
        "fwhm_release_sequence_index",
        "fwhm_release_timestamp_s",
        "live_q",
        "center_age_s",
        "fwhm_age_s",
        "center_release_age_s",
        "fwhm_release_age_s",
        "completed_fast_pairs",
        "completed_sparse_scans",
        "latest_fast_pair",
        "latest_sparse_scan",
    ]
    assert [field.name for field in fields(SparseLinewidthCompositeEstimate)] == [
        "configuration",
        "identities",
        "calibration_source_id",
        "calibration_source_provenance",
        "calibration_budget_treatment",
        "pending_mode",
        "pending_query",
        "incomplete_fast_pair",
        "incomplete_sparse_scan",
        "fast_pair_history",
        "sparse_scan_history",
        "accepted_observations",
        "completed_fast_pairs",
        "completed_sparse_scans",
        "fast_pairs_since_scan",
        "current_sequence_index",
        "current_timestamp_s",
        "fast_tracking_resources",
        "sparse_tracking_resources",
        "tracking_resources",
        "calibration_resources",
        "charged_resources",
        "budget_ceiling",
        "stopped_reason",
        "sparse_geometry_diagnostic",
        "fast_update_cpu_time_s",
        "sparse_update_cpu_time_s",
        "total_update_cpu_time_s",
        "seed",
    ]
    assert [field.name for field in fields(SparseLinewidthCompositeUpdate)] == [
        "query",
        "observation",
        "completed_fast_pair",
        "completed_sparse_scan",
        "estimate",
        "update_cpu_time_s",
    ]


@pytest.mark.parametrize("length", (1, 2, 3, 4))
def test_partial_scan_accepts_exact_prefix_lengths(length: int) -> None:
    queries = tuple(
        make_sparse_query(
            acquisition_index=index,
            point_index=index,
            expected_sequence_index=index,
            expected_end_timestamp_s=(index + 1) * 0.005,
            offset_multiplier=(0.5, -1.0, 0.0, 1.0, -0.5)[index],
            frequency_hz=2.87e9 + (0.5, -1.0, 0.0, 1.0, -0.5)[index] * 1.0e6,
        )
        for index in range(length)
    )
    observations = tuple(
        EstimatorObservation(
            query.expected_sequence_index,
            query.expected_end_timestamp_s,
            query.frequency_hz,
            1.0,
            query.integration_time_s,
            query.expected_nominal_exposure_photons,
        )
        for query in queries
    )
    partial = make_partial_scan(queries=queries, observations=observations)
    assert len(partial.queries) == length


def test_scan_result_requires_all_five_query_echoes_and_status_matrix() -> None:
    assert len(make_scan_result().queries) == 5
    with pytest.raises((TypeError, ValueError)):
        make_scan_result(status="success", failure_code="optimizer_failed")
    with pytest.raises((TypeError, ValueError)):
        make_scan_result(status="failure", failure_code=None)
    optimizer_failure = make_scan_result(
        status="failure",
        failure_code="optimizer_failed",
        fitted_center_correction_hz=None,
        fitted_local_center_hz=None,
        fitted_fwhm_hz=None,
        fitted_amplitude=None,
        fitted_baseline_offset=None,
        fitted_q=None,
        rmse=None,
        amplitude_normalized_rmse=None,
        scaled_jacobian_rank=None,
        scaled_jacobian_condition=None,
    )
    assert optimizer_failure.scipy_status == 1


@pytest.mark.parametrize(
    ("failure_code", "overrides"),
    (
        (
            "model_evaluation_failed",
            {
                "scipy_status": None,
                "scipy_message": None,
                "nfev": None,
                "fitted_center_correction_hz": None,
                "fitted_local_center_hz": None,
                "fitted_fwhm_hz": None,
                "fitted_amplitude": None,
                "fitted_baseline_offset": None,
                "rmse": None,
                "amplitude_normalized_rmse": None,
                "scaled_jacobian_rank": None,
                "scaled_jacobian_condition": None,
            },
        ),
        (
            "optimizer_failed",
            {
                "fitted_center_correction_hz": None,
                "fitted_local_center_hz": None,
                "fitted_fwhm_hz": None,
                "fitted_amplitude": None,
                "fitted_baseline_offset": None,
                "rmse": None,
                "amplitude_normalized_rmse": None,
                "scaled_jacobian_rank": None,
                "scaled_jacobian_condition": None,
            },
        ),
        (
            "nonfinite_solution",
            {
                "fitted_center_correction_hz": None,
                "fitted_local_center_hz": None,
                "fitted_fwhm_hz": None,
                "fitted_amplitude": None,
                "fitted_baseline_offset": None,
                "rmse": None,
                "amplitude_normalized_rmse": None,
                "scaled_jacobian_rank": None,
                "scaled_jacobian_condition": None,
            },
        ),
        (
            "bounds_active",
            {
                "scaled_jacobian_rank": None,
                "scaled_jacobian_condition": None,
            },
        ),
        (
            "rank_deficient",
            {"scaled_jacobian_rank": 3, "scaled_jacobian_condition": None},
        ),
        ("ill_conditioned", {}),
        ("amplitude_unresolved", {}),
        ("residual_quality_failed", {}),
    ),
)
def test_scan_result_accepts_each_ordered_failure_diagnostic_row(
    failure_code: str, overrides: dict[str, object]
) -> None:
    result = make_scan_result(
        status="failure", failure_code=failure_code, fitted_q=None, **overrides
    )
    assert result.failure_code == failure_code


def test_aggregate_enforces_one_incomplete_block_and_history_counter_equations() -> (
    None
):
    estimate = make_composite_estimate()
    assert isinstance(estimate, SparseLinewidthCompositeEstimate)
    with pytest.raises((TypeError, ValueError)):
        make_composite_estimate(
            incomplete_sparse_scan=make_partial_scan(),
            incomplete_fast_pair=object(),
        )


def test_update_requires_exact_query_and_estimate_echo() -> None:
    query = make_sparse_query()
    observation = EstimatorObservation(0, 0.005, query.frequency_hz, 1.0, 0.005, 1.0)
    with pytest.raises(ValueError):
        SparseLinewidthCompositeUpdate(
            query,
            observation,
            None,
            None,
            make_composite_estimate(),
            0.0,
        )


def _diagnostic(**overrides: object) -> SparseGeometryUnavailableDiagnostic:
    values: dict[str, object] = {
        "failure_code": "empty_fit_bounds",
        "scan_index": 0,
        "identity_scan_index": 0,
        "resonance_id": "r0",
        "fast_center_hz": 2.87e9,
        "fast_center_source_kind": "calibration",
        "fast_center_source_pair_index": None,
        "fast_center_reference_timestamp_s": 0.1,
        "fast_center_release_sequence_index": 7,
        "fast_center_release_timestamp_s": 0.1,
        "prior_fwhm_hz": 1.0e6,
        "fwhm_source_kind": "calibration",
        "fwhm_source_scan_index": None,
        "fwhm_reference_timestamp_s": 0.1,
        "fwhm_release_sequence_index": 7,
        "fwhm_release_timestamp_s": 0.1,
        "proposed_frequency_min_hz": 2.869e9,
        "proposed_frequency_max_hz": 2.871e9,
        "calibration_cell_lower_hz": 2.80e9,
        "calibration_cell_upper_hz": 2.94e9,
        "source_frequency_min_hz": 2.74e9,
        "source_frequency_max_hz": 3.02e9,
    }
    values.update(overrides)
    return SparseGeometryUnavailableDiagnostic(**values)  # type: ignore[arg-type]


def test_sparse_linewidth_primitive_aliases_are_closed_and_public() -> None:
    assert get_args(SparseLinewidthFailureCode) == (
        "model_evaluation_failed",
        "optimizer_failed",
        "nonfinite_solution",
        "bounds_active",
        "rank_deficient",
        "ill_conditioned",
        "amplitude_unresolved",
        "residual_quality_failed",
    )
    assert get_args(CompositeMode) == ("fast_pair", "sparse_scan")
    assert get_args(SparseLinewidthSourceKind) == ("calibration", "scan")
    assert get_args(CompositeStopReason) == (
        "budget_exhausted",
        "sparse_geometry_unavailable",
    )
    assert get_args(SparseGeometryFailureCode) == (
        "nonrepresentable_frequency_lower",
        "nonrepresentable_frequency_upper",
        "empty_fit_bounds",
        "calibration_cell_violation",
        "source_domain_violation",
    )
    assert get_args(SparseResetFailureCode) == (
        "invalid_argument_type",
        "configuration_mismatch",
        "calibration_mismatch",
        "metadata_mismatch",
        "invalid_base_sparse_geometry",
        "budget_mismatch",
        "initial_state_construction_failed",
    )
    assert get_args(SparseObservationValidationCode) == (
        "invalid_observation_type",
        "no_pending_query",
        "pending_mode_mismatch",
        "fast_query_echo_mismatch",
        "sparse_query_echo_mismatch",
        "sequence_mismatch",
        "frequency_mismatch",
        "integration_time_mismatch",
        "endpoint_mismatch",
        "nominal_exposure_mismatch",
        "invalid_observation_value",
    )
    assert get_args(SparseUpdateConstructionCode) == (
        "fast_partial_pair_construction_failed",
        "fast_pair_result_construction_failed",
        "fast_identity_estimate_construction_failed",
        "sparse_partial_scan_construction_failed",
        "sparse_scan_result_construction_failed",
        "sparse_identity_estimate_construction_failed",
        "resource_construction_failed",
        "aggregate_estimate_construction_failed",
        "update_construction_failed",
    )


def test_sparse_linewidth_configuration_has_normative_defaults() -> None:
    configuration = SparseLinewidthConfiguration()
    assert configuration.scan_period_fast_pairs == 8
    assert configuration.integration_time_s == 0.005
    assert configuration.center_correction_limit_fwhm_fraction == 0.5
    assert configuration.min_fwhm_prior_ratio == 0.5
    assert configuration.max_fwhm_prior_ratio == 2.0
    assert configuration.min_resolved_amplitude_source_ratio == 0.25
    assert configuration.max_amplitude_source_ratio == 4.0
    assert configuration.baseline_offset_source_amplitude_fraction == 1.0
    assert configuration.rank_rtol == 1.0e-10
    assert configuration.max_scaled_jacobian_condition == 1.0e8
    assert configuration.min_interior_bound_fraction == 1.0e-6
    assert configuration.max_amplitude_normalized_rmse == 0.10
    assert configuration.max_nfev == 4000
    with pytest.raises(TypeError):
        replace(configuration, max_nfev=True)


def test_sparse_linewidth_configuration_has_exact_documented_field_order() -> None:
    assert [item.name for item in fields(SparseLinewidthConfiguration)] == [
        "scan_period_fast_pairs",
        "integration_time_s",
        "center_correction_limit_fwhm_fraction",
        "min_fwhm_prior_ratio",
        "max_fwhm_prior_ratio",
        "min_resolved_amplitude_source_ratio",
        "max_amplitude_source_ratio",
        "baseline_offset_source_amplitude_fraction",
        "rank_rtol",
        "max_scaled_jacobian_condition",
        "min_interior_bound_fraction",
        "max_amplitude_normalized_rmse",
        "max_nfev",
    ]


def test_sparse_linewidth_configuration_is_frozen_slotted_and_canonical() -> None:
    configuration = SparseLinewidthConfiguration(
        scan_period_fast_pairs=np.int64(8),
        integration_time_s=np.float32(0.005),
        center_correction_limit_fwhm_fraction=np.float32(0.5),
        min_fwhm_prior_ratio=np.float32(0.5),
        max_fwhm_prior_ratio=np.float32(2.0),
        min_resolved_amplitude_source_ratio=np.float32(0.25),
        max_amplitude_source_ratio=np.float32(4.0),
        baseline_offset_source_amplitude_fraction=np.float32(1.0),
        rank_rtol=np.float64(1.0e-10),
        max_scaled_jacobian_condition=np.float64(1.0e8),
        min_interior_bound_fraction=np.float64(1.0e-6),
        max_amplitude_normalized_rmse=np.float64(0.1),
        max_nfev=np.int64(4000),
    )

    assert is_dataclass(configuration)
    assert not hasattr(configuration, "__dict__")
    assert type(configuration.scan_period_fast_pairs) is int
    assert type(configuration.max_nfev) is int
    assert all(
        type(getattr(configuration, item.name)) is float
        for item in fields(configuration)
        if item.name not in {"scan_period_fast_pairs", "max_nfev"}
    )
    with pytest.raises(FrozenInstanceError):
        configuration.integration_time_s = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scan_period_fast_pairs", True),
        ("scan_period_fast_pairs", 0),
        ("scan_period_fast_pairs", 1.0),
        ("integration_time_s", True),
        ("integration_time_s", np.array(0.005)),
        ("integration_time_s", 1.0 + 1.0j),
        ("integration_time_s", np.inf),
        ("center_correction_limit_fwhm_fraction", 0.0),
        ("min_fwhm_prior_ratio", 0.0),
        ("max_fwhm_prior_ratio", np.nan),
        ("min_resolved_amplitude_source_ratio", 0.0),
        ("max_amplitude_source_ratio", np.inf),
        ("baseline_offset_source_amplitude_fraction", 0.0),
        ("rank_rtol", 0.0),
        ("rank_rtol", 1.0),
        ("max_scaled_jacobian_condition", 0.999999),
        ("min_interior_bound_fraction", -1.0e-6),
        ("min_interior_bound_fraction", 0.5),
        ("max_amplitude_normalized_rmse", 0.0),
        ("max_nfev", True),
        ("max_nfev", 0),
        ("max_nfev", np.array(4000)),
    ],
)
def test_sparse_linewidth_configuration_rejects_invalid_scalars(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SparseLinewidthConfiguration(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_fwhm_prior_ratio": 2.0, "max_fwhm_prior_ratio": 2.0},
        {"min_fwhm_prior_ratio": 2.1, "max_fwhm_prior_ratio": 2.0},
    ],
)
def test_sparse_linewidth_configuration_enforces_ratio_relations(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SparseLinewidthConfiguration(**overrides)  # type: ignore[arg-type]


def test_sparse_geometry_diagnostic_has_exact_field_order_and_is_frozen_slotted() -> (
    None
):
    diagnostic = _diagnostic()
    assert [item.name for item in fields(diagnostic)] == [
        "failure_code",
        "scan_index",
        "identity_scan_index",
        "resonance_id",
        "fast_center_hz",
        "fast_center_source_kind",
        "fast_center_source_pair_index",
        "fast_center_reference_timestamp_s",
        "fast_center_release_sequence_index",
        "fast_center_release_timestamp_s",
        "prior_fwhm_hz",
        "fwhm_source_kind",
        "fwhm_source_scan_index",
        "fwhm_reference_timestamp_s",
        "fwhm_release_sequence_index",
        "fwhm_release_timestamp_s",
        "proposed_frequency_min_hz",
        "proposed_frequency_max_hz",
        "calibration_cell_lower_hz",
        "calibration_cell_upper_hz",
        "source_frequency_min_hz",
        "source_frequency_max_hz",
    ]
    assert is_dataclass(diagnostic)
    assert not hasattr(diagnostic, "__dict__")
    with pytest.raises(FrozenInstanceError):
        diagnostic.failure_code = "source_domain_violation"  # type: ignore[misc]


@pytest.mark.parametrize(
    "failure_code",
    ["nonrepresentable_frequency_lower", "nonrepresentable_frequency_upper"],
)
def test_unrepresentable_geometry_has_no_proposed_bounds(
    failure_code: str,
) -> None:
    diagnostic = _diagnostic(
        failure_code=failure_code,
        proposed_frequency_min_hz=None,
        proposed_frequency_max_hz=None,
    )
    assert diagnostic.proposed_frequency_min_hz is None
    assert diagnostic.proposed_frequency_max_hz is None
    with pytest.raises(ValueError):
        _diagnostic(failure_code=failure_code)


@pytest.mark.parametrize(
    "failure_code",
    ["empty_fit_bounds", "calibration_cell_violation", "source_domain_violation"],
)
def test_sparse_geometry_diagnostic_requires_finite_ordered_proposed_bounds_otherwise(
    failure_code: str,
) -> None:
    diagnostic = _diagnostic(failure_code=failure_code)
    assert type(diagnostic.proposed_frequency_min_hz) is float
    assert type(diagnostic.proposed_frequency_max_hz) is float
    with pytest.raises(ValueError):
        _diagnostic(
            failure_code=failure_code,
            proposed_frequency_min_hz=None,
            proposed_frequency_max_hz=None,
        )
    with pytest.raises(ValueError):
        _diagnostic(
            failure_code=failure_code,
            proposed_frequency_min_hz=2.871e9,
            proposed_frequency_max_hz=2.869e9,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failure_code", "unknown"),
        ("scan_index", True),
        ("identity_scan_index", -1),
        ("resonance_id", "  "),
        ("fast_center_hz", np.nan),
        ("fast_center_source_kind", "scan"),
        ("fast_center_reference_timestamp_s", np.inf),
        ("prior_fwhm_hz", 0.0),
        ("fwhm_source_kind", "pair"),
        ("calibration_cell_lower_hz", np.nan),
        ("calibration_cell_upper_hz", 2.80e9),
        ("source_frequency_min_hz", np.inf),
        ("source_frequency_max_hz", 2.74e9),
    ],
)
def test_sparse_geometry_diagnostic_rejects_invalid_facts(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _diagnostic(**{field: value})


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (SparseLinewidthResetError, "invalid_argument_type"),
        (SparseLinewidthObservationValidationError, "pending_mode_mismatch"),
        (
            SparseLinewidthUpdateConstructionError,
            "sparse_scan_result_construction_failed",
        ),
    ],
)
def test_sparse_linewidth_errors_validate_and_canonicalize_closed_codes(
    error_type: type[Exception], code: str
) -> None:
    class CapabilityString(str):
        pass

    capable_code = CapabilityString(code)
    capable_code.callback = lambda: None
    error = error_type(capable_code, "failure")  # type: ignore[call-arg]
    assert error.code == code  # type: ignore[attr-defined]
    assert type(error.code) is str  # type: ignore[attr-defined]
    assert not hasattr(error.code, "callback")  # type: ignore[attr-defined]
    assert error.message == "failure"  # type: ignore[attr-defined]
    assert str(error) == "failure"
    with pytest.raises((TypeError, ValueError)):
        error_type("unknown", "failure")  # type: ignore[call-arg]
    with pytest.raises((TypeError, ValueError)):
        error_type(code, "")  # type: ignore[call-arg]
