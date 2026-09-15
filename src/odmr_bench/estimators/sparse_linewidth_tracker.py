"""Composite scheduling shell for sparse five-point linewidth tracking."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields
from numbers import Integral
from typing import final

import numpy as np

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators.sparse_linewidth_fit import (
    _construct_sparse_fit_geometry,
    _SparseFitGeometry,
    _SparseGeometryConstructionError,
    _validate_calibration_sparse_geometry,
    fit_sparse_linewidth,
)
from odmr_bench.estimators.sparse_linewidth_types import (
    CompositeIdentityEstimate,
    SparseGeometryUnavailableDiagnostic,
    SparseLinewidthCompositeEstimate,
    SparseLinewidthCompositeUpdate,
    SparseLinewidthConfiguration,
    SparseLinewidthObservationValidationError,
    SparseLinewidthQuery,
    SparseLinewidthResetError,
    SparseLinewidthScanResult,
    SparseLinewidthUpdateConstructionError,
    SparsePartialScan,
)
from odmr_bench.estimators.two_point_calibration import (
    _validate_two_point_calibration_source_integrity,
)
from odmr_bench.estimators.two_point_resources import _zero_public_resources
from odmr_bench.estimators.two_point_tracker import _pair_model_geometry
from odmr_bench.estimators.two_point_types import (
    PublicAcquisitionResources,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointCalibrationSource,
    TwoPointIdentityCalibration,
    TwoPointPairResult,
    TwoPointPartialPair,
    TwoPointQuery,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
)


@dataclass(frozen=True, slots=True)
class _CompositeTrackerState:
    calibration: TwoPointCalibration
    metadata: TwoPointRunMetadata
    reserved_fast_queries: tuple[TwoPointQuery, TwoPointQuery] | None
    reserved_sparse_queries: tuple[
        SparseLinewidthQuery,
        SparseLinewidthQuery,
        SparseLinewidthQuery,
        SparseLinewidthQuery,
        SparseLinewidthQuery,
    ] | None
    estimate: SparseLinewidthCompositeEstimate


def _snapshot_sparse_configuration(
    configuration: SparseLinewidthConfiguration,
) -> SparseLinewidthConfiguration:
    return SparseLinewidthConfiguration(
        scan_period_fast_pairs=configuration.scan_period_fast_pairs,
        integration_time_s=configuration.integration_time_s,
        center_correction_limit_fwhm_fraction=(
            configuration.center_correction_limit_fwhm_fraction
        ),
        min_fwhm_prior_ratio=configuration.min_fwhm_prior_ratio,
        max_fwhm_prior_ratio=configuration.max_fwhm_prior_ratio,
        min_resolved_amplitude_source_ratio=(
            configuration.min_resolved_amplitude_source_ratio
        ),
        max_amplitude_source_ratio=configuration.max_amplitude_source_ratio,
        baseline_offset_source_amplitude_fraction=(
            configuration.baseline_offset_source_amplitude_fraction
        ),
        rank_rtol=configuration.rank_rtol,
        max_scaled_jacobian_condition=(
            configuration.max_scaled_jacobian_condition
        ),
        min_interior_bound_fraction=configuration.min_interior_bound_fraction,
        max_amplitude_normalized_rmse=(
            configuration.max_amplitude_normalized_rmse
        ),
        max_nfev=configuration.max_nfev,
    )


def _replace_estimate(
    estimate: SparseLinewidthCompositeEstimate, **changes: object
) -> SparseLinewidthCompositeEstimate:
    field_names = tuple(item.name for item in fields(type(estimate)))
    unknown = set(changes).difference(field_names)
    if unknown:
        raise TypeError(f"unknown estimate fields: {sorted(unknown)!r}")
    values = {name: getattr(estimate, name) for name in field_names}
    values.update(changes)
    return SparseLinewidthCompositeEstimate(**values)  # type: ignore[arg-type]


def _advance_query_charge(
    resources: PublicAcquisitionResources,
    metadata: TwoPointRunMetadata,
    query: TwoPointQuery | SparseLinewidthQuery,
) -> PublicAcquisitionResources:
    return PublicAcquisitionResources(
        observations=resources.observations + 1,
        integration_time_s=resources.integration_time_s + query.integration_time_s,
        nominal_exposure_photons=(
            resources.nominal_exposure_photons
            + query.expected_nominal_exposure_photons
        ),
        realized_photons=resources.realized_photons,
        observations_without_realized_counts=(
            resources.observations_without_realized_counts
        ),
        virtual_elapsed_time_s=(
            resources.virtual_elapsed_time_s
            + (metadata.frequency_overhead_s + query.integration_time_s)
        ),
    )


def _reserve_two_atom_block(
    resources: PublicAcquisitionResources,
    metadata: TwoPointRunMetadata,
    queries: tuple[TwoPointQuery, TwoPointQuery],
) -> PublicAcquisitionResources:
    """Replay two indivisible charges with the exact arrival-order recurrence."""
    first_query, second_query = queries
    after_first = _advance_query_charge(resources, metadata, first_query)
    return _advance_query_charge(after_first, metadata, second_query)


def _reserve_five_atom_block(
    resources: PublicAcquisitionResources,
    metadata: TwoPointRunMetadata,
    queries: tuple[
        SparseLinewidthQuery,
        SparseLinewidthQuery,
        SparseLinewidthQuery,
        SparseLinewidthQuery,
        SparseLinewidthQuery,
    ],
) -> PublicAcquisitionResources:
    """Replay five indivisible charges with the exact arrival-order recurrence."""
    first, second, third, fourth, fifth = queries
    after_first = _advance_query_charge(resources, metadata, first)
    after_second = _advance_query_charge(after_first, metadata, second)
    after_third = _advance_query_charge(after_second, metadata, third)
    after_fourth = _advance_query_charge(after_third, metadata, fourth)
    return _advance_query_charge(after_fourth, metadata, fifth)


def _advance_observation_resources(
    resources: PublicAcquisitionResources,
    observation: EstimatorObservation,
    metadata: TwoPointRunMetadata,
) -> PublicAcquisitionResources:
    return PublicAcquisitionResources(
        observations=resources.observations + 1,
        integration_time_s=(
            resources.integration_time_s + observation.integration_time_s
        ),
        nominal_exposure_photons=(
            resources.nominal_exposure_photons
            + observation.nominal_exposure_photons
        ),
        realized_photons=(
            resources.realized_photons + (observation.realized_photons or 0)
        ),
        observations_without_realized_counts=(
            resources.observations_without_realized_counts
            + int(observation.realized_photons is None)
        ),
        virtual_elapsed_time_s=(
            resources.virtual_elapsed_time_s
            + (metadata.frequency_overhead_s + observation.integration_time_s)
        ),
    )


def _within_ceiling(
    resources: PublicAcquisitionResources, ceiling: TwoPointBudgetCeiling
) -> bool:
    observations_fit = (
        ceiling.max_observations is None
        or resources.observations <= ceiling.max_observations
    )
    integration_fits = (
        ceiling.max_integration_time_s is None
        or resources.integration_time_s <= ceiling.max_integration_time_s
    )
    exposure_fits = (
        ceiling.max_nominal_exposure_photons is None
        or resources.nominal_exposure_photons
        <= ceiling.max_nominal_exposure_photons
    )
    elapsed_fits = (
        ceiling.max_virtual_elapsed_time_s is None
        or resources.virtual_elapsed_time_s <= ceiling.max_virtual_elapsed_time_s
    )
    return observations_fit and integration_fits and exposure_fits and elapsed_fits


def _construct_fast_queries(
    calibration: TwoPointCalibration,
    metadata: TwoPointRunMetadata,
    estimate: SparseLinewidthCompositeEstimate,
) -> tuple[TwoPointQuery, TwoPointQuery]:
    fast_configuration = calibration.configuration
    integration_time_s = fast_configuration.integration_time_s
    nominal_exposure_photons = (
        metadata.nominal_photon_rate_hz * integration_time_s
    )
    if not math.isfinite(nominal_exposure_photons):
        raise ValueError("fast-query nominal exposure must remain finite")
    elapsed_atom_s = metadata.frequency_overhead_s + integration_time_s
    if not math.isfinite(elapsed_atom_s):
        raise ValueError("fast-query elapsed charge must remain finite")

    first_sequence_index = (
        0
        if estimate.current_sequence_index is None
        else estimate.current_sequence_index + 1
    )
    second_sequence_index = first_sequence_index + 1
    first_endpoint_s = (
        estimate.current_timestamp_s + metadata.frequency_overhead_s
    ) + integration_time_s
    if not math.isfinite(first_endpoint_s):
        raise ValueError("first fast-query endpoint must remain finite")
    if first_endpoint_s <= estimate.current_timestamp_s:
        raise ValueError("first fast-query endpoint must strictly advance")
    second_endpoint_s = (
        first_endpoint_s + metadata.frequency_overhead_s
    ) + integration_time_s
    if not math.isfinite(second_endpoint_s):
        raise ValueError("second fast-query endpoint must remain finite")
    if second_endpoint_s <= first_endpoint_s:
        raise ValueError("second fast-query endpoint must strictly advance")

    pair_index = estimate.completed_fast_pairs
    query_index = pair_index + pair_index
    identity_index = pair_index % len(estimate.identities)
    identity = estimate.identities[identity_index]
    cell = calibration.identities[identity_index]
    first_side = "minus" if identity.completed_fast_pairs % 2 == 0 else "plus"
    second_side = "plus" if first_side == "minus" else "minus"
    first_frequency_hz = (
        identity.fast_center_hz - cell.offset_hz
        if first_side == "minus"
        else identity.fast_center_hz + cell.offset_hz
    )
    second_frequency_hz = (
        identity.fast_center_hz - cell.offset_hz
        if second_side == "minus"
        else identity.fast_center_hz + cell.offset_hz
    )
    first_query = TwoPointQuery(
        query_index=query_index,
        pair_index=pair_index,
        identity_pair_index=identity.completed_fast_pairs,
        resonance_id=identity.resonance_id,
        side=first_side,
        interrogation_center_hz=identity.fast_center_hz,
        frequency_hz=first_frequency_hz,
        integration_time_s=integration_time_s,
        expected_sequence_index=first_sequence_index,
        expected_end_timestamp_s=first_endpoint_s,
        expected_nominal_exposure_photons=nominal_exposure_photons,
    )
    second_query = TwoPointQuery(
        query_index=query_index + 1,
        pair_index=pair_index,
        identity_pair_index=identity.completed_fast_pairs,
        resonance_id=identity.resonance_id,
        side=second_side,
        interrogation_center_hz=identity.fast_center_hz,
        frequency_hz=second_frequency_hz,
        integration_time_s=integration_time_s,
        expected_sequence_index=second_sequence_index,
        expected_end_timestamp_s=second_endpoint_s,
        expected_nominal_exposure_photons=nominal_exposure_photons,
    )
    return first_query, second_query


def _construct_sparse_queries(
    geometry: _SparseFitGeometry,
    identity: CompositeIdentityEstimate,
    public_metadata: TwoPointRunMetadata,
    *,
    integration_time_s: float,
    first_acquisition_index: int,
    first_sequence_index: int,
    start_timestamp_s: float,
    scan_index: int,
    identity_scan_index: int,
) -> tuple[
    SparseLinewidthQuery,
    SparseLinewidthQuery,
    SparseLinewidthQuery,
    SparseLinewidthQuery,
    SparseLinewidthQuery,
]:
    """Add the current causal clock and policy to one pure frozen geometry."""
    if type(geometry) is not _SparseFitGeometry:
        raise TypeError("geometry must be an exact _SparseFitGeometry")
    if type(identity) is not CompositeIdentityEstimate:
        raise TypeError("identity must be an exact CompositeIdentityEstimate")
    if type(public_metadata) is not TwoPointRunMetadata:
        raise TypeError("public_metadata must be an exact TwoPointRunMetadata")
    if (
        type(integration_time_s) is not float
        or not math.isfinite(integration_time_s)
        or integration_time_s <= 0.0
    ):
        raise ValueError("integration_time_s must be a positive finite float")
    for value, name in (
        (first_acquisition_index, "first_acquisition_index"),
        (first_sequence_index, "first_sequence_index"),
        (scan_index, "scan_index"),
        (identity_scan_index, "identity_scan_index"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Integral, np.integer)
        ) or int(value) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if (
        type(start_timestamp_s) is not float
        or not math.isfinite(start_timestamp_s)
        or start_timestamp_s < 0.0
    ):
        raise ValueError("start_timestamp_s must be a nonnegative finite float")
    if (
        geometry.resonance_id != identity.resonance_id
        or geometry.frozen_fast_center_hz != identity.fast_center_hz
        or geometry.frozen_prior_fwhm_hz != identity.active_fwhm_hz
    ):
        raise ValueError("geometry must equal the target identity's frozen facts")

    nominal_exposure_photons = (
        public_metadata.nominal_photon_rate_hz * integration_time_s
    )
    if not math.isfinite(nominal_exposure_photons):
        raise ValueError("sparse-query nominal exposure must remain finite")
    elapsed_atom_s = public_metadata.frequency_overhead_s + integration_time_s
    if not math.isfinite(elapsed_atom_s):
        raise ValueError("sparse-query elapsed charge must remain finite")

    acquisition_index = int(first_acquisition_index)
    sequence_index = int(first_sequence_index)
    canonical_scan_index = int(scan_index)
    canonical_identity_scan_index = int(identity_scan_index)
    endpoint_s = start_timestamp_s
    queries: list[SparseLinewidthQuery] = []
    for point_index, (offset_multiplier, frequency_hz) in enumerate(
        zip(
            geometry.offset_multipliers,
            geometry.frequencies_hz,
            strict=True,
        )
    ):
        previous_endpoint_s = endpoint_s
        endpoint_s = (
            endpoint_s + public_metadata.frequency_overhead_s
        ) + integration_time_s
        if not math.isfinite(endpoint_s):
            raise ValueError("sparse-query endpoint must remain finite")
        if endpoint_s <= previous_endpoint_s:
            raise ValueError("sparse-query endpoint must strictly advance")
        queries.append(
            SparseLinewidthQuery(
                acquisition_index=acquisition_index + point_index,
                scan_index=canonical_scan_index,
                identity_scan_index=canonical_identity_scan_index,
                point_index=point_index,
                resonance_id=identity.resonance_id,
                offset_multiplier=offset_multiplier,
                frozen_fast_center_hz=identity.fast_center_hz,
                frozen_fast_center_source_kind=identity.fast_center_source_kind,
                frozen_fast_center_source_pair_index=(
                    identity.fast_center_source_pair_index
                ),
                frozen_fast_center_reference_timestamp_s=(
                    identity.fast_center_reference_timestamp_s
                ),
                frozen_fast_center_release_sequence_index=(
                    identity.fast_center_release_sequence_index
                ),
                frozen_fast_center_release_timestamp_s=(
                    identity.fast_center_release_timestamp_s
                ),
                frozen_prior_fwhm_hz=identity.active_fwhm_hz,
                frozen_fwhm_source_kind=identity.fwhm_source_kind,
                frozen_fwhm_source_scan_index=identity.fwhm_source_scan_index,
                frozen_fwhm_reference_timestamp_s=(
                    identity.fwhm_reference_timestamp_s
                ),
                frozen_fwhm_release_sequence_index=(
                    identity.fwhm_release_sequence_index
                ),
                frozen_fwhm_release_timestamp_s=(
                    identity.fwhm_release_timestamp_s
                ),
                frequency_hz=frequency_hz,
                integration_time_s=integration_time_s,
                expected_sequence_index=sequence_index + point_index,
                expected_end_timestamp_s=endpoint_s,
                expected_nominal_exposure_photons=nominal_exposure_photons,
            )
        )
    return tuple(queries)  # type: ignore[return-value]


def _construct_geometry_diagnostic(
    state: _CompositeTrackerState,
    identity: CompositeIdentityEstimate,
    *,
    scan_index: int,
    identity_scan_index: int,
    error: _SparseGeometryConstructionError,
) -> SparseGeometryUnavailableDiagnostic:
    cell = next(
        item
        for item in state.calibration.identities
        if item.resonance_id == identity.resonance_id
    )
    return SparseGeometryUnavailableDiagnostic(
        failure_code=error.geometry_failure_code,  # type: ignore[arg-type]
        scan_index=scan_index,
        identity_scan_index=identity_scan_index,
        resonance_id=identity.resonance_id,
        fast_center_hz=identity.fast_center_hz,
        fast_center_source_kind=identity.fast_center_source_kind,
        fast_center_source_pair_index=identity.fast_center_source_pair_index,
        fast_center_reference_timestamp_s=(
            identity.fast_center_reference_timestamp_s
        ),
        fast_center_release_sequence_index=(
            identity.fast_center_release_sequence_index
        ),
        fast_center_release_timestamp_s=(
            identity.fast_center_release_timestamp_s
        ),
        prior_fwhm_hz=identity.active_fwhm_hz,
        fwhm_source_kind=identity.fwhm_source_kind,
        fwhm_source_scan_index=identity.fwhm_source_scan_index,
        fwhm_reference_timestamp_s=identity.fwhm_reference_timestamp_s,
        fwhm_release_sequence_index=identity.fwhm_release_sequence_index,
        fwhm_release_timestamp_s=identity.fwhm_release_timestamp_s,
        proposed_frequency_min_hz=error.proposed_frequency_min_hz,
        proposed_frequency_max_hz=error.proposed_frequency_max_hz,
        calibration_cell_lower_hz=cell.calibration_cell_lower_hz,
        calibration_cell_upper_hz=cell.calibration_cell_upper_hz,
        source_frequency_min_hz=state.calibration.source.source_frequency_min_hz,
        source_frequency_max_hz=state.calibration.source.source_frequency_max_hz,
    )


def _construct_fast_pair_result(
    state: _CompositeTrackerState,
    query: TwoPointQuery,
    observation: EstimatorObservation,
) -> TwoPointPairResult:
    partial_pair = state.estimate.incomplete_fast_pair
    if partial_pair is None:
        raise ValueError("second-side update requires an incomplete fast pair")
    if partial_pair.first_side == "minus":
        minus_query = partial_pair.first_query
        minus_observation = partial_pair.first_observation
        plus_query = query
        plus_observation = observation
    else:
        minus_query = query
        minus_observation = observation
        plus_query = partial_pair.first_query
        plus_observation = partial_pair.first_observation

    cell_index = state.estimate.completed_fast_pairs % len(
        state.calibration.identities
    )
    cell = state.calibration.identities[cell_index]
    source_fit = state.calibration.source.source_fit
    center_hz = partial_pair.interrogation_center_hz
    observed_sum = minus_observation.fluorescence + plus_observation.fluorescence
    discriminator = None
    common_mode_target_depths = None
    raw_innovation_hz = None
    requested_step_hz = None
    candidate_center_hz = None
    zero_discriminator = None
    discriminator_slope_per_hz = None
    applied_step_hz = 0.0
    lock_state = "lost"
    failure_code = None
    if not math.isfinite(observed_sum):
        failure_code = "numerical_failure"
    elif observed_sum <= 0.0:
        failure_code = "invalid_pair_normalization"
    else:
        model_geometry = _pair_model_geometry(
            source_fit,
            cell.source_fit_index,
            minus_query.frequency_hz,
            plus_query.frequency_hz,
            center_hz,
        )
        if model_geometry is None:
            failure_code = "numerical_failure"
        else:
            model_sum, zero_discriminator, discriminator_slope_per_hz = (
                model_geometry
            )
            computed_discriminator = (
                minus_observation.fluorescence - plus_observation.fluorescence
            ) / observed_sum
            if math.isfinite(computed_discriminator):
                discriminator = computed_discriminator
                computed_common_mode = (
                    observed_sum - model_sum
                ) / cell.target_pair_depth
                if math.isfinite(computed_common_mode):
                    common_mode_target_depths = computed_common_mode
                    computed_raw_innovation_hz = (
                        computed_discriminator - zero_discriminator
                    ) / discriminator_slope_per_hz
                    if math.isfinite(computed_raw_innovation_hz):
                        raw_innovation_hz = computed_raw_innovation_hz
                        computed_requested_step_hz = (
                            state.calibration.configuration.proportional_gain
                            * computed_raw_innovation_hz
                        )
                        if math.isfinite(computed_requested_step_hz):
                            requested_step_hz = computed_requested_step_hz
                            computed_applied_step_hz = max(
                                -cell.max_step_hz,
                                min(computed_requested_step_hz, cell.max_step_hz),
                            )
                            computed_candidate_center_hz = (
                                center_hz + computed_applied_step_hz
                            )
                            if math.isfinite(computed_candidate_center_hz):
                                candidate_center_hz = computed_candidate_center_hz
                            else:
                                failure_code = "numerical_failure"
                        else:
                            failure_code = "numerical_failure"
                    else:
                        failure_code = "numerical_failure"
                else:
                    failure_code = "numerical_failure"
            else:
                failure_code = "numerical_failure"

    if failure_code is None:
        assert common_mode_target_depths is not None
        assert raw_innovation_hz is not None
        assert requested_step_hz is not None
        assert candidate_center_hz is not None
        common_limit = (
            state.calibration.configuration.common_mode_limit_target_depths
        )
        if common_limit is not None and abs(common_mode_target_depths) > common_limit:
            failure_code = "common_mode_limit_exceeded"
        elif abs(raw_innovation_hz) > cell.capture_radius_hz:
            failure_code = "capture_exceeded"
        elif not (
            cell.allowed_center_min_hz
            <= candidate_center_hz
            <= cell.allowed_center_max_hz
        ):
            failure_code = "calibration_domain_exceeded"
        else:
            applied_step_hz = computed_applied_step_hz
            lock_state = (
                "tracking"
                if applied_step_hz == requested_step_hz
                else "step_limited"
            )

    first_observation = partial_pair.first_observation
    first_reference_s = (
        first_observation.timestamp_s - first_observation.integration_time_s / 2.0
    )
    second_reference_s = (
        observation.timestamp_s - observation.integration_time_s / 2.0
    )
    pair_reference_timestamp_s = first_reference_s + (
        second_reference_s - first_reference_s
    ) / 2.0
    return TwoPointPairResult(
        pair_index=query.pair_index,
        identity_pair_index=query.identity_pair_index,
        resonance_id=query.resonance_id,
        interrogation_center_hz=center_hz,
        first_side=partial_pair.first_side,
        minus_query=minus_query,
        plus_query=plus_query,
        minus_observation=minus_observation,
        plus_observation=plus_observation,
        pair_reference_timestamp_s=pair_reference_timestamp_s,
        release_sequence_index=observation.sequence_index,
        release_timestamp_s=observation.timestamp_s,
        discriminator=discriminator,
        zero_discriminator=zero_discriminator,
        discriminator_slope_per_hz=discriminator_slope_per_hz,
        raw_innovation_hz=raw_innovation_hz,
        requested_step_hz=requested_step_hz,
        candidate_center_hz=candidate_center_hz,
        applied_step_hz=applied_step_hz,
        common_mode_target_depths=common_mode_target_depths,
        lock_state=lock_state,
        failure_code=failure_code,
    )


def _identity_after_fast_observation(
    identity: CompositeIdentityEstimate,
    observation: EstimatorObservation,
    *,
    completed_pair: TwoPointPairResult | None = None,
) -> CompositeIdentityEstimate:
    successful_pair = completed_pair is not None and completed_pair.lock_state in {
        "tracking",
        "step_limited",
    }
    fast_center_hz = (
        completed_pair.candidate_center_hz
        if successful_pair and completed_pair is not None
        else identity.fast_center_hz
    )
    assert fast_center_hz is not None
    center_source_kind = "pair" if successful_pair else identity.fast_center_source_kind
    center_source_pair_index = (
        completed_pair.pair_index
        if successful_pair and completed_pair is not None
        else identity.fast_center_source_pair_index
    )
    center_reference_s = (
        completed_pair.pair_reference_timestamp_s
        if successful_pair and completed_pair is not None
        else identity.fast_center_reference_timestamp_s
    )
    center_release_sequence_index = (
        completed_pair.release_sequence_index
        if successful_pair and completed_pair is not None
        else identity.fast_center_release_sequence_index
    )
    center_release_s = (
        completed_pair.release_timestamp_s
        if successful_pair and completed_pair is not None
        else identity.fast_center_release_timestamp_s
    )
    return CompositeIdentityEstimate(
        resonance_id=identity.resonance_id,
        fast_center_hz=fast_center_hz,
        fast_center_source_kind=center_source_kind,
        fast_center_source_pair_index=center_source_pair_index,
        fast_center_reference_timestamp_s=center_reference_s,
        fast_center_release_sequence_index=center_release_sequence_index,
        fast_center_release_timestamp_s=center_release_s,
        active_fwhm_hz=identity.active_fwhm_hz,
        fwhm_source_kind=identity.fwhm_source_kind,
        fwhm_source_scan_index=identity.fwhm_source_scan_index,
        fwhm_reference_timestamp_s=identity.fwhm_reference_timestamp_s,
        fwhm_release_sequence_index=identity.fwhm_release_sequence_index,
        fwhm_release_timestamp_s=identity.fwhm_release_timestamp_s,
        live_q=fast_center_hz / identity.active_fwhm_hz,
        center_age_s=observation.timestamp_s - center_reference_s,
        fwhm_age_s=(
            observation.timestamp_s - identity.fwhm_reference_timestamp_s
        ),
        center_release_age_s=observation.timestamp_s - center_release_s,
        fwhm_release_age_s=(
            observation.timestamp_s - identity.fwhm_release_timestamp_s
        ),
        completed_fast_pairs=(
            identity.completed_fast_pairs + int(completed_pair is not None)
        ),
        completed_sparse_scans=identity.completed_sparse_scans,
        latest_fast_pair=(
            completed_pair if completed_pair is not None else identity.latest_fast_pair
        ),
        latest_sparse_scan=identity.latest_sparse_scan,
    )


def _identity_after_sparse_observation(
    identity: CompositeIdentityEstimate,
    observation: EstimatorObservation,
    *,
    live_q: float,
    completed_scan: SparseLinewidthScanResult | None = None,
) -> CompositeIdentityEstimate:
    successful_scan = completed_scan is not None and completed_scan.status == "success"
    active_fwhm_hz = (
        completed_scan.fitted_fwhm_hz
        if successful_scan and completed_scan is not None
        else identity.active_fwhm_hz
    )
    assert active_fwhm_hz is not None
    fwhm_source_kind = "scan" if successful_scan else identity.fwhm_source_kind
    fwhm_source_scan_index = (
        completed_scan.scan_index
        if successful_scan and completed_scan is not None
        else identity.fwhm_source_scan_index
    )
    fwhm_reference_s = (
        completed_scan.public_reference_timestamp_s
        if successful_scan and completed_scan is not None
        else identity.fwhm_reference_timestamp_s
    )
    fwhm_release_sequence_index = (
        completed_scan.release_sequence_index
        if successful_scan and completed_scan is not None
        else identity.fwhm_release_sequence_index
    )
    fwhm_release_s = (
        completed_scan.release_timestamp_s
        if successful_scan and completed_scan is not None
        else identity.fwhm_release_timestamp_s
    )
    return CompositeIdentityEstimate(
        resonance_id=identity.resonance_id,
        fast_center_hz=identity.fast_center_hz,
        fast_center_source_kind=identity.fast_center_source_kind,
        fast_center_source_pair_index=identity.fast_center_source_pair_index,
        fast_center_reference_timestamp_s=(
            identity.fast_center_reference_timestamp_s
        ),
        fast_center_release_sequence_index=(
            identity.fast_center_release_sequence_index
        ),
        fast_center_release_timestamp_s=identity.fast_center_release_timestamp_s,
        active_fwhm_hz=active_fwhm_hz,
        fwhm_source_kind=fwhm_source_kind,
        fwhm_source_scan_index=fwhm_source_scan_index,
        fwhm_reference_timestamp_s=fwhm_reference_s,
        fwhm_release_sequence_index=fwhm_release_sequence_index,
        fwhm_release_timestamp_s=fwhm_release_s,
        live_q=live_q,
        center_age_s=(
            observation.timestamp_s - identity.fast_center_reference_timestamp_s
        ),
        fwhm_age_s=observation.timestamp_s - fwhm_reference_s,
        center_release_age_s=(
            observation.timestamp_s - identity.fast_center_release_timestamp_s
        ),
        fwhm_release_age_s=observation.timestamp_s - fwhm_release_s,
        completed_fast_pairs=identity.completed_fast_pairs,
        completed_sparse_scans=(
            identity.completed_sparse_scans + int(completed_scan is not None)
        ),
        latest_fast_pair=identity.latest_fast_pair,
        latest_sparse_scan=(
            completed_scan
            if completed_scan is not None
            else identity.latest_sparse_scan
        ),
    )


def _raise_update_error(code: str, message: str, cause: Exception) -> None:
    raise SparseLinewidthUpdateConstructionError(  # type: ignore[arg-type]
        code, message
    ) from cause


def _validate_calibration_integrity(calibration: TwoPointCalibration) -> None:
    source = calibration.source
    if (
        type(source) is not TwoPointCalibrationSource
        or type(calibration.configuration) is not TwoPointTrackerConfiguration
        or type(calibration.identities) is not tuple
        or not all(
            type(identity) is TwoPointIdentityCalibration
            for identity in calibration.identities
        )
    ):
        raise TypeError("calibration graph must retain exact public record types")
    _validate_two_point_calibration_source_integrity(source)
    rebuilt = TwoPointCalibration(
        source=source,
        configuration=calibration.configuration,
        budget_treatment=calibration.budget_treatment,
        identities=calibration.identities,
    )
    if (
        rebuilt.configuration != calibration.configuration
        or rebuilt.budget_treatment != calibration.budget_treatment
        or rebuilt.identities != calibration.identities
    ):
        raise ValueError("calibration must equal its defensive reconstruction")


def _mapped_calibration_epochs(
    public_metadata: TwoPointRunMetadata, calibration: TwoPointCalibration
) -> tuple[float, float]:
    source = calibration.source
    mapping = source.clock_mapping
    if public_metadata.tracker_clock_id != mapping.tracker_clock_id:
        raise ValueError("metadata tracker clock must equal the mapping target")
    if calibration.budget_treatment == "included_same_run" and (
        mapping.kind != "shared_clock"
    ):
        raise ValueError("included calibration requires a shared clock")
    mapped_values = tuple(
        value + mapping.offset_s
        for value in (
            source.source_first_timestamp_s,
            source.source_last_timestamp_s,
            source.physical_fit_epoch_s,
            source.availability_timestamp_s,
        )
    )
    if not all(math.isfinite(value) for value in mapped_values):
        raise ValueError("mapped calibration times must remain finite")
    mapped_physical_epoch_s = mapped_values[2]
    mapped_availability_s = mapped_values[3]
    if (
        mapped_availability_s < 0.0
        or mapped_availability_s > public_metadata.current_timestamp_s
    ):
        raise ValueError("mapped availability must precede the tracking endpoint")
    if calibration.budget_treatment == "included_same_run":
        if (
            public_metadata.nominal_photon_rate_hz
            != source.fluorescence_provenance.nominal_photon_rate_hz
            or public_metadata.frequency_overhead_s
            != source.source_frequency_overhead_s
        ):
            raise ValueError(
                "included metadata resources must equal calibration resources"
            )
        if (
            public_metadata.current_sequence_index
            != source.availability_sequence_index
            or public_metadata.current_timestamp_s != mapped_availability_s
        ):
            raise ValueError(
                "included tracking must start at the calibration boundary"
            )
    return mapped_physical_epoch_s, mapped_availability_s


def _raise_reset_error(code: str, message: str, cause: Exception | None = None) -> None:
    error = SparseLinewidthResetError(code, message)  # type: ignore[arg-type]
    if cause is None:
        raise error
    raise error from cause


@final
class SparseLinewidthCompositeTracker:
    """Own the causal fast/sparse schedule and its immutable public estimate."""

    __slots__ = ("_configuration", "_configuration_snapshot", "_state")

    def __init__(self, configuration: SparseLinewidthConfiguration) -> None:
        if type(configuration) is not SparseLinewidthConfiguration:
            raise TypeError(
                "configuration must be an exact SparseLinewidthConfiguration"
            )
        self._configuration = _snapshot_sparse_configuration(configuration)
        self._configuration_snapshot = _snapshot_sparse_configuration(configuration)
        self._state: _CompositeTrackerState | None = None

    def reset(
        self,
        public_metadata: TwoPointRunMetadata,
        calibration: TwoPointCalibration,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> None:
        """Construct one calibration-seeded boundary, committing only on success."""
        if (
            type(public_metadata) is not TwoPointRunMetadata
            or type(calibration) is not TwoPointCalibration
            or type(budget_ceiling) is not TwoPointBudgetCeiling
            or isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (Integral, np.integer))
            or int(seed) < 0
        ):
            _raise_reset_error(
                "invalid_argument_type",
                "reset requires exact public arguments and a nonnegative integer seed",
            )
        canonical_seed = int(seed)

        if (
            type(self._configuration) is not SparseLinewidthConfiguration
            or type(self._configuration_snapshot) is not SparseLinewidthConfiguration
            or self._configuration != self._configuration_snapshot
        ):
            _raise_reset_error(
                "configuration_mismatch",
                "tracker sparse configuration no longer equals its "
                "constructor snapshot",
            )

        try:
            _validate_calibration_integrity(calibration)
        except Exception as error:
            _raise_reset_error(
                "calibration_mismatch", f"calibration integrity failed: {error}", error
            )

        try:
            mapped_physical_epoch_s, mapped_availability_s = (
                _mapped_calibration_epochs(public_metadata, calibration)
            )
        except Exception as error:
            _raise_reset_error(
                "metadata_mismatch", f"metadata join failed: {error}", error
            )

        try:
            _validate_calibration_sparse_geometry(calibration, self._configuration)
        except SparseLinewidthResetError:
            raise
        except Exception as error:
            _raise_reset_error(
                "invalid_base_sparse_geometry",
                f"base sparse geometry failed: {error}",
                error,
            )

        charged_resources = (
            calibration.source.safe_resources
            if calibration.budget_treatment == "included_same_run"
            else _zero_public_resources()
        )
        if not _within_ceiling(charged_resources, budget_ceiling):
            _raise_reset_error(
                "budget_mismatch",
                "budget ceiling cannot be below charged starting resources",
            )

        try:
            release_sequence_index = (
                calibration.source.availability_sequence_index
                if calibration.budget_treatment == "included_same_run"
                else None
            )
            release_timestamp_s = mapped_availability_s
            identities = tuple(
                CompositeIdentityEstimate(
                    resonance_id=cell.resonance_id,
                    fast_center_hz=cell.calibration_center_hz,
                    fast_center_source_kind="calibration",
                    fast_center_source_pair_index=None,
                    fast_center_reference_timestamp_s=mapped_physical_epoch_s,
                    fast_center_release_sequence_index=release_sequence_index,
                    fast_center_release_timestamp_s=release_timestamp_s,
                    active_fwhm_hz=cell.calibration_fwhm_hz,
                    fwhm_source_kind="calibration",
                    fwhm_source_scan_index=None,
                    fwhm_reference_timestamp_s=mapped_physical_epoch_s,
                    fwhm_release_sequence_index=release_sequence_index,
                    fwhm_release_timestamp_s=release_timestamp_s,
                    live_q=cell.calibration_center_hz
                    / cell.calibration_fwhm_hz,
                    center_age_s=(
                        public_metadata.current_timestamp_s
                        - mapped_physical_epoch_s
                    ),
                    fwhm_age_s=(
                        public_metadata.current_timestamp_s
                        - mapped_physical_epoch_s
                    ),
                    center_release_age_s=(
                        public_metadata.current_timestamp_s - release_timestamp_s
                    ),
                    fwhm_release_age_s=(
                        public_metadata.current_timestamp_s - release_timestamp_s
                    ),
                    completed_fast_pairs=0,
                    completed_sparse_scans=0,
                    latest_fast_pair=None,
                    latest_sparse_scan=None,
                )
                for cell in calibration.identities
            )
            zero_resources = _zero_public_resources()
            estimate = SparseLinewidthCompositeEstimate(
                configuration=self._configuration,
                identities=identities,
                calibration_source_id=calibration.source.source_id,
                calibration_source_provenance=calibration.source.provenance,
                calibration_budget_treatment=calibration.budget_treatment,
                pending_mode=None,
                pending_query=None,
                incomplete_fast_pair=None,
                incomplete_sparse_scan=None,
                fast_pair_history=(),
                sparse_scan_history=(),
                accepted_observations=0,
                completed_fast_pairs=0,
                completed_sparse_scans=0,
                fast_pairs_since_scan=0,
                current_sequence_index=public_metadata.current_sequence_index,
                current_timestamp_s=public_metadata.current_timestamp_s,
                fast_tracking_resources=zero_resources,
                sparse_tracking_resources=zero_resources,
                tracking_resources=zero_resources,
                calibration_resources=calibration.source.safe_resources,
                charged_resources=charged_resources,
                budget_ceiling=budget_ceiling,
                stopped_reason=None,
                sparse_geometry_diagnostic=None,
                fast_update_cpu_time_s=0.0,
                sparse_update_cpu_time_s=0.0,
                total_update_cpu_time_s=0.0,
                seed=canonical_seed,
            )
            initial_queries = _construct_fast_queries(
                calibration, public_metadata, estimate
            )
            _reserve_two_atom_block(
                estimate.charged_resources, public_metadata, initial_queries
            )
            prospective_state = _CompositeTrackerState(
                calibration=calibration,
                metadata=public_metadata,
                reserved_fast_queries=None,
                reserved_sparse_queries=None,
                estimate=estimate,
            )
        except Exception as error:
            _raise_reset_error(
                "initial_state_construction_failed",
                f"initial composite state construction failed: {error}",
                error,
            )
        self._state = prospective_state

    def choose_next_query(self) -> TwoPointQuery | SparseLinewidthQuery | None:
        """Issue the pending query or reserve the next indivisible block."""
        if self._state is None:
            raise RuntimeError("tracker must be reset before choosing a query")
        state = self._state
        if state.estimate.stopped_reason is not None:
            return None
        if state.estimate.pending_query is not None:
            return state.estimate.pending_query

        if state.estimate.incomplete_fast_pair is not None:
            if state.reserved_fast_queries is None:
                raise RuntimeError("incomplete fast pair lacks its reserved block")
            second_query = state.reserved_fast_queries[1]
            estimate = _replace_estimate(
                state.estimate,
                pending_mode="fast_pair",
                pending_query=second_query,
            )
            self._state = _CompositeTrackerState(
                calibration=state.calibration,
                metadata=state.metadata,
                reserved_fast_queries=state.reserved_fast_queries,
                reserved_sparse_queries=None,
                estimate=estimate,
            )
            return second_query
        if state.estimate.incomplete_sparse_scan is not None:
            if state.reserved_sparse_queries is None:
                raise RuntimeError("incomplete sparse scan lacks its reserved block")
            point_index = len(state.estimate.incomplete_sparse_scan.queries)
            if point_index >= len(state.reserved_sparse_queries):
                raise RuntimeError("incomplete sparse scan exhausted its reservation")
            query = state.reserved_sparse_queries[point_index]
            estimate = _replace_estimate(
                state.estimate,
                pending_mode="sparse_scan",
                pending_query=query,
            )
            self._state = _CompositeTrackerState(
                calibration=state.calibration,
                metadata=state.metadata,
                reserved_fast_queries=None,
                reserved_sparse_queries=state.reserved_sparse_queries,
                estimate=estimate,
            )
            return query

        if (
            state.estimate.fast_pairs_since_scan
            == self._configuration.scan_period_fast_pairs
        ):
            scan_index = state.estimate.completed_sparse_scans
            identity_index = scan_index % len(state.estimate.identities)
            identity = state.estimate.identities[identity_index]
            identity_scan_index = identity.completed_sparse_scans
            try:
                geometry = _construct_sparse_fit_geometry(
                    state.calibration,
                    self._configuration,
                    identity,
                    scan_index=scan_index,
                    identity_scan_index=identity_scan_index,
                )
            except _SparseGeometryConstructionError as error:
                diagnostic = _construct_geometry_diagnostic(
                    state,
                    identity,
                    scan_index=scan_index,
                    identity_scan_index=identity_scan_index,
                    error=error,
                )
                stopped = _replace_estimate(
                    state.estimate,
                    stopped_reason="sparse_geometry_unavailable",
                    sparse_geometry_diagnostic=diagnostic,
                )
                self._state = _CompositeTrackerState(
                    calibration=state.calibration,
                    metadata=state.metadata,
                    reserved_fast_queries=None,
                    reserved_sparse_queries=None,
                    estimate=stopped,
                )
                return None

            first_sequence_index = (
                0
                if state.estimate.current_sequence_index is None
                else state.estimate.current_sequence_index + 1
            )
            queries = _construct_sparse_queries(
                geometry,
                identity,
                state.metadata,
                integration_time_s=self._configuration.integration_time_s,
                first_acquisition_index=state.estimate.accepted_observations,
                first_sequence_index=first_sequence_index,
                start_timestamp_s=state.estimate.current_timestamp_s,
                scan_index=scan_index,
                identity_scan_index=identity_scan_index,
            )
            after_block = _reserve_five_atom_block(
                state.estimate.charged_resources, state.metadata, queries
            )
            if not _within_ceiling(after_block, state.estimate.budget_ceiling):
                stopped = _replace_estimate(
                    state.estimate,
                    stopped_reason="budget_exhausted",
                    sparse_geometry_diagnostic=None,
                )
                self._state = _CompositeTrackerState(
                    calibration=state.calibration,
                    metadata=state.metadata,
                    reserved_fast_queries=None,
                    reserved_sparse_queries=None,
                    estimate=stopped,
                )
                return None

            first_query = queries[0]
            estimate = _replace_estimate(
                state.estimate,
                pending_mode="sparse_scan",
                pending_query=first_query,
            )
            self._state = _CompositeTrackerState(
                calibration=state.calibration,
                metadata=state.metadata,
                reserved_fast_queries=None,
                reserved_sparse_queries=queries,
                estimate=estimate,
            )
            return first_query

        queries = _construct_fast_queries(
            state.calibration, state.metadata, state.estimate
        )
        after_block = _reserve_two_atom_block(
            state.estimate.charged_resources, state.metadata, queries
        )
        if not _within_ceiling(after_block, state.estimate.budget_ceiling):
            stopped = _replace_estimate(
                state.estimate,
                stopped_reason="budget_exhausted",
                sparse_geometry_diagnostic=None,
            )
            self._state = _CompositeTrackerState(
                calibration=state.calibration,
                metadata=state.metadata,
                reserved_fast_queries=None,
                reserved_sparse_queries=None,
                estimate=stopped,
            )
            return None

        first_query = queries[0]
        estimate = _replace_estimate(
            state.estimate,
            pending_mode="fast_pair",
            pending_query=first_query,
        )
        self._state = _CompositeTrackerState(
            calibration=state.calibration,
            metadata=state.metadata,
            reserved_fast_queries=queries,
            reserved_sparse_queries=None,
            estimate=estimate,
        )
        return first_query

    def update(
        self, observation: EstimatorObservation
    ) -> SparseLinewidthCompositeUpdate:
        """Accept one estimator-safe observation for the pending query."""
        if type(observation) is not EstimatorObservation:
            raise SparseLinewidthObservationValidationError(
                "invalid_observation_type",
                "observation must be an exact EstimatorObservation",
            )
        if self._state is None or self._state.estimate.pending_query is None:
            raise SparseLinewidthObservationValidationError(
                "no_pending_query", "tracker has no pending query"
            )
        state = self._state
        query = state.estimate.pending_query
        mode = state.estimate.pending_mode
        if (
            (mode == "fast_pair" and type(query) is not TwoPointQuery)
            or (mode == "sparse_scan" and type(query) is not SparseLinewidthQuery)
            or mode not in {"fast_pair", "sparse_scan"}
        ):
            raise SparseLinewidthObservationValidationError(
                "pending_mode_mismatch",
                "pending mode must agree with the exact pending query type",
            )
        if mode == "fast_pair":
            partial = state.estimate.incomplete_fast_pair
            expected_query = (
                None
                if state.reserved_fast_queries is None
                else state.reserved_fast_queries[1 if partial is not None else 0]
            )
            echo_matches = (
                expected_query is query
                and state.reserved_sparse_queries is None
                and state.estimate.incomplete_sparse_scan is None
            )
            if partial is None:
                echo_matches = echo_matches and query.pair_index == (
                    state.estimate.completed_fast_pairs
                )
            else:
                echo_matches = echo_matches and (
                    partial.first_query is state.reserved_fast_queries[0]
                    and query.pair_index == partial.pair_index
                    and query.identity_pair_index == partial.identity_pair_index
                    and query.resonance_id == partial.resonance_id
                    and query.interrogation_center_hz
                    == partial.interrogation_center_hz
                    and query.side != partial.first_side
                )
            if not echo_matches:
                raise SparseLinewidthObservationValidationError(
                    "fast_query_echo_mismatch",
                    "pending fast query must echo its exact reserved pair state",
                )
        else:
            partial_sparse = state.estimate.incomplete_sparse_scan
            expected_point_index = (
                0 if partial_sparse is None else len(partial_sparse.queries)
            )
            reserved_sparse_queries = state.reserved_sparse_queries
            expected_query = (
                None
                if reserved_sparse_queries is None
                or expected_point_index >= len(reserved_sparse_queries)
                else reserved_sparse_queries[expected_point_index]
            )
            echo_matches = (
                expected_query is query
                and state.reserved_fast_queries is None
                and state.estimate.incomplete_fast_pair is None
                and query.point_index == expected_point_index
            )
            if partial_sparse is not None:
                partial_queries_match = (
                    reserved_sparse_queries is not None
                    and len(partial_sparse.queries) < len(reserved_sparse_queries)
                    and all(
                        partial_query is reserved_sparse_queries[index]
                        for index, partial_query in enumerate(
                            partial_sparse.queries
                        )
                    )
                )
                echo_matches = echo_matches and partial_queries_match and (
                    query.scan_index == partial_sparse.scan_index
                    and query.identity_scan_index
                    == partial_sparse.identity_scan_index
                    and query.resonance_id == partial_sparse.resonance_id
                )
            if not echo_matches:
                raise SparseLinewidthObservationValidationError(
                    "sparse_query_echo_mismatch",
                    "pending sparse query must echo its exact reserved scan state",
                )

        if observation.sequence_index != query.expected_sequence_index:
            raise SparseLinewidthObservationValidationError(
                "sequence_mismatch",
                "observation sequence index must equal the pending query",
            )
        if observation.frequency_hz != query.frequency_hz:
            raise SparseLinewidthObservationValidationError(
                "frequency_mismatch",
                "observation frequency must equal the pending query",
            )
        if observation.integration_time_s != query.integration_time_s:
            raise SparseLinewidthObservationValidationError(
                "integration_time_mismatch",
                "observation integration time must equal the pending query",
            )
        if observation.timestamp_s != query.expected_end_timestamp_s:
            raise SparseLinewidthObservationValidationError(
                "endpoint_mismatch",
                "observation endpoint must equal the pending query",
            )
        if (
            observation.nominal_exposure_photons
            != query.expected_nominal_exposure_photons
        ):
            raise SparseLinewidthObservationValidationError(
                "nominal_exposure_mismatch",
                "observation nominal exposure must equal the pending query",
            )
        if (
            type(observation.fluorescence) is not float
            or not math.isfinite(observation.fluorescence)
            or (
                observation.realized_photons is not None
                and (
                    type(observation.realized_photons) is not int
                    or observation.realized_photons < 0
                )
            )
        ):
            raise SparseLinewidthObservationValidationError(
                "invalid_observation_value",
                "observation values must preserve their constructor invariants",
            )
        if mode == "sparse_scan":
            assert type(query) is SparseLinewidthQuery
            if query.point_index == 4:
                prior_partial = state.estimate.incomplete_sparse_scan
                assert prior_partial is not None
                complete_queries = (*prior_partial.queries, query)
                complete_observations = (*prior_partial.observations, observation)
                try:
                    started_ns = time.process_time_ns()
                    completed_scan = fit_sparse_linewidth(
                        state.calibration.source,
                        self._configuration,
                        complete_queries,
                        complete_observations,
                    )
                    if type(completed_scan) is not SparseLinewidthScanResult:
                        raise TypeError(
                            "fit_sparse_linewidth must return an exact "
                            "SparseLinewidthScanResult"
                        )
                except Exception as error:
                    _raise_update_error(
                        "sparse_scan_result_construction_failed",
                        f"sparse scan result construction failed: {error}",
                        error,
                    )

                target_index = state.estimate.completed_sparse_scans % len(
                    state.estimate.identities
                )
                try:
                    active_fwhm_values = tuple(
                        (
                            completed_scan.fitted_fwhm_hz
                            if index == target_index
                            and completed_scan.status == "success"
                            else identity.active_fwhm_hz
                        )
                        for index, identity in enumerate(
                            state.estimate.identities
                        )
                    )
                    assert all(value is not None for value in active_fwhm_values)
                    live_q_values = tuple(
                        identity.fast_center_hz / active_fwhm_hz
                        for identity, active_fwhm_hz in zip(
                            state.estimate.identities,
                            active_fwhm_values,
                            strict=True,
                        )
                    )
                    if not all(math.isfinite(value) for value in live_q_values):
                        raise ValueError(
                            "live Q must remain finite after sparse completion"
                        )
                except Exception as error:
                    _raise_update_error(
                        "aggregate_estimate_construction_failed",
                        f"aggregate estimate construction failed: {error}",
                        error,
                    )

                try:
                    identities = tuple(
                        _identity_after_sparse_observation(
                            identity,
                            observation,
                            live_q=live_q_values[index],
                            completed_scan=(
                                completed_scan if index == target_index else None
                            ),
                        )
                        for index, identity in enumerate(
                            state.estimate.identities
                        )
                    )
                except Exception as error:
                    _raise_update_error(
                        "sparse_identity_estimate_construction_failed",
                        f"sparse identity construction failed: {error}",
                        error,
                    )

                try:
                    sparse_resources = _advance_observation_resources(
                        state.estimate.sparse_tracking_resources,
                        observation,
                        state.metadata,
                    )
                    tracking_resources = _advance_observation_resources(
                        state.estimate.tracking_resources,
                        observation,
                        state.metadata,
                    )
                    charged_resources = _advance_observation_resources(
                        state.estimate.charged_resources,
                        observation,
                        state.metadata,
                    )
                except Exception as error:
                    _raise_update_error(
                        "resource_construction_failed",
                        f"resource construction failed: {error}",
                        error,
                    )

                try:
                    finished_ns = time.process_time_ns()
                    update_cpu_time_s = (
                        finished_ns - started_ns
                    ) / 1_000_000_000.0
                    if not (
                        math.isfinite(update_cpu_time_s)
                        and update_cpu_time_s >= 0.0
                    ):
                        raise ValueError(
                            "sparse update CPU time must be finite and nonnegative"
                        )
                    sparse_history = (
                        *state.estimate.sparse_scan_history,
                        completed_scan,
                    )
                    estimate = _replace_estimate(
                        state.estimate,
                        identities=identities,
                        pending_mode=None,
                        pending_query=None,
                        incomplete_sparse_scan=None,
                        sparse_scan_history=sparse_history,
                        accepted_observations=(
                            state.estimate.accepted_observations + 1
                        ),
                        completed_sparse_scans=len(sparse_history),
                        fast_pairs_since_scan=0,
                        current_sequence_index=observation.sequence_index,
                        current_timestamp_s=observation.timestamp_s,
                        sparse_tracking_resources=sparse_resources,
                        tracking_resources=tracking_resources,
                        charged_resources=charged_resources,
                        sparse_update_cpu_time_s=(
                            state.estimate.sparse_update_cpu_time_s
                            + update_cpu_time_s
                        ),
                        total_update_cpu_time_s=(
                            state.estimate.total_update_cpu_time_s
                            + update_cpu_time_s
                        ),
                    )
                    metadata = TwoPointRunMetadata(
                        tracker_clock_id=state.metadata.tracker_clock_id,
                        current_sequence_index=observation.sequence_index,
                        current_timestamp_s=observation.timestamp_s,
                        nominal_photon_rate_hz=(
                            state.metadata.nominal_photon_rate_hz
                        ),
                        frequency_overhead_s=state.metadata.frequency_overhead_s,
                        fluorescence_quantity=state.metadata.fluorescence_quantity,
                    )
                except Exception as error:
                    _raise_update_error(
                        "aggregate_estimate_construction_failed",
                        f"aggregate estimate construction failed: {error}",
                        error,
                    )

                try:
                    update = SparseLinewidthCompositeUpdate(
                        query=query,
                        observation=observation,
                        completed_fast_pair=None,
                        completed_sparse_scan=completed_scan,
                        estimate=estimate,
                        update_cpu_time_s=update_cpu_time_s,
                    )
                    prospective_state = _CompositeTrackerState(
                        calibration=state.calibration,
                        metadata=metadata,
                        reserved_fast_queries=None,
                        reserved_sparse_queries=None,
                        estimate=estimate,
                    )
                except Exception as error:
                    _raise_update_error(
                        "update_construction_failed",
                        f"composite update construction failed: {error}",
                        error,
                    )
                self._state = prospective_state
                return update

            prior_partial = state.estimate.incomplete_sparse_scan
            try:
                started_ns = time.process_time_ns()
                partial_queries = (
                    (query,)
                    if prior_partial is None
                    else (*prior_partial.queries, query)
                )
                partial_observations = (
                    (observation,)
                    if prior_partial is None
                    else (*prior_partial.observations, observation)
                )
                first_query = partial_queries[0]
                partial_scan = SparsePartialScan(
                    scan_index=first_query.scan_index,
                    identity_scan_index=first_query.identity_scan_index,
                    resonance_id=first_query.resonance_id,
                    frozen_fast_center_hz=first_query.frozen_fast_center_hz,
                    frozen_fast_center_source_kind=(
                        first_query.frozen_fast_center_source_kind
                    ),
                    frozen_fast_center_source_pair_index=(
                        first_query.frozen_fast_center_source_pair_index
                    ),
                    frozen_fast_center_reference_timestamp_s=(
                        first_query.frozen_fast_center_reference_timestamp_s
                    ),
                    frozen_fast_center_release_sequence_index=(
                        first_query.frozen_fast_center_release_sequence_index
                    ),
                    frozen_fast_center_release_timestamp_s=(
                        first_query.frozen_fast_center_release_timestamp_s
                    ),
                    frozen_prior_fwhm_hz=first_query.frozen_prior_fwhm_hz,
                    frozen_fwhm_source_kind=first_query.frozen_fwhm_source_kind,
                    frozen_fwhm_source_scan_index=(
                        first_query.frozen_fwhm_source_scan_index
                    ),
                    frozen_fwhm_reference_timestamp_s=(
                        first_query.frozen_fwhm_reference_timestamp_s
                    ),
                    frozen_fwhm_release_sequence_index=(
                        first_query.frozen_fwhm_release_sequence_index
                    ),
                    frozen_fwhm_release_timestamp_s=(
                        first_query.frozen_fwhm_release_timestamp_s
                    ),
                    queries=partial_queries,
                    observations=partial_observations,
                )
            except Exception as error:
                _raise_update_error(
                    "sparse_partial_scan_construction_failed",
                    f"sparse partial scan construction failed: {error}",
                    error,
                )

            try:
                sparse_resources = _advance_observation_resources(
                    state.estimate.sparse_tracking_resources,
                    observation,
                    state.metadata,
                )
                tracking_resources = _advance_observation_resources(
                    state.estimate.tracking_resources,
                    observation,
                    state.metadata,
                )
                charged_resources = _advance_observation_resources(
                    state.estimate.charged_resources,
                    observation,
                    state.metadata,
                )
            except Exception as error:
                _raise_update_error(
                    "resource_construction_failed",
                    f"resource construction failed: {error}",
                    error,
                )

            try:
                finished_ns = time.process_time_ns()
                update_cpu_time_s = (finished_ns - started_ns) / 1_000_000_000.0
                identities = tuple(
                    _identity_after_fast_observation(identity, observation)
                    for identity in state.estimate.identities
                )
                estimate = _replace_estimate(
                    state.estimate,
                    identities=identities,
                    pending_mode=None,
                    pending_query=None,
                    incomplete_sparse_scan=partial_scan,
                    accepted_observations=(
                        state.estimate.accepted_observations + 1
                    ),
                    current_sequence_index=observation.sequence_index,
                    current_timestamp_s=observation.timestamp_s,
                    sparse_tracking_resources=sparse_resources,
                    tracking_resources=tracking_resources,
                    charged_resources=charged_resources,
                    sparse_update_cpu_time_s=(
                        state.estimate.sparse_update_cpu_time_s
                        + update_cpu_time_s
                    ),
                    total_update_cpu_time_s=(
                        state.estimate.total_update_cpu_time_s
                        + update_cpu_time_s
                    ),
                )
                metadata = TwoPointRunMetadata(
                    tracker_clock_id=state.metadata.tracker_clock_id,
                    current_sequence_index=observation.sequence_index,
                    current_timestamp_s=observation.timestamp_s,
                    nominal_photon_rate_hz=state.metadata.nominal_photon_rate_hz,
                    frequency_overhead_s=state.metadata.frequency_overhead_s,
                    fluorescence_quantity=state.metadata.fluorescence_quantity,
                )
            except Exception as error:
                _raise_update_error(
                    "aggregate_estimate_construction_failed",
                    f"aggregate estimate construction failed: {error}",
                    error,
                )

            try:
                update = SparseLinewidthCompositeUpdate(
                    query=query,
                    observation=observation,
                    completed_fast_pair=None,
                    completed_sparse_scan=None,
                    estimate=estimate,
                    update_cpu_time_s=update_cpu_time_s,
                )
                prospective_state = _CompositeTrackerState(
                    calibration=state.calibration,
                    metadata=metadata,
                    reserved_fast_queries=None,
                    reserved_sparse_queries=state.reserved_sparse_queries,
                    estimate=estimate,
                )
            except Exception as error:
                _raise_update_error(
                    "update_construction_failed",
                    f"composite update construction failed: {error}",
                    error,
                )
            self._state = prospective_state
            return update

        assert type(query) is TwoPointQuery
        first_side = state.estimate.incomplete_fast_pair is None
        completed_pair = None
        partial_pair = None
        identities: tuple[CompositeIdentityEstimate, ...] | None = None
        try:
            started_ns = time.process_time_ns()
            if first_side:
                partial_pair = TwoPointPartialPair(
                    pair_index=query.pair_index,
                    identity_pair_index=query.identity_pair_index,
                    resonance_id=query.resonance_id,
                    interrogation_center_hz=query.interrogation_center_hz,
                    first_side=query.side,
                    first_query=query,
                    first_observation=observation,
                )
            else:
                completed_pair = _construct_fast_pair_result(
                    state, query, observation
                )
        except Exception as error:
            code = (
                "fast_partial_pair_construction_failed"
                if first_side
                else "fast_pair_result_construction_failed"
            )
            _raise_update_error(code, f"fast pair construction failed: {error}", error)

        if not first_side:
            assert completed_pair is not None
            target_index = state.estimate.completed_fast_pairs % len(
                state.estimate.identities
            )
            try:
                identities = tuple(
                    _identity_after_fast_observation(
                        identity,
                        observation,
                        completed_pair=(
                            completed_pair if index == target_index else None
                        ),
                    )
                    for index, identity in enumerate(state.estimate.identities)
                )
            except Exception as error:
                _raise_update_error(
                    "fast_identity_estimate_construction_failed",
                    f"fast identity construction failed: {error}",
                    error,
                )

        try:
            fast_resources = _advance_observation_resources(
                state.estimate.fast_tracking_resources,
                observation,
                state.metadata,
            )
            tracking_resources = _advance_observation_resources(
                state.estimate.tracking_resources,
                observation,
                state.metadata,
            )
            charged_resources = _advance_observation_resources(
                state.estimate.charged_resources,
                observation,
                state.metadata,
            )
        except Exception as error:
            _raise_update_error(
                "resource_construction_failed",
                f"resource construction failed: {error}",
                error,
            )

        try:
            finished_ns = time.process_time_ns()
            update_cpu_time_s = (finished_ns - started_ns) / 1_000_000_000.0
            if first_side:
                identities = tuple(
                    _identity_after_fast_observation(identity, observation)
                    for identity in state.estimate.identities
                )
            assert identities is not None
            pair_history = (
                state.estimate.fast_pair_history
                if completed_pair is None
                else (*state.estimate.fast_pair_history, completed_pair)
            )
            estimate = _replace_estimate(
                state.estimate,
                identities=identities,
                pending_mode=None,
                pending_query=None,
                incomplete_fast_pair=partial_pair,
                fast_pair_history=pair_history,
                accepted_observations=state.estimate.accepted_observations + 1,
                completed_fast_pairs=len(pair_history),
                fast_pairs_since_scan=(
                    state.estimate.fast_pairs_since_scan
                    + int(completed_pair is not None)
                ),
                current_sequence_index=observation.sequence_index,
                current_timestamp_s=observation.timestamp_s,
                fast_tracking_resources=fast_resources,
                tracking_resources=tracking_resources,
                charged_resources=charged_resources,
                fast_update_cpu_time_s=(
                    state.estimate.fast_update_cpu_time_s + update_cpu_time_s
                ),
                total_update_cpu_time_s=(
                    state.estimate.total_update_cpu_time_s + update_cpu_time_s
                ),
            )
            metadata = TwoPointRunMetadata(
                tracker_clock_id=state.metadata.tracker_clock_id,
                current_sequence_index=observation.sequence_index,
                current_timestamp_s=observation.timestamp_s,
                nominal_photon_rate_hz=state.metadata.nominal_photon_rate_hz,
                frequency_overhead_s=state.metadata.frequency_overhead_s,
                fluorescence_quantity=state.metadata.fluorescence_quantity,
            )
        except Exception as error:
            _raise_update_error(
                "aggregate_estimate_construction_failed",
                f"aggregate estimate construction failed: {error}",
                error,
            )

        try:
            update = SparseLinewidthCompositeUpdate(
                query=query,
                observation=observation,
                completed_fast_pair=completed_pair,
                completed_sparse_scan=None,
                estimate=estimate,
                update_cpu_time_s=update_cpu_time_s,
            )
            prospective_state = _CompositeTrackerState(
                calibration=state.calibration,
                metadata=metadata,
                reserved_fast_queries=(
                    state.reserved_fast_queries if first_side else None
                ),
                reserved_sparse_queries=None,
                estimate=estimate,
            )
        except Exception as error:
            _raise_update_error(
                "update_construction_failed",
                f"composite update construction failed: {error}",
                error,
            )
        self._state = prospective_state
        return update

    def estimate(self) -> SparseLinewidthCompositeEstimate:
        """Return the current immutable composite estimate."""
        if self._state is None:
            raise RuntimeError("tracker must be reset before estimating")
        return self._state.estimate


__all__ = ["SparseLinewidthCompositeTracker"]
