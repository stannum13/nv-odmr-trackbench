"""Composite scheduling shell for sparse five-point linewidth tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Integral
from typing import final

import numpy as np

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators.sparse_linewidth_fit import (
    _validate_calibration_sparse_geometry,
)
from odmr_bench.estimators.sparse_linewidth_types import (
    CompositeIdentityEstimate,
    SparseLinewidthCompositeEstimate,
    SparseLinewidthCompositeUpdate,
    SparseLinewidthConfiguration,
    SparseLinewidthQuery,
    SparseLinewidthResetError,
)
from odmr_bench.estimators.two_point_calibration import (
    _validate_two_point_calibration_source_integrity,
)
from odmr_bench.estimators.two_point_resources import _zero_public_resources
from odmr_bench.estimators.two_point_types import (
    PublicAcquisitionResources,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointCalibrationSource,
    TwoPointIdentityCalibration,
    TwoPointQuery,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
)


@dataclass(frozen=True, slots=True)
class _CompositeTrackerState:
    calibration: TwoPointCalibration
    metadata: TwoPointRunMetadata
    reserved_fast_queries: tuple[TwoPointQuery, TwoPointQuery] | None
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
        """Reserve and expose the first fast query, or retain a clean stop."""
        if self._state is None:
            raise RuntimeError("tracker must be reset before choosing a query")
        state = self._state
        if state.estimate.stopped_reason is not None:
            return None
        if state.estimate.pending_query is not None:
            return state.estimate.pending_query
        if (
            state.estimate.accepted_observations != 0
            or state.estimate.incomplete_fast_pair is not None
            or state.estimate.incomplete_sparse_scan is not None
        ):
            raise RuntimeError("post-observation scheduling is reserved for Task 8")

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
            estimate=estimate,
        )
        return first_query

    def update(
        self, observation: EstimatorObservation
    ) -> SparseLinewidthCompositeUpdate:
        """Reserve observation transitions for the next implementation task."""
        del observation
        raise NotImplementedError("composite observation updates begin in Task 8")

    def estimate(self) -> SparseLinewidthCompositeEstimate:
        """Return the current immutable composite estimate."""
        if self._state is None:
            raise RuntimeError("tracker must be reset before estimating")
        return self._state.estimate


__all__ = ["SparseLinewidthCompositeTracker"]
