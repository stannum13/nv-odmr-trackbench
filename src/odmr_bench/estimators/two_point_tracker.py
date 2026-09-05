"""Causal calibrated two-point center tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral
from typing import final

import numpy as np

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators.two_point_resources import _zero_public_resources
from odmr_bench.estimators.two_point_types import (
    PairSide,
    PublicAcquisitionResources,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointEstimate,
    TwoPointIdentityEstimate,
    TwoPointObservationValidationError,
    TwoPointPairResult,
    TwoPointPartialPair,
    TwoPointQuery,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    TwoPointUpdate,
    TwoPointUpdateConstructionError,
)


@dataclass(frozen=True, slots=True)
class _TrackerState:
    calibration: TwoPointCalibration
    metadata: TwoPointRunMetadata
    budget_ceiling: TwoPointBudgetCeiling
    pending_query: TwoPointQuery | None
    pair_history: tuple[TwoPointPairResult, ...]
    estimate: TwoPointEstimate


def _advance_query_charge(
    resources: PublicAcquisitionResources,
    metadata: TwoPointRunMetadata,
    configuration: TwoPointTrackerConfiguration,
) -> PublicAcquisitionResources:
    integration_time_s = configuration.integration_time_s
    nominal_exposure_photons = (
        metadata.nominal_photon_rate_hz * integration_time_s
    )
    virtual_elapsed_time_s = metadata.frequency_overhead_s + integration_time_s
    return PublicAcquisitionResources(
        observations=resources.observations + 1,
        integration_time_s=resources.integration_time_s + integration_time_s,
        nominal_exposure_photons=(
            resources.nominal_exposure_photons + nominal_exposure_photons
        ),
        realized_photons=resources.realized_photons,
        observations_without_realized_counts=(
            resources.observations_without_realized_counts
        ),
        virtual_elapsed_time_s=(
            resources.virtual_elapsed_time_s + virtual_elapsed_time_s
        ),
    )


def _within_ceiling(
    resources: PublicAcquisitionResources,
    ceiling: TwoPointBudgetCeiling,
) -> bool:
    return all(
        limit is None or current <= limit
        for current, limit in (
            (resources.observations, ceiling.max_observations),
            (resources.integration_time_s, ceiling.max_integration_time_s),
            (
                resources.nominal_exposure_photons,
                ceiling.max_nominal_exposure_photons,
            ),
            (resources.virtual_elapsed_time_s, ceiling.max_virtual_elapsed_time_s),
        )
    )


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


def _build_pending_query(
    *,
    query_index: int,
    pair_index: int,
    identity_pair_index: int,
    resonance_id: str,
    side: PairSide,
    interrogation_center_hz: float,
    offset_hz: float,
    metadata: TwoPointRunMetadata,
    configuration: TwoPointTrackerConfiguration,
    estimate: TwoPointEstimate,
) -> TwoPointQuery:
    integration_time_s = configuration.integration_time_s
    expected_sequence_index = (
        0
        if estimate.current_sequence_index is None
        else estimate.current_sequence_index + 1
    )
    expected_end_timestamp_s = (
        estimate.current_timestamp_s + metadata.frequency_overhead_s
    ) + integration_time_s
    expected_nominal_exposure_photons = (
        metadata.nominal_photon_rate_hz * integration_time_s
    )
    frequency_hz = (
        interrogation_center_hz - offset_hz
        if side == "minus"
        else interrogation_center_hz + offset_hz
    )
    return TwoPointQuery(
        query_index=query_index,
        pair_index=pair_index,
        identity_pair_index=identity_pair_index,
        resonance_id=resonance_id,
        side=side,
        interrogation_center_hz=interrogation_center_hz,
        frequency_hz=frequency_hz,
        integration_time_s=integration_time_s,
        expected_sequence_index=expected_sequence_index,
        expected_end_timestamp_s=expected_end_timestamp_s,
        expected_nominal_exposure_photons=expected_nominal_exposure_photons,
    )


def _prospective_first_pair(
    calibration: TwoPointCalibration,
    metadata: TwoPointRunMetadata,
    configuration: TwoPointTrackerConfiguration,
    estimate: TwoPointEstimate,
) -> tuple[TwoPointQuery, PublicAcquisitionResources]:
    integration_time_s = configuration.integration_time_s
    nominal_exposure_photons = (
        metadata.nominal_photon_rate_hz * integration_time_s
    )
    if not math.isfinite(nominal_exposure_photons):
        raise ValueError("first-query nominal exposure must remain finite")
    elapsed_atom_s = metadata.frequency_overhead_s + integration_time_s
    if not math.isfinite(elapsed_atom_s):
        raise ValueError("first-query elapsed charge must remain finite")
    expected_end_timestamp_s = (
        estimate.current_timestamp_s + metadata.frequency_overhead_s
    ) + integration_time_s
    if not math.isfinite(expected_end_timestamp_s):
        raise ValueError("first-query endpoint must remain finite")
    if expected_end_timestamp_s <= estimate.current_timestamp_s:
        raise ValueError("first-query endpoint must strictly advance")
    second_end_timestamp_s = (
        expected_end_timestamp_s + metadata.frequency_overhead_s
    ) + integration_time_s
    if not math.isfinite(second_end_timestamp_s):
        raise ValueError("second-query endpoint must remain finite")
    if second_end_timestamp_s <= expected_end_timestamp_s:
        raise ValueError("second-query endpoint must strictly advance")

    target = estimate.identities[0]
    cell = calibration.identities[0]
    query = _build_pending_query(
        query_index=0,
        pair_index=0,
        identity_pair_index=0,
        resonance_id=target.resonance_id,
        side="minus",
        interrogation_center_hz=target.center_hz,
        offset_hz=cell.offset_hz,
        metadata=metadata,
        configuration=configuration,
        estimate=estimate,
    )
    try:
        after_first = _advance_query_charge(
            estimate.charged_resources,
            metadata,
            configuration,
        )
        after_second = _advance_query_charge(
            after_first,
            metadata,
            configuration,
        )
    except ValueError as error:
        raise ValueError(
            f"first-pair charged resources must remain representable: {error}"
        ) from error
    return query, after_second


@final
class CalibratedTwoPointTracker:
    """Track calibrated resonance centers with adjacent flank pairs."""

    __slots__ = ("_configuration", "_state")

    def __init__(self, configuration: TwoPointTrackerConfiguration) -> None:
        if type(configuration) is not TwoPointTrackerConfiguration:
            raise TypeError(
                "configuration must be an exact TwoPointTrackerConfiguration"
            )
        self._configuration = configuration
        self._state: _TrackerState | None = None

    @property
    def configuration(self) -> TwoPointTrackerConfiguration:
        """Return the immutable tracker configuration."""
        return self._configuration

    @property
    def calibration(self) -> TwoPointCalibration | None:
        """Return the active calibration, if the tracker has been reset."""
        return None if self._state is None else self._state.calibration

    @property
    def pending_query(self) -> TwoPointQuery | None:
        """Return the currently issued query, if any."""
        return None if self._state is None else self._state.pending_query

    @property
    def pair_history(self) -> tuple[TwoPointPairResult, ...]:
        """Return the immutable history of completed pairs."""
        return () if self._state is None else self._state.pair_history

    def reset(
        self,
        public_metadata: TwoPointRunMetadata,
        calibration: TwoPointCalibration,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> None:
        """Reset tracking from one authenticated public calibration boundary."""
        if type(public_metadata) is not TwoPointRunMetadata:
            raise TypeError("public_metadata must be an exact TwoPointRunMetadata")
        if type(calibration) is not TwoPointCalibration:
            raise TypeError("calibration must be an exact TwoPointCalibration")
        if type(budget_ceiling) is not TwoPointBudgetCeiling:
            raise TypeError("budget_ceiling must be an exact TwoPointBudgetCeiling")
        if calibration.configuration != self._configuration:
            raise ValueError(
                "calibration configuration must equal tracker configuration"
            )

        source = calibration.source
        mapping = source.clock_mapping
        if public_metadata.tracker_clock_id != mapping.tracker_clock_id:
            raise ValueError("metadata tracker clock must equal the mapping target")
        if (
            calibration.budget_treatment == "included_same_run"
            and mapping.kind != "shared_clock"
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

        charged_resources = (
            source.safe_resources
            if calibration.budget_treatment == "included_same_run"
            else _zero_public_resources()
        )
        capped_values = (
            (charged_resources.observations, budget_ceiling.max_observations),
            (
                charged_resources.integration_time_s,
                budget_ceiling.max_integration_time_s,
            ),
            (
                charged_resources.nominal_exposure_photons,
                budget_ceiling.max_nominal_exposure_photons,
            ),
            (
                charged_resources.virtual_elapsed_time_s,
                budget_ceiling.max_virtual_elapsed_time_s,
            ),
        )
        if any(
            limit is not None and current > limit
            for current, limit in capped_values
        ):
            raise ValueError(
                "budget ceiling cannot be below charged starting resources"
            )
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (Integral, np.integer)
        ):
            raise TypeError("seed must be an integer")
        canonical_seed = int(seed)
        if canonical_seed < 0:
            raise ValueError("seed must be nonnegative")

        release_sequence_index = (
            source.availability_sequence_index
            if calibration.budget_treatment == "included_same_run"
            else None
        )
        estimate_age_sequence_indices = (
            public_metadata.current_sequence_index - source.availability_sequence_index
            if calibration.budget_treatment == "included_same_run"
            else None
        )
        identities = tuple(
            TwoPointIdentityEstimate(
                resonance_id=cell.resonance_id,
                center_hz=cell.calibration_center_hz,
                calibration_fwhm_hz=cell.calibration_fwhm_hz,
                calibration_cell_lower_hz=cell.calibration_cell_lower_hz,
                calibration_cell_upper_hz=cell.calibration_cell_upper_hz,
                allowed_center_min_hz=cell.allowed_center_min_hz,
                allowed_center_max_hz=cell.allowed_center_max_hz,
                active_source_kind="calibration",
                active_source_pair_index=None,
                active_reference_timestamp_s=mapped_physical_epoch_s,
                active_release_sequence_index=release_sequence_index,
                active_release_timestamp_s=mapped_availability_s,
                estimate_age_sequence_indices=estimate_age_sequence_indices,
                estimate_age_s=(
                    public_metadata.current_timestamp_s - mapped_physical_epoch_s
                ),
                release_age_s=(
                    public_metadata.current_timestamp_s - mapped_availability_s
                ),
                completed_pairs=0,
                lock_state="calibrated",
                failure_code=None,
                latest_pair=None,
            )
            for cell in calibration.identities
        )
        estimate = TwoPointEstimate(
            identities=identities,
            calibration_source_id=source.source_id,
            calibration_source_provenance=source.provenance,
            calibration_budget_treatment=calibration.budget_treatment,
            current_sequence_index=public_metadata.current_sequence_index,
            current_timestamp_s=public_metadata.current_timestamp_s,
            accepted_observations=0,
            completed_pairs=0,
            incomplete_pair=None,
            pending_query=None,
            pair_history=(),
            tracking_resources=_zero_public_resources(),
            calibration_resources=source.safe_resources,
            charged_resources=charged_resources,
            budget_ceiling=budget_ceiling,
            stopped_reason=None,
            seed=canonical_seed,
        )
        _prospective_first_pair(
            calibration,
            public_metadata,
            self._configuration,
            estimate,
        )
        self._state = _TrackerState(
            calibration=calibration,
            metadata=public_metadata,
            budget_ceiling=budget_ceiling,
            pending_query=None,
            pair_history=(),
            estimate=estimate,
        )

    def choose_next_query(self) -> TwoPointQuery | None:
        """Issue or repeat the next causal query."""
        if self._state is None:
            raise RuntimeError("tracker must be reset before choosing a query")
        state = self._state
        if state.estimate.stopped_reason is not None:
            return None
        if state.pending_query is not None:
            return state.pending_query

        partial_pair = state.estimate.incomplete_pair
        if partial_pair is not None:
            cell = state.calibration.identities[
                state.estimate.completed_pairs % len(state.calibration.identities)
            ]
            query = _build_pending_query(
                query_index=state.estimate.accepted_observations,
                pair_index=partial_pair.pair_index,
                identity_pair_index=partial_pair.identity_pair_index,
                resonance_id=partial_pair.resonance_id,
                side="plus",
                interrogation_center_hz=partial_pair.interrogation_center_hz,
                offset_hz=cell.offset_hz,
                metadata=state.metadata,
                configuration=self._configuration,
                estimate=state.estimate,
            )
            estimate = replace(state.estimate, pending_query=query)
            self._state = replace(state, pending_query=query, estimate=estimate)
            return query

        query, after_second = _prospective_first_pair(
            state.calibration,
            state.metadata,
            self._configuration,
            state.estimate,
        )
        if not _within_ceiling(after_second, state.budget_ceiling):
            estimate = replace(state.estimate, stopped_reason="budget_exhausted")
            self._state = replace(state, estimate=estimate)
            return None

        estimate = replace(state.estimate, pending_query=query)
        self._state = replace(state, pending_query=query, estimate=estimate)
        return query

    def update(self, observation: EstimatorObservation) -> TwoPointUpdate:
        """Accept one estimator-safe observation for the pending query."""
        if type(observation) is not EstimatorObservation:
            raise TwoPointObservationValidationError(
                "invalid_observation_type",
                "observation must be an exact EstimatorObservation",
            )
        if self._state is None or self._state.pending_query is None:
            raise TwoPointObservationValidationError(
                "no_pending_query", "tracker has no pending query"
            )
        state = self._state
        query = state.pending_query
        if observation.sequence_index != query.expected_sequence_index:
            raise TwoPointObservationValidationError(
                "sequence_mismatch",
                "observation sequence index must equal the pending query",
            )
        if observation.frequency_hz != query.frequency_hz:
            raise TwoPointObservationValidationError(
                "frequency_mismatch",
                "observation frequency must equal the pending query",
            )
        if observation.integration_time_s != self._configuration.integration_time_s:
            raise TwoPointObservationValidationError(
                "integration_time_mismatch",
                "observation integration time must equal the tracker configuration",
            )
        expected_endpoint_s = (
            state.estimate.current_timestamp_s + state.metadata.frequency_overhead_s
        ) + self._configuration.integration_time_s
        if observation.timestamp_s != expected_endpoint_s:
            raise TwoPointObservationValidationError(
                "endpoint_mismatch",
                "observation endpoint must equal the expected causal endpoint",
            )
        expected_nominal_exposure_photons = (
            state.metadata.nominal_photon_rate_hz
            * self._configuration.integration_time_s
        )
        if (
            observation.nominal_exposure_photons
            != expected_nominal_exposure_photons
        ):
            raise TwoPointObservationValidationError(
                "nominal_exposure_mismatch",
                "observation nominal exposure must equal rate times integration",
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
            raise TwoPointObservationValidationError(
                "invalid_observation_value",
                "observation values must preserve their constructor invariants",
            )

        try:
            partial_pair = TwoPointPartialPair(
                pair_index=query.pair_index,
                identity_pair_index=query.identity_pair_index,
                resonance_id=query.resonance_id,
                interrogation_center_hz=query.interrogation_center_hz,
                first_side=query.side,
                first_query=query,
                first_observation=observation,
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
            identities = tuple(
                replace(
                    identity,
                    estimate_age_sequence_indices=(
                        None
                        if identity.active_release_sequence_index is None
                        else observation.sequence_index
                        - identity.active_release_sequence_index
                    ),
                    estimate_age_s=(
                        observation.timestamp_s
                        - identity.active_reference_timestamp_s
                    ),
                    release_age_s=(
                        observation.timestamp_s
                        - identity.active_release_timestamp_s
                    ),
                )
                for identity in state.estimate.identities
            )
            estimate = TwoPointEstimate(
                identities=identities,
                calibration_source_id=state.estimate.calibration_source_id,
                calibration_source_provenance=(
                    state.estimate.calibration_source_provenance
                ),
                calibration_budget_treatment=(
                    state.estimate.calibration_budget_treatment
                ),
                current_sequence_index=observation.sequence_index,
                current_timestamp_s=observation.timestamp_s,
                accepted_observations=state.estimate.accepted_observations + 1,
                completed_pairs=state.estimate.completed_pairs,
                incomplete_pair=partial_pair,
                pending_query=None,
                pair_history=state.estimate.pair_history,
                tracking_resources=tracking_resources,
                calibration_resources=state.estimate.calibration_resources,
                charged_resources=charged_resources,
                budget_ceiling=state.estimate.budget_ceiling,
                stopped_reason=state.estimate.stopped_reason,
                seed=state.estimate.seed,
            )
            metadata = replace(
                state.metadata,
                current_sequence_index=observation.sequence_index,
                current_timestamp_s=observation.timestamp_s,
            )
            prospective_state = replace(
                state,
                metadata=metadata,
                pending_query=None,
                estimate=estimate,
            )
            update = TwoPointUpdate(query, observation, None, estimate)
        except Exception as error:
            raise TwoPointUpdateConstructionError(
                "partial_pair_construction_failed",
                f"first-side construction failed: {error}",
            ) from error
        self._state = prospective_state
        return update

    def estimate(self) -> TwoPointEstimate:
        """Return the current immutable aggregate estimate."""
        if self._state is None:
            raise RuntimeError("tracker must be reset before estimating")
        return self._state.estimate


__all__ = ["CalibratedTwoPointTracker"]
