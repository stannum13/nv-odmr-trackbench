"""Causal calibrated two-point center tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral
from typing import final

import numpy as np

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators.two_point_calibration import (
    _evaluate_target_only_model,
    _target_center_derivative,
)
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
from odmr_bench.estimators.types import SpectrumFitResult


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


def _pair_model_geometry(
    source_fit: SpectrumFitResult,
    source_fit_index: int,
    minus_frequency_hz: float,
    plus_frequency_hz: float,
    center_hz: float,
) -> tuple[float, float, float] | None:
    """Return finite pair-model sum, zero and slope, or a scientific failure."""
    mu_minus = _evaluate_target_only_model(
        source_fit,
        source_fit_index,
        minus_frequency_hz,
        center_hz,
    )
    mu_plus = _evaluate_target_only_model(
        source_fit,
        source_fit_index,
        plus_frequency_hz,
        center_hz,
    )
    g_minus = _target_center_derivative(
        source_fit,
        source_fit_index,
        minus_frequency_hz,
        center_hz,
    )
    g_plus = _target_center_derivative(
        source_fit,
        source_fit_index,
        plus_frequency_hz,
        center_hz,
    )
    model_sum = mu_minus + mu_plus
    if (
        not all(
            math.isfinite(value)
            for value in (mu_minus, mu_plus, g_minus, g_plus, model_sum)
        )
        or model_sum == 0.0
    ):
        return None
    zero_discriminator = (mu_minus - mu_plus) / model_sum
    discriminator_slope_per_hz = 2.0 * (
        mu_plus * g_minus - mu_minus * g_plus
    ) / model_sum**2
    if (
        not math.isfinite(zero_discriminator)
        or not math.isfinite(discriminator_slope_per_hz)
        or discriminator_slope_per_hz <= 0.0
    ):
        return None
    return model_sum, zero_discriminator, discriminator_slope_per_hz


def _identity_after_observation(
    identity: TwoPointIdentityEstimate,
    observation: EstimatorObservation,
    *,
    completed_pair: TwoPointPairResult | None = None,
) -> TwoPointIdentityEstimate:
    successful_pair = completed_pair is not None and completed_pair.lock_state in {
        "tracking",
        "step_limited",
    }
    center_hz = (
        completed_pair.candidate_center_hz
        if successful_pair and completed_pair is not None
        else identity.center_hz
    )
    active_source_kind = "pair" if successful_pair else identity.active_source_kind
    active_source_pair_index = (
        completed_pair.pair_index
        if successful_pair and completed_pair is not None
        else identity.active_source_pair_index
    )
    active_reference_timestamp_s = (
        completed_pair.pair_reference_timestamp_s
        if successful_pair and completed_pair is not None
        else identity.active_reference_timestamp_s
    )
    active_release_sequence_index = (
        completed_pair.release_sequence_index
        if successful_pair and completed_pair is not None
        else identity.active_release_sequence_index
    )
    active_release_timestamp_s = (
        completed_pair.release_timestamp_s
        if successful_pair and completed_pair is not None
        else identity.active_release_timestamp_s
    )
    estimate_age_sequence_indices = (
        None
        if active_release_sequence_index is None
        else observation.sequence_index - active_release_sequence_index
    )
    return TwoPointIdentityEstimate(
        resonance_id=identity.resonance_id,
        center_hz=center_hz,
        calibration_fwhm_hz=identity.calibration_fwhm_hz,
        calibration_cell_lower_hz=identity.calibration_cell_lower_hz,
        calibration_cell_upper_hz=identity.calibration_cell_upper_hz,
        allowed_center_min_hz=identity.allowed_center_min_hz,
        allowed_center_max_hz=identity.allowed_center_max_hz,
        active_source_kind=active_source_kind,
        active_source_pair_index=active_source_pair_index,
        active_reference_timestamp_s=active_reference_timestamp_s,
        active_release_sequence_index=active_release_sequence_index,
        active_release_timestamp_s=active_release_timestamp_s,
        estimate_age_sequence_indices=estimate_age_sequence_indices,
        estimate_age_s=observation.timestamp_s - active_reference_timestamp_s,
        release_age_s=observation.timestamp_s - active_release_timestamp_s,
        completed_pairs=(
            identity.completed_pairs + int(completed_pair is not None)
        ),
        lock_state=(
            completed_pair.lock_state
            if completed_pair is not None
            else identity.lock_state
        ),
        failure_code=(
            completed_pair.failure_code
            if completed_pair is not None
            else identity.failure_code
        ),
        latest_pair=(
            completed_pair if completed_pair is not None else identity.latest_pair
        ),
    )


def _estimate_after_observation(
    previous: TwoPointEstimate,
    observation: EstimatorObservation,
    *,
    identities: tuple[TwoPointIdentityEstimate, ...],
    incomplete_pair: TwoPointPartialPair | None,
    pair_history: tuple[TwoPointPairResult, ...],
    tracking_resources: PublicAcquisitionResources,
    charged_resources: PublicAcquisitionResources,
) -> TwoPointEstimate:
    return TwoPointEstimate(
        identities=identities,
        calibration_source_id=previous.calibration_source_id,
        calibration_source_provenance=previous.calibration_source_provenance,
        calibration_budget_treatment=previous.calibration_budget_treatment,
        current_sequence_index=observation.sequence_index,
        current_timestamp_s=observation.timestamp_s,
        accepted_observations=previous.accepted_observations + 1,
        completed_pairs=len(pair_history),
        incomplete_pair=incomplete_pair,
        pending_query=None,
        pair_history=pair_history,
        tracking_resources=tracking_resources,
        calibration_resources=previous.calibration_resources,
        charged_resources=charged_resources,
        budget_ceiling=previous.budget_ceiling,
        stopped_reason=previous.stopped_reason,
        seed=previous.seed,
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

    pair_index = estimate.completed_pairs
    identity_index = pair_index % len(estimate.identities)
    target = estimate.identities[identity_index]
    cell = calibration.identities[identity_index]
    identity_pair_index = target.completed_pairs
    first_side: PairSide = (
        "minus" if identity_pair_index % 2 == 0 else "plus"
    )
    query = _build_pending_query(
        query_index=estimate.accepted_observations,
        pair_index=pair_index,
        identity_pair_index=identity_pair_index,
        resonance_id=target.resonance_id,
        side=first_side,
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
                side=("plus" if partial_pair.first_side == "minus" else "minus"),
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

        if state.estimate.incomplete_pair is not None:
            partial_pair = state.estimate.incomplete_pair
            try:
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

                cell_index = state.estimate.completed_pairs % len(
                    state.calibration.identities
                )
                cell = state.calibration.identities[cell_index]
                source_fit = state.calibration.source.source_fit
                center_hz = partial_pair.interrogation_center_hz
                observed_sum = (
                    minus_observation.fluorescence
                    + plus_observation.fluorescence
                )
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
                        (
                            model_sum,
                            zero_discriminator,
                            discriminator_slope_per_hz,
                        ) = model_geometry
                        computed_discriminator = (
                            minus_observation.fluorescence
                            - plus_observation.fluorescence
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
                                        self._configuration.proportional_gain
                                        * computed_raw_innovation_hz
                                    )
                                    if math.isfinite(computed_requested_step_hz):
                                        requested_step_hz = computed_requested_step_hz
                                        computed_applied_step_hz = max(
                                            -cell.max_step_hz,
                                            min(
                                                computed_requested_step_hz,
                                                cell.max_step_hz,
                                            ),
                                        )
                                        computed_candidate_center_hz = (
                                            center_hz + computed_applied_step_hz
                                        )
                                        if math.isfinite(
                                            computed_candidate_center_hz
                                        ):
                                            candidate_center_hz = (
                                                computed_candidate_center_hz
                                            )
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
                    common_limit = self._configuration.common_mode_limit_target_depths
                    if (
                        common_limit is not None
                        and abs(common_mode_target_depths) > common_limit
                    ):
                        raw_innovation_hz = None
                        requested_step_hz = None
                        candidate_center_hz = None
                        failure_code = "common_mode_limit_exceeded"
                    elif abs(raw_innovation_hz) > cell.capture_radius_hz:
                        requested_step_hz = None
                        candidate_center_hz = None
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
                    first_observation.timestamp_s
                    - first_observation.integration_time_s / 2.0
                )
                second_reference_s = (
                    observation.timestamp_s - observation.integration_time_s / 2.0
                )
                pair_reference_timestamp_s = first_reference_s + (
                    second_reference_s - first_reference_s
                ) / 2.0
                pair_result = TwoPointPairResult(
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
            except Exception as error:
                raise TwoPointUpdateConstructionError(
                    "pair_result_construction_failed",
                    f"pair-result construction failed: {error}",
                ) from error
            pair_history = (*state.pair_history, pair_result)

            try:
                identities = tuple(
                    _identity_after_observation(
                        identity,
                        observation,
                        completed_pair=(pair_result if index == cell_index else None),
                    )
                    for index, identity in enumerate(state.estimate.identities)
                )
            except Exception as error:
                raise TwoPointUpdateConstructionError(
                    "identity_estimate_construction_failed",
                    f"identity-estimate construction failed: {error}",
                ) from error

            try:
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
                raise TwoPointUpdateConstructionError(
                    "resource_construction_failed",
                    f"resource construction failed: {error}",
                ) from error

            try:
                estimate = _estimate_after_observation(
                    state.estimate,
                    observation,
                    identities=identities,
                    incomplete_pair=None,
                    pair_history=pair_history,
                    tracking_resources=tracking_resources,
                    charged_resources=charged_resources,
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
                    pair_history=pair_history,
                    estimate=estimate,
                )
                update = TwoPointUpdate(query, observation, pair_result, estimate)
            except Exception as error:
                raise TwoPointUpdateConstructionError(
                    "aggregate_estimate_construction_failed",
                    f"aggregate-estimate construction failed: {error}",
                ) from error
            self._state = prospective_state
            return update

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
                _identity_after_observation(identity, observation)
                for identity in state.estimate.identities
            )
            estimate = _estimate_after_observation(
                state.estimate,
                observation,
                identities=identities,
                incomplete_pair=partial_pair,
                pair_history=state.estimate.pair_history,
                tracking_resources=tracking_resources,
                charged_resources=charged_resources,
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
