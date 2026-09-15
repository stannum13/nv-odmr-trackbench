"""Instrument-owning state machine for calibrated two-point evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Literal

import numpy as np

from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.observations import InstrumentObservation
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators.two_point_tracker import CalibratedTwoPointTracker
from odmr_bench.estimators.two_point_types import (
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointCalibrationSource,
    TwoPointClockMapping,
    TwoPointEstimate,
    TwoPointIdentityBinding,
    TwoPointObservationValidationError,
    TwoPointQuery,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    TwoPointUpdateConstructionError,
)
from odmr_bench.estimators.types import FitConfiguration
from odmr_bench.evaluation.two_point.provenance import (
    _TOKEN_CONSTRUCTION_KEY,
    _lookup_run_token_binding,
    _mint_verified_instrument_run_token,
    _register_run_token,
    _rollback_run_token_registration,
    _RunTokenBinding,
)
from odmr_bench.evaluation.two_point.resource_accounting import (
    _advance_full_resources,
    _resource_mismatch_fields,
    _zero_full_resources,
)
from odmr_bench.evaluation.two_point.types import (
    TwoPointAbortedRun,
    TwoPointAbortReason,
    TwoPointCalibrationPreflightError,
    TwoPointEvaluatorInstrumentConfiguration,
    TwoPointEvaluatorPairTiming,
    TwoPointEvaluatorRunnerState,
    TwoPointInstrumentQueryFailure,
    TwoPointResourceJoinUnavailableAcquisition,
    TwoPointRunnerAborted,
    TwoPointRunnerAccepted,
    TwoPointRunnerBudgetStopped,
    TwoPointRunnerExternallyStopped,
    TwoPointRunnerInstrumentFailure,
    TwoPointRunnerRunOutcome,
    TwoPointRunnerStartError,
    TwoPointRunnerStateError,
    TwoPointRunnerStepOutcome,
    TwoPointTrackingAcquisition,
    VerifiedTwoPointCalibrationOutcome,
    VerifiedTwoPointCalibrationSuccess,
)


@dataclass(frozen=True, slots=True)
class _StartTrackingPlan:
    state_before: TwoPointEvaluatorRunnerState
    tracking_resources_before: ResourceSnapshot


class TwoPointEvaluatorRunner:
    """Own one instrument association and its immutable audit state."""

    __slots__ = ("_instrument", "_state", "_tracker")

    @classmethod
    def bind(cls, instrument: ODMRInstrument) -> TwoPointEvaluatorRunner:
        """Bind a clean exact instrument and register a fresh run token."""
        if type(instrument) is not ODMRInstrument:
            raise TwoPointCalibrationPreflightError("invalid_argument_type")
        try:
            nominal_photon_rate_hz = instrument.nominal_photon_rate_hz
            frequency_overhead_s = instrument.frequency_overhead_s
            resources = instrument.resources
            current_virtual_time_s = instrument.virtual_time_s
        except Exception as error:
            raise TwoPointCalibrationPreflightError("invalid_argument_value") from error
        if (
            type(nominal_photon_rate_hz) is not float
            or type(frequency_overhead_s) is not float
            or type(resources) is not ResourceSnapshot
            or type(current_virtual_time_s) is not float
        ):
            raise TwoPointCalibrationPreflightError("invalid_argument_type")
        try:
            instrument_configuration = TwoPointEvaluatorInstrumentConfiguration(
                nominal_photon_rate_hz=nominal_photon_rate_hz,
                frequency_overhead_s=frequency_overhead_s,
            )
            resources = ResourceSnapshot(
                resources.observations,
                resources.integration_time_s,
                resources.nominal_exposure_photons,
                resources.expected_photons,
                resources.realized_photons,
                resources.observations_without_realized_counts,
                resources.virtual_elapsed_time_s,
            )
            if (
                not math.isfinite(current_virtual_time_s)
                or current_virtual_time_s < 0.0
            ):
                raise ValueError("current virtual time must be finite and nonnegative")
        except TypeError as error:
            raise TwoPointCalibrationPreflightError("invalid_argument_type") from error
        except ValueError as error:
            raise TwoPointCalibrationPreflightError("invalid_argument_value") from error
        if resources != _zero_full_resources() or current_virtual_time_s != 0.0:
            raise TwoPointCalibrationPreflightError("unclean_instrument_boundary")

        token = _mint_verified_instrument_run_token(_TOKEN_CONSTRUCTION_KEY)
        runner = object.__new__(cls)
        object.__setattr__(runner, "_instrument", instrument)
        object.__setattr__(runner, "_tracker", None)
        object.__setattr__(
            runner,
            "_state",
            TwoPointEvaluatorRunnerState(
                phase="ready",
                run_token=token,
                instrument_configuration=instrument_configuration,
                calibration_outcome=None,
                verified_calibration=None,
                calibration=None,
                tracker_estimate=None,
                normal_tracking_trace=(),
                pair_timings=(),
                instrument_resources_at_bind=resources,
                tracking_resources_before=None,
                instrument_resources_current=resources,
                instrument_current_sequence_index=None,
                current_virtual_time_s=current_virtual_time_s,
                last_instrument_failure=None,
                terminal_abort=None,
            ),
        )
        try:
            _register_run_token(
                token,
                runner,
                instrument,
                instrument_configuration,
            )
        except BaseException:
            _rollback_run_token_registration(
                token,
            )
            raise
        return runner

    @property
    def state(self) -> TwoPointEvaluatorRunnerState:
        """Return the current frozen audit snapshot."""
        return self._state

    def acquire_verified_calibration(
        self,
        frequency_hz: Sequence[float],
        integration_time_s: float,
        fit_configuration: FitConfiguration,
        identity_binding: TwoPointIdentityBinding,
        *,
        source_id: str,
        source_clock_id: str,
        tracker_clock_id: str,
        source_to_tracker_offset_s: float,
        physical_fit_epoch_rule: Literal["instrument_midpoint_ordered_mean"],
    ) -> VerifiedTwoPointCalibrationOutcome:
        """Acquire one lossless verified sweep or typed causal failure."""
        from odmr_bench.evaluation.two_point.calibration import (
            _acquire_verified_calibration,
        )

        return _acquire_verified_calibration(
            self,
            frequency_hz,
            integration_time_s,
            fit_configuration,
            identity_binding,
            source_id=source_id,
            source_clock_id=source_clock_id,
            tracker_clock_id=tracker_clock_id,
            source_to_tracker_offset_s=source_to_tracker_offset_s,
            physical_fit_epoch_rule=physical_fit_epoch_rule,
        )

    def start_tracking(
        self,
        tracker: CalibratedTwoPointTracker,
        calibration: TwoPointCalibration,
        verified_calibration: VerifiedTwoPointCalibrationSuccess,
        public_metadata: TwoPointRunMetadata,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> TwoPointEvaluatorRunnerState:
        """Join a verified calibration to tracking without issuing a query."""
        plan = _preflight_start_tracking(
            self,
            tracker,
            calibration,
            verified_calibration,
            public_metadata,
            budget_ceiling,
            seed=seed,
        )
        state_before = plan.state_before
        resources_before = plan.tracking_resources_before

        tracker_configuration_before = tracker._configuration
        tracker_state_before = tracker._state
        try:
            tracker.reset(
                public_metadata,
                calibration,
                budget_ceiling,
                seed=seed,
            )
            tracker_estimate = tracker.estimate()
            state_after = replace(
                state_before,
                phase="tracking",
                verified_calibration=verified_calibration,
                calibration=calibration,
                tracker_estimate=tracker_estimate,
                tracking_resources_before=resources_before,
                instrument_resources_current=resources_before,
            )
        except BaseException as error:
            object.__setattr__(
                tracker,
                "_configuration",
                tracker_configuration_before,
            )
            object.__setattr__(tracker, "_state", tracker_state_before)
            if isinstance(error, Exception):
                raise TwoPointRunnerStartError("tracker_reset_failed") from error
            raise
        object.__setattr__(self, "_tracker", tracker)
        object.__setattr__(self, "_state", state_after)
        return state_after

    def step(self) -> TwoPointRunnerStepOutcome:
        """Acquire and accept one pending two-point observation."""
        state_before = self._state
        tracker = self._tracker
        if state_before.phase != "tracking" or tracker is None:
            raise TwoPointRunnerStateError(
                "step requires a runner in the tracking phase"
            )

        tracker_configuration_before_query = tracker._configuration
        tracker_state_before_query = tracker._state
        try:
            query = tracker.choose_next_query()
            if query is None:
                from odmr_bench.evaluation.two_point.resource_accounting import (
                    build_two_point_evaluator_resources,
                )

                tracker_estimate = tracker.estimate()
                state_after = replace(
                    state_before,
                    phase="budget_stopped",
                    tracker_estimate=tracker_estimate,
                )
                object.__setattr__(self, "_state", state_after)
                resources = build_two_point_evaluator_resources(self)
                if resources is None:
                    raise RuntimeError(
                        "budget stop requires available evaluator resources"
                    )
                return TwoPointRunnerBudgetStopped(
                    kind="budget_stopped",
                    resources=resources,
                    state=state_after,
                )
            tracker_estimate_before = tracker.estimate()
            if tracker_estimate_before.pending_query != query:
                raise RuntimeError("tracker estimate must retain the issued query")

            resources_before = self._instrument.resources
            virtual_time_before = self._instrument.virtual_time_s
            expected_midpoint_s = (
                virtual_time_before
                + state_before.instrument_configuration.frequency_overhead_s
            ) + query.integration_time_s / 2.0
        except BaseException:
            object.__setattr__(self, "_state", state_before)
            _restore_tracker_slots(
                tracker,
                tracker_configuration_before_query,
                tracker_state_before_query,
            )
            raise

        try:
            full_observation = self._instrument.query(
                query.frequency_hz,
                query.integration_time_s,
            )
        except Exception as error:
            try:
                resources_after = self._instrument.resources
                virtual_time_after = self._instrument.virtual_time_s
                from odmr_bench.evaluation.two_point.calibration import (
                    _safe_exception_strings,
                )

                exception_type, exception_message = _safe_exception_strings(error)
                failure = TwoPointInstrumentQueryFailure(
                    query=query,
                    exception_type=exception_type,
                    exception_message=exception_message,
                    instrument_resources_before=resources_before,
                    instrument_resources_after=resources_after,
                )
                state_after = replace(
                    state_before,
                    tracker_estimate=tracker_estimate_before,
                    instrument_resources_current=resources_after,
                    current_virtual_time_s=virtual_time_after,
                    last_instrument_failure=failure,
                )
                outcome = TwoPointRunnerInstrumentFailure(
                    kind="instrument_failure",
                    failure=failure,
                    state=state_after,
                )
            except BaseException:
                _restore_tracker_slots(
                    tracker,
                    tracker_configuration_before_query,
                    tracker_state_before_query,
                )
                raise
            object.__setattr__(self, "_state", state_after)
            return outcome
        except BaseException:
            _restore_tracker_slots(
                tracker,
                tracker_configuration_before_query,
                tracker_state_before_query,
            )
            raise

        tracker_configuration_before_update = tracker._configuration
        tracker_state_before_update = tracker._state
        try:
            resources_after = self._instrument.resources
            virtual_time_after = self._instrument.virtual_time_s
            acquisition = _build_authenticated_tracking_acquisition(
                query=query,
                expected_midpoint_s=expected_midpoint_s,
                full_observation=full_observation,
                resources_before=resources_before,
                resources_after=resources_after,
                virtual_time_after=virtual_time_after,
                overhead_s=(state_before.instrument_configuration.frequency_overhead_s),
            )
            if type(acquisition) is TwoPointResourceJoinUnavailableAcquisition:
                return _finish_aborted_step(
                    self,
                    state_before=state_before,
                    acquisition=acquisition,
                    reason="resource_join_unavailable",
                    exception_type=None,
                    exception_message=None,
                    tracker_estimate_before=tracker_estimate_before,
                    tracker_estimate_after=tracker_estimate_before,
                    resources_after=resources_after,
                    virtual_time_after=virtual_time_after,
                )
            if acquisition.measurement_midpoint_s is None:
                error = TwoPointObservationValidationError(
                    "endpoint_mismatch",
                    "instrument endpoint or live clock does not match "
                    "the pending query",
                )
                from odmr_bench.evaluation.two_point.calibration import (
                    _safe_exception_strings,
                )

                exception_type, exception_message = _safe_exception_strings(error)
                return _finish_aborted_step(
                    self,
                    state_before=state_before,
                    acquisition=acquisition,
                    reason="tracker_observation_validation_error",
                    exception_type=exception_type,
                    exception_message=exception_message,
                    tracker_estimate_before=tracker_estimate_before,
                    tracker_estimate_after=tracker_estimate_before,
                    resources_after=resources_after,
                    virtual_time_after=virtual_time_after,
                )
            try:
                update = tracker.update(acquisition.safe_observation)
            except Exception as error:
                _restore_tracker_slots(
                    tracker,
                    tracker_configuration_before_update,
                    tracker_state_before_update,
                )
                from odmr_bench.evaluation.two_point.calibration import (
                    _safe_exception_strings,
                )

                if isinstance(error, TwoPointObservationValidationError):
                    reason = "tracker_observation_validation_error"
                elif isinstance(error, TwoPointUpdateConstructionError):
                    reason = "tracker_update_construction_error"
                else:
                    reason = "tracker_update_unexpected_error"
                exception_type, exception_message = _safe_exception_strings(error)
                tracker_estimate_after = tracker.estimate()
                return _finish_aborted_step(
                    self,
                    state_before=state_before,
                    acquisition=acquisition,
                    reason=reason,
                    exception_type=exception_type,
                    exception_message=exception_message,
                    tracker_estimate_before=tracker_estimate_before,
                    tracker_estimate_after=tracker_estimate_after,
                    resources_after=resources_after,
                    virtual_time_after=virtual_time_after,
                )

            pair_timings = state_before.pair_timings
            if update.completed_pair is not None:
                first_acquisition = state_before.normal_tracking_trace[-1]
                first_midpoint_s = first_acquisition.measurement_midpoint_s
                second_midpoint_s = acquisition.measurement_midpoint_s
                if first_midpoint_s is None or second_midpoint_s is None:
                    raise RuntimeError("accepted pair must retain both midpoints")
                pair = update.completed_pair
                truth_reference_s = (
                    first_midpoint_s + (second_midpoint_s - first_midpoint_s) / 2.0
                )
                pair_timing = TwoPointEvaluatorPairTiming(
                    pair_index=pair.pair_index,
                    resonance_id=pair.resonance_id,
                    first_measurement_midpoint_s=first_midpoint_s,
                    second_measurement_midpoint_s=second_midpoint_s,
                    truth_reference_timestamp_s=truth_reference_s,
                    public_reference_timestamp_s=(pair.pair_reference_timestamp_s),
                    release_sequence_index=pair.release_sequence_index,
                    release_timestamp_s=pair.release_timestamp_s,
                )
                pair_timings = (*pair_timings, pair_timing)

            state_after = replace(
                state_before,
                tracker_estimate=update.estimate,
                normal_tracking_trace=(
                    *state_before.normal_tracking_trace,
                    acquisition,
                ),
                pair_timings=pair_timings,
                instrument_resources_current=resources_after,
                instrument_current_sequence_index=(full_observation.sequence_index),
                current_virtual_time_s=virtual_time_after,
                last_instrument_failure=None,
            )
            outcome = TwoPointRunnerAccepted(
                kind="accepted",
                acquisition=acquisition,
                update=update,
                state=state_after,
            )
        except BaseException:
            object.__setattr__(self, "_state", state_before)
            _restore_tracker_slots(
                tracker,
                tracker_configuration_before_update,
                tracker_state_before_update,
            )
            raise
        object.__setattr__(self, "_state", state_after)
        return outcome

    def stop_external(self) -> TwoPointRunnerExternallyStopped:
        """Stop tracking at the current causal boundary without acquisition."""
        state_before = self._state
        if state_before.phase != "tracking" or self._tracker is None:
            raise TwoPointRunnerStateError(
                "stop_external requires a runner in the tracking phase"
            )
        from odmr_bench.evaluation.two_point.resource_accounting import (
            build_two_point_evaluator_resources,
        )

        state_after = replace(state_before, phase="externally_stopped")
        object.__setattr__(self, "_state", state_after)
        try:
            resources = build_two_point_evaluator_resources(self)
            if resources is None:
                raise RuntimeError(
                    "external stop requires available evaluator resources"
                )
            outcome = TwoPointRunnerExternallyStopped(
                kind="externally_stopped",
                resources=resources,
                state=state_after,
            )
        except BaseException:
            object.__setattr__(self, "_state", state_before)
            raise
        return outcome

    def run_until_event(self) -> TwoPointRunnerRunOutcome:
        """Advance through accepted observations to the first non-accept event."""
        if self._state.phase != "tracking" or self._tracker is None:
            raise TwoPointRunnerStateError(
                "run_until_event requires a runner in the tracking phase"
            )
        while True:
            outcome = self.step()
            if type(outcome) is not TwoPointRunnerAccepted:
                return outcome


def _finish_aborted_step(
    runner: TwoPointEvaluatorRunner,
    *,
    state_before: TwoPointEvaluatorRunnerState,
    acquisition: (
        TwoPointTrackingAcquisition | TwoPointResourceJoinUnavailableAcquisition
    ),
    reason: TwoPointAbortReason,
    exception_type: str | None,
    exception_message: str | None,
    tracker_estimate_before: TwoPointEstimate,
    tracker_estimate_after: TwoPointEstimate,
    resources_after: ResourceSnapshot,
    virtual_time_after: float,
) -> TwoPointRunnerAborted:
    """Construct one terminal abort and its optional resource aggregate."""
    abort = TwoPointAbortedRun(
        reason=reason,
        exception_type=exception_type,
        exception_message=exception_message,
        unaccepted_acquisition=acquisition,
        unaccepted_observation_count=1,
        tracker_estimate_before=tracker_estimate_before,
        tracker_estimate_after=tracker_estimate_after,
    )
    state_after = replace(
        state_before,
        phase="aborted",
        tracker_estimate=tracker_estimate_after,
        instrument_resources_current=resources_after,
        instrument_current_sequence_index=acquisition.full_observation.sequence_index,
        current_virtual_time_s=virtual_time_after,
        last_instrument_failure=None,
        terminal_abort=abort,
    )
    object.__setattr__(runner, "_state", state_after)
    if type(acquisition) is TwoPointResourceJoinUnavailableAcquisition:
        from odmr_bench.evaluation.two_point.resource_accounting import (
            build_two_point_evaluator_resources,
        )

        resources = build_two_point_evaluator_resources(runner)
        if resources is not None:
            raise RuntimeError("unavailable abort must not fabricate resources")
    else:
        from odmr_bench.evaluation.two_point.resource_accounting import (
            _build_two_point_evaluator_resources_from_context,
        )

        resources = _build_two_point_evaluator_resources_from_context(
            runner,
            authenticated_unaccepted=acquisition,
        )
    return TwoPointRunnerAborted(
        kind="aborted",
        abort=abort,
        resources=resources,
        state=state_after,
    )


def _build_tracking_acquisition(
    *,
    query: TwoPointQuery,
    expected_midpoint_s: float,
    full_observation: InstrumentObservation,
    resources_before: ResourceSnapshot,
    resources_after: ResourceSnapshot,
    virtual_time_after: float,
    overhead_s: float,
) -> TwoPointTrackingAcquisition | TwoPointResourceJoinUnavailableAcquisition:
    safe_observation = full_observation.estimator_view()
    resource_delta = _advance_full_resources(
        _zero_full_resources(),
        full_observation,
        overhead_s,
    )
    expected_resources_after = _advance_full_resources(
        resources_before,
        full_observation,
        overhead_s,
    )
    resource_mismatch_fields = _resource_mismatch_fields(
        expected_resources_after,
        resources_after,
    )
    timing_matches = (
        full_observation.integration_time_s == query.integration_time_s
        and full_observation.timestamp_s == query.expected_end_timestamp_s
        and virtual_time_after == query.expected_end_timestamp_s
    )
    if resource_mismatch_fields:
        return TwoPointResourceJoinUnavailableAcquisition(
            resource_join_status="unavailable",
            query=query,
            expected_measurement_midpoint_s=expected_midpoint_s,
            measurement_midpoint_s=(expected_midpoint_s if timing_matches else None),
            full_observation=full_observation,
            safe_observation=safe_observation,
            resource_mismatch_fields=resource_mismatch_fields,
            instrument_resources_before=resources_before,
            instrument_resources_after=resources_after,
        )
    return TwoPointTrackingAcquisition(
        resource_join_status="authenticated",
        query=query,
        expected_measurement_midpoint_s=expected_midpoint_s,
        measurement_midpoint_s=(expected_midpoint_s if timing_matches else None),
        full_observation=full_observation,
        safe_observation=safe_observation,
        instrument_resources_before=resources_before,
        instrument_resources_after=resources_after,
        instrument_resource_delta=resource_delta,
    )


def _build_authenticated_tracking_acquisition(
    *,
    query: TwoPointQuery,
    expected_midpoint_s: float,
    full_observation: InstrumentObservation,
    resources_before: ResourceSnapshot,
    resources_after: ResourceSnapshot,
    overhead_s: float,
    virtual_time_after: float | None = None,
) -> TwoPointTrackingAcquisition | TwoPointResourceJoinUnavailableAcquisition:
    """Retain the Task 16 private authenticated-construction seam."""
    return _build_tracking_acquisition(
        query=query,
        expected_midpoint_s=expected_midpoint_s,
        full_observation=full_observation,
        resources_before=resources_before,
        resources_after=resources_after,
        virtual_time_after=(
            full_observation.timestamp_s
            if virtual_time_after is None
            else virtual_time_after
        ),
        overhead_s=overhead_s,
    )


def _restore_tracker_slots(
    tracker: CalibratedTwoPointTracker,
    configuration: object,
    state: object,
) -> None:
    object.__setattr__(tracker, "_configuration", configuration)
    object.__setattr__(tracker, "_state", state)


def _preflight_start_tracking(
    runner: TwoPointEvaluatorRunner,
    tracker: CalibratedTwoPointTracker,
    calibration: TwoPointCalibration,
    verified_calibration: VerifiedTwoPointCalibrationSuccess,
    public_metadata: TwoPointRunMetadata,
    budget_ceiling: TwoPointBudgetCeiling,
    *,
    seed: int,
) -> _StartTrackingPlan:
    try:
        state_before = runner._state
    except AttributeError:
        raise TwoPointRunnerStartError("invalid_runner_phase") from None
    if (
        type(state_before) is not TwoPointEvaluatorRunnerState
        or type(state_before.phase) is not str
        or state_before.phase not in ("ready", "calibration_succeeded")
    ):
        raise TwoPointRunnerStartError("invalid_runner_phase")
    if (
        type(tracker) is not CalibratedTwoPointTracker
        or type(calibration) is not TwoPointCalibration
        or type(verified_calibration) is not VerifiedTwoPointCalibrationSuccess
        or type(public_metadata) is not TwoPointRunMetadata
        or type(budget_ceiling) is not TwoPointBudgetCeiling
        or isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (Integral, np.integer))
    ):
        raise TwoPointRunnerStartError("invalid_argument_type")

    source, binding = _authenticate_verified_calibration(verified_calibration)
    try:
        calibration_matches = (
            type(calibration.source) is TwoPointCalibrationSource
            and calibration.source is source
            and type(calibration.configuration) is TwoPointTrackerConfiguration
            and calibration.configuration == tracker.configuration
        )
    except Exception:
        calibration_matches = False
    if not calibration_matches:
        raise TwoPointRunnerStartError("calibration_mismatch")

    same_runner_success = _validate_run_provenance(
        runner,
        state_before,
        binding,
        calibration,
        verified_calibration,
    )
    _validate_start_metadata(
        state_before,
        source,
        calibration,
        public_metadata,
    )
    resources_before = _capture_tracking_boundary(
        runner,
        state_before,
        verified_calibration,
        same_runner_success=same_runner_success,
    )
    return _StartTrackingPlan(state_before, resources_before)


def _authenticate_verified_calibration(
    verified_calibration: VerifiedTwoPointCalibrationSuccess,
) -> tuple[TwoPointCalibrationSource, _RunTokenBinding]:
    source = verified_calibration.source
    binding = _lookup_run_token_binding(verified_calibration.run_token)
    if (
        type(verified_calibration.status) is not str
        or verified_calibration.status != "success"
        or type(source) is not TwoPointCalibrationSource
        or type(source.provenance) is not str
        or source.provenance != "verified_factory_acquisition"
        or binding is None
        or binding.success is not verified_calibration
        or binding.source is not source
    ):
        raise TwoPointRunnerStartError("unverified_calibration")
    return source, binding


def _validate_run_provenance(
    runner: TwoPointEvaluatorRunner,
    state_before: TwoPointEvaluatorRunnerState,
    binding: _RunTokenBinding,
    calibration: TwoPointCalibration,
    verified_calibration: VerifiedTwoPointCalibrationSuccess,
) -> bool:
    try:
        instrument = runner._instrument
        retained_tracker = runner._tracker
    except AttributeError:
        raise TwoPointRunnerStartError("run_provenance_mismatch") from None
    own_binding = _lookup_run_token_binding(state_before.run_token)
    own_binding_matches = (
        own_binding is not None
        and own_binding.issuer_runner is runner
        and own_binding.instrument is instrument
        and own_binding.instrument_configuration
        is state_before.instrument_configuration
        and retained_tracker is None
    )
    same_runner_success = (
        state_before.phase == "calibration_succeeded"
        and state_before.calibration_outcome is verified_calibration
        and state_before.verified_calibration is verified_calibration
        and state_before.run_token is verified_calibration.run_token
        and own_binding is binding
        and binding.issuer_runner is runner
        and binding.instrument is instrument
        and binding.instrument_configuration is state_before.instrument_configuration
    )
    other_runner_success = (
        state_before.phase == "ready"
        and binding.issuer_runner is not runner
        and own_binding is not binding
        and own_binding is not None
        and own_binding.success is None
        and own_binding.source is None
    )
    treatment = calibration.budget_treatment
    valid_treatment = type(treatment) is str and treatment in (
        "included_same_run",
        "conditional_free_precalibration",
    )
    if (
        not own_binding_matches
        or not valid_treatment
        or (
            state_before.phase == "ready"
            and (
                treatment != "conditional_free_precalibration"
                or not other_runner_success
            )
        )
        or (state_before.phase == "calibration_succeeded" and not same_runner_success)
    ):
        raise TwoPointRunnerStartError("run_provenance_mismatch")
    return same_runner_success


def _validate_start_metadata(
    state_before: TwoPointEvaluatorRunnerState,
    source: TwoPointCalibrationSource,
    calibration: TwoPointCalibration,
    public_metadata: TwoPointRunMetadata,
) -> None:
    configuration = state_before.instrument_configuration
    mapping = source.clock_mapping
    if type(mapping) is not TwoPointClockMapping:
        raise TwoPointRunnerStartError("metadata_mismatch")
    try:
        mapped_times = tuple(
            value + mapping.offset_s
            for value in (
                source.source_first_timestamp_s,
                source.source_last_timestamp_s,
                source.physical_fit_epoch_s,
                source.availability_timestamp_s,
            )
        )
        source_rate_hz = source.fluorescence_provenance.nominal_photon_rate_hz
        metadata_matches = (
            public_metadata.tracker_clock_id == mapping.tracker_clock_id
            and public_metadata.current_sequence_index
            == state_before.instrument_current_sequence_index
            and public_metadata.current_timestamp_s
            == state_before.current_virtual_time_s
            and public_metadata.nominal_photon_rate_hz
            == configuration.nominal_photon_rate_hz
            and public_metadata.frequency_overhead_s
            == configuration.frequency_overhead_s
            and all(math.isfinite(value) for value in mapped_times)
            and mapped_times[3] >= 0.0
            and mapped_times[3] <= public_metadata.current_timestamp_s
        )
        if state_before.phase == "ready":
            metadata_matches = metadata_matches and (
                public_metadata.current_sequence_index is None
                and public_metadata.current_timestamp_s == 0.0
            )
        if calibration.budget_treatment == "included_same_run":
            metadata_matches = metadata_matches and (
                mapping.kind == "shared_clock"
                and mapping.source_clock_id == mapping.tracker_clock_id
                and mapping.offset_s == 0.0
                and public_metadata.current_sequence_index
                == source.availability_sequence_index
                and public_metadata.current_timestamp_s
                == source.availability_timestamp_s
                and public_metadata.nominal_photon_rate_hz == source_rate_hz
                and public_metadata.frequency_overhead_s
                == source.source_frequency_overhead_s
            )
    except Exception:
        metadata_matches = False
    if not metadata_matches:
        raise TwoPointRunnerStartError("metadata_mismatch")


def _capture_tracking_boundary(
    runner: TwoPointEvaluatorRunner,
    state_before: TwoPointEvaluatorRunnerState,
    verified_calibration: VerifiedTwoPointCalibrationSuccess,
    *,
    same_runner_success: bool,
) -> ResourceSnapshot:
    try:
        resources_before = runner._instrument.resources
        virtual_time_before = runner._instrument.virtual_time_s
    except Exception as error:
        raise TwoPointRunnerStartError("resource_boundary_mismatch") from error
    if (
        type(resources_before) is not ResourceSnapshot
        or type(virtual_time_before) is not float
        or resources_before != state_before.instrument_resources_current
        or virtual_time_before != state_before.current_virtual_time_s
        or (
            same_runner_success
            and verified_calibration.instrument_resources_after != resources_before
        )
    ):
        raise TwoPointRunnerStartError("resource_boundary_mismatch")
    try:
        return ResourceSnapshot(
            resources_before.observations,
            resources_before.integration_time_s,
            resources_before.nominal_exposure_photons,
            resources_before.expected_photons,
            resources_before.realized_photons,
            resources_before.observations_without_realized_counts,
            resources_before.virtual_elapsed_time_s,
        )
    except (TypeError, ValueError) as error:
        raise TwoPointRunnerStartError("resource_boundary_mismatch") from error
