"""Instrument-owning shell for sparse-linewidth evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Literal

import numpy as np

from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators.sparse_linewidth_tracker import (
    SparseLinewidthCompositeTracker,
)
from odmr_bench.estimators.sparse_linewidth_types import (
    SparseLinewidthQuery,
)
from odmr_bench.estimators.two_point_types import (
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointCalibrationSource,
    TwoPointClockMapping,
    TwoPointIdentityBinding,
    TwoPointQuery,
    TwoPointRunMetadata,
)
from odmr_bench.estimators.types import FitConfiguration
from odmr_bench.evaluation.two_point.provenance import (
    _TOKEN_CONSTRUCTION_KEY,
    _is_exact_registered_runner,
    _lookup_run_token_binding,
    _lookup_verified_calibration_issuer,
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
    TwoPointCalibrationPreflightError,
    TwoPointEvaluatorInstrumentConfiguration,
    TwoPointEvaluatorPairTiming,
    TwoPointEvaluatorRunnerState,
    VerifiedTwoPointCalibrationOutcome,
    VerifiedTwoPointCalibrationSuccess,
)

from .types import (
    SparseEvaluatorRunnerState,
    SparseInstrumentQueryFailure,
    SparseLinewidthEvaluatorScanTiming,
    SparsePreflightError,
    SparseRunnerAccepted,
    SparseRunnerExternallyStopped,
    SparseRunnerInstrumentFailure,
    SparseRunnerRunOutcome,
    SparseRunnerStateError,
    SparseRunnerStepOutcome,
    SparseStartError,
    SparseTrackingAcquisition,
)


@dataclass(frozen=True, slots=True)
class _StartTrackingPlan:
    state_before: SparseEvaluatorRunnerState
    tracking_resources_before: ResourceSnapshot


class SparseLinewidthEvaluatorRunner:
    """Own one instrument association for sparse-linewidth evaluation."""

    __slots__ = ("_instrument", "_state", "_tracker")

    @classmethod
    def bind(cls, instrument: ODMRInstrument) -> SparseLinewidthEvaluatorRunner:
        """Bind one exact clean instrument and register private authority."""
        if (
            cls is not SparseLinewidthEvaluatorRunner
            or type(instrument) is not ODMRInstrument
        ):
            raise SparsePreflightError("invalid_argument_type")
        try:
            nominal_photon_rate_hz = instrument.nominal_photon_rate_hz
            frequency_overhead_s = instrument.frequency_overhead_s
            resources = instrument.resources
            current_virtual_time_s = instrument.virtual_time_s
        except Exception as error:
            raise SparsePreflightError("invalid_argument_value") from error
        if (
            type(nominal_photon_rate_hz) is not float
            or type(frequency_overhead_s) is not float
            or type(resources) is not ResourceSnapshot
            or type(current_virtual_time_s) is not float
        ):
            raise SparsePreflightError("invalid_argument_type")
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
            raise SparsePreflightError("invalid_argument_type") from error
        except ValueError as error:
            raise SparsePreflightError("invalid_argument_value") from error
        if resources != _zero_full_resources() or current_virtual_time_s != 0.0:
            raise SparsePreflightError("unclean_instrument_boundary")

        token = _mint_verified_instrument_run_token(_TOKEN_CONSTRUCTION_KEY)
        runner = object.__new__(cls)
        object.__setattr__(runner, "_instrument", instrument)
        object.__setattr__(runner, "_tracker", None)
        object.__setattr__(
            runner,
            "_state",
            SparseEvaluatorRunnerState(
                phase="ready",
                run_token=token,
                instrument_configuration=instrument_configuration,
                calibration_outcome=None,
                verified_calibration=None,
                calibration=None,
                tracker_estimate=None,
                normal_tracking_trace=(),
                pair_timings=(),
                scan_timings=(),
                instrument_resources_at_bind=resources,
                tracking_resources_before=None,
                instrument_resources_current=resources,
                instrument_current_sequence_index=None,
                current_virtual_time_s=current_virtual_time_s,
                last_instrument_failure=None,
                terminal_abort=None,
                fast_update_cpu_time_s=0.0,
                sparse_update_cpu_time_s=0.0,
                total_update_cpu_time_s=0.0,
            ),
        )
        try:
            _register_run_token(token, runner, instrument, instrument_configuration)
        except BaseException:
            _rollback_run_token_registration(token)
            raise
        return runner

    @property
    def state(self) -> SparseEvaluatorRunnerState:
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
        """Acquire one verified calibration through the private shared issuer."""
        from odmr_bench.evaluation.two_point.calibration import (
            _acquire_verified_calibration_core,
        )

        try:
            return _acquire_verified_calibration_core(
                _lookup_verified_calibration_issuer(self),
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
        except TwoPointCalibrationPreflightError as error:
            raise SparsePreflightError(error.code) from error

    def start_tracking(
        self,
        tracker: SparseLinewidthCompositeTracker,
        calibration: TwoPointCalibration,
        verified_calibration: VerifiedTwoPointCalibrationSuccess,
        public_metadata: TwoPointRunMetadata,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> SparseEvaluatorRunnerState:
        """Authenticate calibration provenance and reset the composite tracker."""
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
        tracker_configuration_snapshot_before = tracker._configuration_snapshot
        tracker_state_before = tracker._state
        try:
            tracker.reset(public_metadata, calibration, budget_ceiling, seed=seed)
            tracker_estimate = tracker.estimate()
            state_after = replace(
                state_before,
                phase="tracking",
                verified_calibration=verified_calibration,
                calibration=calibration,
                tracker_estimate=tracker_estimate,
                tracking_resources_before=resources_before,
                instrument_resources_current=resources_before,
                fast_update_cpu_time_s=tracker_estimate.fast_update_cpu_time_s,
                sparse_update_cpu_time_s=tracker_estimate.sparse_update_cpu_time_s,
                total_update_cpu_time_s=tracker_estimate.total_update_cpu_time_s,
            )
        except BaseException as error:
            object.__setattr__(tracker, "_configuration", tracker_configuration_before)
            object.__setattr__(
                tracker,
                "_configuration_snapshot",
                tracker_configuration_snapshot_before,
            )
            object.__setattr__(tracker, "_state", tracker_state_before)
            if isinstance(error, Exception):
                raise SparseStartError("tracker_reset_failed") from error
            raise
        object.__setattr__(self, "_tracker", tracker)
        object.__setattr__(self, "_state", state_after)
        return state_after

    def step(self) -> SparseRunnerStepOutcome:
        """Acquire and accept one pending composite-tracker observation."""
        state_before = self._state
        tracker = self._tracker
        if state_before.phase != "tracking" or tracker is None:
            raise SparseRunnerStateError("step requires a runner in the tracking phase")

        tracker_slots = _capture_tracker_slots(tracker)
        try:
            query = tracker.choose_next_query()
            if query is None:
                raise RuntimeError("terminal sparse evaluator steps are not installed")
            estimate_before = tracker.estimate()
            if estimate_before.pending_query is not query:
                raise RuntimeError("tracker estimate must retain the issued query")
            mode = _query_mode(query)
            resources_before = self._instrument.resources
            virtual_time_before = self._instrument.virtual_time_s
            expected_midpoint_s = (
                virtual_time_before
                + state_before.instrument_configuration.frequency_overhead_s
                + query.integration_time_s / 2.0
            )
        except BaseException:
            object.__setattr__(self, "_state", state_before)
            _restore_tracker_slots(tracker, tracker_slots)
            raise

        try:
            full_observation = self._instrument.query(
                query.frequency_hz,
                query.integration_time_s,
            )
        except Exception as error:
            try:
                from odmr_bench.evaluation.two_point.calibration import (
                    _safe_exception_strings,
                )

                resources_after = self._instrument.resources
                virtual_time_after = self._instrument.virtual_time_s
                exception_type, exception_message = _safe_exception_strings(error)
                failure = SparseInstrumentQueryFailure(
                    mode=mode,
                    query=query,
                    exception_type=exception_type,
                    exception_message=exception_message,
                    instrument_resources_before=resources_before,
                    instrument_resources_after=resources_after,
                )
                state_after = replace(
                    state_before,
                    tracker_estimate=estimate_before,
                    instrument_resources_current=resources_after,
                    current_virtual_time_s=virtual_time_after,
                    last_instrument_failure=failure,
                )
                outcome = SparseRunnerInstrumentFailure(
                    kind="instrument_failure",
                    failure=failure,
                    state=state_after,
                )
            except BaseException:
                _restore_tracker_slots(tracker, tracker_slots)
                raise
            object.__setattr__(self, "_state", state_after)
            return outcome
        except BaseException:
            _restore_tracker_slots(tracker, tracker_slots)
            raise

        update_slots = _capture_tracker_slots(tracker)
        try:
            resources_after = self._instrument.resources
            virtual_time_after = self._instrument.virtual_time_s
            acquisition = _build_tracking_acquisition(
                mode=mode,
                query=query,
                expected_midpoint_s=expected_midpoint_s,
                full_observation=full_observation,
                resources_before=resources_before,
                resources_after=resources_after,
                virtual_time_after=virtual_time_after,
                overhead_s=state_before.instrument_configuration.frequency_overhead_s,
            )
            if acquisition.measurement_midpoint_s is None:
                raise RuntimeError("returned observation does not match pending query")
            update = tracker.update(acquisition.safe_observation)
            pair_timings = state_before.pair_timings
            if update.completed_fast_pair is not None:
                first_acquisition = state_before.normal_tracking_trace[-1]
                first_midpoint_s = first_acquisition.measurement_midpoint_s
                second_midpoint_s = acquisition.measurement_midpoint_s
                if first_midpoint_s is None:
                    raise RuntimeError("accepted pair must retain both midpoints")
                pair = update.completed_fast_pair
                pair_timings = (
                    *pair_timings,
                    TwoPointEvaluatorPairTiming(
                        pair_index=pair.pair_index,
                        resonance_id=pair.resonance_id,
                        first_measurement_midpoint_s=first_midpoint_s,
                        second_measurement_midpoint_s=second_midpoint_s,
                        truth_reference_timestamp_s=(
                            first_midpoint_s
                            + (second_midpoint_s - first_midpoint_s) / 2.0
                        ),
                        public_reference_timestamp_s=(
                            pair.pair_reference_timestamp_s
                        ),
                        release_sequence_index=pair.release_sequence_index,
                        release_timestamp_s=pair.release_timestamp_s,
                    ),
                )
            scan_timings = state_before.scan_timings
            if update.completed_sparse_scan is not None:
                scan_acquisitions = (
                    *state_before.normal_tracking_trace[-4:],
                    acquisition,
                )
                midpoints = tuple(
                    item.measurement_midpoint_s for item in scan_acquisitions
                )
                if len(midpoints) != 5 or any(value is None for value in midpoints):
                    raise RuntimeError("accepted scan must retain five midpoints")
                exact_midpoints = tuple(float(value) for value in midpoints)
                scan = update.completed_sparse_scan
                scan_timings = (
                    *scan_timings,
                    SparseLinewidthEvaluatorScanTiming(
                        scan_index=scan.scan_index,
                        resonance_id=scan.resonance_id,
                        measurement_midpoints_s=exact_midpoints,  # type: ignore[arg-type]
                        truth_reference_timestamp_s=_ordered_mean(exact_midpoints),
                        public_reference_timestamp_s=(
                            scan.public_reference_timestamp_s
                        ),
                        release_sequence_index=scan.release_sequence_index,
                        release_timestamp_s=scan.release_timestamp_s,
                    ),
                )
            estimate = update.estimate
            state_after = replace(
                state_before,
                tracker_estimate=estimate,
                normal_tracking_trace=(
                    *state_before.normal_tracking_trace,
                    acquisition,
                ),
                pair_timings=pair_timings,
                scan_timings=scan_timings,
                instrument_resources_current=resources_after,
                instrument_current_sequence_index=full_observation.sequence_index,
                current_virtual_time_s=virtual_time_after,
                last_instrument_failure=None,
                fast_update_cpu_time_s=estimate.fast_update_cpu_time_s,
                sparse_update_cpu_time_s=estimate.sparse_update_cpu_time_s,
                total_update_cpu_time_s=estimate.total_update_cpu_time_s,
            )
            outcome = SparseRunnerAccepted(
                kind="accepted",
                acquisition=acquisition,
                update=update,
                state=state_after,
            )
        except BaseException:
            object.__setattr__(self, "_state", state_before)
            _restore_tracker_slots(tracker, update_slots)
            raise
        object.__setattr__(self, "_state", state_after)
        return outcome

    def run_until_event(self) -> SparseRunnerRunOutcome:
        """Reject runs until a later task installs tracking transitions."""
        raise SparseRunnerStateError(
            "run_until_event requires a runner in the tracking phase"
        )

    def stop_external(self) -> SparseRunnerExternallyStopped:
        """Reject stops until a later task installs tracking transitions."""
        raise SparseRunnerStateError(
            "stop_external requires a runner in the tracking phase"
        )


def _preflight_start_tracking(
    runner: SparseLinewidthEvaluatorRunner,
    tracker: SparseLinewidthCompositeTracker,
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
        raise SparseStartError("invalid_runner_phase") from None
    if (
        type(state_before) is not SparseEvaluatorRunnerState
        or type(state_before.phase) is not str
        or state_before.phase not in ("ready", "calibration_succeeded")
    ):
        raise SparseStartError("invalid_runner_phase")
    if (
        type(tracker) is not SparseLinewidthCompositeTracker
        or type(calibration) is not TwoPointCalibration
        or type(verified_calibration) is not VerifiedTwoPointCalibrationSuccess
        or type(public_metadata) is not TwoPointRunMetadata
        or type(budget_ceiling) is not TwoPointBudgetCeiling
        or isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (Integral, np.integer))
    ):
        raise SparseStartError("invalid_argument_type")

    source, binding = _authenticate_verified_calibration(verified_calibration)
    try:
        calibration_matches = (
            type(calibration.source) is TwoPointCalibrationSource
            and calibration.source is source
        )
    except Exception:
        calibration_matches = False
    if not calibration_matches:
        raise SparseStartError("calibration_mismatch")

    same_runner_success = _validate_run_provenance(
        runner,
        state_before,
        binding,
        calibration,
        verified_calibration,
    )
    _validate_start_metadata(state_before, source, calibration, public_metadata)
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
        raise SparseStartError("unverified_calibration")
    return source, binding


def _validate_run_provenance(
    runner: SparseLinewidthEvaluatorRunner,
    state_before: SparseEvaluatorRunnerState,
    binding: _RunTokenBinding,
    calibration: TwoPointCalibration,
    verified_calibration: VerifiedTwoPointCalibrationSuccess,
) -> bool:
    try:
        instrument = runner._instrument
        retained_tracker = runner._tracker
    except AttributeError:
        raise SparseStartError("run_provenance_mismatch") from None
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
    other_runner_success = False
    if state_before.phase == "ready":
        from odmr_bench.evaluation.two_point.runner import TwoPointEvaluatorRunner

        source_runner = binding.issuer_runner
        try:
            source_issuer = _lookup_verified_calibration_issuer(source_runner)
            source_state = source_runner._state
            source_instrument = source_runner._instrument
            source_tracker = source_runner._tracker
            source_state_type_matches = (
                type(source_runner) is SparseLinewidthEvaluatorRunner
                and type(source_state) is SparseEvaluatorRunnerState
            ) or (
                type(source_runner) is TwoPointEvaluatorRunner
                and type(source_state) is TwoPointEvaluatorRunnerState
            )
            other_runner_success = (
                _is_exact_registered_runner(source_runner)
                and source_runner is not runner
                and source_state_type_matches
                and type(source_instrument) is ODMRInstrument
                and type(source_state.instrument_configuration)
                is TwoPointEvaluatorInstrumentConfiguration
                and source_issuer._runner is source_runner
                and source_issuer._instrument is source_instrument
                and source_issuer._run_token is verified_calibration.run_token
                and source_issuer._instrument_configuration
                is source_state.instrument_configuration
                and own_binding is not binding
                and own_binding is not None
                and own_binding.success is None
                and own_binding.source is None
                and source_state.phase == "calibration_succeeded"
                and source_state.run_token is verified_calibration.run_token
                and source_state.calibration_outcome is verified_calibration
                and source_state.verified_calibration is verified_calibration
                and source_instrument is binding.instrument
                and source_state.instrument_configuration
                is binding.instrument_configuration
                and _lookup_run_token_binding(source_state.run_token) is binding
                and source_tracker is None
            )
        except (AttributeError, TypeError):
            other_runner_success = False
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
        raise SparseStartError("run_provenance_mismatch")
    return same_runner_success


def _validate_start_metadata(
    state_before: SparseEvaluatorRunnerState,
    source: TwoPointCalibrationSource,
    calibration: TwoPointCalibration,
    public_metadata: TwoPointRunMetadata,
) -> None:
    configuration = state_before.instrument_configuration
    mapping = source.clock_mapping
    if type(mapping) is not TwoPointClockMapping:
        raise SparseStartError("metadata_mismatch")
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
            and 0.0 <= mapped_times[3] <= public_metadata.current_timestamp_s
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
        raise SparseStartError("metadata_mismatch")


def _capture_tracking_boundary(
    runner: SparseLinewidthEvaluatorRunner,
    state_before: SparseEvaluatorRunnerState,
    verified_calibration: VerifiedTwoPointCalibrationSuccess,
    *,
    same_runner_success: bool,
) -> ResourceSnapshot:
    try:
        resources_before = runner._instrument.resources
        virtual_time_before = runner._instrument.virtual_time_s
    except Exception as error:
        raise SparseStartError("resource_boundary_mismatch") from error
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
        raise SparseStartError("resource_boundary_mismatch")
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
        raise SparseStartError("resource_boundary_mismatch") from error


def _query_mode(query: TwoPointQuery | SparseLinewidthQuery) -> Literal[
    "fast_pair", "sparse_scan"
]:
    if type(query) is TwoPointQuery:
        return "fast_pair"
    if type(query) is SparseLinewidthQuery:
        return "sparse_scan"
    raise TypeError("pending query must be an exact composite query")


def _build_tracking_acquisition(
    *,
    mode: Literal["fast_pair", "sparse_scan"],
    query: TwoPointQuery | SparseLinewidthQuery,
    expected_midpoint_s: float,
    full_observation: object,
    resources_before: ResourceSnapshot,
    resources_after: ResourceSnapshot,
    virtual_time_after: float,
    overhead_s: float,
) -> SparseTrackingAcquisition:
    from odmr_bench.emulator.observations import InstrumentObservation

    if type(full_observation) is not InstrumentObservation:
        raise TypeError("instrument query must return an InstrumentObservation")
    safe_observation = full_observation.estimator_view()
    resource_delta = _advance_full_resources(
        _zero_full_resources(), full_observation, overhead_s
    )
    expected_resources_after = _advance_full_resources(
        resources_before, full_observation, overhead_s
    )
    mismatch_fields = _resource_mismatch_fields(
        expected_resources_after, resources_after
    )
    if mismatch_fields:
        raise RuntimeError("returned observation resource join is unavailable")
    timing_matches = (
        full_observation.sequence_index == query.expected_sequence_index
        and full_observation.frequency_hz == query.frequency_hz
        and full_observation.integration_time_s == query.integration_time_s
        and full_observation.timestamp_s == query.expected_end_timestamp_s
        and virtual_time_after == query.expected_end_timestamp_s
    )
    return SparseTrackingAcquisition(
        resource_join_status="authenticated",
        mode=mode,
        query=query,
        expected_measurement_midpoint_s=expected_midpoint_s,
        measurement_midpoint_s=(expected_midpoint_s if timing_matches else None),
        full_observation=full_observation,
        safe_observation=safe_observation,
        instrument_resources_before=resources_before,
        instrument_resources_after=resources_after,
        instrument_resource_delta=resource_delta,
    )


def _capture_tracker_slots(
    tracker: SparseLinewidthCompositeTracker,
) -> tuple[object, object, object]:
    return (
        tracker._configuration,
        tracker._configuration_snapshot,
        tracker._state,
    )


def _restore_tracker_slots(
    tracker: SparseLinewidthCompositeTracker,
    slots: tuple[object, object, object],
) -> None:
    configuration, configuration_snapshot, state = slots
    object.__setattr__(tracker, "_configuration", configuration)
    object.__setattr__(tracker, "_configuration_snapshot", configuration_snapshot)
    object.__setattr__(tracker, "_state", state)


def _ordered_mean(values: tuple[float, ...]) -> float:
    result = values[0]
    for count, value in enumerate(values[1:], start=2):
        result = result + (value - result) / count
    return result
