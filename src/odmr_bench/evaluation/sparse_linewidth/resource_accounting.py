"""Evaluator-private full resource accounting for sparse-linewidth runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.observations import InstrumentObservation
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators.sparse_linewidth_tracker import (
    SparseLinewidthCompositeTracker,
)
from odmr_bench.estimators.sparse_linewidth_types import (
    SparseLinewidthCompositeEstimate,
)
from odmr_bench.estimators.two_point_types import TwoPointCalibration
from odmr_bench.evaluation.two_point.provenance import (
    _is_exact_registered_runner,
    _lookup_run_token_binding,
    _lookup_verified_calibration_issuer,
)
from odmr_bench.evaluation.two_point.resource_accounting import (
    _advance_full_resources,
    _project_full_resources,
    _replay_full_resources,
    _resource_mismatch_fields,
    _zero_full_resources,
)
from odmr_bench.evaluation.two_point.types import (
    VerifiedTwoPointCalibrationSuccess,
)

from .types import (
    SparseAbortedRun,
    SparseEvaluatorRunnerState,
    SparseLinewidthEvaluatorResources,
    SparseLinewidthEvaluatorScanTiming,
    SparseResourceJoinUnavailableAcquisition,
    SparseRunnerStateError,
    SparseTrackingAcquisition,
)

if TYPE_CHECKING:
    from .runner import SparseLinewidthEvaluatorRunner


def _invalid_context(message: str) -> NoReturn:
    raise ValueError(f"runner resource context is invalid: {message}")


def _authenticate_started_context(
    runner: SparseLinewidthEvaluatorRunner,
) -> tuple[
    SparseEvaluatorRunnerState,
    VerifiedTwoPointCalibrationSuccess,
    TwoPointCalibration,
    SparseLinewidthCompositeEstimate,
]:
    from .runner import SparseLinewidthEvaluatorRunner

    if type(runner) is not SparseLinewidthEvaluatorRunner:
        raise TypeError("runner must be an exact SparseLinewidthEvaluatorRunner")
    try:
        state = runner._state
    except AttributeError:
        _invalid_context("runner state slot is unavailable")
    if type(state) is not SparseEvaluatorRunnerState:
        _invalid_context("runner state type")
    if state.phase in {
        "ready",
        "calibration_succeeded",
        "calibration_failed",
    }:
        raise SparseRunnerStateError(
            "resource accounting requires a runner that has started tracking"
        )
    if state.phase not in {
        "tracking",
        "budget_stopped",
        "geometry_stopped",
        "externally_stopped",
        "aborted",
    }:
        raise SparseRunnerStateError("resource accounting requires a started runner")

    try:
        instrument = runner._instrument
        tracker = runner._tracker
    except AttributeError:
        _invalid_context("runner instrument or tracker slot is unavailable")
    if type(instrument) is not ODMRInstrument:
        _invalid_context("instrument identity")
    if type(tracker) is not SparseLinewidthCompositeTracker:
        _invalid_context("tracker identity")

    verified = state.verified_calibration
    calibration = state.calibration
    estimate = state.tracker_estimate
    if type(verified) is not VerifiedTwoPointCalibrationSuccess:
        _invalid_context("verified calibration outcome")
    if type(calibration) is not TwoPointCalibration:
        _invalid_context("calibration")
    if type(estimate) is not SparseLinewidthCompositeEstimate:
        _invalid_context("tracker estimate")

    own_binding = _lookup_run_token_binding(state.run_token)
    source_binding = _lookup_run_token_binding(verified.run_token)
    source = verified.source
    if (
        own_binding is None
        or own_binding.issuer_runner is not runner
        or own_binding.instrument is not instrument
        or own_binding.instrument_configuration is not state.instrument_configuration
        or source_binding is None
        or source_binding.success is not verified
        or source_binding.source is not source
        or calibration.source is not source
    ):
        _invalid_context("runner, token, source, or calibration identity")
    from odmr_bench.evaluation.two_point.runner import TwoPointEvaluatorRunner
    from odmr_bench.evaluation.two_point.types import TwoPointEvaluatorRunnerState

    source_runner = source_binding.issuer_runner
    if type(source_runner) is SparseLinewidthEvaluatorRunner:
        allowed_source_phases = {
            "calibration_succeeded",
            "tracking",
            "budget_stopped",
            "geometry_stopped",
            "externally_stopped",
            "aborted",
        }
    elif type(source_runner) is TwoPointEvaluatorRunner:
        allowed_source_phases = {
            "calibration_succeeded",
            "tracking",
            "budget_stopped",
            "externally_stopped",
            "aborted",
        }
    else:
        allowed_source_phases = set()
    try:
        source_state = source_runner._state
        source_instrument = source_runner._instrument
        source_issuer = _lookup_verified_calibration_issuer(source_runner)
        source_state_type_matches = (
            type(source_runner) is SparseLinewidthEvaluatorRunner
            and type(source_state) is SparseEvaluatorRunnerState
        ) or (
            type(source_runner) is TwoPointEvaluatorRunner
            and type(source_state) is TwoPointEvaluatorRunnerState
        )
        source_graph_matches = (
            _is_exact_registered_runner(source_runner)
            and source_state_type_matches
            and type(source_instrument) is ODMRInstrument
            and source_issuer._runner is source_runner
            and source_issuer._instrument is source_instrument
            and source_issuer._run_token is verified.run_token
            and source_issuer._instrument_configuration
            is source_state.instrument_configuration
            and source_binding.instrument is source_instrument
            and source_binding.instrument_configuration
            is source_state.instrument_configuration
            and source_state.run_token is verified.run_token
            and source_state.phase in allowed_source_phases
            and source_state.calibration_outcome is verified
            and source_state.verified_calibration is verified
            and source.source_frequency_overhead_s
            == source_state.instrument_configuration.frequency_overhead_s
            and source.fluorescence_provenance.nominal_photon_rate_hz
            == source_state.instrument_configuration.nominal_photon_rate_hz
        )
    except (AttributeError, TypeError):
        source_graph_matches = False
    if not source_graph_matches:
        _invalid_context("verified calibration source runner identity")
    same_runner = state.run_token is verified.run_token
    if same_runner:
        if own_binding is not source_binding:
            _invalid_context("same-run calibration authority")
    elif own_binding.success is not None or own_binding.source is not None:
        _invalid_context("conditional target calibration authority")
    try:
        tracker_state = tracker._state
        live_resources = instrument.resources
        live_time_s = instrument.virtual_time_s
        live_configuration_matches = (
            instrument.nominal_photon_rate_hz
            == state.instrument_configuration.nominal_photon_rate_hz
            and instrument.frequency_overhead_s
            == state.instrument_configuration.frequency_overhead_s
        )
    except Exception:
        _invalid_context("live runner boundary")
    if (
        tracker_state is None
        or tracker_state.calibration is not calibration
        or tracker_state.estimate is not estimate
        or not live_configuration_matches
        or live_resources != state.instrument_resources_current
        or live_time_s != state.current_virtual_time_s
    ):
        _invalid_context("tracker or live instrument boundary")
    return state, verified, calibration, estimate


def _replay_and_validate_calibration(
    state: SparseEvaluatorRunnerState,
    verified: VerifiedTwoPointCalibrationSuccess,
) -> ResourceSnapshot:
    source = verified.source
    full_observations = verified.full_observations
    if (
        tuple(observation.estimator_view() for observation in full_observations)
        != verified.safe_observations
        or verified.safe_observations != source.source_observations
        or source.fluorescence_provenance.sampling_rules
        != tuple(observation.sampling_rule for observation in full_observations)
    ):
        _invalid_context("calibration full and safe traces")
    expected_midpoints: list[float] = []
    previous_endpoint_s = source.source_start_timestamp_s
    for observation in full_observations:
        integration_start_s = previous_endpoint_s + source.source_frequency_overhead_s
        expected_midpoints.append(
            integration_start_s + observation.integration_time_s / 2.0
        )
        if (
            observation.timestamp_s
            != integration_start_s + observation.integration_time_s
        ):
            _invalid_context("calibration timestamp recurrence")
        previous_endpoint_s = observation.timestamp_s
    if verified.measurement_midpoints_s != tuple(expected_midpoints):
        _invalid_context("calibration midpoint recurrence")
    calibration_resources = _replay_full_resources(
        full_observations,
        source.source_frequency_overhead_s,
    )
    if (
        verified.instrument_resources_before != _zero_full_resources()
        or verified.instrument_resources_after != calibration_resources
        or verified.full_resources != calibration_resources
        or _project_full_resources(calibration_resources) != verified.safe_resources
        or verified.safe_resources != source.safe_resources
        or state.instrument_resources_at_bind != _zero_full_resources()
    ):
        _invalid_context("calibration resource boundaries")
    return calibration_resources


def _validate_start_boundary(
    state: SparseEvaluatorRunnerState,
    verified: VerifiedTwoPointCalibrationSuccess,
    calibration: TwoPointCalibration,
    estimate: SparseLinewidthCompositeEstimate,
    calibration_resources: ResourceSnapshot,
    *,
    unaccepted: SparseTrackingAcquisition | None = None,
) -> ResourceSnapshot:
    zero = _zero_full_resources()
    same_runner = state.run_token is verified.run_token
    expected_boundary = calibration_resources if same_runner else zero
    expected_sequence = (
        verified.source.availability_sequence_index if same_runner else None
    )
    expected_time_s = verified.source.availability_timestamp_s if same_runner else 0.0
    expected_charged = (
        calibration_resources
        if calibration.budget_treatment == "included_same_run"
        else zero
    )
    expected_current = expected_boundary
    expected_current_sequence = expected_sequence
    expected_current_time_s = expected_time_s
    if unaccepted is not None:
        expected_current = unaccepted.instrument_resources_after
        expected_current_sequence = unaccepted.full_observation.sequence_index
        expected_current_time_s = state.current_virtual_time_s
    if (
        bool(same_runner) != (calibration.budget_treatment == "included_same_run")
        or (
            same_runner
            and (
                state.calibration_outcome is not verified
                or _lookup_run_token_binding(state.run_token)
                is not _lookup_run_token_binding(verified.run_token)
            )
        )
        or (not same_runner and state.calibration_outcome is not None)
        or state.tracking_resources_before != expected_boundary
        or state.instrument_resources_current != expected_current
        or state.instrument_current_sequence_index != expected_current_sequence
        or state.current_virtual_time_s != expected_current_time_s
        or state.normal_tracking_trace != ()
        or state.pair_timings != ()
        or state.scan_timings != ()
        or (state.terminal_abort is not None) != (unaccepted is not None)
        or estimate.accepted_observations != 0
        or estimate.fast_pair_history != ()
        or estimate.sparse_scan_history != ()
        or estimate.incomplete_fast_pair is not None
        or estimate.incomplete_sparse_scan is not None
        or estimate.fast_tracking_resources != _project_full_resources(zero)
        or estimate.sparse_tracking_resources != _project_full_resources(zero)
        or estimate.tracking_resources != _project_full_resources(zero)
        or estimate.calibration_resources
        != _project_full_resources(calibration_resources)
        or estimate.charged_resources != _project_full_resources(expected_charged)
        or estimate.fast_update_cpu_time_s != 0.0
        or estimate.sparse_update_cpu_time_s != 0.0
        or estimate.total_update_cpu_time_s != 0.0
        or state.fast_update_cpu_time_s != 0.0
        or state.sparse_update_cpu_time_s != 0.0
        or state.total_update_cpu_time_s != 0.0
        or state.fast_update_cpu_time_s != estimate.fast_update_cpu_time_s
        or state.sparse_update_cpu_time_s != estimate.sparse_update_cpu_time_s
        or state.total_update_cpu_time_s != estimate.total_update_cpu_time_s
    ):
        _invalid_context("tracking start boundary")
    _validate_retryable_failure(state, estimate, expected_current)
    return expected_charged


def _validate_retryable_failure(
    state: SparseEvaluatorRunnerState,
    estimate: SparseLinewidthCompositeEstimate,
    physical: ResourceSnapshot,
) -> None:
    failure = state.last_instrument_failure
    if failure is None:
        return
    if (
        state.phase not in {"tracking", "externally_stopped"}
        or state.terminal_abort is not None
        or failure.query is not estimate.pending_query
        or failure.instrument_resources_before != physical
        or failure.instrument_resources_after != physical
        or state.instrument_resources_current != physical
    ):
        _invalid_context("retryable instrument failure")


def _validate_unavailable_abort_context(
    runner: SparseLinewidthEvaluatorRunner,
    state: SparseEvaluatorRunnerState,
    estimate: SparseLinewidthCompositeEstimate,
) -> None:
    abort = state.terminal_abort
    if (
        state.phase != "aborted"
        or abort is None
        or abort.reason != "resource_join_unavailable"
        or abort.exception_type is not None
        or abort.exception_message is not None
        or type(abort.unaccepted_acquisition)
        is not SparseResourceJoinUnavailableAcquisition
        or abort.tracker_estimate_before != estimate
        or abort.tracker_estimate_after != estimate
    ):
        _invalid_context("unavailable terminal abort")
    acquisition = abort.unaccepted_acquisition
    physical = (
        state.normal_tracking_trace[-1].instrument_resources_after
        if state.normal_tracking_trace
        else state.tracking_resources_before
    )
    if physical is None:
        _invalid_context("unavailable tracking boundary")
    overhead_s = state.instrument_configuration.frequency_overhead_s
    prospective = _advance_full_resources(
        physical, acquisition.full_observation, overhead_s
    )
    verified = state.verified_calibration
    if type(verified) is not VerifiedTwoPointCalibrationSuccess:
        _invalid_context("unavailable verified calibration")
    prior_endpoint_s = (
        state.normal_tracking_trace[-1].full_observation.timestamp_s
        if state.normal_tracking_trace
        else (
            verified.source.availability_timestamp_s
            if state.run_token is verified.run_token
            else 0.0
        )
    )
    _validate_unaccepted_midpoint(
        acquisition,
        expected_midpoint_s=(
            prior_endpoint_s + overhead_s + acquisition.query.integration_time_s / 2.0
        ),
        instrument_endpoint_s=state.current_virtual_time_s,
    )
    mismatch_fields = _resource_mismatch_fields(
        prospective, acquisition.instrument_resources_after
    )
    if (
        estimate.pending_query != acquisition.query
        or acquisition.instrument_resources_before != physical
        or acquisition.full_observation.estimator_view() != acquisition.safe_observation
        or not mismatch_fields
        or mismatch_fields != acquisition.resource_mismatch_fields
        or state.instrument_resources_current != acquisition.instrument_resources_after
        or runner._instrument.resources != acquisition.instrument_resources_after
        or state.current_virtual_time_s != runner._instrument.virtual_time_s
        or state.instrument_current_sequence_index
        != acquisition.full_observation.sequence_index
        or any(
            acquisition.full_observation is item.full_observation
            for item in state.normal_tracking_trace
        )
    ):
        _invalid_context("unavailable resource join")


def _validate_unaccepted_midpoint(
    acquisition: SparseTrackingAcquisition | SparseResourceJoinUnavailableAcquisition,
    *,
    expected_midpoint_s: float,
    instrument_endpoint_s: float,
) -> None:
    full = acquisition.full_observation
    query = acquisition.query
    timing_matches = (
        full.integration_time_s == query.integration_time_s
        and full.timestamp_s == query.expected_end_timestamp_s
        and instrument_endpoint_s == query.expected_end_timestamp_s
    )
    expected_measurement = expected_midpoint_s if timing_matches else None
    if (
        acquisition.expected_measurement_midpoint_s != expected_midpoint_s
        or acquisition.measurement_midpoint_s != expected_measurement
    ):
        _invalid_context("unaccepted acquisition midpoint")


def _validate_authenticated_abort_context(
    state: SparseEvaluatorRunnerState,
    estimate: SparseLinewidthCompositeEstimate,
    acquisition: SparseTrackingAcquisition,
) -> None:
    abort = state.terminal_abort
    if (
        state.phase != "aborted"
        or abort is None
        or abort.reason
        not in {
            "tracker_observation_validation_error",
            "tracker_update_construction_error",
            "tracker_update_unexpected_error",
        }
        or type(abort.exception_type) is not str
        or not abort.exception_type
        or type(abort.exception_message) is not str
        or abort.unaccepted_acquisition is not acquisition
        or abort.unaccepted_observation_count != 1
        or abort.tracker_estimate_before is not estimate
        or abort.tracker_estimate_after is not estimate
        or estimate.pending_query is not acquisition.query
        or state.last_instrument_failure is not None
    ):
        _invalid_context("authenticated terminal abort")


def _validate_abort_causal_binding(
    runner: SparseLinewidthEvaluatorRunner,
    abort: object,
) -> None:
    from .runner import _lookup_abort_causal_binding

    if type(abort) is not SparseAbortedRun:
        _invalid_context("terminal abort type")
    binding = _lookup_abort_causal_binding(runner, abort)
    if (
        binding is None
        or binding.reason != abort.reason
        or binding.exception_type != abort.exception_type
        or binding.exception_message != abort.exception_message
        or binding.acquisition is not abort.unaccepted_acquisition
        or binding.tracker_estimate_before is not abort.tracker_estimate_before
        or binding.tracker_estimate_after is not abort.tracker_estimate_after
    ):
        _invalid_context("terminal abort causal binding")


def build_sparse_linewidth_evaluator_resources(
    runner: SparseLinewidthEvaluatorRunner,
) -> SparseLinewidthEvaluatorResources | None:
    """Build the exact full-resource view through accepted and terminal atoms."""
    state, verified, calibration, estimate = _authenticate_started_context(runner)
    abort = state.terminal_abort
    unaccepted: SparseTrackingAcquisition | None = None
    if abort is not None:
        _validate_abort_causal_binding(runner, abort)
        if (
            type(abort.unaccepted_acquisition)
            is SparseResourceJoinUnavailableAcquisition
        ):
            _validate_unavailable_abort_context(runner, state, estimate)
            return None
        if type(abort.unaccepted_acquisition) is not SparseTrackingAcquisition:
            _invalid_context("terminal unaccepted acquisition type")
        unaccepted = abort.unaccepted_acquisition
        _validate_authenticated_abort_context(state, estimate, unaccepted)
    calibration_resources = _replay_and_validate_calibration(state, verified)
    if not state.normal_tracking_trace:
        charged_resources = _validate_start_boundary(
            state,
            verified,
            calibration,
            estimate,
            calibration_resources,
            unaccepted=unaccepted,
        )
        accepted_fast: tuple[InstrumentObservation, ...] = ()
        accepted_sparse: tuple[InstrumentObservation, ...] = ()
        accepted_tracking: tuple[InstrumentObservation, ...] = ()
        fast_resources = _zero_full_resources()
        sparse_resources = _zero_full_resources()
        tracking_resources = _zero_full_resources()
        incomplete_fast = 0
        incomplete_sparse = 0
        physical = state.tracking_resources_before
        if physical is None:
            _invalid_context("tracking resource boundary")
    else:
        (
            accepted_fast,
            accepted_sparse,
            accepted_tracking,
            fast_resources,
            sparse_resources,
            tracking_resources,
            charged_resources,
            incomplete_fast,
            incomplete_sparse,
        ) = _validate_accepted_context(
            runner,
            state,
            verified,
            calibration,
            estimate,
            calibration_resources,
            unaccepted=unaccepted,
        )
        physical = state.normal_tracking_trace[-1].instrument_resources_after
    accepted_charged = charged_resources
    unaccepted_observations: tuple[InstrumentObservation, ...] = ()
    if unaccepted is not None:
        overhead_s = state.instrument_configuration.frequency_overhead_s
        full = unaccepted.full_observation
        expected_after = _advance_full_resources(physical, full, overhead_s)
        prior_endpoint_s = (
            accepted_tracking[-1].timestamp_s
            if accepted_tracking
            else (
                verified.source.availability_timestamp_s
                if state.run_token is verified.run_token
                else 0.0
            )
        )
        _validate_unaccepted_midpoint(
            unaccepted,
            expected_midpoint_s=(
                prior_endpoint_s
                + overhead_s
                + unaccepted.query.integration_time_s / 2.0
            ),
            instrument_endpoint_s=state.current_virtual_time_s,
        )
        if (
            unaccepted.instrument_resources_before != physical
            or unaccepted.instrument_resources_after != expected_after
            or unaccepted.instrument_resource_delta
            != _advance_full_resources(_zero_full_resources(), full, overhead_s)
            or full.estimator_view() != unaccepted.safe_observation
            or state.instrument_resources_current != expected_after
            or runner._instrument.resources != expected_after
            or state.current_virtual_time_s != runner._instrument.virtual_time_s
        ):
            _invalid_context("authenticated unaccepted acquisition join")
        charged_resources = _advance_full_resources(charged_resources, full, overhead_s)
        unaccepted_observations = (full,)
    return SparseLinewidthEvaluatorResources(
        calibration_observations=verified.full_observations,
        accepted_fast_observations=accepted_fast,
        accepted_sparse_observations=accepted_sparse,
        accepted_tracking_observations=accepted_tracking,
        unaccepted_tracking_observations=unaccepted_observations,
        calibration_resources=calibration_resources,
        fast_tracking_resources=fast_resources,
        sparse_tracking_resources=sparse_resources,
        tracking_resources=tracking_resources,
        accepted_charged_resources=accepted_charged,
        charged_resources=charged_resources,
        calibration_budget_treatment=calibration.budget_treatment,
        incomplete_fast_pair_observations=incomplete_fast,  # type: ignore[arg-type]
        incomplete_sparse_scan_observations=incomplete_sparse,  # type: ignore[arg-type]
        unaccepted_observations=len(unaccepted_observations),  # type: ignore[arg-type]
    )


def _safe_arrivals(
    estimate: SparseLinewidthCompositeEstimate,
) -> tuple[tuple[object, object, str], ...]:
    arrivals: list[tuple[object, object, str]] = []
    for pair in estimate.fast_pair_history:
        if pair.first_side == "minus":
            arrivals.extend(
                (
                    (pair.minus_query, pair.minus_observation, "fast_pair"),
                    (pair.plus_query, pair.plus_observation, "fast_pair"),
                )
            )
        else:
            arrivals.extend(
                (
                    (pair.plus_query, pair.plus_observation, "fast_pair"),
                    (pair.minus_query, pair.minus_observation, "fast_pair"),
                )
            )
    if estimate.incomplete_fast_pair is not None:
        partial = estimate.incomplete_fast_pair
        arrivals.append((partial.first_query, partial.first_observation, "fast_pair"))
    for scan in estimate.sparse_scan_history:
        arrivals.extend(
            (query, observation, "sparse_scan")
            for query, observation in zip(scan.queries, scan.observations, strict=True)
        )
    if estimate.incomplete_sparse_scan is not None:
        partial_scan = estimate.incomplete_sparse_scan
        arrivals.extend(
            (query, observation, "sparse_scan")
            for query, observation in zip(
                partial_scan.queries, partial_scan.observations, strict=True
            )
        )
    arrivals.sort(key=lambda item: item[1].sequence_index)
    return tuple(arrivals)


def _validate_accepted_context(
    runner: SparseLinewidthEvaluatorRunner,
    state: SparseEvaluatorRunnerState,
    verified: VerifiedTwoPointCalibrationSuccess,
    calibration: TwoPointCalibration,
    estimate: SparseLinewidthCompositeEstimate,
    calibration_resources: ResourceSnapshot,
    *,
    unaccepted: SparseTrackingAcquisition | None = None,
) -> tuple[
    tuple[InstrumentObservation, ...],
    tuple[InstrumentObservation, ...],
    tuple[InstrumentObservation, ...],
    ResourceSnapshot,
    ResourceSnapshot,
    ResourceSnapshot,
    ResourceSnapshot,
    int,
    int,
]:
    arrivals = _safe_arrivals(estimate)
    trace = state.normal_tracking_trace
    if len(arrivals) != estimate.accepted_observations or len(trace) != len(arrivals):
        _invalid_context("accepted observation cardinality")
    safe_trace = tuple(acquisition.safe_observation for acquisition in trace)
    if safe_trace != tuple(item[1] for item in arrivals):
        _invalid_context("accepted global arrival order")
    overhead_s = state.instrument_configuration.frequency_overhead_s
    physical = state.tracking_resources_before
    if physical is None:
        _invalid_context("tracking resource boundary")
    previous_endpoint_s = (
        verified.source.availability_timestamp_s
        if state.run_token is verified.run_token
        else 0.0
    )
    fast: list[InstrumentObservation] = []
    sparse: list[InstrumentObservation] = []
    for acquisition, (query, observation, mode) in zip(trace, arrivals, strict=True):
        full = acquisition.full_observation
        expected_after = _advance_full_resources(physical, full, overhead_s)
        expected_delta = _advance_full_resources(
            _zero_full_resources(), full, overhead_s
        )
        expected_midpoint_s = (
            previous_endpoint_s + overhead_s + query.integration_time_s / 2.0
        )
        if (
            acquisition.mode != mode
            or acquisition.query is not query
            or acquisition.safe_observation is not observation
            or full.estimator_view() != observation
            or acquisition.instrument_resources_before != physical
            or acquisition.instrument_resources_after != expected_after
            or acquisition.instrument_resource_delta != expected_delta
            or acquisition.expected_measurement_midpoint_s != expected_midpoint_s
            or acquisition.measurement_midpoint_s != expected_midpoint_s
            or full.sequence_index != query.expected_sequence_index
            or full.timestamp_s != query.expected_end_timestamp_s
            or full.frequency_hz != query.frequency_hz
            or full.integration_time_s != query.integration_time_s
        ):
            _invalid_context("accepted acquisition identity or resource join")
        (fast if mode == "fast_pair" else sparse).append(full)
        physical = expected_after
        previous_endpoint_s = full.timestamp_s
    accepted = tuple(acquisition.full_observation for acquisition in trace)
    fast_tuple = tuple(fast)
    sparse_tuple = tuple(sparse)
    fast_resources = _replay_full_resources(fast_tuple, overhead_s)
    sparse_resources = _replay_full_resources(sparse_tuple, overhead_s)
    tracking_resources = _replay_full_resources(accepted, overhead_s)
    charged_resources = (
        calibration_resources
        if calibration.budget_treatment == "included_same_run"
        else _zero_full_resources()
    )
    for observation in accepted:
        charged_resources = _advance_full_resources(
            charged_resources, observation, overhead_s
        )
    live_physical = physical
    live_sequence_index = accepted[-1].sequence_index
    live_endpoint_s = previous_endpoint_s
    if unaccepted is not None:
        live_physical = _advance_full_resources(
            physical, unaccepted.full_observation, overhead_s
        )
        live_sequence_index = unaccepted.full_observation.sequence_index
        live_endpoint_s = state.current_virtual_time_s
    if (
        _project_full_resources(fast_resources) != estimate.fast_tracking_resources
        or _project_full_resources(sparse_resources)
        != estimate.sparse_tracking_resources
        or _project_full_resources(tracking_resources) != estimate.tracking_resources
        or _project_full_resources(charged_resources) != estimate.charged_resources
        or state.instrument_resources_current != live_physical
        or runner._instrument.resources != live_physical
        or state.current_virtual_time_s != live_endpoint_s
        or runner._instrument.virtual_time_s != live_endpoint_s
        or state.instrument_current_sequence_index != live_sequence_index
        or state.fast_update_cpu_time_s != estimate.fast_update_cpu_time_s
        or state.sparse_update_cpu_time_s != estimate.sparse_update_cpu_time_s
        or state.total_update_cpu_time_s != estimate.total_update_cpu_time_s
    ):
        _invalid_context("accepted ledgers, boundary, or CPU totals")
    _validate_retryable_failure(state, estimate, live_physical)
    _validate_timing_context(state, estimate)
    incomplete_fast = int(estimate.incomplete_fast_pair is not None)
    incomplete_sparse = (
        0
        if estimate.incomplete_sparse_scan is None
        else len(estimate.incomplete_sparse_scan.observations)
    )
    return (
        fast_tuple,
        sparse_tuple,
        accepted,
        fast_resources,
        sparse_resources,
        tracking_resources,
        charged_resources,
        incomplete_fast,
        incomplete_sparse,
    )


def _ordered_mean(values: tuple[float, ...]) -> float:
    result = values[0]
    for count, value in enumerate(values[1:], start=2):
        result = result + (value - result) / count
    return result


def _validate_timing_context(
    state: SparseEvaluatorRunnerState,
    estimate: SparseLinewidthCompositeEstimate,
) -> None:
    if len(state.pair_timings) != len(estimate.fast_pair_history) or len(
        state.scan_timings
    ) != len(estimate.sparse_scan_history):
        _invalid_context("completed timing cardinality")
    acquisition_by_sequence = {
        acquisition.full_observation.sequence_index: acquisition
        for acquisition in state.normal_tracking_trace
    }
    if len(acquisition_by_sequence) != len(state.normal_tracking_trace):
        _invalid_context("tracking sequence uniqueness")
    for pair, timing in zip(
        estimate.fast_pair_history, state.pair_timings, strict=True
    ):
        first_observation, second_observation = (
            (pair.minus_observation, pair.plus_observation)
            if pair.first_side == "minus"
            else (pair.plus_observation, pair.minus_observation)
        )
        try:
            first = acquisition_by_sequence[first_observation.sequence_index]
            second = acquisition_by_sequence[second_observation.sequence_index]
        except KeyError:
            _invalid_context("pair timing acquisition membership")
        first_midpoint = first.measurement_midpoint_s
        second_midpoint = second.measurement_midpoint_s
        if first_midpoint is None or second_midpoint is None:
            _invalid_context("pair timing midpoint presence")
        if (
            timing.pair_index != pair.pair_index
            or timing.resonance_id != pair.resonance_id
            or timing.first_measurement_midpoint_s != first_midpoint
            or timing.second_measurement_midpoint_s != second_midpoint
            or timing.truth_reference_timestamp_s
            != first_midpoint + (second_midpoint - first_midpoint) / 2.0
            or timing.public_reference_timestamp_s != pair.pair_reference_timestamp_s
            or timing.release_sequence_index != pair.release_sequence_index
            or timing.release_timestamp_s != pair.release_timestamp_s
        ):
            _invalid_context("pair timing join")
    for scan, timing in zip(
        estimate.sparse_scan_history, state.scan_timings, strict=True
    ):
        if type(timing) is not SparseLinewidthEvaluatorScanTiming:
            _invalid_context("scan timing type")
        try:
            acquisitions = tuple(
                acquisition_by_sequence[observation.sequence_index]
                for observation in scan.observations
            )
        except KeyError:
            _invalid_context("scan timing acquisition membership")
        midpoints = tuple(
            acquisition.measurement_midpoint_s for acquisition in acquisitions
        )
        if len(midpoints) != 5 or any(value is None for value in midpoints):
            _invalid_context("scan timing midpoint presence")
        exact_midpoints = tuple(float(value) for value in midpoints)
        if (
            timing.scan_index != scan.scan_index
            or timing.resonance_id != scan.resonance_id
            or timing.measurement_midpoints_s != exact_midpoints
            or timing.truth_reference_timestamp_s != _ordered_mean(exact_midpoints)
            or timing.public_reference_timestamp_s != scan.public_reference_timestamp_s
            or timing.release_sequence_index != scan.release_sequence_index
            or timing.release_timestamp_s != scan.release_timestamp_s
        ):
            _invalid_context("scan timing join")
