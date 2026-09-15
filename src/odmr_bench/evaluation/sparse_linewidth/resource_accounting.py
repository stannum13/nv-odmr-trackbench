"""Evaluator-private full resource accounting for sparse-linewidth runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from odmr_bench.emulator.instrument import ODMRInstrument
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
    _project_full_resources,
    _replay_full_resources,
    _zero_full_resources,
)
from odmr_bench.evaluation.two_point.types import (
    VerifiedTwoPointCalibrationSuccess,
)

from .types import (
    SparseEvaluatorRunnerState,
    SparseLinewidthEvaluatorResources,
    SparseRunnerStateError,
)

if TYPE_CHECKING:
    from .runner import SparseLinewidthEvaluatorRunner


def _invalid_context(message: str) -> None:
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
    if state.phase != "tracking":
        raise SparseRunnerStateError(
            "resource accounting phase is not implemented by this task"
        )

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
            and source_state.phase not in {"ready", "calibration_failed"}
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
        integration_start_s = (
            previous_endpoint_s + source.source_frequency_overhead_s
        )
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
) -> ResourceSnapshot:
    zero = _zero_full_resources()
    same_runner = state.run_token is verified.run_token
    expected_boundary = calibration_resources if same_runner else zero
    expected_sequence = (
        verified.source.availability_sequence_index if same_runner else None
    )
    expected_time_s = (
        verified.source.availability_timestamp_s if same_runner else 0.0
    )
    expected_charged = (
        calibration_resources
        if calibration.budget_treatment == "included_same_run"
        else zero
    )
    if (
        bool(same_runner)
        != (calibration.budget_treatment == "included_same_run")
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
        or state.instrument_resources_current != expected_boundary
        or state.instrument_current_sequence_index != expected_sequence
        or state.current_virtual_time_s != expected_time_s
        or state.normal_tracking_trace != ()
        or state.pair_timings != ()
        or state.scan_timings != ()
        or state.last_instrument_failure is not None
        or state.terminal_abort is not None
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
        or state.fast_update_cpu_time_s != estimate.fast_update_cpu_time_s
        or state.sparse_update_cpu_time_s != estimate.sparse_update_cpu_time_s
        or state.total_update_cpu_time_s != estimate.total_update_cpu_time_s
    ):
        _invalid_context("tracking start boundary")
    return expected_charged


def build_sparse_linewidth_evaluator_resources(
    runner: SparseLinewidthEvaluatorRunner,
) -> SparseLinewidthEvaluatorResources | None:
    """Build the exact full-resource view at a successful tracking start."""
    state, verified, calibration, estimate = _authenticate_started_context(runner)
    calibration_resources = _replay_and_validate_calibration(state, verified)
    charged_resources = _validate_start_boundary(
        state,
        verified,
        calibration,
        estimate,
        calibration_resources,
    )
    zero = _zero_full_resources()
    return SparseLinewidthEvaluatorResources(
        calibration_observations=verified.full_observations,
        accepted_fast_observations=(),
        accepted_sparse_observations=(),
        accepted_tracking_observations=(),
        unaccepted_tracking_observations=(),
        calibration_resources=calibration_resources,
        fast_tracking_resources=zero,
        sparse_tracking_resources=zero,
        tracking_resources=zero,
        accepted_charged_resources=charged_resources,
        charged_resources=charged_resources,
        calibration_budget_treatment=calibration.budget_treatment,
        incomplete_fast_pair_observations=0,
        incomplete_sparse_scan_observations=0,
        unaccepted_observations=0,
    )
