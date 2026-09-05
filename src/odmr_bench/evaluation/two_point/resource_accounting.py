"""Evaluator-private full resource accounting for two-point traces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.observations import InstrumentObservation
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators.two_point_tracker import CalibratedTwoPointTracker
from odmr_bench.estimators.two_point_types import (
    PublicAcquisitionResources,
    TwoPointCalibration,
    TwoPointEstimate,
)
from odmr_bench.evaluation.two_point.types import (
    ResourceJoinMismatchField,
    TwoPointEvaluatorPairTiming,
    TwoPointEvaluatorResources,
    TwoPointEvaluatorRunnerState,
    TwoPointInstrumentQueryFailure,
    TwoPointResourceJoinUnavailableAcquisition,
    TwoPointTrackingAcquisition,
    VerifiedTwoPointCalibrationSuccess,
)

if TYPE_CHECKING:
    from odmr_bench.evaluation.two_point.runner import TwoPointEvaluatorRunner


def _zero_full_resources() -> ResourceSnapshot:
    return ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)


def _advance_full_resources(
    resources: ResourceSnapshot,
    observation: InstrumentObservation,
    overhead_s: float,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        observations=resources.observations + 1,
        integration_time_s=(
            resources.integration_time_s + observation.integration_time_s
        ),
        nominal_exposure_photons=(
            resources.nominal_exposure_photons
            + observation.nominal_exposure_photons
        ),
        expected_photons=resources.expected_photons + observation.expected_photons,
        realized_photons=(
            resources.realized_photons
            + (
                observation.realized_photons
                if observation.realized_photons is not None
                else 0
            )
        ),
        observations_without_realized_counts=(
            resources.observations_without_realized_counts
            + int(observation.realized_photons is None)
        ),
        virtual_elapsed_time_s=(
            resources.virtual_elapsed_time_s
            + (overhead_s + observation.integration_time_s)
        ),
    )


def _replay_full_resources(
    observations: Sequence[InstrumentObservation],
    overhead_s: float,
) -> ResourceSnapshot:
    resources = _zero_full_resources()
    for observation in observations:
        resources = _advance_full_resources(resources, observation, overhead_s)
    return resources


def _project_full_resources(
    resources: ResourceSnapshot,
) -> PublicAcquisitionResources:
    return PublicAcquisitionResources(
        observations=resources.observations,
        integration_time_s=resources.integration_time_s,
        nominal_exposure_photons=resources.nominal_exposure_photons,
        realized_photons=resources.realized_photons,
        observations_without_realized_counts=(
            resources.observations_without_realized_counts
        ),
        virtual_elapsed_time_s=resources.virtual_elapsed_time_s,
    )


def _resource_mismatch_fields(
    expected: ResourceSnapshot,
    actual: ResourceSnapshot,
) -> tuple[ResourceJoinMismatchField, ...]:
    mismatches: list[ResourceJoinMismatchField] = []
    if expected.observations != actual.observations:
        mismatches.append("observations")
    if expected.integration_time_s != actual.integration_time_s:
        mismatches.append("integration_time_s")
    if expected.nominal_exposure_photons != actual.nominal_exposure_photons:
        mismatches.append("nominal_exposure_photons")
    if expected.expected_photons != actual.expected_photons:
        mismatches.append("expected_photons")
    if expected.realized_photons != actual.realized_photons:
        mismatches.append("realized_photons")
    if (
        expected.observations_without_realized_counts
        != actual.observations_without_realized_counts
    ):
        mismatches.append("observations_without_realized_counts")
    if expected.virtual_elapsed_time_s != actual.virtual_elapsed_time_s:
        mismatches.append("virtual_elapsed_time_s")
    return tuple(mismatches)


def build_two_point_evaluator_resources(
    runner: TwoPointEvaluatorRunner,
) -> TwoPointEvaluatorResources | None:
    """Build the lossless evaluator resource view for one original runner."""
    from odmr_bench.evaluation.two_point.runner import TwoPointEvaluatorRunner

    if type(runner) is not TwoPointEvaluatorRunner:
        raise TypeError("runner must be an exact TwoPointEvaluatorRunner")
    authenticated_unaccepted = None
    state = runner._state
    if (
        type(state) is TwoPointEvaluatorRunnerState
        and state.phase == "aborted"
        and state.terminal_abort is not None
    ):
        terminal_acquisition = state.terminal_abort.unaccepted_acquisition
        if type(terminal_acquisition) is TwoPointResourceJoinUnavailableAcquisition:
            _validate_unavailable_abort_context(runner)
            return None
        if type(terminal_acquisition) is TwoPointTrackingAcquisition:
            authenticated_unaccepted = terminal_acquisition
    return _build_two_point_evaluator_resources_from_context(
        runner,
        authenticated_unaccepted=authenticated_unaccepted,
    )


def _invalid_context(message: str) -> None:
    raise ValueError(f"runner resource context is invalid: {message}")


def _estimate_arrivals(
    estimate: TwoPointEstimate,
) -> tuple[tuple[object, object], ...]:
    arrivals: list[tuple[object, object]] = []
    for pair in estimate.pair_history:
        if pair.first_side == "minus":
            arrivals.extend(
                (
                    (pair.minus_query, pair.minus_observation),
                    (pair.plus_query, pair.plus_observation),
                )
            )
        else:
            arrivals.extend(
                (
                    (pair.plus_query, pair.plus_observation),
                    (pair.minus_query, pair.minus_observation),
                )
            )
    if estimate.incomplete_pair is not None:
        partial = estimate.incomplete_pair
        arrivals.append((partial.first_query, partial.first_observation))
    return tuple(arrivals)


def _authenticate_runner_context(
    runner: TwoPointEvaluatorRunner,
) -> tuple[
    TwoPointEvaluatorRunnerState,
    VerifiedTwoPointCalibrationSuccess,
    TwoPointCalibration,
    TwoPointEstimate,
]:
    from odmr_bench.evaluation.two_point.provenance import (
        _lookup_run_token_binding,
    )
    from odmr_bench.evaluation.two_point.runner import TwoPointEvaluatorRunner

    if type(runner) is not TwoPointEvaluatorRunner:
        raise TypeError("runner must be an exact TwoPointEvaluatorRunner")
    try:
        state = runner._state
        instrument = runner._instrument
        tracker = runner._tracker
    except AttributeError:
        _invalid_context("runner slots are unavailable")
    if type(state) is not TwoPointEvaluatorRunnerState or state.phase not in {
        "tracking",
        "budget_stopped",
        "externally_stopped",
        "aborted",
    }:
        _invalid_context("runner has not started tracking")
    if type(instrument) is not ODMRInstrument:
        _invalid_context("instrument identity is unavailable")
    if type(tracker) is not CalibratedTwoPointTracker:
        _invalid_context("tracker identity is unavailable")
    configuration = state.instrument_configuration
    try:
        instrument_configuration_matches = (
            instrument.nominal_photon_rate_hz
            == configuration.nominal_photon_rate_hz
            and instrument.frequency_overhead_s
            == configuration.frequency_overhead_s
        )
    except Exception:
        instrument_configuration_matches = False
    if not instrument_configuration_matches:
        _invalid_context("live instrument configuration")

    own_binding = _lookup_run_token_binding(state.run_token)
    if (
        own_binding is None
        or own_binding.issuer_runner is not runner
        or own_binding.instrument is not instrument
        or own_binding.instrument_configuration
        is not state.instrument_configuration
    ):
        _invalid_context("runner token, instrument, or configuration identity")

    verified = state.verified_calibration
    calibration = state.calibration
    estimate = state.tracker_estimate
    if type(verified) is not VerifiedTwoPointCalibrationSuccess:
        _invalid_context("verified calibration outcome is unavailable")
    if type(calibration) is not TwoPointCalibration:
        _invalid_context("calibration is unavailable")
    if type(estimate) is not TwoPointEstimate:
        _invalid_context("tracker estimate is unavailable")
    source = verified.source
    verified_binding = _lookup_run_token_binding(verified.run_token)
    if (
        verified_binding is None
        or verified_binding.success is not verified
        or verified_binding.source is not source
        or verified_binding.issuer_runner._instrument
        is not verified_binding.instrument
        or verified_binding.issuer_runner._state.run_token
        is not verified.run_token
        or verified_binding.issuer_runner._state.instrument_configuration
        is not verified_binding.instrument_configuration
    ):
        _invalid_context("verified outcome or source identity")
    same_runner = verified_binding.issuer_runner is runner
    if same_runner:
        if (
            verified_binding is not own_binding
            or state.run_token is not verified.run_token
            or state.calibration_outcome is not verified
        ):
            _invalid_context("same-run provenance")
    elif (
        state.calibration_outcome is not None
        or own_binding.success is not None
        or own_binding.source is not None
    ):
        _invalid_context("conditional cross-run provenance")
    if calibration.budget_treatment == "included_same_run" and not same_runner:
        _invalid_context("included treatment requires the issuing runner")
    if (
        source.provenance != "verified_factory_acquisition"
        or calibration.source is not source
        or tracker._state is None
        or tracker._state.calibration is not calibration
        or tracker._state.estimate is not estimate
        or tracker._configuration != calibration.configuration
        or estimate.calibration_source_id != source.source_id
        or estimate.calibration_source_provenance != source.provenance
        or estimate.calibration_budget_treatment != calibration.budget_treatment
        or estimate.calibration_resources != source.safe_resources
        or source.source_frequency_overhead_s
        != verified_binding.instrument_configuration.frequency_overhead_s
        or source.fluorescence_provenance.nominal_photon_rate_hz
        != verified_binding.instrument_configuration.nominal_photon_rate_hz
    ):
        _invalid_context("calibration, tracker, and estimate join")
    return state, verified, calibration, estimate


def _validate_calibration_context(
    state: TwoPointEvaluatorRunnerState,
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
        or _project_full_resources(calibration_resources)
        != verified.safe_resources
        or verified.safe_resources != source.safe_resources
    ):
        _invalid_context("calibration resource boundaries")
    if state.instrument_resources_at_bind != _zero_full_resources():
        _invalid_context("runner bind boundary")
    return calibration_resources


def _validate_pair_timing_context(
    state: TwoPointEvaluatorRunnerState,
    estimate: TwoPointEstimate,
) -> None:
    if len(state.pair_timings) != len(estimate.pair_history):
        _invalid_context("pair timing cardinality")
    for pair_index, (pair, timing) in enumerate(
        zip(estimate.pair_history, state.pair_timings, strict=True)
    ):
        if type(timing) is not TwoPointEvaluatorPairTiming:
            _invalid_context("pair timing type")
        first = state.normal_tracking_trace[2 * pair_index]
        second = state.normal_tracking_trace[2 * pair_index + 1]
        first_midpoint = first.measurement_midpoint_s
        second_midpoint = second.measurement_midpoint_s
        if first_midpoint is None or second_midpoint is None:
            _invalid_context("accepted pair midpoint")
        truth_reference = first_midpoint + (
            second_midpoint - first_midpoint
        ) / 2.0
        if (
            timing.pair_index != pair.pair_index
            or timing.resonance_id != pair.resonance_id
            or timing.first_measurement_midpoint_s != first_midpoint
            or timing.second_measurement_midpoint_s != second_midpoint
            or timing.truth_reference_timestamp_s != truth_reference
            or timing.public_reference_timestamp_s
            != pair.pair_reference_timestamp_s
            or timing.release_sequence_index != pair.release_sequence_index
            or timing.release_timestamp_s != pair.release_timestamp_s
        ):
            _invalid_context("pair timing and estimate history")


def _validate_accepted_tracking_context(
    state: TwoPointEvaluatorRunnerState,
    verified: VerifiedTwoPointCalibrationSuccess,
    calibration: TwoPointCalibration,
    estimate: TwoPointEstimate,
    calibration_resources: ResourceSnapshot,
) -> tuple[tuple[InstrumentObservation, ...], ResourceSnapshot, ResourceSnapshot]:
    trace = state.normal_tracking_trace
    if not all(type(value) is TwoPointTrackingAcquisition for value in trace):
        _invalid_context("normal tracking acquisition type")
    expected_arrivals = _estimate_arrivals(estimate)
    if len(trace) != len(expected_arrivals):
        _invalid_context("accepted tracking trace cardinality")
    overhead_s = state.instrument_configuration.frequency_overhead_s
    physical_resources = state.tracking_resources_before
    if type(physical_resources) is not ResourceSnapshot:
        _invalid_context("tracking resource boundary")
    same_runner = state.run_token is verified.run_token
    expected_tracking_boundary = (
        verified.instrument_resources_after
        if same_runner
        else state.instrument_resources_at_bind
    )
    if physical_resources != expected_tracking_boundary:
        _invalid_context("tracking start boundary")
    previous_endpoint_s = (
        verified.source.availability_timestamp_s if same_runner else 0.0
    )

    accepted_observations: list[InstrumentObservation] = []
    for acquisition, (expected_query, expected_observation) in zip(
        trace, expected_arrivals, strict=True
    ):
        full_observation = acquisition.full_observation
        if (
            type(full_observation) is not InstrumentObservation
            or full_observation.estimator_view() != acquisition.safe_observation
            or acquisition.query != expected_query
            or acquisition.safe_observation != expected_observation
        ):
            _invalid_context("accepted full, safe, and estimate trace")
        expected_delta = _advance_full_resources(
            _zero_full_resources(), full_observation, overhead_s
        )
        expected_after = _advance_full_resources(
            physical_resources, full_observation, overhead_s
        )
        integration_start_s = previous_endpoint_s + overhead_s
        expected_midpoint_s = (
            integration_start_s + acquisition.query.integration_time_s / 2.0
        )
        if (
            acquisition.instrument_resource_delta != expected_delta
            or acquisition.instrument_resources_before != physical_resources
            or acquisition.instrument_resources_after != expected_after
            or full_observation.sequence_index
            != acquisition.query.expected_sequence_index
            or full_observation.timestamp_s
            != integration_start_s + acquisition.query.integration_time_s
            or full_observation.frequency_hz != acquisition.query.frequency_hz
            or full_observation.integration_time_s
            != acquisition.query.integration_time_s
        ):
            _invalid_context("accepted acquisition resource boundary")
        if (
            acquisition.expected_measurement_midpoint_s
            != acquisition.measurement_midpoint_s
            or acquisition.measurement_midpoint_s is None
            or acquisition.measurement_midpoint_s != expected_midpoint_s
        ):
            _invalid_context("accepted acquisition midpoint")
        accepted_observations.append(full_observation)
        physical_resources = expected_after
        previous_endpoint_s = full_observation.timestamp_s

    accepted_tuple = tuple(accepted_observations)
    failure = state.last_instrument_failure
    if failure is not None and (
        type(failure) is not TwoPointInstrumentQueryFailure
        or estimate.pending_query is None
        or failure.query != estimate.pending_query
        or failure.instrument_resources_before != physical_resources
        or failure.instrument_resources_after != physical_resources
    ):
        _invalid_context("instrument failure resource boundary")
    accepted_resources = _replay_full_resources(accepted_tuple, overhead_s)
    if _project_full_resources(accepted_resources) != estimate.tracking_resources:
        _invalid_context("accepted tracking resource projection")
    accepted_charged = (
        calibration_resources
        if calibration.budget_treatment == "included_same_run"
        else _zero_full_resources()
    )
    for observation in accepted_tuple:
        accepted_charged = _advance_full_resources(
            accepted_charged, observation, overhead_s
        )
    if _project_full_resources(accepted_charged) != estimate.charged_resources:
        _invalid_context("accepted charged resource projection")
    _validate_pair_timing_context(state, estimate)
    return accepted_tuple, physical_resources, accepted_charged


def _validate_unaccepted_midpoint(
    acquisition: (
        TwoPointTrackingAcquisition
        | TwoPointResourceJoinUnavailableAcquisition
    ),
    *,
    expected_midpoint_s: float,
) -> None:
    full_observation = acquisition.full_observation
    query = acquisition.query
    timing_matches = (
        full_observation.integration_time_s == query.integration_time_s
        and full_observation.timestamp_s == query.expected_end_timestamp_s
        and acquisition.instrument_resources_after.virtual_elapsed_time_s
        == query.expected_end_timestamp_s
    )
    expected_measurement_midpoint_s = (
        expected_midpoint_s if timing_matches else None
    )
    if (
        acquisition.expected_measurement_midpoint_s != expected_midpoint_s
        or acquisition.measurement_midpoint_s
        != expected_measurement_midpoint_s
    ):
        _invalid_context("unaccepted acquisition midpoint")


def _validate_unavailable_abort_context(
    runner: TwoPointEvaluatorRunner,
) -> None:
    state, verified, calibration, estimate = _authenticate_runner_context(runner)
    abort = state.terminal_abort
    if (
        state.phase != "aborted"
        or abort is None
        or abort.reason != "resource_join_unavailable"
        or abort.exception_type is not None
        or abort.exception_message is not None
        or type(abort.unaccepted_acquisition)
        is not TwoPointResourceJoinUnavailableAcquisition
        or abort.tracker_estimate_before != estimate
        or abort.tracker_estimate_after != estimate
    ):
        _invalid_context("unavailable terminal abort")
    acquisition = abort.unaccepted_acquisition
    calibration_resources = _validate_calibration_context(state, verified)
    accepted, physical_resources, _ = _validate_accepted_tracking_context(
        state,
        verified,
        calibration,
        estimate,
        calibration_resources,
    )
    full_observation = acquisition.full_observation
    overhead_s = state.instrument_configuration.frequency_overhead_s
    accepted_time_s = (
        accepted[-1].timestamp_s
        if accepted
        else (
            verified.source.availability_timestamp_s
            if state.run_token is verified.run_token
            else 0.0
        )
    )
    expected_midpoint_s = (
        accepted_time_s + overhead_s
    ) + acquisition.query.integration_time_s / 2.0
    _validate_unaccepted_midpoint(
        acquisition,
        expected_midpoint_s=expected_midpoint_s,
    )
    if (
        estimate.pending_query is None
        or acquisition.query != estimate.pending_query
        or type(full_observation) is not InstrumentObservation
        or full_observation.estimator_view() != acquisition.safe_observation
        or acquisition.instrument_resources_before != physical_resources
    ):
        _invalid_context("unavailable raw acquisition")
    prospective_after = _advance_full_resources(
        physical_resources,
        full_observation,
        overhead_s,
    )
    mismatch_fields = _resource_mismatch_fields(
        prospective_after,
        acquisition.instrument_resources_after,
    )
    if (
        not mismatch_fields
        or mismatch_fields != acquisition.resource_mismatch_fields
        or any(
            acquisition.full_observation is accepted_acquisition.full_observation
            for accepted_acquisition in state.normal_tracking_trace
        )
    ):
        _invalid_context("unavailable resource mismatch fields")
    try:
        live_resources = runner._instrument.resources
        live_time_s = runner._instrument.virtual_time_s
    except Exception:
        _invalid_context("unavailable live instrument boundary")
    if (
        state.instrument_resources_current
        != acquisition.instrument_resources_after
        or live_resources != acquisition.instrument_resources_after
        or state.instrument_current_sequence_index
        != full_observation.sequence_index
        or state.current_virtual_time_s != full_observation.timestamp_s
        or live_time_s != full_observation.timestamp_s
    ):
        _invalid_context("unavailable authoritative boundary")


def _build_two_point_evaluator_resources_from_context(
    runner: TwoPointEvaluatorRunner,
    *,
    authenticated_unaccepted: TwoPointTrackingAcquisition | None,
) -> TwoPointEvaluatorResources:
    """Assemble normal resources from one started runner context."""
    state, verified, calibration, estimate = _authenticate_runner_context(runner)
    if authenticated_unaccepted is not None and type(
        authenticated_unaccepted
    ) is not TwoPointTrackingAcquisition:
        raise TypeError(
            "authenticated_unaccepted must be a TwoPointTrackingAcquisition or None"
        )

    source = verified.source
    overhead_s = state.instrument_configuration.frequency_overhead_s
    calibration_observations = verified.full_observations
    calibration_resources = _validate_calibration_context(state, verified)
    (
        accepted_tracking_observations,
        physical_resources,
        accepted_charged_resources,
    ) = _validate_accepted_tracking_context(
        state,
        verified,
        calibration,
        estimate,
        calibration_resources,
    )
    accepted_tracking_resources = _replay_full_resources(
        accepted_tracking_observations,
        overhead_s,
    )
    zero = _zero_full_resources()
    unaccepted_tracking_observations: tuple[InstrumentObservation, ...] = ()
    unaccepted_tracking_resources = zero
    tracking_resources = accepted_tracking_resources
    charged_resources = accepted_charged_resources
    final_physical_resources = physical_resources
    if authenticated_unaccepted is not None:
        acquisition = authenticated_unaccepted
        full_observation = acquisition.full_observation
        if (
            estimate.pending_query is None
            or acquisition.query != estimate.pending_query
            or full_observation.estimator_view() != acquisition.safe_observation
            or any(acquisition is accepted for accepted in state.normal_tracking_trace)
        ):
            _invalid_context("unaccepted acquisition and pending query")
        expected_delta = _advance_full_resources(
            zero, full_observation, overhead_s
        )
        expected_after = _advance_full_resources(
            physical_resources, full_observation, overhead_s
        )
        accepted_time_s = (
            accepted_tracking_observations[-1].timestamp_s
            if accepted_tracking_observations
            else (
                source.availability_timestamp_s
                if state.run_token is verified.run_token
                else 0.0
            )
        )
        expected_midpoint_s = (
            accepted_time_s + overhead_s
        ) + acquisition.query.integration_time_s / 2.0
        _validate_unaccepted_midpoint(
            acquisition,
            expected_midpoint_s=expected_midpoint_s,
        )
        if (
            acquisition.instrument_resources_before != physical_resources
            or acquisition.instrument_resource_delta != expected_delta
            or acquisition.instrument_resources_after != expected_after
        ):
            _invalid_context("unaccepted acquisition resource or time boundary")
        if state.phase == "aborted":
            if (
                state.terminal_abort is None
                or state.terminal_abort.unaccepted_acquisition is not acquisition
            ):
                _invalid_context("terminal abort acquisition identity")
        elif state.phase != "tracking" or state.terminal_abort is not None:
            _invalid_context("unaccepted acquisition phase")
        unaccepted_tracking_observations = (full_observation,)
        unaccepted_tracking_resources = expected_delta
        tracking_resources = _advance_full_resources(
            accepted_tracking_resources, full_observation, overhead_s
        )
        charged_resources = _advance_full_resources(
            accepted_charged_resources, full_observation, overhead_s
        )
        final_physical_resources = expected_after
    try:
        live_resources = runner._instrument.resources
        live_time_s = runner._instrument.virtual_time_s
    except Exception:
        _invalid_context("live instrument boundary")
    accepted_sequence = (
        accepted_tracking_observations[-1].sequence_index
        if accepted_tracking_observations
        else (
            source.availability_sequence_index
            if state.run_token is verified.run_token
            else None
        )
    )
    accepted_time_s = (
        accepted_tracking_observations[-1].timestamp_s
        if accepted_tracking_observations
        else (
            source.availability_timestamp_s
            if state.run_token is verified.run_token
            else 0.0
        )
    )
    final_sequence = (
        authenticated_unaccepted.full_observation.sequence_index
        if authenticated_unaccepted is not None
        else accepted_sequence
    )
    final_time_s = (
        authenticated_unaccepted.full_observation.timestamp_s
        if authenticated_unaccepted is not None
        else accepted_time_s
    )
    state_has_final_atom = (
        authenticated_unaccepted is not None and state.phase == "aborted"
    )
    expected_state_resources = (
        final_physical_resources if state_has_final_atom else physical_resources
    )
    expected_state_sequence = (
        final_sequence if state_has_final_atom else accepted_sequence
    )
    expected_state_time_s = (
        final_time_s if state_has_final_atom else accepted_time_s
    )
    if (
        state.instrument_resources_current != expected_state_resources
        or live_resources != final_physical_resources
        or state.instrument_current_sequence_index != expected_state_sequence
        or state.current_virtual_time_s != expected_state_time_s
        or live_time_s != final_time_s
    ):
        _invalid_context("final instrument boundary")
    return TwoPointEvaluatorResources(
        calibration_observations=calibration_observations,
        accepted_tracking_observations=accepted_tracking_observations,
        unaccepted_tracking_observations=unaccepted_tracking_observations,
        calibration_resources=calibration_resources,
        accepted_tracking_resources=accepted_tracking_resources,
        unaccepted_tracking_resources=unaccepted_tracking_resources,
        tracking_resources=tracking_resources,
        accepted_charged_resources=accepted_charged_resources,
        charged_resources=charged_resources,
        calibration_budget_treatment=calibration.budget_treatment,
        incomplete_pair_observations=int(estimate.incomplete_pair is not None),
        unaccepted_observations=int(authenticated_unaccepted is not None),
    )
