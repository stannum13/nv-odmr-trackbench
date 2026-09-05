"""Runner-private verified calibration acquisition."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Literal, cast, get_args

from odmr_bench.emulator.observations import (
    EstimatorObservation,
    InstrumentObservation,
)
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators.fitting import fit_spectrum
from odmr_bench.estimators.two_point_calibration import (
    _VERIFIED_SOURCE_CONSTRUCTION_KEY,
    _begin_verified_source_construction_transaction,
    _bind_verified_two_point_calibration_source,
    _consume_verified_source_construction_identity,
    _finish_verified_source_construction_transaction,
)
from odmr_bench.estimators.two_point_types import (
    NormalizedFluorescenceProvenance,
    TwoPointClockMapping,
    TwoPointIdentityBinding,
)
from odmr_bench.estimators.types import (
    CompleteSweep,
    FitConfiguration,
    SpectrumFitResult,
)
from odmr_bench.evaluation.two_point.provenance import (
    _bind_run_token_success,
    _lookup_run_token_binding,
    _rollback_run_token_success,
    _snapshot_run_token_binding_before_success,
)
from odmr_bench.evaluation.two_point.resource_accounting import (
    _advance_full_resources,
    _project_full_resources,
    _replay_full_resources,
    _resource_mismatch_fields,
)
from odmr_bench.evaluation.two_point.types import (
    ResourceJoinMismatchField,
    TwoPointCalibrationPreflightError,
    TwoPointEvaluatorRunnerState,
    VerifiedCalibrationFailureCode,
    VerifiedCalibrationPreflightCode,
    VerifiedCalibrationQueryRequest,
    VerifiedTwoPointCalibrationFailure,
    VerifiedTwoPointCalibrationOutcome,
    VerifiedTwoPointCalibrationSuccess,
)

if TYPE_CHECKING:
    from odmr_bench.evaluation.two_point.runner import TwoPointEvaluatorRunner

_RESOURCE_MISMATCH_FIELD_ORDER = cast(
    tuple[ResourceJoinMismatchField, ...],
    get_args(ResourceJoinMismatchField),
)
_STARTED_BOUNDARY_CAPTURE_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class _VerifiedCalibrationPlan:
    frequencies_hz: tuple[float, ...]
    integration_time_s: float
    fit_configuration: FitConfiguration
    identity_binding: TwoPointIdentityBinding
    source_id: str
    clock_mapping: TwoPointClockMapping
    requests: tuple[VerifiedCalibrationQueryRequest, ...]


@dataclass(frozen=True, slots=True)
class _InstrumentBoundary:
    resources: ResourceSnapshot
    virtual_time_s: float


def _safe_exception_strings(error: Exception) -> tuple[str, str]:
    """Render an ordinary causal exception without invoking it after snapshots."""
    try:
        exception_type = type(error).__name__
        if not isinstance(exception_type, str):
            raise TypeError("exception class name must be a string")
        exception_type = str.__str__(exception_type)
        if not exception_type:
            raise ValueError("exception class name must be nonempty")
    except Exception:
        exception_type = "Exception"
    try:
        exception_message = str.__str__(str(error))
    except Exception:
        exception_message = ""
    return exception_type, exception_message


def _capture_started_instrument_boundary(
    runner: TwoPointEvaluatorRunner,
) -> tuple[_InstrumentBoundary | None, tuple[str, str] | None]:
    """Capture one fresh pair without retrying an ordinary fault forever."""
    first_exception_strings = None
    for _ in range(_STARTED_BOUNDARY_CAPTURE_ATTEMPTS):
        try:
            boundary = _InstrumentBoundary(
                resources=runner._instrument.resources,
                virtual_time_s=runner._instrument.virtual_time_s,
            )
        except Exception as error:
            if first_exception_strings is None:
                first_exception_strings = _safe_exception_strings(error)
            continue
        return boundary, first_exception_strings
    return None, first_exception_strings


def _finish_calibration_failure(
    runner: TwoPointEvaluatorRunner,
    state_before: TwoPointEvaluatorRunnerState,
    *,
    failure_code: VerifiedCalibrationFailureCode,
    failed_request: VerifiedCalibrationQueryRequest | None,
    exception_strings: tuple[str, str] | None,
    fit_result: SpectrumFitResult | None,
    resource_mismatch_fields: tuple[ResourceJoinMismatchField, ...],
    full_observations: Sequence[InstrumentObservation],
    safe_observations: Sequence[EstimatorObservation],
    measurement_midpoints_s: Sequence[float | None],
    instrument_resources_after: ResourceSnapshot,
    current_sequence_index: int | None,
    current_virtual_time_s: float,
) -> VerifiedTwoPointCalibrationFailure:
    full_observation_trace = tuple(full_observations)
    if failure_code == "resource_join_unavailable":
        full_resources = None
        safe_resources = None
    else:
        full_resources = _replay_full_resources(
            full_observation_trace,
            state_before.instrument_configuration.frequency_overhead_s,
        )
        safe_resources = _project_full_resources(full_resources)
    outcome = VerifiedTwoPointCalibrationFailure(
        status="failure",
        run_token=state_before.run_token,
        failure_code=failure_code,
        failed_request=failed_request,
        exception_type=(
            None if exception_strings is None else exception_strings[0]
        ),
        exception_message=(
            None if exception_strings is None else exception_strings[1]
        ),
        fit_result=fit_result,
        resource_mismatch_fields=resource_mismatch_fields,
        full_observations=full_observation_trace,
        safe_observations=tuple(safe_observations),
        measurement_midpoints_s=tuple(measurement_midpoints_s),
        instrument_resources_before=state_before.instrument_resources_current,
        instrument_resources_after=instrument_resources_after,
        safe_resources=safe_resources,
        full_resources=full_resources,
    )
    runner._state = replace(
        state_before,
        phase="calibration_failed",
        calibration_outcome=outcome,
        verified_calibration=None,
        instrument_resources_current=outcome.instrument_resources_after,
        instrument_current_sequence_index=current_sequence_index,
        current_virtual_time_s=current_virtual_time_s,
    )
    return outcome


def _raise_preflight(code: VerifiedCalibrationPreflightCode) -> None:
    raise TwoPointCalibrationPreflightError(code)


def _boundary_mismatch_fields(
    expected_resources: ResourceSnapshot,
    expected_virtual_time_s: float,
    actual_resources: ResourceSnapshot,
    actual_virtual_time_s: float,
) -> tuple[ResourceJoinMismatchField, ...]:
    time_fields: tuple[ResourceJoinMismatchField, ...] = (
        ("virtual_elapsed_time_s",)
        if actual_virtual_time_s != expected_virtual_time_s
        else ()
    )
    return _ordered_resource_mismatch_union(
        _resource_mismatch_fields(expected_resources, actual_resources),
        time_fields,
    )


def _ordered_resource_mismatch_union(
    *field_groups: tuple[ResourceJoinMismatchField, ...],
) -> tuple[ResourceJoinMismatchField, ...]:
    present = {field for fields in field_groups for field in fields}
    return tuple(
        field for field in _RESOURCE_MISMATCH_FIELD_ORDER if field in present
    )


def _sequence_index_from_resources(
    resources: ResourceSnapshot,
) -> int | None:
    return None if resources.observations == 0 else resources.observations - 1


def _prepare_verified_calibration(
    runner: TwoPointEvaluatorRunner,
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
) -> _VerifiedCalibrationPlan:
    state = runner._state
    if state.phase != "ready":
        _raise_preflight("invalid_runner_phase")

    binding = _lookup_run_token_binding(state.run_token)
    if (
        binding is None
        or binding.issuer_runner is not runner
        or binding.instrument is not runner._instrument
        or binding.instrument_configuration is not state.instrument_configuration
        or binding.success is not None
        or binding.source is not None
    ):
        _raise_preflight("invalid_runner_phase")

    if (
        type(frequency_hz) not in {list, tuple}
        or type(integration_time_s) is not float
        or type(fit_configuration) is not FitConfiguration
        or type(identity_binding) is not TwoPointIdentityBinding
        or type(source_id) is not str
        or type(source_clock_id) is not str
        or type(tracker_clock_id) is not str
        or type(source_to_tracker_offset_s) is not float
        or type(physical_fit_epoch_rule) is not str
    ):
        _raise_preflight("invalid_argument_type")
    frequencies_hz = tuple(frequency_hz)
    if not all(type(value) is float for value in frequencies_hz):
        _raise_preflight("invalid_argument_type")

    if (
        not math.isfinite(integration_time_s)
        or integration_time_s <= 0.0
        or not source_id.strip()
        or physical_fit_epoch_rule != "instrument_midpoint_ordered_mean"
        or any(
            not math.isfinite(frequency) or frequency <= 0.0
            for frequency in frequencies_hz
        )
    ):
        _raise_preflight("invalid_argument_value")

    configuration = state.instrument_configuration
    requests: list[VerifiedCalibrationQueryRequest] = []
    previous_endpoint_s = state.current_virtual_time_s
    try:
        expected_nominal_exposure_photons = (
            configuration.nominal_photon_rate_hz * integration_time_s
        )
        for point_index, frequency in enumerate(frequencies_hz):
            integration_start_s = (
                previous_endpoint_s + configuration.frequency_overhead_s
            )
            midpoint_s = integration_start_s + integration_time_s / 2.0
            endpoint_s = integration_start_s + integration_time_s
            requests.append(
                VerifiedCalibrationQueryRequest(
                    point_index=point_index,
                    frequency_hz=frequency,
                    integration_time_s=integration_time_s,
                    expected_sequence_index=point_index,
                    expected_measurement_midpoint_s=midpoint_s,
                    expected_end_timestamp_s=endpoint_s,
                    expected_nominal_exposure_photons=(
                        expected_nominal_exposure_photons
                    ),
                )
            )
            previous_endpoint_s = endpoint_s
    except (ArithmeticError, TypeError, ValueError):
        _raise_preflight("invalid_argument_value")

    if len(frequencies_hz) < 2 or any(
        current <= previous
        for previous, current in pairwise(frequencies_hz)
    ):
        _raise_preflight("invalid_frequency_grid")

    try:
        fit_configuration_snapshot = replace(fit_configuration)
        identity_binding_snapshot = TwoPointIdentityBinding(
            identity_binding.mode,
            identity_binding.expected_resonance_ids,
        )
    except (TypeError, ValueError):
        _raise_preflight("invalid_fit_or_identity_configuration")
    if (
        identity_binding_snapshot.mode == "require_expected_ids"
        and identity_binding_snapshot.expected_resonance_ids
        != fit_configuration_snapshot.resonance_ids
    ):
        _raise_preflight("invalid_fit_or_identity_configuration")

    clock_kind = (
        "shared_clock"
        if source_clock_id == tracker_clock_id
        else "unit_scale_offset"
    )
    try:
        clock_mapping = TwoPointClockMapping(
            clock_kind,
            source_clock_id,
            tracker_clock_id,
            1.0,
            source_to_tracker_offset_s,
        )
    except (TypeError, ValueError):
        _raise_preflight("invalid_clock_mapping")

    first_midpoint_s = requests[0].expected_measurement_midpoint_s
    last_midpoint_s = requests[-1].expected_measurement_midpoint_s
    physical_fit_epoch_s = first_midpoint_s + (
        last_midpoint_s - first_midpoint_s
    ) / 2.0
    mapped_times = (
        requests[0].expected_end_timestamp_s + clock_mapping.offset_s,
        requests[-1].expected_end_timestamp_s + clock_mapping.offset_s,
        physical_fit_epoch_s + clock_mapping.offset_s,
    )
    if not all(math.isfinite(value) for value in mapped_times):
        _raise_preflight("invalid_clock_mapping")

    try:
        instrument_resources = runner._instrument.resources
        instrument_time_s = runner._instrument.virtual_time_s
    except Exception:
        _raise_preflight("unclean_instrument_boundary")
    if (
        instrument_resources != state.instrument_resources_current
        or instrument_time_s != state.current_virtual_time_s
        or instrument_resources != state.instrument_resources_at_bind
        or instrument_time_s != 0.0
    ):
        _raise_preflight("unclean_instrument_boundary")

    return _VerifiedCalibrationPlan(
        frequencies_hz=frequencies_hz,
        integration_time_s=integration_time_s,
        fit_configuration=fit_configuration_snapshot,
        identity_binding=identity_binding_snapshot,
        source_id=str.__str__(source_id),
        clock_mapping=clock_mapping,
        requests=tuple(requests),
    )


def _acquire_verified_calibration(
    runner: TwoPointEvaluatorRunner,
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
    plan = _prepare_verified_calibration(
        runner,
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
    instrument = runner._instrument
    state_before = runner._state
    full_observations: list[InstrumentObservation] = []
    safe_observations: list[EstimatorObservation] = []
    measurement_midpoints_s: list[float | None] = []
    instrument_resources_after = state_before.instrument_resources_current
    segment_resources = state_before.instrument_resources_current
    segment_virtual_time_s = state_before.current_virtual_time_s

    for request in plan.requests:
        boundary_before_query, boundary_before_exception_strings = (
            _capture_started_instrument_boundary(runner)
        )
        if boundary_before_query is None:
            return _finish_calibration_failure(
                runner,
                state_before,
                failure_code="resource_join_unavailable",
                failed_request=request,
                exception_strings=None,
                fit_result=None,
                resource_mismatch_fields=_RESOURCE_MISMATCH_FIELD_ORDER,
                full_observations=full_observations,
                safe_observations=safe_observations,
                measurement_midpoints_s=measurement_midpoints_s,
                instrument_resources_after=segment_resources,
                current_sequence_index=_sequence_index_from_resources(
                    segment_resources
                ),
                current_virtual_time_s=segment_virtual_time_s,
            )
        instrument_resources_before_query = boundary_before_query.resources
        continuity_mismatch_fields = _boundary_mismatch_fields(
            segment_resources,
            segment_virtual_time_s,
            instrument_resources_before_query,
            boundary_before_query.virtual_time_s,
        )
        if boundary_before_exception_strings is not None:
            return _finish_calibration_failure(
                runner,
                state_before,
                failure_code=(
                    "resource_join_unavailable"
                    if continuity_mismatch_fields
                    else "instrument_query_failed"
                ),
                failed_request=request,
                exception_strings=(
                    None
                    if continuity_mismatch_fields
                    else boundary_before_exception_strings
                ),
                fit_result=None,
                resource_mismatch_fields=continuity_mismatch_fields,
                full_observations=full_observations,
                safe_observations=safe_observations,
                measurement_midpoints_s=measurement_midpoints_s,
                instrument_resources_after=boundary_before_query.resources,
                current_sequence_index=_sequence_index_from_resources(
                    boundary_before_query.resources
                ),
                current_virtual_time_s=boundary_before_query.virtual_time_s,
            )
        try:
            observation = instrument.query(
                request.frequency_hz, request.integration_time_s
            )
        except Exception as error:
            query_exception_strings = _safe_exception_strings(error)
            boundary_after_query, _ = _capture_started_instrument_boundary(
                runner
            )
            if boundary_after_query is None:
                return _finish_calibration_failure(
                    runner,
                    state_before,
                    failure_code="resource_join_unavailable",
                    failed_request=request,
                    exception_strings=None,
                    fit_result=None,
                    resource_mismatch_fields=_RESOURCE_MISMATCH_FIELD_ORDER,
                    full_observations=full_observations,
                    safe_observations=safe_observations,
                    measurement_midpoints_s=measurement_midpoints_s,
                    instrument_resources_after=(
                        instrument_resources_before_query
                    ),
                    current_sequence_index=_sequence_index_from_resources(
                        instrument_resources_before_query
                    ),
                    current_virtual_time_s=(
                        boundary_before_query.virtual_time_s
                    ),
                )
            instrument_resources_after = boundary_after_query.resources
            instrument_time_after = boundary_after_query.virtual_time_s
            retained_resources = _replay_full_resources(
                full_observations,
                state_before.instrument_configuration.frequency_overhead_s,
            )
            retained_virtual_time_s = (
                state_before.current_virtual_time_s
                if not full_observations
                else full_observations[-1].timestamp_s
            )
            query_failure_mismatch_fields = _ordered_resource_mismatch_union(
                _boundary_mismatch_fields(
                    retained_resources,
                    retained_virtual_time_s,
                    instrument_resources_after,
                    instrument_time_after,
                ),
                continuity_mismatch_fields,
            )
            return _finish_calibration_failure(
                runner,
                state_before,
                failure_code=(
                    "resource_join_unavailable"
                    if query_failure_mismatch_fields
                    else "instrument_query_failed"
                ),
                failed_request=request,
                exception_strings=(
                    None
                    if query_failure_mismatch_fields
                    else query_exception_strings
                ),
                fit_result=None,
                resource_mismatch_fields=query_failure_mismatch_fields,
                full_observations=full_observations,
                safe_observations=safe_observations,
                measurement_midpoints_s=measurement_midpoints_s,
                instrument_resources_after=instrument_resources_after,
                current_sequence_index=_sequence_index_from_resources(
                    instrument_resources_after
                ),
                current_virtual_time_s=instrument_time_after,
            )
        boundary_after_query, boundary_after_exception_strings = (
            _capture_started_instrument_boundary(runner)
        )
        boundary_after_query_unavailable = boundary_after_query is None
        if boundary_after_query is None:
            instrument_resources_after = instrument_resources_before_query
            instrument_time_after = boundary_before_query.virtual_time_s
        else:
            instrument_resources_after = boundary_after_query.resources
            instrument_time_after = boundary_after_query.virtual_time_s
        if type(observation) is not InstrumentObservation:
            retained_resources = _replay_full_resources(
                full_observations,
                state_before.instrument_configuration.frequency_overhead_s,
            )
            retained_virtual_time_s = (
                state_before.current_virtual_time_s
                if not full_observations
                else full_observations[-1].timestamp_s
            )
            return_mismatch_fields = _ordered_resource_mismatch_union(
                (
                    _RESOURCE_MISMATCH_FIELD_ORDER
                    if boundary_after_query_unavailable
                    else ()
                ),
                _boundary_mismatch_fields(
                    retained_resources,
                    retained_virtual_time_s,
                    instrument_resources_after,
                    instrument_time_after,
                ),
                continuity_mismatch_fields,
            )
            current_sequence_index = (
                None
                if not full_observations
                else plan.requests[
                    len(full_observations) - 1
                ].expected_sequence_index
            )
            invalid_return_failure_code = (
                "resource_join_unavailable"
                if return_mismatch_fields
                else (
                    "instrument_query_failed"
                    if boundary_after_exception_strings is not None
                    else "acquisition_contract_mismatch"
                )
            )
            return _finish_calibration_failure(
                runner,
                state_before,
                failure_code=invalid_return_failure_code,
                failed_request=request,
                exception_strings=(
                    boundary_after_exception_strings
                    if invalid_return_failure_code == "instrument_query_failed"
                    else None
                ),
                fit_result=None,
                resource_mismatch_fields=return_mismatch_fields,
                full_observations=full_observations,
                safe_observations=safe_observations,
                measurement_midpoints_s=measurement_midpoints_s,
                instrument_resources_after=instrument_resources_after,
                current_sequence_index=(
                    _sequence_index_from_resources(instrument_resources_after)
                    if return_mismatch_fields
                    else current_sequence_index
                ),
                current_virtual_time_s=instrument_time_after,
            )
        safe_observation = observation.estimator_view()
        full_observations.append(observation)
        safe_observations.append(safe_observation)
        timing_matches = (
            not boundary_after_query_unavailable
            and observation.integration_time_s == request.integration_time_s
            and observation.timestamp_s == request.expected_end_timestamp_s
            and instrument_time_after == request.expected_end_timestamp_s
            and instrument_resources_after.virtual_elapsed_time_s
            == request.expected_end_timestamp_s
        )
        measurement_midpoints_s.append(
            request.expected_measurement_midpoint_s if timing_matches else None
        )
        expected_resources_after = _advance_full_resources(
            instrument_resources_before_query,
            observation,
            state_before.instrument_configuration.frequency_overhead_s,
        )
        local_resource_mismatch_fields = (
            _RESOURCE_MISMATCH_FIELD_ORDER
            if boundary_after_query_unavailable
            else _boundary_mismatch_fields(
                expected_resources_after,
                request.expected_end_timestamp_s,
                instrument_resources_after,
                instrument_time_after,
            )
        )
        resource_mismatch_fields = _ordered_resource_mismatch_union(
            local_resource_mismatch_fields,
            continuity_mismatch_fields,
        )
        if resource_mismatch_fields:
            return _finish_calibration_failure(
                runner,
                state_before,
                failure_code="resource_join_unavailable",
                failed_request=request,
                exception_strings=None,
                fit_result=None,
                resource_mismatch_fields=resource_mismatch_fields,
                full_observations=full_observations,
                safe_observations=safe_observations,
                measurement_midpoints_s=measurement_midpoints_s,
                instrument_resources_after=instrument_resources_after,
                current_sequence_index=_sequence_index_from_resources(
                    instrument_resources_after
                ),
                current_virtual_time_s=instrument_time_after,
            )
        if boundary_after_exception_strings is not None:
            return _finish_calibration_failure(
                runner,
                state_before,
                failure_code="instrument_query_failed",
                failed_request=request,
                exception_strings=boundary_after_exception_strings,
                fit_result=None,
                resource_mismatch_fields=(),
                full_observations=full_observations,
                safe_observations=safe_observations,
                measurement_midpoints_s=measurement_midpoints_s,
                instrument_resources_after=instrument_resources_after,
                current_sequence_index=request.expected_sequence_index,
                current_virtual_time_s=instrument_time_after,
            )
        segment_resources = expected_resources_after
        segment_virtual_time_s = instrument_time_after
        if (
            observation.sequence_index != request.expected_sequence_index
            or observation.frequency_hz != request.frequency_hz
            or not timing_matches
            or observation.nominal_exposure_photons
            != request.expected_nominal_exposure_photons
        ):
            return _finish_calibration_failure(
                runner,
                state_before,
                failure_code="acquisition_contract_mismatch",
                failed_request=request,
                exception_strings=None,
                fit_result=None,
                resource_mismatch_fields=(),
                full_observations=full_observations,
                safe_observations=safe_observations,
                measurement_midpoints_s=measurement_midpoints_s,
                instrument_resources_after=instrument_resources_after,
                current_sequence_index=request.expected_sequence_index,
                current_virtual_time_s=instrument_time_after,
            )

    full_observation_trace = tuple(full_observations)
    safe_observation_trace = tuple(safe_observations)
    full_resources = _replay_full_resources(
        full_observation_trace,
        state_before.instrument_configuration.frequency_overhead_s,
    )
    final_resource_mismatch_fields = _resource_mismatch_fields(
        full_resources, instrument_resources_after
    )
    if final_resource_mismatch_fields:
        return _finish_calibration_failure(
            runner,
            state_before,
            failure_code="resource_join_unavailable",
            failed_request=plan.requests[-1],
            exception_strings=None,
            fit_result=None,
            resource_mismatch_fields=final_resource_mismatch_fields,
            full_observations=full_observation_trace,
            safe_observations=safe_observation_trace,
            measurement_midpoints_s=measurement_midpoints_s,
            instrument_resources_after=instrument_resources_after,
            current_sequence_index=_sequence_index_from_resources(
                instrument_resources_after
            ),
            current_virtual_time_s=instrument_time_after,
        )
    safe_resources = _project_full_resources(full_resources)
    completed_virtual_time_s = instrument_time_after
    fit_result = None
    fit_exception_strings = None
    try:
        sweep = CompleteSweep(
            frequency_hz=tuple(
                observation.frequency_hz
                for observation in safe_observation_trace
            ),
            fluorescence=tuple(
                observation.fluorescence
                for observation in safe_observation_trace
            ),
            last_sequence_index=safe_observation_trace[-1].sequence_index,
            last_timestamp_s=safe_observation_trace[-1].timestamp_s,
            total_integration_time_s=safe_resources.integration_time_s,
            total_nominal_exposure_photons=(
                safe_resources.nominal_exposure_photons
            ),
        )
        fit_result = fit_spectrum(
            sweep,
            plan.fit_configuration,
            initial_guess=None,
        )
        if type(fit_result) is not SpectrumFitResult:
            raise TypeError(
                "fit_spectrum must return an exact SpectrumFitResult"
            )
    except Exception as error:
        fit_exception_strings = _safe_exception_strings(error)

    post_fit_boundary, post_fit_boundary_exception_strings = (
        _capture_started_instrument_boundary(runner)
    )
    if (
        fit_exception_strings is None
        and post_fit_boundary_exception_strings is not None
    ):
        fit_exception_strings = post_fit_boundary_exception_strings
    if post_fit_boundary is not None:
        instrument_resources_after = post_fit_boundary.resources
        instrument_time_after = post_fit_boundary.virtual_time_s
    post_fit_mismatch_fields = (
        _RESOURCE_MISMATCH_FIELD_ORDER
        if post_fit_boundary is None
        else _boundary_mismatch_fields(
            full_resources,
            completed_virtual_time_s,
            instrument_resources_after,
            instrument_time_after,
        )
    )
    if post_fit_mismatch_fields:
        return _finish_calibration_failure(
            runner,
            state_before,
            failure_code="resource_join_unavailable",
            failed_request=plan.requests[-1],
            exception_strings=None,
            fit_result=None,
            resource_mismatch_fields=post_fit_mismatch_fields,
            full_observations=full_observation_trace,
            safe_observations=safe_observation_trace,
            measurement_midpoints_s=measurement_midpoints_s,
            instrument_resources_after=instrument_resources_after,
            current_sequence_index=_sequence_index_from_resources(
                instrument_resources_after
            ),
            current_virtual_time_s=instrument_time_after,
        )
    if fit_exception_strings is not None:
        return _finish_calibration_failure(
            runner,
            state_before,
            failure_code="fit_exception",
            failed_request=None,
            exception_strings=fit_exception_strings,
            fit_result=None,
            resource_mismatch_fields=(),
            full_observations=full_observation_trace,
            safe_observations=safe_observation_trace,
            measurement_midpoints_s=measurement_midpoints_s,
            instrument_resources_after=instrument_resources_after,
            current_sequence_index=plan.requests[-1].expected_sequence_index,
            current_virtual_time_s=instrument_time_after,
        )
    if fit_result is None:
        raise RuntimeError("successful fit must produce an exact result")
    if not fit_result.success:
        return _finish_calibration_failure(
            runner,
            state_before,
            failure_code="fit_failed",
            failed_request=None,
            exception_strings=None,
            fit_result=fit_result,
            resource_mismatch_fields=(),
            full_observations=full_observation_trace,
            safe_observations=safe_observation_trace,
            measurement_midpoints_s=measurement_midpoints_s,
            instrument_resources_after=instrument_resources_after,
            current_sequence_index=plan.requests[-1].expected_sequence_index,
            current_virtual_time_s=instrument_time_after,
        )
    if any(value is None for value in measurement_midpoints_s):
        raise RuntimeError("successful calibration requires exact midpoints")
    successful_midpoints = tuple(
        value for value in measurement_midpoints_s if value is not None
    )
    first_midpoint_s = successful_midpoints[0]
    last_midpoint_s = successful_midpoints[-1]
    physical_fit_epoch_s = first_midpoint_s + (
        last_midpoint_s - first_midpoint_s
    ) / 2.0
    source = None
    outcome = None
    successful_state = None
    binding_before_success = None
    source_exception_strings = None
    try:
        fluorescence_provenance = NormalizedFluorescenceProvenance(
            quantity="normalized_fluorescence",
            normalization_rule="odmr_instrument_normalized_fluorescence_v1",
            nominal_photon_rate_hz=(
                state_before.instrument_configuration.nominal_photon_rate_hz
            ),
            sampling_rules=tuple(
                observation.sampling_rule
                for observation in full_observation_trace
            ),
        )
        source_owner, source_context_token = (
            _begin_verified_source_construction_transaction()
        )
        try:
            source = _bind_verified_two_point_calibration_source(
                fit_result,
                plan.fit_configuration,
                safe_observation_trace,
                plan.identity_binding,
                fluorescence_provenance,
                source_id=plan.source_id,
                source_frequency_overhead_s=(
                    state_before.instrument_configuration.frequency_overhead_s
                ),
                source_start_timestamp_s=state_before.current_virtual_time_s,
                physical_fit_epoch_s=physical_fit_epoch_s,
                availability_sequence_index=(
                    safe_observation_trace[-1].sequence_index
                ),
                availability_timestamp_s=(
                    safe_observation_trace[-1].timestamp_s
                ),
                clock_mapping=plan.clock_mapping,
                construction_key=_VERIFIED_SOURCE_CONSTRUCTION_KEY,
            )
        finally:
            _finish_verified_source_construction_transaction(
                source_owner,
                source_context_token,
                source,
            )
        outcome = VerifiedTwoPointCalibrationSuccess(
            status="success",
            run_token=state_before.run_token,
            source=source,
            full_observations=full_observation_trace,
            safe_observations=safe_observation_trace,
            measurement_midpoints_s=successful_midpoints,
            instrument_resources_before=state_before.instrument_resources_current,
            instrument_resources_after=instrument_resources_after,
            safe_resources=safe_resources,
            full_resources=full_resources,
        )
        successful_state = replace(
            state_before,
            phase="calibration_succeeded",
            calibration_outcome=outcome,
            verified_calibration=outcome,
            instrument_resources_current=instrument_resources_after,
            instrument_current_sequence_index=(
                safe_observation_trace[-1].sequence_index
            ),
            current_virtual_time_s=instrument_time_after,
        )
        binding_before_success = _snapshot_run_token_binding_before_success(
            state_before.run_token,
            runner,
            instrument,
        )
        try:
            _bind_run_token_success(
                state_before.run_token,
                runner,
                instrument,
                outcome,
            )
        except BaseException:
            _rollback_run_token_success(
                state_before.run_token,
                binding_before_success,
            )
            _consume_verified_source_construction_identity(source)
            raise
    except Exception as error:
        source_exception_strings = _safe_exception_strings(error)

    post_source_boundary, post_source_boundary_exception_strings = (
        _capture_started_instrument_boundary(runner)
    )
    if (
        source_exception_strings is None
        and post_source_boundary_exception_strings is not None
    ):
        source_exception_strings = post_source_boundary_exception_strings
    if post_source_boundary is not None:
        instrument_resources_after = post_source_boundary.resources
        instrument_time_after = post_source_boundary.virtual_time_s
    post_source_mismatch_fields = (
        _RESOURCE_MISMATCH_FIELD_ORDER
        if post_source_boundary is None
        else _boundary_mismatch_fields(
            full_resources,
            completed_virtual_time_s,
            instrument_resources_after,
            instrument_time_after,
        )
    )
    if source_exception_strings is not None or post_source_mismatch_fields:
        if binding_before_success is not None:
            _rollback_run_token_success(
                state_before.run_token,
                binding_before_success,
            )
        if source is not None:
            _consume_verified_source_construction_identity(source)
    if post_source_mismatch_fields:
        return _finish_calibration_failure(
            runner,
            state_before,
            failure_code="resource_join_unavailable",
            failed_request=plan.requests[-1],
            exception_strings=None,
            fit_result=None,
            resource_mismatch_fields=post_source_mismatch_fields,
            full_observations=full_observation_trace,
            safe_observations=safe_observation_trace,
            measurement_midpoints_s=successful_midpoints,
            instrument_resources_after=instrument_resources_after,
            current_sequence_index=_sequence_index_from_resources(
                instrument_resources_after
            ),
            current_virtual_time_s=instrument_time_after,
        )
    if source_exception_strings is not None:
        return _finish_calibration_failure(
            runner,
            state_before,
            failure_code="source_binding_failed",
            failed_request=None,
            exception_strings=source_exception_strings,
            fit_result=fit_result,
            resource_mismatch_fields=(),
            full_observations=full_observation_trace,
            safe_observations=safe_observation_trace,
            measurement_midpoints_s=successful_midpoints,
            instrument_resources_after=instrument_resources_after,
            current_sequence_index=plan.requests[-1].expected_sequence_index,
            current_virtual_time_s=instrument_time_after,
        )
    if outcome is None or successful_state is None:
        raise RuntimeError(
            "successful source binding must produce an outcome and state"
        )
    runner._state = successful_state
    return outcome
