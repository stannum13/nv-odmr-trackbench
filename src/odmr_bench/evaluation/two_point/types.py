"""Evaluator-owned primitive contracts for calibrated two-point runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, NoReturn, TypeAlias

import numpy as np

from odmr_bench.emulator.observations import (
    EstimatorObservation,
    InstrumentObservation,
)
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators.two_point_types import (
    CalibrationBudgetTreatment,
    PublicAcquisitionResources,
    TwoPointCalibration,
    TwoPointCalibrationSource,
    TwoPointEstimate,
    TwoPointQuery,
    TwoPointUpdate,
)
from odmr_bench.estimators.types import SpectrumFitResult

VerifiedCalibrationFailureCode: TypeAlias = Literal[
    "instrument_query_failed",
    "resource_join_unavailable",
    "acquisition_contract_mismatch",
    "fit_failed",
    "fit_exception",
    "source_binding_failed",
]

VerifiedCalibrationPreflightCode: TypeAlias = Literal[
    "invalid_runner_phase",
    "invalid_argument_type",
    "invalid_argument_value",
    "invalid_frequency_grid",
    "invalid_fit_or_identity_configuration",
    "invalid_clock_mapping",
    "unclean_instrument_boundary",
]
TwoPointRunnerStartFailureCode: TypeAlias = Literal[
    "invalid_runner_phase",
    "invalid_argument_type",
    "unverified_calibration",
    "calibration_mismatch",
    "run_provenance_mismatch",
    "metadata_mismatch",
    "resource_boundary_mismatch",
    "tracker_reset_failed",
]
ResourceJoinMismatchField: TypeAlias = Literal[
    "observations",
    "integration_time_s",
    "nominal_exposure_photons",
    "expected_photons",
    "realized_photons",
    "observations_without_realized_counts",
    "virtual_elapsed_time_s",
]
TwoPointAbortReason: TypeAlias = Literal[
    "resource_join_unavailable",
    "tracker_observation_validation_error",
    "tracker_update_construction_error",
    "tracker_update_unexpected_error",
]
TwoPointRunnerPhase: TypeAlias = Literal[
    "ready",
    "calibration_succeeded",
    "calibration_failed",
    "tracking",
    "budget_stopped",
    "externally_stopped",
    "aborted",
]

_VERIFIED_CALIBRATION_PREFLIGHT_CODES = frozenset(
    {
        "invalid_runner_phase",
        "invalid_argument_type",
        "invalid_argument_value",
        "invalid_frequency_grid",
        "invalid_fit_or_identity_configuration",
        "invalid_clock_mapping",
        "unclean_instrument_boundary",
    }
)
_TWO_POINT_RUNNER_START_FAILURE_CODES = frozenset(
    {
        "invalid_runner_phase",
        "invalid_argument_type",
        "unverified_calibration",
        "calibration_mismatch",
        "run_provenance_mismatch",
        "metadata_mismatch",
        "resource_boundary_mismatch",
        "tracker_reset_failed",
    }
)
_VERIFIED_CALIBRATION_FAILURE_CODES = frozenset(
    {
        "instrument_query_failed",
        "resource_join_unavailable",
        "acquisition_contract_mismatch",
        "fit_failed",
        "fit_exception",
        "source_binding_failed",
    }
)
_RESOURCE_JOIN_MISMATCH_FIELDS = (
    "observations",
    "integration_time_s",
    "nominal_exposure_photons",
    "expected_photons",
    "realized_photons",
    "observations_without_realized_counts",
    "virtual_elapsed_time_s",
)
_TWO_POINT_ABORT_REASONS = frozenset(
    {
        "resource_join_unavailable",
        "tracker_observation_validation_error",
        "tracker_update_construction_error",
        "tracker_update_unexpected_error",
    }
)
_TWO_POINT_RUNNER_PHASES = frozenset(
    {
        "ready",
        "calibration_succeeded",
        "calibration_failed",
        "tracking",
        "budget_stopped",
        "externally_stopped",
        "aborted",
    }
)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    try:
        canonical = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    return canonical


def _positive_float(value: object, name: str) -> float:
    canonical = _finite_float(value, name)
    if canonical <= 0.0:
        raise ValueError(f"{name} must be positive")
    return canonical


def _nonnegative_float(value: object, name: str) -> float:
    canonical = _finite_float(value, name)
    if canonical < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return canonical


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    canonical = int(value)
    if canonical < 0:
        raise ValueError(f"{name} must be non-negative")
    return canonical


def _closed_error_code(code: object, allowed_codes: frozenset[str]) -> str:
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    canonical = str.__str__(code)
    if canonical not in allowed_codes:
        raise ValueError(f"unknown two-point evaluator error code: {canonical!r}")
    return canonical


def _closed_literal(value: object, name: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    canonical = str.__str__(value)
    if canonical not in allowed:
        raise ValueError(f"{name} must be one of {tuple(sorted(allowed))}")
    return canonical


def _exact_discriminator(value: object, name: str, expected: str) -> str:
    return _closed_literal(value, name, frozenset({expected}))


def _optional_exception_strings(
    exception_type: object, exception_message: object, *, required: bool
) -> tuple[str | None, str | None]:
    if not required:
        if exception_type is not None or exception_message is not None:
            raise ValueError("exception fields must both be absent")
        return None, None
    if not isinstance(exception_type, str):
        raise ValueError("exception_type must be a nonempty string")
    canonical_exception_type = str.__str__(exception_type)
    if not canonical_exception_type:
        raise ValueError("exception_type must be a nonempty string")
    if not isinstance(exception_message, str):
        raise TypeError("exception_message must be a string")
    return canonical_exception_type, str.__str__(exception_message)


def _canonical_resource_mismatch_fields(
    values: object,
) -> tuple[ResourceJoinMismatchField, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("resource_mismatch_fields must be an ordered sequence")
    supplied = tuple(values)
    if any(not isinstance(value, str) for value in supplied):
        raise TypeError("resource_mismatch_fields must contain strings")
    canonical = tuple(str.__str__(value) for value in supplied)
    if any(value not in _RESOURCE_JOIN_MISMATCH_FIELDS for value in canonical):
        raise ValueError("resource_mismatch_fields contains an unknown field")
    expected = tuple(
        value for value in _RESOURCE_JOIN_MISMATCH_FIELDS if value in canonical
    )
    if canonical != expected:
        raise ValueError(
            "resource_mismatch_fields must be unique and in declaration order"
        )
    return canonical  # type: ignore[return-value]


def _validated_observation_trace(
    full_values: object,
    safe_values: object,
    midpoint_values: object,
    *,
    allow_final_missing_midpoint: bool,
) -> tuple[
    tuple[InstrumentObservation, ...],
    tuple[EstimatorObservation, ...],
    tuple[float | None, ...],
]:
    if not isinstance(full_values, (tuple, list)):
        raise TypeError("full_observations must be an ordered sequence")
    if not isinstance(safe_values, (tuple, list)):
        raise TypeError("safe_observations must be an ordered sequence")
    if not isinstance(midpoint_values, (tuple, list)):
        raise TypeError("measurement_midpoints_s must be an ordered sequence")
    full_observations = tuple(full_values)
    safe_observations = tuple(safe_values)
    supplied_midpoints = tuple(midpoint_values)
    if not all(type(value) is InstrumentObservation for value in full_observations):
        raise TypeError("full_observations must contain InstrumentObservation values")
    if not all(type(value) is EstimatorObservation for value in safe_observations):
        raise TypeError("safe_observations must contain EstimatorObservation values")
    if not (
        len(full_observations)
        == len(safe_observations)
        == len(supplied_midpoints)
    ):
        raise ValueError("full, safe, and midpoint tuples must have equal lengths")
    projected_observations = tuple(
        value.estimator_view() for value in full_observations
    )
    if projected_observations != safe_observations:
        raise ValueError("safe observations must exactly project full observations")

    midpoints: list[float | None] = []
    for index, (full_observation, midpoint) in enumerate(
        zip(full_observations, supplied_midpoints, strict=True)
    ):
        if midpoint is None:
            if not allow_final_missing_midpoint or index != len(supplied_midpoints) - 1:
                raise ValueError("only the final failure midpoint may be unavailable")
            midpoints.append(None)
            continue
        canonical = _canonical_optional_midpoint(
            midpoint,
            endpoint_s=full_observation.timestamp_s,
            name="measurement midpoint",
        )
        if canonical is None:
            raise ValueError("measurement midpoint must be available")
        midpoints.append(canonical)
    return full_observations, safe_observations, tuple(midpoints)


def _validate_aggregate_projection(
    full_resources: object,
    safe_resources: object,
    *,
    observation_count: int,
) -> tuple[ResourceSnapshot, PublicAcquisitionResources]:
    if type(full_resources) is not ResourceSnapshot:
        raise TypeError("full_resources must be a ResourceSnapshot")
    if type(safe_resources) is not PublicAcquisitionResources:
        raise TypeError("safe_resources must be PublicAcquisitionResources")
    if full_resources.observations != observation_count:
        raise ValueError("aggregate observation count must match the trace")
    from odmr_bench.evaluation.two_point.resource_accounting import (
        _project_full_resources,
    )

    if _project_full_resources(full_resources) != safe_resources:
        raise ValueError("safe_resources must exactly project full_resources")
    return full_resources, safe_resources


def _canonical_optional_midpoint(
    value: object,
    *,
    endpoint_s: float,
    name: str,
) -> float | None:
    if value is None:
        return None
    midpoint = _nonnegative_float(value, name)
    if midpoint > endpoint_s:
        raise ValueError(f"{name} must not exceed its endpoint")
    return midpoint


def _validated_acquisition_fields(
    query: object,
    expected_measurement_midpoint_s: object,
    measurement_midpoint_s: object,
    full_observation: object,
    safe_observation: object,
    instrument_resources_before: object,
    instrument_resources_after: object,
) -> tuple[float, float | None]:
    if type(query) is not TwoPointQuery:
        raise TypeError("query must be a TwoPointQuery")
    if type(full_observation) is not InstrumentObservation:
        raise TypeError("full_observation must be an InstrumentObservation")
    if type(safe_observation) is not EstimatorObservation:
        raise TypeError("safe_observation must be an EstimatorObservation")
    if full_observation.estimator_view() != safe_observation:
        raise ValueError("safe_observation must exactly project full_observation")
    if type(instrument_resources_before) is not ResourceSnapshot:
        raise TypeError("instrument_resources_before must be a ResourceSnapshot")
    if type(instrument_resources_after) is not ResourceSnapshot:
        raise TypeError("instrument_resources_after must be a ResourceSnapshot")
    expected_midpoint = _canonical_optional_midpoint(
        expected_measurement_midpoint_s,
        endpoint_s=query.expected_end_timestamp_s,
        name="expected_measurement_midpoint_s",
    )
    if expected_midpoint is None:
        raise TypeError("expected_measurement_midpoint_s must be a real scalar")
    measurement_midpoint = _canonical_optional_midpoint(
        measurement_midpoint_s,
        endpoint_s=full_observation.timestamp_s,
        name="measurement_midpoint_s",
    )
    return expected_midpoint, measurement_midpoint


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    canonical = str.__str__(value)
    if not canonical:
        raise ValueError(f"{name} must be nonempty")
    return canonical


def _exact_record_sequence(
    values: object, name: str, record_type: type
) -> tuple[object, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be an ordered sequence")
    canonical = tuple(values)
    if not all(type(value) is record_type for value in canonical):
        raise TypeError(f"{name} must contain exact {record_type.__name__} values")
    return canonical


def _binary_count(value: object, name: str) -> int:
    canonical = _nonnegative_int(value, name)
    if canonical not in {0, 1}:
        raise ValueError(f"{name} must be zero or one")
    return canonical


def _outcome_state(
    value: object, *, phase: TwoPointRunnerPhase
) -> TwoPointEvaluatorRunnerState:
    if type(value) is not TwoPointEvaluatorRunnerState:
        raise TypeError("state must be a TwoPointEvaluatorRunnerState")
    if value.phase != phase:
        raise ValueError(f"outcome requires {phase} state")
    return value


class TwoPointCalibrationPreflightError(ValueError):
    """Reject invalid verified-calibration preflight inputs."""

    code: VerifiedCalibrationPreflightCode

    def __init__(self, code: VerifiedCalibrationPreflightCode) -> None:
        self.code = _closed_error_code(
            code, _VERIFIED_CALIBRATION_PREFLIGHT_CODES
        )
        super().__init__(self.code)


class TwoPointRunnerStartError(ValueError):
    """Reject an invalid transition into two-point tracking."""

    code: TwoPointRunnerStartFailureCode

    def __init__(self, code: TwoPointRunnerStartFailureCode) -> None:
        self.code = _closed_error_code(code, _TWO_POINT_RUNNER_START_FAILURE_CODES)
        super().__init__(self.code)


class TwoPointRunnerStateError(RuntimeError):
    """Reject an operation that is invalid for the runner's current phase."""


class VerifiedInstrumentRunToken:
    """Evaluator-owned run provenance capability."""

    __slots__ = ()

    def __new__(cls) -> NoReturn:
        raise TypeError("ordinary VerifiedInstrumentRunToken construction is disabled")

    def __init_subclass__(cls, **kwargs: object) -> NoReturn:
        del kwargs
        raise TypeError("VerifiedInstrumentRunToken cannot be subclassed")

    def __copy__(self) -> NoReturn:
        raise TypeError("VerifiedInstrumentRunToken cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("VerifiedInstrumentRunToken cannot be deep-copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("VerifiedInstrumentRunToken cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("VerifiedInstrumentRunToken cannot be serialized")


@dataclass(frozen=True, slots=True)
class TwoPointEvaluatorInstrumentConfiguration:
    """Canonical acquisition configuration derived from one instrument."""

    nominal_photon_rate_hz: float
    frequency_overhead_s: float

    def __post_init__(self) -> None:
        nominal_photon_rate_hz = _positive_float(
            self.nominal_photon_rate_hz, "nominal_photon_rate_hz"
        )
        frequency_overhead_s = _nonnegative_float(
            self.frequency_overhead_s, "frequency_overhead_s"
        )
        object.__setattr__(
            self, "nominal_photon_rate_hz", nominal_photon_rate_hz
        )
        object.__setattr__(self, "frequency_overhead_s", frequency_overhead_s)


@dataclass(frozen=True, slots=True)
class VerifiedCalibrationQueryRequest:
    """Expected contract for one verified calibration acquisition."""

    point_index: int
    frequency_hz: float
    integration_time_s: float
    expected_sequence_index: int
    expected_measurement_midpoint_s: float
    expected_end_timestamp_s: float
    expected_nominal_exposure_photons: float

    def __post_init__(self) -> None:
        point_index = _nonnegative_int(self.point_index, "point_index")
        frequency_hz = _positive_float(self.frequency_hz, "frequency_hz")
        integration_time_s = _positive_float(
            self.integration_time_s, "integration_time_s"
        )
        expected_sequence_index = _nonnegative_int(
            self.expected_sequence_index, "expected_sequence_index"
        )
        expected_measurement_midpoint_s = _nonnegative_float(
            self.expected_measurement_midpoint_s,
            "expected_measurement_midpoint_s",
        )
        expected_end_timestamp_s = _nonnegative_float(
            self.expected_end_timestamp_s, "expected_end_timestamp_s"
        )
        expected_nominal_exposure_photons = _nonnegative_float(
            self.expected_nominal_exposure_photons,
            "expected_nominal_exposure_photons",
        )
        if expected_measurement_midpoint_s > expected_end_timestamp_s:
            raise ValueError("expected measurement midpoint must not exceed endpoint")
        if expected_end_timestamp_s < integration_time_s:
            raise ValueError("expected endpoint must include integration time")
        object.__setattr__(self, "point_index", point_index)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "integration_time_s", integration_time_s)
        object.__setattr__(
            self, "expected_sequence_index", expected_sequence_index
        )
        object.__setattr__(
            self,
            "expected_measurement_midpoint_s",
            expected_measurement_midpoint_s,
        )
        object.__setattr__(
            self, "expected_end_timestamp_s", expected_end_timestamp_s
        )
        object.__setattr__(
            self,
            "expected_nominal_exposure_photons",
            expected_nominal_exposure_photons,
        )


@dataclass(frozen=True, slots=True)
class VerifiedTwoPointCalibrationSuccess:
    """Lossless successful verified-calibration acquisition."""

    status: Literal["success"]
    run_token: VerifiedInstrumentRunToken
    source: TwoPointCalibrationSource
    full_observations: tuple[InstrumentObservation, ...]
    safe_observations: tuple[EstimatorObservation, ...]
    measurement_midpoints_s: tuple[float, ...]
    instrument_resources_before: ResourceSnapshot
    instrument_resources_after: ResourceSnapshot
    safe_resources: PublicAcquisitionResources
    full_resources: ResourceSnapshot

    def __post_init__(self) -> None:
        status = _exact_discriminator(self.status, "status", "success")
        if type(self.run_token) is not VerifiedInstrumentRunToken:
            raise TypeError("run_token must be a VerifiedInstrumentRunToken")
        if type(self.source) is not TwoPointCalibrationSource:
            raise TypeError("source must be a TwoPointCalibrationSource")
        full_observations, safe_observations, midpoint_values = (
            _validated_observation_trace(
                self.full_observations,
                self.safe_observations,
                self.measurement_midpoints_s,
                allow_final_missing_midpoint=False,
            )
        )
        measurement_midpoints_s = tuple(
            value for value in midpoint_values if value is not None
        )
        if type(self.instrument_resources_before) is not ResourceSnapshot:
            raise TypeError("instrument_resources_before must be a ResourceSnapshot")
        if type(self.instrument_resources_after) is not ResourceSnapshot:
            raise TypeError("instrument_resources_after must be a ResourceSnapshot")
        full_resources, safe_resources = _validate_aggregate_projection(
            self.full_resources,
            self.safe_resources,
            observation_count=len(full_observations),
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "full_observations", full_observations)
        object.__setattr__(self, "safe_observations", safe_observations)
        object.__setattr__(
            self, "measurement_midpoints_s", measurement_midpoints_s
        )
        object.__setattr__(self, "safe_resources", safe_resources)
        object.__setattr__(self, "full_resources", full_resources)


@dataclass(frozen=True, slots=True)
class VerifiedTwoPointCalibrationFailure:
    """Lossless typed failure after verified acquisition has started."""

    status: Literal["failure"]
    run_token: VerifiedInstrumentRunToken
    failure_code: VerifiedCalibrationFailureCode
    failed_request: VerifiedCalibrationQueryRequest | None
    exception_type: str | None
    exception_message: str | None
    fit_result: SpectrumFitResult | None
    resource_mismatch_fields: tuple[ResourceJoinMismatchField, ...]
    full_observations: tuple[InstrumentObservation, ...]
    safe_observations: tuple[EstimatorObservation, ...]
    measurement_midpoints_s: tuple[float | None, ...]
    instrument_resources_before: ResourceSnapshot
    instrument_resources_after: ResourceSnapshot
    safe_resources: PublicAcquisitionResources | None
    full_resources: ResourceSnapshot | None

    def __post_init__(self) -> None:
        status = _exact_discriminator(self.status, "status", "failure")
        if type(self.run_token) is not VerifiedInstrumentRunToken:
            raise TypeError("run_token must be a VerifiedInstrumentRunToken")
        failure_code = _closed_literal(
            self.failure_code,
            "failure_code",
            _VERIFIED_CALIBRATION_FAILURE_CODES,
        )
        request_required = failure_code in {
            "instrument_query_failed",
            "resource_join_unavailable",
            "acquisition_contract_mismatch",
        }
        if request_required != (self.failed_request is not None):
            raise ValueError("failed_request presence must match failure_code")
        if self.failed_request is not None and type(
            self.failed_request
        ) is not VerifiedCalibrationQueryRequest:
            raise TypeError("failed_request must be a VerifiedCalibrationQueryRequest")
        exception_required = failure_code in {
            "instrument_query_failed",
            "fit_exception",
            "source_binding_failed",
        }
        exception_type, exception_message = _optional_exception_strings(
            self.exception_type,
            self.exception_message,
            required=exception_required,
        )
        fit_required = failure_code in {"fit_failed", "source_binding_failed"}
        if fit_required != (self.fit_result is not None):
            raise ValueError("fit_result presence must match failure_code")
        if self.fit_result is not None:
            if type(self.fit_result) is not SpectrumFitResult:
                raise TypeError("fit_result must be a SpectrumFitResult")
            if self.fit_result.success != (failure_code == "source_binding_failed"):
                raise ValueError("fit_result success must match failure_code")
        resource_mismatch_fields = _canonical_resource_mismatch_fields(
            self.resource_mismatch_fields
        )
        mismatch_required = failure_code == "resource_join_unavailable"
        if mismatch_required != bool(resource_mismatch_fields):
            raise ValueError(
                "resource_mismatch_fields presence must match failure_code"
            )
        full_observations, safe_observations, measurement_midpoints_s = (
            _validated_observation_trace(
                self.full_observations,
                self.safe_observations,
                self.measurement_midpoints_s,
                allow_final_missing_midpoint=failure_code
                in {
                    "resource_join_unavailable",
                    "acquisition_contract_mismatch",
                },
            )
        )
        if type(self.instrument_resources_before) is not ResourceSnapshot:
            raise TypeError("instrument_resources_before must be a ResourceSnapshot")
        if type(self.instrument_resources_after) is not ResourceSnapshot:
            raise TypeError("instrument_resources_after must be a ResourceSnapshot")
        if mismatch_required:
            if self.safe_resources is not None or self.full_resources is not None:
                raise ValueError(
                    "resource_join_unavailable requires absent aggregate resources"
                )
            full_resources = None
            safe_resources = None
        else:
            full_resources, safe_resources = _validate_aggregate_projection(
                self.full_resources,
                self.safe_resources,
                observation_count=len(full_observations),
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "failure_code", failure_code)
        object.__setattr__(self, "exception_type", exception_type)
        object.__setattr__(self, "exception_message", exception_message)
        object.__setattr__(
            self, "resource_mismatch_fields", resource_mismatch_fields
        )
        object.__setattr__(self, "full_observations", full_observations)
        object.__setattr__(self, "safe_observations", safe_observations)
        object.__setattr__(
            self, "measurement_midpoints_s", measurement_midpoints_s
        )
        object.__setattr__(self, "safe_resources", safe_resources)
        object.__setattr__(self, "full_resources", full_resources)


VerifiedTwoPointCalibrationOutcome: TypeAlias = (
    VerifiedTwoPointCalibrationSuccess | VerifiedTwoPointCalibrationFailure
)


@dataclass(frozen=True, slots=True)
class TwoPointTrackingAcquisition:
    """One full evaluator acquisition with an authenticated resource atom."""

    resource_join_status: Literal["authenticated"]
    query: TwoPointQuery
    expected_measurement_midpoint_s: float
    measurement_midpoint_s: float | None
    full_observation: InstrumentObservation
    safe_observation: EstimatorObservation
    instrument_resources_before: ResourceSnapshot
    instrument_resources_after: ResourceSnapshot
    instrument_resource_delta: ResourceSnapshot

    def __post_init__(self) -> None:
        resource_join_status = _exact_discriminator(
            self.resource_join_status, "resource_join_status", "authenticated"
        )
        expected_midpoint, measurement_midpoint = _validated_acquisition_fields(
            self.query,
            self.expected_measurement_midpoint_s,
            self.measurement_midpoint_s,
            self.full_observation,
            self.safe_observation,
            self.instrument_resources_before,
            self.instrument_resources_after,
        )
        if type(self.instrument_resource_delta) is not ResourceSnapshot:
            raise TypeError("instrument_resource_delta must be a ResourceSnapshot")
        overhead_s = (
            self.instrument_resource_delta.virtual_elapsed_time_s
            - self.full_observation.integration_time_s
        )
        if overhead_s < 0.0:
            raise ValueError("instrument_resource_delta must include integration time")
        from odmr_bench.evaluation.two_point.resource_accounting import (
            _advance_full_resources,
            _resource_mismatch_fields,
        )

        zero_resources = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
        expected_delta = _advance_full_resources(
            zero_resources, self.full_observation, overhead_s
        )
        if self.instrument_resource_delta != expected_delta:
            raise ValueError(
                "instrument_resource_delta must be the exact one-observation atom"
            )
        expected_after = _advance_full_resources(
            self.instrument_resources_before, self.full_observation, overhead_s
        )
        if _resource_mismatch_fields(
            expected_after, self.instrument_resources_after
        ):
            raise ValueError(
                "instrument_resources_after must exactly join the acquisition atom"
            )
        object.__setattr__(self, "resource_join_status", resource_join_status)
        object.__setattr__(
            self, "expected_measurement_midpoint_s", expected_midpoint
        )
        object.__setattr__(self, "measurement_midpoint_s", measurement_midpoint)


@dataclass(frozen=True, slots=True)
class TwoPointResourceJoinUnavailableAcquisition:
    """One raw evaluator acquisition whose resource atom cannot be joined."""

    resource_join_status: Literal["unavailable"]
    query: TwoPointQuery
    expected_measurement_midpoint_s: float
    measurement_midpoint_s: float | None
    full_observation: InstrumentObservation
    safe_observation: EstimatorObservation
    resource_mismatch_fields: tuple[ResourceJoinMismatchField, ...]
    instrument_resources_before: ResourceSnapshot
    instrument_resources_after: ResourceSnapshot

    def __post_init__(self) -> None:
        resource_join_status = _exact_discriminator(
            self.resource_join_status, "resource_join_status", "unavailable"
        )
        expected_midpoint, measurement_midpoint = _validated_acquisition_fields(
            self.query,
            self.expected_measurement_midpoint_s,
            self.measurement_midpoint_s,
            self.full_observation,
            self.safe_observation,
            self.instrument_resources_before,
            self.instrument_resources_after,
        )
        resource_mismatch_fields = _canonical_resource_mismatch_fields(
            self.resource_mismatch_fields
        )
        if not resource_mismatch_fields:
            raise ValueError(
                "unavailable acquisition requires resource_mismatch_fields"
            )
        object.__setattr__(self, "resource_join_status", resource_join_status)
        object.__setattr__(
            self, "expected_measurement_midpoint_s", expected_midpoint
        )
        object.__setattr__(self, "measurement_midpoint_s", measurement_midpoint)
        object.__setattr__(
            self, "resource_mismatch_fields", resource_mismatch_fields
        )


@dataclass(frozen=True, slots=True)
class TwoPointEvaluatorPairTiming:
    """Evaluator-only physical and public timing for one completed pair."""

    pair_index: int
    resonance_id: str
    first_measurement_midpoint_s: float
    second_measurement_midpoint_s: float
    truth_reference_timestamp_s: float
    public_reference_timestamp_s: float
    release_sequence_index: int
    release_timestamp_s: float

    def __post_init__(self) -> None:
        pair_index = _nonnegative_int(self.pair_index, "pair_index")
        resonance_id = _nonempty_string(self.resonance_id, "resonance_id")
        first_midpoint = _nonnegative_float(
            self.first_measurement_midpoint_s,
            "first_measurement_midpoint_s",
        )
        second_midpoint = _nonnegative_float(
            self.second_measurement_midpoint_s,
            "second_measurement_midpoint_s",
        )
        if second_midpoint <= first_midpoint:
            raise ValueError("second measurement midpoint must follow the first")
        truth_reference = _nonnegative_float(
            self.truth_reference_timestamp_s,
            "truth_reference_timestamp_s",
        )
        expected_truth_reference = first_midpoint + (
            second_midpoint - first_midpoint
        ) / 2.0
        if truth_reference != expected_truth_reference:
            raise ValueError(
                "truth_reference_timestamp_s must be the ordered midpoint mean"
            )
        public_reference = _nonnegative_float(
            self.public_reference_timestamp_s,
            "public_reference_timestamp_s",
        )
        release_sequence_index = _nonnegative_int(
            self.release_sequence_index, "release_sequence_index"
        )
        release_timestamp = _nonnegative_float(
            self.release_timestamp_s, "release_timestamp_s"
        )
        if release_timestamp < second_midpoint:
            raise ValueError("release timestamp must not precede the second midpoint")
        if release_timestamp < public_reference:
            raise ValueError("release timestamp must not precede the public reference")
        object.__setattr__(self, "pair_index", pair_index)
        object.__setattr__(self, "resonance_id", resonance_id)
        object.__setattr__(self, "first_measurement_midpoint_s", first_midpoint)
        object.__setattr__(self, "second_measurement_midpoint_s", second_midpoint)
        object.__setattr__(
            self, "truth_reference_timestamp_s", truth_reference
        )
        object.__setattr__(
            self, "public_reference_timestamp_s", public_reference
        )
        object.__setattr__(
            self, "release_sequence_index", release_sequence_index
        )
        object.__setattr__(self, "release_timestamp_s", release_timestamp)


@dataclass(frozen=True, slots=True)
class TwoPointInstrumentQueryFailure:
    """One atomic instrument query exception and its unchanged boundary."""

    query: TwoPointQuery
    exception_type: str
    exception_message: str
    instrument_resources_before: ResourceSnapshot
    instrument_resources_after: ResourceSnapshot

    def __post_init__(self) -> None:
        if type(self.query) is not TwoPointQuery:
            raise TypeError("query must be a TwoPointQuery")
        exception_type = _nonempty_string(self.exception_type, "exception_type")
        if not isinstance(self.exception_message, str):
            raise TypeError("exception_message must be a string")
        exception_message = str.__str__(self.exception_message)
        if type(self.instrument_resources_before) is not ResourceSnapshot:
            raise TypeError("instrument_resources_before must be a ResourceSnapshot")
        if type(self.instrument_resources_after) is not ResourceSnapshot:
            raise TypeError("instrument_resources_after must be a ResourceSnapshot")
        if self.instrument_resources_before != self.instrument_resources_after:
            raise ValueError(
                "instrument query failure must preserve its resource boundary"
            )
        object.__setattr__(self, "exception_type", exception_type)
        object.__setattr__(self, "exception_message", exception_message)


@dataclass(frozen=True, slots=True)
class TwoPointEvaluatorResources:
    """Lossless evaluator-owned resources for one tracking run."""

    calibration_observations: tuple[InstrumentObservation, ...]
    accepted_tracking_observations: tuple[InstrumentObservation, ...]
    unaccepted_tracking_observations: tuple[InstrumentObservation, ...]
    calibration_resources: ResourceSnapshot
    accepted_tracking_resources: ResourceSnapshot
    unaccepted_tracking_resources: ResourceSnapshot
    tracking_resources: ResourceSnapshot
    accepted_charged_resources: ResourceSnapshot
    charged_resources: ResourceSnapshot
    calibration_budget_treatment: CalibrationBudgetTreatment
    incomplete_pair_observations: Literal[0, 1]
    unaccepted_observations: Literal[0, 1]

    def __post_init__(self) -> None:
        observation_fields = (
            "calibration_observations",
            "accepted_tracking_observations",
            "unaccepted_tracking_observations",
        )
        for name in observation_fields:
            object.__setattr__(
                self,
                name,
                _exact_record_sequence(
                    getattr(self, name), name, InstrumentObservation
                ),
            )
        resource_fields = (
            "calibration_resources",
            "accepted_tracking_resources",
            "unaccepted_tracking_resources",
            "tracking_resources",
            "accepted_charged_resources",
            "charged_resources",
        )
        for name in resource_fields:
            if type(getattr(self, name)) is not ResourceSnapshot:
                raise TypeError(f"{name} must be a ResourceSnapshot")
        calibration_budget_treatment = _closed_literal(
            self.calibration_budget_treatment,
            "calibration_budget_treatment",
            frozenset(
                {"included_same_run", "conditional_free_precalibration"}
            ),
        )
        incomplete_pair_observations = _binary_count(
            self.incomplete_pair_observations, "incomplete_pair_observations"
        )
        unaccepted_observations = _binary_count(
            self.unaccepted_observations, "unaccepted_observations"
        )
        object.__setattr__(
            self, "calibration_budget_treatment", calibration_budget_treatment
        )
        object.__setattr__(
            self, "incomplete_pair_observations", incomplete_pair_observations
        )
        object.__setattr__(self, "unaccepted_observations", unaccepted_observations)


@dataclass(frozen=True, slots=True)
class TwoPointAbortedRun:
    """Terminal evidence for one returned but unaccepted acquisition."""

    reason: TwoPointAbortReason
    exception_type: str | None
    exception_message: str | None
    unaccepted_acquisition: (
        TwoPointTrackingAcquisition
        | TwoPointResourceJoinUnavailableAcquisition
    )
    unaccepted_observation_count: Literal[1]
    tracker_estimate_before: TwoPointEstimate
    tracker_estimate_after: TwoPointEstimate

    def __post_init__(self) -> None:
        reason = _closed_literal(
            self.reason, "reason", _TWO_POINT_ABORT_REASONS
        )
        unavailable = reason == "resource_join_unavailable"
        expected_acquisition_type = (
            TwoPointResourceJoinUnavailableAcquisition
            if unavailable
            else TwoPointTrackingAcquisition
        )
        if type(self.unaccepted_acquisition) is not expected_acquisition_type:
            raise ValueError("unaccepted acquisition must match abort reason")
        exception_type, exception_message = _optional_exception_strings(
            self.exception_type,
            self.exception_message,
            required=not unavailable,
        )
        unaccepted_observation_count = _nonnegative_int(
            self.unaccepted_observation_count, "unaccepted_observation_count"
        )
        if unaccepted_observation_count != 1:
            raise ValueError("unaccepted_observation_count must be one")
        if type(self.tracker_estimate_before) is not TwoPointEstimate:
            raise TypeError("tracker_estimate_before must be a TwoPointEstimate")
        if type(self.tracker_estimate_after) is not TwoPointEstimate:
            raise TypeError("tracker_estimate_after must be a TwoPointEstimate")
        if self.tracker_estimate_before != self.tracker_estimate_after:
            raise ValueError("tracker estimates before and after abort must be equal")
        if (
            self.tracker_estimate_before.pending_query is None
            or self.tracker_estimate_before.pending_query
            != self.unaccepted_acquisition.query
        ):
            raise ValueError(
                "abort tracker estimates must retain the unaccepted pending query"
            )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "exception_type", exception_type)
        object.__setattr__(self, "exception_message", exception_message)
        object.__setattr__(
            self, "unaccepted_observation_count", unaccepted_observation_count
        )


@dataclass(frozen=True, slots=True)
class TwoPointEvaluatorRunnerState:
    """Frozen audit snapshot of the evaluator runner state machine."""

    phase: TwoPointRunnerPhase
    run_token: VerifiedInstrumentRunToken
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration
    calibration_outcome: VerifiedTwoPointCalibrationOutcome | None
    verified_calibration: VerifiedTwoPointCalibrationSuccess | None
    calibration: TwoPointCalibration | None
    tracker_estimate: TwoPointEstimate | None
    normal_tracking_trace: tuple[TwoPointTrackingAcquisition, ...]
    pair_timings: tuple[TwoPointEvaluatorPairTiming, ...]
    instrument_resources_at_bind: ResourceSnapshot
    tracking_resources_before: ResourceSnapshot | None
    instrument_resources_current: ResourceSnapshot
    instrument_current_sequence_index: int | None
    current_virtual_time_s: float
    last_instrument_failure: TwoPointInstrumentQueryFailure | None
    terminal_abort: TwoPointAbortedRun | None

    def __post_init__(self) -> None:
        phase = _closed_literal(self.phase, "phase", _TWO_POINT_RUNNER_PHASES)
        if type(self.run_token) is not VerifiedInstrumentRunToken:
            raise TypeError("run_token must be a VerifiedInstrumentRunToken")
        if type(
            self.instrument_configuration
        ) is not TwoPointEvaluatorInstrumentConfiguration:
            raise TypeError(
                "instrument_configuration must be a "
                "TwoPointEvaluatorInstrumentConfiguration"
            )
        if self.calibration_outcome is not None and type(
            self.calibration_outcome
        ) not in {
            VerifiedTwoPointCalibrationSuccess,
            VerifiedTwoPointCalibrationFailure,
        }:
            raise TypeError("calibration_outcome has an invalid type")
        if self.verified_calibration is not None and type(
            self.verified_calibration
        ) is not VerifiedTwoPointCalibrationSuccess:
            raise TypeError(
                "verified_calibration must be a VerifiedTwoPointCalibrationSuccess"
            )
        if self.calibration is not None and type(
            self.calibration
        ) is not TwoPointCalibration:
            raise TypeError("calibration must be a TwoPointCalibration")
        if self.tracker_estimate is not None and type(
            self.tracker_estimate
        ) is not TwoPointEstimate:
            raise TypeError("tracker_estimate must be a TwoPointEstimate")
        normal_tracking_trace = _exact_record_sequence(
            self.normal_tracking_trace,
            "normal_tracking_trace",
            TwoPointTrackingAcquisition,
        )
        pair_timings = _exact_record_sequence(
            self.pair_timings, "pair_timings", TwoPointEvaluatorPairTiming
        )
        if type(self.instrument_resources_at_bind) is not ResourceSnapshot:
            raise TypeError("instrument_resources_at_bind must be a ResourceSnapshot")
        if self.tracking_resources_before is not None and type(
            self.tracking_resources_before
        ) is not ResourceSnapshot:
            raise TypeError("tracking_resources_before must be a ResourceSnapshot")
        if type(self.instrument_resources_current) is not ResourceSnapshot:
            raise TypeError("instrument_resources_current must be a ResourceSnapshot")
        instrument_current_sequence_index = (
            None
            if self.instrument_current_sequence_index is None
            else _nonnegative_int(
                self.instrument_current_sequence_index,
                "instrument_current_sequence_index",
            )
        )
        current_virtual_time_s = _nonnegative_float(
            self.current_virtual_time_s, "current_virtual_time_s"
        )
        if self.last_instrument_failure is not None and type(
            self.last_instrument_failure
        ) is not TwoPointInstrumentQueryFailure:
            raise TypeError(
                "last_instrument_failure must be a TwoPointInstrumentQueryFailure"
            )
        if self.terminal_abort is not None and type(
            self.terminal_abort
        ) is not TwoPointAbortedRun:
            raise TypeError("terminal_abort must be a TwoPointAbortedRun")

        active_phase = phase in {
            "tracking",
            "budget_stopped",
            "externally_stopped",
            "aborted",
        }
        if phase == "ready":
            if any(
                value is not None
                for value in (
                    self.calibration_outcome,
                    self.verified_calibration,
                    self.calibration,
                    self.tracker_estimate,
                    self.tracking_resources_before,
                )
            ):
                raise ValueError("ready phase must not contain run state")
        elif phase == "calibration_failed":
            if type(
                self.calibration_outcome
            ) is not VerifiedTwoPointCalibrationFailure or any(
                value is not None
                for value in (
                    self.verified_calibration,
                    self.calibration,
                    self.tracker_estimate,
                    self.tracking_resources_before,
                )
            ):
                raise ValueError(
                    "calibration_failed phase requires only one failure outcome"
                )
        elif phase == "calibration_succeeded":
            if (
                type(self.calibration_outcome)
                is not VerifiedTwoPointCalibrationSuccess
                or self.verified_calibration is not self.calibration_outcome
                or any(
                    value is not None
                    for value in (
                        self.calibration,
                        self.tracker_estimate,
                        self.tracking_resources_before,
                    )
                )
            ):
                raise ValueError(
                    "calibration_succeeded phase requires the exact success outcome"
                )
        elif active_phase:
            if (
                type(self.verified_calibration)
                is not VerifiedTwoPointCalibrationSuccess
                or type(self.calibration) is not TwoPointCalibration
                or type(self.tracker_estimate) is not TwoPointEstimate
                or type(self.tracking_resources_before) is not ResourceSnapshot
            ):
                raise ValueError("active runner phases require tracking state")
            if self.calibration_outcome is not None and (
                type(self.calibration_outcome)
                is not VerifiedTwoPointCalibrationSuccess
                or self.calibration_outcome is not self.verified_calibration
            ):
                raise ValueError(
                    "stored calibration outcome must be the exact verified success"
                )
            if (
                len(normal_tracking_trace)
                != self.tracker_estimate.accepted_observations
            ):
                raise ValueError(
                    "normal tracking trace must match accepted observation count"
                )
            if len(pair_timings) != self.tracker_estimate.completed_pairs:
                raise ValueError("pair timings must match completed pair history")
            stopped = self.tracker_estimate.stopped_reason is not None
            if stopped != (phase == "budget_stopped"):
                raise ValueError("budget-stopped phase must match tracker stop state")

        if not active_phase and (normal_tracking_trace or pair_timings):
            raise ValueError("pre-tracking phases must not contain tracking history")
        if (self.terminal_abort is not None) != (phase == "aborted"):
            raise ValueError("terminal_abort must be present exactly when aborted")
        if self.last_instrument_failure is not None and phase not in {
            "tracking",
            "externally_stopped",
        }:
            raise ValueError(
                "instrument failure may appear only while tracking or "
                "externally stopped"
            )
        if self.last_instrument_failure is not None and (
            self.tracker_estimate is None
            or self.tracker_estimate.pending_query
            != self.last_instrument_failure.query
        ):
            raise ValueError(
                "instrument failure must match the retained pending query"
            )
        if phase == "aborted" and (
            self.terminal_abort is None
            or self.terminal_abort.tracker_estimate_after
            != self.tracker_estimate
        ):
            raise ValueError("terminal abort must match the runner estimate")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "normal_tracking_trace", normal_tracking_trace)
        object.__setattr__(self, "pair_timings", pair_timings)
        object.__setattr__(
            self,
            "instrument_current_sequence_index",
            instrument_current_sequence_index,
        )
        object.__setattr__(self, "current_virtual_time_s", current_virtual_time_s)


@dataclass(frozen=True, slots=True)
class TwoPointRunnerAccepted:
    """One acquisition accepted by the tracker."""

    kind: Literal["accepted"]
    acquisition: TwoPointTrackingAcquisition
    update: TwoPointUpdate
    state: TwoPointEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(self.kind, "kind", "accepted")
        if type(self.acquisition) is not TwoPointTrackingAcquisition:
            raise TypeError("acquisition must be a TwoPointTrackingAcquisition")
        if type(self.update) is not TwoPointUpdate:
            raise TypeError("update must be a TwoPointUpdate")
        state = _outcome_state(self.state, phase="tracking")
        if (
            self.acquisition.query != self.update.query
            or self.acquisition.safe_observation != self.update.observation
            or state.tracker_estimate is not self.update.estimate
            or not state.normal_tracking_trace
            or state.normal_tracking_trace[-1] is not self.acquisition
        ):
            raise ValueError("accepted outcome must match its tracking state")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class TwoPointRunnerInstrumentFailure:
    """One atomic instrument query failure."""

    kind: Literal["instrument_failure"]
    failure: TwoPointInstrumentQueryFailure
    state: TwoPointEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(
            self.kind, "kind", "instrument_failure"
        )
        if type(self.failure) is not TwoPointInstrumentQueryFailure:
            raise TypeError("failure must be a TwoPointInstrumentQueryFailure")
        state = _outcome_state(self.state, phase="tracking")
        if state.last_instrument_failure is not self.failure:
            raise ValueError("instrument failure outcome must match tracking state")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class TwoPointRunnerBudgetStopped:
    """Terminal outcome at an unaffordable pair boundary."""

    kind: Literal["budget_stopped"]
    resources: TwoPointEvaluatorResources
    state: TwoPointEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(self.kind, "kind", "budget_stopped")
        if type(self.resources) is not TwoPointEvaluatorResources:
            raise TypeError("resources must be TwoPointEvaluatorResources")
        _outcome_state(self.state, phase="budget_stopped")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class TwoPointRunnerExternallyStopped:
    """Terminal outcome requested by the caller."""

    kind: Literal["externally_stopped"]
    resources: TwoPointEvaluatorResources
    state: TwoPointEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(
            self.kind, "kind", "externally_stopped"
        )
        if type(self.resources) is not TwoPointEvaluatorResources:
            raise TypeError("resources must be TwoPointEvaluatorResources")
        _outcome_state(self.state, phase="externally_stopped")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class TwoPointRunnerAborted:
    """Terminal outcome after a returned acquisition cannot be accepted."""

    kind: Literal["aborted"]
    abort: TwoPointAbortedRun
    resources: TwoPointEvaluatorResources | None
    state: TwoPointEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(self.kind, "kind", "aborted")
        if type(self.abort) is not TwoPointAbortedRun:
            raise TypeError("abort must be a TwoPointAbortedRun")
        if self.resources is not None and type(
            self.resources
        ) is not TwoPointEvaluatorResources:
            raise TypeError("resources must be TwoPointEvaluatorResources or None")
        state = _outcome_state(self.state, phase="aborted")
        unavailable = type(
            self.abort.unaccepted_acquisition
        ) is TwoPointResourceJoinUnavailableAcquisition
        if (
            state.terminal_abort is not self.abort
            or unavailable != (self.resources is None)
        ):
            raise ValueError("aborted outcome must match its terminal state")
        object.__setattr__(self, "kind", kind)


TwoPointRunnerStepOutcome: TypeAlias = (
    TwoPointRunnerAccepted
    | TwoPointRunnerInstrumentFailure
    | TwoPointRunnerBudgetStopped
    | TwoPointRunnerAborted
)
TwoPointRunnerRunOutcome: TypeAlias = (
    TwoPointRunnerInstrumentFailure
    | TwoPointRunnerBudgetStopped
    | TwoPointRunnerAborted
)
