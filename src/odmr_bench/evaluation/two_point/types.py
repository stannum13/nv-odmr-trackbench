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
    PublicAcquisitionResources,
    TwoPointCalibrationSource,
    TwoPointQuery,
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
