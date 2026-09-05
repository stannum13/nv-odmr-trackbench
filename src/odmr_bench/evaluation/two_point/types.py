"""Evaluator-owned primitive contracts for calibrated two-point runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, NoReturn, TypeAlias

import numpy as np

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
