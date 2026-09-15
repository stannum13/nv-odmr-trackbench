"""Public primitive contracts for sparse five-point linewidth tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np

SparseLinewidthFailureCode: TypeAlias = Literal[
    "model_evaluation_failed",
    "optimizer_failed",
    "nonfinite_solution",
    "bounds_active",
    "rank_deficient",
    "ill_conditioned",
    "amplitude_unresolved",
    "residual_quality_failed",
]
CompositeMode: TypeAlias = Literal["fast_pair", "sparse_scan"]
SparseLinewidthSourceKind: TypeAlias = Literal["calibration", "scan"]
CompositeStopReason: TypeAlias = Literal[
    "budget_exhausted", "sparse_geometry_unavailable"
]
SparseGeometryFailureCode: TypeAlias = Literal[
    "nonrepresentable_frequency_lower",
    "nonrepresentable_frequency_upper",
    "empty_fit_bounds",
    "calibration_cell_violation",
    "source_domain_violation",
]
SparseResetFailureCode: TypeAlias = Literal[
    "invalid_argument_type",
    "configuration_mismatch",
    "calibration_mismatch",
    "metadata_mismatch",
    "invalid_base_sparse_geometry",
    "budget_mismatch",
    "initial_state_construction_failed",
]
SparseObservationValidationCode: TypeAlias = Literal[
    "invalid_observation_type",
    "no_pending_query",
    "pending_mode_mismatch",
    "fast_query_echo_mismatch",
    "sparse_query_echo_mismatch",
    "sequence_mismatch",
    "frequency_mismatch",
    "integration_time_mismatch",
    "endpoint_mismatch",
    "nominal_exposure_mismatch",
    "invalid_observation_value",
]
SparseUpdateConstructionCode: TypeAlias = Literal[
    "fast_partial_pair_construction_failed",
    "fast_pair_result_construction_failed",
    "fast_identity_estimate_construction_failed",
    "sparse_partial_scan_construction_failed",
    "sparse_scan_result_construction_failed",
    "sparse_identity_estimate_construction_failed",
    "resource_construction_failed",
    "aggregate_estimate_construction_failed",
    "update_construction_failed",
]


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


def _nonnegative_float(value: object, name: str) -> float:
    canonical = _finite_float(value, name)
    if canonical < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return canonical


def _positive_float(value: object, name: str) -> float:
    canonical = _finite_float(value, name)
    if canonical <= 0.0:
        raise ValueError(f"{name} must be positive")
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


def _positive_int(value: object, name: str) -> int:
    canonical = _nonnegative_int(value, name)
    if canonical == 0:
        raise ValueError(f"{name} must be positive")
    return canonical


def _required_nonblank_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    canonical = str.__str__(value)
    if not canonical.strip():
        raise ValueError(f"{name} must be nonblank")
    return canonical


def _closed_literal_string(
    value: object, name: str, allowed_values: frozenset[str]
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    canonical = str.__str__(value)
    if canonical not in allowed_values:
        raise ValueError(f"{name} must be a supported value")
    return canonical


def _optional_nonnegative_int(value: object, name: str) -> int | None:
    return None if value is None else _nonnegative_int(value, name)


def _validate_error(
    code: object, message: object, allowed_codes: frozenset[str]
) -> tuple[str, str]:
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    canonical_code = str.__str__(code)
    if canonical_code not in allowed_codes:
        raise ValueError(f"unknown sparse-linewidth error code: {canonical_code!r}")
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    canonical_message = str.__str__(message)
    if not canonical_message:
        raise ValueError("message must be nonempty")
    return canonical_code, canonical_message


_GEOMETRY_FAILURE_CODES = frozenset(
    {
        "nonrepresentable_frequency_lower",
        "nonrepresentable_frequency_upper",
        "empty_fit_bounds",
        "calibration_cell_violation",
        "source_domain_violation",
    }
)
_RESET_FAILURE_CODES = frozenset(
    {
        "invalid_argument_type",
        "configuration_mismatch",
        "calibration_mismatch",
        "metadata_mismatch",
        "invalid_base_sparse_geometry",
        "budget_mismatch",
        "initial_state_construction_failed",
    }
)
_OBSERVATION_VALIDATION_CODES = frozenset(
    {
        "invalid_observation_type",
        "no_pending_query",
        "pending_mode_mismatch",
        "fast_query_echo_mismatch",
        "sparse_query_echo_mismatch",
        "sequence_mismatch",
        "frequency_mismatch",
        "integration_time_mismatch",
        "endpoint_mismatch",
        "nominal_exposure_mismatch",
        "invalid_observation_value",
    }
)
_UPDATE_CONSTRUCTION_CODES = frozenset(
    {
        "fast_partial_pair_construction_failed",
        "fast_pair_result_construction_failed",
        "fast_identity_estimate_construction_failed",
        "sparse_partial_scan_construction_failed",
        "sparse_scan_result_construction_failed",
        "sparse_identity_estimate_construction_failed",
        "resource_construction_failed",
        "aggregate_estimate_construction_failed",
        "update_construction_failed",
    }
)


class SparseLinewidthResetError(ValueError):
    """Raised when a composite reset cannot construct valid public state."""

    code: SparseResetFailureCode
    message: str

    def __init__(self, code: SparseResetFailureCode, message: str) -> None:
        self.code, self.message = _validate_error(code, message, _RESET_FAILURE_CODES)
        super().__init__(self.message)


class SparseLinewidthObservationValidationError(ValueError):
    """Raised when an observation cannot be accepted by the pending query."""

    code: SparseObservationValidationCode
    message: str

    def __init__(self, code: SparseObservationValidationCode, message: str) -> None:
        self.code, self.message = _validate_error(
            code, message, _OBSERVATION_VALIDATION_CODES
        )
        super().__init__(self.message)


class SparseLinewidthUpdateConstructionError(RuntimeError):
    """Raised when a post-validation composite update cannot be constructed."""

    code: SparseUpdateConstructionCode
    message: str

    def __init__(self, code: SparseUpdateConstructionCode, message: str) -> None:
        self.code, self.message = _validate_error(
            code, message, _UPDATE_CONSTRUCTION_CODES
        )
        super().__init__(self.message)


@dataclass(frozen=True, slots=True)
class SparseLinewidthConfiguration:
    """Validated immutable policy for sparse five-point linewidth scans."""

    scan_period_fast_pairs: int = 8
    integration_time_s: float = 0.005
    center_correction_limit_fwhm_fraction: float = 0.5
    min_fwhm_prior_ratio: float = 0.5
    max_fwhm_prior_ratio: float = 2.0
    min_resolved_amplitude_source_ratio: float = 0.25
    max_amplitude_source_ratio: float = 4.0
    baseline_offset_source_amplitude_fraction: float = 1.0
    rank_rtol: float = 1.0e-10
    max_scaled_jacobian_condition: float = 1.0e8
    min_interior_bound_fraction: float = 1.0e-6
    max_amplitude_normalized_rmse: float = 0.10
    max_nfev: int = 4000

    def __post_init__(self) -> None:
        scan_period_fast_pairs = _positive_int(
            self.scan_period_fast_pairs, "scan_period_fast_pairs"
        )
        integration_time_s = _positive_float(
            self.integration_time_s, "integration_time_s"
        )
        center_correction_limit_fwhm_fraction = _positive_float(
            self.center_correction_limit_fwhm_fraction,
            "center_correction_limit_fwhm_fraction",
        )
        min_fwhm_prior_ratio = _positive_float(
            self.min_fwhm_prior_ratio, "min_fwhm_prior_ratio"
        )
        max_fwhm_prior_ratio = _positive_float(
            self.max_fwhm_prior_ratio, "max_fwhm_prior_ratio"
        )
        if min_fwhm_prior_ratio >= max_fwhm_prior_ratio:
            raise ValueError(
                "min_fwhm_prior_ratio must be strictly less than max_fwhm_prior_ratio"
            )
        min_resolved_amplitude_source_ratio = _positive_float(
            self.min_resolved_amplitude_source_ratio,
            "min_resolved_amplitude_source_ratio",
        )
        max_amplitude_source_ratio = _positive_float(
            self.max_amplitude_source_ratio, "max_amplitude_source_ratio"
        )
        baseline_offset_source_amplitude_fraction = _positive_float(
            self.baseline_offset_source_amplitude_fraction,
            "baseline_offset_source_amplitude_fraction",
        )
        rank_rtol = _positive_float(self.rank_rtol, "rank_rtol")
        if rank_rtol >= 1.0:
            raise ValueError("rank_rtol must be within (0, 1)")
        max_scaled_jacobian_condition = _positive_float(
            self.max_scaled_jacobian_condition, "max_scaled_jacobian_condition"
        )
        if max_scaled_jacobian_condition < 1.0:
            raise ValueError("max_scaled_jacobian_condition must be at least one")
        min_interior_bound_fraction = _nonnegative_float(
            self.min_interior_bound_fraction, "min_interior_bound_fraction"
        )
        if min_interior_bound_fraction >= 0.5:
            raise ValueError("min_interior_bound_fraction must be within [0, 0.5)")
        max_amplitude_normalized_rmse = _positive_float(
            self.max_amplitude_normalized_rmse, "max_amplitude_normalized_rmse"
        )
        max_nfev = _positive_int(self.max_nfev, "max_nfev")

        object.__setattr__(self, "scan_period_fast_pairs", scan_period_fast_pairs)
        object.__setattr__(self, "integration_time_s", integration_time_s)
        object.__setattr__(
            self,
            "center_correction_limit_fwhm_fraction",
            center_correction_limit_fwhm_fraction,
        )
        object.__setattr__(self, "min_fwhm_prior_ratio", min_fwhm_prior_ratio)
        object.__setattr__(self, "max_fwhm_prior_ratio", max_fwhm_prior_ratio)
        object.__setattr__(
            self,
            "min_resolved_amplitude_source_ratio",
            min_resolved_amplitude_source_ratio,
        )
        object.__setattr__(
            self, "max_amplitude_source_ratio", max_amplitude_source_ratio
        )
        object.__setattr__(
            self,
            "baseline_offset_source_amplitude_fraction",
            baseline_offset_source_amplitude_fraction,
        )
        object.__setattr__(self, "rank_rtol", rank_rtol)
        object.__setattr__(
            self, "max_scaled_jacobian_condition", max_scaled_jacobian_condition
        )
        object.__setattr__(
            self, "min_interior_bound_fraction", min_interior_bound_fraction
        )
        object.__setattr__(
            self, "max_amplitude_normalized_rmse", max_amplitude_normalized_rmse
        )
        object.__setattr__(self, "max_nfev", max_nfev)


@dataclass(frozen=True, slots=True)
class SparseGeometryUnavailableDiagnostic:
    """Exact diagnostic snapshot for a clean due-scan geometry stop."""

    failure_code: SparseGeometryFailureCode
    scan_index: int
    identity_scan_index: int
    resonance_id: str
    fast_center_hz: float
    fast_center_source_kind: Literal["calibration", "pair"]
    fast_center_source_pair_index: int | None
    fast_center_reference_timestamp_s: float
    fast_center_release_sequence_index: int | None
    fast_center_release_timestamp_s: float
    prior_fwhm_hz: float
    fwhm_source_kind: SparseLinewidthSourceKind
    fwhm_source_scan_index: int | None
    fwhm_reference_timestamp_s: float
    fwhm_release_sequence_index: int | None
    fwhm_release_timestamp_s: float
    proposed_frequency_min_hz: float | None
    proposed_frequency_max_hz: float | None
    calibration_cell_lower_hz: float
    calibration_cell_upper_hz: float
    source_frequency_min_hz: float
    source_frequency_max_hz: float

    def __post_init__(self) -> None:
        failure_code = _closed_literal_string(
            self.failure_code, "failure_code", _GEOMETRY_FAILURE_CODES
        )
        scan_index = _nonnegative_int(self.scan_index, "scan_index")
        identity_scan_index = _nonnegative_int(
            self.identity_scan_index, "identity_scan_index"
        )
        resonance_id = _required_nonblank_string(self.resonance_id, "resonance_id")
        fast_center_hz = _finite_float(self.fast_center_hz, "fast_center_hz")
        fast_center_source_kind = _closed_literal_string(
            self.fast_center_source_kind,
            "fast_center_source_kind",
            frozenset({"calibration", "pair"}),
        )
        fast_center_source_pair_index = _optional_nonnegative_int(
            self.fast_center_source_pair_index, "fast_center_source_pair_index"
        )
        if (fast_center_source_kind == "calibration") != (
            fast_center_source_pair_index is None
        ):
            raise ValueError(
                "fast_center_source_pair_index must agree with fast_center_source_kind"
            )
        fast_center_reference_timestamp_s = _nonnegative_float(
            self.fast_center_reference_timestamp_s,
            "fast_center_reference_timestamp_s",
        )
        fast_center_release_sequence_index = _optional_nonnegative_int(
            self.fast_center_release_sequence_index,
            "fast_center_release_sequence_index",
        )
        fast_center_release_timestamp_s = _nonnegative_float(
            self.fast_center_release_timestamp_s,
            "fast_center_release_timestamp_s",
        )
        prior_fwhm_hz = _positive_float(self.prior_fwhm_hz, "prior_fwhm_hz")
        fwhm_source_kind = _closed_literal_string(
            self.fwhm_source_kind,
            "fwhm_source_kind",
            frozenset({"calibration", "scan"}),
        )
        fwhm_source_scan_index = _optional_nonnegative_int(
            self.fwhm_source_scan_index, "fwhm_source_scan_index"
        )
        if (fwhm_source_kind == "calibration") != (fwhm_source_scan_index is None):
            raise ValueError("fwhm_source_scan_index must agree with fwhm_source_kind")
        fwhm_reference_timestamp_s = _nonnegative_float(
            self.fwhm_reference_timestamp_s, "fwhm_reference_timestamp_s"
        )
        fwhm_release_sequence_index = _optional_nonnegative_int(
            self.fwhm_release_sequence_index, "fwhm_release_sequence_index"
        )
        fwhm_release_timestamp_s = _nonnegative_float(
            self.fwhm_release_timestamp_s, "fwhm_release_timestamp_s"
        )
        proposed_frequency_min_hz = self.proposed_frequency_min_hz
        proposed_frequency_max_hz = self.proposed_frequency_max_hz
        nonrepresentable = failure_code in {
            "nonrepresentable_frequency_lower",
            "nonrepresentable_frequency_upper",
        }
        if nonrepresentable:
            if (
                proposed_frequency_min_hz is not None
                or proposed_frequency_max_hz is not None
            ):
                raise ValueError(
                    "nonrepresentable geometry must not retain proposed frequencies"
                )
        else:
            if proposed_frequency_min_hz is None or proposed_frequency_max_hz is None:
                raise ValueError(
                    "representable geometry must retain both proposed frequencies"
                )
            proposed_frequency_min_hz = _finite_float(
                proposed_frequency_min_hz, "proposed_frequency_min_hz"
            )
            proposed_frequency_max_hz = _finite_float(
                proposed_frequency_max_hz, "proposed_frequency_max_hz"
            )
            if proposed_frequency_min_hz >= proposed_frequency_max_hz:
                raise ValueError(
                    "proposed_frequency_min_hz must be below proposed_frequency_max_hz"
                )
        calibration_cell_lower_hz = _finite_float(
            self.calibration_cell_lower_hz, "calibration_cell_lower_hz"
        )
        calibration_cell_upper_hz = _finite_float(
            self.calibration_cell_upper_hz, "calibration_cell_upper_hz"
        )
        if calibration_cell_lower_hz >= calibration_cell_upper_hz:
            raise ValueError(
                "calibration_cell_lower_hz must be below calibration_cell_upper_hz"
            )
        source_frequency_min_hz = _finite_float(
            self.source_frequency_min_hz, "source_frequency_min_hz"
        )
        source_frequency_max_hz = _finite_float(
            self.source_frequency_max_hz, "source_frequency_max_hz"
        )
        if source_frequency_min_hz >= source_frequency_max_hz:
            raise ValueError(
                "source_frequency_min_hz must be below source_frequency_max_hz"
            )

        object.__setattr__(self, "failure_code", failure_code)
        object.__setattr__(self, "scan_index", scan_index)
        object.__setattr__(self, "identity_scan_index", identity_scan_index)
        object.__setattr__(self, "resonance_id", resonance_id)
        object.__setattr__(self, "fast_center_hz", fast_center_hz)
        object.__setattr__(self, "fast_center_source_kind", fast_center_source_kind)
        object.__setattr__(
            self, "fast_center_source_pair_index", fast_center_source_pair_index
        )
        object.__setattr__(
            self,
            "fast_center_reference_timestamp_s",
            fast_center_reference_timestamp_s,
        )
        object.__setattr__(
            self,
            "fast_center_release_sequence_index",
            fast_center_release_sequence_index,
        )
        object.__setattr__(
            self,
            "fast_center_release_timestamp_s",
            fast_center_release_timestamp_s,
        )
        object.__setattr__(self, "prior_fwhm_hz", prior_fwhm_hz)
        object.__setattr__(self, "fwhm_source_kind", fwhm_source_kind)
        object.__setattr__(self, "fwhm_source_scan_index", fwhm_source_scan_index)
        object.__setattr__(
            self, "fwhm_reference_timestamp_s", fwhm_reference_timestamp_s
        )
        object.__setattr__(
            self, "fwhm_release_sequence_index", fwhm_release_sequence_index
        )
        object.__setattr__(self, "fwhm_release_timestamp_s", fwhm_release_timestamp_s)
        object.__setattr__(self, "proposed_frequency_min_hz", proposed_frequency_min_hz)
        object.__setattr__(
            self,
            "proposed_frequency_max_hz",
            proposed_frequency_max_hz,
        )
        object.__setattr__(self, "calibration_cell_lower_hz", calibration_cell_lower_hz)
        object.__setattr__(self, "calibration_cell_upper_hz", calibration_cell_upper_hz)
        object.__setattr__(self, "source_frequency_min_hz", source_frequency_min_hz)
        object.__setattr__(self, "source_frequency_max_hz", source_frequency_max_hz)
