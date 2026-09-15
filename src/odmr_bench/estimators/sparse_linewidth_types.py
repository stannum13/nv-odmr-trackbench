"""Public primitive contracts for sparse five-point linewidth tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators.two_point_types import (
    CalibrationBudgetTreatment,
    CalibrationSourceProvenance,
    PublicAcquisitionResources,
    TwoPointBudgetCeiling,
    TwoPointPairResult,
    TwoPointPartialPair,
    TwoPointQuery,
)

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


def _optional_finite_float(value: object, name: str) -> float | None:
    return None if value is None else _finite_float(value, name)


def _optional_closed_literal_string(
    value: object, name: str, allowed_values: frozenset[str]
) -> str | None:
    return (
        None if value is None else _closed_literal_string(value, name, allowed_values)
    )


def _validate_observation_echo(
    *,
    frequency_hz: float,
    integration_time_s: float,
    expected_sequence_index: int,
    expected_end_timestamp_s: float,
    expected_nominal_exposure_photons: float,
    observation: object,
    name: str,
) -> EstimatorObservation:
    if type(observation) is not EstimatorObservation:
        raise TypeError(f"{name} must be an exact EstimatorObservation")
    if (
        observation.frequency_hz != frequency_hz
        or observation.integration_time_s != integration_time_s
        or observation.sequence_index != expected_sequence_index
        or observation.timestamp_s != expected_end_timestamp_s
        or observation.nominal_exposure_photons != expected_nominal_exposure_photons
    ):
        raise ValueError(f"{name} must echo its query")
    return observation


def _validate_source_fields(
    *,
    center_kind: object,
    center_index: object,
    center_reference: object,
    center_release_index: object,
    center_release_timestamp: object,
    width_kind: object,
    width_index: object,
    width_reference: object,
    width_release_index: object,
    width_release_timestamp: object,
    prefix: str,
) -> tuple[
    str, int | None, float, int | None, float, str, int | None, float, int | None, float
]:
    center_kind = _closed_literal_string(
        center_kind,
        f"{prefix}fast_center_source_kind",
        frozenset({"calibration", "pair"}),
    )
    center_index = _optional_nonnegative_int(
        center_index, f"{prefix}fast_center_source_pair_index"
    )
    center_reference = _finite_float(
        center_reference, f"{prefix}fast_center_reference_timestamp_s"
    )
    center_release_index = _optional_nonnegative_int(
        center_release_index, f"{prefix}fast_center_release_sequence_index"
    )
    center_release_timestamp = _nonnegative_float(
        center_release_timestamp, f"{prefix}fast_center_release_timestamp_s"
    )
    if (center_kind == "calibration") != (center_index is None):
        raise ValueError("fast-center source kind must agree with source pair index")
    if center_kind == "pair" and (
        center_reference < 0.0 or center_release_index is None
    ):
        raise ValueError(
            "pair fast-center source requires nonnegative reference and release"
        )
    width_kind = _closed_literal_string(
        width_kind, f"{prefix}fwhm_source_kind", frozenset({"calibration", "scan"})
    )
    width_index = _optional_nonnegative_int(
        width_index, f"{prefix}fwhm_source_scan_index"
    )
    width_reference = _finite_float(
        width_reference, f"{prefix}fwhm_reference_timestamp_s"
    )
    width_release_index = _optional_nonnegative_int(
        width_release_index, f"{prefix}fwhm_release_sequence_index"
    )
    width_release_timestamp = _nonnegative_float(
        width_release_timestamp, f"{prefix}fwhm_release_timestamp_s"
    )
    if (width_kind == "calibration") != (width_index is None):
        raise ValueError("FWHM source kind must agree with source scan index")
    if width_kind == "scan" and (width_reference < 0.0 or width_release_index is None):
        raise ValueError("scan FWHM source requires nonnegative reference and release")
    return (
        center_kind,
        center_index,
        center_reference,
        center_release_index,
        center_release_timestamp,
        width_kind,
        width_index,
        width_reference,
        width_release_index,
        width_release_timestamp,
    )


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


_SPARSE_OFFSETS = ((0.5, -1.0, 0.0, 1.0, -0.5), (-0.5, 1.0, 0.0, -1.0, 0.5))
_FIT_FAILURE_CODES = frozenset(_GEOMETRY_FAILURE_CODES)  # Rebound below after aliases.


@dataclass(frozen=True, slots=True)
class SparseLinewidthQuery:
    """One immutable member of a reserved five-observation sparse scan."""

    acquisition_index: int
    scan_index: int
    identity_scan_index: int
    point_index: int
    resonance_id: str
    offset_multiplier: float
    frozen_fast_center_hz: float
    frozen_fast_center_source_kind: Literal["calibration", "pair"]
    frozen_fast_center_source_pair_index: int | None
    frozen_fast_center_reference_timestamp_s: float
    frozen_fast_center_release_sequence_index: int | None
    frozen_fast_center_release_timestamp_s: float
    frozen_prior_fwhm_hz: float
    frozen_fwhm_source_kind: SparseLinewidthSourceKind
    frozen_fwhm_source_scan_index: int | None
    frozen_fwhm_reference_timestamp_s: float
    frozen_fwhm_release_sequence_index: int | None
    frozen_fwhm_release_timestamp_s: float
    frequency_hz: float
    integration_time_s: float
    expected_sequence_index: int
    expected_end_timestamp_s: float
    expected_nominal_exposure_photons: float

    def __post_init__(self) -> None:
        acquisition_index = _nonnegative_int(
            self.acquisition_index, "acquisition_index"
        )
        scan_index = _nonnegative_int(self.scan_index, "scan_index")
        identity_scan_index = _nonnegative_int(
            self.identity_scan_index, "identity_scan_index"
        )
        point_index = _nonnegative_int(self.point_index, "point_index")
        if point_index >= 5:
            raise ValueError("point_index must be in range(5)")
        if identity_scan_index != scan_index // 8:
            raise ValueError("identity_scan_index must equal scan_index // 8")
        resonance_id = _required_nonblank_string(self.resonance_id, "resonance_id")
        offset_multiplier = _finite_float(self.offset_multiplier, "offset_multiplier")
        if offset_multiplier != _SPARSE_OFFSETS[identity_scan_index % 2][point_index]:
            raise ValueError("offset_multiplier must follow the canonical sparse order")
        frozen_fast_center_hz = _finite_float(
            self.frozen_fast_center_hz, "frozen_fast_center_hz"
        )
        frozen_prior_fwhm_hz = _positive_float(
            self.frozen_prior_fwhm_hz, "frozen_prior_fwhm_hz"
        )
        source_fields = _validate_source_fields(
            center_kind=self.frozen_fast_center_source_kind,
            center_index=self.frozen_fast_center_source_pair_index,
            center_reference=self.frozen_fast_center_reference_timestamp_s,
            center_release_index=self.frozen_fast_center_release_sequence_index,
            center_release_timestamp=self.frozen_fast_center_release_timestamp_s,
            width_kind=self.frozen_fwhm_source_kind,
            width_index=self.frozen_fwhm_source_scan_index,
            width_reference=self.frozen_fwhm_reference_timestamp_s,
            width_release_index=self.frozen_fwhm_release_sequence_index,
            width_release_timestamp=self.frozen_fwhm_release_timestamp_s,
            prefix="frozen_",
        )
        frequency_hz = _finite_float(self.frequency_hz, "frequency_hz")
        if (
            frequency_hz
            != frozen_fast_center_hz + offset_multiplier * frozen_prior_fwhm_hz
        ):
            raise ValueError("frequency_hz must equal the frozen sparse geometry")
        integration_time_s = _positive_float(
            self.integration_time_s, "integration_time_s"
        )
        expected_sequence_index = _nonnegative_int(
            self.expected_sequence_index, "expected_sequence_index"
        )
        expected_end_timestamp_s = _nonnegative_float(
            self.expected_end_timestamp_s, "expected_end_timestamp_s"
        )
        if expected_end_timestamp_s < integration_time_s:
            raise ValueError("expected endpoint must include integration time")
        expected_nominal_exposure_photons = _nonnegative_float(
            self.expected_nominal_exposure_photons,
            "expected_nominal_exposure_photons",
        )
        for name, value in (
            ("acquisition_index", acquisition_index),
            ("scan_index", scan_index),
            ("identity_scan_index", identity_scan_index),
            ("point_index", point_index),
            ("resonance_id", resonance_id),
            ("offset_multiplier", offset_multiplier),
            ("frozen_fast_center_hz", frozen_fast_center_hz),
            ("frozen_prior_fwhm_hz", frozen_prior_fwhm_hz),
            ("frequency_hz", frequency_hz),
            ("integration_time_s", integration_time_s),
            ("expected_sequence_index", expected_sequence_index),
            ("expected_end_timestamp_s", expected_end_timestamp_s),
            ("expected_nominal_exposure_photons", expected_nominal_exposure_photons),
        ):
            object.__setattr__(self, name, value)
        for name, value in zip(
            (
                "frozen_fast_center_source_kind",
                "frozen_fast_center_source_pair_index",
                "frozen_fast_center_reference_timestamp_s",
                "frozen_fast_center_release_sequence_index",
                "frozen_fast_center_release_timestamp_s",
                "frozen_fwhm_source_kind",
                "frozen_fwhm_source_scan_index",
                "frozen_fwhm_reference_timestamp_s",
                "frozen_fwhm_release_sequence_index",
                "frozen_fwhm_release_timestamp_s",
            ),
            source_fields,
            strict=True,
        ):
            object.__setattr__(self, name, value)


def _query_snapshot(query: SparseLinewidthQuery) -> tuple[object, ...]:
    return tuple(
        getattr(query, name)
        for name in (
            "scan_index",
            "identity_scan_index",
            "resonance_id",
            "frozen_fast_center_hz",
            "frozen_fast_center_source_kind",
            "frozen_fast_center_source_pair_index",
            "frozen_fast_center_reference_timestamp_s",
            "frozen_fast_center_release_sequence_index",
            "frozen_fast_center_release_timestamp_s",
            "frozen_prior_fwhm_hz",
            "frozen_fwhm_source_kind",
            "frozen_fwhm_source_scan_index",
            "frozen_fwhm_reference_timestamp_s",
            "frozen_fwhm_release_sequence_index",
            "frozen_fwhm_release_timestamp_s",
        )
    )


def _validate_sparse_queries(
    queries: object, observations: object, *, expected_length: int | range, name: str
) -> tuple[tuple[SparseLinewidthQuery, ...], tuple[EstimatorObservation, ...]]:
    if not isinstance(queries, (tuple, list)) or not isinstance(
        observations, (tuple, list)
    ):
        raise TypeError(f"{name} queries and observations must be ordered sequences")
    queries = tuple(queries)
    observations = tuple(observations)
    if len(queries) not in expected_length or len(observations) != len(queries):
        raise ValueError(f"{name} must have its exact query/observation length")
    if not queries or not all(type(query) is SparseLinewidthQuery for query in queries):
        raise TypeError(f"{name} queries must be exact SparseLinewidthQuery values")
    snapshot = _query_snapshot(queries[0])
    for index, (query, observation) in enumerate(
        zip(queries, observations, strict=True)
    ):
        if _query_snapshot(query) != snapshot or query.point_index != index:
            raise ValueError(f"{name} queries must be one exact frozen scan geometry")
        _validate_observation_echo(
            frequency_hz=query.frequency_hz,
            integration_time_s=query.integration_time_s,
            expected_sequence_index=query.expected_sequence_index,
            expected_end_timestamp_s=query.expected_end_timestamp_s,
            expected_nominal_exposure_photons=query.expected_nominal_exposure_photons,
            observation=observation,
            name="sparse observation",
        )
        if index and (
            observation.sequence_index != observations[index - 1].sequence_index + 1
            or observation.timestamp_s <= observations[index - 1].timestamp_s
        ):
            raise ValueError(f"{name} observations must be contiguous in arrival order")
    return queries, observations


@dataclass(frozen=True, slots=True)
class SparsePartialScan:
    scan_index: int
    identity_scan_index: int
    resonance_id: str
    frozen_fast_center_hz: float
    frozen_fast_center_source_kind: Literal["calibration", "pair"]
    frozen_fast_center_source_pair_index: int | None
    frozen_fast_center_reference_timestamp_s: float
    frozen_fast_center_release_sequence_index: int | None
    frozen_fast_center_release_timestamp_s: float
    frozen_prior_fwhm_hz: float
    frozen_fwhm_source_kind: SparseLinewidthSourceKind
    frozen_fwhm_source_scan_index: int | None
    frozen_fwhm_reference_timestamp_s: float
    frozen_fwhm_release_sequence_index: int | None
    frozen_fwhm_release_timestamp_s: float
    queries: tuple[SparseLinewidthQuery, ...]
    observations: tuple[EstimatorObservation, ...]

    def __post_init__(self) -> None:
        queries, observations = _validate_sparse_queries(
            self.queries,
            self.observations,
            expected_length=range(1, 5),
            name="partial scan",
        )
        values = _query_snapshot(queries[0])
        supplied = (
            self.scan_index,
            self.identity_scan_index,
            self.resonance_id,
            self.frozen_fast_center_hz,
            self.frozen_fast_center_source_kind,
            self.frozen_fast_center_source_pair_index,
            self.frozen_fast_center_reference_timestamp_s,
            self.frozen_fast_center_release_sequence_index,
            self.frozen_fast_center_release_timestamp_s,
            self.frozen_prior_fwhm_hz,
            self.frozen_fwhm_source_kind,
            self.frozen_fwhm_source_scan_index,
            self.frozen_fwhm_reference_timestamp_s,
            self.frozen_fwhm_release_sequence_index,
            self.frozen_fwhm_release_timestamp_s,
        )
        if supplied != values:
            raise ValueError("partial scan fields must exactly echo every query")
        for name, value in zip(
            (
                "scan_index",
                "identity_scan_index",
                "resonance_id",
                "frozen_fast_center_hz",
                "frozen_fast_center_source_kind",
                "frozen_fast_center_source_pair_index",
                "frozen_fast_center_reference_timestamp_s",
                "frozen_fast_center_release_sequence_index",
                "frozen_fast_center_release_timestamp_s",
                "frozen_prior_fwhm_hz",
                "frozen_fwhm_source_kind",
                "frozen_fwhm_source_scan_index",
                "frozen_fwhm_reference_timestamp_s",
                "frozen_fwhm_release_sequence_index",
                "frozen_fwhm_release_timestamp_s",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "observations", observations)


_FIT_FAILURE_CODES = frozenset(
    {
        "model_evaluation_failed",
        "optimizer_failed",
        "nonfinite_solution",
        "bounds_active",
        "rank_deficient",
        "ill_conditioned",
        "amplitude_unresolved",
        "residual_quality_failed",
    }
)


@dataclass(frozen=True, slots=True)
class SparseLinewidthScanResult:
    scan_index: int
    identity_scan_index: int
    resonance_id: str
    frozen_fast_center_hz: float
    frozen_fast_center_source_kind: Literal["calibration", "pair"]
    frozen_fast_center_source_pair_index: int | None
    frozen_fast_center_reference_timestamp_s: float
    frozen_fast_center_release_sequence_index: int | None
    frozen_fast_center_release_timestamp_s: float
    frozen_prior_fwhm_hz: float
    frozen_fwhm_source_kind: SparseLinewidthSourceKind
    frozen_fwhm_source_scan_index: int | None
    frozen_fwhm_reference_timestamp_s: float
    frozen_fwhm_release_sequence_index: int | None
    frozen_fwhm_release_timestamp_s: float
    queries: tuple[SparseLinewidthQuery, ...]
    observations: tuple[EstimatorObservation, ...]
    public_reference_timestamp_s: float
    release_sequence_index: int
    release_timestamp_s: float
    status: Literal["success", "failure"]
    failure_code: SparseLinewidthFailureCode | None
    fitted_center_correction_hz: float | None
    fitted_local_center_hz: float | None
    fitted_fwhm_hz: float | None
    fitted_amplitude: float | None
    fitted_baseline_offset: float | None
    fitted_q: float | None
    rmse: float | None
    amplitude_normalized_rmse: float | None
    scaled_jacobian_rank: int | None
    scaled_jacobian_condition: float | None
    scipy_status: int | None
    scipy_message: str | None
    nfev: int | None
    fit_cpu_time_s: float

    def __post_init__(self) -> None:
        queries, observations = _validate_sparse_queries(
            self.queries,
            self.observations,
            expected_length=range(5, 6),
            name="scan result",
        )
        values = _query_snapshot(queries[0])
        supplied = (
            self.scan_index,
            self.identity_scan_index,
            self.resonance_id,
            self.frozen_fast_center_hz,
            self.frozen_fast_center_source_kind,
            self.frozen_fast_center_source_pair_index,
            self.frozen_fast_center_reference_timestamp_s,
            self.frozen_fast_center_release_sequence_index,
            self.frozen_fast_center_release_timestamp_s,
            self.frozen_prior_fwhm_hz,
            self.frozen_fwhm_source_kind,
            self.frozen_fwhm_source_scan_index,
            self.frozen_fwhm_reference_timestamp_s,
            self.frozen_fwhm_release_sequence_index,
            self.frozen_fwhm_release_timestamp_s,
        )
        if supplied != values:
            raise ValueError("scan-result fields must exactly echo every query")
        public_reference_timestamp_s = _nonnegative_float(
            self.public_reference_timestamp_s, "public_reference_timestamp_s"
        )
        midpoints = tuple(
            observation.timestamp_s - observation.integration_time_s / 2.0
            for observation in observations
        )
        expected_reference = midpoints[0]
        for midpoint in midpoints[1:]:
            expected_reference = (
                expected_reference + (midpoint - expected_reference) / 2.0
            )
        # Five point ordered mean is deliberately the ordinary left fold, not np.mean.
        expected_reference = sum(midpoints) / 5.0
        if public_reference_timestamp_s != expected_reference:
            raise ValueError("public reference must equal the five public midpoints")
        release_sequence_index = _nonnegative_int(
            self.release_sequence_index, "release_sequence_index"
        )
        release_timestamp_s = _nonnegative_float(
            self.release_timestamp_s, "release_timestamp_s"
        )
        if (
            release_sequence_index != observations[-1].sequence_index
            or release_timestamp_s != observations[-1].timestamp_s
        ):
            raise ValueError("scan release must equal the fifth observation endpoint")
        status = _closed_literal_string(
            self.status, "status", frozenset({"success", "failure"})
        )
        failure_code = _optional_closed_literal_string(
            self.failure_code, "failure_code", _FIT_FAILURE_CODES
        )
        if (status == "success") != (failure_code is None):
            raise ValueError("status and failure_code must form the exact union")
        fitted = tuple(
            _optional_finite_float(value, name)
            for name, value in (
                ("fitted_center_correction_hz", self.fitted_center_correction_hz),
                ("fitted_local_center_hz", self.fitted_local_center_hz),
                ("fitted_fwhm_hz", self.fitted_fwhm_hz),
                ("fitted_amplitude", self.fitted_amplitude),
                ("fitted_baseline_offset", self.fitted_baseline_offset),
            )
        )
        if fitted[2] is not None and fitted[2] <= 0.0:
            raise ValueError("fitted_fwhm_hz must be positive")
        if fitted[3] is not None and fitted[3] < 0.0:
            raise ValueError("fitted_amplitude must be nonnegative")
        if (
            all(value is not None for value in fitted)
            and fitted[1] != values[3] + fitted[0]
        ):
            raise ValueError(
                "fitted local center must equal frozen center plus correction"
            )
        fitted_q = _optional_finite_float(self.fitted_q, "fitted_q")
        rmse = _optional_finite_float(self.rmse, "rmse")
        normalized_rmse = _optional_finite_float(
            self.amplitude_normalized_rmse, "amplitude_normalized_rmse"
        )
        if (rmse is not None and rmse < 0.0) or (
            normalized_rmse is not None and normalized_rmse < 0.0
        ):
            raise ValueError("RMSE values must be nonnegative")
        rank = (
            None
            if self.scaled_jacobian_rank is None
            else _nonnegative_int(self.scaled_jacobian_rank, "scaled_jacobian_rank")
        )
        if rank is not None and rank > 4:
            raise ValueError("scaled_jacobian_rank must be at most four")
        condition = _optional_finite_float(
            self.scaled_jacobian_condition, "scaled_jacobian_condition"
        )
        if condition is not None and condition <= 0.0:
            raise ValueError("scaled_jacobian_condition must be positive")
        scipy_status = (
            None
            if self.scipy_status is None
            else _nonnegative_int(self.scipy_status, "scipy_status")
        )
        scipy_message = (
            None
            if self.scipy_message is None
            else _required_nonblank_string(self.scipy_message, "scipy_message")
        )
        nfev = None if self.nfev is None else _positive_int(self.nfev, "nfev")
        solver_present = (
            scipy_status is not None,
            scipy_message is not None,
            nfev is not None,
        )
        if len(set(solver_present)) != 1:
            raise ValueError("solver diagnostics must be jointly present or absent")
        fit_group = all(value is not None for value in fitted)
        if any(value is None for value in fitted) and not all(
            value is None for value in fitted
        ):
            raise ValueError("fitted diagnostics must be jointly present or absent")
        rmse_group = rmse is not None and normalized_rmse is not None
        if (rmse is None) != (normalized_rmse is None):
            raise ValueError("RMSE diagnostics must be jointly present or absent")
        expected = {
            "model_evaluation_failed": (False, False, False, False, False),
            "optimizer_failed": (True, False, False, False, False),
            "nonfinite_solution": (True, False, False, False, False),
            "bounds_active": (True, True, True, False, False),
            "rank_deficient": (True, True, True, True, False),
            "ill_conditioned": (True, True, True, True, True),
            "amplitude_unresolved": (True, True, True, True, True),
            "residual_quality_failed": (True, True, True, True, True),
            None: (True, True, True, True, True),
        }[failure_code]
        actual = (
            scipy_status is not None,
            fit_group,
            rmse_group,
            rank is not None,
            condition is not None,
        )
        if actual != expected:
            raise ValueError("result diagnostics must match their ordered fit gate")
        if (
            scipy_status is not None
            and failure_code != "optimizer_failed"
            and scipy_status <= 0
        ):
            raise ValueError("post-optimizer results require a positive scipy status")
        if failure_code == "rank_deficient" and rank not in {0, 1, 2, 3}:
            raise ValueError("rank-deficient results require rank 0 through 3")
        if (
            failure_code
            in {
                "ill_conditioned",
                "amplitude_unresolved",
                "residual_quality_failed",
                None,
            }
            and rank != 4
        ):
            raise ValueError("full-rank late results require rank four")
        if (status == "success") != (fitted_q is not None):
            raise ValueError("fitted_q is present exactly for success")
        if fitted_q is not None and fitted_q != fitted[1] / fitted[2]:
            raise ValueError("fitted_q must equal fitted local center divided by FWHM")
        fit_cpu_time_s = _nonnegative_float(self.fit_cpu_time_s, "fit_cpu_time_s")
        for name, value in zip(
            (
                "scan_index",
                "identity_scan_index",
                "resonance_id",
                "frozen_fast_center_hz",
                "frozen_fast_center_source_kind",
                "frozen_fast_center_source_pair_index",
                "frozen_fast_center_reference_timestamp_s",
                "frozen_fast_center_release_sequence_index",
                "frozen_fast_center_release_timestamp_s",
                "frozen_prior_fwhm_hz",
                "frozen_fwhm_source_kind",
                "frozen_fwhm_source_scan_index",
                "frozen_fwhm_reference_timestamp_s",
                "frozen_fwhm_release_sequence_index",
                "frozen_fwhm_release_timestamp_s",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "observations", observations)
        for name, value in (
            ("public_reference_timestamp_s", public_reference_timestamp_s),
            ("release_sequence_index", release_sequence_index),
            ("release_timestamp_s", release_timestamp_s),
            ("status", status),
            ("failure_code", failure_code),
            ("fitted_center_correction_hz", fitted[0]),
            ("fitted_local_center_hz", fitted[1]),
            ("fitted_fwhm_hz", fitted[2]),
            ("fitted_amplitude", fitted[3]),
            ("fitted_baseline_offset", fitted[4]),
            ("fitted_q", fitted_q),
            ("rmse", rmse),
            ("amplitude_normalized_rmse", normalized_rmse),
            ("scaled_jacobian_rank", rank),
            ("scaled_jacobian_condition", condition),
            ("scipy_status", scipy_status),
            ("scipy_message", scipy_message),
            ("nfev", nfev),
            ("fit_cpu_time_s", fit_cpu_time_s),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CompositeIdentityEstimate:
    resonance_id: str
    fast_center_hz: float
    fast_center_source_kind: Literal["calibration", "pair"]
    fast_center_source_pair_index: int | None
    fast_center_reference_timestamp_s: float
    fast_center_release_sequence_index: int | None
    fast_center_release_timestamp_s: float
    active_fwhm_hz: float
    fwhm_source_kind: SparseLinewidthSourceKind
    fwhm_source_scan_index: int | None
    fwhm_reference_timestamp_s: float
    fwhm_release_sequence_index: int | None
    fwhm_release_timestamp_s: float
    live_q: float
    center_age_s: float
    fwhm_age_s: float
    center_release_age_s: float
    fwhm_release_age_s: float
    completed_fast_pairs: int
    completed_sparse_scans: int
    latest_fast_pair: TwoPointPairResult | None
    latest_sparse_scan: SparseLinewidthScanResult | None

    def __post_init__(self) -> None:
        resonance_id = _required_nonblank_string(self.resonance_id, "resonance_id")
        fast_center_hz = _finite_float(self.fast_center_hz, "fast_center_hz")
        active_fwhm_hz = _positive_float(self.active_fwhm_hz, "active_fwhm_hz")
        source_fields = _validate_source_fields(
            center_kind=self.fast_center_source_kind,
            center_index=self.fast_center_source_pair_index,
            center_reference=self.fast_center_reference_timestamp_s,
            center_release_index=self.fast_center_release_sequence_index,
            center_release_timestamp=self.fast_center_release_timestamp_s,
            width_kind=self.fwhm_source_kind,
            width_index=self.fwhm_source_scan_index,
            width_reference=self.fwhm_reference_timestamp_s,
            width_release_index=self.fwhm_release_sequence_index,
            width_release_timestamp=self.fwhm_release_timestamp_s,
            prefix="",
        )
        live_q = _finite_float(self.live_q, "live_q")
        if live_q != fast_center_hz / active_fwhm_hz:
            raise ValueError("live_q must equal fast center divided by active FWHM")
        ages = tuple(
            _nonnegative_float(value, name)
            for name, value in (
                ("center_age_s", self.center_age_s),
                ("fwhm_age_s", self.fwhm_age_s),
                ("center_release_age_s", self.center_release_age_s),
                ("fwhm_release_age_s", self.fwhm_release_age_s),
            )
        )
        completed_fast_pairs = _nonnegative_int(
            self.completed_fast_pairs, "completed_fast_pairs"
        )
        completed_sparse_scans = _nonnegative_int(
            self.completed_sparse_scans, "completed_sparse_scans"
        )
        if self.latest_fast_pair is not None:
            if type(self.latest_fast_pair) is not TwoPointPairResult:
                raise TypeError("latest_fast_pair must be an exact TwoPointPairResult")
            if (
                self.latest_fast_pair.resonance_id != resonance_id
                or self.latest_fast_pair.identity_pair_index != completed_fast_pairs - 1
            ):
                raise ValueError("latest_fast_pair must agree with identity history")
        elif completed_fast_pairs:
            raise ValueError("completed fast pairs require latest_fast_pair")
        if self.latest_sparse_scan is not None:
            if type(self.latest_sparse_scan) is not SparseLinewidthScanResult:
                raise TypeError(
                    "latest_sparse_scan must be an exact SparseLinewidthScanResult"
                )
            if (
                self.latest_sparse_scan.resonance_id != resonance_id
                or self.latest_sparse_scan.identity_scan_index
                != completed_sparse_scans - 1
            ):
                raise ValueError("latest_sparse_scan must agree with identity history")
        elif completed_sparse_scans:
            raise ValueError("completed sparse scans require latest_sparse_scan")
        if source_fields[0] == "pair" and (
            self.latest_fast_pair is None
            or source_fields[1] != self.latest_fast_pair.pair_index
            or fast_center_hz != self.latest_fast_pair.candidate_center_hz
            or source_fields[2] != self.latest_fast_pair.pair_reference_timestamp_s
            or source_fields[3] != self.latest_fast_pair.release_sequence_index
            or source_fields[4] != self.latest_fast_pair.release_timestamp_s
        ):
            raise ValueError(
                "pair fast-center source must equal latest successful pair"
            )
        if source_fields[5] == "scan" and (
            self.latest_sparse_scan is None
            or self.latest_sparse_scan.status != "success"
            or source_fields[6] != self.latest_sparse_scan.scan_index
            or active_fwhm_hz != self.latest_sparse_scan.fitted_fwhm_hz
            or source_fields[7] != self.latest_sparse_scan.public_reference_timestamp_s
            or source_fields[8] != self.latest_sparse_scan.release_sequence_index
            or source_fields[9] != self.latest_sparse_scan.release_timestamp_s
        ):
            raise ValueError("scan FWHM source must equal latest successful scan")
        for name, value in (
            ("resonance_id", resonance_id),
            ("fast_center_hz", fast_center_hz),
            ("active_fwhm_hz", active_fwhm_hz),
            ("live_q", live_q),
            ("center_age_s", ages[0]),
            ("fwhm_age_s", ages[1]),
            ("center_release_age_s", ages[2]),
            ("fwhm_release_age_s", ages[3]),
            ("completed_fast_pairs", completed_fast_pairs),
            ("completed_sparse_scans", completed_sparse_scans),
        ):
            object.__setattr__(self, name, value)
        for name, value in zip(
            (
                "fast_center_source_kind",
                "fast_center_source_pair_index",
                "fast_center_reference_timestamp_s",
                "fast_center_release_sequence_index",
                "fast_center_release_timestamp_s",
                "fwhm_source_kind",
                "fwhm_source_scan_index",
                "fwhm_reference_timestamp_s",
                "fwhm_release_sequence_index",
                "fwhm_release_timestamp_s",
            ),
            source_fields,
            strict=True,
        ):
            object.__setattr__(self, name, value)


def _resource_sum(
    first: PublicAcquisitionResources, second: PublicAcquisitionResources
) -> tuple[int, float, float, int, int, float]:
    return (
        first.observations + second.observations,
        first.integration_time_s + second.integration_time_s,
        first.nominal_exposure_photons + second.nominal_exposure_photons,
        first.realized_photons + second.realized_photons,
        first.observations_without_realized_counts
        + second.observations_without_realized_counts,
        first.virtual_elapsed_time_s + second.virtual_elapsed_time_s,
    )


@dataclass(frozen=True, slots=True)
class SparseLinewidthCompositeEstimate:
    configuration: SparseLinewidthConfiguration
    identities: tuple[CompositeIdentityEstimate, ...]
    calibration_source_id: str
    calibration_source_provenance: CalibrationSourceProvenance
    calibration_budget_treatment: CalibrationBudgetTreatment
    pending_mode: CompositeMode | None
    pending_query: TwoPointQuery | SparseLinewidthQuery | None
    incomplete_fast_pair: TwoPointPartialPair | None
    incomplete_sparse_scan: SparsePartialScan | None
    fast_pair_history: tuple[TwoPointPairResult, ...]
    sparse_scan_history: tuple[SparseLinewidthScanResult, ...]
    accepted_observations: int
    completed_fast_pairs: int
    completed_sparse_scans: int
    fast_pairs_since_scan: int
    current_sequence_index: int | None
    current_timestamp_s: float
    fast_tracking_resources: PublicAcquisitionResources
    sparse_tracking_resources: PublicAcquisitionResources
    tracking_resources: PublicAcquisitionResources
    calibration_resources: PublicAcquisitionResources
    charged_resources: PublicAcquisitionResources
    budget_ceiling: TwoPointBudgetCeiling
    stopped_reason: CompositeStopReason | None
    sparse_geometry_diagnostic: SparseGeometryUnavailableDiagnostic | None
    fast_update_cpu_time_s: float
    sparse_update_cpu_time_s: float
    total_update_cpu_time_s: float
    seed: int

    def __post_init__(self) -> None:
        if type(self.configuration) is not SparseLinewidthConfiguration:
            raise TypeError(
                "configuration must be an exact SparseLinewidthConfiguration"
            )
        if not isinstance(self.identities, (tuple, list)):
            raise TypeError("identities must be an ordered sequence")
        identities = tuple(self.identities)
        if len(identities) != 8 or not all(
            type(item) is CompositeIdentityEstimate for item in identities
        ):
            raise TypeError(
                "identities must contain exactly eight exact identity estimates"
            )
        if len({item.resonance_id for item in identities}) != 8:
            raise ValueError("identity estimates must have unique IDs")
        source_id = _required_nonblank_string(
            self.calibration_source_id, "calibration_source_id"
        )
        provenance = _closed_literal_string(
            self.calibration_source_provenance,
            "calibration_source_provenance",
            frozenset({"verified_factory_acquisition", "caller_asserted"}),
        )
        treatment = _closed_literal_string(
            self.calibration_budget_treatment,
            "calibration_budget_treatment",
            frozenset({"included_same_run", "conditional_free_precalibration"}),
        )
        if (
            treatment == "included_same_run"
            and provenance != "verified_factory_acquisition"
        ):
            raise ValueError("included treatment requires verified source provenance")
        pending_mode = _optional_closed_literal_string(
            self.pending_mode, "pending_mode", frozenset({"fast_pair", "sparse_scan"})
        )
        if self.pending_query is not None and type(self.pending_query) not in {
            TwoPointQuery,
            SparseLinewidthQuery,
        }:
            raise TypeError("pending_query must be an exact public query")
        if (pending_mode is None) != (self.pending_query is None):
            raise ValueError("pending mode and query must be jointly present")
        if (
            pending_mode == "fast_pair"
            and type(self.pending_query) is not TwoPointQuery
        ):
            raise ValueError("fast pending mode requires a TwoPointQuery")
        if (
            pending_mode == "sparse_scan"
            and type(self.pending_query) is not SparseLinewidthQuery
        ):
            raise ValueError("sparse pending mode requires a SparseLinewidthQuery")
        if (
            self.incomplete_fast_pair is not None
            and type(self.incomplete_fast_pair) is not TwoPointPartialPair
        ):
            raise TypeError("incomplete_fast_pair must be an exact TwoPointPartialPair")
        if (
            self.incomplete_sparse_scan is not None
            and type(self.incomplete_sparse_scan) is not SparsePartialScan
        ):
            raise TypeError("incomplete_sparse_scan must be an exact SparsePartialScan")
        if (
            self.incomplete_fast_pair is not None
            and self.incomplete_sparse_scan is not None
        ):
            raise ValueError("only one incomplete block is permitted")
        if not isinstance(self.fast_pair_history, (tuple, list)) or not isinstance(
            self.sparse_scan_history, (tuple, list)
        ):
            raise TypeError("histories must be ordered sequences")
        fast_history = tuple(self.fast_pair_history)
        sparse_history = tuple(self.sparse_scan_history)
        if not all(type(item) is TwoPointPairResult for item in fast_history):
            raise TypeError(
                "fast_pair_history must contain exact TwoPointPairResult values"
            )
        if not all(type(item) is SparseLinewidthScanResult for item in sparse_history):
            raise TypeError(
                "sparse_scan_history must contain exact "
                "SparseLinewidthScanResult values"
            )
        completed_fast_pairs = _nonnegative_int(
            self.completed_fast_pairs, "completed_fast_pairs"
        )
        completed_sparse_scans = _nonnegative_int(
            self.completed_sparse_scans, "completed_sparse_scans"
        )
        if completed_fast_pairs != len(fast_history) or completed_sparse_scans != len(
            sparse_history
        ):
            raise ValueError("completed counters must equal their history lengths")
        fast_pairs_since_scan = _nonnegative_int(
            self.fast_pairs_since_scan, "fast_pairs_since_scan"
        )
        if fast_pairs_since_scan >= self.configuration.scan_period_fast_pairs:
            raise ValueError("fast_pairs_since_scan must be below the scan period")
        expected_accepted = 2 * completed_fast_pairs + 5 * completed_sparse_scans
        if self.incomplete_fast_pair is not None:
            expected_accepted += 1
        if self.incomplete_sparse_scan is not None:
            expected_accepted += len(self.incomplete_sparse_scan.queries)
        accepted_observations = _nonnegative_int(
            self.accepted_observations, "accepted_observations"
        )
        if accepted_observations != expected_accepted:
            raise ValueError(
                "accepted observations must equal complete and partial blocks"
            )
        current_sequence_index = _optional_nonnegative_int(
            self.current_sequence_index, "current_sequence_index"
        )
        current_timestamp_s = _nonnegative_float(
            self.current_timestamp_s, "current_timestamp_s"
        )
        for name in (
            "fast_tracking_resources",
            "sparse_tracking_resources",
            "tracking_resources",
            "calibration_resources",
            "charged_resources",
        ):
            if type(getattr(self, name)) is not PublicAcquisitionResources:
                raise TypeError(f"{name} must be an exact PublicAcquisitionResources")
        if _resource_sum(
            self.fast_tracking_resources, self.sparse_tracking_resources
        ) != (
            self.tracking_resources.observations,
            self.tracking_resources.integration_time_s,
            self.tracking_resources.nominal_exposure_photons,
            self.tracking_resources.realized_photons,
            self.tracking_resources.observations_without_realized_counts,
            self.tracking_resources.virtual_elapsed_time_s,
        ):
            raise ValueError("tracking resources must equal fast plus sparse resources")
        if treatment == "conditional_free_precalibration":
            if self.charged_resources != self.tracking_resources:
                raise ValueError(
                    "conditional treatment charges tracking resources only"
                )
        elif _resource_sum(self.calibration_resources, self.tracking_resources) != (
            self.charged_resources.observations,
            self.charged_resources.integration_time_s,
            self.charged_resources.nominal_exposure_photons,
            self.charged_resources.realized_photons,
            self.charged_resources.observations_without_realized_counts,
            self.charged_resources.virtual_elapsed_time_s,
        ):
            raise ValueError("included treatment charges calibration plus tracking")
        if type(self.budget_ceiling) is not TwoPointBudgetCeiling:
            raise TypeError("budget_ceiling must be an exact TwoPointBudgetCeiling")
        stopped_reason = _optional_closed_literal_string(
            self.stopped_reason,
            "stopped_reason",
            frozenset({"budget_exhausted", "sparse_geometry_unavailable"}),
        )
        if (
            self.sparse_geometry_diagnostic is not None
            and type(self.sparse_geometry_diagnostic)
            is not SparseGeometryUnavailableDiagnostic
        ):
            raise TypeError("sparse_geometry_diagnostic must be exact")
        if (stopped_reason == "sparse_geometry_unavailable") != (
            self.sparse_geometry_diagnostic is not None
        ):
            raise ValueError("geometry stop and diagnostic must be jointly present")
        if stopped_reason is not None and (
            self.pending_query is not None
            or self.incomplete_fast_pair is not None
            or self.incomplete_sparse_scan is not None
        ):
            raise ValueError("stopped estimates must be at a block boundary")
        for index, pair in enumerate(fast_history):
            if (
                pair.pair_index != index
                or pair.resonance_id != identities[index % 8].resonance_id
            ):
                raise ValueError("fast history must follow global identity order")
        for index, scan in enumerate(sparse_history):
            if (
                scan.scan_index != index
                or scan.resonance_id != identities[index % 8].resonance_id
            ):
                raise ValueError("sparse history must follow global identity order")
        for identity in identities:
            own_pairs = tuple(
                item
                for item in fast_history
                if item.resonance_id == identity.resonance_id
            )
            own_scans = tuple(
                item
                for item in sparse_history
                if item.resonance_id == identity.resonance_id
            )
            if (
                identity.completed_fast_pairs != len(own_pairs)
                or identity.completed_sparse_scans != len(own_scans)
                or identity.latest_fast_pair != (own_pairs[-1] if own_pairs else None)
                or identity.latest_sparse_scan != (own_scans[-1] if own_scans else None)
            ):
                raise ValueError(
                    "identity histories and counters must equal aggregate histories"
                )
            if (
                identity.center_age_s
                != current_timestamp_s - identity.fast_center_reference_timestamp_s
                or identity.fwhm_age_s
                != current_timestamp_s - identity.fwhm_reference_timestamp_s
                or identity.center_release_age_s
                != current_timestamp_s - identity.fast_center_release_timestamp_s
                or identity.fwhm_release_age_s
                != current_timestamp_s - identity.fwhm_release_timestamp_s
            ):
                raise ValueError("identity ages must equal the current endpoint")
        cpu = tuple(
            _nonnegative_float(value, name)
            for name, value in (
                ("fast_update_cpu_time_s", self.fast_update_cpu_time_s),
                ("sparse_update_cpu_time_s", self.sparse_update_cpu_time_s),
                ("total_update_cpu_time_s", self.total_update_cpu_time_s),
            )
        )
        seed = _nonnegative_int(self.seed, "seed")
        for name, value in (
            ("identities", identities),
            ("calibration_source_id", source_id),
            ("calibration_source_provenance", provenance),
            ("calibration_budget_treatment", treatment),
            ("pending_mode", pending_mode),
            ("fast_pair_history", fast_history),
            ("sparse_scan_history", sparse_history),
            ("accepted_observations", accepted_observations),
            ("completed_fast_pairs", completed_fast_pairs),
            ("completed_sparse_scans", completed_sparse_scans),
            ("fast_pairs_since_scan", fast_pairs_since_scan),
            ("current_sequence_index", current_sequence_index),
            ("current_timestamp_s", current_timestamp_s),
            ("stopped_reason", stopped_reason),
            ("fast_update_cpu_time_s", cpu[0]),
            ("sparse_update_cpu_time_s", cpu[1]),
            ("total_update_cpu_time_s", cpu[2]),
            ("seed", seed),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class SparseLinewidthCompositeUpdate:
    query: TwoPointQuery | SparseLinewidthQuery
    observation: EstimatorObservation
    completed_fast_pair: TwoPointPairResult | None
    completed_sparse_scan: SparseLinewidthScanResult | None
    estimate: SparseLinewidthCompositeEstimate
    update_cpu_time_s: float

    def __post_init__(self) -> None:
        if type(self.query) not in {TwoPointQuery, SparseLinewidthQuery}:
            raise TypeError("query must be an exact public query")
        if type(self.query) is TwoPointQuery:
            _validate_observation_echo(
                frequency_hz=self.query.frequency_hz,
                integration_time_s=self.query.integration_time_s,
                expected_sequence_index=self.query.expected_sequence_index,
                expected_end_timestamp_s=self.query.expected_end_timestamp_s,
                expected_nominal_exposure_photons=self.query.expected_nominal_exposure_photons,
                observation=self.observation,
                name="observation",
            )
        else:
            _validate_observation_echo(
                frequency_hz=self.query.frequency_hz,
                integration_time_s=self.query.integration_time_s,
                expected_sequence_index=self.query.expected_sequence_index,
                expected_end_timestamp_s=self.query.expected_end_timestamp_s,
                expected_nominal_exposure_photons=self.query.expected_nominal_exposure_photons,
                observation=self.observation,
                name="observation",
            )
        if (
            self.completed_fast_pair is not None
            and type(self.completed_fast_pair) is not TwoPointPairResult
        ):
            raise TypeError("completed_fast_pair must be an exact TwoPointPairResult")
        if (
            self.completed_sparse_scan is not None
            and type(self.completed_sparse_scan) is not SparseLinewidthScanResult
        ):
            raise TypeError(
                "completed_sparse_scan must be an exact SparseLinewidthScanResult"
            )
        if (
            self.completed_fast_pair is not None
            and self.completed_sparse_scan is not None
        ):
            raise ValueError("one update cannot complete both block kinds")
        if type(self.estimate) is not SparseLinewidthCompositeEstimate:
            raise TypeError(
                "estimate must be an exact SparseLinewidthCompositeEstimate"
            )
        if self.estimate.pending_query is not None:
            raise ValueError("returned estimate must clear the accepted pending query")
        if type(self.query) is SparseLinewidthQuery:
            if self.completed_fast_pair is not None:
                raise ValueError("a sparse query cannot complete a fast pair")
            if self.completed_sparse_scan is None:
                partial = self.estimate.incomplete_sparse_scan
                if (
                    partial is None
                    or partial.queries[-1] != self.query
                    or partial.observations[-1] != self.observation
                ):
                    raise ValueError(
                        "partial sparse update must echo its accepted query"
                    )
            elif (
                self.estimate.incomplete_sparse_scan is not None
                or not self.estimate.sparse_scan_history
                or self.estimate.sparse_scan_history[-1] != self.completed_sparse_scan
                or self.completed_sparse_scan.queries[-1] != self.query
                or self.completed_sparse_scan.observations[-1] != self.observation
            ):
                raise ValueError("completed sparse update must echo the scan tail")
        else:
            if self.completed_sparse_scan is not None:
                raise ValueError("a fast query cannot complete a sparse scan")
            if self.completed_fast_pair is None:
                partial = self.estimate.incomplete_fast_pair
                if (
                    partial is None
                    or partial.first_query != self.query
                    or partial.first_observation != self.observation
                ):
                    raise ValueError("partial fast update must echo its accepted query")
            elif (
                self.estimate.incomplete_fast_pair is not None
                or not self.estimate.fast_pair_history
                or self.estimate.fast_pair_history[-1] != self.completed_fast_pair
            ):
                raise ValueError("completed fast update must echo the history tail")
        object.__setattr__(
            self,
            "update_cpu_time_s",
            _nonnegative_float(self.update_cpu_time_s, "update_cpu_time_s"),
        )
