"""Public primitive contracts for calibrated two-point tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np

CalibrationBudgetTreatment: TypeAlias = Literal[
    "included_same_run", "conditional_free_precalibration"
]
CalibrationSourceProvenance: TypeAlias = Literal[
    "verified_factory_acquisition", "caller_asserted"
]
CalibrationIdentityMode: TypeAlias = Literal[
    "require_expected_ids", "adopt_fit_ids"
]
ClockMappingKind: TypeAlias = Literal["shared_clock", "unit_scale_offset"]
PairSide: TypeAlias = Literal["minus", "plus"]
TwoPointLockState: TypeAlias = Literal["calibrated", "tracking", "step_limited", "lost"]
TwoPointFailureCode: TypeAlias = Literal[
    "invalid_pair_normalization",
    "numerical_failure",
    "common_mode_limit_exceeded",
    "capture_exceeded",
    "calibration_domain_exceeded",
]
TwoPointStopReason: TypeAlias = Literal["budget_exhausted"]
TwoPointCalibrationConstructionCode: TypeAlias = Literal[
    "invalid_argument_type",
    "invalid_argument_value",
    "invalid_provenance_or_quantity",
    "invalid_source_trace",
    "source_resource_mismatch",
    "fit_input_mismatch",
    "source_fit_failed",
    "source_identity_mismatch",
    "invalid_source_epoch",
    "invalid_availability_or_clock",
    "invalid_calibration_geometry",
    "invalid_budget_treatment",
]
TwoPointObservationValidationCode: TypeAlias = Literal[
    "invalid_observation_type",
    "no_pending_query",
    "sequence_mismatch",
    "frequency_mismatch",
    "integration_time_mismatch",
    "endpoint_mismatch",
    "nominal_exposure_mismatch",
    "invalid_observation_value",
]
TwoPointUpdateConstructionCode: TypeAlias = Literal[
    "partial_pair_construction_failed",
    "pair_result_construction_failed",
    "identity_estimate_construction_failed",
    "resource_construction_failed",
    "aggregate_estimate_construction_failed",
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


def _required_nonblank_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be nonblank")
    return value


def _canonical_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be an ordered sequence of strings")
    values = tuple(value)
    return tuple(_required_nonblank_string(item, name) for item in values)


def _validate_error(
    code: object, message: object, allowed_codes: frozenset[str]
) -> str:
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    if code not in allowed_codes:
        raise ValueError(f"unknown two-point error code: {code!r}")
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if not message:
        raise ValueError("message must be nonempty")
    return message


_CALIBRATION_CONSTRUCTION_CODES = frozenset(
    {
        "invalid_argument_type",
        "invalid_argument_value",
        "invalid_provenance_or_quantity",
        "invalid_source_trace",
        "source_resource_mismatch",
        "fit_input_mismatch",
        "source_fit_failed",
        "source_identity_mismatch",
        "invalid_source_epoch",
        "invalid_availability_or_clock",
        "invalid_calibration_geometry",
        "invalid_budget_treatment",
    }
)
_OBSERVATION_VALIDATION_CODES = frozenset(
    {
        "invalid_observation_type",
        "no_pending_query",
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
        "partial_pair_construction_failed",
        "pair_result_construction_failed",
        "identity_estimate_construction_failed",
        "resource_construction_failed",
        "aggregate_estimate_construction_failed",
    }
)


class TwoPointCalibrationConstructionError(ValueError):
    code: TwoPointCalibrationConstructionCode
    message: str

    def __init__(
        self, code: TwoPointCalibrationConstructionCode, message: str
    ) -> None:
        self.message = _validate_error(code, message, _CALIBRATION_CONSTRUCTION_CODES)
        self.code = code
        super().__init__(self.message)


class TwoPointObservationValidationError(ValueError):
    code: TwoPointObservationValidationCode
    message: str

    def __init__(
        self, code: TwoPointObservationValidationCode, message: str
    ) -> None:
        self.message = _validate_error(code, message, _OBSERVATION_VALIDATION_CODES)
        self.code = code
        super().__init__(self.message)


class TwoPointUpdateConstructionError(RuntimeError):
    code: TwoPointUpdateConstructionCode
    message: str

    def __init__(
        self, code: TwoPointUpdateConstructionCode, message: str
    ) -> None:
        self.message = _validate_error(code, message, _UPDATE_CONSTRUCTION_CODES)
        self.code = code
        super().__init__(self.message)


@dataclass(frozen=True, slots=True)
class PublicAcquisitionResources:
    observations: int
    integration_time_s: float
    nominal_exposure_photons: float
    realized_photons: int
    observations_without_realized_counts: int
    virtual_elapsed_time_s: float

    def __post_init__(self) -> None:
        observations = _nonnegative_int(self.observations, "observations")
        integration_time_s = _nonnegative_float(
            self.integration_time_s, "integration_time_s"
        )
        nominal_exposure_photons = _nonnegative_float(
            self.nominal_exposure_photons, "nominal_exposure_photons"
        )
        realized_photons = _nonnegative_int(self.realized_photons, "realized_photons")
        observations_without_realized_counts = _nonnegative_int(
            self.observations_without_realized_counts,
            "observations_without_realized_counts",
        )
        virtual_elapsed_time_s = _nonnegative_float(
            self.virtual_elapsed_time_s, "virtual_elapsed_time_s"
        )
        if observations_without_realized_counts > observations:
            raise ValueError(
                "observations_without_realized_counts cannot exceed observations"
            )
        if virtual_elapsed_time_s < integration_time_s:
            raise ValueError("virtual_elapsed_time_s must include integration_time_s")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "integration_time_s", integration_time_s)
        object.__setattr__(self, "nominal_exposure_photons", nominal_exposure_photons)
        object.__setattr__(self, "realized_photons", realized_photons)
        object.__setattr__(
            self,
            "observations_without_realized_counts",
            observations_without_realized_counts,
        )
        object.__setattr__(self, "virtual_elapsed_time_s", virtual_elapsed_time_s)


@dataclass(frozen=True, slots=True)
class TwoPointBudgetCeiling:
    max_observations: int | None
    max_integration_time_s: float | None
    max_nominal_exposure_photons: float | None
    max_virtual_elapsed_time_s: float | None

    def __post_init__(self) -> None:
        values = (
            self.max_observations,
            self.max_integration_time_s,
            self.max_nominal_exposure_photons,
            self.max_virtual_elapsed_time_s,
        )
        if all(value is None for value in values):
            raise ValueError("at least one budget ceiling must be provided")
        max_observations = (
            None
            if self.max_observations is None
            else _nonnegative_int(self.max_observations, "max_observations")
        )
        max_integration_time_s = (
            None
            if self.max_integration_time_s is None
            else _nonnegative_float(
                self.max_integration_time_s, "max_integration_time_s"
            )
        )
        max_nominal_exposure_photons = (
            None
            if self.max_nominal_exposure_photons is None
            else _nonnegative_float(
                self.max_nominal_exposure_photons, "max_nominal_exposure_photons"
            )
        )
        max_virtual_elapsed_time_s = (
            None
            if self.max_virtual_elapsed_time_s is None
            else _nonnegative_float(
                self.max_virtual_elapsed_time_s, "max_virtual_elapsed_time_s"
            )
        )
        object.__setattr__(self, "max_observations", max_observations)
        object.__setattr__(self, "max_integration_time_s", max_integration_time_s)
        object.__setattr__(
            self, "max_nominal_exposure_photons", max_nominal_exposure_photons
        )
        object.__setattr__(
            self, "max_virtual_elapsed_time_s", max_virtual_elapsed_time_s
        )


@dataclass(frozen=True, slots=True)
class TwoPointIdentityBinding:
    mode: CalibrationIdentityMode
    expected_resonance_ids: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if self.mode not in {"require_expected_ids", "adopt_fit_ids"}:
            raise ValueError("mode must be a supported calibration identity mode")
        if self.mode == "adopt_fit_ids":
            if self.expected_resonance_ids is not None:
                raise ValueError(
                    "adopt_fit_ids requires expected_resonance_ids to be None"
                )
            return
        if self.expected_resonance_ids is None:
            raise ValueError("require_expected_ids requires expected_resonance_ids")
        expected_resonance_ids = _canonical_string_tuple(
            self.expected_resonance_ids, "expected_resonance_ids"
        )
        if len(expected_resonance_ids) != 8:
            raise ValueError(
                "require_expected_ids requires exactly eight resonance IDs"
            )
        if len(set(expected_resonance_ids)) != 8:
            raise ValueError("require_expected_ids requires unique resonance IDs")
        object.__setattr__(self, "expected_resonance_ids", expected_resonance_ids)


@dataclass(frozen=True, slots=True)
class NormalizedFluorescenceProvenance:
    quantity: Literal["normalized_fluorescence"]
    normalization_rule: str
    nominal_photon_rate_hz: float
    sampling_rules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.quantity != "normalized_fluorescence":
            raise ValueError("quantity must be normalized_fluorescence")
        normalization_rule = _required_nonblank_string(
            self.normalization_rule, "normalization_rule"
        )
        nominal_photon_rate_hz = _positive_float(
            self.nominal_photon_rate_hz, "nominal_photon_rate_hz"
        )
        sampling_rules = _canonical_string_tuple(self.sampling_rules, "sampling_rules")
        if not sampling_rules:
            raise ValueError("sampling_rules must be nonempty")
        object.__setattr__(self, "normalization_rule", normalization_rule)
        object.__setattr__(self, "nominal_photon_rate_hz", nominal_photon_rate_hz)
        object.__setattr__(self, "sampling_rules", sampling_rules)


@dataclass(frozen=True, slots=True)
class TwoPointClockMapping:
    kind: ClockMappingKind
    source_clock_id: str
    tracker_clock_id: str
    scale: float
    offset_s: float

    def __post_init__(self) -> None:
        if self.kind not in {"shared_clock", "unit_scale_offset"}:
            raise ValueError("kind must be a supported clock mapping kind")
        source_clock_id = _required_nonblank_string(
            self.source_clock_id, "source_clock_id"
        )
        tracker_clock_id = _required_nonblank_string(
            self.tracker_clock_id, "tracker_clock_id"
        )
        scale = _finite_float(self.scale, "scale")
        offset_s = _finite_float(self.offset_s, "offset_s")
        if scale != 1.0:
            raise ValueError("scale must equal exactly 1.0")
        if self.kind == "shared_clock":
            if source_clock_id != tracker_clock_id:
                raise ValueError("shared_clock requires equal clock IDs")
            if offset_s != 0.0:
                raise ValueError("shared_clock requires a zero offset_s")
        elif source_clock_id == tracker_clock_id:
            raise ValueError("unit_scale_offset requires distinct clock IDs")
        object.__setattr__(self, "source_clock_id", source_clock_id)
        object.__setattr__(self, "tracker_clock_id", tracker_clock_id)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset_s", offset_s)


@dataclass(frozen=True, slots=True)
class TwoPointTrackerConfiguration:
    identity_binding: TwoPointIdentityBinding = field(
        default_factory=lambda: TwoPointIdentityBinding(
            mode="require_expected_ids",
            expected_resonance_ids=("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"),
        )
    )
    offset_fwhm_fraction: float = 0.35
    capture_fwhm_fraction: float = 0.20
    proportional_gain: float = 1.0
    max_step_fwhm_fraction: float = 0.10
    integration_time_s: float = 0.005
    common_mode_limit_target_depths: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity_binding, TwoPointIdentityBinding):
            raise TypeError("identity_binding must be a TwoPointIdentityBinding")
        offset_fwhm_fraction = _positive_float(
            self.offset_fwhm_fraction, "offset_fwhm_fraction"
        )
        capture_fwhm_fraction = _positive_float(
            self.capture_fwhm_fraction, "capture_fwhm_fraction"
        )
        if capture_fwhm_fraction >= offset_fwhm_fraction:
            raise ValueError(
                "capture_fwhm_fraction must be strictly less than offset_fwhm_fraction"
            )
        object.__setattr__(
            self,
            "offset_fwhm_fraction",
            offset_fwhm_fraction,
        )
        object.__setattr__(
            self,
            "capture_fwhm_fraction",
            capture_fwhm_fraction,
        )
        object.__setattr__(
            self,
            "proportional_gain",
            _positive_float(self.proportional_gain, "proportional_gain"),
        )
        object.__setattr__(
            self,
            "max_step_fwhm_fraction",
            _positive_float(self.max_step_fwhm_fraction, "max_step_fwhm_fraction"),
        )
        object.__setattr__(
            self,
            "integration_time_s",
            _positive_float(self.integration_time_s, "integration_time_s"),
        )
        common_mode_limit_target_depths = self.common_mode_limit_target_depths
        if common_mode_limit_target_depths is not None:
            common_mode_limit_target_depths = _positive_float(
                common_mode_limit_target_depths, "common_mode_limit_target_depths"
            )
        object.__setattr__(
            self, "common_mode_limit_target_depths", common_mode_limit_target_depths
        )


@dataclass(frozen=True, slots=True)
class TwoPointRunMetadata:
    tracker_clock_id: str
    current_sequence_index: int | None
    current_timestamp_s: float
    nominal_photon_rate_hz: float
    frequency_overhead_s: float
    fluorescence_quantity: Literal["normalized_fluorescence"]

    def __post_init__(self) -> None:
        tracker_clock_id = _required_nonblank_string(
            self.tracker_clock_id, "tracker_clock_id"
        )
        current_sequence_index = (
            None
            if self.current_sequence_index is None
            else _nonnegative_int(self.current_sequence_index, "current_sequence_index")
        )
        current_timestamp_s = _nonnegative_float(
            self.current_timestamp_s, "current_timestamp_s"
        )
        nominal_photon_rate_hz = _positive_float(
            self.nominal_photon_rate_hz, "nominal_photon_rate_hz"
        )
        frequency_overhead_s = _nonnegative_float(
            self.frequency_overhead_s, "frequency_overhead_s"
        )
        if self.fluorescence_quantity != "normalized_fluorescence":
            raise ValueError("fluorescence_quantity must be normalized_fluorescence")
        object.__setattr__(self, "tracker_clock_id", tracker_clock_id)
        object.__setattr__(self, "current_sequence_index", current_sequence_index)
        object.__setattr__(self, "current_timestamp_s", current_timestamp_s)
        object.__setattr__(self, "nominal_photon_rate_hz", nominal_photon_rate_hz)
        object.__setattr__(self, "frequency_overhead_s", frequency_overhead_s)
