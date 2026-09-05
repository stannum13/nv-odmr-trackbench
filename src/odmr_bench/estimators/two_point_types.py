"""Public primitive contracts for calibrated two-point tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from itertools import pairwise
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators.types import (
    FitConfiguration,
    FitInitialGuess,
    FitUncertainty,
    InitializationDiagnostics,
    SpectrumFitResult,
)

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


def _optional_closed_literal_string(
    value: object, name: str, allowed_values: frozenset[str]
) -> str | None:
    if value is None:
        return None
    return _closed_literal_string(value, name, allowed_values)


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


def _snapshot_tracker_configuration(
    value: TwoPointTrackerConfiguration,
) -> TwoPointTrackerConfiguration:
    """Return an independent value snapshot of tracker configuration data."""
    return TwoPointTrackerConfiguration(
        identity_binding=TwoPointIdentityBinding(
            value.identity_binding.mode,
            value.identity_binding.expected_resonance_ids,
        ),
        offset_fwhm_fraction=value.offset_fwhm_fraction,
        capture_fwhm_fraction=value.capture_fwhm_fraction,
        proportional_gain=value.proportional_gain,
        max_step_fwhm_fraction=value.max_step_fwhm_fraction,
        integration_time_s=value.integration_time_s,
        common_mode_limit_target_depths=value.common_mode_limit_target_depths,
    )


def _snapshot_initialization_diagnostics(
    value: InitializationDiagnostics,
) -> InitializationDiagnostics:
    return InitializationDiagnostics(
        source=value.source,
        candidate_count=value.candidate_count,
        selected_indices=value.selected_indices,
        used_fallback=value.used_fallback,
        messages=value.messages,
    )


def _snapshot_fit_uncertainty(value: FitUncertainty | None) -> FitUncertainty | None:
    if value is None:
        return None
    return FitUncertainty(
        baseline_standard_errors=np.array(
            value.baseline_standard_errors, dtype=np.float64, copy=True
        ),
        center_hz=np.array(value.center_hz, dtype=np.float64, copy=True),
        fwhm_hz=np.array(value.fwhm_hz, dtype=np.float64, copy=True),
        amplitude=np.array(value.amplitude, dtype=np.float64, copy=True),
        eta=(
            None
            if value.eta is None
            else np.array(value.eta, dtype=np.float64, copy=True)
        ),
        method=value.method,
    )


def _snapshot_spectrum_fit_result(value: SpectrumFitResult) -> SpectrumFitResult:
    initial_guess = (
        None
        if value.initial_guess is None
        else FitInitialGuess(
            resonances=value.initial_guess.resonances,
            baseline=value.initial_guess.baseline,
        )
    )
    return SpectrumFitResult(
        success=value.success,
        failure_code=value.failure_code,
        model_kind=value.model_kind,
        baseline_degree=value.baseline_degree,
        resonance_estimates=value.resonance_estimates,
        baseline_estimate=value.baseline_estimate,
        diagnostics=_snapshot_initialization_diagnostics(value.diagnostics),
        initial_guess=initial_guess,
        uncertainty=_snapshot_fit_uncertainty(value.uncertainty),
        uncertainty_reason=value.uncertainty_reason,
        scipy_status=value.scipy_status,
        scipy_message=value.scipy_message,
        nfev=value.nfev,
        cost=value.cost,
        residual_rmse=value.residual_rmse,
        residual_scale=value.residual_scale,
        degrees_of_freedom=value.degrees_of_freedom,
        jacobian_rank=value.jacobian_rank,
    )


@dataclass(frozen=True, slots=True)
class TwoPointCalibrationSource:
    source_id: str
    provenance: CalibrationSourceProvenance
    source_fit: SpectrumFitResult
    fit_configuration: FitConfiguration
    identity_binding: TwoPointIdentityBinding
    resolved_resonance_ids: tuple[str, ...]
    source_observations: tuple[EstimatorObservation, ...]
    fluorescence_provenance: NormalizedFluorescenceProvenance
    source_frequency_overhead_s: float
    source_frequency_min_hz: float
    source_frequency_max_hz: float
    source_first_sequence_index: int
    source_last_sequence_index: int
    source_start_timestamp_s: float
    source_first_timestamp_s: float
    source_last_timestamp_s: float
    physical_fit_epoch_s: float
    availability_sequence_index: int
    availability_timestamp_s: float
    safe_resources: PublicAcquisitionResources
    clock_mapping: TwoPointClockMapping

    def __post_init__(self) -> None:
        source_id = _required_nonblank_string(self.source_id, "source_id")
        if self.provenance == "verified_factory_acquisition":
            raise ValueError(
                "verified_factory_acquisition provenance requires the private factory"
            )
        if self.provenance != "caller_asserted":
            raise ValueError("provenance must be caller_asserted")
        if not isinstance(self.source_fit, SpectrumFitResult):
            raise TypeError("source_fit must be a SpectrumFitResult")
        if not self.source_fit.success:
            raise ValueError("source_fit must be successful")
        source_fit = _snapshot_spectrum_fit_result(self.source_fit)
        if not isinstance(self.fit_configuration, FitConfiguration):
            raise TypeError("fit_configuration must be a FitConfiguration")
        fit_configuration = replace(self.fit_configuration)
        if not isinstance(self.identity_binding, TwoPointIdentityBinding):
            raise TypeError("identity_binding must be a TwoPointIdentityBinding")
        identity_binding = TwoPointIdentityBinding(
            self.identity_binding.mode,
            self.identity_binding.expected_resonance_ids,
        )
        resolved_resonance_ids = _canonical_string_tuple(
            self.resolved_resonance_ids, "resolved_resonance_ids"
        )
        if len(resolved_resonance_ids) != 8 or len(set(resolved_resonance_ids)) != 8:
            raise ValueError("resolved_resonance_ids must contain eight unique IDs")
        if not isinstance(self.source_observations, (tuple, list)):
            raise TypeError("source_observations must be an ordered sequence")
        supplied_observations = tuple(self.source_observations)
        if not supplied_observations:
            raise ValueError("source_observations must be nonempty")
        if not all(
            isinstance(item, EstimatorObservation) for item in supplied_observations
        ):
            raise TypeError(
                "source_observations must contain EstimatorObservation values"
            )
        source_observations = tuple(
            EstimatorObservation(
                sequence_index=item.sequence_index,
                timestamp_s=item.timestamp_s,
                frequency_hz=item.frequency_hz,
                fluorescence=item.fluorescence,
                integration_time_s=item.integration_time_s,
                nominal_exposure_photons=item.nominal_exposure_photons,
                realized_photons=item.realized_photons,
            )
            for item in supplied_observations
        )
        if any(
            current.sequence_index != previous.sequence_index + 1
            for previous, current in pairwise(source_observations)
        ):
            raise ValueError(
                "source observations must have contiguous sequence indices"
            )
        if any(
            current.timestamp_s <= previous.timestamp_s
            for previous, current in pairwise(source_observations)
        ):
            raise ValueError(
                "source observation timestamps must be strictly increasing"
            )
        if any(
            current.frequency_hz <= previous.frequency_hz
            for previous, current in pairwise(source_observations)
        ):
            raise ValueError(
                "source observation frequencies must be strictly increasing"
            )
        if not isinstance(
            self.fluorescence_provenance, NormalizedFluorescenceProvenance
        ):
            raise TypeError(
                "fluorescence_provenance must be NormalizedFluorescenceProvenance"
            )
        fluorescence_provenance = NormalizedFluorescenceProvenance(
            self.fluorescence_provenance.quantity,
            self.fluorescence_provenance.normalization_rule,
            self.fluorescence_provenance.nominal_photon_rate_hz,
            self.fluorescence_provenance.sampling_rules,
        )
        source_frequency_overhead_s = _nonnegative_float(
            self.source_frequency_overhead_s, "source_frequency_overhead_s"
        )
        source_frequency_min_hz = _finite_float(
            self.source_frequency_min_hz, "source_frequency_min_hz"
        )
        source_frequency_max_hz = _finite_float(
            self.source_frequency_max_hz, "source_frequency_max_hz"
        )
        source_first_sequence_index = _nonnegative_int(
            self.source_first_sequence_index, "source_first_sequence_index"
        )
        source_last_sequence_index = _nonnegative_int(
            self.source_last_sequence_index, "source_last_sequence_index"
        )
        source_start_timestamp_s = _nonnegative_float(
            self.source_start_timestamp_s, "source_start_timestamp_s"
        )
        source_first_timestamp_s = _nonnegative_float(
            self.source_first_timestamp_s, "source_first_timestamp_s"
        )
        source_last_timestamp_s = _nonnegative_float(
            self.source_last_timestamp_s, "source_last_timestamp_s"
        )
        physical_fit_epoch_s = _nonnegative_float(
            self.physical_fit_epoch_s, "physical_fit_epoch_s"
        )
        availability_sequence_index = _nonnegative_int(
            self.availability_sequence_index, "availability_sequence_index"
        )
        availability_timestamp_s = _nonnegative_float(
            self.availability_timestamp_s, "availability_timestamp_s"
        )
        if not isinstance(self.safe_resources, PublicAcquisitionResources):
            raise TypeError("safe_resources must be PublicAcquisitionResources")
        safe_resources = PublicAcquisitionResources(
            self.safe_resources.observations,
            self.safe_resources.integration_time_s,
            self.safe_resources.nominal_exposure_photons,
            self.safe_resources.realized_photons,
            self.safe_resources.observations_without_realized_counts,
            self.safe_resources.virtual_elapsed_time_s,
        )
        if not isinstance(self.clock_mapping, TwoPointClockMapping):
            raise TypeError("clock_mapping must be a TwoPointClockMapping")
        clock_mapping = TwoPointClockMapping(
            self.clock_mapping.kind,
            self.clock_mapping.source_clock_id,
            self.clock_mapping.tracker_clock_id,
            self.clock_mapping.scale,
            self.clock_mapping.offset_s,
        )

        first_observation = source_observations[0]
        last_observation = source_observations[-1]
        if (
            source_frequency_min_hz != first_observation.frequency_hz
            or source_frequency_max_hz != last_observation.frequency_hz
            or source_first_sequence_index != first_observation.sequence_index
            or source_last_sequence_index != last_observation.sequence_index
            or source_first_timestamp_s != first_observation.timestamp_s
            or source_last_timestamp_s != last_observation.timestamp_s
        ):
            raise ValueError("source endpoints must equal the safe observation trace")
        previous_timestamp_s = source_start_timestamp_s
        for observation in source_observations:
            expected_timestamp_s = (
                previous_timestamp_s
                + source_frequency_overhead_s
                + observation.integration_time_s
            )
            if observation.timestamp_s != expected_timestamp_s:
                raise ValueError(
                    "source observation timestamps must follow the endpoint recurrence"
                )
            previous_timestamp_s = observation.timestamp_s
        if (
            availability_sequence_index != last_observation.sequence_index
            or availability_timestamp_s != last_observation.timestamp_s
        ):
            raise ValueError("availability must equal the final source observation")
        expected_epoch_s = (
            first_observation.timestamp_s - first_observation.integration_time_s / 2.0
        )
        last_public_midpoint_s = (
            last_observation.timestamp_s - last_observation.integration_time_s / 2.0
        )
        expected_epoch_s += (last_public_midpoint_s - expected_epoch_s) / 2.0
        if physical_fit_epoch_s != expected_epoch_s:
            raise ValueError(
                "caller_asserted physical_fit_epoch_s must be the public midpoint"
            )
        from odmr_bench.estimators.two_point_resources import (
            _replay_public_resources,
        )

        expected_resources = _replay_public_resources(
            source_observations, source_frequency_overhead_s
        )
        if safe_resources != expected_resources:
            raise ValueError("safe_resources must equal the source observation trace")
        fit_ids = tuple(item.resonance_id for item in source_fit.resonance_estimates)
        if (
            fit_ids != resolved_resonance_ids
            or fit_configuration.resonance_ids != resolved_resonance_ids
            or source_fit.model_kind != fit_configuration.model_kind
            or source_fit.baseline_degree != fit_configuration.baseline_degree
        ):
            raise ValueError("source fit, configuration, and resolved IDs must agree")
        if (
            identity_binding.mode == "require_expected_ids"
            and identity_binding.expected_resonance_ids != resolved_resonance_ids
        ):
            raise ValueError("identity binding IDs must equal resolved IDs")

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_fit", source_fit)
        object.__setattr__(self, "fit_configuration", fit_configuration)
        object.__setattr__(self, "identity_binding", identity_binding)
        object.__setattr__(self, "resolved_resonance_ids", resolved_resonance_ids)
        object.__setattr__(self, "source_observations", source_observations)
        object.__setattr__(self, "fluorescence_provenance", fluorescence_provenance)
        object.__setattr__(
            self, "source_frequency_overhead_s", source_frequency_overhead_s
        )
        object.__setattr__(self, "source_frequency_min_hz", source_frequency_min_hz)
        object.__setattr__(self, "source_frequency_max_hz", source_frequency_max_hz)
        object.__setattr__(
            self, "source_first_sequence_index", source_first_sequence_index
        )
        object.__setattr__(
            self, "source_last_sequence_index", source_last_sequence_index
        )
        object.__setattr__(self, "source_start_timestamp_s", source_start_timestamp_s)
        object.__setattr__(self, "source_first_timestamp_s", source_first_timestamp_s)
        object.__setattr__(self, "source_last_timestamp_s", source_last_timestamp_s)
        object.__setattr__(self, "physical_fit_epoch_s", physical_fit_epoch_s)
        object.__setattr__(
            self, "availability_sequence_index", availability_sequence_index
        )
        object.__setattr__(self, "availability_timestamp_s", availability_timestamp_s)
        object.__setattr__(self, "safe_resources", safe_resources)
        object.__setattr__(self, "clock_mapping", clock_mapping)


@dataclass(frozen=True, slots=True)
class TwoPointIdentityCalibration:
    resonance_id: str
    source_fit_index: int
    calibration_center_hz: float
    calibration_fwhm_hz: float
    calibration_amplitude: float
    calibration_eta: float
    offset_hz: float
    capture_radius_hz: float
    max_step_hz: float
    target_pair_depth: float
    calibration_cell_lower_hz: float
    calibration_cell_upper_hz: float
    allowed_center_min_hz: float
    allowed_center_max_hz: float

    def __post_init__(self) -> None:
        resonance_id = _required_nonblank_string(self.resonance_id, "resonance_id")
        source_fit_index = _nonnegative_int(self.source_fit_index, "source_fit_index")
        calibration_center_hz = _finite_float(
            self.calibration_center_hz, "calibration_center_hz"
        )
        calibration_fwhm_hz = _positive_float(
            self.calibration_fwhm_hz, "calibration_fwhm_hz"
        )
        calibration_amplitude = _positive_float(
            self.calibration_amplitude, "calibration_amplitude"
        )
        calibration_eta = _nonnegative_float(self.calibration_eta, "calibration_eta")
        if calibration_eta > 1.0:
            raise ValueError("calibration_eta must not exceed one")
        offset_hz = _positive_float(self.offset_hz, "offset_hz")
        capture_radius_hz = _positive_float(self.capture_radius_hz, "capture_radius_hz")
        max_step_hz = _positive_float(self.max_step_hz, "max_step_hz")
        target_pair_depth = _positive_float(self.target_pair_depth, "target_pair_depth")
        calibration_cell_lower_hz = _finite_float(
            self.calibration_cell_lower_hz, "calibration_cell_lower_hz"
        )
        calibration_cell_upper_hz = _finite_float(
            self.calibration_cell_upper_hz, "calibration_cell_upper_hz"
        )
        allowed_center_min_hz = _finite_float(
            self.allowed_center_min_hz, "allowed_center_min_hz"
        )
        allowed_center_max_hz = _finite_float(
            self.allowed_center_max_hz, "allowed_center_max_hz"
        )
        if calibration_cell_lower_hz >= calibration_cell_upper_hz:
            raise ValueError("calibration cell must be nonempty")
        if allowed_center_min_hz > allowed_center_max_hz:
            raise ValueError("allowed center interval must be nonempty")
        if not (
            calibration_cell_lower_hz
            <= calibration_center_hz
            <= calibration_cell_upper_hz
            and allowed_center_min_hz <= calibration_center_hz <= allowed_center_max_hz
        ):
            raise ValueError("calibration center must remain in its declared geometry")
        object.__setattr__(self, "resonance_id", resonance_id)
        object.__setattr__(self, "source_fit_index", source_fit_index)
        object.__setattr__(self, "calibration_center_hz", calibration_center_hz)
        object.__setattr__(self, "calibration_fwhm_hz", calibration_fwhm_hz)
        object.__setattr__(self, "calibration_amplitude", calibration_amplitude)
        object.__setattr__(self, "calibration_eta", calibration_eta)
        object.__setattr__(self, "offset_hz", offset_hz)
        object.__setattr__(self, "capture_radius_hz", capture_radius_hz)
        object.__setattr__(self, "max_step_hz", max_step_hz)
        object.__setattr__(self, "target_pair_depth", target_pair_depth)
        object.__setattr__(self, "calibration_cell_lower_hz", calibration_cell_lower_hz)
        object.__setattr__(self, "calibration_cell_upper_hz", calibration_cell_upper_hz)
        object.__setattr__(self, "allowed_center_min_hz", allowed_center_min_hz)
        object.__setattr__(self, "allowed_center_max_hz", allowed_center_max_hz)


@dataclass(frozen=True, slots=True)
class TwoPointCalibration:
    source: TwoPointCalibrationSource
    configuration: TwoPointTrackerConfiguration
    budget_treatment: CalibrationBudgetTreatment
    identities: tuple[TwoPointIdentityCalibration, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, TwoPointCalibrationSource):
            raise TypeError("source must be a TwoPointCalibrationSource")
        if not isinstance(self.configuration, TwoPointTrackerConfiguration):
            raise TypeError("configuration must be a TwoPointTrackerConfiguration")
        configuration = _snapshot_tracker_configuration(self.configuration)
        if self.budget_treatment not in {
            "included_same_run",
            "conditional_free_precalibration",
        }:
            raise ValueError("budget_treatment must be supported")
        if (
            self.source.provenance == "caller_asserted"
            and self.budget_treatment != "conditional_free_precalibration"
        ):
            raise ValueError(
                "caller_asserted sources require conditional_free_precalibration"
            )
        if not isinstance(self.identities, (tuple, list)):
            raise TypeError("identities must be an ordered sequence")
        supplied_identities = tuple(self.identities)
        if not all(
            isinstance(item, TwoPointIdentityCalibration)
            for item in supplied_identities
        ):
            raise TypeError(
                "identities must contain TwoPointIdentityCalibration values"
            )
        identities = tuple(replace(item) for item in supplied_identities)
        if len(identities) != 8:
            raise ValueError("identities must contain exactly eight calibrations")
        identity_ids = tuple(item.resonance_id for item in identities)
        if len(set(identity_ids)) != 8:
            raise ValueError("identities must have unique resonance IDs")
        if identity_ids != self.source.resolved_resonance_ids:
            raise ValueError("identity IDs must equal the source resolved IDs in order")
        if configuration.identity_binding != self.source.identity_binding:
            raise ValueError("configuration identity binding must equal source binding")
        for index, identity in enumerate(identities):
            fit_resonance = self.source.source_fit.resonance_estimates[index]
            if (
                identity.source_fit_index != index
                or identity.resonance_id != fit_resonance.resonance_id
                or identity.calibration_center_hz != fit_resonance.center_hz
                or identity.calibration_fwhm_hz != fit_resonance.fwhm_hz
                or identity.calibration_amplitude != fit_resonance.amplitude
                or identity.calibration_eta != fit_resonance.eta
            ):
                raise ValueError(
                    "identities must preserve source fit IDs and model values"
                )
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "identities", identities)


def _opposite_side(side: PairSide) -> PairSide:
    return "plus" if side == "minus" else "minus"


def _optional_finite_float(value: object, name: str) -> float | None:
    return None if value is None else _finite_float(value, name)


def _validate_query_observation_echo(
    query: TwoPointQuery,
    observation: EstimatorObservation,
    name: str,
) -> None:
    if type(observation) is not EstimatorObservation:
        raise TypeError(f"{name} must be an exact EstimatorObservation")
    if (
        observation.sequence_index != query.expected_sequence_index
        or observation.frequency_hz != query.frequency_hz
        or observation.integration_time_s != query.integration_time_s
        or observation.timestamp_s != query.expected_end_timestamp_s
        or observation.nominal_exposure_photons
        != query.expected_nominal_exposure_photons
    ):
        raise ValueError(f"{name} must echo its query")


def _validate_pair_diagnostic_state(
    *,
    lock_state: object,
    failure_code: object,
    zero_discriminator: float | None,
    discriminator_slope_per_hz: float | None,
    discriminator: float | None,
    raw_innovation_hz: float | None,
    requested_step_hz: float | None,
    candidate_center_hz: float | None,
    applied_step_hz: float,
    common_mode_target_depths: float | None,
    interrogation_center_hz: float,
) -> None:
    if lock_state not in {"tracking", "step_limited", "lost"}:
        raise ValueError("pair lock_state must be tracking, step_limited, or lost")
    if failure_code is not None and failure_code not in {
        "invalid_pair_normalization",
        "numerical_failure",
        "common_mode_limit_exceeded",
        "capture_exceeded",
        "calibration_domain_exceeded",
    }:
        raise ValueError("failure_code must be a supported two-point failure")
    diagnostic_values = (
        discriminator,
        common_mode_target_depths,
        raw_innovation_hz,
        requested_step_hz,
        candidate_center_hz,
    )
    geometry_available = zero_discriminator is not None
    if geometry_available != (discriminator_slope_per_hz is not None):
        raise ValueError("pair-local zero and slope must be jointly available")
    if not geometry_available and not (
        lock_state == "lost"
        and failure_code in {"invalid_pair_normalization", "numerical_failure"}
        and all(value is None for value in diagnostic_values)
    ):
        raise ValueError(
            "unavailable pair-local geometry requires an early scientific loss"
        )
    if failure_code == "invalid_pair_normalization" and geometry_available:
        raise ValueError("invalid normalization cannot publish pair-local geometry")
    if lock_state in {"tracking", "step_limited"}:
        if failure_code is not None or any(
            value is None for value in diagnostic_values
        ):
            raise ValueError(
                "successful pairs require complete diagnostics and no failure"
            )
        if candidate_center_hz != interrogation_center_hz + applied_step_hz:
            raise ValueError("successful candidate must equal the applied center step")
        if lock_state == "tracking" and applied_step_hz != requested_step_hz:
            raise ValueError("tracking must apply the requested step exactly")
        if lock_state == "step_limited" and (
            requested_step_hz == 0.0
            or abs(applied_step_hz) >= abs(requested_step_hz)
            or applied_step_hz * requested_step_hz <= 0.0
        ):
            raise ValueError("step_limited must apply a strict same-sign clip")
        return
    if failure_code is None:
        raise ValueError("lost pairs require a failure_code")
    if applied_step_hz != 0.0:
        raise ValueError("lost pairs must apply exactly zero step")
    if failure_code == "invalid_pair_normalization":
        if any(value is not None for value in diagnostic_values):
            raise ValueError("invalid normalization cannot retain later diagnostics")
        return
    if failure_code == "numerical_failure":
        missing_seen = False
        for value in diagnostic_values:
            if value is None:
                missing_seen = True
            elif missing_seen:
                raise ValueError(
                    "numerical failure diagnostics must form a valid prefix"
                )
        return
    expected_presence = {
        "common_mode_limit_exceeded": (True, True, False, False, False),
        "capture_exceeded": (True, True, True, False, False),
        "calibration_domain_exceeded": (True, True, True, True, True),
    }[failure_code]
    actual_presence = tuple(value is not None for value in diagnostic_values)
    if actual_presence != expected_presence:
        raise ValueError("policy failure diagnostics must match their exact prefix")


@dataclass(frozen=True, slots=True)
class TwoPointQuery:
    query_index: int
    pair_index: int
    identity_pair_index: int
    resonance_id: str
    side: PairSide
    interrogation_center_hz: float
    frequency_hz: float
    integration_time_s: float
    expected_sequence_index: int
    expected_end_timestamp_s: float
    expected_nominal_exposure_photons: float

    def __post_init__(self) -> None:
        query_index = _nonnegative_int(self.query_index, "query_index")
        pair_index = _nonnegative_int(self.pair_index, "pair_index")
        identity_pair_index = _nonnegative_int(
            self.identity_pair_index, "identity_pair_index"
        )
        resonance_id = _required_nonblank_string(self.resonance_id, "resonance_id")
        side = _closed_literal_string(
            self.side, "side", frozenset({"minus", "plus"})
        )
        interrogation_center_hz = _finite_float(
            self.interrogation_center_hz, "interrogation_center_hz"
        )
        frequency_hz = _finite_float(self.frequency_hz, "frequency_hz")
        integration_time_s = _positive_float(
            self.integration_time_s, "integration_time_s"
        )
        expected_sequence_index = _nonnegative_int(
            self.expected_sequence_index, "expected_sequence_index"
        )
        expected_end_timestamp_s = _nonnegative_float(
            self.expected_end_timestamp_s, "expected_end_timestamp_s"
        )
        expected_nominal_exposure_photons = _nonnegative_float(
            self.expected_nominal_exposure_photons,
            "expected_nominal_exposure_photons",
        )
        if pair_index != query_index // 2:
            raise ValueError("pair_index must equal query_index // 2")
        if identity_pair_index != pair_index // 8:
            raise ValueError("identity_pair_index must equal pair_index // 8")
        first_side = "minus" if identity_pair_index % 2 == 0 else "plus"
        expected_side = (
            first_side if query_index % 2 == 0 else _opposite_side(first_side)
        )
        if side != expected_side:
            raise ValueError("side must follow the identity pair arrival order")
        if side == "minus" and frequency_hz >= interrogation_center_hz:
            raise ValueError("minus frequency must be below interrogation center")
        if side == "plus" and frequency_hz <= interrogation_center_hz:
            raise ValueError("plus frequency must be above interrogation center")
        if expected_end_timestamp_s < integration_time_s:
            raise ValueError("expected endpoint must include integration time")
        object.__setattr__(self, "query_index", query_index)
        object.__setattr__(self, "pair_index", pair_index)
        object.__setattr__(self, "identity_pair_index", identity_pair_index)
        object.__setattr__(self, "resonance_id", resonance_id)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "interrogation_center_hz", interrogation_center_hz)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "integration_time_s", integration_time_s)
        object.__setattr__(self, "expected_sequence_index", expected_sequence_index)
        object.__setattr__(self, "expected_end_timestamp_s", expected_end_timestamp_s)
        object.__setattr__(
            self,
            "expected_nominal_exposure_photons",
            expected_nominal_exposure_photons,
        )


@dataclass(frozen=True, slots=True)
class TwoPointPartialPair:
    pair_index: int
    identity_pair_index: int
    resonance_id: str
    interrogation_center_hz: float
    first_side: PairSide
    first_query: TwoPointQuery
    first_observation: EstimatorObservation

    def __post_init__(self) -> None:
        pair_index = _nonnegative_int(self.pair_index, "pair_index")
        identity_pair_index = _nonnegative_int(
            self.identity_pair_index, "identity_pair_index"
        )
        resonance_id = _required_nonblank_string(self.resonance_id, "resonance_id")
        interrogation_center_hz = _finite_float(
            self.interrogation_center_hz, "interrogation_center_hz"
        )
        first_side = _closed_literal_string(
            self.first_side, "first_side", frozenset({"minus", "plus"})
        )
        if type(self.first_query) is not TwoPointQuery:
            raise TypeError("first_query must be an exact TwoPointQuery")
        if self.first_query.query_index % 2 != 0:
            raise ValueError("first_query must be the first query of its pair")
        if (
            pair_index != self.first_query.pair_index
            or identity_pair_index != self.first_query.identity_pair_index
            or resonance_id != self.first_query.resonance_id
            or interrogation_center_hz != self.first_query.interrogation_center_hz
            or first_side != self.first_query.side
        ):
            raise ValueError("partial pair fields must echo first_query")
        _validate_query_observation_echo(
            self.first_query, self.first_observation, "first_observation"
        )
        object.__setattr__(self, "pair_index", pair_index)
        object.__setattr__(self, "identity_pair_index", identity_pair_index)
        object.__setattr__(self, "resonance_id", resonance_id)
        object.__setattr__(self, "interrogation_center_hz", interrogation_center_hz)
        object.__setattr__(self, "first_side", first_side)


@dataclass(frozen=True, slots=True)
class TwoPointPairResult:
    pair_index: int
    identity_pair_index: int
    resonance_id: str
    interrogation_center_hz: float
    first_side: PairSide
    minus_query: TwoPointQuery
    plus_query: TwoPointQuery
    minus_observation: EstimatorObservation
    plus_observation: EstimatorObservation
    pair_reference_timestamp_s: float
    release_sequence_index: int
    release_timestamp_s: float
    discriminator: float | None
    zero_discriminator: float | None
    discriminator_slope_per_hz: float | None
    raw_innovation_hz: float | None
    requested_step_hz: float | None
    candidate_center_hz: float | None
    applied_step_hz: float
    common_mode_target_depths: float | None
    lock_state: TwoPointLockState
    failure_code: TwoPointFailureCode | None

    def __post_init__(self) -> None:
        pair_index = _nonnegative_int(self.pair_index, "pair_index")
        identity_pair_index = _nonnegative_int(
            self.identity_pair_index, "identity_pair_index"
        )
        resonance_id = _required_nonblank_string(self.resonance_id, "resonance_id")
        interrogation_center_hz = _finite_float(
            self.interrogation_center_hz, "interrogation_center_hz"
        )
        first_side = _closed_literal_string(
            self.first_side, "first_side", frozenset({"minus", "plus"})
        )
        lock_state = _closed_literal_string(
            self.lock_state,
            "lock_state",
            frozenset({"tracking", "step_limited", "lost"}),
        )
        failure_code = _optional_closed_literal_string(
            self.failure_code,
            "failure_code",
            frozenset(
                {
                    "invalid_pair_normalization",
                    "numerical_failure",
                    "common_mode_limit_exceeded",
                    "capture_exceeded",
                    "calibration_domain_exceeded",
                }
            ),
        )
        if type(self.minus_query) is not TwoPointQuery or type(
            self.plus_query
        ) is not TwoPointQuery:
            raise TypeError("pair queries must be exact TwoPointQuery values")
        if self.minus_query.side != "minus" or self.plus_query.side != "plus":
            raise ValueError("minus_query and plus_query must match their sides")
        for query in (self.minus_query, self.plus_query):
            if (
                query.pair_index != pair_index
                or query.identity_pair_index != identity_pair_index
                or query.resonance_id != resonance_id
                or query.interrogation_center_hz != interrogation_center_hz
            ):
                raise ValueError("pair fields must echo both queries")
        first_query, second_query = (
            (self.minus_query, self.plus_query)
            if first_side == "minus"
            else (self.plus_query, self.minus_query)
        )
        if (
            first_query.query_index != 2 * pair_index
            or second_query.query_index != first_query.query_index + 1
        ):
            raise ValueError("pair queries must be adjacent in arrival order")
        _validate_query_observation_echo(
            self.minus_query, self.minus_observation, "minus_observation"
        )
        _validate_query_observation_echo(
            self.plus_query, self.plus_observation, "plus_observation"
        )
        first_observation, second_observation = (
            (self.minus_observation, self.plus_observation)
            if first_side == "minus"
            else (self.plus_observation, self.minus_observation)
        )
        if (
            second_observation.sequence_index != first_observation.sequence_index + 1
            or second_observation.timestamp_s <= first_observation.timestamp_s
        ):
            raise ValueError("pair observations must be adjacent in arrival order")
        pair_reference_timestamp_s = _nonnegative_float(
            self.pair_reference_timestamp_s, "pair_reference_timestamp_s"
        )
        first_reference_s = (
            first_observation.timestamp_s
            - first_observation.integration_time_s / 2.0
        )
        second_reference_s = (
            second_observation.timestamp_s
            - second_observation.integration_time_s / 2.0
        )
        expected_reference_s = first_reference_s + (
            second_reference_s - first_reference_s
        ) / 2.0
        if pair_reference_timestamp_s != expected_reference_s:
            raise ValueError("pair reference must equal the ordered public midpoint")
        release_sequence_index = _nonnegative_int(
            self.release_sequence_index, "release_sequence_index"
        )
        release_timestamp_s = _nonnegative_float(
            self.release_timestamp_s, "release_timestamp_s"
        )
        if (
            release_sequence_index != second_observation.sequence_index
            or release_timestamp_s != second_observation.timestamp_s
        ):
            raise ValueError("pair release must equal the second observation endpoint")
        discriminator = _optional_finite_float(self.discriminator, "discriminator")
        zero_discriminator = _optional_finite_float(
            self.zero_discriminator, "zero_discriminator"
        )
        discriminator_slope_per_hz = _optional_finite_float(
            self.discriminator_slope_per_hz, "discriminator_slope_per_hz"
        )
        if (
            discriminator_slope_per_hz is not None
            and discriminator_slope_per_hz <= 0.0
        ):
            raise ValueError("discriminator_slope_per_hz must be positive")
        raw_innovation_hz = _optional_finite_float(
            self.raw_innovation_hz, "raw_innovation_hz"
        )
        requested_step_hz = _optional_finite_float(
            self.requested_step_hz, "requested_step_hz"
        )
        candidate_center_hz = _optional_finite_float(
            self.candidate_center_hz, "candidate_center_hz"
        )
        applied_step_hz = _finite_float(self.applied_step_hz, "applied_step_hz")
        common_mode_target_depths = _optional_finite_float(
            self.common_mode_target_depths, "common_mode_target_depths"
        )
        _validate_pair_diagnostic_state(
            lock_state=lock_state,
            failure_code=failure_code,
            zero_discriminator=zero_discriminator,
            discriminator_slope_per_hz=discriminator_slope_per_hz,
            discriminator=discriminator,
            raw_innovation_hz=raw_innovation_hz,
            requested_step_hz=requested_step_hz,
            candidate_center_hz=candidate_center_hz,
            applied_step_hz=applied_step_hz,
            common_mode_target_depths=common_mode_target_depths,
            interrogation_center_hz=interrogation_center_hz,
        )
        object.__setattr__(self, "pair_index", pair_index)
        object.__setattr__(self, "identity_pair_index", identity_pair_index)
        object.__setattr__(self, "resonance_id", resonance_id)
        object.__setattr__(self, "interrogation_center_hz", interrogation_center_hz)
        object.__setattr__(self, "first_side", first_side)
        object.__setattr__(
            self, "pair_reference_timestamp_s", pair_reference_timestamp_s
        )
        object.__setattr__(self, "release_sequence_index", release_sequence_index)
        object.__setattr__(self, "release_timestamp_s", release_timestamp_s)
        object.__setattr__(self, "discriminator", discriminator)
        object.__setattr__(self, "zero_discriminator", zero_discriminator)
        object.__setattr__(
            self, "discriminator_slope_per_hz", discriminator_slope_per_hz
        )
        object.__setattr__(self, "raw_innovation_hz", raw_innovation_hz)
        object.__setattr__(self, "requested_step_hz", requested_step_hz)
        object.__setattr__(self, "candidate_center_hz", candidate_center_hz)
        object.__setattr__(self, "applied_step_hz", applied_step_hz)
        object.__setattr__(
            self, "common_mode_target_depths", common_mode_target_depths
        )
        object.__setattr__(self, "lock_state", lock_state)
        object.__setattr__(self, "failure_code", failure_code)


def _pair_observations_in_arrival_order(
    pair: TwoPointPairResult,
) -> tuple[EstimatorObservation, EstimatorObservation]:
    if pair.first_side == "minus":
        return pair.minus_observation, pair.plus_observation
    return pair.plus_observation, pair.minus_observation


def _estimate_observation_trace(
    pair_history: tuple[TwoPointPairResult, ...],
    incomplete_pair: TwoPointPartialPair | None,
) -> tuple[EstimatorObservation, ...]:
    observations: list[EstimatorObservation] = []
    for pair in pair_history:
        observations.extend(_pair_observations_in_arrival_order(pair))
    if incomplete_pair is not None:
        observations.append(incomplete_pair.first_observation)
    return tuple(observations)


def _validate_estimate_history(
    identities: tuple[TwoPointIdentityEstimate, ...],
    pair_history: tuple[TwoPointPairResult, ...],
) -> None:
    for pair_index, pair in enumerate(pair_history):
        target = identities[pair_index % len(identities)]
        if pair.pair_index != pair_index or pair.resonance_id != target.resonance_id:
            raise ValueError("pair history must follow global index and identity order")
    for identity in identities:
        identity_pairs = tuple(
            pair for pair in pair_history if pair.resonance_id == identity.resonance_id
        )
        expected_latest = identity_pairs[-1] if identity_pairs else None
        if (
            identity.completed_pairs != len(identity_pairs)
            or identity.latest_pair != expected_latest
        ):
            raise ValueError("identity counters and latest pair must equal history")
        successful_pairs = tuple(
            pair
            for pair in identity_pairs
            if pair.lock_state in {"tracking", "step_limited"}
        )
        if identity.active_source_kind == "calibration":
            if successful_pairs:
                raise ValueError("successful history requires a pair active source")
            continue
        active_pair = successful_pairs[-1] if successful_pairs else None
        if active_pair is None or (
            identity.active_source_pair_index != active_pair.pair_index
            or identity.active_reference_timestamp_s
            != active_pair.pair_reference_timestamp_s
            or identity.active_release_sequence_index
            != active_pair.release_sequence_index
            or identity.active_release_timestamp_s != active_pair.release_timestamp_s
            or identity.center_hz != active_pair.candidate_center_hz
        ):
            raise ValueError(
                "pair active source must equal the last successful history pair"
            )
    if sum(identity.completed_pairs for identity in identities) != len(pair_history):
        raise ValueError("identity completed-pair counts must equal aggregate history")


def _validate_accepted_trace_endpoint(
    pair_history: tuple[TwoPointPairResult, ...],
    incomplete_pair: TwoPointPartialPair | None,
    *,
    current_sequence_index: int | None,
    current_timestamp_s: float,
) -> None:
    trace = _estimate_observation_trace(pair_history, incomplete_pair)
    for previous, current in pairwise(trace):
        if (
            current.sequence_index != previous.sequence_index + 1
            or current.timestamp_s <= previous.timestamp_s
        ):
            raise ValueError("accepted observation trace must be contiguous")
    if trace and (
        current_sequence_index != trace[-1].sequence_index
        or current_timestamp_s != trace[-1].timestamp_s
    ):
        raise ValueError("current endpoint must equal the accepted trace tail")


def _validate_estimate_pending_state(
    *,
    identities: tuple[TwoPointIdentityEstimate, ...],
    completed_pairs: int,
    accepted_observations: int,
    incomplete_pair: TwoPointPartialPair | None,
    pending_query: TwoPointQuery | None,
    current_sequence_index: int | None,
    current_timestamp_s: float,
) -> None:
    target = identities[completed_pairs % len(identities)]
    if incomplete_pair is not None and (
        incomplete_pair.pair_index != completed_pairs
        or incomplete_pair.identity_pair_index != target.completed_pairs
        or incomplete_pair.resonance_id != target.resonance_id
        or incomplete_pair.interrogation_center_hz != target.center_hz
        or incomplete_pair.first_query.query_index != accepted_observations - 1
        or current_sequence_index != incomplete_pair.first_observation.sequence_index
        or current_timestamp_s != incomplete_pair.first_observation.timestamp_s
    ):
        raise ValueError(
            "incomplete_pair index, identity, and center must equal the first side"
        )
    if pending_query is not None:
        if (
            pending_query.query_index != accepted_observations
            or pending_query.pair_index != completed_pairs
            or pending_query.identity_pair_index != target.completed_pairs
            or pending_query.resonance_id != target.resonance_id
            or pending_query.interrogation_center_hz != target.center_hz
        ):
            raise ValueError(
                "pending_query index, identity, and center must equal the next query"
            )
        if incomplete_pair is not None and (
            pending_query.side == incomplete_pair.first_side
            or pending_query.interrogation_center_hz
            != incomplete_pair.interrogation_center_hz
        ):
            raise ValueError("pending second query must complete the partial pair")
        expected_sequence_index = (
            0 if current_sequence_index is None else current_sequence_index + 1
        )
        if (
            pending_query.expected_sequence_index != expected_sequence_index
            or pending_query.expected_end_timestamp_s <= current_timestamp_s
        ):
            raise ValueError("pending query must follow the current endpoint")


def _validate_tracking_resource_totals(
    resources: PublicAcquisitionResources,
    observations: tuple[EstimatorObservation, ...],
) -> None:
    integration_time_s = 0.0
    nominal_exposure_photons = 0.0
    realized_photons = 0
    missing = 0
    for observation in observations:
        integration_time_s = integration_time_s + observation.integration_time_s
        nominal_exposure_photons = (
            nominal_exposure_photons + observation.nominal_exposure_photons
        )
        realized_photons += observation.realized_photons or 0
        missing += int(observation.realized_photons is None)
    if (
        resources.observations != len(observations)
        or resources.integration_time_s != integration_time_s
        or resources.nominal_exposure_photons != nominal_exposure_photons
        or resources.realized_photons != realized_photons
        or resources.observations_without_realized_counts != missing
    ):
        raise ValueError("tracking resources must equal accepted observation totals")


def _validate_estimate_identity_ages(
    identities: tuple[TwoPointIdentityEstimate, ...],
    *,
    current_sequence_index: int | None,
    current_timestamp_s: float,
    calibration_budget_treatment: CalibrationBudgetTreatment,
) -> None:
    for identity in identities:
        if (
            identity.estimate_age_s
            != current_timestamp_s - identity.active_reference_timestamp_s
            or identity.release_age_s
            != current_timestamp_s - identity.active_release_timestamp_s
        ):
            raise ValueError("identity time ages must equal the current endpoint")
        if identity.active_release_sequence_index is None:
            if identity.estimate_age_sequence_indices is not None:
                raise ValueError(
                    "missing release sequence requires missing sequence age"
                )
        elif current_sequence_index is None or (
            identity.estimate_age_sequence_indices
            != current_sequence_index - identity.active_release_sequence_index
        ):
            raise ValueError("identity sequence age must equal the current endpoint")
        if identity.active_source_kind == "calibration":
            sequence_fields_present = identity.active_release_sequence_index is not None
            if sequence_fields_present != (
                calibration_budget_treatment == "included_same_run"
            ):
                raise ValueError(
                    "calibration sequence ages must follow budget treatment"
                )


@dataclass(frozen=True, slots=True)
class TwoPointIdentityEstimate:
    resonance_id: str
    center_hz: float
    calibration_fwhm_hz: float
    calibration_cell_lower_hz: float
    calibration_cell_upper_hz: float
    allowed_center_min_hz: float
    allowed_center_max_hz: float
    active_source_kind: Literal["calibration", "pair"]
    active_source_pair_index: int | None
    active_reference_timestamp_s: float
    active_release_sequence_index: int | None
    active_release_timestamp_s: float
    estimate_age_sequence_indices: int | None
    estimate_age_s: float
    release_age_s: float
    completed_pairs: int
    lock_state: TwoPointLockState
    failure_code: TwoPointFailureCode | None
    latest_pair: TwoPointPairResult | None

    def __post_init__(self) -> None:
        resonance_id = _required_nonblank_string(self.resonance_id, "resonance_id")
        active_source_kind = _closed_literal_string(
            self.active_source_kind,
            "active_source_kind",
            frozenset({"calibration", "pair"}),
        )
        lock_state = _closed_literal_string(
            self.lock_state,
            "lock_state",
            frozenset({"calibrated", "tracking", "step_limited", "lost"}),
        )
        failure_code = _optional_closed_literal_string(
            self.failure_code,
            "failure_code",
            frozenset(
                {
                    "invalid_pair_normalization",
                    "numerical_failure",
                    "common_mode_limit_exceeded",
                    "capture_exceeded",
                    "calibration_domain_exceeded",
                }
            ),
        )
        center_hz = _finite_float(self.center_hz, "center_hz")
        calibration_fwhm_hz = _positive_float(
            self.calibration_fwhm_hz, "calibration_fwhm_hz"
        )
        calibration_cell_lower_hz = _finite_float(
            self.calibration_cell_lower_hz, "calibration_cell_lower_hz"
        )
        calibration_cell_upper_hz = _finite_float(
            self.calibration_cell_upper_hz, "calibration_cell_upper_hz"
        )
        allowed_center_min_hz = _finite_float(
            self.allowed_center_min_hz, "allowed_center_min_hz"
        )
        allowed_center_max_hz = _finite_float(
            self.allowed_center_max_hz, "allowed_center_max_hz"
        )
        if not (
            calibration_cell_lower_hz
            <= allowed_center_min_hz
            <= allowed_center_max_hz
            <= calibration_cell_upper_hz
        ):
            raise ValueError("allowed center interval must lie in the calibration cell")
        if not allowed_center_min_hz <= center_hz <= allowed_center_max_hz:
            raise ValueError("center_hz must lie in the allowed center interval")
        active_source_pair_index = (
            None
            if self.active_source_pair_index is None
            else _nonnegative_int(
                self.active_source_pair_index, "active_source_pair_index"
            )
        )
        active_reference_timestamp_s = _finite_float(
            self.active_reference_timestamp_s, "active_reference_timestamp_s"
        )
        active_release_sequence_index = (
            None
            if self.active_release_sequence_index is None
            else _nonnegative_int(
                self.active_release_sequence_index,
                "active_release_sequence_index",
            )
        )
        active_release_timestamp_s = _nonnegative_float(
            self.active_release_timestamp_s, "active_release_timestamp_s"
        )
        estimate_age_sequence_indices = (
            None
            if self.estimate_age_sequence_indices is None
            else _nonnegative_int(
                self.estimate_age_sequence_indices,
                "estimate_age_sequence_indices",
            )
        )
        estimate_age_s = _nonnegative_float(self.estimate_age_s, "estimate_age_s")
        release_age_s = _nonnegative_float(self.release_age_s, "release_age_s")
        completed_pairs = _nonnegative_int(self.completed_pairs, "completed_pairs")
        if active_source_kind == "calibration":
            if active_source_pair_index is not None:
                raise ValueError("calibration source cannot name a pair")
        else:
            if active_source_pair_index is None:
                raise ValueError("pair source requires active_source_pair_index")
            if active_reference_timestamp_s < 0.0:
                raise ValueError("pair active reference must be non-negative")
            if (
                active_release_sequence_index is None
                or estimate_age_sequence_indices is None
            ):
                raise ValueError("pair source requires release and age sequence fields")
        if (active_release_sequence_index is None) != (
            estimate_age_sequence_indices is None
        ):
            raise ValueError("release sequence and sequence age must both be present")
        if lock_state == "lost":
            if failure_code is None:
                raise ValueError("lost identity requires a failure_code")
        elif failure_code is not None:
            raise ValueError("non-lost identity cannot carry a failure_code")
        if self.latest_pair is None:
            if completed_pairs != 0:
                raise ValueError("completed_pairs requires latest_pair")
            if (
                active_source_kind != "calibration"
                or lock_state != "calibrated"
            ):
                raise ValueError("an unpaired identity must remain calibration-seeded")
        else:
            if type(self.latest_pair) is not TwoPointPairResult:
                raise TypeError("latest_pair must be an exact TwoPointPairResult")
            if (
                completed_pairs == 0
                or self.latest_pair.resonance_id != resonance_id
                or self.latest_pair.identity_pair_index != completed_pairs - 1
                or self.latest_pair.lock_state != lock_state
                or self.latest_pair.failure_code != failure_code
            ):
                raise ValueError("latest_pair must match identity history state")
            if self.latest_pair.lock_state in {"tracking", "step_limited"}:
                if (
                    active_source_kind != "pair"
                    or active_source_pair_index != self.latest_pair.pair_index
                    or active_reference_timestamp_s
                    != self.latest_pair.pair_reference_timestamp_s
                    or active_release_sequence_index
                    != self.latest_pair.release_sequence_index
                    or active_release_timestamp_s
                    != self.latest_pair.release_timestamp_s
                    or center_hz != self.latest_pair.candidate_center_hz
                ):
                    raise ValueError("successful latest_pair must be the active source")
            elif active_source_pair_index is not None and (
                active_source_pair_index >= self.latest_pair.pair_index
            ):
                raise ValueError("a lost pair cannot become the active source")
        object.__setattr__(self, "resonance_id", resonance_id)
        object.__setattr__(self, "active_source_kind", active_source_kind)
        object.__setattr__(self, "center_hz", center_hz)
        object.__setattr__(self, "calibration_fwhm_hz", calibration_fwhm_hz)
        object.__setattr__(
            self, "calibration_cell_lower_hz", calibration_cell_lower_hz
        )
        object.__setattr__(
            self, "calibration_cell_upper_hz", calibration_cell_upper_hz
        )
        object.__setattr__(self, "allowed_center_min_hz", allowed_center_min_hz)
        object.__setattr__(self, "allowed_center_max_hz", allowed_center_max_hz)
        object.__setattr__(self, "active_source_pair_index", active_source_pair_index)
        object.__setattr__(
            self, "active_reference_timestamp_s", active_reference_timestamp_s
        )
        object.__setattr__(
            self, "active_release_sequence_index", active_release_sequence_index
        )
        object.__setattr__(
            self, "active_release_timestamp_s", active_release_timestamp_s
        )
        object.__setattr__(
            self, "estimate_age_sequence_indices", estimate_age_sequence_indices
        )
        object.__setattr__(self, "estimate_age_s", estimate_age_s)
        object.__setattr__(self, "release_age_s", release_age_s)
        object.__setattr__(self, "completed_pairs", completed_pairs)
        object.__setattr__(self, "lock_state", lock_state)
        object.__setattr__(self, "failure_code", failure_code)


@dataclass(frozen=True, slots=True)
class TwoPointEstimate:
    identities: tuple[TwoPointIdentityEstimate, ...]
    calibration_source_id: str
    calibration_source_provenance: CalibrationSourceProvenance
    calibration_budget_treatment: CalibrationBudgetTreatment
    current_sequence_index: int | None
    current_timestamp_s: float
    accepted_observations: int
    completed_pairs: int
    incomplete_pair: TwoPointPartialPair | None
    pending_query: TwoPointQuery | None
    pair_history: tuple[TwoPointPairResult, ...]
    tracking_resources: PublicAcquisitionResources
    calibration_resources: PublicAcquisitionResources
    charged_resources: PublicAcquisitionResources
    budget_ceiling: TwoPointBudgetCeiling
    stopped_reason: TwoPointStopReason | None
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.identities, (tuple, list)):
            raise TypeError("identities must be an ordered sequence")
        identities = tuple(self.identities)
        if len(identities) != 8:
            raise ValueError("identities must contain eight identity estimates")
        if not all(
            type(identity) is TwoPointIdentityEstimate for identity in identities
        ):
            raise TypeError(
                "identities must contain exact TwoPointIdentityEstimate values"
            )
        identity_ids = tuple(identity.resonance_id for identity in identities)
        if len(set(identity_ids)) != 8:
            raise ValueError("identity estimates must have unique resonance IDs")
        calibration_source_id = _required_nonblank_string(
            self.calibration_source_id, "calibration_source_id"
        )
        calibration_source_provenance = _closed_literal_string(
            self.calibration_source_provenance,
            "calibration_source_provenance",
            frozenset({"verified_factory_acquisition", "caller_asserted"}),
        )
        calibration_budget_treatment = _closed_literal_string(
            self.calibration_budget_treatment,
            "calibration_budget_treatment",
            frozenset({"included_same_run", "conditional_free_precalibration"}),
        )
        stopped_reason = _optional_closed_literal_string(
            self.stopped_reason,
            "stopped_reason",
            frozenset({"budget_exhausted"}),
        )
        if (
            calibration_budget_treatment == "included_same_run"
            and calibration_source_provenance != "verified_factory_acquisition"
        ):
            raise ValueError("included_same_run requires verified source provenance")
        current_sequence_index = (
            None
            if self.current_sequence_index is None
            else _nonnegative_int(self.current_sequence_index, "current_sequence_index")
        )
        current_timestamp_s = _nonnegative_float(
            self.current_timestamp_s, "current_timestamp_s"
        )
        accepted_observations = _nonnegative_int(
            self.accepted_observations, "accepted_observations"
        )
        completed_pairs = _nonnegative_int(self.completed_pairs, "completed_pairs")
        if self.incomplete_pair is not None and type(
            self.incomplete_pair
        ) is not TwoPointPartialPair:
            raise TypeError("incomplete_pair must be an exact TwoPointPartialPair")
        if self.pending_query is not None and type(
            self.pending_query
        ) is not TwoPointQuery:
            raise TypeError("pending_query must be an exact TwoPointQuery")
        if not isinstance(self.pair_history, (tuple, list)):
            raise TypeError("pair_history must be an ordered sequence")
        pair_history = tuple(self.pair_history)
        if not all(type(pair) is TwoPointPairResult for pair in pair_history):
            raise TypeError("pair_history must contain exact TwoPointPairResult values")
        if completed_pairs != len(pair_history):
            raise ValueError("completed_pairs must equal pair_history length")
        if accepted_observations != 2 * completed_pairs + int(
            self.incomplete_pair is not None
        ):
            raise ValueError("accepted_observations must equal pair and partial counts")
        _validate_estimate_history(identities, pair_history)
        _validate_estimate_pending_state(
            identities=identities,
            completed_pairs=completed_pairs,
            accepted_observations=accepted_observations,
            incomplete_pair=self.incomplete_pair,
            pending_query=self.pending_query,
            current_sequence_index=current_sequence_index,
            current_timestamp_s=current_timestamp_s,
        )
        _validate_accepted_trace_endpoint(
            pair_history,
            self.incomplete_pair,
            current_sequence_index=current_sequence_index,
            current_timestamp_s=current_timestamp_s,
        )
        if type(self.tracking_resources) is not PublicAcquisitionResources:
            raise TypeError(
                "tracking_resources must be exact PublicAcquisitionResources"
            )
        if type(self.calibration_resources) is not PublicAcquisitionResources:
            raise TypeError(
                "calibration_resources must be exact PublicAcquisitionResources"
            )
        if type(self.charged_resources) is not PublicAcquisitionResources:
            raise TypeError(
                "charged_resources must be exact PublicAcquisitionResources"
            )
        accepted_trace = _estimate_observation_trace(
            pair_history, self.incomplete_pair
        )
        _validate_tracking_resource_totals(self.tracking_resources, accepted_trace)
        if calibration_budget_treatment == "conditional_free_precalibration":
            if self.charged_resources != self.tracking_resources:
                raise ValueError("conditional charged resources must equal tracking")
        else:
            if (
                self.charged_resources.observations
                != self.calibration_resources.observations
                + self.tracking_resources.observations
                or self.charged_resources.realized_photons
                != self.calibration_resources.realized_photons
                + self.tracking_resources.realized_photons
                or self.charged_resources.observations_without_realized_counts
                != self.calibration_resources.observations_without_realized_counts
                + self.tracking_resources.observations_without_realized_counts
            ):
                raise ValueError("included charged resource counts must include source")
        if type(self.budget_ceiling) is not TwoPointBudgetCeiling:
            raise TypeError("budget_ceiling must be an exact TwoPointBudgetCeiling")
        if stopped_reason is not None and (
            self.pending_query is not None or self.incomplete_pair is not None
        ):
            raise ValueError("stopped estimate must be at a pair boundary")
        seed = _nonnegative_int(self.seed, "seed")
        _validate_estimate_identity_ages(
            identities,
            current_sequence_index=current_sequence_index,
            current_timestamp_s=current_timestamp_s,
            calibration_budget_treatment=calibration_budget_treatment,
        )
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "calibration_source_id", calibration_source_id)
        object.__setattr__(
            self, "calibration_source_provenance", calibration_source_provenance
        )
        object.__setattr__(
            self, "calibration_budget_treatment", calibration_budget_treatment
        )
        object.__setattr__(self, "current_sequence_index", current_sequence_index)
        object.__setattr__(self, "current_timestamp_s", current_timestamp_s)
        object.__setattr__(self, "accepted_observations", accepted_observations)
        object.__setattr__(self, "completed_pairs", completed_pairs)
        object.__setattr__(self, "pair_history", pair_history)
        object.__setattr__(self, "stopped_reason", stopped_reason)
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True, slots=True)
class TwoPointUpdate:
    query: TwoPointQuery
    observation: EstimatorObservation
    completed_pair: TwoPointPairResult | None
    estimate: TwoPointEstimate

    def __post_init__(self) -> None:
        if type(self.query) is not TwoPointQuery:
            raise TypeError("query must be an exact TwoPointQuery")
        _validate_query_observation_echo(self.query, self.observation, "observation")
        if type(self.estimate) is not TwoPointEstimate:
            raise TypeError("estimate must be an exact TwoPointEstimate")
        if self.estimate.pending_query is not None:
            raise ValueError("returned estimate must clear the accepted pending query")
        if self.completed_pair is None:
            partial = self.estimate.incomplete_pair
            if partial is None or (
                partial.first_query != self.query
                or partial.first_observation != self.observation
            ):
                raise ValueError("first-side update must return its exact partial pair")
            return
        if type(self.completed_pair) is not TwoPointPairResult:
            raise TypeError("completed_pair must be an exact TwoPointPairResult")
        if self.estimate.incomplete_pair is not None or not self.estimate.pair_history:
            raise ValueError("second-side update must clear the partial pair")
        if self.estimate.pair_history[-1] != self.completed_pair:
            raise ValueError("completed_pair must equal the returned history tail")
        second_query, second_observation = (
            (self.completed_pair.plus_query, self.completed_pair.plus_observation)
            if self.completed_pair.first_side == "minus"
            else (
                self.completed_pair.minus_query,
                self.completed_pair.minus_observation,
            )
        )
        if self.query != second_query or self.observation != second_observation:
            raise ValueError("second-side update must echo the completed pair")
