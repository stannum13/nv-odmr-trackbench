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
        expected_resources = PublicAcquisitionResources(
            observations=len(source_observations),
            integration_time_s=sum(
                item.integration_time_s for item in source_observations
            ),
            nominal_exposure_photons=sum(
                item.nominal_exposure_photons for item in source_observations
            ),
            realized_photons=sum(
                item.realized_photons
                for item in source_observations
                if item.realized_photons is not None
            ),
            observations_without_realized_counts=sum(
                item.realized_photons is None for item in source_observations
            ),
            virtual_elapsed_time_s=last_observation.timestamp_s
            - source_start_timestamp_s,
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
