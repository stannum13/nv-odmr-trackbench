"""Immutable validated public contracts for offline spectrum estimation."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from odmr_bench.models import Baseline, Resonance, q_factor


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    try:
        canonical = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite") from None
    if not np.isfinite(canonical):
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


def _int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _optional_int(value: object, name: str) -> int | None:
    return None if value is None else _int(value, name)


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return value


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return value


def _optional_nonnegative_float(value: object, name: str) -> float | None:
    return None if value is None else _nonnegative_float(value, name)


def _optional_positive_float(value: object, name: str) -> float | None:
    return None if value is None else _positive_float(value, name)


def _copied_float64_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Copy a numeric array before its caller applies contract-specific checks."""
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw) or (
            raw.dtype == object and any(np.iscomplexobj(item) for item in raw.ravel())
        ):
            raise TypeError(f"{name} must not contain complex values")
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.ComplexWarning)
            canonical = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, np.exceptions.ComplexWarning) as exc:
        raise TypeError(f"{name} must be a numeric array") from exc
    return canonical


def _immutable_float_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Copy a one-dimensional finite numeric array into a frozen float64 array."""
    canonical = _copied_float64_array(value, name)
    if canonical.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(canonical)):
        raise ValueError(f"{name} must be finite")
    canonical.setflags(write=False)
    return canonical


def _immutable_float_array_with_length(
    value: ArrayLike, name: str, length: int | tuple[int, ...]
) -> NDArray[np.float64]:
    canonical = _immutable_float_array(value, name)
    lengths = (length,) if isinstance(length, int) else length
    if canonical.size not in lengths:
        rendered_lengths = ", ".join(str(item) for item in lengths)
        raise ValueError(f"{name} must have length {rendered_lengths}")
    if np.any(canonical < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return canonical


def _canonical_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{name} must be a sequence of strings")
    try:
        canonical = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of strings") from exc
    for item in canonical:
        if not isinstance(item, str):
            raise TypeError(f"{name} must contain only strings")
    return canonical


def _snapshot_baseline(value: object, name: str) -> Baseline:
    if not isinstance(value, Baseline):
        raise TypeError(f"{name} must be a Baseline")
    return Baseline(
        intercept=value.intercept,
        reference_hz=value.reference_hz,
        slope_per_hz=value.slope_per_hz,
        quadratic_per_hz2=value.quadratic_per_hz2,
    )


def _snapshot_ordered_resonances(value: object, name: str) -> tuple[Resonance, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of Resonance values")
    try:
        supplied = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of Resonance values") from exc
    if len(supplied) != 8:
        raise ValueError(f"{name} must contain exactly eight resonances")
    if not all(isinstance(item, Resonance) for item in supplied):
        raise TypeError(f"{name} must contain only Resonance values")
    canonical = tuple(
        Resonance(
            resonance_id=item.resonance_id,
            center_hz=item.center_hz,
            fwhm_hz=item.fwhm_hz,
            amplitude=item.amplitude,
            eta=item.eta,
        )
        for item in supplied
    )
    if len({item.resonance_id for item in canonical}) != 8:
        raise ValueError(f"{name} must have unique resonance identities")
    if any(right.center_hz <= left.center_hz for left, right in pairwise(canonical)):
        raise ValueError(f"{name} centers must be strictly ordered")
    return canonical


@dataclass(frozen=True, slots=True)
class CompleteSweep:
    """A completed frequency sweep and its optional public completion totals."""

    frequency_hz: ArrayLike
    fluorescence: ArrayLike
    last_sequence_index: int | None = None
    last_timestamp_s: float | None = None
    total_integration_time_s: float | None = None
    total_nominal_exposure_photons: float | None = None

    def __post_init__(self) -> None:
        frequency_hz = _copied_float64_array(self.frequency_hz, "frequency_hz")
        fluorescence = _copied_float64_array(self.fluorescence, "fluorescence")
        if frequency_hz.ndim != 1 or fluorescence.ndim != 1:
            raise ValueError("frequency_hz and fluorescence must be one-dimensional")
        if frequency_hz.size < 2:
            raise ValueError("a complete sweep must contain at least two samples")
        if frequency_hz.shape != fluorescence.shape:
            raise ValueError("frequency_hz and fluorescence must have matching shapes")
        if not np.all(np.isfinite(frequency_hz)) or not np.all(
            np.isfinite(fluorescence)
        ):
            raise ValueError("frequency_hz and fluorescence must be finite")
        if np.any(np.diff(frequency_hz) <= 0.0):
            raise ValueError("frequency_hz must be strictly increasing")
        frequency_hz.setflags(write=False)
        fluorescence.setflags(write=False)

        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "fluorescence", fluorescence)
        if self.last_sequence_index is not None:
            object.__setattr__(
                self,
                "last_sequence_index",
                _nonnegative_int(self.last_sequence_index, "last_sequence_index"),
            )
        if self.last_timestamp_s is not None:
            object.__setattr__(
                self,
                "last_timestamp_s",
                _nonnegative_float(self.last_timestamp_s, "last_timestamp_s"),
            )
        if self.total_integration_time_s is not None:
            object.__setattr__(
                self,
                "total_integration_time_s",
                _positive_float(
                    self.total_integration_time_s, "total_integration_time_s"
                ),
            )
        if self.total_nominal_exposure_photons is not None:
            object.__setattr__(
                self,
                "total_nominal_exposure_photons",
                _nonnegative_float(
                    self.total_nominal_exposure_photons,
                    "total_nominal_exposure_photons",
                ),
            )


@dataclass(frozen=True, slots=True)
class FitConfiguration:
    """Validated fixed configuration for the constrained eight-line fitter."""

    model_kind: str = "pseudo_voigt"
    baseline_degree: int = 1
    resonance_ids: Sequence[str] = tuple(f"r{i}" for i in range(8))
    min_fwhm_hz: float = 2.0e5
    max_fwhm_hz: float = 8.0e6
    max_amplitude: float = 0.25
    min_resolved_amplitude: float = 1.0e-4
    min_center_separation_hz: float = 1.0e6
    savgol_window: int = 11
    savgol_polyorder: int = 2
    relative_prominence: float = 0.01
    allow_fallback: bool = False
    max_nfev: int = 4000
    rank_rtol: float = 1.0e-10
    min_baseline_sse_improvement: float = 1.0e-4
    min_amplitude_significance: float = 5.0

    def __post_init__(self) -> None:
        if self.model_kind not in {"lorentzian", "pseudo_voigt"}:
            raise ValueError("model_kind must be 'lorentzian' or 'pseudo_voigt'")
        baseline_degree = _int(self.baseline_degree, "baseline_degree")
        if baseline_degree not in {1, 2}:
            raise ValueError("baseline_degree must be 1 or 2")
        resonance_ids = _canonical_string_tuple(self.resonance_ids, "resonance_ids")
        if len(resonance_ids) != 8:
            raise ValueError("resonance_ids must contain exactly eight IDs")
        if any(not resonance_id.strip() for resonance_id in resonance_ids):
            raise ValueError("resonance_ids must be nonempty")
        if len(set(resonance_ids)) != 8:
            raise ValueError("resonance_ids must be unique")

        min_fwhm_hz = _positive_float(self.min_fwhm_hz, "min_fwhm_hz")
        max_fwhm_hz = _positive_float(self.max_fwhm_hz, "max_fwhm_hz")
        if max_fwhm_hz <= min_fwhm_hz:
            raise ValueError("max_fwhm_hz must be greater than min_fwhm_hz")
        max_amplitude = _positive_float(self.max_amplitude, "max_amplitude")
        min_resolved_amplitude = _positive_float(
            self.min_resolved_amplitude, "min_resolved_amplitude"
        )
        if min_resolved_amplitude > max_amplitude:
            raise ValueError("min_resolved_amplitude must not exceed max_amplitude")
        min_center_separation_hz = _positive_float(
            self.min_center_separation_hz, "min_center_separation_hz"
        )
        savgol_window = _int(self.savgol_window, "savgol_window")
        if savgol_window < 5 or savgol_window % 2 == 0:
            raise ValueError("savgol_window must be odd and at least five")
        savgol_polyorder = _int(self.savgol_polyorder, "savgol_polyorder")
        if not 0 <= savgol_polyorder < savgol_window:
            raise ValueError("savgol_polyorder must be non-negative and below window")
        relative_prominence = _finite_float(
            self.relative_prominence, "relative_prominence"
        )
        if not 0.0 < relative_prominence <= 1.0:
            raise ValueError("relative_prominence must be within (0, 1]")
        allow_fallback = _bool(self.allow_fallback, "allow_fallback")
        max_nfev = _nonnegative_int(self.max_nfev, "max_nfev")
        if max_nfev == 0:
            raise ValueError("max_nfev must be positive")
        rank_rtol = _finite_float(self.rank_rtol, "rank_rtol")
        if not 0.0 < rank_rtol < 1.0:
            raise ValueError("rank_rtol must be within (0, 1)")
        min_baseline_sse_improvement = _finite_float(
            self.min_baseline_sse_improvement, "min_baseline_sse_improvement"
        )
        if not 0.0 <= min_baseline_sse_improvement < 1.0:
            raise ValueError("min_baseline_sse_improvement must be within [0, 1)")
        min_amplitude_significance = _positive_float(
            self.min_amplitude_significance, "min_amplitude_significance"
        )

        object.__setattr__(self, "baseline_degree", baseline_degree)
        object.__setattr__(self, "resonance_ids", resonance_ids)
        object.__setattr__(self, "min_fwhm_hz", min_fwhm_hz)
        object.__setattr__(self, "max_fwhm_hz", max_fwhm_hz)
        object.__setattr__(self, "max_amplitude", max_amplitude)
        object.__setattr__(self, "min_resolved_amplitude", min_resolved_amplitude)
        object.__setattr__(self, "min_center_separation_hz", min_center_separation_hz)
        object.__setattr__(self, "savgol_window", savgol_window)
        object.__setattr__(self, "savgol_polyorder", savgol_polyorder)
        object.__setattr__(self, "relative_prominence", relative_prominence)
        object.__setattr__(self, "allow_fallback", allow_fallback)
        object.__setattr__(self, "max_nfev", max_nfev)
        object.__setattr__(self, "rank_rtol", rank_rtol)
        object.__setattr__(
            self,
            "min_baseline_sse_improvement",
            min_baseline_sse_improvement,
        )
        object.__setattr__(
            self,
            "min_amplitude_significance",
            min_amplitude_significance,
        )


@dataclass(frozen=True, slots=True)
class FitInitialGuess:
    """An immutable explicitly attempted eight-line starting point."""

    resonances: Sequence[Resonance]
    baseline: Baseline

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resonances",
            _snapshot_ordered_resonances(self.resonances, "resonances"),
        )
        object.__setattr__(
            self, "baseline", _snapshot_baseline(self.baseline, "baseline")
        )


@dataclass(frozen=True, slots=True)
class InitializationDiagnostics:
    """Deterministic initializer provenance, including an explicit fallback state."""

    source: str
    candidate_count: int
    selected_indices: Sequence[int]
    used_fallback: bool
    messages: Sequence[str]

    def __post_init__(self) -> None:
        if self.source not in {"detected", "fallback", "user", "none"}:
            raise ValueError("source must be detected, fallback, user, or none")
        candidate_count = _nonnegative_int(self.candidate_count, "candidate_count")
        if isinstance(self.selected_indices, (str, bytes)):
            raise TypeError("selected_indices must be a sequence of integers")
        try:
            selected_indices = tuple(
                _nonnegative_int(item, "selected_indices item")
                for item in self.selected_indices
            )
        except TypeError as exc:
            raise TypeError("selected_indices must be a sequence of integers") from exc
        if isinstance(self.messages, (str, bytes)):
            raise TypeError("messages must be a sequence of strings")
        messages = _canonical_string_tuple(self.messages, "messages")
        used_fallback = _bool(self.used_fallback, "used_fallback")

        if self.source == "detected":
            if candidate_count < 8:
                raise ValueError(
                    "detected diagnostics require at least eight candidates"
                )
            if len(selected_indices) != 8 or len(set(selected_indices)) != 8:
                raise ValueError("detected diagnostics require eight distinct indices")
            if used_fallback:
                raise ValueError("detected diagnostics cannot use fallback")
        elif self.source == "fallback":
            if selected_indices or not used_fallback:
                raise ValueError("fallback diagnostics require no indices and fallback")
        elif self.source == "user":
            if candidate_count != 0 or selected_indices or used_fallback:
                raise ValueError(f"{self.source} diagnostics have incompatible fields")
        elif selected_indices or used_fallback:
            raise ValueError("none diagnostics require no indices and no fallback")

        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "selected_indices", selected_indices)
        object.__setattr__(self, "used_fallback", used_fallback)
        object.__setattr__(self, "messages", messages)


@dataclass(frozen=True, slots=True)
class FitUncertainty:
    """Local-linearized standard errors in the public, physical units."""

    baseline_standard_errors: ArrayLike
    center_hz: ArrayLike
    fwhm_hz: ArrayLike
    amplitude: ArrayLike
    eta: ArrayLike | None
    method: str = "local_linearized_jacobian_covariance"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_standard_errors",
            _immutable_float_array_with_length(
                self.baseline_standard_errors, "baseline_standard_errors", (2, 3)
            ),
        )
        for attribute in ("center_hz", "fwhm_hz", "amplitude"):
            object.__setattr__(
                self,
                attribute,
                _immutable_float_array_with_length(
                    getattr(self, attribute), attribute, 8
                ),
            )
        if self.eta is not None:
            object.__setattr__(
                self,
                "eta",
                _immutable_float_array_with_length(self.eta, "eta", 8),
            )
        method = _required_string(self.method, "method")
        if method != "local_linearized_jacobian_covariance":
            raise ValueError(
                "method must equal 'local_linearized_jacobian_covariance'"
            )
        object.__setattr__(self, "method", method)


_FAILURE_CODES = frozenset(
    {
        "initialization_failed",
        "insufficient_samples",
        "uninformative_sweep",
        "optimization_failed",
        "quality_failed",
    }
)


def _free_parameter_count(model_kind: str, baseline_degree: int) -> int:
    return baseline_degree + 1 + 8 * (4 if model_kind == "pseudo_voigt" else 3)


def _copy_initial_guess(value: object) -> FitInitialGuess | None:
    if value is None:
        return None
    if not isinstance(value, FitInitialGuess):
        raise TypeError("initial_guess must be a FitInitialGuess or None")
    return FitInitialGuess(resonances=value.resonances, baseline=value.baseline)


@dataclass(frozen=True, slots=True)
class SpectrumFitResult:
    """A complete constrained-fit outcome, including structured failures."""

    success: bool
    failure_code: str | None
    model_kind: str
    baseline_degree: int
    resonance_estimates: Sequence[Resonance]
    baseline_estimate: Baseline | None
    q_values: NDArray[np.float64] = field(init=False, repr=False)
    diagnostics: InitializationDiagnostics
    initial_guess: FitInitialGuess | None
    uncertainty: FitUncertainty | None
    uncertainty_reason: str | None
    scipy_status: int | None
    scipy_message: str | None
    nfev: int
    cost: float | None
    residual_rmse: float | None
    residual_scale: float | None
    degrees_of_freedom: int
    jacobian_rank: int | None

    def __post_init__(self) -> None:
        success = _bool(self.success, "success")
        if self.model_kind not in {"lorentzian", "pseudo_voigt"}:
            raise ValueError("model_kind must be 'lorentzian' or 'pseudo_voigt'")
        baseline_degree = _int(self.baseline_degree, "baseline_degree")
        if baseline_degree not in {1, 2}:
            raise ValueError("baseline_degree must be 1 or 2")
        if not isinstance(self.diagnostics, InitializationDiagnostics):
            raise TypeError("diagnostics must be InitializationDiagnostics")

        failure_code = _optional_string(self.failure_code, "failure_code")
        if success:
            if failure_code is not None:
                raise ValueError("successful results must not have a failure_code")
        elif failure_code not in _FAILURE_CODES:
            raise ValueError("unsuccessful results require a stable failure_code")

        if isinstance(self.resonance_estimates, (str, bytes)):
            raise TypeError("resonance_estimates must be a sequence")
        try:
            supplied_resonances = tuple(self.resonance_estimates)
        except TypeError as exc:
            raise TypeError("resonance_estimates must be a sequence") from exc
        if success:
            resonance_estimates = _snapshot_ordered_resonances(
                supplied_resonances, "resonance_estimates"
            )
            if self.model_kind == "lorentzian" and any(
                resonance.eta != 1.0 for resonance in resonance_estimates
            ):
                raise ValueError("lorentzian results require eta equal to one")
            baseline_estimate = _snapshot_baseline(
                self.baseline_estimate, "baseline_estimate"
            )
        else:
            if supplied_resonances:
                raise ValueError("failed results must not contain resonance estimates")
            if self.baseline_estimate is not None:
                raise ValueError("failed results must not contain a baseline estimate")
            resonance_estimates = ()
            baseline_estimate = None

        initial_guess = _copy_initial_guess(self.initial_guess)
        if (
            self.model_kind == "lorentzian"
            and initial_guess is not None
            and any(resonance.eta != 1.0 for resonance in initial_guess.resonances)
        ):
            raise ValueError("lorentzian guesses require eta equal to one")
        if not success and failure_code in {
            "initialization_failed",
            "insufficient_samples",
            "uninformative_sweep",
        }:
            if initial_guess is not None:
                raise ValueError(
                    "pre-optimization failures cannot retain an initial guess"
                )
        elif initial_guess is None:
            raise ValueError("an optimizer attempt requires an initial_guess snapshot")

        optimizer_attempted = success or failure_code in {
            "optimization_failed",
            "quality_failed",
        }
        if optimizer_attempted and self.diagnostics.source == "none":
            raise ValueError("optimizer attempts require a diagnostic source")
        if failure_code in {"insufficient_samples", "uninformative_sweep"} and (
            self.diagnostics.source != "none"
        ):
            raise ValueError(f"{failure_code} requires diagnostic source 'none'")
        if (
            failure_code == "initialization_failed"
            and self.diagnostics.source == "user"
        ):
            raise ValueError(
                "initialization_failed cannot have diagnostic source 'user'"
            )

        if (
            baseline_degree == 1
            and initial_guess is not None
            and initial_guess.baseline.quadratic_per_hz2 != 0.0
        ):
            raise ValueError("linear-baseline initial guesses require zero quadratic")
        if success:
            assert initial_guess is not None
            assert baseline_estimate is not None
            if baseline_estimate.quadratic_per_hz2 != 0.0 and baseline_degree == 1:
                raise ValueError("linear-baseline estimates require zero quadratic")
            fitted_ids = tuple(item.resonance_id for item in resonance_estimates)
            initial_ids = tuple(item.resonance_id for item in initial_guess.resonances)
            if fitted_ids != initial_ids:
                raise ValueError(
                    "successful fitted resonance IDs and order must match the "
                    "initial guess"
                )
            if baseline_estimate.reference_hz != initial_guess.baseline.reference_hz:
                raise ValueError(
                    "successful final and initial baseline references must match"
                )

        if self.uncertainty is not None and not isinstance(
            self.uncertainty, FitUncertainty
        ):
            raise TypeError("uncertainty must be a FitUncertainty or None")
        uncertainty_reason = _optional_string(
            self.uncertainty_reason, "uncertainty_reason"
        )
        if success:
            if self.uncertainty is None:
                if uncertainty_reason is None or not uncertainty_reason:
                    raise ValueError("missing uncertainty requires a reason")
            elif uncertainty_reason is not None:
                raise ValueError("available uncertainty cannot have a reason")
            if self.uncertainty is not None:
                if (
                    len(self.uncertainty.baseline_standard_errors)
                    != baseline_degree + 1
                ):
                    raise ValueError(
                        "uncertainty baseline shape mismatches baseline_degree"
                    )
                if (self.model_kind == "lorentzian") != (self.uncertainty.eta is None):
                    raise ValueError("uncertainty eta shape mismatches model_kind")
        else:
            if self.uncertainty is not None:
                raise ValueError("failed results cannot claim uncertainty")
            if uncertainty_reason is None or not uncertainty_reason:
                raise ValueError("failed results require an uncertainty_reason")

        scipy_status = _optional_int(self.scipy_status, "scipy_status")
        scipy_message = _optional_string(self.scipy_message, "scipy_message")
        nfev = _nonnegative_int(self.nfev, "nfev")
        cost = _optional_nonnegative_float(self.cost, "cost")
        residual_rmse = _optional_nonnegative_float(self.residual_rmse, "residual_rmse")
        residual_scale = _optional_positive_float(self.residual_scale, "residual_scale")
        degrees_of_freedom = _int(self.degrees_of_freedom, "degrees_of_freedom")
        jacobian_rank = _optional_int(self.jacobian_rank, "jacobian_rank")
        if jacobian_rank is not None and jacobian_rank < 0:
            raise ValueError("jacobian_rank must be non-negative")
        free_parameters = _free_parameter_count(self.model_kind, baseline_degree)
        if jacobian_rank is not None and jacobian_rank > free_parameters:
            raise ValueError("jacobian_rank cannot exceed the free parameter count")

        if optimizer_attempted:
            if (
                scipy_status is None
                or scipy_message is None
                or not scipy_message
                or nfev <= 0
            ):
                raise ValueError(
                    "optimizer attempts require non-None scipy_status, nonempty "
                    "scipy_message, and positive nfev"
                )
            if failure_code == "optimization_failed":
                if scipy_status > 0:
                    raise ValueError(
                        "optimization_failed requires non-positive scipy_status"
                    )
            elif scipy_status <= 0:
                raise ValueError(
                    "success and quality_failed require positive scipy_status"
                )
        elif scipy_status is not None or scipy_message is not None or nfev != 0:
            raise ValueError("pre-optimization failures cannot have SciPy fields")

        if not success:
            assert failure_code is not None
            if failure_code == "insufficient_samples":
                if degrees_of_freedom > 0:
                    raise ValueError(
                        "insufficient_samples requires non-positive degrees_of_freedom"
                    )
            elif degrees_of_freedom <= 0:
                raise ValueError(
                    "non-scarcity failures require positive degrees_of_freedom"
                )
            if failure_code == "initialization_failed":
                if (
                    residual_scale is None
                    or cost is not None
                    or residual_rmse is not None
                ):
                    raise ValueError(
                        "initialization_failed has incompatible residual fields"
                    )
                if jacobian_rank is not None:
                    raise ValueError(
                        "initialization_failed cannot have a Jacobian rank"
                    )
            elif failure_code in {"insufficient_samples", "uninformative_sweep"}:
                if any(
                    value is not None for value in (residual_scale, cost, residual_rmse)
                ):
                    raise ValueError("preflight failures cannot have residual fields")
                if jacobian_rank is not None:
                    raise ValueError("preflight failures cannot have a Jacobian rank")
            elif failure_code == "optimization_failed":
                if residual_scale is None or jacobian_rank is not None:
                    raise ValueError("optimization_failed has incompatible fields")
                if (cost is None) != (residual_rmse is None):
                    raise ValueError("cost and residual_rmse must be jointly available")
            else:
                if residual_scale is None:
                    raise ValueError("quality_failed requires a residual scale")
                if (cost is None) != (residual_rmse is None):
                    raise ValueError("cost and residual_rmse must be jointly available")
                if cost is None and uncertainty_reason != (
                    "optimizer returned non-finite parameters, residuals, or cost"
                ):
                    raise ValueError(
                        "quality_failed may omit residual metrics only for "
                        "non-finite optimizer output"
                    )
                if cost is not None and uncertainty_reason == (
                    "optimizer returned non-finite parameters, residuals, or cost"
                ):
                    raise ValueError(
                        "non-finite optimizer reason must omit residual metrics"
                    )
        else:
            if residual_scale is None or cost is None or residual_rmse is None:
                raise ValueError("successful results require residual fields")
            if degrees_of_freedom <= 0:
                raise ValueError(
                    "successful results require positive degrees_of_freedom"
                )
            if jacobian_rank != free_parameters:
                raise ValueError("successful results require a full-rank Jacobian")

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            q_values = np.asarray(
                q_factor(
                    np.array([item.center_hz for item in resonance_estimates]),
                    np.array([item.fwhm_hz for item in resonance_estimates]),
                ),
                dtype=np.float64,
            ).copy()
        if success and not np.all(np.isfinite(q_values)):
            raise ValueError("successful results require finite representable Q values")
        q_values.setflags(write=False)

        object.__setattr__(self, "success", success)
        object.__setattr__(self, "baseline_degree", baseline_degree)
        object.__setattr__(self, "failure_code", failure_code)
        object.__setattr__(self, "resonance_estimates", resonance_estimates)
        object.__setattr__(self, "baseline_estimate", baseline_estimate)
        object.__setattr__(self, "initial_guess", initial_guess)
        object.__setattr__(self, "uncertainty_reason", uncertainty_reason)
        object.__setattr__(self, "scipy_status", scipy_status)
        object.__setattr__(self, "scipy_message", scipy_message)
        object.__setattr__(self, "nfev", nfev)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "residual_rmse", residual_rmse)
        object.__setattr__(self, "residual_scale", residual_scale)
        object.__setattr__(self, "degrees_of_freedom", degrees_of_freedom)
        object.__setattr__(self, "jacobian_rank", jacobian_rank)
        object.__setattr__(self, "q_values", q_values)


SweepStartKind: TypeAlias = Literal["preflight", "cold", "warm"]
WarmStartDisposition: TypeAlias = Literal[
    "no_successful_prior",
    "used",
    "rejected_age",
    "rejected_compatibility",
    "not_applicable_preflight",
]
WarmStartRejectionCode: TypeAlias = Literal[
    "age_limit_exceeded",
    "baseline_rebase_unrepresentable",
    "center_outside_sweep",
    "center_separation_incompatible",
    "resonance_bounds_incompatible",
    "parameterization_unrepresentable",
]


@dataclass(frozen=True, slots=True)
class SweepFitAttempt:
    """One preflight, cold, or warm fit attempt for a submitted sweep."""

    start_kind: SweepStartKind
    warm_source_update_index: int | None
    fit: SpectrumFitResult
    cpu_time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.start_kind, str):
            raise TypeError("start_kind must be a string")
        if self.start_kind not in {"preflight", "cold", "warm"}:
            raise ValueError("start_kind must be preflight, cold, or warm")
        if not isinstance(self.fit, SpectrumFitResult):
            raise TypeError("fit must be a SpectrumFitResult")

        if self.start_kind == "warm":
            if self.warm_source_update_index is None:
                raise ValueError("warm attempts require a source update index")
            warm_source_update_index = _nonnegative_int(
                self.warm_source_update_index, "warm_source_update_index"
            )
        else:
            if self.warm_source_update_index is not None:
                raise ValueError(
                    "cold and preflight attempts cannot have a warm source"
                )
            warm_source_update_index = None

        failure_code = self.fit.failure_code
        source = self.fit.diagnostics.source
        if self.start_kind == "preflight":
            if (
                self.fit.success
                or failure_code
                not in {"insufficient_samples", "uninformative_sweep"}
                or source != "none"
                or self.fit.nfev != 0
                or self.fit.initial_guess is not None
            ):
                raise ValueError("preflight attempts require a preflight failure")
        elif self.start_kind == "warm":
            if (
                (not self.fit.success and failure_code not in {
                    "optimization_failed",
                    "quality_failed",
                })
                or source != "user"
            ):
                raise ValueError("warm attempts require user-start optimizer outcomes")
        elif failure_code in {"insufficient_samples", "uninformative_sweep"}:
            raise ValueError("cold attempts cannot contain preflight failures")
        elif failure_code == "initialization_failed":
            if source not in {"detected", "fallback", "none"}:
                raise ValueError("cold initialization provenance is invalid")
        elif source not in {"detected", "fallback"}:
            raise ValueError("cold optimizer attempts require automatic initialization")

        object.__setattr__(
            self, "warm_source_update_index", warm_source_update_index
        )
        object.__setattr__(
            self, "cpu_time_s", _nonnegative_float(self.cpu_time_s, "cpu_time_s")
        )


@dataclass(frozen=True, slots=True)
class WarmSweepEstimate:
    """A compound warm-start update and its selected active fit."""

    update_index: int
    attempts: Sequence[SweepFitAttempt]
    warm_start_disposition: WarmStartDisposition
    warm_start_rejection_code: WarmStartRejectionCode | None
    warm_start_message: str | None
    active_fit: SpectrumFitResult | None
    active_source_update_index: int | None
    estimate_age_submitted_observations: int | None
    estimate_age_sequence_indices: int | None
    estimate_age_s: float | None
    observation_count: int
    cumulative_observation_count: int
    first_sequence_index: int | None
    last_sequence_index: int | None
    last_timestamp_s: float | None
    total_integration_time_s: float | None
    total_nominal_exposure_photons: float | None
    cpu_time_s: float

    def __post_init__(self) -> None:
        update_index = _nonnegative_int(self.update_index, "update_index")
        active_source_update_index = _optional_int(
            self.active_source_update_index, "active_source_update_index"
        )
        estimate_age_submitted_observations = _optional_int(
            self.estimate_age_submitted_observations,
            "estimate_age_submitted_observations",
        )
        estimate_age_sequence_indices = _optional_int(
            self.estimate_age_sequence_indices, "estimate_age_sequence_indices"
        )
        estimate_age_s = _optional_nonnegative_float(
            self.estimate_age_s, "estimate_age_s"
        )
        observation_count = _nonnegative_int(
            self.observation_count, "observation_count"
        )
        if observation_count == 0:
            raise ValueError("observation_count must be positive")
        cumulative_observation_count = _nonnegative_int(
            self.cumulative_observation_count, "cumulative_observation_count"
        )
        first_sequence_index = (
            None
            if self.first_sequence_index is None
            else _nonnegative_int(self.first_sequence_index, "first_sequence_index")
        )
        last_sequence_index = (
            None
            if self.last_sequence_index is None
            else _nonnegative_int(self.last_sequence_index, "last_sequence_index")
        )
        last_timestamp_s = _optional_nonnegative_float(
            self.last_timestamp_s, "last_timestamp_s"
        )
        total_integration_time_s = _optional_positive_float(
            self.total_integration_time_s, "total_integration_time_s"
        )
        total_nominal_exposure_photons = _optional_nonnegative_float(
            self.total_nominal_exposure_photons,
            "total_nominal_exposure_photons",
        )
        cpu_time_s = _nonnegative_float(self.cpu_time_s, "cpu_time_s")

        if isinstance(self.attempts, (str, bytes)):
            raise TypeError("attempts must be a sequence of SweepFitAttempt values")
        try:
            attempts = tuple(self.attempts)
        except TypeError as exc:
            raise TypeError(
                "attempts must be a sequence of SweepFitAttempt values"
            ) from exc
        if len(attempts) not in {1, 2}:
            raise ValueError("attempts must contain one or two attempts")
        if not all(isinstance(attempt, SweepFitAttempt) for attempt in attempts):
            raise TypeError("attempts must contain only SweepFitAttempt values")
        if any(
            attempt.start_kind == "warm"
            and (
                attempt.warm_source_update_index is None
                or attempt.warm_source_update_index >= update_index
            )
            for attempt in attempts
        ):
            raise ValueError("warm attempt sources must precede the update")

        if not isinstance(self.warm_start_disposition, str):
            raise TypeError("warm_start_disposition must be a string")
        dispositions = {
            "no_successful_prior",
            "used",
            "rejected_age",
            "rejected_compatibility",
            "not_applicable_preflight",
        }
        if self.warm_start_disposition not in dispositions:
            raise ValueError("warm_start_disposition is not recognized")
        rejection_code = _optional_string(
            self.warm_start_rejection_code, "warm_start_rejection_code"
        )
        rejection_codes = {
            "age_limit_exceeded",
            "baseline_rebase_unrepresentable",
            "center_outside_sweep",
            "center_separation_incompatible",
            "resonance_bounds_incompatible",
            "parameterization_unrepresentable",
        }
        if rejection_code is not None and rejection_code not in rejection_codes:
            raise ValueError("warm_start_rejection_code is not recognized")
        message = _optional_string(self.warm_start_message, "warm_start_message")

        disposition = self.warm_start_disposition
        kinds = tuple(attempt.start_kind for attempt in attempts)
        if disposition == "not_applicable_preflight":
            if kinds != ("preflight",):
                raise ValueError("preflight disposition requires one preflight attempt")
        elif disposition in {
            "no_successful_prior",
            "rejected_age",
            "rejected_compatibility",
        }:
            if kinds != ("cold",):
                raise ValueError("ordinary dispositions require one cold attempt")
        elif kinds == ("warm", "cold"):
            if attempts[0].fit.failure_code not in {
                "optimization_failed",
                "quality_failed",
            }:
                raise ValueError("cold recovery requires a failed warm attempt")
        elif kinds != ("warm",):
            raise ValueError("used disposition requires warm or warm-cold attempts")

        if disposition == "rejected_age":
            if rejection_code != "age_limit_exceeded":
                raise ValueError("rejected_age requires age_limit_exceeded")
            message = _required_string(message, "warm_start_message")
        elif disposition == "rejected_compatibility":
            if rejection_code not in rejection_codes - {"age_limit_exceeded"}:
                raise ValueError("rejected_compatibility requires a compatibility code")
            message = _required_string(message, "warm_start_message")
        elif rejection_code is not None or message is not None:
            raise ValueError("this disposition cannot have a rejection code or message")

        if self.active_fit is not None and not isinstance(
            self.active_fit, SpectrumFitResult
        ):
            raise TypeError("active_fit must be a SpectrumFitResult or None")
        if self.active_fit is not None and not self.active_fit.success:
            raise ValueError("active_fit must be successful")

        sequence_metadata_available = (
            first_sequence_index is not None and last_sequence_index is not None
        )
        if (first_sequence_index is None) != (last_sequence_index is None):
            raise ValueError("sequence endpoints must be jointly available")
        timestamp_available = last_timestamp_s is not None
        current_fit = attempts[-1].fit
        if current_fit.success:
            if self.active_fit is not current_fit:
                raise ValueError("a current success must be the active fit")
            if active_source_update_index != update_index:
                raise ValueError("a current success must use the current source index")
            if estimate_age_submitted_observations != 0:
                raise ValueError("a current success must have zero submitted age")
            expected_sequence_age = 0 if sequence_metadata_available else None
            if estimate_age_sequence_indices != expected_sequence_age:
                raise ValueError(
                    "current sequence age contradicts endpoint availability"
                )
            expected_seconds_age = 0.0 if timestamp_available else None
            if estimate_age_s != expected_seconds_age:
                raise ValueError(
                    "current seconds age contradicts timestamp availability"
                )
        elif self.active_fit is None:
            if any(
                value is not None
                for value in (
                    active_source_update_index,
                    estimate_age_submitted_observations,
                    estimate_age_sequence_indices,
                    estimate_age_s,
                )
            ):
                raise ValueError("no active fit permits no source or age fields")
        else:
            if (
                active_source_update_index is None
                or not 0 <= active_source_update_index < update_index
            ):
                raise ValueError("a stale fit requires an older source update")
            if (
                estimate_age_submitted_observations is None
                or estimate_age_submitted_observations <= 0
            ):
                raise ValueError("a stale fit requires positive submitted age")
            if sequence_metadata_available:
                if (
                    estimate_age_sequence_indices is None
                    or estimate_age_sequence_indices <= 0
                ):
                    raise ValueError("a stale fit requires positive sequence age")
            elif estimate_age_sequence_indices is not None:
                raise ValueError("sequence age requires sequence endpoints")
            if timestamp_available:
                if estimate_age_s is None or estimate_age_s <= 0.0:
                    raise ValueError("a stale fit requires positive seconds age")
            elif estimate_age_s is not None:
                raise ValueError("seconds age requires a completion timestamp")

        if not current_fit.success:
            if disposition == "no_successful_prior" and self.active_fit is not None:
                raise ValueError("no_successful_prior cannot retain an active fit")
            if disposition in {
                "used",
                "rejected_age",
                "rejected_compatibility",
            } and self.active_fit is None:
                raise ValueError("a failed seeded update requires a stale active fit")
            if (
                disposition == "used"
                and active_source_update_index
                != attempts[0].warm_source_update_index
            ):
                raise ValueError("failed warm use must retain its warm source")

        if cumulative_observation_count < observation_count:
            raise ValueError(
                "cumulative_observation_count cannot be below observation_count"
            )
        if sequence_metadata_available:
            assert first_sequence_index is not None
            assert last_sequence_index is not None
            if first_sequence_index != last_sequence_index - observation_count + 1:
                raise ValueError("sequence endpoints contradict observation_count")
        if cpu_time_s < math.fsum(attempt.cpu_time_s for attempt in attempts):
            raise ValueError("cpu_time_s cannot be below the attempt CPU sum")

        object.__setattr__(self, "update_index", update_index)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "warm_start_rejection_code", rejection_code)
        object.__setattr__(self, "warm_start_message", message)
        object.__setattr__(
            self, "active_source_update_index", active_source_update_index
        )
        object.__setattr__(
            self,
            "estimate_age_submitted_observations",
            estimate_age_submitted_observations,
        )
        object.__setattr__(
            self, "estimate_age_sequence_indices", estimate_age_sequence_indices
        )
        object.__setattr__(self, "estimate_age_s", estimate_age_s)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(
            self, "cumulative_observation_count", cumulative_observation_count
        )
        object.__setattr__(self, "first_sequence_index", first_sequence_index)
        object.__setattr__(self, "last_sequence_index", last_sequence_index)
        object.__setattr__(self, "last_timestamp_s", last_timestamp_s)
        object.__setattr__(
            self, "total_integration_time_s", total_integration_time_s
        )
        object.__setattr__(
            self,
            "total_nominal_exposure_photons",
            total_nominal_exposure_photons,
        )
        object.__setattr__(self, "cpu_time_s", cpu_time_s)

    @property
    def current_fit(self) -> SpectrumFitResult:
        return self.attempts[-1].fit

    @property
    def is_stale(self) -> bool:
        return (
            self.active_fit is not None
            and self.active_source_update_index != self.update_index
        )

    @property
    def total_nfev(self) -> int:
        return sum(attempt.fit.nfev for attempt in self.attempts)


@dataclass(frozen=True, slots=True)
class SweepEstimate:
    """A fit outcome paired with one completed sweep's public resource totals."""

    fit: SpectrumFitResult
    last_sequence_index: int | None = None
    last_timestamp_s: float | None = None
    total_integration_time_s: float | None = None
    total_nominal_exposure_photons: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fit, SpectrumFitResult):
            raise TypeError("fit must be a SpectrumFitResult")
        if self.last_sequence_index is not None:
            object.__setattr__(
                self,
                "last_sequence_index",
                _nonnegative_int(self.last_sequence_index, "last_sequence_index"),
            )
        if self.last_timestamp_s is not None:
            object.__setattr__(
                self,
                "last_timestamp_s",
                _nonnegative_float(self.last_timestamp_s, "last_timestamp_s"),
            )
        if self.total_integration_time_s is not None:
            object.__setattr__(
                self,
                "total_integration_time_s",
                _positive_float(
                    self.total_integration_time_s, "total_integration_time_s"
                ),
            )
        if self.total_nominal_exposure_photons is not None:
            object.__setattr__(
                self,
                "total_nominal_exposure_photons",
                _nonnegative_float(
                    self.total_nominal_exposure_photons,
                    "total_nominal_exposure_photons",
                ),
            )
