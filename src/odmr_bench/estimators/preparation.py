"""Shared numerical preparation for cold and warm complete-sweep fits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from odmr_bench.estimators.parameterization import (
    _finite_product_ratio,
    pack_parameters,
    parameter_bounds,
)
from odmr_bench.estimators.types import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    InitializationDiagnostics,
    SpectrumFitResult,
)
from odmr_bench.models import Baseline


class BaselineRebaseError(ValueError):
    """A deliberate finite-float baseline rebase failure."""


class InitialGuessCompatibilityError(ValueError):
    """A deliberate initial guess or parameterization incompatibility."""


WarmStartCompatibilityCode: TypeAlias = Literal[
    "baseline_rebase_unrepresentable",
    "center_outside_sweep",
    "center_separation_incompatible",
    "resonance_bounds_incompatible",
    "parameterization_unrepresentable",
]

_WARM_START_COMPATIBILITY_CODES = frozenset(
    {
        "baseline_rebase_unrepresentable",
        "center_outside_sweep",
        "center_separation_incompatible",
        "resonance_bounds_incompatible",
        "parameterization_unrepresentable",
    }
)


@dataclass(frozen=True, slots=True)
class WarmStartPreparation:
    """An exclusive compatible guess or typed numerical rejection."""

    guess: FitInitialGuess | None
    rejection_code: WarmStartCompatibilityCode | None
    message: str | None

    def __post_init__(self) -> None:
        if self.guess is not None and not isinstance(self.guess, FitInitialGuess):
            raise TypeError("guess must be a FitInitialGuess or None")
        if self.guess is not None:
            if self.rejection_code is not None or self.message is not None:
                raise ValueError("a compatible guess cannot include a rejection")
            object.__setattr__(
                self,
                "guess",
                FitInitialGuess(self.guess.resonances, self.guess.baseline),
            )
            return
        if self.rejection_code not in _WARM_START_COMPATIBILITY_CODES:
            raise ValueError("a rejection requires a closed compatibility code")
        if not isinstance(self.message, str):
            raise TypeError("a rejection message must be a string")
        if not self.message:
            raise ValueError("a rejection message must be nonempty")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    canonical = int(value)
    if canonical <= 0:
        raise ValueError(f"{name} must be positive")
    return canonical


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    try:
        canonical = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    return canonical


@dataclass(frozen=True, slots=True)
class FitPreflight:
    """Start-independent normalization values for one fit-ready sweep."""

    free_parameter_count: int
    degrees_of_freedom: int
    frequency_min_hz: float
    frequency_max_hz: float
    frequency_reference_hz: float
    frequency_half_span_hz: float
    fluorescence_reference: float
    fluorescence_scale: float

    def __post_init__(self) -> None:
        free_parameter_count = _positive_int(
            self.free_parameter_count, "free_parameter_count"
        )
        degrees_of_freedom = _positive_int(
            self.degrees_of_freedom, "degrees_of_freedom"
        )
        frequency_min_hz = _finite_float(self.frequency_min_hz, "frequency_min_hz")
        frequency_max_hz = _finite_float(self.frequency_max_hz, "frequency_max_hz")
        frequency_reference_hz = _finite_float(
            self.frequency_reference_hz, "frequency_reference_hz"
        )
        frequency_half_span_hz = _finite_float(
            self.frequency_half_span_hz, "frequency_half_span_hz"
        )
        fluorescence_reference = _finite_float(
            self.fluorescence_reference, "fluorescence_reference"
        )
        fluorescence_scale = _finite_float(
            self.fluorescence_scale, "fluorescence_scale"
        )
        if frequency_max_hz <= frequency_min_hz:
            raise ValueError("frequency_min_hz must be below frequency_max_hz")
        if frequency_half_span_hz <= 0.0:
            raise ValueError("frequency_half_span_hz must be positive")
        if fluorescence_scale <= 0.0:
            raise ValueError("fluorescence_scale must be positive")
        expected_reference = frequency_min_hz / 2.0 + frequency_max_hz / 2.0
        expected_half_span = frequency_max_hz / 2.0 - frequency_min_hz / 2.0
        if frequency_reference_hz != expected_reference:
            raise ValueError("frequency_reference_hz must equal the endpoint midpoint")
        if frequency_half_span_hz != expected_half_span:
            raise ValueError("frequency_half_span_hz must equal the endpoint half-span")
        object.__setattr__(self, "free_parameter_count", free_parameter_count)
        object.__setattr__(self, "degrees_of_freedom", degrees_of_freedom)
        object.__setattr__(self, "frequency_min_hz", frequency_min_hz)
        object.__setattr__(self, "frequency_max_hz", frequency_max_hz)
        object.__setattr__(self, "frequency_reference_hz", frequency_reference_hz)
        object.__setattr__(self, "frequency_half_span_hz", frequency_half_span_hz)
        object.__setattr__(self, "fluorescence_reference", fluorescence_reference)
        object.__setattr__(self, "fluorescence_scale", fluorescence_scale)


def _free_parameter_count(configuration: FitConfiguration) -> int:
    return (
        configuration.baseline_degree
        + 1
        + 8 * (4 if configuration.model_kind == "pseudo_voigt" else 3)
    )


def _empty_diagnostics(message: str) -> InitializationDiagnostics:
    return InitializationDiagnostics(
        source="none",
        candidate_count=0,
        selected_indices=(),
        used_fallback=False,
        messages=(message,),
    )


def _preoptimization_failure(
    configuration: FitConfiguration,
    failure_code: str,
    diagnostics: InitializationDiagnostics,
    degrees_of_freedom: int,
    *,
    residual_scale: float | None = None,
) -> SpectrumFitResult:
    reason = (
        diagnostics.messages[-1]
        if diagnostics.messages
        else f"{failure_code.replace('_', ' ')} before optimization"
    )
    return SpectrumFitResult(
        success=False,
        failure_code=failure_code,
        model_kind=configuration.model_kind,
        baseline_degree=configuration.baseline_degree,
        resonance_estimates=(),
        baseline_estimate=None,
        diagnostics=diagnostics,
        initial_guess=None,
        uncertainty=None,
        uncertainty_reason=reason,
        scipy_status=None,
        scipy_message=None,
        nfev=0,
        cost=None,
        residual_rmse=None,
        residual_scale=residual_scale,
        degrees_of_freedom=degrees_of_freedom,
        jacobian_rank=None,
    )


def _stable_fluorescence_reference(fluorescence: np.ndarray) -> float:
    """Return the median origin without summing large same-sign endpoints."""
    anchor = float(fluorescence[0])
    return float(anchor + np.median(fluorescence - anchor))


def start_independent_preflight(
    sweep: CompleteSweep,
    configuration: FitConfiguration,
) -> FitPreflight | SpectrumFitResult:
    """Return exact Stage 6.1 failures or shared fit normalization values."""
    if not isinstance(sweep, CompleteSweep):
        raise TypeError("sweep must be a CompleteSweep")
    if not isinstance(configuration, FitConfiguration):
        raise TypeError("configuration must be a FitConfiguration")

    free_parameters = _free_parameter_count(configuration)
    degrees_of_freedom = int(sweep.frequency_hz.size - free_parameters)
    if degrees_of_freedom <= 0:
        return _preoptimization_failure(
            configuration,
            "insufficient_samples",
            _empty_diagnostics("sample count must exceed the free parameter count"),
            degrees_of_freedom,
        )

    frequency_min_hz = float(sweep.frequency_hz[0])
    frequency_max_hz = float(sweep.frequency_hz[-1])
    frequency_reference_hz = frequency_min_hz / 2.0 + frequency_max_hz / 2.0
    frequency_half_span_hz = frequency_max_hz / 2.0 - frequency_min_hz / 2.0
    with np.errstate(over="ignore", invalid="ignore"):
        fluorescence_scale = float(np.ptp(sweep.fluorescence))
    if not np.isfinite(fluorescence_scale) or fluorescence_scale == 0.0:
        return _preoptimization_failure(
            configuration,
            "uninformative_sweep",
            _empty_diagnostics("fluorescence variation is zero or non-finite"),
            degrees_of_freedom,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        fluorescence_reference = _stable_fluorescence_reference(sweep.fluorescence)
    if not np.isfinite(fluorescence_reference):
        return _preoptimization_failure(
            configuration,
            "uninformative_sweep",
            _empty_diagnostics("fluorescence origin is non-finite numerically"),
            degrees_of_freedom,
        )
    return FitPreflight(
        free_parameter_count=free_parameters,
        degrees_of_freedom=degrees_of_freedom,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
        frequency_reference_hz=frequency_reference_hz,
        frequency_half_span_hz=frequency_half_span_hz,
        fluorescence_reference=fluorescence_reference,
        fluorescence_scale=fluorescence_scale,
    )


def validate_initial_guess(
    guess: FitInitialGuess,
    configuration: FitConfiguration,
    preflight: FitPreflight,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate and pack a guess through the sole Stage 6.1 parameterization."""
    if not isinstance(guess, FitInitialGuess):
        raise TypeError("guess must be a FitInitialGuess")
    if not isinstance(configuration, FitConfiguration):
        raise TypeError("configuration must be a FitConfiguration")
    if not isinstance(preflight, FitPreflight):
        raise TypeError("preflight must be a FitPreflight")

    ids = tuple(item.resonance_id for item in guess.resonances)
    if ids != configuration.resonance_ids:
        raise InitialGuessCompatibilityError(
            "guess resonance IDs must exactly match configured order"
        )
    if configuration.model_kind == "lorentzian" and any(
        item.eta != 1.0 for item in guess.resonances
    ):
        raise InitialGuessCompatibilityError(
            "lorentzian guesses require eta exactly equal to one"
        )
    for resonance in guess.resonances:
        if (
            not configuration.min_fwhm_hz
            <= resonance.fwhm_hz
            <= configuration.max_fwhm_hz
        ):
            raise InitialGuessCompatibilityError(
                "guess FWHM lies outside configured bounds"
            )
        if not 0.0 <= resonance.amplitude <= configuration.max_amplitude:
            raise InitialGuessCompatibilityError(
                "guess amplitude lies outside configured bounds"
            )

    try:
        packed = pack_parameters(
            guess,
            configuration,
            frequency_reference_hz=preflight.frequency_reference_hz,
            frequency_half_span_hz=preflight.frequency_half_span_hz,
            fluorescence_reference=preflight.fluorescence_reference,
            fluorescence_scale=preflight.fluorescence_scale,
        )
        lower, upper = parameter_bounds(
            guess,
            configuration,
            frequency_min_hz=preflight.frequency_min_hz,
            frequency_max_hz=preflight.frequency_max_hz,
            frequency_reference_hz=preflight.frequency_reference_hz,
            frequency_half_span_hz=preflight.frequency_half_span_hz,
            fluorescence_scale=preflight.fluorescence_scale,
        )
    except ValueError as error:
        raise InitialGuessCompatibilityError(str(error)) from error

    baseline_size = configuration.baseline_degree + 1
    if (
        not np.all(np.isfinite(packed))
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
    ):
        raise InitialGuessCompatibilityError(
            "numerical parameterization contains invalid values"
        )
    if not np.all(np.isneginf(lower[:baseline_size])) or not np.all(
        np.isposinf(upper[:baseline_size])
    ):
        raise InitialGuessCompatibilityError(
            "baseline parameter bounds must be intentionally infinite"
        )
    if not np.all(np.isfinite(lower[baseline_size:])) or not np.all(
        np.isfinite(upper[baseline_size:])
    ):
        raise InitialGuessCompatibilityError(
            "configured resonance bounds must be finite"
        )
    if np.any(lower >= upper):
        raise InitialGuessCompatibilityError(
            "numerical parameterization must have strict bounds"
        )
    if np.any(packed < lower) or np.any(packed > upper):
        raise InitialGuessCompatibilityError(
            "guess parameters must lie inside optimizer bounds"
        )
    return packed, lower, upper


def rebase_baseline(baseline: Baseline, new_reference_hz: float) -> Baseline:
    """Re-express a public baseline at a new finite reference frequency."""
    if not isinstance(baseline, Baseline):
        raise TypeError("baseline must be a Baseline")
    new_reference = _finite_float(new_reference_hz, "new_reference_hz")
    b0 = baseline.intercept
    b1 = baseline.slope_per_hz
    b2 = baseline.quadratic_per_hz2
    if b1 == 0.0 and b2 == 0.0:
        return Baseline(b0, new_reference, 0.0, 0.0)

    d = new_reference - baseline.reference_hz
    if not math.isfinite(d):
        raise BaselineRebaseError(
            "baseline reference difference is not representable"
        )

    try:
        b1_d = (
            0.0
            if b1 == 0.0 or d == 0.0
            else _finite_product_ratio(
                (b1, d), (), "baseline slope-reference product"
            )
        )
        b2_d2 = (
            0.0
            if b2 == 0.0 or d == 0.0
            else _finite_product_ratio(
                (b2, d, d), (), "baseline quadratic-reference product"
            )
        )
        two_b2_d = (
            0.0
            if b2 == 0.0 or d == 0.0
            else _finite_product_ratio(
                (2.0, b2, d), (), "baseline derivative-reference product"
            )
        )
    except ValueError as error:
        raise BaselineRebaseError(str(error)) from error
    try:
        rebased_intercept = math.fsum((b0, b1_d, b2_d2))
        rebased_slope = math.fsum((b1, two_b2_d))
    except OverflowError as error:
        raise BaselineRebaseError(
            "rebased baseline sum is not representable"
        ) from error
    if not math.isfinite(rebased_intercept) or not math.isfinite(rebased_slope):
        raise BaselineRebaseError("rebased baseline sum is not representable")
    return Baseline(rebased_intercept, new_reference, rebased_slope, b2)


def _warm_rejection(
    code: WarmStartCompatibilityCode, message: str
) -> WarmStartPreparation:
    return WarmStartPreparation(None, code, message)


def prepare_warm_start(
    prior_fit: SpectrumFitResult,
    configuration: FitConfiguration,
    preflight: FitPreflight,
) -> WarmStartPreparation:
    """Prepare a validated prior fit using deterministic failure precedence."""
    if not isinstance(prior_fit, SpectrumFitResult):
        raise TypeError("prior_fit must be a SpectrumFitResult")
    if not isinstance(configuration, FitConfiguration):
        raise TypeError("configuration must be a FitConfiguration")
    if not isinstance(preflight, FitPreflight):
        raise TypeError("preflight must be a FitPreflight")
    if not prior_fit.success:
        raise ValueError("prior_fit must be a successful prior fit")

    prior_ids = tuple(item.resonance_id for item in prior_fit.resonance_estimates)
    if (
        prior_fit.model_kind != configuration.model_kind
        or prior_fit.baseline_degree != configuration.baseline_degree
        or prior_ids != configuration.resonance_ids
    ):
        return _warm_rejection(
            "resonance_bounds_incompatible",
            "prior model, baseline degree, or ordered resonance IDs are incompatible",
        )
    for resonance in prior_fit.resonance_estimates:
        if (
            not configuration.min_fwhm_hz
            <= resonance.fwhm_hz
            <= configuration.max_fwhm_hz
        ):
            return _warm_rejection(
                "resonance_bounds_incompatible",
                "prior FWHM lies outside configured inclusive bounds",
            )
        if not 0.0 <= resonance.amplitude <= configuration.max_amplitude:
            return _warm_rejection(
                "resonance_bounds_incompatible",
                "prior amplitude lies outside configured inclusive bounds",
            )

    assert prior_fit.baseline_estimate is not None
    try:
        baseline = rebase_baseline(
            prior_fit.baseline_estimate, preflight.frequency_reference_hz
        )
    except BaselineRebaseError as error:
        return _warm_rejection("baseline_rebase_unrepresentable", str(error))

    centers = np.asarray(
        [item.center_hz for item in prior_fit.resonance_estimates], dtype=np.float64
    )
    if np.any(centers < preflight.frequency_min_hz) or np.any(
        centers > preflight.frequency_max_hz
    ):
        return _warm_rejection(
            "center_outside_sweep",
            "one or more prior centers lie outside the submitted sweep",
        )
    center_gaps = np.diff(centers)
    if np.any(center_gaps <= 0.0) or np.any(
        center_gaps < configuration.min_center_separation_hz
    ):
        return _warm_rejection(
            "center_separation_incompatible",
            "prior centers are not strictly ordered at the configured separation",
        )

    guess = FitInitialGuess(prior_fit.resonance_estimates, baseline)
    try:
        validate_initial_guess(guess, configuration, preflight)
    except InitialGuessCompatibilityError as error:
        return _warm_rejection("parameterization_unrepresentable", str(error))
    return WarmStartPreparation(guess, None, None)
