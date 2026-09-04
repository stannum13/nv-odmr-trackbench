"""Constrained offline fitting for complete eight-resonance ODMR sweeps."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from odmr_bench.estimators.initialization import initialize_spectrum
from odmr_bench.estimators.parameterization import (
    pack_parameters,
    parameter_bounds,
    public_parameter_transform,
    unpack_parameters,
)
from odmr_bench.estimators.types import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    FitUncertainty,
    InitializationDiagnostics,
    SpectrumFitResult,
)
from odmr_bench.models import multi_resonance_spectrum


def _free_parameter_count(configuration: FitConfiguration) -> int:
    return (
        configuration.baseline_degree
        + 1
        + 8 * (4 if configuration.model_kind == "pseudo_voigt" else 3)
    )


def _real_array(
    value: ArrayLike, name: str
) -> tuple[NDArray[np.float64] | None, str | None]:
    """Validate a real numeric array without silently projecting complex values."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return None, f"{name} must be a numeric array"
    if np.iscomplexobj(raw) or (
        raw.dtype == object and any(np.iscomplexobj(item) for item in raw.ravel())
    ):
        return None, f"{name} must not contain complex values"
    try:
        return np.asarray(raw, dtype=np.float64), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be a numeric array"


def _real_scalar(value: object, name: str) -> tuple[float | None, str | None]:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        return None, f"{name} must be a real scalar"
    try:
        canonical = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be finite"
    if not np.isfinite(canonical):
        return None, f"{name} must be finite"
    return canonical, None


def _stable_row_norms(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return Euclidean row norms without squaring extreme finite entries."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return np.asarray(np.hypot.reduce(np.abs(matrix), axis=1), dtype=np.float64)


def _packed_linearized_covariance_factor(
    jacobian: ArrayLike,
    scaled_cost: object,
    degrees_of_freedom: object,
    rank_rtol: object,
) -> tuple[NDArray[np.float64] | None, int | None, str | None]:
    """Return a packed covariance factor and rank from one validated SVD."""
    matrix, array_reason = _real_array(jacobian, "Jacobian")
    if matrix is None:
        return None, None, array_reason
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return None, None, "Jacobian must be a nonempty two-dimensional matrix"
    column_count = matrix.shape[1]
    if not np.all(np.isfinite(matrix)):
        return None, None, "Jacobian must be finite"
    rank_tolerance, scalar_reason = _real_scalar(rank_rtol, "rank_rtol")
    if rank_tolerance is None:
        return None, None, scalar_reason
    if not 0.0 < rank_tolerance < 1.0:
        return None, None, "rank_rtol must be finite and within (0, 1)"
    try:
        _, singular_values, right_vectors_t = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None, "Jacobian SVD failed"
    if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
        return None, None, "Jacobian SVD produced invalid singular values"
    cutoff = float(np.max(singular_values)) * rank_tolerance
    retained = singular_values > cutoff
    rank = int(np.count_nonzero(retained))
    if rank != column_count:
        return None, rank, "scaled Jacobian is rank deficient"
    if isinstance(degrees_of_freedom, (bool, np.bool_)) or not isinstance(
        degrees_of_freedom, (Integral, np.integer)
    ):
        return None, rank, "degrees of freedom must be an integer"
    canonical_dof = int(degrees_of_freedom)
    if canonical_dof <= 0:
        return None, rank, "degrees of freedom must be positive"
    try:
        dof_scale = float(canonical_dof)
    except (ValueError, OverflowError):
        return None, rank, "degrees of freedom must be finite"
    if not np.isfinite(dof_scale):
        return None, rank, "degrees of freedom must be finite"
    canonical_cost, scalar_reason = _real_scalar(scaled_cost, "scaled cost")
    if canonical_cost is None:
        return None, rank, scalar_reason
    if canonical_cost < 0.0:
        return None, rank, "scaled cost must be finite and non-negative"

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual_scale = (
            np.sqrt(canonical_cost) * np.sqrt(2.0) / np.sqrt(dof_scale)
        )
        covariance_factor = right_vectors_t.T * (
            residual_scale / singular_values
        )
    if not np.all(np.isfinite(covariance_factor)):
        return None, rank, "packed covariance is invalid"
    packed_standard_errors = _stable_row_norms(covariance_factor)
    if not np.all(np.isfinite(packed_standard_errors)) or (
        canonical_cost > 0.0 and np.any(packed_standard_errors == 0.0)
    ):
        return None, rank, "packed covariance is invalid"
    return np.asarray(covariance_factor, dtype=np.float64), rank, None


def _public_standard_errors(
    packed_covariance_factor: NDArray[np.float64], public_transform: ArrayLike
) -> tuple[NDArray[np.float64] | None, str | None]:
    column_count = packed_covariance_factor.shape[0]
    transform, array_reason = _real_array(public_transform, "public transform")
    if transform is None:
        return None, array_reason
    if transform.shape != (column_count, column_count):
        return None, "public transform shape must match Jacobian columns"
    if not np.all(np.isfinite(transform)):
        return None, "public transform must be finite"

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        public_covariance_factor = transform @ packed_covariance_factor
    if not np.all(np.isfinite(public_covariance_factor)):
        return None, "public covariance is invalid"
    standard_errors = _stable_row_norms(public_covariance_factor)
    nonzero_transform_rows = np.any(transform != 0.0, axis=1)
    if not np.all(np.isfinite(standard_errors)) or (
        np.any(packed_covariance_factor != 0.0)
        and np.any((standard_errors == 0.0) & nonzero_transform_rows)
    ):
        return None, "public covariance is invalid"
    return np.asarray(standard_errors, dtype=np.float64), None


def linearized_standard_errors(
    jacobian: ArrayLike,
    scaled_cost: float,
    degrees_of_freedom: int,
    public_transform: ArrayLike,
    rank_rtol: float,
) -> tuple[NDArray[np.float64] | None, int | None, str | None]:
    """Return public-unit local-linearized errors from one scaled-Jacobian SVD."""
    packed_covariance_factor, rank, reason = _packed_linearized_covariance_factor(
        jacobian, scaled_cost, degrees_of_freedom, rank_rtol
    )
    if packed_covariance_factor is None:
        return None, rank, reason
    standard_errors, reason = _public_standard_errors(
        packed_covariance_factor, public_transform
    )
    return standard_errors, rank, reason


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


def _validate_guess(
    guess: FitInitialGuess,
    configuration: FitConfiguration,
    *,
    frequency_min_hz: float,
    frequency_max_hz: float,
    frequency_reference_hz: float,
    frequency_half_span_hz: float,
    fluorescence_reference: float,
    fluorescence_scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    ids = tuple(item.resonance_id for item in guess.resonances)
    if ids != configuration.resonance_ids:
        raise ValueError("guess resonance IDs must exactly match configured order")
    if configuration.model_kind == "lorentzian" and any(
        item.eta != 1.0 for item in guess.resonances
    ):
        raise ValueError("lorentzian guesses require eta exactly equal to one")
    for resonance in guess.resonances:
        if (
            not configuration.min_fwhm_hz
            <= resonance.fwhm_hz
            <= configuration.max_fwhm_hz
        ):
            raise ValueError("guess FWHM lies outside configured bounds")
        if not 0.0 <= resonance.amplitude <= configuration.max_amplitude:
            raise ValueError("guess amplitude lies outside configured bounds")

    packed = pack_parameters(
        guess,
        configuration,
        frequency_reference_hz=frequency_reference_hz,
        frequency_half_span_hz=frequency_half_span_hz,
        fluorescence_reference=fluorescence_reference,
        fluorescence_scale=fluorescence_scale,
    )
    lower, upper = parameter_bounds(
        guess,
        configuration,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
        frequency_reference_hz=frequency_reference_hz,
        frequency_half_span_hz=frequency_half_span_hz,
        fluorescence_scale=fluorescence_scale,
    )
    baseline_size = configuration.baseline_degree + 1
    if (
        not np.all(np.isfinite(packed))
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
    ):
        raise ValueError("numerical parameterization contains invalid values")
    if not np.all(np.isneginf(lower[:baseline_size])) or not np.all(
        np.isposinf(upper[:baseline_size])
    ):
        raise ValueError("baseline parameter bounds must be intentionally infinite")
    if not np.all(np.isfinite(lower[baseline_size:])) or not np.all(
        np.isfinite(upper[baseline_size:])
    ):
        raise ValueError("configured resonance bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("numerical parameterization must have strict bounds")
    if np.any(packed < lower) or np.any(packed > upper):
        raise ValueError("guess parameters must lie inside optimizer bounds")
    return packed, lower, upper


def _append_diagnostic_message(
    diagnostics: InitializationDiagnostics, message: str
) -> InitializationDiagnostics:
    return InitializationDiagnostics(
        source=diagnostics.source,
        candidate_count=diagnostics.candidate_count,
        selected_indices=diagnostics.selected_indices,
        used_fallback=diagnostics.used_fallback,
        messages=(*diagnostics.messages, message),
    )


def _public_parameters_valid(
    fitted: FitInitialGuess,
    configuration: FitConfiguration,
    frequency_min_hz: float,
    frequency_max_hz: float,
) -> bool:
    resonances = fitted.resonances
    centers = np.asarray([item.center_hz for item in resonances])
    widths = np.asarray([item.fwhm_hz for item in resonances])
    amplitudes = np.asarray([item.amplitude for item in resonances])
    etas = np.asarray([item.eta for item in resonances])
    return bool(
        np.all(centers >= frequency_min_hz)
        and np.all(centers <= frequency_max_hz)
        and np.all(np.diff(centers) > 0.0)
        and np.all(np.diff(centers) >= configuration.min_center_separation_hz)
        and np.all(widths >= configuration.min_fwhm_hz)
        and np.all(widths <= configuration.max_fwhm_hz)
        and np.all(amplitudes >= 0.0)
        and np.all(amplitudes <= configuration.max_amplitude)
        and np.all(etas >= 0.0)
        and np.all(etas <= 1.0)
        and (configuration.model_kind == "pseudo_voigt" or np.all(etas == 1.0))
    )


def _scaled_residual_function(
    sweep: CompleteSweep,
    configuration: FitConfiguration,
    *,
    frequency_reference_hz: float,
    frequency_half_span_hz: float,
    fluorescence_reference: float,
    fluorescence_scale: float,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def residual(packed: NDArray[np.float64]) -> NDArray[np.float64]:
        public = unpack_parameters(
            packed,
            configuration,
            frequency_reference_hz=frequency_reference_hz,
            frequency_half_span_hz=frequency_half_span_hz,
            fluorescence_reference=fluorescence_reference,
            fluorescence_scale=fluorescence_scale,
        )
        model = multi_resonance_spectrum(
            sweep.frequency_hz, public.resonances, public.baseline
        )
        return np.asarray(
            (model - sweep.fluorescence) / fluorescence_scale, dtype=np.float64
        )

    return residual


def _raw_fit_metrics(
    fun: object, scaled_cost: object, fluorescence_scale: float
) -> tuple[float | None, float | None]:
    try:
        residual = np.asarray(fun, dtype=np.float64)
        cost_value = float(scaled_cost)
    except (TypeError, ValueError):
        return None, None
    if (
        residual.ndim != 1
        or residual.size == 0
        or not np.all(np.isfinite(residual))
        or not np.isfinite(cost_value)
        or cost_value < 0.0
    ):
        return None, None
    with np.errstate(over="ignore", invalid="ignore"):
        raw_cost = cost_value * fluorescence_scale * fluorescence_scale
        raw_residual = residual * fluorescence_scale
        rmse = float(np.sqrt(np.mean(raw_residual * raw_residual)))
    if not np.isfinite(raw_cost) or not np.isfinite(rmse):
        return None, None
    return raw_cost, rmse


def _baseline_only_sse(
    sweep: CompleteSweep, reference_hz: float, half_span_hz: float, degree: int
) -> float:
    z = (sweep.frequency_hz - reference_hz) / half_span_hz
    design = np.column_stack([z**power for power in range(degree + 1)])
    coefficients, _, _, _ = np.linalg.lstsq(design, sweep.fluorescence, rcond=None)
    residual = design @ coefficients - sweep.fluorescence
    return float(residual @ residual)


def _uncertainty_from_errors(
    errors: NDArray[np.float64], configuration: FitConfiguration
) -> FitUncertainty:
    baseline_size = configuration.baseline_degree + 1
    resonance_errors = errors[baseline_size : baseline_size + 24].reshape(8, 3)
    eta_errors = errors[-8:] if configuration.model_kind == "pseudo_voigt" else None
    return FitUncertainty(
        baseline_standard_errors=errors[:baseline_size],
        amplitude=resonance_errors[:, 0],
        center_hz=resonance_errors[:, 1],
        fwhm_hz=resonance_errors[:, 2],
        eta=eta_errors,
    )


def _amplitudes_have_local_significance(
    fitted: FitInitialGuess,
    packed_covariance_factor: NDArray[np.float64] | None,
    configuration: FitConfiguration,
    fluorescence_scale: float,
) -> tuple[bool, str | None]:
    """Apply a model-conditioned local amplitude/SE gate to every fitted line."""
    unavailable_reason = (
        "model-conditioned local amplitude significance is unavailable or non-finite"
    )
    if packed_covariance_factor is None:
        return False, unavailable_reason

    baseline_size = configuration.baseline_degree + 1
    amplitude_indices = baseline_size + 3 * np.arange(8)
    packed_standard_errors = _stable_row_norms(packed_covariance_factor)
    packed_amplitude_errors = packed_standard_errors[amplitude_indices]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        amplitude_standard_errors = packed_amplitude_errors * fluorescence_scale
    if (
        np.any((packed_amplitude_errors > 0.0) & (amplitude_standard_errors == 0.0))
        or np.any(amplitude_standard_errors < 0.0)
        or not np.all(np.isfinite(amplitude_standard_errors))
    ):
        return False, unavailable_reason

    amplitudes = np.asarray(
        [resonance.amplitude for resonance in fitted.resonances], dtype=np.float64
    )
    significance = np.empty(8, dtype=np.float64)
    zero_error = amplitude_standard_errors == 0.0
    positive_zero_error = zero_error & (amplitudes > 0.0)
    significance[positive_zero_error] = np.inf
    significance[zero_error & ~positive_zero_error] = np.nan
    positive_error = ~zero_error
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        significance[positive_error] = (
            amplitudes[positive_error] / amplitude_standard_errors[positive_error]
        )
    valid_evidence = np.isfinite(significance) | positive_zero_error
    if not np.all(valid_evidence):
        return False, unavailable_reason
    if np.all(significance >= configuration.min_amplitude_significance):
        return True, None
    return (
        False,
        "one or more fitted amplitudes fail the model-conditioned local "
        "min_amplitude_significance threshold",
    )


def fit_spectrum(
    sweep: CompleteSweep,
    configuration: FitConfiguration,
    initial_guess: FitInitialGuess | None = None,
) -> SpectrumFitResult:
    """Fit one complete sweep without truth, future data, or prior-fit state."""
    if not isinstance(sweep, CompleteSweep):
        raise TypeError("sweep must be a CompleteSweep")
    if not isinstance(configuration, FitConfiguration):
        raise TypeError("configuration must be a FitConfiguration")
    if initial_guess is not None and not isinstance(initial_guess, FitInitialGuess):
        raise TypeError("initial_guess must be a FitInitialGuess or None")

    free_parameters = _free_parameter_count(configuration)
    degrees_of_freedom = int(sweep.frequency_hz.size - free_parameters)
    if degrees_of_freedom <= 0:
        return _preoptimization_failure(
            configuration,
            "insufficient_samples",
            _empty_diagnostics("sample count must exceed the free parameter count"),
            degrees_of_freedom,
        )

    frequency_min = float(sweep.frequency_hz[0])
    frequency_max = float(sweep.frequency_hz[-1])
    frequency_reference = frequency_min / 2.0 + frequency_max / 2.0
    frequency_half_span = frequency_max / 2.0 - frequency_min / 2.0
    with np.errstate(over="ignore", invalid="ignore"):
        fluorescence_scale = float(np.ptp(sweep.fluorescence))
    if not np.isfinite(fluorescence_scale) or fluorescence_scale == 0.0:
        return _preoptimization_failure(
            configuration,
            "uninformative_sweep",
            _empty_diagnostics("fluorescence variation is zero or non-finite"),
            degrees_of_freedom,
        )
    fluorescence_anchor = float(sweep.fluorescence[0])
    with np.errstate(over="ignore", invalid="ignore"):
        shifted_fluorescence = sweep.fluorescence - fluorescence_anchor
        fluorescence_reference = float(
            fluorescence_anchor + np.median(shifted_fluorescence)
        )
    if not np.isfinite(fluorescence_reference):
        return _preoptimization_failure(
            configuration,
            "uninformative_sweep",
            _empty_diagnostics("fluorescence origin is non-finite numerically"),
            degrees_of_freedom,
        )

    if initial_guess is None:
        guess, diagnostics = initialize_spectrum(sweep, configuration)
        if guess is None:
            return _preoptimization_failure(
                configuration,
                "initialization_failed",
                diagnostics,
                degrees_of_freedom,
                residual_scale=fluorescence_scale,
            )
    else:
        guess = initial_guess
        diagnostics = InitializationDiagnostics(
            source="user",
            candidate_count=0,
            selected_indices=(),
            used_fallback=False,
            messages=(),
        )
    assert guess is not None

    try:
        packed, lower, upper = _validate_guess(
            guess,
            configuration,
            frequency_min_hz=frequency_min,
            frequency_max_hz=frequency_max,
            frequency_reference_hz=frequency_reference,
            frequency_half_span_hz=frequency_half_span,
            fluorescence_reference=fluorescence_reference,
            fluorescence_scale=fluorescence_scale,
        )
    except ValueError as error:
        if initial_guess is not None:
            raise
        diagnostics = _append_diagnostic_message(
            diagnostics, f"initial guess preflight failed: {error}"
        )
        return _preoptimization_failure(
            configuration,
            "initialization_failed",
            diagnostics,
            degrees_of_freedom,
            residual_scale=fluorescence_scale,
        )

    residual_function = _scaled_residual_function(
        sweep,
        configuration,
        frequency_reference_hz=frequency_reference,
        frequency_half_span_hz=frequency_half_span,
        fluorescence_reference=fluorescence_reference,
        fluorescence_scale=fluorescence_scale,
    )
    optimization = least_squares(
        residual_function,
        packed,
        bounds=(lower, upper),
        method="trf",
        loss="linear",
        x_scale=1.0,
        max_nfev=configuration.max_nfev,
    )
    raw_cost, residual_rmse = _raw_fit_metrics(
        optimization.fun, optimization.cost, fluorescence_scale
    )
    try:
        optimizer_x = np.asarray(optimization.x, dtype=np.float64)
        optimizer_fun = np.asarray(optimization.fun, dtype=np.float64)
        optimizer_cost = float(optimization.cost)
        finite_payload = (
            optimizer_x.shape == packed.shape
            and optimizer_fun.shape == sweep.fluorescence.shape
            and np.all(np.isfinite(optimizer_x))
            and np.all(np.isfinite(optimizer_fun))
            and np.isfinite(optimizer_cost)
            and optimizer_cost >= 0.0
        )
    except (TypeError, ValueError):
        finite_payload = False
    if not optimization.success:
        return SpectrumFitResult(
            success=False,
            failure_code="optimization_failed",
            model_kind=configuration.model_kind,
            baseline_degree=configuration.baseline_degree,
            resonance_estimates=(),
            baseline_estimate=None,
            diagnostics=diagnostics,
            initial_guess=guess,
            uncertainty=None,
            uncertainty_reason="optimizer did not return a finite successful solution",
            scipy_status=int(optimization.status),
            scipy_message=str(optimization.message),
            nfev=int(optimization.nfev),
            cost=raw_cost,
            residual_rmse=residual_rmse,
            residual_scale=fluorescence_scale,
            degrees_of_freedom=degrees_of_freedom,
            jacobian_rank=None,
        )
    if not finite_payload:
        return SpectrumFitResult(
            success=False,
            failure_code="quality_failed",
            model_kind=configuration.model_kind,
            baseline_degree=configuration.baseline_degree,
            resonance_estimates=(),
            baseline_estimate=None,
            diagnostics=diagnostics,
            initial_guess=guess,
            uncertainty=None,
            uncertainty_reason=(
                "optimizer returned non-finite parameters, residuals, or cost"
            ),
            scipy_status=int(optimization.status),
            scipy_message=str(optimization.message),
            nfev=int(optimization.nfev),
            cost=None,
            residual_rmse=None,
            residual_scale=fluorescence_scale,
            degrees_of_freedom=degrees_of_freedom,
            jacobian_rank=None,
        )

    if raw_cost is None or residual_rmse is None:
        return SpectrumFitResult(
            success=False,
            failure_code="quality_failed",
            model_kind=configuration.model_kind,
            baseline_degree=configuration.baseline_degree,
            resonance_estimates=(),
            baseline_estimate=None,
            diagnostics=diagnostics,
            initial_guess=guess,
            uncertainty=None,
            uncertainty_reason=(
                "optimizer returned non-finite parameters, residuals, or cost"
            ),
            scipy_status=int(optimization.status),
            scipy_message=str(optimization.message),
            nfev=int(optimization.nfev),
            cost=None,
            residual_rmse=None,
            residual_scale=fluorescence_scale,
            degrees_of_freedom=degrees_of_freedom,
            jacobian_rank=None,
        )

    packed_covariance_factor, rank, uncertainty_reason = (
        _packed_linearized_covariance_factor(
            optimization.jac,
            float(optimization.cost),
            degrees_of_freedom,
            configuration.rank_rtol,
        )
    )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            fitted = unpack_parameters(
                optimization.x,
                configuration,
                frequency_reference_hz=frequency_reference,
                frequency_half_span_hz=frequency_half_span,
                fluorescence_reference=fluorescence_reference,
                fluorescence_scale=fluorescence_scale,
            )
    except (FloatingPointError, OverflowError, ValueError):
        return SpectrumFitResult(
            success=False,
            failure_code="quality_failed",
            model_kind=configuration.model_kind,
            baseline_degree=configuration.baseline_degree,
            resonance_estimates=(),
            baseline_estimate=None,
            diagnostics=diagnostics,
            initial_guess=guess,
            uncertainty=None,
            uncertainty_reason=(
                "packed solution does not map to finite public parameters"
            ),
            scipy_status=int(optimization.status),
            scipy_message=str(optimization.message),
            nfev=int(optimization.nfev),
            cost=raw_cost,
            residual_rmse=residual_rmse,
            residual_scale=fluorescence_scale,
            degrees_of_freedom=degrees_of_freedom,
            jacobian_rank=rank,
        )
    significant, significance_reason = _amplitudes_have_local_significance(
        fitted, packed_covariance_factor, configuration, fluorescence_scale
    )

    standard_errors: NDArray[np.float64] | None = None
    if packed_covariance_factor is not None:
        try:
            transform = public_parameter_transform(
                configuration,
                frequency_half_span_hz=frequency_half_span,
                fluorescence_scale=fluorescence_scale,
            )
        except ValueError as error:
            uncertainty_reason = f"public parameter transform unavailable: {error}"
        else:
            standard_errors, uncertainty_reason = _public_standard_errors(
                packed_covariance_factor, transform
            )
    uncertainty = (
        _uncertainty_from_errors(standard_errors, configuration)
        if standard_errors is not None
        else None
    )

    centers = np.asarray([item.center_hz for item in fitted.resonances])
    amplitudes = np.asarray([item.amplitude for item in fitted.resonances])
    fitted_values = np.asarray(optimization.x, dtype=np.float64)
    bounds_valid = np.all(fitted_values >= lower) and np.all(fitted_values <= upper)
    public_bounds_valid = _public_parameters_valid(
        fitted, configuration, frequency_min, frequency_max
    )
    separated = np.all(np.diff(centers) >= configuration.min_center_separation_hz)
    resolved = np.all(amplitudes >= configuration.min_resolved_amplitude)
    baseline_sse = _baseline_only_sse(
        sweep, frequency_reference, frequency_half_span, configuration.baseline_degree
    )
    fitted_sse = 2.0 * raw_cost
    improvement = (
        (baseline_sse - fitted_sse) / baseline_sse
        if np.isfinite(baseline_sse) and baseline_sse > 0.0
        else -np.inf
    )
    quality_reasons: list[str] = []
    if rank != free_parameters:
        quality_reasons.append("scaled Jacobian is not full column rank")
    if not bounds_valid:
        quality_reasons.append("fitted parameters violate configured bounds")
    if not public_bounds_valid:
        quality_reasons.append("public fitted parameters violate configured bounds")
    if not separated:
        quality_reasons.append("fitted centers violate minimum separation")
    if not resolved:
        quality_reasons.append("one or more fitted amplitudes are unresolved")
    if not significant:
        assert significance_reason is not None
        quality_reasons.append(significance_reason)
    if not np.isfinite(residual_rmse):
        quality_reasons.append("residual RMSE is non-finite")
    if improvement < configuration.min_baseline_sse_improvement:
        quality_reasons.append("baseline-only SSE improvement is insufficient")

    if quality_reasons:
        return SpectrumFitResult(
            success=False,
            failure_code="quality_failed",
            model_kind=configuration.model_kind,
            baseline_degree=configuration.baseline_degree,
            resonance_estimates=(),
            baseline_estimate=None,
            diagnostics=diagnostics,
            initial_guess=guess,
            uncertainty=None,
            uncertainty_reason="; ".join(quality_reasons),
            scipy_status=int(optimization.status),
            scipy_message=str(optimization.message),
            nfev=int(optimization.nfev),
            cost=raw_cost,
            residual_rmse=residual_rmse,
            residual_scale=fluorescence_scale,
            degrees_of_freedom=degrees_of_freedom,
            jacobian_rank=rank,
        )

    return SpectrumFitResult(
        success=True,
        failure_code=None,
        model_kind=configuration.model_kind,
        baseline_degree=configuration.baseline_degree,
        resonance_estimates=fitted.resonances,
        baseline_estimate=fitted.baseline,
        diagnostics=diagnostics,
        initial_guess=guess,
        uncertainty=uncertainty,
        uncertainty_reason=uncertainty_reason,
        scipy_status=int(optimization.status),
        scipy_message=str(optimization.message),
        nfev=int(optimization.nfev),
        cost=raw_cost,
        residual_rmse=residual_rmse,
        residual_scale=fluorescence_scale,
        degrees_of_freedom=degrees_of_freedom,
        jacobian_rank=rank,
    )
