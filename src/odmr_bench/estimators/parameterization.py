"""Dimensionless parameterization for constrained eight-resonance fitting."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray

from odmr_bench.estimators.types import FitConfiguration, FitInitialGuess
from odmr_bench.models import Baseline, Resonance


def _normalization_values(
    frequency_reference_hz: float,
    frequency_half_span_hz: float,
    fluorescence_scale: float,
) -> tuple[float, float, float]:
    reference = float(frequency_reference_hz)
    half_span = float(frequency_half_span_hz)
    scale = float(fluorescence_scale)
    if not np.isfinite(reference):
        raise ValueError("frequency_reference_hz must be finite")
    if not np.isfinite(half_span) or half_span <= 0.0:
        raise ValueError("frequency_half_span_hz must be finite and positive")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("fluorescence_scale must be finite and positive")
    return reference, half_span, scale


def pack_parameters(
    guess: FitInitialGuess,
    configuration: FitConfiguration,
    *,
    frequency_reference_hz: float,
    frequency_half_span_hz: float,
    fluorescence_reference: float,
    fluorescence_scale: float,
) -> NDArray[np.float64]:
    """Pack one validated public guess into dimensionless optimizer coordinates."""
    if not isinstance(guess, FitInitialGuess):
        raise TypeError("guess must be a FitInitialGuess")
    reference, half_span, scale = _normalization_values(
        frequency_reference_hz, frequency_half_span_hz, fluorescence_scale
    )
    y_reference = float(fluorescence_reference)
    if not np.isfinite(y_reference):
        raise ValueError("fluorescence_reference must be finite")
    if guess.baseline.reference_hz != reference:
        raise ValueError("guess baseline reference must equal the sweep midpoint")

    packed = [
        (guess.baseline.intercept - y_reference) / scale,
        guess.baseline.slope_per_hz * half_span / scale,
    ]
    if configuration.baseline_degree == 2:
        packed.append(guess.baseline.quadratic_per_hz2 * half_span * half_span / scale)
    elif guess.baseline.quadratic_per_hz2 != 0.0:
        raise ValueError("linear baseline guesses require zero quadratic coefficient")

    for resonance in guess.resonances:
        packed.extend(
            (
                resonance.amplitude / scale,
                (resonance.center_hz - reference) / half_span,
                resonance.fwhm_hz / half_span,
            )
        )
    if configuration.model_kind == "pseudo_voigt":
        packed.extend(resonance.eta for resonance in guess.resonances)
    return np.asarray(packed, dtype=np.float64)


def unpack_parameters(
    packed: ArrayLike,
    configuration: FitConfiguration,
    *,
    frequency_reference_hz: float,
    frequency_half_span_hz: float,
    fluorescence_reference: float,
    fluorescence_scale: float,
) -> FitInitialGuess:
    """Convert optimizer coordinates into immutable public fit parameters."""
    reference, half_span, scale = _normalization_values(
        frequency_reference_hz, frequency_half_span_hz, fluorescence_scale
    )
    y_reference = float(fluorescence_reference)
    values = np.asarray(packed, dtype=np.float64)
    baseline_size = configuration.baseline_degree + 1
    expected_size = (
        baseline_size + 24 + (8 if configuration.model_kind == "pseudo_voigt" else 0)
    )
    if values.ndim != 1 or values.size != expected_size:
        raise ValueError(f"packed must be one-dimensional with length {expected_size}")
    if not np.all(np.isfinite(values)) or not np.isfinite(y_reference):
        raise ValueError("packed parameters and fluorescence_reference must be finite")

    baseline = Baseline(
        intercept=y_reference + scale * values[0],
        reference_hz=reference,
        slope_per_hz=scale * values[1] / half_span,
        quadratic_per_hz2=(
            scale * values[2] / (half_span * half_span)
            if configuration.baseline_degree == 2
            else 0.0
        ),
    )
    resonance_values = values[baseline_size : baseline_size + 24].reshape(8, 3)
    etas = (
        values[-8:]
        if configuration.model_kind == "pseudo_voigt"
        else np.ones(8, dtype=np.float64)
    )
    resonances = tuple(
        Resonance(
            resonance_id=resonance_id,
            amplitude=scale * row[0],
            center_hz=reference + half_span * row[1],
            fwhm_hz=half_span * row[2],
            eta=eta,
        )
        for resonance_id, row, eta in zip(
            configuration.resonance_ids, resonance_values, etas, strict=True
        )
    )
    return FitInitialGuess(resonances=resonances, baseline=baseline)


def public_parameter_transform(
    configuration: FitConfiguration,
    *,
    frequency_half_span_hz: float,
    fluorescence_scale: float,
) -> NDArray[np.float64]:
    """Return the packed-to-public diagonal differential transform."""
    _, half_span, scale = _normalization_values(
        0.0, frequency_half_span_hz, fluorescence_scale
    )
    factors = [scale, scale / half_span]
    if configuration.baseline_degree == 2:
        factors.append(scale / (half_span * half_span))
    factors.extend([scale, half_span, half_span] * 8)
    if configuration.model_kind == "pseudo_voigt":
        factors.extend([1.0] * 8)
    return np.diag(np.asarray(factors, dtype=np.float64))


def center_bounds_hz(
    initial_centers_hz: ArrayLike,
    frequency_min_hz: float,
    frequency_max_hz: float,
    min_center_separation_hz: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct non-overlapping center boxes around ordered initial centers."""
    centers = np.asarray(initial_centers_hz, dtype=np.float64)
    minimum = float(min_center_separation_hz)
    frequency_min = float(frequency_min_hz)
    frequency_max = float(frequency_max_hz)
    if centers.shape != (8,) or not np.all(np.isfinite(centers)):
        raise ValueError("initial centers must contain eight finite values")
    if not all(np.isfinite((frequency_min, frequency_max, minimum))):
        raise ValueError("center-bound inputs must be finite")
    if minimum <= 0.0 or frequency_max <= frequency_min:
        raise ValueError("center-bound interval and separation must be positive")
    if frequency_max - frequency_min < 7.0 * minimum:
        raise ValueError("sweep span is infeasible for seven required separations")
    if np.any(np.diff(centers) < minimum):
        raise ValueError("every initial center gap must meet the minimum separation")
    if np.any(centers < frequency_min) or np.any(centers > frequency_max):
        raise ValueError("every initial center must lie inside the sweep")

    lower = np.empty(8, dtype=np.float64)
    upper = np.empty(8, dtype=np.float64)
    lower[0] = frequency_min
    upper[-1] = frequency_max
    for index, (left, right) in enumerate(pairwise(centers)):
        midpoint = left / 2.0 + right / 2.0
        upper[index] = midpoint - minimum / 2.0
        lower[index + 1] = midpoint + minimum / 2.0
    if np.any(lower >= upper):
        raise ValueError("every initial center box must have positive width")
    if np.any(centers < lower) or np.any(centers > upper):
        raise ValueError("every initial center must lie inside its constrained box")
    return lower, upper


def parameter_bounds(
    guess: FitInitialGuess,
    configuration: FitConfiguration,
    *,
    frequency_min_hz: float,
    frequency_max_hz: float,
    frequency_reference_hz: float,
    frequency_half_span_hz: float,
    fluorescence_scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build explicit optimizer bounds in the same order as packed parameters."""
    reference, half_span, scale = _normalization_values(
        frequency_reference_hz, frequency_half_span_hz, fluorescence_scale
    )
    centers = np.asarray([item.center_hz for item in guess.resonances])
    center_lower, center_upper = center_bounds_hz(
        centers,
        frequency_min_hz,
        frequency_max_hz,
        configuration.min_center_separation_hz,
    )
    baseline_size = configuration.baseline_degree + 1
    lower = [-np.inf] * baseline_size
    upper = [np.inf] * baseline_size
    for low_hz, high_hz in zip(center_lower, center_upper, strict=True):
        lower.extend(
            [
                0.0,
                (low_hz - reference) / half_span,
                configuration.min_fwhm_hz / half_span,
            ]
        )
        upper.extend(
            [
                configuration.max_amplitude / scale,
                (high_hz - reference) / half_span,
                configuration.max_fwhm_hz / half_span,
            ]
        )
    if configuration.model_kind == "pseudo_voigt":
        lower.extend([0.0] * 8)
        upper.extend([1.0] * 8)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
