"""Deterministic, baseline-aware initialization for complete ODMR sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_widths, savgol_filter

from odmr_bench.estimators.types import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    InitializationDiagnostics,
)
from odmr_bench.models import Baseline, Resonance

_DISCOVERY_ROUNDOFF_FACTOR = 64.0


@dataclass(frozen=True, slots=True)
class _TrendFit:
    coefficients: NDArray[np.float64]
    mask: NDArray[np.bool_]
    values: NDArray[np.float64]


def _fit_discovery_trend(
    z: NDArray[np.float64],
    fluorescence: NDArray[np.float64],
    degree: int,
) -> _TrendFit | None:
    """Fit the specified baseline polynomial after three rejection updates."""
    mask = np.ones(z.size, dtype=np.bool_)
    minimum_inliers = max(degree + 1, ceil(0.25 * z.size))

    with np.errstate(all="ignore"):
        for _ in range(3):
            coefficients = np.polynomial.polynomial.polyfit(
                z[mask], fluorescence[mask], degree
            )
            trend = np.polynomial.polynomial.polyval(z, coefficients)
            residual = fluorescence - trend
            median = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median)))
            if mad == 0.0:
                mask = residual >= median
            else:
                sigma = 1.4826 * mad
                mask = residual >= median - 2.5 * sigma
            if np.count_nonzero(mask) < minimum_inliers:
                return None

        coefficients = np.asarray(
            np.polynomial.polynomial.polyfit(z[mask], fluorescence[mask], degree),
            dtype=np.float64,
        )
        trend = np.asarray(
            np.polynomial.polynomial.polyval(z, coefficients), dtype=np.float64
        )
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(trend)):
        return None
    return _TrendFit(coefficients=coefficients, mask=mask, values=trend)


def _diagnostic_failure(
    candidate_count: int, message: str
) -> InitializationDiagnostics:
    return InitializationDiagnostics(
        source="none",
        candidate_count=candidate_count,
        selected_indices=(),
        used_fallback=False,
        messages=(message,),
    )


def _baseline_from_scaled_polynomial(
    coefficients: NDArray[np.float64], midpoint_hz: float, half_span_hz: float
) -> Baseline:
    quadratic = (
        (float(coefficients[2]) / half_span_hz) / half_span_hz
        if coefficients.size == 3
        else 0.0
    )
    return Baseline(
        intercept=float(coefficients[0]),
        reference_hz=midpoint_hz,
        slope_per_hz=float(coefficients[1]) / half_span_hz,
        quadratic_per_hz2=quadratic,
    )


def _select_candidate_indices(
    depth: NDArray[np.float64],
    frequency_hz: NDArray[np.float64],
    *,
    relative_prominence: float,
    min_separation_hz: float,
    limit: int = 8,
) -> tuple[tuple[int, ...], int]:
    """Select prominent peaks with separation measured on the physical grid."""
    peak_indices, properties = find_peaks(
        depth,
        prominence=relative_prominence * float(np.max(depth)),
    )
    prominences = np.asarray(properties["prominences"], dtype=np.float64)
    ranked = sorted(
        zip(peak_indices.tolist(), prominences.tolist(), strict=True),
        key=lambda candidate: (-candidate[1], candidate[0]),
    )
    selected_indices: list[int] = []
    for sample_index, _ in ranked:
        candidate_hz = float(frequency_hz[sample_index])
        if all(
            abs(candidate_hz - float(frequency_hz[accepted_index]))
            >= min_separation_hz
            for accepted_index in selected_indices
        ):
            selected_indices.append(sample_index)
            if len(selected_indices) == limit:
                break
    selected_indices.sort(key=lambda index: float(frequency_hz[index]))
    return tuple(selected_indices), int(peak_indices.size)


def _fallback_guess(
    frequency_hz: NDArray[np.float64],
    configuration: FitConfiguration,
    baseline: Baseline,
    maximum_depth: float,
) -> FitInitialGuess | None:
    median_grid_step_hz = float(np.median(np.diff(frequency_hz)))
    edge_margin_hz = max(
        configuration.min_center_separation_hz / 2.0, median_grid_step_hz
    )
    midpoint_hz = float(frequency_hz[0]) / 2.0 + float(frequency_hz[-1]) / 2.0
    half_span_hz = float(frequency_hz[-1]) / 2.0 - float(frequency_hz[0]) / 2.0
    usable_half_span_hz = half_span_hz - edge_margin_hz
    if usable_half_span_hz <= 3.5 * configuration.min_center_separation_hz:
        return None

    centers_hz = midpoint_hz + usable_half_span_hz * np.linspace(
        -1.0, 1.0, 8
    )
    width_hz = float(
        np.clip(
            4.0 * median_grid_step_hz,
            configuration.min_fwhm_hz,
            configuration.max_fwhm_hz,
        )
    )
    amplitude = min(
        configuration.max_amplitude,
        max(2.0 * configuration.min_resolved_amplitude, maximum_depth / 8.0),
    )
    eta = 1.0 if configuration.model_kind == "lorentzian" else 0.5
    return FitInitialGuess(
        resonances=tuple(
            Resonance(
                resonance_id=resonance_id,
                center_hz=float(center_hz),
                fwhm_hz=width_hz,
                amplitude=amplitude,
                eta=eta,
            )
            for resonance_id, center_hz in zip(
                configuration.resonance_ids, centers_hz, strict=True
            )
        ),
        baseline=baseline,
    )


def _scarcity_result(
    frequency_hz: NDArray[np.float64],
    configuration: FitConfiguration,
    baseline: Baseline,
    maximum_depth: float,
    candidate_count: int,
    message: str,
) -> tuple[FitInitialGuess | None, InitializationDiagnostics]:
    if configuration.allow_fallback:
        guess = _fallback_guess(
            frequency_hz, configuration, baseline, maximum_depth
        )
        if guess is not None:
            return guess, InitializationDiagnostics(
                source="fallback",
                candidate_count=candidate_count,
                selected_indices=(),
                used_fallback=True,
                messages=(message,),
            )
        message = "fallback center geometry is infeasible"
    return None, _diagnostic_failure(candidate_count, message)


def _discovery_numerical_floor(
    fluorescence: NDArray[np.float64], trend: NDArray[np.float64]
) -> float:
    """Bound accumulated low-order fit and smoothing roundoff at signal scale."""
    scale = max(
        float(np.max(np.abs(fluorescence))),
        float(np.max(np.abs(trend))),
    )
    return _DISCOVERY_ROUNDOFF_FACTOR * np.finfo(np.float64).eps * scale


def initialize_spectrum(
    sweep: CompleteSweep, configuration: FitConfiguration
) -> tuple[FitInitialGuess | None, InitializationDiagnostics]:
    """Build a deterministic eight-line guess from one complete public sweep."""
    frequency_hz = sweep.frequency_hz
    fluorescence = sweep.fluorescence
    if configuration.savgol_window > frequency_hz.size:
        return None, _diagnostic_failure(
            0, "savgol_window exceeds sweep sample count"
        )

    midpoint_hz = float(frequency_hz[0]) / 2.0 + float(frequency_hz[-1]) / 2.0
    half_span_hz = float(frequency_hz[-1]) / 2.0 - float(frequency_hz[0]) / 2.0
    if (
        not np.isfinite(midpoint_hz)
        or not np.isfinite(half_span_hz)
        or half_span_hz <= 0
    ):
        return None, _diagnostic_failure(
            0, "frequency normalization failed numerically"
        )
    z = np.asarray((frequency_hz - midpoint_hz) / half_span_hz, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        return None, _diagnostic_failure(
            0, "frequency normalization failed numerically"
        )
    try:
        trend_fit = _fit_discovery_trend(
            z, fluorescence, configuration.baseline_degree
        )
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
        return None, _diagnostic_failure(0, "baseline trend fit failed numerically")
    if trend_fit is None:
        return None, _diagnostic_failure(
            0, "baseline rejection left too few samples"
        )

    baseline = _baseline_from_scaled_polynomial(
        trend_fit.coefficients, midpoint_hz, half_span_hz
    )
    smoothed = savgol_filter(
        fluorescence,
        window_length=configuration.savgol_window,
        polyorder=configuration.savgol_polyorder,
    )
    depth = np.maximum(trend_fit.values - smoothed, 0.0)
    if np.all(fluorescence == fluorescence[0]):
        depth = np.zeros_like(depth)
    maximum_depth = float(np.max(depth))
    if not np.isfinite(maximum_depth):
        return _scarcity_result(
            frequency_hz,
            configuration,
            baseline,
            maximum_depth if np.isfinite(maximum_depth) else 0.0,
            0,
            "discovery depth is zero or non-finite",
        )
    numerical_floor = _discovery_numerical_floor(
        fluorescence, trend_fit.values
    )
    if maximum_depth <= numerical_floor:
        return _scarcity_result(
            frequency_hz,
            configuration,
            baseline,
            maximum_depth,
            0,
            "discovery depth is at numerical floor",
        )

    selected_indices, candidate_count = _select_candidate_indices(
        depth,
        frequency_hz,
        relative_prominence=configuration.relative_prominence,
        min_separation_hz=configuration.min_center_separation_hz,
    )
    if len(selected_indices) < 8:
        return _scarcity_result(
            frequency_hz,
            configuration,
            baseline,
            maximum_depth,
            candidate_count,
            "fewer than eight separated dip candidates",
        )

    _, _, left_positions, right_positions = peak_widths(
        depth, np.asarray(selected_indices), rel_height=0.5
    )
    sample_positions = np.arange(frequency_hz.size, dtype=np.float64)
    left_hz = np.interp(left_positions, sample_positions, frequency_hz)
    right_hz = np.interp(right_positions, sample_positions, frequency_hz)
    width_hz = np.clip(
        right_hz - left_hz,
        configuration.min_fwhm_hz,
        configuration.max_fwhm_hz,
    )
    eta = 1.0 if configuration.model_kind == "lorentzian" else 0.5
    resonances = tuple(
        Resonance(
            resonance_id=resonance_id,
            center_hz=float(frequency_hz[sample_index]),
            fwhm_hz=float(candidate_width_hz),
            amplitude=float(
                min(
                    configuration.max_amplitude,
                    max(
                        trend_fit.values[sample_index] - fluorescence[sample_index],
                        0.0,
                    ),
                )
            ),
            eta=eta,
        )
        for resonance_id, sample_index, candidate_width_hz in zip(
            configuration.resonance_ids,
            selected_indices,
            width_hz,
            strict=True,
        )
    )
    guess = FitInitialGuess(
        resonances=resonances,
        baseline=baseline,
    )
    diagnostics = InitializationDiagnostics(
        source="detected",
        candidate_count=candidate_count,
        selected_indices=tuple(selected_indices),
        used_fallback=False,
        messages=(),
    )
    return guess, diagnostics
