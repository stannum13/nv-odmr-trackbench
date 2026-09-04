"""Regression tests for deterministic baseline-aware initialization."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import peak_widths, savgol_filter

from odmr_bench.estimators import (
    CompleteSweep,
    FitConfiguration,
    initialize_spectrum,
)
from odmr_bench.estimators.initialization import (
    _baseline_from_scaled_polynomial,
    _fit_discovery_trend,
    _select_candidate_indices,
)
from odmr_bench.models import Baseline, Resonance, multi_resonance_spectrum

_CENTERS_HZ = np.array(
    [
        2.7583e9,
        2.7917e9,
        2.8231e9,
        2.8579e9,
        2.8912e9,
        2.9268e9,
        2.9634e9,
        3.0011e9,
    ]
)
_FWHM_HZ = np.array([1.6e6, 1.9e6, 2.2e6, 1.75e6, 2.4e6, 2.05e6, 1.85e6, 2.3e6])
_AMPLITUDES = np.array([0.018, 0.031, 0.023, 0.038, 0.027, 0.034, 0.021, 0.029])


def _spectrum_sweep(
    frequency_hz: np.ndarray,
    *,
    centers_hz: np.ndarray = _CENTERS_HZ,
    amplitudes: np.ndarray = _AMPLITUDES,
    baseline_degree: int = 1,
    slope_per_hz: float = 1.7e-10,
    quadratic_per_hz2: float | None = None,
) -> CompleteSweep:
    midpoint_hz = 0.5 * (frequency_hz[0] + frequency_hz[-1])
    quadratic = (
        1.3e-18 if baseline_degree == 2 else 0.0
    ) if quadratic_per_hz2 is None else quadratic_per_hz2
    baseline = Baseline(
        intercept=1.03,
        reference_hz=midpoint_hz,
        slope_per_hz=slope_per_hz,
        quadratic_per_hz2=quadratic,
    )
    resonances = tuple(
        Resonance(
            resonance_id=f"truth-{index}",
            center_hz=float(center_hz),
            fwhm_hz=float(fwhm_hz),
            amplitude=float(amplitude),
            eta=0.25 + 0.07 * index,
        )
        for index, (center_hz, fwhm_hz, amplitude) in enumerate(
            zip(centers_hz, _FWHM_HZ[: centers_hz.size], amplitudes, strict=True)
        )
    )
    fluorescence = multi_resonance_spectrum(frequency_hz, resonances, baseline)
    return CompleteSweep(frequency_hz, fluorescence)


def _generated_sweep(*, baseline_degree: int) -> tuple[CompleteSweep, np.ndarray]:
    frequency_hz = np.linspace(2.74e9, 3.02e9, 2801)
    return (
        _spectrum_sweep(frequency_hz, baseline_degree=baseline_degree),
        frequency_hz,
    )


def _configuration(*, baseline_degree: int) -> FitConfiguration:
    return FitConfiguration(
        baseline_degree=baseline_degree,
        min_fwhm_hz=5.0e5,
        max_fwhm_hz=5.0e6,
        max_amplitude=0.08,
        min_center_separation_hz=8.0e6,
        savgol_window=11,
        savgol_polyorder=2,
        relative_prominence=0.05,
    )


@pytest.mark.parametrize("baseline_degree", [1, 2])
def test_clean_generated_spectrum_is_initialized_deterministically(
    baseline_degree: int,
) -> None:
    sweep, frequency_hz = _generated_sweep(baseline_degree=baseline_degree)
    configuration = _configuration(baseline_degree=baseline_degree)

    first_guess, first_diagnostics = initialize_spectrum(sweep, configuration)
    second_guess, second_diagnostics = initialize_spectrum(sweep, configuration)

    assert first_guess == second_guess
    assert first_diagnostics == second_diagnostics
    assert first_guess is not None
    assert first_diagnostics.source == "detected"
    assert first_diagnostics.selected_indices == (
        183,
        517,
        831,
        1179,
        1512,
        1868,
        2234,
        2611,
    )
    assert not first_diagnostics.used_fallback
    grid_step_hz = float(np.median(np.diff(frequency_hz)))
    guessed_centers_hz = np.array(
        [resonance.center_hz for resonance in first_guess.resonances]
    )
    assert np.all(np.diff(guessed_centers_hz) > 0.0)
    assert np.all(np.abs(guessed_centers_hz - _CENTERS_HZ) <= 2.0 * grid_step_hz)
    for resonance in first_guess.resonances:
        assert configuration.min_fwhm_hz <= resonance.fwhm_hz
        assert resonance.fwhm_hz <= configuration.max_fwhm_hz
        assert 0.0 < resonance.amplitude <= configuration.max_amplitude
        assert resonance.eta == 0.5

    midpoint_hz = 0.5 * float(frequency_hz[0] + frequency_hz[-1])
    half_span_hz = 0.5 * float(frequency_hz[-1] - frequency_hz[0])
    z = (frequency_hz - midpoint_hz) / half_span_hz
    trend = _fit_discovery_trend(z, sweep.fluorescence, baseline_degree)
    assert trend is not None
    selected = np.asarray(first_diagnostics.selected_indices)
    expected_amplitudes = np.clip(
        trend.values[selected] - sweep.fluorescence[selected],
        0.0,
        configuration.max_amplitude,
    )
    assert_allclose(
        [item.amplitude for item in first_guess.resonances],
        expected_amplitudes,
        rtol=0.0,
        atol=1e-15,
    )

    discovery_depth = np.maximum(
        trend.values
        - savgol_filter(
            sweep.fluorescence,
            window_length=configuration.savgol_window,
            polyorder=configuration.savgol_polyorder,
        ),
        0.0,
    )
    _, _, left_positions, right_positions = peak_widths(
        discovery_depth, selected, rel_height=0.5
    )
    sample_positions = np.arange(frequency_hz.size, dtype=np.float64)
    expected_widths = np.clip(
        np.interp(right_positions, sample_positions, frequency_hz)
        - np.interp(left_positions, sample_positions, frequency_hz),
        configuration.min_fwhm_hz,
        configuration.max_fwhm_hz,
    )
    assert_allclose(
        [item.fwhm_hz for item in first_guess.resonances],
        expected_widths,
        rtol=0.0,
        atol=1e-9,
    )


def test_three_rejection_updates_then_final_fit_are_regression_pinned() -> None:
    z = np.linspace(-1.0, 1.0, 21)
    fluorescence = 1.1 + 0.03 * z + 0.015 * z**2
    for index, depth in {2: 0.02, 5: 0.08, 9: 0.035, 14: 0.06, 18: 0.025}.items():
        fluorescence[index] -= depth

    trend = _fit_discovery_trend(z, fluorescence, degree=2)

    assert trend is not None
    assert_allclose(trend.coefficients, [1.1, 0.03, 0.015], rtol=0.0, atol=2e-15)
    assert_array_equal(
        np.flatnonzero(trend.mask),
        [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 19, 20],
    )


def test_scaled_polynomial_is_converted_explicitly_to_public_hz_units() -> None:
    midpoint_hz = 2.88e9
    half_span_hz = 140.0e6

    baseline = _baseline_from_scaled_polynomial(
        np.array([1.1, 0.03, 0.015]), midpoint_hz, half_span_hz
    )

    assert baseline is not None
    assert baseline.intercept == 1.1
    assert baseline.reference_hz == midpoint_hz
    assert baseline.slope_per_hz == pytest.approx(0.03 / half_span_hz)
    assert baseline.quadratic_per_hz2 == pytest.approx(0.015 / half_span_hz**2)


@pytest.mark.parametrize(
    "coefficients",
    [
        np.array([1.0, 1.0e-300]),
        np.array([1.0, 0.0, 1.0e-200]),
    ],
)
def test_scaled_polynomial_rejects_nonzero_coefficients_that_underflow_publicly(
    coefficients: np.ndarray,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        baseline = _baseline_from_scaled_polynomial(
            coefficients, midpoint_hz=0.0, half_span_hz=1.0e100
        )

    assert baseline is None


def test_unrepresentable_public_baseline_conversion_is_structured_without_warning() -> (
    None
):
    frequency_hz = np.linspace(0.0, 1.0e-310, 101)
    sweep = CompleteSweep(frequency_hz, np.linspace(0.9, 1.1, frequency_hz.size))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        guess, diagnostics = initialize_spectrum(
            sweep, FitConfiguration(savgol_window=5)
        )

    assert guess is None
    assert diagnostics.source == "none"
    assert diagnostics.messages == (
        "baseline conversion to public units failed numerically",
    )


def test_nonuniform_grid_uses_physical_frequency_for_selection_and_widths() -> None:
    unit = np.linspace(0.0, 1.0, 4001)
    frequency_hz = 2.74e9 + 280.0e6 * (0.7 * unit + 0.3 * unit**2)
    sweep = _spectrum_sweep(
        frequency_hz,
        baseline_degree=2,
        slope_per_hz=4.8e-10,
        quadratic_per_hz2=2.0e-18,
    )
    configuration = _configuration(baseline_degree=2)

    guess, diagnostics = initialize_spectrum(sweep, configuration)

    assert guess is not None
    assert diagnostics.source == "detected"
    centers_hz = np.array([item.center_hz for item in guess.resonances])
    assert_allclose(
        centers_hz,
        _CENTERS_HZ,
        rtol=0.0,
        atol=2 * np.max(np.diff(frequency_hz)),
    )
    assert all(
        configuration.min_fwhm_hz <= item.fwhm_hz <= configuration.max_fwhm_hz
        for item in guess.resonances
    )
    midpoint_hz = frequency_hz[0] / 2.0 + frequency_hz[-1] / 2.0
    half_span_hz = frequency_hz[-1] / 2.0 - frequency_hz[0] / 2.0
    trend = _fit_discovery_trend(
        (frequency_hz - midpoint_hz) / half_span_hz,
        sweep.fluorescence,
        configuration.baseline_degree,
    )
    assert trend is not None
    depth = np.maximum(
        trend.values
        - savgol_filter(
            sweep.fluorescence,
            window_length=configuration.savgol_window,
            polyorder=configuration.savgol_polyorder,
        ),
        0.0,
    )
    selected = np.asarray(diagnostics.selected_indices)
    _, _, left_positions, right_positions = peak_widths(
        depth, selected, rel_height=0.5
    )
    sample_positions = np.arange(frequency_hz.size, dtype=np.float64)
    expected_widths_hz = np.clip(
        np.interp(right_positions, sample_positions, frequency_hz)
        - np.interp(left_positions, sample_positions, frequency_hz),
        configuration.min_fwhm_hz,
        configuration.max_fwhm_hz,
    )
    assert_allclose(
        [item.fwhm_hz for item in guess.resonances],
        expected_widths_hz,
        rtol=0.0,
        atol=1e-9,
    )


def test_steep_linear_baseline_does_not_displace_detected_centers() -> None:
    frequency_hz = np.linspace(2.74e9, 3.02e9, 2801)
    sweep = _spectrum_sweep(
        frequency_hz, baseline_degree=1, slope_per_hz=8.0e-10
    )

    guess, diagnostics = initialize_spectrum(
        sweep, _configuration(baseline_degree=1)
    )

    assert guess is not None
    assert diagnostics.source == "detected"
    assert_allclose(
        [item.center_hz for item in guess.resonances],
        _CENTERS_HZ,
        rtol=0.0,
        atol=2 * np.median(np.diff(frequency_hz)),
    )


def test_lorentzian_configuration_initializes_eta_to_one() -> None:
    sweep, _ = _generated_sweep(baseline_degree=1)
    configuration = FitConfiguration(
        model_kind="lorentzian",
        baseline_degree=1,
        min_fwhm_hz=5.0e5,
        max_fwhm_hz=5.0e6,
        max_amplitude=0.08,
        min_center_separation_hz=8.0e6,
        relative_prominence=0.05,
    )

    guess, _ = initialize_spectrum(sweep, configuration)

    assert guess is not None
    assert all(item.eta == 1.0 for item in guess.resonances)


def test_flat_sweep_reports_zero_candidates_without_implicit_fallback() -> None:
    frequency_hz = np.linspace(2.74e9, 3.02e9, 2801)
    sweep = CompleteSweep(frequency_hz, np.ones(frequency_hz.size))

    guess, diagnostics = initialize_spectrum(
        sweep, _configuration(baseline_degree=1)
    )

    assert guess is None
    assert diagnostics.source == "none"
    assert diagnostics.candidate_count == 0
    assert not diagnostics.used_fallback


@pytest.mark.parametrize(
    ("baseline_degree", "linear_coefficient", "quadratic_coefficient"),
    [(1, 1.0e-3, 0.0), (2, 1.0e-8, 1.0e-8)],
)
def test_exact_polynomial_baseline_roundoff_does_not_fabricate_candidates(
    baseline_degree: int,
    linear_coefficient: float,
    quadratic_coefficient: float,
) -> None:
    frequency_hz = np.linspace(2.74e9, 3.02e9, 101)
    midpoint_hz = frequency_hz[0] / 2.0 + frequency_hz[-1] / 2.0
    half_span_hz = frequency_hz[-1] / 2.0 - frequency_hz[0] / 2.0
    z = (frequency_hz - midpoint_hz) / half_span_hz
    fluorescence = (
        0.1 + linear_coefficient * z + quadratic_coefficient * z**2
    )

    guess, diagnostics = initialize_spectrum(
        CompleteSweep(frequency_hz, fluorescence),
        FitConfiguration(
            baseline_degree=baseline_degree,
            min_center_separation_hz=1.0,
            relative_prominence=0.01,
            allow_fallback=False,
        ),
    )

    assert guess is None
    assert diagnostics.source == "none"
    assert diagnostics.candidate_count == 0
    assert diagnostics.messages == ("discovery depth is at numerical floor",)


def test_dips_above_numerical_floor_remain_discoverable() -> None:
    frequency_hz = np.linspace(2.74e9, 3.02e9, 2801)
    amplitudes = 1.0e-10 * (1.0 + 0.05 * np.arange(8))
    sweep = _spectrum_sweep(frequency_hz, amplitudes=amplitudes)
    configuration = FitConfiguration(
        min_fwhm_hz=5.0e5,
        max_fwhm_hz=5.0e6,
        max_amplitude=1.0e-8,
        min_resolved_amplitude=1.0e-12,
        min_center_separation_hz=8.0e6,
        relative_prominence=0.05,
    )

    guess, diagnostics = initialize_spectrum(sweep, configuration)

    assert guess is not None
    assert diagnostics.source == "detected"
    assert diagnostics.candidate_count == 8


@pytest.mark.parametrize(
    ("frequency_hz", "baseline_degree"),
    [
        pytest.param(
            np.linspace(1.0, 1.1, 11) * 1.0e308, degree, id=f"same-sign-d{degree}"
        )
        for degree in (1, 2)
    ]
    + [
        pytest.param(
            np.linspace(-1.0, 1.0, 11) * 1.0e308,
            degree,
            id=f"opposite-sign-d{degree}",
        )
        for degree in (1, 2)
    ],
)
def test_extreme_finite_frequency_endpoints_return_without_numerical_warnings(
    frequency_hz: np.ndarray,
    baseline_degree: int,
) -> None:
    fluorescence = np.linspace(0.9, 1.1, frequency_hz.size)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        guess, diagnostics = initialize_spectrum(
            CompleteSweep(frequency_hz, fluorescence),
            FitConfiguration(savgol_window=5, baseline_degree=baseline_degree),
        )

    assert guess is None
    assert diagnostics.source == "none"
    assert diagnostics.candidate_count == 0
    expected_message = (
        "baseline conversion to public units failed numerically"
        if baseline_degree == 2
        else "discovery depth is at numerical floor"
    )
    assert diagnostics.messages == (expected_message,)


def test_polynomial_backend_failure_becomes_stable_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_polynomial_fit(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise np.linalg.LinAlgError("forced numerical failure")

    monkeypatch.setattr(np.polynomial.polynomial, "polyfit", fail_polynomial_fit)
    frequency_hz = np.linspace(2.74e9, 3.02e9, 101)

    guess, diagnostics = initialize_spectrum(
        CompleteSweep(frequency_hz, np.linspace(0.9, 1.1, frequency_hz.size)),
        FitConfiguration(),
    )

    assert guess is None
    assert diagnostics.source == "none"
    assert diagnostics.candidate_count == 0
    assert diagnostics.messages == ("baseline trend fit failed numerically",)


def test_seven_dips_report_candidate_scarcity_without_implicit_fallback() -> None:
    frequency_hz = np.linspace(2.74e9, 3.02e9, 2801)
    sweep = _spectrum_sweep(
        frequency_hz,
        centers_hz=_CENTERS_HZ[:7],
        amplitudes=_AMPLITUDES[:7],
    )

    guess, diagnostics = initialize_spectrum(
        sweep, _configuration(baseline_degree=1)
    )

    assert guess is None
    assert diagnostics.source == "none"
    assert diagnostics.candidate_count == 7
    assert not diagnostics.used_fallback


def test_edge_dip_is_not_fabricated_as_an_interior_candidate() -> None:
    frequency_hz = np.linspace(2.74e9, 3.02e9, 2801)
    centers_hz = np.concatenate(([frequency_hz[0]], _CENTERS_HZ[1:]))
    sweep = _spectrum_sweep(
        frequency_hz,
        centers_hz=centers_hz,
        amplitudes=_AMPLITUDES,
    )

    guess, diagnostics = initialize_spectrum(
        sweep, _configuration(baseline_degree=1)
    )

    assert guess is None
    assert diagnostics.candidate_count == 7
    assert diagnostics.selected_indices == ()


def test_prominence_ranking_wins_for_close_candidates_with_different_heights() -> (
    None
):
    depth = np.zeros(25)
    depth[[2, 3, 4, 5, 20]] = [10.0, 8.0, 8.0, 9.0, 7.0]
    frequency_hz = np.concatenate(
        (
            np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0]),
            np.linspace(102.01, 102.15, 15),
            np.array([102.16, 200.0, 201.0, 202.0]),
        )
    )

    selected, candidate_count = _select_candidate_indices(
        depth, frequency_hz, relative_prominence=0.05, min_separation_hz=0.2, limit=3
    )

    assert candidate_count == 3
    assert 20 in selected
    assert 5 not in selected


@pytest.mark.parametrize("dip_count", [0, 7])
def test_explicit_fallback_uses_pinned_geometry_and_scales(dip_count: int) -> None:
    frequency_hz = np.linspace(2.74e9, 3.02e9, 2801)
    if dip_count:
        sweep = _spectrum_sweep(
            frequency_hz,
            centers_hz=_CENTERS_HZ[:dip_count],
            amplitudes=_AMPLITUDES[:dip_count],
        )
    else:
        sweep = CompleteSweep(frequency_hz, np.ones(frequency_hz.size))
    configuration = FitConfiguration(
        baseline_degree=1,
        min_fwhm_hz=5.0e5,
        max_fwhm_hz=5.0e6,
        max_amplitude=0.08,
        min_resolved_amplitude=2.0e-4,
        min_center_separation_hz=8.0e6,
        relative_prominence=0.05,
        allow_fallback=True,
    )

    guess, diagnostics = initialize_spectrum(sweep, configuration)

    assert guess is not None
    assert diagnostics.source == "fallback"
    assert diagnostics.used_fallback
    assert diagnostics.candidate_count == dip_count
    centers_hz = np.array([item.center_hz for item in guess.resonances])
    median_grid_step = float(np.median(np.diff(frequency_hz)))
    edge_margin = max(configuration.min_center_separation_hz / 2, median_grid_step)
    assert_allclose(
        centers_hz,
        np.linspace(frequency_hz[0] + edge_margin, frequency_hz[-1] - edge_margin, 8),
    )
    assert np.all(centers_hz > frequency_hz[0])
    assert np.all(centers_hz < frequency_hz[-1])
    assert np.all(np.diff(centers_hz) >= configuration.min_center_separation_hz)
    expected_width = np.clip(
        4 * median_grid_step,
        configuration.min_fwhm_hz,
        configuration.max_fwhm_hz,
    )
    assert_allclose([item.fwhm_hz for item in guess.resonances], expected_width)
    if dip_count == 0:
        maximum_depth = 0.0
    else:
        midpoint_hz = 0.5 * float(frequency_hz[0] + frequency_hz[-1])
        half_span_hz = 0.5 * float(frequency_hz[-1] - frequency_hz[0])
        trend = _fit_discovery_trend(
            (frequency_hz - midpoint_hz) / half_span_hz,
            sweep.fluorescence,
            configuration.baseline_degree,
        )
        assert trend is not None
        maximum_depth = float(
            np.max(
                np.maximum(
                    trend.values
                    - savgol_filter(
                        sweep.fluorescence,
                        window_length=configuration.savgol_window,
                        polyorder=configuration.savgol_polyorder,
                    ),
                    0.0,
                )
            )
        )
    expected_amplitude = min(
        configuration.max_amplitude,
        max(2 * configuration.min_resolved_amplitude, maximum_depth / 8),
    )
    assert_allclose(
        [item.amplitude for item in guess.resonances], expected_amplitude
    )


def test_fallback_strict_geometry_boundary_returns_stable_failure() -> None:
    frequency_hz = np.linspace(0.0, 8.0e6, 101)
    configuration = FitConfiguration(
        min_center_separation_hz=1.0e6,
        savgol_window=11,
        allow_fallback=True,
    )

    guess, diagnostics = initialize_spectrum(
        CompleteSweep(frequency_hz, np.ones(frequency_hz.size)), configuration
    )

    assert guess is None
    assert diagnostics.source == "none"
    assert diagnostics.messages == ("fallback center geometry is infeasible",)


def test_smoothing_window_larger_than_sweep_is_a_structured_failure() -> None:
    frequency_hz = np.linspace(2.74e9, 3.02e9, 9)

    guess, diagnostics = initialize_spectrum(
        CompleteSweep(frequency_hz, np.linspace(0.9, 1.1, frequency_hz.size)),
        FitConfiguration(savgol_window=11, allow_fallback=True),
    )

    assert guess is None
    assert diagnostics.source == "none"
    assert diagnostics.candidate_count == 0
    assert diagnostics.messages == ("savgol_window exceeds sweep sample count",)


def test_numerically_too_few_inliers_returns_failure_without_warning() -> None:
    frequency_hz = np.linspace(2.74e9, 3.02e9, 101)
    fluorescence = np.full(frequency_hz.size, 1.0e308)
    fluorescence[-1] = -1.0e308

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        guess, diagnostics = initialize_spectrum(
            CompleteSweep(frequency_hz, fluorescence),
            FitConfiguration(savgol_window=11),
        )

    assert guess is None
    assert diagnostics.source == "none"
    assert diagnostics.messages == ("baseline rejection left too few samples",)
