"""Scientific regressions for the constrained eight-resonance oracle fitter."""

from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from odmr_bench.estimators import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    fit_spectrum,
    linearized_standard_errors,
)
from odmr_bench.estimators.fitting import _scaled_residual_function
from odmr_bench.estimators.parameterization import pack_parameters
from odmr_bench.models import Baseline, Resonance, multi_resonance_spectrum

FREQUENCY_HZ = np.linspace(2.740e9, 3.020e9, 4481)
CENTERS_HZ = 1e9 * np.array([2.760, 2.794, 2.828, 2.862, 2.896, 2.930, 2.964, 2.998])
FWHM_HZ = 1e6 * np.array([1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20])
AMPLITUDES = np.array([0.018, 0.021, 0.024, 0.027, 0.030, 0.033, 0.036, 0.039])
ETAS = np.array([0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.93])
REFERENCE_HZ = 2.880e9


def _configuration(
    model_kind: str = "pseudo_voigt",
    baseline_degree: int = 2,
    **changes: object,
) -> FitConfiguration:
    values: dict[str, object] = {
        "model_kind": model_kind,
        "baseline_degree": baseline_degree,
        "min_fwhm_hz": 2.0e5,
        "max_fwhm_hz": 8.0e6,
        "max_amplitude": 0.08,
        "min_resolved_amplitude": 1.0e-4,
        "min_center_separation_hz": 1.0e6,
        "relative_prominence": 0.01,
        "max_nfev": 4000,
        "rank_rtol": 1.0e-10,
        "min_baseline_sse_improvement": 1.0e-4,
    }
    values.update(changes)
    return FitConfiguration(**values)


def _truth(
    model_kind: str = "pseudo_voigt",
    baseline_degree: int = 2,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
) -> tuple[tuple[Resonance, ...], Baseline]:
    resonances = tuple(
        Resonance(
            resonance_id=f"r{index}",
            center_hz=center,
            fwhm_hz=width,
            amplitude=scale * amplitude,
            eta=1.0 if model_kind == "lorentzian" else eta,
        )
        for index, (center, width, amplitude, eta) in enumerate(
            zip(CENTERS_HZ, FWHM_HZ, AMPLITUDES, ETAS, strict=True)
        )
    )
    baseline = Baseline(
        intercept=scale * 1.0 + offset,
        reference_hz=REFERENCE_HZ,
        slope_per_hz=scale * 2.0e-11,
        quadratic_per_hz2=(scale * -5.0e-20 if baseline_degree == 2 else 0.0),
    )
    return resonances, baseline


def _guess(
    model_kind: str = "pseudo_voigt",
    baseline_degree: int = 2,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
) -> FitInitialGuess:
    resonances, baseline = _truth(
        model_kind, baseline_degree, scale=scale, offset=offset
    )
    center_offsets = 1e5 * np.array([-2, 1, -1, 2, -2, 1, -1, 2])
    return FitInitialGuess(
        resonances=tuple(
            Resonance(
                resonance_id=item.resonance_id,
                center_hz=item.center_hz + center_offset,
                fwhm_hz=item.fwhm_hz * 1.08,
                amplitude=item.amplitude * 0.92,
                eta=(1.0 if model_kind == "lorentzian" else min(1.0, item.eta + 0.04)),
            )
            for item, center_offset in zip(resonances, center_offsets, strict=True)
        ),
        baseline=Baseline(
            intercept=baseline.intercept,
            reference_hz=baseline.reference_hz,
            slope_per_hz=baseline.slope_per_hz,
            quadratic_per_hz2=baseline.quadratic_per_hz2,
        ),
    )


def _sweep(
    model_kind: str = "pseudo_voigt",
    baseline_degree: int = 2,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    noise: np.ndarray | None = None,
) -> CompleteSweep:
    resonances, baseline = _truth(
        model_kind, baseline_degree, scale=scale, offset=offset
    )
    return CompleteSweep(
        FREQUENCY_HZ,
        multi_resonance_spectrum(
            FREQUENCY_HZ, resonances, baseline, additive_noise=noise
        ),
    )


def _assert_recovery(result: object, model_kind: str, degree: int) -> None:
    assert result.success
    fitted = result.resonance_estimates
    assert_array_equal(
        [item.resonance_id for item in fitted], [f"r{i}" for i in range(8)]
    )
    centers = np.array([item.center_hz for item in fitted])
    widths = np.array([item.fwhm_hz for item in fitted])
    amplitudes = np.array([item.amplitude for item in fitted])
    assert np.all(np.abs(centers - CENTERS_HZ) < 0.01 * FWHM_HZ)
    assert_allclose(widths, FWHM_HZ, rtol=0.05, atol=0.0)
    assert_allclose(amplitudes, AMPLITUDES, rtol=0.05, atol=0.0)
    if model_kind == "lorentzian":
        assert_array_equal([item.eta for item in fitted], np.ones(8))
    else:
        assert np.all(np.abs(np.array([item.eta for item in fitted]) - ETAS) < 0.08)
    assert_allclose(result.q_values, centers / widths, rtol=0.0, atol=0.0)
    assert result.baseline_estimate.reference_hz == REFERENCE_HZ
    assert result.baseline_estimate.intercept == pytest.approx(1.0, abs=2e-4)
    assert result.baseline_estimate.slope_per_hz == pytest.approx(2e-11, abs=5e-13)
    expected_quadratic = -5e-20 if degree == 2 else 0.0
    assert result.baseline_estimate.quadratic_per_hz2 == pytest.approx(
        expected_quadratic, abs=5e-21
    )


def test_linearized_standard_errors_use_one_svd_and_public_transform() -> None:
    jacobian = np.diag([4.0, 2.0, 1.0])
    transform = np.diag([2.0, 8.0, 32.0])

    errors, rank, reason = linearized_standard_errors(
        jacobian,
        scaled_cost=6.0,
        degrees_of_freedom=3,
        public_transform=transform,
        rank_rtol=1e-12,
    )

    expected_covariance = (
        transform @ np.diag(4.0 / np.array([16.0, 4.0, 1.0])) @ transform
    )
    assert rank == 3
    assert reason is None
    assert_allclose(errors, np.sqrt(np.diag(expected_covariance)), rtol=1e-14)


@pytest.mark.parametrize(
    ("jacobian", "cost", "dof", "transform", "expected_rank"),
    [
        (np.diag([1.0, 0.0]), 1.0, 2, np.eye(2), 1),
        (np.array([[1.0, np.nan]]), 1.0, 2, np.eye(2), None),
        (np.eye(2), 1.0, 0, np.eye(2), 2),
        (np.eye(2), np.nan, 2, np.eye(2), 2),
        (np.eye(2), 1.0, 2, np.eye(3), None),
        (np.eye(2), 1.0, 2, np.diag([1.0, np.inf]), None),
    ],
)
def test_linearized_standard_error_failures_are_explicit(
    jacobian: np.ndarray,
    cost: float,
    dof: int,
    transform: np.ndarray,
    expected_rank: int | None,
) -> None:
    errors, rank, reason = linearized_standard_errors(
        jacobian, cost, dof, transform, 1e-10
    )
    assert errors is None
    assert rank == expected_rank
    assert reason


@pytest.mark.parametrize(
    ("model_kind", "degree"),
    [
        ("lorentzian", 1),
        ("lorentzian", 2),
        ("pseudo_voigt", 1),
        ("pseudo_voigt", 2),
    ],
)
def test_clean_explicit_guess_recovers_physical_parameters(
    model_kind: str, degree: int
) -> None:
    result = fit_spectrum(
        _sweep(model_kind, degree),
        _configuration(model_kind, degree),
        _guess(model_kind, degree),
    )

    _assert_recovery(result, model_kind, degree)
    assert result.uncertainty is not None
    assert result.uncertainty.method == "local_linearized_jacobian_covariance"
    assert result.uncertainty.baseline_standard_errors.shape == (degree + 1,)
    assert result.uncertainty.center_hz.shape == (8,)
    assert result.uncertainty.fwhm_hz.shape == (8,)
    assert result.uncertainty.amplitude.shape == (8,)
    assert (result.uncertainty.eta is None) == (model_kind == "lorentzian")
    assert np.all(np.isfinite(result.uncertainty.center_hz))


def test_optimizer_failure_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_least_squares(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            success=False,
            status=0,
            message="forced",
            nfev=3,
            x=np.zeros(35),
            fun=np.ones(FREQUENCY_HZ.size),
            cost=1.0,
            jac=np.ones((FREQUENCY_HZ.size, 35)),
        )

    monkeypatch.setattr(
        "odmr_bench.estimators.fitting.least_squares", failed_least_squares
    )
    result = fit_spectrum(_sweep(), _configuration(), _guess())

    assert not result.success
    assert result.failure_code == "optimization_failed"
    assert result.scipy_status == 0
    assert result.scipy_message == "forced"
    assert result.nfev == 3
    assert result.initial_guess == _guess()
    assert result.resonance_estimates == ()
    assert result.baseline_estimate is None


def test_nonfinite_public_parameters_from_successful_optimizer_fail_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overflowing_least_squares(
        fun: object, x0: np.ndarray, **kwargs: object
    ) -> SimpleNamespace:
        del fun, kwargs
        fitted = x0.copy()
        fitted[0] = np.finfo(np.float64).max
        jacobian = np.zeros((FREQUENCY_HZ.size, x0.size))
        jacobian[: x0.size] = np.eye(x0.size)
        return SimpleNamespace(
            success=True,
            status=1,
            message="forced finite packed overflow",
            nfev=1,
            x=fitted,
            fun=np.zeros(FREQUENCY_HZ.size),
            cost=0.0,
            jac=jacobian,
        )

    monkeypatch.setattr(
        "odmr_bench.estimators.fitting.least_squares", overflowing_least_squares
    )

    scale = 1.0e308
    result = fit_spectrum(
        _sweep(scale=scale),
        _configuration(
            max_amplitude=0.08 * scale,
            min_resolved_amplitude=1.0e-4 * scale,
        ),
        _guess(scale=scale),
    )

    assert not result.success
    assert result.failure_code == "quality_failed"
    assert result.resonance_estimates == ()
    assert result.baseline_estimate is None


@pytest.mark.parametrize(
    ("model_kind", "degree"),
    [(model, degree) for model in ("lorentzian", "pseudo_voigt") for degree in (1, 2)],
)
@pytest.mark.parametrize("sample_offset", [-1, 0, 1])
def test_sample_count_preflight_precedes_initialization(
    model_kind: str, degree: int, sample_offset: int
) -> None:
    configuration = _configuration(model_kind, degree)
    free_parameters = degree + 1 + 8 * (4 if model_kind == "pseudo_voigt" else 3)
    sample_count = free_parameters + sample_offset
    sweep = CompleteSweep(
        np.linspace(0.0, 100.0e6, sample_count),
        np.linspace(0.9, 1.1, sample_count),
    )

    result = fit_spectrum(sweep, configuration)

    if sample_offset <= 0:
        assert result.failure_code == "insufficient_samples"
        assert result.degrees_of_freedom == sample_offset
        assert result.initial_guess is None
        assert result.residual_scale is None
        assert result.scipy_status is None
        assert result.nfev == 0
    else:
        assert result.failure_code == "initialization_failed"


@pytest.mark.parametrize("mode", ["auto", "fallback", "user"])
def test_constant_sweep_is_uninformative_before_any_guess(mode: str) -> None:
    configuration = _configuration(allow_fallback=mode == "fallback")
    supplied = _guess() if mode == "user" else None
    result = fit_spectrum(
        CompleteSweep(FREQUENCY_HZ, np.ones(FREQUENCY_HZ.size)),
        configuration,
        supplied,
    )

    assert result.failure_code == "uninformative_sweep"
    assert result.diagnostics.source == "none"
    assert result.initial_guess is None
    assert result.residual_scale is None
    assert result.cost is None
    assert result.residual_rmse is None
    assert result.scipy_status is None
    assert result.scipy_message is None
    assert result.nfev == 0
    assert result.jacobian_rank is None


def test_nonfinite_variation_scale_is_uninformative_without_warning() -> None:
    fluorescence = np.concatenate(
        (
            np.linspace(-1.0e308, -1.0e307, 2240),
            np.linspace(1.0e307, 1.0e308, 2241),
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = fit_spectrum(
            CompleteSweep(FREQUENCY_HZ, fluorescence), _configuration()
        )

    assert result.failure_code == "uninformative_sweep"
    assert result.initial_guess is None
    assert result.residual_scale is None


def test_clean_sweep_auto_initialization_recovers_without_truth_input() -> None:
    result = fit_spectrum(_sweep(), _configuration())

    _assert_recovery(result, "pseudo_voigt", 2)
    assert result.diagnostics.source == "detected"


def test_fixed_seed_low_noise_auto_fit_meets_declared_regression_bounds() -> None:
    noise = np.random.default_rng(6104).normal(0.0, 2.0e-4, FREQUENCY_HZ.size)
    result = fit_spectrum(_sweep(noise=noise), _configuration())

    assert result.success
    fitted = result.resonance_estimates
    centers = np.array([item.center_hz for item in fitted])
    widths = np.array([item.fwhm_hz for item in fitted])
    amplitudes = np.array([item.amplitude for item in fitted])
    etas = np.array([item.eta for item in fitted])
    assert np.all(np.abs(centers - CENTERS_HZ) < 0.08 * FWHM_HZ)
    assert_allclose(widths, FWHM_HZ, rtol=0.15, atol=0.0)
    assert_allclose(amplitudes, AMPLITUDES, rtol=0.12, atol=0.0)
    assert np.all(np.abs(etas - ETAS) < 0.20)
    assert result.baseline_estimate.intercept == pytest.approx(1.0, abs=8e-4)
    assert result.baseline_estimate.slope_per_hz == pytest.approx(2e-11, abs=2e-12)
    assert result.baseline_estimate.quadratic_per_hz2 == pytest.approx(
        -5e-20, abs=2e-20
    )
    assert result.uncertainty is not None
    for errors in (
        result.uncertainty.baseline_standard_errors,
        result.uncertainty.center_hz,
        result.uncertainty.fwhm_hz,
        result.uncertainty.amplitude,
        result.uncertainty.eta,
    ):
        assert errors is not None
        assert np.all(np.isfinite(errors))
        assert np.all(errors >= 0.0)


def test_fallback_can_recover_identifiable_eight_line_fixture() -> None:
    centers = np.linspace(2.7405e9, 3.0195e9, 8)
    resonances = tuple(
        Resonance(f"r{i}", center, 2.0e6, amplitude, 0.5)
        for i, (center, amplitude) in enumerate(zip(centers, AMPLITUDES, strict=True))
    )
    baseline = Baseline(1.0, REFERENCE_HZ, 2e-11)
    sweep = CompleteSweep(
        FREQUENCY_HZ, multi_resonance_spectrum(FREQUENCY_HZ, resonances, baseline)
    )
    configuration = _configuration(
        baseline_degree=1, relative_prominence=1.0, allow_fallback=True
    )

    result = fit_spectrum(sweep, configuration)

    assert result.success
    assert result.diagnostics.source == "fallback"
    assert result.diagnostics.used_fallback
    assert_allclose(
        [item.center_hz for item in result.resonance_estimates], centers, atol=2e4
    )


def test_fallback_does_not_turn_seven_dips_into_success() -> None:
    resonances, baseline = _truth(baseline_degree=1)
    sweep = CompleteSweep(
        FREQUENCY_HZ,
        multi_resonance_spectrum(FREQUENCY_HZ, resonances[:7], baseline),
    )
    result = fit_spectrum(sweep, _configuration(baseline_degree=1, allow_fallback=True))

    assert not result.success
    assert result.failure_code == "quality_failed"
    assert result.diagnostics.source == "fallback"


def _replace_resonance(
    guess: FitInitialGuess, index: int, **changes: object
) -> FitInitialGuess:
    resonances = list(guess.resonances)
    resonances[index] = replace(resonances[index], **changes)
    return FitInitialGuess(resonances=tuple(resonances), baseline=guess.baseline)


@pytest.mark.parametrize(
    "invalid_kind",
    ["ids", "eta", "reference", "interval", "separation", "width", "amplitude"],
)
def test_invalid_user_guesses_raise_before_scipy(invalid_kind: str) -> None:
    guess = _guess()
    if invalid_kind == "ids":
        resonances = list(guess.resonances)
        resonances[0] = replace(resonances[0], resonance_id="wrong")
        guess = FitInitialGuess(tuple(resonances), guess.baseline)
    elif invalid_kind == "eta":
        lorentz = _guess("lorentzian")
        guess = _replace_resonance(lorentz, 0, eta=0.5)
    elif invalid_kind == "reference":
        guess = FitInitialGuess(
            guess.resonances, replace(guess.baseline, reference_hz=REFERENCE_HZ + 1.0)
        )
    elif invalid_kind == "interval":
        guess = _replace_resonance(guess, 0, center_hz=FREQUENCY_HZ[0] - 1.0)
    elif invalid_kind == "separation":
        guess = _replace_resonance(
            guess, 1, center_hz=guess.resonances[0].center_hz + 0.5e6
        )
    elif invalid_kind == "width":
        guess = _replace_resonance(guess, 0, fwhm_hz=1.0e5)
    else:
        guess = _replace_resonance(guess, 0, amplitude=0.09)
    configuration = _configuration(
        model_kind="lorentzian" if invalid_kind == "eta" else "pseudo_voigt"
    )

    with pytest.raises(ValueError):
        fit_spectrum(_sweep(configuration.model_kind), configuration, guess)


def test_valid_user_guess_is_defensively_snapshotted() -> None:
    guess = _guess()
    result = fit_spectrum(_sweep(), _configuration(), guess)

    assert result.success
    assert result.diagnostics.source == "user"
    assert result.initial_guess == guess
    assert result.initial_guess is not guess


def test_identical_calls_are_numerically_repeatable() -> None:
    first = fit_spectrum(_sweep(), _configuration(), _guess())
    second = fit_spectrum(_sweep(), _configuration(), _guess())

    assert first.success == second.success
    assert first.failure_code == second.failure_code
    assert first.scipy_status == second.scipy_status
    assert first.diagnostics == second.diagnostics
    assert_array_equal(
        [item.resonance_id for item in first.resonance_estimates],
        [item.resonance_id for item in second.resonance_estimates],
    )
    for field in ("center_hz", "fwhm_hz", "amplitude", "eta"):
        assert_allclose(
            [getattr(item, field) for item in first.resonance_estimates],
            [getattr(item, field) for item in second.resonance_estimates],
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("scale", [1.0e-3, 1.0e3])
def test_multiplicative_fluorescence_units_preserve_scientific_fit(
    scale: float,
) -> None:
    base = fit_spectrum(_sweep(), _configuration(), _guess())
    scaled = fit_spectrum(
        _sweep(scale=scale),
        _configuration(
            max_amplitude=0.08 * scale,
            min_resolved_amplitude=1.0e-4 * scale,
        ),
        _guess(scale=scale),
    )

    assert scaled.success == base.success
    assert scaled.failure_code == base.failure_code
    assert scaled.jacobian_rank == base.jacobian_rank
    assert_array_equal(
        [item.resonance_id for item in scaled.resonance_estimates],
        [item.resonance_id for item in base.resonance_estimates],
    )
    assert (
        np.max(
            np.abs(
                np.array([x.center_hz for x in scaled.resonance_estimates])
                - np.array([x.center_hz for x in base.resonance_estimates])
            )
        )
        < 10.0
    )
    assert (
        np.max(
            np.abs(
                np.array([x.fwhm_hz for x in scaled.resonance_estimates])
                - np.array([x.fwhm_hz for x in base.resonance_estimates])
            )
        )
        < 1.0e3
    )
    assert (
        np.max(
            np.abs(
                np.array([x.eta for x in scaled.resonance_estimates])
                - np.array([x.eta for x in base.resonance_estimates])
            )
        )
        < 1.0e-4
    )
    assert_allclose(
        np.array([x.amplitude for x in scaled.resonance_estimates]) / scale,
        [x.amplitude for x in base.resonance_estimates],
        rtol=5e-6,
    )
    _assert_recovery(base, "pseudo_voigt", 2)
    inverse_baseline = replace(
        scaled.baseline_estimate,
        intercept=scaled.baseline_estimate.intercept / scale,
        slope_per_hz=scaled.baseline_estimate.slope_per_hz / scale,
        quadratic_per_hz2=scaled.baseline_estimate.quadratic_per_hz2 / scale,
    )
    assert inverse_baseline.intercept == pytest.approx(1.0, abs=2e-4)
    assert inverse_baseline.slope_per_hz == pytest.approx(2e-11, abs=5e-13)
    assert inverse_baseline.quadratic_per_hz2 == pytest.approx(-5e-20, abs=5e-21)


def test_additive_fluorescence_offset_preserves_scientific_fit() -> None:
    base = fit_spectrum(_sweep(), _configuration(), _guess())
    shifted = fit_spectrum(_sweep(offset=1.0e6), _configuration(), _guess(offset=1.0e6))

    assert shifted.success == base.success
    assert shifted.failure_code == base.failure_code
    assert shifted.jacobian_rank == base.jacobian_rank
    assert (
        np.max(
            np.abs(
                np.array([x.center_hz for x in shifted.resonance_estimates])
                - np.array([x.center_hz for x in base.resonance_estimates])
            )
        )
        < 10.0
    )
    assert (
        np.max(
            np.abs(
                np.array([x.fwhm_hz for x in shifted.resonance_estimates])
                - np.array([x.fwhm_hz for x in base.resonance_estimates])
            )
        )
        < 1.0e3
    )
    assert (
        np.max(
            np.abs(
                np.array([x.eta for x in shifted.resonance_estimates])
                - np.array([x.eta for x in base.resonance_estimates])
            )
        )
        < 1.0e-4
    )
    assert_allclose(
        [x.amplitude for x in shifted.resonance_estimates],
        [x.amplitude for x in base.resonance_estimates],
        rtol=5e-6,
    )
    assert shifted.baseline_estimate.intercept - 1.0e6 == pytest.approx(
        base.baseline_estimate.intercept, abs=2e-4
    )
    assert shifted.baseline_estimate.slope_per_hz == pytest.approx(
        base.baseline_estimate.slope_per_hz, abs=5e-13
    )
    assert shifted.baseline_estimate.quadratic_per_hz2 == pytest.approx(
        base.baseline_estimate.quadratic_per_hz2, abs=5e-21
    )


def test_affine_packing_and_scaled_residual_algebra() -> None:
    base_sweep = _sweep()
    scaled_sweep = _sweep(scale=8.0)
    configuration = _configuration()
    scaled_configuration = _configuration(
        max_amplitude=0.64, min_resolved_amplitude=8.0e-4
    )
    base_guess = _guess()
    scaled_guess = _guess(scale=8.0)
    half_span = (FREQUENCY_HZ[-1] - FREQUENCY_HZ[0]) / 2.0
    base_packed = pack_parameters(
        base_guess,
        configuration,
        frequency_reference_hz=REFERENCE_HZ,
        frequency_half_span_hz=half_span,
        fluorescence_reference=float(np.median(base_sweep.fluorescence)),
        fluorescence_scale=float(np.ptp(base_sweep.fluorescence)),
    )
    scaled_packed = pack_parameters(
        scaled_guess,
        scaled_configuration,
        frequency_reference_hz=REFERENCE_HZ,
        frequency_half_span_hz=half_span,
        fluorescence_reference=float(np.median(scaled_sweep.fluorescence)),
        fluorescence_scale=float(np.ptp(scaled_sweep.fluorescence)),
    )
    assert_array_equal(scaled_packed, base_packed)
    base_residual = _scaled_residual_function(
        base_sweep,
        configuration,
        frequency_reference_hz=REFERENCE_HZ,
        frequency_half_span_hz=half_span,
        fluorescence_reference=float(np.median(base_sweep.fluorescence)),
        fluorescence_scale=float(np.ptp(base_sweep.fluorescence)),
    )(base_packed)
    scaled_residual = _scaled_residual_function(
        scaled_sweep,
        scaled_configuration,
        frequency_reference_hz=REFERENCE_HZ,
        frequency_half_span_hz=half_span,
        fluorescence_reference=float(np.median(scaled_sweep.fluorescence)),
        fluorescence_scale=float(np.ptp(scaled_sweep.fluorescence)),
    )(scaled_packed)
    assert_allclose(scaled_residual, base_residual, rtol=0.0, atol=5e-8)


@pytest.mark.parametrize(
    ("improvement", "expected_success"),
    [(0.5 - 1e-12, False), (0.5, True), (0.5 + 1e-12, True)],
)
def test_baseline_improvement_gate_includes_exact_threshold(
    monkeypatch: pytest.MonkeyPatch, improvement: float, expected_success: bool
) -> None:
    sweep = _sweep("lorentzian", 1)
    configuration = _configuration("lorentzian", 1, min_baseline_sse_improvement=0.5)
    z = (FREQUENCY_HZ - REFERENCE_HZ) / ((FREQUENCY_HZ[-1] - FREQUENCY_HZ[0]) / 2)
    design = np.column_stack([np.ones(z.size), z])
    coefficients, _, _, _ = np.linalg.lstsq(design, sweep.fluorescence, rcond=None)
    baseline_sse = float(np.sum((design @ coefficients - sweep.fluorescence) ** 2))
    scale = float(np.ptp(sweep.fluorescence))

    def controlled_least_squares(
        fun: object, x0: np.ndarray, **kwargs: object
    ) -> SimpleNamespace:
        del fun, kwargs
        scaled_cost = baseline_sse * (1.0 - improvement) / (2.0 * scale**2)
        columns = x0.size
        jacobian = np.zeros((FREQUENCY_HZ.size, columns))
        jacobian[:columns] = np.eye(columns)
        return SimpleNamespace(
            success=True,
            status=1,
            message="controlled",
            nfev=1,
            x=x0,
            fun=np.full(
                FREQUENCY_HZ.size, np.sqrt(2 * scaled_cost / FREQUENCY_HZ.size)
            ),
            cost=scaled_cost,
            jac=jacobian,
        )

    monkeypatch.setattr(
        "odmr_bench.estimators.fitting.least_squares", controlled_least_squares
    )
    result = fit_spectrum(sweep, configuration, _guess("lorentzian", 1))

    assert result.success is expected_success
    assert result.failure_code == (None if expected_success else "quality_failed")
