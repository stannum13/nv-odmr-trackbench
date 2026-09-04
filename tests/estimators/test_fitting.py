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
    InitializationDiagnostics,
    fit_spectrum,
    linearized_standard_errors,
)
from odmr_bench.estimators.fitting import (
    _scaled_residual_function,
    _uncertainty_from_errors,
)
from odmr_bench.estimators.parameterization import (
    pack_parameters,
    public_parameter_transform,
)
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


def test_covariance_covers_complete_public_layout_in_declared_order() -> None:
    configuration = _configuration()
    transform = public_parameter_transform(
        configuration,
        frequency_half_span_hz=8.0,
        fluorescence_scale=4.0,
    )
    diagonal = np.linspace(1.0, 2.0, 35)
    jacobian = np.diag(diagonal)

    errors, rank, reason = linearized_standard_errors(
        jacobian,
        scaled_cost=5.0,
        degrees_of_freedom=10,
        public_transform=transform,
        rank_rtol=1.0e-12,
    )

    expected = np.diag(transform) / diagonal
    assert rank == 35
    assert reason is None
    assert_allclose(errors, expected, rtol=2.0e-15, atol=0.0)

    uncertainty = _uncertainty_from_errors(np.arange(35.0), configuration)
    assert_array_equal(uncertainty.baseline_standard_errors, [0.0, 1.0, 2.0])
    resonance_layout = np.arange(3.0, 27.0).reshape(8, 3)
    assert_array_equal(uncertainty.amplitude, resonance_layout[:, 0])
    assert_array_equal(uncertainty.center_hz, resonance_layout[:, 1])
    assert_array_equal(uncertainty.fwhm_hz, resonance_layout[:, 2])
    assert_array_equal(uncertainty.eta, np.arange(27.0, 35.0))


def test_singular_value_exactly_at_cutoff_is_not_retained() -> None:
    errors, rank, reason = linearized_standard_errors(
        np.diag([1.0, 1.0e-4]),
        scaled_cost=1.0,
        degrees_of_freedom=3,
        public_transform=np.eye(2),
        rank_rtol=1.0e-4,
    )

    assert errors is None
    assert rank == 1
    assert reason == "scaled Jacobian is rank deficient"


def test_rank_and_covariance_share_one_svd_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_svd = np.linalg.svd
    calls = 0

    def counting_svd(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counting_svd)

    errors, rank, reason = linearized_standard_errors(
        np.eye(4), 1.0, 5, np.eye(4), 1.0e-12
    )

    assert errors is not None
    assert rank == 4
    assert reason is None
    assert calls == 1


@pytest.mark.parametrize(
    ("jacobian", "cost", "dof", "transform", "expected_rank"),
    [
        (np.diag([1.0, 0.0]), 1.0, 2, np.eye(2), 1),
        (np.array([[1.0, np.nan]]), 1.0, 2, np.eye(2), None),
        (np.eye(2), 1.0, 0, np.eye(2), 2),
        (np.eye(2), np.nan, 2, np.eye(2), 2),
        (np.eye(2), 1.0, 2, np.eye(3), 2),
        (np.eye(2), 1.0, 2, np.diag([1.0, np.inf]), 2),
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


def test_successful_optimizer_with_nonfinite_outputs_fails_quality_without_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite_least_squares(
        fun: object, x0: np.ndarray, **kwargs: object
    ) -> SimpleNamespace:
        del fun, kwargs
        fitted = x0.copy()
        fitted[0] = np.nan
        return SimpleNamespace(
            success=True,
            status=2,
            message="terminated with nonfinite output",
            nfev=4,
            x=fitted,
            fun=np.full(FREQUENCY_HZ.size, np.nan),
            cost=np.nan,
            jac=np.full((FREQUENCY_HZ.size, x0.size), np.nan),
        )

    monkeypatch.setattr(
        "odmr_bench.estimators.fitting.least_squares", nonfinite_least_squares
    )

    result = fit_spectrum(_sweep(), _configuration(), _guess())

    assert not result.success
    assert result.failure_code == "quality_failed"
    assert result.cost is None
    assert result.residual_rmse is None
    assert result.jacobian_rank is None
    assert result.uncertainty is None
    assert result.uncertainty_reason == (
        "optimizer returned non-finite parameters, residuals, or cost"
    )
    assert result.scipy_status == 2
    assert result.nfev == 4


def test_successful_optimizer_with_nan_x_discards_finite_residual_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nan_x_least_squares(
        fun: object, x0: np.ndarray, **kwargs: object
    ) -> SimpleNamespace:
        del fun, kwargs
        fitted = x0.copy()
        fitted[0] = np.nan
        return SimpleNamespace(
            success=True,
            status=2,
            message="terminated with NaN x",
            nfev=4,
            x=fitted,
            fun=np.zeros(FREQUENCY_HZ.size),
            cost=0.0,
            jac=np.eye(FREQUENCY_HZ.size, x0.size),
        )

    monkeypatch.setattr(
        "odmr_bench.estimators.fitting.least_squares", nan_x_least_squares
    )

    result = fit_spectrum(_sweep(), _configuration(), _guess())

    assert result.failure_code == "quality_failed"
    assert result.cost is None
    assert result.residual_rmse is None
    assert result.uncertainty_reason == (
        "optimizer returned non-finite parameters, residuals, or cost"
    )


def test_detected_exact_separation_preflight_failure_has_stable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _configuration(baseline_degree=1)
    centers = 2.75e9 + configuration.min_center_separation_hz * np.arange(8)
    detected_guess = FitInitialGuess(
        tuple(
            Resonance(f"r{i}", center, 5.0e5, 0.01, 0.5)
            for i, center in enumerate(centers)
        ),
        Baseline(1.0, REFERENCE_HZ, 2.0e-11),
    )
    diagnostics = InitializationDiagnostics(
        source="detected",
        candidate_count=8,
        selected_indices=tuple(range(8)),
        used_fallback=False,
        messages=(),
    )
    monkeypatch.setattr(
        "odmr_bench.estimators.fitting.initialize_spectrum",
        lambda sweep, config: (detected_guess, diagnostics),
    )

    def unexpected_scipy(*args: object, **kwargs: object) -> None:
        raise AssertionError("SciPy must not run after guess preflight failure")

    monkeypatch.setattr("odmr_bench.estimators.fitting.least_squares", unexpected_scipy)

    result = fit_spectrum(_sweep(baseline_degree=1), configuration)

    assert result.failure_code == "initialization_failed"
    assert result.diagnostics.source == "detected"
    assert result.diagnostics.messages
    assert result.uncertainty_reason == result.diagnostics.messages[-1]
    assert result.scipy_status is None
    assert result.nfev == 0


def test_public_amplitude_rounding_above_bound_fails_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep = _sweep()
    scale = float(np.ptp(sweep.fluorescence))
    maximum = np.float64(0.08)
    for _ in range(1000):
        if (maximum / scale) * scale > maximum:
            break
        maximum = np.nextafter(maximum, -np.inf)
    else:
        raise AssertionError("failed to construct saturation rounding fixture")
    configuration = _configuration(max_amplitude=float(maximum))

    def saturating_least_squares(
        fun: object, x0: np.ndarray, **kwargs: object
    ) -> SimpleNamespace:
        del fun
        fitted = x0.copy()
        fitted[3] = kwargs["bounds"][1][3]
        jacobian = np.zeros((FREQUENCY_HZ.size, x0.size))
        jacobian[: x0.size] = np.eye(x0.size)
        return SimpleNamespace(
            success=True,
            status=1,
            message="saturated",
            nfev=1,
            x=fitted,
            fun=np.zeros(FREQUENCY_HZ.size),
            cost=0.0,
            jac=jacobian,
        )

    monkeypatch.setattr(
        "odmr_bench.estimators.fitting.least_squares", saturating_least_squares
    )
    result = fit_spectrum(sweep, configuration, _guess())

    assert result.failure_code == "quality_failed"
    assert result.resonance_estimates == ()
    assert "public" in result.uncertainty_reason


def test_underflowed_parameter_bounds_are_rejected_before_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fluorescence = np.linspace(1.0e307, 1.0e308, FREQUENCY_HZ.size)
    configuration = _configuration(
        baseline_degree=1,
        max_amplitude=1.0e-300,
        min_resolved_amplitude=1.0e-300,
    )
    guess = FitInitialGuess(
        tuple(
            replace(item, amplitude=1.0e-300, eta=0.5)
            for item in _guess(baseline_degree=1).resonances
        ),
        Baseline(float(fluorescence[fluorescence.size // 2]), REFERENCE_HZ),
    )

    def unexpected_scipy(*args: object, **kwargs: object) -> None:
        raise AssertionError("SciPy must not receive invalid numerical bounds")

    monkeypatch.setattr("odmr_bench.estimators.fitting.least_squares", unexpected_scipy)

    with pytest.raises(ValueError, match="parameterization"):
        fit_spectrum(CompleteSweep(FREQUENCY_HZ, fluorescence), configuration, guess)


def test_large_finite_baseline_coordinate_is_not_artificially_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guess = _guess()
    guess = FitInitialGuess(
        guess.resonances, replace(guess.baseline, intercept=1.0e100)
    )

    def observing_least_squares(
        fun: object, x0: np.ndarray, **kwargs: object
    ) -> SimpleNamespace:
        del fun
        lower, upper = kwargs["bounds"]
        assert np.all(np.isneginf(lower[:3]))
        assert np.all(np.isposinf(upper[:3]))
        assert np.isfinite(x0[0])
        return SimpleNamespace(
            success=False,
            status=0,
            message="baseline accepted",
            nfev=1,
            x=x0,
            fun=np.zeros(FREQUENCY_HZ.size),
            cost=0.0,
            jac=np.zeros((FREQUENCY_HZ.size, x0.size)),
        )

    monkeypatch.setattr(
        "odmr_bench.estimators.fitting.least_squares", observing_least_squares
    )

    result = fit_spectrum(_sweep(), _configuration(), guess)

    assert result.failure_code == "optimization_failed"


def test_nonfinite_covariance_transform_keeps_identified_fit_without_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = np.ldexp(1.0, -900)
    fluorescence_scale = np.ldexp(1.0, 900)
    frequency = np.linspace(0.0, span, 101)
    sweep = CompleteSweep(
        frequency, np.linspace(0.0, fluorescence_scale, frequency.size)
    )
    centers = span * (2.0 + 4.0 * np.arange(8)) / 32.0
    configuration = FitConfiguration(
        model_kind="pseudo_voigt",
        baseline_degree=2,
        min_fwhm_hz=span / 1024.0,
        max_fwhm_hz=span / 8.0,
        max_amplitude=fluorescence_scale / 32.0,
        min_resolved_amplitude=fluorescence_scale / 4096.0,
        min_center_separation_hz=span / 32.0,
        min_baseline_sse_improvement=0.0,
    )
    guess = FitInitialGuess(
        tuple(
            Resonance(
                f"r{i}",
                center,
                span / 128.0,
                fluorescence_scale / 1024.0,
                0.5,
            )
            for i, center in enumerate(centers)
        ),
        Baseline(fluorescence_scale / 2.0, span / 2.0),
    )

    def successful_least_squares(
        fun: object, x0: np.ndarray, **kwargs: object
    ) -> SimpleNamespace:
        del fun, kwargs
        jacobian = np.zeros((frequency.size, x0.size))
        jacobian[: x0.size] = np.eye(x0.size)
        return SimpleNamespace(
            success=True,
            status=1,
            message="identified",
            nfev=1,
            x=x0,
            fun=np.zeros(frequency.size),
            cost=0.0,
            jac=jacobian,
        )

    svd_calls = 0
    original_svd = np.linalg.svd

    def counting_svd(*args: object, **kwargs: object) -> object:
        nonlocal svd_calls
        svd_calls += 1
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(
        "odmr_bench.estimators.fitting.least_squares", successful_least_squares
    )
    monkeypatch.setattr(
        "odmr_bench.estimators.fitting._baseline_only_sse", lambda *args: 1.0
    )
    monkeypatch.setattr(np.linalg, "svd", counting_svd)

    result = fit_spectrum(sweep, configuration, guess)

    assert result.success
    assert result.jacobian_rank == 35
    assert result.uncertainty is None
    assert result.uncertainty_reason == (
        "public parameter transform unavailable: "
        "quadratic public transform is not representable"
    )
    assert svd_calls == 1


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


def test_same_sign_extreme_fluorescence_has_finite_origin_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequency = np.linspace(FREQUENCY_HZ[0], FREQUENCY_HZ[-1], 4480)
    scale = 1.0e307
    resonances, baseline = _truth(scale=scale, offset=1.0e308)
    sweep = CompleteSweep(
        frequency, multi_resonance_spectrum(frequency, resonances, baseline)
    )
    configuration = _configuration(
        max_amplitude=0.08 * scale,
        min_resolved_amplitude=1.0e-4 * scale,
    )
    guess = _guess(scale=scale, offset=1.0e308)

    def finite_origin_least_squares(
        fun: object, x0: np.ndarray, **kwargs: object
    ) -> SimpleNamespace:
        del fun, kwargs
        assert np.all(np.isfinite(x0))
        return SimpleNamespace(
            success=False,
            status=0,
            message="origin observed",
            nfev=1,
            x=x0,
            fun=np.zeros(frequency.size),
            cost=0.0,
            jac=np.zeros((frequency.size, x0.size)),
        )

    monkeypatch.setattr(
        "odmr_bench.estimators.fitting.least_squares", finite_origin_least_squares
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = fit_spectrum(sweep, configuration, guess)

    assert result.failure_code == "optimization_failed"
    assert result.residual_scale is not None


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
def test_invalid_user_guesses_raise_before_scipy(
    invalid_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    def unexpected_scipy(*args: object, **kwargs: object) -> None:
        raise AssertionError("SciPy must not run for invalid user guesses")

    monkeypatch.setattr("odmr_bench.estimators.fitting.least_squares", unexpected_scipy)

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
    assert_allclose(first.q_values, second.q_values, rtol=1e-12, atol=1e-12)
    assert first.baseline_estimate is not None
    assert second.baseline_estimate is not None
    assert_allclose(
        [
            first.baseline_estimate.intercept,
            first.baseline_estimate.reference_hz,
            first.baseline_estimate.slope_per_hz,
            first.baseline_estimate.quadratic_per_hz2,
        ],
        [
            second.baseline_estimate.intercept,
            second.baseline_estimate.reference_hz,
            second.baseline_estimate.slope_per_hz,
            second.baseline_estimate.quadratic_per_hz2,
        ],
        rtol=1e-12,
        atol=1e-12,
    )
    assert first.uncertainty is not None
    assert second.uncertainty is not None
    for field in (
        "baseline_standard_errors",
        "center_hz",
        "fwhm_hz",
        "amplitude",
        "eta",
    ):
        assert_allclose(
            getattr(first.uncertainty, field),
            getattr(second.uncertainty, field),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("scale", [1.0e-3, 1.0e3])
def test_multiplicative_fluorescence_units_preserve_scientific_fit(
    scale: float,
) -> None:
    noise = np.random.default_rng(6104).normal(0.0, 2.0e-6, FREQUENCY_HZ.size)
    base = fit_spectrum(_sweep(noise=noise), _configuration(), _guess())
    scaled = fit_spectrum(
        _sweep(scale=scale, noise=noise * scale),
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
