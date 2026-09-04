"""Shared fit-preparation and guarded warm-start regressions."""

from __future__ import annotations

import math
import warnings
from dataclasses import FrozenInstanceError, replace
from typing import get_args

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from odmr_bench.estimators import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    InitializationDiagnostics,
    SpectrumFitResult,
)
from odmr_bench.estimators.parameterization import pack_parameters, parameter_bounds
from odmr_bench.estimators.preparation import (
    BaselineRebaseError,
    FitPreflight,
    InitialGuessCompatibilityError,
    WarmStartCompatibilityCode,
    WarmStartPreparation,
    prepare_warm_start,
    rebase_baseline,
    start_independent_preflight,
    validate_initial_guess,
)
from odmr_bench.models import Baseline, Resonance


def _configuration(**changes: object) -> FitConfiguration:
    values: dict[str, object] = {
        "model_kind": "pseudo_voigt",
        "baseline_degree": 1,
        "min_fwhm_hz": 2.0e5,
        "max_fwhm_hz": 8.0e6,
        "max_amplitude": 0.25,
        "min_resolved_amplitude": 1.0e-4,
        "min_center_separation_hz": 1.0e6,
        "savgol_window": 5,
    }
    values.update(changes)
    return FitConfiguration(**values)


def _sweep(sample_count: int, *, constant: bool = False) -> CompleteSweep:
    frequency_hz = np.linspace(2.74e9, 3.02e9, sample_count)
    fluorescence = (
        np.ones(sample_count)
        if constant
        else np.linspace(0.9, 1.1, sample_count)
    )
    return CompleteSweep(frequency_hz, fluorescence)


def _fit_ready_sweep(
    *, frequency_min_hz: float = 0.0, frequency_max_hz: float = 100.0e6
) -> CompleteSweep:
    frequency_hz = np.linspace(frequency_min_hz, frequency_max_hz, 101)
    return CompleteSweep(frequency_hz, np.linspace(0.9, 1.1, frequency_hz.size))


def _valid_guess(
    configuration: FitConfiguration | None = None,
    *,
    baseline: Baseline | None = None,
) -> FitInitialGuess:
    configuration = configuration or _configuration()
    return FitInitialGuess(
        tuple(
            Resonance(
                resonance_id,
                10.0e6 * (index + 1),
                1.0e6,
                0.02,
                0.5 if configuration.model_kind == "pseudo_voigt" else 1.0,
            )
            for index, resonance_id in enumerate(configuration.resonance_ids)
        ),
        baseline or Baseline(1.0, 50.0e6, 2.0e-11),
    )


def _fit_preflight(
    configuration: FitConfiguration | None = None,
    sweep: CompleteSweep | None = None,
) -> FitPreflight:
    result = start_independent_preflight(
        sweep or _fit_ready_sweep(), configuration or _configuration()
    )
    assert isinstance(result, FitPreflight)
    return result


def _successful_prior(
    *,
    model_kind: str = "pseudo_voigt",
    baseline_degree: int = 1,
    resonance_ids: tuple[str, ...] = tuple(f"r{i}" for i in range(8)),
    resonances: tuple[Resonance, ...] | None = None,
    baseline: Baseline | None = None,
) -> SpectrumFitResult:
    prior_configuration = _configuration(
        model_kind=model_kind,
        baseline_degree=baseline_degree,
        resonance_ids=resonance_ids,
    )
    if resonances is None:
        guess = _valid_guess(prior_configuration, baseline=baseline)
    else:
        guess = FitInitialGuess(
            resonances,
            baseline or Baseline(1.0, 50.0e6, 2.0e-11),
        )
    free_parameters = baseline_degree + 1 + 8 * (
        4 if model_kind == "pseudo_voigt" else 3
    )
    return SpectrumFitResult(
        success=True,
        failure_code=None,
        model_kind=model_kind,
        baseline_degree=baseline_degree,
        resonance_estimates=guess.resonances,
        baseline_estimate=guess.baseline,
        diagnostics=InitializationDiagnostics("user", 0, (), False, ()),
        initial_guess=guess,
        uncertainty=None,
        uncertainty_reason="uncertainty omitted from warm-start fixture",
        scipy_status=1,
        scipy_message="converged",
        nfev=1,
        cost=0.0,
        residual_rmse=0.0,
        residual_scale=0.2,
        degrees_of_freedom=67,
        jacobian_rank=free_parameters,
    )


def _optimization_failure() -> SpectrumFitResult:
    guess = _valid_guess()
    return SpectrumFitResult(
        success=False,
        failure_code="optimization_failed",
        model_kind="pseudo_voigt",
        baseline_degree=1,
        resonance_estimates=(),
        baseline_estimate=None,
        diagnostics=InitializationDiagnostics("user", 0, (), False, ()),
        initial_guess=guess,
        uncertainty=None,
        uncertainty_reason="optimizer failed",
        scipy_status=0,
        scipy_message="stopped",
        nfev=1,
        cost=0.1,
        residual_rmse=0.01,
        residual_scale=0.2,
        degrees_of_freedom=67,
        jacobian_rank=None,
    )


def _assert_exact_preflight_failure(
    result: SpectrumFitResult,
    *,
    failure_code: str,
    reason: str,
    degrees_of_freedom: int,
) -> None:
    assert result.success is False
    assert result.failure_code == failure_code
    assert result.model_kind == "pseudo_voigt"
    assert result.baseline_degree == 1
    assert result.resonance_estimates == ()
    assert result.baseline_estimate is None
    assert result.diagnostics.source == "none"
    assert result.diagnostics.candidate_count == 0
    assert result.diagnostics.selected_indices == ()
    assert result.diagnostics.used_fallback is False
    assert result.diagnostics.messages == (reason,)
    assert result.initial_guess is None
    assert result.uncertainty is None
    assert result.uncertainty_reason == reason
    assert result.scipy_status is None
    assert result.scipy_message is None
    assert result.nfev == 0
    assert result.cost is None
    assert result.residual_rmse is None
    assert result.residual_scale is None
    assert result.degrees_of_freedom == degrees_of_freedom
    assert result.jacobian_rank is None
    assert result.q_values.shape == (0,)


def test_preflight_requires_strictly_positive_degrees_of_freedom() -> None:
    configuration = _configuration()

    insufficient = start_independent_preflight(_sweep(34), configuration)
    ready = start_independent_preflight(_sweep(35), configuration)

    assert isinstance(insufficient, SpectrumFitResult)
    _assert_exact_preflight_failure(
        insufficient,
        failure_code="insufficient_samples",
        reason="sample count must exceed the free parameter count",
        degrees_of_freedom=0,
    )
    assert isinstance(ready, FitPreflight)
    assert ready.free_parameter_count == 34
    assert ready.degrees_of_freedom == 1


def test_preflight_constant_sweep_has_exact_uninformative_result() -> None:
    result = start_independent_preflight(_sweep(35, constant=True), _configuration())

    assert isinstance(result, SpectrumFitResult)
    _assert_exact_preflight_failure(
        result,
        failure_code="uninformative_sweep",
        reason="fluorescence variation is zero or non-finite",
        degrees_of_freedom=1,
    )


def test_preflight_sample_count_precedes_variation_and_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_ptp(*args: object, **kwargs: object) -> None:
        raise AssertionError("variation must not run for an insufficient sweep")

    def unexpected_origin(*args: object, **kwargs: object) -> None:
        raise AssertionError("origin must not run for an insufficient sweep")

    monkeypatch.setattr("odmr_bench.estimators.preparation.np.ptp", unexpected_ptp)
    monkeypatch.setattr(
        "odmr_bench.estimators.preparation._stable_fluorescence_reference",
        unexpected_origin,
    )

    result = start_independent_preflight(_sweep(34), _configuration())

    assert isinstance(result, SpectrumFitResult)
    assert result.failure_code == "insufficient_samples"


def test_preflight_nonfinite_origin_has_exact_uninformative_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "odmr_bench.estimators.preparation._stable_fluorescence_reference",
        lambda fluorescence: np.inf,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = start_independent_preflight(_sweep(35), _configuration())

    assert isinstance(result, SpectrumFitResult)
    _assert_exact_preflight_failure(
        result,
        failure_code="uninformative_sweep",
        reason="fluorescence origin is non-finite numerically",
        degrees_of_freedom=1,
    )


@pytest.mark.parametrize("argument", [object(), None])
def test_preflight_rejects_wrong_argument_types(argument: object) -> None:
    with pytest.raises(TypeError):
        start_independent_preflight(argument, _configuration())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        start_independent_preflight(_sweep(35), argument)  # type: ignore[arg-type]


def test_fit_preflight_is_frozen_slotted_and_canonical() -> None:
    preflight = start_independent_preflight(_sweep(35), _configuration())

    assert isinstance(preflight, FitPreflight)
    assert not hasattr(preflight, "__dict__")
    assert type(preflight.free_parameter_count) is int
    assert type(preflight.frequency_reference_hz) is float
    with pytest.raises(FrozenInstanceError):
        preflight.degrees_of_freedom = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"free_parameter_count": True},
        {"free_parameter_count": 0},
        {"degrees_of_freedom": 0},
        {"frequency_min_hz": np.nan},
        {"frequency_max_hz": np.inf},
        {"frequency_max_hz": 10**10000},
        {"frequency_min_hz": 2.0, "frequency_max_hz": 1.0},
        {"frequency_reference_hz": 1.25},
        {"frequency_half_span_hz": 0.25},
        {"fluorescence_reference": np.inf},
        {"fluorescence_scale": 0.0},
    ],
)
def test_fit_preflight_rejects_inconsistent_state(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "free_parameter_count": 34,
        "degrees_of_freedom": 1,
        "frequency_min_hz": 0.0,
        "frequency_max_hz": 2.0,
        "frequency_reference_hz": 1.0,
        "frequency_half_span_hz": 1.0,
        "fluorescence_reference": 1.0,
        "fluorescence_scale": 0.2,
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        FitPreflight(**values)  # type: ignore[arg-type]


def test_validate_initial_guess_returns_exact_parameterization_arrays() -> None:
    configuration = _configuration()
    guess = _valid_guess(configuration)
    preflight = _fit_preflight(configuration)

    packed, lower, upper = validate_initial_guess(guess, configuration, preflight)

    expected_packed = pack_parameters(
        guess,
        configuration,
        frequency_reference_hz=preflight.frequency_reference_hz,
        frequency_half_span_hz=preflight.frequency_half_span_hz,
        fluorescence_reference=preflight.fluorescence_reference,
        fluorescence_scale=preflight.fluorescence_scale,
    )
    expected_lower, expected_upper = parameter_bounds(
        guess,
        configuration,
        frequency_min_hz=preflight.frequency_min_hz,
        frequency_max_hz=preflight.frequency_max_hz,
        frequency_reference_hz=preflight.frequency_reference_hz,
        frequency_half_span_hz=preflight.frequency_half_span_hz,
        fluorescence_scale=preflight.fluorescence_scale,
    )
    assert_array_equal(packed, expected_packed)
    assert_array_equal(lower, expected_lower)
    assert_array_equal(upper, expected_upper)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("ids", "IDs"),
        ("lorentzian_eta", "eta"),
        ("fwhm_below", "FWHM"),
        ("fwhm_above", "FWHM"),
        ("amplitude_above", "amplitude"),
        ("center_outside", "inside the sweep"),
        ("separation_below", "minimum separation"),
    ],
)
def test_validate_initial_guess_classifies_deliberate_incompatibility(
    case: str, message: str
) -> None:
    configuration = _configuration()
    guess = _valid_guess(configuration)
    resonances = list(guess.resonances)
    if case == "ids":
        configuration = replace(
            configuration,
            resonance_ids=("other", *configuration.resonance_ids[1:]),
        )
    elif case == "lorentzian_eta":
        configuration = replace(configuration, model_kind="lorentzian")
    elif case == "fwhm_below":
        resonances[0] = replace(
            resonances[0], fwhm_hz=np.nextafter(configuration.min_fwhm_hz, 0.0)
        )
    elif case == "fwhm_above":
        resonances[0] = replace(
            resonances[0], fwhm_hz=np.nextafter(configuration.max_fwhm_hz, np.inf)
        )
    elif case == "amplitude_above":
        resonances[0] = replace(
            resonances[0], amplitude=np.nextafter(configuration.max_amplitude, np.inf)
        )
    elif case == "center_outside":
        resonances[0] = replace(resonances[0], center_hz=-1.0)
    elif case == "separation_below":
        resonances[1] = replace(
            resonances[1],
            center_hz=resonances[0].center_hz
            + configuration.min_center_separation_hz
            - 1.0,
        )
    guess = FitInitialGuess(tuple(resonances), guess.baseline)

    with pytest.raises(InitialGuessCompatibilityError, match=message):
        validate_initial_guess(guess, configuration, _fit_preflight(configuration))


def test_validate_initial_guess_accepts_inclusive_public_boundaries() -> None:
    configuration = _configuration()
    guess = _valid_guess(configuration)
    resonances = list(guess.resonances)
    resonances[0] = replace(
        resonances[0],
        center_hz=0.0,
        fwhm_hz=configuration.min_fwhm_hz,
        amplitude=0.0,
        eta=0.0,
    )
    resonances[2] = replace(
        resonances[2],
        center_hz=resonances[1].center_hz
        + configuration.min_center_separation_hz,
    )
    resonances[-1] = replace(
        resonances[-1],
        center_hz=100.0e6,
        fwhm_hz=configuration.max_fwhm_hz,
        amplitude=configuration.max_amplitude,
        eta=1.0,
    )

    packed, lower, upper = validate_initial_guess(
        FitInitialGuess(tuple(resonances), guess.baseline),
        configuration,
        _fit_preflight(configuration),
    )

    assert np.all(packed >= lower)
    assert np.all(packed <= upper)


@pytest.mark.parametrize(
    ("packed", "lower", "upper", "message"),
    [
        (np.array([np.inf]), np.array([-np.inf]), np.array([np.inf]), "invalid"),
        (np.array([0.0]), np.array([0.0]), np.array([0.0]), "strict bounds"),
        (np.array([2.0]), np.array([0.0]), np.array([1.0]), "inside optimizer"),
    ],
)
def test_validate_initial_guess_rejects_invalid_parameterization_payloads(
    monkeypatch: pytest.MonkeyPatch,
    packed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    message: str,
) -> None:
    configuration = _configuration(model_kind="lorentzian")
    guess = _valid_guess(configuration)
    baseline_size = configuration.baseline_degree + 1
    size = baseline_size + 24
    supplied_packed = np.zeros(size)
    supplied_lower = np.concatenate((np.full(baseline_size, -np.inf), np.zeros(24)))
    supplied_upper = np.concatenate((np.full(baseline_size, np.inf), np.ones(24)))
    supplied_packed[-1] = packed[0]
    supplied_lower[-1] = lower[0]
    supplied_upper[-1] = upper[0]
    monkeypatch.setattr(
        "odmr_bench.estimators.preparation.pack_parameters",
        lambda *args, **kwargs: supplied_packed,
    )
    monkeypatch.setattr(
        "odmr_bench.estimators.preparation.parameter_bounds",
        lambda *args, **kwargs: (supplied_lower, supplied_upper),
    )

    with pytest.raises(InitialGuessCompatibilityError, match=message):
        validate_initial_guess(guess, configuration, _fit_preflight(configuration))


@pytest.mark.parametrize("helper", ["pack_parameters", "parameter_bounds"])
def test_validate_initial_guess_wraps_only_parameterization_value_errors(
    monkeypatch: pytest.MonkeyPatch, helper: str
) -> None:
    def deliberate_failure(*args: object, **kwargs: object) -> None:
        raise ValueError("injected parameterization failure")

    monkeypatch.setattr(
        f"odmr_bench.estimators.preparation.{helper}", deliberate_failure
    )
    with pytest.raises(
        InitialGuessCompatibilityError, match="injected parameterization failure"
    ) as caught:
        validate_initial_guess(
            _valid_guess(), _configuration(), _fit_preflight()
        )
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("argument_position", [0, 1, 2])
def test_validate_initial_guess_rejects_wrong_argument_types(
    argument_position: int,
) -> None:
    arguments: list[object] = [_valid_guess(), _configuration(), _fit_preflight()]
    arguments[argument_position] = object()
    with pytest.raises(TypeError):
        validate_initial_guess(*arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("quadratic", "expected_intercept", "expected_slope"),
    [
        (0.0, 1.00002, 2.0e-11),
        (
            -5.0e-20,
            1.00001995,
            float.fromhex("0x1.5e15a1f111b1ep-36"),
        ),
    ],
)
def test_rebase_baseline_preserves_linear_and_quadratic_polynomials(
    quadratic: float, expected_intercept: float, expected_slope: float
) -> None:
    old_reference = 2.880e9
    new_reference = 2.881e9
    baseline = Baseline(1.0, old_reference, 2.0e-11, quadratic)
    frequencies = new_reference + np.array([-4.0e6, 0.0, 7.0e6])

    rebased = rebase_baseline(baseline, new_reference)

    assert rebased.intercept == expected_intercept
    assert rebased.slope_per_hz == expected_slope
    assert rebased.quadratic_per_hz2 == quadratic
    assert rebased.reference_hz == new_reference
    np.testing.assert_allclose(
        rebased.evaluate(frequencies),
        baseline.evaluate(frequencies),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


@pytest.mark.parametrize(
    ("baseline", "new_reference_hz", "field"),
    [
        (Baseline(1.0, 0.0, -1.0), 1.0, "intercept"),
        (Baseline(0.0, 0.0, 2.0, -1.0), 1.0, "slope_per_hz"),
    ],
)
def test_rebase_baseline_accepts_exact_zero_cancellation(
    baseline: Baseline, new_reference_hz: float, field: str
) -> None:
    rebased = rebase_baseline(baseline, new_reference_hz)

    assert getattr(rebased, field) == 0.0


def test_rebase_baseline_short_circuits_zero_shape_for_extreme_references() -> None:
    maximum = np.finfo(np.float64).max

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rebased = rebase_baseline(Baseline(1.0, -maximum), maximum)

    assert rebased == Baseline(1.0, maximum)


def test_rebase_baseline_rejects_unrepresentable_reference_difference() -> None:
    maximum = np.finfo(np.float64).max

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            BaselineRebaseError,
            match="baseline reference difference is not representable",
        ):
            rebase_baseline(Baseline(0.0, -maximum, 1.0), maximum)


@pytest.mark.parametrize(
    ("baseline", "new_reference_hz", "expected_intercept", "should_reject"),
    [
        (
            Baseline(0.0, 0.0, slope_per_hz=math.ldexp(1.0, 500)),
            math.ldexp(1.0, 500),
            math.ldexp(1.0, 1000),
            False,
        ),
        (
            Baseline(0.0, 0.0, slope_per_hz=math.ldexp(1.0, 700)),
            math.ldexp(1.0, 400),
            None,
            True,
        ),
        (
            Baseline(0.0, 0.0, quadratic_per_hz2=math.ldexp(1.0, -1000)),
            math.ldexp(1.0, 500),
            1.0,
            False,
        ),
        (
            Baseline(0.0, 0.0, slope_per_hz=np.nextafter(0.0, 1.0)),
            0.5,
            None,
            True,
        ),
    ],
    ids=[
        "representable-linear-product",
        "overflowing-linear-product",
        "representable-quadratic-product",
        "underflowing-linear-product",
    ],
)
def test_rebase_baseline_handles_extreme_products_without_warnings(
    baseline: Baseline,
    new_reference_hz: float,
    expected_intercept: float | None,
    should_reject: bool,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if should_reject:
            with pytest.raises(BaselineRebaseError, match="not representable"):
                rebase_baseline(baseline, new_reference_hz)
        else:
            rebased = rebase_baseline(baseline, new_reference_hz)
            assert rebased.intercept == expected_intercept


@pytest.mark.parametrize(
    ("baseline", "reference"),
    [(object(), 0.0), (Baseline(1.0, 0.0), object()), (Baseline(1.0, 0.0), True)],
)
def test_rebase_baseline_rejects_wrong_argument_types(
    baseline: object, reference: object
) -> None:
    with pytest.raises(TypeError):
        rebase_baseline(baseline, reference)  # type: ignore[arg-type]


def test_rebase_baseline_structures_extreme_integral_reference_failure() -> None:
    with pytest.raises(ValueError, match="new_reference_hz must be finite"):
        rebase_baseline(Baseline(1.0, 0.0), 10**10000)


def test_warm_start_compatibility_codes_are_exactly_closed() -> None:
    assert get_args(WarmStartCompatibilityCode) == (
        "baseline_rebase_unrepresentable",
        "center_outside_sweep",
        "center_separation_incompatible",
        "resonance_bounds_incompatible",
        "parameterization_unrepresentable",
    )


def test_prepare_warm_start_preserves_prior_parameters_and_rebases_only_baseline(
) -> None:
    prior = _successful_prior(
        baseline=Baseline(1.0, 49.0e6, 2.0e-11, 0.0)
    )
    configuration = _configuration()
    preflight = _fit_preflight(configuration)

    preparation = prepare_warm_start(prior, configuration, preflight)

    assert preparation.rejection_code is None
    assert preparation.message is None
    assert preparation.guess is not None
    assert preparation.guess is not prior.initial_guess
    assert preparation.guess.resonances is not prior.resonance_estimates
    assert preparation.guess.resonances == prior.resonance_estimates
    assert preparation.guess.baseline == Baseline(1.00002, 50.0e6, 2.0e-11)
    assert preparation.guess.baseline is not prior.baseline_estimate
    for prepared, previous in zip(
        preparation.guess.resonances, prior.resonance_estimates, strict=True
    ):
        assert prepared is not previous
        assert prepared.resonance_id == previous.resonance_id
        assert prepared.center_hz == previous.center_hz
        assert prepared.fwhm_hz == previous.fwhm_hz
        assert prepared.amplitude == previous.amplitude
        assert prepared.eta == previous.eta
    validate_initial_guess(preparation.guess, configuration, preflight)


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("model", "resonance_bounds_incompatible"),
        ("degree", "resonance_bounds_incompatible"),
        ("ids", "resonance_bounds_incompatible"),
        ("width_below", "resonance_bounds_incompatible"),
        ("width_above", "resonance_bounds_incompatible"),
        ("amplitude_above", "resonance_bounds_incompatible"),
        ("baseline", "baseline_rebase_unrepresentable"),
        ("outside", "center_outside_sweep"),
        ("separation", "center_separation_incompatible"),
        ("parameterization", "parameterization_unrepresentable"),
    ],
)
def test_prepare_warm_start_returns_each_closed_compatibility_code(
    case: str, expected_code: str
) -> None:
    configuration = _configuration()
    preflight = _fit_preflight(configuration)
    prior = _successful_prior()
    if case == "model":
        prior = _successful_prior(model_kind="lorentzian")
    elif case == "degree":
        prior = _successful_prior(baseline_degree=2)
    elif case == "ids":
        prior = _successful_prior(
            resonance_ids=("other", *configuration.resonance_ids[1:])
        )
    elif case in {"width_below", "width_above", "amplitude_above"}:
        resonances = list(prior.resonance_estimates)
        changes = {
            "width_below": {"fwhm_hz": 1.0e5},
            "width_above": {"fwhm_hz": 9.0e6},
            "amplitude_above": {"amplitude": 0.30},
        }
        resonances[0] = replace(resonances[0], **changes[case])
        prior = _successful_prior(resonances=tuple(resonances))
    elif case == "baseline":
        maximum = np.finfo(np.float64).max
        prior = _successful_prior(baseline=Baseline(1.0, -maximum, 1.0))
        preflight = FitPreflight(
            34,
            1,
            np.nextafter(maximum, -np.inf),
            maximum,
            np.nextafter(maximum, -np.inf) / 2.0 + maximum / 2.0,
            maximum / 2.0 - np.nextafter(maximum, -np.inf) / 2.0,
            1.0,
            0.2,
        )
    elif case == "outside":
        preflight = _fit_preflight(
            configuration,
            _fit_ready_sweep(frequency_min_hz=10.0e6 + 1.0),
        )
    elif case == "separation":
        resonances = list(prior.resonance_estimates)
        resonances[1] = replace(
            resonances[1],
            center_hz=resonances[0].center_hz
            + configuration.min_center_separation_hz
            - 1.0,
        )
        prior = _successful_prior(resonances=tuple(resonances))
    elif case == "parameterization":
        configuration = _configuration(min_center_separation_hz=5.0)
        centers = 1.0e16 + 100.0 * np.arange(8)
        resonances = tuple(
            replace(item, center_hz=center)
            for item, center in zip(
                prior.resonance_estimates, centers, strict=True
            )
        )
        prior = _successful_prior(resonances=resonances)
        preflight = _fit_preflight(
            configuration,
            _fit_ready_sweep(
                frequency_min_hz=1.0e16 - 100.0,
                frequency_max_hz=1.0e16 + 800.0,
            ),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        preparation = prepare_warm_start(prior, configuration, preflight)

    assert preparation.guess is None
    assert preparation.rejection_code == expected_code
    assert preparation.rejection_code != "age_limit_exceeded"
    assert isinstance(preparation.message, str)
    assert preparation.message


def test_prepare_warm_start_accepts_inclusive_public_boundaries() -> None:
    configuration = _configuration()
    resonances = list(_successful_prior().resonance_estimates)
    resonances[0] = replace(
        resonances[0],
        center_hz=0.0,
        fwhm_hz=configuration.min_fwhm_hz,
        amplitude=0.0,
        eta=0.0,
    )
    resonances[-1] = replace(
        resonances[-1],
        center_hz=100.0e6,
        fwhm_hz=configuration.max_fwhm_hz,
        amplitude=configuration.max_amplitude,
        eta=1.0,
    )

    preparation = prepare_warm_start(
        _successful_prior(resonances=tuple(resonances)),
        configuration,
        _fit_preflight(configuration),
    )

    assert preparation.guess is not None


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("bounds_before_rebase", "resonance_bounds_incompatible"),
        ("rebase_before_outside", "baseline_rebase_unrepresentable"),
        ("outside_before_separation", "center_outside_sweep"),
        ("separation_before_parameterization", "center_separation_incompatible"),
    ],
)
def test_prepare_warm_start_uses_exact_first_failure_precedence(
    case: str, expected_code: str
) -> None:
    configuration = _configuration()
    prior = _successful_prior()
    preflight = _fit_preflight(configuration)
    if case == "bounds_before_rebase":
        resonances = list(prior.resonance_estimates)
        resonances[0] = replace(resonances[0], fwhm_hz=9.0e6)
        prior = _successful_prior(
            resonances=tuple(resonances),
            baseline=Baseline(1.0, -np.finfo(np.float64).max, 1.0),
        )
        maximum = np.finfo(np.float64).max
        preflight = FitPreflight(
            34,
            1,
            np.nextafter(maximum, -np.inf),
            maximum,
            np.nextafter(maximum, -np.inf) / 2.0 + maximum / 2.0,
            maximum / 2.0 - np.nextafter(maximum, -np.inf) / 2.0,
            1.0,
            0.2,
        )
    elif case == "rebase_before_outside":
        maximum = np.finfo(np.float64).max
        prior = _successful_prior(baseline=Baseline(1.0, -maximum, 1.0))
        preflight = FitPreflight(
            34,
            1,
            np.nextafter(maximum, -np.inf),
            maximum,
            np.nextafter(maximum, -np.inf) / 2.0 + maximum / 2.0,
            maximum / 2.0 - np.nextafter(maximum, -np.inf) / 2.0,
            1.0,
            0.2,
        )
    elif case == "outside_before_separation":
        resonances = list(prior.resonance_estimates)
        resonances[1] = replace(
            resonances[1],
            center_hz=resonances[0].center_hz
            + configuration.min_center_separation_hz
            - 1.0,
        )
        prior = _successful_prior(resonances=tuple(resonances))
        preflight = _fit_preflight(
            configuration,
            _fit_ready_sweep(frequency_min_hz=10.0e6 + 1.0),
        )
    else:
        configuration = _configuration(min_center_separation_hz=101.0)
        centers = 1.0e16 + 100.0 * np.arange(8)
        resonances = tuple(
            replace(item, center_hz=center)
            for item, center in zip(
                prior.resonance_estimates, centers, strict=True
            )
        )
        prior = _successful_prior(resonances=resonances)
        preflight = _fit_preflight(
            configuration,
            _fit_ready_sweep(
                frequency_min_hz=1.0e16 - 100.0,
                frequency_max_hz=1.0e16 + 800.0,
            ),
        )

    preparation = prepare_warm_start(prior, configuration, preflight)

    assert preparation.rejection_code == expected_code


@pytest.mark.parametrize(
    ("guess", "code", "message"),
    [
        (_valid_guess(), None, "unexpected"),
        (None, None, None),
        (_valid_guess(), "center_outside_sweep", "both"),
        (None, "age_limit_exceeded", "age belongs to wrapper"),
        (None, "unknown", "unknown"),
        (None, "center_outside_sweep", ""),
        (None, "center_outside_sweep", None),
    ],
)
def test_warm_start_preparation_rejects_invalid_exclusive_states(
    guess: FitInitialGuess | None, code: object, message: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        WarmStartPreparation(guess, code, message)  # type: ignore[arg-type]


def test_warm_start_preparation_is_frozen_slotted_and_defensive() -> None:
    guess = _valid_guess()

    preparation = WarmStartPreparation(guess, None, None)

    assert preparation.guess == guess
    assert preparation.guess is not guess
    assert not hasattr(preparation, "__dict__")
    with pytest.raises(FrozenInstanceError):
        preparation.message = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("argument_position", [0, 1, 2])
def test_prepare_warm_start_rejects_wrong_argument_types(
    argument_position: int,
) -> None:
    arguments: list[object] = [
        _successful_prior(),
        _configuration(),
        _fit_preflight(),
    ]
    arguments[argument_position] = object()
    with pytest.raises(TypeError):
        prepare_warm_start(*arguments)  # type: ignore[arg-type]


def test_prepare_warm_start_propagates_unsuccessful_prior_value_error() -> None:
    with pytest.raises(ValueError, match="successful prior"):
        prepare_warm_start(
            _optimization_failure(), _configuration(), _fit_preflight()
        )


@pytest.mark.parametrize("helper", ["rebase_baseline", "validate_initial_guess"])
def test_prepare_warm_start_propagates_unexpected_plain_value_errors(
    monkeypatch: pytest.MonkeyPatch, helper: str
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise ValueError("injected unexpected value error")

    monkeypatch.setattr(f"odmr_bench.estimators.preparation.{helper}", unexpected)
    with pytest.raises(
        ValueError, match=r"^injected unexpected value error$"
    ) as caught:
        prepare_warm_start(_successful_prior(), _configuration(), _fit_preflight())
    assert type(caught.value) is ValueError


@pytest.mark.parametrize(
    ("helper", "error", "expected_code"),
    [
        (
            "rebase_baseline",
            BaselineRebaseError("typed rebase"),
            "baseline_rebase_unrepresentable",
        ),
        (
            "validate_initial_guess",
            InitialGuessCompatibilityError("typed parameterization"),
            "parameterization_unrepresentable",
        ),
    ],
)
def test_prepare_warm_start_translates_only_private_typed_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    helper: str,
    error: ValueError,
    expected_code: str,
) -> None:
    def deliberate(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(f"odmr_bench.estimators.preparation.{helper}", deliberate)

    preparation = prepare_warm_start(
        _successful_prior(), _configuration(), _fit_preflight()
    )

    assert preparation.guess is None
    assert preparation.rejection_code == expected_code
    assert preparation.message == str(error)
