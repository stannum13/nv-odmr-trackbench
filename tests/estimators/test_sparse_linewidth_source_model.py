"""Regression tests for the sparse estimator's bound source model seam."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import odmr_bench.estimators.two_point_calibration as calibration_module
from odmr_bench.estimators.two_point_calibration import (
    _evaluate_bound_source_model,
)
from odmr_bench.models import Baseline
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_fit_configuration,
    make_legal_source_fit,
)


def _profile(
    frequency_hz: np.ndarray, center_hz: float, fwhm_hz: float, eta: float
) -> np.ndarray:
    reduced = (frequency_hz - center_hz) / fwhm_hz
    return np.asarray(
        eta / (1.0 + 4.0 * reduced * reduced)
        + (1.0 - eta)
        * np.exp(-4.0 * np.log(2.0) * reduced * reduced),
        dtype=np.float64,
    )


def _independently_ordered_model(
    source: object,
    frequency_hz: float | np.ndarray,
    target_resonance_id: str,
    *,
    center_hz: float,
    fwhm_hz: float,
    amplitude: float,
    baseline_offset: float,
) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    fit = source.source_fit
    result = np.asarray(fit.baseline_estimate.evaluate(frequency), dtype=np.float64)
    result = result + baseline_offset
    for resonance in fit.resonance_estimates:
        target = resonance.resonance_id == target_resonance_id
        result = result - (
            amplitude if target else resonance.amplitude
        ) * _profile(
            frequency,
            center_hz if target else resonance.center_hz,
            fwhm_hz if target else resonance.fwhm_hz,
            resonance.eta,
        )
    return np.asarray(result, dtype=np.float64)


def _source_with_model(
    *,
    model_kind: str,
    quadratic: bool,
) -> object:
    configuration = replace(
        make_legal_fit_configuration(),
        model_kind=model_kind,
        baseline_degree=2 if quadratic else 1,
    )
    fit = make_legal_source_fit(
        replace(configuration, model_kind="pseudo_voigt", baseline_degree=1)
    )
    resonances = tuple(
        replace(resonance, eta=1.0)
        if model_kind == "lorentzian"
        else resonance
        for resonance in fit.resonance_estimates
    )
    baseline = Baseline(
        intercept=1.07,
        reference_hz=2.88e9,
        slope_per_hz=2.3e-11,
        quadratic_per_hz2=1.9e-20 if quadratic else 0.0,
    )
    fit = replace(
        fit,
        model_kind=model_kind,
        baseline_degree=2 if quadratic else 1,
        resonance_estimates=resonances,
        baseline_estimate=baseline,
        initial_guess=replace(
            fit.initial_guess,
            resonances=resonances,
            baseline=baseline,
        ),
        jacobian_rank=(3 if quadratic else 2)
        + 8 * (3 if model_kind == "lorentzian" else 4),
    )
    return make_legal_caller_asserted_source(
        source_fit=fit,
        fit_configuration=configuration,
    )


def _evaluate_default_bound_source_model(
    frequency_hz: float | np.ndarray,
    **overrides: object,
) -> np.ndarray:
    source = make_legal_caller_asserted_source()
    arguments: dict[str, object] = {
        "center_hz": 2.761e9,
        "fwhm_hz": 1.7e6,
        "amplitude": 0.018,
        "baseline_offset": 0.003,
    }
    arguments.update(overrides)
    return _evaluate_bound_source_model(
        frequency_hz,
        source,
        "r0",
        **arguments,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("frequency_hz", (2.76e9, np.array([2.76e9, 2.77e9])))
@pytest.mark.parametrize(
    "parameter_name",
    ("center_hz", "fwhm_hz", "amplitude", "baseline_offset"),
)
@pytest.mark.parametrize("invalid_value", (np.nan, np.inf, -np.inf))
def test_bound_source_model_rejects_nonfinite_scalar_parameters_before_arithmetic(
    frequency_hz: float | np.ndarray,
    parameter_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{parameter_name} must be finite"):
        _evaluate_default_bound_source_model(
            frequency_hz,
            **{parameter_name: invalid_value},
        )


@pytest.mark.parametrize(
    ("parameter_name", "invalid_value"),
    tuple(
        (parameter_name, invalid_value)
        for parameter_name in (
            "center_hz",
            "fwhm_hz",
            "amplitude",
            "baseline_offset",
        )
        for invalid_value in (True, 1.0 + 2.0j, np.array([1.0]))
    ),
)
def test_bound_source_model_rejects_non_scalar_parameter_types(
    parameter_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError, match=rf"{parameter_name} must be a real scalar"):
        _evaluate_default_bound_source_model(
            np.array([2.76e9, 2.77e9]),
            **{parameter_name: invalid_value},
        )


@pytest.mark.parametrize(
    ("parameter_name", "invalid_value", "message"),
    (
        ("fwhm_hz", 0.0, "fwhm_hz must be positive"),
        ("fwhm_hz", -1.0, "fwhm_hz must be positive"),
        ("amplitude", -np.finfo(np.float64).eps, "amplitude must be non-negative"),
    ),
)
def test_bound_source_model_enforces_target_width_and_amplitude_domains(
    parameter_name: str,
    invalid_value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _evaluate_default_bound_source_model(
            2.76e9,
            **{parameter_name: invalid_value},
        )


def test_bound_source_model_changes_only_target_and_constant_offset() -> None:
    source = make_legal_caller_asserted_source()
    frequencies = np.array([2.86e9, 2.87e9, 2.88e9])
    actual = _evaluate_bound_source_model(
        frequencies,
        source,
        "r0",
        center_hz=2.861e9,
        fwhm_hz=8.0e6,
        amplitude=0.02,
        baseline_offset=0.003,
    )
    expected = _independently_ordered_model(
        source,
        frequencies,
        "r0",
        center_hz=2.861e9,
        fwhm_hz=8.0e6,
        amplitude=0.02,
        baseline_offset=0.003,
    )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("model_kind", "quadratic"),
    (("lorentzian", False), ("pseudo_voigt", True)),
)
@pytest.mark.parametrize(
    "frequency_hz",
    (2.760_000_000e9, np.array([2.74e9, 2.86e9, 3.02e9])),
)
def test_bound_source_model_preserves_source_order_for_scalar_and_vector_inputs(
    model_kind: str,
    quadratic: bool,
    frequency_hz: float | np.ndarray,
) -> None:
    source = _source_with_model(model_kind=model_kind, quadratic=quadratic)
    target = source.source_fit.resonance_estimates[3]
    actual = _evaluate_bound_source_model(
        frequency_hz,
        source,
        target.resonance_id,
        center_hz=target.center_hz + 123_456.0,
        fwhm_hz=target.fwhm_hz * 1.17,
        amplitude=target.amplitude * 0.91,
        baseline_offset=-0.004,
    )
    expected = _independently_ordered_model(
        source,
        frequency_hz,
        target.resonance_id,
        center_hz=target.center_hz + 123_456.0,
        fwhm_hz=target.fwhm_hz * 1.17,
        amplitude=target.amplitude * 0.91,
        baseline_offset=-0.004,
    )
    assert actual.dtype == np.float64
    assert np.array_equal(actual, expected)


def test_bound_source_model_uses_the_immutable_fit_tuple_order() -> None:
    source = make_legal_caller_asserted_source()
    reordered_source = SimpleNamespace(
        source_fit=SimpleNamespace(
            baseline_estimate=source.source_fit.baseline_estimate,
            resonance_estimates=tuple(
                reversed(source.source_fit.resonance_estimates)
            ),
        )
    )
    frequency_hz = np.array([2.771e9, 2.876e9, 2.981e9])
    actual = _evaluate_bound_source_model(
        frequency_hz,
        reordered_source,
        "r4",
        center_hz=2.899e9,
        fwhm_hz=1.8e6,
        amplitude=0.015,
        baseline_offset=0.002,
    )
    expected = _independently_ordered_model(
        reordered_source,
        frequency_hz,
        "r4",
        center_hz=2.899e9,
        fwhm_hz=1.8e6,
        amplitude=0.015,
        baseline_offset=0.002,
    )
    assert np.array_equal(actual, expected)


def test_bound_source_model_evaluates_baseline_once_before_source_ordered_dips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_legal_caller_asserted_source()
    calls: list[object] = []
    original_evaluate = Baseline.evaluate

    def counting_evaluate(self: Baseline, frequency_hz: object) -> np.ndarray:
        calls.append(frequency_hz)
        return original_evaluate(self, frequency_hz)

    monkeypatch.setattr(Baseline, "evaluate", counting_evaluate)
    frequency_hz = np.array([2.76e9, 2.82e9])
    _evaluate_bound_source_model(
        frequency_hz,
        source,
        "r2",
        center_hz=2.829e9,
        fwhm_hz=1.1e6,
        amplitude=0.018,
        baseline_offset=0.0,
    )
    assert calls == [frequency_hz]


def test_bound_source_model_does_not_pre_sum_background_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_legal_caller_asserted_source()
    frequency_hz = np.array([2.74e9, 2.89e9, 3.02e9])
    original_profile = calibration_module.np.exp
    calls: list[np.ndarray] = []

    def counted_exp(value: object) -> np.ndarray:
        result = original_profile(value)
        calls.append(np.asarray(value))
        return result

    monkeypatch.setattr(calibration_module.np, "exp", counted_exp)
    _evaluate_bound_source_model(
        frequency_hz,
        source,
        "r1",
        center_hz=2.793e9,
        fwhm_hz=1.9e6,
        amplitude=0.019,
        baseline_offset=0.0,
    )
    assert len(calls) == len(source.source_fit.resonance_estimates)
