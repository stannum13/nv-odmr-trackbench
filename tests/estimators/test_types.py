"""Contract tests for immutable estimator value objects."""

from __future__ import annotations

import math
import warnings
from dataclasses import FrozenInstanceError
from typing import get_args

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from odmr_bench.estimators import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    FitUncertainty,
    InitializationDiagnostics,
    SpectrumFitResult,
    SweepEstimate,
    SweepFitAttempt,
    SweepStartKind,
    WarmStartDisposition,
    WarmStartRejectionCode,
    WarmSweepEstimate,
)
from odmr_bench.models import Baseline, Resonance


def test_warm_sweep_public_names_are_importable() -> None:
    assert get_args(SweepStartKind) == ("preflight", "cold", "warm")
    assert get_args(WarmStartDisposition) == (
        "no_successful_prior",
        "used",
        "rejected_age",
        "rejected_compatibility",
        "not_applicable_preflight",
    )
    assert get_args(WarmStartRejectionCode) == (
        "age_limit_exceeded",
        "baseline_rebase_unrepresentable",
        "center_outside_sweep",
        "center_separation_incompatible",
        "resonance_bounds_incompatible",
        "parameterization_unrepresentable",
    )
    assert SweepFitAttempt is not None
    assert WarmSweepEstimate is not None


def _valid_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linspace(2.74e9, 3.02e9, 128),
        np.linspace(0.98, 1.02, 128),
    )


def test_complete_sweep_copies_freezes_and_canonicalizes_inputs() -> None:
    frequency_hz, fluorescence = _valid_arrays()
    sweep = CompleteSweep(
        frequency_hz=frequency_hz,
        fluorescence=fluorescence,
        last_sequence_index=np.int64(127),
        last_timestamp_s=np.float32(2.5),
        total_integration_time_s=np.float32(1.25),
        total_nominal_exposure_photons=np.float32(2.0e6),
    )

    frequency_hz[:] = 0.0
    fluorescence[:] = 0.0

    assert sweep.frequency_hz.dtype == np.float64
    assert sweep.fluorescence.dtype == np.float64
    assert not sweep.frequency_hz.flags.writeable
    assert not sweep.fluorescence.flags.writeable
    assert sweep.frequency_hz[0] == pytest.approx(2.74e9)
    assert sweep.fluorescence[0] == pytest.approx(0.98)
    assert sweep.last_sequence_index == 127
    assert type(sweep.last_timestamp_s) is float
    assert type(sweep.total_integration_time_s) is float
    assert type(sweep.total_nominal_exposure_photons) is float


def test_complete_sweep_leaves_unavailable_completion_metadata_as_none() -> None:
    frequency_hz, fluorescence = _valid_arrays()
    sweep = CompleteSweep(frequency_hz, fluorescence)

    assert sweep.last_sequence_index is None
    assert sweep.last_timestamp_s is None
    assert sweep.total_integration_time_s is None
    assert sweep.total_nominal_exposure_photons is None


@pytest.mark.parametrize(
    ("frequency_hz", "fluorescence"),
    [
        (np.array([]), np.array([])),
        (np.array([1.0]), np.array([0.9])),
        (np.ones((2, 2)), np.ones(4)),
        (np.ones(4), np.ones((2, 2))),
        (np.arange(4.0), np.ones(3)),
        (np.array([1.0, np.nan]), np.ones(2)),
        (np.array([1.0, 2.0]), np.array([1.0, np.inf])),
        (np.array([1.0, 1.0]), np.ones(2)),
        (np.array([2.0, 1.0]), np.ones(2)),
    ],
)
def test_complete_sweep_rejects_invalid_arrays(
    frequency_hz: np.ndarray, fluorescence: np.ndarray
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CompleteSweep(frequency_hz, fluorescence)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("last_sequence_index", True),
        ("last_sequence_index", -1),
        ("last_timestamp_s", True),
        ("last_timestamp_s", -0.1),
        ("total_integration_time_s", True),
        ("total_integration_time_s", 0.0),
        ("total_nominal_exposure_photons", True),
        ("total_nominal_exposure_photons", -1.0),
    ],
)
def test_complete_sweep_rejects_invalid_metadata(keyword: str, value: object) -> None:
    frequency_hz, fluorescence = _valid_arrays()
    with pytest.raises((TypeError, ValueError)):
        CompleteSweep(frequency_hz, fluorescence, **{keyword: value})


def test_complete_sweep_preserves_all_samples_without_sorting_or_deduplication() -> (
    None
):
    frequency_hz, fluorescence = _valid_arrays()
    sweep = CompleteSweep(frequency_hz, fluorescence)

    assert_array_equal(sweep.frequency_hz, frequency_hz)
    assert_array_equal(sweep.fluorescence, fluorescence)


@pytest.mark.parametrize("complex_field", ["frequency_hz", "fluorescence"])
def test_complete_sweep_rejects_complex_arrays_without_silent_projection(
    complex_field: str,
) -> None:
    frequency_hz, fluorescence = _valid_arrays()
    values: dict[str, np.ndarray] = {
        "frequency_hz": frequency_hz,
        "fluorescence": fluorescence,
    }
    values[complex_field] = values[complex_field].astype(np.complex128) + 1j

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        with pytest.raises(TypeError):
            CompleteSweep(**values)


def _configuration(**overrides: object) -> FitConfiguration:
    values: dict[str, object] = {
        "model_kind": "pseudo_voigt",
        "baseline_degree": 1,
        "resonance_ids": tuple(f"r{i}" for i in range(8)),
        "min_fwhm_hz": 2.0e5,
        "max_fwhm_hz": 8.0e6,
        "max_amplitude": 0.25,
        "min_resolved_amplitude": 1.0e-4,
        "min_amplitude_significance": 5.0,
        "min_center_separation_hz": 1.0e6,
        "savgol_window": 11,
        "savgol_polyorder": 2,
        "relative_prominence": 0.01,
        "allow_fallback": False,
        "max_nfev": 4000,
        "rank_rtol": 1.0e-10,
        "min_baseline_sse_improvement": 1.0e-4,
    }
    values.update(overrides)
    return FitConfiguration(**values)  # type: ignore[arg-type]


def test_fit_configuration_canonicalizes_the_oracle_constraints() -> None:
    configuration = _configuration(
        baseline_degree=np.int64(2),
        resonance_ids=[f"r{i}" for i in range(8)],
        min_fwhm_hz=np.float32(2.0e5),
        allow_fallback=np.bool_(False),
    )

    assert configuration.model_kind == "pseudo_voigt"
    assert configuration.baseline_degree == 2
    assert configuration.resonance_ids == tuple(f"r{i}" for i in range(8))
    assert type(configuration.min_fwhm_hz) is float
    assert type(configuration.min_amplitude_significance) is float
    assert type(configuration.allow_fallback) is bool


def test_fit_configuration_pins_default_amplitude_significance() -> None:
    assert FitConfiguration().min_amplitude_significance == 5.0


def test_fit_configuration_rejects_unrepresentable_amplitude_significance() -> None:
    with pytest.raises(ValueError, match="min_amplitude_significance must be finite"):
        FitConfiguration(min_amplitude_significance=10**10000)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_kind": "gaussian"},
        {"model_kind": True},
        {"baseline_degree": 0},
        {"baseline_degree": True},
        {"resonance_ids": tuple(f"r{i}" for i in range(7))},
        {"resonance_ids": ("r0",) * 8},
        {"resonance_ids": ("", *tuple(f"r{i}" for i in range(7)))},
        {"min_fwhm_hz": True},
        {"min_fwhm_hz": 0.0},
        {"max_fwhm_hz": 2.0e5},
        {"max_amplitude": 0.0},
        {"max_amplitude": True},
        {"min_resolved_amplitude": 0.0},
        {"min_resolved_amplitude": 0.251},
        {"min_amplitude_significance": True},
        {"min_amplitude_significance": 0.0},
        {"min_amplitude_significance": np.inf},
        {"min_center_separation_hz": 0.0},
        {"savgol_window": 10},
        {"savgol_window": 3},
        {"savgol_window": True},
        {"savgol_polyorder": 11},
        {"savgol_polyorder": -1},
        {"savgol_polyorder": True},
        {"relative_prominence": 0.0},
        {"relative_prominence": 1.1},
        {"rank_rtol": 0.0},
        {"rank_rtol": 1.0},
        {"min_baseline_sse_improvement": -0.1},
        {"min_baseline_sse_improvement": 1.0},
        {"allow_fallback": 1},
        {"max_nfev": 0},
        {"max_nfev": True},
    ],
)
def test_fit_configuration_rejects_invalid_constraints(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _configuration(**overrides)


def _resonances(*, signed_first_center: bool = False) -> tuple[Resonance, ...]:
    return tuple(
        Resonance(
            resonance_id=f"r{i}",
            center_hz=(-2.76e9 if signed_first_center and i == 0 else 2.76e9 + i * 2e6),
            fwhm_hz=1.0e6 + i * 1e5,
            amplitude=0.01 + i * 1e-3,
            eta=0.5,
        )
        for i in range(8)
    )


def _initial_guess() -> FitInitialGuess:
    return FitInitialGuess(
        resonances=_resonances(),
        baseline=Baseline(intercept=1.0, reference_hz=2.88e9),
    )


def test_fit_initial_guess_requires_eight_ordered_resonances_and_a_baseline() -> None:
    initial_guess = _initial_guess()

    assert initial_guess.resonances == _resonances()
    assert initial_guess.baseline == Baseline(intercept=1.0, reference_hz=2.88e9)

    with pytest.raises(ValueError):
        FitInitialGuess(_resonances()[:-1], initial_guess.baseline)
    with pytest.raises(ValueError):
        FitInitialGuess(_resonances()[::-1], initial_guess.baseline)
    with pytest.raises(TypeError):
        FitInitialGuess(_resonances(), object())  # type: ignore[arg-type]


def _diagnostics(
    source: str = "detected", **overrides: object
) -> InitializationDiagnostics:
    values: dict[str, object] = {
        "source": source,
        "candidate_count": 8 if source == "detected" else 0,
        "selected_indices": tuple(range(8)) if source == "detected" else (),
        "used_fallback": source == "fallback",
        "messages": ["deterministic discovery"],
    }
    values.update(overrides)
    return InitializationDiagnostics(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("source", ["detected", "fallback", "user", "none"])
def test_initialization_diagnostics_enforces_the_state_table(source: str) -> None:
    diagnostics = _diagnostics(source)

    assert isinstance(diagnostics.selected_indices, tuple)
    assert isinstance(diagnostics.messages, tuple)
    assert diagnostics.source == source


def test_none_diagnostics_can_retain_the_detected_candidate_count() -> None:
    diagnostics = _diagnostics("none", candidate_count=3)

    assert diagnostics.candidate_count == 3


@pytest.mark.parametrize(
    ("source", "overrides"),
    [
        ("invalid", {}),
        ("detected", {"candidate_count": 7}),
        ("detected", {"selected_indices": (0,) * 8}),
        ("detected", {"used_fallback": True}),
        ("fallback", {"selected_indices": (0,)}),
        ("fallback", {"used_fallback": False}),
        ("user", {"candidate_count": 1}),
        ("user", {"selected_indices": (0,)}),
        ("none", {"used_fallback": True}),
        ("none", {"selected_indices": (0,)}),
        ("none", {"candidate_count": True}),
        ("none", {"messages": ("ok", 1)}),
    ],
)
def test_initialization_diagnostics_rejects_contradictory_states(
    source: str, overrides: dict[str, object]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _diagnostics(source, **overrides)


def _uncertainty(*, eta: object = None) -> FitUncertainty:
    return FitUncertainty(
        baseline_standard_errors=np.array([1e-4, 1e-12]),
        center_hz=np.full(8, 1e3),
        fwhm_hz=np.full(8, 1e2),
        amplitude=np.full(8, 1e-3),
        eta=eta,
    )


def test_fit_uncertainty_copies_freezes_and_validates_shapes() -> None:
    center_standard_errors = np.full(8, 1e3)
    uncertainty = FitUncertainty(
        baseline_standard_errors=np.array([1e-4, 1e-12]),
        center_hz=center_standard_errors,
        fwhm_hz=np.full(8, 1e2),
        amplitude=np.full(8, 1e-3),
        eta=np.full(8, 1e-2),
    )
    center_standard_errors[:] = 0.0

    assert uncertainty.center_hz[0] == 1e3
    assert not uncertainty.center_hz.flags.writeable
    assert uncertainty.eta is not None
    assert not uncertainty.eta.flags.writeable

    with pytest.raises(ValueError):
        FitUncertainty(np.ones(1), np.ones(8), np.ones(8), np.ones(8), None)
    with pytest.raises(ValueError):
        FitUncertainty(np.ones(2), np.ones(7), np.ones(8), np.ones(8), None)


@pytest.mark.parametrize(
    "complex_field",
    ["baseline_standard_errors", "center_hz", "fwhm_hz", "amplitude", "eta"],
)
def test_fit_uncertainty_rejects_complex_arrays_without_silent_projection(
    complex_field: str,
) -> None:
    values: dict[str, np.ndarray] = {
        "baseline_standard_errors": np.ones(2),
        "center_hz": np.ones(8),
        "fwhm_hz": np.ones(8),
        "amplitude": np.ones(8),
        "eta": np.ones(8),
    }
    values[complex_field] = values[complex_field].astype(np.complex128) + 1j

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        with pytest.raises(TypeError):
            FitUncertainty(**values)


@pytest.mark.parametrize(
    "method",
    ["bootstrap", "local_linearized_jacobian_covariance_v2"],
)
def test_fit_uncertainty_rejects_any_undeclared_method(method: str) -> None:
    with pytest.raises(ValueError, match="method"):
        FitUncertainty(
            baseline_standard_errors=np.ones(2),
            center_hz=np.ones(8),
            fwhm_hz=np.ones(8),
            amplitude=np.ones(8),
            eta=None,
            method=method,
        )


def _success_result(**overrides: object) -> SpectrumFitResult:
    values: dict[str, object] = {
        "success": True,
        "failure_code": None,
        "model_kind": "pseudo_voigt",
        "baseline_degree": 1,
        "resonance_estimates": _resonances(),
        "baseline_estimate": Baseline(intercept=1.0, reference_hz=2.88e9),
        "diagnostics": _diagnostics(),
        "initial_guess": _initial_guess(),
        "uncertainty": _uncertainty(eta=np.full(8, 1e-3)),
        "uncertainty_reason": None,
        "scipy_status": 1,
        "scipy_message": "converged",
        "nfev": 10,
        "cost": 0.01,
        "residual_rmse": 0.02,
        "residual_scale": 0.1,
        "degrees_of_freedom": 100,
        "jacobian_rank": 34,
    }
    values.update(overrides)
    return SpectrumFitResult(**values)  # type: ignore[arg-type]


def _attempt_result_overrides(outcome: str) -> dict[str, object]:
    if outcome == "success":
        return {}
    common: dict[str, object] = {
        "success": False,
        "failure_code": outcome,
        "resonance_estimates": (),
        "baseline_estimate": None,
        "uncertainty": None,
        "uncertainty_reason": "attempt did not produce an accepted fit",
        "scipy_status": 0 if outcome == "optimization_failed" else 1,
        "scipy_message": "stopped",
        "nfev": 1,
        "jacobian_rank": 10 if outcome == "quality_failed" else None,
    }
    return common


def _optimizer_result(start_kind: str, code: str | None) -> SpectrumFitResult:
    source = "user" if start_kind == "warm" else "detected"
    return _success_result(
        success=code is None,
        failure_code=code,
        resonance_estimates=_resonances() if code is None else (),
        baseline_estimate=(Baseline(1.0, 2.88e9) if code is None else None),
        diagnostics=_diagnostics(source),
        uncertainty=(_uncertainty(eta=np.full(8, 1e-3)) if code is None else None),
        uncertainty_reason=None if code is None else "attempt failed",
        scipy_status=0 if code == "optimization_failed" else 1,
        scipy_message="stopped",
        nfev=4 if code is None else 3,
        cost=0.01,
        residual_rmse=0.02,
        residual_scale=0.1,
        jacobian_rank=(
            34
            if code is None
            else None
            if code == "optimization_failed"
            else 10
        ),
    )


def _cold_initialization_failure(source: str = "none") -> SpectrumFitResult:
    return _success_result(
        success=False,
        failure_code="initialization_failed",
        resonance_estimates=(),
        baseline_estimate=None,
        diagnostics=_diagnostics(source),
        initial_guess=None,
        uncertainty=None,
        uncertainty_reason="initialization failed",
        scipy_status=None,
        scipy_message=None,
        nfev=0,
        cost=None,
        residual_rmse=None,
        residual_scale=0.1,
        jacobian_rank=None,
    )


def _preflight_result(code: str) -> SpectrumFitResult:
    return _success_result(
        success=False,
        failure_code=code,
        resonance_estimates=(),
        baseline_estimate=None,
        diagnostics=_diagnostics("none"),
        initial_guess=None,
        uncertainty=None,
        uncertainty_reason="preflight failed",
        scipy_status=None,
        scipy_message=None,
        nfev=0,
        cost=None,
        residual_rmse=None,
        residual_scale=None,
        degrees_of_freedom=0 if code == "insufficient_samples" else 100,
        jacobian_rank=None,
    )


@pytest.mark.parametrize(
    ("start_kind", "warm_source_update_index", "fit"),
    [
        ("preflight", None, _preflight_result("insufficient_samples")),
        ("preflight", None, _preflight_result("uninformative_sweep")),
        ("cold", None, _optimizer_result("cold", None)),
        ("cold", None, _optimizer_result("cold", "optimization_failed")),
        ("cold", None, _optimizer_result("cold", "quality_failed")),
        (
            "cold",
            None,
            _success_result(diagnostics=_diagnostics("fallback")),
        ),
        (
            "cold",
            None,
            _success_result(
                **{
                    **_attempt_result_overrides("optimization_failed"),
                    "diagnostics": _diagnostics("fallback"),
                }
            ),
        ),
        (
            "cold",
            None,
            _success_result(
                **{
                    **_attempt_result_overrides("quality_failed"),
                    "diagnostics": _diagnostics("fallback"),
                }
            ),
        ),
        ("cold", None, _cold_initialization_failure("detected")),
        ("cold", None, _cold_initialization_failure("fallback")),
        ("cold", None, _cold_initialization_failure("none")),
        ("warm", 0, _optimizer_result("warm", None)),
        ("warm", 0, _optimizer_result("warm", "optimization_failed")),
        ("warm", 0, _optimizer_result("warm", "quality_failed")),
    ],
)
def test_sweep_fit_attempt_accepts_every_legal_state(
    start_kind: str, warm_source_update_index: int | None, fit: SpectrumFitResult
) -> None:
    attempt = SweepFitAttempt(
        start_kind,  # type: ignore[arg-type]
        warm_source_update_index,
        fit,
        np.float32(0.125),
    )

    assert attempt.start_kind == start_kind
    assert attempt.warm_source_update_index == warm_source_update_index
    assert attempt.fit is fit
    assert type(attempt.cpu_time_s) is float


def test_sweep_fit_attempt_canonicalizes_numpy_source_and_is_frozen_slotted() -> None:
    attempt = SweepFitAttempt(
        "warm", np.int64(2), _optimizer_result("warm", None), np.float64(0.25)
    )

    assert attempt.warm_source_update_index == 2
    assert type(attempt.warm_source_update_index) is int
    assert type(attempt.cpu_time_s) is float
    assert not hasattr(attempt, "__dict__")
    with pytest.raises(FrozenInstanceError):
        attempt.cpu_time_s = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("start_kind", "warm_source_update_index", "fit", "cpu_time_s"),
    [
        ("unknown", None, _optimizer_result("cold", None), 0.1),
        ("cold", None, object(), 0.1),
        ("cold", None, _optimizer_result("cold", None), True),
        ("cold", None, _optimizer_result("cold", None), -0.1),
        ("cold", None, _optimizer_result("cold", None), np.inf),
        ("cold", 0, _optimizer_result("cold", None), 0.1),
        ("preflight", 0, _preflight_result("insufficient_samples"), 0.1),
        ("warm", None, _optimizer_result("warm", None), 0.1),
        ("warm", True, _optimizer_result("warm", None), 0.1),
        ("warm", -1, _optimizer_result("warm", None), 0.1),
        ("warm", 1.5, _optimizer_result("warm", None), 0.1),
        ("preflight", None, _optimizer_result("cold", None), 0.1),
        (
            "preflight",
            None,
            _optimizer_result("cold", "optimization_failed"),
            0.1,
        ),
        (
            "preflight",
            None,
            _optimizer_result("cold", "quality_failed"),
            0.1,
        ),
        ("preflight", None, _cold_initialization_failure(), 0.1),
        ("cold", None, _preflight_result("insufficient_samples"), 0.1),
        ("cold", None, _preflight_result("uninformative_sweep"), 0.1),
        ("warm", 0, _preflight_result("insufficient_samples"), 0.1),
        ("warm", 0, _preflight_result("uninformative_sweep"), 0.1),
        ("warm", 0, _cold_initialization_failure(), 0.1),
        ("warm", 0, _optimizer_result("cold", None), 0.1),
        (
            "warm",
            0,
            _success_result(diagnostics=_diagnostics("fallback")),
            0.1,
        ),
        ("cold", None, _optimizer_result("warm", None), 0.1),
    ],
)
def test_sweep_fit_attempt_rejects_invalid_state_or_provenance(
    start_kind: object,
    warm_source_update_index: object,
    fit: object,
    cpu_time_s: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SweepFitAttempt(
            start_kind,  # type: ignore[arg-type]
            warm_source_update_index,  # type: ignore[arg-type]
            fit,  # type: ignore[arg-type]
            cpu_time_s,  # type: ignore[arg-type]
        )


def _fit_attempt(
    start_kind: str, code: str | None = None, *, warm_source: int = 0
) -> SweepFitAttempt:
    if start_kind == "preflight":
        fit = _preflight_result(code or "insufficient_samples")
        source = None
    elif start_kind == "warm":
        fit = _optimizer_result("warm", code)
        source = warm_source
    else:
        fit = _optimizer_result("cold", code)
        source = None
    return SweepFitAttempt(start_kind, source, fit, 0.125)  # type: ignore[arg-type]


def _warm_estimate(
    *,
    attempts: object | None = None,
    update_index: object = 1,
    **overrides: object,
) -> WarmSweepEstimate:
    supplied_attempts: object = (
        [_fit_attempt("cold")] if attempts is None else attempts
    )
    try:
        attempt_tuple = tuple(supplied_attempts)  # type: ignore[arg-type]
    except TypeError:
        attempt_tuple = ()
    selected_fit = (
        attempt_tuple[-1].fit
        if attempt_tuple and isinstance(attempt_tuple[-1], SweepFitAttempt)
        else _optimizer_result("cold", None)
    )
    canonical_update = int(update_index) if not isinstance(update_index, bool) else 1
    active_fit = selected_fit if selected_fit.success else None
    values: dict[str, object] = {
        "update_index": update_index,
        "attempts": supplied_attempts,
        "warm_start_disposition": "no_successful_prior",
        "warm_start_rejection_code": None,
        "warm_start_message": None,
        "active_fit": active_fit,
        "active_source_update_index": canonical_update if active_fit else None,
        "estimate_age_submitted_observations": 0 if active_fit else None,
        "estimate_age_sequence_indices": 0 if active_fit else None,
        "estimate_age_s": 0.0 if active_fit else None,
        "observation_count": 32,
        "cumulative_observation_count": 64,
        "first_sequence_index": 32,
        "last_sequence_index": 63,
        "last_timestamp_s": 1.0,
        "total_integration_time_s": 0.25,
        "total_nominal_exposure_photons": 2.0e6,
        "cpu_time_s": 1.0,
    }
    values.update(overrides)
    return WarmSweepEstimate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("disposition", "attempts", "code", "message"),
    [
        (
            "not_applicable_preflight",
            [_fit_attempt("preflight", "insufficient_samples")],
            None,
            None,
        ),
        ("no_successful_prior", [_fit_attempt("cold")], None, None),
        (
            "rejected_age",
            [_fit_attempt("cold")],
            "age_limit_exceeded",
            "prior is too old",
        ),
        *[
            (
                "rejected_compatibility",
                [_fit_attempt("cold")],
                code,
                f"incompatible: {code}",
            )
            for code in (
                "baseline_rebase_unrepresentable",
                "center_outside_sweep",
                "center_separation_incompatible",
                "resonance_bounds_incompatible",
                "parameterization_unrepresentable",
            )
        ],
        ("used", [_fit_attempt("warm")], None, None),
        (
            "used",
            [
                _fit_attempt("warm", "quality_failed"),
                _fit_attempt("cold"),
            ],
            None,
            None,
        ),
    ],
)
def test_warm_sweep_estimate_accepts_every_disposition_attempt_row(
    disposition: str,
    attempts: list[SweepFitAttempt],
    code: str | None,
    message: str | None,
) -> None:
    estimate = _warm_estimate(
        attempts=attempts,
        warm_start_disposition=disposition,
        warm_start_rejection_code=code,
        warm_start_message=message,
    )

    attempts.clear()
    assert isinstance(estimate.attempts, tuple)
    assert estimate.warm_start_disposition == disposition
    assert estimate.warm_start_rejection_code == code
    assert estimate.warm_start_message == message


def test_warm_sweep_estimate_is_frozen_and_slotted() -> None:
    estimate = _warm_estimate()

    assert not hasattr(estimate, "__dict__")
    with pytest.raises(FrozenInstanceError):
        estimate.update_index = 2  # type: ignore[misc]


def test_warm_sweep_estimate_accepts_failed_warm_before_cold_recovery() -> None:
    estimate = _warm_estimate(
        attempts=[
            _fit_attempt("warm", "optimization_failed"),
            _fit_attempt("cold"),
        ],
        warm_start_disposition="used",
    )

    assert estimate.attempts[0].fit.failure_code == "optimization_failed"
    assert estimate.current_fit is estimate.attempts[1].fit


@pytest.mark.parametrize(
    ("disposition", "attempts", "code", "message"),
    [
        ("not_applicable_preflight", (_fit_attempt("cold"),), None, None),
        (
            "no_successful_prior",
            (_fit_attempt("preflight"),),
            None,
            None,
        ),
        ("no_successful_prior", (_fit_attempt("warm"),), None, None),
        ("rejected_age", (_fit_attempt("cold"),), None, None),
        (
            "rejected_age",
            (_fit_attempt("cold"),),
            "center_outside_sweep",
            "outside",
        ),
        (
            "rejected_compatibility",
            (_fit_attempt("cold"),),
            "age_limit_exceeded",
            "old",
        ),
        (
            "rejected_compatibility",
            (_fit_attempt("cold"),),
            "center_outside_sweep",
            None,
        ),
        ("used", (_fit_attempt("cold"),), None, None),
        ("used", (_fit_attempt("warm"), _fit_attempt("warm")), None, None),
        ("used", (_fit_attempt("warm"), _fit_attempt("cold")), None, None),
        (
            "used",
            (
                _fit_attempt("warm", "quality_failed"),
                _fit_attempt("cold"),
                _fit_attempt("cold"),
            ),
            None,
            None,
        ),
    ],
)
def test_warm_sweep_estimate_rejects_invalid_disposition_attempt_rows(
    disposition: str,
    attempts: tuple[SweepFitAttempt, ...],
    code: str | None,
    message: str | None,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _warm_estimate(
            attempts=attempts,
            warm_start_disposition=disposition,
            warm_start_rejection_code=code,
            warm_start_message=message,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"attempts": ()},
        {
            "attempts": (
                _fit_attempt("warm", "quality_failed"),
                _fit_attempt("cold"),
                _fit_attempt("cold"),
            ),
            "warm_start_disposition": "used",
        },
        {"attempts": (object(),)},
        {"attempts": "cold"},
        {
            "attempts": (_fit_attempt("warm", warm_source=1),),
            "warm_start_disposition": "used",
        },
        {
            "attempts": (_fit_attempt("warm", warm_source=2),),
            "warm_start_disposition": "used",
        },
        {"warm_start_disposition": "unknown"},
        {"warm_start_disposition": True},
        {
            "warm_start_rejection_code": "center_outside_sweep",
            "warm_start_message": "outside",
        },
        {
            "warm_start_disposition": "rejected_compatibility",
            "warm_start_rejection_code": "unknown",
            "warm_start_message": "bad",
        },
        {
            "warm_start_disposition": "rejected_compatibility",
            "warm_start_rejection_code": True,
            "warm_start_message": "bad",
        },
        {
            "warm_start_disposition": "rejected_age",
            "warm_start_rejection_code": "age_limit_exceeded",
            "warm_start_message": "",
        },
        {
            "warm_start_disposition": "rejected_age",
            "warm_start_rejection_code": "age_limit_exceeded",
            "warm_start_message": 1,
        },
        {"warm_start_message": "message without code"},
    ],
)
def test_warm_sweep_estimate_rejects_invalid_attempt_sequence_or_literals(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _warm_estimate(**overrides)


def _stale_warm_estimate(
    disposition: str = "used", **overrides: object
) -> WarmSweepEstimate:
    active_fit = _optimizer_result("cold", None)
    if disposition == "used":
        attempts = [_fit_attempt("warm", "quality_failed", warm_source=0)]
        code = None
        message = None
    elif disposition == "rejected_age":
        attempts = [_fit_attempt("cold", "optimization_failed")]
        code = "age_limit_exceeded"
        message = "prior is too old"
    elif disposition == "rejected_compatibility":
        attempts = [_fit_attempt("cold", "optimization_failed")]
        code = "center_outside_sweep"
        message = "center is outside the submitted sweep"
    elif disposition == "not_applicable_preflight":
        attempts = [_fit_attempt("preflight", "uninformative_sweep")]
        code = None
        message = None
    else:
        attempts = [_fit_attempt("cold", "optimization_failed")]
        code = None
        message = None
    values: dict[str, object] = {
        "attempts": attempts,
        "update_index": 2,
        "warm_start_disposition": disposition,
        "warm_start_rejection_code": code,
        "warm_start_message": message,
        "active_fit": active_fit,
        "active_source_update_index": 0,
        "estimate_age_submitted_observations": 64,
        "estimate_age_sequence_indices": 64,
        "estimate_age_s": 1.25,
        "observation_count": 32,
        "cumulative_observation_count": 96,
        "first_sequence_index": 64,
        "last_sequence_index": 95,
        "last_timestamp_s": 2.0,
    }
    values.update(overrides)
    return _warm_estimate(**values)


def test_warm_sweep_active_state_accepts_current_success_with_available_ages() -> None:
    attempt = _fit_attempt("cold")
    estimate = _warm_estimate(
        attempts=[attempt],
        active_fit=attempt.fit,
        active_source_update_index=1,
        estimate_age_submitted_observations=0,
        estimate_age_sequence_indices=0,
        estimate_age_s=0.0,
    )

    assert estimate.current_fit is attempt.fit
    assert estimate.active_fit is attempt.fit
    assert estimate.is_stale is False


def test_warm_sweep_active_state_accepts_current_success_without_optional_ages(
) -> None:
    attempt = _fit_attempt("cold")
    estimate = _warm_estimate(
        attempts=[attempt],
        active_fit=attempt.fit,
        active_source_update_index=1,
        estimate_age_submitted_observations=0,
        estimate_age_sequence_indices=None,
        estimate_age_s=None,
        first_sequence_index=None,
        last_sequence_index=None,
        last_timestamp_s=None,
    )

    assert estimate.current_fit is estimate.active_fit
    assert estimate.estimate_age_sequence_indices is None
    assert estimate.estimate_age_s is None


def test_warm_sweep_active_state_accepts_stale_success_after_current_failure() -> None:
    estimate = _stale_warm_estimate()

    assert estimate.current_fit.success is False
    assert estimate.active_fit is not None
    assert estimate.active_fit.success is True
    assert estimate.active_fit is not estimate.current_fit
    assert estimate.active_source_update_index == 0
    assert estimate.estimate_age_submitted_observations == 64
    assert estimate.estimate_age_sequence_indices == 64
    assert estimate.estimate_age_s == 1.25
    assert estimate.is_stale is True


def test_warm_sweep_active_state_accepts_failure_before_any_success() -> None:
    attempt = _fit_attempt("cold", "optimization_failed")
    estimate = _warm_estimate(
        update_index=0,
        attempts=[attempt],
        warm_start_disposition="no_successful_prior",
        active_fit=None,
        active_source_update_index=None,
        estimate_age_submitted_observations=None,
        estimate_age_sequence_indices=None,
        estimate_age_s=None,
        cumulative_observation_count=32,
        first_sequence_index=0,
        last_sequence_index=31,
    )

    assert estimate.current_fit is attempt.fit
    assert estimate.active_fit is None
    assert estimate.is_stale is False


@pytest.mark.parametrize("retain_stale", [False, True])
def test_warm_sweep_active_state_accepts_preflight_failure_with_optional_prior(
    retain_stale: bool,
) -> None:
    if retain_stale:
        estimate = _stale_warm_estimate("not_applicable_preflight")
    else:
        estimate = _warm_estimate(
            update_index=0,
            attempts=[_fit_attempt("preflight", "uninformative_sweep")],
            warm_start_disposition="not_applicable_preflight",
            cumulative_observation_count=32,
            first_sequence_index=0,
            last_sequence_index=31,
        )

    assert estimate.current_fit.failure_code == "uninformative_sweep"
    assert estimate.is_stale is retain_stale


@pytest.mark.parametrize(
    "disposition", ["used", "rejected_age", "rejected_compatibility"]
)
def test_warm_sweep_active_state_requires_stale_prior_for_failed_seeded_updates(
    disposition: str,
) -> None:
    if disposition == "used":
        attempts = [_fit_attempt("warm", "quality_failed")]
        code = None
        message = None
    elif disposition == "rejected_age":
        attempts = [_fit_attempt("cold", "quality_failed")]
        code = "age_limit_exceeded"
        message = "old"
    else:
        attempts = [_fit_attempt("cold", "quality_failed")]
        code = "center_outside_sweep"
        message = "outside"

    with pytest.raises(ValueError):
        _warm_estimate(
            attempts=attempts,
            warm_start_disposition=disposition,
            warm_start_rejection_code=code,
            warm_start_message=message,
        )


def test_warm_sweep_active_state_rejects_prior_on_failed_no_successful_prior() -> None:
    with pytest.raises(ValueError):
        _stale_warm_estimate("no_successful_prior")


def test_warm_sweep_active_state_joins_failed_used_to_warm_source() -> None:
    with pytest.raises(ValueError):
        _stale_warm_estimate("used", active_source_update_index=1)


def test_warm_sweep_active_state_rejects_unsuccessful_active_fit() -> None:
    with pytest.raises(ValueError):
        _stale_warm_estimate(
            active_fit=_optimizer_result("cold", "optimization_failed")
        )


def test_warm_sweep_active_state_requires_current_fit_identity() -> None:
    attempt = _fit_attempt("cold")
    equal_but_distinct_fit = _optimizer_result("cold", None)

    with pytest.raises(ValueError):
        _warm_estimate(
            attempts=[attempt],
            active_fit=equal_but_distinct_fit,
            active_source_update_index=1,
            estimate_age_submitted_observations=0,
            estimate_age_sequence_indices=0,
            estimate_age_s=0.0,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"active_fit": None},
        {"active_source_update_index": None},
        {"active_source_update_index": 0},
        {"estimate_age_submitted_observations": None},
        {"estimate_age_submitted_observations": 1},
        {"estimate_age_sequence_indices": None},
        {"estimate_age_sequence_indices": 1},
        {"estimate_age_s": None},
        {"estimate_age_s": 0.5},
        {
            "first_sequence_index": None,
            "last_sequence_index": None,
            "estimate_age_sequence_indices": 0,
        },
        {"last_timestamp_s": None, "estimate_age_s": 0.0},
    ],
)
def test_warm_sweep_active_state_rejects_current_success_contradictions(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _warm_estimate(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"active_source_update_index": None},
        {"active_source_update_index": -1},
        {"active_source_update_index": 2},
        {"active_source_update_index": True},
        {"active_source_update_index": 0.5},
        {"estimate_age_submitted_observations": None},
        {"estimate_age_submitted_observations": 0},
        {"estimate_age_submitted_observations": True},
        {"estimate_age_submitted_observations": 1.5},
        {"estimate_age_sequence_indices": None},
        {"estimate_age_sequence_indices": 0},
        {"estimate_age_sequence_indices": True},
        {"estimate_age_sequence_indices": 1.5},
        {"estimate_age_s": None},
        {"estimate_age_s": 0.0},
        {"estimate_age_s": True},
        {"estimate_age_s": np.inf},
        {
            "first_sequence_index": None,
            "last_sequence_index": None,
            "estimate_age_sequence_indices": 64,
        },
        {"last_timestamp_s": None, "estimate_age_s": 1.25},
    ],
)
def test_warm_sweep_active_state_rejects_stale_failure_contradictions(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _stale_warm_estimate(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"active_source_update_index": 0},
        {"estimate_age_submitted_observations": 1},
        {"estimate_age_sequence_indices": 1},
        {"estimate_age_s": 0.1},
    ],
)
def test_warm_sweep_active_state_rejects_ages_without_active_fit(
    overrides: dict[str, object],
) -> None:
    attempt = _fit_attempt("cold", "optimization_failed")
    with pytest.raises(ValueError):
        _warm_estimate(
            update_index=0,
            attempts=[attempt],
            warm_start_disposition="no_successful_prior",
            active_fit=None,
            cumulative_observation_count=32,
            first_sequence_index=0,
            last_sequence_index=31,
            **overrides,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"observation_count": 0},
        {"observation_count": -1},
        {"observation_count": True},
        {"observation_count": 1.5},
        {"cumulative_observation_count": -1},
        {"cumulative_observation_count": True},
        {"cumulative_observation_count": 1.5},
        {"cumulative_observation_count": 31},
        {"first_sequence_index": None},
        {"last_sequence_index": None},
        {"first_sequence_index": -1},
        {"last_sequence_index": -1},
        {"first_sequence_index": True},
        {"last_sequence_index": 63.5},
        {"first_sequence_index": 31},
        {"last_sequence_index": 62},
        {"last_timestamp_s": -0.1},
        {"last_timestamp_s": True},
        {"last_timestamp_s": np.inf},
        {"total_integration_time_s": 0.0},
        {"total_integration_time_s": True},
        {"total_integration_time_s": np.inf},
        {"total_nominal_exposure_photons": -1.0},
        {"total_nominal_exposure_photons": True},
        {"total_nominal_exposure_photons": np.inf},
        {"cpu_time_s": -0.1},
        {"cpu_time_s": True},
        {"cpu_time_s": np.inf},
        {"cpu_time_s": 0.1},
    ],
)
def test_warm_sweep_resources_reject_invalid_endpoints_or_scalars(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _warm_estimate(**overrides)


def test_warm_sweep_resources_require_cpu_at_least_fsum_of_attempts() -> None:
    warm = SweepFitAttempt(
        "warm", 0, _optimizer_result("warm", "quality_failed"), 0.1
    )
    cold = SweepFitAttempt("cold", None, _optimizer_result("cold", None), 0.2)
    attempt_sum = math.fsum(attempt.cpu_time_s for attempt in (warm, cold))

    with pytest.raises(ValueError):
        _warm_estimate(
            attempts=[warm, cold],
            warm_start_disposition="used",
            cpu_time_s=np.nextafter(attempt_sum, -np.inf),
        )


def test_warm_sweep_resources_canonicalize_numpy_scalars_and_derive_totals() -> None:
    warm = SweepFitAttempt(
        "warm",
        np.int64(0),
        _optimizer_result("warm", "quality_failed"),
        np.float32(0.125),
    )
    cold = SweepFitAttempt(
        "cold", None, _optimizer_result("cold", None), np.float32(0.25)
    )
    estimate = _warm_estimate(
        update_index=np.int64(1),
        attempts=[warm, cold],
        warm_start_disposition="used",
        active_fit=cold.fit,
        active_source_update_index=np.int64(1),
        estimate_age_submitted_observations=np.int64(0),
        estimate_age_sequence_indices=np.int64(0),
        estimate_age_s=np.float32(0.0),
        observation_count=np.int64(32),
        cumulative_observation_count=np.int64(64),
        first_sequence_index=np.int64(32),
        last_sequence_index=np.int64(63),
        last_timestamp_s=np.float32(1.0),
        total_integration_time_s=np.float32(0.25),
        total_nominal_exposure_photons=np.float32(2.0e6),
        cpu_time_s=np.float32(0.5),
    )

    assert estimate.current_fit is cold.fit
    assert estimate.total_nfev == 7
    assert type(estimate.total_nfev) is int
    assert estimate.is_stale is False
    for field_name in (
        "update_index",
        "active_source_update_index",
        "estimate_age_submitted_observations",
        "estimate_age_sequence_indices",
        "observation_count",
        "cumulative_observation_count",
        "first_sequence_index",
        "last_sequence_index",
    ):
        assert type(getattr(estimate, field_name)) is int
    for field_name in (
        "estimate_age_s",
        "last_timestamp_s",
        "total_integration_time_s",
        "total_nominal_exposure_photons",
        "cpu_time_s",
    ):
        assert type(getattr(estimate, field_name)) is float


@pytest.mark.parametrize(
    "invalid_override",
    [
        {"scipy_status": None},
        {"scipy_message": None},
        {"scipy_message": ""},
        {"nfev": 0},
    ],
    ids=["missing-status", "missing-message", "empty-message", "zero-evaluations"],
)
@pytest.mark.parametrize(
    "outcome", ["success", "quality_failed", "optimization_failed"]
)
def test_optimizer_attempts_require_complete_nonempty_provenance(
    outcome: str,
    invalid_override: dict[str, object],
) -> None:
    overrides = _attempt_result_overrides(outcome)
    overrides.update(invalid_override)

    with pytest.raises(ValueError, match="optimizer attempts"):
        _success_result(**overrides)


@pytest.mark.parametrize(
    ("outcome", "scipy_status"),
    [("success", 0), ("quality_failed", 0), ("optimization_failed", 1)],
)
def test_optimizer_attempt_status_sign_matches_outcome(
    outcome: str, scipy_status: int
) -> None:
    overrides = _attempt_result_overrides(outcome)
    overrides["scipy_status"] = scipy_status

    with pytest.raises(ValueError, match="scipy_status"):
        _success_result(**overrides)


def test_successful_fit_derives_read_only_signed_q_and_snapshots_guess() -> None:
    initial_guess = _initial_guess()
    result = _success_result(
        resonance_estimates=_resonances(signed_first_center=True),
        initial_guess=initial_guess,
    )

    assert result.q_values.shape == (8,)
    assert result.q_values[0] < 0.0
    assert result.q_values[0] == pytest.approx(-2760.0)
    assert not result.q_values.flags.writeable
    assert result.initial_guess is not initial_guess
    assert result.initial_guess == initial_guess


def test_successful_fit_rejects_unrepresentable_q_without_warning() -> None:
    resonances = list(_resonances())
    resonances[-1] = Resonance(
        resonance_id=resonances[-1].resonance_id,
        center_hz=np.finfo(np.float64).max,
        fwhm_hz=np.nextafter(0.0, 1.0),
        amplitude=resonances[-1].amplitude,
        eta=resonances[-1].eta,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite representable Q"):
            _success_result(resonance_estimates=tuple(resonances))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "initial_guess": FitInitialGuess(
                    _resonances(),
                    Baseline(1.0, 2.88e9, quadratic_per_hz2=1.0e-20),
                )
            },
            "linear-baseline initial guesses",
        ),
        (
            {
                "baseline_estimate": Baseline(
                    1.0, 2.88e9, quadratic_per_hz2=1.0e-20
                )
            },
            "linear-baseline estimates",
        ),
        (
            {
                "resonance_estimates": tuple(
                    Resonance(
                        resonance_id=(
                            "r1"
                            if index == 0
                            else "r0"
                            if index == 1
                            else item.resonance_id
                        ),
                        center_hz=item.center_hz,
                        fwhm_hz=item.fwhm_hz,
                        amplitude=item.amplitude,
                        eta=item.eta,
                    )
                    for index, item in enumerate(_resonances())
                )
            },
            "IDs and order",
        ),
        (
            {"baseline_estimate": Baseline(1.0, 2.89e9)},
            "baseline references",
        ),
    ],
)
def test_successful_result_rejects_cross_field_contradictions(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _success_result(**overrides)


def test_failed_linear_result_rejects_quadratic_attempted_baseline() -> None:
    invalid_guess = FitInitialGuess(
        _resonances(), Baseline(1.0, 2.88e9, quadratic_per_hz2=1.0e-20)
    )

    with pytest.raises(ValueError, match="linear-baseline initial guesses"):
        _success_result(
            success=False,
            failure_code="optimization_failed",
            resonance_estimates=(),
            baseline_estimate=None,
            initial_guess=invalid_guess,
            uncertainty=None,
            uncertainty_reason="optimizer stopped",
            scipy_status=0,
            scipy_message="stopped",
            nfev=1,
            cost=0.01,
            residual_rmse=0.02,
            residual_scale=0.1,
            jacobian_rank=None,
        )


@pytest.mark.parametrize(
    ("success", "failure_code", "source"),
    [
        (True, None, "none"),
        (False, "optimization_failed", "none"),
        (False, "quality_failed", "none"),
        (False, "insufficient_samples", "user"),
        (False, "uninformative_sweep", "detected"),
        (False, "initialization_failed", "user"),
    ],
)
def test_result_rejects_incompatible_diagnostic_source_and_attempt_state(
    success: bool, failure_code: str | None, source: str
) -> None:
    attempted = success or failure_code in {"optimization_failed", "quality_failed"}
    with pytest.raises(ValueError, match="diagnostic source"):
        _success_result(
            success=success,
            failure_code=failure_code,
            resonance_estimates=_resonances() if success else (),
            baseline_estimate=(Baseline(1.0, 2.88e9) if success else None),
            diagnostics=_diagnostics(source),
            initial_guess=_initial_guess() if attempted else None,
            uncertainty=(
                _uncertainty(eta=np.full(8, 1.0e-3)) if success else None
            ),
            uncertainty_reason=None if success else "not available",
            scipy_status=(
                0 if failure_code == "optimization_failed" else 1 if attempted else None
            ),
            scipy_message="stopped" if attempted else None,
            nfev=1 if attempted else 0,
            cost=0.01 if attempted else None,
            residual_rmse=0.02 if attempted else None,
            residual_scale=(
                None
                if failure_code in {"insufficient_samples", "uninformative_sweep"}
                else 0.1
            ),
            degrees_of_freedom=0 if failure_code == "insufficient_samples" else 100,
            jacobian_rank=(
                34
                if success
                else 10
                if failure_code == "quality_failed"
                else None
            ),
        )


def test_lorentzian_result_rejects_a_retained_guess_with_nonunit_eta() -> None:
    lorentzian_resonances = tuple(
        Resonance(
            resonance_id=resonance.resonance_id,
            center_hz=resonance.center_hz,
            fwhm_hz=resonance.fwhm_hz,
            amplitude=resonance.amplitude,
            eta=1.0,
        )
        for resonance in _resonances()
    )

    with pytest.raises(ValueError):
        _success_result(
            model_kind="lorentzian",
            resonance_estimates=lorentzian_resonances,
            initial_guess=_initial_guess(),
            uncertainty=_uncertainty(),
            jacobian_rank=26,
        )


@pytest.mark.parametrize(
    "failure_code",
    [
        "initialization_failed",
        "insufficient_samples",
        "uninformative_sweep",
        "optimization_failed",
        "quality_failed",
    ],
)
def test_failure_result_has_no_final_estimates_or_uncertainty(
    failure_code: str,
) -> None:
    optimizer_attempted = failure_code in {"optimization_failed", "quality_failed"}
    diagnostics = (
        _diagnostics("none")
        if failure_code in {"insufficient_samples", "uninformative_sweep"}
        else _diagnostics()
    )
    result = _success_result(
        success=False,
        failure_code=failure_code,
        resonance_estimates=(),
        baseline_estimate=None,
        diagnostics=diagnostics,
        initial_guess=_initial_guess() if optimizer_attempted else None,
        uncertainty=None,
        uncertainty_reason="fit did not produce a covariance estimate",
        scipy_status=(
            0
            if failure_code == "optimization_failed"
            else 1
            if optimizer_attempted
            else None
        ),
        scipy_message="stopped" if optimizer_attempted else None,
        nfev=1 if optimizer_attempted else 0,
        cost=0.01
        if failure_code in {"optimization_failed", "quality_failed"}
        else None,
        residual_rmse=0.02
        if failure_code in {"optimization_failed", "quality_failed"}
        else None,
        residual_scale=0.1
        if failure_code not in {"insufficient_samples", "uninformative_sweep"}
        else None,
        degrees_of_freedom=0 if failure_code == "insufficient_samples" else 100,
        jacobian_rank=10 if failure_code == "quality_failed" else None,
    )

    assert result.resonance_estimates == ()
    assert result.baseline_estimate is None
    assert result.q_values.shape == (0,)
    assert not result.q_values.flags.writeable


@pytest.mark.parametrize(
    "overrides",
    [
        {"success": True, "failure_code": "quality_failed"},
        {"success": False, "failure_code": None},
        {"resonance_estimates": _resonances()[:-1]},
        {"baseline_estimate": None},
        {"initial_guess": None},
        {"uncertainty": None, "uncertainty_reason": None},
        {"cost": None},
        {"residual_rmse": None},
        {"residual_scale": None},
        {"jacobian_rank": 33},
    ],
)
def test_successful_result_rejects_missing_or_contradictory_fields(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _success_result(**overrides)


@pytest.mark.parametrize(
    ("failure_code", "overrides"),
    [
        ("initialization_failed", {"initial_guess": _initial_guess()}),
        ("initialization_failed", {"residual_scale": None}),
        ("insufficient_samples", {"residual_scale": 1.0}),
        ("uninformative_sweep", {"cost": 1.0, "residual_rmse": 1.0}),
        ("optimization_failed", {"initial_guess": None}),
        ("optimization_failed", {"cost": 1.0, "residual_rmse": None}),
        ("quality_failed", {"uncertainty": _uncertainty()}),
        ("quality_failed", {"resonance_estimates": _resonances()}),
    ],
)
def test_failure_result_rejects_the_failure_state_matrix(
    failure_code: str, overrides: dict[str, object]
) -> None:
    optimizer_attempted = failure_code in {"optimization_failed", "quality_failed"}
    values: dict[str, object] = {
        "success": False,
        "failure_code": failure_code,
        "resonance_estimates": (),
        "baseline_estimate": None,
        "initial_guess": _initial_guess() if optimizer_attempted else None,
        "uncertainty": None,
        "uncertainty_reason": "not available",
        "scipy_status": (
            0
            if failure_code == "optimization_failed"
            else 1
            if optimizer_attempted
            else None
        ),
        "scipy_message": "stopped" if optimizer_attempted else None,
        "nfev": 1 if optimizer_attempted else 0,
        "cost": 0.01 if optimizer_attempted else None,
        "residual_rmse": 0.02 if optimizer_attempted else None,
        "residual_scale": 0.1
        if failure_code not in {"insufficient_samples", "uninformative_sweep"}
        else None,
        "degrees_of_freedom": 0 if failure_code == "insufficient_samples" else 100,
        "jacobian_rank": 10 if failure_code == "quality_failed" else None,
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        _success_result(**values)


@pytest.mark.parametrize(
    ("failure_code", "degrees_of_freedom"),
    [
        ("initialization_failed", 0),
        ("uninformative_sweep", 0),
        ("optimization_failed", 0),
        ("quality_failed", 0),
        ("insufficient_samples", 1),
    ],
)
def test_failure_result_enforces_degrees_of_freedom_state_semantics(
    failure_code: str, degrees_of_freedom: int
) -> None:
    optimizer_attempted = failure_code in {"optimization_failed", "quality_failed"}
    with pytest.raises(ValueError):
        _success_result(
            success=False,
            failure_code=failure_code,
            resonance_estimates=(),
            baseline_estimate=None,
            initial_guess=_initial_guess() if optimizer_attempted else None,
            uncertainty=None,
            uncertainty_reason="not available",
            scipy_status=(
                0
                if failure_code == "optimization_failed"
                else 1
                if optimizer_attempted
                else None
            ),
            scipy_message="stopped" if optimizer_attempted else None,
            nfev=1 if optimizer_attempted else 0,
            cost=0.01 if optimizer_attempted else None,
            residual_rmse=0.02 if optimizer_attempted else None,
            residual_scale=0.1
            if failure_code not in {"insufficient_samples", "uninformative_sweep"}
            else None,
            degrees_of_freedom=degrees_of_freedom,
            jacobian_rank=10 if failure_code == "quality_failed" else None,
        )


def test_quality_failed_result_rejects_rank_above_its_free_parameter_count() -> None:
    with pytest.raises(ValueError):
        _success_result(
            success=False,
            failure_code="quality_failed",
            resonance_estimates=(),
            baseline_estimate=None,
            initial_guess=_initial_guess(),
            uncertainty=None,
            uncertainty_reason="rank check failed",
            scipy_status=1,
            scipy_message="stopped",
            nfev=1,
            cost=0.01,
            residual_rmse=0.02,
            residual_scale=0.1,
            jacobian_rank=35,
        )


def test_nonfinite_optimizer_reason_rejects_finite_residual_metrics() -> None:
    with pytest.raises(ValueError, match="must omit residual metrics"):
        _success_result(
            success=False,
            failure_code="quality_failed",
            resonance_estimates=(),
            baseline_estimate=None,
            initial_guess=_initial_guess(),
            uncertainty=None,
            uncertainty_reason=(
                "optimizer returned non-finite parameters, residuals, or cost"
            ),
            scipy_status=1,
            scipy_message="stopped",
            nfev=1,
            cost=0.01,
            residual_rmse=0.02,
            residual_scale=0.1,
            jacobian_rank=None,
        )


def test_sweep_estimate_canonicalizes_completion_metadata_for_a_failure() -> None:
    fit = _success_result(
        success=False,
        failure_code="initialization_failed",
        resonance_estimates=(),
        baseline_estimate=None,
        initial_guess=None,
        uncertainty=None,
        uncertainty_reason="no candidate set",
        scipy_status=None,
        scipy_message=None,
        nfev=0,
        cost=None,
        residual_rmse=None,
        residual_scale=0.1,
        jacobian_rank=None,
    )
    estimate = SweepEstimate(
        fit=fit,
        last_sequence_index=np.int64(5),
        last_timestamp_s=np.float32(1.5),
        total_integration_time_s=np.float32(0.2),
        total_nominal_exposure_photons=np.float32(4.0),
    )

    assert estimate.fit is fit
    assert estimate.last_sequence_index == 5
    assert type(estimate.last_timestamp_s) is float
