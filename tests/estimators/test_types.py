"""Contract tests for immutable estimator value objects."""

from __future__ import annotations

import warnings

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
)
from odmr_bench.models import Baseline, Resonance


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
        "min_amplitude_significance": 3.0,
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
    assert FitConfiguration().min_amplitude_significance == 3.0


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
            scipy_status=1 if attempted else None,
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
        scipy_status=1 if optimizer_attempted else None,
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
        "scipy_status": 1 if optimizer_attempted else None,
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
            scipy_status=1 if optimizer_attempted else None,
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
