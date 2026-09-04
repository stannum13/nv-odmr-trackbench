"""Regressions for the cold-start repeated full-sweep estimator."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import odmr_bench.estimators as estimator_api
from odmr_bench.estimators import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    InitializationDiagnostics,
    SpectrumFitResult,
)
from odmr_bench.models import Baseline, Resonance, multi_resonance_spectrum

FREQUENCY_HZ = np.linspace(2.740e9, 3.020e9, 4481)
BASE_CENTERS_HZ = 1e9 * np.array(
    [2.760, 2.794, 2.828, 2.862, 2.896, 2.930, 2.964, 2.998]
)
FWHM_HZ = 1e6 * np.array([1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20])
AMPLITUDES = np.array([0.018, 0.021, 0.024, 0.027, 0.030, 0.033, 0.036, 0.039])
ETAS = np.array([0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.93])

# These bound one generated regression fixture; they are not benchmark results.
CENTER_REGRESSION_TOLERANCE_FWHM = 0.10
FWHM_REGRESSION_RELATIVE_TOLERANCE = 0.18


def _estimator_class() -> type:
    return estimator_api.RepeatedFullSweepEstimator


def _resonances() -> tuple[Resonance, ...]:
    return tuple(
        Resonance(f"r{index}", 2.76e9 + index * 34e6, 1.5e6, 0.02, 0.5)
        for index in range(8)
    )


def _success_result() -> SpectrumFitResult:
    baseline = Baseline(1.0, 2.88e9)
    guess = FitInitialGuess(_resonances(), baseline)
    return SpectrumFitResult(
        success=True,
        failure_code=None,
        model_kind="pseudo_voigt",
        baseline_degree=1,
        resonance_estimates=_resonances(),
        baseline_estimate=baseline,
        diagnostics=InitializationDiagnostics(
            source="user",
            candidate_count=0,
            selected_indices=(),
            used_fallback=False,
            messages=(),
        ),
        initial_guess=guess,
        uncertainty=None,
        uncertainty_reason="not evaluated in wrapper contract test",
        scipy_status=1,
        scipy_message="converged",
        nfev=1,
        cost=0.0,
        residual_rmse=0.0,
        residual_scale=0.1,
        degrees_of_freedom=100,
        jacobian_rank=34,
    )


def _failure_result() -> SpectrumFitResult:
    return SpectrumFitResult(
        success=False,
        failure_code="uninformative_sweep",
        model_kind="pseudo_voigt",
        baseline_degree=1,
        resonance_estimates=(),
        baseline_estimate=None,
        diagnostics=InitializationDiagnostics(
            source="none",
            candidate_count=0,
            selected_indices=(),
            used_fallback=False,
            messages=("no variation",),
        ),
        initial_guess=None,
        uncertainty=None,
        uncertainty_reason="fluorescence variation is zero",
        scipy_status=None,
        scipy_message=None,
        nfev=0,
        cost=None,
        residual_rmse=None,
        residual_scale=None,
        degrees_of_freedom=100,
        jacobian_rank=None,
    )


def _sweep(
    marker: float,
    *,
    last_sequence_index: int,
    last_timestamp_s: float,
) -> CompleteSweep:
    return CompleteSweep(
        np.linspace(2.74e9, 3.02e9, 64),
        np.linspace(marker, marker + 0.1, 64),
        last_sequence_index=last_sequence_index,
        last_timestamp_s=last_timestamp_s,
        total_integration_time_s=marker,
        total_nominal_exposure_photons=marker * 1e6,
    )


def test_each_update_is_cold_start_and_copies_only_that_sweeps_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator_class = _estimator_class()
    configuration = FitConfiguration()
    estimator = estimator_class(configuration)
    first_sweep = _sweep(2.0, last_sequence_index=19, last_timestamp_s=5.0)
    second_sweep = _sweep(1.0, last_sequence_index=4, last_timestamp_s=1.5)
    first_fit = _success_result()
    second_fit = _failure_result()
    calls: list[tuple[CompleteSweep, FitConfiguration, object]] = []
    results = iter((first_fit, second_fit))

    def recording_fit(
        sweep: CompleteSweep,
        supplied_configuration: FitConfiguration,
        *,
        initial_guess: object,
    ) -> SpectrumFitResult:
        calls.append((sweep, supplied_configuration, initial_guess))
        return next(results)

    monkeypatch.setattr("odmr_bench.estimators.full_sweep.fit_spectrum", recording_fit)

    first_estimate = estimator.update_sweep(first_sweep)
    second_estimate = estimator.update_sweep(second_sweep)

    assert calls == [
        (first_sweep, configuration, None),
        (second_sweep, configuration, None),
    ]
    assert first_estimate.fit is first_fit
    assert second_estimate.fit is second_fit
    assert (
        first_estimate.last_sequence_index,
        first_estimate.last_timestamp_s,
        first_estimate.total_integration_time_s,
        first_estimate.total_nominal_exposure_photons,
    ) == (19, 5.0, 2.0, 2.0e6)
    assert (
        second_estimate.last_sequence_index,
        second_estimate.last_timestamp_s,
        second_estimate.total_integration_time_s,
        second_estimate.total_nominal_exposure_photons,
    ) == (4, 1.5, 1.0, 1.0e6)
    assert estimator.latest is second_estimate
    assert estimator.history == (first_estimate, second_estimate)
    assert isinstance(estimator.history, tuple)
    with pytest.raises(FrozenInstanceError):
        second_estimate.last_timestamp_s = 99.0


def test_history_retains_success_failure_success_and_latest_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator_class = _estimator_class()
    estimator = estimator_class(FitConfiguration())
    fit_results = (_success_result(), _failure_result(), _success_result())
    result_iterator = iter(fit_results)
    monkeypatch.setattr(
        "odmr_bench.estimators.full_sweep.fit_spectrum",
        lambda sweep, configuration, *, initial_guess: next(result_iterator),
    )

    first_estimate = estimator.update_sweep(
        _sweep(1.0, last_sequence_index=0, last_timestamp_s=0.5)
    )
    failed_estimate = estimator.update_sweep(
        _sweep(2.0, last_sequence_index=1, last_timestamp_s=0.5)
    )

    assert estimator.latest is failed_estimate
    assert estimator.latest.fit is fit_results[1]

    final_estimate = estimator.update_sweep(
        _sweep(3.0, last_sequence_index=2, last_timestamp_s=0.5)
    )
    estimates = (first_estimate, failed_estimate, final_estimate)

    assert tuple(estimate.fit for estimate in estimator.history) == fit_results
    assert estimator.history == estimates
    assert estimator.history[1].fit.failure_code == "uninformative_sweep"
    assert estimator.latest is final_estimate


def test_latest_starts_empty_reset_clears_state_and_configuration_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator_class = _estimator_class()
    configuration = FitConfiguration()
    estimator = estimator_class(configuration)
    monkeypatch.setattr(
        "odmr_bench.estimators.full_sweep.fit_spectrum",
        lambda sweep, configuration, *, initial_guess: _success_result(),
    )

    assert estimator.configuration is configuration
    assert estimator.latest is None
    assert estimator.history == ()
    estimator.update_sweep(_sweep(1.0, last_sequence_index=0, last_timestamp_s=0.25))
    estimator.reset()

    assert estimator.latest is None
    assert estimator.history == ()
    with pytest.raises(AttributeError):
        estimator.configuration = FitConfiguration()


def test_invalid_input_is_rejected_before_fitting_without_appending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator_class = _estimator_class()
    estimator = estimator_class(FitConfiguration())
    fit_calls = 0

    def recording_fit(*args: object, **kwargs: object) -> SpectrumFitResult:
        nonlocal fit_calls
        fit_calls += 1
        return _success_result()

    monkeypatch.setattr("odmr_bench.estimators.full_sweep.fit_spectrum", recording_fit)
    estimator.update_sweep(_sweep(1.0, last_sequence_index=0, last_timestamp_s=0.25))
    history_before_invalid_input = estimator.history

    with pytest.raises(TypeError, match="CompleteSweep"):
        estimator.update_sweep(object())

    assert fit_calls == 1
    assert estimator.history == history_before_invalid_input


def test_constructor_rejects_non_configuration() -> None:
    estimator_class = _estimator_class()

    with pytest.raises(TypeError, match="FitConfiguration"):
        estimator_class(object())


def test_two_generated_sweeps_fit_independently_with_declared_regression_bounds() -> (
    None
):
    estimator_class = _estimator_class()
    configuration = FitConfiguration(
        model_kind="pseudo_voigt",
        baseline_degree=2,
        max_amplitude=0.08,
        relative_prominence=0.01,
    )
    estimator = estimator_class(configuration)
    shifts_hz = (
        np.zeros(8),
        1e5 * np.array([1.0, -0.5, 0.8, -1.0, 0.6, -0.7, 0.9, -0.4]),
    )
    metadata = ((4480, 4.481), (17, 0.750))
    estimates = []

    for shift_hz, seed, (last_index, last_timestamp_s) in zip(
        shifts_hz, (6201, 6202), metadata, strict=True
    ):
        centers_hz = BASE_CENTERS_HZ + shift_hz
        resonances = tuple(
            Resonance(f"r{index}", center, width, amplitude, eta)
            for index, (center, width, amplitude, eta) in enumerate(
                zip(centers_hz, FWHM_HZ, AMPLITUDES, ETAS, strict=True)
            )
        )
        baseline = Baseline(
            intercept=1.0,
            reference_hz=2.880e9,
            slope_per_hz=2.0e-11,
            quadratic_per_hz2=-5.0e-20,
        )
        noise = np.random.default_rng(seed).normal(0.0, 2.0e-4, FREQUENCY_HZ.size)
        sweep = CompleteSweep(
            FREQUENCY_HZ,
            multi_resonance_spectrum(
                FREQUENCY_HZ, resonances, baseline, additive_noise=noise
            ),
            last_sequence_index=last_index,
            last_timestamp_s=last_timestamp_s,
        )

        estimate = estimator.update_sweep(sweep)
        estimates.append(estimate)

        assert estimate.fit.success
        assert estimate.fit.diagnostics.source == "detected"
        fitted = estimate.fit.resonance_estimates
        assert [item.resonance_id for item in fitted] == [f"r{i}" for i in range(8)]
        fitted_centers_hz = np.array([item.center_hz for item in fitted])
        fitted_fwhm_hz = np.array([item.fwhm_hz for item in fitted])
        assert np.all(np.diff(fitted_centers_hz) > 0.0)
        assert np.all(
            np.abs(fitted_centers_hz - centers_hz)
            < CENTER_REGRESSION_TOLERANCE_FWHM * FWHM_HZ
        )
        np.testing.assert_allclose(
            fitted_fwhm_hz,
            FWHM_HZ,
            rtol=FWHM_REGRESSION_RELATIVE_TOLERANCE,
            atol=0.0,
        )

    assert [estimate.last_sequence_index for estimate in estimates] == [4480, 17]
    assert [estimate.last_timestamp_s for estimate in estimates] == [4.481, 0.750]
    assert estimator.history == tuple(estimates)


def test_synthetic_fit_example_runs_from_an_unrelated_working_directory(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(repository / "examples" / "fit_synthetic_sweep.py")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    rows = completed.stdout.splitlines()
    assert rows[0] == "Synthetic pseudo-Voigt fit diagnostics"
    assert len(rows[1:]) == 8
    for index, row in enumerate(rows[1:]):
        fields = row.split()
        assert fields[0] == f"r{index}"
        diagnostics = {
            name: float(value)
            for name, value in (field.split("=") for field in fields[1:])
        }
        assert diagnostics.keys() == {"center_hz", "fwhm_hz", "q"}
        assert np.all(np.isfinite(tuple(diagnostics.values())))
        assert diagnostics["fwhm_hz"] > 0.0
