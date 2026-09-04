"""Integration regressions for causal warm-started completed sweeps."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from odmr_bench.dynamics import LinearCenterDrift, SpectralSnapshot
from odmr_bench.estimators import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    InitializationDiagnostics,
    RepeatedFullSweepEstimator,
    SpectrumFitResult,
    WarmStartedFullSweepEstimator,
)
from odmr_bench.estimators.fitting import fit_spectrum as real_fit_spectrum
from odmr_bench.models import Baseline, Resonance, multi_resonance_spectrum

BASE_FREQUENCY_HZ = np.linspace(2.740e9, 3.020e9, 4481)
SHIFTED_FREQUENCY_HZ = np.linspace(2.741e9, 3.021e9, 4481)
BASE_CENTERS_HZ = 1e9 * np.array(
    [2.760, 2.794, 2.828, 2.862, 2.896, 2.930, 2.964, 2.998]
)
FWHM_HZ = 1e6 * np.array([1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20])
AMPLITUDES = np.array(
    [0.018, 0.021, 0.024, 0.027, 0.030, 0.033, 0.036, 0.039]
)
ETAS = np.array([0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.93])
COMPLETION_TIMESTAMPS_S = (1.0, 2.0, 3.0)
LAST_SEQUENCE_INDICES = (4480, 8961, 13442)
NOISE_SEEDS = (6211, 6212, 6213)
NOISE_SIGMA = 2.0e-4
CENTER_SLEW_HZ_PER_S = 1.0e5
CENTER_REGRESSION_TOLERANCE_FWHM = 0.10
FWHM_REGRESSION_RELATIVE_TOLERANCE = 0.18
Q_REGRESSION_RELATIVE_TOLERANCE = 0.18

DRIFT_CONFIGURATION = FitConfiguration(
    model_kind="pseudo_voigt",
    baseline_degree=2,
    resonance_ids=tuple(f"r{index}" for index in range(8)),
    min_fwhm_hz=2.0e5,
    max_fwhm_hz=8.0e6,
    max_amplitude=0.08,
    min_resolved_amplitude=1.0e-4,
    min_amplitude_significance=5.0,
    min_center_separation_hz=1.0e6,
    savgol_window=11,
    savgol_polyorder=2,
    relative_prominence=0.01,
    allow_fallback=False,
    max_nfev=4000,
    rank_rtol=1.0e-10,
    min_baseline_sse_improvement=1.0e-4,
)


def _initial_snapshot() -> SpectralSnapshot:
    resonances = tuple(
        Resonance(f"r{index}", center, width, amplitude, eta)
        for index, (center, width, amplitude, eta) in enumerate(
            zip(BASE_CENTERS_HZ, FWHM_HZ, AMPLITUDES, ETAS, strict=True)
        )
    )
    baseline = Baseline(
        intercept=1.0,
        reference_hz=2.880e9,
        slope_per_hz=2.0e-11,
        quadratic_per_hz2=-5.0e-20,
    )
    return SpectralSnapshot(baseline, resonances)


def _build_frozen_drift_fixture(
) -> tuple[tuple[CompleteSweep, ...], tuple[tuple[Resonance, ...], ...]]:
    dynamics = LinearCenterDrift(_initial_snapshot(), CENTER_SLEW_HZ_PER_S)
    grids = (BASE_FREQUENCY_HZ, SHIFTED_FREQUENCY_HZ, BASE_FREQUENCY_HZ)
    sweeps: list[CompleteSweep] = []
    truth_resonances_by_update: list[tuple[Resonance, ...]] = []

    for timestamp_s, last_sequence_index, noise_seed, frequency_hz in zip(
        COMPLETION_TIMESTAMPS_S,
        LAST_SEQUENCE_INDICES,
        NOISE_SEEDS,
        grids,
        strict=True,
    ):
        snapshot = dynamics.snapshot_at(timestamp_s)
        truth_resonances_by_update.append(snapshot.resonances)
        noise = np.random.default_rng(noise_seed).normal(
            0.0, NOISE_SIGMA, frequency_hz.size
        )
        fluorescence = multi_resonance_spectrum(
            frequency_hz,
            snapshot.resonances,
            snapshot.baseline,
            additive_noise=noise,
        )
        sweeps.append(
            CompleteSweep(
                frequency_hz,
                fluorescence,
                last_sequence_index=last_sequence_index,
                last_timestamp_s=timestamp_s,
                total_integration_time_s=4.481,
                total_nominal_exposure_photons=4.481e6,
            )
        )

    return tuple(sweeps), tuple(truth_resonances_by_update)


def _assert_fit_matches_frozen_truth(
    fit: SpectrumFitResult, truth: tuple[Resonance, ...]
) -> None:
    assert fit.success
    assert tuple(item.resonance_id for item in fit.resonance_estimates) == (
        DRIFT_CONFIGURATION.resonance_ids
    )
    fitted_centers_hz = np.array(
        [item.center_hz for item in fit.resonance_estimates]
    )
    fitted_fwhm_hz = np.array([item.fwhm_hz for item in fit.resonance_estimates])
    fitted_q = np.asarray(fit.q_values)
    truth_centers_hz = np.array([item.center_hz for item in truth])
    truth_fwhm_hz = np.array([item.fwhm_hz for item in truth])
    truth_q = truth_centers_hz / truth_fwhm_hz

    assert np.all(
        np.abs(fitted_centers_hz - truth_centers_hz)
        < CENTER_REGRESSION_TOLERANCE_FWHM * truth_fwhm_hz
    )
    np.testing.assert_allclose(
        fitted_fwhm_hz,
        truth_fwhm_hz,
        rtol=FWHM_REGRESSION_RELATIVE_TOLERANCE,
        atol=0.0,
    )
    np.testing.assert_allclose(
        fitted_q,
        truth_q,
        rtol=Q_REGRESSION_RELATIVE_TOLERANCE,
        atol=0.0,
    )
    assert isinstance(fit.nfev, int)
    assert fit.nfev >= 0


def _assert_no_hidden_truth_state(estimator: object) -> None:
    for name in (
        "dynamics",
        "_dynamics",
        "snapshot",
        "_snapshot",
        "snapshots",
        "_snapshots",
        "truth",
        "_truth",
        "truth_resonances_by_update",
        "_truth_resonances_by_update",
    ):
        assert not hasattr(estimator, name)


def test_frozen_snapshot_drift_uses_identical_sweeps_and_declared_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweeps, truth_resonances_by_update = _build_frozen_drift_fixture()
    configuration = DRIFT_CONFIGURATION
    cold = RepeatedFullSweepEstimator(configuration)
    warm = WarmStartedFullSweepEstimator(configuration)
    cold_seen: list[CompleteSweep] = []
    warm_seen: list[CompleteSweep] = []

    def cold_spy(
        sweep: CompleteSweep,
        fit_configuration: FitConfiguration,
        initial_guess: FitInitialGuess | None,
    ) -> SpectrumFitResult:
        cold_seen.append(sweep)
        return real_fit_spectrum(sweep, fit_configuration, initial_guess)

    def warm_spy(
        sweep: CompleteSweep,
        fit_configuration: FitConfiguration,
        initial_guess: FitInitialGuess | None,
    ) -> SpectrumFitResult:
        warm_seen.append(sweep)
        return real_fit_spectrum(sweep, fit_configuration, initial_guess)

    monkeypatch.setattr("odmr_bench.estimators.full_sweep.fit_spectrum", cold_spy)
    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", warm_spy)

    cold_estimates = []
    warm_estimates = []
    for sweep in sweeps:
        cold_estimates.append(cold.update_sweep(sweep))
        warm_estimates.append(warm.update_sweep(sweep))

    assert cold.configuration is configuration
    assert warm.configuration is configuration
    assert len(cold_seen) == len(warm_seen) == len(sweeps)
    for index, sweep in enumerate(sweeps):
        assert cold_seen[index] is sweep
        assert warm_seen[index] is sweep
        _assert_fit_matches_frozen_truth(
            cold_estimates[index].fit, truth_resonances_by_update[index]
        )
        _assert_fit_matches_frozen_truth(
            warm_estimates[index].current_fit,
            truth_resonances_by_update[index],
        )
        assert math.isfinite(warm_estimates[index].cpu_time_s)
        assert warm_estimates[index].cpu_time_s >= 0.0
        assert warm_estimates[index].total_integration_time_s == (
            sweep.total_integration_time_s
        )
        assert warm_estimates[index].total_nominal_exposure_photons == (
            sweep.total_nominal_exposure_photons
        )
        assert warm_estimates[index].last_sequence_index == sweep.last_sequence_index
        assert warm_estimates[index].last_timestamp_s == sweep.last_timestamp_s
        assert cold_estimates[index].total_integration_time_s == (
            sweep.total_integration_time_s
        )
        assert cold_estimates[index].total_nominal_exposure_photons == (
            sweep.total_nominal_exposure_photons
        )
        assert cold_estimates[index].last_sequence_index == sweep.last_sequence_index
        assert cold_estimates[index].last_timestamp_s == sweep.last_timestamp_s

    assert [
        tuple(attempt.start_kind for attempt in estimate.attempts)
        for estimate in warm_estimates
    ] == [("cold",), ("warm",), ("warm",)]
    assert [
        estimate.attempts[0].warm_source_update_index
        for estimate in warm_estimates
    ] == [None, 0, 1]
    assert [
        estimate.active_source_update_index for estimate in warm_estimates
    ] == [0, 1, 2]
    assert [
        estimate.estimate_age_submitted_observations
        for estimate in warm_estimates
    ] == [0, 0, 0]
    assert [
        estimate.estimate_age_sequence_indices for estimate in warm_estimates
    ] == [0, 0, 0]
    assert [estimate.estimate_age_s for estimate in warm_estimates] == [0.0, 0.0, 0.0]
    assert [
        estimate.cumulative_observation_count for estimate in warm_estimates
    ] == [4481, 8962, 13443]
    assert [estimate.observation_count for estimate in warm_estimates] == [4481] * 3
    _assert_no_hidden_truth_state(cold)
    _assert_no_hidden_truth_state(warm)


@pytest.mark.parametrize(
    ("maximum_age", "expected_kind", "expected_source", "expected_disposition"),
    [
        (None, "warm", 0, "used"),
        (1, "cold", None, "rejected_age"),
    ],
)
def test_wrapper_preflight_failure_ages_prior_without_drift_truth(
    maximum_age: int | None,
    expected_kind: str,
    expected_source: int | None,
    expected_disposition: str,
) -> None:
    sweeps, _ = _build_frozen_drift_fixture()
    constant = CompleteSweep(
        SHIFTED_FREQUENCY_HZ,
        np.ones(SHIFTED_FREQUENCY_HZ.size),
        last_sequence_index=LAST_SEQUENCE_INDICES[1],
        last_timestamp_s=COMPLETION_TIMESTAMPS_S[1],
        total_integration_time_s=4.481,
        total_nominal_exposure_photons=4.481e6,
    )
    estimator = WarmStartedFullSweepEstimator(
        DRIFT_CONFIGURATION,
        max_warm_start_age_updates=maximum_age,
    )

    first = estimator.update_sweep(sweeps[0])
    failed = estimator.update_sweep(constant)
    recovered = estimator.update_sweep(sweeps[2])

    assert first.current_fit.success
    fitted_ids = tuple(
        item.resonance_id for item in first.current_fit.resonance_estimates
    )
    assert fitted_ids == DRIFT_CONFIGURATION.resonance_ids
    assert tuple(attempt.start_kind for attempt in failed.attempts) == ("preflight",)
    assert failed.current_fit.failure_code == "uninformative_sweep"
    assert failed.current_fit.resonance_estimates == ()
    assert failed.is_stale
    assert failed.active_fit is first.current_fit
    assert failed.active_source_update_index == 0
    assert (
        failed.estimate_age_submitted_observations,
        failed.estimate_age_sequence_indices,
        failed.estimate_age_s,
    ) == (4481, 4481, 1.0)
    assert tuple(attempt.start_kind for attempt in recovered.attempts) == (
        expected_kind,
    )
    assert recovered.attempts[0].warm_source_update_index == expected_source
    assert recovered.warm_start_disposition == expected_disposition
    assert recovered.current_fit.success
    assert tuple(
        item.resonance_id for item in recovered.current_fit.resonance_estimates
    ) == DRIFT_CONFIGURATION.resonance_ids
    if maximum_age is None:
        assert recovered.warm_start_rejection_code is None
    else:
        assert recovered.warm_start_rejection_code == "age_limit_exceeded"


def test_narrowed_sweep_rejects_prior_center_without_truth_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweeps, truth_resonances_by_update = _build_frozen_drift_fixture()
    estimator = WarmStartedFullSweepEstimator(DRIFT_CONFIGURATION)
    seeded = estimator.update_sweep(sweeps[0])
    first_prior_center_hz = seeded.current_fit.resonance_estimates[0].center_hz
    frequency_hz = np.linspace(first_prior_center_hz + 1.0, 3.020e9, 4481)
    truth = truth_resonances_by_update[1]
    narrowed = CompleteSweep(
        frequency_hz,
        multi_resonance_spectrum(
            frequency_hz,
            truth,
            _initial_snapshot().baseline,
            additive_noise=np.random.default_rng(NOISE_SEEDS[1]).normal(
                0.0, NOISE_SIGMA, frequency_hz.size
            ),
        ),
        last_sequence_index=LAST_SEQUENCE_INDICES[1],
        last_timestamp_s=COMPLETION_TIMESTAMPS_S[1],
        total_integration_time_s=4.481,
        total_nominal_exposure_photons=4.481e6,
    )
    calls: list[tuple[CompleteSweep, FitInitialGuess | None]] = []

    def spy(
        sweep: CompleteSweep,
        configuration: FitConfiguration,
        initial_guess: FitInitialGuess | None,
    ) -> SpectrumFitResult:
        calls.append((sweep, initial_guess))
        return real_fit_spectrum(sweep, configuration, initial_guess)

    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", spy)

    rejected = estimator.update_sweep(narrowed)

    assert calls == [(narrowed, None)]
    assert tuple(attempt.start_kind for attempt in rejected.attempts) == ("cold",)
    assert rejected.warm_start_disposition == "rejected_compatibility"
    assert rejected.warm_start_rejection_code == "center_outside_sweep"
    assert rejected.attempts[0].warm_source_update_index is None
    if rejected.current_fit.success:
        assert tuple(
            item.resonance_id for item in rejected.current_fit.resonance_estimates
        ) == DRIFT_CONFIGURATION.resonance_ids
    else:
        assert rejected.current_fit.resonance_estimates == ()


def _fixed_warm_optimization_failure(
    successful_fit: SpectrumFitResult,
    initial_guess: FitInitialGuess,
) -> SpectrumFitResult:
    return SpectrumFitResult(
        success=False,
        failure_code="optimization_failed",
        model_kind=successful_fit.model_kind,
        baseline_degree=successful_fit.baseline_degree,
        resonance_estimates=(),
        baseline_estimate=None,
        diagnostics=InitializationDiagnostics(
            source="user",
            candidate_count=0,
            selected_indices=(),
            used_fallback=False,
            messages=("fixed integration recovery fixture",),
        ),
        initial_guess=initial_guess,
        uncertainty=None,
        uncertainty_reason="fixed optimizer failure",
        scipy_status=0,
        scipy_message="fixed optimizer failure",
        nfev=7,
        cost=0.25,
        residual_rmse=0.01,
        residual_scale=0.01,
        degrees_of_freedom=successful_fit.degrees_of_freedom,
        jacobian_rank=None,
    )


def test_optimizer_failure_recovers_cold_without_duplicate_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweeps, _ = _build_frozen_drift_fixture()
    estimator = WarmStartedFullSweepEstimator(DRIFT_CONFIGURATION)
    seeded = estimator.update_sweep(sweeps[0])
    fixed_cold_success = replace(seeded.current_fit, nfev=11)
    calls: list[tuple[CompleteSweep, FitInitialGuess | None]] = []
    warm_failure: SpectrumFitResult | None = None

    def fixed_fit(
        sweep: CompleteSweep,
        configuration: FitConfiguration,
        initial_guess: FitInitialGuess | None,
    ) -> SpectrumFitResult:
        nonlocal warm_failure
        calls.append((sweep, initial_guess))
        if initial_guess is not None:
            warm_failure = _fixed_warm_optimization_failure(
                seeded.current_fit, initial_guess
            )
            return warm_failure
        return fixed_cold_success

    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", fixed_fit)

    recovered = estimator.update_sweep(sweeps[2])

    assert len(calls) == 2
    assert calls[0][0] is sweeps[2]
    assert calls[0][1] is not None
    assert calls[1] == (sweeps[2], None)
    assert tuple(attempt.start_kind for attempt in recovered.attempts) == (
        "warm",
        "cold",
    )
    assert recovered.attempts[0].warm_source_update_index == 0
    assert recovered.attempts[1].warm_source_update_index is None
    assert recovered.attempts[0].fit is warm_failure
    assert recovered.current_fit is fixed_cold_success
    assert recovered.active_fit is fixed_cold_success
    assert recovered.active_source_update_index == 1
    assert recovered.observation_count == sweeps[2].frequency_hz.size
    assert recovered.cumulative_observation_count == 8962
    assert recovered.total_integration_time_s == sweeps[2].total_integration_time_s
    assert recovered.total_nominal_exposure_photons == (
        sweeps[2].total_nominal_exposure_photons
    )
    assert recovered.total_nfev == 18


def test_warm_started_example_runs_from_an_unrelated_working_directory(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "examples" / "fit_warm_started_sweeps.py"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    rows = completed.stdout.splitlines()
    assert rows[0] == "Synthetic warm-started sweep diagnostics"
    field_names = [
        "update",
        "disposition",
        "attempts",
        "warm_source",
        "active_source",
        "age_observations",
        "total_nfev",
        "cpu_time_s",
    ]
    assert rows[1].split() == field_names
    assert len(rows[2:]) == 3
    parsed = [dict(zip(field_names, row.split(), strict=True)) for row in rows[2:]]
    assert [row["update"] for row in parsed] == ["0", "1", "2"]
    assert [row["attempts"] for row in parsed] == ["cold", "warm", "warm"]
    assert [row["warm_source"] for row in parsed] == ["none", "0", "1"]
    assert [row["active_source"] for row in parsed] == ["0", "1", "2"]
    assert [row["age_observations"] for row in parsed] == ["0", "0", "0"]
    for row in parsed:
        assert int(row["total_nfev"]) >= 0
        assert math.isfinite(float(row["cpu_time_s"]))
        assert float(row["cpu_time_s"]) >= 0.0


def test_warm_started_guidance_names_operational_boundaries() -> None:
    repository = Path(__file__).resolve().parents[2]
    guidance = (repository / "docs" / "estimators.md").read_text(encoding="utf-8")

    for term in (
        "warm source",
        "baseline rebase",
        "cold recovery",
        "stale",
        "estimate_age_submitted_observations",
        "center boxes",
        "measured update-core process CPU interval",
    ):
        assert term in guidance
