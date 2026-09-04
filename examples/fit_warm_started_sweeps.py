"""Fit three deterministic completed sweeps with causal warm starts."""

from __future__ import annotations

import math

import numpy as np

from odmr_bench.dynamics import LinearCenterDrift, SpectralSnapshot
from odmr_bench.estimators import (
    CompleteSweep,
    FitConfiguration,
    WarmStartedFullSweepEstimator,
)
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


def _generated_sweeps() -> tuple[CompleteSweep, ...]:
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
    dynamics = LinearCenterDrift(
        SpectralSnapshot(baseline, resonances), CENTER_SLEW_HZ_PER_S
    )
    grids = (BASE_FREQUENCY_HZ, SHIFTED_FREQUENCY_HZ, BASE_FREQUENCY_HZ)
    sweeps = []
    for timestamp_s, last_sequence_index, noise_seed, frequency_hz in zip(
        COMPLETION_TIMESTAMPS_S,
        LAST_SEQUENCE_INDICES,
        NOISE_SEEDS,
        grids,
        strict=True,
    ):
        snapshot = dynamics.snapshot_at(timestamp_s)
        noise = np.random.default_rng(noise_seed).normal(
            0.0, NOISE_SIGMA, frequency_hz.size
        )
        sweeps.append(
            CompleteSweep(
                frequency_hz,
                multi_resonance_spectrum(
                    frequency_hz,
                    snapshot.resonances,
                    snapshot.baseline,
                    additive_noise=noise,
                ),
                last_sequence_index=last_sequence_index,
                last_timestamp_s=timestamp_s,
                total_integration_time_s=4.481,
                total_nominal_exposure_photons=4.481e6,
            )
        )
    return tuple(sweeps)


def _source_text(source: int | None) -> str:
    return "none" if source is None else str(source)


def main() -> None:
    estimator = WarmStartedFullSweepEstimator(DRIFT_CONFIGURATION)
    estimates = [estimator.update_sweep(sweep) for sweep in _generated_sweeps()]

    print("Synthetic warm-started sweep diagnostics")
    print(
        "update disposition attempts warm_source active_source "
        "age_observations total_nfev cpu_time_s"
    )
    for estimate in estimates:
        fit = estimate.current_fit
        if not fit.success:
            raise RuntimeError(
                f"synthetic update {estimate.update_index} failed: {fit.failure_code}"
            )
        fitted_ids = tuple(
            resonance.resonance_id for resonance in fit.resonance_estimates
        )
        if fitted_ids != DRIFT_CONFIGURATION.resonance_ids:
            raise RuntimeError("fit returned resonance IDs outside configured order")
        if not math.isfinite(estimate.cpu_time_s):
            raise RuntimeError("fit returned a non-finite CPU diagnostic")
        if estimate.total_nfev < 0:
            raise RuntimeError("fit returned a negative evaluation count")
        age = estimate.estimate_age_submitted_observations
        if age is None:
            raise RuntimeError("successful fit did not expose submitted age")
        warm_source = next(
            (
                attempt.warm_source_update_index
                for attempt in estimate.attempts
                if attempt.start_kind == "warm"
            ),
            None,
        )
        attempts = ",".join(attempt.start_kind for attempt in estimate.attempts)
        print(
            f"{estimate.update_index} {estimate.warm_start_disposition} {attempts} "
            f"{_source_text(warm_source)} "
            f"{_source_text(estimate.active_source_update_index)} "
            f"{age} {estimate.total_nfev} {estimate.cpu_time_s:.9g}"
        )


if __name__ == "__main__":
    main()
