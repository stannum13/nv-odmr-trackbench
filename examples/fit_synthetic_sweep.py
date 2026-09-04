"""Generate and fit one deterministic synthetic eight-resonance sweep."""

from __future__ import annotations

import math

import numpy as np

from odmr_bench.estimators import (
    CompleteSweep,
    FitConfiguration,
    RepeatedFullSweepEstimator,
)
from odmr_bench.models import Baseline, Resonance, multi_resonance_spectrum


def main() -> None:
    frequency_hz = np.linspace(2.740e9, 3.020e9, 4481)
    centers_hz = 1e9 * np.array(
        [2.760, 2.794, 2.828, 2.862, 2.896, 2.930, 2.964, 2.998]
    )
    fwhm_hz = 1e6 * np.array([1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20])
    amplitudes = np.array([0.018, 0.021, 0.024, 0.027, 0.030, 0.033, 0.036, 0.039])
    etas = np.array([0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.93])
    resonances = tuple(
        Resonance(f"r{index}", center, width, amplitude, eta)
        for index, (center, width, amplitude, eta) in enumerate(
            zip(centers_hz, fwhm_hz, amplitudes, etas, strict=True)
        )
    )
    baseline = Baseline(
        intercept=1.0,
        reference_hz=2.880e9,
        slope_per_hz=2.0e-11,
        quadratic_per_hz2=-5.0e-20,
    )
    fluorescence = multi_resonance_spectrum(frequency_hz, resonances, baseline)
    sweep = CompleteSweep(frequency_hz, fluorescence)
    estimator = RepeatedFullSweepEstimator(
        FitConfiguration(
            model_kind="pseudo_voigt",
            baseline_degree=2,
            max_amplitude=0.08,
            relative_prominence=0.01,
        )
    )

    estimate = estimator.update_sweep(sweep)
    if not estimate.fit.success:
        raise RuntimeError(f"synthetic fit failed: {estimate.fit.failure_code}")

    print("Synthetic pseudo-Voigt fit diagnostics")
    for resonance, q_value in zip(
        estimate.fit.resonance_estimates, estimate.fit.q_values, strict=True
    ):
        values = (resonance.center_hz, resonance.fwhm_hz, float(q_value))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("fit returned a non-finite public diagnostic")
        print(
            f"{resonance.resonance_id} "
            f"center_hz={resonance.center_hz:.9g} "
            f"fwhm_hz={resonance.fwhm_hz:.9g} "
            f"q={q_value:.9g}"
        )


if __name__ == "__main__":
    main()
