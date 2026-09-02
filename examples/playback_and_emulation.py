"""Run tiny in-memory playback and synthetic virtual queries without downloads."""

from __future__ import annotations

import json

import numpy as np

from odmr_bench.datasets import SweepDataset, run_playback
from odmr_bench.dynamics import LinearCenterDrift, SpectralSnapshot
from odmr_bench.emulator import ODMRInstrument, PoissonNoise
from odmr_bench.models import Baseline, Resonance


def main() -> int:
    dataset = SweepDataset(
        signal=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        frequency_hz=np.array([2.80e9, 2.81e9, 2.82e9]),
    )
    playback_signals: list[float] = []
    run_playback(
        dataset, lambda observation: playback_signals.append(observation.signal)
    )

    baseline = Baseline(intercept=1.0, reference_hz=2.87e9)
    resonances = tuple(
        Resonance(
            resonance_id=f"r{index}",
            center_hz=2.805e9 + index * 20.0e6,
            fwhm_hz=2.5e6,
            amplitude=0.015,
            eta=0.5,
        )
        for index in range(8)
    )
    instrument = ODMRInstrument(
        dynamics=LinearCenterDrift(
            SpectralSnapshot(baseline=baseline, resonances=resonances),
            center_slew_hz_per_s=1_000.0,
        ),
        noise=PoissonNoise(),
        nominal_photon_rate_hz=100_000.0,
        frequency_overhead_s=0.001,
        seed=7,
    )
    measurements = [
        instrument.query(2.805e9, 0.01),
        instrument.query(2.825e9, 0.01),
    ]
    print(
        json.dumps(
            {
                "in_memory_playback_signals": playback_signals,
                "synthetic_fluorescence": [
                    sample.fluorescence for sample in measurements
                ],
                "virtual_elapsed_time_s": instrument.resources.virtual_elapsed_time_s,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
