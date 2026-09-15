"""Run one deterministic calibrated two-point diagnostic without downloads."""

from __future__ import annotations

import math

from odmr_bench.dynamics import SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import GaussianNoise, ODMRInstrument
from odmr_bench.estimators import (
    CalibratedTwoPointTracker,
    FitConfiguration,
    TwoPointBudgetCeiling,
    TwoPointIdentityBinding,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    calibrate_two_point,
)
from odmr_bench.evaluation.two_point import (
    TwoPointEvaluatorRunner,
    TwoPointRunnerBudgetStopped,
    VerifiedTwoPointCalibrationSuccess,
)
from odmr_bench.models import Baseline, Resonance

_IDS = tuple(f"r{index}" for index in range(8))
_RESONANCE_VALUES = (
    (2.805e9, 2.5e6, 0.018, 0.35),
    (2.825e9, 2.7e6, 0.021, 0.40),
    (2.845e9, 2.9e6, 0.023, 0.45),
    (2.865e9, 3.1e6, 0.025, 0.50),
    (2.875e9, 3.1e6, 0.024, 0.50),
    (2.895e9, 2.9e6, 0.022, 0.45),
    (2.915e9, 2.7e6, 0.020, 0.40),
    (2.935e9, 2.5e6, 0.017, 0.35),
)
_RATE_HZ = 5.0e8
_OVERHEAD_S = 0.001
_CALIBRATION_INTEGRATION_S = 1.0 / 512.0
_TRACKING_INTEGRATION_S = 0.005


def _snapshot() -> SpectralSnapshot:
    return SpectralSnapshot(
        baseline=Baseline(1.0, 2.870e9, 1.0e-11, 0.0),
        resonances=tuple(
            Resonance(resonance_id, *values)
            for resonance_id, values in zip(_IDS, _RESONANCE_VALUES, strict=True)
        ),
    )


def _instrument(
    snapshot: SpectralSnapshot, seed: int, *, overhead_s: float = _OVERHEAD_S
) -> ODMRInstrument:
    return ODMRInstrument(
        dynamics=StationaryDynamics(snapshot),
        noise=GaussianNoise(stddev_at_1s=0.0),
        nominal_photon_rate_hz=_RATE_HZ,
        frequency_overhead_s=overhead_s,
        seed=seed,
    )


def main() -> None:
    snapshot = _snapshot()
    source_frequency_hz = tuple(2.780e9 + index * 100_000.0 for index in range(1801))
    source_runner = TwoPointEvaluatorRunner.bind(
        _instrument(snapshot, seed=20260903, overhead_s=0.0)
    )
    source_duration_s = len(source_frequency_hz) * _CALIBRATION_INTEGRATION_S
    verified = source_runner.acquire_verified_calibration(
        source_frequency_hz,
        _CALIBRATION_INTEGRATION_S,
        FitConfiguration(
            model_kind="pseudo_voigt",
            baseline_degree=1,
            resonance_ids=_IDS,
            min_fwhm_hz=2.0e5,
            max_fwhm_hz=8.0e6,
            max_amplitude=0.08,
            min_resolved_amplitude=1.0e-4,
            min_center_separation_hz=1.0e6,
            savgol_window=11,
            savgol_polyorder=2,
            relative_prominence=0.01,
            max_nfev=4000,
            min_amplitude_significance=5.0,
        ),
        TwoPointIdentityBinding("require_expected_ids", _IDS),
        source_id="synthetic-calibration-v1",
        source_clock_id="synthetic-calibration-clock-v1",
        tracker_clock_id="synthetic-tracker-clock-v1",
        source_to_tracker_offset_s=-source_duration_s,
        physical_fit_epoch_rule="instrument_midpoint_ordered_mean",
    )
    if type(verified) is not VerifiedTwoPointCalibrationSuccess:
        raise RuntimeError(f"synthetic calibration failed: {verified.code}")

    configuration = TwoPointTrackerConfiguration(
        identity_binding=TwoPointIdentityBinding("require_expected_ids", _IDS),
        integration_time_s=_TRACKING_INTEGRATION_S,
    )
    calibration = calibrate_two_point(
        verified.source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    tracker = CalibratedTwoPointTracker(configuration)
    tracking_runner = TwoPointEvaluatorRunner.bind(_instrument(snapshot, seed=20260905))
    tracking_runner.start_tracking(
        tracker,
        calibration,
        verified,
        TwoPointRunMetadata(
            tracker_clock_id="synthetic-tracker-clock-v1",
            current_sequence_index=None,
            current_timestamp_s=0.0,
            nominal_photon_rate_hz=_RATE_HZ,
            frequency_overhead_s=_OVERHEAD_S,
            fluorescence_quantity="normalized_fluorescence",
        ),
        TwoPointBudgetCeiling(
            max_observations=16,
            max_integration_time_s=None,
            max_nominal_exposure_photons=None,
            max_virtual_elapsed_time_s=None,
        ),
        seed=20260904,
    )
    stopped = tracking_runner.run_until_event()
    if type(stopped) is not TwoPointRunnerBudgetStopped:
        raise RuntimeError(f"synthetic tracking stopped unexpectedly: {stopped.kind}")

    estimate = tracker.estimate()
    resources = stopped.resources.tracking_resources
    timing_by_id = {
        timing.resonance_id: timing for timing in tracking_runner.state.pair_timings
    }
    print("Synthetic calibrated two-point diagnostics")
    print(f"treatment={estimate.calibration_budget_treatment}")
    print(
        "resonance_id lock_state center_hz observations integration_time_s "
        "virtual_elapsed_time_s public_reference_timestamp_s release_timestamp_s"
    )
    for identity in estimate.identities:
        timing = timing_by_id[identity.resonance_id]
        values = (
            identity.center_hz,
            resources.observations,
            resources.integration_time_s,
            resources.virtual_elapsed_time_s,
            timing.public_reference_timestamp_s,
            timing.release_timestamp_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("diagnostic output must be finite")
        print(
            f"{identity.resonance_id} {identity.lock_state} "
            f"{identity.center_hz:.9g} {resources.observations} "
            f"{resources.integration_time_s:.9g} "
            f"{resources.virtual_elapsed_time_s:.9g} "
            f"{timing.public_reference_timestamp_s:.9g} "
            f"{timing.release_timestamp_s:.9g}"
        )


if __name__ == "__main__":
    main()
