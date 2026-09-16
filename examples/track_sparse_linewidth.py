"""Run one deterministic five-point sparse-linewidth diagnostic.

This download-free example deliberately reports public/evaluator diagnostics,
not accuracy, sensitivity, or comparative benchmark claims.
"""

from __future__ import annotations

from odmr_bench.dynamics import SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import GaussianNoise, ODMRInstrument, ResourceSnapshot
from odmr_bench.estimators import (
    FitConfiguration,
    SparseLinewidthCompositeTracker,
    SparseLinewidthConfiguration,
    TwoPointBudgetCeiling,
    TwoPointIdentityBinding,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    calibrate_two_point,
)
from odmr_bench.evaluation.sparse_linewidth import (
    SparseLinewidthEvaluatorRunner,
    SparseRunnerBudgetStopped,
)
from odmr_bench.evaluation.two_point import (
    TwoPointEvaluatorRunner,
    VerifiedTwoPointCalibrationSuccess,
)
from odmr_bench.models import Baseline, Resonance

_IDS = tuple(f"r{index}" for index in range(8))
_RATE_HZ = 2.5e9
_OVERHEAD_S = 0.001
_CALIBRATION_INTEGRATION_S = 1.0 / 512.0
_TRACKING_INTEGRATION_S = 0.005


def _snapshot() -> SpectralSnapshot:
    return SpectralSnapshot(
        baseline=Baseline(1.0, 2.88e9, 0.0, 0.0),
        resonances=tuple(
            Resonance(
                resonance_id=resonance_id,
                center_hz=2.76e9 + index * 34.0e6,
                fwhm_hz=1.5e6,
                amplitude=0.02,
                eta=0.5,
            )
            for index, resonance_id in enumerate(_IDS)
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


def _fit_configuration() -> FitConfiguration:
    return FitConfiguration(
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
    )


def _format_ledger(label: str, ledger: ResourceSnapshot) -> str:
    return (
        f"{label}=observations:{ledger.observations},"
        f"integration_time_s:{ledger.integration_time_s:.9g},"
        f"nominal_exposure_photons:{ledger.nominal_exposure_photons:.9g},"
        f"virtual_elapsed_time_s:{ledger.virtual_elapsed_time_s:.9g}"
    )


def main() -> None:
    snapshot = _snapshot()
    source_frequency_hz = tuple(2.74e9 + index * 100_000.0 for index in range(2801))
    source_runner = TwoPointEvaluatorRunner.bind(
        _instrument(snapshot, seed=20260914, overhead_s=0.0)
    )
    source_duration_s = len(source_frequency_hz) * _CALIBRATION_INTEGRATION_S
    verified = source_runner.acquire_verified_calibration(
        source_frequency_hz,
        _CALIBRATION_INTEGRATION_S,
        _fit_configuration(),
        TwoPointIdentityBinding("require_expected_ids", _IDS),
        source_id="synthetic-sparse-calibration-v1",
        source_clock_id="synthetic-calibration-clock-v1",
        tracker_clock_id="synthetic-tracker-clock-v1",
        source_to_tracker_offset_s=-source_duration_s,
        physical_fit_epoch_rule="instrument_midpoint_ordered_mean",
    )
    if type(verified) is not VerifiedTwoPointCalibrationSuccess:
        raise RuntimeError(f"synthetic calibration failed: {verified.code}")

    calibration = calibrate_two_point(
        verified.source,
        TwoPointTrackerConfiguration(integration_time_s=_TRACKING_INTEGRATION_S),
        budget_treatment="conditional_free_precalibration",
    )
    tracker = SparseLinewidthCompositeTracker(
        SparseLinewidthConfiguration(integration_time_s=_TRACKING_INTEGRATION_S)
    )
    runner = SparseLinewidthEvaluatorRunner.bind(_instrument(snapshot, seed=20260915))
    runner.start_tracking(
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
        TwoPointBudgetCeiling(21, None, None, None),
        seed=20260916,
    )
    terminal = runner.run_until_event()
    if type(terminal) is not SparseRunnerBudgetStopped:
        raise RuntimeError(f"synthetic tracking stopped unexpectedly: {terminal.kind}")

    estimate = tracker.estimate()
    identity = estimate.identities[0]
    scan = estimate.sparse_scan_history[0]
    timing = runner.state.scan_timings[0]
    if scan.fitted_q is None:
        raise RuntimeError(f"synthetic sparse fit failed: {scan.failure_code}")

    print("Synthetic sparse-linewidth diagnostics")
    print(f"terminal_phase={terminal.state.phase}")
    print(f"treatment={estimate.calibration_budget_treatment}")
    print(f"live_projection_q={identity.live_q:.12g}")
    print(f"scan_local_q={scan.fitted_q:.12g}")
    print(f"center_source_epoch_s={identity.fast_center_reference_timestamp_s:.12g}")
    print(f"linewidth_source_epoch_s={identity.fwhm_reference_timestamp_s:.12g}")
    print(f"public_reference_timestamp_s={timing.public_reference_timestamp_s:.12g}")
    print(f"truth_reference_timestamp_s={timing.truth_reference_timestamp_s:.12g}")
    print(f"completed_fast_pairs={estimate.completed_fast_pairs}")
    print(f"completed_sparse_scans={estimate.completed_sparse_scans}")
    print(
        _format_ledger(
            "accepted_tracking_ledger", terminal.resources.tracking_resources
        )
    )
    print(_format_ledger("charged_ledger", terminal.resources.charged_resources))


if __name__ == "__main__":
    main()
