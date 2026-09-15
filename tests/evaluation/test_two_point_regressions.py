"""Closed generated acceptance regressions for calibrated two-point tracking.

The numeric bounds in this module characterize one deterministic synthetic
fixture.  They are not experimental results or claims of optimality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import pytest

from odmr_bench.dynamics import LinearCenterDrift, SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import GaussianNoise, PoissonNoise
from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.observations import InstrumentObservation
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

_SOURCE_ID = "stage63-calibration-v1"
_SOURCE_CLOCK_ID = "stage63-calibration-clock-v1"
_TRACKER_CLOCK_ID = "stage63-tracking-clock-v1"
_TRACKER_SEED = 20260904
_SOURCE_SEED = 20260903
_TRACKING_SEED = 20260905
_RATE_HZ = 5.0e8
_OVERHEAD_S = 0.001
_INTEGRATION_S = 0.005
_SOURCE_OFFSET_S = -float.fromhex("0x1.ae2d0e560425fp+4")
_IDS = tuple(f"r{index}" for index in range(8))

_RESONANCES = (
    (2.805e9, 2.5e6, 0.018, 0.35),
    (2.825e9, 2.7e6, 0.021, 0.40),
    (2.845e9, 2.9e6, 0.023, 0.45),
    (2.865e9, 3.1e6, 0.025, 0.50),
    (2.875e9, 3.1e6, 0.024, 0.50),
    (2.895e9, 2.9e6, 0.022, 0.45),
    (2.915e9, 2.7e6, 0.020, 0.40),
    (2.935e9, 2.5e6, 0.017, 0.35),
)


def _source_snapshot() -> SpectralSnapshot:
    return SpectralSnapshot(
        baseline=Baseline(1.0, 2.870e9, 1.0e-11, 0.0),
        resonances=tuple(
            Resonance(resonance_id, *values)
            for resonance_id, values in zip(_IDS, _RESONANCES, strict=True)
        ),
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
        allow_fallback=False,
        max_nfev=4000,
        rank_rtol=1.0e-10,
        min_baseline_sse_improvement=1.0e-4,
        min_amplitude_significance=5.0,
    )


@dataclass(frozen=True)
class _VerifiedFixture:
    runner: TwoPointEvaluatorRunner
    instrument: ODMRInstrument
    success: VerifiedTwoPointCalibrationSuccess
    fitted_snapshot: SpectralSnapshot


def _acquire_source(
    *,
    source_clock_id: str = _SOURCE_CLOCK_ID,
    tracker_clock_id: str = _TRACKER_CLOCK_ID,
    offset_s: float = _SOURCE_OFFSET_S,
) -> _VerifiedFixture:
    instrument = ODMRInstrument(
        dynamics=StationaryDynamics(_source_snapshot()),
        noise=GaussianNoise(stddev_at_1s=0.0),
        nominal_photon_rate_hz=_RATE_HZ,
        frequency_overhead_s=_OVERHEAD_S,
        seed=_SOURCE_SEED,
    )
    runner = TwoPointEvaluatorRunner.bind(instrument)
    configuration = _fit_configuration()
    outcome = runner.acquire_verified_calibration(
        tuple(2.740e9 + index * 62_500.0 for index in range(4481)),
        _INTEGRATION_S,
        configuration,
        TwoPointIdentityBinding("require_expected_ids", _IDS),
        source_id=_SOURCE_ID,
        source_clock_id=source_clock_id,
        tracker_clock_id=tracker_clock_id,
        source_to_tracker_offset_s=offset_s,
        physical_fit_epoch_rule="instrument_midpoint_ordered_mean",
    )
    assert type(outcome) is VerifiedTwoPointCalibrationSuccess, outcome
    assert outcome.source.source_fit.baseline_estimate is not None
    fitted_snapshot = SpectralSnapshot(
        baseline=outcome.source.source_fit.baseline_estimate,
        resonances=tuple(outcome.source.source_fit.resonance_estimates),
    )
    return _VerifiedFixture(runner, instrument, outcome, fitted_snapshot)


@pytest.fixture(scope="module")
def verified_fixture() -> _VerifiedFixture:
    return _acquire_source()


def _tracking_configuration(
    *, common_mode_limit: float | None = None
) -> TwoPointTrackerConfiguration:
    return TwoPointTrackerConfiguration(
        identity_binding=TwoPointIdentityBinding("require_expected_ids", _IDS),
        offset_fwhm_fraction=0.35,
        capture_fwhm_fraction=0.20,
        proportional_gain=1.0,
        max_step_fwhm_fraction=0.10,
        integration_time_s=_INTEGRATION_S,
        common_mode_limit_target_depths=common_mode_limit,
    )


def _start_conditional_run(
    verified: _VerifiedFixture,
    *,
    dynamics: object,
    noise: object,
    seed: int,
    ceiling: TwoPointBudgetCeiling,
    common_mode_limit: float | None = None,
) -> tuple[TwoPointEvaluatorRunner, ODMRInstrument, CalibratedTwoPointTracker]:
    instrument = ODMRInstrument(
        dynamics=dynamics,  # type: ignore[arg-type]
        noise=noise,
        nominal_photon_rate_hz=_RATE_HZ,
        frequency_overhead_s=_OVERHEAD_S,
        seed=seed,
    )
    runner = TwoPointEvaluatorRunner.bind(instrument)
    configuration = _tracking_configuration(common_mode_limit=common_mode_limit)
    tracker = CalibratedTwoPointTracker(configuration)
    calibration = calibrate_two_point(
        verified.success.source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    runner.start_tracking(
        tracker,
        calibration,
        verified.success,
        TwoPointRunMetadata(
            tracker_clock_id=_TRACKER_CLOCK_ID,
            current_sequence_index=None,
            current_timestamp_s=0.0,
            nominal_photon_rate_hz=_RATE_HZ,
            frequency_overhead_s=_OVERHEAD_S,
            fluorescence_quantity="normalized_fluorescence",
        ),
        ceiling,
        seed=_TRACKER_SEED,
    )
    return runner, instrument, tracker


def _cap(
    observations: int,
    integration_hex: str,
    nominal: float,
    elapsed_hex: str,
) -> TwoPointBudgetCeiling:
    return TwoPointBudgetCeiling(
        observations,
        float.fromhex(integration_hex),
        nominal,
        float.fromhex(elapsed_hex),
    )


def _assert_terminal_resources(
    outcome: TwoPointRunnerBudgetStopped,
    instrument: ODMRInstrument,
    *,
    observations: int,
    integration_hex: str,
    nominal: float,
    elapsed_hex: str,
    endpoint_hex: str,
) -> None:
    tracking = outcome.resources.tracking_resources
    assert tracking.observations == observations
    assert tracking.integration_time_s.hex() == integration_hex
    assert tracking.nominal_exposure_photons == nominal
    assert tracking.virtual_elapsed_time_s.hex() == elapsed_hex
    assert outcome.resources.charged_resources == tracking
    assert instrument.resources == tracking
    assert instrument.virtual_time_s.hex() == endpoint_hex
    assert outcome.state.tracker_estimate is not None
    assert outcome.state.tracker_estimate.stopped_reason == "budget_exhausted"


def test_verified_source_fixture_matches_closed_witnesses(
    verified_fixture: _VerifiedFixture,
) -> None:
    success = verified_fixture.success
    source = success.source
    assert source.source_id == _SOURCE_ID
    assert source.source_fit.success
    assert source.fit_configuration == _fit_configuration()
    assert source.identity_binding == TwoPointIdentityBinding(
        "require_expected_ids", _IDS
    )
    assert source.resolved_resonance_ids == _IDS
    assert source.source_frequency_min_hz == 2.740e9
    assert source.source_frequency_max_hz == 3.020e9
    assert source.source_frequency_overhead_s == _OVERHEAD_S
    assert source.fluorescence_provenance.nominal_photon_rate_hz == _RATE_HZ
    assert source.fluorescence_provenance.sampling_rules == ("gaussian",) * 4481
    assert source.clock_mapping.source_clock_id == _SOURCE_CLOCK_ID
    assert source.clock_mapping.tracker_clock_id == _TRACKER_CLOCK_ID
    assert source.clock_mapping.offset_s == _SOURCE_OFFSET_S
    assert source.source_first_sequence_index == 0
    assert source.source_last_sequence_index == 4480
    assert source.source_start_timestamp_s.hex() == "0x0.0p+0"
    assert source.source_first_timestamp_s.hex() == "0x1.89374bc6a7efap-8"
    assert source.source_last_timestamp_s.hex() == "0x1.ae2d0e560425fp+4"
    assert success.measurement_midpoints_s[0].hex() == "0x1.cac083126e979p-9"
    assert success.measurement_midpoints_s[-1].hex() == "0x1.ae22d0e5604efp+4"
    assert source.physical_fit_epoch_s.hex() == "0x1.ae3126e978e27p+3"
    public_last_midpoint = (
        source.source_observations[-1].timestamp_s
        - source.source_observations[-1].integration_time_s / 2.0
    )
    assert public_last_midpoint.hex() == "0x1.ae22d0e5604eep+4"
    public_epoch = (
        source.source_observations[0].timestamp_s
        - source.source_observations[0].integration_time_s / 2.0
    )
    public_epoch += (public_last_midpoint - public_epoch) / 2.0
    assert public_epoch.hex() == "0x1.ae3126e978e26p+3"
    assert (source.physical_fit_epoch_s + _SOURCE_OFFSET_S).hex() == (
        "-0x1.ae28f5c28f697p+3"
    )
    assert (public_epoch + _SOURCE_OFFSET_S).hex() == "-0x1.ae28f5c28f698p+3"
    assert success.full_resources.observations == 4481
    assert success.full_resources.integration_time_s.hex() == "0x1.667ae147ae117p+4"
    assert success.full_resources.nominal_exposure_photons == 11_202_500_000
    assert success.full_resources.realized_photons == 0
    assert success.full_resources.observations_without_realized_counts == 4481
    assert success.full_resources.virtual_elapsed_time_s.hex() == (
        "0x1.ae2d0e5604269p+4"
    )


def test_static_noiseless_two_cycles_match_exact_schedule_resources_and_zero_steps(
    verified_fixture: _VerifiedFixture,
) -> None:
    runner, instrument, tracker = _start_conditional_run(
        verified_fixture,
        dynamics=StationaryDynamics(verified_fixture.fitted_snapshot),
        noise=GaussianNoise(0.0),
        seed=_TRACKING_SEED,
        ceiling=_cap(32, "0x1.47ae147ae147dp-3", 80_000_000, "0x1.89374bc6a7efdp-3"),
    )
    outcome = runner.run_until_event()
    assert type(outcome) is TwoPointRunnerBudgetStopped
    estimate = tracker.estimate()
    fitted_resonances = verified_fixture.success.source.source_fit.resonance_estimates
    assert len(runner.state.normal_tracking_trace) == 32
    assert tuple(pair.resonance_id for pair in estimate.pair_history) == _IDS * 2
    assert tuple(
        acquisition.query.query_index
        for acquisition in runner.state.normal_tracking_trace
    ) == tuple(range(32))
    assert tuple(
        acquisition.full_observation.sequence_index
        for acquisition in runner.state.normal_tracking_trace
    ) == tuple(range(32))
    expected_query_frequencies: list[float] = []
    for cycle in range(2):
        for resonance in fitted_resonances:
            offset_hz = 0.35 * resonance.fwhm_hz
            minus_hz = resonance.center_hz - offset_hz
            plus_hz = resonance.center_hz + offset_hz
            expected_query_frequencies.extend(
                (minus_hz, plus_hz) if cycle == 0 else (plus_hz, minus_hz)
            )
    assert tuple(
        acquisition.query.frequency_hz
        for acquisition in runner.state.normal_tracking_trace
    ) == tuple(expected_query_frequencies)
    assert tuple(
        acquisition.query.side for acquisition in runner.state.normal_tracking_trace
    ) == (("minus", "plus") * 8 + ("plus", "minus") * 8)
    expected_endpoints: list[float] = []
    expected_endpoint_s = 0.0
    for _ in range(32):
        expected_endpoint_s = (expected_endpoint_s + _OVERHEAD_S) + _INTEGRATION_S
        expected_endpoints.append(expected_endpoint_s)
    assert tuple(
        acquisition.full_observation.timestamp_s
        for acquisition in runner.state.normal_tracking_trace
    ) == tuple(expected_endpoints)
    assert tuple(
        acquisition.query.expected_end_timestamp_s
        for acquisition in runner.state.normal_tracking_trace
    ) == tuple(expected_endpoints)
    assert tuple(pair.first_side for pair in estimate.pair_history) == (
        ("minus",) * 8 + ("plus",) * 8
    )
    assert estimate.accepted_observations == 32
    assert estimate.completed_pairs == 16
    assert estimate.current_sequence_index == 31
    assert estimate.current_timestamp_s == instrument.virtual_time_s
    for pair in estimate.pair_history:
        fitted = fitted_resonances[_IDS.index(pair.resonance_id)]
        assert pair.interrogation_center_hz == fitted.center_hz
        assert pair.discriminator == pair.zero_discriminator
        assert pair.raw_innovation_hz == 0.0
        assert pair.requested_step_hz == 0.0
        assert pair.applied_step_hz == 0.0
        assert pair.candidate_center_hz == pair.interrogation_center_hz
        assert pair.lock_state == "tracking"
    assert all(
        identity.active_source_kind == "pair" for identity in estimate.identities
    )
    assert tuple(
        identity.active_source_pair_index for identity in estimate.identities
    ) == tuple(range(8, 16))
    assert (
        tuple(identity.completed_pairs for identity in estimate.identities) == (2,) * 8
    )
    assert tracker.calibration is not None
    for identity, cell in zip(
        estimate.identities, tracker.calibration.identities, strict=True
    ):
        fitted = fitted_resonances[_IDS.index(identity.resonance_id)]
        assert identity.center_hz == fitted.center_hz
        assert identity.calibration_cell_lower_hz == cell.calibration_cell_lower_hz
        assert identity.calibration_cell_upper_hz == cell.calibration_cell_upper_hz
        assert identity.latest_pair is not None
        assert (
            identity.active_reference_timestamp_s
            == identity.latest_pair.pair_reference_timestamp_s
        )
        assert (
            identity.active_release_timestamp_s
            == identity.latest_pair.release_timestamp_s
        )
        assert identity.estimate_age_sequence_indices == (
            estimate.current_sequence_index
            - identity.latest_pair.release_sequence_index
        )
        assert identity.estimate_age_s == (
            estimate.current_timestamp_s
            - identity.latest_pair.pair_reference_timestamp_s
        )
        assert identity.release_age_s == (
            estimate.current_timestamp_s - identity.latest_pair.release_timestamp_s
        )
    _assert_terminal_resources(
        outcome,
        instrument,
        observations=32,
        integration_hex="0x1.47ae147ae147dp-3",
        nominal=80_000_000,
        elapsed_hex="0x1.89374bc6a7efdp-3",
        endpoint_hex="0x1.89374bc6a7efep-3",
    )


def _poisson_run(verified: _VerifiedFixture, seed: int) -> tuple[Any, ...]:
    runner, instrument, tracker = _start_conditional_run(
        verified,
        dynamics=StationaryDynamics(verified.fitted_snapshot),
        noise=PoissonNoise(),
        seed=seed,
        ceiling=_cap(64, "0x1.47ae147ae147ep-2", 160_000_000, "0x1.89374bc6a7efep-2"),
    )
    outcome = runner.run_until_event()
    assert type(outcome) is TwoPointRunnerBudgetStopped
    _assert_terminal_resources(
        outcome,
        instrument,
        observations=64,
        integration_hex="0x1.47ae147ae147ep-2",
        nominal=160_000_000,
        elapsed_hex="0x1.89374bc6a7efep-2",
        endpoint_hex="0x1.89374bc6a7effp-2",
    )
    return (
        tuple(acquisition.query for acquisition in runner.state.normal_tracking_trace),
        tuple(
            acquisition.full_observation
            for acquisition in runner.state.normal_tracking_trace
        ),
        runner.state.pair_timings,
        tracker.estimate(),
        outcome.resources,
    )


def test_static_poisson_four_cycles_are_seed_reproducible(
    verified_fixture: _VerifiedFixture,
) -> None:
    first = _poisson_run(verified_fixture, _TRACKING_SEED)
    second = _poisson_run(verified_fixture, _TRACKING_SEED)
    changed = _poisson_run(verified_fixture, _TRACKING_SEED + 1)
    assert first == second
    first_counts = tuple(observation.realized_photons for observation in first[1])
    changed_counts = tuple(observation.realized_photons for observation in changed[1])
    assert any(
        left != right for left, right in zip(first_counts, changed_counts, strict=True)
    )


@dataclass
class _RejectingPostReleaseTruthOracle:
    dynamics: LinearCenterDrift

    def __post_init__(self) -> None:
        self._released: dict[int, tuple[float, float]] = {}
        self._called: set[int] = set()

    def release(self, pair_index: int, actual_s: float, public_s: float) -> None:
        self._released[pair_index] = (actual_s, public_s)

    def lookup(self, pair_index: int, timestamp_s: float) -> SpectralSnapshot:
        assert pair_index in self._released
        assert pair_index not in self._called
        actual_s, public_s = self._released[pair_index]
        assert timestamp_s == actual_s
        if pair_index == 3:
            assert timestamp_s != public_s
        self._called.add(pair_index)
        return self.dynamics.snapshot_at(timestamp_s)


def test_common_linear_drift_thirty_cycles_tracks_declared_fixture(
    verified_fixture: _VerifiedFixture,
) -> None:
    acquisition_dynamics = LinearCenterDrift(verified_fixture.fitted_snapshot, 5.0e5)
    truth_oracle = _RejectingPostReleaseTruthOracle(
        LinearCenterDrift(verified_fixture.fitted_snapshot, 5.0e5)
    )
    assert acquisition_dynamics is not truth_oracle.dynamics
    runner, instrument, tracker = _start_conditional_run(
        verified_fixture,
        dynamics=acquisition_dynamics,
        noise=GaussianNoise(0.0),
        seed=_TRACKING_SEED,
        ceiling=_cap(
            480, "0x1.33333333332f2p+1", 1_200_000_000, "0x1.70a3d70a3d6c6p+1"
        ),
    )
    outcome = runner.run_until_event()
    assert type(outcome) is TwoPointRunnerBudgetStopped
    estimate = tracker.estimate()
    assert tuple(pair.resonance_id for pair in estimate.pair_history) == _IDS * 30
    for pair, timing in zip(
        estimate.pair_history, runner.state.pair_timings, strict=True
    ):
        truth_oracle.release(
            pair.pair_index,
            timing.truth_reference_timestamp_s,
            timing.public_reference_timestamp_s,
        )
        truth = truth_oracle.lookup(pair.pair_index, timing.truth_reference_timestamp_s)
        true_resonance = next(
            item for item in truth.resonances if item.resonance_id == pair.resonance_id
        )
        true_center = true_resonance.center_hz
        width = true_resonance.fwhm_hz
        displacement = true_center - pair.interrogation_center_hz
        assert pair.lock_state != "lost"
        assert pair.candidate_center_hz is not None
        assert pair.discriminator is not None and math.isfinite(pair.discriminator)
        assert pair.zero_discriminator is not None and math.isfinite(
            pair.zero_discriminator
        )
        assert pair.raw_innovation_hz is not None and math.isfinite(
            pair.raw_innovation_hz
        )
        identity = estimate.identities[_IDS.index(pair.resonance_id)]
        assert (
            identity.calibration_cell_lower_hz
            <= pair.minus_query.frequency_hz
            < pair.plus_query.frequency_hz
            <= identity.calibration_cell_upper_hz
        )
        assert pair.minus_query.frequency_hz < pair.plus_query.frequency_hz
        if abs(displacement) > 1.0e-12 * width:
            assert pair.applied_step_hz * displacement > 0.0
        assert abs(pair.candidate_center_hz - true_center) < 0.05 * width
    assert truth_oracle._called == set(range(240))
    _assert_terminal_resources(
        outcome,
        instrument,
        observations=480,
        integration_hex="0x1.33333333332f2p+1",
        nominal=1_200_000_000,
        elapsed_hex="0x1.70a3d70a3d6c6p+1",
        endpoint_hex="0x1.70a3d70a3d673p+1",
    )


@dataclass(frozen=True)
class _AmplitudeWindowDynamics:
    initial_snapshot: SpectralSnapshot
    start_s: float
    stop_s: float

    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        resonances = tuple(
            replace(resonance, amplitude=0.0)
            if resonance.resonance_id == "r3"
            and self.start_s <= timestamp_s <= self.stop_s
            else resonance
            for resonance in self.initial_snapshot.resonances
        )
        return SpectralSnapshot(self.initial_snapshot.baseline, resonances)


def test_contrast_loss_pair_eleven_and_pair_nineteen_recovery_are_exact(
    verified_fixture: _VerifiedFixture,
) -> None:
    start_s = float.fromhex("0x1.15810624dd2f4p-3")
    stop_s = float.fromhex("0x1.21cac083126ecp-3")
    runner, instrument, tracker = _start_conditional_run(
        verified_fixture,
        dynamics=_AmplitudeWindowDynamics(
            verified_fixture.fitted_snapshot, start_s, stop_s
        ),
        noise=GaussianNoise(0.0),
        seed=_TRACKING_SEED,
        ceiling=_cap(48, "0x1.eb851eb851ebdp-3", 120_000_000, "0x1.26e978d4fdf3ep-2"),
        common_mode_limit=0.5,
    )
    assert tracker.calibration is not None
    cells_before_loss = tracker.calibration.identities
    outcome = runner.run_until_event()
    assert type(outcome) is TwoPointRunnerBudgetStopped
    trace = runner.state.normal_tracking_trace
    assert tuple(trace[index].query.resonance_id for index in (22, 23)) == (
        "r3",
        "r3",
    )
    assert tuple(trace[index].query.side for index in (22, 23)) == ("plus", "minus")
    assert tuple(trace[index].measurement_midpoint_s.hex() for index in (22, 23)) == (
        "0x1.15810624dd2f4p-3",
        "0x1.21cac083126ecp-3",
    )
    loss_pair = tracker.estimate().pair_history[11]
    assert loss_pair.identity_pair_index == 1
    assert loss_pair.resonance_id == "r3"
    assert loss_pair.common_mode_target_depths is not None
    assert abs(loss_pair.common_mode_target_depths - 1.0) <= 1.0e-12
    assert loss_pair.failure_code == "common_mode_limit_exceeded"
    assert loss_pair.lock_state == "lost"
    assert loss_pair.applied_step_hz == 0.0
    assert tracker.calibration is not None
    assert tracker.calibration.identities == cells_before_loss
    pair_history = tracker.estimate().pair_history
    assert tuple(
        pair.pair_index for pair in pair_history if pair.lock_state == "lost"
    ) == (11,)
    assert tuple(
        pair.pair_index
        for pair in pair_history
        if pair.failure_code == "common_mode_limit_exceeded"
    ) == (11,)
    assert all(
        pair.lock_state != "lost" and pair.failure_code is None
        for pair in pair_history
        if pair.pair_index != 11
    )
    recovery = tracker.estimate().pair_history[19]
    assert tuple(trace[index].query.resonance_id for index in (38, 39)) == (
        "r3",
        "r3",
    )
    assert recovery.resonance_id == "r3"
    assert recovery.identity_pair_index == 2
    assert recovery.lock_state in {"tracking", "step_limited"}
    _assert_terminal_resources(
        outcome,
        instrument,
        observations=48,
        integration_hex="0x1.eb851eb851ebdp-3",
        nominal=120_000_000,
        elapsed_hex="0x1.26e978d4fdf3ep-2",
        endpoint_hex="0x1.26e978d4fdf3fp-2",
    )


def test_included_same_run_one_pair_charges_source_once_without_regrouping() -> None:
    verified = _acquire_source(
        source_clock_id=_TRACKER_CLOCK_ID,
        tracker_clock_id=_TRACKER_CLOCK_ID,
        offset_s=0.0,
    )
    configuration = _tracking_configuration()
    tracker = CalibratedTwoPointTracker(configuration)
    calibration = calibrate_two_point(
        verified.success.source,
        configuration,
        budget_treatment="included_same_run",
    )
    verified.runner.start_tracking(
        tracker,
        calibration,
        verified.success,
        TwoPointRunMetadata(
            tracker_clock_id=_TRACKER_CLOCK_ID,
            current_sequence_index=4480,
            current_timestamp_s=verified.instrument.virtual_time_s,
            nominal_photon_rate_hz=_RATE_HZ,
            frequency_overhead_s=_OVERHEAD_S,
            fluorescence_quantity="normalized_fluorescence",
        ),
        _cap(4483, "0x1.66a3d70a3d6d9p+4", 11_207_500_000, "0x1.ae5e353f7cfb9p+4"),
        seed=_TRACKER_SEED,
    )
    outcome = verified.runner.run_until_event()
    assert type(outcome) is TwoPointRunnerBudgetStopped
    assert len(outcome.resources.calibration_observations) == 4481
    assert len(outcome.resources.accepted_tracking_observations) == 2
    assert outcome.resources.charged_resources.observations == 4483
    assert outcome.resources.charged_resources.integration_time_s.hex() == (
        "0x1.66a3d70a3d6d9p+4"
    )
    assert (
        outcome.resources.charged_resources.nominal_exposure_photons == 11_207_500_000
    )
    assert outcome.resources.charged_resources.virtual_elapsed_time_s.hex() == (
        "0x1.ae5e353f7cfb9p+4"
    )
    assert verified.instrument.virtual_time_s.hex() == "0x1.ae5e353f7cfafp+4"
    assert tracker.estimate().stopped_reason == "budget_exhausted"


_FORBIDDEN_RETAINED_NAMES = frozenset(
    {
        "dynamics",
        "expected_photons",
        "full_observation",
        "full_observations",
        "future_observation",
        "future_observations",
        "noiseless_callback",
        "truth",
        "truth_oracle",
    }
)


def _retained_graph(root: object) -> tuple[tuple[str, object], ...]:
    seen: set[int] = set()
    found: list[tuple[str, object]] = []
    pending = [("root", root)]
    while pending:
        path, value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        found.append((path, value))
        if is_dataclass(value):
            pending.extend(
                (f"{path}.{field.name}", getattr(value, field.name))
                for field in fields(value)
            )
        if isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(
                (f"{path}[{index}]", item) for index, item in enumerate(value)
            )
        if isinstance(value, dict):
            pending.extend(
                (f"{path}.key[{index}]", item)
                for index, item in enumerate(value.keys())
            )
            pending.extend((f"{path}[{key!r}]", item) for key, item in value.items())
        instance_dict = getattr(value, "__dict__", None)
        if isinstance(instance_dict, dict):
            pending.extend(
                (f"{path}.{name}", item) for name, item in instance_dict.items()
            )
        for owner in type(value).__mro__:
            slots = owner.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in {"__dict__", "__weakref__"}:
                    continue
                try:
                    item = getattr(value, name)
                except (AttributeError, TypeError, ValueError):
                    continue
                pending.append((f"{path}.{name}", item))
    return tuple(found)


def _retained_graph_violations(root: object) -> tuple[str, ...]:
    forbidden_types = (
        InstrumentObservation,
        SpectralSnapshot,
        LinearCenterDrift,
        StationaryDynamics,
        _AmplitudeWindowDynamics,
        _RejectingPostReleaseTruthOracle,
    )
    violations: list[str] = []
    for path, value in _retained_graph(root):
        terminal_name = path.rsplit(".", maxsplit=1)[-1]
        if isinstance(value, forbidden_types):
            violations.append(f"{path}:forbidden_type")
        if terminal_name in _FORBIDDEN_RETAINED_NAMES:
            violations.append(f"{path}:forbidden_name")
        if callable(getattr(value, "snapshot_at", None)):
            violations.append(f"{path}:hidden_snapshot_capability")
    return tuple(violations)


class _ForbiddenSlotSentinel:
    __slots__ = ("full_observation",)

    def __init__(self) -> None:
        self.full_observation = object()


def test_truth_oracle_rejects_public_reference_pre_release_and_duplicate_lookup(
    verified_fixture: _VerifiedFixture,
) -> None:
    acquisition_dynamics = LinearCenterDrift(verified_fixture.fitted_snapshot, 5.0e5)
    oracle = _RejectingPostReleaseTruthOracle(
        LinearCenterDrift(verified_fixture.fitted_snapshot, 5.0e5)
    )
    runner, _, tracker = _start_conditional_run(
        verified_fixture,
        dynamics=acquisition_dynamics,
        noise=GaussianNoise(0.0),
        seed=_TRACKING_SEED,
        ceiling=_cap(
            8,
            float(8 * _INTEGRATION_S).hex(),
            20_000_000,
            float(8 * (_OVERHEAD_S + _INTEGRATION_S)).hex(),
        ),
    )
    with pytest.raises(AssertionError):
        oracle.lookup(0, 0.0)
    for _ in range(8):
        accepted = runner.step()
        if len(accepted.state.pair_timings) > len(oracle._released):
            timing = accepted.state.pair_timings[-1]
            oracle.release(
                timing.pair_index,
                timing.truth_reference_timestamp_s,
                timing.public_reference_timestamp_s,
            )
            if timing.pair_index == 3:
                with pytest.raises(AssertionError):
                    oracle.lookup(3, timing.public_reference_timestamp_s)
            oracle.lookup(timing.pair_index, timing.truth_reference_timestamp_s)
    pair_three = runner.state.pair_timings[3]
    assert pair_three.truth_reference_timestamp_s.hex() == "0x1.5c28f5c28f5c4p-5"
    assert pair_three.public_reference_timestamp_s.hex() == "0x1.5c28f5c28f5c2p-5"
    with pytest.raises(AssertionError):
        oracle.lookup(3, pair_three.truth_reference_timestamp_s)
    assert oracle._called == {0, 1, 2, 3}

    retained = _retained_graph(tracker)
    assert retained[0] == ("root", tracker)
    assert any(path == "root._state" for path, _ in retained)
    assert any(path == "root._configuration" for path, _ in retained)
    assert _retained_graph_violations(tracker) == ()
    sentinel_violations = _retained_graph_violations(_ForbiddenSlotSentinel())
    assert sentinel_violations == ("root.full_observation:forbidden_name",)
