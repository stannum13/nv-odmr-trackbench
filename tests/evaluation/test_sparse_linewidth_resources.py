"""Resource-accounting tests at the sparse evaluator tracking boundary."""

from __future__ import annotations

import copy

import pytest

from odmr_bench.dynamics import SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import GaussianNoise, InstrumentObservation, ResourceSnapshot
from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.estimators import (
    SparseLinewidthCompositeTracker,
    SparseLinewidthConfiguration,
    TwoPointBudgetCeiling,
    TwoPointIdentityBinding,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    calibrate_two_point,
)
from odmr_bench.evaluation.sparse_linewidth.resource_accounting import (
    build_sparse_linewidth_evaluator_resources,
)
from odmr_bench.evaluation.sparse_linewidth.runner import (
    SparseLinewidthEvaluatorRunner,
)
from odmr_bench.evaluation.sparse_linewidth.types import SparseRunnerStateError
from odmr_bench.evaluation.two_point.types import (
    VerifiedTwoPointCalibrationFailure,
    VerifiedTwoPointCalibrationSuccess,
)
from odmr_bench.models import Baseline, Resonance
from tests.two_point_helpers import make_legal_fit_configuration, make_legal_source_fit


def _snapshot() -> SpectralSnapshot:
    return SpectralSnapshot(
        baseline=Baseline(intercept=1.0, reference_hz=2.88e9),
        resonances=tuple(
            Resonance(
                resonance_id=f"r{index}",
                center_hz=2.76e9 + index * 34.0e6,
                fwhm_hz=1.5e6,
                amplitude=0.02,
                eta=0.5,
            )
            for index in range(8)
        ),
    )


def _instrument() -> ODMRInstrument:
    return ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=GaussianNoise(stddev_at_1s=0.0),
        nominal_photon_rate_hz=2.5e6,
        frequency_overhead_s=0.001,
        seed=29,
    )


def _calibration_arguments(
    *,
    source_clock_id: str = "clock",
    tracker_clock_id: str = "clock",
    source_to_tracker_offset_s: float = 0.0,
) -> dict[str, object]:
    fit_configuration = make_legal_fit_configuration()
    return {
        "frequency_hz": (2.74e9, 3.02e9),
        "integration_time_s": 0.005,
        "fit_configuration": fit_configuration,
        "identity_binding": TwoPointIdentityBinding(
            "require_expected_ids", fit_configuration.resonance_ids
        ),
        "source_id": "verified-source",
        "source_clock_id": source_clock_id,
        "tracker_clock_id": tracker_clock_id,
        "source_to_tracker_offset_s": source_to_tracker_offset_s,
        "physical_fit_epoch_rule": "instrument_midpoint_ordered_mean",
    }


def _acquire_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_clock_id: str = "clock",
    tracker_clock_id: str = "clock",
    source_to_tracker_offset_s: float = 0.0,
) -> tuple[
    SparseLinewidthEvaluatorRunner,
    ODMRInstrument,
    VerifiedTwoPointCalibrationSuccess,
]:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    arguments = _calibration_arguments(
        source_clock_id=source_clock_id,
        tracker_clock_id=tracker_clock_id,
        source_to_tracker_offset_s=source_to_tracker_offset_s,
    )
    fit_configuration = arguments["fit_configuration"]
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: make_legal_source_fit(
            fit_configuration  # type: ignore[arg-type]
        ),
    )
    instrument = _instrument()
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)
    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]
    assert type(outcome) is VerifiedTwoPointCalibrationSuccess
    return runner, instrument, outcome


def _start_included(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    SparseLinewidthEvaluatorRunner,
    ODMRInstrument,
    VerifiedTwoPointCalibrationSuccess,
]:
    runner, instrument, success = _acquire_success(monkeypatch)
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="included_same_run",
    )
    runner.start_tracking(
        SparseLinewidthCompositeTracker(SparseLinewidthConfiguration()),
        calibration,
        success,
        TwoPointRunMetadata(
            tracker_clock_id="clock",
            current_sequence_index=runner.state.instrument_current_sequence_index,
            current_timestamp_s=runner.state.current_virtual_time_s,
            nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
            frequency_overhead_s=instrument.frequency_overhead_s,
            fluorescence_quantity="normalized_fluorescence",
        ),
        TwoPointBudgetCeiling(100, None, None, None),
        seed=3,
    )
    return runner, instrument, success


def _start_conditional(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    SparseLinewidthEvaluatorRunner,
    ODMRInstrument,
    VerifiedTwoPointCalibrationSuccess,
]:
    _, _, success = _acquire_success(
        monkeypatch,
        source_clock_id="source-clock",
        tracker_clock_id="tracker-clock",
        source_to_tracker_offset_s=-0.012,
    )
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="conditional_free_precalibration",
    )
    instrument = _instrument()
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)
    runner.start_tracking(
        SparseLinewidthCompositeTracker(SparseLinewidthConfiguration()),
        calibration,
        success,
        TwoPointRunMetadata(
            tracker_clock_id="tracker-clock",
            current_sequence_index=None,
            current_timestamp_s=0.0,
            nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
            frequency_overhead_s=instrument.frequency_overhead_s,
            fluorescence_quantity="normalized_fluorescence",
        ),
        TwoPointBudgetCeiling(100, None, None, None),
        seed=5,
    )
    return runner, instrument, success


def _replay(
    observations: tuple[InstrumentObservation, ...], overhead_s: float
) -> ResourceSnapshot:
    result = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    for observation in observations:
        result = ResourceSnapshot(
            result.observations + 1,
            result.integration_time_s + observation.integration_time_s,
            result.nominal_exposure_photons
            + observation.nominal_exposure_photons,
            result.expected_photons + observation.expected_photons,
            result.realized_photons
            + (
                observation.realized_photons
                if observation.realized_photons is not None
                else 0
            ),
            result.observations_without_realized_counts
            + int(observation.realized_photons is None),
            result.virtual_elapsed_time_s
            + overhead_s
            + observation.integration_time_s,
        )
    return result


def _assert_prestart_state_is_zero(runner: SparseLinewidthEvaluatorRunner) -> None:
    state = runner.state
    assert state.normal_tracking_trace == ()
    assert state.pair_timings == ()
    assert state.scan_timings == ()
    assert state.tracker_estimate is None
    assert state.tracking_resources_before is None
    assert state.fast_update_cpu_time_s == 0.0
    assert state.sparse_update_cpu_time_s == 0.0
    assert state.total_update_cpu_time_s == 0.0


def test_resource_builder_rejects_every_prestart_phase_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = SparseLinewidthEvaluatorRunner.bind(_instrument())
    succeeded, _, _ = _acquire_success(monkeypatch)
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    failed = SparseLinewidthEvaluatorRunner.bind(_instrument())
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    failure = failed.acquire_verified_calibration(  # type: ignore[arg-type]
        **_calibration_arguments()
    )
    assert type(failure) is VerifiedTwoPointCalibrationFailure

    for runner, phase in (
        (ready, "ready"),
        (succeeded, "calibration_succeeded"),
        (failed, "calibration_failed"),
    ):
        state_before = runner.state
        _assert_prestart_state_is_zero(runner)
        assert state_before.instrument_resources_at_bind == ResourceSnapshot(
            0, 0.0, 0.0, 0.0, 0, 0, 0.0
        )
        if state_before.calibration_outcome is None:
            assert state_before.instrument_resources_current == (
                state_before.instrument_resources_at_bind
            )
        else:
            assert state_before.instrument_resources_current == (
                state_before.calibration_outcome.instrument_resources_after
            )
        original_instrument = runner._instrument
        object.__setattr__(runner, "_instrument", object())
        try:
            with pytest.raises(SparseRunnerStateError):
                build_sparse_linewidth_evaluator_resources(runner)
        finally:
            object.__setattr__(runner, "_instrument", original_instrument)
        assert runner.state is state_before
        assert runner.state.phase == phase


@pytest.mark.parametrize("treatment", ["included", "conditional"])
def test_started_resource_builder_replays_source_and_respects_treatment(
    treatment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, instrument, success = (
        _start_included(monkeypatch)
        if treatment == "included"
        else _start_conditional(monkeypatch)
    )
    state_before = runner.state

    resources = build_sparse_linewidth_evaluator_resources(runner)

    zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    expected_calibration = _replay(
        success.full_observations, success.source.source_frequency_overhead_s
    )
    assert resources is not None
    assert resources.calibration_observations == success.full_observations
    assert tuple(
        observation.estimator_view()
        for observation in resources.calibration_observations
    ) == success.safe_observations == success.source.source_observations
    assert resources.calibration_resources == expected_calibration
    assert resources.accepted_fast_observations == ()
    assert resources.accepted_sparse_observations == ()
    assert resources.accepted_tracking_observations == ()
    assert resources.unaccepted_tracking_observations == ()
    assert resources.fast_tracking_resources == zero
    assert resources.sparse_tracking_resources == zero
    assert resources.tracking_resources == zero
    assert resources.incomplete_fast_pair_observations == 0
    assert resources.incomplete_sparse_scan_observations == 0
    assert resources.unaccepted_observations == 0
    if treatment == "included":
        assert resources.calibration_budget_treatment == "included_same_run"
        assert resources.accepted_charged_resources == expected_calibration
        assert resources.charged_resources == expected_calibration
        assert runner.state.tracking_resources_before == (
            success.instrument_resources_after
        )
    else:
        assert (
            resources.calibration_budget_treatment
            == "conditional_free_precalibration"
        )
        assert resources.accepted_charged_resources == zero
        assert resources.charged_resources == zero
        assert runner.state.tracking_resources_before == zero
    assert runner.state.instrument_resources_current == instrument.resources
    assert runner.state.tracking_resources_before == instrument.resources
    assert runner.state is state_before


@pytest.mark.parametrize(
    "attack", ["safe_trace", "midpoints", "full_resources", "start_boundary"]
)
def test_started_resource_builder_rejects_broken_calibration_or_start_join(
    attack: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _, success = _start_included(monkeypatch)
    state = runner.state
    if attack == "safe_trace":
        target = success
        field = "safe_observations"
        original = success.safe_observations
        changed = tuple(reversed(original))
    elif attack == "midpoints":
        target = success
        field = "measurement_midpoints_s"
        original = success.measurement_midpoints_s
        changed = tuple(reversed(original))
    elif attack == "full_resources":
        target = success
        field = "full_resources"
        original = success.full_resources
        changed = copy.copy(original)
        object.__setattr__(changed, "expected_photons", original.expected_photons + 1.0)
    else:
        target = state
        field = "tracking_resources_before"
        original = state.tracking_resources_before
        assert original is not None
        changed = copy.copy(original)
        object.__setattr__(changed, "observations", original.observations + 1)
    object.__setattr__(target, field, changed)
    try:
        with pytest.raises(ValueError, match="resource context"):
            build_sparse_linewidth_evaluator_resources(runner)
    finally:
        object.__setattr__(target, field, original)


def test_started_resource_builder_reauthenticates_external_source_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    arguments = _calibration_arguments(
        source_clock_id="source-clock",
        tracker_clock_id="tracker-clock",
        source_to_tracker_offset_s=-0.012,
    )
    fit_configuration = arguments["fit_configuration"]
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: make_legal_source_fit(
            fit_configuration  # type: ignore[arg-type]
        ),
    )
    source_runner = SparseLinewidthEvaluatorRunner.bind(_instrument())
    success = source_runner.acquire_verified_calibration(  # type: ignore[arg-type]
        **arguments
    )
    assert type(success) is VerifiedTwoPointCalibrationSuccess
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="conditional_free_precalibration",
    )
    target = SparseLinewidthEvaluatorRunner.bind(_instrument())
    target.start_tracking(
        SparseLinewidthCompositeTracker(SparseLinewidthConfiguration()),
        calibration,
        success,
        TwoPointRunMetadata(
            tracker_clock_id="tracker-clock",
            current_sequence_index=None,
            current_timestamp_s=0.0,
            nominal_photon_rate_hz=target._instrument.nominal_photon_rate_hz,
            frequency_overhead_s=target._instrument.frequency_overhead_s,
            fluorescence_quantity="normalized_fluorescence",
        ),
        TwoPointBudgetCeiling(100, None, None, None),
        seed=7,
    )
    original_phase = source_runner.state.phase
    object.__setattr__(source_runner.state, "phase", "ready")
    try:
        with pytest.raises(ValueError, match="resource context"):
            build_sparse_linewidth_evaluator_resources(target)
    finally:
        object.__setattr__(source_runner.state, "phase", original_phase)


def test_resource_builder_requires_exact_runner_type() -> None:
    with pytest.raises(TypeError, match="exact SparseLinewidthEvaluatorRunner"):
        build_sparse_linewidth_evaluator_resources(object())  # type: ignore[arg-type]
