"""Resource-accounting tests at the sparse evaluator tracking boundary."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from odmr_bench.dynamics import SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import (
    GaussianNoise,
    InstrumentObservation,
    PoissonNoise,
    ResourceSnapshot,
)
from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.estimators import (
    CalibratedTwoPointTracker,
    SparseLinewidthCompositeTracker,
    SparseLinewidthConfiguration,
    TwoPointBudgetCeiling,
    TwoPointIdentityBinding,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    calibrate_two_point,
)
from odmr_bench.estimators.sparse_linewidth_types import (
    SparseLinewidthObservationValidationError,
)
from odmr_bench.evaluation.sparse_linewidth.resource_accounting import (
    build_sparse_linewidth_evaluator_resources,
)
from odmr_bench.evaluation.sparse_linewidth.runner import (
    SparseLinewidthEvaluatorRunner,
)
from odmr_bench.evaluation.sparse_linewidth.types import (
    SparseRunnerAborted,
    SparseRunnerStateError,
)
from odmr_bench.evaluation.two_point.runner import TwoPointEvaluatorRunner
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
    *,
    configuration: SparseLinewidthConfiguration | None = None,
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
        SparseLinewidthCompositeTracker(
            configuration or SparseLinewidthConfiguration()
        ),
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
    *,
    configuration: SparseLinewidthConfiguration | None = None,
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
    return _start_conditional_from_success(success, configuration=configuration)


def _start_conditional_from_success(
    success: VerifiedTwoPointCalibrationSuccess,
    *,
    configuration: SparseLinewidthConfiguration | None = None,
) -> tuple[
    SparseLinewidthEvaluatorRunner,
    ODMRInstrument,
    VerifiedTwoPointCalibrationSuccess,
]:
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="conditional_free_precalibration",
    )
    instrument = _instrument()
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)
    runner.start_tracking(
        SparseLinewidthCompositeTracker(
            configuration or SparseLinewidthConfiguration()
        ),
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


def _acquire_two_point_success(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    TwoPointEvaluatorRunner,
    ODMRInstrument,
    VerifiedTwoPointCalibrationSuccess,
]:
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
    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]
    assert type(outcome) is VerifiedTwoPointCalibrationSuccess
    return runner, instrument, outcome


def _replay(
    observations: tuple[InstrumentObservation, ...], overhead_s: float
) -> ResourceSnapshot:
    result = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    for observation in observations:
        result = ResourceSnapshot(
            result.observations + 1,
            result.integration_time_s + observation.integration_time_s,
            result.nominal_exposure_photons + observation.nominal_exposure_photons,
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
            + (overhead_s + observation.integration_time_s),
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
    assert (
        tuple(
            observation.estimator_view()
            for observation in resources.calibration_observations
        )
        == success.safe_observations
        == success.source.source_observations
    )
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
            resources.calibration_budget_treatment == "conditional_free_precalibration"
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


@pytest.mark.parametrize("source_kind", ["sparse", "two_point"])
def test_conditional_builder_accepts_real_source_tracking_progression_and_closes_phases(
    source_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if source_kind == "sparse":
        source_runner, source_instrument, success = _acquire_success(
            monkeypatch,
            source_clock_id="source-clock",
            tracker_clock_id="tracker-clock",
            source_to_tracker_offset_s=-0.012,
        )
    else:
        source_runner, source_instrument, success = _acquire_two_point_success(
            monkeypatch
        )
    target, _, _ = _start_conditional_from_success(success)
    assert build_sparse_linewidth_evaluator_resources(target) is not None

    source_calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="conditional_free_precalibration",
    )
    metadata = TwoPointRunMetadata(
        tracker_clock_id="tracker-clock",
        current_sequence_index=source_runner.state.instrument_current_sequence_index,
        current_timestamp_s=source_runner.state.current_virtual_time_s,
        nominal_photon_rate_hz=source_instrument.nominal_photon_rate_hz,
        frequency_overhead_s=source_instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    if source_kind == "sparse":
        source_runner.start_tracking(
            SparseLinewidthCompositeTracker(SparseLinewidthConfiguration()),
            source_calibration,
            success,
            metadata,
            TwoPointBudgetCeiling(100, None, None, None),
            seed=11,
        )
    else:
        source_runner.start_tracking(
            CalibratedTwoPointTracker(TwoPointTrackerConfiguration()),
            source_calibration,
            success,
            metadata,
            TwoPointBudgetCeiling(100, None, None, None),
            seed=11,
        )
    assert source_runner.state.phase == "tracking"
    assert build_sparse_linewidth_evaluator_resources(target) is not None

    original_phase = source_runner.state.phase
    object.__setattr__(source_runner.state, "phase", "hostile_unknown")
    try:
        with pytest.raises(ValueError, match="resource context"):
            build_sparse_linewidth_evaluator_resources(target)
    finally:
        object.__setattr__(source_runner.state, "phase", original_phase)

    if source_kind == "sparse":
        object.__setattr__(source_runner.state, "phase", "geometry_stopped")
        try:
            assert build_sparse_linewidth_evaluator_resources(target) is not None
        finally:
            object.__setattr__(source_runner.state, "phase", original_phase)
    else:
        object.__setattr__(source_runner.state, "phase", "geometry_stopped")
        try:
            with pytest.raises(ValueError, match="resource context"):
                build_sparse_linewidth_evaluator_resources(target)
        finally:
            object.__setattr__(source_runner.state, "phase", original_phase)


def test_conditional_builder_rejects_polluted_target_token_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point.provenance import _lookup_run_token_binding

    runner, _, success = _start_conditional(monkeypatch)
    binding = _lookup_run_token_binding(runner.state.run_token)
    assert binding is not None
    assert binding.success is None
    assert binding.source is None
    object.__setattr__(binding, "success", success)
    object.__setattr__(binding, "source", success.source)
    try:
        with pytest.raises(ValueError, match="resource context"):
            build_sparse_linewidth_evaluator_resources(runner)
    finally:
        object.__setattr__(binding, "success", None)
        object.__setattr__(binding, "source", None)


def test_started_builder_rejects_coordinated_nonzero_cpu_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _start_included(monkeypatch)
    state = runner.state
    estimate = state.tracker_estimate
    assert estimate is not None
    field_names = (
        "fast_update_cpu_time_s",
        "sparse_update_cpu_time_s",
        "total_update_cpu_time_s",
    )
    for target in (estimate, state):
        for field_name in field_names:
            object.__setattr__(target, field_name, 1.0)
    try:
        with pytest.raises(ValueError, match="resource context"):
            build_sparse_linewidth_evaluator_resources(runner)
    finally:
        for target in (estimate, state):
            for field_name in field_names:
                object.__setattr__(target, field_name, 0.0)


def test_resource_builder_requires_exact_runner_type() -> None:
    with pytest.raises(TypeError, match="exact SparseLinewidthEvaluatorRunner"):
        build_sparse_linewidth_evaluator_resources(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("treatment", ["included", "conditional"])
def test_tracking_resources_replay_fast_sparse_and_interleaved_arrival_atoms(
    treatment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = SparseLinewidthConfiguration(scan_period_fast_pairs=1)
    runner, _, success = (
        _start_included(monkeypatch, configuration=configuration)
        if treatment == "included"
        else _start_conditional(monkeypatch, configuration=configuration)
    )

    outcomes = [runner.step() for _ in range(5)]
    assert all(outcome.kind == "accepted" for outcome in outcomes)
    resources = build_sparse_linewidth_evaluator_resources(runner)
    assert resources is not None
    trace = runner.state.normal_tracking_trace
    assert tuple(acquisition.mode for acquisition in trace) == (
        "fast_pair",
        "fast_pair",
        "sparse_scan",
        "sparse_scan",
        "sparse_scan",
    )
    assert tuple(
        acquisition.full_observation.sequence_index for acquisition in trace
    ) == tuple(
        range(
            trace[0].full_observation.sequence_index,
            trace[0].full_observation.sequence_index + 5,
        )
    )
    assert len(trace) == 5
    fast = tuple(
        acquisition.full_observation
        for acquisition in trace
        if acquisition.mode == "fast_pair"
    )
    sparse = tuple(
        acquisition.full_observation
        for acquisition in trace
        if acquisition.mode == "sparse_scan"
    )
    interleaved = tuple(acquisition.full_observation for acquisition in trace)
    overhead_s = runner.state.instrument_configuration.frequency_overhead_s
    assert resources.accepted_fast_observations == fast
    assert resources.accepted_sparse_observations == sparse
    assert resources.accepted_tracking_observations == interleaved
    assert resources.fast_tracking_resources == _replay(fast, overhead_s)
    assert resources.sparse_tracking_resources == _replay(sparse, overhead_s)
    assert resources.tracking_resources == _replay(interleaved, overhead_s)
    assert resources.incomplete_fast_pair_observations == 0
    assert resources.incomplete_sparse_scan_observations == 3
    assert resources.unaccepted_observations == 0
    replayed_interleaved = _replay(interleaved, overhead_s)
    assert (
        resources.tracking_resources.expected_photons
        == replayed_interleaved.expected_photons
    )
    assert (
        resources.tracking_resources.realized_photons
        == replayed_interleaved.realized_photons
    )
    calibration_resources = _replay(
        success.full_observations, success.source.source_frequency_overhead_s
    )
    charged_start = (
        calibration_resources
        if treatment == "included"
        else ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    )
    charged = charged_start
    for observation in interleaved:
        atom = _replay((observation,), overhead_s)
        charged = ResourceSnapshot(
            charged.observations + atom.observations,
            charged.integration_time_s + atom.integration_time_s,
            charged.nominal_exposure_photons + atom.nominal_exposure_photons,
            charged.expected_photons + atom.expected_photons,
            charged.realized_photons + atom.realized_photons,
            charged.observations_without_realized_counts
            + atom.observations_without_realized_counts,
            charged.virtual_elapsed_time_s + atom.virtual_elapsed_time_s,
        )
    assert resources.accepted_charged_resources == charged
    assert resources.charged_resources == charged


def test_tracking_resources_count_one_incomplete_fast_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _start_included(monkeypatch)
    outcome = runner.step()
    assert outcome.kind == "accepted"

    resources = build_sparse_linewidth_evaluator_resources(runner)

    assert resources is not None
    assert len(resources.accepted_fast_observations) == 1
    assert resources.accepted_sparse_observations == ()
    assert resources.incomplete_fast_pair_observations == 1
    assert resources.incomplete_sparse_scan_observations == 0


@pytest.mark.parametrize("partial_count", [1, 2, 3, 4])
def test_tracking_resources_retain_each_partial_sparse_prefix(
    partial_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _start_conditional(
        monkeypatch,
        configuration=SparseLinewidthConfiguration(scan_period_fast_pairs=1),
    )
    for _ in range(2 + partial_count):
        assert runner.step().kind == "accepted"

    resources = build_sparse_linewidth_evaluator_resources(runner)

    assert resources is not None
    assert len(resources.accepted_fast_observations) == 2
    assert len(resources.accepted_sparse_observations) == partial_count
    assert resources.incomplete_fast_pair_observations == 0
    assert resources.incomplete_sparse_scan_observations == partial_count


def test_accepted_atom_retains_expected_and_realized_photon_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, _ = _start_conditional(monkeypatch)
    object.__setattr__(instrument, "_noise", PoissonNoise())

    outcome = runner.step()
    resources = build_sparse_linewidth_evaluator_resources(runner)

    assert outcome.kind == "accepted"
    full = outcome.acquisition.full_observation
    assert full.realized_photons is not None
    assert resources is not None
    assert resources.tracking_resources.expected_photons == full.expected_photons
    assert resources.tracking_resources.realized_photons == full.realized_photons
    assert resources.tracking_resources.observations_without_realized_counts == 0


def test_scientifically_failed_scan_is_fully_accepted_and_charged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.estimators import sparse_linewidth_tracker as tracker_module

    original_fit = tracker_module.fit_sparse_linewidth

    def scientific_failure(*args: object, **kwargs: object):
        result = original_fit(*args, **kwargs)  # type: ignore[arg-type]
        return replace(
            result,
            status="failure",
            failure_code="model_evaluation_failed",
            fitted_center_correction_hz=None,
            fitted_local_center_hz=None,
            fitted_fwhm_hz=None,
            fitted_amplitude=None,
            fitted_baseline_offset=None,
            fitted_q=None,
            rmse=None,
            amplitude_normalized_rmse=None,
            scaled_jacobian_rank=None,
            scaled_jacobian_condition=None,
            scipy_status=None,
            scipy_message=None,
            nfev=None,
        )

    monkeypatch.setattr(tracker_module, "fit_sparse_linewidth", scientific_failure)
    runner, _, _ = _start_included(
        monkeypatch,
        configuration=SparseLinewidthConfiguration(scan_period_fast_pairs=1),
    )

    outcomes = [runner.step() for _ in range(7)]

    assert outcomes[-1].update.completed_sparse_scan.status == "failure"
    resources = build_sparse_linewidth_evaluator_resources(runner)
    assert resources is not None
    assert len(resources.accepted_fast_observations) == 2
    assert len(resources.accepted_sparse_observations) == 5
    assert len(resources.accepted_tracking_observations) == 7
    assert resources.incomplete_fast_pair_observations == 0
    assert resources.incomplete_sparse_scan_observations == 0


def test_authenticated_abort_resource_view_separates_accepted_and_final_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _start_included(monkeypatch)

    def reject_returned_observation(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise SparseLinewidthObservationValidationError(
            "endpoint_mismatch", "injected returned-observation rejection"
        )

    monkeypatch.setattr(
        SparseLinewidthCompositeTracker,
        "update",
        reject_returned_observation,
    )

    outcome = runner.step()
    rebuilt = build_sparse_linewidth_evaluator_resources(runner)

    assert type(outcome) is SparseRunnerAborted
    assert rebuilt == outcome.resources
    assert rebuilt is not None
    assert rebuilt.accepted_tracking_observations == ()
    assert len(rebuilt.unaccepted_tracking_observations) == 1
    assert rebuilt.unaccepted_observations == 1
    assert rebuilt.accepted_charged_resources.observations + 1 == (
        rebuilt.charged_resources.observations
    )
    assert rebuilt.fast_tracking_resources.observations == 0
    assert rebuilt.sparse_tracking_resources.observations == 0
    assert rebuilt.tracking_resources.observations == 0


@pytest.mark.parametrize("join_kind", ["authenticated", "unavailable"])
def test_abort_resource_builder_authenticates_unaccepted_midpoint_recurrence(
    join_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _start_included(monkeypatch)
    if join_kind == "authenticated":
        monkeypatch.setattr(
            SparseLinewidthCompositeTracker,
            "update",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                SparseLinewidthObservationValidationError(
                    "endpoint_mismatch", "injected rejection"
                )
            ),
        )
    else:
        original_query = ODMRInstrument.query

        def query_then_corrupt(
            self: ODMRInstrument, frequency_hz: float, integration_time_s: float
        ):
            observation = original_query(self, frequency_hz, integration_time_s)
            object.__setattr__(
                self._ledger, "_observations", self._ledger._observations + 1
            )
            return observation

        monkeypatch.setattr(ODMRInstrument, "query", query_then_corrupt)

    outcome = runner.step()
    assert type(outcome) is SparseRunnerAborted
    acquisition = outcome.abort.unaccepted_acquisition
    original_midpoint = acquisition.expected_measurement_midpoint_s
    object.__setattr__(
        acquisition,
        "expected_measurement_midpoint_s",
        original_midpoint + 1.0,
    )
    try:
        with pytest.raises(ValueError, match="resource context"):
            build_sparse_linewidth_evaluator_resources(runner)
    finally:
        object.__setattr__(
            acquisition,
            "expected_measurement_midpoint_s",
            original_midpoint,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "reason",
        "exception_type",
        "exception_message",
        "unaccepted_count",
        "estimate_before",
        "estimate_after",
        "phase",
        "pending_query",
    ],
)
def test_authenticated_abort_resource_builder_reauthenticates_terminal_record(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _start_included(monkeypatch)

    monkeypatch.setattr(
        SparseLinewidthCompositeTracker,
        "update",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SparseLinewidthObservationValidationError(
                "endpoint_mismatch", "injected rejection"
            )
        ),
    )
    outcome = runner.step()
    assert type(outcome) is SparseRunnerAborted
    abort = outcome.abort
    acquisition = abort.unaccepted_acquisition

    target: object
    attribute: str
    replacement: object
    if mutation == "reason":
        target, attribute, replacement = (
            abort,
            "reason",
            "tracker_update_construction_error",
        )
    elif mutation == "exception_type":
        target, attribute, replacement = abort, "exception_type", None
    elif mutation == "exception_message":
        target, attribute, replacement = abort, "exception_message", None
    elif mutation == "unaccepted_count":
        target, attribute, replacement = abort, "unaccepted_observation_count", 0
    elif mutation == "estimate_before":
        target, attribute, replacement = abort, "tracker_estimate_before", object()
    elif mutation == "estimate_after":
        target, attribute, replacement = abort, "tracker_estimate_after", object()
    elif mutation == "phase":
        target, attribute, replacement = outcome.state, "phase", "tracking"
    else:
        target, attribute = acquisition, "query"
        replacement = replace(
            acquisition.query,
            frequency_hz=acquisition.query.frequency_hz + 1.0,
        )
    original = getattr(target, attribute)
    object.__setattr__(target, attribute, replacement)
    try:
        with pytest.raises(ValueError, match="resource context"):
            build_sparse_linewidth_evaluator_resources(runner)
    finally:
        object.__setattr__(target, attribute, original)
