"""Tests for the evaluator-owned two-point tracking runner."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from odmr_bench.dynamics import SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import GaussianNoise
from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.observations import (
    EstimatorObservation,
    InstrumentObservation,
)
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators import (
    CalibratedTwoPointTracker,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointEstimate,
    TwoPointIdentityBinding,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    TwoPointUpdate,
    calibrate_two_point,
)
from odmr_bench.evaluation.two_point import (
    TwoPointEvaluatorRunner,
    TwoPointRunnerAccepted,
    TwoPointRunnerInstrumentFailure,
    TwoPointRunnerStartError,
    VerifiedTwoPointCalibrationSuccess,
)
from odmr_bench.evaluation.two_point.provenance import _lookup_run_token_binding
from odmr_bench.models import Baseline, Resonance
from tests.two_point_helpers import (
    make_legal_fit_configuration,
    make_legal_source_fit,
)


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


def _instrument(
    *,
    nominal_photon_rate_hz: float = 2.5e6,
    frequency_overhead_s: float = 0.001,
) -> ODMRInstrument:
    return ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=GaussianNoise(stddev_at_1s=0.0),
        nominal_photon_rate_hz=nominal_photon_rate_hz,
        frequency_overhead_s=frequency_overhead_s,
        seed=13,
    )


def _verified_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_clock_id: str,
    tracker_clock_id: str,
    source_to_tracker_offset_s: float,
) -> tuple[
    TwoPointEvaluatorRunner,
    ODMRInstrument,
    VerifiedTwoPointCalibrationSuccess,
]:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    fit_configuration = make_legal_fit_configuration()
    returned_fit = make_legal_source_fit(fit_configuration)
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: returned_fit,
    )
    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    outcome = runner.acquire_verified_calibration(
        (2.74e9, 3.02e9),
        0.005,
        fit_configuration,
        TwoPointIdentityBinding(
            "require_expected_ids", fit_configuration.resonance_ids
        ),
        source_id="verified-source",
        source_clock_id=source_clock_id,
        tracker_clock_id=tracker_clock_id,
        source_to_tracker_offset_s=source_to_tracker_offset_s,
        physical_fit_epoch_rule="instrument_midpoint_ordered_mean",
    )
    assert type(outcome) is VerifiedTwoPointCalibrationSuccess
    return runner, instrument, outcome


def _included_inputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_clock_id: str = "shared-clock",
    tracker_clock_id: str = "shared-clock",
    source_to_tracker_offset_s: float = 0.0,
) -> tuple[
    TwoPointEvaluatorRunner,
    ODMRInstrument,
    CalibratedTwoPointTracker,
    TwoPointCalibration,
    VerifiedTwoPointCalibrationSuccess,
    TwoPointRunMetadata,
    TwoPointBudgetCeiling,
]:
    runner, instrument, success = _verified_success(
        monkeypatch,
        source_clock_id=source_clock_id,
        tracker_clock_id=tracker_clock_id,
        source_to_tracker_offset_s=source_to_tracker_offset_s,
    )
    configuration = TwoPointTrackerConfiguration()
    tracker = CalibratedTwoPointTracker(configuration)
    calibration = calibrate_two_point(
        success.source,
        configuration,
        budget_treatment="included_same_run",
    )
    metadata = TwoPointRunMetadata(
        tracker_clock_id=tracker_clock_id,
        current_sequence_index=runner.state.instrument_current_sequence_index,
        current_timestamp_s=runner.state.current_virtual_time_s,
        nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
        frequency_overhead_s=instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    budget = TwoPointBudgetCeiling(4, 0.02, 50_000.0, 0.024)
    return runner, instrument, tracker, calibration, success, metadata, budget


def _conditional_tracking_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    TwoPointEvaluatorRunner,
    ODMRInstrument,
    CalibratedTwoPointTracker,
]:
    _, _, success = _verified_success(
        monkeypatch,
        source_clock_id="source-clock",
        tracker_clock_id="tracking-clock",
        source_to_tracker_offset_s=-0.012,
    )
    configuration = TwoPointTrackerConfiguration()
    calibration = calibrate_two_point(
        success.source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    tracker = CalibratedTwoPointTracker(configuration)
    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    metadata = TwoPointRunMetadata(
        tracker_clock_id="tracking-clock",
        current_sequence_index=None,
        current_timestamp_s=0.0,
        nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
        frequency_overhead_s=instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    runner.start_tracking(
        tracker,
        calibration,
        success,
        metadata,
        TwoPointBudgetCeiling(8, 0.04, 100_000.0, 0.048),
        seed=20260904,
    )
    return runner, instrument, tracker


def test_conditional_start_on_clean_other_runner_preserves_identity_and_zero_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_runner, source_instrument, success = _verified_success(
        monkeypatch,
        source_clock_id="source-clock",
        tracker_clock_id="tracking-clock",
        source_to_tracker_offset_s=-0.012,
    )
    configuration = TwoPointTrackerConfiguration()
    calibration = calibrate_two_point(
        success.source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    tracker = CalibratedTwoPointTracker(configuration)
    tracking_instrument = _instrument(
        nominal_photon_rate_hz=4.0e6,
        frequency_overhead_s=0.002,
    )
    runner = TwoPointEvaluatorRunner.bind(tracking_instrument)
    state_before = runner.state
    metadata = TwoPointRunMetadata(
        tracker_clock_id="tracking-clock",
        current_sequence_index=None,
        current_timestamp_s=0.0,
        nominal_photon_rate_hz=4.0e6,
        frequency_overhead_s=0.002,
        fluorescence_quantity="normalized_fluorescence",
    )
    budget = TwoPointBudgetCeiling(2, 0.01, 40_000.0, 0.014)
    reset_calls: list[tuple[object, ...]] = []
    original_reset = CalibratedTwoPointTracker.reset

    def reset_spy(
        self: CalibratedTwoPointTracker,
        public_metadata: TwoPointRunMetadata,
        supplied_calibration: object,
        budget_ceiling: object,
        *,
        seed: int,
    ) -> None:
        reset_calls.append(
            (
                self,
                public_metadata,
                supplied_calibration,
                budget_ceiling,
                seed,
            )
        )
        original_reset(
            self,
            public_metadata,
            supplied_calibration,  # type: ignore[arg-type]
            budget_ceiling,  # type: ignore[arg-type]
            seed=seed,
        )

    query_calls = 0

    def reject_query(*args: object, **kwargs: object) -> object:
        nonlocal query_calls
        query_calls += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(CalibratedTwoPointTracker, "reset", reset_spy)
    monkeypatch.setattr(ODMRInstrument, "query", reject_query)

    state = runner.start_tracking(
        tracker,
        calibration,
        success,
        metadata,
        budget,
        seed=20260904,
    )

    zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    assert state is runner.state
    assert state.phase == "tracking"
    assert state.run_token is state_before.run_token
    assert state.run_token is not success.run_token
    assert state.calibration_outcome is None
    assert state.verified_calibration is success
    assert state.verified_calibration.source is success.source
    assert state.verified_calibration.run_token is success.run_token
    assert state.calibration is calibration
    assert tracker.calibration is calibration
    assert state.tracker_estimate is tracker.estimate()
    assert state.tracker_estimate.accepted_observations == 0
    assert state.tracker_estimate.completed_pairs == 0
    assert state.tracker_estimate.pending_query is None
    assert state.tracker_estimate.pair_history == ()
    assert state.tracker_estimate.calibration_resources == success.safe_resources
    assert state.tracker_estimate.charged_resources.observations == 0
    assert state.normal_tracking_trace == ()
    assert state.pair_timings == ()
    assert state.tracking_resources_before == zero
    assert state.instrument_resources_current == zero
    assert state.instrument_current_sequence_index is None
    assert state.current_virtual_time_s == 0.0
    assert state.last_instrument_failure is None
    assert state.terminal_abort is None
    assert reset_calls == [(tracker, metadata, calibration, budget, 20260904)]
    assert query_calls == 0
    assert tracking_instrument.resources == zero
    assert tracking_instrument.virtual_time_s == 0.0
    assert source_runner.state.calibration_outcome is success
    assert source_runner.state.verified_calibration is success
    assert source_instrument.resources == success.instrument_resources_after


def test_conditional_start_on_same_calibration_succeeded_runner_is_legal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, success = _verified_success(
        monkeypatch,
        source_clock_id="shared-clock",
        tracker_clock_id="shared-clock",
        source_to_tracker_offset_s=0.0,
    )
    configuration = TwoPointTrackerConfiguration()
    calibration = calibrate_two_point(
        success.source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    tracker = CalibratedTwoPointTracker(configuration)
    state_before = runner.state
    metadata = TwoPointRunMetadata(
        tracker_clock_id="shared-clock",
        current_sequence_index=state_before.instrument_current_sequence_index,
        current_timestamp_s=state_before.current_virtual_time_s,
        nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
        frequency_overhead_s=instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    budget = TwoPointBudgetCeiling(2, 0.01, 25_000.0, 0.012)
    query_calls = 0

    def reject_query(*args: object, **kwargs: object) -> object:
        nonlocal query_calls
        query_calls += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(ODMRInstrument, "query", reject_query)

    state = runner.start_tracking(
        tracker,
        calibration,
        success,
        metadata,
        budget,
        seed=19,
    )

    assert state is runner.state
    assert state.phase == "tracking"
    assert state.calibration_outcome is success
    assert state.verified_calibration is success
    assert state.calibration_outcome is state.verified_calibration
    assert state.calibration is calibration
    assert state.tracker_estimate is tracker.estimate()
    assert state.tracker_estimate.calibration_resources == success.safe_resources
    assert state.tracker_estimate.charged_resources.observations == 0
    assert state.tracker_estimate.charged_resources.integration_time_s == 0.0
    assert state.tracker_estimate.charged_resources.nominal_exposure_photons == 0.0
    assert state.tracker_estimate.charged_resources.virtual_elapsed_time_s == 0.0
    assert state.normal_tracking_trace == ()
    assert state.pair_timings == ()
    assert state.tracking_resources_before == success.instrument_resources_after
    assert state.instrument_resources_current == success.instrument_resources_after
    assert state.instrument_current_sequence_index == 1
    assert state.current_virtual_time_s == 0.012
    assert query_calls == 0
    assert instrument.resources == success.instrument_resources_after


def test_included_start_requires_exact_original_capabilities_and_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _included_inputs(monkeypatch)

    copied_outcome = _included_inputs(monkeypatch)
    copied_outcome = (
        *copied_outcome[:4],
        replace(copied_outcome[4]),
        *copied_outcome[5:],
    )

    copied_source = _included_inputs(monkeypatch)
    copied_source_value = copy.copy(copied_source[4].source)
    copied_source_calibration = copy.copy(copied_source[3])
    object.__setattr__(
        copied_source_calibration, "source", copied_source_value
    )
    copied_source = (
        *copied_source[:3],
        copied_source_calibration,
        copied_source[4],
        *copied_source[5:],
    )

    copied_source_outcome = _included_inputs(monkeypatch)
    copied_source_value = copy.copy(copied_source_outcome[4].source)
    copied_success = replace(
        copied_source_outcome[4], source=copied_source_value
    )
    copied_calibration = copy.copy(copied_source_outcome[3])
    object.__setattr__(copied_calibration, "source", copied_source_value)
    copied_source_outcome = (
        *copied_source_outcome[:3],
        copied_calibration,
        copied_success,
        *copied_source_outcome[5:],
    )

    unregistered_token = _included_inputs(monkeypatch)
    token = object.__new__(type(unregistered_token[4].run_token))
    unregistered_token = (
        *unregistered_token[:4],
        replace(unregistered_token[4], run_token=token),
        *unregistered_token[5:],
    )

    copied_runner = _included_inputs(monkeypatch)
    copied_runner = (copy.copy(copied_runner[0]), *copied_runner[1:])

    copied_instrument = _included_inputs(monkeypatch)
    object.__setattr__(
        copied_instrument[0], "_instrument", copy.copy(copied_instrument[1])
    )

    nonshared_clock = _included_inputs(
        monkeypatch,
        source_clock_id="source-clock",
        tracker_clock_id="tracking-clock",
        source_to_tracker_offset_s=0.0,
    )

    discontinuous_boundary = _included_inputs(monkeypatch)
    zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    object.__setattr__(
        discontinuous_boundary[0].state,
        "instrument_resources_current",
        zero,
    )

    metadata_rate = _included_inputs(monkeypatch)
    metadata_rate = (
        *metadata_rate[:5],
        replace(metadata_rate[5], nominal_photon_rate_hz=3.0e6),
        metadata_rate[6],
    )
    metadata_overhead = _included_inputs(monkeypatch)
    metadata_overhead = (
        *metadata_overhead[:5],
        replace(metadata_overhead[5], frequency_overhead_s=0.002),
        metadata_overhead[6],
    )

    source_rate = _included_inputs(monkeypatch)
    object.__setattr__(
        source_rate[4].source.fluorescence_provenance,
        "nominal_photon_rate_hz",
        3.0e6,
    )
    source_overhead = _included_inputs(monkeypatch)
    object.__setattr__(
        source_overhead[4].source,
        "source_frequency_overhead_s",
        0.002,
    )

    reset_calls: list[CalibratedTwoPointTracker] = []
    query_calls = 0
    original_reset = CalibratedTwoPointTracker.reset

    def reset_spy(
        self: CalibratedTwoPointTracker,
        public_metadata: TwoPointRunMetadata,
        calibration: TwoPointCalibration,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> None:
        reset_calls.append(self)
        original_reset(
            self,
            public_metadata,
            calibration,
            budget_ceiling,
            seed=seed,
        )

    def reject_query(*args: object, **kwargs: object) -> object:
        nonlocal query_calls
        query_calls += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(CalibratedTwoPointTracker, "reset", reset_spy)
    monkeypatch.setattr(ODMRInstrument, "query", reject_query)

    runner, instrument, tracker, calibration, success, metadata, budget = valid
    state = runner.start_tracking(
        tracker,
        calibration,
        success,
        metadata,
        budget,
        seed=23,
    )

    binding = _lookup_run_token_binding(success.run_token)
    assert binding is not None
    assert binding.issuer_runner is runner
    assert binding.instrument is instrument
    assert binding.success is success
    assert binding.source is success.source
    assert state.run_token is success.run_token
    assert state.calibration_outcome is success
    assert state.verified_calibration is success
    assert state.calibration is calibration
    assert state.tracking_resources_before == success.instrument_resources_after
    assert state.tracking_resources_before == instrument.resources
    assert success.source.clock_mapping.kind == "shared_clock"
    assert metadata.current_sequence_index == success.source.availability_sequence_index
    assert metadata.current_timestamp_s == success.source.availability_timestamp_s
    assert (
        metadata.nominal_photon_rate_hz
        == state.instrument_configuration.nominal_photon_rate_hz
        == success.source.fluorescence_provenance.nominal_photon_rate_hz
    )
    assert (
        metadata.frequency_overhead_s
        == state.instrument_configuration.frequency_overhead_s
        == success.source.source_frequency_overhead_s
    )
    assert state.tracker_estimate is tracker.estimate()
    assert state.tracker_estimate.charged_resources == success.safe_resources
    assert state.normal_tracking_trace == ()
    assert state.pair_timings == ()

    invalid_cases = (
        ("copied outcome", copied_outcome, "unverified_calibration"),
        ("copied calibration source", copied_source, "calibration_mismatch"),
        ("copied outcome source", copied_source_outcome, "unverified_calibration"),
        ("unregistered token", unregistered_token, "unverified_calibration"),
        ("copied runner", copied_runner, "run_provenance_mismatch"),
        ("copied instrument", copied_instrument, "run_provenance_mismatch"),
        ("nonshared clock", nonshared_clock, "metadata_mismatch"),
        (
            "discontinuous boundary",
            discontinuous_boundary,
            "resource_boundary_mismatch",
        ),
        ("metadata rate", metadata_rate, "metadata_mismatch"),
        ("metadata overhead", metadata_overhead, "metadata_mismatch"),
        ("source rate", source_rate, "metadata_mismatch"),
        ("source overhead", source_overhead, "metadata_mismatch"),
    )
    for label, arguments, expected_code in invalid_cases:
        (
            invalid_runner,
            invalid_instrument,
            invalid_tracker,
            invalid_calibration,
            invalid_success,
            invalid_metadata,
            invalid_budget,
        ) = arguments
        state_before = invalid_runner.state
        tracker_calibration_before = invalid_tracker.calibration
        resources_before = invalid_instrument.resources
        time_before = invalid_instrument.virtual_time_s
        with pytest.raises(TwoPointRunnerStartError) as raised:
            invalid_runner.start_tracking(
                invalid_tracker,
                invalid_calibration,
                invalid_success,
                invalid_metadata,
                invalid_budget,
                seed=23,
            )
        assert raised.value.code == expected_code, label
        assert invalid_runner.state is state_before, label
        assert invalid_tracker.calibration is tracker_calibration_before, label
        assert invalid_instrument.resources == resources_before, label
        assert invalid_instrument.virtual_time_s == time_before, label

    assert reset_calls == [tracker]
    assert query_calls == 0


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("invalid_runner_phase", "invalid_runner_phase"),
        ("invalid_argument_type", "invalid_argument_type"),
        ("unverified_calibration", "unverified_calibration"),
        ("calibration_mismatch", "calibration_mismatch"),
        ("run_provenance_mismatch", "run_provenance_mismatch"),
        ("metadata_mismatch", "metadata_mismatch"),
        ("resource_boundary_mismatch", "resource_boundary_mismatch"),
        ("tracker_reset_failed", "tracker_reset_failed"),
    ],
)
def test_start_error_precedence_and_atomicity(
    case: str,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        runner,
        instrument,
        tracker,
        calibration,
        success,
        metadata,
        budget,
    ) = _included_inputs(monkeypatch)
    original_reset = CalibratedTwoPointTracker.reset

    if case == "invalid_runner_phase":
        runner.start_tracking(
            tracker,
            calibration,
            success,
            metadata,
            budget,
            seed=5,
        )
    else:
        original_reset(
            tracker,
            metadata,
            calibration,
            budget,
            seed=7,
        )

    call_runner = runner
    call_tracker = tracker
    call_success = success
    call_metadata = metadata
    call_seed: object = 29
    if case == "invalid_runner_phase":
        call_seed = True
    elif case == "invalid_argument_type":
        call_seed = True
        call_success = replace(success)
    elif case == "unverified_calibration":
        call_success = replace(success)
        call_tracker = CalibratedTwoPointTracker(
            replace(tracker.configuration, proportional_gain=0.5)
        )
        call_metadata = replace(metadata, nominal_photon_rate_hz=3.0e6)
    elif case == "calibration_mismatch":
        call_tracker = CalibratedTwoPointTracker(
            replace(tracker.configuration, proportional_gain=0.5)
        )
        call_metadata = replace(metadata, nominal_photon_rate_hz=3.0e6)
    elif case == "run_provenance_mismatch":
        call_runner = copy.copy(runner)
        call_metadata = replace(metadata, nominal_photon_rate_hz=3.0e6)
    elif case == "metadata_mismatch":
        call_metadata = replace(metadata, nominal_photon_rate_hz=3.0e6)
        instrument.query(2.8e9, 0.005)
    elif case == "resource_boundary_mismatch":
        instrument.query(2.8e9, 0.005)

    runner_state_before = call_runner.state
    tracker_configuration_before = call_tracker._configuration
    tracker_state_before = call_tracker._state
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s
    runner_binding_before = _lookup_run_token_binding(
        call_runner.state.run_token
    )
    success_binding_before = _lookup_run_token_binding(success.run_token)
    reset_calls: list[CalibratedTwoPointTracker] = []
    query_calls = 0
    reset_failure = RuntimeError("reset committed then failed")

    def reset_sentinel(
        self: CalibratedTwoPointTracker,
        public_metadata: TwoPointRunMetadata,
        supplied_calibration: TwoPointCalibration,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> None:
        reset_calls.append(self)
        if case != "tracker_reset_failed":
            raise AssertionError("start preflight reached tracker.reset")
        original_reset(
            self,
            public_metadata,
            supplied_calibration,
            budget_ceiling,
            seed=seed,
        )
        object.__setattr__(
            self,
            "_configuration",
            replace(self._configuration, proportional_gain=0.5),
        )
        raise reset_failure

    def reject_query(*args: object, **kwargs: object) -> object:
        nonlocal query_calls
        query_calls += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(CalibratedTwoPointTracker, "reset", reset_sentinel)
    monkeypatch.setattr(ODMRInstrument, "query", reject_query)

    with pytest.raises(TwoPointRunnerStartError) as raised:
        call_runner.start_tracking(
            call_tracker,
            calibration,
            call_success,
            call_metadata,
            budget,
            seed=call_seed,  # type: ignore[arg-type]
        )

    assert raised.value.code == expected_code
    assert call_runner.state is runner_state_before
    assert call_tracker._configuration is tracker_configuration_before
    assert call_tracker._state is tracker_state_before
    assert instrument.resources == resources_before
    assert instrument.virtual_time_s == time_before
    assert query_calls == 0
    assert len(reset_calls) == (1 if case == "tracker_reset_failed" else 0)
    assert _lookup_run_token_binding(call_runner.state.run_token) is (
        runner_binding_before
    )
    assert _lookup_run_token_binding(success.run_token) is success_binding_before
    if case == "tracker_reset_failed":
        assert raised.value.__cause__ is reset_failure


def test_start_base_exception_restores_exact_tracker_slots_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        runner,
        instrument,
        tracker,
        calibration,
        success,
        metadata,
        budget,
    ) = _included_inputs(monkeypatch)
    original_reset = CalibratedTwoPointTracker.reset
    original_reset(
        tracker,
        metadata,
        calibration,
        budget,
        seed=7,
    )
    runner_state_before = runner.state
    tracker_configuration_before = tracker._configuration
    tracker_state_before = tracker._state
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s
    query_calls = 0
    failure = KeyboardInterrupt("reset committed then interrupted")

    def reset_then_interrupt(
        self: CalibratedTwoPointTracker,
        public_metadata: TwoPointRunMetadata,
        supplied_calibration: TwoPointCalibration,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> None:
        original_reset(
            self,
            public_metadata,
            supplied_calibration,
            budget_ceiling,
            seed=seed,
        )
        object.__setattr__(
            self,
            "_configuration",
            replace(self._configuration, proportional_gain=0.5),
        )
        raise failure

    def reject_query(*args: object, **kwargs: object) -> object:
        nonlocal query_calls
        query_calls += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(
        CalibratedTwoPointTracker,
        "reset",
        reset_then_interrupt,
    )
    monkeypatch.setattr(ODMRInstrument, "query", reject_query)

    with pytest.raises(KeyboardInterrupt) as raised:
        runner.start_tracking(
            tracker,
            calibration,
            success,
            metadata,
            budget,
            seed=29,
        )

    assert raised.value is failure
    assert runner.state is runner_state_before
    assert runner._tracker is None
    assert tracker._configuration is tracker_configuration_before
    assert tracker._state is tracker_state_before
    assert instrument.resources == resources_before
    assert instrument.virtual_time_s == time_before
    assert query_calls == 0


def test_step_accepts_first_and_second_sides_and_records_pair_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, _ = _conditional_tracking_inputs(monkeypatch)
    original_query = ODMRInstrument.query
    original_update = CalibratedTwoPointTracker.update
    query_entries: list[tuple[ResourceSnapshot, float, float, float]] = []
    update_entries: list[tuple[TwoPointEstimate, EstimatorObservation]] = []

    def query_spy(
        self: ODMRInstrument,
        frequency_hz: float,
        integration_time_s: float,
    ) -> InstrumentObservation:
        query_entries.append(
            (
                self.resources,
                self.virtual_time_s,
                frequency_hz,
                integration_time_s,
            )
        )
        return original_query(self, frequency_hz, integration_time_s)

    def update_spy(
        self: CalibratedTwoPointTracker,
        observation: EstimatorObservation,
    ) -> TwoPointUpdate:
        update_entries.append((self.estimate(), observation))
        return original_update(self, observation)

    monkeypatch.setattr(ODMRInstrument, "query", query_spy)
    monkeypatch.setattr(CalibratedTwoPointTracker, "update", update_spy)

    first = runner.step()

    assert type(first) is TwoPointRunnerAccepted
    first_query = first.acquisition.query
    first_observation = first.acquisition.full_observation
    first_safe = first.acquisition.safe_observation
    first_delta = ResourceSnapshot(
        observations=1,
        integration_time_s=first_observation.integration_time_s,
        nominal_exposure_photons=first_observation.nominal_exposure_photons,
        expected_photons=first_observation.expected_photons,
        realized_photons=0,
        observations_without_realized_counts=1,
        virtual_elapsed_time_s=(
            instrument.frequency_overhead_s
            + first_observation.integration_time_s
        ),
    )
    assert first.kind == "accepted"
    assert first_query.side == "minus"
    assert first.acquisition.resource_join_status == "authenticated"
    assert first.acquisition.expected_measurement_midpoint_s == 0.0035
    assert first.acquisition.measurement_midpoint_s == 0.0035
    assert first.acquisition.instrument_resources_before == ResourceSnapshot(
        0, 0.0, 0.0, 0.0, 0, 0, 0.0
    )
    assert first.acquisition.instrument_resources_after == first_delta
    assert first.acquisition.instrument_resource_delta == first_delta
    assert first_observation.estimator_view() == first_safe
    assert first_safe is first.update.observation
    assert first_query is first.update.query
    assert first.update.completed_pair is None
    assert first.state is runner.state
    assert first.state.tracker_estimate is first.update.estimate
    assert first.state.normal_tracking_trace == (first.acquisition,)
    assert first.state.normal_tracking_trace[0] is first.acquisition
    assert first.state.pair_timings == ()
    assert first.state.instrument_resources_current == first_delta
    assert first.state.instrument_current_sequence_index == 0
    assert first.state.current_virtual_time_s == 0.006
    assert first.state.last_instrument_failure is None
    assert update_entries[0][0].pending_query is first_query
    assert update_entries[0][1] is first_safe

    second = runner.step()

    assert type(second) is TwoPointRunnerAccepted
    second_query = second.acquisition.query
    second_observation = second.acquisition.full_observation
    second_safe = second.acquisition.safe_observation
    timing = second.state.pair_timings[0]
    pair = second.update.completed_pair
    assert second.kind == "accepted"
    assert second_query.side == "plus"
    assert second.acquisition.resource_join_status == "authenticated"
    assert second.acquisition.expected_measurement_midpoint_s == 0.0095
    assert second.acquisition.measurement_midpoint_s == 0.0095
    assert second.acquisition.instrument_resources_before == first_delta
    assert second.acquisition.instrument_resources_after == instrument.resources
    assert second.acquisition.instrument_resource_delta.observations == 1
    assert second.acquisition.instrument_resource_delta.integration_time_s == 0.005
    assert second.acquisition.instrument_resource_delta.virtual_elapsed_time_s == 0.006
    assert second_observation.estimator_view() == second_safe
    assert second_safe is second.update.observation
    assert second_query is second.update.query
    assert pair is not None
    assert pair.pair_index == 0
    assert pair.resonance_id == "r0"
    assert second.state is runner.state
    assert second.state.tracker_estimate is second.update.estimate
    assert second.state.normal_tracking_trace == (
        first.acquisition,
        second.acquisition,
    )
    assert second.state.normal_tracking_trace[0] is first.acquisition
    assert second.state.normal_tracking_trace[1] is second.acquisition
    assert len(second.state.pair_timings) == 1
    assert timing.pair_index == pair.pair_index
    assert timing.resonance_id == pair.resonance_id
    assert timing.first_measurement_midpoint_s == 0.0035
    assert timing.second_measurement_midpoint_s == 0.0095
    assert timing.truth_reference_timestamp_s == (
        0.0035 + (0.0095 - 0.0035) / 2.0
    )
    assert timing.public_reference_timestamp_s == pair.pair_reference_timestamp_s
    assert timing.release_sequence_index == second_observation.sequence_index == 1
    assert timing.release_timestamp_s == second_observation.timestamp_s == 0.012
    assert second.state.instrument_resources_current == instrument.resources
    assert second.state.instrument_current_sequence_index == 1
    assert second.state.current_virtual_time_s == 0.012
    assert second.state.last_instrument_failure is None
    assert first.state.normal_tracking_trace == (first.acquisition,)
    assert first.state.pair_timings == ()
    assert update_entries[1][0].pending_query is second_query
    assert update_entries[1][0].accepted_observations == 1
    assert update_entries[1][1] is second_safe
    assert len(update_entries) == 2
    assert query_entries == [
        (
            ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0),
            0.0,
            first_query.frequency_hz,
            first_query.integration_time_s,
        ),
        (
            first_delta,
            0.006,
            second_query.frequency_hz,
            second_query.integration_time_s,
        ),
    ]


def test_pair_three_truth_and_public_references_use_distinct_associations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _conditional_tracking_inputs(monkeypatch)

    outcomes = tuple(runner.step() for _ in range(8))

    assert all(type(outcome) is TwoPointRunnerAccepted for outcome in outcomes)
    pair_three = outcomes[-1]
    timing = pair_three.state.pair_timings[3]
    pair = pair_three.update.completed_pair
    first_acquisition = pair_three.state.normal_tracking_trace[6]
    second_acquisition = pair_three.state.normal_tracking_trace[7]
    assert pair is not None
    assert timing.pair_index == pair.pair_index == 3
    assert timing.resonance_id == pair.resonance_id == "r3"
    assert timing.first_measurement_midpoint_s == (
        first_acquisition.measurement_midpoint_s
    )
    assert timing.second_measurement_midpoint_s == (
        second_acquisition.measurement_midpoint_s
    )
    assert (
        timing.truth_reference_timestamp_s.hex()
        == "0x1.5c28f5c28f5c4p-5"
    )
    assert (
        timing.public_reference_timestamp_s.hex()
        == "0x1.5c28f5c28f5c2p-5"
    )
    assert (
        timing.truth_reference_timestamp_s
        != timing.public_reference_timestamp_s
    )
    assert timing.public_reference_timestamp_s == pair.pair_reference_timestamp_s
    assert timing.release_sequence_index == pair.release_sequence_index == 7
    assert (
        timing.release_timestamp_s
        == pair.release_timestamp_s
        == second_acquisition.full_observation.timestamp_s
        == 0.048
    )
    assert timing.release_timestamp_s.hex() == "0x1.89374bc6a7efap-5"


def test_instrument_exception_preserves_identical_pending_query_and_can_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, tracker = _conditional_tracking_inputs(monkeypatch)
    original_query = ODMRInstrument.query
    original_update = CalibratedTwoPointTracker.update
    runner_state_before = runner.state
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s
    query_entries: list[
        tuple[TwoPointEstimate, ResourceSnapshot, float, float, float]
    ] = []
    update_calls: list[EstimatorObservation] = []
    transient = RuntimeError("transient instrument fault")

    def flaky_query(
        self: ODMRInstrument,
        frequency_hz: float,
        integration_time_s: float,
    ) -> InstrumentObservation:
        query_entries.append(
            (
                tracker.estimate(),
                self.resources,
                self.virtual_time_s,
                frequency_hz,
                integration_time_s,
            )
        )
        if len(query_entries) == 1:
            raise transient
        return original_query(self, frequency_hz, integration_time_s)

    def update_spy(
        self: CalibratedTwoPointTracker,
        observation: EstimatorObservation,
    ) -> TwoPointUpdate:
        update_calls.append(observation)
        return original_update(self, observation)

    monkeypatch.setattr(ODMRInstrument, "query", flaky_query)
    monkeypatch.setattr(CalibratedTwoPointTracker, "update", update_spy)

    failed = runner.step()

    assert type(failed) is TwoPointRunnerInstrumentFailure
    pending_query = failed.failure.query
    estimate_at_failure = query_entries[0][0]
    assert failed.kind == "instrument_failure"
    assert failed.failure.exception_type == "RuntimeError"
    assert failed.failure.exception_message == "transient instrument fault"
    assert failed.failure.instrument_resources_before == resources_before
    assert failed.failure.instrument_resources_after == resources_before
    assert failed.state is runner.state
    assert failed.state.phase == "tracking"
    assert failed.state.last_instrument_failure is failed.failure
    assert failed.state.tracker_estimate is estimate_at_failure
    assert failed.state.tracker_estimate is tracker.estimate()
    assert failed.state.tracker_estimate.pending_query is pending_query
    assert (
        failed.state.normal_tracking_trace
        is runner_state_before.normal_tracking_trace
    )
    assert failed.state.normal_tracking_trace == ()
    assert failed.state.pair_timings is runner_state_before.pair_timings
    assert failed.state.pair_timings == ()
    assert failed.state.instrument_resources_current == resources_before
    assert failed.state.instrument_current_sequence_index is None
    assert failed.state.current_virtual_time_s == time_before
    assert instrument.resources == resources_before
    assert instrument.virtual_time_s == time_before
    assert update_calls == []

    accepted = runner.step()

    assert type(accepted) is TwoPointRunnerAccepted
    assert accepted.acquisition.query is pending_query
    assert accepted.update.query is pending_query
    assert query_entries[1][0] is estimate_at_failure
    assert query_entries[1][3:] == query_entries[0][3:]
    assert accepted.state is runner.state
    assert accepted.state.phase == "tracking"
    assert accepted.state.last_instrument_failure is None
    assert accepted.state.normal_tracking_trace == (accepted.acquisition,)
    assert accepted.state.pair_timings == ()
    assert accepted.state.tracker_estimate is accepted.update.estimate
    assert accepted.state.tracker_estimate.accepted_observations == 1
    assert accepted.state.tracker_estimate.pending_query is None
    assert update_calls == [accepted.acquisition.safe_observation]
    assert failed.state.last_instrument_failure is failed.failure
    assert failed.state.normal_tracking_trace == ()


@pytest.mark.parametrize(
    ("case", "failure"),
    [
        ("query_issuance", KeyboardInterrupt("query issuance interrupted")),
        ("tracker_update", RuntimeError("update committed then failed")),
        ("runner_state", SystemExit("state constructed then interrupted")),
    ],
)
def test_step_transaction_restores_tracker_slots_and_reraises_identically(
    case: str,
    failure: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import runner as runner_module

    runner, instrument, tracker = _conditional_tracking_inputs(monkeypatch)
    runner_state_before = runner.state
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s
    configuration_before = tracker._configuration
    tracker_state_before = tracker._state
    original_choose = CalibratedTwoPointTracker.choose_next_query
    original_update = CalibratedTwoPointTracker.update
    update_checkpoint: list[tuple[object, object]] = []

    if case == "query_issuance":

        def choose_then_interrupt(self: CalibratedTwoPointTracker) -> object:
            original_choose(self)
            object.__setattr__(
                self,
                "_configuration",
                replace(self._configuration, proportional_gain=0.5),
            )
            raise failure

        monkeypatch.setattr(
            CalibratedTwoPointTracker,
            "choose_next_query",
            choose_then_interrupt,
        )
    else:

        def update_spy(
            self: CalibratedTwoPointTracker,
            observation: EstimatorObservation,
        ) -> TwoPointUpdate:
            update_checkpoint.append((self._configuration, self._state))
            update = original_update(self, observation)
            if case == "tracker_update":
                object.__setattr__(
                    self,
                    "_configuration",
                    replace(self._configuration, proportional_gain=0.5),
                )
                raise failure
            return update

        monkeypatch.setattr(CalibratedTwoPointTracker, "update", update_spy)

        if case == "runner_state":
            original_replace = runner_module.replace

            def replace_then_interrupt(
                instance: object,
                **changes: object,
            ) -> object:
                constructed = original_replace(instance, **changes)
                if "normal_tracking_trace" in changes:
                    object.__setattr__(
                        tracker,
                        "_configuration",
                        replace(
                            tracker._configuration,
                            proportional_gain=0.5,
                        ),
                    )
                    raise failure
                return constructed

            monkeypatch.setattr(runner_module, "replace", replace_then_interrupt)

    with pytest.raises(type(failure)) as raised:
        runner.step()

    assert raised.value is failure
    assert runner.state is runner_state_before
    assert runner.state.normal_tracking_trace is (
        runner_state_before.normal_tracking_trace
    )
    assert runner.state.pair_timings is runner_state_before.pair_timings
    if case == "query_issuance":
        assert tracker._configuration is configuration_before
        assert tracker._state is tracker_state_before
        assert instrument.resources == resources_before
        assert instrument.virtual_time_s == time_before
    else:
        assert len(update_checkpoint) == 1
        assert tracker._configuration is update_checkpoint[0][0]
        assert tracker._state is update_checkpoint[0][1]
        assert tracker.pending_query is not None
        assert instrument.resources.observations == resources_before.observations + 1
        assert instrument.virtual_time_s > time_before
