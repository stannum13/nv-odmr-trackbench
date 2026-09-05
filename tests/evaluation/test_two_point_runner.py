"""Tests for the evaluator-owned two-point tracking runner."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from odmr_bench.dynamics import SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import GaussianNoise
from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators import (
    CalibratedTwoPointTracker,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointIdentityBinding,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    calibrate_two_point,
)
from odmr_bench.evaluation.two_point import (
    TwoPointEvaluatorRunner,
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
