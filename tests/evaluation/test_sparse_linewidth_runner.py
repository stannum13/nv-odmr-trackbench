"""Contract tests for the sparse-linewidth evaluator runner shell."""

from __future__ import annotations

import inspect
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace

import pytest

from odmr_bench.dynamics import (
    SpectralDynamics,
    SpectralSnapshot,
    StationaryDynamics,
)
from odmr_bench.emulator import GaussianNoise
from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators import (
    SparseLinewidthCompositeTracker,
    SparseLinewidthConfiguration,
    TwoPointBudgetCeiling,
    TwoPointIdentityBinding,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    calibrate_two_point,
)
from odmr_bench.evaluation.sparse_linewidth import (
    SparseEvaluatorRunnerState,
    SparsePreflightError,
    SparseRunnerStateError,
    SparseStartError,
)
from odmr_bench.evaluation.sparse_linewidth.runner import (
    SparseLinewidthEvaluatorRunner,
)
from odmr_bench.evaluation.sparse_linewidth.types import (
    SparseRunnerAccepted,
    SparseRunnerInstrumentFailure,
)
from odmr_bench.evaluation.two_point.types import (
    TwoPointEvaluatorInstrumentConfiguration,
    TwoPointEvaluatorRunnerState,
    VerifiedTwoPointCalibrationSuccess,
)
from odmr_bench.models import Baseline, Resonance
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
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


def _instrument() -> ODMRInstrument:
    return ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=GaussianNoise(stddev_at_1s=0.0),
        nominal_photon_rate_hz=2.5e6,
        frequency_overhead_s=0.001,
        seed=13,
    )


def _calibration_arguments() -> dict[str, object]:
    fit_configuration = make_legal_fit_configuration()
    return {
        "frequency_hz": (2.74e9, 3.02e9),
        "integration_time_s": 0.005,
        "fit_configuration": fit_configuration,
        "identity_binding": TwoPointIdentityBinding(
            "require_expected_ids", fit_configuration.resonance_ids
        ),
        "source_id": "verified-source",
        "source_clock_id": "clock",
        "tracker_clock_id": "clock",
        "source_to_tracker_offset_s": 0.0,
        "physical_fit_epoch_rule": "instrument_midpoint_ordered_mean",
    }


@dataclass(frozen=True, slots=True)
class QueryScopedDynamicsCall:
    timestamp_s: float
    inside_instrument_query: bool


class QueryScopedDynamicsSpy:
    def __init__(self, base_dynamics: SpectralDynamics) -> None:
        self._base_dynamics = base_dynamics
        self._inside_instrument_query = False
        self._calls: list[QueryScopedDynamicsCall] = []

    @property
    def calls(self) -> tuple[QueryScopedDynamicsCall, ...]:
        return tuple(self._calls)

    def clear_calls(self) -> None:
        self._calls.clear()

    @contextmanager
    def instrument_query_scope(self) -> AbstractContextManager[None]:
        previous = self._inside_instrument_query
        self._inside_instrument_query = True
        try:
            yield
        finally:
            self._inside_instrument_query = previous

    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        self._calls.append(
            QueryScopedDynamicsCall(timestamp_s, self._inside_instrument_query)
        )
        return self._base_dynamics.snapshot_at(timestamp_s)


def _started_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scan_period_fast_pairs: int = 8,
    dynamics: SpectralDynamics | None = None,
) -> tuple[SparseLinewidthEvaluatorRunner, ODMRInstrument]:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    arguments = _calibration_arguments()
    fit_configuration = arguments["fit_configuration"]
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: make_legal_source_fit(
            fit_configuration  # type: ignore[arg-type]
        ),
    )
    instrument = ODMRInstrument(
        dynamics=dynamics or StationaryDynamics(_snapshot()),
        noise=GaussianNoise(stddev_at_1s=0.0),
        nominal_photon_rate_hz=2.5e6,
        frequency_overhead_s=0.001,
        seed=13,
    )
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)
    success = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]
    assert type(success) is VerifiedTwoPointCalibrationSuccess
    tracker = SparseLinewidthCompositeTracker(
        SparseLinewidthConfiguration(
            scan_period_fast_pairs=scan_period_fast_pairs,
        )
    )
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="included_same_run",
    )
    metadata = TwoPointRunMetadata(
        tracker_clock_id="clock",
        current_sequence_index=runner.state.instrument_current_sequence_index,
        current_timestamp_s=runner.state.current_virtual_time_s,
        nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
        frequency_overhead_s=instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    runner.start_tracking(
        tracker,
        calibration,
        success,
        metadata,
        TwoPointBudgetCeiling(100, None, None, None),
        seed=17,
    )
    return runner, instrument


def _ordered_mean(values: tuple[float, ...]) -> float:
    result = values[0]
    for count, value in enumerate(values[1:], start=2):
        result = result + (value - result) / count
    return result


def _verified_success(
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

    arguments = _calibration_arguments()
    arguments.update(
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


def test_sparse_runner_signatures_are_exact() -> None:
    signatures = {
        "bind": inspect.signature(SparseLinewidthEvaluatorRunner.bind),
        "state": inspect.signature(SparseLinewidthEvaluatorRunner.state.fget),
        "acquire": inspect.signature(
            SparseLinewidthEvaluatorRunner.acquire_verified_calibration
        ),
        "start": inspect.signature(SparseLinewidthEvaluatorRunner.start_tracking),
        "step": inspect.signature(SparseLinewidthEvaluatorRunner.step),
        "run": inspect.signature(SparseLinewidthEvaluatorRunner.run_until_event),
        "stop": inspect.signature(SparseLinewidthEvaluatorRunner.stop_external),
    }

    assert str(signatures["bind"]) == (
        "(instrument: 'ODMRInstrument') -> 'SparseLinewidthEvaluatorRunner'"
    )
    assert str(signatures["state"]) == ("(self) -> 'SparseEvaluatorRunnerState'")
    assert str(signatures["acquire"]) == (
        "(self, frequency_hz: 'Sequence[float]', integration_time_s: 'float', "
        "fit_configuration: 'FitConfiguration', identity_binding: "
        "'TwoPointIdentityBinding', *, source_id: 'str', source_clock_id: "
        "'str', tracker_clock_id: 'str', source_to_tracker_offset_s: 'float', "
        "physical_fit_epoch_rule: \"Literal['instrument_midpoint_ordered_mean']\") "
        "-> 'VerifiedTwoPointCalibrationOutcome'"
    )
    assert str(signatures["start"]) == (
        "(self, tracker: 'SparseLinewidthCompositeTracker', calibration: "
        "'TwoPointCalibration', verified_calibration: "
        "'VerifiedTwoPointCalibrationSuccess', public_metadata: "
        "'TwoPointRunMetadata', budget_ceiling: 'TwoPointBudgetCeiling', *, "
        "seed: 'int') -> 'SparseEvaluatorRunnerState'"
    )
    assert str(signatures["step"]) == "(self) -> 'SparseRunnerStepOutcome'"
    assert str(signatures["run"]) == "(self) -> 'SparseRunnerRunOutcome'"
    assert str(signatures["stop"]) == ("(self) -> 'SparseRunnerExternallyStopped'")


def test_sparse_runner_bind_captures_exact_clean_boundary_and_read_only_state() -> None:
    instrument = _instrument()

    runner = SparseLinewidthEvaluatorRunner.bind(instrument)

    zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    state = runner.state
    assert type(runner) is SparseLinewidthEvaluatorRunner
    assert type(state) is SparseEvaluatorRunnerState
    assert state is runner.state
    assert state.phase == "ready"
    assert state.instrument_configuration == (
        TwoPointEvaluatorInstrumentConfiguration(2.5e6, 0.001)
    )
    assert state.instrument_resources_at_bind == zero
    assert state.instrument_resources_current == zero
    assert state.instrument_resources_at_bind is state.instrument_resources_current
    assert state.instrument_current_sequence_index is None
    assert state.current_virtual_time_s == 0.0
    assert state.calibration_outcome is None
    assert state.verified_calibration is None
    assert state.calibration is None
    assert state.tracker_estimate is None
    assert state.normal_tracking_trace == ()
    assert state.pair_timings == ()
    assert state.scan_timings == ()
    assert state.tracking_resources_before is None
    assert state.last_instrument_failure is None
    assert state.terminal_abort is None
    assert state.fast_update_cpu_time_s == 0.0
    assert state.sparse_update_cpu_time_s == 0.0
    assert state.total_update_cpu_time_s == 0.0
    assert instrument.resources == zero
    assert instrument.virtual_time_s == 0.0
    from odmr_bench.evaluation.two_point.provenance import (
        _lookup_run_token_binding,
        _lookup_verified_calibration_issuer,
    )

    binding = _lookup_run_token_binding(state.run_token)
    assert binding is not None
    assert binding.issuer_runner is runner
    assert binding.instrument is instrument
    assert binding.instrument_configuration is state.instrument_configuration
    issuer = _lookup_verified_calibration_issuer(runner)
    assert issuer._runner is runner
    assert issuer._instrument is instrument
    assert issuer._run_token is state.run_token
    assert issuer._instrument_configuration is state.instrument_configuration
    with pytest.raises(AttributeError):
        runner.state = state  # type: ignore[misc]


@pytest.mark.parametrize(
    ("instrument", "code"),
    [
        (object(), "invalid_argument_type"),
        (_instrument(), "unclean_instrument_boundary"),
    ],
)
def test_sparse_runner_bind_rejects_nonexact_or_unclean_instrument(
    instrument: object, code: str
) -> None:
    if type(instrument) is ODMRInstrument:
        instrument.query(2.8e9, 0.005)
    with pytest.raises(SparsePreflightError) as raised:
        SparseLinewidthEvaluatorRunner.bind(instrument)  # type: ignore[arg-type]
    assert raised.value.code == code


def test_sparse_runner_rejects_tracking_operations_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrument = _instrument()
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)

    def reject_query(*args: object, **kwargs: object) -> object:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(ODMRInstrument, "query", reject_query)
    state_before = runner.state
    with pytest.raises(SparseRunnerStateError):
        runner.step()
    with pytest.raises(SparseRunnerStateError):
        runner.run_until_event()
    with pytest.raises(SparseRunnerStateError):
        runner.stop_external()
    assert runner.state is state_before
    assert instrument.resources == state_before.instrument_resources_current


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("phase", "invalid_runner_phase"),
        ("type", "invalid_argument_type"),
        ("value", "invalid_argument_value"),
        ("grid", "invalid_frequency_grid"),
        ("fit", "invalid_fit_or_identity_configuration"),
        ("clock", "invalid_clock_mapping"),
        ("boundary", "unclean_instrument_boundary"),
    ],
)
def test_sparse_calibration_preflight_precedence_and_atomicity(
    case: str, expected_code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    instrument = _instrument()
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)
    arguments = _calibration_arguments()
    if case == "phase":
        object.__setattr__(runner.state, "phase", "calibration_failed")
        arguments["frequency_hz"] = object()
    elif case == "type":
        arguments["frequency_hz"] = (True, 3.02e9)
        arguments["integration_time_s"] = 0.0
    elif case == "value":
        arguments["frequency_hz"] = (-1.0, -1.0)
        arguments["fit_configuration"] = make_legal_fit_configuration(
            tuple(f"x{i}" for i in range(8))
        )
    elif case == "grid":
        arguments["frequency_hz"] = (2.74e9, 2.74e9)
        arguments["fit_configuration"] = make_legal_fit_configuration(
            tuple(f"x{i}" for i in range(8))
        )
    elif case == "fit":
        arguments["fit_configuration"] = make_legal_fit_configuration(
            tuple(f"x{i}" for i in range(8))
        )
        arguments["source_clock_id"] = ""
    elif case == "clock":
        arguments["source_to_tracker_offset_s"] = 1.0
    else:
        instrument.query(2.8e9, 0.005)

    query_calls = 0

    def reject_query(*args: object, **kwargs: object) -> object:
        nonlocal query_calls
        query_calls += 1
        raise AssertionError((args, kwargs))

    if case != "boundary":
        monkeypatch.setattr(ODMRInstrument, "query", reject_query)
    state_before = runner.state
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s
    with pytest.raises(SparsePreflightError) as raised:
        runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]
    assert raised.value.code == expected_code
    assert runner.state is state_before
    assert instrument.resources == resources_before
    assert instrument.virtual_time_s == time_before
    assert query_calls == 0


def test_sparse_calibration_success_uses_private_core_and_enters_success_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, outcome = _verified_success(monkeypatch)

    assert runner.state.phase == "calibration_succeeded"
    assert runner.state.calibration_outcome is outcome
    assert runner.state.verified_calibration is outcome
    assert runner.state.instrument_resources_current == instrument.resources
    assert runner.state.current_virtual_time_s == instrument.virtual_time_s


def test_sparse_calibration_fit_failure_enters_exact_failure_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module
    from odmr_bench.evaluation.two_point.types import VerifiedTwoPointCalibrationFailure

    instrument = _instrument()
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit exploded")),
    )

    outcome = runner.acquire_verified_calibration(  # type: ignore[arg-type]
        **_calibration_arguments()
    )

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "fit_exception"
    assert outcome.exception_type == "RuntimeError"
    assert outcome.exception_message == "fit exploded"
    assert runner.state.phase == "calibration_failed"
    assert runner.state.calibration_outcome is outcome
    assert runner.state.verified_calibration is None
    assert runner.state.instrument_resources_current == instrument.resources
    assert runner.state.current_virtual_time_s == instrument.virtual_time_s
    assert len(outcome.full_observations) == 2
    assert outcome.safe_observations == tuple(
        observation.estimator_view() for observation in outcome.full_observations
    )


def test_sparse_calibration_rolls_back_private_success_binding_before_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module
    from odmr_bench.evaluation.two_point.provenance import _lookup_run_token_binding
    from odmr_bench.evaluation.two_point.types import VerifiedTwoPointCalibrationFailure

    arguments = _calibration_arguments()
    successful_fit = make_legal_source_fit(arguments["fit_configuration"])
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda *args, **kwargs: successful_fit,
    )
    instrument = _instrument()
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)
    original_bind = calibration_module._bind_run_token_success

    def commit_then_fail(*args: object, **kwargs: object) -> None:
        original_bind(*args, **kwargs)
        raise RuntimeError("success binding exploded")

    monkeypatch.setattr(
        calibration_module, "_bind_run_token_success", commit_then_fail
    )

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "source_binding_failed"
    assert runner.state.phase == "calibration_failed"
    binding = _lookup_run_token_binding(runner.state.run_token)
    assert binding is not None
    assert binding.success is None
    assert binding.source is None
    assert runner.state.instrument_resources_current == instrument.resources


def test_sparse_start_rejects_calibration_mismatch_before_tracker_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, success = _verified_success(monkeypatch)
    configuration = SparseLinewidthConfiguration()
    tracker = SparseLinewidthCompositeTracker(configuration)
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="included_same_run",
    )
    mismatched = calibrate_two_point(
        make_legal_caller_asserted_source(),
        TwoPointTrackerConfiguration(),
        budget_treatment="conditional_free_precalibration",
    )
    metadata = TwoPointRunMetadata(
        tracker_clock_id="clock",
        current_sequence_index=runner.state.instrument_current_sequence_index,
        current_timestamp_s=runner.state.current_virtual_time_s,
        nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
        frequency_overhead_s=instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    budget = TwoPointBudgetCeiling(100, None, None, None)
    reset_calls = 0

    def reset_spy(*args: object, **kwargs: object) -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(SparseLinewidthCompositeTracker, "reset", reset_spy)
    with pytest.raises(SparseStartError) as caught:
        runner.start_tracking(
            tracker, mismatched, success, metadata, budget, seed=0
        )
    assert caught.value.code == "calibration_mismatch"
    assert reset_calls == 0
    assert runner.state.phase == "calibration_succeeded"
    assert instrument.resources == runner.state.instrument_resources_current
    assert calibration.source is success.source


def test_sparse_start_same_run_enters_tracking_with_exact_reset_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, success = _verified_success(monkeypatch)
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="included_same_run",
    )
    metadata = TwoPointRunMetadata(
        tracker_clock_id="clock",
        current_sequence_index=runner.state.instrument_current_sequence_index,
        current_timestamp_s=runner.state.current_virtual_time_s,
        nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
        frequency_overhead_s=instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    budget = TwoPointBudgetCeiling(100, None, None, None)

    state = runner.start_tracking(
        tracker, calibration, success, metadata, budget, seed=7
    )

    assert state is runner.state
    assert state.phase == "tracking"
    assert state.calibration_outcome is success
    assert state.verified_calibration is success
    assert state.calibration is calibration
    assert state.tracker_estimate is tracker.estimate()
    assert state.tracking_resources_before == instrument.resources
    assert state.instrument_resources_current == instrument.resources
    assert state.normal_tracking_trace == ()
    assert state.pair_timings == ()
    assert state.scan_timings == ()
    assert state.fast_update_cpu_time_s == state.tracker_estimate.fast_update_cpu_time_s
    assert state.sparse_update_cpu_time_s == (
        state.tracker_estimate.sparse_update_cpu_time_s
    )
    assert state.total_update_cpu_time_s == (
        state.tracker_estimate.total_update_cpu_time_s
    )


def test_sparse_start_allows_authenticated_conditional_other_runner_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_runner, _, success = _verified_success(
        monkeypatch,
        source_clock_id="source-clock",
        tracker_clock_id="tracking-clock",
        source_to_tracker_offset_s=-0.012,
    )
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="conditional_free_precalibration",
    )
    instrument = _instrument()
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    metadata = TwoPointRunMetadata(
        tracker_clock_id="tracking-clock",
        current_sequence_index=None,
        current_timestamp_s=0.0,
        nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
        frequency_overhead_s=instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )

    state = runner.start_tracking(
        tracker,
        calibration,
        success,
        metadata,
        TwoPointBudgetCeiling(100, None, None, None),
        seed=9,
    )

    assert state.phase == "tracking"
    assert state.calibration_outcome is None
    assert state.verified_calibration is success
    assert state.run_token is not success.run_token
    assert state.tracking_resources_before == ResourceSnapshot(
        0, 0.0, 0.0, 0.0, 0, 0, 0.0
    )
    assert source_runner.state.phase == "calibration_succeeded"


@pytest.mark.parametrize(
    "attack",
    [
        "binding_runner_object",
        "binding_runner_registered",
        "binding_instrument",
        "binding_configuration",
        "source_token",
        "source_instrument",
        "source_configuration",
        "source_phase",
        "source_calibration_outcome",
        "source_verified_calibration",
        "source_cross_class_state",
    ],
)
def test_conditional_start_rejects_broken_live_source_identity_graph_before_reset(
    attack: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point.provenance import _lookup_run_token_binding

    source_runner, source_instrument, success = _verified_success(
        monkeypatch,
        source_clock_id="source-clock",
        tracker_clock_id="tracking-clock",
        source_to_tracker_offset_s=-0.012,
    )
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="conditional_free_precalibration",
    )
    target_instrument = _instrument()
    target_runner = SparseLinewidthEvaluatorRunner.bind(target_instrument)
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    metadata = TwoPointRunMetadata(
        tracker_clock_id="tracking-clock",
        current_sequence_index=None,
        current_timestamp_s=0.0,
        nominal_photon_rate_hz=target_instrument.nominal_photon_rate_hz,
        frequency_overhead_s=target_instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    binding = _lookup_run_token_binding(success.run_token)
    assert binding is not None
    assert binding.issuer_runner is source_runner
    assert binding.instrument is source_instrument
    assert binding.instrument_configuration is (
        source_runner.state.instrument_configuration
    )
    assert binding.success is success
    assert binding.source is success.source

    restoration: tuple[object, str, object] | None
    if attack == "binding_runner_object":
        restoration = (binding, "issuer_runner", binding.issuer_runner)
        object.__setattr__(binding, "issuer_runner", object())
    elif attack == "binding_runner_registered":
        other_runner = SparseLinewidthEvaluatorRunner.bind(_instrument())
        restoration = (binding, "issuer_runner", binding.issuer_runner)
        object.__setattr__(binding, "issuer_runner", other_runner)
    elif attack == "binding_instrument":
        restoration = (binding, "instrument", binding.instrument)
        object.__setattr__(binding, "instrument", _instrument())
    elif attack == "binding_configuration":
        restoration = (
            binding,
            "instrument_configuration",
            binding.instrument_configuration,
        )
        object.__setattr__(
            binding,
            "instrument_configuration",
            TwoPointEvaluatorInstrumentConfiguration(3.0e6, 0.002),
        )
    elif attack == "source_token":
        other_runner = SparseLinewidthEvaluatorRunner.bind(_instrument())
        restoration = (source_runner.state, "run_token", source_runner.state.run_token)
        object.__setattr__(
            source_runner.state, "run_token", other_runner.state.run_token
        )
    elif attack == "source_instrument":
        restoration = (source_runner, "_instrument", source_runner._instrument)
        object.__setattr__(source_runner, "_instrument", _instrument())
    elif attack == "source_configuration":
        restoration = (
            source_runner.state,
            "instrument_configuration",
            source_runner.state.instrument_configuration,
        )
        object.__setattr__(
            source_runner.state,
            "instrument_configuration",
            TwoPointEvaluatorInstrumentConfiguration(3.0e6, 0.002),
        )
    elif attack == "source_phase":
        restoration = (source_runner.state, "phase", source_runner.state.phase)
        object.__setattr__(source_runner.state, "phase", "calibration_failed")
    elif attack == "source_calibration_outcome":
        restoration = (
            source_runner.state,
            "calibration_outcome",
            source_runner.state.calibration_outcome,
        )
        object.__setattr__(
            source_runner.state, "calibration_outcome", replace(success)
        )
    elif attack == "source_verified_calibration":
        restoration = (
            source_runner.state,
            "verified_calibration",
            source_runner.state.verified_calibration,
        )
        object.__setattr__(
            source_runner.state, "verified_calibration", replace(success)
        )
    else:
        source_state = source_runner.state
        restoration = (source_runner, "_state", source_state)
        object.__setattr__(
            source_runner,
            "_state",
            TwoPointEvaluatorRunnerState(
                phase=source_state.phase,
                run_token=source_state.run_token,
                instrument_configuration=source_state.instrument_configuration,
                calibration_outcome=source_state.calibration_outcome,
                verified_calibration=source_state.verified_calibration,
                calibration=source_state.calibration,
                tracker_estimate=None,
                normal_tracking_trace=(),
                pair_timings=(),
                instrument_resources_at_bind=source_state.instrument_resources_at_bind,
                tracking_resources_before=None,
                instrument_resources_current=source_state.instrument_resources_current,
                instrument_current_sequence_index=(
                    source_state.instrument_current_sequence_index
                ),
                current_virtual_time_s=source_state.current_virtual_time_s,
                last_instrument_failure=None,
                terminal_abort=None,
            ),
        )

    attacked_source_state = source_runner.state
    target_state_before = target_runner.state
    source_resources_before = source_instrument.resources
    source_time_before = source_instrument.virtual_time_s
    target_resources_before = target_instrument.resources
    target_time_before = target_instrument.virtual_time_s
    reset_calls = 0

    def reset_spy(*args: object, **kwargs: object) -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(SparseLinewidthCompositeTracker, "reset", reset_spy)
    try:
        with pytest.raises(SparseStartError) as raised:
            target_runner.start_tracking(
                tracker,
                calibration,
                success,
                metadata,
                TwoPointBudgetCeiling(100, None, None, None),
                seed=9,
            )
        assert raised.value.code == "run_provenance_mismatch"
        assert reset_calls == 0
        assert target_runner.state is target_state_before
        assert source_runner.state is attacked_source_state
        assert source_instrument.resources == source_resources_before
        assert source_instrument.virtual_time_s == source_time_before
        assert target_instrument.resources == target_resources_before
        assert target_instrument.virtual_time_s == target_time_before
    finally:
        object.__setattr__(restoration[0], restoration[1], restoration[2])


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
def test_sparse_start_error_precedence_and_exact_rollback(
    case: str,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, success = _verified_success(monkeypatch)
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="included_same_run",
    )
    metadata = TwoPointRunMetadata(
        tracker_clock_id="clock",
        current_sequence_index=runner.state.instrument_current_sequence_index,
        current_timestamp_s=runner.state.current_virtual_time_s,
        nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
        frequency_overhead_s=instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    budget = TwoPointBudgetCeiling(100, None, None, None)
    call_seed: object = 17
    call_success = success
    call_calibration = calibration
    call_metadata = metadata
    if case == "invalid_runner_phase":
        object.__setattr__(runner.state, "phase", "tracking")
        call_seed = True
        call_success = replace(success)
    elif case == "invalid_argument_type":
        call_seed = True
        call_success = replace(success)
    elif case == "unverified_calibration":
        call_success = replace(success)
        call_metadata = replace(metadata, nominal_photon_rate_hz=3.0e6)
    elif case == "calibration_mismatch":
        call_calibration = calibrate_two_point(
            make_legal_caller_asserted_source(),
            TwoPointTrackerConfiguration(),
            budget_treatment="conditional_free_precalibration",
        )
        call_metadata = replace(metadata, nominal_photon_rate_hz=3.0e6)
    elif case == "run_provenance_mismatch":
        object.__setattr__(runner.state, "calibration_outcome", replace(success))
        call_metadata = replace(metadata, nominal_photon_rate_hz=3.0e6)
    elif case == "metadata_mismatch":
        call_metadata = replace(metadata, nominal_photon_rate_hz=3.0e6)
    elif case == "resource_boundary_mismatch":
        instrument.query(2.8e9, 0.005)

    runner_state_before = runner.state
    tracker_configuration_before = tracker._configuration
    tracker_configuration_snapshot_before = tracker._configuration_snapshot
    tracker_state_before = tracker._state
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s
    reset_calls = 0
    original_reset = SparseLinewidthCompositeTracker.reset
    reset_failure = RuntimeError("reset committed then failed")

    def reset_sentinel(
        self: SparseLinewidthCompositeTracker,
        public_metadata: TwoPointRunMetadata,
        supplied_calibration: object,
        budget_ceiling: object,
        *,
        seed: int,
    ) -> None:
        nonlocal reset_calls
        reset_calls += 1
        if case != "tracker_reset_failed":
            raise AssertionError("start preflight reached tracker.reset")
        original_reset(
            self,
            public_metadata,
            supplied_calibration,  # type: ignore[arg-type]
            budget_ceiling,  # type: ignore[arg-type]
            seed=seed,
        )
        object.__setattr__(
            self, "_configuration", SparseLinewidthConfiguration(max_nfev=9)
        )
        object.__setattr__(
            self,
            "_configuration_snapshot",
            SparseLinewidthConfiguration(max_nfev=10),
        )
        raise reset_failure

    monkeypatch.setattr(SparseLinewidthCompositeTracker, "reset", reset_sentinel)

    with pytest.raises(SparseStartError) as raised:
        runner.start_tracking(
            tracker,
            call_calibration,
            call_success,
            call_metadata,
            budget,
            seed=call_seed,  # type: ignore[arg-type]
        )

    assert raised.value.code == expected_code
    assert runner.state is runner_state_before
    assert tracker._configuration is tracker_configuration_before
    assert tracker._configuration_snapshot is tracker_configuration_snapshot_before
    assert tracker._state is tracker_state_before
    assert instrument.resources == resources_before
    assert instrument.virtual_time_s == time_before
    assert reset_calls == (1 if case == "tracker_reset_failed" else 0)
    if case == "tracker_reset_failed":
        assert raised.value.__cause__ is reset_failure


@pytest.mark.parametrize(
    ("join_case", "expected_code"),
    [
        ("token", "unverified_calibration"),
        ("runner", "run_provenance_mismatch"),
        ("instrument", "run_provenance_mismatch"),
        ("treatment", "run_provenance_mismatch"),
        ("clock", "metadata_mismatch"),
    ],
)
def test_sparse_start_authenticates_each_private_and_public_join_before_reset(
    join_case: str,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point.provenance import _lookup_run_token_binding

    runner, instrument, success = _verified_success(monkeypatch)
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="included_same_run",
    )
    metadata = TwoPointRunMetadata(
        tracker_clock_id="clock",
        current_sequence_index=runner.state.instrument_current_sequence_index,
        current_timestamp_s=runner.state.current_virtual_time_s,
        nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
        frequency_overhead_s=instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    call_success = success
    binding = _lookup_run_token_binding(success.run_token)
    assert binding is not None
    restore: tuple[object, str, object] | None = None
    if join_case == "token":
        other = SparseLinewidthEvaluatorRunner.bind(_instrument())
        call_success = replace(success, run_token=other.state.run_token)
    elif join_case == "runner":
        restore = (binding, "issuer_runner", binding.issuer_runner)
        object.__setattr__(binding, "issuer_runner", object())
    elif join_case == "instrument":
        restore = (binding, "instrument", binding.instrument)
        object.__setattr__(binding, "instrument", _instrument())
    elif join_case == "treatment":
        restore = (calibration, "budget_treatment", calibration.budget_treatment)
        object.__setattr__(calibration, "budget_treatment", "invalid")
    else:
        metadata = replace(metadata, tracker_clock_id="different-clock")

    reset_calls = 0

    def reset_spy(*args: object, **kwargs: object) -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(SparseLinewidthCompositeTracker, "reset", reset_spy)
    try:
        with pytest.raises(SparseStartError) as raised:
            runner.start_tracking(
                tracker,
                calibration,
                call_success,
                metadata,
                TwoPointBudgetCeiling(100, None, None, None),
                seed=3,
            )
    finally:
        if restore is not None:
            target, attribute, value = restore
            object.__setattr__(target, attribute, value)

    assert raised.value.code == expected_code
    assert reset_calls == 0
    assert runner.state.phase == "calibration_succeeded"


def test_sparse_start_reset_process_control_restores_tracker_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopNow(BaseException):
        pass

    runner, instrument, success = _verified_success(monkeypatch)
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    calibration = calibrate_two_point(
        success.source,
        TwoPointTrackerConfiguration(),
        budget_treatment="included_same_run",
    )
    metadata = TwoPointRunMetadata(
        tracker_clock_id="clock",
        current_sequence_index=runner.state.instrument_current_sequence_index,
        current_timestamp_s=runner.state.current_virtual_time_s,
        nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
        frequency_overhead_s=instrument.frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )
    runner_state_before = runner.state
    configuration_before = tracker._configuration
    configuration_snapshot_before = tracker._configuration_snapshot
    tracker_state_before = tracker._state
    stop = StopNow()

    def reset_then_stop(*args: object, **kwargs: object) -> None:
        object.__setattr__(
            tracker, "_configuration", SparseLinewidthConfiguration(max_nfev=9)
        )
        object.__setattr__(
            tracker,
            "_configuration_snapshot",
            SparseLinewidthConfiguration(max_nfev=10),
        )
        object.__setattr__(tracker, "_state", object())
        raise stop

    monkeypatch.setattr(
        SparseLinewidthCompositeTracker, "reset", reset_then_stop
    )

    with pytest.raises(StopNow) as raised:
        runner.start_tracking(
            tracker,
            calibration,
            success,
            metadata,
            TwoPointBudgetCeiling(100, None, None, None),
            seed=3,
        )

    assert raised.value is stop
    assert runner.state is runner_state_before
    assert tracker._configuration is configuration_before
    assert tracker._configuration_snapshot is configuration_snapshot_before
    assert tracker._state is tracker_state_before


def test_accepted_fast_step_retains_exact_causal_join_and_cpu_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument = _started_runner(monkeypatch)
    state_before = runner.state
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s

    outcome = runner.step()

    assert type(outcome) is SparseRunnerAccepted
    acquisition = outcome.acquisition
    assert acquisition.mode == "fast_pair"
    assert acquisition.instrument_resources_before == resources_before
    assert acquisition.instrument_resources_after == instrument.resources
    assert acquisition.expected_measurement_midpoint_s == (
        time_before
        + runner.state.instrument_configuration.frequency_overhead_s
        + acquisition.query.integration_time_s / 2.0
    )
    assert acquisition.measurement_midpoint_s == (
        acquisition.full_observation.timestamp_s
        - acquisition.full_observation.integration_time_s / 2.0
    )
    assert acquisition.safe_observation == acquisition.full_observation.estimator_view()
    assert outcome.update.query is acquisition.query
    assert outcome.update.observation is acquisition.safe_observation
    assert outcome.state.normal_tracking_trace == (acquisition,)
    assert outcome.state.tracker_estimate is outcome.update.estimate
    assert outcome.state.instrument_current_sequence_index == (
        acquisition.full_observation.sequence_index
    )
    assert outcome.state.current_virtual_time_s == instrument.virtual_time_s
    assert outcome.state.last_instrument_failure is None
    assert outcome.state.fast_update_cpu_time_s == (
        outcome.update.estimate.fast_update_cpu_time_s
    )
    assert outcome.state.sparse_update_cpu_time_s == (
        outcome.update.estimate.sparse_update_cpu_time_s
    )
    assert outcome.state.total_update_cpu_time_s == (
        outcome.update.estimate.total_update_cpu_time_s
    )
    assert outcome.state is not state_before


def test_instrument_query_exception_is_retryable_and_uncharged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument = _started_runner(monkeypatch)
    original_query = ODMRInstrument.query
    pending_query = runner._tracker.choose_next_query()
    state_before = runner.state
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s
    calls = 0

    def fail_once(self: ODMRInstrument, *args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary acquisition failure")
        return original_query(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ODMRInstrument, "query", fail_once)

    failure = runner.step()

    assert type(failure) is SparseRunnerInstrumentFailure
    assert failure.failure.mode == "fast_pair"
    assert failure.failure.query is pending_query
    assert failure.failure.exception_type == "RuntimeError"
    assert failure.failure.exception_message == "temporary acquisition failure"
    assert failure.failure.instrument_resources_before == resources_before
    assert failure.failure.instrument_resources_after == resources_before
    assert failure.state.normal_tracking_trace == ()
    assert failure.state.tracker_estimate is runner._tracker.estimate()
    assert failure.state.tracker_estimate != state_before.tracker_estimate
    assert failure.state.tracker_estimate.pending_query is pending_query
    assert failure.state.instrument_resources_current == resources_before
    assert failure.state.current_virtual_time_s == time_before
    assert instrument.resources == resources_before
    assert instrument.virtual_time_s == time_before

    accepted = runner.step()
    assert type(accepted) is SparseRunnerAccepted
    assert accepted.acquisition.query is pending_query


def test_completed_fast_pair_retains_existing_pair_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = _started_runner(monkeypatch)
    first = runner.step()
    second = runner.step()
    assert type(first) is SparseRunnerAccepted
    assert type(second) is SparseRunnerAccepted
    pair = second.update.completed_fast_pair
    assert pair is not None

    timing = second.state.pair_timings[-1]
    first_midpoint = first.acquisition.measurement_midpoint_s
    second_midpoint = second.acquisition.measurement_midpoint_s
    assert first_midpoint is not None
    assert second_midpoint is not None
    assert timing.pair_index == pair.pair_index
    assert timing.resonance_id == pair.resonance_id
    assert timing.first_measurement_midpoint_s == first_midpoint
    assert timing.second_measurement_midpoint_s == second_midpoint
    assert timing.truth_reference_timestamp_s == (
        first_midpoint + (second_midpoint - first_midpoint) / 2.0
    )
    assert timing.public_reference_timestamp_s == pair.pair_reference_timestamp_s
    assert timing.release_sequence_index == pair.release_sequence_index
    assert timing.release_timestamp_s == pair.release_timestamp_s


def test_completed_scan_retains_timing_without_out_of_query_dynamics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = QueryScopedDynamicsSpy(StationaryDynamics(_snapshot()))
    runner, _ = _started_runner(
        monkeypatch, scan_period_fast_pairs=1, dynamics=spy
    )
    original_query = ODMRInstrument.query

    def query_in_declared_signal_scope(
        self: ODMRInstrument, *args: object, **kwargs: object
    ) -> object:
        with spy.instrument_query_scope():
            return original_query(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ODMRInstrument, "query", query_in_declared_signal_scope)
    spy.clear_calls()
    outcomes = [runner.step() for _ in range(7)]
    assert all(type(outcome) is SparseRunnerAccepted for outcome in outcomes)
    outcome = outcomes[-1]
    assert type(outcome) is SparseRunnerAccepted
    scan = outcome.update.completed_sparse_scan
    assert scan is not None
    timing = outcome.state.scan_timings[-1]
    scan_acquisitions = outcome.state.normal_tracking_trace[-5:]
    actual_midpoints = tuple(
        acquisition.measurement_midpoint_s for acquisition in scan_acquisitions
    )
    assert all(value is not None for value in actual_midpoints)
    actual_midpoints = tuple(float(value) for value in actual_midpoints)
    public_midpoints = tuple(
        observation.timestamp_s - observation.integration_time_s / 2.0
        for observation in scan.observations
    )
    assert timing.measurement_midpoints_s == actual_midpoints
    assert timing.public_reference_timestamp_s == _ordered_mean(public_midpoints)
    assert timing.truth_reference_timestamp_s == _ordered_mean(actual_midpoints)
    assert timing.release_sequence_index == outcome.update.observation.sequence_index
    assert timing.release_timestamp_s == outcome.update.observation.timestamp_s
    assert spy.calls
    assert all(call.inside_instrument_query for call in spy.calls)
