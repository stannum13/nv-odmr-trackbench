"""Contract tests for the sparse-linewidth evaluator runner shell."""

from __future__ import annotations

import gc
import inspect
import weakref
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
    SparseRunnerAborted,
    SparseRunnerAccepted,
    SparseRunnerBudgetStopped,
    SparseRunnerExternallyStopped,
    SparseRunnerGeometryStopped,
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
    budget_ceiling: TwoPointBudgetCeiling | None = None,
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
        budget_ceiling or TwoPointBudgetCeiling(100, None, None, None),
        seed=17,
    )
    return runner, instrument


def _accept_steps(
    runner: SparseLinewidthEvaluatorRunner, count: int
) -> tuple[SparseRunnerAccepted, ...]:
    outcomes = tuple(runner.step() for _ in range(count))
    assert all(type(outcome) is SparseRunnerAccepted for outcome in outcomes)
    return outcomes  # type: ignore[return-value]


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

    monkeypatch.setattr(calibration_module, "_bind_run_token_success", commit_then_fail)

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
        runner.start_tracking(tracker, mismatched, success, metadata, budget, seed=0)
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
        object.__setattr__(source_runner.state, "calibration_outcome", replace(success))
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

    monkeypatch.setattr(SparseLinewidthCompositeTracker, "reset", reset_then_stop)

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
    runner, _ = _started_runner(monkeypatch, scan_period_fast_pairs=1, dynamics=spy)
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


def test_budget_stop_is_clean_before_query_and_preserves_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument = _started_runner(
        monkeypatch,
        budget_ceiling=TwoPointBudgetCeiling(2, None, None, None),
    )
    state_before = runner.state
    resources_before = instrument.resources

    outcome = runner.step()

    assert type(outcome) is SparseRunnerBudgetStopped
    assert outcome.state.phase == "budget_stopped"
    assert outcome.state.tracker_estimate is not None
    assert outcome.state.tracker_estimate.stopped_reason == "budget_exhausted"
    assert instrument.resources == resources_before
    assert outcome.state.normal_tracking_trace == state_before.normal_tracking_trace
    assert outcome.resources.unaccepted_observations == 0
    with pytest.raises(SparseRunnerStateError):
        runner.step()


def test_due_geometry_stop_is_clean_and_retains_exact_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.estimators import sparse_linewidth_tracker as tracker_module

    runner, instrument = _started_runner(monkeypatch)
    _accept_steps(runner, 16)
    resources_before = instrument.resources

    def fail_geometry(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise tracker_module._SparseGeometryConstructionError(
            "empty_fit_bounds",
            "injected due geometry",
            proposed_frequency_min_hz=1.0,
            proposed_frequency_max_hz=2.0,
        )

    monkeypatch.setattr(tracker_module, "_construct_sparse_fit_geometry", fail_geometry)

    outcome = runner.step()

    assert type(outcome) is SparseRunnerGeometryStopped
    assert outcome.state.phase == "geometry_stopped"
    assert outcome.state.tracker_estimate is not None
    assert (
        outcome.diagnostic is outcome.state.tracker_estimate.sparse_geometry_diagnostic
    )
    assert instrument.resources == resources_before
    assert outcome.resources.unaccepted_observations == 0


def test_external_stop_performs_no_query_and_preserves_partial_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument = _started_runner(monkeypatch)
    _accept_steps(runner, 1)
    state_before = runner.state
    resources_before = instrument.resources

    outcome = runner.stop_external()

    assert type(outcome) is SparseRunnerExternallyStopped
    assert outcome.state.phase == "externally_stopped"
    assert outcome.state.tracker_estimate is state_before.tracker_estimate
    assert outcome.state.normal_tracking_trace == state_before.normal_tracking_trace
    assert outcome.resources.incomplete_fast_pair_observations == 1
    assert instrument.resources == resources_before
    with pytest.raises(SparseRunnerStateError):
        runner.stop_external()


def test_external_stop_preserves_partial_sparse_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument = _started_runner(
        monkeypatch, scan_period_fast_pairs=1
    )
    _accept_steps(runner, 3)
    estimate_before = runner.state.tracker_estimate
    resources_before = instrument.resources
    assert estimate_before is not None
    assert estimate_before.incomplete_sparse_scan is not None
    assert estimate_before.pending_query is None

    outcome = runner.stop_external()

    assert outcome.state.tracker_estimate is estimate_before
    assert outcome.state.tracker_estimate.incomplete_sparse_scan is not None
    assert outcome.resources.incomplete_sparse_scan_observations == 1
    assert instrument.resources == resources_before


def test_run_until_event_continues_only_accepted_steps_to_budget_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = _started_runner(
        monkeypatch,
        budget_ceiling=TwoPointBudgetCeiling(4, None, None, None),
    )

    outcome = runner.run_until_event()

    assert type(outcome) is SparseRunnerBudgetStopped
    assert len(outcome.state.normal_tracking_trace) == 2


def test_run_until_event_returns_first_instrument_failure_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = _started_runner(monkeypatch)
    calls = 0

    def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal calls
        del args, kwargs
        calls += 1
        raise RuntimeError("instrument unavailable")

    monkeypatch.setattr(ODMRInstrument, "query", fail_once)

    outcome = runner.run_until_event()

    assert type(outcome) is SparseRunnerInstrumentFailure
    assert calls == 1
    assert outcome.state.phase == "tracking"
    assert outcome.state.tracker_estimate is not None
    assert outcome.state.tracker_estimate.pending_query == outcome.failure.query


@pytest.mark.parametrize(
    ("error_factory", "reason"),
    [
        (
            lambda: __import__(
                "odmr_bench.estimators.sparse_linewidth_types",
                fromlist=["SparseLinewidthObservationValidationError"],
            ).SparseLinewidthObservationValidationError(
                "endpoint_mismatch", "injected validation"
            ),
            "tracker_observation_validation_error",
        ),
        (
            lambda: __import__(
                "odmr_bench.estimators.sparse_linewidth_types",
                fromlist=["SparseLinewidthUpdateConstructionError"],
            ).SparseLinewidthUpdateConstructionError(
                "update_construction_failed", "injected construction"
            ),
            "tracker_update_construction_error",
        ),
        (
            lambda: RuntimeError("injected unexpected"),
            "tracker_update_unexpected_error",
        ),
    ],
)
def test_returned_authenticated_observation_aborts_with_one_uncharged_cpu_atom(
    monkeypatch: pytest.MonkeyPatch,
    error_factory,
    reason: str,
) -> None:
    runner, instrument = _started_runner(monkeypatch)
    state_before = runner.state
    estimate_before = state_before.tracker_estimate
    assert estimate_before is not None
    error = error_factory()

    def reject_update(self: object, observation: object) -> object:
        del self, observation
        raise error

    monkeypatch.setattr(SparseLinewidthCompositeTracker, "update", reject_update)

    outcome = runner.step()

    assert type(outcome) is SparseRunnerAborted
    assert outcome.abort.reason == reason
    assert outcome.abort.exception_type == type(error).__name__
    assert outcome.abort.exception_message == str(error)
    assert outcome.abort.tracker_estimate_before.pending_query is not None
    assert outcome.abort.tracker_estimate_after == outcome.abort.tracker_estimate_before
    assert outcome.resources is not None
    assert outcome.resources.unaccepted_observations == 1
    assert len(outcome.resources.unaccepted_tracking_observations) == 1
    assert outcome.resources.charged_resources.observations == (
        outcome.resources.accepted_charged_resources.observations + 1
    )
    assert outcome.state.total_update_cpu_time_s == state_before.total_update_cpu_time_s
    assert instrument.resources == outcome.state.instrument_resources_current


@pytest.mark.parametrize(
    "reserved_name",
    [
        "SparseLinewidthObservationValidationError",
        "SparseLinewidthUpdateConstructionError",
    ],
)
def test_foreign_reserved_exception_name_retains_unexpected_abort_evidence(
    reserved_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = _started_runner(monkeypatch)
    foreign_type = type(reserved_name, (RuntimeError,), {})
    error = foreign_type("foreign same-name exception")

    monkeypatch.setattr(
        SparseLinewidthCompositeTracker,
        "update",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    outcome = runner.step()

    assert type(outcome) is SparseRunnerAborted
    assert outcome.abort.reason == "tracker_update_unexpected_error"
    assert outcome.abort.exception_type == reserved_name
    assert outcome.abort.exception_message == str(error)


@pytest.mark.parametrize(
    ("base_name", "reason"),
    [
        (
            "SparseLinewidthObservationValidationError",
            "tracker_observation_validation_error",
        ),
        (
            "SparseLinewidthUpdateConstructionError",
            "tracker_update_construction_error",
        ),
    ],
)
def test_public_sparse_error_subclass_retains_documented_abort_classification(
    base_name: str,
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.estimators import sparse_linewidth_types as error_types

    runner, _ = _started_runner(monkeypatch)
    base = getattr(error_types, base_name)
    subclass = type(f"Derived{base_name}", (base,), {})
    code = (
        "endpoint_mismatch"
        if base_name == "SparseLinewidthObservationValidationError"
        else "update_construction_failed"
    )
    error = subclass(code, "derived public error")

    monkeypatch.setattr(
        SparseLinewidthCompositeTracker,
        "update",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    outcome = runner.step()

    assert type(outcome) is SparseRunnerAborted
    assert outcome.abort.reason == reason
    assert outcome.abort.exception_type == subclass.__name__
    assert outcome.abort.exception_message == str(error)


@pytest.mark.parametrize("failure_site", ["resources", "outcome"])
def test_terminal_construction_failure_does_not_overwrite_tracker_exception(
    failure_site: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        resource_accounting as resource_module,
    )
    from odmr_bench.evaluation.sparse_linewidth import runner as runner_module

    runner, _ = _started_runner(monkeypatch)
    original = RuntimeError("original tracker exception")
    construction = RuntimeError(f"injected terminal {failure_site} failure")
    original_resource_builder = (
        resource_module.build_sparse_linewidth_evaluator_resources
    )
    monkeypatch.setattr(
        SparseLinewidthCompositeTracker,
        "update",
        lambda *args, **kwargs: (_ for _ in ()).throw(original),
    )
    if failure_site == "resources":
        monkeypatch.setattr(
            resource_module,
            "build_sparse_linewidth_evaluator_resources",
            lambda *args, **kwargs: (_ for _ in ()).throw(construction),
        )
    else:
        monkeypatch.setattr(
            runner_module,
            "SparseRunnerAborted",
            lambda *args, **kwargs: (_ for _ in ()).throw(construction),
        )

    with pytest.raises(RuntimeError) as raised:
        runner.step()

    assert raised.value is construction
    assert runner.state.phase == "aborted"
    abort = runner.state.terminal_abort
    assert abort is not None
    assert abort.reason == "tracker_update_unexpected_error"
    assert abort.exception_type == "RuntimeError"
    assert abort.exception_message == str(original)
    if failure_site == "resources":
        monkeypatch.setattr(
            resource_module,
            "build_sparse_linewidth_evaluator_resources",
            original_resource_builder,
        )
    first = original_resource_builder(runner)
    second = original_resource_builder(runner)
    assert first is not None
    assert second == first


@pytest.mark.parametrize("abort_kind", ["authenticated", "unavailable"])
def test_discarded_aborted_runners_do_not_leave_global_causal_retention(
    abort_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point.provenance import (
        _rollback_run_token_registration,
    )

    original_query = ODMRInstrument.query
    corrupt_instruments: set[ODMRInstrument] = set()
    if abort_kind == "authenticated":
        monkeypatch.setattr(
            SparseLinewidthCompositeTracker,
            "update",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("discarded authenticated abort")
            ),
        )
    else:

        def query_then_corrupt_ledger(
            self: ODMRInstrument,
            frequency_hz: float,
            integration_time_s: float,
        ):
            observation = original_query(self, frequency_hz, integration_time_s)
            if self in corrupt_instruments:
                object.__setattr__(
                    self._ledger,
                    "_observations",
                    self._ledger._observations + 1,
                )
            return observation

        monkeypatch.setattr(ODMRInstrument, "query", query_then_corrupt_ledger)

    dynamics_refs: list[weakref.ReferenceType[QueryScopedDynamicsSpy]] = []
    for _ in range(4):
        dynamics = QueryScopedDynamicsSpy(StationaryDynamics(_snapshot()))
        runner, instrument = _started_runner(monkeypatch, dynamics=dynamics)
        corrupt_instruments.add(instrument)
        outcome = runner.step()
        assert type(outcome) is SparseRunnerAborted
        corrupt_instruments.discard(instrument)
        _rollback_run_token_registration(runner.state.run_token)
        dynamics_refs.append(weakref.ref(dynamics))
        del outcome, runner, instrument, dynamics

    gc.collect()

    assert all(reference() is None for reference in dynamics_refs)


@pytest.mark.parametrize(
    "fault_type",
    (RuntimeError, KeyboardInterrupt),
    ids=("exception", "baseexception"),
)
def test_sparse_bind_state_construction_fault_revokes_minted_token_and_reraises_identical_baseexception(  # noqa: E501
    fault_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.sparse_linewidth import runner as runner_module
    from odmr_bench.evaluation.two_point import provenance as provenance_module

    registries = (
        provenance_module._MINTED_RUN_TOKEN_IDENTITIES,
        provenance_module._RUN_TOKEN_BINDINGS,
        provenance_module._VERIFIED_CALIBRATION_ISSUERS,
        provenance_module._VERIFIED_CALIBRATION_ISSUER_REGISTRATIONS,
    )
    registry_snapshots = tuple(tuple(registry.items()) for registry in registries)
    captured_tokens: list[object] = []
    fault = fault_type("ready-state construction exploded")

    def fail_state_construction(*args: object, **kwargs: object) -> None:
        del args
        captured_tokens.append(kwargs["run_token"])
        raise fault

    monkeypatch.setattr(
        runner_module,
        "SparseEvaluatorRunnerState",
        fail_state_construction,
    )

    with pytest.raises(fault_type) as raised:
        SparseLinewidthEvaluatorRunner.bind(_instrument())

    assert raised.value is fault
    assert len(captured_tokens) == 1
    token = captured_tokens[0]
    assert tuple(tuple(registry.items()) for registry in registries) == (
        registry_snapshots
    )
    assert provenance_module._MINTED_RUN_TOKEN_IDENTITIES.get(id(token)) is not token
    assert provenance_module._lookup_run_token_binding(token) is None
    assert token not in provenance_module._VERIFIED_CALIBRATION_ISSUER_REGISTRATIONS


def test_unavailable_join_aborts_without_fabricated_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument = _started_runner(monkeypatch)
    original_query = ODMRInstrument.query

    def query_then_corrupt_ledger(
        self: ODMRInstrument, frequency_hz: float, integration_time_s: float
    ):
        observation = original_query(self, frequency_hz, integration_time_s)
        object.__setattr__(
            self._ledger, "_observations", self._ledger._observations + 1
        )
        return observation

    monkeypatch.setattr(ODMRInstrument, "query", query_then_corrupt_ledger)

    outcome = runner.step()

    assert type(outcome) is SparseRunnerAborted
    assert outcome.abort.reason == "resource_join_unavailable"
    assert outcome.abort.exception_type is None
    assert outcome.abort.exception_message is None
    assert outcome.resources is None
    assert outcome.abort.tracker_estimate_before == outcome.abort.tracker_estimate_after
    assert outcome.state.instrument_resources_current == instrument.resources


def test_returned_endpoint_mismatch_is_authenticated_validation_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument = _started_runner(monkeypatch)
    original_query = ODMRInstrument.query

    def query_with_wrong_returned_endpoint(
        self: ODMRInstrument, frequency_hz: float, integration_time_s: float
    ):
        observation = original_query(self, frequency_hz, integration_time_s)
        return replace(observation, timestamp_s=observation.timestamp_s + 1.0)

    monkeypatch.setattr(
        ODMRInstrument, "query", query_with_wrong_returned_endpoint
    )

    outcome = runner.step()

    assert type(outcome) is SparseRunnerAborted
    assert outcome.abort.reason == "tracker_observation_validation_error"
    assert outcome.abort.unaccepted_acquisition.measurement_midpoint_s is None
    assert outcome.resources is not None
    assert outcome.resources.unaccepted_observations == 1
    assert outcome.resources.charged_resources.observations == (
        outcome.resources.accepted_charged_resources.observations + 1
    )
    assert outcome.state.current_virtual_time_s == instrument.virtual_time_s


@pytest.mark.parametrize("echo_field", ["sequence_index", "frequency_hz"])
def test_returned_echo_mismatch_is_charged_validation_abort_with_timing_midpoint(
    echo_field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = _started_runner(monkeypatch)
    original_query = ODMRInstrument.query

    def query_with_wrong_echo(
        self: ODMRInstrument, frequency_hz: float, integration_time_s: float
    ):
        observation = original_query(self, frequency_hz, integration_time_s)
        replacement = (
            observation.sequence_index + 1
            if echo_field == "sequence_index"
            else observation.frequency_hz + 1.0
        )
        return replace(observation, **{echo_field: replacement})

    monkeypatch.setattr(ODMRInstrument, "query", query_with_wrong_echo)

    outcome = runner.step()

    assert type(outcome) is SparseRunnerAborted
    assert outcome.state.phase == "aborted"
    assert outcome.abort.reason == "tracker_observation_validation_error"
    acquisition = outcome.abort.unaccepted_acquisition
    assert (
        acquisition.measurement_midpoint_s
        == acquisition.expected_measurement_midpoint_s
    )
    assert outcome.resources is not None
    assert outcome.resources.unaccepted_observations == 1
    assert outcome.resources.charged_resources.observations == (
        outcome.resources.accepted_charged_resources.observations + 1
    )


@pytest.mark.parametrize(
    ("construction_site", "accepted_prefix"),
    [
        ("pair_timing", 1),
        ("scan_timing", 6),
        ("runner_state", 0),
        ("accepted_outcome", 0),
    ],
)
def test_ordinary_post_return_construction_fault_becomes_unexpected_abort(
    construction_site: str,
    accepted_prefix: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.sparse_linewidth import runner as runner_module

    runner, _ = _started_runner(monkeypatch, scan_period_fast_pairs=1)
    for _ in range(accepted_prefix):
        assert type(runner.step()) is SparseRunnerAccepted
    state_before = runner.state
    tracker = runner._tracker
    assert tracker is not None
    update_slots: list[tuple[object, object, object]] = []
    original_update = SparseLinewidthCompositeTracker.update

    def capture_update_boundary(
        self: SparseLinewidthCompositeTracker, observation: object
    ):
        update_slots.append(
            (self._configuration, self._configuration_snapshot, self._state)
        )
        return original_update(self, observation)  # type: ignore[arg-type]

    monkeypatch.setattr(
        SparseLinewidthCompositeTracker, "update", capture_update_boundary
    )
    error = RuntimeError(f"injected {construction_site}")

    if construction_site == "pair_timing":
        monkeypatch.setattr(
            runner_module,
            "TwoPointEvaluatorPairTiming",
            lambda *args, **kwargs: (_ for _ in ()).throw(error),
        )
    elif construction_site == "scan_timing":
        monkeypatch.setattr(
            runner_module,
            "SparseLinewidthEvaluatorScanTiming",
            lambda *args, **kwargs: (_ for _ in ()).throw(error),
        )
    elif construction_site == "runner_state":
        original_replace = runner_module.replace
        failed = False

        def fail_accepted_state(instance: object, **changes: object):
            nonlocal failed
            if not failed and "normal_tracking_trace" in changes:
                failed = True
                raise error
            return original_replace(instance, **changes)

        monkeypatch.setattr(runner_module, "replace", fail_accepted_state)
    else:
        monkeypatch.setattr(
            runner_module,
            "SparseRunnerAccepted",
            lambda *args, **kwargs: (_ for _ in ()).throw(error),
        )

    outcome = runner.step()

    assert type(outcome) is SparseRunnerAborted
    assert outcome.state.phase == "aborted"
    assert outcome.abort.reason == "tracker_update_unexpected_error"
    assert outcome.abort.exception_type == "RuntimeError"
    assert outcome.abort.exception_message == str(error)
    assert (
        outcome.abort.tracker_estimate_before
        == outcome.abort.tracker_estimate_after
    )
    assert outcome.state.fast_update_cpu_time_s == state_before.fast_update_cpu_time_s
    assert (
        outcome.state.sparse_update_cpu_time_s
        == state_before.sparse_update_cpu_time_s
    )
    assert (
        outcome.state.total_update_cpu_time_s
        == state_before.total_update_cpu_time_s
    )
    assert update_slots
    assert (
        tracker._configuration,
        tracker._configuration_snapshot,
        tracker._state,
    ) == update_slots[-1]
    assert outcome.resources is not None
    assert outcome.resources.unaccepted_observations == 1
    assert outcome.resources.charged_resources.observations == (
        outcome.resources.accepted_charged_resources.observations + 1
    )


@pytest.mark.parametrize(
    ("accepted_prefix", "scan_period_fast_pairs"),
    [(0, 8), (1, 8), (3, 1)],
)
def test_retryable_failure_resources_and_external_stop_preserve_causal_evidence(
    accepted_prefix: int,
    scan_period_fast_pairs: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.sparse_linewidth.resource_accounting import (
        build_sparse_linewidth_evaluator_resources,
    )

    runner, _ = _started_runner(
        monkeypatch, scan_period_fast_pairs=scan_period_fast_pairs
    )
    for _ in range(accepted_prefix):
        assert type(runner.step()) is SparseRunnerAccepted

    def fail_query(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("instrument unavailable")

    monkeypatch.setattr(ODMRInstrument, "query", fail_query)
    failed = runner.step()
    assert type(failed) is SparseRunnerInstrumentFailure
    failure = failed.failure
    pending = failed.state.tracker_estimate.pending_query
    resources_before_stop = build_sparse_linewidth_evaluator_resources(runner)
    assert resources_before_stop is not None
    assert resources_before_stop.unaccepted_observations == 0
    assert resources_before_stop.charged_resources == (
        resources_before_stop.accepted_charged_resources
    )

    stopped = runner.stop_external()

    assert stopped.state.phase == "externally_stopped"
    assert stopped.state.last_instrument_failure is failure
    assert stopped.state.tracker_estimate.pending_query == pending
    assert stopped.resources == resources_before_stop
    assert stopped.state.fast_update_cpu_time_s == failed.state.fast_update_cpu_time_s
    assert (
        stopped.state.sparse_update_cpu_time_s
        == failed.state.sparse_update_cpu_time_s
    )
    assert (
        stopped.state.total_update_cpu_time_s
        == failed.state.total_update_cpu_time_s
    )


def test_returned_observation_process_control_restores_tracker_and_reraises_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopNow(BaseException):
        pass

    runner, instrument = _started_runner(monkeypatch)
    tracker = runner._tracker
    assert tracker is not None
    state_before = runner.state
    tracker_slots_before = (
        tracker._configuration,
        tracker._configuration_snapshot,
        tracker._state,
    )
    stop = StopNow()

    def mutate_then_stop(self: SparseLinewidthCompositeTracker, observation: object):
        del observation
        object.__setattr__(self, "_state", object())
        raise stop

    monkeypatch.setattr(SparseLinewidthCompositeTracker, "update", mutate_then_stop)

    with pytest.raises(StopNow) as raised:
        runner.step()

    assert raised.value is stop
    assert runner.state is state_before
    assert (
        tracker._configuration,
        tracker._configuration_snapshot,
        tracker._state,
    ) == tracker_slots_before
    assert instrument.resources.observations == (
        state_before.instrument_resources_current.observations + 1
    )


@pytest.mark.parametrize(
    "phase",
    [
        "ready",
        "calibration_succeeded",
        "calibration_failed",
        "tracking",
        "budget_stopped",
        "geometry_stopped",
        "externally_stopped",
        "aborted",
    ],
)
def test_all_eight_phases_reject_every_illegal_operation_before_side_effects(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if phase == "ready":
        runner = SparseLinewidthEvaluatorRunner.bind(_instrument())
    else:
        runner, _ = _started_runner(monkeypatch)
        object.__setattr__(runner.state, "phase", phase)
    state_before = runner.state
    instrument = runner._instrument
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s

    if phase != "ready":
        with pytest.raises(SparsePreflightError) as raised:
            runner.acquire_verified_calibration(  # type: ignore[arg-type]
                **_calibration_arguments()
            )
        assert raised.value.code == "invalid_runner_phase"
    if phase not in {"ready", "calibration_succeeded"}:
        with pytest.raises(SparseStartError) as raised:
            runner.start_tracking(  # type: ignore[arg-type]
                object(), object(), object(), object(), object(), seed=0
            )
        assert raised.value.code == "invalid_runner_phase"
    if phase != "tracking":
        for operation in (
            runner.step,
            runner.run_until_event,
            runner.stop_external,
        ):
            with pytest.raises(SparseRunnerStateError):
                operation()

    assert runner.state is state_before
    assert instrument.resources == resources_before
    assert instrument.virtual_time_s == time_before
