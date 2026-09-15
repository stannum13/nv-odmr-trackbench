"""Contract tests for the sparse-linewidth evaluator runner shell."""

from __future__ import annotations

import inspect

import pytest

from odmr_bench.dynamics import SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import GaussianNoise
from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.evaluation.sparse_linewidth import (
    SparseEvaluatorRunnerState,
    SparsePreflightError,
    SparseRunnerStateError,
    SparseStartError,
)
from odmr_bench.evaluation.sparse_linewidth.runner import (
    SparseLinewidthEvaluatorRunner,
)
from odmr_bench.evaluation.two_point.types import (
    TwoPointEvaluatorInstrumentConfiguration,
)
from odmr_bench.models import Baseline, Resonance


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


def test_sparse_runner_shell_rejects_operations_not_implemented_in_current_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrument = _instrument()
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)

    def reject_query(*args: object, **kwargs: object) -> object:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(ODMRInstrument, "query", reject_query)
    state_before = runner.state
    with pytest.raises(SparsePreflightError) as acquire_error:
        runner.acquire_verified_calibration(  # type: ignore[arg-type]
            (),
            0.0,
            object(),
            object(),
            source_id="",
            source_clock_id="",
            tracker_clock_id="",
            source_to_tracker_offset_s=0.0,
            physical_fit_epoch_rule="instrument_midpoint_ordered_mean",
        )
    assert acquire_error.value.code == "invalid_runner_phase"
    with pytest.raises(SparseStartError) as start_error:
        runner.start_tracking(  # type: ignore[arg-type]
            object(), object(), object(), object(), object(), seed=0
        )
    assert start_error.value.code == "invalid_runner_phase"
    with pytest.raises(SparseRunnerStateError):
        runner.step()
    with pytest.raises(SparseRunnerStateError):
        runner.run_until_event()
    with pytest.raises(SparseRunnerStateError):
        runner.stop_external()
    assert runner.state is state_before
    assert instrument.resources == state_before.instrument_resources_current
