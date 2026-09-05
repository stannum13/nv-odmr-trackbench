"""Tests for evaluator-private full two-point resource accounting."""

from __future__ import annotations

import ast
import copy
import math
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path

import pytest

from odmr_bench.dynamics import SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import GaussianNoise
from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.noise import NoiseResult
from odmr_bench.emulator.observations import InstrumentObservation
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
from odmr_bench.estimators.two_point_types import (
    CalibrationBudgetTreatment,
    PublicAcquisitionResources,
    TwoPointCalibrationSource,
)
from odmr_bench.evaluation import two_point
from odmr_bench.evaluation.two_point import (
    TwoPointAbortedRun,
    TwoPointEvaluatorResources,
    TwoPointEvaluatorRunner,
    TwoPointResourceJoinUnavailableAcquisition,
    TwoPointRunnerAccepted,
    VerifiedInstrumentRunToken,
    VerifiedTwoPointCalibrationSuccess,
    build_two_point_evaluator_resources,
)
from odmr_bench.evaluation.two_point.resource_accounting import (
    _advance_full_resources,
    _build_two_point_evaluator_resources_from_context,
    _project_full_resources,
    _replay_full_resources,
    _resource_mismatch_fields,
    _zero_full_resources,
)
from odmr_bench.evaluation.two_point.runner import (
    _build_authenticated_tracking_acquisition,
)
from odmr_bench.models import Baseline, Resonance
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_fit_configuration,
    make_legal_source_fit,
)


class _MixedCountNoise:
    sampling_rule = "task-16-mixed-counts"

    def __init__(self, zero_indices: frozenset[int]) -> None:
        self.index = 0
        self.zero_indices = zero_indices

    def checkpoint(self) -> int:
        return self.index

    def restore(self, checkpoint: object) -> None:
        self.index = int(checkpoint)

    def sample(
        self,
        expected_fluorescence: float,
        nominal_rate_hz: float,
        integration_time_s: float,
        rng: object,
    ) -> NoiseResult:
        del nominal_rate_hz, integration_time_s, rng
        index = self.index
        self.index += 1
        return NoiseResult(
            fluorescence=(
                0.0 if index in self.zero_indices else expected_fluorescence
            ),
            realized_photons=None if index % 2 == 0 else 100 + index,
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
    noise: object | None = None,
) -> ODMRInstrument:
    return ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=GaussianNoise(stddev_at_1s=0.0) if noise is None else noise,
        nominal_photon_rate_hz=nominal_photon_rate_hz,
        frequency_overhead_s=frequency_overhead_s,
        seed=16,
    )


def _verified_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_clock_id: str,
    tracker_clock_id: str,
    source_to_tracker_offset_s: float,
    noise: object | None = None,
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
    instrument = _instrument(noise=noise)
    runner = TwoPointEvaluatorRunner.bind(instrument)
    outcome = runner.acquire_verified_calibration(
        (2.74e9, 3.02e9),
        0.005,
        fit_configuration,
        TwoPointIdentityBinding(
            "require_expected_ids", fit_configuration.resonance_ids
        ),
        source_id="task-16-source",
        source_clock_id=source_clock_id,
        tracker_clock_id=tracker_clock_id,
        source_to_tracker_offset_s=source_to_tracker_offset_s,
        physical_fit_epoch_rule="instrument_midpoint_ordered_mean",
    )
    assert type(outcome) is VerifiedTwoPointCalibrationSuccess
    return runner, instrument, outcome


def _started_runner(
    monkeypatch: pytest.MonkeyPatch,
    treatment: CalibrationBudgetTreatment,
    *,
    tracking_noise: object | None = None,
) -> tuple[
    TwoPointEvaluatorRunner,
    ODMRInstrument,
    CalibratedTwoPointTracker,
    TwoPointCalibration,
    VerifiedTwoPointCalibrationSuccess,
]:
    included = treatment == "included_same_run"
    source_runner, source_instrument, success = _verified_success(
        monkeypatch,
        source_clock_id="shared-clock" if included else "source-clock",
        tracker_clock_id="shared-clock" if included else "tracking-clock",
        source_to_tracker_offset_s=0.0 if included else -0.012,
        noise=tracking_noise if included else None,
    )
    configuration = TwoPointTrackerConfiguration()
    calibration = calibrate_two_point(
        success.source,
        configuration,
        budget_treatment=treatment,
    )
    tracker = CalibratedTwoPointTracker(configuration)
    runner = (
        source_runner
        if included
        else TwoPointEvaluatorRunner.bind(_instrument(noise=tracking_noise))
    )
    instrument = source_instrument if included else runner._instrument
    metadata = TwoPointRunMetadata(
        tracker_clock_id="shared-clock" if included else "tracking-clock",
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
        seed=20260904,
    )
    return runner, instrument, tracker, calibration, success


def _changed_resource_field(
    resources: ResourceSnapshot,
    field_name: str,
) -> ResourceSnapshot:
    value = getattr(resources, field_name)
    if field_name == "observations_without_realized_counts":
        changed: float | int = value - 1 if value else 1
    elif isinstance(value, int):
        changed = value + 1
    else:
        changed = math.nextafter(value, math.inf)
    return replace(resources, **{field_name: changed})


def _state_copy_with(state: object, **changes: object) -> object:
    copied = copy.copy(state)
    for name, value in changes.items():
        object.__setattr__(copied, name, value)
    return copied


def _instrument_observation(
    *,
    sequence_index: int,
    integration_time_s: float,
    nominal_exposure_photons: float,
    expected_photons: float,
    realized_photons: int | None,
) -> InstrumentObservation:
    return InstrumentObservation(
        sequence_index=sequence_index,
        timestamp_s=(sequence_index + 1) * integration_time_s,
        frequency_hz=2.80e9 + sequence_index * 1.0e6,
        fluorescence=0.98,
        integration_time_s=integration_time_s,
        nominal_exposure_photons=nominal_exposure_photons,
        expected_photons=expected_photons,
        realized_photons=realized_photons,
        sampling_rule="test-rule",
    )


def test_full_resource_helpers_are_evaluator_private_and_atomic() -> None:
    observations = tuple(
        _instrument_observation(
            sequence_index=index,
            integration_time_s=0.005,
            nominal_exposure_photons=12_500.0 + index,
            expected_photons=12_000.0 + index,
            realized_photons=realized,
        )
        for index, realized in enumerate((None, 3, None, 5, 7, None))
    )

    zero = _zero_full_resources()
    assert zero == ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    first = _advance_full_resources(zero, observations[0], 0.001)
    assert first == ResourceSnapshot(
        observations=1,
        integration_time_s=0.005,
        nominal_exposure_photons=12_500.0,
        expected_photons=12_000.0,
        realized_photons=0,
        observations_without_realized_counts=1,
        virtual_elapsed_time_s=0.006,
    )

    total = zero
    for observation in observations:
        total = _advance_full_resources(total, observation, 0.001)
    assert total == _replay_full_resources(observations, 0.001)
    assert total == ResourceSnapshot(
        observations=6,
        integration_time_s=total.integration_time_s,
        nominal_exposure_photons=75_015.0,
        expected_photons=72_015.0,
        realized_photons=15,
        observations_without_realized_counts=3,
        virtual_elapsed_time_s=total.virtual_elapsed_time_s,
    )
    assert total.integration_time_s.hex() == "0x1.eb851eb851eb9p-6"
    assert total.integration_time_s != math.fsum([0.005] * 6)
    assert total.virtual_elapsed_time_s.hex() == "0x1.26e978d4fdf3bp-5"

    projected = _project_full_resources(total)
    assert projected == PublicAcquisitionResources(
        observations=total.observations,
        integration_time_s=total.integration_time_s,
        nominal_exposure_photons=total.nominal_exposure_photons,
        realized_photons=total.realized_photons,
        observations_without_realized_counts=(
            total.observations_without_realized_counts
        ),
        virtual_elapsed_time_s=total.virtual_elapsed_time_s,
    )
    assert projected.realized_photons == total.realized_photons
    assert not hasattr(projected, "expected_photons")

    private_helper_names = (
        "_zero_full_resources",
        "_advance_full_resources",
        "_replay_full_resources",
        "_project_full_resources",
        "_resource_mismatch_fields",
    )
    assert not any(hasattr(two_point, name) for name in private_helper_names)


def _estimator_resource_boundary_violations(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    forbidden_identifiers = {
        "ResourceSnapshot",
        "expected_photons",
        "_zero_full_resources",
        "_advance_full_resources",
        "_replay_full_resources",
        "_project_full_resources",
        "_resource_mismatch_fields",
    }
    forbidden_modules = {
        "odmr_bench.emulator.resources",
        "odmr_bench.evaluation.two_point.resource_accounting",
    }
    violations: set[str] = set()

    def check_identifier(identifier: str, location: str) -> None:
        if identifier in forbidden_identifiers or "full_resources" in identifier:
            violations.add(f"{location}:{identifier}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            check_identifier(node.id, "name")
        elif isinstance(node, ast.Attribute):
            check_identifier(node.attr, "attribute")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            check_identifier(node.name, "definition")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    violations.add(f"module:{alias.name}")
                check_identifier(alias.name.rsplit(".", 1)[-1], "import")
                if alias.asname is not None:
                    check_identifier(alias.asname, "import-alias")
        elif isinstance(node, ast.ImportFrom):
            if node.module in forbidden_modules:
                violations.add(f"module:{node.module}")
            for alias in node.names:
                check_identifier(alias.name, "from-import")
                if alias.asname is not None:
                    check_identifier(alias.asname, "import-alias")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            check_identifier(node.value, "string-reference")
    return tuple(sorted(violations))


def test_full_resource_boundary_is_absent_from_every_estimator_module() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    estimator_root = repository_root / "src/odmr_bench/estimators"
    estimator_paths = tuple(sorted(estimator_root.rglob("*.py")))

    assert estimator_paths
    violations = {
        path.relative_to(repository_root).as_posix(): found
        for path in estimator_paths
        if (found := _estimator_resource_boundary_violations(path))
    }
    assert violations == {}


def test_full_resource_replay_preserves_order_sensitive_binary64_arrivals() -> (
    None
):
    observations = tuple(
        _instrument_observation(
            sequence_index=index,
            integration_time_s=0.005,
            nominal_exposure_photons=nominal,
            expected_photons=expected,
            realized_photons=index + 1,
        )
        for index, (nominal, expected) in enumerate(
            (
                (float(2**53), float(2**52)),
                (1.0, 0.5),
                (1.0, 0.5),
            )
        )
    )

    forward = _zero_full_resources()
    for observation in observations:
        forward = _advance_full_resources(forward, observation, 0.001)
    reverse = _zero_full_resources()
    for observation in reversed(observations):
        reverse = _advance_full_resources(reverse, observation, 0.001)

    assert forward.nominal_exposure_photons.hex() == "0x1.0000000000000p+53"
    assert reverse.nominal_exposure_photons.hex() == "0x1.0000000000001p+53"
    assert forward.expected_photons.hex() == "0x1.0000000000000p+52"
    assert reverse.expected_photons.hex() == "0x1.0000000000001p+52"
    assert forward != reverse
    assert _replay_full_resources(observations, 0.001) == forward


def test_resource_mismatch_fields_are_exact_complete_and_declaration_ordered() -> (
    None
):
    expected = ResourceSnapshot(
        observations=2,
        integration_time_s=0.03,
        nominal_exposure_photons=25_000.0,
        expected_photons=24_000.0,
        realized_photons=23_975,
        observations_without_realized_counts=1,
        virtual_elapsed_time_s=0.04,
    )
    field_names = tuple(field.name for field in fields(ResourceSnapshot))
    mutations: dict[str, float | int] = {
        "observations": 3,
        "integration_time_s": math.nextafter(
            expected.integration_time_s, math.inf
        ),
        "nominal_exposure_photons": math.nextafter(
            expected.nominal_exposure_photons, math.inf
        ),
        "expected_photons": math.nextafter(expected.expected_photons, math.inf),
        "realized_photons": expected.realized_photons + 1,
        "observations_without_realized_counts": 2,
        "virtual_elapsed_time_s": math.nextafter(
            expected.virtual_elapsed_time_s, math.inf
        ),
    }

    assert field_names == (
        "observations",
        "integration_time_s",
        "nominal_exposure_photons",
        "expected_photons",
        "realized_photons",
        "observations_without_realized_counts",
        "virtual_elapsed_time_s",
    )
    assert _resource_mismatch_fields(expected, expected) == ()
    for field_name, changed_value in mutations.items():
        actual = replace(expected, **{field_name: changed_value})
        assert _resource_mismatch_fields(expected, actual) == (field_name,)

    actual = replace(expected, **mutations)
    assert _resource_mismatch_fields(expected, actual) == field_names


def test_public_builder_zero_trace_and_accepted_trace_join_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, _, calibration, success = _started_runner(
        monkeypatch, "included_same_run"
    )

    zero_trace = build_two_point_evaluator_resources(runner)

    assert type(zero_trace) is TwoPointEvaluatorResources
    zero = _zero_full_resources()
    assert zero_trace.calibration_observations == success.full_observations
    assert zero_trace.accepted_tracking_observations == ()
    assert zero_trace.unaccepted_tracking_observations == ()
    assert zero_trace.calibration_resources == success.full_resources
    assert zero_trace.accepted_tracking_resources == zero
    assert zero_trace.unaccepted_tracking_resources == zero
    assert zero_trace.tracking_resources == zero
    assert zero_trace.accepted_charged_resources == success.full_resources
    assert zero_trace.charged_resources == success.full_resources
    assert (
        zero_trace.calibration_budget_treatment
        == calibration.budget_treatment
        == "included_same_run"
    )
    assert zero_trace.incomplete_pair_observations == 0
    assert zero_trace.unaccepted_observations == 0

    accepted = runner.step()
    assert type(accepted) is TwoPointRunnerAccepted
    accepted_trace = build_two_point_evaluator_resources(runner)

    assert type(accepted_trace) is TwoPointEvaluatorResources
    full_observation = accepted.acquisition.full_observation
    expected_tracking = _replay_full_resources(
        (full_observation,), instrument.frequency_overhead_s
    )
    expected_charged = _replay_full_resources(
        (*success.full_observations, full_observation),
        instrument.frequency_overhead_s,
    )
    assert accepted_trace.calibration_observations == success.full_observations
    assert accepted_trace.accepted_tracking_observations == (full_observation,)
    assert accepted_trace.unaccepted_tracking_observations == ()
    assert accepted_trace.calibration_resources == success.full_resources
    assert accepted_trace.accepted_tracking_resources == expected_tracking
    assert accepted_trace.unaccepted_tracking_resources == zero
    assert accepted_trace.tracking_resources == expected_tracking
    assert accepted_trace.accepted_charged_resources == expected_charged
    assert accepted_trace.charged_resources == expected_charged
    assert _project_full_resources(expected_tracking) == (
        runner.state.tracker_estimate.tracking_resources
    )
    assert _project_full_resources(expected_charged) == (
        runner.state.tracker_estimate.charged_resources
    )
    assert accepted_trace.incomplete_pair_observations == 1
    assert accepted_trace.unaccepted_observations == 0


def test_builder_rejects_nonoriginal_or_inconsistent_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, tracker, calibration, success = _started_runner(
        monkeypatch, "conditional_free_precalibration"
    )
    outcomes = tuple(runner.step() for _ in range(4))
    assert all(type(outcome) is TwoPointRunnerAccepted for outcome in outcomes)
    original_state = runner.state
    original_instrument = runner._instrument
    trace = original_state.normal_tracking_trace
    resource_fields = tuple(field.name for field in fields(ResourceSnapshot))

    def rejects_state(label: str, state: object) -> None:
        object.__setattr__(runner, "_state", state)
        try:
            with pytest.raises((TypeError, ValueError), match=r"runner|context"):
                build_two_point_evaluator_resources(runner)
        finally:
            object.__setattr__(runner, "_state", original_state)

    copied_runner = copy.copy(runner)
    with pytest.raises(ValueError, match=r"runner|context"):
        build_two_point_evaluator_resources(copied_runner)

    caller_source = make_legal_caller_asserted_source()
    assert type(caller_source) is TwoPointCalibrationSource
    caller_outcome = copy.copy(success)
    object.__setattr__(caller_outcome, "source", caller_source)
    caller_calibration = copy.copy(calibration)
    object.__setattr__(caller_calibration, "source", caller_source)
    rejects_state(
        "caller source",
        _state_copy_with(
            original_state,
            verified_calibration=caller_outcome,
            calibration=caller_calibration,
        ),
    )

    copied_outcome = replace(success)
    rejects_state(
        "copied outcome",
        _state_copy_with(original_state, verified_calibration=copied_outcome),
    )

    copied_token = object.__new__(VerifiedInstrumentRunToken)
    rejects_state(
        "copied token",
        _state_copy_with(original_state, run_token=copied_token),
    )

    copied_source = copy.copy(success.source)
    copied_source_outcome = copy.copy(success)
    object.__setattr__(copied_source_outcome, "source", copied_source)
    copied_source_calibration = copy.copy(calibration)
    object.__setattr__(copied_source_calibration, "source", copied_source)
    rejects_state(
        "copied source",
        _state_copy_with(
            original_state,
            verified_calibration=copied_source_outcome,
            calibration=copied_source_calibration,
        ),
    )

    copied_instrument = object.__new__(ODMRInstrument)
    for slot in ODMRInstrument.__slots__:
        object.__setattr__(copied_instrument, slot, getattr(instrument, slot))
    object.__setattr__(runner, "_instrument", copied_instrument)
    try:
        with pytest.raises(ValueError, match=r"runner|context"):
            build_two_point_evaluator_resources(runner)
    finally:
        object.__setattr__(runner, "_instrument", original_instrument)

    trace_mutations = {
        "extra observation": (*trace, trace[-1]),
        "missing observation": trace[:-1],
        "duplicate observation": (trace[0], trace[0], *trace[2:]),
        "reordered observation": (trace[1], trace[0], *trace[2:]),
    }
    mismatched = copy.copy(trace[0])
    object.__setattr__(mismatched, "safe_observation", trace[1].safe_observation)
    trace_mutations["safe-view-mismatched observation"] = (
        mismatched,
        *trace[1:],
    )
    for label, changed_trace in trace_mutations.items():
        rejects_state(
            label,
            _state_copy_with(
                original_state, normal_tracking_trace=changed_trace
            ),
        )

    for field_name in resource_fields:
        rejects_state(
            f"final boundary {field_name}",
            _state_copy_with(
                original_state,
                instrument_resources_current=_changed_resource_field(
                    original_state.instrument_resources_current, field_name
                ),
            ),
        )

    rejects_state(
        "tracking boundary",
        _state_copy_with(
            original_state,
            tracking_resources_before=_changed_resource_field(
                original_state.tracking_resources_before, "expected_photons"
            ),
        ),
    )
    rejects_state(
        "bind boundary",
        _state_copy_with(
            original_state,
            instrument_resources_at_bind=_changed_resource_field(
                original_state.instrument_resources_at_bind, "observations"
            ),
        ),
    )
    changed_acquisition = copy.copy(trace[1])
    object.__setattr__(
        changed_acquisition,
        "instrument_resources_before",
        _changed_resource_field(
            changed_acquisition.instrument_resources_before,
            "integration_time_s",
        ),
    )
    rejects_state(
        "acquisition boundary",
        _state_copy_with(
            original_state,
            normal_tracking_trace=(trace[0], changed_acquisition, *trace[2:]),
        ),
    )
    rejects_state(
        "sequence boundary",
        _state_copy_with(
            original_state,
            instrument_current_sequence_index=(
                original_state.instrument_current_sequence_index + 1
            ),
        ),
    )
    rejects_state(
        "time boundary",
        _state_copy_with(
            original_state,
            current_virtual_time_s=math.nextafter(
                original_state.current_virtual_time_s, math.inf
            ),
        ),
    )
    assert runner.state is original_state
    assert runner._instrument is original_instrument
    assert tracker.estimate() is original_state.tracker_estimate


def test_accepted_charged_prefix_uses_arrival_atoms_in_both_treatments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for treatment in (
        "included_same_run",
        "conditional_free_precalibration",
    ):
        source_prefix = 2 if treatment == "included_same_run" else 0
        noise = _MixedCountNoise(
            frozenset({source_prefix + 2, source_prefix + 3})
        )
        runner, instrument, _, _, success = _started_runner(
            monkeypatch,
            treatment,
            tracking_noise=noise,
        )
        outcomes = tuple(runner.step() for _ in range(5))
        assert all(type(outcome) is TwoPointRunnerAccepted for outcome in outcomes)
        assert outcomes[3].update.completed_pair is not None
        assert outcomes[3].update.completed_pair.lock_state == "lost"
        assert (
            outcomes[3].update.completed_pair.failure_code
            == "invalid_pair_normalization"
        )
        assert runner.state.tracker_estimate.incomplete_pair is not None

        original_query = ODMRInstrument.query

        def fail_pending_query(
            self: ODMRInstrument,
            frequency_hz: float,
            integration_time_s: float,
            *,
            target_instrument: ODMRInstrument = instrument,
            retained_query: Callable[
                [ODMRInstrument, float, float], InstrumentObservation
            ] = original_query,
        ) -> InstrumentObservation:
            if self is target_instrument:
                raise RuntimeError("retain pending second side")
            return retained_query(self, frequency_hz, integration_time_s)

        with monkeypatch.context() as query_patch:
            query_patch.setattr(ODMRInstrument, "query", fail_pending_query)
            failure = runner.step()
        assert failure.kind == "instrument_failure"
        assert runner.state.tracker_estimate.pending_query is failure.failure.query

        resources = build_two_point_evaluator_resources(runner)

        assert type(resources) is TwoPointEvaluatorResources
        accepted = tuple(
            outcome.acquisition.full_observation for outcome in outcomes
        )
        expected_tracking = _zero_full_resources()
        for observation in accepted:
            expected_tracking = _advance_full_resources(
                expected_tracking,
                observation,
                instrument.frequency_overhead_s,
            )
        expected_charged = (
            _replay_full_resources(
                success.full_observations,
                success.source.source_frequency_overhead_s,
            )
            if treatment == "included_same_run"
            else _zero_full_resources()
        )
        for observation in accepted:
            expected_charged = _advance_full_resources(
                expected_charged,
                observation,
                instrument.frequency_overhead_s,
            )

        assert resources.accepted_tracking_observations == accepted
        assert resources.accepted_tracking_resources == expected_tracking
        assert resources.tracking_resources == expected_tracking
        assert resources.accepted_charged_resources == expected_charged
        assert resources.charged_resources == expected_charged
        assert _project_full_resources(expected_tracking) == (
            runner.state.tracker_estimate.tracking_resources
        )
        assert _project_full_resources(expected_charged) == (
            runner.state.tracker_estimate.charged_resources
        )
        assert expected_tracking.observations == 5
        expected_realized = 0
        for observation in accepted:
            expected_realized += observation.realized_photons or 0
        assert expected_tracking.realized_photons == expected_realized
        assert expected_tracking.observations_without_realized_counts == 3
        assert resources.incomplete_pair_observations == 1
        assert resources.unaccepted_observations == 0
        assert resources.unaccepted_tracking_observations == ()
        assert failure.failure.instrument_resources_before == (
            resources.accepted_charged_resources
            if treatment == "conditional_free_precalibration"
            else resources.charged_resources
        )


@pytest.mark.parametrize("abort_side", ["first", "second"])
def test_private_resource_assembly_continues_with_one_authenticated_unaccepted_atom(
    monkeypatch: pytest.MonkeyPatch,
    abort_side: str,
) -> None:
    runner, instrument, tracker, _, _ = _started_runner(
        monkeypatch, "conditional_free_precalibration"
    )
    accepted_outcomes: tuple[TwoPointRunnerAccepted, ...] = ()
    if abort_side == "second":
        first = runner.step()
        assert type(first) is TwoPointRunnerAccepted
        accepted_outcomes = (first,)

    query = tracker.choose_next_query()
    assert query is not None
    pending_estimate = tracker.estimate()
    assert pending_estimate.pending_query is query
    state_with_pending = replace(runner.state, tracker_estimate=pending_estimate)
    object.__setattr__(runner, "_state", state_with_pending)
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s
    expected_midpoint_s = (
        time_before + instrument.frequency_overhead_s
    ) + query.integration_time_s / 2.0
    full_observation = instrument.query(
        query.frequency_hz, query.integration_time_s
    )
    unaccepted = _build_authenticated_tracking_acquisition(
        query=query,
        expected_midpoint_s=expected_midpoint_s,
        full_observation=full_observation,
        resources_before=resources_before,
        resources_after=instrument.resources,
        overhead_s=instrument.frequency_overhead_s,
    )

    resources = _build_two_point_evaluator_resources_from_context(
        runner,
        authenticated_unaccepted=unaccepted,
    )

    accepted = tuple(
        outcome.acquisition.full_observation for outcome in accepted_outcomes
    )
    expected_accepted = _replay_full_resources(
        accepted, instrument.frequency_overhead_s
    )
    expected_unaccepted = _advance_full_resources(
        _zero_full_resources(),
        full_observation,
        instrument.frequency_overhead_s,
    )
    expected_tracking = expected_accepted
    for observation in (full_observation,):
        expected_tracking = _advance_full_resources(
            expected_tracking,
            observation,
            instrument.frequency_overhead_s,
        )
    expected_charged = resources.accepted_charged_resources
    for observation in (full_observation,):
        expected_charged = _advance_full_resources(
            expected_charged,
            observation,
            instrument.frequency_overhead_s,
        )

    assert resources.accepted_tracking_observations == accepted
    assert resources.unaccepted_tracking_observations == (full_observation,)
    assert resources.accepted_tracking_resources == expected_accepted
    assert resources.unaccepted_tracking_resources == expected_unaccepted
    assert resources.tracking_resources == expected_tracking
    assert resources.accepted_charged_resources == expected_accepted
    assert _project_full_resources(resources.accepted_charged_resources) == (
        pending_estimate.charged_resources
    )
    assert resources.charged_resources == expected_charged
    assert resources.charged_resources == instrument.resources
    assert resources.incomplete_pair_observations == int(abort_side == "second")
    assert resources.unaccepted_observations == 1
    assert len(resources.accepted_tracking_observations) == int(
        abort_side == "second"
    )


def test_public_builder_returns_none_for_unavailable_abort_without_fabrication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import (
        resource_accounting as resource_module,
    )

    runner, instrument, tracker, _, _ = _started_runner(
        monkeypatch, "conditional_free_precalibration"
    )
    query = tracker.choose_next_query()
    assert query is not None
    pending_estimate = tracker.estimate()
    pending_state = replace(runner.state, tracker_estimate=pending_estimate)
    object.__setattr__(runner, "_state", pending_state)
    resources_before = instrument.resources
    time_before = instrument.virtual_time_s
    expected_midpoint_s = (
        time_before + instrument.frequency_overhead_s
    ) + query.integration_time_s / 2.0
    returned_observation = instrument.query(
        query.frequency_hz, query.integration_time_s
    )
    raw_observation = replace(
        returned_observation,
        expected_photons=math.nextafter(
            returned_observation.expected_photons, math.inf
        ),
    )
    prospective = _advance_full_resources(
        resources_before,
        raw_observation,
        instrument.frequency_overhead_s,
    )
    mismatch_fields = _resource_mismatch_fields(
        prospective, instrument.resources
    )
    assert mismatch_fields == ("expected_photons",)
    unavailable = TwoPointResourceJoinUnavailableAcquisition(
        resource_join_status="unavailable",
        query=query,
        expected_measurement_midpoint_s=expected_midpoint_s,
        measurement_midpoint_s=expected_midpoint_s,
        full_observation=raw_observation,
        safe_observation=raw_observation.estimator_view(),
        resource_mismatch_fields=mismatch_fields,
        instrument_resources_before=resources_before,
        instrument_resources_after=instrument.resources,
    )
    abort = TwoPointAbortedRun(
        reason="resource_join_unavailable",
        exception_type=None,
        exception_message=None,
        unaccepted_acquisition=unavailable,
        unaccepted_observation_count=1,
        tracker_estimate_before=pending_estimate,
        tracker_estimate_after=pending_estimate,
    )
    aborted_state = replace(
        pending_state,
        phase="aborted",
        instrument_resources_current=instrument.resources,
        instrument_current_sequence_index=returned_observation.sequence_index,
        current_virtual_time_s=instrument.virtual_time_s,
        terminal_abort=abort,
    )
    object.__setattr__(runner, "_state", aborted_state)

    original_advance = resource_module._advance_full_resources
    raw_advance_starts: list[ResourceSnapshot] = []

    def advance_spy(
        resources: ResourceSnapshot,
        observation: InstrumentObservation,
        overhead_s: float,
    ) -> ResourceSnapshot:
        if observation is raw_observation:
            raw_advance_starts.append(resources)
        return original_advance(resources, observation, overhead_s)

    def reject_aggregate(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("unavailable abort must not fabricate an aggregate")

    monkeypatch.setattr(resource_module, "_advance_full_resources", advance_spy)
    monkeypatch.setattr(
        resource_module, "TwoPointEvaluatorResources", reject_aggregate
    )

    result = build_two_point_evaluator_resources(runner)

    assert result is None
    assert raw_advance_starts == [resources_before]
    assert runner.state is aborted_state
    assert runner.state.terminal_abort is abort
    assert runner.state.instrument_resources_current == instrument.resources
    assert tracker.estimate() is pending_estimate


def test_private_builder_rejects_authenticated_unaccepted_missing_midpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, tracker, _, _ = _started_runner(
        monkeypatch, "conditional_free_precalibration"
    )
    query = tracker.choose_next_query()
    assert query is not None
    pending_estimate = tracker.estimate()
    object.__setattr__(
        runner,
        "_state",
        replace(runner.state, tracker_estimate=pending_estimate),
    )
    resources_before = instrument.resources
    expected_midpoint_s = (
        instrument.virtual_time_s + instrument.frequency_overhead_s
    ) + query.integration_time_s / 2.0
    full_observation = instrument.query(
        query.frequency_hz, query.integration_time_s
    )
    authenticated = _build_authenticated_tracking_acquisition(
        query=query,
        expected_midpoint_s=expected_midpoint_s,
        full_observation=full_observation,
        resources_before=resources_before,
        resources_after=instrument.resources,
        overhead_s=instrument.frequency_overhead_s,
    )
    midpoint_erased = replace(authenticated, measurement_midpoint_s=None)

    with pytest.raises(ValueError, match="midpoint"):
        _build_two_point_evaluator_resources_from_context(
            runner,
            authenticated_unaccepted=midpoint_erased,
        )


def test_public_builder_rejects_expected_only_unavailable_missing_midpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, instrument, tracker, _, _ = _started_runner(
        monkeypatch, "conditional_free_precalibration"
    )
    query = tracker.choose_next_query()
    assert query is not None
    pending_estimate = tracker.estimate()
    pending_state = replace(runner.state, tracker_estimate=pending_estimate)
    object.__setattr__(runner, "_state", pending_state)
    resources_before = instrument.resources
    expected_midpoint_s = (
        instrument.virtual_time_s + instrument.frequency_overhead_s
    ) + query.integration_time_s / 2.0
    returned_observation = instrument.query(
        query.frequency_hz, query.integration_time_s
    )
    raw_observation = replace(
        returned_observation,
        expected_photons=math.nextafter(
            returned_observation.expected_photons, math.inf
        ),
    )
    prospective = _advance_full_resources(
        resources_before,
        raw_observation,
        instrument.frequency_overhead_s,
    )
    unavailable = TwoPointResourceJoinUnavailableAcquisition(
        resource_join_status="unavailable",
        query=query,
        expected_measurement_midpoint_s=expected_midpoint_s,
        measurement_midpoint_s=None,
        full_observation=raw_observation,
        safe_observation=raw_observation.estimator_view(),
        resource_mismatch_fields=_resource_mismatch_fields(
            prospective, instrument.resources
        ),
        instrument_resources_before=resources_before,
        instrument_resources_after=instrument.resources,
    )
    abort = TwoPointAbortedRun(
        reason="resource_join_unavailable",
        exception_type=None,
        exception_message=None,
        unaccepted_acquisition=unavailable,
        unaccepted_observation_count=1,
        tracker_estimate_before=pending_estimate,
        tracker_estimate_after=pending_estimate,
    )
    object.__setattr__(
        runner,
        "_state",
        replace(
            pending_state,
            phase="aborted",
            instrument_resources_current=instrument.resources,
            instrument_current_sequence_index=(
                returned_observation.sequence_index
            ),
            current_virtual_time_s=instrument.virtual_time_s,
            terminal_abort=abort,
        ),
    )

    with pytest.raises(ValueError, match="midpoint"):
        build_two_point_evaluator_resources(runner)


def test_builder_authenticates_retriable_failure_resource_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _, _, _ = _started_runner(
        monkeypatch, "conditional_free_precalibration"
    )
    accepted = runner.step()
    assert type(accepted) is TwoPointRunnerAccepted

    def fail_query(
        self: ODMRInstrument,
        frequency_hz: float,
        integration_time_s: float,
    ) -> InstrumentObservation:
        del self, frequency_hz, integration_time_s
        raise RuntimeError("retain pending query")

    with monkeypatch.context() as query_patch:
        query_patch.setattr(ODMRInstrument, "query", fail_query)
        failed = runner.step()
    assert failed.kind == "instrument_failure"
    assert build_two_point_evaluator_resources(runner) is not None

    failure = failed.failure
    changed_boundary = replace(
        failure.instrument_resources_before,
        expected_photons=math.nextafter(
            failure.instrument_resources_before.expected_photons,
            math.inf,
        ),
    )
    changed_failure = replace(
        failure,
        instrument_resources_before=changed_boundary,
        instrument_resources_after=changed_boundary,
    )
    object.__setattr__(
        runner,
        "_state",
        replace(runner.state, last_instrument_failure=changed_failure),
    )

    with pytest.raises(ValueError, match=r"failure.*boundary"):
        build_two_point_evaluator_resources(runner)
