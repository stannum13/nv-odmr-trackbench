"""Tests for runner-owned verified two-point calibration acquisition."""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import replace

import pytest

from odmr_bench.dynamics import SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import GaussianNoise
from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.resources import ResourceLedger, ResourceSnapshot
from odmr_bench.estimators import CompleteSweep
from odmr_bench.estimators.two_point_types import (
    TwoPointCalibrationConstructionError,
)
from odmr_bench.evaluation.two_point import TwoPointEvaluatorRunner
from odmr_bench.evaluation.two_point.provenance import _lookup_run_token_binding
from odmr_bench.evaluation.two_point.types import (
    TwoPointCalibrationPreflightError,
    TwoPointEvaluatorInstrumentConfiguration,
    TwoPointEvaluatorRunnerState,
    VerifiedTwoPointCalibrationFailure,
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


def _instrument(
    *, nominal_photon_rate_hz: float = 2.5e6, frequency_overhead_s: float = 0.001
) -> ODMRInstrument:
    return ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=GaussianNoise(stddev_at_1s=0.0),
        nominal_photon_rate_hz=nominal_photon_rate_hz,
        frequency_overhead_s=frequency_overhead_s,
        seed=13,
    )


def test_runner_bind_derives_clean_configuration_and_registers_identity() -> None:
    instrument = _instrument()

    runner = TwoPointEvaluatorRunner.bind(instrument)

    zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    assert type(runner.state) is TwoPointEvaluatorRunnerState
    assert runner.state is runner.state
    assert runner.state.phase == "ready"
    assert runner.state.instrument_configuration == (
        TwoPointEvaluatorInstrumentConfiguration(2.5e6, 0.001)
    )
    assert runner.state.instrument_resources_at_bind == zero
    assert runner.state.instrument_resources_current == zero
    assert runner.state.instrument_resources_at_bind is (
        runner.state.instrument_resources_current
    )
    assert runner.state.instrument_current_sequence_index is None
    assert runner.state.current_virtual_time_s == 0.0
    assert runner.state.calibration_outcome is None
    assert runner.state.verified_calibration is None
    assert runner.state.calibration is None
    assert runner.state.tracker_estimate is None
    assert runner.state.normal_tracking_trace == ()
    assert runner.state.pair_timings == ()
    assert runner.state.tracking_resources_before is None
    assert runner.state.last_instrument_failure is None
    assert runner.state.terminal_abort is None
    assert instrument.resources == zero
    assert instrument.virtual_time_s == 0.0

    binding = _lookup_run_token_binding(runner.state.run_token)
    assert binding is not None
    assert binding.issuer_runner is runner
    assert binding.instrument is instrument
    assert binding.instrument_configuration is runner.state.instrument_configuration
    assert binding.success is None
    assert binding.source is None


def _valid_calibration_arguments() -> dict[str, object]:
    fit_configuration = make_legal_fit_configuration()
    from odmr_bench.estimators import TwoPointIdentityBinding

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


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("bind_type_before_boundary", "invalid_argument_type"),
        ("bind_value_before_boundary", "invalid_argument_value"),
        ("bind_time_value_before_boundary", "invalid_argument_value"),
        ("bind_boundary", "unclean_instrument_boundary"),
        ("phase_before_type", "invalid_runner_phase"),
        ("type_before_value", "invalid_argument_type"),
        ("value_before_grid", "invalid_argument_value"),
        ("grid_before_fit", "invalid_frequency_grid"),
        ("fit_before_clock", "invalid_fit_or_identity_configuration"),
        ("clock", "invalid_clock_mapping"),
    ],
)
def test_bind_and_calibration_preflight_precedence_has_zero_calls(
    case: str,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrument = _instrument()
    if case.startswith("bind_"):
        instrument.query(2.8e9, 0.005)
        if case == "bind_type_before_boundary":
            object.__setattr__(instrument, "_nominal_photon_rate_hz", object())
        elif case == "bind_value_before_boundary":
            object.__setattr__(instrument, "_nominal_photon_rate_hz", 0.0)
        elif case == "bind_time_value_before_boundary":
            object.__setattr__(instrument, "_virtual_time_s", float("inf"))
        with pytest.raises(TwoPointCalibrationPreflightError) as raised:
            TwoPointEvaluatorRunner.bind(instrument)
        assert raised.value.code == expected_code
        return

    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    if case == "phase_before_type":
        object.__setattr__(runner.state, "phase", "calibration_failed")
        arguments["frequency_hz"] = object()
    elif case == "type_before_value":
        arguments["frequency_hz"] = (True, 3.02e9)
        arguments["integration_time_s"] = 0.0
    elif case == "value_before_grid":
        arguments["frequency_hz"] = (-1.0, -1.0)
        bad_fit = make_legal_fit_configuration(tuple(f"x{i}" for i in range(8)))
        arguments["fit_configuration"] = bad_fit
    elif case == "grid_before_fit":
        arguments["frequency_hz"] = (2.74e9, 2.74e9)
        arguments["fit_configuration"] = make_legal_fit_configuration(
            tuple(f"x{i}" for i in range(8))
        )
    elif case == "fit_before_clock":
        arguments["fit_configuration"] = make_legal_fit_configuration(
            tuple(f"x{i}" for i in range(8))
        )
        arguments["source_clock_id"] = ""
    elif case == "clock":
        arguments["source_to_tracker_offset_s"] = 1.0

    query_calls = 0
    fit_calls = 0

    def reject_query(*args: object, **kwargs: object) -> object:
        nonlocal query_calls
        query_calls += 1
        raise AssertionError((args, kwargs))

    def reject_fit(*args: object, **kwargs: object) -> object:
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(ODMRInstrument, "query", reject_query)
    if importlib.util.find_spec(
        "odmr_bench.evaluation.two_point.calibration"
    ) is not None:
        from odmr_bench.evaluation.two_point import calibration as calibration_module

        monkeypatch.setattr(calibration_module, "fit_spectrum", reject_fit)

    resources_before = instrument.resources
    time_before = instrument.virtual_time_s
    state_before = runner.state
    with pytest.raises(TwoPointCalibrationPreflightError) as raised:
        runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]
    assert raised.value.code == expected_code
    assert query_calls == 0
    assert fit_calls == 0
    assert instrument.resources == resources_before
    assert instrument.virtual_time_s == time_before
    assert runner.state is state_before


def test_verified_calibration_success_uses_safe_fit_and_identity_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    fit_configuration = arguments["fit_configuration"]
    assert type(fit_configuration) is type(make_legal_fit_configuration())
    returned_fit = make_legal_source_fit(fit_configuration)
    fit_calls: list[tuple[CompleteSweep, object, object]] = []

    def fit_spy(
        sweep: CompleteSweep,
        configuration: object,
        initial_guess: object = object(),
    ) -> object:
        fit_calls.append((sweep, configuration, initial_guess))
        return returned_fit

    monkeypatch.setattr(calibration_module, "fit_spectrum", fit_spy)

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationSuccess
    assert len(fit_calls) == 1
    sweep, fitted_configuration, initial_guess = fit_calls[0]
    assert type(sweep) is CompleteSweep
    assert tuple(sweep.frequency_hz) == (2.74e9, 3.02e9)
    assert tuple(sweep.fluorescence) == tuple(
        observation.fluorescence for observation in outcome.safe_observations
    )
    assert sweep.last_sequence_index == 1
    assert sweep.last_timestamp_s == 0.012
    assert sweep.total_integration_time_s == 0.01
    assert sweep.total_nominal_exposure_photons == 25_000.0
    assert fitted_configuration == fit_configuration
    assert fitted_configuration is not fit_configuration
    assert initial_guess is None
    assert not hasattr(sweep, "expected_photons")

    assert outcome.status == "success"
    assert type(outcome.full_observations) is tuple
    assert outcome.safe_observations == tuple(
        observation.estimator_view()
        for observation in outcome.full_observations
    )
    assert outcome.measurement_midpoints_s == (0.0035, 0.0095)
    assert tuple(item.sequence_index for item in outcome.full_observations) == (0, 1)
    assert tuple(item.timestamp_s for item in outcome.full_observations) == (
        0.006,
        0.012,
    )
    assert tuple(item.integration_time_s for item in outcome.full_observations) == (
        0.005,
        0.005,
    )
    assert tuple(
        item.nominal_exposure_photons for item in outcome.full_observations
    ) == (12_500.0, 12_500.0)
    assert tuple(item.sampling_rule for item in outcome.full_observations) == (
        "gaussian",
        "gaussian",
    )
    assert outcome.instrument_resources_before == ResourceSnapshot(
        0, 0.0, 0.0, 0.0, 0, 0, 0.0
    )
    assert outcome.instrument_resources_after == instrument.resources
    assert outcome.full_resources == instrument.resources
    assert outcome.safe_resources.observations == 2
    assert outcome.safe_resources.integration_time_s == 0.01
    assert outcome.safe_resources.nominal_exposure_photons == 25_000.0
    assert outcome.safe_resources.realized_photons == 0
    assert outcome.safe_resources.observations_without_realized_counts == 2
    assert outcome.safe_resources.virtual_elapsed_time_s == 0.012
    assert not hasattr(outcome.safe_resources, "expected_photons")

    source = outcome.source
    assert source.provenance == "verified_factory_acquisition"
    assert source.source_id == "verified-source"
    assert source.source_fit is not returned_fit
    assert source.source_fit.success is True
    assert source.source_fit.model_kind == returned_fit.model_kind
    assert source.source_fit.baseline_degree == returned_fit.baseline_degree
    assert source.source_fit.resonance_estimates == returned_fit.resonance_estimates
    assert source.fit_configuration == fit_configuration
    assert source.fit_configuration is not fit_configuration
    assert source.identity_binding == arguments["identity_binding"]
    assert source.identity_binding is not arguments["identity_binding"]
    assert source.resolved_resonance_ids == tuple(f"r{i}" for i in range(8))
    assert source.source_observations == outcome.safe_observations
    assert source.source_observations is not outcome.safe_observations
    assert source.fluorescence_provenance.quantity == "normalized_fluorescence"
    assert source.fluorescence_provenance.normalization_rule == (
        "odmr_instrument_normalized_fluorescence_v1"
    )
    assert source.fluorescence_provenance.nominal_photon_rate_hz == 2.5e6
    assert source.fluorescence_provenance.sampling_rules == (
        "gaussian",
        "gaussian",
    )
    assert source.source_frequency_overhead_s == 0.001
    assert source.source_frequency_min_hz == 2.74e9
    assert source.source_frequency_max_hz == 3.02e9
    assert source.source_first_sequence_index == 0
    assert source.source_last_sequence_index == 1
    assert source.source_start_timestamp_s == 0.0
    assert source.source_first_timestamp_s == 0.006
    assert source.source_last_timestamp_s == 0.012
    assert source.physical_fit_epoch_s == 0.0035 + (0.0095 - 0.0035) / 2.0
    assert source.availability_sequence_index == 1
    assert source.availability_timestamp_s == 0.012
    assert source.safe_resources == outcome.safe_resources
    assert source.clock_mapping.kind == "shared_clock"
    assert source.clock_mapping.source_clock_id == "clock"
    assert source.clock_mapping.tracker_clock_id == "clock"
    assert source.clock_mapping.scale == 1.0
    assert source.clock_mapping.offset_s == 0.0

    binding = _lookup_run_token_binding(outcome.run_token)
    assert binding is not None
    assert binding.issuer_runner is runner
    assert binding.instrument is instrument
    assert binding.success is outcome
    assert binding.source is outcome.source
    assert runner.state.phase == "calibration_succeeded"
    assert runner.state.calibration_outcome is outcome
    assert runner.state.verified_calibration is outcome
    assert runner.state.instrument_resources_current == instrument.resources
    assert runner.state.instrument_current_sequence_index == 1
    assert runner.state.current_virtual_time_s == 0.012


def test_verified_source_identity_matrix_rejects_public_mint_and_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.estimators.two_point_calibration import (
        _VERIFIED_SOURCE_CONSTRUCTION_KEY,
        _bind_verified_two_point_calibration_source,
    )
    from odmr_bench.evaluation.two_point import calibration as calibration_module
    from odmr_bench.evaluation.two_point.provenance import (
        _bind_run_token_success,
        _register_run_token,
    )

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    fit_configuration = arguments["fit_configuration"]
    returned_fit = make_legal_source_fit(fit_configuration)
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: returned_fit,
    )
    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]
    assert type(outcome) is VerifiedTwoPointCalibrationSuccess

    with pytest.raises(TypeError, match="construction key"):
        _bind_verified_two_point_calibration_source(
            outcome.source.source_fit,
            outcome.source.fit_configuration,
            outcome.source.source_observations,
            outcome.source.identity_binding,
            outcome.source.fluorescence_provenance,
            source_id=outcome.source.source_id,
            source_frequency_overhead_s=(
                outcome.source.source_frequency_overhead_s
            ),
            source_start_timestamp_s=outcome.source.source_start_timestamp_s,
            physical_fit_epoch_s=outcome.source.physical_fit_epoch_s,
            availability_sequence_index=(
                outcome.source.availability_sequence_index
            ),
            availability_timestamp_s=outcome.source.availability_timestamp_s,
            clock_mapping=outcome.source.clock_mapping,
            construction_key=object(),
        )

    with pytest.raises(TwoPointCalibrationConstructionError) as raised_epoch:
        _bind_verified_two_point_calibration_source(
            outcome.source.source_fit,
            outcome.source.fit_configuration,
            outcome.source.source_observations,
            outcome.source.identity_binding,
            outcome.source.fluorescence_provenance,
            source_id=outcome.source.source_id,
            source_frequency_overhead_s=(
                outcome.source.source_frequency_overhead_s
            ),
            source_start_timestamp_s=outcome.source.source_start_timestamp_s,
            physical_fit_epoch_s=outcome.source.physical_fit_epoch_s + 0.001,
            availability_sequence_index=(
                outcome.source.availability_sequence_index
            ),
            availability_timestamp_s=outcome.source.availability_timestamp_s,
            clock_mapping=outcome.source.clock_mapping,
            construction_key=_VERIFIED_SOURCE_CONSTRUCTION_KEY,
        )
    assert raised_epoch.value.code == "invalid_source_epoch"

    caller_source = make_legal_caller_asserted_source()
    with pytest.raises(ValueError, match="private factory"):
        replace(caller_source, provenance="verified_factory_acquisition")
    with pytest.raises(ValueError, match="private factory"):
        replace(outcome.source)

    copied_source = copy.copy(outcome.source)
    copied_outcome = replace(outcome)
    binding = _lookup_run_token_binding(outcome.run_token)
    assert binding is not None
    assert binding.success is outcome
    assert binding.success is not copied_outcome
    assert binding.source is outcome.source
    assert binding.source is not copied_source

    unregistered_token = object.__new__(type(outcome.run_token))
    assert _lookup_run_token_binding(unregistered_token) is None
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(outcome.run_token)

    other_instrument = _instrument()
    other_runner = TwoPointEvaluatorRunner.bind(other_instrument)
    with pytest.raises(ValueError, match="runner-minted identity"):
        _register_run_token(
            unregistered_token,
            other_runner,
            other_instrument,
            other_runner.state.instrument_configuration,
        )
    assert _lookup_run_token_binding(unregistered_token) is None

    copied_source_outcome = replace(
        outcome,
        run_token=other_runner.state.run_token,
        source=copied_source,
    )
    with pytest.raises(ValueError, match="registered identity"):
        _bind_run_token_success(
            other_runner.state.run_token,
            other_runner,
            other_instrument,
            copied_source_outcome,
        )

    directly_assembled = replace(
        outcome,
        run_token=other_runner.state.run_token,
        source=caller_source,
    )
    with pytest.raises(ValueError, match="registered identity"):
        _bind_run_token_success(
            other_runner.state.run_token,
            other_runner,
            other_instrument,
            directly_assembled,
        )
    other_binding = _lookup_run_token_binding(other_runner.state.run_token)
    assert other_binding is not None
    assert other_binding.success is None
    assert other_binding.source is None


def _acquire_with_second_observation_mutation(
    monkeypatch: pytest.MonkeyPatch,
    **changes: float | int,
) -> tuple[VerifiedTwoPointCalibrationFailure, TwoPointEvaluatorRunner]:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    original_query = ODMRInstrument.query

    def corrupt_second(
        target: ODMRInstrument, frequency_hz: float, integration_time_s: float
    ) -> object:
        observation = original_query(target, frequency_hz, integration_time_s)
        if observation.sequence_index == 1:
            return replace(observation, **changes)
        return observation

    monkeypatch.setattr(ODMRInstrument, "query", corrupt_second)
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError((args, kwargs))
        ),
    )
    outcome = runner.acquire_verified_calibration(  # type: ignore[assignment]
        **_valid_calibration_arguments()  # type: ignore[arg-type]
    )
    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    return outcome, runner


@pytest.mark.parametrize(
    ("changes", "failure_code"),
    [
        ({"timestamp_s": 0.0125}, "acquisition_contract_mismatch"),
        ({"integration_time_s": 0.006}, "resource_join_unavailable"),
    ],
)
def test_calibration_timing_or_integration_mismatch_sets_final_midpoint_none(
    changes: dict[str, float],
    failure_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, runner = _acquire_with_second_observation_mutation(
        monkeypatch, **changes
    )

    assert outcome.failure_code == failure_code
    assert outcome.failed_request is not None
    assert outcome.failed_request.point_index == 1
    assert outcome.measurement_midpoints_s == (0.0035, None)
    assert len(outcome.full_observations) == 2
    assert outcome.safe_observations == tuple(
        observation.estimator_view()
        for observation in outcome.full_observations
    )
    assert runner.state.phase == "calibration_failed"
    assert runner.state.calibration_outcome is outcome
    assert runner.state.verified_calibration is None


@pytest.mark.parametrize(
    "changes",
    [
        {"frequency_hz": 3.020000001e9},
        {"sequence_index": 7},
    ],
)
def test_calibration_frequency_or_sequence_mismatch_retains_exact_midpoint(
    changes: dict[str, float | int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, runner = _acquire_with_second_observation_mutation(
        monkeypatch, **changes
    )

    assert outcome.failure_code == "acquisition_contract_mismatch"
    assert outcome.failed_request is not None
    assert outcome.failed_request.point_index == 1
    assert outcome.measurement_midpoints_s == (0.0035, 0.0095)
    assert outcome.safe_resources is not None
    assert outcome.full_resources is not None
    assert runner.state.instrument_current_sequence_index == 1


@pytest.mark.parametrize(
    ("changes", "mismatch_fields", "expected_final_midpoint"),
    [
        (
            {"nominal_exposure_photons": 12_501.0},
            ("nominal_exposure_photons",),
            0.0095,
        ),
        ({"expected_photons": 1.0}, ("expected_photons",), 0.0095),
        (
            {"integration_time_s": 0.006},
            ("integration_time_s", "virtual_elapsed_time_s"),
            None,
        ),
    ],
)
def test_calibration_resource_corruption_midpoint_slots_follow_timing_only(
    changes: dict[str, float],
    mismatch_fields: tuple[str, ...],
    expected_final_midpoint: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, runner = _acquire_with_second_observation_mutation(
        monkeypatch, **changes
    )

    assert outcome.failure_code == "resource_join_unavailable"
    assert outcome.resource_mismatch_fields == mismatch_fields
    assert outcome.measurement_midpoints_s == (
        0.0035,
        expected_final_midpoint,
    )
    assert outcome.safe_resources is None
    assert outcome.full_resources is None
    assert runner.state.instrument_resources_current == (
        outcome.instrument_resources_after
    )


def test_calibration_rejects_discontinuous_authoritative_resource_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    resources_getter = ODMRInstrument.resources.fget
    assert resources_getter is not None
    resource_reads = 0

    def discontinuous_resources(target: ODMRInstrument) -> ResourceSnapshot:
        nonlocal resource_reads
        resource_reads += 1
        snapshot = resources_getter(target)
        if resource_reads in {4, 5}:
            return replace(
                snapshot,
                expected_photons=snapshot.expected_photons + 1.0,
            )
        return snapshot

    monkeypatch.setattr(
        ODMRInstrument,
        "resources",
        property(discontinuous_resources),
    )
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError((args, kwargs))
        ),
    )

    outcome = runner.acquire_verified_calibration(  # type: ignore[assignment]
        **_valid_calibration_arguments()  # type: ignore[arg-type]
    )

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "resource_join_unavailable"
    assert outcome.resource_mismatch_fields == ("expected_photons",)
    assert len(outcome.full_observations) == 2
    assert outcome.measurement_midpoints_s == (0.0035, 0.0095)
    assert outcome.safe_resources is None
    assert outcome.full_resources is None
    assert runner.state.phase == "calibration_failed"


@pytest.mark.parametrize(
    ("case", "expected_fields"),
    [
        (
            "local_before_continuity",
            ("nominal_exposure_photons", "expected_photons"),
        ),
        (
            "continuity_before_local",
            ("nominal_exposure_photons", "expected_photons"),
        ),
        ("overlapping", ("expected_photons",)),
    ],
)
def test_calibration_resource_join_reports_complete_ordered_union(
    case: str,
    expected_fields: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    resources_getter = ODMRInstrument.resources.fget
    assert resources_getter is not None
    resource_reads = 0

    def corrupt_resources(target: ODMRInstrument) -> ResourceSnapshot:
        nonlocal resource_reads
        resource_reads += 1
        snapshot = resources_getter(target)
        if resource_reads == 4:
            return replace(
                snapshot,
                **(
                    {
                        "nominal_exposure_photons": (
                            snapshot.nominal_exposure_photons + 1.0
                        )
                    }
                    if case == "continuity_before_local"
                    else {"expected_photons": snapshot.expected_photons + 1.0}
                ),
            )
        if resource_reads == 5:
            if case == "overlapping":
                return snapshot
            return replace(
                snapshot,
                nominal_exposure_photons=(
                    snapshot.nominal_exposure_photons + 1.0
                ),
                expected_photons=snapshot.expected_photons + 1.0,
            )
        return snapshot

    monkeypatch.setattr(
        ODMRInstrument,
        "resources",
        property(corrupt_resources),
    )
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError((args, kwargs))
        ),
    )

    outcome = runner.acquire_verified_calibration(  # type: ignore[assignment]
        **_valid_calibration_arguments()  # type: ignore[arg-type]
    )

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "resource_join_unavailable"
    assert outcome.resource_mismatch_fields == expected_fields
    assert len(outcome.full_observations) == 2
    assert outcome.measurement_midpoints_s == (0.0035, 0.0095)
    assert runner.state.phase == "calibration_failed"


@pytest.mark.parametrize(
    "case",
    (
        "instrument_exception",
        "instrument_exception_unprintable",
        "instrument_exception_stringification_moves_instrument",
        "instrument_commit_then_raises",
        "invalid_query_result",
        "fit_failed",
        "fit_exception",
        "fit_invalid_result",
        "post_fit_snapshot_failed",
        "source_binding_failed",
        "source_binding_constructs_then_raises",
        "success_binding_failed",
        "success_binding_commits_then_raises",
        "success_state_construction_failed",
        "fit_mutates_instrument",
        "fit_resets_resource_ledger",
        "source_binding_mutates_instrument",
        "post_source_snapshot_failed",
        "resource_unavailable",
    ),
)
def test_verified_calibration_preserves_every_typed_post_start_failure(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    class QueryFailure(RuntimeError):
        pass

    class SourceBindingFailure(RuntimeError):
        pass

    class UnprintableQueryFailure(RuntimeError):
        def __str__(self) -> str:
            raise ValueError("query exception stringification exploded")

    class MutatingStringQueryFailure(RuntimeError):
        def __str__(self) -> str:
            nonlocal string_calls
            string_calls += 1
            original_query(instrument, 2.9e9, 0.005)
            return "query exception moved instrument"

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    original_query = ODMRInstrument.query
    query_calls = 0
    fit_calls = 0
    string_calls = 0
    captured_sources: list[object] = []
    successful_fit = make_legal_source_fit(arguments["fit_configuration"])
    structured_failure = replace(
        successful_fit,
        success=False,
        failure_code="quality_failed",
        resonance_estimates=(),
        baseline_estimate=None,
    )

    if case in {
        "instrument_exception",
        "instrument_exception_unprintable",
        "instrument_exception_stringification_moves_instrument",
        "instrument_commit_then_raises",
        "invalid_query_result",
    }:
        arguments["frequency_hz"] = (2.74e9, 2.88e9, 3.02e9)

        def query(target: ODMRInstrument, frequency: float, integration: float):
            nonlocal query_calls
            query_calls += 1
            if query_calls == 3:
                if case == "instrument_exception":
                    raise QueryFailure("query exploded")
                if case == "instrument_exception_unprintable":
                    raise UnprintableQueryFailure()
                if case == "instrument_exception_stringification_moves_instrument":
                    raise MutatingStringQueryFailure()
                if case == "instrument_commit_then_raises":
                    original_query(target, frequency, integration)
                    raise QueryFailure("query committed then exploded")
                return object()
            return original_query(target, frequency, integration)

        fit_behavior: object = AssertionError("fit must not be called")
    elif case == "resource_unavailable":

        def query(target: ODMRInstrument, frequency: float, integration: float):
            nonlocal query_calls
            query_calls += 1
            observation = original_query(target, frequency, integration)
            if observation.sequence_index == 1:
                return replace(
                    observation,
                    integration_time_s=0.006,
                    nominal_exposure_photons=12_501.0,
                    expected_photons=1.0,
                    realized_photons=7,
                )
            return observation

        fit_behavior = AssertionError("fit must not be called")
    else:

        def query(target: ODMRInstrument, frequency: float, integration: float):
            nonlocal query_calls
            query_calls += 1
            return original_query(target, frequency, integration)

        if case == "fit_failed":
            fit_behavior = structured_failure
        elif case == "fit_exception":
            fit_behavior = RuntimeError("fit exploded")
        elif case == "fit_invalid_result":
            fit_behavior = object()
        else:
            fit_behavior = successful_fit

    def fit(*args: object, **kwargs: object) -> object:
        nonlocal fit_calls
        fit_calls += 1
        if case == "fit_mutates_instrument":
            original_query(instrument, 2.9e9, 0.005)
        elif case == "fit_resets_resource_ledger":
            object.__setattr__(instrument, "_ledger", ResourceLedger())
        if isinstance(fit_behavior, BaseException):
            raise fit_behavior
        return fit_behavior

    monkeypatch.setattr(ODMRInstrument, "query", query)
    monkeypatch.setattr(calibration_module, "fit_spectrum", fit)
    if case == "source_binding_failed":
        monkeypatch.setattr(
            calibration_module,
            "_bind_verified_two_point_calibration_source",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                SourceBindingFailure("source bind exploded")
            ),
        )
    elif case == "source_binding_constructs_then_raises":
        original_bind_source = (
            calibration_module._bind_verified_two_point_calibration_source
        )

        def construct_then_fail(*args: object, **kwargs: object) -> object:
            constructed = original_bind_source(*args, **kwargs)
            captured_sources.append(constructed)
            raise SourceBindingFailure("source bind failed after construction")

        monkeypatch.setattr(
            calibration_module,
            "_bind_verified_two_point_calibration_source",
            construct_then_fail,
        )
    elif case == "source_binding_mutates_instrument":
        def mutate_then_fail(*args: object, **kwargs: object) -> object:
            del args, kwargs
            original_query(instrument, 2.9e9, 0.005)
            raise SourceBindingFailure("source bind moved instrument")

        monkeypatch.setattr(
            calibration_module,
            "_bind_verified_two_point_calibration_source",
            mutate_then_fail,
        )
    elif case == "success_binding_failed":
        monkeypatch.setattr(
            calibration_module,
            "_bind_run_token_success",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                SourceBindingFailure("success bind exploded")
            ),
        )
    elif case == "success_binding_commits_then_raises":
        original_bind_success = calibration_module._bind_run_token_success

        def commit_then_fail(*args: object, **kwargs: object) -> None:
            original_bind_success(*args, **kwargs)
            raise SourceBindingFailure("success bind failed after commit")

        monkeypatch.setattr(
            calibration_module,
            "_bind_run_token_success",
            commit_then_fail,
        )
    elif case == "success_state_construction_failed":
        original_replace = calibration_module.replace

        def reject_success_state(*args: object, **changes: object) -> object:
            if changes.get("phase") == "calibration_succeeded":
                raise RuntimeError("success state construction exploded")
            return original_replace(*args, **changes)

        monkeypatch.setattr(calibration_module, "replace", reject_success_state)
    elif case in {"post_fit_snapshot_failed", "post_source_snapshot_failed"}:
        resources_getter = ODMRInstrument.resources.fget
        assert resources_getter is not None
        resource_reads = 0

        def fail_post_source_snapshot(
            target: ODMRInstrument,
        ) -> ResourceSnapshot:
            nonlocal resource_reads
            resource_reads += 1
            failed_read = 6 if case == "post_fit_snapshot_failed" else 8
            if resource_reads == failed_read:
                boundary = "post-fit" if failed_read == 6 else "post-source"
                raise RuntimeError(f"{boundary} snapshot exploded")
            return resources_getter(target)

        monkeypatch.setattr(
            ODMRInstrument,
            "resources",
            property(fail_post_source_snapshot),
        )

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.status == "failure"
    expected_code = {
        "instrument_exception": "instrument_query_failed",
        "instrument_exception_unprintable": "instrument_query_failed",
        "instrument_exception_stringification_moves_instrument": (
            "resource_join_unavailable"
        ),
        "instrument_commit_then_raises": "resource_join_unavailable",
        "invalid_query_result": "acquisition_contract_mismatch",
        "fit_failed": "fit_failed",
        "fit_exception": "fit_exception",
        "fit_invalid_result": "fit_exception",
        "post_fit_snapshot_failed": "fit_exception",
        "source_binding_failed": "source_binding_failed",
        "source_binding_constructs_then_raises": "source_binding_failed",
        "success_binding_failed": "source_binding_failed",
        "success_binding_commits_then_raises": "source_binding_failed",
        "success_state_construction_failed": "source_binding_failed",
        "fit_mutates_instrument": "resource_join_unavailable",
        "fit_resets_resource_ledger": "resource_join_unavailable",
        "source_binding_mutates_instrument": "resource_join_unavailable",
        "post_source_snapshot_failed": "source_binding_failed",
        "resource_unavailable": "resource_join_unavailable",
    }[case]
    assert outcome.failure_code == expected_code
    assert len(outcome.full_observations) == 2
    assert len(outcome.safe_observations) == 2
    assert outcome.safe_observations == tuple(
        observation.estimator_view()
        for observation in outcome.full_observations
    )
    assert outcome.measurement_midpoints_s == (
        (0.0035, None)
        if case == "resource_unavailable"
        else (0.0035, 0.0095)
    )
    assert outcome.instrument_resources_before == ResourceSnapshot(
        0, 0.0, 0.0, 0.0, 0, 0, 0.0
    )
    assert outcome.instrument_resources_after == instrument.resources
    assert runner.state.phase == "calibration_failed"
    assert runner.state.calibration_outcome is outcome
    assert runner.state.verified_calibration is None
    assert runner.state.instrument_resources_current == instrument.resources
    collaborator_moved_instrument = case in {
        "instrument_commit_then_raises",
        "instrument_exception_stringification_moves_instrument",
        "fit_mutates_instrument",
        "source_binding_mutates_instrument",
    }
    assert runner.state.current_virtual_time_s == instrument.virtual_time_s
    assert runner.state.current_virtual_time_s == pytest.approx(
        0.018 if collaborator_moved_instrument else 0.012
    )
    assert runner.state.instrument_current_sequence_index == (
        None
        if case == "fit_resets_resource_ledger"
        else (2 if collaborator_moved_instrument else 1)
    )

    if case in {
        "instrument_exception",
        "instrument_exception_unprintable",
        "invalid_query_result",
    }:
        assert query_calls == 3
        assert fit_calls == 0
        assert outcome.failed_request is not None
        assert outcome.failed_request.point_index == 2
        assert outcome.exception_type == (
            "QueryFailure"
            if case == "instrument_exception"
            else (
                "UnprintableQueryFailure"
                if case == "instrument_exception_unprintable"
                else None
            )
        )
        assert outcome.exception_message == (
            "query exploded"
            if case == "instrument_exception"
            else ("" if case == "instrument_exception_unprintable" else None)
        )
        assert outcome.fit_result is None
    elif case in {
        "instrument_commit_then_raises",
        "instrument_exception_stringification_moves_instrument",
    }:
        assert query_calls == 3
        assert fit_calls == 0
        assert outcome.failed_request is not None
        assert outcome.failed_request.point_index == 2
        assert outcome.exception_type is None
        assert outcome.exception_message is None
        assert outcome.fit_result is None
        assert "observations" in outcome.resource_mismatch_fields
        assert "virtual_elapsed_time_s" in outcome.resource_mismatch_fields
        assert string_calls == (
            1
            if case == "instrument_exception_stringification_moves_instrument"
            else 0
        )
    elif case == "fit_failed":
        assert query_calls == 2
        assert fit_calls == 1
        assert outcome.failed_request is None
        assert outcome.exception_type is None
        assert outcome.fit_result is structured_failure
    elif case in {
        "fit_exception",
        "fit_invalid_result",
        "post_fit_snapshot_failed",
    }:
        assert query_calls == 2
        assert fit_calls == 1
        assert outcome.failed_request is None
        assert outcome.exception_type == (
            "TypeError" if case == "fit_invalid_result" else "RuntimeError"
        )
        assert outcome.exception_message == (
            "fit_spectrum must return an exact SpectrumFitResult"
            if case == "fit_invalid_result"
            else (
                "fit exploded"
                if case == "fit_exception"
                else "post-fit snapshot exploded"
            )
        )
        assert outcome.fit_result is None
    elif case in {
        "source_binding_failed",
        "source_binding_constructs_then_raises",
        "success_binding_failed",
        "success_binding_commits_then_raises",
        "success_state_construction_failed",
        "post_source_snapshot_failed",
    }:
        assert query_calls == 2
        assert fit_calls == 1
        assert outcome.failed_request is None
        assert outcome.exception_type == (
            "RuntimeError"
            if case in {
                "success_state_construction_failed",
                "post_source_snapshot_failed",
            }
            else "SourceBindingFailure"
        )
        assert outcome.exception_message == (
            "source bind exploded"
            if case == "source_binding_failed"
            else (
                "source bind failed after construction"
                if case == "source_binding_constructs_then_raises"
                else (
                    "success bind exploded"
                    if case == "success_binding_failed"
                    else (
                        "success bind failed after commit"
                        if case == "success_binding_commits_then_raises"
                        else (
                            "success state construction exploded"
                            if case == "success_state_construction_failed"
                            else "post-source snapshot exploded"
                        )
                    )
                )
            )
        )
        assert outcome.fit_result is successful_fit
    elif collaborator_moved_instrument or case == "fit_resets_resource_ledger":
        assert query_calls == (
            3 if case == "instrument_commit_then_raises" else 2
        )
        assert fit_calls == 1
        assert outcome.failed_request is not None
        assert outcome.failed_request.point_index == 1
        assert outcome.exception_type is None
        assert outcome.exception_message is None
        assert outcome.fit_result is None
        assert "observations" in outcome.resource_mismatch_fields
        assert "virtual_elapsed_time_s" in outcome.resource_mismatch_fields
    else:
        assert query_calls == 2
        assert fit_calls == 0
        assert outcome.failed_request is not None
        assert outcome.failed_request.point_index == 1
        assert outcome.exception_type is None
        assert outcome.fit_result is None
        assert outcome.resource_mismatch_fields == (
            "integration_time_s",
            "nominal_exposure_photons",
            "expected_photons",
            "realized_photons",
            "observations_without_realized_counts",
            "virtual_elapsed_time_s",
        )

    resource_failure = (
        case in {"resource_unavailable", "fit_resets_resource_ledger"}
        or collaborator_moved_instrument
    )
    if resource_failure:
        assert outcome.safe_resources is None
        assert outcome.full_resources is None
    else:
        assert outcome.safe_resources is not None
        assert outcome.full_resources is not None
        assert outcome.full_resources == instrument.resources
    assert outcome.resource_mismatch_fields == (
        outcome.resource_mismatch_fields
        if resource_failure
        else ()
    )

    binding = _lookup_run_token_binding(outcome.run_token)
    assert binding is not None
    assert binding.issuer_runner is runner
    assert binding.instrument is instrument
    assert binding.success is None
    assert binding.source is None
    if case == "source_binding_constructs_then_raises":
        from odmr_bench.estimators.two_point_calibration import (
            _consume_verified_source_construction_identity,
        )

        assert len(captured_sources) == 1
        assert not _consume_verified_source_construction_identity(
            captured_sources[0]
        )
    resources_after_failure = instrument.resources
    queries_after_failure = query_calls
    with pytest.raises(TwoPointCalibrationPreflightError) as raised:
        runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]
    assert raised.value.code == "invalid_runner_phase"
    assert query_calls == queries_after_failure
    assert instrument.resources == resources_after_failure


@pytest.mark.parametrize("boundary", ("pre_query", "post_query"))
def test_query_boundary_snapshot_failures_terminalize_with_committed_prefix(
    boundary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    query_calls = 0
    fit_calls = 0
    original_query = ODMRInstrument.query
    resources_getter = ODMRInstrument.resources.fget
    time_getter = ODMRInstrument.virtual_time_s.fget
    assert resources_getter is not None
    assert time_getter is not None

    def query(
        target: ODMRInstrument, frequency_hz: float, integration_time_s: float
    ) -> object:
        nonlocal query_calls
        query_calls += 1
        return original_query(target, frequency_hz, integration_time_s)

    def fit(*args: object, **kwargs: object) -> object:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        raise AssertionError("fit must not be called")

    resource_reads = 0
    time_reads = 0

    def resources(target: ODMRInstrument) -> ResourceSnapshot:
        nonlocal resource_reads
        resource_reads += 1
        if boundary == "pre_query" and resource_reads == 2:
            raise RuntimeError("pre-query resources snapshot exploded")
        return resources_getter(target)

    def virtual_time_s(target: ODMRInstrument) -> float:
        nonlocal time_reads
        time_reads += 1
        if boundary == "post_query" and time_reads == 3:
            raise RuntimeError("post-query time snapshot exploded")
        return time_getter(target)

    monkeypatch.setattr(ODMRInstrument, "query", query)
    monkeypatch.setattr(ODMRInstrument, "resources", property(resources))
    monkeypatch.setattr(
        ODMRInstrument, "virtual_time_s", property(virtual_time_s)
    )
    monkeypatch.setattr(calibration_module, "fit_spectrum", fit)

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "instrument_query_failed"
    assert outcome.failed_request is not None
    assert outcome.failed_request.point_index == 0
    assert outcome.exception_type == "RuntimeError"
    assert outcome.exception_message == (
        "pre-query resources snapshot exploded"
        if boundary == "pre_query"
        else "post-query time snapshot exploded"
    )
    expected_count = 0 if boundary == "pre_query" else 1
    assert query_calls == expected_count
    assert fit_calls == 0
    assert len(outcome.full_observations) == expected_count
    assert len(outcome.safe_observations) == expected_count
    assert outcome.measurement_midpoints_s == (
        () if boundary == "pre_query" else (0.0035,)
    )
    assert outcome.instrument_resources_after == instrument.resources
    assert runner.state.phase == "calibration_failed"
    assert runner.state.calibration_outcome is outcome
    assert runner.state.instrument_resources_current == instrument.resources
    assert runner.state.current_virtual_time_s == instrument.virtual_time_s
    assert runner.state.instrument_current_sequence_index == (
        None if boundary == "pre_query" else 0
    )


@pytest.mark.parametrize("stage", ("post_fit", "post_source"))
def test_partial_boundary_capture_resnapshots_after_hostile_exception_rendering(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    original_query = ODMRInstrument.query
    time_getter = ODMRInstrument.virtual_time_s.fget
    assert time_getter is not None
    successful_fit = make_legal_source_fit(arguments["fit_configuration"])
    string_calls = 0
    time_reads = 0

    class MutatingSnapshotFailure(RuntimeError):
        def __str__(self) -> str:
            nonlocal string_calls
            string_calls += 1
            original_query(instrument, 2.9e9, 0.005)
            return f"{stage} time snapshot moved instrument"

    def virtual_time_s(target: ODMRInstrument) -> float:
        nonlocal time_reads
        time_reads += 1
        failed_read = 6 if stage == "post_fit" else 8
        if time_reads == failed_read:
            raise MutatingSnapshotFailure()
        return time_getter(target)

    monkeypatch.setattr(
        ODMRInstrument, "virtual_time_s", property(virtual_time_s)
    )
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: successful_fit,
    )

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "resource_join_unavailable"
    assert outcome.failed_request is not None
    assert outcome.failed_request.point_index == 1
    assert outcome.exception_type is None
    assert outcome.exception_message is None
    assert outcome.fit_result is None
    assert len(outcome.full_observations) == 2
    assert outcome.measurement_midpoints_s == (0.0035, 0.0095)
    assert outcome.instrument_resources_after == instrument.resources
    assert outcome.instrument_resources_after.observations == 3
    assert "observations" in outcome.resource_mismatch_fields
    assert "virtual_elapsed_time_s" in outcome.resource_mismatch_fields
    assert string_calls == 1
    assert runner.state.phase == "calibration_failed"
    assert runner.state.instrument_resources_current == instrument.resources
    assert runner.state.instrument_current_sequence_index == 2
    assert runner.state.current_virtual_time_s == instrument.virtual_time_s
    assert runner.state.current_virtual_time_s == pytest.approx(0.018)
    binding = _lookup_run_token_binding(outcome.run_token)
    assert binding is not None
    assert binding.success is None
    assert binding.source is None


def test_post_query_snapshot_rendering_uses_fresh_time_for_resource_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    time_getter = ODMRInstrument.virtual_time_s.fget
    assert time_getter is not None
    time_reads = 0
    string_calls = 0

    class MutatingSnapshotFailure(RuntimeError):
        def __str__(self) -> str:
            nonlocal string_calls
            string_calls += 1
            object.__setattr__(instrument, "_virtual_time_s", 0.007)
            return "post-query snapshot rendering moved time"

    def virtual_time_s(target: ODMRInstrument) -> float:
        nonlocal time_reads
        time_reads += 1
        if time_reads == 3:
            raise MutatingSnapshotFailure()
        return time_getter(target)

    monkeypatch.setattr(
        ODMRInstrument, "virtual_time_s", property(virtual_time_s)
    )
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fit must not be called")
        ),
    )

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "resource_join_unavailable"
    assert outcome.resource_mismatch_fields == ("virtual_elapsed_time_s",)
    assert len(outcome.full_observations) == 1
    assert outcome.measurement_midpoints_s == (None,)
    assert outcome.instrument_resources_after == instrument.resources
    assert runner.state.current_virtual_time_s == 0.007
    assert runner.state.phase == "calibration_failed"
    assert string_calls == 1


def test_verified_source_consumption_rejects_post_factory_field_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.estimators.two_point_calibration import (
        _consume_verified_source_construction_identity,
    )
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    successful_fit = make_legal_source_fit(arguments["fit_configuration"])
    original_bind_source = (
        calibration_module._bind_verified_two_point_calibration_source
    )
    captured_sources: list[object] = []

    def mutate_verified_fields(*args: object, **kwargs: object) -> object:
        source = original_bind_source(*args, **kwargs)
        captured_sources.append(source)
        object.__setattr__(source, "physical_fit_epoch_s", 123.0)
        object.__setattr__(
            source, "resolved_resonance_ids", tuple(f"x{i}" for i in range(8))
        )
        return source

    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: successful_fit,
    )
    monkeypatch.setattr(
        calibration_module,
        "_bind_verified_two_point_calibration_source",
        mutate_verified_fields,
    )

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "source_binding_failed"
    assert outcome.exception_type == "ValueError"
    assert outcome.exception_message == (
        "run token success does not match its registered identity"
    )
    assert runner.state.phase == "calibration_failed"
    binding = _lookup_run_token_binding(outcome.run_token)
    assert binding is not None
    assert binding.success is None
    assert binding.source is None
    assert len(captured_sources) == 1
    assert not _consume_verified_source_construction_identity(
        captured_sources[0]
    )


def test_verified_source_cleanup_revokes_identity_after_class_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.estimators.two_point_calibration import (
        _consume_verified_source_construction_identity,
    )
    from odmr_bench.estimators.two_point_types import TwoPointCalibrationSource
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    class MutatedSource(TwoPointCalibrationSource):
        __slots__ = ()

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    successful_fit = make_legal_source_fit(arguments["fit_configuration"])
    original_bind_source = (
        calibration_module._bind_verified_two_point_calibration_source
    )
    captured_sources: list[TwoPointCalibrationSource] = []

    def mutate_source_class(*args: object, **kwargs: object) -> object:
        source = original_bind_source(*args, **kwargs)
        captured_sources.append(source)
        object.__setattr__(source, "__class__", MutatedSource)
        return source

    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: successful_fit,
    )
    monkeypatch.setattr(
        calibration_module,
        "_bind_verified_two_point_calibration_source",
        mutate_source_class,
    )

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "source_binding_failed"
    assert len(captured_sources) == 1
    captured_source = captured_sources[0]
    object.__setattr__(
        captured_source, "__class__", TwoPointCalibrationSource
    )
    assert not _consume_verified_source_construction_identity(captured_source)
    binding = _lookup_run_token_binding(outcome.run_token)
    assert binding is not None
    assert binding.success is None
    assert binding.source is None


def test_success_rollback_ignores_mutated_success_source_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    successful_fit = make_legal_source_fit(arguments["fit_configuration"])
    original_bind_success = calibration_module._bind_run_token_success
    captured_successes: list[VerifiedTwoPointCalibrationSuccess] = []
    captured_sources: list[object] = []

    def commit_mutate_then_fail(
        token: object,
        issuer_runner: object,
        target_instrument: object,
        success: VerifiedTwoPointCalibrationSuccess,
    ) -> None:
        original_bind_success(
            token, issuer_runner, target_instrument, success
        )
        captured_successes.append(success)
        captured_sources.append(success.source)
        object.__setattr__(
            success, "source", make_legal_caller_asserted_source()
        )
        raise RuntimeError("success binding mutated source after commit")

    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: successful_fit,
    )
    monkeypatch.setattr(
        calibration_module,
        "_bind_run_token_success",
        commit_mutate_then_fail,
    )

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "source_binding_failed"
    assert outcome.exception_type == "RuntimeError"
    assert outcome.exception_message == (
        "success binding mutated source after commit"
    )
    assert runner.state.phase == "calibration_failed"
    assert len(captured_successes) == 1
    assert len(captured_sources) == 1
    binding = _lookup_run_token_binding(outcome.run_token)
    assert binding is not None
    assert binding.success is None
    assert binding.source is None


@pytest.mark.parametrize("failure_point", ("before_commit", "after_commit"))
def test_runner_bind_rolls_back_every_provisional_token_identity(
    failure_point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import provenance as provenance_module
    from odmr_bench.evaluation.two_point import runner as runner_module

    instrument = _instrument()
    original_register = runner_module._register_run_token
    captured: list[tuple[object, object, object]] = []

    def fail_registration(
        token: object,
        issuer_runner: object,
        target_instrument: object,
        instrument_configuration: object,
    ) -> None:
        captured.append((token, issuer_runner, instrument_configuration))
        if failure_point == "after_commit":
            original_register(
                token,
                issuer_runner,
                target_instrument,
                instrument_configuration,
            )
        raise RuntimeError(f"registration {failure_point} exploded")

    monkeypatch.setattr(
        runner_module, "_register_run_token", fail_registration
    )

    with pytest.raises(RuntimeError, match=failure_point):
        TwoPointEvaluatorRunner.bind(instrument)

    assert len(captured) == 1
    token, captured_runner, instrument_configuration = captured[0]
    assert _lookup_run_token_binding(token) is None
    assert provenance_module._MINTED_RUN_TOKEN_IDENTITIES.get(id(token)) is not token
    assert captured_runner.state.run_token is token
    assert captured_runner.state.instrument_configuration is instrument_configuration


@pytest.mark.parametrize("stage", ("post_query", "post_source"))
def test_persistent_boundary_fault_terminates_with_bounded_resource_failure(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    class CaptureLoopExceeded(BaseException):
        pass

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    original_query = ODMRInstrument.query
    resources_getter = ODMRInstrument.resources.fget
    time_getter = ODMRInstrument.virtual_time_s.fget
    assert resources_getter is not None
    assert time_getter is not None
    successful_fit = make_legal_source_fit(arguments["fit_configuration"])
    armed = False
    failed_reads = 0
    captured_successes: list[VerifiedTwoPointCalibrationSuccess] = []

    def resources(target: ODMRInstrument) -> ResourceSnapshot:
        nonlocal failed_reads
        if not armed:
            return resources_getter(target)
        failed_reads += 1
        if failed_reads > 4:
            raise CaptureLoopExceeded()
        raise RuntimeError(f"persistent {stage} resources failure")

    def query(
        target: ODMRInstrument, frequency_hz: float, integration_time_s: float
    ) -> object:
        nonlocal armed
        observation = original_query(target, frequency_hz, integration_time_s)
        if stage == "post_query":
            armed = True
        return observation

    original_bind_success = calibration_module._bind_run_token_success

    def bind_success(*args: object, **kwargs: object) -> None:
        nonlocal armed
        original_bind_success(*args, **kwargs)
        success = args[-1]
        assert type(success) is VerifiedTwoPointCalibrationSuccess
        captured_successes.append(success)
        armed = True

    monkeypatch.setattr(ODMRInstrument, "resources", property(resources))
    monkeypatch.setattr(ODMRInstrument, "query", query)
    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: successful_fit,
    )
    if stage == "post_source":
        monkeypatch.setattr(
            calibration_module, "_bind_run_token_success", bind_success
        )

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "resource_join_unavailable"
    assert outcome.resource_mismatch_fields == (
        "observations",
        "integration_time_s",
        "nominal_exposure_photons",
        "expected_photons",
        "realized_photons",
        "observations_without_realized_counts",
        "virtual_elapsed_time_s",
    )
    expected_count = 1 if stage == "post_query" else 2
    assert len(outcome.full_observations) == expected_count
    assert len(outcome.safe_observations) == expected_count
    assert outcome.measurement_midpoints_s == (
        (None,) if stage == "post_query" else (0.0035, 0.0095)
    )
    assert failed_reads == 2
    assert runner.state.phase == "calibration_failed"
    assert runner.state.calibration_outcome is outcome
    trusted_count = 0 if stage == "post_query" else 2
    assert outcome.instrument_resources_after.observations == trusted_count
    assert runner.state.instrument_resources_current is (
        outcome.instrument_resources_after
    )
    assert resources_getter(instrument).observations == expected_count
    assert time_getter(instrument) == pytest.approx(0.006 * expected_count)
    binding = _lookup_run_token_binding(outcome.run_token)
    assert binding is not None
    assert binding.success is None
    assert binding.source is None
    assert len(captured_successes) == (1 if stage == "post_source" else 0)


def test_bind_rollback_revokes_fresh_token_after_binding_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import provenance as provenance_module
    from odmr_bench.evaluation.two_point import runner as runner_module

    instrument = _instrument()
    original_register = runner_module._register_run_token
    captured_tokens: list[object] = []

    def mutate_binding_then_fail(*args: object, **kwargs: object) -> None:
        original_register(*args, **kwargs)
        token = args[0]
        captured_tokens.append(token)
        binding = _lookup_run_token_binding(token)
        assert binding is not None
        object.__setattr__(binding, "issuer_runner", object())
        object.__setattr__(binding, "instrument", object())
        raise RuntimeError("registration binding mutated after commit")

    monkeypatch.setattr(
        runner_module, "_register_run_token", mutate_binding_then_fail
    )

    with pytest.raises(RuntimeError, match="mutated after commit"):
        TwoPointEvaluatorRunner.bind(instrument)

    assert len(captured_tokens) == 1
    token = captured_tokens[0]
    assert _lookup_run_token_binding(token) is None
    assert provenance_module._MINTED_RUN_TOKEN_IDENTITIES.get(id(token)) is not token


def test_success_rollback_restores_trusted_prebind_binding_after_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    successful_fit = make_legal_source_fit(arguments["fit_configuration"])
    original_bind_success = calibration_module._bind_run_token_success

    def mutate_registry_then_fail(*args: object, **kwargs: object) -> None:
        original_bind_success(*args, **kwargs)
        token = args[0]
        binding = _lookup_run_token_binding(token)
        assert binding is not None
        object.__setattr__(binding, "issuer_runner", object())
        object.__setattr__(binding, "instrument", object())
        object.__setattr__(
            binding, "source", make_legal_caller_asserted_source()
        )
        raise RuntimeError("success registry mutated after commit")

    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: successful_fit,
    )
    monkeypatch.setattr(
        calibration_module,
        "_bind_run_token_success",
        mutate_registry_then_fail,
    )

    outcome = runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert type(outcome) is VerifiedTwoPointCalibrationFailure
    assert outcome.failure_code == "source_binding_failed"
    assert outcome.exception_message == "success registry mutated after commit"
    binding = _lookup_run_token_binding(outcome.run_token)
    assert binding is not None
    assert binding.issuer_runner is runner
    assert binding.instrument is instrument
    assert binding.instrument_configuration is runner.state.instrument_configuration
    assert binding.success is None
    assert binding.source is None
    assert runner.state.phase == "calibration_failed"


def test_bind_registration_keyboard_interrupt_rolls_back_then_reraises_same_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.evaluation.two_point import provenance as provenance_module
    from odmr_bench.evaluation.two_point import runner as runner_module

    instrument = _instrument()
    original_register = runner_module._register_run_token
    interruption = KeyboardInterrupt("registration interrupted after commit")
    captured_tokens: list[object] = []

    def commit_then_interrupt(*args: object, **kwargs: object) -> None:
        original_register(*args, **kwargs)
        captured_tokens.append(args[0])
        raise interruption

    monkeypatch.setattr(
        runner_module, "_register_run_token", commit_then_interrupt
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        TwoPointEvaluatorRunner.bind(instrument)

    assert raised.value is interruption
    assert len(captured_tokens) == 1
    token = captured_tokens[0]
    assert _lookup_run_token_binding(token) is None
    assert provenance_module._MINTED_RUN_TOKEN_IDENTITIES.get(id(token)) is not token


def test_success_binding_system_exit_rolls_back_then_reraises_same_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.estimators.two_point_calibration import (
        _consume_verified_source_construction_identity,
    )
    from odmr_bench.evaluation.two_point import calibration as calibration_module

    instrument = _instrument()
    runner = TwoPointEvaluatorRunner.bind(instrument)
    arguments = _valid_calibration_arguments()
    successful_fit = make_legal_source_fit(arguments["fit_configuration"])
    original_bind_success = calibration_module._bind_run_token_success
    interruption = SystemExit(17)
    captured_successes: list[VerifiedTwoPointCalibrationSuccess] = []

    def commit_then_exit(*args: object, **kwargs: object) -> None:
        original_bind_success(*args, **kwargs)
        success = args[-1]
        assert type(success) is VerifiedTwoPointCalibrationSuccess
        captured_successes.append(success)
        raise interruption

    monkeypatch.setattr(
        calibration_module,
        "fit_spectrum",
        lambda sweep, configuration, initial_guess=None: successful_fit,
    )
    monkeypatch.setattr(
        calibration_module, "_bind_run_token_success", commit_then_exit
    )

    with pytest.raises(SystemExit) as raised:
        runner.acquire_verified_calibration(**arguments)  # type: ignore[arg-type]

    assert raised.value is interruption
    assert len(captured_successes) == 1
    success = captured_successes[0]
    binding = _lookup_run_token_binding(runner.state.run_token)
    assert binding is not None
    assert binding.issuer_runner is runner
    assert binding.instrument is instrument
    assert binding.success is None
    assert binding.source is None
    assert not _consume_verified_source_construction_identity(success.source)
