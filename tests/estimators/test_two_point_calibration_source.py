"""Tests for caller-asserted two-point calibration source binding."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import odmr_bench.estimators.two_point_calibration as calibration_module
from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators import (
    CompleteSweep,
    FitConfiguration,
    NormalizedFluorescenceProvenance,
    SpectrumFitResult,
    TwoPointCalibrationConstructionError,
    TwoPointClockMapping,
    TwoPointIdentityBinding,
    bind_caller_asserted_two_point_calibration_source,
)
from odmr_bench.estimators.two_point_resources import _replay_public_resources
from tests.two_point_helpers import (
    make_legal_fit_configuration,
    make_legal_source_fit,
    make_legal_source_observations,
)

FactoryArguments = dict[str, Any]
Defect = Callable[[FactoryArguments], None]


class _StringSubclass(str):
    pass


def _factory_arguments() -> FactoryArguments:
    fit_configuration = make_legal_fit_configuration()
    observations = list(make_legal_source_observations())
    first_midpoint_s = (
        observations[0].timestamp_s - observations[0].integration_time_s / 2.0
    )
    last_midpoint_s = (
        observations[-1].timestamp_s - observations[-1].integration_time_s / 2.0
    )
    return {
        "source_fit": make_legal_source_fit(fit_configuration),
        "fit_configuration": fit_configuration,
        "source_observations": observations,
        "identity_binding": TwoPointIdentityBinding("adopt_fit_ids", None),
        "fluorescence_provenance": NormalizedFluorescenceProvenance(
            "normalized_fluorescence", "declared", 2.5e6, ("declared",)
        ),
        "source_id": "caller-source",
        "source_frequency_overhead_s": 0.0,
        "source_start_timestamp_s": 0.0,
        "physical_fit_epoch_s": (
            first_midpoint_s + (last_midpoint_s - first_midpoint_s) / 2.0
        ),
        "availability_sequence_index": observations[-1].sequence_index,
        "availability_timestamp_s": observations[-1].timestamp_s,
        "clock_mapping": TwoPointClockMapping(
            "shared_clock", "source-clock", "source-clock", 1.0, 0.0
        ),
    }


def _invalid_argument_type(arguments: FactoryArguments) -> None:
    arguments["availability_sequence_index"] = 1.0


def _invalid_argument_value(arguments: FactoryArguments) -> None:
    arguments["source_id"] = ""


def _invalid_provenance(arguments: FactoryArguments) -> None:
    provenance = arguments["fluorescence_provenance"]
    object.__setattr__(provenance, "quantity", "raw_voltage")


def _invalid_source_trace(arguments: FactoryArguments) -> None:
    observations = list(arguments["source_observations"])
    observations[-1] = replace(observations[-1], timestamp_s=0.011)
    arguments["source_observations"] = observations


def _source_resource_mismatch(arguments: FactoryArguments) -> None:
    observations = list(arguments["source_observations"])
    observations[0] = replace(
        observations[0], nominal_exposure_photons=12_499.0
    )
    arguments["source_observations"] = observations


def _fit_input_mismatch(arguments: FactoryArguments) -> None:
    arguments["fit_configuration"] = FitConfiguration(model_kind="lorentzian")


def _source_fit_failed(arguments: FactoryArguments) -> None:
    source_fit: SpectrumFitResult = arguments["source_fit"]
    arguments["source_fit"] = replace(
        source_fit,
        success=False,
        failure_code="quality_failed",
        resonance_estimates=(),
        baseline_estimate=None,
    )


def _source_identity_mismatch(arguments: FactoryArguments) -> None:
    arguments["identity_binding"] = TwoPointIdentityBinding(
        "require_expected_ids", tuple(f"expected-{index}" for index in range(8))
    )


def _invalid_source_epoch(arguments: FactoryArguments) -> None:
    arguments["physical_fit_epoch_s"] = math.nextafter(
        arguments["physical_fit_epoch_s"], math.inf
    )


def _invalid_availability_or_clock(arguments: FactoryArguments) -> None:
    arguments["availability_sequence_index"] = 0


_ORDERED_DEFECTS: tuple[tuple[str, Defect], ...] = (
    ("invalid_argument_type", _invalid_argument_type),
    ("invalid_argument_value", _invalid_argument_value),
    ("invalid_provenance_or_quantity", _invalid_provenance),
    ("invalid_source_trace", _invalid_source_trace),
    ("source_resource_mismatch", _source_resource_mismatch),
    ("fit_input_mismatch", _fit_input_mismatch),
    ("source_fit_failed", _source_fit_failed),
    ("source_identity_mismatch", _source_identity_mismatch),
    ("invalid_source_epoch", _invalid_source_epoch),
    ("invalid_availability_or_clock", _invalid_availability_or_clock),
)


def test_caller_asserted_factory_binds_exact_public_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_configuration = make_legal_fit_configuration()
    source_fit = make_legal_source_fit(fit_configuration)
    supplied_observations = list(make_legal_source_observations())
    identity_binding = TwoPointIdentityBinding("adopt_fit_ids", None)
    fluorescence_provenance = NormalizedFluorescenceProvenance(
        "normalized_fluorescence",
        "caller-declared normalized fluorescence",
        2.5e6,
        ("caller-declared",),
    )
    clock_mapping = TwoPointClockMapping(
        "unit_scale_offset", "source-clock", "tracker-clock", 1.0, -0.25
    )
    first_midpoint_s = (
        supplied_observations[0].timestamp_s
        - supplied_observations[0].integration_time_s / 2.0
    )
    last_midpoint_s = (
        supplied_observations[-1].timestamp_s
        - supplied_observations[-1].integration_time_s / 2.0
    )
    physical_fit_epoch_s = (
        first_midpoint_s + (last_midpoint_s - first_midpoint_s) / 2.0
    )
    fit_inputs: list[CompleteSweep] = []

    def record_fit_input(*args: Any, **kwargs: Any) -> CompleteSweep:
        fit_input = CompleteSweep(*args, **kwargs)
        fit_inputs.append(fit_input)
        return fit_input

    monkeypatch.setattr(calibration_module, "CompleteSweep", record_fit_input)

    source = bind_caller_asserted_two_point_calibration_source(
        source_fit,
        fit_configuration,
        supplied_observations,
        identity_binding,
        fluorescence_provenance,
        source_id="caller-source",
        source_frequency_overhead_s=0.0,
        source_start_timestamp_s=0.0,
        physical_fit_epoch_s=physical_fit_epoch_s,
        availability_sequence_index=1,
        availability_timestamp_s=0.010,
        clock_mapping=clock_mapping,
    )

    supplied_observations.clear()
    assert source.provenance == "caller_asserted"
    assert source.source_id == "caller-source"
    assert source.source_fit is not source_fit
    assert source.source_fit.model_kind == source_fit.model_kind
    assert source.source_fit.baseline_degree == source_fit.baseline_degree
    assert source.source_fit.resonance_estimates == source_fit.resonance_estimates
    assert source.source_fit.baseline_estimate == source_fit.baseline_estimate
    assert source.fit_configuration == fit_configuration
    assert source.fit_configuration is not fit_configuration
    assert source.identity_binding == identity_binding
    assert source.identity_binding is not identity_binding
    assert source.fluorescence_provenance == fluorescence_provenance
    assert source.fluorescence_provenance is not fluorescence_provenance
    assert source.clock_mapping == clock_mapping
    assert source.clock_mapping is not clock_mapping
    assert source.source_observations == make_legal_source_observations()
    assert source.source_frequency_min_hz == source.source_observations[0].frequency_hz
    assert source.source_frequency_max_hz == source.source_observations[-1].frequency_hz
    assert source.source_first_sequence_index == 0
    assert source.source_last_sequence_index == 1
    assert source.source_first_timestamp_s == 0.005
    assert source.source_last_timestamp_s == 0.010
    assert source.resolved_resonance_ids == tuple(f"r{index}" for index in range(8))
    assert source.identity_binding.mode == "adopt_fit_ids"
    assert source.safe_resources == _replay_public_resources(
        source.source_observations, 0.0
    )

    assert len(fit_inputs) == 1
    fit_input = fit_inputs[0]
    np.testing.assert_array_equal(
        fit_input.frequency_hz,
        [item.frequency_hz for item in source.source_observations],
    )
    np.testing.assert_array_equal(
        fit_input.fluorescence,
        [item.fluorescence for item in source.source_observations],
    )
    assert fit_input.last_sequence_index == source.source_last_sequence_index
    assert fit_input.last_timestamp_s == source.source_last_timestamp_s
    assert (
        fit_input.total_integration_time_s
        == source.safe_resources.integration_time_s
    )
    assert (
        fit_input.total_nominal_exposure_photons
        == source.safe_resources.nominal_exposure_photons
    )


@pytest.mark.parametrize(
    ("earlier_index", "expected_code"),
    tuple((index, code) for index, (code, _) in enumerate(_ORDERED_DEFECTS[:-1])),
)
def test_caller_asserted_factory_construction_code_precedence(
    earlier_index: int,
    expected_code: str,
) -> None:
    arguments = _factory_arguments()
    _ORDERED_DEFECTS[earlier_index][1](arguments)
    _ORDERED_DEFECTS[earlier_index + 1][1](arguments)

    with pytest.raises(TwoPointCalibrationConstructionError) as raised:
        bind_caller_asserted_two_point_calibration_source(**arguments)

    assert raised.value.code == expected_code
    assert raised.value.message


def _exact_epoch_witness_arguments() -> tuple[FactoryArguments, float, float, float]:
    integration_time_s = 0.005
    overhead_s = 0.001
    endpoint_s = 0.0
    observations: list[EstimatorObservation] = []
    for index in range(4_481):
        endpoint_s = (endpoint_s + overhead_s) + integration_time_s
        observations.append(
            EstimatorObservation(
                sequence_index=index,
                timestamp_s=endpoint_s,
                frequency_hz=2.740e9 + index * 62_500.0,
                fluorescence=1.0,
                integration_time_s=integration_time_s,
                nominal_exposure_photons=12_500.0,
            )
        )
    first_public_midpoint_s = (
        observations[0].timestamp_s - observations[0].integration_time_s / 2.0
    )
    last_public_midpoint_s = (
        observations[-1].timestamp_s
        - observations[-1].integration_time_s / 2.0
    )
    physical_fit_epoch_s = first_public_midpoint_s + (
        last_public_midpoint_s - first_public_midpoint_s
    ) / 2.0
    arguments = _factory_arguments()
    arguments.update(
        source_observations=observations,
        source_frequency_overhead_s=overhead_s,
        physical_fit_epoch_s=physical_fit_epoch_s,
        availability_sequence_index=4_480,
        availability_timestamp_s=observations[-1].timestamp_s,
    )
    return (
        arguments,
        first_public_midpoint_s,
        last_public_midpoint_s,
        physical_fit_epoch_s,
    )


def test_caller_asserted_epoch_requires_exact_public_midpoint_mean() -> None:
    arguments, first_midpoint_s, last_midpoint_s, exact_epoch_s = (
        _exact_epoch_witness_arguments()
    )
    assert exact_epoch_s.hex() == "0x1.ae3126e978e26p+3"
    source = bind_caller_asserted_two_point_calibration_source(**arguments)
    assert source.physical_fit_epoch_s.hex() == "0x1.ae3126e978e26p+3"

    rejected_epochs = (
        math.nextafter(exact_epoch_s, -math.inf),
        math.nextafter(exact_epoch_s, math.inf),
        first_midpoint_s,
        last_midpoint_s,
        math.nextafter(first_midpoint_s, -math.inf),
        math.nextafter(last_midpoint_s, math.inf),
    )
    for rejected_epoch_s in rejected_epochs:
        invalid_arguments = dict(arguments)
        invalid_arguments["physical_fit_epoch_s"] = rejected_epoch_s
        with pytest.raises(TwoPointCalibrationConstructionError) as raised:
            bind_caller_asserted_two_point_calibration_source(**invalid_arguments)
        assert raised.value.code == "invalid_source_epoch"


def test_caller_asserted_long_trace_resources_are_replace_stable() -> None:
    arguments, _, _, _ = _exact_epoch_witness_arguments()
    source = bind_caller_asserted_two_point_calibration_source(**arguments)
    expected_resources = _replay_public_resources(
        source.source_observations, source.source_frequency_overhead_s
    )
    assert source.safe_resources == expected_resources

    reconstructed = replace(source)
    assert reconstructed is not source
    assert repr(reconstructed) == repr(source)
    assert reconstructed.safe_resources == expected_resources

    wrong_resources = replace(
        source.safe_resources,
        observations=source.safe_resources.observations + 1,
    )
    with pytest.raises(ValueError, match="safe_resources"):
        replace(source, safe_resources=wrong_resources)


@pytest.mark.parametrize(
    ("malformation", "expected_code"),
    (
        ("invalid_identity_mode", "source_identity_mismatch"),
        ("adopt_mode_with_expected_ids", "source_identity_mismatch"),
        ("nonnumeric_clock_offset", "invalid_availability_or_clock"),
        ("successful_fit_without_initial_guess", "source_fit_failed"),
        ("malformed_initial_guess_resonances", "source_fit_failed"),
        ("malformed_initial_guess_resonance_id", "source_fit_failed"),
        ("mismatched_string_subclass_fit_ids", "fit_input_mismatch"),
        ("nonscalar_source_fit_model_kind", "fit_input_mismatch"),
        ("string_observation_integration_time", "invalid_source_trace"),
    ),
)
def test_caller_asserted_factory_closes_malformed_exact_record_boundary(
    malformation: str,
    expected_code: str,
) -> None:
    arguments = _factory_arguments()
    if malformation == "invalid_identity_mode":
        object.__setattr__(arguments["identity_binding"], "mode", "invalid")
    elif malformation == "adopt_mode_with_expected_ids":
        object.__setattr__(
            arguments["identity_binding"],
            "expected_resonance_ids",
            tuple(f"r{index}" for index in range(8)),
        )
    elif malformation == "nonnumeric_clock_offset":
        object.__setattr__(arguments["clock_mapping"], "offset_s", "zero")
    elif malformation == "successful_fit_without_initial_guess":
        object.__setattr__(arguments["source_fit"], "initial_guess", None)
    elif malformation == "malformed_initial_guess_resonances":
        object.__setattr__(
            arguments["source_fit"].initial_guess,
            "resonances",
            None,
        )
    elif malformation == "malformed_initial_guess_resonance_id":
        object.__setattr__(
            arguments["source_fit"].initial_guess.resonances[0],
            "resonance_id",
            np.array(["r0", "r0-copy"]),
        )
    elif malformation == "mismatched_string_subclass_fit_ids":
        changed_id = _StringSubclass("changed-r0")
        object.__setattr__(
            arguments["source_fit"].initial_guess.resonances[0],
            "resonance_id",
            changed_id,
        )
        object.__setattr__(
            arguments["source_fit"].resonance_estimates[0],
            "resonance_id",
            changed_id,
        )
    elif malformation == "nonscalar_source_fit_model_kind":
        object.__setattr__(
            arguments["source_fit"],
            "model_kind",
            np.array(["pseudo_voigt", "lorentzian"]),
        )
    else:
        observations = arguments["source_observations"]
        object.__setattr__(observations[0], "integration_time_s", "0.005")

    with pytest.raises(TwoPointCalibrationConstructionError) as raised:
        bind_caller_asserted_two_point_calibration_source(**arguments)

    assert raised.value.code == expected_code
    assert raised.value.message


def test_fit_input_id_mismatch_precedes_malformed_source_fit() -> None:
    arguments = _factory_arguments()
    arguments["fit_configuration"] = make_legal_fit_configuration(
        tuple(f"configured-{index}" for index in range(8))
    )
    source_fit = arguments["source_fit"]
    assert source_fit.model_kind == arguments["fit_configuration"].model_kind
    assert (
        source_fit.baseline_degree
        == arguments["fit_configuration"].baseline_degree
    )
    assert tuple(
        resonance.resonance_id
        for resonance in source_fit.resonance_estimates
    ) != arguments["fit_configuration"].resonance_ids
    object.__setattr__(source_fit, "initial_guess", None)

    with pytest.raises(TwoPointCalibrationConstructionError) as raised:
        bind_caller_asserted_two_point_calibration_source(**arguments)

    assert raised.value.code == "fit_input_mismatch"
    assert raised.value.message
