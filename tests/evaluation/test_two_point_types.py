"""Tests for evaluator-owned calibrated two-point primitive contracts."""

import copy
import json
import pickle
from dataclasses import FrozenInstanceError, asdict, astuple, dataclass, fields, replace
from typing import get_args

import numpy as np
import pytest


def test_evaluator_primitive_names_are_public() -> None:
    from odmr_bench.evaluation import two_point
    from odmr_bench.evaluation.two_point import (
        ResourceJoinMismatchField,
        TwoPointCalibrationPreflightError,
        TwoPointEvaluatorInstrumentConfiguration,
        TwoPointRunnerStartError,
        TwoPointRunnerStateError,
        VerifiedCalibrationQueryRequest,
        VerifiedInstrumentRunToken,
    )

    assert tuple(two_point.__all__) == (
        "ResourceJoinMismatchField",
        "TwoPointCalibrationPreflightError",
        "TwoPointEvaluatorInstrumentConfiguration",
        "TwoPointRunnerStartError",
        "TwoPointRunnerStateError",
        "VerifiedCalibrationQueryRequest",
        "VerifiedInstrumentRunToken",
    )
    assert get_args(ResourceJoinMismatchField) == (
        "observations",
        "integration_time_s",
        "nominal_exposure_photons",
        "expected_photons",
        "realized_photons",
        "observations_without_realized_counts",
        "virtual_elapsed_time_s",
    )
    forbidden_later_names = (
        "TwoPointAbortedRun",
        "TwoPointEvaluatorPairTiming",
        "TwoPointEvaluatorResources",
        "TwoPointEvaluatorRunner",
        "TwoPointEvaluatorRunnerState",
        "TwoPointInstrumentQueryFailure",
        "TwoPointResourceJoinUnavailableAcquisition",
        "TwoPointRunnerAborted",
        "TwoPointRunnerAccepted",
        "TwoPointRunnerBudgetStopped",
        "TwoPointRunnerExternallyStopped",
        "TwoPointRunnerInstrumentFailure",
        "TwoPointRunnerRunOutcome",
        "TwoPointRunnerStepOutcome",
        "TwoPointTrackingAcquisition",
        "VerifiedTwoPointCalibrationFailure",
        "VerifiedTwoPointCalibrationOutcome",
        "VerifiedTwoPointCalibrationSuccess",
        "build_two_point_evaluator_resources",
    )
    assert not any(hasattr(two_point, name) for name in forbidden_later_names)

    assert TwoPointCalibrationPreflightError
    assert TwoPointEvaluatorInstrumentConfiguration
    assert TwoPointRunnerStartError
    assert TwoPointRunnerStateError
    assert VerifiedCalibrationQueryRequest
    assert VerifiedInstrumentRunToken


def test_verified_token_is_opaque_identity_capability() -> None:
    from odmr_bench.evaluation import two_point
    from odmr_bench.evaluation.two_point import VerifiedInstrumentRunToken, provenance

    with pytest.raises(TypeError):
        VerifiedInstrumentRunToken()
    with pytest.raises(TypeError):
        provenance._mint_verified_instrument_run_token(object())

    token = provenance._mint_verified_instrument_run_token(
        provenance._TOKEN_CONSTRUCTION_KEY
    )
    other = provenance._mint_verified_instrument_run_token(
        provenance._TOKEN_CONSTRUCTION_KEY
    )
    arbitrary_allocation = object.__new__(VerifiedInstrumentRunToken)

    assert type(token) is VerifiedInstrumentRunToken
    assert type(other) is VerifiedInstrumentRunToken
    assert token is not other
    assert token != other
    assert token == token
    assert type(arbitrary_allocation) is VerifiedInstrumentRunToken
    assert arbitrary_allocation is not token
    assert arbitrary_allocation is not other
    assert arbitrary_allocation != token
    assert type(token).__eq__ is object.__eq__
    assert type(token).__hash__ is object.__hash__
    assert "__eq__" not in VerifiedInstrumentRunToken.__dict__
    assert VerifiedInstrumentRunToken.__slots__ == ()
    assert not hasattr(token, "__dict__")
    assert not hasattr(token, "value")
    assert not hasattr(arbitrary_allocation, "value")
    assert not hasattr(arbitrary_allocation, "issuer")
    assert "issuer" not in repr(token).lower()
    assert "runner" not in repr(token).lower()
    assert "issuer" not in repr(arbitrary_allocation).lower()
    assert "runner" not in repr(arbitrary_allocation).lower()
    assert not hasattr(two_point, "_TOKEN_CONSTRUCTION_KEY")
    assert not hasattr(two_point, "_mint_verified_instrument_run_token")

    with pytest.raises(TypeError):
        class ForgedVerifiedInstrumentRunToken(VerifiedInstrumentRunToken):
            def __new__(cls) -> object:
                return object.__new__(cls)

    @dataclass(frozen=True)
    class TokenHolder:
        token: VerifiedInstrumentRunToken

    holder = TokenHolder(token)
    with pytest.raises(TypeError):
        asdict(holder)
    with pytest.raises(TypeError):
        astuple(holder)

    for candidate in (token, arbitrary_allocation):
        with pytest.raises(TypeError):
            copy.copy(candidate)
        with pytest.raises(TypeError):
            copy.deepcopy(candidate)
        with pytest.raises(TypeError):
            pickle.dumps(candidate)
        with pytest.raises(TypeError):
            json.dumps(candidate)
        with pytest.raises(TypeError):
            candidate.__reduce__()
        with pytest.raises(TypeError):
            candidate.__reduce_ex__(pickle.HIGHEST_PROTOCOL)


def test_evaluator_primitives_validate_intrinsic_matrix() -> None:
    from odmr_bench.evaluation.two_point import (
        TwoPointCalibrationPreflightError,
        TwoPointEvaluatorInstrumentConfiguration,
        TwoPointRunnerStartError,
        TwoPointRunnerStateError,
        VerifiedCalibrationQueryRequest,
    )
    from odmr_bench.evaluation.two_point.types import (
        TwoPointRunnerStartFailureCode,
        VerifiedCalibrationPreflightCode,
    )

    configuration = TwoPointEvaluatorInstrumentConfiguration(2.5e6, 0.001)
    request = VerifiedCalibrationQueryRequest(
        0,
        2.87e9,
        0.5,
        0,
        0.75,
        1.0,
        1.25e6,
    )

    assert tuple(field.name for field in fields(configuration)) == (
        "nominal_photon_rate_hz",
        "frequency_overhead_s",
    )
    assert tuple(field.name for field in fields(request)) == (
        "point_index",
        "frequency_hz",
        "integration_time_s",
        "expected_sequence_index",
        "expected_measurement_midpoint_s",
        "expected_end_timestamp_s",
        "expected_nominal_exposure_photons",
    )
    assert configuration.__slots__ == (
        "nominal_photon_rate_hz",
        "frequency_overhead_s",
    )
    assert request.__slots__ == tuple(field.name for field in fields(request))
    assert type(configuration.nominal_photon_rate_hz) is float
    assert type(configuration.frequency_overhead_s) is float
    assert type(request.point_index) is int
    assert type(request.frequency_hz) is float
    assert type(request.integration_time_s) is float
    assert type(request.expected_sequence_index) is int
    assert type(request.expected_measurement_midpoint_s) is float
    assert type(request.expected_end_timestamp_s) is float
    assert type(request.expected_nominal_exposure_photons) is float
    with pytest.raises(FrozenInstanceError):
        configuration.nominal_photon_rate_hz = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        request.point_index = 1  # type: ignore[misc]

    invalid_configuration_rows = (
        ("nominal_photon_rate_hz", True, TypeError),
        ("nominal_photon_rate_hz", 0.0, ValueError),
        ("nominal_photon_rate_hz", float("inf"), ValueError),
        ("nominal_photon_rate_hz", 1.0 + 0.0j, TypeError),
        ("frequency_overhead_s", np.bool_(False), TypeError),
        ("frequency_overhead_s", -1.0, ValueError),
        ("frequency_overhead_s", float("nan"), ValueError),
        ("frequency_overhead_s", np.array(0.0), TypeError),
    )
    for field_name, value, error_type in invalid_configuration_rows:
        with pytest.raises(error_type):
            replace(configuration, **{field_name: value})

    invalid_request_rows = (
        ("point_index", True, TypeError),
        ("point_index", -1, ValueError),
        ("point_index", 0.0, TypeError),
        ("frequency_hz", False, TypeError),
        ("frequency_hz", 0.0, ValueError),
        ("frequency_hz", float("inf"), ValueError),
        ("integration_time_s", np.bool_(True), TypeError),
        ("integration_time_s", 0.0, ValueError),
        ("integration_time_s", float("nan"), ValueError),
        ("expected_sequence_index", True, TypeError),
        ("expected_sequence_index", -1, ValueError),
        ("expected_sequence_index", 0.0, TypeError),
        ("expected_measurement_midpoint_s", False, TypeError),
        ("expected_measurement_midpoint_s", -1.0, ValueError),
        ("expected_measurement_midpoint_s", float("inf"), ValueError),
        ("expected_end_timestamp_s", np.bool_(False), TypeError),
        ("expected_end_timestamp_s", -1.0, ValueError),
        ("expected_end_timestamp_s", float("nan"), ValueError),
        ("expected_nominal_exposure_photons", True, TypeError),
        ("expected_nominal_exposure_photons", -1.0, ValueError),
        ("expected_nominal_exposure_photons", float("inf"), ValueError),
    )
    for field_name, value, error_type in invalid_request_rows:
        with pytest.raises(error_type):
            replace(request, **{field_name: value})
    with pytest.raises(ValueError, match="midpoint"):
        replace(request, expected_measurement_midpoint_s=1.1)
    with pytest.raises(ValueError, match="integration"):
        replace(
            request,
            expected_measurement_midpoint_s=0.125,
            expected_end_timestamp_s=0.25,
        )

    numpy_configuration = TwoPointEvaluatorInstrumentConfiguration(
        np.float64(2.5e6), np.float64(0.001)
    )
    numpy_request = VerifiedCalibrationQueryRequest(
        np.int64(0),
        np.float64(2.87e9),
        np.float64(0.5),
        np.int64(0),
        np.float64(0.75),
        np.float64(1.0),
        np.float64(1.25e6),
    )
    assert type(numpy_configuration.nominal_photon_rate_hz) is float
    assert type(numpy_configuration.frequency_overhead_s) is float
    assert type(numpy_request.point_index) is int
    assert type(numpy_request.frequency_hz) is float
    assert type(numpy_request.integration_time_s) is float
    assert type(numpy_request.expected_sequence_index) is int
    assert type(numpy_request.expected_measurement_midpoint_s) is float
    assert type(numpy_request.expected_end_timestamp_s) is float
    assert type(numpy_request.expected_nominal_exposure_photons) is float

    error_contracts = (
        (
            TwoPointCalibrationPreflightError,
            VerifiedCalibrationPreflightCode,
            (
                "invalid_runner_phase",
                "invalid_argument_type",
                "invalid_argument_value",
                "invalid_frequency_grid",
                "invalid_fit_or_identity_configuration",
                "invalid_clock_mapping",
                "unclean_instrument_boundary",
            ),
        ),
        (
            TwoPointRunnerStartError,
            TwoPointRunnerStartFailureCode,
            (
                "invalid_runner_phase",
                "invalid_argument_type",
                "unverified_calibration",
                "calibration_mismatch",
                "run_provenance_mismatch",
                "metadata_mismatch",
                "resource_boundary_mismatch",
                "tracker_reset_failed",
            ),
        ),
    )
    for error_class, code_alias, codes in error_contracts:
        assert get_args(code_alias) == codes
        for code in codes:
            error = error_class(code)
            assert error.code == code
            assert error.args == (code,)
        with pytest.raises(ValueError):
            error_class("unknown")
        with pytest.raises(TypeError):
            error_class(True)

    state_error = TwoPointRunnerStateError("runner is not tracking")
    assert str(state_error) == "runner is not tracking"
