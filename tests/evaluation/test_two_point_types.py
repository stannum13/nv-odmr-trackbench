"""Tests for evaluator-owned calibrated two-point primitive contracts."""

import ast
import copy
import inspect
import json
import math
import pickle
from dataclasses import (
    MISSING,
    FrozenInstanceError,
    asdict,
    astuple,
    dataclass,
    fields,
    replace,
)
from pathlib import Path
from typing import get_args

import numpy as np
import pytest


def _capability_record_subclass(value: object) -> object:
    capable_type = type(
        f"CapabilityBearing{type(value).__name__}",
        (type(value),),
        {"__slots__": ("slot_capability", "__dict__")},
    )
    candidate = object.__new__(capable_type)
    for record_field in fields(value):
        object.__setattr__(
            candidate,
            record_field.name,
            getattr(value, record_field.name),
        )
    object.__setattr__(candidate, "callback", lambda: None)
    object.__setattr__(candidate, "payload", {"retained": object()})
    object.__setattr__(candidate, "slot_capability", object())
    return candidate


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
        "TwoPointEvaluatorPairTiming",
        "TwoPointInstrumentQueryFailure",
        "TwoPointResourceJoinUnavailableAcquisition",
        "TwoPointRunnerStartError",
        "TwoPointRunnerStateError",
        "TwoPointTrackingAcquisition",
        "VerifiedCalibrationQueryRequest",
        "VerifiedInstrumentRunToken",
        "VerifiedTwoPointCalibrationFailure",
        "VerifiedTwoPointCalibrationOutcome",
        "VerifiedTwoPointCalibrationSuccess",
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
        "TwoPointEvaluatorResources",
        "TwoPointEvaluatorRunner",
        "TwoPointEvaluatorRunnerState",
        "TwoPointRunnerAborted",
        "TwoPointRunnerAccepted",
        "TwoPointRunnerBudgetStopped",
        "TwoPointRunnerExternallyStopped",
        "TwoPointRunnerInstrumentFailure",
        "TwoPointRunnerRunOutcome",
        "TwoPointRunnerStepOutcome",
        "build_two_point_evaluator_resources",
    )
    assert not any(hasattr(two_point, name) for name in forbidden_later_names)

    assert TwoPointCalibrationPreflightError
    assert TwoPointEvaluatorInstrumentConfiguration
    assert TwoPointRunnerStartError
    assert TwoPointRunnerStateError
    assert VerifiedCalibrationQueryRequest
    assert VerifiedInstrumentRunToken


def test_calibration_outcome_and_acquisition_names_are_public() -> None:
    from odmr_bench.evaluation import two_point
    from odmr_bench.evaluation.two_point import (
        TwoPointEvaluatorPairTiming,
        TwoPointInstrumentQueryFailure,
        TwoPointResourceJoinUnavailableAcquisition,
        TwoPointTrackingAcquisition,
        VerifiedTwoPointCalibrationFailure,
        VerifiedTwoPointCalibrationOutcome,
        VerifiedTwoPointCalibrationSuccess,
    )

    introduced_names = (
        "TwoPointEvaluatorPairTiming",
        "TwoPointInstrumentQueryFailure",
        "TwoPointResourceJoinUnavailableAcquisition",
        "TwoPointTrackingAcquisition",
        "VerifiedTwoPointCalibrationFailure",
        "VerifiedTwoPointCalibrationOutcome",
        "VerifiedTwoPointCalibrationSuccess",
    )
    assert tuple(
        name for name in two_point.__all__ if name in introduced_names
    ) == introduced_names
    success_fields = tuple(
        field.name for field in fields(VerifiedTwoPointCalibrationSuccess)
    )
    assert success_fields == (
        "status",
        "run_token",
        "source",
        "full_observations",
        "safe_observations",
        "measurement_midpoints_s",
        "instrument_resources_before",
        "instrument_resources_after",
        "safe_resources",
        "full_resources",
    )
    failure_fields = tuple(
        field.name for field in fields(VerifiedTwoPointCalibrationFailure)
    )
    assert failure_fields == (
        "status",
        "run_token",
        "failure_code",
        "failed_request",
        "exception_type",
        "exception_message",
        "fit_result",
        "resource_mismatch_fields",
        "full_observations",
        "safe_observations",
        "measurement_midpoints_s",
        "instrument_resources_before",
        "instrument_resources_after",
        "safe_resources",
        "full_resources",
    )
    assert tuple(field.name for field in fields(TwoPointTrackingAcquisition)) == (
        "resource_join_status",
        "query",
        "expected_measurement_midpoint_s",
        "measurement_midpoint_s",
        "full_observation",
        "safe_observation",
        "instrument_resources_before",
        "instrument_resources_after",
        "instrument_resource_delta",
    )
    assert tuple(
        field.name for field in fields(TwoPointResourceJoinUnavailableAcquisition)
    ) == (
        "resource_join_status",
        "query",
        "expected_measurement_midpoint_s",
        "measurement_midpoint_s",
        "full_observation",
        "safe_observation",
        "resource_mismatch_fields",
        "instrument_resources_before",
        "instrument_resources_after",
    )
    assert tuple(field.name for field in fields(TwoPointEvaluatorPairTiming)) == (
        "pair_index",
        "resonance_id",
        "first_measurement_midpoint_s",
        "second_measurement_midpoint_s",
        "truth_reference_timestamp_s",
        "public_reference_timestamp_s",
        "release_sequence_index",
        "release_timestamp_s",
    )
    assert tuple(field.name for field in fields(TwoPointInstrumentQueryFailure)) == (
        "query",
        "exception_type",
        "exception_message",
        "instrument_resources_before",
        "instrument_resources_after",
    )
    assert get_args(VerifiedTwoPointCalibrationOutcome) == (
        VerifiedTwoPointCalibrationSuccess,
        VerifiedTwoPointCalibrationFailure,
    )
    record_classes = (
        VerifiedTwoPointCalibrationSuccess,
        VerifiedTwoPointCalibrationFailure,
        TwoPointTrackingAcquisition,
        TwoPointResourceJoinUnavailableAcquisition,
        TwoPointEvaluatorPairTiming,
        TwoPointInstrumentQueryFailure,
    )
    for record_class in record_classes:
        assert record_class.__slots__ == tuple(
            field.name for field in fields(record_class)
        )


def _calibration_outcome_arguments(
    failure_code: str | None,
) -> dict[str, object]:
    from odmr_bench.emulator import InstrumentObservation, ResourceSnapshot
    from odmr_bench.estimators import PublicAcquisitionResources
    from odmr_bench.evaluation.two_point import (
        VerifiedCalibrationQueryRequest,
        VerifiedInstrumentRunToken,
    )
    from tests.two_point_helpers import (
        make_legal_caller_asserted_source,
        make_legal_source_fit,
    )

    full_observation = InstrumentObservation(
        sequence_index=0,
        timestamp_s=0.006,
        frequency_hz=2.76e9,
        fluorescence=0.98,
        integration_time_s=0.005,
        nominal_exposure_photons=12_500.0,
        expected_photons=12_000.0,
        realized_photons=12_250,
        sampling_rule="test-rule",
    )
    full_resources = ResourceSnapshot(
        1, 0.005, 12_500.0, 12_000.0, 12_250, 0, 0.006
    )
    common: dict[str, object] = {
        "status": "success" if failure_code is None else "failure",
        "run_token": object.__new__(VerifiedInstrumentRunToken),
        "full_observations": (full_observation,),
        "safe_observations": (full_observation.estimator_view(),),
        "measurement_midpoints_s": (0.0035,),
        "instrument_resources_before": ResourceSnapshot(
            0, 0.0, 0.0, 0.0, 0, 0, 0.0
        ),
        "instrument_resources_after": full_resources,
        "safe_resources": PublicAcquisitionResources(
            1, 0.005, 12_500.0, 12_250, 0, 0.006
        ),
        "full_resources": full_resources,
    }
    if failure_code is None:
        common["source"] = make_legal_caller_asserted_source()
        return common

    exception_codes = {
        "instrument_query_failed",
        "fit_exception",
        "source_binding_failed",
    }
    common.update(
        failure_code=failure_code,
        failed_request=(
            VerifiedCalibrationQueryRequest(
                1, 2.77e9, 0.005, 1, 0.0095, 0.012, 12_500.0
            )
            if failure_code
            in {
                "instrument_query_failed",
                "resource_join_unavailable",
                "acquisition_contract_mismatch",
            }
            else None
        ),
        exception_type="RuntimeError" if failure_code in exception_codes else None,
        exception_message="boom" if failure_code in exception_codes else None,
        fit_result=(
            replace(
                make_legal_source_fit(),
                success=False,
                failure_code="quality_failed",
                resonance_estimates=(),
                baseline_estimate=None,
            )
            if failure_code == "fit_failed"
            else make_legal_source_fit()
            if failure_code == "source_binding_failed"
            else None
        ),
        resource_mismatch_fields=(
            ("expected_photons",)
            if failure_code == "resource_join_unavailable"
            else ()
        ),
    )
    if failure_code == "resource_join_unavailable":
        common["safe_resources"] = None
        common["full_resources"] = None
    return common


@pytest.mark.parametrize(
    "invalid_case",
    (
        "success_status",
        "success_full_safe_lengths",
        "success_midpoint_lengths",
        "success_safe_projection",
        "success_aggregate_projection",
        "failure_status",
        "failure_code",
        "failed_request_presence",
        "exception_presence",
        "fit_presence",
        "mismatch_presence",
        "resource_unavailable_aggregates",
        "available_failure_aggregates",
        "failure_aligned_lengths",
        "failure_safe_projection",
    ),
)
def test_verified_calibration_outcome_discriminator_matrix(
    invalid_case: str,
) -> None:
    from odmr_bench.emulator import EstimatorObservation
    from odmr_bench.evaluation.two_point import (
        VerifiedTwoPointCalibrationFailure,
        VerifiedTwoPointCalibrationSuccess,
    )
    from odmr_bench.evaluation.two_point.types import VerifiedCalibrationFailureCode

    success = VerifiedTwoPointCalibrationSuccess(
        **_calibration_outcome_arguments(None)  # type: ignore[arg-type]
    )
    failure_codes = (
        "instrument_query_failed",
        "resource_join_unavailable",
        "acquisition_contract_mismatch",
        "fit_failed",
        "fit_exception",
        "source_binding_failed",
    )
    assert get_args(VerifiedCalibrationFailureCode) == failure_codes
    failures = {
        code: VerifiedTwoPointCalibrationFailure(
            **_calibration_outcome_arguments(code)  # type: ignore[arg-type]
        )
        for code in failure_codes
    }

    assert success.status == "success"
    assert len(success.full_observations) == len(success.safe_observations) == len(
        success.measurement_midpoints_s
    )
    assert tuple(item.estimator_view() for item in success.full_observations) == (
        success.safe_observations
    )
    for code, failure in failures.items():
        assert failure.status == "failure"
        assert len(failure.full_observations) == len(failure.safe_observations) == len(
            failure.measurement_midpoints_s
        )
        assert tuple(item.estimator_view() for item in failure.full_observations) == (
            failure.safe_observations
        )
        assert (failure.safe_resources is None) == (
            code == "resource_join_unavailable"
        )
        assert (failure.full_resources is None) == (
            code == "resource_join_unavailable"
        )

    if invalid_case == "success_status":
        target, overrides = success, {"status": "failure"}
    elif invalid_case == "success_full_safe_lengths":
        target, overrides = success, {"safe_observations": ()}
    elif invalid_case == "success_midpoint_lengths":
        target, overrides = success, {"measurement_midpoints_s": ()}
    elif invalid_case == "success_safe_projection":
        safe = success.safe_observations[0]
        target, overrides = success, {
            "safe_observations": (replace(safe, fluorescence=0.97),)
        }
    elif invalid_case == "success_aggregate_projection":
        target, overrides = success, {
            "safe_resources": replace(success.safe_resources, realized_photons=12_249)
        }
    elif invalid_case == "failure_status":
        target, overrides = failures["fit_failed"], {"status": "success"}
    elif invalid_case == "failure_code":
        target, overrides = failures["fit_failed"], {"failure_code": "unknown"}
    elif invalid_case == "failed_request_presence":
        target, overrides = failures["fit_failed"], {
            "failed_request": failures["instrument_query_failed"].failed_request
        }
    elif invalid_case == "exception_presence":
        target, overrides = failures["fit_exception"], {"exception_type": None}
    elif invalid_case == "fit_presence":
        target, overrides = failures["fit_failed"], {"fit_result": None}
    elif invalid_case == "mismatch_presence":
        target, overrides = failures["resource_join_unavailable"], {
            "resource_mismatch_fields": ()
        }
    elif invalid_case == "resource_unavailable_aggregates":
        target, overrides = failures["resource_join_unavailable"], {
            "full_resources": success.full_resources,
            "safe_resources": success.safe_resources,
        }
    elif invalid_case == "available_failure_aggregates":
        target, overrides = failures["acquisition_contract_mismatch"], {
            "full_resources": None,
            "safe_resources": None,
        }
    elif invalid_case == "failure_aligned_lengths":
        target, overrides = failures["fit_exception"], {
            "measurement_midpoints_s": ()
        }
    else:
        target = failures["fit_exception"]
        safe = target.safe_observations[0]
        assert isinstance(safe, EstimatorObservation)
        overrides = {"safe_observations": (replace(safe, fluorescence=0.97),)}

    with pytest.raises((TypeError, ValueError)):
        replace(target, **overrides)


def _tracking_acquisition_arguments(*, unavailable: bool) -> dict[str, object]:
    from odmr_bench.emulator import InstrumentObservation, ResourceSnapshot
    from odmr_bench.evaluation.two_point.resource_accounting import (
        _advance_full_resources,
    )
    from tests.two_point_helpers import make_legal_query

    before = ResourceSnapshot(2, 0.01, 25_000.0, 24_000.0, 24_500, 0, 0.012)
    query = make_legal_query(
        expected_sequence_index=2,
        expected_end_timestamp_s=0.018,
    )
    full_observation = InstrumentObservation(
        sequence_index=2,
        timestamp_s=0.018,
        frequency_hz=query.frequency_hz,
        fluorescence=0.98,
        integration_time_s=query.integration_time_s,
        nominal_exposure_photons=query.expected_nominal_exposure_photons,
        expected_photons=12_000.0,
        realized_photons=12_250,
        sampling_rule="test-rule",
    )
    delta = ResourceSnapshot(1, 0.005, 12_500.0, 12_000.0, 12_250, 0, 0.006)
    after = _advance_full_resources(before, full_observation, 0.001)
    common: dict[str, object] = {
        "resource_join_status": "unavailable" if unavailable else "authenticated",
        "query": query,
        "expected_measurement_midpoint_s": 0.0155,
        "measurement_midpoint_s": 0.0155,
        "full_observation": full_observation,
        "safe_observation": full_observation.estimator_view(),
        "instrument_resources_before": before,
        "instrument_resources_after": after,
    }
    if unavailable:
        common["resource_mismatch_fields"] = ("expected_photons",)
        common["instrument_resources_after"] = replace(
            after, expected_photons=after.expected_photons + 1.0
        )
    else:
        common["instrument_resource_delta"] = delta
    return common


@pytest.mark.parametrize(
    "invalid_case",
    (
        "authenticated_status",
        "authenticated_safe_projection",
        "authenticated_delta",
        "authenticated_after",
        "authenticated_expected_midpoint",
        "authenticated_measurement_midpoint",
        "unavailable_status",
        "unavailable_empty_mismatch",
        "unavailable_mismatch_order",
        "unavailable_safe_projection",
        "pair_truth_reference",
        "pair_release",
        "instrument_failure_type",
        "instrument_failure_boundary",
    ),
)
def test_authenticated_and_unavailable_acquisition_intrinsic_matrix(
    invalid_case: str,
) -> None:
    from odmr_bench.emulator import ResourceSnapshot
    from odmr_bench.evaluation.two_point import (
        TwoPointEvaluatorPairTiming,
        TwoPointInstrumentQueryFailure,
        TwoPointResourceJoinUnavailableAcquisition,
        TwoPointTrackingAcquisition,
    )

    authenticated = TwoPointTrackingAcquisition(
        **_tracking_acquisition_arguments(unavailable=False)  # type: ignore[arg-type]
    )
    unavailable = TwoPointResourceJoinUnavailableAcquisition(
        **_tracking_acquisition_arguments(unavailable=True)  # type: ignore[arg-type]
    )
    first_midpoint = 0.0035
    second_midpoint = 0.0095
    truth_reference = first_midpoint + (second_midpoint - first_midpoint) / 2.0
    timing = TwoPointEvaluatorPairTiming(
        pair_index=0,
        resonance_id="r0",
        first_measurement_midpoint_s=first_midpoint,
        second_measurement_midpoint_s=second_midpoint,
        truth_reference_timestamp_s=truth_reference,
        public_reference_timestamp_s=0.0065,
        release_sequence_index=1,
        release_timestamp_s=0.012,
    )
    failure = TwoPointInstrumentQueryFailure(
        query=authenticated.query,
        exception_type="RuntimeError",
        exception_message="",
        instrument_resources_before=authenticated.instrument_resources_before,
        instrument_resources_after=authenticated.instrument_resources_before,
    )

    assert authenticated.full_observation.estimator_view() == (
        authenticated.safe_observation
    )
    assert unavailable.full_observation.estimator_view() == unavailable.safe_observation
    for acquisition in (authenticated, unavailable):
        assert acquisition.query.expected_sequence_index == (
            acquisition.full_observation.sequence_index
        )
        assert acquisition.query.frequency_hz == (
            acquisition.full_observation.frequency_hz
        )
        assert acquisition.query.integration_time_s == (
            acquisition.full_observation.integration_time_s
        )
        assert acquisition.query.expected_end_timestamp_s == (
            acquisition.full_observation.timestamp_s
        )
        assert acquisition.query.expected_nominal_exposure_photons == (
            acquisition.full_observation.nominal_exposure_photons
        )
    assert not hasattr(unavailable, "instrument_resource_delta")
    assert (
        replace(authenticated, measurement_midpoint_s=None).measurement_midpoint_s
        is None
    )
    assert (
        replace(unavailable, measurement_midpoint_s=None).measurement_midpoint_s
        is None
    )
    assert timing.truth_reference_timestamp_s == truth_reference
    assert timing.release_timestamp_s >= timing.second_measurement_midpoint_s
    assert failure.instrument_resources_before == failure.instrument_resources_after

    if invalid_case == "authenticated_status":
        target, overrides = authenticated, {"resource_join_status": "unavailable"}
    elif invalid_case == "authenticated_safe_projection":
        target, overrides = authenticated, {
            "safe_observation": replace(
                authenticated.safe_observation, fluorescence=0.97
            )
        }
    elif invalid_case == "authenticated_delta":
        target, overrides = authenticated, {
            "instrument_resource_delta": replace(
                authenticated.instrument_resource_delta,
                expected_photons=11_999.0,
            )
        }
    elif invalid_case == "authenticated_after":
        target, overrides = authenticated, {
            "instrument_resources_after": replace(
                authenticated.instrument_resources_after,
                nominal_exposure_photons=(
                    authenticated.instrument_resources_after.nominal_exposure_photons
                    + 1.0
                ),
            )
        }
    elif invalid_case == "authenticated_expected_midpoint":
        target, overrides = authenticated, {
            "expected_measurement_midpoint_s": (
                authenticated.query.expected_end_timestamp_s + 0.001
            )
        }
    elif invalid_case == "authenticated_measurement_midpoint":
        target, overrides = authenticated, {
            "measurement_midpoint_s": authenticated.full_observation.timestamp_s + 0.001
        }
    elif invalid_case == "unavailable_status":
        target, overrides = unavailable, {"resource_join_status": "authenticated"}
    elif invalid_case == "unavailable_empty_mismatch":
        target, overrides = unavailable, {"resource_mismatch_fields": ()}
    elif invalid_case == "unavailable_mismatch_order":
        target, overrides = unavailable, {
            "resource_mismatch_fields": ("expected_photons", "observations")
        }
    elif invalid_case == "unavailable_safe_projection":
        target, overrides = unavailable, {
            "safe_observation": replace(unavailable.safe_observation, fluorescence=0.97)
        }
    elif invalid_case == "pair_truth_reference":
        target, overrides = timing, {"truth_reference_timestamp_s": 0.0064}
    elif invalid_case == "pair_release":
        target, overrides = timing, {"release_timestamp_s": 0.009}
    elif invalid_case == "instrument_failure_type":
        target, overrides = failure, {"exception_type": ""}
    else:
        target, overrides = failure, {
            "instrument_resources_after": ResourceSnapshot(
                3, 0.015, 37_500.0, 36_000.0, 36_750, 0, 0.018
            )
        }

    with pytest.raises((TypeError, ValueError)):
        replace(target, **overrides)


def test_exact_instrument_midpoint_survives_noninvertible_endpoint_rounding() -> None:
    from odmr_bench.emulator import InstrumentObservation, ResourceSnapshot
    from odmr_bench.estimators import PublicAcquisitionResources, TwoPointQuery
    from odmr_bench.evaluation.two_point import (
        TwoPointResourceJoinUnavailableAcquisition,
        TwoPointTrackingAcquisition,
        VerifiedTwoPointCalibrationFailure,
        VerifiedTwoPointCalibrationSuccess,
    )
    from odmr_bench.evaluation.two_point.resource_accounting import (
        _advance_full_resources,
        _project_full_resources,
    )
    from tests.two_point_helpers import make_legal_caller_asserted_source

    start = float.fromhex("0x1.0000000000001p+0")
    integration_time_s = float.fromhex("0x1.0000000000000p-53")
    actual_midpoint_s = start + integration_time_s / 2.0
    endpoint_s = start + integration_time_s
    assert actual_midpoint_s.hex() == "0x1.0000000000001p+0"
    assert endpoint_s.hex() == "0x1.0000000000002p+0"
    assert (endpoint_s - integration_time_s).hex() == "0x1.0000000000002p+0"

    full_observation = InstrumentObservation(
        sequence_index=0,
        timestamp_s=endpoint_s,
        frequency_hz=2.76e9 - 525_000.0,
        fluorescence=0.98,
        integration_time_s=integration_time_s,
        nominal_exposure_photons=1.0,
        expected_photons=0.98,
        realized_photons=None,
        sampling_rule="exact-midpoint-witness",
    )
    zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    atom = _advance_full_resources(zero, full_observation, start)
    query = TwoPointQuery(
        query_index=0,
        pair_index=0,
        identity_pair_index=0,
        resonance_id="r0",
        side="minus",
        interrogation_center_hz=2.76e9,
        frequency_hz=full_observation.frequency_hz,
        integration_time_s=integration_time_s,
        expected_sequence_index=0,
        expected_end_timestamp_s=endpoint_s,
        expected_nominal_exposure_photons=1.0,
    )
    safe_resources = _project_full_resources(atom)
    assert type(safe_resources) is PublicAcquisitionResources

    success_arguments = _calibration_outcome_arguments(None)
    success_arguments.update(
        source=make_legal_caller_asserted_source(),
        full_observations=(full_observation,),
        safe_observations=(full_observation.estimator_view(),),
        measurement_midpoints_s=(actual_midpoint_s,),
        instrument_resources_before=zero,
        instrument_resources_after=atom,
        safe_resources=safe_resources,
        full_resources=atom,
    )
    success = VerifiedTwoPointCalibrationSuccess(
        **success_arguments  # type: ignore[arg-type]
    )

    failure_arguments = _calibration_outcome_arguments("fit_failed")
    failure_arguments.update(
        full_observations=(full_observation,),
        safe_observations=(full_observation.estimator_view(),),
        measurement_midpoints_s=(actual_midpoint_s,),
        instrument_resources_before=zero,
        instrument_resources_after=atom,
        safe_resources=safe_resources,
        full_resources=atom,
    )
    failure = VerifiedTwoPointCalibrationFailure(
        **failure_arguments  # type: ignore[arg-type]
    )

    authenticated = TwoPointTrackingAcquisition(
        resource_join_status="authenticated",
        query=query,
        expected_measurement_midpoint_s=actual_midpoint_s,
        measurement_midpoint_s=actual_midpoint_s,
        full_observation=full_observation,
        safe_observation=full_observation.estimator_view(),
        instrument_resources_before=zero,
        instrument_resources_after=atom,
        instrument_resource_delta=atom,
    )
    unavailable = TwoPointResourceJoinUnavailableAcquisition(
        resource_join_status="unavailable",
        query=query,
        expected_measurement_midpoint_s=actual_midpoint_s,
        measurement_midpoint_s=actual_midpoint_s,
        full_observation=full_observation,
        safe_observation=full_observation.estimator_view(),
        resource_mismatch_fields=("expected_photons",),
        instrument_resources_before=zero,
        instrument_resources_after=replace(
            atom, expected_photons=atom.expected_photons + 1.0
        ),
    )

    assert success.measurement_midpoints_s == (actual_midpoint_s,)
    assert failure.measurement_midpoints_s == (actual_midpoint_s,)
    assert authenticated.measurement_midpoint_s == actual_midpoint_s
    assert unavailable.measurement_midpoint_s == actual_midpoint_s


def test_public_and_truth_pair_references_keep_distinct_clock_conventions() -> None:
    from odmr_bench.evaluation.two_point import TwoPointEvaluatorPairTiming

    first_actual_midpoint_s = float.fromhex("0x1.0000000000000p+0")
    second_actual_midpoint_s = float.fromhex("0x1.0000000000001p+0")
    truth_reference_s = first_actual_midpoint_s + (
        second_actual_midpoint_s - first_actual_midpoint_s
    ) / 2.0
    public_reference_s = float.fromhex("0x1.0000000000002p+0")
    assert truth_reference_s.hex() == "0x1.0000000000000p+0"
    assert public_reference_s.hex() == "0x1.0000000000002p+0"
    assert public_reference_s > second_actual_midpoint_s

    timing = TwoPointEvaluatorPairTiming(
        pair_index=0,
        resonance_id="r0",
        first_measurement_midpoint_s=first_actual_midpoint_s,
        second_measurement_midpoint_s=second_actual_midpoint_s,
        truth_reference_timestamp_s=truth_reference_s,
        public_reference_timestamp_s=public_reference_s,
        release_sequence_index=1,
        release_timestamp_s=float.fromhex("0x1.0000000000003p+0"),
    )

    assert timing.truth_reference_timestamp_s.hex() == "0x1.0000000000000p+0"
    assert timing.public_reference_timestamp_s.hex() == "0x1.0000000000002p+0"


@pytest.mark.parametrize(
    ("failure_code", "allows_final_none"),
    (
        ("instrument_query_failed", False),
        ("resource_join_unavailable", True),
        ("acquisition_contract_mismatch", True),
        ("fit_failed", False),
        ("fit_exception", False),
        ("source_binding_failed", False),
    ),
)
def test_verified_failure_final_none_midpoint_matches_failure_code(
    failure_code: str,
    allows_final_none: bool,
) -> None:
    from odmr_bench.evaluation.two_point import VerifiedTwoPointCalibrationFailure

    arguments = _calibration_outcome_arguments(failure_code)
    arguments["measurement_midpoints_s"] = (None,)
    if allows_final_none:
        failure = VerifiedTwoPointCalibrationFailure(
            **arguments  # type: ignore[arg-type]
        )
        assert failure.measurement_midpoints_s == (None,)
    else:
        with pytest.raises(ValueError, match="midpoint"):
            VerifiedTwoPointCalibrationFailure(
                **arguments  # type: ignore[arg-type]
            )


def test_verified_failure_rejects_an_interior_missing_midpoint() -> None:
    from odmr_bench.evaluation.two_point import VerifiedTwoPointCalibrationFailure

    arguments = _calibration_outcome_arguments("resource_join_unavailable")
    full_observation = arguments["full_observations"][0]  # type: ignore[index]
    safe_observation = arguments["safe_observations"][0]  # type: ignore[index]
    concrete_midpoint = arguments["measurement_midpoints_s"][0]  # type: ignore[index]
    arguments["full_observations"] = (full_observation, full_observation)
    arguments["safe_observations"] = (safe_observation, safe_observation)
    arguments["measurement_midpoints_s"] = (None, concrete_midpoint)

    with pytest.raises(ValueError, match="final failure midpoint"):
        VerifiedTwoPointCalibrationFailure(
            **arguments  # type: ignore[arg-type]
        )


def test_verified_failure_presence_and_fit_polarity_matrix_is_bidirectional() -> None:
    from odmr_bench.evaluation.two_point import (
        VerifiedTwoPointCalibrationFailure,
        VerifiedTwoPointCalibrationSuccess,
    )
    from tests.two_point_helpers import make_legal_source_fit

    failure_codes = (
        "instrument_query_failed",
        "resource_join_unavailable",
        "acquisition_contract_mismatch",
        "fit_failed",
        "fit_exception",
        "source_binding_failed",
    )
    request_codes = {
        "instrument_query_failed",
        "resource_join_unavailable",
        "acquisition_contract_mismatch",
    }
    exception_codes = {
        "instrument_query_failed",
        "fit_exception",
        "source_binding_failed",
    }
    fit_codes = {"fit_failed", "source_binding_failed"}

    legal = {
        code: VerifiedTwoPointCalibrationFailure(
            **_calibration_outcome_arguments(code)  # type: ignore[arg-type]
        )
        for code in failure_codes
    }
    request_donor = legal["instrument_query_failed"].failed_request
    successful_fit = make_legal_source_fit()
    unsuccessful_fit = replace(
        successful_fit,
        success=False,
        failure_code="quality_failed",
        resonance_estimates=(),
        baseline_estimate=None,
    )

    for code, failure in legal.items():
        if code in request_codes:
            with pytest.raises(ValueError, match="failed_request"):
                replace(failure, failed_request=None)
        else:
            with pytest.raises(ValueError, match="failed_request"):
                replace(failure, failed_request=request_donor)

        if code in exception_codes:
            with pytest.raises(ValueError, match="exception"):
                replace(failure, exception_type=None)
            with pytest.raises((TypeError, ValueError), match="exception"):
                replace(failure, exception_message=None)
            empty_message = replace(failure, exception_message="")
            assert empty_message.exception_message == ""
        else:
            with pytest.raises(ValueError, match="exception"):
                replace(failure, exception_type="RuntimeError")
            with pytest.raises(ValueError, match="exception"):
                replace(failure, exception_message="unexpected")

        if code in fit_codes:
            with pytest.raises(ValueError, match="fit_result"):
                replace(failure, fit_result=None)
        else:
            with pytest.raises(ValueError, match="fit_result"):
                replace(failure, fit_result=successful_fit)

        if code == "resource_join_unavailable":
            with pytest.raises(ValueError, match="mismatch"):
                replace(failure, resource_mismatch_fields=())
            for aggregate_field, aggregate_value in (
                ("safe_resources", legal["fit_exception"].safe_resources),
                ("full_resources", legal["fit_exception"].full_resources),
            ):
                with pytest.raises(ValueError, match="aggregate"):
                    replace(failure, **{aggregate_field: aggregate_value})
        else:
            with pytest.raises(ValueError, match="mismatch"):
                replace(failure, resource_mismatch_fields=("expected_photons",))
            with pytest.raises((TypeError, ValueError)):
                replace(failure, safe_resources=None)
            with pytest.raises((TypeError, ValueError)):
                replace(failure, full_resources=None)

    with pytest.raises(ValueError, match="success"):
        replace(legal["fit_failed"], fit_result=successful_fit)
    with pytest.raises(ValueError, match="success"):
        replace(legal["source_binding_failed"], fit_result=unsuccessful_fit)

    success = VerifiedTwoPointCalibrationSuccess(
        **_calibration_outcome_arguments(None)  # type: ignore[arg-type]
    )
    mismatched_full_resources = replace(success.full_resources, observations=2)
    mismatched_safe_resources = replace(success.safe_resources, observations=2)
    with pytest.raises(ValueError, match="observation count"):
        replace(
            success,
            full_resources=mismatched_full_resources,
            safe_resources=mismatched_safe_resources,
        )


def test_exception_type_emptiness_uses_the_canonical_builtin_string() -> None:
    from odmr_bench.evaluation.two_point import VerifiedTwoPointCalibrationFailure

    class TruthyEmpty(str):
        def __bool__(self) -> bool:
            return True

    class FalsyNonempty(str):
        def __bool__(self) -> bool:
            return False

    failure = VerifiedTwoPointCalibrationFailure(
        **_calibration_outcome_arguments("fit_exception")  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="exception_type"):
        replace(failure, exception_type=TruthyEmpty(""))

    canonical = replace(failure, exception_type=FalsyNonempty("X"))
    assert type(canonical.exception_type) is str
    assert canonical.exception_type == "X"
    assert replace(canonical, exception_message="").exception_message == ""


@pytest.mark.parametrize("invalid_midpoint", (-1.0, float("nan"), float("inf")))
def test_acquisition_rejects_negative_or_nonfinite_actual_midpoint(
    invalid_midpoint: float,
) -> None:
    from odmr_bench.evaluation.two_point import TwoPointTrackingAcquisition

    with pytest.raises(ValueError, match="measurement_midpoint_s"):
        TwoPointTrackingAcquisition(
            **(
                _tracking_acquisition_arguments(unavailable=False)
                | {"measurement_midpoint_s": invalid_midpoint}
            )  # type: ignore[arg-type]
        )


def test_pair_public_reference_has_inclusive_release_bounds() -> None:
    from odmr_bench.evaluation.two_point import TwoPointEvaluatorPairTiming

    timing = TwoPointEvaluatorPairTiming(
        0, "r0", 0.0035, 0.0095, 0.006500000000000001, 0.0065, 1, 0.012
    )
    for invalid_reference in (-1.0, float("nan"), float("inf"), 0.0121):
        with pytest.raises(
            ValueError, match=r"public reference|non-negative|finite"
        ):
            replace(timing, public_reference_timestamp_s=invalid_reference)

    public_at_release = replace(timing, public_reference_timestamp_s=0.012)
    release_at_second_midpoint = replace(timing, release_timestamp_s=0.0095)
    assert (
        public_at_release.public_reference_timestamp_s
        == public_at_release.release_timestamp_s
    )
    assert (
        release_at_second_midpoint.release_timestamp_s
        == release_at_second_midpoint.second_measurement_midpoint_s
    )


def test_authenticated_acquisition_accepts_exact_zero_overhead() -> None:
    from odmr_bench.emulator import ResourceSnapshot
    from odmr_bench.evaluation.two_point import TwoPointTrackingAcquisition
    from odmr_bench.evaluation.two_point.resource_accounting import (
        _advance_full_resources,
    )

    arguments = _tracking_acquisition_arguments(unavailable=False)
    observation = arguments["full_observation"]
    before = arguments["instrument_resources_before"]
    zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    arguments["instrument_resource_delta"] = _advance_full_resources(
        zero, observation, 0.0  # type: ignore[arg-type]
    )
    arguments["instrument_resources_after"] = _advance_full_resources(
        before, observation, 0.0  # type: ignore[arg-type]
    )

    acquisition = TwoPointTrackingAcquisition(
        **arguments  # type: ignore[arg-type]
    )
    assert (
        acquisition.instrument_resource_delta.virtual_elapsed_time_s
        == acquisition.full_observation.integration_time_s
    )


def test_unavailable_acquisition_rejects_duplicate_ordered_mismatch_fields() -> None:
    from odmr_bench.evaluation.two_point import (
        TwoPointResourceJoinUnavailableAcquisition,
    )

    arguments = _tracking_acquisition_arguments(unavailable=True)
    arguments["resource_mismatch_fields"] = (
        "expected_photons",
        "expected_photons",
    )
    with pytest.raises(ValueError, match="unique"):
        TwoPointResourceJoinUnavailableAcquisition(
            **arguments  # type: ignore[arg-type]
        )


def test_task11_nested_record_edges_reject_wrong_types_and_capable_subclasses() -> None:
    from odmr_bench.evaluation.two_point import (
        TwoPointInstrumentQueryFailure,
        TwoPointResourceJoinUnavailableAcquisition,
        TwoPointTrackingAcquisition,
        VerifiedTwoPointCalibrationFailure,
        VerifiedTwoPointCalibrationSuccess,
    )

    success = VerifiedTwoPointCalibrationSuccess(
        **_calibration_outcome_arguments(None)  # type: ignore[arg-type]
    )
    failure = VerifiedTwoPointCalibrationFailure(
        **_calibration_outcome_arguments("source_binding_failed")  # type: ignore[arg-type]
    )
    request_failure = VerifiedTwoPointCalibrationFailure(
        **_calibration_outcome_arguments("instrument_query_failed")  # type: ignore[arg-type]
    )
    authenticated = TwoPointTrackingAcquisition(
        **_tracking_acquisition_arguments(unavailable=False)  # type: ignore[arg-type]
    )
    unavailable = TwoPointResourceJoinUnavailableAcquisition(
        **_tracking_acquisition_arguments(unavailable=True)  # type: ignore[arg-type]
    )
    query_failure = TwoPointInstrumentQueryFailure(
        authenticated.query,
        "RuntimeError",
        "",
        authenticated.instrument_resources_before,
        replace(authenticated.instrument_resources_before),
    )
    nested_edges = (
        (success, "source", success.source, False),
        (success, "full_observations", success.full_observations[0], True),
        (success, "safe_observations", success.safe_observations[0], True),
        (
            success,
            "instrument_resources_before",
            success.instrument_resources_before,
            False,
        ),
        (
            success,
            "instrument_resources_after",
            success.instrument_resources_after,
            False,
        ),
        (success, "safe_resources", success.safe_resources, False),
        (success, "full_resources", success.full_resources, False),
        (
            request_failure,
            "failed_request",
            request_failure.failed_request,
            False,
        ),
        (failure, "fit_result", failure.fit_result, False),
        (
            failure,
            "instrument_resources_before",
            failure.instrument_resources_before,
            False,
        ),
        (
            failure,
            "instrument_resources_after",
            failure.instrument_resources_after,
            False,
        ),
        (authenticated, "query", authenticated.query, False),
        (authenticated, "full_observation", authenticated.full_observation, False),
        (authenticated, "safe_observation", authenticated.safe_observation, False),
        (
            authenticated,
            "instrument_resources_before",
            authenticated.instrument_resources_before,
            False,
        ),
        (
            authenticated,
            "instrument_resources_after",
            authenticated.instrument_resources_after,
            False,
        ),
        (
            authenticated,
            "instrument_resource_delta",
            authenticated.instrument_resource_delta,
            False,
        ),
        (unavailable, "query", unavailable.query, False),
        (unavailable, "full_observation", unavailable.full_observation, False),
        (unavailable, "safe_observation", unavailable.safe_observation, False),
        (
            unavailable,
            "instrument_resources_before",
            unavailable.instrument_resources_before,
            False,
        ),
        (
            unavailable,
            "instrument_resources_after",
            unavailable.instrument_resources_after,
            False,
        ),
        (query_failure, "query", query_failure.query, False),
        (
            query_failure,
            "instrument_resources_before",
            query_failure.instrument_resources_before,
            False,
        ),
        (
            query_failure,
            "instrument_resources_after",
            query_failure.instrument_resources_after,
            False,
        ),
    )
    for target, field_name, legal_value, wrap_in_tuple in nested_edges:
        assert legal_value is not None
        capable_value = _capability_record_subclass(legal_value)
        for invalid_value in (object(), capable_value):
            override = (invalid_value,) if wrap_in_tuple else invalid_value
            with pytest.raises((TypeError, ValueError)):
                replace(target, **{field_name: override})

    for target in (success, failure):
        with pytest.raises(TypeError, match="run_token"):
            replace(target, run_token=object())


def test_task11_text_edges_canonicalize_capability_bearing_string_subclasses() -> None:
    from odmr_bench.evaluation.two_point import (
        TwoPointEvaluatorPairTiming,
        TwoPointInstrumentQueryFailure,
        TwoPointResourceJoinUnavailableAcquisition,
        TwoPointTrackingAcquisition,
        VerifiedTwoPointCalibrationFailure,
        VerifiedTwoPointCalibrationSuccess,
    )

    class CapabilityString(str):
        __slots__ = ("__dict__", "slot_capability")

    def capable(value: str) -> CapabilityString:
        candidate = CapabilityString(value)
        candidate.callback = lambda: None
        candidate.payload = {"retained": object()}
        candidate.slot_capability = object()
        return candidate

    success = VerifiedTwoPointCalibrationSuccess(
        **_calibration_outcome_arguments(None)  # type: ignore[arg-type]
    )
    failure = VerifiedTwoPointCalibrationFailure(
        **_calibration_outcome_arguments("source_binding_failed")  # type: ignore[arg-type]
    )
    authenticated = TwoPointTrackingAcquisition(
        **_tracking_acquisition_arguments(unavailable=False)  # type: ignore[arg-type]
    )
    unavailable = TwoPointResourceJoinUnavailableAcquisition(
        **_tracking_acquisition_arguments(unavailable=True)  # type: ignore[arg-type]
    )
    timing = TwoPointEvaluatorPairTiming(
        0, "r0", 0.0035, 0.0095, 0.006500000000000001, 0.0065, 1, 0.012
    )
    query_failure = TwoPointInstrumentQueryFailure(
        authenticated.query,
        "RuntimeError",
        "",
        authenticated.instrument_resources_before,
        replace(authenticated.instrument_resources_before),
    )
    scalar_edges = (
        (success, "status", "success"),
        (failure, "status", "failure"),
        (failure, "failure_code", "source_binding_failed"),
        (failure, "exception_type", "RuntimeError"),
        (failure, "exception_message", "boom"),
        (authenticated, "resource_join_status", "authenticated"),
        (unavailable, "resource_join_status", "unavailable"),
        (timing, "resonance_id", "r0"),
        (query_failure, "exception_type", "RuntimeError"),
        (query_failure, "exception_message", ""),
    )
    for target, field_name, value in scalar_edges:
        canonical = replace(target, **{field_name: capable(value)})
        retained = getattr(canonical, field_name)
        assert type(retained) is str
        assert retained == value
        assert not hasattr(retained, "callback")

    mismatch_failure = VerifiedTwoPointCalibrationFailure(
        **_calibration_outcome_arguments("resource_join_unavailable")  # type: ignore[arg-type]
    )
    for target in (mismatch_failure, unavailable):
        canonical = replace(
            target,
            resource_mismatch_fields=(capable("expected_photons"),),
        )
        retained = canonical.resource_mismatch_fields[0]
        assert type(retained) is str
        assert retained == "expected_photons"
        assert not hasattr(retained, "callback")


def test_task11_record_schemas_pin_annotations_signatures_and_frozen_fields() -> None:
    from odmr_bench.evaluation.two_point import (
        TwoPointEvaluatorPairTiming,
        TwoPointInstrumentQueryFailure,
        TwoPointResourceJoinUnavailableAcquisition,
        TwoPointTrackingAcquisition,
        VerifiedTwoPointCalibrationFailure,
        VerifiedTwoPointCalibrationSuccess,
    )

    expected_annotations = {
        VerifiedTwoPointCalibrationSuccess: {
            "status": "Literal['success']",
            "run_token": "VerifiedInstrumentRunToken",
            "source": "TwoPointCalibrationSource",
            "full_observations": "tuple[InstrumentObservation, ...]",
            "safe_observations": "tuple[EstimatorObservation, ...]",
            "measurement_midpoints_s": "tuple[float, ...]",
            "instrument_resources_before": "ResourceSnapshot",
            "instrument_resources_after": "ResourceSnapshot",
            "safe_resources": "PublicAcquisitionResources",
            "full_resources": "ResourceSnapshot",
        },
        VerifiedTwoPointCalibrationFailure: {
            "status": "Literal['failure']",
            "run_token": "VerifiedInstrumentRunToken",
            "failure_code": "VerifiedCalibrationFailureCode",
            "failed_request": "VerifiedCalibrationQueryRequest | None",
            "exception_type": "str | None",
            "exception_message": "str | None",
            "fit_result": "SpectrumFitResult | None",
            "resource_mismatch_fields": (
                "tuple[ResourceJoinMismatchField, ...]"
            ),
            "full_observations": "tuple[InstrumentObservation, ...]",
            "safe_observations": "tuple[EstimatorObservation, ...]",
            "measurement_midpoints_s": "tuple[float | None, ...]",
            "instrument_resources_before": "ResourceSnapshot",
            "instrument_resources_after": "ResourceSnapshot",
            "safe_resources": "PublicAcquisitionResources | None",
            "full_resources": "ResourceSnapshot | None",
        },
        TwoPointTrackingAcquisition: {
            "resource_join_status": "Literal['authenticated']",
            "query": "TwoPointQuery",
            "expected_measurement_midpoint_s": "float",
            "measurement_midpoint_s": "float | None",
            "full_observation": "InstrumentObservation",
            "safe_observation": "EstimatorObservation",
            "instrument_resources_before": "ResourceSnapshot",
            "instrument_resources_after": "ResourceSnapshot",
            "instrument_resource_delta": "ResourceSnapshot",
        },
        TwoPointResourceJoinUnavailableAcquisition: {
            "resource_join_status": "Literal['unavailable']",
            "query": "TwoPointQuery",
            "expected_measurement_midpoint_s": "float",
            "measurement_midpoint_s": "float | None",
            "full_observation": "InstrumentObservation",
            "safe_observation": "EstimatorObservation",
            "resource_mismatch_fields": (
                "tuple[ResourceJoinMismatchField, ...]"
            ),
            "instrument_resources_before": "ResourceSnapshot",
            "instrument_resources_after": "ResourceSnapshot",
        },
        TwoPointEvaluatorPairTiming: {
            "pair_index": "int",
            "resonance_id": "str",
            "first_measurement_midpoint_s": "float",
            "second_measurement_midpoint_s": "float",
            "truth_reference_timestamp_s": "float",
            "public_reference_timestamp_s": "float",
            "release_sequence_index": "int",
            "release_timestamp_s": "float",
        },
        TwoPointInstrumentQueryFailure: {
            "query": "TwoPointQuery",
            "exception_type": "str",
            "exception_message": "str",
            "instrument_resources_before": "ResourceSnapshot",
            "instrument_resources_after": "ResourceSnapshot",
        },
    }
    for record_class, annotations in expected_annotations.items():
        record_fields = fields(record_class)
        assert record_class.__annotations__ == annotations
        assert record_class.__dataclass_params__.frozen is True
        assert record_class.__slots__ == tuple(field.name for field in record_fields)
        assert all(field.default is MISSING for field in record_fields)
        assert all(field.default_factory is MISSING for field in record_fields)
        signature = inspect.signature(record_class)
        assert tuple(signature.parameters) == tuple(
            field.name for field in record_fields
        )
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and parameter.default is inspect.Parameter.empty
            for parameter in signature.parameters.values()
        )

    authenticated = TwoPointTrackingAcquisition(
        **_tracking_acquisition_arguments(unavailable=False)  # type: ignore[arg-type]
    )
    records = (
        VerifiedTwoPointCalibrationSuccess(
            **_calibration_outcome_arguments(None)  # type: ignore[arg-type]
        ),
        VerifiedTwoPointCalibrationFailure(
            **_calibration_outcome_arguments("fit_failed")  # type: ignore[arg-type]
        ),
        authenticated,
        TwoPointResourceJoinUnavailableAcquisition(
            **_tracking_acquisition_arguments(unavailable=True)  # type: ignore[arg-type]
        ),
        TwoPointEvaluatorPairTiming(
            0,
            "r0",
            0.0035,
            0.0095,
            0.006500000000000001,
            0.0065,
            1,
            0.012,
        ),
        TwoPointInstrumentQueryFailure(
            authenticated.query,
            "RuntimeError",
            "",
            authenticated.instrument_resources_before,
            replace(authenticated.instrument_resources_before),
        ),
    )
    for record in records:
        first_field = fields(record)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(record, first_field, getattr(record, first_field))


def test_pair_truth_reference_rejects_both_adjacent_binary64_values() -> None:
    from odmr_bench.evaluation.two_point import TwoPointEvaluatorPairTiming

    timing = TwoPointEvaluatorPairTiming(
        0, "r0", 0.0035, 0.0095, 0.006500000000000001, 0.0065, 1, 0.012
    )
    assert timing.truth_reference_timestamp_s.hex() == "0x1.a9fbe76c8b43ap-8"
    for direction in (-math.inf, math.inf):
        neighbor = math.nextafter(timing.truth_reference_timestamp_s, direction)
        assert neighbor.hex() in {
            "0x1.a9fbe76c8b439p-8",
            "0x1.a9fbe76c8b43bp-8",
        }
        with pytest.raises(ValueError, match="ordered midpoint mean"):
            replace(timing, truth_reference_timestamp_s=neighbor)


def test_query_failure_accepts_equal_but_distinct_resource_snapshots() -> None:
    from odmr_bench.evaluation.two_point import (
        TwoPointInstrumentQueryFailure,
        TwoPointTrackingAcquisition,
    )

    authenticated = TwoPointTrackingAcquisition(
        **_tracking_acquisition_arguments(unavailable=False)  # type: ignore[arg-type]
    )
    before = authenticated.instrument_resources_before
    after = replace(before)
    assert after == before
    assert after is not before

    failure = TwoPointInstrumentQueryFailure(
        authenticated.query,
        "RuntimeError",
        "",
        before,
        after,
    )
    assert failure.instrument_resources_before == failure.instrument_resources_after
    assert failure.instrument_resources_before is not failure.instrument_resources_after


def test_task11_resource_delegation_is_local_exact_and_association_sensitive() -> None:
    from odmr_bench.emulator import InstrumentObservation, ResourceSnapshot
    from odmr_bench.estimators import TwoPointQuery
    from odmr_bench.evaluation.two_point import TwoPointTrackingAcquisition
    from odmr_bench.evaluation.two_point.resource_accounting import (
        _advance_full_resources,
    )

    large = float(2**53)
    before = ResourceSnapshot(1, 0.0, large, large, 0, 1, large)
    observation = InstrumentObservation(
        1,
        large + 2.0,
        2.76e9 - 525_000.0,
        0.98,
        1.0,
        1.0,
        1.0,
        None,
        "association-sensitive",
    )
    query = TwoPointQuery(
        0,
        0,
        0,
        "r0",
        "minus",
        2.76e9,
        observation.frequency_hz,
        1.0,
        1,
        observation.timestamp_s,
        1.0,
    )
    zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    atom = _advance_full_resources(zero, observation, 1.0)
    after = _advance_full_resources(before, observation, 1.0)
    reassociated_elapsed = (before.virtual_elapsed_time_s + 1.0) + 1.0
    assert before.virtual_elapsed_time_s.hex() == "0x1.0000000000000p+53"
    assert after.virtual_elapsed_time_s.hex() == "0x1.0000000000001p+53"
    assert reassociated_elapsed.hex() == "0x1.0000000000000p+53"

    acquisition = TwoPointTrackingAcquisition(
        "authenticated",
        query,
        large,
        large,
        observation,
        observation.estimator_view(),
        before,
        after,
        atom,
    )
    assert acquisition.instrument_resources_after is after

    types_path = (
        Path(__file__).resolve().parents[2]
        / "src/odmr_bench/evaluation/two_point/types.py"
    )
    tree = ast.parse(types_path.read_text(), filename=str(types_path))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    resource_imports = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "odmr_bench.evaluation.two_point.resource_accounting"
    )
    imported_names = tuple(
        sorted(alias.name for node in resource_imports for alias in node.names)
    )
    assert imported_names == (
        "_advance_full_resources",
        "_project_full_resources",
        "_resource_mismatch_fields",
    )
    assert all(
        alias.asname is None for node in resource_imports for alias in node.names
    )
    for import_node in resource_imports:
        ancestor = parents[import_node]
        while not isinstance(ancestor, ast.FunctionDef | ast.Module):
            ancestor = parents[ancestor]
        assert isinstance(ancestor, ast.FunctionDef)

    helper_names = set(imported_names)
    helper_calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in helper_names
    )
    calls_by_name = {
        helper_name: tuple(
            call for call in helper_calls if call.func.id == helper_name
        )
        for helper_name in helper_names
    }
    assert {name: len(calls) for name, calls in calls_by_name.items()} == {
        "_advance_full_resources": 2,
        "_project_full_resources": 1,
        "_resource_mismatch_fields": 1,
    }
    assert {
        ast.unparse(call) for call in calls_by_name["_advance_full_resources"]
    } == {
        "_advance_full_resources(zero_resources, self.full_observation, overhead_s)",
        "_advance_full_resources(self.instrument_resources_before, "
        "self.full_observation, overhead_s)",
    }
    advance_assignments = {
        ast.unparse(parents[call].targets[0]): ast.unparse(call)
        for call in calls_by_name["_advance_full_resources"]
        if isinstance(parents[call], ast.Assign)
    }
    assert set(advance_assignments) == {"expected_delta", "expected_after"}

    (project_call,) = calls_by_name["_project_full_resources"]
    project_comparison = parents[project_call]
    assert ast.unparse(project_call) == "_project_full_resources(full_resources)"
    assert isinstance(project_comparison, ast.Compare)
    assert project_comparison.left is project_call
    assert isinstance(project_comparison.ops[0], ast.NotEq)
    assert ast.unparse(project_comparison.comparators[0]) == "safe_resources"
    assert isinstance(parents[project_comparison], ast.If)
    assert parents[project_comparison].test is project_comparison

    (mismatch_call,) = calls_by_name["_resource_mismatch_fields"]
    assert ast.unparse(mismatch_call) == (
        "_resource_mismatch_fields(expected_after, "
        "self.instrument_resources_after)"
    )
    assert isinstance(parents[mismatch_call], ast.If)
    assert parents[mismatch_call].test is mismatch_call

    source = types_path.read_text()
    assert "_zero_full_resources" not in source
    assert "_replay_full_resources" not in source
    resource_fields = {
        "observations",
        "integration_time_s",
        "nominal_exposure_photons",
        "expected_photons",
        "realized_photons",
        "observations_without_realized_counts",
        "virtual_elapsed_time_s",
    }
    allowed_resource_arithmetic = {
        "self.instrument_resource_delta.virtual_elapsed_time_s - "
        "self.full_observation.integration_time_s"
    }
    duplicated_arithmetic = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and any(
            isinstance(child, ast.Attribute) and child.attr in resource_fields
            for child in ast.walk(node)
        )
        and ast.unparse(node) not in allowed_resource_arithmetic
    )
    assert duplicated_arithmetic == ()
    suspicious_aggregate_calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "sum")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"add", "fsum", "sum"}
            )
        )
        and any(
            isinstance(child, ast.Attribute) and child.attr in resource_fields
            for child in ast.walk(node)
        )
    )
    assert suspicious_aggregate_calls == ()


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
