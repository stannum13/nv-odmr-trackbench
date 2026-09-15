"""Intrinsic contracts for sparse-linewidth evaluator values."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import get_args, get_type_hints

import pytest


def _pending_estimate():
    from tests.sparse_linewidth_helpers import (
        make_composite_estimate,
        make_composite_identity,
    )
    from tests.two_point_helpers import make_legal_estimate

    query = make_legal_estimate("first").pending_query
    assert query is not None
    base = make_composite_estimate()
    identities = list(base.identities)
    identities[0] = make_composite_identity(
        fast_center_hz=query.interrogation_center_hz,
        live_q=query.interrogation_center_hz / 1.0e6,
    )
    return make_composite_estimate(
        identities=identities,
        pending_mode="fast_pair",
        pending_query=query,
    )


def _full_acquisition_arguments(*, unavailable: bool = False) -> dict[str, object]:
    from dataclasses import replace

    from odmr_bench.emulator import InstrumentObservation, ResourceSnapshot
    from odmr_bench.evaluation.two_point.resource_accounting import (
        _advance_full_resources,
    )

    estimate = _pending_estimate()
    query = estimate.pending_query
    assert query is not None
    full = InstrumentObservation(
        sequence_index=query.expected_sequence_index,
        timestamp_s=query.expected_end_timestamp_s,
        frequency_hz=query.frequency_hz,
        fluorescence=0.98,
        integration_time_s=query.integration_time_s,
        nominal_exposure_photons=query.expected_nominal_exposure_photons,
        expected_photons=12_000.0,
        realized_photons=12_250,
        sampling_rule="test-rule",
    )
    before = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    delta = _advance_full_resources(before, full, 0.001)
    values: dict[str, object] = {
        "resource_join_status": "unavailable" if unavailable else "authenticated",
        "mode": "fast_pair",
        "query": query,
        "expected_measurement_midpoint_s": (
            query.expected_end_timestamp_s - query.integration_time_s / 2.0
        ),
        "measurement_midpoint_s": (
            full.timestamp_s - full.integration_time_s / 2.0
        ),
        "full_observation": full,
        "safe_observation": full.estimator_view(),
        "instrument_resources_before": before,
        "instrument_resources_after": delta,
    }
    if unavailable:
        values["resource_mismatch_fields"] = ["expected_photons"]
        values["instrument_resources_after"] = replace(
            delta, expected_photons=delta.expected_photons + 1.0
        )
    else:
        values["instrument_resource_delta"] = delta
    return values


def _geometry_diagnostic():
    from odmr_bench.estimators import SparseGeometryUnavailableDiagnostic

    return SparseGeometryUnavailableDiagnostic(
        failure_code="calibration_cell_violation",
        scan_index=0,
        identity_scan_index=0,
        resonance_id="r0",
        fast_center_hz=2.87e9,
        fast_center_source_kind="calibration",
        fast_center_source_pair_index=None,
        fast_center_reference_timestamp_s=0.0,
        fast_center_release_sequence_index=None,
        fast_center_release_timestamp_s=0.0,
        prior_fwhm_hz=1.0e6,
        fwhm_source_kind="calibration",
        fwhm_source_scan_index=None,
        fwhm_reference_timestamp_s=0.0,
        fwhm_release_sequence_index=None,
        fwhm_release_timestamp_s=0.0,
        proposed_frequency_min_hz=2.869e9,
        proposed_frequency_max_hz=2.871e9,
        calibration_cell_lower_hz=2.87e9,
        calibration_cell_upper_hz=2.872e9,
        source_frequency_min_hz=2.80e9,
        source_frequency_max_hz=2.94e9,
    )


def _state_arguments(phase: str) -> dict[str, object]:
    from odmr_bench.emulator import ResourceSnapshot
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseAbortedRun,
        SparseResourceJoinUnavailableAcquisition,
    )
    from tests.evaluation.test_two_point_types import _runner_state_arguments
    from tests.sparse_linewidth_helpers import make_composite_estimate

    two_point_ready = _runner_state_arguments("ready")
    success_state = _runner_state_arguments("calibration_succeeded")
    failure_state = _runner_state_arguments("calibration_failed")
    active_state = _runner_state_arguments("tracking")
    active = phase in {
        "tracking",
        "budget_stopped",
        "geometry_stopped",
        "externally_stopped",
        "aborted",
    }
    if phase == "budget_stopped":
        estimate = make_composite_estimate(stopped_reason="budget_exhausted")
    elif phase == "geometry_stopped":
        diagnostic = _geometry_diagnostic()
        estimate = make_composite_estimate(
            stopped_reason="sparse_geometry_unavailable",
            sparse_geometry_diagnostic=diagnostic,
        )
    elif phase == "aborted":
        estimate = _pending_estimate()
    else:
        estimate = make_composite_estimate()
    terminal_abort = None
    if phase == "aborted":
        unavailable = SparseResourceJoinUnavailableAcquisition(
            **_full_acquisition_arguments(unavailable=True)  # type: ignore[arg-type]
        )
        terminal_abort = SparseAbortedRun(
            "resource_join_unavailable",
            None,
            None,
            unavailable,
            1,
            estimate,
            replace(estimate),
        )
    zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    return {
        "phase": phase,
        "run_token": two_point_ready["run_token"],
        "instrument_configuration": two_point_ready["instrument_configuration"],
        "calibration_outcome": (
            failure_state["calibration_outcome"]
            if phase == "calibration_failed"
            else success_state["calibration_outcome"]
            if phase == "calibration_succeeded"
            else None
        ),
        "verified_calibration": (
            success_state["verified_calibration"]
            if active or phase == "calibration_succeeded"
            else None
        ),
        "calibration": active_state["calibration"] if active else None,
        "tracker_estimate": estimate if active else None,
        "normal_tracking_trace": (),
        "pair_timings": (),
        "scan_timings": (),
        "instrument_resources_at_bind": zero,
        "tracking_resources_before": zero if active else None,
        "instrument_resources_current": (
            failure_state["instrument_resources_current"]
            if phase == "calibration_failed"
            else success_state["instrument_resources_current"]
            if phase == "calibration_succeeded"
            else zero
        ),
        "instrument_current_sequence_index": (
            0
            if phase in {"calibration_failed", "calibration_succeeded"}
            else None
        ),
        "current_virtual_time_s": (
            0.006
            if phase in {"calibration_failed", "calibration_succeeded"}
            else 0.0
        ),
        "last_instrument_failure": None,
        "terminal_abort": terminal_abort,
        "fast_update_cpu_time_s": (
            estimate.fast_update_cpu_time_s if active else 0.0
        ),
        "sparse_update_cpu_time_s": (
            estimate.sparse_update_cpu_time_s if active else 0.0
        ),
        "total_update_cpu_time_s": (
            estimate.total_update_cpu_time_s if active else 0.0
        ),
    }


def _resources_arguments() -> dict[str, object]:
    from odmr_bench.emulator import ResourceSnapshot

    full = _full_acquisition_arguments()["full_observation"]
    zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    return {
        "calibration_observations": [full],
        "accepted_fast_observations": [],
        "accepted_sparse_observations": [],
        "accepted_tracking_observations": [full],
        "unaccepted_tracking_observations": [],
        "calibration_resources": zero,
        "fast_tracking_resources": zero,
        "sparse_tracking_resources": zero,
        "tracking_resources": zero,
        "accepted_charged_resources": zero,
        "charged_resources": zero,
        "calibration_budget_treatment": "conditional_free_precalibration",
        "incomplete_fast_pair_observations": 0,
        "incomplete_sparse_scan_observations": 0,
        "unaccepted_observations": 0,
    }


def test_sparse_runner_phase_is_closed_and_geometry_is_distinct() -> None:
    from odmr_bench.evaluation.sparse_linewidth import SparseRunnerPhase

    assert get_args(SparseRunnerPhase) == (
        "ready",
        "calibration_succeeded",
        "calibration_failed",
        "tracking",
        "budget_stopped",
        "geometry_stopped",
        "externally_stopped",
        "aborted",
    )


def test_sparse_evaluator_public_value_surface_is_exact() -> None:
    from odmr_bench.evaluation import sparse_linewidth

    assert tuple(sparse_linewidth.__all__) == (
        "SparseAbortReason",
        "SparseAbortedRun",
        "SparseEvaluatorRunnerState",
        "SparseInstrumentQueryFailure",
        "SparseLinewidthEvaluatorResources",
        "SparseLinewidthEvaluatorScanTiming",
        "SparsePreflightCode",
        "SparsePreflightError",
        "SparseResourceJoinUnavailableAcquisition",
        "SparseRunnerAborted",
        "SparseRunnerAccepted",
        "SparseRunnerBudgetStopped",
        "SparseRunnerExternallyStopped",
        "SparseRunnerGeometryStopped",
        "SparseRunnerInstrumentFailure",
        "SparseRunnerPhase",
        "SparseRunnerRunOutcome",
        "SparseRunnerStateError",
        "SparseRunnerStepOutcome",
        "SparseStartCode",
        "SparseStartError",
        "SparseTrackingAcquisition",
    )
    source = Path(inspect.getsourcefile(sparse_linewidth) or "").read_text()
    assert ".runner" not in source


def test_sparse_aliases_and_outcome_unions_are_exact() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseAbortReason,
        SparsePreflightCode,
        SparseRunnerAborted,
        SparseRunnerAccepted,
        SparseRunnerBudgetStopped,
        SparseRunnerGeometryStopped,
        SparseRunnerInstrumentFailure,
        SparseRunnerRunOutcome,
        SparseRunnerStepOutcome,
        SparseStartCode,
    )

    assert get_args(SparseAbortReason) == (
        "resource_join_unavailable",
        "tracker_observation_validation_error",
        "tracker_update_construction_error",
        "tracker_update_unexpected_error",
    )
    assert get_args(SparsePreflightCode) == (
        "invalid_runner_phase",
        "invalid_argument_type",
        "invalid_argument_value",
        "invalid_frequency_grid",
        "invalid_fit_or_identity_configuration",
        "invalid_clock_mapping",
        "unclean_instrument_boundary",
    )
    assert get_args(SparseStartCode) == (
        "invalid_runner_phase",
        "invalid_argument_type",
        "unverified_calibration",
        "calibration_mismatch",
        "run_provenance_mismatch",
        "metadata_mismatch",
        "resource_boundary_mismatch",
        "tracker_reset_failed",
    )
    assert get_args(SparseRunnerStepOutcome) == (
        SparseRunnerAccepted,
        SparseRunnerInstrumentFailure,
        SparseRunnerBudgetStopped,
        SparseRunnerGeometryStopped,
        SparseRunnerAborted,
    )
    assert get_args(SparseRunnerRunOutcome) == (
        SparseRunnerInstrumentFailure,
        SparseRunnerBudgetStopped,
        SparseRunnerGeometryStopped,
        SparseRunnerAborted,
    )


def test_sparse_record_schemas_are_exact_frozen_and_slotted() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseAbortedRun,
        SparseEvaluatorRunnerState,
        SparseInstrumentQueryFailure,
        SparseLinewidthEvaluatorResources,
        SparseLinewidthEvaluatorScanTiming,
        SparseResourceJoinUnavailableAcquisition,
        SparseRunnerAborted,
        SparseRunnerAccepted,
        SparseRunnerBudgetStopped,
        SparseRunnerExternallyStopped,
        SparseRunnerGeometryStopped,
        SparseRunnerInstrumentFailure,
        SparseTrackingAcquisition,
    )

    expected = {
        SparseTrackingAcquisition: (
            "resource_join_status",
            "mode",
            "query",
            "expected_measurement_midpoint_s",
            "measurement_midpoint_s",
            "full_observation",
            "safe_observation",
            "instrument_resources_before",
            "instrument_resources_after",
            "instrument_resource_delta",
        ),
        SparseResourceJoinUnavailableAcquisition: (
            "resource_join_status",
            "mode",
            "query",
            "expected_measurement_midpoint_s",
            "measurement_midpoint_s",
            "full_observation",
            "safe_observation",
            "resource_mismatch_fields",
            "instrument_resources_before",
            "instrument_resources_after",
        ),
        SparseInstrumentQueryFailure: (
            "mode",
            "query",
            "exception_type",
            "exception_message",
            "instrument_resources_before",
            "instrument_resources_after",
        ),
        SparseAbortedRun: (
            "reason",
            "exception_type",
            "exception_message",
            "unaccepted_acquisition",
            "unaccepted_observation_count",
            "tracker_estimate_before",
            "tracker_estimate_after",
        ),
        SparseLinewidthEvaluatorScanTiming: (
            "scan_index",
            "resonance_id",
            "measurement_midpoints_s",
            "truth_reference_timestamp_s",
            "public_reference_timestamp_s",
            "release_sequence_index",
            "release_timestamp_s",
        ),
        SparseLinewidthEvaluatorResources: (
            "calibration_observations",
            "accepted_fast_observations",
            "accepted_sparse_observations",
            "accepted_tracking_observations",
            "unaccepted_tracking_observations",
            "calibration_resources",
            "fast_tracking_resources",
            "sparse_tracking_resources",
            "tracking_resources",
            "accepted_charged_resources",
            "charged_resources",
            "calibration_budget_treatment",
            "incomplete_fast_pair_observations",
            "incomplete_sparse_scan_observations",
            "unaccepted_observations",
        ),
        SparseEvaluatorRunnerState: (
            "phase",
            "run_token",
            "instrument_configuration",
            "calibration_outcome",
            "verified_calibration",
            "calibration",
            "tracker_estimate",
            "normal_tracking_trace",
            "pair_timings",
            "scan_timings",
            "instrument_resources_at_bind",
            "tracking_resources_before",
            "instrument_resources_current",
            "instrument_current_sequence_index",
            "current_virtual_time_s",
            "last_instrument_failure",
            "terminal_abort",
            "fast_update_cpu_time_s",
            "sparse_update_cpu_time_s",
            "total_update_cpu_time_s",
        ),
        SparseRunnerAccepted: ("kind", "acquisition", "update", "state"),
        SparseRunnerInstrumentFailure: ("kind", "failure", "state"),
        SparseRunnerBudgetStopped: ("kind", "resources", "state"),
        SparseRunnerGeometryStopped: (
            "kind",
            "diagnostic",
            "resources",
            "state",
        ),
        SparseRunnerExternallyStopped: ("kind", "resources", "state"),
        SparseRunnerAborted: ("kind", "abort", "resources", "state"),
    }
    for record_class, expected_fields in expected.items():
        record_fields = tuple(field.name for field in fields(record_class))
        assert record_fields == expected_fields
        assert record_class.__dataclass_params__.frozen is True
        assert record_class.__slots__ == expected_fields
        assert tuple(get_type_hints(record_class)) == expected_fields


def test_sparse_errors_are_closed_and_canonical() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparsePreflightError,
        SparseRunnerStateError,
        SparseStartError,
    )

    class Code(str):
        pass

    for error_class, codes in (
        (
            SparsePreflightError,
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
            SparseStartError,
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
    ):
        for code in codes:
            error = error_class(Code(code))
            assert type(error.code) is str
            assert error.code == code
            assert str(error) == code
        with pytest.raises(TypeError, match="code must be a string"):
            error_class(object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown sparse evaluator error code"):
            error_class("unknown")  # type: ignore[arg-type]
    assert issubclass(SparseRunnerStateError, RuntimeError)


def test_authenticated_and_unavailable_acquisition_matrix() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseResourceJoinUnavailableAcquisition,
        SparseTrackingAcquisition,
    )

    authenticated = SparseTrackingAcquisition(
        **_full_acquisition_arguments()  # type: ignore[arg-type]
    )
    unavailable = SparseResourceJoinUnavailableAcquisition(
        **_full_acquisition_arguments(unavailable=True)  # type: ignore[arg-type]
    )
    assert authenticated.resource_join_status == "authenticated"
    assert unavailable.resource_join_status == "unavailable"
    assert unavailable.resource_mismatch_fields == ("expected_photons",)
    with pytest.raises(FrozenInstanceError):
        authenticated.mode = "sparse_scan"  # type: ignore[misc]
    with pytest.raises(ValueError, match="mode must match query type"):
        replace(authenticated, mode="sparse_scan")
    with pytest.raises(ValueError, match="resource_join_status"):
        replace(authenticated, resource_join_status="unavailable")
    with pytest.raises(TypeError, match="query"):
        replace(authenticated, query=object())
    with pytest.raises(ValueError, match="exactly project"):
        replace(
            authenticated,
            safe_observation=replace(
                authenticated.safe_observation, fluorescence=0.9
            ),
        )
    with pytest.raises(ValueError, match="one-observation atom"):
        replace(
            authenticated,
            instrument_resource_delta=replace(
                authenticated.instrument_resource_delta,
                expected_photons=(
                    authenticated.instrument_resource_delta.expected_photons + 1.0
                ),
            ),
        )
    with pytest.raises(ValueError, match="requires resource_mismatch_fields"):
        replace(unavailable, resource_mismatch_fields=())
    with pytest.raises(ValueError, match="unique and in declaration order"):
        replace(
            unavailable,
            resource_mismatch_fields=("expected_photons", "expected_photons"),
        )


def test_query_failure_requires_mode_canonical_strings_and_unchanged_boundary() -> None:
    from odmr_bench.evaluation.sparse_linewidth import SparseInstrumentQueryFailure

    class Text(str):
        pass

    acquisition = _full_acquisition_arguments()
    before = acquisition["instrument_resources_before"]
    failure = SparseInstrumentQueryFailure(
        "fast_pair",
        acquisition["query"],
        Text("RuntimeError"),
        Text(""),
        before,
        replace(before),
    )
    assert type(failure.exception_type) is str
    assert type(failure.exception_message) is str
    with pytest.raises(ValueError, match="mode must match query type"):
        replace(failure, mode="sparse_scan")
    with pytest.raises(ValueError, match="nonempty"):
        replace(failure, exception_type="")
    with pytest.raises(ValueError, match="preserve its resource boundary"):
        replace(
            failure,
            instrument_resources_after=_full_acquisition_arguments()[
                "instrument_resources_after"
            ],
        )


def test_scan_timing_has_exact_five_value_fold_and_defensive_tuple() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseLinewidthEvaluatorScanTiming,
    )

    midpoints = [1.0, 2.0, 4.0, 8.0, 16.0]
    folded = midpoints[0]
    for count, value in enumerate(midpoints[1:], start=2):
        folded = folded + (value - folded) / count
    timing = SparseLinewidthEvaluatorScanTiming(
        0, "r0", midpoints, folded, 6.25, 4, 17.0
    )
    midpoints.append(32.0)
    assert timing.measurement_midpoints_s == (1.0, 2.0, 4.0, 8.0, 16.0)
    assert all(type(value) is float for value in timing.measurement_midpoints_s)
    with pytest.raises(ValueError, match="exactly five"):
        replace(timing, measurement_midpoints_s=(1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(timing, measurement_midpoints_s=(1.0, 2.0, 2.0, 8.0, 16.0))
    with pytest.raises(ValueError, match="ordered five-value mean"):
        replace(timing, truth_reference_timestamp_s=folded + 1.0)
    with pytest.raises(ValueError, match="fifth midpoint"):
        replace(timing, release_timestamp_s=15.0)


def test_resources_defensively_copy_all_observation_tuples_and_count_matrix() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseLinewidthEvaluatorResources,
    )

    arguments = _resources_arguments()
    calibration = arguments["calibration_observations"]
    assert isinstance(calibration, list)
    full_observation = calibration[0]
    resources = SparseLinewidthEvaluatorResources(
        **arguments  # type: ignore[arg-type]
    )
    calibration.clear()
    assert len(resources.calibration_observations) == 1
    for name in (
        "calibration_observations",
        "accepted_fast_observations",
        "accepted_sparse_observations",
        "accepted_tracking_observations",
        "unaccepted_tracking_observations",
    ):
        assert type(getattr(resources, name)) is tuple
    for incomplete_fast in (0, 1):
        for incomplete_sparse in (0, 1, 2, 3, 4):
            if incomplete_fast and incomplete_sparse:
                continue
            for unaccepted in (0, 1):
                value = replace(
                    resources,
                    accepted_fast_observations=(full_observation,)
                    * incomplete_fast,
                    accepted_sparse_observations=(full_observation,)
                    * incomplete_sparse,
                    unaccepted_tracking_observations=(full_observation,)
                    * unaccepted,
                    incomplete_fast_pair_observations=incomplete_fast,
                    incomplete_sparse_scan_observations=incomplete_sparse,
                    unaccepted_observations=unaccepted,
                )
                assert value.incomplete_sparse_scan_observations == incomplete_sparse
    invalid = (
        ("incomplete_fast_pair_observations", True, TypeError),
        ("incomplete_fast_pair_observations", 2, ValueError),
        ("incomplete_sparse_scan_observations", -1, ValueError),
        ("incomplete_sparse_scan_observations", 5, ValueError),
        ("unaccepted_observations", False, TypeError),
        ("unaccepted_observations", 2, ValueError),
        ("calibration_resources", object(), TypeError),
        ("accepted_sparse_observations", (object(),), TypeError),
    )
    for name, value, error in invalid:
        with pytest.raises(error):
            replace(resources, **{name: value})
    with pytest.raises(ValueError, match="only one incomplete block"):
        replace(
            resources,
            accepted_fast_observations=(full_observation,),
            accepted_sparse_observations=(full_observation,),
            incomplete_fast_pair_observations=1,
            incomplete_sparse_scan_observations=1,
        )


def test_resource_fast_incomplete_count_matches_fast_tuple_remainder() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseLinewidthEvaluatorResources,
    )

    arguments = _resources_arguments()
    full_observation = arguments["calibration_observations"][0]  # type: ignore[index]
    arguments["accepted_fast_observations"] = (full_observation,)
    with pytest.raises(ValueError, match="incomplete fast-pair count"):
        SparseLinewidthEvaluatorResources(
            **arguments  # type: ignore[arg-type]
        )


def test_resource_sparse_incomplete_count_matches_sparse_tuple_remainder() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseLinewidthEvaluatorResources,
    )

    arguments = _resources_arguments()
    full_observation = arguments["calibration_observations"][0]  # type: ignore[index]
    arguments["accepted_sparse_observations"] = (full_observation,) * 3
    with pytest.raises(ValueError, match="incomplete sparse-scan count"):
        SparseLinewidthEvaluatorResources(
            **arguments  # type: ignore[arg-type]
        )


def test_resource_unaccepted_count_matches_unaccepted_tuple_length() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseLinewidthEvaluatorResources,
    )

    arguments = _resources_arguments()
    full_observation = arguments["calibration_observations"][0]  # type: ignore[index]
    arguments["unaccepted_tracking_observations"] = (full_observation,)
    with pytest.raises(ValueError, match="unaccepted count"):
        SparseLinewidthEvaluatorResources(
            **arguments  # type: ignore[arg-type]
        )


def test_abort_presence_matrix_distinguishes_unavailable_and_authenticated() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseAbortedRun,
        SparseResourceJoinUnavailableAcquisition,
        SparseTrackingAcquisition,
    )

    estimate = _pending_estimate()
    authenticated = SparseTrackingAcquisition(
        **_full_acquisition_arguments()  # type: ignore[arg-type]
    )
    unavailable = SparseResourceJoinUnavailableAcquisition(
        **_full_acquisition_arguments(unavailable=True)  # type: ignore[arg-type]
    )
    for reason in (
        "resource_join_unavailable",
        "tracker_observation_validation_error",
        "tracker_update_construction_error",
        "tracker_update_unexpected_error",
    ):
        missing = reason == "resource_join_unavailable"
        abort = SparseAbortedRun(
            reason,
            None if missing else "RuntimeError",
            None if missing else "",
            unavailable if missing else authenticated,
            1,
            estimate,
            replace(estimate),
        )
        if missing:
            assert abort.exception_message is None
        else:
            assert type(abort.exception_message) is str
        with pytest.raises(ValueError):
            replace(
                abort,
                unaccepted_acquisition=authenticated if missing else unavailable,
            )
    unavailable_abort = SparseAbortedRun(
        "resource_join_unavailable",
        None,
        None,
        unavailable,
        1,
        estimate,
        replace(estimate),
    )
    with pytest.raises(ValueError, match="both be absent"):
        replace(
            unavailable_abort,
            exception_type="RuntimeError",
            exception_message="boom",
        )
    with pytest.raises(ValueError, match="must be one"):
        replace(unavailable_abort, unaccepted_observation_count=0)
    with pytest.raises(ValueError, match="must be equal"):
        replace(unavailable_abort, tracker_estimate_after=replace(estimate, seed=1))


@pytest.mark.parametrize(
    "phase",
    (
        "ready",
        "calibration_succeeded",
        "calibration_failed",
        "tracking",
        "budget_stopped",
        "geometry_stopped",
        "externally_stopped",
        "aborted",
    ),
)
def test_runner_state_phase_presence_matrix_and_cpu_fields(phase: str) -> None:
    from odmr_bench.evaluation.sparse_linewidth import SparseEvaluatorRunnerState

    state = SparseEvaluatorRunnerState(
        **_state_arguments(phase)  # type: ignore[arg-type]
    )
    assert state.phase == phase
    if state.tracker_estimate is None:
        assert (
            state.fast_update_cpu_time_s,
            state.sparse_update_cpu_time_s,
            state.total_update_cpu_time_s,
        ) == (0.0, 0.0, 0.0)
    else:
        assert (
            state.fast_update_cpu_time_s,
            state.sparse_update_cpu_time_s,
            state.total_update_cpu_time_s,
        ) == (
            state.tracker_estimate.fast_update_cpu_time_s,
            state.tracker_estimate.sparse_update_cpu_time_s,
            state.tracker_estimate.total_update_cpu_time_s,
        )
    if phase == "ready":
        invalid = {"calibration": _state_arguments("tracking")["calibration"]}
    elif phase == "calibration_succeeded":
        assert state.verified_calibration is not None
        invalid = {"verified_calibration": replace(state.verified_calibration)}
    elif phase == "calibration_failed":
        invalid = {
            "calibration_outcome": _state_arguments("calibration_succeeded")[
                "calibration_outcome"
            ]
        }
    elif phase == "tracking":
        invalid = {"tracking_resources_before": None}
    elif phase == "budget_stopped":
        invalid = {
            "tracker_estimate": _state_arguments("tracking")["tracker_estimate"]
        }
    elif phase == "geometry_stopped":
        invalid = {
            "tracker_estimate": _state_arguments("budget_stopped")[
                "tracker_estimate"
            ]
        }
    elif phase == "externally_stopped":
        invalid = {
            "terminal_abort": _state_arguments("aborted")["terminal_abort"]
        }
    else:
        invalid = {"terminal_abort": None}
    with pytest.raises(ValueError):
        replace(state, **invalid)
    with pytest.raises(ValueError, match="CPU"):
        replace(state, fast_update_cpu_time_s=1.0)


def test_runner_state_defensive_timing_tuples_and_cardinality() -> None:
    from odmr_bench.evaluation.sparse_linewidth import SparseEvaluatorRunnerState

    arguments = _state_arguments("tracking")
    for name in ("normal_tracking_trace", "pair_timings", "scan_timings"):
        arguments[name] = []
    state = SparseEvaluatorRunnerState(**arguments)  # type: ignore[arg-type]
    assert state.normal_tracking_trace == ()
    assert state.pair_timings == ()
    assert state.scan_timings == ()
    with pytest.raises(TypeError, match="pair_timings"):
        replace(state, pair_timings=[object()])
    with pytest.raises(TypeError, match="scan_timings"):
        replace(state, scan_timings=[object()])


def _resources():
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseLinewidthEvaluatorResources,
    )

    return SparseLinewidthEvaluatorResources(
        **_resources_arguments()  # type: ignore[arg-type]
    )


def test_terminal_outcome_kind_phase_and_abort_resource_matrix() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseEvaluatorRunnerState,
        SparseRunnerAborted,
        SparseRunnerBudgetStopped,
        SparseRunnerExternallyStopped,
        SparseRunnerGeometryStopped,
    )

    resources = _resources()
    budget_state = SparseEvaluatorRunnerState(
        **_state_arguments("budget_stopped")  # type: ignore[arg-type]
    )
    geometry_state = SparseEvaluatorRunnerState(
        **_state_arguments("geometry_stopped")  # type: ignore[arg-type]
    )
    external_state = SparseEvaluatorRunnerState(
        **_state_arguments("externally_stopped")  # type: ignore[arg-type]
    )
    aborted_state = SparseEvaluatorRunnerState(
        **_state_arguments("aborted")  # type: ignore[arg-type]
    )
    diagnostic = geometry_state.tracker_estimate.sparse_geometry_diagnostic  # type: ignore[union-attr]
    abort = aborted_state.terminal_abort
    assert diagnostic is not None and abort is not None
    budget = SparseRunnerBudgetStopped("budget_stopped", resources, budget_state)
    geometry = SparseRunnerGeometryStopped(
        "geometry_stopped", diagnostic, resources, geometry_state
    )
    external = SparseRunnerExternallyStopped(
        "externally_stopped", resources, external_state
    )
    aborted = SparseRunnerAborted("aborted", abort, None, aborted_state)
    assert (budget.kind, geometry.kind, external.kind, aborted.kind) == (
        "budget_stopped",
        "geometry_stopped",
        "externally_stopped",
        "aborted",
    )
    with pytest.raises(ValueError):
        replace(budget, state=external_state)
    with pytest.raises(ValueError):
        replace(geometry, diagnostic=replace(diagnostic, scan_index=1))
    with pytest.raises(ValueError):
        replace(external, kind="budget_stopped")
    with pytest.raises(ValueError, match="resources"):
        replace(aborted, resources=resources)
    authenticated_abort = replace(
        abort,
        reason="tracker_update_unexpected_error",
        exception_type="RuntimeError",
        exception_message="boom",
        unaccepted_acquisition=__import__(
            "odmr_bench.evaluation.sparse_linewidth",
            fromlist=["SparseTrackingAcquisition"],
        ).SparseTrackingAcquisition(
            **_full_acquisition_arguments()  # type: ignore[arg-type]
        ),
    )
    authenticated_state = replace(aborted_state, terminal_abort=authenticated_abort)
    authenticated = SparseRunnerAborted(
        "aborted", authenticated_abort, resources, authenticated_state
    )
    assert authenticated.resources is resources
    with pytest.raises(ValueError, match="resources"):
        replace(authenticated, resources=None)


def test_accepted_and_instrument_failure_outcomes_match_tracking_state() -> None:
    from odmr_bench.evaluation.sparse_linewidth import (
        SparseEvaluatorRunnerState,
        SparseInstrumentQueryFailure,
        SparseRunnerAccepted,
        SparseRunnerInstrumentFailure,
        SparseTrackingAcquisition,
    )
    from tests.estimators.test_sparse_linewidth_tracker import (
        _calibration,
        _fast_observation,
        _reset_tracker,
    )

    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    query = tracker.choose_next_query()
    assert query is not None
    update = tracker.update(_fast_observation(calibration, query))  # type: ignore[arg-type]
    full_args = _full_acquisition_arguments()
    full = full_args["full_observation"]
    full = replace(
        full,
        sequence_index=query.expected_sequence_index,
        timestamp_s=query.expected_end_timestamp_s,
        frequency_hz=query.frequency_hz,
        fluorescence=update.observation.fluorescence,
        integration_time_s=query.integration_time_s,
        nominal_exposure_photons=query.expected_nominal_exposure_photons,
        realized_photons=update.observation.realized_photons,
    )
    from odmr_bench.emulator import ResourceSnapshot
    from odmr_bench.evaluation.two_point.resource_accounting import (
        _advance_full_resources,
    )

    before = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    delta = _advance_full_resources(before, full, 0.001)
    acquisition = SparseTrackingAcquisition(
        "authenticated",
        "fast_pair",
        query,
        query.expected_end_timestamp_s - query.integration_time_s / 2.0,
        full.timestamp_s - full.integration_time_s / 2.0,
        full,
        full.estimator_view(),
        before,
        delta,
        delta,
    )
    accepted_args = _state_arguments("tracking")
    accepted_args.update(
        tracker_estimate=update.estimate,
        normal_tracking_trace=(acquisition,),
        instrument_current_sequence_index=query.expected_sequence_index,
        current_virtual_time_s=query.expected_end_timestamp_s,
        fast_update_cpu_time_s=update.estimate.fast_update_cpu_time_s,
        sparse_update_cpu_time_s=update.estimate.sparse_update_cpu_time_s,
        total_update_cpu_time_s=update.estimate.total_update_cpu_time_s,
    )
    accepted_state = SparseEvaluatorRunnerState(**accepted_args)  # type: ignore[arg-type]
    accepted = SparseRunnerAccepted("accepted", acquisition, update, accepted_state)
    assert accepted.state.tracker_estimate is update.estimate
    with pytest.raises(ValueError, match="match its tracking state"):
        replace(accepted, acquisition=replace(acquisition))

    pending = _pending_estimate()
    failure = SparseInstrumentQueryFailure(
        "fast_pair",
        pending.pending_query,
        "RuntimeError",
        "boom",
        before,
        replace(before),
    )
    failure_state = SparseEvaluatorRunnerState(
        **(
            _state_arguments("tracking")
            | {"tracker_estimate": pending, "last_instrument_failure": failure}
        )  # type: ignore[arg-type]
    )
    outcome = SparseRunnerInstrumentFailure(
        "instrument_failure", failure, failure_state
    )
    assert outcome.state.last_instrument_failure is failure
    externally_stopped = replace(failure_state, phase="externally_stopped")
    assert externally_stopped.last_instrument_failure is failure
    with pytest.raises(ValueError):
        replace(outcome, failure=replace(failure))
