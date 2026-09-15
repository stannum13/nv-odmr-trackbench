"""Evaluator-owned value contracts for sparse-linewidth tracking runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np

from odmr_bench.emulator.observations import (
    EstimatorObservation,
    InstrumentObservation,
)
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators.sparse_linewidth_types import (
    CompositeMode,
    SparseGeometryUnavailableDiagnostic,
    SparseLinewidthCompositeEstimate,
    SparseLinewidthCompositeUpdate,
    SparseLinewidthQuery,
)
from odmr_bench.estimators.two_point_types import (
    CalibrationBudgetTreatment,
    TwoPointCalibration,
    TwoPointQuery,
)
from odmr_bench.evaluation.two_point.types import (
    ResourceJoinMismatchField,
    TwoPointEvaluatorInstrumentConfiguration,
    TwoPointEvaluatorPairTiming,
    VerifiedInstrumentRunToken,
    VerifiedTwoPointCalibrationFailure,
    VerifiedTwoPointCalibrationOutcome,
    VerifiedTwoPointCalibrationSuccess,
)

SparseRunnerPhase: TypeAlias = Literal[
    "ready",
    "calibration_succeeded",
    "calibration_failed",
    "tracking",
    "budget_stopped",
    "geometry_stopped",
    "externally_stopped",
    "aborted",
]
SparseAbortReason: TypeAlias = Literal[
    "resource_join_unavailable",
    "tracker_observation_validation_error",
    "tracker_update_construction_error",
    "tracker_update_unexpected_error",
]
SparsePreflightCode: TypeAlias = Literal[
    "invalid_runner_phase",
    "invalid_argument_type",
    "invalid_argument_value",
    "invalid_frequency_grid",
    "invalid_fit_or_identity_configuration",
    "invalid_clock_mapping",
    "unclean_instrument_boundary",
]
SparseStartCode: TypeAlias = Literal[
    "invalid_runner_phase",
    "invalid_argument_type",
    "unverified_calibration",
    "calibration_mismatch",
    "run_provenance_mismatch",
    "metadata_mismatch",
    "resource_boundary_mismatch",
    "tracker_reset_failed",
]

_SPARSE_RUNNER_PHASES = frozenset(
    {
        "ready",
        "calibration_succeeded",
        "calibration_failed",
        "tracking",
        "budget_stopped",
        "geometry_stopped",
        "externally_stopped",
        "aborted",
    }
)
_SPARSE_ABORT_REASONS = frozenset(
    {
        "resource_join_unavailable",
        "tracker_observation_validation_error",
        "tracker_update_construction_error",
        "tracker_update_unexpected_error",
    }
)
_SPARSE_PREFLIGHT_CODES = frozenset(
    {
        "invalid_runner_phase",
        "invalid_argument_type",
        "invalid_argument_value",
        "invalid_frequency_grid",
        "invalid_fit_or_identity_configuration",
        "invalid_clock_mapping",
        "unclean_instrument_boundary",
    }
)
_SPARSE_START_CODES = frozenset(
    {
        "invalid_runner_phase",
        "invalid_argument_type",
        "unverified_calibration",
        "calibration_mismatch",
        "run_provenance_mismatch",
        "metadata_mismatch",
        "resource_boundary_mismatch",
        "tracker_reset_failed",
    }
)
_RESOURCE_JOIN_MISMATCH_FIELDS = (
    "observations",
    "integration_time_s",
    "nominal_exposure_photons",
    "expected_photons",
    "realized_photons",
    "observations_without_realized_counts",
    "virtual_elapsed_time_s",
)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    try:
        canonical = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    return canonical


def _nonnegative_float(value: object, name: str) -> float:
    canonical = _finite_float(value, name)
    if canonical < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return canonical


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    canonical = int(value)
    if canonical < 0:
        raise ValueError(f"{name} must be non-negative")
    return canonical


def _closed_error_code(code: object, allowed: frozenset[str]) -> str:
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    canonical = str.__str__(code)
    if canonical not in allowed:
        raise ValueError(f"unknown sparse evaluator error code: {canonical!r}")
    return canonical


def _closed_literal(
    value: object, name: str, allowed: frozenset[str]
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    canonical = str.__str__(value)
    if canonical not in allowed:
        raise ValueError(f"{name} must be one of {tuple(sorted(allowed))}")
    return canonical


def _exact_discriminator(value: object, name: str, expected: str) -> str:
    return _closed_literal(value, name, frozenset({expected}))


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    canonical = str.__str__(value)
    if not canonical:
        raise ValueError(f"{name} must be nonempty")
    return canonical


def _optional_exception_strings(
    exception_type: object, exception_message: object, *, required: bool
) -> tuple[str | None, str | None]:
    if not required:
        if exception_type is not None or exception_message is not None:
            raise ValueError("exception fields must both be absent")
        return None, None
    exception_type = _nonempty_string(exception_type, "exception_type")
    if not isinstance(exception_message, str):
        raise TypeError("exception_message must be a string")
    return exception_type, str.__str__(exception_message)


def _canonical_resource_mismatch_fields(
    values: object,
) -> tuple[ResourceJoinMismatchField, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("resource_mismatch_fields must be an ordered sequence")
    supplied = tuple(values)
    if any(not isinstance(value, str) for value in supplied):
        raise TypeError("resource_mismatch_fields must contain strings")
    canonical = tuple(str.__str__(value) for value in supplied)
    if any(value not in _RESOURCE_JOIN_MISMATCH_FIELDS for value in canonical):
        raise ValueError("resource_mismatch_fields contains an unknown field")
    expected = tuple(
        value for value in _RESOURCE_JOIN_MISMATCH_FIELDS if value in canonical
    )
    if canonical != expected:
        raise ValueError(
            "resource_mismatch_fields must be unique and in declaration order"
        )
    return canonical  # type: ignore[return-value]


def _query_mode(
    mode: object, query: object
) -> tuple[CompositeMode, TwoPointQuery | SparseLinewidthQuery]:
    canonical_mode = _closed_literal(
        mode, "mode", frozenset({"fast_pair", "sparse_scan"})
    )
    if type(query) not in {TwoPointQuery, SparseLinewidthQuery}:
        raise TypeError("query must be an exact public query")
    expected_mode = "fast_pair" if type(query) is TwoPointQuery else "sparse_scan"
    if canonical_mode != expected_mode:
        raise ValueError("mode must match query type")
    return canonical_mode, query  # type: ignore[return-value]


def _canonical_optional_midpoint(
    value: object, *, endpoint_s: float, name: str
) -> float | None:
    if value is None:
        return None
    midpoint = _nonnegative_float(value, name)
    if midpoint > endpoint_s:
        raise ValueError(f"{name} must not exceed its endpoint")
    return midpoint


def _validated_acquisition_fields(
    mode: object,
    query: object,
    expected_measurement_midpoint_s: object,
    measurement_midpoint_s: object,
    full_observation: object,
    safe_observation: object,
    instrument_resources_before: object,
    instrument_resources_after: object,
) -> tuple[CompositeMode, float, float | None]:
    canonical_mode, canonical_query = _query_mode(mode, query)
    if type(full_observation) is not InstrumentObservation:
        raise TypeError("full_observation must be an InstrumentObservation")
    if type(safe_observation) is not EstimatorObservation:
        raise TypeError("safe_observation must be an EstimatorObservation")
    if full_observation.estimator_view() != safe_observation:
        raise ValueError("safe_observation must exactly project full_observation")
    if type(instrument_resources_before) is not ResourceSnapshot:
        raise TypeError("instrument_resources_before must be a ResourceSnapshot")
    if type(instrument_resources_after) is not ResourceSnapshot:
        raise TypeError("instrument_resources_after must be a ResourceSnapshot")
    expected_midpoint = _canonical_optional_midpoint(
        expected_measurement_midpoint_s,
        endpoint_s=canonical_query.expected_end_timestamp_s,
        name="expected_measurement_midpoint_s",
    )
    if expected_midpoint is None:
        raise TypeError("expected_measurement_midpoint_s must be a real scalar")
    measurement_midpoint = _canonical_optional_midpoint(
        measurement_midpoint_s,
        endpoint_s=full_observation.timestamp_s,
        name="measurement_midpoint_s",
    )
    return canonical_mode, expected_midpoint, measurement_midpoint


def _exact_record_sequence(
    values: object, name: str, record_type: type
) -> tuple[object, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be an ordered sequence")
    canonical = tuple(values)
    if not all(type(value) is record_type for value in canonical):
        raise TypeError(f"{name} must contain exact {record_type.__name__} values")
    return canonical


def _bounded_count(value: object, name: str, maximum: int) -> int:
    canonical = _nonnegative_int(value, name)
    if canonical > maximum:
        if maximum == 1:
            raise ValueError(f"{name} must be zero or one")
        raise ValueError(f"{name} must be between zero and {maximum}")
    return canonical


class SparsePreflightError(ValueError):
    """Reject invalid sparse verified-calibration preflight inputs."""

    code: SparsePreflightCode

    def __init__(self, code: SparsePreflightCode) -> None:
        self.code = _closed_error_code(code, _SPARSE_PREFLIGHT_CODES)
        super().__init__(self.code)


class SparseStartError(ValueError):
    """Reject an invalid transition into sparse-linewidth tracking."""

    code: SparseStartCode

    def __init__(self, code: SparseStartCode) -> None:
        self.code = _closed_error_code(code, _SPARSE_START_CODES)
        super().__init__(self.code)


class SparseRunnerStateError(RuntimeError):
    """Reject an operation invalid for the current sparse runner phase."""


@dataclass(frozen=True, slots=True)
class SparseTrackingAcquisition:
    """One full sparse evaluator acquisition with an authenticated atom."""

    resource_join_status: Literal["authenticated"]
    mode: CompositeMode
    query: TwoPointQuery | SparseLinewidthQuery
    expected_measurement_midpoint_s: float
    measurement_midpoint_s: float | None
    full_observation: InstrumentObservation
    safe_observation: EstimatorObservation
    instrument_resources_before: ResourceSnapshot
    instrument_resources_after: ResourceSnapshot
    instrument_resource_delta: ResourceSnapshot

    def __post_init__(self) -> None:
        status = _exact_discriminator(
            self.resource_join_status,
            "resource_join_status",
            "authenticated",
        )
        mode, expected_midpoint, midpoint = _validated_acquisition_fields(
            self.mode,
            self.query,
            self.expected_measurement_midpoint_s,
            self.measurement_midpoint_s,
            self.full_observation,
            self.safe_observation,
            self.instrument_resources_before,
            self.instrument_resources_after,
        )
        if type(self.instrument_resource_delta) is not ResourceSnapshot:
            raise TypeError("instrument_resource_delta must be a ResourceSnapshot")
        overhead_s = (
            self.instrument_resource_delta.virtual_elapsed_time_s
            - self.full_observation.integration_time_s
        )
        if overhead_s < 0.0:
            raise ValueError("instrument_resource_delta must include integration time")
        from odmr_bench.evaluation.two_point.resource_accounting import (
            _advance_full_resources,
            _resource_mismatch_fields,
        )

        zero = ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
        expected_delta = _advance_full_resources(
            zero, self.full_observation, overhead_s
        )
        if self.instrument_resource_delta != expected_delta:
            raise ValueError(
                "instrument_resource_delta must be the exact one-observation atom"
            )
        expected_after = _advance_full_resources(
            self.instrument_resources_before, self.full_observation, overhead_s
        )
        if _resource_mismatch_fields(
            expected_after, self.instrument_resources_after
        ):
            raise ValueError(
                "instrument_resources_after must exactly join the acquisition atom"
            )
        object.__setattr__(self, "resource_join_status", status)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self, "expected_measurement_midpoint_s", expected_midpoint
        )
        object.__setattr__(self, "measurement_midpoint_s", midpoint)


@dataclass(frozen=True, slots=True)
class SparseResourceJoinUnavailableAcquisition:
    """One raw acquisition whose evaluator resource atom cannot be joined."""

    resource_join_status: Literal["unavailable"]
    mode: CompositeMode
    query: TwoPointQuery | SparseLinewidthQuery
    expected_measurement_midpoint_s: float
    measurement_midpoint_s: float | None
    full_observation: InstrumentObservation
    safe_observation: EstimatorObservation
    resource_mismatch_fields: tuple[ResourceJoinMismatchField, ...]
    instrument_resources_before: ResourceSnapshot
    instrument_resources_after: ResourceSnapshot

    def __post_init__(self) -> None:
        status = _exact_discriminator(
            self.resource_join_status, "resource_join_status", "unavailable"
        )
        mode, expected_midpoint, midpoint = _validated_acquisition_fields(
            self.mode,
            self.query,
            self.expected_measurement_midpoint_s,
            self.measurement_midpoint_s,
            self.full_observation,
            self.safe_observation,
            self.instrument_resources_before,
            self.instrument_resources_after,
        )
        mismatch_fields = _canonical_resource_mismatch_fields(
            self.resource_mismatch_fields
        )
        if not mismatch_fields:
            raise ValueError(
                "unavailable acquisition requires resource_mismatch_fields"
            )
        object.__setattr__(self, "resource_join_status", status)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self, "expected_measurement_midpoint_s", expected_midpoint
        )
        object.__setattr__(self, "measurement_midpoint_s", midpoint)
        object.__setattr__(self, "resource_mismatch_fields", mismatch_fields)


@dataclass(frozen=True, slots=True)
class SparseInstrumentQueryFailure:
    """One atomic instrument exception and its unchanged resource boundary."""

    mode: CompositeMode
    query: TwoPointQuery | SparseLinewidthQuery
    exception_type: str
    exception_message: str
    instrument_resources_before: ResourceSnapshot
    instrument_resources_after: ResourceSnapshot

    def __post_init__(self) -> None:
        mode, _ = _query_mode(self.mode, self.query)
        exception_type = _nonempty_string(self.exception_type, "exception_type")
        if not isinstance(self.exception_message, str):
            raise TypeError("exception_message must be a string")
        exception_message = str.__str__(self.exception_message)
        if type(self.instrument_resources_before) is not ResourceSnapshot:
            raise TypeError("instrument_resources_before must be a ResourceSnapshot")
        if type(self.instrument_resources_after) is not ResourceSnapshot:
            raise TypeError("instrument_resources_after must be a ResourceSnapshot")
        if self.instrument_resources_before != self.instrument_resources_after:
            raise ValueError(
                "instrument query failure must preserve its resource boundary"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "exception_type", exception_type)
        object.__setattr__(self, "exception_message", exception_message)


@dataclass(frozen=True, slots=True)
class SparseAbortedRun:
    """Terminal evidence for one returned but unaccepted acquisition."""

    reason: SparseAbortReason
    exception_type: str | None
    exception_message: str | None
    unaccepted_acquisition: (
        SparseTrackingAcquisition | SparseResourceJoinUnavailableAcquisition
    )
    unaccepted_observation_count: Literal[1]
    tracker_estimate_before: SparseLinewidthCompositeEstimate
    tracker_estimate_after: SparseLinewidthCompositeEstimate

    def __post_init__(self) -> None:
        reason = _closed_literal(
            self.reason, "reason", _SPARSE_ABORT_REASONS
        )
        unavailable = reason == "resource_join_unavailable"
        expected_type = (
            SparseResourceJoinUnavailableAcquisition
            if unavailable
            else SparseTrackingAcquisition
        )
        if type(self.unaccepted_acquisition) is not expected_type:
            raise ValueError("unaccepted acquisition must match abort reason")
        exception_type, exception_message = _optional_exception_strings(
            self.exception_type,
            self.exception_message,
            required=not unavailable,
        )
        unaccepted_count = _nonnegative_int(
            self.unaccepted_observation_count, "unaccepted_observation_count"
        )
        if unaccepted_count != 1:
            raise ValueError("unaccepted_observation_count must be one")
        if type(self.tracker_estimate_before) is not SparseLinewidthCompositeEstimate:
            raise TypeError(
                "tracker_estimate_before must be a SparseLinewidthCompositeEstimate"
            )
        if type(self.tracker_estimate_after) is not SparseLinewidthCompositeEstimate:
            raise TypeError(
                "tracker_estimate_after must be a SparseLinewidthCompositeEstimate"
            )
        if self.tracker_estimate_before != self.tracker_estimate_after:
            raise ValueError("tracker estimates before and after abort must be equal")
        if (
            self.tracker_estimate_before.pending_query is None
            or self.tracker_estimate_before.pending_query
            != self.unaccepted_acquisition.query
        ):
            raise ValueError(
                "abort tracker estimates must retain the unaccepted pending query"
            )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "exception_type", exception_type)
        object.__setattr__(self, "exception_message", exception_message)
        object.__setattr__(
            self, "unaccepted_observation_count", unaccepted_count
        )


@dataclass(frozen=True, slots=True)
class SparseLinewidthEvaluatorScanTiming:
    """Evaluator-only physical and public timing for one completed scan."""

    scan_index: int
    resonance_id: str
    measurement_midpoints_s: tuple[float, float, float, float, float]
    truth_reference_timestamp_s: float
    public_reference_timestamp_s: float
    release_sequence_index: int
    release_timestamp_s: float

    def __post_init__(self) -> None:
        scan_index = _nonnegative_int(self.scan_index, "scan_index")
        resonance_id = _nonempty_string(self.resonance_id, "resonance_id")
        if not isinstance(self.measurement_midpoints_s, (tuple, list)):
            raise TypeError("measurement_midpoints_s must be an ordered sequence")
        if len(self.measurement_midpoints_s) != 5:
            raise ValueError("measurement_midpoints_s must contain exactly five values")
        midpoints = tuple(
            _nonnegative_float(value, "measurement midpoint")
            for value in self.measurement_midpoints_s
        )
        if any(current <= previous for previous, current in pairwise(midpoints)):
            raise ValueError("measurement midpoints must be strictly increasing")
        truth_reference = _nonnegative_float(
            self.truth_reference_timestamp_s, "truth_reference_timestamp_s"
        )
        expected_truth = midpoints[0]
        for count, value in enumerate(midpoints[1:], start=2):
            expected_truth = expected_truth + (value - expected_truth) / count
        if truth_reference != expected_truth:
            raise ValueError(
                "truth_reference_timestamp_s must be the ordered five-value mean"
            )
        public_reference = _nonnegative_float(
            self.public_reference_timestamp_s,
            "public_reference_timestamp_s",
        )
        release_sequence_index = _nonnegative_int(
            self.release_sequence_index, "release_sequence_index"
        )
        release_timestamp = _nonnegative_float(
            self.release_timestamp_s, "release_timestamp_s"
        )
        if release_timestamp < midpoints[-1]:
            raise ValueError("release timestamp must not precede the fifth midpoint")
        if release_timestamp < public_reference:
            raise ValueError("release timestamp must not precede the public reference")
        object.__setattr__(self, "scan_index", scan_index)
        object.__setattr__(self, "resonance_id", resonance_id)
        object.__setattr__(self, "measurement_midpoints_s", midpoints)
        object.__setattr__(self, "truth_reference_timestamp_s", truth_reference)
        object.__setattr__(self, "public_reference_timestamp_s", public_reference)
        object.__setattr__(self, "release_sequence_index", release_sequence_index)
        object.__setattr__(self, "release_timestamp_s", release_timestamp)


@dataclass(frozen=True, slots=True)
class SparseLinewidthEvaluatorResources:
    """Lossless evaluator-owned resources for one composite tracking run."""

    calibration_observations: tuple[InstrumentObservation, ...]
    accepted_fast_observations: tuple[InstrumentObservation, ...]
    accepted_sparse_observations: tuple[InstrumentObservation, ...]
    accepted_tracking_observations: tuple[InstrumentObservation, ...]
    unaccepted_tracking_observations: tuple[InstrumentObservation, ...]
    calibration_resources: ResourceSnapshot
    fast_tracking_resources: ResourceSnapshot
    sparse_tracking_resources: ResourceSnapshot
    tracking_resources: ResourceSnapshot
    accepted_charged_resources: ResourceSnapshot
    charged_resources: ResourceSnapshot
    calibration_budget_treatment: CalibrationBudgetTreatment
    incomplete_fast_pair_observations: Literal[0, 1]
    incomplete_sparse_scan_observations: Literal[0, 1, 2, 3, 4]
    unaccepted_observations: Literal[0, 1]

    def __post_init__(self) -> None:
        for name in (
            "calibration_observations",
            "accepted_fast_observations",
            "accepted_sparse_observations",
            "accepted_tracking_observations",
            "unaccepted_tracking_observations",
        ):
            object.__setattr__(
                self,
                name,
                _exact_record_sequence(
                    getattr(self, name), name, InstrumentObservation
                ),
            )
        for name in (
            "calibration_resources",
            "fast_tracking_resources",
            "sparse_tracking_resources",
            "tracking_resources",
            "accepted_charged_resources",
            "charged_resources",
        ):
            if type(getattr(self, name)) is not ResourceSnapshot:
                raise TypeError(f"{name} must be a ResourceSnapshot")
        treatment = _closed_literal(
            self.calibration_budget_treatment,
            "calibration_budget_treatment",
            frozenset({"included_same_run", "conditional_free_precalibration"}),
        )
        incomplete_fast = _bounded_count(
            self.incomplete_fast_pair_observations,
            "incomplete_fast_pair_observations",
            1,
        )
        incomplete_sparse = _bounded_count(
            self.incomplete_sparse_scan_observations,
            "incomplete_sparse_scan_observations",
            4,
        )
        if incomplete_fast and incomplete_sparse:
            raise ValueError("only one incomplete block may be present")
        unaccepted = _bounded_count(
            self.unaccepted_observations, "unaccepted_observations", 1
        )
        if incomplete_fast != len(self.accepted_fast_observations) % 2:
            raise ValueError(
                "incomplete fast-pair count must match accepted fast observations"
            )
        if incomplete_sparse != len(self.accepted_sparse_observations) % 5:
            raise ValueError(
                "incomplete sparse-scan count must match accepted sparse observations"
            )
        if unaccepted != len(self.unaccepted_tracking_observations):
            raise ValueError(
                "unaccepted count must match unaccepted tracking observations"
            )
        object.__setattr__(self, "calibration_budget_treatment", treatment)
        object.__setattr__(
            self, "incomplete_fast_pair_observations", incomplete_fast
        )
        object.__setattr__(
            self, "incomplete_sparse_scan_observations", incomplete_sparse
        )
        object.__setattr__(self, "unaccepted_observations", unaccepted)


@dataclass(frozen=True, slots=True)
class SparseEvaluatorRunnerState:
    """Frozen audit snapshot of the sparse evaluator state machine."""

    phase: SparseRunnerPhase
    run_token: VerifiedInstrumentRunToken
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration
    calibration_outcome: VerifiedTwoPointCalibrationOutcome | None
    verified_calibration: VerifiedTwoPointCalibrationSuccess | None
    calibration: TwoPointCalibration | None
    tracker_estimate: SparseLinewidthCompositeEstimate | None
    normal_tracking_trace: tuple[SparseTrackingAcquisition, ...]
    pair_timings: tuple[TwoPointEvaluatorPairTiming, ...]
    scan_timings: tuple[SparseLinewidthEvaluatorScanTiming, ...]
    instrument_resources_at_bind: ResourceSnapshot
    tracking_resources_before: ResourceSnapshot | None
    instrument_resources_current: ResourceSnapshot
    instrument_current_sequence_index: int | None
    current_virtual_time_s: float
    last_instrument_failure: SparseInstrumentQueryFailure | None
    terminal_abort: SparseAbortedRun | None
    fast_update_cpu_time_s: float
    sparse_update_cpu_time_s: float
    total_update_cpu_time_s: float

    def __post_init__(self) -> None:
        phase = _closed_literal(self.phase, "phase", _SPARSE_RUNNER_PHASES)
        if type(self.run_token) is not VerifiedInstrumentRunToken:
            raise TypeError("run_token must be a VerifiedInstrumentRunToken")
        if (
            type(self.instrument_configuration)
            is not TwoPointEvaluatorInstrumentConfiguration
        ):
            raise TypeError(
                "instrument_configuration must be a "
                "TwoPointEvaluatorInstrumentConfiguration"
            )
        if self.calibration_outcome is not None and type(
            self.calibration_outcome
        ) not in {
            VerifiedTwoPointCalibrationSuccess,
            VerifiedTwoPointCalibrationFailure,
        }:
            raise TypeError("calibration_outcome has an invalid type")
        if self.verified_calibration is not None and type(
            self.verified_calibration
        ) is not VerifiedTwoPointCalibrationSuccess:
            raise TypeError(
                "verified_calibration must be a VerifiedTwoPointCalibrationSuccess"
            )
        if (
            self.calibration is not None
            and type(self.calibration) is not TwoPointCalibration
        ):
            raise TypeError("calibration must be a TwoPointCalibration")
        if self.tracker_estimate is not None and type(
            self.tracker_estimate
        ) is not SparseLinewidthCompositeEstimate:
            raise TypeError(
                "tracker_estimate must be a SparseLinewidthCompositeEstimate"
            )
        normal_trace = _exact_record_sequence(
            self.normal_tracking_trace,
            "normal_tracking_trace",
            SparseTrackingAcquisition,
        )
        pair_timings = _exact_record_sequence(
            self.pair_timings, "pair_timings", TwoPointEvaluatorPairTiming
        )
        scan_timings = _exact_record_sequence(
            self.scan_timings,
            "scan_timings",
            SparseLinewidthEvaluatorScanTiming,
        )
        if type(self.instrument_resources_at_bind) is not ResourceSnapshot:
            raise TypeError("instrument_resources_at_bind must be a ResourceSnapshot")
        if self.tracking_resources_before is not None and type(
            self.tracking_resources_before
        ) is not ResourceSnapshot:
            raise TypeError("tracking_resources_before must be a ResourceSnapshot")
        if type(self.instrument_resources_current) is not ResourceSnapshot:
            raise TypeError("instrument_resources_current must be a ResourceSnapshot")
        sequence_index = (
            None
            if self.instrument_current_sequence_index is None
            else _nonnegative_int(
                self.instrument_current_sequence_index,
                "instrument_current_sequence_index",
            )
        )
        current_virtual_time_s = _nonnegative_float(
            self.current_virtual_time_s, "current_virtual_time_s"
        )
        if self.last_instrument_failure is not None and type(
            self.last_instrument_failure
        ) is not SparseInstrumentQueryFailure:
            raise TypeError(
                "last_instrument_failure must be a SparseInstrumentQueryFailure"
            )
        if self.terminal_abort is not None and type(
            self.terminal_abort
        ) is not SparseAbortedRun:
            raise TypeError("terminal_abort must be a SparseAbortedRun")
        cpu = tuple(
            _nonnegative_float(value, name)
            for name, value in (
                ("fast_update_cpu_time_s", self.fast_update_cpu_time_s),
                ("sparse_update_cpu_time_s", self.sparse_update_cpu_time_s),
                ("total_update_cpu_time_s", self.total_update_cpu_time_s),
            )
        )

        active = phase in {
            "tracking",
            "budget_stopped",
            "geometry_stopped",
            "externally_stopped",
            "aborted",
        }
        if phase == "ready":
            if any(
                value is not None
                for value in (
                    self.calibration_outcome,
                    self.verified_calibration,
                    self.calibration,
                    self.tracker_estimate,
                    self.tracking_resources_before,
                )
            ):
                raise ValueError("ready phase must not contain run state")
        elif phase == "calibration_failed":
            if (
                type(self.calibration_outcome)
                is not VerifiedTwoPointCalibrationFailure
                or any(
                    value is not None
                    for value in (
                        self.verified_calibration,
                        self.calibration,
                        self.tracker_estimate,
                        self.tracking_resources_before,
                    )
                )
            ):
                raise ValueError(
                    "calibration_failed phase requires only one failure outcome"
                )
        elif phase == "calibration_succeeded":
            if (
                type(self.calibration_outcome) is not VerifiedTwoPointCalibrationSuccess
                or self.verified_calibration is not self.calibration_outcome
                or any(
                    value is not None
                    for value in (
                        self.calibration,
                        self.tracker_estimate,
                        self.tracking_resources_before,
                    )
                )
            ):
                raise ValueError(
                    "calibration_succeeded phase requires the exact success outcome"
                )
        elif active:
            if (
                type(self.verified_calibration)
                is not VerifiedTwoPointCalibrationSuccess
                or type(self.calibration) is not TwoPointCalibration
                or type(self.tracker_estimate) is not SparseLinewidthCompositeEstimate
                or type(self.tracking_resources_before) is not ResourceSnapshot
            ):
                raise ValueError("active runner phases require tracking state")
            if self.calibration_outcome is not None and (
                type(self.calibration_outcome) is not VerifiedTwoPointCalibrationSuccess
                or self.calibration_outcome is not self.verified_calibration
            ):
                raise ValueError(
                    "stored calibration outcome must be the exact verified success"
                )
            if len(normal_trace) != self.tracker_estimate.accepted_observations:
                raise ValueError(
                    "normal tracking trace must match accepted observation count"
                )
            if len(pair_timings) != self.tracker_estimate.completed_fast_pairs:
                raise ValueError("pair timings must match completed fast-pair history")
            if len(scan_timings) != self.tracker_estimate.completed_sparse_scans:
                raise ValueError(
                    "scan timings must match completed sparse-scan history"
                )
            expected_stop = {
                "budget_stopped": "budget_exhausted",
                "geometry_stopped": "sparse_geometry_unavailable",
            }.get(phase)
            if self.tracker_estimate.stopped_reason != expected_stop:
                raise ValueError("terminal phase must match tracker stop state")
            estimate_cpu = (
                self.tracker_estimate.fast_update_cpu_time_s,
                self.tracker_estimate.sparse_update_cpu_time_s,
                self.tracker_estimate.total_update_cpu_time_s,
            )
            if cpu != estimate_cpu:
                raise ValueError("runner CPU totals must match tracker CPU totals")

        if not active and (normal_trace or pair_timings or scan_timings):
            raise ValueError("pre-tracking phases must not contain tracking history")
        if not active and any(cpu):
            raise ValueError("pre-tracking phases require zero CPU totals")
        if (self.terminal_abort is not None) != (phase == "aborted"):
            raise ValueError("terminal_abort must be present exactly when aborted")
        if self.last_instrument_failure is not None and phase != "tracking":
            raise ValueError("instrument failure may appear only while tracking")
        if self.last_instrument_failure is not None and (
            self.tracker_estimate is None
            or self.tracker_estimate.pending_query
            != self.last_instrument_failure.query
        ):
            raise ValueError(
                "instrument failure must match the retained pending query"
            )
        if phase == "aborted" and (
            self.terminal_abort is None
            or self.terminal_abort.tracker_estimate_after != self.tracker_estimate
        ):
            raise ValueError("terminal abort must match the runner estimate")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "normal_tracking_trace", normal_trace)
        object.__setattr__(self, "pair_timings", pair_timings)
        object.__setattr__(self, "scan_timings", scan_timings)
        object.__setattr__(self, "instrument_current_sequence_index", sequence_index)
        object.__setattr__(self, "current_virtual_time_s", current_virtual_time_s)
        object.__setattr__(self, "fast_update_cpu_time_s", cpu[0])
        object.__setattr__(self, "sparse_update_cpu_time_s", cpu[1])
        object.__setattr__(self, "total_update_cpu_time_s", cpu[2])


def _outcome_state(
    value: object, *, phase: SparseRunnerPhase
) -> SparseEvaluatorRunnerState:
    if type(value) is not SparseEvaluatorRunnerState:
        raise TypeError("state must be a SparseEvaluatorRunnerState")
    if value.phase != phase:
        raise ValueError(f"outcome requires {phase} state")
    return value


@dataclass(frozen=True, slots=True)
class SparseRunnerAccepted:
    """One composite acquisition accepted by the tracker."""

    kind: Literal["accepted"]
    acquisition: SparseTrackingAcquisition
    update: SparseLinewidthCompositeUpdate
    state: SparseEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(self.kind, "kind", "accepted")
        if type(self.acquisition) is not SparseTrackingAcquisition:
            raise TypeError("acquisition must be a SparseTrackingAcquisition")
        if type(self.update) is not SparseLinewidthCompositeUpdate:
            raise TypeError("update must be a SparseLinewidthCompositeUpdate")
        state = _outcome_state(self.state, phase="tracking")
        if (
            self.acquisition.query != self.update.query
            or self.acquisition.safe_observation != self.update.observation
            or state.tracker_estimate is not self.update.estimate
            or not state.normal_tracking_trace
            or state.normal_tracking_trace[-1] is not self.acquisition
        ):
            raise ValueError("accepted outcome must match its tracking state")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class SparseRunnerInstrumentFailure:
    """One retryable atomic instrument query failure."""

    kind: Literal["instrument_failure"]
    failure: SparseInstrumentQueryFailure
    state: SparseEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(self.kind, "kind", "instrument_failure")
        if type(self.failure) is not SparseInstrumentQueryFailure:
            raise TypeError("failure must be a SparseInstrumentQueryFailure")
        state = _outcome_state(self.state, phase="tracking")
        if state.last_instrument_failure is not self.failure:
            raise ValueError("instrument failure outcome must match tracking state")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class SparseRunnerBudgetStopped:
    """Terminal outcome at an unaffordable acquisition-block boundary."""

    kind: Literal["budget_stopped"]
    resources: SparseLinewidthEvaluatorResources
    state: SparseEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(self.kind, "kind", "budget_stopped")
        if type(self.resources) is not SparseLinewidthEvaluatorResources:
            raise TypeError("resources must be SparseLinewidthEvaluatorResources")
        _outcome_state(self.state, phase="budget_stopped")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class SparseRunnerGeometryStopped:
    """Terminal outcome at a due but unavailable sparse geometry boundary."""

    kind: Literal["geometry_stopped"]
    diagnostic: SparseGeometryUnavailableDiagnostic
    resources: SparseLinewidthEvaluatorResources
    state: SparseEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(self.kind, "kind", "geometry_stopped")
        if type(self.diagnostic) is not SparseGeometryUnavailableDiagnostic:
            raise TypeError("diagnostic must be a SparseGeometryUnavailableDiagnostic")
        if type(self.resources) is not SparseLinewidthEvaluatorResources:
            raise TypeError("resources must be SparseLinewidthEvaluatorResources")
        state = _outcome_state(self.state, phase="geometry_stopped")
        if (
            state.tracker_estimate is None
            or state.tracker_estimate.sparse_geometry_diagnostic != self.diagnostic
        ):
            raise ValueError("geometry outcome diagnostic must match its state")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class SparseRunnerExternallyStopped:
    """Terminal outcome requested by the caller."""

    kind: Literal["externally_stopped"]
    resources: SparseLinewidthEvaluatorResources
    state: SparseEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(self.kind, "kind", "externally_stopped")
        if type(self.resources) is not SparseLinewidthEvaluatorResources:
            raise TypeError("resources must be SparseLinewidthEvaluatorResources")
        _outcome_state(self.state, phase="externally_stopped")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class SparseRunnerAborted:
    """Terminal outcome after a returned acquisition cannot be accepted."""

    kind: Literal["aborted"]
    abort: SparseAbortedRun
    resources: SparseLinewidthEvaluatorResources | None
    state: SparseEvaluatorRunnerState

    def __post_init__(self) -> None:
        kind = _exact_discriminator(self.kind, "kind", "aborted")
        if type(self.abort) is not SparseAbortedRun:
            raise TypeError("abort must be a SparseAbortedRun")
        if self.resources is not None and type(
            self.resources
        ) is not SparseLinewidthEvaluatorResources:
            raise TypeError(
                "resources must be SparseLinewidthEvaluatorResources or None"
            )
        state = _outcome_state(self.state, phase="aborted")
        unavailable = type(
            self.abort.unaccepted_acquisition
        ) is SparseResourceJoinUnavailableAcquisition
        if (
            state.terminal_abort is not self.abort
            or unavailable != (self.resources is None)
        ):
            raise ValueError("aborted outcome resources must match its terminal state")
        object.__setattr__(self, "kind", kind)


SparseRunnerStepOutcome: TypeAlias = (
    SparseRunnerAccepted
    | SparseRunnerInstrumentFailure
    | SparseRunnerBudgetStopped
    | SparseRunnerGeometryStopped
    | SparseRunnerAborted
)
SparseRunnerRunOutcome: TypeAlias = (
    SparseRunnerInstrumentFailure
    | SparseRunnerBudgetStopped
    | SparseRunnerGeometryStopped
    | SparseRunnerAborted
)
