# Calibrated Two-Point Center Tracker Design

## Purpose and status

Stage 6.3 adds the first adaptive within-sweep estimator. It tracks the center
of each configured resonance by acquiring two local flanks, converting their
normalized signed difference to hertz through a public calibration, and
applying a bounded proportional correction. It does not estimate linewidth,
amplitude, eta, contrast, Q, or uncertainty.

This document fixes the implementation contract; it is not an implementation
plan and does not claim a benchmark result. Stage 6.3 must establish causal
acquisition, checkable calibration provenance, stable identity, auditable
resources, deterministic scheduling, and truth isolation. Accuracy and
bandwidth comparisons against completed sweeps remain Stage 6.5 work.

## Approaches considered

Three approaches were considered:

1. A calibrated local-slope proportional loop is selected. Each completed pair
   produces a dimensionless discriminator, subtracts its public-model zero,
   divides by an analytic slope in inverse hertz, and applies a bounded
   correction in hertz. It isolates the two-point measurement hypothesis with
   the fewest dynamic assumptions.
2. A tabulated nonlinear inverse discriminator could use more of the fitted
   line shape and enlarge the local inversion interval. It would introduce
   interpolation, branch selection, and out-of-table behavior before the local
   linear baseline has been tested.
3. An extended Kalman or other state-space tracker could combine the nonlinear
   observation with a motion prior. It would confound the two-point measurement
   with a predictor and require process-noise choices outside Stage 6.3.

A raw fluorescence difference multiplied by a tuned gain is rejected. Its
units depend on normalization, baseline, amplitude, and integration, so its
numerical gain has no stable frequency meaning across identities or runs.

## Architecture and ownership

```text
evaluator-controlled calibration acquisition
        | full observations retained only by evaluator
        v
safe observations -> public full-spectrum fit -> immutable bound source
        |                                      (fit + trace + scale + clock
        |                                       + bounds + safe resources)
        v
required calibration-budget treatment + conservative identity cells
        |
        v
fixed-ID round robin -> adjacent alternating flank pair
        |
        v
normalized discriminator - public-model zero
        |
        v
analytic target-only slope -> capture -> gain -> step/domain limit
        |
        v
immutable pair result + eight identity-keyed policy cells
```

The evaluator owns the virtual instrument, dynamics, full
`InstrumentObservation` values, signal-conditioned expected photons, and
post-release truth. The tracker owns only immutable public configuration,
calibration, run metadata, budget, estimator-safe observations, pair buffers,
and identity cells. No tracker method receives an instrument, dynamics object,
`SpectralSnapshot`, expected photon count, noiseless callback, or evaluator
reference.

All new estimator-facing records are frozen, slotted, defensively snapshotted,
and exported from `odmr_bench.estimators`. Evaluator-only records and joins live
under `odmr_bench.evaluation.two_point`. Valid NumPy scalar integers/reals are
canonicalized to Python `int`/`float`; booleans, complex values, arrays in
scalar positions, and non-finite values are rejected.

## Exact public contracts

The following schemas fix names and ownership. Implementations may add private
helpers but may not replace fields with unvalidated metadata mappings.

```python
CalibrationBudgetTreatment = Literal[
    "included_same_run",
    "conditional_free_precalibration",
]
CalibrationSourceProvenance = Literal[
    "verified_factory_acquisition",
    "caller_asserted",
]
CalibrationIdentityMode = Literal[
    "require_expected_ids",
    "adopt_fit_ids",
]
ClockMappingKind = Literal["shared_clock", "unit_scale_offset"]
PairSide = Literal["minus", "plus"]
TwoPointLockState = Literal[
    "calibrated",
    "tracking",
    "step_limited",
    "lost",
]
TwoPointFailureCode = Literal[
    "invalid_pair_normalization",
    "numerical_failure",
    "common_mode_limit_exceeded",
    "capture_exceeded",
    "calibration_domain_exceeded",
]
TwoPointStopReason = Literal["budget_exhausted"]
TwoPointAbortReason = Literal[
    "resource_join_unavailable",
    "tracker_observation_validation_error",
    "tracker_update_construction_error",
    "tracker_update_unexpected_error",
]
VerifiedCalibrationFailureCode = Literal[
    "instrument_query_failed",
    "resource_join_unavailable",
    "acquisition_contract_mismatch",
    "fit_failed",
    "fit_exception",
    "source_binding_failed",
]
VerifiedCalibrationPreflightCode = Literal[
    "invalid_runner_phase",
    "invalid_argument_type",
    "invalid_argument_value",
    "invalid_frequency_grid",
    "invalid_fit_or_identity_configuration",
    "invalid_clock_mapping",
    "unclean_instrument_boundary",
]
TwoPointRunnerPhase = Literal[
    "ready",
    "calibration_succeeded",
    "calibration_failed",
    "tracking",
    "budget_stopped",
    "externally_stopped",
    "aborted",
]
TwoPointRunnerStartFailureCode = Literal[
    "invalid_runner_phase",
    "invalid_argument_type",
    "unverified_calibration",
    "calibration_mismatch",
    "run_provenance_mismatch",
    "metadata_mismatch",
    "resource_boundary_mismatch",
    "tracker_reset_failed",
]
TwoPointCalibrationConstructionCode = Literal[
    "invalid_argument_type",
    "invalid_argument_value",
    "invalid_provenance_or_quantity",
    "invalid_source_trace",
    "source_resource_mismatch",
    "fit_input_mismatch",
    "source_fit_failed",
    "source_identity_mismatch",
    "invalid_source_epoch",
    "invalid_availability_or_clock",
    "invalid_calibration_geometry",
    "invalid_budget_treatment",
]
TwoPointObservationValidationCode = Literal[
    "invalid_observation_type",
    "no_pending_query",
    "sequence_mismatch",
    "frequency_mismatch",
    "integration_time_mismatch",
    "endpoint_mismatch",
    "nominal_exposure_mismatch",
    "invalid_observation_value",
]
TwoPointUpdateConstructionCode = Literal[
    "partial_pair_construction_failed",
    "pair_result_construction_failed",
    "identity_estimate_construction_failed",
    "resource_construction_failed",
    "aggregate_estimate_construction_failed",
]
ResourceJoinMismatchField = Literal[
    "observations",
    "integration_time_s",
    "nominal_exposure_photons",
    "expected_photons",
    "realized_photons",
    "observations_without_realized_counts",
    "virtual_elapsed_time_s",
]

PublicAcquisitionResources(
    observations: int,
    integration_time_s: float,
    nominal_exposure_photons: float,
    realized_photons: int,
    observations_without_realized_counts: int,
    virtual_elapsed_time_s: float,
)

TwoPointBudgetCeiling(
    max_observations: int | None,
    max_integration_time_s: float | None,
    max_nominal_exposure_photons: float | None,
    max_virtual_elapsed_time_s: float | None,
)

TwoPointIdentityBinding(
    mode: CalibrationIdentityMode,
    expected_resonance_ids: tuple[str, ...] | None,
)

NormalizedFluorescenceProvenance(
    quantity: Literal["normalized_fluorescence"],
    normalization_rule: str,
    nominal_photon_rate_hz: float,
    sampling_rules: tuple[str, ...],
)

TwoPointClockMapping(
    kind: ClockMappingKind,
    source_clock_id: str,
    tracker_clock_id: str,
    scale: float,               # required to equal exactly 1.0 in Stage 6.3
    offset_s: float,            # tracker_time = source_time + offset_s
)

TwoPointCalibrationSource(
    source_id: str,
    provenance: CalibrationSourceProvenance,
    source_fit: SpectrumFitResult,
    fit_configuration: FitConfiguration,
    identity_binding: TwoPointIdentityBinding,
    resolved_resonance_ids: tuple[str, ...],
    source_observations: tuple[EstimatorObservation, ...],
    fluorescence_provenance: NormalizedFluorescenceProvenance,
    source_frequency_overhead_s: float,
    source_frequency_min_hz: float,
    source_frequency_max_hz: float,
    source_first_sequence_index: int,
    source_last_sequence_index: int,
    source_start_timestamp_s: float,
    source_first_timestamp_s: float,
    source_last_timestamp_s: float,
    physical_fit_epoch_s: float,
    availability_sequence_index: int,
    availability_timestamp_s: float,
    safe_resources: PublicAcquisitionResources,
    clock_mapping: TwoPointClockMapping,
)

TwoPointTrackerConfiguration(
    identity_binding: TwoPointIdentityBinding = TwoPointIdentityBinding(
        mode="require_expected_ids",
        expected_resonance_ids=("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"),
    ),
    offset_fwhm_fraction: float = 0.35,
    capture_fwhm_fraction: float = 0.20,
    proportional_gain: float = 1.0,
    max_step_fwhm_fraction: float = 0.10,
    integration_time_s: float = 0.005,
    common_mode_limit_target_depths: float | None = None,
)

TwoPointIdentityCalibration(
    resonance_id: str,
    source_fit_index: int,
    calibration_center_hz: float,
    calibration_fwhm_hz: float,
    calibration_amplitude: float,
    calibration_eta: float,
    offset_hz: float,
    capture_radius_hz: float,
    max_step_hz: float,
    target_pair_depth: float,
    calibration_cell_lower_hz: float,
    calibration_cell_upper_hz: float,
    allowed_center_min_hz: float,
    allowed_center_max_hz: float,
)

TwoPointCalibration(
    source: TwoPointCalibrationSource,
    configuration: TwoPointTrackerConfiguration,
    budget_treatment: CalibrationBudgetTreatment,  # required; no default
    identities: tuple[TwoPointIdentityCalibration, ...],
)

TwoPointRunMetadata(
    tracker_clock_id: str,
    current_sequence_index: int | None,
    current_timestamp_s: float,
    nominal_photon_rate_hz: float,
    frequency_overhead_s: float,
    fluorescence_quantity: Literal["normalized_fluorescence"],
)

TwoPointQuery(
    query_index: int,
    pair_index: int,
    identity_pair_index: int,
    resonance_id: str,
    side: PairSide,
    interrogation_center_hz: float,
    frequency_hz: float,
    integration_time_s: float,
    expected_sequence_index: int,
    expected_end_timestamp_s: float,
    expected_nominal_exposure_photons: float,
)

TwoPointPartialPair(
    pair_index: int,
    identity_pair_index: int,
    resonance_id: str,
    interrogation_center_hz: float,
    first_side: PairSide,
    first_query: TwoPointQuery,
    first_observation: EstimatorObservation,
)

TwoPointPairResult(
    pair_index: int,
    identity_pair_index: int,
    resonance_id: str,
    interrogation_center_hz: float,
    first_side: PairSide,
    minus_query: TwoPointQuery,
    plus_query: TwoPointQuery,
    minus_observation: EstimatorObservation,
    plus_observation: EstimatorObservation,
    pair_reference_timestamp_s: float,
    release_sequence_index: int,
    release_timestamp_s: float,
    discriminator: float | None,
    zero_discriminator: float,
    discriminator_slope_per_hz: float,
    raw_innovation_hz: float | None,
    requested_step_hz: float | None,
    candidate_center_hz: float | None,
    applied_step_hz: float,
    common_mode_target_depths: float | None,
    lock_state: TwoPointLockState,
    failure_code: TwoPointFailureCode | None,
)

TwoPointIdentityEstimate(
    resonance_id: str,
    center_hz: float,
    calibration_fwhm_hz: float,
    calibration_cell_lower_hz: float,
    calibration_cell_upper_hz: float,
    allowed_center_min_hz: float,
    allowed_center_max_hz: float,
    active_source_kind: Literal["calibration", "pair"],
    active_source_pair_index: int | None,
    active_reference_timestamp_s: float,
    active_release_sequence_index: int | None,
    active_release_timestamp_s: float,
    estimate_age_sequence_indices: int | None,
    estimate_age_s: float,
    release_age_s: float,
    completed_pairs: int,
    lock_state: TwoPointLockState,
    failure_code: TwoPointFailureCode | None,
    latest_pair: TwoPointPairResult | None,
)

TwoPointEstimate(
    identities: tuple[TwoPointIdentityEstimate, ...],
    calibration_source_id: str,
    calibration_source_provenance: CalibrationSourceProvenance,
    calibration_budget_treatment: CalibrationBudgetTreatment,
    current_sequence_index: int | None,
    current_timestamp_s: float,
    accepted_observations: int,
    completed_pairs: int,
    incomplete_pair: TwoPointPartialPair | None,
    pending_query: TwoPointQuery | None,
    pair_history: tuple[TwoPointPairResult, ...],
    tracking_resources: PublicAcquisitionResources,
    calibration_resources: PublicAcquisitionResources,
    charged_resources: PublicAcquisitionResources,
    budget_ceiling: TwoPointBudgetCeiling,
    stopped_reason: TwoPointStopReason | None,
    seed: int,
)

TwoPointUpdate(
    query: TwoPointQuery,
    observation: EstimatorObservation,
    completed_pair: TwoPointPairResult | None,
    estimate: TwoPointEstimate,
)

# Evaluator-only; none of these records is an estimator input. The token has
# no public constructor, value representation, copy/deepcopy, serialization,
# or user-defined equality. The evaluator module can validate its private
# issuer/outcome/source binding; public code can only carry the same object.
VerifiedInstrumentRunToken()

TwoPointEvaluatorInstrumentConfiguration(
    nominal_photon_rate_hz: float,
    frequency_overhead_s: float,
)

VerifiedCalibrationQueryRequest(
    point_index: int,
    frequency_hz: float,
    integration_time_s: float,
    expected_sequence_index: int,
    expected_measurement_midpoint_s: float,
    expected_end_timestamp_s: float,
    expected_nominal_exposure_photons: float,
)

VerifiedTwoPointCalibrationSuccess(
    status: Literal["success"],
    run_token: VerifiedInstrumentRunToken,
    source: TwoPointCalibrationSource,
    full_observations: tuple[InstrumentObservation, ...],
    safe_observations: tuple[EstimatorObservation, ...],
    measurement_midpoints_s: tuple[float, ...],
    instrument_resources_before: ResourceSnapshot,
    instrument_resources_after: ResourceSnapshot,
    safe_resources: PublicAcquisitionResources,
    full_resources: ResourceSnapshot,
)

VerifiedTwoPointCalibrationFailure(
    status: Literal["failure"],
    run_token: VerifiedInstrumentRunToken,
    failure_code: VerifiedCalibrationFailureCode,
    failed_request: VerifiedCalibrationQueryRequest | None,
    exception_type: str | None,
    exception_message: str | None,
    fit_result: SpectrumFitResult | None,
    resource_mismatch_fields: tuple[ResourceJoinMismatchField, ...],
    full_observations: tuple[InstrumentObservation, ...],
    safe_observations: tuple[EstimatorObservation, ...],
    measurement_midpoints_s: tuple[float | None, ...],
    instrument_resources_before: ResourceSnapshot,
    instrument_resources_after: ResourceSnapshot,
    safe_resources: PublicAcquisitionResources | None,
    full_resources: ResourceSnapshot | None,
)

VerifiedTwoPointCalibrationOutcome = (
    VerifiedTwoPointCalibrationSuccess | VerifiedTwoPointCalibrationFailure
)

TwoPointTrackingAcquisition(
    resource_join_status: Literal["authenticated"],
    query: TwoPointQuery,
    expected_measurement_midpoint_s: float,
    measurement_midpoint_s: float | None,
    full_observation: InstrumentObservation,
    safe_observation: EstimatorObservation,
    instrument_resources_before: ResourceSnapshot,
    instrument_resources_after: ResourceSnapshot,
    instrument_resource_delta: ResourceSnapshot,
)

TwoPointResourceJoinUnavailableAcquisition(
    resource_join_status: Literal["unavailable"],
    query: TwoPointQuery,
    expected_measurement_midpoint_s: float,
    measurement_midpoint_s: float | None,
    full_observation: InstrumentObservation,
    safe_observation: EstimatorObservation,
    resource_mismatch_fields: tuple[ResourceJoinMismatchField, ...],
    instrument_resources_before: ResourceSnapshot,
    instrument_resources_after: ResourceSnapshot,
)

TwoPointEvaluatorPairTiming(
    pair_index: int,
    resonance_id: str,
    first_measurement_midpoint_s: float,
    second_measurement_midpoint_s: float,
    truth_reference_timestamp_s: float,
    public_reference_timestamp_s: float,
    release_sequence_index: int,
    release_timestamp_s: float,
)

TwoPointInstrumentQueryFailure(
    query: TwoPointQuery,
    exception_type: str,
    exception_message: str,
    instrument_resources_before: ResourceSnapshot,
    instrument_resources_after: ResourceSnapshot,
)

TwoPointEvaluatorResources(
    calibration_observations: tuple[InstrumentObservation, ...],
    accepted_tracking_observations: tuple[InstrumentObservation, ...],
    unaccepted_tracking_observations: tuple[InstrumentObservation, ...],
    calibration_resources: ResourceSnapshot,
    accepted_tracking_resources: ResourceSnapshot,
    unaccepted_tracking_resources: ResourceSnapshot,
    tracking_resources: ResourceSnapshot,
    accepted_charged_resources: ResourceSnapshot,
    charged_resources: ResourceSnapshot,
    calibration_budget_treatment: CalibrationBudgetTreatment,
    incomplete_pair_observations: Literal[0, 1],
    unaccepted_observations: Literal[0, 1],
)

TwoPointAbortedRun(
    reason: TwoPointAbortReason,
    exception_type: str | None,
    exception_message: str | None,
    unaccepted_acquisition: (
        TwoPointTrackingAcquisition
        | TwoPointResourceJoinUnavailableAcquisition
    ),
    unaccepted_observation_count: Literal[1],
    tracker_estimate_before: TwoPointEstimate,
    tracker_estimate_after: TwoPointEstimate,
)

TwoPointEvaluatorRunnerState(
    phase: TwoPointRunnerPhase,
    run_token: VerifiedInstrumentRunToken,
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration,
    calibration_outcome: VerifiedTwoPointCalibrationOutcome | None,
    verified_calibration: VerifiedTwoPointCalibrationSuccess | None,
    calibration: TwoPointCalibration | None,
    tracker_estimate: TwoPointEstimate | None,
    normal_tracking_trace: tuple[TwoPointTrackingAcquisition, ...],
    pair_timings: tuple[TwoPointEvaluatorPairTiming, ...],
    instrument_resources_at_bind: ResourceSnapshot,
    tracking_resources_before: ResourceSnapshot | None,
    instrument_resources_current: ResourceSnapshot,
    instrument_current_sequence_index: int | None,
    current_virtual_time_s: float,
    last_instrument_failure: TwoPointInstrumentQueryFailure | None,
    terminal_abort: TwoPointAbortedRun | None,
)

TwoPointRunnerAccepted(
    kind: Literal["accepted"],
    acquisition: TwoPointTrackingAcquisition,
    update: TwoPointUpdate,
    state: TwoPointEvaluatorRunnerState,
)

TwoPointRunnerInstrumentFailure(
    kind: Literal["instrument_failure"],
    failure: TwoPointInstrumentQueryFailure,
    state: TwoPointEvaluatorRunnerState,
)

TwoPointRunnerBudgetStopped(
    kind: Literal["budget_stopped"],
    resources: TwoPointEvaluatorResources,
    state: TwoPointEvaluatorRunnerState,
)

TwoPointRunnerExternallyStopped(
    kind: Literal["externally_stopped"],
    resources: TwoPointEvaluatorResources,
    state: TwoPointEvaluatorRunnerState,
)

TwoPointRunnerAborted(
    kind: Literal["aborted"],
    abort: TwoPointAbortedRun,
    resources: TwoPointEvaluatorResources | None,
    state: TwoPointEvaluatorRunnerState,
)

TwoPointRunnerStepOutcome = (
    TwoPointRunnerAccepted
    | TwoPointRunnerInstrumentFailure
    | TwoPointRunnerBudgetStopped
    | TwoPointRunnerAborted
)
TwoPointRunnerRunOutcome = (
    TwoPointRunnerInstrumentFailure
    | TwoPointRunnerBudgetStopped
    | TwoPointRunnerAborted
)
```

`TwoPointPairResult` stores queries and observations by side, while
`first_side` preserves arrival order. A successful pair has
`failure_code=None`; `tracking` applies the requested step exactly and
`step_limited` applies the clipped step. `lost` requires one failure code and
applies exactly zero. `calibrated` occurs only before an identity's first pair.

`active_source_pair_index` is `None` for a calibration seed and otherwise
names the global pair that last successfully produced or refreshed the active
center estimate, including a successful pair whose applied step is exactly
zero. A failed pair remains visible as `latest_pair` and changes the policy
state to `lost`, but does not replace the older active center source.
`calibration_fwhm_hz` is provenance, not a live linewidth estimate.

`TwoPointCalibration.configuration` is a defensive immutable snapshot. Reset
requires exact equality with the tracker's constructor configuration, including
identity binding, integration, offset, capture, gain, step, and common-mode
policy. A source or calibration cannot be silently reused under a different
tracker configuration.

The public factories, tracker, and evaluator-runner interfaces are:

```python
class TwoPointCalibrationPreflightError(ValueError):
    code: VerifiedCalibrationPreflightCode

class TwoPointCalibrationConstructionError(ValueError):
    code: TwoPointCalibrationConstructionCode
    message: str
    def __init__(
        self, code: TwoPointCalibrationConstructionCode, message: str
    ) -> None: ...

class TwoPointRunnerStartError(ValueError):
    code: TwoPointRunnerStartFailureCode

class TwoPointRunnerStateError(RuntimeError): ...

class TwoPointObservationValidationError(ValueError):
    code: TwoPointObservationValidationCode
    message: str
    def __init__(
        self, code: TwoPointObservationValidationCode, message: str
    ) -> None: ...

class TwoPointUpdateConstructionError(RuntimeError):
    code: TwoPointUpdateConstructionCode
    message: str
    def __init__(
        self, code: TwoPointUpdateConstructionCode, message: str
    ) -> None: ...

def bind_caller_asserted_two_point_calibration_source(
    source_fit: SpectrumFitResult,
    fit_configuration: FitConfiguration,
    source_observations: Sequence[EstimatorObservation],
    identity_binding: TwoPointIdentityBinding,
    fluorescence_provenance: NormalizedFluorescenceProvenance,
    *,
    source_id: str,
    source_frequency_overhead_s: float,
    source_start_timestamp_s: float,
    physical_fit_epoch_s: float,
    availability_sequence_index: int,
    availability_timestamp_s: float,
    clock_mapping: TwoPointClockMapping,
) -> TwoPointCalibrationSource: ...

def calibrate_two_point(
    source: TwoPointCalibrationSource,
    configuration: TwoPointTrackerConfiguration,
    *,
    budget_treatment: CalibrationBudgetTreatment,  # required; no default
) -> TwoPointCalibration: ...

class CalibratedTwoPointTracker:
    def __init__(self, configuration: TwoPointTrackerConfiguration): ...

    @property
    def configuration(self) -> TwoPointTrackerConfiguration: ...
    @property
    def calibration(self) -> TwoPointCalibration | None: ...
    @property
    def pending_query(self) -> TwoPointQuery | None: ...
    @property
    def pair_history(self) -> tuple[TwoPointPairResult, ...]: ...

    def reset(
        self,
        public_metadata: TwoPointRunMetadata,
        calibration: TwoPointCalibration,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> None: ...

    def choose_next_query(self) -> TwoPointQuery | None: ...
    def update(self, observation: EstimatorObservation) -> TwoPointUpdate: ...
    def estimate(self) -> TwoPointEstimate: ...

# These two read-only instrument properties expose acquisition configuration,
# not hidden spectral state, and are never passed to an estimator.
class ODMRInstrument:
    @property
    def nominal_photon_rate_hz(self) -> float: ...
    @property
    def frequency_overhead_s(self) -> float: ...

class TwoPointEvaluatorRunner:
    @classmethod
    def bind(
        cls,
        instrument: ODMRInstrument,
    ) -> TwoPointEvaluatorRunner: ...

    @property
    def state(self) -> TwoPointEvaluatorRunnerState: ...

    def acquire_verified_calibration(
        self,
        frequency_hz: Sequence[float],
        integration_time_s: float,
        fit_configuration: FitConfiguration,
        identity_binding: TwoPointIdentityBinding,
        *,
        source_id: str,
        source_clock_id: str,
        tracker_clock_id: str,
        source_to_tracker_offset_s: float,
        physical_fit_epoch_rule: Literal[
            "instrument_midpoint_ordered_mean"
        ],
    ) -> VerifiedTwoPointCalibrationOutcome: ...

    def start_tracking(
        self,
        tracker: CalibratedTwoPointTracker,
        calibration: TwoPointCalibration,
        verified_calibration: VerifiedTwoPointCalibrationSuccess,
        public_metadata: TwoPointRunMetadata,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> TwoPointEvaluatorRunnerState: ...

    def step(self) -> TwoPointRunnerStepOutcome: ...
    def run_until_event(self) -> TwoPointRunnerRunOutcome: ...
    def stop_external(self) -> TwoPointRunnerExternallyStopped: ...

def build_two_point_evaluator_resources(
    runner: TwoPointEvaluatorRunner,
) -> TwoPointEvaluatorResources | None: ...
```

The safe records, construction/observation/update code aliases and their three
declared exception classes, asserted-source factory, calibration factory, and
tracker are exported from `odmr_bench.estimators`. The preflight/start code
aliases and errors, instrument-owning runner, opaque run token, typed verified-
calibration outcomes, both acquisition variants, timing traces, evaluator
join/resources (including `ResourceJoinMismatchField`), step/run outcomes, and
aborted-run record are exported from
`odmr_bench.evaluation.two_point`; this is the only public module in this stage
that accepts full observations or an instrument.

Calling `choose_next_query`, `update`, or `estimate` before a successful reset
raises. Returned tuples and nested model values are immutable snapshots.

## Construction, provenance, and validation precedence

### One bound calibration source

The tracker accepts exactly one `TwoPointCalibrationSource`; it never accepts a
fit, timing declaration, resource total, quantity label, or ID list as a
separate calibration argument. The source binds all of them so consistency is
validated once and cannot change between calibration and reset.

`verified_factory_acquisition` has a narrow, checkable meaning. A
`TwoPointEvaluatorRunner` mints one nonconstructible
`VerifiedInstrumentRunToken` when it binds one concrete instrument object and
retains both for its lifetime. At bind it derives the immutable
`TwoPointEvaluatorInstrumentConfiguration` from the instrument's two read-only
configuration properties; callers cannot independently declare these values.
The evaluator module privately binds the token to its issuing runner,
instrument identity, derived configuration, and, once successful, the exact
success outcome and source object. That one-time binding is what another
tracking runner checks for conditional use; possession of a token reference
does not let a caller construct a different verified outcome. This is an
in-process provenance capability, not a cryptographic or serializable claim.
Its `acquire_verified_calibration` method issues the declared monotonically
increasing frequencies, retains every returned full observation and its safe
view, and computes the expected instrument-evaluation midpoint before each
call using that bound configuration. Once the returned observation satisfies
the complete query contract, that value is the exact measurement midpoint for
the current `ODMRInstrument` association. It constructs one `CompleteSweep`
from only the safe values, sets that sweep's optional last-index, last-time,
integration-total, and nominal-total fields from the same atomic trace, and
invokes `fit_spectrum` with the stored `FitConfiguration` and no supplied
initial guess. On success it derives source bounds, first/last indices and
times, physical epoch, availability endpoint, normalization provenance, and
safe resources from that same acquisition. Thus it verifies linkage within
this emulator runner; it is not a cryptographic statement about external
hardware or metrology.

The verified path is deliberately conservative: the bound source instrument
must be at virtual time zero with an all-zero resource snapshot. These are
preflight checks. The first returned observation must also have sequence zero;
because the current instrument exposes no next-index property, a mismatch is a
post-query `acquisition_contract_mismatch` outcome that preserves that committed
observation. A successful verified source consequently starts at sequence and
time zero, and `safe_resources` is both the source-segment replay and the safe
projection of the instrument's cumulative source endpoint. Nonzero-start
external traces use the caller-asserted factory and can only be conditional
pre-calibration results.

Direct construction cannot mint `verified_factory_acquisition`. The asserted
factory produces `caller_asserted` and means only that the caller supplied a
self-consistent immutable bundle and asserted that its fit came from its safe
observations on the named normalized scale. It does not verify that causal
history. The status is retained in every estimate and cannot be upgraded by
copying. Caller-asserted sources are allowed only with
`conditional_free_precalibration` and are ineligible for an end-to-end
benchmark or evaluator expected-photon join.

For a verified source the normalization rule is exactly
`"odmr_instrument_normalized_fluorescence_v1"`; the stored nominal rate is the
runner's instrument-derived `TwoPointEvaluatorInstrumentConfiguration` rate,
every returned observation's nominal exposure must exactly equal that rate
times its integration, the endpoint recurrence verifies the bound overhead,
and `sampling_rules` is the arrival-ordered tuple from the full observations. The
source outcome therefore checks the declarations against the returned trace
without reading private attributes. Since the instrument settings are
immutable, the retained run token and instrument identity extend that evidence
to later included tracking on the same runner. The fit consumes normalized
fluorescence, never counts or expected photons. For a caller-asserted source the
non-empty rule and sampling labels are declarations, not verification. Both
statuses require the literal quantity `normalized_fluorescence`; unresolved
analog units or an undocumented affine rescaling cannot calibrate this tracker.

The source observations form one complete sweep: exact type only, nonempty,
contiguous sequence indices, strictly increasing timestamps and frequencies,
and the stored first/last fields equal the first/last safe observation. For
each source observation, the timestamp must equal

```text
(previous_endpoint + source_frequency_overhead_s) + integration_time_s
```

using the displayed association, beginning from `source_start_timestamp_s`.
Bounds are exactly the first and last frequencies. Availability index equals
the last source index and availability timestamp equals the last source
endpoint in Stage 6.3. For a verified acquisition, the distinct physical fit
epoch is the overflow-safe ordered mean of the first and last evaluator-owned
instrument midpoints, each computed before its query as
`(previous_instrument_endpoint + overhead) + integration / 2.0`. A
caller-asserted source instead uses the only timing convention available in its
safe trace. Define

```text
first_public_midpoint =
    first_observation.timestamp_s
    - first_observation.integration_time_s / 2.0
last_public_midpoint =
    last_observation.timestamp_s
    - last_observation.integration_time_s / 2.0
physical_fit_epoch_s =
    first_public_midpoint
    + (last_public_midpoint - first_public_midpoint) / 2.0
```

The asserted epoch must equal that result bit-for-bit and therefore lies in the
closed public-midpoint interval. The displayed association is normative;
equivalent regrouping and either neighboring ULP fail with
`invalid_source_epoch`. Endpoint reconstruction is not used for a verified
physical epoch because it can differ from the actual instrument association by
one binary64 ULP.

The fit-complete availability event is causally ordered after the last
observation even though estimator CPU does not advance virtual acquisition
time, so the two events may share one timestamp and acquisition index. CPU time
is evaluator-reported separately and never backdated into the fit epoch.

Verified acquisition has a lossless success/failure boundary. Exact argument
types, scalar domains, frequency-grid monotonicity, identity-binding shape,
clock-map shape, runner phase, and the clean time/resource boundary are checked
before the first query; `TwoPointCalibrationPreflightError(code=...)` may raise
there, with no acquisition or runner-state mutation. Once the acquisition loop
starts, no ordinary `Exception` is allowed to discard its trace:

- every successfully returned observation is appended to both the full and
  safe tuples, an aligned exact-midpoint-or-`None` slot is appended as defined
  below, and its authoritative instrument before/after snapshots are retained;
- an ordinary instrument `Exception` returns `instrument_query_failed`, with the exact
  failed request and exception, all earlier committed observations, and no
  invented observation for the atomic query that raised;
- immediately after a return, the evaluator applies the full raw record and
  bound overhead as one prospective ledger transition from `before`. If any
  resulting field differs from the authoritative instrument `after`, it returns
  `resource_join_unavailable`, lists every unequal
  `ResourceJoinMismatchField`, retains the raw full/safe record and snapshots,
  and sets both aggregate replay fields to `None` rather than subtracting,
  fabricating, or discarding the acquisition. This resource classification
  precedes every other returned-observation defect;
- only after that replay joins exactly, a returned observation that
  contradicts the bound frequency, sequence, endpoint, integration, nominal
  rate, or overhead returns `acquisition_contract_mismatch`; and
- an unsuccessful structured `SpectrumFitResult` returns `fit_failed` and
  retains that result plus the complete acquisition; a fitting exception
  returns `fit_exception` with the complete acquisition; and
- failure to bind the successful fit and complete trace into the source returns
  `source_binding_failed` with the complete acquisition and successful fit.

Both outcome variants carry the same opaque runner token, every committed full
observation and exact safe view, aligned midpoint slots, and the captured
authoritative instrument before/after snapshots. A success, and every failure
except `resource_join_unavailable`, also carries mandatory safe/full segment
resource replays from zero that join the final snapshot exactly. The
resource-unavailable failure carries `None` for both aggregates; its final
authoritative cumulative `instrument_resources_after` still preserves
expected-photon and every other ledger total evaluator-side.

A success has one finite exact midpoint per observation. A failure has one slot
per returned observation: a slot is the exact midpoint after timing checks pass,
while a timing-mismatched final observation stores `None`. Earlier slots remain
exact. On a mid-acquisition instrument exception the `after` snapshot is the
current cumulative ledger after the earlier successes and equals the snapshot
immediately before the failed atomic query. `resource_mismatch_fields` is
nonempty exactly for `resource_join_unavailable` and otherwise empty. Its
members appear once each in `ResourceJoinMismatchField` declaration order.
`fit_result` is non-`None` exactly for structured fit failure or post-fit source
binding failure; exception fields are non-`None` exactly for exception-based
failures. `failed_request` is non-`None` exactly for
`instrument_query_failed`, `resource_join_unavailable`, or
`acquisition_contract_mismatch`; fit and binding failures occur after the
request loop and store `None`. An acquisition failure is terminal for that
runner; a new calibration attempt requires a newly bound runner. Process-
control `BaseException` subclasses are outside these typed outcome guarantees.

`TwoPointIdentityBinding(mode="require_expected_ids", expected_resonance_ids=x)`
requires exactly eight unique nonempty IDs and exact ordered equality among
`x`, `FitConfiguration.resonance_ids`, and successful fit IDs.
`mode="adopt_fit_ids"` requires `expected_resonance_ids is None`; adoption is
therefore explicit rather than an omitted check. It adopts the successful fit
IDs in their existing order and still requires exactly eight unique IDs.

### Clock and availability rules

Stage 6.3 maps clock values only by
`tracker_time = source_time + offset_s`. `scale` must equal exactly `1.0`; no
drift-rate clock conversion is implied. `shared_clock` requires identical
nonempty clock IDs and `offset_s == 0.0`. `unit_scale_offset` requires distinct
nonempty IDs and a finite offset for which mapped first, last, physical-epoch,
and availability times are finite. Sequence indices are never mapped across
distinct clocks.

At reset, `tracker_clock_id` must equal the mapping target and mapped
availability must be nonnegative and not after `current_timestamp_s`.

Raw source-observation timestamps, source start/availability endpoints,
tracking observations, run endpoints, pair releases, and the public
endpoint-reconstructed pair reference are finite and nonnegative. A mapped
source coordinate and the mapped physical fit/reference epoch are finite signed
floats; the fixed fixture intentionally maps its physical fit epoch below zero.
Accordingly `TwoPointIdentityEstimate.active_reference_timestamp_s` accepts a
finite signed value while a cell is calibration-seeded. Its
`active_release_timestamp_s`, every pair-derived active reference/release, and
all sequence/time age fields remain finite and nonnegative. Every age
subtraction is validated nonnegative rather than silently clamped.

- `included_same_run` requires verified provenance, `shared_clock`, and
  evaluator-runner context carrying the exact successful source outcome. The
  runner requires object identity of its retained opaque token, the original
  bound instrument, and the source object in the calibration. It also requires
  `current_sequence_index == availability_sequence_index`,
  `current_timestamp_s == availability_timestamp_s`, an exact continuous
  boundary from calibration `instrument_resources_after` to tracking
  `instrument_resources_before`, and both configuration equalities:

  ```text
  public_metadata.nominal_photon_rate_hz
      == runner.instrument_configuration.nominal_photon_rate_hz
      == source.fluorescence_provenance.nominal_photon_rate_hz
  public_metadata.frequency_overhead_s
      == runner.instrument_configuration.frequency_overhead_s
      == source.source_frequency_overhead_s
  ```

  The first tracking index is `availability_sequence_index + 1`. A copied or
  deserialized source without its original runner/token context cannot make an
  included same-run claim; it may be used only as conditional pre-calibration.
- `conditional_free_precalibration` permits either provenance status. Source
  indices remain in their source domain, and source rate/overhead may differ
  from tracking. The run metadata rate/overhead must instead exactly equal the
  tracking runner's bound instrument configuration. `current_sequence_index=None`
  means a fresh tracker stream whose first index is zero; otherwise the first
  index is `current_sequence_index + 1`.

The physical epoch seeds estimate age. Availability seeds causal release age.
They are never substituted for each other.

### Required calibration budget treatment

`budget_treatment` is a required keyword with no default.

- `included_same_run` starts charged resources from the source's complete safe
  acquisition and applies every later tracking acquisition to that running
  value. Calibration is charged exactly once.
- `conditional_free_precalibration` reports source resources separately but
  starts charged resources at zero. Every dependent result remains labeled
  conditional on free pre-calibration; the source cost does not disappear.

Expected photons are evaluator-only and are not a source safe-resource field or
a budget input.

### Failure boundaries and precedence

Verified acquisition has two deliberately different failure mechanisms.
Before acquisition, `TwoPointCalibrationPreflightError` carries one
`VerifiedCalibrationPreflightCode`. `TwoPointEvaluatorRunner.bind` applies
`invalid_argument_type` before `unclean_instrument_boundary` while deriving the
instrument configuration and clean boundary. On an already bound runner,
`acquire_verified_calibration` applies this order to defects visible to that
call:

| Order | Preflight code |
| ---: | --- |
| 1 | `invalid_runner_phase` |
| 2 | `invalid_argument_type` |
| 3 | `invalid_argument_value` |
| 4 | `invalid_frequency_grid` |
| 5 | `invalid_fit_or_identity_configuration` |
| 6 | `invalid_clock_mapping` |

`invalid_argument_type` covers exact container/record/scalar types and
forbidden booleans. `invalid_argument_value` covers correctly typed general
values outside their scalar/text/literal domain, including nonpositive or
nonfinite integration, a blank `source_id`, a nonpositive/nonfinite frequency
element, and any epoch-rule value other than
`"instrument_midpoint_ordered_mean"`. `invalid_frequency_grid` then covers an
empty, duplicate, or non-increasing grid whose elements already pass their
individual domains. Fit/identity structure precedes clock-ID/offset/mapping
structure. At `bind`, where only instrument/configuration and boundary defects
exist, the order is type, value, then `unclean_instrument_boundary`.

Preflight constructs defensive argument snapshots before acquisition. It
performs no query, and a raised error leaves runner and instrument unchanged.
Once the first query is attempted, causal order fixes outcome precedence: the
first instrument exception or returned-observation contract mismatch stops
acquisition; otherwise fitting happens after the full trace and source binding
happens only after fit success. Those post-start paths return the typed failure
outcome defined above instead of raising an ordinary exception and losing
acquisitions.

The nonacquiring asserted-source and `calibrate_two_point` paths raise
`TwoPointCalibrationConstructionError(code, message)` with a nonempty message.
They use this exact first-applicable precedence:

| Order | Construction code | Scope |
| ---: | --- | --- |
| 1 | `invalid_argument_type` | exact record/container/scalar types and forbidden booleans |
| 2 | `invalid_argument_value` | correctly typed scalar/text/literal domains not assigned to a later semantic row |
| 3 | `invalid_provenance_or_quantity` | provenance minting, fluorescence quantity, normalization, and sampling labels |
| 4 | `invalid_source_trace` | nonempty sweep, sequence/frequency/timestamp order and recurrence, source bounds/endpoints |
| 5 | `source_resource_mismatch` | atomic safe-resource replay versus stored source total |
| 6 | `fit_input_mismatch` | fit configuration, trace, and successful-result input provenance |
| 7 | `source_fit_failed` | unsuccessful or malformed source fit, including missing baseline |
| 8 | `source_identity_mismatch` | expected/adopted ID mode, uniqueness, cardinality, and order |
| 9 | `invalid_source_epoch` | verified actual-midpoint rule or asserted exact public-midpoint rule |
| 10 | `invalid_availability_or_clock` | availability endpoint/index and clock mapping |
| 11 | `invalid_calibration_geometry` | model, slope, cell, and capture-envelope construction |
| 12 | `invalid_budget_treatment` | unknown treatment or asserted-source/same-run combination |

Omitting the required `budget_treatment` keyword is rejected by Python
argument binding before this semantic precedence applies. Tests combine each
adjacent pair of defects and require the earlier literal code.

`TwoPointObservationValidationError(code, message)` also requires a nonempty
message and uses its alias order: invalid exact observation type; no pending
query; sequence; frequency; integration; endpoint; nominal exposure; then
observation-value defect. `TwoPointUpdateConstructionError(code, message)` is
reserved for prospective immutable state construction after observation
validation. First-side construction uses `partial_pair_construction_failed`.
Second-side construction proceeds through pair result, identity estimate,
resource state, then aggregate estimate and uses the corresponding first
failing code. Scientific gate outcomes are records, not construction
exceptions.

### Intrinsic record invariants

Constructors validate only facts represented by their own fields. They do not
claim to authenticate acquisition history, a reset input, a prior state, or an
instrument that is absent from the record.

| Record group | Intrinsic constructor invariant |
| --- | --- |
| resources and budget | canonical nonnegative fields; missing observations do not exceed observations; elapsed is at least integration; at least one finite nonnegative ceiling |
| identity binding and clock | valid mode/optional-ID combination; finite unit-scale map; nonempty clock IDs and correct shared/distinct relation |
| source | successful fit shape, internally equal trace endpoints/bounds/IDs/resource totals, nonnegative raw endpoints/epoch/availability and finite signed map offset; caller-asserted epoch exactly follows its public-midpoint convention; direct construction cannot mint verified provenance |
| tracker configuration and metadata | finite scalar domains and fraction ordering; exact normalized quantity; metadata endpoint/index/rate/overhead domains and finite one/two-query products |
| calibration and identity calibration | stored source/config/ID equality; eight unique ordered identities; positive finite model/depth values; internally nonempty fixed geometry |
| query, partial pair, and pair result | local scalar domains and field-to-field ID/index/side/query/observation echoes; adjacent sides; release selection; gate-state/diagnostic combinations |
| identity and aggregate estimate | calibration-seeded active reference may be finite signed; pair references/releases and all ages/counters/resources are nonnegative; history/counter/identity/pending/partial equalities that use fields present in the estimate; valid source/stop/seed labels |
| update | local query/observation echo and completed-pair versus returned-estimate combination |
| verified acquisition outcomes | discriminator-specific request/fit/exception/mismatch fields; equal lengths and projections of full/safe/midpoint tuples; aggregate resources are `None` exactly for `resource_join_unavailable` and otherwise valid and present |
| evaluator acquisition/timing/state/outcome records | authenticated acquisition has an exact atomic delta; unavailable acquisition has nonempty mismatch fields, authoritative boundaries, and no delta; other local type, phase, cardinality, endpoint, and union combinations are valid |
| evaluator resources | valid full observation tuples and resource snapshots; accepted charged prefix and final charged resource are distinct; incomplete and unaccepted counts are independently zero or one |
| abort | one unaccepted acquisition and equal pending-bearing tracker snapshots; resource-unavailable reason requires the unavailable variant and `None` exception fields, while exception reasons require the authenticated variant, nonempty exception type, and exact possibly-empty message |

### Contextual joins and their owners

Facts requiring external context are checked by the operation that owns that
context, before it constructs the returned record:

| Owner | Contextual guarantees |
| --- | --- |
| `TwoPointEvaluatorRunner.bind` | concrete instrument identity, instrument-derived immutable runner configuration, clean initial resource/time boundary, and opaque run-token minting |
| `acquire_verified_calibration` | token/instrument continuity, actual instrument midpoints, query/observation recurrence, full-to-safe projections, exact replay when available or an explicit unavailable join with authoritative snapshots, fit-from-safe-trace linkage, and verified source provenance |
| asserted-source factory | caller-asserted label plus internal trace/config/resource consistency; it does not authenticate origin or fit derivation |
| `calibrate_two_point` | source/config identity, model derivative/monotonicity, fixed-cell geometry, and treatment compatibility |
| tracker `reset` | constructor configuration, calibration, metadata, clock, starting sequence, charged source resources, ceiling, and seed joins |
| tracker `choose_next_query` / `update` | prior pending/partial/schedule state, pair reservation, exact accepted echo, state transition, history, source refresh, and tracker resource joins |
| `build_two_point_evaluator_resources` | runner-private instrument/token context, verified calibration trace, normal tracking trace, accepted charged-prefix equality, optional authenticated unaccepted atom, full/safe resources, and instrument boundary joins; returns `None` for an unavailable resource join |
| evaluator runner tracking step | issued pending query, pre-update estimate, actual instrument midpoint when authenticated, original full observation, authoritative before/after ledger, exact replay or unavailable status, tracker exception class, normal-trace append or terminal abort, and no post-abort call |

For example, a directly constructed `TwoPointUpdate` can prove its observation
echoes its stored query; only `update` can prove that query was the pending
query. A `TwoPointEstimate` can prove its own counters equal its own history;
only reset/transition code can prove equality to reset-time inputs or genuine
budget unaffordability. `TwoPointEvaluatorResources` and `TwoPointAbortedRun`
likewise become contextually authoritative only when produced by their named
builder/runner.

## Public-model discriminator calibration
### Units and canonical evaluation

The repository fit defines

\[
M(f)=B(f)-\sum_j A_jP_j(f;c_j,w_j,\eta_j),
\]

where `P` is the FWHM-matched pseudo-Voigt profile (`eta=1` for a
Lorentzian fit). The conceptual target-frozen background is

\[
H_i(f)=B(f)-\sum_{j\ne i} A_jP_j(f;c_j,w_j,\eta_j),
\]

and the target-only local model is

\[
M_i(f;c)=H_i(f)-A_iP_i(f;c,w_i,\eta_i).
\]

Production must not materialize `H_i` by regrouping the sum. It evaluates the
baseline once and subtracts each resonance in the immutable source-fit tuple
order, exactly like `multi_resonance_spectrum`, replacing only the target's
center with `c`. This canonical order is also used by generated hidden fixtures
so an exact-model static test is not defeated by reassociation roundoff.

| Quantity | Unit |
| --- | --- |
| `f`, `c`, `q`, `w`, `delta`, `r`, raw/requested/applied step | Hz |
| integration, overhead, reference/release/age | s |
| nominal photon rate | photons/s |
| nominal/expected/realized exposure | photons |
| `P`, `A`, `B`, `M`, `mu`, `D`, observed fluorescence | normalized-fluorescence units (dimensionless) |
| `e`, `e0`, gain, FWHM fractions, `m` | dimensionless |
| target-center derivative and `kappa` | Hz^-1 |

At a pair's frozen interrogation center `q_i`, define
`delta_i = offset_fwhm_fraction * w_i`, `f_minus=q_i-delta_i`, and
`f_plus=q_i+delta_i`. Only the target center moves in the derivative; all other
fit values remain frozen.

For `u=(f-c)/w_i`,

\[
P(u)=\frac{\eta_i}{1+4u^2}
 +(1-\eta_i)e^{-4\ln(2)u^2},
\]

and

\[
\frac{\partial M_i}{\partial c}
=-\frac{8A_i u}{w_i}
\left[
\frac{\eta_i}{(1+4u^2)^2}
+(1-\eta_i)\ln(2)e^{-4\ln(2)u^2}
\right].
\]

At `c=q_i`, let `mu_minus=M_i(f_minus;q_i)`,
`mu_plus=M_i(f_plus;q_i)`, and let `g_minus`, `g_plus` be their analytic
target-center derivatives. Then

\[
e(y_-,y_+)=\frac{y_- - y_+}{y_-+y_+},
\qquad
e_0=\frac{\mu_- - \mu_+}{\mu_-+\mu_+},
\]

\[
\kappa_i=
\frac{2(\mu_+g_- - \mu_-g_+)}{(\mu_-+\mu_+)^2}.
\]

On the minus flank `u<0`, so `g_minus>0`; on the plus flank `u>0`,
so `g_plus<0`. With positive modeled flank fluorescence, `kappa_i>0`: a
target center displaced upward produces a positive innovation. A non-positive
or non-finite slope fails calibration; it is never sign-flipped or absolutized.

For `r_i=capture_fwhm_fraction*w_i`, `r_i<delta_i` keeps both target
derivatives on their original sides throughout the declared capture interval.
The closest-to-center modeled value on each flank is checked at the relevant
capture endpoint and must remain strictly positive. Analytically this makes
the displayed discriminator strictly increasing on `[-r_i,+r_i]`. The
implementation also verifies this derivation against a centered numerical
difference with `h=1e-5*w_i`, `rtol=1e-8`, and `atol=1e-15 Hz^-1`; production
still uses the analytic expression. A 1001-point inclusive test grid must have
strictly positive consecutive discriminator differences for all eight fixture
lines.

The target-only pair-depth scale is, with the reduced-coordinate arguments
shown explicitly,

\[
D_i=A_i\left[
P_i(-\delta_i/w_i)+P_i(+\delta_i/w_i)
\right] > 0.
\]

Equivalently it is the sum of the two full target profiles evaluated at
`q_i +/- delta_i` with center `q_i`. This dimensionless quantity is stored for
the common-mode diagnostic.

## Fixed conservative identity cells

Identity is fixed by calibration ID, never by the current frequency order. Let
the successful source-fit centers be strictly ordered
`c_0 < ... < c_7`, and let the inclusive source sweep bounds be `F_min` and
`F_max`. Fixed calibration Voronoi cells are

```text
cell_lower[0] = F_min
cell_upper[7] = F_max
internal_boundary[i] = c_i + (c_(i+1) - c_i) / 2
cell_upper[i] = internal_boundary[i]
cell_lower[i+1] = internal_boundary[i]
```

The ordered-difference midpoint avoids overflow. Cells are derived once from
calibration centers and never move when estimates move. Shared boundaries are
allowed because each ID's usable center interval is inset.

For identity `i`, define its full capture-plus-probe half-span

\[
a_i=\delta_i+r_i.
\]

The only allowed interrogation-center interval is inclusive:

```text
allowed_center_min_hz = calibration_cell_lower_hz + a_i
allowed_center_max_hz = calibration_cell_upper_hz - a_i
```

It must be nonempty and contain the calibration center. For every initial and
candidate `q`, the tracker requires the complete interval
`[q-a_i, q+a_i]` inside both the identity cell and source domain. This stronger
condition entails that the center, both actual probes `q +/- delta_i`, the
full possible center interval `q +/- r_i`, and both probes around every center
in that capture interval remain inside the fixed calibration cell/source
domain. It also handles unequal neighboring widths because each identity uses
its own `delta_i+r_i` inset.

Calibration fails if any initial geometry is empty or outside its domain. On a
pair update, candidate arithmetic is checked for finiteness first. A finite
candidate outside the inclusive allowed interval produces
`lost/calibration_domain_exceeded`, records that candidate, applies exactly
`0.0 Hz`, and retains the prior center. A non-finite candidate is instead
`lost/numerical_failure`. Domain failure can therefore never be used to relabel
or cross identities.

Geometry tests cover all eight cells, both outer source bounds, every internal
boundary, unequal FWHM in both neighbor directions, exact allowed endpoints,
one-ULP outward candidates via `math.nextafter`, an empty-cell construction,
and repeated legal maximum steps toward the close `r3/r4` pair. Exact allowed
endpoints pass; the first finite outward candidate fails with zero step.

## Scheduling, budget binding, and pending state

Pairs follow a fixed round-robin source-ID order. Observations for a pair are
adjacent in the accepted stream and never interleaved. Each identity has a
zero-based `identity_pair_index`: even values arrive `minus, plus`; odd values
arrive `plus, minus`. Alternation is per ID, not global. Scientific failures
are completed pairs and advance schedule/history/alternation even though they
do not change the center.

At a pair boundary:

```text
query_index = accepted_observations
pair_index = completed_pairs
identity_pair_index = target_identity.completed_pairs
target identity = identities[pair_index % 8]
```

Both frequencies and pair-local model values are frozen from the center at
pair start. The first observation cannot change the second frequency.

The total budget ceiling is immutable and bound by `reset`; query selection has
no budget argument. A cap below the initial charged usage is invalid. Equality
is valid but cannot start a pair. The budget dimensions are observations,
integration time, nominal exposure, and virtual elapsed time. Expected and
realized photons are never affordability inputs.

One prospective query charge is:

```text
observations + 1
integration_time_s + configured_integration_time_s
nominal_exposure_photons +
    (nominal_photon_rate_hz * configured_integration_time_s)
virtual_elapsed_time_s +
    (frequency_overhead_s + configured_integration_time_s)
```

The association shown is normative. Pair affordability starts from current
charged usage and applies this exact one-query transition twice in sequence;
it does not use `current + 2*cost`, multiplication, regrouping, `sum`, or
`math.fsum`. Every resulting capped field must be `<=` its ceiling. This is the
same floating transition that eventual observations use.

The four legal pending states are:

| State | Partial pair | Pending query | Legal action |
| --- | --- | --- | --- |
| pair boundary | `None` | `None` | reserve two charges; issue first, or mark budget stopped |
| first issued | `None` | first side | repeated choose returns same query; matching update accepts first |
| first accepted | exact first side | `None` | choose issues the already-reserved second side |
| second issued | exact first side | second side | repeated choose returns same query; matching update completes pair |

No other pending/partial combination is constructible. A pending first query
has `query_index=2*completed_pairs`; a pending second has
`query_index=2*completed_pairs+1`. Repeated selection with a pending query
returns the exact same frozen value and mutates nothing. After accepting the
first side, the second query is returned without rechecking affordability.

At an unreserved boundary, the first unaffordable selection atomically sets
`stopped_reason="budget_exhausted"` and returns `None`; all subsequent calls
return `None` unchanged. A stopped estimate has no pending or partial pair. A
normal budget stop therefore never creates an incomplete pair.

An ordinary instrument `Exception` before a successful query leaves the
pending query unchanged and may be retried because the instrument guarantees
ordinary-exception rollback. Process-control exceptions are outside that
guarantee. The rule is different after a successful instrument query, as
defined under aborted runs below.

## Observation acceptance, updates, and state atomicity

`update` accepts exactly an `EstimatorObservation`, not an
`InstrumentObservation`, subclass with extra fields, or arbitrary duck type.
It checks in this exact code/precedence order before mutation:

1. exact concrete type (`invalid_observation_type`);
2. one pending query (`no_pending_query`);
3. exact pending expected sequence index (`sequence_mismatch`);
4. exact pending frequency (`frequency_mismatch`);
5. exact configured integration duration (`integration_time_mismatch`);
6. exact expected endpoint (`endpoint_mismatch`), computed as
   `(previous_endpoint + overhead) + integration`;
7. exact nominal exposure (`nominal_exposure_mismatch`), computed as
   `rate * integration`; and
8. constructor-established finite fluorescence and nonnegative optional
   realized count (`invalid_observation_value`).

There is no tolerance window, sorting, resampling, duplicate suppression, or
timestamp inference. Duplicate and out-of-order delivery fail the exact
pending/sequence check. The tracker cannot infer a Poisson normalization from
the safe view and does not try.

The first valid side commits exactly one observation, one atomic tracking and
charged resource transition, its endpoint, and one immutable partial pair. It
does not update a center, pair history, pair counter, identity pair counter, or
lock state. The second side constructs and validates the complete prospective
pair result, candidate identity state, aggregate estimate, resource state,
history, and next schedule state before committing all of them together.
A scientific gate failure is a valid committed `lost` pair and charges both
observations. Any validation, arithmetic, or record-construction exception
leaves tracker state value-equal to its pre-call state.

`reset` likewise constructs the entire source/config/metadata/budget/seed/cell
state before replacing an existing run. A successful reset clears prior
pending/partial/history/resources/endpoints/counters/stopped reason and seed.
An invalid reset leaves the old run untouched.

## Hertz update and policy-state semantics

For a valid complete pair,

\[
\widehat{\Delta f_i}=\frac{e-e_0}{\kappa_i},
\qquad
s_i^{\mathrm{requested}}=K_p\widehat{\Delta f_i}.
\]

A positive innovation requests an upward frequency correction. Capture is
tested on the raw hertz innovation before gain or clipping. If gates pass,

\[
s_i=\operatorname{clip}
(s_i^{\mathrm{requested}},-s_i^{\max},+s_i^{\max}),
\qquad
q_i^{\mathrm{candidate}}=q_i+s_i.
\]

No predictor, smoother, integral term, derivative term, or cross-ID correction
is present.

The exact pair-decision precedence is:

1. Compute the pair sum from finite observations. A non-finite sum is
   `numerical_failure`; a finite sum `<=0` is `invalid_pair_normalization`.
2. Any non-finite discriminator, common diagnostic, innovation, requested
   step, clipped step, candidate, model value, or derived arithmetic is
   `numerical_failure`. Later gates are not evaluated.
3. If configured, `abs(common_mode_target_depths) > limit` gives
   `common_mode_limit_exceeded`; equality passes.
4. `abs(raw_innovation_hz) > capture_radius_hz` gives `capture_exceeded`;
   equality passes.
5. A finite candidate outside the inclusive allowed center interval gives
   `calibration_domain_exceeded`.
6. Otherwise apply the correction. Exact equality between requested and
   applied step gives `tracking`; strict clipping gives `step_limited`.

Every failure yields `lost`, preserves the active center/source epoch, and
applies exactly `0.0`. Invalid normalization has downstream discriminator,
innovation, step, candidate, and common diagnostic `None`. Numerical failure
retains only earlier finite diagnostics. Common, capture, and domain failures
retain every finite diagnostic computed before the gate; domain failure also
retains its finite rejected candidate.

The four lock values are policy classifications, not truth-certified lock
states. `tracking` means only that the public pair passed local policy gates;
the hidden target can still be outside the true capture interval under model
mismatch. Conversely, `lost` means a public gate failed, not that evaluator
truth proved loss. Stage 6.5 computes a separate truth-based lock metric after
release. A policy-lost cell is nonabsorbing: its next scheduled pair probes the
retained center and can return to `tracking` or `step_limited`; no reacquisition
scan is implied.

## Common-mode diagnostic and confounding

With pair-local modeled sum `S0=mu_minus+mu_plus`,

\[
m_i=\frac{(y_-+y_+)-S_0}{D_i}.
\]

It is dimensionless in calibrated target-pair-depth units and never enters the
hertz correction. In a static centered exact-model calculation, removing only
the target dip while retaining baseline and all other lines gives `m_i` equal
to one up to declared floating tolerance. The optional threshold is a
conservative policy gate only.

This value is not a target-contrast estimator. Baseline motion, common
fluorescence gain, target or neighboring linewidth/amplitude/center changes,
within-pair evolution, line-shape mismatch, and normalization error are
confounded. Similar asynchronous values across IDs are diagnostic evidence,
not proof of simultaneity or physical cause. Stage 6.3 does not feed an
aggregate correction back into any cell.

## Public reference, instrument truth time, release, and ages

An `EstimatorObservation.timestamp_s` is an integration endpoint; it does not
carry the instrument's exact evaluation midpoint. Stage 6.3 therefore names two
nearby but distinct binary64 conventions:

1. The tracker reconstructs a **public observation reference** only from its
   safe fields as `endpoint - integration / 2.0`. It takes the overflow-safe
   ordered mean of the first and second reconstructed references:

   ```text
   pair_reference_timestamp_s =
       first_public_midpoint
       + (second_public_midpoint - first_public_midpoint) / 2.0
   ```

   `TwoPointPairResult.pair_reference_timestamp_s` and a pair-derived
   `active_reference_timestamp_s` use this convention. It is a deterministic
   estimator timestamp, not a claim that it is bitwise the hidden evaluation
   time.
2. Before each query, `TwoPointEvaluatorRunner` computes the instrument's exact
   **measurement midpoint** with the current instrument association:

   ```text
   measurement_midpoint_s =
       (instrument.virtual_time_s + frequency_overhead_s)
       + integration_time_s / 2.0
   ```

   It stores this evaluator-only value in `TwoPointTrackingAcquisition` (and in
   the aligned verified-calibration midpoint tuple). A completed pair's
   `TwoPointEvaluatorPairTiming.truth_reference_timestamp_s` is the overflow-
   safe ordered mean of its two actual measurement midpoints. Evaluator truth
   and Stage 6.5 error/lock calculations use this value exactly, never the
   endpoint-reconstructed public reference.

`TwoPointEvaluatorPairTiming.public_reference_timestamp_s` must exactly equal
the pair result, so both conventions remain auditable. The pair release
sequence/timestamp is the second-arriving observation's endpoint. A correction
is unavailable at either reference and becomes causal only at release.

For the first fresh fixture pair, both midpoint associations happen to agree:

| Value | Decimal display | Exact binary64 witness |
| --- | ---: | --- |
| first endpoint | `0.006` | `0x1.89374bc6a7efap-8` |
| second endpoint/release | `0.012` | `0x1.89374bc6a7efap-7` |
| first midpoint under either convention | `0.0035` | `0x1.cac083126e979p-9` |
| second midpoint under either convention | `0.0095` | `0x1.374bc6a7ef9dbp-7` |
| ordered-mean reference | `0.006500000000000001` | `0x1.a9fbe76c8b43ap-8` |
| release minus public reference | `0.0055` | `0x1.6872b020c49bap-8` |

The distinction is pinned by first-cycle pair 3 (observation indices 6/7):
the evaluator truth reference is `0x1.5c28f5c28f5c4p-5`, while the public
endpoint-reconstructed reference is `0x1.5c28f5c28f5c2p-5`. Tests require this
inequality and require truth lookup at the former. All ordered means use
`first + (second - first) / 2.0`; `(first + second) / 2.0` is forbidden because
it can overflow and has different rounding.

For a pair-derived active center at the current accepted endpoint:

```text
estimate_age_sequence_indices =
    current_sequence_index - active_release_sequence_index
estimate_age_s =
    current_timestamp_s - active_reference_timestamp_s
release_age_s =
    current_timestamp_s - active_release_timestamp_s
```

The tracker's `estimate_age_s` is age relative to its public reference
convention. Calibration-seeded centers instead use the mapped, actual physical
fit epoch from the verified source and mapped availability. Sequence and
release ages are zero immediately on pair release, while public-reference age
is positive. Calibration sequence age exists only in included same-run mode;
it is `None` across separate sequence domains. Failed pairs and accepted first
sides age older active centers without pretending to refresh them. Evaluator-
reported truth-reference latency, when needed, uses the separate exact
instrument reference.

## Canonical public and evaluator resource arithmetic

`PublicAcquisitionResources` structurally mirrors `ResourceSnapshot` except
that it omits expected photons. The instrument ledger is authoritative about
the arithmetic: each successful acquisition performs one left-associated
binary64 `old + atomic_value` transition per floating field. The tracker and
evaluator replay the same transition, in observation arrival order, beginning
from an all-zero record (or, for included charging, from the already replayed
source record). They never recompute a subtotal by subtraction and never add a
calibration subtotal to a tracking subtotal.

For one safe observation and known overhead the transition is exactly:

```text
observations = old.observations + 1
integration_time_s = old.integration_time_s + observation.integration_time_s
nominal_exposure_photons =
    old.nominal_exposure_photons + observation.nominal_exposure_photons
realized_photons = old.realized_photons + (observation.realized_photons or 0)
observations_without_realized_counts =
    old.missing + int(observation.realized_photons is None)
virtual_elapsed_time_s =
    old.virtual_elapsed_time_s + (overhead + observation.integration_time_s)
```

The full evaluator replay adds
`old.expected_photons + observation.expected_photons` in the same step.
Canonical joins require exact field equality, including binary64 equality,
because both sides replay identical atomic terms in identical order. There is
no ULP window that could hide a regrouping error. For example, six successive
`0.005` additions must produce
`0x1.eb851eb851eb9p-6`; `math.fsum`'s neighboring value is not accepted.

The public estimate's `tracking_resources` and `charged_resources` count only
accepted tracker observations, including a first side and both sides of a
scientifically failed pair. Source resources are always reported. Evaluator
final resources additionally include one authenticated observation whose
tracker update aborted. A reserved but unreturned query and an instrument query
that raises before return are not resources. A returned raw observation whose
atom cannot join the authoritative ledger is preserved, but has no fabricated
aggregate evaluator replay.

`build_two_point_evaluator_resources` performs the only safe/full join:

1. It accepts only the original runner object after that runner has started
   tracking from one exact `VerifiedTwoPointCalibrationSuccess`. Calibration
   full observations and exact safe views come from that outcome and must match
   the immutable source one-to-one and in sequence. Caller-asserted sources have
   no eligible runner state and are rejected.
2. `accepted_tracking_observations` is the full-observation projection of
   `normal_tracking_trace`. Its safe views must exactly equal the observations
   reconstructed in arrival order from `pair_history` plus an optional partial
   pair, and its count equals `accepted_observations`. A pending query and a
   failed instrument call contribute nothing.
3. If the terminal abort holds a
   `TwoPointResourceJoinUnavailableAcquisition`, the builder verifies its raw
   record, nonempty mismatch tuple, authoritative before/after snapshots, and
   final runner snapshot, then returns `None`. It does not construct
   `TwoPointEvaluatorResources`, a one-observation delta, or any aggregate that
   purports to include the malformed atom.
4. Otherwise, `unaccepted_tracking_observations` is empty or is the one full
   observation in the authenticated terminal-abort acquisition. The accepted
   tuple never includes it. Thus a second-side abort may have both
   `incomplete_pair_observations == 1` and `unaccepted_observations == 1`;
   these describe different facts.
5. The evaluator replays calibration, accepted tracking, authenticated
   unaccepted tracking, total tracking, accepted charged, and final charged
   full resources from the atomic observations.
   `accepted_tracking_resources` replays the normal tuple from zero;
   `unaccepted_tracking_resources` replays the zero-or-one unaccepted tuple
   from zero; `tracking_resources` replays their acquisition-order
   concatenation from zero. Mixed counted and count-free observations update
   realized/missing totals independently.
6. Every authenticated acquisition's stored `instrument_resources_before`
   advances by its
   exact full atomic observation to its stored `after`; consecutive boundaries
   are exactly contiguous. The final boundary equals
   `state.instrument_resources_current`. A query exception instead requires
   equal before/after snapshots because current `ODMRInstrument.query` is
   atomic.
7. Included mode requires the same runner token and instrument, an all-zero
   calibration `before`, exact calibration-after/tracking-before continuity,
   and a final cumulative instrument snapshot equal to charged replay of
   calibration followed by all tracking observations. Conditional mode reports
   calibration from zero but charges only all tracking observations from zero;
   no boundary is claimed between a source runner and a distinct tracking
   runner.
8. Every non-expected field of `calibration_resources` equals the source safe
   resources, and every non-expected field of
   `accepted_tracking_resources` equals the safe estimate's tracking resources.
   The evaluator separately performs an accepted-only charged replay: source
   then accepted tracking atoms for included treatment, or accepted tracking
   atoms from zero for conditional treatment. It stores that full replay as
   `accepted_charged_resources` and requires its exact safe projection to equal
   `estimate.charged_resources` field-for-field.
9. `charged_resources` continues that same arrival-ordered accepted prefix with
   the optional authenticated unaccepted atom. It equals
   `accepted_charged_resources` when none exists and may differ only by that
   exact atom after an abort. Expected photons in both records remain
   evaluator-only; they are never projected into the estimate.
10. Within a constructed evaluator-resource record,
   `incomplete_pair_observations` is one exactly when the safe estimate has a
   partial pair, and `unaccepted_observations` is one exactly when an
   authenticated terminal-abort acquisition exists. Both are otherwise zero;
   the resource-unavailable case has no evaluator-resource record.

An extra, missing, duplicate, reordered, or safe-view-mismatched full
observation makes the join fail. Tests include six-observation regrouping,
exact-ceiling, mixed-count, scientific-failure, pending-query, incomplete-pair,
and first-/second-side abort cases.

Stage 6.5 uses equal total integration time as its primary matched budget. If
all methods share one nominal photon rate, equal integration also implies equal
nominal exposure; if rates differ, it does not and the mismatch is reported.
Signal-conditioned expected photons are evaluator-only reported outcomes, not
adaptive inputs or the primary stopping budget.

## Global state joins and transitions

Every `TwoPointEstimate` must satisfy all of these equalities:

```text
accepted_observations ==
    2 * completed_pairs + int(incomplete_pair is not None)
tracking_resources.observations == accepted_observations
sum(identity.completed_pairs for identity in identities) == completed_pairs
completed_pairs == len(pair_history)
pair_history[k].pair_index == k
if reset current index is n:
    current_sequence_index == n + accepted_observations
if reset current index is None and accepted_observations == 0:
    current_sequence_index is None
if reset current index is None and accepted_observations > 0:
    current_sequence_index == accepted_observations - 1
```

Identity tuple order, IDs, calibration FWHM, cell bounds, and allowed bounds
equal the calibration exactly. Each identity's `latest_pair` is the last
history pair for that ID or `None`; its `completed_pairs` is that count. A
successful latest pair is the active pair source and refreshes its reference
and release ages even when the applied step and numeric center change are
exactly zero. A failed latest pair is visible as policy state/failure while
active source fields still join to the most recent earlier successful pair or
calibration seed.

Resource joins are exact under the canonical replay above. `calibration_source_id`,
source provenance, budget treatment, calibration resources, budget ceiling,
and seed equal reset inputs in every estimate. `stopped_reason` is non-`None`
only in the boundary/no-pending/no-partial state.

Transition joins are:

| Transition | Required post-state |
| --- | --- |
| reset | zero tracking observations/pairs/history; no pending/partial/stop; calibration-seeded cells; source resources reported; charged start follows treatment |
| choose first | only `pending_query` appears; counters/resources/endpoints/cells unchanged |
| accept first | pending clears; exact partial appears; accepted/resources/endpoint advance once; pair/history/cells do not |
| choose second | exact partial remains; only matching second pending appears |
| accept second success | pending/partial clear; pair/history and both counters advance; target active source refreshes even for an exact zero step; all other cells value-equal |
| accept second policy failure | same schedule/history/counter advancement; target active source retained and policy state becomes lost |
| unaffordable boundary | only stopped reason changes; returns `None` |
| validation/calculation exception | every tracker field remains value-equal |

For a first-side `TwoPointUpdate`, `completed_pair is None` and its estimate's
partial query/observation equal the update fields. For a second-side update,
`completed_pair` equals the new history tail, both side query/observation joins
are exact, and the returned estimate has no partial or pending value.

## Evaluator runner, incomplete pairs, and terminal aborts

`TwoPointEvaluatorRunner` is the single owner of the emulator-side run
protocol. It retains the bound instrument object privately; its frozen `state`
is an immutable audit snapshot, not a replacement instrument handle. `bind`
reads the instrument's immutable rate and overhead properties, mints the opaque
token, and captures its current time/resource boundary. The official Stage 6.3
runner accepts only a clean instrument (`virtual_time_s == 0.0` and an all-zero
resource snapshot), so `instrument_current_sequence_index` begins as `None`.
An unclean bind raises
`TwoPointCalibrationPreflightError(code="unclean_instrument_boundary")` before
minting a usable runner.

The callable state machine is exact:

| Operation and input phase | Calls and successful transition |
| --- | --- |
| `acquire_verified_calibration` in `ready` | Uses the typed lossless acquisition path above. Success stores the exact outcome in both `calibration_outcome` and `verified_calibration`, advances the known current sequence/time/resource boundary, and enters `calibration_succeeded`. A typed failure is stored in `calibration_outcome`, advances only by its retained committed observations, and enters terminal `calibration_failed`. |
| `start_tracking` in `ready` | Allowed only for `conditional_free_precalibration` with a successful verified source from another runner. The clean tracking boundary requires metadata index `None` and time `0.0`; success stores the passed object in `verified_calibration` while `calibration_outcome` remains `None` because this runner did not acquire it. |
| `start_tracking` in `calibration_succeeded` | Accepts that exact success outcome and retains it in both calibration fields. `included_same_run` additionally requires the same outcome/source/token/instrument context and continuous boundary; conditional treatment may deliberately leave the source cost uncharged. |
| either successful `start_tracking` | Validates all joins below, calls `tracker.reset` exactly once, snapshots `tracking_resources_before`, stores the returned estimate, and enters `tracking`. It performs no instrument query. |
| `step` in `tracking`, unaffordable pair boundary | `choose_next_query()` returns `None` with the tracker stopped at its bound ceiling. The runner performs no instrument call, builds resources, and enters terminal `budget_stopped`. |
| `step` in `tracking`, query raises an ordinary `Exception` | Captures equal instrument before/after snapshots, leaves the tracker pending query and all normal traces unchanged, stores `last_instrument_failure`, and returns `TwoPointRunnerInstrumentFailure` while remaining in `tracking`. |
| `step` in `tracking`, query and update succeed | Stores one accepted acquisition, appends it to `normal_tracking_trace`, clears `last_instrument_failure`, and returns `TwoPointRunnerAccepted`. A completed pair also appends exactly one aligned evaluator pair timing. |
| `step` after a query return but before an accepted update | Performs the terminal-abort protocol below, appends nothing to the normal trace, and enters `aborted`. |
| `run_until_event` in `tracking` | Repeats `step` only across `TwoPointRunnerAccepted`; it returns immediately on the first instrument failure, budget stop, or abort. It never automatically retries an instrument failure. |
| `stop_external` in `tracking` | Performs no query/update, preserves the exact pending/partial state and any immediately preceding instrument-failure diagnostic, builds resources, and enters terminal `externally_stopped`. |

Every phase-invalid call fails before touching the instrument or tracker:
`acquire_verified_calibration` uses
`TwoPointCalibrationPreflightError(code="invalid_runner_phase")`,
`start_tracking` uses
`TwoPointRunnerStartError(code="invalid_runner_phase")`, and `step`,
`run_until_event`, and `stop_external` use `TwoPointRunnerStateError`. In
particular, after `calibration_failed`, `budget_stopped`, `externally_stopped`,
or `aborted` there is no retry, next query, update, calibration, restart, or
external-stop transition; inspection of `state` and resource building remain
read-only. A new run requires a newly bound runner.

`start_tracking` requires object identity between `calibration.source` and
`verified_calibration.source`, exact calibration/configuration equality with
the tracker, and exact tracking metadata rate/overhead equality with the
instrument-derived runner configuration. It also requires metadata
time/sequence to equal the runner's known current boundary. Included treatment
then requires that `verified_calibration` is the exact success stored by this
runner, its token is the runner's token, the private bound instrument is the
one that produced it, and its final resource/time/index boundary is the current
tracking boundary. Conditional treatment permits an exact success whose token
privately authenticates a different issuing runner/outcome/source, but makes no
cross-run boundary claim. The runner applies failures in
this stable `TwoPointRunnerStartError` precedence: runner phase, argument type,
verified-success shape, calibration/source/configuration equality, run-token
and treatment provenance, metadata/clock/sequence/time/configuration, resource
boundary, then tracker reset. These correspond in order to the declared start
failure codes; a reset exception is chained under `tracker_reset_failed`.
Every failure precedes an instrument query and leaves runner, instrument, and
tracker value-equal to entry.

At the start of `step`, the runner first calls `choose_next_query()`. If it
returns `None`, the budget-stop transition above completes without an
instrument call. Otherwise the returned query is the tracker's idempotent
pending query. Only then does the runner snapshot
`tracker_estimate_before = tracker.estimate()` and require
`tracker_estimate_before.pending_query == query`; thus the snapshot already
contains a newly issued first query as well as a previously pending retry. It
then snapshots instrument resources/time and computes
`expected_measurement_midpoint_s` with the instrument's exact association.

After a full observation returns, the runner retains its exact safe view and
authoritative instrument before/after snapshots before any validation outcome.
It first prospectively replays the raw full atom from `before`, using bound
overhead, and compares every field to `after` without subtraction. A mismatch
creates `TwoPointResourceJoinUnavailableAcquisition` with every unequal field,
sets `measurement_midpoint_s` only if its independent timing checks passed, and
enters terminal abort with reason `resource_join_unavailable`, `None` exception
fields, and `resources=None`; it does not call `tracker.update`.

When the full resource transition joins exactly,
`TwoPointTrackingAcquisition` is mandatory and contains the exact canonical
one-observation delta. The remaining query/observation contract is then checked
against the immutable instrument configuration and before boundary before the
midpoint is promoted to the non-`None` exact `measurement_midpoint_s`. A
non-resource mismatch remains a committed, resource-authenticated acquisition;
the runner creates a coded `TwoPointObservationValidationError` and aborts
without calling `tracker.update`. Otherwise it calls
`tracker.update(full_observation.estimator_view())` exactly once.

Once `ODMRInstrument.query` returns successfully, that full acquisition and
cost are committed even if update raises. The runner catches every ordinary
`Exception`, constructs `TwoPointAbortedRun`, and enters terminal `aborted`.
Every abort retains one of the two exact acquisition variants, a stable reason,
and an unaccepted count of one. An authenticated variant retains its canonical
one-observation delta, concrete exception type/message, and mandatory evaluator
resources. The resource-unavailable variant instead retains the malformed raw
full/safe record, mismatch fields, and authoritative snapshots, with no delta,
exception, or aggregate evaluator resources. Neither path subtracts cumulative
floats or discards the observation. Because the pre-update snapshot is taken
after query issuance and every covered tracker failure is ordinary-`Exception`
transactional, tracker before/after estimates are value-equal while the
instrument sequence, endpoint, and authoritative resources advanced once.

`TwoPointObservationValidationError` maps to
`tracker_observation_validation_error`; `TwoPointUpdateConstructionError` maps
to `tracker_update_construction_error`; every other ordinary update exception
maps to `tracker_update_unexpected_error`. The two named tracker exceptions are
public classes, so classification never parses messages. Process-control
`BaseException` subclasses are outside `step` outcome and transactional-abort
guarantees and may propagate unchanged. Stage 6.3 makes no promise about runner
terminalization or equal tracker snapshots after such an asynchronous process-
control interruption; cooperative cancellation requiring a typed outcome must
use an ordinary `Exception` covered above.

External interruption can leave one accepted first side even though normal
budget stopping reserves pairs. The estimate and evaluator resources then
retain that exact partial pair and its charge but emit no discriminator,
center update, completed pair, or fabricated partner. The half-pair is neither
discarded nor classified as scientific lock failure. An external stop may also
retain an already issued pending second query; it costs nothing. If the second
query instead returned and its update aborted, the accepted first side remains
the incomplete pair and the second observation appears only as the single
unaccepted abort acquisition.

Runner-state constructors enforce only local phase/discriminator structure;
the runner transition authenticates history. `ready` has no calibration,
tracker, trace, or tracking boundary. `calibration_failed` has exactly one
failure outcome and no tracker. `calibration_succeeded` has the same success
object in its two calibration fields and no tracker. Every tracking or later
phase has a verified success, calibration, estimate, and tracking boundary;
its instrument-current sequence/time/resource fields equal the runner-observed
endpoint.
Only `aborted` has `terminal_abort`; only `tracking` or an external stop taken
immediately after a query failure may retain `last_instrument_failure`. Pair
timings are one-to-one with completed pairs in the normal trace and never
include an unaccepted acquisition.

## Deterministic contract verification matrix

In addition to the scientific fixtures below, implementation tests pin these
causal boundaries:

- Instrument configuration properties reproduce constructor-canonical rate
  and overhead exactly. Included start succeeds only with exact metadata,
  source, success object, token, instrument identity, clock/index/time, and
  continuous resource boundary. Separate tests perturb only rate, only
  overhead, token/runner, copied source, calibration source, and one boundary
  field; each raises the stable start error before reset or query. Conditional
  start on a clean second runner succeeds while preserving different source
  and tracking rate/overhead declarations.
- Each public record rejects violations of its intrinsic fields. Context-only
  counterexamples are then passed to the named owner: a locally valid update
  with the wrong prior pending query, estimate state inconsistent with reset,
  fault-injected evaluator trace inconsistent with the runner boundary, and an
  abort inconsistent with its runner acquisition are rejected by that
  transition or builder. A directly constructed resources/abort record is not
  upgraded to authoritative provenance. No constructor test pretends those
  absent facts self-authenticate.
- Exception-contract tests exhaust every literal construction, observation,
  update, and verified-preflight code, check declared inheritance and nonempty
  messages, and combine adjacent preflight/construction defects to pin exact
  precedence, including `invalid_argument_value` and `invalid_source_epoch`.
- Verified acquisition tests cover clean success; a preflight defect with zero
  queries; a returned first-observation contract mismatch with one `None`
  midpoint slot; an instrument exception after two successes; a structured fit
  failure; a fitting exception; and source-binding failure. Every post-start
  outcome preserves exactly the committed full/safe observations, aligned
  midpoint slots, fit/exception discriminator, and before/after snapshots.
  Every non-resource-mismatch outcome has mandatory canonical safe/full replay.
  A failed runner accepts no later call.
- Separate calibration and tracking fault-seam tests alter only raw
  `integration_time_s`, only raw `nominal_exposure_photons`, and only evaluator-
  side `expected_photons`, each far enough to change the prospective ledger.
  Each calibration case returns `resource_join_unavailable`; each tracking case
  returns a terminal abort of that reason. All six retain the raw record and
  authoritative snapshots, list exactly every resulting unequal resource
  field in alias order, expose no delta or aggregate resources, and never use
  subtraction. Integration corruption also names virtual elapsed whenever its
  prospective replay changes that field. Expected corruption is never visible
  in the safe observation or tracker.
- Runner tests cover accepted first and second sides, completed-pair timing,
  exact budget stop, external stop both before and after issuing a pending
  second query, a nonmutating instrument failure, and an explicit later retry
  of the identical pending query. `run_until_event` returns the first instrument
  failure without retrying it. Fresh and retry aborts snapshot only after the
  pending query exists, so their before/after tracker estimates are equal and
  contain that exact query. Tests make no transactional claim for arbitrary
  process-control `BaseException`.
- Abort tests inject evaluator acquisition mismatch, the public validation
  exception, a late `TwoPointUpdateConstructionError`, and an arbitrary
  ordinary update exception. Each resource-authenticated case retains exactly
  one full/safe observation and canonical delta, keeps tracker snapshots equal,
  excludes it from the normal trace, includes it in evaluator resources, and
  rejects subsequent mutation. The unavailable variant has the same terminal
  behavior but returns `resources=None`. First- and second-side aborts pin the
  independent incomplete/unaccepted counts.
- Midpoint tests reproduce the exact source actual/public hexadecimal
  witnesses and the pair-3 truth/public mismatch. A rejecting dynamics sentinel
  accepts only the actual evaluator truth reference. The fixed negative mapped
  calibration epoch constructs successfully; negative observation endpoints,
  release endpoints, or ages fail, and age subtraction is never clamped.
- Caller-asserted source tests require the exact overflow-safe ordered mean of
  first/last public reconstructed midpoints. The exact public witness passes;
  each neighboring ULP, either interval endpoint, and one ULP outside either
  endpoint fail with `invalid_source_epoch`, while the verified actual-midpoint
  witness remains distinct.
- Resource-builder tests require the accepted-only full charged prefix's safe
  projection to equal `estimate.charged_resources` in both treatments. A normal
  abort advances only final evaluator charged resources by its authenticated
  unaccepted atom; an unavailable abort returns no evaluator aggregate.
- Every noiseless zero-step pair replaces the active source pair index and
  timestamps and resets that identity's release age while preserving its
  center bit-for-bit. A failed pair still retains the preceding successful
  source.

## Seed and deterministic policy

`reset` requires a seed keyword with no default. It accepts Python/NumPy
integers, rejects booleans and negative integers, stores a canonical
nonnegative Python `int`, and reserves it for future estimator-local
randomness. Stage 6.3 draws no random numbers: ID order and side alternation are
fixed, and repeated selection consumes no RNG state. A paired test changes only
the tracker seed and requires every query, update, estimate, and resource to
remain equal except the reported seed. Instrument seed remains independently
capable of changing Poisson observations.

## Closed generated-regression fixture

These values are proposed regression defaults, not measured performance or
claims of optimality.

### Shared tracker policy and source model

| Quantity | Exact fixture value |
| --- | ---: |
| offset | `0.35 * fitted FWHM` |
| capture radius | `0.20 * fitted FWHM` |
| proportional gain | `1.0` |
| maximum step | `0.10 * fitted FWHM` |
| integration per query | `0.005 s` |
| frequency overhead | `0.001 s` |
| nominal photon rate | `5.0e8 photons/s` |
| tracker seed | `20260904` |
| source-instrument seed | `20260903` |
| tracking-instrument seed | `20260905` |

The named source family is `stage63_eight_line_v1`, copied from the physical
model values in `configs/drift.yaml` but constructed directly in the test so a
later YAML edit cannot silently change it:

```text
baseline = (intercept=1.0, reference_hz=2.870e9,
            slope_per_hz=1.0e-11, quadratic_per_hz2=0.0)
r0 = (2.805e9, 2.5e6, 0.018, 0.35)
r1 = (2.825e9, 2.7e6, 0.021, 0.40)
r2 = (2.845e9, 2.9e6, 0.023, 0.45)
r3 = (2.865e9, 3.1e6, 0.025, 0.50)
r4 = (2.875e9, 3.1e6, 0.024, 0.50)
r5 = (2.895e9, 2.9e6, 0.022, 0.45)
r6 = (2.915e9, 2.7e6, 0.020, 0.40)
r7 = (2.935e9, 2.5e6, 0.017, 0.35)
# tuple entries are (center_hz, fwhm_hz, amplitude, eta)
```

The verified source uses a stationary version of this model, zero-standard-
deviation Gaussian normalized noise, and frequencies
`2.740e9 + k*62_500.0 Hz` for every integer `k` from 0 through 4480. It makes
4481 adjacent `0.005 s` acquisitions with `0.001 s` overhead on a clean source
instrument. The public fit uses `initial_guess=None` and exactly:

```python
FitConfiguration(
    model_kind="pseudo_voigt",
    baseline_degree=1,
    resonance_ids=("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"),
    min_fwhm_hz=2.0e5,
    max_fwhm_hz=8.0e6,
    max_amplitude=0.08,
    min_resolved_amplitude=1.0e-4,
    min_center_separation_hz=1.0e6,
    savgol_window=11,
    savgol_polyorder=2,
    relative_prominence=0.01,
    allow_fallback=False,
    max_nfev=4000,
    rank_rtol=1.0e-10,
    min_baseline_sse_improvement=1.0e-4,
    min_amplitude_significance=5.0,
)
```

The source must fit successfully; the tracker fixture then constructs its
hidden initial spectrum from that returned public fit, in fit tuple order. Fit
outputs are not replaced by generator truth or hard-coded expected centers.
The exact source boundaries and public acquisition values are:

| Field | Exact value |
| --- | --- |
| source ID | `stage63-calibration-v1` |
| source clock | `stage63-calibration-clock-v1` |
| first/last sequence | `0` / `4480` |
| source start | `0x0.0p+0` |
| first endpoint | `0x1.89374bc6a7efap-8` |
| last endpoint and availability | `0x1.ae2d0e560425fp+4` |
| first/last actual instrument midpoint | `0x1.cac083126e979p-9` / `0x1.ae22d0e5604efp+4` |
| physical fit epoch, actual ordered mean | `0x1.ae3126e978e27p+3` |
| last public reconstructed midpoint (diagnostic only) | `0x1.ae22d0e5604eep+4` |
| public reconstructed endpoint mean (exact caller-asserted convention) | `0x1.ae3126e978e26p+3` |
| frequency bounds | `2.740e9` / `3.020e9 Hz` |
| observations | `4481` |
| integration total | `0x1.667ae147ae117p+4 s` |
| nominal exposure | `11_202_500_000 photons` |
| realized/missing-count observations | `0` / `4481` |
| virtual elapsed | `0x1.ae2d0e5604269p+4 s` |

Core tracker regressions use a clean separate tracker clock named
`stage63-tracking-clock-v1`, map source availability to tracker time zero with
offset `-0x1.ae2d0e560425fp+4`, and therefore map the physical fit epoch to
`-0x1.ae28f5c28f697p+3`. The neighboring public endpoint-reconstructed value is
`-0x1.ae28f5c28f698p+3` and is diagnostic only. They set
`current_sequence_index=None`, `current_timestamp_s=0.0`, and explicitly use
`conditional_free_precalibration`. The source model was stationary throughout
calibration, so the tracking hidden model begins at the fitted public centers
at tracker time zero; the nonzero reported calibration age is still retained.

### Exact run cases and ceilings

Each cap below is generated by the canonical repeated one-query transition,
not decimal multiplication. All four caps are present.

| Case | Cycles / observations | Integration cap | Nominal cap | Elapsed cap |
| --- | ---: | --- | ---: | --- |
| static/noiseless | `2 / 32` | `0x1.47ae147ae147dp-3` | `80_000_000` | `0x1.89374bc6a7efdp-3` |
| static/Poisson | `4 / 64` | `0x1.47ae147ae147ep-2` | `160_000_000` | `0x1.89374bc6a7efep-2` |
| common linear drift | `30 / 480` | `0x1.33333333332f2p+1` | `1_200_000_000` | `0x1.70a3d70a3d6c6p+1` |
| deterministic loss/recovery | `3 / 48` | `0x1.eb851eb851ebdp-3` | `120_000_000` | `0x1.26e978d4fdf3ep-2` |

The corresponding clean instrument endpoints are, respectively,
`0x1.89374bc6a7efep-3`, `0x1.89374bc6a7effp-2`,
`0x1.70a3d70a3d673p+1`, and `0x1.26e978d4fdf3fp-2`. The one-ULP differences
from resource elapsed totals are expected because the clock performs overhead
and integration additions separately.

The cases are fixed as follows:

- **Static/noiseless:** zero Gaussian noise and the exact fitted hidden model.
  Every pair must have `discriminator == zero_discriminator`, exact zero raw,
  requested, and applied correction, and policy state `tracking`. Every such
  zero-step success becomes that identity's active pair source and refreshes
  its reference/release timestamps and ages while leaving only the numeric
  center unchanged. The schedule, IDs, endpoints, ages, cells, counters, and
  resources are exact. The next pair after observation 32 is unaffordable and
  records `budget_exhausted`.
- **Static/Poisson:** `PoissonNoise` with tracking instrument seed `20260905`.
  Two independently constructed four-cycle runs must match every query,
  realized count, public/evaluator resource, diagnostic, policy state, and
  estimate. A run differing only in instrument seed must differ in at least one
  of the 64 realized counts. No accuracy superiority is inferred.
- **Common linear drift:** all fitted centers begin at their public fit values
  at tracker time zero and slew at exactly `5.0e5 Hz/s`; all other parameters
  remain fixed; noise is zero Gaussian. Thirty full cycles must preserve ID
  order and domain, produce finite diagnostics, have the correction sign of
  the evaluator truth displacement whenever its magnitude exceeds
  `1e-12*w_i`, avoid policy loss, and keep the pair-release center within
  `0.05*w_i` of truth at that pair's evaluator-owned actual instrument
  reference. This is a fixture acceptance guard, not a general tracking claim.
- **Deterministic contrast loss/recovery:** zero Gaussian noise,
  `common_mode_limit_target_depths=0.5`, and otherwise the static model. Only
  target `r3` has amplitude zero from
  `0x1.15810624dd2f4p-3` through `0x1.21cac083126ecp-3` tracker seconds,
  inclusive. Those are observation indices 22 and 23, global pair 11,
  identity pair 1, arriving `plus` then `minus`. Its diagnostic must satisfy
  `abs(m-1.0) <= 1e-12`, the common-mode gate wins, applied step is exact zero,
  and every other cell is value-equal to its pre-pair value. Amplitude is
  restored immediately afterward. The next `r3` pair is global pair 19 at
  observation indices 38/39 and must return to `tracking` or `step_limited`.

The included-budget contract has a separate verified, static, zero-noise test
that continues the same clean source instrument and shared clock. Reset starts
at sequence 4480 and source availability, uses `included_same_run`, and binds a
one-pair total ceiling of 4483 observations, integration
`0x1.66a3d70a3d6d9p+4`, nominal exposure `11_207_500_000`, and elapsed
`0x1.ae5e353f7cfb9p+4`. One pair fits exactly; the next does not. Its final
instrument endpoint is `0x1.ae5e353f7cfafp+4`. This pins continuous source-plus-
tracking arithmetic and prevents subtotal regrouping.

### Photon-rate sizing, not a result

At `0.005 s`, the declared `5.0e8 photons/s` rate gives `2.5e6` nominal
photons per query. For an isolated sizing calculation using the weakest source
target (`A=0.017`, `w=2.5e6 Hz`, `eta=0.35`, unit background), the model at
`0.35 FWHM` gives approximately `mu=0.98814` and
`kappa=9.219e-9 Hz^-1`. First-order independent Poisson propagation gives

\[
\sigma_e\simeq(2\mu N)^{-1/2}=4.50\times10^{-4},
\qquad
\sigma_f/w\simeq\sigma_e/(\kappa w)=0.0195.
\]

This analytic scale only makes the seeded stochastic regression meaningful
relative to the `0.10 FWHM` step and `0.20 FWHM` capture settings. It is not an
observed error, hardware recommendation, sensitivity estimate, or benchmark
claim.

## Truth-isolation verification

For sign/accuracy assertions, the evaluator calls hidden dynamics exactly once
at each completed pair's
`TwoPointEvaluatorPairTiming.truth_reference_timestamp_s`, only after the
tracker has released the pair result, and stores that snapshot in a separate
fixture-owned truth record. That timestamp is the actual instrument-midpoint
ordered mean retained by the runner, not the tracker's public endpoint-
reconstructed reference. The pair-3 neighboring hexadecimal values above are
a mandatory sentinel: truth lookup at the public value fails the regression.
Truth is never passed into calibration, reset, query selection, update,
estimate, budget checks, common-mode gates, or resource replay.

The verified calibration factory uses only safe views to construct and fit its
sweep; full source observations remain evaluator-side solely for expected-
photon/resource joining. Tests use rejecting sentinels and field inspection to
prove that a full observation, snapshot, dynamics object, noiseless callback,
future observation, or expected count cannot enter or be retained by the
tracker. Post-run truth cannot change a query or estimator record.

## Documentation and completion boundary

Implementation will later add researcher guidance for source provenance,
required budget labels, fixed cells, alternating side order, discriminator
sign, policy-state caveats, common-mode confounding, partial/aborted resource
reporting, and physical-reference versus release age. A download-free example
may print these diagnostics but may not claim realtime, photon, or accuracy
superiority.

Stage 6.3 is complete only after a separate reviewed implementation plan,
red/green implementation, all closed fixtures, documentation/example, full
test suite, Ruff, package build, installed-wheel smoke, and independent review.
This design revision alone does not complete the stage.

## Explicitly out of scope

- new FWHM, amplitude, eta, contrast, Q, uncertainty, or sensitivity estimates;
- the five-point local linewidth update reserved for Stage 6.4;
- matched-budget estimator comparisons and metrics reserved for Stage 6.5;
- nonlinear discriminator inversion, velocity prediction, smoothing, integral
  or derivative control, Kalman/state-space models, or shared-Hamiltonian
  feedback;
- truth-certified online lock status or using post-release truth for a gate;
- aggregate common-mode correction or a claim that the diagnostic identifies a
  baseline, contrast, temperature, magnetic-field, or other physical cause;
- scan-based reacquisition, adaptive offsets, pair interleaving, random pair
  order, continuing after an update abort, or dropping an acquired half-pair;
- dynamic calibration cells, collision resolution, frequency-sorted identity
  reassignment, unresolved hyperfine guarantees, or model-mismatch robustness;
- recorded-playback evaluation at frequencies absent from a recording;
- using expected photons, noiseless fluorescence, hidden trajectory, future
  observations, or post-run truth in calibration or online decisions; and
- any benchmark result, realtime-performance claim, hardware-readiness claim,
  or assertion that the proposed regression settings are optimal.
