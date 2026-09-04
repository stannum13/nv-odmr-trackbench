# Calibrated Two-Point Center Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Stage 6.3 calibrated two-point center tracker,
its lossless evaluator-owned calibration/run protocol, exact resource joins,
and closed generated regressions without starting Stage 6.5 comparisons.

**Architecture:** Estimator-safe immutable records, public-resource replay,
calibration binding, analytic local calibration, and the tracker state machine
remain under `odmr_bench.estimators`. Full observations, expected photons,
actual instrument midpoints, opaque run provenance, resource authentication,
and the instrument-owning state machine live only in the focused
`odmr_bench.evaluation.two_point` package. The evaluator releases one safe
observation at a time; pair-local public calculations are committed
transactionally only after the second flank, while evaluator truth lookup stays
post-release and fixture-owned.

**Tech Stack:** Python 3.11+, frozen/slotted dataclasses, NumPy, SciPy,
pytest, Ruff, Hatch build, existing `odmr_bench.models`,
`odmr_bench.dynamics`, `odmr_bench.emulator`, and `odmr_bench.estimators`
APIs.

## Global Constraints

- The binding source of truth is
  `docs/superpowers/specs/2026-09-04-calibrated-two-point-tracker-design.md`;
  implementation may add private helpers but may not rename, weaken, or extend
  its public fields, aliases, error codes, signatures, precedence, or state
  transitions.
- Every production function, method, property, record, and private arithmetic
  helper is introduced only after a focused behavioral test exists and has
  been observed failing for the expected missing behavior. A collection error
  is acceptable only for the first public surface in a task; subsequent REDs
  must reach the intended behavior.
- Public frequency/linewidth values are Hz, time is seconds, normalized
  fluorescence is dimensionless, FWHM is the linewidth convention, and the
  tracker estimates centers only. It does not estimate FWHM, amplitude, eta,
  contrast, Q, uncertainty, sensitivity, or a truth-certified lock state.
- The tracker may retain only immutable public configuration, one immutable
  calibration, run metadata, budget, estimator-safe observations, pending/pair
  state, and eight identity cells. It never receives or retains an
  `ODMRInstrument`, `SpectralDynamics`, `SpectralSnapshot`,
  `InstrumentObservation`, expected photons, noiseless callback, evaluator
  reference, future observation, or post-run truth.
- `included_same_run` and `conditional_free_precalibration` remain explicit,
  required, machine-readable treatments. Calibration is charged once in the
  former and separately reported but uncharged in the latter.
- Verified calibration uses model kind `"pseudo_voigt"`, quantity
  `"normalized_fluorescence"`, and normalization rule
  `"odmr_instrument_normalized_fluorescence_v1"` exactly; caller-asserted
  normalization and sampling labels remain nonempty declarations.
- All resource floats use the instrument's left-associated, one-observation
  arrival-order transitions. Never derive a delta by subtraction, add
  calibration and tracking subtotals, multiply a two-query cost, call `sum` or
  `math.fsum` for ledger accumulation, or compare resource floats with a
  tolerance.
- Expected photons remain evaluator-only and are never an affordability input.
  A malformed raw atom produces typed `resource_join_unavailable`; it is
  retained with authoritative boundaries and never given a fabricated delta or
  aggregate replay.
- Pairs are adjacent, fixed-ID round robin, and per-ID side alternating. Pair
  frequencies/model values freeze at pair start. Budget affordability reserves
  two sequential atomic charges before a first side and is not rechecked before
  the reserved second side.
- All public records are frozen and slotted and validate only intrinsic fields.
  Defensive copying is selective: mutable/value-bearing inputs are snapshotted,
  while capability-bearing provenance retains exact identity. The named
  factory, tracker transition, runner, or resource builder owns every
  contextual join.
- Accepted NumPy scalar integers/reals canonicalize to Python `int`/`float`;
  booleans, complex values, arrays in scalar positions, invalid exact record or
  container types, and non-finite values are rejected in the specified order.
- Ordered means use `first + (second - first) / 2.0`. The tracker uses public
  endpoint-reconstructed midpoints; evaluator truth uses actual instrument
  midpoints. The two are never substituted or regrouped.
- An ordinary post-query `Exception` becomes one terminal typed abort with
  equal pending-bearing tracker snapshots. `BaseException` process-control
  subclasses may propagate and carry no transactional guarantee.
- The Stage 6.3 seed is stored but consumes no RNG state. The schedule and
  estimates are seed-independent; the separately seeded instrument may change
  Poisson observations.
- Tests and examples are generated and download-free. Public text calls policy
  states diagnostics, not physical truth, and makes no realtime, photon,
  accuracy, hardware, optimality, or matched-budget-superiority claim.
- Equal-integration matched comparison, truth-based benchmark metrics, and any
  superiority claim remain Stage 6.5. Do not add comparison metrics, ranking,
  plots, or assertions to Stage 6.3.
- Use `.venv/bin/python -m pytest` for every RED/GREEN. Each task ends with its
  focused files, `tests/estimators tests/evaluation tests/emulator`, full
  pytest, Ruff, `git diff --check`, diff inspection, one atomic commit, and a
  fresh task review before the next task begins. An immediate task-review fix
  first gets a RED regression and is amended into that task commit. Integrated
  review fixes are new atomic commits, one complete fix wave per commit.

### Exact Copy-versus-Identity Matrix

| Owner/field | Required handling | Assertion |
|---|---|---|
| Caller-provided mutable sequences, configuration ID inputs, observation collections, and fit-owned arrays | Defensive value snapshot into the declared immutable tuple/record/array representation | Mutating the caller's original does not change the stored value. |
| `EstimatorObservation`, fit configuration, fitted model/result values, resource values, and public metadata/configuration values | Reconstruct or defensively snapshot value data where the owning public record accepts it | Stored values remain immutable and equality-stable. |
| `TwoPointCalibration.configuration` | Defensive value snapshot, exactly as required by the design | `calibration.configuration == supplied` and `calibration.configuration is not supplied`. |
| `TwoPointCalibration.source` and every verified outcome/source field | Retain the exact object | `calibration.source is source`; a value-equal copied source fails authentication. |
| `VerifiedInstrumentRunToken` | Retain opaque object identity; never copy, serialize, or compare by value | Only the original minted token resolves in the private registry. |
| `VerifiedTwoPointCalibrationSuccess` stored in runner state | Retain the exact outcome object in both success fields | `state.calibration_outcome is state.verified_calibration is success`. |
| `TwoPointEvaluatorRunner` and `ODMRInstrument` in private registry/builder context | Retain and authenticate exact identity | Copied/rebuilt/value-equal objects fail before any collaborator call. |

No snapshot operation may traverse or recreate a token, verified source,
verified outcome, runner, or instrument. The verified-source factory snapshots
its mutable fit/configuration/trace inputs once; consumers thereafter retain
that exact source object.

## File and Responsibility Map

| Path | Responsibility |
|---|---|
| `src/odmr_bench/estimators/two_point_types.py` | All estimator-facing aliases, errors, frozen records, intrinsic invariants, and defensive snapshots. |
| `src/odmr_bench/estimators/two_point_resources.py` | Private zero/advance/replay helpers for exact estimator-safe resource arithmetic and pair affordability; it never imports `ResourceSnapshot`. |
| `src/odmr_bench/estimators/two_point_calibration.py` | Caller-asserted source binding, evaluator-private verified binding seam, canonical target-only model/derivative, calibration slope/depth, and fixed cells. |
| `src/odmr_bench/estimators/two_point_tracker.py` | `CalibratedTwoPointTracker`, reset joins, pair reservation/scheduling, observation validation, pair policy, ages, and atomic state commits. |
| `src/odmr_bench/estimators/__init__.py` | Export every approved estimator-facing alias, error, record, factory, and tracker; keep private helpers internal. |
| `src/odmr_bench/emulator/instrument.py` | Add read-only canonical `nominal_photon_rate_hz` and `frequency_overhead_s` acquisition-configuration properties only. |
| `src/odmr_bench/evaluation/__init__.py` | Establish the evaluation namespace without re-exporting hidden implementation helpers. |
| `src/odmr_bench/evaluation/two_point/types.py` | Evaluator aliases/errors, opaque token, full-observation records, runner states/outcomes, and intrinsic invariants. |
| `src/odmr_bench/evaluation/two_point/provenance.py` | Private token mint/registry and exact runner/instrument/success/source identity binding. |
| `src/odmr_bench/evaluation/two_point/resource_accounting.py` | Private full-resource atomic replay/join helpers and public `build_two_point_evaluator_resources`. |
| `src/odmr_bench/evaluation/two_point/calibration.py` | Verified acquisition preflight, request loop, fit invocation, exact midpoint trace, typed success/failure construction, and token/source binding. |
| `src/odmr_bench/evaluation/two_point/runner.py` | `TwoPointEvaluatorRunner`, private token registry, start joins, tracking step/run/stop transitions, timing trace, and abort classification. |
| `src/odmr_bench/evaluation/two_point/__init__.py` | The sole evaluator public surface for all approved two-point evaluator names. |
| `tests/two_point_helpers.py` | Test-only legal public-fit/source/calibration/observation factories shared by estimator/evaluator tests; no production import may depend on it. |
| `tests/estimators/test_two_point_types.py` | Estimator-record intrinsic invariants, scalar canonicalization, error aliases, immutability, and truth-excluding fields. |
| `tests/estimators/test_two_point_resources.py` | Exact safe atomic replay, six-addition witness, mixed counts, projection, and sequential affordability. |
| `tests/estimators/test_two_point_calibration_source.py` | Asserted-source provenance, trace/fit/resource/epoch/clock/ID validation, and construction precedence. |
| `tests/estimators/test_two_point_calibration.py` | Analytic calibration, numerical derivative/monotonicity, all fixed cells, boundary geometry, and budget treatment. |
| `tests/estimators/test_two_point_tracker.py` | Reset, idempotent queries, schedule, reservation, observation precedence, pair gates, ages, seed, and state joins. |
| `tests/estimators/test_two_point_tracker_atomicity.py` | Reset/update construction failures, late arithmetic/record faults, and value-equal rollback. |
| `tests/emulator/test_instrument.py` | Exact read-only instrument configuration properties and unchanged atomic query behavior. |
| `tests/evaluation/test_two_point_types.py` | Opaque token and all evaluator record/state/outcome intrinsic matrices. |
| `tests/evaluation/test_two_point_calibration.py` | Verified preflight precedence, success, all typed calibration failures, midpoint alignment, and calibration resource-unavailable seams. |
| `tests/evaluation/test_two_point_runner.py` | Start provenance/boundary checks, step/retry/stop/budget state machine, pair timing, and ordinary-exception aborts. |
| `tests/evaluation/test_two_point_resources.py` | Full/safe joins, accepted charged prefix, optional abort atom, mixed counts, regrouping rejection, and unavailable result. |
| `tests/evaluation/test_two_point_regressions.py` | Closed static, Poisson, drift, contrast-loss/recovery, included-budget, midpoint sentinel, and post-release truth-isolation regressions. |
| `examples/track_two_point_centers.py` | Download-free conditional-precalibration diagnostic run with policy/resource/timing output and no comparison claim. |
| `docs/estimators.md`, `README.md` | Researcher guidance, public semantics, limitations, and example entry point. |
| `tests/test_package.py` | Public import smoke for installed estimator and evaluator surfaces. |
| `PROJECT_STATE.md`, `CHANGELOG.md` | Per-task evidence and final implementation/review status. |

The split is deliberate. Record validation can be rejected without touching
arithmetic; asserted provenance without accepting calibration geometry;
scheduling without accepting the discriminator policy; verified acquisition
without accepting tracking; runner transitions without accepting aggregate
joins; and scientific fixture acceptance without accepting public claims.

## Exact Public Interface Allocation

`src/odmr_bench/estimators/two_point_types.py` defines these exact aliases:

```python
CalibrationBudgetTreatment = Literal[
    "included_same_run", "conditional_free_precalibration"
]
CalibrationSourceProvenance = Literal[
    "verified_factory_acquisition", "caller_asserted"
]
CalibrationIdentityMode = Literal["require_expected_ids", "adopt_fit_ids"]
ClockMappingKind = Literal["shared_clock", "unit_scale_offset"]
PairSide = Literal["minus", "plus"]
TwoPointLockState = Literal["calibrated", "tracking", "step_limited", "lost"]
TwoPointFailureCode = Literal[
    "invalid_pair_normalization",
    "numerical_failure",
    "common_mode_limit_exceeded",
    "capture_exceeded",
    "calibration_domain_exceeded",
]
TwoPointStopReason = Literal["budget_exhausted"]
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
```

It defines these exact public classes:

```python
class TwoPointCalibrationConstructionError(ValueError):
    code: TwoPointCalibrationConstructionCode
    message: str
    def __init__(
        self, code: TwoPointCalibrationConstructionCode, message: str
    ) -> None: ...

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
```

The estimator records have exactly these fields; constructor syntax below is a
schema, and implementations use frozen/slotted dataclasses rather than custom
untyped mappings:

```python
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
    scale: float,
    offset_s: float,
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
    budget_treatment: CalibrationBudgetTreatment,
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
```

`TwoPointCalibration.budget_treatment` and tracker `reset(..., *, seed)` have
no defaults.

`src/odmr_bench/estimators/two_point_calibration.py` and
`src/odmr_bench/estimators/two_point_tracker.py` provide exactly:

```python
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
    budget_treatment: CalibrationBudgetTreatment,
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
```

`src/odmr_bench/evaluation/two_point/types.py` defines the approved aliases
`VerifiedCalibrationFailureCode`, `VerifiedCalibrationPreflightCode`,
`TwoPointRunnerPhase`, `TwoPointRunnerStartFailureCode`,
`TwoPointAbortReason`, `ResourceJoinMismatchField`,
`VerifiedTwoPointCalibrationOutcome`, `TwoPointRunnerStepOutcome`, and
`TwoPointRunnerRunOutcome` with every literal exactly as follows:

```python
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
ResourceJoinMismatchField = Literal[
    "observations",
    "integration_time_s",
    "nominal_exposure_photons",
    "expected_photons",
    "realized_photons",
    "observations_without_realized_counts",
    "virtual_elapsed_time_s",
]
VerifiedTwoPointCalibrationOutcome = (
    VerifiedTwoPointCalibrationSuccess | VerifiedTwoPointCalibrationFailure
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

It defines the errors
`TwoPointCalibrationPreflightError(ValueError)` with
`code: VerifiedCalibrationPreflightCode`,
`TwoPointRunnerStartError(ValueError)` with
`code: TwoPointRunnerStartFailureCode`, and
`TwoPointRunnerStateError(RuntimeError)`; opaque
`VerifiedInstrumentRunToken`; and these exact frozen/slotted records:
`TwoPointEvaluatorInstrumentConfiguration`,
`VerifiedCalibrationQueryRequest`, `VerifiedTwoPointCalibrationSuccess`,
`VerifiedTwoPointCalibrationFailure`, `TwoPointTrackingAcquisition`,
`TwoPointResourceJoinUnavailableAcquisition`,
`TwoPointEvaluatorPairTiming`, `TwoPointInstrumentQueryFailure`,
`TwoPointEvaluatorResources`, `TwoPointAbortedRun`,
`TwoPointEvaluatorRunnerState`, `TwoPointRunnerAccepted`,
`TwoPointRunnerInstrumentFailure`, `TwoPointRunnerBudgetStopped`,
`TwoPointRunnerExternallyStopped`, and `TwoPointRunnerAborted`. Fields and
literal discriminators remain exactly those in **Exact public contracts** of
the approved design.

The evaluator record field schemas are:

```python
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
```

`src/odmr_bench/evaluation/two_point/runner.py` and
`resource_accounting.py` provide exactly:

```python
class TwoPointEvaluatorRunner:
    @classmethod
    def bind(cls, instrument: ODMRInstrument) -> TwoPointEvaluatorRunner: ...
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

The two new `ODMRInstrument` properties have signatures
`nominal_photon_rate_hz(self) -> float` and
`frequency_overhead_s(self) -> float`; they expose constructor-canonical
configuration only.

---

## Execution Preflight

Before Task 1, record the exact implementation range base in the durable SDD
ledger:

- [ ] Run:

  ```bash
  IMPLEMENTATION_BASE=$(git rev-parse HEAD)
  test -n "$IMPLEMENTATION_BASE"
  mkdir -p .superpowers/sdd
  printf 'IMPLEMENTATION_BASE=%s
' "$IMPLEMENTATION_BASE" \
    >> .superpowers/sdd/progress.md
  ```

  Expected: the recorded hash is the reviewed plan commit from which
  implementation begins. Never recompute or replace it after Task 1.

- [ ] For each task, generate the task brief from this plan, run its RED and
  GREEN commands exactly, then run the task gate. Immediate task-review fixes
  get a focused RED and are amended into that task commit with
  `git commit --amend --no-edit`. Integrated-review fixes in Task 20 are new
  atomic commits, one commit per combined review-fix wave; do not rewrite
  already reviewed task history.

Every behavioral increment below is a strict six-action cycle. A checkbox that
says **Add** changes only the named test file. Its following **Run RED** checkbox
uses the literal command shown and must fail for the stated feature-missing
reason (an unexpected collection/import/fixture error is repaired and rerun,
not accepted). The next **Implement GREEN** checkbox changes only the listed
production files and follows the shown exact assertions/signatures/formulas;
the following **Run GREEN** repeats the identical node and must report PASS.
The subsequent **Refactor** checkbox is behavior-neutral and repeats that node.
The final task gate and commit are separate checkboxes. Do not combine, skip,
or reorder these actions, and never introduce a production function or method
before its owning observed RED.

---

### Task 1: Public Resource, Budget, Identity, Clock, Configuration, and Error Primitives

**Files:**

- Create: `src/odmr_bench/estimators/two_point_types.py`
- Create: `tests/two_point_helpers.py`
- Create: `tests/estimators/test_two_point_types.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Imports introduced now:** `PublicAcquisitionResources`,
`TwoPointBudgetCeiling`, `TwoPointIdentityBinding`,
`NormalizedFluorescenceProvenance`, `TwoPointClockMapping`,
`TwoPointTrackerConfiguration`, `TwoPointRunMetadata`,
`TwoPointCalibrationConstructionError`,
`TwoPointObservationValidationError`, and
`TwoPointUpdateConstructionError`; and aliases
`CalibrationBudgetTreatment`, `CalibrationSourceProvenance`,
`CalibrationIdentityMode`, `ClockMappingKind`, `PairSide`,
`TwoPointLockState`, `TwoPointFailureCode`, `TwoPointStopReason`,
`TwoPointCalibrationConstructionCode`, `TwoPointObservationValidationCode`,
and `TwoPointUpdateConstructionCode`. Source, calibration, query, pair,
estimate, update, factories, and tracker imports remain absent until their
owning tasks.

- [ ] **Write the import RED.** Add:

  ```python
  def test_two_point_primitive_names_are_public() -> None:
      from odmr_bench.estimators import (
          CalibrationBudgetTreatment,
          CalibrationIdentityMode,
          CalibrationSourceProvenance,
          ClockMappingKind,
          NormalizedFluorescenceProvenance,
          PairSide,
          PublicAcquisitionResources,
          TwoPointBudgetCeiling,
          TwoPointCalibrationConstructionCode,
          TwoPointCalibrationConstructionError,
          TwoPointClockMapping,
          TwoPointFailureCode,
          TwoPointIdentityBinding,
          TwoPointLockState,
          TwoPointObservationValidationCode,
          TwoPointObservationValidationError,
          TwoPointRunMetadata,
          TwoPointStopReason,
          TwoPointTrackerConfiguration,
          TwoPointUpdateConstructionCode,
          TwoPointUpdateConstructionError,
      )
      assert CalibrationBudgetTreatment
      assert CalibrationIdentityMode
      assert CalibrationSourceProvenance
      assert ClockMappingKind
      assert PublicAcquisitionResources
      assert TwoPointBudgetCeiling
      assert TwoPointIdentityBinding
      assert NormalizedFluorescenceProvenance
      assert PairSide
      assert TwoPointClockMapping
      assert TwoPointTrackerConfiguration
      assert TwoPointRunMetadata
      assert TwoPointCalibrationConstructionCode
      assert TwoPointCalibrationConstructionError
      assert TwoPointFailureCode
      assert TwoPointLockState
      assert TwoPointObservationValidationCode
      assert TwoPointObservationValidationError
      assert TwoPointStopReason
      assert TwoPointUpdateConstructionCode
      assert TwoPointUpdateConstructionError
  ```

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_two_point_primitive_names_are_public -q`.
  Expected RED: collection fails with
  `ImportError: cannot import name 'PublicAcquisitionResources'`.

- [ ] Add the exact aliases, error class signatures, and frozen/slotted field
  surfaces from **Exact Public Interface Allocation**; export only the names
  listed in this task's **Imports introduced now** block.

- [ ] Repeat
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_two_point_primitive_names_are_public -q`.
  Expected GREEN: `1 passed`.

- [ ] **Write the resource/budget behavior RED.** Add
  `test_public_resources_and_budget_validate_intrinsic_domains` with exact
  assertions:

  ```python
  resources = PublicAcquisitionResources(1, 0.005, 2.5e6, 0, 1, 0.006)
  assert resources.observations == 1
  assert type(resources.integration_time_s) is float
  numpy_resources = PublicAcquisitionResources(
      np.int64(1), np.float64(0.005), np.float64(2.5e6),
      np.int64(0), np.int64(1), np.float64(0.006)
  )
  assert type(numpy_resources.observations) is int
  with pytest.raises(ValueError, match="cannot exceed"):
      replace(resources, observations_without_realized_counts=2)
  with pytest.raises(ValueError, match="include integration"):
      replace(resources, virtual_elapsed_time_s=0.004)
  with pytest.raises(ValueError, match="at least one"):
      TwoPointBudgetCeiling(None, None, None, None)
  with pytest.raises(TypeError):
      replace(resources, observations=True)
  ```

  Do not add production serialization support.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_public_resources_and_budget_validate_intrinsic_domains -q`.
  Expected RED: at least the missing-count, elapsed, or
  empty-ceiling row reports `DID NOT RAISE`.

- [ ] Implement only canonical scalar and resource/budget intrinsic validation.

- [ ] Repeat
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_public_resources_and_budget_validate_intrinsic_domains -q`.
  Expected GREEN: `1 passed`.

- [ ] **Write identity/clock/configuration/error REDs.** Add parameterized node
  `test_identity_clock_configuration_and_errors_are_closed` asserting:
  require mode has exactly eight unique nonblank IDs; adopt mode has `None`;
  shared clock has equal nonblank IDs/zero offset; unit offset has distinct IDs;
  scale is exactly `1.0`; tracker defaults equal the approved values; metadata
  quantity is exactly `"normalized_fluorescence"`; each error accepts every
  declared code, rejects an unknown code, and requires a nonempty message.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_identity_clock_configuration_and_errors_are_closed -q`.
  Expected RED: an invalid mode/clock/error combination constructs.

- [ ] Implement only those intrinsic matrices.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Refactor duplicated scalar guards without changing behavior; repeat the
  three Task 1 nodes. Expected GREEN: `3 passed`.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py -q`,
  `.venv/bin/python -m pytest tests/estimators tests/emulator -q`,
  `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check .`, and
  `git diff --check`. Record counts and inspect only Task 1 files.

- [ ] Stage only the files listed for Task 1 and commit with message
  `feat: add two-point primitive contracts`.

---

### Task 2: Calibration Source and Calibration Record Contracts

**Files:**

- Modify: `src/odmr_bench/estimators/two_point_types.py`
- Modify: `tests/two_point_helpers.py`
- Modify: `tests/estimators/test_two_point_types.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Imports introduced now:** `TwoPointCalibrationSource`,
`TwoPointIdentityCalibration`, and `TwoPointCalibration`.

- [ ] Add
  `test_calibration_record_names_are_public` importing exactly those three
  names.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_calibration_record_names_are_public -q`.
  Expected RED: `ImportError` for `TwoPointCalibrationSource`.

- [ ] Add the exact frozen/slotted field surfaces and exports, retaining the
  source and configuration arguments temporarily without semantic validation.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_caller_asserted_source_snapshots_values_and_rejects_verified_direct_construction`:

  ```python
  mutable_observation_list = list(make_legal_source_observations())
  source = make_legal_caller_asserted_source(
      source_observations=mutable_observation_list
  )
  assert source.provenance == "caller_asserted"
  assert source.source_observations is not mutable_observation_list
  assert source.source_observations == tuple(mutable_observation_list)
  with pytest.raises(ValueError, match="verified"):
      replace(source, provenance="verified_factory_acquisition")
  ```

  Also mutate the original list, fit-owned arrays, configuration ID list, and
  observation inputs after construction; stored value data remains unchanged.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_caller_asserted_source_snapshots_values_and_rejects_verified_direct_construction -q`.
  Expected RED: direct verified construction succeeds or a
  mutable/value alias changes the record.

- [ ] Implement source value snapshots and local endpoint/bounds/ID/resource
  equalities only.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_calibration_records_preserve_source_identity_and_snapshot_configuration`.
  Assert:

  ```python
  mutable_configuration = make_legal_tracker_configuration()
  identities = make_legal_identity_calibrations()
  treatment = "conditional_free_precalibration"
  calibration = TwoPointCalibration(source, mutable_configuration, treatment, identities)
  assert calibration.source is source
  assert calibration.configuration == mutable_configuration
  assert calibration.configuration is not mutable_configuration
  assert calibration.identities == tuple(identities)
  ```

  Require exactly eight unique ordered identities, source/configuration/ID
  equality, positive geometry, and nonempty allowed intervals.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_calibration_records_preserve_source_identity_and_snapshot_configuration -q`.
  Expected RED: source is copied, configuration is aliased, or an invalid row
  constructs.

- [ ] Implement the calibration intrinsic matrix. Retain the exact source
  object; defensively snapshot configuration/value records and sequences.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Refactor snapshot helpers without traversing capability-bearing fields;
  repeat all three Task 2 nodes. Expected GREEN: `3 passed`.

- [ ] Run `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py -q`,
  `.venv/bin/python -m pytest tests/estimators tests/emulator -q`,
  `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check .`, and
  `git diff --check`; record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: add two-point calibration contracts`.

---

### Task 3: Query, Pair, Identity-Estimate, Aggregate-Estimate, and Update Contracts

**Files:**

- Modify: `src/odmr_bench/estimators/two_point_types.py`
- Modify: `tests/two_point_helpers.py`
- Modify: `tests/estimators/test_two_point_types.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Imports introduced now:** `TwoPointQuery`, `TwoPointPartialPair`,
`TwoPointPairResult`, `TwoPointIdentityEstimate`, `TwoPointEstimate`, and
`TwoPointUpdate`.

- [ ] Add `test_two_point_state_record_names_are_public` importing exactly the
  six names.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_two_point_state_record_names_are_public -q`.
  Expected RED: `ImportError` for `TwoPointQuery`.

- [ ] Add the exact frozen/slotted field surfaces and exports.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add parameterized
  `test_query_partial_and_pair_intrinsic_state_matrix`. Use a legal pair and
  one-field `replace` rows to pin index/ID/side/query/observation echoes,
  adjacent sides, arrival order, release selection, and every
  success/step-limited/lost diagnostic combination. Representative assertions:

  ```python
  assert pair.minus_query.side == "minus"
  assert pair.plus_query.side == "plus"
  assert pair.release_sequence_index == second_observation.sequence_index
  with pytest.raises(ValueError):
      replace(pair, lock_state="lost", failure_code=None)
  with pytest.raises(ValueError):
      replace(pair, lock_state="tracking", failure_code="capture_exceeded")
  ```

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_query_partial_and_pair_intrinsic_state_matrix -q`.
  Expected RED: at least one echo/state contradiction constructs.

- [ ] Implement only local query/partial/pair validation.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add
  `test_identity_estimate_aggregate_estimate_and_update_intrinsic_matrix`.
  Pin signed calibration reference, nonnegative pair references/releases/ages,
  active-source rules, history/counter/identity/pending/partial equations,
  stopped boundary, resource counts, seed, and first-/second-side update echoes.
  Use object identity only where the schema requires it; never compare
  `SpectrumFitResult` arrays through generated dataclass equality.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_identity_estimate_aggregate_estimate_and_update_intrinsic_matrix -q`.
  Expected RED: an invalid aggregate/history or update echo constructs.

- [ ] Implement intrinsic-only validation. Do not
  authenticate a reset, pending tracker state, runner, or acquisition here.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add
  `test_estimator_record_graph_contains_no_truth_or_full_resource_type`.
  Recursively inspect `dataclasses.fields` and values; assert no
  `ODMRInstrument`, `SpectralDynamics`, `SpectralSnapshot`,
  `InstrumentObservation`, `ResourceSnapshot`, expected-photon field,
  callback, evaluator, future, or truth reference is reachable. This is a
  post-GREEN structural characterization, not a behavior introduction; no
  production change is permitted unless it exposes a new focused RED.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_types.py::test_estimator_record_graph_contains_no_truth_or_full_resource_type -q`.
  Expected GREEN; a failure requires a focused leakage RED before production
  correction.

- [ ] Refactor only duplicated intrinsic validators; repeat all four Task 3
  nodes. Expected GREEN.

- [ ] Run focused/full/lint/diff gates, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: add two-point state contracts`.

---

### Task 4: Estimator-Safe Atomic Resources and Caller-Asserted Source Factory

**Files:**

- Create: `src/odmr_bench/estimators/two_point_resources.py`
- Create: `src/odmr_bench/estimators/two_point_calibration.py`
- Create: `tests/estimators/test_two_point_resources.py`
- Create: `tests/estimators/test_two_point_calibration_source.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Private interfaces:**

```python
def _zero_public_resources() -> PublicAcquisitionResources: ...
def _advance_public_resources(
    resources: PublicAcquisitionResources,
    observation: EstimatorObservation,
    overhead_s: float,
) -> PublicAcquisitionResources: ...
def _replay_public_resources(
    observations: Sequence[EstimatorObservation],
    overhead_s: float,
) -> PublicAcquisitionResources: ...
```

No estimator module imports `ResourceSnapshot` or defines a full-to-safe
projection.

- [ ] Add
  `test_public_resource_replay_uses_one_arrival_order_atom` with:

  ```python
  observations = tuple(
      make_estimator_observation(
          sequence_index=index,
          integration_time_s=0.005,
          realized_photons=realized,
      )
      for index, realized in enumerate((None, 3, None, 5, 7, None))
  )
  total = _zero_public_resources()
  for observation in observations:
      total = _advance_public_resources(total, observation, 0.001)
  assert total == _replay_public_resources(observations, 0.001)
  assert total.integration_time_s.hex() == "0x1.eb851eb851eb9p-6"
  assert total.integration_time_s != math.fsum([0.005] * 6)
  assert total.virtual_elapsed_time_s.hex() == "0x1.26e978d4fdf3bp-5"
  assert total.realized_photons == 15
  assert total.observations_without_realized_counts == 3
  ```

  Include mixed realized/missing counts.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_resources.py::test_public_resource_replay_uses_one_arrival_order_atom -q`.
  Expected RED: `ModuleNotFoundError: odmr_bench.estimators.two_point_resources`.

- [ ] Implement the three helpers using only
  `PublicAcquisitionResources` and `EstimatorObservation`; each atom uses:

  ```python
  PublicAcquisitionResources(
      observations=resources.observations + 1,
      integration_time_s=resources.integration_time_s + observation.integration_time_s,
      nominal_exposure_photons=(
          resources.nominal_exposure_photons
          + observation.nominal_exposure_photons
      ),
      realized_photons=(
          resources.realized_photons
          + (
              observation.realized_photons
              if observation.realized_photons is not None
              else 0
          )
      ),
      observations_without_realized_counts=(
          resources.observations_without_realized_counts
          + int(observation.realized_photons is None)
      ),
      virtual_elapsed_time_s=(
          resources.virtual_elapsed_time_s
          + (overhead_s + observation.integration_time_s)
      ),
  )
  ```

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_caller_asserted_factory_binds_exact_public_trace` using the
  complete public signature. Assert defensive value snapshots, caller label,
  exact source bounds/endpoints, ID mode, resource replay, and exact
  `CompleteSweep` fit-input provenance.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_calibration_source.py::test_caller_asserted_factory_binds_exact_public_trace -q`.
  Expected RED:
  `ImportError` for
  `bind_caller_asserted_two_point_calibration_source`.

- [ ] Add the exact public signature and enough valid-path construction to make
  the node GREEN. It constructs fit-input facts but does not refit.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add parameterized
  `test_caller_asserted_factory_construction_code_precedence` with adjacent
  defect pairs for codes 1–10. Add
  `test_caller_asserted_epoch_requires_exact_public_midpoint_mean`, requiring
  the exact witness and rejecting both neighboring ULPs, interval endpoints,
  and one ULP beyond each endpoint.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_calibration_source.py::test_caller_asserted_factory_construction_code_precedence tests/estimators/test_two_point_calibration_source.py::test_caller_asserted_epoch_requires_exact_public_midpoint_mean -q`.
  Expected RED: first wrong code or epoch neighbor accepted.

- [ ] Implement exact precedence and epoch association.

- [ ] Repeat the exact two-node command. Expected GREEN: `2 passed`.

- [ ] Refactor only shared source validation; repeat all three Task 4 nodes.
  Expected GREEN.

- [ ] Run focused/full/lint/diff gates, assert
  `rg 'ResourceSnapshot|expected_photons' src/odmr_bench/estimators/two_point_resources.py`
  returns no matches, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: bind caller-asserted two-point sources`.

---

### Task 5: Canonical Local Model, Analytic Derivative, and Calibration Factory

**Files:**

- Modify: `src/odmr_bench/estimators/two_point_calibration.py`
- Create: `tests/estimators/test_two_point_calibration.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

- [ ] Add `test_target_only_model_and_center_derivative_are_canonical`.
  Import private
  `_evaluate_target_only_model(source_fit, target_index, frequency_hz, center_hz) -> float`
  and
  `_target_center_derivative(source_fit, target_index, frequency_hz, center_hz) -> float`.
  Assert baseline-once/source-order subtraction equals the fixture expression,
  only the target center changes, derivative signs are plus/minus on the two
  flanks, and centered numerical difference matches at `rtol=1e-8`,
  `atol=1e-15`.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_calibration.py::test_target_only_model_and_center_derivative_are_canonical -q`.
  Expected RED: import fails for `_evaluate_target_only_model`.

- [ ] Implement only those two private functions with the approved formulas:

  ```python
  u = (frequency_hz - center_hz) / target.fwhm_hz
  profile = (
      target.eta / (1.0 + 4.0 * u * u)
      + (1.0 - target.eta) * math.exp(-4.0 * math.log(2.0) * u * u)
  )
  derivative = -(8.0 * target.amplitude * u / target.fwhm_hz) * (
      target.eta / (1.0 + 4.0 * u * u) ** 2
      + (1.0 - target.eta)
      * math.log(2.0)
      * math.exp(-4.0 * math.log(2.0) * u * u)
  )
  ```

  Evaluate the baseline once and subtract all eight dips in immutable fit tuple
  order, using `profile` only for the target and the stored centers for every
  non-target; do not use `sum` or regroup the subtraction.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_calibrate_two_point_public_signature_exists`:

  ```python
  signature = inspect.signature(calibrate_two_point)
  assert tuple(signature.parameters) == (
      "source", "configuration", "budget_treatment"
  )
  assert signature.parameters["budget_treatment"].kind is inspect.Parameter.KEYWORD_ONLY
  with pytest.raises(NotImplementedError, match="geometry"):
      calibrate_two_point(source, configuration, budget_treatment="conditional_free_precalibration")
  ```

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_calibration.py::test_calibrate_two_point_public_signature_exists -q`.
  Expected RED: import fails for `calibrate_two_point`.

- [ ] Add the exact shell, validate argument exact types, and raise
  `NotImplementedError("calibration geometry not implemented")` only after
  those checks. Export it.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_calibration_builds_analytic_slope_depth_and_all_fixed_cells`.
  Assert every exact `delta/r/max_step`, outer bound, ordered-difference
  internal boundary, allowed inset, calibration center containment, target pair
  depth, zero discriminator, and positive slope. The 1001-point inclusive grid
  must have strictly positive consecutive discriminator values for all eight
  identities.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_calibration.py::test_calibration_builds_analytic_slope_depth_and_all_fixed_cells -q`.
  Expected RED: `NotImplementedError: calibration geometry not implemented`.

- [ ] Implement geometry only, including the numerical derivative cross-check
  and failures through `invalid_calibration_geometry`.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_calibration_geometry_accepts_endpoints_and_rejects_empty_or_one_ulp_outward`
  and
  `test_calibration_budget_and_adjacent_construction_precedence`. Pin unequal
  neighbor widths, exact endpoints, one-ULP outward/empty geometry, required
  keyword binding, asserted/included rejection, verified/conditional
  acceptance, and adjacent code pairs 10/11 and 11/12.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_calibration.py::test_calibration_geometry_accepts_endpoints_and_rejects_empty_or_one_ulp_outward tests/estimators/test_two_point_calibration.py::test_calibration_budget_and_adjacent_construction_precedence -q`.
  Expected RED: at least budget treatment or boundary
  precedence is wrong; the function is present and geometry test no longer
  fails on missing behavior.

- [ ] Implement treatment/precedence without changing model math.

- [ ] Repeat the exact two-node command. Expected GREEN: `2 passed`.

- [ ] Refactor only duplicated geometry validation; repeat all four Task 5
  nodes. Expected GREEN: `4 passed`.

- [ ] Run focused/full/lint/diff gates, inspect formula order and no hidden
  inputs, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: calibrate two-point discriminator cells`.

---

### Task 6: Tracker Construction, Reset, Pair-Boundary Budget, and First Query

**Files:**

- Create: `src/odmr_bench/estimators/two_point_tracker.py`
- Create: `tests/estimators/test_two_point_tracker.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Scope boundary:** no accepted observation, partial pair, second query,
completed pair, next identity, or odd alternation is tested or implemented here.

**Import introduced now:** `CalibratedTwoPointTracker`; no evaluator name is
introduced by this task.

- [ ] Add `test_tracker_constructor_and_pre_reset_surface`. Assert exact
  constructor configuration, empty properties, and state errors from
  `choose_next_query` and `estimate`.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_tracker_constructor_and_pre_reset_surface -q`.
  Expected RED: import fails for `CalibratedTwoPointTracker`.

- [ ] Add the class, final slots, four properties, and pre-reset guards only.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_reset_builds_calibrated_cells_and_mode_specific_resource_start`.
  Pin configuration equality, mapped availability, signed mapped physical
  epoch, included versus conditional sequence age, source resources, charged
  start, zero tracking/history/pending/partial/stop, and required seed
  canonicalization.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_reset_builds_calibrated_cells_and_mode_specific_resource_start -q`.
  Expected RED: missing `reset` or `NotImplementedError`.

- [ ] Add only the valid reset path using direct field assignments in declared
  state order: assign calibration/cells before validating the later
  metadata/resource/ceiling joins. Do not add prospective construction or
  rollback protection yet.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_invalid_reset_is_atomic_and_uses_exact_join_precedence`,
  perturbing calibration/configuration, clock, availability, metadata
  rate/overhead, starting boundary, ceiling below charged start, and seed.
  Snapshot `tracker.configuration`, `calibration`, `pending_query`,
  `pair_history`, and `estimate()` before each call.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_invalid_reset_is_atomic_and_uses_exact_join_precedence -q`.
  Expected RED: a late invalid row leaves at least one directly assigned field
  changed, proving the atomicity test does not pass on arrival.

- [ ] Replace direct assignments with prospective reset construction in locals
  and one commit-last state replacement; preserve the exact error precedence.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_choose_first_query_reserves_two_atomic_charges_and_is_idempotent`.
  Pin query/pair/identity indices all zero, r0, minus, frozen center/frequency,
  expected sequence/endpoint/nominal exposure, exact repeated object/value,
  and no state/resource mutation except pending. Caps one ULP below/equal/above
  each sequential two-atom result prove no `current + 2*cost`.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_choose_first_query_reserves_two_atomic_charges_and_is_idempotent -q`.
  Expected RED: missing `choose_next_query` behavior.

- [ ] Implement pair-boundary two-transition affordability and first query only.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_unaffordable_boundary_stops_atomically_without_partial_pair`.
  Assert first `None`, exact `budget_exhausted`, no pending/partial, and
  repeat `None` with equal estimate.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_unaffordable_boundary_stops_atomically_without_partial_pair -q`.
  Expected RED: the unaffordable branch returns no terminal estimate.

- [ ] Implement only the atomic budget-stop transition.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Refactor only duplicated prospective-state construction; repeat all five
  Task 6 nodes. Expected GREEN: `5 passed`.

- [ ] Run focused/full/lint/diff gates, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: reset and reserve two-point queries`.

---

### Task 7: First-Side Update and Reserved Second Query

**Files:**

- Modify: `src/odmr_bench/estimators/two_point_tracker.py`
- Modify: `tests/estimators/test_two_point_tracker.py`
- Create: `tests/estimators/test_two_point_tracker_atomicity.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

- [ ] Add `test_update_surface_and_pre_reset_guard` with the exact annotated
  signature and a pre-reset `TwoPointObservationValidationError` assertion.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_update_surface_and_pre_reset_guard -q`.
  Expected RED:
  `AttributeError: 'CalibratedTwoPointTracker' object has no attribute 'update'`.

- [ ] Add the exact annotated method with pre-reset guard only.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add parameterized
  `test_update_validation_code_precedence_is_exact_and_atomic` with actual
  `(mutation, code)` rows for exact type, no pending, sequence, frequency,
  integration, endpoint, nominal exposure, and invalid value. Combine adjacent
  defects and assert:

  ```python
  before = tracker.estimate()
  with pytest.raises(TwoPointObservationValidationError) as caught:
      tracker.update(observation)
  assert caught.value.code == expected_code
  assert caught.value.message
  assert tracker.estimate() == before
  ```

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_update_validation_code_precedence_is_exact_and_atomic -q`.
  Expected RED: first valid-looking observation reaches the
  pre-reset-only placeholder or an invalid row is accepted.

- [ ] Implement validation only, in declared order.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_first_side_commits_one_atom_and_exact_partial_pair`. Assert
  returned update has no completed pair; pending clears; exact partial appears;
  one tracking/charged atom and endpoint/index advance; center, history, pair
  count, identity count, and lock state remain unchanged.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_first_side_commits_one_atom_and_exact_partial_pair -q`.
  Expected RED: valid update does not return `TwoPointUpdate`.

- [ ] Implement only the valid first-side path with direct state assignments in
  declared order: clear `self._pending_query` before constructing the partial
  pair, then assign partial/resource/estimate values as each succeeds. Do not
  add prospective construction or rollback protection.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add parameterized
  `test_first_side_construction_faults_roll_back_every_field`, monkeypatching
  the partial-pair, resource, and aggregate-estimate constructors one at a time.
  Per the design's first-side rule, assert
  `partial_pair_construction_failed` for each prospective construction failure,
  a nonempty chained message, and exact equality of configuration, calibration,
  pending query, pair history, and `estimate()` before/after.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker_atomicity.py::test_first_side_construction_faults_roll_back_every_field -q`.
  Expected RED: a late constructor fault leaves at least one directly assigned
  field changed.

- [ ] Replace direct first-side assignments with prospective local construction,
  exact exception-code mapping, and one commit-last state replacement.

- [ ] Repeat the exact node and the first-side behavior node. Expected GREEN:
  all parameter rows and the behavior node pass.

- [ ] Add
  `test_choose_reserved_second_query_without_budget_recheck_or_center_change`.
  Tighten the bound ceiling after reservation only through the reset fixture's
  exact cap, accept first, then assert second query is plus for identity pair
  zero, uses frozen pair center, has query index one, and repeated choose is
  idempotent. Spy on affordability helper to raise if called after first
  acceptance.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_choose_reserved_second_query_without_budget_recheck_or_center_change -q`.
  Expected RED: second query absent or budget rechecked.

- [ ] Implement the reserved second-query transition.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Refactor only the shared pending-query builder; repeat the five Task 7
  nodes. Expected GREEN.

- [ ] Run focused/full/lint/diff gates, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: accept first two-point flank`.

---

### Task 8: Completed Pair Policy, Schedule Advancement, Ages, and Update Atomicity

**Files:**

- Modify: `src/odmr_bench/estimators/two_point_tracker.py`
- Modify: `tests/estimators/test_two_point_tracker.py`
- Modify: `tests/estimators/test_two_point_tracker_atomicity.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

- [ ] Add
  `test_second_side_computes_signed_hertz_update_and_refreshes_zero_step_source`.
  Feed exact public-model flanks. Assert side-stored observations, first-side
  arrival, public midpoint ordered mean, second-arrival release,
  `discriminator == zero_discriminator`, exact zero innovation/request/
  applied step, `tracking`, pair history/counters, and active source refresh.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_second_side_computes_signed_hertz_update_and_refreshes_zero_step_source -q`.
  Expected RED: second valid update raises the first-side-only branch.

- [ ] Implement only the valid second-side success using direct state
  assignments in declared order: clear `self._partial_pair` before constructing
  the pair result, then assign identity/resource/history/estimate values as each
  succeeds. Do not add prospective construction or rollback protection.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add parameterized
  `test_second_side_construction_faults_roll_back_every_field`, monkeypatching
  pair-result, identity-estimate, resource, and aggregate-estimate constructors
  plus one derived-arithmetic overflow and one arbitrary ordinary exception.
  Assert that an exception is raised and configuration, calibration, pending
  query, pair history, and `estimate()` remain value-equal before/after; this
  node owns rollback, not the per-stage type/code matrix.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker_atomicity.py::test_second_side_construction_faults_roll_back_every_field -q`.
  Expected RED: a late failure leaves at least one directly assigned field
  changed.

- [ ] Replace direct second-side assignments with prospective local
  construction and one commit-last state replacement. Temporarily map all
  construction failures to `pair_result_construction_failed`; per-stage code
  mapping belongs to the next RED.

- [ ] Repeat the exact node and the second-side behavior node. Expected GREEN:
  all parameter rows and the behavior node pass.

- [ ] Add parameterized
  `test_second_side_construction_exception_codes_follow_stage_order`. Inject
  pair-result, identity-estimate, resource-state, and aggregate-estimate
  failures and require respectively `pair_result_construction_failed`,
  `identity_estimate_construction_failed`, `resource_construction_failed`, and
  `aggregate_estimate_construction_failed`, each with a nonempty chained
  message and unchanged state.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker_atomicity.py::test_second_side_construction_exception_codes_follow_stage_order -q`.
  Expected RED: the last three rows still report
  `pair_result_construction_failed`.

- [ ] Add the exact per-stage exception-code mapping without changing the
  already-green prospective commit boundary.

- [ ] Repeat the exact node and both rollback nodes. Expected GREEN: all rows
  pass.

- [ ] Add parameterized
  `test_pair_gate_precedence_and_retained_diagnostics` for invalid sum,
  nonfinite derived arithmetic, common strict exceed/equality, capture strict
  exceed/equality, domain, tracking, and step-limited. Assert every specified
  retained-versus-`None` diagnostic and zero-step lost behavior.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_pair_gate_precedence_and_retained_diagnostics -q`.
  Expected RED: at least one gate/state is wrong.

- [ ] Implement the exact six-stage decision tree.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add
  `test_public_schedule_advances_second_side_next_identity_and_odd_alternation`.
  Drive only public `choose/update` calls through 18 pairs. Assert adjacency,
  r0…r7 round robin, next identity after completion, scientific failure still
  advances, and r0's second identity pair arrives plus/minus.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_public_schedule_advances_second_side_next_identity_and_odd_alternation -q`.
  Expected RED: the completed pair does not advance to the next identity or
  flip r0's later first side.

- [ ] Implement schedule advancement after completed success or scientific
  failure.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_domain_endpoints_and_active_age_transitions_are_exact`. Cover exact
  allowed endpoints, `math.nextafter` outward, repeated legal maximum steps
  toward r3/r4, fresh/nonfresh sequence equations, positive public-reference
  age at release, zero release age, failed-pair retention, and first-side aging.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_domain_endpoints_and_active_age_transitions_are_exact -q`.
  Expected RED: boundary or age mismatch.

- [ ] Implement the exact domain and age joins.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_seed_is_observationally_inert_and_tracker_has_no_truth_path`.
  Compare different tracker seeds field-by-field except seed; use a rejecting
  future-observation container and recursive forbidden-type/field inspection.
  This is a post-GREEN seed/truth-isolation characterization; any failure gets
  a new focused RED node before removing leaked state.

- [ ] Run
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py::test_seed_is_observationally_inert_and_tracker_has_no_truth_path -q`.
  Expected GREEN; on failure, stop and introduce a new focused unit RED before
  changing production.

- [ ] Refactor only duplicate pair/state construction; repeat all Task 8 nodes.
  Expected GREEN.

- [ ] Run both tracker files, estimator/emulator suites, full pytest, Ruff, and
  diff check; record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: complete calibrated two-point updates`.

---

### Task 9: Instrument Configuration and Evaluator Primitive Errors/Token

**Files:**

- Modify: `src/odmr_bench/emulator/instrument.py`
- Modify: `tests/emulator/test_instrument.py`
- Create: `src/odmr_bench/evaluation/__init__.py`
- Create: `src/odmr_bench/evaluation/two_point/__init__.py`
- Create: `src/odmr_bench/evaluation/two_point/types.py`
- Create: `src/odmr_bench/evaluation/two_point/provenance.py`
- Create: `tests/evaluation/test_two_point_types.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Imports introduced now:** `TwoPointCalibrationPreflightError`,
`TwoPointRunnerStartError`, `TwoPointRunnerStateError`,
`VerifiedInstrumentRunToken`,
`TwoPointEvaluatorInstrumentConfiguration`, and
`VerifiedCalibrationQueryRequest`, plus the
`ResourceJoinMismatchField` alias needed by Task 10's private comparison
helper. Runner/outcome/resource-record imports remain absent.

- [ ] Add
  `test_instrument_exposes_exact_read_only_acquisition_configuration`.

- [ ] Run
  `.venv/bin/python -m pytest tests/emulator/test_instrument.py::test_instrument_exposes_exact_read_only_acquisition_configuration -q`.
  Expected RED: `AttributeError` for `nominal_photon_rate_hz`.

- [ ] Add both read-only properties returning constructor-canonical fields.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_evaluator_primitive_names_are_public` importing exactly the
  seven names in this task's **Imports introduced now** block.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_types.py::test_evaluator_primitive_names_are_public -q`.
  Expected RED: `ModuleNotFoundError` for the evaluator two-point package.

- [ ] Add exact aliases/errors/two record surfaces and exports.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_verified_token_is_opaque_identity_capability`. Assert public
  construction, copy, deepcopy, pickle, and serialization fail; no value field
  or issuer-bearing repr exists; equality/hash remain object identity with no
  user-defined `__eq__`.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_types.py::test_verified_token_is_opaque_identity_capability -q`.
  Expected RED: direct construction succeeds or token lacks the guarded
  private mint path.

- [ ] Implement private
  `_mint_verified_instrument_run_token(construction_key: object) -> VerifiedInstrumentRunToken`
  and module-private `_TOKEN_CONSTRUCTION_KEY`; export neither.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_evaluator_primitives_validate_intrinsic_matrix` for instrument
  configuration, request timing/index/exposure, and error-code closure.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_types.py::test_evaluator_primitives_validate_intrinsic_matrix -q`.
  Expected RED: the first invalid exact-type/value row constructs.

- [ ] Implement only the local intrinsic matrix.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Refactor only duplicate scalar guards; repeat all four Task 9 nodes.
  Expected GREEN.

- [ ] Run focused/full/lint/diff gates, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: add two-point evaluator primitives`.

---

### Task 10: Evaluator-Private Full Resource Replay, Projection, and Mismatch

**Files:**

- Create: `src/odmr_bench/evaluation/two_point/resource_accounting.py`
- Create: `tests/evaluation/test_two_point_resources.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Private interfaces introduced before any consumer:**

```python
def _zero_full_resources() -> ResourceSnapshot: ...
def _advance_full_resources(
    resources: ResourceSnapshot,
    observation: InstrumentObservation,
    overhead_s: float,
) -> ResourceSnapshot: ...
def _replay_full_resources(
    observations: Sequence[InstrumentObservation],
    overhead_s: float,
) -> ResourceSnapshot: ...
def _project_full_resources(
    resources: ResourceSnapshot,
) -> PublicAcquisitionResources: ...
def _resource_mismatch_fields(
    expected: ResourceSnapshot,
    actual: ResourceSnapshot,
) -> tuple[ResourceJoinMismatchField, ...]: ...
```

- [ ] Add `test_full_resource_helpers_are_evaluator_private_and_atomic`.
  Assert zero/advance/replay, expected photons, mixed counts, six-addition hex,
  safe projection omission of expected photons, and no estimator-module symbol:

  ```python
  assert total.integration_time_s.hex() == "0x1.eb851eb851eb9p-6"
  assert total.virtual_elapsed_time_s.hex() == "0x1.26e978d4fdf3bp-5"
  assert projected.realized_photons == total.realized_photons
  assert not hasattr(projected, "expected_photons")
  ```

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_resources.py::test_full_resource_helpers_are_evaluator_private_and_atomic -q`.
  Expected RED: module/import failure for `resource_accounting`.

- [ ] Implement zero/advance/replay/projection only in the evaluator module and
  do not expose them. The full atom uses `observation.realized_photons`, adds
  zero exactly when that field is `None`, and associates elapsed as
  `old + (overhead_s + observation.integration_time_s)`.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_resource_mismatch_fields_are_exact_complete_and_declaration_ordered`.
  Alter every field independently and together; assert exact tuple order and no
  tolerance.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_resources.py::test_resource_mismatch_fields_are_exact_complete_and_declaration_ordered -q`.
  Expected RED: missing `_resource_mismatch_fields`.

- [ ] Implement fieldwise exact comparison in declaration order.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Refactor only the shared atom construction; repeat both Task 10 nodes.
  Expected GREEN: `2 passed`.

- [ ] Run focused/full/lint/diff gates, inspect no subtraction/regrouping and no
  estimator full-resource import, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: add evaluator resource replay`.

---

### Task 11: Verified Calibration Outcome and Acquisition Record Contracts

**Files:**

- Modify: `src/odmr_bench/evaluation/two_point/types.py`
- Modify: `src/odmr_bench/evaluation/two_point/__init__.py`
- Modify: `tests/evaluation/test_two_point_types.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Imports introduced now:** `VerifiedTwoPointCalibrationSuccess`,
`VerifiedTwoPointCalibrationFailure`,
`VerifiedTwoPointCalibrationOutcome`, `TwoPointTrackingAcquisition`,
`TwoPointResourceJoinUnavailableAcquisition`,
`TwoPointEvaluatorPairTiming`, and `TwoPointInstrumentQueryFailure`.

**Consumes from Task 10:** `_advance_full_resources`,
`_project_full_resources`, and `_resource_mismatch_fields`; record validators
call these reviewed evaluator-private primitives through method-local imports
after `types.py` has initialized, avoiding an import cycle and duplicate
resource arithmetic in `types.py`.

- [ ] Add `test_calibration_outcome_and_acquisition_names_are_public` with only
  the names above.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_types.py::test_calibration_outcome_and_acquisition_names_are_public -q`.
  Expected RED: `ImportError` for `VerifiedTwoPointCalibrationSuccess`.

- [ ] Add the exact field surfaces and exports.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add parameterized
  `test_verified_calibration_outcome_discriminator_matrix`. Pin success versus
  failure status, failed-request/fit/exception/mismatch combinations, aligned
  full/safe/midpoint tuple lengths, exact safe projection equality, and
  aggregate resources `None` exactly for `resource_join_unavailable`.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_types.py::test_verified_calibration_outcome_discriminator_matrix -q`.
  Expected RED: an invalid discriminator row constructs.

- [ ] Implement only local outcome validation.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add
  `test_authenticated_and_unavailable_acquisition_intrinsic_matrix`. Require
  authenticated exact one-atom delta and unavailable nonempty mismatch/no
  delta; pin full-to-safe view equality, query echoes, optional midpoint timing,
  timing release, and instrument failure equal-boundary possibilities.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_types.py::test_authenticated_and_unavailable_acquisition_intrinsic_matrix -q`.
  Expected RED: an invalid delta/mismatch row constructs.

- [ ] Implement only local acquisition validation.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Refactor only duplicate discriminator validation; repeat the three Task
  11 nodes. Expected GREEN.

- [ ] Run focused/full/lint/diff gates, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: add verified calibration outcome contracts`.

---

### Task 12: Evaluator Resource, Abort, Runner-State, and Outcome Contracts

**Files:**

- Modify: `src/odmr_bench/evaluation/two_point/types.py`
- Modify: `src/odmr_bench/evaluation/two_point/__init__.py`
- Modify: `tests/evaluation/test_two_point_types.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Imports introduced now:** `TwoPointEvaluatorResources`,
`TwoPointAbortedRun`, `TwoPointEvaluatorRunnerState`,
`TwoPointRunnerAccepted`, `TwoPointRunnerInstrumentFailure`,
`TwoPointRunnerBudgetStopped`, `TwoPointRunnerExternallyStopped`,
`TwoPointRunnerAborted`, `TwoPointRunnerStepOutcome`, and
`TwoPointRunnerRunOutcome`.

- [ ] Add `test_runner_state_and_outcome_names_are_public`.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_types.py::test_runner_state_and_outcome_names_are_public -q`.
  Expected RED: `ImportError` for `TwoPointEvaluatorResources`.

- [ ] Add the exact surfaces and exports.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add `test_evaluator_resource_and_abort_intrinsic_matrix`. Assert accepted
  charged prefix and final charged are distinct fields, incomplete/unaccepted
  counts independently 0/1, resource-unavailable abort requires unavailable
  acquisition and no exception fields, and authenticated reasons require
  authenticated acquisition/nonempty exception type/exact message.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_types.py::test_evaluator_resource_and_abort_intrinsic_matrix -q`.
  Expected RED: an invalid resource/abort row constructs.

- [ ] Implement intrinsic validation only.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add parameterized
  `test_runner_phase_and_outcome_discriminator_matrix` for all seven phases.
  Pin ready/calibration-failed/calibration-succeeded/tracking/terminal local
  shapes, exact same success object in both calibration fields, terminal abort
  only in aborted, instrument failure placement, timing/history cardinality,
  and step/run outcome kind/state compatibility.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_types.py::test_runner_phase_and_outcome_discriminator_matrix -q`.
  Expected RED: at least one invalid phase combination constructs.

- [ ] Implement the local phase/discriminator matrix.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Refactor only duplicate state guards; repeat all three Task 12 nodes.
  Expected GREEN.

- [ ] Run focused/full/lint/diff gates, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: add two-point runner state contracts`.

---

### Task 13: Runner Binding and Verified Calibration Acquisition

**Files:**

- Modify: `src/odmr_bench/estimators/two_point_calibration.py`
- Create: `src/odmr_bench/evaluation/two_point/calibration.py`
- Create: `src/odmr_bench/evaluation/two_point/runner.py`
- Modify: `src/odmr_bench/evaluation/two_point/provenance.py`
- Modify: `src/odmr_bench/evaluation/two_point/__init__.py`
- Create: `tests/evaluation/test_two_point_calibration.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Exact private producer/consumer interfaces:**

```python
# estimators/two_point_calibration.py; imported only by evaluator calibration
_VERIFIED_SOURCE_CONSTRUCTION_KEY: object
def _bind_verified_two_point_calibration_source(
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
    construction_key: object,
) -> TwoPointCalibrationSource: ...

# evaluation/two_point/provenance.py
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .runner import TwoPointEvaluatorRunner

@dataclass(frozen=True, slots=True)
class _RunTokenBinding:
    issuer_runner: "TwoPointEvaluatorRunner"
    instrument: ODMRInstrument
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration
    success: VerifiedTwoPointCalibrationSuccess | None
    source: TwoPointCalibrationSource | None

def _register_run_token(
    token: VerifiedInstrumentRunToken,
    issuer_runner: "TwoPointEvaluatorRunner",
    instrument: ODMRInstrument,
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration,
) -> None: ...
def _bind_run_token_success(
    token: VerifiedInstrumentRunToken,
    issuer_runner: "TwoPointEvaluatorRunner",
    instrument: ODMRInstrument,
    success: VerifiedTwoPointCalibrationSuccess,
) -> None: ...
def _lookup_run_token_binding(
    token: VerifiedInstrumentRunToken,
) -> _RunTokenBinding | None: ...
```

`runner.bind` produces/registers the token. Verified acquisition consumes the
construction key, creates the source, creates the success, then binds that exact
success/source to the registry. Start/resource code only looks up; it never
re-registers or upgrades copied values.

- [ ] Add `test_runner_bind_derives_clean_configuration_and_registers_identity`.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_calibration.py::test_runner_bind_derives_clean_configuration_and_registers_identity -q`.
  Expected RED: `ImportError` for `TwoPointEvaluatorRunner`.

- [ ] Add class/bind/state with clean boundary and exact instrument
  configuration, token mint, runner construction, then registry registration.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add parameterized
  `test_bind_and_calibration_preflight_precedence_has_zero_calls`. Pin bind
  type/value/unclean order and acquisition phase/type/value/grid/fit-identity/
  clock order with instrument query and fit sentinels that raise if called.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_calibration.py::test_bind_and_calibration_preflight_precedence_has_zero_calls -q`.
  Expected RED: the first unhandled row reaches a sentinel or returns the wrong
  code.

- [ ] Implement the complete defensive preflight before every collaborator
  call. Add the exact public `acquire_verified_calibration` signature and end
  the valid-preflight path with
  `NotImplementedError("verified calibration acquisition not implemented")`;
  do not acquire an observation yet.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add `test_verified_calibration_success_uses_safe_fit_and_identity_binding`.
  Spy on `fit_spectrum`; assert exact `CompleteSweep`, `initial_guess=None`,
  normalized rule, actual midpoint/endpoint/resource witnesses, exact full/safe
  projections, source fields, and registry identities:

  ```python
  binding = _lookup_run_token_binding(outcome.run_token)
  assert binding is not None
  assert binding.issuer_runner is runner
  assert binding.instrument is instrument
  assert binding.success is outcome
  assert binding.source is outcome.source
  assert runner.state.calibration_outcome is outcome
  assert runner.state.verified_calibration is outcome
  ```

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_calibration.py::test_verified_calibration_success_uses_safe_fit_and_identity_binding -q`.
  Expected RED: `acquire_verified_calibration` is absent or raises its
  feature-missing branch.

- [ ] Implement the success loop using Task 10 helpers and the exact private
  source seam.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_verified_source_identity_matrix_rejects_public_mint_and_copies`.
  Require wrong construction key failure, public/direct verified failure,
  original source/outcome/token registry lookup success, and
  `replace(outcome.source)` absent from the binding.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_calibration.py::test_verified_source_identity_matrix_rejects_public_mint_and_copies -q`.
  Expected RED: a copied source or wrong construction key authenticates.

- [ ] Enforce the construction-key and exact identity checks.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add named midpoint-slot nodes:
  `test_calibration_timing_or_integration_mismatch_sets_final_midpoint_none`;
  `test_calibration_frequency_or_sequence_mismatch_retains_exact_midpoint`;
  and
  `test_calibration_resource_corruption_midpoint_slots_follow_timing_only`.
  The last parameterizes nominal-only and expected-only corruption to exact
  midpoint, integration/timing corruption to `None`, and preserves every
  earlier exact slot.

- [ ] Run each literal command separately:

  ```bash
  .venv/bin/python -m pytest tests/evaluation/test_two_point_calibration.py::test_calibration_timing_or_integration_mismatch_sets_final_midpoint_none -q
  .venv/bin/python -m pytest tests/evaluation/test_two_point_calibration.py::test_calibration_frequency_or_sequence_mismatch_retains_exact_midpoint -q
  .venv/bin/python -m pytest tests/evaluation/test_two_point_calibration.py::test_calibration_resource_corruption_midpoint_slots_follow_timing_only -q
  ```

  Expected RED: the final slot uses one generic failure rule.

- [ ] Implement timing checks independently from frequency/sequence/nominal/
  expected checks.

- [ ] Repeat each literal command. Expected GREEN: `1 passed` for each.

- [ ] Add
  `test_verified_calibration_preserves_every_typed_post_start_failure`
  covering instrument exception after two successes, structured fit failure,
  fitting exception, source-binding failure, and resource-unavailable alias
  order/None aggregates.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_calibration.py::test_verified_calibration_preserves_every_typed_post_start_failure -q`.
  Expected RED: the first unimplemented causal failure propagates or loses its
  accepted prefix.

- [ ] Implement one causal branch at a time in parameter order, rerunning the
  exact node after each branch. Expected final GREEN: all parameter rows pass.

- [ ] Refactor only shared acquisition finalization; repeat all eight Task 13
  nodes. Expected GREEN.

- [ ] Run focused/full/lint/diff gates, inspect safe-only fitting, identity,
  midpoint alignment, resource-first classification, and terminal calibration
  failure, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: acquire verified two-point calibration`.

---

### Task 14: Runner Start and Zero-Trace Tracking State

**Files:**

- Modify: `src/odmr_bench/evaluation/two_point/runner.py`
- Create: `tests/evaluation/test_two_point_runner.py`
- Modify: `src/odmr_bench/evaluation/two_point/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

- [ ] Add `test_conditional_start_on_clean_other_runner_preserves_identity_and_zero_trace`.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_conditional_start_on_clean_other_runner_preserves_identity_and_zero_trace -q`.
  Expected RED: `AttributeError` for `start_tracking`.

- [ ] Add the exact method signature and conditional-other-runner valid path;
  retain exact verified outcome/source/token identities, call reset once, and
  perform no query.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_conditional_start_on_same_calibration_succeeded_runner_is_legal`.
  Use the same runner's success but conditional treatment; assert
  `calibration_outcome is verified_calibration is success`, zero normal trace,
  and calibration resources uncharged.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_conditional_start_on_same_calibration_succeeded_runner_is_legal -q`.
  Expected RED: implementation accepts only included treatment on the issuing
  runner.

- [ ] Add this same-runner conditional legal branch.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_included_start_requires_exact_original_capabilities_and_boundary`.
  Assert original source/outcome/token/instrument, shared clock, continuous
  calibration-after/tracking-before, and exact three-way rate/overhead equality.
  Perturb copies individually.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_included_start_requires_exact_original_capabilities_and_boundary -q`.
  Expected RED: at least one copied capability or discontinuous boundary is
  accepted.

- [ ] Implement the included-treatment identity and boundary joins.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add parameterized
  `test_start_error_precedence_and_atomicity` for all eight exact start codes,
  with tracker reset/instrument query sentinels and state snapshots.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_start_error_precedence_and_atomicity -q`.
  Expected RED: the first unhandled row reaches a no-call sentinel, returns the
  wrong code, or mutates state.

- [ ] Implement precedence through chained `tracker_reset_failed`.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Refactor only the start preflight; repeat all four Task 14 nodes.
  Expected GREEN.

- [ ] Run focused/full/lint/diff gates, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: start two-point evaluator tracking`.

---

### Task 15: Accepted Tracking Step, Pair Timing, and Retriable Instrument Failure

**Files:**

- Modify: `src/odmr_bench/evaluation/two_point/runner.py`
- Modify: `tests/evaluation/test_two_point_runner.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

- [ ] Add `test_step_accepts_first_and_second_sides_and_records_pair_timing`.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_step_accepts_first_and_second_sides_and_records_pair_timing -q`.
  Expected RED: `AttributeError` for `step`.

- [ ] Add the exact method signature and accepted path only. Snapshot after
  query issuance, compute actual midpoint before query, authenticate one atom,
  call update once, append normal trace, and append timing only for a completed
  pair.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_pair_three_truth_and_public_references_use_distinct_associations`.
  Assert exact hexadecimal witnesses and release fields.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_pair_three_truth_and_public_references_use_distinct_associations -q`.
  Expected RED: regrouped midpoint logic produces the wrong hexadecimal
  witness.

- [ ] Implement both ordered means without substituting the public midpoint
  for evaluator truth.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_instrument_exception_preserves_identical_pending_query_and_can_retry`.
  Assert equal instrument before/after, unchanged normal trace/tracker estimate,
  stored failure, tracking phase, explicit later retry uses the identical
  pending query, and acceptance clears failure.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_instrument_exception_preserves_identical_pending_query_and_can_retry -q`.
  Expected RED: the exception propagates or retry does not reuse the identical
  pending query.

- [ ] Implement only the ordinary query-exception/retry branch.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Refactor only accepted-acquisition construction; repeat all three Task 15
  nodes. Expected GREEN.

- [ ] Run focused/full/lint/diff gates, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: accept two-point evaluator observations`.

---

### Task 16: Public Evaluator Resource Builder and Optional Authenticated Abort Atom

**Files:**

- Modify: `src/odmr_bench/evaluation/two_point/resource_accounting.py`
- Modify: `src/odmr_bench/evaluation/two_point/__init__.py`
- Modify: `tests/evaluation/test_two_point_resources.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Exact private assembly seam, introduced before terminal step consumes it:**

```python
def _build_two_point_evaluator_resources_from_context(
    runner: TwoPointEvaluatorRunner,
    *,
    authenticated_unaccepted: TwoPointTrackingAcquisition | None,
) -> TwoPointEvaluatorResources: ...
```

The public `build_two_point_evaluator_resources(runner)` derives the optional
atom only from authenticated runner terminal state. It returns `None` for an
unavailable abort.

- [ ] Add
  `test_public_builder_zero_trace_and_accepted_trace_join_exactly`.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_resources.py::test_public_builder_zero_trace_and_accepted_trace_join_exactly -q`.
  Expected RED: `ImportError` for `build_two_point_evaluator_resources`.

- [ ] Add the exact public signature and normal zero/accepted trace assembly
  using started/accepted runners from Tasks 14–15.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_builder_rejects_nonoriginal_or_inconsistent_context`, covering caller
  source, copied runner/outcome/token/source, extra/missing/duplicate/reordered/
  safe-view-mismatched observation, and each boundary field.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_resources.py::test_builder_rejects_nonoriginal_or_inconsistent_context -q`.
  Expected RED: at least one copied capability or inconsistent trace is
  accepted.

- [ ] Implement contextual authentication with exact registry identities.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add
  `test_accepted_charged_prefix_uses_arrival_atoms_in_both_treatments`. Assert
  exact projection to estimate charged resources without subtotal addition and
  cover mixed counts/scientific lost/pending/incomplete first side.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_resources.py::test_accepted_charged_prefix_uses_arrival_atoms_in_both_treatments -q`.
  Expected RED: accepted charged resources are regrouped or omit an arrival
  atom.

- [ ] Implement the accepted prefix as one arrival-order replay.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add
  `test_private_resource_assembly_continues_with_one_authenticated_unaccepted_atom`.
  Call the private seam on first- and second-side contexts. Assert accepted
  tuple excludes the atom, final tracking/charged includes it exactly,
  accepted charged projection joins estimate, and incomplete/unaccepted counts
  independently equal one.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_resources.py::test_private_resource_assembly_continues_with_one_authenticated_unaccepted_atom -q`.
  Expected RED: the private seam lacks the optional atom or regrouping changes
  the exact result.

- [ ] Implement optional atom continuation after the accepted prefix.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add
  `test_public_builder_returns_none_for_unavailable_abort_without_fabrication`
  using an intrinsically valid unavailable terminal fixture.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_resources.py::test_public_builder_returns_none_for_unavailable_abort_without_fabrication -q`.
  Expected RED: builder tries aggregate replay.

- [ ] Add the `None` branch with boundary/raw validation.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Refactor only resource context validation; repeat all five Task 16 nodes.
  Expected GREEN.

- [ ] Run focused/full/lint/diff gates, inspect original-runner authentication,
  accepted prefix, no subtraction/regrouping, and evaluator-only expected
  photons, record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `feat: join two-point evaluator resources`.

---

### Task 17: Budget/External Stops, Authenticated/Unavailable Aborts, Run Loop, and Phase Matrix

**Files:**

- Modify: `src/odmr_bench/evaluation/two_point/runner.py`
- Modify: `tests/evaluation/test_two_point_runner.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

- [ ] Add `test_budget_stop_builds_resources_without_query`.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_budget_stop_builds_resources_without_query -q`.
  Expected RED: `step` lacks the `choose_next_query() is None` branch.

- [ ] Implement the budget terminal outcome through the already-green public
  builder.

- [ ] Repeat the exact node. Expected GREEN: `1 passed`.

- [ ] Add parameterized
  `test_authenticated_update_exceptions_abort_with_equal_pending_snapshots`
  for validation error, update-construction error, and arbitrary
  `Exception`, on first and second side and on fresh/retry pending queries.
  Assert exact reason class mapping, one authenticated unaccepted atom, mandatory
  resources from Task 16, no normal append, terminal state, and no later call.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_authenticated_update_exceptions_abort_with_equal_pending_snapshots -q`.
  Expected RED: update exception propagates or loses acquisition.

- [ ] Implement authenticated abort using the private optional-atom assembly.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add
  `test_resource_join_unavailable_aborts_without_update_or_aggregate`.
  Parameterize raw integration, nominal, and expected corruption; assert ordered
  mismatch fields, raw full/safe/boundaries, timing-derived midpoint rule,
  `resources is None`, no tracker update, equal snapshots, and terminal state.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_resource_join_unavailable_aborts_without_update_or_aggregate -q`.
  Expected RED: raw mismatch reaches tracker update or fabricates resources.

- [ ] Implement unavailable abort using Task 10 primitives and Task 16 public
  `None` behavior.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add
  `test_external_stop_preserves_boundary_partial_pending_and_failure_states`.
  Cover before pair, accepted first, pending second, and immediately after
  instrument failure.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_external_stop_preserves_boundary_partial_pending_and_failure_states -q`.
  Expected RED: `AttributeError` for `stop_external`.

- [ ] Add the exact method with no-query/no-update behavior, builder result,
  and terminal state.

- [ ] Repeat the exact node. Expected GREEN: all parameter rows pass.

- [ ] Add three exact `run_until_event` nodes:
  `test_run_until_event_loops_across_accepted_steps_to_budget_stop`,
  `test_run_until_event_loops_across_accepted_steps_to_first_abort`, and
  `test_run_until_event_returns_first_instrument_failure_without_retry`.

- [ ] Run each literal command separately:

  ```bash
  .venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_run_until_event_loops_across_accepted_steps_to_budget_stop -q
  .venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_run_until_event_loops_across_accepted_steps_to_first_abort -q
  .venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_run_until_event_returns_first_instrument_failure_without_retry -q
  ```

  Expected RED: missing method or wrong loop exit.

- [ ] Implement the exact accepted-only loop and three terminal exits.

- [ ] Repeat each literal command. Expected GREEN: `1 passed` for each.

- [ ] Add parameterized
  `test_operation_by_phase_matrix_rejects_before_any_call` over operations
  `acquire_verified_calibration`, `start_tracking`, `step`,
  `run_until_event`, and `stop_external` and phases `ready`,
  `calibration_succeeded`, `calibration_failed`, `tracking`,
  `budget_stopped`, `externally_stopped`, and `aborted`. Encode only
  acquire-ready, start-ready-conditional, start-calibration-succeeded, and
  step/run/stop-tracking as legal. For every illegal cell, instrument-query,
  tracker choose/reset/update, fit, builder, and registry sentinels raise if
  called; state remains equal.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_operation_by_phase_matrix_rejects_before_any_call -q`.
  Expected RED: at least one illegal cell reaches a no-call sentinel or one
  legal cell is rejected.

- [ ] Implement all phase guards before collaborator calls.

- [ ] Repeat the exact node. Expected GREEN: all operation×phase rows pass.

- [ ] Add `test_base_exception_is_not_converted_to_typed_abort`. This is a
  post-GREEN characterization of the ordinary-`Exception` boundary; if it
  fails, add a distinct focused RED before narrowing catches.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_runner.py::test_base_exception_is_not_converted_to_typed_abort -q`.
  Expected GREEN; any failure becomes a distinct focused RED/fix/GREEN cycle.

- [ ] Refactor only terminal-outcome construction; repeat all Task 17 nodes.
  Expected GREEN.

- [ ] Run focused/full/lint/diff gates with zero skips and record evidence.

- [ ] Stage only the files listed for Task 17 and commit with message
  `feat: complete two-point runner terminal protocol`.

---

### Task 18: Closed Generated Scientific Acceptance and Truth Isolation

**Files:**

- Create: `tests/evaluation/test_two_point_regressions.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Classification:** these are acceptance/characterization checks and are
expected to PASS against Tasks 1–17. They are not mislabeled REDs. If any check
fails, first add one new focused unit RED to the owning earlier test file,
observe its specific behavioral failure, make the minimal production change,
run that unit GREEN, then rerun the acceptance node. Commit each discovered gap
as a new atomic `fix:` commit before the Task 18 acceptance commit.

**Exact fixture assertions:** use `stage63-calibration-v1`, source clock
`stage63-calibration-clock-v1`, tracker clock `stage63-tracking-clock-v1`,
tracker/source/tracking seeds `20260904`/`20260903`/`20260905`, rate `5.0e8`,
overhead `0.001`, and integration `0.005`. The source grid is
`2.740e9 + k * 62_500.0` for `k=0..4480`; the eight `(center, FWHM, amplitude,
eta)` tuples are `(2.805e9,2.5e6,.018,.35)`, `(2.825e9,2.7e6,.021,.40)`,
`(2.845e9,2.9e6,.023,.45)`, `(2.865e9,3.1e6,.025,.50)`,
`(2.875e9,3.1e6,.024,.50)`, `(2.895e9,2.9e6,.022,.45)`,
`(2.915e9,2.7e6,.020,.40)`, `(2.935e9,2.5e6,.017,.35)`. The baseline is
intercept `1.0`, reference `2.870e9`, slope `1.0e-11`, and quadratic `0.0`.
Use exactly:

```python
FitConfiguration(
    model_kind="pseudo_voigt", baseline_degree=1,
    resonance_ids=("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"),
    min_fwhm_hz=2.0e5, max_fwhm_hz=8.0e6, max_amplitude=0.08,
    min_resolved_amplitude=1.0e-4, min_center_separation_hz=1.0e6,
    savgol_window=11, savgol_polyorder=2, relative_prominence=0.01,
    allow_fallback=False, max_nfev=4000, rank_rtol=1.0e-10,
    min_baseline_sse_improvement=1.0e-4, min_amplitude_significance=5.0,
)
```

Assert source
first/last endpoints `0x1.89374bc6a7efap-8`/
`0x1.ae2d0e560425fp+4`, actual first/last midpoints
`0x1.cac083126e979p-9`/`0x1.ae22d0e5604efp+4`, actual epoch
`0x1.ae3126e978e27p+3`, public last midpoint
`0x1.ae22d0e5604eep+4`, public endpoint epoch
`0x1.ae3126e978e26p+3`, mapped actual/public epochs
`-0x1.ae28f5c28f697p+3`/`-0x1.ae28f5c28f698p+3`, observations `4481`,
integration `0x1.667ae147ae117p+4`, nominal exposure `11_202_500_000`,
realized/missing `0/4481`, and elapsed `0x1.ae2d0e5604269p+4`.

The static/Poisson/drift/loss caps are respectively observations
`32/64/480/48`, integration `0x1.47ae147ae147dp-3`/
`0x1.47ae147ae147ep-2`/`0x1.33333333332f2p+1`/
`0x1.eb851eb851ebdp-3`, nominal `80_000_000`/`160_000_000`/
`1_200_000_000`/`120_000_000`, elapsed `0x1.89374bc6a7efdp-3`/
`0x1.89374bc6a7efep-2`/`0x1.70a3d70a3d6c6p+1`/
`0x1.26e978d4fdf3ep-2`, and instrument endpoints
`0x1.89374bc6a7efep-3`/`0x1.89374bc6a7effp-2`/
`0x1.70a3d70a3d673p+1`/`0x1.26e978d4fdf3fp-2`. Loss affects only r3
from `0x1.15810624dd2f4p-3` through `0x1.21cac083126ecp-3` inclusive at
observations 22/23 (pair 11, identity-pair 1, plus/minus); recovery is pair 19
at observations 38/39. Included same-run uses totals `4483`,
`0x1.66a3d70a3d6d9p+4`, `11_207_500_000`,
`0x1.ae5e353f7cfb9p+4`, and endpoint `0x1.ae5e353f7cfafp+4`.

- [ ] Add the exact 4481-point `stage63_eight_line_v1` source fixture,
  published model/configuration/seeds/rate/overhead/integration, and all source
  endpoint/midpoint/epoch/resource hexadecimal witnesses. Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_regressions.py::test_verified_source_fixture_matches_closed_witnesses -q`.
  Expected: PASS.

- [ ] Add
  `test_static_noiseless_two_cycles_match_exact_schedule_resources_and_zero_steps`.
  Require 32 observations, exact caps, refreshed zero-step sources, and next
  budget stop.

- [ ] Run twice:
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_regressions.py::test_static_noiseless_two_cycles_match_exact_schedule_resources_and_zero_steps -q`.
  Expected: PASS twice.

- [ ] Add
  `test_static_poisson_four_cycles_are_seed_reproducible`. Require all
  query/count/resource/diagnostic/policy/estimate fields match for seed
  `20260905` and at least one of 64 counts differs for a different instrument
  seed; make no accuracy assertion.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_regressions.py::test_static_poisson_four_cycles_are_seed_reproducible -q`.
  Expected: PASS.

- [ ] Add
  `test_common_linear_drift_thirty_cycles_tracks_declared_fixture`. Construct
  two separate identically configured dynamics instances:

  ```python
  acquisition_dynamics = LinearCenterDrift(initial_snapshot, 5.0e5)
  truth_oracle = RejectingPostReleaseTruthOracle(
      LinearCenterDrift(initial_snapshot, 5.0e5)
  )
  ```

  The instrument owns `acquisition_dynamics`; the fixture owns
  `truth_oracle`. Call only the oracle once per completed pair after release
  at the evaluator actual reference. Assert sign/domain/no loss/`0.05*w`
  bounds.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_regressions.py::test_common_linear_drift_thirty_cycles_tracks_declared_fixture -q`.
  Expected: PASS.

- [ ] Add
  `test_contrast_loss_pair_eleven_and_pair_nineteen_recovery_are_exact` with
  the specified hex interval and common-mode assertions.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_regressions.py::test_contrast_loss_pair_eleven_and_pair_nineteen_recovery_are_exact -q`.
  Expected: PASS.

- [ ] Add
  `test_included_same_run_one_pair_charges_source_once_without_regrouping`.
  Require exact total ceilings/final endpoint and next budget stop.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_regressions.py::test_included_same_run_one_pair_charges_source_once_without_regrouping -q`.
  Expected: PASS.

- [ ] Add
  `test_truth_oracle_rejects_public_reference_pre_release_and_duplicate_lookup`.
  Require pair-3 distinct hex values, post-release-only call, exactly one call
  per scored pair, and no truth/full/future object reachable from tracker.

- [ ] Run
  `.venv/bin/python -m pytest tests/evaluation/test_two_point_regressions.py::test_truth_oracle_rejects_public_reference_pre_release_and_duplicate_lookup -q`.
  Expected: PASS.

- [ ] Run the whole acceptance file twice, then dynamics/models and all
  estimator/evaluator/emulator tests, full pytest, Ruff, and diff check. Record
  results.

- [ ] Stage only the acceptance file, `PROJECT_STATE.md`, and `CHANGELOG.md`;
  commit with message `test: close two-point tracker regressions`.

---

### Task 19: Public Documentation, Example, Exports, Build, and Wheel Smoke

**Files:**

- Create: `examples/track_two_point_centers.py`
- Modify: `docs/estimators.md`
- Modify: `README.md`
- Modify: `tests/test_package.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

The initial import node is a post-GREEN export characterization and changes no
production code. The example and documentation nodes that follow are distinct
strict RED → implementation → identical-node GREEN cycles.

- [ ] Add
  `test_two_point_public_surfaces_import_from_installed_modules`, explicitly
  importing every estimator name from Tasks 1–8 and every evaluator name from
  Tasks 9–17.

- [ ] Run
  `.venv/bin/python -m pytest tests/test_package.py::test_two_point_public_surfaces_import_from_installed_modules -q`.
  Expected GREEN in source checkout; this characterizes the final export set
  before wheel smoke.

- [ ] Add `test_two_point_example_runs_out_of_tree`. From `tmp_path`, subprocess
  the nonexistent example and assert exit zero, header
  `Synthetic calibrated two-point diagnostics`, the literal treatment
  `conditional_free_precalibration`, finite policy/resource/timing columns,
  and absence of truth-error, expected-photon, comparison, or superiority
  output.

- [ ] Run
  `.venv/bin/python -m pytest tests/test_package.py::test_two_point_example_runs_out_of_tree -q`.
  Expected RED: subprocess exits 2 because
  `examples/track_two_point_centers.py` is absent.

- [ ] Add the download-free conditional-precalibration example using only
  public APIs. Print no truth/error/expected-photon/comparison output.

- [ ] Repeat the exact example node. Expected GREEN: `1 passed`.

- [ ] Add `test_two_point_guidance_terms`. Read `docs/estimators.md` and
  `README.md`; require explicit text for caller-asserted versus verified
  provenance, both mandatory budget treatments, fixed cells, side order and
  discriminator sign, policy-only lock/common-mode state, zero-step refresh,
  partial/unaccepted resources, token continuity, public versus actual
  midpoint and release/age semantics, ordinary terminal aborts, inert tracker
  seed, the example link, and the Stage 6.5 matched-budget-superiority boundary.

- [ ] Run
  `.venv/bin/python -m pytest tests/test_package.py::test_two_point_guidance_terms -q`.
  Expected RED: the first required Stage 6.3 guidance term or example link is
  absent.

- [ ] Add researcher guidance for provenance, mandatory treatments, fixed cells,
  side order/sign, gates/policy state, common-mode confounding, zero-step
  refresh, partial/unaccepted resources, token continuity, public/actual/release
  timing and ages, terminal aborts, seed, and explicit no Stage 6.5 superiority.
  Update README example link.

- [ ] Repeat the exact documentation node. Expected GREEN: `1 passed`.

- [ ] Run this exact fail-fast build/isolated-wheel smoke from the repository
  root; the temporary working directory and `-I` prevent source-tree fallback:

  ```bash
  set -eu
  STAGE63_DIST=$(mktemp -d)
  STAGE63_VENV=$(mktemp -d)
  STAGE63_RUN=$(mktemp -d)
  .venv/bin/python -m build --outdir "$STAGE63_DIST"
  test "$(find "$STAGE63_DIST" -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')" = 1
  test "$(find "$STAGE63_DIST" -maxdepth 1 -name '*.tar.gz' | wc -l | tr -d ' ')" = 1
  python3 -m venv "$STAGE63_VENV"
  "$STAGE63_VENV/bin/python" -m pip install "$STAGE63_DIST"/*.whl
  cd "$STAGE63_RUN"
  "$STAGE63_VENV/bin/python" -I - <<'PY'
  import odmr_bench.estimators as estimators
  import odmr_bench.evaluation.two_point as evaluation

  estimator_names = {
      "CalibrationBudgetTreatment", "CalibrationIdentityMode",
      "CalibrationSourceProvenance", "CalibratedTwoPointTracker",
      "ClockMappingKind", "NormalizedFluorescenceProvenance", "PairSide",
      "PublicAcquisitionResources", "TwoPointBudgetCeiling",
      "TwoPointCalibration", "TwoPointCalibrationConstructionCode",
      "TwoPointCalibrationConstructionError", "TwoPointCalibrationSource",
      "TwoPointClockMapping", "TwoPointEstimate", "TwoPointFailureCode",
      "TwoPointIdentityBinding", "TwoPointIdentityCalibration",
      "TwoPointIdentityEstimate", "TwoPointLockState",
      "TwoPointObservationValidationCode", "TwoPointObservationValidationError",
      "TwoPointPairResult", "TwoPointPartialPair", "TwoPointQuery",
      "TwoPointRunMetadata", "TwoPointStopReason",
      "TwoPointTrackerConfiguration", "TwoPointUpdate",
      "TwoPointUpdateConstructionCode", "TwoPointUpdateConstructionError",
      "bind_caller_asserted_two_point_calibration_source", "calibrate_two_point",
  }
  evaluator_names = {
      "ResourceJoinMismatchField", "TwoPointAbortReason", "TwoPointAbortedRun",
      "TwoPointCalibrationPreflightError", "TwoPointEvaluatorInstrumentConfiguration",
      "TwoPointEvaluatorPairTiming", "TwoPointEvaluatorResources",
      "TwoPointEvaluatorRunner", "TwoPointEvaluatorRunnerState",
      "TwoPointInstrumentQueryFailure", "TwoPointResourceJoinUnavailableAcquisition",
      "TwoPointRunnerAborted", "TwoPointRunnerAccepted", "TwoPointRunnerBudgetStopped",
      "TwoPointRunnerExternallyStopped", "TwoPointRunnerInstrumentFailure",
      "TwoPointRunnerPhase", "TwoPointRunnerRunOutcome",
      "TwoPointRunnerStartError", "TwoPointRunnerStartFailureCode",
      "TwoPointRunnerStateError", "TwoPointRunnerStepOutcome",
      "TwoPointTrackingAcquisition", "VerifiedCalibrationFailureCode",
      "VerifiedCalibrationPreflightCode", "VerifiedCalibrationQueryRequest",
      "VerifiedInstrumentRunToken", "VerifiedTwoPointCalibrationFailure",
      "VerifiedTwoPointCalibrationOutcome", "VerifiedTwoPointCalibrationSuccess",
      "build_two_point_evaluator_resources",
  }
  assert not (estimator_names - set(dir(estimators)))
  assert not (evaluator_names - set(dir(evaluation)))
  PY
  "$STAGE63_VENV/bin/python" -I \
    "$OLDPWD/examples/track_two_point_centers.py"
  ```

  Expected: one exact sdist, one exact wheel, all enumerated imports succeed,
  and the example exits zero outside the source tree.

- [ ] Run full pytest, Ruff, and diff check; inspect claims/exports/artifacts,
  record evidence.

- [ ] Stage only the files listed for this task and commit with message
  `docs: document calibrated two-point tracking`.

---

### Task 20: Integrated Scientific and Software Review, Fix Waves, and Closeout

**Files:** review-driven focused tests/production files only for new fix-wave
commits; after clean re-review modify `PROJECT_STATE.md` and `CHANGELOG.md`.

- [ ] Read `IMPLEMENTATION_BASE` from
  `.superpowers/sdd/progress.md`; do not recompute it. Generate the final code
  review package for exact range `$IMPLEMENTATION_BASE..HEAD`. If a wider
  design+plan+implementation audit is explicitly requested, record a separately
  named `AUDIT_BASE`; never substitute it for `IMPLEMENTATION_BASE`.

- [ ] Run independent scientific review of units, canonical subtraction/
  derivative/slope, cells/envelope, discriminator sign/gates, epochs/
  availability, midpoint/release/truth ordering, resources, exact fixtures, and
  Stage 6.5 boundary.

- [ ] Run independent software/protocol review of every literal/intrinsic/
  contextual owner, copy-versus-identity rule, registry capability, phase
  matrix, no-call guards, no-future boundary, reservation/update atomicity,
  ordinary abort, unavailable join, accepted prefix/optional atom, exports,
  wheel, and repository hygiene.

- [ ] One fix owner handles the complete Critical/Important set from both
  reviews as one review wave. For each finding: write one focused unit RED with
  an exact node, observe it fail for the reported behavior, make the minimal
  production fix, run the node GREEN, then run affected/full gates. Commit the
  wave as a new atomic `fix:` commit. Do not amend Tasks 1–19. Rebuild the
  exact `$IMPLEMENTATION_BASE..HEAD` package and re-review until zero Critical
  and Important findings remain.

- [ ] Run twice where deterministic:

  ```bash
  .venv/bin/python -m pytest tests/estimators tests/evaluation tests/emulator -q
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check .
  .venv/bin/python -m build --outdir "$(mktemp -d)/dist"
  git diff --check
  ```

  Repeat the Task 19 isolated-wheel smoke. Expected: zero skips, clean tests,
  lint, build, installed imports/example, and no diff-check output.

- [ ] Only after clean re-review, update state/changelog with exact counts,
  review range/fix waves, build/wheel evidence, and Stage 6.3 completion.
  Preserve the no-Stage-6.5 claim boundary.

- [ ] Stage only `PROJECT_STATE.md` and `CHANGELOG.md`; commit with message
  `docs: close calibrated two-point tracking stage`.

## Final Coverage and Dependency Audit

| Concern | Owning tasks |
|---|---|
| Primitive estimator records/errors | 1–3 |
| Estimator-safe resources and asserted provenance | 4 |
| Canonical local model, slope, fixed cells, calibration treatment | 5 |
| Reset, sequential pair budget, first query/idempotence | 6 |
| First acceptance, partial state, reserved second query | 7 |
| Pair policy, completed scheduling/alternation, ages, atomicity | 8 |
| Instrument properties and opaque token primitives | 9 |
| Evaluator-private full-resource replay/projection/mismatch | 10 |
| Verified outcome/acquisition records consuming reviewed primitives | 11 |
| Runner/resource/abort/outcome record graph | 12 |
| Bind, verified acquisition, midpoint slots, source/token registry identity | 13 |
| Other-runner/same-runner conditional and included starts | 14 |
| Accepted steps, timing, retriable instrument failure | 15 |
| Normal/accepted-prefix/optional-abort resource joins | 16 |
| Budget/external stops, aborts, run loop, full operation×phase matrix | 17 |
| Static/Poisson/drift/loss/included-budget acceptance and separate truth oracle | 18 |
| Exact exports, docs, example, build, isolated wheel | 19 |
| Exact implementation-base review, atomic fix waves, closeout | 20 |

The dependency order is strict: estimator primitives precede factories and
tracker transitions; evaluator resource primitives precede outcome/acquisition
record validators and verified acquisition; verified acquisition precedes
start; zero-trace start precedes accepted step; accepted trace precedes public
builder; normal/optional-atom builder joins precede terminal step/stop
outcomes. No task claims a missing-helper RED after an earlier task has
consumed that helper.

The acceptance claim remains limited to the approved Stage 6.3
tracker/protocol and closed fixtures. It is not evidence of physical lock,
realtime readiness, optimal settings, sensitivity, or comparative superiority.
