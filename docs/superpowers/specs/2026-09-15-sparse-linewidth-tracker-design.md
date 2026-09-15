# Sparse Five-Point Linewidth Tracker Design

## Purpose and status

Stage 6.4 adds a lower-rate five-point local fit to calibrated fast center
tracking. It publishes active FWHM and an asynchronous live-Q projection; each
scan reports local center, FWHM, amplitude, constant offset, and single-scan Q.

This contract implements no code and claims no benchmark winner, accuracy,
latency, robustness, sensitivity, or resource superiority; those remain 6.5.

The following decisions are binding:

- Stage 6.4 is an additive composite tracker and evaluator. No Stage 6.3 public
  record, field, constructor invariant, failure meaning, or runner transition is
  changed.
- A sparse scan starts only at a fast-pair boundary and is an indivisible
  reserved block of five adjacent acquisitions.
- Its offsets are exactly `(-1.0, -0.5, 0.0, 0.5, 1.0)` times one FWHM frozen
  before the scan. Acquisition order alternates with exact time reversal.
- The fit has four free public parameters: local center correction, FWHM,
  amplitude, and a constant baseline offset. Target `eta`, the source baseline
  shape, and every non-target line tail remain frozen.
- The fitted center correction is diagnostic. It is never applied to, used to
  seed, or otherwise fed back into the fast center tracker.
- The asynchronous live-Q projection is current fast center divided by active
  sparse FWHM, with separate epochs. Scan Q is fitted center divided by FWHM.

## Approaches considered

1. **Additive composite state machine (selected).** It owns the interleaved
   schedule, reuses bound calibration and reviewed numerical helpers, and keeps
   Stage 6.3 independently stable with explicit separate update rates.
2. **Extend Stage 6.3 records/runner.** Optional linewidth fields look smaller
   but change old constructor/state meanings, so this is rejected.
3. **Joint center/linewidth filter.** It could use every sample but would
   confound sparse measurement with a motion model, so it is deferred.

The composite cannot wrap a running `CalibratedTwoPointTracker`: sparse samples
break its contiguous sequence/resource recurrence. It owns global recurrence,
uses existing calibration and characterized helpers, may reuse compatible
pair records, and never fabricates a `TwoPointEstimate` over a gapped stream.

## Architecture and ownership

```text
verified calibration -> one global scheduler/clock
                    -> repeated fast pairs -> per-ID center source
                    -> due five-point block -> per-ID FWHM source
center source + FWHM source -> asynchronous live-Q projection

evaluator: full observations, actual midpoints, expected photons, truth
tracker:   safe observations, public midpoints, policy state, safe ledgers
```

The estimator never receives instrument/dynamics/truth, expected photons,
actual midpoints, callbacks, or evaluator references. The evaluator owns them
and the full/safe join. Public records are frozen, slotted defensive snapshots.

The planned code boundaries are:

- `estimators/sparse_linewidth_{types,fit,tracker}.py`: contracts, fit, schedule;
- `evaluation/sparse_linewidth/{types,runner,resource_accounting}.py`: evaluator;
- `evaluation/two_point/calibration.py`: private runner-neutral acquisition core;
- `evaluation/two_point/provenance.py`: private exact runner allowlisting.

The composite implements its own public calibration operation; both runners use
private issuer adapters into the extracted core. It accepts only exact runner
types and registry-bound instances, then authenticates exact runner/token/
instrument/configuration/source/outcome identity. Subclasses and equal copies
fail. Every failure revokes attempt bindings and preserves existing rollback,
including commit-then-raise. Public constructors/factories, class membership,
copies, and `object.__new__` cannot mint authority. Differential tests require
bitwise/value-identical Stage 6.3 outcomes and unchanged pair/retry/stop/abort/
resource semantics.

## Exact acquisition and fit configuration

The sparse policy defaults are normative for Stage 6.4 acceptance. Every run
stores its immutable configured values. Non-default values are allowed within
the domains below, but are labeled non-acceptance policies and do not satisfy
the built-in Stage 6.4 generated acceptance fixtures:

```python
SparseLinewidthConfiguration(
    scan_period_fast_pairs=8,
    integration_time_s=0.005,
    center_correction_limit_fwhm_fraction=0.5,
    min_fwhm_prior_ratio=0.5,
    max_fwhm_prior_ratio=2.0,
    min_resolved_amplitude_source_ratio=0.25,
    max_amplitude_source_ratio=4.0,
    baseline_offset_source_amplitude_fraction=1.0,
    rank_rtol=1.0e-10,
    max_scaled_jacobian_condition=1.0e8,
    min_interior_bound_fraction=1.0e-6,
    max_amplitude_normalized_rmse=0.10,
    max_nfev=4000,
)
```

All values are finite real scalars of exact documented units; booleans, complex
values, arrays in scalar positions, and invalid NumPy scalar categories are
rejected. Period and evaluation counts are positive exact integers. Bound
ratios and the normalized-RMSE limit are positive, `rank_rtol` is in `(0, 1)`,
the condition limit is at least `1`, the interior fraction is in `[0, 0.5)`,
and the minimum FWHM ratio is strictly below its maximum. The Stage 6.4 offset
set and alternating order are not configurable.

After every eight completed fast pairs, before another fast pair, one scan is
due. The target ID is `source_ids[completed_scans % 8]`. A successful or
scientifically failed five-observation scan increments `completed_scans`,
advances that ID's scan parity, and resets the fast-pairs-since-scan count.
Partial, rejected, or aborted scans do not. Thus the first scan follows one
full r0...r7 fast round and targets r0; later scans visit IDs round-robin.

At scan start freeze:

```text
q0 = target's current fast center
w0 = target's active sparse FWHM, or calibration FWHM before first success
A0 = target amplitude from the bound calibration source
offset frequencies = q0 + multiplier * w0
```

`reset` prospectively constructs the complete five-point geometry and fit-bound
intersection for every calibration-seeded identity before committing any state.
Invalid base geometry is a typed reset construction error. At a later due-scan
boundary, the current fast center or active FWHM may make the five-query
envelope or fit bounds invalid even though reset geometry was valid. In that
case selection performs no query, changes only the terminal stop fields, and
returns `None` with `stopped_reason="sparse_geometry_unavailable"` and an exact
diagnostic snapshot of the target identity, center/width sources, proposed
envelope, calibration cell, and source domain. It never relabels this clean
boundary condition as a Stage 6.3 lost pair or narrows Stage 6.3 pair semantics.
Geometry-code precedence is nonrepresentable lower, nonrepresentable upper,
empty fit bounds, cell violation, then source-domain violation. Proposed
minimum and maximum are jointly `None` for either nonrepresentable code and are
jointly present, finite, and ordered for the other three codes. `q0`, `w0`,
cell bounds, and source bounds are always finite and present; a lower/upper
arithmetic failure therefore never erases the exact prior facts that caused it.

The canonical even per-ID scan order is:

```text
(+0.5, -1.0, 0.0, +1.0, -0.5)
```

The odd per-ID order is its exact time reversal:

```text
(-0.5, +1.0, 0.0, -1.0, +0.5)
```

Every completed attempt therefore has the same mathematical design matrix,
while successive attempts for one ID reverse every frequency's temporal
position. With equal time spacing and centered time indices
`t=(-2,-1,0,+1,+2)`, both orders satisfy exact `sum(t*x) == 0` for offset
multiplier `x`. This cancels the design's linear time-frequency correlation;
it does not identify, correct, or guarantee robustness to nonlinear or
parameter dynamics within a scan.
All five frequencies, their order, integration times, expected sequence
indices, expected endpoints, and nominal exposures are constructed and frozen
before the first query. No result from points one through four changes point
five or starts a fast pair.

### Stateful query-constructor interface correction

The original implementation ledger listed the private stateful query
constructor without any input carrying the sparse integration time. That
signature was insufficient: `_SparseFitGeometry` is intentionally pure and
retains no scheduler policy, `TwoPointRunMetadata` contains rate/overhead but
not sparse integration time, and `SparseLinewidthConfiguration.integration_time_s`
is explicitly configurable. Inferring the default `0.005` would silently break
legal non-default configurations; adding scheduler policy to pure fit geometry
would weaken the model/scheduling separation.

The stateful constructor therefore receives one additional private keyword,
`integration_time_s: float`, supplied from the tracker-owned immutable sparse
configuration. This makes the dependency explicit and testable while leaving
the public API and pure geometry unchanged. The constructor still obtains the
nominal photon rate and frequency overhead from public run metadata. The
tradeoff is a one-argument extension of a private helper signature relative to
the original ledger; no estimator-facing contract changes.

## Canonical local model

For source baseline `B`, source-ordered resonances, target `i`, source mixture
`eta_i`, and free parameters `(dc, w, A, b0)`, the five predicted values are

```text
M_i(f; dc, w, A, b0) =
    B(f) + b0
    - sum over j != i in immutable source order of
        A_j * P(f; c_j, w_j, eta_j)
    - A * P(f; q0 + dc, w, eta_i)
```

`P` is the repository's explicit-FWHM pseudo-Voigt; a Lorentzian source has
`eta_i == 1.0`. Production evaluates the baseline once, then subtracts lines
in immutable source order. It must not materialize or regroup a pre-summed
background. The source baseline intercept, slope, quadratic term, and reference
remain frozen. Only the additive constant `b0` moves; Stage 6.4 does not fit an
affine local slope. Every non-target center, FWHM, amplitude, and eta remains
frozen. Target eta also remains frozen.

Under the normative acceptance configuration, the four free public parameters
have these exact closed bounds, intersected with the source fit's absolute
FWHM and amplitude bounds. A labeled non-acceptance configuration substitutes
its recorded fractions and ratios in the same formulas:

```text
-0.5*w0 <= dc <= +0.5*w0
max(0.5*w0, source_min_fwhm_hz) <= w
    <= min(2.0*w0, source_max_fwhm_hz)
0.0 <= A <= min(4.0*A0, source_max_amplitude)
-1.0*A0 <= b0 <= +1.0*A0
```

The local-center interval `q0 + dc` must also lie inside the target's fixed
Stage 6.3 calibration cell. All five frozen queries must lie inside that cell
and source frequency domain. An empty intersection or non-representable bound
is a reset/scan-construction error, not a fit attempt. The public initial guess
must be strictly inside every intersected bound. Exact endpoints are legal
optimizer outputs but do not pass the post-fit interior gate.

Optimization uses `scipy.optimize.least_squares(method="trf")`, the public
initial guess `(0.0, w0, A0, 0.0)`, the exact bounds above, and no stochastic
restart. Residuals are model minus observed fluorescence. The packed variables
are dimensionless:

```text
x = (dc/w0, w/w0, A/A0, b0/A0)
```

The optimizer Jacobian and all rank/condition decisions are for these scaled
variables. This prevents unit choice alone from deciding identifiability.

## Ordered fit gates and failure semantics

Exactly five points and four parameters give one residual degree of freedom.
A completed scan evaluates these gates in order; the first failure is retained:

| Order | Failure code | Exact gate |
| ---: | --- | --- |
| 1 | `model_evaluation_failed` | Initial model/residual/bounds/scales return a non-finite or non-representable value without raising. |
| 2 | `optimizer_failed` | `least_squares` returns `status <= 0` or `nfev >= max_nfev`; these are the only scientific optimizer-failure paths. |
| 3 | `nonfinite_solution` | A returned successful solver result has the wrong residual/Jacobian shape, or any fitted public parameter, prediction, residual, cost, RMSE, or singular value is non-finite or not representable. |
| 4 | `bounds_active` | A parameter is outside its closed bound or its distance to either bound divided by that bound's span is `< 1.0e-6`. Equality to `1.0e-6` passes. |
| 5 | `rank_deficient` | With singular values in descending order and cutoff `s_max * 1.0e-10`, the count satisfying `s > cutoff` is not exactly four. Equality to the cutoff is discarded. |
| 6 | `ill_conditioned` | Full rank holds but `s_max / s_min > 1.0e8`. Equality passes. |
| 7 | `amplitude_unresolved` | Fitted amplitude is below `max(0.25*A0, source_min_resolved_amplitude)` after public conversion. Equality passes. |
| 8 | `residual_quality_failed` | `sqrt(sum(residual**2)/5) / fitted_amplitude > 0.10`. Equality passes. |

The bound span and normalized distances are evaluated in scaled coordinates.
Rank uses one deterministic NumPy SVD of the final scaled Jacobian; condition
uses the same singular values and never recomputes another factorization. The
RMSE numerator uses the optimizer residual's fixed five-element arrival order
with ordinary NumPy binary64 operations. No uncertainty claim is published
from the single remaining degree of freedom in Stage 6.4.

Diagnostic presence is exact. `fit_cpu_time_s` is present for every completed
scan. Solver fields mean the exact canonical built-in `int` status, built-in
string message, and positive `nfev` returned by SciPy:

| Result | Solver fields | Five fitted public fields | RMSE fields | Rank | Condition |
| --- | --- | --- | --- | --- | --- |
| `model_evaluation_failed` | absent | absent | absent | absent | absent |
| `optimizer_failed` | present | absent | absent | absent | absent |
| `nonfinite_solution` | present | absent | absent | absent | absent |
| `bounds_active` | present | present | present | absent | absent |
| `rank_deficient` | present | present | present | present, `0..3` | absent |
| `ill_conditioned` | present | present | present | exactly `4` | present |
| `amplitude_unresolved` | present | present | present | exactly `4` | present |
| `residual_quality_failed` | present | present | present | exactly `4` | present |
| `success` | present | present | present | exactly `4` | present |

The five fitted public fields are center correction, local center, FWHM,
amplitude, and baseline offset; fitted Q is additionally present only on
`success`. No partially finite group is published. `scipy_status > 0` is
required for every row after `optimizer_failed`. Status/failure-code presence
is `success`/`None` or `failure`/exactly one code.

`success` requires all eight gates to pass. It publishes `dc`, local center
`q0 + dc`, FWHM, amplitude, baseline offset, RMSE, normalized RMSE, rank `4`,
condition number, and

```text
scan_q = fitted_local_center_hz / fitted_fwhm_hz
```

Q preserves the repository convention exactly: `center/FWHM` may be finite
negative, zero, or positive. Only a non-finite/nonrepresentable division makes
the scan `nonfinite_solution`; generated physical scenarios separately require
positive absolute centers. A scientific failure is a completed, charged scan. It
publishes every finite diagnostic available before its failing gate, applies no
new FWHM, and retains the older active FWHM source. Failure is not an exception.
Any ordinary `Exception` raised by model arithmetic, scaling, `least_squares`,
SVD, Q derivation, timing, or an unexpected collaborator is not a scientific
fit result: it rolls back the whole tracker update and becomes a typed
construction/unexpected evaluator abort after a returned acquisition.
Validation and immutable-record construction errors follow the same rollback.
Process-control `BaseException` values propagate unchanged after transaction
cleanup and are never converted.

## Public contracts

The exact field surfaces are fixed below; implementations may add private
helpers but not unvalidated metadata maps.

```python
SparseLinewidthFailureCode = Literal[
    "model_evaluation_failed", "optimizer_failed", "nonfinite_solution",
    "bounds_active", "rank_deficient", "ill_conditioned",
    "amplitude_unresolved", "residual_quality_failed",
]
CompositeMode = Literal["fast_pair", "sparse_scan"]
SparseLinewidthSourceKind = Literal["calibration", "scan"]
CompositeStopReason = Literal[
    "budget_exhausted", "sparse_geometry_unavailable",
]
SparseGeometryFailureCode = Literal[
    "nonrepresentable_frequency_lower", "nonrepresentable_frequency_upper",
    "empty_fit_bounds", "calibration_cell_violation",
    "source_domain_violation",
]
SparseGeometryUnavailableDiagnostic(
    failure_code: SparseGeometryFailureCode,
    scan_index: int, identity_scan_index: int, resonance_id: str,
    fast_center_hz: float,
    fast_center_source_kind: Literal["calibration", "pair"],
    fast_center_source_pair_index: int | None, fast_center_reference_timestamp_s: float,
    fast_center_release_sequence_index: int | None,
    fast_center_release_timestamp_s: float,
    prior_fwhm_hz: float, fwhm_source_kind: SparseLinewidthSourceKind,
    fwhm_source_scan_index: int | None, fwhm_reference_timestamp_s: float,
    fwhm_release_sequence_index: int | None, fwhm_release_timestamp_s: float,
    proposed_frequency_min_hz: float | None, proposed_frequency_max_hz: float | None,
    calibration_cell_lower_hz: float, calibration_cell_upper_hz: float,
    source_frequency_min_hz: float, source_frequency_max_hz: float,
)

SparseLinewidthQuery(
    acquisition_index: int, scan_index: int, identity_scan_index: int,
    point_index: int, resonance_id: str, offset_multiplier: float,
    frozen_fast_center_hz: float,
    frozen_fast_center_source_kind: Literal["calibration", "pair"],
    frozen_fast_center_source_pair_index: int | None,
    frozen_fast_center_reference_timestamp_s: float,
    frozen_fast_center_release_sequence_index: int | None,
    frozen_fast_center_release_timestamp_s: float,
    frozen_prior_fwhm_hz: float, frozen_fwhm_source_kind: SparseLinewidthSourceKind,
    frozen_fwhm_source_scan_index: int | None, frozen_fwhm_reference_timestamp_s: float,
    frozen_fwhm_release_sequence_index: int | None,
    frozen_fwhm_release_timestamp_s: float,
    frequency_hz: float, integration_time_s: float, expected_sequence_index: int,
    expected_end_timestamp_s: float,
    expected_nominal_exposure_photons: float,
)

SparsePartialScan(
    scan_index: int, identity_scan_index: int, resonance_id: str,
    frozen_fast_center_hz: float,
    frozen_fast_center_source_kind: Literal["calibration", "pair"],
    frozen_fast_center_source_pair_index: int | None,
    frozen_fast_center_reference_timestamp_s: float,
    frozen_fast_center_release_sequence_index: int | None,
    frozen_fast_center_release_timestamp_s: float,
    frozen_prior_fwhm_hz: float, frozen_fwhm_source_kind: SparseLinewidthSourceKind,
    frozen_fwhm_source_scan_index: int | None, frozen_fwhm_reference_timestamp_s: float,
    frozen_fwhm_release_sequence_index: int | None,
    frozen_fwhm_release_timestamp_s: float,
    queries: tuple[SparseLinewidthQuery, ...],       # length 1..4
    observations: tuple[EstimatorObservation, ...], # same length/order
)

SparseLinewidthScanResult(
    scan_index: int, identity_scan_index: int, resonance_id: str,
    frozen_fast_center_hz: float,
    frozen_fast_center_source_kind: Literal["calibration", "pair"],
    frozen_fast_center_source_pair_index: int | None,
    frozen_fast_center_reference_timestamp_s: float,
    frozen_fast_center_release_sequence_index: int | None,
    frozen_fast_center_release_timestamp_s: float,
    frozen_prior_fwhm_hz: float, frozen_fwhm_source_kind: SparseLinewidthSourceKind,
    frozen_fwhm_source_scan_index: int | None, frozen_fwhm_reference_timestamp_s: float,
    frozen_fwhm_release_sequence_index: int | None,
    frozen_fwhm_release_timestamp_s: float,
    queries: tuple[SparseLinewidthQuery, ...],       # length exactly 5
    observations: tuple[EstimatorObservation, ...], # same arrival order
    public_reference_timestamp_s: float, release_sequence_index: int,
    release_timestamp_s: float, status: Literal["success", "failure"],
    failure_code: SparseLinewidthFailureCode | None,
    fitted_center_correction_hz: float | None,
    fitted_local_center_hz: float | None,
    fitted_fwhm_hz: float | None, fitted_amplitude: float | None,
    fitted_baseline_offset: float | None, fitted_q: float | None,
    rmse: float | None,
    amplitude_normalized_rmse: float | None,
    scaled_jacobian_rank: int | None, scaled_jacobian_condition: float | None,
    scipy_status: int | None, scipy_message: str | None,
    nfev: int | None, fit_cpu_time_s: float,
)

CompositeIdentityEstimate(
    resonance_id: str, fast_center_hz: float,
    fast_center_source_kind: Literal["calibration", "pair"],
    fast_center_source_pair_index: int | None, fast_center_reference_timestamp_s: float,
    fast_center_release_sequence_index: int | None,
    fast_center_release_timestamp_s: float,
    active_fwhm_hz: float, fwhm_source_kind: SparseLinewidthSourceKind,
    fwhm_source_scan_index: int | None, fwhm_reference_timestamp_s: float,
    fwhm_release_sequence_index: int | None, fwhm_release_timestamp_s: float,
    live_q: float, center_age_s: float, fwhm_age_s: float,
    center_release_age_s: float, fwhm_release_age_s: float,
    completed_fast_pairs: int, completed_sparse_scans: int,
    latest_fast_pair: TwoPointPairResult | None,
    latest_sparse_scan: SparseLinewidthScanResult | None,
)

SparseLinewidthCompositeEstimate(
    configuration: SparseLinewidthConfiguration,
    identities: tuple[CompositeIdentityEstimate, ...],
    calibration_source_id: str, calibration_source_provenance: CalibrationSourceProvenance,
    calibration_budget_treatment: CalibrationBudgetTreatment,
    pending_mode: CompositeMode | None, pending_query: TwoPointQuery | SparseLinewidthQuery | None,
    incomplete_fast_pair: TwoPointPartialPair | None,
    incomplete_sparse_scan: SparsePartialScan | None,
    fast_pair_history: tuple[TwoPointPairResult, ...],
    sparse_scan_history: tuple[SparseLinewidthScanResult, ...],
    accepted_observations: int, completed_fast_pairs: int,
    completed_sparse_scans: int, fast_pairs_since_scan: int,
    current_sequence_index: int | None, current_timestamp_s: float,
    fast_tracking_resources: PublicAcquisitionResources,
    sparse_tracking_resources: PublicAcquisitionResources,
    tracking_resources: PublicAcquisitionResources,
    calibration_resources: PublicAcquisitionResources,
    charged_resources: PublicAcquisitionResources,
    budget_ceiling: TwoPointBudgetCeiling, stopped_reason: CompositeStopReason | None,
    sparse_geometry_diagnostic: SparseGeometryUnavailableDiagnostic | None,
    fast_update_cpu_time_s: float, sparse_update_cpu_time_s: float,
    total_update_cpu_time_s: float, seed: int,
)

SparseLinewidthCompositeUpdate(
    query: TwoPointQuery | SparseLinewidthQuery,
    observation: EstimatorObservation,
    completed_fast_pair: TwoPointPairResult | None,
    completed_sparse_scan: SparseLinewidthScanResult | None,
    estimate: SparseLinewidthCompositeEstimate,
    update_cpu_time_s: float,
)

SparseLinewidthEvaluatorScanTiming(
    scan_index: int, resonance_id: str,
    measurement_midpoints_s: tuple[float, float, float, float, float],
    truth_reference_timestamp_s: float, public_reference_timestamp_s: float,
    release_sequence_index: int, release_timestamp_s: float,
)

SparseLinewidthEvaluatorResources(
    calibration_observations: tuple[InstrumentObservation, ...],
    accepted_fast_observations: tuple[InstrumentObservation, ...],
    accepted_sparse_observations: tuple[InstrumentObservation, ...],
    accepted_tracking_observations: tuple[InstrumentObservation, ...],
    unaccepted_tracking_observations: tuple[InstrumentObservation, ...],
    calibration_resources: ResourceSnapshot, fast_tracking_resources: ResourceSnapshot,
    sparse_tracking_resources: ResourceSnapshot, tracking_resources: ResourceSnapshot,
    accepted_charged_resources: ResourceSnapshot, charged_resources: ResourceSnapshot,
    calibration_budget_treatment: CalibrationBudgetTreatment,
    incomplete_fast_pair_observations: Literal[0, 1],
    incomplete_sparse_scan_observations: Literal[0, 1, 2, 3, 4],
    unaccepted_observations: Literal[0, 1],
)
```

The tracker interface mirrors Stage 6.3 at the composite boundary:

```python
class SparseLinewidthCompositeTracker:
    def __init__(self, configuration: SparseLinewidthConfiguration): ...

    def reset(
        self,
        public_metadata: TwoPointRunMetadata,
        calibration: TwoPointCalibration,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> None: ...
    def choose_next_query(
        self,
    ) -> TwoPointQuery | SparseLinewidthQuery | None: ...
    def update(
        self, observation: EstimatorObservation
    ) -> SparseLinewidthCompositeUpdate: ...
    def estimate(self) -> SparseLinewidthCompositeEstimate: ...
```

The evaluator contract is additive and exact:

```python
SparseRunnerPhase = Literal[
    "ready", "calibration_succeeded", "calibration_failed", "tracking",
    "budget_stopped", "geometry_stopped", "externally_stopped", "aborted",
]
SparseAbortReason = Literal[
    "resource_join_unavailable", "tracker_observation_validation_error",
    "tracker_update_construction_error", "tracker_update_unexpected_error",
]
SparsePreflightCode = Literal[
    "invalid_runner_phase", "invalid_argument_type", "invalid_argument_value",
    "invalid_frequency_grid", "invalid_fit_or_identity_configuration",
    "invalid_clock_mapping", "unclean_instrument_boundary",
]
SparseStartCode = Literal[
    "invalid_runner_phase", "invalid_argument_type", "unverified_calibration",
    "calibration_mismatch", "run_provenance_mismatch", "metadata_mismatch",
    "resource_boundary_mismatch", "tracker_reset_failed",
]
SparseTrackingAcquisition(
    resource_join_status: Literal["authenticated"], mode: CompositeMode,
    query: TwoPointQuery | SparseLinewidthQuery,
    expected_measurement_midpoint_s: float, measurement_midpoint_s: float | None,
    full_observation: InstrumentObservation, safe_observation: EstimatorObservation,
    instrument_resources_before: ResourceSnapshot,
    instrument_resources_after: ResourceSnapshot,
    instrument_resource_delta: ResourceSnapshot,
)
SparseResourceJoinUnavailableAcquisition(
    resource_join_status: Literal["unavailable"], mode: CompositeMode,
    query: TwoPointQuery | SparseLinewidthQuery,
    expected_measurement_midpoint_s: float, measurement_midpoint_s: float | None,
    full_observation: InstrumentObservation, safe_observation: EstimatorObservation,
    resource_mismatch_fields: tuple[ResourceJoinMismatchField, ...],
    instrument_resources_before: ResourceSnapshot,
    instrument_resources_after: ResourceSnapshot,
)
SparseInstrumentQueryFailure(
    mode: CompositeMode, query: TwoPointQuery | SparseLinewidthQuery,
    exception_type: str, exception_message: str,
    instrument_resources_before: ResourceSnapshot,
    instrument_resources_after: ResourceSnapshot,
)
SparseAbortedRun(
    reason: SparseAbortReason, exception_type: str | None,
    exception_message: str | None,
    unaccepted_acquisition: SparseTrackingAcquisition |
        SparseResourceJoinUnavailableAcquisition,
    unaccepted_observation_count: Literal[1],
    tracker_estimate_before: SparseLinewidthCompositeEstimate,
    tracker_estimate_after: SparseLinewidthCompositeEstimate,
)
SparseEvaluatorRunnerState(
    phase: SparseRunnerPhase, run_token: VerifiedInstrumentRunToken,
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration,
    calibration_outcome: VerifiedTwoPointCalibrationOutcome | None,
    verified_calibration: VerifiedTwoPointCalibrationSuccess | None,
    calibration: TwoPointCalibration | None,
    tracker_estimate: SparseLinewidthCompositeEstimate | None,
    normal_tracking_trace: tuple[SparseTrackingAcquisition, ...],
    pair_timings: tuple[TwoPointEvaluatorPairTiming, ...],
    scan_timings: tuple[SparseLinewidthEvaluatorScanTiming, ...],
    instrument_resources_at_bind: ResourceSnapshot,
    tracking_resources_before: ResourceSnapshot | None,
    instrument_resources_current: ResourceSnapshot,
    instrument_current_sequence_index: int | None, current_virtual_time_s: float,
    last_instrument_failure: SparseInstrumentQueryFailure | None,
    terminal_abort: SparseAbortedRun | None,
    fast_update_cpu_time_s: float,
    sparse_update_cpu_time_s: float,
    total_update_cpu_time_s: float,
)
SparseRunnerAccepted(kind: Literal["accepted"], acquisition: SparseTrackingAcquisition,
    update: SparseLinewidthCompositeUpdate, state: SparseEvaluatorRunnerState)
SparseRunnerInstrumentFailure(kind: Literal["instrument_failure"],
    failure: SparseInstrumentQueryFailure, state: SparseEvaluatorRunnerState)
SparseRunnerBudgetStopped(kind: Literal["budget_stopped"],
    resources: SparseLinewidthEvaluatorResources, state: SparseEvaluatorRunnerState)
SparseRunnerGeometryStopped(kind: Literal["geometry_stopped"],
    diagnostic: SparseGeometryUnavailableDiagnostic,
    resources: SparseLinewidthEvaluatorResources, state: SparseEvaluatorRunnerState)
SparseRunnerExternallyStopped(kind: Literal["externally_stopped"],
    resources: SparseLinewidthEvaluatorResources, state: SparseEvaluatorRunnerState)
SparseRunnerAborted(kind: Literal["aborted"], abort: SparseAbortedRun,
    resources: SparseLinewidthEvaluatorResources | None,
    state: SparseEvaluatorRunnerState)
SparseRunnerStepOutcome = SparseRunnerAccepted | SparseRunnerInstrumentFailure |
    SparseRunnerBudgetStopped | SparseRunnerGeometryStopped | SparseRunnerAborted
SparseRunnerRunOutcome = SparseRunnerInstrumentFailure | SparseRunnerBudgetStopped |
    SparseRunnerGeometryStopped | SparseRunnerAborted
class SparsePreflightError(ValueError): code: SparsePreflightCode
class SparseStartError(ValueError): code: SparseStartCode
class SparseRunnerStateError(RuntimeError): ...

class SparseLinewidthEvaluatorRunner:
    @classmethod
    def bind(cls, instrument: ODMRInstrument) -> SparseLinewidthEvaluatorRunner: ...
    @property
    def state(self) -> SparseEvaluatorRunnerState: ...
    def acquire_verified_calibration(
        self, frequency_hz: Sequence[float], integration_time_s: float,
        fit_configuration: FitConfiguration,
        identity_binding: TwoPointIdentityBinding, *, source_id: str,
        source_clock_id: str, tracker_clock_id: str,
        source_to_tracker_offset_s: float,
        physical_fit_epoch_rule: Literal["instrument_midpoint_ordered_mean"],
    ) -> VerifiedTwoPointCalibrationOutcome: ...
    def start_tracking(
        self, tracker: SparseLinewidthCompositeTracker,
        calibration: TwoPointCalibration,
        verified_calibration: VerifiedTwoPointCalibrationSuccess,
        public_metadata: TwoPointRunMetadata,
        budget_ceiling: TwoPointBudgetCeiling, *, seed: int,
    ) -> SparseEvaluatorRunnerState: ...
    def step(self) -> SparseRunnerStepOutcome: ...
    def run_until_event(self) -> SparseRunnerRunOutcome: ...
    def stop_external(self) -> SparseRunnerExternallyStopped: ...
```

`bind` accepts one exact clean `ODMRInstrument` and returns `ready`.
`acquire_verified_calibration` is legal only in `ready` and terminates in
`calibration_succeeded` or `calibration_failed`. `start_tracking` is legal in
`ready` only with an exact other-runner conditional source, or in
`calibration_succeeded` with its exact outcome; success enters `tracking`.
`step`, `run_until_event`, and `stop_external` are legal only in `tracking`.
`step` returns exactly one declared step union member. `run_until_event` loops
only over accepted steps and returns its first declared terminal/retryable
member; `stop_external` performs no query and returns its dedicated outcome.
Every illegal phase fails before touching tracker or instrument: acquire uses
`SparsePreflightError`, start uses `SparseStartError`, and the last three use
`SparseRunnerStateError`. Terminal phases expose only read-only state/resources.

Exact-type preflight precedes value, grid, fit/identity, clock, then clean-boundary
checks; start precedence is the `SparseStartCode` order above. `ready` has no
calibration/tracker; calibration phases hold one matching verified outcome;
tracking/terminal phases hold exact calibration and tracker. Outcome kind equals
state phase; stopped resources equal the builder, and geometry diagnostic equals
the estimate. Normal trace safe projections equal the accepted global
stream. Pair timings align one-to-one with fast-pair history under the existing
`TwoPointEvaluatorPairTiming` joins; scan timings align one-to-one with sparse
history and its exact five actual/public midpoint folds. A query
failure exists only in nonterminal tracking, has equal resource boundaries, and
leaves the pending query unchanged. Abort requires equal before/after tracker
snapshots; unavailable joins require the unavailable acquisition, no exception
strings, and `resources=None`, while other aborts require an authenticated atom,
canonical exception strings, and exact final resources. Token, source,
calibration, tracker, instrument, sequence/time, and full/safe identity joins
are checked by the owning runner before record construction. Existing Stage
6.3 exported names and objects remain unchanged; Stage 6.4 names are appended.

Every sparse query snapshots both active-source tuples shown in its fields; all
five carry bitwise/value-identical snapshots. The partial and result repeat
those fields and require exact equality to every query rather than reconstructing
them from current state. The immutable estimate configuration and its source
ID, provenance, and treatment equal the bound calibration on every transition.

`fit_cpu_time_s` is the finite nonnegative delta of `time.process_time_ns()`
around model preparation, solver call, and fit gates. `update_cpu_time_s` is
the finite nonnegative process-CPU delta around the accepted estimator update,
including fit work on a fifth sparse point. Neither includes instrument
acquisition or waiting, enters any acquisition resource ledger, or supports a
realtime-performance claim; both are descriptive machine-dependent diagnostics.
After every accepted update, `total_update_cpu_time_s` performs exactly
`old_total + update.update_cpu_time_s` in global arrival order. The matching
mode subtotal performs the same left-associated addition and the other subtotal
is unchanged. Total is never formed by adding fast and sparse subtotals. Query
failure, clean stop, external stop, and abort add nothing. Runner CPU fields
equal tracker-estimate totals (all-zero pre-reset) through terminal outcomes.

## Asynchronous live-Q projection, epochs, and timing

For each identity, the live projection is always

```text
live_q = fast_center_hz / active_fwhm_hz
```

It is recomputed after every successful fast pair and successful sparse scan,
but is not a single-epoch Q. Finite signed and zero results are valid; only a
nonfinite/nonrepresentable division fails, transactionally, as
`aggregate_estimate_construction_failed`. The record keeps separate source
kinds, source indices, public reference timestamps,
release sequence indices, release timestamps, and ages. It never fabricates a
single Q epoch. Before the first successful scan, FWHM comes from calibration
with the calibration physical-fit and availability epochs. A failed scan ages
but does not refresh that source. A successful scan refreshes the FWHM source
even if its fitted FWHM is bitwise equal to the prior value.

The fitted local center belongs only to its `SparseLinewidthScanResult` and its
`fitted_q`. The fast center, fast source fields, and next fast interrogation
center remain value-equal across sparse completion. This no-feedback invariant
is checked structurally and transitionally.

For each safe observation, the tracker reconstructs
`public_midpoint = endpoint - integration/2`. The five-point public reference
is the left fold of overflow-safe ordered means:

```text
mean = public_midpoint[0]
for count, value in enumerate(public_midpoint[1:], start=2):
    mean = mean + (value - mean) / count
```

The evaluator records each actual midpoint from the instrument's pre-query
clock association and computes the truth reference with the same fold over
actual midpoints. It stores both values in a
`SparseLinewidthEvaluatorScanTiming`. Truth lookup and eventual Stage 6.5
scoring use only the actual-midpoint truth reference. The estimator sees only
the public reference. Release is the fifth-arriving observation's sequence and
endpoint; neither reference is a causal availability time.

## Schedule, reservation, and state transitions

Only two acquisition blocks exist: a two-observation fast pair and a
five-observation sparse scan. At a clean pair/scan boundary, the scheduler first
checks whether a sparse scan is due; otherwise it selects the next Stage 6.3
round-robin pair. A due scan checks geometry before affordability; a valid
selected block is fully reserved before its first query is exposed.

Each prospective observation applies exactly one Stage 6.3 canonical charge:

```text
observations = old + 1
integration = old + query.integration_time_s
nominal exposure = old + rate * query.integration_time_s
elapsed = old + (frequency_overhead_s + query.integration_time_s)
```

Affordability applies that transition sequentially two or five times from the
current charged state, in planned query order. It does not multiply one charge,
add a block subtotal, use subtraction, `sum`, or `math.fsum`. Every capped field
must be `<=` its ceiling. Expected and realized photons are not affordability
inputs.

If a due block is unaffordable, selection atomically sets
`budget_exhausted`, returns `None`, and never falls back to the other block.
Normal budget exhaustion therefore leaves no partial block. Once reserved, all
remaining queries issue without another budget check. A repeated selection
with a pending query returns the exact same object. A first through fourth
sparse acceptance charges one observation and exposes an exact partial scan but
does not fit, update FWHM, append history, advance parity, or schedule a pair.
The fifth acceptance constructs the complete result and prospective aggregate
before one commit.

An ordinary instrument exception before a returned observation leaves the
query pending and is retryable. A returned observation that cannot be joined or
accepted produces a terminal typed abort retaining the unaccepted acquisition.
External stop preserves a partial reserved block and pending query. No later
operation can complete or reinterpret that partial block. Scientific fit
failure is a normal completed scan, not an abort.

## Exact resource ledgers

Public state retains five separate immutable ledgers:

- `calibration_resources`: the bound source's safe resources;
- `fast_tracking_resources`: accepted fast observations replayed from zero;
- `sparse_tracking_resources`: accepted sparse observations replayed from zero;
- `tracking_resources`: all accepted fast and sparse observations replayed in
  actual interleaved arrival order from zero;
- `charged_resources`: source atoms followed by interleaved tracking atoms for
  `included_same_run`, or interleaved tracking atoms from zero for
  `conditional_free_precalibration`.

The aggregate is never formed by adding the fast and sparse subtotals. Each
atomic replay uses the Stage 6.3 left-associated binary64 rules for observations,
integration, nominal exposure, realized/missing counts, and elapsed time.
Reserved, unissued, and ordinary-exception queries charge nothing. Every
accepted point of a partial scan is charged and appears in both sparse and
interleaved tracking ledgers. Every point of a scientifically failed complete
scan is also charged.

The evaluator mirrors those ledgers with full `ResourceSnapshot` values and
expected photons, plus zero-or-one authenticated unaccepted observation. It
replays calibration, accepted interleaved tracking, accepted charged prefix,
and final charged resources atom by atom. Full observations must project
exactly to estimator-safe history/partial state. A resource-join-unavailable
abort retains raw boundaries and returns no fabricated aggregate resource
record, matching Stage 6.3 semantics.

Core state joins include:

```text
accepted_observations ==
    2*completed_fast_pairs + 5*completed_sparse_scans
    + length(incomplete_fast_pair observations)
    + length(incomplete_sparse_scan observations)
completed_fast_pairs == len(fast_pair_history)
completed_sparse_scans == len(sparse_scan_history)
tracking_resources.observations == accepted_observations
fast_tracking_resources.observations ==
    2*completed_fast_pairs + length(incomplete_fast_pair observations)
sparse_tracking_resources.observations ==
    5*completed_sparse_scans + length(incomplete_sparse_scan observations)
```

Exactly one incomplete block kind may exist. A pending query agrees with that
block and mode, except that the first pending query has no partial record.
Sequence index and endpoint advance for every accepted global observation,
regardless of mode.

## Validation and failure boundaries

```python
SparseResetFailureCode = Literal[
    "invalid_argument_type", "configuration_mismatch", "calibration_mismatch",
    "metadata_mismatch", "invalid_base_sparse_geometry", "budget_mismatch",
    "initial_state_construction_failed",
]
SparseObservationValidationCode = Literal[
    "invalid_observation_type", "no_pending_query", "pending_mode_mismatch",
    "fast_query_echo_mismatch", "sparse_query_echo_mismatch",
    "sequence_mismatch", "frequency_mismatch", "integration_time_mismatch",
    "endpoint_mismatch", "nominal_exposure_mismatch",
    "invalid_observation_value",
]
SparseUpdateConstructionCode = Literal[
    "fast_partial_pair_construction_failed",
    "fast_pair_result_construction_failed",
    "fast_identity_estimate_construction_failed",
    "sparse_partial_scan_construction_failed",
    "sparse_scan_result_construction_failed",
    "sparse_identity_estimate_construction_failed",
    "resource_construction_failed", "aggregate_estimate_construction_failed",
    "update_construction_failed",
]
class SparseLinewidthResetError(ValueError): code: SparseResetFailureCode
class SparseLinewidthObservationValidationError(ValueError):
    code: SparseObservationValidationCode
class SparseLinewidthUpdateConstructionError(RuntimeError):
    code: SparseUpdateConstructionCode
```

Reset code precedence is declaration order and reset rolls back to the exact
prior configuration/state on every failure. Observation precedence is exact
safe type, pending query, mode, the mode-specific pending-query/state echo,
sequence, frequency, integration, endpoint, nominal exposure, then value.
There is no tolerance, sorting, resampling, duplicate suppression, or timestamp
inference. The fast branch may call the reviewed Stage 6.3 validator privately,
but catches and translates it; no Stage 6.3 validation/update exception crosses
the public composite boundary.

After validation, fast construction precedence is partial pair (first side) or
pair result then fast identity (second side), followed by resources, aggregate,
and update. Sparse precedence is partial scan (points one through four) or scan
result then sparse identity (point five), followed by the same final three.
The first failing exact code is chained from the cause. Any such failure or
unexpected ordinary exception leaves every tracker field and all three CPU
totals value-equal to entry; `BaseException` cleanup preserves the same rollback
and re-raises the identical object.

The runner adds `geometry_stopped` to Stage 6.3's ready, calibration
success/failure, tracking, budget/external stop, and abort phases. Starting the
composite authenticates the exact verified calibration source, token, runner,
instrument, clock, metadata, treatment, and resource boundary before reset.
Included and conditional calibration treatments keep their Stage 6.3 meanings.
No new record can mint verified provenance.

At evaluator level, a returned full observation remains an authenticated
unaccepted atom when its join is valid; the runner never continues after abort.

## Verification and test contract

Implementation is not complete until the following deterministic groups pass:

1. **Records/exports:** exact fields, frozen/slotted defensive snapshots, union
   invariants, scalar canonicalization, `__all__`, and isolated-wheel imports.
2. **Schedule/model:** first scan after eight pairs, ID rotation, both exact
   orders, frozen sources/frequencies, no interleaving/feedback, all eight IDs,
   source-order tails, frozen terms, and both supported line shapes.
3. **Gates:** every first-applicable code; rank 3/4 and cutoff ULPs; condition
   `1e8` equality/outward ULP; bound-margin `1e-6` equality/inward ULP;
   amplitude/RMSE boundaries; exception rollback; and the presence matrix.
4. **Q/epochs/timing:** calibration seed, success refresh, failure retention,
   signed/zero asynchronous and scan Q, nonrepresentable division,
   both five-value time folds, neighboring ULPs, release, and truth isolation.
5. **Reservation/resources/atomicity:** exact ceilings, every geometry code and
   presence case, retry/partial/abort/rollback, both treatments, interleaved
   acquisition/CPU replay, pair/scan timing joins, and unavailable joins.
6. **Compatibility/isolation:** full old suite plus bitwise Stage 6.3 calibration
   and fast-only differential traces; retained graphs forbid full observations,
   expected photons, truth/dynamics, callbacks, instruments, and evaluators.

Generated acceptance fixtures may pin exact values and tolerances for this
declared model and noise setup. They must be labeled synthetic contract tests,
not general performance evidence.

## Documentation deliverables

Estimator guidance and package/API smoke must cover the fixed offsets/orders,
fitted versus frozen parameters, gate boundaries and presence matrix,
asynchronous live projection versus scan Q, separate sources and public/truth
times, all ledgers/treatments, stop/failure/abort/partial behavior, and declared
model mismatch. A download-free synthetic example may print finite diagnostics
but may not claim superiority or experimental validity.

## Limitations and non-goals

The local model can absorb only a constant baseline displacement. Any true
within-scan affine baseline change is model mismatch and can bias center,
linewidth, amplitude, and Q; the source baseline's existing slope is frozen,
not re-estimated. Adding a fifth free slope to five observations would remove
positive residual degrees of freedom and is explicitly rejected.

The five acquisitions are not simultaneous. Alternating exact time reversal
balances deterministic order across repeated scans but cannot identify or
remove center drift, linewidth drift, amplitude drift, baseline drift, or
neighbor-tail evolution within one scan. The fitted local center correction is
therefore a diagnostic of the frozen local model, not an independent fast lock
update. A state-space motion model or randomized scan is future work.

Out of scope are Stage 6.3 contract/pair changes; fitting eta, baseline
slope/curvature, or non-target parameters; uncertainty from one residual degree
of freedom; sensitivity inference from Q; unrecorded adaptive playback;
matched-budget/error/statistical/experimental Stage 6.5 claims; Stage 6.6
artifacts/plots/reporting; and adversarial in-process isolation guarantees.

Implementation completion requires generated regressions, documentation, package smoke, and independent reviews; this document completes only design.
