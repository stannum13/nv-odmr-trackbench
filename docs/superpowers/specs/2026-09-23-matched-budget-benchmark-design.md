# Matched-Budget Estimator Benchmark Design

## Purpose and status

Stage 6.5 performs the first causal, matched-acquisition comparison of the four
implemented estimator families:

1. repeated independent full-sweep fitting;
2. warm-started full-sweep fitting;
3. calibrated two-point center tracking; and
4. calibrated two-point tracking with periodic five-point linewidth fitting.

The primary matched resource is total integration time, including calibration.
Every result also reports observation count, nominal photon exposure,
signal-conditioned expected photons, realized photons when present, virtual
elapsed time, measurement latency, and estimator CPU time. The benchmark must
be capable of falsifying the sparse-tracking hypothesis; it does not assume a
winner. The comparable CPU field is explicitly
`post_calibration_update_cpu_time_s`; calibration fitting CPU is outside that
field for every method.

This design adds machine-readable comparison results and deterministic
scientific regressions. Stage 6.6 remains responsible for YAML-driven runs,
multi-seed uncertainty summaries, publication plots, committed result
artifacts, and README headline claims. Stage 6.5 therefore makes no general
accuracy, bandwidth, robustness, sensitivity, or superiority claim.

## Approaches considered

### Selected: dual-view causal evaluation

Each method emits same-epoch source releases plus explicitly labeled
release-aligned operational projections. A second evaluator constructs causal
sample-and-hold values on an explicit shared virtual-time checkpoint grid using
only public state already released at each checkpoint.

This preserves the scientific meaning of a completed sweep or sparse scan while
also measuring operational behavior between updates. It exposes cadence and
latency rather than hiding them inside a single pooled error number.

### Rejected: update-event metrics only

Scoring only when an estimator publishes rewards slow methods by omitting the
intervals between estimates. It is useful as one view, but insufficient as the
comparison.

### Rejected: checkpoint sample-and-hold metrics only

Checkpoint scoring measures operational tracking but erases whether center,
linewidth, and Q came from one same-epoch fit, a held calibration width, or
asynchronous center and width sources. That distinction is scientifically
material and remains explicit in the selected design.

### Rejected: force every method into one estimator interface

The reviewed sweep estimators and closed-loop evaluators have intentionally
different causal contracts. A new common estimator abstraction would either
weaken their provenance/resource guarantees or duplicate them. Stage 6.5 uses
method adapters that emit a common evaluator-owned result schema without
changing the estimator APIs.

## Scope boundary

Stage 6.5 includes:

- production linear linewidth drift composable with existing center dynamics;
- evaluator-only, release-gated truth sampling;
- a closed-loop full-sweep evaluator shared by repeated and warm estimators;
- immutable comparison event, checkpoint, resource, and summary records;
- pure deterministic metric functions;
- two-point and sparse adapters over their existing evaluator runners;
- one matched-integration orchestrator over four isolated instruments;
- exact static and deterministic center-plus-linewidth-drift regressions;
- one seeded-Poisson smoke that establishes reproducibility, not uncertainty;
- documentation of the result semantics and retained limitations.

Stage 6.5 excludes:

- YAML configuration and the final benchmark CLI;
- plots, report directories, or committed numerical result artifacts;
- multi-seed confidence intervals or ranking stability;
- slew-to-lock-loss curves, peak-collision identity scoring, or dropout
  reacquisition studies;
- real-data playback;
- lock-in and physics-informed filters;
- a claim that Q is magnetometric sensitivity.

## Architecture and ownership

```text
immutable scenario + method policy + budget
                    |
                    +--> fresh instrument + repeated-sweep adapter
                    +--> fresh instrument + warm-sweep adapter
                    +--> fresh instrument + two-point evaluator adapter
                    +--> fresh instrument + sparse evaluator adapter
                                      |
                         released estimator events
                                      |
                 evaluator-only truth at recorded event epochs
                                      |
       same-epoch + release-aligned errors + shared checkpoints
                                      |
                     per-ID and macro comparison summary
```

The planned production boundaries are:

- `dynamics/linewidth_drift.py`: deterministic composable linewidth motion;
- `evaluation/full_sweep/`: acquisition, timing, resource, and adapter records
  for repeated and warm full-sweep estimation;
- `evaluation/truth.py`: private evaluator-only release-gated truth authority;
- `metrics/tracking.py`: immutable error records and pure scalar summaries;
- `benchmark/matched_budget.py`: method adapters and four-method orchestration;
- `benchmark/types.py`: closed public comparison configuration/result records.

No estimator module imports `evaluation`, `benchmark`, instrument, dynamics,
truth, expected-photon, or full-observation types. Truth is sampled only by the
comparison evaluator after the corresponding public estimate release exists.

## Immutable configuration contracts

The public configuration is a frozen/slotted composition rather than an open
mapping:

```python
MatchedScenarioConfiguration(
    scenario_id: str,
    initial_snapshot: SpectralSnapshot,
    center_slew_hz_per_s: float | Mapping[str, float],
    fwhm_slew_hz_per_s: float | Mapping[str, float],
    noise_kind: Literal["gaussian", "poisson"],
    gaussian_stddev_at_1s: float | None,
    nominal_photon_rate_hz: float,
    frequency_overhead_s: float,
    seed: int,
)

MatchedAcquisitionConfiguration(
    frequency_hz: Sequence[float],
    integration_time_s: float,
    total_integration_time_s: float,
    checkpoint_timestamps_s: Sequence[float],
)

MatchedBudgetConfiguration(
    scenario: MatchedScenarioConfiguration,
    acquisition: MatchedAcquisitionConfiguration,
    fit: FitConfiguration,
    two_point: TwoPointTrackerConfiguration,
    sparse: SparseLinewidthConfiguration,
)
```

The frequency grid is used for both included calibration and later full sweeps.
The fit configuration is the exact common calibration/sweep model contract.
`fit.resonance_ids` must equal the initial snapshot's exact ordered IDs.
`two_point.identity_binding.mode` must be `require_expected_ids`, and its exact
ordered `expected_resonance_ids` must equal both `fit.resonance_ids` and the
initial snapshot IDs. Stage 6.5 rejects `adopt_fit_ids` rather than silently
replacing caller policy. `two_point.integration_time_s` and
`sparse.integration_time_s` must each equal
`acquisition.integration_time_s`. The private factory preserves the validated
tracker configurations, constructs `TwoPointBudgetCeiling` from
`acquisition.total_integration_time_s`, and supplies `included_same_run` only
through verified calibration construction. Neither tracker configuration is
treated as containing budget treatment or a ceiling; sparse identity order is
derived from the authenticated fit/calibration. Availability timestamps, clock
mapping, calibration resources, and starting sequence boundaries are derived
from authenticated acquired calibration rather than configured by the caller.

`noise_kind="gaussian"` requires finite nonnegative
`gaussian_stddev_at_1s`; `noise_kind="poisson"` requires it to be absent. The
first pass intentionally excludes empirical residual configuration, which
needs Stage 6.6 dataset/run provenance. Scenario mappings are copied into exact
immutable source-ID order. The private factory consumes only these records and
constructs four fresh dynamics, noise, RNG, instrument, estimator, and runner
graphs.

Stage 6.5 assumes resolved, noncrossing stable physical IDs. Preflight requires
initial resonance centers to be finite, positive, and strictly increasing in
exact `fit.resonance_ids` order. It computes the maximum nominal comparison
horizon from the configured checkpoints and successful block schedules, then
requires center order and positive finite FWHM at both `t=0` and that maximum
horizon. For linear motion, each adjacent center difference and each width is
affine, so endpoint validation proves the condition throughout the closed
interval. A touching/crossing, nonpositive, or nonfinite trajectory is rejected
before any instrument is constructed. Peak collision and identity reassignment
remain later scenario contracts.

## Comparison cohort and shared calibration

Every method runs on a distinct `ODMRInstrument` constructed from the same
immutable scenario configuration. Mutable dynamics, noise, instrument, runner,
tracker, or random-number-generator objects are never shared between methods.
The scenario factory recreates equivalent deterministic dynamics and uses the
same declared seed for the same noise policy. Identical seeds mean reproducible
method-local noise streams. Separate RNG objects initialized with one seed are
common-random-number coupled while their sampling calls remain aligned; adaptive
schedules and variable random consumption can desynchronize them. They are not
independent streams and do not imply paired physical photon draws. Intentional
common-random-number analysis versus method-derived independent sub-seeds is a
Stage 6.6 multi-seed design decision.

The first comparison treatment is exactly `included_same_run`:

- all four methods acquire the same ordered calibration frequency grid;
- every calibration observation and its integration time is charged;
- the grid, integration time, photon rate, overhead, noise configuration, and
  starting virtual clock are identical across isolated instruments;
- repeated and warm sweep methods must submit that acquired calibration sweep
  as their first causal estimator update, so warm state is seeded and repeated
  history has the same causal boundary;
- two-point and sparse methods use the exact verified calibration result
  required by their existing runners.

Every method emits one authenticated eight-identity `calibration_seed` release
batch at the calibration endpoint. Its center, FWHM, and Q values have the
closed semantic `same_epoch_calibration_fit` and are available to checkpoint
projection. All four calibration seed batches use the existing verified-
calibration physical epoch: the overflow-safe mean of the first and last actual
calibration integration midpoints. Full-sweep methods may use the ordered mean
of all actual midpoints only for later operational sweep batches. These two
conventions remain distinct and adjacent-ULP regressions prevent accidental
substitution.
The sweep methods use their first fit result; the interactive methods use their
verified calibration fit. A failed calibration produces a calibration-failure
outcome and no numerical seed batch. Calibration-derived events are therefore
included symmetrically rather than optional.

Calibration fit CPU is excluded from the primary comparable estimator-update
CPU total for all four methods because the existing interactive calibration
transaction does not isolate fit CPU from acquisition/evaluator work. Each
result labels the comparable CPU field `post_calibration_update_cpu_time_s`.
Calibration observation/integration/photon/virtual-time resources remain fully
included. Stage 6.5 does not compare a sweep calibration-fit CPU value against
an unavailable interactive calibration-fit CPU value.

Conditional pre-calibration remains representable by the low-level evaluator
APIs but is not a valid treatment in the first Stage 6.5 comparison.

## Primary budget and block stopping

`configuration.acquisition.total_integration_time_s` is the primary ceiling.
It includes calibration and every later accepted or physically returned
observation. It is finite, positive, and large enough to acquire the configured
calibration block for every method.

Each method declares the complete next acquisition block before querying:

- a full sweep is one ordered frequency-grid block;
- a fast center update is the existing two-query pair block;
- a due sparse linewidth update is the existing indivisible five-query block.

The adapter starts a block only when its complete nominal integration charge
fits within the ceiling under the repository's exact prospective-addition
rules. It never acquires a partial sweep, pair, or sparse scan merely to consume
the residual budget. A returned-but-unaccepted physical observation remains
charged according to the owning evaluator contract.

Each method result records:

```text
configured integration ceiling
actual charged integration
signed discrepancy = ceiling - actual charged integration
exactly_matched = discrepancy == 0.0
stopping block kind and nominal charge
```

The orchestrator never describes unequal actual charges as exactly matched.
The deterministic acceptance configuration uses one shared per-observation
integration time and constructs a reachable post-calibration schedule prefix.
With the default sparse cadence, one cycle is eight two-query pairs followed by
one five-query scan, or 21 observations; scanning all eight identities requires
168 observations. If the sweep contains `M` points, the acceptance prefix is a
positive multiple of `lcm(M, 168)`, which is also a pair boundary. The ceiling
is constructed by the same left-associated prospective additions used by the
resource ledger over calibration plus post-calibration observations; it is not
computed as `count * integration_time_s`. The acceptance factory simulates the
deterministic block schedule before a run and rejects a ceiling that is not an
exact clean boundary for every method. The general API still preserves and
reports block mismatch.

Equal integration time implies equal nominal exposure only when every method
uses the same public nominal photon rate. The result records that condition; it
does not infer equal signal-conditioned expected photons.

## Full-sweep evaluator

The full-sweep evaluator owns one exact ordered frequency grid and one
integration time. Frequencies are finite, positive, strictly increasing, and
contain enough distinct points for the existing fit configuration. A sweep
block is affordable or rejected before its first query.

Before every query the evaluator captures the exact pre-query virtual clock and
immutable frequency overhead. It computes the hidden physical midpoint using
the instrument's exact arithmetic association:

```text
pre_query_virtual_time + frequency_overhead + integration_time / 2
```

After return it authenticates the endpoint recurrence and before/after resource
boundaries. It never reconstructs the physical midpoint as
`endpoint - integration_time / 2`, because the two associations may differ by
an ULP. If an endpoint-derived public midpoint is retained for compatibility,
it is explicitly distinct from the physical truth-reference midpoint.

For every returned observation the evaluator retains:

- the full `InstrumentObservation`;
- the estimator-safe frequency, fluorescence, sequence, endpoint, integration,
  and nominal-exposure projection;
- expected and realized photon metadata in evaluator-only resources;
- the precomputed and authenticated physical integration midpoint.

A completed block becomes one `CompleteSweep` without sorting, interpolation,
dropping, or future access. Its public endpoint is the final observation
endpoint. `FullSweepTiming` retains both independently authenticated values:

```text
ordered_all_midpoints_reference_s
first_last_calibration_reference_s
release_sequence_index
release_timestamp_s
block_role = calibration | operational
```

The first included calibration block selects the overflow-safe first/last
reference for its `calibration_seed` batch. Every later operational sweep
selects the ordered binary64 mean of all actual midpoints. The truth authority
authenticates the selected reference against `block_role`; adjacent-ULP tests
prove that neither field can substitute for the other. The release time is the
final returned endpoint, and measurement latency is release minus the selected
truth reference.

The repeated adapter calls `RepeatedFullSweepEstimator.update_sweep` once per
completed block. The warm adapter calls
`WarmStartedFullSweepEstimator.update_sweep` once per completed block and
retains its documented warm/cold-attempt diagnostics. Every post-calibration
sweep update is timed around the update call and accumulated by arrival-order
addition independently of acquisition time. The calibration update seeds state
but is excluded from the comparable CPU total. Fit failure is a released event
with status and resource charge, not a missing sweep.

Full-sweep acquisition exceptions follow the same physical boundary used by the
interactive runners:

- an exception before an observation returns is retryable and uncharged;
- a malformed returned observation terminates the method run and retains any
  physically authoritative charge that can be authenticated;
- process-control `BaseException` values propagate identically after restoring
  only state that can still be restored.

The full-sweep evaluator is a supported `evaluation.full_sweep` API analogous
to the existing two-point and sparse evaluators, not a benchmark-only helper.
Its closed public phases are `ready`, `running`, `budget_stopped`,
`aborted`, and `externally_stopped`. A running state may contain one immutable
reserved sweep plus an accepted prefix of length zero through `M - 1`; it never
contains a second reservation. The public outcome union is:

- `FullSweepObservationAccepted` for a nonfinal returned point;
- `FullSweepReleased` for the final point plus estimator result;
- `FullSweepInstrumentFailure` for a retryable pre-return exception;
- `FullSweepBudgetStopped` selected before the first point of an unaffordable
  sweep;
- `FullSweepAborted` for a returned-but-unaccepted observation or an
  authenticated update/construction failure;
- `FullSweepExternallyStopped` with the exact retained partial prefix.

Each acquisition record joins pending query, full observation, estimator-safe
projection, precomputed physical midpoint, before/after instrument resources,
and sequence/endpoint recurrence. A resource builder independently replays the
accepted prefix and one optional unaccepted physical atom. An unavailable
physical join preserves raw evidence and returns no invented aggregate. The
runner exposes `step`, `run_until_event`, and `stop_external`; the loop continues
only across accepted/released outcomes and starts another sweep only after a
complete release. Exact instrument operation normally returns a valid record,
while injected instrument seams exercise malformed returns, construction
faults, retry, rollback, and `BaseException` behavior.

## Release-gated truth authority

Production truth evaluation is evaluator-only and requires a completed timing
record plus its exact release identity. It has no estimator-facing export.

The private scenario factory mints one non-public `MethodRunAuthority` per
method. A weak-lifetime binding joins the exact scenario configuration,
dynamics, noise, RNG, instrument, adapter/runner, estimator/tracker, and method
token identities. Full-sweep timing and release records are minted only by the
bound full-sweep evaluator. Two-point and sparse adapters must authenticate the
exact existing runner, current state, timing record, resource record, and
method binding before requesting truth. Equal copies, foreign runners, foreign
scenario objects, stale tokens, subclasses, and reconstructed records fail.

The authority lifecycle follows the reviewed provenance rules: successful live
runs own their bindings while global indexes are weak; discarded runs release
their complete instrument/dynamics graphs; partial registration failure rolls
back every private slot; dead or revoked tokens fail closed. No public
constructor, serialization path, copy, or `object.__new__` allocation can mint
authority.

For a source-aligned event, truth is sampled exactly once at the timing record's
`truth_reference_timestamp_s` only after the event is released. The authority
rejects incomplete timing, mismatched release sequence, a timestamp beyond the
release, a foreign scenario identity, and duplicate or reordered acquisition
evidence before invoking `snapshot_at`.

For an operational checkpoint, truth is sampled at the checkpoint timestamp
after the complete method run. The authority returns exactly one frozen
snapshot per `(method_run_identity, checkpoint_index)`; all identity and
quantity scores at that checkpoint derive from it. This is post-hoc scoring,
never an estimator input. The checkpoint evaluator selects only public active
state whose release timestamp is less than or equal to the checkpoint. Later
estimates cannot influence an earlier checkpoint.

Estimator graphs are recursively checked in tests for retained truth,
instrument, evaluator, future, callback, full-observation, expected-photon, or
dynamics capability.

## Deterministic linewidth dynamics

`LinearLinewidthDrift(base_dynamics, reference_fwhm_hz,
fwhm_slew_hz_per_s)` composes over any `SpectralDynamics` and replaces only
each resonance FWHM:

```text
fwhm_i(t) = reference_fwhm_i + slew_i * t
```

The reference and slew may be one scalar or exact immutable ID mappings. The
constructor validates exact ID coverage, finite values, and positive reference
widths. `snapshot_at` reuses the repository timestamp validation, calls the
base dynamics exactly once, preserves tuple order, IDs, centers, amplitudes,
eta, and baseline exactly, and rejects generated nonpositive or nonfinite
widths. It has no stochastic state.

The private matched-scenario factory derives `reference_fwhm_hz` from
`scenario.initial_snapshot` in exact resonance tuple/ID order; callers cannot
supply a second conflicting benchmark reference map.

The Stage 6.5 drift acceptance scenario composes `LinearCenterDrift` with
`LinearLinewidthDrift`. This deterministic scenario isolates estimator cadence
and model behavior from Monte Carlo variation. A separate seeded-Poisson smoke
checks reproducibility and resource accounting without supporting a statistical
performance claim.

## Release batches and common event schema

One physical estimator update first creates an immutable acyclic record graph:

```text
BenchmarkReleaseHeader  <- exact parent of each ReleasedEstimate
        |
BenchmarkReleaseBatch(header, source_releases, operational_projections,
                      post_update_projection/status)
```

The public header owns one timing/resource boundary, one CPU charge, release
sequence, and public scalar batch index. It contains no authority, token,
instrument, evaluator, or scenario capability. Each `ReleasedEstimate`
retains the exact already-constructed header, never the enclosing batch. The
batch then owns the header and child tuples. This avoids cyclic frozen
construction. A weak private registry binds exact header and child identities
to their exact method-run authority/batch transaction; scalar batch indices are
display/serialization metadata and never authority. Any internal construction
token stays only in the private adapter/authority graph.

Because frozen value records may be equal, this registry is never equality-
keyed. It uses an identity key plus weak reference and callback cleanup, and
every lookup verifies `referent is supplied_object` before granting authority.
ID reuse or an equal foreign record therefore cannot authenticate.

The batch owns an exact post-update public projection/status and two distinct
tuples: `source_releases` for genuinely refreshed quantities and
`operational_projections` for held/recomputed state at release.

- a successful sweep batch source-releases all eight same-epoch fits and makes
  them the operational projection;
- a two-point pair source-releases only the tracked center; held calibration
  width and held-width Q are operational projections;
- a sparse fast pair source-releases only fast center; asynchronous live Q is
  an operational projection and active width remains held;
- a successful sparse scan source-releases diagnostic scan-local center/FWHM/Q
  plus the operational active-width refresh, then projects refreshed live Q;
- a failed sparse scan source-releases no width and projects the exact retained
  prior operational width/live state;
- stale identities present in a state snapshot are not duplicated as new
  source releases.

A failure batch may have empty numerical tuples while still retaining the exact
post-update status/projection needed for checkpoint state transitions. Thus
checkpoint construction replays release batches, not merely the latest
successful `ReleasedEstimate`.

Acquisition, photons, virtual time, and CPU live only on the batch and are never
copied into eight additive per-ID records. The truth authority samples one
`SpectralSnapshot` per successful same-epoch release batch and scores every
same-epoch value in that batch from the snapshot. It does not call
`snapshot_at` once per resonance. Batch identity, timing, release, truth, and
resource joins are authenticated before per-ID values are scored.

If a batch also releases an asynchronous operational projection, the authority
may additionally sample exactly once at that batch's release timestamp. All
release-aligned quantities share that snapshot. Thus lookup cardinality is one
per unique authenticated `(batch, scoring_timestamp_kind)`, never one per
identity or quantity.

Each batch contains immutable `ReleasedEstimate` records with exactly one
quantity (`center_hz`, `fwhm_hz`, or `q`) and:

- method, stream key, and event kind;
- resonance ID and stable source ordering;
- released status and explicit unavailable/failure reason;
- one value and one closed `quantity_semantics` value;
- zero, one, or two source-reference timestamps and source epochs;
- release timestamp plus the exact parent `BenchmarkReleaseHeader` identity.

A separate immutable `ScoredEstimate` contains the exact released object plus
the evaluator truth timestamp/value and signed error. Scoring never mutates or
value-copies a frozen release record. A private authority binding requires exact
released-record identity before constructing the scored record.

Closed quantity semantics are:

- `same_epoch_calibration_fit` for every method's calibration seed center,
  FWHM, and Q;
- `same_epoch_sweep_fit` for repeated and warm sweep center/FWHM/Q;
- `tracked_center` for the two-point center;
- `held_calibration_fwhm` for two-point linewidth;
- `held_width_live_q` for two-point center divided by calibration FWHM;
- `sparse_scan_local_fit` for five-point local center/FWHM/Q;
- `active_sparse_fwhm` for the operational width refreshed only by a successful
  sparse scan and otherwise held;
- `asynchronous_sparse_live` for current fast center divided by active sparse
  FWHM with separate source epochs.

Closed stream keys are:

- `calibration_seed`;
- `sweep_operational`;
- `two_point_operational`;
- `sparse_operational`;
- `sparse_scan_diagnostic`.

Checkpoint selection starts from the exact `calibration_seed` batch as the
authenticated initial active projection, then moves only through the
operational stream declared for the method. In particular, a newer
`sparse_scan_diagnostic` center never replaces the `sparse_operational` fast
center; the production no-feedback rule remains observable in the benchmark.

The two-point held width and its derived Q are never labeled as updated
linewidth or same-epoch Q. Sparse same-epoch FWHM/Q accuracy uses completed
five-point scan results. Sparse asynchronous live Q is reported separately and
cannot replace the scan-local Q summary.

Same-epoch calibration, sweep, and scan-local quantities have one source
timestamp. Both `held_width_live_q` and `asynchronous_sparse_live` retain
separate center-source and width-source timestamps/epochs and forbid a singular
source timestamp or source age. Their release-aligned operational errors are
scored against truth at batch release, and their checkpoint operational errors
against truth at the checkpoint. Both report separate center age, width age,
and release age; neither is called source-aligned same-epoch Q.

All eight resonance identities remain present in method results. Adapters do
not reorder by estimated frequency or silently relabel a crossing.

## Shared causal checkpoints

`configuration.acquisition.checkpoint_timestamps_s` is an explicit finite,
strictly increasing candidate tuple. The acceptance factory derives it before
any method runs from the immutable scenario, complete-block schedules, budget,
integration time, and overhead. Validation requires every checkpoint to lie at
or before the minimum nominal successful terminal time computed before
execution. The full configured tuple is the scoring denominator. Actual early
failure or termination never truncates this grid for any method.

At checkpoint `t`, each method uses its latest authenticated public active-state
projection with `release_timestamp_s <= t`:

- repeated sweep uses an explicit benchmark-evaluator policy that causally
  holds the last successfully released sweep output across later failed
  sweeps; this is not described as estimator-owned active state, although the
  successful output remains in evaluator-visible history;
- warm sweep uses the exact `active_fit` and original source update retained by
  `WarmSweepEstimate`;
- two-point uses the public tracked center plus held calibration width, while
  retaining lost/step-limited policy state;
- sparse uses the public fast center plus active sparse width; a failed sparse
  fit retains its prior width exactly as the tracker does.

Failure/update outcomes are recorded separately and never masquerade as new
successful source samples. After a terminal method event, the last public active
projection, if one exists, remains sample-and-held through the fixed nominal
horizon with terminal status and increasing ages visible. If no active
projection exists, that method/identity/quantity is unavailable at the
checkpoint; the denominator is unchanged.

The held value is scored against truth at `t`. Same-epoch quantities report
`source_age_s = t - source_timestamp_s`; asynchronous quantities report
separate center-source age, width-source age, and release age and forbid a
singular source age.

Same-epoch source-aligned metrics and checkpoint sample-and-hold metrics are
stored in separate summaries. They are never pooled. Actual completed coverage,
terminal status, held-after-terminal counts, and unavailable counts are
reported per method without changing the configured checkpoint denominator.

## Metrics and aggregation

Public aggregation accepts frozen `ErrorSample` records, not bare floats. Only
the evaluator-owned scoring factory may construct one, from an exact privately
authenticated `ScoredEstimate`. The sample must echo the scored record's exact
resonance ID, quantity, unit, semantic, evaluation view, stream key, and signed
error; equal foreign copies fail. A closed compatibility matrix over
`(quantity, semantic, view, stream_key)` additionally requires:

- `same_epoch_calibration_fit`: center/FWHM/Q with matching units and
  `source_aligned` on `calibration_seed`, or `checkpoint` when that exact seed
  on `calibration_seed` remains the authenticated initial active projection;
- `same_epoch_sweep_fit`: center/FWHM/Q with matching units and
  `source_aligned` on its sweep release, always `release_aligned` on
  `sweep_operational`, or `checkpoint` on `sweep_operational`;
- `tracked_center`: center/Hz with `source_aligned` on the owning fast-pair
  release, or `release_aligned`/`checkpoint` on `two_point_operational` or
  `sparse_operational` as applicable;
- `held_calibration_fwhm`: FWHM/Hz with `release_aligned` or `checkpoint` on
  `two_point_operational` only; the original calibration measurement uses
  `same_epoch_calibration_fit`;
- `held_width_live_q`: Q/dimensionless with `release_aligned` or `checkpoint`
  on `two_point_operational` only;
- `sparse_scan_local_fit`: center/FWHM/Q with matching units and
  `source_aligned` on `sparse_scan_diagnostic` only;
- `active_sparse_fwhm`: FWHM/Hz with `release_aligned` or `checkpoint` on
  `sparse_operational` only;
- `asynchronous_sparse_live`: Q/dimensionless with `release_aligned` or
  `checkpoint` on `sparse_operational` only.

No diagnostic sparse-scan center or Q is admitted into an operational
checkpoint stream.

Every `ErrorSample` also exact-echoes method, exact method-run identity,
comparison/scenario identity, stream key, and resonance ID from its authenticated
`ScoredEstimate`.

Typed per-ID aggregation requires one exact method-run identity, comparison/
scenario identity, method, stream key, resonance ID, quantity, unit, semantic,
and view. Pooled aggregation may vary resonance ID only within the exact
configured eight-ID set; it retains per-ID counts and a `pooled` label. Macro
aggregation consumes exactly eight completed per-ID summaries with one exact
method-run/comparison/method/stream/quantity/unit/semantic/view and the exact
configured eight-ID set; it never aggregates raw mixed samples. Cross-method
aggregation is not a scalar error summary, and cross-run/seed uncertainty is a
separate Stage 6.6 contract. Mixed groups fail before arithmetic. A private
scalar kernel receives only the validated signed-error tuple.

For signed finite errors `e_j = estimate_j - truth_j`, let
`scale = max(abs(e_j))`. An all-zero sample returns canonical positive zero for
all five metrics. Otherwise pure metric functions compute deterministic scaled
values in frozen sample order:

```text
bias = scale * (fsum(e_j / scale) / n)
MAE = scale * (fsum(abs(e_j / scale)) / n)
RMSE = scale * sqrt(fsum((e_j / scale)^2) / n)
normalized_bias = fsum(e_j / scale) / n
standard deviation = scale * sqrt(
    fsum((e_j / scale - normalized_bias)^2) / n
)
95th-percentile absolute error = deterministic linear percentile of abs(e_j)
```

The standard deviation is the population value for the evaluated event set;
no Bessel correction is implied. Percentile interpolation is fixed to the
NumPy `quantile(..., method="linear")` convention and regression-tested at
exact ranks and neighboring values. Scaling prevents avoidable square overflow
for representable finite results. The final multiplication must itself remain
finite or the metric is unavailable with `nonrepresentable_metric`; it is never
clipped. Nonfinite inputs, booleans, empty samples, and inconsistent units are
rejected rather than silently filtered. Extreme finite values, cancellation,
permutation behavior, and signed-zero canonicalization are regression-tested.

Metrics are computed separately for center Hz, FWHM Hz, and dimensionless Q,
and separately for every quantity semantic and evaluation view. Per-ID
summaries are authoritative. The primary eight-resonance aggregate is the
unweighted macro mean of exactly eight per-ID metric values. If any identity
lacks a valid per-ID metric, the primary macro metric is unavailable with the
missing identities recorded; it is never recomputed over a smaller successful
subset. A pooled available-sample summary may also be reported, clearly labeled
`pooled` with complete eligible/available/failure counts; it never replaces the
macro result. Photon-weighted error is outside the first Stage 6.5 result.

Q truth is always `truth_center_hz / truth_fwhm_hz` at the applicable truth
timestamp. It is never interpreted as sensitivity. Width and Q errors for a
held calibration width are labeled operational held-width errors.

## Failures, missingness, and lock language

Failures are never silently dropped. Every method result records distinct units:

- scheduled, returned, accepted, unaccepted, and charged observation counts;
- reserved, completed, failed, partial, and terminal acquisition-block counts
  by sweep/pair/scan kind;
- release-batch and per-ID quantity-record counts;
- scientific-failure, validation-failure, software-failure, abort, and
  unavailable checkpoint counts at their owning block/update level;
- checkpoint availability numerator and declared denominator;
- first terminal reason and causal boundary;
- identities and quantities affected.

Numerical error metrics use only finite available estimates, but every summary
also carries its eligible, available, failed, and unavailable counts. If no
sample is available, the metric fields are absent with a closed reason; zeros
are not fabricated. A stale last-known estimate may be used at a checkpoint
only when the method's public active state retains it or the declared repeated-
sweep evaluator hold policy retains an already released success; its original
source and increasing age remain visible.

Estimator policy states such as `locked` or `lost` remain policy diagnostics.
Stage 6.5 does not call them truth-certified lock. Truth-based lock-loss,
reacquisition, collision identity, and maximum-slew metrics require later
scenario-specific contracts.

## Resource and latency records

Each `BenchmarkMethodResult` contains both accepted and final physically charged
resources where its evaluator distinguishes them. The common resource record
includes:

- observations;
- total integration time;
- nominal photon exposure;
- expected photons;
- realized photons and missing-count observations;
- virtual elapsed time;
- calibration and post-calibration partitions;
- complete/partial pair, sweep, and sparse-scan counts;
- benchmark-defined estimator update CPU total and any estimator-reported CPU
  total;
- per-release-batch observations, integration, photons, virtual time, and CPU.

Per-quantity `ReleasedEstimate` records carry no additive resource fields and
must never be summed to reconstruct method resources.

Expected and realized photons are evaluator outputs only. Estimators receive no
photon expectation, truth, or scenario object. CPU time is measured separately
from acquisition and is not added to virtual instrument time. The common CPU
boundary is the estimator update call only: the repeated and warm full-sweep
adapters time `update_sweep`, while two-point and sparse adapters reuse their
tracker-owned update clocks rather than timing instrument queries or the outer
runner. Warm-sweep estimator-reported CPU must be less than or equal to its
enclosing adapter measurement within declared clock resolution. Interactive
estimator-reported CPU is itself the benchmark-defined value because the
existing runner isolates that boundary. Both fields are retained where both
exist and are never added together. Tests inject clocks; production uses the
documented process CPU clock. The primary method CPU total begins after the
calibration seed release for every method. Realtime interpretation
reports measurement latency, integration, query count/overhead, and compute
latency separately.

Resource totals use the existing exact arrival-order ledger behavior. The
comparison layer replays each method trace independently and requires the replay
to equal its evaluator's final record before constructing a result.

## Orchestrator transaction

`run_matched_budget_comparison(configuration)` performs these steps:

1. validate the complete immutable configuration without constructing an
   instrument;
2. create four isolated scenario/instrument graphs;
3. acquire and validate included same-run calibration for each method;
4. run each adapter to its complete-block integration ceiling or terminal
   event;
5. freeze the four method traces and resources;
6. perform release-gated same-epoch source-aligned and release-aligned
   operational truth scoring;
7. build checkpoint sample-and-hold records;
8. compute per-ID, macro, and optional pooled summaries;
9. verify budget/resource joins and publish one immutable comparison result.

Method execution order is fixed and recorded but cannot change deterministic
scenario configuration. A method failure does not prevent later methods from
running on their isolated instruments. Configuration/construction failures
before method execution fail the comparison atomically. An ordinary exception
after a method begins becomes that method's terminal software-failure result
only when its physical/resource boundary can be authenticated; otherwise the
comparison fails closed. Process-control `BaseException` values propagate
identically.

The result stores the package version, seed, complete configuration, method
order, and deterministic scenario identifier. Git hash, YAML serialization,
run-directory layout, plots, and repeated-seed uncertainty belong to Stage 6.6.

## Public surface and immutability

Stage 6.5 exports the approved benchmark configuration and immutable result
records, pure metric functions, production linewidth dynamics, one
matched-budget entry point, and the full-sweep evaluator's phases, outcomes,
runner, timing, and resource builder from `evaluation.full_sweep`. Benchmark
method adapters, truth authority, run-provenance tokens, and scenario factories
remain private.

All public records are frozen and slotted. Input mappings and sequences are
defensively copied into canonical immutable order. Scalar validation rejects
booleans, nonfinite values, and lossy implicit array conversion. Closed strings,
statuses, method names, quantity semantics, failure reasons, and aggregation
kinds reject unknown values. Equality never substitutes for required identity
joins inside evaluator transactions.

No existing Stage 6.1–6.4 public signature, record meaning, state transition,
resource boundary, or export is changed.

## Determinism and first acceptance cases

The exact-static case uses noiseless deterministic observations and constant
centers/widths. It establishes acquisition, release, truth, identity, resource,
metric, and budget joins. It is not a noise-floor claim.

The center-plus-linewidth-drift case uses deterministic linear motion small
enough that all methods remain in their documented model domains. It establishes
different cadence and quantity semantics without claiming a winner.

The seeded-Poisson smoke runs the same immutable configuration twice per method
and requires identical observations, non-CPU traces, acquisition resources,
and metric summaries after projecting out production-clock CPU fields. CPU
fields must be finite and nonnegative. A separate injected deterministic-clock
test requires complete equality including CPU. The one seed is not used for
uncertainty bars, hypothesis acceptance, or ranking.

The acceptance target includes all eight resonances and requires at least one
post-calibration same-epoch sweep event for both sweep methods, at least one
completed pair for every two-point identity, and at least one completed sparse
scan for every sparse identity. The configured integration ceiling is chosen to
make those requirements possible without special-casing the scheduler.

## Testing and review gates

Unit tests cover:

- linewidth-drift validation, composition, and generated physicality;
- immutable comparison records and every closed discriminator;
- scalar metric definitions, order, percentile convention, and invalid input;
- full-sweep block affordability, timing, truth release, resources, failures,
  CPU accounting, and rollback;
- two-point/sparse adapter projections without bypassing their runners;
- held-width and asynchronous-Q labeling;
- checkpoint causality and no future-event selection;
- failure/missingness denominators and macro versus pooled aggregation;
- exact and inexact budget records;
- identical seed/config reproducibility and isolated object graphs;
- recursive absence of truth/full-resource capability in estimators.

Differential tests require unchanged Stage 6.1–6.4 traces and public exports.
Scientific review audits matched resources, truth epochs, Q semantics, failure
inclusion, and claim language. Software review audits immutability, closed
unions, identity joins, rollback, `Exception` versus `BaseException`, weak
provenance lifetime, deterministic folds, and wheel contents.

The final Stage 6.5 gate runs focused predecessor compatibility suites, the
complete repository twice, Ruff, an isolated wheel build/install/import smoke,
diff validation, and clean status. Only then may the project state say Stage
6.5 is implemented. Numerical performance conclusions remain pending Stage 6.6
multi-seed artifacts even if deterministic Stage 6.5 fixtures show a difference.

## Retained scientific limitations

- The simulator begins with eight electronic resonances and omits hyperfine
  substructure; unresolved hyperfine can bias fitted FWHM and Q.
- Linear center/linewidth drift is a controlled scenario, not a complete NV
  ensemble dynamics model.
- Pseudo-Voigt FWHM is an effective lineshape width, not intrinsic decoherence.
- Sparse five-point fits can be biased by baseline curvature, non-target tails,
  peak approach, and within-scan dynamics.
- Full-sweep truth reference at the mean sample midpoint is a declared scoring
  convention, not proof that one spectrum existed instantaneously there.
- Identical seeds across adaptive schedules do not create paired physical photon
  histories.
- CPU timing is workstation-dependent and is reported, not normalized into a
  hardware-independent realtime claim.
- Q alone is not magnetometric sensitivity; contrast, slope, photon rate, and
  noise remain separately relevant.
