# Warm-Started Causal Sweep Fitting Design

## Purpose

Stage 6.2 adds a practical completed-sweep baseline: use only an earlier
successful public fit to initialize the next causally submitted sweep. It tests
whether warm starting preserves the Stage 6.1 fit definition while reducing
optimizer work on deterministic noncrossing drift. It does not claim higher
temporal bandwidth, lower measurement cost, or realtime superiority.

## Approaches considered

Three designs were considered:

1. A stateful wrapper around unchanged `fit_spectrum` is selected. It owns
   causal prior state, prepares a compatible guess, records compound attempts,
   and can retry cold without coupling temporal policy into the oracle.
2. Fitter-native temporal state would make `fit_spectrum` accept prior results.
   This obscures its single-sweep oracle contract and cannot represent both a
   failed warm attempt and a cold recovery cleanly.
3. Two-success velocity extrapolation could tolerate more slew, but it is a
   predictor rather than the required warm-start baseline and would confound
   later tracker comparisons.

Running warm and cold on every sweep and selecting by hindsight is rejected:
it doubles compute and gives the estimator an undeclared model-selection
advantage. Stage 6.2 runs cold only when no compatible prior is available or as
one explicit recovery attempt after a start-dependent warm failure.

## Architecture

```text
current completed sweep
          +
latest earlier successful public fit
          ↓
causal endpoint validation
          ↓
baseline rebase + compatibility check
          ↓
warm fit ──failure eligible for recovery──→ one cold fit
          ↓
attempt record + current result + explicitly aged active result
```

The Stage 6.1 fitter remains stateless. `WarmStartedFullSweepEstimator` owns
only immutable configuration, earlier public outputs, causal endpoints, and
history. It never receives hidden dynamics, truth, future sweeps,
signal-conditioned expected photons, or evaluator references.

## Public records

All new public records are frozen, slotted, validated, and exported from
`odmr_bench.estimators`.

```python
SweepFitAttempt(
    start_kind,                  # "preflight", "cold", or "warm"
    warm_source_update_index,    # present only for warm
    fit,                         # complete SpectrumFitResult
    cpu_time_s,                  # non-negative process CPU time
)

WarmSweepEstimate(
    update_index,
    attempts,                    # one, or warm then cold
    warm_start_disposition,      # exact values below
    warm_start_rejection_code,
    warm_start_message,
    active_fit,                  # current success or older success, else None
    active_source_update_index,
    estimate_age_submitted_observations,
    estimate_age_sequence_indices,
    estimate_age_s,
    observation_count,
    cumulative_observation_count,
    first_sequence_index,
    last_sequence_index,
    last_timestamp_s,
    total_integration_time_s,
    total_nominal_exposure_photons,
    cpu_time_s,
)
```

`warm_start_disposition` is one of `no_successful_prior`, `used`,
`rejected_age`, `rejected_compatibility`, or `not_applicable_preflight`.
Rejection codes are closed to `age_limit_exceeded`,
`baseline_rebase_unrepresentable`, `center_outside_sweep`,
`center_separation_incompatible`, `resonance_bounds_incompatible`, and
`parameterization_unrepresentable`. `rejected_age` requires exactly
`age_limit_exceeded`; `rejected_compatibility` requires one of the other five
codes; `no_successful_prior`, `used`, and `not_applicable_preflight` require no
code. `warm_start_message` is present only with a rejection code and explains
it but is not a machine-readable key. A
rejected disposition produces one cold attempt. `used` produces a warm attempt
and optionally a second cold attempt. A start-independent fitter preflight
failure produces one `preflight` attempt and `not_applicable_preflight`.

`current_fit` is a derived property returning `attempts[-1].fit`; it never hides
a failed update. `active_fit` supports continuous
consumption: it is the selected current success, or the most recent older
success after a failed update. In the latter case `is_stale=True`, the active
source is explicit, and age is positive. With no success, `active_fit` and its
source/ages are `None` and `is_stale=False`.

`is_stale` is derived from current/active success and source indices.
`total_nfev` is a derived sum across retained attempts. Acquisition resources are
recorded once from the submitted sweep even when computation is retried.
Per-attempt and whole-update CPU use a patchable module-level
`time.process_time_ns` clock. Whole-update time covers validation, preparation,
and fit calls through the instant immediately before record construction and
must be at least the attempt sum. Timing is diagnostic, machine-dependent, and
excluded from deterministic equality assertions.

Constructors enforce the following cross-field rules without array-valued
dataclass equality: attempts contain exactly one item or exactly `warm, cold`;
cold/preflight attempts have no warm source; warm requires a smaller source
index; `used` begins warm; non-used ordinary dispositions contain one cold;
preflight disposition contains one preflight; rejection dispositions require
one closed code; `active_fit`, when present, is successful. Current success
uses itself as active at the current zero-based update index, submitted-
observation age zero, sequence-index age zero iff sequence metadata exists, and
seconds age zero iff timestamp metadata exists; unavailable bases remain `None`.
Current failure with active output uses an older source, positive submitted-
observation age, and positive sequence/time age when those bases exist, with
unavailable bases `None`. No
active fit implies no active source or ages. Resource metadata is copied from
the submitted sweep by the wrapper and regression-tested there; the standalone
record constructor validates its scalar domains but cannot inspect the source
sweep. Derived fields cannot be supplied inconsistently.

The estimator interface is:

```python
class WarmStartedFullSweepEstimator:
    def __init__(
        self,
        configuration: FitConfiguration,
        *,
        retry_cold_on_warm_failure: bool = True,
        max_warm_start_age_updates: int | None = None,
    ): ...

    @property
    def configuration(self) -> FitConfiguration: ...
    @property
    def latest(self) -> WarmSweepEstimate | None: ...
    @property
    def history(self) -> tuple[WarmSweepEstimate, ...]: ...
    @property
    def latest_success(self) -> WarmSweepEstimate | None: ...
    def update_sweep(self, sweep: CompleteSweep) -> WarmSweepEstimate: ...
    def reset(self) -> None: ...
```

Invalid input and noncausal metadata neither call the fitter nor advance state.
`latest_success` is the update that produced the active successful fit, even
when `latest` is a later failed update.

## Warm-start preparation

Only `SpectrumFitResult.success=True` may seed a future sweep. Its model kind,
baseline degree, ordered IDs, resonance bounds, and line-shape semantics must
match the immutable configuration.

The public baseline is rebased from old reference `r` to the new sweep midpoint
`r'`. With `d = r' - r`:

```text
intercept' = intercept + slope*d + quadratic*d²
slope'     = slope + 2*quadratic*d
quadratic' = quadratic
reference' = r'
```

The transformation preserves the same polynomial. It uses the fitter's exact
overflow-safe midpoint convention, `f_min/2 + f_max/2`. Zero slope and
quadratic terms short-circuit before computing an unnecessary reference
difference. Otherwise a finite materialized reference difference is required; nonzero
products use the Stage 6.1 exponent-aware product/ratio helper so underflow or
overflow is explicit, and final sums use `math.fsum`. An exactly zero final
intercept or slope caused by cancellation among representable float terms is
valid. The implementation conservatively rejects a nonfinite reference
difference or any individually unrepresentable product even when symbolic
cancellation might yield a finite coefficient. Such products or sums produce
`baseline_rebase_unrepresentable`; no arbitrary-precision public coefficient is
invented.

The prepared `FitInitialGuess` keeps configured ID order and the earlier
center, FWHM, amplitude, and eta values. It is compatible only when every
center lies inside the new sweep, centers remain strictly ordered and meet the
configured minimum separation, widths/amplitudes/eta remain valid, the rebased
baseline is representable, and the Stage 6.1 parameterization can build
strictly feasible center/resonance bounds. A changed grid or span is allowed;
exact grid equality is not required.

Warm preparation returns a typed internal `WarmStartPreparation` containing
either the immutable guess or one closed rejection code/message. Guess
validation is a single package-internal helper shared with `fit_spectrum`, not
a duplicate of its checks. A prepared-compatible guess therefore cannot later
raise a known compatibility `ValueError`; unexpected errors still propagate.
Known incompatibility becomes `rejected_compatibility` and a cold attempt.
Unexpected programming errors propagate and do not mutate history.

Before compatibility preparation, the wrapper calls the same package-internal
start-independent sweep-preflight helper used by `fit_spectrum`. It returns
either no failure or the exact `insufficient_samples`/`uninformative_sweep`
result. This prevents an uninformative sweep from being misclassified as prior
incompatibility and avoids duplicating fitter logic.

## Causal state machine and recovery

The estimator is configured with
`retry_cold_on_warm_failure: bool = True` and
`max_warm_start_age_updates: int | None = None`. Boolean values are not accepted
as integers; a configured value must be positive. `None` is the scientifically neutral default and disables the
seeding-age limit; explicit finite values are benchmark conditions. Age one
means the immediately preceding update. Age rejection affects seeding only and
never removes an older success from `active_fit`.

For update `k`:

1. Validate the sweep and any declared endpoint before fitting. API call order
   is causal even without external timestamps.
2. If sequence indices are supplied in a run, every sweep must supply one and
   declares consecutive observations. Derive
   `first_sequence_index=last_sequence_index-observation_count+1` and require it
   to be non-negative and exceed the prior last index, preventing overlap. Timestamp availability
   is likewise consistent; supplied timestamps increase strictly. Callers reset
   between recordings.
3. With no successful prior, run one cold fit.
4. Reject an over-age or incompatible prior and run one cold fit.
5. Otherwise run one warm fit using the exact rebased prior.
6. If the warm result is `optimization_failed` or `quality_failed` and cold
   retry is enabled, run exactly one cold fit
   on the same already acquired sweep. Do not retry start-independent
   `insufficient_samples` or `uninformative_sweep`.
7. Select the warm result when it succeeds; otherwise select the cold retry
   when present. Retain both attempts even if both fail.
8. Promote only a selected successful current fit. Failed current updates keep
   the older successful active fit, explicitly stale and aged.
9. A fitter result of `insufficient_samples` or `uninformative_sweep` is
   start-independent: record it as one `preflight` attempt with no warm source,
   and prepare or claim no warm guess for it.
10. Append state only after every required fit/record validates. An unexpected
   exception leaves history, prior, endpoints, update index, and cumulative
   observations unchanged and aborts the benchmark run; a harness must not
   skip that acquired sweep and continue as if it were recorded successfully.

Cold reacquisition assigns configured IDs left-to-right. Warm and cold paths
therefore assume eight resolved, noncrossing components. Ordered IDs are not
evidence of physical identity through a collision.

## Age and resource semantics

The estimator maintains a deterministic cumulative submitted-observation
count, exposed as the endpoint when external sequence indices are absent. A
current success has every available age zero. After failure:

```text
estimate_age_submitted_observations =
    cumulative_observations_now - cumulative_observations_at_active_success

estimate_age_sequence_indices =
    current_last_sequence_index - active_source_last_sequence_index

estimate_age_s =
    current_completion_timestamp - active_success_timestamp
```

Sequence-index age is `None` without external indices. Seconds age is `None`
when timestamps are unavailable; neither is inferred. The two observation-age
bases are never silently mixed.
The current sweep's `observation_count`, integration time, nominal exposure,
and endpoint metadata are copied unchanged. Signal-conditioned expected photons
remain hidden evaluator data; realized counts are public when an observation
model exposes them, but the current `CompleteSweep` does not retain their
aggregate. Evaluator joins for realized counts and virtual elapsed time remain
Stage 6.5 work. A cold retry consumes CPU and optimizer evaluations but no
additional observations, integration time, or photon exposure.

## Failure and reset behavior

A failed warm attempt remains visible even after cold recovery. A current
failure remains visible even while `active_fit` exposes a stale prior. Fit
failure codes retain their Stage 6.1 meaning; the wrapper does not relabel them.
`start_kind="preflight"` occurs if and only if the fit is unsuccessful with
`insufficient_samples` or `uninformative_sweep`, has `nfev=0`, and used no
initial guess. Those failure codes never appear as cold or warm attempts.

`reset()` removes history, successful-prior state, endpoint-availability mode,
last endpoints, cumulative observations, and update numbering. Configuration
is retained.

## Verification contract

Generated fixed-seed, noncrossing center-drift sweeps must establish:

- first update is cold and the next compatible successful update is warm from
  the exact preceding public success;
- linear and quadratic baselines rebase algebraically to a changed midpoint;
- rebasing accepts exact-zero cancellation and handles extreme signed
  references plus representable and unrepresentable products explicitly;
- compatible changed grids work and incompatible span/bounds produce one cold
  attempt with an explicit disposition;
- success, failed update, then success uses the most recent eligible successful
  source and exposes exact stale submitted-observation, sequence-index, and time
  age during failure;
- warm failure followed by cold success records both attempts and promotes only
  the cold result;
- start-independent failures are not retried;
- age rejection uses cold only;
- IDs never come from a failed result or truth and remain configured/ordered;
- warm and cold estimates recover center/FWHM/Q within declared fixture bounds;
- overlapping/nonmonotonic indexed sweeps and inconsistent endpoint availability
  are rejected before fitting; injected exceptions leave all state unchanged
  and abort the run;
- record constructors reject contradictory attempt/disposition/active-source/
  age/resource combinations and derived fields are exact;
- a mocked CPU clock pins attempt sums and whole-update aggregation while real
  CPU values receive only finite nonnegative checks;
- reset clears every causal/provenance field; and
- identical logical runs match apart from CPU timing and platform-sensitive
  SciPy iteration/status details.

Tests report optimizer evaluations and CPU descriptively. They do not assert a
universal speedup. The Stage 6.2 demonstration may compare fixed-fixture cold
and warm `nfev`, but any observed difference is a regression result, not a
headline performance conclusion.

Every controlled drift regression constructs each complete sweep from one
frozen spectral snapshot at its declared completion/reference timestamp. Cold
and warm estimators receive the exact same immutable sweeps and configuration.
These are completed-sweep estimator tests, not event-driven within-sweep motion
or virtual-instrument realtime results.

## Documentation and completion boundary

`docs/estimators.md` will describe warm-source provenance, baseline rebasing,
conditional cold recovery, stale estimates, age, center-box limitations, and
resource accounting. It calls `cpu_time_s` the measured update-core process CPU
interval and states that record construction/state append are outside it. A download-free example will process a short generated
drift sequence and print attempt/source/age diagnostics without claiming
realtime performance.

Stage 6.2 is complete when the wrapper, records, preparation helper, generated
causal regressions, docs/example, full tests, Ruff, build, and installed-wheel
smokes pass independent task and integrated review.

## Explicitly out of scope

- velocity prediction or smoothing across two or more successful fits;
- adaptive queries, sparse sampling, lock-in behavior, or within-sweep motion;
- matched-budget accuracy, photon, latency, or speedup claims;
- choosing between successful warm and cold fits by hindsight score;
- CPU normalization across machines;
- collision/identity-swap resolution and dropout benchmarks; and
- uncertainty calibration or calibrated physical line detection.
