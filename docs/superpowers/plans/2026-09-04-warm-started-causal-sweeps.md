# Warm-Started Causal Sweep Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a causal completed-sweep estimator that seeds only from its most
recent eligible successful public fit, records every warm/cold attempt and
stale-estimate age, and preserves the Stage 6.1 fit definition and acquisition
accounting.

**Architecture:** Keep `fit_spectrum` stateless and move its start-independent
preflight and initial-guess validation into one package-internal preparation
module. That module also performs overflow-safe polynomial rebasing and returns
a typed compatibility outcome. A separate stateful wrapper owns immutable
configuration, causal endpoint modes, earlier public outputs, and append-only
history; frozen public records make attempts, source provenance, age, and CPU
cost explicit.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`optimize`), pytest, Ruff, Hatch
build, existing `odmr_bench.models` and `odmr_bench.dynamics` APIs.

## Global Constraints

- All scientific implementation remains under `src/odmr_bench/estimators/` and
  public types are exported from `odmr_bench.estimators`.
- Public frequency and linewidth values are Hz, time is seconds, FWHM is the
  only public linewidth convention, and `Q = center_hz / fwhm_hz` without an
  absolute value.
- The fitter remains a stateless single-sweep oracle. The wrapper may retain
  only immutable configuration, public fit results, public endpoints, and
  history; it never receives truth, future sweeps, dynamics objects,
  signal-conditioned expected photons, or evaluator references.
- A warm guess comes only from an earlier `SpectrumFitResult(success=True)`.
  Failed fits, synthetic truth, and offline reference trajectories are never
  seed sources.
- Ordered IDs retain `FitConfiguration.resonance_ids` only in the resolved,
  noncrossing eight-component scope. They are not evidence of physical identity
  through a collision.
- Changed frequency grids and spans are allowed. Compatibility is determined
  by the new sweep interval, rebased baseline, configured resonance bounds,
  and the same parameterization preflight used by `fit_spectrum`, never by
  exact grid equality.
- A rejected warm source causes one cold attempt. A compatible warm source
  causes one warm attempt and at most one cold recovery. The estimator never
  runs successful warm and cold fits and chooses between them by hindsight.
- Cold recovery is permitted only after `optimization_failed` or
  `quality_failed`; `insufficient_samples` and `uninformative_sweep` are
  start-independent preflight outcomes and are never retried.
- A cold retry reuses the same immutable `CompleteSweep`; it adds optimizer
  evaluations and CPU time but no observations, integration time, or nominal
  photon exposure.
- Do not add signal-conditioned expected-photon or realized-count aggregates to
  `CompleteSweep` or the new estimate. Only the already-public nominal exposure
  total is copied; evaluator joins for other photon/time bases remain Stage 6.5.
- Every controlled drift regression constructs a complete sweep from exactly
  one frozen `SpectralSnapshot` at the sweep's declared completion/reference
  timestamp. Cold and warm estimators consume the same `CompleteSweep` objects
  and `FitConfiguration`.
- CPU timing uses the patchable module-level `time.process_time_ns` clock.
  Timing is finite and non-negative but is excluded from deterministic result
  equality and from any realtime or universal-speedup claim.
- Invalid input, noncausal endpoints, unexpected preparation/fitter/timer
  exceptions, and record-construction failures append nothing and advance no
  endpoint mode, cumulative observation count, source state, or update index.
- `.venv/bin/python -m pytest` is used for every RED and GREEN run. Each task
  ends with focused tests, the complete estimator suite, full pytest, Ruff,
  `git diff --check`, one commit, and an independent review before the next
  task begins.
- If a task review finds a defect before the next task starts, add its focused
  RED regression, fix it, rerun that task's gates, and amend the same task
  commit with `git commit --amend --no-edit`; this preserves one reviewable
  commit per task.

## File and responsibility map

| Path | Responsibility |
|---|---|
| `src/odmr_bench/estimators/types.py` | Frozen `SweepFitAttempt` and `WarmSweepEstimate` contracts, closed literals, scalar/cross-field validation, and derived properties. |
| `src/odmr_bench/estimators/preparation.py` | Shared typed start-independent preflight, shared guess validation, exact baseline rebase, and typed warm compatibility outcome. |
| `src/odmr_bench/estimators/fitting.py` | Preserve public `fit_spectrum`; consume the shared preflight/guess helpers without changing Stage 6.1 behavior. |
| `src/odmr_bench/estimators/warm_sweep.py` | Causal endpoint/state machine, warm-source selection, cold recovery, aging, resource copying, CPU timing, and atomic append. |
| `src/odmr_bench/estimators/__init__.py` | Export only the new public records and estimator; preparation details stay internal. |
| `tests/estimators/test_types.py` | Exhaustive standalone and cross-field public-record invariants. |
| `tests/estimators/test_preparation.py` | Shared preflight parity, shared guess validation, baseline-rebase extremes, compatibility codes, and typed outcomes. |
| `tests/estimators/test_warm_sweep.py` | Mocked causal state/recovery/age/resource/CPU/atomicity behavior. |
| `tests/estimators/test_warm_sweep_integration.py` | Frozen-snapshot drift, unchanged/changed grids, public recovery tolerances, truth isolation, and example smoke. |
| `examples/fit_warm_started_sweeps.py` | Download-free short drift sequence with attempt/source/age diagnostics. |
| `docs/estimators.md`, `README.md` | Researcher-facing semantics, limitations, and example entry point. |
| `PROJECT_STATE.md`, `CHANGELOG.md` | Per-task implementation status and final verified counts. |

This split is intentional: Task 1 freezes records without fitting behavior;
Task 2 can be rejected independently if the numerical preparation boundary is
wrong; Task 3 can be reviewed with deterministic mocked fits; Task 4 alone
introduces generated scientific integration, documentation, and packaging
evidence.

## Frozen public contract

Add these exact aliases and dataclass surfaces to
`src/odmr_bench/estimators/types.py`:

```python
from typing import Literal, TypeAlias

SweepStartKind: TypeAlias = Literal["preflight", "cold", "warm"]
WarmStartDisposition: TypeAlias = Literal[
    "no_successful_prior",
    "used",
    "rejected_age",
    "rejected_compatibility",
    "not_applicable_preflight",
]
WarmStartRejectionCode: TypeAlias = Literal[
    "age_limit_exceeded",
    "baseline_rebase_unrepresentable",
    "center_outside_sweep",
    "center_separation_incompatible",
    "resonance_bounds_incompatible",
    "parameterization_unrepresentable",
]

@dataclass(frozen=True, slots=True)
class SweepFitAttempt:
    start_kind: SweepStartKind
    warm_source_update_index: int | None
    fit: SpectrumFitResult
    cpu_time_s: float

@dataclass(frozen=True, slots=True)
class WarmSweepEstimate:
    update_index: int
    attempts: Sequence[SweepFitAttempt]
    warm_start_disposition: WarmStartDisposition
    warm_start_rejection_code: WarmStartRejectionCode | None
    warm_start_message: str | None
    active_fit: SpectrumFitResult | None
    active_source_update_index: int | None
    estimate_age_submitted_observations: int | None
    estimate_age_sequence_indices: int | None
    estimate_age_s: float | None
    observation_count: int
    cumulative_observation_count: int
    first_sequence_index: int | None
    last_sequence_index: int | None
    last_timestamp_s: float | None
    total_integration_time_s: float | None
    total_nominal_exposure_photons: float | None
    cpu_time_s: float

    @property
    def current_fit(self) -> SpectrumFitResult:
        return self.attempts[-1].fit

    @property
    def is_stale(self) -> bool:
        return (
            self.active_fit is not None
            and self.active_source_update_index != self.update_index
        )

    @property
    def total_nfev(self) -> int:
        return sum(attempt.fit.nfev for attempt in self.attempts)
```

`attempts` is copied to an immutable tuple. Literal-valued fields require
actual strings in the declared closed sets; booleans and numerics are not
coerced. Integer fields accept Python/NumPy integral scalars except booleans
and are stored as Python `int`. Float fields accept real Python/NumPy scalars
except booleans, must be finite, and are stored as Python `float`.

### `SweepFitAttempt` state table

| `start_kind` | `warm_source_update_index` | permitted `fit` |
|---|---|---|
| `preflight` | exactly `None` | unsuccessful `insufficient_samples` or `uninformative_sweep`, `diagnostics.source == "none"`, `nfev == 0`, and `initial_guess is None` |
| `cold` | exactly `None` | success/optimizer/quality outcomes require `diagnostics.source` `"detected"` or `"fallback"`; `initialization_failed` also permits `"none"` |
| `warm` | non-negative integer | success, `optimization_failed`, or `quality_failed`, always with `diagnostics.source == "user"` |

Every row requires `fit` to be a `SpectrumFitResult` and `cpu_time_s` to be a
finite non-negative float. The containing `WarmSweepEstimate` additionally
requires every warm source to be strictly less than its `update_index`.
Preflight failure codes are therefore impossible on cold/warm attempts, and an
initialization failure is impossible on a warm attempt whose validated user
guess bypasses automatic initialization.

### Disposition and attempt matrix

| disposition | attempts | rejection code/message |
|---|---|---|
| `not_applicable_preflight` | exactly one `preflight` | both `None` |
| `no_successful_prior` | exactly one `cold` | both `None` |
| `rejected_age` | exactly one `cold` | code exactly `age_limit_exceeded`; nonempty message |
| `rejected_compatibility` | exactly one `cold` | exactly one of the five non-age codes; nonempty message |
| `used` | exactly one `warm`, or exactly `warm, cold` | both `None` |

For `used` with two attempts, the warm attempt must fail with
`optimization_failed` or `quality_failed`; the cold attempt is the selected
`current_fit`. All other two-item sequences are invalid. A one-item failed warm
attempt is valid only when cold retry was disabled. Constructors do not infer
or overwrite disposition fields.

Disposition meaning also constrains failed-current active state. A failed
`no_successful_prior` has no active fit/source/age. Failed `used`,
`rejected_age`, and `rejected_compatibility` require a stale active successful
fit; failed `used` additionally requires
`active_source_update_index == attempts[0].warm_source_update_index`. A
`not_applicable_preflight` failure may have no active result or retain an older
stale result. A successful selected current fit follows the ordinary current-
success row regardless of disposition and becomes active itself.

### Active result, age, endpoint, and resource matrix

`current_fit` is exactly `attempts[-1].fit`; `total_nfev` is
`sum(attempt.fit.nfev for attempt in attempts)` as a Python integer; and
`is_stale` is exactly
`active_fit is not None and active_source_update_index != update_index`. These
properties are derived and have no constructor parameters.

| current/active state | source | submitted age | sequence age | seconds age | stale |
|---|---:|---:|---:|---:|---|
| current success | exactly `update_index` | `0` | `0` iff sequence metadata exists, otherwise `None` | `0.0` iff timestamp metadata exists, otherwise `None` | `False` |
| current failure, older active success | integer in `[0, update_index)` | positive integer | positive integer iff sequence metadata exists, otherwise `None` | positive finite float iff timestamp metadata exists, otherwise `None` | `True` |
| current failure, no success yet | `None` | `None` | `None` | `None` | `False` |

When present, `active_fit` must be successful. Current success requires object
identity (`active_fit is current_fit`) rather than array-valued dataclass
equality. Current failure requires `active_fit is not current_fit`; the record
cannot prove which historical object supplied it, so the wrapper tests pin
identity against `history[active_source_update_index].current_fit`.

`observation_count` is positive.
`cumulative_observation_count >= observation_count`. Sequence metadata is
all-or-none per record: `first_sequence_index` and `last_sequence_index` are
both `None`, or both non-negative integers satisfying

```text
first_sequence_index = last_sequence_index - observation_count + 1 >= 0
```

Timestamp, integration, and exposure retain the `CompleteSweep` domains:
`last_timestamp_s` is optional non-negative finite;
`total_integration_time_s` is optional positive finite; and
`total_nominal_exposure_photons` is optional non-negative finite. Whole-update
`cpu_time_s` is finite, non-negative, and at least
`math.fsum(attempt.cpu_time_s for attempt in attempts)`.

## Internal preparation contract

Create `src/odmr_bench/estimators/preparation.py`; none of its names are added
to `odmr_bench.estimators.__all__`.

```text
WarmStartCompatibilityCode: TypeAlias = Literal[
    "baseline_rebase_unrepresentable",
    "center_outside_sweep",
    "center_separation_incompatible",
    "resonance_bounds_incompatible",
    "parameterization_unrepresentable",
]

class BaselineRebaseError(ValueError)

class InitialGuessCompatibilityError(ValueError)

@dataclass(frozen=True, slots=True)
class FitPreflight:
    free_parameter_count: int
    degrees_of_freedom: int
    frequency_min_hz: float
    frequency_max_hz: float
    frequency_reference_hz: float
    frequency_half_span_hz: float
    fluorescence_reference: float
    fluorescence_scale: float

@dataclass(frozen=True, slots=True)
class WarmStartPreparation:
    guess: FitInitialGuess | None
    rejection_code: WarmStartCompatibilityCode | None
    message: str | None

def start_independent_preflight(
    sweep: CompleteSweep,
    configuration: FitConfiguration,
) -> FitPreflight | SpectrumFitResult

def validate_initial_guess(
    guess: FitInitialGuess,
    configuration: FitConfiguration,
    preflight: FitPreflight,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]

def rebase_baseline(
    baseline: Baseline,
    new_reference_hz: float,
) -> Baseline

def prepare_warm_start(
    prior_fit: SpectrumFitResult,
    configuration: FitConfiguration,
    preflight: FitPreflight,
) -> WarmStartPreparation
```

`start_independent_preflight` validates both argument types, runs sample
count before fluorescence variation, and returns the exact existing Stage 6.1
`insufficient_samples` or `uninformative_sweep` `SpectrumFitResult`. A ready
`FitPreflight` contains the same values currently computed in `fit_spectrum`:

```python
free_parameters = baseline_degree + 1 + 8 * (
    4 if model_kind == "pseudo_voigt" else 3
)
degrees_of_freedom = len(sweep.frequency_hz) - free_parameters
frequency_min_hz = float(sweep.frequency_hz[0])
frequency_max_hz = float(sweep.frequency_hz[-1])
frequency_reference_hz = frequency_min_hz / 2.0 + frequency_max_hz / 2.0
frequency_half_span_hz = frequency_max_hz / 2.0 - frequency_min_hz / 2.0
fluorescence_scale = float(np.ptp(sweep.fluorescence))
fluorescence_reference = _stable_fluorescence_reference(sweep.fluorescence)
```

The ready record stores `free_parameter_count=free_parameters`; failure result
construction uses the same local before a ready record exists.

It preserves preflight ordering and exact diagnostics/reasons: non-positive
degrees of freedom returns `insufficient_samples` before reading a variation
scale; zero/non-finite `ptp` returns `uninformative_sweep` with
`"fluorescence variation is zero or non-finite"`; a non-finite stable origin
returns `uninformative_sweep` with
`"fluorescence origin is non-finite numerically"`. Both failures have
`source="none"`, no initial guess, no SciPy fields, `nfev=0`, and no residual
fields. `fit_spectrum` consumes this union and otherwise behaves identically.
It retains its existing `initial_guess` type check before calling the helper, so
an invalid supplied guess type still raises before any data-dependent branch.

`validate_initial_guess` is the current `_validate_guess` logic moved without
weakening any check. It requires exact configured ID order, Lorentzian eta
exactly one, width/amplitude bounds, finite packed values, intentional infinite
baseline bounds, finite resonance bounds, strict lower/upper bounds, and a
packed point within bounds. It calls the existing `pack_parameters` and
`parameter_bounds` once and returns `(packed, lower, upper)`. Its own deliberate
validation failures and every `ValueError` raised by those two named
parameterization helpers are re-raised as
`InitialGuessCompatibilityError` with the original message and exception
chaining. This remains compatible with Stage 6.1 because the subtype is still a
`ValueError`; exceptions of any other type propagate.

`WarmStartPreparation` enforces an exclusive state: success is exactly
`guess is not None`, `rejection_code is None`, `message is None`; rejection is
exactly `guess is None`, one of the five non-age rejection codes, and a
nonempty message. Age rejection belongs to the wrapper and cannot be returned
by this numerical helper.

`FitPreflight` is also constructor-validated: free-parameter count and degrees
of freedom are positive non-boolean integers; all six float fields are finite;
fluorescence scale and frequency half-span are positive; minimum is below
maximum; and reference and half-span exactly equal the overflow-safe endpoint
formulas used above. This
rejects an internally inconsistent normalization record. `rebase_baseline`
requires a `Baseline` and a finite non-boolean real reference, implements the
algorithm below, and is internal rather than exported. Its deliberate
representability failures are `BaselineRebaseError`; wrong argument types and
unexpected exceptions retain their original types.

### Exact baseline midpoint/rebase algorithm

For old baseline `(b0, r, b1, b2)` and new midpoint `r_prime`, implement this
exact finite-float sequence under no ignored warning state:

```python
if b1 == 0.0 and b2 == 0.0:
    return Baseline(b0, r_prime, 0.0, 0.0)

d = r_prime - r
if not math.isfinite(d):
    raise BaselineRebaseError(
        "baseline reference difference is not representable"
    )

try:
    b1_d = 0.0 if b1 == 0.0 or d == 0.0 else _finite_product_ratio(
        (b1, d), (), "baseline slope-reference product"
    )
    b2_d2 = 0.0 if b2 == 0.0 or d == 0.0 else _finite_product_ratio(
        (b2, d, d), (), "baseline quadratic-reference product"
    )
    two_b2_d = 0.0 if b2 == 0.0 or d == 0.0 else _finite_product_ratio(
        (2.0, b2, d), (), "baseline derivative-reference product"
    )
except ValueError as error:
    raise BaselineRebaseError(str(error)) from error
try:
    rebased_intercept = math.fsum((b0, b1_d, b2_d2))
    rebased_slope = math.fsum((b1, two_b2_d))
except OverflowError as error:
    raise BaselineRebaseError(
        "rebased baseline sum is not representable"
    ) from error
if not math.isfinite(rebased_intercept) or not math.isfinite(rebased_slope):
    raise BaselineRebaseError("rebased baseline sum is not representable")
return Baseline(rebased_intercept, r_prime, rebased_slope, b2)
```

The new midpoint is always the preflight value
`frequency_min_hz / 2 + frequency_max_hz / 2`. Zero linear/quadratic terms
short-circuit before computing `d`. Every mathematically nonzero product uses
the Stage 6.1 `_finite_product_ratio`, which rejects both overflow and nonzero
underflow. Exactly zero `math.fsum` results from representable cancellation are
accepted. Non-finite differences or individually unrepresentable terms remain
conservative `baseline_rebase_unrepresentable` rejections even if symbolic
arbitrary-precision cancellation could have produced a finite value.

### Compatibility classification order

`prepare_warm_start` requires a successful prior and classifies the first
failure in this exact order so messages remain deterministic:

1. `resonance_bounds_incompatible`: prior model kind, baseline degree, or
   ordered IDs differ from configuration; any width is outside inclusive
   `[min_fwhm_hz, max_fwhm_hz]`; any amplitude is outside inclusive
   `[0, max_amplitude]`. A successful Lorentzian result already guarantees eta
   exactly one, every public resonance already bounds eta to `[0, 1]`, and a
   successful degree-one result already guarantees zero quadratic, so those
   same-model defects are unreachable defensive facts rather than standalone
   successful-prior test fixtures. Cross-model eta semantics are rejected by
   the model-kind check.
2. `baseline_rebase_unrepresentable`: the exact algorithm above rejects the
   reference difference, a nonzero product, a sum, or the final public
   `Baseline`.
3. `center_outside_sweep`: any earlier center is below
   `frequency_min_hz` or above `frequency_max_hz`.
4. `center_separation_incompatible`: centers are not strictly increasing or
   any adjacent gap is below `min_center_separation_hz`.
5. `parameterization_unrepresentable`: `validate_initial_guess` rejects any
   remaining center-box, packing, scale, or finite-bound condition.

On success, copy all eight prior public resonance values in configured order,
replace only the baseline with its rebased value, build a new immutable
`FitInitialGuess`, and call `validate_initial_guess` once before returning it.
`prepare_warm_start` catches only `BaselineRebaseError` and
`InitialGuessCompatibilityError` and translates them to their declared codes.
An injected plain `ValueError`, a non-successful prior, wrong argument type,
assertion failure, or any other exception propagates and is never converted to
a compatibility result.

## Causal estimator contract

Add this exact public surface in `src/odmr_bench/estimators/warm_sweep.py`:

```text
class WarmStartedFullSweepEstimator:
    def __init__(
        self,
        configuration: FitConfiguration,
        *,
        retry_cold_on_warm_failure: bool = True,
        max_warm_start_age_updates: int | None = None,
    ) -> None

    @property
    def configuration(self) -> FitConfiguration

    @property
    def latest(self) -> WarmSweepEstimate | None

    @property
    def history(self) -> tuple[WarmSweepEstimate, ...]

    @property
    def latest_success(self) -> WarmSweepEstimate | None

    def update_sweep(self, sweep: CompleteSweep) -> WarmSweepEstimate

    def reset(self) -> None
```

`configuration` must be a `FitConfiguration`.
`retry_cold_on_warm_failure` accepts only Python/NumPy booleans and is stored as
a Python `bool`. `max_warm_start_age_updates` is `None` or a positive
Python/NumPy integral scalar, rejects booleans, and is stored as a Python
`int`. `None` disables the seed-age limit. For update `k`, source age is
`k - source.update_index`; the immediately preceding update has age one, and
rejection occurs only when that value is strictly greater than the configured
maximum. Age rejection never clears the older active result.

### Endpoint validation and staged commit

At method entry, sample
`update_started_ns = _sample_monotonic_cpu_ns(None)` and retain it as
`last_sample_ns`. Every later call passes the immediately previous raw sample
and replaces `last_sample_ns`; `_sample_monotonic_cpu_ns` propagates clock
exceptions and raises `RuntimeError("process CPU clock moved backwards")` if a
sample is lower. This enforces one globally nondecreasing stream, including
gaps and nesting boundaries, rather than validating paired durations alone.
Its exact body is:

```python
def _sample_monotonic_cpu_ns(previous_ns: int | None) -> int:
    raw = time.process_time_ns()
    if isinstance(raw, (bool, np.bool_)) or not isinstance(
        raw, (Integral, np.integer)
    ):
        raise TypeError("process_time_ns must return an integer")
    sample_ns = int(raw)
    if previous_ns is not None and sample_ns < previous_ns:
        raise RuntimeError("process CPU clock moved backwards")
    return sample_ns
```

Then type-check the sweep, derive
`observation_count = len(sweep.frequency_hz)`, and compute the prospective
cumulative count. Validate all prospective state in locals:

- Sequence availability is fixed by the first accepted update. If present,
  derive
  `first_sequence_index = last_sequence_index - observation_count + 1`, require
  it non-negative, and after the first update require
  `first_sequence_index > previous_last_sequence_index`. Gaps are permitted;
  overlap is not.
- Timestamp availability is fixed independently by the first accepted update.
  If present after the first update, require
  `last_timestamp_s > previous_last_timestamp_s`.
- A later present value after an absent first endpoint, or a later absent value
  after a present first endpoint, raises before preflight/fitting.

Call `start_independent_preflight` before inspecting prior age or compatibility.
Time it separately. If it returns a fit failure, create one `preflight` attempt
with that helper interval and disposition `not_applicable_preflight`; do not
prepare or claim a warm source.

For a ready preflight, the update decision is:

| condition | disposition | fitter calls |
|---|---|---|
| no earlier selected success | `no_successful_prior` | one cold with `initial_guess=None` |
| source age exceeds configured limit | `rejected_age` / `age_limit_exceeded` | one cold with `initial_guess=None` |
| typed warm preparation rejects | `rejected_compatibility` / helper code+message | one cold with `initial_guess=None` |
| compatible source | `used` | one warm with exactly `preparation.guess` |
| warm fails optimization/quality and retry enabled | `used` | then exactly one cold with `initial_guess=None` |

Each ordinary attempt samples `attempt_started_ns` through the global sampler,
calls `fit_spectrum` once, then globally samples `attempt_finished_ns`; no
acquisition object is recreated. A preflight helper interval becomes attempt
CPU only when preflight itself is the recorded attempt. Stage raw entries as
`(start_kind, warm_source_update_index, fit, elapsed_ns)` and construct no
public attempt record yet. After every required fit, globally sample
`update_finished_ns`. If that final call raises or regresses, the already-run
fit remains unrecorded and all estimator state is unchanged.

Only after the final clock succeeds, convert staged integer durations by
division by `1_000_000_000.0` and construct `SweepFitAttempt` records. Because
independent float conversion can round a containing interval below the sum of
subintervals, set

```python
attempt_sum_s = math.fsum(
    elapsed_ns / 1_000_000_000.0
    for _, _, _, elapsed_ns in raw_attempts
)
measured_update_s = (update_finished_ns - update_started_ns) / 1_000_000_000.0
cpu_time_s = max(measured_update_s, attempt_sum_s)
```

This is only a rounding guard; the integer intervals remain properly nested.
All public attempt/update record construction and history append occur after
the final clock sample and are outside `cpu_time_s`.

Select only `attempts[-1].fit`. If it succeeds, it becomes the current active
fit and the newly constructed estimate becomes `latest_success`. If it fails,
retain the earlier active fit/source unchanged. Compute ages from prospective
current/source records exactly as:

```python
submitted_age = (
    cumulative_observations_now
    - source_estimate.cumulative_observation_count
)
sequence_age = (
    current_last_sequence_index
    - source_estimate.last_sequence_index
    if current_last_sequence_index is not None
    else None
)
seconds_age = (
    current_last_timestamp_s - source_estimate.last_timestamp_s
    if current_last_timestamp_s is not None
    else None
)
```

For current success all available ages are literal zero. For stale output all
available ages are positive by endpoint validation. Construct every attempt
and the final estimate first. Only after all construction succeeds append the
estimate and commit endpoint modes, last endpoints, cumulative count, and
latest-success reference. `update_index` is always `len(_history)` before
append, so an exception cannot consume an index.

`reset()` clears history, latest-success state, both endpoint-availability
modes, both last endpoints, and cumulative observations while retaining the
three immutable constructor settings.

## Fixed generated regression family

Use these constants in `tests/estimators/test_warm_sweep_integration.py` and
the example. Truth remains local to fixture construction and is never passed
to an estimator:

```python
BASE_FREQUENCY_HZ = np.linspace(2.740e9, 3.020e9, 4481)
SHIFTED_FREQUENCY_HZ = np.linspace(2.741e9, 3.021e9, 4481)
BASE_CENTERS_HZ = 1e9 * np.array(
    [2.760, 2.794, 2.828, 2.862, 2.896, 2.930, 2.964, 2.998]
)
FWHM_HZ = 1e6 * np.array([1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20])
AMPLITUDES = np.array(
    [0.018, 0.021, 0.024, 0.027, 0.030, 0.033, 0.036, 0.039]
)
ETAS = np.array([0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.93])
COMPLETION_TIMESTAMPS_S = (1.0, 2.0, 3.0)
LAST_SEQUENCE_INDICES = (4480, 8961, 13442)
NOISE_SEEDS = (6211, 6212, 6213)
NOISE_SIGMA = 2.0e-4
CENTER_SLEW_HZ_PER_S = 1.0e5
DRIFT_CONFIGURATION = FitConfiguration(
    model_kind="pseudo_voigt",
    baseline_degree=2,
    resonance_ids=tuple(f"r{index}" for index in range(8)),
    min_fwhm_hz=2.0e5,
    max_fwhm_hz=8.0e6,
    max_amplitude=0.08,
    min_resolved_amplitude=1.0e-4,
    min_amplitude_significance=5.0,
    min_center_separation_hz=1.0e6,
    savgol_window=11,
    savgol_polyorder=2,
    relative_prominence=0.01,
    allow_fallback=False,
    max_nfev=4000,
    rank_rtol=1.0e-10,
    min_baseline_sse_improvement=1.0e-4,
)
```

Build an initial `SpectralSnapshot` with the configured `r0` through `r7` order,
quadratic `Baseline(intercept=1.0, reference_hz=2.880e9,
slope_per_hz=2.0e-11, quadratic_per_hz2=-5.0e-20)`, and the arrays above.
Use `LinearCenterDrift(initial_snapshot, CENTER_SLEW_HZ_PER_S)` and the exact
`DRIFT_CONFIGURATION` object for both cold and warm estimators. For each
timestamp, call `snapshot_at(timestamp)` exactly once, evaluate
`multi_resonance_spectrum` on grids `(BASE, SHIFTED, BASE)`, add the one seeded
Gaussian vector, and construct one `CompleteSweep` with the listed endpoint,
`total_integration_time_s=4.481`, and
`total_nominal_exposure_photons=4.481e6`.

Both estimators receive the same tuple elements by identity. Require successful
cold and warm fits, ordered IDs, center error
`< 0.10 * true_fwhm`, FWHM relative error `< 0.18`, and Q relative error
`< 0.18` for every component. These are fixed-fixture regression bounds, not
benchmark results. Assert only finite non-negative CPU and integral non-negative
evaluation totals; print or retain `nfev` descriptively but never require warm
to use fewer evaluations.

---

### Task 1: Frozen Attempt and Warm-Estimate Contracts

**Files:**

- Modify: `src/odmr_bench/estimators/types.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `tests/estimators/test_types.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes unchanged `SpectrumFitResult` and its Stage 6.1 failure-state
  invariants.
- Produces the exact public aliases, `SweepFitAttempt`, `WarmSweepEstimate`,
  and three derived properties specified in **Frozen public contract**.
- Later tasks may rely on tuple-canonicalized attempts and constructor-enforced
  provenance; they must not reconstruct these invariants in the wrapper.

- [ ] **Step 1: Write and run the public-name import RED**

  Add only a `test_warm_sweep_public_names_are_importable` smoke that imports
  the three aliases plus `SweepFitAttempt` and `WarmSweepEstimate` from
  `odmr_bench.estimators`. Run:

  ```bash
  .venv/bin/python -m pytest \
    tests/estimators/test_types.py::test_warm_sweep_public_names_are_importable -q
  ```

  Expected: collection fails with
  `ImportError: cannot import name 'SweepFitAttempt' from
  'odmr_bench.estimators'`.

- [ ] **Step 2: Add the minimal frozen/slotted surfaces and make import GREEN**

  Add the exact aliases and dataclass fields from **Frozen public contract**
  after `SpectrumFitResult`; initially canonicalize `attempts` to a tuple and
  implement only the three derived properties. Export all five names. Run the
  Step 1 command and require one pass. This deliberately leaves behavioral
  validation absent so the next RED is attributable to attempt semantics.

- [ ] **Step 3: Add legal factories and run the attempt-provenance RED**

  Extend existing factories with separate legal cold and warm results; never
  reuse a `source="user"` optimizer result as a cold result:

  ```python
  def _optimizer_result(start_kind: str, code: str | None) -> SpectrumFitResult:
      source = "user" if start_kind == "warm" else "detected"
      return _success_result(
          success=code is None,
          failure_code=code,
          resonance_estimates=_resonances() if code is None else (),
          baseline_estimate=(Baseline(1.0, 2.88e9) if code is None else None),
          diagnostics=_diagnostics(source),
          uncertainty=(
              _uncertainty(eta=np.full(8, 1e-3)) if code is None else None
          ),
          uncertainty_reason=None if code is None else "attempt failed",
          scipy_status=0 if code == "optimization_failed" else 1,
          scipy_message="stopped",
          nfev=4 if code is None else 3,
          cost=0.01,
          residual_rmse=0.02,
          residual_scale=0.1,
          jacobian_rank=34 if code is None else (
              None if code == "optimization_failed" else 10
          ),
      )
  ```

  Retain exact factories for cold `initialization_failed` with source
  `"none"`, plus both preflight failures. Test every legal attempt-table row,
  NumPy scalar canonicalization, frozen/slotted behavior, and reject:

  ```text
  unknown kind; non-fit; bad CPU/source scalar; non-None warm-source index on
  cold/preflight; missing warm-source index on warm; success/ordinary failure
  labeled preflight;
  preflight failure labeled cold/warm; initialization failure labeled warm;
  warm optimizer result whose diagnostics source is detected/fallback;
  cold optimizer result whose diagnostics source is user.
  ```

  Run the attempt-test selection. Expected RED: at least the warm/detected and
  cold/user rows construct without raising because Step 2 has no provenance
  validation.

- [ ] **Step 4: Implement attempt validation and make that group GREEN**

  Reuse existing scalar validators. Enforce: preflight requires source
  `"none"`; warm optimizer outcomes require source `"user"`; cold optimizer
  outcomes require source `"detected"` or `"fallback"`; cold
  `initialization_failed` permits `"detected"`, `"fallback"`, or `"none"`.
  Enforce the complete attempt table and run the Step 3 selection to all-pass.

- [ ] **Step 5: Add disposition/attempt tests and run their behavioral RED**

  Add every legal disposition row, both one-attempt `used` and warm-failure /
  cold-recovery `used`, all five compatibility codes, and these invalid rows:

  ```python
  INVALID_DISPOSITIONS = (
      ("not_applicable_preflight", (cold,), None, None),
      ("no_successful_prior", (preflight,), None, None),
      ("no_successful_prior", (warm,), None, None),
      ("rejected_age", (cold,), None, None),
      ("rejected_age", (cold,), "center_outside_sweep", "outside"),
      ("rejected_compatibility", (cold,), "age_limit_exceeded", "old"),
      ("rejected_compatibility", (cold,), "center_outside_sweep", None),
      ("used", (cold,), None, None),
      ("used", (warm, warm), None, None),
      ("used", (warm_success, cold), None, None),
      ("used", (warm_failure, cold, cold), None, None),
  )
  ```

  Also reject empty/more-than-two/non-attempt sequences, warm source at or
  after the update, stray/unknown code, empty required message, and message
  without code. Expected RED: invalid attempt/disposition combinations are
  accepted by the Step 4 record.

- [ ] **Step 6: Implement the disposition matrix and make it GREEN**

  Canonicalize the attempt tuple, code, and message; enforce the exact matrix
  and the failed-first requirement for two-attempt `used`. Run the Step 5
  selection and require all pass before adding active-state tests.

- [ ] **Step 7: Add active/disposition/resource tests and run their behavioral RED**

  Add all three base active rows plus the cross-disposition rules. A stale
  update at index 2/source 0 uses submitted age 64, sequence age 64, seconds
  age 1.25, 32 observations/cumulative 96, first/last 64/95, timestamp 2.0.
  Reject every one-field contradiction from **Active result, age, endpoint, and
  resource matrix**, plus these joins:

  ```text
  failed no_successful_prior with any active fit/source/age;
  failed used/rejected_age/rejected_compatibility without a stale active fit;
  failed used whose active source differs from its warm attempt source;
  unsuccessful active fit; current success not identical to active_fit;
  endpoint/age availability mismatch; invalid resource/CPU scalar;
  whole CPU below math.fsum(attempt CPU).
  ```

  Preflight failure is tested with no active fit and with one optional stale
  active fit. Verify derived `current_fit`, `is_stale`, and `total_nfev=7`
  without generated dataclass equality. Expected RED: Step 6 accepts at least
  the failed-disposition/active contradictions.

- [ ] **Step 8: Implement active/resource validation and make it GREEN**

  Implement validation in this order: canonical scalars; disposition/attempt
  table; active-state/disposition joins; endpoints/resources; CPU aggregation;
  canonical field assignment. Use object identity only for current success.
  Failed `used` checks only scalar source equality with its first warm attempt;
  wrapper tests later prove history-object identity. Run the Step 7 selection
  to all-pass.

- [ ] **Step 9: Run Task 1 complete GREEN and regression gates**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/estimators/test_types.py -q
  .venv/bin/python -m pytest tests/estimators -q
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check .
  git diff --check
  ```

  Expected: all commands pass; the existing baseline before implementation is
  289 estimator tests and 489 full tests, so counts may only increase. Inspect
  `git diff -- src/odmr_bench/estimators/types.py
  src/odmr_bench/estimators/__init__.py tests/estimators/test_types.py` and
  verify frozen/slotted behavior, exact literals, diagnostic/start provenance,
  disposition/active joins, no array-valued equality, and no wrapper behavior.

- [ ] **Step 10: Record, commit, and review Task 1**

  Update `PROJECT_STATE.md` with the passing counts and Task 1 contract status;
  add one concise `CHANGELOG.md` entry. Commit exactly:

  ```bash
  git add src/odmr_bench/estimators/types.py \
    src/odmr_bench/estimators/__init__.py \
    tests/estimators/test_types.py PROJECT_STATE.md CHANGELOG.md
  git commit -m "feat: add warm-sweep result contracts"
  ```

  A fresh reviewer must check the commit against every row of the three
  matrices, literal closure, NumPy scalar/boolean handling, derived properties,
  immutable tuple copying, and object-identity comparisons. Resolve all
  Critical/Important findings and rerun Step 7 before beginning Task 2.

---

### Task 2: Shared Fit Preparation and Guarded Warm Guess

**Files:**

- Create: `src/odmr_bench/estimators/preparation.py`
- Create: `tests/estimators/test_preparation.py`
- Modify: `src/odmr_bench/estimators/fitting.py`
- Modify: `tests/estimators/test_fitting.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes `CompleteSweep`, `FitConfiguration`, successful public
  `SpectrumFitResult`, `pack_parameters`, `parameter_bounds`, and
  `_finite_product_ratio`.
- Produces internal `FitPreflight`, `WarmStartPreparation`,
  `WarmStartCompatibilityCode`, `BaselineRebaseError`,
  `InitialGuessCompatibilityError`, `start_independent_preflight`,
  `validate_initial_guess`, `rebase_baseline`, and `prepare_warm_start` with the
  exact signatures above.
- `fit_spectrum(sweep, configuration, initial_guess=None) ->
  SpectrumFitResult` remains the unchanged public API and sole optimizer entry
  point.

- [ ] **Step 1: Write a focused exact-result preflight RED**

  Add valid pseudo-Voigt config/sweep helpers. For sample counts 34 and 35 with
  degree one (34 free parameters), assert `insufficient_samples`, degrees of
  freedom 0 and 1 behavior exactly as Stage 6.1 requires: count 34 fails and 35
  is ready. For a 35-sample constant sweep, assert `uninformative_sweep`. Pin
  every failure field, diagnostic source/message, `nfev=0`, and absence of an
  initial guess/residual fields. Monkeypatch `np.ptp` and
  `_stable_fluorescence_reference` to prove sample count runs first and the
  origin branch returns its exact reason.

  Run:

  ```bash
  .venv/bin/python -m pytest tests/estimators/test_preparation.py -q
  ```

  Expected: collection fails with
  `ModuleNotFoundError: No module named 'odmr_bench.estimators.preparation'`.

- [ ] **Step 2: Extract the start-independent helper and keep fitting green**

  Before editing fitter code, add two focused monkeypatched-optimizer
  characterization regressions that reach the final
  `rank != free_parameters` comparison: a full-rank accepted result and a
  rank-deficient `quality_failed` result. Run those two selections against the
  current fitter and require GREEN; they pin the live post-optimization local
  across the refactor.

  Move `_free_parameter_count`, `_empty_diagnostics`,
  `_preoptimization_failure`, and `_stable_fluorescence_reference` only as
  needed by the helper; import them privately in `fitting.py` if its remaining
  functions still use them. Replace the opening calculation in `fit_spectrum`
  with:

  ```python
  preflight = start_independent_preflight(sweep, configuration)
  if isinstance(preflight, SpectrumFitResult):
      return preflight
  free_parameters = preflight.free_parameter_count
  degrees_of_freedom = preflight.degrees_of_freedom
  frequency_min = preflight.frequency_min_hz
  frequency_max = preflight.frequency_max_hz
  frequency_reference = preflight.frequency_reference_hz
  frequency_half_span = preflight.frequency_half_span_hz
  fluorescence_reference = preflight.fluorescence_reference
  fluorescence_scale = preflight.fluorescence_scale
  ```

  Preserve public type validation at helper entry. Run:

  ```bash
  .venv/bin/python -m pytest \
    tests/estimators/test_preparation.py \
    tests/estimators/test_fitting.py -q
  ```

  Expected: all preflight and existing fitter tests pass with no change to fit
  classifications, diagnostics, initialization ordering, or numerical fields.
  The two characterization tests prove the extracted local remains defined on
  both rank branches rather than relying on broad suite coverage to expose a
  `NameError`.

- [ ] **Step 3: Write shared guess-validation parity tests**

  Build `FitPreflight` through the helper, then call
  `validate_initial_guess`. Compare its returned packed/lower/upper arrays
  exactly to `pack_parameters` and `parameter_bounds`. Parameterize exact
  configured-ID mismatch, Lorentzian eta mismatch, FWHM below/above inclusive
  limits, amplitude above maximum, center outside, separation below
  the configured minimum, collapsed boxes, non-finite/unrepresentable packing,
  and packed-outside-bounds cases. Monkeypatch
  `odmr_bench.estimators.preparation.validate_initial_guess` and assert one
  valid user-guess call to `fit_spectrum` invokes that exact shared symbol once.

  Expected RED before refactoring `_validate_guess`: focused fitting test fails
  because the shared symbol is not called. Move the current logic, replace the
  private call in `fit_spectrum`, and rerun the focused files to GREEN.

- [ ] **Step 4: Write exact linear/quadratic rebase tests**

  Test polynomial preservation at `new_reference + [-4e6, 0, 7e6]` with
  `rtol=2e-15, atol=2e-15` using:

  ```text
  old reference = 2.880e9
  new reference = 2.881e9
  intercept = 1.0
  slope = 2.0e-11
  quadratic = 0.0, then -5.0e-20
  expected linear = (1.00002, 2.0e-11, 0.0)
  expected quadratic intercept = 1.00001995
  expected quadratic slope = float.fromhex("0x1.5e15a1f111b1ep-36")
                           = 1.9899999999999997e-11
  expected quadratic coefficient = -5.0e-20
  ```

  The quadratic slope is mathematically
  `2e-11 + 2*(-5e-20)*(1e6) = 1.99e-11`, but the exact binary64 algorithm
  yields the hex value above, one ULP below the decimal literal `1.99e-11`.
  Assert the scalar slope against `float.fromhex` exactly. The intercept follows
  `1 + 2e-11*1e6 - 5e-20*(1e6)^2 = 1.00001995`.
  Independently compare old/new polynomial evaluations at the three frequencies
  with `rtol=2e-15, atol=2e-15`; do not replace that preservation check with
  coefficient-only assertions.

  Add binary-exact cancellation cases where `math.fsum` returns exactly zero:

  ```text
  (b0, b1, b2, d) = (1.0, -1.0, 0.0, 1.0) -> intercept' == 0.0
  (b0, b1, b2, d) = (0.0, 2.0, -1.0, 1.0) -> slope' == 0.0
  ```

  Add a zero-slope/quadratic baseline with old/new references
  `-max_float`/`max_float`; it must succeed because no difference is needed.
  With nonzero slope, the same references reject
  `baseline_rebase_unrepresentable`. Pin exact extreme products with direct
  `rebase_baseline` calls:

  ```python
  # Linear product is exactly 2**1000 and representable.
  Baseline(0.0, 0.0, slope_per_hz=math.ldexp(1.0, 500))
  new_reference_hz = math.ldexp(1.0, 500)

  # Linear product is mathematically 2**1100 and must reject.
  Baseline(0.0, 0.0, slope_per_hz=math.ldexp(1.0, 700))
  new_reference_hz = math.ldexp(1.0, 400)

  # Quadratic product is exactly one with no intermediate overflow.
  Baseline(0.0, 0.0, quadratic_per_hz2=math.ldexp(1.0, -1000))
  new_reference_hz = math.ldexp(1.0, 500)

  # A nonzero subnormal linear product underflows and must reject.
  Baseline(0.0, 0.0, slope_per_hz=np.nextafter(0.0, 1.0))
  new_reference_hz = 0.5
  ```

  Wrap the calls in `warnings.catch_warnings()` with
  `simplefilter("error")`; no numerical warning may escape.

  Run:

  ```bash
  .venv/bin/python -m pytest tests/estimators/test_preparation.py -k rebase -q
  ```

  Expected RED: collection or selected tests fail because `rebase_baseline`
  does not exist; the already-green shared preflight/validator tests still
  collect and are not the reason for this failure.

- [ ] **Step 5: Implement exact rebasing and make it GREEN**

  Implement `BaselineRebaseError`, the zero-coefficient short circuit, and the
  exact finite-float algorithm in **Exact baseline midpoint/rebase algorithm**.
  Run the Step 4 command and require all selected tests pass before adding
  compatibility behavior.

- [ ] **Step 6: Write and run the complete typed compatibility-code matrix RED**

  Start from one successful prior built with the Stage 6.1 public result
  factory, exact configured IDs, a midpoint baseline, and eight valid ordered
  components. Assert the returned guess is a defensive immutable value, keeps
  every center/FWHM/amplitude/eta and ID exactly, changes only baseline
  coefficients/reference, and passes `validate_initial_guess`.

  Parameterize each closed code with a deterministic trigger:

  | code | trigger |
  |---|---|
  | `baseline_rebase_unrepresentable` | nonzero slope and old/new references whose subtraction is infinite |
  | `center_outside_sweep` | change interval so the first old center is 1 Hz below `f_min` |
  | `center_separation_incompatible` | prior adjacent centers differ by `min_center_separation_hz - 1` |
  | `resonance_bounds_incompatible` | prior model/degree/IDs mismatch; with current bounds `2e5..8e6` Hz and amplitude maximum `0.25`, use prior widths `1e5` and `9e6` plus amplitude `0.30` as separate rows |
  | `parameterization_unrepresentable` | centers `1e16 + 100*np.arange(8)`, sweep endpoints `1e16 - 100` and `1e16 + 800`, and separation `5.0`; individually valid centers produce unrepresentable rounded box separation |

  Boundary acceptance is inclusive for sweep centers, FWHM, amplitude, and eta;
  center ordering is strict and configured separation is inclusive. Assert
  every rejection has a nonempty explanatory message, no guess, and no age
  code. Assert a `WarmStartPreparation` constructor rejects both/neither
  states, an age code, unknown code, empty message, and a message on success.

  Add multi-defect rows to freeze first-failure precedence:

  ```text
  width 9e6 plus overflowing baseline rebase -> resonance_bounds_incompatible;
  overflowing baseline rebase plus first center outside -> baseline_rebase_unrepresentable;
  first center outside plus an adjacent gap below minimum -> center_outside_sweep;
  adjacent 100 Hz gaps with requested separation 101 Hz plus rounded box
      failure near 1e16 -> center_separation_incompatible.
  ```

  Call `prepare_warm_start(object(), configuration, preflight)`,
  `prepare_warm_start(success, object(), preflight)`, and
  `prepare_warm_start(success, configuration, object())`; each must raise
  `TypeError` and return no `WarmStartPreparation`. Pass a legal
  `optimization_failed` prior and require plain `ValueError` propagation with
  no compatibility result. Monkeypatch `rebase_baseline` and
  `validate_initial_guess` separately to raise plain
  `ValueError("injected unexpected value error")`; require that exact exception
  to propagate. Then raise each private typed exception and require only those
  to become the corresponding compatibility result.

  Run:

  ```bash
  .venv/bin/python -m pytest tests/estimators/test_preparation.py \
    -k 'warm or compatibility' -q
  ```

  Expected RED: failures identify missing `WarmStartPreparation` and
  `prepare_warm_start`, while the completed rebase tests remain green.

- [ ] **Step 7: Implement typed compatibility and make it GREEN**

  Implement the classification order in **Internal preparation contract**.
  Catch only
  `BaselineRebaseError`/`InitialGuessCompatibilityError` at the translation
  boundary. Do not use `decimal`, arbitrary precision, clipped coefficients,
  reordered IDs, guessed truth, or a looser second validation path.

  Rerun:

  ```bash
  .venv/bin/python -m pytest tests/estimators/test_preparation.py -q
  .venv/bin/python -m pytest tests/estimators/test_fitting.py \
    tests/estimators/test_parameterization.py -q
  ```

  Expected: all pass. The second command proves the extraction preserved the
  Stage 6.1 scaled residual, quality, covariance, and center-box behavior.

- [ ] **Step 8: Run Task 2 full gates, inspect, commit, and review**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/estimators -q
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check .
  git diff --check
  ```

  Inspect the diff for one shared preflight, one shared guess validator, exact
  Stage 6.1 failure payloads, no import cycle, finite-float short-circuits,
  first-failure code determinism, and no public export of preparation types.
  Update project state/changelog with passing counts, then commit exactly:

  ```bash
  git add src/odmr_bench/estimators/preparation.py \
    src/odmr_bench/estimators/fitting.py \
    tests/estimators/test_preparation.py \
    tests/estimators/test_fitting.py PROJECT_STATE.md CHANGELOG.md
  git commit -m "refactor: share fit preparation for warm starts"
  ```

  A fresh reviewer must audit midpoint arithmetic, nonzero underflow/overflow,
  exact-zero cancellation, compatibility classification, shared-call evidence,
  Stage 6.1 output parity, and truth isolation. Resolve all
  Critical/Important findings and rerun this step before Task 3.

---

### Task 3: Causal Warm-Started Full-Sweep State Machine

**Files:**

- Create: `src/odmr_bench/estimators/warm_sweep.py`
- Create: `tests/estimators/test_warm_sweep.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes public `CompleteSweep`, `FitConfiguration`, `SweepFitAttempt`, and
  `WarmSweepEstimate`; internal `FitPreflight`,
  `start_independent_preflight`, and `prepare_warm_start`; and unchanged
  `fit_spectrum`.
- Produces and exports `WarmStartedFullSweepEstimator` with the exact constructor,
  properties, update, and reset signatures in **Causal estimator contract**.
- Exposes no dynamics, truth, fitter-choice, mutable-history, or endpoint
  mutation hook.

- [ ] **Step 1: Add legal wrapper factories and run the public-class RED**

  In `tests/estimators/test_warm_sweep.py`, define the exact 64-point `_sweep`
  from Task 1's contract tests. Provide distinct `_cold_success` and
  `_warm_success` factories with diagnostics sources `"detected"` and
  `"user"`; distinct cold/warm optimization and quality failures with the same
  legal sources; a cold `initialization_failed`; and both exact preflight
  failures. Keep all other Stage 6.1 fields valid. Add only the import test and
  run:

  ```bash
  .venv/bin/python -m pytest \
    tests/estimators/test_warm_sweep.py::test_warm_estimator_is_public -q
  ```

  Expected: collection fails with
  `ImportError: cannot import name 'WarmStartedFullSweepEstimator'`.

- [ ] **Step 2: Add the minimal class surface and make constructor tests GREEN**

  Add/export the class with final slots, constructor validation, empty
  properties, and reset. Its temporary `update_sweep` has the final signature
  and raises `NotImplementedError("update_sweep state machine not added")`;
  this scaffold is removed in Step 4 and never committed. Add constructor tests
  for invalid configuration; retry values `0`, `1`, `"yes"`; invalid ages
  `True`, `False`, `0`, `-1`, `1.5`, `np.inf`, string; and canonical accepted
  `np.bool_(False)`/`np.int64(2)`. Require immutable configuration and empty
  `latest`, `latest_success`, tuple history. Run the public/constructor/reset
  selection and require GREEN.

- [ ] **Step 3: Add no-prior and preflight behavioral RED tests**

  Produce one real ready `FitPreflight`, then monkeypatch module helpers. Test
  first cold success, repeated cold failures with no success (each remains
  `no_successful_prior`), and both preflight codes before/after an existing
  success. Preflight must bypass preparation/fitter; with a prior it is the only
  disposition whose failed update may keep an optional stale active fit.
  Expected RED: calls reach the temporary `NotImplementedError`, not an import
  failure or invalid result factory.

- [ ] **Step 4: Implement no-prior/preflight paths and make them GREEN**

  Replace the scaffold with a minimal update that validates input, calls shared
  preflight first, runs one cold fit only without a success, constructs legal
  records, and appends last. Use zero CPU temporarily; timer behavior is added
  in Step 12. Run the Step 3 selection and require all pass.

- [ ] **Step 5: Add source/age/compatibility branch-join RED tests**

  Require a cold success at update 0 to seed update 1 with the exact prior fit
  and exact prepared guess. Pin age maximum boundaries (`1` rejects source age
  2; `2` and `None` accept), and compatibility rejection with copied code/
  message and no warm fitter call. Add the missing joins:

  ```text
  age-rejected cold success becomes the next update's warm source;
  compatibility-rejected cold success becomes the next warm source;
  compatibility-rejected cold failure retains the exact prior active fit,
      source index 0, submitted age 64, and unavailable sequence/time ages;
  failed used disposition keeps the identical prior active fit and the same
      source index as its warm attempt.
  ```

  Expected RED: Step 4 always cold-starts and therefore fails the used,
  rejection, and new-source assertions.

- [ ] **Step 6: Implement source selection/age/preparation and make it GREEN**

  Add latest-success source selection, strict age comparison, typed preparation,
  and cold rejection branches. Promote every selected cold success, including
  after rejection; leave the prior active only after selected failure. Run the
  Step 5 selection to all-pass.

- [ ] **Step 7: Add warm recovery behavioral RED tests**

  Parameterize legal warm `optimization_failed`/`quality_failed`. With retry,
  require `(warm_guess, None)`, attempts `("warm", "cold")`, source only on
  warm, both retained, cold selection, exact summed `nfev`, and one-sweep
  acquisition totals. With retry disabled require one failed warm and stale
  prior. Warm success stops. Warm returning initialization/preflight codes must
  fail record construction atomically and never relabel/retry. Expected RED:
  Step 6 has no conditional cold-recovery branch.

- [ ] **Step 8: Implement recovery and make it GREEN**

  Retry exactly once only for the two eligible warm failures, select only the
  last attempt, and never compare successful-start cost/RMSE/parameters. Run
  the Step 7 selection and require all pass.

- [ ] **Step 9: Add endpoint/age/resource/reset behavioral RED tests**

  Test indices 63 then 127 (first 0/64), overlap ending 126, accepted gap ending
  191, negative first from initial last 62; timestamps 1.0 then 1.5 with equal/
  decreasing rejection; and independent availability changes
  `present->absent`/`absent->present`. Reject before helper/fitter and preserve
  state. Pin success/failure/success ages `(0,0,0.0)`, `(64,64,1.5)`, then
  zeros; without external bases keep only submitted age. With a gap, pin
  submitted age 64 versus sequence age 128. Distinct per-sweep integration and
  exposure copy once through two-attempt recovery. After reset, retain settings
  but accept endpoint-free update 0/cumulative 64. Expected RED: Step 8 lacks
  endpoint modes and exact age/resource derivation.

- [ ] **Step 10: Implement endpoints/ages/resources and make them GREEN**

  Add exact `_ProspectiveEndpoints` and `_validate_endpoints` signatures below,
  compute prospective state locally, construct records, then commit. Run the
  Step 9 selection to all-pass.

- [ ] **Step 11: Add global CPU/exception atomicity behavioral RED tests**

  Patch `time.process_time_ns` with these exact streams:

  | path | clock values (ns) | expected attempt CPU | expected update CPU |
  |---|---|---|---|
  | preflight failure | `100, 110, 140, 160` | `30e-9` | `60e-9` |
  | one cold | `100, 110, 120, 130, 160, 180` | `30e-9` | `80e-9` |
  | warm then cold | `100, 110, 120, 130, 150, 160, 200, 220` | `20e-9, 40e-9` | `120e-9` |

  Add contiguous attempt durations `693945840953877919 ns` and
  `169434960463419660 ns`; their separately converted `math.fsum` is
  `863380801.4172976`, one ULP above containing interval conversion
  `863380801.4172975`, so update CPU must use the former. Also add globally
  regressing but pairwise-valid
  `100, 110, 120, 105, 115, 130`. The latter must raise at `105`; the `max`
  guard may not mask it. Inject clock `RuntimeError("timer failed")` after exact
  cold-path samples `100, 110, 120, 130, 160`, so it occurs on the final sample
  after fitter work. Monkeypatch the module's `WarmSweepEstimate` constructor
  with a raising factory for the final-record-construction case. Also inject
  exceptions from preflight, preparation, warm fit, and cold recovery. Every
  case preserves
  history/latest/latest-success, endpoint modes, last endpoints, cumulative
  count, and update index; fault removal permits the exact same sweep/index.
  Expected RED: Step 10 has zero CPU/no global sampler and does not call the
  patched final timer.

- [ ] **Step 12: Implement globally monotonic timing and make it GREEN**

  Follow **Endpoint validation and staged commit** exactly: keep raw outcomes
  and nanoseconds local, globally sample every boundary, sample update end,
  then construct public records and commit. Implement these exact helpers:

  ```text
  @dataclass(frozen=True, slots=True)
  class _ProspectiveEndpoints:
      sequence_available: bool
      first_sequence_index: int | None
      last_sequence_index: int | None
      timestamp_available: bool
      last_timestamp_s: float | None

  def _sample_monotonic_cpu_ns(previous_ns: int | None) -> int

  def _nonnegative_elapsed_ns(start_ns: int, end_ns: int) -> int

  def _timed_fit(
      self,
      sweep: CompleteSweep,
      initial_guess: FitInitialGuess | None,
      previous_sample_ns: int,
  ) -> tuple[SpectrumFitResult, int, int]

  def _validate_endpoints(
      self,
      sweep: CompleteSweep,
      observation_count: int,
  ) -> _ProspectiveEndpoints
  ```

  `_timed_fit` returns `(fit, elapsed_ns, final_sample_ns)` and uses immutable
  `self._configuration`. Compare two logical runs with different clocks by
  explicit non-CPU fields, never nested dataclass equality. Real clocks receive
  only finite/non-negative and whole-at-least-attempt-sum assertions. Run the
  Step 11 selection and the whole focused file to GREEN.

- [ ] **Step 13: Run Task 3 gates, inspect, commit, and review**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/estimators/test_types.py \
    tests/estimators/test_preparation.py \
    tests/estimators/test_warm_sweep.py -q
  .venv/bin/python -m pytest tests/estimators -q
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check .
  git diff --check
  ```

  Inspect source and tests for successful-prior-only seeding, no successful
  warm/cold hindsight choice, preflight-first ordering, endpoint-base
  independence, strict timestamps/nonoverlap, exact aging, same-sweep resource
  reuse, timer nesting, and append-last atomicity. Update state/changelog and
  commit exactly:

  ```bash
  git add src/odmr_bench/estimators/warm_sweep.py \
    src/odmr_bench/estimators/__init__.py \
    tests/estimators/test_warm_sweep.py PROJECT_STATE.md CHANGELOG.md
  git commit -m "feat: add causal warm-started sweep estimator"
  ```

  A fresh reviewer must trace every branch in the decision table and verify no
  truth/future/evaluator reference, no swallowed unexpected exception, no
  acquisition double count, no failed-result promotion, and no state mutation
  before final record construction. Resolve all Critical/Important findings and
  rerun this step before Task 4.

---

### Task 4: Frozen-Snapshot Drift Integration, Guidance, and Package Smoke

**Files:**

- Create: `tests/estimators/test_warm_sweep_integration.py`
- Create: `examples/fit_warm_started_sweeps.py`
- Modify: `docs/estimators.md`
- Modify: `README.md`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes the public Stage 6.2 API only; production integration tests do not
  import preparation internals.
- Reuses `SpectralSnapshot`, `LinearCenterDrift`, `Baseline`, `Resonance`, and
  `multi_resonance_spectrum` only in evaluator/test/example fixture
  construction.
- Produces no new estimator API. The example is download-free and reports
  diagnostics, not benchmark superiority.

- [ ] **Step 1: Write frozen-snapshot drift integration tests**

  Implement the exact family in **Fixed generated regression family**. Return
  `(sweeps, truth_resonances_by_update)` from the fixture builder, where every
  `truth_resonances_by_update[k]` comes from the single snapshot used to build
  `sweeps[k]`. Do not call `snapshot_at` again during assertions.

  Instantiate cold and warm estimators with the same immutable configuration
  and submit in causal order as:

  ```python
  for sweep in sweeps:
      cold_estimates.append(cold.update_sweep(sweep))
      warm_estimates.append(warm.update_sweep(sweep))
  ```

  Spy on `odmr_bench.estimators.full_sweep.fit_spectrum` and
  `odmr_bench.estimators.warm_sweep.fit_spectrum` with wrappers that append the
  received sweep and then call a saved reference to the real
  `odmr_bench.estimators.fitting.fit_spectrum`. Assert every cold/warm spy entry
  `is sweeps[k]`; never clone or regenerate one path.
  Require attempt kinds `[("cold",), ("warm",), ("warm",)]`, warm source
  indices `[None, 0, 1]`, current active sources `[0, 1, 2]`, all available ages
  zero, exact cumulative observations `[4481, 8962, 13443]`, and unchanged
  per-sweep resources. Apply the declared center/FWHM/Q regression bounds to
  both estimator outputs. Assert every estimator instance has no attribute
  holding the dynamics/snapshots/truth tuple.

- [ ] **Step 2: Add changed-grid, stale recovery, and ID provenance regressions**

  The three-grid sequence already proves a successful prior rebases across a
  changed midpoint and then back. Add a separate deterministic wrapper-only
  success/failure/success regression with the middle fluorescence constant.
  Label this test `test_wrapper_preflight_failure_ages_prior_without_drift_truth`;
  it is not a controlled drift regression and constructs its constant
  `CompleteSweep` directly, without creating then overwriting a snapshot-based
  spectrum. The successful outer sweeps may reuse the fixed public spectral
  arrays solely to establish wrapper state. The sequence must produce cold
  success, `preflight` uninformative failure with stale
  ages `(4481, 4481, 1.0)`, then a warm attempt sourced from update 0 (or cold
  if the explicitly configured age limit rejects it). Use
  `max_warm_start_age_updates=None` in the acceptance case and `1` in the age
  rejection case.

  Assert fitted IDs always equal `configuration.resonance_ids`. A failed fit
  has no resonance IDs and can never seed. Truth IDs are used only to build and
  score snapshots; no production call accepts them separately. A deliberately
  narrowed sweep that excludes the first prior center must create exactly one
  cold attempt with `rejected_compatibility` /
  `center_outside_sweep`, not a warm call or a truth-derived correction.

- [ ] **Step 3: Add an optimizer-recovery integration without brittle SciPy behavior**

  Keep scientific fitting in the frozen-snapshot test. For recovery provenance,
  monkeypatch only the module-level `fit_spectrum` so the compatible warm call
  returns a legal fixed `optimization_failed` result and the next cold call
  returns a legal success. Assert both attempts, source index, selected cold
  result, current active promotion, one-sweep acquisition resources, and exact
  `total_nfev`. Do not force real SciPy failure with `max_nfev`; that is
  platform-sensitive and already tested at the fitter boundary.

- [ ] **Step 4: Write documentation/example RED tests**

  Add a subprocess test modeled on the existing synthetic example test. Run
  `examples/fit_warm_started_sweeps.py` from `tmp_path` with `sys.executable`.
  Require exit zero, header
  `Synthetic warm-started sweep diagnostics`, exactly three update rows, and
  parse fields:

  ```text
  update disposition attempts warm_source active_source age_observations
  total_nfev cpu_time_s
  ```

  Expected attempt strings are `cold`, `warm`, `warm`; sources are `none`, `0`,
  `1`; ages are zero after each success; `total_nfev` is non-negative integral;
  CPU is finite non-negative. Read `docs/estimators.md` and require exact terms
  `warm source`, `baseline rebase`, `cold recovery`, `stale`,
  `estimate_age_submitted_observations`, `center boxes`, and
  `measured update-core process CPU interval`.

  Run:

  ```bash
  .venv/bin/python -m pytest tests/estimators/test_warm_sweep_integration.py -q
  ```

  Expected RED: the example subprocess returns status 2 because
  `examples/fit_warm_started_sweeps.py` does not exist, and the documentation
  term assertions fail. Any drift regression failure is a Stage 6.2 integration
  defect to correct within this task, not a tolerance to loosen without
  numerical evidence.

- [ ] **Step 5: Implement the download-free example**

  Generate the three fixed-seed sweeps from **Fixed generated regression
  family**, submit them causally, fail loudly if a
  current fit is unsuccessful, and print one row per update. Render missing
  sources as `none`, attempts as comma-joined start kinds, ages as decimal
  integers, `total_nfev` as decimal, and CPU with `:.9g`. Do not print truth
  errors or claim lower CPU/evaluations. Guard every printed float with
  `math.isfinite` and every public fit with ordered configured IDs.

- [ ] **Step 6: Extend researcher guidance and README**

  Add a `Warm-started completed sweeps` section to `docs/estimators.md` that
  states:

  - only the latest earlier successful public fit may seed;
  - polynomial rebasing preserves the old baseline at the new overflow-safe
    midpoint, while unrepresentable products reject warm use;
  - changed grids are supported subject to current center boxes/bounds;
  - a failed warm attempt remains visible before conditional one-cold recovery;
  - current failure and older active success are distinct, and stale age uses
    submitted-observation, sequence-index, and timestamp bases without mixing;
  - retries add CPU/`nfev` but no acquisition resources;
  - `cpu_time_s` is the measured update-core process CPU interval through the
    instant before record construction/state append and is machine-dependent;
  - endpoint inconsistency/overlap and unexpected exceptions abort an update
    atomically, and a harness must not skip the acquired sweep;
  - callers reset between independent recordings so endpoint-availability modes
    and source provenance cannot cross recording boundaries;
  - ordered center boxes assume resolved noncrossing components; and
  - this completed-sweep baseline does not establish temporal bandwidth,
    realtime utility, universal speedup, collision identity, or matched-budget
    superiority.

  Document both constructor options and the public fields/properties. Update
  `README.md` to name `WarmStartedFullSweepEstimator`, link the same guidance,
  and show `python examples/fit_warm_started_sweeps.py` as a diagnostic
  generated example beside the cold example.

- [ ] **Step 7: Run focused and scientific GREEN gates**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/estimators/test_warm_sweep_integration.py -q
  .venv/bin/python -m pytest tests/dynamics tests/models \
    tests/estimators -q
  .venv/bin/python examples/fit_warm_started_sweeps.py
  ```

  Expected: all tests pass; the example emits one header plus three finite
  diagnostic rows. Capture cold/warm `nfev` values in the test report only as
  descriptive regression observations. The test suite must not assert a
  universal warm-start speedup or platform-stable SciPy status/message/count.

- [ ] **Step 8: Run the complete build and isolated-wheel smoke**

  Use a temporary build directory so no stale repository `dist/` artifact can
  be selected:

  ```bash
  bash <<'STAGE_6_2_SMOKE'
  set -euo pipefail
  repo=$(pwd -P)
  smoke_root=$(mktemp -d)
  "$repo/.venv/bin/python" -m pytest -q
  "$repo/.venv/bin/ruff" check .
  "$repo/.venv/bin/python" -m build --outdir "$smoke_root/dist"
  wheel="$smoke_root/dist/nv_odmr_trackbench-0.1.0-py3-none-any.whl"
  sdist="$smoke_root/dist/nv_odmr_trackbench-0.1.0.tar.gz"
  test -f "$wheel"
  test -f "$sdist"
  test "$(find "$smoke_root/dist" -maxdepth 1 -name '*.whl' -type f | wc -l | tr -d ' ')" = 1
  test "$(find "$smoke_root/dist" -maxdepth 1 -name '*.tar.gz' -type f | wc -l | tr -d ' ')" = 1
  "$repo/.venv/bin/python" -m venv "$smoke_root/venv"
  "$smoke_root/venv/bin/python" -m pip install "$wheel"
  cd "$smoke_root"
  "$smoke_root/venv/bin/python" -c \
    'from odmr_bench.estimators import FitConfiguration, SweepFitAttempt, SweepStartKind, WarmStartedFullSweepEstimator, WarmStartDisposition, WarmStartRejectionCode, WarmSweepEstimate; e = WarmStartedFullSweepEstimator(FitConfiguration()); assert e.latest is None and e.history == ()'
  "$smoke_root/venv/bin/python" "$repo/examples/fit_warm_started_sweeps.py"
  git -C "$repo" diff --check
  STAGE_6_2_SMOKE
  ```

  Expected: full pytest passes with a count greater than the 489-test baseline;
  Ruff reports `All checks passed!`; build creates exactly one sdist and the
  exact wheel; isolated import/constructor smoke includes all three public
  aliases and exits zero; the example runs with the wheel-only
  temporary interpreter and emits three rows; diff check emits no output. Any
  intermediate failure terminates the declared Bash script immediately.

- [ ] **Step 9: Inspect, record, commit, and review Task 4**

  Inspect the complete Stage 6.2 diff for frozen-snapshot construction,
  same-object cold/warm inputs, declared tolerances, no hidden truth access,
  exact age/resource semantics, unsupported performance language, source-only
  example behavior, public API packaging, secrets/absolute paths, and generated
  artifacts. Remove no user files and commit no build/cache artifacts.

  Update `PROJECT_STATE.md` with the exact estimator/full test counts, Ruff,
  build, wheel smoke, Stage 6.2 behavior, and next integrated-review action.
  Add a concise `CHANGELOG.md` entry, then commit exactly:

  ```bash
  git add tests/estimators/test_warm_sweep_integration.py \
    examples/fit_warm_started_sweeps.py docs/estimators.md README.md \
    PROJECT_STATE.md CHANGELOG.md
  git commit -m "feat: validate warm-started drift sweeps"
  ```

  A fresh reviewer must audit snapshot timing, estimator/truth separation,
  numerical bounds, changed-grid rebase, stale/age and recovery evidence,
  resource honesty, documentation claims, example output, and installed-wheel
  public imports. Resolve all Critical/Important findings and rerun Steps 7–8.

---

## Final Integrated Review Gate

After all four task reviews are clean, create one review package covering
`e8c8b44` (the approved Stage 6.2 design commit) through the final Task 4
implementation commit. A fresh senior reviewer must inspect the actual diff
and run evidence for:

- public record closure and every attempt/disposition/active/age invariant;
- midpoint/rebase overflow, underflow, cancellation, and unchanged polynomial;
- exact Stage 6.1 preflight and guess-validation reuse without fitter drift;
- successful-prior-only causality, endpoint modes, recovery selection, and
  append-last atomicity;
- distinct submitted-observation/sequence/time age bases;
- per-attempt versus whole-update process CPU and acquisition non-duplication;
- frozen-snapshot timing, same-sweep cold/warm inputs, IDs, and truth isolation;
- fixed-fixture center/FWHM/Q tolerances without a universal speedup claim; and
- docs, example, exports, build, isolated wheel, secrets, and repository
  hygiene.

One fix owner addresses the complete Critical/Important finding set in atomic
commits, reruns the focused and complete gates, and requests re-review of the
full range. Stage 6.2 is complete only when re-review has no remaining
Critical/Important findings and these final commands pass:

```bash
.venv/bin/python -m pytest tests/estimators -q
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/python -m build --outdir "$(mktemp -d)/dist"
git diff --check
```

Final acceptance means only that the declared causal completed-sweep baseline
is implemented and regression-backed. It does not establish realtime utility,
matched-budget superiority, collision-safe identity, physical eight-line
detection, or a machine-independent compute advantage.
