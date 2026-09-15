# Sparse Five-Point Linewidth Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage 6.4 as an additive composite two-point-center and
five-point-linewidth tracker/evaluator with exact causal scheduling, numerical
gates, provenance, timing, resources, generated acceptance, and no Stage 6.5
comparative claim.

**Architecture:** Estimator-safe immutable contracts, the canonical local
model/fitter, and one composite causal state machine live under
`odmr_bench.estimators`. Evaluator-only full observations, expected photons,
actual midpoints, opaque authority, resource joins, and instrument-owning run
transitions live in a new `odmr_bench.evaluation.sparse_linewidth` package.
Both evaluator runners use one private runner-neutral verified-calibration core
with exact runner allowlisting, while every Stage 6.3 public object and pair
semantic remains unchanged.

**Tech Stack:** Python 3.11+, frozen/slotted dataclasses, NumPy 1.26+, SciPy
1.11+ bounded `least_squares`, `time.process_time_ns`, pytest 8+, Ruff, Hatch,
and the existing `odmr_bench.models`, `dynamics`, `emulator`, `estimators`, and
`evaluation.two_point` APIs.

## Global Constraints

- The binding source of truth is
  `docs/superpowers/specs/2026-09-15-sparse-linewidth-tracker-design.md`;
  implementation may add private helpers but may not rename, weaken, omit, or
  extend its public records, aliases, error codes, signatures, precedence,
  timing rules, presence matrices, or state transitions.
- Stage 6.4 is additive. No Stage 6.3 public record, field, constructor
  invariant, failure meaning, pair schedule, runner transition, export object,
  or resource meaning changes. Shared-helper extraction requires bitwise/value
  differential coverage of Stage 6.3 before and after extraction.
- Internal frequency and linewidth units are Hz; time is seconds; normalized
  fluorescence is dimensionless; FWHM is the linewidth convention. Physical
  quantities never enter public APIs as ambiguous unitless values.
- The estimator receives only immutable public configuration/calibration,
  estimator-safe observations, public endpoints, budget, and its own state. It
  never receives or retains an instrument, dynamics, truth/snapshot, full
  observation, expected photons, actual midpoint, callback, evaluator handle,
  future record, or post-release truth.
- The composite owns one global accepted-observation sequence/time recurrence.
  It does not wrap a running `CalibratedTwoPointTracker` and does not construct
  a `TwoPointEstimate` over a stream containing sparse gaps.
- Fast pairs retain the exact Stage 6.3 round-robin, per-ID side alternation,
  discriminator, capture/domain/common-mode, source-refresh, and lost-pair
  semantics. Sparse work may not relabel or narrow a valid Stage 6.3 pair.
- A sparse scan is due after exactly eight completed fast pairs, targets
  `source_ids[completed_scans % 8]`, starts only at a clean pair boundary, and
  reserves five adjacent observations indivisibly. Scientific scan failure
  advances scan count/ID parity; partial, rejected, and aborted scans do not.
- Sparse multipliers are exactly `(-1.0, -0.5, 0.0, 0.5, 1.0)` times the
  FWHM prior frozen at scan start. Even per-ID acquisition order is exactly
  `(+0.5, -1.0, 0.0, +1.0, -0.5)` and odd order is its exact reversal;
  `sum(t*x) == 0` for equal spacing is not a nonlinear-dynamics guarantee.
- Every sparse query freezes and echoes the exact fast-center and FWHM-prior
  source kinds, indices, public references, release sequence indices, and
  release timestamps. Points one through four cannot change point five.
- The local fit has exactly four free public parameters: local center
  correction, FWHM, amplitude, and constant baseline offset. Target eta, source
  baseline shape, and every non-target center/FWHM/amplitude/eta and tail are
  frozen and evaluated in immutable source order.
- Normative fit bounds are center correction `[-0.5*w0,+0.5*w0]`, FWHM
  `[max(0.5*w0, source_min), min(2.0*w0, source_max)]`, amplitude
  `[0.0,min(4.0*A0,source_max)]`, and baseline offset `[-A0,+A0]`.
- Final scaled-Jacobian gates are ordered exactly: returned model validity;
  returned solver status/`nfev`; finite solution; `1e-6` interior bound margin;
  rank four with `rank_rtol=1e-10`; condition `<=1e8`; amplitude
  `>=max(0.25*A0, source_min_resolved_amplitude)`; normalized RMSE `<=0.10`.
- Raised ordinary exceptions from model arithmetic, SciPy, SVD, Q, timing, or
  construction roll back and become typed construction/evaluator aborts. Only
  returned `status <= 0` or `nfev >= 4000` is scientific `optimizer_failed`.
  Transaction-cleaned `BaseException` values re-raise unchanged.
- The fit diagnostic presence matrix is exact: solver fields begin at
  `optimizer_failed`; fitted/RMSE groups begin at `bounds_active`; rank begins
  at `rank_deficient`; condition begins at `ill_conditioned`; fitted scan Q is
  present only on success. `fit_cpu_time_s` is always present on a completed
  scan.
- Local center correction is diagnostic only and never feeds the fast center.
  Scan Q is fitted local center/FWHM. The live value is explicitly an
  asynchronous projection `fast_center_hz/active_fwhm_hz` with separate source
  epochs; finite negative, zero, and positive Q are valid.
- Public scan time uses the exact five-value left fold of endpoint-reconstructed
  midpoints. Evaluator truth time uses the same fold over actual instrument
  midpoints. Release is the fifth-arriving observation endpoint/index, and
  truth is never requested before release or more than once.
- Reset prospectively validates every calibration-seeded geometry. Later due
  geometry failure stops cleanly with exact first-applicable code and source/
  envelope/cell/domain facts before affordability or query. Proposed min/max
  are jointly absent only for nonrepresentable lower/upper codes.
- Pair and five-point affordability replays two or five atomic charges in
  query order from current charged state. Never multiply a block cost, add
  subtotals, subtract snapshots, use `sum`/`math.fsum`, or use resource-float
  tolerances. Expected/realized photons are not affordability inputs.
- Public calibration, fast, sparse, interleaved tracking, and charged ledgers
  use exact left-associated arrival-order atoms. The interleaved total is never
  `fast + sparse`. Evaluator ledgers independently retain expected photons and
  at most one authenticated unaccepted atom.
- `fast_update_cpu_time_s`, `sparse_update_cpu_time_s`, and
  `total_update_cpu_time_s` use accepted update deltas and exact left folds;
  total follows global arrival order and is never formed from subtotals. CPU
  time is nonnegative process CPU, not acquisition resource or a realtime claim.
- Verified authority is private, exact-identity, and exact-runner-allowlisted.
  Public constructors/factories, subclasses, copies, serialization, class
  membership, and `object.__new__` allocations cannot mint or transfer it.
- `included_same_run` and `conditional_free_precalibration` remain explicit and
  exact. Source cost is charged once in the former and reported but uncharged
  in the latter. Caller-asserted sources never gain same-run authority.
- Generated fixtures and examples are deterministic and download-free. They
  are synthetic contract evidence, not hardware, experimental, uncertainty,
  sensitivity, realtime, optimality, or matched-budget-superiority evidence.
- Stage 6.5 metrics/ranking and Stage 6.6 artifacts/plots/report automation are
  out of scope. Affine within-scan baseline change and all within-scan parameter
  dynamics remain explicit model-mismatch limitations.
- Every behavioral production change follows RED, observed intended failure,
  minimal GREEN, focused refactor, focused gate, affected estimator/evaluator/
  emulator gate, full pytest, Ruff, `git diff --check`, diff inspection, and one
  atomic task commit. Review fixes first receive a focused regression.

---

## File and Responsibility Map

| Path | Responsibility |
|---|---|
| `src/odmr_bench/estimators/sparse_linewidth_types.py` | All estimator aliases, errors, configuration, scan/identity/aggregate/update records, intrinsic matrices, and defensive snapshots. |
| `src/odmr_bench/estimators/sparse_linewidth_fit.py` | Fit geometry, canonical target-local model, packed scaling, one bounded solver attempt, gates, and CPU diagnostic. |
| `src/odmr_bench/estimators/sparse_linewidth_tracker.py` | Composite reset, scheduling, pair/sparse transitions, global clocks, ages, safe resources, live projection, CPU totals, and atomic commit. |
| `src/odmr_bench/estimators/two_point_calibration.py` | Extract reusable private canonical target-local source-model helpers without public change. |
| `src/odmr_bench/estimators/__init__.py` | Add only approved Stage 6.4 estimator exports. |
| `src/odmr_bench/evaluation/two_point/provenance.py` | Generalize private registry to exact-allowlist both runner types and exact instances. |
| `src/odmr_bench/evaluation/two_point/calibration.py` | Extract private runner-neutral verified acquisition transaction; preserve public Stage 6.3 entry point. |
| `src/odmr_bench/evaluation/two_point/runner.py` | Adapt private core call only; preserve all public behavior. |
| `src/odmr_bench/evaluation/sparse_linewidth/types.py` | Evaluator errors, acquisition/query-failure, timing, abort, resources, runner states, and outcomes. |
| `src/odmr_bench/evaluation/sparse_linewidth/resource_accounting.py` | Exact authenticated full/safe joins and evaluator arrival-order ledgers. |
| `src/odmr_bench/evaluation/sparse_linewidth/runner.py` | Public bind/calibration/start/step/run/stop state machine. |
| `src/odmr_bench/evaluation/sparse_linewidth/__init__.py` | Exact evaluator public surface. |
| `src/odmr_bench/dynamics/linewidth_drift.py` | Deterministic per-ID FWHM dynamics for generated tests only. |
| `src/odmr_bench/dynamics/__init__.py` | Export the new public dynamics strategy. |
| `tests/sparse_linewidth_helpers.py` | Legal source/calibration/instrument/query helper factories shared only by tests. |
| `tests/estimators/test_sparse_linewidth_types.py` | Exact estimator contracts, presence matrices, defensive copies, signed Q, and isolation surfaces. |
| `tests/estimators/test_sparse_linewidth_fit.py` | Local model, scaling, solver, every numerical gate/ULP, exception, and CPU behavior. |
| `tests/estimators/test_sparse_linewidth_tracker.py` | Reset, geometry, schedule, reservation, fast/sparse transitions, epochs, resources, and CPU totals. |
| `tests/estimators/test_sparse_linewidth_atomicity.py` | Typed precedence and value-equal rollback for every prospective construction fault. |
| `tests/evaluation/test_sparse_linewidth_types.py` | Evaluator intrinsic field/phase/outcome/presence matrices and exact public signatures. |
| `tests/evaluation/test_sparse_linewidth_resources.py` | Full/safe joins, ledgers, partial/failed scans, abort atom, and unavailable join. |
| `tests/evaluation/test_sparse_linewidth_runner.py` | Bind/calibration/start, accepted steps, retries, timing, stops, aborts, and phase matrix. |
| `tests/evaluation/test_sparse_linewidth_regressions.py` | Closed static/noisy/drift/mismatch acceptance and truth isolation. |
| `tests/evaluation/test_two_point_calibration.py` | Runner-neutral core differential, allowlist, rollback, and Stage 6.3 compatibility. |
| `tests/evaluation/test_two_point_runner.py` | Exact unchanged Stage 6.3 start/step/terminal traces after extraction. |
| `tests/dynamics/test_linewidth_drift.py` | FWHM dynamics construction, interpolation, identity, and snapshot validation. |
| `examples/track_sparse_linewidth.py` | Download-free conditional-precalibration diagnostic example. |
| `docs/estimators.md`, `README.md`, `tests/test_package.py` | Guidance, example entry, exports, source-tree and isolated-wheel smoke. |
| `PROJECT_STATE.md`, `CHANGELOG.md` | Per-task evidence, review status, and scope boundary. |

## Cross-Task Interface Ledger

The approved spec owns every public field list. The following producer names
and signatures are fixed so task briefs can be executed independently:

```python
def _evaluate_bound_source_model(
    frequency_hz: NDArray[np.float64], source: TwoPointCalibrationSource,
    target_resonance_id: str, *, center_hz: float, fwhm_hz: float,
    amplitude: float, baseline_offset: float,
) -> NDArray[np.float64]

def _construct_sparse_geometry(
    calibration: TwoPointCalibration, configuration: SparseLinewidthConfiguration,
    identity: CompositeIdentityEstimate, *, scan_index: int,
    identity_scan_index: int,
) -> tuple[tuple[SparseLinewidthQuery, ...], _SparseFitGeometry]

def _validate_calibration_sparse_geometry(
    calibration: TwoPointCalibration,
    configuration: SparseLinewidthConfiguration,
) -> None

def fit_sparse_linewidth(
    source: TwoPointCalibrationSource,
    configuration: SparseLinewidthConfiguration,
    queries: tuple[SparseLinewidthQuery, ...],
    observations: tuple[EstimatorObservation, ...],
) -> SparseLinewidthScanResult

class SparseLinewidthCompositeTracker:
    def __init__(self, configuration: SparseLinewidthConfiguration) -> None
    def reset(self, public_metadata: TwoPointRunMetadata,
              calibration: TwoPointCalibration,
              budget_ceiling: TwoPointBudgetCeiling, *, seed: int) -> None
    def choose_next_query(self) -> TwoPointQuery | SparseLinewidthQuery | None
    def update(self, observation: EstimatorObservation) -> SparseLinewidthCompositeUpdate
    def estimate(self) -> SparseLinewidthCompositeEstimate

def build_sparse_linewidth_evaluator_resources(
    runner: SparseLinewidthEvaluatorRunner,
) -> SparseLinewidthEvaluatorResources | None

class SparseLinewidthEvaluatorRunner:
    @classmethod
    def bind(cls, instrument: ODMRInstrument) -> SparseLinewidthEvaluatorRunner
    @property
    def state(self) -> SparseEvaluatorRunnerState
    def acquire_verified_calibration(
        self, frequency_hz: Sequence[float], integration_time_s: float,
        fit_configuration: FitConfiguration,
        identity_binding: TwoPointIdentityBinding, *, source_id: str,
        source_clock_id: str, tracker_clock_id: str,
        source_to_tracker_offset_s: float,
        physical_fit_epoch_rule: Literal["instrument_midpoint_ordered_mean"],
    ) -> VerifiedTwoPointCalibrationOutcome
    def start_tracking(
        self, tracker: SparseLinewidthCompositeTracker,
        calibration: TwoPointCalibration,
        verified_calibration: VerifiedTwoPointCalibrationSuccess,
        public_metadata: TwoPointRunMetadata,
        budget_ceiling: TwoPointBudgetCeiling, *, seed: int,
    ) -> SparseEvaluatorRunnerState
    def step(self) -> SparseRunnerStepOutcome
    def run_until_event(self) -> SparseRunnerRunOutcome
    def stop_external(self) -> SparseRunnerExternallyStopped
```

Private verified-calibration extraction introduces `_VerifiedCalibrationIssuer`
with exact `runner`, `instrument`, `run_token`, and
`instrument_configuration` identities; `_acquire_verified_calibration_core`
consumes that issuer plus the public acquisition arguments and returns
`VerifiedTwoPointCalibrationOutcome`. Only exact registered
`TwoPointEvaluatorRunner` and `SparseLinewidthEvaluatorRunner` instances can
obtain an issuer.

```python
def _acquire_verified_calibration_core(
    issuer: _VerifiedCalibrationIssuer,
    frequency_hz: Sequence[float],
    integration_time_s: float,
    fit_configuration: FitConfiguration,
    identity_binding: TwoPointIdentityBinding,
    *,
    source_id: str,
    source_clock_id: str,
    tracker_clock_id: str,
    source_to_tracker_offset_s: float,
    physical_fit_epoch_rule: Literal["instrument_midpoint_ordered_mean"],
) -> VerifiedTwoPointCalibrationOutcome
```

## Execution Waves and Conflict Rules

| Wave | Tasks | Dependency |
|---|---|---|
| 1 | 1-2 | Estimator contracts only; Task 2 consumes Task 1. |
| 2 | 3-6 | Shared model, geometry, fitter success, then failure gates. |
| 3 | 7-10 | Tracker shell, fast branch, sparse partial, sparse completion. |
| 4 | 11-13 | Private provenance, evaluator contracts, resource builder. |
| 5 | 14-16 | Runner calibration/start, accepted steps, terminal protocol. |
| 6 | 17-19 | Dynamics, closed acceptance, public docs/package. |
| 7 | 20 | Integrated scientific/software review and closeout. |

Tasks in a wave are listed by dependency, not parallel permission. A worker
owns only the files listed in its task, must preserve concurrent edits, and must
not update `PROJECT_STATE.md` or `CHANGELOG.md` outside that task's evidence
entry. Tasks 3 and 11 are the only feature tasks permitted to modify Stage 6.3
production modules; a Task 20 fix wave may touch one only after adding a focused
differential RED for its concrete review finding. Tasks 3, 11, 14, and 20 must
run the complete Stage 6.3 focused suites before commit.

## Mandatory Per-Task Gate

After each task's focused GREEN, run exactly:

```bash
.venv/bin/python -m pytest tests/estimators tests/evaluation tests/emulator -q
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
git diff --check
git status --short
```

Expected: every pytest command exits zero, Ruff prints `All checks passed!`,
`git diff --check` is silent, and `git status --short` lists only that task's
declared files. Record the fresh counts in `PROJECT_STATE.md` and a concise
scope-safe entry in `CHANGELOG.md` before the task commit.

---

### Task 1: Estimator Error, Configuration, and Geometry-Diagnostic Primitives

**Files:** Create `src/odmr_bench/estimators/sparse_linewidth_types.py` and
`tests/estimators/test_sparse_linewidth_types.py`; modify
`src/odmr_bench/estimators/__init__.py`, `PROJECT_STATE.md`, and `CHANGELOG.md`.

**Interfaces:** Define the exact spec aliases `SparseLinewidthFailureCode`,
`CompositeMode`, `SparseLinewidthSourceKind`, `CompositeStopReason`,
`SparseGeometryFailureCode`, `SparseResetFailureCode`,
`SparseObservationValidationCode`, and `SparseUpdateConstructionCode`; frozen,
slotted `SparseLinewidthConfiguration`; frozen, slotted
`SparseGeometryUnavailableDiagnostic`; and the three public errors
`SparseLinewidthResetError`, `SparseLinewidthObservationValidationError`, and
`SparseLinewidthUpdateConstructionError`, each with its exact closed `code`.

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

```python
SparseGeometryFailureCode = Literal[
    "nonrepresentable_frequency_lower", "nonrepresentable_frequency_upper",
    "empty_fit_bounds", "calibration_cell_violation", "source_domain_violation",
]
SparseResetFailureCode = Literal[
    "invalid_argument_type", "configuration_mismatch", "calibration_mismatch",
    "metadata_mismatch", "invalid_base_sparse_geometry", "budget_mismatch",
    "initial_state_construction_failed",
]
SparseObservationValidationCode = Literal[
    "invalid_observation_type", "no_pending_query", "pending_mode_mismatch",
    "fast_query_echo_mismatch", "sparse_query_echo_mismatch",
    "sequence_mismatch", "frequency_mismatch", "integration_time_mismatch",
    "endpoint_mismatch", "nominal_exposure_mismatch", "invalid_observation_value",
]
SparseUpdateConstructionCode = Literal[
    "fast_partial_pair_construction_failed", "fast_pair_result_construction_failed",
    "fast_identity_estimate_construction_failed",
    "sparse_partial_scan_construction_failed",
    "sparse_scan_result_construction_failed",
    "sparse_identity_estimate_construction_failed", "resource_construction_failed",
    "aggregate_estimate_construction_failed", "update_construction_failed",
]
```

**First RED witness:**

```python
def test_sparse_linewidth_configuration_has_normative_defaults() -> None:
    configuration = SparseLinewidthConfiguration()
    assert configuration.scan_period_fast_pairs == 8
    assert configuration.integration_time_s == 0.005
    assert configuration.max_nfev == 4000
    with pytest.raises(TypeError):
        replace(configuration, max_nfev=True)
```

- [ ] **RED:** Test exact defaults, signatures/fields, canonical built-in
  scalars, frozen/slots behavior, rejected booleans/complex/arrays/nonfinite
  values, every ratio/count relation, and diagnostic proposed-bound presence.
  Run `.venv/bin/python -m pytest tests/estimators/test_sparse_linewidth_types.py
  -q`; expect import failure because the module does not exist.
- [ ] **GREEN:** Implement only these primitives. Preserve the fixed offset and
  order policy outside configuration. Enforce geometry code precedence and
  require proposed min/max jointly absent only for nonrepresentable lower or
  upper, jointly finite/ordered otherwise; all q0/width/cell/source facts are
  always finite. Re-run the RED command; expect all tests green.
- [ ] Run the Mandatory Per-Task Gate. Commit the declared files with
  `git commit -m "feat: define sparse linewidth primitives"`.

---

### Task 2: Query, Partial, Result, Identity, Aggregate, and Update Records

**Files:** Modify `src/odmr_bench/estimators/sparse_linewidth_types.py`,
`src/odmr_bench/estimators/__init__.py`, and
`tests/estimators/test_sparse_linewidth_types.py`; create
`tests/sparse_linewidth_helpers.py`; modify `PROJECT_STATE.md` and `CHANGELOG.md`.

**Interfaces:** Add the exact spec fields for `SparseLinewidthQuery`,
`SparsePartialScan`, `SparseLinewidthScanResult`, `CompositeIdentityEstimate`,
`SparseLinewidthCompositeEstimate`, and `SparseLinewidthCompositeUpdate`.

**First RED witness:**

```python
def test_scan_q_preserves_repository_signed_convention() -> None:
    result = make_scan_result(status="success", fitted_local_center_hz=-1.0,
                              fitted_fwhm_hz=2.0, fitted_q=-0.5)
    assert result.fitted_q == -0.5
    assert type(result.fitted_q) is float
```

`tests/sparse_linewidth_helpers.py` produces `make_sparse_query`,
`make_partial_scan`, `make_scan_result`, `make_composite_identity`, and
`make_composite_estimate`, each returning a legal exact public base-class
record and accepting only declared keyword overrides.

- [ ] **RED:** Add exact `dataclasses.fields` assertions and tests for immutable
  tuple/copy boundaries, five-query echo identity, partial lengths 1..4,
  completed length 5, source-kind/index/release joins, status/failure union,
  the full gate-presence matrix, signed/zero finite Q, history/counter/resource
  equations, one-incomplete-block rule, asynchronous epoch/age joins, and exact
  query/update echoes. Run `.venv/bin/python -m pytest
  tests/estimators/test_sparse_linewidth_types.py -q`; expect missing records.
- [ ] **GREEN:** Implement records as frozen/slotted validated snapshots. The
  result exposes solver fields from `optimizer_failed`, fitted and RMSE groups
  from `bounds_active`, rank from `rank_deficient`, condition from
  `ill_conditioned`, and fitted Q only on success. The aggregate stores immutable
  configuration plus calibration source ID/provenance/treatment and all three
  independent CPU folds. Re-run RED; expect green.
- [ ] Run the Mandatory Per-Task Gate. Commit with
  `git commit -m "feat: add sparse linewidth estimator records"`.

---

### Task 3: Canonical Bound-Source Model Extraction and Stage 6.3 Differential

**Files:** Modify `src/odmr_bench/estimators/two_point_calibration.py` and
`tests/estimators/test_two_point_calibration.py`; create
`tests/estimators/test_sparse_linewidth_source_model.py`; modify
`PROJECT_STATE.md` and `CHANGELOG.md`.

**Interface:** Add private `_evaluate_bound_source_model(frequency_hz, source,
target_resonance_id, *, center_hz, fwhm_hz, amplitude, baseline_offset) ->
NDArray[np.float64]` and any private validation helper needed by both fits.

**First RED witness:**

```python
def test_bound_source_model_changes_only_target_and_constant_offset() -> None:
    source = make_legal_caller_asserted_source()
    frequencies = np.array([2.86e9, 2.87e9, 2.88e9])
    actual = _evaluate_bound_source_model(
        frequencies, source, "r0", center_hz=2.861e9,
        fwhm_hz=8.0e6, amplitude=0.02, baseline_offset=0.003,
    )
    assert np.array_equal(actual, independently_ordered_model(source, frequencies))
```

- [ ] **RED:** Freeze Stage 6.3 calibration outputs/exceptions over Lorentzian
  and pseudo-Voigt sources, affine/quadratic baselines, overlapping tails,
  source-order permutations, scalar/vector frequencies, and boundary values;
  add local-model tests proving baseline is evaluated once, non-target lines
  are subtracted in immutable source order, and only target center/FWHM/amplitude
  plus a constant offset vary. Run `.venv/bin/python -m pytest
  tests/estimators/test_two_point_calibration.py
  tests/estimators/test_sparse_linewidth_source_model.py -q`; expect the private
  helper import to fail while legacy tests stay green.
- [ ] **GREEN:** Extract/generalize the private source-bound core and delegate
  Stage 6.3 without changing a public object, exception, arithmetic order, or
  result bit pattern. Do not pre-sum a background. Re-run RED plus
  `.venv/bin/python -m pytest tests/estimators/test_two_point_tracker.py -q`;
  expect differential equality and all legacy tests green.
- [ ] Run the Mandatory Per-Task Gate. Commit with
  `git commit -m "refactor: share bound spectral source model"`.

---

### Task 4: Sparse Fit Geometry and Prospective Calibration Validation

**Files:** Create `src/odmr_bench/estimators/sparse_linewidth_fit.py` and
`tests/estimators/test_sparse_linewidth_fit.py`; modify `PROJECT_STATE.md` and
`CHANGELOG.md`.

**Interfaces:** Add private frozen `_SparseFitGeometry`; exact
`_validate_calibration_sparse_geometry(calibration, configuration) -> None`; and
`_construct_sparse_geometry(calibration, configuration, identity, *, scan_index,
identity_scan_index) -> tuple[tuple[SparseLinewidthQuery, ...],
_SparseFitGeometry]`.

**First RED witness:**

```python
@pytest.mark.parametrize("parity, expected", [
    (0, (0.5, -1.0, 0.0, 1.0, -0.5)),
    (1, (-0.5, 1.0, 0.0, -1.0, 0.5)),
])
def test_sparse_geometry_uses_exact_time_symmetric_order(parity, expected) -> None:
    queries, _ = _construct_sparse_geometry(
        calibration, configuration, identity, scan_index=parity,
        identity_scan_index=parity,
    )
    assert tuple(query.offset_multiplier for query in queries) == expected
```

- [ ] **RED:** Test all eight calibration-seeded identities, exact multiplier
  set, even order `(+.5,-1,0,+1,-.5)`, exact odd reversal, equal-spacing
  `sum(t*x)==0`, all five preconstructed query clocks/resources/source echoes,
  normative/intersected bounds, strict-interior initial guess, ULP boundaries,
  and first-applicable geometry codes including proposed-bound presence. Run
  `.venv/bin/python -m pytest tests/estimators/test_sparse_linewidth_fit.py -q`;
  expect missing geometry helpers.
- [ ] **GREEN:** Construct lower then upper with representability checks, then
  empty bounds, cell, and source-domain checks in that order. Validation raises
  typed construction errors and never invokes SciPy. Re-run RED; expect green.
- [ ] Run the Mandatory Per-Task Gate. Commit with
  `git commit -m "feat: construct sparse linewidth fit geometry"`.

---

### Task 5: Isolated Four-Parameter Local Fit Success Path

**Files:** Modify `src/odmr_bench/estimators/sparse_linewidth_fit.py` and
`tests/estimators/test_sparse_linewidth_fit.py`; modify `PROJECT_STATE.md` and
`CHANGELOG.md`.

**Interface:** Implement `fit_sparse_linewidth(source, configuration, queries,
observations) -> SparseLinewidthScanResult` using one
`least_squares(method="trf")` attempt in scaled variables
`(dc/w0,w/w0,A/A0,b0/A0)`.

**First RED witness:**

```python
def test_noiseless_fit_recovers_exact_four_parameter_local_model() -> None:
    result = fit_sparse_linewidth(source, configuration, queries, observations)
    assert result.status == "success"
    assert result.fitted_center_correction_hz == pytest.approx(expected_dc)
    assert result.fitted_fwhm_hz == pytest.approx(expected_fwhm)
    assert result.fitted_amplitude == pytest.approx(expected_amplitude)
    assert result.fitted_baseline_offset == pytest.approx(expected_offset)
```

- [ ] **RED:** Add noiseless recovery tests for both supported line shapes,
  sloped/quadratic frozen baselines, non-target tails, source order, exact
  residual sign/model formula, one solver invocation, exact public initial
  guess/bounds/scaling, five-element arrival order, fitted local center/Q, and
  no mutation. Run the focused fit file; expect the fitter to be absent.
- [ ] **GREEN:** Implement the exact four-free-parameter model; freeze target
  eta, baseline shape, and all non-target values. Use model-minus-observation
  residuals and no restart. Construct a success record through the Task 2
  validator. Run `.venv/bin/python -m pytest
  tests/estimators/test_sparse_linewidth_fit.py -k 'success or model' -q`;
  expect green.
- [ ] Run the Mandatory Per-Task Gate. Commit with
  `git commit -m "feat: fit sparse local linewidth model"`.

---

### Task 6: Fit Failure Gates, Diagnostic Presence, Exceptions, and CPU

**Files:** Modify `src/odmr_bench/estimators/sparse_linewidth_fit.py` and
`tests/estimators/test_sparse_linewidth_fit.py`; modify `PROJECT_STATE.md` and
`CHANGELOG.md`.

**Interfaces:** Complete all `SparseLinewidthScanResult` status paths and exact
`fit_cpu_time_s` measurement around preparation, solve, gates, and construction.

**First RED witness:**

```python
@pytest.mark.parametrize("status,nfev", [(0, 1), (1, 4000)])
def test_only_returned_solver_failure_is_scientific_optimizer_failure(
    monkeypatch, status, nfev,
) -> None:
    monkeypatch.setattr(sparse_fit, "least_squares",
                        fake_solver_result(status=status, nfev=nfev))
    result = fit_sparse_linewidth(source, configuration, queries, observations)
    assert (result.status, result.failure_code) == ("failure", "optimizer_failed")
    assert result.scipy_status == status
```

- [ ] **RED:** Parameterize the eight first-applicable failures and exact
  presence rows. Pin returned `status<=0` and `nfev>=4000` as the only
  `optimizer_failed` paths; wrong shapes/nonfinite public solution as
  `nonfinite_solution`; scaled bound-margin equality/inward ULP; rank cutoff
  equality/outward ULP; rank 3/4; condition `1e8` equality/outward ULP;
  amplitude and RMSE equality/outward ULP; one SVD; ordered binary64 RMSE; signed
  and zero Q; and nonrepresentable Q. Monkeypatch process CPU and collaborators
  for raised `Exception` and identical `BaseException`. Run the focused fit
  file; expect incomplete status paths.
- [ ] **GREEN:** Apply gates exactly in design order with one final scaled
  Jacobian SVD. Returned scientific failures are records; ordinary raised
  exceptions escape to the tracker construction boundary; `BaseException`
  remains identical. Record a finite nonnegative process-CPU delta on every
  completed scan. Re-run the full fit file; expect green.
- [ ] Run the Mandatory Per-Task Gate. Commit with
  `git commit -m "feat: enforce sparse fit quality gates"`.

---

### Task 7: Composite Tracker Reset, Geometry Stop, Global Schedule, and Reservation

**Files:** Create `src/odmr_bench/estimators/sparse_linewidth_tracker.py`,
`tests/estimators/test_sparse_linewidth_tracker.py`, and
`tests/estimators/test_sparse_linewidth_atomicity.py`; modify estimator exports,
`PROJECT_STATE.md`, and `CHANGELOG.md`.

**Interface:** Implement the fixed public `SparseLinewidthCompositeTracker`
constructor, `reset`, `choose_next_query`, and `estimate`; reserve `update` for
Tasks 8-10.

**First RED witness:**

```python
def test_first_sparse_scan_is_due_after_exactly_eight_fast_pairs() -> None:
    tracker = reset_tracker()
    accept_fast_pairs(tracker, count=8)
    query = tracker.choose_next_query()
    assert isinstance(query, SparseLinewidthQuery)
    assert (query.scan_index, query.resonance_id) == (0, "r0")
```

- [ ] **RED:** Test reset-code precedence and value-equal rollback; prospective
  geometry for all IDs; calibration-seeded centers/FWHMs/epochs; first scan only
  after eight completed pairs; target `source_ids[completed_scans%8]`; geometry
  before affordability; sequential two/five atom reservation at exact ceilings;
  no multiply/subtract/sum; same pending object; no fallback; and atomic
  `budget_exhausted` or `sparse_geometry_unavailable` with full diagnostic and
  no query. Run tracker/atomicity files; expect missing tracker.
- [ ] **GREEN:** Build one composite state machine and global sequence/time
  recurrence. Do not wrap a running Stage 6.3 tracker. Reset calls the Task 4
  prospective validator. A later invalid due geometry changes only stop fields;
  a valid affordable block freezes every query before exposing point one.
  Re-run RED; expect reset/schedule selections green.
- [ ] Run the Mandatory Per-Task Gate. Commit with
  `git commit -m "feat: schedule composite sparse tracking"`.

---

### Task 8: Exact Fast-Pair Branch and Stage 6.3 Differential

**Files:** Modify `src/odmr_bench/estimators/sparse_linewidth_tracker.py`,
`tests/estimators/test_sparse_linewidth_tracker.py`, and
`tests/estimators/test_sparse_linewidth_atomicity.py`; modify
`PROJECT_STATE.md` and `CHANGELOG.md`.

**Interface:** Complete the fast branch of `update(observation) ->
SparseLinewidthCompositeUpdate`, translating internal Stage 6.3 errors to the
new sparse observation/construction error types.

**First RED witness:**

```python
def test_fast_only_trace_is_stage_63_differentially_identical() -> None:
    legacy_results = run_legacy_fast_trace(observations)
    composite_results = run_composite_fast_trace(observations)
    assert composite_results == legacy_results
```

- [ ] **RED:** Compare pure-fast traces to Stage 6.3 for all eight IDs, repeated
  rounds, side parity, discriminator, capture/domain/common-mode, source refresh,
  and lost-pair cases. Test exact validation precedence, first-side partial,
  second-side result then identity construction, no sparse relabeling, no
  acceptance across gaps, and rollback at each construction code/ordinary
  exception/identical `BaseException`. Run `.venv/bin/python -m pytest
  tests/estimators/test_sparse_linewidth_tracker.py
  tests/estimators/test_sparse_linewidth_atomicity.py -k fast -q`; expect missing
  fast update behavior.
- [ ] **GREEN:** Reuse reviewed Stage 6.3 numerical/validation helpers privately,
  but catch and translate every public composite failure. Preserve pair semantics
  exactly while committing global clocks/resources/CPU only after prospective
  aggregate and update construction. Re-run RED plus the full Stage 6.3 tracker
  suite; expect bitwise/value differential equality.
- [ ] Run the Mandatory Per-Task Gate. Commit with
  `git commit -m "feat: preserve fast pairs in composite tracker"`.

---

### Task 9: Sparse Query Block and Partial-Scan Transitions

**Files:** Modify `src/odmr_bench/estimators/sparse_linewidth_tracker.py`,
`tests/estimators/test_sparse_linewidth_tracker.py`, and
`tests/estimators/test_sparse_linewidth_atomicity.py`; modify
`PROJECT_STATE.md` and `CHANGELOG.md`.

**Interface:** Complete sparse update points one through four, yielding exact
`SparsePartialScan` snapshots and normal `SparseLinewidthCompositeUpdate` values.

**First RED witness:**

```python
def test_first_four_sparse_points_are_frozen_partial_transitions() -> None:
    tracker = tracker_at_due_scan()
    frozen = tuple(accept_next_sparse_point(tracker) for _ in range(4))
    estimate = tracker.estimate()
    assert len(estimate.incomplete_sparse_scan.observations) == 4
    assert estimate.completed_sparse_scans == 0
    assert tuple(update.completed_sparse_scan for update in frozen) == (None,) * 4
```

- [ ] **RED:** Test sparse echo/sequence/frequency/integration/endpoint/nominal-
  exposure/value precedence; five frozen queries and exact source snapshots;
  partial lengths 1..4; no fit/history/FWHM/parity/counter advance; charge each
  accepted point in sparse and interleaved order; next pending query identity;
  and rollback for partial/resource/aggregate/update failures. Run tracker and
  atomicity files with `-k 'sparse and (partial or validation)'`; expect failures.
- [ ] **GREEN:** Translate all sparse-specific validation through the new error
  union. Construct each prospective partial, resources, aggregate, and update
  before one commit; never consult current centers/widths to alter points 2..5.
  Re-run RED; expect green.
- [ ] Run the Mandatory Per-Task Gate. Commit with
  `git commit -m "feat: track partial sparse scans"`.

---

### Task 10: Sparse Completion, Live Projection, Epochs, Resources, and CPU

**Files:** Modify `src/odmr_bench/estimators/sparse_linewidth_tracker.py`,
`tests/estimators/test_sparse_linewidth_types.py`,
`tests/estimators/test_sparse_linewidth_tracker.py`,
`tests/estimators/test_sparse_linewidth_atomicity.py`, `PROJECT_STATE.md`, and
`CHANGELOG.md`.

**Interface:** Complete point-five update, scientific success/failure, all
aggregate/resource/epoch fields, and `update_cpu_time_s`.

**First RED witness:**

```python
def test_sparse_success_refreshes_width_but_never_fast_center() -> None:
    before = tracker.estimate().identities[0]
    update = accept_fifth_sparse_point(tracker)
    after = update.estimate.identities[0]
    assert after.fast_center_hz == before.fast_center_hz
    assert after.fwhm_source_kind == "scan"
    assert after.live_q == after.fast_center_hz / after.active_fwhm_hz
```

- [ ] **RED:** Test public midpoint left fold and neighboring ULPs; fifth release;
  success refresh versus scientific-failure retention; local center never feeds
  fast center; asynchronous signed/zero live Q and nonrepresentable division;
  separate center/width ages; exact calibration/fast/sparse/interleaved/charged
  ledgers for both treatments; history/counters/parity; arrival-order mode and
  total CPU folds where total differs from subtotal addition; and every sparse
  construction rollback. Run all sparse estimator tests; expect incomplete
  completion/aggregate behavior.
- [ ] **GREEN:** Call the isolated fitter once, advance scan count/parity for
  success or scientific failure, update FWHM only on success, and preserve fast
  center value/source across completion. Measure accepted update CPU with
  `process_time_ns`; left-fold the mode subtotal and global total independently.
  Re-run all sparse estimator tests; expect green.
- [ ] Run the Mandatory Per-Task Gate. Commit with
  `git commit -m "feat: complete composite linewidth updates"`.

---

### Task 11: Private Runner-Neutral Verified-Calibration Authority

**Files:** Modify `src/odmr_bench/evaluation/two_point/provenance.py`,
`src/odmr_bench/evaluation/two_point/calibration.py`,
`src/odmr_bench/evaluation/two_point/runner.py`,
`tests/evaluation/test_two_point_provenance.py`,
`tests/evaluation/test_two_point_calibration.py`, and
`tests/evaluation/test_two_point_runner.py`, `PROJECT_STATE.md`, and
`CHANGELOG.md`.

**Interfaces:** Add private `_VerifiedCalibrationIssuer` holding exact runner,
instrument, run-token, and configuration identities; a private exact-class
runner registration/issuer path; and `_acquire_verified_calibration_core(issuer,
frequency_hz, integration_time_s, fit_configuration, identity_binding, *,
source_id, source_clock_id, tracker_clock_id, source_to_tracker_offset_s,
physical_fit_epoch_rule) -> VerifiedTwoPointCalibrationOutcome` with the exact
ledger annotations. Public Stage 6.3 signatures remain byte-for-byte/type-for-
type unchanged.

**First RED witness:**

```python
def test_public_allocations_cannot_mint_verified_calibration_authority() -> None:
    forged = object.__new__(_VerifiedCalibrationIssuer)
    with pytest.raises(TypeError, match="registered exact runner"):
        _acquire_verified_calibration_core(forged, **legal_acquisition_arguments())
```

- [ ] **RED:** Snapshot Stage 6.3 calibration success/failure, resource atoms,
  identity, token, exception rollback, and runner traces. Add attacks using
  subclasses, copies, serialization, public constructors/factories,
  `object.__new__`, wrong runner/instrument/token/configuration, and unregistered
  exact types. Run the three Stage 6.3 evaluator files; expect new private API
  tests red and old tests green.
- [ ] **GREEN:** Extract the transaction into the private runner-neutral core;
  allow only exact registered runner classes and exact live instances to receive
  issuers. Adapt Stage 6.3 internally without changing public objects or
  semantics. Re-run RED; expect authority attacks rejected and differential
  traces identical.
- [ ] Run Mandatory Gate and commit with
  `git commit -m "refactor: generalize verified calibration authority"`.

---

### Task 12: Evaluator Contracts, Signatures, States, and Outcomes

**Files:** Create `src/odmr_bench/evaluation/sparse_linewidth/types.py`, package
`__init__.py`, and `tests/evaluation/test_sparse_linewidth_types.py`; modify
`PROJECT_STATE.md` and `CHANGELOG.md`.

**Interfaces:** Define every exact spec evaluator record and alias:
`SparseRunnerPhase`, `SparseAbortReason`, `SparsePreflightCode`,
`SparseStartCode`, acquisition/unavailable/query-failure/abort/timing/resources,
state, five step outcomes, run union, and the three evaluator error classes.
`SparseEvaluatorRunnerState` includes
`pair_timings: tuple[TwoPointEvaluatorPairTiming, ...]` and
`scan_timings: tuple[SparseLinewidthEvaluatorScanTiming, ...]` alongside exact
`fast_update_cpu_time_s`, `sparse_update_cpu_time_s`, and
`total_update_cpu_time_s` mirrors of the tracker estimate.

**First RED witness:**

```python
def test_sparse_runner_phase_is_closed_and_geometry_is_distinct() -> None:
    assert get_args(SparseRunnerPhase) == (
        "ready", "calibration_succeeded", "calibration_failed", "tracking",
        "budget_stopped", "geometry_stopped", "externally_stopped", "aborted",
    )
```

- [ ] **RED:** Assert exact fields, frozen/slots, defensive tuples, canonical
  exception strings, all joins and phase presence, outcome-kind equality,
  unavailable-versus-authenticated abort matrix, pair/scan timing joins, CPU
  equality, and `inspect.signature` for all seven runner entry points from the
  spec ledger. Run the new types file; expect import failures.
- [ ] **GREEN:** Implement intrinsic validators and closed code sets without a
  free metadata map. The phase set is exactly ready, calibration_succeeded,
  calibration_failed, tracking, budget_stopped, geometry_stopped,
  externally_stopped, and aborted. Step/run unions exclude external stop from
  `run_until_event`. Re-run RED; expect green.
- [ ] Run Mandatory Gate and commit with
  `git commit -m "feat: define sparse evaluator contracts"`.

---

### Task 13: Evaluator Full/Safe Resource Builder

**Files:** Create
`src/odmr_bench/evaluation/sparse_linewidth/resource_accounting.py` and
`tests/evaluation/test_sparse_linewidth_resources.py`; modify
`src/odmr_bench/evaluation/sparse_linewidth/__init__.py`, `PROJECT_STATE.md`,
and `CHANGELOG.md`.

**Interface:** Implement
`build_sparse_linewidth_evaluator_resources(runner) ->
SparseLinewidthEvaluatorResources | None`.

**First RED witness:**

```python
def test_tracking_total_replays_interleaved_atoms_not_mode_subtotals() -> None:
    resources = build_sparse_linewidth_evaluator_resources(interleaved_runner())
    assert resources.tracking_resources == replay_full_atoms(arrival_order_atoms)
    assert resources.tracking_resources != add_snapshots(
        resources.fast_tracking_resources, resources.sparse_tracking_resources
    )
```

- [ ] **RED:** Cover both calibration treatments, calibration/fast/sparse/global
  arrival-order replays, partial and scientifically failed scans, expected and
  realized photons, exact full-to-safe identity joins, included source charged
  once, conditional source uncharged, zero-or-one authenticated unaccepted atom,
  unavailable join returning `None`, and no float tolerances/subtotal addition.
  Run the new resource file; expect missing builder.
- [ ] **GREEN:** Replay immutable atoms left-associatively from zero in each
  ledger; form total charged from source prefix then global accepted stream and
  optional authenticated atom. Never infer full fields from safe records. Re-run
  RED; expect green.
- [ ] Run Mandatory Gate and commit with
  `git commit -m "feat: account sparse evaluator resources"`.

---

### Task 14: Composite Runner Bind, Verified Calibration, and Start

**Files:** Create `src/odmr_bench/evaluation/sparse_linewidth/runner.py` and
`tests/evaluation/test_sparse_linewidth_runner.py`; modify
`src/odmr_bench/evaluation/sparse_linewidth/__init__.py`, `PROJECT_STATE.md`,
and `CHANGELOG.md`.

**Interfaces:** Implement exact `bind`, read-only `state`,
`acquire_verified_calibration`, and `start_tracking` signatures from the ledger;
register the exact sparse runner through Task 11's private API.

**First RED witness:**

```python
def test_start_rejects_calibration_mismatch_before_tracker_reset(spy_tracker) -> None:
    runner = calibrated_runner()
    with pytest.raises(SparseStartError) as caught:
        runner.start_tracking(spy_tracker, mismatched_calibration,
                              runner.state.verified_calibration,
                              public_metadata, budget_ceiling, seed=0)
    assert caught.value.code == "calibration_mismatch"
    assert spy_tracker.reset_calls == 0
```

- [ ] **RED:** Assert bind accepts only an exact clean instrument and captures
  exact token/config/resource boundary. Test acquire only in ready, exact-type
  then value/grid/fit-identity/clock/clean-boundary precedence, success/failure
  phases, and rollback. Test start in ready only for authenticated conditional
  other-runner source or calibration_succeeded for its exact outcome; pin the
  ordered start codes, all identity/provenance/metadata/resource joins, reset
  translation, and no tracker/instrument touch on failure. Run runner tests with
  `-k 'bind or calibration or start'`; expect missing runner behavior.
- [ ] **GREEN:** Delegate acquisition only through a private issuer. Authenticate
  treatment/source/token/runner/instrument/clocks/boundary before tracker reset;
  success enters tracking with zero CPU and empty traces/timings. Re-run RED and
  complete Stage 6.3 evaluator suites; expect all green.
- [ ] Run Mandatory Gate and commit with
  `git commit -m "feat: start sparse evaluator runs"`.

---

### Task 15: Accepted Runner Steps, Retry, and Pair/Scan Timing

**Files:** Modify `src/odmr_bench/evaluation/sparse_linewidth/runner.py`,
`tests/evaluation/test_sparse_linewidth_runner.py`, `PROJECT_STATE.md`, and
`CHANGELOG.md`.

**Interface:** Implement the accepted and retryable-instrument-failure branches
of `step() -> SparseRunnerStepOutcome`.

**First RED witness:**

```python
def test_completed_scan_joins_public_and_truth_timing_once() -> None:
    outcome = accept_one_complete_scan(runner)
    timing = outcome.state.scan_timings[-1]
    assert timing.public_reference_timestamp_s == ordered_mean(public_midpoints)
    assert timing.truth_reference_timestamp_s == ordered_mean(actual_midpoints)
    assert instrument.truth_lookup_calls == 1
```

- [ ] **RED:** For both modes, assert expected midpoint captured pre-query,
  returned full/safe identity, exact resource before/after/delta join, normal
  trace equals accepted stream, tracker update echo, runner CPU equals estimate,
  and instrument exceptions retain equal resource boundaries/pending query and
  nonterminal tracking state. Complete pairs must append exactly one existing
  `TwoPointEvaluatorPairTiming`; complete scans append exactly one five-midpoint
  timing using ordered-mean folds for actual truth and public endpoint-derived
  values, with release joins and exactly one truth lookup after release. Run
  runner tests with `-k 'accepted or retry or timing'`; expect failures.
- [ ] **GREEN:** Query once per step, keep full observations evaluator-only, and
  commit trace/timing/state only after tracker acceptance. A pre-return ordinary
  instrument exception produces `SparseRunnerInstrumentFailure` and no charge;
  `BaseException` propagates. Re-run RED; expect green.
- [ ] Run Mandatory Gate and commit with
  `git commit -m "feat: step sparse evaluator tracking"`.

---

### Task 16: Clean Stops, Aborts, Run Loop, and Phase Matrix

**Files:** Modify `src/odmr_bench/evaluation/sparse_linewidth/runner.py`,
`src/odmr_bench/evaluation/sparse_linewidth/resource_accounting.py`,
`src/odmr_bench/evaluation/sparse_linewidth/types.py`,
`tests/evaluation/test_sparse_linewidth_types.py`,
`tests/evaluation/test_sparse_linewidth_resources.py`,
`tests/evaluation/test_sparse_linewidth_runner.py`, `PROJECT_STATE.md`, and
`CHANGELOG.md`.

**Interfaces:** Complete `step`, `run_until_event`, and `stop_external` for exact
budget/geometry/abort/external outcomes and terminal state preservation.

**First RED witness:**

```python
def test_unavailable_join_aborts_without_fabricated_resources() -> None:
    outcome = corrupt_next_resource_join(runner).step()
    assert isinstance(outcome, SparseRunnerAborted)
    assert outcome.abort.reason == "resource_join_unavailable"
    assert outcome.resources is None
    assert outcome.abort.tracker_estimate_before == outcome.abort.tracker_estimate_after
```

- [ ] **RED:** Cover all eight phases and illegal-call error classes; clean
  budget/geometry stops before query; exact diagnostic/resources joins; external
  stop preserving pending/partial blocks; run loop continuing only accepted
  outcomes and returning first retryable/terminal member; resource-unavailable
  abort with unavailable atom/no exception strings/`resources=None`; the three
  authenticated validation/construction/unexpected abort reasons with canonical
  strings/resources; value-equal tracker rollback; one unaccepted atom; no CPU
  increments; and identical `BaseException` propagation after cleanup. Run all
  sparse evaluator tests; expect terminal paths incomplete.
- [ ] **GREEN:** Implement state transitions transactionally. Query failure is
  nonterminal; any returned but unaccepted observation is terminal. Terminal
  state is immutable/read-only and preserves global arrival-order compute
  accounting. Re-run all sparse evaluator tests; expect green.
- [ ] Run Mandatory Gate and commit with
  `git commit -m "feat: finalize sparse evaluator state machine"`.

---

### Task 17: Deterministic Linewidth Dynamics

**Files:** Create `src/odmr_bench/dynamics/linewidth_drift.py` and
`tests/dynamics/test_linewidth_drift.py`; modify
`src/odmr_bench/dynamics/__init__.py`, `PROJECT_STATE.md`, and `CHANGELOG.md`.

**Interface:** Add frozen/slotted `LinearLinewidthDrift(base_dynamics:
SpectralDynamics, reference_fwhm_hz: Mapping[str,float],
fwhm_slew_hz_per_s: float | Mapping[str,float])` with
`snapshot_at(timestamp_s) -> SpectralSnapshot`.

**First RED witness:**

```python
def test_linewidth_drift_composes_without_changing_base_center() -> None:
    dynamics = LinearLinewidthDrift(center_drift, reference_widths, 100.0)
    actual = dynamics.snapshot_at(2.0)
    base = center_drift.snapshot_at(2.0)
    assert actual.resonances[0].center_hz == base.resonances[0].center_hz
    assert actual.resonances[0].fwhm_hz == reference_widths["r0"] + 200.0
```

- [ ] **RED:** Test scalar/exact-ID slews, canonical immutable mappings, rejected
  types/nonfinite/missing/extra IDs, deterministic calls, negative/nonfinite
  time, no input mutation, and composition over `LinearCenterDrift`: centers,
  amplitudes, eta, and baseline equal the base snapshot at t while each width is
  reference plus slew*t. Reject generated nonpositive/nonfinite widths. Run the
  new dynamics test; expect missing class.
- [ ] **GREEN:** Reuse `validate_timestamp_s`, preserve tuple order/IDs, and
  return a new physical snapshot without stochastic state or callbacks. Re-run
  RED; expect green.
- [ ] Run Mandatory Gate and commit with
  `git commit -m "feat: add deterministic linewidth dynamics"`.

---

### Task 18: Closed Scientific Regressions and Acceptance Scenarios

**Files:** Create `tests/evaluation/test_sparse_linewidth_regressions.py` and
`tests/evaluation/fixtures/sparse_linewidth_acceptance.py`, `PROJECT_STATE.md`,
and `CHANGELOG.md`.

**Interfaces:** Fixture functions build deterministic public configurations,
sources, dynamics, instruments, and budgets; they expose no estimator truth
handle and own no assertion/tolerance.

**First RED witness:**

```python
def test_exact_static_acceptance_recovers_local_linewidth() -> None:
    outcome = run_acceptance_case(exact_static_case())
    scan = outcome.state.tracker_estimate.sparse_scan_history[0]
    assert scan.status == "success"
    assert scan.fitted_fwhm_hz == pytest.approx(EXACT_STATIC_FWHM_HZ)
    assert scan.fitted_q == scan.fitted_local_center_hz / scan.fitted_fwhm_hz
```

- [ ] **RED:** Add named exact-static, seeded-Poisson-static, composed center/
  width drift, independent-source-epoch, included-accounting, retained-width
  scientific failure, prospective/due geometry stop, affine-baseline mismatch,
  and within-scan-dynamics cases. Assert exact noiseless four-parameter recovery,
  fixed stochastic tolerances, identity/source/timing/resource/CPU joins, scan
  and asynchronous signed Q meanings, and block indivisibility. Run the new
  regression file; expect any integration contract defect to fail by name.
- [ ] **GREEN:** Correct only fixture/test construction in this task. Production
  defects return to the owning task with the failing node before proceeding.
  Mismatch cases assert coherent diagnostics and documented limitation, never
  unbiased recovery or performance. Audit with `rg -n
  "current_snapshot|snapshot_at|truth|_dynamics" src/odmr_bench/estimators
  src/odmr_bench/evaluation/sparse_linewidth` and a spy test proving truth is
  read exactly once only after release. Re-run regression plus focused sparse
  tracker/runner/resources tests; expect green.
- [ ] Run Mandatory Gate and commit with
  `git commit -m "test: lock sparse linewidth acceptance"`.

---

### Task 19: Public Exports, Documentation, Example, and Wheel Smoke

**Files:** Modify `src/odmr_bench/estimators/__init__.py`,
`src/odmr_bench/evaluation/sparse_linewidth/__init__.py`,
`src/odmr_bench/dynamics/__init__.py`, `tests/test_package.py`, `README.md`,
`docs/estimators.md`, `PROJECT_STATE.md`, and `CHANGELOG.md`; create
`examples/track_sparse_linewidth.py`.

**Interfaces:** Export all approved configuration/records/errors/tracker,
runner/states/outcomes/resources/builder, and linewidth dynamics; keep fit,
authority, token, registration, and model helpers private. The exact export
names are the public classes, aliases, and functions allocated in Tasks 1, 2,
7, 10, 12-14, and 17; no new callable signature is introduced here.

**First RED witness:**

```python
def test_installed_stage_64_public_surface() -> None:
    from odmr_bench.dynamics import LinearLinewidthDrift
    from odmr_bench.estimators import SparseLinewidthCompositeTracker
    from odmr_bench.evaluation.sparse_linewidth import SparseLinewidthEvaluatorRunner
    assert LinearLinewidthDrift and SparseLinewidthCompositeTracker
    assert SparseLinewidthEvaluatorRunner
```

- [ ] **RED:** Assert exact `__all__`, source-tree imports, and installed-wheel
  imports. Add a download-free conditional-precalibration example that uses
  only public APIs and prints terminal phase, asynchronous live projection,
  scan-local Q, both source epochs, ledgers, pair/scan counts, and public/truth
  labels. Run `.venv/bin/python -m pytest tests/test_package.py -q` and
  `.venv/bin/python examples/track_sparse_linewidth.py`; expect missing exports
  or example.
- [ ] **GREEN:** Document offsets/orders, free/frozen parameters, gate/presence
  table, Q distinction, epochs/times, ledgers/treatments, stop/failure/abort/
  partial behavior, mismatch limitations, and all Stage 6.5/6.6 nonclaims.
  Build into `temp_dir="$(mktemp -d)"` with `.venv/bin/python -m build --outdir
  "$temp_dir/dist"`, create an isolated venv, install `"$temp_dir"/dist/*.whl`,
  and import the three public entry points. Expect build/install/import/example
  green; remove only the validated mktemp directory.
- [ ] Run Mandatory Gate and commit with
  `git commit -m "docs: publish sparse linewidth tracking workflow"`.

---

### Task 20: Integrated Scientific and Software Gates

**Files:** Review `src/odmr_bench/estimators/sparse_linewidth_types.py`,
`sparse_linewidth_fit.py`, `sparse_linewidth_tracker.py`, and
`two_point_calibration.py`; `src/odmr_bench/evaluation/two_point/provenance.py`,
`calibration.py`, and `runner.py`; every file in
`src/odmr_bench/evaluation/sparse_linewidth/`;
`src/odmr_bench/dynamics/linewidth_drift.py`; all `test_sparse_linewidth_*`
files declared above; `tests/dynamics/test_linewidth_drift.py`,
`tests/test_package.py`, `README.md`, `docs/estimators.md`, and
`examples/track_sparse_linewidth.py`. Modify a reviewed source/test/doc only in
a separate named fix wave with its reproducing RED test and atomic commit.
After closure modify only `PROJECT_STATE.md` and `CHANGELOG.md`.

**Interfaces:** Consume every fixed public/private signature in the Cross-Task
Interface Ledger. Produce no new API; the only closeout outputs are focused
regression fixes, verified project-state evidence, and the changelog record.

**Review RED rule:**

```python
@pytest.mark.regression
def test_estimator_graph_retains_no_truth_capability() -> None:
    estimate = completed_tracker.estimate()
    assert recursively_find_forbidden_types(estimate) == ()
```

Every additional finding gets an equally concrete named node before its fix;
each fix wave records that exact node ID in `PROJECT_STATE.md`.

- [ ] **RED — baseline and review gate:** In a clean tree run `.venv/bin/python -m pytest
  tests/estimators tests/evaluation tests/emulator tests/dynamics -q`, then
  `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check .`,
  `git diff --check`, and `git status --short`. Expect both pytest commands and
  Ruff green, silent diff check, and clean status; record exact counts.
- [ ] **Scientific review:** Verify offsets/parity/time-frequency sum limitation,
  four free/frozen terms, every numerical precedence/presence ULP, signed Q,
  frozen source echoes, failure retention, no center feedback, asynchronous
  epochs, left-fold CPU/resources/times, geometry-before-budget, truth isolation,
  and mismatch/nonclaim wording. Every finding gets a RED and fix-wave commit.
- [ ] **Software review:** Verify immutable copies, exact closed errors/signatures/
  phases/unions, identity/resource joins, retry/rollback, `Exception` versus
  `BaseException`, exact authority allowlist/non-mintability, unchanged Stage 6.3
  differential traces, imports, and wheel contents. Every finding gets a RED
  and fix-wave commit.
- [ ] **GREEN — final gate:** After every finding has its reproducing RED and
  atomic fix, run focused Stage 6.3 calibration/tracker/runner/provenance
  suites, sparse acceptance, two consecutive `.venv/bin/python -m pytest -q`
  runs, `.venv/bin/ruff check .`, `.venv/bin/python -m build`,
  `git diff --check`, and `git status --short`. Expect all green and clean.
- [ ] Record commands/counts/wheel names and retained declared limitations. Mark
  implemented only if all gates pass; commit only records with
  `git commit -m "docs: close sparse linewidth tracking stage"`. Otherwise leave
  Stage 6.4 pending and do not make the closeout commit.
