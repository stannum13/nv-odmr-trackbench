# Offline Oracle and Repeated Full-Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, constrained eight-component Lorentzian and
pseudo-Voigt offline fitter plus a cold-start repeated full-sweep estimator.
Its success label is conditional on the model, initializer, and configured
quality thresholds, not proof of eight physical resonances.

**Implementation status (2026-09-04):** Tasks 1–4, the integrated-review fix
set, and the final fluorescence-origin/provenance follow-up are implemented
locally. Tests, lint, build, isolated-wheel smoke, example execution, and
independent re-review are clean; only authorized synchronization remains.

**Architecture:** Validated immutable sweep/config/result contracts isolate raw
observations from truth. A deterministic initializer produces an explicit guess
and diagnostics. A scaled bounded least-squares core reuses the repository's
FWHM-native spectrum model, and a causal wrapper fits each completed sweep
independently without warm starts.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`signal`, `optimize`), pytest, Ruff.

## Global Constraints

- All scientifically important implementation lives under
  `src/odmr_bench/estimators/` and is importable from `odmr_bench.estimators`.
- Public frequency and linewidth values are Hz; FWHM is the only public
  linewidth convention and `Q = center_hz / fwhm_hz`.
- Every successful fit contains exactly eight unique model-component
  identities; centers are strictly ordered for this resolved, non-crossing
  oracle scope. This structural contract does not establish physical identity
  or the existence of eight physical lines.
- Lorentzian mode fixes `eta = 1`; pseudo-Voigt mode bounds every eta to
  `[0, 1]`.
- Fits never receive synthetic truth, future sweeps, hidden dynamics,
  signal-conditioned expected photons, or an offline reference trajectory.
- Invalid arrays/configuration fail before optimization. Scientifically valid
  inputs that cannot initialize or converge return a structured unsuccessful
  result and are never silently dropped.
- The optional Figshare recording provides apparent fitted observables only;
  no real-data truth RMSE, photon calibration, timing, adaptive, or
  eight-identity claim is permitted.
- Initialization and fitting are deterministic for identical inputs and
  configuration.
- Tests and CI use generated spectra and tiny fixtures only; the external file
  is never downloaded implicitly or required.
- Use `.venv/bin/python -m pytest` for RED and GREEN runs so test import
  environments are identical.
- Each implementation task follows red/green TDD, focused and full tests, Ruff,
  diff inspection, scientific/software self-review, one local commit, and an
  independent review before the commit is pushed to `origin/main`.

## Exact result and failure contract

The public frozen slotted records have these stable fields:

```python
CompleteSweep(
    frequency_hz,
    fluorescence,
    last_sequence_index=None,
    last_timestamp_s=None,
    total_integration_time_s=None,
    total_nominal_exposure_photons=None,
)

InitializationDiagnostics(
    source,                 # "detected", "fallback", "user", or "none"
    candidate_count,
    selected_indices,
    used_fallback,
    messages,
)

FitUncertainty(
    baseline_standard_errors,  # length 2 or 3: intercept, slope, quadratic
    center_hz,
    fwhm_hz,
    amplitude,
    eta,                       # None for Lorentzian, length 8 otherwise
    method="local_linearized_jacobian_covariance",
)

SpectrumFitResult(
    success,
    failure_code,           # stable code below, None only on success
    model_kind,
    baseline_degree,
    resonance_estimates,    # length 8 on success, empty on failure
    baseline_estimate,      # Baseline on success, None on failure
    q_values,               # length 8 on success, empty read-only array otherwise
    diagnostics,
    initial_guess,          # immutable attempted guess; None before optimization
    uncertainty,
    uncertainty_reason,
    scipy_status,
    scipy_message,
    nfev,
    cost,                   # raw squared-fluorescence half sum of squared residuals
    residual_rmse,          # raw fluorescence units
    residual_scale,         # positive fluorescence units used internally
    degrees_of_freedom,
    jacobian_rank,
)

SweepEstimate(
    fit,
    last_sequence_index,
    last_timestamp_s,
    total_integration_time_s,
    total_nominal_exposure_photons,
)
```

Stable failure codes are `initialization_failed`, `insufficient_samples`,
`uninformative_sweep`, `optimization_failed`, and `quality_failed`. SciPy status remains a separate
optional integer. Invalid types, non-finite arrays, configuration incompatibility,
or an invalid caller-supplied guess raise before optimization. Scientifically
valid data that cannot identify or converge to eight model components return an
unsuccessful result. A failed fit carries its completed sweep's metadata and
resource totals in `SweepEstimate`.

Diagnostics combinations are validated as follows:

| `source` | `candidate_count` | `selected_indices` | `used_fallback` |
|---|---:|---|---|
| `detected` | at least 8 | exactly 8 distinct non-negative indices | `False` |
| `fallback` | 0 or more detected candidates | empty | `True` |
| `user` | 0 | empty | `False` |
| `none` | 0 or more detected candidates | empty | `False` |

Diagnostic provenance and optimizer state are also cross-validated. `none` is
restricted to pre-optimization failures; `user` requires an optimizer attempt;
and `detected` or `fallback` accompanies an optimizer attempt or an
`initialization_failed` result when the generated guess itself fails numerical
preflight. Sample-count and uninformative-sweep preflights always use `none`.

The result scalar fields use these types:
`scipy_status: int | None`, `scipy_message: str | None`, `nfev: int`,
`cost: float | None`, `residual_rmse: float | None`,
`residual_scale: float | None`, `degrees_of_freedom: int`, and
`jacobian_rank: int | None`. `initial_guess: FitInitialGuess | None` is a
defensive immutable snapshot of the actual detected, fallback, or caller-
supplied guess used for an optimizer attempt, never synthetic truth.

| result state | optimizer fields | residual fields | rank/uncertainty |
|---|---|---|---|
| `initialization_failed` | status/message `None`, `nfev=0` | positive scale; cost/RMSE `None` | rank `None`; uncertainty `None` with required reason; initial guess `None` |
| `insufficient_samples` | status/message `None`, `nfev=0` | scale/cost/RMSE `None` because sample-count preflight runs first | rank `None`; uncertainty `None` with required reason; initial guess `None` |
| `uninformative_sweep` | status/message `None`, `nfev=0` | scale/cost/RMSE `None` because `ptp(y)` is zero or non-finite | rank `None`; uncertainty `None` with required reason; initial guess `None` |
| `optimization_failed` | integral status `<=0`, nonempty message, `nfev>0` | positive scale; finite cost/RMSE retained when available, otherwise `None` | rank `None`; uncertainty `None` with required reason; initial guess present |
| `quality_failed` | integral status `>0`, nonempty message, `nfev>0` | positive scale and normally finite cost/RMSE; both are `None` only when a nominally successful optimizer returns non-finite parameters, residuals, or cost, and that special reason is invalid when finite metrics are present | computed rank retained when available; uncertainty `None` with required reason; initial guess present |
| success | integral status `>0`, nonempty message, `nfev>0` | positive scale and finite cost/RMSE | full rank; uncertainty present or `None` only with a required covariance reason; initial guess present |

All unsuccessful results have empty final resonance/Q arrays and no final
baseline. Constructors reject field combinations that contradict this table;
NaN is never used as a missing-value sentinel.

The number of free parameters is `baseline_degree + 1 + 8 * 3` for Lorentzian
and eight larger for pseudo-Voigt. The fitter requires
`n_samples > n_free_parameters`; fewer or equal samples return
`insufficient_samples` without calling SciPy, giving
`degrees_of_freedom = n_samples - n_free_parameters`.

For post-fit identifiability, singular values `s` of the scaled residual
Jacobian define rank as `count(s > max(s) * rank_rtol)`. Success requires full
column rank, positive degrees of freedom, every fitted amplitude at least
`min_resolved_amplitude`, every fitted amplitude divided by its local
linearized public amplitude standard error to be at least
`min_amplitude_significance` (positive finite, default `5.0`), and

```text
(baseline_only_sse - fitted_sse) / baseline_only_sse
    >= min_baseline_sse_improvement
```

with a strictly positive finite baseline-only SSE. Flat data therefore cannot
be successful. If the baseline-only SSE is zero, the result is
`quality_failed`. A fallback guess may enter optimization, but it passes or
fails these same checks. The amplitude evidence uses the packed covariance and
fluorescence-scale factor before the unrelated full public transform is
attempted. A positive amplitude with zero standard error passes as positive
infinity; unavailable or non-finite evidence fails conservatively. This is a
model-conditioned local diagnostic, not calibrated detection evidence.
The threshold failure reason is exactly `one or more fitted amplitudes fail the
model-conditioned local min_amplitude_significance threshold`; unavailable or
non-finite evidence uses `model-conditioned local amplitude significance is
unavailable or non-finite`.

`baseline_only_sse` is the minimum unweighted SSE from
`numpy.linalg.lstsq` applied to the centered target `y-y_ref` for all finite
samples using columns `[1, z]` or `[1, z, z**2]`, where `z` uses the same sweep
midpoint and half-span as the fitter. `fitted_sse = 2 * cost` in raw
squared-fluorescence units. The quality gate tests a zero-SSE polynomial, noisy
seven-line spectra with seeds 1, 2, and 23
at noise sigma `2e-4` (including a false component inserted between true
lines and direct `+1e6` fluorescence shifts), and improvement values immediately
below, exactly at, and immediately above the configured threshold.

`RepeatedFullSweepEstimator.latest` is `None` before the first attempted fit and
otherwise refers to the most recently completed attempt, including a structured
failure. `history` is `tuple[SweepEstimate, ...]`. Cross-sweep monotonicity of
indices/timestamps belongs to the evaluator because independent recordings may
be submitted; the wrapper only preserves each sweep's own completion metadata.

## Fixed numerical regression fixture

All clean/noisy fit acceptance tests use one declared family unless a boundary
test states otherwise:

```text
frequency_hz = linspace(2.740e9, 3.020e9, 4481)  # 62.5 kHz spacing
centers_hz = 1e9 * [2.760, 2.794, 2.828, 2.862,
                    2.896, 2.930, 2.964, 2.998]
fwhm_hz = 1e6 * [1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20]
amplitude = [0.018, 0.021, 0.024, 0.027,
             0.030, 0.033, 0.036, 0.039]
eta = [0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.93]
baseline reference_hz = 2.880e9  # sweep midpoint
baseline intercept = 1.0
baseline slope_per_hz = 2.0e-11
baseline quadratic_per_hz2 = -5.0e-20  # zero for linear tests
explicit-guess center perturbation_hz =
    1e5 * [-2, 1, -1, 2, -2, 1, -1, 2]
explicit-guess width multiplier = 1.08
explicit-guess amplitude multiplier = 0.92
explicit-guess eta offset = 0.04, clipped to [0, 1]
noisy seed = 6104
Gaussian sigma in normalized fluorescence = 2.0e-4
```

The forced-fallback success regression is a separate exact fixture on the same
4481-point scan. With `d=1.0e6 Hz`, its margin is `e=0.5e6 Hz` and its truth
centers are exactly `linspace(2.7405e9, 3.0195e9, 8)`, matching the fallback
guess. It uses FWHM `2.0e6 Hz`, amplitudes
`[0.018, 0.021, 0.024, 0.027, 0.030, 0.033, 0.036, 0.039]`, eta `0.5`, and the
linear baseline above. A deliberately high discovery threshold forces
fallback. Every truth center is consequently inside its fallback midpoint box.

Clean explicit-guess tolerances are center error `< 0.01 * true_fwhm`, FWHM
relative error `< 0.05`, amplitude relative error `< 0.05`, eta absolute error
`< 0.08`, intercept absolute error `< 2e-4`, slope absolute error `< 5e-13`,
and quadratic absolute error `< 5e-21`. Noisy/auto-initialized tolerances are
center error `< 0.08 * true_fwhm`, FWHM relative error `< 0.15`, amplitude
relative error `< 0.12`, eta absolute error `< 0.20`, intercept absolute error
`< 8e-4`, slope absolute error `< 2e-12`, and quadratic absolute error
`< 2e-20`. These are regression bounds for fixed fixtures, not benchmark
results.

---

### Task 1: Immutable Sweep, Configuration, and Result Contracts

**Files:**
- Create: `src/odmr_bench/estimators/__init__.py`
- Create: `src/odmr_bench/estimators/types.py`
- Create: `tests/estimators/test_types.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: `CompleteSweep`, `FitConfiguration`, `FitInitialGuess`,
  `InitializationDiagnostics`, `FitUncertainty`, `SpectrumFitResult`, and
  `SweepEstimate`.
- Reuses: immutable `Baseline`, `Resonance`, and existing `q_factor`.
- All input NumPy arrays are copied, canonicalized to float64, and made
  read-only.

- [ ] **Step 1: Write failing `CompleteSweep` tests**

  Construct a valid sweep with 128 strictly increasing finite frequencies and
  fluorescence samples. Assert defensive copies, read-only arrays, and
  canonical optional `last_sequence_index`, `last_timestamp_s`,
  `total_integration_time_s`, and `total_nominal_exposure_photons`. These are
  sums over every observation in the completed sweep. Parameterize failures
  for empty/non-1D/mismatched/non-finite/non-increasing arrays, bool metadata,
  negative sequence/time/exposure, and non-positive integration.

- [ ] **Step 2: Run the sweep tests and observe the missing-package RED**

  Run `.venv/bin/python -m pytest tests/estimators/test_types.py -q` and confirm
  collection fails because `odmr_bench.estimators` does not exist.

- [ ] **Step 3: Implement `CompleteSweep` and scalar validators**

  Use a frozen slotted dataclass. The constructor copies both arrays before
  validation, rejects fewer than two samples, and never sorts, deduplicates, or
  removes values. Optional metadata remains `None` when unavailable.

- [ ] **Step 4: Write failing configuration/guess/result tests**

  Require `FitConfiguration` to canonicalize and validate:

  ```python
  FitConfiguration(
      model_kind="pseudo_voigt",       # or "lorentzian"
      baseline_degree=1,               # 1 or 2
      resonance_ids=tuple(f"r{i}" for i in range(8)),
      min_fwhm_hz=2.0e5,
      max_fwhm_hz=8.0e6,
      max_amplitude=0.25,
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

  Reject invalid model/degree, anything other than eight unique nonempty IDs,
  invalid width/amplitude/resolved-amplitude/significance/separation bounds,
  even or
  undersized Savitzky–Golay
  windows, `relative_prominence` outside `(0, 1]`, `rank_rtol` outside `(0, 1)`,
  improvement outside `[0, 1)`, invalid polynomial order/fallback/max
  evaluations, and bool-as-number inputs. Require `FitInitialGuess` to contain exactly eight
  ordered valid resonances and a baseline. Require diagnostics to record
  candidate count, selected indices, fallback flag, and messages immutably,
  enforcing the diagnostics invariant table above. `SpectrumFitResult` also
  retains a defensive immutable snapshot of the actual initial guess whenever
  optimization was attempted.

  Test successful `SpectrumFitResult` requires exactly eight ordered valid
  resonances and derives a read-only Q array using `q_factor`. Include a
  contract-only signed-coordinate case and assert a negative center produces
  negative Q without `abs`. An
  unsuccessful result must contain no resonance/baseline estimate, must retain
  status/diagnostics, and cannot claim uncertainties. Validate residual RMSE,
  cost, degrees of freedom, evaluation count, and optional uncertainty shapes.
  Parameterize all failure codes and reject every contradictory combination in
  the failure-field matrix above. For degree one, require zero quadratic terms
  in retained initial and final baselines. Successful fitted IDs/order and the
  final baseline reference must exactly match the retained initial guess.
  Diagnostic sources must be compatible with whether optimization was
  attempted. Every optimizer attempt requires an integral status, nonempty
  message, and positive `nfev`; success/quality failure require status `>0`,
  optimizer failure status `<=0`, and pre-optimizer failures retain
  `None`/`None`/`0`. Reject every `FitUncertainty.method` except
  `local_linearized_jacobian_covariance`. Derive Q under a guarded numerical
  context and reject a successful result whose Q is non-finite or
  unrepresentable.

- [ ] **Step 5: Implement contracts and verify Task 1**

  Implement only validation/immutability and derived Q behavior; no optimizer.
  Run focused tests, full pytest, Ruff, and `git diff --check`; inspect unit,
  truth-isolation, and failed-result semantics; update state/changelog and
  commit.

---

### Task 2: Deterministic Baseline-Aware Initialization

**Files:**
- Create: `src/odmr_bench/estimators/initialization.py`
- Create: `tests/estimators/test_initialization.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `CompleteSweep`, `FitConfiguration`.
- Produces:
  `initialize_spectrum(sweep, configuration) -> tuple[FitInitialGuess | None, InitializationDiagnostics]`.
- Never consumes truth or a previous fit.

- [ ] **Step 1: Add deterministic generated-spectrum fixtures**

  Create a test helper that evaluates eight well-separated pseudo-Voigt dips on
  a linear or quadratic baseline using the production spectral model. Truth is
  retained only in the test. Use at least 16 samples per narrowest FWHM and
  avoid symmetric/equal-depth candidates that create arbitrary tie ordering.

- [ ] **Step 2: Write and run failing initialization tests**

  Require two identical calls to return equal guesses/diagnostics. On clean
  linear and quadratic cases, selected centers must be within two grid steps,
  widths must lie inside configured bounds, amplitudes must be positive and no
  larger than `max_amplitude`, centers must be ordered, and fallback must be
  false. Run the focused file and confirm failure is the missing initializer.

- [ ] **Step 3: Implement deterministic trend and candidate discovery**

  Smooth with `scipy.signal.savgol_filter` only after verifying the configured
  window fits the sweep. Let `z=(f-midpoint)/half_span`. Begin with every sample
  in the baseline mask and repeat exactly three times: fit the configured
  polynomial to masked `(z, y)`; compute `r=y-trend`, `m=median(r)`,
  `mad=median(abs(r-m))`, and `sigma=1.4826*mad`; retain samples with
  `r >= m - 2.5*sigma`. When `mad == 0`, retain samples with `r >= m`. If fewer
  than `max(degree + 1, ceil(0.25*n_samples))` remain, return a structured
  initialization failure. Perform three rejection updates and then one final
  polynomial fit on the third updated mask; that fourth fit is the discovery
  trend. Regression-test the final coefficients and selected indices.

  Define the discovery vector as `depth=max(trend-smoothed_signal, 0)` and the
  prominence reference as `max(depth)`. Zero/non-finite depth is candidate
  scarcity. Detect all interior candidates with `scipy.signal.find_peaks`
  using `relative_prominence * max(depth)` and no sample-distance argument.
  Sort candidates by `(-prominence, sample_index)` and greedily retain one only
  when its actual frequency is at least `min_center_separation_hz` from every
  accepted candidate. Stop at eight, then sort those eight by frequency. This
  remains correct for strictly increasing nonuniform grids.

- [ ] **Step 4: Implement amplitude and FWHM guesses**

  Initialize amplitude from positive local detrended depth, clipped only to
  the declared upper bound. Use `scipy.signal.peak_widths` at half prominence
  on the same discovery vector; convert fractional left/right sample positions
  to Hz with `np.interp` over the actual frequency axis, then
  bound it to `[min_fwhm_hz, max_fwhm_hz]`. Initialize eta to `1.0` for
  Lorentzian and `0.5` for pseudo-Voigt. Convert the scaled baseline polynomial
  explicitly into `Baseline(intercept, reference_hz, slope_per_hz,
  quadratic_per_hz2)`. If a finite scaled polynomial cannot be represented in
  public per-Hz units, including nonzero coefficients that underflow to zero,
  return `source="none"`; a finite `0..1e-310` sweep must therefore produce
  `initialization_failed` without an exception or warning.

- [ ] **Step 5: Write failing scarcity/fallback/window tests**

  A flat sweep and a seven-dip sweep return `(None, diagnostics)` with detected
  candidate count and no fabricated successful guess when fallback is false.
  When fallback is true, the same valid frequency interval returns eight evenly
  spaced centers strictly inside the observed bounds and separated by at least
  the configured minimum. Use edge margin
  `max(min_center_separation_hz / 2, median_grid_step)`, width
  `clip(4 * median_grid_step, min_fwhm_hz, max_fwhm_hz)`, and amplitude
  `min(max_amplitude, max(2 * min_resolved_amplitude, max(depth) / 8))`.
  Record `source="fallback"` and `used_fallback=True`. A configured smoothing window
  longer than the sweep returns a structured initialization failure rather than
  allowing SciPy to throw an opaque exception.

  Before constructing fallback centers, require
  `span - 2*edge_margin > 7*min_center_separation_hz`; strict `>` ensures every
  midpoint constraint box has positive width. If infeasible, return
  `source="none"` with a stable diagnostic message. The separate exact fallback
  fixture above—not the ordinary regression family—demonstrates successful
  model fitting.

- [ ] **Step 6: Implement fallback/failure diagnostics and verify Task 2**

  Add regressions for a nonuniform grid, steep linear and curved baselines,
  zero discovery span, an edge dip that `find_peaks` cannot identify, a close
  candidate pair whose height and prominence ranks differ, and the too-few-
  inlier branch. Ensure fallback cannot activate implicitly. Run focused/full tests, Ruff,
  diff checks, deterministic reruns, scientific/code review, state/changelog,
  and commit.

---

### Task 3: Constrained Lorentzian and Pseudo-Voigt Oracle Fitter

**Files:**
- Create: `src/odmr_bench/estimators/parameterization.py`
- Create: `src/odmr_bench/estimators/fitting.py`
- Create: `tests/estimators/test_parameterization.py`
- Create: `tests/estimators/test_fitting.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `CompleteSweep`, `FitConfiguration`, optional validated
  `FitInitialGuess`.
- Produces:
  `fit_spectrum(sweep, configuration, initial_guess=None) -> SpectrumFitResult`.
- Produces public helper:
  `linearized_standard_errors(jacobian, scaled_cost, degrees_of_freedom, public_transform, rank_rtol) -> tuple[np.ndarray | None, int | None, str | None]`.

- [ ] **Step 1: Write failing model-pack/unpack tests**

  Test pure internal `pack_parameters`, `unpack_parameters`, and
  `public_parameter_transform` helpers directly so normalized frequency,
  fluorescence, and baseline scaling round-trip algebraically to public
  Hz/per-Hz/per-Hz² parameters without optimizer tolerance.
  Confirm Lorentzian results fix eta at one and pseudo-Voigt results retain eight
  independently bounded eta values. For adjacent guesses `g_i < g_(i+1)`, let
  `m_i=(g_i+g_(i+1))/2` and separation `d`. Define
  `upper_i=m_i-d/2`, `lower_(i+1)=m_i+d/2`, with `lower_0=f_min` and
  `upper_7=f_max`. Preflight requires `f_max-f_min >= 7*d`, every initial gap
  at least `d`, every guess center inside the sweep and its box, and every
  `lower_i < upper_i`. Test exact boundary values, infeasible span/gaps/boxes,
  and prove any accepted adjacent boxes remain at least `d` apart.

- [ ] **Step 2: Implement scaled parameter packing and residual evaluation**

  Fix the public baseline reference to sweep midpoint `f_ref`, set
  `h=(f_max-f_min)/2` and `z=(f-f_ref)/h`. Define fluorescence scale
  origin `y_ref=median(y)` and variation scale `S=ptp(y)`. Sample-count
  preflight runs first; then, if `S` is zero or non-finite, return
  `uninformative_sweep` before initialization and SciPy;
  do not replace it with an offset-dependent floor. Pack the baseline as
  `[(b0-y_ref)/S, b1*h/S, b2*h**2/S]` (omit quadratic for degree one), then for each
  resonance `[amplitude/S, (center-f_ref)/h, fwhm/h]`, followed by eight etas
  only for pseudo-Voigt. Reject a caller guess whose baseline reference is not
  exactly `f_ref`; exact ID order must equal `configuration.resonance_ids`, eta
  must match model semantics, and every center/width/amplitude/separation bound
  must hold before SciPy. Evaluate the quadratic expressions with an
  exponent-aware product/ratio operation; do not form `h**2` or depend on a
  multiplication association that may overflow when the mathematical result
  is finite and representable.

  Unpack the final public intercept as `y_ref + S*b0_scaled`. For optimizer
  evaluation, form `y_centered=y-y_ref` once and evaluate the existing
  FWHM-native `multi_resonance_spectrum` with centered baseline intercept
  `S*b0_scaled`; use residual `(model_centered-y_centered)/S` so the residual and
  finite-difference Jacobian never reconstruct and subtract two large absolute
  fluorescence origins. Pass deterministic `x_scale=1.0` because every
  packed coordinate is dimensionless. Baseline coordinates use intentional
  `(-inf, +inf)` SciPy bounds because their public coefficients are constrained
  only to be finite; resonance bounds must remain finite, strictly ordered, and
  numerically feasible. The public transform `T` is diagonal in
  this fixed-reference layout: factors are `S`, `S/h`, `S/h**2` for baseline;
  `S`, `h`, `h` for each amplitude/center/FWHM; and one for eta. There is no
  baseline re-referencing in Stage 6.1.

- [ ] **Step 3: Write failing clean-recovery tests**

  Use the fixed fixture table below. Fit deterministic 8-dip Lorentzian and pseudo-Voigt spectra with explicit
  test guesses perturbed from truth. Require success and per-resonance center
  error below 1% of FWHM, FWHM error below 5%, amplitude error below 5%, eta
  absolute error below 0.08 for identifiable pseudo-Voigt fixtures, Q equal to
  fitted center/FWHM, and baseline values within declared fixture tolerances.
  Cover both linear and quadratic baselines. Public-fit comparisons use
  `np.testing.assert_allclose`; pure pack/unpack tests require exact values for
  binary-exact inputs. Identical fit calls require exact success/status/IDs and
  `rtol=1e-12, atol=1e-12` for floating result arrays.
  Add affine-fluorescence regressions: multiply every fluorescence-valued input,
  amplitude, baseline coefficient, noise, and configuration amplitude threshold
  by `1e-3` and `1e3`, and separately add `1e6` to the baseline intercept.
  For multiplicative cases scale observed fluorescence/noise, true and guessed
  baseline coefficients/amplitudes, `max_amplitude`, and
  `min_resolved_amplitude`; do not scale eta, relative prominence, rank
  tolerance, or the SSE-improvement fraction. Divide fitted fluorescence-valued
  results by that factor before comparison. For the additive case add `1e6`
  directly to one realized noisy observation array and to the explicit-guess
  intercept, subtract it only from the fitted intercept for comparison, and
  leave other fields unchanged. Require identical success/failure code, ordered
  IDs, and full-rank decision plus `rtol=1e-6` agreement for raw cost, RMSE,
  every public standard-error field, and derived amplitude significance. A
  deliberately high local threshold must make both origins take the same
  significance-gate branch.
  At one fixed public parameter vector, compare the base and inverse-transformed
  affine packed parameters exactly where binary-representable and their scaled
  residual vectors with `rtol=0, atol=5e-8`; this directly tests the scaling
  algebra. Independently optimized affine cases must have the same public
  success/failure classification, failure code, ordered IDs, and full-rank
  decision, but need not have identical SciPy status/message or `nfev`.
  After inverse transformation, require center differences `< 10 Hz`, FWHM
  differences `< 1 kHz`, eta differences `< 1e-4`, and require both fits to
  satisfy the clean recovery bounds above. The identical-input `1e-12`
  repeatability assertion applies only to byte-identical inputs, not affine
  representations with different floating-point resolution.

- [ ] **Step 4: Implement bounded least squares and post-fit validity**

  Call `scipy.optimize.least_squares` with deterministic `method="trf"`, linear
  loss, configured `max_nfev`, and explicit bounds. A successful result requires
  optimizer success, finite parameters/residuals/cost, centers separated by at
  least `min_center_separation_hz`, all public bounds satisfied, positive
  degrees of freedom, full scaled-Jacobian column rank, every amplitude at least
  `min_resolved_amplitude`, every amplitude meeting the configured
  model-conditioned local `min_amplitude_significance`, finite residual RMSE,
  and configured baseline-only SSE improvement. Populate public raw-fluorescence cost as
  `scaled_optimizer_cost * S**2`; compute RMSE from unscaled residuals. No guess
  means `initialization_failed` without SciPy. Insufficient sample count means
  `insufficient_samples`. A constant sweep means `uninformative_sweep` before
  initialization regardless of `allow_fallback` or a supplied valid user guess;
  because no optimizer attempt occurs, `initial_guess` remains `None`.
  Optimizer termination failure means `optimization_failed`; a nominally
  successful termination with non-finite optimizer outputs is a metric-less
  `quality_failed`, and any other post-fit check is `quality_failed`. Failed
  results retain diagnostics but no baseline/resonance estimate, and NaN is
  never used as a missing-value sentinel.
  Compute the degree-matched baseline-only reference using the exact all-sample
  least-squares definition above on `y-y_ref`, never the large absolute target.

- [ ] **Step 5: Write failing uncertainty tests**

  For a full-column-rank scaled Jacobian and positive degrees of freedom,
  compute one SVD `J=U @ diag(s) @ Vt`. Define retained singular values with
  the same strict gate `s > max(s)*rank_rtol`; only full rank proceeds. Compute
  `sigma=sqrt(scaled_cost)*sqrt(2)/sqrt(dof)` and form the square-root factor
  `F=Vt.T * (sigma/s)` columnwise, then `C_packed=F @ F.T`. This is algebraically
  `Vt.T @ diag((2*scaled_cost/dof)/s**2) @ Vt` without prematurely squaring
  extreme singular values. Then compute
  `C_public=T @ C_packed @ T.T`; compare standard errors to
  `sqrt(diag(C_public))`. Include a diagonal transform with distinct Hz,
  FWHM, amplitude, and baseline factors. Baseline re-reference covariance is
  not tested because Stage 6.1 fixes the public reference to sweep midpoint and
  rejects re-referenced guesses. For rank
  deficiency, non-finite input, non-positive degrees of freedom, invalid shapes,
  or invalid covariance, require `(None, computed_rank_or_None, reason)`. A
  public transform that is not representable is uncertainty unavailability:
  run the same one Jacobian SVD, retain its computed rank, do not fabricate
  standard errors, and permit an otherwise full-rank fit to remain successful.
  In a well-conditioned noisy fit, assert
  finite non-negative public standard-error arrays with shapes matching baseline
  coefficients and eight resonance parameters; label the method local linearized
  Jacobian covariance. The public helper rejects boolean or non-integral degrees
  of freedom, non-real scalar cost/tolerance values, and complex Jacobian or
  transform arrays with structured reasons before any lossy conversion. Extreme
  integral scalar values that cannot enter floating-point covariance arithmetic
  also return structured unavailability rather than escaping as exceptions.

  After the full-rank SVD covariance is available, derive each public amplitude
  standard error directly from its packed covariance diagonal and the
  fluorescence scale. Apply `amplitude / amplitude_se >=
  min_amplitude_significance` before attempting the full public transform, so
  an unrepresentable slope or quadratic transform does not erase otherwise
  available amplitude evidence. Zero amplitude SE with positive amplitude is
  positive infinity; unavailable or non-finite evidence is a conservative
  quality failure. Do not describe this local, model-conditioned ratio as a
  calibrated detection statistic.

- [ ] **Step 6: Write failing initializer/failure/noisy regression tests**

  Preflight each degree/model combination immediately below, at, and above its
  free-parameter count; at/below returns `insufficient_samples` without SciPy.
  Fit a clean generated sweep without a supplied guess to prove integration
  with Task 2. A fixed-seed low-noise fixture must recover within explicitly
  looser bounds from the fixed table. A constant sweep with auto initialization,
  enabled fallback, or a valid user guess must always return
  `uninformative_sweep` with the exact preflight fields above. With fallback
  enabled, a seven-dip input must return `quality_failed`. Noisy seven-line
  fixtures at sigma `2e-4` with fixed seeds 1, 2, and 23 must also return
  `quality_failed`, including after a direct `+1e6` shift and when noise
  initialization inserts a false component between true lines. An identifiable
  eight-dip fixture
  whose deliberately high discovery prominence forces fallback must succeed and
  retain `source="fallback"`. Invalid user guesses raise for ID order, eta,
  baseline reference, interval, separation, or configured bounds and record
  `source="user"` plus an exactly retained defensive initial-guess snapshot on
  valid use. Force optimizer failure through a monkeypatched
  deterministic SciPy result with `success=False`, rather than incidental
  `max_nfev`. Two identical fits use the repeatability tolerance above.

- [ ] **Step 7: Complete fitting, verify, review, and commit**

  Run focused/full pytest, Ruff, diff checks, inspect residual scaling,
  FWHM/Q, bounds, uncertainty labels, failure honesty, and absence of truth
  inputs; update state/changelog and commit.

---

### Task 4: Cold-Start Repeated Full-Sweep Estimator and Researcher Guidance

**Files:**
- Create: `src/odmr_bench/estimators/full_sweep.py`
- Create: `tests/estimators/test_full_sweep.py`
- Create: `docs/estimators.md`
- Create: `examples/fit_synthetic_sweep.py`
- Modify: `src/odmr_bench/estimators/__init__.py`
- Modify: `README.md`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: `RepeatedFullSweepEstimator(configuration)` with `reset()`,
  `update_sweep(sweep) -> SweepEstimate`, `latest`, and evaluator-only immutable
  `history`.
- Every update calls `fit_spectrum(sweep, configuration, initial_guess=None)`.
- `SweepEstimate` records the fit plus the sweep's last observation
  index/timestamp and is unavailable before the first update.

- [ ] **Step 1: Write failing cold-start/causal-wrapper tests**

  Monkeypatch the module-level fitter with a recording callable and submit two
  sweeps. Assert both calls receive `initial_guess=None`, the second call does
  not receive the first result, input order is retained, metadata in each
  `SweepEstimate` comes only from its own completed sweep, and `latest/history`
  expose frozen result references without mutable lists. `reset()` removes all
  estimator state. Add three attempted updates returning success, structured
  failure, success; history must retain all three in order and `latest` must
  advance to the failed attempt rather than silently retaining the prior
  success. Invalid input rejected before fitting must not append history.

- [ ] **Step 2: Implement the repeated-sweep wrapper**

  Store immutable configuration, a private result list, and no fit parameters
  outside returned history. Reject non-`CompleteSweep` inputs before fitting.
  Copy `last_sequence_index`, `last_timestamp_s`,
  `total_integration_time_s`, and `total_nominal_exposure_photons` into every
  estimate, including failures. Do not impose cross-recording monotonicity;
  evaluator orchestration owns that check.
  The class is a complete-sweep estimator, not the sample-wise adaptive
  estimator interface planned for Stage 6.3.

- [ ] **Step 3: Write a generated end-to-end estimator regression**

  Generate two independent complete sweeps with known small center shifts and
  fixed noise seeds. Require two successful cold-start fits, correct last
  sequence/timestamp propagation, ordered eight-resonance estimates, and error
  bounds declared as regression tolerances rather than benchmark results.

- [ ] **Step 4: Add documentation and a download-free example**

  Document the model, initialization, bounds, failure semantics, local
  Jacobian uncertainty, ordered-center limitation, and cold-start behavior.
  Explain that real recording fits are apparent observables, not truth. The
  example generates one eight-dip pseudo-Voigt sweep, fits it, and prints finite
  center/FWHM/Q diagnostics; it neither downloads data nor claims realtime or
  comparative performance.

- [ ] **Step 5: Run the complete Stage 6.1 verification gate**

  Capture `repo=$(pwd -P)`, remove only the repository's existing `dist/`
  artifacts through an explicit checked path, and run estimator tests, full
  pytest, Ruff, `.venv/bin/python -m build`. Set
  `wheel="$repo/dist/nv_odmr_trackbench-0.1.0-py3-none-any.whl"` and assert it
  is the one expected file explicitly so no stale artifact can be selected. Create `smoke=$(mktemp -d)`
  and its environment with `"$repo/.venv/bin/python" -m venv "$smoke/venv"`;
  install exactly `"$wheel"`, change to `$smoke`, and import
  `odmr_bench.estimators` with `"$smoke/venv/bin/python"`. Run the example as
  `"$repo/.venv/bin/python" "$repo/examples/fit_synthetic_sweep.py"`, then run
  `git -C "$repo" diff --check`. Inspect the full Stage 6.1 diff for scientific
  correctness, public API stability, secrets/absolute paths, generated
  artifacts, and unsupported claims. Update `PROJECT_STATE.md` with the exact
  passing test count and next Stage 6.2 action, update `CHANGELOG.md`, and commit.

---

## Final Integrated Review Gate

After every task review is clean, create one review package from the Stage 6.1
design commit through the final implementation commit. A fresh senior reviewer
must audit initialization bias, center-order enforcement, FWHM/Q conventions,
fit failure honesty, uncertainty scaling, truth isolation, cold-start behavior,
real-data claims, API/package quality, and regression strength. One fix agent
addresses the complete Critical/Important finding set, followed by re-review.
Only a clean integrated range receives final local verification and authorized
push to `origin/main`.

Integrated-review acceptance does not promote the Stage 6.1 success bit into a
physical line detector. It means only that the fitted eight-component solution
passed the declared model, initializer, rank, amplitude, local-significance,
and baseline-improvement checks. Unresolved or model-mismatched spectral
features remain outside the validated scope and must be counted as a limitation
in later benchmark reporting.
