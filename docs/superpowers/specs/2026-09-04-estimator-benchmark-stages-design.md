# Estimator Benchmark Stages Design

## Purpose

Build the first falsifiable estimator comparison as six independently reviewed
vertical slices rather than one large implementation. Each slice must leave the
repository runnable and scientifically interpretable. The sequence begins with
a slow offline reference because center and linewidth trackers cannot be judged
credibly without a tested full-spectrum estimator.

## Approaches considered

Three implementation orders were considered:

1. An estimator-first staged sequence adds the oracle, causal sweep baselines,
   sparse trackers, matched-budget harness, then reporting. This is selected
   because every later result depends on the preceding scientific reference.
2. A tracker-first demonstration would show motion sooner, but its error and
   linewidth claims would lack a defensible reference and stable failure labels.
3. A benchmark-harness-first sequence would establish infrastructure early but
   delay testing the central estimation hypothesis and encourage abstractions
   before their data contracts are known.

## Six-stage decomposition

### Stage 6.1 — Offline oracle and repeated full-sweep fit

Implement deterministic constrained eight-component Lorentzian and
pseudo-Voigt fits with explicit FWHM, polynomial baseline, initialization,
diagnostics, and synthetic truth regression. Wrap the fitter as a repeated
full-sweep estimator that fits each completed sweep independently. Passing the
fit contract is conditional on this model, initializer, and quality thresholds;
it does not establish that eight physical resonances exist.

### Stage 6.2 — Warm-started causal sweep fitting

Carry the preceding successful fit into the next sweep while preserving causal
ordering, stable resonance identities, fit-failure state, estimate age, and
resource use. Compare cold and warm starts on deterministic center drift before
making performance claims.

### Stage 6.3 — Calibrated two-point center tracking

Add an interactive interrogation policy around each current center using a
declared offset and local discriminator calibration. The update converts the
normalized signed fluorescence error into a frequency innovation; it never uses
an arbitrary uncalibrated intensity gain or hidden truth.

### Stage 6.4 — Five-point sparse linewidth and Q tracking

Add a lower-rate local acquisition around the fast center prior. Fit center
correction, FWHM, amplitude, and local baseline from five points with explicit
identifiability/failure checks. Center and linewidth update rates remain
separate, and Q is always recomputed as center divided by FWHM.

### Stage 6.5 — Matched-budget benchmark and metrics

Run repeated-sweep, warm-start, two-point, and two-point-plus-five-point methods
on the same hidden scenario. The primary normalization is equal total
integration time; equal nominal photon exposure and observation count are also
reported. Implement center/FWHM/Q RMSE, MAE, bias, percentile error, estimate
age, latency, samples, photons, virtual time, and estimator CPU time.

### Stage 6.6 — Reproducible result artifacts and CLI

Add deterministic YAML-driven benchmark execution, machine-readable run
records, uncertainty across seeds, and programmatic plots for truth-versus-
estimate and error-versus-budget. The README may summarize measured outputs
only after committed run artifacts reproduce them.

Stages 6.2 through 6.6 receive separate design/implementation review before
execution. This document fixes their dependency order and scope but does not
prejudge their numerical results.

## Stage 6.1 architecture

```text
frequency and fluorescence arrays
              ↓
validated CompleteSweep
              ↓
baseline-aware dip initialization
              ↓
bounded least-squares parameterization
              ↓
immutable SpectrumFitResult
              ↓
RepeatedFullSweepEstimator estimate
```

Scientifically important fitting code lives in
`src/odmr_bench/estimators/`. Input validation and fit results are independent
of the dataset adapter so the same fitter accepts generated spectra and locally
loaded complete real sweeps.

### Complete sweep input

`CompleteSweep` contains one-dimensional finite arrays of strictly increasing
frequency in Hz and fluorescence in declared units, plus optional acquisition
metadata. It requires enough distinct samples to identify the configured model.
The fitter never sorts or silently removes samples. A caller must explicitly
prepare non-monotonic or missing data.

The fitting boundary does not accept truth, future sweeps, expected photons, or
an instrument dynamics object. Synthetic evaluation retains truth separately.

### Fit model and parameterization

Both supported modes reuse the repository's explicit-FWHM line shapes:

- `lorentzian`: eight dips with `eta = 1` fixed;
- `pseudo_voigt`: eight dips with each `eta` bounded to `[0, 1]`.

The baseline is linear by default and optionally quadratic, centered on an
explicit reference frequency. Public parameters remain center Hz, FWHM Hz,
amplitude in spectrum units, eta, and Q. Optimization may use transformed or
scaled internal variables, but conversion to public units is explicit and
tested. Quadratic packing, unpacking, and public covariance factors use
exponent-aware product/ratio evaluation so intermediate overflow cannot reject
a mathematically finite representable result.

Constraints require:

- exactly eight unique model-component output identities;
- strictly ordered centers within the supplied fitting interval;
- positive FWHM with configurable lower and upper bounds;
- non-negative amplitudes with a finite configurable upper bound;
- eta in `[0, 1]` for pseudo-Voigt and exactly one for Lorentzian; and
- finite baseline coefficients.

Center ordering is valid for the initial resolved, non-crossing oracle scope.
It must not later be used to conceal identity swaps in collision scenarios or
be interpreted as evidence of physical identity by itself.

### Initialization

Initialization is deterministic and data-derived:

1. estimate a low-order baseline or smooth trend;
2. smooth only for candidate discovery using a validated Savitzky–Golay window;
3. detect dips with `scipy.signal.find_peaks` on the inverted detrended signal;
4. select eight candidates using prominence and separation constraints;
5. initialize amplitudes from local depth and FWHM from half-prominence
   crossings;
   and
6. fall back to eight evenly spaced centers only when explicitly enabled and
   record that fallback in diagnostics.

The initializer never receives synthetic centers. User-provided initial values
are allowed only through a separate validated `FitInitialGuess` object and are
recorded in fit metadata. Warm-start behavior belongs to Stage 6.2.

The optimizer's center boxes are local and candidate-conditioned. Given
adjacent ordered initial centers `g_i < g_(i+1)`, their midpoint is
`m_i = (g_i + g_(i+1)) / 2`; for configured separation `d`, the left center's
upper bound is `m_i - d / 2` and the right center's lower bound is
`m_i + d / 2`. The outer bounds are the sweep endpoints. This construction
preserves separation but can trap a component around a noise peak or unresolved
feature selected by initialization. Empty or unrepresentable boundary boxes
fail preflight rather than being repaired silently. Scaled-polynomial baseline
conversion likewise fails initialization when finite coefficients cannot be
represented in public per-Hz units.
Nonzero scaled slope or quadratic coefficients that would underflow to public
zero are unrepresentable and follow the same structured failure path.

### Optimization and result

Use `scipy.optimize.least_squares` with bounded parameters and residual scaling
appropriate to the declared fluorescence units. Scaling separates a fixed
data-derived origin `y_ref = median(y)` from the signal-variation scale
`S = ptp(y)`: the packed intercept is `(b0-y_ref)/S`, spectral amplitudes and
baseline variation use `S`, and a zero/non-finite variation is rejected before
initialization and optimization as a structured `uninformative_sweep` failure.
That preflight does not retain an unused auto, fallback, or user guess.
Consequently, in exact arithmetic adding a constant fluorescence offset does
not change the spectral residual Jacobian. Finite-precision offset
representations may change SciPy's discrete termination status or evaluation
count without changing the public scientific success/rank decision. A finite
packed initial point is required before SciPy. Baseline coordinates use
intentional infinite optimizer bounds because their public coefficients are
otherwise unconstrained finite values; all configured resonance bounds must be
finite, strictly ordered, and numerically feasible. A
`SpectrumFitResult` contains:

- model kind and baseline degree;
- eight immutable resonance estimates with stable output IDs;
- baseline estimate and explicit reference frequency;
- Q values derived from fitted center/FWHM;
- success flag, termination status/message, evaluation count, and cost;
- residual RMSE and degrees of freedom;
- initialization diagnostics, including fallback use; and
- an immutable snapshot of the exact initial guess used for any optimizer
  attempt; and
- optional standard errors derived from the final Jacobian.

The result constructor enforces cross-field provenance: a degree-one result has
zero quadratic terms in every retained initial or final baseline; successful
fitted IDs and order equal the retained initial IDs and order; initial and final
baseline references match exactly; and diagnostic sources agree with whether
optimization was attempted. Successful Q derivation runs under guarded
floating-point handling and rejects non-finite or unrepresentable ratios.

Jacobian uncertainties are reported only when the Jacobian has full numerical
column rank under the configured deterministic relative singular-value
tolerance, degrees of freedom are positive, and the covariance approximation
is finite. Rank and covariance reuse one SVD and exactly the same singular-value
cutoff rather than forming a separately truncated normal-matrix pseudoinverse.
Full numerical rank and positive degrees of freedom are Stage 6.1
fit-success requirements, not merely uncertainty requirements. This prevents a
flat or under-resolved fallback fit from being reported as an identified
eight-component solution in the validated fixtures.

Optimization uses dimensionless scaled parameters and residuals, but public
uncertainty never remains in optimizer coordinates. With the public baseline
reference fixed to the sweep midpoint, the fitter computes the complete linear
map from packed parameters to public intercept, per-Hz slope, per-Hz²
quadratic, amplitude, center Hz, FWHM Hz, and eta. It transforms the full
covariance as `C_public = T @ C_packed @ T.T` before extracting standard errors.
The packed covariance is represented by square-root SVD factors,
`sigma = sqrt(cost)*sqrt(2)/sqrt(dof)` and `F = V * (sigma/s)`, so that
algebraically `C_packed = F @ F.T`. Packed errors are stable row norms of `F`,
and public errors are stable row norms of `T @ F`; the dense covariance need
not be formed, so extreme but representable errors are not squared prematurely.
The reported fit cost is converted to raw squared-fluorescence units and
residual RMSE to fluorescence units. Uncertainties are labeled local linearized fit uncertainties, not
experimental coverage guarantees. If covariance is non-finite after the public
transform, uncertainty is `None` with a diagnostic reason and the otherwise
identified fit may remain successful. The same rule applies when the public
transform itself is not representable: rank is still computed from the shared
single SVD, no public standard errors are fabricated, and the transform failure
is reported precisely.

The same packed covariance supplies truth-independent evidence for every fitted
component. Before attempting the unrelated full public transform, the fitter
multiplies each packed amplitude standard error by the fluorescence scale and
requires `amplitude / amplitude_se >= min_amplitude_significance`, whose
positive finite default is `3.0`. A positive amplitude with exactly zero
standard error has positive-infinite significance. Unavailable or non-finite
evidence fails conservatively. This ratio is a model-conditioned local
diagnostic, not a calibrated detection statistic or false-discovery guarantee.
The public standard-error helper rejects boolean or non-integral degrees of
freedom, invalid scalar cost/tolerance values, and complex Jacobian or transform
arrays with structured reasons instead of silently coercing them. Integral
scalar values too large for floating-point covariance arithmetic also produce
structured unavailability rather than an exception.

An optimizer termination flag alone is insufficient for scientific success.
The result is unsuccessful when parameters are non-finite, bounds are violated,
center ordering/separation fails, residuals are non-finite, degrees of freedom
are non-positive, the Jacobian is rank deficient, any line is below the
configured resolved-amplitude or model-conditioned local-significance
threshold, or the fit fails the configured minimum improvement over a
baseline-only model. Failed results retain
diagnostics but expose no plausible-looking parameter estimate as a successful
observation.
An unsuccessful optimizer termination is `optimization_failed`. A nominally
successful termination with non-finite parameters, residuals, or cost is
instead `quality_failed`; its raw cost and RMSE remain `None` rather than using
NaN or manufacturing finite metrics. This special non-finite-output reason is
invalid when finite raw metrics are present. Other quality failures retain
finite raw cost and RMSE diagnostics.

Initialization follows one explicit state machine. No guess means
`initialization_failed` and SciPy is not called. An explicitly enabled fallback
guess may reach optimization and records `used_fallback`; low candidate count
then remains diagnostic rather than forcing failure. The same post-fit rank,
amplitude, local-significance, separation, and baseline-improvement checks
apply, so fallback can recover a well-conditioned eight-component fixture
missed by conservative detection. These checks reject the declared flat and
fixed-seed seven-line regressions but do not prove arbitrary model-mismatched
spectra contain eight physical lines.
Evenly spaced fallback is constructed only when the usable span after edge
margins is strictly greater than seven minimum separations, ensuring every
ordered-center constraint box has positive width; otherwise initialization
fails explicitly.

### Repeated full-sweep estimator

`RepeatedFullSweepEstimator` accepts complete sweeps one at a time. It fits each
sweep independently using the same immutable configuration and emits an
estimate valid at that sweep's last observation timestamp/index. It retains
completed result history only for evaluator/reporting use; its next fit receives
no previous parameters. This is intentionally slow and serves as the cold-start
scanning baseline.

## Real-data role

The verified Figshare recording is useful for parsing-to-fit integration,
baseline/lineshape stress, convergence diagnostics, and apparent fitted
parameters. It does not supply exact eight-resonance identities, field/current
trajectory, timestamps, photon calibration, or center/FWHM/Q truth. Therefore:

- real-file fits are labeled apparent observables or offline references;
- fit residuals are not automatically pure detector noise;
- no truth RMSE is computed from the recording;
- no adaptive or saved-acquisition claim is made from subsampling it; and
- CI uses generated spectra and tiny parser fixtures, never the external file.

An optional local-data smoke command may be documented after the synthetic
oracle is stable, but Stage 6.1 does not make the external dataset mandatory.

## Errors and diagnostics

Malformed configuration and invalid arrays raise concise validation errors
before optimization. Candidate scarcity, rank deficiency, optimizer failure,
or failed post-fit checks return structured unsuccessful results when enough
valid input exists to attempt a fit. This distinction lets benchmark metrics
count fit failures rather than silently dropping them.

No fit path catches arbitrary programming exceptions and relabels them as a
scientific failure.

## Verification contract

Stage 6.1 tests must establish:

- Lorentzian and pseudo-Voigt recovery on deterministic eight-dip spectra;
- center, FWHM, amplitude, eta, baseline, and Q definitions and tolerances;
- linear and quadratic baseline behavior;
- deterministic initialization and repeated fits;
- ordered centers and positive bounded widths/amplitudes;
- no use of hidden truth or future sweeps;
- honest failure for fewer than eight detectable dips, including the fixed
  noisy seven-line seeds whose initializer inserts a false component;
- fallback diagnostics when fallback is explicitly enabled;
- uncertainty availability for a well-conditioned fit and unavailability for a
  rank-deficient case;
- conservative model-conditioned local amplitude-significance gating using the
  same packed covariance even when another public transform is unavailable;
- independent repeated-sweep fits do not reuse earlier fit parameters; and
- generated noisy regression cases remain within declared accuracy bounds for
  fixed seeds without turning those fixtures into headline benchmark results.

Every implementation task follows red/green TDD, independent task review, full
pytest/Ruff verification, diff inspection, scientific/software review, state
and changelog updates, one local commit, and authorized synchronization to
`origin/main` only after its review is clean.

## Stage 6.1 completion boundary

Stage 6.1 is complete when another researcher can construct a complete synthetic
sweep, run either supported constrained fit, inspect diagnostics and fitted
FWHM/Q, and process successive sweeps through the cold-start estimator with no
truth leakage. It does not yet claim realtime performance, matched-budget
superiority, or robustness under collisions, dropout, Hamiltonian motion,
unresolved structure, or other model mismatch. In particular, Stage 6.1
success is not proof of eight-line physical truth.
