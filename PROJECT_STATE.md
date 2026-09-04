# Project State

Last updated: 2026-09-04

## Current stage

Stage 6.2 locally complete — The causal warm-started sweep estimator, generated
drift regression, documentation, package smoke, and integrated re-review are
clean. The reviewed range is ready for synchronization before Stage 6.3 design.

## Completed work

- Inspected the repository and confirmed it began empty, without prior commits
  or benchmark outputs.
- Defined the scientific observables and unambiguous FWHM/Q conventions.
- Separated causal recorded playback from interactive closed-loop emulation.
- Defined acquisition-resource accounting and matched-budget comparison rules.
- Defined estimator truth-isolation and causal-access requirements.
- Scoped the first end-to-end milestone and its required outputs.
- Created the public GitHub repository and configured it as the `origin` remote:
  `https://github.com/stannum13/nv-odmr-trackbench`.
- Renamed the public repository and distribution to `nv-odmr-trackbench` while
  retaining the `odmr_bench` import package and `odmrbench` CLI.
- Received user approval of the Stage 0 scientific specification.
- Added the installable Python package scaffold and `odmrbench` console entry
  point.
- Added the MIT license, citation metadata, and GitHub Actions CI workflow.
- Added normalized Lorentzian, Gaussian, and FWHM-matched pseudo-Voigt line
  shapes with explicit linewidth conversion helpers and Q calculation.
- Added immutable, validated baseline and resonance parameters using explicit
  Hz-valued fields, canonical Python-float storage, and a reference-centered
  polynomial baseline.
- Added deterministic eight-dip spectrum composition with stable parent IDs
  and caller-supplied, already-realized additive noise.
- Added a YAML-driven script that generates the synthetic eight-resonance
  demonstration plot from reusable package configuration and curve helpers.
- Extended CI to build and install the wheel before testing and smoke-test the
  installed `odmrbench --version` entry point.
- Completed independent task reviews and a clean final senior review for the
  repository-scaffold and spectral-model stages.
- Approved the event-driven virtual-instrument design, including normalized
  fluorescence, photon accounting, truth isolation, and virtual-clock semantics.
- Verified Figshare DOI `10.6084/m9.figshare.28788437.v1` as the first optional
  real-data anchor: 4,693 sweeps by 311 points, CC BY 4.0, checksum-matched.
- Added the task-by-task implementation plan for verified playback, hidden
  dynamics, observation noise/resources, the virtual instrument, and POC CLI.
- Added a checked, explicit-path Figshare registry/loader that preserves the
  raw analog signal and unresolved units, without downloading or redistributing
  external data.
- Added immutable sweep data and causal row-major recorded playback, with
  timestamps unavailable unless a caller explicitly assumes a nominal clock.
- Replaced the deficient generator-only estimator boundary: Python generator
  frames retain the offline dataset and can reveal future samples. Estimator
  playback now uses an evaluator-owned causal callback runner that supplies one
  frozen observation at a time; `iter_playback_for_analysis` is trusted
  evaluator-only tooling.
- Hardened local verified loading by checking and parsing one immutable byte
  snapshot, added full YAML-to-record parity coverage, deterministic verified
  loader success coverage, and non-empty sweep-dimension validation.
- Added frozen hidden `SpectralSnapshot` truth records that require exactly
  eight unique stable physical IDs and positive absolute centers.
- Added runtime-checkable spectral-dynamics protocol plus stationary and
  deterministic common/per-ID linear-center-drift implementations. They accept
  only finite non-negative virtual timestamps and never sort resonance order.
- Added seeded Poisson shot-noise and controlled Gaussian normalized-fluorescence
  models, with photons retained only when the generative model produces counts.
- Added provenance-bearing empirical residual noise with explicit replay,
  independent-sample, and contiguous-block correlation modes.
- Added frozen full and estimator-safe observation records; signal-conditioned
  expected photons remain evaluator-only and are absent from estimator objects.
- Added atomic virtual-acquisition resource accounting for observations,
  integration, nominal/expected/realized photons, uncounted observations, and
  elapsed virtual time.
- Added an event-driven virtual ODMR instrument that evaluates hidden truth at
  the integration midpoint, returns end-of-integration timestamps, advances
  only virtual time, and commits sequence, clock, and resource totals atomically.
- Preserved seeded stochastic reproducibility across failed queries by restoring
  NumPy-generator and stateful-noise state; empirical replay remains usable
  without copying its immutable configuration.
- Added deterministic JSON CLI summaries for checked optional-dataset metadata,
  explicit-local raw playback, and a fixed seeded synthetic-drift scenario.
- Hardened CLI configuration and packaging review findings: the complete query
  schedule is scalar-canonicalized and checked for finite virtual timing before
  instrument construction, expected input failures have concise exit-2 error
  messages, playback streams aggregates, and `bundled:drift` is wheel-packaged
  for arbitrary-working-directory use.
- Added an explicit eight-resonance Poisson drift configuration, a download-free
  in-memory playback/emulation example, and researcher-facing raw-data and
  synthetic-emulation guidance.
- Approved a six-stage estimator sequence and specified Stage 6.1: constrained
  offline Lorentzian/pseudo-Voigt oracle plus an independent repeated-sweep
  baseline.
- Added the task-by-task Stage 6.1 implementation plan covering immutable fit
  contracts, deterministic initialization, constrained fitting/uncertainty,
  and the cold-start repeated full-sweep estimator.
- Corrected the plan after adversarial review by defining the fit-failure state
  machine, full-rank quality gate, scaled-to-public covariance transform, exact
  center bounds, public schemas, initializer formulas, and fixed regressions.
- Corrected the second review findings with fluorescence origin/scale
  parameterization, one-cutoff SVD covariance, an exact baseline-only SSE
  reference, feasible fallback geometry/fixture, a denser scan grid, immutable
  initial-guess provenance, and a typed failure-field matrix. The final review
  closes the remaining numerical gap by evaluating both spectral residuals and
  the baseline-only target in centered fluorescence coordinates.
- Defined pre-initialization `uninformative_sweep` handling for zero-variation
  data and corrected the final smoke-test paths and affine-scaling fixture rules.
- Separated exact identical-input repeatability from physically negligible
  affine-representation roundoff and SciPy termination details.
- Added immutable validated full-sweep, fitting-configuration, initialization,
  uncertainty, fit-result, and per-sweep-estimate contracts for the offline
  oracle. Result records enforce the structured failure state machine, preserve
  attempted-guess provenance, and derive read-only signed Q values from public
  fitted centers and FWHM values without estimator access to hidden truth.
- Added deterministic baseline-aware eight-line initialization using three
  robust rejection updates and a final scaled polynomial trend, prominence-
  ranked dip discovery, actual-frequency separation and width interpolation,
  raw detrended amplitudes, and explicit-only feasible fallback geometry.
  Structured diagnostics preserve candidate scarcity and invalid-window or
  numerical baseline failures without fabricating a detected solution. A
  signal-scaled floating-point discovery floor rejects polynomial/smoothing
  roundoff, while overflow-safe frequency normalization supports extreme finite
  same-sign and opposite-sign endpoints.
- Added deterministic bounded Lorentzian and pseudo-Voigt oracle fitting with
  dimensionless frequency/fluorescence scaling, midpoint-referenced linear or
  quadratic baselines, exact non-crossing center boxes, structured scientific
  failures, raw-unit residual diagnostics, full-rank quality gates, and
  public-unit local-linearized covariance from one shared SVD cutoff. Fixed
  regressions cover clean/noisy recovery, affine fluorescence changes,
  initialization/fallback behavior, and exact baseline-improvement boundaries.
- Hardened the oracle review boundary with nonempty initializer-preflight
  reasons, intentional infinite baseline bounds plus finite feasible resonance
  bounds, explicit public-parameter and rounded center-box checks,
  exponent-aware quadratic scaling, and overflow-safe fluorescence origins. A
  distinct metric-less `quality_failed` state covers nominally successful
  optimizers with any non-finite required output, while an unrepresentable
  public covariance transform preserves rank and can leave an otherwise valid
  fit successful without uncertainty. Covariance tests pin the full public
  layout, strict SVD cutoff, and single-SVD implementation.
- Added `RepeatedFullSweepEstimator`, which passes `initial_guess=None` for
  every completed sweep, retains successful and structured failed attempts in
  immutable evaluator history, advances `latest` on failures, copies only that
  sweep's public completion metadata, and clears all retained state on reset.
- Added a fixed-seed two-sweep generated regression with declared numerical
  tolerances, researcher guidance for the model, initialization, bounds,
  failures, local-linearized uncertainty, ordered-center scope, and recording
  interpretation, plus a download-free synthetic fitting example.
- Addressed the integrated Stage 6.1 findings with a positive finite
  `min_amplitude_significance` configuration (default `5.0`) and an all-line
  model-conditioned local amplitude/standard-error gate derived from the same
  packed covariance as public uncertainty. Fixed noisy seven-line seeds 1, 2,
  and 23 now fail quality before and after direct `+1e6` fluorescence shifts
  instead of promoting a noise-supported eighth component; normal clean/noisy
  eight-component fixtures retain their declared recovery.
- Guarded unrepresentable initializer baseline conversion, strengthened result
  provenance across baseline degree/reference, IDs, diagnostic source, and
  optimizer-attempt state/status/message/evaluation count, restricted the
  uncertainty method to its exact local-Jacobian label, and rejected non-finite
  Q without leaking numerical warnings. The public linearized-error helper now
  rejects complex arrays, bool/non-integral degrees of freedom, and invalid
  scalar inputs without lossy coercion. Its covariance uses square-root SVD
  factors to avoid premature overflow/underflow, and finite scaled baseline
  coefficients that would overflow or underflow in public units fail
  initialization explicitly.
- Added fit-level regressions for zero baseline-only SSE, rank deficiency,
  affine raw diagnostics and public errors, nonuniform grids, covariance
  unavailability, exact-zero amplitude errors, and weak false components near
  the configured significance threshold. A noisy direct-addition regression
  pins classification, IDs, rank, cost, RMSE, every public SE field, and local
  significance behavior under a `+1e6` fluorescence origin shift.
- Specified the Stage 6.2 causal warm-start state machine: successful-prior-only
  seeding, guarded polynomial rebasing, shared sweep/guess preflight, explicit
  warm/cold attempt provenance, cold recovery, stale active estimates with
  distinct age bases, nonoverlapping endpoints, and acquisition-versus-compute
  resource accounting.
- Drafted the complete four-gate Stage 6.2 TDD implementation plan: frozen
  attempt/estimate contracts, shared fitter preparation and exact finite-float
  baseline rebasing, the causal warm-start/recovery/age/CPU state machine, and
  frozen-snapshot drift integration with documentation, example, build, and
  isolated-wheel smoke. After its initial 0-Critical/7-Important/4-Minor review,
  corrected the lost post-preflight rank variable, attempt/source and
  disposition/active invariants, typed compatibility translation/precedence,
  global timer ordering/atomicity, behavioral TDD increments, exact drift
  configuration, wrapper-only constant failure, branch joins, and fail-fast
  wheel/sdist smoke. The revised plan subsequently passed adversarial
  re-review with no findings and preceded feature implementation.
- Added frozen, slotted `SweepFitAttempt` and `WarmSweepEstimate` records with
  closed public literals, immutable attempt tuples, exact warm/cold/preflight
  provenance, disposition and rejection-code matrices, identity-based active
  result selection, explicit stale-age semantics, endpoint/resource domains,
  compound CPU accounting, and derived current-fit, staleness, and evaluation
  totals. Constructors canonicalize supported NumPy scalars without using
  array-valued fit-result equality.
- Extracted the Stage 6.1 sample/variation/origin preflight and initial-guess
  validation into one package-internal preparation path shared by the unchanged
  `fit_spectrum` entry point and future warm-start orchestration. Added exact
  finite-float linear/quadratic baseline rebasing with zero-coefficient and
  exact-cancellation handling, explicit overflow/nonzero-underflow rejection,
  and a guarded successful-prior conversion that deterministically returns one
  of five closed compatibility codes in the specified first-failure order.
- Added `WarmStartedFullSweepEstimator`, which validates causal endpoints before
  preparation, seeds only from the latest selected success, rejects over-age or
  incompatible starts explicitly, retains one eligible same-sweep cold retry,
  exposes separately aged stale active fits, counts acquisition resources once,
  and commits history/endpoints only after globally monotonic process-CPU timing
  and every public record construction succeed.
- Validated the completed-sweep wrapper on one immutable, fixed-seed three-grid
  drift family. Cold and warm estimators receive the identical frozen sweep
  objects, changed-midpoint baseline rebasing remains compatible, ordered IDs
  and fixed-fixture center/FWHM/Q bounds hold, and cumulative observations and
  zero-age source promotion are exact. Separate regressions cover constant-
  sweep preflight staleness, update-age rejection, center-outside-sweep cold
  fallback, and deterministic failed-warm/one-cold recovery without duplicating
  acquisition resources. The download-free example reports source, attempt,
  age, `nfev`, and measured process CPU diagnostics without a speedup claim.

## Important scientific and design decisions

- Project identity is `nv-odmr-trackbench`; the existing checkout
  directory is retained.
- Internal frequency and linewidth units are Hz. Internal time units are
  seconds. Public APIs must not accept ambiguous unitless physical quantities.
- Each line component uses FWHM directly. Lorentzian HWHM/gamma and Gaussian
  sigma conversions must be explicit and unit-tested.
- Q is defined as resonance center divided by FWHM and is not treated as a
  proxy for magnetometric sensitivity by itself.
- Low-level spectral functions permit finite signed frequency coordinates for
  generality; physical scenarios will require positive absolute resonance
  centers at the instrument-validation boundary without redefining Q.
- The initial benchmark represents eight electronic resonances. Optional
  hyperfine components may later retain parent electronic-resonance identities.
- An offline-oracle success means an eight-component fit passed the declared
  model, candidate-conditioned initializer, and configured quality thresholds.
  It is not calibrated evidence that eight physical resonances are present.
- Scenario truth belongs to the virtual instrument and evaluation harness; an
  estimator receives only observations and permitted public metadata.
- Recorded playback cannot evaluate adaptive frequencies that were not present
  in the recording.
- A callback runner protects the normal estimator API from accidental future
  data access, but it is not a security sandbox against adversarial Python stack
  introspection; use process isolation for that threat model.
- Budget matching and any unavoidable budget mismatch must be explicit in
  machine-readable results and figures.
- Development prioritizes the first falsifiable vertical slice over completing
  every planned abstraction in advance.
- Empirical residual correlation is a declared experimental condition: replay
  preserves supplied order, sample draws independent residuals, and block draws
  seeded contiguous blocks with deterministic wrapping.

## Tests currently passing

- `pytest tests/estimators`: 597 passed.
- `pytest`: 797 passed.
- `ruff check .`: All checks passed.
- The fail-fast package smoke built exactly one
  `nv_odmr_trackbench-0.1.0.tar.gz` and one
  `nv_odmr_trackbench-0.1.0-py3-none-any.whl`, then installed the wheel into a
  fresh environment, imported all Stage 6.2 public aliases and
  `WarmStartedFullSweepEstimator`, and ran the three-update example there.

## Known scientific limitations

- The current dynamics layer provides only stationary and deterministic linear
  center drift; no Hamiltonian model exists yet.
- No external real-data file is bundled or attached, and the recording has no
  verified tracking truth.
- Hyperfine structure, ensemble inhomogeneity, optical power broadening,
  microwave power broadening, temperature coupling, and instrument transfer
  functions are not yet modeled.
- No benchmark results exist, so neither primary nor secondary hypothesis has
  supporting evidence.
- Raw playback retains unresolved analog units and has no measured timestamps;
  its CLI summary labels both its `unknown_analog_signal` quantity and its
  `conflicted_unverified` unit status; it is not a photon-count or timing claim.
- The fixed CLI emulator is synthetic. Its seeded output does not establish
  estimator accuracy, realtime performance, or agreement with the recording.
- The Stage 6.1 model does not resolve arbitrary overlapping, hyperfine-rich,
  asymmetric, or otherwise model-mismatched features. Its local amplitude
  significance is not a calibrated detector or false-discovery guarantee.

## Known software limitations

- End-to-end benchmark reproducibility has not yet been demonstrated beyond the
  installable package, command smoke tests, and deterministic synthetic
  configuration fixtures.
- The proof-of-concept emulator CLI accepts only its explicit Poisson-noise
  schema and fixed query schedule; adaptive estimator orchestration is not yet
  implemented.
- Warm-started fitting operates only after a sweep completes. Its measured
  update-core process CPU interval and optimizer evaluation count are
  machine-dependent descriptive diagnostics and establish neither within-sweep
  realtime utility nor universal computational improvement.

## Next actions

1. Synchronize the verified Stage 6.2 commit range and confirm remote CI.
2. Design Stage 6.3 calibrated two-point center tracking without hidden-truth or
   arbitrary-gain shortcuts.
