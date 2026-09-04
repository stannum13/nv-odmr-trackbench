# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow [Semantic Versioning](https://semver.org/)
once a package version is introduced.

## [Unreleased]

### Changed

- Closed the integrated Stage 6.1 oracle findings. Fit success now requires
  every component to meet a configurable model-conditioned local amplitude
  significance threshold (`5.0` by default) derived from the same packed
  covariance as public uncertainty; unavailable/non-finite evidence fails
  conservatively, while unrelated public-transform unavailability does not
  erase usable amplitude evidence. Noisy seven-line seeds 1, 2, and 23 fail
  before and after a direct `+1e6` fluorescence shift rather than promoting a
  noise-supported eighth component. Centered model/data residuals and centered
  baseline-only least squares eliminate avoidable additive-origin cancellation
  while preserving public cost/RMSE units. Initializer baseline conversion now
  rejects both overflow and nonzero-to-zero underflow. Result cross-field and
  optimizer-status provenance, the exact local-Jacobian uncertainty method,
  finite-Q derivation, and public linearized-error input validation are guarded
  at their numerical/type boundaries. The covariance is formed from square-root
  SVD factors so representable errors are not lost to premature singular-value
  squaring. Documentation now states explicitly that an oracle success is
  model/initializer/threshold-conditioned, not proof of eight physical lines.
- Hardened constrained oracle fitting against empty initialization reasons,
  rounded public-bound violations, unrepresentable center separations,
  collapsed resonance bounds, and extreme finite fluorescence origins.
  Quadratic scaling now uses exponent-aware product/ratio evaluation so any
  representable result is independent of multiplication association. Baseline
  optimizer coordinates remain intentionally unbounded while every resonance
  bound is finite and numerically feasible. Successful optimizer termination
  with any non-finite required output is now an explicit metric-less
  `quality_failed`; that reason cannot accompany finite residual metrics.
  A non-representable public covariance transform leaves uncertainty
  unavailable without discarding an otherwise identified fit.
- Built-in Poisson and Gaussian noise strategies now fix their recorded
  sampling-rule provenance at construction, and public Figshare attribution
  consistently uses the checked creator string, `Liu`.
- Clarified that the optional external recording is verified by metadata and
  checksum but is neither bundled nor attached and has no verified tracking
  truth; the static-spectrum script is a source-checkout demo after install.
- CLI simulation now canonicalizes every fixed-schedule frequency and duration
  before virtual-instrument construction, validates the entire prospective
  virtual timeline, and reports expected path/YAML/configuration/data failures
  as concise stderr errors with exit status 2. Playback aggregates stream
  without retaining every observation object.
- Replaced generator-only recorded playback at the estimator boundary with an
  evaluator-owned callback runner. Python generator frames retain offline source
  state, so the retained `iter_playback_for_analysis` iterator is explicitly
  trusted evaluator tooling and is never represented as causally isolated.
- Verified local Figshare loading now checks the length and digest of one
  immutable byte snapshot and parses that same snapshot.
- Rejected empty sweep and frequency dimensions in `SweepDataset`.
- Renamed the public distribution and repository to `nv-odmr-trackbench`;
  the `odmr_bench` import package and `odmrbench` CLI remain unchanged.
- Canonicalized validated real parameter scalars to immutable Python floats and
  rejected boolean, complex, array, non-finite, and non-string-ID inputs.
- Made the deterministic additive-noise boundary explicit: callers supply an
  already-realized broadcastable perturbation and sampling remains external.
- Extended CI to build and install the wheel and smoke-test the installed CLI.

### Added

- Added fixed-snapshot drift integration coverage and a download-free
  warm-started completed-sweep example. The regressions freeze truth at sweep
  construction, feed identical immutable sweeps to cold and warm paths, cover
  changed-grid rebasing, stale/age and cold-recovery provenance, preserve
  acquisition accounting, and treat CPU/`nfev` as descriptive diagnostics
  rather than realtime or universal-speedup evidence. Researcher guidance now
  documents warm-source, rebase, retry, staleness, age-base, resource, reset,
  atomicity, ordered-center, and interpretation boundaries.
- Added a causal warm-started completed-sweep estimator with successful-prior-
  only seeding, strict endpoint modes, explicit age/compatibility rejection,
  one eligible same-acquisition cold recovery, independently derived stale-fit
  ages, and globally monotonic process-CPU timing with atomic state commits.
- Added one internal fit-preparation contract shared by cold and future warm
  paths: exact Stage 6.1 sample/variation/origin preflight results, typed
  initial-guess compatibility validation, overflow-safe linear/quadratic
  baseline rebasing, and guarded successful-prior conversion with five closed,
  deterministic first-failure compatibility codes. The public `fit_spectrum`
  API and its sole-optimizer-entry-point behavior remain unchanged.
- Added frozen public warm-sweep attempt and estimate contracts with closed
  provenance/disposition states, explicit active-result age, immutable compound
  attempts, and validated acquisition/CPU resource accounting.
- Drafted the four-task Stage 6.2 test-driven implementation plan covering
  exhaustive warm-attempt/active-age contracts, shared start-independent fitter
  preparation, overflow-safe polynomial rebasing, causal recovery and CPU
  timing, frozen-snapshot drift regressions, researcher guidance, and isolated
  wheel verification. Revised it after adversarial review to preserve the
  post-preflight rank local, close diagnostic/disposition/active-state
  invariants, type compatibility failures, globally validate timer ordering and
  final-clock atomicity, freeze the scientific configuration, use behavioral
  RED/GREEN increments, cover every source-selection join, and make exact
  sdist/wheel/example smoke fail fast. The corrected plan passed adversarial
  re-review with no findings before implementation began.
- Specified Stage 6.2 warm-started causal completed-sweep fitting: successful-
  prior-only seeding, polynomial-baseline rebasing, explicit compatibility and
  age rejection, retained warm/cold attempts, conditional same-sweep cold
  recovery, stale active-estimate age, and acquisition-versus-compute resource
  separation.

- A cold-start `RepeatedFullSweepEstimator` that independently fits each
  completed sweep with `initial_guess=None`, retains successful and failed
  attempts in immutable evaluator history, advances `latest` on every attempt,
  and preserves only the submitted sweep's public completion metadata. Fixed-
  seed two-sweep regression coverage pins cold-start provenance, resonance
  ordering, metadata propagation, and declared center/FWHM tolerances.
- Researcher guidance for the full-sweep model, initialization, constraints,
  structured failures, local Jacobian uncertainty, ordered-center limitation,
  and interpretation of external-recording fits, plus a download-free
  pseudo-Voigt fitting example that prints finite center/FWHM/Q diagnostics.
- Constrained eight-resonance Lorentzian and pseudo-Voigt oracle fitting with
  scaled bounded parameters, midpoint-fixed linear/quadratic baselines,
  deterministic TRF optimization, non-crossing center boxes, structured
  preflight/optimization/quality failures, exact baseline-only SSE gating,
  raw-fluorescence diagnostics, and public-unit local-linearized uncertainties
  derived from one scaled-Jacobian SVD. Regression coverage includes clean and
  noisy recovery, affine fluorescence invariance, deterministic initialization,
  explicit fallback honesty, parameter validation, and repeatability.
- Deterministic baseline-aware initialization for complete sweeps, including
  three robust baseline-rejection updates and a final discovery fit, physical-
  frequency candidate separation and linewidth interpolation on nonuniform
  grids, model-specific eta guesses, raw detrended amplitudes, explicit-only
  feasible fallback guesses, stable scarcity/failure diagnostics, a signal-
  scaled floating-point discovery floor, and overflow-safe extreme-frequency
  normalization.
- Frozen, slotted estimator contracts for complete sweeps, constrained fit
  configuration, initial guesses, initializer diagnostics, local covariance
  uncertainties, structured spectrum-fit outcomes, and sweep-level resource
  totals. Arrays are defensive float64 read-only copies; fit failures preserve
  diagnostics without final estimates or uncertainty, and fitted Q values are
  derived directly as signed center/FWHM values.
- Approved six-stage estimator-benchmark decomposition and detailed Stage 6.1
  design for constrained eight-resonance offline fitting and an independent
  repeated full-sweep baseline.
- Added the test-driven Stage 6.1 oracle/full-sweep implementation plan.
- Hardened that plan after internal scientific review with explicit
  identifiability/failure rules, public-unit covariance transformation, exact
  ordered-center bounds, deterministic initializer formulas, public schemas,
  and numerical regression fixtures.
- Closed its second review findings with affine fluorescence scaling, a shared
  SVD cutoff for rank and covariance, an exact baseline-only quality reference,
  feasible fallback geometry, immutable initial-guess provenance, and explicit
  failure-field invariants.
- Added explicit zero-variation sweep failure semantics and fully pinned the
  Stage 6.1 clean-environment verification commands.
- Separated algebraic affine-scaling tests from optimizer-level scientific
  equivalence tolerances so harmless floating-point termination changes are not
  treated as scientific failures.

- Packaged the deterministic drift configuration for wheel-only,
  arbitrary-working-directory use via
  `odmrbench simulate --config bundled:drift`; CI verifies wheel contents and a
  clean-virtual-environment smoke. Source `configs/drift.yaml` remains
  available as the single canonical scenario source.
- Added `unknown_analog_signal` alongside `conflicted_unverified` to
  dataset-info and playback JSON summaries.
- `odmrbench dataset-info`, explicit-local `odmrbench playback`, and
  YAML-driven `odmrbench simulate` commands with sorted, finite JSON summaries.
  The commands distinguish unverified raw recorded signal from seeded synthetic
  emulation and make no fitted-tracking or performance claim.
- An eight-resonance linear-drift, Poisson-noise configuration; optional-dataset
  provenance/limitation documentation; synthetic-emulator documentation; and a
  small in-memory playback/emulation example that never downloads external data.
- Event-driven `ODMRInstrument` queries with overhead-before-integration
  virtual-time sequencing, midpoint hidden-truth evaluation, end timestamps,
  normalized-fluorescence photon accounting, and no wall-clock dependency.
- Atomic query commits that preserve clock, sequence, resources, seeded RNG
  behavior, and stateful empirical-noise cursors when validation or sampling
  fails. Noise now has an explicit in-place checkpoint/restore extension
  contract, so third-party strategies cannot corrupt external aliases through
  reflective deep-copy rollback and are rejected before sampling without it.
  Full records retain expected photons while estimator views do not.
- Seeded Poisson shot noise, controlled Gaussian normalized-fluorescence noise,
  and provenance-bearing empirical residual noise with explicit replay, sample,
  and contiguous-block correlation semantics.
- Frozen evaluator-owned `InstrumentObservation` records and separate
  estimator-safe observations that structurally exclude signal-conditioned
  expected photon counts and other hidden truth.
- Atomic resource-ledger snapshots covering observation count, integration,
  nominal/expected/realized photons, observations without counts, and virtual
  elapsed time.
- Frozen hidden eight-resonance `SpectralSnapshot` records and a
  runtime-checkable `SpectralDynamics` interface for virtual-time truth.
- Deterministic stationary and common/per-ID linear center drift that preserve
  physical parent IDs and input order without frequency sorting.
- `load_verified_sweep_file` for deterministic, injected-record verification
  without a network dependency, plus complete YAML/dataclass parity coverage.
- `run_playback`, which supplies estimator callbacks with one frozen
  `PlaybackObservation` at a time. Process isolation remains required for
  deliberately adversarial estimator code.
- Checked versioned provenance for Figshare DOI `10.6084/m9.figshare.28788437.v1`,
  an explicit-local-path verifier, and a parser that preserves original raw
  sweep order without normalization or implicit downloads.
- Immutable raw `SweepDataset` records and causal row-major playback
  observations, with nominal timestamps available only through an explicit
  caller-supplied clock assumption.
- Vectorized Lorentzian, Gaussian, and FWHM-matched pseudo-Voigt profiles.
- Explicit Lorentzian HWHM and Gaussian sigma conversion helpers with tests.
- Validated baseline and resonance parameter objects, Q calculation, and deterministic multi-resonance composition.
- YAML-driven script for generating the synthetic eight-resonance demonstration plot.
- Reusable static-spectrum configuration and numerical curve helpers with
  regression coverage for all eight unique resonance identities and dips.
- Installable Python 3.11+ package scaffold and `odmrbench` console entry point.
- MIT license, citation metadata, Ruff/pytest configuration, and GitHub Actions CI.
- Stage 0 scientific specification defining observables, linewidth conventions,
  causal benchmark modes, resource budgets, fairness rules, noise semantics,
  truth isolation, and the first end-to-end milestone.
- Initial project status and README documents.
- Persisted design record for the Stage 0 specification.
- Public GitHub repository configuration for ongoing synchronized development.
- Detailed, test-driven implementation plan for the repository scaffold and
  explicit-FWHM spectral-model stages.
- Approved real-data-grounded playback and event-driven virtual-instrument
  design covering causal order/time, estimator-safe observations, photon
  accounting, modular empirical/synthetic noise, and linear center drift.
- Verified the provenance, license, checksum, matrix structure, frequency grid,
  and scientific limitations of Figshare DOI
  `10.6084/m9.figshare.28788437.v1` for optional playback and semi-empirical use.
- Added the test-driven playback and virtual-instrument implementation plan.
