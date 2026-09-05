# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow [Semantic Versioning](https://semver.org/)
once a package version is introduced.

## [Unreleased]

### Changed

- Made repository-local test support and estimator test directories explicit
  packages so clean Linux direct-pytest collection resolves
  `tests.two_point_helpers`; the repair passed the complete GitHub CI matrix on
  Python 3.11 and 3.12.
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

- Added the public final/slotted `CalibratedTwoPointTracker` construction,
  reset, pair-boundary affordability, first-minus query, and budget-stop
  surface. Reset transactionally binds exact configuration, clock,
  availability, metadata-resource, starting-boundary, charged-resource,
  ceiling, and seed joins; seeds eight calibration-derived identity estimates
  with signed mapped physical epochs and mode-specific sequence ages; and
  reports source resources while charging them only for included same-run
  calibration. First-query selection reserves two left-associated atomic
  charges across all four capped dimensions, is idempotent while pending, and
  stops an unaffordable boundary without creating a partial pair. Observation
  acceptance, second queries, completed pairs, identity advancement, odd
  alternation, and evaluator APIs remain intentionally unimplemented. Review
  hardening moves first-query and both sequential-charge representability into
  the atomic reset boundary, rejecting constructor-valid cross-record nominal,
  elapsed, endpoint, and charged-total overflow before state commit so query
  selection cannot raise from accepted reset state. The final task re-review
  reported zero Critical, Important, or Minor findings.
- Added the public calibrated two-point discriminator-cell factory and its
  canonical scalar target-only model. The model evaluates the baseline once,
  subtracts all eight dips in immutable source-fit order, and changes only the
  selected target center; the analytic center derivative is checked against a
  centered numerical discriminator derivative. Per-identity calibration fixes
  ordered-difference Voronoi cells, capture-plus-probe allowed-center insets,
  target-pair depth, and positive slope geometry, including exact endpoint,
  one-ULP outward, empty-cell, unequal-width, and construction-precedence
  coverage. Caller-asserted sources reject included-same-run treatment while
  verified provenance supports both declared budget treatments. Review
  hardening adds constructible-subclass witnesses for every exact public
  argument boundary, pins type-error precedence over later geometry/treatment
  defects, and proves a sign-preserving analytic derivative perturbation is
  rejected when it disagrees with the independent numerical slope. The
  zero-discriminator expectation now uses a separate explicit model expression.
- Added estimator-safe, arrival-ordered public resource atoms and replay for
  calibrated two-point traces. The new caller-asserted calibration-source
  factory snapshots the exact public trace, binds derived sweep endpoints,
  fit-input facts, IDs, and safe resources without refitting, applies the
  closed construction-error precedence, and requires the bit-exact ordered
  mean of reconstructed first/last public midpoints. The calibration-source
  intrinsic resource check now calls the same canonical replay, so long source
  snapshots survive reconstruction without arithmetic regrouping. Review
  hardening reconstructs every exact nested public record at its owning
  precedence stage, preserving typed construction errors for malformed trace,
  fit, identity, provenance, and clock values. A combined fitted-ID mismatch
  and malformed-fit witness additionally fixes code-6/code-7 precedence; the
  final task re-review reported zero Critical or Important findings.
- Added exact frozen/slotted two-point query, partial-pair, pair-result,
  identity-estimate, aggregate-estimate, and update contracts. Intrinsic
  validation closes local query/observation echoes, alternating adjacent pair
  sides, public reference/release selection, success/failure diagnostic
  matrices, active-source and history equations, causal ages, pending/partial
  states, stopped boundaries, safe resource counts, and update-side echoes.
  Accepted IDs and closed string literals are canonicalized to exact built-in
  strings, preventing capability-bearing string subclasses from remaining
  reachable through the estimator record graph.
  The estimator record graph remains free of hidden truth, full instrument
  observations/resources, expected photons, callbacks, evaluator objects, and
  futures; contextual reset, tracker, runner, and acquisition authentication
  remains deliberately outside these constructors. After focused fixes for
  diagnostic prefixes, accepted-trace endpoints, target-center echoes, exact
  nested record types, and capability-bearing string subclasses, the task
  passed final spec and quality re-review with no findings.
- Added frozen/slotted Stage 6.3 calibration-source, per-identity-calibration,
  and aggregate-calibration contracts. Caller-asserted sources snapshot their
  declared fit, configuration, identity, observation, provenance, resource,
  and clock values while direct verified provenance construction is rejected.
  Aggregate calibrations retain their exact authenticated source object and
  independently snapshot configuration and identity-calibration value records.
  Task review added explicit nested reconstruction for fit diagnostics,
  optional uncertainty arrays, initial guesses, fitted parameters, and Q
  values; the amended task passed spec and quality re-review with no findings.
- Added the Stage 6.3 calibrated two-point tracker's initial public contracts:
  immutable/slotted resource, budget, identity, normalized-fluorescence,
  clock, configuration, and run-metadata records; closed typed construction
  errors; and only their task-owned estimator exports. Canonical scalar and
  intrinsic cross-field validation preserves the estimator-safe public boundary
  without exposing calibration, query, update, or tracker implementation APIs.
  Task review then added strict capture-fraction-below-offset validation and
  restricted ID/sampling-rule snapshots to ordered tuple/list inputs, rejecting
  unordered sets, mappings, and generators. The amended task passed spec and
  quality re-review with no findings.
- Drafted and revised the twenty-task Stage 6.3 test-driven implementation
  plan for the approved calibrated two-point tracker. Its independently reviewable gates
  cover truth-excluding public contracts, exact safe/full resource arithmetic,
  caller-asserted and opaque-token verified calibration, analytic discriminator
  geometry and fixed identity cells, deterministic pair scheduling and atomic
  policy updates, lossless calibration outcomes, runner retry/stop/ordinary-
  exception abort semantics, typed resource-unavailable joins, exact public
  versus actual midpoint handling, closed static/Poisson/drift/contrast-loss
  regressions, documentation/example/package smoke, and final independent
  scientific/software review. The revision closes all eight Important and
  three Minor findings from its first adversarial review by making each
  dependency and RED/GREEN boundary explicit. A second bounded correction
  fixes exact estimator resource signatures/field names/addition association,
  moves evaluator resource primitives before record validators, places reset
  and first-/second-side rollback REDs before transactional implementations,
  and separates the example and documentation TDD cycles. Its final bounded
  adversarial review reported zero Critical, Important, or Minor findings.
  The plan changes no production code and preserves the Stage 6.5 matched-
  budget-superiority boundary.
- Specified the Stage 6.3 calibrated two-point center tracker without starting
  implementation: one immutable calibration source binds its public fit,
  exact-ID check or explicit fit-ID adoption, normalized safe trace, sweep
  bounds, resources, physical epoch, availability endpoint, and clock mapping;
  same-run versus conditional-free calibration treatment remains required with
  no default. The corrected design also fixes conservative source/Voronoi
  identity domains, analytic target-only normalized-discriminator slopes,
  adjacent/alternating pairs, reset-bound ceilings, exact pending/query/update
  and aggregate-state joins, instrument-compatible atomic resource arithmetic,
  bounded hertz corrections, policy-only lock/common-mode diagnostics,
  distinct overflow-safe public and actual-instrument pair references,
  evaluator full-observation joins, terminal post-query abort records,
  nonnegative deterministic seeds, and fully frozen static/Poisson/drift/
  contrast-loss regressions. The second-review corrections bind verified
  source provenance to an opaque runner token, exact instrument rate/overhead,
  and a continuous same-run ledger; split intrinsic record validation from
  contextual joins; preserve every committed calibration observation in typed
  success/failure outcomes; and give one evaluator runner exact normal,
  instrument-failure, external-stop, budget-stop, and terminal-abort semantics.
  They also pin successful zero-step source refresh and signed mapped-reference
  domains. The latest re-review corrections snapshot the issued pending query
  before covered update failures; separate resource-unavailable raw records
  from authenticated replays; close public construction, observation, update,
  and preflight error codes; require caller-asserted public-midpoint epochs; and
  join accepted charged prefixes before an optional authenticated abort atom.
  The final bounded adversarial design review reported zero Critical,
  Important, or Minor findings, closing the design gate before implementation
  planning.
  The scientific specification fixes equal integration time as the Stage 6.5
  primary budget, with equal nominal exposure only under a shared nominal rate
  and expected photons reported evaluator-side only.
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
