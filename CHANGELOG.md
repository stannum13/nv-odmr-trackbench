# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow [Semantic Versioning](https://semver.org/)
once a package version is introduced.

## [Unreleased]

### Changed

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
