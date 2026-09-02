# Project State

Last updated: 2026-09-03

## Current stage

Stage 5 — Event-driven virtual instrument is complete. The playback and
emulation proof-of-concept CLI is next.

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

- `pytest`: 183 passed.
- `ruff check .`: All checks passed.

## Known scientific limitations

- The current dynamics layer provides only stationary and deterministic linear
  center drift; no Hamiltonian model exists yet.
- No real dataset has been verified or attached.
- Hyperfine structure, ensemble inhomogeneity, optical power broadening,
  microwave power broadening, temperature coupling, and instrument transfer
  functions are not yet modeled.
- No benchmark results exist, so neither primary nor secondary hypothesis has
  supporting evidence.

## Known software limitations

- End-to-end benchmark reproducibility has not yet been demonstrated beyond the
  installable package, continuous-integration scaffold, and deterministic
  configuration-generated spectrum fixture.

## Next actions

1. Add proof-of-concept playback and emulation CLI commands and their explicit
   deterministic drift configuration.
2. Exercise the deterministic linear-drift scenario before the matched-budget
   full-sweep versus sparse-tracker milestone.
