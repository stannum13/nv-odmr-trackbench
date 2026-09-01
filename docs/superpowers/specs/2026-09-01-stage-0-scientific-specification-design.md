# Stage 0 Scientific Specification Design

## Purpose

Establish the scientific contracts and fairness rules for a public benchmark of
causal tracking strategies for eight resolved NV-ensemble ODMR resonances. This
design governs the initial vertical slice and provides constraints for later
physics, dataset, estimator, metric, and reporting stages.

## Chosen approach

Use a vertical-slice-first architecture. Define stable boundaries for spectra,
observations, estimator access, truth, and resource ledgers, then implement only
the components required to compare a full sweep with a two-point center tracker
and five-point local linewidth tracker. Broader abstractions are added when a
second implementation requires them.

This was selected over a layer-complete approach, which would delay a
falsifiable benchmark, and a tightly coupled prototype, which would make causal
integrity and matched-budget accounting difficult to audit.

## Scientific boundary

The benchmark initially models eight parent electronic resonances. Each has a
center, explicit FWHM, amplitude or declared contrast, pseudo-Voigt mixture, and
Q defined as center divided by FWHM. Lorentzian, Gaussian, and pseudo-Voigt
profiles share an explicit FWHM convention. Hyperfine subcomponents remain a
future parent-linked extension.

Q is treated as a fitted spectral descriptor, not a standalone magnetometric
sensitivity measure or an intrinsic coherence quantity.

## Execution architecture

Recorded playback streams immutable observations in original order and never
invokes estimator query selection. Closed-loop execution gives a restricted
instrument interface to an interrogation policy and advances a virtual clock
according to integration and configured overheads. Neither mode gives an
estimator a reference to scenario truth.

The evaluation layer alone joins estimator outputs with hidden truth or an
offline real-data reference. Estimate records identify the final observation
and timestamp incorporated, allowing latency and causal audits.

## Resource and fairness architecture

A resource ledger accumulates observation count, integration time, expected and
realized photons, virtual elapsed time, and estimator CPU time. Benchmark
comparisons declare one primary matched resource while reporting every resource.
Block-size mismatches use explicit stopping rules and are reported rather than
silently called equal.

Expected photons are used for deterministic photon-budget matching. Realized
Poisson counts remain a stochastic result. Compute time is measured but not
mixed into virtual acquisition time.

## Data and noise flow

In closed loop, a query specifies frequency and integration duration. Hidden
dynamics are evaluated over the resulting virtual-time interval, the spectral
model determines the expected rate or fluorescence, the selected noise model
produces an observation, and the resource ledger advances. Gaussian,
photon-counting, and empirical-residual noise are separate strategies with
recorded parameters and deterministic random generators.

For real data, online replay is completed without an offline reference being
visible to the estimator. Adaptive queries require the emulator. A real fitted
spectrum plus experimental residuals and synthetic dynamics is labeled
semi-empirical simulation.

## Error handling and invariants

Public constructors and query methods reject non-finite frequencies,
non-positive linewidths, negative photon rates, non-positive integration
durations, invalid pseudo-Voigt fractions, and non-monotonic playback times.
Physical units appear in field names. Resonance IDs remain physical labels even
when their sorted frequency order changes.

Configuration resolution must fail before execution when units, budgets,
scenario types, or estimator capabilities are inconsistent. Fit failures and
lock loss are recorded as outcomes rather than replaced with hidden truth.

## Verification strategy

The first implementation plan will use test-driven tasks. Unit tests will prove
line-height and FWHM conventions, conversion formulas, eight-resonance
composition, deterministic noise, virtual-time monotonicity, causal estimator
updates, Q calculation, and resource-ledger arithmetic. Regression tests will
bound the first deterministic tracking scenario. CLI reproducibility tests will
compare resolved outputs for identical seeds while excluding measured CPU time
from byte-for-byte deterministic assertions.

At every completed stage, tests and linting are run, the diff receives
scientific and software reviews, project state and changelog are updated, and a
single descriptive local commit is created. Configured remotes may be pushed
only with explicit user authorization.

## Scope boundary

Stage 0 writes documentation only. It does not introduce package metadata,
models, estimators, benchmark results, dataset claims, or dependencies. The next
plan begins with repository scaffolding and the spectral-model portion of the
first vertical slice.
