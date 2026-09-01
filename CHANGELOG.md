# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow [Semantic Versioning](https://semver.org/)
once a package version is introduced.

## [Unreleased]

### Changed

- Renamed the public distribution and repository to `nv-odmr-trackbench`;
  the `odmr_bench` import package and `odmrbench` CLI remain unchanged.
- Canonicalized validated real parameter scalars to immutable Python floats and
  rejected boolean, complex, array, non-finite, and non-string-ID inputs.
- Made the deterministic additive-noise boundary explicit: callers supply an
  already-realized broadcastable perturbation and sampling remains external.
- Extended CI to build and install the wheel and smoke-test the installed CLI.

### Added

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
