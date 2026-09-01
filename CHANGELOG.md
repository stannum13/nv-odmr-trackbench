# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow [Semantic Versioning](https://semver.org/)
once a package version is introduced.

## [Unreleased]

### Changed

- Renamed the public distribution and repository to `nv-odmr-trackbench`;
  the `odmr_bench` import package and `odmrbench` CLI remain unchanged.

### Added

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
