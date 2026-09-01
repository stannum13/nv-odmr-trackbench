# Project State

Last updated: 2026-09-01

## Current stage

Stage 2 — Spectral models complete; next: virtual NV instrument

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
  Hz-valued fields and a reference-centered polynomial baseline.
- Added deterministic eight-dip spectrum composition with stable parent IDs
  and caller-supplied additive noise.
- Added a YAML-driven script that generates the synthetic eight-resonance
  demonstration plot from package code.

## Important scientific and design decisions

- Project identity is `nv-odmr-trackbench`; the existing checkout
  directory is retained.
- Internal frequency and linewidth units are Hz. Internal time units are
  seconds. Public APIs must not accept ambiguous unitless physical quantities.
- Each line component uses FWHM directly. Lorentzian HWHM/gamma and Gaussian
  sigma conversions must be explicit and unit-tested.
- Q is defined as resonance center divided by FWHM and is not treated as a
  proxy for magnetometric sensitivity by itself.
- The initial benchmark represents eight electronic resonances. Optional
  hyperfine components may later retain parent electronic-resonance identities.
- Scenario truth belongs to the virtual instrument and evaluation harness; an
  estimator receives only observations and permitted public metadata.
- Recorded playback cannot evaluate adaptive frequencies that were not present
  in the recording.
- Budget matching and any unavoidable budget mismatch must be explicit in
  machine-readable results and figures.
- Development prioritizes the first falsifiable vertical slice over completing
  every planned abstraction in advance.

## Tests currently passing

- `pytest`: 22 passed.
- `ruff check .`: All checks passed.

## Known scientific limitations

- No stochastic observation-noise, dynamics, or Hamiltonian model exists yet;
  the current spectrum model accepts only explicit deterministic additive noise.
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

1. Add the event-driven virtual instrument and virtual clock.
2. Implement seeded Poisson and Gaussian observation-noise models.
3. Add the deterministic linear-drift scenario before the matched-budget
   full-sweep versus sparse-tracker milestone.
