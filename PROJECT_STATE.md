# Project State

Last updated: 2026-09-01

## Current stage

Stage 0 — Scientific specification complete and user-approved. The detailed
plan for the repository scaffold and spectral-model stages is complete and
awaiting execution.

## Completed work

- Inspected the repository and confirmed it began empty, without prior commits
  or benchmark outputs.
- Defined the scientific observables and unambiguous FWHM/Q conventions.
- Separated causal recorded playback from interactive closed-loop emulation.
- Defined acquisition-resource accounting and matched-budget comparison rules.
- Defined estimator truth-isolation and causal-access requirements.
- Scoped the first end-to-end milestone and its required outputs.
- Created the public GitHub repository and configured it as the `origin` remote:
  `https://github.com/stannum13/nv-odmr-realtime-benchmark`.
- Received user approval of the Stage 0 scientific specification.

## Important scientific and design decisions

- Project identity is `nv-odmr-realtime-benchmark`; the existing checkout
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

- No executable test suite exists yet because Stage 0 contains documentation
  only.
- Documentation whitespace and repository-reference checks are run manually at
  the end of this stage.

## Known scientific limitations

- No implemented spectral, noise, dynamics, or Hamiltonian model exists yet.
- No real dataset has been verified or attached.
- Hyperfine structure, ensemble inhomogeneity, optical power broadening,
  microwave power broadening, temperature coupling, and instrument transfer
  functions are not yet modeled.
- No benchmark results exist, so neither primary nor secondary hypothesis has
  supporting evidence.

## Known software limitations

- The Python package, dependency metadata, CLI, configs, tests, and CI do not
  yet exist.
- Reproducibility has been specified but not yet demonstrated in code.

## Next actions

1. Review and execute the detailed implementation plan for the repository
   scaffold and spectral-model stages.
2. Implement the scaffold and spectral models with explicit FWHM conversion
   tests.
3. Add the event-driven virtual instrument and deterministic drift scenario.
4. Complete the matched-budget full-sweep versus sparse-tracker milestone.
