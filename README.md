# NV ODMR TrackBench

A reproducible benchmark for comparing realtime ODMR resonance and linewidth
estimators under matched acquisition budgets.

The project targets an NV ensemble with four crystallographic orientations and
two electronic spin transitions per orientation. It asks which causal strategy
best tracks the center frequency, FWHM, and Q factor of all eight resolved ODMR
resonances when measurement time, fluorescence samples, or photon budget is
held fixed.

Two benchmark modes are in scope:

- recorded playback, preserving the original causal acquisition order; and
- closed-loop emulation, in which an estimator selects the next microwave
  frequency to interrogate.

The scientific contracts and fairness requirements are specified in
[docs/scientific_spec.md](docs/scientific_spec.md). The installable package and
CI scaffold are in place; executable spectral models are the current stage.

## Planned first milestone

The first end-to-end result will compare repeated full-spectrum fitting with a
two-point center tracker plus a lower-rate five-point linewidth estimator on a
virtual eight-resonance spectrum. Comparisons will report center, FWHM, and Q
errors together with acquisition and compute resources.

## Scientific caution

The Q factor used by this project is the fitted resonance center divided by its
FWHM. Q alone is not equivalent to magnetometric sensitivity. Contrast,
fluorescence or photon rate, spectral slope, acquisition bandwidth, and the
noise model must also be considered.

## Project status

See [PROJECT_STATE.md](PROJECT_STATE.md) for completed work, limitations, and
the next planned actions. Installation and executable examples will be added as
the first vertical slice is implemented.
