# Offline full-sweep estimation

The Stage 6.1 estimator fits a completed ODMR frequency sweep with a constrained
eight-component model. A successful result is conditional on that model, its
initializer, and the configured quality thresholds; it is not proof that the
spectrum contains eight physical resonances. The fit is a reference for later
estimator evaluation, not a result from a matched-budget comparison.

## Input and model

Construct a `CompleteSweep` from one-dimensional, finite arrays with strictly
increasing frequency in Hz. Optional completion metadata records the final
sequence index and timestamp plus cumulative integration time and nominal
photon exposure. The fitter does not sort samples, fill missing values, or
receive synthetic truth.

`FitConfiguration` selects either eight Lorentzian dips or eight FWHM-matched
pseudo-Voigt dips. Lorentzian fits fix the mixing value `eta` to one;
pseudo-Voigt fits constrain it to `[0, 1]`. A linear baseline is the default,
and a quadratic baseline can be selected. Public linewidths are FWHM in Hz and
each reported Q is calculated as `center_hz / fwhm_hz`.

## Initialization and bounds

Automatic initialization is deterministic and data-derived. It estimates and
removes a low-order trend, smooths only for candidate discovery, finds dips by
prominence, and estimates their local depths and half-prominence widths. No
expected center positions are supplied by the generated-data evaluator. If
discovery does not find eight candidates, fitting stops unless the
configuration explicitly enables the evenly spaced fallback; diagnostics
record that choice.

The center search is local and candidate-conditioned. For adjacent ordered
initial centers `g_i < g_(i+1)`, let `m_i = (g_i + g_(i+1)) / 2` and let `d` be
`min_center_separation_hz`. The boxes use
`upper_i = m_i - d / 2` and `lower_(i+1) = m_i + d / 2`, with
`lower_0 = f_min` and `upper_7 = f_max`. Preflight rejects empty boxes and
infeasible boundary geometry. These boxes preserve minimum separation, but a
noise peak or unresolved feature chosen by the initializer can constrain the
optimizer to the wrong neighborhood. Ordered output IDs likewise do not prove
physical identity.

Optimization constrains eight unique resonance identities to strictly ordered
centers inside the sweep interval. FWHM is positive and bounded by the
configured minimum and maximum, amplitudes are non-negative and bounded, and
pseudo-Voigt mixing values remain in `[0, 1]`. The linear and quadratic
baseline coefficients are finite but otherwise unconstrained.

Strict center ordering is suitable for the initial resolved, non-crossing
scope. It does not establish physical identity through crossings or collisions,
and it must not be used to hide an identity swap in a later scenario.

## Results, failures, and uncertainty

`fit_spectrum` returns an immutable `SpectrumFitResult`. Successful results
contain eight ordered resonance records, a baseline, residual diagnostics, and
derived Q values. Valid spectra can instead return structured failures for
insufficient samples, uninformative fluorescence, failed initialization,
optimization failure, or post-fit quality failure. Failed attempts retain
diagnostics but do not expose plausible-looking resonance or baseline
estimates. Malformed arrays or configurations raise validation errors before
optimization.

Success also requires every amplitude to meet both the absolute
`min_resolved_amplitude` gate and the configured
positive finite `min_amplitude_significance` gate, whose default is 3.0. For
each component, the latter is fitted amplitude divided by its public-unit
amplitude standard error from the same local packed covariance used for
uncertainty. A positive
amplitude with an exactly zero standard error has infinite significance;
unavailable or non-finite evidence fails conservatively.

When the final scaled residual Jacobian has full numerical column rank and
positive degrees of freedom, standard errors are transformed back to public
units. These are local linearized Jacobian uncertainties. They are not
experimental coverage guarantees. In particular, amplitude significance is a
model-conditioned local diagnostic, not a calibrated line-detection statistic
or false-discovery guarantee. An unavailable or numerically unrepresentable
covariance is reported with a reason rather than fabricated. Unresolved,
overlapping, hyperfine-rich, or otherwise model-mismatched spectra can still be
misclassified and remain outside the Stage 6.1 oracle's validated scope.

## Repeated cold-start sweeps

`RepeatedFullSweepEstimator` accepts one `CompleteSweep` at a time. Every call
uses `fit_spectrum(..., initial_guess=None)`, so neither a successful nor a
failed fit supplies parameters to the next sweep. Each immutable
`SweepEstimate` copies completion metadata only from that input sweep.

`latest` is `None` before the first update and then refers to the most recent
attempt, including a structured failure. `history` returns an immutable tuple
of every attempt in input order for evaluator/reporting use. `reset()` clears
that retained history. The wrapper deliberately accepts metadata that is not
monotonic across calls because an evaluator may submit independent recordings;
cross-recording causal checks belong to that evaluator.

This complete-sweep API is distinct from the sample-wise adaptive estimator
interface planned for a later stage.

## Synthetic example and recording interpretation

From a source checkout with the package installed, run:

```bash
python examples/fit_synthetic_sweep.py
```

The example generates one deterministic pseudo-Voigt sweep in memory and
prints finite fitted center, FWHM, and Q diagnostics. It does not download a
recording and its generated fixture is not a benchmark measurement.

A fit to the optional external recording can be described only as an apparent
observable or offline reference. That recording has no verified eight-line
identity, trajectory, timestamp, photon calibration, or center/FWHM/Q truth.
Residual structure may include model mismatch and instrument effects rather
than detector noise alone.
