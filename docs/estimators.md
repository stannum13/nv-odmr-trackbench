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

The fitter derives one fixed fluorescence origin `y_ref` from the observations.
Its optimizer compares `y - y_ref` with a model whose baseline intercept is
`b0 - y_ref`; it never reconstructs and subtracts two large absolute model/data
origins inside the residual. The degree-matched baseline-only quality reference
uses the same centered target. Reported intercepts remain in public fluorescence
units, and cost and RMSE remain in raw squared-fluorescence and fluorescence
units respectively. Directly adding a representable constant may still quantize
the input samples, so exact equality or threshold decisions arbitrarily close
to a boundary are not universally guaranteed. Fixed noisy direct-addition
regressions pin the scientific classification, ordered IDs, and rank decision
away from such a boundary.

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

Every optimizer attempt retains a nonempty SciPy message, an integral status,
and a positive evaluation count. Success and `quality_failed` represent a
nominally successful optimizer termination and therefore require a positive
status; `optimization_failed` requires a non-positive status. Pre-optimizer
failures retain status/message as `None` and `nfev=0`.

Success also requires every amplitude to meet both the absolute
`min_resolved_amplitude` gate and the configured
positive finite `min_amplitude_significance` gate, whose default is 5.0. For
each component, the latter is fitted amplitude divided by its public-unit
amplitude standard error from the same local packed covariance used for
uncertainty. A positive
amplitude with an exactly zero standard error has infinite significance;
unavailable or non-finite evidence fails conservatively.

When the final scaled residual Jacobian has full numerical column rank and
positive degrees of freedom, standard errors are transformed back to public
units. `FitUncertainty.method` is exactly
`local_linearized_jacobian_covariance`; other method labels are rejected.
These are local linearized Jacobian uncertainties. They are not
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

## Warm-started completed sweeps

`WarmStartedFullSweepEstimator` is a causal, completed-sweep baseline. It
accepts one already acquired `CompleteSweep` at a time and only the latest
earlier successful public fit may be the next warm source. It does not receive
dynamics, snapshots, truth, future observations, or evaluator-private photon
information. Callers should submit every acquired sweep in order.

The constructor takes an immutable `FitConfiguration` plus two keyword
options. `retry_cold_on_warm_failure=True` retains a failed warm attempt whose
failure code is exactly `optimization_failed` or `quality_failed`, then
performs at most one cold recovery on the same acquired sweep. Other failure
codes do not trigger a retry. `max_warm_start_age_updates=None` disables an
update-count age limit; an explicitly configured positive integer rejects older
warm sources and runs one cold attempt. Rejection affects seeding, not the
availability of an older successful estimate.

Before a warm fit, the prior polynomial baseline is transformed to the current
overflow-safe sweep midpoint. This baseline rebase preserves the old baseline
function at the new reference; any overflow, nonzero underflow, or otherwise
unrepresentable product rejects warm use rather than inventing a coefficient.
Changed grids and spans are supported when the prior resonance parameters are
compatible with the current center boxes and configured bounds. The ordered
center boxes assume eight resolved, noncrossing components; fitted ID order is
not evidence of collision identity.

Each accepted update is a `WarmSweepEstimate`. Its `attempts` retain one
`SweepFitAttempt`, or the ordered failed-warm/conditional-cold pair. Every
attempt exposes `start_kind`, `warm_source_update_index`, its full `fit`, and
attempt CPU time. `warm_start_disposition`, `warm_start_rejection_code`, and
`warm_start_message` distinguish use, age rejection, compatibility rejection,
no successful prior, and a start-independent preflight failure. A failed warm
attempt therefore remains visible before conditional cold recovery.

`current_fit` is the final attempt on the current update. `active_fit` is
instead the current success or the latest older success, with
`active_source_update_index` identifying its origin. Thus current failure and
an older active success are distinct; `is_stale` reports that distinction.
Stale age is recorded independently as
`estimate_age_submitted_observations`, `estimate_age_sequence_indices`, and
`estimate_age_s`. These submitted-observation, external sequence-index, and
timestamp bases are never substituted for or mixed with one another. A
successful current fit has zero age on every available basis.

Each record's `update_index` is its zero-based accepted-update position.
Resource fields copy or derive from the submitted sweep exactly:
`observation_count`, `cumulative_observation_count`, `first_sequence_index`,
`last_sequence_index`, `last_timestamp_s`, `total_integration_time_s`, and
`total_nominal_exposure_photons`. `first_sequence_index` is derived from the
submitted observation count and inclusive last sequence index when that basis
is available. Retries add CPU and `nfev` but no acquisition resources.
`total_nfev` sums retained attempts. `cpu_time_s` is the
measured update-core process CPU interval through the instant before record
construction/state append, is at least the attempt sum, and is
machine-dependent diagnostic data rather than an acquisition-time or
performance guarantee.

The estimator exposes immutable `configuration`, `latest`, `history`, and
`latest_success` properties, plus `update_sweep()` and `reset()`. Sequence and
timestamp endpoint availability must remain consistent within a recording;
sequence ranges cannot overlap and timestamps must increase. Endpoint
inconsistency, overlap, and unexpected exceptions abort an update atomically.
A benchmark harness must not skip that acquired sweep and continue as though it
had been recorded. Call `reset()` between independent recordings so endpoint-
availability modes, earlier successes, stale age, and source provenance cannot
cross recording boundaries.

This completed-sweep baseline makes no within-sweep realtime claim. It does not
establish temporal bandwidth, realtime utility, universal speedup, collision
identity, or matched-budget superiority. CPU time and evaluation counts are
descriptive diagnostics only and must be interpreted alongside the unchanged
acquisition resources and machine/software environment.

## Synthetic example and recording interpretation

From a source checkout with the package installed, run:

```bash
python examples/fit_synthetic_sweep.py
python examples/fit_warm_started_sweeps.py
```

The first example generates one deterministic pseudo-Voigt sweep in memory and
prints finite fitted center, FWHM, and Q diagnostics. The warm-started example
generates three causally submitted drift sweeps and reports source, age,
attempt, optimizer-evaluation, and process-CPU diagnostics. Neither downloads
a recording, and neither generated fixture is a benchmark measurement or a
performance comparison.

A fit to the optional external recording can be described only as an apparent
observable or offline reference. That recording has no verified eight-line
identity, trajectory, timestamp, photon calibration, or center/FWHM/Q truth.
Residual structure may include model mismatch and instrument effects rather
than detector noise alone.
