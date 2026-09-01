# Scientific Specification

## 1. Research question

Given a fixed measurement-time, fluorescence-sample, or photon budget, which
realtime estimation strategy best tracks the center frequency, linewidth, and Q
factor of the eight resolved electronic ODMR resonances of an NV ensemble under
realistic dynamics and noise?

This question is comparative rather than confirmatory. The benchmark must be
able to falsify the hypothesis that fast two-point center discrimination plus
lower-rate sparse linewidth fitting retains most of the accuracy of repeated
full-spectrum fitting while providing greater temporal bandwidth. It must also
be able to show when a shared physics-informed tracker is worse than independent
trackers because its physical model is mismatched.

No estimator may access simulation truth, future observations, or observations
outside its declared acquisition budget.

## 2. System under study

The initial system is an NV ensemble with four crystallographic NV orientations
and two electronic ODMR transitions per orientation. These define eight parent
resonance identities. A spectrum at frequency \(f\) and time \(t\) is modeled
conceptually as

\[
I(f,t) = b(f,t) - \sum_{i=1}^{8}D_i(f,t) + \epsilon(f,t),
\]

where \(b\) is the fluorescence baseline, \(D_i\) is a non-negative dip depth,
and \(\epsilon\) represents the declared observation-noise process.

The default baseline is

\[
b(f,t)=b_0(t)+b_1(t)(f-f_{\mathrm{ref}}),
\]

with an optional quadratic term

\[
b_2(t)(f-f_{\mathrm{ref}})^2.
\]

The reference frequency must be stored with the model. Centering the polynomial
at \(f_{\mathrm{ref}}\) reduces coefficient correlation and avoids applying
large powers to absolute microwave frequencies.

## 3. Observables and conventions

For parent resonance \(i\), the primary observables are:

- center frequency \(f_i\), in Hz;
- full width at half maximum \(w_i\), in Hz;
- dip amplitude \(A_i\), in the same fluorescence units as the spectrum, or
  contrast \(C_i\), which must be dimensionless and explicitly normalized;
- pseudo-Voigt Lorentzian mixture fraction \(\eta_i\), dimensionless and in
  \([0,1]\); and
- Q factor \(Q_i=f_i/w_i\), dimensionless.

All internal frequency quantities use Hz and all internal time quantities use
seconds. User-facing configuration may offer documented convenience units, but
configuration parsing must convert them to explicit SI-valued fields before a
simulation or benchmark begins. APIs must not silently infer units.

Low-level spectral functions allow finite signed frequency coordinates for
generality. Physical benchmark scenarios must use positive absolute resonance
centers; scenario-level validation arrives with the instrument stage. This
policy does not silently redefine Q: the benchmark definition remains
\(Q_i=f_i/w_i\), without taking an absolute value or otherwise changing the
signed low-level coordinate.

### 3.1 FWHM-normalized line shapes

Let \(x=f-f_i\) and \(w_i>0\) be the FWHM. Unit-height component profiles are
defined as

\[
L(f;f_i,w_i)=\frac{1}{1+4x^2/w_i^2}
\]

and

\[
G(f;f_i,w_i)=\exp\!\left[-4\ln(2)x^2/w_i^2\right].
\]

Both equal one at \(f=f_i\) and one half at
\(f=f_i\pm w_i/2\). The default pseudo-Voigt dip is

\[
D_i(f)=A_i\left[\eta_i L(f;f_i,w_i)
 +(1-\eta_i)G(f;f_i,w_i)\right].
\]

Because both components have the same half-maximum points, this mixture also
has FWHM \(w_i\). The following conversions are permitted only through named,
tested helpers:

\[
\gamma_{\mathrm{HWHM}}=w/2,
\qquad
\sigma_{\mathrm{Gaussian}}=\frac{w}{2\sqrt{2\ln 2}}.
\]

Lorentzian gamma, HWHM, Gaussian sigma, and FWHM must never share an ambiguous
field named `width` at an interface boundary.

Lorentzian, Gaussian, and pseudo-Voigt profiles are required. A true Voigt
profile may be added later, but its Gaussian and Lorentzian width parameters
must be named separately because a true Voigt FWHM is not equal to either
component width.

### 3.2 Amplitude and contrast

In the additive dip model, \(A_i\) is the isolated dip depth at its center, not
necessarily the total drop of a multi-resonance spectrum at that frequency,
because neighboring tails add. If contrast is reported, its normalization must
be declared, for example \(C_i=A_i/b(f_i)\). Fits and plots must label whether
they report fluorescence amplitude or normalized contrast.

### 3.3 Q and discrimination-related quantities

The benchmark definition is

\[
Q_i=\frac{f_i}{w_i}.
\]

This fitted spectral Q must not be described as an intrinsic decoherence metric
without a justified physical model. Power broadening, unresolved hyperfine
structure, inhomogeneous broadening, strain distributions, and line-shape
mismatch can all change fitted FWHM.

Q alone is not equivalent to magnetometric sensitivity. Where supported, the
benchmark should additionally report:

- maximum absolute ODMR slope in fluorescence per Hz;
- normalized contrast;
- fluorescence or photon rate;
- FWHM;
- acquisition bandwidth or update interval; and
- an explicitly labeled shot-noise-limited sensitivity proxy.

Any sensitivity proxy must state its normalization and assumptions and must not
be labeled as measured sensitivity.

## 4. Benchmark execution modes

Recorded playback and closed-loop emulation answer different questions and are
separate execution protocols. Results must identify their mode.

### 4.1 Mode A: recorded playback

The estimator receives recorded observations in their original causal order:

```python
estimator.update(
    frequency_hz=frequency_hz,
    fluorescence=fluorescence,
    timestamp_s=timestamp_s,
)
```

At update \(k\), the estimator may access observations with indices no greater
than \(k\), acquisition metadata declared public before replay, and its own
previous state. It must not access future samples, future scan boundaries,
future field/current metadata, or an offline reference trajectory.

An evaluation harness may use future data after replay to construct a reference
and calculate metrics, but that reference must not share an object or callback
with the estimator. Offline preprocessing that uses a complete recording must
be labeled noncausal and cannot be counted as online estimator performance.

Playback evaluates algorithms compatible with the recorded acquisition
schedule. It cannot fairly evaluate an adaptive query at a frequency that was
never acquired. Interpolation of missing adaptive observations does not turn a
recording into a closed-loop benchmark.

### 4.2 Mode B: closed-loop virtual instrument

An estimator or interrogation policy chooses each requested measurement:

```python
observation = instrument.query(
    frequency_hz=frequency_hz,
    integration_time_s=integration_time_s,
)
```

The instrument evaluates hidden state at the causal measurement time, samples
the configured observation model, returns an observation, and advances virtual
time without calling `sleep()`. Adaptive frequency sampling, two-point and
multi-point tracking, FSK, and lock-in-inspired modulation must be evaluated in
this mode.

The initial observation schema contains at least:

```python
Observation(
    timestamp_s: float,
    frequency_hz: float,
    fluorescence: float,
    integration_time_s: float,
    expected_photons: float,
)
```

This is the instrument/evaluation record, not necessarily the estimator's
object view. If `expected_photons` is computed from the hidden noiseless
spectrum, it must be withheld from the estimator because it would reveal the
observation mean. The estimator instead receives timestamp, query settings,
measured fluorescence or realized counts, and only acquisition metadata known
without evaluating hidden truth. A separately declared nominal exposure such
as baseline photon rate times integration duration may be public when it is
known before the query.

The timestamp convention must be explicit. The default is the end of the
integration interval: a query beginning at virtual time \(t\) with duration
\(\Delta t\) observes the declared time-averaged signal over
\([t,t+\Delta t]\) and returns timestamp \(t+\Delta t\). A point-sampled
approximation may be used only when declared by the scenario.

Instrument overheads such as frequency-settling or readout dead time may later
be configured separately. When present, they advance virtual elapsed time but
do not contribute photon integration time.

### 4.3 Common estimator boundary

Realtime estimators should converge on a common behavioral contract:

```python
class Estimator:
    def reset(self, public_metadata, seed): ...
    def choose_next_query(self, budget_state): ...
    def update(self, observation): ...
    def estimate(self): ...
```

Playback-compatible estimators may not implement active query selection. An
estimate must record the observation index and virtual timestamp through which
it is valid. Estimator random-number generators must derive deterministically
from a recorded run seed.

## 5. Truth isolation and causal integrity

Scenario truth is owned by the virtual instrument or evaluation layer. It
includes true resonance parameters, disturbance trajectories, latent magnetic
field, noiseless spectra, signal-conditioned expected photon counts, and future
random draws. None is estimator input.

Permitted public information must be explicit in configuration. Examples
include nominal scan bounds, a calibration spectrum, the number of parent
resonances, nominal modulation offsets, and a Hamiltonian model selected for a
physics-informed estimator. The acquisition resources used to obtain a
calibration spectrum count toward the benchmark unless a result is explicitly
labeled as conditional on free pre-calibration.

Tests must include sentinels or separated types that make accidental truth
access difficult. Benchmark orchestration must not pass a scenario or
instrument object directly to an estimator when a restricted observation view
is sufficient.

## 6. Resource accounting and fair comparisons

Each run and estimate must account for:

- `observations`: number of fluorescence observations returned;
- `integration_time_s`: sum of photon-collecting intervals;
- `expected_photons`: sum of configured expected counts before random sampling;
- `realized_photons`: sum of sampled counts when the observation model exposes
  integer photon counts;
- `virtual_elapsed_time_s`: integration plus configured acquisition overheads;
- `cpu_time_s`: measured estimator compute time, reported separately from
  virtual acquisition time.

Fluorescence samples and photon counts are not interchangeable. One sample may
represent a different integration duration and expected photon count from
another. Expected photon budget is the primary deterministic budget for
cross-seed matching; realized photon count is an outcome and is reported for
diagnostics.

Comparisons may match one of the following primary budgets:

1. equal number of observations;
2. equal total integration time;
3. equal expected photon count; or
4. equal virtual elapsed time when overheads matter.

Every comparison must name its primary matched budget and report all resource
dimensions. If exact equality is impossible because an estimator updates in
blocks, the harness must use a declared stopping rule, report the discrepancy,
and avoid presenting the result as exactly matched. A full sweep receiving
substantially more photons than a sparse tracker is not a fair accuracy
comparison unless accuracy is explicitly plotted against that resource.

Realtime performance is determined by measurement latency, integration time,
query count and schedule, acquisition overhead, and compute latency. Fast
execution on one workstation does not by itself establish realtime utility.

## 7. Noise and observation semantics

Noise and dynamics must be modular and separately configurable.

### 7.1 Ideal Gaussian noise

Additive Gaussian noise is permitted for controlled tests. Its standard
deviation and fluorescence units must be recorded. It is not a photon-counting
model and must not be presented as one.

At the low-level deterministic spectrum boundary, `additive_noise` is an
already-realized caller-provided perturbation that may broadcast over the
frequency coordinates. Stochastic sampling, random-number generator ownership,
and noise-process configuration remain outside `multi_resonance_spectrum`.

### 7.2 Photon shot noise

A Poisson model samples non-negative photon counts from the time-integrated
expected photon rate. Configuration must state whether spectral functions
produce a rate, a normalized multiplier on a baseline rate, or expected counts.
Returned normalized fluorescence must preserve or accompany the underlying
count and normalization metadata so photon budgets remain auditable.

### 7.3 Empirical residual noise

Empirical residuals may be sampled or replayed only after their source data and
preparation method have been verified. The implementation must record whether
residuals are sampled independently, in blocks, or replayed in order, because
these choices change temporal correlation. A fitted real spectrum combined
with synthetic dynamics and experimental residuals is a semi-empirical
simulation, not raw experimental playback.

Noise processes must receive deterministic per-run random generators. Identical
configuration, seed, dataset version, and software version must reproduce the
same synthetic observations apart from explicitly documented platform-level
floating-point differences.

## 8. Dynamics and the virtual clock

Dynamics are functions of hidden virtual time, not loop iterations or wall
clock time. Required disturbances will include stationary parameters, linear
center drift, center steps, linewidth drift, contrast drift, baseline drift,
temporary contrast loss, peak approach/collision, and magnetic-field-generated
motion.

During a finite integration, the scientifically preferred expected signal is a
time average over the integration window. An efficient quadrature or midpoint
approximation may be selected per scenario, but the choice must be recorded.
Rapid dynamics relative to integration time must not be hidden by evaluating
only at the query start without disclosure.

## 9. Eight-resonance identity and ordering

Parent resonance IDs are stable physical labels assigned by the scenario, not
indices obtained by sorting the frequencies independently at every timestamp.
Frequency order may change during a collision. Evaluation must therefore
distinguish physical-ID error, sorted-set error, and an identity swap.

An offline fit may enforce ordered centers in regimes where the order is known
not to cross. Such a constraint must be disabled or interpreted carefully in a
collision scenario because it can conceal identity failure.

The initial spectrum has eight parent electronic resonances. Future hyperfine
support may expand each parent into approximately three components for
\(^{14}\mathrm{N}\), while preserving the eight parent IDs. Unresolved or
partially resolved hyperfine structure may bias fitted single-line FWHM and Q;
such values must not be interpreted silently as intrinsic linewidths.

## 10. Ground truth and offline oracle

Synthetic scenarios provide exact hidden truth for evaluation. Real recordings
generally do not. For real data, a high-SNR constrained offline fit may serve as
a reference estimate, but it must be labeled an offline reference or oracle,
not exact ground truth. Its model assumptions, fit failures, and uncertainty
limitations must be retained with results.

The oracle may use complete scans and future data only after the online replay
has been isolated. It is an accuracy reference and is not automatically a
realtime competitor.

## 11. Required outputs and metrics

The first milestone must report, per resonance and in aggregate:

- center-frequency RMSE;
- FWHM RMSE;
- Q RMSE;
- tracking latency or estimate age;
- observations used;
- integration time used;
- expected photon count used;
- virtual elapsed time; and
- estimator CPU time.

Later benchmark stages add MAE, bias, error standard deviation, 95th-percentile
absolute error, identity swaps, lock-loss probability, reacquisition latency,
settling time, maximum reliable slew, and uncertainty calibration where
available.

Aggregation across eight resonances must retain per-resonance results and state
whether a summary is a macro-average, photon-weighted average, or pooled error.
Monte Carlo summaries must report seed count and uncertainty across independent
runs.

Central comparisons will show frequency and Q/FWHM accuracy against photon
budget and temporal bandwidth, plus lock probability against resonance slew.

## 12. First end-to-end milestone

The first executable vertical slice is:

```text
eight pseudo-Voigt resonances
    -> event-driven virtual instrument
    -> deterministic frequency drift plus declared noise
    -> repeated full-sweep fit
       versus
       calibrated two-point center tracker plus periodic five-point local fit
    -> matched acquisition budget
    -> center, FWHM and Q RMSE
       tracking latency, samples, integration time, photons and CPU time
    -> reproducible CLI result and generated figure
```

The two-point discriminator must convert its signed fluorescence difference to
frequency displacement through a calibrated local slope, normalized
discriminator, local model, or state-space observation model. An arbitrary gain
applied to raw intensity difference is not sufficient. The local linewidth fit
uses the fast center estimate as a prior and may update less frequently than the
center loop.

The first benchmark must state exactly how the full sweep and sparse tracker
budgets are matched. Estimator-specific query scheduling is exercised only in
closed-loop mode.

## 13. Scientific claims and limitations

Before executable results exist, the repository claims only to define and
implement a benchmark. It does not claim superior performance, state of the
art, experimental sensitivity, or hardware readiness.

The minimum Hamiltonian planned for later stages omits or approximates
hyperfine structure, ensemble strain distributions, temperature-dependent
zero-field splitting, optical and microwave power broadening, charge-state
dynamics, and instrument response. Exact diagonalization of a spin-1 model can
produce physically coupled transition motion, but it does not make the complete
emulator a quantitatively validated model of a particular apparatus.

These limitations and any later model mismatch must accompany results rather
than being removed from public-facing summaries.

## 14. Reproducibility contract

Every benchmark run will eventually record the resolved configuration, random
seed, software version, Git commit when available, scenario and estimator
parameters, raw and summary metrics, resource accounting, runtime, and generated
plots. Generated figures must be produced by scripts or CLI commands and must
not be edited manually.

No external large dataset is required for tests or CI. A tiny deterministic
synthetic fixture will support installation and smoke testing. The first
verified external anchor is Figshare DOI `10.6084/m9.figshare.28788437.v1`
(Liu Liu, CC BY 4.0): 4,693 complete sweeps over 311 frequency points from
2.740 GHz through 3.050 GHz in 1 MHz steps. Its detector units, timestamps,
current/field trajectory, and exact resonance truth are unresolved or absent,
so it supports playback, morphology checks, and semi-empirical residual work,
not photon calibration or exact tracking-error claims. Large third-party data
will not be silently downloaded or redistributed.
