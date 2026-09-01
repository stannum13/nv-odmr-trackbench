# Event-Driven Virtual Instrument Design

## Purpose

Build the causal measurement substrate for closed-loop ODMR experiments. The
instrument must turn hidden time-dependent eight-resonance truth into noisy
fluorescence observations, advance a virtual clock without sleeping, and
account for acquisition resources without leaking the noiseless signal to an
estimator.

This stage deliberately stops before fitting or tracking. It provides the
boundary that later full-sweep, two-point, sparse-linewidth, lock-in, and
physics-informed estimators must use. A small verified real-data layer is
implemented first so playback semantics and empirical-noise interfaces are
constrained by an actual ODMR file rather than invented in isolation.

## Real-data anchor

The first external anchor is Figshare DOI
`10.6084/m9.figshare.28788437.v1`, authored by Liu Liu and licensed CC BY 4.0.
The versioned file is a regular matrix of 4,693 stored sweeps by 311 frequency
points spanning 2.740 GHz through 3.050 GHz inclusive in 1 MHz steps. Its MD5
checksum is `df03ef2385cdd64d2f0e117ecd9d6c7e`.

The source description identifies an NV-ensemble ODMR acquisition under
magnetic fields produced by high current. The file does not contain per-sweep
timestamps, current, field, field direction, or an exact resonance trajectory.
Its header labels the data `counts/s`, while the named acquisition channel is
`/Dev1/AI0` and the stored values are approximately 3.17--3.45. The canonical
signal quantity is therefore `unknown_analog_signal` with unit status
`conflicted_unverified`; the loader must preserve both declared metadata and
this qualification without silently converting the values to photons or volts.

The repository will not redistribute or download the 19 MB source during
installation, tests, CI, or ordinary benchmark commands. The loader accepts an
explicit local path. A separate opt-in fetch command may download the exact
version into an ignored cache after displaying provenance, size, and license
and must verify its checksum. CI uses a tiny generated parser fixture with the
same structural conventions, not a claimed experimental excerpt.

Defensible initial uses are recorded-order full-sweep playback, parser and
fit stress testing, baseline and lineshape characterization, and extraction of
provenance-bearing empirical residual candidates. Exact center/FWHM/Q truth,
photon budgets, field calibration, exact timing, and counterfactual adaptive
queries are not available from this recording. Any fitted reference is an
offline oracle rather than ground truth. Any real fit plus synthetic dynamics
and prepared residuals is labeled semi-empirical simulation.

## Selected approach

Use a normalized-fluorescence instrument with an explicit nominal
off-resonance photon rate. The deterministic spectral model produces a
dimensionless fluorescence multiplier near unity. A noise strategy converts
that expected normalized signal and integration duration into a measured
observation.

This approach was selected over two alternatives:

1. Arbitrary fluorescence units plus a separate photon-budget estimator would
   accommodate more raw files directly but would make shot-noise and
   cross-estimator photon comparisons ambiguous.
2. A counts-only public interface would be maximally direct for photon-counting
   hardware but would make Gaussian controlled tests and normalized real-data
   playback unnecessarily awkward.

The selected boundary preserves raw counts when they exist while giving all
estimators one explicit normalized-fluorescence field.

Raw playback is intentionally separate: the Figshare adapter returns the
unmodified signal and unresolved unit metadata. A later explicit,
provenance-bearing normalization transform may create dimensionless input for
an estimator or residual model; loading never normalizes implicitly.

## Playback contract

`SweepDataset` stores the two-dimensional raw signal, frequency axis, sweep
indices, sample indices, declared metadata, and provenance. Its arrays are
read-only. Original row and column order is canonical and is never silently
sorted. With no measured timestamps, observations expose sequence indices and
`timestamp_s=None`.

An explicit nominal-timing transform may derive sample times using the declared
200 Hz clock. These times are labeled inferred and do not establish uninterrupted
acquisition, integration duration, inter-sweep dead time, or temporal bandwidth.
The playback iterator yields only already-reached observations and cannot expose
future rows, offline fit results, or a future scan boundary to an estimator.

## Package boundaries

```text
src/odmr_bench/
├── datasets/
│   ├── models.py        # immutable raw sweep data and provenance
│   ├── registry.py      # checked versioned metadata
│   └── figshare.py      # explicit-path loader and opt-in fetch
├── dynamics/
│   ├── base.py          # hidden spectral snapshot and dynamics protocol
│   └── center_drift.py  # stationary and linear center drift
└── emulator/
    ├── observations.py  # full and estimator-safe immutable records
    ├── noise.py         # Gaussian, Poisson, empirical residual strategies
    ├── resources.py     # immutable snapshots and mutable internal ledger
    └── instrument.py    # query validation, virtual clock, orchestration
```

Dynamics own hidden time-varying parameters. Noise owns stochastic sampling.
The instrument owns virtual-time sequencing and the random-number generator.
The resource ledger owns acquisition accounting. None of these modules owns an
estimator.

## Hidden dynamics contract

A `SpectralDynamics` protocol exposes only to the instrument:

```python
def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot: ...
```

`SpectralSnapshot` contains a validated `Baseline` and exactly eight
`Resonance` objects with unique stable IDs and positive absolute centers. It is
hidden truth and is never passed to estimators.

The initial implementations are:

- `StationaryDynamics`, returning one immutable snapshot; and
- `LinearCenterDrift`, applying a configured `center_slew_hz_per_s` to each
  parent ID while preserving widths, amplitudes, eta values, and baseline.

The drift model accepts either one common slew or a complete ID-to-slew map.
Missing or extra IDs fail during construction. Later center steps, linewidth
drift, contrast loss, collision, and Hamiltonian dynamics implement the same
protocol without changing the instrument.

## Query and clock semantics

The public query remains:

```python
instrument.query(frequency_hz: float, integration_time_s: float)
```

Both arguments must be finite and positive. A configured
`frequency_overhead_s` must be finite and non-negative.

For a query issued at virtual time \(t\):

1. frequency-setting overhead advances the clock over
   \([t,t+t_{\mathrm{overhead}}]\);
2. fluorescence integrates over the following interval of duration
   \(\Delta t\);
3. the expected spectrum is evaluated at the integration midpoint;
4. the observation timestamp is the end of integration, when the observation
   becomes available; and
5. the virtual clock equals that timestamp when the query returns.

With zero overhead, this reduces to the scientific specification's default
interval \([t,t+\Delta t]\). Midpoint evaluation is an explicit approximation,
stored as acquisition metadata. A future quadrature strategy may replace it
without altering noise or estimator interfaces.

No code path calls `sleep()` or derives dynamics from wall-clock time.

## Photon and fluorescence semantics

The spectrum output \(s(f,t)\) is a normalized expected fluorescence
multiplier. With nominal off-resonance photon rate \(R_0>0\), the instantaneous
expected detected rate is

\[
R(f,t)=R_0 s(f,t).
\]

The midpoint approximation gives

\[
\lambda=R_0 s(f,t_{\mathrm{mid}})\Delta t
\]

expected photons for one query. The instrument rejects non-finite or negative
expected normalized fluorescence because a negative Poisson intensity is not a
valid observation model. Zero expected fluorescence is allowed.

The nominal exposure \(R_0\Delta t\) may be public because it is known before
the hidden spectrum is evaluated. Signal-conditioned \(\lambda\) remains
evaluator-only.

## Noise strategies

Every strategy receives expected normalized fluorescence, nominal photon rate,
integration duration, and an explicit NumPy random generator. It returns a
measured normalized fluorescence and an optional realized integer photon count.

### Poisson shot noise

Sample

\[
n\sim\operatorname{Poisson}(\lambda)
\]

and return

\[
y=n/(R_0\Delta t).
\]

The integer `realized_photons` is retained. The exact signal-conditioned
`expected_photons` is stored only in the full instrument record and resource
ledger.

### Ideal Gaussian noise

Configure `stddev_at_1s`, in normalized-fluorescence units at one second. The
per-query standard deviation is

\[
\sigma(\Delta t)=\frac{\sigma_{1\mathrm{s}}}{\sqrt{\Delta t}}.
\]

This controlled model returns no realized photon count. Expected photons are
still tracked as an acquisition-budget quantity and must not be misdescribed as
the generative distribution for Gaussian noise.

### Empirical residual noise

Accept a finite, non-empty residual array supplied by the caller. `replay`
mode cycles causally through the stored order. `sample` mode draws indices from
the explicit seeded generator. Residuals are added in normalized-fluorescence
units.

Residual preparation is a separate, provenance-bearing transformation. A
residual artifact records the source dataset/version, sweep selection, fitted
reference method, normalization, and correlation treatment. Initial support
replays complete supplied residual sequences and contiguous blocks; independent
sampling remains available only when explicitly selected. The Figshare source
can ground residual morphology, but its unresolved units prevent photon-noise
calibration or claims that the residuals are pure detector noise.

## Observation privacy boundary

The immutable full `InstrumentObservation` contains:

```text
sequence_index
timestamp_s
frequency_hz
fluorescence
integration_time_s
nominal_exposure_photons
expected_photons
realized_photons (optional)
sampling_rule
```

`InstrumentObservation.estimator_view()` constructs a separate immutable
`EstimatorObservation` containing only:

```text
sequence_index
timestamp_s
frequency_hz
fluorescence
integration_time_s
nominal_exposure_photons
realized_photons (optional)
```

It has no `expected_photons`, hidden snapshot, noiseless fluorescence, dynamics
object, or future-randomness reference. Tests must prove the field is absent,
not merely set to `None`.

## Resource accounting

After each successful query, the ledger accumulates:

- observation count;
- total integration time;
- total nominal exposure photons;
- total signal-conditioned expected photons;
- total realized photons for count-producing observations;
- number of observations without realized counts; and
- virtual elapsed time including frequency overhead.

The public `ResourceSnapshot` is immutable. Ledger mutation occurs only after
validation and noise sampling succeed, so a failed query consumes no virtual
time or resource budget.

Estimator compute time is outside the instrument and will be added by the
benchmark runner.

## Determinism and seed ownership

The instrument constructor accepts a seed or explicit NumPy generator and owns
that generator for the run. Dynamics are deterministic functions of virtual
time in this stage. Identical initial state, configuration, query sequence, and
seed produce identical observations and resource snapshots.

The generator itself never appears in an estimator-facing object.

## Validation and failure behavior

Construction fails for:

- non-positive nominal photon rate;
- negative or non-finite overhead;
- snapshots other than exactly eight unique parent resonances;
- non-positive physical resonance centers;
- inconsistent drift-ID maps;
- invalid Gaussian standard deviation; or
- empty/non-finite empirical residual arrays.

Queries fail without advancing state for non-positive/non-finite frequency or
integration duration, non-finite spectral output, negative expected
fluorescence, or invalid noise results.

Noise strategies must reject non-finite returned fluorescence and negative or
non-integral realized counts. Poisson overflow or invalid rate errors are
surfaced with contextual `ValueError` rather than silently clipped.

## Verification strategy

Tests will prove:

- the Figshare-format loader preserves a generated fixture's matrix and order;
- verified registry metadata encodes the DOI, version, shape, grid, license,
  checksum, and unresolved-unit/timing/ground-truth status;
- malformed shapes, grids, non-finite data, and mismatched checksums fail;
- playback exposes no future observation and nominal timestamps require an
  explicit assumption-bearing transform;

- stationary and per-ID linear drift snapshots are deterministic and preserve
  stable identity;
- physical snapshots reject negative centers and invalid cardinality;
- timestamps and sequence indices increase monotonically;
- overhead and integration advance virtual time exactly;
- midpoint truth is used for a known linear drift;
- no real sleep or wall-clock dependence occurs;
- Poisson counts and normalized fluorescence match a fixed seed;
- Gaussian variance scaling follows \(1/\sqrt{\Delta t}\);
- empirical replay order and seeded sampling are reproducible;
- identical seeds and queries yield identical records;
- expected photons never appear in the estimator view;
- failed queries do not mutate clock, RNG-observable sequence, or resources;
- resource arithmetic is exact within floating-point tolerances; and
- existing 39 spectral/package tests remain passing.

## Scope boundary and next dependency

This stage does not implement fitting, tracking, metrics, benchmark output
directories, or magnetic-field dynamics. It implements a verified real-file
adapter and causal playback substrate, but not claims derived from that file.
Its next consumer is a deterministic frequency-drift scenario exercised first by a
repeated full-sweep estimator and then by a calibrated two-point center tracker
with a periodic five-point local linewidth fit.
