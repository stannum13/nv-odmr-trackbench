# Playback and Virtual Instrument Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a causal proof-of-concept substrate that can replay a verified
real ODMR sweep file and query a deterministic eight-resonance virtual
instrument with auditable virtual time and acquisition resources.

**Architecture:** Keep recorded playback and closed-loop emulation as separate
protocols sharing small immutable observation records. The real-data adapter
preserves raw order and uncertain units; the emulator owns hidden dynamics,
noise sampling, virtual time, and resource accounting. CLI smoke commands expose
both paths without introducing estimators or unsupported experimental claims.

**Tech Stack:** Python 3.11+, NumPy, PyYAML, argparse, pytest, Ruff.

## Global Constraints

- All scientifically important code lives under `src/odmr_bench/`.
- Recorded playback never exposes future observations or offline references.
- Closed-loop emulation advances virtual time and never calls `sleep()`.
- The Figshare signal remains `unknown_analog_signal` with
  `conflicted_unverified` units; it is never silently treated as photons,
  normalized fluorescence, or volts.
- Figshare DOI `10.6084/m9.figshare.28788437.v1` is optional, never downloaded
  implicitly, and is never required by tests or CI.
- External source checksum is MD5
  `df03ef2385cdd64d2f0e117ecd9d6c7e`; expected shape is `(4693, 311)` and the
  grid is 2.740 GHz through 3.050 GHz inclusive at 1 MHz spacing.
- Synthetic stochastic behavior is deterministic for identical seeds, configs,
  and query sequences.
- Hidden noiseless fluorescence and signal-conditioned expected photons never
  appear in estimator-facing observations.
- FWHM remains the only public linewidth convention and `Q = center / FWHM`.
- Production code is added test-first; each red test must be observed failing
  for the intended missing behavior before implementation.
- Each completed task updates `PROJECT_STATE.md` and `CHANGELOG.md`, runs focused
  tests plus the full `pytest` and `ruff` suites, inspects the diff, commits once,
  and pushes `main` to `origin` only after its task review is clean.

---

### Task 1: Verified Sweep Dataset and Causal Playback

**Files:**
- Create: `datasets/README.md`
- Create: `datasets/registry.yaml`
- Create: `src/odmr_bench/datasets/__init__.py`
- Create: `src/odmr_bench/datasets/models.py`
- Create: `src/odmr_bench/datasets/registry.py`
- Create: `src/odmr_bench/datasets/figshare.py`
- Create: `src/odmr_bench/datasets/playback.py`
- Create: `tests/datasets/test_registry.py`
- Create: `tests/datasets/test_figshare.py`
- Create: `tests/datasets/test_playback.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: `DatasetRecord`, `SweepDataset`, `PlaybackObservation`,
  `FIGSHARE_28788437_V1`, `parse_figshare_sweep_file`,
  `load_figshare_28788437`, and `iter_playback`.
- `SweepDataset.signal` has shape `(n_sweeps, n_frequencies)` and all NumPy
  arrays are copies marked read-only.
- `iter_playback(dataset, nominal_clock_hz=None)` yields flattened row-major
  observations with monotonic `sequence_index`; timestamps are `None` unless
  the caller explicitly supplies the nominal-clock assumption.

- [ ] **Step 1: Write failing registry and immutable-model tests**

  Add tests that require exact DOI/version/license/author/checksum/shape/grid,
  `signal_quantity == "unknown_analog_signal"`,
  `unit_status == "conflicted_unverified"`,
  `timing_status == "nominal_without_timestamps"`, and
  `ground_truth_status == "none"`. Construct a 2-by-3 `SweepDataset`, mutate
  the caller arrays afterward, and assert the stored arrays are unchanged and
  reject assignment. Assert non-2D, non-finite, mismatched-axis, and
  non-increasing-frequency inputs raise `ValueError`.

- [ ] **Step 2: Run the model tests and observe the missing-module failure**

  Run `pytest tests/datasets/test_registry.py -q` and confirm collection fails
  because `odmr_bench.datasets` does not exist.

- [ ] **Step 3: Implement registry records and immutable sweep data**

  Implement frozen slotted dataclasses. `DatasetRecord` contains stable ID,
  title, DOI, version, author tuple, license/SPDX URL, canonical URL, filename,
  download URL, byte size, checksum algorithm/value, shape, frequency start/
  stop/step/count, declared signal label, canonical signal quantity, unit
  status, detector channel, nominal clock, timing status, missing-metadata
  tuple, and ground-truth status. Add the exact verified Figshare record and a
  test that `datasets/registry.yaml` carries matching public metadata.

- [ ] **Step 4: Write failing parser tests using a generated tiny fixture**

  Generate a temporary text file with the same comment-header conventions and
  two rows by three tab-separated values. Test exact data/order preservation,
  parsed declared metadata, generated frequency axis, and failures for wrong
  row width, non-finite cells, and header/grid mismatch. Test that the verified
  loader rejects the tiny fixture because its checksum differs.

- [ ] **Step 5: Run parser tests and observe the missing parser failure**

  Run `pytest tests/datasets/test_figshare.py -q` and confirm failures name the
  absent parser/loader behavior.

- [ ] **Step 6: Implement the generic parser and verified loader**

  `parse_figshare_sweep_file(path, *, expected_shape=None)` parses only the
  documented header fields, loads finite floats, validates a regular matrix and
  inclusive frequency grid, and returns `SweepDataset` without normalization or
  sorting. `load_figshare_28788437(path)` verifies size, checksum, exact shape,
  and exact grid before returning. It performs no network operation.

- [ ] **Step 7: Write failing causal-playback tests**

  Assert row-major frequency/signal order, sequence indices `0..N-1`, sweep and
  sample indices, absent timestamps by default, explicit inferred timestamps
  `sequence_index / nominal_clock_hz`, and rejection of non-positive/non-finite
  clock assumptions. Confirm consuming one iterator item exposes no container
  or callback for future signal values.

- [ ] **Step 8: Implement playback and verify Task 1**

  Implement frozen `PlaybackObservation` and a generator-only `iter_playback`.
  Run `pytest tests/datasets -q`, `pytest -q`, `ruff check .`, and
  `git diff --check`; inspect scientific wording and commit.

---

### Task 2: Hidden Eight-Resonance Dynamics

**Files:**
- Create: `src/odmr_bench/dynamics/__init__.py`
- Create: `src/odmr_bench/dynamics/base.py`
- Create: `src/odmr_bench/dynamics/center_drift.py`
- Create: `tests/dynamics/test_snapshots.py`
- Create: `tests/dynamics/test_center_drift.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: existing immutable `Baseline` and `Resonance`.
- Produces: frozen `SpectralSnapshot`, runtime-checkable `SpectralDynamics`
  protocol, `StationaryDynamics`, and `LinearCenterDrift`.
- `snapshot_at(timestamp_s)` accepts finite non-negative virtual time and always
  returns exactly eight unique stable resonance IDs with positive centers.

- [ ] **Step 1: Write failing snapshot-invariant tests**

  Build eight valid resonances and assert snapshots reject 7/9 resonances,
  duplicate IDs, non-positive centers, and invalid timestamps while copying the
  resonance tuple immutably.

- [ ] **Step 2: Run the snapshot tests and observe the missing API failure**

  Run `pytest tests/dynamics/test_snapshots.py -q` and confirm the dynamics
  module is absent.

- [ ] **Step 3: Implement the snapshot and stationary dynamics**

  Add the protocol and validation helpers. `StationaryDynamics.snapshot_at`
  returns a new validated snapshot with the same immutable parameters for every
  valid timestamp.

- [ ] **Step 4: Write failing linear-drift tests**

  Assert common slew and complete ID-to-slew mapping move centers by
  `slew_hz_per_s * timestamp_s`, preserve all other fields and IDs, and are
  deterministic. Reject missing/extra IDs, bool/non-finite slew values, and any
  configuration that yields a non-positive center at the queried time.

- [ ] **Step 5: Implement linear drift and verify Task 2**

  Implement `LinearCenterDrift` without sorting by frequency. Run focused/full
  tests, Ruff, diff checks, inspect identity semantics, and commit.

---

### Task 3: Observation Noise and Resource Accounting

**Files:**
- Create: `src/odmr_bench/emulator/__init__.py`
- Create: `src/odmr_bench/emulator/noise.py`
- Create: `src/odmr_bench/emulator/observations.py`
- Create: `src/odmr_bench/emulator/resources.py`
- Create: `tests/emulator/test_noise.py`
- Create: `tests/emulator/test_observations.py`
- Create: `tests/emulator/test_resources.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: `NoiseResult`, `PoissonNoise`, `GaussianNoise`,
  `EmpiricalResidualNoise`, `InstrumentObservation`, `EstimatorObservation`,
  `ResourceLedger`, and immutable `ResourceSnapshot`.
- Every noise strategy implements
  `sample(expected_fluorescence, nominal_rate_hz, integration_time_s, rng)`.
- Empirical residual mode is exactly one of `replay`, `sample`, or `block`; block
  mode preserves contiguous residual order and wraps deterministically.

- [ ] **Step 1: Write and run failing noise tests**

  Test fixed-seed Poisson counts and normalization, Gaussian
  `1 / sqrt(integration_time_s)` scaling using identically seeded draws,
  empirical replay/wrap order, seeded independent samples, contiguous block
  order, and validation of rates/durations/residuals/modes. Run the focused test
  and confirm missing behavior.

- [ ] **Step 2: Implement minimal noise strategies**

  Return measured normalized fluorescence plus optional integral realized
  photons. Reject negative/non-finite expected fluorescence and invalid results;
  never clip. Keep residual provenance as an immutable mapping containing source
  ID, preparation label, normalization label, and correlation mode.

- [ ] **Step 3: Write and run failing privacy-boundary tests**

  Require full observations to store sequence, end timestamp, frequency,
  measured fluorescence, integration, nominal exposure, expected photons,
  optional realized photons, and sampling rule. Assert `estimator_view()` is a
  separate frozen object with no `expected_photons`, noiseless signal, snapshot,
  dynamics, or RNG field—even when inspected with `dataclasses.fields`.

- [ ] **Step 4: Implement observation records**

  Canonicalize finite scalar inputs, require non-negative sequence index and
  photon fields, and return an estimator-safe record containing only public
  acquisition metadata.

- [ ] **Step 5: Write and run failing resource-ledger tests**

  Assert exact accumulation of observations, integration time, nominal and
  expected photons, realized photons, observations without realized counts,
  and virtual elapsed time. Assert snapshots are immutable and invalid commits
  leave the ledger unchanged.

- [ ] **Step 6: Implement resources and verify Task 3**

  Keep mutation private to `ResourceLedger.record(...)`; validate all inputs
  before mutation. Run focused/full tests, Ruff, diff checks, privacy/scientific
  review, and commit.

---

### Task 4: Event-Driven Virtual Instrument

**Files:**
- Create: `src/odmr_bench/emulator/instrument.py`
- Create: `tests/emulator/test_instrument.py`
- Modify: `src/odmr_bench/emulator/__init__.py`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `SpectralDynamics`, `multi_resonance_spectrum`, noise strategies,
  observations, and resource ledger.
- Produces: `ODMRInstrument.query(frequency_hz, integration_time_s)` returning
  full `InstrumentObservation`, `ODMRInstrument.resources`, and
  `ODMRInstrument.virtual_time_s`.
- The constructor accepts dynamics, noise, positive nominal photon rate,
  non-negative frequency overhead, and exactly one of seed or explicit NumPy
  generator.

- [ ] **Step 1: Write failing clock and midpoint tests**

  Use a known linear drift and deterministic zero Gaussian noise. Assert a query
  advances overhead then integration, evaluates the spectrum at the integration
  midpoint, timestamps availability at integration end, increments sequence,
  and never calls wall-clock time or sleep.

- [ ] **Step 2: Run the focused test and observe the missing instrument failure**

  Run `pytest tests/emulator/test_instrument.py -q` and confirm failure is due to
  the absent instrument.

- [ ] **Step 3: Implement query orchestration**

  Validate all constructor/query values before use, obtain the hidden midpoint
  snapshot, evaluate one-point normalized fluorescence, calculate nominal and
  signal-conditioned expected photons, sample noise, construct the observation,
  then atomically advance clock/sequence/resources. Failed queries consume
  nothing and do not advance RNG-observable behavior.

- [ ] **Step 4: Add deterministic, privacy, and failure-atomicity tests**

  Assert identical seeds/queries reproduce observations/resources; different
  seeds change stochastic results; estimator views omit truth; Poisson realized
  counts agree with returned normalized fluorescence; invalid frequency,
  duration, negative spectrum, and invalid noise output preserve clock,
  sequence, resources, and the next valid seeded result.

- [ ] **Step 5: Verify Task 4**

  Run emulator/dynamics/model tests, then full tests, Ruff, diff checks, inspect
  the estimator boundary and resource arithmetic, and commit.

---

### Task 5: Proof-of-Concept Playback and Emulation CLI

**Files:**
- Create: `configs/drift.yaml`
- Create: `docs/emulator.md`
- Create: `docs/datasets.md`
- Create: `examples/playback_and_emulation.py`
- Create: `tests/test_cli_playback.py`
- Create: `tests/test_cli_simulate.py`
- Modify: `src/odmr_bench/cli.py`
- Modify: `README.md`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: `odmrbench dataset-info`,
  `odmrbench playback --path PATH [--max-observations N]`, and
  `odmrbench simulate --config configs/drift.yaml`.
- Commands print deterministic JSON summaries and never imply fitted tracking
  accuracy; estimators are the next stage.

- [ ] **Step 1: Write failing CLI tests**

  Require `dataset-info` to emit DOI/license/size/checksum and limitation flags;
  playback on a tiny generated fixture to emit observation count, sweep/point
  counts, signal min/max, unit/timing status, and no inferred timestamps by
  default; simulate to emit seed, query count, final virtual time, integration,
  nominal/expected/realized photon totals, and the first/last measured sample.

- [ ] **Step 2: Run CLI tests and observe parser-command failures**

  Run both focused CLI test files and confirm argparse rejects the missing
  subcommands.

- [ ] **Step 3: Implement commands and deterministic drift config**

  Parse YAML into eight explicit pseudo-Voigt resonances, baseline, linear slew,
  Poisson noise, overhead, seed, and a fixed query schedule. Keep configuration
  validation in package code rather than the example. JSON uses sorted keys and
  finite native numeric types.

- [ ] **Step 4: Add researcher-facing documentation and example**

  Document the explicit Figshare download URL, 19 MB size, checksum verification,
  CC BY attribution, absent metadata, and local-path playback. Explain which
  comparisons require synthetic emulation. The example runs a tiny in-memory
  playback followed by virtual queries without downloading external data.

- [ ] **Step 5: Perform stage verification and commit**

  Run `ruff check .`, `pytest -q`, `python -m build`, install the wheel into a
  clean temporary virtual environment, smoke-test all three CLI commands, run
  `git diff --check`, inspect the full stage diff, complete scientific and
  software-quality reviews, update state/changelog, and commit.

---

## Final Review Gate

After all task-scoped reviews are clean, create a review package for the full
range beginning at the design commit. A fresh senior reviewer checks causal
isolation, raw-unit honesty, checksum/registry correctness, virtual-time and
photon arithmetic, seed determinism, failure atomicity, public API clarity,
tests, packaging, docs, and claim discipline. One fixer handles all Critical or
Important findings, reruns covering/full verification, and receives re-review.
Only a clean reviewed range is pushed as the completed playback/instrument
stage.
