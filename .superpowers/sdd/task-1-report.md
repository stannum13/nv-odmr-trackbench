# Task 1 Report — Verified Sweep Dataset and Causal Playback

## Scope

Added the verified Figshare registry, immutable raw sweep model, local-only
parser/verifier, and causal row-major playback. The external source is neither
downloaded by the package nor redistributed. Its header label remains preserved
as `count data (counts/s)`, while its canonical quantity remains
`unknown_analog_signal` with `conflicted_unverified` units.

## TDD evidence

Observed RED cycles:

1. `pytest tests/datasets/test_registry.py -q` failed during collection with
   `ModuleNotFoundError: No module named 'odmr_bench'`. The host `pytest`
   installation does not have this `src`-layout package installed; this still
   established that the requested dataset package was absent before creation.
2. `PYTHONPATH=src pytest tests/datasets/test_figshare.py -q` failed during
   collection because `load_figshare_28788437` and
   `parse_figshare_sweep_file` were absent from `odmr_bench.datasets`.
3. `PYTHONPATH=src pytest tests/datasets/test_playback.py -q` failed during
   collection because `iter_playback` was absent from `odmr_bench.datasets`.

Each test group was then run green after the smallest corresponding
implementation. The final focused suite is `20 passed`.

## Verification and review

Final commands and results:

```console
.venv/bin/python -m pytest tests/datasets -q  # 20 passed
.venv/bin/python -m pytest -q                 # 59 passed
.venv/bin/ruff check .                        # All checks passed!
git diff --cached --check                      # clean
```

Scientific self-review confirms that code and documentation do not call the raw
values photons, volts, normalized fluorescence, experimental truth, or measured
timing. The nominal 200 Hz field is only a caller-opted inferred-time transform.
The code-quality review found immutable copied NumPy arrays, a local-path-only
checksum verifier, no network operation, no normalization/sorting, and no
future-data object or callback on a playback observation.

## Commit

`a4e2fd7 feat: add verified sweep playback`

## Concern

The bare host `pytest` command lacks the editable package installation in this
workspace. Verification uses the repository's `.venv/bin/python -m pytest`,
which has the package installed; this is an environment issue, not a package
test failure.

---

## Review-fix report (2026-09-03)

### Scope and resolution

Resolved every Task 1 review finding without downloading or redistributing the
external Figshare file.

- **C1:** The original `iter_playback` generator was deficient as an
  estimator-facing boundary because its inspectable frame retained `dataset`
  and the current row. Estimator execution now uses
  `run_playback(dataset, on_observation, nominal_clock_hz=None)`, whose normal
  callback API supplies exactly one frozen `PlaybackObservation` at a time.
  The renamed `iter_playback_for_analysis` remains explicitly trusted
  evaluator-only tooling and is not exported as a causal iterator. This is an
  API-boundary guarantee, not a sandbox against malicious Python stack
  introspection; adversarial estimators require process isolation.
- **I1:** `load_verified_sweep_file` reads one immutable byte snapshot, checks
  its digest and length, and parses that same byte sequence. The fixed Figshare
  loader delegates to it with the checked record.
- **I2:** YAML parity now compares every `DatasetRecord` field, including the
  scientifically critical signal, unit, timing, missing-metadata, and
  ground-truth classifications.
- **I3:** A generated tiny fixture and injected expected record cover the
  verified loader's deterministic success path, including size/digest, exact
  shape/grid, parsing of the verified bytes despite later path replacement, and
  provenance attachment.
- **M1:** `SweepDataset` now rejects zero sweeps and zero frequency samples.

### Test-first evidence

Observed RED before production changes:

```console
.venv/bin/python -m pytest tests/datasets/test_figshare.py tests/datasets/test_playback.py -q
# collection errors: load_verified_sweep_file and iter_playback_for_analysis
# were absent from odmr_bench.datasets

.venv/bin/python -m pytest tests/datasets/test_registry.py -q
# 2 failed: empty (0, 3) and (2, 0) SweepDataset inputs did not raise ValueError
```

The expanded YAML parity assertions were green immediately because the checked
YAML already matched the immutable record; they close the review's future-drift
coverage gap rather than change registry data.

Observed GREEN after the minimal implementation:

```console
.venv/bin/python -m pytest tests/datasets -q
# 24 passed

.venv/bin/python -m pytest -q
# 63 passed

.venv/bin/ruff check .
# All checks passed!

git diff --check
# clean
```

An intermediate focused Ruff run reported only an import-order violation in the
new Figshare test; it was corrected before the final clean Ruff gate.

### Documentation and self-review

Updated the virtual-instrument design, implementation plan, scientific
specification, project state, changelog, and dataset README to record the
generator-frame deficiency, runner alternative, trusted-analysis tradeoff, and
process-isolation limitation. Self-review confirmed no estimator-facing API
exports `iter_playback`, while `run_playback` only calls the supplied callback
with `PlaybackObservation` values.

### Commit

Included in the local `fix: harden verified playback boundary` commit.
