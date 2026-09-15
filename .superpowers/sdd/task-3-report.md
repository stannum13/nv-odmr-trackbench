# Stage 6.4 Task 3 Implementer Report

## Scope

Extracted the private `_evaluate_bound_source_model` seam in
`two_point_calibration.py`. It consumes the frozen calibration source, evaluates
its baseline exactly once, adds an optional constant offset, and subtracts every
source resonance in immutable tuple order. Only the named target's center,
FWHM, and amplitude can vary. The existing scalar Stage 6.3 target-only helper
now delegates through the shared scalar core; it retains the prior arithmetic
and invalid nonmatching-index behavior exactly. No public export, source
validation, or pre-summed background was introduced.

## TDD Evidence

- RED: after creating `test_sparse_linewidth_source_model.py`,
  `.venv/bin/python -m pytest tests/estimators/test_two_point_calibration.py
  tests/estimators/test_sparse_linewidth_source_model.py -q` stopped during
  collection with the expected `ImportError`: `_evaluate_bound_source_model`
  was absent. The pre-existing calibration file alone remained green at
  `12 passed`.
- GREEN: implemented the smallest private shared fit core, the source-ID-bound
  wrapper, and scalar legacy delegation. The same two-file command passed
  `24 passed`; after adding the explicit legacy invalid-index differential it
  passed `27 passed in 1.14s`.
- The new tests independently evaluate Lorentzian and pseudo-Voigt source
  tuples with affine and quadratic baselines, scalar and vector frequencies,
  boundary frequencies, a reversed immutable tuple seam, one baseline call,
  no background pre-sum, target-only center/FWHM/amplitude replacement, and a
  constant offset. Calibration tests retain exact scalar-byte comparisons and
  boundary/exception coverage.

## Differential and Verification

```text
.venv/bin/python -m pytest \
  tests/estimators/test_two_point_calibration.py \
  tests/estimators/test_sparse_linewidth_source_model.py \
  tests/estimators/test_two_point_tracker.py -q
81 passed in 1.26s

.venv/bin/python -m pytest tests/estimators -q
890 passed in 24.69s

.venv/bin/python -m pytest tests/dynamics tests/models tests/evaluation tests/emulator -q
384 passed in 8.53s

.venv/bin/python -m pytest tests/datasets tests/test_cli_errors.py \
  tests/test_cli_playback.py tests/test_cli_simulate.py tests/test_package.py \
  tests/test_plot_script.py tests/test_plotting.py -q
46 passed in 4.24s

Full repository partition total: 1,320 passed.
Affected estimator/evaluator/emulator integration total: 1,274 passed.

.venv/bin/ruff check .
All checks passed!
```

The full suite was deliberately run in exhaustive disjoint repository
partitions because this terminal stops one streaming command after 30 seconds;
the direct all-tests command reached 79% before that host limit. The partition
commands cover every collected `tests/test_*.py` file and yielded the exact
full-suite total above.

## Diff and Scope Inspection

- `git diff --check` exited 0 with no output.
- The production diff adds only private helpers and no package export.
- `rg "sum\\(|fsum|background" src/odmr_bench/estimators/two_point_calibration.py`
  found no new pre-summed model path.
- Only the declared production/test/state/changelog files plus this required
  task report are staged for the task commit.

## Commit

Atomic commit created with message:
`refactor: share bound spectral source model`.

## Concerns

None within Task 3 scope. The helper is intentionally private and the next
sparse-estimator task owns its first production consumer.

## Review-Fix Pass

### Finding and RED

The Task 3 re-review found that `_evaluate_bound_source_model` accepted NaN,
both infinities, booleans, complex values, arrays, zero/negative FWHM, and
negative amplitude. Some values propagated NaNs, projected complex values with
a warning, or reached a scalar divide-by-zero path.

The focused RED command added scalar and vector witnesses for NaN and both
infinities on every supplied parameter, all four parameter names crossed with
boolean/complex/array inputs, and target FWHM/amplitude domain failures:

```text
.venv/bin/python -m pytest \
  tests/estimators/test_sparse_linewidth_source_model.py::test_bound_source_model_rejects_nonfinite_scalar_parameters_before_arithmetic \
  tests/estimators/test_sparse_linewidth_source_model.py::test_bound_source_model_rejects_non_scalar_parameter_types \
  tests/estimators/test_sparse_linewidth_source_model.py::test_bound_source_model_enforces_target_width_and_amplitude_domains -q
39 failed, 6 warnings
```

### GREEN and Final Verification

Added a private finite-real scalar canonicalizer only at the new source-bound
entry point. It canonicalizes supported NumPy scalar values to built-in floats,
rejects boolean/complex/array values and non-finite values before source lookup
or arithmetic, requires FWHM `> 0`, and requires amplitude `>= 0`. The shared
legacy core remains unvalidated so the Stage 6.3 target-only helper's prior
inputs, outputs, and exception behavior remain unchanged.

```text
Focused validation GREEN: 39 passed in 0.77s
Stage 6.3 calibration + sparse model + tracker: 120 passed in 1.32s
Estimator integration: 929 passed in 25.95s
Dynamics/models/evaluator/emulator integration: 384 passed in 8.65s
Remaining datasets/CLI/package/plotting suite: 46 passed in 4.40s
Full repository partition total: 1,359 passed
Affected integration total: 1,313 passed
.venv/bin/ruff check .: All checks passed!
git diff --check: clean
```

The terminal cannot retain a process session after its 30-second stream limit,
so the full suite again uses exhaustive disjoint repository partitions rather
than claiming a truncated single-command run.
