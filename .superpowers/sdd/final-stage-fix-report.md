# Final Stage Fix Report — Playback and Virtual Instrument

Date: 2026-09-03

## Scope addressed

- Made `PoissonNoise.sampling_rule` and `GaussianNoise.sampling_rule` fixed,
  non-constructor dataclass fields. Callers can no longer relabel either
  built-in sampling strategy, and full instrument observations record their
  exact fixed built-in rule.
- Corrected the public Figshare creator attribution to `Liu` in the dataset
  guide, scientific specification, and virtual-instrument design record.
  A registry-driven regression now checks each public attribution form against
  the checked creator string.
- Clarified that the external recording's metadata/checksum are verified while
  its file is neither bundled nor attached and it has no verified tracking
  truth.
- Clarified that the static spectrum plotting command is run from a source
  checkout after installation.

## TDD evidence

RED:

```text
.venv/bin/pytest \
  tests/emulator/test_noise.py::test_builtin_noise_constructors_cannot_relabel_their_sampling_rules \
  tests/emulator/test_instrument.py::test_builtin_noise_observation_records_the_exact_sampling_rule \
  tests/datasets/test_registry.py::test_public_dataset_docs_match_the_checked_creator_string -q

2 failed, 2 passed
```

The failures demonstrated that constructors accepted a contradictory
`sampling_rule` and that the first checked public document still used
`Liu Liu`. The observation checks already passed for normal built-in instances
and were retained as regression coverage for the evaluator-side provenance
record.

GREEN:

```text
4 passed
```

## Final verification

- Focused regressions: 4 passed.
- Full suite: `200 passed`.
- `.venv/bin/ruff check .`: all checks passed.
- `git diff --check`: clean.
- Build: fresh wheel and sdist built successfully with `python -m build`.
- Wheel contents: confirmed `odmr_bench/emulator/noise.py` and
  `odmr_bench/configs/drift.yaml` are present.
- Clean external-wheel smokes from an unrelated temporary directory:
  `odmrbench dataset-info`, explicit-path `odmrbench playback`, and
  `odmrbench simulate --config bundled:drift` all exited successfully.

## Scientific and software review

The provenance field now cannot contradict the built-in Poisson or Gaussian
formula at construction, and no estimator-visible fields were added. The
attribution correction follows the checked registry creator string without
changing DOI, version, license, checksum, raw-unit qualification, timing
qualification, or truth limitations. The revised project-state limitation no
longer contradicts verified metadata/checksum/header evidence.

## Known historical note

The Task 2 initial RED evidence remains a historical process-only limitation:
its first run failed from a source-layout import environment mismatch rather
than the intended missing dynamics behavior. This final fix does not rewrite
that record; later targeted RED/GREEN evidence and current runtime verification
remain the relevant implementation evidence.
