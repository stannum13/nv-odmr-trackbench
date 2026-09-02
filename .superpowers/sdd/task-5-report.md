# Task 5 Report — Proof-of-Concept Playback and Emulation CLI

## Scope delivered

- Added deterministic, sorted JSON commands:
  - `odmrbench dataset-info`
  - `odmrbench playback --path PATH [--max-observations N]`
  - `odmrbench simulate --config configs/drift.yaml`
- Added validated YAML loading in `odmr_bench.cli`, creating eight explicit
  pseudo-Voigt resonances, a reference-centered baseline, common linear center
  slew, seeded Poisson noise, overhead, and a non-empty fixed query schedule.
- Added `configs/drift.yaml`, raw-data and emulator documentation, an
  in-memory/no-download example, README guidance, and stage state/changelog
  updates.

## TDD record

### RED

Before the CLI implementation, ran:

```text
.venv/bin/python -m pytest tests/test_cli_playback.py tests/test_cli_simulate.py -q
```

Result: `4 failed`. Each failure was argparse rejecting a missing command:
`dataset-info`, `playback`, or `simulate`.

During scientific wording review, changed the playback test to require the
repository's canonical raw-data statuses. Ran:

```text
.venv/bin/python -m pytest tests/test_cli_playback.py -q
```

Result: `1 failed, 1 passed`; the implementation still emitted the former
non-canonical labels, proving the assertion exercised the intended behavior.

### GREEN

After implementation and the terminology correction, ran:

```text
.venv/bin/python -m pytest tests/test_cli_playback.py tests/test_cli_simulate.py -q
```

Result: `4 passed`.

## Final verification

```text
.venv/bin/python -m pytest -q              # 187 passed
.venv/bin/ruff check .                      # All checks passed
.venv/bin/python -m build                   # sdist and wheel built
git diff --check                            # clean
```

Installed the rebuilt wheel into a new temporary virtual environment and ran
all three commands successfully. The smoke outputs confirmed the checked
Figshare DOI/license/size/checksum, raw playback with `inferred_timestamps:
false` plus `conflicted_unverified` / `nominal_without_timestamps`, and the
fixed simulation's deterministic resource summary. The installed commands used
no implicit download or package-local configuration resource; `simulate`
receives its scenario through the explicit `--config` path.

## Scientific self-review

- Raw recorded signal remains explicitly `conflicted_unverified`; no command
  labels it as photons, volts, or normalized fluorescence.
- Playback leaves timestamps uninferred; its nominal clock status does not make
  a timing or bandwidth claim.
- Simulation output is identified as synthetic emulation, not an estimator fit,
  tracking-accuracy result, realtime claim, or experimental agreement claim.
- The default scenario has exactly eight unique parent IDs and invokes the
  existing FWHM-based pseudo-Voigt spectrum path.
- Virtual time and photon totals come from `ODMRInstrument`/`ResourceLedger`.
  The default smoke's 8 queries produce `0.048 s = 8 * (0.001 + 0.005)`, total
  integration `0.040 s`, and nominal exposure `20,000` photons; expected and
  realized totals remain separately reported.

## Software-quality self-review

- YAML structure is validated in package code before the first query: exact
  root/baseline/noise/query/resonance keys, eight resonances, Poisson mode,
  integer seed, and non-empty schedule. Existing domain/instrument validation
  rejects invalid physical scalars and failed acquisitions remain atomic.
- JSON is emitted with sorted keys and `allow_nan=False`; summary values are
  native validated `float`/`int` fields.
- The local-file playback path uses the existing no-network parser and only
  retains the requested prefix for its summary.
- The wheel contains all imported package code and dependencies; configuration
  is intentionally an explicit caller-supplied file rather than an untracked
  runtime resource. The project-provided `configs/drift.yaml` was used in the
  installed-wheel smoke.
- No Critical or Important issue was found in this task-scoped self-review.

## Remaining concern / handoff

The plan's final fresh-senior review covers the full playback/instrument range
beginning at the design commit. This task has completed its scoped self-review
and local verification, but that range review and any push are intentionally
left to the parent-stage workflow.
