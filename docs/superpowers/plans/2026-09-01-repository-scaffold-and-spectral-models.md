# Repository Scaffold and Spectral Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable, CI-tested Python 3.11 package with explicit-FWHM Lorentzian, Gaussian, pseudo-Voigt, baseline, Q, and eight-resonance spectrum models plus a reproducible demonstration plot.

**Architecture:** Keep deterministic spectral physics under `src/odmr_bench/models/`; stochastic observation noise remains outside these modules for the later virtual-instrument stage. Immutable parameter objects validate physical invariants at construction, while vectorized functions accept NumPy-compatible frequency arrays. A YAML-driven script imports the same tested package code used by future benchmarks and writes an unedited figure artifact.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pandas, Matplotlib, PyYAML, pytest, Ruff, Hatchling, GitHub Actions.

## Global Constraints

- The distribution name is `nv-odmr-trackbench`; the import package is `odmr_bench`.
- Internal frequency and linewidth units are Hz; internal time units are seconds.
- Every linewidth-facing API uses a field or argument named `fwhm_hz`; HWHM and Gaussian sigma appear only in named conversion helpers.
- Q is exactly `center_hz / fwhm_hz` and must not be presented as magnetometric sensitivity.
- Spectrum functions are deterministic; random noise belongs to the later instrument/noise boundary.
- Random behavior introduced in future stages must use explicit deterministic random generators.
- Scientifically important code lives under `src/odmr_bench/`; scripts only orchestrate package APIs.
- Generated figures are never edited manually and are ignored by Git unless a later publication stage explicitly selects them.
- No external dataset or network access is required by tests or CI.
- Use one clean commit for the repository-scaffold stage and one clean commit for the spectral-model stage; push each completed stage to `origin/main`.

## Planned file structure

```text
.
├── .github/workflows/ci.yml       # Python 3.11/3.12 lint and test workflow
├── .gitignore                     # Python, environment, run, and figure artifacts
├── CITATION.cff                   # project citation metadata without fabricated authors
├── LICENSE                        # MIT license
├── pyproject.toml                 # package, dependencies, CLI, pytest, Ruff
├── configs/static.yaml            # deterministic eight-resonance plot configuration
├── scripts/plot_spectrum.py       # config-driven generated demonstration figure
├── src/odmr_bench/__init__.py     # public version
├── src/odmr_bench/cli.py          # initial `odmrbench --version` entry point
├── src/odmr_bench/models/__init__.py
├── src/odmr_bench/models/lineshapes.py
├── src/odmr_bench/models/parameters.py
├── src/odmr_bench/models/spectrum.py
├── tests/test_package.py
├── tests/models/test_lineshapes.py
├── tests/models/test_parameters.py
├── tests/models/test_spectrum.py
└── tests/test_plot_script.py
```

The three model files each have one responsibility: normalized functions and
conversions, validated immutable parameters, and multi-resonance composition.
This prevents later fitting and instrument code from becoming coupled to plot
or configuration concerns.

## Explicitly deferred to later stage plans

This plan implements the deterministic spectral foundation only. The virtual
clock, Gaussian/Poisson/empirical noise strategies, dynamic disturbances,
restricted estimator observation view, resource ledger, replay adapters,
offline and realtime estimators, matched-budget runner, metrics, Hamiltonian,
datasets, and publication results remain governed by `docs/scientific_spec.md`
and will be planned at their dependency boundaries. No test or README text in
this plan may imply that those capabilities already exist.

---

### Task 1: Stage 1 — Installable repository scaffold and CI

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `CITATION.cff`
- Create: `.github/workflows/ci.yml`
- Create: `src/odmr_bench/__init__.py`
- Create: `src/odmr_bench/cli.py`
- Create: `tests/test_package.py`
- Modify: `README.md`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: Python 3.11+ and the Stage 0 scientific specification.
- Produces: importable `odmr_bench.__version__: str`, console entry point `odmrbench`, and standard commands `pytest` and `ruff check .` used by every later task.

- [ ] **Step 1: Write the failing package and CLI tests**

Create `tests/test_package.py` with this complete content:

```python
from __future__ import annotations

import subprocess
import sys

import odmr_bench


def test_package_exposes_version() -> None:
    assert odmr_bench.__version__ == "0.1.0"


def test_cli_reports_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "odmr_bench.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "odmrbench 0.1.0"
```

- [ ] **Step 2: Run the test to verify the package does not exist yet**

Run:

```bash
pytest tests/test_package.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'odmr_bench'`.

- [ ] **Step 3: Create package metadata and the minimal CLI**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "nv-odmr-trackbench"
version = "0.1.0"
description = "A reproducible benchmark for realtime NV-center ODMR resonance tracking."
readme = "README.md"
requires-python = ">=3.11"
license = {file = "LICENSE"}
keywords = ["NV center", "ODMR", "realtime tracking", "benchmark"]
classifiers = [
  "Development Status :: 2 - Pre-Alpha",
  "License :: OSI Approved :: MIT License",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Physics",
]
dependencies = [
  "matplotlib>=3.8",
  "numpy>=1.26",
  "pandas>=2.1",
  "pyyaml>=6.0",
  "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
  "build>=1.2",
  "pytest>=8.0",
  "ruff>=0.6",
]

[project.scripts]
odmrbench = "odmr_bench.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/odmr_bench"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]
```

Create `src/odmr_bench/__init__.py`:

```python
"""Realtime NV-ensemble ODMR estimation benchmark."""

__version__ = "0.1.0"
```

Create `src/odmr_bench/cli.py`:

```python
"""Command-line entry point for the ODMR benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from odmr_bench import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odmrbench")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
.venv/
venv/
dist/
build/
artifacts/
runs/
*.ipynb_checkpoints/
.DS_Store
```

Create `LICENSE`:

```text
MIT License

Copyright (c) 2026 NV ODMR TrackBench contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Create `CITATION.cff`:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it using this metadata."
title: "NV ODMR TrackBench"
type: software
authors:
  - alias: "stannum13"
version: 0.1.0
license: MIT
repository-code: "https://github.com/stannum13/nv-odmr-trackbench"
```

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest
```

- [ ] **Step 4: Install the editable package and verify the tests pass**

Run:

```bash
python -m pip install -e ".[dev]"
pytest tests/test_package.py -v
odmrbench --version
```

Expected: two tests pass and the command prints `odmrbench 0.1.0`.

- [ ] **Step 5: Run the initial lint and packaging checks**

Run:

```bash
ruff check .
python -m pip check
python -m build --wheel
```

Verify that exactly one wheel named like
`nv_odmr_trackbench-0.1.0-py3-none-any.whl` appears in `dist/`.
Expected: every command exits zero.

- [ ] **Step 6: Update the public status documents for Stage 1**

In `README.md`, replace the sentence beginning `The repository is currently at
Stage 0` with:

```markdown
The scientific contracts and fairness requirements are specified in
[docs/scientific_spec.md](docs/scientific_spec.md). The installable package and
CI scaffold are in place; executable spectral models are the current stage.
```

In `PROJECT_STATE.md`:

- set `Current stage` to `Stage 1 — Repository scaffold complete; Stage 2 — Spectral models in progress`;
- add the package scaffold, CLI, MIT license, citation metadata, and CI workflow to completed work;
- replace the test section with the exact pytest and Ruff counts observed;
- remove package metadata, tests, and CI from known software limitations; and
- make explicit-FWHM spectral functions the first next action.

In `CHANGELOG.md`, add these bullets under `Unreleased / Added`:

```markdown
- Installable Python 3.11+ package scaffold and `odmrbench` console entry point.
- MIT license, citation metadata, Ruff/pytest configuration, and GitHub Actions CI.
```

- [ ] **Step 7: Perform the Stage 1 review, commit, and push**

Run:

```bash
pytest
ruff check .
git diff --check
git diff --stat
git status --short
```

Review the full diff and confirm: no absolute paths, no secrets, no generated
wheel is staged, the CLI imports without optional runtime state, CI requires no
external data, and no README claim describes benchmark results. Then run:

```bash
git add .github .gitignore CITATION.cff LICENSE pyproject.toml src tests README.md PROJECT_STATE.md CHANGELOG.md
git commit -m "build: scaffold installable benchmark package"
git push origin main
```

Expected: the commit succeeds, the push updates `origin/main`, and `git status
--short --branch` reports a clean tracking branch.

---

### Task 2: Stage 2 — Explicit-FWHM spectral models and generated figure

**Files:**
- Create: `src/odmr_bench/models/__init__.py`
- Create: `src/odmr_bench/models/lineshapes.py`
- Create: `src/odmr_bench/models/parameters.py`
- Create: `src/odmr_bench/models/spectrum.py`
- Create: `tests/models/test_lineshapes.py`
- Create: `tests/models/test_parameters.py`
- Create: `tests/models/test_spectrum.py`
- Create: `tests/test_plot_script.py`
- Create: `configs/static.yaml`
- Create: `scripts/plot_spectrum.py`
- Modify: `README.md`
- Modify: `PROJECT_STATE.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: NumPy and PyYAML from Task 1; `odmr_bench` import package.
- Produces: `lorentzian`, `gaussian`, `pseudo_voigt`, FWHM conversion helpers, `q_factor`, immutable `Resonance` and `Baseline`, `multi_resonance_spectrum`, `configs/static.yaml`, and `scripts/plot_spectrum.py`.

- [ ] **Step 1: Write failing line-shape, FWHM, limiting-case, and Q tests**

Create `tests/models/test_lineshapes.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from odmr_bench.models.lineshapes import (
    fwhm_to_gaussian_sigma,
    fwhm_to_lorentzian_hwhm,
    gaussian,
    gaussian_sigma_to_fwhm,
    lorentzian,
    lorentzian_hwhm_to_fwhm,
    pseudo_voigt,
    q_factor,
)


@pytest.mark.parametrize("profile", [lorentzian, gaussian])
def test_profile_center_height_and_fwhm(profile) -> None:
    center_hz = 2.87e9
    fwhm_hz = 4.0e6
    frequency_hz = np.array(
        [center_hz - fwhm_hz / 2, center_hz, center_hz + fwhm_hz / 2]
    )
    assert_allclose(profile(frequency_hz, center_hz, fwhm_hz), [0.5, 1.0, 0.5])


def test_pseudo_voigt_limits_and_shared_fwhm() -> None:
    center_hz = 2.87e9
    fwhm_hz = 3.0e6
    frequency_hz = center_hz + np.linspace(-2.0, 2.0, 21) * fwhm_hz
    assert_allclose(
        pseudo_voigt(frequency_hz, center_hz, fwhm_hz, eta=0.0),
        gaussian(frequency_hz, center_hz, fwhm_hz),
    )
    assert_allclose(
        pseudo_voigt(frequency_hz, center_hz, fwhm_hz, eta=1.0),
        lorentzian(frequency_hz, center_hz, fwhm_hz),
    )
    half_max = pseudo_voigt(
        np.array([center_hz - fwhm_hz / 2, center_hz + fwhm_hz / 2]),
        center_hz,
        fwhm_hz,
        eta=0.37,
    )
    assert_allclose(half_max, [0.5, 0.5])


def test_named_linewidth_conversions_round_trip() -> None:
    fwhm_hz = np.array([1.0, 2.5e6, 7.0e6])
    assert_allclose(
        lorentzian_hwhm_to_fwhm(fwhm_to_lorentzian_hwhm(fwhm_hz)),
        fwhm_hz,
    )
    assert_allclose(
        gaussian_sigma_to_fwhm(fwhm_to_gaussian_sigma(fwhm_hz)),
        fwhm_hz,
    )


def test_q_factor_uses_center_divided_by_fwhm() -> None:
    assert_allclose(q_factor(2.87e9, 2.0e6), 1435.0)


@pytest.mark.parametrize("invalid_fwhm_hz", [0.0, -1.0, np.nan, np.inf])
def test_profiles_reject_invalid_fwhm(invalid_fwhm_hz: float) -> None:
    with pytest.raises(ValueError, match="fwhm_hz must be finite and positive"):
        lorentzian(np.array([2.87e9]), 2.87e9, invalid_fwhm_hz)


@pytest.mark.parametrize("invalid_eta", [-0.01, 1.01, np.nan])
def test_pseudo_voigt_rejects_invalid_eta(invalid_eta: float) -> None:
    with pytest.raises(ValueError, match="eta must be finite and within"):
        pseudo_voigt(np.array([2.87e9]), 2.87e9, 2.0e6, invalid_eta)
```

- [ ] **Step 2: Run the line-shape tests to verify the module is absent**

Run:

```bash
pytest tests/models/test_lineshapes.py -v
```

Expected: collection fails because `odmr_bench.models.lineshapes` does not
exist.

- [ ] **Step 3: Implement normalized line shapes and named conversions**

Create `src/odmr_bench/models/lineshapes.py`:

```python
"""Vectorized ODMR line shapes with explicit FWHM conventions."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _finite_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _positive_fwhm(fwhm_hz: ArrayLike) -> FloatArray:
    fwhm = np.asarray(fwhm_hz, dtype=np.float64)
    if not np.all(np.isfinite(fwhm)) or np.any(fwhm <= 0.0):
        raise ValueError("fwhm_hz must be finite and positive")
    return fwhm


def lorentzian(
    frequency_hz: ArrayLike,
    center_hz: ArrayLike,
    fwhm_hz: ArrayLike,
) -> FloatArray:
    """Return a unit-height Lorentzian whose width argument is FWHM."""
    frequency = _finite_array(frequency_hz, "frequency_hz")
    center = _finite_array(center_hz, "center_hz")
    fwhm = _positive_fwhm(fwhm_hz)
    reduced = (frequency - center) / fwhm
    return np.asarray(1.0 / (1.0 + 4.0 * reduced**2), dtype=np.float64)


def gaussian(
    frequency_hz: ArrayLike,
    center_hz: ArrayLike,
    fwhm_hz: ArrayLike,
) -> FloatArray:
    """Return a unit-height Gaussian whose width argument is FWHM."""
    frequency = _finite_array(frequency_hz, "frequency_hz")
    center = _finite_array(center_hz, "center_hz")
    fwhm = _positive_fwhm(fwhm_hz)
    reduced = (frequency - center) / fwhm
    return np.asarray(np.exp(-4.0 * np.log(2.0) * reduced**2), dtype=np.float64)


def pseudo_voigt(
    frequency_hz: ArrayLike,
    center_hz: ArrayLike,
    fwhm_hz: ArrayLike,
    eta: ArrayLike,
) -> FloatArray:
    """Return an FWHM-matched linear mixture of Lorentzian and Gaussian profiles."""
    mixture = np.asarray(eta, dtype=np.float64)
    if not np.all(np.isfinite(mixture)) or np.any((mixture < 0.0) | (mixture > 1.0)):
        raise ValueError("eta must be finite and within [0, 1]")
    return np.asarray(
        mixture * lorentzian(frequency_hz, center_hz, fwhm_hz)
        + (1.0 - mixture) * gaussian(frequency_hz, center_hz, fwhm_hz),
        dtype=np.float64,
    )


def fwhm_to_lorentzian_hwhm(fwhm_hz: ArrayLike) -> FloatArray:
    return np.asarray(_positive_fwhm(fwhm_hz) / 2.0, dtype=np.float64)


def lorentzian_hwhm_to_fwhm(hwhm_hz: ArrayLike) -> FloatArray:
    hwhm = _finite_array(hwhm_hz, "hwhm_hz")
    if np.any(hwhm <= 0.0):
        raise ValueError("hwhm_hz must be positive")
    return np.asarray(2.0 * hwhm, dtype=np.float64)


def fwhm_to_gaussian_sigma(fwhm_hz: ArrayLike) -> FloatArray:
    return np.asarray(
        _positive_fwhm(fwhm_hz) / (2.0 * np.sqrt(2.0 * np.log(2.0))),
        dtype=np.float64,
    )


def gaussian_sigma_to_fwhm(sigma_hz: ArrayLike) -> FloatArray:
    sigma = _finite_array(sigma_hz, "sigma_hz")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_hz must be positive")
    return np.asarray(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma, dtype=np.float64)


def q_factor(center_hz: ArrayLike, fwhm_hz: ArrayLike) -> FloatArray:
    center = _finite_array(center_hz, "center_hz")
    return np.asarray(center / _positive_fwhm(fwhm_hz), dtype=np.float64)
```

- [ ] **Step 4: Run the line-shape tests and lint the implementation**

Run:

```bash
pytest tests/models/test_lineshapes.py -v
ruff check src/odmr_bench/models/lineshapes.py tests/models/test_lineshapes.py
```

Expected: all parameterized cases pass and Ruff exits zero.

- [ ] **Step 5: Write failing immutable-parameter and spectrum-composition tests**

Create `tests/models/test_parameters.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from odmr_bench.models.parameters import Baseline, Resonance


def test_baseline_is_centered_at_reference_frequency() -> None:
    baseline = Baseline(
        intercept=1.0,
        slope_per_hz=2.0e-9,
        quadratic_per_hz2=3.0e-18,
        reference_hz=2.87e9,
    )
    offsets_hz = np.array([-2.0e6, 0.0, 2.0e6])
    expected = 1.0 + 2.0e-9 * offsets_hz + 3.0e-18 * offsets_hz**2
    assert_allclose(baseline.evaluate(2.87e9 + offsets_hz), expected)


def test_resonance_rejects_nonphysical_parameters() -> None:
    with pytest.raises(ValueError, match="fwhm_hz"):
        Resonance("r0", center_hz=2.87e9, fwhm_hz=0.0, amplitude=0.02, eta=0.5)
    with pytest.raises(ValueError, match="amplitude"):
        Resonance("r0", center_hz=2.87e9, fwhm_hz=2.0e6, amplitude=-0.01, eta=0.5)
    with pytest.raises(ValueError, match="eta"):
        Resonance("r0", center_hz=2.87e9, fwhm_hz=2.0e6, amplitude=0.02, eta=1.1)


def test_resonance_requires_a_stable_nonempty_id() -> None:
    with pytest.raises(ValueError, match="resonance_id"):
        Resonance("", center_hz=2.87e9, fwhm_hz=2.0e6, amplitude=0.02, eta=0.5)
```

Create `tests/models/test_spectrum.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from odmr_bench.models.lineshapes import pseudo_voigt
from odmr_bench.models.parameters import Baseline, Resonance
from odmr_bench.models.spectrum import multi_resonance_spectrum


def _eight_resonances() -> tuple[Resonance, ...]:
    centers_hz = 2.80e9 + np.arange(8) * 20.0e6
    return tuple(
        Resonance(
            resonance_id=f"r{index}",
            center_hz=float(center_hz),
            fwhm_hz=2.0e6,
            amplitude=0.01 + index * 0.001,
            eta=0.35,
        )
        for index, center_hz in enumerate(centers_hz)
    )


def test_eight_resonance_composition_matches_explicit_sum() -> None:
    frequency_hz = np.linspace(2.78e9, 2.96e9, 901)
    baseline = Baseline(intercept=1.0, reference_hz=2.87e9)
    resonances = _eight_resonances()
    expected = baseline.evaluate(frequency_hz)
    for resonance in resonances:
        expected -= resonance.amplitude * pseudo_voigt(
            frequency_hz,
            resonance.center_hz,
            resonance.fwhm_hz,
            resonance.eta,
        )
    assert_allclose(
        multi_resonance_spectrum(frequency_hz, resonances, baseline),
        expected,
    )


def test_isolated_dip_has_requested_amplitude_at_center() -> None:
    resonance = Resonance("r0", 2.87e9, 2.0e6, 0.025, 0.4)
    baseline = Baseline(intercept=1.2, reference_hz=2.87e9)
    value = multi_resonance_spectrum(
        np.array([resonance.center_hz]),
        [resonance],
        baseline,
    )
    assert_allclose(value, [1.175])


def test_explicit_additive_noise_is_applied_without_randomness() -> None:
    frequency_hz = np.array([2.86e9, 2.87e9, 2.88e9])
    baseline = Baseline(intercept=1.0, reference_hz=2.87e9)
    noise = np.array([0.01, -0.02, 0.03])
    clean = multi_resonance_spectrum(frequency_hz, [], baseline)
    noisy = multi_resonance_spectrum(
        frequency_hz,
        [],
        baseline,
        additive_noise=noise,
    )
    assert_allclose(noisy, clean + noise)


def test_duplicate_resonance_ids_are_rejected() -> None:
    resonance = Resonance("r0", 2.87e9, 2.0e6, 0.025, 0.4)
    with pytest.raises(ValueError, match="unique"):
        multi_resonance_spectrum(
            np.array([2.87e9]),
            [resonance, resonance],
            Baseline(intercept=1.0, reference_hz=2.87e9),
        )
```

- [ ] **Step 6: Run the new tests to verify parameter and spectrum modules are absent**

Run:

```bash
pytest tests/models/test_parameters.py tests/models/test_spectrum.py -v
```

Expected: collection fails because `odmr_bench.models.parameters` and
`odmr_bench.models.spectrum` do not exist.

- [ ] **Step 7: Implement immutable model parameters and deterministic composition**

Create `src/odmr_bench/models/parameters.py`:

```python
"""Validated immutable parameters for ODMR spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _require_finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class Resonance:
    resonance_id: str
    center_hz: float
    fwhm_hz: float
    amplitude: float
    eta: float

    def __post_init__(self) -> None:
        if not self.resonance_id.strip():
            raise ValueError("resonance_id must be nonempty")
        _require_finite(self.center_hz, "center_hz")
        _require_finite(self.fwhm_hz, "fwhm_hz")
        _require_finite(self.amplitude, "amplitude")
        _require_finite(self.eta, "eta")
        if self.fwhm_hz <= 0.0:
            raise ValueError("fwhm_hz must be positive")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class Baseline:
    intercept: float
    reference_hz: float
    slope_per_hz: float = 0.0
    quadratic_per_hz2: float = 0.0

    def __post_init__(self) -> None:
        _require_finite(self.intercept, "intercept")
        _require_finite(self.reference_hz, "reference_hz")
        _require_finite(self.slope_per_hz, "slope_per_hz")
        _require_finite(self.quadratic_per_hz2, "quadratic_per_hz2")

    def evaluate(self, frequency_hz: ArrayLike) -> NDArray[np.float64]:
        frequency = np.asarray(frequency_hz, dtype=np.float64)
        if not np.all(np.isfinite(frequency)):
            raise ValueError("frequency_hz must be finite")
        offset_hz = frequency - self.reference_hz
        return np.asarray(
            self.intercept
            + self.slope_per_hz * offset_hz
            + self.quadratic_per_hz2 * offset_hz**2,
            dtype=np.float64,
        )
```

Create `src/odmr_bench/models/spectrum.py`:

```python
"""Composition of deterministic multi-resonance ODMR spectra."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from odmr_bench.models.lineshapes import pseudo_voigt
from odmr_bench.models.parameters import Baseline, Resonance


def multi_resonance_spectrum(
    frequency_hz: ArrayLike,
    resonances: Sequence[Resonance],
    baseline: Baseline,
    *,
    additive_noise: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Evaluate baseline minus all dip components plus explicit noise."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if not np.all(np.isfinite(frequency)):
        raise ValueError("frequency_hz must be finite")
    resonance_ids = [resonance.resonance_id for resonance in resonances]
    if len(set(resonance_ids)) != len(resonance_ids):
        raise ValueError("resonance IDs must be unique")

    fluorescence = baseline.evaluate(frequency).copy()
    for resonance in resonances:
        fluorescence -= resonance.amplitude * pseudo_voigt(
            frequency,
            resonance.center_hz,
            resonance.fwhm_hz,
            resonance.eta,
        )

    if additive_noise is not None:
        noise = np.asarray(additive_noise, dtype=np.float64)
        if not np.all(np.isfinite(noise)):
            raise ValueError("additive_noise must be finite")
        try:
            fluorescence += np.broadcast_to(noise, fluorescence.shape)
        except ValueError as error:
            raise ValueError("additive_noise must broadcast to frequency_hz") from error
    return np.asarray(fluorescence, dtype=np.float64)
```

Create `src/odmr_bench/models/__init__.py`:

```python
"""Spectral models with explicit physical-unit conventions."""

from odmr_bench.models.lineshapes import gaussian, lorentzian, pseudo_voigt, q_factor
from odmr_bench.models.parameters import Baseline, Resonance
from odmr_bench.models.spectrum import multi_resonance_spectrum

__all__ = [
    "Baseline",
    "Resonance",
    "gaussian",
    "lorentzian",
    "multi_resonance_spectrum",
    "pseudo_voigt",
    "q_factor",
]
```

- [ ] **Step 8: Run all model tests and lint the model package**

Run:

```bash
pytest tests/models -v
ruff check src/odmr_bench/models tests/models
```

Expected: all model tests pass and Ruff exits zero.

- [ ] **Step 9: Write a failing generated-figure integration test**

Create `tests/test_plot_script.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_static_spectrum_plot_is_generated(tmp_path: Path) -> None:
    output_path = tmp_path / "spectrum.png"
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_spectrum.py",
            "--config",
            "configs/static.yaml",
            "--output",
            str(output_path),
        ],
        check=True,
    )
    assert output_path.is_file()
    assert output_path.stat().st_size > 10_000
```

- [ ] **Step 10: Run the figure test to verify the config and script are absent**

Run:

```bash
pytest tests/test_plot_script.py -v
```

Expected: the subprocess exits nonzero because `scripts/plot_spectrum.py` does
not exist.

- [ ] **Step 11: Create the deterministic eight-resonance config and plotting script**

Create `configs/static.yaml`:

```yaml
frequency:
  start_hz: 2790000000.0
  stop_hz: 2950000000.0
  points: 2401
baseline:
  intercept: 1.0
  reference_hz: 2870000000.0
  slope_per_hz: 1.0e-11
  quadratic_per_hz2: 0.0
resonances:
  - {resonance_id: r0, center_hz: 2805000000.0, fwhm_hz: 2500000.0, amplitude: 0.018, eta: 0.35}
  - {resonance_id: r1, center_hz: 2825000000.0, fwhm_hz: 2700000.0, amplitude: 0.021, eta: 0.40}
  - {resonance_id: r2, center_hz: 2845000000.0, fwhm_hz: 2900000.0, amplitude: 0.023, eta: 0.45}
  - {resonance_id: r3, center_hz: 2865000000.0, fwhm_hz: 3100000.0, amplitude: 0.025, eta: 0.50}
  - {resonance_id: r4, center_hz: 2875000000.0, fwhm_hz: 3100000.0, amplitude: 0.024, eta: 0.50}
  - {resonance_id: r5, center_hz: 2895000000.0, fwhm_hz: 2900000.0, amplitude: 0.022, eta: 0.45}
  - {resonance_id: r6, center_hz: 2915000000.0, fwhm_hz: 2700000.0, amplitude: 0.020, eta: 0.40}
  - {resonance_id: r7, center_hz: 2935000000.0, fwhm_hz: 2500000.0, amplitude: 0.017, eta: 0.35}
```

Create `scripts/plot_spectrum.py`:

```python
"""Generate the deterministic eight-dip spectrum demonstration figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from odmr_bench.models import (
    Baseline,
    Resonance,
    multi_resonance_spectrum,
    pseudo_voigt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    frequency_config = config["frequency"]
    frequency_hz = np.linspace(
        float(frequency_config["start_hz"]),
        float(frequency_config["stop_hz"]),
        int(frequency_config["points"]),
    )
    baseline = Baseline(**config["baseline"])
    resonances = tuple(Resonance(**item) for item in config["resonances"])
    if len(resonances) != 8:
        raise ValueError("static demonstration requires exactly eight resonances")

    fluorescence = multi_resonance_spectrum(frequency_hz, resonances, baseline)
    baseline_values = baseline.evaluate(frequency_hz)

    figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    for resonance in resonances:
        component = baseline_values - resonance.amplitude * pseudo_voigt(
            frequency_hz,
            resonance.center_hz,
            resonance.fwhm_hz,
            resonance.eta,
        )
        axis.plot(frequency_hz / 1.0e9, component, color="0.78", linewidth=0.8)
    axis.plot(
        frequency_hz / 1.0e9,
        fluorescence,
        color="#1f5a94",
        linewidth=1.8,
        label="eight-resonance spectrum",
    )
    axis.plot(
        frequency_hz / 1.0e9,
        baseline_values,
        color="#b24a33",
        linestyle="--",
        linewidth=1.1,
        label="baseline",
    )
    axis.set(
        xlabel="Microwave frequency (GHz)",
        ylabel="Normalized fluorescence",
        title="Synthetic NV-ensemble ODMR spectrum",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 12: Generate and inspect the figure without editing it**

Run:

```bash
pytest tests/test_plot_script.py -v
python scripts/plot_spectrum.py --config configs/static.yaml --output artifacts/spectrum_demo.png
```

Expected: the test passes and `artifacts/spectrum_demo.png` exists. Inspect the
image using the local image-viewing tool. Confirm that eight distinct minima,
the linear baseline, labeled GHz axis, and component curves are visible; fix
only the configuration or plotting code if they are not.

- [ ] **Step 13: Run fresh scientific invariants and the complete quality suite**

Run:

```bash
pytest
ruff check .
python -m pip check
git diff --check
```

Expected: every test passes, Ruff and dependency checks exit zero, and the diff
check is silent. Confirm from test output that all of these are exercised:

- Lorentzian and Gaussian equal one at center and one half at ±FWHM/2;
- pseudo-Voigt limiting cases exactly recover the component profiles;
- all named width conversions round-trip;
- Q equals center/FWHM;
- amplitudes are isolated dip depths;
- exactly eight stable parent IDs compose correctly in the fixture;
- invalid widths, amplitudes, eta values, and duplicate IDs fail loudly; and
- the plot is regenerated from configuration and package code.

- [ ] **Step 14: Update Stage 2 documentation and state**

Add this section to `README.md` after `Planned first milestone`:

````markdown
## Synthetic spectrum demonstration

After installing the package, generate the deterministic eight-resonance
spectrum with:

```bash
python scripts/plot_spectrum.py \
  --config configs/static.yaml \
  --output artifacts/spectrum_demo.png
```

The line-shape APIs use FWHM in Hz explicitly. The generated image is an
illustrative synthetic fixture, not experimental data or a benchmark result.
````

In `PROJECT_STATE.md`:

- set `Current stage` to `Stage 2 — Spectral models complete; next: virtual NV instrument`;
- add normalized line shapes, conversions, validated parameters, eight-dip composition, and the generated plot to completed work;
- record the exact current pytest count and Ruff result;
- remove the missing spectral model from scientific limitations while retaining all unimplemented noise, dynamics, Hamiltonian, estimator, and real-data limitations; and
- make the event-driven instrument, virtual clock, Poisson/Gaussian noise, and linear drift scenario the next actions.

Add these bullets to `CHANGELOG.md` under `Unreleased / Added`:

```markdown
- Vectorized Lorentzian, Gaussian, and FWHM-matched pseudo-Voigt profiles.
- Explicit Lorentzian HWHM and Gaussian sigma conversion helpers with tests.
- Validated baseline and resonance parameter objects, Q calculation, and deterministic multi-resonance composition.
- YAML-driven script for generating the synthetic eight-resonance demonstration plot.
```

- [ ] **Step 15: Inspect, review, commit, and push Stage 2**

Run:

```bash
git status --short
git diff --stat
git diff
```

Perform the scientific review against `docs/scientific_spec.md`: formulas use
the same FWHM convention; the pseudo-Voigt mixture has shared half-maximum
points; Q has no sensitivity claim; baseline powers use frequency relative to
`reference_hz`; noise is explicit and deterministic; parent IDs are not inferred
from sorting. Perform the software review: public names use units; validation is
centralized; plotting imports package code; generated artifacts and build output
remain ignored; tests do not require network access.

Run the final verification after all documentation edits:

```bash
pytest
ruff check .
git diff --check
```

Only if all commands exit zero, run:

```bash
git add configs scripts src/odmr_bench/models tests/models tests/test_plot_script.py README.md PROJECT_STATE.md CHANGELOG.md
git commit -m "feat: add explicit-FWHM spectral models"
git push origin main
git status --short --branch
```

Expected: the stage commit and push succeed, and the final status reports a clean
`main...origin/main` tracking branch.
