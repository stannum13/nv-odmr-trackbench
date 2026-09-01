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


def test_parameter_scalars_are_canonical_python_floats() -> None:
    source = np.array(2.87e9)
    resonance = Resonance(
        "r0",
        center_hz=source[()],
        fwhm_hz=np.int64(2_000_000),
        amplitude=np.float32(0.02),
        eta=0,
    )
    baseline = Baseline(
        intercept=np.float64(1.0),
        reference_hz=source[()],
        slope_per_hz=np.int32(0),
    )

    source[...] = 1.0

    assert resonance.center_hz == 2.87e9
    assert baseline.reference_hz == 2.87e9
    for value in (
        resonance.center_hz,
        resonance.fwhm_hz,
        resonance.amplitude,
        resonance.eta,
        baseline.intercept,
        baseline.reference_hz,
        baseline.slope_per_hz,
        baseline.quadratic_per_hz2,
    ):
        assert type(value) is float


@pytest.mark.parametrize(
    "invalid",
    [True, 1.0 + 0.0j, np.array([1.0]), np.array([[1.0]])],
)
def test_parameter_fields_reject_non_real_scalars(invalid: object) -> None:
    with pytest.raises(TypeError, match=r"intercept.*real scalar"):
        Baseline(intercept=invalid, reference_hz=2.87e9)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_parameter_fields_reject_non_finite_scalars(invalid: float) -> None:
    with pytest.raises(ValueError, match=r"center_hz.*finite"):
        Resonance("r0", invalid, 2.0e6, 0.02, 0.5)


@pytest.mark.parametrize("invalid", [None, 1, b"r0"])
def test_resonance_id_must_be_a_string(invalid: object) -> None:
    with pytest.raises(TypeError, match=r"resonance_id.*string"):
        Resonance(invalid, 2.87e9, 2.0e6, 0.02, 0.5)  # type: ignore[arg-type]
