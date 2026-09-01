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
