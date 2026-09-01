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


def test_scalar_frequency_returns_scalar_shaped_array() -> None:
    baseline = Baseline(intercept=1.0, reference_hz=2.87e9)
    result = multi_resonance_spectrum(2.87e9, [], baseline)

    assert result.shape == ()
    assert_allclose(result, 1.0)


def test_multidimensional_frequency_and_noise_broadcasting() -> None:
    frequency_hz = np.array(
        [[2.86e9, 2.87e9, 2.88e9], [2.865e9, 2.875e9, 2.885e9]]
    )
    baseline = Baseline(intercept=1.0, reference_hz=2.87e9)
    resonance = Resonance("r0", 2.87e9, 2.0e6, 0.025, 0.4)
    column_noise = np.array([[0.01], [-0.02]])

    result = multi_resonance_spectrum(
        frequency_hz,
        [resonance],
        baseline,
        additive_noise=column_noise,
    )
    expected = (
        baseline.evaluate(frequency_hz)
        - resonance.amplitude
        * pseudo_voigt(
            frequency_hz,
            resonance.center_hz,
            resonance.fwhm_hz,
            resonance.eta,
        )
        + column_noise
    )

    assert result.shape == frequency_hz.shape
    assert_allclose(result, expected)


def test_scalar_additive_noise_broadcasts_to_every_frequency() -> None:
    frequency_hz = np.array([2.86e9, 2.87e9, 2.88e9])
    baseline = Baseline(intercept=1.0, reference_hz=2.87e9)

    result = multi_resonance_spectrum(
        frequency_hz,
        [],
        baseline,
        additive_noise=0.125,
    )

    assert_allclose(result, [1.125, 1.125, 1.125])


def test_incompatible_additive_noise_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="broadcast to frequency_hz"):
        multi_resonance_spectrum(
            np.zeros((2, 3)),
            [],
            Baseline(intercept=1.0, reference_hz=0.0),
            additive_noise=np.zeros((4,)),
        )
