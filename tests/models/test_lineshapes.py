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
