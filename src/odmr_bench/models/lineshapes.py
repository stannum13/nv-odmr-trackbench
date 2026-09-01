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
