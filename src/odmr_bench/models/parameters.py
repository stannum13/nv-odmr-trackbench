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
