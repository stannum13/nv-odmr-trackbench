"""Validated immutable parameters for ODMR spectra."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _canonical_real_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    canonical = float(value)
    if not np.isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    return canonical


@dataclass(frozen=True, slots=True)
class Resonance:
    resonance_id: str
    center_hz: float
    fwhm_hz: float
    amplitude: float
    eta: float

    def __post_init__(self) -> None:
        if not isinstance(self.resonance_id, str):
            raise TypeError("resonance_id must be a string")
        if not self.resonance_id.strip():
            raise ValueError("resonance_id must be nonempty")
        for name in ("center_hz", "fwhm_hz", "amplitude", "eta"):
            object.__setattr__(
                self,
                name,
                _canonical_real_scalar(getattr(self, name), name),
            )
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
        for name in (
            "intercept",
            "reference_hz",
            "slope_per_hz",
            "quadratic_per_hz2",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_real_scalar(getattr(self, name), name),
            )

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
