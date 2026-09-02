"""Hidden, physically valid spectral states and their dynamics interface."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np

from odmr_bench.models import Baseline, Resonance


def validate_timestamp_s(timestamp_s: object) -> float:
    """Return a finite, non-negative virtual timestamp in seconds."""
    if isinstance(timestamp_s, bool | np.bool_) or not isinstance(
        timestamp_s, Real | np.integer | np.floating
    ):
        raise TypeError("timestamp_s must be a real scalar")
    canonical = float(timestamp_s)
    if not np.isfinite(canonical):
        raise ValueError("timestamp_s must be finite")
    if canonical < 0.0:
        raise ValueError("timestamp_s must be non-negative")
    return canonical


@dataclass(frozen=True, slots=True)
class SpectralSnapshot:
    """One hidden eight-resonance physical state at virtual time."""

    baseline: Baseline
    resonances: tuple[Resonance, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, Baseline):
            raise TypeError("baseline must be a Baseline")
        try:
            resonances = tuple([*self.resonances])
        except TypeError as error:
            raise TypeError("resonances must be an iterable of Resonance") from error
        if len(resonances) != 8:
            raise ValueError("resonances must contain exactly eight entries")
        if not all(isinstance(resonance, Resonance) for resonance in resonances):
            raise TypeError("resonances must contain only Resonance objects")
        resonance_ids = [resonance.resonance_id for resonance in resonances]
        if len(set(resonance_ids)) != len(resonance_ids):
            raise ValueError("resonance IDs must be unique")
        if any(resonance.center_hz <= 0.0 for resonance in resonances):
            raise ValueError("resonance centers must be positive")
        object.__setattr__(self, "resonances", resonances)


@runtime_checkable
class SpectralDynamics(Protocol):
    """Provider of hidden physical states evaluated at virtual time."""

    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        """Return the physical state at a finite, non-negative timestamp."""
        ...
