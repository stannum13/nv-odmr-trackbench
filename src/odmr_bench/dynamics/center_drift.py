"""Deterministic stationary and linear-center-drift hidden dynamics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from types import MappingProxyType

import numpy as np

from odmr_bench.dynamics.base import SpectralSnapshot, validate_timestamp_s


def _validate_slew_hz_per_s(value: object) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, Real | np.integer | np.floating
    ):
        raise TypeError("center_slew_hz_per_s must be a real scalar")
    canonical = float(value)
    if not np.isfinite(canonical):
        raise ValueError("center_slew_hz_per_s must be finite")
    return canonical


@dataclass(frozen=True, slots=True)
class StationaryDynamics:
    """Dynamics that return the same physical parameters at every valid time."""

    initial_snapshot: SpectralSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.initial_snapshot, SpectralSnapshot):
            raise TypeError("initial_snapshot must be a SpectralSnapshot")

    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        validate_timestamp_s(timestamp_s)
        return SpectralSnapshot(
            baseline=self.initial_snapshot.baseline,
            resonances=self.initial_snapshot.resonances,
        )


@dataclass(frozen=True, slots=True)
class LinearCenterDrift:
    """Apply a common or per-physical-ID linear center slew."""

    initial_snapshot: SpectralSnapshot
    center_slew_hz_per_s: float | Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.initial_snapshot, SpectralSnapshot):
            raise TypeError("initial_snapshot must be a SpectralSnapshot")
        resonance_ids = tuple(
            resonance.resonance_id for resonance in self.initial_snapshot.resonances
        )
        configured_slew = self.center_slew_hz_per_s
        if isinstance(configured_slew, Mapping):
            configured_ids = set(configured_slew)
            required_ids = set(resonance_ids)
            if configured_ids != required_ids:
                raise ValueError(
                    "center_slew_hz_per_s mapping must contain exactly the snapshot IDs"
                )
            slews = {
                resonance_id: _validate_slew_hz_per_s(configured_slew[resonance_id])
                for resonance_id in resonance_ids
            }
            object.__setattr__(
                self,
                "center_slew_hz_per_s",
                MappingProxyType(slews),
            )
            return
        common_slew = _validate_slew_hz_per_s(configured_slew)
        object.__setattr__(self, "center_slew_hz_per_s", common_slew)

    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        timestamp = validate_timestamp_s(timestamp_s)
        configured_slew = self.center_slew_hz_per_s
        resonances = tuple(
            replace(
                resonance,
                center_hz=resonance.center_hz
                + self._slew_for(resonance.resonance_id, configured_slew) * timestamp,
            )
            for resonance in self.initial_snapshot.resonances
        )
        if any(resonance.center_hz <= 0.0 for resonance in resonances):
            raise ValueError("resonance centers must remain positive at timestamp_s")
        return SpectralSnapshot(
            baseline=self.initial_snapshot.baseline,
            resonances=resonances,
        )

    @staticmethod
    def _slew_for(
        resonance_id: str,
        configured_slew: float | Mapping[str, float],
    ) -> float:
        if isinstance(configured_slew, Mapping):
            return configured_slew[resonance_id]
        return configured_slew
