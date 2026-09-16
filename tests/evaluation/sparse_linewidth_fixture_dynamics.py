"""Deterministic test support for composing linewidth and center drift."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from types import MappingProxyType

import numpy as np

from odmr_bench.dynamics import (
    SpectralDynamics,
    SpectralSnapshot,
)
from odmr_bench.dynamics.base import validate_timestamp_s


def _canonical_real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, Real | np.integer | np.floating
    ):
        raise TypeError(f"{name} values must be real scalars")
    canonical = float(value)
    if not np.isfinite(canonical):
        raise ValueError(f"{name} values must be finite")
    if positive and canonical <= 0.0:
        raise ValueError(f"{name} values must be positive")
    return canonical


def _canonical_mapping(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    canonical: dict[str, float] = {}
    for resonance_id, configured_value in value.items():
        if not isinstance(resonance_id, str) or not resonance_id.strip():
            raise TypeError(f"{name} keys must be nonempty resonance ID strings")
        canonical[resonance_id] = _canonical_real(
            configured_value, name, positive=positive
        )
    return MappingProxyType(canonical)


@dataclass(frozen=True, slots=True)
class DeterministicLinewidthDrift:
    """Apply deterministic scalar or per-ID linewidth slew to base dynamics."""

    base_dynamics: SpectralDynamics
    reference_fwhm_hz: Mapping[str, float]
    fwhm_slew_hz_per_s: float | Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.base_dynamics, SpectralDynamics):
            raise TypeError("base_dynamics must implement SpectralDynamics")
        reference = _canonical_mapping(
            self.reference_fwhm_hz,
            "reference_fwhm_hz",
            positive=True,
        )
        object.__setattr__(self, "reference_fwhm_hz", reference)

        configured_slew = self.fwhm_slew_hz_per_s
        if isinstance(configured_slew, Mapping):
            slews = _canonical_mapping(configured_slew, "fwhm_slew_hz_per_s")
            if set(slews) != set(reference):
                raise ValueError(
                    "fwhm_slew_hz_per_s mapping must contain exactly the "
                    "reference_fwhm_hz IDs"
                )
            object.__setattr__(self, "fwhm_slew_hz_per_s", slews)
            return
        object.__setattr__(
            self,
            "fwhm_slew_hz_per_s",
            _canonical_real(configured_slew, "fwhm_slew_hz_per_s"),
        )

    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        timestamp = validate_timestamp_s(timestamp_s)
        base = self.base_dynamics.snapshot_at(timestamp)
        if not isinstance(base, SpectralSnapshot):
            raise TypeError("base_dynamics.snapshot_at must return a SpectralSnapshot")
        base_ids = tuple(resonance.resonance_id for resonance in base.resonances)
        if set(base_ids) != set(self.reference_fwhm_hz):
            raise ValueError(
                "reference_fwhm_hz mapping must contain exactly the base snapshot IDs"
            )

        resonances = tuple(
            replace(
                resonance,
                fwhm_hz=self._width_at(resonance.resonance_id, timestamp),
            )
            for resonance in base.resonances
        )
        return SpectralSnapshot(baseline=base.baseline, resonances=resonances)

    def _width_at(self, resonance_id: str, timestamp_s: float) -> float:
        configured_slew = self.fwhm_slew_hz_per_s
        slew = (
            configured_slew[resonance_id]
            if isinstance(configured_slew, Mapping)
            else configured_slew
        )
        width = self.reference_fwhm_hz[resonance_id] + slew * timestamp_s
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("generated fwhm_hz must remain finite and positive")
        return width
