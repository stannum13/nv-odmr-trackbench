"""Immutable full and estimator-safe virtual-instrument observations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, Real | np.integer | np.floating
    ):
        raise TypeError(f"{name} must be a real scalar")
    canonical = float(value)
    if not np.isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    return canonical


def _nonnegative_float(value: object, name: str) -> float:
    canonical = _finite_float(value, name)
    if canonical < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return canonical


def _positive_float(value: object, name: str) -> float:
    canonical = _finite_float(value, name)
    if canonical <= 0.0:
        raise ValueError(f"{name} must be positive")
    return canonical


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, Integral | np.integer
    ):
        raise TypeError(f"{name} must be an integer")
    canonical = int(value)
    if canonical < 0:
        raise ValueError(f"{name} must be non-negative")
    return canonical


@dataclass(frozen=True, slots=True)
class EstimatorObservation:
    """Public observation payload, intentionally isolated from hidden truth."""

    sequence_index: int
    timestamp_s: float
    frequency_hz: float
    fluorescence: float
    integration_time_s: float
    nominal_exposure_photons: float
    realized_photons: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_index",
            _nonnegative_int(self.sequence_index, "sequence_index"),
        )
        object.__setattr__(
            self, "timestamp_s", _nonnegative_float(self.timestamp_s, "timestamp_s")
        )
        object.__setattr__(
            self, "frequency_hz", _finite_float(self.frequency_hz, "frequency_hz")
        )
        object.__setattr__(
            self, "fluorescence", _finite_float(self.fluorescence, "fluorescence")
        )
        object.__setattr__(
            self,
            "integration_time_s",
            _positive_float(self.integration_time_s, "integration_time_s"),
        )
        object.__setattr__(
            self,
            "nominal_exposure_photons",
            _nonnegative_float(
                self.nominal_exposure_photons, "nominal_exposure_photons"
            ),
        )
        if self.realized_photons is not None:
            object.__setattr__(
                self,
                "realized_photons",
                _nonnegative_int(self.realized_photons, "realized_photons"),
            )


@dataclass(frozen=True, slots=True)
class InstrumentObservation:
    """Evaluator-owned record with signal-conditioned expected photons."""

    sequence_index: int
    timestamp_s: float
    frequency_hz: float
    fluorescence: float
    integration_time_s: float
    nominal_exposure_photons: float
    expected_photons: float
    realized_photons: int | None = None
    sampling_rule: str = ""

    def __post_init__(self) -> None:
        public = EstimatorObservation(
            sequence_index=self.sequence_index,
            timestamp_s=self.timestamp_s,
            frequency_hz=self.frequency_hz,
            fluorescence=self.fluorescence,
            integration_time_s=self.integration_time_s,
            nominal_exposure_photons=self.nominal_exposure_photons,
            realized_photons=self.realized_photons,
        )
        object.__setattr__(self, "sequence_index", public.sequence_index)
        object.__setattr__(self, "timestamp_s", public.timestamp_s)
        object.__setattr__(self, "frequency_hz", public.frequency_hz)
        object.__setattr__(self, "fluorescence", public.fluorescence)
        object.__setattr__(self, "integration_time_s", public.integration_time_s)
        object.__setattr__(
            self, "nominal_exposure_photons", public.nominal_exposure_photons
        )
        object.__setattr__(self, "realized_photons", public.realized_photons)
        object.__setattr__(
            self,
            "expected_photons",
            _nonnegative_float(self.expected_photons, "expected_photons"),
        )
        if not isinstance(self.sampling_rule, str) or not self.sampling_rule:
            raise ValueError("sampling_rule must be a non-empty string")

    def estimator_view(self) -> EstimatorObservation:
        """Return a separately allocated immutable observation safe for estimators."""
        return EstimatorObservation(
            sequence_index=self.sequence_index,
            timestamp_s=self.timestamp_s,
            frequency_hz=self.frequency_hz,
            fluorescence=self.fluorescence,
            integration_time_s=self.integration_time_s,
            nominal_exposure_photons=self.nominal_exposure_photons,
            realized_photons=self.realized_photons,
        )
