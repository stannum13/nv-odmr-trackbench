"""Atomic acquisition-resource accounting for virtual-instrument queries."""

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
class ResourceSnapshot:
    """Immutable public totals for all completed virtual acquisitions."""

    observations: int
    integration_time_s: float
    nominal_exposure_photons: float
    expected_photons: float
    realized_photons: int
    observations_without_realized_counts: int
    virtual_elapsed_time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observations", _nonnegative_int(self.observations, "observations")
        )
        object.__setattr__(
            self,
            "integration_time_s",
            _nonnegative_float(self.integration_time_s, "integration_time_s"),
        )
        object.__setattr__(
            self,
            "nominal_exposure_photons",
            _nonnegative_float(
                self.nominal_exposure_photons, "nominal_exposure_photons"
            ),
        )
        object.__setattr__(
            self,
            "expected_photons",
            _nonnegative_float(self.expected_photons, "expected_photons"),
        )
        object.__setattr__(
            self,
            "realized_photons",
            _nonnegative_int(self.realized_photons, "realized_photons"),
        )
        object.__setattr__(
            self,
            "observations_without_realized_counts",
            _nonnegative_int(
                self.observations_without_realized_counts,
                "observations_without_realized_counts",
            ),
        )
        object.__setattr__(
            self,
            "virtual_elapsed_time_s",
            _nonnegative_float(
                self.virtual_elapsed_time_s, "virtual_elapsed_time_s"
            ),
        )
        if self.observations_without_realized_counts > self.observations:
            raise ValueError(
                "observations_without_realized_counts cannot exceed observations"
            )
        if self.virtual_elapsed_time_s < self.integration_time_s:
            raise ValueError("virtual_elapsed_time_s must include integration_time_s")


class ResourceLedger:
    """Private mutable resource totals committed only after query success."""

    __slots__ = (
        "_expected_photons",
        "_integration_time_s",
        "_nominal_exposure_photons",
        "_observations",
        "_observations_without_realized_counts",
        "_realized_photons",
        "_virtual_elapsed_time_s",
    )

    def __init__(self) -> None:
        self._observations = 0
        self._integration_time_s = 0.0
        self._nominal_exposure_photons = 0.0
        self._expected_photons = 0.0
        self._realized_photons = 0
        self._observations_without_realized_counts = 0
        self._virtual_elapsed_time_s = 0.0

    def record(
        self,
        *,
        integration_time_s: float,
        nominal_exposure_photons: float,
        expected_photons: float,
        realized_photons: int | None,
        virtual_elapsed_time_s: float,
    ) -> None:
        """Atomically add one successful acquisition's declared resource use."""
        integration = _positive_float(integration_time_s, "integration_time_s")
        nominal = _nonnegative_float(
            nominal_exposure_photons, "nominal_exposure_photons"
        )
        expected = _nonnegative_float(expected_photons, "expected_photons")
        elapsed = _nonnegative_float(virtual_elapsed_time_s, "virtual_elapsed_time_s")
        if elapsed < integration:
            raise ValueError("virtual_elapsed_time_s must include integration_time_s")
        realized = (
            None
            if realized_photons is None
            else _nonnegative_int(realized_photons, "realized_photons")
        )

        prospective = ResourceSnapshot(
            observations=self._observations + 1,
            integration_time_s=self._integration_time_s + integration,
            nominal_exposure_photons=self._nominal_exposure_photons + nominal,
            expected_photons=self._expected_photons + expected,
            realized_photons=self._realized_photons
            if realized is None
            else self._realized_photons + realized,
            observations_without_realized_counts=(
                self._observations_without_realized_counts + 1
                if realized is None
                else self._observations_without_realized_counts
            ),
            virtual_elapsed_time_s=self._virtual_elapsed_time_s + elapsed,
        )

        self._observations = prospective.observations
        self._integration_time_s = prospective.integration_time_s
        self._nominal_exposure_photons = prospective.nominal_exposure_photons
        self._expected_photons = prospective.expected_photons
        self._realized_photons = prospective.realized_photons
        self._observations_without_realized_counts = (
            prospective.observations_without_realized_counts
        )
        self._virtual_elapsed_time_s = prospective.virtual_elapsed_time_s

    def snapshot(self) -> ResourceSnapshot:
        """Return a frozen copy of the completed acquisition totals."""
        return ResourceSnapshot(
            observations=self._observations,
            integration_time_s=self._integration_time_s,
            nominal_exposure_photons=self._nominal_exposure_photons,
            expected_photons=self._expected_photons,
            realized_photons=self._realized_photons,
            observations_without_realized_counts=self._observations_without_realized_counts,
            virtual_elapsed_time_s=self._virtual_elapsed_time_s,
        )
