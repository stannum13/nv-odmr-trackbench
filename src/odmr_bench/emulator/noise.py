"""Declared stochastic noise models for normalized fluorescence observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, Real | np.integer | np.floating
    ):
        raise TypeError(f"{name} must be a real scalar")
    canonical = float(value)
    if not np.isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    return canonical


def _positive_float(value: object, name: str) -> float:
    canonical = _finite_float(value, name)
    if canonical <= 0.0:
        raise ValueError(f"{name} must be positive")
    return canonical


def _validate_sample_inputs(
    expected_fluorescence: object,
    nominal_rate_hz: object,
    integration_time_s: object,
    rng: object,
) -> tuple[float, float, float, np.random.Generator]:
    fluorescence = _finite_float(expected_fluorescence, "expected_fluorescence")
    if fluorescence < 0.0:
        raise ValueError("expected_fluorescence must be non-negative")
    rate_hz = _positive_float(nominal_rate_hz, "nominal_rate_hz")
    duration_s = _positive_float(integration_time_s, "integration_time_s")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return fluorescence, rate_hz, duration_s, rng


def _nonnegative_integral(value: object, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, Integral | np.integer
    ):
        raise TypeError(f"{name} must be an integer")
    canonical = int(value)
    if canonical < 0:
        raise ValueError(f"{name} must be non-negative")
    return canonical


@dataclass(frozen=True, slots=True)
class NoiseResult:
    """One measured fluorescence sample and optional observed photon count."""

    fluorescence: float
    realized_photons: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fluorescence", _finite_float(self.fluorescence, "fluorescence")
        )
        if self.realized_photons is not None:
            object.__setattr__(
                self,
                "realized_photons",
                _nonnegative_integral(self.realized_photons, "realized_photons"),
            )


@dataclass(frozen=True, slots=True)
class PoissonNoise:
    """Photon shot noise with counts retained alongside normalized fluorescence."""

    sampling_rule: str = "poisson"

    def sample(
        self,
        expected_fluorescence: float,
        nominal_rate_hz: float,
        integration_time_s: float,
        rng: np.random.Generator,
    ) -> NoiseResult:
        fluorescence, rate_hz, duration_s, validated_rng = _validate_sample_inputs(
            expected_fluorescence, nominal_rate_hz, integration_time_s, rng
        )
        expected_count = fluorescence * rate_hz * duration_s
        try:
            count = validated_rng.poisson(expected_count)
        except ValueError as error:
            raise ValueError(
                "Poisson sampling failed for the configured expected photon count"
            ) from error
        realized_photons = _nonnegative_integral(count, "realized_photons")
        return NoiseResult(
            fluorescence=realized_photons / (rate_hz * duration_s),
            realized_photons=realized_photons,
        )


@dataclass(frozen=True, slots=True)
class GaussianNoise:
    """Controlled additive normalized-fluorescence noise, not a count model."""

    stddev_at_1s: float
    sampling_rule: str = "gaussian"

    def __post_init__(self) -> None:
        stddev_at_1s = _finite_float(self.stddev_at_1s, "stddev_at_1s")
        if stddev_at_1s < 0.0:
            raise ValueError("stddev_at_1s must be non-negative")
        object.__setattr__(
            self, "stddev_at_1s", stddev_at_1s
        )

    def sample(
        self,
        expected_fluorescence: float,
        nominal_rate_hz: float,
        integration_time_s: float,
        rng: np.random.Generator,
    ) -> NoiseResult:
        fluorescence, _, duration_s, validated_rng = _validate_sample_inputs(
            expected_fluorescence, nominal_rate_hz, integration_time_s, rng
        )
        stddev = self.stddev_at_1s / np.sqrt(duration_s)
        return NoiseResult(fluorescence + validated_rng.normal(scale=stddev))


class EmpiricalResidualNoise:
    """Supplied normalized residuals with explicit temporal-correlation semantics."""

    __slots__ = (
        "_block_offset",
        "_block_size",
        "_block_start",
        "_configuration_locked",
        "_cursor",
        "_mode",
        "_provenance",
        "_residuals",
    )

    _REQUIRED_PROVENANCE_KEYS = frozenset(
        {"source_id", "preparation_label", "normalization_label", "correlation_mode"}
    )
    _IMMUTABLE_CONFIGURATION_NAMES = frozenset(
        {
            "_block_size",
            "_configuration_locked",
            "_mode",
            "_provenance",
            "_residuals",
            "block_size",
            "mode",
            "provenance",
            "residuals",
        }
    )

    def __init__(
        self,
        residuals: ArrayLike,
        *,
        mode: str,
        provenance: Mapping[str, str],
        block_size: int | None = None,
    ) -> None:
        values = np.asarray(residuals, dtype=np.float64).copy()
        if values.ndim != 1 or values.size == 0:
            raise ValueError("residuals must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(values)):
            raise ValueError("residuals must be finite")
        if mode not in {"replay", "sample", "block"}:
            raise ValueError("mode must be exactly 'replay', 'sample', or 'block'")
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        copied_provenance = dict(provenance)
        if set(copied_provenance) != self._REQUIRED_PROVENANCE_KEYS:
            raise ValueError(
                "provenance must contain exactly the required residual fields"
            )
        if any(
            not isinstance(value, str) or not value
            for value in copied_provenance.values()
        ):
            raise ValueError("residual provenance values must be non-empty strings")
        if copied_provenance["correlation_mode"] != mode:
            raise ValueError("provenance correlation_mode must match mode")
        if mode == "block":
            if block_size is None:
                raise ValueError("block_size is required for block residual mode")
            block_size = _nonnegative_integral(block_size, "block_size")
            if block_size == 0:
                raise ValueError("block_size must be positive")
        elif block_size is not None:
            raise ValueError("block_size is only valid for block residual mode")

        values.setflags(write=False)
        object.__setattr__(self, "_residuals", values)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_block_size", block_size)
        object.__setattr__(self, "_provenance", MappingProxyType(copied_provenance))
        self._cursor = 0
        self._block_start = 0
        self._block_offset = 0
        object.__setattr__(self, "_configuration_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if (
            getattr(self, "_configuration_locked", False)
            and name in self._IMMUTABLE_CONFIGURATION_NAMES
        ):
            raise AttributeError("empirical residual configuration is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._IMMUTABLE_CONFIGURATION_NAMES:
            raise AttributeError("empirical residual configuration is immutable")
        object.__delattr__(self, name)

    @property
    def residuals(self) -> np.ndarray:
        """Return the defensively copied, read-only residual sequence."""
        copied_residuals = self._residuals.copy()
        copied_residuals.setflags(write=False)
        return copied_residuals

    @property
    def mode(self) -> str:
        """Return the fixed temporal-correlation mode."""
        return self._mode

    @property
    def block_size(self) -> int | None:
        """Return the fixed contiguous-block length, when applicable."""
        return self._block_size

    @property
    def provenance(self) -> Mapping[str, str]:
        """Return immutable residual-source provenance."""
        return self._provenance

    @property
    def sampling_rule(self) -> str:
        """Return a rule that retains the selected residual correlation mode."""
        return f"empirical_{self.mode}"

    def sample(
        self,
        expected_fluorescence: float,
        nominal_rate_hz: float,
        integration_time_s: float,
        rng: np.random.Generator,
    ) -> NoiseResult:
        fluorescence, _, _, validated_rng = _validate_sample_inputs(
            expected_fluorescence, nominal_rate_hz, integration_time_s, rng
        )
        if self._mode == "replay":
            residual = self._residuals[self._cursor]
            self._cursor = (self._cursor + 1) % self._residuals.size
        elif self._mode == "sample":
            residual = self._residuals[
                validated_rng.integers(0, self._residuals.size)
            ]
        else:
            if self._block_offset == 0:
                self._block_start = int(
                    validated_rng.integers(0, self._residuals.size)
                )
            residual = self._residuals[
                (self._block_start + self._block_offset) % self._residuals.size
            ]
            self._block_offset = (self._block_offset + 1) % self._block_size  # type: ignore[operator]
        return NoiseResult(fluorescence + float(residual))
