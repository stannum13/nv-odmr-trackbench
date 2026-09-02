"""Event-driven, truth-isolating ODMR virtual-instrument queries."""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real
from typing import Any

import numpy as np

from odmr_bench.dynamics import SpectralDynamics, SpectralSnapshot
from odmr_bench.emulator.noise import NoiseResult
from odmr_bench.emulator.observations import InstrumentObservation
from odmr_bench.emulator.resources import ResourceLedger, ResourceSnapshot
from odmr_bench.models import multi_resonance_spectrum


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


def _nonnegative_float(value: object, name: str) -> float:
    canonical = _finite_float(value, name)
    if canonical < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return canonical


def _validate_noise(noise: object) -> None:
    if not callable(getattr(noise, "sample", None)):
        raise TypeError("noise must provide a callable sample method")
    sampling_rule = getattr(noise, "sampling_rule", None)
    if not isinstance(sampling_rule, str) or not sampling_rule:
        raise ValueError("noise.sampling_rule must be a non-empty string")


def _checkpoint_noise(noise: object) -> object:
    """Capture opaque strategy state before sampling, or reject unsafe plugins."""
    checkpoint = getattr(noise, "checkpoint", None)
    restore = getattr(noise, "restore", None)
    if not callable(checkpoint) or not callable(restore):
        raise TypeError(
            "noise must implement callable checkpoint/restore methods before sampling"
        )
    return checkpoint()


class ODMRInstrument:
    """Causal virtual ODMR acquisition with a midpoint spectrum approximation."""

    __slots__ = (
        "_dynamics",
        "_frequency_overhead_s",
        "_ledger",
        "_noise",
        "_nominal_photon_rate_hz",
        "_rng",
        "_sequence_index",
        "_virtual_time_s",
    )

    def __init__(
        self,
        *,
        dynamics: SpectralDynamics,
        noise: object,
        nominal_photon_rate_hz: float,
        frequency_overhead_s: float = 0.0,
        seed: int | np.integer | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not isinstance(dynamics, SpectralDynamics):
            raise TypeError("dynamics must implement SpectralDynamics")
        _validate_noise(noise)
        nominal_rate = _positive_float(
            nominal_photon_rate_hz, "nominal_photon_rate_hz"
        )
        overhead = _nonnegative_float(frequency_overhead_s, "frequency_overhead_s")
        if (seed is None) == (rng is None):
            raise ValueError("provide exactly one of seed or rng")
        if rng is not None:
            if not isinstance(rng, np.random.Generator):
                raise TypeError("rng must be a numpy.random.Generator")
            generator = rng
        else:
            if isinstance(seed, bool | np.bool_) or not isinstance(
                seed, Integral | np.integer
            ):
                raise TypeError("seed must be an integer")
            generator = np.random.default_rng(int(seed))

        # Validate the hidden scenario before the first query without exposing
        # its snapshot through the public instrument interface.
        initial_snapshot = dynamics.snapshot_at(0.0)
        if not isinstance(initial_snapshot, SpectralSnapshot):
            raise TypeError("dynamics.snapshot_at must return a SpectralSnapshot")

        self._dynamics = dynamics
        self._noise = noise
        self._nominal_photon_rate_hz = nominal_rate
        self._frequency_overhead_s = overhead
        self._rng = generator
        self._ledger = ResourceLedger()
        self._virtual_time_s = 0.0
        self._sequence_index = 0

    @property
    def resources(self) -> ResourceSnapshot:
        """Return immutable acquisition totals for successful queries only."""
        return self._ledger.snapshot()

    @property
    def virtual_time_s(self) -> float:
        """Return the endpoint of the latest successful virtual acquisition."""
        return self._virtual_time_s

    def query(
        self, frequency_hz: float, integration_time_s: float
    ) -> InstrumentObservation:
        """Acquire one noisy observation at a chosen frequency and duration.

        Setting overhead precedes integration.  The deterministic spectrum is
        sampled at the integration midpoint, while the returned timestamp is
        the integration endpoint.  State commits only after all checks pass.
        """
        frequency = _positive_float(frequency_hz, "frequency_hz")
        integration = _positive_float(integration_time_s, "integration_time_s")
        integration_start = self._virtual_time_s + self._frequency_overhead_s
        midpoint = integration_start + integration / 2.0
        endpoint = integration_start + integration
        if not all(
            np.isfinite(timestamp)
            for timestamp in (integration_start, midpoint, endpoint)
        ):
            raise ValueError("query virtual timestamps must remain finite")

        snapshot = self._dynamics.snapshot_at(midpoint)
        if not isinstance(snapshot, SpectralSnapshot):
            raise TypeError("dynamics.snapshot_at must return a SpectralSnapshot")
        spectrum = np.asarray(
            multi_resonance_spectrum(
                np.asarray([frequency]), snapshot.resonances, snapshot.baseline
            ),
            dtype=np.float64,
        )
        if spectrum.shape != (1,):
            raise ValueError("one-point spectrum evaluation must return one value")
        expected_fluorescence = _finite_float(
            spectrum[0], "expected normalized fluorescence"
        )
        if expected_fluorescence < 0.0:
            raise ValueError("expected normalized fluorescence must be non-negative")
        nominal_exposure = self._nominal_photon_rate_hz * integration
        expected_photons = expected_fluorescence * nominal_exposure
        nominal_exposure = _nonnegative_float(
            nominal_exposure, "nominal_exposure_photons"
        )
        expected_photons = _nonnegative_float(expected_photons, "expected_photons")

        # Sampling can advance both the generator and a stateful strategy.
        # The explicit noise protocol restores aliases and externally held
        # mutable objects in place if a query cannot commit.
        rng_state: dict[str, Any] = deepcopy(self._rng.bit_generator.state)
        noise_state = _checkpoint_noise(self._noise)
        try:
            noise_result = self._noise.sample(
                expected_fluorescence,
                self._nominal_photon_rate_hz,
                integration,
                self._rng,
            )
            if not isinstance(noise_result, NoiseResult):
                raise TypeError("noise.sample must return a NoiseResult")
            sampling_rule = getattr(self._noise, "sampling_rule", None)
            if not isinstance(sampling_rule, str) or not sampling_rule:
                raise ValueError("noise.sampling_rule must be a non-empty string")
            observation = InstrumentObservation(
                sequence_index=self._sequence_index,
                timestamp_s=endpoint,
                frequency_hz=frequency,
                fluorescence=noise_result.fluorescence,
                integration_time_s=integration,
                nominal_exposure_photons=nominal_exposure,
                expected_photons=expected_photons,
                realized_photons=noise_result.realized_photons,
                sampling_rule=sampling_rule,
            )
            self._ledger.record(
                integration_time_s=integration,
                nominal_exposure_photons=nominal_exposure,
                expected_photons=expected_photons,
                realized_photons=noise_result.realized_photons,
                virtual_elapsed_time_s=self._frequency_overhead_s + integration,
            )
        except Exception:
            self._rng.bit_generator.state = rng_state
            self._noise.restore(noise_state)
            raise

        self._virtual_time_s = endpoint
        self._sequence_index += 1
        return observation
