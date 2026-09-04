"""Cold-start estimation from independently completed ODMR sweeps."""

from __future__ import annotations

from odmr_bench.estimators.fitting import fit_spectrum
from odmr_bench.estimators.types import (
    CompleteSweep,
    FitConfiguration,
    SweepEstimate,
)


class RepeatedFullSweepEstimator:
    """Fit every completed sweep independently and retain evaluator history."""

    __slots__ = ("_configuration", "_history")

    def __init__(self, configuration: FitConfiguration) -> None:
        if not isinstance(configuration, FitConfiguration):
            raise TypeError("configuration must be a FitConfiguration")
        self._configuration = configuration
        self._history: list[SweepEstimate] = []

    @property
    def configuration(self) -> FitConfiguration:
        """Return the immutable fitting configuration."""
        return self._configuration

    @property
    def latest(self) -> SweepEstimate | None:
        """Return the most recent attempted fit, including a failure."""
        return self._history[-1] if self._history else None

    @property
    def history(self) -> tuple[SweepEstimate, ...]:
        """Return an immutable evaluator-only snapshot of attempted fits."""
        return tuple(self._history)

    def reset(self) -> None:
        """Remove all retained sweep estimates."""
        self._history.clear()

    def update_sweep(self, sweep: CompleteSweep) -> SweepEstimate:
        """Cold-start fit one completed sweep and retain its public metadata."""
        if not isinstance(sweep, CompleteSweep):
            raise TypeError("sweep must be a CompleteSweep")

        fit = fit_spectrum(sweep, self._configuration, initial_guess=None)
        estimate = SweepEstimate(
            fit=fit,
            last_sequence_index=sweep.last_sequence_index,
            last_timestamp_s=sweep.last_timestamp_s,
            total_integration_time_s=sweep.total_integration_time_s,
            total_nominal_exposure_photons=(sweep.total_nominal_exposure_photons),
        )
        self._history.append(estimate)
        return estimate


__all__ = ["RepeatedFullSweepEstimator"]
