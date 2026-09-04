"""Causal warm-start estimation from completed ODMR sweeps."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from odmr_bench.estimators.fitting import fit_spectrum
from odmr_bench.estimators.preparation import (
    FitPreflight,
    prepare_warm_start,
    start_independent_preflight,
)
from odmr_bench.estimators.types import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    SpectrumFitResult,
    SweepFitAttempt,
    WarmSweepEstimate,
)


@dataclass(frozen=True, slots=True)
class _ProspectiveEndpoints:
    sequence_available: bool
    first_sequence_index: int | None
    last_sequence_index: int | None
    timestamp_available: bool
    last_timestamp_s: float | None


def _sample_monotonic_cpu_ns(previous_ns: int | None) -> int:
    raw = time.process_time_ns()
    if isinstance(raw, (bool, np.bool_)) or not isinstance(
        raw, (Integral, np.integer)
    ):
        raise TypeError("process_time_ns must return an integer")
    sample_ns = int(raw)
    if previous_ns is not None and sample_ns < previous_ns:
        raise RuntimeError("process CPU clock moved backwards")
    return sample_ns


def _nonnegative_elapsed_ns(start_ns: int, end_ns: int) -> int:
    elapsed_ns = end_ns - start_ns
    if elapsed_ns < 0:
        raise RuntimeError("process CPU clock moved backwards")
    return elapsed_ns


class WarmStartedFullSweepEstimator:
    """Warm-start complete sweeps from the latest earlier successful fit."""

    __slots__ = (
        "_configuration",
        "_cumulative_observation_count",
        "_history",
        "_last_sequence_index",
        "_last_timestamp_s",
        "_latest_success",
        "_max_warm_start_age_updates",
        "_retry_cold_on_warm_failure",
        "_sequence_available",
        "_timestamp_available",
    )

    def __init__(
        self,
        configuration: FitConfiguration,
        *,
        retry_cold_on_warm_failure: bool = True,
        max_warm_start_age_updates: int | None = None,
    ) -> None:
        if not isinstance(configuration, FitConfiguration):
            raise TypeError("configuration must be a FitConfiguration")
        if not isinstance(retry_cold_on_warm_failure, (bool, np.bool_)):
            raise TypeError("retry_cold_on_warm_failure must be a boolean")
        if max_warm_start_age_updates is not None:
            if (
                isinstance(max_warm_start_age_updates, (bool, np.bool_))
                or not isinstance(
                    max_warm_start_age_updates, (Integral, np.integer)
                )
            ):
                raise TypeError("max_warm_start_age_updates must be an integer or None")
            if int(max_warm_start_age_updates) <= 0:
                raise ValueError("max_warm_start_age_updates must be positive")
            max_warm_start_age_updates = int(max_warm_start_age_updates)

        self._configuration = configuration
        self._retry_cold_on_warm_failure = bool(retry_cold_on_warm_failure)
        self._max_warm_start_age_updates = max_warm_start_age_updates
        self._history: list[WarmSweepEstimate] = []
        self._latest_success: WarmSweepEstimate | None = None
        self._sequence_available: bool | None = None
        self._last_sequence_index: int | None = None
        self._timestamp_available: bool | None = None
        self._last_timestamp_s: float | None = None
        self._cumulative_observation_count = 0

    @property
    def configuration(self) -> FitConfiguration:
        """Return the immutable fitting configuration."""
        return self._configuration

    @property
    def latest(self) -> WarmSweepEstimate | None:
        """Return the latest attempted update, including a failure."""
        return self._history[-1] if self._history else None

    @property
    def history(self) -> tuple[WarmSweepEstimate, ...]:
        """Return an immutable snapshot of all accepted updates."""
        return tuple(self._history)

    @property
    def latest_success(self) -> WarmSweepEstimate | None:
        """Return the update that produced the active successful fit."""
        return self._latest_success

    def update_sweep(self, sweep: CompleteSweep) -> WarmSweepEstimate:
        """Process one causally submitted completed sweep."""
        update_started_ns = _sample_monotonic_cpu_ns(None)
        last_sample_ns = update_started_ns
        if not isinstance(sweep, CompleteSweep):
            raise TypeError("sweep must be a CompleteSweep")

        update_index = len(self._history)
        observation_count = int(sweep.frequency_hz.size)
        cumulative_observation_count = (
            self._cumulative_observation_count + observation_count
        )
        endpoints = self._validate_endpoints(sweep, observation_count)

        preflight_started_ns = _sample_monotonic_cpu_ns(last_sample_ns)
        last_sample_ns = preflight_started_ns
        preflight = start_independent_preflight(sweep, self._configuration)
        preflight_finished_ns = _sample_monotonic_cpu_ns(last_sample_ns)
        last_sample_ns = preflight_finished_ns
        raw_attempts: list[
            tuple[str, int | None, SpectrumFitResult, int]
        ]
        if isinstance(preflight, SpectrumFitResult):
            raw_attempts = [
                (
                    "preflight",
                    None,
                    preflight,
                    _nonnegative_elapsed_ns(
                        preflight_started_ns, preflight_finished_ns
                    ),
                )
            ]
            disposition = "not_applicable_preflight"
            rejection_code = None
            rejection_message = None
        else:
            if not isinstance(preflight, FitPreflight):
                raise TypeError("preflight helper returned an invalid result")
            source_estimate = self._latest_success
            if source_estimate is None:
                fit, fit_elapsed_ns, last_sample_ns = self._timed_fit(
                    sweep, None, last_sample_ns
                )
                raw_attempts = [("cold", None, fit, fit_elapsed_ns)]
                disposition = "no_successful_prior"
                rejection_code = None
                rejection_message = None
            else:
                source_age_updates = update_index - source_estimate.update_index
                if (
                    self._max_warm_start_age_updates is not None
                    and source_age_updates > self._max_warm_start_age_updates
                ):
                    fit, fit_elapsed_ns, last_sample_ns = self._timed_fit(
                        sweep, None, last_sample_ns
                    )
                    raw_attempts = [("cold", None, fit, fit_elapsed_ns)]
                    disposition = "rejected_age"
                    rejection_code = "age_limit_exceeded"
                    rejection_message = (
                        f"warm-start source age {source_age_updates} exceeds "
                        f"configured maximum {self._max_warm_start_age_updates}"
                    )
                else:
                    assert source_estimate.active_fit is not None
                    preparation = prepare_warm_start(
                        source_estimate.active_fit, self._configuration, preflight
                    )
                    if preparation.guess is None:
                        fit, fit_elapsed_ns, last_sample_ns = self._timed_fit(
                            sweep, None, last_sample_ns
                        )
                        raw_attempts = [("cold", None, fit, fit_elapsed_ns)]
                        disposition = "rejected_compatibility"
                        rejection_code = preparation.rejection_code
                        rejection_message = preparation.message
                    else:
                        fit, fit_elapsed_ns, last_sample_ns = self._timed_fit(
                            sweep,
                            preparation.guess,
                            last_sample_ns,
                        )
                        raw_attempts = [
                            (
                                "warm",
                                source_estimate.update_index,
                                fit,
                                fit_elapsed_ns,
                            )
                        ]
                        if (
                            self._retry_cold_on_warm_failure
                            and fit.failure_code
                            in {"optimization_failed", "quality_failed"}
                        ):
                            (
                                cold_fit,
                                cold_elapsed_ns,
                                last_sample_ns,
                            ) = self._timed_fit(
                                sweep, None, last_sample_ns
                            )
                            raw_attempts.append(
                                ("cold", None, cold_fit, cold_elapsed_ns)
                            )
                        disposition = "used"
                        rejection_code = None
                        rejection_message = None

        update_finished_ns = _sample_monotonic_cpu_ns(last_sample_ns)
        attempts = tuple(
            SweepFitAttempt(
                start_kind, source_update_index, fit, elapsed_ns / 1_000_000_000.0
            )
            for start_kind, source_update_index, fit, elapsed_ns in raw_attempts
        )
        attempt_sum_s = math.fsum(
            elapsed_ns / 1_000_000_000.0
            for _, _, _, elapsed_ns in raw_attempts
        )
        measured_update_s = (
            update_finished_ns - update_started_ns
        ) / 1_000_000_000.0
        cpu_time_s = max(measured_update_s, attempt_sum_s)

        current_fit = attempts[-1].fit
        source_estimate = self._latest_success
        if current_fit.success:
            active_fit = current_fit
            active_source_update_index = update_index
            submitted_age = 0
            sequence_age = 0 if endpoints.sequence_available else None
            seconds_age = 0.0 if endpoints.timestamp_available else None
        elif source_estimate is not None:
            active_fit = source_estimate.active_fit
            active_source_update_index = source_estimate.update_index
            submitted_age = (
                cumulative_observation_count
                - source_estimate.cumulative_observation_count
            )
            sequence_age = (
                endpoints.last_sequence_index - source_estimate.last_sequence_index
                if endpoints.last_sequence_index is not None
                and source_estimate.last_sequence_index is not None
                else None
            )
            seconds_age = (
                endpoints.last_timestamp_s - source_estimate.last_timestamp_s
                if endpoints.last_timestamp_s is not None
                and source_estimate.last_timestamp_s is not None
                else None
            )
        else:
            active_fit = None
            active_source_update_index = None
            submitted_age = None
            sequence_age = None
            seconds_age = None

        estimate = WarmSweepEstimate(
            update_index=update_index,
            attempts=attempts,
            warm_start_disposition=disposition,
            warm_start_rejection_code=rejection_code,
            warm_start_message=rejection_message,
            active_fit=active_fit,
            active_source_update_index=active_source_update_index,
            estimate_age_submitted_observations=submitted_age,
            estimate_age_sequence_indices=sequence_age,
            estimate_age_s=seconds_age,
            observation_count=observation_count,
            cumulative_observation_count=cumulative_observation_count,
            first_sequence_index=endpoints.first_sequence_index,
            last_sequence_index=endpoints.last_sequence_index,
            last_timestamp_s=endpoints.last_timestamp_s,
            total_integration_time_s=sweep.total_integration_time_s,
            total_nominal_exposure_photons=sweep.total_nominal_exposure_photons,
            cpu_time_s=cpu_time_s,
        )
        self._history.append(estimate)
        self._cumulative_observation_count = cumulative_observation_count
        self._sequence_available = endpoints.sequence_available
        self._last_sequence_index = endpoints.last_sequence_index
        self._timestamp_available = endpoints.timestamp_available
        self._last_timestamp_s = endpoints.last_timestamp_s
        if current_fit.success:
            self._latest_success = estimate
        return estimate

    def _timed_fit(
        self,
        sweep: CompleteSweep,
        initial_guess: FitInitialGuess | None,
        previous_sample_ns: int,
    ) -> tuple[SpectrumFitResult, int, int]:
        attempt_started_ns = _sample_monotonic_cpu_ns(previous_sample_ns)
        fit = fit_spectrum(
            sweep, self._configuration, initial_guess=initial_guess
        )
        attempt_finished_ns = _sample_monotonic_cpu_ns(attempt_started_ns)
        return (
            fit,
            _nonnegative_elapsed_ns(attempt_started_ns, attempt_finished_ns),
            attempt_finished_ns,
        )

    def _validate_endpoints(
        self,
        sweep: CompleteSweep,
        observation_count: int,
    ) -> _ProspectiveEndpoints:
        sequence_available = sweep.last_sequence_index is not None
        if (
            self._sequence_available is not None
            and sequence_available != self._sequence_available
        ):
            raise ValueError("sequence endpoint availability must remain consistent")
        if sequence_available:
            assert sweep.last_sequence_index is not None
            first_sequence_index = (
                sweep.last_sequence_index - observation_count + 1
            )
            if first_sequence_index < 0:
                raise ValueError("first_sequence_index must be non-negative")
            if (
                self._last_sequence_index is not None
                and first_sequence_index <= self._last_sequence_index
            ):
                raise ValueError("sequence endpoints overlap an earlier sweep")
            last_sequence_index = sweep.last_sequence_index
        else:
            first_sequence_index = None
            last_sequence_index = None

        timestamp_available = sweep.last_timestamp_s is not None
        if (
            self._timestamp_available is not None
            and timestamp_available != self._timestamp_available
        ):
            raise ValueError("timestamp availability must remain consistent")
        if (
            timestamp_available
            and self._last_timestamp_s is not None
            and sweep.last_timestamp_s <= self._last_timestamp_s
        ):
            raise ValueError("completion timestamps must strictly increase")

        return _ProspectiveEndpoints(
            sequence_available=sequence_available,
            first_sequence_index=first_sequence_index,
            last_sequence_index=last_sequence_index,
            timestamp_available=timestamp_available,
            last_timestamp_s=sweep.last_timestamp_s,
        )

    def reset(self) -> None:
        """Clear causal state while retaining constructor settings."""
        self._history.clear()
        self._latest_success = None
        self._sequence_available = None
        self._last_sequence_index = None
        self._timestamp_available = None
        self._last_timestamp_s = None
        self._cumulative_observation_count = 0


__all__ = ["WarmStartedFullSweepEstimator"]
