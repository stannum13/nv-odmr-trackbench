"""Causal, row-major playback of immutable recorded sweep datasets."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from numbers import Real

import numpy as np

from odmr_bench.datasets.models import SweepDataset


@dataclass(frozen=True, slots=True)
class PlaybackObservation:
    """One recorded sample available to a causal playback estimator."""

    sequence_index: int
    sweep_index: int
    sample_index: int
    timestamp_s: float | None
    frequency_hz: float
    signal: float


def _validated_clock(nominal_clock_hz: float | None) -> float | None:
    if nominal_clock_hz is None:
        return None
    if isinstance(nominal_clock_hz, (bool, np.bool_)) or not isinstance(
        nominal_clock_hz, Real
    ):
        raise ValueError("nominal_clock_hz must be a finite positive real number")
    clock_hz = float(nominal_clock_hz)
    if not np.isfinite(clock_hz) or clock_hz <= 0.0:
        raise ValueError("nominal_clock_hz must be a finite positive real number")
    return clock_hz


def iter_playback_for_analysis(
    dataset: SweepDataset, nominal_clock_hz: float | None = None
) -> Generator[PlaybackObservation, None, None]:
    """Yield recorded samples for trusted evaluator-side analysis only.

    A timestamp is deliberately unavailable unless a caller supplies an explicit
    nominal clock assumption. Such timestamps are inferred sequence times, not
    measured acquisition timestamps.

    This generator retains the complete offline dataset in its frame and is not
    causally isolated. Never pass it to estimator-controlled code; use
    :func:`run_playback` for estimator callbacks.
    """
    clock_hz = _validated_clock(nominal_clock_hz)

    def observations() -> Generator[PlaybackObservation, None, None]:
        sequence_index = 0
        for sweep_index, sweep in enumerate(dataset.signal):
            for sample_index, signal in enumerate(sweep):
                timestamp_s = (
                    None if clock_hz is None else sequence_index / clock_hz
                )
                yield PlaybackObservation(
                    sequence_index=sequence_index,
                    sweep_index=sweep_index,
                    sample_index=sample_index,
                    timestamp_s=timestamp_s,
                    frequency_hz=float(dataset.frequency_hz[sample_index]),
                    signal=float(signal),
                )
                sequence_index += 1

    return observations()


def run_playback(
    dataset: SweepDataset,
    on_observation: Callable[[PlaybackObservation], None],
    nominal_clock_hz: float | None = None,
) -> None:
    """Call estimator code with exactly one frozen observation at a time.

    Iteration remains evaluator-owned, so normal estimator interfaces receive
    only ``PlaybackObservation`` values. Python callbacks are not a security
    sandbox against deliberate stack introspection; use process isolation for
    adversarial estimator code.
    """
    for observation in iter_playback_for_analysis(dataset, nominal_clock_hz):
        on_observation(observation)


__all__ = [
    "PlaybackObservation",
    "iter_playback_for_analysis",
    "run_playback",
]
