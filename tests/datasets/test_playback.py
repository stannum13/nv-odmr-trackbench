"""Tests for causal row-major playback of recorded raw sweeps."""

from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

from odmr_bench.datasets import SweepDataset, iter_playback


@pytest.fixture
def dataset() -> SweepDataset:
    return SweepDataset(
        signal=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        frequency_hz=np.array([10.0, 20.0, 30.0]),
    )


def test_playback_yields_row_major_observations_without_timestamps(
    dataset: SweepDataset,
) -> None:
    observations = list(iter_playback(dataset))

    assert [item.sequence_index for item in observations] == list(range(6))
    assert [item.sweep_index for item in observations] == [0, 0, 0, 1, 1, 1]
    assert [item.sample_index for item in observations] == [0, 1, 2, 0, 1, 2]
    assert [item.frequency_hz for item in observations] == [10.0, 20.0, 30.0] * 2
    assert [item.signal for item in observations] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert [item.timestamp_s for item in observations] == [None] * 6


def test_playback_uses_explicit_nominal_clock_only_when_supplied(
    dataset: SweepDataset,
) -> None:
    observations = list(iter_playback(dataset, nominal_clock_hz=2.0))

    assert [item.timestamp_s for item in observations] == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
    ]


@pytest.mark.parametrize("clock_hz", [0.0, -1.0, np.nan, np.inf])
def test_playback_rejects_invalid_nominal_clock_assumptions(
    dataset: SweepDataset, clock_hz: float
) -> None:
    with pytest.raises(ValueError, match="nominal_clock_hz"):
        iter_playback(dataset, nominal_clock_hz=clock_hz)


def test_first_playback_observation_does_not_expose_future_values(
    dataset: SweepDataset,
) -> None:
    iterator = iter_playback(dataset)
    observation = next(iterator)

    assert inspect.isgenerator(iterator)
    assert {field.name for field in fields(observation)} == {
        "sequence_index",
        "sweep_index",
        "sample_index",
        "timestamp_s",
        "frequency_hz",
        "signal",
    }
    assert not hasattr(observation, "__dict__")
    assert not any(
        callable(getattr(observation, field.name)) for field in fields(observation)
    )
