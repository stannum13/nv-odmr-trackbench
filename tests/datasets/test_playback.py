"""Tests for causal row-major playback of recorded raw sweeps."""

from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

import odmr_bench.datasets as datasets
from odmr_bench.datasets import (
    PlaybackObservation,
    SweepDataset,
    iter_playback_for_analysis,
    run_playback,
)


@pytest.fixture
def dataset() -> SweepDataset:
    return SweepDataset(
        signal=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        frequency_hz=np.array([10.0, 20.0, 30.0]),
    )


def test_playback_yields_row_major_observations_without_timestamps(
    dataset: SweepDataset,
) -> None:
    observations = list(iter_playback_for_analysis(dataset))

    assert [item.sequence_index for item in observations] == list(range(6))
    assert [item.sweep_index for item in observations] == [0, 0, 0, 1, 1, 1]
    assert [item.sample_index for item in observations] == [0, 1, 2, 0, 1, 2]
    assert [item.frequency_hz for item in observations] == [10.0, 20.0, 30.0] * 2
    assert [item.signal for item in observations] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert [item.timestamp_s for item in observations] == [None] * 6


def test_playback_uses_explicit_nominal_clock_only_when_supplied(
    dataset: SweepDataset,
) -> None:
    observations = list(iter_playback_for_analysis(dataset, nominal_clock_hz=2.0))

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
        iter_playback_for_analysis(dataset, nominal_clock_hz=clock_hz)


def test_causal_runner_passes_only_one_frozen_observation_per_callback(
    dataset: SweepDataset,
) -> None:
    received: list[PlaybackObservation] = []

    run_playback(dataset, received.append)

    assert [observation.sequence_index for observation in received] == list(range(6))
    assert {field.name for field in fields(received[0])} == {
        "sequence_index",
        "sweep_index",
        "sample_index",
        "timestamp_s",
        "frequency_hz",
        "signal",
    }
    assert not hasattr(received[0], "__dict__")
    assert not any(
        callable(getattr(received[0], field.name)) for field in fields(received[0])
    )


def test_analysis_iterator_is_explicitly_named_and_not_estimator_facing(
    dataset: SweepDataset,
) -> None:
    iterator = iter_playback_for_analysis(dataset)

    assert inspect.isgenerator(iterator)
    assert not hasattr(datasets, "iter_playback")
