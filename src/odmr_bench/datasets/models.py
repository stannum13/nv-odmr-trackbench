"""Immutable raw sweep datasets and their checked provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _readonly_float_array(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Checked, version-specific provenance for one external data file."""

    dataset_id: str
    title: str
    doi: str
    version: int
    authors: tuple[str, ...]
    license_name: str
    license_spdx: str
    license_url: str
    canonical_url: str
    filename: str
    download_url: str
    byte_size: int
    checksum_algorithm: str
    checksum_value: str
    shape: tuple[int, int]
    frequency_start_hz: float
    frequency_stop_hz: float
    frequency_step_hz: float
    frequency_count: int
    declared_signal_label: str
    signal_quantity: str
    unit_status: str
    detector_channel: str
    nominal_clock_hz: float
    timing_status: str
    missing_metadata: tuple[str, ...]
    ground_truth_status: str

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be nonempty")
        if self.version < 1:
            raise ValueError("version must be positive")
        if len(self.shape) != 2 or any(size < 1 for size in self.shape):
            raise ValueError("shape must contain two positive dimensions")
        if self.frequency_count != self.shape[1]:
            raise ValueError("frequency_count must match the signal width")
        grid = np.linspace(
            self.frequency_start_hz,
            self.frequency_stop_hz,
            self.frequency_count,
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(grid))
            or self.frequency_step_hz <= 0.0
            or not np.allclose(
                np.diff(grid), self.frequency_step_hz, rtol=0.0, atol=0.0
            )
        ):
            raise ValueError("frequency metadata must define an inclusive regular grid")


@dataclass(frozen=True, slots=True)
class SweepDataset:
    """An immutable, ordered matrix of unnormalised recorded sweep values."""

    signal: ArrayLike
    frequency_hz: ArrayLike
    declared_metadata: Mapping[str, str] = field(default_factory=dict)
    record: DatasetRecord | None = None
    sweep_indices: NDArray[np.intp] = field(init=False, repr=False)
    sample_indices: NDArray[np.intp] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        signal = _readonly_float_array(self.signal)
        frequency_hz = _readonly_float_array(self.frequency_hz)
        if signal.ndim != 2:
            raise ValueError("signal must be two-dimensional")
        if frequency_hz.ndim != 1:
            raise ValueError("frequency_hz must be one-dimensional")
        if signal.shape[1] != frequency_hz.size:
            raise ValueError("frequency_hz length must match the signal width")
        if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(frequency_hz)):
            raise ValueError("signal and frequency_hz must be finite")
        if not np.all(np.diff(frequency_hz) > 0.0):
            raise ValueError("frequency_hz must be strictly increasing")

        sweep_indices = np.arange(signal.shape[0], dtype=np.intp)
        sample_indices = np.arange(signal.shape[1], dtype=np.intp)
        sweep_indices.setflags(write=False)
        sample_indices.setflags(write=False)
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        metadata = MappingProxyType(dict(self.declared_metadata))
        object.__setattr__(self, "declared_metadata", metadata)
        object.__setattr__(self, "sweep_indices", sweep_indices)
        object.__setattr__(self, "sample_indices", sample_indices)
