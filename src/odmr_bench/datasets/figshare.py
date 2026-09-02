"""Explicit-path parsing and verification for the initial Figshare ODMR file."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from odmr_bench.datasets.models import DatasetRecord, SweepDataset
from odmr_bench.datasets.registry import FIGSHARE_28788437_V1

_SCALAR_HEADERS = {
    "Number of frequency sweeps (#)": "number_of_frequency_sweeps",
    "Clock Frequencies (Hz)": "nominal_clock_hz",
}
_GRID_HEADERS = {
    "Start Frequencies (Hz)": "start_frequency_hz",
    "Stop Frequencies (Hz)": "stop_frequency_hz",
    "Step sizes (Hz)": "step_size_hz",
}


def _header_value(comment: str) -> tuple[str, str] | None:
    content = comment[1:].strip()
    key, separator, value = content.partition(":")
    if not separator:
        return None
    return key.strip(), value.strip()


def _single_bracketed_float(value: str, field: str) -> float:
    match = re.fullmatch(r"\[\s*([^\[\],]+)\s*\]", value)
    if match is None:
        raise ValueError(f"{field} must be a one-element bracketed list")
    try:
        number = float(match.group(1))
    except ValueError as error:
        raise ValueError(f"{field} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _regular_frequency_axis(
    start_hz: float, stop_hz: float, step_hz: float, frequency_count: int
) -> np.ndarray:
    if step_hz <= 0.0:
        raise ValueError("step_size_hz must be positive")
    intervals = (stop_hz - start_hz) / step_hz
    rounded_intervals = round(intervals)
    if (
        rounded_intervals < 0
        or not np.isclose(intervals, rounded_intervals, rtol=0.0, atol=1e-9)
        or frequency_count != rounded_intervals + 1
    ):
        raise ValueError("header does not define an inclusive frequency grid")
    return start_hz + step_hz * np.arange(frequency_count, dtype=np.float64)


def _parse_figshare_sweep_bytes(
    source_bytes: bytes, *, expected_shape: tuple[int, int] | None = None
) -> SweepDataset:
    """Parse one immutable Qudi-style file snapshot without reordering or scaling."""
    declared: dict[str, str] = {}
    data_rows: list[list[float]] = []
    signal_label: str | None = None

    try:
        lines = source_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("sweep file must be UTF-8 text") from error

    for line in lines:
        if line.startswith("#"):
            parsed = _header_value(line)
            if parsed is not None:
                key, value = parsed
                if key in _SCALAR_HEADERS:
                    declared[_SCALAR_HEADERS[key]] = value
                elif key in _GRID_HEADERS:
                    declared[_GRID_HEADERS[key]] = str(
                        _single_bracketed_float(value, key)
                    )
                elif key == "Channel":
                    _, _, channel = value.partition(":")
                    if not channel.strip():
                        raise ValueError("Channel must include a detector channel")
                    declared["detector_channel"] = channel.strip()
                elif key == "count data (counts/s)":
                    signal_label = key
            elif line[1:].strip() == "count data (counts/s)":
                signal_label = "count data (counts/s)"
            continue
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split("\t")]
        except ValueError as error:
            raise ValueError(
                "data cells must be finite floating-point values"
            ) from error
        if not values or not np.all(np.isfinite(values)):
            raise ValueError("data cells must be finite")
        data_rows.append(values)

    required = {
        "number_of_frequency_sweeps",
        "start_frequency_hz",
        "stop_frequency_hz",
        "step_size_hz",
        "nominal_clock_hz",
        "detector_channel",
    }
    missing = sorted(required - declared.keys())
    if missing:
        raise ValueError(f"missing documented header fields: {', '.join(missing)}")
    if signal_label is None:
        raise ValueError("missing declared signal label")
    if not data_rows:
        raise ValueError("sweep file has no data rows")
    width = len(data_rows[0])
    if any(len(row) != width for row in data_rows):
        raise ValueError("data rows must form a regular matrix")

    try:
        sweep_count = int(declared["number_of_frequency_sweeps"])
    except ValueError as error:
        raise ValueError("number_of_frequency_sweeps must be an integer") from error
    if sweep_count != len(data_rows):
        raise ValueError("header sweep count does not match data rows")
    start_hz = float(declared["start_frequency_hz"])
    stop_hz = float(declared["stop_frequency_hz"])
    step_hz = float(declared["step_size_hz"])
    frequency_hz = _regular_frequency_axis(start_hz, stop_hz, step_hz, width)
    signal = np.asarray(data_rows, dtype=np.float64)
    if expected_shape is not None and signal.shape != expected_shape:
        raise ValueError(f"expected shape {expected_shape}, got {signal.shape}")

    declared["declared_signal_label"] = signal_label
    return SweepDataset(
        signal=signal,
        frequency_hz=frequency_hz,
        declared_metadata=declared,
    )


def parse_figshare_sweep_file(
    path: str | Path, *, expected_shape: tuple[int, int] | None = None
) -> SweepDataset:
    """Parse a Qudi-style Figshare sweep file without reordering or scaling it."""
    source = Path(path)
    try:
        source_bytes = source.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read sweep file: {source}") from error
    return _parse_figshare_sweep_bytes(source_bytes, expected_shape=expected_shape)


def _load_verified_sweep_bytes(
    source_bytes: bytes, *, expected_record: DatasetRecord
) -> SweepDataset:
    digest = hashlib.new(
        expected_record.checksum_algorithm, source_bytes
    ).hexdigest()
    if digest != expected_record.checksum_value:
        raise ValueError("file checksum does not match Figshare record")
    if len(source_bytes) != expected_record.byte_size:
        raise ValueError("file size does not match Figshare record")
    parsed = _parse_figshare_sweep_bytes(
        source_bytes, expected_shape=expected_record.shape
    )
    expected_frequency_hz = (
        expected_record.frequency_start_hz
        + expected_record.frequency_step_hz
        * np.arange(expected_record.frequency_count, dtype=np.float64)
    )
    if not np.array_equal(parsed.frequency_hz, expected_frequency_hz):
        raise ValueError("file frequency grid does not match Figshare record")
    return SweepDataset(
        signal=parsed.signal,
        frequency_hz=parsed.frequency_hz,
        declared_metadata=parsed.declared_metadata,
        record=expected_record,
    )


def load_verified_sweep_file(
    path: str | Path, *, expected_record: DatasetRecord
) -> SweepDataset:
    """Verify and parse one local byte snapshot against an injected record."""
    source = Path(path)
    try:
        source_bytes = source.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read sweep file: {source}") from error
    return _load_verified_sweep_bytes(source_bytes, expected_record=expected_record)


def load_figshare_28788437(path: str | Path) -> SweepDataset:
    """Verify a local copy of Figshare article 28788437 version 1; never download."""
    return load_verified_sweep_file(path, expected_record=FIGSHARE_28788437_V1)


__all__ = [
    "load_figshare_28788437",
    "load_verified_sweep_file",
    "parse_figshare_sweep_file",
]
