"""Tests for verified dataset provenance and immutable raw sweep data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from odmr_bench.datasets import FIGSHARE_28788437_V1, SweepDataset


def test_figshare_record_has_exact_verified_public_metadata() -> None:
    record = FIGSHARE_28788437_V1

    assert record.dataset_id == "figshare-28788437-v1"
    assert record.doi == "10.6084/m9.figshare.28788437.v1"
    assert record.version == 1
    assert record.authors == ("Liu",)
    assert record.license_name == "CC BY 4.0"
    assert record.license_spdx == "CC-BY-4.0"
    assert record.license_url == "https://creativecommons.org/licenses/by/4.0/"
    assert record.checksum_algorithm == "md5"
    assert record.checksum_value == "df03ef2385cdd64d2f0e117ecd9d6c7e"
    assert record.shape == (4693, 311)
    assert record.frequency_start_hz == 2_740_000_000.0
    assert record.frequency_stop_hz == 3_050_000_000.0
    assert record.frequency_step_hz == 1_000_000.0
    assert record.frequency_count == 311
    assert record.declared_signal_label == "count data (counts/s)"
    assert record.signal_quantity == "unknown_analog_signal"
    assert record.unit_status == "conflicted_unverified"
    assert record.detector_channel == "/Dev1/AI0"
    assert record.nominal_clock_hz == 200.0
    assert record.timing_status == "nominal_without_timestamps"
    assert record.ground_truth_status == "none"


def test_registry_yaml_matches_the_checked_figshare_record() -> None:
    registry_path = Path(__file__).parents[2] / "datasets" / "registry.yaml"
    parsed = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    public = parsed["datasets"][0]
    record = FIGSHARE_28788437_V1

    assert public["dataset_id"] == record.dataset_id
    assert public["title"] == record.title
    assert public["doi"] == record.doi
    assert public["version"] == record.version
    assert tuple(public["authors"]) == record.authors
    assert public["license_name"] == record.license_name
    assert public["license_spdx"] == record.license_spdx
    assert public["license_url"] == record.license_url
    assert public["canonical_url"] == record.canonical_url
    assert public["filename"] == record.filename
    assert public["download_url"] == record.download_url
    assert public["byte_size"] == record.byte_size
    assert public["checksum_algorithm"] == record.checksum_algorithm
    assert public["checksum_value"] == record.checksum_value
    assert tuple(public["shape"]) == record.shape
    assert public["frequency"] == {
        "start_hz": record.frequency_start_hz,
        "stop_hz": record.frequency_stop_hz,
        "step_hz": record.frequency_step_hz,
        "count": record.frequency_count,
    }
    assert public["declared_signal_label"] == record.declared_signal_label
    assert public["signal_quantity"] == record.signal_quantity
    assert public["unit_status"] == record.unit_status
    assert public["detector_channel"] == record.detector_channel
    assert public["nominal_clock_hz"] == record.nominal_clock_hz
    assert public["timing_status"] == record.timing_status
    assert tuple(public["missing_metadata"]) == record.missing_metadata
    assert public["ground_truth_status"] == record.ground_truth_status


def test_sweep_dataset_copies_and_freezes_caller_arrays() -> None:
    signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    frequency_hz = np.array([10.0, 20.0, 30.0])

    dataset = SweepDataset(signal=signal, frequency_hz=frequency_hz)
    signal[0, 0] = 99.0
    frequency_hz[0] = 99.0

    np.testing.assert_array_equal(
        dataset.signal, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    np.testing.assert_array_equal(dataset.frequency_hz, np.array([10.0, 20.0, 30.0]))
    assert not dataset.signal.flags.writeable
    assert not dataset.frequency_hz.flags.writeable
    assert not dataset.sweep_indices.flags.writeable
    assert not dataset.sample_indices.flags.writeable
    with pytest.raises(ValueError):
        dataset.signal[0, 0] = 0.0
    with pytest.raises(ValueError):
        dataset.frequency_hz[0] = 0.0


@pytest.mark.parametrize(
    ("signal", "frequency_hz", "message"),
    [
        (np.array([1.0, 2.0]), np.array([10.0, 20.0]), "two-dimensional"),
        (
            np.array([[1.0, np.nan]]),
            np.array([10.0, 20.0]),
            "finite",
        ),
        (np.array([[1.0, 2.0]]), np.array([10.0]), "length"),
        (np.array([[1.0, 2.0]]), np.array([20.0, 10.0]), "increasing"),
        (np.empty((0, 3)), np.array([10.0, 20.0, 30.0]), "at least one"),
        (np.empty((2, 0)), np.array([]), "at least one"),
    ],
)
def test_sweep_dataset_rejects_invalid_arrays(
    signal: np.ndarray, frequency_hz: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SweepDataset(signal=signal, frequency_hz=frequency_hz)
