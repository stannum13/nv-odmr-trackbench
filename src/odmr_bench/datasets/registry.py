"""Checked registry entries for optional external datasets."""

from __future__ import annotations

from odmr_bench.datasets.models import DatasetRecord

FIGSHARE_28788437_V1 = DatasetRecord(
    dataset_id="figshare-28788437-v1",
    title="20241224-1651-50_ODMR_data_ch0_raw.dat",
    doi="10.6084/m9.figshare.28788437.v1",
    version=1,
    authors=("Liu",),
    license_name="CC BY 4.0",
    license_spdx="CC-BY-4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    canonical_url=(
        "https://figshare.com/articles/dataset/"
        "20241224-1651-50_ODMR_data_ch0_raw_dat/28788437"
    ),
    filename="20241224-1651-50_ODMR_data_ch0_raw.dat",
    download_url="https://ndownloader.figshare.com/files/53646563",
    byte_size=18_974_276,
    checksum_algorithm="md5",
    checksum_value="df03ef2385cdd64d2f0e117ecd9d6c7e",
    shape=(4_693, 311),
    frequency_start_hz=2_740_000_000.0,
    frequency_stop_hz=3_050_000_000.0,
    frequency_step_hz=1_000_000.0,
    frequency_count=311,
    declared_signal_label="count data (counts/s)",
    signal_quantity="unknown_analog_signal",
    unit_status="conflicted_unverified",
    detector_channel="/Dev1/AI0",
    nominal_clock_hz=200.0,
    timing_status="nominal_without_timestamps",
    missing_metadata=(
        "per_sweep_timestamps",
        "current",
        "field",
        "field_direction",
        "resonance_trajectory",
    ),
    ground_truth_status="none",
)


__all__ = ["FIGSHARE_28788437_V1"]
