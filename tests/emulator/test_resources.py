from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from odmr_bench.emulator.resources import ResourceLedger, ResourceSnapshot


def test_resource_ledger_accumulates_all_acquisition_dimensions() -> None:
    ledger = ResourceLedger()
    ledger.record(
        integration_time_s=0.2,
        nominal_exposure_photons=2_000.0,
        expected_photons=1_900.0,
        realized_photons=1_876,
        virtual_elapsed_time_s=0.25,
    )
    ledger.record(
        integration_time_s=0.1,
        nominal_exposure_photons=1_000.0,
        expected_photons=980.0,
        realized_photons=None,
        virtual_elapsed_time_s=0.15,
    )

    snapshot = ledger.snapshot()
    assert snapshot.observations == 2
    assert snapshot.integration_time_s == pytest.approx(0.3)
    assert snapshot.nominal_exposure_photons == pytest.approx(3_000.0)
    assert snapshot.expected_photons == pytest.approx(2_880.0)
    assert snapshot.realized_photons == 1_876
    assert snapshot.observations_without_realized_counts == 1
    assert snapshot.virtual_elapsed_time_s == pytest.approx(0.4)


def test_resource_snapshot_is_immutable() -> None:
    snapshot = ResourceLedger().snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.observations = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observations": -1},
        {"observations": True},
        {"integration_time_s": float("nan")},
        {"nominal_exposure_photons": float("inf")},
        {"expected_photons": -0.1},
        {"realized_photons": -1},
        {"observations_without_realized_counts": 1.5},
        {"virtual_elapsed_time_s": float("-inf")},
        {"observations": 0, "observations_without_realized_counts": 1},
        {"integration_time_s": 0.2, "virtual_elapsed_time_s": 0.1},
    ],
)
def test_resource_snapshot_rejects_invalid_public_totals(
    kwargs: dict[str, float | int | bool],
) -> None:
    values: dict[str, float | int | bool] = {
        "observations": 0,
        "integration_time_s": 0.0,
        "nominal_exposure_photons": 0.0,
        "expected_photons": 0.0,
        "realized_photons": 0,
        "observations_without_realized_counts": 0,
        "virtual_elapsed_time_s": 0.0,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        ResourceSnapshot(**values)


def test_resource_snapshot_canonicalizes_valid_numeric_scalars() -> None:
    snapshot = ResourceSnapshot(
        observations=np.int64(1),
        integration_time_s=np.float64(0.1),
        nominal_exposure_photons=np.float64(2.0),
        expected_photons=np.float64(1.9),
        realized_photons=np.int64(2),
        observations_without_realized_counts=np.int64(0),
        virtual_elapsed_time_s=np.float64(0.2),
    )

    assert type(snapshot.observations) is int
    assert type(snapshot.integration_time_s) is float
    assert type(snapshot.realized_photons) is int


@pytest.mark.parametrize(
    "kwargs",
    [
        {"integration_time_s": 0.0},
        {"nominal_exposure_photons": -0.1},
        {"expected_photons": float("nan")},
        {"realized_photons": -1},
        {"realized_photons": 1.5},
        {"virtual_elapsed_time_s": 0.01},
    ],
)
def test_invalid_commit_leaves_resource_ledger_unchanged(
    kwargs: dict[str, float | int],
) -> None:
    ledger = ResourceLedger()
    before = ledger.snapshot()
    values: dict[str, float | int | None] = {
        "integration_time_s": 0.1,
        "nominal_exposure_photons": 10.0,
        "expected_photons": 9.0,
        "realized_photons": 8,
        "virtual_elapsed_time_s": 0.2,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        ledger.record(**values)

    assert ledger.snapshot() == before


def test_overflowing_prospective_totals_leave_resource_ledger_unchanged() -> None:
    ledger = ResourceLedger()
    values = {
        "integration_time_s": 1e308,
        "nominal_exposure_photons": 1e308,
        "expected_photons": 1e308,
        "realized_photons": 1,
        "virtual_elapsed_time_s": 1e308,
    }
    ledger.record(**values)
    before = ledger.snapshot()

    with pytest.raises(ValueError):
        ledger.record(**values)

    assert ledger.snapshot() == before
