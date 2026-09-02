from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from odmr_bench.emulator.resources import ResourceLedger


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
