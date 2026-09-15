"""Reset precedence and rollback tests for the sparse composite tracker."""

from __future__ import annotations

from copy import copy
from dataclasses import replace

import pytest

from odmr_bench.estimators import (
    SparseLinewidthCompositeTracker,
    SparseLinewidthConfiguration,
    SparseLinewidthResetError,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointRunMetadata,
    calibrate_two_point,
)
from odmr_bench.estimators import sparse_linewidth_tracker as tracker_module
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_tracker_configuration,
)


def _calibration(*, included: bool = False) -> TwoPointCalibration:
    source = make_legal_caller_asserted_source()
    if included:
        object.__setattr__(source, "provenance", "verified_factory_acquisition")
    return calibrate_two_point(
        source,
        make_legal_tracker_configuration(),
        budget_treatment=(
            "included_same_run" if included else "conditional_free_precalibration"
        ),
    )


def _metadata(
    calibration: TwoPointCalibration, *, included: bool = False
) -> TwoPointRunMetadata:
    return TwoPointRunMetadata(
        tracker_clock_id=calibration.source.clock_mapping.tracker_clock_id,
        current_sequence_index=(
            calibration.source.availability_sequence_index if included else None
        ),
        current_timestamp_s=(
            calibration.source.availability_timestamp_s if included else 0.020
        ),
        nominal_photon_rate_hz=(
            calibration.source.fluorescence_provenance.nominal_photon_rate_hz
        ),
        frequency_overhead_s=(
            calibration.source.source_frequency_overhead_s if included else 0.001
        ),
        fluorescence_quantity="normalized_fluorescence",
    )


def _valid_tracker() -> SparseLinewidthCompositeTracker:
    calibration = _calibration()
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    tracker.reset(
        _metadata(calibration),
        calibration,
        TwoPointBudgetCeiling(100, None, None, None),
        seed=7,
    )
    assert tracker.choose_next_query() is not None
    return tracker


def _snapshot(tracker: SparseLinewidthCompositeTracker) -> tuple[object, ...]:
    return tracker._configuration, tracker._configuration_snapshot, tracker._state


def _corrupt_calibration(calibration: TwoPointCalibration) -> TwoPointCalibration:
    corrupted = copy(calibration)
    object.__setattr__(corrupted, "identities", tuple(reversed(calibration.identities)))
    return corrupted


def _invalid_geometry_calibration(
    calibration: TwoPointCalibration,
) -> TwoPointCalibration:
    late = calibration.identities[-1]
    invalid = replace(
        late,
        calibration_cell_lower_hz=late.calibration_center_hz - 600_000.0,
        calibration_cell_upper_hz=late.calibration_center_hz + 600_000.0,
    )
    return replace(calibration, identities=(*calibration.identities[:-1], invalid))


def test_invalid_argument_type_precedes_every_later_reset_failure() -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _invalid_geometry_calibration(_calibration())

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(  # type: ignore[arg-type]
            object(),
            calibration,
            TwoPointBudgetCeiling(0, None, None, None),
            seed=-1,
        )

    assert raised.value.code == "invalid_argument_type"
    assert _snapshot(tracker) == before


def test_configuration_integrity_precedes_calibration_and_metadata() -> None:
    tracker = _valid_tracker()
    object.__setattr__(
        tracker,
        "_configuration",
        replace(tracker._configuration, scan_period_fast_pairs=7),
    )
    before = _snapshot(tracker)
    calibration = _corrupt_calibration(_calibration())

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            replace(_metadata(calibration), tracker_clock_id="wrong-clock"),
            calibration,
            TwoPointBudgetCeiling(0, None, None, None),
            seed=11,
        )

    assert raised.value.code == "configuration_mismatch"
    assert _snapshot(tracker) == before


def test_calibration_mismatch_precedes_metadata_geometry_and_budget() -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _corrupt_calibration(_invalid_geometry_calibration(_calibration()))

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            replace(_metadata(calibration), tracker_clock_id="wrong-clock"),
            calibration,
            TwoPointBudgetCeiling(0, None, None, None),
            seed=11,
        )

    assert raised.value.code == "calibration_mismatch"
    assert _snapshot(tracker) == before


def test_metadata_mismatch_precedes_geometry_and_budget() -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _invalid_geometry_calibration(_calibration())

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            replace(_metadata(calibration), tracker_clock_id="wrong-clock"),
            calibration,
            TwoPointBudgetCeiling(0, None, None, None),
            seed=11,
        )

    assert raised.value.code == "metadata_mismatch"
    assert _snapshot(tracker) == before


def test_geometry_mismatch_precedes_budget_and_rolls_back_value_equally() -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _invalid_geometry_calibration(_calibration())

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            _metadata(calibration),
            calibration,
            TwoPointBudgetCeiling(0, None, None, None),
            seed=11,
        )

    assert raised.value.code == "invalid_base_sparse_geometry"
    assert _snapshot(tracker) == before


def test_starting_budget_mismatch_rolls_back_value_equally() -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _calibration(included=True)

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            _metadata(calibration, included=True),
            calibration,
            TwoPointBudgetCeiling(1, None, None, None),
            seed=11,
        )

    assert raised.value.code == "budget_mismatch"
    assert _snapshot(tracker) == before


@pytest.mark.parametrize(
    "constructor_name",
    ("CompositeIdentityEstimate", "SparseLinewidthCompositeEstimate"),
)
def test_initial_state_construction_failure_rolls_back_value_equally(
    monkeypatch: pytest.MonkeyPatch, constructor_name: str
) -> None:
    tracker = _valid_tracker()
    before = _snapshot(tracker)
    calibration = _calibration()

    def fail_construction(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(f"injected {constructor_name} failure")

    monkeypatch.setattr(tracker_module, constructor_name, fail_construction)

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            _metadata(calibration),
            calibration,
            TwoPointBudgetCeiling(100, None, None, None),
            seed=11,
        )

    assert raised.value.code == "initial_state_construction_failed"
    assert raised.value.__cause__ is not None
    assert _snapshot(tracker) == before


def test_budget_stop_record_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration()
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    tracker.reset(
        _metadata(calibration),
        calibration,
        TwoPointBudgetCeiling(1, None, None, None),
        seed=13,
    )
    before = _snapshot(tracker)

    def fail_construction(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected stop construction failure")

    monkeypatch.setattr(
        tracker_module, "SparseLinewidthCompositeEstimate", fail_construction
    )

    with pytest.raises(RuntimeError, match="stop construction"):
        tracker.choose_next_query()

    assert _snapshot(tracker) == before
