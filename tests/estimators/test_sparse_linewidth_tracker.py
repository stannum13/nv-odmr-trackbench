"""Reset and initial reservation tests for the sparse composite tracker."""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from odmr_bench.estimators import (
    PublicAcquisitionResources,
    SparseLinewidthCompositeTracker,
    SparseLinewidthConfiguration,
    SparseLinewidthResetError,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointQuery,
    TwoPointRunMetadata,
    calibrate_two_point,
)
from odmr_bench.estimators import sparse_linewidth_tracker as tracker_module
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_tracker_configuration,
)


def _calibration(
    *, included: bool = False, offset_s: float = 0.0
) -> TwoPointCalibration:
    source = make_legal_caller_asserted_source()
    if included:
        object.__setattr__(source, "provenance", "verified_factory_acquisition")
    if offset_s:
        mapping = replace(
            source.clock_mapping,
            kind="unit_scale_offset",
            source_clock_id="source-clock",
            tracker_clock_id="tracker-clock",
            offset_s=offset_s,
        )
        source = replace(source, clock_mapping=mapping)
    configuration = make_legal_tracker_configuration()
    return calibrate_two_point(
        source,
        configuration,
        budget_treatment=(
            "included_same_run" if included else "conditional_free_precalibration"
        ),
    )


def _metadata(
    calibration: TwoPointCalibration,
    *, included: bool = False,
    current_timestamp_s: float = 0.020,
    frequency_overhead_s: float = 0.001,
) -> TwoPointRunMetadata:
    if included:
        return TwoPointRunMetadata(
            tracker_clock_id=calibration.source.clock_mapping.tracker_clock_id,
            current_sequence_index=calibration.source.availability_sequence_index,
            current_timestamp_s=(
                calibration.source.availability_timestamp_s
                + calibration.source.clock_mapping.offset_s
            ),
            nominal_photon_rate_hz=(
                calibration.source.fluorescence_provenance.nominal_photon_rate_hz
            ),
            frequency_overhead_s=calibration.source.source_frequency_overhead_s,
            fluorescence_quantity="normalized_fluorescence",
        )
    return TwoPointRunMetadata(
        tracker_clock_id=calibration.source.clock_mapping.tracker_clock_id,
        current_sequence_index=None,
        current_timestamp_s=current_timestamp_s,
        nominal_photon_rate_hz=2.5e6,
        frequency_overhead_s=frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )


def _reset_tracker(
    *,
    configuration: SparseLinewidthConfiguration | None = None,
    calibration: TwoPointCalibration | None = None,
    metadata: TwoPointRunMetadata | None = None,
    ceiling: TwoPointBudgetCeiling | None = None,
) -> SparseLinewidthCompositeTracker:
    configuration = configuration or SparseLinewidthConfiguration()
    calibration = calibration or _calibration()
    metadata = metadata or _metadata(calibration)
    ceiling = ceiling or TwoPointBudgetCeiling(100, None, None, None)
    tracker = SparseLinewidthCompositeTracker(configuration)
    tracker.reset(metadata, calibration, ceiling, seed=23)
    return tracker


def test_constructor_and_pre_reset_surface_are_closed() -> None:
    configuration = SparseLinewidthConfiguration()
    tracker = SparseLinewidthCompositeTracker(configuration)

    assert SparseLinewidthCompositeTracker.__final__ is True
    assert not hasattr(tracker, "__dict__")
    with pytest.raises(RuntimeError, match="reset"):
        tracker.choose_next_query()
    with pytest.raises(RuntimeError, match="reset"):
        tracker.estimate()
    with pytest.raises(TypeError):
        SparseLinewidthCompositeTracker(object())  # type: ignore[arg-type]


def test_reset_seeds_all_identity_values_and_independent_epochs() -> None:
    calibration = _calibration(offset_s=-0.010)
    metadata = _metadata(calibration, current_timestamp_s=0.020)
    tracker = _reset_tracker(calibration=calibration, metadata=metadata)

    estimate = tracker.estimate()
    mapped_fit_epoch_s = calibration.source.physical_fit_epoch_s - 0.010
    zero = PublicAcquisitionResources(0, 0.0, 0.0, 0, 0, 0.0)
    assert estimate.current_sequence_index is None
    assert estimate.current_timestamp_s == metadata.current_timestamp_s
    assert estimate.calibration_resources == calibration.source.safe_resources
    assert estimate.fast_tracking_resources == zero
    assert estimate.sparse_tracking_resources == zero
    assert estimate.tracking_resources == zero
    assert estimate.charged_resources == zero
    assert estimate.accepted_observations == 0
    assert estimate.completed_fast_pairs == 0
    assert estimate.completed_sparse_scans == 0
    assert estimate.fast_pairs_since_scan == 0
    assert estimate.pending_mode is None
    assert estimate.pending_query is None
    assert estimate.incomplete_fast_pair is None
    assert estimate.incomplete_sparse_scan is None
    assert estimate.fast_pair_history == ()
    assert estimate.sparse_scan_history == ()
    assert estimate.stopped_reason is None
    assert estimate.sparse_geometry_diagnostic is None
    assert estimate.fast_update_cpu_time_s == 0.0
    assert estimate.sparse_update_cpu_time_s == 0.0
    assert estimate.total_update_cpu_time_s == 0.0
    assert estimate.seed == 23

    for identity, cell in zip(
        estimate.identities, calibration.identities, strict=True
    ):
        assert identity.resonance_id == cell.resonance_id
        assert identity.fast_center_hz == cell.calibration_center_hz
        assert identity.fast_center_source_kind == "calibration"
        assert identity.fast_center_source_pair_index is None
        assert identity.fast_center_reference_timestamp_s == mapped_fit_epoch_s
        assert identity.fast_center_release_sequence_index is None
        assert identity.fast_center_release_timestamp_s == 0.0
        assert identity.active_fwhm_hz == cell.calibration_fwhm_hz
        assert identity.fwhm_source_kind == "calibration"
        assert identity.fwhm_source_scan_index is None
        assert identity.fwhm_reference_timestamp_s == mapped_fit_epoch_s
        assert identity.fwhm_release_sequence_index is None
        assert identity.fwhm_release_timestamp_s == 0.0
        assert identity.live_q == (
            cell.calibration_center_hz / cell.calibration_fwhm_hz
        )
        assert (
            identity.center_age_s
            == metadata.current_timestamp_s - mapped_fit_epoch_s
        )
        assert (
            identity.fwhm_age_s
            == metadata.current_timestamp_s - mapped_fit_epoch_s
        )
        assert identity.center_release_age_s == metadata.current_timestamp_s
        assert identity.fwhm_release_age_s == metadata.current_timestamp_s
        assert identity.completed_fast_pairs == 0
        assert identity.completed_sparse_scans == 0
        assert identity.latest_fast_pair is None
        assert identity.latest_sparse_scan is None


def test_included_reset_seeds_source_cost_and_availability_epochs() -> None:
    calibration = _calibration(included=True)
    metadata = _metadata(calibration, included=True)
    tracker = _reset_tracker(
        calibration=calibration,
        metadata=metadata,
        ceiling=TwoPointBudgetCeiling(100, None, None, None),
    )

    estimate = tracker.estimate()
    assert estimate.charged_resources == calibration.source.safe_resources
    for identity in estimate.identities:
        assert (
            identity.fast_center_reference_timestamp_s
            == calibration.source.physical_fit_epoch_s
        )
        assert (
            identity.fwhm_reference_timestamp_s
            == calibration.source.physical_fit_epoch_s
        )
        assert (
            identity.fast_center_release_sequence_index
            == calibration.source.availability_sequence_index
        )
        assert (
            identity.fwhm_release_sequence_index
            == calibration.source.availability_sequence_index
        )
        assert (
            identity.fast_center_release_timestamp_s
            == calibration.source.availability_timestamp_s
        )
        assert (
            identity.fwhm_release_timestamp_s
            == calibration.source.availability_timestamp_s
        )
        assert identity.center_release_age_s == 0.0
        assert identity.fwhm_release_age_s == 0.0


def test_reset_prospectively_checks_the_late_identity_geometry() -> None:
    calibration = _calibration()
    late = calibration.identities[-1]
    narrow_late = replace(
        late,
        calibration_cell_lower_hz=late.calibration_center_hz - 600_000.0,
        calibration_cell_upper_hz=late.calibration_center_hz + 600_000.0,
    )
    calibration = replace(
        calibration, identities=(*calibration.identities[:-1], narrow_late)
    )
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            _metadata(calibration),
            calibration,
            TwoPointBudgetCeiling(100, None, None, None),
            seed=23,
        )

    assert raised.value.code == "invalid_base_sparse_geometry"
    assert "calibration_cell_violation" in str(raised.value)
    with pytest.raises(RuntimeError, match="reset"):
        tracker.estimate()


def test_reset_exposes_only_the_first_reserved_fast_query() -> None:
    tracker = _reset_tracker()

    query = tracker.choose_next_query()

    assert type(query) is TwoPointQuery
    assert (query.pair_index, query.resonance_id) == (0, "r0")
    assert query.query_index == 0
    assert query.identity_pair_index == 0
    assert query.side == "minus"


def test_initial_fast_reservation_is_idempotent_and_does_not_charge_resources() -> None:
    tracker = _reset_tracker()
    before = tracker.estimate()

    first = tracker.choose_next_query()
    second = tracker.choose_next_query()
    after = tracker.estimate()

    assert first is second
    assert after.pending_query is first
    assert after.pending_mode == "fast_pair"
    assert after.charged_resources == before.charged_resources
    assert after.tracking_resources == before.tracking_resources
    assert after.current_sequence_index == before.current_sequence_index
    assert after.current_timestamp_s == before.current_timestamp_s
    assert after.accepted_observations == 0


def _initial_block_totals(
    calibration: TwoPointCalibration, metadata: TwoPointRunMetadata
) -> tuple[float, float, float]:
    integration_time_s = calibration.configuration.integration_time_s
    integration_total_s = (0.0 + integration_time_s) + integration_time_s
    nominal_atom = metadata.nominal_photon_rate_hz * integration_time_s
    nominal_total = (0.0 + nominal_atom) + nominal_atom
    elapsed_atom_s = metadata.frequency_overhead_s + integration_time_s
    elapsed_total_s = (0.0 + elapsed_atom_s) + elapsed_atom_s
    return integration_total_s, nominal_total, elapsed_total_s


@pytest.mark.parametrize(
    "ceiling_factory",
    (
        pytest.param(
            lambda totals: TwoPointBudgetCeiling(2, None, None, None),
            id="observation",
        ),
        pytest.param(
            lambda totals: TwoPointBudgetCeiling(None, totals[0], None, None),
            id="integration",
        ),
        pytest.param(
            lambda totals: TwoPointBudgetCeiling(None, None, totals[1], None),
            id="nominal-exposure",
        ),
        pytest.param(
            lambda totals: TwoPointBudgetCeiling(None, None, None, totals[2]),
            id="elapsed",
        ),
        pytest.param(
            lambda totals: TwoPointBudgetCeiling(
                2, totals[0], totals[1], totals[2]
            ),
            id="all",
        ),
    ),
)
def test_initial_pair_is_affordable_at_every_exact_ceiling(ceiling_factory) -> None:
    calibration = _calibration()
    metadata = _metadata(calibration)
    totals = _initial_block_totals(calibration, metadata)
    tracker = _reset_tracker(
        calibration=calibration,
        metadata=metadata,
        ceiling=ceiling_factory(totals),
    )

    assert type(tracker.choose_next_query()) is TwoPointQuery
    assert tracker.estimate().stopped_reason is None


@pytest.mark.parametrize(
    "ceiling_factory",
    (
        pytest.param(
            lambda totals: TwoPointBudgetCeiling(1, None, None, None),
            id="observation",
        ),
        pytest.param(
            lambda totals: TwoPointBudgetCeiling(
                None, math.nextafter(totals[0], -math.inf), None, None
            ),
            id="integration",
        ),
        pytest.param(
            lambda totals: TwoPointBudgetCeiling(
                None, None, math.nextafter(totals[1], -math.inf), None
            ),
            id="nominal-exposure",
        ),
        pytest.param(
            lambda totals: TwoPointBudgetCeiling(
                None, None, None, math.nextafter(totals[2], -math.inf)
            ),
            id="elapsed",
        ),
    ),
)
def test_unaffordable_initial_pair_stops_atomically_without_a_query(
    ceiling_factory,
) -> None:
    calibration = _calibration()
    metadata = _metadata(calibration)
    totals = _initial_block_totals(calibration, metadata)
    tracker = _reset_tracker(
        calibration=calibration,
        metadata=metadata,
        ceiling=ceiling_factory(totals),
    )
    before = tracker.estimate()

    assert tracker.choose_next_query() is None
    stopped = tracker.estimate()

    assert stopped == replace(before, stopped_reason="budget_exhausted")
    assert stopped.pending_query is None
    assert stopped.pending_mode is None
    assert stopped.sparse_geometry_diagnostic is None
    assert tracker.choose_next_query() is None
    assert tracker.estimate() is stopped


def test_two_and_five_atom_helpers_are_literal_sequential_left_folds() -> None:
    atom = 2.0**-53
    start = PublicAcquisitionResources(0, 1.0, 1.0, 0, 0, 1.0)
    metadata = SimpleNamespace(frequency_overhead_s=0.0)
    queries = tuple(
        SimpleNamespace(
            integration_time_s=atom,
            expected_nominal_exposure_photons=atom,
        )
        for _ in range(5)
    )

    after_two = tracker_module._reserve_two_atom_block(
        start, metadata, queries[:2]
    )
    after_five = tracker_module._reserve_five_atom_block(start, metadata, queries)

    expected_two = (1.0 + atom) + atom
    expected_five = 1.0
    for _ in range(5):
        expected_five = expected_five + atom
    assert after_two.observations == 2
    assert after_two.integration_time_s == expected_two
    assert after_two.nominal_exposure_photons == expected_two
    assert after_two.virtual_elapsed_time_s == expected_two
    assert after_five.observations == 5
    assert after_five.integration_time_s == expected_five
    assert after_five.nominal_exposure_photons == expected_five
    assert after_five.virtual_elapsed_time_s == expected_five
    assert expected_two != 1.0 + 2.0 * atom
    assert expected_five != 1.0 + 5.0 * atom

    for helper in (
        tracker_module._reserve_two_atom_block,
        tracker_module._reserve_five_atom_block,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(helper)))
        assert not any(
            isinstance(node, (ast.Mult, ast.Sub)) for node in ast.walk(tree)
        )
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "sum" not in called_names


def test_reset_accepts_numpy_integer_seed_and_canonicalizes_it() -> None:
    calibration = _calibration()
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    tracker.reset(
        _metadata(calibration),
        calibration,
        TwoPointBudgetCeiling(2, None, None, None),
        seed=np.int64(17),
    )

    assert tracker.estimate().seed == 17
    assert type(tracker.estimate().seed) is int


def test_initial_fast_shell_uses_the_calibration_fast_policy() -> None:
    calibration = _calibration()
    fast_configuration = replace(
        calibration.configuration, integration_time_s=0.007
    )
    calibration = replace(calibration, configuration=fast_configuration)
    metadata = _metadata(calibration)
    tracker = _reset_tracker(calibration=calibration, metadata=metadata)

    query = tracker.choose_next_query()

    assert type(query) is TwoPointQuery
    assert query.integration_time_s == 0.007
    assert (
        query.integration_time_s
        != tracker.estimate().configuration.integration_time_s
    )


def test_reset_does_not_retain_or_wrap_a_stage_63_tracker() -> None:
    tracker = _reset_tracker()

    assert "CalibratedTwoPointTracker" not in inspect.getsource(tracker_module)
    assert not any(
        type(getattr(tracker, slot)).__name__ == "CalibratedTwoPointTracker"
        for slot in tracker.__slots__
    )
