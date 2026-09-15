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

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators import (
    CalibratedTwoPointTracker,
    PublicAcquisitionResources,
    SparseLinewidthCompositeTracker,
    SparseLinewidthConfiguration,
    SparseLinewidthQuery,
    SparseLinewidthResetError,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointQuery,
    TwoPointRunMetadata,
    calibrate_two_point,
)
from odmr_bench.estimators import sparse_linewidth_tracker as tracker_module
from odmr_bench.estimators import two_point_calibration as calibration_module
from odmr_bench.estimators.sparse_linewidth_fit import (
    _SparseGeometryConstructionError,
)
from odmr_bench.estimators.two_point_calibration import (
    _evaluate_target_only_model,
)
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_tracker_configuration,
)


def _calibration(
    *, included: bool = False, offset_s: float = 0.0
) -> TwoPointCalibration:
    source = make_legal_caller_asserted_source()
    if included:
        source = replace(
            source,
            fluorescence_provenance=replace(
                source.fluorescence_provenance,
                normalization_rule=(
                    "odmr_instrument_normalized_fluorescence_v1"
                ),
            ),
        )
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


def test_conditional_free_reset_retains_mapped_calibration_availability() -> None:
    calibration = _calibration(offset_s=-0.005)
    metadata = _metadata(calibration, current_timestamp_s=0.020)
    tracker = _reset_tracker(calibration=calibration, metadata=metadata)

    mapped_availability_s = (
        calibration.source.availability_timestamp_s
        + calibration.source.clock_mapping.offset_s
    )
    assert mapped_availability_s == 0.005
    for identity in tracker.estimate().identities:
        assert identity.fast_center_release_sequence_index is None
        assert identity.fast_center_release_timestamp_s == mapped_availability_s
        assert identity.fwhm_release_sequence_index is None
        assert identity.fwhm_release_timestamp_s == mapped_availability_s
        assert identity.center_release_age_s == 0.015
        assert identity.fwhm_release_age_s == 0.015


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


def test_reset_does_not_consume_verified_source_construction_authority() -> None:
    template = make_legal_caller_asserted_source()
    fluorescence_provenance = replace(
        template.fluorescence_provenance,
        normalization_rule="odmr_instrument_normalized_fluorescence_v1",
    )
    source = calibration_module._bind_verified_two_point_calibration_source(
        template.source_fit,
        template.fit_configuration,
        template.source_observations,
        template.identity_binding,
        fluorescence_provenance,
        source_id=template.source_id,
        source_frequency_overhead_s=template.source_frequency_overhead_s,
        source_start_timestamp_s=template.source_start_timestamp_s,
        physical_fit_epoch_s=template.physical_fit_epoch_s,
        availability_sequence_index=template.availability_sequence_index,
        availability_timestamp_s=template.availability_timestamp_s,
        clock_mapping=template.clock_mapping,
        construction_key=calibration_module._VERIFIED_SOURCE_CONSTRUCTION_KEY,
    )
    calibration = calibrate_two_point(
        source,
        make_legal_tracker_configuration(),
        budget_treatment="included_same_run",
    )
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())

    tracker.reset(
        _metadata(calibration, included=True),
        calibration,
        TwoPointBudgetCeiling(100, None, None, None),
        seed=23,
    )

    assert calibration_module._consume_verified_source_construction_identity(source)


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


def _fast_observation(
    calibration: TwoPointCalibration,
    query: TwoPointQuery,
    *,
    fluorescence: float | None = None,
    realized_photons: int | None = None,
) -> EstimatorObservation:
    cell = calibration.identities[query.pair_index % 8]
    value = (
        _evaluate_target_only_model(
            calibration.source.source_fit,
            cell.source_fit_index,
            query.frequency_hz,
            query.interrogation_center_hz,
        )
        if fluorescence is None
        else fluorescence
    )
    return EstimatorObservation(
        sequence_index=query.expected_sequence_index,
        timestamp_s=query.expected_end_timestamp_s,
        frequency_hz=query.frequency_hz,
        fluorescence=value,
        integration_time_s=query.integration_time_s,
        nominal_exposure_photons=query.expected_nominal_exposure_photons,
        realized_photons=realized_photons,
    )


def _accept_fast_pairs(
    tracker: SparseLinewidthCompositeTracker,
    calibration: TwoPointCalibration,
    *,
    count: int,
) -> None:
    for _ in range(count * 2):
        query = tracker.choose_next_query()
        assert type(query) is TwoPointQuery
        tracker.update(_fast_observation(calibration, query))


def test_fast_only_trace_is_stage_63_differentially_identical() -> None:
    calibration = _calibration()
    metadata = _metadata(calibration)
    ceiling = TwoPointBudgetCeiling(100, None, None, None)
    legacy = CalibratedTwoPointTracker(calibration.configuration)
    composite = SparseLinewidthCompositeTracker(
        SparseLinewidthConfiguration(scan_period_fast_pairs=24)
    )
    legacy.reset(metadata, calibration, ceiling, seed=29)
    composite.reset(metadata, calibration, ceiling, seed=29)

    for pair_index in range(20):
        for _ in range(2):
            legacy_query = legacy.choose_next_query()
            composite_query = composite.choose_next_query()
            assert type(legacy_query) is TwoPointQuery
            assert composite_query == legacy_query
            fluorescence = -1.0 if pair_index == 6 else None
            legacy_update = legacy.update(
                _fast_observation(
                    calibration,
                    legacy_query,
                    fluorescence=fluorescence,
                    realized_photons=pair_index,
                )
            )
            composite_update = composite.update(
                _fast_observation(
                    calibration,
                    composite_query,
                    fluorescence=fluorescence,
                    realized_photons=pair_index,
                )
            )
            assert composite_update.completed_fast_pair == legacy_update.completed_pair

    legacy_estimate = legacy.estimate()
    composite_estimate = composite.estimate()
    assert composite_estimate.fast_pair_history == legacy_estimate.pair_history
    assert composite_estimate.fast_tracking_resources == (
        legacy_estimate.tracking_resources
    )
    assert tuple(
        item.fast_center_hz for item in composite_estimate.identities
    ) == tuple(item.center_hz for item in legacy_estimate.identities)
    assert tuple(
        item.completed_fast_pairs for item in composite_estimate.identities
    ) == tuple(item.completed_pairs for item in legacy_estimate.identities)
    assert composite_estimate.fast_pair_history[6].lock_state == "lost"
    assert tuple(
        pair.first_side for pair in composite_estimate.fast_pair_history[8:12]
    ) == ("plus", "plus", "plus", "plus")


def test_first_fast_side_is_partial_and_second_refreshes_identity() -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    before = tracker.estimate()
    first_query = tracker.choose_next_query()
    assert type(first_query) is TwoPointQuery
    first_observation = _fast_observation(calibration, first_query)

    first_update = tracker.update(first_observation)

    assert first_update.query is first_query
    assert first_update.observation is first_observation
    assert first_update.completed_fast_pair is None
    assert first_update.estimate.incomplete_fast_pair is not None
    assert first_update.estimate.completed_fast_pairs == 0
    assert first_update.estimate.accepted_observations == 1
    assert first_update.estimate.fast_pairs_since_scan == 0
    assert first_update.estimate.fast_tracking_resources.observations == 1
    assert first_update.estimate.tracking_resources.observations == 1
    assert first_update.estimate.charged_resources.observations == 1
    assert first_update.estimate.sparse_tracking_resources == (
        before.sparse_tracking_resources
    )

    second_query = tracker.choose_next_query()
    assert type(second_query) is TwoPointQuery
    second_observation = _fast_observation(calibration, second_query)
    second_update = tracker.update(second_observation)
    completed = second_update.completed_fast_pair

    assert completed is not None
    assert completed.first_side == "minus"
    assert completed.lock_state == "tracking"
    assert second_update.estimate.incomplete_fast_pair is None
    assert second_update.estimate.completed_fast_pairs == 1
    assert second_update.estimate.fast_pairs_since_scan == 1
    identity = second_update.estimate.identities[0]
    assert identity.fast_center_source_kind == "pair"
    assert identity.fast_center_source_pair_index == 0
    assert identity.fast_center_reference_timestamp_s == (
        completed.pair_reference_timestamp_s
    )
    assert identity.fast_center_release_sequence_index == (
        completed.release_sequence_index
    )
    assert identity.fast_center_release_timestamp_s == completed.release_timestamp_s
    assert identity.active_fwhm_hz == before.identities[0].active_fwhm_hz
    assert identity.fwhm_source_kind == "calibration"


def test_eighth_fast_pair_selects_one_frozen_sparse_block() -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    _accept_fast_pairs(tracker, calibration, count=8)

    query = tracker.choose_next_query()

    assert isinstance(query, SparseLinewidthQuery)
    assert (query.scan_index, query.resonance_id, query.point_index) == (0, "r0", 0)
    estimate = tracker.estimate()
    identity = estimate.identities[0]
    assert estimate.pending_mode == "sparse_scan"
    assert estimate.pending_query is query
    assert query.acquisition_index == 16
    assert query.identity_scan_index == 0
    assert query.offset_multiplier == 0.5
    assert query.frozen_fast_center_hz == identity.fast_center_hz
    assert query.frozen_fast_center_source_kind == identity.fast_center_source_kind
    assert query.frozen_fast_center_source_pair_index == (
        identity.fast_center_source_pair_index
    )
    assert query.frozen_prior_fwhm_hz == identity.active_fwhm_hz
    assert query.frozen_fwhm_source_kind == identity.fwhm_source_kind
    assert query.integration_time_s == estimate.configuration.integration_time_s
    assert query.expected_nominal_exposure_photons == (
        tracker._state.metadata.nominal_photon_rate_hz * query.integration_time_s
    )
    assert tracker.choose_next_query() is query


def test_due_sparse_block_uses_nondefault_policy_and_exact_recurrences() -> None:
    configuration = SparseLinewidthConfiguration(integration_time_s=0.007)
    calibration = _calibration()
    metadata = _metadata(calibration, frequency_overhead_s=0.003)
    tracker = _reset_tracker(
        configuration=configuration,
        calibration=calibration,
        metadata=metadata,
    )
    _accept_fast_pairs(tracker, calibration, count=8)
    before = tracker.estimate()

    first = tracker.choose_next_query()

    assert type(first) is SparseLinewidthQuery
    state = tracker._state
    assert state is not None
    queries = state.reserved_sparse_queries
    assert queries is not None
    assert len(queries) == 5
    expected_endpoint = before.current_timestamp_s
    for point_index, query in enumerate(queries):
        expected_endpoint = (
            expected_endpoint + metadata.frequency_overhead_s
        ) + configuration.integration_time_s
        assert query.point_index == point_index
        assert query.acquisition_index == before.accepted_observations + point_index
        assert query.expected_sequence_index == (
            before.current_sequence_index + point_index + 1
        )
        assert query.expected_end_timestamp_s == expected_endpoint
        assert query.integration_time_s == 0.007
        assert query.expected_nominal_exposure_photons == 2.5e6 * 0.007
    assert tuple(query.offset_multiplier for query in queries) == (
        0.5,
        -1.0,
        0.0,
        1.0,
        -0.5,
    )
    assert all(
        query.frozen_fast_center_release_timestamp_s
        == first.frozen_fast_center_release_timestamp_s
        and query.frozen_fwhm_release_timestamp_s
        == first.frozen_fwhm_release_timestamp_s
        for query in queries
    )


def test_due_sparse_block_does_not_fall_back_to_affordable_fast_work() -> None:
    calibration = _calibration()
    tracker = _reset_tracker(
        calibration=calibration,
        ceiling=TwoPointBudgetCeiling(20, None, None, None),
    )
    _accept_fast_pairs(tracker, calibration, count=8)
    before = tracker.estimate()

    assert tracker.choose_next_query() is None

    stopped = tracker.estimate()
    assert stopped == replace(before, stopped_reason="budget_exhausted")
    assert stopped.pending_query is None
    assert stopped.incomplete_fast_pair is None
    assert stopped.incomplete_sparse_scan is None
    assert stopped.fast_pairs_since_scan == 8


@pytest.mark.parametrize(
    ("failure_code", "proposed"),
    (
        ("nonrepresentable_frequency_lower", (None, None)),
        ("nonrepresentable_frequency_upper", (None, None)),
        ("empty_fit_bounds", (1.0, 2.0)),
        ("calibration_cell_violation", (1.0, 2.0)),
        ("source_domain_violation", (1.0, 2.0)),
    ),
)
def test_due_sparse_geometry_stops_before_affordability_with_full_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str,
    proposed: tuple[float | None, float | None],
) -> None:
    calibration = _calibration()
    tracker = _reset_tracker(
        calibration=calibration,
        ceiling=TwoPointBudgetCeiling(16, None, None, None),
    )
    _accept_fast_pairs(tracker, calibration, count=8)
    before = tracker.estimate()

    def fail_geometry(*args, **kwargs):
        del args, kwargs
        raise _SparseGeometryConstructionError(
            failure_code,
            "injected due geometry",
            proposed_frequency_min_hz=proposed[0],
            proposed_frequency_max_hz=proposed[1],
        )

    def reject_affordability(*args, **kwargs):
        del args, kwargs
        raise AssertionError("affordability must follow valid geometry")

    monkeypatch.setattr(tracker_module, "_construct_sparse_fit_geometry", fail_geometry)
    monkeypatch.setattr(
        tracker_module, "_reserve_five_atom_block", reject_affordability
    )

    assert tracker.choose_next_query() is None

    stopped = tracker.estimate()
    diagnostic = stopped.sparse_geometry_diagnostic
    identity = before.identities[0]
    cell = calibration.identities[0]
    assert stopped.stopped_reason == "sparse_geometry_unavailable"
    assert diagnostic is not None
    assert diagnostic.failure_code == failure_code
    assert diagnostic.scan_index == 0
    assert diagnostic.identity_scan_index == 0
    assert diagnostic.resonance_id == "r0"
    assert diagnostic.fast_center_hz == identity.fast_center_hz
    assert diagnostic.prior_fwhm_hz == identity.active_fwhm_hz
    assert (
        diagnostic.proposed_frequency_min_hz,
        diagnostic.proposed_frequency_max_hz,
    ) == proposed
    assert diagnostic.calibration_cell_lower_hz == cell.calibration_cell_lower_hz
    assert diagnostic.calibration_cell_upper_hz == cell.calibration_cell_upper_hz
    assert diagnostic.source_frequency_min_hz == (
        calibration.source.source_frequency_min_hz
    )
    assert diagnostic.source_frequency_max_hz == (
        calibration.source.source_frequency_max_hz
    )
