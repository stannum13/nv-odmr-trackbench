"""Reset and initial reservation tests for the sparse composite tracker."""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from copy import copy
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
    SparseLinewidthObservationValidationError,
    SparseLinewidthQuery,
    SparseLinewidthResetError,
    SparseLinewidthScanResult,
    SparseLinewidthUpdateConstructionError,
    SparsePartialScan,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointEstimate,
    TwoPointIdentityEstimate,
    TwoPointQuery,
    TwoPointRunMetadata,
    calibrate_two_point,
)
from odmr_bench.estimators import sparse_linewidth_tracker as tracker_module
from odmr_bench.estimators import two_point_calibration as calibration_module
from odmr_bench.estimators.sparse_linewidth_fit import (
    _construct_sparse_fit_geometry,
    _SparseGeometryConstructionError,
)
from odmr_bench.estimators.sparse_linewidth_fit import (
    fit_sparse_linewidth as _real_fit_sparse_linewidth,
)
from odmr_bench.estimators.two_point_calibration import (
    _evaluate_bound_source_model,
    _evaluate_target_only_model,
    _target_center_derivative,
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


def _sparse_observation(
    query: SparseLinewidthQuery,
    *,
    fluorescence: float = 0.9,
    realized_photons: int | None = None,
) -> EstimatorObservation:
    return EstimatorObservation(
        sequence_index=query.expected_sequence_index,
        timestamp_s=query.expected_end_timestamp_s,
        frequency_hz=query.frequency_hz,
        fluorescence=fluorescence,
        integration_time_s=query.integration_time_s,
        nominal_exposure_photons=query.expected_nominal_exposure_photons,
        realized_photons=realized_photons,
    )


def _model_sparse_observation(
    calibration: TwoPointCalibration,
    query: SparseLinewidthQuery,
    *,
    center_correction_fwhm_fraction: float = 0.1,
    fwhm_prior_ratio: float = 1.2,
    amplitude_source_ratio: float = 1.1,
    baseline_offset_source_ratio: float = 0.02,
    realized_photons: int | None = None,
) -> EstimatorObservation:
    target = next(
        resonance
        for resonance in calibration.source.source_fit.resonance_estimates
        if resonance.resonance_id == query.resonance_id
    )
    fluorescence = float(
        _evaluate_bound_source_model(
            query.frequency_hz,
            calibration.source,
            query.resonance_id,
            center_hz=(
                query.frozen_fast_center_hz
                + center_correction_fwhm_fraction * query.frozen_prior_fwhm_hz
            ),
            fwhm_hz=fwhm_prior_ratio * query.frozen_prior_fwhm_hz,
            amplitude=amplitude_source_ratio * target.amplitude,
            baseline_offset=baseline_offset_source_ratio * target.amplitude,
        )
    )
    return _sparse_observation(
        query,
        fluorescence=fluorescence,
        realized_photons=realized_photons,
    )


def _scan_public_reference(
    observations: tuple[EstimatorObservation, ...],
) -> float:
    reference_s = (
        observations[0].timestamp_s - observations[0].integration_time_s / 2.0
    )
    for count, observation in enumerate(observations[1:], start=2):
        midpoint_s = observation.timestamp_s - observation.integration_time_s / 2.0
        reference_s = reference_s + (midpoint_s - reference_s) / count
    return reference_s


def _synthetic_fit_result(
    queries: tuple[SparseLinewidthQuery, ...],
    observations: tuple[EstimatorObservation, ...],
    *,
    success: bool,
    fitted_fwhm_hz: float | None = None,
    fitted_local_center_hz: float | None = None,
) -> SparseLinewidthScanResult:
    first = queries[0]
    common = {
        "scan_index": first.scan_index,
        "identity_scan_index": first.identity_scan_index,
        "resonance_id": first.resonance_id,
        "frozen_fast_center_hz": first.frozen_fast_center_hz,
        "frozen_fast_center_source_kind": first.frozen_fast_center_source_kind,
        "frozen_fast_center_source_pair_index": (
            first.frozen_fast_center_source_pair_index
        ),
        "frozen_fast_center_reference_timestamp_s": (
            first.frozen_fast_center_reference_timestamp_s
        ),
        "frozen_fast_center_release_sequence_index": (
            first.frozen_fast_center_release_sequence_index
        ),
        "frozen_fast_center_release_timestamp_s": (
            first.frozen_fast_center_release_timestamp_s
        ),
        "frozen_prior_fwhm_hz": first.frozen_prior_fwhm_hz,
        "frozen_fwhm_source_kind": first.frozen_fwhm_source_kind,
        "frozen_fwhm_source_scan_index": first.frozen_fwhm_source_scan_index,
        "frozen_fwhm_reference_timestamp_s": (
            first.frozen_fwhm_reference_timestamp_s
        ),
        "frozen_fwhm_release_sequence_index": (
            first.frozen_fwhm_release_sequence_index
        ),
        "frozen_fwhm_release_timestamp_s": (
            first.frozen_fwhm_release_timestamp_s
        ),
        "queries": queries,
        "observations": observations,
        "public_reference_timestamp_s": _scan_public_reference(observations),
        "release_sequence_index": observations[-1].sequence_index,
        "release_timestamp_s": observations[-1].timestamp_s,
        "fit_cpu_time_s": 0.002,
    }
    if not success:
        return SparseLinewidthScanResult(
            **common,
            status="failure",
            failure_code="model_evaluation_failed",
            fitted_center_correction_hz=None,
            fitted_local_center_hz=None,
            fitted_fwhm_hz=None,
            fitted_amplitude=None,
            fitted_baseline_offset=None,
            fitted_q=None,
            rmse=None,
            amplitude_normalized_rmse=None,
            scaled_jacobian_rank=None,
            scaled_jacobian_condition=None,
            scipy_status=None,
            scipy_message=None,
            nfev=None,
        )
    fitted_fwhm_hz = (
        first.frozen_prior_fwhm_hz * 1.25
        if fitted_fwhm_hz is None
        else fitted_fwhm_hz
    )
    fitted_local_center_hz = (
        first.frozen_fast_center_hz
        if fitted_local_center_hz is None
        else fitted_local_center_hz
    )
    fitted_center_correction_hz = (
        fitted_local_center_hz - first.frozen_fast_center_hz
    )
    return SparseLinewidthScanResult(
        **common,
        status="success",
        failure_code=None,
        fitted_center_correction_hz=fitted_center_correction_hz,
        fitted_local_center_hz=fitted_local_center_hz,
        fitted_fwhm_hz=fitted_fwhm_hz,
        fitted_amplitude=1.0,
        fitted_baseline_offset=0.0,
        fitted_q=fitted_local_center_hz / fitted_fwhm_hz,
        rmse=0.0,
        amplitude_normalized_rmse=0.0,
        scaled_jacobian_rank=4,
        scaled_jacobian_condition=1.0,
        scipy_status=1,
        scipy_message="synthetic success",
        nfev=1,
    )


def _synthetic_failure_result(
    queries: tuple[SparseLinewidthQuery, ...],
    observations: tuple[EstimatorObservation, ...],
    failure_code: str,
) -> SparseLinewidthScanResult:
    successful = _synthetic_fit_result(queries, observations, success=True)
    replacements: dict[str, object] = {
        "status": "failure",
        "failure_code": failure_code,
        "fitted_q": None,
    }
    if failure_code == "model_evaluation_failed":
        replacements.update(
            fitted_center_correction_hz=None,
            fitted_local_center_hz=None,
            fitted_fwhm_hz=None,
            fitted_amplitude=None,
            fitted_baseline_offset=None,
            rmse=None,
            amplitude_normalized_rmse=None,
            scaled_jacobian_rank=None,
            scaled_jacobian_condition=None,
            scipy_status=None,
            scipy_message=None,
            nfev=None,
        )
    elif failure_code in {"optimizer_failed", "nonfinite_solution"}:
        replacements.update(
            fitted_center_correction_hz=None,
            fitted_local_center_hz=None,
            fitted_fwhm_hz=None,
            fitted_amplitude=None,
            fitted_baseline_offset=None,
            rmse=None,
            amplitude_normalized_rmse=None,
            scaled_jacobian_rank=None,
            scaled_jacobian_condition=None,
            scipy_status=0 if failure_code == "optimizer_failed" else 1,
        )
    elif failure_code == "bounds_active":
        replacements.update(
            scaled_jacobian_rank=None,
            scaled_jacobian_condition=None,
        )
    elif failure_code == "rank_deficient":
        replacements.update(
            scaled_jacobian_rank=3,
            scaled_jacobian_condition=None,
        )
    return replace(successful, **replacements)


def _seed_signed_target_at_fifth_point(
    tracker: SparseLinewidthCompositeTracker,
    calibration: TwoPointCalibration,
    fast_center_hz: float,
) -> tuple[tuple[SparseLinewidthQuery, ...], object]:
    for _ in range(2):
        query = tracker.choose_next_query()
        assert type(query) is TwoPointQuery
        tracker.update(_fast_observation(calibration, query, fluorescence=-1.0))
    _accept_fast_pairs(tracker, calibration, count=7)
    reserved, _ = _accept_sparse_prefix(tracker)
    fifth = tracker.choose_next_query()
    assert fifth is reserved[4]
    state = tracker._state
    assert state is not None
    target = state.estimate.identities[0]
    assert target.fast_center_source_kind == "calibration"
    assert target.latest_fast_pair is not None
    assert target.latest_fast_pair.lock_state == "lost"
    seeded_target = replace(
        target,
        fast_center_hz=fast_center_hz,
        live_q=fast_center_hz / target.active_fwhm_hz,
    )
    seeded_estimate = copy(state.estimate)
    object.__setattr__(
        seeded_estimate,
        "identities",
        (seeded_target, *state.estimate.identities[1:]),
    )
    object.__setattr__(tracker, "_state", replace(state, estimate=seeded_estimate))
    return reserved, seeded_target


def _accept_sparse_prefix(
    tracker: SparseLinewidthCompositeTracker,
    *,
    length: int = 4,
    observation_factory=_sparse_observation,
) -> tuple[
    tuple[SparseLinewidthQuery, ...],
    tuple[EstimatorObservation, ...],
]:
    first = tracker.choose_next_query()
    assert type(first) is SparseLinewidthQuery
    state = tracker._state
    assert state is not None
    reserved = state.reserved_sparse_queries
    assert reserved is not None
    observations: list[EstimatorObservation] = []
    for point_index in range(length):
        query = tracker.choose_next_query()
        assert query is reserved[point_index]
        observation = observation_factory(query)
        tracker.update(observation)
        observations.append(observation)
    return reserved, tuple(observations)


def _identity_source_snapshot(identity) -> tuple[object, ...]:
    return (
        identity.fast_center_hz,
        identity.fast_center_source_kind,
        identity.fast_center_source_pair_index,
        identity.fast_center_reference_timestamp_s,
        identity.fast_center_release_sequence_index,
        identity.fast_center_release_timestamp_s,
        identity.active_fwhm_hz,
        identity.fwhm_source_kind,
        identity.fwhm_source_scan_index,
        identity.fwhm_reference_timestamp_s,
        identity.fwhm_release_sequence_index,
        identity.fwhm_release_timestamp_s,
        identity.live_q,
        identity.completed_fast_pairs,
        identity.completed_sparse_scans,
        identity.latest_fast_pair,
        identity.latest_sparse_scan,
    )


def _frozen_source_snapshot(value) -> tuple[object, ...]:
    return (
        value.frozen_fast_center_hz,
        value.frozen_fast_center_source_kind,
        value.frozen_fast_center_source_pair_index,
        value.frozen_fast_center_reference_timestamp_s,
        value.frozen_fast_center_release_sequence_index,
        value.frozen_fast_center_release_timestamp_s,
        value.frozen_prior_fwhm_hz,
        value.frozen_fwhm_source_kind,
        value.frozen_fwhm_source_scan_index,
        value.frozen_fwhm_reference_timestamp_s,
        value.frozen_fwhm_release_sequence_index,
        value.frozen_fwhm_release_timestamp_s,
    )


def _independent_resource_transition(
    resources: PublicAcquisitionResources,
    observation: EstimatorObservation,
    metadata: TwoPointRunMetadata,
) -> PublicAcquisitionResources:
    realized_increment = (
        0 if observation.realized_photons is None else observation.realized_photons
    )
    missing_increment = 1 if observation.realized_photons is None else 0
    return PublicAcquisitionResources(
        observations=resources.observations + 1,
        integration_time_s=(
            resources.integration_time_s + observation.integration_time_s
        ),
        nominal_exposure_photons=(
            resources.nominal_exposure_photons
            + observation.nominal_exposure_photons
        ),
        realized_photons=resources.realized_photons + realized_increment,
        observations_without_realized_counts=(
            resources.observations_without_realized_counts + missing_increment
        ),
        virtual_elapsed_time_s=(
            resources.virtual_elapsed_time_s
            + (metadata.frequency_overhead_s + observation.integration_time_s)
        ),
    )


def _assert_resource_fields(
    actual: PublicAcquisitionResources,
    expected: PublicAcquisitionResources,
) -> None:
    assert actual.observations == expected.observations
    assert actual.integration_time_s == expected.integration_time_s
    assert actual.nominal_exposure_photons == expected.nominal_exposure_photons
    assert actual.realized_photons == expected.realized_photons
    assert (
        actual.observations_without_realized_counts
        == expected.observations_without_realized_counts
    )
    assert actual.virtual_elapsed_time_s == expected.virtual_elapsed_time_s


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


def _install_completed_sparse_scan_fixture(
    tracker: SparseLinewidthCompositeTracker,
) -> None:
    """Install one legal failed scan without exercising Task 9 update behavior."""
    first = tracker.choose_next_query()
    assert type(first) is SparseLinewidthQuery
    state = tracker._state
    assert state is not None
    queries = state.reserved_sparse_queries
    assert queries is not None
    observations = tuple(
        EstimatorObservation(
            sequence_index=query.expected_sequence_index,
            timestamp_s=query.expected_end_timestamp_s,
            frequency_hz=query.frequency_hz,
            fluorescence=1.0,
            integration_time_s=query.integration_time_s,
            nominal_exposure_photons=query.expected_nominal_exposure_photons,
            realized_photons=None,
        )
        for query in queries
    )
    reference_timestamp_s = (
        observations[0].timestamp_s - observations[0].integration_time_s / 2.0
    )
    for count, observation in enumerate(observations[1:], start=2):
        midpoint_s = observation.timestamp_s - observation.integration_time_s / 2.0
        reference_timestamp_s = reference_timestamp_s + (
            midpoint_s - reference_timestamp_s
        ) / count
    scan = SparseLinewidthScanResult(
        scan_index=first.scan_index,
        identity_scan_index=first.identity_scan_index,
        resonance_id=first.resonance_id,
        frozen_fast_center_hz=first.frozen_fast_center_hz,
        frozen_fast_center_source_kind=first.frozen_fast_center_source_kind,
        frozen_fast_center_source_pair_index=(
            first.frozen_fast_center_source_pair_index
        ),
        frozen_fast_center_reference_timestamp_s=(
            first.frozen_fast_center_reference_timestamp_s
        ),
        frozen_fast_center_release_sequence_index=(
            first.frozen_fast_center_release_sequence_index
        ),
        frozen_fast_center_release_timestamp_s=(
            first.frozen_fast_center_release_timestamp_s
        ),
        frozen_prior_fwhm_hz=first.frozen_prior_fwhm_hz,
        frozen_fwhm_source_kind=first.frozen_fwhm_source_kind,
        frozen_fwhm_source_scan_index=first.frozen_fwhm_source_scan_index,
        frozen_fwhm_reference_timestamp_s=(
            first.frozen_fwhm_reference_timestamp_s
        ),
        frozen_fwhm_release_sequence_index=(
            first.frozen_fwhm_release_sequence_index
        ),
        frozen_fwhm_release_timestamp_s=first.frozen_fwhm_release_timestamp_s,
        queries=queries,
        observations=observations,
        public_reference_timestamp_s=reference_timestamp_s,
        release_sequence_index=observations[-1].sequence_index,
        release_timestamp_s=observations[-1].timestamp_s,
        status="failure",
        failure_code="model_evaluation_failed",
        fitted_center_correction_hz=None,
        fitted_local_center_hz=None,
        fitted_fwhm_hz=None,
        fitted_amplitude=None,
        fitted_baseline_offset=None,
        fitted_q=None,
        rmse=None,
        amplitude_normalized_rmse=None,
        scaled_jacobian_rank=None,
        scaled_jacobian_condition=None,
        scipy_status=None,
        scipy_message=None,
        nfev=None,
        fit_cpu_time_s=0.0,
    )
    endpoint_s = observations[-1].timestamp_s
    identities = tuple(
        replace(
            identity,
            center_age_s=endpoint_s - identity.fast_center_reference_timestamp_s,
            fwhm_age_s=endpoint_s - identity.fwhm_reference_timestamp_s,
            center_release_age_s=(
                endpoint_s - identity.fast_center_release_timestamp_s
            ),
            fwhm_release_age_s=endpoint_s - identity.fwhm_release_timestamp_s,
            completed_sparse_scans=(
                identity.completed_sparse_scans
                + int(identity.resonance_id == scan.resonance_id)
            ),
            latest_sparse_scan=(
                scan
                if identity.resonance_id == scan.resonance_id
                else identity.latest_sparse_scan
            ),
        )
        for identity in state.estimate.identities
    )

    def advance_resources(
        resources: PublicAcquisitionResources,
    ) -> PublicAcquisitionResources:
        for observation in observations:
            resources = tracker_module._advance_observation_resources(
                resources, observation, state.metadata
            )
        return resources

    sparse_resources = advance_resources(state.estimate.sparse_tracking_resources)
    tracking_resources = advance_resources(state.estimate.tracking_resources)
    estimate = tracker_module._replace_estimate(
        state.estimate,
        identities=identities,
        pending_mode=None,
        pending_query=None,
        sparse_scan_history=(*state.estimate.sparse_scan_history, scan),
        accepted_observations=state.estimate.accepted_observations + 5,
        completed_sparse_scans=state.estimate.completed_sparse_scans + 1,
        fast_pairs_since_scan=0,
        current_sequence_index=observations[-1].sequence_index,
        current_timestamp_s=endpoint_s,
        sparse_tracking_resources=sparse_resources,
        tracking_resources=tracking_resources,
        charged_resources=tracking_resources,
    )
    object.__setattr__(
        tracker,
        "_state",
        replace(state, reserved_sparse_queries=None, estimate=estimate),
    )


def _advance_to_due_sparse_scan(
    tracker: SparseLinewidthCompositeTracker,
    calibration: TwoPointCalibration,
    *,
    scan_index: int,
) -> None:
    _accept_fast_pairs(tracker, calibration, count=8)
    for completed_scan_index in range(scan_index):
        assert tracker.estimate().completed_sparse_scans == completed_scan_index
        _install_completed_sparse_scan_fixture(tracker)
        _accept_fast_pairs(tracker, calibration, count=8)


def _project_composite_fast_view(
    composite: SparseLinewidthCompositeTracker,
    calibration: TwoPointCalibration,
) -> TwoPointEstimate:
    estimate = composite.estimate()
    identities = tuple(
        TwoPointIdentityEstimate(
            resonance_id=identity.resonance_id,
            center_hz=identity.fast_center_hz,
            calibration_fwhm_hz=cell.calibration_fwhm_hz,
            calibration_cell_lower_hz=cell.calibration_cell_lower_hz,
            calibration_cell_upper_hz=cell.calibration_cell_upper_hz,
            allowed_center_min_hz=cell.allowed_center_min_hz,
            allowed_center_max_hz=cell.allowed_center_max_hz,
            active_source_kind=identity.fast_center_source_kind,
            active_source_pair_index=identity.fast_center_source_pair_index,
            active_reference_timestamp_s=(
                identity.fast_center_reference_timestamp_s
            ),
            active_release_sequence_index=(
                identity.fast_center_release_sequence_index
            ),
            active_release_timestamp_s=(
                identity.fast_center_release_timestamp_s
            ),
            estimate_age_sequence_indices=(
                None
                if identity.fast_center_release_sequence_index is None
                else estimate.current_sequence_index
                - identity.fast_center_release_sequence_index
            ),
            estimate_age_s=identity.center_age_s,
            release_age_s=identity.center_release_age_s,
            completed_pairs=identity.completed_fast_pairs,
            lock_state=(
                "calibrated"
                if identity.latest_fast_pair is None
                else identity.latest_fast_pair.lock_state
            ),
            failure_code=(
                None
                if identity.latest_fast_pair is None
                else identity.latest_fast_pair.failure_code
            ),
            latest_pair=identity.latest_fast_pair,
        )
        for identity, cell in zip(
            estimate.identities, calibration.identities, strict=True
        )
    )
    return TwoPointEstimate(
        identities=identities,
        calibration_source_id=estimate.calibration_source_id,
        calibration_source_provenance=estimate.calibration_source_provenance,
        calibration_budget_treatment=estimate.calibration_budget_treatment,
        current_sequence_index=estimate.current_sequence_index,
        current_timestamp_s=estimate.current_timestamp_s,
        accepted_observations=estimate.accepted_observations,
        completed_pairs=estimate.completed_fast_pairs,
        incomplete_pair=estimate.incomplete_fast_pair,
        pending_query=(
            estimate.pending_query if estimate.pending_mode == "fast_pair" else None
        ),
        pair_history=estimate.fast_pair_history,
        tracking_resources=estimate.fast_tracking_resources,
        calibration_resources=estimate.calibration_resources,
        charged_resources=estimate.charged_resources,
        budget_ceiling=estimate.budget_ceiling,
        stopped_reason=estimate.stopped_reason,
        seed=estimate.seed,
    )


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
            fluorescence = -1.0 if pair_index == 14 else None
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
            assert _project_composite_fast_view(composite, calibration) == (
                legacy.estimate()
            )

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
    assert composite_estimate.fast_pair_history[14].lock_state == "lost"
    r6 = composite_estimate.identities[6]
    assert r6.fast_center_source_kind == "pair"
    assert r6.fast_center_source_pair_index == 6
    assert tuple(
        pair.first_side for pair in composite_estimate.fast_pair_history[8:12]
    ) == ("plus", "plus", "plus", "plus")


def _differential_gate_case(
    case: str,
) -> tuple[TwoPointCalibration, tuple[float, float]]:
    base_calibration = _calibration()
    base_configuration = base_calibration.configuration
    base_cell = base_calibration.identities[0]
    source_fit = base_calibration.source.source_fit
    center_hz = base_cell.calibration_center_hz
    minus_frequency_hz = center_hz - base_cell.offset_hz
    plus_frequency_hz = center_hz + base_cell.offset_hz
    mu_minus = _evaluate_target_only_model(
        source_fit,
        base_cell.source_fit_index,
        minus_frequency_hz,
        center_hz,
    )
    mu_plus = _evaluate_target_only_model(
        source_fit,
        base_cell.source_fit_index,
        plus_frequency_hz,
        center_hz,
    )
    g_minus = _target_center_derivative(
        source_fit,
        base_cell.source_fit_index,
        minus_frequency_hz,
        center_hz,
    )
    g_plus = _target_center_derivative(
        source_fit,
        base_cell.source_fit_index,
        plus_frequency_hz,
        center_hz,
    )
    model_sum = mu_minus + mu_plus
    zero_discriminator = (mu_minus - mu_plus) / model_sum
    slope_per_hz = 2.0 * (
        mu_plus * g_minus - mu_minus * g_plus
    ) / model_sum**2
    desired_raw_hz = {
        "tracking": 0.05 * base_cell.calibration_fwhm_hz,
        "common": 0.0,
        "capture": 0.05 * base_cell.calibration_fwhm_hz,
        "domain": 0.05 * base_cell.calibration_fwhm_hz,
        "step-limited": 0.15 * base_cell.calibration_fwhm_hz,
        "numerical": 2.0,
    }[case]
    desired_common = 0.25 if case == "common" else 0.0
    observed_sum = model_sum + desired_common * base_cell.target_pair_depth
    desired_discriminator = zero_discriminator + slope_per_hz * desired_raw_hz
    minus_value = observed_sum * (1.0 + desired_discriminator) / 2.0
    plus_value = observed_sum - minus_value
    actual_discriminator = (minus_value - plus_value) / observed_sum
    actual_common = (observed_sum - model_sum) / base_cell.target_pair_depth
    actual_raw_hz = (
        actual_discriminator - zero_discriminator
    ) / slope_per_hz
    configuration = replace(
        base_configuration,
        proportional_gain=(1.0e308 if case == "numerical" else 1.0),
        common_mode_limit_target_depths=(
            math.nextafter(abs(actual_common), 0.0)
            if case == "common"
            else None
        ),
    )
    calibration = calibrate_two_point(
        base_calibration.source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    cell = calibration.identities[0]
    if case == "capture":
        calibration = replace(
            calibration,
            identities=(
                replace(
                    cell,
                    capture_radius_hz=math.nextafter(abs(actual_raw_hz), 0.0),
                ),
                *calibration.identities[1:],
            ),
        )
    if case == "domain":
        candidate_hz = center_hz + min(actual_raw_hz, cell.max_step_hz)
        calibration = replace(
            calibration,
            identities=(
                replace(
                    cell,
                    allowed_center_max_hz=math.nextafter(candidate_hz, -math.inf),
                ),
                *calibration.identities[1:],
            ),
        )
    return calibration, (minus_value, plus_value)


@pytest.mark.parametrize(
    ("case", "expected_lock", "expected_failure"),
    (
        ("tracking", "tracking", None),
        ("common", "lost", "common_mode_limit_exceeded"),
        ("capture", "lost", "capture_exceeded"),
        ("domain", "lost", "calibration_domain_exceeded"),
        ("step-limited", "step_limited", None),
        ("numerical", "lost", "numerical_failure"),
    ),
)
def test_fast_gate_views_are_completely_stage_63_differential(
    case: str,
    expected_lock: str,
    expected_failure: str | None,
) -> None:
    calibration, fluorescence_values = _differential_gate_case(case)
    metadata = _metadata(calibration)
    ceiling = TwoPointBudgetCeiling(20, None, None, None)
    legacy = CalibratedTwoPointTracker(calibration.configuration)
    composite = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    legacy.reset(metadata, calibration, ceiling, seed=31)
    composite.reset(metadata, calibration, ceiling, seed=31)

    for fluorescence in fluorescence_values:
        legacy_query = legacy.choose_next_query()
        composite_query = composite.choose_next_query()
        assert type(legacy_query) is TwoPointQuery
        assert composite_query == legacy_query
        legacy.update(
            _fast_observation(
                calibration,
                legacy_query,
                fluorescence=fluorescence,
            )
        )
        composite.update(
            _fast_observation(
                calibration,
                composite_query,
                fluorescence=fluorescence,
            )
        )
        assert _project_composite_fast_view(composite, calibration) == (
            legacy.estimate()
        )

    pair = composite.estimate().fast_pair_history[-1]
    assert pair.lock_state == expected_lock
    assert pair.failure_code == expected_failure


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


@pytest.mark.parametrize(
    ("scan_index", "identity_index", "identity_scan_index", "expected_order"),
    (
        (0, 0, 0, (0.5, -1.0, 0.0, 1.0, -0.5)),
        (1, 1, 0, (0.5, -1.0, 0.0, 1.0, -0.5)),
        (8, 0, 1, (-0.5, 1.0, 0.0, -1.0, 0.5)),
    ),
)
def test_sparse_query_constructor_rotates_targets_orders_and_all_echoes(
    scan_index: int,
    identity_index: int,
    identity_scan_index: int,
    expected_order: tuple[float, ...],
) -> None:
    configuration = SparseLinewidthConfiguration(integration_time_s=0.007)
    calibration = _calibration()
    metadata = _metadata(calibration, frequency_overhead_s=0.003)
    tracker = _reset_tracker(
        configuration=configuration,
        calibration=calibration,
        metadata=metadata,
    )
    _accept_fast_pairs(tracker, calibration, count=8)
    estimate = tracker.estimate()
    identity = estimate.identities[identity_index]
    geometry = _construct_sparse_fit_geometry(
        calibration,
        configuration,
        identity,
        scan_index=scan_index,
        identity_scan_index=identity_scan_index,
    )

    queries = tracker_module._construct_sparse_queries(
        geometry,
        identity,
        metadata,
        integration_time_s=configuration.integration_time_s,
        first_acquisition_index=40,
        first_sequence_index=73,
        start_timestamp_s=0.5,
        scan_index=scan_index,
        identity_scan_index=identity_scan_index,
    )

    assert tuple(query.offset_multiplier for query in queries) == expected_order
    assert {query.resonance_id for query in queries} == {f"r{identity_index}"}
    expected_frozen = (
        identity.fast_center_hz,
        identity.fast_center_source_kind,
        identity.fast_center_source_pair_index,
        identity.fast_center_reference_timestamp_s,
        identity.fast_center_release_sequence_index,
        identity.fast_center_release_timestamp_s,
        identity.active_fwhm_hz,
        identity.fwhm_source_kind,
        identity.fwhm_source_scan_index,
        identity.fwhm_reference_timestamp_s,
        identity.fwhm_release_sequence_index,
        identity.fwhm_release_timestamp_s,
    )
    for point_index, query in enumerate(queries):
        assert (
            query.frozen_fast_center_hz,
            query.frozen_fast_center_source_kind,
            query.frozen_fast_center_source_pair_index,
            query.frozen_fast_center_reference_timestamp_s,
            query.frozen_fast_center_release_sequence_index,
            query.frozen_fast_center_release_timestamp_s,
            query.frozen_prior_fwhm_hz,
            query.frozen_fwhm_source_kind,
            query.frozen_fwhm_source_scan_index,
            query.frozen_fwhm_reference_timestamp_s,
            query.frozen_fwhm_release_sequence_index,
            query.frozen_fwhm_release_timestamp_s,
        ) == expected_frozen
        assert query.acquisition_index == 40 + point_index
        assert query.expected_sequence_index == 73 + point_index
        assert query.integration_time_s == 0.007
        assert query.expected_nominal_exposure_photons == 2.5e6 * 0.007


@pytest.mark.parametrize(
    ("scan_index", "resonance_id", "identity_scan_index", "expected_order"),
    (
        (1, "r1", 0, (0.5, -1.0, 0.0, 1.0, -0.5)),
        (8, "r0", 1, (-0.5, 1.0, 0.0, -1.0, 0.5)),
    ),
)
def test_due_scheduler_rotates_target_and_uses_identity_scan_order(
    scan_index: int,
    resonance_id: str,
    identity_scan_index: int,
    expected_order: tuple[float, ...],
) -> None:
    calibration = _calibration()
    tracker = _reset_tracker(
        calibration=calibration,
        ceiling=TwoPointBudgetCeiling(1_000, None, None, None),
    )
    _advance_to_due_sparse_scan(
        tracker,
        calibration,
        scan_index=scan_index,
    )
    before = tracker.estimate()
    target = before.identities[scan_index % 8]
    assert before.completed_sparse_scans == scan_index
    assert before.fast_pairs_since_scan == 8
    assert target.resonance_id == resonance_id
    assert target.completed_sparse_scans == identity_scan_index

    first = tracker.choose_next_query()

    assert type(first) is SparseLinewidthQuery
    state = tracker._state
    assert state is not None
    queries = state.reserved_sparse_queries
    assert queries is not None
    assert first is queries[0]
    assert {
        (query.scan_index, query.resonance_id, query.identity_scan_index)
        for query in queries
    } == {(scan_index, resonance_id, identity_scan_index)}
    assert tuple(query.offset_multiplier for query in queries) == expected_order


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


def test_sparse_pending_query_requires_exact_reserved_object_before_sequence() -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    _accept_fast_pairs(tracker, calibration, count=8)
    query = tracker.choose_next_query()
    assert type(query) is SparseLinewidthQuery
    state = tracker._state
    assert state is not None
    cloned_query = replace(query)
    corrupted_estimate = replace(state.estimate, pending_query=cloned_query)
    object.__setattr__(
        tracker,
        "_state",
        replace(state, estimate=corrupted_estimate),
    )
    observation = EstimatorObservation(
        sequence_index=query.expected_sequence_index + 1,
        timestamp_s=query.expected_end_timestamp_s,
        frequency_hz=query.frequency_hz,
        fluorescence=0.9,
        integration_time_s=query.integration_time_s,
        nominal_exposure_photons=query.expected_nominal_exposure_photons,
        realized_photons=None,
    )
    before = tracker._state

    with pytest.raises(SparseLinewidthObservationValidationError) as raised:
        tracker.update(observation)

    assert raised.value.code == "sparse_query_echo_mismatch"
    assert tracker._state == before


@pytest.mark.parametrize("stale_kind", ("reservation", "partial"))
def test_sparse_pending_query_rejects_stale_fast_state_before_sequence(
    stale_kind: str,
) -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    first_fast_query = tracker.choose_next_query()
    assert type(first_fast_query) is TwoPointQuery
    state = tracker._state
    assert state is not None
    stale_fast_reservation = state.reserved_fast_queries
    assert stale_fast_reservation is not None
    tracker.update(_fast_observation(calibration, first_fast_query))
    stale_fast_partial = tracker.estimate().incomplete_fast_pair
    assert stale_fast_partial is not None
    for _ in range(15):
        query = tracker.choose_next_query()
        assert type(query) is TwoPointQuery
        tracker.update(_fast_observation(calibration, query))
    sparse_query = tracker.choose_next_query()
    assert type(sparse_query) is SparseLinewidthQuery
    state = tracker._state
    assert state is not None
    if stale_kind == "reservation":
        corrupted_state = replace(
            state, reserved_fast_queries=stale_fast_reservation
        )
    else:
        corrupted_estimate = copy(state.estimate)
        object.__setattr__(
            corrupted_estimate, "incomplete_fast_pair", stale_fast_partial
        )
        corrupted_state = replace(state, estimate=corrupted_estimate)
    object.__setattr__(tracker, "_state", corrupted_state)
    observation = EstimatorObservation(
        sequence_index=sparse_query.expected_sequence_index + 1,
        timestamp_s=sparse_query.expected_end_timestamp_s,
        frequency_hz=sparse_query.frequency_hz,
        fluorescence=0.9,
        integration_time_s=sparse_query.integration_time_s,
        nominal_exposure_photons=sparse_query.expected_nominal_exposure_photons,
        realized_photons=None,
    )
    before = tracker._state

    with pytest.raises(SparseLinewidthObservationValidationError) as raised:
        tracker.update(observation)

    assert raised.value.code == "sparse_query_echo_mismatch"
    assert tracker._state == before


def test_first_four_sparse_points_are_frozen_partial_transitions() -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    _accept_fast_pairs(tracker, calibration, count=8)
    before = tracker.estimate()
    first = tracker.choose_next_query()
    assert type(first) is SparseLinewidthQuery
    state = tracker._state
    assert state is not None
    reserved = state.reserved_sparse_queries
    assert reserved is not None
    source_snapshots = tuple(
        _identity_source_snapshot(identity) for identity in before.identities
    )
    observations: list[EstimatorObservation] = []
    updates = []
    expected_sparse_cpu = before.sparse_update_cpu_time_s
    expected_total_cpu = before.total_update_cpu_time_s

    for point_index in range(4):
        query = tracker.choose_next_query()
        assert query is reserved[point_index]
        assert tracker.choose_next_query() is query
        observation = _sparse_observation(
            query,
            fluorescence=0.91 + 0.01 * point_index,
            realized_photons=None if point_index % 2 else point_index + 3,
        )
        observations.append(observation)

        update = tracker.update(observation)
        updates.append(update)
        partial = update.estimate.incomplete_sparse_scan
        assert type(partial) is SparsePartialScan
        assert len(partial.queries) == point_index + 1
        assert _frozen_source_snapshot(partial) == _frozen_source_snapshot(
            reserved[0]
        )
        assert all(
            actual is expected
            for actual, expected in zip(
                partial.queries, reserved[: point_index + 1], strict=True
            )
        )
        assert all(
            actual is expected
            for actual, expected in zip(
                partial.observations, observations, strict=True
            )
        )
        assert update.query is query
        assert update.observation is observation
        assert update.completed_fast_pair is None
        assert update.completed_sparse_scan is None
        assert update.estimate.pending_mode is None
        assert update.estimate.pending_query is None
        assert update.estimate.sparse_scan_history == before.sparse_scan_history
        assert update.estimate.completed_sparse_scans == 0
        assert update.estimate.completed_fast_pairs == before.completed_fast_pairs
        assert update.estimate.fast_pairs_since_scan == before.fast_pairs_since_scan
        assert update.estimate.accepted_observations == (
            before.accepted_observations + point_index + 1
        )
        assert update.estimate.current_sequence_index == observation.sequence_index
        assert update.estimate.current_timestamp_s == observation.timestamp_s
        assert tuple(
            _identity_source_snapshot(identity)
            for identity in update.estimate.identities
        ) == source_snapshots
        expected_sparse_cpu = expected_sparse_cpu + update.update_cpu_time_s
        expected_total_cpu = expected_total_cpu + update.update_cpu_time_s
        assert update.estimate.sparse_update_cpu_time_s == expected_sparse_cpu
        assert update.estimate.total_update_cpu_time_s == expected_total_cpu

    assert tuple(update.completed_sparse_scan for update in updates) == (None,) * 4
    fifth = tracker.choose_next_query()
    assert fifth is reserved[4]
    assert tracker.estimate().pending_query is fifth
    assert (
        tracker.estimate().incomplete_sparse_scan
        is updates[-1].estimate.incomplete_sparse_scan
    )
    assert all(
        _identity_source_snapshot(tracker.estimate().identities[index])
        == source_snapshots[index]
        for index in range(8)
    )
    frozen_query_facts = tuple(_frozen_source_snapshot(query) for query in reserved)
    assert len(set(frozen_query_facts)) == 1


def test_sparse_partial_resource_ledgers_follow_exact_arrival_order() -> None:
    configuration = SparseLinewidthConfiguration(integration_time_s=0.007)
    calibration = _calibration(included=True)
    metadata = _metadata(calibration, included=True)
    tracker = _reset_tracker(
        configuration=configuration,
        calibration=calibration,
        metadata=metadata,
    )
    _accept_fast_pairs(tracker, calibration, count=8)
    before = tracker.estimate()
    expected_sparse = before.sparse_tracking_resources
    expected_tracking = before.tracking_resources
    expected_charged = before.charged_resources
    assert expected_charged != expected_tracking

    for point_index in range(4):
        query = tracker.choose_next_query()
        assert type(query) is SparseLinewidthQuery
        observation = _sparse_observation(
            query,
            realized_photons=None if point_index == 1 else 10 + point_index,
        )
        expected_sparse = _independent_resource_transition(
            expected_sparse, observation, metadata
        )
        expected_tracking = _independent_resource_transition(
            expected_tracking, observation, metadata
        )
        expected_charged = _independent_resource_transition(
            expected_charged, observation, metadata
        )

        update = tracker.update(observation)

        assert update.estimate.fast_tracking_resources == before.fast_tracking_resources
        _assert_resource_fields(
            update.estimate.sparse_tracking_resources, expected_sparse
        )
        _assert_resource_fields(update.estimate.tracking_resources, expected_tracking)
        _assert_resource_fields(update.estimate.charged_resources, expected_charged)
        assert (
            update.estimate.sparse_tracking_resources.observations
            == point_index + 1
        )
        assert update.estimate.tracking_resources.observations == (
            before.tracking_resources.observations + point_index + 1
        )


def test_sparse_success_refreshes_width_but_never_fast_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    _accept_fast_pairs(tracker, calibration, count=8)
    reserved, observations = _accept_sparse_prefix(
        tracker,
        observation_factory=lambda query: _model_sparse_observation(
            calibration, query, realized_photons=31 + query.point_index
        ),
    )
    before = tracker.estimate()
    before_target = before.identities[0]
    fifth = tracker.choose_next_query()
    assert fifth is reserved[4]
    fifth_observation = _model_sparse_observation(
        calibration, fifth, realized_photons=35
    )
    calls: list[tuple[object, ...]] = []

    def counted_fit(source, configuration, queries, fit_observations):
        calls.append((source, configuration, queries, fit_observations))
        return _real_fit_sparse_linewidth(
            source, configuration, queries, fit_observations
        )

    monkeypatch.setattr(
        tracker_module, "fit_sparse_linewidth", counted_fit, raising=False
    )

    update = tracker.update(fifth_observation)

    assert len(calls) == 1
    source, configuration, queries, fit_observations = calls[0]
    assert source is calibration.source
    assert configuration is tracker._configuration
    assert all(
        actual is expected
        for actual, expected in zip(queries, reserved, strict=True)
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            fit_observations, (*observations, fifth_observation), strict=True
        )
    )
    scan = update.completed_sparse_scan
    assert type(scan) is SparseLinewidthScanResult
    assert scan.status == "success"
    assert scan.fitted_local_center_hz != before_target.fast_center_hz
    assert scan.public_reference_timestamp_s == _scan_public_reference(
        scan.observations
    )
    assert scan.release_sequence_index == fifth_observation.sequence_index
    assert scan.release_timestamp_s == fifth_observation.timestamp_s
    after = update.estimate
    after_target = after.identities[0]
    assert after_target.fast_center_hz == before_target.fast_center_hz
    assert after_target.fast_center_source_kind == before_target.fast_center_source_kind
    assert (
        after_target.fast_center_source_pair_index
        == before_target.fast_center_source_pair_index
    )
    assert (
        after_target.fast_center_reference_timestamp_s
        == before_target.fast_center_reference_timestamp_s
    )
    assert (
        after_target.fast_center_release_sequence_index
        == before_target.fast_center_release_sequence_index
    )
    assert (
        after_target.fast_center_release_timestamp_s
        == before_target.fast_center_release_timestamp_s
    )
    assert after_target.active_fwhm_hz == scan.fitted_fwhm_hz
    assert after_target.fwhm_source_kind == "scan"
    assert after_target.fwhm_source_scan_index == scan.scan_index
    assert (
        after_target.fwhm_reference_timestamp_s
        == scan.public_reference_timestamp_s
    )
    assert after_target.fwhm_release_sequence_index == scan.release_sequence_index
    assert after_target.fwhm_release_timestamp_s == scan.release_timestamp_s
    assert after_target.live_q == (
        after_target.fast_center_hz / after_target.active_fwhm_hz
    )
    assert after_target.center_age_s == (
        scan.release_timestamp_s - before_target.fast_center_reference_timestamp_s
    )
    assert after_target.center_release_age_s == (
        scan.release_timestamp_s - before_target.fast_center_release_timestamp_s
    )
    assert after_target.fwhm_age_s == (
        scan.release_timestamp_s - scan.public_reference_timestamp_s
    )
    assert after_target.fwhm_release_age_s == 0.0
    assert after.completed_sparse_scans == 1
    assert after.fast_pairs_since_scan == 0
    assert after.sparse_scan_history[-1] is scan
    assert after_target.latest_sparse_scan is scan
    assert after_target.completed_sparse_scans == 1
    assert after.incomplete_sparse_scan is None
    assert after.pending_mode is None
    assert after.pending_query is None
    assert update.query is fifth
    assert update.observation is fifth_observation
    assert tracker._state is not None
    assert tracker._state.reserved_sparse_queries is None
    next_query = tracker.choose_next_query()
    assert type(next_query) is TwoPointQuery
    assert next_query.interrogation_center_hz == before_target.fast_center_hz


def test_scientific_sparse_failure_is_completed_but_retains_both_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    _accept_fast_pairs(tracker, calibration, count=8)
    reserved, observations = _accept_sparse_prefix(tracker)
    before = tracker.estimate()
    before_target = before.identities[0]
    fifth = tracker.choose_next_query()
    assert fifth is reserved[4]
    fifth_observation = _sparse_observation(fifth)
    calls = 0

    def failed_fit(source, configuration, queries, fit_observations):
        nonlocal calls
        calls += 1
        assert source is calibration.source
        assert configuration is tracker._configuration
        return _synthetic_fit_result(
            queries, fit_observations, success=False
        )

    monkeypatch.setattr(
        tracker_module, "fit_sparse_linewidth", failed_fit, raising=False
    )

    update = tracker.update(fifth_observation)

    assert calls == 1
    scan = update.completed_sparse_scan
    assert type(scan) is SparseLinewidthScanResult
    assert scan.status == "failure"
    assert scan.failure_code == "model_evaluation_failed"
    assert all(
        actual is expected
        for actual, expected in zip(
            scan.observations, (*observations, fifth_observation), strict=True
        )
    )
    after = update.estimate
    after_target = after.identities[0]
    assert (
        after_target.fast_center_hz,
        after_target.fast_center_source_kind,
        after_target.fast_center_source_pair_index,
        after_target.fast_center_reference_timestamp_s,
        after_target.fast_center_release_sequence_index,
        after_target.fast_center_release_timestamp_s,
    ) == (
        before_target.fast_center_hz,
        before_target.fast_center_source_kind,
        before_target.fast_center_source_pair_index,
        before_target.fast_center_reference_timestamp_s,
        before_target.fast_center_release_sequence_index,
        before_target.fast_center_release_timestamp_s,
    )
    assert (
        after_target.active_fwhm_hz,
        after_target.fwhm_source_kind,
        after_target.fwhm_source_scan_index,
        after_target.fwhm_reference_timestamp_s,
        after_target.fwhm_release_sequence_index,
        after_target.fwhm_release_timestamp_s,
    ) == (
        before_target.active_fwhm_hz,
        before_target.fwhm_source_kind,
        before_target.fwhm_source_scan_index,
        before_target.fwhm_reference_timestamp_s,
        before_target.fwhm_release_sequence_index,
        before_target.fwhm_release_timestamp_s,
    )
    assert after_target.live_q == before_target.live_q
    assert after_target.center_age_s == (
        scan.release_timestamp_s - before_target.fast_center_reference_timestamp_s
    )
    assert after_target.fwhm_age_s == (
        scan.release_timestamp_s - before_target.fwhm_reference_timestamp_s
    )
    assert after_target.center_release_age_s == (
        scan.release_timestamp_s - before_target.fast_center_release_timestamp_s
    )
    assert after_target.fwhm_release_age_s == (
        scan.release_timestamp_s - before_target.fwhm_release_timestamp_s
    )
    assert after.completed_sparse_scans == before.completed_sparse_scans + 1
    assert after.fast_pairs_since_scan == 0
    assert after.sparse_scan_history[-1] is scan
    assert after_target.latest_sparse_scan is scan
    assert after_target.completed_sparse_scans == 1
    assert update.estimate.accepted_observations == before.accepted_observations + 1


@pytest.mark.parametrize(
    "fast_center_hz",
    (
        pytest.param(-2.0, id="negative"),
        pytest.param(-0.0, id="negative-zero"),
        pytest.param(0.0, id="positive-zero"),
        pytest.param(2.0, id="positive"),
    ),
)
@pytest.mark.parametrize("fit_success", (False, True), ids=("failure", "success"))
def test_fifth_sparse_transition_recomputes_signed_and_zero_live_q(
    monkeypatch: pytest.MonkeyPatch,
    fast_center_hz: float,
    fit_success: bool,
) -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    reserved, before_target = _seed_signed_target_at_fifth_point(
        tracker, calibration, fast_center_hz
    )
    fifth = reserved[4]
    observation = _sparse_observation(fifth)

    def fit_result(source, configuration, queries, observations):
        del source, configuration
        return _synthetic_fit_result(
            queries,
            observations,
            success=fit_success,
        )

    monkeypatch.setattr(tracker_module, "fit_sparse_linewidth", fit_result)

    update = tracker.update(observation)

    scan = update.completed_sparse_scan
    assert scan is not None
    after_target = update.estimate.identities[0]
    assert after_target.fast_center_hz == before_target.fast_center_hz
    assert math.copysign(1.0, after_target.fast_center_hz) == math.copysign(
        1.0, before_target.fast_center_hz
    )
    assert (
        after_target.fast_center_source_kind,
        after_target.fast_center_source_pair_index,
        after_target.fast_center_reference_timestamp_s,
        after_target.fast_center_release_sequence_index,
        after_target.fast_center_release_timestamp_s,
    ) == (
        before_target.fast_center_source_kind,
        before_target.fast_center_source_pair_index,
        before_target.fast_center_reference_timestamp_s,
        before_target.fast_center_release_sequence_index,
        before_target.fast_center_release_timestamp_s,
    )
    expected_fwhm_hz = (
        scan.fitted_fwhm_hz if fit_success else before_target.active_fwhm_hz
    )
    assert expected_fwhm_hz is not None
    assert after_target.active_fwhm_hz == expected_fwhm_hz
    expected_live_q = before_target.fast_center_hz / expected_fwhm_hz
    assert after_target.live_q == expected_live_q
    assert math.copysign(1.0, after_target.live_q) == math.copysign(
        1.0, expected_live_q
    )


@pytest.mark.parametrize(
    "failure_code",
    (
        "model_evaluation_failed",
        "optimizer_failed",
        "nonfinite_solution",
        "bounds_active",
        "rank_deficient",
        "ill_conditioned",
        "amplitude_unresolved",
        "residual_quality_failed",
    ),
)
def test_every_scientific_failure_shape_completes_without_refreshing_width(
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str,
) -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    _accept_fast_pairs(tracker, calibration, count=8)
    reserved, _ = _accept_sparse_prefix(tracker)
    before = tracker.estimate()
    before_target = before.identities[0]
    fifth = tracker.choose_next_query()
    assert fifth is reserved[4]
    observation = _sparse_observation(fifth)

    def failed_fit(source, configuration, queries, observations):
        del source, configuration
        return _synthetic_failure_result(queries, observations, failure_code)

    monkeypatch.setattr(tracker_module, "fit_sparse_linewidth", failed_fit)

    update = tracker.update(observation)

    scan = update.completed_sparse_scan
    assert scan is not None
    assert scan.status == "failure"
    assert scan.failure_code == failure_code
    after = update.estimate
    after_target = after.identities[0]
    assert (
        after_target.active_fwhm_hz,
        after_target.fwhm_source_kind,
        after_target.fwhm_source_scan_index,
        after_target.fwhm_reference_timestamp_s,
        after_target.fwhm_release_sequence_index,
        after_target.fwhm_release_timestamp_s,
    ) == (
        before_target.active_fwhm_hz,
        before_target.fwhm_source_kind,
        before_target.fwhm_source_scan_index,
        before_target.fwhm_reference_timestamp_s,
        before_target.fwhm_release_sequence_index,
        before_target.fwhm_release_timestamp_s,
    )
    assert after_target.live_q == (
        after_target.fast_center_hz / before_target.active_fwhm_hz
    )
    assert after.completed_sparse_scans == before.completed_sparse_scans + 1
    assert after.fast_pairs_since_scan == 0
    assert after.sparse_scan_history[-1] is scan
    assert after_target.latest_sparse_scan is scan
    assert (
        after_target.completed_sparse_scans
        == before_target.completed_sparse_scans + 1
    )
    if failure_code in {
        "bounds_active",
        "rank_deficient",
        "ill_conditioned",
        "amplitude_unresolved",
        "residual_quality_failed",
    }:
        assert scan.fitted_fwhm_hz is not None
        assert scan.fitted_fwhm_hz != after_target.active_fwhm_hz


@pytest.mark.parametrize("included", (False, True))
@pytest.mark.parametrize("success", (False, True))
def test_fifth_point_replays_every_resource_ledger_for_both_treatments(
    monkeypatch: pytest.MonkeyPatch, included: bool, success: bool
) -> None:
    configuration = SparseLinewidthConfiguration(integration_time_s=0.007)
    calibration = _calibration(included=included)
    metadata = _metadata(calibration, included=included)
    tracker = _reset_tracker(
        configuration=configuration,
        calibration=calibration,
        metadata=metadata,
    )
    _accept_fast_pairs(tracker, calibration, count=8)
    before = tracker.estimate()
    point_index = 0

    def observation_factory(query: SparseLinewidthQuery) -> EstimatorObservation:
        nonlocal point_index
        observation = _sparse_observation(
            query,
            fluorescence=0.87 + 0.01 * point_index,
            realized_photons=None if point_index in {1, 4} else 41 + point_index,
        )
        point_index += 1
        return observation

    reserved, prefix = _accept_sparse_prefix(
        tracker, observation_factory=observation_factory
    )
    fifth = tracker.choose_next_query()
    assert fifth is reserved[4]
    fifth_observation = observation_factory(fifth)

    def fit_result(source, fit_configuration, queries, fit_observations):
        assert source is calibration.source
        assert fit_configuration is tracker._configuration
        return _synthetic_fit_result(
            queries, fit_observations, success=success
        )

    monkeypatch.setattr(
        tracker_module, "fit_sparse_linewidth", fit_result, raising=False
    )
    expected_sparse = before.sparse_tracking_resources
    expected_tracking = before.tracking_resources
    expected_charged = before.charged_resources
    for observation in (*prefix, fifth_observation):
        expected_sparse = _independent_resource_transition(
            expected_sparse, observation, metadata
        )
        expected_tracking = _independent_resource_transition(
            expected_tracking, observation, metadata
        )
        expected_charged = _independent_resource_transition(
            expected_charged, observation, metadata
        )

    update = tracker.update(fifth_observation)

    estimate = update.estimate
    assert estimate.calibration_budget_treatment == calibration.budget_treatment
    assert estimate.calibration_resources == before.calibration_resources
    assert estimate.fast_tracking_resources == before.fast_tracking_resources
    _assert_resource_fields(estimate.sparse_tracking_resources, expected_sparse)
    _assert_resource_fields(estimate.tracking_resources, expected_tracking)
    _assert_resource_fields(estimate.charged_resources, expected_charged)
    assert estimate.sparse_tracking_resources.observations == 5
    assert estimate.tracking_resources.observations == before.accepted_observations + 5
    assert estimate.charged_resources != estimate.tracking_resources or not included


def test_fifth_point_cpu_interval_and_mode_total_folds_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    _accept_fast_pairs(tracker, calibration, count=8)
    reserved, _ = _accept_sparse_prefix(tracker)
    fifth = tracker.choose_next_query()
    assert fifth is reserved[4]
    fifth_observation = _sparse_observation(fifth)
    state = tracker._state
    assert state is not None
    old_fast_cpu = float(2**53)
    old_sparse_cpu = 2.0
    old_total_cpu = float(2**53)
    estimate = replace(
        state.estimate,
        fast_update_cpu_time_s=old_fast_cpu,
        sparse_update_cpu_time_s=old_sparse_cpu,
        total_update_cpu_time_s=old_total_cpu,
    )
    object.__setattr__(tracker, "_state", replace(state, estimate=estimate))
    events: list[str] = []
    clock_values = iter((10_000_000_000, 11_000_000_000))

    def clock() -> int:
        events.append("clock")
        return next(clock_values)

    def fit_result(source, configuration, queries, observations):
        del source, configuration
        events.append("fit")
        return _synthetic_fit_result(queries, observations, success=False)

    identity_constructor = tracker_module.CompositeIdentityEstimate
    resources_constructor = tracker_module.PublicAcquisitionResources

    def construct_identity(*args, **kwargs):
        events.append("identity")
        return identity_constructor(*args, **kwargs)

    def construct_resources(*args, **kwargs):
        events.append("resource")
        return resources_constructor(*args, **kwargs)

    monkeypatch.setattr(tracker_module.time, "process_time_ns", clock)
    monkeypatch.setattr(
        tracker_module, "fit_sparse_linewidth", fit_result, raising=False
    )
    monkeypatch.setattr(tracker_module, "CompositeIdentityEstimate", construct_identity)
    monkeypatch.setattr(
        tracker_module, "PublicAcquisitionResources", construct_resources
    )

    update = tracker.update(fifth_observation)

    assert update.update_cpu_time_s == 1.0
    assert events[0:2] == ["clock", "fit"]
    assert events.count("identity") == 8
    assert events.count("resource") == 3
    assert events[-1] == "clock"
    assert events.index("fit") < events.index("identity")
    assert events.index("identity") < events.index("resource")
    assert update.estimate.fast_update_cpu_time_s == old_fast_cpu
    assert update.estimate.sparse_update_cpu_time_s == old_sparse_cpu + 1.0
    assert update.estimate.total_update_cpu_time_s == old_total_cpu + 1.0
    assert update.estimate.total_update_cpu_time_s != (
        update.estimate.fast_update_cpu_time_s
        + update.estimate.sparse_update_cpu_time_s
    )


def test_nonrepresentable_live_q_rolls_back_as_aggregate_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration()
    tracker = _reset_tracker(calibration=calibration)
    _accept_fast_pairs(tracker, calibration, count=8)
    reserved, _ = _accept_sparse_prefix(tracker)
    fifth = tracker.choose_next_query()
    assert fifth is reserved[4]
    fifth_observation = _sparse_observation(fifth)
    before = tracker._state
    calls = 0

    def unrepresentable_live_q(source, configuration, queries, observations):
        nonlocal calls
        del source, configuration
        calls += 1
        return _synthetic_fit_result(
            queries,
            observations,
            success=True,
            fitted_fwhm_hz=np.nextafter(0.0, 1.0),
            fitted_local_center_hz=0.0,
        )

    monkeypatch.setattr(
        tracker_module,
        "fit_sparse_linewidth",
        unrepresentable_live_q,
        raising=False,
    )

    with pytest.raises(SparseLinewidthUpdateConstructionError) as raised:
        tracker.update(fifth_observation)

    assert calls == 1
    assert raised.value.code == "aggregate_estimate_construction_failed"
    assert tracker._state == before
