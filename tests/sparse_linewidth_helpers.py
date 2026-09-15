"""Legal public sparse-linewidth record fixtures for contract tests."""

from __future__ import annotations

from dataclasses import replace

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators import (
    CompositeIdentityEstimate,
    SparseLinewidthCompositeEstimate,
    SparseLinewidthConfiguration,
    SparseLinewidthQuery,
    SparseLinewidthScanResult,
    SparsePartialScan,
)
from odmr_bench.estimators.two_point_types import (
    PublicAcquisitionResources,
    TwoPointBudgetCeiling,
    TwoPointPairResult,
)
from tests.two_point_helpers import make_legal_pair_result


def _overridden(
    values: dict[str, object], overrides: dict[str, object]
) -> dict[str, object]:
    unknown = set(overrides).difference(values)
    if unknown:
        raise TypeError(f"unknown fixture overrides: {sorted(unknown)!r}")
    return values | overrides


def make_sparse_query(**overrides: object) -> SparseLinewidthQuery:
    values: dict[str, object] = {
        "acquisition_index": 0,
        "scan_index": 0,
        "identity_scan_index": 0,
        "point_index": 0,
        "resonance_id": "r0",
        "offset_multiplier": 0.5,
        "frozen_fast_center_hz": 2.87e9,
        "frozen_fast_center_source_kind": "calibration",
        "frozen_fast_center_source_pair_index": None,
        "frozen_fast_center_reference_timestamp_s": 0.0,
        "frozen_fast_center_release_sequence_index": None,
        "frozen_fast_center_release_timestamp_s": 0.0,
        "frozen_prior_fwhm_hz": 1.0e6,
        "frozen_fwhm_source_kind": "calibration",
        "frozen_fwhm_source_scan_index": None,
        "frozen_fwhm_reference_timestamp_s": 0.0,
        "frozen_fwhm_release_sequence_index": None,
        "frozen_fwhm_release_timestamp_s": 0.0,
        "frequency_hz": 2.8705e9,
        "integration_time_s": 0.005,
        "expected_sequence_index": 0,
        "expected_end_timestamp_s": 0.005,
        "expected_nominal_exposure_photons": 1.0,
    }
    return SparseLinewidthQuery(**_overridden(values, overrides))  # type: ignore[arg-type]


def _observation(query: SparseLinewidthQuery) -> EstimatorObservation:
    return EstimatorObservation(
        sequence_index=query.expected_sequence_index,
        timestamp_s=query.expected_end_timestamp_s,
        frequency_hz=query.frequency_hz,
        fluorescence=1.0,
        integration_time_s=query.integration_time_s,
        nominal_exposure_photons=query.expected_nominal_exposure_photons,
    )


def _queries(length: int) -> tuple[SparseLinewidthQuery, ...]:
    offsets = (0.5, -1.0, 0.0, 1.0, -0.5)
    return tuple(
        make_sparse_query(
            acquisition_index=index,
            point_index=index,
            offset_multiplier=offsets[index],
            frequency_hz=2.87e9 + offsets[index] * 1.0e6,
            expected_sequence_index=index,
            expected_end_timestamp_s=(index + 1) * 0.005,
        )
        for index in range(length)
    )


def make_partial_scan(**overrides: object) -> SparsePartialScan:
    queries = _queries(1)
    values: dict[str, object] = {
        **{
            name: getattr(queries[0], name)
            for name in (
                "scan_index",
                "identity_scan_index",
                "resonance_id",
                "frozen_fast_center_hz",
                "frozen_fast_center_source_kind",
                "frozen_fast_center_source_pair_index",
                "frozen_fast_center_reference_timestamp_s",
                "frozen_fast_center_release_sequence_index",
                "frozen_fast_center_release_timestamp_s",
                "frozen_prior_fwhm_hz",
                "frozen_fwhm_source_kind",
                "frozen_fwhm_source_scan_index",
                "frozen_fwhm_reference_timestamp_s",
                "frozen_fwhm_release_sequence_index",
                "frozen_fwhm_release_timestamp_s",
            )
        },
        "queries": queries,
        "observations": tuple(_observation(query) for query in queries),
    }
    return SparsePartialScan(**_overridden(values, overrides))  # type: ignore[arg-type]


def make_scan_result(**overrides: object) -> SparseLinewidthScanResult:
    queries = _queries(5)
    values: dict[str, object] = {
        **{
            name: getattr(queries[0], name)
            for name in (
                "scan_index",
                "identity_scan_index",
                "resonance_id",
                "frozen_fast_center_hz",
                "frozen_fast_center_source_kind",
                "frozen_fast_center_source_pair_index",
                "frozen_fast_center_reference_timestamp_s",
                "frozen_fast_center_release_sequence_index",
                "frozen_fast_center_release_timestamp_s",
                "frozen_prior_fwhm_hz",
                "frozen_fwhm_source_kind",
                "frozen_fwhm_source_scan_index",
                "frozen_fwhm_reference_timestamp_s",
                "frozen_fwhm_release_sequence_index",
                "frozen_fwhm_release_timestamp_s",
            )
        },
        "queries": queries,
        "observations": tuple(_observation(query) for query in queries),
        "public_reference_timestamp_s": 0.0125,
        "release_sequence_index": 4,
        "release_timestamp_s": 0.025,
        "status": "success",
        "failure_code": None,
        "fitted_center_correction_hz": 0.0,
        "fitted_local_center_hz": 2.87e9,
        "fitted_fwhm_hz": 1.0e6,
        "fitted_amplitude": 1.0,
        "fitted_baseline_offset": 0.0,
        "fitted_q": 2870.0,
        "rmse": 0.01,
        "amplitude_normalized_rmse": 0.01,
        "scaled_jacobian_rank": 4,
        "scaled_jacobian_condition": 2.0,
        "scipy_status": 1,
        "scipy_message": "success",
        "nfev": 1,
        "fit_cpu_time_s": 0.001,
    }
    values = _overridden(values, overrides)
    if (
        "fitted_local_center_hz" in overrides
        and "fitted_center_correction_hz" not in overrides
    ):
        values["fitted_center_correction_hz"] = (
            values["fitted_local_center_hz"] - values["frozen_fast_center_hz"]
        )
    return SparseLinewidthScanResult(**values)  # type: ignore[arg-type]


def make_composite_identity(**overrides: object) -> CompositeIdentityEstimate:
    values: dict[str, object] = {
        "resonance_id": "r0",
        "fast_center_hz": 2.87e9,
        "fast_center_source_kind": "calibration",
        "fast_center_source_pair_index": None,
        "fast_center_reference_timestamp_s": 0.0,
        "fast_center_release_sequence_index": None,
        "fast_center_release_timestamp_s": 0.0,
        "active_fwhm_hz": 1.0e6,
        "fwhm_source_kind": "calibration",
        "fwhm_source_scan_index": None,
        "fwhm_reference_timestamp_s": 0.0,
        "fwhm_release_sequence_index": None,
        "fwhm_release_timestamp_s": 0.0,
        "live_q": 2870.0,
        "center_age_s": 0.0,
        "fwhm_age_s": 0.0,
        "center_release_age_s": 0.0,
        "fwhm_release_age_s": 0.0,
        "completed_fast_pairs": 0,
        "completed_sparse_scans": 0,
        "latest_fast_pair": None,
        "latest_sparse_scan": None,
    }
    return CompositeIdentityEstimate(**_overridden(values, overrides))  # type: ignore[arg-type]


def _resources() -> PublicAcquisitionResources:
    return PublicAcquisitionResources(0, 0.0, 0.0, 0, 0, 0.0)


def make_composite_estimate(**overrides: object) -> SparseLinewidthCompositeEstimate:
    identities = tuple(
        make_composite_identity(resonance_id=f"r{index}") for index in range(8)
    )
    resources = _resources()
    values: dict[str, object] = {
        "configuration": SparseLinewidthConfiguration(),
        "identities": identities,
        "calibration_source_id": "source",
        "calibration_source_provenance": "verified_factory_acquisition",
        "calibration_budget_treatment": "conditional_free_precalibration",
        "pending_mode": None,
        "pending_query": None,
        "incomplete_fast_pair": None,
        "incomplete_sparse_scan": None,
        "fast_pair_history": (),
        "sparse_scan_history": (),
        "accepted_observations": 0,
        "completed_fast_pairs": 0,
        "completed_sparse_scans": 0,
        "fast_pairs_since_scan": 0,
        "current_sequence_index": None,
        "current_timestamp_s": 0.0,
        "fast_tracking_resources": resources,
        "sparse_tracking_resources": resources,
        "tracking_resources": resources,
        "calibration_resources": resources,
        "charged_resources": resources,
        "budget_ceiling": TwoPointBudgetCeiling(
            max_observations=1,
            max_integration_time_s=None,
            max_nominal_exposure_photons=None,
            max_virtual_elapsed_time_s=None,
        ),
        "stopped_reason": None,
        "sparse_geometry_diagnostic": None,
        "fast_update_cpu_time_s": 0.0,
        "sparse_update_cpu_time_s": 0.0,
        "total_update_cpu_time_s": 0.0,
        "seed": 0,
    }
    return SparseLinewidthCompositeEstimate(**_overridden(values, overrides))  # type: ignore[arg-type]


def _scheduled_fast_pair(
    pair_index: int, first_sequence_index: int
) -> TwoPointPairResult:
    identity_pair_index = pair_index // 8
    resonance_id = f"r{pair_index % 8}"
    base = make_legal_pair_result(
        pair_index=pair_index,
        identity_pair_index=identity_pair_index,
        resonance_id=resonance_id,
    )
    minus_sequence_index = first_sequence_index + int(base.first_side == "plus")
    plus_sequence_index = first_sequence_index + int(base.first_side == "minus")
    minus_query = replace(
        base.minus_query,
        expected_sequence_index=minus_sequence_index,
        expected_end_timestamp_s=(minus_sequence_index + 1) * 0.005,
    )
    plus_query = replace(
        base.plus_query,
        expected_sequence_index=plus_sequence_index,
        expected_end_timestamp_s=(plus_sequence_index + 1) * 0.005,
    )
    minus_observation = replace(
        base.minus_observation,
        sequence_index=minus_query.expected_sequence_index,
        timestamp_s=minus_query.expected_end_timestamp_s,
    )
    plus_observation = replace(
        base.plus_observation,
        sequence_index=plus_query.expected_sequence_index,
        timestamp_s=plus_query.expected_end_timestamp_s,
    )
    first_observation, second_observation = (
        (minus_observation, plus_observation)
        if base.first_side == "minus"
        else (plus_observation, minus_observation)
    )
    first_midpoint = (
        first_observation.timestamp_s - first_observation.integration_time_s / 2.0
    )
    second_midpoint = (
        second_observation.timestamp_s
        - second_observation.integration_time_s / 2.0
    )
    return replace(
        base,
        minus_query=minus_query,
        plus_query=plus_query,
        minus_observation=minus_observation,
        plus_observation=plus_observation,
        pair_reference_timestamp_s=first_midpoint
        + (second_midpoint - first_midpoint) / 2.0,
        release_sequence_index=second_observation.sequence_index,
        release_timestamp_s=second_observation.timestamp_s,
    )


def _scheduled_sparse_scan(
    scan_index: int,
    first_acquisition_index: int,
    fast_pair: TwoPointPairResult,
    fwhm_source: SparseLinewidthScanResult | None,
    *,
    failed: bool,
) -> SparseLinewidthScanResult:
    identity_scan_index = scan_index // 8
    offsets = (
        (0.5, -1.0, 0.0, 1.0, -0.5)
        if identity_scan_index % 2 == 0
        else (-0.5, 1.0, 0.0, -1.0, 0.5)
    )
    fast_center_hz = fast_pair.candidate_center_hz
    assert fast_center_hz is not None
    prior_fwhm_hz = (
        1.0e6 if fwhm_source is None else fwhm_source.fitted_fwhm_hz
    )
    assert prior_fwhm_hz is not None
    source_snapshot: dict[str, object] = {
        "frozen_fast_center_hz": fast_center_hz,
        "frozen_fast_center_source_kind": "pair",
        "frozen_fast_center_source_pair_index": fast_pair.pair_index,
        "frozen_fast_center_reference_timestamp_s": (
            fast_pair.pair_reference_timestamp_s
        ),
        "frozen_fast_center_release_sequence_index": (
            fast_pair.release_sequence_index
        ),
        "frozen_fast_center_release_timestamp_s": fast_pair.release_timestamp_s,
        "frozen_prior_fwhm_hz": prior_fwhm_hz,
        "frozen_fwhm_source_kind": (
            "calibration" if fwhm_source is None else "scan"
        ),
        "frozen_fwhm_source_scan_index": (
            None if fwhm_source is None else fwhm_source.scan_index
        ),
        "frozen_fwhm_reference_timestamp_s": (
            0.0
            if fwhm_source is None
            else fwhm_source.public_reference_timestamp_s
        ),
        "frozen_fwhm_release_sequence_index": (
            None if fwhm_source is None else fwhm_source.release_sequence_index
        ),
        "frozen_fwhm_release_timestamp_s": (
            0.0 if fwhm_source is None else fwhm_source.release_timestamp_s
        ),
    }
    queries = tuple(
        make_sparse_query(
            acquisition_index=first_acquisition_index + point_index,
            scan_index=scan_index,
            identity_scan_index=identity_scan_index,
            point_index=point_index,
            resonance_id=f"r{scan_index % 8}",
            offset_multiplier=offset_multiplier,
            frequency_hz=fast_center_hz + offset_multiplier * prior_fwhm_hz,
            expected_sequence_index=first_acquisition_index + point_index,
            expected_end_timestamp_s=(
                first_acquisition_index + point_index + 1
            )
            * 0.005,
            **source_snapshot,
        )
        for point_index, offset_multiplier in enumerate(offsets)
    )
    observations = tuple(_observation(query) for query in queries)
    midpoints = tuple(
        item.timestamp_s - item.integration_time_s / 2.0 for item in observations
    )
    public_reference_timestamp_s = midpoints[0]
    for count, midpoint in enumerate(midpoints[1:], start=2):
        public_reference_timestamp_s = public_reference_timestamp_s + (
            midpoint - public_reference_timestamp_s
        ) / count
    source_overrides: dict[str, object] = {
        "queries": queries,
        "observations": observations,
        "scan_index": scan_index,
        "identity_scan_index": identity_scan_index,
        "resonance_id": f"r{scan_index % 8}",
        **source_snapshot,
        "public_reference_timestamp_s": public_reference_timestamp_s,
        "release_sequence_index": observations[-1].sequence_index,
        "release_timestamp_s": observations[-1].timestamp_s,
        "fitted_center_correction_hz": 0.0,
        "fitted_local_center_hz": fast_center_hz,
        "fitted_fwhm_hz": prior_fwhm_hz,
        "fitted_q": fast_center_hz / prior_fwhm_hz,
    }
    if failed:
        source_overrides.update(
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
    return make_scan_result(**source_overrides)


def _resources_for(
    observations: tuple[EstimatorObservation, ...],
) -> PublicAcquisitionResources:
    integration_time_s = 0.0
    nominal_exposure_photons = 0.0
    realized_photons = 0
    observations_without_realized_counts = 0
    for observation in observations:
        integration_time_s = integration_time_s + observation.integration_time_s
        nominal_exposure_photons = (
            nominal_exposure_photons + observation.nominal_exposure_photons
        )
        realized_photons += (
            0 if observation.realized_photons is None else observation.realized_photons
        )
        observations_without_realized_counts += int(
            observation.realized_photons is None
        )
    return PublicAcquisitionResources(
        len(observations),
        integration_time_s,
        nominal_exposure_photons,
        realized_photons,
        observations_without_realized_counts,
        integration_time_s,
    )


def make_scan_source_retention_estimate() -> SparseLinewidthCompositeEstimate:
    """Build the public period-1 schedule through r0's second sparse scan."""
    fast_history = tuple(
        _scheduled_fast_pair(pair_index, 7 * pair_index) for pair_index in range(9)
    )
    sparse_history: list[SparseLinewidthScanResult] = []
    for scan_index, fast_pair in enumerate(fast_history):
        sparse_history.append(
            _scheduled_sparse_scan(
                scan_index,
                7 * scan_index + 2,
                fast_pair,
                sparse_history[0] if scan_index == 8 else None,
                failed=scan_index == 8,
            )
        )
    sparse_history_tuple = tuple(sparse_history)
    current_timestamp_s = sparse_history_tuple[-1].release_timestamp_s
    identities = []
    for identity_index in range(8):
        own_pairs = fast_history[identity_index::8]
        own_scans = sparse_history_tuple[identity_index::8]
        fast_source = own_pairs[-1]
        successful_scan = own_scans[0]
        active_fwhm_hz = successful_scan.fitted_fwhm_hz
        assert fast_source.candidate_center_hz is not None
        assert active_fwhm_hz is not None
        identities.append(
            make_composite_identity(
                resonance_id=f"r{identity_index}",
                fast_center_hz=fast_source.candidate_center_hz,
                fast_center_source_kind="pair",
                fast_center_source_pair_index=fast_source.pair_index,
                fast_center_reference_timestamp_s=(
                    fast_source.pair_reference_timestamp_s
                ),
                fast_center_release_sequence_index=fast_source.release_sequence_index,
                fast_center_release_timestamp_s=fast_source.release_timestamp_s,
                active_fwhm_hz=active_fwhm_hz,
                fwhm_source_kind="scan",
                fwhm_source_scan_index=successful_scan.scan_index,
                fwhm_reference_timestamp_s=successful_scan.public_reference_timestamp_s,
                fwhm_release_sequence_index=successful_scan.release_sequence_index,
                fwhm_release_timestamp_s=successful_scan.release_timestamp_s,
                live_q=fast_source.candidate_center_hz / active_fwhm_hz,
                center_age_s=current_timestamp_s
                - fast_source.pair_reference_timestamp_s,
                fwhm_age_s=current_timestamp_s
                - successful_scan.public_reference_timestamp_s,
                center_release_age_s=current_timestamp_s
                - fast_source.release_timestamp_s,
                fwhm_release_age_s=current_timestamp_s
                - successful_scan.release_timestamp_s,
                completed_fast_pairs=len(own_pairs),
                completed_sparse_scans=len(own_scans),
                latest_fast_pair=fast_source,
                latest_sparse_scan=own_scans[-1],
            )
        )
    fast_observations = tuple(
        observation
        for pair in fast_history
        for observation in (
            (pair.minus_observation, pair.plus_observation)
            if pair.first_side == "minus"
            else (pair.plus_observation, pair.minus_observation)
        )
    )
    sparse_observations = tuple(
        observation
        for scan in sparse_history_tuple
        for observation in scan.observations
    )
    tracking_observations = tuple(
        observation
        for pair, scan in zip(fast_history, sparse_history_tuple, strict=True)
        for observation in (
            *(
                (pair.minus_observation, pair.plus_observation)
                if pair.first_side == "minus"
                else (pair.plus_observation, pair.minus_observation)
            ),
            *scan.observations,
        )
    )
    zero_resources = _resources()
    tracking_resources = _resources_for(tracking_observations)
    return make_composite_estimate(
        configuration=SparseLinewidthConfiguration(scan_period_fast_pairs=1),
        identities=tuple(identities),
        fast_pair_history=fast_history,
        sparse_scan_history=sparse_history_tuple,
        accepted_observations=len(tracking_observations),
        completed_fast_pairs=len(fast_history),
        completed_sparse_scans=len(sparse_history_tuple),
        fast_pairs_since_scan=0,
        current_sequence_index=tracking_observations[-1].sequence_index,
        current_timestamp_s=current_timestamp_s,
        fast_tracking_resources=_resources_for(fast_observations),
        sparse_tracking_resources=_resources_for(sparse_observations),
        tracking_resources=tracking_resources,
        calibration_resources=zero_resources,
        charged_resources=tracking_resources,
        budget_ceiling=TwoPointBudgetCeiling(100, None, None, None),
    )
