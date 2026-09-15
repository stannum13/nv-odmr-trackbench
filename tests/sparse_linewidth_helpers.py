"""Legal public sparse-linewidth record fixtures for contract tests."""

from __future__ import annotations

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
)


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
