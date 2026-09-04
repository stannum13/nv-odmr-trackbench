"""Shared helpers for calibrated two-point tracker tests."""

from __future__ import annotations

from collections.abc import Sequence

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators import (
    FitConfiguration,
    FitInitialGuess,
    InitializationDiagnostics,
    NormalizedFluorescenceProvenance,
    PublicAcquisitionResources,
    SpectrumFitResult,
    TwoPointCalibrationSource,
    TwoPointClockMapping,
    TwoPointIdentityBinding,
    TwoPointIdentityCalibration,
    TwoPointTrackerConfiguration,
)
from odmr_bench.models import Baseline, Resonance


def make_legal_fit_configuration(
    resonance_ids: Sequence[str] | None = None,
) -> FitConfiguration:
    return FitConfiguration(
        resonance_ids=(
            tuple(f"r{index}" for index in range(8))
            if resonance_ids is None
            else resonance_ids
        )
    )


def make_legal_source_fit(
    configuration: FitConfiguration | None = None,
) -> SpectrumFitResult:
    configuration = (
        make_legal_fit_configuration() if configuration is None else configuration
    )
    resonances = tuple(
        Resonance(
            resonance_id=resonance_id,
            center_hz=2.76e9 + index * 34e6,
            fwhm_hz=1.5e6,
            amplitude=0.02,
            eta=0.5,
        )
        for index, resonance_id in enumerate(configuration.resonance_ids)
    )
    baseline = Baseline(intercept=1.0, reference_hz=2.88e9)
    return SpectrumFitResult(
        success=True,
        failure_code=None,
        model_kind="pseudo_voigt",
        baseline_degree=1,
        resonance_estimates=resonances,
        baseline_estimate=baseline,
        diagnostics=InitializationDiagnostics(
            source="user",
            candidate_count=0,
            selected_indices=(),
            used_fallback=False,
            messages=(),
        ),
        initial_guess=FitInitialGuess(resonances, baseline),
        uncertainty=None,
        uncertainty_reason="not evaluated in two-point contract tests",
        scipy_status=1,
        scipy_message="converged",
        nfev=1,
        cost=0.0,
        residual_rmse=0.0,
        residual_scale=0.1,
        degrees_of_freedom=100,
        jacobian_rank=34,
    )


def make_legal_source_observations() -> tuple[EstimatorObservation, ...]:
    return (
        EstimatorObservation(0, 0.005, 2.74e9, 1.0, 0.005, 12_500.0),
        EstimatorObservation(1, 0.010, 3.02e9, 0.99, 0.005, 12_500.0),
    )


def make_legal_tracker_configuration() -> TwoPointTrackerConfiguration:
    return TwoPointTrackerConfiguration()


def make_legal_caller_asserted_source(
    *,
    source_observations: Sequence[EstimatorObservation] | None = None,
    source_fit: SpectrumFitResult | None = None,
    fit_configuration: FitConfiguration | None = None,
) -> TwoPointCalibrationSource:
    fit_configuration = (
        make_legal_fit_configuration()
        if fit_configuration is None
        else fit_configuration
    )
    source_fit = (
        make_legal_source_fit(fit_configuration) if source_fit is None else source_fit
    )
    source_observations = (
        make_legal_source_observations()
        if source_observations is None
        else source_observations
    )
    first_public_midpoint_s = (
        source_observations[0].timestamp_s
        - source_observations[0].integration_time_s / 2.0
    )
    last_public_midpoint_s = (
        source_observations[-1].timestamp_s
        - source_observations[-1].integration_time_s / 2.0
    )
    return TwoPointCalibrationSource(
        source_id="source-1",
        provenance="caller_asserted",
        source_fit=source_fit,
        fit_configuration=fit_configuration,
        identity_binding=TwoPointIdentityBinding(
            "require_expected_ids", tuple(f"r{index}" for index in range(8))
        ),
        resolved_resonance_ids=tuple(f"r{index}" for index in range(8)),
        source_observations=source_observations,
        fluorescence_provenance=NormalizedFluorescenceProvenance(
            "normalized_fluorescence", "declared", 2.5e6, ("declared",)
        ),
        source_frequency_overhead_s=0.0,
        source_frequency_min_hz=2.74e9,
        source_frequency_max_hz=3.02e9,
        source_first_sequence_index=0,
        source_last_sequence_index=1,
        source_start_timestamp_s=0.0,
        source_first_timestamp_s=0.005,
        source_last_timestamp_s=0.010,
        physical_fit_epoch_s=(
            first_public_midpoint_s
            + (last_public_midpoint_s - first_public_midpoint_s) / 2.0
        ),
        availability_sequence_index=1,
        availability_timestamp_s=0.010,
        safe_resources=PublicAcquisitionResources(2, 0.01, 25_000.0, 0, 2, 0.01),
        clock_mapping=TwoPointClockMapping("shared_clock", "clock", "clock", 1.0, 0.0),
    )


def make_legal_identity_calibrations(
    source: TwoPointCalibrationSource,
) -> tuple[TwoPointIdentityCalibration, ...]:
    return tuple(
        TwoPointIdentityCalibration(
            resonance_id=resonance.resonance_id,
            source_fit_index=index,
            calibration_center_hz=resonance.center_hz,
            calibration_fwhm_hz=resonance.fwhm_hz,
            calibration_amplitude=resonance.amplitude,
            calibration_eta=resonance.eta,
            offset_hz=0.35 * resonance.fwhm_hz,
            capture_radius_hz=0.20 * resonance.fwhm_hz,
            max_step_hz=0.10 * resonance.fwhm_hz,
            target_pair_depth=0.01,
            calibration_cell_lower_hz=resonance.center_hz - 10.0e6,
            calibration_cell_upper_hz=resonance.center_hz + 10.0e6,
            allowed_center_min_hz=resonance.center_hz - 8.0e6,
            allowed_center_max_hz=resonance.center_hz + 8.0e6,
        )
        for index, resonance in enumerate(source.source_fit.resonance_estimates)
    )
