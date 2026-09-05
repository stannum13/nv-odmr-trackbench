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
    TwoPointBudgetCeiling,
    TwoPointCalibrationSource,
    TwoPointClockMapping,
    TwoPointEstimate,
    TwoPointIdentityBinding,
    TwoPointIdentityCalibration,
    TwoPointIdentityEstimate,
    TwoPointPairResult,
    TwoPointPartialPair,
    TwoPointQuery,
    TwoPointTrackerConfiguration,
    TwoPointUpdate,
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


def make_legal_query(
    *,
    side: str = "minus",
    query_index: int = 0,
    pair_index: int = 0,
    identity_pair_index: int = 0,
    resonance_id: str = "r0",
    interrogation_center_hz: float = 2.76e9,
    expected_sequence_index: int = 0,
    expected_end_timestamp_s: float = 0.006,
) -> TwoPointQuery:
    offset_hz = 525_000.0
    return TwoPointQuery(
        query_index=query_index,
        pair_index=pair_index,
        identity_pair_index=identity_pair_index,
        resonance_id=resonance_id,
        side=side,
        interrogation_center_hz=interrogation_center_hz,
        frequency_hz=(
            interrogation_center_hz - offset_hz
            if side == "minus"
            else interrogation_center_hz + offset_hz
        ),
        integration_time_s=0.005,
        expected_sequence_index=expected_sequence_index,
        expected_end_timestamp_s=expected_end_timestamp_s,
        expected_nominal_exposure_photons=12_500.0,
    )


def make_legal_partial_pair() -> TwoPointPartialPair:
    query = make_legal_query()
    observation = EstimatorObservation(
        query.expected_sequence_index,
        query.expected_end_timestamp_s,
        query.frequency_hz,
        0.98,
        query.integration_time_s,
        query.expected_nominal_exposure_photons,
        12_250,
    )
    return TwoPointPartialPair(
        pair_index=query.pair_index,
        identity_pair_index=query.identity_pair_index,
        resonance_id=query.resonance_id,
        interrogation_center_hz=query.interrogation_center_hz,
        first_side=query.side,
        first_query=query,
        first_observation=observation,
    )


def make_legal_pair_result(
    *,
    pair_index: int = 0,
    identity_pair_index: int = 0,
    resonance_id: str = "r0",
    first_side: str | None = None,
    lock_state: str = "tracking",
    failure_code: str | None = None,
    numerical_prefix: int = 0,
) -> TwoPointPairResult:
    first_side = (
        "minus" if identity_pair_index % 2 == 0 else "plus"
    ) if first_side is None else first_side
    minus_is_first = first_side == "minus"
    minus_query = make_legal_query(
        side="minus",
        query_index=2 * pair_index + int(not minus_is_first),
        pair_index=pair_index,
        identity_pair_index=identity_pair_index,
        resonance_id=resonance_id,
        expected_sequence_index=int(not minus_is_first),
        expected_end_timestamp_s=0.006 if minus_is_first else 0.012,
    )
    plus_query = make_legal_query(
        side="plus",
        query_index=2 * pair_index + int(minus_is_first),
        pair_index=pair_index,
        identity_pair_index=identity_pair_index,
        resonance_id=resonance_id,
        expected_sequence_index=int(minus_is_first),
        expected_end_timestamp_s=0.012 if minus_is_first else 0.006,
    )
    minus_observation = EstimatorObservation(
        minus_query.expected_sequence_index,
        minus_query.expected_end_timestamp_s,
        minus_query.frequency_hz,
        0.98,
        0.005,
        12_500.0,
        12_250,
    )
    plus_observation = EstimatorObservation(
        plus_query.expected_sequence_index,
        plus_query.expected_end_timestamp_s,
        plus_query.frequency_hz,
        0.99,
        0.005,
        12_500.0,
        12_375,
    )
    successful = lock_state in {"tracking", "step_limited"}
    invalid_normalization = failure_code == "invalid_pair_normalization"
    numerical_failure = failure_code == "numerical_failure"
    common_mode_failure = failure_code == "common_mode_limit_exceeded"
    capture_failure = failure_code == "capture_exceeded"
    discriminator_present = (
        not invalid_normalization
        and (not numerical_failure or numerical_prefix >= 1)
    )
    common_mode_present = (
        not invalid_normalization
        and (not numerical_failure or numerical_prefix >= 2)
    )
    innovation_present = (
        not invalid_normalization
        and not common_mode_failure
        and (not numerical_failure or numerical_prefix >= 3)
    )
    requested_step_present = (
        not invalid_normalization
        and not common_mode_failure
        and not capture_failure
        and (not numerical_failure or numerical_prefix >= 4)
    )
    candidate_present = (
        not invalid_normalization
        and not common_mode_failure
        and not capture_failure
        and (not numerical_failure or numerical_prefix >= 5)
    )
    geometry_available = not invalid_normalization and (
        not numerical_failure or numerical_prefix >= 1
    )
    return TwoPointPairResult(
        pair_index=pair_index,
        identity_pair_index=identity_pair_index,
        resonance_id=resonance_id,
        interrogation_center_hz=2.76e9,
        first_side=first_side,
        minus_query=minus_query,
        plus_query=plus_query,
        minus_observation=minus_observation,
        plus_observation=plus_observation,
        pair_reference_timestamp_s=(
            (0.006 - 0.005 / 2.0)
            + ((0.012 - 0.005 / 2.0) - (0.006 - 0.005 / 2.0)) / 2.0
        ),
        release_sequence_index=1,
        release_timestamp_s=0.012,
        discriminator=0.01 if discriminator_present else None,
        zero_discriminator=0.0 if geometry_available else None,
        discriminator_slope_per_hz=1.0e-6 if geometry_available else None,
        raw_innovation_hz=10_000.0 if innovation_present else None,
        requested_step_hz=10_000.0 if requested_step_present else None,
        candidate_center_hz=(
            2.760005e9
            if lock_state == "step_limited"
            else 2.76001e9 if candidate_present else None
        ),
        applied_step_hz=(
            10_000.0
            if lock_state == "tracking"
            else 5_000.0
            if lock_state == "step_limited"
            else 0.0
        ),
        common_mode_target_depths=(
            2.0
            if failure_code == "common_mode_limit_exceeded"
            else 0.0 if common_mode_present else None
        ),
        lock_state=lock_state,
        failure_code=None if successful else failure_code,
    )


def make_legal_identity_estimates(
    *,
    current_timestamp_s: float,
    latest_pair: TwoPointPairResult | None = None,
) -> tuple[TwoPointIdentityEstimate, ...]:
    source = make_legal_caller_asserted_source()
    calibrations = make_legal_identity_calibrations(source)
    identities = []
    for calibration in calibrations:
        is_pair_identity = (
            latest_pair is not None
            and latest_pair.resonance_id == calibration.resonance_id
        )
        if is_pair_identity:
            active_reference_timestamp_s = latest_pair.pair_reference_timestamp_s
            active_release_timestamp_s = latest_pair.release_timestamp_s
            center_hz = latest_pair.candidate_center_hz
            assert center_hz is not None
            identity = TwoPointIdentityEstimate(
                resonance_id=calibration.resonance_id,
                center_hz=center_hz,
                calibration_fwhm_hz=calibration.calibration_fwhm_hz,
                calibration_cell_lower_hz=calibration.calibration_cell_lower_hz,
                calibration_cell_upper_hz=calibration.calibration_cell_upper_hz,
                allowed_center_min_hz=calibration.allowed_center_min_hz,
                allowed_center_max_hz=calibration.allowed_center_max_hz,
                active_source_kind="pair",
                active_source_pair_index=latest_pair.pair_index,
                active_reference_timestamp_s=active_reference_timestamp_s,
                active_release_sequence_index=latest_pair.release_sequence_index,
                active_release_timestamp_s=active_release_timestamp_s,
                estimate_age_sequence_indices=0,
                estimate_age_s=current_timestamp_s - active_reference_timestamp_s,
                release_age_s=current_timestamp_s - active_release_timestamp_s,
                completed_pairs=1,
                lock_state=latest_pair.lock_state,
                failure_code=latest_pair.failure_code,
                latest_pair=latest_pair,
            )
        else:
            identity = TwoPointIdentityEstimate(
                resonance_id=calibration.resonance_id,
                center_hz=calibration.calibration_center_hz,
                calibration_fwhm_hz=calibration.calibration_fwhm_hz,
                calibration_cell_lower_hz=calibration.calibration_cell_lower_hz,
                calibration_cell_upper_hz=calibration.calibration_cell_upper_hz,
                allowed_center_min_hz=calibration.allowed_center_min_hz,
                allowed_center_max_hz=calibration.allowed_center_max_hz,
                active_source_kind="calibration",
                active_source_pair_index=None,
                active_reference_timestamp_s=-1.0,
                active_release_sequence_index=None,
                active_release_timestamp_s=0.0,
                estimate_age_sequence_indices=None,
                estimate_age_s=current_timestamp_s + 1.0,
                release_age_s=current_timestamp_s,
                completed_pairs=0,
                lock_state="calibrated",
                failure_code=None,
                latest_pair=None,
            )
        identities.append(identity)
    return tuple(identities)


def make_legal_estimate(state: str = "boundary") -> TwoPointEstimate:
    source = make_legal_caller_asserted_source()
    zero_resources = PublicAcquisitionResources(0, 0.0, 0.0, 0, 0, 0.0)
    incomplete_pair = (
        make_legal_partial_pair() if state in {"partial", "second"} else None
    )
    pair = (
        make_legal_pair_result()
        if state in {"complete", "next", "stopped"}
        else None
    )
    if pair is not None:
        current_sequence_index = pair.release_sequence_index
        current_timestamp_s = pair.release_timestamp_s
        accepted_observations = 2
        completed_pairs = 1
        pair_history = (pair,)
        tracking_resources = PublicAcquisitionResources(
            2, 0.01, 25_000.0, 24_625, 0, 0.012
        )
    elif incomplete_pair is not None:
        current_sequence_index = incomplete_pair.first_observation.sequence_index
        current_timestamp_s = incomplete_pair.first_observation.timestamp_s
        accepted_observations = 1
        completed_pairs = 0
        pair_history = ()
        tracking_resources = PublicAcquisitionResources(
            1, 0.005, 12_500.0, 12_250, 0, 0.006
        )
    else:
        current_sequence_index = None
        current_timestamp_s = 0.0
        accepted_observations = 0
        completed_pairs = 0
        pair_history = ()
        tracking_resources = zero_resources
    pending_query = None
    if state == "first":
        pending_query = make_legal_query()
    elif state == "second":
        pending_query = make_legal_query(
            side="plus",
            query_index=1,
            expected_sequence_index=1,
            expected_end_timestamp_s=0.012,
        )
    elif state == "next":
        pending_query = make_legal_query(
            side="minus",
            query_index=2,
            pair_index=1,
            resonance_id="r1",
            interrogation_center_hz=2.794e9,
            expected_sequence_index=2,
            expected_end_timestamp_s=0.018,
        )
    identities = make_legal_identity_estimates(
        current_timestamp_s=current_timestamp_s,
        latest_pair=pair,
    )
    return TwoPointEstimate(
        identities=identities,
        calibration_source_id=source.source_id,
        calibration_source_provenance=source.provenance,
        calibration_budget_treatment="conditional_free_precalibration",
        current_sequence_index=current_sequence_index,
        current_timestamp_s=current_timestamp_s,
        accepted_observations=accepted_observations,
        completed_pairs=completed_pairs,
        incomplete_pair=incomplete_pair,
        pending_query=pending_query,
        pair_history=pair_history,
        tracking_resources=tracking_resources,
        calibration_resources=source.safe_resources,
        charged_resources=tracking_resources,
        budget_ceiling=TwoPointBudgetCeiling(100, None, None, None),
        stopped_reason="budget_exhausted" if state == "stopped" else None,
        seed=7,
    )


def make_legal_update(side: str = "first") -> TwoPointUpdate:
    if side == "first":
        estimate = make_legal_estimate("partial")
        partial = estimate.incomplete_pair
        assert partial is not None
        return TwoPointUpdate(
            partial.first_query,
            partial.first_observation,
            None,
            estimate,
        )
    estimate = make_legal_estimate("complete")
    pair = estimate.pair_history[-1]
    query = pair.plus_query if pair.first_side == "minus" else pair.minus_query
    observation = (
        pair.plus_observation
        if pair.first_side == "minus"
        else pair.minus_observation
    )
    return TwoPointUpdate(query, observation, pair, estimate)
