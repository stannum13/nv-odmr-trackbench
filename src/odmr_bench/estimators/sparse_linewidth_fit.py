"""Pure geometry for one frozen sparse five-point linewidth fit."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators.sparse_linewidth_types import (
    CompositeIdentityEstimate,
    SparseLinewidthConfiguration,
    SparseLinewidthQuery,
    SparseLinewidthResetError,
    SparseLinewidthScanResult,
)
from odmr_bench.estimators.two_point_calibration import _evaluate_bound_source_model
from odmr_bench.estimators.two_point_types import (
    TwoPointCalibration,
    TwoPointCalibrationSource,
    TwoPointIdentityCalibration,
)

_EVEN_MULTIPLIERS = (0.5, -1.0, 0.0, 1.0, -0.5)
_ODD_MULTIPLIERS = tuple(reversed(_EVEN_MULTIPLIERS))


@dataclass(frozen=True, slots=True)
class _SparseFitGeometry:
    """Immutable fit-only facts, deliberately separate from query scheduling."""

    resonance_id: str
    offset_multipliers: tuple[float, float, float, float, float]
    frequencies_hz: tuple[float, float, float, float, float]
    frozen_fast_center_hz: float
    frozen_prior_fwhm_hz: float
    frozen_source_amplitude: float
    calibration_cell_lower_hz: float
    calibration_cell_upper_hz: float
    source_frequency_min_hz: float
    source_frequency_max_hz: float
    lower_bounds_scaled: tuple[float, float, float, float]
    upper_bounds_scaled: tuple[float, float, float, float]
    initial_guess_scaled: tuple[float, float, float, float]


class _SparseGeometryConstructionError(SparseLinewidthResetError):
    """Typed reset failure retaining private diagnostic construction facts."""

    geometry_failure_code: str
    proposed_frequency_min_hz: float | None
    proposed_frequency_max_hz: float | None

    def __init__(
        self,
        failure_code: str,
        message: str,
        *,
        proposed_frequency_min_hz: float | None,
        proposed_frequency_max_hz: float | None,
    ) -> None:
        self.geometry_failure_code = failure_code
        self.proposed_frequency_min_hz = proposed_frequency_min_hz
        self.proposed_frequency_max_hz = proposed_frequency_max_hz
        super().__init__("invalid_base_sparse_geometry", f"{failure_code}: {message}")


def _geometry_error(
    failure_code: str,
    message: str,
    *,
    proposed_frequency_min_hz: float | None = None,
    proposed_frequency_max_hz: float | None = None,
) -> None:
    raise _SparseGeometryConstructionError(
        failure_code,
        message,
        proposed_frequency_min_hz=proposed_frequency_min_hz,
        proposed_frequency_max_hz=proposed_frequency_max_hz,
    )


def _nonnegative_index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _geometry_error("empty_fit_bounds", f"{name} must be a nonnegative integer")
    canonical = int(value)
    if canonical < 0:
        _geometry_error("empty_fit_bounds", f"{name} must be a nonnegative integer")
    return canonical


def _identity_calibration(
    calibration: TwoPointCalibration, resonance_id: str
) -> TwoPointIdentityCalibration:
    for candidate in calibration.identities:
        if candidate.resonance_id == resonance_id:
            return candidate
    _geometry_error("empty_fit_bounds", "identity is absent from calibration")
    raise AssertionError("unreachable")


def _construct_geometry(
    calibration: TwoPointCalibration,
    configuration: SparseLinewidthConfiguration,
    seeded_identity: TwoPointIdentityCalibration,
    *,
    q0_hz: float,
    w0_hz: float,
    scan_index: int,
    identity_scan_index: int,
) -> _SparseFitGeometry:
    del scan_index
    a0 = seeded_identity.calibration_amplitude
    if not (
        math.isfinite(q0_hz)
        and math.isfinite(w0_hz)
        and w0_hz > 0.0
        and math.isfinite(a0)
        and a0 > 0.0
    ):
        _geometry_error("empty_fit_bounds", "frozen fit facts must be finite")

    proposed_lower_hz = q0_hz - w0_hz
    if not math.isfinite(proposed_lower_hz):
        _geometry_error("nonrepresentable_frequency_lower", "q0 - w0 is not finite")
    proposed_upper_hz = q0_hz + w0_hz
    if not math.isfinite(proposed_upper_hz):
        _geometry_error("nonrepresentable_frequency_upper", "q0 + w0 is not finite")

    offset_multipliers = (
        _EVEN_MULTIPLIERS if identity_scan_index % 2 == 0 else _ODD_MULTIPLIERS
    )
    frequencies_hz = tuple(
        q0_hz + multiplier * w0_hz for multiplier in offset_multipliers
    )
    if not all(math.isfinite(frequency_hz) for frequency_hz in frequencies_hz):
        _geometry_error(
            "nonrepresentable_frequency_upper", "query frequency is not finite"
        )

    source_configuration = calibration.source.fit_configuration
    lower_bounds = (
        -configuration.center_correction_limit_fwhm_fraction,
        max(
            configuration.min_fwhm_prior_ratio,
            source_configuration.min_fwhm_hz / w0_hz,
        ),
        0.0,
        -configuration.baseline_offset_source_amplitude_fraction,
    )
    upper_bounds = (
        configuration.center_correction_limit_fwhm_fraction,
        min(
            configuration.max_fwhm_prior_ratio,
            source_configuration.max_fwhm_hz / w0_hz,
        ),
        min(
            configuration.max_amplitude_source_ratio,
            source_configuration.max_amplitude / a0,
        ),
        configuration.baseline_offset_source_amplitude_fraction,
    )
    initial_guess = (0.0, 1.0, 1.0, 0.0)
    if not (
        all(math.isfinite(value) for value in (*lower_bounds, *upper_bounds))
        and all(
            lower < initial < upper
            for lower, initial, upper in zip(
                lower_bounds, initial_guess, upper_bounds, strict=True
            )
        )
    ):
        _geometry_error(
            "empty_fit_bounds",
            "scaled fit bounds lack a strict interior",
            proposed_frequency_min_hz=proposed_lower_hz,
            proposed_frequency_max_hz=proposed_upper_hz,
        )

    cell_lower_hz = seeded_identity.calibration_cell_lower_hz
    cell_upper_hz = seeded_identity.calibration_cell_upper_hz
    local_center_lower_hz = (
        q0_hz - configuration.center_correction_limit_fwhm_fraction * w0_hz
    )
    local_center_upper_hz = (
        q0_hz + configuration.center_correction_limit_fwhm_fraction * w0_hz
    )
    if not (
        cell_lower_hz <= local_center_lower_hz <= local_center_upper_hz <= cell_upper_hz
        and all(
            cell_lower_hz <= frequency_hz <= cell_upper_hz
            for frequency_hz in frequencies_hz
        )
    ):
        _geometry_error(
            "calibration_cell_violation",
            "fit geometry leaves calibration cell",
            proposed_frequency_min_hz=proposed_lower_hz,
            proposed_frequency_max_hz=proposed_upper_hz,
        )

    source_lower_hz = calibration.source.source_frequency_min_hz
    source_upper_hz = calibration.source.source_frequency_max_hz
    if not all(
        source_lower_hz <= frequency_hz <= source_upper_hz
        for frequency_hz in frequencies_hz
    ):
        _geometry_error(
            "source_domain_violation",
            "query frequency leaves source domain",
            proposed_frequency_min_hz=proposed_lower_hz,
            proposed_frequency_max_hz=proposed_upper_hz,
        )

    return _SparseFitGeometry(
        resonance_id=seeded_identity.resonance_id,
        offset_multipliers=offset_multipliers,
        frequencies_hz=frequencies_hz,
        frozen_fast_center_hz=q0_hz,
        frozen_prior_fwhm_hz=w0_hz,
        frozen_source_amplitude=a0,
        calibration_cell_lower_hz=cell_lower_hz,
        calibration_cell_upper_hz=cell_upper_hz,
        source_frequency_min_hz=source_lower_hz,
        source_frequency_max_hz=source_upper_hz,
        lower_bounds_scaled=lower_bounds,
        upper_bounds_scaled=upper_bounds,
        initial_guess_scaled=initial_guess,
    )


def _validate_calibration_sparse_geometry(
    calibration: TwoPointCalibration, configuration: SparseLinewidthConfiguration
) -> None:
    """Prospectively reject calibration seeds that cannot form a sparse fit."""
    if type(calibration) is not TwoPointCalibration:
        _geometry_error("empty_fit_bounds", "calibration must be exact")
    if type(configuration) is not SparseLinewidthConfiguration:
        _geometry_error("empty_fit_bounds", "configuration must be exact")
    for scan_index, seeded_identity in enumerate(calibration.identities):
        _construct_geometry(
            calibration,
            configuration,
            seeded_identity,
            q0_hz=seeded_identity.calibration_center_hz,
            w0_hz=seeded_identity.calibration_fwhm_hz,
            scan_index=scan_index,
            identity_scan_index=0,
        )


def _construct_sparse_fit_geometry(
    calibration: TwoPointCalibration,
    configuration: SparseLinewidthConfiguration,
    identity: CompositeIdentityEstimate,
    *,
    scan_index: int,
    identity_scan_index: int,
) -> _SparseFitGeometry:
    """Freeze the exact local fit geometry without constructing a query clock."""
    if type(calibration) is not TwoPointCalibration:
        _geometry_error("empty_fit_bounds", "calibration must be exact")
    if type(configuration) is not SparseLinewidthConfiguration:
        _geometry_error("empty_fit_bounds", "configuration must be exact")
    if type(identity) is not CompositeIdentityEstimate:
        _geometry_error("empty_fit_bounds", "identity must be exact")
    scan_index = _nonnegative_index(scan_index, "scan_index")
    identity_scan_index = _nonnegative_index(identity_scan_index, "identity_scan_index")
    seeded_identity = _identity_calibration(calibration, identity.resonance_id)
    return _construct_geometry(
        calibration,
        configuration,
        seeded_identity,
        q0_hz=identity.fast_center_hz,
        w0_hz=identity.active_fwhm_hz,
        scan_index=scan_index,
        identity_scan_index=identity_scan_index,
    )


def _public_reference_timestamp_s(
    observations: tuple[EstimatorObservation, ...],
) -> float:
    reference_s = observations[0].timestamp_s - observations[0].integration_time_s / 2.0
    for count, observation in enumerate(observations[1:], start=2):
        midpoint_s = observation.timestamp_s - observation.integration_time_s / 2.0
        reference_s = reference_s + (midpoint_s - reference_s) / count
    return reference_s


def _derive_fitted_q(fitted_local_center_hz: float, fitted_fwhm_hz: float) -> float:
    return float(fitted_local_center_hz / fitted_fwhm_hz)


def _finish_sparse_fit(
    *,
    started_ns: int,
    queries: tuple[SparseLinewidthQuery, ...],
    observations: tuple[EstimatorObservation, ...],
    public_reference_timestamp_s: float,
    status: str,
    failure_code: str | None,
    fitted_center_correction_hz: float | None = None,
    fitted_local_center_hz: float | None = None,
    fitted_fwhm_hz: float | None = None,
    fitted_amplitude: float | None = None,
    fitted_baseline_offset: float | None = None,
    fitted_q: float | None = None,
    rmse: float | None = None,
    amplitude_normalized_rmse: float | None = None,
    scaled_jacobian_rank: int | None = None,
    scaled_jacobian_condition: float | None = None,
    scipy_status: int | None = None,
    scipy_message: str | None = None,
    nfev: int | None = None,
) -> SparseLinewidthScanResult:
    first_query = queries[0]
    finished_ns = time.process_time_ns()
    fit_cpu_time_s = (finished_ns - started_ns) / 1_000_000_000.0
    return SparseLinewidthScanResult(
        scan_index=first_query.scan_index,
        identity_scan_index=first_query.identity_scan_index,
        resonance_id=first_query.resonance_id,
        frozen_fast_center_hz=first_query.frozen_fast_center_hz,
        frozen_fast_center_source_kind=first_query.frozen_fast_center_source_kind,
        frozen_fast_center_source_pair_index=(
            first_query.frozen_fast_center_source_pair_index
        ),
        frozen_fast_center_reference_timestamp_s=(
            first_query.frozen_fast_center_reference_timestamp_s
        ),
        frozen_fast_center_release_sequence_index=(
            first_query.frozen_fast_center_release_sequence_index
        ),
        frozen_fast_center_release_timestamp_s=(
            first_query.frozen_fast_center_release_timestamp_s
        ),
        frozen_prior_fwhm_hz=first_query.frozen_prior_fwhm_hz,
        frozen_fwhm_source_kind=first_query.frozen_fwhm_source_kind,
        frozen_fwhm_source_scan_index=first_query.frozen_fwhm_source_scan_index,
        frozen_fwhm_reference_timestamp_s=(
            first_query.frozen_fwhm_reference_timestamp_s
        ),
        frozen_fwhm_release_sequence_index=(
            first_query.frozen_fwhm_release_sequence_index
        ),
        frozen_fwhm_release_timestamp_s=(
            first_query.frozen_fwhm_release_timestamp_s
        ),
        queries=queries,
        observations=observations,
        public_reference_timestamp_s=public_reference_timestamp_s,
        release_sequence_index=observations[-1].sequence_index,
        release_timestamp_s=observations[-1].timestamp_s,
        status=status,
        failure_code=failure_code,
        fitted_center_correction_hz=fitted_center_correction_hz,
        fitted_local_center_hz=fitted_local_center_hz,
        fitted_fwhm_hz=fitted_fwhm_hz,
        fitted_amplitude=fitted_amplitude,
        fitted_baseline_offset=fitted_baseline_offset,
        fitted_q=fitted_q,
        rmse=rmse,
        amplitude_normalized_rmse=amplitude_normalized_rmse,
        scaled_jacobian_rank=scaled_jacobian_rank,
        scaled_jacobian_condition=scaled_jacobian_condition,
        scipy_status=scipy_status,
        scipy_message=scipy_message,
        nfev=nfev,
        fit_cpu_time_s=fit_cpu_time_s,
    )


def _canonical_solver_diagnostics(
    optimization: object,
) -> tuple[int, str, int]:
    raw_status = optimization.status  # type: ignore[attr-defined]
    raw_message = optimization.message  # type: ignore[attr-defined]
    raw_nfev = optimization.nfev  # type: ignore[attr-defined]
    if isinstance(raw_status, (bool, np.bool_)) or not isinstance(
        raw_status, (Integral, np.integer)
    ):
        raise TypeError("solver status must be an integer")
    if not isinstance(raw_message, str):
        raise TypeError("solver message must be a string")
    if isinstance(raw_nfev, (bool, np.bool_)) or not isinstance(
        raw_nfev, (Integral, np.integer)
    ):
        raise TypeError("solver nfev must be an integer")
    status = int(raw_status)
    message = str(raw_message)
    nfev = int(raw_nfev)
    if not message.strip():
        raise ValueError("solver message must be nonblank")
    if nfev <= 0:
        raise ValueError("solver nfev must be positive")
    return status, message, nfev


def fit_sparse_linewidth(
    source: TwoPointCalibrationSource,
    configuration: SparseLinewidthConfiguration,
    queries: Sequence[SparseLinewidthQuery],
    observations: Sequence[EstimatorObservation],
) -> SparseLinewidthScanResult:
    """Fit one completed sparse scan's four-parameter local source model."""
    started_ns = time.process_time_ns()
    frozen_queries = tuple(queries)
    frozen_observations = tuple(observations)
    first_query = frozen_queries[0]
    q0_hz = first_query.frozen_fast_center_hz
    w0_hz = first_query.frozen_prior_fwhm_hz
    target = next(
        resonance
        for resonance in source.source_fit.resonance_estimates
        if resonance.resonance_id == first_query.resonance_id
    )
    a0 = target.amplitude
    frequency_hz = np.asarray(
        [query.frequency_hz for query in frozen_queries], dtype=np.float64
    )
    observed = np.asarray(
        [observation.fluorescence for observation in frozen_observations],
        dtype=np.float64,
    )
    lower_bounds = np.asarray(
        (
            -configuration.center_correction_limit_fwhm_fraction,
            max(
                configuration.min_fwhm_prior_ratio,
                source.fit_configuration.min_fwhm_hz / w0_hz,
            ),
            0.0,
            -configuration.baseline_offset_source_amplitude_fraction,
        ),
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        (
            configuration.center_correction_limit_fwhm_fraction,
            min(
                configuration.max_fwhm_prior_ratio,
                source.fit_configuration.max_fwhm_hz / w0_hz,
            ),
            min(
                configuration.max_amplitude_source_ratio,
                source.fit_configuration.max_amplitude / a0,
            ),
            configuration.baseline_offset_source_amplitude_fraction,
        ),
        dtype=np.float64,
    )
    initial_guess = np.asarray((0.0, 1.0, 1.0, 0.0), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        bound_spans = upper_bounds - lower_bounds
    public_reference_timestamp_s = _public_reference_timestamp_s(frozen_observations)

    def scaled_prediction(packed: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            dc_hz = packed[0] * w0_hz
            fwhm_hz = packed[1] * w0_hz
            amplitude = packed[2] * a0
            baseline_offset = packed[3] * a0
            center_hz = q0_hz + dc_hz
        return np.asarray(
            _evaluate_bound_source_model(
                frequency_hz,
                source,
                first_query.resonance_id,
                center_hz=center_hz,
                fwhm_hz=fwhm_hz,
                amplitude=amplitude,
                baseline_offset=baseline_offset,
            ),
            dtype=np.float64,
        )

    def scaled_residual(packed: np.ndarray) -> np.ndarray:
        model = scaled_prediction(packed)
        return np.asarray(model - observed, dtype=np.float64)

    preparation_is_finite = (
        lower_bounds.shape == (4,)
        and upper_bounds.shape == (4,)
        and bound_spans.shape == (4,)
        and initial_guess.shape == (4,)
        and frequency_hz.shape == (5,)
        and observed.shape == (5,)
        and np.all(np.isfinite(lower_bounds))
        and np.all(np.isfinite(upper_bounds))
        and np.all(np.isfinite(bound_spans))
        and np.all(bound_spans > 0.0)
        and np.all(np.isfinite(initial_guess))
        and np.all(np.isfinite(frequency_hz))
        and np.all(np.isfinite(observed))
        and np.all(lower_bounds < initial_guess)
        and np.all(initial_guess < upper_bounds)
    )
    if not preparation_is_finite:
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="model_evaluation_failed",
        )
    initial_prediction = scaled_prediction(initial_guess)
    initial_residual = np.asarray(initial_prediction - observed, dtype=np.float64)
    if not (
        initial_prediction.shape == (5,)
        and initial_residual.shape == (5,)
        and np.all(np.isfinite(initial_prediction))
        and np.all(np.isfinite(initial_residual))
    ):
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="model_evaluation_failed",
        )

    optimization = least_squares(
        scaled_residual,
        initial_guess,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        max_nfev=configuration.max_nfev,
    )
    scipy_status, scipy_message, nfev = _canonical_solver_diagnostics(optimization)
    solver_diagnostics = {
        "scipy_status": scipy_status,
        "scipy_message": scipy_message,
        "nfev": nfev,
    }
    if scipy_status <= 0 or nfev >= configuration.max_nfev:
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="optimizer_failed",
            **solver_diagnostics,
        )

    fitted_scaled = np.asarray(optimization.x, dtype=np.float64)
    residual = np.asarray(optimization.fun, dtype=np.float64)
    scaled_jacobian = np.asarray(optimization.jac, dtype=np.float64)
    cost = float(optimization.cost)
    if not (
        fitted_scaled.shape == (4,)
        and residual.shape == (5,)
        and scaled_jacobian.shape == (5, 4)
    ):
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="nonfinite_solution",
            **solver_diagnostics,
        )
    if not (
        np.all(np.isfinite(fitted_scaled))
        and np.all(np.isfinite(residual))
        and math.isfinite(cost)
    ):
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="nonfinite_solution",
            **solver_diagnostics,
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        fitted_center_correction_hz = float(fitted_scaled[0] * w0_hz)
        fitted_local_center_hz = float(q0_hz + fitted_center_correction_hz)
        fitted_fwhm_hz = float(fitted_scaled[1] * w0_hz)
        fitted_amplitude = float(fitted_scaled[2] * a0)
        fitted_baseline_offset = float(fitted_scaled[3] * a0)
    public_parameters = (
        fitted_center_correction_hz,
        fitted_local_center_hz,
        fitted_fwhm_hz,
        fitted_amplitude,
        fitted_baseline_offset,
    )
    if not all(math.isfinite(value) for value in public_parameters):
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="nonfinite_solution",
            **solver_diagnostics,
        )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        final_prediction = scaled_prediction(fitted_scaled)
        rmse = float(np.sqrt(np.sum(residual**2) / 5.0))
        amplitude_normalized_rmse = float(rmse / fitted_amplitude)
    if not (
        final_prediction.shape == (5,)
        and np.all(np.isfinite(final_prediction))
        and fitted_fwhm_hz > 0.0
        and fitted_amplitude >= 0.0
        and math.isfinite(rmse)
        and rmse >= 0.0
        and math.isfinite(amplitude_normalized_rmse)
        and amplitude_normalized_rmse >= 0.0
    ):
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="nonfinite_solution",
            **solver_diagnostics,
        )
    singular_values = np.asarray(
        np.linalg.svd(scaled_jacobian, compute_uv=False), dtype=np.float64
    )
    if not (
        singular_values.shape == (4,) and np.all(np.isfinite(singular_values))
    ):
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="nonfinite_solution",
            **solver_diagnostics,
        )

    fitted_diagnostics = {
        "fitted_center_correction_hz": fitted_center_correction_hz,
        "fitted_local_center_hz": fitted_local_center_hz,
        "fitted_fwhm_hz": fitted_fwhm_hz,
        "fitted_amplitude": fitted_amplitude,
        "fitted_baseline_offset": fitted_baseline_offset,
        "rmse": rmse,
        "amplitude_normalized_rmse": amplitude_normalized_rmse,
    }
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lower_margins = (fitted_scaled - lower_bounds) / bound_spans
        upper_margins = (upper_bounds - fitted_scaled) / bound_spans
    if np.any(
        (fitted_scaled < lower_bounds)
        | (fitted_scaled > upper_bounds)
        | (lower_margins < configuration.min_interior_bound_fraction)
        | (upper_margins < configuration.min_interior_bound_fraction)
    ):
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="bounds_active",
            **fitted_diagnostics,
            **solver_diagnostics,
        )

    cutoff = singular_values[0] * configuration.rank_rtol
    scaled_jacobian_rank = int(np.count_nonzero(singular_values > cutoff))
    if scaled_jacobian_rank != 4:
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="rank_deficient",
            scaled_jacobian_rank=scaled_jacobian_rank,
            **fitted_diagnostics,
            **solver_diagnostics,
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled_jacobian_condition = float(singular_values[0] / singular_values[-1])
    if not math.isfinite(scaled_jacobian_condition):
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="nonfinite_solution",
            **solver_diagnostics,
        )
    late_diagnostics = {
        "scaled_jacobian_rank": scaled_jacobian_rank,
        "scaled_jacobian_condition": scaled_jacobian_condition,
        **fitted_diagnostics,
        **solver_diagnostics,
    }
    if scaled_jacobian_condition > configuration.max_scaled_jacobian_condition:
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="ill_conditioned",
            **late_diagnostics,
        )

    resolved_amplitude = max(
        configuration.min_resolved_amplitude_source_ratio * a0,
        source.fit_configuration.min_resolved_amplitude,
    )
    if fitted_amplitude < resolved_amplitude:
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="amplitude_unresolved",
            **late_diagnostics,
        )
    if amplitude_normalized_rmse > configuration.max_amplitude_normalized_rmse:
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="residual_quality_failed",
            **late_diagnostics,
        )

    fitted_q = _derive_fitted_q(fitted_local_center_hz, fitted_fwhm_hz)
    if not math.isfinite(fitted_q):
        return _finish_sparse_fit(
            started_ns=started_ns,
            queries=frozen_queries,
            observations=frozen_observations,
            public_reference_timestamp_s=public_reference_timestamp_s,
            status="failure",
            failure_code="nonfinite_solution",
            **solver_diagnostics,
        )
    return _finish_sparse_fit(
        started_ns=started_ns,
        queries=frozen_queries,
        observations=frozen_observations,
        public_reference_timestamp_s=public_reference_timestamp_s,
        status="success",
        failure_code=None,
        fitted_q=fitted_q,
        **late_diagnostics,
    )
