"""Pure geometry for one frozen sparse five-point linewidth fit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

from odmr_bench.estimators.sparse_linewidth_types import (
    CompositeIdentityEstimate,
    SparseLinewidthConfiguration,
    SparseLinewidthResetError,
)
from odmr_bench.estimators.two_point_types import (
    TwoPointCalibration,
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
