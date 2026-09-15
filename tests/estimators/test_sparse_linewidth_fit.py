"""Pure sparse five-point fit-geometry contract tests."""

from __future__ import annotations

import inspect
import math
import sys
from dataclasses import fields, replace

import pytest

from odmr_bench.estimators import (
    SparseLinewidthConfiguration,
    SparseLinewidthResetError,
    TwoPointCalibration,
)
from odmr_bench.estimators.sparse_linewidth_fit import (
    _construct_sparse_fit_geometry,
    _SparseFitGeometry,
    _validate_calibration_sparse_geometry,
)
from tests.sparse_linewidth_helpers import make_composite_identity
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_identity_calibrations,
    make_legal_tracker_configuration,
)


def _calibration() -> TwoPointCalibration:
    source = make_legal_caller_asserted_source()
    return TwoPointCalibration(
        source=source,
        configuration=make_legal_tracker_configuration(),
        budget_treatment="conditional_free_precalibration",
        identities=make_legal_identity_calibrations(source),
    )


def _seeded_identity(calibration: TwoPointCalibration, index: int = 0):
    seeded = calibration.identities[index]
    return make_composite_identity(
        resonance_id=seeded.resonance_id,
        fast_center_hz=seeded.calibration_center_hz,
        active_fwhm_hz=seeded.calibration_fwhm_hz,
        live_q=seeded.calibration_center_hz / seeded.calibration_fwhm_hz,
    )


@pytest.mark.parametrize(
    ("parity", "expected"),
    (
        (0, (0.5, -1.0, 0.0, 1.0, -0.5)),
        (1, (-0.5, 1.0, 0.0, -1.0, 0.5)),
    ),
)
def test_sparse_geometry_uses_exact_time_symmetric_order(
    parity: int, expected: tuple[float, ...]
) -> None:
    calibration = _calibration()
    geometry = _construct_sparse_fit_geometry(
        calibration,
        SparseLinewidthConfiguration(),
        _seeded_identity(calibration),
        scan_index=parity,
        identity_scan_index=parity,
    )
    assert geometry.offset_multipliers == expected


def test_calibration_validation_constructs_every_seeded_identity() -> None:
    calibration = _calibration()
    configuration = SparseLinewidthConfiguration()

    _validate_calibration_sparse_geometry(calibration, configuration)

    for index, seeded in enumerate(calibration.identities):
        geometry = _construct_sparse_fit_geometry(
            calibration,
            configuration,
            _seeded_identity(calibration, index),
            scan_index=index,
            identity_scan_index=0,
        )
        assert geometry.resonance_id == seeded.resonance_id
        assert geometry.frozen_fast_center_hz == seeded.calibration_center_hz
        assert geometry.frozen_prior_fwhm_hz == seeded.calibration_fwhm_hz
        assert geometry.frozen_source_amplitude == seeded.calibration_amplitude


def test_calibration_validation_rejects_late_identity_sparse_geometry() -> None:
    calibration = _calibration()
    late_seed = calibration.identities[7]
    narrow_late_cell = replace(
        late_seed,
        calibration_cell_lower_hz=late_seed.calibration_center_hz - 600_000.0,
        calibration_cell_upper_hz=late_seed.calibration_center_hz + 600_000.0,
    )
    calibration = replace(
        calibration,
        identities=(*calibration.identities[:7], narrow_late_cell),
    )

    with pytest.raises(SparseLinewidthResetError) as raised:
        _validate_calibration_sparse_geometry(
            calibration, SparseLinewidthConfiguration()
        )

    assert raised.value.code == "invalid_base_sparse_geometry"
    assert raised.value.geometry_failure_code == "calibration_cell_violation"
    assert (
        raised.value.proposed_frequency_min_hz < raised.value.proposed_frequency_max_hz
    )


def test_geometry_keeps_the_exact_five_frequency_design_and_time_balance() -> None:
    calibration = _calibration()
    identity = _seeded_identity(calibration)
    geometry = _construct_sparse_fit_geometry(
        calibration,
        SparseLinewidthConfiguration(),
        identity,
        scan_index=9,
        identity_scan_index=0,
    )

    assert set(geometry.offset_multipliers) == {-1.0, -0.5, 0.0, 0.5, 1.0}
    assert geometry.frequencies_hz == tuple(
        identity.fast_center_hz + offset * identity.active_fwhm_hz
        for offset in geometry.offset_multipliers
    )
    assert (
        sum(
            time_index * offset
            for time_index, offset in zip(
                (-2, -1, 0, 1, 2), geometry.offset_multipliers, strict=True
            )
        )
        == 0.0
    )


def test_geometry_uses_normative_intersected_scaled_bounds_and_interior_guess() -> None:
    calibration = _calibration()
    seeded = calibration.identities[0]
    source_configuration = calibration.source.fit_configuration
    geometry = _construct_sparse_fit_geometry(
        calibration,
        SparseLinewidthConfiguration(),
        _seeded_identity(calibration),
        scan_index=0,
        identity_scan_index=0,
    )

    assert geometry.lower_bounds_scaled == pytest.approx(
        (
            -0.5,
            max(0.5, source_configuration.min_fwhm_hz / seeded.calibration_fwhm_hz),
            0.0,
            -1.0,
        )
    )
    assert geometry.upper_bounds_scaled == pytest.approx(
        (
            0.5,
            min(2.0, source_configuration.max_fwhm_hz / seeded.calibration_fwhm_hz),
            min(4.0, source_configuration.max_amplitude / seeded.calibration_amplitude),
            1.0,
        )
    )
    assert geometry.initial_guess_scaled == (0.0, 1.0, 1.0, 0.0)
    assert all(
        lower < initial < upper
        for lower, initial, upper in zip(
            geometry.lower_bounds_scaled,
            geometry.initial_guess_scaled,
            geometry.upper_bounds_scaled,
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "configuration",
    (
        SparseLinewidthConfiguration(center_correction_limit_fwhm_fraction=0.5),
        SparseLinewidthConfiguration(
            min_fwhm_prior_ratio=0.75,
            max_fwhm_prior_ratio=1.25,
            max_amplitude_source_ratio=1.5,
            baseline_offset_source_amplitude_fraction=0.75,
        ),
    ),
)
def test_every_geometry_bound_and_initial_guess_is_strictly_interior(
    configuration: SparseLinewidthConfiguration,
) -> None:
    calibration = _calibration()
    geometry = _construct_sparse_fit_geometry(
        calibration,
        configuration,
        _seeded_identity(calibration),
        scan_index=0,
        identity_scan_index=0,
    )

    assert all(
        lower < initial < upper
        for lower, initial, upper in zip(
            geometry.lower_bounds_scaled,
            geometry.initial_guess_scaled,
            geometry.upper_bounds_scaled,
            strict=True,
        )
    )


def test_nonrepresentable_lower_precedes_every_other_geometry_failure() -> None:
    calibration = _calibration()
    identity = replace(
        _seeded_identity(calibration),
        fast_center_hz=-sys.float_info.max,
        active_fwhm_hz=sys.float_info.max,
        live_q=-1.0,
    )

    with pytest.raises(SparseLinewidthResetError) as raised:
        _construct_sparse_fit_geometry(
            calibration,
            SparseLinewidthConfiguration(),
            identity,
            scan_index=0,
            identity_scan_index=0,
        )

    assert raised.value.code == "invalid_base_sparse_geometry"
    assert "nonrepresentable_frequency_lower" in str(raised.value)
    assert raised.value.geometry_failure_code == "nonrepresentable_frequency_lower"
    assert raised.value.proposed_frequency_min_hz is None
    assert raised.value.proposed_frequency_max_hz is None


def test_nonrepresentable_upper_follows_representable_lower() -> None:
    calibration = _calibration()
    identity = replace(
        _seeded_identity(calibration),
        fast_center_hz=sys.float_info.max,
        active_fwhm_hz=sys.float_info.max,
        live_q=1.0,
    )

    with pytest.raises(SparseLinewidthResetError) as raised:
        _construct_sparse_fit_geometry(
            calibration,
            SparseLinewidthConfiguration(),
            identity,
            scan_index=0,
            identity_scan_index=0,
        )

    assert raised.value.code == "invalid_base_sparse_geometry"
    assert "nonrepresentable_frequency_upper" in str(raised.value)
    assert raised.value.geometry_failure_code == "nonrepresentable_frequency_upper"
    assert raised.value.proposed_frequency_min_hz is None
    assert raised.value.proposed_frequency_max_hz is None


def test_empty_bounds_precede_cell_and_source_domain_checks() -> None:
    calibration = _calibration()
    identity = replace(
        _seeded_identity(calibration), active_fwhm_hz=20.0e6, live_q=138.0
    )

    with pytest.raises(SparseLinewidthResetError) as raised:
        _construct_sparse_fit_geometry(
            calibration,
            SparseLinewidthConfiguration(),
            identity,
            scan_index=0,
            identity_scan_index=0,
        )

    assert raised.value.code == "invalid_base_sparse_geometry"
    assert "empty_fit_bounds" in str(raised.value)
    assert raised.value.geometry_failure_code == "empty_fit_bounds"
    assert (
        raised.value.proposed_frequency_min_hz < raised.value.proposed_frequency_max_hz
    )


def test_calibration_cell_violation_precedes_source_domain_violation() -> None:
    calibration = _calibration()
    seeded = calibration.identities[0]
    identity = replace(
        _seeded_identity(calibration),
        fast_center_hz=seeded.calibration_cell_lower_hz,
        live_q=seeded.calibration_cell_lower_hz / seeded.calibration_fwhm_hz,
    )

    with pytest.raises(SparseLinewidthResetError) as raised:
        _construct_sparse_fit_geometry(
            calibration,
            SparseLinewidthConfiguration(),
            identity,
            scan_index=0,
            identity_scan_index=0,
        )

    assert raised.value.code == "invalid_base_sparse_geometry"
    assert "calibration_cell_violation" in str(raised.value)
    assert raised.value.geometry_failure_code == "calibration_cell_violation"
    assert (
        raised.value.proposed_frequency_min_hz < raised.value.proposed_frequency_max_hz
    )


def test_closed_cell_boundary_passes_and_one_ulp_outward_fails() -> None:
    calibration = _calibration()
    seeded = calibration.identities[0]
    width = seeded.calibration_fwhm_hz
    boundary_center = seeded.calibration_cell_lower_hz + width
    boundary_identity = replace(
        _seeded_identity(calibration),
        fast_center_hz=boundary_center,
        live_q=boundary_center / width,
    )

    geometry = _construct_sparse_fit_geometry(
        calibration,
        SparseLinewidthConfiguration(),
        boundary_identity,
        scan_index=0,
        identity_scan_index=0,
    )
    assert geometry.frequencies_hz[1] == seeded.calibration_cell_lower_hz

    outward_identity = replace(
        boundary_identity,
        fast_center_hz=math.nextafter(boundary_center, -math.inf),
        live_q=math.nextafter(boundary_center, -math.inf) / width,
    )
    with pytest.raises(SparseLinewidthResetError) as raised:
        _construct_sparse_fit_geometry(
            calibration,
            SparseLinewidthConfiguration(),
            outward_identity,
            scan_index=0,
            identity_scan_index=0,
        )
    assert raised.value.geometry_failure_code == "calibration_cell_violation"


def test_source_domain_violation_follows_a_valid_calibration_cell() -> None:
    calibration = _calibration()
    widened = replace(
        calibration.identities[0],
        calibration_cell_lower_hz=2.70e9,
        calibration_cell_upper_hz=2.80e9,
    )
    calibration = replace(
        calibration, identities=(widened, *calibration.identities[1:])
    )
    identity = replace(
        _seeded_identity(calibration),
        fast_center_hz=2.741e9,
        live_q=2.741e9 / calibration.identities[0].calibration_fwhm_hz,
    )

    with pytest.raises(SparseLinewidthResetError) as raised:
        _construct_sparse_fit_geometry(
            calibration,
            SparseLinewidthConfiguration(),
            identity,
            scan_index=0,
            identity_scan_index=0,
        )

    assert raised.value.code == "invalid_base_sparse_geometry"
    assert "source_domain_violation" in str(raised.value)
    assert raised.value.geometry_failure_code == "source_domain_violation"
    assert (
        raised.value.proposed_frequency_min_hz < raised.value.proposed_frequency_max_hz
    )


def test_closed_source_boundary_passes_and_one_ulp_outward_fails() -> None:
    calibration = _calibration()
    widened = replace(
        calibration.identities[0],
        calibration_cell_lower_hz=2.70e9,
        calibration_cell_upper_hz=2.80e9,
    )
    calibration = replace(
        calibration, identities=(widened, *calibration.identities[1:])
    )
    width = widened.calibration_fwhm_hz
    boundary_center = calibration.source.source_frequency_min_hz + width
    boundary_identity = replace(
        _seeded_identity(calibration),
        fast_center_hz=boundary_center,
        live_q=boundary_center / width,
    )

    geometry = _construct_sparse_fit_geometry(
        calibration,
        SparseLinewidthConfiguration(),
        boundary_identity,
        scan_index=0,
        identity_scan_index=0,
    )
    assert geometry.frequencies_hz[1] == calibration.source.source_frequency_min_hz

    outward_center = math.nextafter(boundary_center, -math.inf)
    outward_identity = replace(
        boundary_identity,
        fast_center_hz=outward_center,
        live_q=outward_center / width,
    )
    with pytest.raises(SparseLinewidthResetError) as raised:
        _construct_sparse_fit_geometry(
            calibration,
            SparseLinewidthConfiguration(),
            outward_identity,
            scan_index=0,
            identity_scan_index=0,
        )
    assert raised.value.geometry_failure_code == "source_domain_violation"


def test_initial_guess_at_a_closed_bound_is_an_empty_fit_geometry() -> None:
    calibration = _calibration()

    with pytest.raises(SparseLinewidthResetError) as raised:
        _construct_sparse_fit_geometry(
            calibration,
            SparseLinewidthConfiguration(max_amplitude_source_ratio=1.0),
            _seeded_identity(calibration),
            scan_index=0,
            identity_scan_index=0,
        )

    assert "empty_fit_bounds" in str(raised.value)


def test_geometry_is_frozen_and_contains_no_query_clock_or_resource_metadata() -> None:
    field_names = tuple(field.name for field in fields(_SparseFitGeometry))
    assert field_names == (
        "resonance_id",
        "offset_multipliers",
        "frequencies_hz",
        "frozen_fast_center_hz",
        "frozen_prior_fwhm_hz",
        "frozen_source_amplitude",
        "calibration_cell_lower_hz",
        "calibration_cell_upper_hz",
        "source_frequency_min_hz",
        "source_frequency_max_hz",
        "lower_bounds_scaled",
        "upper_bounds_scaled",
        "initial_guess_scaled",
    )
    assert _SparseFitGeometry.__slots__ == field_names
    forbidden = ("acquisition", "sequence", "endpoint", "exposure", "clock", "resource")
    assert all(
        all(term not in field_name for term in forbidden) for field_name in field_names
    )


def test_geometry_module_has_no_scipy_or_stateful_query_clock_construction() -> None:
    import odmr_bench.estimators.sparse_linewidth_fit as sparse_fit

    module_source = inspect.getsource(sparse_fit)
    assert "scipy" not in module_source
    assert "SparseLinewidthQuery" not in module_source
    assert "TwoPointRunMetadata" not in module_source
