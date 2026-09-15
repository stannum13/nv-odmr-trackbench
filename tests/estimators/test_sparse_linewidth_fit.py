"""Pure sparse five-point fit-geometry contract tests."""

from __future__ import annotations

import inspect
import math
import sys
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators import (
    SparseLinewidthConfiguration,
    SparseLinewidthResetError,
    TwoPointCalibration,
)
from odmr_bench.estimators.sparse_linewidth_fit import (
    _construct_sparse_fit_geometry,
    _SparseFitGeometry,
    _validate_calibration_sparse_geometry,
    fit_sparse_linewidth,
)
from odmr_bench.estimators.two_point_calibration import _evaluate_bound_source_model
from odmr_bench.models import Baseline
from tests.sparse_linewidth_helpers import make_composite_identity, make_sparse_query
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_fit_configuration,
    make_legal_identity_calibrations,
    make_legal_source_fit,
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


def _fit_source(*, model_kind: str, quadratic: bool):
    configuration = replace(
        make_legal_fit_configuration(),
        model_kind=model_kind,
        baseline_degree=2 if quadratic else 1,
    )
    fit = make_legal_source_fit(configuration)
    resonances = tuple(
        replace(resonance, eta=1.0)
        if model_kind == "lorentzian"
        else resonance
        for resonance in fit.resonance_estimates
    )
    baseline = Baseline(
        intercept=1.03,
        reference_hz=2.88e9,
        slope_per_hz=2.1e-11,
        quadratic_per_hz2=1.7e-20 if quadratic else 0.0,
    )
    fit = replace(
        fit,
        model_kind=model_kind,
        baseline_degree=2 if quadratic else 1,
        resonance_estimates=resonances,
        baseline_estimate=baseline,
        initial_guess=replace(
            fit.initial_guess,
            resonances=resonances,
            baseline=baseline,
        ),
        jacobian_rank=(3 if quadratic else 2)
        + 8 * (3 if model_kind == "lorentzian" else 4),
    )
    return make_legal_caller_asserted_source(
        source_fit=fit,
        fit_configuration=configuration,
    )


def _fit_inputs(
    source: object,
    *,
    scan_index: int = 0,
    q0_hz: float | None = None,
    dc_hz: float = 120_000.0,
    fwhm_hz: float = 1.65e6,
    amplitude: float = 0.018,
    baseline_offset: float = 0.002,
):
    target = source.source_fit.resonance_estimates[0]
    q0_hz = target.center_hz if q0_hz is None else q0_hz
    w0_hz = target.fwhm_hz
    multipliers = (
        (0.5, -1.0, 0.0, 1.0, -0.5)
        if (scan_index // 8) % 2 == 0
        else (-0.5, 1.0, 0.0, -1.0, 0.5)
    )
    queries = tuple(
        make_sparse_query(
            acquisition_index=10 + point_index,
            scan_index=scan_index,
            identity_scan_index=scan_index // 8,
            point_index=point_index,
            resonance_id=target.resonance_id,
            offset_multiplier=multiplier,
            frozen_fast_center_hz=q0_hz,
            frozen_prior_fwhm_hz=w0_hz,
            frequency_hz=q0_hz + multiplier * w0_hz,
            expected_sequence_index=20 + point_index,
            expected_end_timestamp_s=0.105 + 0.005 * point_index,
            expected_nominal_exposure_photons=12_500.0,
        )
        for point_index, multiplier in enumerate(multipliers)
    )
    fluorescence = _evaluate_bound_source_model(
        np.asarray([query.frequency_hz for query in queries], dtype=np.float64),
        source,
        target.resonance_id,
        center_hz=q0_hz + dc_hz,
        fwhm_hz=fwhm_hz,
        amplitude=amplitude,
        baseline_offset=baseline_offset,
    )
    observations = tuple(
        EstimatorObservation(
            sequence_index=query.expected_sequence_index,
            timestamp_s=query.expected_end_timestamp_s,
            frequency_hz=query.frequency_hz,
            fluorescence=float(value),
            integration_time_s=query.integration_time_s,
            nominal_exposure_photons=query.expected_nominal_exposure_photons,
        )
        for query, value in zip(queries, fluorescence, strict=True)
    )
    return queries, observations


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


def test_geometry_module_has_no_stateful_query_clock_construction() -> None:
    import odmr_bench.estimators.sparse_linewidth_fit as sparse_fit

    module_source = inspect.getsource(sparse_fit)
    assert "TwoPointRunMetadata" not in module_source


@pytest.mark.parametrize(
    ("model_kind", "quadratic"),
    (("lorentzian", False), ("pseudo_voigt", True)),
)
def test_noiseless_success_recovers_exact_four_parameter_local_model(
    model_kind: str, quadratic: bool
) -> None:
    source = _fit_source(model_kind=model_kind, quadratic=quadratic)
    configuration = SparseLinewidthConfiguration()
    expected = (120_000.0, 1.65e6, 0.018, 0.002)
    queries, observations = _fit_inputs(
        source,
        dc_hz=expected[0],
        fwhm_hz=expected[1],
        amplitude=expected[2],
        baseline_offset=expected[3],
    )

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert result.status == "success"
    assert result.failure_code is None
    assert result.fitted_center_correction_hz == pytest.approx(expected[0], rel=1.0e-6)
    assert result.fitted_local_center_hz == (
        queries[0].frozen_fast_center_hz + result.fitted_center_correction_hz
    )
    assert result.fitted_local_center_hz == pytest.approx(
        queries[0].frozen_fast_center_hz + expected[0], abs=0.1
    )
    assert result.fitted_fwhm_hz == pytest.approx(expected[1], rel=1.0e-6)
    assert result.fitted_amplitude == pytest.approx(expected[2], rel=1.0e-6)
    assert result.fitted_baseline_offset == pytest.approx(expected[3], rel=2.0e-6)
    assert result.fitted_q == result.fitted_local_center_hz / result.fitted_fwhm_hz
    assert result.scaled_jacobian_rank == 4
    assert result.scaled_jacobian_condition is not None
    assert result.rmse == pytest.approx(0.0, abs=1.0e-8)
    assert result.amplitude_normalized_rmse == pytest.approx(0.0, abs=1.0e-6)


def test_success_uses_one_scaled_solver_call_and_preserves_arrival_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fit_source(model_kind="pseudo_voigt", quadratic=True)
    configuration = SparseLinewidthConfiguration(max_nfev=123)
    expected = (90_000.0, 1.62e6, 0.017, -0.0015)
    queries, observations = _fit_inputs(
        source,
        scan_index=8,
        q0_hz=-2.76e9,
        dc_hz=expected[0],
        fwhm_hz=expected[1],
        amplitude=expected[2],
        baseline_offset=expected[3],
    )
    query_ids = tuple(id(query) for query in queries)
    observation_ids = tuple(id(observation) for observation in observations)
    source_resonances = source.source_fit.resonance_estimates
    calls: list[tuple[object, np.ndarray, dict[str, object]]] = []

    def observing_least_squares(
        residual: object, x0: object, **kwargs: object
    ) -> SimpleNamespace:
        packed = np.asarray(x0, dtype=np.float64)
        calls.append((residual, packed.copy(), kwargs.copy()))
        w0 = queries[0].frozen_prior_fwhm_hz
        a0 = source.source_fit.resonance_estimates[0].amplitude
        probe = np.asarray((0.07, 1.08, 0.91, -0.04), dtype=np.float64)
        expected_model = _evaluate_bound_source_model(
            np.asarray([query.frequency_hz for query in queries], dtype=np.float64),
            source,
            queries[0].resonance_id,
            center_hz=queries[0].frozen_fast_center_hz + probe[0] * w0,
            fwhm_hz=probe[1] * w0,
            amplitude=probe[2] * a0,
            baseline_offset=probe[3] * a0,
        )
        expected_residual = expected_model - np.asarray(
            [observation.fluorescence for observation in observations],
            dtype=np.float64,
        )
        assert np.array_equal(residual(probe), expected_residual)
        solution = np.asarray(
            (expected[0] / w0, expected[1] / w0, expected[2] / a0, expected[3] / a0),
            dtype=np.float64,
        )
        fun = np.asarray(residual(solution), dtype=np.float64)
        return SimpleNamespace(
            x=solution,
            fun=fun,
            jac=np.vstack((np.eye(4), np.ones(4))),
            cost=float(np.dot(fun, fun) / 2.0),
            status=1,
            message="converged",
            nfev=7,
        )

    monkeypatch.setattr(
        "odmr_bench.estimators.sparse_linewidth_fit.least_squares",
        observing_least_squares,
    )

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert len(calls) == 1
    _, initial, kwargs = calls[0]
    target = source.source_fit.resonance_estimates[0]
    w0 = queries[0].frozen_prior_fwhm_hz
    a0 = target.amplitude
    assert np.array_equal(initial, np.asarray((0.0, 1.0, 1.0, 0.0)))
    lower, upper = kwargs.pop("bounds")
    assert np.array_equal(
        np.asarray(lower),
        np.asarray(
            (
                -0.5,
                max(0.5, source.fit_configuration.min_fwhm_hz / w0),
                0.0,
                -1.0,
            )
        ),
    )
    assert np.array_equal(
        np.asarray(upper),
        np.asarray(
            (
                0.5,
                min(2.0, source.fit_configuration.max_fwhm_hz / w0),
                min(4.0, source.fit_configuration.max_amplitude / a0),
                1.0,
            )
        ),
    )
    assert kwargs == {"method": "trf", "max_nfev": 123}
    assert result.queries == queries
    assert result.observations == observations
    assert tuple(id(query) for query in queries) == query_ids
    assert tuple(id(observation) for observation in observations) == observation_ids
    assert source.source_fit.resonance_estimates == source_resonances
    assert result.release_sequence_index == observations[-1].sequence_index
    assert result.release_timestamp_s == observations[-1].timestamp_s
    assert result.fitted_center_correction_hz == expected[0]
    assert result.fitted_local_center_hz == (
        queries[0].frozen_fast_center_hz + expected[0]
    )
    assert result.fitted_fwhm_hz == expected[1]
    assert result.fitted_amplitude == expected[2]
    assert result.fitted_baseline_offset == expected[3]
    assert result.fitted_q == result.fitted_local_center_hz / expected[1]
    assert result.fitted_q < 0.0
