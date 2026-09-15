"""Pure sparse five-point fit-geometry contract tests."""

from __future__ import annotations

import inspect
import math
import sys
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

import odmr_bench.estimators.sparse_linewidth_fit as sparse_fit
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


def _solver_result(
    *,
    x: object = (0.0, 1.0, 1.0, 0.0),
    fun: object = (0.0, 0.0, 0.0, 0.0, 0.0),
    jac: object | None = None,
    cost: object = 0.0,
    status: object = 1,
    message: object = "controlled convergence",
    nfev: object = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray(x, dtype=np.float64),
        fun=np.asarray(fun, dtype=np.float64),
        jac=(
            np.vstack((np.eye(4, dtype=np.float64), np.zeros(4)))
            if jac is None
            else np.asarray(jac, dtype=np.float64)
        ),
        cost=cost,
        status=status,
        message=message,
        nfev=nfev,
    )


def _controlled_fit_inputs(*, q0_hz: float | None = None):
    source = _fit_source(model_kind="pseudo_voigt", quadratic=True)
    queries, observations = _fit_inputs(source, q0_hz=q0_hz)
    return source, SparseLinewidthConfiguration(), queries, observations


def _patch_solver(
    monkeypatch: pytest.MonkeyPatch, optimization: SimpleNamespace
) -> None:
    monkeypatch.setattr(
        sparse_fit, "least_squares", lambda *args, **kwargs: optimization
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


@pytest.mark.parametrize(
    ("scenario", "expected_code", "expected_presence"),
    (
        ("model", "model_evaluation_failed", (False, False, False, False, False)),
        ("optimizer", "optimizer_failed", (True, False, False, False, False)),
        ("nonfinite", "nonfinite_solution", (True, False, False, False, False)),
        ("bounds", "bounds_active", (True, True, True, False, False)),
        ("rank", "rank_deficient", (True, True, True, True, False)),
        ("condition", "ill_conditioned", (True, True, True, True, True)),
        ("amplitude", "amplitude_unresolved", (True, True, True, True, True)),
        ("rmse", "residual_quality_failed", (True, True, True, True, True)),
        ("success", None, (True, True, True, True, True)),
    ),
)
def test_first_applicable_gate_and_exact_diagnostic_presence_matrix(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_code: str | None,
    expected_presence: tuple[bool, bool, bool, bool, bool],
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    solution = np.asarray((0.0, 1.0, 1.0, 0.0), dtype=np.float64)
    residual = np.zeros(5, dtype=np.float64)
    status = 1
    singular_values = np.ones(4, dtype=np.float64)

    if scenario == "model":
        monkeypatch.setattr(
            sparse_fit,
            "_evaluate_bound_source_model",
            lambda *args, **kwargs: np.full(5, np.nan),
        )
        monkeypatch.setattr(
            sparse_fit,
            "least_squares",
            lambda *args, **kwargs: pytest.fail("solver called after initial failure"),
        )
    else:
        if scenario == "optimizer":
            status = 0
            solution[0] = np.nan
            residual[:] = np.inf
        elif scenario == "nonfinite":
            solution[0] = np.nan
            residual[:] = np.inf
        elif scenario == "bounds":
            solution[:] = (0.5, 1.0, 0.20, 0.0)
            residual[:] = math.nextafter(0.002, math.inf)
            singular_values[-1] = 0.0
        elif scenario == "rank":
            solution[2] = 0.20
            residual[:] = math.nextafter(0.002, math.inf)
            singular_values[-1] = 0.0
        elif scenario == "condition":
            solution[2] = 0.20
            residual[:] = math.nextafter(0.002, math.inf)
            singular_values[-1] = math.nextafter(1.0e-8, 0.0)
        elif scenario == "amplitude":
            solution[2] = 0.20
            residual[:] = math.nextafter(0.002, math.inf)
        elif scenario == "rmse":
            residual[:] = math.nextafter(0.002, math.inf)
        _patch_solver(
            monkeypatch,
            _solver_result(x=solution, fun=residual, status=status),
        )
    svd_calls = 0

    def controlled_svd(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal svd_calls
        svd_calls += 1
        return singular_values.copy()

    monkeypatch.setattr(sparse_fit.np.linalg, "svd", controlled_svd)
    clock_values = iter((100, 400))
    monkeypatch.setattr(
        sparse_fit.time, "process_time_ns", lambda: next(clock_values)
    )

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert result.status == ("success" if expected_code is None else "failure")
    assert result.failure_code == expected_code
    fitted = (
        result.fitted_center_correction_hz,
        result.fitted_local_center_hz,
        result.fitted_fwhm_hz,
        result.fitted_amplitude,
        result.fitted_baseline_offset,
    )
    actual_presence = (
        all(
            value is not None
            for value in (result.scipy_status, result.scipy_message, result.nfev)
        ),
        all(value is not None for value in fitted),
        result.rmse is not None and result.amplitude_normalized_rmse is not None,
        result.scaled_jacobian_rank is not None,
        result.scaled_jacobian_condition is not None,
    )
    assert actual_presence == expected_presence
    assert (
        result.fitted_q is not None
        if expected_code is None
        else result.fitted_q is None
    )
    assert result.fit_cpu_time_s == 300 / 1_000_000_000.0
    assert math.isfinite(result.fit_cpu_time_s) and result.fit_cpu_time_s >= 0.0
    assert svd_calls == (0 if scenario in {"model", "optimizer", "nonfinite"} else 1)


def test_nonrepresentable_preparation_fails_before_model_or_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, configuration, ordinary_queries, ordinary_observations = (
        _controlled_fit_inputs()
    )
    frozen_width = 5.0e-324
    queries = tuple(
        replace(
            query,
            frozen_prior_fwhm_hz=frozen_width,
            frequency_hz=query.frozen_fast_center_hz,
        )
        for query in ordinary_queries
    )
    observations = tuple(
        replace(observation, frequency_hz=query.frequency_hz)
        for query, observation in zip(queries, ordinary_observations, strict=True)
    )
    monkeypatch.setattr(
        sparse_fit,
        "_evaluate_bound_source_model",
        lambda *args, **kwargs: pytest.fail("model called after invalid preparation"),
    )
    monkeypatch.setattr(
        sparse_fit,
        "least_squares",
        lambda *args, **kwargs: pytest.fail("solver called after invalid preparation"),
    )

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert (result.status, result.failure_code) == (
        "failure",
        "model_evaluation_failed",
    )
    assert result.scipy_status is None


@pytest.mark.parametrize("status", (0, -1))
def test_nonpositive_solver_status_is_optimizer_failure(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    configuration = replace(configuration, max_nfev=17)
    _patch_solver(
        monkeypatch,
        _solver_result(
            x=(np.nan, 1.0, 1.0, 0.0),
            fun=np.full(5, np.inf),
            status=status,
            nfev=1,
        ),
    )

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert (result.status, result.failure_code) == ("failure", "optimizer_failed")
    assert result.scipy_status == status
    assert result.nfev == 1


@pytest.mark.parametrize(
    ("nfev", "expected_code"), ((17, "optimizer_failed"), (16, None))
)
def test_optimizer_failure_uses_nondefault_configured_max_nfev(
    monkeypatch: pytest.MonkeyPatch, nfev: int, expected_code: str | None
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    configuration = replace(configuration, max_nfev=17)
    _patch_solver(monkeypatch, _solver_result(status=1, nfev=nfev))

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert result.failure_code == expected_code
    assert result.scipy_status == 1
    assert result.nfev == nfev


@pytest.mark.parametrize(
    "malformation",
    (
        "wrong_x_shape",
        "wrong_residual_shape",
        "wrong_jacobian_shape",
        "nonfinite_parameter",
        "nonfinite_public_parameter",
        "nonfinite_prediction",
        "nonfinite_residual",
        "nonfinite_cost",
        "nonfinite_rmse",
        "nonfinite_singular_value",
    ),
)
def test_malformed_or_nonfinite_returned_solution_is_nonfinite_solution(
    monkeypatch: pytest.MonkeyPatch, malformation: str
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    x: object = (0.0, 1.0, 1.0, 0.0)
    fun: object = np.zeros(5, dtype=np.float64)
    jac: object = np.vstack((np.eye(4), np.zeros(4)))
    cost: object = 0.0
    singular_values = np.ones(4, dtype=np.float64)
    if malformation == "wrong_x_shape":
        x = (0.0, 1.0, 1.0)
    elif malformation == "wrong_residual_shape":
        fun = np.zeros(4, dtype=np.float64)
    elif malformation == "wrong_jacobian_shape":
        jac = np.eye(4, dtype=np.float64)
    elif malformation == "nonfinite_parameter":
        x = (np.nan, 1.0, 1.0, 0.0)
    elif malformation == "nonfinite_public_parameter":
        x = (sys.float_info.max, 1.0, 1.0, 0.0)
    elif malformation == "nonfinite_residual":
        fun = (np.inf, 0.0, 0.0, 0.0, 0.0)
    elif malformation == "nonfinite_cost":
        cost = np.nan
    elif malformation == "nonfinite_rmse":
        fun = (sys.float_info.max, 0.0, 0.0, 0.0, 0.0)
    elif malformation == "nonfinite_singular_value":
        singular_values[-1] = np.nan
    if malformation == "nonfinite_prediction":
        original_model = sparse_fit._evaluate_bound_source_model
        model_calls = 0

        def nonfinite_final_model(*args: object, **kwargs: object) -> np.ndarray:
            nonlocal model_calls
            model_calls += 1
            if model_calls == 1:
                return original_model(*args, **kwargs)
            return np.full(5, np.inf)

        monkeypatch.setattr(
            sparse_fit, "_evaluate_bound_source_model", nonfinite_final_model
        )
    _patch_solver(monkeypatch, _solver_result(x=x, fun=fun, jac=jac, cost=cost))
    monkeypatch.setattr(
        sparse_fit.np.linalg,
        "svd",
        lambda *args, **kwargs: singular_values.copy(),
    )

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert (result.status, result.failure_code) == ("failure", "nonfinite_solution")
    assert result.scipy_status == 1
    assert result.fitted_center_correction_hz is None
    assert result.rmse is None
    assert result.scaled_jacobian_rank is None


@pytest.mark.parametrize(
    ("direction", "expected_code"),
    (("equality", "amplitude_unresolved"), ("inward_ulp", "bounds_active")),
)
def test_scaled_bound_margin_equality_passes_and_inward_ulp_fails(
    monkeypatch: pytest.MonkeyPatch, direction: str, expected_code: str
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    amplitude_scaled = 4.0 * configuration.min_interior_bound_fraction
    assert amplitude_scaled / 4.0 == configuration.min_interior_bound_fraction
    if direction == "inward_ulp":
        amplitude_scaled = math.nextafter(amplitude_scaled, 0.0)
    _patch_solver(monkeypatch, _solver_result(x=(0.0, 1.0, amplitude_scaled, 0.0)))

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert result.failure_code == expected_code


@pytest.mark.parametrize(
    ("direction", "expected_code", "expected_rank"),
    (("equality", "rank_deficient", 3), ("outward_ulp", None, 4)),
)
def test_rank_cutoff_equality_is_discarded_and_outward_ulp_is_full_rank(
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    expected_code: str | None,
    expected_rank: int,
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    configuration = replace(configuration, max_scaled_jacobian_condition=1.0e12)
    cutoff = 1.0 * configuration.rank_rtol
    smallest = cutoff
    if direction == "outward_ulp":
        smallest = math.nextafter(cutoff, math.inf)
    calls = 0

    def controlled_svd(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.asarray((1.0, 1.0, 1.0, smallest), dtype=np.float64)

    _patch_solver(monkeypatch, _solver_result())
    monkeypatch.setattr(sparse_fit.np.linalg, "svd", controlled_svd)

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert result.failure_code == expected_code
    assert result.scaled_jacobian_rank == expected_rank
    assert calls == 1


@pytest.mark.parametrize(
    ("direction", "expected_code"),
    (("equality", None), ("outward_ulp", "ill_conditioned")),
)
def test_condition_limit_equality_passes_and_outward_ulp_fails(
    monkeypatch: pytest.MonkeyPatch, direction: str, expected_code: str | None
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    smallest = 1.0e-8
    if direction == "outward_ulp":
        smallest = math.nextafter(smallest, 0.0)
    _patch_solver(monkeypatch, _solver_result())
    monkeypatch.setattr(
        sparse_fit.np.linalg,
        "svd",
        lambda *args, **kwargs: np.asarray((1.0, 1.0, 1.0, smallest)),
    )

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert result.failure_code == expected_code
    assert result.scaled_jacobian_condition == 1.0 / smallest
    if direction == "equality":
        assert result.scaled_jacobian_condition == 1.0e8


@pytest.mark.parametrize(
    ("direction", "expected_code"),
    (("equality", None), ("outward_ulp", "amplitude_unresolved")),
)
def test_resolved_amplitude_equality_passes_and_outward_ulp_fails(
    monkeypatch: pytest.MonkeyPatch, direction: str, expected_code: str | None
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    amplitude_scaled = configuration.min_resolved_amplitude_source_ratio
    if direction == "outward_ulp":
        amplitude_scaled = math.nextafter(amplitude_scaled, 0.0)
    _patch_solver(monkeypatch, _solver_result(x=(0.0, 1.0, amplitude_scaled, 0.0)))

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    expected_threshold = max(
        configuration.min_resolved_amplitude_source_ratio
        * source.source_fit.resonance_estimates[0].amplitude,
        source.fit_configuration.min_resolved_amplitude,
    )
    assert result.failure_code == expected_code
    assert result.fitted_amplitude is not None
    if direction == "equality":
        assert result.fitted_amplitude == expected_threshold
    else:
        assert result.fitted_amplitude < expected_threshold


@pytest.mark.parametrize(
    ("direction", "expected_code"),
    (("equality", None), ("outward_ulp", "residual_quality_failed")),
)
def test_normalized_rmse_equality_passes_and_outward_ulp_fails(
    monkeypatch: pytest.MonkeyPatch, direction: str, expected_code: str | None
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    residual_value = 0.002
    if direction == "outward_ulp":
        residual_value = math.nextafter(residual_value, math.inf)
    _patch_solver(monkeypatch, _solver_result(fun=np.full(5, residual_value)))

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert result.failure_code == expected_code
    assert result.amplitude_normalized_rmse is not None
    if direction == "equality":
        assert result.amplitude_normalized_rmse == 0.10
    else:
        assert result.amplitude_normalized_rmse > 0.10


def test_rmse_uses_optimizer_residual_in_fixed_arrival_order_binary64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    residual = np.asarray(
        tuple(
            float.fromhex(value)
            for value in (
                "0x1.06a410420cd63p-85",
                "0x1.f5b0003225350p+83",
                "0x1.a602fe967b819p+94",
                "0x1.b5c95b46562e8p+105",
                "0x1.7ba191c8d1b75p+115",
            )
        ),
        dtype=np.float64,
    )
    expected = float(np.sqrt(np.sum(residual**2) / 5.0))
    regrouped = float(np.sqrt(np.sum((residual**2)[::-1]) / 5.0))
    assert expected != regrouped
    _patch_solver(monkeypatch, _solver_result(fun=residual))

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert result.failure_code == "residual_quality_failed"
    assert result.rmse == expected
    assert result.rmse != regrouped


@pytest.mark.parametrize(("q0_hz", "expected_sign"), ((-2.76e9, -1), (0.0, 0)))
def test_success_preserves_signed_and_zero_scan_q(
    monkeypatch: pytest.MonkeyPatch, q0_hz: float, expected_sign: int
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs(q0_hz=q0_hz)
    _patch_solver(monkeypatch, _solver_result())

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert result.status == "success"
    assert result.fitted_q == q0_hz / queries[0].frozen_prior_fwhm_hz
    assert (result.fitted_q > 0.0) - (result.fitted_q < 0.0) == expected_sign


def test_nonrepresentable_scan_q_is_nonfinite_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_configuration = replace(
        make_legal_fit_configuration(), min_fwhm_hz=5.0e-324, max_fwhm_hz=1.0e-299
    )
    source_fit = make_legal_source_fit(source_configuration)
    tiny_resonances = tuple(
        replace(resonance, center_hz=index * 1.0e6, fwhm_hz=1.0e-300)
        for index, resonance in enumerate(source_fit.resonance_estimates)
    )
    source_fit = replace(
        source_fit,
        resonance_estimates=tiny_resonances,
        initial_guess=replace(source_fit.initial_guess, resonances=tiny_resonances),
    )
    source = make_legal_caller_asserted_source(
        source_fit=source_fit, fit_configuration=source_configuration
    )
    q0_hz = sys.float_info.max
    multipliers = (0.5, -1.0, 0.0, 1.0, -0.5)
    queries = tuple(
        make_sparse_query(
            acquisition_index=10 + point_index,
            point_index=point_index,
            offset_multiplier=multiplier,
            resonance_id="r0",
            frozen_fast_center_hz=q0_hz,
            frozen_prior_fwhm_hz=1.0e-300,
            frequency_hz=q0_hz + multiplier * 1.0e-300,
            expected_sequence_index=20 + point_index,
            expected_end_timestamp_s=0.105 + 0.005 * point_index,
        )
        for point_index, multiplier in enumerate(multipliers)
    )
    observations = tuple(
        EstimatorObservation(
            sequence_index=query.expected_sequence_index,
            timestamp_s=query.expected_end_timestamp_s,
            frequency_hz=query.frequency_hz,
            fluorescence=1.0,
            integration_time_s=query.integration_time_s,
            nominal_exposure_photons=query.expected_nominal_exposure_photons,
        )
        for query in queries
    )
    monkeypatch.setattr(
        sparse_fit,
        "_evaluate_bound_source_model",
        lambda *args, **kwargs: np.zeros(5, dtype=np.float64),
    )
    _patch_solver(monkeypatch, _solver_result())

    result = fit_sparse_linewidth(
        source, SparseLinewidthConfiguration(), queries, observations
    )

    assert (result.status, result.failure_code) == ("failure", "nonfinite_solution")
    assert result.fitted_q is None
    assert result.fitted_fwhm_hz is None


class _FitControlFlow(BaseException):
    pass


@pytest.mark.parametrize(
    "seam", ("model", "solver", "svd", "q", "clock_start", "clock_end", "record")
)
@pytest.mark.parametrize("exception_type", (RuntimeError, _FitControlFlow))
def test_raised_collaborator_exceptions_escape_as_the_identical_object(
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
    exception_type: type[BaseException],
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    raised_exception = exception_type(f"raised by {seam}")

    def raise_exact(*args: object, **kwargs: object) -> object:
        raise raised_exception

    _patch_solver(monkeypatch, _solver_result())
    if seam == "model":
        monkeypatch.setattr(sparse_fit, "_evaluate_bound_source_model", raise_exact)
    elif seam == "solver":
        monkeypatch.setattr(sparse_fit, "least_squares", raise_exact)
    elif seam == "svd":
        monkeypatch.setattr(sparse_fit.np.linalg, "svd", raise_exact)
    elif seam == "q":
        monkeypatch.setattr(sparse_fit, "_derive_fitted_q", raise_exact)
    elif seam == "clock_start":
        monkeypatch.setattr(sparse_fit.time, "process_time_ns", raise_exact)
    elif seam == "clock_end":
        clock_calls = iter((100, raised_exception))

        def fail_second_clock() -> int:
            value = next(clock_calls)
            if isinstance(value, BaseException):
                raise value
            return value

        monkeypatch.setattr(sparse_fit.time, "process_time_ns", fail_second_clock)
    elif seam == "record":
        monkeypatch.setattr(sparse_fit, "SparseLinewidthScanResult", raise_exact)

    with pytest.raises(BaseException) as caught:
        fit_sparse_linewidth(source, configuration, queries, observations)

    assert caught.value is raised_exception


def test_fit_cpu_clock_stops_before_public_result_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, configuration, queries, observations = _controlled_fit_inputs()
    _patch_solver(monkeypatch, _solver_result())
    events: list[str] = []
    clock_values = iter((1_000, 501_000, 9_000_000_000))

    def controlled_clock() -> int:
        value = next(clock_values)
        events.append(f"clock:{value}")
        return value

    captured: dict[str, object] = {}

    def slow_result_constructor(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        events.append("constructor:start")
        controlled_clock()
        events.append("constructor:end")
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(sparse_fit.time, "process_time_ns", controlled_clock)
    monkeypatch.setattr(
        sparse_fit, "SparseLinewidthScanResult", slow_result_constructor
    )

    result = fit_sparse_linewidth(source, configuration, queries, observations)

    assert events == [
        "clock:1000",
        "clock:501000",
        "constructor:start",
        "clock:9000000000",
        "constructor:end",
    ]
    assert captured["fit_cpu_time_s"] == 500_000 / 1_000_000_000.0
    assert result.fit_cpu_time_s == captured["fit_cpu_time_s"]
