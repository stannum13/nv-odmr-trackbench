"""Tests for analytic calibrated two-point discriminator geometry."""

from __future__ import annotations

import inspect
import math
from dataclasses import fields, replace
from itertools import pairwise

import pytest

import odmr_bench.estimators.two_point_calibration as calibration_module
from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators import (
    TwoPointCalibrationConstructionError,
    TwoPointCalibrationSource,
    TwoPointIdentityBinding,
    TwoPointTrackerConfiguration,
    calibrate_two_point,
)
from odmr_bench.estimators.two_point_calibration import (
    _evaluate_target_only_model,
    _target_center_derivative,
)
from odmr_bench.models import Baseline, Resonance
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_source_fit,
    make_legal_tracker_configuration,
)


def _profile(frequency_hz: float, resonance: Resonance, center_hz: float) -> float:
    u = (frequency_hz - center_hz) / resonance.fwhm_hz
    return resonance.eta / (1.0 + 4.0 * u * u) + (
        1.0 - resonance.eta
    ) * math.exp(-4.0 * math.log(2.0) * u * u)


def _model_fixture() -> tuple[object, int]:
    fit = make_legal_source_fit()
    resonances = tuple(
        Resonance(
            resonance.resonance_id,
            resonance.center_hz + index * 125_000.0,
            resonance.fwhm_hz + index * 71_000.0,
            resonance.amplitude + index * 0.0013,
            0.08 + index * 0.11,
        )
        for index, resonance in enumerate(fit.resonance_estimates)
    )
    return (
        replace(
            fit,
            resonance_estimates=resonances,
            baseline_estimate=Baseline(
                intercept=1.07,
                reference_hz=2.88e9,
                slope_per_hz=2.3e-11,
            ),
        ),
        3,
    )


def test_target_only_model_and_center_derivative_are_canonical() -> None:
    source_fit, target_index = _model_fixture()
    target = source_fit.resonance_estimates[target_index]
    center_hz = target.center_hz + 0.07 * target.fwhm_hz
    frequency_hz = center_hz - 0.31 * target.fwhm_hz

    expected = float(source_fit.baseline_estimate.evaluate(frequency_hz))
    for index, resonance in enumerate(source_fit.resonance_estimates):
        evaluated_center_hz = (
            center_hz if index == target_index else resonance.center_hz
        )
        expected -= resonance.amplitude * _profile(
            frequency_hz, resonance, evaluated_center_hz
        )
    assert (
        _evaluate_target_only_model(
            source_fit, target_index, frequency_hz, center_hz
        )
        == expected
    )

    stored_center_model = _evaluate_target_only_model(
        source_fit, target_index, frequency_hz, target.center_hz
    )
    assert stored_center_model != expected

    offset_hz = 0.35 * target.fwhm_hz
    minus_frequency_hz = target.center_hz - offset_hz
    plus_frequency_hz = target.center_hz + offset_hz
    assert (
        _target_center_derivative(
            source_fit, target_index, minus_frequency_hz, target.center_hz
        )
        > 0.0
    )
    assert (
        _target_center_derivative(
            source_fit, target_index, plus_frequency_hz, target.center_hz
        )
        < 0.0
    )

    step_hz = 1e-5 * target.fwhm_hz
    numerical = (
        _evaluate_target_only_model(
            source_fit,
            target_index,
            frequency_hz,
            center_hz + step_hz,
        )
        - _evaluate_target_only_model(
            source_fit,
            target_index,
            frequency_hz,
            center_hz - step_hz,
        )
    ) / (2.0 * step_hz)
    analytic = _target_center_derivative(
        source_fit, target_index, frequency_hz, center_hz
    )
    assert analytic == pytest.approx(numerical, rel=1e-8, abs=1e-15)


def test_calibrate_two_point_public_signature_exists() -> None:
    source = make_legal_caller_asserted_source()
    configuration = make_legal_tracker_configuration()

    signature = inspect.signature(calibrate_two_point)
    assert tuple(signature.parameters) == (
        "source",
        "configuration",
        "budget_treatment",
    )
    assert (
        signature.parameters["budget_treatment"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    calibration = calibrate_two_point(
        source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    assert calibration.source is source


def _discriminator(
    source_fit: object,
    target_index: int,
    minus_frequency_hz: float,
    plus_frequency_hz: float,
    center_hz: float,
) -> float:
    minus = _evaluate_target_only_model(
        source_fit, target_index, minus_frequency_hz, center_hz
    )
    plus = _evaluate_target_only_model(
        source_fit, target_index, plus_frequency_hz, center_hz
    )
    return (minus - plus) / (minus + plus)


def _independent_model_value(source_fit: object, frequency_hz: float) -> float:
    value = float(source_fit.baseline_estimate.evaluate(frequency_hz))
    for resonance in source_fit.resonance_estimates:
        value -= resonance.amplitude * _profile(
            frequency_hz, resonance, resonance.center_hz
        )
    return value


def test_calibration_builds_analytic_slope_depth_and_all_fixed_cells() -> None:
    source_fit = make_legal_source_fit()
    unequal_fwhm_hz = (
        1.1e6,
        2.3e6,
        1.4e6,
        2.8e6,
        1.2e6,
        2.0e6,
        1.6e6,
        2.4e6,
    )
    source_fit = replace(
        source_fit,
        resonance_estimates=tuple(
            replace(resonance, fwhm_hz=unequal_fwhm_hz[index])
            for index, resonance in enumerate(source_fit.resonance_estimates)
        ),
    )
    source = make_legal_caller_asserted_source(source_fit=source_fit)
    configuration = make_legal_tracker_configuration()
    calibration = calibrate_two_point(
        source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )

    assert calibration.source is source
    assert calibration.configuration == configuration
    assert calibration.configuration is not configuration
    assert calibration.budget_treatment == "conditional_free_precalibration"
    centers = tuple(
        resonance.center_hz for resonance in source.source_fit.resonance_estimates
    )
    internal_boundaries = tuple(
        left + (right - left) / 2.0 for left, right in pairwise(centers)
    )
    expected_lowers = (source.source_frequency_min_hz, *internal_boundaries)
    expected_uppers = (*internal_boundaries, source.source_frequency_max_hz)

    for index, (identity, resonance) in enumerate(
        zip(calibration.identities, source.source_fit.resonance_estimates, strict=True)
    ):
        delta_hz = configuration.offset_fwhm_fraction * resonance.fwhm_hz
        capture_radius_hz = (
            configuration.capture_fwhm_fraction * resonance.fwhm_hz
        )
        max_step_hz = configuration.max_step_fwhm_fraction * resonance.fwhm_hz
        assert identity.source_fit_index == index
        assert identity.offset_hz == delta_hz
        assert identity.capture_radius_hz == capture_radius_hz
        assert identity.max_step_hz == max_step_hz
        assert identity.calibration_cell_lower_hz == expected_lowers[index]
        assert identity.calibration_cell_upper_hz == expected_uppers[index]
        inset_hz = delta_hz + capture_radius_hz
        assert identity.allowed_center_min_hz == expected_lowers[index] + inset_hz
        assert identity.allowed_center_max_hz == expected_uppers[index] - inset_hz
        assert (
            identity.allowed_center_min_hz
            <= identity.calibration_center_hz
            <= identity.allowed_center_max_hz
        )

        target_pair_depth = resonance.amplitude * (
            _profile(
                resonance.center_hz - delta_hz,
                resonance,
                resonance.center_hz,
            )
            + _profile(
                resonance.center_hz + delta_hz,
                resonance,
                resonance.center_hz,
            )
        )
        assert identity.target_pair_depth == target_pair_depth

        minus_frequency_hz = resonance.center_hz - delta_hz
        plus_frequency_hz = resonance.center_hz + delta_hz
        mu_minus = _evaluate_target_only_model(
            source.source_fit,
            index,
            minus_frequency_hz,
            resonance.center_hz,
        )
        mu_plus = _evaluate_target_only_model(
            source.source_fit,
            index,
            plus_frequency_hz,
            resonance.center_hz,
        )
        zero_discriminator = (mu_minus - mu_plus) / (mu_minus + mu_plus)
        expected_mu_minus = _independent_model_value(
            source.source_fit, minus_frequency_hz
        )
        expected_mu_plus = _independent_model_value(
            source.source_fit, plus_frequency_hz
        )
        expected_zero_discriminator = (
            expected_mu_minus - expected_mu_plus
        ) / (expected_mu_minus + expected_mu_plus)
        assert zero_discriminator == expected_zero_discriminator
        assert zero_discriminator == pytest.approx(
            _discriminator(
                source.source_fit,
                index,
                minus_frequency_hz,
                plus_frequency_hz,
                resonance.center_hz,
            ),
            rel=0.0,
            abs=0.0,
        )
        g_minus = _target_center_derivative(
            source.source_fit,
            index,
            minus_frequency_hz,
            resonance.center_hz,
        )
        g_plus = _target_center_derivative(
            source.source_fit,
            index,
            plus_frequency_hz,
            resonance.center_hz,
        )
        slope_per_hz = 2.0 * (mu_plus * g_minus - mu_minus * g_plus) / (
            mu_minus + mu_plus
        ) ** 2
        assert slope_per_hz > 0.0

        grid = tuple(
            resonance.center_hz
            - capture_radius_hz
            + 2.0 * capture_radius_hz * grid_index / 1000.0
            for grid_index in range(1001)
        )
        discriminators = tuple(
            _discriminator(
                source.source_fit,
                index,
                minus_frequency_hz,
                plus_frequency_hz,
                center_hz,
            )
            for center_hz in grid
        )
        assert all(right > left for left, right in pairwise(discriminators))


def test_calibration_rejects_analytic_numerical_slope_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_legal_caller_asserted_source()
    configuration = make_legal_tracker_configuration()
    original_derivative = calibration_module._target_center_derivative
    first = source.source_fit.resonance_estimates[0]
    offset_hz = configuration.offset_fwhm_fraction * first.fwhm_hz
    original_minus = original_derivative(
        source.source_fit,
        0,
        first.center_hz - offset_hz,
        first.center_hz,
    )
    original_plus = original_derivative(
        source.source_fit,
        0,
        first.center_hz + offset_hz,
        first.center_hz,
    )

    def perturbed_derivative(
        source_fit: object,
        target_index: int,
        frequency_hz: float,
        center_hz: float,
    ) -> float:
        return 1.000_001 * original_derivative(
            source_fit, target_index, frequency_hz, center_hz
        )

    assert original_minus > 0.0
    assert original_plus < 0.0
    assert 1.000_001 * original_minus > 0.0
    assert 1.000_001 * original_plus < 0.0
    monkeypatch.setattr(
        calibration_module,
        "_target_center_derivative",
        perturbed_derivative,
    )

    _assert_calibration_code(
        source,
        configuration,
        "conditional_free_precalibration",
        "invalid_calibration_geometry",
    )


def _source_with_bounds(lower_hz: float, upper_hz: float) -> object:
    source = make_legal_caller_asserted_source()
    first, last = source.source_observations
    observations = (
        EstimatorObservation(
            first.sequence_index,
            first.timestamp_s,
            lower_hz,
            first.fluorescence,
            first.integration_time_s,
            first.nominal_exposure_photons,
            first.realized_photons,
        ),
        EstimatorObservation(
            last.sequence_index,
            last.timestamp_s,
            upper_hz,
            last.fluorescence,
            last.integration_time_s,
            last.nominal_exposure_photons,
            last.realized_photons,
        ),
    )
    return replace(
        source,
        source_observations=observations,
        source_frequency_min_hz=lower_hz,
        source_frequency_max_hz=upper_hz,
    )


def _assert_calibration_code(
    source: object,
    configuration: object,
    budget_treatment: object,
    expected_code: str,
) -> None:
    with pytest.raises(TwoPointCalibrationConstructionError) as raised:
        calibrate_two_point(
            source,
            configuration,
            budget_treatment=budget_treatment,
        )
    assert raised.value.code == expected_code
    assert raised.value.message


class _CalibrationSourceSubclass(TwoPointCalibrationSource):
    pass


class _TrackerConfigurationSubclass(TwoPointTrackerConfiguration):
    pass


class _BudgetTreatmentSubclass(str):
    pass


@pytest.mark.parametrize(
    "case",
    (
        "wrong_source",
        "source_subclass",
        "wrong_configuration",
        "configuration_subclass",
        "wrong_budget_treatment",
        "budget_treatment_subclass",
    ),
)
def test_calibration_rejects_wrong_and_subclass_argument_types_first(
    case: str,
) -> None:
    source: object = make_legal_caller_asserted_source()
    configuration: object = make_legal_tracker_configuration()
    budget_treatment: object = "not_a_treatment"
    invalid_geometry_configuration = replace(
        configuration,
        identity_binding=TwoPointIdentityBinding("adopt_fit_ids", None),
    )

    if case == "wrong_source":
        source = object()
        configuration = invalid_geometry_configuration
    elif case == "source_subclass":
        source = _CalibrationSourceSubclass(
            **{
                field.name: getattr(source, field.name)
                for field in fields(source)
            }
        )
        configuration = invalid_geometry_configuration
    elif case == "wrong_configuration":
        configuration = object()
    elif case == "configuration_subclass":
        configuration = _TrackerConfigurationSubclass(
            **{
                field.name: getattr(invalid_geometry_configuration, field.name)
                for field in fields(invalid_geometry_configuration)
            }
        )
    elif case == "wrong_budget_treatment":
        configuration = invalid_geometry_configuration
        budget_treatment = object()
    else:
        configuration = invalid_geometry_configuration
        budget_treatment = _BudgetTreatmentSubclass(
            "conditional_free_precalibration"
        )

    _assert_calibration_code(
        source,
        configuration,
        budget_treatment,
        "invalid_argument_type",
    )


def test_calibration_geometry_accepts_endpoints_and_rejects_empty_or_one_ulp_outward(
) -> None:
    configuration = make_legal_tracker_configuration()
    ordinary_source = make_legal_caller_asserted_source()
    first = ordinary_source.source_fit.resonance_estimates[0]
    last = ordinary_source.source_fit.resonance_estimates[-1]
    first_inset_hz = (
        configuration.offset_fwhm_fraction * first.fwhm_hz
        + configuration.capture_fwhm_fraction * first.fwhm_hz
    )
    last_inset_hz = (
        configuration.offset_fwhm_fraction * last.fwhm_hz
        + configuration.capture_fwhm_fraction * last.fwhm_hz
    )
    exact_lower_hz = first.center_hz - first_inset_hz
    exact_upper_hz = last.center_hz + last_inset_hz
    endpoint_source = _source_with_bounds(exact_lower_hz, exact_upper_hz)
    calibration = calibrate_two_point(
        endpoint_source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    assert calibration.identities[0].allowed_center_min_hz == first.center_hz
    assert calibration.identities[-1].allowed_center_max_hz == last.center_hz

    lower_outward_source = _source_with_bounds(
        math.nextafter(exact_lower_hz, math.inf), exact_upper_hz
    )
    upper_outward_source = _source_with_bounds(
        exact_lower_hz, math.nextafter(exact_upper_hz, -math.inf)
    )
    for source in (lower_outward_source, upper_outward_source):
        _assert_calibration_code(
            source,
            configuration,
            "conditional_free_precalibration",
            "invalid_calibration_geometry",
        )

    source = make_legal_caller_asserted_source()
    resonances = list(source.source_fit.resonance_estimates)
    middle_center_hz = resonances[3].center_hz
    resonances[2] = replace(resonances[2], center_hz=middle_center_hz - 500_000.0)
    resonances[4] = replace(resonances[4], center_hz=middle_center_hz + 500_000.0)
    empty_fit = replace(source.source_fit, resonance_estimates=tuple(resonances))
    empty_source = replace(source, source_fit=empty_fit)
    _assert_calibration_code(
        empty_source,
        configuration,
        "conditional_free_precalibration",
        "invalid_calibration_geometry",
    )


def test_calibration_budget_and_adjacent_construction_precedence() -> None:
    source = make_legal_caller_asserted_source()
    configuration = make_legal_tracker_configuration()
    with pytest.raises(TypeError, match="budget_treatment"):
        calibrate_two_point(source, configuration)

    _assert_calibration_code(
        source,
        configuration,
        "included_same_run",
        "invalid_budget_treatment",
    )
    verified_source = make_legal_caller_asserted_source()
    object.__setattr__(
        verified_source, "provenance", "verified_factory_acquisition"
    )
    assert (
        calibrate_two_point(
            verified_source,
            configuration,
            budget_treatment="conditional_free_precalibration",
        ).source
        is verified_source
    )

    invalid_geometry_configuration = replace(
        configuration,
        identity_binding=TwoPointIdentityBinding("adopt_fit_ids", None),
    )
    invalid_availability_source = make_legal_caller_asserted_source()
    object.__setattr__(
        invalid_availability_source,
        "availability_timestamp_s",
        math.nextafter(invalid_availability_source.availability_timestamp_s, math.inf),
    )
    _assert_calibration_code(
        invalid_availability_source,
        invalid_geometry_configuration,
        "conditional_free_precalibration",
        "invalid_availability_or_clock",
    )
    _assert_calibration_code(
        source,
        invalid_geometry_configuration,
        "not_a_treatment",
        "invalid_calibration_geometry",
    )
    _assert_calibration_code(
        source,
        configuration,
        "not_a_treatment",
        "invalid_budget_treatment",
    )
