"""Tests for the dimensionless constrained fit parameterization."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from odmr_bench.estimators import FitConfiguration, FitInitialGuess
from odmr_bench.estimators.parameterization import (
    center_bounds_hz,
    pack_parameters,
    parameter_bounds,
    public_parameter_transform,
    unpack_parameters,
)
from odmr_bench.models import Baseline, Resonance


def _guess(model_kind: str = "pseudo_voigt", degree: int = 2) -> FitInitialGuess:
    centers = np.arange(8, dtype=np.float64) * 8.0 + 16.0
    return FitInitialGuess(
        resonances=tuple(
            Resonance(
                resonance_id=f"r{index}",
                center_hz=float(center),
                fwhm_hz=4.0,
                amplitude=float(index + 1),
                eta=1.0 if model_kind == "lorentzian" else (index + 1) / 8.0,
            )
            for index, center in enumerate(centers)
        ),
        baseline=Baseline(
            intercept=64.0,
            reference_hz=64.0,
            slope_per_hz=0.5,
            quadratic_per_hz2=0.25 if degree == 2 else 0.0,
        ),
    )


@pytest.mark.parametrize(
    ("model_kind", "degree", "size"),
    [
        ("lorentzian", 1, 26),
        ("lorentzian", 2, 27),
        ("pseudo_voigt", 1, 34),
        ("pseudo_voigt", 2, 35),
    ],
)
def test_pack_unpack_round_trip_is_exact_for_binary_values(
    model_kind: str, degree: int, size: int
) -> None:
    configuration = FitConfiguration(
        model_kind=model_kind,
        baseline_degree=degree,
        min_fwhm_hz=1.0,
        max_fwhm_hz=8.0,
        max_amplitude=16.0,
        min_resolved_amplitude=0.5,
        min_center_separation_hz=4.0,
    )
    guess = _guess(model_kind, degree)

    packed = pack_parameters(
        guess,
        configuration,
        frequency_reference_hz=64.0,
        frequency_half_span_hz=64.0,
        fluorescence_reference=32.0,
        fluorescence_scale=16.0,
    )
    unpacked = unpack_parameters(
        packed,
        configuration,
        frequency_reference_hz=64.0,
        frequency_half_span_hz=64.0,
        fluorescence_reference=32.0,
        fluorescence_scale=16.0,
    )

    assert packed.shape == (size,)
    assert unpacked == guess
    if model_kind == "lorentzian":
        assert all(item.eta == 1.0 for item in unpacked.resonances)
    else:
        assert_array_equal(packed[-8:], np.arange(1, 9) / 8.0)


def test_public_transform_has_declared_diagonal_factors() -> None:
    configuration = FitConfiguration(model_kind="pseudo_voigt", baseline_degree=2)
    transform = public_parameter_transform(
        configuration, frequency_half_span_hz=8.0, fluorescence_scale=4.0
    )
    expected = [4.0, 0.5, 0.0625]
    expected.extend([4.0, 8.0, 8.0] * 8)
    expected.extend([1.0] * 8)
    assert_array_equal(transform, np.diag(expected))


def test_quadratic_scaling_is_overflow_safe_at_extreme_binary_span() -> None:
    half_span = np.ldexp(1.0, 520)
    quadratic = np.ldexp(1.0, -1040)
    configuration = FitConfiguration(
        model_kind="lorentzian",
        baseline_degree=2,
        min_fwhm_hz=1.0,
        max_fwhm_hz=8.0,
        max_amplitude=16.0,
        min_resolved_amplitude=0.5,
        min_center_separation_hz=1.0,
    )
    guess = FitInitialGuess(
        resonances=_guess("lorentzian", 2).resonances,
        baseline=Baseline(0.0, 0.0, quadratic_per_hz2=quadratic),
    )

    packed = pack_parameters(
        guess,
        configuration,
        frequency_reference_hz=0.0,
        frequency_half_span_hz=half_span,
        fluorescence_reference=0.0,
        fluorescence_scale=1.0,
    )
    unpacked = unpack_parameters(
        packed,
        configuration,
        frequency_reference_hz=0.0,
        frequency_half_span_hz=half_span,
        fluorescence_reference=0.0,
        fluorescence_scale=1.0,
    )
    transform = public_parameter_transform(
        configuration,
        frequency_half_span_hz=half_span,
        fluorescence_scale=1.0,
    )

    assert packed[2] == 1.0
    assert unpacked.baseline.quadratic_per_hz2 == quadratic
    assert transform[2, 2] == quadratic


def test_public_transform_rejects_nonfinite_sequential_factors() -> None:
    configuration = FitConfiguration(model_kind="lorentzian", baseline_degree=2)

    with pytest.raises(ValueError, match="transform failed numerically"):
        public_parameter_transform(
            configuration,
            frequency_half_span_hz=np.nextafter(0.0, 1.0),
            fluorescence_scale=np.finfo(np.float64).max,
        )


def test_center_boxes_enforce_minimum_separation_including_exact_gaps() -> None:
    centers = np.array([10.0, 20.0, 35.0, 50.0, 65.0, 80.0, 95.0, 110.0])
    lower, upper = center_bounds_hz(centers, 0.0, 120.0, 10.0)

    assert lower[1] == centers[1]
    assert upper[0] == centers[0]
    assert np.all(lower[1:] - upper[:-1] == 10.0)


def test_center_boxes_reject_unrepresentable_requested_separation() -> None:
    centers = 1.0e16 + 100.0 * np.arange(8)

    with pytest.raises(ValueError, match="representable"):
        center_bounds_hz(centers, 1.0e16 - 100.0, 1.0e16 + 800.0, 5.0)


@pytest.mark.parametrize(
    ("centers", "minimum", "match"),
    [
        (np.arange(8) * 10.0, 12.0, "span"),
        ([10, 20, 30, 40, 50, 60, 70, 70], 10.0, "gap"),
        ([-1, 10, 20, 30, 40, 50, 60, 70], 5.0, "sweep"),
        ([0, 10, 20, 30, 40, 50, 60, 80], 10.0, "box"),
    ],
)
def test_infeasible_center_geometry_is_rejected(
    centers: object, minimum: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        center_bounds_hz(np.asarray(centers, dtype=np.float64), 0.0, 80.0, minimum)


def test_parameter_bounds_scale_all_public_limits() -> None:
    configuration = FitConfiguration(
        model_kind="pseudo_voigt",
        baseline_degree=1,
        min_fwhm_hz=2.0,
        max_fwhm_hz=8.0,
        max_amplitude=16.0,
        min_resolved_amplitude=1.0,
        min_center_separation_hz=4.0,
    )
    guess = _guess("pseudo_voigt", 1)
    lower, upper = parameter_bounds(
        guess,
        configuration,
        frequency_min_hz=0.0,
        frequency_max_hz=100.0,
        frequency_reference_hz=50.0,
        frequency_half_span_hz=50.0,
        fluorescence_scale=8.0,
    )

    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    for index in range(8):
        start = 2 + 3 * index
        assert_array_equal(lower[start : start + 3][[0, 2]], [0.0, 0.04])
        assert_array_equal(upper[start : start + 3][[0, 2]], [2.0, 0.16])
    assert_array_equal(lower[-8:], np.zeros(8))
    assert_array_equal(upper[-8:], np.ones(8))
