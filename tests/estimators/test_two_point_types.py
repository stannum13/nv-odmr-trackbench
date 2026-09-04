"""Tests for calibrated two-point public primitive contracts."""

from dataclasses import replace

import numpy as np
import pytest

from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_fit_configuration,
    make_legal_identity_calibrations,
    make_legal_source_fit,
    make_legal_source_observations,
    make_legal_tracker_configuration,
)


def test_two_point_primitive_names_are_public() -> None:
    from odmr_bench.estimators import (
        CalibrationBudgetTreatment,
        CalibrationIdentityMode,
        CalibrationSourceProvenance,
        ClockMappingKind,
        NormalizedFluorescenceProvenance,
        PairSide,
        PublicAcquisitionResources,
        TwoPointBudgetCeiling,
        TwoPointCalibrationConstructionCode,
        TwoPointCalibrationConstructionError,
        TwoPointClockMapping,
        TwoPointFailureCode,
        TwoPointIdentityBinding,
        TwoPointLockState,
        TwoPointObservationValidationCode,
        TwoPointObservationValidationError,
        TwoPointRunMetadata,
        TwoPointStopReason,
        TwoPointTrackerConfiguration,
        TwoPointUpdateConstructionCode,
        TwoPointUpdateConstructionError,
    )

    assert CalibrationBudgetTreatment
    assert CalibrationIdentityMode
    assert CalibrationSourceProvenance
    assert ClockMappingKind
    assert PublicAcquisitionResources
    assert TwoPointBudgetCeiling
    assert TwoPointIdentityBinding
    assert NormalizedFluorescenceProvenance
    assert PairSide
    assert TwoPointClockMapping
    assert TwoPointTrackerConfiguration
    assert TwoPointRunMetadata
    assert TwoPointCalibrationConstructionCode
    assert TwoPointCalibrationConstructionError
    assert TwoPointFailureCode
    assert TwoPointLockState
    assert TwoPointObservationValidationCode
    assert TwoPointObservationValidationError
    assert TwoPointStopReason
    assert TwoPointUpdateConstructionCode
    assert TwoPointUpdateConstructionError


def test_calibration_record_names_are_public() -> None:
    from odmr_bench.estimators import (
        TwoPointCalibration,
        TwoPointCalibrationSource,
        TwoPointIdentityCalibration,
    )

    assert TwoPointCalibrationSource
    assert TwoPointIdentityCalibration
    assert TwoPointCalibration


def test_caller_asserted_source_snapshots_values_and_rejects_verified_direct_construction(  # noqa: E501
) -> None:
    mutable_observation_list = list(make_legal_source_observations())
    mutable_resonance_ids = [f"r{index}" for index in range(8)]
    fit_configuration = make_legal_fit_configuration(mutable_resonance_ids)
    source_fit = make_legal_source_fit(fit_configuration)
    source = make_legal_caller_asserted_source(
        source_observations=mutable_observation_list,
        source_fit=source_fit,
        fit_configuration=fit_configuration,
    )

    assert source.provenance == "caller_asserted"
    assert source.source_observations is not mutable_observation_list
    assert source.source_observations == tuple(mutable_observation_list)
    assert source.source_observations[0] is not mutable_observation_list[0]
    assert source.source_fit is not source_fit
    assert source.fit_configuration is not fit_configuration
    with pytest.raises(ValueError, match="verified"):
        replace(source, provenance="verified_factory_acquisition")

    original_frequency_hz = source.source_observations[0].frequency_hz
    original_q_value = source.source_fit.q_values[0]
    mutable_observation_list.clear()
    mutable_resonance_ids[-1] = "mutated"
    object.__setattr__(fit_configuration, "resonance_ids", ("mutated",) * 8)
    source_fit.q_values.setflags(write=True)
    source_fit.q_values[0] = 0.0
    assert source.source_observations[0].frequency_hz == original_frequency_hz
    assert source.fit_configuration.resonance_ids[-1] == "r7"
    assert source.source_fit.q_values[0] == original_q_value


def test_caller_asserted_source_deeply_snapshots_nested_fit_values() -> None:
    from odmr_bench.estimators import FitUncertainty

    uncertainty = FitUncertainty(
        baseline_standard_errors=np.array([0.01, 0.02]),
        center_hz=np.full(8, 10.0),
        fwhm_hz=np.full(8, 20.0),
        amplitude=np.full(8, 0.03),
        eta=np.full(8, 0.04),
    )
    source_fit = replace(
        make_legal_source_fit(),
        uncertainty=uncertainty,
        uncertainty_reason=None,
    )
    source = make_legal_caller_asserted_source(source_fit=source_fit)
    stored_fit = source.source_fit
    stored_diagnostics = stored_fit.diagnostics
    stored_uncertainty = stored_fit.uncertainty
    assert stored_diagnostics is not source_fit.diagnostics
    assert stored_uncertainty is not uncertainty
    assert stored_uncertainty is not None
    expected_arrays = {
        name: np.array(getattr(stored_uncertainty, name), copy=True)
        for name in (
            "baseline_standard_errors",
            "center_hz",
            "fwhm_hz",
            "amplitude",
            "eta",
        )
    }
    expected_q_values = np.array(stored_fit.q_values, copy=True)

    object.__setattr__(source_fit.diagnostics, "messages", ("mutated",))
    object.__setattr__(source_fit.diagnostics, "source", "none")
    object.__setattr__(source_fit, "resonance_estimates", ())
    object.__setattr__(source_fit, "baseline_estimate", None)
    object.__setattr__(source_fit, "initial_guess", None)
    object.__setattr__(source_fit, "nfev", 0)
    source_fit.q_values.setflags(write=True)
    source_fit.q_values[0] = 0.0
    object.__setattr__(uncertainty, "method", "mutated")
    for name in expected_arrays:
        original = getattr(uncertainty, name)
        assert original is not None
        original.setflags(write=True)
        original[...] = 0.0

    assert stored_diagnostics.source == "user"
    assert stored_diagnostics.messages == ()
    assert len(stored_fit.resonance_estimates) == 8
    assert stored_fit.baseline_estimate is not None
    assert stored_fit.initial_guess is not None
    assert stored_fit.nfev == 1
    assert np.array_equal(stored_fit.q_values, expected_q_values)
    assert not stored_fit.q_values.flags.writeable
    assert stored_uncertainty.method == "local_linearized_jacobian_covariance"
    for name, expected in expected_arrays.items():
        stored = getattr(stored_uncertainty, name)
        assert stored is not None
        assert np.array_equal(stored, expected)
        assert not stored.flags.writeable


def test_calibration_records_preserve_source_identity_and_snapshot_configuration(
) -> None:
    from odmr_bench.estimators import TwoPointCalibration

    source = make_legal_caller_asserted_source()
    mutable_configuration = make_legal_tracker_configuration()
    identities = make_legal_identity_calibrations(source)
    treatment = "conditional_free_precalibration"
    calibration = TwoPointCalibration(
        source, mutable_configuration, treatment, identities
    )

    assert calibration.source is source
    assert calibration.configuration == mutable_configuration
    assert calibration.configuration is not mutable_configuration
    assert calibration.identities == tuple(identities)
    assert calibration.identities is not identities

    object.__setattr__(
        mutable_configuration.identity_binding,
        "expected_resonance_ids",
        ("mutated",) * 8,
    )
    assert calibration.configuration.identity_binding.expected_resonance_ids == tuple(
        f"r{index}" for index in range(8)
    )
    with pytest.raises(ValueError, match="conditional_free"):
        TwoPointCalibration(
            source, make_legal_tracker_configuration(), "included_same_run", identities
        )
    with pytest.raises(ValueError, match="unique"):
        TwoPointCalibration(
            source,
            make_legal_tracker_configuration(),
            treatment,
            (*identities[:-1], replace(identities[-1], resonance_id="r0")),
        )
    with pytest.raises(ValueError, match="positive"):
        replace(identities[0], calibration_fwhm_hz=0.0)
    with pytest.raises(ValueError, match="nonempty"):
        replace(
            identities[0],
            allowed_center_min_hz=identities[0].allowed_center_max_hz + 1.0,
        )


def test_public_resources_and_budget_validate_intrinsic_domains() -> None:
    from odmr_bench.estimators import (
        PublicAcquisitionResources,
        TwoPointBudgetCeiling,
    )

    resources = PublicAcquisitionResources(1, 0.005, 2.5e6, 0, 1, 0.006)
    assert resources.observations == 1
    assert type(resources.integration_time_s) is float
    numpy_resources = PublicAcquisitionResources(
        np.int64(1),
        np.float64(0.005),
        np.float64(2.5e6),
        np.int64(0),
        np.int64(1),
        np.float64(0.006),
    )
    assert type(numpy_resources.observations) is int
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(resources, observations_without_realized_counts=2)
    with pytest.raises(ValueError, match="include integration"):
        replace(resources, virtual_elapsed_time_s=0.004)
    with pytest.raises(ValueError, match="at least one"):
        TwoPointBudgetCeiling(None, None, None, None)
    with pytest.raises(TypeError):
        replace(resources, observations=True)


@pytest.mark.parametrize("capture_fwhm_fraction", (0.35, 0.36))
def test_tracker_configuration_requires_capture_fraction_below_offset(
    capture_fwhm_fraction: float,
) -> None:
    from odmr_bench.estimators import TwoPointTrackerConfiguration

    with pytest.raises(ValueError, match="strictly less"):
        TwoPointTrackerConfiguration(capture_fwhm_fraction=capture_fwhm_fraction)


def test_string_tuple_contracts_snapshot_lists_and_reject_non_sequences() -> None:
    from odmr_bench.estimators import (
        NormalizedFluorescenceProvenance,
        TwoPointIdentityBinding,
    )

    expected_ids = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]
    canonical_ids = tuple(expected_ids)
    sampling_rules = ["poisson"]
    binding = TwoPointIdentityBinding("require_expected_ids", expected_ids)
    provenance = NormalizedFluorescenceProvenance(
        "normalized_fluorescence", "instrument_v1", 2.5e6, sampling_rules
    )
    expected_ids[-1] = "mutated"
    sampling_rules.append("mutated")
    assert binding.expected_resonance_ids == canonical_ids
    assert provenance.sampling_rules == ("poisson",)

    for value in (
        set(canonical_ids),
        dict.fromkeys(canonical_ids),
        (resonance_id for resonance_id in canonical_ids),
    ):
        with pytest.raises(TypeError, match="ordered sequence"):
            TwoPointIdentityBinding("require_expected_ids", value)
    for value in (
        {"poisson"},
        {"poisson": None},
        (rule for rule in ("poisson",)),
    ):
        with pytest.raises(TypeError, match="ordered sequence"):
            NormalizedFluorescenceProvenance(
                "normalized_fluorescence", "instrument_v1", 2.5e6, value
            )


@pytest.mark.parametrize(
    ("error_type", "codes"),
    [
        (
            "calibration",
            (
                "invalid_argument_type",
                "invalid_argument_value",
                "invalid_provenance_or_quantity",
                "invalid_source_trace",
                "source_resource_mismatch",
                "fit_input_mismatch",
                "source_fit_failed",
                "source_identity_mismatch",
                "invalid_source_epoch",
                "invalid_availability_or_clock",
                "invalid_calibration_geometry",
                "invalid_budget_treatment",
            ),
        ),
        (
            "observation",
            (
                "invalid_observation_type",
                "no_pending_query",
                "sequence_mismatch",
                "frequency_mismatch",
                "integration_time_mismatch",
                "endpoint_mismatch",
                "nominal_exposure_mismatch",
                "invalid_observation_value",
            ),
        ),
        (
            "update",
            (
                "partial_pair_construction_failed",
                "pair_result_construction_failed",
                "identity_estimate_construction_failed",
                "resource_construction_failed",
                "aggregate_estimate_construction_failed",
            ),
        ),
    ],
)
def test_identity_clock_configuration_and_errors_are_closed(
    error_type: str, codes: tuple[str, ...]
) -> None:
    from odmr_bench.estimators import (
        NormalizedFluorescenceProvenance,
        TwoPointCalibrationConstructionError,
        TwoPointClockMapping,
        TwoPointIdentityBinding,
        TwoPointObservationValidationError,
        TwoPointRunMetadata,
        TwoPointTrackerConfiguration,
        TwoPointUpdateConstructionError,
    )

    expected_ids = ("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7")
    required = TwoPointIdentityBinding("require_expected_ids", expected_ids)
    assert required.expected_resonance_ids == expected_ids
    assert TwoPointIdentityBinding("adopt_fit_ids", None).expected_resonance_ids is None
    with pytest.raises(ValueError):
        TwoPointIdentityBinding("require_expected_ids", expected_ids[:-1])
    with pytest.raises(ValueError):
        TwoPointIdentityBinding("require_expected_ids", (*expected_ids[:-1], "r0"))
    with pytest.raises(ValueError):
        TwoPointIdentityBinding("require_expected_ids", (*expected_ids[:-1], " "))
    with pytest.raises(ValueError):
        TwoPointIdentityBinding("adopt_fit_ids", expected_ids)
    with pytest.raises(ValueError):
        TwoPointIdentityBinding("unknown", None)

    assert TwoPointClockMapping("shared_clock", "clock", "clock", 1.0, 0.0)
    assert TwoPointClockMapping("unit_scale_offset", "source", "tracker", 1.0, 2.0)
    with pytest.raises(ValueError):
        TwoPointClockMapping("shared_clock", "source", "tracker", 1.0, 0.0)
    with pytest.raises(ValueError):
        TwoPointClockMapping("shared_clock", "clock", "clock", 1.0, 0.1)
    with pytest.raises(ValueError):
        TwoPointClockMapping("unit_scale_offset", "clock", "clock", 1.0, 0.0)
    with pytest.raises(ValueError):
        TwoPointClockMapping("shared_clock", "clock", "clock", 1.01, 0.0)
    with pytest.raises(ValueError):
        TwoPointClockMapping("shared_clock", " ", " ", 1.0, 0.0)

    assert NormalizedFluorescenceProvenance(
        "normalized_fluorescence", "instrument_v1", 2.5e6, ("poisson",)
    )
    with pytest.raises(ValueError):
        NormalizedFluorescenceProvenance("raw_counts", "instrument_v1", 2.5e6, ())
    with pytest.raises(ValueError):
        NormalizedFluorescenceProvenance(
            "normalized_fluorescence", "instrument_v1", 0.0, ()
        )

    configuration = TwoPointTrackerConfiguration()
    assert configuration.identity_binding == required
    assert configuration.offset_fwhm_fraction == 0.35
    assert configuration.capture_fwhm_fraction == 0.20
    assert configuration.proportional_gain == 1.0
    assert configuration.max_step_fwhm_fraction == 0.10
    assert configuration.integration_time_s == 0.005
    assert configuration.common_mode_limit_target_depths is None
    with pytest.raises(ValueError):
        TwoPointTrackerConfiguration(integration_time_s=0.0)
    assert TwoPointRunMetadata(
        "clock", None, 0.0, 2.5e6, 0.001, "normalized_fluorescence"
    )
    with pytest.raises(ValueError):
        TwoPointRunMetadata("clock", None, 0.0, 2.5e6, 0.001, "raw_counts")

    error_class = {
        "calibration": TwoPointCalibrationConstructionError,
        "observation": TwoPointObservationValidationError,
        "update": TwoPointUpdateConstructionError,
    }[error_type]
    for code in codes:
        error = error_class(code, "details")
        assert error.code == code
        assert error.message == "details"
    with pytest.raises(ValueError):
        error_class("unknown", "details")
    with pytest.raises(ValueError):
        error_class(codes[0], "")
