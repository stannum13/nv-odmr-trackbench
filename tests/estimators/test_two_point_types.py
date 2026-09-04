"""Tests for calibrated two-point public primitive contracts."""

from dataclasses import replace

import numpy as np
import pytest


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
