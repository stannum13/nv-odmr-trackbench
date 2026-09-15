from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import odmr_bench


def test_package_exposes_version() -> None:
    assert odmr_bench.__version__ == "0.1.0"


def test_cli_reports_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "odmr_bench.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "odmrbench 0.1.0"


def test_two_point_public_surfaces_import_from_installed_modules() -> None:
    from odmr_bench.estimators import (
        CalibratedTwoPointTracker,
        CalibrationBudgetTreatment,
        CalibrationIdentityMode,
        CalibrationSourceProvenance,
        ClockMappingKind,
        NormalizedFluorescenceProvenance,
        PairSide,
        PublicAcquisitionResources,
        TwoPointBudgetCeiling,
        TwoPointCalibration,
        TwoPointCalibrationConstructionCode,
        TwoPointCalibrationConstructionError,
        TwoPointCalibrationSource,
        TwoPointClockMapping,
        TwoPointEstimate,
        TwoPointFailureCode,
        TwoPointIdentityBinding,
        TwoPointIdentityCalibration,
        TwoPointIdentityEstimate,
        TwoPointLockState,
        TwoPointObservationValidationCode,
        TwoPointObservationValidationError,
        TwoPointPairResult,
        TwoPointPartialPair,
        TwoPointQuery,
        TwoPointRunMetadata,
        TwoPointStopReason,
        TwoPointTrackerConfiguration,
        TwoPointUpdate,
        TwoPointUpdateConstructionCode,
        TwoPointUpdateConstructionError,
        bind_caller_asserted_two_point_calibration_source,
        calibrate_two_point,
    )
    from odmr_bench.evaluation.two_point import (
        ResourceJoinMismatchField,
        TwoPointAbortedRun,
        TwoPointAbortReason,
        TwoPointCalibrationPreflightError,
        TwoPointEvaluatorInstrumentConfiguration,
        TwoPointEvaluatorPairTiming,
        TwoPointEvaluatorResources,
        TwoPointEvaluatorRunner,
        TwoPointEvaluatorRunnerState,
        TwoPointInstrumentQueryFailure,
        TwoPointResourceJoinUnavailableAcquisition,
        TwoPointRunnerAborted,
        TwoPointRunnerAccepted,
        TwoPointRunnerBudgetStopped,
        TwoPointRunnerExternallyStopped,
        TwoPointRunnerInstrumentFailure,
        TwoPointRunnerPhase,
        TwoPointRunnerRunOutcome,
        TwoPointRunnerStartError,
        TwoPointRunnerStartFailureCode,
        TwoPointRunnerStateError,
        TwoPointRunnerStepOutcome,
        TwoPointTrackingAcquisition,
        VerifiedCalibrationFailureCode,
        VerifiedCalibrationPreflightCode,
        VerifiedCalibrationQueryRequest,
        VerifiedInstrumentRunToken,
        VerifiedTwoPointCalibrationFailure,
        VerifiedTwoPointCalibrationOutcome,
        VerifiedTwoPointCalibrationSuccess,
        build_two_point_evaluator_resources,
    )

    assert all(
        value is not None
        for value in (
            CalibrationBudgetTreatment,
            CalibrationIdentityMode,
            CalibrationSourceProvenance,
            CalibratedTwoPointTracker,
            ClockMappingKind,
            NormalizedFluorescenceProvenance,
            PairSide,
            PublicAcquisitionResources,
            TwoPointBudgetCeiling,
            TwoPointCalibration,
            TwoPointCalibrationConstructionCode,
            TwoPointCalibrationConstructionError,
            TwoPointCalibrationSource,
            TwoPointClockMapping,
            TwoPointEstimate,
            TwoPointFailureCode,
            TwoPointIdentityBinding,
            TwoPointIdentityCalibration,
            TwoPointIdentityEstimate,
            TwoPointLockState,
            TwoPointObservationValidationCode,
            TwoPointObservationValidationError,
            TwoPointPairResult,
            TwoPointPartialPair,
            TwoPointQuery,
            TwoPointRunMetadata,
            TwoPointStopReason,
            TwoPointTrackerConfiguration,
            TwoPointUpdate,
            TwoPointUpdateConstructionCode,
            TwoPointUpdateConstructionError,
            bind_caller_asserted_two_point_calibration_source,
            calibrate_two_point,
            ResourceJoinMismatchField,
            TwoPointAbortReason,
            TwoPointAbortedRun,
            TwoPointCalibrationPreflightError,
            TwoPointEvaluatorInstrumentConfiguration,
            TwoPointEvaluatorPairTiming,
            TwoPointEvaluatorResources,
            TwoPointEvaluatorRunner,
            TwoPointEvaluatorRunnerState,
            TwoPointInstrumentQueryFailure,
            TwoPointResourceJoinUnavailableAcquisition,
            TwoPointRunnerAborted,
            TwoPointRunnerAccepted,
            TwoPointRunnerBudgetStopped,
            TwoPointRunnerExternallyStopped,
            TwoPointRunnerInstrumentFailure,
            TwoPointRunnerPhase,
            TwoPointRunnerRunOutcome,
            TwoPointRunnerStartError,
            TwoPointRunnerStartFailureCode,
            TwoPointRunnerStateError,
            TwoPointRunnerStepOutcome,
            TwoPointTrackingAcquisition,
            VerifiedCalibrationFailureCode,
            VerifiedCalibrationPreflightCode,
            VerifiedCalibrationQueryRequest,
            VerifiedInstrumentRunToken,
            VerifiedTwoPointCalibrationFailure,
            VerifiedTwoPointCalibrationOutcome,
            VerifiedTwoPointCalibrationSuccess,
            build_two_point_evaluator_resources,
        )
    )


def test_two_point_example_runs_out_of_tree(tmp_path: Path) -> None:
    example = Path(__file__).parents[1] / "examples" / "track_two_point_centers.py"
    completed = subprocess.run(
        [sys.executable, "-I", str(example)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == "Synthetic calibrated two-point diagnostics"
    assert lines[1] == "treatment=conditional_free_precalibration"
    assert lines[2].split() == [
        "resonance_id",
        "lock_state",
        "center_hz",
        "observations",
        "integration_time_s",
        "virtual_elapsed_time_s",
        "public_reference_timestamp_s",
        "release_timestamp_s",
    ]
    assert len(lines) == 11
    for index, line in enumerate(lines[3:]):
        columns = line.split()
        assert columns[0] == f"r{index}"
        assert columns[1] in {"searching", "tracking", "lost"}
        assert all(math.isfinite(float(value)) for value in columns[2:])

    forbidden = (
        "truth",
        "error",
        "expected_photons",
        "comparison",
        "superiority",
    )
    assert not any(term in completed.stdout.lower() for term in forbidden)


def test_two_point_guidance_terms() -> None:
    root = Path(__file__).parents[1]
    estimator_guidance = (root / "docs" / "estimators.md").read_text().lower()
    readme = (root / "README.md").read_text().lower()
    combined = estimator_guidance + "\n" + readme

    required_terms = (
        "caller-asserted",
        "verified_factory_acquisition",
        "included_same_run",
        "conditional_free_precalibration",
        "fixed calibration cells",
        "minus then plus",
        "plus then minus",
        "positive discriminator slope",
        "policy-only",
        "common-mode",
        "zero-step refresh",
        "partial pair",
        "unaccepted observation",
        "token continuity",
        "public reference timestamp",
        "actual measurement midpoint",
        "release timestamp",
        "estimate age",
        "release age",
        "terminal abort",
        "inert",
        "examples/track_two_point_centers.py",
        "no stage 6.5 matched-budget superiority result",
    )
    assert all(term in combined for term in required_terms)
