from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

import odmr_bench

_PRE_STAGE_64_ESTIMATOR_EXPORTS = (
    "CalibratedTwoPointTracker",
    "CalibrationBudgetTreatment",
    "CalibrationIdentityMode",
    "CalibrationSourceProvenance",
    "ClockMappingKind",
    "CompleteSweep",
    "FitConfiguration",
    "FitInitialGuess",
    "FitUncertainty",
    "InitializationDiagnostics",
    "NormalizedFluorescenceProvenance",
    "PairSide",
    "PublicAcquisitionResources",
    "RepeatedFullSweepEstimator",
    "SpectrumFitResult",
    "SweepEstimate",
    "SweepFitAttempt",
    "SweepStartKind",
    "TwoPointBudgetCeiling",
    "TwoPointCalibration",
    "TwoPointCalibrationConstructionCode",
    "TwoPointCalibrationConstructionError",
    "TwoPointCalibrationSource",
    "TwoPointClockMapping",
    "TwoPointEstimate",
    "TwoPointFailureCode",
    "TwoPointIdentityBinding",
    "TwoPointIdentityCalibration",
    "TwoPointIdentityEstimate",
    "TwoPointLockState",
    "TwoPointObservationValidationCode",
    "TwoPointObservationValidationError",
    "TwoPointPairResult",
    "TwoPointPartialPair",
    "TwoPointQuery",
    "TwoPointRunMetadata",
    "TwoPointStopReason",
    "TwoPointTrackerConfiguration",
    "TwoPointUpdate",
    "TwoPointUpdateConstructionCode",
    "TwoPointUpdateConstructionError",
    "WarmStartDisposition",
    "WarmStartRejectionCode",
    "WarmStartedFullSweepEstimator",
    "WarmSweepEstimate",
    "bind_caller_asserted_two_point_calibration_source",
    "calibrate_two_point",
    "fit_spectrum",
    "initialize_spectrum",
    "linearized_standard_errors",
)

_SPARSE_ESTIMATOR_EXPORTS = (
    "CompositeIdentityEstimate",
    "CompositeMode",
    "CompositeStopReason",
    "SparseGeometryFailureCode",
    "SparseGeometryUnavailableDiagnostic",
    "SparseLinewidthCompositeEstimate",
    "SparseLinewidthCompositeTracker",
    "SparseLinewidthCompositeUpdate",
    "SparseLinewidthConfiguration",
    "SparseLinewidthFailureCode",
    "SparseLinewidthObservationValidationError",
    "SparseLinewidthQuery",
    "SparseLinewidthResetError",
    "SparseLinewidthScanResult",
    "SparseLinewidthSourceKind",
    "SparseLinewidthUpdateConstructionError",
    "SparseObservationValidationCode",
    "SparsePartialScan",
    "SparseResetFailureCode",
    "SparseUpdateConstructionCode",
)

_SPARSE_EVALUATOR_EXPORTS = (
    "SparseAbortReason",
    "SparseAbortedRun",
    "SparseEvaluatorRunnerState",
    "SparseInstrumentQueryFailure",
    "SparseLinewidthEvaluatorResources",
    "SparseLinewidthEvaluatorRunner",
    "SparseLinewidthEvaluatorScanTiming",
    "SparsePreflightCode",
    "SparsePreflightError",
    "SparseResourceJoinUnavailableAcquisition",
    "SparseRunnerAborted",
    "SparseRunnerAccepted",
    "SparseRunnerBudgetStopped",
    "SparseRunnerExternallyStopped",
    "SparseRunnerGeometryStopped",
    "SparseRunnerInstrumentFailure",
    "SparseRunnerPhase",
    "SparseRunnerRunOutcome",
    "SparseRunnerStateError",
    "SparseRunnerStepOutcome",
    "SparseStartCode",
    "SparseStartError",
    "SparseTrackingAcquisition",
    "build_sparse_linewidth_evaluator_resources",
)

_EXPECTED_ESTIMATOR_EXPORTS = (
    "CalibratedTwoPointTracker",
    "CalibrationBudgetTreatment",
    "CalibrationIdentityMode",
    "CalibrationSourceProvenance",
    "ClockMappingKind",
    "CompleteSweep",
    "CompositeIdentityEstimate",
    "CompositeMode",
    "CompositeStopReason",
    "FitConfiguration",
    "FitInitialGuess",
    "FitUncertainty",
    "InitializationDiagnostics",
    "NormalizedFluorescenceProvenance",
    "PairSide",
    "PublicAcquisitionResources",
    "RepeatedFullSweepEstimator",
    "SparseGeometryFailureCode",
    "SparseGeometryUnavailableDiagnostic",
    "SparseLinewidthCompositeEstimate",
    "SparseLinewidthCompositeTracker",
    "SparseLinewidthCompositeUpdate",
    "SparseLinewidthConfiguration",
    "SparseLinewidthFailureCode",
    "SparseLinewidthObservationValidationError",
    "SparseLinewidthQuery",
    "SparseLinewidthResetError",
    "SparseLinewidthScanResult",
    "SparseLinewidthSourceKind",
    "SparseLinewidthUpdateConstructionError",
    "SparseObservationValidationCode",
    "SparsePartialScan",
    "SparseResetFailureCode",
    "SparseUpdateConstructionCode",
    "SpectrumFitResult",
    "SweepEstimate",
    "SweepFitAttempt",
    "SweepStartKind",
    "TwoPointBudgetCeiling",
    "TwoPointCalibration",
    "TwoPointCalibrationConstructionCode",
    "TwoPointCalibrationConstructionError",
    "TwoPointCalibrationSource",
    "TwoPointClockMapping",
    "TwoPointEstimate",
    "TwoPointFailureCode",
    "TwoPointIdentityBinding",
    "TwoPointIdentityCalibration",
    "TwoPointIdentityEstimate",
    "TwoPointLockState",
    "TwoPointObservationValidationCode",
    "TwoPointObservationValidationError",
    "TwoPointPairResult",
    "TwoPointPartialPair",
    "TwoPointQuery",
    "TwoPointRunMetadata",
    "TwoPointStopReason",
    "TwoPointTrackerConfiguration",
    "TwoPointUpdate",
    "TwoPointUpdateConstructionCode",
    "TwoPointUpdateConstructionError",
    "WarmStartDisposition",
    "WarmStartRejectionCode",
    "WarmStartedFullSweepEstimator",
    "WarmSweepEstimate",
    "bind_caller_asserted_two_point_calibration_source",
    "calibrate_two_point",
    "fit_spectrum",
    "initialize_spectrum",
    "linearized_standard_errors",
)


def _assert_exact_estimator_exports(candidate: tuple[str, ...]) -> None:
    assert candidate == _EXPECTED_ESTIMATOR_EXPORTS
    assert tuple(
        name for name in candidate if name in _PRE_STAGE_64_ESTIMATOR_EXPORTS
    ) == _PRE_STAGE_64_ESTIMATOR_EXPORTS


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


def test_stage_64_public_surface_is_exact_and_importable() -> None:
    from odmr_bench import estimators
    from odmr_bench.evaluation import sparse_linewidth

    estimator_exports = tuple(estimators.__all__)
    _assert_exact_estimator_exports(estimator_exports)
    sparse_exports = tuple(
        name for name in estimator_exports if name in _SPARSE_ESTIMATOR_EXPORTS
    )
    assert sparse_exports == _SPARSE_ESTIMATOR_EXPORTS
    assert tuple(sparse_linewidth.__all__) == _SPARSE_EVALUATOR_EXPORTS
    assert all(
        getattr(estimators, name) is not None for name in _SPARSE_ESTIMATOR_EXPORTS
    )
    assert all(
        getattr(sparse_linewidth, name) is not None
        for name in _SPARSE_EVALUATOR_EXPORTS
    )

    forbidden = (
        "fit_sparse_linewidth",
        "_VerifiedCalibrationIssuer",
        "VerifiedInstrumentRunToken",
        "_register_run_token",
        "_evaluate_bound_source_model",
    )
    assert not any(name in estimators.__all__ for name in forbidden)
    assert not any(name in sparse_linewidth.__all__ for name in forbidden)


def test_exact_estimator_export_contract_rejects_private_fitter_mutation() -> None:
    from odmr_bench import estimators

    mutated = (*estimators.__all__, "fit_sparse_linewidth")
    with pytest.raises(AssertionError):
        _assert_exact_estimator_exports(mutated)


def test_sparse_linewidth_example_runs_out_of_tree(tmp_path: Path) -> None:
    example = Path(__file__).parents[1] / "examples" / "track_sparse_linewidth.py"
    completed = subprocess.run(
        [sys.executable, "-I", str(example)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    required_labels = (
        "terminal_phase=budget_stopped",
        "live_projection_q=",
        "scan_local_q=",
        "center_source_epoch_s=",
        "linewidth_source_epoch_s=",
        "accepted_tracking_ledger=",
        "charged_ledger=",
        "completed_fast_pairs=",
        "completed_sparse_scans=",
        "public_reference_timestamp_s=",
        "truth_reference_timestamp_s=",
    )
    assert all(label in completed.stdout for label in required_labels)
    forbidden_claims = ("superior", "experimental result", "sensitivity result")
    assert not any(claim in completed.stdout.lower() for claim in forbidden_claims)


def test_sparse_linewidth_guidance_covers_scientific_contract() -> None:
    root = Path(__file__).parents[1]
    combined = "\n".join(
        (
            (root / "README.md").read_text().lower(),
            (root / "docs" / "estimators.md").read_text().lower(),
        )
    )
    required_terms = (
        "(+0.5, -1.0, 0.0, +1.0, -0.5)",
        "(-0.5, +1.0, 0.0, -1.0, +0.5)",
        "four free parameters",
        "fitted local center",
        "frozen",
        "diagnostic presence",
        "| `nonfinite_solution` | present | absent | absent | absent | absent |",
        "asynchronous live q",
        "scan-local q",
        "public reference timestamp",
        "truth reference timestamp",
        "conditional_free_precalibration",
        "partial sparse scan",
        "retryable instrument failure",
        "terminal abort",
        "affine baseline",
        "within-scan dynamics",
        "no stage 6.5 matched-budget",
        "no stage 6.6",
        "examples/track_sparse_linewidth.py",
    )
    assert all(term in combined for term in required_terms)


def test_sparse_docs_state_per_scan_zero_time_frequency_sum_and_limit() -> None:
    root = Path(__file__).parents[1]
    guidance = " ".join(
        (root / "docs" / "estimators.md").read_text().lower().split()
    )
    centered_indices = (-2.0, -1.0, 0.0, 1.0, 2.0)
    parity_orders = (
        (0.5, -1.0, 0.0, 1.0, -0.5),
        (-0.5, 1.0, 0.0, -1.0, 0.5),
    )

    assert tuple(
        sum(
            time * offset
            for time, offset in zip(centered_indices, order, strict=True)
        )
        for order in parity_orders
    ) == (0.0, 0.0)
    assert "each parity order independently" in guidance
    assert "sum(t*x) == 0" in guidance
    assert "balances acquisition-order systematics across repeated scans" in guidance
    assert "does not identify or correct within-scan parameter dynamics" in guidance
    assert "does not establish robustness to within-scan dynamics" in guidance


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
