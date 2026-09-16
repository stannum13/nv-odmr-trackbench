"""Deterministic, download-free Stage 6.4 scientific acceptance fixtures.

This module deliberately lives under ``tests``.  It may inspect hidden truth,
but only through :func:`evaluate_released_scan_truth` after a scan's public
release record exists.  Production tracking receives no truth handle.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import cast
from unittest.mock import patch

from odmr_bench.dynamics import SpectralDynamics, SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import GaussianNoise, PoissonNoise
from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.noise import CheckpointableNoise
from odmr_bench.estimators import (
    SparseLinewidthCompositeTracker,
    SparseLinewidthConfiguration,
    TwoPointBudgetCeiling,
    TwoPointIdentityBinding,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    calibrate_two_point,
)
from odmr_bench.evaluation.sparse_linewidth.runner import (
    SparseLinewidthEvaluatorRunner,
)
from odmr_bench.evaluation.sparse_linewidth.types import (
    SparseLinewidthEvaluatorScanTiming,
    SparseRunnerAccepted,
    SparseRunnerExternallyStopped,
    SparseRunnerStepOutcome,
)
from odmr_bench.evaluation.two_point.types import VerifiedTwoPointCalibrationSuccess
from odmr_bench.models import Baseline, Resonance
from tests.evaluation.sparse_linewidth_fixture_dynamics import (
    DeterministicLinewidthDrift,
)
from tests.two_point_helpers import make_legal_fit_configuration, make_legal_source_fit


class QueryScopedDynamicsSpy:
    """Record whether each hidden-truth evaluation occurs inside a query."""

    def __init__(self, base: SpectralDynamics) -> None:
        self.base = base
        self.inside_query = False
        self.calls: list[tuple[float, bool]] = []

    @contextmanager
    def query_scope(self):  # type: ignore[no-untyped-def]
        previous = self.inside_query
        self.inside_query = True
        try:
            yield
        finally:
            self.inside_query = previous

    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        self.calls.append((timestamp_s, self.inside_query))
        return self.base.snapshot_at(timestamp_s)

    def clear_calls(self) -> None:
        self.calls.clear()


@dataclass(frozen=True, slots=True)
class AffineBaselineDrift:
    base: SpectralSnapshot
    slope_slew_per_hz_per_s: float

    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        baseline = replace(
            self.base.baseline,
            slope_per_hz=(
                self.base.baseline.slope_per_hz
                + self.slope_slew_per_hz_per_s * timestamp_s
            ),
        )
        return SpectralSnapshot(baseline=baseline, resonances=self.base.resonances)


@dataclass(frozen=True, slots=True)
class QuadraticWithinScanWidthDynamics:
    base: SpectralSnapshot
    curvature_hz_per_s2: float

    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        resonances = tuple(
            replace(
                item,
                fwhm_hz=item.fwhm_hz + self.curvature_hz_per_s2 * timestamp_s**2,
            )
            for item in self.base.resonances
        )
        return SpectralSnapshot(baseline=self.base.baseline, resonances=resonances)


@dataclass(frozen=True, slots=True)
class SparseAcceptanceCase:
    """Complete public-input recipe for one deterministic acceptance run."""

    name: str
    dynamics: SpectralDynamics
    noise: CheckpointableNoise
    source_snapshot: SpectralSnapshot
    seed: int = 2718
    nominal_photon_rate_hz: float = 2.5e9
    frequency_overhead_s: float = 0.001
    integration_time_s: float = 0.005
    accepted_steps: int = 21
    calibration_frequency_hz: tuple[float, float] = (2.74e9, 3.02e9)
    source_clock_id: str = "acceptance-clock"
    tracker_clock_id: str = "acceptance-clock"
    source_to_tracker_offset_s: float = 0.0
    configuration: SparseLinewidthConfiguration = field(
        default_factory=SparseLinewidthConfiguration
    )
    budget: TwoPointBudgetCeiling = field(
        default_factory=lambda: TwoPointBudgetCeiling(100, None, None, None)
    )
    terminal_action: str = "external_stop"


def acceptance_snapshot(
    *,
    baseline_slope_per_hz: float = 0.0,
    fwhm_hz: float = 1.5e6,
) -> SpectralSnapshot:
    """Return the canonical eight-resonance acceptance truth/source snapshot."""
    return SpectralSnapshot(
        baseline=Baseline(
            intercept=1.0,
            slope_per_hz=baseline_slope_per_hz,
            reference_hz=2.88e9,
        ),
        resonances=tuple(
            Resonance(
                resonance_id=f"r{index}",
                center_hz=2.76e9 + index * 34.0e6,
                fwhm_hz=fwhm_hz,
                amplitude=0.02,
                eta=0.5,
            )
            for index in range(8)
        ),
    )


def exact_static_case() -> SparseAcceptanceCase:
    snapshot = acceptance_snapshot()
    return SparseAcceptanceCase(
        name="exact-static-four-parameter-recovery",
        dynamics=StationaryDynamics(snapshot),
        noise=GaussianNoise(stddev_at_1s=0.0),
        source_snapshot=snapshot,
    )


def seeded_poisson_static_case() -> SparseAcceptanceCase:
    snapshot = acceptance_snapshot()
    return SparseAcceptanceCase(
        name="seeded-poisson-static-fixed-tolerance",
        dynamics=StationaryDynamics(snapshot),
        noise=PoissonNoise(),
        source_snapshot=snapshot,
        seed=1729,
        nominal_photon_rate_hz=1.0e10,
    )


def composed_center_width_drift_case() -> SparseAcceptanceCase:
    from odmr_bench.dynamics import LinearCenterDrift

    snapshot = acceptance_snapshot()
    center = LinearCenterDrift(snapshot, 60_000.0)
    widths = {item.resonance_id: item.fwhm_hz for item in snapshot.resonances}
    return SparseAcceptanceCase(
        name="composed-center-and-width-drift",
        dynamics=DeterministicLinewidthDrift(center, widths, 40_000.0),
        noise=GaussianNoise(stddev_at_1s=0.0),
        source_snapshot=snapshot,
    )


def independent_source_epoch_case() -> SparseAcceptanceCase:
    base = exact_static_case()
    return replace(
        base,
        name="independent-source-and-tracker-epochs",
    )


def retained_width_failure_case() -> SparseAcceptanceCase:
    """High deterministic shot noise creates a scientific, not runner, failure."""
    base = seeded_poisson_static_case()
    return replace(
        base,
        name="scientific-failure-retains-prior-width",
        nominal_photon_rate_hz=2.0e5,
        seed=91,
    )


def affine_baseline_mismatch_case() -> SparseAcceptanceCase:
    snapshot = acceptance_snapshot()
    return SparseAcceptanceCase(
        name="affine-within-scan-baseline-mismatch",
        dynamics=AffineBaselineDrift(snapshot, 8.0e-10),
        noise=GaussianNoise(stddev_at_1s=0.0),
        source_snapshot=snapshot,
    )


def within_scan_dynamics_case() -> SparseAcceptanceCase:
    snapshot = acceptance_snapshot()
    return SparseAcceptanceCase(
        name="nonlinear-within-scan-width-mismatch",
        dynamics=QuadraticWithinScanWidthDynamics(snapshot, 2.0e7),
        noise=GaussianNoise(stddev_at_1s=0.0),
        source_snapshot=snapshot,
    )


def due_geometry_stop_case() -> SparseAcceptanceCase:
    from odmr_bench.dynamics import LinearCenterDrift

    snapshot = acceptance_snapshot()
    return SparseAcceptanceCase(
        name="due-geometry-clean-stop",
        dynamics=LinearCenterDrift(snapshot, -2.0e6),
        noise=GaussianNoise(stddev_at_1s=0.0),
        source_snapshot=snapshot,
        accepted_steps=16,
        calibration_frequency_hz=(2.758485e9, 3.02e9),
        terminal_action="next_event",
    )


def unaffordable_sparse_block_case() -> SparseAcceptanceCase:
    base = exact_static_case()
    return replace(
        base,
        name="unaffordable-five-point-block",
        accepted_steps=16,
        budget=TwoPointBudgetCeiling(22, None, None, None),
        terminal_action="next_event",
    )


def run_acceptance_case_with_trace(
    case: SparseAcceptanceCase,
) -> tuple[
    SparseRunnerExternallyStopped | SparseRunnerStepOutcome,
    tuple[SparseRunnerAccepted, ...],
]:
    """Run one acceptance recipe and retain every accepted public outcome."""
    instrument = ODMRInstrument(
        dynamics=case.dynamics,
        noise=case.noise,
        nominal_photon_rate_hz=case.nominal_photon_rate_hz,
        frequency_overhead_s=case.frequency_overhead_s,
        seed=case.seed,
    )
    if hasattr(case.dynamics, "clear_calls"):
        case.dynamics.clear_calls()  # type: ignore[attr-defined]
    runner = SparseLinewidthEvaluatorRunner.bind(instrument)
    fit_configuration = make_legal_fit_configuration()
    source_fit = make_legal_source_fit(fit_configuration)
    source_fit = replace(
        source_fit,
        baseline_estimate=case.source_snapshot.baseline,
        resonance_estimates=case.source_snapshot.resonances,
        initial_guess=replace(
            source_fit.initial_guess,
            resonances=case.source_snapshot.resonances,
            baseline=case.source_snapshot.baseline,
        ),
    )
    original_query = ODMRInstrument.query

    def scoped_query(queried: ODMRInstrument, *args: object, **kwargs: object):
        scope = (
            case.dynamics.query_scope()  # type: ignore[attr-defined]
            if queried is instrument and hasattr(case.dynamics, "query_scope")
            else nullcontext()
        )
        with scope:
            return original_query(queried, *args, **kwargs)  # type: ignore[arg-type]

    with (
        patch.object(ODMRInstrument, "query", scoped_query),
        patch(
            "odmr_bench.evaluation.two_point.calibration.fit_spectrum",
            return_value=source_fit,
        ),
    ):
        verified = runner.acquire_verified_calibration(
            case.calibration_frequency_hz,
            case.integration_time_s,
            fit_configuration,
            TwoPointIdentityBinding(
                "require_expected_ids", fit_configuration.resonance_ids
            ),
            source_id=f"{case.name}-source",
            source_clock_id=case.source_clock_id,
            tracker_clock_id=case.tracker_clock_id,
            source_to_tracker_offset_s=case.source_to_tracker_offset_s,
            physical_fit_epoch_rule="instrument_midpoint_ordered_mean",
        )
        assert type(verified) is VerifiedTwoPointCalibrationSuccess
        calibration = calibrate_two_point(
            verified.source,
            TwoPointTrackerConfiguration(),
            budget_treatment="included_same_run",
        )
        tracker = SparseLinewidthCompositeTracker(case.configuration)
        metadata = TwoPointRunMetadata(
            tracker_clock_id=case.tracker_clock_id,
            current_sequence_index=runner.state.instrument_current_sequence_index,
            current_timestamp_s=runner.state.current_virtual_time_s,
            nominal_photon_rate_hz=instrument.nominal_photon_rate_hz,
            frequency_overhead_s=instrument.frequency_overhead_s,
            fluorescence_quantity="normalized_fluorescence",
        )
        runner.start_tracking(
            tracker,
            calibration,
            verified,
            metadata,
            case.budget,
            seed=case.seed,
        )
        accepted: list[SparseRunnerAccepted] = []
        for _ in range(case.accepted_steps):
            outcome = runner.step()
            if outcome.kind != "accepted":
                raise AssertionError(f"{case.name} ended early: {outcome!r}")
            accepted.append(cast(SparseRunnerAccepted, outcome))
        if case.terminal_action == "external_stop":
            result = runner.stop_external()
        elif case.terminal_action == "next_event":
            result = runner.step()
        else:
            raise AssertionError(f"unknown terminal action: {case.terminal_action}")
    return (
        cast(SparseRunnerExternallyStopped | SparseRunnerStepOutcome, result),
        tuple(accepted),
    )


def run_acceptance_case(
    case: SparseAcceptanceCase,
) -> SparseRunnerExternallyStopped | SparseRunnerStepOutcome:
    """Run one acceptance recipe and return only its declared terminal event."""
    terminal, _ = run_acceptance_case_with_trace(case)
    return terminal


def evaluate_released_scan_truth(
    dynamics: SpectralDynamics,
    timing: SparseLinewidthEvaluatorScanTiming,
    *,
    completed_release_sequence_index: int,
) -> SpectralSnapshot:
    """Evaluate hidden truth once, only after the scan is publicly released."""
    if not isinstance(dynamics, SpectralDynamics):
        raise TypeError("dynamics must implement SpectralDynamics")
    if type(timing) is not SparseLinewidthEvaluatorScanTiming:
        raise TypeError("timing must be a SparseLinewidthEvaluatorScanTiming")
    if isinstance(completed_release_sequence_index, bool) or not isinstance(
        completed_release_sequence_index, int
    ):
        raise TypeError("completed_release_sequence_index must be an integer")
    if completed_release_sequence_index != timing.release_sequence_index:
        raise ValueError("timing has not been completed and released")
    return dynamics.snapshot_at(timing.truth_reference_timestamp_s)
