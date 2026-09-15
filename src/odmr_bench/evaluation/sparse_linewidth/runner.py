"""Instrument-owning shell for sparse-linewidth evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators.sparse_linewidth_tracker import (
    SparseLinewidthCompositeTracker,
)
from odmr_bench.estimators.two_point_types import (
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointIdentityBinding,
    TwoPointRunMetadata,
)
from odmr_bench.estimators.types import FitConfiguration
from odmr_bench.evaluation.two_point.provenance import (
    _TOKEN_CONSTRUCTION_KEY,
    _mint_verified_instrument_run_token,
    _register_run_token,
    _rollback_run_token_registration,
)
from odmr_bench.evaluation.two_point.resource_accounting import (
    _zero_full_resources,
)
from odmr_bench.evaluation.two_point.types import (
    TwoPointEvaluatorInstrumentConfiguration,
    VerifiedTwoPointCalibrationOutcome,
    VerifiedTwoPointCalibrationSuccess,
)

from .types import (
    SparseEvaluatorRunnerState,
    SparsePreflightError,
    SparseRunnerExternallyStopped,
    SparseRunnerRunOutcome,
    SparseRunnerStateError,
    SparseRunnerStepOutcome,
    SparseStartError,
)


class SparseLinewidthEvaluatorRunner:
    """Own one instrument association for sparse-linewidth evaluation."""

    __slots__ = ("_instrument", "_state", "_tracker")

    @classmethod
    def bind(cls, instrument: ODMRInstrument) -> SparseLinewidthEvaluatorRunner:
        """Bind one exact clean instrument and register private authority."""
        if (
            cls is not SparseLinewidthEvaluatorRunner
            or type(instrument) is not ODMRInstrument
        ):
            raise SparsePreflightError("invalid_argument_type")
        try:
            nominal_photon_rate_hz = instrument.nominal_photon_rate_hz
            frequency_overhead_s = instrument.frequency_overhead_s
            resources = instrument.resources
            current_virtual_time_s = instrument.virtual_time_s
        except Exception as error:
            raise SparsePreflightError("invalid_argument_value") from error
        if (
            type(nominal_photon_rate_hz) is not float
            or type(frequency_overhead_s) is not float
            or type(resources) is not ResourceSnapshot
            or type(current_virtual_time_s) is not float
        ):
            raise SparsePreflightError("invalid_argument_type")
        try:
            instrument_configuration = TwoPointEvaluatorInstrumentConfiguration(
                nominal_photon_rate_hz=nominal_photon_rate_hz,
                frequency_overhead_s=frequency_overhead_s,
            )
            resources = ResourceSnapshot(
                resources.observations,
                resources.integration_time_s,
                resources.nominal_exposure_photons,
                resources.expected_photons,
                resources.realized_photons,
                resources.observations_without_realized_counts,
                resources.virtual_elapsed_time_s,
            )
            if (
                not math.isfinite(current_virtual_time_s)
                or current_virtual_time_s < 0.0
            ):
                raise ValueError("current virtual time must be finite and nonnegative")
        except TypeError as error:
            raise SparsePreflightError("invalid_argument_type") from error
        except ValueError as error:
            raise SparsePreflightError("invalid_argument_value") from error
        if resources != _zero_full_resources() or current_virtual_time_s != 0.0:
            raise SparsePreflightError("unclean_instrument_boundary")

        token = _mint_verified_instrument_run_token(_TOKEN_CONSTRUCTION_KEY)
        runner = object.__new__(cls)
        object.__setattr__(runner, "_instrument", instrument)
        object.__setattr__(runner, "_tracker", None)
        object.__setattr__(
            runner,
            "_state",
            SparseEvaluatorRunnerState(
                phase="ready",
                run_token=token,
                instrument_configuration=instrument_configuration,
                calibration_outcome=None,
                verified_calibration=None,
                calibration=None,
                tracker_estimate=None,
                normal_tracking_trace=(),
                pair_timings=(),
                scan_timings=(),
                instrument_resources_at_bind=resources,
                tracking_resources_before=None,
                instrument_resources_current=resources,
                instrument_current_sequence_index=None,
                current_virtual_time_s=current_virtual_time_s,
                last_instrument_failure=None,
                terminal_abort=None,
                fast_update_cpu_time_s=0.0,
                sparse_update_cpu_time_s=0.0,
                total_update_cpu_time_s=0.0,
            ),
        )
        try:
            _register_run_token(token, runner, instrument, instrument_configuration)
        except BaseException:
            _rollback_run_token_registration(token)
            raise
        return runner

    @property
    def state(self) -> SparseEvaluatorRunnerState:
        """Return the current frozen audit snapshot."""
        return self._state

    def acquire_verified_calibration(
        self,
        frequency_hz: Sequence[float],
        integration_time_s: float,
        fit_configuration: FitConfiguration,
        identity_binding: TwoPointIdentityBinding,
        *,
        source_id: str,
        source_clock_id: str,
        tracker_clock_id: str,
        source_to_tracker_offset_s: float,
        physical_fit_epoch_rule: Literal["instrument_midpoint_ordered_mean"],
    ) -> VerifiedTwoPointCalibrationOutcome:
        """Reject calibration until the Task 13 transition is installed."""
        del (
            frequency_hz,
            integration_time_s,
            fit_configuration,
            identity_binding,
            source_id,
            source_clock_id,
            tracker_clock_id,
            source_to_tracker_offset_s,
            physical_fit_epoch_rule,
        )
        raise SparsePreflightError("invalid_runner_phase")

    def start_tracking(
        self,
        tracker: SparseLinewidthCompositeTracker,
        calibration: TwoPointCalibration,
        verified_calibration: VerifiedTwoPointCalibrationSuccess,
        public_metadata: TwoPointRunMetadata,
        budget_ceiling: TwoPointBudgetCeiling,
        *,
        seed: int,
    ) -> SparseEvaluatorRunnerState:
        """Reject tracking start until the Task 13 transition is installed."""
        del (
            tracker,
            calibration,
            verified_calibration,
            public_metadata,
            budget_ceiling,
            seed,
        )
        raise SparseStartError("invalid_runner_phase")

    def step(self) -> SparseRunnerStepOutcome:
        """Reject steps until a later task installs tracking transitions."""
        raise SparseRunnerStateError("step requires a runner in the tracking phase")

    def run_until_event(self) -> SparseRunnerRunOutcome:
        """Reject runs until a later task installs tracking transitions."""
        raise SparseRunnerStateError(
            "run_until_event requires a runner in the tracking phase"
        )

    def stop_external(self) -> SparseRunnerExternallyStopped:
        """Reject stops until a later task installs tracking transitions."""
        raise SparseRunnerStateError(
            "stop_external requires a runner in the tracking phase"
        )
