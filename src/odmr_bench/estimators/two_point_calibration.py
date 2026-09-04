"""Construction helpers for calibrated two-point sources."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from itertools import pairwise

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators.two_point_resources import _replay_public_resources
from odmr_bench.estimators.two_point_types import (
    NormalizedFluorescenceProvenance,
    PublicAcquisitionResources,
    TwoPointCalibrationConstructionCode,
    TwoPointCalibrationConstructionError,
    TwoPointCalibrationSource,
    TwoPointClockMapping,
    TwoPointIdentityBinding,
)
from odmr_bench.estimators.types import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    SpectrumFitResult,
)
from odmr_bench.models import Resonance


def _fail(code: TwoPointCalibrationConstructionCode, message: str) -> None:
    raise TwoPointCalibrationConstructionError(code, message)


def _validate_argument_types(
    source_fit: object,
    fit_configuration: object,
    source_observations: object,
    identity_binding: object,
    fluorescence_provenance: object,
    source_id: object,
    source_frequency_overhead_s: object,
    source_start_timestamp_s: object,
    physical_fit_epoch_s: object,
    availability_sequence_index: object,
    availability_timestamp_s: object,
    clock_mapping: object,
) -> None:
    records_have_exact_types = (
        type(source_fit) is SpectrumFitResult
        and type(fit_configuration) is FitConfiguration
        and type(identity_binding) is TwoPointIdentityBinding
        and type(fluorescence_provenance) is NormalizedFluorescenceProvenance
        and type(clock_mapping) is TwoPointClockMapping
    )
    scalars_have_exact_types = (
        type(source_id) is str
        and type(source_frequency_overhead_s) is float
        and type(source_start_timestamp_s) is float
        and type(physical_fit_epoch_s) is float
        and type(availability_sequence_index) is int
        and type(availability_timestamp_s) is float
    )
    if type(source_observations) not in {list, tuple}:
        _fail(
            "invalid_argument_type",
            "source_observations must be an exact list or tuple",
        )
    observations = tuple(source_observations)
    if not records_have_exact_types or not scalars_have_exact_types or not all(
        type(item) is EstimatorObservation for item in observations
    ):
        _fail(
            "invalid_argument_type",
            "caller-asserted source arguments must have exact public types",
        )


def _validate_argument_values(
    *,
    source_id: str,
    source_frequency_overhead_s: float,
    source_start_timestamp_s: float,
    physical_fit_epoch_s: float,
    availability_sequence_index: int,
    availability_timestamp_s: float,
) -> None:
    if not source_id.strip():
        _fail("invalid_argument_value", "source_id must be nonblank")
    scalar_values = (
        source_frequency_overhead_s,
        source_start_timestamp_s,
        physical_fit_epoch_s,
        availability_timestamp_s,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in scalar_values):
        _fail(
            "invalid_argument_value",
            "source timing values must be finite and nonnegative",
        )
    if availability_sequence_index < 0:
        _fail(
            "invalid_argument_value",
            "availability_sequence_index must be nonnegative",
        )


def _validate_fluorescence_provenance(
    provenance: NormalizedFluorescenceProvenance,
) -> NormalizedFluorescenceProvenance:
    try:
        return NormalizedFluorescenceProvenance(
            provenance.quantity,
            provenance.normalization_rule,
            provenance.nominal_photon_rate_hz,
            provenance.sampling_rules,
        )
    except (TypeError, ValueError) as error:
        _fail(
            "invalid_provenance_or_quantity",
            f"invalid fluorescence provenance: {error}",
        )


def _validate_source_trace(
    observations: tuple[EstimatorObservation, ...],
    overhead_s: float,
    start_timestamp_s: float,
) -> tuple[EstimatorObservation, ...]:
    try:
        observations = tuple(
            EstimatorObservation(
                sequence_index=observation.sequence_index,
                timestamp_s=observation.timestamp_s,
                frequency_hz=observation.frequency_hz,
                fluorescence=observation.fluorescence,
                integration_time_s=observation.integration_time_s,
                nominal_exposure_photons=observation.nominal_exposure_photons,
                realized_photons=observation.realized_photons,
            )
            for observation in observations
        )
    except (TypeError, ValueError) as error:
        _fail("invalid_source_trace", f"invalid source observation: {error}")
    if len(observations) < 2:
        _fail(
            "invalid_source_trace",
            "source_observations must contain a complete sweep",
        )
    if any(
        current.sequence_index != previous.sequence_index + 1
        or current.timestamp_s <= previous.timestamp_s
        or current.frequency_hz <= previous.frequency_hz
        for previous, current in pairwise(observations)
    ):
        _fail(
            "invalid_source_trace",
            "source trace must be contiguous and strictly ordered",
        )
    previous_endpoint_s = start_timestamp_s
    for observation in observations:
        expected_endpoint_s = (
            previous_endpoint_s + overhead_s
        ) + observation.integration_time_s
        if observation.timestamp_s != expected_endpoint_s:
            _fail(
                "invalid_source_trace",
                "source timestamps must follow the exact endpoint recurrence",
            )
        previous_endpoint_s = observation.timestamp_s
    return observations


def _validate_source_resources(
    observations: tuple[EstimatorObservation, ...],
    provenance: NormalizedFluorescenceProvenance,
) -> None:
    if any(
        observation.nominal_exposure_photons
        != provenance.nominal_photon_rate_hz * observation.integration_time_s
        for observation in observations
    ):
        _fail(
            "source_resource_mismatch",
            "source nominal exposure must match the declared nominal photon rate",
        )


def _construct_fit_input(
    observations: tuple[EstimatorObservation, ...],
    resources: PublicAcquisitionResources,
) -> CompleteSweep:
    try:
        return CompleteSweep(
            frequency_hz=tuple(item.frequency_hz for item in observations),
            fluorescence=tuple(item.fluorescence for item in observations),
            last_sequence_index=observations[-1].sequence_index,
            last_timestamp_s=observations[-1].timestamp_s,
            total_integration_time_s=resources.integration_time_s,
            total_nominal_exposure_photons=resources.nominal_exposure_photons,
        )
    except (TypeError, ValueError) as error:
        _fail("fit_input_mismatch", f"invalid complete-sweep fit input: {error}")


def _validate_fit_input_provenance(
    source_fit: SpectrumFitResult,
    fit_configuration: FitConfiguration,
) -> FitConfiguration:
    try:
        fit_configuration = replace(fit_configuration)
    except (TypeError, ValueError) as error:
        _fail("fit_input_mismatch", f"invalid fit configuration: {error}")
    if (
        type(source_fit.model_kind) is not str
        or type(source_fit.baseline_degree) is not int
    ):
        _fail(
            "fit_input_mismatch",
            "source fit model and baseline provenance must use exact scalar types",
        )
    initial_guess = source_fit.initial_guess
    initial_ids = None
    if (
        type(initial_guess) is FitInitialGuess
        and type(initial_guess.resonances) is tuple
        and all(
            type(resonance) is Resonance
            and type(resonance.resonance_id) is str
            for resonance in initial_guess.resonances
        )
    ):
        initial_ids = tuple(
            resonance.resonance_id for resonance in initial_guess.resonances
        )
    fitted_resonances = source_fit.resonance_estimates
    fitted_ids = None
    if (
        source_fit.success is True
        and type(fitted_resonances) is tuple
        and len(fitted_resonances) == 8
        and all(
            type(resonance) is Resonance
            and type(resonance.resonance_id) is str
            for resonance in fitted_resonances
        )
    ):
        fitted_ids = tuple(
            resonance.resonance_id for resonance in fitted_resonances
        )
    if (
        source_fit.model_kind != fit_configuration.model_kind
        or source_fit.baseline_degree != fit_configuration.baseline_degree
        or (
            initial_ids is not None
            and initial_ids != fit_configuration.resonance_ids
        )
        or (
            fitted_ids is not None
            and fitted_ids != fit_configuration.resonance_ids
        )
    ):
        _fail(
            "fit_input_mismatch",
            "source fit must match the supplied fit configuration and trace facts",
        )
    return fit_configuration


def _validated_source_fit(
    source_fit: SpectrumFitResult,
) -> tuple[SpectrumFitResult, tuple[str, ...]]:
    try:
        source_fit = replace(source_fit)
    except (TypeError, ValueError) as error:
        _fail("source_fit_failed", f"invalid source fit: {error}")
    if (
        not source_fit.success
        or source_fit.failure_code is not None
        or source_fit.baseline_estimate is None
        or len(source_fit.resonance_estimates) != 8
    ):
        _fail("source_fit_failed", "source_fit must be a successful eight-line fit")
    return source_fit, tuple(
        item.resonance_id for item in source_fit.resonance_estimates
    )


def _validate_source_identity(
    identity_binding: TwoPointIdentityBinding,
    resolved_ids: tuple[str, ...],
) -> TwoPointIdentityBinding:
    try:
        identity_binding = TwoPointIdentityBinding(
            identity_binding.mode,
            identity_binding.expected_resonance_ids,
        )
    except (TypeError, ValueError) as error:
        _fail("source_identity_mismatch", f"invalid identity binding: {error}")
    if len(set(resolved_ids)) != 8 or (
        identity_binding.mode == "require_expected_ids"
        and identity_binding.expected_resonance_ids != resolved_ids
    ):
        _fail(
            "source_identity_mismatch",
            "source fit identities do not satisfy the requested identity mode",
        )
    return identity_binding


def _validate_source_epoch(
    observations: tuple[EstimatorObservation, ...], physical_fit_epoch_s: float
) -> None:
    first_midpoint_s = (
        observations[0].timestamp_s - observations[0].integration_time_s / 2.0
    )
    last_midpoint_s = (
        observations[-1].timestamp_s
        - observations[-1].integration_time_s / 2.0
    )
    expected_epoch_s = first_midpoint_s + (
        last_midpoint_s - first_midpoint_s
    ) / 2.0
    if physical_fit_epoch_s != expected_epoch_s:
        _fail(
            "invalid_source_epoch",
            "physical_fit_epoch_s must equal the exact public-midpoint mean",
        )


def _validate_availability_and_clock(
    observations: tuple[EstimatorObservation, ...],
    availability_sequence_index: int,
    availability_timestamp_s: float,
    clock_mapping: TwoPointClockMapping,
) -> TwoPointClockMapping:
    last_observation = observations[-1]
    try:
        clock_mapping = TwoPointClockMapping(
            clock_mapping.kind,
            clock_mapping.source_clock_id,
            clock_mapping.tracker_clock_id,
            clock_mapping.scale,
            clock_mapping.offset_s,
        )
    except (TypeError, ValueError) as error:
        _fail(
            "invalid_availability_or_clock",
            f"invalid clock mapping: {error}",
        )
    if (
        availability_sequence_index != last_observation.sequence_index
        or availability_timestamp_s != last_observation.timestamp_s
    ):
        _fail(
            "invalid_availability_or_clock",
            "availability and clock mapping must match the completed source trace",
        )
    return clock_mapping


def bind_caller_asserted_two_point_calibration_source(
    source_fit: SpectrumFitResult,
    fit_configuration: FitConfiguration,
    source_observations: Sequence[EstimatorObservation],
    identity_binding: TwoPointIdentityBinding,
    fluorescence_provenance: NormalizedFluorescenceProvenance,
    *,
    source_id: str,
    source_frequency_overhead_s: float,
    source_start_timestamp_s: float,
    physical_fit_epoch_s: float,
    availability_sequence_index: int,
    availability_timestamp_s: float,
    clock_mapping: TwoPointClockMapping,
) -> TwoPointCalibrationSource:
    """Bind a caller's self-consistent public calibration trace without refitting."""
    _validate_argument_types(
        source_fit,
        fit_configuration,
        source_observations,
        identity_binding,
        fluorescence_provenance,
        source_id,
        source_frequency_overhead_s,
        source_start_timestamp_s,
        physical_fit_epoch_s,
        availability_sequence_index,
        availability_timestamp_s,
        clock_mapping,
    )
    _validate_argument_values(
        source_id=source_id,
        source_frequency_overhead_s=source_frequency_overhead_s,
        source_start_timestamp_s=source_start_timestamp_s,
        physical_fit_epoch_s=physical_fit_epoch_s,
        availability_sequence_index=availability_sequence_index,
        availability_timestamp_s=availability_timestamp_s,
    )
    fluorescence_provenance = _validate_fluorescence_provenance(
        fluorescence_provenance
    )
    observations = tuple(source_observations)
    observations = _validate_source_trace(
        observations, source_frequency_overhead_s, source_start_timestamp_s
    )
    _validate_source_resources(observations, fluorescence_provenance)
    resources = _replay_public_resources(observations, source_frequency_overhead_s)
    first_observation = observations[0]
    last_observation = observations[-1]
    _construct_fit_input(observations, resources)
    fit_configuration = _validate_fit_input_provenance(
        source_fit, fit_configuration
    )
    source_fit, resolved_resonance_ids = _validated_source_fit(source_fit)
    if fit_configuration.resonance_ids != resolved_resonance_ids:
        _fail(
            "fit_input_mismatch",
            "source fit identities must match the supplied fit configuration",
        )
    identity_binding = _validate_source_identity(
        identity_binding, resolved_resonance_ids
    )
    _validate_source_epoch(observations, physical_fit_epoch_s)
    clock_mapping = _validate_availability_and_clock(
        observations,
        availability_sequence_index,
        availability_timestamp_s,
        clock_mapping,
    )

    return TwoPointCalibrationSource(
        source_id=source_id,
        provenance="caller_asserted",
        source_fit=source_fit,
        fit_configuration=fit_configuration,
        identity_binding=identity_binding,
        resolved_resonance_ids=resolved_resonance_ids,
        source_observations=observations,
        fluorescence_provenance=fluorescence_provenance,
        source_frequency_overhead_s=source_frequency_overhead_s,
        source_frequency_min_hz=first_observation.frequency_hz,
        source_frequency_max_hz=last_observation.frequency_hz,
        source_first_sequence_index=first_observation.sequence_index,
        source_last_sequence_index=last_observation.sequence_index,
        source_start_timestamp_s=source_start_timestamp_s,
        source_first_timestamp_s=first_observation.timestamp_s,
        source_last_timestamp_s=last_observation.timestamp_s,
        physical_fit_epoch_s=physical_fit_epoch_s,
        availability_sequence_index=availability_sequence_index,
        availability_timestamp_s=availability_timestamp_s,
        safe_resources=resources,
        clock_mapping=clock_mapping,
    )
