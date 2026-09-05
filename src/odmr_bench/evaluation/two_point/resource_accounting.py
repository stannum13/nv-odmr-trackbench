"""Evaluator-private full resource accounting for two-point traces."""

from __future__ import annotations

from collections.abc import Sequence

from odmr_bench.emulator.observations import InstrumentObservation
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators.two_point_types import PublicAcquisitionResources
from odmr_bench.evaluation.two_point.types import ResourceJoinMismatchField


def _zero_full_resources() -> ResourceSnapshot:
    return ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)


def _advance_full_resources(
    resources: ResourceSnapshot,
    observation: InstrumentObservation,
    overhead_s: float,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        observations=resources.observations + 1,
        integration_time_s=(
            resources.integration_time_s + observation.integration_time_s
        ),
        nominal_exposure_photons=(
            resources.nominal_exposure_photons
            + observation.nominal_exposure_photons
        ),
        expected_photons=resources.expected_photons + observation.expected_photons,
        realized_photons=(
            resources.realized_photons
            + (
                observation.realized_photons
                if observation.realized_photons is not None
                else 0
            )
        ),
        observations_without_realized_counts=(
            resources.observations_without_realized_counts
            + int(observation.realized_photons is None)
        ),
        virtual_elapsed_time_s=(
            resources.virtual_elapsed_time_s
            + (overhead_s + observation.integration_time_s)
        ),
    )


def _replay_full_resources(
    observations: Sequence[InstrumentObservation],
    overhead_s: float,
) -> ResourceSnapshot:
    resources = _zero_full_resources()
    for observation in observations:
        resources = _advance_full_resources(resources, observation, overhead_s)
    return resources


def _project_full_resources(
    resources: ResourceSnapshot,
) -> PublicAcquisitionResources:
    return PublicAcquisitionResources(
        observations=resources.observations,
        integration_time_s=resources.integration_time_s,
        nominal_exposure_photons=resources.nominal_exposure_photons,
        realized_photons=resources.realized_photons,
        observations_without_realized_counts=(
            resources.observations_without_realized_counts
        ),
        virtual_elapsed_time_s=resources.virtual_elapsed_time_s,
    )


def _resource_mismatch_fields(
    expected: ResourceSnapshot,
    actual: ResourceSnapshot,
) -> tuple[ResourceJoinMismatchField, ...]:
    mismatches: list[ResourceJoinMismatchField] = []
    if expected.observations != actual.observations:
        mismatches.append("observations")
    if expected.integration_time_s != actual.integration_time_s:
        mismatches.append("integration_time_s")
    if expected.nominal_exposure_photons != actual.nominal_exposure_photons:
        mismatches.append("nominal_exposure_photons")
    if expected.expected_photons != actual.expected_photons:
        mismatches.append("expected_photons")
    if expected.realized_photons != actual.realized_photons:
        mismatches.append("realized_photons")
    if (
        expected.observations_without_realized_counts
        != actual.observations_without_realized_counts
    ):
        mismatches.append("observations_without_realized_counts")
    if expected.virtual_elapsed_time_s != actual.virtual_elapsed_time_s:
        mismatches.append("virtual_elapsed_time_s")
    return tuple(mismatches)
