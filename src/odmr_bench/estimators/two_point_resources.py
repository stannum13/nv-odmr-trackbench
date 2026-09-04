"""Estimator-safe atomic resource accounting for two-point traces."""

from __future__ import annotations

from collections.abc import Sequence

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators.two_point_types import PublicAcquisitionResources


def _zero_public_resources() -> PublicAcquisitionResources:
    return PublicAcquisitionResources(0, 0.0, 0.0, 0, 0, 0.0)


def _advance_public_resources(
    resources: PublicAcquisitionResources,
    observation: EstimatorObservation,
    overhead_s: float,
) -> PublicAcquisitionResources:
    return PublicAcquisitionResources(
        observations=resources.observations + 1,
        integration_time_s=(
            resources.integration_time_s + observation.integration_time_s
        ),
        nominal_exposure_photons=(
            resources.nominal_exposure_photons
            + observation.nominal_exposure_photons
        ),
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


def _replay_public_resources(
    observations: Sequence[EstimatorObservation],
    overhead_s: float,
) -> PublicAcquisitionResources:
    resources = _zero_public_resources()
    for observation in observations:
        resources = _advance_public_resources(resources, observation, overhead_s)
    return resources
