"""Tests for calibrated two-point tracker reset and pair reservation."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from concurrent.futures import Future
from copy import copy
from dataclasses import dataclass, fields, is_dataclass, replace

import numpy as np
import pytest

from odmr_bench.dynamics import SpectralSnapshot
from odmr_bench.emulator.observations import (
    EstimatorObservation,
    InstrumentObservation,
)
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators import (
    CalibratedTwoPointTracker,
    NormalizedFluorescenceProvenance,
    PublicAcquisitionResources,
    TwoPointBudgetCeiling,
    TwoPointCalibration,
    TwoPointClockMapping,
    TwoPointIdentityBinding,
    TwoPointObservationValidationError,
    TwoPointRunMetadata,
    TwoPointTrackerConfiguration,
    TwoPointUpdate,
    bind_caller_asserted_two_point_calibration_source,
    calibrate_two_point,
)
from odmr_bench.estimators import two_point_tracker as tracker_module
from odmr_bench.estimators.two_point_calibration import (
    _evaluate_target_only_model,
    _target_center_derivative,
)
from odmr_bench.models import Baseline
from tests.two_point_helpers import (
    make_legal_caller_asserted_source,
    make_legal_fit_configuration,
    make_legal_source_fit,
    make_legal_tracker_configuration,
)


def test_tracker_constructor_and_pre_reset_surface() -> None:
    configuration = make_legal_tracker_configuration()
    tracker = CalibratedTwoPointTracker(configuration)

    assert tracker.configuration is configuration
    assert tracker.calibration is None
    assert tracker.pending_query is None
    assert tracker.pair_history == ()
    assert CalibratedTwoPointTracker.__final__ is True
    assert not hasattr(tracker, "__dict__")

    with pytest.raises(RuntimeError, match="reset"):
        tracker.choose_next_query()
    with pytest.raises(RuntimeError, match="reset"):
        tracker.estimate()


def test_update_surface_and_pre_reset_guard() -> None:
    tracker = CalibratedTwoPointTracker(make_legal_tracker_configuration())

    signature = inspect.signature(CalibratedTwoPointTracker.update)
    assert tuple(signature.parameters) == ("self", "observation")
    assert signature.parameters["observation"].annotation == "EstimatorObservation"
    assert signature.return_annotation == "TwoPointUpdate"

    observation = EstimatorObservation(0, 0.1, 2.8e9, 1.0, 0.1, 1.0)
    with pytest.raises(TwoPointObservationValidationError) as caught:
        tracker.update(observation)
    assert caught.value.code == "no_pending_query"
    assert caught.value.message
    assert str(caught.value) == caught.value.message
    assert caught.value.args == (caught.value.message,)


def _make_calibration(*, included: bool, offset_s: float = 0.0):
    configuration = make_legal_tracker_configuration()
    source = make_legal_caller_asserted_source()
    if offset_s != 0.0:
        source = replace(
            source,
            clock_mapping=TwoPointClockMapping(
                "unit_scale_offset",
                "source-clock",
                "tracker-clock",
                1.0,
                offset_s,
            ),
        )
    if included:
        object.__setattr__(
            source, "provenance", "verified_factory_acquisition"
        )
    calibration = calibrate_two_point(
        source,
        configuration,
        budget_treatment=(
            "included_same_run"
            if included
            else "conditional_free_precalibration"
        ),
    )
    return configuration, calibration


def test_reset_builds_calibrated_cells_and_mode_specific_resource_start() -> None:
    zero = PublicAcquisitionResources(0, 0.0, 0.0, 0, 0, 0.0)

    included_configuration, included_calibration = _make_calibration(included=True)
    included_tracker = CalibratedTwoPointTracker(included_configuration)
    included_metadata = TwoPointRunMetadata(
        tracker_clock_id="clock",
        current_sequence_index=1,
        current_timestamp_s=0.010,
        nominal_photon_rate_hz=2.5e6,
        frequency_overhead_s=0.0,
        fluorescence_quantity="normalized_fluorescence",
    )
    with pytest.raises(TypeError, match="seed"):
        included_tracker.reset(
            included_metadata,
            included_calibration,
            TwoPointBudgetCeiling(100, None, None, None),
        )
    included_tracker.reset(
        included_metadata,
        included_calibration,
        TwoPointBudgetCeiling(100, None, None, None),
        seed=np.int64(7),
    )
    included = included_tracker.estimate()

    assert included_tracker.configuration is included_configuration
    assert included_tracker.calibration is included_calibration
    assert included.current_sequence_index == 1
    assert included.current_timestamp_s == 0.010
    assert included.calibration_resources == included_calibration.source.safe_resources
    assert included.charged_resources == included_calibration.source.safe_resources
    assert included.tracking_resources == zero
    assert included.seed == 7
    assert type(included.seed) is int
    assert included.accepted_observations == included.completed_pairs == 0
    assert included.incomplete_pair is None
    assert included.pending_query is None
    assert included.pair_history == ()
    assert included.stopped_reason is None
    for identity, cell in zip(
        included.identities, included_calibration.identities, strict=True
    ):
        assert identity.resonance_id == cell.resonance_id
        assert identity.center_hz == cell.calibration_center_hz
        assert identity.calibration_fwhm_hz == cell.calibration_fwhm_hz
        assert identity.calibration_cell_lower_hz == cell.calibration_cell_lower_hz
        assert identity.calibration_cell_upper_hz == cell.calibration_cell_upper_hz
        assert identity.allowed_center_min_hz == cell.allowed_center_min_hz
        assert identity.allowed_center_max_hz == cell.allowed_center_max_hz
        assert (
            identity.active_reference_timestamp_s
            == included_calibration.source.physical_fit_epoch_s
        )
        assert identity.active_release_sequence_index == 1
        assert identity.active_release_timestamp_s == 0.010
        assert identity.estimate_age_sequence_indices == 0
        assert (
            identity.estimate_age_s
            == 0.010 - included_calibration.source.physical_fit_epoch_s
        )
        assert identity.release_age_s == 0.0
        assert identity.completed_pairs == 0
        assert identity.lock_state == "calibrated"
        assert identity.failure_code is None
        assert identity.latest_pair is None

    conditional_configuration, conditional_calibration = _make_calibration(
        included=False, offset_s=-0.010
    )
    conditional_tracker = CalibratedTwoPointTracker(conditional_configuration)
    conditional_metadata = TwoPointRunMetadata(
        tracker_clock_id="tracker-clock",
        current_sequence_index=None,
        current_timestamp_s=0.020,
        nominal_photon_rate_hz=3.0e6,
        frequency_overhead_s=0.001,
        fluorescence_quantity="normalized_fluorescence",
    )
    conditional_tracker.reset(
        conditional_metadata,
        conditional_calibration,
        TwoPointBudgetCeiling(None, 1.0, None, None),
        seed=11,
    )
    conditional = conditional_tracker.estimate()

    assert conditional.current_sequence_index is None
    assert (
        conditional.calibration_resources
        == conditional_calibration.source.safe_resources
    )
    assert conditional.charged_resources == zero
    assert conditional.tracking_resources == zero
    for identity in conditional.identities:
        expected_reference_s = (
            conditional_calibration.source.physical_fit_epoch_s - 0.010
        )
        assert identity.active_reference_timestamp_s == expected_reference_s
        assert identity.active_reference_timestamp_s < 0.0
        assert identity.active_release_sequence_index is None
        assert identity.active_release_timestamp_s == 0.0
        assert identity.estimate_age_sequence_indices is None
        assert identity.estimate_age_s == 0.020 - expected_reference_s
        assert identity.release_age_s == 0.020


def _tracker_snapshot(tracker: CalibratedTwoPointTracker) -> tuple[object, ...]:
    return (
        tracker.configuration,
        tracker.calibration,
        tracker.pending_query,
        tracker.pair_history,
        tracker.estimate(),
    )


def _calibration_with_source_id(calibration, source_id: str):
    source = copy(calibration.source)
    object.__setattr__(source, "source_id", source_id)
    return replace(calibration, source=source)


def test_invalid_reset_is_atomic_and_uses_exact_join_precedence() -> None:
    configuration, calibration = _make_calibration(included=True)
    tracker = CalibratedTwoPointTracker(configuration)
    metadata = TwoPointRunMetadata(
        tracker_clock_id="clock",
        current_sequence_index=1,
        current_timestamp_s=0.010,
        nominal_photon_rate_hz=2.5e6,
        frequency_overhead_s=0.0,
        fluorescence_quantity="normalized_fluorescence",
    )
    ceiling = TwoPointBudgetCeiling(100, 1.0, 1.0e9, 1.0)
    tracker.reset(metadata, calibration, ceiling, seed=7)

    mismatched_configuration = replace(configuration, integration_time_s=0.006)
    mismatched_calibration = replace(
        calibration, configuration=mismatched_configuration
    )
    distinct_calibration = _calibration_with_source_id(calibration, "source-2")
    offset_source = copy(calibration.source)
    object.__setattr__(offset_source, "source_id", "offset-source")
    object.__setattr__(
        offset_source,
        "clock_mapping",
        TwoPointClockMapping(
            "unit_scale_offset", "source-clock", "clock", 1.0, -0.020
        ),
    )
    offset_calibration = replace(calibration, source=offset_source)
    rows = (
        (
            metadata,
            mismatched_calibration,
            TwoPointBudgetCeiling(1, None, None, None),
            -1,
            ValueError,
            "configuration",
        ),
        (
            replace(
                metadata,
                tracker_clock_id="wrong-clock",
                current_timestamp_s=0.0,
                nominal_photon_rate_hz=3.0e6,
                current_sequence_index=2,
            ),
            distinct_calibration,
            TwoPointBudgetCeiling(1, None, None, None),
            -1,
            ValueError,
            "clock",
        ),
        (
            replace(
                metadata,
                nominal_photon_rate_hz=3.0e6,
                current_sequence_index=2,
            ),
            offset_calibration,
            TwoPointBudgetCeiling(1, None, None, None),
            -1,
            ValueError,
            "clock",
        ),
        (
            replace(
                metadata,
                current_timestamp_s=0.009,
                nominal_photon_rate_hz=3.0e6,
                current_sequence_index=2,
            ),
            distinct_calibration,
            TwoPointBudgetCeiling(1, None, None, None),
            -1,
            ValueError,
            "availability",
        ),
        (
            replace(metadata, nominal_photon_rate_hz=3.0e6, current_sequence_index=2),
            distinct_calibration,
            TwoPointBudgetCeiling(1, None, None, None),
            -1,
            ValueError,
            "metadata resources",
        ),
        (
            replace(metadata, frequency_overhead_s=0.001, current_sequence_index=2),
            distinct_calibration,
            TwoPointBudgetCeiling(1, None, None, None),
            -1,
            ValueError,
            "metadata resources",
        ),
        (
            replace(metadata, current_sequence_index=2),
            distinct_calibration,
            TwoPointBudgetCeiling(1, None, None, None),
            -1,
            ValueError,
            "boundary",
        ),
        (
            metadata,
            distinct_calibration,
            TwoPointBudgetCeiling(1, None, None, None),
            -1,
            ValueError,
            "ceiling",
        ),
        (
            metadata,
            distinct_calibration,
            ceiling,
            True,
            TypeError,
            "seed",
        ),
        (
            metadata,
            distinct_calibration,
            ceiling,
            -1,
            ValueError,
            "seed",
        ),
    )

    for bad_metadata, bad_calibration, bad_ceiling, bad_seed, error, match in rows:
        before = _tracker_snapshot(tracker)
        with pytest.raises(error, match=match):
            tracker.reset(
                bad_metadata,
                bad_calibration,
                bad_ceiling,
                seed=bad_seed,
            )
        assert _tracker_snapshot(tracker) == before


def _make_rounding_witness_calibration():
    fit_configuration = make_legal_fit_configuration()
    source_fit = make_legal_source_fit(fit_configuration)
    frequencies = (2.74e9, 2.75e9, 2.76e9, 2.90e9, 3.00e9, 3.02e9)
    source_observations = []
    endpoint_s = 0.0
    for index, frequency_hz in enumerate(frequencies):
        endpoint_s = (endpoint_s + 0.1) + 0.1
        source_observations.append(
            EstimatorObservation(
                index,
                endpoint_s,
                frequency_hz,
                1.0,
                0.1,
                0.1,
            )
        )
    first_midpoint_s = source_observations[0].timestamp_s - 0.1 / 2.0
    last_midpoint_s = source_observations[-1].timestamp_s - 0.1 / 2.0
    physical_fit_epoch_s = first_midpoint_s + (
        last_midpoint_s - first_midpoint_s
    ) / 2.0
    source = bind_caller_asserted_two_point_calibration_source(
        source_fit,
        fit_configuration,
        source_observations,
        TwoPointIdentityBinding(
            "require_expected_ids", tuple(f"r{index}" for index in range(8))
        ),
        NormalizedFluorescenceProvenance(
            "normalized_fluorescence", "declared", 1.0, ("declared",)
        ),
        source_id="rounding-witness",
        source_frequency_overhead_s=0.1,
        source_start_timestamp_s=0.0,
        physical_fit_epoch_s=physical_fit_epoch_s,
        availability_sequence_index=5,
        availability_timestamp_s=endpoint_s,
        clock_mapping=TwoPointClockMapping(
            "shared_clock", "clock", "clock", 1.0, 0.0
        ),
    )
    object.__setattr__(source, "provenance", "verified_factory_acquisition")
    configuration = TwoPointTrackerConfiguration(integration_time_s=0.1)
    calibration = calibrate_two_point(
        source,
        configuration,
        budget_treatment="included_same_run",
    )
    metadata = TwoPointRunMetadata(
        tracker_clock_id="clock",
        current_sequence_index=5,
        current_timestamp_s=endpoint_s,
        nominal_photon_rate_hz=1.0,
        frequency_overhead_s=0.1,
        fluorescence_quantity="normalized_fluorescence",
    )
    return configuration, calibration, metadata


def _reset_rounding_witness(ceiling: TwoPointBudgetCeiling):
    configuration, calibration, metadata = _make_rounding_witness_calibration()
    tracker = CalibratedTwoPointTracker(configuration)
    tracker.reset(metadata, calibration, ceiling, seed=19)
    return tracker, calibration, metadata


class _EstimatorObservationSubclass(EstimatorObservation):
    pass


def _clear_pending_query(
    tracker: CalibratedTwoPointTracker, observation: EstimatorObservation
) -> EstimatorObservation:
    state = tracker._state
    assert state is not None
    estimate = replace(state.estimate, pending_query=None)
    object.__setattr__(
        tracker,
        "_state",
        replace(state, pending_query=None, estimate=estimate),
    )
    return observation


def _inexact_observation_without_pending(
    tracker: CalibratedTwoPointTracker, observation: EstimatorObservation
) -> EstimatorObservation:
    observation = _clear_pending_query(tracker, observation)
    return _EstimatorObservationSubclass(
        observation.sequence_index,
        observation.timestamp_s,
        observation.frequency_hz,
        observation.fluorescence,
        observation.integration_time_s,
        observation.nominal_exposure_photons,
        observation.realized_photons,
    )


def _no_pending_and_sequence_mismatch(
    tracker: CalibratedTwoPointTracker, observation: EstimatorObservation
) -> EstimatorObservation:
    observation = _clear_pending_query(tracker, observation)
    return replace(
        observation,
        sequence_index=observation.sequence_index + 1,
    )


def _invalid_observation_value(
    tracker: CalibratedTwoPointTracker, observation: EstimatorObservation
) -> EstimatorObservation:
    del tracker
    object.__setattr__(observation, "fluorescence", float("nan"))
    object.__setattr__(observation, "realized_photons", -1)
    return observation


def _nominal_and_value_mismatch(
    tracker: CalibratedTwoPointTracker, observation: EstimatorObservation
) -> EstimatorObservation:
    del tracker
    changed = replace(
        observation,
        nominal_exposure_photons=observation.nominal_exposure_photons + 1.0,
    )
    object.__setattr__(changed, "fluorescence", float("nan"))
    return changed


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        (_inexact_observation_without_pending, "invalid_observation_type"),
        (_no_pending_and_sequence_mismatch, "no_pending_query"),
        (
            lambda tracker, observation: replace(
                observation,
                sequence_index=observation.sequence_index + 1,
                frequency_hz=observation.frequency_hz + 1.0,
            ),
            "sequence_mismatch",
        ),
        (
            lambda tracker, observation: replace(
                observation,
                frequency_hz=observation.frequency_hz + 1.0,
                integration_time_s=observation.integration_time_s * 2.0,
            ),
            "frequency_mismatch",
        ),
        (
            lambda tracker, observation: replace(
                observation,
                integration_time_s=observation.integration_time_s * 2.0,
                timestamp_s=observation.timestamp_s + 1.0,
            ),
            "integration_time_mismatch",
        ),
        (
            lambda tracker, observation: replace(
                observation,
                timestamp_s=observation.timestamp_s + 1.0,
                nominal_exposure_photons=(
                    observation.nominal_exposure_photons + 1.0
                ),
            ),
            "endpoint_mismatch",
        ),
        (_nominal_and_value_mismatch, "nominal_exposure_mismatch"),
        (_invalid_observation_value, "invalid_observation_value"),
    ),
)
def test_update_validation_code_precedence_is_exact_and_atomic(
    mutation, expected_code: str
) -> None:
    tracker, calibration, _ = _reset_rounding_witness(
        TwoPointBudgetCeiling(None, 100.0, None, None)
    )
    query = tracker.choose_next_query()
    assert query is not None
    observation = EstimatorObservation(
        sequence_index=query.expected_sequence_index,
        timestamp_s=query.expected_end_timestamp_s,
        frequency_hz=query.frequency_hz,
        fluorescence=0.9,
        integration_time_s=query.integration_time_s,
        nominal_exposure_photons=query.expected_nominal_exposure_photons,
        realized_photons=7,
    )
    observation = mutation(tracker, observation)
    before = tracker.estimate()
    with pytest.raises(TwoPointObservationValidationError) as caught:
        tracker.update(observation)
    assert caught.value.code == expected_code
    assert caught.value.message
    assert tracker.estimate() == before
    assert tracker.calibration is calibration


def test_first_side_commits_one_atom_and_exact_partial_pair() -> None:
    tracker, _, metadata = _reset_rounding_witness(
        TwoPointBudgetCeiling(None, 100.0, None, None)
    )
    query = tracker.choose_next_query()
    assert query is not None
    before = tracker.estimate()
    observation = EstimatorObservation(
        query.expected_sequence_index,
        query.expected_end_timestamp_s,
        query.frequency_hz,
        0.9,
        query.integration_time_s,
        query.expected_nominal_exposure_photons,
        7,
    )

    update = tracker.update(observation)

    assert type(update) is TwoPointUpdate
    assert update.query is query
    assert update.observation is observation
    assert update.completed_pair is None
    after = update.estimate
    assert tracker.estimate() is after
    assert tracker.pending_query is None
    assert after.pending_query is None
    partial = after.incomplete_pair
    assert partial is not None
    assert partial.pair_index == query.pair_index
    assert partial.identity_pair_index == query.identity_pair_index
    assert partial.resonance_id == query.resonance_id
    assert partial.interrogation_center_hz == query.interrogation_center_hz
    assert partial.first_side == query.side
    assert partial.first_query is query
    assert partial.first_observation is observation
    assert after.accepted_observations == before.accepted_observations + 1
    assert after.current_sequence_index == observation.sequence_index
    assert after.current_timestamp_s == observation.timestamp_s
    assert after.tracking_resources == PublicAcquisitionResources(
        1,
        observation.integration_time_s,
        observation.nominal_exposure_photons,
        7,
        0,
        metadata.frequency_overhead_s + observation.integration_time_s,
    )
    charged_before = before.charged_resources
    assert after.charged_resources == PublicAcquisitionResources(
        charged_before.observations + 1,
        charged_before.integration_time_s + observation.integration_time_s,
        charged_before.nominal_exposure_photons
        + observation.nominal_exposure_photons,
        charged_before.realized_photons + 7,
        charged_before.observations_without_realized_counts,
        charged_before.virtual_elapsed_time_s
        + (metadata.frequency_overhead_s + observation.integration_time_s),
    )
    for identity_after, identity_before in zip(
        after.identities, before.identities, strict=True
    ):
        assert identity_after.center_hz == identity_before.center_hz
        assert identity_after.completed_pairs == identity_before.completed_pairs == 0
        assert identity_after.lock_state == identity_before.lock_state == "calibrated"
        assert identity_after.active_source_kind == identity_before.active_source_kind
        assert (
            identity_after.active_reference_timestamp_s
            == identity_before.active_reference_timestamp_s
        )
        assert (
            identity_after.active_release_sequence_index
            == identity_before.active_release_sequence_index
        )
        assert identity_after.estimate_age_s == (
            observation.timestamp_s - identity_after.active_reference_timestamp_s
        )
        assert identity_after.release_age_s == (
            observation.timestamp_s - identity_after.active_release_timestamp_s
        )
        assert identity_after.estimate_age_sequence_indices == (
            observation.sequence_index
            - identity_after.active_release_sequence_index
        )
    assert after.completed_pairs == before.completed_pairs == 0
    assert after.pair_history == before.pair_history == ()


def test_invalid_reset_rejects_nonrepresentable_first_pair_atomically() -> None:
    source = make_legal_caller_asserted_source()
    rows = (
        (
            TwoPointTrackerConfiguration(integration_time_s=2.0),
            TwoPointRunMetadata(
                "clock",
                None,
                0.010,
                1.0e308,
                0.0,
                "normalized_fluorescence",
            ),
            "nominal exposure",
        ),
        (
            TwoPointTrackerConfiguration(integration_time_s=8.0e307),
            TwoPointRunMetadata(
                "clock",
                None,
                0.010,
                1.0e-308,
                1.0e308,
                "normalized_fluorescence",
            ),
            "elapsed",
        ),
        (
            TwoPointTrackerConfiguration(integration_time_s=8.0e307),
            TwoPointRunMetadata(
                "clock",
                None,
                1.0e308,
                1.0e-308,
                0.0,
                "normalized_fluorescence",
            ),
            "endpoint",
        ),
        (
            TwoPointTrackerConfiguration(integration_time_s=1.0),
            TwoPointRunMetadata(
                "clock",
                None,
                0.010,
                1.0e308,
                0.0,
                "normalized_fluorescence",
            ),
            "charged resources",
        ),
    )

    for configuration, metadata, match in rows:
        calibration = calibrate_two_point(
            source,
            configuration,
            budget_treatment="conditional_free_precalibration",
        )
        tracker = CalibratedTwoPointTracker(configuration)
        tracker.reset(
            TwoPointRunMetadata(
                "clock",
                None,
                0.010,
                1.0e-308 if configuration.integration_time_s > 2.0 else 1.0,
                0.0,
                "normalized_fluorescence",
            ),
            calibration,
            TwoPointBudgetCeiling(2, None, None, None),
            seed=17,
        )
        assert tracker.choose_next_query() is not None
        before = _tracker_snapshot(tracker)
        with pytest.raises(ValueError, match=match):
            tracker.reset(
                metadata,
                calibration,
                TwoPointBudgetCeiling(2, None, None, None),
                seed=23,
            )
        assert _tracker_snapshot(tracker) == before


@pytest.mark.parametrize(
    ("configuration", "metadata", "failure_kind"),
    (
        (
            TwoPointTrackerConfiguration(integration_time_s=2.0**-53),
            TwoPointRunMetadata(
                "clock",
                None,
                np.nextafter(1.0, -np.inf),
                1.0,
                0.0,
                "normalized_fluorescence",
            ),
            "nonadvancing",
        ),
        (
            TwoPointTrackerConfiguration(integration_time_s=1.0e307),
            TwoPointRunMetadata(
                "clock",
                None,
                1.0e308,
                1.0e-308,
                3.0e307,
                "normalized_fluorescence",
            ),
            "overflowing",
        ),
    ),
)
def test_invalid_reset_rejects_nonadvancing_or_overflowing_second_endpoint_atomically(
    configuration: TwoPointTrackerConfiguration,
    metadata: TwoPointRunMetadata,
    failure_kind: str,
) -> None:
    source = make_legal_caller_asserted_source()
    calibration = calibrate_two_point(
        source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    tracker = CalibratedTwoPointTracker(configuration)
    tracker.reset(
        TwoPointRunMetadata(
            "clock",
            None,
            0.010,
            metadata.nominal_photon_rate_hz,
            0.0,
            "normalized_fluorescence",
        ),
        calibration,
        TwoPointBudgetCeiling(2, None, None, None),
        seed=29,
    )
    assert tracker.choose_next_query() is not None
    before = _tracker_snapshot(tracker)

    first_endpoint_s = (
        metadata.current_timestamp_s + metadata.frequency_overhead_s
    ) + configuration.integration_time_s
    second_endpoint_s = (
        first_endpoint_s + metadata.frequency_overhead_s
    ) + configuration.integration_time_s
    assert math.isfinite(first_endpoint_s)
    if failure_kind == "nonadvancing":
        assert first_endpoint_s == 1.0
        assert second_endpoint_s == first_endpoint_s
    else:
        assert first_endpoint_s == 1.4e308
        assert not math.isfinite(second_endpoint_s)

    with pytest.raises(ValueError, match="second-query endpoint"):
        tracker.reset(
            metadata,
            calibration,
            TwoPointBudgetCeiling(2, None, None, None),
            seed=31,
        )
    assert _tracker_snapshot(tracker) == before


def test_invalid_reset_rejects_nonadvancing_first_endpoint_atomically() -> None:
    configuration = TwoPointTrackerConfiguration(integration_time_s=2.0**-54)
    calibration = calibrate_two_point(
        make_legal_caller_asserted_source(),
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    tracker = CalibratedTwoPointTracker(configuration)
    tracker.reset(
        TwoPointRunMetadata(
            "clock", None, 0.010, 1.0, 0.0, "normalized_fluorescence"
        ),
        calibration,
        TwoPointBudgetCeiling(2, None, None, None),
        seed=37,
    )
    assert tracker.choose_next_query() is not None
    before = _tracker_snapshot(tracker)
    metadata = TwoPointRunMetadata(
        "clock", None, 1.0, 1.0, 0.0, "normalized_fluorescence"
    )
    first_endpoint_s = (
        metadata.current_timestamp_s + metadata.frequency_overhead_s
    ) + configuration.integration_time_s
    assert first_endpoint_s == metadata.current_timestamp_s

    with pytest.raises(ValueError, match="first-query endpoint"):
        tracker.reset(
            metadata,
            calibration,
            TwoPointBudgetCeiling(2, None, None, None),
            seed=41,
        )
    assert _tracker_snapshot(tracker) == before


def test_choose_first_query_reserves_two_atomic_charges_and_is_idempotent() -> None:
    configuration, calibration, metadata = _make_rounding_witness_calibration()
    start = calibration.source.safe_resources
    integration_after_two = (
        start.integration_time_s + configuration.integration_time_s
    ) + configuration.integration_time_s
    nominal_atom = (
        metadata.nominal_photon_rate_hz * configuration.integration_time_s
    )
    nominal_after_two = (
        start.nominal_exposure_photons + nominal_atom
    ) + nominal_atom
    elapsed_atom = metadata.frequency_overhead_s + configuration.integration_time_s
    elapsed_after_two = (
        start.virtual_elapsed_time_s + elapsed_atom
    ) + elapsed_atom
    assert integration_after_two != (
        start.integration_time_s + 2.0 * configuration.integration_time_s
    )
    assert nominal_after_two != start.nominal_exposure_photons + 2.0 * nominal_atom
    assert elapsed_after_two != start.virtual_elapsed_time_s + 2.0 * elapsed_atom

    dimension_cases = (
        (
            "observations",
            8,
            7,
            9,
        ),
        (
            "integration",
            integration_after_two,
            np.nextafter(integration_after_two, -np.inf),
            np.nextafter(integration_after_two, np.inf),
        ),
        (
            "nominal",
            nominal_after_two,
            np.nextafter(nominal_after_two, -np.inf),
            np.nextafter(nominal_after_two, np.inf),
        ),
        (
            "elapsed",
            elapsed_after_two,
            np.nextafter(elapsed_after_two, -np.inf),
            np.nextafter(elapsed_after_two, np.inf),
        ),
    )
    for dimension, exact, below, above in dimension_cases:
        for cap, affordable in ((below, False), (exact, True), (above, True)):
            ceiling = TwoPointBudgetCeiling(
                max_observations=cap if dimension == "observations" else None,
                max_integration_time_s=cap if dimension == "integration" else None,
                max_nominal_exposure_photons=(
                    cap if dimension == "nominal" else None
                ),
                max_virtual_elapsed_time_s=cap if dimension == "elapsed" else None,
            )
            tracker, _, _ = _reset_rounding_witness(ceiling)
            query = tracker.choose_next_query()
            assert (query is not None) is affordable

    ceiling = TwoPointBudgetCeiling(
        8,
        integration_after_two,
        nominal_after_two,
        elapsed_after_two,
    )
    tracker, calibration, metadata = _reset_rounding_witness(ceiling)
    before = tracker.estimate()
    query = tracker.choose_next_query()
    assert query is not None
    assert query.query_index == 0
    assert query.pair_index == 0
    assert query.identity_pair_index == 0
    assert query.resonance_id == "r0"
    assert query.side == "minus"
    assert (
        query.interrogation_center_hz
        == calibration.identities[0].calibration_center_hz
    )
    assert query.frequency_hz == (
        query.interrogation_center_hz - calibration.identities[0].offset_hz
    )
    assert query.integration_time_s == 0.1
    assert query.expected_sequence_index == 6
    assert query.expected_end_timestamp_s == (
        metadata.current_timestamp_s + metadata.frequency_overhead_s
    ) + configuration.integration_time_s
    assert query.expected_nominal_exposure_photons == 0.1

    after = tracker.estimate()
    assert after.pending_query is query
    assert tracker.pending_query is query
    assert replace(after, pending_query=None) == before
    repeated = tracker.choose_next_query()
    assert repeated is query
    assert tracker.estimate() is after


def test_choose_reserved_second_query_without_budget_recheck_or_center_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration, calibration, metadata = _make_rounding_witness_calibration()
    exact_cap = calibration.source.safe_resources.observations + 2
    tracker, _, _ = _reset_rounding_witness(
        TwoPointBudgetCeiling(exact_cap, None, None, None)
    )
    first_query = tracker.choose_next_query()
    assert first_query is not None
    frozen_center_hz = first_query.interrogation_center_hz
    first_observation = EstimatorObservation(
        first_query.expected_sequence_index,
        first_query.expected_end_timestamp_s,
        first_query.frequency_hz,
        0.9,
        first_query.integration_time_s,
        first_query.expected_nominal_exposure_photons,
        7,
    )
    tracker.update(first_observation)
    after_first = tracker.estimate()
    assert after_first.charged_resources.observations == exact_cap - 1
    assert after_first.identities[0].center_hz == frozen_center_hz

    def reject_affordability_recheck(*args, **kwargs):
        del args, kwargs
        raise AssertionError("reserved second query must not recheck affordability")

    monkeypatch.setattr(
        tracker_module, "_within_ceiling", reject_affordability_recheck
    )
    second_query = tracker.choose_next_query()

    assert second_query is not None
    assert second_query.query_index == 1
    assert second_query.pair_index == 0
    assert second_query.identity_pair_index == 0
    assert second_query.resonance_id == first_query.resonance_id == "r0"
    assert second_query.side == "plus"
    assert second_query.interrogation_center_hz == frozen_center_hz
    assert second_query.frequency_hz == (
        frozen_center_hz + calibration.identities[0].offset_hz
    )
    assert second_query.integration_time_s == configuration.integration_time_s
    assert second_query.expected_sequence_index == first_observation.sequence_index + 1
    assert second_query.expected_end_timestamp_s == (
        first_observation.timestamp_s + metadata.frequency_overhead_s
    ) + configuration.integration_time_s
    assert (
        second_query.expected_nominal_exposure_photons
        == metadata.nominal_photon_rate_hz * configuration.integration_time_s
    )
    after_second = tracker.estimate()
    assert after_second.incomplete_pair is after_first.incomplete_pair
    assert replace(after_second, pending_query=None) == after_first
    assert tracker.choose_next_query() is second_query
    assert tracker.estimate() is after_second


def test_unaffordable_boundary_stops_atomically_without_partial_pair() -> None:
    configuration, calibration, metadata = _make_rounding_witness_calibration()
    tracker = CalibratedTwoPointTracker(configuration)
    tracker.reset(
        metadata,
        calibration,
        TwoPointBudgetCeiling(
            calibration.source.safe_resources.observations, None, None, None
        ),
        seed=19,
    )
    before = tracker.estimate()

    assert tracker.choose_next_query() is None
    stopped = tracker.estimate()
    assert stopped.stopped_reason == "budget_exhausted"
    assert stopped.pending_query is None
    assert stopped.incomplete_pair is None
    assert stopped.accepted_observations == 0
    assert stopped.completed_pairs == 0
    assert stopped.pair_history == ()
    assert replace(stopped, stopped_reason=None) == before

    assert tracker.choose_next_query() is None
    assert tracker.estimate() is stopped


def test_second_side_computes_signed_hertz_update_and_refreshes_zero_step_source(
) -> None:
    tracker, calibration, _ = _reset_rounding_witness(
        TwoPointBudgetCeiling(None, 100.0, None, None)
    )
    cell = calibration.identities[0]
    observations = []
    updates = []

    for expected_side in ("minus", "plus"):
        query = tracker.choose_next_query()
        assert query is not None
        assert query.side == expected_side
        fluorescence = _evaluate_target_only_model(
            calibration.source.source_fit,
            cell.source_fit_index,
            query.frequency_hz,
            query.interrogation_center_hz,
        )
        observation = EstimatorObservation(
            query.expected_sequence_index,
            query.expected_end_timestamp_s,
            query.frequency_hz,
            fluorescence,
            query.integration_time_s,
            query.expected_nominal_exposure_photons,
            None,
        )
        observations.append(observation)
        updates.append(tracker.update(observation))

    first_observation, second_observation = observations
    assert updates[0].completed_pair is None
    completed = updates[1].completed_pair
    assert completed is not None
    assert completed.minus_observation is first_observation
    assert completed.plus_observation is second_observation
    assert completed.first_side == "minus"
    first_reference_s = (
        first_observation.timestamp_s - first_observation.integration_time_s / 2.0
    )
    second_reference_s = (
        second_observation.timestamp_s
        - second_observation.integration_time_s / 2.0
    )
    assert completed.pair_reference_timestamp_s == first_reference_s + (
        second_reference_s - first_reference_s
    ) / 2.0
    assert completed.release_sequence_index == second_observation.sequence_index
    assert completed.release_timestamp_s == second_observation.timestamp_s
    assert completed.discriminator == completed.zero_discriminator
    assert completed.raw_innovation_hz == 0.0
    assert completed.requested_step_hz == 0.0
    assert completed.applied_step_hz == 0.0
    assert completed.candidate_center_hz == completed.interrogation_center_hz
    assert completed.lock_state == "tracking"
    assert completed.failure_code is None

    estimate = tracker.estimate()
    assert updates[1].estimate is estimate
    assert estimate.incomplete_pair is None
    assert estimate.pending_query is None
    assert tracker.pending_query is None
    assert estimate.pair_history == tracker.pair_history == (completed,)
    assert estimate.accepted_observations == 2
    assert estimate.completed_pairs == 1
    identity = estimate.identities[0]
    assert identity.completed_pairs == 1
    assert identity.latest_pair is completed
    assert identity.center_hz == completed.interrogation_center_hz
    assert identity.active_source_kind == "pair"
    assert identity.active_source_pair_index == 0
    assert (
        identity.active_reference_timestamp_s
        == completed.pair_reference_timestamp_s
    )
    assert identity.active_release_sequence_index == completed.release_sequence_index
    assert identity.active_release_timestamp_s == completed.release_timestamp_s
    assert identity.estimate_age_sequence_indices == 0
    assert identity.estimate_age_s == (
        completed.release_timestamp_s - completed.pair_reference_timestamp_s
    )
    assert identity.release_age_s == 0.0


@pytest.mark.parametrize(
    "budget_treatment",
    ("included_same_run", "conditional_free_precalibration"),
)
def test_second_arrival_resources_are_exact_left_associated_binary64(
    budget_treatment: str,
) -> None:
    configuration, included_calibration, metadata = (
        _make_rounding_witness_calibration()
    )
    calibration = calibrate_two_point(
        included_calibration.source,
        configuration,
        budget_treatment=budget_treatment,
    )
    tracker = CalibratedTwoPointTracker(configuration)
    tracker.reset(
        metadata,
        calibration,
        TwoPointBudgetCeiling(None, 100.0, None, None),
        seed=37,
    )
    cell = calibration.identities[0]
    for realized_photons in (7, 11):
        query = tracker.choose_next_query()
        assert query is not None
        tracker.update(
            EstimatorObservation(
                query.expected_sequence_index,
                query.expected_end_timestamp_s,
                query.frequency_hz,
                _evaluate_target_only_model(
                    calibration.source.source_fit,
                    cell.source_fit_index,
                    query.frequency_hz,
                    query.interrogation_center_hz,
                ),
                query.integration_time_s,
                query.expected_nominal_exposure_photons,
                realized_photons,
            )
        )

    atom_integration = configuration.integration_time_s
    atom_nominal = metadata.nominal_photon_rate_hz * atom_integration
    atom_elapsed = metadata.frequency_overhead_s + atom_integration
    tracking_expected = PublicAcquisitionResources(
        2,
        (0.0 + atom_integration) + atom_integration,
        (0.0 + atom_nominal) + atom_nominal,
        18,
        0,
        (0.0 + atom_elapsed) + atom_elapsed,
    )
    source = calibration.source.safe_resources
    included_expected = PublicAcquisitionResources(
        (source.observations + 1) + 1,
        (source.integration_time_s + atom_integration) + atom_integration,
        (source.nominal_exposure_photons + atom_nominal) + atom_nominal,
        (source.realized_photons + 7) + 11,
        (source.observations_without_realized_counts + 0) + 0,
        (source.virtual_elapsed_time_s + atom_elapsed) + atom_elapsed,
    )
    assert included_expected.integration_time_s != (
        source.integration_time_s + 2.0 * atom_integration
    )
    assert included_expected.nominal_exposure_photons != (
        source.nominal_exposure_photons + 2.0 * atom_nominal
    )
    assert included_expected.virtual_elapsed_time_s != (
        source.virtual_elapsed_time_s + 2.0 * atom_elapsed
    )

    estimate = tracker.estimate()
    charged_expected = (
        included_expected
        if budget_treatment == "included_same_run"
        else tracking_expected
    )
    for field in fields(PublicAcquisitionResources):
        assert getattr(estimate.tracking_resources, field.name) == getattr(
            tracking_expected, field.name
        )
        assert getattr(estimate.charged_resources, field.name) == getattr(
            charged_expected, field.name
        )


def _complete_pair_for_gate(case: str):
    base_configuration, base_calibration, metadata = (
        _make_rounding_witness_calibration()
    )
    base_cell = base_calibration.identities[0]
    source_fit = base_calibration.source.source_fit
    center_hz = base_cell.calibration_center_hz
    minus_frequency_hz = center_hz - base_cell.offset_hz
    plus_frequency_hz = center_hz + base_cell.offset_hz
    mu_minus = _evaluate_target_only_model(
        source_fit, base_cell.source_fit_index, minus_frequency_hz, center_hz
    )
    mu_plus = _evaluate_target_only_model(
        source_fit, base_cell.source_fit_index, plus_frequency_hz, center_hz
    )
    g_minus = _target_center_derivative(
        source_fit, base_cell.source_fit_index, minus_frequency_hz, center_hz
    )
    g_plus = _target_center_derivative(
        source_fit, base_cell.source_fit_index, plus_frequency_hz, center_hz
    )
    model_sum = mu_minus + mu_plus
    zero_discriminator = (mu_minus - mu_plus) / model_sum
    slope_per_hz = 2.0 * (
        mu_plus * g_minus - mu_minus * g_plus
    ) / model_sum**2

    desired_raw_hz = 0.0
    desired_common = 0.0
    proportional_gain = base_configuration.proportional_gain
    if case in {"numerical", "numerical-combined"}:
        desired_raw_hz = 2.0
        proportional_gain = 1.0e308
        if case == "numerical-combined":
            desired_common = -0.25
    elif case in {"common-strict", "common-equality"}:
        desired_common = 0.25
    elif case in {
        "common-negative-strict-combined",
        "common-negative-equality",
    }:
        desired_common = -0.25
        desired_raw_hz = -0.15 * base_cell.calibration_fwhm_hz
    elif case in {"capture-strict", "capture-equality", "domain", "tracking"}:
        desired_raw_hz = 0.05 * base_cell.calibration_fwhm_hz
    elif case in {"capture-negative-strict-combined", "capture-negative-equality"}:
        desired_raw_hz = -0.05 * base_cell.calibration_fwhm_hz
    elif case == "step-limited":
        desired_raw_hz = 0.15 * base_cell.calibration_fwhm_hz

    if case == "invalid-sum":
        minus_fluorescence = 1.0
        plus_fluorescence = -1.0
    else:
        observed_sum = model_sum + desired_common * base_cell.target_pair_depth
        desired_discriminator = (
            zero_discriminator + slope_per_hz * desired_raw_hz
        )
        minus_fluorescence = (
            observed_sum * (1.0 + desired_discriminator) / 2.0
        )
        plus_fluorescence = observed_sum - minus_fluorescence

    observed_sum = minus_fluorescence + plus_fluorescence
    actual_discriminator = (
        None
        if observed_sum <= 0.0
        else (minus_fluorescence - plus_fluorescence) / observed_sum
    )
    actual_common = (
        None
        if observed_sum <= 0.0
        else (observed_sum - model_sum) / base_cell.target_pair_depth
    )
    actual_raw_hz = (
        None
        if actual_discriminator is None
        else (actual_discriminator - zero_discriminator) / slope_per_hz
    )
    common_limit = None
    if case in {
        "common-strict",
        "common-negative-strict-combined",
        "numerical-combined",
    }:
        assert actual_common is not None
        common_limit = math.nextafter(abs(actual_common), 0.0)
    elif case in {
        "common-equality",
        "common-negative-equality",
    }:
        assert actual_common is not None
        common_limit = abs(actual_common)

    configuration = replace(
        base_configuration,
        proportional_gain=proportional_gain,
        common_mode_limit_target_depths=common_limit,
    )
    calibration = calibrate_two_point(
        base_calibration.source,
        configuration,
        budget_treatment="included_same_run",
    )
    cell = calibration.identities[0]
    if case in {
        "capture-strict",
        "capture-equality",
        "common-negative-strict-combined",
        "capture-negative-strict-combined",
        "capture-negative-equality",
        "numerical-combined",
    }:
        assert actual_raw_hz is not None
        capture_radius_hz = abs(actual_raw_hz)
        if case in {
            "capture-strict",
            "common-negative-strict-combined",
            "capture-negative-strict-combined",
            "numerical-combined",
        }:
            capture_radius_hz = math.nextafter(capture_radius_hz, 0.0)
        calibration = replace(
            calibration,
            identities=(
                replace(cell, capture_radius_hz=capture_radius_hz),
                *calibration.identities[1:],
            ),
        )
    if case in {
        "domain",
        "common-negative-strict-combined",
        "capture-negative-strict-combined",
        "numerical-combined",
    }:
        assert actual_raw_hz is not None
        requested_step_hz = (
            base_configuration.proportional_gain * actual_raw_hz
            if case == "numerical-combined"
            else configuration.proportional_gain * actual_raw_hz
        )
        applied_step_hz = max(
            -cell.max_step_hz,
            min(requested_step_hz, cell.max_step_hz),
        )
        candidate_center_hz = center_hz + applied_step_hz
        domain_changes = (
            {
                "allowed_center_min_hz": math.nextafter(
                    candidate_center_hz, math.inf
                )
            }
            if candidate_center_hz < center_hz
            else {
                "allowed_center_max_hz": math.nextafter(
                    candidate_center_hz, -math.inf
                )
            }
        )
        current_cell = calibration.identities[0]
        calibration = replace(
            calibration,
            identities=(
                replace(current_cell, **domain_changes),
                *calibration.identities[1:],
            ),
        )

    tracker = CalibratedTwoPointTracker(configuration)
    tracker.reset(
        metadata,
        calibration,
        TwoPointBudgetCeiling(None, 100.0, None, None),
        seed=43,
    )
    completed = None
    for fluorescence in (minus_fluorescence, plus_fluorescence):
        query = tracker.choose_next_query()
        assert query is not None
        update = tracker.update(
            EstimatorObservation(
                query.expected_sequence_index,
                query.expected_end_timestamp_s,
                query.frequency_hz,
                fluorescence,
                query.integration_time_s,
                query.expected_nominal_exposure_photons,
                None,
            )
        )
        completed = update.completed_pair
    assert completed is not None
    return completed, configuration, calibration.identities[0]


@pytest.mark.parametrize(
    ("case", "lock_state", "failure_code", "diagnostic_presence"),
    (
        pytest.param(
            "invalid-sum",
            "lost",
            "invalid_pair_normalization",
            (False, False, False, False, False),
            id="invalid-sum",
        ),
        pytest.param(
            "numerical",
            "lost",
            "numerical_failure",
            (True, True, True, False, False),
            id="nonfinite-derived-arithmetic",
        ),
        pytest.param(
            "numerical-combined",
            "lost",
            "numerical_failure",
            (True, True, True, False, False),
            id="numerical-precedes-common-capture-domain",
        ),
        pytest.param(
            "common-strict",
            "lost",
            "common_mode_limit_exceeded",
            (True, True, False, False, False),
            id="common-strict-exceed",
        ),
        pytest.param(
            "common-equality",
            "tracking",
            None,
            (True, True, True, True, True),
            id="common-equality-passes",
        ),
        pytest.param(
            "common-negative-strict-combined",
            "lost",
            "common_mode_limit_exceeded",
            (True, True, False, False, False),
            id="negative-common-precedes-capture-domain",
        ),
        pytest.param(
            "common-negative-equality",
            "step_limited",
            None,
            (True, True, True, True, True),
            id="negative-common-equality-passes",
        ),
        pytest.param(
            "capture-strict",
            "lost",
            "capture_exceeded",
            (True, True, True, False, False),
            id="capture-strict-exceed",
        ),
        pytest.param(
            "capture-equality",
            "tracking",
            None,
            (True, True, True, True, True),
            id="capture-equality-passes",
        ),
        pytest.param(
            "capture-negative-strict-combined",
            "lost",
            "capture_exceeded",
            (True, True, True, False, False),
            id="negative-capture-precedes-domain",
        ),
        pytest.param(
            "capture-negative-equality",
            "tracking",
            None,
            (True, True, True, True, True),
            id="negative-capture-equality-passes",
        ),
        pytest.param(
            "domain",
            "lost",
            "calibration_domain_exceeded",
            (True, True, True, True, True),
            id="domain",
        ),
        pytest.param(
            "tracking",
            "tracking",
            None,
            (True, True, True, True, True),
            id="tracking",
        ),
        pytest.param(
            "step-limited",
            "step_limited",
            None,
            (True, True, True, True, True),
            id="step-limited",
        ),
    ),
)
def test_pair_gate_precedence_and_retained_diagnostics(
    case: str,
    lock_state: str,
    failure_code: str | None,
    diagnostic_presence: tuple[bool, bool, bool, bool, bool],
) -> None:
    pair, configuration, cell = _complete_pair_for_gate(case)

    assert pair.lock_state == lock_state
    assert pair.failure_code == failure_code
    assert tuple(
        value is not None
        for value in (
            pair.discriminator,
            pair.common_mode_target_depths,
            pair.raw_innovation_hz,
            pair.requested_step_hz,
            pair.candidate_center_hz,
        )
    ) == diagnostic_presence
    if case == "invalid-sum":
        assert (pair.zero_discriminator, pair.discriminator_slope_per_hz) == (
            None,
            None,
        )
    else:
        assert pair.zero_discriminator is not None
        assert pair.discriminator_slope_per_hz is not None
        assert math.isfinite(pair.zero_discriminator)
        assert math.isfinite(pair.discriminator_slope_per_hz)
        assert pair.discriminator_slope_per_hz > 0.0
    if lock_state == "lost":
        assert pair.applied_step_hz == 0.0
        assert pair.candidate_center_hz != pair.interrogation_center_hz
    elif lock_state == "tracking":
        assert pair.applied_step_hz == pair.requested_step_hz
        assert pair.candidate_center_hz == (
            pair.interrogation_center_hz + pair.applied_step_hz
        )
    else:
        assert pair.requested_step_hz is not None
        assert pair.applied_step_hz * pair.requested_step_hz > 0.0
        assert abs(pair.applied_step_hz) < abs(pair.requested_step_hz)

    if case in {"common-strict", "common-negative-strict-combined"}:
        assert pair.common_mode_target_depths is not None
        assert configuration.common_mode_limit_target_depths is not None
        assert (
            abs(pair.common_mode_target_depths)
            > configuration.common_mode_limit_target_depths
        )
    elif case in {"common-equality", "common-negative-equality"}:
        assert pair.common_mode_target_depths is not None
        assert (
            abs(pair.common_mode_target_depths)
            == configuration.common_mode_limit_target_depths
        )
    elif case in {"capture-strict", "capture-negative-strict-combined"}:
        assert pair.raw_innovation_hz is not None
        assert abs(pair.raw_innovation_hz) > cell.capture_radius_hz
    elif case in {"capture-equality", "capture-negative-equality"}:
        assert pair.raw_innovation_hz is not None
        assert abs(pair.raw_innovation_hz) == cell.capture_radius_hz
    elif case == "domain":
        assert pair.candidate_center_hz is not None
        assert pair.candidate_center_hz > cell.allowed_center_max_hz


def test_public_schedule_advances_second_side_next_identity_and_odd_alternation(
) -> None:
    tracker, calibration, _ = _reset_rounding_witness(
        TwoPointBudgetCeiling(None, 100.0, None, None)
    )
    queries = []
    failed_pair_index = 5

    for pair_index in range(18):
        identity_index = pair_index % 8
        identity_pair_index = pair_index // 8
        first_side = "minus" if identity_pair_index % 2 == 0 else "plus"
        expected_sides = (
            (first_side, "plus" if first_side == "minus" else "minus")
        )
        completed = None
        for arrival_index, expected_side in enumerate(expected_sides):
            query = tracker.choose_next_query()
            assert query is not None
            queries.append(query)
            assert query.query_index == 2 * pair_index + arrival_index
            assert query.pair_index == pair_index
            assert query.identity_pair_index == identity_pair_index
            assert query.resonance_id == f"r{identity_index}"
            assert query.side == expected_side

            if pair_index == failed_pair_index:
                fluorescence = 1.0 if arrival_index == 0 else -1.0
            else:
                cell = calibration.identities[identity_index]
                fluorescence = _evaluate_target_only_model(
                    calibration.source.source_fit,
                    cell.source_fit_index,
                    query.frequency_hz,
                    query.interrogation_center_hz,
                )
            update = tracker.update(
                EstimatorObservation(
                    query.expected_sequence_index,
                    query.expected_end_timestamp_s,
                    query.frequency_hz,
                    fluorescence,
                    query.integration_time_s,
                    query.expected_nominal_exposure_photons,
                    None,
                )
            )
            completed = update.completed_pair
            if arrival_index == 0:
                assert completed is None

        assert completed is not None
        assert completed.pair_index == pair_index
        if pair_index == failed_pair_index:
            assert completed.lock_state == "lost"
            assert completed.failure_code == "invalid_pair_normalization"
        if pair_index < 17:
            next_query = tracker.choose_next_query()
            assert next_query is not None
            assert next_query.pair_index == pair_index + 1
            assert next_query.resonance_id == f"r{(pair_index + 1) % 8}"
            assert tracker.choose_next_query() is next_query

    assert tuple(query.resonance_id for query in queries) == tuple(
        f"r{pair_index % 8}"
        for pair_index in range(18)
        for _ in range(2)
    )
    assert all(
        second.query_index == first.query_index + 1
        and second.expected_sequence_index == first.expected_sequence_index + 1
        for first, second in zip(queries[::2], queries[1::2], strict=True)
    )
    assert tuple(query.side for query in queries[16:18]) == ("plus", "minus")
    estimate = tracker.estimate()
    assert estimate.accepted_observations == 36
    assert estimate.completed_pairs == 18
    assert len(estimate.pair_history) == 18
    assert tuple(identity.completed_pairs for identity in estimate.identities) == (
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
    )


def _pair_fluorescence_for_raw(
    calibration, identity_index: int, center_hz: float, raw_hz: float
) -> tuple[float, float, float]:
    cell = calibration.identities[identity_index]
    source_fit = calibration.source.source_fit
    minus_frequency_hz = center_hz - cell.offset_hz
    plus_frequency_hz = center_hz + cell.offset_hz
    mu_minus = _evaluate_target_only_model(
        source_fit, cell.source_fit_index, minus_frequency_hz, center_hz
    )
    mu_plus = _evaluate_target_only_model(
        source_fit, cell.source_fit_index, plus_frequency_hz, center_hz
    )
    g_minus = _target_center_derivative(
        source_fit, cell.source_fit_index, minus_frequency_hz, center_hz
    )
    g_plus = _target_center_derivative(
        source_fit, cell.source_fit_index, plus_frequency_hz, center_hz
    )
    model_sum = mu_minus + mu_plus
    zero_discriminator = (mu_minus - mu_plus) / model_sum
    slope_per_hz = 2.0 * (
        mu_plus * g_minus - mu_minus * g_plus
    ) / model_sum**2
    discriminator = zero_discriminator + slope_per_hz * raw_hz
    minus_fluorescence = model_sum * (1.0 + discriminator) / 2.0
    plus_fluorescence = model_sum - minus_fluorescence
    actual_discriminator = (
        minus_fluorescence - plus_fluorescence
    ) / (minus_fluorescence + plus_fluorescence)
    actual_raw_hz = (
        actual_discriminator - zero_discriminator
    ) / slope_per_hz
    return minus_fluorescence, plus_fluorescence, actual_raw_hz


def _complete_pair_with_raw(tracker, calibration, raw_hz: float):
    first_query = tracker.choose_next_query()
    assert first_query is not None
    identity_index = first_query.pair_index % 8
    minus_fluorescence, plus_fluorescence, _ = _pair_fluorescence_for_raw(
        calibration,
        identity_index,
        first_query.interrogation_center_hz,
        raw_hz,
    )
    first_fluorescence = (
        minus_fluorescence if first_query.side == "minus" else plus_fluorescence
    )
    first_update = tracker.update(
        EstimatorObservation(
            first_query.expected_sequence_index,
            first_query.expected_end_timestamp_s,
            first_query.frequency_hz,
            first_fluorescence,
            first_query.integration_time_s,
            first_query.expected_nominal_exposure_photons,
            None,
        )
    )
    assert first_update.completed_pair is None
    second_query = tracker.choose_next_query()
    assert second_query is not None
    second_fluorescence = (
        minus_fluorescence if second_query.side == "minus" else plus_fluorescence
    )
    completed = tracker.update(
        EstimatorObservation(
            second_query.expected_sequence_index,
            second_query.expected_end_timestamp_s,
            second_query.frequency_hz,
            second_fluorescence,
            second_query.integration_time_s,
            second_query.expected_nominal_exposure_photons,
            None,
        )
    ).completed_pair
    assert completed is not None
    return completed


def test_domain_endpoints_and_active_age_transitions_are_exact() -> None:
    base_configuration, base_calibration, metadata = (
        _make_rounding_witness_calibration()
    )
    base_cell = base_calibration.identities[0]
    for direction in (-1.0, 1.0):
        requested_raw_hz = (
            direction * 0.05 * base_cell.calibration_fwhm_hz
        )
        _, _, actual_raw_hz = _pair_fluorescence_for_raw(
            base_calibration,
            0,
            base_cell.calibration_center_hz,
            requested_raw_hz,
        )
        exact_candidate_hz = base_cell.calibration_center_hz + actual_raw_hz
        if direction < 0.0:
            exact_cell = replace(
                base_cell, allowed_center_min_hz=exact_candidate_hz
            )
            outward_cell = replace(
                base_cell,
                allowed_center_min_hz=math.nextafter(
                    exact_candidate_hz, math.inf
                ),
            )
        else:
            exact_cell = replace(
                base_cell, allowed_center_max_hz=exact_candidate_hz
            )
            outward_cell = replace(
                base_cell,
                allowed_center_max_hz=math.nextafter(
                    exact_candidate_hz, -math.inf
                ),
            )

        for cell, should_pass in ((exact_cell, True), (outward_cell, False)):
            calibration = replace(
                base_calibration,
                identities=(cell, *base_calibration.identities[1:]),
            )
            tracker = CalibratedTwoPointTracker(base_configuration)
            tracker.reset(
                metadata,
                calibration,
                TwoPointBudgetCeiling(None, 100.0, None, None),
                seed=47,
            )
            pair = _complete_pair_with_raw(
                tracker, calibration, requested_raw_hz
            )
            assert pair.candidate_center_hz == exact_candidate_hz
            if should_pass:
                assert pair.lock_state == "tracking"
                assert pair.failure_code is None
                assert tracker.estimate().identities[0].center_hz == (
                    cell.allowed_center_min_hz
                    if direction < 0.0
                    else cell.allowed_center_max_hz
                )
            else:
                assert pair.lock_state == "lost"
                assert pair.failure_code == "calibration_domain_exceeded"
                endpoint_hz = (
                    cell.allowed_center_min_hz
                    if direction < 0.0
                    else cell.allowed_center_max_hz
                )
                assert pair.candidate_center_hz == math.nextafter(
                    endpoint_hz,
                    -math.inf if direction < 0.0 else math.inf,
                )
                assert tracker.estimate().identities[0].center_hz == (
                    base_cell.calibration_center_hz
                )

    tracker = CalibratedTwoPointTracker(base_configuration)
    tracker.reset(
        metadata,
        base_calibration,
        TwoPointBudgetCeiling(None, 100.0, None, None),
        seed=53,
    )
    expected_centers = tuple(
        identity.center_hz for identity in tracker.estimate().identities
    )
    for pair_index in range(21):
        identity_index = pair_index % 8
        cell = base_calibration.identities[identity_index]
        raw_hz = 0.0
        if identity_index == 3:
            raw_hz = 0.15 * cell.calibration_fwhm_hz
        elif identity_index == 4:
            raw_hz = -0.15 * cell.calibration_fwhm_hz
        before_center_hz = tracker.estimate().identities[identity_index].center_hz
        pair = _complete_pair_with_raw(tracker, base_calibration, raw_hz)
        if identity_index in {3, 4}:
            direction = 1.0 if identity_index == 3 else -1.0
            assert pair.lock_state == "step_limited"
            assert pair.applied_step_hz == direction * cell.max_step_hz
            expected_centers = tuple(
                center + direction * cell.max_step_hz
                if index == identity_index
                else center
                for index, center in enumerate(expected_centers)
            )
            assert pair.candidate_center_hz == (
                before_center_hz + direction * cell.max_step_hz
            )
            assert (
                cell.allowed_center_min_hz
                <= pair.candidate_center_hz
                <= cell.allowed_center_max_hz
            )
        assert tuple(
            identity.center_hz for identity in tracker.estimate().identities
        ) == expected_centers

    tracker = CalibratedTwoPointTracker(base_configuration)
    tracker.reset(
        metadata,
        base_calibration,
        TwoPointBudgetCeiling(None, 100.0, None, None),
        seed=59,
    )
    initial = tracker.estimate()
    r0_initial = initial.identities[0]
    r0_initial_epoch = (
        r0_initial.center_hz,
        r0_initial.active_source_kind,
        r0_initial.active_source_pair_index,
        r0_initial.active_reference_timestamp_s,
        r0_initial.active_release_sequence_index,
        r0_initial.active_release_timestamp_s,
    )
    first_query = tracker.choose_next_query()
    assert first_query is not None
    first_value, _, _ = _pair_fluorescence_for_raw(
        base_calibration, 0, first_query.interrogation_center_hz, 0.0
    )
    tracker.update(
        EstimatorObservation(
            first_query.expected_sequence_index,
            first_query.expected_end_timestamp_s,
            first_query.frequency_hz,
            first_value,
            first_query.integration_time_s,
            first_query.expected_nominal_exposure_photons,
            None,
        )
    )
    after_first = tracker.estimate()
    r0_after_first = after_first.identities[0]
    assert (
        r0_after_first.center_hz,
        r0_after_first.active_source_kind,
        r0_after_first.active_source_pair_index,
        r0_after_first.active_reference_timestamp_s,
        r0_after_first.active_release_sequence_index,
        r0_after_first.active_release_timestamp_s,
    ) == r0_initial_epoch
    assert r0_after_first.active_source_kind == "calibration"
    assert r0_after_first.estimate_age_sequence_indices == (
        after_first.current_sequence_index
        - r0_after_first.active_release_sequence_index
    )
    assert r0_after_first.estimate_age_s == (
        after_first.current_timestamp_s
        - r0_after_first.active_reference_timestamp_s
    )
    assert r0_after_first.release_age_s == (
        after_first.current_timestamp_s
        - r0_after_first.active_release_timestamp_s
    )
    assert r0_after_first.estimate_age_s > r0_initial.estimate_age_s

    second_query = tracker.choose_next_query()
    assert second_query is not None
    _, second_value, _ = _pair_fluorescence_for_raw(
        base_calibration, 0, second_query.interrogation_center_hz, 0.0
    )
    released = tracker.update(
        EstimatorObservation(
            second_query.expected_sequence_index,
            second_query.expected_end_timestamp_s,
            second_query.frequency_hz,
            second_value,
            second_query.integration_time_s,
            second_query.expected_nominal_exposure_photons,
            None,
        )
    ).completed_pair
    assert released is not None
    r0_fresh = tracker.estimate().identities[0]
    assert r0_fresh.active_source_kind == "pair"
    assert r0_fresh.estimate_age_sequence_indices == 0
    assert r0_fresh.estimate_age_s == (
        released.release_timestamp_s - released.pair_reference_timestamp_s
    )
    assert r0_fresh.estimate_age_s > 0.0
    assert r0_fresh.release_age_s == 0.0

    r1_before_failure = tracker.estimate().identities[1]
    failed_first = tracker.choose_next_query()
    assert failed_first is not None
    tracker.update(
        EstimatorObservation(
            failed_first.expected_sequence_index,
            failed_first.expected_end_timestamp_s,
            failed_first.frequency_hz,
            1.0,
            failed_first.integration_time_s,
            failed_first.expected_nominal_exposure_photons,
            None,
        )
    )
    r0_nonfresh = tracker.estimate().identities[0]
    assert r0_nonfresh.estimate_age_sequence_indices == (
        tracker.estimate().current_sequence_index
        - r0_nonfresh.active_release_sequence_index
    )
    failed_second = tracker.choose_next_query()
    assert failed_second is not None
    failed = tracker.update(
        EstimatorObservation(
            failed_second.expected_sequence_index,
            failed_second.expected_end_timestamp_s,
            failed_second.frequency_hz,
            -1.0,
            failed_second.integration_time_s,
            failed_second.expected_nominal_exposure_photons,
            None,
        )
    ).completed_pair
    assert failed is not None
    assert failed.failure_code == "invalid_pair_normalization"
    r1_after_failure = tracker.estimate().identities[1]
    assert r1_after_failure.center_hz == r1_before_failure.center_hz
    assert (
        r1_after_failure.active_source_kind
        == r1_before_failure.active_source_kind
        == "calibration"
    )
    assert (
        r1_after_failure.active_reference_timestamp_s
        == r1_before_failure.active_reference_timestamp_s
    )
    assert (
        r1_after_failure.active_release_sequence_index
        == r1_before_failure.active_release_sequence_index
    )
    assert r1_after_failure.estimate_age_sequence_indices == (
        tracker.estimate().current_sequence_index
        - r1_after_failure.active_release_sequence_index
    )
    assert r1_after_failure.estimate_age_s == (
        tracker.estimate().current_timestamp_s
        - r1_after_failure.active_reference_timestamp_s
    )
    assert r1_after_failure.release_age_s == (
        tracker.estimate().current_timestamp_s
        - r1_after_failure.active_release_timestamp_s
    )

    for _ in range(6):
        retained_pair = _complete_pair_with_raw(tracker, base_calibration, 0.0)
        assert retained_pair.lock_state == "tracking"
    r0_before_failure = tracker.estimate().identities[0]
    r0_pair_epoch = (
        r0_before_failure.center_hz,
        r0_before_failure.active_source_kind,
        r0_before_failure.active_source_pair_index,
        r0_before_failure.active_reference_timestamp_s,
        r0_before_failure.active_release_sequence_index,
        r0_before_failure.active_release_timestamp_s,
    )
    assert r0_pair_epoch[1:3] == ("pair", 0)

    failed_r0 = None
    for arrival_index, fluorescence in enumerate((1.0, -1.0)):
        query = tracker.choose_next_query()
        assert query is not None
        assert query.pair_index == 8
        failed_r0 = tracker.update(
            EstimatorObservation(
                query.expected_sequence_index,
                query.expected_end_timestamp_s,
                query.frequency_hz,
                fluorescence,
                query.integration_time_s,
                query.expected_nominal_exposure_photons,
                None,
            )
        ).completed_pair
        r0_after_arrival = tracker.estimate().identities[0]
        assert (
            r0_after_arrival.center_hz,
            r0_after_arrival.active_source_kind,
            r0_after_arrival.active_source_pair_index,
            r0_after_arrival.active_reference_timestamp_s,
            r0_after_arrival.active_release_sequence_index,
            r0_after_arrival.active_release_timestamp_s,
        ) == r0_pair_epoch
        if arrival_index == 0:
            assert failed_r0 is None
    assert failed_r0 is not None
    assert failed_r0.failure_code == "invalid_pair_normalization"
    assert tracker.estimate().identities[0].estimate_age_sequence_indices == (
        tracker.estimate().current_sequence_index - r0_pair_epoch[4]
    )
    assert tracker.estimate().identities[0].estimate_age_s == (
        tracker.estimate().current_timestamp_s - r0_pair_epoch[3]
    )
    assert tracker.estimate().identities[0].release_age_s == (
        tracker.estimate().current_timestamp_s - r0_pair_epoch[5]
    )


class _RejectingFutureObservationContainer:
    def __getattribute__(self, name: str):
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError("tracker inspected a future-observation container")


def _assert_tracker_graph_has_no_truth_path(value: object) -> None:
    forbidden_types = (
        InstrumentObservation,
        ResourceSnapshot,
        SpectralSnapshot,
        Future,
    )
    forbidden_name_fragments = (
        "expected_photons",
        "truth",
        "dynamics",
        "future",
        "noise",
        "callback",
        "evaluator",
        "capability",
    )
    visited: set[int] = set()

    def inspect_name(name: object, path: str) -> None:
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        if isinstance(name, str):
            normalized = name.casefold()
            assert not any(
                fragment in normalized for fragment in forbidden_name_fragments
            ), f"forbidden name or token at {path}: {name!r}"

    def inspect_value(item: object, path: str) -> None:
        if isinstance(item, forbidden_types):
            raise AssertionError(f"forbidden full-resource type at {path}")
        inspect_name(item, path)
        type_name = f"{type(item).__module__}.{type(item).__qualname__}"
        inspect_name(type_name, f"{path}.__type__")
        if callable(item) and not isinstance(item, type):
            raise AssertionError(f"forbidden retained callback at {path}")
        if item is None or isinstance(item, str | bytes | int | float | bool):
            return
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        represented_names: set[str] = set()
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                represented_names.add(field.name)
                inspect_name(field.name, f"{path}.{field.name}")
                inspect_value(getattr(item, field.name), f"{path}.{field.name}")
        if isinstance(item, Mapping):
            for key, nested in item.items():
                inspect_name(key, f"{path}.key")
                inspect_value(key, f"{path}.key")
                inspect_value(nested, f"{path}[{key!r}]")
            return
        if isinstance(item, tuple | list | set | frozenset):
            for index, nested in enumerate(item):
                inspect_value(nested, f"{path}[{index}]")
            return
        instance_dictionary = getattr(item, "__dict__", None)
        if isinstance(instance_dictionary, Mapping):
            for key, nested in instance_dictionary.items():
                if key in represented_names:
                    continue
                inspect_name(key, f"{path}.__dict__.key")
                inspect_value(nested, f"{path}.{key}")
        for owner in type(item).__mro__:
            slots = getattr(owner, "__slots__", ())
            slots = (slots,) if isinstance(slots, str) else slots
            for slot in slots:
                if slot in {"__dict__", "__weakref__"}:
                    continue
                if slot in represented_names:
                    continue
                inspect_name(slot, f"{path}.{slot}")
                if hasattr(item, slot):
                    inspect_value(getattr(item, slot), f"{path}.{slot}")

    inspect_value(value, "tracker")


class _EvaluatorSentinel:
    pass


class _CapabilitySentinel:
    pass


class _DictionaryHolder:
    pass


class _BaseSlotHolder:
    __slots__ = ("retained",)

    def __init__(self, retained: object) -> None:
        self.retained = retained


class _DerivedSlotHolder(_BaseSlotHolder):
    __slots__ = ()


@dataclass
class _DynamicDataclassHolder:
    public_value: int


def test_truth_graph_checker_rejects_every_forbidden_storage_form() -> None:
    dictionary_holder = _DictionaryHolder()
    dictionary_holder.retained = Future()
    dataclass_holder = _DynamicDataclassHolder(1)
    dataclass_holder.retained = Future()
    rejecting_holder = _BaseSlotHolder(_RejectingFutureObservationContainer())
    forbidden_values = (
        object.__new__(InstrumentObservation),
        object.__new__(ResourceSnapshot),
        object.__new__(SpectralSnapshot),
        Future(),
        _EvaluatorSentinel(),
        _CapabilitySentinel(),
        {"expected_photons": 1.0},
        {"safe": Future()},
        dictionary_holder,
        dataclass_holder,
        _DerivedSlotHolder(Future()),
        rejecting_holder,
    )

    for forbidden in forbidden_values:
        with pytest.raises(AssertionError):
            _assert_tracker_graph_has_no_truth_path(forbidden)


def test_tracker_public_surface_has_no_evaluator_or_capability_api() -> None:
    assert tuple(
        name for name in dir(CalibratedTwoPointTracker) if not name.startswith("_")
    ) == (
        "calibration",
        "choose_next_query",
        "configuration",
        "estimate",
        "pair_history",
        "pending_query",
        "reset",
        "update",
    )


def test_seed_is_observationally_inert_and_tracker_has_no_truth_path() -> None:
    configuration, calibration, metadata = _make_rounding_witness_calibration()
    seeds = (60, 997, (1 << 127) + 3)
    assert {seed % 2 for seed in seeds} == {0, 1}
    assert len({seed % 8 for seed in seeds}) == 3
    trackers = tuple(CalibratedTwoPointTracker(configuration) for _ in seeds)
    for tracker, seed in zip(trackers, seeds, strict=True):
        tracker.reset(
            metadata,
            calibration,
            TwoPointBudgetCeiling(None, 100.0, None, None),
            seed=seed,
        )
    reference = trackers[0]
    for comparison in trackers[1:]:
        assert replace(
            reference.estimate(), seed=comparison.estimate().seed
        ) == comparison.estimate()

    for _ in range(18):
        queries = tuple(tracker.choose_next_query() for tracker in trackers)
        assert all(query == queries[0] for query in queries[1:])
        reference_query = queries[0]
        assert reference_query is not None
        identity_index = reference_query.pair_index % 8
        cell = calibration.identities[identity_index]
        fluorescence = _evaluate_target_only_model(
            calibration.source.source_fit,
            cell.source_fit_index,
            reference_query.frequency_hz,
            reference_query.interrogation_center_hz,
        )
        observation = EstimatorObservation(
            reference_query.expected_sequence_index,
            reference_query.expected_end_timestamp_s,
            reference_query.frequency_hz,
            fluorescence,
            reference_query.integration_time_s,
            reference_query.expected_nominal_exposure_photons,
            None,
        )
        updates = tuple(tracker.update(observation) for tracker in trackers)
        reference_update = updates[0]
        for comparison, comparison_update in zip(
            trackers[1:], updates[1:], strict=True
        ):
            assert replace(
                reference_update.estimate,
                seed=comparison_update.estimate.seed,
            ) == comparison_update.estimate
            assert reference_update.query == comparison_update.query
            assert reference_update.observation == comparison_update.observation
            assert reference_update.completed_pair == comparison_update.completed_pair
            assert reference.pending_query == comparison.pending_query
            assert reference.pair_history == comparison.pair_history

    pending = reference.choose_next_query()
    assert pending is not None
    before = _tracker_snapshot(reference)
    with pytest.raises(TwoPointObservationValidationError) as caught:
        reference.update(  # type: ignore[arg-type]
            _RejectingFutureObservationContainer()
        )
    assert caught.value.code == "invalid_observation_type"
    assert _tracker_snapshot(reference) == before
    for tracker in trackers:
        _assert_tracker_graph_has_no_truth_path(tracker)


def _tracker_at_public_zero_model_sum_endpoint():
    fit_configuration = replace(make_legal_fit_configuration(), baseline_degree=2)
    source_fit = make_legal_source_fit(fit_configuration)
    endpoint_baseline = Baseline(
        0.004290640100839996,
        2.777e9,
        0.0,
        1.0e-14,
    )
    assert source_fit.initial_guess is not None
    source_fit = replace(
        source_fit,
        baseline_degree=2,
        baseline_estimate=endpoint_baseline,
        initial_guess=replace(
            source_fit.initial_guess,
            baseline=endpoint_baseline,
        ),
        jacobian_rank=35,
    )
    source = make_legal_caller_asserted_source(
        source_fit=source_fit,
        fit_configuration=fit_configuration,
    )
    configuration = replace(
        make_legal_tracker_configuration(),
        proportional_gain=100.0,
        max_step_fwhm_fraction=16_175_000.0 / 1_500_000.0,
    )
    calibration = calibrate_two_point(
        source,
        configuration,
        budget_treatment="conditional_free_precalibration",
    )
    tracker = CalibratedTwoPointTracker(configuration)
    tracker.reset(
        TwoPointRunMetadata(
            "clock",
            None,
            0.010,
            2.5e6,
            0.0,
            "normalized_fluorescence",
        ),
        calibration,
        TwoPointBudgetCeiling(100, None, None, None),
        seed=67,
    )

    endpoint_pair = _complete_pair_with_raw(
        tracker,
        calibration,
        0.15 * calibration.identities[0].calibration_fwhm_hz,
    )
    assert endpoint_pair.lock_state == "step_limited"
    assert endpoint_pair.candidate_center_hz == 2_776_175_000.0
    assert (
        tracker.estimate().identities[0].center_hz
        == calibration.identities[0].allowed_center_max_hz
        == 2_776_175_000.0
    )
    cell = calibration.identities[0]
    endpoint_center_hz = tracker.estimate().identities[0].center_hz
    endpoint_models = (
        _evaluate_target_only_model(
            calibration.source.source_fit,
            cell.source_fit_index,
            endpoint_center_hz - cell.offset_hz,
            endpoint_center_hz,
        ),
        _evaluate_target_only_model(
            calibration.source.source_fit,
            cell.source_fit_index,
            endpoint_center_hz + cell.offset_hz,
            endpoint_center_hz,
        ),
    )
    assert tuple(value.hex() for value in endpoint_models) == (
        "0x1.1be389a7e357ep-7",
        "-0x1.1be389a7e357ep-7",
    )
    assert math.copysign(1.0, endpoint_models[0] + endpoint_models[1]) == 1.0
    assert endpoint_models[0] + endpoint_models[1] == 0.0
    for _ in range(7):
        pair = _complete_pair_with_raw(tracker, calibration, 0.0)
        assert pair.lock_state == "tracking"
    assert tracker.estimate().completed_pairs == 8
    return tracker, calibration


def _tracker_at_shifted_r0_pair(
) -> tuple[CalibratedTwoPointTracker, TwoPointCalibration]:
    tracker, calibration, _ = _reset_rounding_witness(
        TwoPointBudgetCeiling(None, 100.0, None, None)
    )
    first_cell = calibration.identities[0]
    moved = _complete_pair_with_raw(
        tracker,
        calibration,
        0.05 * first_cell.calibration_fwhm_hz,
    )
    assert moved.lock_state == "tracking"
    assert moved.candidate_center_hz != first_cell.calibration_center_hz
    for _ in range(7):
        assert _complete_pair_with_raw(tracker, calibration, 0.0).lock_state == (
            "tracking"
        )
    assert tracker.estimate().completed_pairs == 8
    return tracker, calibration


@pytest.mark.parametrize("invalid_normalization", (False, True))
def test_pair_geometry_is_current_center_only_or_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    invalid_normalization: bool,
) -> None:
    tracker, calibration = _tracker_at_shifted_r0_pair()
    cell = calibration.identities[0]
    center_hz = tracker.estimate().identities[0].center_hz
    minus_frequency_hz = center_hz - cell.offset_hz
    plus_frequency_hz = center_hz + cell.offset_hz
    mu_minus = _evaluate_target_only_model(
        calibration.source.source_fit,
        cell.source_fit_index,
        minus_frequency_hz,
        center_hz,
    )
    mu_plus = _evaluate_target_only_model(
        calibration.source.source_fit,
        cell.source_fit_index,
        plus_frequency_hz,
        center_hz,
    )
    g_minus = _target_center_derivative(
        calibration.source.source_fit,
        cell.source_fit_index,
        minus_frequency_hz,
        center_hz,
    )
    g_plus = _target_center_derivative(
        calibration.source.source_fit,
        cell.source_fit_index,
        plus_frequency_hz,
        center_hz,
    )
    model_sum = mu_minus + mu_plus
    expected_geometry = (
        (mu_minus - mu_plus) / model_sum,
        2.0 * (mu_plus * g_minus - mu_minus * g_plus) / model_sum**2,
    )
    calibration_center_hz = cell.calibration_center_hz
    calibration_mu_minus = _evaluate_target_only_model(
        calibration.source.source_fit,
        cell.source_fit_index,
        calibration_center_hz - cell.offset_hz,
        calibration_center_hz,
    )
    calibration_mu_plus = _evaluate_target_only_model(
        calibration.source.source_fit,
        cell.source_fit_index,
        calibration_center_hz + cell.offset_hz,
        calibration_center_hz,
    )
    calibration_g_minus = _target_center_derivative(
        calibration.source.source_fit,
        cell.source_fit_index,
        calibration_center_hz - cell.offset_hz,
        calibration_center_hz,
    )
    calibration_g_plus = _target_center_derivative(
        calibration.source.source_fit,
        cell.source_fit_index,
        calibration_center_hz + cell.offset_hz,
        calibration_center_hz,
    )
    calibration_sum = calibration_mu_minus + calibration_mu_plus
    calibration_geometry = (
        (calibration_mu_minus - calibration_mu_plus) / calibration_sum,
        2.0
        * (
            calibration_mu_plus * calibration_g_minus
            - calibration_mu_minus * calibration_g_plus
        )
        / calibration_sum**2,
    )
    assert expected_geometry != calibration_geometry

    fluorescence_by_side = (
        {"minus": 1.0, "plus": -1.0}
        if invalid_normalization
        else {"minus": mu_minus, "plus": mu_plus}
    )
    completed = None
    for arrival_index in range(2):
        query = tracker.choose_next_query()
        assert query is not None
        if arrival_index == 1 and invalid_normalization:
            def reject_model_evaluation(*args, **kwargs):
                del args, kwargs
                raise AssertionError("invalid normalization must skip local geometry")

            monkeypatch.setattr(
                tracker_module,
                "_evaluate_target_only_model",
                reject_model_evaluation,
            )
        completed = tracker.update(
            EstimatorObservation(
                query.expected_sequence_index,
                query.expected_end_timestamp_s,
                query.frequency_hz,
                fluorescence_by_side[query.side],
                query.integration_time_s,
                query.expected_nominal_exposure_photons,
                None,
            )
        ).completed_pair

    assert completed is not None
    if invalid_normalization:
        assert completed.failure_code == "invalid_pair_normalization"
        assert (
            completed.zero_discriminator,
            completed.discriminator_slope_per_hz,
        ) == (None, None)
    else:
        assert completed.failure_code is None
        assert completed.lock_state == "tracking"
        assert (
            completed.zero_discriminator,
            completed.discriminator_slope_per_hz,
        ) == expected_geometry


@pytest.mark.parametrize(
    ("arrival_fluorescence", "expected_failure"),
    (
        pytest.param((1.0, 1.0), "numerical_failure", id="positive-observed-sum"),
        pytest.param(
            (1.0, -1.0),
            "invalid_pair_normalization",
            id="zero-observed-sum",
        ),
    ),
)
def test_public_endpoint_bad_model_commits_pair_and_advances_without_wedging(
    monkeypatch: pytest.MonkeyPatch,
    arrival_fluorescence: tuple[float, float],
    expected_failure: str,
) -> None:
    tracker, calibration = _tracker_at_public_zero_model_sum_endpoint()
    active_before = tracker.estimate().identities[0]
    active_epoch_before = (
        active_before.center_hz,
        active_before.active_source_kind,
        active_before.active_source_pair_index,
        active_before.active_reference_timestamp_s,
        active_before.active_release_sequence_index,
        active_before.active_release_timestamp_s,
    )
    queries = []
    completed = None
    for arrival_index, fluorescence in enumerate(arrival_fluorescence):
        query = tracker.choose_next_query()
        assert query is not None
        queries.append(query)
        if arrival_index == 1 and expected_failure == "invalid_pair_normalization":
            def reject_lower_precedence_model(*args, **kwargs):
                del args, kwargs
                raise AssertionError(
                    "invalid observed normalization must precede model arithmetic"
                )

            monkeypatch.setattr(
                tracker_module,
                "_evaluate_target_only_model",
                reject_lower_precedence_model,
            )
        completed = tracker.update(
            EstimatorObservation(
                query.expected_sequence_index,
                query.expected_end_timestamp_s,
                query.frequency_hz,
                fluorescence,
                query.integration_time_s,
                query.expected_nominal_exposure_photons,
                None,
            )
        ).completed_pair

    assert tuple(query.side for query in queries) == ("plus", "minus")
    assert completed is not None
    assert completed.pair_index == 8
    assert completed.failure_code == expected_failure
    assert completed.lock_state == "lost"
    assert completed.applied_step_hz == 0.0
    assert (
        completed.zero_discriminator,
        completed.discriminator_slope_per_hz,
    ) == (None, None)
    assert (
        completed.discriminator,
        completed.common_mode_target_depths,
        completed.raw_innovation_hz,
        completed.requested_step_hz,
        completed.candidate_center_hz,
    ) == (None, None, None, None, None)
    after = tracker.estimate()
    assert after.completed_pairs == 9
    assert after.accepted_observations == 18
    assert after.pair_history[-1] is completed
    active_after = after.identities[0]
    assert active_after.latest_pair is completed
    assert active_after.completed_pairs == 2
    assert (
        active_after.center_hz,
        active_after.active_source_kind,
        active_after.active_source_pair_index,
        active_after.active_reference_timestamp_s,
        active_after.active_release_sequence_index,
        active_after.active_release_timestamp_s,
    ) == active_epoch_before
    next_query = tracker.choose_next_query()
    assert next_query is not None
    assert next_query.pair_index == 9
    assert next_query.resonance_id == calibration.identities[1].resonance_id
    assert next_query.identity_pair_index == 1
    assert next_query.side == "plus"


def _complete_numeric_witness_pair(
    fluorescence: tuple[float, float],
    monkeypatch: pytest.MonkeyPatch | None = None,
    injected_stage: str | None = None,
):
    configuration, calibration, _ = _make_rounding_witness_calibration()
    tracker, _, _ = _reset_rounding_witness(
        TwoPointBudgetCeiling(None, 100.0, None, None)
    )
    completed = None
    for arrival_index, value in enumerate(fluorescence):
        query = tracker.choose_next_query()
        assert query is not None
        if arrival_index == 1 and injected_stage is not None:
            assert monkeypatch is not None
            if injected_stage == "model":
                original_evaluator = tracker_module._evaluate_target_only_model
                calls = 0

                def nonfinite_current_model(
                    *args,
                    _original_evaluator=original_evaluator,
                    **kwargs,
                ):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return math.inf
                    return _original_evaluator(*args, **kwargs)

                monkeypatch.setattr(
                    tracker_module,
                    "_evaluate_target_only_model",
                    nonfinite_current_model,
                )
            elif injected_stage == "raw":
                monkeypatch.setattr(
                    tracker_module,
                    "_pair_model_geometry",
                    lambda *args, **kwargs: (
                        1.0,
                        0.0,
                        float.fromhex("0x0.0000000000001p-1022"),
                    ),
                )
            elif injected_stage == "candidate":
                monkeypatch.setattr(
                    tracker_module,
                    "max",
                    lambda *args: math.inf,
                    raising=False,
                )
            else:
                raise AssertionError(f"unknown injected stage: {injected_stage}")
        completed = tracker.update(
            EstimatorObservation(
                query.expected_sequence_index,
                query.expected_end_timestamp_s,
                query.frequency_hz,
                value,
                query.integration_time_s,
                query.expected_nominal_exposure_photons,
                None,
            )
        ).completed_pair
    assert completed is not None
    return completed, configuration, calibration


@pytest.mark.parametrize(
    ("stage", "fluorescence", "diagnostic_presence"),
    (
        pytest.param(
            "observed",
            (1.0e308, 1.0e308),
            (False, False, False, False, False),
            id="observed-sum",
        ),
        pytest.param(
            "discriminator",
            (1.0e308, -8.0e307),
            (False, False, False, False, False),
            id="discriminator",
        ),
        pytest.param(
            "common",
            (5.0e307, 5.0e307),
            (True, False, False, False, False),
            id="common",
        ),
    ),
)
def test_numerical_failure_retains_exact_public_arithmetic_prefix(
    stage: str,
    fluorescence: tuple[float, float],
    diagnostic_presence: tuple[bool, bool, bool, bool, bool],
) -> None:
    pair, _, _ = _complete_numeric_witness_pair(fluorescence)

    assert pair.failure_code == "numerical_failure"
    assert pair.lock_state == "lost"
    assert pair.applied_step_hz == 0.0
    assert tuple(
        value is not None
        for value in (
            pair.discriminator,
            pair.common_mode_target_depths,
            pair.raw_innovation_hz,
            pair.requested_step_hz,
            pair.candidate_center_hz,
        )
    ) == diagnostic_presence
    if stage == "observed":
        assert (pair.zero_discriminator, pair.discriminator_slope_per_hz) == (
            None,
            None,
        )
    else:
        assert pair.zero_discriminator is not None
        assert pair.discriminator_slope_per_hz is not None
        assert pair.discriminator_slope_per_hz > 0.0


@pytest.mark.parametrize(
    ("stage", "fluorescence", "diagnostic_presence"),
    (
        pytest.param(
            "model",
            (0.5, 0.5),
            (False, False, False, False, False),
            id="model-value",
        ),
        pytest.param(
            "raw",
            (1.0, 0.0),
            (True, True, False, False, False),
            id="raw-innovation",
        ),
        pytest.param(
            "candidate",
            (1.0, 0.0),
            (True, True, True, True, False),
            id="candidate-center",
        ),
    ),
)
def test_numerical_failure_retains_exact_fault_injected_arithmetic_prefix(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    fluorescence: tuple[float, float],
    diagnostic_presence: tuple[bool, bool, bool, bool, bool],
) -> None:
    pair, _, _ = _complete_numeric_witness_pair(
        fluorescence,
        monkeypatch,
        stage,
    )

    assert pair.failure_code == "numerical_failure"
    assert pair.lock_state == "lost"
    assert pair.applied_step_hz == 0.0
    assert tuple(
        value is not None
        for value in (
            pair.discriminator,
            pair.common_mode_target_depths,
            pair.raw_innovation_hz,
            pair.requested_step_hz,
            pair.candidate_center_hz,
        )
    ) == diagnostic_presence
    if stage == "model":
        assert (pair.zero_discriminator, pair.discriminator_slope_per_hz) == (
            None,
            None,
        )
    else:
        assert pair.zero_discriminator is not None
        assert pair.discriminator_slope_per_hz is not None
        assert pair.discriminator_slope_per_hz > 0.0


def test_requested_step_numerical_failure_retains_exact_prefix() -> None:
    pair, _, _ = _complete_pair_for_gate("numerical")

    assert pair.failure_code == "numerical_failure"
    assert pair.lock_state == "lost"
    assert pair.applied_step_hz == 0.0
    assert tuple(
        value is not None
        for value in (
            pair.discriminator,
            pair.common_mode_target_depths,
            pair.raw_innovation_hz,
            pair.requested_step_hz,
            pair.candidate_center_hz,
        )
    ) == (True, True, True, False, False)
    assert pair.zero_discriminator is not None
    assert pair.discriminator_slope_per_hz is not None
    assert pair.discriminator_slope_per_hz > 0.0
