"""Tests for calibrated two-point tracker reset and pair reservation."""

from __future__ import annotations

import inspect
import math
from copy import copy
from dataclasses import replace

import numpy as np
import pytest

from odmr_bench.emulator.observations import EstimatorObservation
from odmr_bench.estimators import (
    CalibratedTwoPointTracker,
    NormalizedFluorescenceProvenance,
    PublicAcquisitionResources,
    TwoPointBudgetCeiling,
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
