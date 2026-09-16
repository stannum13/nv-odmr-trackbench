"""Closed scientific acceptance regressions for sparse linewidth tracking."""

import importlib
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, fields, is_dataclass, replace

import pytest

from odmr_bench.dynamics import SpectralDynamics, SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import (
    EmpiricalResidualNoise,
    GaussianNoise,
    InstrumentObservation,
    ODMRInstrument,
    PoissonNoise,
    ResourceLedger,
    ResourceSnapshot,
)
from odmr_bench.estimators import (
    SparseLinewidthCompositeTracker,
    SparseLinewidthConfiguration,
    SparseLinewidthResetError,
    TwoPointBudgetCeiling,
    TwoPointRunMetadata,
)
from odmr_bench.evaluation.sparse_linewidth.runner import (
    SparseLinewidthEvaluatorRunner,
)
from odmr_bench.evaluation.sparse_linewidth.types import (
    SparseLinewidthEvaluatorScanTiming,
)
from tests.evaluation.sparse_linewidth_acceptance import (
    QueryScopedDynamicsSpy,
    affine_baseline_mismatch_case,
    composed_center_width_drift_case,
    due_geometry_stop_case,
    evaluate_released_scan_truth,
    exact_static_case,
    independent_source_epoch_case,
    retained_width_failure_case,
    run_acceptance_case,
    run_acceptance_case_with_trace,
    seeded_poisson_static_case,
    unaffordable_sparse_block_case,
    within_scan_dynamics_case,
)
from tests.sparse_linewidth_helpers import make_composite_identity


def recursively_find_forbidden_types(value: object) -> tuple[str, ...]:
    """Return every forbidden truth-capability path in one retained graph."""
    forbidden_types = (
        InstrumentObservation,
        ResourceSnapshot,
        ResourceLedger,
        SpectralSnapshot,
        SpectralDynamics,
        ODMRInstrument,
        GaussianNoise,
        PoissonNoise,
        EmpiricalResidualNoise,
        Future,
    )
    forbidden_name_fragments = (
        "expected_photons",
        "truth",
        "future",
        "dynamics",
        "noise",
        "callback",
        "evaluator",
        "instrument",
        "capability",
    )
    visited: set[int] = set()
    violations: list[str] = []
    reported_paths: set[str] = set()

    def report(path: str) -> None:
        if path not in reported_paths:
            reported_paths.add(path)
            violations.append(path)

    def name_is_forbidden(name: object) -> bool:
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return isinstance(name, str) and any(
            fragment in name.casefold() for fragment in forbidden_name_fragments
        )

    def inspect(item: object, path: str) -> None:
        if isinstance(item, forbidden_types):
            report(path)
            return
        type_name = f"{type(item).__module__}.{type(item).__name__}"
        if name_is_forbidden(type_name):
            report(f"{path}.__type__")
        if callable(item) and not isinstance(item, type):
            report(path)
            return
        if item is None or type(item) in {str, bytes, int, float, bool}:
            return
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)

        represented_names: set[str] = set()
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                represented_names.add(field.name)
                nested_path = f"{path}.{field.name}"
                if name_is_forbidden(field.name):
                    report(nested_path)
                inspect(getattr(item, field.name), nested_path)

        if isinstance(item, Mapping):
            for index, (key, nested) in enumerate(item.items()):
                key_path = f"{path}.key[{index}]"
                if name_is_forbidden(key):
                    report(key_path)
                inspect(key, key_path)
                inspect(nested, f"{path}.value[{index}]")
        if isinstance(item, tuple | list | set | frozenset):
            for index, nested in enumerate(item):
                inspect(nested, f"{path}[{index}]")

        instance_dictionary = getattr(item, "__dict__", None)
        if isinstance(instance_dictionary, Mapping):
            for name, nested in instance_dictionary.items():
                if name in represented_names:
                    continue
                nested_path = f"{path}.{name}"
                if name_is_forbidden(name):
                    report(nested_path)
                inspect(nested, nested_path)

        for owner in type(item).__mro__:
            slots = getattr(owner, "__slots__", ())
            slots = (slots,) if isinstance(slots, str) else slots
            for slot in slots:
                if slot in {"__dict__", "__weakref__"} or slot in represented_names:
                    continue
                nested_path = f"{path}.{slot}"
                if name_is_forbidden(slot):
                    report(nested_path)
                if hasattr(item, slot):
                    inspect(getattr(item, slot), nested_path)

    inspect(value, "estimate")
    return tuple(violations)


class _TruthOracleSentinel:
    pass


class _EvaluatorSentinel:
    pass


class _ExpectedPhotonsCapability:
    pass


class _DictionaryHolder:
    pass


class DictCapabilityCarrier(dict[object, object]):
    pass


class ListCapabilityCarrier(list[object]):
    pass


class CyclicDictCarrier(dict[object, object]):
    pass


class CyclicListCarrier(list[object]):
    pass


class _BaseSlotHolder:
    __slots__ = ("retained",)

    def __init__(self, retained: object) -> None:
        self.retained = retained


class _DerivedSlotHolder(_BaseSlotHolder):
    __slots__ = ()


@dataclass
class _DataclassHolder:
    retained: object


def test_exact_static_acceptance_recovers_local_linewidth() -> None:
    case = exact_static_case()
    outcome = run_acceptance_case(case)
    scan = outcome.state.tracker_estimate.sparse_scan_history[0]
    truth = evaluate_released_scan_truth(
        case.dynamics,
        outcome.state.scan_timings[0],
        completed_release_sequence_index=scan.release_sequence_index,
    )
    assert scan.status == "success"
    assert scan.fitted_center_correction_hz == 0.0
    assert scan.fitted_fwhm_hz == truth.resonances[0].fwhm_hz
    assert scan.fitted_amplitude == truth.resonances[0].amplitude
    assert scan.fitted_baseline_offset == 0.0
    assert scan.fitted_q == scan.fitted_local_center_hz / scan.fitted_fwhm_hz


def test_seeded_poisson_acceptance_has_fixed_scientific_tolerances() -> None:
    case = seeded_poisson_static_case()
    outcome = run_acceptance_case(case)
    scan = outcome.state.tracker_estimate.sparse_scan_history[0]
    truth = evaluate_released_scan_truth(
        case.dynamics,
        outcome.state.scan_timings[0],
        completed_release_sequence_index=scan.release_sequence_index,
    )

    assert scan.status == "success"
    assert abs(scan.fitted_local_center_hz - truth.resonances[0].center_hz) < 10_000.0
    assert abs(scan.fitted_fwhm_hz - truth.resonances[0].fwhm_hz) < 25_000.0
    assert abs(scan.fitted_amplitude - truth.resonances[0].amplitude) < 5.0e-4
    assert scan.amplitude_normalized_rmse < 0.01


def test_composed_center_width_drift_uses_release_gated_truth() -> None:
    case = composed_center_width_drift_case()
    outcome = run_acceptance_case(case)
    scan = outcome.state.tracker_estimate.sparse_scan_history[0]
    timing = outcome.state.scan_timings[0]
    truth = evaluate_released_scan_truth(
        case.dynamics,
        timing,
        completed_release_sequence_index=scan.release_sequence_index,
    )

    assert scan.status == "success"
    assert abs(scan.fitted_local_center_hz - truth.resonances[0].center_hz) < 2_000.0
    assert abs(scan.fitted_fwhm_hz - truth.resonances[0].fwhm_hz) < 3_000.0
    assert timing.public_reference_timestamp_s == scan.public_reference_timestamp_s
    assert timing.release_sequence_index == scan.release_sequence_index


def test_center_and_width_sources_retain_independent_epochs() -> None:
    outcome = run_acceptance_case(independent_source_epoch_case())
    identity = outcome.state.tracker_estimate.identities[0]
    source = outcome.state.verified_calibration.source

    assert source.physical_fit_epoch_s == pytest.approx(0.0065)
    assert identity.fast_center_source_kind == "pair"
    assert identity.fwhm_source_kind == "scan"
    assert identity.fast_center_reference_timestamp_s == pytest.approx(0.0185)
    assert identity.fwhm_reference_timestamp_s == pytest.approx(0.1235)
    assert identity.fast_center_release_sequence_index == 3
    assert identity.fwhm_release_sequence_index == 22
    assert (
        source.physical_fit_epoch_s
        < identity.fast_center_reference_timestamp_s
        < identity.fwhm_reference_timestamp_s
    )
    assert identity.live_q == identity.fast_center_hz / identity.active_fwhm_hz


def test_included_accounting_and_cpu_joins_cover_every_accepted_atom() -> None:
    outcome = run_acceptance_case(exact_static_case())
    state = outcome.state
    resources = outcome.resources

    assert len(state.normal_tracking_trace) == 21
    assert len(resources.accepted_fast_observations) == 16
    assert len(resources.accepted_sparse_observations) == 5
    assert resources.accepted_tracking_observations == tuple(
        acquisition.full_observation for acquisition in state.normal_tracking_trace
    )
    assert resources.calibration_resources.observations == 2
    assert resources.tracking_resources.observations == 21
    assert resources.charged_resources.observations == 23
    assert resources.calibration_budget_treatment == "included_same_run"
    assert state.fast_update_cpu_time_s >= 0.0
    assert state.sparse_update_cpu_time_s >= 0.0
    assert state.total_update_cpu_time_s >= 0.0
    assert (
        state.total_update_cpu_time_s
        == state.tracker_estimate.total_update_cpu_time_s
    )


def test_included_cpu_join_replays_every_accepted_update_in_arrival_order() -> None:
    terminal, accepted = run_acceptance_case_with_trace(exact_static_case())
    assert len(accepted) == 21
    fast = 0.0
    sparse = 0.0
    total = 0.0
    for item in accepted:
        total = total + item.update.update_cpu_time_s
        if item.acquisition.mode == "fast_pair":
            fast = fast + item.update.update_cpu_time_s
        else:
            sparse = sparse + item.update.update_cpu_time_s
    assert terminal.state.fast_update_cpu_time_s == fast
    assert terminal.state.sparse_update_cpu_time_s == sparse
    assert terminal.state.total_update_cpu_time_s == total


def test_scientific_failure_retains_calibration_width_and_completes_block() -> None:
    case = retained_width_failure_case()
    outcome = run_acceptance_case(case)
    estimate = outcome.state.tracker_estimate
    scan = estimate.sparse_scan_history[0]
    identity = estimate.identities[0]

    assert scan.status == "failure"
    assert scan.failure_code == "residual_quality_failed"
    assert len(scan.queries) == len(scan.observations) == 5
    assert tuple(item.sequence_index for item in scan.observations) == tuple(
        range(
            scan.observations[0].sequence_index,
            scan.observations[0].sequence_index + 5,
        )
    )
    assert identity.active_fwhm_hz == case.source_snapshot.resonances[0].fwhm_hz
    assert identity.fwhm_source_kind == "calibration"


def test_released_truth_rejects_incomplete_and_calls_once_after_release() -> None:
    case = exact_static_case()
    outcome = run_acceptance_case(case)
    timing = outcome.state.scan_timings[0]
    spy = QueryScopedDynamicsSpy(case.dynamics)

    with pytest.raises(ValueError, match="completed and released"):
        evaluate_released_scan_truth(
            spy,
            timing,
            completed_release_sequence_index=timing.release_sequence_index - 1,
        )
    assert spy.calls == []
    snapshot = evaluate_released_scan_truth(
        spy,
        timing,
        completed_release_sequence_index=timing.release_sequence_index,
    )
    assert snapshot.resonances[0].resonance_id == timing.resonance_id
    assert spy.calls == [(timing.truth_reference_timestamp_s, False)]


def test_production_tracking_has_no_out_of_query_truth_evaluation() -> None:
    base = exact_static_case()
    spy = QueryScopedDynamicsSpy(base.dynamics)
    run_acceptance_case(replace(base, dynamics=spy))

    assert spy.calls
    assert all(inside_query for _, inside_query in spy.calls)


@pytest.mark.parametrize(
    "forbidden",
    (
        object.__new__(InstrumentObservation),
        object.__new__(ResourceSnapshot),
        object.__new__(ResourceLedger),
        object.__new__(SpectralSnapshot),
        object.__new__(StationaryDynamics),
        object.__new__(ODMRInstrument),
        object.__new__(GaussianNoise),
        object.__new__(PoissonNoise),
        object.__new__(EmpiricalResidualNoise),
        object.__new__(SparseLinewidthEvaluatorRunner),
        Future(),
        _TruthOracleSentinel(),
        _EvaluatorSentinel(),
        _ExpectedPhotonsCapability(),
        lambda: None,
    ),
)
def test_truth_graph_checker_detects_every_forbidden_type_family(
    forbidden: object,
) -> None:
    assert recursively_find_forbidden_types([forbidden])


@pytest.mark.parametrize(
    "carrier",
    (
        _DataclassHolder(Future()),
        (lambda: None),
        _DerivedSlotHolder(Future()),
        {Future(): "safe"},
        {"safe": Future()},
        [Future()],
        (Future(),),
        {Future()},
        frozenset((Future(),)),
        {"expected_photons": 1.0},
    ),
)
def test_truth_graph_checker_detects_every_storage_form(
    carrier: object,
) -> None:
    if callable(carrier):
        holder = _DictionaryHolder()
        holder.retained = carrier
        carrier = holder
    assert recursively_find_forbidden_types(carrier)


@pytest.mark.parametrize(
    "carrier", (DictCapabilityCarrier(), ListCapabilityCarrier())
)
def test_truth_graph_checker_detects_capability_on_container_subclass(
    carrier: object,
) -> None:
    carrier.retained = Future()  # type: ignore[attr-defined]
    violations = recursively_find_forbidden_types(carrier)
    assert violations
    assert "estimate.retained" in violations


@pytest.mark.parametrize(
    "carrier", (CyclicDictCarrier(), CyclicListCarrier())
)
def test_truth_graph_checker_handles_clean_cyclic_container_subclass(
    carrier: object,
) -> None:
    if isinstance(carrier, dict):
        carrier["self"] = carrier
    else:
        carrier.append(carrier)  # type: ignore[attr-defined]
    assert recursively_find_forbidden_types(carrier) == ()


def test_truth_graph_checker_is_import_mode_invariant() -> None:
    qualified = importlib.import_module(
        "tests.evaluation.test_sparse_linewidth_regressions"
    )
    clean_dict = qualified.CyclicDictCarrier()
    clean_dict["self"] = clean_dict
    clean_list = qualified.CyclicListCarrier()
    clean_list.append(clean_list)

    assert qualified.recursively_find_forbidden_types(clean_dict) == ()
    assert qualified.recursively_find_forbidden_types(clean_list) == ()

    capable = qualified.DictCapabilityCarrier()
    capable.retained = Future()
    assert "estimate.retained" in qualified.recursively_find_forbidden_types(
        capable
    )
    evaluator = object.__new__(qualified.SparseLinewidthEvaluatorRunner)
    assert qualified.recursively_find_forbidden_types(evaluator)


def test_truth_graph_checker_reports_each_forbidden_path_once() -> None:
    @dataclass
    class Holder:
        truth: object

    assert recursively_find_forbidden_types(Holder(Future())) == (
        "estimate.truth",
    )

    dictionary_holder = _DictionaryHolder()
    dictionary_holder.truth = Future()
    assert recursively_find_forbidden_types(dictionary_holder) == (
        "estimate.truth",
    )
    assert recursively_find_forbidden_types({"truth": Future()}) == (
        "estimate.key[0]",
        "estimate.value[0]",
    )


def test_truth_graph_checker_is_cycle_safe_and_nonvacuous() -> None:
    cycle = _DictionaryHolder()
    cycle.self = cycle
    cycle.retained = _EvaluatorSentinel()

    assert recursively_find_forbidden_types(cycle)

    del cycle.retained
    assert recursively_find_forbidden_types(cycle) == ()


def test_estimator_graph_retains_no_truth_capability() -> None:
    terminal = run_acceptance_case(exact_static_case())
    estimate = terminal.state.tracker_estimate

    assert recursively_find_forbidden_types(estimate) == ()


@pytest.mark.parametrize(
    ("case_factory", "minimum_width_bias_hz"),
    [
        (affine_baseline_mismatch_case, 25_000.0),
        (within_scan_dynamics_case, 3_000.0),
    ],
)
def test_documented_model_mismatch_has_fixed_nontrivial_bias(
    case_factory,  # type: ignore[no-untyped-def]
    minimum_width_bias_hz: float,
) -> None:
    case = case_factory()
    outcome = run_acceptance_case(case)
    scan = outcome.state.tracker_estimate.sparse_scan_history[0]
    truth = evaluate_released_scan_truth(
        case.dynamics,
        outcome.state.scan_timings[0],
        completed_release_sequence_index=scan.release_sequence_index,
    )

    assert scan.status == "success"
    assert scan.amplitude_normalized_rmse is not None
    assert scan.amplitude_normalized_rmse > 0.0
    assert (
        abs(scan.fitted_fwhm_hz - truth.resonances[0].fwhm_hz)
        > minimum_width_bias_hz
    )


@pytest.mark.parametrize("center", [-2.87e9, -0.0, 0.0, 2.87e9])
def test_asynchronous_q_preserves_finite_sign_and_signed_zero(center: float) -> None:
    expected = center / 1.0e6
    identity = make_composite_identity(fast_center_hz=center, live_q=expected)
    assert identity.live_q == expected
    if center == 0.0:
        assert str(identity.live_q).startswith("-") == str(center).startswith("-")


def test_sparse_block_is_indivisible_and_uses_frozen_geometry() -> None:
    outcome = run_acceptance_case(exact_static_case())
    scan = outcome.state.tracker_estimate.sparse_scan_history[0]

    assert tuple(query.offset_multiplier for query in scan.queries) == (
        0.5,
        -1.0,
        0.0,
        1.0,
        -0.5,
    )
    assert all(
        query.frozen_fast_center_hz == scan.queries[0].frozen_fast_center_hz
        and query.frozen_prior_fwhm_hz == scan.queries[0].frozen_prior_fwhm_hz
        for query in scan.queries
    )
    assert tuple(query.expected_sequence_index for query in scan.queries) == tuple(
        range(
            scan.queries[0].expected_sequence_index,
            scan.queries[0].expected_sequence_index + 5,
        )
    )


def test_prospective_geometry_rejects_late_invalid_identity_before_run() -> None:
    outcome = run_acceptance_case(exact_static_case())
    calibration = outcome.state.calibration
    assert calibration is not None
    late = calibration.identities[-1]
    invalid = replace(
        calibration,
        identities=(
            *calibration.identities[:-1],
            replace(
                late,
                calibration_cell_lower_hz=late.calibration_center_hz - 600_000.0,
                calibration_cell_upper_hz=late.calibration_center_hz + 600_000.0,
            ),
        ),
    )
    source = invalid.source
    tracker = SparseLinewidthCompositeTracker(SparseLinewidthConfiguration())
    metadata = TwoPointRunMetadata(
        tracker_clock_id=source.clock_mapping.tracker_clock_id,
        current_sequence_index=source.availability_sequence_index,
        current_timestamp_s=source.availability_timestamp_s,
        nominal_photon_rate_hz=source.fluorescence_provenance.nominal_photon_rate_hz,
        frequency_overhead_s=source.source_frequency_overhead_s,
        fluorescence_quantity="normalized_fluorescence",
    )

    with pytest.raises(SparseLinewidthResetError) as raised:
        tracker.reset(
            metadata,
            invalid,
            TwoPointBudgetCeiling(100, None, None, None),
            seed=31,
        )
    assert raised.value.code == "invalid_base_sparse_geometry"
    assert "calibration_cell_violation" in str(raised.value)


def test_due_geometry_stops_cleanly_before_reserving_sparse_block() -> None:
    outcome = run_acceptance_case(due_geometry_stop_case())

    assert outcome.kind == "geometry_stopped"
    assert outcome.state.phase == "geometry_stopped"
    assert outcome.state.tracker_estimate.accepted_observations == 16
    assert outcome.state.tracker_estimate.incomplete_sparse_scan is None
    assert outcome.diagnostic.failure_code == "calibration_cell_violation"
    assert outcome.resources.tracking_resources.observations == 16


def test_unaffordable_five_point_block_charges_none_of_the_block() -> None:
    outcome = run_acceptance_case(unaffordable_sparse_block_case())

    assert outcome.kind == "budget_stopped"
    estimate = outcome.state.tracker_estimate
    assert estimate.accepted_observations == 16
    assert estimate.completed_sparse_scans == 0
    assert estimate.incomplete_sparse_scan is None
    assert outcome.resources.accepted_sparse_observations == ()
    assert outcome.resources.tracking_resources.observations == 16


def test_truth_timing_rejects_noninteger_completed_release_index() -> None:
    timing = SparseLinewidthEvaluatorScanTiming(
        0, "r0", (1.0, 2.0, 3.0, 4.0, 5.0), 3.0, 3.0, 4, 5.5
    )
    with pytest.raises(TypeError, match="completed_release_sequence_index"):
        evaluate_released_scan_truth(
            StationaryDynamics(exact_static_case().source_snapshot),
            replace(timing, release_sequence_index=4),  # accepted exact record
            completed_release_sequence_index=True,  # type: ignore[arg-type]
        )
