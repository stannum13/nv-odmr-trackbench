"""Behavioral tests for the causal warm-started full-sweep estimator."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

from odmr_bench.estimators import (
    CompleteSweep,
    FitConfiguration,
    FitInitialGuess,
    InitializationDiagnostics,
    SpectrumFitResult,
    WarmStartedFullSweepEstimator,
)
from odmr_bench.estimators.preparation import (
    FitPreflight,
    WarmStartPreparation,
    start_independent_preflight,
)
from odmr_bench.models import Baseline, Resonance


def _configuration() -> FitConfiguration:
    return FitConfiguration()


def _sweep(
    *,
    last_sequence_index: int | None = None,
    last_timestamp_s: float | None = None,
    total_integration_time_s: float | None = None,
    total_nominal_exposure_photons: float | None = None,
) -> CompleteSweep:
    return CompleteSweep(
        frequency_hz=np.linspace(2.74e9, 3.02e9, 64),
        fluorescence=np.linspace(0.98, 1.02, 64),
        last_sequence_index=last_sequence_index,
        last_timestamp_s=last_timestamp_s,
        total_integration_time_s=total_integration_time_s,
        total_nominal_exposure_photons=total_nominal_exposure_photons,
    )


def _diagnostics(source: str) -> InitializationDiagnostics:
    return InitializationDiagnostics(
        source=source,
        candidate_count=8 if source == "detected" else 0,
        selected_indices=tuple(range(8)) if source == "detected" else (),
        used_fallback=source == "fallback",
        messages=("test result",),
    )


def _resonances() -> tuple[Resonance, ...]:
    return tuple(
        Resonance(
            resonance_id=f"r{index}",
            center_hz=2.76e9 + index * 34e6,
            fwhm_hz=1.0e6,
            amplitude=0.02,
            eta=0.5,
        )
        for index in range(8)
    )


def _guess() -> FitInitialGuess:
    return FitInitialGuess(_resonances(), Baseline(1.0, 2.88e9))


def _optimizer_result(start_kind: str, code: str | None) -> SpectrumFitResult:
    source = "user" if start_kind == "warm" else "detected"
    success = code is None
    return SpectrumFitResult(
        success=success,
        failure_code=code,
        model_kind="pseudo_voigt",
        baseline_degree=1,
        resonance_estimates=_resonances() if success else (),
        baseline_estimate=Baseline(1.0, 2.88e9) if success else None,
        diagnostics=_diagnostics(source),
        initial_guess=_guess(),
        uncertainty=None,
        uncertainty_reason=(
            "uncertainty not calculated" if success else "attempt failed"
        ),
        scipy_status=0 if code == "optimization_failed" else 1,
        scipy_message="stopped",
        nfev=4 if success else 3,
        cost=0.01,
        residual_rmse=0.02,
        residual_scale=0.1,
        degrees_of_freedom=30,
        jacobian_rank=34 if success else (10 if code == "quality_failed" else None),
    )


def _cold_success() -> SpectrumFitResult:
    return _optimizer_result("cold", None)


def _warm_success() -> SpectrumFitResult:
    return _optimizer_result("warm", None)


def _cold_failure(code: str = "optimization_failed") -> SpectrumFitResult:
    return _optimizer_result("cold", code)


def _warm_failure(code: str = "optimization_failed") -> SpectrumFitResult:
    return _optimizer_result("warm", code)


def _cold_initialization_failure() -> SpectrumFitResult:
    return SpectrumFitResult(
        success=False,
        failure_code="initialization_failed",
        model_kind="pseudo_voigt",
        baseline_degree=1,
        resonance_estimates=(),
        baseline_estimate=None,
        diagnostics=_diagnostics("none"),
        initial_guess=None,
        uncertainty=None,
        uncertainty_reason="initialization failed",
        scipy_status=None,
        scipy_message=None,
        nfev=0,
        cost=None,
        residual_rmse=None,
        residual_scale=0.1,
        degrees_of_freedom=30,
        jacobian_rank=None,
    )


def _preflight_failure(code: str) -> SpectrumFitResult:
    return SpectrumFitResult(
        success=False,
        failure_code=code,
        model_kind="pseudo_voigt",
        baseline_degree=1,
        resonance_estimates=(),
        baseline_estimate=None,
        diagnostics=_diagnostics("none"),
        initial_guess=None,
        uncertainty=None,
        uncertainty_reason="preflight failed",
        scipy_status=None,
        scipy_message=None,
        nfev=0,
        cost=None,
        residual_rmse=None,
        residual_scale=None,
        degrees_of_freedom=0 if code == "insufficient_samples" else 30,
        jacobian_rank=None,
    )


def _ready() -> FitPreflight:
    result = start_independent_preflight(_sweep(), _configuration())
    assert isinstance(result, FitPreflight)
    return result


def test_warm_estimator_is_public() -> None:
    assert WarmStartedFullSweepEstimator is not None


def test_constructor_starts_empty_and_exposes_immutable_configuration() -> None:
    configuration = _configuration()
    estimator = WarmStartedFullSweepEstimator(
        configuration,
        retry_cold_on_warm_failure=np.bool_(False),
        max_warm_start_age_updates=np.int64(2),
    )

    assert estimator.configuration is configuration
    assert estimator.latest is None
    assert estimator.latest_success is None
    assert estimator.history == ()
    assert isinstance(estimator.history, tuple)
    with pytest.raises(AttributeError):
        estimator.configuration = _configuration()  # type: ignore[misc]


@pytest.mark.parametrize("configuration", [None, object(), "config"])
def test_constructor_rejects_invalid_configuration(configuration: object) -> None:
    with pytest.raises(TypeError, match="configuration"):
        WarmStartedFullSweepEstimator(configuration)  # type: ignore[arg-type]


@pytest.mark.parametrize("retry", [0, 1, "yes"])
def test_constructor_rejects_nonboolean_retry(retry: object) -> None:
    with pytest.raises(TypeError, match="retry_cold_on_warm_failure"):
        WarmStartedFullSweepEstimator(
            _configuration(), retry_cold_on_warm_failure=retry  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "age", [True, False, 0, -1, 1.5, np.inf, "one"]
)
def test_constructor_rejects_invalid_maximum_age(age: object) -> None:
    with pytest.raises((TypeError, ValueError), match="max_warm_start_age_updates"):
        WarmStartedFullSweepEstimator(
            _configuration(), max_warm_start_age_updates=age  # type: ignore[arg-type]
        )


def test_reset_clears_empty_state_and_retains_settings() -> None:
    configuration = _configuration()
    estimator = WarmStartedFullSweepEstimator(
        configuration,
        retry_cold_on_warm_failure=False,
        max_warm_start_age_updates=2,
    )

    estimator.reset()

    assert estimator.configuration is configuration
    assert estimator.latest is None
    assert estimator.latest_success is None
    assert estimator.history == ()


def _patch_ready(monkeypatch: pytest.MonkeyPatch) -> FitPreflight:
    ready = _ready()
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.start_independent_preflight",
        lambda sweep, configuration: ready,
    )
    return ready


def test_first_success_is_one_cold_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ready(monkeypatch)
    success = _cold_success()
    guesses: list[FitInitialGuess | None] = []

    def fake_fit(
        sweep: CompleteSweep,
        configuration: FitConfiguration,
        initial_guess: FitInitialGuess | None,
    ) -> SpectrumFitResult:
        guesses.append(initial_guess)
        return success

    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", fake_fit)
    estimator = WarmStartedFullSweepEstimator(_configuration())

    result = estimator.update_sweep(_sweep())

    assert guesses == [None]
    assert result.update_index == 0
    assert result.warm_start_disposition == "no_successful_prior"
    assert tuple(attempt.start_kind for attempt in result.attempts) == ("cold",)
    assert result.current_fit is success
    assert result.active_fit is success
    assert result.active_source_update_index == 0
    assert result.estimate_age_submitted_observations == 0
    assert estimator.latest is result
    assert estimator.latest_success is result


def test_repeated_cold_failures_never_create_a_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    failures = iter((_cold_failure(), _cold_initialization_failure()))
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: next(failures),
    )
    estimator = WarmStartedFullSweepEstimator(_configuration())

    first = estimator.update_sweep(_sweep())
    second = estimator.update_sweep(_sweep())

    assert first.warm_start_disposition == "no_successful_prior"
    assert second.warm_start_disposition == "no_successful_prior"
    assert first.active_fit is None
    assert second.active_fit is None
    assert estimator.latest_success is None


@pytest.mark.parametrize("code", ["insufficient_samples", "uninformative_sweep"])
@pytest.mark.parametrize("with_prior", [False, True])
def test_preflight_failure_bypasses_preparation_and_fitter(
    monkeypatch: pytest.MonkeyPatch, code: str, with_prior: bool
) -> None:
    estimator = WarmStartedFullSweepEstimator(_configuration())
    prior = _cold_success()
    if with_prior:
        _patch_ready(monkeypatch)
        monkeypatch.setattr(
            "odmr_bench.estimators.warm_sweep.fit_spectrum",
            lambda sweep, configuration, initial_guess: prior,
        )
        seeded = estimator.update_sweep(_sweep())

    failure = _preflight_failure(code)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.start_independent_preflight",
        lambda sweep, configuration: failure,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("preparation/fitter must be bypassed")

    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start", forbidden
    )
    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", forbidden)

    result = estimator.update_sweep(_sweep())

    assert result.warm_start_disposition == "not_applicable_preflight"
    assert result.current_fit is failure
    assert tuple(attempt.start_kind for attempt in result.attempts) == ("preflight",)
    if with_prior:
        assert result.active_fit is prior
        assert result.active_fit is seeded.active_fit
        assert result.active_source_update_index == 0
        assert result.estimate_age_submitted_observations == 64
        assert estimator.latest_success is seeded
    else:
        assert result.active_fit is None
        assert estimator.latest_success is None


@pytest.mark.parametrize(
    ("maximum", "expected_disposition", "expected_guess"),
    [(1, "rejected_age", None), (2, "used", "warm"), (None, "used", "warm")],
)
def test_source_age_limit_uses_strict_update_age_boundary(
    monkeypatch: pytest.MonkeyPatch,
    maximum: int | None,
    expected_disposition: str,
    expected_guess: str | None,
) -> None:
    _patch_ready(monkeypatch)
    prepared = WarmStartPreparation(_guess(), None, None)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: prepared,
    )
    final_fit = _cold_success() if expected_guess is None else _warm_success()
    fits = iter((_cold_success(), _warm_failure(), final_fit))
    guesses: list[FitInitialGuess | None] = []

    def fake_fit(
        sweep: CompleteSweep,
        configuration: FitConfiguration,
        initial_guess: FitInitialGuess | None,
    ) -> SpectrumFitResult:
        guesses.append(initial_guess)
        return next(fits)

    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", fake_fit)
    estimator = WarmStartedFullSweepEstimator(
        _configuration(),
        retry_cold_on_warm_failure=False,
        max_warm_start_age_updates=maximum,
    )
    estimator.update_sweep(_sweep())
    estimator.update_sweep(_sweep())

    result = estimator.update_sweep(_sweep())

    assert result.warm_start_disposition == expected_disposition
    if expected_guess is None:
        assert guesses[-1] is None
        assert result.warm_start_rejection_code == "age_limit_exceeded"
    else:
        assert guesses[-1] is prepared.guess
        assert result.attempts[0].warm_source_update_index == 0


def test_compatible_latest_success_seeds_exact_prepared_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    cold = _cold_success()
    warm = _warm_success()
    prepared = WarmStartPreparation(_guess(), None, None)
    prior_seen: list[SpectrumFitResult] = []
    guesses: list[FitInitialGuess | None] = []
    fits = iter((cold, warm))

    def fake_prepare(
        prior: SpectrumFitResult,
        configuration: FitConfiguration,
        preflight: FitPreflight,
    ) -> WarmStartPreparation:
        prior_seen.append(prior)
        return prepared

    def fake_fit(
        sweep: CompleteSweep,
        configuration: FitConfiguration,
        initial_guess: FitInitialGuess | None,
    ) -> SpectrumFitResult:
        guesses.append(initial_guess)
        return next(fits)

    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start", fake_prepare
    )
    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", fake_fit)
    estimator = WarmStartedFullSweepEstimator(_configuration())
    first = estimator.update_sweep(_sweep())
    second = estimator.update_sweep(_sweep())

    assert prior_seen == [cold]
    assert prior_seen[0] is first.active_fit
    assert guesses == [None, prepared.guess]
    assert second.warm_start_disposition == "used"
    assert second.attempts[0].warm_source_update_index == 0


def test_compatibility_rejection_copies_reason_and_failed_cold_keeps_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    rejection = WarmStartPreparation(
        None, "center_outside_sweep", "prior center outside current sweep"
    )
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: rejection,
    )
    prior = _cold_success()
    failure = _cold_failure("quality_failed")
    fits = iter((prior, failure))
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: next(fits),
    )
    estimator = WarmStartedFullSweepEstimator(_configuration())
    seeded = estimator.update_sweep(_sweep())

    result = estimator.update_sweep(_sweep())

    assert result.warm_start_disposition == "rejected_compatibility"
    assert result.warm_start_rejection_code == rejection.rejection_code
    assert result.warm_start_message == rejection.message
    assert result.current_fit is failure
    assert result.active_fit is seeded.active_fit
    assert result.active_source_update_index == 0
    assert result.estimate_age_submitted_observations == 64
    assert result.estimate_age_sequence_indices is None
    assert result.estimate_age_s is None


@pytest.mark.parametrize("rejection_kind", ["age", "compatibility"])
def test_rejected_cold_success_becomes_next_warm_source(
    monkeypatch: pytest.MonkeyPatch, rejection_kind: str
) -> None:
    ready = _patch_ready(monkeypatch)
    prepared = WarmStartPreparation(_guess(), None, None)
    source_priors: list[SpectrumFitResult] = []
    preparations: Iterator[WarmStartPreparation]
    if rejection_kind == "compatibility":
        preparations = iter(
            (
                WarmStartPreparation(None, "center_outside_sweep", "outside"),
                prepared,
            )
        )
        maximum = None
    else:
        preparations = iter((prepared,))
        maximum = 1

    def fake_prepare(
        prior: SpectrumFitResult,
        configuration: FitConfiguration,
        preflight: FitPreflight,
    ) -> WarmStartPreparation:
        source_priors.append(prior)
        return next(preparations)

    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start", fake_prepare
    )
    first_success = _cold_success()
    rejected_success = _cold_success()
    final_success = _warm_success()
    fit_values: list[SpectrumFitResult]
    if rejection_kind == "age":
        preflight_failure = _preflight_failure("uninformative_sweep")
        preflights = iter((ready, preflight_failure, ready, ready))
        monkeypatch.setattr(
            "odmr_bench.estimators.warm_sweep.start_independent_preflight",
            lambda sweep, configuration: next(preflights),
        )
        fit_values = [first_success, rejected_success, final_success]
    else:
        fit_values = [first_success, rejected_success, final_success]
    fits = iter(fit_values)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: next(fits),
    )
    estimator = WarmStartedFullSweepEstimator(
        _configuration(), max_warm_start_age_updates=maximum
    )
    estimator.update_sweep(_sweep())
    if rejection_kind == "age":
        estimator.update_sweep(_sweep())
    rejected = estimator.update_sweep(_sweep())
    promoted = estimator.update_sweep(_sweep())

    assert rejected.current_fit is rejected_success
    assert promoted.warm_start_disposition == "used"
    assert promoted.attempts[0].warm_source_update_index == rejected.update_index
    assert source_priors[-1] is rejected_success


@pytest.mark.parametrize("failure_code", ["optimization_failed", "quality_failed"])
def test_eligible_warm_failure_retries_cold_once_on_same_sweep(
    monkeypatch: pytest.MonkeyPatch, failure_code: str
) -> None:
    _patch_ready(monkeypatch)
    prepared = WarmStartPreparation(_guess(), None, None)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: prepared,
    )
    warm_failure = _warm_failure(failure_code)
    cold_recovery = _cold_success()
    fits = iter((_cold_success(), warm_failure, cold_recovery))
    calls: list[tuple[CompleteSweep, FitInitialGuess | None]] = []

    def fake_fit(
        sweep: CompleteSweep,
        configuration: FitConfiguration,
        initial_guess: FitInitialGuess | None,
    ) -> SpectrumFitResult:
        calls.append((sweep, initial_guess))
        return next(fits)

    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", fake_fit)
    estimator = WarmStartedFullSweepEstimator(_configuration())
    estimator.update_sweep(_sweep())
    submitted = _sweep(
        total_integration_time_s=0.75,
        total_nominal_exposure_photons=1234.0,
    )

    recovered = estimator.update_sweep(submitted)

    assert len(calls) == 3
    assert calls[-2][0] is submitted
    assert calls[-2][1] is prepared.guess
    assert calls[-1][0] is submitted
    assert calls[-1][1] is None
    assert tuple(attempt.start_kind for attempt in recovered.attempts) == (
        "warm",
        "cold",
    )
    assert recovered.attempts[0].warm_source_update_index == 0
    assert recovered.attempts[1].warm_source_update_index is None
    assert recovered.attempts[0].fit is warm_failure
    assert recovered.current_fit is cold_recovery
    assert recovered.active_fit is cold_recovery
    assert recovered.total_nfev == warm_failure.nfev + cold_recovery.nfev
    assert recovered.observation_count == 64
    assert recovered.cumulative_observation_count == 128
    assert recovered.total_integration_time_s == 0.75
    assert recovered.total_nominal_exposure_photons == 1234.0


def test_retry_disabled_keeps_failed_warm_and_identical_prior_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    prepared = WarmStartPreparation(_guess(), None, None)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: prepared,
    )
    warm_failure = _warm_failure()
    fits = iter((_cold_success(), warm_failure))
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: next(fits),
    )
    estimator = WarmStartedFullSweepEstimator(
        _configuration(), retry_cold_on_warm_failure=False
    )
    source = estimator.update_sweep(_sweep())

    failed = estimator.update_sweep(_sweep())

    assert tuple(attempt.start_kind for attempt in failed.attempts) == ("warm",)
    assert failed.current_fit is warm_failure
    assert failed.active_fit is source.active_fit
    assert failed.active_source_update_index == 0
    assert (
        failed.active_source_update_index
        == failed.attempts[0].warm_source_update_index
    )
    assert estimator.latest_success is source


def test_warm_success_stops_without_hindsight_cold_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    prepared = WarmStartPreparation(_guess(), None, None)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: prepared,
    )
    fits = iter((_cold_success(), _warm_success()))
    calls = 0

    def fake_fit(
        sweep: CompleteSweep,
        configuration: FitConfiguration,
        initial_guess: FitInitialGuess | None,
    ) -> SpectrumFitResult:
        nonlocal calls
        calls += 1
        return next(fits)

    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", fake_fit)
    estimator = WarmStartedFullSweepEstimator(_configuration())
    estimator.update_sweep(_sweep())

    result = estimator.update_sweep(_sweep())

    assert calls == 2
    assert tuple(attempt.start_kind for attempt in result.attempts) == ("warm",)


@pytest.mark.parametrize(
    "invalid_warm",
    [
        _cold_initialization_failure(),
        _preflight_failure("insufficient_samples"),
        _preflight_failure("uninformative_sweep"),
    ],
)
def test_illegal_warm_outcome_fails_atomically_without_retry_or_relabel(
    monkeypatch: pytest.MonkeyPatch, invalid_warm: SpectrumFitResult
) -> None:
    _patch_ready(monkeypatch)
    prepared = WarmStartPreparation(_guess(), None, None)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: prepared,
    )
    fits = iter((_cold_success(), invalid_warm))
    calls = 0

    def fake_fit(
        sweep: CompleteSweep,
        configuration: FitConfiguration,
        initial_guess: FitInitialGuess | None,
    ) -> SpectrumFitResult:
        nonlocal calls
        calls += 1
        return next(fits)

    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", fake_fit)
    estimator = WarmStartedFullSweepEstimator(_configuration())
    source = estimator.update_sweep(_sweep())

    with pytest.raises(ValueError, match="warm attempts"):
        estimator.update_sweep(_sweep())

    assert calls == 2
    assert estimator.history == (source,)
    assert estimator.latest is source
    assert estimator.latest_success is source


def test_sequence_endpoints_require_nonoverlap_and_allow_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: _cold_failure(),
    )
    estimator = WarmStartedFullSweepEstimator(_configuration())

    first = estimator.update_sweep(_sweep(last_sequence_index=63))
    second = estimator.update_sweep(_sweep(last_sequence_index=127))
    before = estimator.history
    with pytest.raises(ValueError, match="overlap"):
        estimator.update_sweep(_sweep(last_sequence_index=126))
    gap = estimator.update_sweep(_sweep(last_sequence_index=191))

    assert (first.first_sequence_index, first.last_sequence_index) == (0, 63)
    assert (second.first_sequence_index, second.last_sequence_index) == (64, 127)
    assert estimator.history[:2] == before
    assert (gap.first_sequence_index, gap.last_sequence_index) == (128, 191)


def test_negative_first_sequence_index_rejects_before_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = WarmStartedFullSweepEstimator(_configuration())

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("helper must not run")

    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.start_independent_preflight", forbidden
    )
    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", forbidden)

    with pytest.raises(ValueError, match="first_sequence_index"):
        estimator.update_sweep(_sweep(last_sequence_index=62))
    assert estimator.history == ()


@pytest.mark.parametrize("bad_timestamp", [1.0, 0.5])
def test_timestamps_must_strictly_increase_without_state_change(
    monkeypatch: pytest.MonkeyPatch, bad_timestamp: float
) -> None:
    _patch_ready(monkeypatch)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: _cold_failure(),
    )
    estimator = WarmStartedFullSweepEstimator(_configuration())
    accepted = estimator.update_sweep(_sweep(last_timestamp_s=1.0))

    with pytest.raises(ValueError, match="timestamp"):
        estimator.update_sweep(_sweep(last_timestamp_s=bad_timestamp))

    assert estimator.history == (accepted,)
    later = estimator.update_sweep(_sweep(last_timestamp_s=1.5))
    assert later.update_index == 1


@pytest.mark.parametrize(
    ("first_sequence", "second_sequence", "first_time", "second_time"),
    [
        (63, None, None, None),
        (None, 127, None, None),
        (None, None, 1.0, None),
        (None, None, None, 1.5),
    ],
)
def test_endpoint_availability_mode_is_independently_consistent(
    monkeypatch: pytest.MonkeyPatch,
    first_sequence: int | None,
    second_sequence: int | None,
    first_time: float | None,
    second_time: float | None,
) -> None:
    _patch_ready(monkeypatch)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: _cold_failure(),
    )
    estimator = WarmStartedFullSweepEstimator(_configuration())
    accepted = estimator.update_sweep(
        _sweep(
            last_sequence_index=first_sequence,
            last_timestamp_s=first_time,
        )
    )

    with pytest.raises(ValueError, match="availability"):
        estimator.update_sweep(
            _sweep(
                last_sequence_index=second_sequence,
                last_timestamp_s=second_time,
            )
        )

    assert estimator.history == (accepted,)


def test_success_failure_success_exposes_distinct_exact_ages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    prepared = WarmStartPreparation(_guess(), None, None)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: prepared,
    )
    source_fit = _cold_success()
    failed_fit = _warm_failure()
    current_fit = _warm_success()
    fits = iter((source_fit, failed_fit, current_fit))
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: next(fits),
    )
    estimator = WarmStartedFullSweepEstimator(
        _configuration(), retry_cold_on_warm_failure=False
    )

    first = estimator.update_sweep(
        _sweep(last_sequence_index=63, last_timestamp_s=1.0)
    )
    failed = estimator.update_sweep(
        _sweep(last_sequence_index=127, last_timestamp_s=2.5)
    )
    current = estimator.update_sweep(
        _sweep(last_sequence_index=191, last_timestamp_s=3.0)
    )

    assert (
        first.estimate_age_submitted_observations,
        first.estimate_age_sequence_indices,
        first.estimate_age_s,
    ) == (0, 0, 0.0)
    assert failed.active_fit is first.active_fit
    assert (
        failed.estimate_age_submitted_observations,
        failed.estimate_age_sequence_indices,
        failed.estimate_age_s,
    ) == (64, 64, 1.5)
    assert (
        current.estimate_age_submitted_observations,
        current.estimate_age_sequence_indices,
        current.estimate_age_s,
    ) == (0, 0, 0.0)


def test_gap_keeps_submitted_and_sequence_ages_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    prepared = WarmStartPreparation(_guess(), None, None)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: prepared,
    )
    fits = iter((_cold_success(), _warm_failure()))
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: next(fits),
    )
    estimator = WarmStartedFullSweepEstimator(
        _configuration(), retry_cold_on_warm_failure=False
    )
    estimator.update_sweep(_sweep(last_sequence_index=63))

    failed = estimator.update_sweep(_sweep(last_sequence_index=191))

    assert failed.estimate_age_submitted_observations == 64
    assert failed.estimate_age_sequence_indices == 128
    assert failed.estimate_age_s is None


def test_reset_clears_endpoint_modes_and_numbering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: _cold_failure(),
    )
    estimator = WarmStartedFullSweepEstimator(
        _configuration(),
        retry_cold_on_warm_failure=False,
        max_warm_start_age_updates=2,
    )
    estimator.update_sweep(
        _sweep(last_sequence_index=63, last_timestamp_s=1.0)
    )

    estimator.reset()
    fresh = estimator.update_sweep(_sweep())

    assert fresh.update_index == 0
    assert fresh.cumulative_observation_count == 64
    assert fresh.first_sequence_index is None
    assert fresh.last_sequence_index is None
    assert fresh.last_timestamp_s is None


def _set_clock(monkeypatch: pytest.MonkeyPatch, values: list[object]) -> None:
    samples = iter(values)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.time.process_time_ns",
        lambda: next(samples),
    )


def test_preflight_failure_cpu_intervals_are_nested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = _preflight_failure("uninformative_sweep")
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.start_independent_preflight",
        lambda sweep, configuration: failure,
    )
    _set_clock(monkeypatch, [100, 110, 140, 160])

    result = WarmStartedFullSweepEstimator(_configuration()).update_sweep(_sweep())

    assert result.attempts[0].cpu_time_s == 30e-9
    assert result.cpu_time_s == 60e-9


def test_one_cold_fit_cpu_intervals_are_nested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: _cold_success(),
    )
    _set_clock(monkeypatch, [100, 110, 120, 130, 160, 180])

    result = WarmStartedFullSweepEstimator(_configuration()).update_sweep(_sweep())

    assert result.attempts[0].cpu_time_s == 30e-9
    assert result.cpu_time_s == 80e-9


def _seed_for_timing(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fit_results: Iterator[SpectrumFitResult],
) -> WarmStartedFullSweepEstimator:
    _patch_ready(monkeypatch)
    prepared = WarmStartPreparation(_guess(), None, None)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: prepared,
    )
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: next(fit_results),
    )
    _set_clock(monkeypatch, [0, 1, 2, 3, 4, 5])
    estimator = WarmStartedFullSweepEstimator(_configuration())
    estimator.update_sweep(_sweep())
    return estimator


def test_warm_then_cold_cpu_intervals_and_attempt_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fits = iter((_cold_success(), _warm_failure(), _cold_success()))
    estimator = _seed_for_timing(monkeypatch, fit_results=fits)
    _set_clock(monkeypatch, [100, 110, 120, 130, 150, 160, 200, 220])

    result = estimator.update_sweep(_sweep())

    assert [attempt.cpu_time_s for attempt in result.attempts] == [20e-9, 40e-9]
    assert result.cpu_time_s == 120e-9


def test_update_cpu_uses_separately_converted_attempt_sum_rounding_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fits = iter((_cold_success(), _warm_failure(), _cold_success()))
    estimator = _seed_for_timing(monkeypatch, fit_results=fits)
    warm_ns = 693_945_840_953_877_919
    cold_ns = 169_434_960_463_419_660
    _set_clock(
        monkeypatch,
        [
            0,
            1,
            2,
            2,
            warm_ns + 2,
            warm_ns + 2,
            warm_ns + cold_ns + 2,
            warm_ns + cold_ns + 2,
        ],
    )

    result = estimator.update_sweep(_sweep())

    expected = math.fsum((warm_ns / 1e9, cold_ns / 1e9))
    assert expected == 863380801.4172976
    assert (warm_ns + cold_ns + 2) / 1e9 == 863380801.4172975
    assert result.cpu_time_s == expected


def test_global_clock_regression_is_not_masked_by_nonnegative_durations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    calls = 0

    def fake_fit(*args: object, **kwargs: object) -> SpectrumFitResult:
        nonlocal calls
        calls += 1
        return _cold_success()

    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", fake_fit)
    _set_clock(monkeypatch, [100, 110, 120, 105, 115, 130])
    estimator = WarmStartedFullSweepEstimator(_configuration())

    with pytest.raises(RuntimeError, match="process CPU clock moved backwards"):
        estimator.update_sweep(_sweep())

    assert calls == 0
    assert estimator.history == ()


@pytest.mark.parametrize("bad_sample", [True, np.bool_(False), 1.5, np.float64(2)])
def test_cpu_clock_requires_raw_integral_samples(
    monkeypatch: pytest.MonkeyPatch, bad_sample: object
) -> None:
    _set_clock(monkeypatch, [bad_sample])
    estimator = WarmStartedFullSweepEstimator(_configuration())

    with pytest.raises(TypeError, match="process_time_ns"):
        estimator.update_sweep(_sweep())

    assert estimator.history == ()


def test_final_timer_failure_after_fitter_work_is_atomic_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ready(monkeypatch)
    calls = 0

    def fake_fit(*args: object, **kwargs: object) -> SpectrumFitResult:
        nonlocal calls
        calls += 1
        return _cold_success()

    monkeypatch.setattr("odmr_bench.estimators.warm_sweep.fit_spectrum", fake_fit)
    samples = iter((100, 110, 120, 130, 160))

    def failing_clock() -> int:
        try:
            return next(samples)
        except StopIteration:
            raise RuntimeError("timer failed") from None

    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.time.process_time_ns", failing_clock
    )
    estimator = WarmStartedFullSweepEstimator(_configuration())
    submitted = _sweep(last_sequence_index=63, last_timestamp_s=1.0)

    with pytest.raises(RuntimeError, match="timer failed"):
        estimator.update_sweep(submitted)
    assert calls == 1
    assert estimator.history == ()

    _set_clock(monkeypatch, [200, 210, 220, 230, 260, 280])
    retried = estimator.update_sweep(submitted)
    assert retried.update_index == 0
    assert retried.first_sequence_index == 0


def _private_state(estimator: WarmStartedFullSweepEstimator) -> tuple[object, ...]:
    return (
        estimator.history,
        estimator.latest,
        estimator.latest_success,
        estimator._sequence_available,  # type: ignore[attr-defined]
        estimator._last_sequence_index,  # type: ignore[attr-defined]
        estimator._timestamp_available,  # type: ignore[attr-defined]
        estimator._last_timestamp_s,  # type: ignore[attr-defined]
        estimator._cumulative_observation_count,  # type: ignore[attr-defined]
    )


def _assert_private_state_unchanged(
    estimator: WarmStartedFullSweepEstimator, before: tuple[object, ...]
) -> None:
    after = _private_state(estimator)
    assert len(after[0]) == len(before[0])  # type: ignore[arg-type]
    assert all(
        current is previous
        for current, previous in zip(after[0], before[0], strict=True)  # type: ignore[arg-type]
    )
    assert after[1] is before[1]
    assert after[2] is before[2]
    assert after[3:] == before[3:]


def test_final_record_construction_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import odmr_bench.estimators.warm_sweep as warm_sweep_module

    _patch_ready(monkeypatch)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: _cold_success(),
    )
    _set_clock(monkeypatch, [100, 110, 120, 130, 160, 180])
    estimator = WarmStartedFullSweepEstimator(_configuration())
    before = _private_state(estimator)
    record_type = warm_sweep_module.WarmSweepEstimate
    submitted = _sweep(last_sequence_index=63, last_timestamp_s=1.0)

    def fail_record(*args: object, **kwargs: object) -> object:
        raise RuntimeError("record failed")

    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.WarmSweepEstimate", fail_record
    )
    with pytest.raises(RuntimeError, match="record failed"):
        estimator.update_sweep(submitted)
    _assert_private_state_unchanged(estimator, before)

    monkeypatch.setattr(warm_sweep_module, "WarmSweepEstimate", record_type)
    _set_clock(monkeypatch, [200, 210, 220, 230, 260, 280])
    retried = estimator.update_sweep(submitted)
    assert retried.update_index == 0
    assert retried.first_sequence_index == 0


@pytest.mark.parametrize(
    "fault_site", ["preflight", "preparation", "warm_fit", "cold_recovery"]
)
def test_helper_and_fitter_exceptions_preserve_all_causal_state_and_retry_index(
    monkeypatch: pytest.MonkeyPatch, fault_site: str
) -> None:
    ready = _patch_ready(monkeypatch)
    prepared = WarmStartPreparation(_guess(), None, None)
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: _cold_success(),
    )
    estimator = WarmStartedFullSweepEstimator(_configuration())
    source = estimator.update_sweep(
        _sweep(last_sequence_index=63, last_timestamp_s=1.0)
    )
    submitted = _sweep(last_sequence_index=127, last_timestamp_s=2.0)
    before = _private_state(estimator)

    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.start_independent_preflight",
        lambda sweep, configuration: ready,
    )
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: prepared,
    )
    if fault_site == "preflight":
        monkeypatch.setattr(
            "odmr_bench.estimators.warm_sweep.start_independent_preflight",
            lambda sweep, configuration: (_ for _ in ()).throw(
                RuntimeError("preflight failed")
            ),
        )
    elif fault_site == "preparation":
        monkeypatch.setattr(
            "odmr_bench.estimators.warm_sweep.prepare_warm_start",
            lambda prior, configuration, preflight: (_ for _ in ()).throw(
                RuntimeError("preparation failed")
            ),
        )
    elif fault_site == "warm_fit":
        monkeypatch.setattr(
            "odmr_bench.estimators.warm_sweep.fit_spectrum",
            lambda sweep, configuration, initial_guess: (_ for _ in ()).throw(
                RuntimeError("warm fit failed")
            ),
        )
    else:
        monkeypatch.setattr(
            "odmr_bench.estimators.warm_sweep.fit_spectrum",
            lambda sweep, configuration, initial_guess: (
                _warm_failure()
                if initial_guess is not None
                else (_ for _ in ()).throw(RuntimeError("cold recovery failed"))
            ),
        )

    with pytest.raises(RuntimeError, match="failed"):
        estimator.update_sweep(submitted)
    _assert_private_state_unchanged(estimator, before)

    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.start_independent_preflight",
        lambda sweep, configuration: ready,
    )
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.prepare_warm_start",
        lambda prior, configuration, preflight: prepared,
    )
    retry_fits: Iterator[SpectrumFitResult]
    if fault_site == "cold_recovery":
        retry_fits = iter((_warm_failure(), _cold_success()))
    else:
        retry_fits = iter((_warm_success(),))
    monkeypatch.setattr(
        "odmr_bench.estimators.warm_sweep.fit_spectrum",
        lambda sweep, configuration, initial_guess: next(retry_fits),
    )

    retried = estimator.update_sweep(submitted)

    assert retried.update_index == 1
    assert retried.first_sequence_index == 64
    assert estimator.history[0] is source
