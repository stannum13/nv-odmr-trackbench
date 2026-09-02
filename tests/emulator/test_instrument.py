from __future__ import annotations

import time
from dataclasses import fields

import numpy as np
import pytest

from odmr_bench.dynamics import LinearCenterDrift, SpectralSnapshot, StationaryDynamics
from odmr_bench.emulator import (
    EmpiricalResidualNoise,
    GaussianNoise,
    NoiseResult,
    PoissonNoise,
)
from odmr_bench.models import Baseline, Resonance


def _snapshot(
    *, baseline_intercept: float = 1.0, drifting_line: bool = False
) -> SpectralSnapshot:
    return SpectralSnapshot(
        baseline=Baseline(intercept=baseline_intercept, reference_hz=2.87e9),
        resonances=tuple(
            Resonance(
                resonance_id=f"nv-{index}",
                center_hz=2.86e9 + index * 2.0e6,
                fwhm_hz=1.0 if drifting_line and index == 0 else 1.0e6,
                amplitude=1.0 if drifting_line and index == 0 else 0.0,
                eta=0.5,
            )
            for index in range(8)
        ),
    )


def test_query_uses_midpoint_truth_and_end_timestamp_without_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    dynamics = LinearCenterDrift(
        _snapshot(drifting_line=True), center_slew_hz_per_s=1.0
    )
    instrument = ODMRInstrument(
        dynamics=dynamics,
        noise=GaussianNoise(stddev_at_1s=0.0),
        nominal_photon_rate_hz=100.0,
        frequency_overhead_s=0.25,
        seed=3,
    )
    monkeypatch.setattr(time, "sleep", lambda *_: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(time, "time", lambda: (_ for _ in ()).throw(AssertionError))

    # A center is observed through its drifting Lorentzian.  This makes the
    # expected fluorescence distinguish midpoint sampling from start/end use.
    first = instrument.query(2.86e9, 0.5)

    assert first.sequence_index == 0
    assert first.timestamp_s == pytest.approx(0.75)
    assert first.fluorescence == pytest.approx(1.0 - 1.0 / (1.0 + 4.0 * 0.5**2))
    assert first.expected_photons == pytest.approx(100.0 * first.fluorescence * 0.5)
    assert first.nominal_exposure_photons == pytest.approx(50.0)
    assert instrument.virtual_time_s == pytest.approx(0.75)
    assert instrument.resources.virtual_elapsed_time_s == pytest.approx(0.75)

    second = instrument.query(2.86e9, 0.5)
    assert second.sequence_index == 1
    assert second.timestamp_s == pytest.approx(1.5)
    assert instrument.virtual_time_s == pytest.approx(1.5)


def test_identical_seeded_queries_reproduce_observations_and_resources() -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    kwargs = {
        "dynamics": StationaryDynamics(_snapshot(baseline_intercept=0.8)),
        "noise": PoissonNoise(),
        "nominal_photon_rate_hz": 500.0,
        "frequency_overhead_s": 0.01,
        "seed": 41,
    }
    first = ODMRInstrument(**kwargs)
    second = ODMRInstrument(**kwargs)
    queries = [(2.86e9, 0.02), (2.87e9, 0.04), (2.88e9, 0.01)]

    assert [first.query(*query) for query in queries] == [
        second.query(*query) for query in queries
    ]
    assert first.resources == second.resources


def test_different_seeds_change_stochastic_results() -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    common = {
        "dynamics": StationaryDynamics(_snapshot(baseline_intercept=0.8)),
        "noise": PoissonNoise(),
        "nominal_photon_rate_hz": 1_000.0,
    }
    first = ODMRInstrument(**common, seed=1)
    second = ODMRInstrument(**common, seed=2)

    assert first.query(2.86e9, 0.1).realized_photons != second.query(
        2.86e9, 0.1
    ).realized_photons


def test_estimator_view_omits_hidden_truth() -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    observation = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=PoissonNoise(),
        nominal_photon_rate_hz=100.0,
        seed=2,
    ).query(2.86e9, 0.1)

    estimator_view = observation.estimator_view()
    assert "expected_photons" not in {field.name for field in fields(estimator_view)}
    assert not hasattr(estimator_view, "expected_photons")
    assert not hasattr(estimator_view, "snapshot")


def test_poisson_counts_match_returned_normalized_fluorescence() -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    observation = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot(baseline_intercept=0.75)),
        noise=PoissonNoise(),
        nominal_photon_rate_hz=100.0,
        seed=29,
    ).query(2.86e9, 0.2)

    assert observation.realized_photons is not None
    assert observation.fluorescence == pytest.approx(
        observation.realized_photons / observation.nominal_exposure_photons
    )
    assert observation.expected_photons == pytest.approx(
        0.75 * observation.nominal_exposure_photons
    )


def test_empirical_replay_noise_remains_usable_by_the_instrument() -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    noise = EmpiricalResidualNoise(
        [0.1, 0.2],
        mode="replay",
        provenance={
            "source_id": "generated-fixture",
            "preparation_label": "known-reference-subtraction",
            "normalization_label": "normalized-fluorescence",
            "correlation_mode": "replay",
        },
    )
    instrument = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=noise,
        nominal_photon_rate_hz=100.0,
        seed=1,
    )

    assert [instrument.query(2.86e9, 0.1).fluorescence for _ in range(3)] == (
        pytest.approx([1.1, 1.2, 1.1])
    )


class _NegativeSpectrumDynamics:
    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        return _snapshot(baseline_intercept=-0.1)


class _InvalidNoise:
    sampling_rule = "poisson"

    def __init__(self) -> None:
        self.valid = False

    def sample(
        self,
        expected_fluorescence: float,
        nominal_rate_hz: float,
        integration_time_s: float,
        rng: np.random.Generator,
    ) -> object:
        if not self.valid:
            rng.normal()
            return object()
        count = rng.poisson(
            expected_fluorescence * nominal_rate_hz * integration_time_s
        )
        return NoiseResult(
            fluorescence=count / (nominal_rate_hz * integration_time_s),
            realized_photons=count,
        )


class _UncheckpointableCounter:
    def __init__(self) -> None:
        self.value = 0

    def __deepcopy__(self, memo: object) -> object:
        raise TypeError("cannot checkpoint counter")


class _UncheckpointableNoise:
    sampling_rule = "invalid"

    def __init__(self) -> None:
        self.counter = _UncheckpointableCounter()
        self.sample_calls = 0

    def sample(
        self,
        expected_fluorescence: float,
        nominal_rate_hz: float,
        integration_time_s: float,
        rng: np.random.Generator,
    ) -> object:
        self.sample_calls += 1
        self.counter.value += 1
        rng.normal()
        return object()


@pytest.mark.parametrize("frequency_hz,integration_time_s", [
    (0.0, 0.1),
    (np.nan, 0.1),
    (2.86e9, 0.0),
    (2.86e9, np.inf),
])
def test_invalid_query_values_are_fully_atomic(
    frequency_hz: float, integration_time_s: float
) -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    instrument = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot(baseline_intercept=0.8)),
        noise=PoissonNoise(),
        nominal_photon_rate_hz=100.0,
        seed=17,
    )
    untouched = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot(baseline_intercept=0.8)),
        noise=PoissonNoise(),
        nominal_photon_rate_hz=100.0,
        seed=17,
    )
    before = instrument.resources

    with pytest.raises((TypeError, ValueError)):
        instrument.query(frequency_hz, integration_time_s)

    assert instrument.virtual_time_s == 0.0
    assert instrument.resources == before
    assert instrument.query(2.86e9, 0.1) == untouched.query(2.86e9, 0.1)


@pytest.mark.parametrize("dynamics,noise", [
    (_NegativeSpectrumDynamics(), PoissonNoise()),
    (StationaryDynamics(_snapshot()), _InvalidNoise()),
])
def test_failed_query_preserves_clock_resources_sequence_and_rng(
    dynamics: object, noise: object
) -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    instrument = ODMRInstrument(
        dynamics=dynamics,
        noise=noise,
        nominal_photon_rate_hz=100.0,
        seed=17,
    )
    control = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=PoissonNoise(),
        nominal_photon_rate_hz=100.0,
        seed=17,
    )
    before = instrument.resources

    with pytest.raises((TypeError, ValueError)):
        instrument.query(2.86e9, 0.1)

    assert instrument.virtual_time_s == 0.0
    assert instrument.resources == before
    if isinstance(noise, _InvalidNoise):
        # The invalid strategy consumed a draw before returning invalid output.
        # Its externally selected next valid mode must still see the seed state.
        noise.valid = True
        assert instrument.query(2.86e9, 0.1) == control.query(2.86e9, 0.1)


def test_query_rejects_uncheckpointable_noise_state_before_sampling() -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    noise = _UncheckpointableNoise()
    instrument = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=noise,
        nominal_photon_rate_hz=100.0,
        seed=17,
    )

    with pytest.raises(TypeError, match="checkpointable"):
        instrument.query(2.86e9, 0.1)

    assert noise.counter.value == 0
    assert noise.sample_calls == 0
    assert instrument.virtual_time_s == 0.0
    assert instrument.resources.observations == 0


def test_constructor_requires_one_rng_source_and_valid_rates() -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    common = {
        "dynamics": StationaryDynamics(_snapshot()),
        "noise": GaussianNoise(stddev_at_1s=0.0),
        "nominal_photon_rate_hz": 100.0,
    }
    with pytest.raises(ValueError):
        ODMRInstrument(**common)
    with pytest.raises(ValueError):
        ODMRInstrument(**common, seed=1, rng=np.random.default_rng(1))
    with pytest.raises(ValueError):
        ODMRInstrument(**common, seed=1, frequency_overhead_s=-0.1)
    with pytest.raises(ValueError):
        ODMRInstrument(
            dynamics=common["dynamics"],
            noise=common["noise"],
            nominal_photon_rate_hz=0.0,
            seed=1,
        )
