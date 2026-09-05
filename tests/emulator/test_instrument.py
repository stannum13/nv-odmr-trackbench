from __future__ import annotations

import time
from dataclasses import fields

import numpy as np
import pytest

import odmr_bench.emulator.instrument as instrument_module
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


def test_instrument_exposes_exact_read_only_acquisition_configuration() -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    instrument = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=PoissonNoise(),
        nominal_photon_rate_hz=np.float64(123.5),
        frequency_overhead_s=np.float64(0.25),
        seed=3,
    )

    assert instrument.nominal_photon_rate_hz == 123.5
    assert type(instrument.nominal_photon_rate_hz) is float
    assert instrument.frequency_overhead_s == 0.25
    assert type(instrument.frequency_overhead_s) is float
    with pytest.raises(AttributeError):
        instrument.nominal_photon_rate_hz = 1.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        instrument.frequency_overhead_s = 0.0  # type: ignore[misc]


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


@pytest.mark.parametrize(
    ("noise", "expected_rule"),
    [
        (PoissonNoise(), "poisson"),
        (GaussianNoise(stddev_at_1s=0.0), "gaussian"),
    ],
)
def test_builtin_noise_observation_records_the_exact_sampling_rule(
    noise: PoissonNoise | GaussianNoise, expected_rule: str
) -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    observation = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=noise,
        nominal_photon_rate_hz=100.0,
        seed=2,
    ).query(2.86e9, 0.1)

    assert observation.sampling_rule == expected_rule


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

    def checkpoint(self) -> None:
        return None

    def restore(self, checkpoint: object) -> None:
        assert checkpoint is None

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


class _NoCheckpointNoise:
    sampling_rule = "invalid"

    def __init__(self) -> None:
        self.sample_calls = 0

    def sample(
        self,
        expected_fluorescence: float,
        nominal_rate_hz: float,
        integration_time_s: float,
        rng: np.random.Generator,
    ) -> object:
        self.sample_calls += 1
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


def test_query_rejects_noise_without_checkpoint_protocol_before_sampling() -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    noise = _NoCheckpointNoise()
    instrument = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=noise,
        nominal_photon_rate_hz=100.0,
        seed=17,
    )

    with pytest.raises(TypeError, match="checkpoint/restore"):
        instrument.query(2.86e9, 0.1)

    assert noise.sample_calls == 0
    assert instrument.virtual_time_s == 0.0
    assert instrument.resources.observations == 0


class _AliasedMutableNoise:
    sampling_rule = "aliased-test"

    def __init__(self, *, fail: bool) -> None:
        self.fail = fail
        self.left = [0]
        self.right = self.left
        self.cursor = 0

    def checkpoint(self) -> tuple[int, tuple[int, ...]]:
        return self.cursor, tuple(self.left)

    def restore(self, checkpoint: object) -> None:
        cursor, values = checkpoint  # type: ignore[misc]
        self.cursor = cursor
        self.left[:] = values

    def sample(
        self,
        expected_fluorescence: float,
        nominal_rate_hz: float,
        integration_time_s: float,
        rng: np.random.Generator,
    ) -> object:
        self.cursor += 1
        self.left[0] += 1
        rng.normal()
        if self.fail:
            return object()
        return NoiseResult(fluorescence=float(self.left[0]))


class _RecoveringDynamics:
    def __init__(self) -> None:
        self.fail = True

    def snapshot_at(self, timestamp_s: float) -> SpectralSnapshot:
        if self.fail and timestamp_s > 0.0:
            raise RuntimeError("transient dynamics failure")
        return _snapshot()


@pytest.mark.parametrize(
    "failure_phase",
    [
        "dynamics",
        "spectrum_shape",
        "spectrum_raises",
        "noise",
        "observation",
        "ledger",
    ],
)
def test_failure_phases_restore_all_query_state_and_next_seeded_result(
    failure_phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from odmr_bench.emulator.instrument import ODMRInstrument

    noise = _AliasedMutableNoise(fail=failure_phase == "noise")
    dynamics: object = (
        _RecoveringDynamics()
        if failure_phase == "dynamics"
        else StationaryDynamics(_snapshot())
    )
    instrument = ODMRInstrument(
        dynamics=dynamics,
        noise=noise,
        nominal_photon_rate_hz=100.0,
        seed=17,
    )
    control = ODMRInstrument(
        dynamics=StationaryDynamics(_snapshot()),
        noise=_AliasedMutableNoise(fail=False),
        nominal_photon_rate_hz=100.0,
        seed=17,
    )
    before_resources = instrument.resources
    externally_held_state = noise.left
    original_spectrum = instrument_module.multi_resonance_spectrum
    original_observation = instrument_module.InstrumentObservation
    original_record = instrument_module.ResourceLedger.record

    if failure_phase == "spectrum_shape":
        monkeypatch.setattr(
            instrument_module,
            "multi_resonance_spectrum",
            lambda *_args, **_kwargs: np.asarray([1.0, 1.0]),
        )
    elif failure_phase == "spectrum_raises":
        def raise_from_spectrum(*_args: object, **_kwargs: object) -> np.ndarray:
            raise RuntimeError("spectrum evaluation failure")

        monkeypatch.setattr(
            instrument_module, "multi_resonance_spectrum", raise_from_spectrum
        )
    elif failure_phase == "observation":
        def raise_from_observation(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("observation construction failure")

        monkeypatch.setattr(
            instrument_module, "InstrumentObservation", raise_from_observation
        )
    elif failure_phase == "ledger":
        def raise_from_ledger(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("ledger failure")

        monkeypatch.setattr(
            instrument_module.ResourceLedger, "record", raise_from_ledger
        )

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        instrument.query(2.86e9, 0.1)

    assert instrument.virtual_time_s == 0.0
    assert instrument.resources == before_resources
    assert noise.left is externally_held_state
    assert noise.right is externally_held_state
    assert noise.left == [0]
    assert noise.cursor == 0

    if failure_phase == "dynamics":
        assert isinstance(dynamics, _RecoveringDynamics)
        dynamics.fail = False
    if failure_phase == "noise":
        noise.fail = False
    monkeypatch.setattr(
        instrument_module, "multi_resonance_spectrum", original_spectrum
    )
    monkeypatch.setattr(
        instrument_module, "InstrumentObservation", original_observation
    )
    monkeypatch.setattr(instrument_module.ResourceLedger, "record", original_record)

    assert instrument.query(2.86e9, 0.1) == control.query(2.86e9, 0.1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dynamics", object()),
        ("noise", object()),
        ("nominal_photon_rate_hz", np.nan),
        ("nominal_photon_rate_hz", np.inf),
        ("frequency_overhead_s", np.nan),
        ("frequency_overhead_s", np.inf),
        ("seed", True),
        ("seed", 1.5),
        ("seed", "invalid"),
        ("rng", np.random.RandomState(1)),
        ("rng", object()),
    ],
)
def test_constructor_rejects_invalid_dependencies_and_rng_configuration(
    field: str, value: object
) -> None:
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
    invalid = dict(common)
    invalid[field] = value
    if field == "rng":
        invalid["rng"] = value
    else:
        invalid["seed"] = value if field == "seed" else 1
    with pytest.raises((TypeError, ValueError)):
        ODMRInstrument(**invalid)
