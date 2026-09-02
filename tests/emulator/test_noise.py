from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from odmr_bench.emulator.noise import (
    EmpiricalResidualNoise,
    GaussianNoise,
    PoissonNoise,
)


def _provenance(correlation_mode: str) -> dict[str, str]:
    return {
        "source_id": "generated-fixture-v1",
        "preparation_label": "known-reference-subtraction",
        "normalization_label": "normalized-fluorescence",
        "correlation_mode": correlation_mode,
    }


def test_poisson_noise_returns_seeded_count_and_normalized_fluorescence() -> None:
    rate_hz = 25.0
    duration_s = 0.2
    expected_fluorescence = 0.8
    expected_count = np.random.default_rng(23).poisson(
        expected_fluorescence * rate_hz * duration_s
    )

    result = PoissonNoise().sample(
        expected_fluorescence,
        rate_hz,
        duration_s,
        np.random.default_rng(23),
    )

    assert result.realized_photons == expected_count
    assert result.fluorescence == expected_count / (rate_hz * duration_s)


def test_gaussian_noise_scales_standard_deviation_with_integration_time() -> None:
    noise = GaussianNoise(stddev_at_1s=0.4)
    expected_fluorescence = 1.2
    unit_duration = noise.sample(
        expected_fluorescence,
        10.0,
        1.0,
        np.random.default_rng(5),
    )
    quarter_duration = noise.sample(
        expected_fluorescence,
        10.0,
        0.25,
        np.random.default_rng(5),
    )

    assert unit_duration.realized_photons is None
    assert quarter_duration.realized_photons is None
    assert quarter_duration.fluorescence - expected_fluorescence == pytest.approx(
        2.0 * (unit_duration.fluorescence - expected_fluorescence)
    )


def test_zero_gaussian_noise_is_a_deterministic_control() -> None:
    result = GaussianNoise(stddev_at_1s=0.0).sample(
        0.83, 10.0, 0.2, np.random.default_rng(5)
    )

    assert result.fluorescence == 0.83
    assert result.realized_photons is None


def test_builtin_noise_constructors_cannot_relabel_their_sampling_rules() -> None:
    with pytest.raises(TypeError, match="sampling_rule"):
        PoissonNoise(sampling_rule="gaussian")
    with pytest.raises(TypeError, match="sampling_rule"):
        GaussianNoise(stddev_at_1s=0.0, sampling_rule="poisson")

    assert PoissonNoise().sampling_rule == "poisson"
    assert GaussianNoise(stddev_at_1s=0.0).sampling_rule == "gaussian"


@pytest.mark.parametrize(
    "noise",
    [PoissonNoise(), GaussianNoise(stddev_at_1s=0.0)],
)
def test_stateless_noise_strategies_use_immutable_noop_checkpoints(
    noise: PoissonNoise | GaussianNoise,
) -> None:
    checkpoint = noise.checkpoint()

    assert checkpoint is None
    noise.restore(checkpoint)


def test_empirical_checkpoint_restores_only_mutable_replay_cursor_in_place() -> None:
    noise = EmpiricalResidualNoise(
        [-0.1, 0.25, 0.5],
        mode="replay",
        provenance=_provenance("replay"),
    )
    residuals = noise.residuals
    provenance = noise.provenance
    checkpoint = noise.checkpoint()

    noise.sample(1.0, 10.0, 0.1, np.random.default_rng(99))
    noise.restore(checkpoint)

    assert noise.residuals.tolist() == residuals.tolist()
    assert noise.provenance is provenance
    assert noise.sample(1.0, 10.0, 0.1, np.random.default_rng(99)).fluorescence == (
        pytest.approx(0.9)
    )


def test_empirical_replay_cycles_through_residual_order() -> None:
    noise = EmpiricalResidualNoise(
        [-0.1, 0.25, 0.5],
        mode="replay",
        provenance=_provenance("replay"),
    )

    values = [
        noise.sample(1.0, 10.0, 0.1, np.random.default_rng(99)).fluorescence
        for _ in range(5)
    ]

    assert values == pytest.approx([0.9, 1.25, 1.5, 0.9, 1.25])
    assert noise.provenance == MappingProxyType(_provenance("replay"))
    with pytest.raises(TypeError):
        noise.provenance["source_id"] = "mutated"  # type: ignore[index]


def test_empirical_configuration_cannot_be_rebound_after_construction() -> None:
    supplied_residuals = np.array([-0.1, 0.25, 0.5])
    supplied_provenance = _provenance("replay")
    noise = EmpiricalResidualNoise(
        supplied_residuals,
        mode="replay",
        provenance=supplied_provenance,
    )

    supplied_residuals[0] = 99.0
    supplied_provenance["source_id"] = "mutated-source"

    with pytest.raises(AttributeError):
        noise.mode = "sample"  # type: ignore[assignment]
    with pytest.raises(AttributeError):
        noise.block_size = 2  # type: ignore[assignment]
    with pytest.raises(AttributeError):
        noise.residuals = np.array([99.0])  # type: ignore[assignment]
    with pytest.raises(AttributeError):
        noise.provenance = _provenance("sample")  # type: ignore[assignment]
    with pytest.raises(AttributeError):
        noise._mode = "sample"  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        noise._configuration_locked = False  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        noise.residuals[0] = 99.0

    assert noise.mode == "replay"
    assert noise.block_size is None
    assert noise.residuals.tolist() == [-0.1, 0.25, 0.5]
    assert noise.provenance["source_id"] == "generated-fixture-v1"


@pytest.mark.parametrize(
    "attribute",
    [
        "mode",
        "block_size",
        "residuals",
        "provenance",
        "_mode",
        "_block_size",
        "_residuals",
        "_provenance",
        "_configuration_locked",
    ],
)
def test_empirical_configuration_cannot_be_deleted_after_construction(
    attribute: str,
) -> None:
    noise = EmpiricalResidualNoise(
        [-0.1, 0.25, 0.5],
        mode="replay",
        provenance=_provenance("replay"),
    )
    sampling_rule = noise.sampling_rule
    provenance = dict(noise.provenance)

    with pytest.raises(AttributeError):
        delattr(noise, attribute)

    assert noise.sampling_rule == sampling_rule
    assert dict(noise.provenance) == provenance


def test_public_residual_copy_cannot_alter_internal_sample_sequence() -> None:
    noise = EmpiricalResidualNoise(
        [-0.1, 0.25, 0.5],
        mode="replay",
        provenance=_provenance("replay"),
    )
    exposed_residuals = noise.residuals

    exposed_residuals.setflags(write=True)
    exposed_residuals[:] = [99.0, 98.0, 97.0]
    actual = [
        noise.sample(1.0, 10.0, 0.1, np.random.default_rng(99)).fluorescence
        for _ in range(3)
    ]

    assert actual == pytest.approx([0.9, 1.25, 1.5])


def test_empirical_sample_draws_seeded_independent_residual_indices() -> None:
    residuals = np.array([-0.2, 0.0, 0.4])
    expected_rng = np.random.default_rng(11)
    expected = [
        1.0 + residuals[expected_rng.integers(0, residuals.size)]
        for _ in range(6)
    ]
    noise = EmpiricalResidualNoise(
        residuals,
        mode="sample",
        provenance=_provenance("sample"),
    )
    actual_rng = np.random.default_rng(11)

    actual = [noise.sample(1.0, 10.0, 0.1, actual_rng).fluorescence for _ in range(6)]

    assert actual == pytest.approx(expected)


def test_empirical_block_preserves_contiguous_order_across_boundaries() -> None:
    residuals = np.array([-0.2, 0.0, 0.3, 0.5])
    expected_rng = np.random.default_rng(17)
    expected: list[float] = []
    for index in range(7):
        if index % 3 == 0:
            start = expected_rng.integers(0, residuals.size)
        expected.append(1.0 + residuals[(start + index % 3) % residuals.size])
    noise = EmpiricalResidualNoise(
        residuals,
        mode="block",
        block_size=3,
        provenance=_provenance("block"),
    )
    actual_rng = np.random.default_rng(17)

    actual = [
        noise.sample(1.0, 10.0, 0.1, actual_rng).fluorescence
        for _ in range(7)
    ]

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("expected_fluorescence", "nominal_rate_hz", "integration_time_s"),
    [
        (-0.01, 1.0, 1.0),
        (np.nan, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, np.inf, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, np.nan),
    ],
)
def test_noise_strategies_reject_invalid_sample_inputs(
    expected_fluorescence: float, nominal_rate_hz: float, integration_time_s: float
) -> None:
    with pytest.raises((TypeError, ValueError)):
        PoissonNoise().sample(
            expected_fluorescence,
            nominal_rate_hz,
            integration_time_s,
            np.random.default_rng(1),
        )


@pytest.mark.parametrize(
    ("residuals", "mode", "provenance", "block_size"),
    [
        ([], "replay", _provenance("replay"), None),
        ([0.0, np.inf], "sample", _provenance("sample"), None),
        ([0.0], "unknown", _provenance("unknown"), None),
        ([0.0], "replay", {"source_id": "missing-labels"}, None),
        ([0.0], "block", _provenance("block"), 0),
        ([0.0], "block", _provenance("replay"), 2),
    ],
)
def test_empirical_noise_rejects_invalid_configuration(
    residuals: list[float],
    mode: str,
    provenance: dict[str, str],
    block_size: int | None,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        EmpiricalResidualNoise(
            residuals,
            mode=mode,
            provenance=provenance,
            block_size=block_size,
        )


@pytest.mark.parametrize("stddev_at_1s", [-0.01, np.nan, np.inf])
def test_gaussian_noise_rejects_invalid_standard_deviation(stddev_at_1s: float) -> None:
    with pytest.raises(ValueError):
        GaussianNoise(stddev_at_1s)
