from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from odmr_bench.plotting.static_spectrum import (
    generate_static_spectrum_curves,
    load_static_spectrum_config,
)


def test_static_config_has_eight_unique_resonance_ids_and_centers() -> None:
    config = load_static_spectrum_config(Path("configs/static.yaml"))

    resonance_ids = [resonance.resonance_id for resonance in config.resonances]
    centers_hz = [resonance.center_hz for resonance in config.resonances]

    assert len(resonance_ids) == len(set(resonance_ids)) == 8
    assert len(centers_hz) == len(set(centers_hz)) == 8
    assert_allclose(
        centers_hz,
        np.array([2.805, 2.825, 2.845, 2.865, 2.875, 2.895, 2.915, 2.935])
        * 1.0e9,
    )


def test_static_curves_contain_eight_numerically_resolved_dips() -> None:
    config = load_static_spectrum_config(Path("configs/static.yaml"))
    curves = generate_static_spectrum_curves(config)

    local_minimum_indices = np.flatnonzero(
        (curves.fluorescence[1:-1] < curves.fluorescence[:-2])
        & (curves.fluorescence[1:-1] < curves.fluorescence[2:])
    ) + 1

    assert curves.components.shape == (8, config.frequency_hz.size)
    assert local_minimum_indices.size == 8
    assert_allclose(
        curves.frequency_hz[local_minimum_indices],
        [resonance.center_hz for resonance in config.resonances],
        atol=np.diff(config.frequency_hz).max(),
        rtol=0.0,
    )
    assert np.all(
        curves.fluorescence[local_minimum_indices]
        < curves.baseline[local_minimum_indices]
    )
