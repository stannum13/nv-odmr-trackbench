"""Composition of deterministic multi-resonance ODMR spectra."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from odmr_bench.models.lineshapes import pseudo_voigt
from odmr_bench.models.parameters import Baseline, Resonance


def multi_resonance_spectrum(
    frequency_hz: ArrayLike,
    resonances: Sequence[Resonance],
    baseline: Baseline,
    *,
    additive_noise: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Evaluate baseline minus all dip components plus explicit noise."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if not np.all(np.isfinite(frequency)):
        raise ValueError("frequency_hz must be finite")
    resonance_ids = [resonance.resonance_id for resonance in resonances]
    if len(set(resonance_ids)) != len(resonance_ids):
        raise ValueError("resonance IDs must be unique")

    fluorescence = baseline.evaluate(frequency).copy()
    for resonance in resonances:
        fluorescence -= resonance.amplitude * pseudo_voigt(
            frequency,
            resonance.center_hz,
            resonance.fwhm_hz,
            resonance.eta,
        )

    if additive_noise is not None:
        noise = np.asarray(additive_noise, dtype=np.float64)
        if not np.all(np.isfinite(noise)):
            raise ValueError("additive_noise must be finite")
        try:
            fluorescence += np.broadcast_to(noise, fluorescence.shape)
        except ValueError as error:
            raise ValueError("additive_noise must broadcast to frequency_hz") from error
    return np.asarray(fluorescence, dtype=np.float64)
