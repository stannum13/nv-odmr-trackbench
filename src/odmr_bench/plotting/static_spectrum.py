"""Configuration loading and numerical curves for the static spectrum plot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from odmr_bench.models import Baseline, Resonance, multi_resonance_spectrum
from odmr_bench.models.lineshapes import pseudo_voigt


@dataclass(frozen=True, slots=True)
class StaticSpectrumConfig:
    """Validated inputs for the deterministic eight-resonance demonstration."""

    frequency_hz: NDArray[np.float64]
    baseline: Baseline
    resonances: tuple[Resonance, ...]


@dataclass(frozen=True, slots=True)
class StaticSpectrumCurves:
    """Numerical curves rendered by the static spectrum plotting script."""

    frequency_hz: NDArray[np.float64]
    fluorescence: NDArray[np.float64]
    baseline: NDArray[np.float64]
    components: NDArray[np.float64]


def load_static_spectrum_config(path: Path) -> StaticSpectrumConfig:
    """Load the static demonstration YAML into validated model objects."""
    with path.open(encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)

    frequency_config = raw_config["frequency"]
    frequency_hz = np.linspace(
        float(frequency_config["start_hz"]),
        float(frequency_config["stop_hz"]),
        int(frequency_config["points"]),
    )
    baseline = Baseline(**raw_config["baseline"])
    resonances = tuple(Resonance(**item) for item in raw_config["resonances"])
    if len(resonances) != 8:
        raise ValueError("static demonstration requires exactly eight resonances")
    if len({resonance.resonance_id for resonance in resonances}) != 8:
        raise ValueError("static demonstration resonance IDs must be unique")
    if len({resonance.center_hz for resonance in resonances}) != 8:
        raise ValueError("static demonstration resonance centers must be unique")
    return StaticSpectrumConfig(frequency_hz, baseline, resonances)


def generate_static_spectrum_curves(
    config: StaticSpectrumConfig,
) -> StaticSpectrumCurves:
    """Generate the deterministic aggregate, baseline, and component curves."""
    baseline_values = config.baseline.evaluate(config.frequency_hz)
    components = np.stack(
        [
            baseline_values
            - resonance.amplitude
            * pseudo_voigt(
                config.frequency_hz,
                resonance.center_hz,
                resonance.fwhm_hz,
                resonance.eta,
            )
            for resonance in config.resonances
        ]
    )
    fluorescence = multi_resonance_spectrum(
        config.frequency_hz,
        config.resonances,
        config.baseline,
    )
    return StaticSpectrumCurves(
        config.frequency_hz,
        fluorescence,
        baseline_values,
        components,
    )
