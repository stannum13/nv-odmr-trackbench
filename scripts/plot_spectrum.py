"""Generate the deterministic eight-dip spectrum demonstration figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from odmr_bench.models import (
    Baseline,
    Resonance,
    multi_resonance_spectrum,
    pseudo_voigt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    frequency_config = config["frequency"]
    frequency_hz = np.linspace(
        float(frequency_config["start_hz"]),
        float(frequency_config["stop_hz"]),
        int(frequency_config["points"]),
    )
    baseline = Baseline(**config["baseline"])
    resonances = tuple(Resonance(**item) for item in config["resonances"])
    if len(resonances) != 8:
        raise ValueError("static demonstration requires exactly eight resonances")

    fluorescence = multi_resonance_spectrum(frequency_hz, resonances, baseline)
    baseline_values = baseline.evaluate(frequency_hz)

    figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    for resonance in resonances:
        component = baseline_values - resonance.amplitude * pseudo_voigt(
            frequency_hz,
            resonance.center_hz,
            resonance.fwhm_hz,
            resonance.eta,
        )
        axis.plot(frequency_hz / 1.0e9, component, color="0.78", linewidth=0.8)
    axis.plot(
        frequency_hz / 1.0e9,
        fluorescence,
        color="#1f5a94",
        linewidth=1.8,
        label="eight-resonance spectrum",
    )
    axis.plot(
        frequency_hz / 1.0e9,
        baseline_values,
        color="#b24a33",
        linestyle="--",
        linewidth=1.1,
        label="baseline",
    )
    axis.set(
        xlabel="Microwave frequency (GHz)",
        ylabel="Normalized fluorescence",
        title="Synthetic NV-ensemble ODMR spectrum",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
