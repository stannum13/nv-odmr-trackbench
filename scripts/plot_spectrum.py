"""Generate the deterministic eight-dip spectrum demonstration figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from odmr_bench.plotting import (
    generate_static_spectrum_curves,
    load_static_spectrum_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_static_spectrum_config(args.config)
    curves = generate_static_spectrum_curves(config)

    figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    for component in curves.components:
        axis.plot(
            curves.frequency_hz / 1.0e9,
            component,
            color="0.78",
            linewidth=0.8,
        )
    axis.plot(
        curves.frequency_hz / 1.0e9,
        curves.fluorescence,
        color="#1f5a94",
        linewidth=1.8,
        label="eight-resonance spectrum",
    )
    axis.plot(
        curves.frequency_hz / 1.0e9,
        curves.baseline,
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
