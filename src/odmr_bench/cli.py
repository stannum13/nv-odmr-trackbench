"""Command-line entry point for raw playback and synthetic emulation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from odmr_bench import __version__
from odmr_bench.datasets import FIGSHARE_28788437_V1, parse_figshare_sweep_file
from odmr_bench.datasets.playback import iter_playback_for_analysis
from odmr_bench.dynamics import LinearCenterDrift, SpectralSnapshot
from odmr_bench.emulator import InstrumentObservation, ODMRInstrument, PoissonNoise
from odmr_bench.models import Baseline, Resonance


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _exact_keys(
    value: Mapping[str, object], *, name: str, required: frozenset[str]
) -> None:
    if set(value) != required:
        raise ValueError(
            f"{name} must contain exactly {', '.join(sorted(required))}"
        )


def _load_drift_scenario(
    path: Path,
) -> tuple[ODMRInstrument, list[Mapping[str, object]], int]:
    """Validate a deterministic, Poisson-noise linear-drift scenario YAML file."""
    with path.open(encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)
    config = _mapping(raw_config, "simulation config")
    _exact_keys(
        config,
        name="simulation config",
        required=frozenset(
            {
                "baseline",
                "center_slew_hz_per_s",
                "frequency_overhead_s",
                "noise",
                "nominal_photon_rate_hz",
                "queries",
                "resonances",
                "seed",
            }
        ),
    )

    baseline_config = _mapping(config["baseline"], "baseline")
    _exact_keys(
        baseline_config,
        name="baseline",
        required=frozenset(
            {"intercept", "quadratic_per_hz2", "reference_hz", "slope_per_hz"}
        ),
    )
    baseline = Baseline(**baseline_config)

    raw_resonances = config["resonances"]
    if not isinstance(raw_resonances, list) or len(raw_resonances) != 8:
        raise ValueError("resonances must contain exactly eight entries")
    resonances = tuple(
        _resonance_from_config(raw_resonance, index)
        for index, raw_resonance in enumerate(raw_resonances)
    )
    snapshot = SpectralSnapshot(baseline=baseline, resonances=resonances)

    noise_config = _mapping(config["noise"], "noise")
    _exact_keys(noise_config, name="noise", required=frozenset({"kind"}))
    if noise_config["kind"] != "poisson":
        raise ValueError("noise.kind must be 'poisson' for this command")

    raw_queries = config["queries"]
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError("queries must be a non-empty list")
    queries = [
        _query_from_config(raw_query, index)
        for index, raw_query in enumerate(raw_queries)
    ]
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    instrument = ODMRInstrument(
        dynamics=LinearCenterDrift(snapshot, config["center_slew_hz_per_s"]),
        noise=PoissonNoise(),
        nominal_photon_rate_hz=config["nominal_photon_rate_hz"],
        frequency_overhead_s=config["frequency_overhead_s"],
        seed=seed,
    )
    return instrument, queries, seed


def _resonance_from_config(value: object, index: int) -> Resonance:
    resonance = _mapping(value, f"resonances[{index}]")
    _exact_keys(
        resonance,
        name=f"resonances[{index}]",
        required=frozenset({"amplitude", "center_hz", "eta", "fwhm_hz", "id"}),
    )
    return Resonance(
        resonance_id=resonance["id"],
        center_hz=resonance["center_hz"],
        fwhm_hz=resonance["fwhm_hz"],
        amplitude=resonance["amplitude"],
        eta=resonance["eta"],
    )


def _query_from_config(value: object, index: int) -> Mapping[str, object]:
    query = _mapping(value, f"queries[{index}]")
    _exact_keys(
        query,
        name=f"queries[{index}]",
        required=frozenset({"frequency_hz", "integration_time_s"}),
    )
    return query


def _print_json(payload: Mapping[str, Any]) -> None:
    """Print one stable, finite JSON summary for scripts and smoke tests."""
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


def _dataset_info() -> int:
    record = FIGSHARE_28788437_V1
    _print_json(
        {
            "byte_size": record.byte_size,
            "checksum": {
                "algorithm": record.checksum_algorithm,
                "value": record.checksum_value,
            },
            "doi": record.doi,
            "download_url": record.download_url,
            "license": record.license_name,
            "limitations": {
                "ground_truth_status": record.ground_truth_status,
                "missing_metadata": list(record.missing_metadata),
                "timing_status": record.timing_status,
                "unit_status": record.unit_status,
            },
        }
    )
    return 0


def _playback(path: Path, max_observations: int | None) -> int:
    if max_observations is not None and max_observations <= 0:
        raise ValueError("max_observations must be positive")
    dataset = parse_figshare_sweep_file(path)
    observations = iter_playback_for_analysis(dataset)
    selected = []
    for observation in observations:
        if max_observations is not None and len(selected) >= max_observations:
            break
        selected.append(observation)
    if not selected:
        raise ValueError("playback requires at least one observation")
    signals = [observation.signal for observation in selected]
    _print_json(
        {
            "inferred_timestamps": False,
            "mode": "recorded_playback",
            "observation_count": len(selected),
            "points_per_sweep": int(dataset.signal.shape[1]),
            "signal_max": max(signals),
            "signal_min": min(signals),
            "sweep_count": int(dataset.signal.shape[0]),
            "timing_status": "nominal_without_timestamps",
            "unit_status": "conflicted_unverified",
        }
    )
    return 0


def _sample_summary(sample: InstrumentObservation) -> dict[str, float | int]:
    return {
        "fluorescence": sample.fluorescence,
        "frequency_hz": sample.frequency_hz,
        "sequence_index": sample.sequence_index,
        "timestamp_s": sample.timestamp_s,
    }


def _simulate(config_path: Path) -> int:
    instrument, queries, seed = _load_drift_scenario(config_path)
    samples = [
        instrument.query(
            frequency_hz=query["frequency_hz"],
            integration_time_s=query["integration_time_s"],
        )
        for query in queries
    ]
    resources = instrument.resources
    _print_json(
        {
            "expected_photons": resources.expected_photons,
            "final_virtual_time_s": instrument.virtual_time_s,
            "first_sample": _sample_summary(samples[0]),
            "integration_time_s": resources.integration_time_s,
            "last_sample": _sample_summary(samples[-1]),
            "mode": "synthetic_emulation",
            "nominal_exposure_photons": resources.nominal_exposure_photons,
            "query_count": resources.observations,
            "realized_photons": resources.realized_photons,
            "seed": seed,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odmrbench")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "dataset-info", help="print checked optional-dataset metadata"
    )
    playback = commands.add_parser(
        "playback", help="summarize an explicit raw sweep file"
    )
    playback.add_argument("--path", type=Path, required=True)
    playback.add_argument("--max-observations", type=int)
    simulate = commands.add_parser(
        "simulate", help="run a fixed synthetic drift schedule"
    )
    simulate.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dataset-info":
        return _dataset_info()
    if args.command == "playback":
        return _playback(args.path, args.max_observations)
    if args.command == "simulate":
        return _simulate(args.config)
    raise RuntimeError("unreachable command dispatch")


if __name__ == "__main__":
    raise SystemExit(main())
