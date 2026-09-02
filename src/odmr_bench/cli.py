"""Command-line entry point for raw playback and synthetic emulation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from importlib.resources import files
from math import isfinite
from numbers import Real
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


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        canonical = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    if canonical <= 0.0:
        raise ValueError(f"{name} must be positive")
    return canonical


def _nonnegative_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        canonical = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    if canonical < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return canonical


def _validate_prospective_schedule(
    queries: Sequence[Mapping[str, float]], frequency_overhead_s: float
) -> None:
    """Reject overflow before the instrument exists or any query can run."""
    virtual_time_s = 0.0
    for query in queries:
        integration_start_s = virtual_time_s + frequency_overhead_s
        midpoint_s = integration_start_s + query["integration_time_s"] / 2.0
        endpoint_s = integration_start_s + query["integration_time_s"]
        if not all(
            isfinite(value)
            for value in (integration_start_s, midpoint_s, endpoint_s)
        ):
            raise ValueError("query schedule virtual timestamps must remain finite")
        virtual_time_s = endpoint_s


def _load_drift_scenario(
    path: Path,
) -> tuple[ODMRInstrument, list[Mapping[str, float]], int]:
    """Validate a deterministic, Poisson-noise linear-drift scenario YAML file."""
    try:
        with path.open(encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream)
    except OSError as error:
        raise ValueError(f"could not read simulation config: {path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"could not parse simulation config: {path}") from error
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
    queries: list[Mapping[str, float]] = [
        _query_from_config(raw_query, index)
        for index, raw_query in enumerate(raw_queries)
    ]
    frequency_overhead_s = _nonnegative_finite_float(
        config["frequency_overhead_s"], "frequency_overhead_s"
    )
    _validate_prospective_schedule(queries, frequency_overhead_s)
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    instrument = ODMRInstrument(
        dynamics=LinearCenterDrift(snapshot, config["center_slew_hz_per_s"]),
        noise=PoissonNoise(),
        nominal_photon_rate_hz=config["nominal_photon_rate_hz"],
        frequency_overhead_s=frequency_overhead_s,
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


def _query_from_config(value: object, index: int) -> Mapping[str, float]:
    query = _mapping(value, f"queries[{index}]")
    _exact_keys(
        query,
        name=f"queries[{index}]",
        required=frozenset({"frequency_hz", "integration_time_s"}),
    )
    return {
        "frequency_hz": _positive_finite_float(
            query["frequency_hz"], f"queries[{index}].frequency_hz"
        ),
        "integration_time_s": _positive_finite_float(
            query["integration_time_s"], f"queries[{index}].integration_time_s"
        ),
    }


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
                "signal_quantity": record.signal_quantity,
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
    observation_count = 0
    signal_min: float | None = None
    signal_max: float | None = None
    for observation in observations:
        if max_observations is not None and observation_count >= max_observations:
            break
        observation_count += 1
        signal_min = (
            observation.signal
            if signal_min is None
            else min(signal_min, observation.signal)
        )
        signal_max = (
            observation.signal
            if signal_max is None
            else max(signal_max, observation.signal)
        )
    if observation_count == 0 or signal_min is None or signal_max is None:
        raise ValueError("playback requires at least one observation")
    _print_json(
        {
            "inferred_timestamps": False,
            "mode": "recorded_playback",
            "observation_count": observation_count,
            "points_per_sweep": int(dataset.signal.shape[1]),
            "signal_max": signal_max,
            "signal_min": signal_min,
            "signal_quantity": FIGSHARE_28788437_V1.signal_quantity,
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


def _simulation_config_path(config: str) -> Path:
    if config == "bundled:drift":
        return Path(files("odmr_bench").joinpath("configs", "drift.yaml"))
    if config.startswith("bundled:"):
        raise ValueError(f"unknown bundled simulation config: {config}")
    return Path(config)


def _simulate(config: str) -> int:
    try:
        instrument, queries, seed = _load_drift_scenario(
            _simulation_config_path(config)
        )
    except OverflowError as error:
        raise ValueError(
            "simulation config has a numeric value outside the finite float range"
        ) from error
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
    simulate.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "dataset-info":
            return _dataset_info()
        if args.command == "playback":
            return _playback(args.path, args.max_observations)
        if args.command == "simulate":
            return _simulate(args.config)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        return 2
    raise RuntimeError("unreachable command dispatch")


if __name__ == "__main__":
    raise SystemExit(main())
