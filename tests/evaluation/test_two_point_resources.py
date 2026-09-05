"""Tests for evaluator-private full two-point resource accounting."""

from __future__ import annotations

import ast
import math
from dataclasses import fields, replace
from pathlib import Path

from odmr_bench.emulator.observations import InstrumentObservation
from odmr_bench.emulator.resources import ResourceSnapshot
from odmr_bench.estimators.two_point_types import PublicAcquisitionResources
from odmr_bench.evaluation import two_point
from odmr_bench.evaluation.two_point.resource_accounting import (
    _advance_full_resources,
    _project_full_resources,
    _replay_full_resources,
    _resource_mismatch_fields,
    _zero_full_resources,
)


def _instrument_observation(
    *,
    sequence_index: int,
    integration_time_s: float,
    nominal_exposure_photons: float,
    expected_photons: float,
    realized_photons: int | None,
) -> InstrumentObservation:
    return InstrumentObservation(
        sequence_index=sequence_index,
        timestamp_s=(sequence_index + 1) * integration_time_s,
        frequency_hz=2.80e9 + sequence_index * 1.0e6,
        fluorescence=0.98,
        integration_time_s=integration_time_s,
        nominal_exposure_photons=nominal_exposure_photons,
        expected_photons=expected_photons,
        realized_photons=realized_photons,
        sampling_rule="test-rule",
    )


def test_full_resource_helpers_are_evaluator_private_and_atomic() -> None:
    observations = tuple(
        _instrument_observation(
            sequence_index=index,
            integration_time_s=0.005,
            nominal_exposure_photons=12_500.0 + index,
            expected_photons=12_000.0 + index,
            realized_photons=realized,
        )
        for index, realized in enumerate((None, 3, None, 5, 7, None))
    )

    zero = _zero_full_resources()
    assert zero == ResourceSnapshot(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    first = _advance_full_resources(zero, observations[0], 0.001)
    assert first == ResourceSnapshot(
        observations=1,
        integration_time_s=0.005,
        nominal_exposure_photons=12_500.0,
        expected_photons=12_000.0,
        realized_photons=0,
        observations_without_realized_counts=1,
        virtual_elapsed_time_s=0.006,
    )

    total = zero
    for observation in observations:
        total = _advance_full_resources(total, observation, 0.001)
    assert total == _replay_full_resources(observations, 0.001)
    assert total == ResourceSnapshot(
        observations=6,
        integration_time_s=total.integration_time_s,
        nominal_exposure_photons=75_015.0,
        expected_photons=72_015.0,
        realized_photons=15,
        observations_without_realized_counts=3,
        virtual_elapsed_time_s=total.virtual_elapsed_time_s,
    )
    assert total.integration_time_s.hex() == "0x1.eb851eb851eb9p-6"
    assert total.integration_time_s != math.fsum([0.005] * 6)
    assert total.virtual_elapsed_time_s.hex() == "0x1.26e978d4fdf3bp-5"

    projected = _project_full_resources(total)
    assert projected == PublicAcquisitionResources(
        observations=total.observations,
        integration_time_s=total.integration_time_s,
        nominal_exposure_photons=total.nominal_exposure_photons,
        realized_photons=total.realized_photons,
        observations_without_realized_counts=(
            total.observations_without_realized_counts
        ),
        virtual_elapsed_time_s=total.virtual_elapsed_time_s,
    )
    assert projected.realized_photons == total.realized_photons
    assert not hasattr(projected, "expected_photons")

    private_helper_names = (
        "_zero_full_resources",
        "_advance_full_resources",
        "_replay_full_resources",
        "_project_full_resources",
        "_resource_mismatch_fields",
    )
    assert not any(hasattr(two_point, name) for name in private_helper_names)


def _estimator_resource_boundary_violations(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    forbidden_identifiers = {
        "ResourceSnapshot",
        "expected_photons",
        "_zero_full_resources",
        "_advance_full_resources",
        "_replay_full_resources",
        "_project_full_resources",
        "_resource_mismatch_fields",
    }
    forbidden_modules = {
        "odmr_bench.emulator.resources",
        "odmr_bench.evaluation.two_point.resource_accounting",
    }
    violations: set[str] = set()

    def check_identifier(identifier: str, location: str) -> None:
        if identifier in forbidden_identifiers or "full_resources" in identifier:
            violations.add(f"{location}:{identifier}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            check_identifier(node.id, "name")
        elif isinstance(node, ast.Attribute):
            check_identifier(node.attr, "attribute")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            check_identifier(node.name, "definition")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    violations.add(f"module:{alias.name}")
                check_identifier(alias.name.rsplit(".", 1)[-1], "import")
                if alias.asname is not None:
                    check_identifier(alias.asname, "import-alias")
        elif isinstance(node, ast.ImportFrom):
            if node.module in forbidden_modules:
                violations.add(f"module:{node.module}")
            for alias in node.names:
                check_identifier(alias.name, "from-import")
                if alias.asname is not None:
                    check_identifier(alias.asname, "import-alias")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            check_identifier(node.value, "string-reference")
    return tuple(sorted(violations))


def test_full_resource_boundary_is_absent_from_every_estimator_module() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    estimator_root = repository_root / "src/odmr_bench/estimators"
    estimator_paths = tuple(sorted(estimator_root.rglob("*.py")))

    assert estimator_paths
    violations = {
        path.relative_to(repository_root).as_posix(): found
        for path in estimator_paths
        if (found := _estimator_resource_boundary_violations(path))
    }
    assert violations == {}


def test_full_resource_replay_preserves_order_sensitive_binary64_arrivals() -> (
    None
):
    observations = tuple(
        _instrument_observation(
            sequence_index=index,
            integration_time_s=0.005,
            nominal_exposure_photons=nominal,
            expected_photons=expected,
            realized_photons=index + 1,
        )
        for index, (nominal, expected) in enumerate(
            (
                (float(2**53), float(2**52)),
                (1.0, 0.5),
                (1.0, 0.5),
            )
        )
    )

    forward = _zero_full_resources()
    for observation in observations:
        forward = _advance_full_resources(forward, observation, 0.001)
    reverse = _zero_full_resources()
    for observation in reversed(observations):
        reverse = _advance_full_resources(reverse, observation, 0.001)

    assert forward.nominal_exposure_photons.hex() == "0x1.0000000000000p+53"
    assert reverse.nominal_exposure_photons.hex() == "0x1.0000000000001p+53"
    assert forward.expected_photons.hex() == "0x1.0000000000000p+52"
    assert reverse.expected_photons.hex() == "0x1.0000000000001p+52"
    assert forward != reverse
    assert _replay_full_resources(observations, 0.001) == forward


def test_resource_mismatch_fields_are_exact_complete_and_declaration_ordered() -> (
    None
):
    expected = ResourceSnapshot(
        observations=2,
        integration_time_s=0.03,
        nominal_exposure_photons=25_000.0,
        expected_photons=24_000.0,
        realized_photons=23_975,
        observations_without_realized_counts=1,
        virtual_elapsed_time_s=0.04,
    )
    field_names = tuple(field.name for field in fields(ResourceSnapshot))
    mutations: dict[str, float | int] = {
        "observations": 3,
        "integration_time_s": math.nextafter(
            expected.integration_time_s, math.inf
        ),
        "nominal_exposure_photons": math.nextafter(
            expected.nominal_exposure_photons, math.inf
        ),
        "expected_photons": math.nextafter(expected.expected_photons, math.inf),
        "realized_photons": expected.realized_photons + 1,
        "observations_without_realized_counts": 2,
        "virtual_elapsed_time_s": math.nextafter(
            expected.virtual_elapsed_time_s, math.inf
        ),
    }

    assert field_names == (
        "observations",
        "integration_time_s",
        "nominal_exposure_photons",
        "expected_photons",
        "realized_photons",
        "observations_without_realized_counts",
        "virtual_elapsed_time_s",
    )
    assert _resource_mismatch_fields(expected, expected) == ()
    for field_name, changed_value in mutations.items():
        actual = replace(expected, **{field_name: changed_value})
        assert _resource_mismatch_fields(expected, actual) == (field_name,)

    actual = replace(expected, **mutations)
    assert _resource_mismatch_fields(expected, actual) == field_names
