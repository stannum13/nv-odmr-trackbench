"""Private provenance capability construction for two-point evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from odmr_bench.emulator.instrument import ODMRInstrument
from odmr_bench.estimators.two_point_types import TwoPointCalibrationSource
from odmr_bench.evaluation.two_point.types import (
    TwoPointEvaluatorInstrumentConfiguration,
    VerifiedInstrumentRunToken,
    VerifiedTwoPointCalibrationSuccess,
)

if TYPE_CHECKING:
    from .runner import TwoPointEvaluatorRunner

_TOKEN_CONSTRUCTION_KEY: object = object()
_MINTED_RUN_TOKEN_IDENTITIES: dict[int, VerifiedInstrumentRunToken] = {}


@dataclass(frozen=True, slots=True)
class _RunTokenBinding:
    """Exact in-process identities associated with one runner-issued token."""

    issuer_runner: TwoPointEvaluatorRunner
    instrument: ODMRInstrument
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration
    success: VerifiedTwoPointCalibrationSuccess | None
    source: TwoPointCalibrationSource | None


_RUN_TOKEN_BINDINGS: dict[VerifiedInstrumentRunToken, _RunTokenBinding] = {}


def _mint_verified_instrument_run_token(
    construction_key: object,
) -> VerifiedInstrumentRunToken:
    """Create a runner-issued token after the evaluator-private key check."""
    if construction_key is not _TOKEN_CONSTRUCTION_KEY:
        raise TypeError("invalid verified instrument run token construction key")
    token = object.__new__(VerifiedInstrumentRunToken)
    _MINTED_RUN_TOKEN_IDENTITIES[id(token)] = token
    return token


def _register_run_token(
    token: VerifiedInstrumentRunToken,
    issuer_runner: TwoPointEvaluatorRunner,
    instrument: ODMRInstrument,
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration,
) -> None:
    """Register the initial exact issuer/instrument identity for one token."""
    from .runner import TwoPointEvaluatorRunner

    if type(token) is not VerifiedInstrumentRunToken:
        raise TypeError("token must be an exact VerifiedInstrumentRunToken")
    if type(issuer_runner) is not TwoPointEvaluatorRunner:
        raise TypeError("issuer_runner must be an exact TwoPointEvaluatorRunner")
    if type(instrument) is not ODMRInstrument:
        raise TypeError("instrument must be an exact ODMRInstrument")
    if type(
        instrument_configuration
    ) is not TwoPointEvaluatorInstrumentConfiguration:
        raise TypeError(
            "instrument_configuration must be an exact "
            "TwoPointEvaluatorInstrumentConfiguration"
        )
    if token in _RUN_TOKEN_BINDINGS:
        raise ValueError("run token is already registered")
    if _MINTED_RUN_TOKEN_IDENTITIES.get(id(token)) is not token:
        raise ValueError("run token lacks its runner-minted identity")
    if (
        issuer_runner._instrument is not instrument
        or issuer_runner._state.run_token is not token
        or issuer_runner._state.instrument_configuration
        is not instrument_configuration
    ):
        raise ValueError(
            "run token does not match its runner/instrument/configuration identity"
        )
    del _MINTED_RUN_TOKEN_IDENTITIES[id(token)]
    _RUN_TOKEN_BINDINGS[token] = _RunTokenBinding(
        issuer_runner=issuer_runner,
        instrument=instrument,
        instrument_configuration=instrument_configuration,
        success=None,
        source=None,
    )


def _rollback_run_token_registration(
    token: VerifiedInstrumentRunToken,
) -> None:
    """Unconditionally revoke one freshly minted bind-attempt token."""
    if _MINTED_RUN_TOKEN_IDENTITIES.get(id(token)) is token:
        del _MINTED_RUN_TOKEN_IDENTITIES[id(token)]
    _RUN_TOKEN_BINDINGS.pop(token, None)


def _lookup_run_token_binding(
    token: VerifiedInstrumentRunToken,
) -> _RunTokenBinding | None:
    """Return the exact registered binding, never class-membership authority."""
    if type(token) is not VerifiedInstrumentRunToken:
        return None
    return _RUN_TOKEN_BINDINGS.get(token)


def _snapshot_run_token_binding_before_success(
    token: VerifiedInstrumentRunToken,
    issuer_runner: TwoPointEvaluatorRunner,
    instrument: ODMRInstrument,
) -> _RunTokenBinding:
    """Copy the trusted empty binding before a success-bind transaction."""
    binding = _RUN_TOKEN_BINDINGS.get(token)
    if (
        binding is None
        or binding.issuer_runner is not issuer_runner
        or binding.instrument is not instrument
        or binding.instrument_configuration
        is not issuer_runner._state.instrument_configuration
        or binding.success is not None
        or binding.source is not None
    ):
        raise ValueError("run token lacks its trusted pre-success binding")
    return _RunTokenBinding(
        issuer_runner=binding.issuer_runner,
        instrument=binding.instrument,
        instrument_configuration=binding.instrument_configuration,
        success=None,
        source=None,
    )


def _bind_run_token_success(
    token: VerifiedInstrumentRunToken,
    issuer_runner: TwoPointEvaluatorRunner,
    instrument: ODMRInstrument,
    success: VerifiedTwoPointCalibrationSuccess,
) -> None:
    """Bind one exact successful outcome/source to its existing token record."""
    from odmr_bench.estimators.two_point_calibration import (
        _consume_verified_source_construction_identity,
    )

    from .runner import TwoPointEvaluatorRunner

    if type(token) is not VerifiedInstrumentRunToken:
        raise TypeError("token must be an exact VerifiedInstrumentRunToken")
    if type(issuer_runner) is not TwoPointEvaluatorRunner:
        raise TypeError("issuer_runner must be an exact TwoPointEvaluatorRunner")
    if type(instrument) is not ODMRInstrument:
        raise TypeError("instrument must be an exact ODMRInstrument")
    if type(success) is not VerifiedTwoPointCalibrationSuccess:
        raise TypeError(
            "success must be an exact VerifiedTwoPointCalibrationSuccess"
        )
    from .resource_accounting import _replay_full_resources

    binding = _RUN_TOKEN_BINDINGS.get(token)
    source = success.source
    current_resources = instrument.resources
    current_virtual_time_s = instrument.virtual_time_s
    last_safe_observation = (
        None if not success.safe_observations else success.safe_observations[-1]
    )
    expected_midpoints: list[float] = []
    previous_endpoint_s = source.source_start_timestamp_s
    for observation in success.safe_observations:
        integration_start_s = (
            previous_endpoint_s
            + binding.instrument_configuration.frequency_overhead_s
            if binding is not None
            else previous_endpoint_s
        )
        expected_midpoints.append(
            integration_start_s + observation.integration_time_s / 2.0
        )
        previous_endpoint_s = observation.timestamp_s
    replayed_resources = (
        None
        if binding is None
        else _replay_full_resources(
            success.full_observations,
            binding.instrument_configuration.frequency_overhead_s,
        )
    )
    if (
        binding is None
        or binding.issuer_runner is not issuer_runner
        or binding.instrument is not instrument
        or binding.instrument_configuration
        is not issuer_runner._state.instrument_configuration
        or success.run_token is not token
        or source.provenance != "verified_factory_acquisition"
        or source.source_observations != success.safe_observations
        or source.safe_resources != success.safe_resources
        or source.source_frequency_overhead_s
        != binding.instrument_configuration.frequency_overhead_s
        or source.fluorescence_provenance.nominal_photon_rate_hz
        != binding.instrument_configuration.nominal_photon_rate_hz
        or source.source_start_timestamp_s
        != issuer_runner._state.current_virtual_time_s
        or success.measurement_midpoints_s != tuple(expected_midpoints)
        or success.instrument_resources_before
        != issuer_runner._state.instrument_resources_current
        or success.instrument_resources_after != current_resources
        or success.full_resources != current_resources
        or replayed_resources != success.full_resources
        or last_safe_observation is None
        or source.availability_sequence_index
        != last_safe_observation.sequence_index
        or source.availability_timestamp_s
        != last_safe_observation.timestamp_s
        or current_virtual_time_s != source.availability_timestamp_s
        or binding.success is not None
        or binding.source is not None
        or any(
            existing.success is success or existing.source is source
            for existing in _RUN_TOKEN_BINDINGS.values()
        )
    ):
        raise ValueError("run token success does not match its registered identity")
    if not _consume_verified_source_construction_identity(source):
        raise ValueError("run token success does not match its registered identity")
    _RUN_TOKEN_BINDINGS[token] = _RunTokenBinding(
        issuer_runner=binding.issuer_runner,
        instrument=binding.instrument,
        instrument_configuration=binding.instrument_configuration,
        success=success,
        source=source,
    )


def _rollback_run_token_success(
    token: VerifiedInstrumentRunToken,
    binding_before: _RunTokenBinding,
) -> None:
    """Restore the trusted binding captured before this success transaction."""
    _RUN_TOKEN_BINDINGS[token] = binding_before
