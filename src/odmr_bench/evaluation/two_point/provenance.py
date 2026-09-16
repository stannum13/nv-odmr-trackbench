"""Private provenance capability construction for two-point evaluation."""

from __future__ import annotations

import weakref
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
    from odmr_bench.evaluation.sparse_linewidth.runner import (
        SparseLinewidthEvaluatorRunner,
    )

    from .runner import TwoPointEvaluatorRunner

    _RegisteredEvaluatorRunner = (
        TwoPointEvaluatorRunner | SparseLinewidthEvaluatorRunner
    )
else:
    _RegisteredEvaluatorRunner = object

_TOKEN_CONSTRUCTION_KEY: object = object()
_MINTED_RUN_TOKEN_IDENTITIES: dict[int, VerifiedInstrumentRunToken] = {}


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _RunTokenBinding:
    """Exact in-process identities associated with one runner-issued token."""

    issuer_runner: _RegisteredEvaluatorRunner
    instrument: ODMRInstrument
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration
    success: VerifiedTwoPointCalibrationSuccess | None
    source: TwoPointCalibrationSource | None


_RUN_TOKEN_BINDINGS: weakref.WeakValueDictionary[
    VerifiedInstrumentRunToken, _RunTokenBinding
] = weakref.WeakValueDictionary()


class _VerifiedCalibrationIssuer:
    """Unforgeable in-process capability for one exact registered runner."""

    __slots__ = (
        "__weakref__",
        "_instrument",
        "_instrument_configuration",
        "_run_token",
        "_runner",
    )

    def __new__(cls) -> _VerifiedCalibrationIssuer:
        raise TypeError("private verified calibration issuer cannot be constructed")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("private verified calibration issuer may not be subclassed")

    def __copy__(self) -> _VerifiedCalibrationIssuer:
        raise TypeError("private verified calibration issuer cannot be copied")

    def __deepcopy__(self, memo: object) -> _VerifiedCalibrationIssuer:
        del memo
        raise TypeError("private verified calibration issuer cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("private verified calibration issuer cannot be serialized")


_VERIFIED_CALIBRATION_ISSUERS: weakref.WeakKeyDictionary[
    object, weakref.ReferenceType[_VerifiedCalibrationIssuer]
] = weakref.WeakKeyDictionary()
_VERIFIED_CALIBRATION_ISSUER_REGISTRATIONS: weakref.WeakValueDictionary[
    VerifiedInstrumentRunToken, _VerifiedCalibrationIssuer
] = weakref.WeakValueDictionary()


def _registered_runner_types() -> tuple[type[object], type[object]]:
    """Resolve the closed runner allowlist after both classes are defined."""
    from odmr_bench.evaluation.sparse_linewidth.runner import (
        SparseLinewidthEvaluatorRunner,
    )

    from .runner import TwoPointEvaluatorRunner

    return TwoPointEvaluatorRunner, SparseLinewidthEvaluatorRunner


def _is_exact_registered_runner(runner: object) -> bool:
    return type(runner) in _registered_runner_types()


def _mint_verified_calibration_issuer(
    runner: _RegisteredEvaluatorRunner,
    instrument: ODMRInstrument,
    token: VerifiedInstrumentRunToken,
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration,
) -> None:
    issuer = object.__new__(_VerifiedCalibrationIssuer)
    object.__setattr__(issuer, "_runner", runner)
    object.__setattr__(issuer, "_instrument", instrument)
    object.__setattr__(issuer, "_run_token", token)
    object.__setattr__(
        issuer, "_instrument_configuration", instrument_configuration
    )
    object.__setattr__(runner, "_provenance_issuer", issuer)
    _VERIFIED_CALIBRATION_ISSUERS[runner] = weakref.ref(issuer)
    _VERIFIED_CALIBRATION_ISSUER_REGISTRATIONS[token] = issuer


def _lookup_verified_calibration_issuer(
    runner: object,
) -> _VerifiedCalibrationIssuer:
    """Return authority only for the exact live registered runner identity."""
    if not _is_exact_registered_runner(runner):
        raise TypeError("issuer requires a registered exact runner")
    issuer_reference = _VERIFIED_CALIBRATION_ISSUERS.get(runner)
    issuer = None if issuer_reference is None else issuer_reference()
    try:
        state = runner._state
        instrument = runner._instrument
    except AttributeError:
        raise TypeError("issuer requires a registered exact runner") from None
    binding = _RUN_TOKEN_BINDINGS.get(state.run_token)
    registration = _VERIFIED_CALIBRATION_ISSUER_REGISTRATIONS.get(
        state.run_token
    )
    if (
        type(issuer) is not _VerifiedCalibrationIssuer
        or issuer._runner is not runner
        or issuer._instrument is not instrument
        or issuer._run_token is not state.run_token
        or issuer._instrument_configuration is not state.instrument_configuration
        or registration is None
        or registration is not issuer
        or binding is None
        or binding.issuer_runner is not runner
        or binding.instrument is not instrument
        or binding.instrument_configuration is not state.instrument_configuration
    ):
        raise TypeError("issuer requires a registered exact runner")
    return issuer


def _runner_from_verified_calibration_issuer(
    issuer: _VerifiedCalibrationIssuer,
) -> _RegisteredEvaluatorRunner:
    """Authenticate every issuer-held identity against the live registries."""
    if type(issuer) is not _VerifiedCalibrationIssuer:
        raise TypeError("issuer requires a registered exact runner")
    try:
        runner = issuer._runner
    except AttributeError:
        raise TypeError("issuer requires a registered exact runner") from None
    if _lookup_verified_calibration_issuer(runner) is not issuer:
        raise TypeError("issuer requires a registered exact runner")
    return runner


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
    issuer_runner: _RegisteredEvaluatorRunner,
    instrument: ODMRInstrument,
    instrument_configuration: TwoPointEvaluatorInstrumentConfiguration,
) -> None:
    """Register the initial exact issuer/instrument identity for one token."""
    if type(token) is not VerifiedInstrumentRunToken:
        raise TypeError("token must be an exact VerifiedInstrumentRunToken")
    if not _is_exact_registered_runner(issuer_runner):
        raise TypeError("issuer_runner must be a registered exact runner type")
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
    binding = _RunTokenBinding(
        issuer_runner=issuer_runner,
        instrument=instrument,
        instrument_configuration=instrument_configuration,
        success=None,
        source=None,
    )
    object.__setattr__(issuer_runner, "_provenance_binding", binding)
    _RUN_TOKEN_BINDINGS[token] = binding
    _mint_verified_calibration_issuer(
        issuer_runner,
        instrument,
        token,
        instrument_configuration,
    )


def _rollback_run_token_registration(
    token: VerifiedInstrumentRunToken,
    transaction_runner: _RegisteredEvaluatorRunner | None = None,
) -> None:
    """Unconditionally revoke one freshly minted bind-attempt token."""
    if _MINTED_RUN_TOKEN_IDENTITIES.get(id(token)) is token:
        del _MINTED_RUN_TOKEN_IDENTITIES[id(token)]
    binding = _RUN_TOKEN_BINDINGS.pop(token, None)
    issuer = _VERIFIED_CALIBRATION_ISSUER_REGISTRATIONS.pop(token, None)
    runner = None
    for candidate, issuer_reference in tuple(
        _VERIFIED_CALIBRATION_ISSUERS.items()
    ):
        candidate_issuer = issuer_reference()
        if (issuer is not None and candidate_issuer is issuer) or (
            candidate_issuer is not None
            and candidate_issuer._run_token is token
        ):
            runner = candidate
            if issuer is None:
                issuer = candidate_issuer
            _VERIFIED_CALIBRATION_ISSUERS.pop(candidate, None)
            break
    if runner is None and binding is not None:
        candidate = binding.issuer_runner
        if _is_exact_registered_runner(candidate):
            runner = candidate
            _VERIFIED_CALIBRATION_ISSUERS.pop(candidate, None)
    if _is_exact_registered_runner(transaction_runner):
        runner = transaction_runner
    if runner is not None:
        object.__setattr__(runner, "_provenance_binding", None)
        object.__setattr__(runner, "_provenance_issuer", None)


def _lookup_run_token_binding(
    token: VerifiedInstrumentRunToken,
) -> _RunTokenBinding | None:
    """Return the exact registered binding, never class-membership authority."""
    if type(token) is not VerifiedInstrumentRunToken:
        return None
    return _RUN_TOKEN_BINDINGS.get(token)


def _snapshot_run_token_binding_before_success(
    token: VerifiedInstrumentRunToken,
    issuer_runner: _RegisteredEvaluatorRunner,
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
    issuer_runner: _RegisteredEvaluatorRunner,
    instrument: ODMRInstrument,
    success: VerifiedTwoPointCalibrationSuccess,
) -> None:
    """Bind one exact successful outcome/source to its existing token record."""
    from odmr_bench.estimators.two_point_calibration import (
        _consume_verified_source_construction_identity,
    )

    if type(token) is not VerifiedInstrumentRunToken:
        raise TypeError("token must be an exact VerifiedInstrumentRunToken")
    if not _is_exact_registered_runner(issuer_runner):
        raise TypeError("issuer_runner must be a registered exact runner type")
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
    bound = _RunTokenBinding(
        issuer_runner=binding.issuer_runner,
        instrument=binding.instrument,
        instrument_configuration=binding.instrument_configuration,
        success=success,
        source=source,
    )
    object.__setattr__(issuer_runner, "_provenance_binding", bound)
    _RUN_TOKEN_BINDINGS[token] = bound


def _rollback_run_token_success(
    token: VerifiedInstrumentRunToken,
    binding_before: _RunTokenBinding,
) -> None:
    """Restore the trusted binding captured before this success transaction."""
    object.__setattr__(
        binding_before.issuer_runner,
        "_provenance_binding",
        binding_before,
    )
    _RUN_TOKEN_BINDINGS[token] = binding_before
