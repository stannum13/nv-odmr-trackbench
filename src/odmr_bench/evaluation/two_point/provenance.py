"""Private provenance capability construction for two-point evaluation."""

from odmr_bench.evaluation.two_point.types import VerifiedInstrumentRunToken

_TOKEN_CONSTRUCTION_KEY: object = object()


def _mint_verified_instrument_run_token(
    construction_key: object,
) -> VerifiedInstrumentRunToken:
    """Create a runner-issued token after the evaluator-private key check."""
    if construction_key is not _TOKEN_CONSTRUCTION_KEY:
        raise TypeError("invalid verified instrument run token construction key")
    return object.__new__(VerifiedInstrumentRunToken)
