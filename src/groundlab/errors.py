"""Exception types and the CLI exit codes they map to."""

from __future__ import annotations

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_VERIFICATION_FAILED = 3
EXIT_REFUSED = 4
EXIT_CUDA_UNAVAILABLE = 5


class GroundLabError(Exception):
    """Base class for expected, user-facing failures."""

    exit_code = EXIT_ERROR


class ConfigError(GroundLabError):
    exit_code = EXIT_USAGE


class VerificationError(GroundLabError):
    """A hash, count or cross-check did not match its expected value."""

    exit_code = EXIT_VERIFICATION_FAILED


class SplitAccessError(GroundLabError):
    """The test split was requested without a valid freeze."""

    exit_code = EXIT_REFUSED


class CudaUnavailableError(GroundLabError):
    exit_code = EXIT_CUDA_UNAVAILABLE
