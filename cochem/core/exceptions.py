"""Authoritative Core Exception Hierarchy for CoChem Core.

Adheres to:
- Method Matrix [M] & Provenance Standards
- Zero-Mock Anti-Spoofing Protocol
- Dynamic Mendeleev Invariant Mandate
"""

from __future__ import annotations

from typing import Any, Optional

class CoChemError(Exception):
    """TORQ-local error contract; importing it does not require a BASE service.

    BASE adapters translate errors at their boundary rather than impersonating
    an unavailable sibling package or coupling standalone numerical imports to it.
    """
    def __init__(self, message: str, error_code: str | None = None, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details


class AirGapBoundaryError(CoChemError):
    """AirGapBoundary operation failed."""


class CoordinateShapeError(CoChemError):
    """CoordinateShape operation failed."""


class IsotopeMassResolutionError(CoChemError):
    """IsotopeMassResolution operation failed."""


class IsotopeStabilityError(CoChemError):
    """IsotopeStability operation failed."""


class PESStorageError(CoChemError):
    """PESStorage operation failed."""


class ProcessReaperError(CoChemError):
    """ProcessReaper operation failed."""


class RadiusNotFoundError(CoChemError):
    """RadiusNotFound operation failed."""


class SchemaMigrationError(CoChemError):
    """SchemaMigration operation failed."""


class SubprocessBrokerError(CoChemError):
    """SubprocessBroker operation failed."""


class ThermodynamicsParameterError(CoChemError):
    """ThermodynamicsParameter operation failed."""


class BaseMissingDataError(CoChemError, KeyError):
    """A required value was not provided or could not be resolved."""


class SingularityError(CoChemError, ValueError):
    """The requested operation is undefined at a singular input."""


class MissingDataError(BaseMissingDataError):
    """Raised when required element, isotope, basis set, or calculation data is missing."""

    def __init__(self, message: str, symbol_or_query: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.symbol_or_query = symbol_or_query


class MendeleevInvariantError(MissingDataError):
    """Raised when chemical element or isotopic queries violate Mendeleev physical invariants."""

    def __init__(self, message: str, symbol_or_query: Optional[Any] = None) -> None:
        super().__init__(message, symbol_or_query=symbol_or_query)


class RotationalGridInstabilityError(CoChemError, ValueError):
    """Raised when Cartesian DFT integration grid breaks rotational invariance or induces imaginary modes."""

    def __init__(self, message: str, delta_cm1: Optional[float] = None) -> None:
        super().__init__(message, error_code="COCHEM_E_ROT_GRID_INSTABILITY")
        self.message = message
        self.delta_cm1 = delta_cm1


class JobTimeoutError(CoChemError, TimeoutError):
    """Raised when an asynchronous calculation or subprocess job exceeds temporal limits."""

    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message, error_code="COCHEM_E_JOB_TIMEOUT")


__all__ = [
    "CoChemError",
    "MissingDataError",
    "MendeleevInvariantError",
    "RotationalGridInstabilityError",
    "JobTimeoutError",
    "CoordinateShapeError",
    "AirGapBoundaryError",
    "SchemaMigrationError",
    "PESStorageError",
    "ProcessReaperError",
    "SubprocessBrokerError",
    "ThermodynamicsParameterError",
    "IsotopeMassResolutionError",
    "IsotopeStabilityError",
    "RadiusNotFoundError",
]
