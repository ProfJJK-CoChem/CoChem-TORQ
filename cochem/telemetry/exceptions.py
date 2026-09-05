"""
Typed exception hierarchy for CoChem Mobile and Telemetry subsystems.

Invariants:
- Zero-Mock & Anti-Spoofing Protocol: Strict typed exception definitions.
- ISO/IEC/IEEE 29148:2018 & CoChem-Mobile SRS Chunk 09.
"""

from __future__ import annotations


class CoChemMOBException(Exception):
    """Base exception for CoChem Mobile and Telemetry subsystems."""


class TelemetryLockError(CoChemMOBException):
    """Raised when file locking on telemetry log stream fails or times out."""


class RingBufferOverflowError(CoChemMOBException):
    """Raised when telemetry ring buffer encounters unrecoverable overflow."""


class VAPIDEncryptionError(CoChemMOBException):
    """Raised when RFC 8291 payload encryption or key derivation fails."""


class PayloadSanitizationViolationError(CoChemMOBException):
    """Raised when sensitive chemical structures or unredacted traces enter push payload."""


class AirGapPushBlockedError(CoChemMOBException):
    """Raised when an unauthorized external SaaS relay is invoked in air-gap mode."""
