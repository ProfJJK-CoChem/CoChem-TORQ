"""
CoChem Telemetry & Real-Time Monitoring Subsystem.

Invariants:
- Zero-Mock & Anti-Spoofing Protocol.
- Real Cryptography (RFC 8291 / RFC 8292).
- OS-Agnostic File Locking via filelock.FileLock.
- Bounded Circular FIFO Ring Buffer (<= 64 KB).
"""

from __future__ import annotations

from cochem.telemetry.buffer import MAX_BUFFER_BYTES, MAX_BUFFER_RECORDS, BoundedRingBuffer
from cochem.telemetry.exceptions import (
    AirGapPushBlockedError,
    CoChemMOBException,
    PayloadSanitizationViolationError,
    RingBufferOverflowError,
    TelemetryLockError,
    VAPIDEncryptionError,
)
from cochem.telemetry.logger import FileLockLogger, resolve_log_path
from cochem.telemetry.push import (
    VAPIDPushEngine,
    check_airgap_compliance,
    sanitize_push_payload,
)
from cochem.telemetry.schemas import (
    PushJobStatus,
    PushNotificationPayload,
    PushSubscriptionEndpoint,
    TelemetryLogLevel,
    TelemetryRecord,
    VAPIDSubscriptionKeys,
)

__all__ = [
    "AirGapPushBlockedError",
    "BoundedRingBuffer",
    "CoChemMOBException",
    "FileLockLogger",
    "MAX_BUFFER_BYTES",
    "MAX_BUFFER_RECORDS",
    "PayloadSanitizationViolationError",
    "PushJobStatus",
    "PushNotificationPayload",
    "PushSubscriptionEndpoint",
    "RingBufferOverflowError",
    "TelemetryLockError",
    "TelemetryLogLevel",
    "TelemetryRecord",
    "VAPIDEncryptionError",
    "VAPIDPushEngine",
    "VAPIDSubscriptionKeys",
    "check_airgap_compliance",
    "resolve_log_path",
    "sanitize_push_payload",
]
