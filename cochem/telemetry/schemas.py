"""
Pydantic v2 schemas for CoChem Mobile & Telemetry subsystems.

Invariants:
- Python 3.11+ type hints, Pydantic v2 schemas.
- Strict validation with extra='forbid' to eliminate IP leakage.
- ISO 8601 UTC timestamp formatting.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TelemetryLogLevel(str, Enum):
    """Log severity levels for structured telemetry."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TelemetryRecord(BaseModel):
    """Structured telemetry log record for append-only JSONL streaming."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(..., ge=0, description="Monotonic sequential record index")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    level: TelemetryLogLevel = Field(
        default=TelemetryLogLevel.INFO, description="Log severity level"
    )
    module: str = Field(..., min_length=1, description="Originating Python module or worker name")
    job_id: Optional[str] = Field(default=None, description="Optional UUIDv4 of calculation job")
    message: str = Field(..., min_length=1, description="Telemetry log message text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Structured key-value metrics"
    )


class PushJobStatus(str, Enum):
    """Terminal state classifications for push notifications."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    HARD_ABORT = "HARD_ABORT"


class PushNotificationPayload(BaseModel):
    """Sanitized zero-trust push alert payload strictly stripped of chemical/code IP."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(..., min_length=1, description="UUIDv4 calculation identifier")
    status: PushJobStatus = Field(..., description="Terminal state of calculation")
    timestamp: str = Field(..., description="ISO 8601 UTC completion timestamp")
    duration_seconds: float = Field(..., ge=0.0, description="Total execution duration in seconds")
    exit_code: int = Field(..., description="Process exit code (0 for success)")
    summary: str = Field(..., max_length=256, description="Redacted operational summary message")


class VAPIDSubscriptionKeys(BaseModel):
    """Client Web Push subscription keys (P-256 and auth token)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p256dh: str = Field(..., min_length=16, description="Base64url-encoded client P-256 public key")
    auth: str = Field(
        ..., min_length=16, description="Base64url-encoded client 16-byte auth secret"
    )


class PushSubscriptionEndpoint(BaseModel):
    """Web Push subscription endpoint schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str = Field(..., min_length=10, description="Browser push service URL")
    keys: VAPIDSubscriptionKeys = Field(..., description="Client cryptographic keys")
