"""Job State and Lifecycle Data Models for CoChem Mobile Asynchronous Execution.

Strictly adhering to SRS Chunk 08 (REQ-MOB-070 through REQ-MOB-073) and the Zero-Mock Mandate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, ConfigDict, Field


class ExecutionTier(str, Enum):
    """Multi-tier environment classification across 6-Tier Environment Matrix."""

    LOCAL_TIER_1 = "TIER_1_LOCAL"
    CONTAINER_TIER_2 = "TIER_2_CONTAINER"
    SSH_HPC_GATEWAY_TIER_3 = "TIER_3_SSH_GATEWAY"
    CODESPACES_TIER_4 = "TIER_4_CODESPACES"
    GITHUB_ACTIONS_TIER_5 = "TIER_5_GHA_CLOUD"
    HPC_SLURM_TIER_6 = "TIER_6_HPC_SLURM"


class JobStatus(str, Enum):
    """Asynchronous job execution lifecycle statuses."""

    QUEUED = "QUEUED"
    STAGED = "STAGED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    ABORTED = "ABORTED"

    def is_terminal(self) -> bool:
        """Return True if status is terminal."""
        return self in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMED_OUT,
            JobStatus.ABORTED,
        }

    def is_active(self) -> bool:
        """Return True if job is still in-flight."""
        return not self.is_terminal()


VALID_STATUS_TRANSITIONS: Dict[JobStatus, Set[JobStatus]] = {
    JobStatus.QUEUED: {
        JobStatus.STAGED,
        JobStatus.RUNNING,
        JobStatus.CANCELLED,
        JobStatus.FAILED,
        JobStatus.ABORTED,
    },
    JobStatus.STAGED: {
        JobStatus.RUNNING,
        JobStatus.CANCELLED,
        JobStatus.FAILED,
        JobStatus.ABORTED,
    },
    JobStatus.RUNNING: {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMED_OUT,
        JobStatus.ABORTED,
    },
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
    JobStatus.TIMED_OUT: set(),
    JobStatus.ABORTED: set(),
}


def validate_status_transition(current: JobStatus, target: JobStatus) -> bool:
    """Validate whether transitioning from current to target status is permitted."""
    if current == target:
        return True
    return target in VALID_STATUS_TRANSITIONS.get(current, set())


class ExecutionPayload(BaseModel):
    """Canonical execution payload for offloading computational chemistry pipelines."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    job_id: str = Field(..., description="Unique UUIDv4 calculation identifier")
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp",
    )
    tier: ExecutionTier = Field(..., description="Target execution tier")
    workflow_type: str = Field(
        ...,
        description="Computational chemistry workflow type (e.g. ORCA_OPT_FREQ, XTB_CONFORMER, RDKIT_CLEAN)",
    )
    molecule_xyz: str = Field(..., description="Standard multi-line Cartesian XYZ coordinate block")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Method Matrix configuration parameters"
    )
    output_artifact_dir: str = Field(
        ..., description="POSIX path under $COCH_ARTIFACTS/jobs/{job_id}/"
    )
    hmac_sha256: Optional[str] = Field(
        default=None, description="HMAC-SHA256 integrity signature computed over canonical payload"
    )

    def to_canonical_dict(self, exclude_signature: bool = True) -> Dict[str, Any]:
        """Convert payload to canonical dictionary representation for signing/hashing."""
        data = self.model_dump()
        if exclude_signature and "hmac_sha256" in data:
            data.pop("hmac_sha256", None)
        return data


class ManifestReference(BaseModel):
    """Staged manifest reference transmitted when execution payload exceeds 64 KB threshold."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    job_id: str = Field(..., description="Unique UUIDv4 calculation identifier")
    manifest_uri: str = Field(..., description="POSIX path or URI to staged payload.json")
    file_size_bytes: int = Field(..., ge=0, description="Size of staged payload file in bytes")
    hmac_sha256: str = Field(
        ..., description="HMAC-SHA256 cryptographic digest of the staged payload file"
    )
    created_at_utc: str = Field(..., description="ISO 8601 creation timestamp")


class JobStatusRecord(BaseModel):
    """Atomic status record persisted to $COCHEM_STATE_DIR/{job_id}.status.json."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    job_id: str = Field(..., description="Unique UUIDv4 calculation identifier")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Current execution status")
    tier: ExecutionTier = Field(..., description="Execution tier")
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Completion percentage"
    )
    current_step: str = Field(default="", description="Human-readable description of current step")
    error_message: Optional[str] = Field(
        default=None, description="Error diagnostics if job failed"
    )
    output_hdf5_path: Optional[str] = Field(
        default=None, description="POSIX path to generated HDF5 dataset"
    )
    updated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 update timestamp",
    )

    def is_terminal(self) -> bool:
        """Check if current job status is terminal."""
        return self.status.is_terminal()
