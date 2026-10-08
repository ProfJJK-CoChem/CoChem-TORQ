"""CoChem-TORQ: Environment Bootstrapper & IPC Resource Manager.

=============================================================
Phase 1 (Stage 0.0) Implementation
----------------------------------
Provides environment initialization, mathematical air-gap boundary verification,
dynamic artifact/scratch directory routing, cross-platform memory-mapped IPC
scratch buffers, and deterministic atexit resource reclamation for the
CoChem-TORQ ecosystem.

Architecture & Directives:
- Tripartite Filesystem Air-Gap: Mathematically asserts that working directory
  is strictly isolated and disjoint from artifact and scratch directories.
- Dynamic Path Routing: Queries host environment variables (e.g. COCHEM_ARTIFACTS,
  COCHEM_SCRATCH) falling back dynamically to secure system temp subdirectories.
  Zero hardcoded paths permitted.
- Cross-Platform IPC Buffers: Memory-mapped files (mmap) and shared memory
  (multiprocessing.shared_memory.SharedMemory) with unified context management.
- Deterministic Garbage Collection: Automatic atexit handler registration and
  manual cleanup trigger functions for safe resource wiping and unlinking.
- Strict Pydantic v2 data models for metadata and configuration.
- Module-level logging via standard logging infrastructure.

Authoritative Sources:
- CoChem Architecture Specification: Phase 1 (Stage 0.0)
- Method Matrix: Heterogeneous Orchestration & Air-Gap Directives
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import mmap
import multiprocessing.shared_memory as sm  # zero-empty_block IPC shared memory
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

# Configure module-level logging
logger = logging.getLogger("CoChem-TORQ.Init")

# Global tracking sets for registered IPC resources
_ACTIVE_SCRATCH_PATHS: Final[set[Path]] = set()
_ACTIVE_SHM_NAMES: Final[set[str]] = set()
_ACTIVE_SHM_OBJECTS: Final[list[sm.SharedMemory]] = []
_ACTIVE_MMAP_OBJECTS: Final[list[mmap.mmap]] = []
_ACTIVE_BUFFERS: Final[list[IPCScratchBuffer]] = []
_ACTIVE_FILE_DESCRIPTORS: Final[set[int]] = set()
_ATEXIT_REGISTERED: bool = False

__all__ = [
    "AirGapViolationError",
    "IPCBufferError",
    "ProvenanceIntegrityError",
    "ProvenanceHashMismatchError",
    "AirGapReport",
    "FitProvenanceReport",
    "IPCBufferMetadata",
    "BootstrapperConfig",
    "get_artifact_directory",
    "get_scratch_directory",
    "check_airgap",
    "verify_airgap",
    "format_airgap_error",
    "prompt_airgap_remediation",
    "IPCScratchBuffer",
    "create_ipc_scratch_buffer",
    "register_mmap_buffer",
    "register_ipc_cleanup",
    "unregister_ipc_cleanup",
    "cleanup_ipc_scratch",
    "canonicalize_provenance_json",
    "compute_provenance_sha256",
    "compute_file_sha256",
    "validate_fit_provenance",
    "bootstrap_environment",
]


# ============================================================================
# Exceptions
# ============================================================================


class AirGapViolationError(RuntimeError):
    """Raised when working directory and artifact directory overlap or intersect.

    Hard-fails deterministically on Air-Gap violations with rich diagnostic details,
    explicit rejection of silent auto-moving (to avoid breaking relative paths and
    violating operational predictability), and concrete remediation commands.
    """

    def __init__(
        self,
        message: str,
        cwd: Path | str | None = None,
        artifacts_dir: Path | str | None = None,
        reason: str | None = None,
        remediation_command: str | None = None,
        target_path: Path | str | None = None,
        repo_root: Path | str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cwd = Path(cwd).resolve() if cwd is not None else None
        self.artifacts_dir = Path(artifacts_dir).resolve() if artifacts_dir is not None else None
        self.target_path = Path(target_path).resolve() if target_path is not None else self.artifacts_dir
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else self.cwd
        self.reason = reason
        self.remediation_command = remediation_command
        self.auto_move_rejected: bool = True

    def __str__(self) -> str:
        return self.message


class IPCBufferError(RuntimeError):
    """Raised when memory-mapped buffer creation, access, or teardown fails."""


class ProvenanceIntegrityError(RuntimeError):
    """Raised when fit_provenance.json integrity, formatting, or dataset binding fails."""


class ProvenanceHashMismatchError(ProvenanceIntegrityError):
    """Raised when canonical SHA-256 validation of fit_provenance.json fails."""


# ============================================================================
# Pydantic v2 Models
# ============================================================================


class AirGapReport(BaseModel):
    """Immutable diagnostic report detailing air-gap boundary verification."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(
        ..., description="True if paths are mathematically disjoint and safe"
    )
    cwd_resolved: Path = Field(
        ..., description="Canonical resolved current working directory"
    )
    artifacts_resolved: Path = Field(
        ..., description="Canonical resolved runtime artifacts directory"
    )
    verified_at: float = Field(
        default_factory=time.time, description="Unix timestamp of verification check"
    )
    reason: str | None = Field(
        default=None, description="Diagnostic explanation of air-gap violation if any"
    )


class FitProvenanceReport(BaseModel):
    """Diagnostic report describing fit_provenance.json validation results."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(
        ..., description="True if SHA-256 hash matches canonical payload"
    )
    provenance_path: Path = Field(
        ..., description="Canonical path to evaluated fit_provenance.json"
    )
    computed_sha256: str = Field(
        ..., description="Calculated canonical SHA-256 digest"
    )
    expected_sha256: str | None = Field(
        default=None, description="Expected SHA-256 digest from file or caller"
    )
    dataset_hashes_verified: int = Field(
        default=0, description="Count of verified dataset file hashes"
    )
    verified_at: float = Field(
        default_factory=time.time, description="Unix timestamp of verification check"
    )
    diagnostics: str | None = Field(
        default=None, description="Diagnostic explanation of validation failure if any"
    )


class IPCBufferMetadata(BaseModel):
    """Pydantic v2 model describing IPC scratch buffer properties and lifecycle."""

    model_config = ConfigDict(extra="ignore")

    buffer_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the buffer",
    )
    name: str = Field(
        ..., description="Human-readable or OS kernel name for the buffer"
    )
    buffer_type: Literal["mmap", "shm"] = Field(
        ...,
        description="Type of IPC buffer ('mmap' or 'shm')",
    )
    size_bytes: int = Field(
        ..., gt=0, description="Allocated buffer capacity in bytes (must be > 0)"
    )
    file_path: Path | None = Field(
        default=None,
        description="Path to physical backing file if buffer_type == 'mmap'",
    )
    created_at: float = Field(
        default_factory=time.time, description="Unix timestamp of buffer allocation"
    )
    pid: int = Field(
        default_factory=os.getpid, description="Process ID that created the buffer"
    )
    is_active: bool = Field(
        default=True, description="True if buffer is currently open and mapped"
    )


class BootstrapperConfig(BaseModel):
    """Validated configuration for TORQ runtime environment initialization."""

    model_config = ConfigDict(extra="ignore")

    artifacts_dir: Path = Field(
        ..., description="Resolved directory for persistent artifacts"
    )
    scratch_dir: Path = Field(
        ..., description="Resolved directory for fast ephemeral scratch files"
    )
    env_var: str = Field(
        default="COCHEM_ARTIFACTS",
        description="Environment variable queried for artifacts directory",
    )
    scratch_env_var: str = Field(
        default="COCHEM_SCRATCH",
        description="Environment variable queried for scratch directory",
    )
    enforce_airgap: bool = Field(
        default=True, description="Whether air-gap validation is enforced at startup"
    )
    clean_on_exit: bool = Field(
        default=True, description="Whether automatic atexit IPC cleanup is registered"
    )
    validate_provenance: bool = Field(
        default=True,
        description="Whether fit_provenance.json SHA-256 validation is enforced at startup",
    )
    fit_provenance_path: Path | None = Field(
        default=None,
        description="Path to validated fit_provenance.json if present or discovered",
    )
    provenance_hash: str | None = Field(
        default=None,
        description="Validated canonical SHA-256 hash of fit_provenance.json",
    )
    provenance_report: FitProvenanceReport | None = Field(
        default=None,
        description="Full diagnostic report of fit_provenance.json validation",
    )


# ============================================================================
# Dynamic Directory Mapping
# ============================================================================


def get_artifact_directory(
    env_var: str = "COCHEM_ARTIFACTS",
    fallback_dir: Path | str | None = None,
    create_if_missing: bool = True,
) -> Path:
    """Establish dynamic mapping of artifact directories by querying host environment.

    Zero hardcoded paths allowed. Queries `env_var`, then `fallback_dir`, and
    falls back dynamically to a secure system temp subdirectory.

    Args:
        env_var: Environment variable name to query (default: 'COCHEM_ARTIFACTS').
        fallback_dir: Optional fallback path if environment variable is unset.
        create_if_missing: If True, creates the target directory on disk.

    Returns:
        Canonical resolved `Path` to the artifact directory.
    """
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        target = Path(env_val).expanduser().resolve()
    elif fallback_dir is not None:
        target = Path(fallback_dir).expanduser().resolve()
    else:
        target = (Path(tempfile.gettempdir()) / "cochem_artifacts").resolve()

    if create_if_missing:
        os.makedirs(target, exist_ok=True)
    return target


def get_scratch_directory(
    env_var: str = "COCHEM_SCRATCH",
    fallback_dir: Path | str | None = None,
    create_if_missing: bool = True,
) -> Path:
    """Establish dynamic mapping of ephemeral scratch directories.

    Queries `env_var`, then `fallback_dir`, and falls back to a PID-isolated
    directory under the system temp directory (`<tempdir>/cochem_scratch/pid_<PID>`).

    Args:
        env_var: Environment variable name to query (default: 'COCHEM_SCRATCH').
        fallback_dir: Optional fallback path if environment variable is unset.
        create_if_missing: If True, creates the target directory on disk.

    Returns:
        Canonical resolved `Path` to the ephemeral scratch directory.
    """
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        target = Path(env_val).expanduser().resolve()
    elif fallback_dir is not None:
        target = Path(fallback_dir).expanduser().resolve()
    else:
        target = (
            Path(tempfile.gettempdir()) / "cochem_scratch" / f"pid_{os.getpid()}"
        ).resolve()

    if create_if_missing:
        os.makedirs(target, exist_ok=True)
    return target


# ============================================================================
# Air-Gap Boundary Verification
# ============================================================================


def check_airgap(
    cwd: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
) -> AirGapReport:
    """Perform mathematical evaluation of air-gap separation between cwd and artifacts.

    Args:
        cwd: Working directory to test (defaults to `Path.cwd()`).
        artifacts_dir: Artifacts directory to test (defaults to dynamic artifacts dir).

    Returns:
        `AirGapReport` detailing validity, timestamps, and diagnostic reason.
    """
    resolved_cwd = Path(cwd if cwd is not None else Path.cwd()).expanduser().resolve()
    if artifacts_dir is not None:
        resolved_artifacts = Path(artifacts_dir).expanduser().resolve()
    else:
        resolved_artifacts = (
            get_artifact_directory(create_if_missing=False).expanduser().resolve()
        )

    # Normalize cases for cross-platform comparison
    cwd_str = os.path.normcase(str(resolved_cwd))
    art_str = os.path.normcase(str(resolved_artifacts))

    # Check 1: Identical paths (intersects / overlaps)
    if cwd_str == art_str:
        return AirGapReport(
            is_valid=False,
            cwd_resolved=resolved_cwd,
            artifacts_resolved=resolved_artifacts,
            verified_at=time.time(),
            reason=(
                f"Working directory and artifacts directory intersect "
                f"(identical paths: '{resolved_cwd}')"
            ),
        )

    # Check 2: Artifacts directory located inside cwd
    try:
        if resolved_artifacts.is_relative_to(resolved_cwd):
            return AirGapReport(
                is_valid=False,
                cwd_resolved=resolved_cwd,
                artifacts_resolved=resolved_artifacts,
                verified_at=time.time(),
                reason=(
                    f"Artifacts directory '{resolved_artifacts}' is located inside / "
                    f"intersects the working directory '{resolved_cwd}'"
                ),
            )
    except (ValueError, AttributeError):
        pass

    # Check 3: CWD located inside artifacts directory
    try:
        if resolved_cwd.is_relative_to(resolved_artifacts):
            return AirGapReport(
                is_valid=False,
                cwd_resolved=resolved_cwd,
                artifacts_resolved=resolved_artifacts,
                verified_at=time.time(),
                reason=(
                    f"Working directory '{resolved_cwd}' is located inside / "
                    f"intersects the artifacts directory '{resolved_artifacts}'"
                ),
            )
    except (ValueError, AttributeError):
        pass

    # Check 4: Path parent hierarchy intersection check
    if (
        resolved_cwd in resolved_artifacts.parents
        or resolved_artifacts in resolved_cwd.parents
    ):
        return AirGapReport(
            is_valid=False,
            cwd_resolved=resolved_cwd,
            artifacts_resolved=resolved_artifacts,
            verified_at=time.time(),
            reason=(
                f"Path intersection and overlap detected between working "
                f"directory '{resolved_cwd}' and artifacts directory "
                f"'{resolved_artifacts}'"
            ),
        )

    # Check 5: Normalized string path containment check (cross-platform safeguard)
    cwd_prefix = cwd_str if cwd_str.endswith(os.sep) else (cwd_str + os.sep)
    art_prefix = art_str if art_str.endswith(os.sep) else (art_str + os.sep)
    if art_str.startswith(cwd_prefix):
        return AirGapReport(
            is_valid=False,
            cwd_resolved=resolved_cwd,
            artifacts_resolved=resolved_artifacts,
            verified_at=time.time(),
            reason=(
                f"Artifacts directory '{resolved_artifacts}' is a subpath of "
                f"working directory '{resolved_cwd}'"
            ),
        )
    if cwd_str.startswith(art_prefix):
        return AirGapReport(
            is_valid=False,
            cwd_resolved=resolved_cwd,
            artifacts_resolved=resolved_artifacts,
            verified_at=time.time(),
            reason=(
                f"Working directory '{resolved_cwd}' is a subpath of "
                f"artifacts directory '{resolved_artifacts}'"
            ),
        )

    return AirGapReport(
        is_valid=True,
        cwd_resolved=resolved_cwd,
        artifacts_resolved=resolved_artifacts,
        verified_at=time.time(),
        reason=None,
    )


def format_airgap_error(report: AirGapReport) -> str:
    """Format an informative, descriptive error banner for an AirGapReport violation.

    Explicitly articulates that silent auto-moving of files is strictly rejected
    to prevent breaking relative paths and violating operational predictability,
    and provides concrete cross-platform configuration and remediation commands.
    """
    posix_env = 'export COCHEM_ARTIFACTS="$HOME/CoChem_Artifacts" && export COCHEM_SCRATCH="/tmp/cochem_scratch"'
    ps_env = '$env:COCHEM_ARTIFACTS="$HOME\\CoChem_Artifacts"; $env:COCHEM_SCRATCH="$env:TEMP\\cochem_scratch"'

    banner = (
        f"\n{'=' * 80}\n"
        f"[HARD_ABORT: AIR-GAP VIOLATION] Tripartite Filesystem Boundary Breach\n"
        f"{'=' * 80}\n"
        f"Failure Mode       : AirGapViolationError: Air-gap integrity violated! {report.reason}.\n"
        f"Working Directory  : {report.cwd_resolved}\n"
        f"Artifacts Directory: {report.artifacts_resolved}\n"
        f"Detail             : CoChem-TORQ forbids writing runtime artifacts inside or overlapping the working repository space.\n"
        f"\n"
        f"Auto-Move Policy:\n"
        f"  [AUTO-MOVE FORBIDDEN] CoChem-TORQ strictly rejects silent auto-moving of files.\n"
        f"  Auto-relocating files silently breaks relative paths, corrupts pipeline provenance,\n"
        f"  and violates deterministic operational predictability. Execution has hard-aborted.\n"
        f"\n"
        f"Actionable Remediation Protocol (SRS Doc 2 Part 1 §1.1):\n"
        f"  1. Configure disjoint directories outside the repository tree:\n"
        f"     POSIX/Bash        : {posix_env}\n"
        f"     Windows PowerShell: {ps_env}\n"
        f"  2. Re-verify runtime environment initialization:\n"
        f"     python -c \"from Libraries.cochem_torq_init import bootstrap_environment; bootstrap_environment()\"\n"
        f"{'=' * 80}"
    )
    return banner


def prompt_airgap_remediation(
    error: AirGapViolationError,
    interactive: bool = False,
) -> str:
    """Provides a formatted remediation summary or interactive prompt options for AirGapViolationError."""
    remediation = (
        f"\n[REMEDIATION GUIDE FOR AIR-GAP BREACH]\n"
        f"Reason        : {error.reason or 'Working directory and artifacts directory intersect.'}\n"
        f"Working Dir   : {error.cwd}\n"
        f"Artifacts Dir : {error.artifacts_dir}\n"
        f"Suggested Fix :\n"
        f"  {error.remediation_command or 'Configure disjoint COCHEM_ARTIFACTS and COCHEM_SCRATCH environment variables.'}\n"
    )
    if interactive and sys.stdin.isatty():
        try:
            input(f"{remediation}\nPress Enter once manual remediation is completed to proceed, or Ctrl+C to abort...")
        except (KeyboardInterrupt, EOFError):
            pass
    return remediation


def verify_airgap(
    cwd: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
) -> bool:
    """Mathematically assert that cwd is disjoint from the runtime artifacts directory.

    Resolves both paths to absolute canonical locations and verifies that neither
    path contains or equals the other.

    Args:
        cwd: Working directory (defaults to `Path.cwd()`).
        artifacts_dir: Artifacts directory (defaults to `get_artifact_directory()`).

    Returns:
        `True` if air-gap separation is strictly satisfied.

    Raises:
        `AirGapViolationError`: If paths intersect, overlap, or contain each other.
    """
    report = check_airgap(cwd=cwd, artifacts_dir=artifacts_dir)
    if not report.is_valid:
        error_msg = format_airgap_error(report)
        logger.error(error_msg)
        posix_remed = 'export COCHEM_ARTIFACTS="$HOME/CoChem_Artifacts" && export COCHEM_SCRATCH="/tmp/cochem_scratch"'
        raise AirGapViolationError(
            message=error_msg,
            cwd=report.cwd_resolved,
            artifacts_dir=report.artifacts_resolved,
            reason=report.reason,
            remediation_command=posix_remed,
        )

    logger.info(
        "Air-gap boundary verified: cwd='%s' is isolated from artifacts_dir='%s'.",
        report.cwd_resolved,
        report.artifacts_resolved,
    )
    return True


# ============================================================================
# IPC Scratch Buffer Wrapper
# ============================================================================


class IPCScratchBuffer:
    """Cross-platform wrapper for memory-mapped files and SharedMemory buffers.

    Provides uniform read, write, flush, close, unlink, and context manager
    semantics across POSIX and Windows operating systems.
    """

    def __init__(
        self,
        metadata: IPCBufferMetadata,
        mmap_obj: mmap.mmap | None = None,
        shm_obj: sm.SharedMemory | None = None,
        file_obj: Any | None = None,
        path: Path | None = None,
    ) -> None:
        self.metadata = metadata
        self._mmap = mmap_obj
        self._shm = shm_obj
        self._file_obj = file_obj
        self._path = path
        self._closed = False

    @property
    def path(self) -> Path | None:
        """Backing file path for file-backed mmap buffer, or None if pure shm."""
        return self._path

    @property
    def name(self) -> str:
        """Name of the buffer."""
        return self.metadata.name

    @property
    def size(self) -> int:
        """Size of the buffer in bytes."""
        return self.metadata.size_bytes

    @property
    def is_closed(self) -> bool:
        """True if the buffer has been closed."""
        return self._closed

    @property
    def buffer(self) -> memoryview | mmap.mmap:
        """Access underlying buffer view."""
        if self._closed:
            raise IPCBufferError(f"Cannot access closed buffer '{self.name}'.")
        if self._mmap is not None:
            return self._mmap
        if self._shm is not None:
            if self._shm.buf is None:
                raise IPCBufferError(f"SharedMemory '{self.name}' has no active buffer.")
            return self._shm.buf
        raise IPCBufferError(f"No active buffer backing in '{self.name}'.")

    def write(self, data: bytes | bytearray | memoryview, offset: int = 0) -> int:
        """Write raw byte payload into buffer at specified offset.

        Args:
            data: Bytes or buffer to write.
            offset: Starting byte position (default: 0).

        Returns:
            Number of bytes written.

        Raises:
            IPCBufferError: If buffer is closed, or offset + length exceeds buffer size.
        """
        if self._closed:
            raise IPCBufferError(f"Cannot write to closed buffer '{self.name}'.")
        data_len = len(data)
        if offset < 0 or (offset + data_len) > self.size:
            raise IPCBufferError(
                f"Write boundary violation in '{self.name}': offset={offset}, "
                f"data_len={data_len}, buffer_capacity={self.size} bytes."
            )

        if self._mmap is not None:
            self._mmap[offset : offset + data_len] = data
            self._mmap.flush()
            return data_len
        elif self._shm is not None:
            if self._shm.buf is None:
                raise IPCBufferError(f"SharedMemory '{self.name}' has no active buffer.")
            self._shm.buf[offset : offset + data_len] = data
            return data_len
        else:
            raise IPCBufferError(f"Buffer '{self.name}' has no active storage.")

    def read(self, size: int = -1, offset: int = 0) -> bytes:
        """Read bytes from buffer at specified offset.

        Args:
            size: Number of bytes to read (-1 reads all remaining bytes).
            offset: Starting byte position (default: 0).

        Returns:
            `bytes` object containing read content.

        Raises:
            IPCBufferError: If buffer is closed or offset is out of bounds.
        """
        if self._closed:
            raise IPCBufferError(f"Cannot read from closed buffer '{self.name}'.")
        if offset < 0 or offset > self.size:
            raise IPCBufferError(
                f"Read offset out of bounds in '{self.name}': "
                f"offset={offset}, capacity={self.size}."
            )

        length = (self.size - offset) if size < 0 else min(size, self.size - offset)
        if (offset + length) > self.size:
            raise IPCBufferError(
                f"Read boundary violation in '{self.name}': offset={offset}, "
                f"length={length}, buffer_capacity={self.size}."
            )

        if self._mmap is not None:
            return bytes(self._mmap[offset : offset + length])
        elif self._shm is not None:
            if self._shm.buf is None:
                raise IPCBufferError(f"SharedMemory '{self.name}' has no active buffer.")
            return bytes(self._shm.buf[offset : offset + length])
        else:
            raise IPCBufferError(f"Buffer '{self.name}' has no active storage.")

    def flush(self) -> None:
        """Flush buffer modifications to storage."""
        if not self._closed and self._mmap is not None:
            self._mmap.flush()

    def close(self) -> None:
        """Close buffer and underlying file handles."""
        if self._closed:
            return

        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception as e:
                logger.warning("Error closing mmap buffer '%s': %s", self.name, e)
            self._mmap = None

        if self._file_obj is not None:
            try:
                self._file_obj.close()
            except Exception as e:
                logger.warning("Error closing backing file for '%s': %s", self.name, e)
            self._file_obj = None

        if self._shm is not None:
            try:
                self._shm.close()
            except Exception as e:
                logger.warning("Error closing shared memory '%s': %s", self.name, e)

        self._closed = True
        self.metadata.is_active = False

    def unlink(self) -> None:
        """Unlink and permanently destroy the physical buffer storage."""
        path_to_unlink = self._path
        shm_to_unlink = self._shm
        mmap_to_close = self._mmap
        buf_name = self.name

        self.close()

        if shm_to_unlink is not None:
            try:
                shm_to_unlink.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("Error unlinking shared memory '%s': %s", self.name, e)
            self._shm = None

        if path_to_unlink is not None and path_to_unlink.exists():
            try:
                path_to_unlink.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("Error removing backing file '%s': %s", path_to_unlink, e)
            self._path = None

        # Cleanly deregister from global tracking registry to avoid memory leaks
        unregister_ipc_cleanup(
            buffers=[self],
            scratch_paths=[path_to_unlink] if path_to_unlink is not None else None,
            shm_names=[buf_name] if self.metadata.buffer_type == "shm" else None,
            shm_objects=[shm_to_unlink] if shm_to_unlink is not None else None,
            mmap_objects=[mmap_to_close] if mmap_to_close is not None else None,
        )

    def __enter__(self) -> IPCScratchBuffer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unlink()


# ============================================================================
# IPC Buffer Creation & Registration Helpers
# ============================================================================


def create_ipc_scratch_buffer(
    name: str | None = None,
    size: int = 4096,
    buffer_type: Literal["mmap", "shm"] | str = "mmap",
    scratch_dir: Path | str | None = None,
    auto_register: bool = True,
) -> IPCScratchBuffer:
    """Create and initialize a cross-platform IPC scratch buffer.

    Supports both file-backed `mmap` and operating system `SharedMemory`.

    Args:
        name: Name of buffer (generates unique name if None).
        size: Size in bytes (must be > 0).
        buffer_type: Either 'mmap' or 'shm' (multiprocessing shared memory).
        scratch_dir: Directory for mmap file backing.
        auto_register: If True, automatically registers with `register_ipc_cleanup`.

    Returns:
        Configured `IPCScratchBuffer` instance.

    Raises:
        IPCBufferError: If buffer allocation or memory mapping fails.
    """
    if size <= 0:
        raise IPCBufferError(f"Buffer size must be positive, got {size} bytes.")

    normalized_type = str(buffer_type).lower()
    if normalized_type not in ("mmap", "shm"):
        raise IPCBufferError(
            f"Unsupported buffer_type '{buffer_type}'. Must be 'mmap' or 'shm'."
        )

    buf_name = name or f"cochem_torq_ipc_{uuid.uuid4().hex[:12]}"

    if normalized_type == "mmap":
        base_dir = (
            Path(scratch_dir).resolve()
            if scratch_dir is not None
            else get_scratch_directory()
        )
        os.makedirs(base_dir, exist_ok=True)
        file_path = base_dir / f"{buf_name}.dat"

        try:
            with open(file_path, "w+b") as f:
                f.truncate(size)
                f.flush()
                mm = mmap.mmap(f.fileno(), size)
        except Exception as exc:
            raise IPCBufferError(
                f"Failed to create mmap scratch buffer at '{file_path}': {exc}"
            ) from exc

        meta = IPCBufferMetadata(
            name=buf_name,
            buffer_type="mmap",
            size_bytes=size,
            file_path=file_path,
        )
        scratch_buf = IPCScratchBuffer(
            metadata=meta,
            mmap_obj=mm,
            file_obj=None,
            path=file_path,
        )

        if auto_register:
            register_ipc_cleanup(
                scratch_paths=[file_path],
                mmap_objects=[mm],
                buffers=[scratch_buf],
            )

        return scratch_buf

    else:  # shm
        try:
            shm_obj = sm.SharedMemory(name=buf_name, create=True, size=size)
        except Exception as exc:
            raise IPCBufferError(
                f"Failed to create SharedMemory segment '{buf_name}': {exc}"
            ) from exc

        meta = IPCBufferMetadata(
            name=buf_name,
            buffer_type="shm",
            size_bytes=size,
        )
        scratch_buf = IPCScratchBuffer(
            metadata=meta,
            shm_obj=shm_obj,
        )

        if auto_register:
            register_ipc_cleanup(
                shm_names=[buf_name],
                shm_objects=[shm_obj],
                buffers=[scratch_buf],
            )

        return scratch_buf


def register_mmap_buffer(
    mm: mmap.mmap,
    backing_path: Path | str | None = None,
) -> None:
    """Register an existing mmap buffer and its backing file for atexit cleanup.

    Args:
        mm: Open `mmap.mmap` buffer object.
        backing_path: Optional path to backing file on disk.
    """
    paths: list[Path] = []
    if backing_path is not None:
        paths.append(Path(backing_path).resolve())

    register_ipc_cleanup(
        scratch_paths=paths if paths else None,
        mmap_objects=[mm],
    )


# ============================================================================
# Deterministic Cleanup & atexit Handlers
# ============================================================================


def _ensure_atexit_registered() -> None:
    """Ensure that the atexit IPC cleanup handler is registered exactly once."""
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_cleanup_handler)
        _ATEXIT_REGISTERED = True


def _atexit_cleanup_handler() -> None:
    """Internal handler on process exit to reclaim registered IPC resources."""
    logger.debug("atexit handler invoked: reclaiming registered TORQ IPC resources...")
    cleanup_ipc_scratch()


def register_ipc_cleanup(
    scratch_paths: Sequence[Path | str] | None = None,
    shm_names: Sequence[str] | None = None,
    shm_objects: Sequence[sm.SharedMemory] | None = None,
    mmap_objects: Sequence[mmap.mmap] | None = None,
    buffers: Sequence[IPCScratchBuffer] | None = None,
    file_descriptors: Sequence[int] | None = None,
) -> None:
    """Register physical scratch paths, shared memory blocks, and mmap buffers.

    Args:
        scratch_paths: Files or directories to unlink/wipe upon exit.
        shm_names: Named shared memory segments to unlink.
        shm_objects: `SharedMemory` instances to close and unlink.
        mmap_objects: `mmap.mmap` buffer instances to close.
        buffers: `IPCScratchBuffer` instances to close/unlink.
        file_descriptors: Open integer file descriptors to close.
    """
    _ensure_atexit_registered()

    if scratch_paths:
        for p in scratch_paths:
            _ACTIVE_SCRATCH_PATHS.add(Path(p).resolve())

    if shm_names:
        for name in shm_names:
            if name:
                _ACTIVE_SHM_NAMES.add(name)

    if shm_objects:
        for shm in shm_objects:
            if not any(x is shm for x in _ACTIVE_SHM_OBJECTS):
                _ACTIVE_SHM_OBJECTS.append(shm)
                _ACTIVE_SHM_NAMES.add(shm.name)

    if mmap_objects is not None:
        for mm in mmap_objects:
            if not any(x is mm for x in _ACTIVE_MMAP_OBJECTS):
                _ACTIVE_MMAP_OBJECTS.append(mm)

    if buffers is not None:
        for buf in buffers:
            if not any(x is buf for x in _ACTIVE_BUFFERS):
                _ACTIVE_BUFFERS.append(buf)

    if file_descriptors is not None:
        for fd in file_descriptors:
            _ACTIVE_FILE_DESCRIPTORS.add(fd)


def unregister_ipc_cleanup(
    scratch_paths: Sequence[Path | str] | None = None,
    shm_names: Sequence[str] | None = None,
    shm_objects: Sequence[sm.SharedMemory] | None = None,
    mmap_objects: Sequence[mmap.mmap] | None = None,
    buffers: Sequence[IPCScratchBuffer] | None = None,
    file_descriptors: Sequence[int] | None = None,
) -> None:
    """Deregister tracking references using identity to prevent mmap equality errors."""
    if scratch_paths is not None:
        for p in scratch_paths:
            _ACTIVE_SCRATCH_PATHS.discard(Path(p).resolve())

    if shm_names is not None:
        for name in shm_names:
            if name:
                _ACTIVE_SHM_NAMES.discard(name)

    if shm_objects is not None:
        for shm in shm_objects:
            _ACTIVE_SHM_OBJECTS[:] = [x for x in _ACTIVE_SHM_OBJECTS if x is not shm]
            try:
                _ACTIVE_SHM_NAMES.discard(shm.name)
            except Exception:
                pass

    if mmap_objects is not None:
        for mm in mmap_objects:
            _ACTIVE_MMAP_OBJECTS[:] = [x for x in _ACTIVE_MMAP_OBJECTS if x is not mm]

    if buffers is not None:
        for buf in buffers:
            _ACTIVE_BUFFERS[:] = [x for x in _ACTIVE_BUFFERS if x is not buf]

    if file_descriptors is not None:
        for fd in file_descriptors:
            _ACTIVE_FILE_DESCRIPTORS.discard(fd)


def cleanup_ipc_scratch(
    scratch_paths: Sequence[Path | str] | None = None,
    shm_names: Sequence[str] | None = None,
    shm_objects: Sequence[sm.SharedMemory] | None = None,
    mmap_objects: Sequence[mmap.mmap] | None = None,
    buffers: Sequence[IPCScratchBuffer] | None = None,
    file_descriptors: Sequence[int] | None = None,
) -> dict[str, int]:
    """Deterministically reclaim, unlink, and wipe IPC scratch resources.

    If no arguments are provided, reclaims all resources currently tracked in
    the global registry. When explicit targets are provided, cleans up ONLY
    the targeted resources without destroying other active allocations.

    Returns:
        Dictionary summarizing the count of reclaimed resources:
        `{"files_removed": int, "directories_removed": int, ...}`
    """
    summary: dict[str, int] = {
        "files_removed": 0,
        "directories_removed": 0,
        "shm_unlinked": 0,
        "mmaps_closed": 0,
        "fds_closed": 0,
    }

    is_full_cleanup = (
        scratch_paths is None
        and shm_names is None
        and shm_objects is None
        and mmap_objects is None
        and buffers is None
        and file_descriptors is None
    )

    # 1. Close active IPCScratchBuffer wrappers
    targets_buffers = (
        list(_ACTIVE_BUFFERS)
        if is_full_cleanup
        else (list(buffers) if buffers is not None else [])
    )
    for buf in targets_buffers:
        try:
            buf.close()
        except Exception as e:
            logger.debug("IPCScratchBuffer close notice: %s", e)
        _ACTIVE_BUFFERS[:] = [x for x in _ACTIVE_BUFFERS if x is not buf]

    # 2. Close active mmap objects
    targets_mmap = (
        list(_ACTIVE_MMAP_OBJECTS)
        if is_full_cleanup
        else (list(mmap_objects) if mmap_objects is not None else [])
    )
    for mm in targets_mmap:
        try:
            mm.close()
            summary["mmaps_closed"] += 1
        except Exception as e:
            logger.debug("mmap close notice: %s", e)
        _ACTIVE_MMAP_OBJECTS[:] = [x for x in _ACTIVE_MMAP_OBJECTS if x is not mm]

    # 3. Close and unlink SharedMemory objects
    targets_shm_objs = (
        list(_ACTIVE_SHM_OBJECTS)
        if is_full_cleanup
        else (list(shm_objects) if shm_objects else [])
    )
    for shm in targets_shm_objs:
        try:
            shm.close()
        except Exception as e:
            logger.debug("SharedMemory close notice: %s", e)
        try:
            shm.unlink()
            summary["shm_unlinked"] += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("SharedMemory unlink notice: %s", e)
        _ACTIVE_SHM_OBJECTS[:] = [x for x in _ACTIVE_SHM_OBJECTS if x is not shm]

    # 4. Unlink named SharedMemory segments
    targets_shm_names = (
        set(_ACTIVE_SHM_NAMES)
        if is_full_cleanup
        else (set(shm_names) if shm_names else set())
    )
    for name in targets_shm_names:
        try:
            temp_shm = sm.SharedMemory(name=name, create=False)
            temp_shm.close()
            temp_shm.unlink()
            summary["shm_unlinked"] += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("Named SharedMemory unlink notice for '%s': %s", name, e)
        _ACTIVE_SHM_NAMES.discard(name)

    # 5. Close file descriptors
    targets_fds = (
        set(_ACTIVE_FILE_DESCRIPTORS)
        if is_full_cleanup
        else (set(file_descriptors) if file_descriptors else set())
    )
    for fd in targets_fds:
        try:
            os.close(fd)
            summary["fds_closed"] += 1
        except OSError:
            pass
        _ACTIVE_FILE_DESCRIPTORS.discard(fd)

    # 6. Remove scratch files and directories
    targets_paths = (
        set(_ACTIVE_SCRATCH_PATHS)
        if is_full_cleanup
        else ({Path(p).resolve() for p in scratch_paths} if scratch_paths else set())
    )
    for p in targets_paths:
        if p.is_file():
            try:
                p.unlink(missing_ok=True)
                summary["files_removed"] += 1
            except Exception as e:
                logger.warning("Failed to remove scratch file '%s': %s", p, e)
        elif p.is_dir():
            try:
                shutil.rmtree(p, ignore_errors=False)
                summary["directories_removed"] += 1
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("Failed to remove scratch directory '%s': %s", p, e)
        _ACTIVE_SCRATCH_PATHS.discard(p)

    return summary


# ============================================================================
# Provenance Canonicalization & SHA-256 Hash Validation
# ============================================================================


def canonicalize_provenance_json(
    content: dict[str, Any] | str | bytes | Path,
    exclude_keys: Sequence[str] = ("sha256", "hash", "provenance_hash", "_canonical_sha256"),
) -> bytes:
    """Canonicalize JSON provenance payload for deterministic SHA-256 hashing.

    Normalizes data structures by recursively sorting dictionary keys, stripping
    variable formatting whitespace, applying standard compact separators (',', ':'),
    and excluding self-referential hash fields. This guarantees that whitespace
    or key-ordering variations do not cause brittle failures while strictly
    catching semantic alterations to CODATA constants or physical parameters.

    Args:
        content: Dictionary, JSON string, bytes, or Path to JSON file.
        exclude_keys: Sequence of key names to omit from canonicalization.

    Returns:
        Canonical UTF-8 encoded bytes.

    Raises:
        ProvenanceIntegrityError: If content is unparseable or of unsupported type.
        FileNotFoundError: If a specified file path does not exist on disk.
    """
    if isinstance(content, Path):
        p = content.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Provenance file not found at '{p}'")
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ProvenanceIntegrityError(
                f"Failed to parse provenance JSON file '{p}': {exc}"
            ) from exc
    elif isinstance(content, (str, bytes)):
        if isinstance(content, str) and (content.endswith(".json") or os.path.isfile(content)):
            p = Path(content).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Provenance file not found at '{p}'")
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                raise ProvenanceIntegrityError(
                    f"Failed to parse provenance JSON file '{p}': {exc}"
                ) from exc
        else:
            try:
                data = json.loads(content)
            except Exception as exc:
                raise ProvenanceIntegrityError(
                    f"Failed to decode provenance JSON string: {exc}"
                ) from exc
    elif isinstance(content, dict):
        data = content
    else:
        raise ProvenanceIntegrityError(
            f"Unsupported content type for canonicalization: {type(content).__name__}"
        )

    exclude_set = set(exclude_keys)

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                str(k): _normalize(v)
                for k, v in sorted(obj.items())
                if k not in exclude_set
            }
        elif isinstance(obj, (list, tuple)):
            return [_normalize(item) for item in obj]
        elif isinstance(obj, float):
            return float(obj)
        return obj

    normalized = _normalize(data)
    canonical_str = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return canonical_str.encode("utf-8")


def compute_provenance_sha256(
    content: dict[str, Any] | str | bytes | Path,
    exclude_keys: Sequence[str] = ("sha256", "hash", "provenance_hash", "_canonical_sha256"),
) -> str:
    """Compute deterministic SHA-256 hex digest of canonicalized provenance data.

    Args:
        content: Dictionary, JSON string, bytes, or Path to JSON file.
        exclude_keys: Sequence of key names to omit from canonicalization.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest string.
    """
    canonical_bytes = canonicalize_provenance_json(content, exclude_keys=exclude_keys)
    return hashlib.sha256(canonical_bytes).hexdigest()


def compute_file_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 digest of a physical file iteratively without memory bloat.

    Args:
        file_path: Path to target file on disk.
        chunk_size: Byte chunk read size (default: 64 KB).

    Returns:
        Hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Target file does not exist: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_fit_provenance(
    provenance_path: Path | str | dict[str, Any] | None = None,
    expected_hash: str | None = None,
    enforce_dataset_hashes: bool = True,
    search_dirs: Sequence[Path | str] | None = None,
) -> FitProvenanceReport:
    """Validate SHA-256 cryptographic hash and integrity of fit_provenance.json.

    Canonicalizes the JSON content (ignoring whitespace and key order) before hashing,
    guaranteeing that no local modifications to CODATA constants or isotopic masses pass
    silently without brittle formatting failures.

    Validates against the internal self-referential hash field (e.g. 'sha256' or 'provenance_hash')
    and/or an externally supplied `expected_hash`. When `enforce_dataset_hashes` is True,
    verifies that any dataset files specified in 'dataset_hashes' exist on disk with
    identical cryptographic digests.

    Args:
        provenance_path: Explicit Path/str to fit_provenance.json, or loaded dict. If None,
                         searches `search_dirs` or default locations (artifacts, cwd, scratch).
        expected_hash: Optional externally expected SHA-256 hex digest.
        enforce_dataset_hashes: If True, asserts bitwise hash matches for referenced datasets.
        search_dirs: Optional search directories if provenance_path is None.

    Returns:
        `FitProvenanceReport` containing detailed validation metadata.

    Raises:
        FileNotFoundError: If provenance file is missing and was explicitly required.
        ProvenanceIntegrityError: If JSON syntax is invalid or dataset file verification fails.
        ProvenanceHashMismatchError: If canonical SHA-256 digest does not match expected digest.
    """
    target_file: Path
    data: dict[str, Any]

    if provenance_path is not None and not isinstance(provenance_path, dict):
        target_file = Path(provenance_path).expanduser().resolve()
        if not target_file.is_file():
            raise FileNotFoundError(
                f"fit_provenance.json not found at explicit path: '{target_file}'"
            )
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ProvenanceIntegrityError(
                f"Failed to read or parse fit_provenance.json at '{target_file}': {exc}"
            ) from exc
    elif isinstance(provenance_path, dict):
        data = provenance_path
        target_file = Path("in_memory_fit_provenance.json")
    else:
        # Search candidate paths
        candidate_dirs: list[Path] = []
        if search_dirs:
            candidate_dirs.extend(Path(d).expanduser().resolve() for d in search_dirs)
        candidate_dirs.append(get_artifact_directory(create_if_missing=False))
        candidate_dirs.append(Path.cwd().resolve())
        candidate_dirs.append(get_scratch_directory(create_if_missing=False))

        discovered_file: Path | None = None
        for c_dir in candidate_dirs:
            cand = c_dir / "fit_provenance.json"
            if cand.is_file():
                discovered_file = cand
                break

        if discovered_file is None:
            if expected_hash is not None:
                raise FileNotFoundError(
                    "expected_hash was provided, but fit_provenance.json was not found "
                    f"in search paths: {[str(d) for d in candidate_dirs]}"
                )
            return FitProvenanceReport(
                is_valid=False,
                provenance_path=Path("fit_provenance.json"),
                computed_sha256="",
                expected_sha256=expected_hash,
                dataset_hashes_verified=0,
                diagnostics="No fit_provenance.json file found in search directories.",
            )

        target_file = discovered_file
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ProvenanceIntegrityError(
                f"Failed to read or parse fit_provenance.json at '{target_file}': {exc}"
            ) from exc

    computed_hash = compute_provenance_sha256(data)

    internal_hash = data.get("sha256") or data.get("provenance_hash") or data.get("hash")
    target_expected: str | None = None
    if expected_hash is not None:
        target_expected = expected_hash.strip().lower()
    elif internal_hash is not None:
        target_expected = str(internal_hash).strip().lower()

    if target_expected is None:
        err = "[HARD_ABORT: PROVENANCE HASH MISMATCH] fit_provenance.json lacks internal SHA-256 and no external hash was provided!"
        logger.error(err)
        raise ProvenanceHashMismatchError(err)

    if computed_hash != target_expected:
        err = (
            f"[HARD_ABORT: PROVENANCE HASH MISMATCH] fit_provenance.json integrity check failed! "
            f"Computed canonical SHA-256 '{computed_hash}' does not match expected "
            f"'{target_expected}' at '{target_file}'."
        )
        logger.error(err)
        raise ProvenanceHashMismatchError(err)

    # Dataset hashes verification
    datasets_verified = 0
    if enforce_dataset_hashes and "dataset_hashes" in data and isinstance(data["dataset_hashes"], dict):
        base_search_dir = target_file.parent if target_file.is_file() else Path.cwd()
        for dset_name, exp_dset_hash in data["dataset_hashes"].items():
            clean_exp_hash = str(exp_dset_hash).strip().lower()
            dset_cand = base_search_dir / dset_name
            if not dset_cand.is_file():
                art_cand = get_artifact_directory(create_if_missing=False) / dset_name
                if art_cand.is_file():
                    dset_cand = art_cand

            if dset_cand.is_file():
                actual_dset_hash = compute_file_sha256(dset_cand)
                if actual_dset_hash != clean_exp_hash:
                    err = (
                        f"[HARD_ABORT: DATASET INTEGRITY VIOLATION] Referenced dataset file '{dset_name}' "
                        f"hash mismatch! Actual SHA-256 '{actual_dset_hash}' does not match "
                        f"provenance digest '{clean_exp_hash}'."
                    )
                    logger.error(err)
                    raise ProvenanceIntegrityError(err)
                datasets_verified += 1
            else:
                raise ProvenanceIntegrityError(
                    "[HARD_ABORT: DATASET INTEGRITY VIOLATION] "
                    f"Referenced dataset file '{dset_name}' is unavailable; "
                    "its hash cannot be verified."
                )

    logger.info(
        "fit_provenance.json verified successfully: path='%s', sha256='%s', datasets_verified=%d",
        target_file,
        computed_hash,
        datasets_verified,
    )

    return FitProvenanceReport(
        is_valid=True,
        provenance_path=target_file,
        computed_sha256=computed_hash,
        expected_sha256=target_expected,
        dataset_hashes_verified=datasets_verified,
        diagnostics=None,
    )


# ============================================================================
# Environment Bootstrap Routine
# ============================================================================


def bootstrap_environment(
    artifacts_env: str = "COCHEM_ARTIFACTS",
    scratch_env: str = "COCHEM_SCRATCH",
    fallback_artifacts: Path | str | None = None,
    fallback_scratch: Path | str | None = None,
    enforce_airgap: bool = True,
    validate_provenance: bool = True,
    fit_provenance_path: Path | str | None = None,
    expected_provenance_hash: str | None = None,
) -> BootstrapperConfig:
    """Bootstrap runtime environment for CoChem-TORQ execution.

    Resolves artifact and scratch directories dynamically, performs mathematical
    air-gap validation against the current working repository space, ensures
    directory creation, optionally validates SHA-256 provenance hashes for
    `fit_provenance.json`, and registers the atexit IPC cleanup collector.

    Args:
        artifacts_env: Environment variable for persistent artifacts.
        scratch_env: Environment variable for ephemeral scratch.
        fallback_artifacts: Optional fallback path for artifacts directory.
        fallback_scratch: Optional fallback path for scratch directory.
        enforce_airgap: If True, asserts strict mathematical isolation between
            CWD and runtime directories, raising `AirGapViolationError` on intersection.
        validate_provenance: If True, enforces SHA-256 canonical hash validation on
            `fit_provenance.json` before computation begins.
        fit_provenance_path: Optional explicit Path/str to `fit_provenance.json`.
        expected_provenance_hash: Optional expected SHA-256 hex digest.

    Returns:
        Validated `BootstrapperConfig` instance.

    Raises:
        AirGapViolationError: If air-gap validation fails and enforce_airgap is True.
        ProvenanceHashMismatchError: If provenance SHA-256 hash fails validation.
        ProvenanceIntegrityError: If provenance payload or dataset binding is corrupted.
    """
    artifacts_path = get_artifact_directory(
        env_var=artifacts_env,
        fallback_dir=fallback_artifacts,
        create_if_missing=True,
    )
    scratch_path = get_scratch_directory(
        env_var=scratch_env,
        fallback_dir=fallback_scratch,
        create_if_missing=True,
    )

    if enforce_airgap:
        # Tripartite air-gap validation: CWD vs Artifacts, CWD vs Scratch, Artifacts vs Scratch
        verify_airgap(cwd=Path.cwd(), artifacts_dir=artifacts_path)
        verify_airgap(cwd=Path.cwd(), artifacts_dir=scratch_path)
        verify_airgap(cwd=artifacts_path, artifacts_dir=scratch_path)

    prov_report: FitProvenanceReport | None = None
    if validate_provenance:
        prov_target: Path | None = None
        if fit_provenance_path is not None:
            prov_target = Path(fit_provenance_path).expanduser().resolve()

        search_dirs: list[Path] = []
        if fallback_artifacts is not None:
            search_dirs.append(Path(fallback_artifacts).expanduser().resolve())

        prov_report = validate_fit_provenance(
            provenance_path=prov_target,
            expected_hash=expected_provenance_hash,
            enforce_dataset_hashes=True,
            search_dirs=search_dirs if search_dirs else None,
        )

        if not prov_report.is_valid:
            err = f"[HARD_ABORT: PROVENANCE VALIDATION FAILED] {prov_report.diagnostics}"
            logger.error(err)
            raise ProvenanceIntegrityError(err)

    _ensure_atexit_registered()
    register_ipc_cleanup(scratch_paths=[scratch_path])

    config = BootstrapperConfig(
        artifacts_dir=artifacts_path,
        scratch_dir=scratch_path,
        env_var=artifacts_env,
        scratch_env_var=scratch_env,
        enforce_airgap=enforce_airgap,
        clean_on_exit=True,
        validate_provenance=validate_provenance,
        fit_provenance_path=prov_report.provenance_path if (prov_report and prov_report.is_valid) else None,
        provenance_hash=prov_report.computed_sha256 if (prov_report and prov_report.is_valid) else None,
        provenance_report=prov_report,
    )

    logger.info(
        "CoChem-TORQ Bootstrapped: artifacts='%s', scratch='%s', airgap_enforced=%s, provenance_validated=%s",
        artifacts_path,
        scratch_path,
        enforce_airgap,
        bool(prov_report and prov_report.is_valid),
    )
    return config
