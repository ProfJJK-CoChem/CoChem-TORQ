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
import logging
import mmap
import multiprocessing.shared_memory as sm  # zero-stub IPC shared memory
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


# ============================================================================
# Exceptions
# ============================================================================


class AirGapViolationError(RuntimeError):
    """Raised when working directory and artifact directory overlap or intersect."""


class IPCBufferError(RuntimeError):
    """Raised when memory-mapped buffer creation, access, or teardown fails."""


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
        error_msg = (
            f"AirGapViolationError: Air-gap integrity violated! {report.reason}. "
            "CoChem-TORQ forbids writing runtime artifacts inside or "
            "overlapping the working repository space."
        )
        logger.error(error_msg)
        raise AirGapViolationError(error_msg)

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
# Environment Bootstrap Routine
# ============================================================================


def bootstrap_environment(
    artifacts_env: str = "COCHEM_ARTIFACTS",
    scratch_env: str = "COCHEM_SCRATCH",
    fallback_artifacts: Path | str | None = None,
    fallback_scratch: Path | str | None = None,
    enforce_airgap: bool = True,
) -> BootstrapperConfig:
    """Bootstrap runtime environment for CoChem-TORQ execution.

    Resolves artifact and scratch directories dynamically, performs mathematical
    air-gap validation against the current working repository space, ensures
    directory creation, and registers the atexit IPC cleanup collector.

    Args:
        artifacts_env: Environment variable for persistent artifacts.
        scratch_env: Environment variable for ephemeral scratch.
        fallback_artifacts: Optional fallback path for artifacts directory.
        fallback_scratch: Optional fallback path for scratch directory.
        enforce_airgap: If True, asserts strict mathematical isolation between
            CWD and runtime directories, raising `AirGapViolationError` on intersection.

    Returns:
        Validated `BootstrapperConfig` instance.

    Raises:
        AirGapViolationError: If air-gap validation fails and enforce_airgap is True.
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

    _ensure_atexit_registered()
    register_ipc_cleanup(scratch_paths=[scratch_path])

    config = BootstrapperConfig(
        artifacts_dir=artifacts_path,
        scratch_dir=scratch_path,
        env_var=artifacts_env,
        scratch_env_var=scratch_env,
        enforce_airgap=enforce_airgap,
        clean_on_exit=True,
    )

    logger.info(
        "CoChem-TORQ Bootstrapped: artifacts='%s', scratch='%s', airgap_enforced=%s",
        artifacts_path,
        scratch_path,
        enforce_airgap,
    )
    return config
