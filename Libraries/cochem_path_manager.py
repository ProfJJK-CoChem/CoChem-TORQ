"""CoChem-TORQ: Dynamic Path Resolution & Air-Gap Verification Contracts.

=============================================================================
Phase 1 (Stage 0.0) / Task 1 & Task 5 Core Foundation
-----------------------------------------------------
Authoritative master library providing the central 6-Tier Execution Context,
dynamic path resolution hierarchy, Tripartite Filesystem Air-Gap security ring
enforcement (verify_air_gap_boundary), cross-platform POSIX/Darwin/Windows
shared memory management (CoChemSharedMemoryBuffer), out-of-core scratch buffer
allocation, physical buffer synchronization, ghost output purging, and deterministic
atexit/signal IPC resource reclamation.

Architectural Philosophy & Tripartite Air-Gap Model:
1. Domain A (Ring 1 Static Execution Tier):
   - Immutable Git repository root ($HOME/CoChem-TORQ / $COCHEM_REPO_DIR).
   - Read-only source code, tests, and configuration.
   - Any runtime write or mutation attempt raises AirGapViolationError.
2. Domain B (Ring 3 Dynamic Artifact Tier):
   - Curated persistent scientific outputs, HDF5 vault, Parquet catalogs, SI exports.
   - Resolved via 6-tier hierarchy ($COCHEM_DELIVERABLES, $COCHEM_ARTIFACTS, etc.).
   - Mathematically disjoint from Domain A.
3. Domain C (Ring 2 Volatile Compute Tier):
   - High-throughput ephemeral scratch, RAM disks (/dev/shm, /tmp, %TEMP%),
     inter-process communication (IPC) buffers, and memory-mapped files.
   - Cleaned deterministically on process termination or garbage collection.

Method Matrix v4 & Physical Invariant Compliance:
- Zero-Mock Anti-Spoofing Mandate: All operations perform authentic physical disk I/O,
  real OS kernel shared memory allocation, genuine memory mapping, and cryptographic
  checksum hashing (xxhash.xxh64 / SHA-256).
- Mendeleev Mandate: Dynamic atomic and isotopic masses dynamically retrieved via
  the `mendeleev` library with zero hardcoded atomic weights.
- Cross-Platform Hardware Saturation: Fully compatible with Windows (NTFS), macOS (Darwin),
  Linux (POSIX), GitHub Actions CI runners, Codespaces, and HPC supercomputer nodes (SLURM/PBS).

Authoritative References:
- SRS Task 1: Dynamic Path Resolution & Air-Gap Verification Contracts
- SRS Task 5: Path Manager & Cross-Platform Shared Memory Backplane
- Method Matrix v4: Heterogeneous Orchestration, Concurrency & Air-Gap Directives
"""

from __future__ import annotations

import atexit
import ctypes
import enum
import functools
import gc
import hashlib
import io
import logging
import mmap
import multiprocessing.shared_memory as sm
import os
import platform
import re
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import (
    Any,
    Final,
    Literal,
)

import numpy as np
from mendeleev import element as mendeleev_element
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("CoChem-TORQ.PathManager")

try:
    import xxhash
    _XXHASH_AVAILABLE = True
except ImportError:
    _XXHASH_AVAILABLE = False


# ============================================================================
# 1. Custom Exceptions
# ============================================================================


class AirGapViolationError(PermissionError):
    """Raised when an operation breaches Domain A static repository space or intersects air-gap boundaries."""


class IPCBufferError(RuntimeError):
    """Base exception for all IPC and shared memory buffer operations."""


class SharedMemoryAllocationError(IPCBufferError):
    """Raised when shared memory segment creation or mapping fails."""


class BufferCorruptedError(IPCBufferError):
    """Raised when memory segment content is corrupted or unparseable."""


class ChecksumMismatchError(IPCBufferError):
    """Raised when buffer checksum verification fails due to corruption or mismatch."""


class OrphanedResourceError(IPCBufferError):
    """Raised when an orphaned IPC resource is detected or fails cleanup."""


class CoChemIntegrityError(Exception):
    """Raised when physical file integrity, buffer lock sync, or proof-of-work validation fails."""

    def __init__(self, message: str, path: Path | str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.path = Path(path) if path is not None else None


class MethodMatrixViolationError(ValueError):
    """Raised when a Method Matrix v4 invariant or restriction is breached."""


# ============================================================================
# 2. Enumerations & Constants
# ============================================================================


class CoChemAirGapRing(enum.IntEnum):
    """Tripartite Filesystem Domain Rings."""

    RING_1_DOMAIN_A_STATIC = 1  # Immutable Git Repository & Static Source Code
    RING_2_DOMAIN_C_EPHEMERAL = 2  # Volatile RAM disk, SHM & Temporary Scratch
    RING_3_DOMAIN_B_PERSISTENT = 3  # Dynamic HDF5 Vault, Curated Artifacts & Deliverables

    # String & integer aliases for cross-module compatibility
    RING1_STATIC_REPO = 1
    RING2_SCRATCH = 2
    RING3_ARTIFACTS = 3


class EnvironmentTier(str, enum.Enum):
    """6-Tier Environment Matrix defining host execution environments."""

    LOCAL_WINDOWS = "LOCAL_WINDOWS"
    LOCAL_MACOS = "LOCAL_MACOS"
    LOCAL_LINUX = "LOCAL_LINUX"
    CODESPACES = "CODESPACES"
    GITHUB_ACTIONS = "GITHUB_ACTIONS"
    HPC_NODES = "HPC_NODES"


class IPCSegmentType(str, enum.Enum):
    """Backing storage type for IPC buffers."""

    SHARED_MEMORY = "shm"
    MEMORY_MAPPED_FILE = "mmap"


class IPCPayloadType(str, enum.Enum):
    """Data payload representation type serialized into IPC buffers."""

    TABLE = "table"
    RECORD_BATCH = "record_batch"
    TENSOR = "tensor"
    NUMPY_ARRAY = "numpy_array"
    RAW_BYTES = "raw_bytes"


# ============================================================================
# 3. Pydantic v2 Data Models
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
    target_resolved: Path = Field(
        ..., description="Canonical resolved target path tested"
    )
    artifacts_resolved: Path | None = Field(
        default=None, description="Canonical resolved runtime artifacts directory"
    )
    verified_at: float = Field(
        default_factory=time.time, description="Unix timestamp of verification check"
    )
    reason: str | None = Field(
        default=None, description="Diagnostic explanation of air-gap violation if any"
    )


class TierCapabilities(BaseModel):
    """Hardware and runtime capabilities for an execution tier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: EnvironmentTier = Field(
        ..., description="Canonical 6-tier environment identifier"
    )
    is_ci: bool = Field(
        default=False, description="True if running in an automated CI/CD pipeline"
    )
    is_hpc: bool = Field(
        default=False, description="True if executing on a clustered batch scheduler"
    )
    is_container: bool = Field(
        default=False, description="True if executing within a container or WSL layer"
    )
    has_gpu_acceleration: bool = Field(
        default=False, description="True if GPU compute acceleration is supported"
    )
    supports_openmpi: bool = Field(
        default=False, description="True if OpenMPI multi-node parallelization is enabled"
    )
    max_worker_processes: int = Field(
        default=4, description="Recommended concurrent worker process limit"
    )
    default_scratch_mount: str = Field(
        ..., description="Default path string for volatile compute scratch buffers"
    )
    description: str = Field(
        ..., description="Detailed description of the environment tier"
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
        ..., description="Type of IPC buffer ('mmap' or 'shm')"
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
    checksum_xxh64: str = Field(
        default="", description="xxhash.xxh64 checksum hex string"
    )
    checksum_sha256: str = Field(
        default="", description="SHA-256 fallback checksum hex string"
    )
    extra_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary extra metadata"
    )


# ============================================================================
# 4. Checksum & Cryptographic Integrity Helpers
# ============================================================================


def compute_buffer_checksum(
    data: bytes | bytearray | memoryview | mmap.mmap | np.ndarray,
) -> tuple[str, str]:
    """Computes xxhash.xxh64 and SHA-256 cryptographic checksums for memory buffers.

    Args:
        data: Buffer or byte sequence to hash.

    Returns:
        Tuple of (xxh64_hex_digest, sha256_hex_digest).
    """
    raw_view: memoryview
    if isinstance(data, (bytes, bytearray)):
        raw_view = memoryview(data)
    elif isinstance(data, memoryview):
        raw_view = data.cast("B") if data.format != "B" else data
    elif isinstance(data, mmap.mmap):
        raw_view = memoryview(data)
    elif isinstance(data, np.ndarray):
        raw_view = memoryview(np.ascontiguousarray(data)).cast("B")
    else:
        raw_view = memoryview(bytes(data))

    sha256_hash = hashlib.sha256(raw_view).hexdigest()

    if _XXHASH_AVAILABLE:
        xxh64_hash = xxhash.xxh64(raw_view).hexdigest()
    else:
        xxh64_hash = sha256_hash[:16]

    return xxh64_hash, sha256_hash


# ============================================================================
# 5. Global IPC Registry & Garbage Collection
# ============================================================================

_ACTIVE_BUFFERS_LOCK = threading.RLock()
_ACTIVE_BUFFERS: set[CoChemSharedMemoryBuffer] = set()
_ACTIVE_SHM_NAMES: set[str] = set()
_ACTIVE_SCRATCH_PATHS: set[Path] = set()
_ATEXIT_REGISTERED: bool = False


def register_shm_buffer(buffer: CoChemSharedMemoryBuffer) -> None:
    """Registers an active buffer in the global tracking registry."""
    with _ACTIVE_BUFFERS_LOCK:
        _ACTIVE_BUFFERS.add(buffer)
        if buffer.name:
            _ACTIVE_SHM_NAMES.add(buffer.name)
        if buffer.path:
            _ACTIVE_SCRATCH_PATHS.add(buffer.path)
    _ensure_atexit_registered()


def unregister_shm_buffer(buffer: CoChemSharedMemoryBuffer) -> None:
    """Unregisters a buffer from the global tracking registry."""
    with _ACTIVE_BUFFERS_LOCK:
        _ACTIVE_BUFFERS.discard(buffer)
        if buffer.name:
            _ACTIVE_SHM_NAMES.discard(buffer.name)
        if buffer.path:
            _ACTIVE_SCRATCH_PATHS.discard(buffer.path)


def cleanup_ipc_scratch(force_unlink: bool = True) -> int:
    """Cleans up all tracked active shared memory segments and scratch files.

    Args:
        force_unlink: If True, unlinks/deletes all active segments and temp files.

    Returns:
        Number of cleaned-up resources.
    """
    cleaned_count = 0
    with _ACTIVE_BUFFERS_LOCK:
        gc.collect()
        buffers_to_clean = list(_ACTIVE_BUFFERS)
        _ACTIVE_BUFFERS.clear()

        for buf in buffers_to_clean:
            try:
                if not buf.is_closed:
                    buf.close()
                if force_unlink:
                    buf.unlink()
                cleaned_count += 1
            except Exception as exc:
                logger.warning(f"Error during buffer cleanup ({buf.name}): {exc}")

        for shm_name in list(_ACTIVE_SHM_NAMES):
            try:
                shm_temp = sm.SharedMemory(name=shm_name, create=False)
                if hasattr(shm_temp, "buf") and shm_temp.buf is not None:
                    try:
                        shm_temp.buf.release()
                    except Exception:
                        pass
                shm_temp.close()
                shm_temp.unlink()
                cleaned_count += 1
            except Exception:
                pass
        _ACTIVE_SHM_NAMES.clear()

        for scratch_file in list(_ACTIVE_SCRATCH_PATHS):
            try:
                if scratch_file.exists():
                    scratch_file.unlink(missing_ok=True)
                    cleaned_count += 1
            except Exception as exc:
                logger.warning(f"Error deleting scratch file {scratch_file}: {exc}")
        _ACTIVE_SCRATCH_PATHS.clear()

    return cleaned_count


def _atexit_cleanup_handler() -> None:
    """Invoked automatically by atexit on normal or unhandled termination."""
    cleanup_ipc_scratch(force_unlink=True)


def _signal_cleanup_handler(signum: int, frame: Any) -> None:
    """Signal trap for SIGINT and SIGTERM to guarantee zero orphaned IPC buffers."""
    cleanup_ipc_scratch(force_unlink=True)
    sys.exit(128 + signum)


def _ensure_atexit_registered() -> None:
    """Ensures atexit and signal handlers are registered exactly once."""
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_cleanup_handler)
        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if sig is not None:
                try:
                    signal.signal(sig, _signal_cleanup_handler)
                except (ValueError, OSError):
                    pass
        _ATEXIT_REGISTERED = True


# ============================================================================
# 6. Cross-Platform Shared Memory Buffer (CoChemSharedMemoryBuffer)
# ============================================================================


class CoChemSharedMemoryBuffer:
    """Cross-platform POSIX, Darwin, and Windows shared memory buffer manager.

    Provides zero-copy shared memory access via `multiprocessing.shared_memory.SharedMemory`
    with automatic transparent fallback to memory-mapped files (`mmap.mmap`) if POSIX
    `/dev/shm` or Darwin shm limits are reached or unavailable.
    """

    def __init__(
        self,
        name: str | None = None,
        size: int = 4096,
        create: bool = True,
        prefer_shm: bool = True,
        backing_dir: Path | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError(f"Buffer size must be > 0 bytes, got {size}.")

        self._name = name or f"cochem_shm_{uuid.uuid4().hex[:12]}"
        self._size = size
        self._create = create
        self._prefer_shm = prefer_shm
        self._backing_dir = backing_dir
        self._closed = False
        self._is_shm = False
        self._payload_size = 0
        self._shm: sm.SharedMemory | None = None
        self._mmap: mmap.mmap | None = None
        self._file_obj: io.BufferedRandom | None = None
        self._path: Path | None = None

        self._allocate_buffer()
        register_shm_buffer(self)

    def _allocate_buffer(self) -> None:
        """Attempts SharedMemory allocation, falling back to mmap if shm fails."""
        if self._prefer_shm:
            try:
                if self._create:
                    self._shm = sm.SharedMemory(name=self._name, create=True, size=self._size)
                else:
                    self._shm = sm.SharedMemory(name=self._name, create=False)
                    self._size = self._shm.size
                self._is_shm = True
                self.metadata = IPCBufferMetadata(
                    name=self._name,
                    buffer_type="shm",
                    size_bytes=self._size,
                    file_path=None,
                )
                logger.debug(f"Allocated SharedMemory segment '{self._name}' ({self._size} bytes).")
                return
            except (FileExistsError, sm.SharedMemoryError, OSError, PermissionError) as exc:
                logger.info(
                    f"SharedMemory allocation failed for '{self._name}': {exc}. "
                    "Falling back transparently to memory-mapped file (mmap)."
                )

        self._is_shm = False
        target_dir = self._backing_dir or Path(tempfile.gettempdir()) / "cochem_scratch"
        target_dir.mkdir(parents=True, exist_ok=True)
        self._path = target_dir / f"{self._name}.mmap"

        try:
            if self._create:
                self._file_obj = open(self._path, "wb+")
                self._file_obj.seek(self._size - 1)
                self._file_obj.write(b"\x00")
                self._file_obj.flush()
                self._mmap = mmap.mmap(self._file_obj.fileno(), self._size, access=mmap.ACCESS_WRITE)
            else:
                if not self._path.exists():
                    raise SharedMemoryAllocationError(f"Backing mmap file does not exist: {self._path}")
                self._size = self._path.stat().st_size
                self._file_obj = open(self._path, "r+b")
                self._mmap = mmap.mmap(self._file_obj.fileno(), self._size, access=mmap.ACCESS_WRITE)

            self.metadata = IPCBufferMetadata(
                name=self._name,
                buffer_type="mmap",
                size_bytes=self._size,
                file_path=self._path,
            )
            logger.debug(f"Allocated mmap buffer '{self._name}' at '{self._path}' ({self._size} bytes).")
        except Exception as mmap_exc:
            raise SharedMemoryAllocationError(
                f"Failed to allocate fallback mmap buffer '{self._name}': {mmap_exc}"
            ) from mmap_exc

    @property
    def name(self) -> str:
        """Name identifier of the buffer."""
        return self._name

    @property
    def size(self) -> int:
        """Capacity of the buffer in bytes."""
        return self._size

    @property
    def is_closed(self) -> bool:
        """True if the buffer has been closed."""
        return self._closed

    @property
    def is_shm(self) -> bool:
        """True if backed by kernel SharedMemory, False if backed by mmap."""
        return self._is_shm

    @property
    def path(self) -> Path | None:
        """Path to physical backing file for mmap buffers, or None for shm."""
        return self._path

    @property
    def buf(self) -> memoryview:
        """Access direct memoryview of the underlying buffer."""
        if self._closed:
            raise IPCBufferError(f"Cannot access closed buffer '{self._name}'.")
        if self._is_shm and self._shm is not None:
            if self._shm.buf is None:
                raise IPCBufferError(f"SharedMemory '{self._name}' has no active buffer.")
            return self._shm.buf
        if self._mmap is not None:
            return memoryview(self._mmap)
        raise IPCBufferError(f"Buffer '{self._name}' has no valid memory allocation.")

    @property
    def buffer(self) -> memoryview:
        """Alias for buf property."""
        return self.buf

    def write(
        self,
        data: bytes | bytearray | memoryview | np.ndarray,
        offset: int = 0,
    ) -> int:
        """Writes binary payload into the shared buffer starting at offset.

        Args:
            data: Data payload to write.
            offset: Byte offset within the buffer.

        Returns:
            Number of bytes written.

        Raises:
            IPCBufferError: If buffer is closed or payload exceeds buffer capacity.
        """
        if self._closed:
            raise IPCBufferError(f"Cannot write to closed buffer '{self._name}'.")

        raw_bytes: memoryview
        if isinstance(data, (bytes, bytearray)):
            raw_bytes = memoryview(data)
        elif isinstance(data, memoryview):
            raw_bytes = data.cast("B") if data.format != "B" else data
        elif isinstance(data, np.ndarray):
            raw_bytes = memoryview(np.ascontiguousarray(data)).cast("B")
        else:
            raw_bytes = memoryview(bytes(data))

        data_len = len(raw_bytes)
        if offset + data_len > self._size:
            raise IPCBufferError(
                f"Payload of {data_len} bytes at offset {offset} exceeds buffer capacity of {self._size} bytes."
            )

        target_view = self.buf
        target_view[offset : offset + data_len] = raw_bytes
        self.flush()

        self._payload_size = max(self._payload_size, offset + data_len)
        self.metadata.extra_metadata["payload_size"] = self._payload_size

        xxh, sha = compute_buffer_checksum(target_view[offset : offset + data_len])
        self.metadata.checksum_xxh64 = xxh
        self.metadata.checksum_sha256 = sha

        return data_len

    def read(self, size: int | None = None, offset: int = 0) -> bytes:
        """Reads byte contents from the buffer.

        Args:
            size: Number of bytes to read (default: full payload from offset).
            offset: Starting byte offset.

        Returns:
            Bytes read from buffer.
        """
        if self._closed:
            raise IPCBufferError(f"Cannot read from closed buffer '{self._name}'.")

        target_view = self.buf
        effective_capacity = self._payload_size if (self._payload_size > 0 and size is None) else self._size
        total_avail = effective_capacity - offset
        read_len = total_avail if size is None else min(size, self._size - offset)

        if read_len < 0:
            return b""

        return bytes(target_view[offset : offset + read_len])

    def get_memoryview(self, offset: int = 0, size: int | None = None) -> memoryview:
        """Returns a non-copying slice memoryview of the buffer."""
        if self._closed:
            raise IPCBufferError(f"Cannot slice closed buffer '{self._name}'.")
        target_view = self.buf
        max_len = self._size - offset
        slice_len = max_len if size is None else min(size, max_len)
        return target_view[offset : offset + slice_len]

    def flush(self) -> None:
        """Flushes memory buffers to kernel or disk."""
        if not self._closed:
            if self._mmap is not None:
                try:
                    self._mmap.flush()
                except OSError:
                    pass
            if self._file_obj is not None:
                try:
                    self._file_obj.flush()
                except OSError:
                    pass

    def compute_checksum(self) -> str:
        """Computes current cryptographic checksum of the buffer payload."""
        data_to_hash = self.buf[0 : self._payload_size] if self._payload_size > 0 else self.buf
        xxh, _ = compute_buffer_checksum(data_to_hash)
        return xxh

    def verify_checksum(self, expected_checksum: str | None = None) -> bool:
        """Verifies buffer integrity against expected checksum."""
        target_checksum = expected_checksum or self.metadata.checksum_xxh64
        if not target_checksum:
            return True
        current_checksum = self.compute_checksum()
        if current_checksum != target_checksum:
            raise ChecksumMismatchError(
                f"Checksum mismatch on buffer '{self._name}': expected '{target_checksum}', got '{current_checksum}'."
            )
        return True

    def close(self) -> None:
        """Closes memory-map and file handles without unlinking."""
        if self._closed:
            return
        self._closed = True

        if self._shm is not None:
            shm_obj = self._shm
            self._shm = None
            if hasattr(shm_obj, "buf") and shm_obj.buf is not None:
                try:
                    shm_obj.buf.release()
                except Exception:
                    pass
            try:
                shm_obj.close()
            except Exception:
                pass

        if self._mmap is not None:
            mmap_obj = self._mmap
            self._mmap = None
            try:
                mmap_obj.close()
            except Exception:
                pass

        if self._file_obj is not None:
            file_obj = self._file_obj
            self._file_obj = None
            try:
                file_obj.close()
            except Exception:
                pass

        gc.collect()

    def unlink(self) -> None:
        """Unlinks shared memory segment or deletes physical backing temporary file."""
        unregister_shm_buffer(self)

        if self._is_shm:
            if self._shm is not None:
                try:
                    if hasattr(self._shm, "buf") and self._shm.buf is not None:
                        try:
                            self._shm.buf.release()
                        except Exception:
                            pass
                    self._shm.unlink()
                except Exception:
                    pass
            else:
                try:
                    temp_shm = sm.SharedMemory(name=self._name, create=False)
                    if hasattr(temp_shm, "buf") and temp_shm.buf is not None:
                        try:
                            temp_shm.buf.release()
                        except Exception:
                            pass
                    temp_shm.unlink()
                    temp_shm.close()
                except Exception:
                    pass

        self.close()
        gc.collect()

        if self._path is not None and self._path.exists():
            try:
                remove_readonly_seal(self._path)
                self._path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(f"Failed to unlink mmap backing file '{self._path}': {exc}")

    def __enter__(self) -> CoChemSharedMemoryBuffer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ============================================================================
# 7. 6-Tier CoChemPathManager Master Implementation
# ============================================================================


class CoChemPathManager:
    """Master dynamic path resolution and Tripartite Air-Gap enforcement engine.

    Enforces the strict 6-Tier Scratch Resolution Hierarchy and Deliverables Resolution Hierarchy:
    - Tier 1: Explicit custom_path argument passed to method/constructor.
    - Tier 2: COCHEM_SCRATCH or COCHEM_SCRATCH_DIR environment variables.
    - Tier 3: COCHEM_TMP, TMPDIR, TEMP, or TMP environment variables.
    - Tier 4: XDG_CACHE_HOME / cochem / scratch (or ~/.cache/cochem/scratch, macOS Caches, Windows LocalAppData).
    - Tier 5: tempfile.gettempdir() / cochem_scratch.
    - Tier 6: Path.home() / .cochem / scratch fallback.
    """

    ENV_REPO_DIR: Final[str] = "COCHEM_REPO_DIR"
    ENV_SCRATCH_DIR: Final[str] = "COCHEM_SCRATCH_DIR"
    ENV_SCRATCH: Final[str] = "COCHEM_SCRATCH"
    ENV_ARTIFACTS_DIR: Final[str] = "COCHEM_ARTIFACTS_DIR"
    ENV_ARTIFACTS: Final[str] = "COCHEM_ARTIFACTS"
    ENV_ARTIFACTS_FALLBACK: Final[str] = "COCH_ARTIFACTS"
    ENV_DELIVERABLES_DIR: Final[str] = "COCHEM_DELIVERABLES_DIR"
    ENV_DELIVERABLES: Final[str] = "COCHEM_DELIVERABLES"

    def __init__(
        self,
        base_dir: str | Path | None = None,
        scratch_dir: str | Path | None = None,
        deliverables_dir: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        self._custom_scratch = Path(scratch_dir).resolve() if scratch_dir is not None else None
        target_deliv = deliverables_dir or artifacts_dir
        self._custom_deliverables = Path(target_deliv).resolve() if target_deliv is not None else None

    @classmethod
    def detect_tier(cls) -> EnvironmentTier:
        """Autonomously detects the active environment tier from OS telemetry and environment variables."""
        if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("RUNNER_TEMP"):
            return EnvironmentTier.GITHUB_ACTIONS

        if os.environ.get("CODESPACES") == "true" or os.environ.get("CODESPACE_NAME"):
            return EnvironmentTier.CODESPACES

        if any(
            os.environ.get(var)
            for var in ("SLURM_TMPDIR", "SLURM_JOB_ID", "PFSDIR", "PBS_O_WORKDIR")
        ):
            return EnvironmentTier.HPC_NODES

        sys_name = platform.system()
        if sys_name == "Windows" or os.environ.get("WSL_DISTRO_NAME"):
            return EnvironmentTier.LOCAL_WINDOWS
        elif sys_name == "Darwin":
            return EnvironmentTier.LOCAL_MACOS
        else:
            return EnvironmentTier.LOCAL_LINUX

    @classmethod
    def get_tier_capabilities(cls, tier: EnvironmentTier | None = None) -> TierCapabilities:
        """Retrieves runtime capability descriptors for the specified or detected environment tier."""
        active_tier = tier or cls.detect_tier()

        capabilities_map: dict[EnvironmentTier, TierCapabilities] = {
            EnvironmentTier.LOCAL_WINDOWS: TierCapabilities(
                tier=EnvironmentTier.LOCAL_WINDOWS,
                is_ci=False,
                is_hpc=False,
                is_container=bool(os.environ.get("WSL_DISTRO_NAME")),
                has_gpu_acceleration=True,
                supports_openmpi=bool(os.environ.get("COCHEM_FORCE_MPI") == "1"),
                max_worker_processes=os.cpu_count() or 8,
                default_scratch_mount=str(Path(tempfile.gettempdir()) / "cochem_scratch"),
                description="Local Windows workstation or WSL subsystem developer environment",
            ),
            EnvironmentTier.LOCAL_MACOS: TierCapabilities(
                tier=EnvironmentTier.LOCAL_MACOS,
                is_ci=False,
                is_hpc=False,
                is_container=False,
                has_gpu_acceleration=True,
                supports_openmpi=bool(shutil.which("mpirun")),
                max_worker_processes=os.cpu_count() or 8,
                default_scratch_mount=str(Path.home() / "Library" / "Caches" / "CoChem" / "scratch"),
                description="Local Apple Silicon or Intel macOS environment with Metal acceleration",
            ),
            EnvironmentTier.LOCAL_LINUX: TierCapabilities(
                tier=EnvironmentTier.LOCAL_LINUX,
                is_ci=False,
                is_hpc=False,
                is_container=False,
                has_gpu_acceleration=True,
                supports_openmpi=bool(shutil.which("mpirun") or shutil.which("orterun")),
                max_worker_processes=os.cpu_count() or 16,
                default_scratch_mount="/tmp/cochem_scratch",
                description="Native Linux (Debian/Ubuntu/RHEL) workstation compute node",
            ),
            EnvironmentTier.CODESPACES: TierCapabilities(
                tier=EnvironmentTier.CODESPACES,
                is_ci=False,
                is_hpc=False,
                is_container=True,
                has_gpu_acceleration=False,
                supports_openmpi=bool(shutil.which("mpirun")),
                max_worker_processes=4,
                default_scratch_mount=str(Path.home() / ".cochem" / "scratch"),
                description="Cloud-hosted GitHub Codespace containerized developer environment",
            ),
            EnvironmentTier.GITHUB_ACTIONS: TierCapabilities(
                tier=EnvironmentTier.GITHUB_ACTIONS,
                is_ci=True,
                is_hpc=False,
                is_container=True,
                has_gpu_acceleration=False,
                supports_openmpi=False,
                max_worker_processes=2,
                default_scratch_mount=os.environ.get("RUNNER_TEMP", str(Path(tempfile.gettempdir()) / "cochem_scratch")),
                description="Automated GitHub Actions CI/CD regression test runner",
            ),
            EnvironmentTier.HPC_NODES: TierCapabilities(
                tier=EnvironmentTier.HPC_NODES,
                is_ci=False,
                is_hpc=True,
                is_container=False,
                has_gpu_acceleration=True,
                supports_openmpi=True,
                max_worker_processes=32,
                default_scratch_mount=os.environ.get(
                    "SLURM_TMPDIR", os.environ.get("PFSDIR", str(Path(tempfile.gettempdir()) / "cochem_scratch"))
                ),
                description="Clustered High-Performance Computing SLURM/PBS supercomputing node",
            ),
        }
        return capabilities_map[active_tier]

    @classmethod
    def get_repo_root(cls) -> Path:
        """Domain A / Ring 1: Static immutable repository root."""
        env_val = os.environ.get(cls.ENV_REPO_DIR)
        if env_val and env_val.strip():
            repo_path = Path(env_val.strip()).resolve()
            if repo_path.is_dir():
                return repo_path

        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                return parent.resolve()

        return Path.cwd().resolve()

    @classmethod
    def _is_inside_repo(cls, target: Path) -> bool:
        """Check whether target is located inside the Domain A static repository root."""
        resolved = target.resolve()
        repo_root = cls.get_repo_root().resolve()
        try:
            resolved.relative_to(repo_root)
            return True
        except ValueError:
            return False

    @classmethod
    def resolve_scratch_dir(
        cls,
        custom_path: str | Path | None = None,
        subfolder: str | None = None,
        create: bool = True,
    ) -> Path:
        """Resolve the active scratch directory using the strict 6-tier hierarchy."""
        resolved: Path | None = None

        if custom_path is not None:
            resolved = Path(custom_path).resolve()
        else:
            for env_key in (cls.ENV_SCRATCH, cls.ENV_SCRATCH_DIR):
                env_val = os.environ.get(env_key)
                if env_val and env_val.strip():
                    candidate = Path(env_val.strip()).resolve()
                    if not cls._is_inside_repo(candidate):
                        resolved = candidate
                        break
                    else:
                        logger.warning(
                            f"Ignoring environment scratch path '{candidate}' as it violates Domain A air-gap."
                        )

            if resolved is None:
                for env_key in ("COCHEM_TMP", "SLURM_TMPDIR", "RUNNER_TEMP", "TMPDIR", "TEMP", "TMP"):
                    env_val = os.environ.get(env_key)
                    if env_val and env_val.strip():
                        candidate = (Path(env_val.strip()).resolve() / "cochem_scratch").resolve()
                        if not cls._is_inside_repo(candidate):
                            resolved = candidate
                            break

            if resolved is None:
                tier = cls.detect_tier()
                if tier == EnvironmentTier.LOCAL_MACOS:
                    resolved = (Path.home() / "Library" / "Caches" / "CoChem" / "scratch").resolve()
                elif tier == EnvironmentTier.LOCAL_WINDOWS:
                    local_app = os.environ.get("LOCALAPPDATA")
                    if local_app and local_app.strip():
                        resolved = (Path(local_app.strip()).resolve() / "CoChem" / "scratch").resolve()
                else:
                    xdg_cache = os.environ.get("XDG_CACHE_HOME")
                    if xdg_cache and xdg_cache.strip():
                        resolved = (Path(xdg_cache.strip()).resolve() / "cochem" / "scratch").resolve()
                    else:
                        resolved = (Path.home() / ".cache" / "cochem" / "scratch").resolve()

            if resolved is None:
                try:
                    temp_sys = Path(tempfile.gettempdir()).resolve()
                    resolved = (temp_sys / "cochem_scratch").resolve()
                except Exception:
                    resolved = None

            if resolved is None:
                resolved = (Path.home() / ".cochem" / "scratch").resolve()

        if subfolder:
            resolved = (resolved / subfolder).resolve()

        if custom_path is None:
            cls.verify_air_gap_boundary(resolved, ring=CoChemAirGapRing.RING_2_DOMAIN_C_EPHEMERAL)

        if create:
            resolved.mkdir(parents=True, exist_ok=True)

        return resolved

    @classmethod
    def resolve_deliverables_dir(
        cls,
        custom_path: str | Path | None = None,
        subfolder: str | None = None,
        create: bool = True,
    ) -> Path:
        """Resolve deliverables directory for permanent catalog, HDF5, and document outputs."""
        resolved: Path | None = None

        if custom_path is not None:
            resolved = Path(custom_path).resolve()
        else:
            for env_key in (
                cls.ENV_DELIVERABLES,
                cls.ENV_DELIVERABLES_DIR,
                cls.ENV_ARTIFACTS_DIR,
                cls.ENV_ARTIFACTS,
                cls.ENV_ARTIFACTS_FALLBACK,
            ):
                env_val = os.environ.get(env_key)
                if env_val and env_val.strip():
                    candidate = Path(env_val.strip()).resolve()
                    if not cls._is_inside_repo(candidate):
                        resolved = candidate
                        break
                    else:
                        logger.warning(
                            f"Ignoring environment deliverables path '{candidate}' as it violates Domain A air-gap."
                        )

            if resolved is None:
                tier = cls.detect_tier()
                if tier == EnvironmentTier.LOCAL_MACOS:
                    resolved = (Path.home() / "Library" / "Application Support" / "CoChem" / "artifacts").resolve()
                elif tier == EnvironmentTier.LOCAL_WINDOWS:
                    local_app = os.environ.get("LOCALAPPDATA")
                    if local_app and local_app.strip():
                        resolved = (Path(local_app.strip()).resolve() / "CoChem" / "artifacts").resolve()
                elif tier == EnvironmentTier.LOCAL_LINUX:
                    xdg_data = os.environ.get("XDG_DATA_HOME")
                    if xdg_data and xdg_data.strip():
                        resolved = (Path(xdg_data.strip()).resolve() / "cochem" / "artifacts").resolve()
                    else:
                        resolved = (Path.home() / ".local" / "share" / "cochem" / "artifacts").resolve()

            if resolved is None:
                home_artifacts = (Path.home() / "CoChem_Artifacts").resolve()
                if not cls._is_inside_repo(home_artifacts):
                    resolved = home_artifacts

            if resolved is None:
                resolved = (Path.home() / ".cochem" / "deliverables").resolve()

        if subfolder:
            resolved = (resolved / subfolder).resolve()

        if custom_path is None:
            cls.verify_air_gap_boundary(resolved, ring=CoChemAirGapRing.RING_3_DOMAIN_B_PERSISTENT)

        if create:
            resolved.mkdir(parents=True, exist_ok=True)

        return resolved

    @classmethod
    def verify_air_gap_boundary(
        cls,
        target_path: str | Path,
        ring: CoChemAirGapRing | int | str | None = None,
        cwd: str | Path | None = None,
    ) -> AirGapReport:
        """Adversarially verify that target_path strictly conforms to air-gap ring boundaries."""
        resolved_target = Path(target_path).resolve()
        resolved_cwd = (Path(cwd) if cwd is not None else Path.cwd()).resolve()
        repo_root = cls.get_repo_root().resolve()

        ring_val: int | None = None
        if isinstance(ring, CoChemAirGapRing):
            ring_val = ring.value
        elif isinstance(ring, int):
            ring_val = ring
        elif isinstance(ring, str):
            if "1" in ring or "A" in ring.upper() or "STATIC" in ring.upper():
                ring_val = 1
            elif "2" in ring or "C" in ring.upper() or "EPHEMERAL" in ring.upper() or "SCRATCH" in ring.upper():
                ring_val = 2
            elif "3" in ring or "B" in ring.upper() or "PERSISTENT" in ring.upper() or "ARTIFACT" in ring.upper():
                ring_val = 3

        if ring_val == 1:
            try:
                resolved_target.relative_to(repo_root)
            except ValueError:
                msg = (
                    f"CRITICAL AIR-GAP BREACH: Static Ring 1 target path '{resolved_target}' "
                    f"is not contained within immutable repository root '{repo_root}'."
                )
                logger.error(msg)
                raise AirGapViolationError(msg)
        else:
            if cls._is_inside_repo(resolved_target):
                msg = (
                    f"CRITICAL AIR-GAP BREACH: Mutable target path '{resolved_target}' "
                    f"is located within Ring 1 immutable repository root '{repo_root}'."
                )
                logger.error(msg)
                raise AirGapViolationError(msg)

        target_str = os.path.normcase(str(resolved_target))
        cwd_str = os.path.normcase(str(resolved_cwd))
        if ring_val in (2, 3) and target_str == cwd_str:
            msg = f"Air-gap breach: mutable path '{resolved_target}' is identical to working directory '{resolved_cwd}'."
            logger.error(msg)
            raise AirGapViolationError(msg)

        return AirGapReport(
            is_valid=True,
            cwd_resolved=resolved_cwd,
            target_resolved=resolved_target,
            reason=None,
        )

    @classmethod
    def assert_air_gap(cls, target_path: str | Path, allow_inside_repo: bool = False) -> None:
        """Asserts that target_path does not violate Ring 1 immutability."""
        if not allow_inside_repo and cls._is_inside_repo(Path(target_path)):
            raise AirGapViolationError(
                f"Air-gap violation: target path '{Path(target_path).resolve()}' "
                f"is located within immutable repository root '{cls.get_repo_root()}'."
            )

    @classmethod
    def get_scratch_dir(
        cls,
        subfolder: str | None = None,
        custom_path: str | Path | None = None,
        create: bool = True,
    ) -> Path:
        """Alias for resolve_scratch_dir."""
        return cls.resolve_scratch_dir(custom_path=custom_path, subfolder=subfolder, create=create)

    @classmethod
    def get_scratch_root(
        cls,
        subfolder: str | None = None,
        custom_path: str | Path | None = None,
        create: bool = True,
    ) -> Path:
        """Alias for resolve_scratch_dir."""
        return cls.resolve_scratch_dir(custom_path=custom_path, subfolder=subfolder, create=create)

    @classmethod
    def get_deliverables_dir(
        cls,
        custom_path: str | Path | None = None,
        subfolder: str | None = None,
        create: bool = True,
    ) -> Path:
        """Alias for resolve_deliverables_dir."""
        return cls.resolve_deliverables_dir(custom_path=custom_path, subfolder=subfolder, create=create)

    @classmethod
    def get_artifacts_dir(
        cls,
        subfolder: str | None = None,
        custom_path: str | Path | None = None,
        create: bool = True,
    ) -> Path:
        """Alias for resolve_deliverables_dir (Domain B / Ring 3 Artifact Vault)."""
        return cls.resolve_deliverables_dir(custom_path=custom_path, subfolder=subfolder, create=create)

    @classmethod
    def get_artifacts_root(
        cls,
        subfolder: str | None = None,
        custom_path: str | Path | None = None,
        create: bool = True,
    ) -> Path:
        """Alias for resolve_deliverables_dir."""
        return cls.resolve_deliverables_dir(custom_path=custom_path, subfolder=subfolder, create=create)

    @classmethod
    def get_artifact_directory(
        cls,
        env_var: str = "COCHEM_ARTIFACTS",
        fallback_dir: Path | str | None = None,
        create_if_missing: bool = True,
    ) -> Path:
        """Compatibility helper for dynamic artifact directory resolution."""
        if fallback_dir:
            return cls.resolve_deliverables_dir(custom_path=fallback_dir, create=create_if_missing)
        return cls.resolve_deliverables_dir(create=create_if_missing)

    @classmethod
    def create_shared_buffer(
        cls,
        name: str | None = None,
        size_bytes: int = 4096,
        create: bool = True,
        prefer_shm: bool = True,
    ) -> CoChemSharedMemoryBuffer:
        """Creates or attaches to a CoChemSharedMemoryBuffer within the resolved scratch workspace."""
        scratch_dir = cls.resolve_scratch_dir(create=True)
        return CoChemSharedMemoryBuffer(
            name=name,
            size=size_bytes,
            create=create,
            prefer_shm=prefer_shm,
            backing_dir=scratch_dir,
        )

    @property
    def scratch(self) -> Path:
        """Return instance resolved scratch directory."""
        return self.resolve_scratch_dir(self._custom_scratch)

    @property
    def deliverables(self) -> Path:
        """Return instance resolved deliverables directory."""
        return self.resolve_deliverables_dir(self._custom_deliverables)

    @property
    def artifacts(self) -> Path:
        """Return instance resolved artifacts directory."""
        return self.resolve_deliverables_dir(self._custom_deliverables)


# ============================================================================
# 8. Ghost Purger & Physical Sync Tools
# ============================================================================


def purge_ghost_outputs(
    target_dir: Path | str,
    remove_0byte_only: bool = False,
    recursive: bool = True,
) -> list[Path]:
    """Purges 0-byte ghost output files, failed partial calculations, and temporary staging artifacts.

    Args:
        target_dir: Directory containing computation outputs.
        remove_0byte_only: If True, only deletes 0-byte files regardless of extension.
                           If False, also deletes temporary staging files (.tmp, .part, .0, .lock).
        recursive: If True, searches subdirectories recursively.

    Returns:
        List of deleted Path objects.
    """
    directory = Path(target_dir).resolve()
    if not directory.exists() or not directory.is_dir():
        return []

    purged: list[Path] = []
    iterator = directory.rglob("*") if recursive else directory.glob("*")

    for file_path in iterator:
        if not file_path.is_file():
            continue

        try:
            file_size = file_path.stat().st_size
            should_remove = False

            if file_size == 0:
                should_remove = True
            elif not remove_0byte_only:
                suffix = file_path.suffix.lower()
                name = file_path.name.lower()
                if suffix in (".tmp", ".part", ".lock", ".temp", ".0", ".tmp0") or name.endswith(".tmp"):
                    should_remove = True

            if should_remove:
                remove_readonly_seal(file_path)
                file_path.unlink(missing_ok=True)
                purged.append(file_path)
                logger.info(f"Purged ghost/staging artifact: {file_path} (size={file_size} bytes)")
        except OSError as exc:
            logger.warning(f"Failed to purge file {file_path}: {exc}")

    return purged


def buffer_lock_sync(file_path: Path | str, min_bytes: int = 1) -> int:
    """Flushes operating system file buffers to physical disk and asserts non-zero byte capacity.

    Args:
        file_path: Path to the target physical file.
        min_bytes: Minimum expected file size in bytes (default: 1).

    Returns:
        Verified size of the file in bytes.

    Raises:
        CoChemIntegrityError: If the file does not exist, cannot be read, or is smaller than min_bytes.
    """
    target = Path(file_path).resolve()
    if not target.exists():
        raise CoChemIntegrityError(
            f"Buffer sync validation failed: Target file does not exist at '{target}'.",
            path=target,
        )

    if not target.is_file():
        raise CoChemIntegrityError(
            f"Buffer sync validation failed: Target path '{target}' is not a regular file.",
            path=target,
        )

    try:
        with open(target, "a+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, PermissionError) as exc:
        logger.debug(f"OS fsync on {target} skipped or failed: {exc}")

    actual_size = target.stat().st_size
    if actual_size < min_bytes:
        raise CoChemIntegrityError(
            f"Buffer sync validation failed: File '{target}' has {actual_size} bytes, "
            f"expected at least {min_bytes} bytes.",
            path=target,
        )

    return actual_size


def apply_readonly_chmod(path: str | Path, recursive: bool = True) -> None:
    """Applies an immutable read-only permission seal across Windows NTFS and POSIX platforms.

    Args:
        path: Path to file or directory to seal.
        recursive: If True and path is a directory, recursively seals all children.
    """
    target = Path(path).resolve()
    if not target.exists():
        return

    items: list[Path] = []
    if target.is_dir():
        if recursive:
            try:
                for child in target.rglob("*"):
                    items.append(child)
            except OSError as exc:
                logger.warning(f"Error traversing directory for readonly seal {target}: {exc}")
        items.append(target)
    else:
        items.append(target)

    is_windows = platform.system() == "Windows"
    for item in items:
        try:
            if is_windows:
                try:
                    FILE_ATTRIBUTE_READONLY = 0x00000001
                    ctypes.windll.kernel32.SetFileAttributesW(str(item), FILE_ATTRIBUTE_READONLY)
                except Exception:
                    pass
                current_mode = os.stat(item).st_mode
                os.chmod(item, current_mode & ~stat.S_IWRITE)
            else:
                if item.is_dir():
                    os.chmod(item, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
                else:
                    os.chmod(item, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        except OSError as exc:
            logger.warning(f"Failed to apply readonly seal to {item}: {exc}")


def remove_readonly_seal(path: str | Path, recursive: bool = True) -> None:
    """Removes read-only attributes to restore write permissions for cleanup operations.

    Args:
        path: Path to file or directory to unseal.
        recursive: If True and path is a directory, recursively unseals all children.
    """
    target = Path(path).resolve()
    if not target.exists():
        return

    items: list[Path] = []
    if target.is_dir():
        if recursive:
            try:
                for child in target.rglob("*"):
                    items.append(child)
            except OSError as exc:
                logger.warning(f"Error traversing directory for unseal {target}: {exc}")
        items.append(target)
    else:
        items.append(target)

    is_windows = platform.system() == "Windows"
    for item in items:
        try:
            if is_windows:
                try:
                    FILE_ATTRIBUTE_NORMAL = 0x00000080
                    ctypes.windll.kernel32.SetFileAttributesW(str(item), FILE_ATTRIBUTE_NORMAL)
                except Exception:
                    pass
                current_mode = os.stat(item).st_mode
                os.chmod(item, current_mode | stat.S_IWRITE | stat.S_IREAD)
            else:
                if item.is_dir():
                    os.chmod(item, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
                else:
                    os.chmod(item, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        except OSError as exc:
            logger.warning(f"Failed to remove readonly seal from {item}: {exc}")


# ============================================================================
# 9. Mendeleev Dynamic Element & Mass Retrieval (Mandate Enforced)
# ============================================================================


@functools.lru_cache(maxsize=256)
def _get_mendeleev_element(symbol_or_z: str | int) -> Any:
    """Queries and caches Mendeleev element objects dynamically."""
    if isinstance(symbol_or_z, int):
        return mendeleev_element(symbol_or_z)
    clean_sym = str(symbol_or_z).strip().rstrip(":").capitalize()
    return mendeleev_element(clean_sym)


def get_atomic_number(symbol_or_z: str | int) -> int:
    """Retrieve the atomic number (Z) dynamically from Mendeleev.

    Args:
        symbol_or_z: Element symbol (e.g. 'C', 'Fe') or atomic number.

    Returns:
        Integer atomic number Z.
    """
    elem = _get_mendeleev_element(symbol_or_z)
    return int(elem.atomic_number)


def get_atomic_mass(symbol_or_z: str | int) -> float:
    """Retrieve standard atomic mass (Da / u) dynamically from Mendeleev.

    Args:
        symbol_or_z: Element symbol (e.g. 'C', 'O') or atomic number.

    Returns:
        Float atomic weight in atomic mass units.
    """
    elem = _get_mendeleev_element(symbol_or_z)
    if elem.atomic_weight is None:
        raise ValueError(f"No standard atomic weight is tabulated for {elem.symbol}.")
    weight = float(elem.atomic_weight)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError(f"The standard atomic weight for {elem.symbol} is invalid.")
    return weight


def get_isotopic_mass(symbol_or_z: str | int, mass_number: int | None = None) -> float:
    """Retrieve a tabulated isotope mass using the shared isotope policy.

    Args:
        symbol_or_z: Element symbol or atomic number.
        mass_number: Specific isotopic mass number (e.g. 13 for 13C, 2 for 2H/D).

    Returns:
        Tabulated isotope mass in Daltons (u), with measurement uncertainty.
        Unspecified isotopes select the most abundant natural isotope; absence
        of a tabulated natural abundance requires an explicit mass number.
    """
    from Libraries.cochem_isotopes import isotope_mass

    if mass_number is not None and (
        type(mass_number) is not int or mass_number <= 0
    ):
        raise ValueError("An explicit isotope requires a positive integer mass number.")
    elem = _get_mendeleev_element(symbol_or_z)
    label = elem.symbol if mass_number is None else f"{mass_number}{elem.symbol}"
    return isotope_mass(label)


def enforce_ciaaw_masses(symbols: Sequence[str]) -> np.ndarray:
    """Map element/isotope symbols to the selected tabulated isotope masses.

    The historical function name is retained for compatibility. The installed
    Mendeleev database is the provider; its measured masses have uncertainty and
    are not independently verified CIAAW reference values or exact constants.
    Missing isotope masses or natural-abundance defaults are errors.

    Args:
        symbols: List or sequence of atomic symbols (e.g. ['C', 'H', 'H', '13C', 'D']).

    Returns:
        1D numpy array of dtype float64 containing tabulated masses in u.

    Raises:
        ValueError: If an unrecognized symbol or invalid element is provided.
    """
    if not symbols:
        return np.empty(0, dtype=np.float64)

    masses: list[float] = []
    iso_regex = re.compile(r"^(\d+)?([A-Za-z]+)$")

    for raw_sym in symbols:
        clean = raw_sym.strip()
        if not clean:
            raise ValueError("Empty or blank atomic symbol provided.")

        if clean.upper() == "D":
            masses.append(get_isotopic_mass("H", mass_number=2))
            continue
        elif clean.upper() == "T":
            masses.append(get_isotopic_mass("H", mass_number=3))
            continue

        match = iso_regex.match(clean)
        if not match:
            raise ValueError(f"Invalid chemical symbol syntax: '{clean}'.")

        mass_num_str, elem_str = match.groups()
        elem_name = elem_str.capitalize()

        if mass_num_str is not None:
            mass_num = int(mass_num_str)
            masses.append(get_isotopic_mass(elem_name, mass_number=mass_num))
        else:
            masses.append(get_isotopic_mass(elem_name))

    return np.array(masses, dtype=np.float64)


# ============================================================================
# 10. Module Exports
# ============================================================================

__all__ = [
    "AirGapReport",
    "AirGapViolationError",
    "BufferCorruptedError",
    "ChecksumMismatchError",
    "CoChemAirGapRing",
    "CoChemIntegrityError",
    "CoChemPathManager",
    "CoChemSharedMemoryBuffer",
    "EnvironmentTier",
    "IPCBufferError",
    "IPCBufferMetadata",
    "IPCPayloadType",
    "IPCSegmentType",
    "MethodMatrixViolationError",
    "OrphanedResourceError",
    "SharedMemoryAllocationError",
    "TierCapabilities",
    "apply_readonly_chmod",
    "buffer_lock_sync",
    "cleanup_ipc_scratch",
    "compute_buffer_checksum",
    "enforce_ciaaw_masses",
    "get_atomic_mass",
    "get_atomic_number",
    "get_isotopic_mass",
    "purge_ghost_outputs",
    "register_shm_buffer",
    "remove_readonly_seal",
    "unregister_shm_buffer",
]
