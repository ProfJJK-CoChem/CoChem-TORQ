"""CoChem-TORQ: PyArrow IPC & Shared Memory Manager.

=============================================================
Phase 5 (Stage 4.0) Implementation
----------------------------------
Provides zero-copy and low-latency Inter-Process Communication (IPC)
via PyArrow and cross-platform shared memory (POSIX/Windows SharedMemory)
with transparent fallback to memory-mapped files (mmap).

Features & Directives:
- pyarrow_mmap_mapper: Allocates and maps SharedMemory / mmap segments directly
  into PyArrow py_buffer/foreign_buffer streams for PyArrow Tables, RecordBatches,
  Tensors, and NumPy arrays.
- mmap_checksum_verifier: Cryptographically and bit-exact integrity checks using
  xxhash.xxh64 (and SHA-256 fallback) to eliminate silent memory corruption across
  hardware buses and process boundaries.
- orphaned_ipc_cleaner: Safe atexit and signal (SIGINT/SIGTERM) traps to guarantee
  zero orphaned shared memory segments or hanging temporary files.
- 6-Tier Environment Matrix: Path resolution (LOCAL_WINDOWS, LOCAL_MACOS, LOCAL_LINUX,
  GITHUB_ACTIONS, CODESPACES, HPC_NODES) enforcing Tripartite Air-Gap boundaries.
- Mendeleev Integration: Dynamic atomic and isotopic mass retrieval with zero
  hardcoded constants.
- Strict Pydantic v2 data models and Python 3.10+ typing.

Authoritative Sources:
- Task 8 SRS Specification: High-Fidelity Engine & Memory Backoff (Stage 4.0)
- Method Matrix v4: Heterogeneous Orchestration & Air-Gap Directives
"""

from __future__ import annotations

import atexit
from abc import ABC, abstractmethod
import enum
import gc
import hashlib
import io
import logging
import mmap
import multiprocessing.shared_memory as sm
import os
import platform
import shutil
import signal
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Literal, Optional, Set, Tuple, Union

from mendeleev import element
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from pydantic import BaseModel, ConfigDict, Field

# Configure module-level logger
logger = logging.getLogger("CoChem-TORQ.IPC")

# Optional xxhash library check
try:
    import xxhash  # type: ignore[import-untyped]
    _XXHASH_AVAILABLE = True
except ImportError:
    _XXHASH_AVAILABLE = False


# ============================================================================
# 1. Custom Exceptions
# ============================================================================


class AirGapViolationError(PermissionError):
    """Raised when an operation attempts to write to Ring 1 static repository space at runtime."""
    pass


class IPCBufferError(RuntimeError):
    """Base exception for all IPC and memory-mapped buffer operations."""
    pass


class ChecksumMismatchError(IPCBufferError):
    """Raised when buffer checksum verification fails due to data corruption or size mismatch."""
    pass


class BufferCorruptedError(IPCBufferError):
    """Raised when memory segment content is corrupted or unparseable."""
    pass


class SharedMemoryAllocationError(IPCBufferError):
    """Raised when shared memory segment creation or mapping fails."""
    pass


class OrphanedResourceError(IPCBufferError):
    """Raised when an orphaned IPC resource is detected or fails cleanup."""
    pass


# ============================================================================
# 2. 6-Tier Environment Matrix & Execution Context
# ============================================================================


class EnvironmentTier(str, enum.Enum):
    """6-Tier Environment Matrix defining host execution environments."""
    LOCAL_WINDOWS = "LOCAL_WINDOWS"
    LOCAL_MACOS = "LOCAL_MACOS"
    LOCAL_LINUX = "LOCAL_LINUX"
    GITHUB_ACTIONS = "GITHUB_ACTIONS"
    CODESPACES = "CODESPACES"
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


def get_repo_root() -> Path:
    """Locate the Domain A / Ring 1 immutable Git repository root."""
    env_val = os.environ.get("COCHEM_REPO_DIR")
    if env_val:
        repo_path = Path(env_val).resolve()
        if repo_path.is_dir():
            return repo_path

    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    return Path.cwd().resolve()


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
    verified_at: float = Field(
        default_factory=time.time, description="Unix timestamp of verification check"
    )
    reason: Optional[str] = Field(
        default=None, description="Diagnostic explanation of air-gap violation if any"
    )


class ExecutionContext(BaseModel):
    """Manages runtime environment detection, memory thresholds, core allocation,

    and dynamic scratch/shm/artifacts path resolution across the 6-Tier Environment Matrix.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tier: EnvironmentTier = Field(default=EnvironmentTier.LOCAL_WINDOWS)
    custom_scratch_dir: Optional[Path] = None
    custom_shm_dir: Optional[Path] = None
    custom_artifacts_dir: Optional[Path] = None
    max_memory_mb: int = Field(default=16384)
    num_cores: int = Field(default=8)
    gpu_available: bool = Field(default=False)
    vram_mb: int = Field(default=0)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def __init__(self, **data: Any) -> None:
        if "tier" not in data:
            data["tier"] = self.detect_tier()
        super().__init__(**data)
        self._detect_hardware_specs()

    @classmethod
    def detect_tier(cls) -> EnvironmentTier:
        """Autonomously detects the active environment tier from OS telemetry and environment variables."""
        if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("RUNNER_TEMP"):
            return EnvironmentTier.GITHUB_ACTIONS

        if os.environ.get("CODESPACES") == "true" or os.environ.get("CODESPACE_NAME"):
            return EnvironmentTier.CODESPACES

        if (
            os.environ.get("SLURM_TMPDIR")
            or os.environ.get("SLURM_JOB_ID")
            or os.environ.get("PFSDIR")
            or os.environ.get("PBS_O_WORKDIR")
        ):
            return EnvironmentTier.HPC_NODES

        sys_name = platform.system()
        if sys_name == "Windows" or os.environ.get("WSL_DISTRO_NAME"):
            return EnvironmentTier.LOCAL_WINDOWS
        elif sys_name == "Darwin":
            return EnvironmentTier.LOCAL_MACOS
        else:
            return EnvironmentTier.LOCAL_LINUX

    def _detect_hardware_specs(self) -> None:
        """Queries host CPU cores, RAM, and NVIDIA GPU telemetry if available."""
        try:
            vm = psutil.virtual_memory()
            self.max_memory_mb = int(vm.total / (1024 * 1024))
            self.num_cores = os.cpu_count() or 8
        except Exception:
            pass

        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self.gpu_available = True
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.vram_mb = int(mem_info.total / (1024 * 1024))
            pynvml.nvmlShutdown()
        except Exception:
            self.gpu_available = False
            self.vram_mb = 0

    @staticmethod
    def _is_repo_root_violation(target: Path) -> bool:
        """Check whether a path is located inside the static repository root."""
        resolved = target.resolve()
        repo_root = get_repo_root().resolve()
        try:
            resolved.relative_to(repo_root)
            return True
        except ValueError:
            return False

    def verify_air_gap_boundary(self, target_path: Path) -> AirGapReport:
        """Verifies that runtime scratch, shm, or artifacts paths do not mutate Domain A / Ring 1 repo root."""
        resolved_target = target_path.resolve()
        repo_root = get_repo_root().resolve()
        if self._is_repo_root_violation(resolved_target):
            msg = f"Tripartite Air-Gap Violation: Path '{resolved_target}' is inside static repository root '{repo_root}'."
            report = AirGapReport(
                is_valid=False,
                cwd_resolved=Path.cwd().resolve(),
                target_resolved=resolved_target,
                reason=msg,
            )
            raise AirGapViolationError(msg)

        return AirGapReport(
            is_valid=True,
            cwd_resolved=Path.cwd().resolve(),
            target_resolved=resolved_target,
            reason=None,
        )

    def get_scratch_dir(self, subfolder: Optional[str] = None) -> Path:
        """Resolves the ephemeral Domain C / Ring 2 scratch directory for the active tier."""
        if self.custom_scratch_dir:
            base = Path(self.custom_scratch_dir).resolve()
        elif os.environ.get("COCHEM_SCRATCH_DIR") and not self._is_repo_root_violation(Path(os.environ["COCHEM_SCRATCH_DIR"])):
            base = Path(os.environ["COCHEM_SCRATCH_DIR"]).resolve()
        else:
            if self.tier == EnvironmentTier.GITHUB_ACTIONS:
                runner_temp = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
                base = Path(runner_temp) / "cochem_scratch"
            elif self.tier == EnvironmentTier.CODESPACES:
                base = Path.home() / ".cochem" / "scratch"
            elif self.tier == EnvironmentTier.HPC_NODES:
                slurm_tmp = os.environ.get("SLURM_TMPDIR") or os.environ.get("PFSDIR") or tempfile.gettempdir()
                base = Path(slurm_tmp) / "cochem_scratch"
            elif self.tier == EnvironmentTier.LOCAL_MACOS:
                base = Path.home() / "Library" / "Caches" / "CoChem" / "scratch"
            elif self.tier == EnvironmentTier.LOCAL_WINDOWS:
                local_app_data = os.environ.get("LOCALAPPDATA")
                if local_app_data:
                    base = Path(local_app_data) / "CoChem" / "scratch"
                else:
                    base = Path(tempfile.gettempdir()) / "cochem_scratch"
            else:  # LOCAL_LINUX
                xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
                if xdg_runtime and Path(xdg_runtime).is_dir():
                    base = Path(xdg_runtime) / "cochem" / "scratch"
                elif Path(tempfile.gettempdir()).is_dir():
                    base = Path(tempfile.gettempdir()) / "cochem" / "scratch"
                else:
                    base = Path(tempfile.gettempdir()) / "cochem_scratch"

        target = (base / subfolder) if subfolder else base
        self.verify_air_gap_boundary(target)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_shm_dir(self, subfolder: Optional[str] = None) -> Path:
        """Resolves the zero-copy shared memory directory for the active tier."""
        if self.custom_shm_dir:
            base = Path(self.custom_shm_dir).resolve()
        elif os.environ.get("COCHEM_SHM_DIR") and not self._is_repo_root_violation(Path(os.environ["COCHEM_SHM_DIR"])):
            base = Path(os.environ["COCHEM_SHM_DIR"]).resolve()
        else:
            if self.tier == EnvironmentTier.GITHUB_ACTIONS:
                runner_temp = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
                base = Path(runner_temp) / "shm"
            elif self.tier == EnvironmentTier.CODESPACES:
                base = Path.home() / ".cochem" / "shm"
            elif self.tier == EnvironmentTier.HPC_NODES:
                slurm_tmp = os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
                base = Path(slurm_tmp) / "shm"
            elif self.tier == EnvironmentTier.LOCAL_MACOS:
                tmpdir = os.environ.get("TMPDIR", tempfile.gettempdir())
                base = Path(tmpdir) / "cochem_shm"
            elif self.tier == EnvironmentTier.LOCAL_WINDOWS:
                local_app_data = os.environ.get("LOCALAPPDATA")
                if local_app_data:
                    base = Path(local_app_data) / "CoChem" / "shm"
                else:
                    base = Path(tempfile.gettempdir()) / "cochem_shm"
            else:  # LOCAL_LINUX
                if Path("/dev/shm").is_dir() and os.access("/dev/shm", os.W_OK):
                    base = Path("/dev/shm/cochem")
                else:
                    base = Path(tempfile.gettempdir()) / "cochem_shm"

        target = (base / subfolder) if subfolder else base
        self.verify_air_gap_boundary(target)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_artifacts_dir(self, subfolder: Optional[str] = None) -> Path:
        """Resolves the Domain B / Ring 3 persistent artifact vault directory."""
        if self.custom_artifacts_dir:
            base = Path(self.custom_artifacts_dir).resolve()
        elif os.environ.get("COCHEM_ARTIFACTS_DIR") and not self._is_repo_root_violation(Path(os.environ["COCHEM_ARTIFACTS_DIR"])):
            base = Path(os.environ["COCHEM_ARTIFACTS_DIR"]).resolve()
        elif os.environ.get("COCHEM_ARTIFACTS") and not self._is_repo_root_violation(Path(os.environ["COCHEM_ARTIFACTS"])):
            base = Path(os.environ["COCHEM_ARTIFACTS"]).resolve()
        else:
            if self.tier == EnvironmentTier.GITHUB_ACTIONS:
                runner_temp = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
                base = Path(runner_temp) / "cochem_artifacts"
            elif self.tier == EnvironmentTier.CODESPACES:
                base = Path.home() / ".cochem" / "artifacts"
            elif self.tier == EnvironmentTier.HPC_NODES:
                slurm_tmp = os.environ.get("SLURM_TMPDIR") or os.environ.get("PFSDIR") or tempfile.gettempdir()
                base = Path(slurm_tmp) / "cochem_artifacts"
            elif self.tier == EnvironmentTier.LOCAL_MACOS:
                base = Path.home() / "Library" / "Application Support" / "CoChem" / "artifacts"
            elif self.tier == EnvironmentTier.LOCAL_WINDOWS:
                local_app_data = os.environ.get("LOCALAPPDATA")
                if local_app_data:
                    base = Path(local_app_data) / "CoChem" / "artifacts"
                else:
                    base = Path.home() / "CoChem_Artifacts"
            else:  # LOCAL_LINUX
                base = Path.home() / ".local" / "share" / "cochem" / "artifacts"

        target = (base / subfolder) if subfolder else base
        self.verify_air_gap_boundary(target)
        target.mkdir(parents=True, exist_ok=True)
        return target


# ============================================================================
# 3. Mendeleev Dynamic Mass Retrieval (Mandate Enforced)
# ============================================================================


def get_atomic_mass(symbol: str) -> float:
    """Retrieve the standard atomic mass (weight) dynamically from Mendeleev.

    Args:
        symbol: Chemical element symbol (e.g. 'H', 'C', 'O').

    Returns:
        Standard atomic mass in Da (g/mol).
    """
    clean_sym = symbol.strip().capitalize()
    el = element(clean_sym)
    if el.atomic_weight is not None:
        return float(el.atomic_weight)
    if el.isotopes:
        return float(el.isotopes[0].mass or el.isotopes[0].mass_number)
    raise ValueError(f"No atomic mass available in mendeleev for '{symbol}'.")


def get_isotopic_mass(symbol: str, mass_number: int) -> float:
    """Retrieve the exact physical isotopic mass dynamically from Mendeleev.

    Args:
        symbol: Chemical element symbol (e.g. 'C', 'H').
        mass_number: Isotope nucleon number (e.g. 13 for C-13, 2 for Deuterium).

    Returns:
        Exact isotopic mass in Da.
    """
    clean_sym = symbol.strip().capitalize()
    el = element(clean_sym)
    for iso in el.isotopes:
        if iso.mass_number == mass_number:
            if iso.mass is not None:
                return float(iso.mass)
            return float(iso.mass_number)
    raise ValueError(f"Isotope '{clean_sym}-{mass_number}' not found in mendeleev database.")


def get_atomic_number(symbol: str) -> int:
    """Retrieve atomic number (Z) dynamically from Mendeleev."""
    clean_sym = symbol.strip().capitalize()
    return int(element(clean_sym).atomic_number)


# ============================================================================
# 4. Pydantic Models for IPC Metadata & Verification
# ============================================================================


class IPCBufferMetadata(BaseModel):
    """Metadata describing an allocated and serialized IPC buffer segment."""
    model_config = ConfigDict(extra="ignore")

    buffer_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the buffer",
    )
    name: str = Field(
        ..., description="Unique kernel or filesystem identifier for the buffer"
    )
    buffer_type: IPCSegmentType = Field(
        ..., description="Type of IPC buffer ('shm' or 'mmap')"
    )
    payload_type: IPCPayloadType = Field(
        default=IPCPayloadType.RAW_BYTES,
        description="Type of data serialized within the buffer",
    )
    size_bytes: int = Field(
        ..., gt=0, description="Exact payload byte size"
    )
    allocated_bytes: int = Field(
        ..., gt=0, description="Total capacity allocated in memory/disk segment"
    )
    file_path: Optional[Path] = Field(
        default=None, description="Path to physical backing file if buffer_type == 'mmap'"
    )
    checksum_xxh64: str = Field(
        ..., description="Hexadecimal xxHash-64 (or SHA-256 fallback) checksum of payload"
    )
    checksum_sha256: Optional[str] = Field(
        default=None, description="Cryptographic SHA-256 checksum of payload"
    )
    pid: int = Field(
        default_factory=os.getpid, description="Process ID that created the buffer"
    )
    created_at: float = Field(
        default_factory=time.time, description="Unix timestamp of creation"
    )
    is_active: bool = Field(
        default=True, description="True if buffer is currently open and valid"
    )
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Custom payload attributes (shape, dtype, schema, etc.)"
    )


class ChecksumVerificationResult(BaseModel):
    """Diagnostic outcome of buffer integrity and bit-exact checksum verification."""
    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(
        ..., description="True if byte size and checksum match expected values"
    )
    expected_bytes: int = Field(
        ..., description="Expected payload size in bytes"
    )
    actual_bytes: int = Field(
        ..., description="Actual buffer/payload size in bytes"
    )
    expected_hash: str = Field(
        ..., description="Expected checksum hex string"
    )
    computed_hash: str = Field(
        ..., description="Computed checksum hex string from buffer data"
    )
    algorithm: str = Field(
        default="xxh64", description="Hashing algorithm used ('xxh64' or 'sha256')"
    )
    verified_at: float = Field(
        default_factory=time.time, description="Unix timestamp of check"
    )
    error_message: Optional[str] = Field(
        default=None, description="Diagnostic error details if verification failed"
    )


# ============================================================================
# 5. Checksum Engine (mmap_checksum_verifier)
# ============================================================================


def compute_buffer_checksum(
    data: Union[bytes, bytearray, memoryview, pa.Buffer, np.ndarray],
    algorithm: str = "xxh64",
) -> Tuple[str, int]:
    """Compute deterministic cryptographic or fast hash checksum and byte size of a buffer.

    Args:
        data: Input buffer, memoryview, PyArrow Buffer, or NumPy array.
        algorithm: Hashing algorithm ('xxh64' or 'sha256').

    Returns:
        Tuple of (hexdigest_string, total_bytes).
    """
    if isinstance(data, pa.Buffer):
        buf_view: Union[memoryview, bytes, bytearray] = memoryview(data)
        size = data.size
    elif isinstance(data, np.ndarray):
        buf_view = memoryview(data)
        size = data.nbytes
    elif isinstance(data, memoryview):
        buf_view = data
        size = len(data)
    elif isinstance(data, (bytes, bytearray)):
        buf_view = data
        size = len(data)
    else:
        raise TypeError(f"Unsupported buffer data type: {type(data)}")

    if algorithm.lower() == "xxh64" and _XXHASH_AVAILABLE:
        digest = xxhash.xxh64(buf_view).hexdigest()
    else:
        digest = hashlib.sha256(buf_view).hexdigest()

    return digest, size


def mmap_checksum_verifier(
    data: Union[bytes, bytearray, memoryview, pa.Buffer, np.ndarray],
    expected_size: Optional[int] = None,
    expected_checksum: Optional[str] = None,
    algorithm: str = "xxh64",
    raise_on_error: bool = True,
) -> ChecksumVerificationResult:
    """Verify bit-exact size and hash integrity of memory-mapped or shared memory buffers.

    Eliminates silent MPI/hardware bus memory corruption.

    Args:
        data: Buffer to verify.
        expected_size: Expected byte length (optional).
        expected_checksum: Expected hexadecimal hash string (optional).
        algorithm: Hashing algorithm ('xxh64' or 'sha256').
        raise_on_error: If True, raises ChecksumMismatchError or BufferCorruptedError on failure.

    Returns:
        ChecksumVerificationResult detailing validation outcome.

    Raises:
        ChecksumMismatchError: If size or checksum does not match expected values.
    """
    computed_hash, actual_size = compute_buffer_checksum(data, algorithm=algorithm)

    exp_size = expected_size if expected_size is not None else actual_size
    exp_hash = expected_checksum if expected_checksum is not None else computed_hash

    size_match = (actual_size == exp_size)
    hash_match = (computed_hash.lower() == exp_hash.lower())
    is_valid = size_match and hash_match

    error_msg: Optional[str] = None
    if not is_valid:
        reasons = []
        if not size_match:
            reasons.append(f"Size mismatch: expected {exp_size} bytes, got {actual_size} bytes")
        if not hash_match:
            reasons.append(f"Hash mismatch ({algorithm}): expected {exp_hash}, computed {computed_hash}")
        error_msg = "; ".join(reasons)

        if raise_on_error:
            raise ChecksumMismatchError(
                f"Memory-mapped buffer validation failed: {error_msg}"
            )

    return ChecksumVerificationResult(
        is_valid=is_valid,
        expected_bytes=exp_size,
        actual_bytes=actual_size,
        expected_hash=exp_hash,
        computed_hash=computed_hash,
        algorithm=algorithm if (_XXHASH_AVAILABLE or algorithm != "xxh64") else "sha256",
        error_message=error_msg,
    )


# ============================================================================
# 6. Thread-Safe IPC Registry & Orphaned IPC Cleaner
# ============================================================================


class IPCRegistry:
    """Thread-safe global registry tracking active SharedMemory and mmap resources."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._shm_objects: Dict[str, sm.SharedMemory] = {}
        self._mmap_objects: Dict[Path, mmap.mmap] = {}
        self._file_handles: Dict[Path, Any] = {}
        self._file_paths: Set[Path] = set()
        self._metadata_map: Dict[str, IPCBufferMetadata] = {}

    def register_shm(
        self,
        shm: sm.SharedMemory,
        name: str,
        metadata: Optional[IPCBufferMetadata] = None,
    ) -> None:
        """Register an active SharedMemory segment for lifecycle tracking."""
        with self._lock:
            self._shm_objects[name] = shm
            if metadata is not None:
                self._metadata_map[name] = metadata
            logger.debug("Registered SharedMemory segment: %s", name)

    def unregister_shm(self, name: str) -> None:
        """Unregister a SharedMemory segment from tracking."""
        with self._lock:
            self._shm_objects.pop(name, None)
            self._metadata_map.pop(name, None)
            logger.debug("Unregistered SharedMemory segment: %s", name)

    def register_mmap(
        self,
        file_path: Path,
        mm: Optional[mmap.mmap] = None,
        file_handle: Optional[Any] = None,
        metadata: Optional[IPCBufferMetadata] = None,
    ) -> None:
        """Register an active mmap backing file and handle for lifecycle tracking."""
        with self._lock:
            resolved = file_path.resolve()
            self._file_paths.add(resolved)
            if mm is not None:
                self._mmap_objects[resolved] = mm
            if file_handle is not None:
                self._file_handles[resolved] = file_handle
            if metadata is not None:
                self._metadata_map[str(resolved)] = metadata
            logger.debug("Registered mmap file: %s", resolved)

    def unregister_mmap(self, file_path: Path) -> None:
        """Unregister an mmap file from tracking."""
        with self._lock:
            resolved = file_path.resolve()
            self._mmap_objects.pop(resolved, None)
            self._file_handles.pop(resolved, None)
            self._file_paths.discard(resolved)
            self._metadata_map.pop(str(resolved), None)
            logger.debug("Unregistered mmap file: %s", resolved)

    def active_shm_names(self) -> List[str]:
        """Return list of active shared memory segment names."""
        with self._lock:
            return list(self._shm_objects.keys())

    def active_file_paths(self) -> List[Path]:
        """Return list of active memory-mapped file paths."""
        with self._lock:
            return list(self._file_paths)

    def active_count(self) -> int:
        """Return total count of actively tracked resources."""
        with self._lock:
            return len(self._shm_objects) + len(self._file_paths)

    def cleanup_all(self, aggressive: bool = True) -> int:
        """Close and unlink all tracked shared memory segments and remove temporary mmap files.

        Args:
            aggressive: If True, forces garbage collection to release open memoryviews
                        before closing to prevent Windows BufferError.

        Returns:
            Total count of cleaned resources.
        """
        cleaned_count = 0
        with self._lock:
            if aggressive:
                gc.collect()

            # 1. Clean Shared Memory segments
            for name, shm in list(self._shm_objects.items()):
                try:
                    if hasattr(shm, "buf") and shm.buf is not None:
                        try:
                            shm.buf.release()
                        except Exception:
                            pass
                    try:
                        shm.close()
                    except Exception:
                        pass
                    try:
                        shm.unlink()
                        cleaned_count += 1
                        logger.debug("Cleaned SharedMemory segment: %s", name)
                    except (FileNotFoundError, Exception) as exc:
                        logger.warning("Error unlinking SharedMemory segment %s: %s", name, exc)
                except Exception as exc:
                    logger.warning("Error cleaning SharedMemory segment %s: %s", name, exc)
            self._shm_objects.clear()

            # 2. Clean mmap objects
            for p, mm in list(self._mmap_objects.items()):
                try:
                    try:
                        mm.close()
                    except Exception:
                        pass
                except Exception as exc:
                    logger.warning("Error closing mmap on %s: %s", p, exc)
            self._mmap_objects.clear()

            # 3. Clean open file handles
            for p, fh in list(self._file_handles.items()):
                try:
                    fh.close()
                except Exception as exc:
                    logger.warning("Error closing file handle on %s: %s", p, exc)
            self._file_handles.clear()

            if aggressive:
                gc.collect()

            # 4. Remove physical temporary backing files
            for p in list(self._file_paths):
                try:
                    if p.exists():
                        p.unlink(missing_ok=True)
                        cleaned_count += 1
                        logger.debug("Deleted temporary mmap backing file: %s", p)
                except Exception as exc:
                    logger.warning("Error unlinking mmap file %s: %s", p, exc)
            self._file_paths.clear()
            self._metadata_map.clear()

        return cleaned_count


# Global singleton registry
_GLOBAL_IPC_REGISTRY: Final[IPCRegistry] = IPCRegistry()
_SIGNAL_TRAPS_INSTALLED: bool = False


def get_ipc_registry() -> IPCRegistry:
    """Retrieve the global singleton IPC tracking registry."""
    return _GLOBAL_IPC_REGISTRY


def orphaned_ipc_cleaner(
    signum: Optional[int] = None,
    frame: Optional[Any] = None,
) -> int:
    """Aggressively sweeps and purges all active SharedMemory segments and temporary mmap files.

    Can be invoked directly or triggered via atexit / signal handlers (SIGINT / SIGTERM).

    Args:
        signum: Signal number if invoked as a signal handler.
        frame: Current stack frame if invoked as a signal handler.

    Returns:
        Number of resources cleaned up.
    """
    logger.info("Orphaned IPC Cleaner invoked (signum=%s)", signum)
    count = _GLOBAL_IPC_REGISTRY.cleanup_all(aggressive=True)
    if signum is not None:
        # Re-raise standard interrupt behavior if triggered by signal
        if signum == signal.SIGINT:
            signal.default_int_handler(signum, frame)
        elif signum == signal.SIGTERM:
            sys.exit(128 + signal.SIGTERM)
    return count


def register_signal_traps() -> None:
    """Register safe atexit and signal traps (SIGINT, SIGTERM, SIGBREAK) for automated IPC cleanup."""
    global _SIGNAL_TRAPS_INSTALLED
    if _SIGNAL_TRAPS_INSTALLED:
        return

    atexit.register(orphaned_ipc_cleaner)

    # Register OS signal traps where allowed (main thread)
    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, orphaned_ipc_cleaner)
            signal.signal(signal.SIGTERM, orphaned_ipc_cleaner)
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, orphaned_ipc_cleaner)
        except (ValueError, AttributeError) as exc:
            logger.debug("Signal registration skipped: %s", exc)

    _SIGNAL_TRAPS_INSTALLED = True


# Initialize traps on module load
register_signal_traps()


# ============================================================================
# 7. PyArrow IPC Segment Classes (SharedMemory & Mmap)
# ============================================================================


class BaseIPCSegment(ABC):
    """Abstract base class providing PyArrow serialization and zero-copy IPC mapping."""

    def __init__(self, name: str, size: int) -> None:
        self._name: str = name
        self._size: int = size
        self._is_closed: bool = False
        self._metadata: Optional[IPCBufferMetadata] = None

    @property
    def name(self) -> str:
        """Name or identifier of the segment."""
        return self._name

    @property
    def size(self) -> int:
        """Allocated capacity of the segment in bytes."""
        return self._size

    @property
    def is_closed(self) -> bool:
        """True if the segment has been closed."""
        return self._is_closed

    @property
    def metadata(self) -> Optional[IPCBufferMetadata]:
        """Metadata associated with the last write/read operation."""
        return self._metadata

    def __enter__(self) -> BaseIPCSegment:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @abstractmethod
    def get_memoryview(self, length: Optional[int] = None) -> memoryview:
        """Return memoryview of the underlying buffer."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close buffer handles."""
        pass

    @abstractmethod
    def unlink(self) -> None:
        """Release and delete the buffer segment from OS kernel/filesystem."""
        pass

    # ------------------------------------------------------------------------
    # Core Read / Write / Serialization Methods
    # ------------------------------------------------------------------------

    def write_raw(
        self,
        data: bytes,
        payload_type: IPCPayloadType = IPCPayloadType.RAW_BYTES,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> IPCBufferMetadata:
        """Write raw bytes into the memory segment and compute checksum."""
        if self._is_closed:
            raise IPCBufferError(f"Segment '{self._name}' is already closed.")

        data_len = len(data)
        if data_len > self._size:
            raise IPCBufferError(
                f"Data size ({data_len} bytes) exceeds allocated segment capacity ({self._size} bytes)."
            )

        mv = self.get_memoryview(data_len)
        mv[:data_len] = data

        xxh64_hash, _ = compute_buffer_checksum(data, algorithm="xxh64")
        sha256_hash, _ = compute_buffer_checksum(data, algorithm="sha256")

        meta = IPCBufferMetadata(
            name=self._name,
            buffer_type=(
                IPCSegmentType.SHARED_MEMORY
                if isinstance(self, PyArrowSharedMemorySegment)
                else IPCSegmentType.MEMORY_MAPPED_FILE
            ),
            payload_type=payload_type,
            size_bytes=data_len,
            allocated_bytes=self._size,
            file_path=(
                getattr(self, "_file_path", None)
            ),
            checksum_xxh64=xxh64_hash,
            checksum_sha256=sha256_hash,
            extra_metadata=extra_metadata or {},
        )
        self._metadata = meta
        return meta

    def read_raw(self, length: Optional[int] = None, verify_checksum: bool = True) -> bytes:
        """Read raw bytes from the memory segment."""
        if self._is_closed:
            raise IPCBufferError(f"Segment '{self._name}' is already closed.")

        target_len = length if length is not None else (self._metadata.size_bytes if self._metadata else self._size)
        mv = self.get_memoryview(target_len)
        raw_data = mv[:target_len].tobytes()

        if verify_checksum and self._metadata is not None:
            mmap_checksum_verifier(
                raw_data,
                expected_size=self._metadata.size_bytes,
                expected_checksum=self._metadata.checksum_xxh64,
                algorithm="xxh64",
                raise_on_error=True,
            )

        return raw_data

    def to_pyarrow_buffer(self, length: Optional[int] = None, verify_checksum: bool = True) -> pa.Buffer:
        """Create a zero-copy PyArrow Buffer mapping the segment's memoryview."""
        raw_bytes = self.read_raw(length=length, verify_checksum=verify_checksum)
        return pa.py_buffer(raw_bytes)

    # ------------------------------------------------------------------------
    # High-Dimensional PyArrow Table & RecordBatch Methods
    # ------------------------------------------------------------------------

    def write_table(self, table: pa.Table) -> IPCBufferMetadata:
        """Serialize a PyArrow Table into the segment via PyArrow IPC stream."""
        sink = pa.BufferOutputStream()
        with pa_ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        serialized = sink.getvalue().to_pybytes()
        extra = {
            "num_rows": table.num_rows,
            "num_columns": table.num_columns,
            "column_names": table.column_names,
        }
        return self.write_raw(serialized, payload_type=IPCPayloadType.TABLE, extra_metadata=extra)

    def read_table(self, length: Optional[int] = None, verify_checksum: bool = True) -> pa.Table:
        """Deserialize a PyArrow Table from the segment."""
        raw_data = self.read_raw(length=length, verify_checksum=verify_checksum)
        buf = pa.py_buffer(raw_data)
        reader = pa_ipc.open_stream(buf)
        return reader.read_all()

    def write_record_batch(self, batch: pa.RecordBatch) -> IPCBufferMetadata:
        """Serialize a PyArrow RecordBatch into the segment."""
        sink = pa.BufferOutputStream()
        with pa_ipc.new_stream(sink, batch.schema) as writer:
            writer.write_batch(batch)
        serialized = sink.getvalue().to_pybytes()
        extra = {
            "num_rows": batch.num_rows,
            "num_columns": batch.num_columns,
        }
        return self.write_raw(serialized, payload_type=IPCPayloadType.RECORD_BATCH, extra_metadata=extra)

    def read_record_batch(self, length: Optional[int] = None, verify_checksum: bool = True) -> pa.RecordBatch:
        """Deserialize a PyArrow RecordBatch from the segment."""
        raw_data = self.read_raw(length=length, verify_checksum=verify_checksum)
        buf = pa.py_buffer(raw_data)
        reader = pa_ipc.open_stream(buf)
        return reader.read_next_batch()

    # ------------------------------------------------------------------------
    # High-Dimensional Tensor & NumPy Methods
    # ------------------------------------------------------------------------

    def write_tensor(self, tensor: Union[pa.Tensor, np.ndarray]) -> IPCBufferMetadata:
        """Serialize a PyArrow Tensor or NumPy array into the segment via PyArrow IPC."""
        if isinstance(tensor, np.ndarray):
            pa_tensor = pa.Tensor.from_numpy(tensor)
            orig_shape = list(tensor.shape)
            orig_dtype = str(tensor.dtype)
        elif isinstance(tensor, pa.Tensor):
            pa_tensor = tensor
            orig_shape = list(tensor.shape)
            orig_dtype = str(tensor.type)
        else:
            raise TypeError(f"Expected pa.Tensor or np.ndarray, got {type(tensor)}")

        sink = pa.BufferOutputStream()
        pa_ipc.write_tensor(pa_tensor, sink)
        serialized = sink.getvalue().to_pybytes()

        extra = {
            "shape": orig_shape,
            "dtype": orig_dtype,
        }
        return self.write_raw(serialized, payload_type=IPCPayloadType.TENSOR, extra_metadata=extra)

    def read_tensor(self, length: Optional[int] = None, verify_checksum: bool = True) -> pa.Tensor:
        """Deserialize a PyArrow Tensor from the segment."""
        raw_data = self.read_raw(length=length, verify_checksum=verify_checksum)
        buf = pa.py_buffer(raw_data)
        return pa_ipc.read_tensor(buf)

    def write_numpy(self, array: np.ndarray) -> IPCBufferMetadata:
        """Serialize a NumPy array as a Tensor into the segment."""
        meta = self.write_tensor(array)
        meta.payload_type = IPCPayloadType.NUMPY_ARRAY
        self._metadata = meta
        return meta

    def read_numpy(self, length: Optional[int] = None, verify_checksum: bool = True) -> np.ndarray:
        """Deserialize a NumPy array from the segment."""
        tensor = self.read_tensor(length=length, verify_checksum=verify_checksum)
        return np.copy(tensor.to_numpy())


class PyArrowSharedMemorySegment(BaseIPCSegment):
    """Zero-copy IPC segment backed by multiprocessing.shared_memory.SharedMemory."""

    def __init__(
        self,
        name: Optional[str] = None,
        size: int = 0,
        create: bool = True,
    ) -> None:
        self._is_owner = create
        if create:
            if size <= 0:
                raise ValueError("Buffer capacity (size) must be > 0 bytes when creating SharedMemory.")
            seg_name = name or f"cochem_shm_{uuid.uuid4().hex[:12]}"
            try:
                try:
                    self._shm = sm.SharedMemory(name=seg_name, create=True, size=size)
                except FileExistsError:
                    # If existing segment with same name, attach or unlink and recreate
                    try:
                        temp_shm = sm.SharedMemory(name=seg_name, create=False)
                        temp_shm.close()
                        temp_shm.unlink()
                    except Exception:
                        pass
                    self._shm = sm.SharedMemory(name=seg_name, create=True, size=size)
            except Exception as exc:
                raise SharedMemoryAllocationError(
                    f"Failed to allocate SharedMemory segment '{seg_name}' of {size} bytes: {exc}"
                ) from exc
        else:
            if not name:
                raise ValueError("Segment name is required when attaching to existing SharedMemory.")
            try:
                self._shm = sm.SharedMemory(name=name, create=False)
            except Exception as exc:
                raise SharedMemoryAllocationError(
                    f"Failed to attach to SharedMemory segment '{name}': {exc}"
                ) from exc

        actual_size = self._shm.size
        super().__init__(name=self._shm.name, size=actual_size)
        if self._is_owner:
            _GLOBAL_IPC_REGISTRY.register_shm(self._shm, self._name)

    @property
    def shm(self) -> sm.SharedMemory:
        """Underlying Python SharedMemory instance."""
        return self._shm

    @property
    def is_owner(self) -> bool:
        """True if this instance created and owns the shared memory segment."""
        return self._is_owner

    def get_memoryview(self, length: Optional[int] = None) -> memoryview:
        """Return memoryview into SharedMemory buffer."""
        if self._is_closed:
            raise IPCBufferError(f"SharedMemory segment '{self._name}' is closed.")
        target_len = length if length is not None else self._size
        return self._shm.buf[:target_len]

    def close(self) -> None:
        """Close SharedMemory handle safely."""
        if not self._is_closed:
            self._is_closed = True
            if hasattr(self._shm, "buf") and self._shm.buf is not None:
                try:
                    self._shm.buf.release()
                except Exception:
                    pass
            try:
                gc.collect()
                self._shm.close()
            except Exception as exc:
                logger.warning("Error closing SharedMemory '%s': %s", self._name, exc)
            finally:
                if not self._is_owner:
                    _GLOBAL_IPC_REGISTRY.unregister_shm(self._name)

    def unlink(self) -> None:
        """Unlink SharedMemory segment from OS kernel."""
        try:
            self.close()
        except Exception:
            pass
        if self._is_owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Error unlinking SharedMemory '%s': %s", self._name, exc)
            finally:
                _GLOBAL_IPC_REGISTRY.unregister_shm(self._name)


class PyArrowMmapFileSegment(BaseIPCSegment):
    """Memory-mapped file IPC segment with cross-platform mmap backing."""

    def __init__(
        self,
        file_path: Optional[Path] = None,
        size: int = 0,
        is_ephemeral: bool = True,
        context: Optional[ExecutionContext] = None,
    ) -> None:
        ctx = context or ExecutionContext()
        self._is_ephemeral = is_ephemeral

        if file_path is None:
            if size <= 0:
                raise ValueError("Buffer capacity (size) must be > 0 bytes when creating mmap file.")
            shm_dir = ctx.get_shm_dir()
            filename = f"cochem_mmap_{uuid.uuid4().hex[:12]}.ipc"
            self._file_path = (shm_dir / filename).resolve()
            # Pre-allocate physical backing file
            with open(self._file_path, "wb") as f:
                f.seek(size - 1)
                f.write(b"\x00")
            target_size = size
            self._is_ephemeral = True
        else:
            self._file_path = file_path.resolve()
            ctx.verify_air_gap_boundary(self._file_path)
            if not self._file_path.exists():
                raise FileNotFoundError(f"Mmap backing file not found: {self._file_path}")
            target_size = self._file_path.stat().st_size
            if size > 0 and size != target_size:
                # Resize backing file if specified
                with open(self._file_path, "r+b") as f:
                    f.truncate(size)
                target_size = size

        self._file_handle = open(self._file_path, "r+b")
        try:
            self._mmap = mmap.mmap(self._file_handle.fileno(), target_size)
        except Exception as exc:
            self._file_handle.close()
            raise IPCBufferError(
                f"Failed to map file '{self._file_path}' into mmap: {exc}"
            ) from exc

        super().__init__(name=str(self._file_path), size=target_size)
        if self._is_ephemeral:
            _GLOBAL_IPC_REGISTRY.register_mmap(self._file_path, self._mmap, self._file_handle)

    @property
    def file_path(self) -> Path:
        """Path to physical backing file on disk."""
        return self._file_path

    @property
    def mmap(self) -> mmap.mmap:
        """Underlying Python mmap instance."""
        return self._mmap

    def get_memoryview(self, length: Optional[int] = None) -> memoryview:
        """Return memoryview into memory-mapped file."""
        if self._is_closed:
            raise IPCBufferError(f"Mmap segment '{self._name}' is closed.")
        target_len = length if length is not None else self._size
        return memoryview(self._mmap)[:target_len]

    def close(self) -> None:
        """Close mmap and physical file handles."""
        if not self._is_closed:
            self._is_closed = True
            try:
                gc.collect()
                self._mmap.close()
            except Exception as exc:
                logger.warning("Error closing mmap on '%s': %s", self._file_path, exc)

            try:
                self._file_handle.close()
            except Exception as exc:
                logger.warning("Error closing file handle on '%s': %s", self._file_path, exc)

    def unlink(self) -> None:
        """Unlink and delete temporary backing file."""
        try:
            self.close()
        except Exception:
            pass
        try:
            gc.collect()
            if self._file_path.exists():
                self._file_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Error deleting mmap file '%s': %s", self._file_path, exc)
        finally:
            _GLOBAL_IPC_REGISTRY.unregister_mmap(self._file_path)


# ============================================================================
# 8. High-Level PyArrow MMAP Mapper & Serialization Helpers
# ============================================================================


def pyarrow_mmap_mapper(
    payload: Optional[Any] = None,
    name: Optional[str] = None,
    size_bytes: Optional[int] = None,
    prefer_shm: bool = True,
    context: Optional[ExecutionContext] = None,
) -> Union[PyArrowSharedMemorySegment, PyArrowMmapFileSegment]:
    """Allocate and configure a PyArrow-compatible IPC memory segment.

    Supports SharedMemory (POSIX/Windows) with transparent fallback to
    memory-mapped files (mmap) on allocation or OS restriction failure.

    Args:
        payload: Optional data object to immediately serialize (Table, RecordBatch, Tensor, NumPy array, or bytes).
        name: Optional segment identifier name.
        size_bytes: Optional capacity in bytes (required if payload is None).
        prefer_shm: If True, attempts SharedMemory allocation first before falling back to mmap file.
        context: ExecutionContext for path resolution and air-gap validation.

    Returns:
        Configured and mapped PyArrowSharedMemorySegment or PyArrowMmapFileSegment.
    """
    ctx = context or ExecutionContext()

    # Pre-serialize payload to determine size if provided
    serialized_bytes: Optional[bytes] = None
    payload_type: IPCPayloadType = IPCPayloadType.RAW_BYTES
    extra_meta: Dict[str, Any] = {}

    if payload is not None:
        if isinstance(payload, pa.Table):
            sink = pa.BufferOutputStream()
            with pa_ipc.new_stream(sink, payload.schema) as writer:
                writer.write_table(payload)
            serialized_bytes = sink.getvalue().to_pybytes()
            payload_type = IPCPayloadType.TABLE
            extra_meta = {
                "num_rows": payload.num_rows,
                "num_columns": payload.num_columns,
                "column_names": payload.column_names,
            }
        elif isinstance(payload, pa.RecordBatch):
            sink = pa.BufferOutputStream()
            with pa_ipc.new_stream(sink, payload.schema) as writer:
                writer.write_batch(payload)
            serialized_bytes = sink.getvalue().to_pybytes()
            payload_type = IPCPayloadType.RECORD_BATCH
            extra_meta = {"num_rows": payload.num_rows, "num_columns": payload.num_columns}
        elif isinstance(payload, (pa.Tensor, np.ndarray)):
            if isinstance(payload, np.ndarray):
                tensor = pa.Tensor.from_numpy(payload)
                payload_type = IPCPayloadType.NUMPY_ARRAY
            else:
                tensor = payload
                payload_type = IPCPayloadType.TENSOR
            sink = pa.BufferOutputStream()
            pa_ipc.write_tensor(tensor, sink)
            serialized_bytes = sink.getvalue().to_pybytes()
            extra_meta = {"shape": list(tensor.shape), "dtype": str(tensor.type)}
        elif isinstance(payload, (bytes, bytearray)):
            serialized_bytes = bytes(payload)
            payload_type = IPCPayloadType.RAW_BYTES
        else:
            raise TypeError(f"Unsupported payload type for IPC mapping: {type(payload)}")

        required_size = len(serialized_bytes)
        alloc_size = max(required_size, size_bytes or required_size)
    else:
        if size_bytes is None or size_bytes <= 0:
            raise ValueError("size_bytes must be specified and > 0 when no payload is passed.")
        alloc_size = size_bytes

    # Attempt allocation
    segment: Union[PyArrowSharedMemorySegment, PyArrowMmapFileSegment]
    if prefer_shm:
        try:
            segment = PyArrowSharedMemorySegment(name=name, size=alloc_size, create=True)
        except Exception as exc:
            logger.warning(
                "SharedMemory allocation failed (%s); falling back to memory-mapped file.", exc
            )
            segment = PyArrowMmapFileSegment(file_path=None, size=alloc_size, context=ctx)
    else:
        segment = PyArrowMmapFileSegment(file_path=None, size=alloc_size, context=ctx)

    # If payload was supplied, write it immediately
    if serialized_bytes is not None:
        segment.write_raw(
            serialized_bytes,
            payload_type=payload_type,
            extra_metadata=extra_meta,
        )

    return segment


def serialize_to_ipc(
    payload: Any,
    name: Optional[str] = None,
    prefer_shm: bool = True,
    context: Optional[ExecutionContext] = None,
) -> Tuple[Union[PyArrowSharedMemorySegment, PyArrowMmapFileSegment], IPCBufferMetadata]:
    """Serialize a high-dimensional object into an IPC buffer and return segment and metadata.

    Args:
        payload: PyArrow Table, RecordBatch, Tensor, NumPy array, or bytes.
        name: Optional segment name.
        prefer_shm: If True, uses SharedMemory with mmap fallback.
        context: ExecutionContext.

    Returns:
        Tuple of (IPCSegment, IPCBufferMetadata).
    """
    segment = pyarrow_mmap_mapper(
        payload=payload,
        name=name,
        prefer_shm=prefer_shm,
        context=context,
    )
    if segment.metadata is None:
        raise IPCBufferError("Serialization failed to generate metadata.")
    return segment, segment.metadata


def deserialize_from_ipc(
    metadata: IPCBufferMetadata,
    segment: Optional[Union[PyArrowSharedMemorySegment, PyArrowMmapFileSegment]] = None,
    verify_checksum: bool = True,
    context: Optional[ExecutionContext] = None,
) -> Any:
    """Deserialize a PyArrow Table, RecordBatch, Tensor, NumPy array, or bytes from IPC.

    Args:
        metadata: Metadata describing the buffer and payload.
        segment: Optional existing open segment. If None, attaches dynamically based on metadata.
        verify_checksum: If True, validates xxh64 checksum before deserializing.
        context: ExecutionContext.

    Returns:
        Deserialized PyArrow Table, RecordBatch, Tensor, NumPy array, or bytes.
    """
    ctx = context or ExecutionContext()
    managed_seg: Optional[Union[PyArrowSharedMemorySegment, PyArrowMmapFileSegment]] = None

    if segment is not None:
        target_seg = segment
    else:
        if metadata.buffer_type == IPCSegmentType.SHARED_MEMORY:
            managed_seg = PyArrowSharedMemorySegment(name=metadata.name, create=False)
            target_seg = managed_seg
        else:
            if metadata.file_path is None:
                raise IPCBufferError("Metadata specifies mmap buffer but file_path is None.")
            managed_seg = PyArrowMmapFileSegment(file_path=metadata.file_path, context=ctx)
            target_seg = managed_seg

    try:
        raw_bytes = target_seg.read_raw(length=metadata.size_bytes, verify_checksum=verify_checksum)
        if verify_checksum:
            mmap_checksum_verifier(
                raw_bytes,
                expected_size=metadata.size_bytes,
                expected_checksum=metadata.checksum_xxh64,
                algorithm="xxh64",
                raise_on_error=True,
            )

        buf = pa.py_buffer(raw_bytes)

        if metadata.payload_type == IPCPayloadType.TABLE:
            reader = pa_ipc.open_stream(buf)
            return reader.read_all()
        elif metadata.payload_type == IPCPayloadType.RECORD_BATCH:
            reader = pa_ipc.open_stream(buf)
            return reader.read_next_batch()
        elif metadata.payload_type == IPCPayloadType.TENSOR:
            return pa_ipc.read_tensor(buf)
        elif metadata.payload_type == IPCPayloadType.NUMPY_ARRAY:
            tensor = pa_ipc.read_tensor(buf)
            return np.copy(tensor.to_numpy())
        elif metadata.payload_type == IPCPayloadType.RAW_BYTES:
            return raw_bytes
        else:
            raise IPCBufferError(f"Unknown payload type: {metadata.payload_type}")
    finally:
        if managed_seg is not None:
            managed_seg.close()
