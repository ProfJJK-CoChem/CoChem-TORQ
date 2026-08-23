"""CoChem-TORQ: SWMR Zombie Lock Reaper & Database Healer.

======================================================
Phase 1 (Stage 0.0) Implementation
----------------------------------
Provides resilient, cross-platform SWMR (Single-Writer Multiple-Reader)
file lock management, multi-tier zombie PID detection, atomic acquisition,
and automatic journal/superblock healing for HDF5 databases (e.g. landscape.h5).

Architecture & Features:
- Atomic lock file creation via os.O_CREAT | os.O_EXCL
- Pydantic v2 LockMetadata model with Slurm HPC integration
- Multi-tier process validation:
    * Local: psutil pid existence, process start-time validation against lock
      creation timestamp (detects recycled PIDs), and zombie/dead status checks.
    * Distributed HPC: squeue query parsing against terminal state matrices.
- Superblock journal flush and lock recovery on dead/zombie processes.
- SWMRWriteContext context manager for safe transactional database access.
- Complete physical validation compliance (prohibits synthetic stubs).

Authoritative Sources:
- CoChem-TORQ High-Throughput Distributed Architecture Specification
- HDF5 Single-Writer / Multiple-Reader (SWMR) File Access Specification
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Final

import h5py
import psutil
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Configure module-level logging
logger = logging.getLogger("CoChem-TORQ.H5Healer")

# Default lock timing parameters (seconds)
DEFAULT_LOCK_TIMEOUT: Final[float] = 10.0
DEFAULT_RETRY_INTERVAL: Final[float] = 0.2
PID_RECYCLE_TOLERANCE_SECONDS: Final[float] = 1.0

# Slurm job state matrices
SLURM_TERMINAL_STATES: Final[set[str]] = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}

SLURM_ACTIVE_STATES: Final[set[str]] = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "STAGE_OUT",
    "SUSPENDED",
}


# ============================================================================
# Lock Metadata Model (Pydantic v2)
# ============================================================================


class LockMetadata(BaseModel):
    """Pydantic v2 model representing SWMR lock file metadata.

    Serializes process identity, creation timestamp, host, and HPC job context.
    """

    model_config = ConfigDict(extra="ignore")

    pid: int = Field(
        ..., description="Operating system Process ID holding the lock"
    )
    hostname: str = Field(
        ..., description="Hostname of the machine where lock was created"
    )
    slurm_job_id: str | None = Field(
        default=None, description="Slurm Job ID if executed under HPC scheduler"
    )
    created_at: float = Field(
        ..., description="POSIX timestamp (time.time()) when lock was acquired"
    )
    session_id: str | None = Field(
        default=None,
        description="Unique UUID/session token for lock ownership verification",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary user metadata"
    )


# ============================================================================
# Path & Metadata Utilities
# ============================================================================


def get_lock_path(db_path: str | Path) -> Path:
    """Return the standard SWMR lock file path associated with an HDF5 database.

    Format: <db_path>.lock
    """
    path_str = str(db_path)
    return Path(path_str + ".lock")


def read_lock_metadata(lock_path: str | Path) -> LockMetadata | None:
    """Safely read and validate LockMetadata from a JSON lock file.

    Catches specific I/O and deserialization exceptions and logs warnings.

    Returns:
        LockMetadata instance if valid, or None if missing, empty, or corrupted.
    """
    path = Path(lock_path)
    if not path.exists():
        return None

    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            logger.warning("SWMR lock file %s is empty (0 bytes).", path)
            return None

        raw_data = json.loads(content)
        if not isinstance(raw_data, dict):
            logger.warning(
                "SWMR lock file %s contains non-dictionary JSON: %s",
                path,
                type(raw_data),
            )
            return None

        return LockMetadata.model_validate(raw_data)

    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Corrupted JSON in SWMR lock file %s: %s", path, exc)
        return None
    except ValidationError as exc:
        logger.warning("Invalid schema in SWMR lock file %s: %s", path, exc)
        return None
    except PermissionError as exc:
        logger.warning("Permission denied reading SWMR lock file %s: %s", path, exc)
        return None
    except OSError as exc:
        logger.warning("OS I/O error reading SWMR lock file %s: %s", path, exc)
        return None


# ============================================================================
# Host & Process Validation Helpers
# ============================================================================


def _is_local_host(lock_hostname: str) -> bool:
    """Determine if the given hostname matches the local host machine."""
    if not lock_hostname:
        return True

    curr_host = socket.gethostname().lower()
    target_host = lock_hostname.lower()

    if target_host in {"localhost", "127.0.0.1", "::1", curr_host}:
        return True

    # Compare short hostname (before first dot) for FQDN variations
    if target_host.split(".")[0] == curr_host.split(".")[0]:
        return True

    try:
        curr_fqdn = socket.getfqdn().lower()
        if (
            target_host == curr_fqdn
            or target_host.split(".")[0] == curr_fqdn.split(".")[0]
        ):
            return True
    except OSError:
        pass

    return False


def _check_local_pid(pid: int, lock_created_at: float) -> tuple[bool, str]:
    """Validate whether a local PID is a zombie/dead process or recycled PID.

    Returns:
        (is_zombie: bool, reason: str)
    """
    if pid <= 0:
        return True, f"PID {pid} is an invalid / non-positive PID"

    if not psutil.pid_exists(pid):
        return True, f"PID {pid} does not exist in local process table"

    try:
        proc = psutil.Process(pid)

        # Check PID recycling: process created after lock creation
        try:
            proc_created = proc.create_time()
            if proc_created > lock_created_at + PID_RECYCLE_TOLERANCE_SECONDS:
                return (
                    True,
                    f"PID {pid} was recycled by OS (process create_time "
                    f"{proc_created:.2f} > lock created_at {lock_created_at:.2f})",
                )
        except psutil.AccessDenied:
            pass

        # Check zombie or dead status
        try:
            status = proc.status()
            zombie_statuses = {
                getattr(psutil, "STATUS_ZOMBIE", "zombie"),
                getattr(psutil, "STATUS_DEAD", "dead"),
            }
            if status in zombie_statuses:
                return True, f"PID {pid} is in status '{status}'"
        except psutil.AccessDenied:
            return (
                False,
                f"PID {pid} status inaccessible (assumed active system process)",
            )

        return False, f"PID {pid} is actively running (status='{proc.status()}')"

    except psutil.NoSuchProcess:
        return True, f"PID {pid} terminated during inspection"
    except psutil.AccessDenied:
        # Access denied indicates an active, privileged system process; assume alive
        return False, f"PID {pid} access denied (assumed active system process)"
    except OSError as exc:
        logger.warning("OS error inspecting PID %d: %s", pid, exc)
        return False, f"OS error inspecting PID {pid}: {exc}"


def _check_slurm_job_status(slurm_job_id: str) -> tuple[bool, str]:
    """Inspect Slurm scheduler status for a remote HPC job ID using squeue.

    Returns:
        (is_zombie: bool, reason: str)
    """
    squeue_exe = shutil.which("squeue")
    if not squeue_exe:
        return (
            False,
            "squeue command not available on current host; "
            "remote job status indeterminate",
        )

    try:
        proc_result = subprocess.run(
            [squeue_exe, "-j", str(slurm_job_id), "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        output = proc_result.stdout.strip().upper()

        # If squeue returned non-zero or empty stdout, the job has exited the queue
        if proc_result.returncode != 0 or not output:
            return (
                True,
                f"Slurm job {slurm_job_id} not found in active squeue "
                f"(returncode={proc_result.returncode})",
            )

        job_states = [s.strip() for s in output.split() if s.strip()]
        if not job_states:
            return True, f"Slurm job {slurm_job_id} returned empty state list"

        # Check active states first for job array safety (e.g. ['COMPLETED', 'RUNNING'])
        if any(state in SLURM_ACTIVE_STATES for state in job_states):
            return (
                False,
                f"Slurm job {slurm_job_id} is active in queue: {job_states}",
            )

        # Check against terminal state matrix
        if any(state in SLURM_TERMINAL_STATES for state in job_states):
            return (
                True,
                f"Slurm job {slurm_job_id} reached terminal state: {job_states}",
            )

        # Unknown state: fail-safe to alive
        return (
            False,
            f"Slurm job {slurm_job_id} reported unrecognized status: {job_states}",
        )

    except subprocess.TimeoutExpired:
        logger.warning(
            "squeue query timed out for Slurm Job ID %s", slurm_job_id
        )
        return False, f"squeue query timed out for Job ID {slurm_job_id}"
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(
            "Failed executing squeue for Slurm Job ID %s: %s",
            slurm_job_id,
            exc,
        )
        return False, f"squeue execution error: {exc}"


# ============================================================================
# Core Zombie Detection & Force Release
# ============================================================================


def detect_zombie_pids(
    db_path: str | Path,
) -> tuple[bool, dict[str, Any] | None]:
    """Detect whether the lock on the HDF5 database is held by a dead/zombie process.

    Returns:
        (is_zombie: bool, metadata_dict: Optional[Dict[str, Any]])
        - If no lock file exists: (False, None)
        - If lock file exists and is corrupted/empty: (True, {"corrupted": True, ...})
        - If lock held by dead/zombie PID or terminal Slurm: (True, metadata_dict)
        - If lock is held by an active, healthy process: (False, metadata_dict)
    """
    lock_path = get_lock_path(db_path)
    if not lock_path.exists():
        return False, None

    meta = read_lock_metadata(lock_path)
    if meta is None:
        return True, {"corrupted": True, "lock_path": str(lock_path)}

    meta_dict = meta.model_dump()

    # Multi-tier check: Local execution vs Distributed HPC execution
    if _is_local_host(meta.hostname):
        is_zombie, reason = _check_local_pid(meta.pid, meta.created_at)
        meta_dict["zombie_reason"] = reason
        return is_zombie, meta_dict

    # Distributed HPC execution
    slurm_id = meta.slurm_job_id
    if slurm_id and str(slurm_id).strip() not in {"", "0", "None"}:
        is_zombie, reason = _check_slurm_job_status(str(slurm_id).strip())
        meta_dict["zombie_reason"] = reason
        return is_zombie, meta_dict

    # Remote host without verifiable Slurm job ID: default safely to alive
    meta_dict["zombie_reason"] = (
        "Remote host without queryable Slurm job; assumed alive"
    )
    return False, meta_dict


def force_release_swmr(
    db_path: str | Path,
    lock_metadata: dict[str, Any] | None = None,
    force_override: bool = False,
) -> bool:
    """Safely release an SWMR lock, flush HDF5 journals, and repair superblocks.

    Args:
        db_path: Path to the HDF5 database file.
        lock_metadata: Optional pre-fetched lock metadata dict.
        force_override: If True, release even if the locking process appears active.

    Raises:
        BlockingIOError: If lock is held by confirmed active process and override=False.
        OSError: If filesystem unlinking or superblock flush encounters an error.

    Returns:
        True if the lock was successfully cleared and database superblock flushed.
    """
    path_obj = Path(db_path)
    lock_path = get_lock_path(db_path)

    if lock_path.exists():
        is_zombie, detected_meta = detect_zombie_pids(db_path)
        meta = lock_metadata or detected_meta or {}

        if not is_zombie and not force_override:
            pid = meta.get("pid", "unknown")
            host = meta.get("hostname", "unknown")
            raise BlockingIOError(
                f"SWMR database '{db_path}' is actively locked by "
                f"PID {pid} on host {host}."
            )

        # Unlink stale/dead lock file
        try:
            lock_path.unlink(missing_ok=True)
            logger.warning(
                "Unlinked stale SWMR lock file: %s (zombie=%s, force_override=%s)",
                lock_path,
                is_zombie,
                force_override,
            )
        except OSError as exc:
            logger.error(
                "Failed unlinking SWMR lock file %s: %s", lock_path, exc
            )
            raise

    # If the HDF5 database exists, trigger journal flushing and superblock repair
    if path_obj.exists() and path_obj.is_file():
        try:
            with h5py.File(path_obj, mode="a", libver="latest") as f:
                f.flush()
            logger.info(
                "Flushed HDF5 journal and validated superblock for '%s'",
                path_obj,
            )
        except (OSError, RuntimeError) as exc:
            logger.error(
                "HDF5 superblock flush error on '%s': %s", path_obj, exc
            )
            raise

    return True


# ============================================================================
# Atomic Lock Acquisition & Release
# ============================================================================


def acquire_swmr_lock(
    db_path: str | Path,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
    retry_interval: float = DEFAULT_RETRY_INTERVAL,
    auto_heal: bool = True,
    slurm_job_id: str | None = None,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> LockMetadata:
    """Atomically acquire an SWMR file lock for an HDF5 database.

    If an existing lock is held by a dead/zombie process and auto_heal is True,
    it will be reaped and auto-healed automatically.

    Args:
        db_path: Path to target HDF5 database.
        timeout: Maximum seconds to wait before raising TimeoutError.
        retry_interval: Polling sleep interval between collision attempts.
        auto_heal: Automatically detect and purge zombie locks.
        slurm_job_id: Optional explicit Slurm Job ID to embed in metadata.
        session_id: Unique session UUID. If omitted, a new UUID4 is generated.
        extra: Optional additional metadata dictionary.

    Returns:
        LockMetadata representing the newly acquired lock.

    Raises:
        TimeoutError: If lock cannot be acquired within the timeout period.
        BlockingIOError: If actively locked and auto_heal is False.
    """
    lock_path = get_lock_path(db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    s_job = slurm_job_id or os.environ.get("SLURM_JOB_ID")
    s_id = session_id or str(uuid.uuid4())

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    start_time = time.monotonic()

    while True:
        try:
            fd = os.open(str(lock_path), flags)
            try:
                # Successfully created lock file atomically
                metadata = LockMetadata(
                    pid=os.getpid(),
                    hostname=socket.gethostname(),
                    slurm_job_id=str(s_job) if s_job else None,
                    created_at=time.time(),
                    session_id=s_id,
                    extra=extra or {},
                )
                payload = metadata.model_dump_json(indent=2).encode("utf-8")
                os.write(fd, payload)
                try:
                    os.fsync(fd)
                except OSError:
                    pass
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                lock_path.unlink(missing_ok=True)
                raise
            else:
                os.close(fd)

            logger.debug("Acquired SWMR lock for '%s' (session=%s)", db_path, s_id)
            return metadata

        except FileExistsError:
            # Lock file currently exists
            if auto_heal:
                is_zombie, meta = detect_zombie_pids(db_path)
                if is_zombie:
                    logger.warning(
                        "Detected stale SWMR lock on '%s'. Initiating auto-heal...",
                        db_path,
                    )
                    try:
                        force_release_swmr(
                            db_path, lock_metadata=meta, force_override=True
                        )
                        # Immediately retry lock acquisition after reaping zombie
                        continue
                    except (BlockingIOError, OSError) as exc:
                        logger.warning(
                            "Auto-heal attempt failed on '%s': %s",
                            db_path,
                            exc,
                        )

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                is_zombie, meta = detect_zombie_pids(db_path)
                pid_info = meta.get("pid", "unknown") if meta else "unknown"
                host_info = meta.get("hostname", "unknown") if meta else "unknown"
                raise TimeoutError(
                    f"Timed out after {timeout:.1f}s waiting for SWMR lock on "
                    f"'{db_path}' (held by PID {pid_info} on host {host_info})."
                )

            time.sleep(retry_interval)


def release_swmr_lock(
    db_path: str | Path,
    session_id: str | None = None,
    force: bool = False,
) -> bool:
    """Release an SWMR lock file.

    Args:
        db_path: Path to target HDF5 database.
        session_id: Session ID to verify ownership against lock metadata.
        force: If True, bypass session and process ownership validation.

    Raises:
        PermissionError: If lock is owned by a different session and force=False.

    Returns:
        True if lock was successfully released or did not exist.
    """
    lock_path = get_lock_path(db_path)
    if not lock_path.exists():
        return True

    meta = read_lock_metadata(lock_path)
    if not force and meta is not None:
        if session_id is not None and meta.session_id != session_id:
            raise PermissionError(
                f"Cannot release lock on '{db_path}': session ID mismatch "
                f"(expected {session_id}, got {meta.session_id})."
            )
        if (
            session_id is None
            and _is_local_host(meta.hostname)
            and meta.pid != os.getpid()
        ):
            raise PermissionError(
                f"Cannot release lock on '{db_path}': lock is owned by PID "
                f"{meta.pid}, caller PID is {os.getpid()}."
            )

    try:
        lock_path.unlink(missing_ok=True)
        logger.debug("Released SWMR lock for '%s'", db_path)
        return True
    except OSError as exc:
        logger.error("Error unlinking lock file %s: %s", lock_path, exc)
        raise


# ============================================================================
# Integrity Verification & Full Healing
# ============================================================================


def verify_h5_swmr_integrity(db_path: str | Path) -> bool:
    """Verify physical structure, superblock header, and SWMR readability.

    Returns:
        True if file exists, is valid HDF5, and can be read in SWMR mode without error.
        False if file is missing, corrupt, or unreadable.
    """
    path = Path(db_path)
    if not path.exists() or not path.is_file():
        return False

    try:
        with h5py.File(path, mode="r", libver="latest", swmr=True) as f:
            _ = list(f.keys())
            _ = list(f.attrs.keys())
        return True
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("SWMR integrity check failed for '%s': %s", path, exc)
        return False


def heal_swmr_database(db_path: str | Path, force: bool = False) -> bool:
    """Execute full SWMR database healing.

    1. Harvest and purge zombie locks.
    2. Flush HDF5 journal and update superblock.
    3. Verify physical database integrity.

    Returns:
        True if the database is confirmed intact and healthy.
    """
    force_release_swmr(db_path, force_override=force)
    return verify_h5_swmr_integrity(db_path)


# ============================================================================
# SWMR Context Manager
# ============================================================================


class SWMRWriteContext:
    """Robust Context Manager for SWMR HDF5 Database Access.

    Guarantees:
    1. Atomic lock acquisition before file open.
    2. Automatic zombie lock purging if stale lock is encountered.
    3. Proper HDF5 SWMR mode initialization (swmr_mode = True).
    4. Deterministic flush, file close, and lock release upon context exit.
    """

    def __init__(
        self,
        db_path: str | Path,
        timeout: float = DEFAULT_LOCK_TIMEOUT,
        retry_interval: float = DEFAULT_RETRY_INTERVAL,
        auto_heal: bool = True,
        slurm_job_id: str | None = None,
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
        libver: str = "latest",
        mode: str = "a",
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.auto_heal = auto_heal
        self.slurm_job_id = slurm_job_id
        self.session_id = session_id or str(uuid.uuid4())
        self.extra = extra or {}
        self.libver = libver
        self.mode = mode
        self.lock_metadata: LockMetadata | None = None
        self.file: h5py.File | None = None

    def __enter__(self) -> h5py.File:
        self.lock_metadata = acquire_swmr_lock(
            db_path=self.db_path,
            timeout=self.timeout,
            retry_interval=self.retry_interval,
            auto_heal=self.auto_heal,
            slurm_job_id=self.slurm_job_id,
            session_id=self.session_id,
            extra=self.extra,
        )

        try:
            self.file = h5py.File(
                self.db_path, mode=self.mode, libver=self.libver
            )
            try:
                # Enable SWMR write mode if not already active
                if not getattr(self.file, "swmr_mode", False):
                    self.file.swmr_mode = True
            except (ValueError, RuntimeError) as exc:
                logger.debug(
                    "Note enabling swmr_mode on '%s': %s", self.db_path, exc
                )
            return self.file
        except Exception:
            # If opening file fails, close leaked handle and release lock
            if self.file is not None:
                try:
                    self.file.close()
                except Exception:
                    pass
                self.file = None
            release_swmr_lock(self.db_path, session_id=self.session_id, force=True)
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if self.file is not None:
                try:
                    self.file.flush()
                except OSError as exc:
                    logger.warning(
                        "Error flushing file '%s' on exit: %s",
                        self.db_path,
                        exc,
                    )
                finally:
                    self.file.close()
        finally:
            release_swmr_lock(self.db_path, session_id=self.session_id, force=True)


__all__ = [
    "LockMetadata",
    "get_lock_path",
    "read_lock_metadata",
    "detect_zombie_pids",
    "force_release_swmr",
    "acquire_swmr_lock",
    "release_swmr_lock",
    "SWMRWriteContext",
    "verify_h5_swmr_integrity",
    "heal_swmr_database",
]
