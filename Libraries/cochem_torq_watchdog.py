"""CoChem-TORQ: The Step-Back Recovery Guard & Hardware Watchdog.

=============================================================================
Phase 5 (Stage 4.0 / Task 8) Implementation
-------------------------------------------
Implements an asynchronous OS-level watchdog and recovery engine designed to
prevent High-Performance Computing (HPC) and local workstation crashes during
multi-day, multi-dimensional torsional potential energy surface (PES) sweeps.

Core Directives & Capabilities:
1. Grid-Collapse Step-Back:
   Live telemetry monitoring for Self-Consistent Field (SCF) divergence,
   severe linear dependence, and gradient stagnation. Safely terminates hanging
   engine processes, widens spatial torsional grid steps, injects damped
   convergence directives (SlowConv, Damp, SOSCF, Shift, GridX), and produces
   autonomous restart payloads.

2. Dynamic Memory Backoff:
   Continuous host RAM monitoring via psutil. If memory pressure exceeds the
   configured threshold (e.g. >90%), gracefully halts execution, scales down
   %maxcore allocations by 30%, purges bloated scratch matrices (.tmp, .mat),
   and prepares a restart payload.

3. CUDA Memory Leak Guard:
   VRAM monitoring and proactive cache clearing (CuPy memory pools, PyTorch
   CUDA allocator, garbage collection) to maintain a flat VRAM baseline across
   multi-day sweeps.

4. Safe Process Tree Teardown & Zombie MPI Reaping:
   Recursive process tree discovery and teardown (SIGTERM -> SIGKILL) without
   leaving orphaned OpenMPI or quantum chemistry daemons. Automatic cleanup
   registration via atexit.

5. Tripartite Filesystem Air-Gap Compliance:
   Strict domain isolation between Domain A (Ring 1 Static Repo, read-only),
   Domain B (Ring 2 Ephemeral Scratch), and Domain C (Ring 3 Persistent Artifacts).
   Runtime writes directly to Domain A raise AirGapViolationError.

6. Mendeleev Dynamic Integration:
   Zero-tolerance for hardcoded atomic weights; dynamic periodic property
   querying via the Mendeleev library.
"""

from __future__ import annotations

import atexit
import datetime
import enum
import gc
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Union

from mendeleev import element as mendeleev_element
import psutil
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Configure module-level logging
logger = logging.getLogger("CoChem-TORQ.Watchdog")

# Optional PyTorch check
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Optional CuPy check
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

# Optional pynvml check
try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


# =============================================================================
# 1. Custom Exceptions
# =============================================================================


class WatchdogError(Exception):
    """Base exception for CoChem-TORQ watchdog and recovery operations."""

    pass


class AirGapViolationError(PermissionError):
    """Raised when an operation attempts to write runtime data to Ring 1 static repository space."""

    pass


class ProcessTeardownError(WatchdogError):
    """Raised when a process tree fails to terminate within the allocated timeout window."""

    pass


class StepBackRecoveryError(WatchdogError):
    """Raised when grid collapse step-back recovery cannot construct a valid recovery payload."""

    pass


class MemoryBackoffError(WatchdogError):
    """Raised when memory backoff exceeds permissible minimum resource boundaries."""

    pass


class CudaLeakGuardError(WatchdogError):
    """Raised when CUDA memory cleanup operations encounter an unrecoverable failure."""

    pass


# =============================================================================
# 2. Tripartite Filesystem Air-Gap Architecture
# =============================================================================


def get_repo_root() -> Path:
    """Locate the Domain A / Ring 1 immutable Git repository root."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent.resolve()

    env_val = os.environ.get("COCHEM_REPO_DIR")
    if env_val:
        repo_path = Path(env_val).resolve()
        if repo_path.is_dir():
            return repo_path

    return Path.cwd().resolve()


def get_artifacts_dir() -> Path:
    """Return the dynamically resolved persistent artifacts directory (Domain C / Ring 3)."""
    env_dir = os.environ.get("COCHEM_ARTIFACTS_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
    else:
        p = Path.home() / "cochem_artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_scratch_dir() -> Path:
    """Return the dynamically resolved ephemeral scratch directory (Domain B / Ring 2)."""
    env_dir = os.environ.get("COCHEM_SCRATCH_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
    else:
        system_temp = tempfile.gettempdir()
        p = Path(system_temp) / "cochem_scratch"
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_runtime_write_path(target_path: Union[str, Path]) -> Path:
    """Verify that runtime output paths strictly respect Tripartite Air-Gap boundaries.

    Parameters
    ----------
    target_path : Union[str, Path]
        The destination filesystem path to evaluate.

    Returns
    -------
    Path
        The resolved absolute path if validated.

    Raises
    ------
    AirGapViolationError
        If the target path points inside Domain A (Ring 1 static repository space).
    """
    resolved_path = Path(target_path).resolve()
    repo_root = get_repo_root()

    try:
        resolved_path.relative_to(repo_root)
        raise AirGapViolationError(
            f"Air-Gap Boundary Violation: Attempted runtime write to Ring 1 static repository space: {resolved_path}"
        )
    except ValueError:
        pass

    return resolved_path


# =============================================================================
# 3. Mendeleev Dynamic Integration
# =============================================================================


def get_atomic_mass(element: Union[str, int]) -> float:
    """Retrieve standard atomic weight dynamically via Mendeleev library.

    Parameters
    ----------
    element : Union[str, int]
        Chemical element symbol or atomic number.

    Returns
    -------
    float
        Standard atomic mass in atomic mass units (Da).
    """
    if isinstance(element, int):
        elem_obj = mendeleev_element(element)
    else:
        elem_obj = mendeleev_element(str(element).strip().capitalize())
    return float(elem_obj.atomic_weight)


def get_isotopic_mass(element: Union[str, int], mass_number: Optional[int] = None) -> float:
    """Retrieve exact isotopic mass dynamically via Mendeleev library.

    Parameters
    ----------
    element : Union[str, int]
        Chemical element symbol or atomic number.
    mass_number : Optional[int]
        Specific isotopic mass number (A). If None, most abundant isotope is chosen.

    Returns
    -------
    float
        Exact isotopic mass in atomic mass units (Da).
    """
    if isinstance(element, int):
        elem_obj = mendeleev_element(element)
    else:
        elem_obj = mendeleev_element(str(element).strip().capitalize())

    isotopes = elem_obj.isotopes
    if not isotopes:
        return float(elem_obj.atomic_weight)

    if mass_number is not None:
        for iso in isotopes:
            if iso.mass_number == mass_number:
                return float(iso.mass)
        raise ValueError(f"Isotope A={mass_number} not found for element {elem_obj.symbol}")

    abundant = max(isotopes, key=lambda x: (x.abundance if x.abundance is not None else 0.0))
    return float(abundant.mass if abundant.mass is not None else elem_obj.atomic_weight)


def get_element_symbol(element: Union[str, int]) -> str:
    """Retrieve canonical element symbol dynamically via Mendeleev library."""
    if isinstance(element, int):
        elem_obj = mendeleev_element(element)
    else:
        elem_obj = mendeleev_element(str(element).strip().capitalize())
    return str(elem_obj.symbol)


def get_atomic_number(element: Union[str, int]) -> int:
    """Retrieve atomic number (Z) dynamically via Mendeleev library."""
    if isinstance(element, int):
        elem_obj = mendeleev_element(element)
    else:
        elem_obj = mendeleev_element(str(element).strip().capitalize())
    return int(elem_obj.atomic_number)


# =============================================================================
# 4. Enums and Pydantic v2 Models
# =============================================================================


class RecoveryActionType(str, enum.Enum):
    """Enumeration of autonomous recovery actions executed by the watchdog."""

    NO_ACTION = "NO_ACTION"
    GRID_COLLAPSE_STEP_BACK = "GRID_COLLAPSE_STEP_BACK"
    MEMORY_BACKOFF = "MEMORY_BACKOFF"
    CUDA_CACHE_CLEAR = "CUDA_CACHE_CLEAR"
    PROCESS_TEARDOWN = "PROCESS_TEARDOWN"
    RESTART_CALCULATION = "RESTART_CALCULATION"
    ABORT = "ABORT"


# Alias for backward/forward compatibility
RecoveryAction = RecoveryActionType


class FailureMode(str, enum.Enum):
    """Enumeration of quantum chemical and system hardware failure modes."""

    NONE = "NONE"
    SCF_DIVERGENCE = "SCF_DIVERGENCE"
    LINEAR_DEPENDENCE = "LINEAR_DEPENDENCE"
    GRADIENT_STAGNATION = "GRADIENT_STAGNATION"
    RAM_EXHAUSTION = "RAM_EXHAUSTION"
    VRAM_EXHAUSTION = "VRAM_EXHAUSTION"
    PROCESS_HANG = "PROCESS_HANG"
    ORPHANED_PROCESS = "ORPHANED_PROCESS"


class WatchdogConfig(BaseModel):
    """Configuration parameters for the CoChem-TORQ Watchdog & Step-Back Guard."""

    ram_threshold_pct: float = Field(
        default=90.0,
        ge=10.0,
        le=100.0,
        description="Host RAM utilization percentage threshold to trigger dynamic memory backoff.",
    )
    vram_threshold_pct: float = Field(
        default=85.0,
        ge=10.0,
        le=100.0,
        description="GPU VRAM utilization percentage threshold to trigger CUDA memory leak purge.",
    )
    grid_step_shift_deg: float = Field(
        default=3.0,
        gt=0.0,
        description="Torsional coordinate angular shift (degrees) applied during grid-collapse recovery.",
    )
    maxcore_reduction_factor: float = Field(
        default=0.7,
        gt=0.1,
        lt=1.0,
        description="Multiplicative scaling factor applied to %maxcore during memory backoff (e.g. 0.7 = 30% reduction).",
    )
    min_maxcore_mb: int = Field(
        default=500,
        ge=100,
        description="Lower bound on %maxcore memory in MB below which further reduction is prohibited.",
    )
    process_teardown_timeout_sec: float = Field(
        default=3.0,
        gt=0.0,
        description="Timeout in seconds before escalating SIGTERM to SIGKILL during process tree teardown.",
    )
    poll_interval_sec: float = Field(
        default=0.5,
        gt=0.01,
        description="Telemetry stream and hardware polling interval in seconds.",
    )
    scf_damped_keywords: List[str] = Field(
        default_factory=lambda: ["SlowConv", "Damp", "SOSCF"],
        description="Damped convergence keywords injected into quantum chemical input files upon SCF divergence.",
    )
    scf_grid_fallback: str = Field(
        default="GridX",
        description="Integration grid fallback directive injected upon linear dependence or convergence failure.",
    )
    max_recovery_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum successive recovery attempts per geometry point before aborting.",
    )
    cuda_flush_interval_mb: float = Field(
        default=10.0,
        gt=0.0,
        description="Threshold byte flush interval in MB triggering proactive CUDA and garbage collector cleaning.",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProcessTelemetry(BaseModel):
    """Snapshot of OS process and hardware utilization telemetry."""

    pid: int = Field(..., description="Monitored OS process identifier.")
    cpu_percent: float = Field(default=0.0, description="Process CPU utilization percentage.")
    ram_used_bytes: int = Field(default=0, description="Process resident memory usage in bytes.")
    ram_percent: float = Field(default=0.0, description="Process memory percentage of total system RAM.")
    host_ram_percent: float = Field(default=0.0, description="System-wide total RAM utilization percentage.")
    vram_used_bytes: int = Field(default=0, description="GPU VRAM used in bytes.")
    vram_total_bytes: int = Field(default=0, description="Total GPU VRAM in bytes.")
    vram_percent: float = Field(default=0.0, description="GPU VRAM utilization percentage.")
    num_threads: int = Field(default=0, description="Number of active execution threads.")
    num_children: int = Field(default=0, description="Number of active child processes.")
    is_running: bool = Field(default=True, description="Whether the process is currently active.")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="UTC timestamp of telemetry capture.",
    )

    model_config = ConfigDict(frozen=True)


class RecoveryEvent(BaseModel):
    """Audit record capturing an autonomous recovery action taken by the watchdog."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the recovery event.",
    )
    action: RecoveryActionType = Field(..., description="Specific recovery action executed.")
    failure_mode: FailureMode = Field(..., description="Failure mode that triggered recovery.")
    pid: Optional[int] = Field(default=None, description="PID of the terminated or recovered process.")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="UTC timestamp of the recovery event.",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic metadata and parameter adjustments.",
    )
    success: bool = Field(default=True, description="Whether the recovery intervention succeeded.")
    error_message: Optional[str] = Field(default=None, description="Diagnostic error details if recovery failed.")

    model_config = ConfigDict(frozen=True)


class StepBackResult(BaseModel):
    """Result payload returned by grid-collapse step-back execution."""

    success: bool = Field(..., description="Whether step-back input generation succeeded.")
    new_input_path: Optional[Path] = Field(default=None, description="Filesystem path to generated recovery input file.")
    new_grid_angle_deg: float = Field(..., description="Adjusted torsional grid coordinate in degrees.")
    applied_flags: List[str] = Field(
        default_factory=list,
        description="List of convergence and integration directives injected into input.",
    )
    original_pid: Optional[int] = Field(default=None, description="Terminated engine process PID.")
    event: Optional[RecoveryEvent] = Field(default=None, description="Associated recovery event audit record.")

    model_config = ConfigDict(frozen=True)


class MemoryBackoffResult(BaseModel):
    """Result payload returned by dynamic memory backoff execution."""

    success: bool = Field(..., description="Whether memory backoff and scratch purge succeeded.")
    previous_maxcore_mb: int = Field(..., description="Prior %maxcore allocation in MB.")
    new_maxcore_mb: int = Field(..., description="Scaled-down %maxcore allocation in MB.")
    purged_scratch_files: List[Path] = Field(
        default_factory=list,
        description="List of purged scratch temporary matrix files.",
    )
    purged_bytes: int = Field(default=0, description="Total bytes freed across purged scratch files.")
    new_input_path: Optional[Path] = Field(default=None, description="Filesystem path to updated input file.")
    event: Optional[RecoveryEvent] = Field(default=None, description="Associated recovery event audit record.")

    model_config = ConfigDict(frozen=True)


class CudaGuardResult(BaseModel):
    """Result payload returned by CUDA memory leak cleanup."""

    vram_percent_before: float = Field(..., description="VRAM utilization percentage before cleanup.")
    vram_percent_after: float = Field(..., description="VRAM utilization percentage after cleanup.")
    freed_bytes: int = Field(default=0, description="Estimated byte reduction across GPU memory pools.")
    cupy_freed: bool = Field(default=False, description="Whether CuPy memory pools were cleared.")
    torch_freed: bool = Field(default=False, description="Whether PyTorch CUDA cache was cleared.")
    gc_collected_count: int = Field(default=0, description="Number of unreachable objects collected by gc.")
    event: Optional[RecoveryEvent] = Field(default=None, description="Associated recovery event audit record.")

    model_config = ConfigDict(frozen=True)


class WatchdogStatus(BaseModel):
    """Overall status and telemetry log of the watchdog monitoring session."""

    active: bool = Field(default=False, description="Whether the watchdog daemon is currently monitoring.")
    monitored_pid: Optional[int] = Field(default=None, description="Currently tracked root engine PID.")
    events: List[RecoveryEvent] = Field(default_factory=list, description="Historical sequence of recovery events.")
    telemetry_history: List[ProcessTelemetry] = Field(
        default_factory=list,
        description="Historical sequence of captured process telemetry snapshots.",
    )
    active_maxcore_mb: Optional[int] = Field(default=None, description="Current effective %maxcore memory in MB.")
    active_grid_angle_deg: Optional[float] = Field(default=None, description="Current torsional coordinate in degrees.")
    restart_count: int = Field(default=0, description="Number of restarts executed during current session.")

    model_config = ConfigDict(frozen=True)


# =============================================================================
# 5. Safe Process Tree Teardown & Zombie MPI Reaper
# =============================================================================


def safe_process_tree_teardown(
    pid: int,
    timeout: float = 3.0,
    sig: int = signal.SIGTERM,
) -> List[int]:
    """Recursively discover and gracefully terminate an entire process subtree.

    Progresses from SIGTERM to SIGKILL if processes do not exit within the
    timeout window. Reaps zombie process handles on Windows and POSIX systems.

    Parameters
    ----------
    pid : int
        The root process identifier to terminate alongside all descendents.
    timeout : float
        Grace period in seconds to wait for voluntary termination before SIGKILL.
    sig : int
        Initial termination signal (defaults to SIGTERM).

    Returns
    -------
    List[int]
        List of all terminated process IDs (including children and parent).
    """
    terminated_pids: List[int] = []

    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        logger.debug(f"Process PID {pid} is already terminated or nonexistent.")
        return terminated_pids
    except psutil.AccessDenied as e:
        logger.warning(f"Access denied attempting to inspect process PID {pid}: {e}")
        return terminated_pids

    try:
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []

    all_procs = children + [parent]

    # Phase 1: Send initial graceful termination signal
    for proc in all_procs:
        try:
            proc.send_signal(sig)
            terminated_pids.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
        except psutil.AccessDenied as e:
            logger.warning(f"Access denied sending signal {sig} to PID {proc.pid}: {e}")

    # Phase 2: Wait for processes to exit
    gone, alive = psutil.wait_procs(all_procs, timeout=timeout)

    # Phase 3: Forcefully terminate remaining processes via SIGKILL / kill()
    if alive:
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            except psutil.AccessDenied as e:
                logger.warning(f"Access denied forcefully killing PID {proc.pid}: {e}")

        # Final wait
        psutil.wait_procs(alive, timeout=1.0)

    # Phase 4: Reap zombie handles
    for proc in all_procs:
        try:
            if hasattr(proc, "wait"):
                proc.wait(timeout=0)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied):
            pass

    return list(dict.fromkeys(terminated_pids))


def zombie_mpi_reaper(pgid: Optional[int] = None) -> List[int]:
    """Scan and aggressively terminate orphaned OpenMPI, MPICH, and quantum chemistry binaries.

    If a Process Group ID (PGID) is provided on POSIX systems, sends SIGKILL to
    the entire process group. Also scans system processes for orphaned quantum
    chemistry workers whose parents have terminated.

    Parameters
    ----------
    pgid : Optional[int]
        Optional process group ID to terminate via os.killpg.

    Returns
    -------
    List[int]
        List of terminated orphaned process PIDs.
    """
    reaped_pids: List[int] = []

    # 1. Kill process group if PGID provided on POSIX
    if pgid is not None and hasattr(os, "killpg"):
        try:
            sig_to_send = getattr(signal, "SIGKILL", getattr(signal, "SIGTERM", 9))
            os.killpg(pgid, sig_to_send)
            reaped_pids.append(pgid)
            logger.info(f"Terminated process group PGID {pgid} via os.killpg.")
        except (ProcessLookupError, PermissionError) as e:
            logger.debug(f"Process group PGID {pgid} termination notice: {e}")

    # 2. Targeted scanning for orphaned quantum chemistry and MPI workers
    known_worker_patterns = {
        "orca", "orca_scf", "orca_gto", "orca_mdci", "orca_casscf",
        "mpirun", "mpiexec", "orterun", "hydra_pmi_proxy", "pmi_proxy",
        "xtb", "crest", "qcxms", "pyscf",
    }

    try:
        current_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline", "ppid"]):
            try:
                pinfo = proc.info
                p_pid = pinfo.get("pid")
                p_name = str(pinfo.get("name") or "").lower()
                p_cmd = " ".join(pinfo.get("cmdline") or []).lower()
                p_ppid = pinfo.get("ppid")

                if p_pid == current_pid:
                    continue

                is_target_binary = any(kw in p_name or kw in p_cmd for kw in known_worker_patterns)
                if is_target_binary:
                    # Check if parent is dead / init (ppid == 1 or NoSuchProcess)
                    parent_dead = False
                    if p_ppid is None or p_ppid == 1:
                        parent_dead = True
                    else:
                        try:
                            parent_proc = psutil.Process(p_ppid)
                            if not parent_proc.is_running():
                                parent_dead = True
                        except (psutil.NoSuchProcess, psutil.ZombieProcess):
                            parent_dead = True

                    if parent_dead:
                        try:
                            proc.kill()
                            reaped_pids.append(proc.pid)
                            logger.info(f"Reaped orphaned quantum chemistry/MPI worker PID {proc.pid} ({p_name}).")
                        except (psutil.NoSuchProcess, psutil.ZombieProcess):
                            pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.debug(f"Notice during zombie process scanning: {e}")

    return reaped_pids


# Register automatic zombie MPI reaper at Python interpreter exit
atexit.register(zombie_mpi_reaper)


# =============================================================================
# 6. Telemetry & Log Stream Monitoring Utilities
# =============================================================================


SCF_FAILURE_PATTERNS: Final[List[Tuple[re.Pattern[str], FailureMode, str]]] = [
    (
        re.compile(r"SCF NOT CONVERGED|SCF cycle did not converge|SCF failed to converge", re.IGNORECASE),
        FailureMode.SCF_DIVERGENCE,
        "SCF failed to achieve convergence within maximum iteration limit.",
    ),
    (
        re.compile(r"LINEAR DEPENDENCE|NEAR LINEAR DEPENDENCE|Basis set linear dependence|OVERLAP MATRIX ILL-CONDITIONED", re.IGNORECASE),
        FailureMode.LINEAR_DEPENDENCE,
        "Severe basis set linear dependence detected in overlap metric.",
    ),
    (
        re.compile(r"GEOMETRY OPTIMIZATION FAILED|Gradient norm exceeded|Step size too large|Stagnation in step size", re.IGNORECASE),
        FailureMode.GRADIENT_STAGNATION,
        "Geometric step stagnation or gradient explosion encountered.",
    ),
    (
        re.compile(r"OUT OF MEMORY|bad_alloc|Cannot allocate memory|insufficient memory", re.IGNORECASE),
        FailureMode.RAM_EXHAUSTION,
        "Quantum chemistry engine reported host Out-Of-Memory failure.",
    ),
]


def scan_log_for_scf_failure(log_content: str) -> Tuple[bool, FailureMode, str]:
    """Scan execution log text for quantum chemical convergence failures and numerical anomalies.

    Parameters
    ----------
    log_content : str
        The raw text content of the quantum chemistry log or output stream.

    Returns
    -------
    Tuple[bool, FailureMode, str]
        (has_failed, failure_mode, detail_message)
    """
    for pattern, failure_mode, description in SCF_FAILURE_PATTERNS:
        if pattern.search(log_content):
            return True, failure_mode, description

    return False, FailureMode.NONE, ""


def tail_log_file(file_path: Union[str, Path], n_lines: int = 100) -> List[str]:
    """Retrieve the trailing N lines from a monitored execution log file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the log file.
    n_lines : int
        Number of trailing lines to return.

    Returns
    -------
    List[str]
        List of trailing line strings.
    """
    p = Path(file_path)
    if not p.is_file():
        return []

    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return [line.rstrip("\r\n") for line in lines[-n_lines:]]
    except OSError as e:
        logger.warning(f"Failed to tail log file {p}: {e}")
        return []


def collect_process_telemetry(pid: int) -> ProcessTelemetry:
    """Capture a comprehensive hardware and process telemetry snapshot.

    Parameters
    ----------
    pid : int
        The target process identifier.

    Returns
    -------
    ProcessTelemetry
        Pydantic model containing CPU, RAM, VRAM, thread, and process status metrics.
    """
    try:
        proc = psutil.Process(pid)
        is_running = proc.is_running()
        cpu_pct = proc.cpu_percent(interval=None)
        mem_info = proc.memory_info()
        ram_bytes = mem_info.rss
        ram_pct = proc.memory_percent()
        num_threads = proc.num_threads()
        num_children = len(proc.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        is_running = False
        cpu_pct = 0.0
        ram_bytes = 0
        ram_pct = 0.0
        num_threads = 0
        num_children = 0

    host_ram_pct = psutil.virtual_memory().percent

    # Query GPU VRAM metrics if available
    vram_used = 0
    vram_total = 0
    vram_pct = 0.0

    if _TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            vram_used = torch.cuda.memory_allocated()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            vram_total = total_mem
            if vram_total > 0:
                vram_pct = (vram_used / vram_total) * 100.0
        except Exception:
            pass
    elif _PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used = int(info.used)
            vram_total = int(info.total)
            if vram_total > 0:
                vram_pct = (vram_used / vram_total) * 100.0
        except Exception:
            pass

    return ProcessTelemetry(
        pid=pid,
        cpu_percent=cpu_pct,
        ram_used_bytes=ram_bytes,
        ram_percent=ram_pct,
        host_ram_percent=host_ram_pct,
        vram_used_bytes=vram_used,
        vram_total_bytes=vram_total,
        vram_percent=vram_pct,
        num_threads=num_threads,
        num_children=num_children,
        is_running=is_running,
    )


# =============================================================================
# 7. Grid-Collapse Step-Back Recovery
# =============================================================================


def execute_grid_collapse_step_back(
    engine_pid: Optional[int],
    input_file_path: Union[str, Path],
    current_grid_angle_deg: float,
    output_file_path: Optional[Union[str, Path]] = None,
    config: Optional[WatchdogConfig] = None,
    scratch_dir: Optional[Union[str, Path]] = None,
    shift_deg: Optional[float] = None,
) -> StepBackResult:
    """Execute autonomous Step-Back recovery upon SCF divergence or numerical stagnation.

    Tears down hanging quantum chemical engine processes, widens the target
    torsional coordinate step, injects damped convergence directives (SlowConv,
    Damp, SOSCF, Shift, GridX), and generates a validated restart input payload
    strictly in the designated scratch space.

    Parameters
    ----------
    engine_pid : Optional[int]
        The PID of the hanging quantum chemistry calculation to terminate.
    input_file_path : Union[str, Path]
        Filesystem path to the failing input file (.inp).
    current_grid_angle_deg : float
        Current torsional coordinate angle in degrees.
    output_file_path : Optional[Union[str, Path]]
        Optional path to the associated output log for diagnostic inspection.
    config : Optional[WatchdogConfig]
        Watchdog configuration settings (uses defaults if None).
    scratch_dir : Optional[Union[str, Path]]
        Directory where recovery payloads will be written (Domain B).
    shift_deg : Optional[float]
        Explicit angular shift to apply (defaults to config.grid_step_shift_deg).

    Returns
    -------
    StepBackResult
        Result containing the adjusted grid angle, injected flags, and path to the rewritten input.

    Raises
    ------
    StepBackRecoveryError
        If the original input file cannot be read or recovery payload generation fails.
    AirGapViolationError
        If payload generation targets Domain A static repository space.
    """
    if config is None:
        config = WatchdogConfig()

    resolved_input_path = Path(input_file_path).resolve()
    if not resolved_input_path.is_file():
        raise StepBackRecoveryError(f"Target input file does not exist: {resolved_input_path}")

    # 1. Determine failure diagnostic from output file if available
    failure_mode = FailureMode.SCF_DIVERGENCE
    failure_desc = "Autonomous grid-collapse step-back invoked."
    if output_file_path:
        out_p = Path(output_file_path).resolve()
        if out_p.is_file():
            try:
                log_text = out_p.read_text(encoding="utf-8", errors="replace")
                has_failed, mode, desc = scan_log_for_scf_failure(log_text)
                if has_failed:
                    failure_mode = mode
                    failure_desc = desc
            except OSError as e:
                logger.warning(f"Could not read output log {out_p}: {e}")

    # 2. Terminate hanging process tree if PID specified
    if engine_pid is not None and engine_pid > 0:
        logger.info(f"Tearing down hanging engine process tree PID {engine_pid} due to {failure_mode.value}.")
        safe_process_tree_teardown(engine_pid, timeout=config.process_teardown_timeout_sec)

    # 3. Compute widened torsional step angle
    delta_shift = shift_deg if shift_deg is not None else config.grid_step_shift_deg
    new_grid_angle = (current_grid_angle_deg + delta_shift) % 360.0

    # 4. Ingest and modify quantum chemical input file
    try:
        original_content = resolved_input_path.read_text(encoding="utf-8")
    except OSError as e:
        raise StepBackRecoveryError(f"Failed to read input file {resolved_input_path}: {e}") from e

    modified_lines: List[str] = []
    applied_flags: List[str] = []

    # Prepare convergence injection directives
    damped_keywords = list(config.scf_damped_keywords)
    if config.scf_grid_fallback not in damped_keywords:
        damped_keywords.append(config.scf_grid_fallback)

    # Process input content line by line
    scf_block_found = False
    in_scf_block = False

    for line in original_content.splitlines():
        trimmed = line.strip()

        # Handle simple keyword line (starts with !)
        if trimmed.startswith("!"):
            existing_tokens = set(trimmed[1:].split())
            new_tokens = list(trimmed[1:].split())
            for kw in damped_keywords:
                if kw not in existing_tokens:
                    new_tokens.append(kw)
                    applied_flags.append(kw)
            modified_lines.append("! " + " ".join(new_tokens))
            continue

        # Handle %scf block
        if trimmed.lower().startswith("%scf"):
            scf_block_found = True
            in_scf_block = True
            modified_lines.append(line)
            # Inject robust convergence sub-directives
            modified_lines.append("    MaxIter 300")
            modified_lines.append("    Damp 0.7")
            modified_lines.append("    Shift 0.2")
            modified_lines.append("    DIIS true")
            modified_lines.append("    SOSCF true")
            applied_flags.extend(["MaxIter 300", "Damp 0.7", "Shift 0.2", "DIIS", "SOSCF"])
            continue

        if in_scf_block and trimmed.lower().startswith("end"):
            in_scf_block = False
            modified_lines.append(line)
            continue

        # Modify dihedral scan angle constraint if present in input
        if re.search(r"\b(scan|dihedral|d\s+\d+\s+\d+\s+\d+\s+\d+)\b", trimmed, re.IGNORECASE):
            angle_match = re.search(r"=\s*([+-]?\d+\.?\d*)", trimmed)
            if angle_match:
                old_val = angle_match.group(1)
                new_val_str = f"{new_grid_angle:.2f}"
                line = line.replace(f"={old_val}", f"={new_val_str}").replace(f"= {old_val}", f"= {new_val_str}")
                applied_flags.append(f"GridAngleShift({old_val}->{new_val_str})")

        modified_lines.append(line)

    # If no %scf block existed, append one
    if not scf_block_found:
        modified_lines.append("")
        modified_lines.append("%scf")
        modified_lines.append("    MaxIter 300")
        modified_lines.append("    Damp 0.7")
        modified_lines.append("    Shift 0.2")
        modified_lines.append("    DIIS true")
        modified_lines.append("    SOSCF true")
        modified_lines.append("end")
        applied_flags.extend(["%scf Block Injected"])

    # 5. Resolve destination scratch directory with Air-Gap enforcement
    if scratch_dir is not None:
        target_scratch = Path(scratch_dir).resolve()
    else:
        target_scratch = get_scratch_dir() / "orca_step_back"

    validate_runtime_write_path(target_scratch)
    target_scratch.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    new_input_filename = f"step_back_{timestamp_str}_{resolved_input_path.name}"
    new_input_path = target_scratch / new_input_filename

    # Write modified input payload
    try:
        new_input_path.write_text("\n".join(modified_lines) + "\n", encoding="utf-8")
    except OSError as e:
        raise StepBackRecoveryError(f"Failed to write step-back recovery input to {new_input_path}: {e}") from e

    # 6. Construct audit event record
    event = RecoveryEvent(
        action=RecoveryActionType.GRID_COLLAPSE_STEP_BACK,
        failure_mode=failure_mode,
        pid=engine_pid,
        details={
            "original_grid_angle_deg": current_grid_angle_deg,
            "new_grid_angle_deg": new_grid_angle,
            "delta_shift_deg": delta_shift,
            "applied_flags": applied_flags,
            "original_input": str(resolved_input_path),
            "new_input": str(new_input_path),
            "diagnostic_message": failure_desc,
        },
        success=True,
    )

    logger.info(
        f"Step-Back Guard successfully shifted torsional grid ({current_grid_angle_deg:.1f}° -> {new_grid_angle:.1f}°) "
        f"and generated recovery input at {new_input_path}."
    )

    return StepBackResult(
        success=True,
        new_input_path=new_input_path,
        new_grid_angle_deg=new_grid_angle,
        applied_flags=applied_flags,
        original_pid=engine_pid,
        event=event,
    )


# =============================================================================
# 8. Dynamic Memory Backoff
# =============================================================================


def purge_bloated_scratch_matrices(scratch_path: Union[str, Path]) -> Tuple[List[Path], int]:
    """Scan and delete bloated temporary quantum chemistry matrix files (.tmp, .mat).

    Parameters
    ----------
    scratch_path : Union[str, Path]
        The scratch directory to sanitize.

    Returns
    -------
    Tuple[List[Path], int]
        (purged_file_paths, total_purged_bytes)
    """
    purged_files: List[Path] = []
    total_bytes: int = 0

    p = Path(scratch_path).resolve()
    if not p.is_dir():
        return purged_files, total_bytes

    # Patterns matching scratch matrices and temporary buffers
    target_extensions = {".tmp", ".mat", ".densities", ".scftmp", ".overlap_tmp", ".mo_tmp"}

    for item in p.rglob("*"):
        if item.is_file():
            is_match = item.suffix.lower() in target_extensions or ".tmp." in item.name.lower() or item.name.endswith(".tmp")
            if is_match:
                try:
                    file_size = item.stat().st_size
                    item.unlink()
                    purged_files.append(item)
                    total_bytes += file_size
                except OSError as e:
                    logger.warning(f"Could not purge temporary file {item}: {e}")

    return purged_files, total_bytes


def dynamic_memory_backoff(
    engine_pid: Optional[int],
    input_file_path: Union[str, Path],
    current_maxcore_mb: int,
    scratch_dir: Optional[Union[str, Path]] = None,
    config: Optional[WatchdogConfig] = None,
    force_backoff: bool = False,
) -> MemoryBackoffResult:
    """Execute dynamic memory backoff to prevent host Out-Of-Memory kernel panics.

    Evaluates system RAM pressure via psutil. If RAM usage exceeds the threshold
    (or if forced), safely terminates the engine process tree, reduces %maxcore
    allocation by 30%, purges bloated scratch matrices, and outputs an updated
    input file to scratch space.

    Parameters
    ----------
    engine_pid : Optional[int]
        The PID of the memory-intensive engine process to terminate.
    input_file_path : Union[str, Path]
        Filesystem path to the input file (.inp).
    current_maxcore_mb : int
        Current %maxcore memory allocation in MB per core.
    scratch_dir : Optional[Union[str, Path]]
        Scratch directory where temporary files reside and payload will be written.
    config : Optional[WatchdogConfig]
        Watchdog configuration settings (uses defaults if None).
    force_backoff : bool
        If True, executes backoff unconditionally regardless of current RAM percentage.

    Returns
    -------
    MemoryBackoffResult
        Result containing the previous and new maxcore values, purged file list, and new input path.

    Raises
    ------
    MemoryBackoffError
        If maxcore cannot be reduced further or input modification fails.
    AirGapViolationError
        If payload generation targets Domain A static repository space.
    """
    if config is None:
        config = WatchdogConfig()

    host_ram_pct = psutil.virtual_memory().percent
    should_backoff = force_backoff or (host_ram_pct >= config.ram_threshold_pct)

    if not should_backoff:
        return MemoryBackoffResult(
            success=False,
            previous_maxcore_mb=current_maxcore_mb,
            new_maxcore_mb=current_maxcore_mb,
            purged_scratch_files=[],
            purged_bytes=0,
            new_input_path=None,
            event=None,
        )

    resolved_input_path = Path(input_file_path).resolve()
    if not resolved_input_path.is_file():
        raise MemoryBackoffError(f"Target input file does not exist: {resolved_input_path}")

    # 1. Terminate memory-intensive process tree
    if engine_pid is not None and engine_pid > 0:
        logger.info(f"Dynamic memory backoff terminating PID {engine_pid} (RAM: {host_ram_pct:.1f}%).")
        safe_process_tree_teardown(engine_pid, timeout=config.process_teardown_timeout_sec)

    # 2. Scale down %maxcore by factor (e.g. 0.7 = 30% reduction)
    calculated_maxcore = int(current_maxcore_mb * config.maxcore_reduction_factor)
    new_maxcore = max(calculated_maxcore, config.min_maxcore_mb)

    # 3. Purge scratch matrices
    if scratch_dir is not None:
        target_scratch = Path(scratch_dir).resolve()
    else:
        target_scratch = get_scratch_dir() / "orca_tmp"

    validate_runtime_write_path(target_scratch)
    target_scratch.mkdir(parents=True, exist_ok=True)
    purged_files, purged_bytes = purge_bloated_scratch_matrices(target_scratch)

    # 4. Ingest and rewrite input file with scaled %maxcore
    try:
        original_content = resolved_input_path.read_text(encoding="utf-8")
    except OSError as e:
        raise MemoryBackoffError(f"Failed to read input file {resolved_input_path}: {e}") from e

    modified_lines: List[str] = []
    maxcore_replaced = False

    for line in original_content.splitlines():
        trimmed = line.strip()
        if trimmed.lower().startswith("%maxcore"):
            modified_lines.append(f"%maxcore {new_maxcore}")
            maxcore_replaced = True
        else:
            modified_lines.append(line)

    if not maxcore_replaced:
        modified_lines.insert(0, f"%maxcore {new_maxcore}")

    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    new_input_filename = f"mem_backoff_{timestamp_str}_{resolved_input_path.name}"
    new_input_path = target_scratch / new_input_filename

    try:
        new_input_path.write_text("\n".join(modified_lines) + "\n", encoding="utf-8")
    except OSError as e:
        raise MemoryBackoffError(f"Failed to write memory backoff input to {new_input_path}: {e}") from e

    # 5. Record recovery audit event
    event = RecoveryEvent(
        action=RecoveryActionType.MEMORY_BACKOFF,
        failure_mode=FailureMode.RAM_EXHAUSTION,
        pid=engine_pid,
        details={
            "host_ram_percent": host_ram_pct,
            "ram_threshold_pct": config.ram_threshold_pct,
            "previous_maxcore_mb": current_maxcore_mb,
            "new_maxcore_mb": new_maxcore,
            "purged_scratch_files_count": len(purged_files),
            "purged_bytes": purged_bytes,
            "new_input_path": str(new_input_path),
        },
        success=True,
    )

    logger.info(
        f"Dynamic Memory Backoff complete: reduced %maxcore from {current_maxcore_mb} MB to {new_maxcore} MB, "
        f"purged {len(purged_files)} scratch files ({purged_bytes / (1024*1024):.2f} MB freed)."
    )

    return MemoryBackoffResult(
        success=True,
        previous_maxcore_mb=current_maxcore_mb,
        new_maxcore_mb=new_maxcore,
        purged_scratch_files=purged_files,
        purged_bytes=purged_bytes,
        new_input_path=new_input_path,
        event=event,
    )


# =============================================================================
# 9. CUDA Memory Leak Guard
# =============================================================================


def cuda_memory_leak_guard(
    threshold_vram_pct: Optional[float] = None,
    config: Optional[WatchdogConfig] = None,
    force_flush: bool = False,
) -> CudaGuardResult:
    """Proactively monitor GPU VRAM utilization and release cached tensor memory.

    Flushes CuPy memory pools and PyTorch CUDA cache, and invokes Python garbage
    collection to ensure a flat VRAM baseline across multi-day execution loops.

    Parameters
    ----------
    threshold_vram_pct : Optional[float]
        Optional override for VRAM threshold (defaults to config.vram_threshold_pct).
    config : Optional[WatchdogConfig]
        Watchdog configuration settings (uses defaults if None).
    force_flush : bool
        If True, forces memory pool cleanup regardless of current VRAM level.

    Returns
    -------
    CudaGuardResult
        Detailed report of VRAM levels before/after, freed bytes, and cleanup actions taken.
    """
    if config is None:
        config = WatchdogConfig()

    active_threshold = threshold_vram_pct if threshold_vram_pct is not None else config.vram_threshold_pct

    # Measure initial VRAM utilization
    vram_before_pct = 0.0
    vram_before_bytes = 0

    if _TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            vram_before_bytes = torch.cuda.memory_allocated()
            total_vram = torch.cuda.get_device_properties(0).total_memory
            if total_vram > 0:
                vram_before_pct = (vram_before_bytes / total_vram) * 100.0
        except Exception as e:
            logger.debug(f"Torch CUDA memory inquiry notice: {e}")

    should_clean = force_flush or (vram_before_pct >= active_threshold)

    cupy_freed = False
    torch_freed = False
    gc_count = 0

    if should_clean:
        # 1. Release CuPy memory pool allocations
        if _CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                cupy_freed = True
                logger.debug("CuPy default and pinned memory pools released.")
            except Exception as e:
                logger.warning(f"Failed to release CuPy memory pool: {e}")

        # 2. Release PyTorch CUDA cache and IPC handles
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    getattr(torch.cuda, "ipc_collect")()
                torch_freed = True
                logger.debug("PyTorch CUDA cache emptied.")
            except Exception as e:
                logger.warning(f"Failed to clear PyTorch CUDA cache: {e}")

        # 3. Force Python garbage collection
        gc_count = gc.collect()

    # Measure post-cleanup VRAM utilization
    vram_after_pct = 0.0
    vram_after_bytes = 0

    if _TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            vram_after_bytes = torch.cuda.memory_allocated()
            total_vram = torch.cuda.get_device_properties(0).total_memory
            if total_vram > 0:
                vram_after_pct = (vram_after_bytes / total_vram) * 100.0
        except Exception:
            pass

    freed_bytes = max(0, vram_before_bytes - vram_after_bytes)

    event: Optional[RecoveryEvent] = None
    if should_clean:
        event = RecoveryEvent(
            action=RecoveryActionType.CUDA_CACHE_CLEAR,
            failure_mode=FailureMode.VRAM_EXHAUSTION if vram_before_pct >= active_threshold else FailureMode.NONE,
            details={
                "vram_percent_before": vram_before_pct,
                "vram_percent_after": vram_after_pct,
                "freed_bytes": freed_bytes,
                "cupy_freed": cupy_freed,
                "torch_freed": torch_freed,
                "gc_collected_count": gc_count,
            },
            success=True,
        )

    return CudaGuardResult(
        vram_percent_before=vram_before_pct,
        vram_percent_after=vram_after_pct,
        freed_bytes=freed_bytes,
        cupy_freed=cupy_freed,
        torch_freed=torch_freed,
        gc_collected_count=gc_count,
        event=event,
    )


# =============================================================================
# 10. Autonomous Watchdog Monitor & Daemon
# =============================================================================


class TorqWatchdogDaemon:
    """Autonomous OS-level watchdog monitoring daemon for CoChem-TORQ calculations.

    Monitors host hardware pressure (RAM, VRAM), tails execution logs in real-time,
    and coordinates autonomous error recovery interventions without interrupting
    the high-level workflow.
    """

    def __init__(
        self,
        config: Optional[WatchdogConfig] = None,
        scratch_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the Watchdog Daemon with configuration and scratch directory.

        Parameters
        ----------
        config : Optional[WatchdogConfig]
            Watchdog configuration settings.
        scratch_dir : Optional[Union[str, Path]]
            Ephemeral scratch storage directory (Domain B).
        """
        self.config = config if config is not None else WatchdogConfig()
        if scratch_dir is not None:
            self.scratch_dir = Path(scratch_dir).resolve()
        else:
            self.scratch_dir = get_scratch_dir()
        validate_runtime_write_path(self.scratch_dir)

        self._active: bool = False
        self._monitored_pid: Optional[int] = None
        self._input_path: Optional[Path] = None
        self._log_path: Optional[Path] = None
        self._current_grid_angle: float = 0.0
        self._current_maxcore_mb: int = 2000
        self._events: List[RecoveryEvent] = []
        self._telemetry_history: List[ProcessTelemetry] = []
        self._restart_count: int = 0

    @property
    def is_active(self) -> bool:
        """Return whether monitoring is active."""
        return self._active

    def start_monitoring(
        self,
        pid: int,
        input_path: Union[str, Path],
        log_path: Optional[Union[str, Path]] = None,
        current_grid_angle: float = 0.0,
        current_maxcore_mb: int = 2000,
    ) -> None:
        """Arm the watchdog to monitor an active engine process.

        Parameters
        ----------
        pid : int
            OS process identifier of the quantum chemistry calculation.
        input_path : Union[str, Path]
            Path to the calculation input file.
        log_path : Optional[Union[str, Path]]
            Path to the output log file.
        current_grid_angle : float
            Current torsional coordinate angle in degrees.
        current_maxcore_mb : int
            Current %maxcore memory allocation in MB.
        """
        self._monitored_pid = pid
        self._input_path = Path(input_path).resolve()
        self._log_path = Path(log_path).resolve() if log_path else None
        self._current_grid_angle = current_grid_angle
        self._current_maxcore_mb = current_maxcore_mb
        self._active = True
        logger.info(f"Watchdog armed for PID {pid} (Maxcore: {current_maxcore_mb} MB, Grid: {current_grid_angle:.1f}°).")

    def stop_monitoring(self) -> None:
        """Disarm the watchdog monitoring session."""
        self._active = False
        self._monitored_pid = None
        logger.info("Watchdog disarmed.")

    def poll_health(self) -> Optional[RecoveryEvent]:
        """Poll telemetry and logs once; trigger recovery if an anomaly is detected.

        Returns
        -------
        Optional[RecoveryEvent]
            The recovery event executed, or None if hardware and calculations are healthy.
        """
        if not self._active or self._monitored_pid is None:
            return None

        # 1. Capture process and hardware telemetry
        telemetry = collect_process_telemetry(self._monitored_pid)
        self._telemetry_history.append(telemetry)

        # 2. Check RAM pressure -> Dynamic Memory Backoff
        if telemetry.host_ram_percent >= self.config.ram_threshold_pct:
            logger.warning(
                f"RAM threshold breached ({telemetry.host_ram_percent:.1f}% >= {self.config.ram_threshold_pct:.1f}%). "
                f"Triggering dynamic memory backoff."
            )
            backoff_res = dynamic_memory_backoff(
                engine_pid=self._monitored_pid,
                input_file_path=self._input_path if self._input_path else Path("job.inp"),
                current_maxcore_mb=self._current_maxcore_mb,
                scratch_dir=self.scratch_dir,
                config=self.config,
                force_backoff=True,
            )
            if backoff_res.event:
                self._events.append(backoff_res.event)
                self._current_maxcore_mb = backoff_res.new_maxcore_mb
                if backoff_res.new_input_path:
                    self._input_path = backoff_res.new_input_path
                self._restart_count += 1
                return backoff_res.event

        # 3. Check VRAM pressure -> CUDA Memory Leak Guard
        if telemetry.vram_percent >= self.config.vram_threshold_pct:
            logger.warning(
                f"VRAM threshold breached ({telemetry.vram_percent:.1f}% >= {self.config.vram_threshold_pct:.1f}%). "
                f"Triggering CUDA memory leak guard."
            )
            cuda_res = cuda_memory_leak_guard(
                threshold_vram_pct=self.config.vram_threshold_pct,
                config=self.config,
                force_flush=True,
            )
            if cuda_res.event:
                self._events.append(cuda_res.event)
                return cuda_res.event

        # 4. Tail and inspect log output if available -> Grid-Collapse Step-Back
        if self._log_path and self._log_path.is_file():
            recent_lines = tail_log_file(self._log_path, n_lines=150)
            log_text = "\n".join(recent_lines)
            has_failed, mode, desc = scan_log_for_scf_failure(log_text)

            if has_failed and mode in {FailureMode.SCF_DIVERGENCE, FailureMode.LINEAR_DEPENDENCE, FailureMode.GRADIENT_STAGNATION}:
                logger.warning(f"Quantum chemical failure detected ({mode.value}): {desc}. Triggering Step-Back Guard.")
                stepback_res = execute_grid_collapse_step_back(
                    engine_pid=self._monitored_pid,
                    input_file_path=self._input_path if self._input_path else Path("job.inp"),
                    current_grid_angle_deg=self._current_grid_angle,
                    output_file_path=self._log_path,
                    config=self.config,
                    scratch_dir=self.scratch_dir,
                )
                if stepback_res.event:
                    self._events.append(stepback_res.event)
                    self._current_grid_angle = stepback_res.new_grid_angle_deg
                    if stepback_res.new_input_path:
                        self._input_path = stepback_res.new_input_path
                    self._restart_count += 1
                    return stepback_res.event

        return None

    def get_status(self) -> WatchdogStatus:
        """Generate a complete status model for the current watchdog session."""
        return WatchdogStatus(
            active=self._active,
            monitored_pid=self._monitored_pid,
            events=list(self._events),
            telemetry_history=list(self._telemetry_history),
            active_maxcore_mb=self._current_maxcore_mb,
            active_grid_angle_deg=self._current_grid_angle,
            restart_count=self._restart_count,
        )


try:
    from cochem_base.cochem_torq_watchdog import (
        DynamicMemoryResult,
        dynamic_memory_backoff,
    )
except ImportError:
    pass
