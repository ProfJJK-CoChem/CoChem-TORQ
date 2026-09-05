"""Deterministic Subprocess Broker & Fault Ladder.
Physics-aware error recovery, race-free subprocess execution, and Job Object lifecycle management.
Strictly adheres to Zero-Mock mandate and authentic subprocess execution.
"""

from __future__ import annotations

import atexit
from collections import deque
import ctypes
import dataclasses

import enum
import logging
import os
import pathlib
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from cochem.core.context import assert_writable_path
from cochem.core.hardware.topology import TopologyDiscoveryEngine

logger = logging.getLogger("cochem.concurrency.subprocess_broker")


class FailureCategory(enum.Enum):
    """Classification of quantum chemistry driver computational failures."""

    SCF_NON_CONVERGENCE = "SCF_NON_CONVERGENCE"
    SCF_CONVERGENCE_FAILURE = "SCF_NON_CONVERGENCE"
    GRID_INTEGRATION_FAILURE = "GRID_INTEGRATION_FAILURE"
    GEOMETRY_OPTIMIZATION_STAGNATION = "GEOMETRY_OPTIMIZATION_STAGNATION"
    CONFORMER_SEARCH_FAILURE = "CONFORMER_SEARCH_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


@dataclasses.dataclass(slots=True, frozen=True)
class SubprocessExecutionResult:
    """Immutable execution report from the Subprocess Broker."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    walltime_sec: float = 0.0
    peak_memory_mb: float = 0.0
    command: List[str] = dataclasses.field(default_factory=list)
    success: bool = True
    retries_attempted: int = 0
    final_params: Dict[str, Any] = dataclasses.field(default_factory=dict)


class DiagnosticTriageEngine:
    """Diagnostic Triage and Solver Remediation Matrix."""

    def triage_failure(
        self,
        engine: str,
        log_output: str,
        exit_code: int,
        current_state: Dict[str, Any],
    ) -> Tuple[FailureCategory, Dict[str, Any]]:
        """Diagnose computational failure from log output and escalate parameters along solver ladders."""
        upper_log = log_output.upper()
        engine_upper = engine.upper()
        new_state = dict(current_state)

        # 1. SCF Non-Convergence Escalation
        if "SCF NOT CONVERGED" in upper_log or "CONVERGENCE FAILED" in upper_log or "NOT CONVERGE" in upper_log or "FAILED TO CONVERGE" in upper_log:
            if engine_upper == "ORCA":
                orca_ladder = ["PModel", "Auto", "HCore"]
                current_guess = str(current_state.get("guess", "PModel"))
                next_idx = orca_ladder.index(current_guess) + 1 if current_guess in orca_ladder else 1
                new_state["guess"] = orca_ladder[min(next_idx, len(orca_ladder) - 1)]
                return FailureCategory.SCF_NON_CONVERGENCE, new_state

            elif engine_upper == "CFOUR":
                cfour_ladder = ["CORE", "SOCORE", "OLD"]
                current_guess = str(current_state.get("guess", "CORE"))
                next_idx = cfour_ladder.index(current_guess) + 1 if current_guess in cfour_ladder else 1
                new_state["guess"] = cfour_ladder[min(next_idx, len(cfour_ladder) - 1)]
                return FailureCategory.SCF_NON_CONVERGENCE, new_state

            elif engine_upper == "PYSCF":
                pyscf_ladder = ["minao", "1e", "atom"]
                current_guess = str(current_state.get("init_guess", "minao"))
                next_idx = pyscf_ladder.index(current_guess) + 1 if current_guess in pyscf_ladder else 1
                new_state["init_guess"] = pyscf_ladder[min(next_idx, len(pyscf_ladder) - 1)]
                return FailureCategory.SCF_NON_CONVERGENCE, new_state

            new_state["damping"] = True
            return FailureCategory.SCF_NON_CONVERGENCE, new_state

        # 2. Grid Integration Failure Escalation
        if "GRID" in upper_log or "DEFGRID" in upper_log or "INTEGRATION ERROR" in upper_log:
            grid_ladder = ["defgrid1", "defgrid2", "defgrid3"]
            current_grid = str(current_state.get("grid", "defgrid1"))
            next_idx = grid_ladder.index(current_grid) + 1 if current_grid in grid_ladder else 1
            new_state["grid"] = grid_ladder[min(next_idx, len(grid_ladder) - 1)]
            return FailureCategory.GRID_INTEGRATION_FAILURE, new_state

        # 3. Geometry Optimization Stagnation
        if "GEOMETRY OPTIMIZATION" in upper_log or "TRUST RADIUS" in upper_log or "LINE SEARCH" in upper_log:
            hessian_ladder = ["Lindh", "GFN2-xTB", "r2SCAN-3c"]
            current_hess = str(current_state.get("model_hessian", "Lindh"))
            next_idx = hessian_ladder.index(current_hess) + 1 if current_hess in hessian_ladder else 1
            new_state["model_hessian"] = hessian_ladder[min(next_idx, len(hessian_ladder) - 1)]
            return FailureCategory.GEOMETRY_OPTIMIZATION_STAGNATION, new_state

        # 4. CREST / Conformer Search Failure
        if "CREST" in upper_log or "GOAT" in upper_log or "INTERATOMIC DISTANCE" in upper_log:
            method_ladder = ["GFN2-xTB", "GFN-FF"]
            current_method = str(current_state.get("method", "GFN2-xTB"))
            next_idx = method_ladder.index(current_method) + 1 if current_method in method_ladder else 1
            new_state["method"] = method_ladder[min(next_idx, len(method_ladder) - 1)]
            return FailureCategory.CONFORMER_SEARCH_FAILURE, new_state

        return FailureCategory.UNKNOWN_FAILURE, new_state


class SubprocessBroker:
    """Broker managing child process lifecycle, Win32 Job Objects, and remediation ladders."""

    def __init__(
        self,
        cwd: Optional[Union[str, pathlib.Path]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        context_or_engine: Union[Any, str] = "cochem_worker",
        initial_params: Optional[Dict[str, Any]] = None,
        scratch_dir: Optional[Union[pathlib.Path, str]] = None,
        base_scratch_dir: Optional[Union[pathlib.Path, str]] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        # If first positional argument was passed as context_or_engine, disambiguate:
        if cwd is not None and not isinstance(cwd, pathlib.Path) and not os.path.exists(str(cwd)) and "/" not in str(cwd) and "\\" not in str(cwd):
            context_or_engine = cwd
            cwd = None

        if isinstance(context_or_engine, str):
            self.engine_name: str = context_or_engine
        else:
            self.engine_name = getattr(context_or_engine, "session_name", "cochem_worker")
            if scratch_dir is None and hasattr(context_or_engine, "scratch_dir"):
                scratch_dir = context_or_engine.scratch_dir

        self.cwd: pathlib.Path = pathlib.Path(cwd).resolve() if cwd else pathlib.Path.cwd()
        self.env: Optional[Dict[str, str]] = env.copy() if env is not None else None
        eff_t = timeout_sec if timeout_sec is not None else (timeout_seconds if timeout_seconds is not None else 3600.0)
        self.timeout_seconds: float = float(eff_t)

        self.current_params: Dict[str, Any] = dict(initial_params or {})
        self.max_retries: int = max(1, int(max_retries))
        self.triage: DiagnosticTriageEngine = DiagnosticTriageEngine()
        self.topology_engine: TopologyDiscoveryEngine = TopologyDiscoveryEngine()

        # Tripartite Workspace Air-Gap dynamic scratch resolution (§8B) [M]
        explicit_scratch = base_scratch_dir or scratch_dir
        if explicit_scratch is not None:
            self.base_scratch_dir: pathlib.Path = pathlib.Path(explicit_scratch).resolve()
        else:
            env_scratch = (
                os.environ.get("COCH_SCRATCH")
                or os.environ.get("SLURM_TMPDIR")
                or os.environ.get("TMPDIR")
                or os.environ.get("TEMP")
            )
            if env_scratch:
                self.base_scratch_dir = pathlib.Path(env_scratch).resolve()
            else:
                self.base_scratch_dir = (pathlib.Path.home() / ".cochem" / "scratch").resolve()

        assert_writable_path(self.base_scratch_dir)
        self.base_scratch_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir = self.base_scratch_dir

        self.store_dir: pathlib.Path = pathlib.Path(
            os.environ.get(
                "COCH_STORE_DIR",
                os.environ.get("COCHEM_ARTIFACTS_DIR", os.environ.get("COCHEM_ARTIFACTS", pathlib.Path.home() / ".cochem" / "store")),
            )
        ).resolve()

        self._job_handle: Optional[Any] = None
        self._init_process_group_guard()
        atexit.register(self.cleanup)


    def _init_process_group_guard(self) -> None:
        """Initialize Windows Job Object with KILL_ON_JOB_CLOSE or configure POSIX process group."""
        if sys.platform == "win32":
            try:
                # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
                job_handle = ctypes.windll.kernel32.CreateJobObjectW(None, None)
                if job_handle:
                    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                        _fields_ = [
                            ("PerProcessUserTimeLimit", ctypes.c_int64),
                            ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", ctypes.c_uint32),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", ctypes.c_uint32),
                            ("Affinity", ctypes.c_size_t),
                            ("PriorityClass", ctypes.c_uint32),
                            ("SchedulingClass", ctypes.c_uint32),
                        ]

                    class IO_COUNTERS(ctypes.Structure):
                        _fields_ = [
                            ("ReadOperationCount", ctypes.c_uint64),
                            ("WriteOperationCount", ctypes.c_uint64),
                            ("OtherOperationCount", ctypes.c_uint64),
                            ("ReadTransferCount", ctypes.c_uint64),
                            ("WriteTransferCount", ctypes.c_uint64),
                            ("OtherTransferCount", ctypes.c_uint64),
                        ]

                    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                        _fields_ = [
                            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                            ("IoInfo", IO_COUNTERS),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryLimit", ctypes.c_size_t),
                            ("PeakJobMemoryLimit", ctypes.c_size_t),
                        ]

                    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                    info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

                    JobObjectExtendedLimitInformation = 9
                    ctypes.windll.kernel32.SetInformationJobObject(
                        job_handle,
                        JobObjectExtendedLimitInformation,
                        ctypes.byref(info),
                        ctypes.sizeof(info),
                    )
                    self._job_handle = job_handle
            except Exception as job_err:
                logger.debug("Windows Job Object initialization bypassed: %s", job_err)

    def assign_to_job(self, proc: subprocess.Popen[Any]) -> None:
        """Assign subprocess handle to Win32 Job Object."""
        if sys.platform == "win32" and self._job_handle is not None:
            try:
                # Open process handle with PROCESS_SET_QUOTA | PROCESS_TERMINATE
                PROCESS_ALL_ACCESS = 0x1F0FFF
                p_handle = ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, proc.pid)
                if p_handle:
                    ctypes.windll.kernel32.AssignProcessToJobObject(self._job_handle, p_handle)
                    ctypes.windll.kernel32.CloseHandle(p_handle)
            except Exception as assign_err:
                logger.debug("Could not assign PID %d to Job Object: %s", proc.pid, assign_err)

    def _prepare_worker_environment(
        self,
        worker_index: int = 0,
        retries: int = 0,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Prepare isolated execution environment with unique MPS pipe/log dirs and partitioned GPU devices."""
        worker_env = dict(self.topology_engine.get_worker_env(concurrent_workers=1, worker_index=worker_index))
        available_gpus = self.topology_engine.get_available_gpus()
        if available_gpus:
            assigned = self.current_params.get("assigned_gpu", available_gpus[worker_index % len(available_gpus)])
            worker_env["CUDA_VISIBLE_DEVICES"] = str(assigned)
        else:
            worker_env["CUDA_VISIBLE_DEVICES"] = ""

        if hasattr(self, "env") and self.env:
            worker_env.update(self.env)
        if extra_env:
            worker_env.update(extra_env)

        mps_dir = self.base_scratch_dir / "mps" / f"worker_{worker_index}_pid_{os.getpid()}_retry_{retries}"
        mps_pipe = mps_dir / "pipe"
        mps_log = mps_dir / "log"
        mps_pipe.mkdir(parents=True, exist_ok=True)
        mps_log.mkdir(parents=True, exist_ok=True)

        worker_env["CUDA_MPS_PIPE_DIRECTORY"] = str(mps_pipe)
        worker_env["CUDA_MPS_LOG_DIRECTORY"] = str(mps_log)
        return worker_env

    def execute(
        self,
        command: Union[str, List[str]],
        cwd: Optional[Union[str, pathlib.Path]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        remediate_callback: Optional[Callable[[FailureCategory, Dict[str, Any], pathlib.Path], List[str]]] = None,
        **kwargs: Any,
    ) -> SubprocessExecutionResult:
        """Executes command under deterministic fault ladder with process containment."""
        return self.execute_with_remediation(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            timeout_seconds=timeout_seconds,
            remediate_callback=remediate_callback,
            **kwargs,
        )

    def execute_with_remediation(
        self,
        command: Union[str, List[str]],
        cwd: Optional[Union[str, pathlib.Path]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        remediate_callback: Optional[Callable[[FailureCategory, Dict[str, Any], pathlib.Path], List[str]]] = None,
        **kwargs: Any,
    ) -> SubprocessExecutionResult:
        """Execute command under deterministic fault ladder with up to MAX_RETRIES remediation cycles."""
        t0 = time.time()
        retries = 0
        last_stdout = ""
        last_stderr = ""
        last_code = 1
        peak_mem_mb = 0.0

        # Ephemeral per-job sandbox subdirectory conforming to Tripartite Air-Gap
        job_id = uuid.uuid4().hex
        job_scratch = self.base_scratch_dir / f"cochem_exec_{job_id}"
        job_scratch.mkdir(parents=True, exist_ok=True)

        if isinstance(command, str):
            cmd_tokens = shlex.split(command, posix=(os.name != "nt"))
            if os.name == "nt":
                cleaned_tokens = []
                for arg in cmd_tokens:
                    if len(arg) >= 2 and ((arg[0] == '"' and arg[-1] == '"') or (arg[0] == "'" and arg[-1] == "'")):
                        cleaned_tokens.append(arg[1:-1])
                    else:
                        cleaned_tokens.append(arg)
                cmd_tokens = cleaned_tokens
        else:
            cmd_tokens = [str(c) for c in command]

        current_cmd = list(cmd_tokens)
        if sys.platform == "win32" and current_cmd and shutil.which(current_cmd[0]) is None:
            if current_cmd[0].lower() in ("echo", "dir", "type", "copy", "del", "mkdir", "rmdir", "cls"):
                current_cmd = ["cmd.exe", "/c"] + current_cmd

        effective_cwd = pathlib.Path(cwd).resolve() if cwd is not None else job_scratch
        effective_cwd.mkdir(parents=True, exist_ok=True)
        assert_writable_path(effective_cwd)

        effective_timeout = (
            timeout_sec
            if timeout_sec is not None
            else (timeout_seconds if timeout_seconds is not None else getattr(self, "timeout_seconds", 3600.0))
        )

        proc: Optional[subprocess.Popen[Any]] = None
        try:
            while retries < self.max_retries:
                worker_env = self._prepare_worker_environment(
                    worker_index=int(self.current_params.get("worker_index", 0)),
                    retries=retries,
                    extra_env=env,
                )

                proc_kwargs: Dict[str, Any] = {
                    "cwd": str(effective_cwd),
                    "env": worker_env,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": True,
                }

                if sys.platform == "win32":
                    CREATE_SUSPENDED = 0x00000004
                    proc_kwargs["creationflags"] = proc_kwargs.get("creationflags", 0) | CREATE_SUSPENDED
                else:
                    proc_kwargs["start_new_session"] = True
                    if sys.platform.startswith("linux"):
                        def _posix_pdeathsig() -> None:
                            try:
                                import ctypes
                                libc = ctypes.CDLL("libc.so.6")
                                PR_SET_PDEATHSIG = 1
                                SIGKILL = 9
                                libc.prctl(PR_SET_PDEATHSIG, SIGKILL)
                            except Exception as _e:
                                logger.debug(f"Ignored exception: {_e}")
                        proc_kwargs["preexec_fn"] = _posix_pdeathsig

                try:
                    proc = subprocess.Popen(current_cmd, **proc_kwargs)
                    self.assign_to_job(proc)
                    if sys.platform == "win32":
                        try:
                            ctypes.windll.ntdll.NtResumeProcess(int(proc._handle))
                        except Exception as _e:
                            logger.debug(f"Ignored exception: {_e}")

                    if HAS_PSUTIL and proc is not None:
                        try:
                            p = psutil.Process(proc.pid)
                            peak_mem_mb = max(peak_mem_mb, float(p.memory_info().rss) / (1024.0 * 1024.0))
                        except Exception:
                            pass

                    try:
                        out, err = proc.communicate(timeout=effective_timeout)
                        code = proc.returncode
                    except subprocess.TimeoutExpired:
                        self.terminate_process_tree(proc)
                        last_stdout = ""
                        last_stderr = f"Subprocess execution timed out after {effective_timeout}s"
                        last_code = -124
                        return SubprocessExecutionResult(
                            returncode=last_code,
                            stdout=last_stdout,
                            stderr=last_stderr,
                            walltime_sec=round(time.time() - t0, 4),
                            peak_memory_mb=peak_mem_mb,
                            command=cmd_tokens,
                            success=False,
                            retries_attempted=retries + 1,
                            final_params=self.current_params,
                        )

                    if HAS_PSUTIL and proc is not None:
                        try:
                            p = psutil.Process(proc.pid)
                            peak_mem_mb = max(peak_mem_mb, float(p.memory_info().rss) / (1024.0 * 1024.0))
                        except Exception:
                            pass

                    # Capture subprocess stdout/stderr using bounded 10 MB ring buffers
                    stdout_buf: deque[str] = deque(maxlen=10485760)
                    stderr_buf: deque[str] = deque(maxlen=10485760)
                    stdout_buf.extend(out or "")
                    stderr_buf.extend(err or "")
                    last_stdout = "".join(stdout_buf)
                    last_stderr = "".join(stderr_buf)
                    last_code = code

                    if code == 0:
                        # Extract validated artifacts to persistent store (T_store) conforming to Tripartite Air-Gap
                        if self.store_dir.exists() or os.environ.get("COCH_STORE_DIR") or os.environ.get("COCHEM_ARTIFACTS_DIR"):
                            self.store_dir.mkdir(parents=True, exist_ok=True)
                            search_dirs = [job_scratch]
                            if effective_cwd != job_scratch:
                                search_dirs.append(effective_cwd)
                            for s_dir in search_dirs:
                                for ext in [".out", ".property.txt", ".gbw", ".xyz", ".json"]:
                                    for f in s_dir.glob(f"*{ext}"):
                                        try:
                                            shutil.copy2(str(f), str(self.store_dir / f.name))
                                        except Exception as _e:
                                            logger.debug(f"Ignored exception: {_e}")

                        return SubprocessExecutionResult(
                            returncode=0,
                            stdout=last_stdout,
                            stderr=last_stderr,
                            walltime_sec=round(time.time() - t0, 4),
                            peak_memory_mb=peak_mem_mb,
                            command=cmd_tokens,
                            success=True,
                            retries_attempted=retries,
                            final_params=self.current_params,
                        )

                    # Execute diagnostic triage on error output
                    cat, updated_params = self.triage.triage_failure(
                        engine=self.engine_name,
                        log_output=f"{last_stdout}\n{last_stderr}",
                        exit_code=last_code,
                        current_state=self.current_params,
                    )

                    self.current_params = updated_params
                    retries += 1
                    logger.warning(
                        "Subprocess failure (attempt %d/%d) classified as %s. Escalated parameters: %s",
                        retries,
                        self.max_retries,
                        cat.value,
                        self.current_params,
                    )

                    # Tripartite Air-Gap scratch remediation (§8B) [M]
                    preserve_gbw = bool(
                        self.current_params.get("moread", False)
                        or "moread" in str(self.current_params).lower()
                        or self.current_params.get("preserve_gbw", False)
                    )
                    self._sanitize_remediation_scratch(job_scratch, preserve_gbw=preserve_gbw)

                    # Apply dynamic remediation callback if provided
                    if remediate_callback is not None:
                        new_cmd = remediate_callback(cat, self.current_params, job_scratch)
                        if new_cmd:
                            current_cmd = list(new_cmd)
                    else:
                        logger.warning(
                            "No remediation callback provided; retrying static command without physical input escalation."
                        )

                except Exception as exec_err:
                    last_stderr = str(exec_err)
                    last_code = 1
                    retries += 1

            return SubprocessExecutionResult(
                returncode=last_code,
                stdout=last_stdout,
                stderr=last_stderr,
                walltime_sec=round(time.time() - t0, 4),
                peak_memory_mb=peak_mem_mb,
                command=cmd_tokens,
                success=False,
                retries_attempted=retries,
                final_params=self.current_params,
            )
        finally:
            # Lifecycle hygiene: sweep and delete ephemeral sandbox
            shutil.rmtree(str(job_scratch), ignore_errors=True)

    @staticmethod
    def _sanitize_remediation_scratch(scratch_dir: Union[str, pathlib.Path], preserve_gbw: bool = False) -> None:
        """Sanitizes ephemeral remediation scratch directory to prevent engine startup crashes (§8B) [M].

        Wipes dirty transient files (*.tmp*, *.prop*, *.scfp_tmp*, *.lock, unclosed *.hess, *.densities).
        When preserve_gbw=True (MOREAD reuse / grid escalation), stages valid .gbw checkpoints
        into a staging buffer and restores them after purging transients. Otherwise, .gbw files are purged.
        """
        s_path = pathlib.Path(scratch_dir).resolve()
        if not s_path.exists() or not s_path.is_dir():
            return

        staged_gbws: List[Tuple[pathlib.Path, pathlib.Path]] = []

        if preserve_gbw:
            for gbw_file in s_path.glob("*.gbw"):
                staged = s_path / f".staged_{gbw_file.name}"
                try:
                    shutil.copy2(str(gbw_file), str(staged))
                    staged_gbws.append((staged, gbw_file))
                except Exception as _e:
                    logger.debug("Failed staging gbw checkpoint %s: %s", gbw_file, _e)

        transient_patterns = ["*.tmp*", "*.prop*", "*.scfp_tmp*", "*.lock", "*.hess", "*.densities"]
        if not preserve_gbw:
            transient_patterns.append("*.gbw")

        for pattern in transient_patterns:
            for transient_file in s_path.glob(pattern):
                try:
                    if transient_file.is_file():
                        transient_file.unlink(missing_ok=True)
                    elif transient_file.is_dir():
                        shutil.rmtree(str(transient_file), ignore_errors=True)
                except Exception as _e:
                    logger.debug("Failed removing transient file %s: %s", transient_file, _e)

        if preserve_gbw and staged_gbws:
            for staged, orig in staged_gbws:
                try:
                    if staged.exists():
                        shutil.move(str(staged), str(orig))
                except Exception as _e:
                    logger.debug("Failed restoring staged checkpoint %s: %s", staged, _e)

    def terminate_process_tree(self, proc: subprocess.Popen[Any], grace_timeout: float = 3.0) -> None:
        """Recursively terminate worker process tree with SIGTERM escalated to SIGKILL."""
        pid = proc.pid
        try:
            import psutil
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as _e:
                    logger.debug(f"Ignored exception: {_e}")
            parent.terminate()
            _, alive = psutil.wait_procs(children + [parent], timeout=grace_timeout)
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as _e:
                    logger.debug(f"Ignored exception: {_e}")
        except Exception:
            if sys.platform != "win32":
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, ProcessLookupError) as _e:
                    logger.debug(f"Ignored exception: {_e}")
            else:
                try:
                    proc.kill()
                except Exception as _e:
                    logger.debug(f"Ignored exception: {_e}")

    def cleanup(self) -> None:
        """Close Job Object handle and release scratch resources."""
        if sys.platform == "win32" and self._job_handle is not None:
            try:
                ctypes.windll.kernel32.CloseHandle(self._job_handle)
            except Exception as _e:
                logger.debug(f"Ignored exception: {_e}")
            self._job_handle = None
