"""Core AsyncProcessRunner integrating CUDA budgeting, Slurm preflight, and MPI supervision.

Enforces Tripartite Air-Gapped Storage ($COCH_SRC, $COCH_ARTIFACTS, $COCH_SCRATCH)
and concurrency-safe Single-Writer Multiple-Reader (SWMR) HDF5 telemetry.
"""

import os
import math
import re
import tempfile
import threading
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import filelock
import h5py

from cochem.hpc.models import (
    MpiClusterExecutionConfig,
    SlurmDryRunResult,
    SlurmJobDirectiveSpec,
    SlurmResourceValidationError,
)
from cochem.hpc.slurm_generator import SlurmDryRunGenerator
from cochem.runners.cuda_budget import CudaMemoryManager
from cochem.runners.mpi_supervisor import MpiProcessSupervisor


class AsyncProcessRunner:
    """Orchestrates high-performance computational workflows across heterogeneous cluster resources."""

    def __init__(
        self,
        src_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
        scratch_dir: Optional[Path] = None,
        cuda_manager: Optional[CudaMemoryManager] = None,
        slurm_generator: Optional[SlurmDryRunGenerator] = None,
        mpi_supervisor: Optional[MpiProcessSupervisor] = None,
    ) -> None:
        src_env = os.environ.get("COCH_SRC")
        self.src_dir = Path(
            src_dir if src_dir is not None else (src_env if src_env is not None else Path.cwd() / "src")
        ).resolve()

        art_env = os.environ.get("COCH_ARTIFACTS")
        self.artifacts_dir = Path(
            artifacts_dir
            if artifacts_dir is not None
            else (art_env if art_env is not None else Path.cwd() / "artifacts")
        ).resolve()

        scratch_env = os.environ.get("COCH_SCRATCH")
        slurm_env = os.environ.get("SLURM_TMPDIR")
        fallback_scratch = (
            scratch_env if scratch_env is not None else (slurm_env if slurm_env is not None else Path.cwd() / "scratch")
        )
        self.scratch_dir = Path(
            scratch_dir if scratch_dir is not None else fallback_scratch
        ).resolve()

        self.cuda_manager = cuda_manager or CudaMemoryManager(allow_cpu_fallback=False)
        self.slurm_generator = slurm_generator or SlurmDryRunGenerator()
        self.mpi_supervisor = mpi_supervisor or MpiProcessSupervisor()

        self.telemetry_lock_path = self.scratch_dir / ".telemetry.lock"
        self._active_writers: Dict[str, h5py.File] = {}
        self._writer_locks: Dict[str, filelock.FileLock] = {}
        self._telemetry_guard = threading.RLock()
        self._owned_scratch_dirs: set[Path] = set()

        # Ensure write destinations exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def validate_write_path(self, target_path: Path) -> None:
        """Enforce Tripartite architecture: strictly forbid writing into read-only $COCH_SRC."""
        resolved = target_path.resolve()
        try:
            resolved.relative_to(self.src_dir)
            is_in_src = True
        except ValueError:
            is_in_src = False

        if is_in_src or resolved == self.src_dir:
            raise PermissionError(
                f"Tripartite storage violation: Cannot write to read-only source directory ($COCH_SRC): '{resolved}'."
            )

    def init_telemetry(self, telemetry_file: Path) -> None:
        """Create or reopen telemetry without truncating existing observations.

        A per-file lock is held for the writer's lifetime. Only committed rows
        may be consumed; the availability mask distinguishes unknown energy
        from a measured/calculated zero. Legacy ambiguous files need migration.
        """
        self.validate_write_path(telemetry_file)
        telemetry_file.parent.mkdir(parents=True, exist_ok=True)
        key = str(telemetry_file.resolve())
        with self._telemetry_guard:
            if key in self._active_writers:
                return
            ownership = filelock.FileLock(key + ".writer.lock", timeout=0, thread_local=False)
            ownership.acquire()
            handle = None
            try:
                exists = telemetry_file.exists()
                handle = h5py.File(telemetry_file, "r+" if exists else "x", libver="latest")
                if exists:
                    if handle.attrs.get("telemetry_schema_version") != 2:
                        raise ValueError("Existing telemetry has no qualified availability schema; explicit migration is required.")
                    group = handle["telemetry"]
                    count = int(group["committed_rows"][()])
                    if count < 0:
                        raise ValueError("Existing telemetry has a negative committed row count.")
                    for name in ("step", "energy", "energy_available", "walltime"):
                        dataset = group[name]
                        if dataset.ndim != 1 or dataset.shape[0] < count:
                            raise ValueError("Existing telemetry has inconsistent committed rows.")
                    # An interrupted append may leave an uncommitted suffix.
                    for name in ("step", "energy", "energy_available", "walltime"):
                        group[name].resize((count,))
                else:
                    handle.attrs["telemetry_schema_version"] = 2
                    group = handle.create_group("telemetry")
                    for name, dtype in (("step", "int64"), ("energy", "float64"),
                                        ("energy_available", "bool"), ("walltime", "float64")):
                        group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)
                    group.create_dataset("committed_rows", data=0, dtype="int64")
                    group["energy"].attrs["units"] = "hartree"
                    group["energy"].attrs["missingness"] = "NaN with energy_available=false"
                    group["walltime"].attrs["units"] = "seconds"
                handle.swmr_mode = True
                handle.flush()
                self._active_writers[key] = handle
                self._writer_locks[key] = ownership
            except BaseException:
                if handle is not None:
                    handle.close()
                ownership.release()
                raise

    def close_telemetry(self, telemetry_file: Path) -> None:
        """Close a writer and release its per-file ownership lock."""
        key = str(telemetry_file.resolve())
        with self._telemetry_guard:
            handle = self._active_writers.pop(key, None)
            ownership = self._writer_locks.pop(key, None)
            try:
                if handle is not None and handle.id.valid:
                    try:
                        handle.flush()
                    finally:
                        handle.close()
            finally:
                if ownership is not None:
                    ownership.release()

    def close(self) -> None:
        for key in list(self._active_writers):
            self.close_telemetry(Path(key))

    def __enter__(self) -> "AsyncProcessRunner":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
        self.cleanup_scratch()

    def cleanup_scratch(self, task_name: Optional[str] = None) -> None:
        """Remove only scratch directories created by this runner instance."""
        for path in tuple(self._owned_scratch_dirs):
            if task_name is None or path.name == task_name:
                shutil.rmtree(path)
                self._owned_scratch_dirs.remove(path)

    def record_telemetry_metric(
        self, telemetry_file: Path, step: int, energy: Optional[float], walltime: float,
    ) -> None:
        """Publish one committed telemetry row with explicit energy availability."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Telemetry step must be a nonnegative integer.")
        if not math.isfinite(walltime) or walltime < 0:
            raise ValueError("Telemetry walltime must be finite and nonnegative.")
        if energy is not None and not math.isfinite(energy):
            raise ValueError("Available telemetry energy must be finite; use None for unavailable.")
        self.validate_write_path(telemetry_file)
        key = str(telemetry_file.resolve())
        with self._telemetry_guard:
            if key not in self._active_writers:
                self.init_telemetry(telemetry_file)
            handle = self._active_writers[key]
            group = handle["telemetry"]
            count = int(group["committed_rows"][()])
            values = {"step": step, "energy": energy if energy is not None else float("nan"),
                      "energy_available": energy is not None, "walltime": walltime}
            for name, value in values.items():
                dataset = group[name]
                dataset.resize((count + 1,))
                dataset[count] = value
                dataset.flush()
            handle.flush()
            group["committed_rows"][()] = count + 1
            group["committed_rows"].flush()
            handle.flush()

    async def dispatch_task(
        self,
        task_name: str,
        binary_args: List[str],
        gpu_required_mb: Optional[int] = None,
        device_id: Optional[int] = None,
        slurm_spec: Optional[SlurmJobDirectiveSpec] = None,
        partition_limits: Optional[Dict[str, Any]] = None,
        mpi_config: Optional[MpiClusterExecutionConfig] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        telemetry_callback: Optional[Callable[[str], None]] = None,
        rank_trace_callback: Optional[Callable[[str], None]] = None,
        cleanup_on_completion: bool = True,
    ) -> Dict[str, Any]:
        """Dispatch a high-performance computation task adhering to tripartite and cluster invariants."""
        # Setup job scratch space
        if re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", task_name) is None:
            raise ValueError("Task names must be nonempty safe filename components.")
        job_scratch = Path(tempfile.mkdtemp(prefix=task_name + "_", dir=self.scratch_dir))
        self._owned_scratch_dirs.add(job_scratch)
        execution_cwd = cwd or job_scratch

        # Prepare environment with Tripartite storage locations
        run_env = dict(os.environ) if env is None else dict(env)
        run_env["COCH_SRC"] = str(self.src_dir)
        run_env["COCH_ARTIFACTS"] = str(self.artifacts_dir)
        run_env["COCH_SCRATCH"] = str(job_scratch)
        run_env["SLURM_TMPDIR"] = str(job_scratch)

        # 1. GPU VRAM Budgeting & Environment Isolation
        if gpu_required_mb is not None and gpu_required_mb > 0:
            assigned_dev = await self.cuda_manager.schedule_gpu_task(
                required_mb=gpu_required_mb,
                requested_device_id=device_id,
            )
            run_env = self.cuda_manager.prepare_worker_environment(
                device_id=assigned_dev,
                base_env=run_env,
            )

        # 2. Slurm Preflight Validation
        slurm_result: Optional[SlurmDryRunResult] = None
        if slurm_spec is not None:
            slurm_result = self.slurm_generator.generate_sbatch_script(
                spec=slurm_spec,
                partition_limits=partition_limits,
            )
            if not slurm_result.is_valid:
                error_summary = "; ".join(slurm_result.validation_errors)
                raise SlurmResourceValidationError(
                    f"Slurm preflight validation failed for task '{task_name}': {error_summary}"
                )

        # 3. Initialize SWMR HDF5 Telemetry
        telemetry_file = self.artifacts_dir / "telemetry" / f"{job_scratch.name}.h5"
        self.init_telemetry(telemetry_file)

        try:
            # 4. Supervise Process Execution
            start_time = time.monotonic()
            if mpi_config is not None:
                config_with_env = mpi_config.model_copy(
                    update={"environment_vars": {**mpi_config.environment_vars, **run_env}}
                )
                exec_result = await self.mpi_supervisor.run_mpi_task(
                    config=config_with_env,
                    binary_args=binary_args,
                    cwd=execution_cwd,
                    telemetry_callback=telemetry_callback,
                    rank_trace_callback=rank_trace_callback,
                )
            else:
                exec_result = await self.mpi_supervisor.run_command(
                    command=binary_args,
                    cwd=execution_cwd,
                    env=run_env,
                    telemetry_callback=telemetry_callback,
                    rank_trace_callback=rank_trace_callback,
                )

            elapsed = time.monotonic() - start_time

            # 5. Record Completion Metric to Telemetry
            self.record_telemetry_metric(
                telemetry_file=telemetry_file,
                step=1,
                energy=None,
                walltime=elapsed,
            )

        finally:
            self.close_telemetry(telemetry_file)

        if cleanup_on_completion:
            self.cleanup_scratch(task_name=job_scratch.name)

        return {
            "task_name": task_name,
            "exit_code": exec_result["exit_code"],
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
            "telemetry_file": str(telemetry_file),
            "scratch_dir": str(job_scratch),
            "artifacts_dir": str(self.artifacts_dir),
            "slurm_result": slurm_result,
            "elapsed_seconds": elapsed,
            "energy_hartree": None,
            "energy_status": "unavailable",
            "scientific_validation": "not_performed_by_process_runner",
        }
