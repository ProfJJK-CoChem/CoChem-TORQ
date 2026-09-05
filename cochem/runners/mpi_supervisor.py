"""Asynchronous multi-node MPI process supervisor.

Provides stream multiplexing, rank-0 telemetry segregation, and cross-platform
process tree lifecycle management (POSIX process groups and Windows CREATE_NEW_PROCESS_GROUP).
"""

import asyncio
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.cochem.hpc.models import MpiClusterExecutionConfig, MpiProcessSupervisorError

logger = logging.getLogger(__name__)


class MpiProcessSupervisor:
    """Supervises multi-rank MPI executions with non-blocking stream multiplexing."""

    CHUNK_SIZE: int = 65536  # 64 KB stream chunk buffer to prevent OS pipe deadlocks

    def __init__(self, grace_period_seconds: float = 15.0) -> None:
        self.grace_period_seconds = grace_period_seconds
        self._rank_pattern = re.compile(r"^\[(?:rank\s*)?(\d+)\]", re.IGNORECASE)

    def build_mpi_command(
        self,
        config: MpiClusterExecutionConfig,
        binary_args: List[str],
    ) -> List[str]:
        """Construct launcher-specific MPI invocation arguments."""
        total_ranks = config.n_nodes * config.n_tasks_per_node
        launcher = config.launcher.lower().strip()

        if launcher == "srun":
            cmd = [
                "srun",
                f"--nodes={config.n_nodes}",
                f"--ntasks-per-node={config.n_tasks_per_node}",
                f"--cpus-per-task={config.cpus_per_task}",
                "--mpi=pmi2",
            ]
        elif launcher == "mpirun":
            cmd = [
                "mpirun",
                "-np",
                str(total_ranks),
                "-N",
                str(config.n_tasks_per_node),
                "--bind-to",
                "core",
            ]
        elif launcher == "mpiexec":
            cmd = [
                "mpiexec",
                "-n",
                str(total_ranks),
                "-ppn",
                str(config.n_tasks_per_node),
            ]
        else:
            raise MpiProcessSupervisorError(f"Unsupported MPI launcher designated: '{launcher}'.")

        cmd.extend(binary_args)
        return cmd

    async def _read_stream(
        self,
        stream: Optional[asyncio.StreamReader],
        on_line: Callable[[str], None],
        accumulator: List[str],
    ) -> None:
        """Read stream asynchronously in 64 KB chunks to prevent pipe buffer saturation."""
        if stream is None:
            return

        buffered = ""
        while True:
            chunk = await stream.read(self.CHUNK_SIZE)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            buffered += text
            lines = buffered.splitlines(keepends=True)
            # Retain trailing incomplete segment if chunk ended mid-line
            if lines and not lines[-1].endswith(("\n", "\r")):
                buffered = lines.pop()
            else:
                buffered = ""

            for line in lines:
                clean_line = line.rstrip("\r\n")
                accumulator.append(clean_line)
                on_line(clean_line)

        if buffered:
            clean_line = buffered.rstrip("\r\n")
            accumulator.append(clean_line)
            on_line(clean_line)

    async def _terminate_process_tree(self, proc: asyncio.subprocess.Process) -> None:
        """Terminate process tree cross-platform using signal propagation or forced kill."""
        pid = proc.pid
        is_windows = sys.platform == "win32"

        try:
            if is_windows:
                # Send CTRL_BREAK_EVENT to process group
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # POSIX process group signal
                killpg = getattr(os, "killpg", None)
                getpgid = getattr(os, "getpgid", None)
                if killpg is not None and getpgid is not None:
                    killpg(getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            logger.debug("Initial termination signal to process %s failed non-fatally: %s", pid, exc)

        # Wait during grace period
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.grace_period_seconds)
            return
        except asyncio.TimeoutError:
            logger.debug("Process %s exceeded grace period; escalating to forced kill", pid)

        # Escalate to forceful kill
        try:
            if is_windows:
                # Execute taskkill for complete process tree destruction
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                killpg = getattr(os, "killpg", None)
                getpgid = getattr(os, "getpgid", None)
                sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
                if killpg is not None and getpgid is not None:
                    killpg(getpgid(pid), sigkill)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            logger.debug("Forceful kill of process %s encountered non-fatal error: %s", pid, exc)

        try:
            await proc.wait()
        except Exception as exc:
            logger.debug("Final wait on process %s encountered non-fatal error: %s", pid, exc)

    async def run_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        timeout_seconds: float = 3600.0,
        env: Optional[Dict[str, str]] = None,
        telemetry_callback: Optional[Callable[[str], None]] = None,
        rank_trace_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute command asynchronously with stream segregation and process group safety."""
        is_windows = sys.platform == "win32"
        extra_kwargs: Dict[str, Any] = {}

        if is_windows:
            extra_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            setsid = getattr(os, "setsid", None)
            if setsid is not None:
                extra_kwargs["preexec_fn"] = setsid

        run_env = dict(os.environ) if env is None else dict(env)
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        rank_trace_lines: List[str] = []

        def handle_stdout_line(line: str) -> None:
            match = self._rank_pattern.match(line)
            if match:
                rank_id = int(match.group(1))
                if rank_id == 0:
                    if telemetry_callback:
                        telemetry_callback(line)
                else:
                    rank_trace_lines.append(line)
                    if rank_trace_callback:
                        rank_trace_callback(line)
            else:
                # Default non-prefixed stdout belongs to primary rank 0 telemetry
                if telemetry_callback:
                    telemetry_callback(line)

        def handle_stderr_line(line: str) -> None:
            stderr_lines.append(line)

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            env=run_env,
            **extra_kwargs,
        )

        stdout_task = asyncio.create_task(
            self._read_stream(proc.stdout, handle_stdout_line, stdout_lines)
        )
        stderr_task = asyncio.create_task(
            self._read_stream(proc.stderr, handle_stderr_line, stderr_lines)
        )

        try:
            await asyncio.wait_for(
                asyncio.gather(proc.wait(), stdout_task, stderr_task),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            await self._terminate_process_tree(proc)
            raise MpiProcessSupervisorError(
                f"Process execution timed out after {timeout_seconds:.2f} seconds: {' '.join(command)}"
            ) from exc
        except asyncio.CancelledError:
            await self._terminate_process_tree(proc)
            raise

        exit_code = proc.returncode
        if exit_code != 0:
            err_msg = "\n".join(stderr_lines) if stderr_lines else "\n".join(stdout_lines[-10:])
            raise MpiProcessSupervisorError(
                f"Process exited with non-zero return code {exit_code}: {err_msg}"
            )

        return {
            "exit_code": exit_code,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "rank_traces": rank_trace_lines,
        }

    async def run_mpi_task(
        self,
        config: MpiClusterExecutionConfig,
        binary_args: List[str],
        cwd: Optional[Path] = None,
        telemetry_callback: Optional[Callable[[str], None]] = None,
        rank_trace_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Build MPI arguments and supervise execution across designated nodes."""
        mpi_cmd = self.build_mpi_command(config, binary_args)
        return await self.run_command(
            command=mpi_cmd,
            cwd=cwd,
            timeout_seconds=config.timeout_seconds,
            env=config.environment_vars if config.environment_vars else None,
            telemetry_callback=telemetry_callback,
            rank_trace_callback=rank_trace_callback,
        )
