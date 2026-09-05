"""# zero-stub anti-spoofing engine
Zero-Trust Quarantine Runner Module (ci_tools/zero_trust_runner.py)

Creates sterile, ephemeral execution directories (/tmp/cochem_exec_<uuid>/ or %TEMP%/cochem_exec_<uuid>)
for isolated execution, tests, and verification. Manages child process lifetimes with psutil-based
zombie process sweeping.

Complies with Method Matrix v4, CoChem Anti-Spoofing Protocol v2, and WBS Task 1.2.2.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("zero_trust_runner")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class QuarantineResult:
    """Result of a command executed within the sterile quarantine environment."""
    exit_code: int
    stdout: str
    stderr: str
    quarantine_dir: str
    duration_s: float
    passed: bool
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "quarantine_dir": self.quarantine_dir,
            "duration_s": self.duration_s,
            "passed": self.passed,
            "timed_out": self.timed_out,
        }


def sweep_zombie_processes() -> int:
    """Safely terminate orphaned and zombie child processes using psutil."""
    if not psutil:
        return 0

    current_pid = os.getpid()
    terminated_count = 0
    try:
        parent = psutil.Process(current_pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
                terminated_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if children:
            _, alive = psutil.wait_procs(children, timeout=3.0)
            for proc in alive:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return terminated_count


# Register automatic cleanup on process exit
atexit.register(sweep_zombie_processes)


class QuarantineEnvironment:
    """Context manager providing an isolated, ephemeral execution directory."""

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        prefix: str = "cochem_exec_",
        copy_paths: Optional[Sequence[Union[str, Path]]] = None,
        preserve_on_failure: bool = False,
    ) -> None:
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            self.base_dir = Path(r"d:\__CoChem").resolve()

        self.prefix = prefix
        self.copy_paths = [Path(p).resolve() for p in copy_paths] if copy_paths else []
        self.preserve_on_failure = preserve_on_failure
        self.quarantine_id = str(uuid.uuid4())
        self.quarantine_dir = self.base_dir / f"{self.prefix}{self.quarantine_id}"

    def __enter__(self) -> "QuarantineEnvironment":
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        for src_path in self.copy_paths:
            if src_path.exists():
                dest_path = self.quarantine_dir / src_path.name
                if src_path.is_file():
                    shutil.copy2(src_path, dest_path)
                elif src_path.is_dir():
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None and self.preserve_on_failure:
            logger.warning(f"Preserving failed quarantine directory: {self.quarantine_dir}")
        else:
            shutil.rmtree(self.quarantine_dir, ignore_errors=True)
        sweep_zombie_processes()

    def run_command(
        self,
        command: Sequence[str],
        timeout: int = 300,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> QuarantineResult:
        """Execute a command strictly inside the quarantine directory."""
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        root_env = os.environ.get("COCHEM_ROOT")
        if root_env:
            cwd_path = Path(root_env).resolve()
            env["PYTHONPATH"] = str(cwd_path)

        start_time = time.monotonic()
        timed_out = False

        try:
            result = subprocess.run(
                list(command),
                cwd=str(self.quarantine_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.monotonic() - start_time
            passed = (result.returncode == 0)
            return QuarantineResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                quarantine_dir=str(self.quarantine_dir),
                duration_s=round(duration, 4),
                passed=passed,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as e:
            duration = time.monotonic() - start_time
            logger.error(f"Quarantine command timed out after {timeout} seconds: {e}")
            return QuarantineResult(
                exit_code=124,
                stdout=e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr=e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or f"TimeoutExpired: {e}"),
                quarantine_dir=str(self.quarantine_dir),
                duration_s=round(duration, 4),
                passed=False,
                timed_out=True,
            )
        except Exception as e:
            duration = time.monotonic() - start_time
            logger.error(f"Quarantine execution failed with error: {e}")
            return QuarantineResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                quarantine_dir=str(self.quarantine_dir),
                duration_s=round(duration, 4),
                passed=False,
                timed_out=False,
            )


def run_in_quarantine(
    command: Sequence[str],
    timeout: int = 300,
    env_overrides: Optional[Dict[str, str]] = None,
    copy_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> QuarantineResult:
    """Convenience function to run a command inside a fresh quarantine directory."""
    with QuarantineEnvironment(copy_paths=copy_paths) as qe:
        return qe.run_command(command, timeout=timeout, env_overrides=env_overrides)


def main() -> int:
    import argparse
    import re
    import hashlib
    
    parser = argparse.ArgumentParser(description="Zero-Trust Quarantine Runner")
    parser.add_argument("--nonce", type=str, help="Cryptographic nonce for ExecutionReceipt", default="UNKNOWN_NONCE")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run in quarantine")
    args = parser.parse_args()

    if not args.command:
        logger.error("Usage: zero_trust_runner.py [--nonce NONCE] <command...>")
        return 1
        
    command = args.command
    if command[0] == "--":
        command = command[1:]

    cwd = Path.cwd().resolve()
    
    with QuarantineEnvironment() as qe:
        # 1. Enforce copy of current repository to quarantine directory
        shutil.copytree(cwd, qe.quarantine_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".agent_artifacts"))
        
        # 2. PathCanonicalRewriter: Regex-based case-insensitive path replacement
        new_command = []
        cwd_str = str(cwd).replace("\\", "\\\\")
        # Regex to match the cwd path (case insensitive) anywhere in the argument
        cwd_pattern = re.compile(re.escape(str(cwd)), re.IGNORECASE)
        cwd_forward_pattern = re.compile(re.escape(str(cwd).replace("\\", "/")), re.IGNORECASE)
        
        for arg in command:
            new_arg = cwd_pattern.sub(str(qe.quarantine_dir).replace("\\", "\\\\"), arg)
            new_arg = cwd_forward_pattern.sub(str(qe.quarantine_dir).replace("\\", "/"), new_arg)
            new_command.append(new_arg)
                
        # 3. Set PYTHONPATH and COCHEM_ROOT strictly to quarantine directory
        extra_paths = [str(qe.quarantine_dir)]
        ci_tools_dir = Path(__file__).resolve().parent
        if ci_tools_dir.exists():
            extra_paths.append(str(ci_tools_dir))
        for sibling_name in ("CoChem-BASE", "CoChem-KINETIC"):
            sibling_dir = Path("d:/__CoChem/GitHub-Repo") / sibling_name
            if sibling_dir.exists():
                extra_paths.append(str(sibling_dir))

        env_overrides = {
            "PYTHONPATH": os.pathsep.join(extra_paths),
            "COCHEM_ROOT": str(qe.quarantine_dir)
        }
        
        res = qe.run_command(new_command, timeout=300, env_overrides=env_overrides)

    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
        
    # Generate Cryptographic ExecutionReceipt
    stdout_hash = hashlib.sha256((res.stdout or "").encode('utf-8')).hexdigest()
    stderr_hash = hashlib.sha256((res.stderr or "").encode('utf-8')).hexdigest()
    receipt = f"\n=== EXECUTION RECEIPT ===\nNONCE: {args.nonce}\nEXIT_CODE: {res.exit_code}\nDURATION: {res.duration_s}s\nSTDOUT_HASH: {stdout_hash}\nSTDERR_HASH: {stderr_hash}\n===========================\n"
    print(receipt)
    
    return res.exit_code

if __name__ == "__main__":
    sys.exit(main())
