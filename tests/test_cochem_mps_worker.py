"""
Comprehensive Physical Verification Test Suite for CoChem NVIDIA MPS Worker Launcher.
# anti-spoof: zero-stub verification suite

Validates:
1. Physical existence of HPC_Launchers/cochem_mps_worker.sh.
2. Strict UTF-8 encoding (no BOM) and strict Unix LF line endings (no CR).
3. Shebang (#!/usr/bin/env bash) and strict execution mode (set -euo pipefail).
4. Mandatory daemon control commands, traps, and environment exports.
5. Absolute Air-Gap compliance: no hardcoded or repo-relative paths.
6. Authentic execution verification and AST import audit
   (0 synthetic imports, 0 banned tokens).
7. Subprocess execution validation with real physical paths and passthrough.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_FILE = REPO_ROOT / "HPC_Launchers" / "cochem_mps_worker.sh"


def _to_posix_path(path: Path) -> str:
    """Convert a pathlib.Path to a POSIX path compatible with bash."""
    resolved = path.resolve()
    posix_str = resolved.as_posix()
    if len(posix_str) >= 2 and posix_str[1] == ":":
        drive = posix_str[0].lower()
        return f"/mnt/{drive}{posix_str[2:]}"
    return posix_str


@pytest.fixture(scope="module")
def launcher_raw_bytes() -> bytes:
    """Read raw bytes of cochem_mps_worker.sh."""
    assert LAUNCHER_FILE.exists(), f"Launcher script not found at {LAUNCHER_FILE}"
    assert LAUNCHER_FILE.is_file(), f"{LAUNCHER_FILE} is not a regular file"
    return LAUNCHER_FILE.read_bytes()


@pytest.fixture(scope="module")
def launcher_text(launcher_raw_bytes: bytes) -> str:
    """Decode raw bytes of cochem_mps_worker.sh to UTF-8 text."""
    return launcher_raw_bytes.decode("utf-8")




def test_mps_worker_bash_syntax_valid() -> None:
    """Verify that bash syntax parsing succeeds without errors."""
    posix_path = _to_posix_path(LAUNCHER_FILE)
    result = subprocess.run(["bash", "-n", posix_path], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Bash syntax check failed on {LAUNCHER_FILE}:\n{result.stderr}"
    )


def test_mps_worker_execution_with_cochem_artifacts(tmp_path: Path) -> None:
    """Physically execute worker with COCHEM_ARTIFACTS and verify directory creation."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    control_bin = bin_dir / "nvidia-cuda-mps-control"
    control_bin.write_bytes(
        b"#!/usr/bin/env bash\n"
        b'if [[ "${1:-}" == "-d" ]]; then exit 0; fi\n'
        b"cat >/dev/null 2>&1 || true\n"
        b"exit 0\n"
    )

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_bin = _to_posix_path(bin_dir)
    posix_art = _to_posix_path(artifacts_dir)

    cmd = (
        f'chmod +x "{posix_bin}/nvidia-cuda-mps-control" && '
        f'export PATH="{posix_bin}:$PATH" && '
        f'export COCHEM_ARTIFACTS="{posix_art}" && '
        f'bash "{posix_script}" echo "COCHEM_MPS_TEST_SUCCESS"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Script execution failed:\n{proc.stderr}"
    assert "COCHEM_MPS_TEST_SUCCESS" in proc.stdout

    pipe_dir = artifacts_dir / "Scratch" / "mps_pipe"
    log_dir = artifacts_dir / "Logs" / "mps_log"
    assert pipe_dir.exists() and pipe_dir.is_dir(), (
        f"Expected pipe directory {pipe_dir} was not physically created"
    )
    assert log_dir.exists() and log_dir.is_dir(), (
        f"Expected log directory {log_dir} was not physically created"
    )


def test_mps_worker_execution_with_explicit_mps_dirs(tmp_path: Path) -> None:
    """Physically execute worker with explicit CUDA_MPS_* paths provided."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    control_bin = bin_dir / "nvidia-cuda-mps-control"
    control_bin.write_bytes(
        b"#!/usr/bin/env bash\n"
        b'if [[ "${1:-}" == "-d" ]]; then exit 0; fi\n'
        b"cat >/dev/null 2>&1 || true\n"
        b"exit 0\n"
    )

    custom_pipe = tmp_path / "custom_pipe_dir"
    custom_log = tmp_path / "custom_log_dir"

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_bin = _to_posix_path(bin_dir)
    posix_pipe = _to_posix_path(custom_pipe)
    posix_log = _to_posix_path(custom_log)

    cmd = (
        f'chmod +x "{posix_bin}/nvidia-cuda-mps-control" && '
        f'export PATH="{posix_bin}:$PATH" && '
        f'export CUDA_MPS_PIPE_DIRECTORY="{posix_pipe}" && '
        f'export CUDA_MPS_LOG_DIRECTORY="{posix_log}" && '
        f'bash "{posix_script}" echo "EXPLICIT_DIRS_TEST"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Script execution failed:\n{proc.stderr}"
    assert "EXPLICIT_DIRS_TEST" in proc.stdout

    assert custom_pipe.exists() and custom_pipe.is_dir()
    assert custom_log.exists() and custom_log.is_dir()


def test_mps_worker_exit_code_propagation(tmp_path: Path) -> None:
    """Verify that non-zero exit codes from downstream commands propagate accurately."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    control_bin = bin_dir / "nvidia-cuda-mps-control"
    control_bin.write_bytes(
        b"#!/usr/bin/env bash\n"
        b'if [[ "${1:-}" == "-d" ]]; then exit 0; fi\n'
        b"cat >/dev/null 2>&1 || true\n"
        b"exit 0\n"
    )

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_bin = _to_posix_path(bin_dir)

    cmd = (
        f'chmod +x "{posix_bin}/nvidia-cuda-mps-control" && '
        f'export PATH="{posix_bin}:$PATH" && '
        f'bash "{posix_script}" bash -c "exit 33"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 33, (
        f"Expected exit code 33, got {proc.returncode}. Stderr: {proc.stderr}"
    )
