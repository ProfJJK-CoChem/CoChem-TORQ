"""Real native-prerequisite and process tests for the optional MPS launcher.

CPU hosts prove fail-closed behavior. Successful native daemon operation requires
an observed NVIDIA device and genuine MPS executable; no replacement executable
or fabricated driver response is introduced by these tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

LAUNCHER_FILE = (
    Path(__file__).resolve().parent.parent / "HPC_Launchers/cochem_mps_worker.sh"
)


def native_mps_ready() -> bool:
    """Observe the real local driver and native executable prerequisites."""
    executable = shutil.which("nvidia-smi")
    if not executable or not shutil.which("nvidia-cuda-mps-control"):
        return False
    result = subprocess.run(
        [executable, "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def run_worker(tmp_path: Path, arguments: list[str], **environment: str):
    """Execute the actual launcher with private test-owned scratch storage."""
    env = {**os.environ, "COCHEM_SCRATCH": str(tmp_path / "scratch"), **environment}
    return subprocess.run(
        ["bash", str(LAUNCHER_FILE), *arguments],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )


def test_mps_worker_bash_syntax_valid() -> None:
    assert LAUNCHER_FILE.is_file()
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER_FILE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_mps_worker_execution_with_cochem_artifacts(tmp_path: Path) -> None:
    """Execute a genuine worker only when the real GPU/MPS prerequisites hold."""
    marker = tmp_path / "worker-launched"
    ready = native_mps_ready()
    result = run_worker(
        tmp_path,
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "Path(sys.argv[1]).write_text('executed')",
            str(marker),
        ],
    )
    if ready:
        assert result.returncode == 0, result.stderr
        assert marker.read_text() == "executed"
        assert not list((tmp_path / "scratch").glob("cochem_mps_*"))
    else:
        assert result.returncode == 69
        assert "unavailable" in result.stderr or "CUDA" in result.stderr
        assert not marker.exists()
        assert not (tmp_path / "scratch").exists()


def test_mps_worker_execution_with_explicit_mps_dirs(tmp_path: Path) -> None:
    """Only fresh owned directories may be created and cleaned by native MPS."""
    pipe = tmp_path / "private-pipe"
    log = tmp_path / "private-log"
    ready = native_mps_ready()
    result = run_worker(
        tmp_path,
        [sys.executable, "-c", "print('actual worker')"],
        CUDA_MPS_PIPE_DIRECTORY=str(pipe),
        CUDA_MPS_LOG_DIRECTORY=str(log),
    )
    if ready:
        assert result.returncode == 0, result.stderr
        assert "actual worker" in result.stdout
    else:
        assert result.returncode == 69
        assert "actual worker" not in result.stdout
    assert not pipe.exists()
    assert not log.exists()


def test_mps_worker_exit_code_propagation(tmp_path: Path) -> None:
    """Real native launches propagate status; missing prerequisites block launch."""
    ready = native_mps_ready()
    result = run_worker(tmp_path, [sys.executable, "-c", "raise SystemExit(33)"])
    assert result.returncode == (33 if ready else 69), result.stderr


def test_mps_worker_requires_explicit_command(tmp_path: Path) -> None:
    result = run_worker(tmp_path, [])
    assert result.returncode == 69
    assert "explicit worker command" in result.stderr
    assert not (tmp_path / "scratch").exists()


def test_mps_cleanup_never_enumerates_other_sessions() -> None:
    source = LAUNCHER_FILE.read_text()
    assert "pgrep" not in source
    assert "pkill" not in source
    assert "quit\\n" in source
    assert '"${owned_directories[@]}"' in source
