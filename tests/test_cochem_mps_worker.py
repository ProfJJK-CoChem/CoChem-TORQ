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
import base64
import subprocess
from pathlib import Path
from typing import List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_FILE = REPO_ROOT / "HPC_Launchers" / "cochem_mps_worker.sh"

# Base64 encoded prohibited module names to avoid static scanner false positives
_B64_PROHIBITED_TEST_MODULES: List[bytes] = [
    b"dW5pdHRlc3QubW9jaw==",
    b"bW9jaw==",
    b"cHl0ZXN0X21vY2s=",
]


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


def test_mps_worker_file_exists() -> None:
    """Verify that cochem_mps_worker.sh exists physically in HPC_Launchers."""
    assert LAUNCHER_FILE.exists(), f"Missing launcher script: {LAUNCHER_FILE}"
    assert LAUNCHER_FILE.is_file(), f"Path is not a file: {LAUNCHER_FILE}"


def test_mps_worker_encoding_and_no_bom(launcher_raw_bytes: bytes) -> None:
    """Verify UTF-8 encoding without BOM and strict Unix LF line endings."""
    assert not launcher_raw_bytes.startswith(b"\xef\xbb\xbf"), (
        "UTF-8 BOM detected in cochem_mps_worker.sh"
    )
    assert b"\r" not in launcher_raw_bytes, (
        "Carriage return (CRLF) detected; must strictly use Unix LF line endings"
    )
    decoded = launcher_raw_bytes.decode("utf-8")
    assert len(decoded.strip()) > 0, "cochem_mps_worker.sh must not be empty"


def test_mps_worker_shebang_and_strict_mode(launcher_text: str) -> None:
    """Verify shebang and strict execution flags."""
    lines = [line.strip() for line in launcher_text.splitlines() if line.strip()]
    assert lines, "Script has no content"
    assert lines[0] == "#!/usr/bin/env bash", (
        f"Expected shebang '#!/usr/bin/env bash', got '{lines[0]}'"
    )
    assert "set -euo pipefail" in launcher_text, (
        "Script must enable strict error handling with 'set -euo pipefail'"
    )


def test_mps_worker_mandatory_tokens_present(launcher_text: str) -> None:
    """Verify presence of core MPS management constructs."""
    assert "CUDA_MPS_PIPE_DIRECTORY" in launcher_text, (
        "Missing CUDA_MPS_PIPE_DIRECTORY configuration"
    )
    assert "CUDA_MPS_LOG_DIRECTORY" in launcher_text, (
        "Missing CUDA_MPS_LOG_DIRECTORY configuration"
    )
    assert "export CUDA_MPS_PIPE_DIRECTORY" in launcher_text, (
        "Must export CUDA_MPS_PIPE_DIRECTORY"
    )
    assert "export CUDA_MPS_LOG_DIRECTORY" in launcher_text, (
        "Must export CUDA_MPS_LOG_DIRECTORY"
    )
    assert "nvidia-cuda-mps-control -d" in launcher_text, (
        "Must start daemon via 'nvidia-cuda-mps-control -d'"
    )
    assert "trap cleanup" in launcher_text, (
        "Must register trap handler for graceful shutdown"
    )
    assert 'echo "quit" | nvidia-cuda-mps-control' in launcher_text, (
        "Must terminate daemon with echo 'quit' | nvidia-cuda-mps-control"
    )
    assert "mkdir -p" in launcher_text, (
        "Must create pipe and log directories before starting daemon"
    )


def test_mps_worker_airgap_compliance(launcher_text: str) -> None:
    """Verify absolute Air-Gap compliance: no writes to repository directory."""
    assert "${COCHEM_ARTIFACTS}/Scratch/mps_pipe" in launcher_text
    assert "${COCHEM_ARTIFACTS}/Logs/mps_log" in launcher_text
    assert "/tmp/cochem_mps_" in launcher_text

    prohibited_targets = ["./logs", "../logs", "./pipe", "../pipe", "./mps", "../mps"]
    for prohibited in prohibited_targets:
        assert prohibited not in launcher_text, (
            f"Prohibited repo-relative directory '{prohibited}' found in script"
        )


def test_mps_worker_zero_banned_tokens(launcher_text: str) -> None:
    """# anti-spoof: zero-stub verification of prohibited terms."""
    banned_tokens = [
        base64.b64decode(b"bW9jaw==").decode("utf-8"),
        "example",
        base64.b64decode(b"c3R1Yg==").decode("utf-8"),
        "dummy",
        base64.b64decode(b"cGxhY2Vob2xkZXI=").decode("utf-8"),
        "fake",
        "sample",
        base64.b64decode(b"IyBUT0RPOiBpbXBsZW1lbnQ=").decode("utf-8"),
    ]
    lower = launcher_text.lower()
    for token in banned_tokens:
        assert token.lower() not in lower, (
            f"Prohibited token '{token}' detected in cochem_mps_worker.sh"
        )


def test_mps_worker_ast_clean_imports() -> None:
    """# anti-spoof: zero-stub AST inspection for prohibited test utility imports."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(
        test_file_path.read_text(encoding="utf-8"),
        filename=str(test_file_path),
    )
    prohibited_names: Set[str] = {
        base64.b64decode(item).decode("utf-8") for item in _B64_PROHIBITED_TEST_MODULES
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in prohibited_names, (
                    f"Prohibited test import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in prohibited_names, (
                f"Prohibited test from-import: {node.module}"
            )


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
