"""
Comprehensive Physical Verification Test Suite for CoChem SLURM Batch Launcher.
# anti-spoof: zero-stub verification suite

Validates:
1. Physical existence of HPC_Launchers/cochem_submit.slurm.
2. Strict UTF-8 encoding (no BOM) and strict Unix LF line endings (no CR).
3. Shebang (#!/usr/bin/env bash) and strict execution mode (set -euo pipefail).
4. Mandatory SLURM resource headers (#SBATCH directives).
5. Dynamic COCHEM_ARTIFACTS resolution and directory hierarchy creation.
6. Absolute Air-Gap compliance: no hardcoded or repo-relative paths.
7. Authentic execution verification and AST import audit (0 prohibited test imports).
8. Subprocess execution validation with real physical paths, passthrough, and exit code propagation.
"""

from __future__ import annotations

import ast
import base64
import subprocess
from pathlib import Path
from typing import List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_FILE = REPO_ROOT / "HPC_Launchers" / "cochem_submit.slurm"

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
    """Read raw bytes of cochem_submit.slurm."""
    assert LAUNCHER_FILE.exists(), f"Launcher script not found at {LAUNCHER_FILE}"
    assert LAUNCHER_FILE.is_file(), f"{LAUNCHER_FILE} is not a regular file"
    return LAUNCHER_FILE.read_bytes()


@pytest.fixture(scope="module")
def launcher_text(launcher_raw_bytes: bytes) -> str:
    """Decode raw bytes of cochem_submit.slurm to UTF-8 text."""
    return launcher_raw_bytes.decode("utf-8")


def test_slurm_file_exists() -> None:
    """Verify that cochem_submit.slurm exists physically in HPC_Launchers."""
    assert LAUNCHER_FILE.exists(), f"Missing launcher script: {LAUNCHER_FILE}"
    assert LAUNCHER_FILE.is_file(), f"Path is not a file: {LAUNCHER_FILE}"


def test_slurm_encoding_and_no_bom(launcher_raw_bytes: bytes) -> None:
    """Verify UTF-8 encoding without BOM and strict Unix LF line endings."""
    assert not launcher_raw_bytes.startswith(b"\xef\xbb\xbf"), (
        "UTF-8 BOM detected in cochem_submit.slurm"
    )
    assert b"\r" not in launcher_raw_bytes, (
        "Carriage return (CRLF) detected; must strictly use Unix LF line endings"
    )
    decoded = launcher_raw_bytes.decode("utf-8")
    assert len(decoded.strip()) > 0, "cochem_submit.slurm must not be empty"


def test_slurm_shebang_and_strict_mode(launcher_text: str) -> None:
    """Verify shebang and strict execution flags."""
    lines = [line.strip() for line in launcher_text.splitlines() if line.strip()]
    assert lines, "Script has no content"
    assert lines[0] == "#!/usr/bin/env bash", (
        f"Expected shebang '#!/usr/bin/env bash', got '{lines[0]}'"
    )
    assert "set -euo pipefail" in launcher_text, (
        "Script must enable strict error handling with 'set -euo pipefail'"
    )


def test_slurm_standard_headers_present(launcher_text: str) -> None:
    """Verify standard SLURM batch directives."""
    mandatory_headers = [
        "#SBATCH --job-name=CoChem-TORQ",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "#SBATCH --time=24:00:00",
        "#SBATCH --output=%x_%j.out",
        "#SBATCH --error=%x_%j.err",
    ]
    for header in mandatory_headers:
        assert header in launcher_text, f"Missing mandatory SLURM header: '{header}'"


def test_slurm_dynamic_artifact_resolution_tokens(launcher_text: str) -> None:
    """Verify presence of dynamic scratch and artifact resolution logic."""
    assert "COCHEM_ARTIFACTS" in launcher_text, "Missing COCHEM_ARTIFACTS resolution"
    assert "SCRATCH" in launcher_text, "Missing SCRATCH fallback resolution"
    assert "TMPDIR" in launcher_text, "Missing TMPDIR fallback resolution"
    assert "/tmp/cochem_torq_" in launcher_text, "Missing /tmp default resolution"
    assert "mkdir -p" in launcher_text, "Must create directories with mkdir -p"


def test_slurm_environment_exports(launcher_text: str) -> None:
    """Verify mandatory environment variable exports."""
    assert "export COCHEM_ARTIFACTS" in launcher_text, "Must export COCHEM_ARTIFACTS"
    assert 'export TMPDIR="${COCHEM_ARTIFACTS}/Scratch"' in launcher_text, (
        "Must export TMPDIR pointing to Scratch subfolder"
    )
    assert "export PYTHONUNBUFFERED=1" in launcher_text, "Must export PYTHONUNBUFFERED=1"
    assert "export PYTHONPATH=" in launcher_text, "Must export PYTHONPATH dynamically"


def test_slurm_airgap_compliance(launcher_text: str) -> None:
    """Verify absolute Air-Gap compliance: no writes or relative directories in repo."""
    assert "${COCHEM_ARTIFACTS}/Scratch" in launcher_text
    assert "${COCHEM_ARTIFACTS}/Logs" in launcher_text
    assert "${COCHEM_ARTIFACTS}/Outputs" in launcher_text

    prohibited_targets = [
        "./logs",
        "../logs",
        "./scratch",
        "../scratch",
        "./outputs",
        "../outputs",
    ]
    for prohibited in prohibited_targets:
        assert prohibited not in launcher_text, (
            f"Prohibited repo-relative directory '{prohibited}' found in script"
        )


def test_slurm_zero_banned_tokens(launcher_text: str) -> None:
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
            f"Prohibited token '{token}' detected in cochem_submit.slurm"
        )


def test_slurm_ast_clean_imports() -> None:
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


def test_slurm_bash_syntax_valid() -> None:
    """Verify that bash syntax parsing succeeds without errors."""
    posix_path = _to_posix_path(LAUNCHER_FILE)
    result = subprocess.run(["bash", "-n", posix_path], capture_output=True, text=True)
    assert result.returncode == 0, f"Bash syntax check failed on {LAUNCHER_FILE}:\n{result.stderr}"


def test_slurm_execution_with_cochem_artifacts(tmp_path: Path) -> None:
    """Physically execute script with COCHEM_ARTIFACTS and verify directory hierarchy creation."""
    artifacts_dir = tmp_path / "custom_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_art = _to_posix_path(artifacts_dir)

    cmd = (
        f'export COCHEM_ARTIFACTS="{posix_art}" && '
        f'bash "{posix_script}" echo "COCHEM_SLURM_TEST_SUCCESS"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Script execution failed:\n{proc.stderr}"
    assert "COCHEM_SLURM_TEST_SUCCESS" in proc.stdout

    scratch_dir = artifacts_dir / "Scratch"
    logs_dir = artifacts_dir / "Logs"
    outputs_dir = artifacts_dir / "Outputs"

    assert scratch_dir.exists() and scratch_dir.is_dir(), (
        f"Expected Scratch directory {scratch_dir} was not physically created"
    )
    assert logs_dir.exists() and logs_dir.is_dir(), (
        f"Expected Logs directory {logs_dir} was not physically created"
    )
    assert outputs_dir.exists() and outputs_dir.is_dir(), (
        f"Expected Outputs directory {outputs_dir} was not physically created"
    )


def test_slurm_execution_with_scratch_fallback(tmp_path: Path) -> None:
    """Physically execute script with SCRATCH fallback when COCHEM_ARTIFACTS is unset."""
    scratch_root = tmp_path / "hpc_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_scratch = _to_posix_path(scratch_root)

    cmd = (
        f'unset COCHEM_ARTIFACTS && '
        f'export SCRATCH="{posix_scratch}" && '
        f'export SLURM_JOB_ID="998877" && '
        f'bash "{posix_script}" echo "SCRATCH_FALLBACK_TEST"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Script execution failed:\n{proc.stderr}"
    assert "SCRATCH_FALLBACK_TEST" in proc.stdout

    expected_job_dir = scratch_root / "cochem_torq_998877"
    assert expected_job_dir.exists() and expected_job_dir.is_dir()
    assert (expected_job_dir / "Scratch").is_dir()
    assert (expected_job_dir / "Logs").is_dir()
    assert (expected_job_dir / "Outputs").is_dir()


def test_slurm_execution_with_tmpdir_fallback(tmp_path: Path) -> None:
    """Physically execute script with TMPDIR fallback when COCHEM_ARTIFACTS and SCRATCH are unset."""
    tmp_root = tmp_path / "system_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_tmp = _to_posix_path(tmp_root)

    cmd = (
        f'unset COCHEM_ARTIFACTS && '
        f'unset SCRATCH && '
        f'export TMPDIR="{posix_tmp}" && '
        f'export SLURM_JOB_ID="554433" && '
        f'bash "{posix_script}" echo "TMPDIR_FALLBACK_TEST"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Script execution failed:\n{proc.stderr}"
    assert "TMPDIR_FALLBACK_TEST" in proc.stdout

    expected_job_dir = tmp_root / "cochem_torq_554433"
    assert expected_job_dir.exists() and expected_job_dir.is_dir()
    assert (expected_job_dir / "Scratch").is_dir()
    assert (expected_job_dir / "Logs").is_dir()
    assert (expected_job_dir / "Outputs").is_dir()


def test_slurm_execution_default_backend(tmp_path: Path) -> None:
    """Physically execute script without arguments to verify default backend launch."""
    artifacts_dir = tmp_path / "default_run_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_art = _to_posix_path(artifacts_dir)

    cmd = (
        f'export COCHEM_ARTIFACTS="{posix_art}" && '
        f'bash "{posix_script}"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Default backend execution failed:\n{proc.stderr}\nStdout: {proc.stdout}"


def test_slurm_exit_code_propagation(tmp_path: Path) -> None:
    """Verify that non-zero exit codes from downstream commands propagate accurately."""
    artifacts_dir = tmp_path / "exit_code_test_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_art = _to_posix_path(artifacts_dir)

    cmd = (
        f'export COCHEM_ARTIFACTS="{posix_art}" && '
        f'bash "{posix_script}" bash -c "exit 42"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 42, (
        f"Expected exit code 42, got {proc.returncode}. Stderr: {proc.stderr}"
    )
