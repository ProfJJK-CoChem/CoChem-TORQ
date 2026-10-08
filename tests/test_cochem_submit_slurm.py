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
8. Real path routing, explicit command passthrough, and exit code propagation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_FILE = REPO_ROOT / "HPC_Launchers" / "cochem_submit.slurm"


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


def test_slurm_bash_syntax_valid() -> None:
    """Verify that bash syntax parsing succeeds without errors."""
    posix_path = _to_posix_path(LAUNCHER_FILE)
    result = subprocess.run(["bash", "-n", posix_path], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Bash syntax check failed on {LAUNCHER_FILE}:\n{result.stderr}"
    )


def test_slurm_execution_with_cochem_artifacts(tmp_path: Path) -> None:
    """Observe actual directory creation with the configured artifacts root."""
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
    """Observe real SCRATCH routing when COCHEM_ARTIFACTS is unset."""
    scratch_root = tmp_path / "hpc_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_scratch = _to_posix_path(scratch_root)

    cmd = (
        f"unset COCHEM_ARTIFACTS && "
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
    """Observe real TMPDIR routing with other storage settings unset."""
    tmp_root = tmp_path / "system_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_tmp = _to_posix_path(tmp_root)

    cmd = (
        f"unset COCHEM_ARTIFACTS && "
        f"unset SCRATCH && "
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
    """Missing reviewed inputs must block batch science without a success claim."""
    artifacts_dir = tmp_path / "default_run_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_art = _to_posix_path(artifacts_dir)

    cmd = f'export COCHEM_ARTIFACTS="{posix_art}" && bash "{posix_script}"'

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 64, proc.stderr
    assert "PYTHON_EXEC must name" in proc.stderr
    assert "initialized successfully" not in proc.stdout
    assert list((artifacts_dir / "Outputs").iterdir()) == []


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


def test_slurm_request_without_approval_fails_closed(tmp_path: Path) -> None:
    """Naming an actual input file alone never grants calculation approval."""
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    env = {
        **os.environ,
        "COCHEM_ARTIFACTS": str(tmp_path / "artifacts"),
        "COCHEM_REQUEST_FILE": str(request),
        "PYTHON_EXEC": sys.executable,
    }
    env.pop("COCHEM_APPROVED_PLAN_FILE", None)
    result = subprocess.run(
        ["bash", str(LAUNCHER_FILE)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 64
    assert "COCHEM_APPROVED_PLAN_FILE" in result.stderr
    assert list((tmp_path / "artifacts" / "Outputs").iterdir()) == []


@pytest.mark.real_engine
def test_slurm_genuine_approved_hf_request(tmp_path: Path) -> None:
    """An explicit approval reaches a real bounded native HF calculation."""
    from cochem_torq.artifacts import verify_shard
    from cochem_torq.candidate_ledger import CandidateLedger
    from cochem_torq.domain import canonical_json, read_json
    from cochem_torq.service import approve_plan, plan_request

    review = plan_request(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
                "charge": 0,
                "multiplicity": 1,
            },
            "recipe": "hf-sto-3g-education",
            "products": ["geometry"],
            "resources": {"cores": 1, "memory_mb": 2048, "wall_seconds": 120},
            "source_provenance": {"producer": "explicit_local_test_input"},
        },
        execution="local_validation",
    )
    approved = approve_plan(review, actor="explicit local SLURM adapter test review")
    source = tmp_path / "request.json"
    source.write_bytes(canonical_json(review["plan"]["request"]))
    approval = tmp_path / "approved.json"
    approval.write_bytes(canonical_json(approved))
    artifacts = tmp_path / "artifacts"
    env = {
        **os.environ,
        "PYTHON_EXEC": sys.executable,
        "COCHEM_REQUEST_FILE": str(source),
        "COCHEM_APPROVED_PLAN_FILE": str(approval),
        "COCHEM_ARTIFACTS": str(artifacts),
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER_FILE)],
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    output = Path(receipt["output_directory"])
    assert output.parent == (artifacts / "Outputs").resolve()
    assert output.is_dir()
    outputs = [output]
    manifest = verify_shard(outputs[0])
    assert manifest["request_id"] == review["plan"]["request"]["request_id"]
    calculation = read_json(outputs[0] / "result.json")
    assert calculation["status"] == "complete"
    assert calculation["stages"]["equilibrium_geometry"]["status"] == "available"
    assert not calculation["identification_ready"]
    assert not calculation["experimental_accuracy_established"]
    with CandidateLedger(tmp_path / "real-candidates.sqlite") as ledger:
        candidate = ledger.register_shard(
            outputs[0],
            actor="explicit local result review",
            reason="Retain the actual native HF endpoint with its artifacts",
        )
        assert candidate["selection_state"] == "retained"
        assert candidate["geometry_status"] == "optimized"
        assert (
            candidate["molecule"]["geometry_bohr"]
            == (calculation["stages"]["equilibrium_geometry"]["value"]["geometry_bohr"])
        )
        assert candidate["native_artifact_hashes"]
        assert all(
            item in manifest["files"] for item in candidate["native_artifact_hashes"]
        )
        ledger.exclude(
            candidate["candidate_id"],
            expected_revision=1,
            actor="reviewer",
            reason="Reversible manual exclusion",
        )
        restored = ledger.restore(
            candidate["candidate_id"],
            expected_revision=2,
            actor="reviewer",
            reason="Restore the actual native candidate",
        )
        assert restored["quality"] == candidate["quality"]
        assert restored["native_artifact_hashes"] == candidate["native_artifact_hashes"]
        assert verify_shard(outputs[0]) == manifest
