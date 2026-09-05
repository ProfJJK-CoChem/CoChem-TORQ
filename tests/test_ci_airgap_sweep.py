"""Zero-Mock Unit and Integration Tests for CI Air-Gap Sweep (ci_tools/ci_airgap_sweep.py).

Mandated by SRS Doc 2 Part 1 (§1.1) and Method Matrix v4 (Sections 8A.5, 8C.1-8C.3).
Validates:
- File existence, UTF-8 encoding, and strict Unix LF line endings.
- Shannon entropy mathematical precision and threshold boundary checks.
- Magic number byte signature detection across disguised binary containers.
- Quantum chemistry simulation output signatures (.log, .out).
- Atomic Cartesian coordinate (.xyz) payload heuristics.
- Localized path leak and active execution state detection in configuration JSONs.
- Multi-OS cross-platform scanning (git ls-files -z and filesystem traversal).
- CLI subprocess execution with return code 0 (clean) and return code 1 (violations).
- Structured JSON serialization and summary reporting.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Set

import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
CI_TOOLS_DIR = REPO_ROOT / "ci_tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ci_tools.ci_airgap_sweep import (  # noqa: E402
    DEFAULT_ENTROPY_THRESHOLD,
    FORBIDDEN_EXTENSIONS,
    AirgapSweepSummary,
    AirgapViolation,
    calculate_shannon_entropy,
    check_config_pollution,
    check_qm_log_signatures,
    format_airgap_report,
    inspect_magic_number,
    is_restricted_extension,
    is_xyz_coordinate_payload,
    run_airgap_sweep,
)


@pytest.fixture
def script_path() -> Path:
    """Fixture providing absolute path to ci_tools/ci_airgap_sweep.py."""
    path = CI_TOOLS_DIR / "ci_airgap_sweep.py"
    assert path.exists(), f"ci_airgap_sweep.py not found at {path}"
    return path


def test_script_exists_and_lf_endings(script_path: Path) -> None:
    """Verify ci_airgap_sweep.py exists, has substantive size, no BOM, and uses LF line endings."""
    stat = script_path.stat()
    assert stat.st_size > 2000, f"Script size too small ({stat.st_size} bytes)"
    raw = script_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "Script contains UTF-8 BOM"
    assert b"\r\n" not in raw, "Script contains Windows CRLF line endings"
    assert b"\n" in raw, "Script missing newline characters"


def test_shannon_entropy_calculation() -> None:
    """Verify Shannon entropy mathematical properties."""
    # Empty data -> 0.0
    assert calculate_shannon_entropy(b"") == 0.0

    # Uniform single byte -> 0.0
    assert calculate_shannon_entropy(b"A" * 1024) == 0.0

    # Two equally distributed bytes -> 1.0 bit/byte
    assert abs(calculate_shannon_entropy(b"AB" * 512) - 1.0) < 1e-6

    # Normal English source code -> roughly 4.0 - 5.5 bits/byte
    code_sample = b"def calculate_energy(mass: float, velocity: float) -> float:\n    return 0.5 * mass * (velocity ** 2)\n" * 10
    h_code = calculate_shannon_entropy(code_sample)
    assert 3.5 <= h_code <= 5.8

    # High entropy pseudorandom binary payload -> > 7.5 bits/byte
    high_ent_bytes = bytes((i * 137 + 29) % 256 for i in range(2048))
    h_rand = calculate_shannon_entropy(high_ent_bytes)
    assert h_rand >= 7.8


def test_restricted_extensions_detection() -> None:
    """Verify detection of quantum chemical and restricted extensions."""
    blocked_set: Set[str] = set(FORBIDDEN_EXTENSIONS)

    assert is_restricted_extension(Path("orbitals.gbw"), blocked_set) == (True, ".gbw")
    assert is_restricted_extension(Path("state.h5"), blocked_set) == (True, ".h5")
    assert is_restricted_extension(Path("geometry.XYZ"), blocked_set) == (True, ".xyz")
    assert is_restricted_extension(Path("hessian.opt"), blocked_set) == (True, ".opt")
    assert is_restricted_extension(Path("run.parsl"), blocked_set) == (True, ".parsl")
    assert is_restricted_extension(Path("data.h5.bak"), blocked_set) == (True, ".h5")
    assert is_restricted_extension(Path(".tmp"), blocked_set) == (True, ".tmp")

    # Allowed source files
    assert is_restricted_extension(Path("main.py"), blocked_set) == (False, "")
    assert is_restricted_extension(Path("README.md"), blocked_set) == (False, "")
    assert is_restricted_extension(Path("config.json"), blocked_set) == (False, "")


def test_magic_number_detection() -> None:
    """Verify magic number detection identifies disguised binaries."""
    # HDF5
    assert inspect_magic_number(b"\x89HDF\r\n\x1a\n\x00\x00" + b"\x00" * 32) is not None
    # SQLite
    assert inspect_magic_number(b"SQLite format 3\x00\x10\x00" + b"\x00" * 32) is not None
    # NumPy
    assert inspect_magic_number(b"\x93NUMPY\x01\x00v\x00" + b"\x00" * 32) is not None
    # Parquet
    assert inspect_magic_number(b"PAR1\x00\x01\x02\x03" + b"\x00" * 32) is not None
    # Linux ELF
    assert inspect_magic_number(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 32) is not None
    # Windows PE
    assert inspect_magic_number(b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 32) is not None
    # Benign text
    assert inspect_magic_number(b"# CoChem Static Repository Tier\n") is None


def test_qm_log_signatures_detection() -> None:
    """Verify quantum chemistry calculation banner and log detection."""
    orca_hdr = b"=========================================\n       * O R C A *\n========================================="
    assert check_qm_log_signatures(orca_hdr) is not None

    energy_hdr = b"FINAL SINGLE POINT ENERGY   -152.8732148\n"
    assert check_qm_log_signatures(energy_hdr) is not None

    benign_text = b"def run_calculation(): pass\n"
    assert check_qm_log_signatures(benign_text) is None


def test_xyz_coordinate_payload_detection() -> None:
    """Verify detection of XYZ molecular coordinate format."""
    xyz_data = (
        b"3\n"
        b"Water dimer fragment\n"
        b"O 0.000000 0.000000 0.117300\n"
        b"H 0.000000 0.757200 -0.469200\n"
        b"H 0.000000 -0.757200 -0.469200\n"
    )
    assert is_xyz_coordinate_payload(xyz_data) is True

    # Benign text is not XYZ
    doc_data = b"# Overview\nThis document details the method matrix.\n1. First section.\n2. Second section.\n"
    assert is_xyz_coordinate_payload(doc_data) is False


def test_config_pollution_detection(tmp_path: Path) -> None:
    """Verify config pollution detection flags active jobs and localized paths."""
    # Clean config
    clean_cfg = tmp_path / "cochem_system_config.json"
    clean_cfg.write_text(json.dumps({"environment": "production", "active_jobs": []}), encoding="utf-8")
    assert len(check_config_pollution(clean_cfg)) == 0

    # Polluted config with active jobs
    dirty_jobs = tmp_path / "dirty_jobs.json"
    dirty_jobs.write_text(json.dumps({"active_jobs": ["job_1234"]}), encoding="utf-8")
    issues_jobs = check_config_pollution(dirty_jobs)
    assert any("Active jobs" in iss for iss in issues_jobs)

    # Polluted config with localized path leak
    dirty_path = tmp_path / "dirty_path.json"
    dirty_path.write_text(json.dumps({"data_dir": "C:\\Users\\researcher\\scratch"}), encoding="utf-8")
    issues_path = check_config_pollution(dirty_path)
    assert any("Localized system path leak" in iss for iss in issues_path)


def test_clean_workspace_scan(tmp_path: Path) -> None:
    """Verify that a compliant static workspace passes air-gap sweep cleanly."""
    (tmp_path / "main.py").write_text("print('Clean static repository')", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project CoChem", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'cochem'", encoding="utf-8")

    sub = tmp_path / "src" / "pkg"
    sub.mkdir(parents=True)
    (sub / "engine.py").write_text("class Engine:\n    pass\n", encoding="utf-8")

    summary: AirgapSweepSummary = run_airgap_sweep(repo_root=tmp_path, use_git=False)

    assert summary.is_clean is True
    assert summary.violation_count == 0
    assert summary.scanned_files_count == 4


def test_disguised_binary_detection_in_sweep(tmp_path: Path) -> None:
    """Verify that disguised binary payloads with innocent extensions are flagged."""
    disguised_file = tmp_path / "notes.txt.raw"
    disguised_file.write_bytes(b"\x89HDF\r\n\x1a\n\x00\x00" + b"\x00" * 128)

    summary = run_airgap_sweep(repo_root=tmp_path, use_git=False)

    assert summary.is_clean is False
    assert summary.violation_count >= 1
    assert any(v.violation_type == "MAGIC_NUMBER_VIOLATION" for v in summary.violations)


def test_high_entropy_detection_in_sweep(tmp_path: Path) -> None:
    """Verify that high-entropy disguised binary data is detected."""
    encrypted_blob = tmp_path / "secrets.dat"
    # Generate high entropy bytes
    encrypted_blob.write_bytes(bytes((i * 199 + 43) % 256 for i in range(4096)))

    summary = run_airgap_sweep(repo_root=tmp_path, use_git=False)

    assert summary.is_clean is False
    assert any(v.violation_type == "HIGH_ENTROPY_VIOLATION" for v in summary.violations)


def test_cli_execution_clean(tmp_path: Path) -> None:
    """Verify CLI execution returns exit code 0 on clean workspace."""
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Documentation\n", encoding="utf-8")

    cmd = [
        sys.executable,
        str(CI_TOOLS_DIR / "ci_airgap_sweep.py"),
        "--repo-root",
        str(tmp_path),
        "--no-git",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, f"Expected 0, got {proc.returncode}. STDERR: {proc.stderr}"
    assert "[PASSED] CLEAN" in proc.stdout


def test_cli_execution_violation(tmp_path: Path) -> None:
    """Verify CLI execution returns exit code 1 on restricted files."""
    (tmp_path / "scratch_run.gbw").write_bytes(b"\x00" * 64)

    cmd = [
        sys.executable,
        str(CI_TOOLS_DIR / "ci_airgap_sweep.py"),
        "--repo-root",
        str(tmp_path),
        "--no-git",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 1
    assert "[FAILED] AIR-GAP BREACH DETECTED" in proc.stderr
    assert "scratch_run.gbw" in proc.stderr


def test_cli_json_output(tmp_path: Path) -> None:
    """Verify CLI --json outputs valid JSON summary."""
    (tmp_path / "valid.py").write_text("z = 100\n", encoding="utf-8")

    cmd = [
        sys.executable,
        str(CI_TOOLS_DIR / "ci_airgap_sweep.py"),
        "--repo-root",
        str(tmp_path),
        "--no-git",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["is_clean"] is True
    assert data["violation_count"] == 0
    assert data["scanned_files_count"] == 1


def test_default_entropy_threshold_constant() -> None:
    """Verify default Shannon entropy threshold matches specification."""
    assert DEFAULT_ENTROPY_THRESHOLD == 7.8


def test_format_airgap_report(tmp_path: Path) -> None:
    """Verify format_airgap_report renders both clean and violation states."""
    clean_summary = AirgapSweepSummary(
        repo_root=tmp_path,
        is_clean=True,
        scanned_files_count=10,
        scanned_directories_count=2,
        violations=[],
        scan_method="git_ls_files",
        execution_time_seconds=0.0123,
        max_entropy_observed=4.5,
        high_entropy_files_count=0,
    )
    clean_report = format_airgap_report(clean_summary)
    assert "[PASSED] CLEAN (0 violations)" in clean_report
    assert "Static Execution Tier is 100% immutable" in clean_report

    violation = AirgapViolation(
        file_path=tmp_path / "orbitals.gbw",
        relative_path="orbitals.gbw",
        violation_type="FORBIDDEN_EXTENSION",
        detail="File matches forbidden runtime extension '.gbw'",
        severity="ERROR",
        entropy=7.92,
        matched_pattern=".gbw",
    )
    dirty_summary = AirgapSweepSummary(
        repo_root=tmp_path,
        is_clean=False,
        scanned_files_count=10,
        scanned_directories_count=2,
        violations=[violation],
        scan_method="filesystem_traversal",
        execution_time_seconds=0.0456,
        max_entropy_observed=7.92,
        high_entropy_files_count=1,
    )
    dirty_report = format_airgap_report(dirty_summary)
    assert "[FAILED] AIR-GAP BREACH DETECTED (1 violations)" in dirty_report
    assert "FORBIDDEN_EXTENSION" in dirty_report
    assert "MANDATED REMEDIATION PROTOCOL" in dirty_report


def test_restricted_directory_detection(tmp_path: Path) -> None:
    """Verify restricted runtime directory patterns are flagged."""
    restricted_dir = tmp_path / "cochem_artifacts"
    restricted_dir.mkdir(parents=True)
    (restricted_dir / "output.txt").write_text("data", encoding="utf-8")

    summary = run_airgap_sweep(repo_root=tmp_path, use_git=False)
    assert summary.is_clean is False
    assert any(v.violation_type == "RESTRICTED_DIRECTORY" for v in summary.violations)


def test_invalid_repo_paths(tmp_path: Path) -> None:
    """Verify run_airgap_sweep validates repository root existence and directory type."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="Repository root path does not exist"):
        run_airgap_sweep(repo_root=non_existent)

    regular_file = tmp_path / "file.txt"
    regular_file.write_text("hello", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="Repository root path is not a directory"):
        run_airgap_sweep(repo_root=regular_file)

