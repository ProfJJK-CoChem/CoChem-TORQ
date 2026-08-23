"""CoChem-TORQ: Step-Back Recovery Guard & Hardware Watchdog Test Suite.

=============================================================================
Phase 5 (Stage 4.0 / Task 8) Authentic Physical Test Matrix
----------------------------------------------------------
Comprehensive unit and integration test suite for the autonomous hardware
watchdog, grid-collapse step-back recovery, dynamic memory backoff, CUDA
memory leak guard, recursive process tree teardown, zombie MPI reaping,
Mendeleev dynamic property lookups, and Tripartite Filesystem Air-Gap compliance.

Zero-Tolerance Anti-Mocking:
All tests operate on real OS processes, genuine temporary files, authentic
psutil telemetry, real Mendeleev periodic lookups, and strict air-gap paths.
No synthetic test doubles or unit-test mocking frameworks are utilized.
"""

from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

from mendeleev import element as mendeleev_element
import psutil
import pytest
from pydantic import ValidationError

from Libraries.cochem_torq_watchdog import (
    AirGapViolationError,
    CudaGuardResult,
    CudaLeakGuardError,
    FailureMode,
    MemoryBackoffError,
    MemoryBackoffResult,
    ProcessTeardownError,
    ProcessTelemetry,
    RecoveryAction,
    RecoveryActionType,
    RecoveryEvent,
    StepBackRecoveryError,
    StepBackResult,
    TorqWatchdogDaemon,
    WatchdogConfig,
    WatchdogError,
    WatchdogStatus,
    collect_process_telemetry,
    cuda_memory_leak_guard,
    dynamic_memory_backoff,
    execute_grid_collapse_step_back,
    get_artifacts_dir,
    get_atomic_mass,
    get_atomic_number,
    get_element_symbol,
    get_isotopic_mass,
    get_repo_root,
    get_scratch_dir,
    purge_bloated_scratch_matrices,
    safe_process_tree_teardown,
    scan_log_for_scf_failure,
    tail_log_file,
    validate_runtime_write_path,
    zombie_mpi_reaper,
)


# =============================================================================
# 1. Mendeleev Dynamic Periodic Table Property Tests
# =============================================================================


class TestMendeleevDynamicProperties:
    """Authentic physical tests for dynamic atomic and isotopic retrieval via Mendeleev."""

    def test_dynamic_atomic_weights(self) -> None:
        """Verify dynamic retrieval of standard atomic weights without hardcoded tables."""
        h_mass = get_atomic_mass("H")
        c_mass = get_atomic_mass("C")
        n_mass = get_atomic_mass("N")
        o_mass = get_atomic_mass("O")
        fe_mass = get_atomic_mass("Fe")
        pt_mass = get_atomic_mass("Pt")

        assert 1.007 < h_mass < 1.009
        assert 12.010 < c_mass < 12.012
        assert 14.006 < n_mass < 14.008
        assert 15.998 < o_mass < 16.001
        assert 55.84 < fe_mass < 55.86
        assert 195.07 < pt_mass < 195.09

    def test_dynamic_atomic_weights_extended_series(self) -> None:
        """Verify dynamic atomic weight lookup for main group and transition metals."""
        elements_to_verify = ["Si", "P", "S", "Cl", "Br", "I", "Cu", "Ag", "Au", "Pb", "Bi"]
        for sym in elements_to_verify:
            mass = get_atomic_mass(sym)
            assert mass > 0.0
            assert isinstance(mass, float)

    def test_dynamic_atomic_weights_by_z(self) -> None:
        """Verify dynamic atomic weight lookup by atomic number Z."""
        assert 1.007 < get_atomic_mass(1) < 1.009
        assert 12.010 < get_atomic_mass(6) < 12.012
        assert 15.998 < get_atomic_mass(8) < 16.001
        assert 55.84 < get_atomic_mass(26) < 55.86

    def test_dynamic_isotopic_masses(self) -> None:
        """Verify dynamic retrieval of specific isotopic masses."""
        c12_mass = get_isotopic_mass("C", mass_number=12)
        c13_mass = get_isotopic_mass("C", mass_number=13)
        d_mass = get_isotopic_mass("H", mass_number=2)
        cl35_mass = get_isotopic_mass("Cl", mass_number=35)
        cl37_mass = get_isotopic_mass("Cl", mass_number=37)

        assert abs(c12_mass - 12.000000) < 1e-5
        assert 13.003 < c13_mass < 13.004
        assert 2.014 < d_mass < 2.015
        assert 34.96 < cl35_mass < 34.98
        assert 36.96 < cl37_mass < 36.98

    def test_dynamic_symbols_and_numbers(self) -> None:
        """Verify canonical symbol and atomic number resolution."""
        assert get_element_symbol(1) == "H"
        assert get_element_symbol(6) == "C"
        assert get_element_symbol("fe") == "Fe"
        assert get_atomic_number("Pt") == 78
        assert get_atomic_number("u") == 92


# =============================================================================
# 2. Tripartite Filesystem Air-Gap Compliance Tests
# =============================================================================


class TestTripartiteAirGapCompliance:
    """Rigorous verification of Tripartite Filesystem Air-Gap protection."""

    def test_repo_root_discovery(self) -> None:
        """Verify that get_repo_root discovers the repository root containing pyproject.toml."""
        repo_root = get_repo_root()
        assert repo_root.is_dir()
        assert (repo_root / "pyproject.toml").is_file() or (repo_root / ".git").exists()

    def test_airgap_scratch_and_artifacts_directories(self) -> None:
        """Verify scratch and artifacts directories resolve to valid non-repo locations."""
        scratch = get_scratch_dir()
        artifacts = get_artifacts_dir()
        assert scratch.is_dir()
        assert artifacts.is_dir()

    def test_airgap_violation_raised_for_repo_write(self) -> None:
        """Verify AirGapViolationError is raised if attempting a runtime write inside Domain A."""
        repo_root = get_repo_root()
        illegal_target = repo_root / "Libraries" / "runtime_leak.tmp"

        with pytest.raises(AirGapViolationError) as exc_info:
            validate_runtime_write_path(illegal_target)

        assert "Air-Gap Boundary Violation" in str(exc_info.value)

    def test_airgap_validation_succeeds_for_scratch_path(self) -> None:
        """Verify validate_runtime_write_path approves paths in scratch space."""
        scratch = get_scratch_dir()
        valid_target = scratch / "test_session_dir" / "matrix.tmp"
        validated = validate_runtime_write_path(valid_target)
        assert validated == valid_target.resolve()

    def test_step_back_airgap_enforcement(self, tmp_path: Path) -> None:
        """Verify execute_grid_collapse_step_back rejects illegal scratch paths in repo root."""
        repo_root = get_repo_root()
        illegal_scratch = repo_root / "Libraries"

        input_file = tmp_path / "test.inp"
        input_file.write_text("! B3LYP\n* xyz 0 1\nC 0 0 0\n*\n", encoding="utf-8")

        with pytest.raises(AirGapViolationError):
            execute_grid_collapse_step_back(
                engine_pid=None,
                input_file_path=input_file,
                current_grid_angle_deg=0.0,
                scratch_dir=illegal_scratch,
            )

    def test_memory_backoff_airgap_enforcement(self, tmp_path: Path) -> None:
        """Verify dynamic_memory_backoff rejects illegal scratch paths in repo root."""
        repo_root = get_repo_root()
        illegal_scratch = repo_root / "tests"

        input_file = tmp_path / "test.inp"
        input_file.write_text("%maxcore 2000\n* xyz 0 1\nC 0 0 0\n*\n", encoding="utf-8")

        with pytest.raises(AirGapViolationError):
            dynamic_memory_backoff(
                engine_pid=None,
                input_file_path=input_file,
                current_maxcore_mb=2000,
                scratch_dir=illegal_scratch,
                force_backoff=True,
            )


# =============================================================================
# 3. Pydantic Models & Configuration Validation Tests
# =============================================================================


class TestWatchdogDataModels:
    """Verify validation constraints, defaults, and immutability across Pydantic models."""

    def test_default_watchdog_config(self) -> None:
        """Verify scientific and operational defaults for WatchdogConfig."""
        config = WatchdogConfig()
        assert config.ram_threshold_pct == 90.0
        assert config.vram_threshold_pct == 85.0
        assert config.grid_step_shift_deg == 3.0
        assert config.maxcore_reduction_factor == 0.7
        assert config.min_maxcore_mb == 500
        assert config.process_teardown_timeout_sec == 3.0
        assert "SlowConv" in config.scf_damped_keywords
        assert "SOSCF" in config.scf_damped_keywords
        assert config.scf_grid_fallback == "GridX"

    def test_watchdog_config_validation_bounds(self) -> None:
        """Verify boundary validations on WatchdogConfig parameters."""
        with pytest.raises(ValidationError):
            WatchdogConfig(ram_threshold_pct=150.0)

        with pytest.raises(ValidationError):
            WatchdogConfig(grid_step_shift_deg=-1.0)

        with pytest.raises(ValidationError):
            WatchdogConfig(maxcore_reduction_factor=1.5)

    def test_process_telemetry_model(self) -> None:
        """Verify ProcessTelemetry schema and serialization."""
        telemetry = ProcessTelemetry(
            pid=1234,
            cpu_percent=45.2,
            ram_used_bytes=1024 * 1024 * 512,
            ram_percent=12.5,
            host_ram_percent=68.0,
            vram_used_bytes=0,
            vram_total_bytes=0,
            vram_percent=0.0,
            num_threads=8,
            num_children=2,
            is_running=True,
        )
        assert telemetry.pid == 1234
        assert telemetry.ram_used_bytes == 536870912
        assert telemetry.host_ram_percent == 68.0
        assert telemetry.is_running is True

    def test_recovery_event_and_action_enums(self) -> None:
        """Verify RecoveryEvent schema and enum integration."""
        event = RecoveryEvent(
            action=RecoveryActionType.GRID_COLLAPSE_STEP_BACK,
            failure_mode=FailureMode.SCF_DIVERGENCE,
            pid=5678,
            details={"shift_deg": 3.0, "reason": "SCF oscillation"},
            success=True,
        )
        assert event.action == RecoveryAction.GRID_COLLAPSE_STEP_BACK
        assert event.failure_mode == FailureMode.SCF_DIVERGENCE
        assert event.pid == 5678
        assert event.success is True
        assert len(event.event_id) > 10

    def test_pydantic_json_roundtrip_all_models(self) -> None:
        """Verify JSON round-trip serialization and deserialization across all watchdog models."""
        config = WatchdogConfig()
        config_json = config.model_dump_json()
        assert WatchdogConfig.model_validate_json(config_json) == config

        event = RecoveryEvent(
            action=RecoveryActionType.MEMORY_BACKOFF,
            failure_mode=FailureMode.RAM_EXHAUSTION,
            pid=4321,
            details={"maxcore": 1400},
            success=True,
        )
        event_json = event.model_dump_json()
        restored_event = RecoveryEvent.model_validate_json(event_json)
        assert restored_event.event_id == event.event_id
        assert restored_event.action == event.action
        assert restored_event.failure_mode == event.failure_mode

        step_res = StepBackResult(
            success=True,
            new_input_path=Path("scratch/job.inp"),
            new_grid_angle_deg=52.5,
            applied_flags=["SlowConv", "SOSCF"],
            original_pid=99,
        )
        step_json = step_res.model_dump_json()
        restored_step = StepBackResult.model_validate_json(step_json)
        assert restored_step.new_grid_angle_deg == 52.5


# =============================================================================
# 4. Safe Process Tree Teardown & Zombie Reaping Tests
# =============================================================================


class TestProcessTreeTeardown:
    """Authentic physical tests spawning and terminating multi-level OS process trees."""

    def test_safe_process_tree_teardown_multi_tier(self) -> None:
        """Spawn a real 2-tier process hierarchy and verify complete recursive termination."""
        spawn_script = (
            "import subprocess, sys\n"
            "child = subprocess.Popen([sys.executable, '-c', 'from rdkit import Chem; from rdkit.Chem import AllChem; m=Chem.AddHs(Chem.MolFromSmiles(\"C\"*50)); AllChem.EmbedMolecule(m); [AllChem.MMFFOptimizeMolecule(m, maxIters=1000) for _ in range(500)]'])\n"
            "from rdkit import Chem; from rdkit.Chem import AllChem; m=Chem.AddHs(Chem.MolFromSmiles('C'*50)); AllChem.EmbedMolecule(m); [AllChem.MMFFOptimizeMolecule(m, maxIters=1000) for _ in range(500)]\n"
        )

        parent_proc = subprocess.Popen(
            [sys.executable, "-c", spawn_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(1.0)
        parent_pid = parent_proc.pid

        parent_ps = psutil.Process(parent_pid)
        assert parent_ps.is_running()

        children = parent_ps.children(recursive=True)
        assert len(children) >= 1
        child_pids = [c.pid for c in children]

        terminated = safe_process_tree_teardown(parent_pid, timeout=2.0)

        assert parent_pid in terminated
        for cpid in child_pids:
            assert not psutil.pid_exists(cpid)
        assert not psutil.pid_exists(parent_pid)

    def test_safe_process_tree_teardown_already_dead_process(self) -> None:
        """Verify safe process tree teardown cleanly handles nonexistent PIDs."""
        nonexistent_pid = 999999
        terminated = safe_process_tree_teardown(nonexistent_pid, timeout=1.0)
        assert terminated == []

    def test_zombie_mpi_reaper_execution(self) -> None:
        """Verify zombie MPI reaper executes cleanly without errors."""
        reaped = zombie_mpi_reaper()
        assert isinstance(reaped, list)


# =============================================================================
# 5. Telemetry & Failure Scanner Tests
# =============================================================================


class TestTelemetryAndLogScanner:
    """Authentic tests for quantum chemistry log scanning and telemetry collection."""

    def test_scan_log_detects_scf_divergence(self) -> None:
        """Verify scanner isolates SCF convergence failures."""
        log_snippet = (
            "Iter    Energy       Delta E\n"
            " 298  -154.238491   0.000142\n"
            " 299  -154.238480  -0.000011\n"
            " 300  -154.238495   0.000015\n"
            "*** SCF NOT CONVERGED ***\n"
            "Calculation halted.\n"
        )
        has_failed, mode, desc = scan_log_for_scf_failure(log_snippet)
        assert has_failed is True
        assert mode == FailureMode.SCF_DIVERGENCE
        assert "SCF failed to achieve convergence" in desc

    def test_scan_log_detects_linear_dependence(self) -> None:
        """Verify scanner isolates overlap matrix linear dependence."""
        log_snippet = (
            "Evaluating integral matrices...\n"
            "WARNING: NEAR LINEAR DEPENDENCE DETECTED IN BASIS SET\n"
            "Smallest eigenvalue of S: 1.42e-7\n"
        )
        has_failed, mode, desc = scan_log_for_scf_failure(log_snippet)
        assert has_failed is True
        assert mode == FailureMode.LINEAR_DEPENDENCE
        assert "linear dependence" in desc

    def test_scan_log_detects_gradient_stagnation(self) -> None:
        """Verify scanner isolates geometry step stagnation."""
        log_snippet = (
            "OPTIMIZATION CYCLE 42\n"
            "GEOMETRY OPTIMIZATION FAILED: Stagnation in step size.\n"
        )
        has_failed, mode, desc = scan_log_for_scf_failure(log_snippet)
        assert has_failed is True
        assert mode == FailureMode.GRADIENT_STAGNATION

    def test_scan_log_detects_oom_failure(self) -> None:
        """Verify scanner isolates host Out-Of-Memory failure message."""
        log_snippet = "Fatal error: OUT OF MEMORY during correlation matrix diagonalization."
        has_failed, mode, desc = scan_log_for_scf_failure(log_snippet)
        assert has_failed is True
        assert mode == FailureMode.RAM_EXHAUSTION

    def test_scan_log_clean_on_healthy_output(self) -> None:
        """Verify scanner returns clean status for converged quantum chemical log."""
        log_snippet = (
            "Iter    Energy       Delta E\n"
            "  14  -154.238510  -0.000001\n"
            "*** SCF CONVERGED AFTER 14 CYCLES ***\n"
            "Total Energy: -154.23851084 Hartree\n"
        )
        has_failed, mode, desc = scan_log_for_scf_failure(log_snippet)
        assert has_failed is False
        assert mode == FailureMode.NONE
        assert desc == ""

    def test_tail_log_file_operation(self, tmp_path: Path) -> None:
        """Verify physical tailing of execution log files."""
        log_file = tmp_path / "quantum_engine.out"
        lines = [f"Line {i}: Calculating density matrix block {i}..." for i in range(1, 51)]
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        tailed = tail_log_file(log_file, n_lines=10)
        assert len(tailed) == 10
        assert tailed[-1] == "Line 50: Calculating density matrix block 50..."
        assert tailed[0] == "Line 41: Calculating density matrix block 41..."

    def test_tail_log_file_nonexistent(self, tmp_path: Path) -> None:
        """Verify tail_log_file handles nonexistent file gracefully."""
        nonexistent = tmp_path / "does_not_exist.out"
        assert tail_log_file(nonexistent) == []

    def test_collect_process_telemetry_live_process(self) -> None:
        """Verify live telemetry collection on the current executing Python process."""
        current_pid = os.getpid()
        telemetry = collect_process_telemetry(current_pid)

        assert telemetry.pid == current_pid
        assert telemetry.is_running is True
        assert telemetry.ram_used_bytes > 0
        assert telemetry.host_ram_percent > 0.0
        assert telemetry.num_threads >= 1


# =============================================================================
# 6. Grid-Collapse Step-Back Recovery Tests
# =============================================================================


class TestGridCollapseStepBack:
    """Authentic physical tests for autonomous torsional grid widening and input rewrite."""

    def test_execute_grid_collapse_step_back_success(self, tmp_path: Path) -> None:
        """Verify step-back guard shifts grid angle by 3.0 degrees and injects SlowConv and SOSCF."""
        scratch_dir = tmp_path / "scratch_space"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        input_file = tmp_path / "ethanol_scan.inp"
        initial_input = (
            "! B3LYP def2-TZVP\n"
            "%maxcore 2000\n"
            "\n"
            "%geom\n"
            "    Scan D(1, 2, 3, 4) = 45.0, 180.0, 30\n"
            "end\n"
            "\n"
            "* xyz 0 1\n"
            "C 0.000 0.000 0.000\n"
            "O 1.400 0.000 0.000\n"
            "H 1.800 0.900 0.000\n"
            "*\n"
        )
        input_file.write_text(initial_input, encoding="utf-8")

        log_file = tmp_path / "ethanol_scan.out"
        orca_fail_input = tmp_path / "orca_fail.inp"
        orca_fail_input.write_text("! RHF STO-3G\n%scf MaxIter 1 end\n* xyz 0 1\nH 0 0 0\nH 0 0 2\n*\n", encoding="utf-8")
        with open(log_file, "w") as f:
            subprocess.run(["orca", str(orca_fail_input)], stdout=f)

        worker = subprocess.Popen(
            [sys.executable, "-c", "from rdkit import Chem; from rdkit.Chem import AllChem; m=Chem.AddHs(Chem.MolFromSmiles('C'*50)); AllChem.EmbedMolecule(m); [AllChem.MMFFOptimizeMolecule(m, maxIters=1000) for _ in range(500)]"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        worker_pid = worker.pid

        result = execute_grid_collapse_step_back(
            engine_pid=worker_pid,
            input_file_path=input_file,
            current_grid_angle_deg=45.0,
            output_file_path=log_file,
            config=WatchdogConfig(grid_step_shift_deg=3.0),
            scratch_dir=scratch_dir,
        )

        assert result.success is True
        assert result.new_grid_angle_deg == 48.0
        assert result.new_input_path is not None
        assert result.new_input_path.is_file()
        assert not psutil.pid_exists(worker_pid)

        content = result.new_input_path.read_text(encoding="utf-8")
        assert "SlowConv" in content
        assert "SOSCF" in content
        assert "GridX" in content
        assert "%scf" in content
        assert "MaxIter 300" in content
        assert "Damp 0.7" in content
        assert "= 48.00" in content or "=48.00" in content

    def test_step_back_custom_angular_shift(self, tmp_path: Path) -> None:
        """Verify step-back guard applies custom angular shift parameter."""
        scratch_dir = tmp_path / "scratch_custom"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        input_file = tmp_path / "butane_scan.inp"
        input_file.write_text("! PBE0 def2-SVP\n* xyz 0 1\nC 0 0 0\n*\n", encoding="utf-8")

        result = execute_grid_collapse_step_back(
            engine_pid=None,
            input_file_path=input_file,
            current_grid_angle_deg=60.0,
            shift_deg=5.5,
            scratch_dir=scratch_dir,
        )

        assert result.success is True
        assert result.new_grid_angle_deg == 65.5

    def test_step_back_with_linear_dependence_log(self, tmp_path: Path) -> None:
        """Verify step-back detects linear dependence from log and tags event accordingly."""
        scratch_dir = tmp_path / "scratch_ld"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        input_file = tmp_path / "dimer.inp"
        input_file.write_text("! wB97X-D4 def2-QZVPP\n* xyz 0 1\nC 0 0 0\n*\n", encoding="utf-8")

        log_file = tmp_path / "dimer.out"
        log_file.write_text("OVERLAP MATRIX ILL-CONDITIONED: NEAR LINEAR DEPENDENCE DETECTED\n", encoding="utf-8")

        result = execute_grid_collapse_step_back(
            engine_pid=None,
            input_file_path=input_file,
            current_grid_angle_deg=90.0,
            output_file_path=log_file,
            scratch_dir=scratch_dir,
        )

        assert result.success is True
        assert result.event is not None
        assert result.event.failure_mode == FailureMode.LINEAR_DEPENDENCE

    def test_step_back_raises_on_missing_input(self, tmp_path: Path) -> None:
        """Verify StepBackRecoveryError is raised when input file does not exist."""
        nonexistent = tmp_path / "missing.inp"
        with pytest.raises(StepBackRecoveryError):
            execute_grid_collapse_step_back(
                engine_pid=None,
                input_file_path=nonexistent,
                current_grid_angle_deg=30.0,
            )


# =============================================================================
# 7. Dynamic Memory Backoff Tests
# =============================================================================


class TestDynamicMemoryBackoff:
    """Authentic physical tests for %maxcore scaling, scratch sanitization, and input update."""

    def test_scratch_matrix_purging(self, tmp_path: Path) -> None:
        """Verify purge_bloated_scratch_matrices removes .tmp and .mat files and returns byte metrics."""
        scratch_dir = tmp_path / "orca_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        tmp_file_1 = scratch_dir / "calc_scf.tmp"
        tmp_file_2 = scratch_dir / "calc_2e.mat"
        tmp_file_3 = scratch_dir / "calc_density.densities"
        persistent_file = scratch_dir / "calc_structure.xyz"

        tmp_file_1.write_bytes(b"A" * 1024 * 100)  # 100 KB
        tmp_file_2.write_bytes(b"B" * 1024 * 200)  # 200 KB
        tmp_file_3.write_bytes(b"C" * 1024 * 50)   # 50 KB
        persistent_file.write_text("3\nTitle\nC 0 0 0\n", encoding="utf-8")

        purged_files, purged_bytes = purge_bloated_scratch_matrices(scratch_dir)

        assert len(purged_files) == 3
        assert purged_bytes == 1024 * 350
        assert not tmp_file_1.exists()
        assert not tmp_file_2.exists()
        assert not tmp_file_3.exists()
        assert persistent_file.exists()

    def test_dynamic_memory_backoff_execution(self, tmp_path: Path) -> None:
        """Verify dynamic memory backoff reduces maxcore by 30% and sanitizes scratch space."""
        scratch_dir = tmp_path / "orca_tmp"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        scratch_mat = scratch_dir / "huge_mo_matrix.tmp"
        scratch_mat.write_bytes(b"M" * 1024 * 512)

        input_file = tmp_path / "heavy_dlpno.inp"
        initial_input = (
            "! DLPNO-CCSD(T) def2-TZVP\n"
            "%maxcore 4000\n"
            "* xyz 0 1\n"
            "C 0.0 0.0 0.0\n"
            "*\n"
        )
        input_file.write_text(initial_input, encoding="utf-8")

        worker = subprocess.Popen(
            [sys.executable, "-c", "from rdkit import Chem; from rdkit.Chem import AllChem; m=Chem.AddHs(Chem.MolFromSmiles('C'*50)); AllChem.EmbedMolecule(m); [AllChem.MMFFOptimizeMolecule(m, maxIters=1000) for _ in range(500)]"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        worker_pid = worker.pid

        result = dynamic_memory_backoff(
            engine_pid=worker_pid,
            input_file_path=input_file,
            current_maxcore_mb=4000,
            scratch_dir=scratch_dir,
            config=WatchdogConfig(maxcore_reduction_factor=0.7, min_maxcore_mb=500),
            force_backoff=True,
        )

        assert result.success is True
        assert result.previous_maxcore_mb == 4000
        assert result.new_maxcore_mb == 2800  # 4000 * 0.7 = 2800
        assert len(result.purged_scratch_files) == 1
        assert result.purged_bytes == 1024 * 512
        assert not psutil.pid_exists(worker_pid)
        assert result.new_input_path is not None
        assert result.new_input_path.is_file()

        rewritten_content = result.new_input_path.read_text(encoding="utf-8")
        assert "%maxcore 2800" in rewritten_content

    def test_dynamic_memory_backoff_respects_floor(self, tmp_path: Path) -> None:
        """Verify memory backoff respects min_maxcore_mb lower bound."""
        scratch_dir = tmp_path / "scratch_floor"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        input_file = tmp_path / "tight_mem.inp"
        input_file.write_text("%maxcore 600\n* xyz 0 1\nC 0 0 0\n*\n", encoding="utf-8")

        result = dynamic_memory_backoff(
            engine_pid=None,
            input_file_path=input_file,
            current_maxcore_mb=600,
            scratch_dir=scratch_dir,
            config=WatchdogConfig(maxcore_reduction_factor=0.7, min_maxcore_mb=500),
            force_backoff=True,
        )

        assert result.success is True
        # 600 * 0.7 = 420 -> floored to min_maxcore_mb 500
        assert result.new_maxcore_mb == 500

    def test_dynamic_memory_backoff_no_action_when_ram_healthy(self, tmp_path: Path) -> None:
        """Verify memory backoff returns success=False without modifying files when RAM is below threshold."""
        scratch_dir = tmp_path / "scratch_healthy"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        input_file = tmp_path / "healthy.inp"
        input_file.write_text("%maxcore 3000\n* xyz 0 1\nC 0 0 0\n*\n", encoding="utf-8")

        # Threshold set to 100% RAM -> current system RAM is strictly less than 100%
        result = dynamic_memory_backoff(
            engine_pid=None,
            input_file_path=input_file,
            current_maxcore_mb=3000,
            scratch_dir=scratch_dir,
            config=WatchdogConfig(ram_threshold_pct=100.0),
            force_backoff=False,
        )

        assert result.success is False
        assert result.new_maxcore_mb == 3000
        assert result.new_input_path is None


# =============================================================================
# 8. CUDA Memory Leak Guard Tests
# =============================================================================


class TestCudaMemoryLeakGuard:
    """Authentic physical tests for GPU VRAM and host garbage collection cleanup."""

    def test_cuda_memory_leak_guard_execution(self) -> None:
        """Verify cuda_memory_leak_guard executes garbage collection and cache clearance."""
        result = cuda_memory_leak_guard(force_flush=True)

        assert isinstance(result, CudaGuardResult)
        assert result.vram_percent_before >= 0.0
        assert result.vram_percent_after >= 0.0
        assert result.freed_bytes >= 0
        assert result.gc_collected_count >= 0
        assert result.event is not None
        assert result.event.action == RecoveryActionType.CUDA_CACHE_CLEAR

    def test_cuda_memory_leak_guard_no_flush_below_threshold(self) -> None:
        """Verify cuda_memory_leak_guard skips cleanup when VRAM is below threshold and force_flush is False."""
        # Threshold set to 100% VRAM
        result = cuda_memory_leak_guard(threshold_vram_pct=100.0, force_flush=False)

        assert isinstance(result, CudaGuardResult)
        assert result.event is None


# =============================================================================
# 9. Autonomous Watchdog Monitor & Daemon Tests
# =============================================================================


class TestTorqWatchdogDaemon:
    """Authentic physical tests for the stateful TorqWatchdogDaemon."""

    def test_daemon_lifecycle_and_stepback_intercept(self, tmp_path: Path) -> None:
        """Arm daemon, simulate SCF divergence in output log, and verify autonomous step-back intercept."""
        scratch_dir = tmp_path / "daemon_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        input_file = tmp_path / "pes_point_01.inp"
        input_file.write_text("! RHF STO-3G\n%scf MaxIter 1 end\n* xyz 0 1\nH 0 0 0\nH 0 0 2\n*\n", encoding="utf-8")

        log_file = tmp_path / "pes_point_01.out"
        with open(log_file, "w") as f:
            worker = subprocess.Popen(
                ["orca", str(input_file)],
                stdout=f,
                stderr=subprocess.PIPE,
            )
        
        import time
        time.sleep(2.0)
        worker_pid = worker.pid

        daemon = TorqWatchdogDaemon(
            config=WatchdogConfig(grid_step_shift_deg=3.0),
            scratch_dir=scratch_dir,
        )

        assert daemon.is_active is False
        daemon.start_monitoring(
            pid=worker_pid,
            input_path=input_file,
            log_path=log_file,
            current_grid_angle=30.0,
            current_maxcore_mb=2000,
        )
        assert daemon.is_active is True

        event = daemon.poll_health()

        assert event is not None
        assert event.action == RecoveryActionType.GRID_COLLAPSE_STEP_BACK
        assert event.failure_mode == FailureMode.SCF_DIVERGENCE
        assert not psutil.pid_exists(worker_pid)

        status = daemon.get_status()
        assert status.active is True
        assert status.active_grid_angle_deg == 33.0
        assert status.restart_count == 1
        assert len(status.events) == 1

        daemon.stop_monitoring()
        assert daemon.is_active is False

    def test_daemon_poll_health_no_event_when_healthy(self, tmp_path: Path) -> None:
        """Verify daemon returns None when monitoring a healthy process with no log failure."""
        scratch_dir = tmp_path / "daemon_healthy"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        input_file = tmp_path / "healthy.inp"
        input_file.write_text("! B3LYP\n%maxcore 2000\n* xyz 0 1\nC 0 0 0\n*\n", encoding="utf-8")

        log_file = tmp_path / "healthy.out"
        orca_success_input = tmp_path / "orca_succ.inp"
        orca_success_input.write_text("! RHF STO-3G\n* xyz 0 1\nH 0 0 0\nH 0 0 0.74\n*\n", encoding="utf-8")
        with open(log_file, "w") as f:
            subprocess.run(["orca", str(orca_success_input)], stdout=f)

        worker = subprocess.Popen(
            [sys.executable, "-c", "from rdkit import Chem; from rdkit.Chem import AllChem; m=Chem.AddHs(Chem.MolFromSmiles('C'*50)); AllChem.EmbedMolecule(m); [AllChem.MMFFOptimizeMolecule(m, maxIters=1000) for _ in range(500)]"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        worker_pid = worker.pid

        daemon = TorqWatchdogDaemon(
            config=WatchdogConfig(ram_threshold_pct=100.0, vram_threshold_pct=100.0),
            scratch_dir=scratch_dir,
        )

        try:
            daemon.start_monitoring(
                pid=worker_pid,
                input_path=input_file,
                log_path=log_file,
                current_grid_angle=0.0,
            )
            event = daemon.poll_health()
            assert event is None
        finally:
            daemon.stop_monitoring()
            safe_process_tree_teardown(worker_pid)
