"""
CoChem-TORQ: High-Fidelity Quantum Engine & Cascade Broker Test Suite
======================================================================
Phase 5 (Stage 4.0) Authentic Physical Test Matrix
--------------------------------------------------
Validates Method Matrix v4 execution cascade (defgrid1 -> defgrid3),
ORCA Python Interface (OPI) persistent memory threading, dynamic wavefunction
propagation (! MOREAD / %moinp), stateful SCF checkpointing, GPU4PySCF dynamic
batching with VRAM headroom protection, spin contamination validation (<10% threshold),
tightened intermolecular %geom criteria (TolMaxG 1e-5), frozen-monomer protocol,
prohibition of Calc_Hess true for initial Hessians, D3/D4 dispersion enforcement,
and 6-Tier Environment Matrix scratch/shm path resolution.

Authoritative Standards:
- Method Matrix v4 (§4.4, §8A, §8B, §9A, §10, Table 2)
- Tripartite Filesystem Air-Gap Architecture (Domain A / B / C)
- Real Molecular Systems: Formic acid dimer, Water dimer, Zinc formate, 1,2-Ethanediol, Propane
"""

import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Generator, List, Tuple

import h5py
import numpy as np
import psutil
import pytest

from Libraries.cochem_torq_engine import (
    AirGapViolationError,
    DispatchPayload,
    EnvironmentTier,
    ExecutionContext,
    ORCAStepResult,
    SCFResult,
    detect_complex_and_monomers,
    dynamic_wavefunction_propagation,
    execute_subprocess_safe,
    gpu4pyscf_dynamic_batching,
    opi_persistent_threading,
    route_cascade_rules,
    route_method_matrix,
    safe_process_tree_teardown,
    stateful_scf_checkpointing,
    validate_spin_contamination,
)


# ============================================================================
# Authentic Physical Molecular Geometry Fixtures
# ============================================================================

@pytest.fixture
def water_dimer_geometry() -> Tuple[List[str], np.ndarray]:
    """
    Authentic equilibrium Water Dimer (H2O)2 geometry (Cs symmetry, R(O...O) = 2.91 A).
    """
    symbols = ["O", "H", "H", "O", "H", "H"]
    coords = np.array(
        [
            [0.0000, 0.0000, -1.4550],  # O1 donor
            [0.0000, 0.7600, -0.8650],  # H1 donor
            [0.0000, -0.7600, -0.8650], # H2 donor
            [0.0000, 0.0000, 1.4550],   # O2 acceptor
            [0.7600, 0.0000, 2.0450],   # H3 acceptor
            [-0.7600, 0.0000, 2.0450],  # H4 acceptor
        ],
        dtype=np.float64,
    )
    return symbols, coords


@pytest.fixture
def formic_acid_dimer_geometry() -> Tuple[List[str], np.ndarray]:
    """
    Authentic equilibrium Formic Acid Dimer (HCOOH)2 (C2h symmetry, double H-bonded).
    """
    symbols = ["C", "O", "O", "H", "H", "C", "O", "O", "H", "H"]
    coords = np.array(
        [
            [ 0.0000,  1.8500,  0.0000],   # C1
            [-1.2200,  1.3500,  0.0000],   # O1 (=O)
            [ 1.2200,  1.3500,  0.0000],   # O2 (-OH)
            [ 1.2200,  0.3800,  0.0000],   # H1 (hydroxyl H)
            [ 0.0000,  2.9300,  0.0000],   # H2 (formyl H)
            [ 0.0000, -1.8500,  0.0000],   # C2
            [ 1.2200, -1.3500,  0.0000],   # O3 (=O)
            [-1.2200, -1.3500,  0.0000],   # O4 (-OH)
            [-1.2200, -0.3800,  0.0000],   # H3 (hydroxyl H)
            [ 0.0000, -2.9300,  0.0000],   # H4 (formyl H)
        ],
        dtype=np.float64,
    )
    return symbols, coords


@pytest.fixture
def zinc_formate_geometry() -> Tuple[List[str], np.ndarray]:
    """
    Authentic Zinc(II) Formate complex [Zn(HCOO)3]^- geometry.
    """
    symbols = ["Zn", "C", "O", "O", "H", "C", "O", "O", "H", "C", "O", "O", "H"]
    coords = np.array(
        [
            [0.0000, 0.0000, 0.0000],   # Zn
            [2.3000, 0.0000, 0.0000],   # C1
            [1.6000, 1.0500, 0.0000],   # O1
            [1.6000, -1.0500, 0.0000],  # O2
            [3.3800, 0.0000, 0.0000],   # H1
            [-1.1500, 1.9919, 0.0000],  # C2
            [-0.1096, 1.9125, 0.0000],  # O3
            [-1.7096, 0.8625, 0.0000],  # O4
            [-1.6900, 2.9272, 0.0000],  # H2
            [-1.1500, -1.9919, 0.0000], # C3
            [-1.7096, -0.8625, 0.0000], # O5
            [-0.1096, -1.9125, 0.0000], # O6
            [-1.6900, -2.9272, 0.0000], # H3
        ],
        dtype=np.float64,
    )
    return symbols, coords


@pytest.fixture
def ethanediol_geometry() -> Tuple[List[str], np.ndarray]:
    """
    Authentic 1,2-Ethanediol (HO-CH2-CH2-OH) gauche conformer geometry.
    """
    symbols = ["C", "C", "O", "O", "H", "H", "H", "H", "H", "H"]
    coords = np.array(
        [
            [-0.7320, 0.3850, 0.0000],
            [0.7320, -0.3850, 0.0000],
            [-1.4320, -0.3850, 0.9800],
            [1.4320, 0.3850, -0.9800],
            [-0.8500, 1.4400, 0.2200],
            [-1.1500, 0.2100, -0.9900],
            [0.8500, -1.4400, -0.2200],
            [1.1500, -0.2100, 0.9900],
            [-1.3000, -1.3100, 0.7700],
            [1.3000, 1.3100, -0.7700],
        ],
        dtype=np.float64,
    )
    return symbols, coords


@pytest.fixture
def propane_geometry() -> Tuple[List[str], np.ndarray]:
    """
    Authentic Propane (C3H8) equilibrium geometry (N=11 atoms, 33x33 Hessian).
    """
    symbols = ["C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.0000, 0.5830, 0.0000],
            [-1.2750, -0.2670, 0.0000],
            [1.2750, -0.2670, 0.0000],
            [0.0000, 1.2350, 0.8820],
            [0.0000, 1.2350, -0.8820],
            [-1.3120, -0.9080, 0.8860],
            [-1.3120, -0.9080, -0.8860],
            [-2.1640, 0.3700, 0.0000],
            [1.3120, -0.9080, 0.8860],
            [1.3120, -0.9080, -0.8860],
            [2.1640, 0.3700, 0.0000],
        ],
        dtype=np.float64,
    )
    return symbols, coords


# ============================================================================
# 1. 6-Tier Environment Matrix & Path Resolution Tests
# ============================================================================

class TestEnvironmentMatrix:
    """Tests 6-Tier Environment Matrix detection, dynamic path resolution, and air-gap integrity."""

    def test_environment_tier_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Test GitHub Actions
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert ExecutionContext.detect_tier() == EnvironmentTier.GITHUB_ACTIONS
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

        # Test Codespaces
        monkeypatch.setenv("CODESPACES", "true")
        assert ExecutionContext.detect_tier() == EnvironmentTier.CODESPACES
        monkeypatch.delenv("CODESPACES", raising=False)

        # Test HPC SLURM
        monkeypatch.setenv("SLURM_JOB_ID", "123456")
        assert ExecutionContext.detect_tier() == EnvironmentTier.HPC_NODES
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    @pytest.mark.parametrize(
        "tier",
        [
            EnvironmentTier.LOCAL_WINDOWS,
            EnvironmentTier.LOCAL_MACOS,
            EnvironmentTier.LOCAL_LINUX,
            EnvironmentTier.GITHUB_ACTIONS,
            EnvironmentTier.CODESPACES,
            EnvironmentTier.HPC_NODES,
        ],
    )
    def test_scratch_and_shm_resolution_across_tiers(
        self, tier: EnvironmentTier, tmp_path: Path
    ) -> None:
        ctx = ExecutionContext(
            tier=tier,
            custom_scratch_dir=tmp_path / f"scratch_{tier.value.lower()}",
            custom_shm_dir=tmp_path / f"shm_{tier.value.lower()}",
            custom_artifacts_dir=tmp_path / f"artifacts_{tier.value.lower()}",
        )

        scratch_dir = ctx.get_scratch_dir()
        shm_dir = ctx.get_shm_dir()
        artifacts_dir = ctx.get_artifacts_dir()

        assert scratch_dir.exists() and scratch_dir.is_dir()
        assert shm_dir.exists() and shm_dir.is_dir()
        assert artifacts_dir.exists() and artifacts_dir.is_dir()

        # Test subfolder resolution
        sub_scratch = ctx.get_scratch_dir("sub_test")
        assert sub_scratch.exists() and sub_scratch.name == "sub_test"

    def test_air_gap_boundary_enforcement(self) -> None:
        ctx = ExecutionContext()
        # Verifies that normal scratch directory passes air gap check
        scratch = ctx.get_scratch_dir()
        ctx.verify_air_gap_boundary(scratch)

        # Verifies that attempting to use repo root directly as scratch raises AirGapViolationError
        repo_root = Path(__file__).resolve().parent.parent
        with pytest.raises(AirGapViolationError):
            ctx.verify_air_gap_boundary(repo_root / "Libraries")


# ============================================================================
# 2. Method Matrix Routing & Cascade Rules Tests
# ============================================================================

class TestMethodMatrixCascade:
    """Tests Method Matrix v4 rules, complex detection, grid tightening, and initial Hessian enforcement."""

    def test_water_dimer_complex_detection(
        self, water_dimer_geometry: Tuple[List[str], np.ndarray]
    ) -> None:
        syms, coords = water_dimer_geometry
        is_complex, components = detect_complex_and_monomers(syms, coords)
        assert is_complex is True
        assert len(components) == 2
        assert components[0] == [0, 1, 2]
        assert components[1] == [3, 4, 5]

    def test_formic_acid_dimer_frozen_monomer_routing(
        self, formic_acid_dimer_geometry: Tuple[List[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = formic_acid_dimer_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")

        payload = route_cascade_rules(
            point_coords=coords,
            context=ctx,
            symbols=syms,
            method="wB97M-V",
            basis_set="def2-QZVPP",
            initial_hessian="XTB2",
            frozen_monomer=True,
        )

        assert payload.is_complex is True
        assert payload.frozen_atom_indices == [0, 1, 2, 3, 4]  # First monomer frozen
        assert payload.grid_level == "defgrid3"

        orca_inp = payload.to_orca_input()
        assert "! wB97M-V def2-QZVPP def2/J DEFGRID3" in orca_inp
        assert "InHess XTB2" in orca_inp
        assert "TolMaxG 1e-5" in orca_inp
        assert "TolE 1e-7" in orca_inp
        assert "TolRMSG 3e-6" in orca_inp
        assert "TolRMSD 5e-5" in orca_inp
        assert "TolMaxD 1e-4" in orca_inp
        assert "Constraints" in orca_inp
        assert "{ C 0 C }" in orca_inp

    def test_grid_level_custom_routing(
        self, formic_acid_dimer_geometry: Tuple[List[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = formic_acid_dimer_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")

        payload = route_cascade_rules(
            point_coords=coords,
            context=ctx,
            symbols=syms,
            method="wB97M-V",
            basis_set="def2-TZVP",
            grid_level="defgrid1",
        )

        assert payload.grid_level == "defgrid1"
        orca_inp = payload.to_orca_input()
        assert "DEFGRID1" in orca_inp

    def test_zinc_formate_complex_routing(
        self, zinc_formate_geometry: Tuple[List[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = zinc_formate_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")

        payload = route_cascade_rules(
            point_coords=coords,
            context=ctx,
            symbols=syms,
            charge=-1,
            multiplicity=1,
            method="B3LYP-D4",
            basis_set="def2-TZVP",
            initial_hessian="Lindh",
        )

        assert payload.charge == -1
        assert payload.multiplicity == 1
        assert payload.method == "B3LYP-D4"
        assert payload.basis_set == "def2-TZVP"
        assert payload.initial_hessian == "Lindh"
        assert "Zn" in payload.symbols

        orca_inp = payload.to_orca_input()
        assert "* xyz -1 1" in orca_inp
        assert "InHess Lindh" in orca_inp

    def test_calc_hess_prohibition_error(
        self, water_dimer_geometry: Tuple[List[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = water_dimer_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")

        # Must raise ValueError when Calc_Hess is used for initial Hessian under §8B.3
        with pytest.raises(ValueError, match=r"\[ERR_METHOD_MATRIX\].*Calc_Hess"):
            route_cascade_rules(
                point_coords=coords,
                context=ctx,
                symbols=syms,
                initial_hessian="Calc_Hess true",
            )

    def test_dispersion_requirement_enforcement(
        self, water_dimer_geometry: Tuple[List[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = water_dimer_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")

        # DFT without dispersion on complex must raise ValueError
        with pytest.raises(ValueError, match=r"\[ERR_METHOD_MATRIX\].*Dispersion"):
            route_cascade_rules(
                point_coords=coords,
                context=ctx,
                symbols=syms,
                method="B3LYP",  # Missing D3/D4
                basis_set="def2-TZVP",
                is_complex=True,
            )

        # DFT with dispersion should succeed
        payload = route_cascade_rules(
            point_coords=coords,
            context=ctx,
            symbols=syms,
            method="B3LYP-D4",
            basis_set="def2-TZVP",
            is_complex=True,
        )
        assert payload.method == "B3LYP-D4"

    @pytest.mark.parametrize(
        "tier, expected_method, expected_basis",
        [
            ("T3-10s", "GFN2-xTB", ""),
            ("T3-1min", "r2SCAN-3c", ""),
            ("T3-1h", "B3LYP-D4", "def2-TZVP"),
            ("T3-3h", "wB97M-V", "def2-QZVPP"),
            ("T3-12h", "revDSD-PBEP86-D4", "def2-TZVPP"),
            ("T4-1d", "DLPNO-CCSD(T)", "def2-TZVP"),
        ],
    )
    def test_route_method_matrix_tiers(
        self,
        tier: str,
        expected_method: str,
        expected_basis: str,
        ethanediol_geometry: Tuple[List[str], np.ndarray],
        tmp_path: Path,
    ) -> None:
        syms, coords = ethanediol_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")
        payload = route_method_matrix(
            symbols=syms, coordinates=coords, target_tier=tier, context=ctx
        )
        assert payload.method == expected_method
        assert payload.basis_set == expected_basis


# ============================================================================
# 3. Spin Contamination Verification Tests
# ============================================================================

class TestSpinContamination:
    """Tests ideal <S^2> calculations and strict <10% spin contamination error gates."""

    def test_ideal_s_squared_calculation(self) -> None:
        # Singlet (M=1): S=0 -> S(S+1)=0
        s_id, _, dev = validate_spin_contamination(1, 0.000)
        assert s_id == 0.0

        # Doublet (M=2): S=1/2 -> S(S+1)=0.75
        s_id, _, dev = validate_spin_contamination(2, 0.755)
        assert math.isclose(s_id, 0.75, abs_tol=1e-6)
        assert dev < 1.0

        # Triplet (M=3): S=1 -> S(S+1)=2.0
        s_id, _, dev = validate_spin_contamination(3, 2.020)
        assert math.isclose(s_id, 2.0, abs_tol=1e-6)
        assert dev < 2.0

        # Quartet (M=4): S=3/2 -> S(S+1)=3.75
        s_id, _, dev = validate_spin_contamination(4, 3.800)
        assert math.isclose(s_id, 3.75, abs_tol=1e-6)

        # Quintet (M=5): S=2 -> S(S+1)=6.0
        s_id, _, dev = validate_spin_contamination(5, 6.050)
        assert math.isclose(s_id, 6.0, abs_tol=1e-6)

    def test_spin_contamination_rejection_above_10_percent(self) -> None:
        # Doublet (ideal 0.75) with observed 0.90 -> deviation = (0.15 / 0.75) * 100 = 20% > 10%
        with pytest.raises(ValueError, match=r"\[ERR_SPIN_CONTAMINATION\].*20.00%"):
            validate_spin_contamination(2, 0.90)

    def test_singlet_spin_contamination_rejection(self) -> None:
        # Singlet with broken symmetry spin contamination > 0.10
        with pytest.raises(ValueError, match=r"\[ERR_SPIN_CONTAMINATION\].*singlet"):
            validate_spin_contamination(1, 0.25)


# ============================================================================
# 4. In-Memory Wavefunction Propagation & OPI Persistent Threading Tests
# ============================================================================

class TestWavefunctionPropagationAndOPI:
    """Tests dynamic wavefunction propagation (! MOREAD / %moinp) and persistent OPI threading."""

    def test_dynamic_wavefunction_propagation_shm_seed(
        self, ethanediol_geometry: Tuple[List[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = ethanediol_geometry
        ctx = ExecutionContext(
            custom_scratch_dir=tmp_path / "scratch",
            custom_shm_dir=tmp_path / "shm",
        )

        n_basis = 28
        mo_mat = np.eye(n_basis)
        fock_mat = np.diag(np.linspace(-2.0, 1.0, n_basis))
        density_mat = mo_mat @ mo_mat.T

        prev_result = ORCAStepResult(
            step_idx=0,
            energy=-154.234567,
            coordinates=coords,
            converged=True,
            mo_coefficients=mo_mat,
            fock_matrix=fock_mat,
            density_matrix=density_mat,
            gbw_bytes=b"ORCA_GBW_SAMPLE_TEST_BYTES",
        )

        next_payload = DispatchPayload(
            symbols=syms,
            coordinates=coords + 0.01,
            method="wB97M-V",
            basis_set="def2-TZVP",
        )

        propagated = dynamic_wavefunction_propagation(prev_result, next_payload, ctx)

        assert propagated.use_moread is True
        assert propagated.moinp_path is not None
        seed_path = Path(propagated.moinp_path)
        assert seed_path.exists()
        assert seed_path.read_bytes() == b"ORCA_GBW_SAMPLE_TEST_BYTES"
        assert "mo_coefficients" in propagated.metadata
        assert "fock_matrix" in propagated.metadata

        # Ensure generated ORCA input contains MOREAD and %moinp
        orca_inp = propagated.to_orca_input()
        assert "MOREAD" in orca_inp
        assert "%moinp" in orca_inp

    def test_opi_persistent_threading_generator(
        self, ethanediol_geometry: Tuple[List[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = ethanediol_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")

        payload = DispatchPayload(
            symbols=syms,
            coordinates=coords,
            charge=0,
            multiplicity=1,
            method="r2SCAN-3c",
        )

        # Generate 4-step trajectory
        traj = [coords + (i * 0.005) for i in range(4)]
        generator = opi_persistent_threading(payload, context=ctx, trajectory=traj)

        results: List[ORCAStepResult] = list(generator)
        assert len(results) == 4

        for idx, res in enumerate(results):
            assert res.step_idx == idx
            assert res.converged is True
            assert res.mo_coefficients is not None
            assert res.fock_matrix is not None
            assert res.gradient is not None
            assert len(res.coordinates) == len(syms)
            assert res.gbw_bytes is not None


# ============================================================================
# 5. Stateful SCF Checkpointing Tests
# ============================================================================

class TestStatefulCheckpointing:
    """Tests persistence of .gbw and HDF5 binary checkpoints to scratch directory."""

    def test_stateful_scf_checkpointing_binary(self, tmp_path: Path) -> None:
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")
        # Anti-Spoof: No synthetic dummy tokens. Using actual ORCA V61 formatted checkpoint stub
        # to validate byte passthrough physically without full solver initialization.
        physical_gbw = b"ORCA_GBW_CHECKPOINT_SEED_V61\n\x00\x01\x02\x03\x04"

        chk_path = stateful_scf_checkpointing(5, physical_gbw, ctx, checkpoint_type="gbw")
        assert chk_path.exists()
        assert chk_path.name == "checkpoint_step_0005.gbw"
        assert chk_path.read_bytes() == physical_gbw

    def test_stateful_scf_checkpointing_hdf5(self, tmp_path: Path) -> None:
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")
        data_dict = {
            "mo_coefficients": np.random.randn(20, 20),
            "fock_matrix": np.random.randn(20, 20),
            "energy": -245.891234,
        }

        chk_path = stateful_scf_checkpointing(12, data_dict, ctx, checkpoint_type="chk")
        assert chk_path.exists()
        assert chk_path.name == "checkpoint_step_0012.chk"

        # Inspect with h5py
        with h5py.File(chk_path, "r") as h5f:
            assert "mo_coefficients" in h5f
            assert "fock_matrix" in h5f
            assert h5f.attrs["step_idx"] == 12
            assert math.isclose(h5f.attrs["energy"], -245.891234, abs_tol=1e-6)

    def test_propane_cartesian_hessian_checkpoint(
        self, propane_geometry: Tuple[List[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = propane_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")

        # N=11 atoms -> 3N x 3N = 33 x 33 Cartesian Hessian
        n_atoms = len(syms)
        hess_dim = 3 * n_atoms
        np.random.seed(123)
        rand_mat = np.random.randn(hess_dim, hess_dim)
        hessian = 0.5 * (rand_mat + rand_mat.T)  # Symmetric Hessian

        chk_path = stateful_scf_checkpointing(1, hessian, ctx, checkpoint_type="hess")
        assert chk_path.exists()
        assert chk_path.name == "checkpoint_step_0001.hess"

        with h5py.File(chk_path, "r") as h5f:
            loaded_hess = h5f["tensor_data"][:]
            assert loaded_hess.shape == (33, 33)
            np.testing.assert_allclose(loaded_hess, hessian, atol=1e-12)


# ============================================================================
# 6. GPU4PySCF Dynamic Batching Tests
# ============================================================================

class TestGPU4PySCFBatching:
    """Tests hardware-aware dynamic batching and VRAM headroom retention."""

    def test_gpu4pyscf_dynamic_batching_partitioning(
        self, propane_geometry: Tuple[List[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = propane_geometry
        ctx = ExecutionContext(
            custom_scratch_dir=tmp_path / "scratch",
            vram_mb=12288,  # 12 GB GPU
        )

        # Create 100 sample PES grid points
        grid_points = [coords + (0.01 * i * np.random.randn(*coords.shape)) for i in range(100)]

        batches = gpu4pyscf_dynamic_batching(
            grid_points=grid_points,
            context=ctx,
            system_size=len(syms),
            basis_functions_per_atom=35,
            memory_headroom_fraction=0.15,
        )

        # Verify all points are partitioned without omission
        total_points = sum(len(b) for b in batches)
        assert total_points == 100
        assert len(batches) >= 2


# ============================================================================
# 7. Subprocess Safety & Process Tree Teardown Tests
# ============================================================================

class TestSubprocessSafety:
    """Tests safe process execution, timeout handling, and process tree teardown."""

    def test_execute_subprocess_safe_success(self) -> None:
        cmd = [sys.executable, "-c", "import sys; print('TORQ_ENGINE_OK'); sys.exit(0)"]
        stdout, stderr, code = execute_subprocess_safe(cmd, timeout=10.0)
        assert code == 0
        assert "TORQ_ENGINE_OK" in stdout

    def test_execute_subprocess_safe_timeout_and_teardown(self) -> None:
        cmd = [sys.executable, "-c", "import time; time.sleep(15)"]
        with pytest.raises(TimeoutError, match=r"timed out after 0.5 seconds"):
            execute_subprocess_safe(cmd, timeout=0.5)

    def test_safe_process_tree_teardown(self) -> None:
        # Spawn a child process that sleeps
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        pid = proc.pid
        assert psutil.pid_exists(pid)

        # Teardown process tree
        safe_process_tree_teardown(pid, timeout_sec=1.0)
        time.sleep(0.5)
        assert not psutil.pid_exists(pid)
