"""
CoChem-TORQ: Engine contract, mathematical input geometry and actual native-evidence tests
======================================================================
Phase 5 (Stage 4.0) Declared mathematical Test Matrix
--------------------------------------------------
Validates Method Matrix v4 execution cascade (defgrid1 -> defgrid3),
ORCA Python Interface (OPI) persistent memory threading, dynamic wavefunction
propagation (! MOREAD / %moinp), stateful SCF checkpointing, GPU4PySCF dynamic
batching with VRAM headroom protection, spin contamination validation (<10% threshold),
tightened intermolecular %geom criteria (TolMaxG 1e-5), frozen-monomer protocol,
prohibition of Calc_Hess true for initial Hessians, D3/D4 dispersion enforcement,
Counterpoise ghost atoms, dynamic atomic masses via Mendeleev, and 6-Tier Environment
Matrix scratch/shm path resolution.

Authoritative Standards:
- Method Matrix v4 (§4.4, §8A, §8B, §9A, §10, Table 2)
- Tripartite Filesystem Air-Gap Architecture (Domain A / B / C)
- Real Molecular Systems: Formic acid dimer, Water dimer, Zinc formate, 1,2-Ethanediol, Propane
"""

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psutil
import pytest
from mendeleev import element

from Libraries.cochem_torq_engine import (
    AirGapViolationError,
    DispatchPayload,
    EnvironmentTier,
    ExecutionContext,
    SpinContaminationError,
    detect_complex_and_monomers,
    detect_non_covalent_contacts,
    execute_subprocess_safe,
    get_atomic_mass,
    get_isotopic_mass,
    get_pyykko_radius,
    get_vdw_radius,
    gpu4pyscf_dynamic_batching,
    opi_persistent_threading,
    route_cascade_rules,
    route_method_matrix,
    safe_process_tree_teardown,
    stateful_scf_checkpointing,
    validate_spin_contamination,
)

# ============================================================================
# Declared mathematical Molecular Geometry Fixtures
# ============================================================================


@pytest.fixture
def water_dimer_geometry() -> tuple[list[str], np.ndarray]:
    """
    Declared model Water Dimer (H2O)2 geometry (Cs symmetry, R(O...O) = 2.91 A).
    """
    symbols = ["O", "H", "H", "O", "H", "H"]
    coords = np.array(
        [
            [0.0000, 0.0000, -1.4550],  # O1 donor
            [0.0000, 0.7600, -0.8650],  # H1 donor
            [0.0000, -0.7600, -0.8650],  # H2 donor
            [0.0000, 0.0000, 1.4550],  # O2 acceptor
            [0.7600, 0.0000, 2.0450],  # H3 acceptor
            [-0.7600, 0.0000, 2.0450],  # H4 acceptor
        ],
        dtype=np.float64,
    )
    return symbols, coords


@pytest.fixture
def formic_acid_dimer_geometry() -> tuple[list[str], np.ndarray]:
    """
    Declared model Formic Acid Dimer (HCOOH)2 (C2h symmetry, double H-bonded).
    """
    symbols = ["C", "O", "O", "H", "H", "C", "O", "O", "H", "H"]
    coords = np.array(
        [
            [0.0000, 1.8500, 0.0000],  # C1
            [-1.2200, 1.3500, 0.0000],  # O1 (=O)
            [1.2200, 1.3500, 0.0000],  # O2 (-OH)
            [1.2200, 0.3800, 0.0000],  # H1 (hydroxyl H)
            [0.0000, 2.9300, 0.0000],  # H2 (formyl H)
            [0.0000, -1.8500, 0.0000],  # C2
            [1.2200, -1.3500, 0.0000],  # O3 (=O)
            [-1.2200, -1.3500, 0.0000],  # O4 (-OH)
            [-1.2200, -0.3800, 0.0000],  # H3 (hydroxyl H)
            [0.0000, -2.9300, 0.0000],  # H4 (formyl H)
        ],
        dtype=np.float64,
    )
    return symbols, coords


@pytest.fixture
def zinc_formate_geometry() -> tuple[list[str], np.ndarray]:
    """
    Declared Zinc-shaped model(II) Formate complex [Zn(HCOO)3]^- geometry.
    """
    symbols = ["Zn", "C", "O", "O", "H", "C", "O", "O", "H", "C", "O", "O", "H"]
    coords = np.array(
        [
            [0.0000, 0.0000, 0.0000],  # Zn
            [2.3000, 0.0000, 0.0000],  # C1
            [1.6000, 1.0500, 0.0000],  # O1
            [1.6000, -1.0500, 0.0000],  # O2
            [3.3800, 0.0000, 0.0000],  # H1
            [-1.1500, 1.9919, 0.0000],  # C2
            [-0.1096, 1.9125, 0.0000],  # O3
            [-1.7096, 0.8625, 0.0000],  # O4
            [-1.6900, 2.9272, 0.0000],  # H2
            [-1.1500, -1.9919, 0.0000],  # C3
            [-1.7096, -0.8625, 0.0000],  # O5
            [-0.1096, -1.9125, 0.0000],  # O6
            [-1.6900, -2.9272, 0.0000],  # H3
        ],
        dtype=np.float64,
    )
    return symbols, coords


@pytest.fixture
def ethanediol_geometry() -> tuple[list[str], np.ndarray]:
    """
    Declared 1,2-ethanediol-shaped model (HO-CH2-CH2-OH) gauche conformer geometry.
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
def propane_geometry() -> tuple[list[str], np.ndarray]:
    """
    Declared propane-shaped model (C3H8) equilibrium geometry (N=11 atoms, 33x33 Hessian).
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
# 1. Mendeleev Dynamic Retrieval Tests (Mendeleev Mandate)
# ============================================================================


class TestMendeleevDynamicRetrieval:
    """Tests dynamic atomic mass, isotopic mass, and covalent/vdW radii retrieval via Mendeleev."""

    def test_dynamic_atomic_mass_retrieval(self) -> None:
        c_mass = get_atomic_mass("C")
        h_mass = get_atomic_mass("H")
        o_mass = get_atomic_mass("O")
        zn_mass = get_atomic_mass("Zn")

        assert math.isclose(c_mass, float(element("C").atomic_weight), rel_tol=1e-5)
        assert math.isclose(h_mass, float(element("H").atomic_weight), rel_tol=1e-5)
        assert math.isclose(o_mass, float(element("O").atomic_weight), rel_tol=1e-5)
        assert math.isclose(zn_mass, float(element("Zn").atomic_weight), rel_tol=1e-5)

    def test_dynamic_isotopic_mass_retrieval(self) -> None:
        c13_mass = get_isotopic_mass("C", 13)
        h2_mass = get_isotopic_mass("H", 2)

        assert 13.0 < c13_mass < 13.01
        assert 2.0 < h2_mass < 2.02

    def test_dynamic_radii_retrieval(self) -> None:
        c_cov = get_pyykko_radius("C")
        h_cov = get_pyykko_radius("H")
        c_vdw = get_vdw_radius("C")
        o_vdw = get_vdw_radius("O")

        assert math.isclose(c_cov, 0.75, abs_tol=0.05)
        assert math.isclose(h_cov, 0.32, abs_tol=0.05)
        assert c_vdw > 1.50
        assert o_vdw > 1.40


# ============================================================================
# 2. 6-Tier Environment Matrix & Path Resolution Tests
# ============================================================================


class TestEnvironmentMatrix:
    """Tests 6-Tier Environment Matrix detection, dynamic path resolution, and air-gap integrity."""

    def test_environment_tier_detection(self) -> None:
        """Exercise actual selectors without inheriting or changing host context."""
        import platform

        selectors = (
            "GITHUB_ACTIONS", "RUNNER_TEMP", "CODESPACES", "CODESPACE_NAME",
            "SLURM_TMPDIR", "SLURM_JOB_ID", "PFSDIR", "PBS_O_WORKDIR",
            "WSL_DISTRO_NAME",
        )
        host_tier = {
            "Windows": EnvironmentTier.LOCAL_WINDOWS,
            "Darwin": EnvironmentTier.LOCAL_MACOS,
        }.get(platform.system(), EnvironmentTier.LOCAL_LINUX)
        cases = [
            ({}, host_tier),
            ({"GITHUB_ACTIONS": "false", "CODESPACES": "false"}, host_tier),
            ({"GITHUB_ACTIONS": "true"}, EnvironmentTier.GITHUB_ACTIONS),
            ({"RUNNER_TEMP": "/tmp"}, EnvironmentTier.GITHUB_ACTIONS),
            ({"CODESPACES": "true"}, EnvironmentTier.CODESPACES),
            ({"CODESPACE_NAME": "explicit-test-context"}, EnvironmentTier.CODESPACES),
            ({"SLURM_TMPDIR": "/tmp"}, EnvironmentTier.HPC_NODES),
            ({"SLURM_JOB_ID": "123456"}, EnvironmentTier.HPC_NODES),
            ({"PFSDIR": "/tmp"}, EnvironmentTier.HPC_NODES),
            ({"PBS_O_WORKDIR": "/tmp"}, EnvironmentTier.HPC_NODES),
            ({"WSL_DISTRO_NAME": "explicit-test-context"}, EnvironmentTier.LOCAL_WINDOWS),
            ({"RUNNER_TEMP": "/tmp", "CODESPACES": "true", "SLURM_JOB_ID": "123456"},
             EnvironmentTier.GITHUB_ACTIONS),
            ({"CODESPACE_NAME": "explicit-test-context", "SLURM_JOB_ID": "123456",
              "WSL_DISTRO_NAME": "explicit-test-context"}, EnvironmentTier.CODESPACES),
            ({"PBS_O_WORKDIR": "/tmp", "WSL_DISTRO_NAME": "explicit-test-context"},
             EnvironmentTier.HPC_NODES),
        ]
        original_env = dict(os.environ)
        try:
            for configuration, expected in cases:
                for key in selectors:
                    os.environ.pop(key, None)
                os.environ.update(configuration)
                assert ExecutionContext.detect_tier() == expected, configuration
        finally:
            os.environ.clear()
            os.environ.update(original_env)
        assert dict(os.environ) == original_env

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

        sub_scratch = ctx.get_scratch_dir("sub_test")
        assert sub_scratch.exists() and sub_scratch.name == "sub_test"

    def test_air_gap_boundary_enforcement(self) -> None:
        ctx = ExecutionContext()
        scratch = ctx.get_scratch_dir()
        ctx.verify_air_gap_boundary(scratch)

        repo_root = Path(__file__).resolve().parent.parent
        with pytest.raises(AirGapViolationError):
            ctx.verify_air_gap_boundary(repo_root / "Libraries")


# ============================================================================
# 3. Method Matrix Routing & Cascade Rules Tests
# ============================================================================


class TestMethodMatrixCascade:
    """Tests Method Matrix v4 rules, complex detection, grid tightening, and initial Hessian enforcement."""

    def test_water_dimer_complex_detection(
        self, water_dimer_geometry: tuple[list[str], np.ndarray]
    ) -> None:
        syms, coords = water_dimer_geometry
        is_complex, components = detect_complex_and_monomers(syms, coords)
        assert is_complex is True
        assert len(components) == 2
        assert components[0] == [0, 1, 2]
        assert components[1] == [3, 4, 5]

    def test_formic_acid_dimer_frozen_monomer_routing(
        self, formic_acid_dimer_geometry: tuple[list[str], np.ndarray], tmp_path: Path
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
        assert payload.frozen_atom_indices == [0, 1, 2, 3, 4]
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
        self, formic_acid_dimer_geometry: tuple[list[str], np.ndarray], tmp_path: Path
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
        self, zinc_formate_geometry: tuple[list[str], np.ndarray], tmp_path: Path
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
        self, water_dimer_geometry: tuple[list[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = water_dimer_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")

        with pytest.raises(ValueError, match=r"\[ERR_METHOD_MATRIX\].*Calc_Hess"):
            route_cascade_rules(
                point_coords=coords,
                context=ctx,
                symbols=syms,
                initial_hessian="Calc_Hess true",
            )

    def test_dispersion_requirement_enforcement(
        self, water_dimer_geometry: tuple[list[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = water_dimer_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")

        with pytest.raises(ValueError, match=r"\[ERR_METHOD_MATRIX\].*Dispersion"):
            route_cascade_rules(
                point_coords=coords,
                context=ctx,
                symbols=syms,
                method="B3LYP",
                basis_set="def2-TZVP",
                is_complex=True,
            )

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
        ethanediol_geometry: tuple[list[str], np.ndarray],
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
# 4. Counterpoise & Ghost Atoms Tests
# ============================================================================


class TestCounterpoiseAndGhostAtoms:
    """Tests Counterpoise ghost atom routing and non-covalent contact detection."""

    def test_water_dimer_counterpoise_ghost_routing(
        self, water_dimer_geometry: tuple[list[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = water_dimer_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")

        payload = route_cascade_rules(
            point_coords=coords,
            context=ctx,
            symbols=syms,
            method="wB97M-V",
            basis_set="def2-TZVP",
            counterpoise=True,
        )

        assert payload.counterpoise is True
        assert payload.ghost_atom_indices == [3, 4, 5]

        orca_inp = payload.to_orca_input()
        assert "O: " in orca_inp
        assert "H: " in orca_inp

    def test_non_covalent_contact_detection(
        self, water_dimer_geometry: tuple[list[str], np.ndarray]
    ) -> None:
        syms, coords = water_dimer_geometry
        has_contacts, components, pairs = detect_non_covalent_contacts(syms, coords)
        assert has_contacts is True
        assert len(components) == 2
        assert len(pairs) > 0


# ============================================================================
# 5. Spin Contamination Verification Tests
# ============================================================================


class TestSpinContamination:
    """Tests ideal <S^2> calculations and strict <10% spin contamination error gates."""

    def test_ideal_s_squared_calculation(self) -> None:
        s_id, _, dev = validate_spin_contamination(1, 0.000)
        assert s_id == 0.0

        s_id, _, dev = validate_spin_contamination(2, 0.755)
        assert math.isclose(s_id, 0.75, abs_tol=1e-6)
        assert dev < 1.0

        s_id, _, dev = validate_spin_contamination(3, 2.020)
        assert math.isclose(s_id, 2.0, abs_tol=1e-6)
        assert dev < 2.0

        s_id, _, dev = validate_spin_contamination(4, 3.800)
        assert math.isclose(s_id, 3.75, abs_tol=1e-6)

        s_id, _, dev = validate_spin_contamination(5, 6.050)
        assert math.isclose(s_id, 6.0, abs_tol=1e-6)

    def test_spin_contamination_rejection_above_10_percent(self) -> None:
        with pytest.raises(
            SpinContaminationError, match=r"\[ERR_SPIN_CONTAMINATION\].*20.0%"
        ):
            validate_spin_contamination(2, 0.90)

    def test_singlet_spin_contamination_rejection(self) -> None:
        with pytest.raises(SpinContaminationError, match=r"Ideal = 0.0000"):
            validate_spin_contamination(1, 0.25)


# ============================================================================
# 6. In-Memory Wavefunction Propagation & OPI Persistent Threading Tests
# ============================================================================


class TestWavefunctionPropagationAndOPI:
    """Tests dynamic wavefunction propagation (! MOREAD / %moinp) and persistent OPI threading."""

    def test_opi_persistent_threading_generator(
        self, ethanediol_geometry, tmp_path: Path
    ) -> None:
        """Run ORCA only when installed; otherwise require an explicit missing-engine error."""
        import shutil

        symbols, coordinates = ethanediol_geometry
        context = ExecutionContext(
            custom_scratch_dir=tmp_path / "scratch", max_memory_mb=4000
        )
        payload = DispatchPayload(
            symbols=symbols,
            coordinates=coordinates,
            charge=0,
            multiplicity=1,
            method="r2SCAN-3c",
        )
        iterator = opi_persistent_threading(payload, context=context, n_steps=1)
        if not shutil.which(os.environ.get("ORCA_PATH", "orca")):
            with pytest.raises(FileNotFoundError, match="unavailable"):
                list(iterator)
        else:
            observed = list(iterator)
            assert len(observed) == 1
            assert observed[0].scf_converged and observed[0].normally_terminated
            assert observed[0].gradient is not None


# ============================================================================
# 7. Stateful SCF Checkpointing Tests
# ============================================================================


class TestStatefulCheckpointing:
    """Tests persistence of .gbw and HDF5 binary checkpoints to scratch directory."""

    def test_stateful_scf_checkpointing_binary(self, tmp_path: Path) -> None:
        """An unavailable native checkpoint is rejected instead of creating GBW bytes."""
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")
        with pytest.raises(ValueError, match="Empty checkpoint"):
            stateful_scf_checkpointing(5, b"", ctx, checkpoint_type="gbw")

    def test_stateful_scf_checkpointing_hdf5(self, tmp_path: Path) -> None:
        """Archive actual repository ORCA energy/gradient/coordinates as TORQ HDF5."""
        from Libraries.cochem_torq_engine import _read_orca_engrad

        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")
        energy, gradient, _, coordinates = _read_orca_engrad(
            Path(__file__).parents[1] / "test.engrad", 10
        )
        data = {"energy": energy, "gradient": gradient, "coordinates": coordinates}
        checkpoint = stateful_scf_checkpointing(12, data, ctx, checkpoint_type="chk")
        assert checkpoint.suffix == ".h5"
        with h5py.File(checkpoint, "r") as handle:
            np.testing.assert_array_equal(handle["gradient"][:], gradient)
            np.testing.assert_array_equal(handle["coordinates"][:], coordinates)
            assert handle.attrs["energy"] == energy
            assert handle.attrs["step_idx"] == 12

    def test_propane_cartesian_hessian_checkpoint(self, tmp_path: Path) -> None:
        """Persist a genuine H2 HF Hessian, rather than random data labeled propane."""
        pytest.importorskip("pyscf")
        from pyscf import gto, lib, scf

        old_threads = lib.num_threads()
        lib.num_threads(1)
        try:
            molecule = gto.M(
                atom="H 0 0 -.7; H 0 0 .7", unit="Bohr", basis="sto-3g", verbose=0
            )
            method = scf.RHF(molecule).run(conv_tol=1e-11)
            assert method.converged
            tensor = method.Hessian().kernel().transpose(0, 2, 1, 3).reshape(6, 6)
        finally:
            lib.num_threads(old_threads)
        context = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")
        checkpoint = stateful_scf_checkpointing(
            1, tensor, context, checkpoint_type="hess"
        )
        assert checkpoint.suffix == ".h5"
        with h5py.File(checkpoint, "r") as handle:
            np.testing.assert_array_equal(handle["tensor_data"][:], tensor)


# ============================================================================
# 8. GPU4PySCF Dynamic Batching Tests
# ============================================================================


class TestGPU4PySCFBatching:
    """Tests hardware-aware dynamic batching and VRAM headroom retention."""

    def test_gpu_batching_requires_observed_device_capacity(
        self, propane_geometry: tuple[list[str], np.ndarray], tmp_path: Path
    ) -> None:
        """Use actual hardware discovery; missing GPUs must never gain assumed VRAM."""
        symbols, coordinates = propane_geometry
        context = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")
        # Declared PES input coordinates; no engine energies are supplied or inferred.
        points = [coordinates + i * 0.001 for i in range(100)]
        if not context.gpu_available or context.vram_mb <= 0:
            with pytest.raises(ValueError, match="requires observed GPU memory"):
                gpu4pyscf_dynamic_batching(points, context, system_size=len(symbols))
        else:
            batches = gpu4pyscf_dynamic_batching(
                points, context, system_size=len(symbols), memory_headroom_fraction=0.15
            )
            assert [id(point) for batch in batches for point in batch] == [
                id(point) for point in points
            ]
            assert all(batch for batch in batches)


# ============================================================================
# 9. Subprocess Safety & Process Tree Teardown Tests
# ============================================================================


class TestSubprocessSafety:
    """Tests safe process execution, timeout handling, and process tree teardown."""

    def test_execute_subprocess_safe_success(self) -> None:
        cmd = [sys.executable, "-c", "import sys; print('TORQ_ENGINE_OK'); sys.exit(0)"]
        stdout, stderr, code = execute_subprocess_safe(cmd, timeout=10.0)
        assert code == 0
        assert "TORQ_ENGINE_OK" in stdout

    def test_execute_subprocess_safe_timeout_and_teardown(self) -> None:
        cmd = [
            sys.executable,
            "-c",
            "from rdkit import Chem; from rdkit.Chem import AllChem; m=Chem.AddHs(Chem.MolFromSmiles('C'*50)); AllChem.EmbedMolecule(m); [AllChem.MMFFOptimizeMolecule(m, maxIters=1000) for _ in range(50)]",
        ]
        with pytest.raises(TimeoutError, match=r"timed out after 0.5 seconds"):
            execute_subprocess_safe(cmd, timeout=0.5)

    def test_safe_process_tree_teardown(self) -> None:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from rdkit import Chem; from rdkit.Chem import AllChem; m=Chem.AddHs(Chem.MolFromSmiles('C'*50)); AllChem.EmbedMolecule(m); [AllChem.MMFFOptimizeMolecule(m, maxIters=1000) for _ in range(50)]",
            ]
        )
        pid = proc.pid
        assert psutil.pid_exists(pid)

        safe_process_tree_teardown(pid, timeout_sec=1.0)
        time.sleep(0.5)
        assert not psutil.pid_exists(pid)
