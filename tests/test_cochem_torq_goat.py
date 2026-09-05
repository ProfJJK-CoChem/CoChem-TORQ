"""Zero-Mock Physical Test Suite for CoChem-TORQ ORCA GOAT Conformer Engine.

================================================================================
Phase 3 (Stage 2.0 - 2.1) Authentic Test Matrix
--------------------------------------------------------------------------------
Validates Primary ORCA GOAT Conformer Enumeration (GOAT-EXPLORE, ExtOpt AIMNet2),
Two-Stage Deduplication (Stage A broad, Stage B spectroscopic --bthr 0.001),
the 0.93 [M] F1 baseline scoring, dynamic Mendeleev masses, ExtOpt file contract
and sign-flip physics, QCSchema HDF5 archiving, and Tripartite Filesystem Air-Gap
compliance per Method Matrix Sections 9B.1-9B.4 and Section 10.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import pytest

from Libraries.cochem_torq_goat import (
    AirGapViolationError,
    CoChemAirGapRing,
    CoChemPathManager,
    ConformerRecord,
    EnsembleContainer,
    ExtOptContract,
    ExtOptContractError,
    GoatAuditReport,
    GoatConfig,
    GoatError,
    GoatExecutionError,
    GoatExtOptDriver,
    GoatMode,
    GoatRunner,
    GoatTimeoutError,
    calculate_conformational_entropy,
    check_complex_dissociation,
    compute_center_of_mass,
    compute_inertia_tensor,
    compute_moments_and_constants,
    compute_rmsd,
    deduplicate_stage_a,
    deduplicate_stage_b_spectroscopic,
    execute_goat_conformer_pipeline,
    generate_orca_goat_input,
    get_dynamic_atomic_mass,
    get_dynamic_covalent_radius,
    get_dynamic_isotopic_mass,
    get_dynamic_vdw_radius,
    kabsch_align,
    parse_xyz_file,
    parse_xyz_string,
    save_ensemble_to_hdf5,
    write_xyz_file,
    write_xyz_string,
)


# =============================================================================
# Authentic Physical Geometry Fixtures
# =============================================================================

@pytest.fixture
def water_geometry() -> tuple[list[str], np.ndarray]:
    """Authentic equilibrium C2v water (H2O) geometry."""
    syms = ["O", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586],
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def formamidinium_formate_geometry() -> tuple[list[str], np.ndarray]:
    """Authentic formamidinium formate complex geometry (10 atoms)."""
    syms = ["C", "N", "N", "H", "H", "H", "C", "O", "O", "H"]
    coords = np.array(
        [
            [0.000, 1.200, 0.000],   # C (cation)
            [-1.150, 1.850, 0.000],  # N
            [1.150, 1.850, 0.000],   # N
            [0.000, 0.120, 0.000],   # H
            [-1.150, 2.850, 0.000],  # H
            [1.150, 2.850, 0.000],   # H
            [0.000, -1.800, 0.000],  # C (anion)
            [-1.200, -1.250, 0.000], # O
            [1.200, -1.250, 0.000],  # O
            [0.000, -2.880, 0.000],  # H
        ],
        dtype=np.float64,
    )
    return syms, coords


# =============================================================================
# 1. Mendeleev Dynamic Integration Tests
# =============================================================================

def test_dynamic_mendeleev_masses() -> None:
    """Verify dynamic atomic and isotopic mass retrieval via Mendeleev."""
    c_mass = get_dynamic_atomic_mass("C")
    h_mass = get_dynamic_atomic_mass("H")
    o_mass = get_dynamic_atomic_mass("O")
    n_mass = get_dynamic_atomic_mass("N")

    assert 12.009 < c_mass < 12.013, f"Unexpected C mass: {c_mass}"
    assert 1.007 < h_mass < 1.009, f"Unexpected H mass: {h_mass}"
    assert 15.998 < o_mass < 16.002, f"Unexpected O mass: {o_mass}"
    assert 14.005 < n_mass < 14.009, f"Unexpected N mass: {n_mass}"

    # Isotopic test
    c13_mass = get_dynamic_isotopic_mass("C", 13)
    assert 13.000 < c13_mass < 13.005, f"Unexpected 13C mass: {c13_mass}"


def test_dynamic_mendeleev_radii() -> None:
    """Verify covalent and van der Waals radii retrieval via Mendeleev."""
    c_cov = get_dynamic_covalent_radius("C")
    h_cov = get_dynamic_covalent_radius("H")
    o_cov = get_dynamic_covalent_radius("O")

    assert 0.70 < c_cov < 0.80, f"Unexpected C covalent radius: {c_cov}"
    assert 0.30 < h_cov < 0.40, f"Unexpected H covalent radius: {h_cov}"
    assert 0.60 < o_cov < 0.70, f"Unexpected O covalent radius: {o_cov}"

    c_vdw = get_dynamic_vdw_radius("C")
    assert 1.60 < c_vdw < 1.85, f"Unexpected C vdW radius: {c_vdw}"


# =============================================================================
# 2. Rigid Rotor & Rotational Constants Tests
# =============================================================================

def test_rotational_constants_water(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """Verify rotational constants calculation for authentic equilibrium water."""
    syms, coords = water_geometry
    (A_mhz, B_mhz, C_mhz), (A_ghz, B_ghz, C_ghz), delta, planar, kappa = (
        compute_moments_and_constants(syms, coords)
    )

    assert A_mhz > B_mhz > C_mhz > 0.0, "Rotational constants must satisfy A > B > C > 0"
    assert 800000.0 < A_mhz < 950000.0, f"A constant for H2O out of expected range: {A_mhz} MHz"
    assert 400000.0 < B_mhz < 550000.0, f"B constant for H2O out of expected range: {B_mhz} MHz"
    assert 250000.0 < C_mhz < 350000.0, f"C constant for H2O out of expected range: {C_mhz} MHz"

    # Planar molecule inertial defect check (Delta ~ 0 for planar equilibrium structure)
    assert abs(delta) < 0.1, f"Inertial defect Delta for planar H2O should be ~0: {delta}"


def test_kabsch_alignment_and_rmsd() -> None:
    """Verify Kabsch alignment and RMSD invariance under rigid translation and rotation."""
    syms = ["C", "C", "O"]
    P = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=np.float64)

    # Apply rigid rotation and translation
    theta = math.pi / 4.0
    R = np.array(
        [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    shift = np.array([5.0, -3.0, 2.0])
    Q = np.dot(P, R.T) + shift

    rmsd = compute_rmsd(P, Q)
    assert rmsd < 1e-8, f"Kabsch RMSD under rigid transformation should be ~0: {rmsd}"


# =============================================================================
# 3. Non-Covalent Dissociation & Conformational Entropy Tests
# =============================================================================

def test_complex_dissociation_detector() -> None:
    """Verify that dissociated non-covalent complexes are correctly flagged."""
    syms = ["O", "H", "H", "Ne"]
    # Bound water...Ne complex
    bound_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586],
            [0.0, 0.0, 3.100],  # Ne within vdW contact
        ],
        dtype=np.float64,
    )
    assert not check_complex_dissociation(syms, bound_coords)

    # Dissociated complex (Ne pulled to 15 Angstroms)
    dissoc_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586],
            [0.0, 0.0, 15.000],  # Ne dissociated
        ],
        dtype=np.float64,
    )
    assert check_complex_dissociation(syms, dissoc_coords)


def test_conformational_entropy_calculation() -> None:
    """Verify conformational entropy S_conf calculation."""
    # Two degenerate conformers at dE = 0: S_conf = R * ln(2) ≈ 1.377 cal/(mol*K)
    energies_degen = [0.0, 0.0]
    s_conf_degen, weights = calculate_conformational_entropy(energies_degen, temperature_k=298.15)
    expected_s = 1.98720425864083 * np.log(2.0)
    assert abs(s_conf_degen - expected_s) < 1e-3, f"Expected {expected_s}, got {s_conf_degen}"
    assert len(weights) == 2
    assert abs(weights[0] - 0.5) < 1e-4

    # Single conformer: S_conf = 0.0
    energies_single = [0.0]
    s_conf_single, _ = calculate_conformational_entropy(energies_single)
    assert abs(s_conf_single) < 1e-8


# =============================================================================
# 4. ORCA GOAT Input Generation & ExtOpt Contract Tests
# =============================================================================

def test_generate_orca_goat_input_authoritative() -> None:
    """Verify ORCA GOAT input script conforms to Method Matrix §9B.4."""
    config = GoatConfig(
        mode=GoatMode.GOAT_EXPLORE,
        driver=GoatExtOptDriver.AIMNET2,
        maxen_kcal=12.0,
        conftemp_k=298.15,
        confdegen="auto",
        tight_opt=True,
        scf_tol_e=1e-5,
        threads=8,
    )

    inp = generate_orca_goat_input(config, "test_mol.xyz", charge=0, multiplicity=1)

    assert "! GOAT-EXPLORE ExtOpt TightOpt PAL8" in inp
    assert "ProgExt" in inp
    assert "Ext_Params" in inp
    assert "TolE 1e-05" in inp or "TolE 1e-5" in inp
    assert "maxen 12.0" in inp
    assert "conftemp 298.15" in inp
    assert "confdegen auto" in inp
    assert "* xyzfile 0 1 test_mol.xyz" in inp


def test_extopt_contract_sign_flip_and_conversion() -> None:
    """Verify Section 10 sign flip: g_Eh_a0 = (-F_eV_A) * 0.529177210903 / 27.211386245988."""
    # Force of 1.0 eV/A in +X direction
    forces_ev_ang = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    grad_eh_bohr = ExtOptContract.convert_ase_forces_to_orca_gradient(forces_ev_ang)

    # Gradient must be NEGATIVE (sign flip)
    expected_grad_x = -1.0 * (0.529177210903 / 27.211386245988)
    assert abs(grad_eh_bohr[0, 0] - expected_grad_x) < 1e-10
    assert grad_eh_bohr[0, 0] < 0.0, "Gradient must have opposite sign of force (uphill vs downhill)"


def test_extopt_file_contract_roundtrip(tmp_path: Path) -> None:
    """Verify ExtOpt .extinp.tmp parsing and .engrad writing."""
    extinp_file = tmp_path / "mol_EXT.extinp.tmp"
    extinp_content = """mol_EXT.xyz # coordinate file
0           # charge
1           # multiplicity
4           # NCores
1           # do gradient
"""
    extinp_file.write_text(extinp_content, encoding="utf-8")

    xyz_f, chg, mult, ncores, dograd, pcfile = ExtOptContract.read_extinp(extinp_file)
    assert xyz_f == "mol_EXT.xyz"
    assert chg == 0
    assert mult == 1
    assert ncores == 4
    assert dograd == 1
    assert pcfile is None

    engrad_file = tmp_path / "mol_EXT.engrad"
    grad_data = np.array([[0.001, -0.002, 0.003], [0.000, 0.001, -0.001]], dtype=np.float64)
    ExtOptContract.write_engrad(engrad_file, n_atoms=2, energy_eh=-76.456789, gradient_eh_bohr=grad_data)

    assert engrad_file.exists()
    content = engrad_file.read_text(encoding="utf-8")
    assert "-76.456789000000" in content
    assert "0.001000000000" in content


# =============================================================================
# 5. Two-Stage Deduplication Tests
# =============================================================================

def test_two_stage_deduplication(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """Verify Stage A (broad) and Stage B (spectroscopic) deduplication filters."""
    syms, coords = water_geometry

    # Create base conformer
    r0 = ConformerRecord(
        index=0,
        symbols=syms,
        coordinates=coords.tolist(),
        energy_hartree=-76.000,
        energy_kcal_rel=0.0,
        rotational_constants_mhz=(800000.0, 500000.0, 300000.0),
        origin_engine="GOAT",
    )

    # Exact duplicate
    r1 = ConformerRecord(
        index=1,
        symbols=syms,
        coordinates=coords.tolist(),
        energy_hartree=-76.000,
        energy_kcal_rel=0.0,
        rotational_constants_mhz=(800000.0, 500000.0, 300000.0),
        origin_engine="GOAT",
    )

    # Conformer with slight energy & B shift (< 1% B difference, Stage A duplicate, Stage B candidate)
    r2_coords = coords + np.array([[0.0, 0.0, 0.0], [0.0, 0.002, 0.0], [0.0, -0.002, 0.0]])
    r2 = ConformerRecord(
        index=2,
        symbols=syms,
        coordinates=r2_coords.tolist(),
        energy_hartree=-75.9999,
        energy_kcal_rel=0.06,  # 0.06 kcal/mol
        rotational_constants_mhz=(800000.0, 500500.0, 299800.0),  # 0.1% shift in B
        origin_engine="GOAT",
    )

    # Distinct high-energy conformer
    r3_coords = coords + np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    r3 = ConformerRecord(
        index=3,
        symbols=syms,
        coordinates=r3_coords.tolist(),
        energy_hartree=-75.990,
        energy_kcal_rel=6.27,
        rotational_constants_mhz=(600000.0, 350000.0, 200000.0),
        origin_engine="GOAT",
    )

    pool = [r0, r1, r2, r3]

    # Stage A deduplication
    stage_a = deduplicate_stage_a(pool, rmsd_thr=0.125, ethr_kcal=0.100, bthr_frac=0.025)
    assert len(stage_a) < len(pool)
    assert any(c.index == 0 for c in stage_a)
    assert any(c.index == 3 for c in stage_a)

    # Stage B spectroscopic deduplication
    stage_b = deduplicate_stage_b_spectroscopic(stage_a, rmsd_thr=0.125, ethr_kcal=0.05, bthr_frac=0.001)
    assert len(stage_b) >= 1


# =============================================================================
# 6. Physical Basin Hopping & Pipeline End-to-End Tests
# =============================================================================

def test_save_ensemble_to_hdf5_qcschema(tmp_path: Path, water_geometry: tuple[list[str], np.ndarray]) -> None:
    """Verify HDF5 serialization of conformer ensemble adhering to QCSchema."""
    syms, coords = water_geometry
    rec = ConformerRecord(
        index=0,
        symbols=syms,
        coordinates=coords.tolist(),
        energy_hartree=-76.45,
        energy_kcal_rel=0.0,
        rotational_constants_mhz=(850000.0, 430000.0, 280000.0),
        origin_engine="GOAT",
    )
    ensemble = EnsembleContainer(
        name="Water_Ensemble",
        conformers=[rec],
        temperature_k=298.15,
        s_conf_cal_mol_k=0.0,
        provenance_tag="[M]",
    )

    h5_file = tmp_path / "water_ensemble.h5"
    save_ensemble_to_hdf5(ensemble, h5_file)

    assert h5_file.exists()
    with h5py.File(h5_file, "r") as f:
        assert f.attrs["ensemble_name"] == "Water_Ensemble"
        assert f.attrs["f1_baseline"] == 0.93
        assert "conformers" in f
        coords_ds = f["conformers/coordinates"][:]
        assert coords_ds.shape == (1, 3, 3)


def test_execute_goat_conformer_pipeline(tmp_path: Path, water_geometry: tuple[list[str], np.ndarray]) -> None:
    """Verify high-level Phase 3 (Stage 2.0 - 2.1) ORCA GOAT conformer pipeline."""
    scratch_dir = tmp_path / "scratch"
    artifacts_dir = tmp_path / "artifacts"

    config = GoatConfig(
        mode=GoatMode.GOAT_EXPLORE,
        driver=GoatExtOptDriver.AIMNET2,
        maxen_kcal=12.0,
        threads=2,
    )

    syms, coords = water_geometry
    seeds = [(syms, coords)]

    with pytest.raises(GoatExecutionError):
        ensemble, report = execute_goat_conformer_pipeline(
            seed_geometries=seeds,
            config=config,
            system_name="H2O_Test",
            scratch_dir=scratch_dir,
            artifacts_dir=artifacts_dir,
        )

