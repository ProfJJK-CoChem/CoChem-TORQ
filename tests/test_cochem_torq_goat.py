"""GOAT adapter contracts, real archived data, and genuine HF deduplication inputs.

These tests do not qualify ORCA GOAT conformer-search completeness or accuracy.
Declared geometries are input examples, never inferred equilibrium structures.
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
# Declared Geometry Inputs
# =============================================================================


@pytest.fixture
def water_geometry() -> tuple[list[str], np.ndarray]:
    """Declared C2v water-shaped geometry for numerical checks."""
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
            [0.000, 1.200, 0.000],  # C (cation)
            [-1.150, 1.850, 0.000],  # N
            [1.150, 1.850, 0.000],  # N
            [0.000, 0.120, 0.000],  # H
            [-1.150, 2.850, 0.000],  # H
            [1.150, 2.850, 0.000],  # H
            [0.000, -1.800, 0.000],  # C (anion)
            [-1.200, -1.250, 0.000],  # O
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

    assert c_mass == 12.0, f"Unexpected C-12 mass: {c_mass}"
    assert 1.00782 < h_mass < 1.00783, f"Unexpected H-1 mass: {h_mass}"
    assert 15.99491 < o_mass < 15.99492, f"Unexpected O-16 mass: {o_mass}"
    assert 14.00307 < n_mass < 14.00308, f"Unexpected N-14 mass: {n_mass}"

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


def test_rotational_constants_water(
    water_geometry: tuple[list[str], np.ndarray],
) -> None:
    """Verify rigid-rotor constants for a declared water-shaped input."""
    syms, coords = water_geometry
    (A_mhz, B_mhz, C_mhz), (A_ghz, B_ghz, C_ghz), delta, planar, kappa = (
        compute_moments_and_constants(syms, coords)
    )

    assert A_mhz > B_mhz > C_mhz > 0.0, (
        "Rotational constants must satisfy A > B > C > 0"
    )
    assert 800000.0 < A_mhz < 950000.0, (
        f"A constant for H2O out of expected range: {A_mhz} MHz"
    )
    assert 400000.0 < B_mhz < 550000.0, (
        f"B constant for H2O out of expected range: {B_mhz} MHz"
    )
    assert 250000.0 < C_mhz < 350000.0, (
        f"C constant for H2O out of expected range: {C_mhz} MHz"
    )

    # Planar molecule inertial defect check (Delta ~ 0 for declared planar input)
    assert abs(delta) < 0.1, (
        f"Inertial defect Delta for planar H2O should be ~0: {delta}"
    )


def test_kabsch_alignment_and_rmsd() -> None:
    """Verify Kabsch alignment and RMSD invariance under rigid translation and rotation."""
    syms = ["C", "C", "O"]
    P = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=np.float64)

    # Apply rigid rotation and translation
    theta = math.pi / 4.0
    R = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
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
    s_conf_degen, weights = calculate_conformational_entropy(
        energies_degen, temperature_k=298.15
    )
    expected_s = 1.98720425864083 * np.log(2.0)
    assert abs(s_conf_degen - expected_s) < 1e-3, (
        f"Expected {expected_s}, got {s_conf_degen}"
    )
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
    assert grad_eh_bohr[0, 0] < 0.0, (
        "Gradient must have opposite sign of force (uphill vs downhill)"
    )


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
    from Libraries.cochem_torq_engine import _read_orca_engrad

    archive = Path(__file__).resolve().parents[1] / "test.engrad"
    energy, gradient, atom_numbers, coordinates = _read_orca_engrad(archive, 10)
    ExtOptContract.write_engrad(
        engrad_file,
        n_atoms=len(atom_numbers),
        energy_eh=energy,
        gradient_eh_bohr=gradient,
    )
    # The ExtOpt format deliberately contains no coordinate block: compare its
    # numeric payload with the unchanged, genuinely archived ORCA calculation.
    content = engrad_file.read_text(encoding="utf-8")
    numbers = [
        float(line)
        for line in content.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert int(numbers[0]) == len(atom_numbers)
    assert numbers[1] == pytest.approx(energy, abs=5e-13)
    assert np.allclose(np.asarray(numbers[2:]).reshape(10, 3), gradient, atol=5e-13)


# =============================================================================
# 5. Two-Stage Deduplication Tests
# =============================================================================


def _real_hf_records(
    symbols: list[str], geometries: list[np.ndarray], workspace: Path
) -> list[ConformerRecord]:
    """Evaluate every input with actual HF/STO-3G and retain native evidence."""
    pytest.importorskip("pyscf")
    from scipy.constants import physical_constants

    from cochem_torq.engines import PySCFBackend

    bohr_angstrom = physical_constants["Bohr radius"][0] / 1e-10
    records = []
    for index, coordinates in enumerate(geometries):
        result = PySCFBackend().evaluate(
            {
                "molecule": {
                    "symbols": symbols,
                    "geometry_bohr": (coordinates / bohr_angstrom).tolist(),
                    "charge": 0,
                    "multiplicity": 1,
                },
                "method": {
                    "name": "hf",
                    "basis": "sto-3g",
                    "reference": "restricted",
                    "frozen_core": False,
                },
                "properties": ["energy"],
            },
            workspace / str(index),
        )
        assert result["status"] == "complete"
        moments = compute_moments_and_constants(symbols, coordinates)
        records.append(
            ConformerRecord(
                index=index,
                symbols=symbols,
                coordinates=coordinates.tolist(),
                energy_hartree=result["energy_hartree"],
                rotational_constants_mhz=moments[0],
                rotational_constants_ghz=moments[1],
                inertial_defect_u_a2=moments[2],
                planar_moments_u_a2=moments[3],
                ray_asymmetry_kappa=moments[4],
                origin_engine="PySCF/HF/STO-3G",
                provenance_tag="actual_native_evidence",
            )
        )
    minimum = min(record.energy_hartree for record in records)
    for record in records:
        record.energy_kcal_rel = (record.energy_hartree - minimum) * 627.5094740631
    return records


def test_two_stage_deduplication(
    tmp_path: Path, water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Deduplicate actual same-method HF inputs; no search completeness is claimed."""
    symbols, coordinates = water_geometry
    near = coordinates + np.array([[0, 0, 0], [0, 0.002, 0], [0, -0.002, 0]])
    distant = coordinates + np.array([[0, 0, 0], [0, 0.08, 0], [0, -0.08, 0]])
    pool = _real_hf_records(
        symbols, [coordinates, coordinates.copy(), near, distant], tmp_path / "hf"
    )
    stage_a = deduplicate_stage_a(
        pool, rmsd_thr=0.125, ethr_kcal=0.100, bthr_frac=0.025
    )
    assert len(stage_a) < len(pool)
    assert any(c.index == 0 for c in stage_a)
    assert any(c.index == 3 for c in stage_a)
    assert not any(c.index == 1 for c in stage_a)
    stage_b = deduplicate_stage_b_spectroscopic(
        stage_a, rmsd_thr=0.125, ethr_kcal=0.05, bthr_frac=0.001
    )
    assert len(stage_b) >= 1


# =============================================================================
# 6. Physical Basin Hopping & Pipeline End-to-End Tests
# =============================================================================


def test_save_ensemble_to_hdf5_qcschema(
    tmp_path: Path, water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Archive a genuinely calculated energy and geometrically derived constants."""
    symbols, coordinates = water_geometry
    record = _real_hf_records(symbols, [coordinates], tmp_path / "hf")[0]
    ensemble = EnsembleContainer(
        name="HF_Input_Ensemble",
        conformers=[record],
        temperature_k=298.15,
        s_conf_cal_mol_k=None,
        provenance_tag="actual_native_evidence",
    )
    target = tmp_path / "ensemble.h5"
    save_ensemble_to_hdf5(ensemble, target)
    with h5py.File(target, "r") as archive:
        assert archive.attrs["ensemble_name"] == "HF_Input_Ensemble"
        assert "f1_baseline" not in archive.attrs
        assert "s_conf_cal_mol_k" not in archive.attrs
        assert np.array_equal(archive["conformers/coordinates"][:], [coordinates])
        assert archive["conformers/energies_hartree"][0] == record.energy_hartree
        assert np.array_equal(
            archive["conformers/rotational_constants_mhz"][0],
            record.rotational_constants_mhz,
        )


def test_execute_goat_conformer_pipeline(
    tmp_path: Path, water_geometry: tuple[list[str], np.ndarray]
) -> None:
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


def test_rotor_undefined_axes_are_not_finite_floored_values() -> None:
    """Check exact one-point and two-point mass geometry, not engine observations."""
    from scipy.constants import atomic_mass, h

    atom = compute_moments_and_constants(["He"], np.array([[0.0, 0.0, 0.0]]))
    assert atom[0] == (None, None, None)
    assert atom[1] == (None, None, None)
    assert atom[4] is None
    coordinates = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    linear = compute_moments_and_constants(["H", "H"], coordinates)
    assert linear[0][0] is None and linear[1][0] is None
    inertia_si = 0.5 * get_dynamic_atomic_mass("H") * atomic_mass * 1e-20
    expected_b = h / (8 * np.pi**2 * inertia_si * 1e6)
    assert linear[0][1] == pytest.approx(expected_b, rel=1e-10)
    assert linear[0][2] == pytest.approx(expected_b, rel=1e-10)
    assert linear[4] is None
    with pytest.raises(ValueError, match="nonempty finite"):
        compute_moments_and_constants([], np.empty((0, 3)))
    with pytest.raises(ValueError, match="Coincident nuclei"):
        compute_moments_and_constants(["H", "H"], np.zeros((2, 3)))


def test_undefined_linear_axis_cannot_authorize_three_axis_pruning(
    tmp_path: Path,
) -> None:
    coordinates = np.array([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]])
    first = _real_hf_records(["H", "H"], [coordinates], tmp_path / "hf")[0]
    second = first.model_copy(
        update={"index": 1}
    )  # Same genuine request, no invented measurement.
    assert first.rotational_constants_mhz[0] is None
    assert len(deduplicate_stage_a([first, second])) == 2
    assert len(deduplicate_stage_b_spectroscopic([first, second])) == 2
    ensemble = EnsembleContainer(name="Linear_actual_HF", conformers=[first])
    path = tmp_path / "linear.h5"
    save_ensemble_to_hdf5(ensemble, path)
    with h5py.File(path, "r") as archive:
        assert np.isnan(archive["conformers/rotational_constants_mhz"][0, 0])
        assert np.isfinite(archive["conformers/rotational_constants_mhz"][0, 1:]).all()
        assert "unavailable" in archive.attrs["missing_physical_values"]


def test_search_configuration_never_asserts_unrun_benchmark_f1() -> None:
    assert GoatConfig().f1_baseline is None
    with pytest.raises(ValueError):
        GoatConfig(f1_baseline=0.93)
