"""CREST adapter contracts, actual HF numerical inputs, and missing-evidence handling.

No conformer-search completeness, CREST execution, or benchmark F1 is inferred
from mathematical geometry checks or serializer roundtrips.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import pytest

from Libraries.cochem_torq_crest import (
    AirGapViolationError,
    CoChemAirGapRing,
    CoChemPathManager,
    ConformerRecord,
    CregenConfig,
    CregenReferee,
    CrestConfig,
    CrestError,
    CrestExecutionError,
    CrestRunner,
    CrestTimeoutError,
    EnsembleContainer,
    UnionAuditReport,
    UnionConformerReferee,
    calculate_conformational_entropy,
    check_complex_dissociation,
    compute_center_of_mass,
    compute_moments_and_constants,
    compute_rmsd,
    deduplicate_spectroscopic,
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
def hfip_ne_complex_geometry() -> tuple[list[str], np.ndarray]:
    """Declared 13-atom complex-shaped input; no binding or optimization is established."""
    syms = ["C", "C", "C", "O", "H", "H", "F", "F", "F", "F", "F", "F", "Ne"]
    # Model coordinates for 13-atom vdW complex
    coords = np.array(
        [
            [0.000, 0.000, 0.000],  # C (central)
            [1.250, 0.850, 0.000],  # C (CF3)
            [-1.250, 0.850, 0.000],  # C (CF3)
            [0.000, -0.800, 1.150],  # O
            [0.000, -0.650, -0.890],  # H (CH)
            [0.000, -1.700, 0.850],  # H (OH)
            [1.300, 1.650, 1.080],  # F
            [2.350, 0.050, 0.000],  # F
            [1.300, 1.650, -1.080],  # F
            [-1.300, 1.650, 1.080],  # F
            [-2.350, 0.050, 0.000],  # F
            [-1.300, 1.650, -1.080],  # F
            [0.000, -3.200, 2.800],  # Ne (declared contact position)
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
    ne_mass = get_dynamic_atomic_mass("Ne")

    assert c_mass == 12.0, f"Unexpected dominant-isotope C mass: {c_mass}"
    assert 1.00782 < h_mass < 1.00783, f"Unexpected H-1 mass: {h_mass}"
    assert 15.99491 < o_mass < 15.99492, f"Unexpected O-16 mass: {o_mass}"
    assert 19.99243 < ne_mass < 19.99245, f"Unexpected Ne-20 mass: {ne_mass}"


def test_dynamic_mendeleev_radii() -> None:
    """Verify covalent and van der Waals radii retrieval via Mendeleev."""
    c_cov = get_dynamic_covalent_radius("C")
    h_cov = get_dynamic_covalent_radius("H")
    ne_vdw = get_dynamic_vdw_radius("Ne")
    ar_vdw = get_dynamic_vdw_radius("Ar")

    assert 0.70 < c_cov < 0.80, f"Unexpected C covalent radius: {c_cov}"
    assert 0.30 < h_cov < 0.35, f"Unexpected H covalent radius: {h_cov}"
    assert 1.40 < ne_vdw < 1.70, f"Unexpected Ne vdW radius: {ne_vdw}"
    assert 1.80 < ar_vdw < 2.00, f"Unexpected Ar vdW radius: {ar_vdw}"


# =============================================================================
# 2. Moments of Inertia & Rotational Constants
# =============================================================================


def test_water_moments_and_rotational_constants(
    water_geometry: tuple[list[str], np.ndarray],
) -> None:
    """Verify rigid-rotor mathematics on a declared water-shaped input."""
    syms, coords = water_geometry
    res = compute_moments_and_constants(syms, coords)

    a_mhz, b_mhz, c_mhz = res["rotational_constants_mhz"]
    delta = res["inertial_defect_u_a2"]

    # Water is an asymmetric top with A > B > C
    assert a_mhz > b_mhz > c_mhz > 0.0
    assert 700000.0 < a_mhz < 900000.0, f"Unexpected A: {a_mhz}"
    assert 350000.0 < b_mhz < 500000.0, f"Unexpected B: {b_mhz}"
    assert 250000.0 < c_mhz < 350000.0, f"Unexpected C: {c_mhz}"

    # Planar molecule must have inertial defect Delta ≈ 0.0 u*Å^2 [M]
    assert abs(delta) < 1e-4, f"Non-zero planar inertial defect: {delta}"


def test_center_of_mass(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """Verify mass-weighted center of mass calculation."""
    syms, coords = water_geometry
    com = compute_center_of_mass(syms, coords)
    assert com.shape == (3,)
    assert abs(com[0]) < 1e-12, "COM x should be zero by symmetry"


# =============================================================================
# 3. Kabsch Alignment & RMSD Tests
# =============================================================================


def test_kabsch_alignment_invariance(
    water_geometry: tuple[list[str], np.ndarray],
) -> None:
    """Verify Kabsch alignment is invariant under translation and pure rotation."""
    syms, coords = water_geometry

    # Apply rigid translation + rotation
    theta = np.radians(45.0)
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed_coords = np.dot(coords, rot.T) + np.array([2.5, -1.0, 0.5])

    rmsd_val, aligned_p = kabsch_align(transformed_coords, coords)
    assert rmsd_val < 1e-10, f"Kabsch RMSD non-zero under rigid rotation: {rmsd_val}"


# =============================================================================
# 4. Complex Dissociation Check
# =============================================================================


def test_check_complex_dissociation(
    hfip_ne_complex_geometry: tuple[list[str], np.ndarray],
) -> None:
    """Verify vdW complex dissociation detection."""
    syms, coords = hfip_ne_complex_geometry

    # The declared contact geometry passes the distance heuristic
    is_dissoc_bound = check_complex_dissociation(syms, coords, max_vdw_factor=2.4)
    assert not is_dissoc_bound, "Bound HFIP...Ne complex falsely flagged as dissociated"

    # Separate Ne fragment by 15 Angstroms
    dissociated_coords = coords.copy()
    dissociated_coords[-1] += np.array([0.0, 0.0, 15.0])

    is_dissoc_split = check_complex_dissociation(
        syms, dissociated_coords, max_vdw_factor=2.4
    )
    assert is_dissoc_split, "Dissociated complex was not detected"


# =============================================================================
# 5. Multi-Structure XYZ I/O Tests
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
                rotational_constants_mhz=moments["rotational_constants_mhz"],
                origin_engine="PySCF/HF/STO-3G",
                provenance_tag="actual_native_evidence",
            )
        )
    minimum = min(record.energy_hartree for record in records)
    for record in records:
        record.energy_kcal_rel = (record.energy_hartree - minimum) * 627.5094740631
    return records


def test_multi_xyz_serialization_and_parsing(
    tmp_path: Path, water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Roundtrip actual HF/STO-3G energies; these are not CREST/GOAT results."""
    symbols, coordinates = water_geometry
    displaced = coordinates + np.array([[0, 0, 0], [0, 0.02, 0], [0, -0.02, 0]])
    records = _real_hf_records(symbols, [coordinates, displaced], tmp_path / "hf")
    parsed = parse_xyz_string(write_xyz_string(records))
    assert len(parsed) == 2
    for source, restored in zip(records, parsed):
        assert restored.symbols == symbols
        assert restored.energy_hartree == pytest.approx(
            source.energy_hartree, abs=1e-10
        )
        assert np.allclose(restored.coordinates, source.coordinates, atol=5e-9)


# =============================================================================
# 6. CREST Command Line Construction & Validation
# =============================================================================


def test_crest_command_line_flags() -> None:
    """Verify that CrestRunner strictly includes mandatory flags (--nci, --nocross, --noreftopo, --ewin 12)."""
    cfg = CrestConfig(
        crest_bin="crest",
        ewin_kcal=12.0,
        nci=True,
        nocross=True,
        noreftopo=True,
        gfn_level="gfn2",
        threads=8,
        niceprint=True,
    )
    runner = CrestRunner(config=cfg)
    cmd = runner.build_command_line("input.xyz")

    assert "crest" in cmd[0]
    assert "input.xyz" in cmd[1]
    assert "--nci" in cmd, "Mandatory --nci flag missing"
    assert "--nocross" in cmd, "Mandatory --nocross flag missing"
    assert "--noreftopo" in cmd, "Mandatory --noreftopo flag missing"
    assert "--gfn2" in cmd, "Mandatory --gfn2 flag missing"
    assert "--ewin" in cmd and "12.0" in cmd, "Mandatory --ewin 12.0 flag missing"
    assert "--T" in cmd and "8" in cmd, "Thread count missing"
    assert "--niceprint" in cmd, "--niceprint flag missing"


# =============================================================================
# 7. CREGEN Referee & Deduplication Engine Tests
# =============================================================================


def test_cregen_referee_deduplication(
    tmp_path: Path, water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Genuine HF observations cannot be culled by an unverified identity sieve."""
    symbols, coordinates = water_geometry
    near = coordinates + np.array([[0, 0, 0], [0, 0.0001, 0], [0, -0.0001, 0]])
    distant = coordinates + np.array([[0, 0, 0], [0, 0.08, 0], [0, -0.08, 0]])
    records = _real_hf_records(symbols, [coordinates, near, distant], tmp_path / "hf")
    referee = CregenReferee(
        config=CregenConfig(ewin_kcal=100.0, ethr_kcal=0.05, rthr_angstrom=0.125)
    )
    survivors = referee.referee_ensemble(records)
    assert len(survivors) == len(records)
    assert all(retained is original for retained, original in zip(survivors, records))
    assert [record.index for record in survivors] == [
        record.index for record in records
    ]
    assert all(record.origin_engine == "PySCF/HF/STO-3G" for record in survivors)
    assert all(
        record.energy_hartree in {item.energy_hartree for item in records}
        for record in survivors
    )


# =============================================================================
# 8. Two-Stage Spectroscopic Deduplication Tests
# =============================================================================


def test_spectroscopic_deduplication_tight_threshold(
    tmp_path: Path, water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Numerical agreement cannot discard genuine observations without identity."""
    symbols, coordinates = water_geometry
    near = coordinates + np.array([[0, 0, 0], [0, 0.0003, 0], [0, -0.0003, 0]])
    records = _real_hf_records(symbols, [coordinates, near], tmp_path / "hf")
    survivors = deduplicate_spectroscopic(records, bthr_spectroscopic=0.001)
    assert len(survivors) == len(records)
    assert all(retained is original for retained, original in zip(survivors, records))


# =============================================================================
# 9. The 6-Step Master Union Protocol Orchestrator Tests
# =============================================================================


def test_union_protocol_full_orchestration(
    hfip_ne_complex_geometry: tuple[list[str], np.ndarray],
) -> None:
    """A supplied seed is not a completed GOAT ensemble or validated union."""
    symbols, coordinates = hfip_ne_complex_geometry
    referee = UnionConformerReferee()
    with pytest.raises(ValueError, match="energy-model provenance"):
        referee.execute_union_protocol(
            seed_inputs=[(symbols, coordinates)], system_name="Missing_GOAT"
        )
    with pytest.raises(FileNotFoundError, match="completed GOAT ensemble"):
        referee.execute_union_protocol(
            seed_inputs=[(symbols, coordinates)],
            system_name="Missing_GOAT",
            goat_energy_model="gfn2",
        )


# =============================================================================
# 10. FAIR-Compliant HDF5 Serialization Tests
# =============================================================================


def test_hdf5_serialization_and_attributes(
    tmp_path: Path, water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """An empty observed ensemble must not invent F1 or thermodynamic results."""
    ensemble = EnsembleContainer(name="No_calculation", conformers=[])
    report = UnionAuditReport(
        system_name="No_calculation",
        n_seeds=0,
        n_goat_raw=0,
        n_crest_raw=0,
        n_union_raw=0,
        n_goat_unique=None,
        n_crest_unique=None,
        n_shared_intersection=None,
        n_survivors_stage_a=0,
        n_survivors_stage_b=0,
        provenance_tag="serialization_only_no_engine_execution",
    )
    path = save_ensemble_to_hdf5(ensemble, report, tmp_path / "unavailable.h5")
    with h5py.File(path, "r") as archive:
        assert archive.attrs["system_name"] == "No_calculation"
        assert "goat_f1_baseline" not in archive.attrs
        assert "crest_f1_baseline" not in archive.attrs
        assert "s_conf_union_cal_mol_k" not in archive.attrs
        assert archive["conformers"].attrs["count"] == 0
        assert not list(archive["conformers"].keys())


# =============================================================================
# 11. Tripartite Filesystem Air-Gap Security Tests
# =============================================================================


def test_air_gap_violation_prevention() -> None:
    """Verify AirGapViolationError is raised when trying to write inside Ring 1 repo."""
    repo_root = CoChemPathManager.get_repo_root()
    illegal_target = repo_root / "illegal_output.xyz"

    with pytest.raises(AirGapViolationError):
        CoChemPathManager.assert_air_gap(illegal_target)
