"""Zero-Mock Physical Test Suite for CoChem-TORQ CREST & Union Referee Engine.

================================================================================
Phase 3 (Stage 2.0 - 2.1) Authentic Test Matrix
--------------------------------------------------------------------------------
Validates Independent CREST Conformer Exploration (--nci --nocross --noreftopo),
CREGEN Refereeing, Two-Stage Deduplication (Stage A broad, Stage B spectroscopic),
the 6-Step Union Protocol, 0.93 [M] F1 baseline scoring, dynamic Mendeleev masses,
and Tripartite Filesystem Air-Gap compliance per Method Matrix Sections 9B.1-9B.4.
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
def hfip_ne_complex_geometry() -> tuple[list[str], np.ndarray]:
    """Authentic hexafluoroisopropanol...Neon van der Waals complex geometry."""
    syms = ["C", "C", "C", "O", "H", "H", "F", "F", "F", "F", "F", "F", "Ne"]
    # Model coordinates for 13-atom vdW complex
    coords = np.array(
        [
            [0.000, 0.000, 0.000],   # C (central)
            [1.250, 0.850, 0.000],   # C (CF3)
            [-1.250, 0.850, 0.000],  # C (CF3)
            [0.000, -0.800, 1.150],  # O
            [0.000, -0.650, -0.890], # H (CH)
            [0.000, -1.700, 0.850],  # H (OH)
            [1.300, 1.650, 1.080],   # F
            [2.350, 0.050, 0.000],   # F
            [1.300, 1.650, -1.080],  # F
            [-1.300, 1.650, 1.080],  # F
            [-2.350, 0.050, 0.000],  # F
            [-1.300, 1.650, -1.080], # F
            [0.000, -3.200, 2.800],  # Ne (vdW bound)
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

    assert 12.009 < c_mass < 12.013, f"Unexpected C mass: {c_mass}"
    assert 1.007 < h_mass < 1.009, f"Unexpected H mass: {h_mass}"
    assert 15.998 < o_mass < 16.002, f"Unexpected O mass: {o_mass}"
    assert 20.170 < ne_mass < 20.190, f"Unexpected Ne mass: {ne_mass}"


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
    water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Verify moments of inertia and rotational constants on equilibrium H2O."""
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
    water_geometry: tuple[list[str], np.ndarray]
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
    hfip_ne_complex_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Verify vdW complex dissociation detection."""
    syms, coords = hfip_ne_complex_geometry

    # Bound complex should not be dissociated
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

def test_multi_xyz_serialization_and_parsing(
    water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Verify lossless roundtrip serialization and parsing of multi-structure XYZ files."""
    syms, coords = water_geometry

    phys1 = compute_moments_and_constants(syms, coords)
    rec1 = ConformerRecord(
        index=0,
        symbols=syms,
        coordinates=coords.tolist(),
        energy_hartree=-76.425,
        energy_kcal_rel=0.0,
        rotational_constants_mhz=phys1["rotational_constants_mhz"],
        rotational_constants_ghz=phys1["rotational_constants_ghz"],
        inertial_defect_u_a2=phys1["inertial_defect_u_a2"],
        planar_moments_u_a2=phys1["planar_moments_u_a2"],
        ray_asymmetry_kappa=phys1["ray_asymmetry_kappa"],
        origin_engine="GOAT",
    )

    perturbed_coords = coords + np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, -0.02, 0.0]]
    )
    phys2 = compute_moments_and_constants(syms, perturbed_coords)
    rec2 = ConformerRecord(
        index=1,
        symbols=syms,
        coordinates=perturbed_coords.tolist(),
        energy_hartree=-76.423,
        energy_kcal_rel=1.255,
        rotational_constants_mhz=phys2["rotational_constants_mhz"],
        rotational_constants_ghz=phys2["rotational_constants_ghz"],
        inertial_defect_u_a2=phys2["inertial_defect_u_a2"],
        planar_moments_u_a2=phys2["planar_moments_u_a2"],
        ray_asymmetry_kappa=phys2["ray_asymmetry_kappa"],
        origin_engine="CREST",
    )

    xyz_str = write_xyz_string([rec1, rec2])
    parsed_records = parse_xyz_string(xyz_str)

    assert len(parsed_records) == 2
    assert parsed_records[0].symbols == syms
    assert parsed_records[1].symbols == syms
    assert abs(parsed_records[0].energy_hartree - (-76.425)) < 1e-4


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
    water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Verify CREGEN 5-step sieve deduplication and threshold enforcement."""
    syms, coords = water_geometry

    # Build exact duplicate + slightly displaced copy within RMSD threshold 0.125 Å
    phys = compute_moments_and_constants(syms, coords)
    rec1 = ConformerRecord(
        index=0,
        symbols=syms,
        coordinates=coords.tolist(),
        energy_hartree=-76.425,
        energy_kcal_rel=0.0,
        rotational_constants_mhz=phys["rotational_constants_mhz"],
        rotational_constants_ghz=phys["rotational_constants_ghz"],
        inertial_defect_u_a2=phys["inertial_defect_u_a2"],
        planar_moments_u_a2=phys["planar_moments_u_a2"],
        origin_engine="GOAT",
    )

    # Near-duplicate: 0.001 Å displacement (within 0.125 Å RMSD, bthr 1.0% and 0.01 ΔE)
    near_coords = coords + np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, -0.001, 0.0]]
    )
    phys_near = compute_moments_and_constants(syms, near_coords)
    rec2 = ConformerRecord(
        index=1,
        symbols=syms,
        coordinates=near_coords.tolist(),
        energy_hartree=-76.42502,
        energy_kcal_rel=0.01,
        rotational_constants_mhz=phys_near["rotational_constants_mhz"],
        rotational_constants_ghz=phys_near["rotational_constants_ghz"],
        inertial_defect_u_a2=phys_near["inertial_defect_u_a2"],
        planar_moments_u_a2=phys_near["planar_moments_u_a2"],
        origin_engine="CREST",
    )

    # Distinct isomer: 0.35 Å displacement (beyond 0.125 Å RMSD)
    distinct_coords = coords + np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.35, 0.0], [0.0, -0.35, 0.0]]
    )
    phys_dist = compute_moments_and_constants(syms, distinct_coords)
    rec3 = ConformerRecord(
        index=2,
        symbols=syms,
        coordinates=distinct_coords.tolist(),
        energy_hartree=-76.420,
        energy_kcal_rel=3.137,
        rotational_constants_mhz=phys_dist["rotational_constants_mhz"],
        rotational_constants_ghz=phys_dist["rotational_constants_ghz"],
        inertial_defect_u_a2=phys_dist["inertial_defect_u_a2"],
        planar_moments_u_a2=phys_dist["planar_moments_u_a2"],
        origin_engine="CREST",
    )

    referee = CregenReferee(
        config=CregenConfig(ewin_kcal=12.0, ethr_kcal=0.05, rthr_angstrom=0.125)
    )
    survivors = referee.referee_ensemble([rec1, rec2, rec3])

    # Near-duplicate should be filtered out, leaving 2 unique conformers
    assert len(survivors) == 2, f"Expected 2 unique conformers, got {len(survivors)}"
    assert survivors[0].origin_engine == "UNION"
    assert survivors[1].origin_engine == "UNION"


# =============================================================================
# 8. Two-Stage Spectroscopic Deduplication Tests
# =============================================================================

def test_spectroscopic_deduplication_tight_threshold(
    water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Verify Stage B Spectroscopic Deduplication tightens threshold to Delta B/B <= 0.1%."""
    syms, coords = water_geometry
    phys = compute_moments_and_constants(syms, coords)

    rec1 = ConformerRecord(
        index=0,
        symbols=syms,
        coordinates=coords.tolist(),
        energy_hartree=-76.425,
        energy_kcal_rel=0.0,
        rotational_constants_mhz=phys["rotational_constants_mhz"],
        origin_engine="GOAT",
    )

    # Conformer with ΔB/B ≈ 0.05% (below 0.1% microwave resolution)
    b_perturbed_coords = coords + np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0003, 0.0], [0.0, -0.0003, 0.0]]
    )
    phys_p = compute_moments_and_constants(syms, b_perturbed_coords)
    rec2 = ConformerRecord(
        index=1,
        symbols=syms,
        coordinates=b_perturbed_coords.tolist(),
        energy_hartree=-76.425,
        energy_kcal_rel=0.001,
        rotational_constants_mhz=phys_p["rotational_constants_mhz"],
        origin_engine="CREST",
    )

    survivors = deduplicate_spectroscopic([rec1, rec2], bthr_spectroscopic=0.001)
    assert len(survivors) == 1, "Spectroscopically indistinguishable conformer was not merged"


# =============================================================================
# 9. The 6-Step Master Union Protocol Orchestrator Tests
# =============================================================================

def test_union_protocol_full_orchestration(
    hfip_ne_complex_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Verify complete execution of 6-Step Union Protocol, F1 scoring, and audit metrics."""
    syms, coords = hfip_ne_complex_geometry

    referee = UnionConformerReferee()
    ensemble, report = referee.execute_union_protocol(
        seed_inputs=[(syms, coords)],
        system_name="HFIP_Ne_Benchmark",
    )

    assert isinstance(ensemble, EnsembleContainer)
    assert isinstance(report, UnionAuditReport)
    assert len(ensemble.conformers) >= 1
    assert report.system_name == "HFIP_Ne_Benchmark"
    assert report.goat_f1_baseline == 0.93, "GOAT F1 benchmark baseline must equal 0.93 [M]"
    assert report.crest_f1_baseline == 0.77, "CREST F1 baseline must reflect 0.74-0.80 [M]"
    assert "Both GOAT and CREST are stochastic global optimizers" in report.completeness_disclaimer
    assert report.provenance_tag == "[M]"


# =============================================================================
# 10. FAIR-Compliant HDF5 Serialization Tests
# =============================================================================

def test_hdf5_serialization_and_attributes(
    water_geometry: tuple[list[str], np.ndarray]
) -> None:
    """Verify HDF5 QCSchema serialization and persistent state preservation."""
    syms, coords = water_geometry
    referee = UnionConformerReferee()
    ensemble, report = referee.execute_union_protocol(
        seed_inputs=[(syms, coords)],
        system_name="Water_HDF5_Test",
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        h5_path = Path(tmp.name)

    try:
        saved_path = save_ensemble_to_hdf5(ensemble, report, h5_path)
        assert saved_path.exists()

        with h5py.File(str(saved_path), "r") as f:
            assert f.attrs["system_name"] == "Water_HDF5_Test"
            assert f.attrs["goat_f1_baseline"] == 0.93
            assert "conformers" in f
            assert "union_audit" in f
            confs_grp = f["conformers"]
            assert confs_grp.attrs["count"] == len(ensemble.conformers)
            assert "conformer_0000" in confs_grp
            c0 = confs_grp["conformer_0000"]
            assert "rotational_constants_mhz" in c0
            assert "coordinates" in c0
            assert "symbols" in c0
    finally:
        if h5_path.exists():
            h5_path.unlink()


# =============================================================================
# 11. Tripartite Filesystem Air-Gap Security Tests
# =============================================================================

def test_air_gap_violation_prevention() -> None:
    """Verify AirGapViolationError is raised when trying to write inside Ring 1 repo."""
    repo_root = CoChemPathManager.get_repo_root()
    illegal_target = repo_root / "illegal_output.xyz"

    with pytest.raises(AirGapViolationError):
        CoChemPathManager.assert_air_gap(illegal_target)
