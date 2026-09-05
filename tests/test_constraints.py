"""Zero-Mock Physical Test Suite for CoChem-TORQ Constraints Manager.

Validates Phase 4 (Stage 3.0–3.5) Frozen-Monomer Spatial Protections, Rotational
Constant Sensitivity Budgeting, Dynamic Mendeleev Integration, and TolMaxG
Convergence Enforcement according to Method Matrix Sections 9A.1–9A.2 and 9A.7.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from Libraries.cochem_torq_constraints import (
    AMU_ANGSTROM2_TO_MHZ,
    BOHR_TO_ANGSTROM,
    HARTREE_PER_BOHR_TO_NEWTON,
    TOL_MAX_G_DEFAULT_EH_BOHR,
    TOL_RMS_G_DEFAULT_EH_BOHR,
    TOL_RESIDUAL_G_FROZEN_EH_BOHR,
    MonomerConstraintMode,
    ConstraintCoordinateType,
    MonomerFragment,
    IntermolecularMetrics,
    RotationalConstants,
    RotationalErrorBudget,
    ConvergenceAudit,
    ConstraintEnforcementResult,
    MethodMatrixConstraintError,
    ConstraintConvergenceError,
    SpatialCollisionError,
    FragmentDissociationError,
    get_dynamic_atomic_mass,
    get_dynamic_isotopic_mass,
    get_dynamic_pyykko_radius,
    get_dynamic_vdw_radius,
    get_dynamic_atomic_number,
    partition_monomer_fragments,
    compute_rotational_constants,
    propagate_rotational_error,
    audit_gradient_convergence,
    generate_orca_constraint_block,
    generate_cfour_constraint_block,
    TorqConstraintManager,
)


# =============================================================================
# 1. Mendeleev Dynamic Integration Tests
# =============================================================================

def test_dynamic_mendeleev_masses() -> None:
    """Verify dynamic atomic and isotopic mass retrieval via Mendeleev."""
    c_mass = get_dynamic_atomic_mass("C")
    h_mass = get_dynamic_atomic_mass("H")
    o_mass = get_dynamic_atomic_mass("O")

    assert 12.009 < c_mass < 12.013, f"Unexpected C mass: {c_mass}"
    assert 1.007 < h_mass < 1.009, f"Unexpected H mass: {h_mass}"
    assert 15.998 < o_mass < 16.002, f"Unexpected O mass: {o_mass}"

    # Isotopic masses
    c13_mass = get_dynamic_isotopic_mass("C", 13)
    assert 13.003 < c13_mass < 13.004, f"Unexpected 13C mass: {c13_mass}"


def test_dynamic_mendeleev_radii() -> None:
    """Verify dynamic Pyykkö and vdW radius retrieval via Mendeleev."""
    c_cov = get_dynamic_pyykko_radius("C")
    o_cov = get_dynamic_pyykko_radius("O")
    c_vdw = get_dynamic_vdw_radius("C")

    assert 0.70 <= c_cov <= 0.80, f"Unexpected C covalent radius: {c_cov}"
    assert 0.60 <= o_cov <= 0.70, f"Unexpected O covalent radius: {o_cov}"
    assert 1.50 <= c_vdw <= 1.90, f"Unexpected C vdW radius: {c_vdw}"

    z_c = get_dynamic_atomic_number("C")
    assert z_c == 6


# =============================================================================
# 2. Monomer Partitioning & Intermolecular Metrics Tests
# =============================================================================

def test_partition_monomers_dimer() -> None:
    """Verify covalent partitioning of CO2...H2O complex."""
    symbols = ["C", "O", "O", "O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],       # C (CO2)
        [0.0, 0.0, 1.162],     # O (CO2)
        [0.0, 0.0, -1.162],    # O (CO2)
        [2.836, 0.0, 0.0],     # O (H2O)
        [2.836, 0.757, 0.586], # H (H2O)
        [2.836, -0.757, 0.586] # H (H2O)
    ], dtype=np.float64)

    monomers, metrics = partition_monomer_fragments(
        symbols=symbols,
        coordinates=coords,
        constraint_mode=MonomerConstraintMode.FROZEN_ISO
    )

    assert len(monomers) == 2, f"Expected 2 monomers, got {len(monomers)}"
    assert monomers[0].molecular_formula in ("CO2", "O2C")
    assert monomers[1].molecular_formula in ("H2O", "OH2")
    assert monomers[0].atom_indices == [0, 1, 2]
    assert monomers[1].atom_indices == [3, 4, 5]

    assert not metrics.is_collision
    assert not metrics.is_dissociated
    assert 2.70 < metrics.center_of_mass_distance_angstrom < 3.00


def test_spatial_collision_detection() -> None:
    """Verify that unphysically close nuclear positions raise SpatialCollisionError."""
    symbols = ["O", "H", "H", "O", "H", "H"]
    colliding_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
        [0.5, 0.0, 0.0],      # Nuclear collision (0.5 Å separation)
        [0.5, 0.757, 0.586],
        [0.5, -0.757, 0.586]
    ], dtype=np.float64)

    mgr = TorqConstraintManager(collision_threshold_angstrom=1.20)
    with pytest.raises(SpatialCollisionError):
        mgr.analyze_and_enforce(symbols, colliding_coords)


def test_fragment_dissociation_detection() -> None:
    """Verify that runaway dissociated fragments raise FragmentDissociationError."""
    symbols = ["He", "Ne"]
    coords = np.array([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]], dtype=np.float64)

    mgr = TorqConstraintManager(dissociation_threshold_angstrom=12.0)
    with pytest.raises(FragmentDissociationError):
        mgr.analyze_and_enforce(symbols, coords)


# =============================================================================
# 3. Rotational Constants & Error Propagation Tests
# =============================================================================

def test_rotational_constants_water() -> None:
    """Verify rotational constants calculation for authentic water molecule."""
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586]
    ], dtype=np.float64)

    rot = compute_rotational_constants(symbols, coords)

    assert rot.I_a_amu_angstrom2 > 0.0
    assert rot.I_a_amu_angstrom2 <= rot.I_b_amu_angstrom2 <= rot.I_c_amu_angstrom2
    assert rot.A_MHz >= rot.B_MHz >= rot.C_MHz
    assert rot.rotor_type == "asymmetric"
    # Planar water has small inertial defect (close to zero for equilibrium geometry)
    assert abs(rot.inertial_defect_amu_angstrom2) < 0.05


def test_error_propagation_rule() -> None:
    """Verify Method Matrix Section 9A.1 & 4.5 error propagation: Delta B / B = -2 Delta R / R."""
    R = 2.836
    delta_R = 0.002
    monomer_err = 0.010

    budget = propagate_rotational_error(
        R_angstrom=R,
        delta_R_angstrom=delta_R,
        monomer_bond_error_angstrom=monomer_err
    )

    expected_delta_B_over_B = -2.0 * (delta_R / R) * 100.0
    assert math.isclose(budget.delta_B_over_B_percent, expected_delta_B_over_B, rel_tol=1e-5)
    # Monomer error sensitivity on A is significant (~ -1.7%)
    assert budget.delta_A_over_A_percent < -1.5
    # Equivalent monomer error is ~ 16.8–17.6 mÅ
    assert 0.015 < budget.equivalent_monomer_error_for_B_angstrom < 0.020


# =============================================================================
# 4. Strict TolMaxG & Convergence Auditing Tests
# =============================================================================

def test_gradient_convergence_success() -> None:
    """Verify gradient audit passes when gradient is within TolMaxG (1e-5) and TolRMSG (3e-6)."""
    gradient = np.full((6, 3), 2.0e-6, dtype=np.float64)
    audit = audit_gradient_convergence(
        gradient_eh_bohr=gradient,
        frozen_atom_indices=[0, 1, 2],
        tol_max_g=TOL_MAX_G_DEFAULT_EH_BOHR,
        tol_rms_g=TOL_RMS_G_DEFAULT_EH_BOHR
    )

    assert audit.converged
    assert not audit.deformation_channel_active
    assert not audit.escalation_required


def test_gradient_convergence_unconverged_free() -> None:
    """Verify gradient audit flags failure when free coordinates exceed TolMaxG."""
    gradient = np.zeros((6, 3), dtype=np.float64)
    # Free atom (index 3) has gradient 2.5e-5 > 1e-5
    gradient[3, 0] = 2.5e-5
    audit = audit_gradient_convergence(
        gradient_eh_bohr=gradient,
        frozen_atom_indices=[0, 1, 2],
        tol_max_g=TOL_MAX_G_DEFAULT_EH_BOHR,
        tol_rms_g=TOL_RMS_G_DEFAULT_EH_BOHR
    )

    assert not audit.converged
    assert audit.max_g_observed_eh_bohr == pytest.approx(2.5e-5)


def test_residual_gradient_deformation_escalation() -> None:
    """Verify residual gradient on frozen coordinates triggers deformation warning (§9A.1)."""
    gradient = np.zeros((6, 3), dtype=np.float64)
    # Free atoms converged (1e-6)
    gradient[3:6] = 1.0e-6
    # Frozen atom (index 1) has high residual gradient 4e-5 > 1e-5
    gradient[1, 2] = 4.0e-5

    audit = audit_gradient_convergence(
        gradient_eh_bohr=gradient,
        frozen_atom_indices=[0, 1, 2],
        tol_max_g=TOL_MAX_G_DEFAULT_EH_BOHR,
        tol_rms_g=TOL_RMS_G_DEFAULT_EH_BOHR
    )

    assert audit.converged  # Free coordinates are converged
    assert audit.deformation_channel_active  # Frozen residual gradient is non-negligible
    assert audit.escalation_required
    assert any("Deformation Warning" in note for note in audit.audit_notes)


# =============================================================================
# 5. ORCA & CFOUR Constraint Block Generation Tests
# =============================================================================

def test_orca_constraint_block_generation() -> None:
    """Verify generation of ORCA %geom constraint block with strict thresholds."""
    frozen_indices = [0, 1, 2]
    block = generate_orca_constraint_block(frozen_atom_indices=frozen_indices)

    assert "%geom" in block
    assert "TolMaxG 1e-5" in block
    assert "TolRMSG 3e-6" in block
    assert "TolE 1e-7" in block
    assert "Constraints" in block
    assert "{ C 0 C }" in block
    assert "{ C 1 C }" in block
    assert "{ C 2 C }" in block


def test_cfour_constraint_block_generation() -> None:
    """Verify generation of CFOUR ZMAT active / frozen variable markers."""
    zmat_vars = ["R_CO1", "R_CO2", "A_OCO", "R_COM", "A_TILT"]
    frozen_vars = ["R_CO1", "R_CO2", "A_OCO"]

    block = generate_cfour_constraint_block(zmat_vars, frozen_vars, geo_conv=5)

    assert "GEO_CONV=5" in block
    assert "GEO_MAXCYC=50" in block
    assert "R_CO1 = [FROZEN]" in block
    assert "R_COM* = [ACTIVE]" in block


# =============================================================================
# 6. End-to-End Manager Integration Tests
# =============================================================================

def test_manager_full_enforcement() -> None:
    """Verify complete execution of TorqConstraintManager on CO2...H2O complex."""
    symbols = ["C", "O", "O", "O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.162],
        [0.0, 0.0, -1.162],
        [2.836, 0.0, 0.0],
        [2.836, 0.757, 0.586],
        [2.836, -0.757, 0.586]
    ], dtype=np.float64)

    gradient = np.full((6, 3), 1.0e-6, dtype=np.float64)

    mgr = TorqConstraintManager()
    result: ConstraintEnforcementResult = mgr.analyze_and_enforce(
        symbols=symbols,
        coordinates=coords,
        constraint_mode=MonomerConstraintMode.FROZEN_ISO,
        gradient=gradient,
        freeze_monomer_index=0
    )

    assert result.success
    assert result.num_monomers == 2
    assert len(result.monomer_fragments) == 2
    assert result.convergence_audit is not None
    assert result.convergence_audit.converged
    assert "{ C 0 C }" in result.orca_geom_constraints_block
    assert "TolMaxG" in result.provenance_tags
