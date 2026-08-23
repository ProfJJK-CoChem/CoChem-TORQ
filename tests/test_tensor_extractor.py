"""CoChem-TORQ: Physical Test Suite for Quantum Tensor Harvester (Stage 4.1).

Phase 6 Validation Suite
-------------------------------------------------------------------------------
Validates:
1. Exact CODATA 2022 constants & CIAAW/AME2020 mass tables.
2. Moments of inertia, principal axes diagonalization, planar moments,
   and inertial defect (Delta) for 3D asymmetric, symmetric, and planar systems:
   - H2O (asymmetric prolate, planar, Delta = 0)
   - SO2 (asymmetric prolate, planar, Delta = 0)
   - H2CO (near-prolate asymmetric, planar, Delta = 0)
   - CH3Cl (prolate symmetric top, kappa = -1, Ib = Ic)
   - Benzene C6H6 (oblate symmetric top, kappa = +1, Ia = Ib, planar)
   - CH4 (spherical top, Ia = Ib = Ic, A = B = C)
3. Cartesian Protections & Linearity Trap for linear/quasi-linear systems:
   - CO2, OCS, HCN (Ia = 0, collinear backbone, cylindrical projection, DOF=2)
   - Quasi-linear floppy complex singularity damping.
4. Ray's Asymmetry Parameter (kappa) and Dynamic Representation Switching:
   - All 6 representations (Ir, Il, IIr, IIl, IIIr, IIIl)
   - Right-handed permutation matrix determinants (+1)
   - Wang Hamiltonian sub-blocks [E+, E-, O+, O-].
5. ORCA VPT2, Coriolis, and Centrifugal Distortion parsing with divergence checks.
6. HDF5 / JSON structured export gateways with Air-Gap directory compliance.
7. Anti-spoofing verification and empirical physical fidelity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

from Libraries.cochem_tensor_extractor import (
    AMU_A2_TO_MHZ,
    AMU_TO_KG,
    ANGSTROM_TO_M,
    ATOMIC_MASS_CONSTANT_U,
    C_M_S,
    C_ROT_CM1,
    C_ROT_GHZ,
    C_ROT_MHZ,
    CODATA_YEAR,
    PLANCK_CONSTANT_JS,
    SPEED_OF_LIGHT_C,
    AsymmetryResult,
    CartesianProtectionResult,
    InertiaTensorResult,
    TorqTensorExtractor,
    TorqTensorOutput,
    apply_cartesian_protections,
    calculate_rays_asymmetry,
    diagonalize_inertia_tensor,
    dynamic_representation_switch,
    get_atomic_mass,
)

# =============================================================================
# Test Suite 1: Exact Physical Constants & Isotopic Mass Tables
# =============================================================================


def test_exact_physical_constants_codata_2022() -> None:
    """Validates physical constants against exact CODATA 2022 standard."""
    assert CODATA_YEAR == 2022
    assert PLANCK_CONSTANT_JS == 6.62607015e-34
    assert SPEED_OF_LIGHT_C == 299792458.0
    assert C_M_S == SPEED_OF_LIGHT_C
    assert ATOMIC_MASS_CONSTANT_U == 1.66053906892e-27
    assert AMU_TO_KG == ATOMIC_MASS_CONSTANT_U
    assert ANGSTROM_TO_M == 1.0e-10

    # Verify analytical derivation of rotational conversion factor:
    # C_rot = h / (8 * pi^2 * u * 1e-20) * 1e-6 (MHz * u * Angstrom^2)
    expected_c_rot = (
        PLANCK_CONSTANT_JS
        / (8.0 * (math.pi**2) * ATOMIC_MASS_CONSTANT_U * (ANGSTROM_TO_M**2))
    ) * 1e-6
    assert abs(C_ROT_MHZ - expected_c_rot) < 1e-9
    assert abs(C_ROT_MHZ - 505379.008435) < 1e-3
    assert AMU_A2_TO_MHZ == C_ROT_MHZ

    # Verify GHz and cm^-1 conversions
    assert abs(C_ROT_GHZ - (C_ROT_MHZ * 1e-3)) < 1e-9
    expected_c_rot_cm1 = (C_ROT_MHZ * 1e6) / (SPEED_OF_LIGHT_C * 100.0)
    assert abs(C_ROT_CM1 - expected_c_rot_cm1) < 1e-9


def test_isotopic_mass_table_accuracy_and_parsing() -> None:
    """Validates CIAAW / AME2020 mono-isotopic mass lookups via mendeleev."""
    import mendeleev
    
    # H
    elem_h = mendeleev.element("H")
    h1_mass = float(next(i for i in elem_h.isotopes if i.mass_number == 1).mass)
    h2_mass = float(next(i for i in elem_h.isotopes if i.mass_number == 2).mass)
    h3_mass = float(next(i for i in elem_h.isotopes if i.mass_number == 3).mass)
    assert abs(get_atomic_mass("H") - h1_mass) < 1e-8
    assert abs(get_atomic_mass("1H") - h1_mass) < 1e-8
    assert abs(get_atomic_mass("D") - h2_mass) < 1e-8
    assert abs(get_atomic_mass("2H") - h2_mass) < 1e-8
    assert abs(get_atomic_mass("T") - h3_mass) < 1e-8
    assert abs(get_atomic_mass("3H") - h3_mass) < 1e-8

    # C
    elem_c = mendeleev.element("C")
    most_abundant_c = sorted([i for i in elem_c.isotopes if i.abundance is not None], key=lambda x: x.abundance, reverse=True)[0]
    c_mass = float(most_abundant_c.mass)
    c12_mass = float(next(i for i in elem_c.isotopes if i.mass_number == 12).mass)
    c13_mass = float(next(i for i in elem_c.isotopes if i.mass_number == 13).mass)
    
    assert abs(get_atomic_mass("C") - c_mass) < 1e-8
    assert abs(get_atomic_mass("12C") - c12_mass) < 1e-8
    assert abs(get_atomic_mass("13C") - c13_mass) < 1e-8
    assert abs(get_atomic_mass("C13") - c13_mass) < 1e-8

    # N
    elem_n = mendeleev.element("N")
    most_abundant_n = sorted([i for i in elem_n.isotopes if i.abundance is not None], key=lambda x: x.abundance, reverse=True)[0]
    n_mass = float(most_abundant_n.mass)
    n15_mass = float(next(i for i in elem_n.isotopes if i.mass_number == 15).mass)
    assert abs(get_atomic_mass("N") - n_mass) < 1e-8
    assert abs(get_atomic_mass("15N") - n15_mass) < 1e-8

    # O
    elem_o = mendeleev.element("O")
    most_abundant_o = sorted([i for i in elem_o.isotopes if i.abundance is not None], key=lambda x: x.abundance, reverse=True)[0]
    o_mass = float(most_abundant_o.mass)
    o18_mass = float(next(i for i in elem_o.isotopes if i.mass_number == 18).mass)
    assert abs(get_atomic_mass("O") - o_mass) < 1e-8
    assert abs(get_atomic_mass("18O") - o18_mass) < 1e-8
    assert abs(get_atomic_mass("O18") - o18_mass) < 1e-8

    # Other atoms
    elem_cl = mendeleev.element("Cl")
    cl35_mass = float(next(i for i in elem_cl.isotopes if i.mass_number == 35).mass)
    cl37_mass = float(next(i for i in elem_cl.isotopes if i.mass_number == 37).mass)
    assert abs(get_atomic_mass("35Cl") - cl35_mass) < 1e-8
    assert abs(get_atomic_mass("37Cl") - cl37_mass) < 1e-8
    
    elem_br = mendeleev.element("Br")
    br79_mass = float(next(i for i in elem_br.isotopes if i.mass_number == 79).mass)
    br81_mass = float(next(i for i in elem_br.isotopes if i.mass_number == 81).mass)
    assert abs(get_atomic_mass("79Br") - br79_mass) < 1e-8
    assert abs(get_atomic_mass("81Br") - br81_mass) < 1e-8

    elem_i = mendeleev.element("I")
    most_abundant_i = sorted([i for i in elem_i.isotopes if i.abundance is not None], key=lambda x: x.abundance, reverse=True)[0]
    i_mass = float(most_abundant_i.mass)
    assert abs(get_atomic_mass("I") - i_mass) < 1e-8

    # Fallback to default for unrecognized elements
    assert get_atomic_mass("UnknownElement") == 12.0


# =============================================================================
# Test Suite 2: Real Asymmetric Tops & Planar Systems (H2O, SO2, H2CO)
# =============================================================================


def test_water_molecule_h2o_planar_asymmetric_top() -> None:
    """Validates inertia tensor, planar moments, inertial defect, and constants

    for real H2O geometry.
    """
    # Equilibrium C2v geometry of H2O in yz plane (Angstroms)
    symbols = ["O", "H", "H"]
    coords = [
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ]

    res = diagonalize_inertia_tensor(coords, symbols=symbols)

    # 1. Mass and COM
    expected_mass = get_atomic_mass("O") + 2.0 * get_atomic_mass("H")
    assert abs(res.total_mass_u - expected_mass) < 1e-6
    # COM must be close to origin on y, and centered
    assert abs(res.center_of_mass_A[0]) < 1e-10
    assert abs(res.center_of_mass_A[1]) < 1e-10

    # 2. Moments of Inertia: Ia <= Ib <= Ic
    ia, ib, ic = res.principal_moments_u_A2
    assert 0.0 < ia < ib < ic

    # 3. Planar defect: Delta = Ic - Ia - Ib == 0.0 for planar geometry
    assert abs(res.inertial_defect_u_A2) < 1e-6
    assert res.is_planar is True
    # For molecule in principal plane (a, b), P_cc = sum m * c^2 = 0
    assert abs(res.planar_moments.P_cc) < 1e-6
    assert res.planar_moments.P_aa > 0.0
    assert res.planar_moments.P_bb > 0.0

    # 4. Rotational constants A >= B >= C
    rc = res.rotational_constants
    assert rc.A_MHz is not None
    assert rc.A_GHz is not None
    assert rc.A_MHz > rc.B_MHz > rc.C_MHz
    # Literature H2O equilibrium: A ~ 835 GHz, B ~ 435 GHz, C ~ 278 GHz
    assert 800.0 < rc.A_GHz < 900.0
    assert 400.0 < rc.B_GHz < 500.0
    assert 250.0 < rc.C_GHz < 350.0

    # 5. Ray's Asymmetry parameter kappa for H2O: kappa ~ -0.46 (Asymmetric Prolate)
    asym = calculate_rays_asymmetry(rc.A_MHz, rc.B_MHz, rc.C_MHz)
    assert -0.60 < asym.kappa < -0.30
    assert asym.rotor_type == "Asymmetric Prolate"
    assert asym.recommended_representation == "Ir"


def test_sulfur_dioxide_so2_planar_asymmetric_top() -> None:
    """Validates planar SO2 molecule moments of inertia and near-prolate asymmetry."""
    symbols = ["S", "O", "O"]
    coords = [
        [0.000000, 0.000000, 0.364200],
        [0.000000, 1.237000, -0.364200],
        [0.000000, -1.237000, -0.364200],
    ]

    res = diagonalize_inertia_tensor(coords, symbols=symbols)

    # In-plane equilibrium -> inertial defect = 0
    assert abs(res.inertial_defect_u_A2) < 1e-6
    assert res.is_planar is True

    ia, ib, ic = res.principal_moments_u_A2
    assert ia < ib < ic

    rc = res.rotational_constants
    assert rc.A_MHz is not None
    # SO2: A ~ 60 GHz, B ~ 10 GHz, C ~ 8 GHz
    assert 50000.0 < rc.A_MHz < 70000.0
    assert 8000.0 < rc.B_MHz < 12000.0
    assert 7000.0 < rc.C_MHz < 10000.0

    asym = calculate_rays_asymmetry(rc.A_MHz, rc.B_MHz, rc.C_MHz)
    # SO2 kappa is near -0.94 (very prolate)
    assert -0.98 < asym.kappa < -0.90
    assert asym.rotor_type == "Asymmetric Prolate"
    assert asym.recommended_representation == "Ir"


def test_formaldehyde_h2co_planar_asymmetric_top() -> None:
    """Validates formaldehyde H2CO (near-prolate symmetric top)."""
    symbols = ["C", "O", "H", "H"]
    coords = [
        [0.000000, 0.000000, -0.597600],
        [0.000000, 0.000000, 0.607400],
        [0.000000, 0.934300, -1.171200],
        [0.000000, -0.934300, -1.171200],
    ]

    extractor = TorqTensorExtractor(symbols, coords, point_id="h2co_01")
    tensors = extractor.extract_tensors()

    assert abs(tensors["inertial_defect_u_A2"]) < 1e-6
    assert tensors["is_planar"] is True

    rc = tensors["rotational_constants"]
    assert rc["A"] > rc["B"] > rc["C"]

    # H2CO kappa ~ -0.96
    asym = tensors["asymmetry"]
    assert -0.99 < asym["kappa"] < -0.93
    assert asym["rotor_type"] == "Asymmetric Prolate"


# =============================================================================
# Test Suite 3: Symmetric Tops (Prolate CH3Cl, Oblate Benzene) & Spherical Top (CH4)
# =============================================================================


def test_methyl_chloride_ch3cl_prolate_symmetric_top() -> None:
    """Validates methyl chloride CH3Cl as prolate symmetric top (kappa=-1, Ib=Ic)."""
    symbols = ["C", "Cl", "H", "H", "H"]
    r_ch = 1.09
    theta = math.radians(109.5)
    r_ccl = 1.78

    coords = [
        [0.000000, 0.000000, 0.000000],  # C
        [0.000000, 0.000000, r_ccl],  # Cl
        [0.000000, r_ch * math.sin(theta), r_ch * math.cos(theta)],  # H1
        [
            r_ch * math.sin(theta) * math.cos(math.radians(210)),
            r_ch * math.sin(theta) * math.sin(math.radians(210)),
            r_ch * math.cos(theta),
        ],  # H2
        [
            r_ch * math.sin(theta) * math.cos(math.radians(330)),
            r_ch * math.sin(theta) * math.sin(math.radians(330)),
            r_ch * math.cos(theta),
        ],  # H3
    ]

    res = diagonalize_inertia_tensor(coords, symbols=symbols)
    ia, ib, ic = res.principal_moments_u_A2

    # Prolate top: Ia < Ib == Ic
    assert ia < ib
    assert abs(ib - ic) < 1e-4

    rc = res.rotational_constants
    assert rc.A_MHz is not None
    assert abs(rc.B_MHz - rc.C_MHz) < 1e-2

    asym = calculate_rays_asymmetry(rc.A_MHz, rc.B_MHz, rc.C_MHz)
    assert abs(asym.kappa - (-1.0)) < 1e-4
    assert asym.rotor_type == "Prolate Symmetric"
    assert asym.recommended_representation == "Ir"


def test_benzene_c6h6_oblate_symmetric_top() -> None:
    """Validates Benzene C6H6 as planar oblate symmetric top (kappa=+1, Ia=Ib)."""
    symbols = ["C"] * 6 + ["H"] * 6
    r_cc = 1.397
    r_ch = 1.084
    r_tot = r_cc + r_ch

    coords = []
    # Carbons
    for i in range(6):
        angle = math.radians(60.0 * i)
        coords.append([r_cc * math.cos(angle), r_cc * math.sin(angle), 0.0])
    # Hydrogens
    for i in range(6):
        angle = math.radians(60.0 * i)
        coords.append([r_tot * math.cos(angle), r_tot * math.sin(angle), 0.0])

    res = diagonalize_inertia_tensor(coords, symbols=symbols)
    ia, ib, ic = res.principal_moments_u_A2

    # Oblate symmetric top: Ia == Ib < Ic
    assert abs(ia - ib) < 1e-4
    # Planar exact condition: Ic = Ia + Ib = 2*Ia
    assert abs(ic - (ia + ib)) < 1e-4
    assert abs(res.inertial_defect_u_A2) < 1e-4
    assert res.is_planar is True

    rc = res.rotational_constants
    assert rc.A_MHz is not None
    assert abs(rc.A_MHz - rc.B_MHz) < 1e-2
    assert rc.B_MHz > rc.C_MHz

    asym = calculate_rays_asymmetry(rc.A_MHz, rc.B_MHz, rc.C_MHz)
    assert abs(asym.kappa - 1.0) < 1e-4
    assert asym.rotor_type == "Oblate Symmetric"
    assert asym.recommended_representation == "IIIr"


def test_methane_ch4_spherical_top() -> None:
    """Validates methane CH4 as isotropic spherical top (Ia = Ib = Ic, A = B = C)."""
    symbols = ["C", "H", "H", "H", "H"]
    d = 1.089 / math.sqrt(3.0)
    coords = [
        [0.0, 0.0, 0.0],
        [d, d, d],
        [d, -d, -d],
        [-d, d, -d],
        [-d, -d, d],
    ]

    res = diagonalize_inertia_tensor(coords, symbols=symbols)
    ia, ib, ic = res.principal_moments_u_A2

    assert abs(ia - ib) < 1e-6
    assert abs(ib - ic) < 1e-6

    rc = res.rotational_constants
    assert rc.A_MHz is not None
    assert abs(rc.A_MHz - rc.B_MHz) < 1e-3
    assert abs(rc.B_MHz - rc.C_MHz) < 1e-3

    asym = calculate_rays_asymmetry(rc.A_MHz, rc.B_MHz, rc.C_MHz)
    assert asym.rotor_type == "Spherical Top"


# =============================================================================
# Test Suite 4: Cartesian Protections & Linearity Trap (CO2, OCS, HCN)
# =============================================================================


def test_cartesian_protections_linear_co2_and_ocs() -> None:
    """Validates Cartesian protections for linear molecules CO2 and OCS."""
    # CO2 along z-axis
    co2_symbols = ["O", "C", "O"]
    co2_coords = [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]

    co2_prot = apply_cartesian_protections(co2_coords, symbols=co2_symbols)
    assert co2_prot.is_linear is True
    assert co2_prot.rotational_dof == 2
    assert co2_prot.singularity_damping_applied is True
    assert co2_prot.protected_rotational_constants.A_MHz is None
    assert co2_prot.protected_rotational_constants.B_MHz > 0.0
    assert (
        co2_prot.protected_rotational_constants.B_MHz
        == co2_prot.protected_rotational_constants.C_MHz
    )

    # Cylindrical coordinates verification: radial rho == 0.0 for all atoms
    for cyl in co2_prot.cylindrical_coordinates:
        assert abs(cyl["rho_radial"]) < 1e-6

    # OCS along arbitrary rotated line
    ocs_symbols = ["O", "C", "S"]
    v = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    ocs_coords = [
        (0.0 * v).tolist(),
        (1.16 * v).tolist(),
        ((1.16 + 1.56) * v).tolist(),
    ]

    ocs_prot = apply_cartesian_protections(ocs_coords, symbols=ocs_symbols)
    assert ocs_prot.is_linear is True
    assert ocs_prot.rotational_dof == 2
    for cyl in ocs_prot.cylindrical_coordinates:
        assert abs(cyl["rho_radial"]) < 1e-4


def test_cartesian_protections_linear_hcn() -> None:
    """Validates linear HCN."""
    symbols = ["H", "C", "N"]
    coords = [[0.0, 0.0, -1.066], [0.0, 0.0, 0.0], [0.0, 0.0, 1.153]]

    extractor = TorqTensorExtractor(symbols, coords, point_id="hcn_linear")
    output = extractor.get_full_output()

    assert output.cartesian_protection.is_linear is True
    assert output.cartesian_protection.rotational_dof == 2
    assert output.inertia.rotational_constants.A_MHz is None
    assert output.inertia.rotational_constants.B_MHz > 0.0


def test_cartesian_protections_quasi_linear_complex() -> None:
    """Validates quasi-linear floppy complex protection with 179.5 degree angle."""
    symbols = ["Ne", "C", "O"]
    # Slight bend of 0.5 degrees
    angle_rad = math.radians(179.5)
    r1 = 3.2
    r2 = 1.13
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, r1],
        [
            r2 * math.sin(math.pi - angle_rad),
            0.0,
            r1 + r2 * math.cos(math.pi - angle_rad),
        ],
    ]

    prot = apply_cartesian_protections(
        coords, symbols=symbols, threshold_linear=1e-2, angle_tolerance_deg=1.0
    )
    assert prot.is_quasi_linear is True
    assert prot.rotational_dof == 2
    assert prot.singularity_damping_applied is True


# =============================================================================
# Test Suite 5: Dynamic Representation Switch (6 Representations)
# =============================================================================


def test_dynamic_representation_switch_all_six_representations() -> None:
    """Validates all 6 standard King-Hainer-Cross representations."""
    from typing import Literal

    representations: list[Literal["Ir", "Il", "IIr", "IIl", "IIIr", "IIIl"]] = [
        "Ir",
        "Il",
        "IIr",
        "IIl",
        "IIIr",
        "IIIl",
    ]

    for rep in representations:
        res = dynamic_representation_switch(kappa=0.5, preferred_type=rep)
        assert res["representation"] == rep
        assert "axis_mapping" in res
        assert "transformation_matrix" in res
        t_mat = np.array(res["transformation_matrix"])
        det = np.linalg.det(t_mat)
        if res["is_right_handed"]:
            assert abs(det - 1.0) < 1e-6
        else:
            assert abs(det - (-1.0)) < 1e-6
        assert res["wang_subblocks"] == ["E+", "E-", "O+", "O-"]

    # Auto selection based on kappa
    prolate_rep = dynamic_representation_switch(kappa=-0.8, preferred_type="auto")
    assert prolate_rep["representation"] == "Ir"

    oblate_rep = dynamic_representation_switch(kappa=+0.8, preferred_type="auto")
    assert oblate_rep["representation"] == "IIIr"


# =============================================================================
# Test Suite 6: ORCA VPT2, Coriolis, Centrifugal Distortion Parser
# =============================================================================


def test_orca_vpt2_and_coriolis_parser(tmp_path: Path) -> None:
    """Validates ORCA %vib block parsing for resonances and distortion constants."""
    orca_output_text = """
================================================================================
                               ORCA VPT2 MODULE
================================================================================
Darling-Dennison Mode 1 Mode 2 K = -14.2857
Darling-Dennison Mode 3 Mode 4 K = 2.4510

----------------------------------------
Coriolis Coupling Matrix (X)
----------------------------------------
  0.000000  0.845120 -0.124500
 -0.845120  0.000000  0.512340
  0.124500 -0.512340  0.000000

----------------------------------------
Coriolis Coupling Matrix (Y)
----------------------------------------
  0.000000  0.221100  0.781200
 -0.221100  0.000000 -0.114400
 -0.781200  0.114400  0.000000

----------------------------------------
Coriolis Coupling Matrix (Z)
----------------------------------------
  0.000000  0.000000  0.000000
  0.000000  0.000000  0.998120
  0.000000 -0.998120  0.000000

Centrifugal Distortion Constants (A-Reduction):
  D_J  = 0.034512
  D_JK = -0.124500
  D_K  = 1.542100
  d_1  = -0.004120
  d_2  = 0.000850

Polarizability derivative: 1.254100
Polarizability derivative: 0.895400
Polarizability derivative: 2.145000
"""
    orca_file = tmp_path / "orca_test.out"
    orca_file.write_text(orca_output_text, encoding="utf-8")

    extractor = TorqTensorExtractor(
        symbols=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.1], [0.0, 0.7, -0.4], [0.0, -0.7, -0.4]],
        point_id="vpt2_h2o",
        orca_file=orca_file,
    )

    vpt2_res = extractor.extract_vpt2_data(orca_file)

    # 1. Darling-Dennison
    assert len(vpt2_res["darling_dennison"]) == 2
    assert vpt2_res["darling_dennison"][0]["mode1"] == 1
    assert vpt2_res["darling_dennison"][0]["mode2"] == 2
    assert abs(vpt2_res["darling_dennison"][0]["resonance"] - (-14.2857)) < 1e-4

    # 2. Coriolis Couplings
    assert len(vpt2_res["coriolis_couplings"]["x"]) == 9
    assert abs(vpt2_res["coriolis_couplings"]["x"][1] - 0.845120) < 1e-5
    assert len(vpt2_res["coriolis_couplings"]["z"]) == 9

    # 3. Distortion Constants
    cd = vpt2_res["centrifugal_distortion"]
    assert abs(cd["D_J"][0] - 0.034512) < 1e-6
    assert abs(cd["D_JK"][0] - (-0.124500)) < 1e-6
    assert abs(cd["D_K"][0] - 1.542100) < 1e-6
    assert abs(cd["d_1"][0] - (-0.004120)) < 1e-6
    assert abs(cd["d_2"][0] - 0.000850) < 1e-6

    # 4. Polarizabilities
    assert len(vpt2_res["raman_polarizability"]) == 3
    assert abs(vpt2_res["raman_polarizability"][0] - 1.254100) < 1e-6

    # 5. Divergence check
    assert vpt2_res["is_divergent"] is False


def test_orca_vpt2_divergence_detection(tmp_path: Path) -> None:
    """Validates unphysical divergence detection for distortion constants."""
    orca_divergent_text = """
Centrifugal Distortion Constants:
  D_J  = 1.5e7
  D_JK = 2.4e8
  D_K  = -9.9e9
"""
    orca_file = tmp_path / "orca_div.out"
    orca_file.write_text(orca_divergent_text, encoding="utf-8")

    extractor = TorqTensorExtractor(
        symbols=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.1], [0.0, 0.7, -0.4], [0.0, -0.7, -0.4]],
    )
    vpt2_res = extractor.extract_vpt2_data(orca_file)

    assert vpt2_res["is_divergent"] is True
    assert len(vpt2_res["divergence_details"]) > 0


# =============================================================================
# Test Suite 7: Thermal NMR & Raman Polarizability Extractors
# =============================================================================


def test_thermal_nmr_extraction(tmp_path: Path) -> None:
    """Validates thermal NMR shielding extraction from AIMD trajectory file."""
    traj_text = """3
Frame 1
O  0.0  0.0  0.11
H  0.0  0.75 -0.46
H  0.0 -0.75 -0.46
3
Frame 2
O  0.0  0.0  0.12
H  0.0  0.76 -0.47
H  0.0 -0.76 -0.47
"""
    traj_file = tmp_path / "aimd_traj.xyz"
    traj_file.write_text(traj_text, encoding="utf-8")

    extractor = TorqTensorExtractor(
        symbols=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.11], [0.0, 0.75, -0.46], [0.0, -0.75, -0.46]],
    )

    nmr_res = extractor.extract_thermal_nmr(traj_file)
    assert nmr_res["frame_count"] == 2
    assert nmr_res["thermal_average"] > 0.0
    assert len(nmr_res["isotropic_shielding"]) == 2


# =============================================================================
# Test Suite 8: JSON and HDF5 Export Gateways & Air-Gap Compliance
# =============================================================================


def test_export_tensor_json_and_hdf5(tmp_path: Path) -> None:
    """Validates JSON and HDF5 serialization with Air-Gap directory compliance."""
    export_dir = tmp_path / "artifacts" / "Tensors"
    export_dir.mkdir(parents=True, exist_ok=True)

    json_file = export_dir / "torq_tensors.json"
    vpt2_file = export_dir / "torq_vpt2.json"
    lam_file = export_dir / "torq_lam_vpt2.json"
    h5_file = export_dir / "torq_tensors.h5"

    symbols = ["O", "H", "H"]
    coords = [
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ]

    extractor = TorqTensorExtractor(symbols, coords, point_id="h2o_pt01")

    # 1. Export JSON
    extractor.export_tensor(json_file)
    assert json_file.exists()
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
        assert data["point_id"] == "h2o_pt01"
        assert "rotational_constants" in data
        assert "inertia_tensor" in data

    # 2. Export VPT2 JSON
    extractor.export_vpt2_tensor(vpt2_file)
    assert vpt2_file.exists()

    # 3. Export LAM VPT2 JSON
    extractor.export_lam_vpt2_tensor(lam_file)
    assert lam_file.exists()

    # 4. Export HDF5
    payload_dict = {
        "rotational_constants_mhz": [
            data["rotational_constants"]["A"],
            data["rotational_constants"]["B"],
            data["rotational_constants"]["C"],
        ],
        "inertia_tensor": data["inertia_tensor"],
        "is_planar": True,
        "rotor_type": "Asymmetric Prolate",
    }
    extractor.export_to_hdf5(h5_file, payload_dict)
    assert h5_file.exists()

    # Verify HDF5 contents
    with h5py.File(h5_file, "r") as f:
        assert "point_h2o_pt01" in f
        grp = f["point_h2o_pt01"]
        assert "rotational_constants_mhz" in grp
        assert "inertia_tensor" in grp
        assert bool(grp.attrs["is_planar"]) is True
        assert str(grp.attrs["rotor_type"]) == "Asymmetric Prolate"

    # 5. Export Sinc-DVR HDF5
    dvr_h5 = export_dir / "sinc_dvr.h5"
    dvr_payload = {
        "wavefunction": np.ones((50, 50)).tolist(),
        "energy_levels": [0.0, 125.4, 250.8, 375.2],
        "tunneling_splitting": 1.458e-4,
        "kraitchman_coords": [[0.0, 0.0, 0.5]],
    }
    extractor.export_to_hdf5_with_sinc_dvr(dvr_h5, dvr_payload)
    assert dvr_h5.exists()
    with h5py.File(dvr_h5, "r") as f:
        grp = f["point_h2o_pt01"]
        assert "wavefunction" in grp
        assert "energy_levels" in grp
        assert abs(grp.attrs["tunneling_splitting"] - 1.458e-4) < 1e-8


# =============================================================================
# Test Suite 9: Pydantic Data Models & Anti-Spoofing Protocols
# =============================================================================


def test_pydantic_payload_models_integrity() -> None:
    """Validates Pydantic schema validation and immutable contract."""
    symbols = ["C", "O", "O"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.16], [0.0, 0.0, 1.16]]
    extractor = TorqTensorExtractor(symbols, coords, point_id="co2_pydantic")
    full_output = extractor.get_full_output()

    assert isinstance(full_output, TorqTensorOutput)
    assert isinstance(full_output.inertia, InertiaTensorResult)
    assert isinstance(full_output.cartesian_protection, CartesianProtectionResult)
    assert isinstance(full_output.asymmetry, AsymmetryResult)

    dumped = full_output.model_dump()
    assert dumped["point_id"] == "co2_pydantic"
    assert dumped["cartesian_protection"]["is_linear"] is True


def test_anti_spoofing_spin_hamiltonian_guard() -> None:
    """Validates that unverified Spin Hamiltonian calls raise strict RuntimeError."""
    extractor = TorqTensorExtractor(
        symbols=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.1], [0.0, 0.7, -0.4], [0.0, -0.7, -0.4]],
    )
    with pytest.raises(RuntimeError, match="Anti-spoofing mandate"):
        extractor.extract_spin_hamiltonian()
