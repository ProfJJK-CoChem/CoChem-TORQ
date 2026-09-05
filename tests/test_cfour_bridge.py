"""
Unit Tests for CoChem-TORQ CFOUR Coupled-Cluster Bridge
======================================================
Tests Phase 5 (Stage 4.5) CFOUR Bridge:
- Z-matrix generation with dummy atom singularity protection (Method Matrix §9.5)
- Coupled cluster analytic second derivatives and VPT2 parsing
- Watson A- and S-reduction quartic and sextic centrifugal distortion extraction
- Electric Field Gradient to nuclear quadrupole coupling conversion (chi = EFG * Q * 234.96474)
- Nuclear spin-rotation tensor parsing
- SPCAT payload conversion and HDF5 SWMR archiving
- Zero-mock / anti-spoofing compliance
"""

import math
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
import h5py

from Libraries.cochem_torq_cfour_bridge import (
    CFOURPhysicalConstants,
    CONSTANTS,
    get_atomic_mass,
    get_isotopic_mass,
    get_atomic_number,
    compute_center_of_mass,
    compute_inertia_tensor,
    CFOURRotationalConstants,
    CFOURQuarticDistortion,
    CFOURSexticDistortion,
    CFOURVibrationalData,
    CFOURDipoleMoment,
    CFOURNuclearQuadrupole,
    CFOURSpinRotation,
    CFOUREnergies,
    CFOUROutputPayload,
    CFOURZmatBuilder,
    CFOUROutputParser,
    TorqCfourExecutor,
    export_cfour_to_spcat_dict,
    save_cfour_to_hdf5,
    CFOURDecompositionManager,
    cleanup_zombies,
)


# ============================================================================
# 1. Fundamental Constants and Dynamic Mendeleev Mass Retrieval
# ============================================================================

def test_cfour_constants() -> None:
    assert math.isclose(CONSTANTS.BOHR_TO_ANGSTROM, 0.529177210903, rel_tol=1e-8)
    assert math.isclose(CONSTANTS.CM1_TO_MHZ, 29979.2458, rel_tol=1e-8)
    assert math.isclose(CONSTANTS.EFG_TO_KHZ_FACTOR, 234.96474, rel_tol=1e-8)
    assert math.isclose(CONSTANTS.AU_TO_DEBYE, 2.5417464519, rel_tol=1e-8)


def test_mendeleev_mass_retrieval() -> None:
    # Carbon mass (amu)
    c_mass = get_atomic_mass("C")
    assert 12.0 <= c_mass <= 12.02

    # Hydrogen mass
    h_mass = get_atomic_mass("H")
    assert 1.0 <= h_mass <= 1.01

    # Oxygen mass
    o_mass = get_atomic_mass("O")
    assert 15.99 <= o_mass <= 16.01

    # Isotopic masses
    c13_mass = get_isotopic_mass("C", 13)
    assert 13.0 <= c13_mass <= 13.01

    d_mass = get_isotopic_mass("H", 2)
    assert 2.01 <= d_mass <= 2.02

    # Atomic numbers
    assert get_atomic_number("C") == 6
    assert get_atomic_number("N") == 7
    assert get_atomic_number("O") == 8


# ============================================================================
# 2. Geometry and Inertia Tensor Calculations
# ============================================================================

def test_inertia_tensor_water() -> None:
    symbols = ["O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692]
    ]
    com = compute_center_of_mass(symbols, np.array(coords))
    assert abs(com[0]) < 1e-6
    assert abs(com[1]) < 1e-6

    moments, axes, rot_mhz = compute_inertia_tensor(symbols, np.array(coords))
    # Moments of inertia should be sorted I_A <= I_B <= I_C
    assert moments[0] <= moments[1] <= moments[2]
    # Rotational constants A >= B >= C (MHz)
    assert rot_mhz[0] >= rot_mhz[1] >= rot_mhz[2]
    assert rot_mhz[0] > 500000.0  # Water A ~ 830-850 GHz
    assert rot_mhz[1] > 200000.0  # Water B ~ 430-440 GHz
    assert rot_mhz[2] > 150000.0  # Water C ~ 270-290 GHz


# ============================================================================
# 3. Z-Matrix Generation with Dummy-Atom Singularity Protection
# ============================================================================

def test_zmat_generation_water() -> None:
    symbols = ["O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.7572, 0.586],
        [0.0, -0.7572, 0.586]
    ]
    zmat_text, vars_dict = CFOURZmatBuilder.cartesian_to_zmat(symbols, coords, optimize_all=True)
    assert "O" in zmat_text
    assert "H 1 R1" in zmat_text
    assert "H 2 R2 1 A1" in zmat_text
    assert "R1* =" in zmat_text
    assert "R2* =" in zmat_text
    assert "A1* =" in zmat_text
    assert len(vars_dict) == 3


def test_zmat_generation_linear_dummy_atom_protection() -> None:
    """
    Tests Method Matrix §9.5 requirement:
    Angles near 0 or 180 degrees must insert a dummy atom 'X' to avoid CFOUR singularity crashes.
    """
    # Linear Ar-H-C-N chain: Ar at (0,0,0), H at (0,0,3.0), C at (0,0,4.065), N at (0,0,5.221)
    symbols = ["Ar", "H", "C", "N"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 4.065],
        [0.0, 0.0, 5.221]
    ]
    zmat_text, vars_dict = CFOURZmatBuilder.cartesian_to_zmat(symbols, coords, linear_angle_threshold_deg=5.0)
    # The linear chain should trigger dummy atom insertion 'X'
    assert "X 2 R" in zmat_text
    assert "90.0" in zmat_text  # Dummy atom placed at 90 degrees


def test_generate_full_zmat_input_deck() -> None:
    symbols = ["O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.7572, 0.586],
        [0.0, -0.7572, 0.586]
    ]
    deck = CFOURZmatBuilder.generate_full_zmat_input(
        symbols=symbols,
        coordinates=coords,
        method="CCSD(T)",
        basis="ANO1",
        reference="RHF",
        memory_gb=32,
        anharm="VPT2",
        props="FIRST_ORDER",
        isotopes=[16, 1, 1],
    )
    # Check Method Matrix invariants
    assert "*CFOUR(" in deck
    assert "CALC=CCSD(T)" in deck
    assert "BASIS=ANO1" in deck
    assert "ABCDTYPE=AOBASIS" in deck
    assert "CC_PROG=ECC" in deck
    assert "MEMORY_SIZE=32" in deck
    assert "MEM_UNIT=GB" in deck
    assert "VIB=EXACT" in deck
    assert "ANHARM=VPT2" in deck
    assert "PROPS=FIRST_ORDER" in deck
    assert "%isotopes" in deck
    assert "16" in deck


# ============================================================================
# 4. CFOUR Output Parser: Energies, Rotational Constants & VPT2
# ============================================================================

CFOUR_SAMPLE_OUTPUT = """
------------------------------------------------------------------------
                        CFOUR Version 2.1
------------------------------------------------------------------------

Title: Water CCSD(T)/ANO1 VPT2 Anharmonic Force Field

Reference energy is -76.0625482341 a.u.
Total MP2 energy : -76.3214567890
Total CCSD energy : -76.3345123456
Total CCSD(T) energy : -76.3421987654 a.u.

Total DBOC = 0.00045123 a.u.
Total Relativistic Energy : -0.05432100

Rotational constants (in MHz):
A =   835412.34    B =   435678.90    C =   278912.45

Harmonic vibrational frequencies (cm-1):
  1595.23   3657.45   3756.89

Anharmonic vibrational frequencies (cm-1):
  1552.10   3582.40   3678.90

Zero point vibrational energy : 13.250 kcal/mol
Anharmonic zero point energy : 13.010 kcal/mol

Be, B0 AND B-B0 SHIFTS FOR SINGLY EXCITED VIBRATIONAL STATES (CM-1)
Ground State    27.8654   14.5321    9.3032
State 1          0.1200    0.0450    0.0230

Delta_J = 0.1234 kHz
Delta_JK = -0.5678 kHz
Delta_K = 2.3456 kHz
delta_J = 0.0234 kHz
delta_K = 0.1456 kHz

Phi_J = 0.0123 Hz
Phi_JK = -0.0456 Hz
Phi_KJ = 0.1234 Hz
Phi_K = 0.4567 Hz
phi_j = 0.0023 Hz
phi_jk = -0.0123 Hz
phi_k = 0.0456 Hz

Dipole moment (Debye): mu_a = 0.0000  mu_b = 0.0000  mu_c = 1.8540  Total = 1.8540

Electric field gradient tensor for atom 1 (N) (in a.u.):
 0.12345   0.00000   0.00000
 0.00000  -0.05432   0.00000
 0.00000   0.00000  -0.06913

Spin-rotation tensor for atom 1 (N) (in kHz):
  12.3400    0.0000    0.0000
   0.0000    15.4200    0.0000
   0.0000    0.0000    18.6500
"""

def test_cfour_output_parser_full() -> None:
    payload = CFOUROutputParser.parse_full_output(CFOUR_SAMPLE_OUTPUT, molecule_name="Water_VPT2")

    # Energies
    assert math.isclose(payload.energies.scf_energy_hartree, -76.0625482341, rel_tol=1e-8)
    assert math.isclose(payload.energies.mp2_energy_hartree, -76.3214567890, rel_tol=1e-8)
    assert math.isclose(payload.energies.ccsd_energy_hartree, -76.3345123456, rel_tol=1e-8)
    assert math.isclose(payload.energies.ccsd_t_energy_hartree, -76.3421987654, rel_tol=1e-8)
    assert math.isclose(payload.energies.final_energy_hartree, -76.3421987654, rel_tol=1e-8)
    assert math.isclose(payload.energies.dboc_correction_hartree, 0.00045123, rel_tol=1e-8)
    assert math.isclose(payload.energies.relativistic_correction_hartree, -0.05432100, rel_tol=1e-8)

    # Rotational Constants
    assert math.isclose(payload.rotational_constants.Ae_MHz, 835412.34, rel_tol=1e-6)
    assert math.isclose(payload.rotational_constants.Be_MHz, 435678.90, rel_tol=1e-6)
    assert math.isclose(payload.rotational_constants.Ce_MHz, 278912.45, rel_tol=1e-6)
    assert payload.rotational_constants.A0_MHz > 0.0
    assert payload.rotational_constants.B0_MHz > 0.0
    assert payload.rotational_constants.C0_MHz > 0.0

    # Vibrational Frequencies
    assert len(payload.vibrational_data.harmonic_frequencies_cm1) == 3
    assert math.isclose(payload.vibrational_data.harmonic_frequencies_cm1[0], 1595.23, rel_tol=1e-6)
    assert len(payload.vibrational_data.anharmonic_frequencies_cm1) == 3
    assert math.isclose(payload.vibrational_data.anharmonic_frequencies_cm1[0], 1552.10, rel_tol=1e-6)
    assert math.isclose(payload.vibrational_data.harmonic_zpe_kcal_mol, 13.250, rel_tol=1e-6)
    assert math.isclose(payload.vibrational_data.anharmonic_zpe_kcal_mol, 13.010, rel_tol=1e-6)

    # Quartic Centrifugal Distortion (Watson A)
    assert math.isclose(payload.quartic_distortion.Delta_J_kHz, 0.1234, rel_tol=1e-6)
    assert math.isclose(payload.quartic_distortion.Delta_JK_kHz, -0.5678, rel_tol=1e-6)
    assert math.isclose(payload.quartic_distortion.Delta_K_kHz, 2.3456, rel_tol=1e-6)
    assert math.isclose(payload.quartic_distortion.delta_J_kHz, 0.0234, rel_tol=1e-6)
    assert math.isclose(payload.quartic_distortion.delta_K_kHz, 0.1456, rel_tol=1e-6)

    # Sextic Centrifugal Distortion (Watson A) - CFOUR Unique Capability
    assert math.isclose(payload.sextic_distortion.Phi_J_Hz, 0.0123, rel_tol=1e-6)
    assert math.isclose(payload.sextic_distortion.Phi_JK_Hz, -0.0456, rel_tol=1e-6)
    assert math.isclose(payload.sextic_distortion.Phi_KJ_Hz, 0.1234, rel_tol=1e-6)
    assert math.isclose(payload.sextic_distortion.Phi_K_Hz, 0.4567, rel_tol=1e-6)
    assert math.isclose(payload.sextic_distortion.phi_J_Hz, 0.0023, rel_tol=1e-6)
    assert math.isclose(payload.sextic_distortion.phi_JK_Hz, -0.0123, rel_tol=1e-6)
    assert math.isclose(payload.sextic_distortion.phi_K_Hz, 0.0456, rel_tol=1e-6)

    # Dipole Moment
    assert math.isclose(payload.dipole_moment.mu_c_debye, 1.8540, rel_tol=1e-6)
    assert math.isclose(payload.dipole_moment.mu_total_debye, 1.8540, rel_tol=1e-6)

    # Nuclear Quadrupole Coupling (converted from EFG with Q(14N) = 20.44 mbarn)
    assert len(payload.quadrupole_coupling) == 1
    nq = payload.quadrupole_coupling[0]
    assert nq.element == "N"
    # chi_zz [kHz] = 0.12345 * 20.44 * 234.96474 ~ 592.88 kHz
    expected_chi = 0.12345 * 20.44 * 234.96474
    assert math.isclose(nq.chi_cc_kHz, expected_chi, rel_tol=1e-3)

    # Spin-Rotation Tensor
    assert len(payload.spin_rotation) == 1
    sr = payload.spin_rotation[0]
    assert sr.element == "N"
    assert math.isclose(sr.C_aa_kHz, 12.34, rel_tol=1e-6)
    assert math.isclose(sr.C_bb_kHz, 15.42, rel_tol=1e-6)
    assert math.isclose(sr.C_cc_kHz, 18.65, rel_tol=1e-6)
    assert math.isclose(sr.C_iso_kHz, (12.34 + 15.42 + 18.65) / 3.0, rel_tol=1e-6)

    # Provenance Hash
    assert len(payload.raw_output_sha256) == 64


# ============================================================================
# 5. Bridge Interfacing & HDF5 Persistence Tests
# ============================================================================

def test_export_to_spcat_dict() -> None:
    payload = CFOUROutputParser.parse_full_output(CFOUR_SAMPLE_OUTPUT, molecule_name="Water_VPT2")
    spcat_data = export_cfour_to_spcat_dict(payload)

    assert spcat_data["eccsd_t"] == -76.3421987654
    assert len(spcat_data["frequencies"]) == 3
    assert len(spcat_data["anharmonic_frequencies"]) == 3
    assert spcat_data["dipoles"]["mu_c"] == 1.8540
    assert "sextic_distortion_hz" in spcat_data
    assert spcat_data["sextic_distortion_hz"]["Phi_J"] == 0.0123


def test_save_to_hdf5(tmp_path: Path) -> None:
    payload = CFOUROutputParser.parse_full_output(CFOUR_SAMPLE_OUTPUT, molecule_name="Water_VPT2")
    h5_path = tmp_path / "test_cfour_landscape.h5"

    save_cfour_to_hdf5(payload, h5_path, dataset_group="ab_initio/cfour/water")

    assert h5_path.exists()
    with h5py.File(str(h5_path), "r") as f:
        grp = f["ab_initio/cfour/water"]
        assert math.isclose(grp.attrs["final_energy_hartree"], -76.3421987654, rel_tol=1e-8)
        assert math.isclose(grp.attrs["Ae_MHz"], 835412.34, rel_tol=1e-6)
        assert math.isclose(grp.attrs["Phi_J_Hz"], 0.0123, rel_tol=1e-6)
        assert "harmonic_frequencies_cm1" in grp
        assert len(grp["harmonic_frequencies_cm1"]) == 3


# ============================================================================
# 6. Queue Decomposition Manager Tests
# ============================================================================

def test_decomposition_manager() -> None:
    base_deck = "*CFOUR(\nCALC=CCSD(T)\nBASIS=ANO1\n)"
    subjob_zmat = CFOURDecompositionManager.generate_irrep_subjob_input(base_deck, irrep_index=2)
    assert "FD_IRREP=2" in subjob_zmat
    assert "FD_PROJECT=OFF" in subjob_zmat
    assert "FREQ_ALGORITHM=PARALLEL" in subjob_zmat
    assert "CALC=CCSD(T)" in subjob_zmat

    cmds = CFOURDecompositionManager.get_assembly_command_chain()
    assert cmds == ["xjoda", "xsymcor", "xja2fja", "xcubic"]


# ============================================================================
# 7. Additional Exhaustive Tests: S-Reduction, Alphas & Anti-Spoofing
# ============================================================================

CFOUR_S_REDUCTION_SAMPLE = """
Rotational constants (in cm-1): A = 10.500000 B = 5.250000 C = 2.625000

D_J = 0.5432 kHz
D_JK = -1.2345 kHz
D_K = 4.5678 kHz
d_1 = 0.0432 kHz
d_2 = 0.0012 kHz

H_J = 0.0054 Hz
H_JK = -0.0123 Hz
H_KJ = 0.0456 Hz
H_K = 0.1234 Hz
h_1 = 0.0012 Hz
h_2 = -0.0003 Hz
h_3 = 0.0001 Hz
"""

def test_cfour_output_parser_s_reduction() -> None:
    quartic, sextic = CFOUROutputParser.parse_centrifugal_distortion(CFOUR_S_REDUCTION_SAMPLE)
    assert quartic.reduction_type == "S"
    assert math.isclose(quartic.D_J_kHz, 0.5432, rel_tol=1e-6)
    assert math.isclose(quartic.D_JK_kHz, -1.2345, rel_tol=1e-6)
    assert math.isclose(quartic.D_K_kHz, 4.5678, rel_tol=1e-6)
    assert math.isclose(quartic.d_1_kHz, 0.0432, rel_tol=1e-6)
    assert math.isclose(quartic.d_2_kHz, 0.0012, rel_tol=1e-6)

    assert sextic.reduction_type == "S"
    assert math.isclose(sextic.H_J_Hz, 0.0054, rel_tol=1e-6)
    assert math.isclose(sextic.H_JK_Hz, -0.0123, rel_tol=1e-6)
    assert math.isclose(sextic.H_KJ_Hz, 0.0456, rel_tol=1e-6)
    assert math.isclose(sextic.H_K_Hz, 0.1234, rel_tol=1e-6)
    assert math.isclose(sextic.h_1_Hz, 0.0012, rel_tol=1e-6)
    assert math.isclose(sextic.h_2_Hz, -0.0003, rel_tol=1e-6)
    assert math.isclose(sextic.h_3_Hz, 0.0001, rel_tol=1e-6)


CFOUR_ALPHA_SAMPLE = """
Rotational constants (in MHz): A = 50000.00 B = 25000.00 C = 10000.00

Vibration-rotation interaction constants (in MHz):
 1   150.00   75.00   25.00
 2   100.00   50.00   15.00
 3    50.00   25.00   10.00
"""

def test_cfour_output_parser_alphas() -> None:
    rc = CFOUROutputParser.parse_rotational_constants(CFOUR_ALPHA_SAMPLE)
    assert len(rc.alpha_A_MHz) == 3
    assert math.isclose(rc.alpha_A_MHz[0], 150.0, rel_tol=1e-6)
    assert math.isclose(rc.alpha_B_MHz[1], 50.0, rel_tol=1e-6)
    assert math.isclose(rc.alpha_C_MHz[2], 10.0, rel_tol=1e-6)

    # B0 = Be - 0.5 * sum(alpha)
    # A0 = 50000 - 0.5 * (150 + 100 + 50) = 50000 - 150 = 49850 MHz
    assert math.isclose(rc.A0_MHz, 49850.0, rel_tol=1e-6)
    # B0 = 25000 - 0.5 * (75 + 50 + 25) = 25000 - 75 = 24925 MHz
    assert math.isclose(rc.B0_MHz, 24925.0, rel_tol=1e-6)
    # C0 = 10000 - 0.5 * (25 + 15 + 10) = 10000 - 25 = 9975 MHz
    assert math.isclose(rc.C0_MHz, 9975.0, rel_tol=1e-6)


def test_anti_spoofing_no_stubs() -> None:
    """Anti-Spoofing Protocol v2 verification: ensure zero stubs/mocking."""
    import inspect
    import Libraries.cochem_torq_cfour_bridge as bridge_mod

    source = inspect.getsource(bridge_mod)
    assert "NotImplementedError" not in source
    assert "unittest.mock" not in source
    assert "MagicMock" not in source

