"""CFOUR mathematics, input contracts and explicit missing-result tests.

No positive CFOUR output, force field, VPT2 observable or tensor is generated in
this module. Native positive qualification requires genuine archived or live
CFOUR evidence; absent values remain unavailable and cannot become a catalog.
"""

import math
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from Libraries.cochem_torq_cfour_bridge import (
    CONSTANTS,
    CFOURDecompositionManager,
    CFOURDipoleMoment,
    CFOUREnergies,
    CFOURNuclearQuadrupole,
    CFOUROutputParser,
    CFOUROutputPayload,
    CFOURPhysicalConstants,
    CFOURQuarticDistortion,
    CFOURRotationalConstants,
    CFOURSexticDistortion,
    CFOURSpinRotation,
    CFOURVibrationalData,
    CFOURZmatBuilder,
    TorqCfourExecutor,
    cleanup_zombies,
    compute_center_of_mass,
    compute_inertia_tensor,
    export_cfour_to_spcat_dict,
    get_atomic_mass,
    get_atomic_number,
    get_isotopic_mass,
    save_cfour_to_hdf5,
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
    coords = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
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
    coords = [[0.0, 0.0, 0.0], [0.0, 0.7572, 0.586], [0.0, -0.7572, 0.586]]
    zmat_text, vars_dict = CFOURZmatBuilder.cartesian_to_zmat(
        symbols, coords, optimize_all=True
    )
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
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.065], [0.0, 0.0, 5.221]]
    zmat_text, vars_dict = CFOURZmatBuilder.cartesian_to_zmat(
        symbols, coords, linear_angle_threshold_deg=5.0
    )
    # The linear chain should trigger dummy atom insertion 'X'
    assert "X 2 R" in zmat_text
    assert "90.0" in zmat_text  # Dummy atom placed at 90 degrees


def test_generate_full_zmat_input_deck() -> None:
    symbols = ["O", "H", "H"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.7572, 0.586], [0.0, -0.7572, 0.586]]
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


def test_cfour_output_parser_full() -> None:
    """Missing CFOUR evidence stays unavailable; no vendor output is invented."""
    payload = CFOUROutputParser.parse_full_output("")
    assert payload.calculation_converged is None
    assert payload.calc_method is None and payload.basis_set is None
    assert payload.energies.final_energy_hartree is None
    assert payload.vibrational_data.harmonic_frequencies_cm1 is None
    assert payload.vibrational_data.anharmonic_frequencies_cm1 is None
    assert payload.validation_issues


# ============================================================================
# 5. Bridge Interfacing & HDF5 Persistence Tests
# ============================================================================


def test_export_to_spcat_dict() -> None:
    """Unknown convergence cannot be promoted to a physical catalog."""
    payload = CFOUROutputParser.parse_full_output("")
    with pytest.raises(ValueError, match="positively verified"):
        export_cfour_to_spcat_dict(payload)


def test_save_to_hdf5(tmp_path: Path) -> None:
    """Actual HDF5 persistence records missingness without invented observables."""
    import json

    payload = CFOUROutputParser.parse_full_output("")
    h5_path = tmp_path / "unavailable-cfour.h5"
    save_cfour_to_hdf5(payload, h5_path, dataset_group="ab_initio/cfour/unavailable")
    with h5py.File(h5_path, "r") as handle:
        group = handle["ab_initio/cfour/unavailable"]
        assert "final_energy_hartree" not in group.attrs
        assert "Ae_MHz" not in group.attrs
        assert "harmonic_frequencies_cm1" not in group
        saved = json.loads(group["payload_json"].asstr()[()])
        assert saved["energies"]["final_energy_hartree"] is None


# ============================================================================
# 6. Queue Decomposition Manager Tests
# ============================================================================


def test_decomposition_manager() -> None:
    base_deck = "*CFOUR(\nCALC=CCSD(T)\nBASIS=ANO1\n)"
    subjob_zmat = CFOURDecompositionManager.generate_irrep_subjob_input(
        base_deck, irrep_index=2
    )
    assert "FD_IRREP=2" in subjob_zmat
    assert "FD_PROJECT=OFF" in subjob_zmat
    assert "FREQ_ALGORITHM=PARALLEL" in subjob_zmat
    assert "CALC=CCSD(T)" in subjob_zmat

    cmds = CFOURDecompositionManager.get_assembly_command_chain()
    assert cmds == ["xjoda", "xsymcor", "xja2fja", "xcubic"]


# ============================================================================
# 7. Additional Exhaustive Tests: S-Reduction, Alphas & Anti-Spoofing
# ============================================================================


def test_cfour_output_parser_s_reduction() -> None:
    """No Watson reduction or distortion constants are assumed without evidence."""
    quartic, sextic = CFOUROutputParser.parse_centrifugal_distortion("")
    assert all(value is None for value in quartic.model_dump().values())
    assert all(value is None for value in sextic.model_dump().values())


def test_cfour_output_parser_alphas() -> None:
    """An absent anharmonic calculation cannot generate alpha constants or B0."""
    result = CFOUROutputParser.parse_rotational_constants("")
    assert result.alpha_A_MHz is None
    assert result.alpha_B_MHz is None
    assert result.alpha_C_MHz is None
    assert result.A0_MHz is result.B0_MHz is result.C0_MHz is None


def test_anti_spoofing_no_stubs() -> None:
    """Anti-Spoofing Protocol v2 verification: ensure zero stubs/mocking."""
    import inspect

    import Libraries.cochem_torq_cfour_bridge as bridge_mod

    source = inspect.getsource(bridge_mod)
    assert "NotImplementedError" not in source
    assert "unittest.mock" not in source
    assert "MagicMock" not in source


@pytest.mark.parametrize(
    "points",
    [
        ([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
        ([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([[float("nan"), 0, 0], [0, 0, 0], [1, 0, 0]]),
    ],
)
def test_zmat_angle_never_invents_zero_for_undefined_geometry(points):
    with pytest.raises(ValueError, match="Undefined Z-matrix angle"):
        CFOURZmatBuilder._compute_angle(
            *(np.asarray(point, dtype=float) for point in points)
        )


@pytest.mark.parametrize(
    "points",
    [
        ([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
        ([[0, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]]),
    ],
)
def test_zmat_dihedral_never_invents_zero_for_degenerate_planes(points):
    with pytest.raises(ValueError, match="Undefined Z-matrix dihedral"):
        CFOURZmatBuilder._compute_dihedral(
            *(np.asarray(point, dtype=float) for point in points)
        )


def test_zmat_rejects_coincident_nuclei_and_preserves_defined_angles():
    with pytest.raises(ValueError, match="Coincident nuclei"):
        CFOURZmatBuilder.cartesian_to_zmat(["H", "H"], [[0, 0, 0], [0, 0, 0]])
    assert CFOURZmatBuilder._compute_angle(
        np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 1.0, 0])
    ) == pytest.approx(90.0)
    assert abs(
        CFOURZmatBuilder._compute_dihedral(
            np.array([0.0, 1, 0]),
            np.zeros(3),
            np.array([1.0, 0, 0]),
            np.array([1.0, 0, 1]),
        )
    ) == pytest.approx(90.0)
