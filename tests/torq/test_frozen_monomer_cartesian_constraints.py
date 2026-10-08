"""
Test Frozen-Monomer Cartesian Coordinate Constraints
SRS Chunk 09, Suggestion #82 (Method Matrix v4 §9A.1-§9A.2)
Zero-Mock compliant: Explicit water dimer coordinate geometry.
"""

import numpy as np
import pytest

from Libraries.cochem_torq_constraints import (
    HARTREE_PER_BOHR_TO_NEWTON,
    ConstraintCoordinateType,
    TorqConstraintManager,
    audit_gradient_convergence,
    compute_rotational_constants,
    generate_orca_constraint_block,
    generate_orca_frozen_monomer_constraints_block,
    get_dynamic_isotopic_mass,
    get_dynamic_vdw_radius,
    propagate_rotational_error,
)
from Libraries.cochem_torq_engine import DispatchPayload


def test_water_dimer_cartesian_constraints_generation():
    """Freeze Monomer A atom rows without angle or dihedral constraints."""
    symbols = ["O", "H", "H", "O", "H", "H"]
    coords = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.7570, 0.5860],
            [0.0000, -0.7570, 0.5860],
            [2.9000, 0.0000, 0.0000],
            [3.5000, 0.7570, 0.5860],
            [3.5000, -0.7570, 0.5860],
        ]
    )

    atoms_a = [0, 1, 2]
    atoms_b = [3, 4, 5]

    geom_block = generate_orca_frozen_monomer_constraints_block(
        atoms_a=atoms_a,
        atoms_b=atoms_b,
        symbols=symbols,
        coordinates=coords,
        frozen_monomer="A",
        mode="cartesian",
    )

    assert "TolE 1e-7" in geom_block
    assert "TolMaxG 1e-5" in geom_block

    for idx in atoms_a:
        expected_str = f"{{ C {idx} C }}"
        assert expected_str in geom_block

    for idx in atoms_b:
        unexpected_str = f"{{ C {idx} C }}"
        assert unexpected_str not in geom_block

    assert "{ B " not in geom_block, "Found illegal distance constraint { B ... }"
    assert "{ A " not in geom_block, "Found illegal angle constraint { A ... }"
    assert "{ D " not in geom_block, "Found illegal dihedral constraint { D ... }"


def test_internal_request_is_never_silently_replaced_by_cartesian_freezing():
    with pytest.raises(RuntimeError, match="reviewed isolated BASE"):
        generate_orca_frozen_monomer_constraints_block(
            atoms_a=[0],
            atoms_b=[1],
            symbols=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 1.4]],
            mode="internal",
        )


def test_single_monomer_has_no_invented_intermolecular_reference_distance():
    with pytest.raises(ValueError, match="measured positive finite"):
        TorqConstraintManager().analyze_and_enforce(
            ["O", "H", "H"],
            [[0, 0, 0], [0, 0.757, 0.586], [0, -0.757, 0.586]],
        )


def test_linear_and_atomic_rotors_preserve_undefined_observables():
    rotor = compute_rotational_constants(["H", "H"], [[0, 0, -0.37], [0, 0, 0.37]])
    assert rotor.I_a_amu_angstrom2 == 0
    assert rotor.A_MHz is None and rotor.A_cm1 is None and rotor.kappa is None
    assert rotor.B_MHz == rotor.C_MHz and rotor.B_MHz > 0
    assert rotor.rotor_type == "linear"
    atom = compute_rotational_constants(["He"], [[1, 2, 3]])
    assert atom.rotor_type == "atom"
    assert atom.A_MHz is atom.B_MHz is atom.C_MHz is atom.kappa is None


def test_unknown_base_constants_and_monomer_sensitivity_remain_absent():
    budget = propagate_rotational_error(R_angstrom=3.0, delta_R_angstrom=0.003)
    assert budget.delta_B_over_B_percent == pytest.approx(-0.2)
    assert budget.delta_B_MHz is None
    assert budget.delta_A_MHz is None and budget.delta_A_over_A_percent is None
    assert budget.equivalent_monomer_error_for_B_angstrom is None
    assert budget.reference_bond_length_angstrom is None
    assert budget.monomer_B_sensitivity_percent_per_angstrom is None
    assert budget.accuracy_qualified is False
    assert "no accuracy target" in budget.headline_verdict


def test_missing_tabulated_values_are_not_replaced_by_generic_values():
    with pytest.raises(ValueError, match="No tabulated mass"):
        get_dynamic_isotopic_mass("C", 1000)
    with pytest.raises(ValueError, match="No tabulated van der Waals radius"):
        get_dynamic_vdw_radius("Og")
    assert get_dynamic_isotopic_mass("H") == get_dynamic_isotopic_mass("H", 1)


def test_missing_gradient_does_not_claim_convergence_or_hydrogen_bond():
    geometry = [
        [0, 0, 0],
        [0, 0, 1.162],
        [0, 0, -1.162],
        [2.836, 0, 0],
        [2.836, 0.757, 0.586],
        [2.836, -0.757, 0.586],
    ]
    result = TorqConstraintManager().analyze_and_enforce(
        ["C", "O", "O", "O", "H", "H"],
        geometry,
    )
    assert result.success is False and result.convergence_status == "unassessed"
    assert result.convergence_audit is None
    assert result.cfour_zmat_constraints_block is None
    assert result.error_budget.accuracy_qualified is False
    assert all(
        fragment.is_hydrogen_bonded is None for fragment in result.monomer_fragments
    )
    assert result.monomer_fragments[1].has_potential_hydrogen_bond_donor is True


def test_absolute_distance_sensitivity_uses_actual_geometric_rotor():
    rotor = compute_rotational_constants(
        ["O", "H", "H"],
        [[0, 0, 0], [0, 0.757, 0.586], [0, -0.757, 0.586]],
    )
    budget = propagate_rotational_error(3.0, 0.003, base_constants=rotor)
    assert budget.delta_B_MHz == pytest.approx(rotor.B_MHz * -0.002)
    assert budget.delta_A_MHz is None and budget.accuracy_qualified is False


@pytest.mark.parametrize("distance", [0, -1, float("nan"), float("inf")])
def test_undefined_effective_distance_cannot_become_an_error_budget(distance):
    with pytest.raises(ValueError):
        propagate_rotational_error(distance)


def test_hartree_per_bohr_force_conversion_matches_published_si_formula():
    from scipy.constants import physical_constants

    expected = (
        physical_constants["Hartree energy"][0] / physical_constants["Bohr radius"][0]
    )
    assert HARTREE_PER_BOHR_TO_NEWTON == pytest.approx(expected, rel=1e-15)
    assert HARTREE_PER_BOHR_TO_NEWTON * 1e-5 == pytest.approx(8.2387e-13, rel=1e-5)


@pytest.mark.parametrize(
    "gradient",
    [np.empty((0, 3)), [[float("nan"), 0, 0]], [[complex(0, 1), 0, 0]], [[0, 0]]],
)
def test_invalid_gradients_cannot_establish_convergence(gradient):
    with pytest.raises(ValueError):
        audit_gradient_convergence(gradient)


@pytest.mark.parametrize(
    "mode", [ConstraintCoordinateType.INTERNAL_ALL, ConstraintCoordinateType.RIGID_BODY]
)
def test_unimplemented_constraint_modes_never_emit_empty_successful_blocks(mode):
    with pytest.raises(RuntimeError, match="no substitute"):
        generate_orca_constraint_block([0], ["H"], [[0, 0, 0]], mode)


def test_internal_request_cannot_substitute_cartesian_when_geometry_missing():
    with pytest.raises(ValueError, match="require explicit symbols and geometry"):
        generate_orca_constraint_block(
            [0], mode=ConstraintCoordinateType.INTERNAL_BONDS
        )


@pytest.mark.parametrize(
    ("atoms_a", "atoms_b", "coordinates"),
    [
        ([0], [0], [[0, 0, 0], [0, 0, 1.4]]),
        ([True], [1], [[0, 0, 0], [0, 0, 1.4]]),
        ([0], [2], [[0, 0, 0], [0, 0, 1.4]]),
        ([0], [1], [[0, 0, 0], [0, 0, float("nan")]]),
    ],
)
def test_cartesian_constraints_reject_ambiguous_rows_and_invalid_geometry(
    atoms_a,
    atoms_b,
    coordinates,
):
    with pytest.raises(ValueError):
        generate_orca_frozen_monomer_constraints_block(
            atoms_a,
            atoms_b,
            ["H", "H"],
            coordinates,
            mode="cartesian",
        )


def test_dispatch_payload_orca_deck_emits_cartesian_constraints():
    """Emit Cartesian constraints for exactly the explicitly frozen atom rows."""
    symbols = ["O", "H", "H", "O", "H", "H"]
    coords = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.7570, 0.5860],
            [0.0000, -0.7570, 0.5860],
            [2.9000, 0.0000, 0.0000],
            [3.5000, 0.7570, 0.5860],
            [3.5000, -0.7570, 0.5860],
        ]
    )

    payload = DispatchPayload(
        symbols=symbols,
        coordinates=coords,
        charge=0,
        multiplicity=1,
        method="wB97M-V",
        basis_set="def2-TZVP",
        frozen_atom_indices=[0, 1, 2],
        is_complex=True,
    )
    deck = payload.to_orca_input()

    assert "{ C 0 C }" in deck
    assert "{ C 1 C }" in deck
    assert "{ C 2 C }" in deck
    assert "{ C 3 C }" not in deck
    assert "{ B " not in deck
    assert "{ A " not in deck
    assert "{ D " not in deck
