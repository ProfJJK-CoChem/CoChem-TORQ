"""
Test Counterpoise Multi-Job Decoupling & Frozen-Monomer Protocol
SRS Chunk 09, Suggestion #81 (Method Matrix v4 §9A.1-§9A.2)
Zero-Mock compliant: Real coordinate definitions, no mocks.
"""
import numpy as np
import pytest

from cochem_base.calc.cochem_calc_execution_router import (
    compute_counterpoise_interaction_energy,
    validate_counterpoise_request,
)
from Libraries.cochem_torq_engine import (
    DispatchPayload,
    calculate_discrete_counterpoise_energy,
)


def test_counterpoise_opt_deck_purges_ghost_atoms():
    """Verify that counterpoise decks do not place ghost basis functions into unconstrained ! Opt jobs."""
    symbols = ["O", "H", "H", "O", "H", "H"]
    coords = np.array([
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.7570, 0.5860],
        [0.0000, -0.7570, 0.5860],
        [2.9000, 0.0000, 0.0000],
        [3.5000, 0.7570, 0.5860],
        [3.5000, -0.7570, 0.5860],
    ])

    payload_opt = DispatchPayload(
        symbols=symbols,
        coordinates=coords,
        charge=0,
        multiplicity=1,
        method="wB97M-V",
        basis_set="def2-TZVP",
        extra_options="! Opt",
        counterpoise=True,
        ghost_atom_indices=[3, 4, 5],
        is_complex=True,
    )
    deck_opt = payload_opt.to_orca_input()
    assert ":" not in deck_opt, f"Found ghost atom syntax in ! Opt deck:\n{deck_opt}"

    payload_sp = DispatchPayload(
        symbols=symbols,
        coordinates=coords,
        charge=0,
        multiplicity=1,
        method="wB97M-V",
        basis_set="def2-TZVP",
        extra_options="",
        counterpoise=True,
        ghost_atom_indices=[3, 4, 5],
        is_complex=True,
    )
    deck_sp = payload_sp.to_orca_input()
    assert "O:" in deck_sp or "H:" in deck_sp, f"Ghost atoms missing from SP deck:\n{deck_sp}"


def test_discrete_counterpoise_interaction_energy():
    """Assert that interaction energy evaluation correctly computes E_AB^{AB} - E_A^{AB} - E_B^{AB}."""
    e_ab = -152.8500
    e_a_ghost = -76.4200
    e_b_ghost = -76.4210
    expected_delta_e = e_ab - e_a_ghost - e_b_ghost

    delta_e_engine = calculate_discrete_counterpoise_energy(e_ab, e_a_ghost, e_b_ghost)
    assert abs(delta_e_engine - expected_delta_e) < 1e-12

    delta_e_router = compute_counterpoise_interaction_energy(e_ab, e_a_ghost, e_b_ghost)
    assert abs(delta_e_router - expected_delta_e) < 1e-12


def test_unconstrained_counterpoise_opt_rejection():
    """Assert that unconstrained CP optimization on complexes is rejected."""
    with pytest.raises(ValueError, match="Unconstrained counterpoise geometry optimization is strictly prohibited"):
        validate_counterpoise_request(is_opt=True, has_frozen_constraints=False, is_complex=True)

    validate_counterpoise_request(is_opt=True, has_frozen_constraints=True, is_complex=True)
