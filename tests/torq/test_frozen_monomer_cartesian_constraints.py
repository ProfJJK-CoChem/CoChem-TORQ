"""
Test Frozen-Monomer Cartesian Coordinate Constraints
SRS Chunk 09, Suggestion #82 (Method Matrix v4 §9A.1-§9A.2)
Zero-Mock compliant: Explicit water dimer coordinate geometry.
"""
import numpy as np

from Libraries.cochem_torq_constraints import generate_orca_frozen_monomer_constraints_block
from Libraries.cochem_torq_engine import DispatchPayload


def test_water_dimer_cartesian_constraints_generation():
    """Assert all atoms of Monomer A are assigned { C i C } Cartesian constraints without angle/dihedral drift."""
    symbols = ["O", "H", "H", "O", "H", "H"]
    coords = np.array([
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.7570, 0.5860],
        [0.0000, -0.7570, 0.5860],
        [2.9000, 0.0000, 0.0000],
        [3.5000, 0.7570, 0.5860],
        [3.5000, -0.7570, 0.5860],
    ])

    atoms_a = [0, 1, 2]
    atoms_b = [3, 4, 5]

    geom_block = generate_orca_frozen_monomer_constraints_block(
        atoms_a=atoms_a,
        atoms_b=atoms_b,
        symbols=symbols,
        coordinates=coords,
        frozen_monomer="A",
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


def test_dispatch_payload_orca_deck_emits_cartesian_constraints():
    """Verify that DispatchPayload.to_orca_input emits { C i C } for frozen_atom_indices."""
    symbols = ["O", "H", "H", "O", "H", "H"]
    coords = np.array([
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.7570, 0.5860],
        [0.0000, -0.7570, 0.5860],
        [2.9000, 0.0000, 0.0000],
        [3.5000, 0.7570, 0.5860],
        [3.5000, -0.7570, 0.5860],
    ])

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
