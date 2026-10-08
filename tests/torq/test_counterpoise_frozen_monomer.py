"""Actual CP evidence and honest legacy deck capability checks.

Deck serialization is a software check, not an ORCA scientific qualification.
TORQ never imports the separately isolated BASE engine environment here.
"""

import numpy as np
import pytest

from cochem_torq.energetics import evaluate_interaction
from Libraries.cochem_torq_engine import (
    DispatchPayload,
    calculate_discrete_counterpoise_energy,
)


@pytest.mark.parametrize(
    "options", ["! Opt", "! TightOpt", "! VeryTightOpt", "! OptTS", "! NEB-TS"]
)
@pytest.mark.parametrize("frozen", [None, [0]])
def test_unqualified_cp_optimization_rejected_without_silent_ghost_removal(
    options, frozen
):
    payload = DispatchPayload(
        symbols=["He", "He"],
        coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        charge=0,
        multiplicity=1,
        method="HF",
        basis_set="cc-pVDZ",
        extra_options=options,
        counterpoise=True,
        ghost_atom_indices=[1],
        frozen_atom_indices=frozen,
        initial_hessian=None,
    )
    with pytest.raises(
        NotImplementedError, match="consistently differentiated CP surface"
    ):
        payload.to_orca_input()
    assert payload.ghost_atom_indices == [1]


def test_single_point_deck_retains_declared_ghost_centers_and_fragment_state():
    payload = DispatchPayload(
        symbols=["He", "He"],
        coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        charge=0,
        multiplicity=1,
        method="HF",
        basis_set="cc-pVDZ",
        counterpoise=True,
        ghost_atom_indices=[1],
        initial_hessian=None,
        # A filename containing 'optimization' is not an optimization keyword.
        moinp_path="archived/optimization-reference.gbw",
    )
    deck = payload.to_orca_input()
    assert "He:" in deck
    assert "* xyz 0 1" in deck


@pytest.mark.real_engine
def test_actual_counterpoise_components_agree_with_legacy_arithmetic(tmp_path):
    result = evaluate_interaction(
        {
            "molecule": {
                "symbols": ["He", "He"],
                "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": ["a", "b"],
            },
            "method": {
                "name": "hf",
                "basis": "cc-pvdz",
                "reference": "restricted",
                "frozen_core": False,
            },
            "fragments": [
                {"fragment_id": "a", "atom_ids": ["a"], "charge": 0, "multiplicity": 1},
                {"fragment_id": "b", "atom_ids": ["b"], "charge": 0, "multiplicity": 1},
            ],
            "geometry_protocol": "counterpoise_single_point",
        },
        tmp_path / "actual-native-cp",
    )
    assert result.status == "available", result.errors
    components = result.components
    legacy_sum = calculate_discrete_counterpoise_energy(
        components["complex"]["energy_hartree"],
        components["a_full_complex_basis"]["energy_hartree"],
        components["b_full_complex_basis"]["energy_hartree"],
    )
    assert legacy_sum == pytest.approx(
        result.quantities["interaction_counterpoise"].value_hartree, abs=1e-13
    )
    assert result.quantities["De"].status == "unavailable"
