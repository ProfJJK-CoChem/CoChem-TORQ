"""Actual RDKit forcefield and Mendeleev checks; no substituted calculators."""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
from pathlib import Path

import pytest
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import AllChem

from cochem.mobile.rdkit_bridge import (
    InvalidSmilesError,
    RDKit3DResult,
    Smiles3DConformerEngine,
    compute_quantum_electronic_states,
    relax_geometry_and_calculate_energy,
    smiles_to_3d,
    smiles_to_3d_async,
)
from Libraries.cochem_isotopes import isotope_mass


def test_unparameterized_actual_helium_pair_has_no_forcefield_result() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("[He].[He]"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    assert not AllChem.MMFFHasAllMoleculeParams(mol)
    assert not AllChem.UFFHasAllMoleculeParams(mol)
    assert relax_geometry_and_calculate_energy(mol) == (None, "unavailable", False)
    result = smiles_to_3d("[He].[He]")
    assert result.energy_kcal_mol is None
    assert not result.converged
    assert result.force_field_method == "unavailable"
    assert result.model_dump(mode="json")["energy_kcal_mol"] is None
    assert "energy_kcal_mol=unavailable" in result.xyz_block


def test_single_atom_origin_is_not_a_calculated_or_optimized_energy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "helium.xyz"
    result = smiles_to_3d("[He]", output_path=path)
    assert result.coordinates_3d == [[0.0, 0.0, 0.0]]
    assert result.energy_kcal_mol is None
    assert result.force_field_method == "unavailable"
    assert not result.converged
    assert path.read_text(encoding="utf-8") == result.xyz_block
    assert "converged=False" in result.xyz_block
    assert "electronic_state_verified=False" in result.xyz_block


def test_actual_mmff_converged_energy_matches_independent_forcefield_evaluation() -> (
    None
):
    result = smiles_to_3d("CCO", multiplicity=1)
    assert result.force_field_method == "MMFF94s"
    assert result.converged
    assert result.energy_kcal_mol is not None and math.isfinite(result.energy_kcal_mol)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, xyz in enumerate(result.coordinates_3d):
        conformer.SetAtomPosition(index, xyz)
    mol.AddConformer(conformer)
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    field = AllChem.MMFFGetMoleculeForceField(mol, properties)
    assert result.energy_kcal_mol == pytest.approx(field.CalcEnergy(), abs=1e-10)


def test_actual_unconverged_mmff_retains_evaluated_energy_and_method() -> None:
    result = smiles_to_3d("CCO", max_iters=0)
    assert result.force_field_method == "MMFF94s"
    assert not result.converged
    assert result.energy_kcal_mol is not None and result.energy_kcal_mol > 0
    assert "converged=False" in result.xyz_block


def test_actual_uff_is_explicitly_identified_for_unparameterized_mmff_radical() -> None:
    result = smiles_to_3d("[O][O]", multiplicity=3)
    assert result.force_field_method == "UFF"
    assert result.energy_kcal_mol is not None and math.isfinite(result.energy_kcal_mol)
    assert result.spin_multiplicity == 3
    assert result.spin_multiplicity_source == "user_supplied"
    assert not result.electronic_state_verified


@pytest.mark.parametrize("smiles", ["[CH3]", "[O][O]"])
def test_radical_representation_does_not_establish_ground_state(smiles: str) -> None:
    with pytest.raises(ValueError, match="Specify multiplicity explicitly"):
        smiles_to_3d(smiles)


def test_coupled_radical_and_unradicalized_oxygen_accept_explicit_valid_spins() -> None:
    for smiles in ("[O][O]", "O=O"):
        mol = Chem.MolFromSmiles(smiles)
        assert compute_quantum_electronic_states(mol, multiplicity=1) == (0, 1)
        assert compute_quantum_electronic_states(mol, multiplicity=3) == (0, 3)
    result = smiles_to_3d("O=O")
    assert result.spin_multiplicity_source == "closed_shell_valence_default"
    assert not result.electronic_state_verified


@pytest.mark.parametrize("multiplicity", [0, -1, 1.5, True, "2", 2])
def test_invalid_or_parity_incompatible_multiplicity_is_rejected(
    multiplicity: object,
) -> None:
    with pytest.raises(ValueError, match="multiplicity|Multiplicity"):
        compute_quantum_electronic_states(
            Chem.MolFromSmiles("O"), multiplicity=multiplicity
        )


def test_actual_formal_charge_and_zero_electron_spin_bound_are_respected() -> None:
    assert compute_quantum_electronic_states(Chem.MolFromSmiles("[NH4+]")) == (1, 1)
    proton = Chem.MolFromSmiles("[H+]")
    assert compute_quantum_electronic_states(proton, multiplicity=1) == (1, 1)
    with pytest.raises(ValueError, match="electron count"):
        compute_quantum_electronic_states(proton, multiplicity=3)


def test_explicit_isotopes_survive_real_rdkit_embedding_and_xyz_serialization() -> None:
    result = smiles_to_3d("[2H]O[2H]")
    assert result.atomic_symbols == ["2H", "O", "2H"]
    assert result.atomic_masses == [
        isotope_mass("2H"),
        isotope_mass("O"),
        isotope_mass("2H"),
    ]
    assert [record["mass_number"] for record in result.isotope_records] == [2, 16, 2]
    assert result.isotope_records[0]["selection_policy"] == "explicit_mass_number"
    assert (
        result.isotope_records[1]["selection_policy"]
        == "most_abundant_naturally_occurring_isotope"
    )
    assert len(result.isotope_records[0]["source"]["database_sha256"]) == 64
    assert not result.isotope_records[0]["source"]["tabulated_mass_is_exact"]
    assert result.xyz_block.splitlines()[2].split()[0] == "2H"
    assert result.xyz_block.splitlines()[4].split()[0] == "2H"
    assert result.total_mass_amu == pytest.approx(sum(result.atomic_masses))


def test_async_and_executor_paths_retain_explicit_multiplicity() -> None:
    async def run() -> None:
        direct = await smiles_to_3d_async("[H]", multiplicity=2)
        assert direct.spin_multiplicity == 2 and direct.energy_kcal_mol is None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            external = await smiles_to_3d_async(
                "[H]", executor=executor, multiplicity=2
            )
            assert external.spin_multiplicity_source == "user_supplied"
        with Smiles3DConformerEngine(max_workers=2) as engine:
            assert engine.convert("[H]", multiplicity=2).spin_multiplicity == 2
            asynchronous = await engine.convert_async("[H]", multiplicity=2)
            assert asynchronous.spin_multiplicity == 2
            batch = await engine.convert_batch_async(
                ["[H]", "[He]"], multiplicities=[2, 1]
            )
            assert [entry.spin_multiplicity for entry in batch] == [2, 1]
            with pytest.raises(ValueError, match="one multiplicity"):
                await engine.convert_batch_async(["[H]", "[He]"], multiplicities=[2])

    asyncio.run(run())


def test_schema_rejects_unavailable_energy_paired_with_success_claim() -> None:
    unavailable = smiles_to_3d("[He]").model_dump()
    for changed in (
        {"converged": True},
        {"force_field_method": "UFF"},
        {"energy_kcal_mol": 0.0},
    ):
        with pytest.raises(ValidationError):
            RDKit3DResult.model_validate(dict(unavailable, **changed))


@pytest.mark.parametrize("smiles", ["", "   ", "C(C", "C" * 2049])
def test_malformed_smiles_has_no_result(smiles: str) -> None:
    with pytest.raises(InvalidSmilesError):
        smiles_to_3d(smiles)
