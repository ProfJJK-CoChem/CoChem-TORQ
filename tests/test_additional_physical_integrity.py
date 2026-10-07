"""Actual RDKit/Mendeleev and numerical missingness checks, with no engine mocks."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest


@pytest.fixture(scope="module")
def inorganic_models():
    pytest.importorskip("rdkit", reason="The actual mobile package requires RDKit")
    from cochem.mobile.inorganic import models

    return models


@pytest.fixture(scope="module")
def telemetry():
    pytest.importorskip(
        "plotly", reason="Actual telemetry import requires its visualization dependency"
    )
    from Libraries import cochem_torq_telemetry

    return cochem_torq_telemetry


def _payload(smiles):
    from rdkit import Chem
    from cochem.mobile.schemas import SketcherPayloadSchema

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return SketcherPayloadSchema(smiles=smiles, molfile_v2000=Chem.MolToMolBlock(mol))


@pytest.mark.real_engine
def test_actual_rdkit_minimizer_and_energy_at_returned_coordinates():
    pytest.importorskip("rdkit", reason="Genuine force-field execution is required")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from cochem.mobile.conformer_engine import generate_3d_conformer

    result = generate_3d_conformer(_payload("CCO"))
    assert result.success and result.validation.success
    assert result.force_field_used == "MMFF94"
    assert math.isfinite(result.energy_kcal_mol)
    assert "minimizer converged" in result.validation.diagnostic_message
    # Re-evaluate a native RDKit potential at the actual returned full-precision coordinates.
    reconstructed = Chem.MolFromMolBlock(result.molfile_v3000, removeHs=False)
    conf = reconstructed.GetConformer()
    for atom in result.coordinates_3d:
        conf.SetAtomPosition(atom.atom_index, (atom.x, atom.y, atom.z))
    props = AllChem.MMFFGetMoleculeProperties(reconstructed, mmffVariant="MMFF94")
    field = AllChem.MMFFGetMoleculeForceField(reconstructed, props)
    assert result.energy_kcal_mol == pytest.approx(field.CalcEnergy(), abs=1e-10)


@pytest.mark.real_engine
def test_actual_nonconvergence_keeps_partial_geometry_without_success():
    pytest.importorskip(
        "rdkit", reason="Genuine force-field nonconvergence is required"
    )
    from cochem.mobile.conformer_engine import generate_3d_conformer

    result = generate_3d_conformer(_payload("CCO"), max_minimization_iterations=0)
    assert result.success is False
    assert (
        result.validation.success is True
    )  # Sanitization passed; optimization did not.
    assert result.energy_kcal_mol is None
    assert result.force_field_used == "MMFF94"
    assert len(result.coordinates_3d) == 9
    assert "did not converge" in result.validation.diagnostic_message
    assert "status 1" in result.validation.diagnostic_message
    assert "UFF" not in result.validation.diagnostic_message


@pytest.mark.real_engine
def test_real_parameter_unavailability_never_claims_minimization():
    pytest.importorskip(
        "rdkit", reason="Actual RDKit parameter availability is required"
    )
    from cochem.mobile.conformer_engine import generate_3d_conformer

    # RDKit does not provide MMFF94 or UFF parameters for He; no patched calculator.
    result = generate_3d_conformer(_payload("[He]"))
    assert result.success is False
    assert result.validation.success is True
    assert result.energy_kcal_mol is None and result.force_field_used is None
    assert len(result.coordinates_3d) == 1
    assert "MMFF94 parameters unavailable" in result.validation.diagnostic_message
    assert "UFF parameters unavailable" in result.validation.diagnostic_message


def test_formula_molar_mass_against_independent_rdkit_average_weight(inorganic_models):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    calculated = inorganic_models.calculate_formula_weight("C2H6O")
    independently_tabulated = Descriptors.MolWt(Chem.MolFromSmiles("CCO"))
    # Different elemental tables round their conventional standard atomic weights differently.
    assert calculated == pytest.approx(independently_tabulated, abs=0.003)


@pytest.mark.parametrize(
    "formula,counts",
    [
        ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}),
        ("[Fe(CN)6]4-", {"Fe": 1, "C": 6, "N": 6}),
        ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}),
        ("SO4^2-", {"S": 1, "O": 4}),
        ("NH4+", {"N": 1, "H": 4}),
    ],
)
def test_nested_adduct_and_charge_grammar_uses_actual_weights(
    inorganic_models, formula, counts
):
    from mendeleev import element

    independently_counted = sum(
        float(element(symbol).atomic_weight) * number
        for symbol, number in counts.items()
    )
    assert inorganic_models.calculate_formula_weight(formula) == pytest.approx(
        independently_counted, abs=1e-10
    )


@pytest.mark.parametrize(
    "formula",
    [
        "",
        "not-a-formula",
        "C2H6O???",
        "Xx2O",
        "H0",
        "H01",
        "C(H2",
        "C)H2",
        "[]",
        "H2O..NaCl",
        "H2O garbage",
        "H2O+garbage",
        "Fe3+",
        "Cu0.5O",
    ],
)
def test_invalid_or_unknown_formula_cannot_supply_partial_mass(
    inorganic_models, formula
):
    with pytest.raises(ValueError):
        inorganic_models.calculate_formula_weight(formula)


def test_metal_and_radius_data_do_not_become_invented_defaults(inorganic_models):
    from mendeleev import element

    donor = inorganic_models.DonorAtom(symbol="N", index=0)
    assert donor.covalent_radius_angstrom == pytest.approx(
        element("N").covalent_radius / 100.0
    )
    iron = inorganic_models.MetalCenter(symbol="Fe", oxidation_state=2)
    assert iron.covalent_radius_angstrom == pytest.approx(
        element("Fe").covalent_radius / 100.0
    )
    with pytest.raises(ValueError, match="outside the implemented"):
        inorganic_models.MetalCenter(symbol="Ca", oxidation_state=2).d_electrons
    cerium = inorganic_models.MetalCenter(symbol="Ce", oxidation_state=3)
    assert cerium.group_id is None  # Actual missing table datum, not patched metadata.
    with pytest.raises(ValueError, match="unavailable for f-block"):
        _ = cerium.d_electrons
    with pytest.raises(ValueError, match="qualified state/configuration"):
        _ = cerium.f_electrons
    with pytest.raises(ValueError, match="cannot be inferred"):
        cerium.determine_spin_multiplicity()
    with pytest.raises(ValueError, match="No qualified spin estimate"):
        iron.determine_spin_multiplicity(inorganic_models.CoordinationPolyhedron.LINEAR)


def test_pair_distance_is_actual_and_single_atom_is_not_applicable(telemetry):
    coordinates = np.asarray([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 2.0]])
    distance, pair = telemetry._compute_pairwise_distances(coordinates)
    assert distance == 2.0 and pair == (1, 2)
    moved = coordinates @ np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    ) + [17.0, -5.0, 3.0]
    assert telemetry._compute_pairwise_distances(moved) == (distance, pair)
    assert telemetry._compute_pairwise_distances([[0.0, 0.0, 0.0]]) == (None, None)
    # A genuinely coincident pair has a computed zero, with its real atom indices.
    assert telemetry._compute_pairwise_distances(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ) == (0.0, (0, 1))


@pytest.mark.parametrize(
    "coordinates",
    [
        [],
        [1.0, 2.0, 3.0],
        [[0.0, 0.0]],
        [[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        [[np.inf, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[1e300, 0.0, 0.0], [-1e300, 0.0, 0.0]],
    ],
)
def test_invalid_distances_raise_without_inventing_a_pair(telemetry, coordinates):
    with pytest.raises(ValueError):
        telemetry._compute_pairwise_distances(coordinates)


def test_single_atom_crash_artifact_keeps_missing_distance_explicit(
    telemetry, tmp_path
):
    exported = telemetry.export_crash_animation(
        np.asarray([[[0.123456789123, 0.0, 0.0]]]),
        symbols=["He"],
        artifact_dir=tmp_path / "artifacts",
        scratch_dir=tmp_path / "scratch",
        abort_reason="Actual input diagnostic",
    )
    report = json.loads(exported["diagnostic_path"].read_text())
    assert report["min_interatomic_distance"] is None
    assert report["colliding_pair"] is None
    assert report["distance_status"] == "not_applicable"
    assert "Fewer than two atoms" in report["distance_reason"]
    assert report["initial_energy_hartree"] is None
    assert report["max_gradient_norm"] is None
    xyz = exported["xyz_path"].read_text().splitlines()
    assert "not applicable" in xyz[1]
    assert float(xyz[2].split()[1]) == 0.123456789123


def test_crash_minimum_and_small_distance_remain_real(telemetry, tmp_path):
    trajectory = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0]]]
    )
    exported = telemetry.export_crash_animation(
        trajectory,
        symbols=["H", "H"],
        artifact_dir=tmp_path / "artifacts",
        scratch_dir=tmp_path / "scratch",
    )
    report = json.loads(exported["diagnostic_path"].read_text())
    assert report["min_interatomic_distance"] == 1e-9
    assert report["colliding_pair"] == [0, 1]
    assert report["crash_frame_index"] == 1
    assert report["distance_status"] == "computed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"symbols": ["X", "H"]},
        {"symbols": ["H"]},
        {"symbols": ["H", "H"], "energies": [np.nan]},
        {"symbols": ["H", "H"], "gradients": [[[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]]},
        {"symbols": ["H", "H"], "error_node_id": "../../escape"},
    ],
)
def test_crash_export_rejects_missing_identity_and_invalid_observations(
    telemetry, tmp_path, kwargs
):
    with pytest.raises(ValueError):
        telemetry.export_crash_animation(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            artifact_dir=tmp_path / "artifacts",
            scratch_dir=tmp_path / "scratch",
            **kwargs,
        )
    assert not list(tmp_path.rglob("*.xyz"))
    assert not list(tmp_path.rglob("*.json"))


def test_coordination_template_cannot_certify_electronic_geometry(inorganic_models):
    metal = inorganic_models.MetalCenter(symbol="Fe", oxidation_state=2)
    for template in inorganic_models.CoordinationPolyhedron:
        compatible, reason = metal.check_geometry_compatibility(template)
        assert compatible is None
        assert "undetermined" in reason
        assert "electronic state" in reason
