"""Actual declared-state, geometric/database and classical UFF integrity checks.

These establish software/model contracts, not coordination-complex stability,
quantum-state assignment, or experimentally validated molecular accuracy.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from mendeleev import element
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from cochem.mobile.assembly.alignment import (
    align_ligand_to_template,
    calculate_metal_donor_distance,
    kabsch_fit_proper,
    rotation_matrix_from_vectors,
)
from cochem.mobile.assembly.clash import (
    evaluate_steric_clashes,
    resolve_clashes_and_report,
    run_constrained_uff_relaxation,
)
from cochem.mobile.assembly.engine import assemble_complex_sync, infer_ligand_charge
from cochem.mobile.assembly.exceptions import (
    CoordinationGeometryMismatchError,
    QuantumParityError,
    SingularRotationAxisError,
    StericClashDetectedError,
)
from cochem.mobile.assembly.geometries import get_coordination_template_vectors
from cochem.mobile.assembly.models import (
    ComplexAssemblyRequest,
    CoordinationGeometryEnum,
    LigandAttachment,
)
from cochem.mobile.assembly.storage import read_complex_from_hdf5


def _chlorine(name="chloride", *, charge=None, slot=0):
    return LigandAttachment(
        ligand_id=name,
        donor_atom_indices=[0],
        denticity=1,
        target_vector_slots=[slot],
        atomic_symbols=["Cl"],
        coordinates=[(0.0, 0.0, 0.0)],
        formal_charge=charge,
    )


def _request(*, charges=(-1, -1), multiplicity=1, metal="Ni", oxidation=2):
    return ComplexAssemblyRequest(
        metal_symbol=metal,
        oxidation_state=oxidation,
        geometry=CoordinationGeometryEnum.LINEAR,
        ligands=[_chlorine(charge=q, slot=i) for i, q in enumerate(charges)],
        spin_multiplicity=multiplicity,
    )


@pytest.mark.parametrize("name", ["chloride", "cyanide", "acetate", "neutral", "CN-"])
def test_name_or_composition_never_establishes_charge(name):
    inspected = _chlorine(name)
    assert inspected.formal_charge is None
    template = get_coordination_template_vectors(CoordinationGeometryEnum.LINEAR)
    assert np.isfinite(align_ligand_to_template(inspected, "Ni", template)).all()
    with pytest.raises(ValueError, match="requires an explicit formal_charge"):
        infer_ligand_charge(inspected)


@pytest.mark.parametrize("charge", [-2, -1, 0, 1, 2])
def test_explicit_declared_charge_is_preserved_independently_of_name(charge):
    assert infer_ligand_charge(_chlorine("neutral_cyanide", charge=charge)) == charge
    assert infer_ligand_charge(_chlorine("chloride", charge=charge)) == charge


@pytest.mark.parametrize("charge", [True, False, -1.0, "-1"])
def test_charge_requires_a_typed_integer(charge):
    with pytest.raises(ValidationError):
        _chlorine(charge=charge)


def test_missing_charge_fails_before_any_state_bearing_artifact(tmp_path):
    target = tmp_path / "not-created.h5"
    with pytest.raises(ValueError, match="formal_charge"):
        assemble_complex_sync(_request(charges=(None, -1)), h5_path=target)
    assert not target.exists()
    assert not target.with_suffix(".lock").exists()


def test_whole_complex_parity_accepts_declared_odd_state_and_preserves_hdf5(tmp_path):
    # A caller-declared odd-electron proposal: no stability claim is made.
    request = _request(charges=(-1, 0), multiplicity=2)
    target = tmp_path / "declared-state.h5"
    result = assemble_complex_sync(request, h5_path=target)
    assert result.total_formal_charge == 1
    total_z = element("Ni").atomic_number + 2 * element("Cl").atomic_number
    assert result.total_electron_count == total_z - 1 == 61
    assert result.spin_multiplicity == 2
    assert result.geometry_status == "initial_geometric_proposal"
    assert result.chemical_state_source == "caller_declared_not_independently_verified"
    assert result.scientific_qualification is False
    assert result.quantum_calculation_performed is False
    assert result.actual_d_population is None
    assert result.uff_energy_kcal_mol is None
    assert result.uff_status == "unavailable_without_declared_molecular_graph"
    stored = read_complex_from_hdf5(target, result.hdf5_record_key)
    np.testing.assert_array_equal(stored["atomic_numbers"], [28, 17, 17])
    radius = calculate_metal_donor_distance("Ni", "Cl")
    np.testing.assert_allclose(
        stored["coordinates"], [[0, 0, 0], [0, 0, radius], [0, 0, -radius]]
    )
    assert stored["metadata"]["ligand_formal_charges"] == [-1, 0]
    assert stored["metadata"]["total_electron_count"] == 61
    assert stored["metadata"]["scientific_qualification"] is False
    assert stored["metadata"]["uff_energy_kcal_mol"] is None
    assert result.clash_report.initial_clash_detected is False
    assert result.clash_report.clash_resolved is False
    assert result.clash_report.radius_source == "mendeleev.vdw_radius"
    assert "bondi_threshold" not in result.clash_report.model_dump()


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_whole_complex_parity_rejects_even_spin_for_odd_electron_state(
    tmp_path, multiplicity
):
    target = tmp_path / "rejected.h5"
    with pytest.raises(QuantumParityError, match="parity"):
        assemble_complex_sync(
            _request(charges=(-1, 0), multiplicity=multiplicity), h5_path=target
        )
    assert not target.exists()


def test_nonpositive_whole_complex_electron_count_is_rejected(tmp_path):
    with pytest.raises(QuantumParityError, match="positive whole-complex"):
        assemble_complex_sync(
            _request(charges=(40, 40)), h5_path=tmp_path / "absent.h5"
        )


def test_spin_larger_than_actual_whole_electron_count_is_rejected(tmp_path):
    with pytest.raises(QuantumParityError):
        assemble_complex_sync(
            _request(multiplicity=101), h5_path=tmp_path / "absent.h5"
        )


def test_formal_bookkeeping_outside_d_shell_is_retained_as_unavailable(tmp_path):
    # Actual Zn group 12 yields raw formal bookkeeping 12 for declared Zn(0).
    result = assemble_complex_sync(
        _request(metal="Zn", oxidation=0), h5_path=tmp_path / "proposal.h5"
    )
    assert result.formal_valence_electron_count == element("Zn").group_id == 12
    assert result.d_electron_count is None
    assert result.d_electron_count_status == "outside_d_shell_range"
    assert result.actual_d_population is None
    assert result.scientific_qualification is False


def test_declared_charge_changes_provenance_and_preserves_previous_state(tmp_path):
    path = tmp_path / "states.h5"
    first = assemble_complex_sync(_request(charges=(-1, -1)), h5_path=path)
    record = read_complex_from_hdf5(path, first.hdf5_record_key)
    digest = hashlib.sha256(record["coordinates"].tobytes()).hexdigest()
    second = assemble_complex_sync(_request(charges=(0, 0)), h5_path=path)
    assert first.sha256_provenance != second.sha256_provenance
    assert (first.total_formal_charge, second.total_formal_charge) == (0, 2)
    retained = read_complex_from_hdf5(path, first.hdf5_record_key)
    assert hashlib.sha256(retained["coordinates"].tobytes()).hexdigest() == digest
    assert retained["metadata"]["ligand_formal_charges"] == [-1, -1]


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), True, "0.0"]
)
def test_coordinates_require_finite_actual_numeric_values(value):
    payload = _chlorine(charge=-1).model_dump()
    payload["coordinates"] = [(value, 0.0, 0.0)]
    with pytest.raises(ValidationError):
        LigandAttachment.model_validate(payload)


@pytest.mark.parametrize("field", ["oxidation_state", "spin_multiplicity"])
def test_request_state_integers_do_not_accept_boolean_or_float(field):
    payload = _request().model_dump(mode="json")
    for value in (True, 2.0, "2"):
        payload[field] = value
        with pytest.raises(ValidationError):
            ComplexAssemblyRequest.model_validate(payload)


def test_duplicate_coordination_slots_fail_before_persistence(tmp_path):
    payload = _request().model_dump(mode="json")
    payload["ligands"][1]["target_vector_slots"] = [0]
    with pytest.raises(CoordinationGeometryMismatchError):
        assemble_complex_sync(
            ComplexAssemblyRequest.model_validate(payload),
            h5_path=tmp_path / "absent.h5",
        )


@pytest.mark.parametrize("geometry", list(CoordinationGeometryEnum))
def test_actual_template_vectors_are_finite_unit_proposal_directions(geometry):
    vectors = get_coordination_template_vectors(geometry)
    assert vectors.shape == (geometry.coordination_number, 3)
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-12)


@pytest.mark.parametrize(
    "source,target", [([0, 0, 0], [0, 0, 1]), ([1, 0, 0], [0, 0, 0])]
)
def test_zero_vector_never_substitutes_identity_rotation(source, target):
    with pytest.raises(SingularRotationAxisError):
        rotation_matrix_from_vectors(np.array(source), np.array(target))


@pytest.mark.parametrize(
    "bad",
    [
        np.array([np.nan, 0, 1]),
        np.array([np.inf, 0, 1]),
        np.array([1j, 0, 1]),
        np.array([1, 0]),
    ],
)
def test_nonreal_nonfinite_or_wrong_shape_rotation_is_rejected(bad):
    with pytest.raises(ValueError):
        rotation_matrix_from_vectors(bad, np.array([1.0, 0.0, 0.0]))


@pytest.mark.parametrize("target", [[1, 0, 0], [-1, 0, 0], [0, 0, 1]])
def test_parallel_antiparallel_and_orthogonal_rotations_remain_proper(target):
    source = np.array([1.0, 0.0, 0.0])
    matrix = rotation_matrix_from_vectors(source, np.array(target))
    np.testing.assert_allclose(matrix @ source, target, atol=1e-12)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1)


def test_rigid_alignment_preserves_internal_distances_without_hydrogen_reassignment():
    # Declared inspection conformer: the former proximity-based reflection
    # changed an individual H position, violating the rigid-conformer contract.
    coords = np.array([[0, 0, 0], [0, 0, 1], [0.2, 0, -0.8]])
    ligand = LigandAttachment(
        ligand_id="declared-inspection",
        donor_atom_indices=[0],
        denticity=1,
        target_vector_slots=[0],
        atomic_symbols=["O", "H", "H"],
        coordinates=coords.tolist(),
        formal_charge=0,
    )
    template = get_coordination_template_vectors(CoordinationGeometryEnum.LINEAR)
    aligned = align_ligand_to_template(ligand, "Ni", template)
    before = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    after = np.linalg.norm(aligned[:, None] - aligned[None, :], axis=-1)
    np.testing.assert_allclose(after, before, atol=1e-12)


def test_actual_kabsch_rotation_preserves_geometry_and_matches_target():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([[2.0, 3.0, 1.0], [2.0, 4.0, 1.0], [1.0, 3.0, 1.0]])
    matrix, source_center, target_center = kabsch_fit_proper(source, target)
    np.testing.assert_allclose(
        (source - source_center) @ matrix.T + target_center, target, atol=1e-12
    )
    assert np.linalg.det(matrix) == pytest.approx(1)


def test_database_contact_cutoff_uses_named_source_and_actual_pair_distance():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    clash, pairs, distance, cutoff, penalty = evaluate_steric_clashes(
        ["Ni", "Cl", "Cl"], coords, [(1, 2), (2, 3)], {1, 2}
    )
    assert clash is False and pairs == [] and penalty == 0
    assert distance == 4
    assert cutoff == pytest.approx(0.6 * 2 * element("Cl").vdw_radius / 100)


def test_no_eligible_pair_is_unavailable_not_fabricated_zero():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    values = evaluate_steric_clashes(["Ni", "Cl", "Cl"], coords, [(1, 3)], {1, 2})
    assert values == (False, [], None, None, 0.0)
    final, report, energy = resolve_clashes_and_report(
        ["Ni", "Cl", "Cl"], coords, [(1, 3)], [], {1, 2}
    )
    np.testing.assert_array_equal(final, coords)
    assert report.evaluated_pair_count == 0
    assert report.min_observed_distance is None
    assert report.vdw_contact_threshold is None
    assert report.clash_resolved is False and energy is None
    assert json.loads(report.model_dump_json())["min_observed_distance"] is None


def test_unresolved_contacts_cannot_trigger_guessed_graph_uff():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.01]])
    with pytest.raises(StericClashDetectedError, match="explicit molecular graph"):
        resolve_clashes_and_report(
            ["Ni", "Cl", "Cl"], coords, [(1, 2), (2, 3)], [], {1, 2}
        )


def test_missing_uff_graph_retains_proposal_and_unavailable_energy():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    result, energy = run_constrained_uff_relaxation(["Ni", "Cl", "Cl"], coords, {1, 2})
    np.testing.assert_array_equal(result, coords)
    assert energy is None


def test_actual_graph_uff_energy_matches_returned_geometry_without_mutating_input():
    graph = Chem.AddHs(Chem.MolFromSmiles("O"))
    before = Chem.MolToMolBlock(graph)
    initial = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 3.0, 0.0]])
    coords, energy = run_constrained_uff_relaxation(
        ["O", "H", "H"], initial, set(), molecule=graph
    )
    assert energy is not None and np.isfinite(energy)
    np.testing.assert_array_equal(coords[0], initial[0])
    assert Chem.MolToMolBlock(graph) == before
    retained = Chem.Mol(graph)
    conformer = Chem.Conformer(3)
    for index, position in enumerate(coords):
        conformer.SetAtomPosition(index, Point3D(*(float(v) for v in position)))
    retained.AddConformer(conformer)
    force_field = AllChem.UFFGetMoleculeForceField(
        retained, ignoreInterfragInteractions=False
    )
    assert force_field.CalcEnergy() == pytest.approx(energy, abs=1e-12)


def test_actual_uff_nonconvergence_cannot_publish_energy_or_geometry():
    graph = Chem.AddHs(Chem.MolFromSmiles("O"))
    initial = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 3.0, 0.0]])
    with pytest.raises(RuntimeError, match="did not converge.*status 1"):
        run_constrained_uff_relaxation(
            ["O", "H", "H"], initial, set(), molecule=graph, max_iterations=1
        )
    np.testing.assert_array_equal(initial, [[0, 0, 0], [1.5, 0, 0], [0, 3, 0]])


def test_actual_unparameterized_graph_has_no_substituted_uff_energy():
    graph = Chem.MolFromSmiles("[He]")
    assert not AllChem.UFFHasAllMoleculeParams(graph)
    with pytest.raises(ValueError, match="complete UFF parameters"):
        run_constrained_uff_relaxation(["He"], np.zeros((1, 3)), set(), molecule=graph)


def test_graph_order_and_implicit_hydrogens_must_match_declared_atoms():
    graph = Chem.MolFromSmiles("O")
    with pytest.raises(ValueError, match="explicitly contain all hydrogen"):
        run_constrained_uff_relaxation(["O"], np.zeros((1, 3)), set(), molecule=graph)
    with pytest.raises(ValueError, match="atom order"):
        run_constrained_uff_relaxation(["He"], np.zeros((1, 3)), set(), molecule=graph)
