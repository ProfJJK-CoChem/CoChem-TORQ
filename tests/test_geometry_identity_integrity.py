"""Real RDKit graph operations and explicit mathematical Cartesian examples."""

import copy

import numpy as np
import pytest

from cochem_torq.domain import Molecule, PrerequisiteError
from cochem_torq.geometry_identity import (
    ComparisonPolicy,
    GeometryComparison,
    IndexedGeometry,
    SymmetryPolicy,
    SymmetryProposal,
    compare_indexed_geometries,
    indexed_geometry_from_rdkit,
    propose_nuclear_symmetry,
)

pytestmark = pytest.mark.cpu_ml


@pytest.fixture(scope="module")
def chem():
    return pytest.importorskip("rdkit.Chem")


def record(chem, smiles, coordinates, *, ids=None):
    graph = chem.AddHs(chem.MolFromSmiles(smiles))
    molecule = Molecule(
        symbols=[atom.GetSymbol() for atom in graph.GetAtoms()],
        isotopes=[atom.GetIsotope() or None for atom in graph.GetAtoms()],
        atom_ids=ids or [f"atom-{index}" for index in range(graph.GetNumAtoms())],
        geometry_bohr=np.asarray(coordinates, dtype=float).tolist(),
        charge=sum(atom.GetFormalCharge() for atom in graph.GetAtoms()),
        multiplicity=1,
    )
    return graph, indexed_geometry_from_rdkit(
        graph, molecule, graph_source="explicit mathematical graph/geometry case"
    )


def tetrahedron(chem, smiles="[C@H](F)(Cl)Br"):
    return record(
        chem,
        smiles,
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
    )


def water(chem, *, deuterated=False, distortion=0.0):
    return record(
        chem,
        "[2H]O" if deuterated else "O",
        [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [-1.0 + distortion, 1.0, 0.0]]
        if deuterated
        else [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0 + distortion, 1.0, 0.0]],
    )


def policy(**changes):
    return ComparisonPolicy(
        equivalent_rmsd_bohr=1e-8, distinct_rmsd_bohr=1e-5, **changes
    )


def test_real_rdkit_indexed_graph_preserves_every_explicit_atom_and_stereo(chem):
    _, original = tetrahedron(chem)
    restored = IndexedGeometry.model_validate_json(original.model_dump_json())
    assert restored == original
    assert "@" in original.indexed_graph_smiles
    assert original.stereo_completeness == "supported_assignments_complete"
    assert original.molecule.atom_ids == [
        "atom-0",
        "atom-1",
        "atom-2",
        "atom-3",
        "atom-4",
    ]


def test_known_permutation_translation_and_proper_rotation_recover_stable_ids(chem):
    graph, original = tetrahedron(chem)
    order = [4, 2, 0, 3, 1]
    transformed_graph = chem.RenumberAtoms(graph, order)
    angle = 0.61
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([4.0, -3.0, 2.0])
    coordinates = (
        np.asarray(original.molecule.geometry_bohr) @ rotation + translation
    )[order]
    target = indexed_geometry_from_rdkit(
        transformed_graph,
        Molecule(
            symbols=[original.molecule.symbols[index] for index in order],
            isotopes=[None] * 5,
            atom_ids=[original.molecule.atom_ids[index] for index in order],
            geometry_bohr=coordinates.tolist(),
            charge=0,
            multiplicity=1,
        ),
        graph_source="actual RDKit RenumberAtoms",
    )
    report = compare_indexed_geometries(
        original, target, policy(atom_id_policy="shared_ids_authoritative")
    )
    assert report.status == "equivalent"
    assert report.graph_isomorphism == "verified" and report.enumeration_complete
    assert report.mappings_evaluated == report.acceptable_mapping_count == 1
    alignment = report.alignments[0]
    assert alignment.rmsd_bohr < 1e-14
    assert np.linalg.det(alignment.rotation_row_convention) == pytest.approx(
        1.0, abs=1e-12
    )
    assert all(left == right for left, right in alignment.source_to_target_atom_ids)
    assert not report.scientific_accuracy_established
    assert report.disposition == "retain_all_candidates; no_automatic_exclusion"


def test_actual_enantiomer_graphs_do_not_become_equivalent(chem):
    _, left = tetrahedron(chem)
    _, right = tetrahedron(chem, "[C@@H](F)(Cl)Br")
    report = compare_indexed_geometries(left, right, policy())
    assert report.status == "distinct" and report.graph_isomorphism == "not_isomorphic"
    assert not report.alignments


def test_reflected_cartesian_geometry_cannot_be_aligned_by_an_improper_rotation(chem):
    graph, original = tetrahedron(chem)
    payload = original.molecule.model_dump(mode="json")
    reflected = np.asarray(payload["geometry_bohr"])
    reflected[:, 0] *= -1
    payload["geometry_bohr"] = reflected.tolist()
    mirror = indexed_geometry_from_rdkit(
        graph,
        Molecule.model_validate(payload),
        graph_source="explicit reflected mathematical geometry",
    )
    report = compare_indexed_geometries(original, mirror, policy())
    assert report.graph_isomorphism == "verified" and report.status == "distinct"
    assert report.alignments[0].rmsd_bohr > 0.1
    assert np.linalg.det(report.alignments[0].rotation_row_convention) > 0.999999
    assert report.alignments[0].orientation_status == "underdetermined"


def test_e_z_double_bond_stereochemistry_is_preserved_by_real_graph_matching(chem):
    coordinates = [
        [-2.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    _, trans = record(chem, "F/C=C/F", coordinates)
    _, cis = record(chem, "F/C=C\\F", coordinates)
    report = compare_indexed_geometries(trans, cis, policy())
    assert report.status == "distinct" and report.graph_isomorphism == "not_isomorphic"


def test_actual_enhanced_stereo_groups_cannot_silently_lose_relative_stereo(chem):
    smiles = "C[C@H](F)[C@H](Cl)Br |&1:1,3|"
    graph = chem.AddHs(chem.MolFromSmiles(smiles))
    assert graph.GetStereoGroups()
    coordinates = [[float(index), 0.0, 0.0] for index in range(graph.GetNumAtoms())]
    with pytest.raises(PrerequisiteError, match="relative-stereo adapter"):
        record(chem, smiles, coordinates)


def test_unassigned_actual_stereo_is_retained_for_review_even_at_zero_rmsd(chem):
    _, uncertain = tetrahedron(chem, "C(F)(Cl)Br")
    assert uncertain.stereo_completeness == "unassigned"
    report = compare_indexed_geometries(uncertain, uncertain, policy())
    assert report.status == "ambiguous" and report.acceptable_mapping_count == 1


@pytest.mark.parametrize("unassigned_is_source", [True, False])
def test_missing_stereo_cannot_prove_distinctness_from_an_assigned_graph(
    chem, unassigned_is_source
):
    _, uncertain = tetrahedron(chem, "C(F)(Cl)Br")
    _, assigned = tetrahedron(chem)
    source, target = (
        (uncertain, assigned) if unassigned_is_source else (assigned, uncertain)
    )
    report = compare_indexed_geometries(source, target, policy())
    assert report.status == "ambiguous"
    assert report.graph_isomorphism == "not_established"
    assert report.disposition == "retain_all_candidates; no_automatic_exclusion"


def test_multiple_verified_symmetric_atom_mappings_are_retained(chem):
    _, original = water(chem)
    report = compare_indexed_geometries(original, original, policy())
    assert report.status == "ambiguous" and report.acceptable_mapping_count == 2
    assert report.enumeration_complete and len(report.alignments) == 2
    assert report.disposition.startswith("retain_all")


def test_bounded_graph_enumeration_cannot_claim_mapping_completeness(chem):
    _, original = water(chem)
    report = compare_indexed_geometries(original, original, policy(max_mappings=1))
    assert not report.enumeration_complete and report.status == "ambiguous"


def test_borderline_thresholds_keep_the_match_uncertain(chem):
    graph, original = tetrahedron(chem)
    changed = original.molecule.model_dump(mode="json")
    changed["geometry_bohr"][1][0] += 0.004
    target = indexed_geometry_from_rdkit(
        graph,
        Molecule.model_validate(changed),
        graph_source="declared mathematical perturbation",
    )
    report = compare_indexed_geometries(
        original,
        target,
        ComparisonPolicy(equivalent_rmsd_bohr=1e-5, distinct_rmsd_bohr=0.02),
    )
    assert report.status == "ambiguous"
    assert 1e-5 < report.alignments[0].rmsd_bohr < 0.02


def test_isotopes_cannot_be_swapped_under_an_authoritative_atom_map(chem):
    graph, original = water(chem, deuterated=True)
    isotope_indices = [
        index for index, atom in enumerate(graph.GetAtoms()) if atom.GetSymbol() == "H"
    ]
    changed_graph = chem.Mol(graph)
    for index in isotope_indices:
        changed_graph.GetAtomWithIdx(index).SetIsotope(
            0 if graph.GetAtomWithIdx(index).GetIsotope() else 2
        )
    molecule = original.molecule.model_dump(mode="json")
    molecule["isotopes"] = [
        atom.GetIsotope() or None for atom in changed_graph.GetAtoms()
    ]
    other = indexed_geometry_from_rdkit(
        changed_graph,
        Molecule.model_validate(molecule),
        graph_source="actual isotope assertion swap",
    )
    report = compare_indexed_geometries(
        original, other, policy(atom_id_policy="shared_ids_authoritative")
    )
    assert report.status == "distinct"


def test_different_connectivity_with_same_elements_does_not_pass(chem):
    coordinates = [[float(i), float((i * i) % 3), 0.0] for i in range(9)]
    # C2H6O has two genuine RDKit topologies, with all explicit atom rows present.
    _, ether = record(chem, "COC", coordinates)
    _, alcohol = record(chem, "CCO", coordinates)
    report = compare_indexed_geometries(ether, alcohol, policy())
    assert report.status == "distinct" and not report.alignments


def test_implicit_hydrogens_cannot_supply_missing_cartesian_rows(chem):
    graph = chem.MolFromSmiles("O")
    molecule = Molecule(
        symbols=["O"],
        geometry_bohr=[[0.0, 0.0, 0.0]],
        atom_ids=["oxygen"],
        charge=0,
        multiplicity=1,
    )
    with pytest.raises(ValueError, match="unrepresented hydrogens"):
        indexed_geometry_from_rdkit(
            graph, molecule, graph_source="supplied incomplete geometry"
        )


def test_missing_stable_atom_ids_are_rejected(chem):
    graph, original = water(chem)
    payload = original.molecule.model_dump(mode="json")
    payload["atom_ids"] = None
    with pytest.raises(ValueError, match="stable atom identifiers"):
        indexed_geometry_from_rdkit(
            graph,
            Molecule.model_validate(payload),
            graph_source="missing atom identities",
        )


def test_declared_graph_charge_cannot_override_electronic_state(chem):
    graph, original = water(chem)
    payload = original.molecule.model_dump(mode="json")
    payload["charge"] = 2
    with pytest.raises(ValueError, match="state"):
        indexed_geometry_from_rdkit(
            graph,
            Molecule.model_validate(payload),
            graph_source="deliberate inconsistent state",
        )


def test_comparison_result_labels_are_recomputed_when_parsed(chem):
    _, original = water(chem)
    actual = compare_indexed_geometries(original, original, policy())
    damaged = actual.model_dump(mode="json")
    damaged["status"] = "equivalent"
    with pytest.raises(ValueError, match="actual graph/geometry evidence"):
        GeometryComparison.model_validate(damaged)


def test_water_point_group_proposal_has_actions_residuals_and_tolerance_sweep(chem):
    _, original = water(chem)
    report = propose_nuclear_symmetry(
        original, SymmetryPolicy(tolerances_bohr=(1e-8, 1e-6))
    )
    assert [item.proposed_point_group for item in report.tolerance_sweep] == [
        "C2v",
        "C2v",
    ]
    assert len(report.actions) == 4 and not report.tolerance_sensitive
    assert all(
        item.action_set_closed and item.maximum_residual_bohr < 1e-14
        for item in report.tolerance_sweep
    )
    assert all(action.resolved_mass_identity == "verified" for action in report.actions)
    assert report.mass_source_status == "actual_tabulated_isotope_records"
    assert report.feasible_permutation_inversion_group.status == "unavailable"
    assert report.nuclear_spin_weights.status == "unavailable"
    assert report.tunneling_selection_rules.status == "unavailable"
    assert report.computational_subgroup.status == "unavailable"
    assert not report.unconstrained_minimum_established


def test_deuteration_changes_mass_actions_without_renaming_geometric_group(chem):
    _, original = water(chem, deuterated=True)
    report = propose_nuclear_symmetry(
        original, SymmetryPolicy(tolerances_bohr=(1e-8, 1e-6))
    )
    assert report.tolerance_sweep[0].proposed_point_group == "C2v"
    assert len(report.tolerance_sweep[0].mass_preserving_action_indices) == 2
    assert sum(action.isotope_assertions_preserved for action in report.actions) == 2
    assert {record["mass_number"] for record in report.resolved_isotope_records} >= {
        1,
        2,
        16,
    }


def test_geometry_perturbation_reports_symmetry_tolerance_sensitivity(chem):
    _, original = water(chem, distortion=0.002)
    report = propose_nuclear_symmetry(
        original, SymmetryPolicy(tolerances_bohr=(1e-7, 0.01))
    )
    assert report.tolerance_sensitive and report.needs_review
    assert report.tolerance_sweep[0].proposed_point_group == "Cs"
    assert report.tolerance_sweep[1].proposed_point_group == "C2v"


def test_linear_symmetry_actions_do_not_invent_a_finite_point_group(chem):
    _, original = record(chem, "[H][H]", [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    report = propose_nuclear_symmetry(
        original, SymmetryPolicy(tolerances_bohr=(1e-8, 1e-6))
    )
    assert report.geometry_rank == 1 and report.needs_review
    assert all(item.proposed_point_group is None for item in report.tolerance_sweep)


def test_bounded_symmetry_enumeration_is_reported_instead_of_certified(chem):
    _, original = water(chem)
    report = propose_nuclear_symmetry(
        original, SymmetryPolicy(tolerances_bohr=(1e-8, 1e-6), max_permutations=1)
    )
    assert not report.enumeration_complete and report.needs_review
    assert all(item.proposed_point_group is None for item in report.tolerance_sweep)


def test_symmetry_profile_limits_nuclear_rows_before_combinatorial_enumeration(chem):
    graph = chem.AddHs(chem.MolFromSmiles("CCCCCCCC"))
    coordinates = [[float(index), 0.0, 0.0] for index in range(graph.GetNumAtoms())]
    _, original = record(chem, "CCCCCCCC", coordinates)
    with pytest.raises(PrerequisiteError, match="at most 24 nuclear rows"):
        propose_nuclear_symmetry(original, SymmetryPolicy(tolerances_bohr=(1e-8, 1e-6)))


def test_methane_actions_are_retained_when_group_classifier_is_unsupported(chem):
    _, original = record(
        chem,
        "C",
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
    )
    report = propose_nuclear_symmetry(
        original, SymmetryPolicy(tolerances_bohr=(1e-8, 1e-6))
    )
    assert len(report.actions) == 24 and report.enumeration_complete
    assert report.needs_review and all(
        item.proposed_point_group is None for item in report.tolerance_sweep
    )


def test_constrained_geometry_is_separate_from_an_unconstrained_minimum_claim(chem):
    _, original = water(chem)
    p = SymmetryPolicy(tolerances_bohr=(1e-8, 1e-6))
    with pytest.raises(ValueError, match="explicit decision"):
        propose_nuclear_symmetry(
            original, p, optimization_constraint_status="constrained"
        )
    report = propose_nuclear_symmetry(
        original,
        p,
        optimization_constraint_status="constrained",
        constraint_evidence="declared mathematical constraint case",
    )
    assert report.optimization_constraint_status == "constrained"
    assert not report.unconstrained_minimum_established


@pytest.mark.parametrize("field", ["group", "residual", "mass", "permutation"])
def test_supplied_symmetry_labels_cannot_be_accepted_as_computed_proof(chem, field):
    _, original = water(chem)
    actual = propose_nuclear_symmetry(
        original, SymmetryPolicy(tolerances_bohr=(1e-8, 1e-6))
    )
    damaged = copy.deepcopy(actual.model_dump(mode="json"))
    if field == "group":
        damaged["tolerance_sweep"][0]["proposed_point_group"] = "Ci"
    elif field == "residual":
        damaged["actions"][0]["max_atom_residual_bohr"] = 1.0
    elif field == "mass":
        damaged["actions"][0]["resolved_mass_identity"] = "not_preserved"
    else:
        damaged["actions"][0]["source_to_target_indices"] = [0, 2, 1]
    with pytest.raises(ValueError, match="actual geometry proposals"):
        SymmetryProposal.model_validate(damaged)


@pytest.mark.parametrize(
    "tolerances", [(1e-6,), (1e-6, 1e-6), (1e-6, 1e-8), (-1.0, 1e-6)]
)
def test_invalid_symmetry_tolerance_sweeps_reject(tolerances):
    with pytest.raises(ValueError):
        SymmetryPolicy(tolerances_bohr=tolerances)


def test_invalid_comparison_threshold_order_rejects():
    with pytest.raises(ValueError, match="threshold"):
        ComparisonPolicy(equivalent_rmsd_bohr=1e-3, distinct_rmsd_bohr=1e-4)
