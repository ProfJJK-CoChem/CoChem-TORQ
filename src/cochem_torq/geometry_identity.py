"""Bounded graph, proper-frame and geometric-symmetry evidence.

Supplied topology/stereochemistry assertions are never inferred from XYZ or
hashes. Results describe supplied graphs and coordinates, not minima, chemical
accuracy, feasible PI operations, nuclear-spin weights or tunneling dynamics.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictFloat, StrictInt, model_validator

from cochem_torq.domain import Contract, Molecule, PrerequisiteError, digest

Matrix = tuple[tuple[StrictFloat, StrictFloat, StrictFloat], ...]
FloatArray = NDArray[np.float64]
StereoCompleteness = Literal["supported_assignments_complete", "unassigned"]
PointGroup = Literal["C1", "Cs", "Ci", "C2", "C2v", "C2h", "D2"]
MassIdentity = Literal["verified", "not_preserved", "unavailable"]


class IndexedGeometry(Contract):
    schema_version: Literal["cochem.torq.indexed-geometry/1"] = (
        "cochem.torq.indexed-geometry/1"
    )
    molecule: Molecule
    indexed_graph_smiles: str = Field(min_length=1, max_length=100_000)
    graph_source: str = Field(min_length=1, max_length=512)
    rdkit_version: str = Field(min_length=1, max_length=80)
    topology_origin: Literal["supplied_assertion_not_inferred"] = (
        "supplied_assertion_not_inferred"
    )
    stereo_completeness: StereoCompleteness

    @model_validator(mode="after")
    def actual_indexed_graph(self) -> IndexedGeometry:
        _checked_graph(self)
        return self


class ComparisonPolicy(Contract):
    equivalent_rmsd_bohr: StrictFloat = Field(gt=0)
    distinct_rmsd_bohr: StrictFloat = Field(gt=0)
    max_mappings: StrictInt = Field(default=256, ge=1, le=10_000)
    atom_id_policy: Literal[
        "verified_graph_permutations", "shared_ids_authoritative"
    ] = "verified_graph_permutations"
    metric: Literal["unweighted_atom_rmsd_after_translation_and_SO3"] = (
        "unweighted_atom_rmsd_after_translation_and_SO3"
    )

    @model_validator(mode="after")
    def thresholds(self) -> ComparisonPolicy:
        if self.distinct_rmsd_bohr <= self.equivalent_rmsd_bohr:
            raise ValueError(
                "Distinct RMSD threshold must exceed the equivalence threshold."
            )
        return self


class ProperAlignment(Contract):
    source_to_target_indices: tuple[StrictInt, ...]
    source_to_target_atom_ids: tuple[tuple[str, str], ...]
    rotation_row_convention: Matrix
    source_centroid_bohr: tuple[StrictFloat, StrictFloat, StrictFloat]
    target_centroid_bohr: tuple[StrictFloat, StrictFloat, StrictFloat]
    translation_bohr: tuple[StrictFloat, StrictFloat, StrictFloat]
    rmsd_bohr: StrictFloat = Field(ge=0)
    max_atom_residual_bohr: StrictFloat = Field(ge=0)
    covariance_rank: StrictInt = Field(ge=0, le=3)
    orientation_status: Literal["unique_for_selected_mapping", "underdetermined"]
    transformation: Literal["target = source @ rotation + translation; SO(3)"] = (
        "target = source @ rotation + translation; SO(3)"
    )

    @model_validator(mode="after")
    def proper_rotation_and_permutation(self) -> ProperAlignment:
        count = len(self.source_to_target_indices)
        if (
            set(self.source_to_target_indices) != set(range(count))
            or len(self.source_to_target_atom_ids) != count
            or (
                self.covariance_rank < 2
                and self.orientation_status != "underdetermined"
            )
        ):
            raise ValueError(
                "Alignment requires a full permutation and truthful frame rank."
            )
        _orthogonal_matrix(self.rotation_row_convention, parity=1)
        return self


class GeometryComparison(Contract):
    schema_version: Literal["cochem.torq.geometry-comparison/1"] = (
        "cochem.torq.geometry-comparison/1"
    )
    evidence_class: Literal["supplied_graph_and_geometry_software_evidence"] = (
        "supplied_graph_and_geometry_software_evidence"
    )
    source: IndexedGeometry
    target: IndexedGeometry
    source_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy: ComparisonPolicy
    status: Literal["equivalent", "distinct", "ambiguous", "unavailable"]
    reason: str = Field(min_length=1)
    graph_isomorphism: Literal["verified", "not_isomorphic", "not_established"]
    enumeration_complete: bool
    mappings_evaluated: StrictInt = Field(ge=0)
    acceptable_mapping_count: StrictInt = Field(ge=0)
    alignments: tuple[ProperAlignment, ...]
    disposition: Literal["retain_all_candidates; no_automatic_exclusion"] = (
        "retain_all_candidates; no_automatic_exclusion"
    )
    scientific_accuracy_established: Literal[False] = False

    @model_validator(mode="after")
    def recomputed_mapping_evidence(self) -> GeometryComparison:
        expected = _comparison_data(self.source, self.target, self.policy)
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError(
                "Comparison labels/transforms differ from actual "
                "graph/geometry evidence."
            )
        return self


class UnavailableSymmetry(Contract):
    status: Literal["unavailable"] = "unavailable"
    absence_kind: Literal["unsupported", "not_computed"]
    reason: str = Field(min_length=1)


class SymmetryPolicy(Contract):
    tolerances_bohr: tuple[StrictFloat, ...] = Field(min_length=2, max_length=16)
    max_permutations: StrictInt = Field(default=256, ge=1, le=512)
    action_closure_tolerance: StrictFloat = Field(default=1e-7, gt=0, le=1e-2)
    mass_relative_tolerance: StrictFloat = Field(default=1e-12, gt=0, le=1e-4)

    @model_validator(mode="after")
    def tolerance_sweep(self) -> SymmetryPolicy:
        if (
            any(value <= 0 for value in self.tolerances_bohr)
            or tuple(sorted(set(self.tolerances_bohr))) != self.tolerances_bohr
        ):
            raise ValueError(
                "Symmetry tolerances must be positive, distinct and increasing."
            )
        return self


class GeometricAction(Contract):
    source_to_target_indices: tuple[StrictInt, ...]
    source_to_target_atom_ids: tuple[tuple[str, str], ...]
    rotation_row_convention: Matrix
    parity: Literal[1, -1]
    rmsd_bohr: StrictFloat = Field(ge=0)
    max_atom_residual_bohr: StrictFloat = Field(ge=0)
    isotope_assertions_preserved: bool
    resolved_mass_identity: MassIdentity
    action_origin: Literal["least_squares_nuclear_position_proposal"] = (
        "least_squares_nuclear_position_proposal"
    )

    @model_validator(mode="after")
    def actual_orthogonal_action(self) -> GeometricAction:
        count = len(self.source_to_target_indices)
        if (
            set(self.source_to_target_indices) != set(range(count))
            or len(self.source_to_target_atom_ids) != count
        ):
            raise ValueError("A symmetry action requires a full nuclear permutation.")
        _orthogonal_matrix(self.rotation_row_convention, parity=self.parity)
        return self


class ToleranceObservation(Contract):
    tolerance_bohr: StrictFloat = Field(gt=0)
    accepted_action_indices: tuple[StrictInt, ...]
    mass_preserving_action_indices: tuple[StrictInt, ...]
    maximum_residual_bohr: StrictFloat | None = Field(default=None, ge=0)
    closure_residual: StrictFloat | None = Field(default=None, ge=0)
    action_set_closed: bool
    proposed_point_group: PointGroup | None
    classification_status: Literal["bounded_proposal", "unassigned"]


class SymmetryProposal(Contract):
    schema_version: Literal["cochem.torq.symmetry-proposal/1"] = (
        "cochem.torq.symmetry-proposal/1"
    )
    geometry: IndexedGeometry
    geometry_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_class: Literal["nuclear_geometry_software_evidence"] = (
        "nuclear_geometry_software_evidence"
    )
    policy: SymmetryPolicy
    geometry_centroid_bohr: tuple[StrictFloat, StrictFloat, StrictFloat]
    geometry_rank: StrictInt = Field(ge=0, le=3)
    enumeration_complete: bool
    actions: tuple[GeometricAction, ...]
    tolerance_sweep: tuple[ToleranceObservation, ...]
    tolerance_sensitive: bool
    resolved_isotope_records: tuple[dict[str, Any], ...]
    mass_source_status: Literal["actual_tabulated_isotope_records", "unavailable"]
    computational_subgroup: UnavailableSymmetry
    feasible_permutation_inversion_group: UnavailableSymmetry
    nuclear_spin_weights: UnavailableSymmetry
    tunneling_selection_rules: UnavailableSymmetry
    optimization_constraint_status: Literal[
        "unconstrained", "constrained", "not_computed"
    ]
    constraint_evidence: str | None
    unconstrained_minimum_established: Literal[False] = False
    needs_review: bool
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def recomputed_geometric_actions(self) -> SymmetryProposal:
        expected = _symmetry_data(
            self.geometry,
            self.policy,
            optimization_constraint_status=self.optimization_constraint_status,
            constraint_evidence=self.constraint_evidence,
        )
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError(
                "Symmetry labels/actions/residuals differ from "
                "actual geometry proposals."
            )
        return self


def _rdkit() -> Any:
    try:
        from rdkit import Chem
    except ImportError as error:
        raise PrerequisiteError(
            "Verified graph mappings require the real RDKit adapter."
        ) from error
    return Chem


def _supported_stereo(mol: Any) -> StereoCompleteness:
    chem = _rdkit()
    if mol.GetStereoGroups():
        raise PrerequisiteError(
            "Enhanced stereo groups require a separate relative-stereo adapter."
        )
    supported_atoms = {
        chem.ChiralType.CHI_UNSPECIFIED,
        chem.ChiralType.CHI_TETRAHEDRAL_CW,
        chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    if any(atom.GetChiralTag() not in supported_atoms for atom in mol.GetAtoms()):
        raise PrerequisiteError(
            "Only asserted tetrahedral atom stereochemistry is supported."
        )
    supported_bonds = {
        chem.BondStereo.STEREONONE,
        chem.BondStereo.STEREOANY,
        chem.BondStereo.STEREOE,
        chem.BondStereo.STEREOZ,
        chem.BondStereo.STEREOCIS,
        chem.BondStereo.STEREOTRANS,
    }
    if any(bond.GetStereo() not in supported_bonds for bond in mol.GetBonds()):
        raise PrerequisiteError(
            "The asserted bond stereochemistry has no supported adapter."
        )
    allowed_types = {chem.StereoType.Atom_Tetrahedral, chem.StereoType.Bond_Double}
    potentials = chem.FindPotentialStereo(mol)
    if any(item.type not in allowed_types for item in potentials):
        raise PrerequisiteError(
            "Non-tetrahedral/atropisomer stereo requires a separate adapter."
        )
    return (
        "unassigned"
        if any(item.specified != chem.StereoSpecified.Specified for item in potentials)
        else "supported_assignments_complete"
    )


def _checked_graph(record: IndexedGeometry) -> Any:
    chem = _rdkit()
    import rdkit

    if record.molecule.atom_ids is None:
        raise ValueError("Geometry mapping requires explicit stable atom identifiers.")
    if rdkit.__version__ != record.rdkit_version:
        raise PrerequisiteError(
            "Revalidate the indexed graph under its recorded RDKit version."
        )
    parameters = chem.SmilesParserParams()
    parameters.removeHs = False
    parameters.sanitize = True
    mol = chem.MolFromSmiles(record.indexed_graph_smiles, parameters)
    if mol is None or mol.GetNumAtoms() != len(record.molecule.symbols):
        raise ValueError(
            "The supplied graph must contain every explicit geometry atom."
        )
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if set(maps) != set(range(1, len(maps) + 1)):
        raise ValueError(
            "Indexed graph atom maps must identify exactly every original row."
        )
    order = [maps.index(index) for index in range(1, len(maps) + 1)]
    mol = chem.RenumberAtoms(mol, order)
    isotopes = record.molecule.isotopes or [None] * len(maps)
    if (
        any(
            atom.GetSymbol() != record.molecule.symbols[index]
            or (atom.GetIsotope() or None) != isotopes[index]
            or atom.GetNumImplicitHs() != 0
            or atom.GetNumExplicitHs() != 0
            for index, atom in enumerate(mol.GetAtoms())
        )
        or sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        != record.molecule.charge
    ):
        raise ValueError(
            "Graph symbols/isotopes/state or unrepresented hydrogens "
            "differ from geometry."
        )
    chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    if _supported_stereo(mol) != record.stereo_completeness:
        raise ValueError(
            "Declared stereochemistry completeness differs from the actual graph."
        )
    if (
        chem.MolToSmiles(mol, canonical=True, allHsExplicit=True, allBondsExplicit=True)
        != record.indexed_graph_smiles
    ):
        raise ValueError(
            "Indexed graph must preserve a canonical "
            "stereo/isotope-explicit round trip."
        )
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def indexed_geometry_from_rdkit(
    mol: Any, molecule: Molecule, *, graph_source: str
) -> IndexedGeometry:
    """Capture supplied explicit RDKit topology while retaining original row IDs."""
    chem = _rdkit()
    import rdkit

    if not isinstance(mol, chem.Mol):
        raise ValueError("Provide a genuine RDKit molecular graph.")
    clone = chem.Mol(mol)
    if clone.GetNumAtoms() != len(molecule.symbols):
        raise ValueError(
            "RDKit atom rows must already match the complete supplied geometry."
        )
    if molecule.atom_ids is None:
        raise ValueError("Geometry mapping requires explicit stable atom identifiers.")
    declared_chirality = [atom.GetChiralTag() for atom in clone.GetAtoms()]
    chem.AssignStereochemistry(clone, cleanIt=True, force=True)
    if any(
        before != chem.ChiralType.CHI_UNSPECIFIED
        and atom.GetChiralTag() == chem.ChiralType.CHI_UNSPECIFIED
        for before, atom in zip(declared_chirality, clone.GetAtoms(), strict=True)
    ):
        raise ValueError(
            "The supplied atom stereochemistry cannot be retained "
            "by the supported graph adapter."
        )
    completeness = _supported_stereo(clone)
    for index, atom in enumerate(clone.GetAtoms(), start=1):
        atom.SetAtomMapNum(index)
    smiles = chem.MolToSmiles(
        clone, canonical=True, allHsExplicit=True, allBondsExplicit=True
    )
    return IndexedGeometry(
        molecule=molecule,
        indexed_graph_smiles=smiles,
        graph_source=graph_source,
        rdkit_version=rdkit.__version__,
        stereo_completeness=completeness,
    )


def _atom_signature(atom: Any) -> tuple[Any, ...]:
    return (
        atom.GetAtomicNum(),
        atom.GetIsotope(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        atom.GetIsAromatic(),
        atom.GetChiralTag() != _rdkit().ChiralType.CHI_UNSPECIFIED,
        atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
    )


def _verified_graph_mapping(source: Any, target: Any, mapping: tuple[int, ...]) -> bool:
    if (
        set(mapping) != set(range(source.GetNumAtoms()))
        or source.GetNumBonds() != target.GetNumBonds()
    ):
        return False
    if any(
        _atom_signature(atom) != _atom_signature(target.GetAtomWithIdx(mapping[index]))
        for index, atom in enumerate(source.GetAtoms())
    ):
        return False
    for bond in source.GetBonds():
        other = target.GetBondBetweenAtoms(
            mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
        )
        if other is None or (
            bond.GetBondType(),
            bond.GetIsAromatic(),
            str(bond.GetStereo()),
        ) != (other.GetBondType(), other.GetIsAromatic(), str(other.GetStereo())):
            return False
    return True


def _matrix_tuple(matrix: FloatArray) -> Matrix:
    return tuple((float(row[0]), float(row[1]), float(row[2])) for row in matrix)


def _orthogonal_matrix(matrix: Matrix, *, parity: int) -> FloatArray:
    array = np.asarray(matrix, dtype=float)
    if (
        array.shape != (3, 3)
        or not np.isfinite(array).all()
        or not np.allclose(array.T @ array, np.eye(3), rtol=0, atol=1e-10)
        or not math.isclose(
            float(np.linalg.det(array)), parity, rel_tol=0, abs_tol=1e-10
        )
    ):
        raise ValueError("An action must have the declared orthogonal parity.")
    return array


def _fit_action(
    source: FloatArray, target: FloatArray, parity: int
) -> tuple[FloatArray, float, float, int, bool]:
    covariance = source.T @ target
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = parity * float(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    residual = np.linalg.norm(source @ rotation - target, axis=1)
    rank = int(np.linalg.matrix_rank(covariance))
    unconstrained_parity = float(np.linalg.det(u @ vt))
    gap_tolerance = float(singular[0]) * 3 * np.finfo(float).eps
    isolated = bool(
        rank >= 2
        and (
            parity * unconstrained_parity > 0
            or float(singular[1] - singular[2]) > gap_tolerance
        )
    )
    return (
        rotation,
        float(np.sqrt(np.mean(residual**2))),
        float(np.max(residual)),
        rank,
        isolated,
    )


def compare_indexed_geometries(
    source: IndexedGeometry, target: IndexedGeometry, policy: ComparisonPolicy
) -> GeometryComparison:
    return GeometryComparison(**_comparison_data(source, target, policy))


def _comparison_data(
    source: IndexedGeometry, target: IndexedGeometry, policy: ComparisonPolicy
) -> dict[str, Any]:
    source = IndexedGeometry.model_validate(source.model_dump(mode="json"))
    target = IndexedGeometry.model_validate(target.model_dump(mode="json"))
    policy = ComparisonPolicy.model_validate(policy.model_dump(mode="json"))
    original, other = _checked_graph(source), _checked_graph(target)
    base = {
        "source": source,
        "target": target,
        "source_record_sha256": digest(source.model_dump(mode="json")),
        "target_record_sha256": digest(target.model_dump(mode="json")),
        "policy": policy,
    }
    if len(source.molecule.symbols) != len(target.molecule.symbols) or (
        source.molecule.charge,
        source.molecule.multiplicity,
    ) != (target.molecule.charge, target.molecule.multiplicity):
        return {
            **base,
            "status": "distinct",
            "reason": "Atom count/electronic state differs.",
            "graph_isomorphism": "not_isomorphic",
            "enumeration_complete": True,
            "mappings_evaluated": 0,
            "acceptable_mapping_count": 0,
            "alignments": (),
        }
    candidates = other.GetSubstructMatches(
        original, uniquify=False, useChirality=True, maxMatches=policy.max_mappings + 1
    )
    complete = len(candidates) <= policy.max_mappings
    source_ids = source.molecule.atom_ids
    target_ids = target.molecule.atom_ids
    if source_ids is None or target_ids is None:
        raise ValueError("Verified alignment requires stable atom IDs.")
    mappings = []
    for raw in candidates[: policy.max_mappings]:
        mapping = tuple(int(index) for index in raw)
        if _verified_graph_mapping(original, other, mapping) and (
            policy.atom_id_policy != "shared_ids_authoritative"
            or all(
                source_ids[index] == target_ids[other_index]
                for index, other_index in enumerate(mapping)
            )
        ):
            mappings.append(mapping)
    coordinates = np.asarray(source.molecule.geometry_bohr, dtype=float)
    target_coordinates = np.asarray(target.molecule.geometry_bohr, dtype=float)
    source_center, target_center = (
        coordinates.mean(axis=0),
        target_coordinates.mean(axis=0),
    )
    alignments = []
    for mapping in mappings:
        rotation, rmsd, maximum, rank, isolated = _fit_action(
            coordinates - source_center,
            (target_coordinates - target_center)[list(mapping)],
            1,
        )
        alignments.append(
            ProperAlignment(
                source_to_target_indices=mapping,
                source_to_target_atom_ids=tuple(
                    (source_ids[index], target_ids[j])
                    for index, j in enumerate(mapping)
                ),
                rotation_row_convention=_matrix_tuple(rotation),
                source_centroid_bohr=tuple(source_center),
                target_centroid_bohr=tuple(target_center),
                translation_bohr=tuple(target_center - source_center @ rotation),
                rmsd_bohr=rmsd,
                max_atom_residual_bohr=maximum,
                covariance_rank=rank,
                orientation_status="underdetermined"
                if not isolated
                else "unique_for_selected_mapping",
            )
        )
    alignments.sort(
        key=lambda record: (record.rmsd_bohr, record.source_to_target_indices)
    )
    acceptable = sum(
        record.rmsd_bohr <= policy.equivalent_rmsd_bohr for record in alignments
    )
    incomplete_stereo = "unassigned" in {
        source.stereo_completeness,
        target.stereo_completeness,
    }
    if not alignments:
        if incomplete_stereo:
            status, reason = (
                "ambiguous",
                "Missing stereochemical assignments cannot prove the absence "
                "of a full stereo-preserving mapping; candidates retained.",
            )
        elif complete:
            status, reason = (
                "distinct",
                "No full stereo/isotope/connectivity-preserving mapping exists.",
            )
        else:
            status, reason = (
                "ambiguous",
                "Bounded graph enumeration ended before mapping absence "
                "could be established.",
            )
    elif not complete or incomplete_stereo or acceptable > 1:
        status, reason = (
            "ambiguous",
            "Enumeration/stereochemistry/multiple atom mappings require review; "
            "candidates retained.",
        )
    elif alignments[0].rmsd_bohr <= policy.equivalent_rmsd_bohr:
        status, reason = (
            "equivalent",
            "Verified full graph mapping and proper-frame RMSD "
            "pass the declared threshold.",
        )
    elif alignments[0].rmsd_bohr >= policy.distinct_rmsd_bohr:
        status, reason = (
            "distinct",
            "Every enumerated verified mapping exceeds "
            "the declared distinct threshold.",
        )
    else:
        status, reason = (
            "ambiguous",
            "Best proper-frame RMSD lies between the declared thresholds.",
        )
    return {
        **base,
        "status": status,
        "reason": reason,
        "graph_isomorphism": "verified"
        if alignments
        else "not_isomorphic"
        if complete and not incomplete_stereo
        else "not_established",
        "enumeration_complete": complete,
        "mappings_evaluated": len(alignments),
        "acceptable_mapping_count": acceptable,
        "alignments": tuple(alignments),
    }


def _action_closure(actions: Sequence[GeometricAction]) -> float | None:
    if not actions:
        return None
    maximum = 0.0
    indexed: dict[tuple[int, ...], list[GeometricAction]] = {}
    for action in actions:
        indexed.setdefault(action.source_to_target_indices, []).append(action)
    for left, right in itertools.product(actions, repeat=2):
        # Source index i first moves to left[i], then to right[left[i]].
        composition = tuple(
            right.source_to_target_indices[j] for j in left.source_to_target_indices
        )
        matrix = np.asarray(left.rotation_row_convention) @ np.asarray(
            right.rotation_row_convention
        )
        candidates = indexed.get(composition, [])
        if not candidates:
            return None
        distance = min(
            float(
                np.max(np.abs(matrix - np.asarray(candidate.rotation_row_convention)))
            )
            for candidate in candidates
        )
        maximum = max(maximum, distance)
    return maximum


def _bounded_group(
    actions: Sequence[GeometricAction],
    *,
    complete: bool,
    rank: int,
    closed: bool,
    tolerance: float,
) -> PointGroup | None:
    if not complete or rank < 2 or not closed:
        return None
    matrices = [np.asarray(action.rotation_row_convention) for action in actions]
    identities = sum(
        np.allclose(matrix, np.eye(3), rtol=0, atol=tolerance) for matrix in matrices
    )
    proper = [
        matrix
        for action, matrix in zip(actions, matrices, strict=True)
        if action.parity == 1
    ]
    improper = [
        matrix
        for action, matrix in zip(actions, matrices, strict=True)
        if action.parity == -1
    ]
    inversions = sum(
        np.allclose(matrix, -np.eye(3), rtol=0, atol=tolerance) for matrix in improper
    )
    reflections = sum(
        abs(float(np.trace(matrix)) - 1.0) <= tolerance for matrix in improper
    )
    half_turns = sum(
        abs(float(np.trace(matrix)) + 1.0) <= tolerance for matrix in proper
    )
    if identities != 1:
        return None
    if len(actions) == 1:
        return "C1"
    if len(actions) == 2:
        return (
            "Ci"
            if inversions == 1
            else "Cs"
            if reflections == 1
            else "C2"
            if half_turns == 1
            else None
        )
    if len(actions) == 4 and half_turns == 1 and len(proper) == 2:
        return (
            "C2v"
            if reflections == 2
            else "C2h"
            if reflections == 1 and inversions == 1
            else None
        )
    if len(actions) == 4 and len(proper) == 4 and half_turns == 3:
        return "D2"
    return None


def propose_nuclear_symmetry(
    geometry: IndexedGeometry,
    policy: SymmetryPolicy,
    *,
    optimization_constraint_status: Literal[
        "unconstrained", "constrained", "not_computed"
    ] = "not_computed",
    constraint_evidence: str | None = None,
) -> SymmetryProposal:
    return SymmetryProposal(
        **_symmetry_data(
            geometry,
            policy,
            optimization_constraint_status=optimization_constraint_status,
            constraint_evidence=constraint_evidence,
        )
    )


def _symmetry_data(
    geometry: IndexedGeometry,
    policy: SymmetryPolicy,
    *,
    optimization_constraint_status: Literal[
        "unconstrained", "constrained", "not_computed"
    ],
    constraint_evidence: str | None,
) -> dict[str, Any]:
    geometry = IndexedGeometry.model_validate(geometry.model_dump(mode="json"))
    policy = SymmetryPolicy.model_validate(policy.model_dump(mode="json"))
    if len(geometry.molecule.symbols) > 24:
        raise PrerequisiteError(
            "The bounded symmetry proposal profile supports at most 24 nuclear rows."
        )
    if optimization_constraint_status == "constrained" and not constraint_evidence:
        raise ValueError(
            "A constrained geometry requires its explicit decision/evidence reference."
        )
    coordinates = np.asarray(geometry.molecule.geometry_bohr, dtype=float)
    center = coordinates.mean(axis=0)
    centered = coordinates - center
    rank = int(np.linalg.matrix_rank(centered))
    graph = nx.Graph()
    for index, symbol in enumerate(geometry.molecule.symbols):
        graph.add_node(index, element=symbol)
    for left, right in itertools.combinations(range(len(centered)), 2):
        graph.add_edge(
            left,
            right,
            distance=float(np.linalg.norm(centered[left] - centered[right])),
        )
    maximum_tolerance = policy.tolerances_bohr[-1]
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph,
        graph,
        node_match=lambda left, right: left["element"] == right["element"],
        edge_match=lambda left, right: (
            abs(left["distance"] - right["distance"]) <= 2 * maximum_tolerance
        ),
    )
    permutations = list(
        itertools.islice(matcher.isomorphisms_iter(), policy.max_permutations + 1)
    )
    complete = len(permutations) <= policy.max_permutations
    isotope_numbers = geometry.molecule.isotopes or [None] * len(centered)
    records: tuple[dict[str, Any], ...] = ()
    masses = None
    mass_status = "unavailable"
    try:
        from Libraries.cochem_isotopes import isotope_record

        records = tuple(
            isotope_record(f"{number}{symbol}" if number is not None else symbol)
            for symbol, number in zip(
                geometry.molecule.symbols, isotope_numbers, strict=True
            )
        )
        masses = np.asarray([record["mass_u"] for record in records], dtype=float)
        if not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError(
                "Actual isotope records must contain finite positive masses."
            )
        mass_status = "actual_tabulated_isotope_records"
    except (ImportError, ValueError, RuntimeError):
        # No mass-based action is accepted if the actual isotope service fails.
        records, masses = (), None
    atom_ids = geometry.molecule.atom_ids
    if atom_ids is None:
        raise ValueError("Symmetry proposals require stable atom identities.")
    actions = []
    for proposed in permutations[: policy.max_permutations]:
        permutation = tuple(int(proposed[index]) for index in range(len(centered)))
        for parity in (1, -1):
            rotation, rmsd, residual, _, _isolated = _fit_action(
                centered, centered[list(permutation)], parity
            )
            if residual > maximum_tolerance:
                continue
            mass_identity: MassIdentity = (
                "unavailable"
                if masses is None
                else "verified"
                if np.allclose(
                    masses,
                    masses[list(permutation)],
                    rtol=policy.mass_relative_tolerance,
                    atol=0,
                )
                else "not_preserved"
            )
            actions.append(
                GeometricAction(
                    source_to_target_indices=permutation,
                    source_to_target_atom_ids=tuple(
                        (atom_ids[index], atom_ids[j])
                        for index, j in enumerate(permutation)
                    ),
                    rotation_row_convention=_matrix_tuple(rotation),
                    parity=parity,
                    rmsd_bohr=rmsd,
                    max_atom_residual_bohr=residual,
                    isotope_assertions_preserved=all(
                        isotope_numbers[i] == isotope_numbers[j]
                        for i, j in enumerate(permutation)
                    ),
                    resolved_mass_identity=mass_identity,
                )
            )
    observations = []
    for tolerance in policy.tolerances_bohr:
        indices = tuple(
            index
            for index, action in enumerate(actions)
            if action.max_atom_residual_bohr <= tolerance
        )
        selected = [actions[index] for index in indices]
        closure = _action_closure(selected)
        closed = closure is not None and closure <= policy.action_closure_tolerance
        group = _bounded_group(
            selected,
            complete=complete,
            rank=rank,
            closed=closed,
            tolerance=policy.action_closure_tolerance,
        )
        observations.append(
            ToleranceObservation(
                tolerance_bohr=tolerance,
                accepted_action_indices=indices,
                mass_preserving_action_indices=tuple(
                    index
                    for index in indices
                    if actions[index].resolved_mass_identity == "verified"
                ),
                maximum_residual_bohr=max(
                    (action.max_atom_residual_bohr for action in selected), default=None
                ),
                closure_residual=closure,
                action_set_closed=closed,
                proposed_point_group=group,
                classification_status="bounded_proposal"
                if group is not None
                else "unassigned",
            )
        )
    sensitivity = (
        len({observation.accepted_action_indices for observation in observations}) > 1
    )
    return {
        "geometry": geometry,
        "geometry_record_sha256": digest(geometry.model_dump(mode="json")),
        "policy": policy,
        "geometry_centroid_bohr": tuple(center),
        "geometry_rank": rank,
        "enumeration_complete": complete,
        "actions": tuple(actions),
        "tolerance_sweep": tuple(observations),
        "tolerance_sensitive": sensitivity,
        "resolved_isotope_records": records,
        "mass_source_status": mass_status,
        "computational_subgroup": UnavailableSymmetry(
            absence_kind="not_computed",
            reason=(
                "No authentic engine subgroup parser/evidence was supplied; "
                "geometry cannot identify an engine's applied subgroup."
            ),
        ),
        "feasible_permutation_inversion_group": UnavailableSymmetry(
            absence_kind="unsupported",
            reason=(
                "Nuclear position actions do not establish dynamically feasible "
                "permutations/inversions; a validated nuclear-motion model is required."
            ),
        ),
        "nuclear_spin_weights": UnavailableSymmetry(
            absence_kind="unsupported",
            reason=(
                "No authenticated nuclear-spin data and feasible PI "
                "representations were supplied."
            ),
        ),
        "tunneling_selection_rules": UnavailableSymmetry(
            absence_kind="unsupported",
            reason=(
                "No qualified tunneling Hamiltonian or transition "
                "representation was supplied."
            ),
        ),
        "optimization_constraint_status": optimization_constraint_status,
        "constraint_evidence": constraint_evidence,
        "needs_review": not complete
        or rank < 2
        or sensitivity
        or any(o.proposed_point_group is None for o in observations),
        "limitations": (
            "Geometric proposals describe supplied nuclear positions only; "
            "no stationary/minimum or accuracy claim.",
            "C1/Cs/Ci/C2/C2v/C2h/D2 classification is bounded; "
            "other finite and continuous groups remain unassigned.",
            "Molecular geometry, mass/isotope actions, engine subgroup "
            "and feasible PI symmetry remain separate.",
        ),
    }
