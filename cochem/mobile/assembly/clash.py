"""Database-radius contact heuristics and genuine graph-qualified UFF relaxation.

Radius contact checks and template rotations propose geometry, without
equilibrium-structure or quantum-calculation claims. UFF requires a supplied
molecular graph; XYZ distances cannot establish one.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from cochem.mobile.assembly.constants import get_vdw_radius_angstrom
from cochem.mobile.assembly.exceptions import (
    SingularRotationAxisError,
    StericClashDetectedError,
)
from cochem.mobile.assembly.models import StericClashReport

logger = logging.getLogger(__name__)


def _validate_coordinates(
    atomic_symbols: list[str], coordinates: np.ndarray
) -> np.ndarray:
    if np.iscomplexobj(coordinates):
        raise ValueError("Coordinates must be real finite Cartesian values.")
    array = np.asarray(coordinates, dtype=np.float64)
    if array.shape != (len(atomic_symbols), 3) or not np.isfinite(array).all():
        raise ValueError("Coordinates must match all atoms in a finite (N,3) array.")
    return array


def _eligible_pairs(
    num_atoms: int,
    ligand_slices: list[tuple[int, int]],
    donor_global_indices: set[int],
) -> list[tuple[int, int]]:
    """Return the explicitly defined contact-policy pairs, without inferring bonds."""
    atom_to_ligand = [-1] * num_atoms
    covered: set[int] = set()
    for ligand_index, (start, end) in enumerate(ligand_slices):
        if (
            type(start) is not int
            or type(end) is not int
            or not (1 <= start < end <= num_atoms)
        ):
            raise ValueError(
                "Ligand slices must be nonempty ranges excluding the metal."
            )
        indices = set(range(start, end))
        if covered & indices:
            raise ValueError("Ligand slices must not overlap.")
        covered.update(indices)
        for index in indices:
            atom_to_ligand[index] = ligand_index
    if covered != set(range(1, num_atoms)):
        raise ValueError("Ligand slices must account for every nonmetal atom.")
    if any(
        type(index) is not int or index not in covered for index in donor_global_indices
    ):
        raise ValueError("Donor indices must identify actual ligand atoms.")
    return [
        (i, j)
        for i in range(num_atoms)
        for j in range(i + 1, num_atoms)
        if (i == 0 and j not in donor_global_indices)
        or (i != 0 and atom_to_ligand[i] != atom_to_ligand[j])
    ]


def rodrigues_rotate_point(
    point: np.ndarray,
    origin: np.ndarray,
    unit_axis: np.ndarray,
    theta_rad: float,
) -> np.ndarray:
    """Rotate a 3D point around an arbitrary axis passing through origin by angle theta.

    v_rot = v*cos(theta) + (k x v)*sin(theta) + k*(k . v)*(1 - cos(theta))

    Args:
        point: 3D point to rotate.
        origin: Point on rotation axis.
        unit_axis: Unit normal vector k_hat defining rotation axis.
        theta_rad: Rotation angle in radians.

    Returns:
        Rotated 3D point.
    """
    for value in (point, origin, unit_axis):
        if (
            np.iscomplexobj(value)
            or np.asarray(value).shape != (3,)
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                "Rodrigues rotation requires finite real Cartesian vectors."
            )
    if not math.isfinite(theta_rad) or not np.isclose(
        np.linalg.norm(unit_axis), 1.0, atol=1e-12
    ):
        raise SingularRotationAxisError(
            "Rodrigues rotation requires a unit axis and finite angle."
        )
    if abs(theta_rad) < 1e-12:
        return point.copy()

    v_vec = point - origin
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    dot_prod = float(np.dot(v_vec, unit_axis))
    cross_prod = np.cross(unit_axis, v_vec)

    v_rot = (
        (v_vec * cos_t)
        + (cross_prod * sin_t)
        + (unit_axis * (dot_prod * (1.0 - cos_t)))
    )
    result: np.ndarray = np.asarray(origin + v_rot, dtype=np.float64)
    return result


def evaluate_steric_clashes(
    atomic_symbols: list[str],
    coordinates: np.ndarray,
    ligand_slices: list[tuple[int, int]],
    donor_global_indices: set[int],
    alpha_vdw: float = 0.60,
) -> tuple[bool, list[tuple[int, int]], float | None, float | None, float]:
    """Evaluate an explicit scaled Mendeleev ``vdw_radius`` contact heuristic.

    Non-bonded pairs evaluated:
        - (0, j): Metal to non-donor atom j > 0
        - (i, j): Atom i in ligand A and Atom j in ligand B (A != B)

    Args:
        atomic_symbols: List of elemental symbols for all N atoms (metal at index 0).
        coordinates: Cartesian coordinates array of shape (N, 3).
        ligand_slices: List of (start_idx, end_idx) for each ligand attachment.
        donor_global_indices: Set of global atom indices directly bonded to metal.
        alpha_vdw: Database-radius sum scale factor (default: 0.60).

    Returns:
        Tuple of:
            (clash_detected, clash_pairs, min_observed_distance,
            vdw_cutoff_at_closest_pair, total_penalty)
            A missing eligible pair has null distance and cutoff.
    """
    num_atoms = len(atomic_symbols)
    coordinates = _validate_coordinates(atomic_symbols, coordinates)
    if (
        isinstance(alpha_vdw, bool)
        or not math.isfinite(alpha_vdw)
        or not 0.1 <= alpha_vdw <= 1.0
    ):
        raise ValueError("The radius scale must be finite and within [0.1, 1.0].")
    eligible = _eligible_pairs(num_atoms, ligand_slices, donor_global_indices)
    vdw_radii = [get_vdw_radius_angstrom(sym) for sym in atomic_symbols]

    clash_pairs: list[tuple[int, int]] = []
    min_observed_distance: float | None = None
    vdw_threshold_min: float | None = None
    total_penalty = 0.0

    for i, j in eligible:
        dist = float(np.linalg.norm(coordinates[i] - coordinates[j]))
        if not math.isfinite(dist):
            raise ValueError("Contact distances must be finite.")
        cutoff = alpha_vdw * (vdw_radii[i] + vdw_radii[j])
        if min_observed_distance is None or dist < min_observed_distance:
            min_observed_distance = dist
            vdw_threshold_min = cutoff
        if dist < cutoff:
            clash_pairs.append((i, j))
            total_penalty += (cutoff - dist) ** 2

    clash_detected = len(clash_pairs) > 0
    return (
        clash_detected,
        clash_pairs,
        min_observed_distance,
        vdw_threshold_min,
        total_penalty,
    )


def perform_rodrigues_dihedral_sweep(
    atomic_symbols: list[str],
    coordinates: np.ndarray,
    ligand_slices: list[tuple[int, int]],
    monodentate_info: list[
        tuple[int, int, int]
    ],  # (ligand_idx, start_idx, donor_global_idx)
    donor_global_indices: set[int],
    alpha_vdw: float = 0.60,
) -> tuple[np.ndarray, bool]:
    """Perform deterministic 15-degree dihedral sweeps around metal-donor bonds for
    clashing ligands.

    Args:
        atomic_symbols: List of elemental symbols.
        coordinates: Mutable or input coordinates array of shape (N, 3).
        ligand_slices: Slices for all ligands.
        monodentate_info: List of (ligand_idx, start_idx, donor_global_idx) for
        monodentate ligands.
        donor_global_indices: Set of donor atom indices.
        alpha_vdw: Scaled database-radius contact tolerance factor.

    Returns:
        Tuple of (optimized_coordinates, any_rotation_applied).

    Raises:
        SingularRotationAxisError: If metal-donor vector has zero norm.
    """
    opt_coords = coordinates.copy()
    any_rotation = False

    # Sweep angles: 24 discrete steps of 15 degrees in [0, 360)
    theta_steps = [math.radians(deg) for deg in range(0, 360, 15)]

    for lig_idx, _start_idx, donor_idx in monodentate_info:
        start, end = ligand_slices[lig_idx]
        donor_pos = opt_coords[donor_idx].copy()
        metal_pos = opt_coords[0].copy()

        axis_vec = donor_pos - metal_pos
        axis_norm = float(np.linalg.norm(axis_vec))

        if axis_norm < 1e-12:
            raise SingularRotationAxisError(
                f"Singular rotation axis for monodentate ligand {lig_idx}: "
                f"||u|| = {axis_norm:.6e} < 1e-12"
            )

        unit_axis = axis_vec / axis_norm

        best_theta = 0.0
        best_penalty = float("inf")
        initial_lig_coords = opt_coords[start:end].copy()

        for theta in theta_steps:
            # Rotate all atoms in this ligand around donor_pos along unit_axis
            test_coords = opt_coords.copy()
            for local_idx, global_idx in enumerate(range(start, end)):
                test_coords[global_idx] = rodrigues_rotate_point(
                    initial_lig_coords[local_idx],
                    donor_pos,
                    unit_axis,
                    theta,
                )

            _, _, _, _, penalty = evaluate_steric_clashes(
                atomic_symbols,
                test_coords,
                ligand_slices,
                donor_global_indices,
                alpha_vdw,
            )

            if penalty < best_penalty:
                best_penalty = penalty
                best_theta = theta
                if penalty == 0.0:
                    break

        if abs(best_theta) > 1e-12:
            any_rotation = True
            for local_idx, global_idx in enumerate(range(start, end)):
                opt_coords[global_idx] = rodrigues_rotate_point(
                    initial_lig_coords[local_idx],
                    donor_pos,
                    unit_axis,
                    best_theta,
                )

    return opt_coords, any_rotation


def run_constrained_uff_relaxation(
    atomic_symbols: list[str],
    coordinates: np.ndarray,
    donor_global_indices: set[int],
    *,
    molecule: Chem.Mol | None = None,
    max_iterations: int = 500,
) -> tuple[np.ndarray, float | None]:
    """Run genuine constrained UFF using an explicit caller-supplied graph.

    Atom zero and declared donor atoms are fixed. The molecular graph must
    include every atom in the same order; explicit graph charges and bonds
    are retained. No connectivity or formal charge is inferred from XYZ.
    Missing graph returns the unchanged proposal and unavailable energy.
    An invalid/unparameterized graph or failed minimization raises with a
    reason. A returned finite energy is the converged UFF model energy at
    the returned coordinates, not a quantum energy or a certified minimum.
    """
    coordinates = _validate_coordinates(atomic_symbols, coordinates)
    if type(max_iterations) is not int or not 1 <= max_iterations <= 500:
        raise ValueError("UFF max_iterations must be an integer within [1,500].")
    if not atomic_symbols:
        raise ValueError("UFF requires at least one explicit atom.")
    if any(
        type(index) is not int or not 0 < index < len(atomic_symbols)
        for index in donor_global_indices
    ):
        raise ValueError("UFF donor indices must identify actual nonzero atom indices.")
    if molecule is None:
        logger.info("UFF unavailable: no explicit molecular graph was declared.")
        return coordinates.copy(), None
    if not isinstance(molecule, Chem.Mol):
        raise ValueError("UFF requires an actual RDKit molecular graph.")
    graph_symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    if graph_symbols != atomic_symbols:
        raise ValueError(
            "UFF graph atom order must match every declared coordinate atom."
        )
    if any(atom.GetNumImplicitHs() for atom in molecule.GetAtoms()):
        raise ValueError("UFF graph must explicitly contain all hydrogen atoms.")

    ro_mol = Chem.Mol(molecule)
    Chem.SanitizeMol(ro_mol)
    ro_mol.RemoveAllConformers()
    conf = Chem.Conformer(len(atomic_symbols))
    for index, position in enumerate(coordinates):
        conf.SetAtomPosition(index, Point3D(*(float(value) for value in position)))
    ro_mol.AddConformer(conf, assignId=True)
    if not AllChem.UFFHasAllMoleculeParams(ro_mol):
        raise ValueError(
            "UFF unavailable: the supplied graph lacks complete UFF parameters."
        )
    ff = AllChem.UFFGetMoleculeForceField(
        ro_mol, confId=0, ignoreInterfragInteractions=False
    )
    if ff is None:
        raise ValueError("UFF unavailable: RDKit did not provide a force field.")
    ff.AddFixedPoint(0)
    for donor_index in sorted(donor_global_indices):
        ff.AddFixedPoint(donor_index)
    ff.Initialize()
    optimizer_status = ff.Minimize(maxIts=max_iterations, forceTol=1e-4)
    if optimizer_status != 0:
        raise RuntimeError(
            f"UFF minimization did not converge (RDKit status {optimizer_status}); "
            "no relaxed geometry or energy is qualified."
        )
    final_energy = float(ff.CalcEnergy())
    relaxed_positions = np.asarray(
        ro_mol.GetConformer().GetPositions(), dtype=np.float64
    )
    if not math.isfinite(final_energy) or not np.isfinite(relaxed_positions).all():
        raise RuntimeError("UFF returned nonfinite coordinates or energy.")
    fixed_indices = [0, *sorted(donor_global_indices)]
    if not np.allclose(
        relaxed_positions[fixed_indices], coordinates[fixed_indices], rtol=0, atol=1e-12
    ):
        raise RuntimeError("UFF altered a declared fixed atom; result rejected.")
    return relaxed_positions, final_energy


def resolve_clashes_and_report(
    atomic_symbols: list[str],
    initial_coordinates: np.ndarray,
    ligand_slices: list[tuple[int, int]],
    monodentate_info: list[tuple[int, int, int]],
    donor_global_indices: set[int],
    alpha_vdw: float = 0.60,
) -> tuple[np.ndarray, StericClashReport, float | None]:
    """Apply the contact heuristic and rigid dihedral sweeps to a proposal.

    This interface contains no declared molecular graph. UFF energy is
    therefore unavailable, and unresolved contact conflicts fail explicitly.
    Internal distances are preserved by rigid rotations; no equilibrium
    structure or minimum is certified by this geometric procedure.
    """
    initial_coordinates = _validate_coordinates(atomic_symbols, initial_coordinates)
    initial_clash, initial_pairs, min_distance, cutoff, _ = evaluate_steric_clashes(
        atomic_symbols,
        initial_coordinates,
        ligand_slices,
        donor_global_indices,
        alpha_vdw,
    )
    current_coords = initial_coordinates.copy()
    final_clash = initial_clash
    final_pairs = list(initial_pairs)
    if initial_clash:
        current_coords, _ = perform_rodrigues_dihedral_sweep(
            atomic_symbols,
            current_coords,
            ligand_slices,
            monodentate_info,
            donor_global_indices,
            alpha_vdw,
        )
        final_clash, final_pairs, min_distance, cutoff, _ = evaluate_steric_clashes(
            atomic_symbols,
            current_coords,
            ligand_slices,
            donor_global_indices,
            alpha_vdw,
        )
        if final_clash:
            raise StericClashDetectedError(
                "Scaled database-radius contact conflicts remain "
                f"for pairs {final_pairs}; "
                "rigid dihedral sweeps could not resolve them. UFF is unavailable "
                "without an explicit molecular graph; no force-field or quantum "
                "result was substituted."
            )
    report = StericClashReport(
        evaluated_pair_count=len(
            _eligible_pairs(len(atomic_symbols), ligand_slices, donor_global_indices)
        ),
        initial_clash_detected=initial_clash,
        initial_clash_pairs=initial_pairs,
        clash_detected=final_clash,
        clash_pairs=final_pairs,
        min_observed_distance=min_distance,
        vdw_contact_threshold=cutoff,
        clash_resolved=initial_clash and not final_clash,
    )
    return current_coords, report, None
