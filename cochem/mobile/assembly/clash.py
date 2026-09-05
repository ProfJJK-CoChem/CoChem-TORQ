"""Deterministic pairwise Bondi contact clash detection, Rodrigues dihedral sweeps, and UFF relaxation.

Strict adherence to the Zero-Mock mandate with physical force field energy minimization.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Geometry import Point3D

from cochem.mobile.assembly.constants import get_vdw_radius_angstrom
from cochem.mobile.assembly.exceptions import (
    SingularRotationAxisError,
    StericClashDetectedError,
)
from cochem.mobile.assembly.models import StericClashReport

logger = logging.getLogger(__name__)


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
    if abs(theta_rad) < 1e-12:
        return point.copy()

    v_vec = point - origin
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    dot_prod = float(np.dot(v_vec, unit_axis))
    cross_prod = np.cross(unit_axis, v_vec)

    v_rot = (v_vec * cos_t) + (cross_prod * sin_t) + (unit_axis * (dot_prod * (1.0 - cos_t)))
    result: np.ndarray = np.asarray(origin + v_rot, dtype=np.float64)
    return result


def evaluate_steric_clashes(
    atomic_symbols: list[str],
    coordinates: np.ndarray,
    ligand_slices: list[tuple[int, int]],
    donor_global_indices: set[int],
    alpha_vdw: float = 0.60,
) -> tuple[bool, list[tuple[int, int]], float, float, float]:
    """Evaluate pairwise non-bonded Bondi contact distances across the assembled complex.

    Non-bonded pairs evaluated:
        - (0, j): Metal to non-donor atom j > 0
        - (i, j): Atom i in ligand A and Atom j in ligand B (A != B)

    Args:
        atomic_symbols: List of elemental symbols for all N atoms (metal at index 0).
        coordinates: Cartesian coordinates array of shape (N, 3).
        ligand_slices: List of (start_idx, end_idx) for each ligand attachment.
        donor_global_indices: Set of global atom indices directly bonded to metal.
        alpha_vdw: Scaled Bondi contact tolerance factor (default: 0.60).

    Returns:
        Tuple of:
            (clash_detected, clash_pairs, min_observed_distance, bondi_threshold_min, total_penalty)
    """
    num_atoms = len(atomic_symbols)
    vdw_radii = [get_vdw_radius_angstrom(sym) for sym in atomic_symbols]

    # Map atom index to ligand index (-1 for metal at index 0)
    atom_to_ligand = [-1] * num_atoms
    for lig_idx, (start, end) in enumerate(ligand_slices):
        for a_idx in range(start, end):
            atom_to_ligand[a_idx] = lig_idx

    clash_pairs: list[tuple[int, int]] = []
    min_observed_distance = float("inf")
    bondi_threshold_min = 0.0
    total_penalty = 0.0

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # Check if pair is non-bonded
            if i == 0:
                # Metal-ligand pair
                if j in donor_global_indices:
                    # Bonded metal-donor pair: skip
                    continue
            else:
                # Inter-ligand pair check
                lig_i = atom_to_ligand[i]
                lig_j = atom_to_ligand[j]
                if lig_i == lig_j and lig_i >= 0:
                    # Intra-ligand pair: skip
                    continue

            dist = float(np.linalg.norm(coordinates[i] - coordinates[j]))
            cutoff = alpha_vdw * (vdw_radii[i] + vdw_radii[j])

            if dist < min_observed_distance:
                min_observed_distance = dist
                bondi_threshold_min = cutoff

            if dist < cutoff:
                clash_pairs.append((i, j))
                overlap = cutoff - dist
                total_penalty += overlap**2

    if min_observed_distance == float("inf"):
        min_observed_distance = 0.0
        bondi_threshold_min = 0.0

    clash_detected = len(clash_pairs) > 0
    return (
        clash_detected,
        clash_pairs,
        min_observed_distance,
        bondi_threshold_min,
        total_penalty,
    )


def perform_rodrigues_dihedral_sweep(
    atomic_symbols: list[str],
    coordinates: np.ndarray,
    ligand_slices: list[tuple[int, int]],
    monodentate_info: list[tuple[int, int, int]],  # (ligand_idx, start_idx, donor_global_idx)
    donor_global_indices: set[int],
    alpha_vdw: float = 0.60,
) -> tuple[np.ndarray, bool]:
    """Perform deterministic 15-degree dihedral sweeps around metal-donor bonds for clashing ligands.

    Args:
        atomic_symbols: List of elemental symbols.
        coordinates: Mutable or input coordinates array of shape (N, 3).
        ligand_slices: Slices for all ligands.
        monodentate_info: List of (ligand_idx, start_idx, donor_global_idx) for monodentate ligands.
        donor_global_indices: Set of donor atom indices.
        alpha_vdw: Bondi tolerance factor.

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
                f"Singular rotation axis for monodentate ligand {lig_idx}: ||u|| = {axis_norm:.6e} < 1e-12"
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
) -> tuple[np.ndarray, float | None]:
    """Execute constrained RDKit UFF energy minimization with fixed metal and donor positions.

    Args:
        atomic_symbols: List of elemental symbols for all N atoms.
        coordinates: Initial coordinates array of shape (N, 3).
        donor_global_indices: Set of global indices for donor atoms.

    Returns:
        Tuple of (relaxed_coordinates, final_uff_energy_kcal_mol).
    """
    blocker = rdBase.BlockLogs()
    try:
        num_atoms = len(atomic_symbols)
        rw_mol = Chem.RWMol()
        for sym in atomic_symbols:
            atom = Chem.Atom(sym)
            atom.SetNoImplicit(True)
            rw_mol.AddAtom(atom)

        ro_mol = rw_mol.GetMol()
        conf = Chem.Conformer(num_atoms)
        for i in range(num_atoms):
            p = coordinates[i]
            conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
        ro_mol.AddConformer(conf, assignId=True)

        try:
            rdDetermineBonds.DetermineConnectivity(ro_mol)
        except (ValueError, RuntimeError) as exc:
            logger.debug("rdDetermineBonds could not determine connectivity: %s", exc)

        ro_mol.UpdatePropertyCache(strict=False)

        ff = AllChem.UFFGetMoleculeForceField(ro_mol, confId=0, ignoreInterfragInteractions=False)
        if ff is None:
            return coordinates.copy(), None

        # Fix metal (index 0) and all donor atoms
        ff.AddFixedPoint(0)
        for d_idx in donor_global_indices:
            ff.AddFixedPoint(d_idx)

        ff.Initialize()
        ff.Minimize(maxIts=500, forceTol=1e-4)
        final_energy = float(ff.CalcEnergy())

        relaxed_positions = ro_mol.GetConformer().GetPositions()
        return relaxed_positions, final_energy

    except Exception as exc:
        logger.warning("Constrained UFF relaxation encountered non-fatal error: %s", exc)
        return coordinates.copy(), None
    finally:
        del blocker


def resolve_clashes_and_report(
    atomic_symbols: list[str],
    initial_coordinates: np.ndarray,
    ligand_slices: list[tuple[int, int]],
    monodentate_info: list[tuple[int, int, int]],
    donor_global_indices: set[int],
    alpha_vdw: float = 0.60,
) -> tuple[np.ndarray, StericClashReport, float | None]:
    """Orchestrate clash evaluation, deterministic dihedral sweep, and UFF relaxation.

    Args:
        atomic_symbols: List of elemental symbols.
        initial_coordinates: Assembled coordinates array of shape (N, 3).
        ligand_slices: Slices for all ligands.
        monodentate_info: Monodentate metadata list.
        donor_global_indices: Set of donor atom indices.
        alpha_vdw: Bondi tolerance factor.

    Returns:
        Tuple of (final_coordinates, StericClashReport, uff_energy).

    Raises:
        StericClashDetectedError: If steric clash cannot be resolved below threshold.
    """
    # 1. Initial clash evaluation
    initial_clash, clash_pairs, min_d, bondi_thresh, penalty = evaluate_steric_clashes(
        atomic_symbols,
        initial_coordinates,
        ligand_slices,
        donor_global_indices,
        alpha_vdw,
    )

    current_coords = initial_coordinates.copy()
    uff_energy: float | None = None

    if initial_clash:
        # 2. Deterministic Rodrigues Dihedral Sweeps
        current_coords, _ = perform_rodrigues_dihedral_sweep(
            atomic_symbols,
            current_coords,
            ligand_slices,
            monodentate_info,
            donor_global_indices,
            alpha_vdw,
        )

        # Re-evaluate
        clash_detected, clash_pairs, min_d, bondi_thresh, penalty = evaluate_steric_clashes(
            atomic_symbols,
            current_coords,
            ligand_slices,
            donor_global_indices,
            alpha_vdw,
        )

        if clash_detected:
            # 3. Constrained UFF relaxation
            current_coords, uff_energy = run_constrained_uff_relaxation(
                atomic_symbols,
                current_coords,
                donor_global_indices,
            )

            # Re-evaluate
            clash_detected, clash_pairs, min_d, bondi_thresh, penalty = evaluate_steric_clashes(
                atomic_symbols,
                current_coords,
                ligand_slices,
                donor_global_indices,
                alpha_vdw,
            )

            if clash_detected:
                raise StericClashDetectedError(
                    f"Steric clash detected between atom pairs {clash_pairs} "
                    f"(min distance {min_d:.3f} A < scaled Bondi threshold {bondi_thresh:.3f} A) "
                    "that could not be resolved via dihedral sweeps or UFF relaxation."
                )
    else:
        # Calculate baseline UFF energy for the unconstrained/relaxed structure if possible
        _, uff_energy = run_constrained_uff_relaxation(
            atomic_symbols,
            current_coords,
            donor_global_indices,
        )

    report = StericClashReport(
        clash_detected=initial_clash,
        clash_pairs=clash_pairs,
        min_observed_distance=min_d,
        bondi_threshold=bondi_thresh,
        clash_resolved=True,
    )

    return current_coords, report, uff_energy
