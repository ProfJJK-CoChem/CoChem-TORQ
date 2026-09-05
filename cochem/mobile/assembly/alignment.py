"""Geometric alignment, Kabsch SO(3) transformation, and distance bounds verification.

Strict adherence to the Zero-Mock mandate with physical distance calculations and proper SO(3) rotations.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from cochem.mobile.assembly.constants import (
    get_covalent_radius_angstrom,
    get_standard_atomic_weight,
)
from cochem.mobile.assembly.exceptions import (
    KabschReflectionError,
    UnphysicalMonomerSeparationError,
)
from cochem.mobile.assembly.models import LigandAttachment

logger = logging.getLogger(__name__)


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Calculate 3x3 proper rotation matrix that rotates unit vector vec1 to unit vector vec2.

    Args:
        vec1: Source 3D vector.
        vec2: Target 3D vector.

    Returns:
        3x3 rotation matrix R in SO(3).
    """
    n1 = float(np.linalg.norm(vec1))
    n2 = float(np.linalg.norm(vec2))
    if n1 < 1e-12 or n2 < 1e-12:
        return np.eye(3, dtype=np.float64)

    a = vec1 / n1
    b = vec2 / n2
    c = float(np.dot(a, b))

    if c > 0.9999999999:
        return np.eye(3, dtype=np.float64)
    if c < -0.9999999999:
        # 180-degree rotation around any perpendicular axis
        orth = (
            np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(a[0]) < 0.9
            else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        )
        axis = np.cross(a, orth)
        axis = axis / np.linalg.norm(axis)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    kmat = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )

    r_mat = np.eye(3, dtype=np.float64) + kmat + (kmat @ kmat) * ((1.0 - c) / (s**2))
    return r_mat


def kabsch_fit_proper(
    p_coords: np.ndarray, q_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform Kabsch alignment with strict SO(3) sign-correction to prevent improper reflections.

    Args:
        p_coords: Mobile donor coordinates of shape (M, 3).
        q_coords: Target positions of shape (M, 3).

    Returns:
        Tuple of (R, p_centroid, q_centroid) where R is a 3x3 proper rotation matrix.

    Raises:
        KabschReflectionError: If proper SO(3) rotation cannot be achieved.
    """
    p_cent = np.mean(p_coords, axis=0)
    q_cent = np.mean(q_coords, axis=0)

    p_centered = p_coords - p_cent
    q_centered = q_coords - q_cent

    h_mat = p_centered.T @ q_centered
    u, _, vt = np.linalg.svd(h_mat)
    v = vt.T

    # Compute sign of determinant
    d_sign = np.linalg.det(v @ u.T)
    d_mat = np.diag([1.0, 1.0, 1.0 if d_sign >= 0 else -1.0])

    r_mat = v @ d_mat @ u.T
    det_r = float(np.linalg.det(r_mat))

    if abs(det_r - 1.0) > 1e-5:
        raise KabschReflectionError(
            f"Kabsch alignment failed proper SO(3) rotation constraint: det(R) = {det_r:.6f}"
        )

    return r_mat, p_cent, q_cent


def calculate_metal_donor_distance(metal_symbol: str, donor_symbol: str) -> float:
    """Calculate dynamic equilibrium metal-donor bond distance in Angstroms.

    R_ML = r_cov(Metal) + r_cov(Donor)

    Args:
        metal_symbol: Transition metal element symbol.
        donor_symbol: Donor atom element symbol.

    Returns:
        Equilibrium bond length in Angstroms.
    """
    r_metal = get_covalent_radius_angstrom(metal_symbol)
    r_donor = get_covalent_radius_angstrom(donor_symbol)
    return float(r_metal + r_donor)


def align_ligand_to_template(
    ligand: LigandAttachment,
    metal_symbol: str,
    template_vectors: np.ndarray,
) -> np.ndarray:
    """Rigidly align a ligand attachment to the specified coordination template vector slots.

    Args:
        ligand: LigandAttachment specification with 3D conformer coordinates.
        metal_symbol: Central transition metal symbol.
        template_vectors: Normalized coordination template vectors of shape (CN, 3).

    Returns:
        Aligned coordinates of shape (N_atoms, 3) in Angstroms.

    Raises:
        KabschReflectionError: If multidentate alignment produces improper reflection.
        UnphysicalMonomerSeparationError: If COM distance is outside [1.5, 12.0] Angstroms.
    """
    coords = np.array(ligand.coordinates, dtype=np.float64)
    num_atoms = len(coords)

    # Compute target donor positions
    target_positions = []
    for slot_idx, donor_idx in zip(
        ligand.target_vector_slots, ligand.donor_atom_indices, strict=True
    ):
        slot_vec = template_vectors[slot_idx]
        donor_sym = ligand.atomic_symbols[donor_idx]
        r_ml = calculate_metal_donor_distance(metal_symbol, donor_sym)
        target_pos = r_ml * slot_vec
        target_positions.append(target_pos)

    target_positions_arr = np.array(target_positions, dtype=np.float64)

    if ligand.denticity == 1:
        # Monodentate alignment
        donor_idx = ligand.donor_atom_indices[0]
        target_pos = target_positions_arr[0]
        slot_vec = template_vectors[ligand.target_vector_slots[0]]

        # Translate donor atom to origin
        p_donor = coords[donor_idx].copy()
        coords_rel = coords - p_donor

        if num_atoms == 1:
            # Monatomic ligand
            aligned_coords = coords_rel + target_pos
        else:
            # Compute centroid of non-donor atoms relative to donor
            non_donor_indices = [i for i in range(num_atoms) if i != donor_idx]
            c_others = np.mean(coords_rel[non_donor_indices], axis=0)
            norm_c = float(np.linalg.norm(c_others))

            if norm_c < 1e-12:
                # Symmetric or linear offset fallback
                u_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                u_dir = c_others / norm_c

            # Rotate ligand so non-donor atoms point outward along +slot_vec
            r_mat = rotation_matrix_from_vectors(u_dir, slot_vec)
            aligned_coords = (coords_rel @ r_mat.T) + target_pos

    else:
        # Multidentate chelator alignment via Kabsch algorithm
        donor_coords = coords[ligand.donor_atom_indices]
        r_mat, p_cent, q_cent = kabsch_fit_proper(donor_coords, target_positions_arr)

        # Apply rotation and translation
        aligned_coords = ((coords - p_cent) @ r_mat.T) + q_cent

        # Orient non-donor backbone atoms outward away from metal origin (0, 0, 0)
        non_donor_indices = [i for i in range(num_atoms) if i not in ligand.donor_atom_indices]
        if non_donor_indices and len(ligand.donor_atom_indices) == 2:
            norm_q = float(np.linalg.norm(q_cent))
            if norm_q > 1e-6:
                q_unit = q_cent / norm_q
                d_axis = target_positions_arr[1] - target_positions_arr[0]
                norm_axis = float(np.linalg.norm(d_axis))
                if norm_axis > 1e-6:
                    axis_unit = d_axis / norm_axis
                    best_theta = 0.0
                    best_proj = -float("inf")
                    # Sweep angles around donor axis to maximize backbone projection along q_unit
                    for deg in range(0, 360, 5):
                        theta_rad = math.radians(deg)
                        cos_t = math.cos(theta_rad)
                        sin_t = math.sin(theta_rad)
                        # Compute rotated backbone centroid relative to q_cent
                        c_back_rot_sum = np.zeros(3, dtype=np.float64)
                        for idx in non_donor_indices:
                            v_vec = aligned_coords[idx] - q_cent
                            dot_val = float(np.dot(v_vec, axis_unit))
                            cross_val = np.cross(axis_unit, v_vec)
                            v_rot = (
                                (v_vec * cos_t)
                                + (cross_val * sin_t)
                                + (axis_unit * (dot_val * (1.0 - cos_t)))
                            )
                            c_back_rot_sum += v_rot
                        c_back_mean = c_back_rot_sum / len(non_donor_indices)
                        proj = float(np.dot(c_back_mean, q_unit))
                        if proj > best_proj:
                            best_proj = proj
                            best_theta = theta_rad

                    if abs(best_theta) > 1e-12:
                        cos_t = math.cos(best_theta)
                        sin_t = math.sin(best_theta)
                        for i in range(num_atoms):
                            v_vec = aligned_coords[i] - q_cent
                            dot_val = float(np.dot(v_vec, axis_unit))
                            cross_val = np.cross(axis_unit, v_vec)
                            aligned_coords[i] = (
                                q_cent
                                + (v_vec * cos_t)
                                + (cross_val * sin_t)
                                + (axis_unit * (dot_val * (1.0 - cos_t)))
                            )

        # Validate bite angle RMSD
        for i in range(len(ligand.donor_atom_indices)):
            for j in range(i + 1, len(ligand.donor_atom_indices)):
                da_idx = ligand.donor_atom_indices[i]
                db_idx = ligand.donor_atom_indices[j]
                va = aligned_coords[da_idx]
                vb = aligned_coords[db_idx]
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na > 1e-12 and nb > 1e-12:
                    cos_calc = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
                    angle_calc = math.degrees(math.acos(cos_calc))

                    sa = template_vectors[ligand.target_vector_slots[i]]
                    sb = template_vectors[ligand.target_vector_slots[j]]
                    cos_tmpl = np.clip(np.dot(sa, sb), -1.0, 1.0)
                    angle_tmpl = math.degrees(math.acos(cos_tmpl))

                    deviation = abs(angle_calc - angle_tmpl)
                    if deviation > 5.0:
                        logger.warning(
                            "Bite angle deviation for ligand '%s' between donors (%d, %d): "
                            "calc=%.2f deg, template=%.2f deg, diff=%.2f deg",
                            ligand.ligand_id,
                            da_idx,
                            db_idx,
                            angle_calc,
                            angle_tmpl,
                            deviation,
                        )

    # Ensure all hydrogens bonded to donor atoms point outward away from metal (0, 0, 0)
    for donor_idx in ligand.donor_atom_indices:
        d_pos = aligned_coords[donor_idx]
        norm_d = float(np.linalg.norm(d_pos))
        if norm_d > 1e-12:
            u_d = d_pos / norm_d
            for h_idx, sym in enumerate(ligand.atomic_symbols):
                if sym == "H":
                    dist_to_donor = float(np.linalg.norm(aligned_coords[h_idx] - d_pos))
                    if dist_to_donor < 1.30:
                        w_vec = aligned_coords[h_idx] - d_pos
                        dot_proj = float(np.dot(w_vec, u_d))
                        if dot_proj < 0.0:
                            # Invert component pointing into metal so it points outward
                            w_corr = w_vec - 2.0 * dot_proj * u_d
                            aligned_coords[h_idx] = d_pos + w_corr

    # Validate physical asymptotic distance bounds: 1.5 A <= R_COM <= 12.0 A
    weights = np.array(
        [get_standard_atomic_weight(sym) for sym in ligand.atomic_symbols], dtype=np.float64
    )
    total_mass = float(np.sum(weights))
    if total_mass <= 0:
        total_mass = float(len(weights))
        weights = np.ones(len(weights), dtype=np.float64)

    com = np.sum(aligned_coords * weights[:, np.newaxis], axis=0) / total_mass
    r_com = float(np.linalg.norm(com))

    if not (1.5 <= r_com <= 12.0):
        raise UnphysicalMonomerSeparationError(
            f"Ligand '{ligand.ligand_id}' center-of-mass distance {r_com:.4f} A "
            f"violates physical asymptotic bounds [1.5, 12.0] A."
        )

    final_coords: np.ndarray = np.asarray(aligned_coords, dtype=np.float64)
    return final_coords
