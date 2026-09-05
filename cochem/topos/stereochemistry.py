"""Graph-Level Stereocenter & Chirality Detector.

Implements coordinate-frame-invariant vector triple product parity for tetrahedral
stereocenters (R/S) and torsional dihedral boundary assignment for double bonds (E/Z).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from cochem.topos.exceptions import ChiralityAssignmentError

logger = logging.getLogger("cochem.topos.stereochemistry")

DEGENERATE_CHIRAL_THRESHOLD: float = 1e-6


def assign_tetrahedral_chirality(
    coords: np.ndarray,
    center_idx: int,
    priority_indices: tuple[int, int, int, int],
) -> str:
    """Assigns R/S tetrahedral chirality via coordinate-frame-invariant vector triple product parity.
    
    Given central chiral atom at r0 and four prioritized ligand coordinates r1, r2, r3, r4
    (where priority 1 > 2 > 3 > 4), evaluates:
        Delta_chiral = [(r1 - r0) x (r2 - r0)] . (r0 - r4)
        
    Parity:
      - Delta_chiral < 0 => "R" (Rectus, clockwise 1 -> 2 -> 3 looking from ligand 4 toward r0)
      - Delta_chiral > 0 => "S" (Sinister, counter-clockwise 1 -> 2 -> 3)
      - |Delta_chiral| < 1e-6 => raises ChiralityAssignmentError("Planar or degenerate chiral configuration")
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ChiralityAssignmentError(f"Coordinates must be (N, 3), got {coords.shape}")

    r0 = coords[center_idx]
    r1 = coords[priority_indices[0]]
    r2 = coords[priority_indices[1]]
    r3 = coords[priority_indices[2]]
    r4 = coords[priority_indices[3]]

    v1 = r1 - r0
    v2 = r2 - r0
    v4 = r0 - r4

    cross_12 = np.cross(v1, v2)
    delta_chiral = float(np.dot(cross_12, v4))

    if abs(delta_chiral) < DEGENERATE_CHIRAL_THRESHOLD:
        raise ChiralityAssignmentError(
            f"Planar or degenerate chiral configuration detected at center {center_idx} "
            f"(Delta_chiral = {delta_chiral:.2e})"
        )

    if delta_chiral < 0.0:
        return "R"
    return "S"


def compute_dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates 4-atom torsional dihedral angle in degrees in [-180, 180]."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    norm_b2 = np.linalg.norm(b2)

    if norm_n1 < 1e-8 or norm_n2 < 1e-8 or norm_b2 < 1e-8:
        return 0.0

    u1 = n1 / norm_n1
    u2 = n2 / norm_n2
    ub2 = b2 / norm_b2

    m1 = np.cross(u1, ub2)

    x = float(np.dot(u1, u2))
    y = float(np.dot(m1, u2))

    angle_rad = math.atan2(y, x)
    return math.degrees(angle_rad)


def assign_double_bond_stereo(
    coords: np.ndarray,
    substituent_a: int,
    terminus_a: int,
    terminus_b: int,
    substituent_b: int,
) -> str:
    """Assigns E/Z configuration to a double bond based on torsional dihedral angle phi.
    
    phi(substituent_a, terminus_a, terminus_b, substituent_b):
      - |phi| < 90 deg => "Z" (Zusammen, cisoid)
      - |phi| >= 90 deg => "E" (Entgegen, transoid)
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ChiralityAssignmentError(f"Coordinates must be (N, 3), got {coords.shape}")

    p0 = coords[substituent_a]
    p1 = coords[terminus_a]
    p2 = coords[terminus_b]
    p3 = coords[substituent_b]

    phi = compute_dihedral_angle(p0, p1, p2, p3)

    if abs(phi) < 90.0:
        return "Z"
    return "E"
