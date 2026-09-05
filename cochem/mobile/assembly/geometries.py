"""Ideal coordination polyhedral template vector generators.

Produces normalized Cartesian unit vectors in R^3 (||v_k|| = 1.0) centered at (0, 0, 0)
for coordination numbers CN=2 to CN=8.
"""

from __future__ import annotations

import math

import numpy as np

from cochem.mobile.assembly.models import CoordinationGeometryEnum


def get_coordination_template_vectors(geometry: CoordinationGeometryEnum) -> np.ndarray:
    """Generate normalized Cartesian unit vectors for the specified coordination geometry.

    Args:
        geometry: Target CoordinationGeometryEnum value.

    Returns:
        np.ndarray of shape (CN, 3) where each row is a unit vector (norm = 1.0).

    Raises:
        ValueError: If geometry is unhandled.
    """
    if geometry == CoordinationGeometryEnum.LINEAR:
        # CN = 2: along z-axis
        vectors = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

    elif geometry == CoordinationGeometryEnum.TRIGONAL_PLANAR:
        # CN = 3: in xy-plane at 120 degree separation
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [-0.5, math.sqrt(3.0) / 2.0, 0.0],
                [-0.5, -math.sqrt(3.0) / 2.0, 0.0],
            ],
            dtype=np.float64,
        )

    elif geometry == CoordinationGeometryEnum.TETRAHEDRAL:
        # CN = 4: ideal tetrahedral angle 109.4712 degrees
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        vectors = np.array(
            [
                [1.0 * inv_sqrt3, 1.0 * inv_sqrt3, 1.0 * inv_sqrt3],
                [1.0 * inv_sqrt3, -1.0 * inv_sqrt3, -1.0 * inv_sqrt3],
                [-1.0 * inv_sqrt3, 1.0 * inv_sqrt3, -1.0 * inv_sqrt3],
                [-1.0 * inv_sqrt3, -1.0 * inv_sqrt3, 1.0 * inv_sqrt3],
            ],
            dtype=np.float64,
        )

    elif geometry == CoordinationGeometryEnum.SQUARE_PLANAR:
        # CN = 4: in xy-plane at 90 degree separation
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )

    elif geometry == CoordinationGeometryEnum.TRIGONAL_BIPYRAMIDAL:
        # CN = 5: 2 axial (z-axis) + 3 equatorial (xy-plane, 120 deg)
        vectors = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [-0.5, math.sqrt(3.0) / 2.0, 0.0],
                [-0.5, -math.sqrt(3.0) / 2.0, 0.0],
            ],
            dtype=np.float64,
        )

    elif geometry == CoordinationGeometryEnum.SQUARE_PYRAMIDAL:
        # CN = 5: 1 axial (z-axis) + 4 basal (xy-plane, 90 deg)
        vectors = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )

    elif geometry == CoordinationGeometryEnum.OCTAHEDRAL:
        # CN = 6: +/- x, +/- y, +/- z
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

    elif geometry == CoordinationGeometryEnum.PENTAGONAL_BIPYRAMIDAL:
        # CN = 7: 2 axial (+/- z) + 5 equatorial in xy-plane (72 deg intervals)
        eq_vecs = []
        for k in range(5):
            angle = k * 2.0 * math.pi / 5.0
            eq_vecs.append([math.cos(angle), math.sin(angle), 0.0])
        vectors = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                *eq_vecs,
            ],
            dtype=np.float64,
        )

    elif geometry == CoordinationGeometryEnum.SQUARE_ANTIPRISMATIC:
        # CN = 8: Archimedean square antiprism with 45 degree staggered rings
        theta = math.radians(59.26)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        top_vecs = [
            [sin_t * math.cos(math.radians(phi)), sin_t * math.sin(math.radians(phi)), cos_t]
            for phi in [0, 90, 180, 270]
        ]
        bot_vecs = [
            [sin_t * math.cos(math.radians(phi)), sin_t * math.sin(math.radians(phi)), -cos_t]
            for phi in [45, 135, 225, 315]
        ]
        vectors = np.array([*top_vecs, *bot_vecs], dtype=np.float64)

    else:
        raise ValueError(f"Unsupported coordination geometry: {geometry}")

    # Ensure exact unit normalization
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors: np.ndarray = np.asarray(vectors / norms, dtype=np.float64)
    return normalized_vectors
