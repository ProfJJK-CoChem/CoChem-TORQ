"""Declared internal-coordinate mathematics for fixed molecular PES samples.

An embedding minimizes Cartesian displacement from the supplied geometry subject
to explicitly requested coordinates. This is geometry construction, not an
electronic energy optimization or evidence of a stationary molecular structure.
Only declared movable atom rows change. No graph, conformer equivalence, symmetry
division or chemical connectivity is inferred from the coordinate operations.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from scipy.optimize import minimize

from .units import convert, real_values


def coordinate_value(geometry_bohr: Any, coordinate: Any) -> float:
    """Evaluate a bond in bohr or an angle/torsion in radians.

    Torsions use atan2 of oriented normals around the middle atom pair. Linear
    adjacent bonds do not define a torsion and are rejected rather than assigned
    an arbitrary zero. Coordinate atom indices are the immutable request rows.
    """
    from .scan import InternalCoordinate

    if not isinstance(coordinate, InternalCoordinate):
        raise TypeError("Use an explicitly validated InternalCoordinate definition.")
    geometry = real_values(geometry_bohr)
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise ValueError("Coordinates require a finite N by 3 Cartesian geometry.")
    indices = coordinate.atom_indices
    if max(indices) >= len(geometry):
        raise ValueError("Internal-coordinate atom indices exceed the geometry.")
    rows = geometry[list(indices)]
    if coordinate.kind == "bond":
        value = float(np.linalg.norm(rows[1] - rows[0]))
        if value <= 1e-10:
            raise ValueError("A coincident bond has no defined direction.")
        return value
    if coordinate.kind == "angle":
        left, right = rows[0] - rows[1], rows[2] - rows[1]
        scale = float(np.linalg.norm(left) * np.linalg.norm(right))
        if scale <= 1e-20:
            raise ValueError("Coincident atoms do not define a valence angle.")
        return float(math.atan2(np.linalg.norm(np.cross(left, right)), left @ right))
    first, central, last = rows[0] - rows[1], rows[2] - rows[1], rows[3] - rows[2]
    axis_length = float(np.linalg.norm(central))
    if axis_length <= 1e-10:
        raise ValueError("A coincident middle bond cannot define torsion.")
    axis = central / axis_length
    left = first - (first @ axis) * axis
    right = last - (last @ axis) * axis
    if min(np.linalg.norm(left), np.linalg.norm(right)) <= 1e-10:
        raise ValueError("Collinear adjacent bonds do not define torsion.")
    return float(math.atan2(np.cross(axis, left) @ right, left @ right))


def coordinate_target(value: float, coordinate: Any) -> float:
    if coordinate.kind == "bond":
        return float(convert(value, coordinate.unit, "bohr"))
    return float(value * math.pi / 180 if coordinate.unit == "degree" else value)


def coordinate_residual(value: float, target: float, coordinate: Any) -> float:
    difference = value - target
    if coordinate.kind == "dihedral":
        return float(math.atan2(math.sin(difference), math.cos(difference)))
    return float(difference)


def constraint_jacobian(
    geometry_bohr: Any, coordinates: Any, *, step_bohr: float = 1e-5
) -> np.ndarray:
    """Exact bond and centered angular coordinate Jacobian with wrapped torsions."""
    if not math.isfinite(step_bohr) or step_bohr <= 0:
        raise ValueError(
            "A positive finite coordinate differentiation step is required."
        )
    geometry = real_values(geometry_bohr)
    result = np.empty((len(coordinates), geometry.size))
    for row, coordinate in enumerate(coordinates):
        if coordinate.kind == "bond":
            length = coordinate_value(geometry, coordinate)
            left, right = coordinate.atom_indices
            direction = (geometry[right] - geometry[left]) / length
            result[row] = 0.0
            result[row, 3 * left : 3 * left + 3] = -direction
            result[row, 3 * right : 3 * right + 3] = direction
    for column in range(geometry.size):
        plus, minus = geometry.copy(), geometry.copy()
        plus.flat[column] += step_bohr
        minus.flat[column] -= step_bohr
        for row, coordinate in enumerate(coordinates):
            if coordinate.kind == "bond":
                continue
            result[row, column] = coordinate_residual(
                coordinate_value(plus, coordinate),
                coordinate_value(minus, coordinate),
                coordinate,
            ) / (2 * step_bohr)
    return result


def embed_fixed_coordinates(
    reference_bohr: Any,
    coordinates: Any,
    values: Any,
    movable_atoms: tuple[int, ...],
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 300,
) -> dict[str, Any]:
    """Solve declared fixed-coordinate construction without evaluating an engine.

    The deterministic minimum-displacement objective chooses one embedding; it
    does not characterize other embeddings or establish a complete PES branch.
    Every unselected Cartesian atom row stays bit-for-bit identical to input.
    """
    reference = real_values(reference_bohr)
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 2:
        raise ValueError("A molecular embedding requires a finite N by 3 geometry.")
    if (
        not movable_atoms
        or any(
            type(index) is not int or not 0 <= index < len(reference)
            for index in movable_atoms
        )
        or len(set(movable_atoms)) != len(movable_atoms)
    ):
        raise ValueError("Movable atoms require distinct valid immutable atom rows.")
    targets_input = real_values(values)
    if targets_input.shape != (len(coordinates),) or not coordinates:
        raise ValueError("Each coordinate requires one declared target value.")
    if not math.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
        raise ValueError("Embedding tolerance must be positive and at most 1e-6.")
    if type(max_iterations) is not int or not 1 <= max_iterations <= 1000:
        raise ValueError("Embedding iterations must be an integer in [1,1000].")
    targets = tuple(
        coordinate_target(value, coordinate)
        for value, coordinate in zip(targets_input, coordinates)
    )
    for coordinate, target in zip(coordinates, targets):
        if coordinate.kind == "angle" and not 0 < target < math.pi:
            raise ValueError(
                "Executable valence angles must lie strictly between zero and pi."
            )
    selected = list(movable_atoms)
    origin = reference[selected].reshape(-1)

    def geometry(vector: np.ndarray) -> np.ndarray:
        result = reference.copy()
        result[selected] = vector.reshape(-1, 3)
        return cast(np.ndarray, result)

    def residual(vector: np.ndarray) -> np.ndarray:
        sample = geometry(vector)
        return np.asarray(
            [
                coordinate_residual(
                    coordinate_value(sample, coordinate), target, coordinate
                )
                for coordinate, target in zip(coordinates, targets)
            ]
        )

    columns = np.asarray(
        [[3 * atom + axis for axis in range(3)] for atom in selected]
    ).reshape(-1)

    def jacobian(vector: np.ndarray) -> np.ndarray:
        return np.asarray(
            constraint_jacobian(geometry(vector), coordinates)[:, columns], dtype=float
        )

    initial_jacobian = jacobian(origin)
    if np.linalg.matrix_rank(initial_jacobian, tol=1e-8) != len(coordinates):
        raise ValueError(
            "Declared coordinates are dependent or cannot be changed by movable atoms."
        )
    result = minimize(
        lambda vector: float(0.5 * np.sum((vector - origin) ** 2)),
        origin,
        jac=lambda vector: vector - origin,
        constraints={"type": "eq", "fun": residual, "jac": jacobian},
        method="SLSQP",
        # Objective stopping and coordinate acceptance are separate dimensions.
        # Asking SLSQP for sub-machine objective changes can fail line search
        # even for a translated exact bond solution. Independent residual and
        # KKT checks below still enforce the declared geometry accuracy.
        options={"ftol": 1e-13, "maxiter": max_iterations},
    )
    errors = residual(result.x)
    sample = geometry(result.x)
    if not result.success or float(np.max(np.abs(errors))) > tolerance:
        raise ValueError(
            f"Declared coordinate embedding did not converge: {result.message}; "
            f"residuals={errors.tolist()}"
        )
    final_jacobian = jacobian(result.x)
    if np.linalg.matrix_rank(final_jacobian, tol=1e-8) != len(coordinates):
        raise ValueError("The constructed sample has dependent coordinate constraints.")
    displacement_gradient = result.x - origin
    multipliers = np.linalg.lstsq(final_jacobian.T, displacement_gradient, rcond=None)[
        0
    ]
    stationarity_residual = float(
        np.linalg.norm(displacement_gradient - final_jacobian.T @ multipliers)
    )
    if stationarity_residual > 1e-7 * max(
        1.0, float(np.linalg.norm(displacement_gradient))
    ):
        raise ValueError(
            "Coordinate construction failed its independent objective KKT check."
        )
    distances = np.linalg.norm(sample[:, None] - sample[None, :], axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    if np.min(distances) <= 1e-6:
        raise ValueError("The constructed sample contains coincident nuclei.")
    return {
        "geometry_bohr": sample.tolist(),
        "construction": (
            "minimum_cartesian_displacement_with_declared_internal_constraints"
        ),
        "evidence_class": "derived_coordinate_mathematics",
        "movable_atom_indices": list(movable_atoms),
        "residuals_bohr_or_radian": errors.tolist(),
        "jacobian_rank": len(coordinates),
        "iterations": int(result.nit),
        "objective_stationarity_residual": stationarity_residual,
        "global_embedding_minimum_claimed": False,
        "stationary_point_claimed": False,
    }
