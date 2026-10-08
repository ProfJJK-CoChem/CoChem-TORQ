"""Isotope-specific inertia and projected Cartesian-Hessian analysis.

Atomic units are used for the electronic derivatives. Normal modes have
L.T @ L = I in mass-weighted electron-mass coordinates; Cartesian displacement
is M**(-1/2) L Q, where Q has units bohr*sqrt(electron mass).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..units import (
    ATOMIC_MASS_ELECTRON,
    BOHR_METRE,
    HARTREE_CM1,
    atomic_mass,
    h,
    pi,
)
from ..units import (
    HARTREE_JOULE as HARTREE_JOULE,
)


def finite_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(
            "Physical arrays must be real; imaginary parts cannot be discarded."
        )
    array = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(array)) or (shape is not None and array.shape != shape):
        raise ValueError(
            f"Expected finite array with shape {shape}; got {array.shape}."
        )
    array.setflags(write=False)
    return array


def artifact_digest(value: Any) -> str:
    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {key: convert(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(child) for child in item]
        return item

    from ..domain import digest

    return digest(convert(value))


@dataclass(frozen=True)
class EquilibriumRotor:
    constants_mhz: tuple[float | None, float | None, float | None]
    principal_moments_u_bohr2: np.ndarray
    principal_axes_columns: np.ndarray
    rotor_type: str
    isotope_masses_u: np.ndarray
    geometry_digest: str
    observable: str = "Be"


@dataclass(frozen=True)
class HarmonicResult:
    coordinates_bohr: np.ndarray
    isotope_masses_u: np.ndarray
    frequencies_cm1: np.ndarray
    angular_frequencies_au: np.ndarray
    mass_weighted_modes: np.ndarray
    cartesian_modes: np.ndarray
    dimensionless_to_cartesian: np.ndarray | None
    eigenvalues_au: np.ndarray
    external_rank: int
    external_residual_relative: float
    symmetry_residual_relative: float
    stationary_character: str
    harmonic_zpe_hartree: float | None
    source_digest: str
    convention: str = (
        "orthonormal mass-weighted modes; electron masses; signed wavenumbers"
    )


def equilibrium_rotor(coordinates_bohr: Any, masses_u: Any) -> EquilibriumRotor:
    coordinates = finite_array(coordinates_bohr)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not len(coordinates):
        raise ValueError("Geometry must have nonempty shape [N,3].")
    masses = finite_array(masses_u, (len(coordinates),))
    if np.any(masses <= 0):
        raise ValueError("Explicit isotope masses must be positive.")
    centered = coordinates - np.average(coordinates, axis=0, weights=masses)
    inertia = np.eye(3) * np.sum(masses * np.sum(centered**2, axis=1))
    inertia -= np.einsum("i,ij,ik->jk", masses, centered, centered)
    moments, axes = np.linalg.eigh(inertia)
    tolerance = max(float(np.max(np.abs(moments))) * 1e-12, 1e-14)
    if np.any(moments < -tolerance):
        raise ValueError("Negative moment of inertia.")
    moments[np.abs(moments) < tolerance] = 0.0
    if len(coordinates) > 1 and moments[-1] == 0:
        raise ValueError("Coincident nuclei do not define a rotor.")
    for index in range(3):
        pivot = np.argmax(np.abs(axes[:, index]))
        if axes[pivot, index] < 0:
            axes[:, index] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    factor = h / (8 * pi * pi * atomic_mass * BOHR_METRE**2 * 1e6)
    constants = tuple(float(factor / x) if x > 0 else None for x in moments)
    return EquilibriumRotor(
        constants,
        finite_array(moments),
        finite_array(axes),
        "atom"
        if len(coordinates) == 1
        else "linear"
        if moments[0] == 0
        else "nonlinear",
        masses,
        artifact_digest({"geometry_bohr": coordinates, "masses_u": masses}),
    )


def analyze_hessian(
    coordinates_bohr: Any,
    masses_u: Any,
    hessian_hartree_bohr2: Any,
    *,
    symmetry_tolerance: float = 1e-8,
    zero_frequency_tolerance_cm1: float = 0.1,
) -> HarmonicResult:
    """Project translations/rotations, retaining all signed internal modes.

    The external residual is returned for qualification; projection never hides
    that a supplied Hessian fails invariance or was evaluated away from stationarity.
    """
    coordinates = finite_array(coordinates_bohr)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0:
        raise ValueError("Coordinates require shape [N,3].")
    count = len(coordinates)
    masses = finite_array(masses_u, (count,))
    if (
        np.any(masses <= 0)
        or symmetry_tolerance < 0
        or zero_frequency_tolerance_cm1 < 0
    ):
        raise ValueError("Masses must be positive and tolerances nonnegative.")
    hessian = finite_array(hessian_hartree_bohr2, (3 * count, 3 * count))
    norm = float(np.linalg.norm(hessian))
    symmetry_residual = float(
        np.linalg.norm(hessian - hessian.T) / max(norm, np.finfo(float).tiny)
    )
    if symmetry_residual > symmetry_tolerance:
        raise ValueError(
            f"Hessian symmetry residual {symmetry_residual} "
            f"exceeds {symmetry_tolerance}."
        )
    hessian_symmetric = (hessian + hessian.T) / 2
    electron_masses = masses * ATOMIC_MASS_ELECTRON
    mass_vector = np.repeat(electron_masses, 3)
    centered = coordinates - np.average(coordinates, axis=0, weights=masses)
    external = []
    for axis in np.eye(3):
        external.append(
            (np.tile(axis, (count, 1)) * np.sqrt(electron_masses)[:, None]).ravel()
        )
        external.append(
            (np.cross(axis, centered) * np.sqrt(electron_masses)[:, None]).ravel()
        )
    external_matrix = np.asarray(external).T
    u, singular, _ = np.linalg.svd(external_matrix, full_matrices=True)
    rank = int(np.sum(singular > max(float(singular.max()) * 1e-10, 1e-13)))
    weighted = hessian_symmetric / np.sqrt(mass_vector[:, None] * mass_vector[None, :])
    external_residual = float(
        np.linalg.norm(weighted @ u[:, :rank])
        / max(float(np.linalg.norm(weighted)), np.finfo(float).tiny)
    )
    internal_basis = u[:, rank:]
    eigenvalues, vectors = np.linalg.eigh(internal_basis.T @ weighted @ internal_basis)
    modes = internal_basis @ vectors
    signed_frequencies = (
        np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * HARTREE_CM1
    )
    angular = np.sqrt(np.abs(eigenvalues))
    cartesian = modes / np.sqrt(mass_vector)[:, None]
    if np.any(signed_frequencies < -zero_frequency_tolerance_cm1):
        character = "nonminimum"
    elif np.any(signed_frequencies <= zero_frequency_tolerance_cm1):
        character = "unresolved_zero_modes"
    else:
        character = "positive_definite_vibrational_hessian"
    transform = (
        cartesian / np.sqrt(angular)[None, :]
        if character == "positive_definite_vibrational_hessian"
        else None
    )
    zpe = (
        float(0.5 * np.sum(angular))
        if character == "positive_definite_vibrational_hessian"
        else None
    )
    digest = artifact_digest(
        {
            "coordinates_bohr": coordinates,
            "masses_u": masses,
            "hessian_hartree_bohr2": hessian,
        }
    )
    return HarmonicResult(
        coordinates,
        masses,
        finite_array(signed_frequencies),
        finite_array(angular),
        finite_array(modes),
        finite_array(cartesian),
        None if transform is None else finite_array(transform),
        finite_array(eigenvalues),
        rank,
        external_residual,
        symmetry_residual,
        character,
        zpe,
        digest,
    )


def analyze_isotopologues(
    coordinates_bohr: Any, hessian_hartree_bohr2: Any, isotope_masses: dict[str, Any]
) -> dict[str, HarmonicResult]:
    """Born–Oppenheimer isotope substitution: unchanged Cartesian Hessian.

    The approximation excludes diagonal Born–Oppenheimer and other isotope-specific
    electronic corrections. Each isotopologue recomputes COM and external projection.
    """
    if not isotope_masses:
        raise ValueError("At least one explicitly named isotopologue is required.")
    return {
        name: analyze_hessian(coordinates_bohr, masses, hessian_hartree_bohr2)
        for name, masses in isotope_masses.items()
    }
