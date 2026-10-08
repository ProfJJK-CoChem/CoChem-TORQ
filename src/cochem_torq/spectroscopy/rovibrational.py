"""Geometric mode couplings and fixed-equilibrium-frame inertia derivatives.

Q has units bohr*sqrt(electron mass), r(Q)=r_e+M**(-1/2)LQ, and L.T L=1.
These algebraic precursors do not implement a rovibrational kinetic operator,
rotation-vibration alpha, B0, centrifugal distortion or a full VPT2 solver.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from ..units import ATOMIC_MASS_ELECTRON, HARTREE_CM1, constants_provenance
from .harmonic import (
    EquilibriumRotor,
    HarmonicResult,
    artifact_digest,
    equilibrium_rotor,
    finite_array,
)
from .results import ScientificContext

Matrix = tuple[tuple[float, ...], ...]
Tensor3 = tuple[tuple[tuple[float, ...], ...], ...]
Tensor4 = tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]
_CONVENTION = "orthonormal mass-weighted modes; electron masses; signed wavenumbers"
_SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


class _Immutable(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )


class PrecursorIdentity(_Immutable):
    """Immutable copy of validated source, isotope, mode and frame declarations.

    Parent digests identify retained artifacts. The calling workflow must verify
    their actual bytes; these geometric calculations do not authenticate an engine.
    """

    input_context_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    input_evidence_class: Literal["engine_calculation", "mathematical_model"]
    symbols: tuple[str, ...]
    atom_ids: tuple[str, ...]
    atom_id_policy: Literal["source_identifiers", "input_ordinal"]
    isotope_numbers: tuple[int, ...]
    isotope_masses_u: tuple[float, ...]
    isotope_selection_policies: tuple[str, ...]
    isotope_reference_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    charge: int
    multiplicity: int
    geometry_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    harmonic_source_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    harmonic_declaration_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    mode_basis_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    mode_order: tuple[str, ...]
    frame_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    equilibrium_rotor_geometry_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    recipe_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    protocol_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    parent_artifact_sha256: tuple[str, ...]
    algorithm_source_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    constants_sha256: str = Field(pattern="^[a-f0-9]{64}$")


class UnavailableRovibrationalProduct(_Immutable):
    name: Literal[
        "rovibrational_kinetic_operator",
        "rotation_vibration_alpha",
        "vibration_corrected_constants",
        "centrifugal_distortion",
        "full_rovibrational_vpt2",
    ]
    status: Literal["unavailable"] = "unavailable"
    value: None = None
    reason: str = Field(min_length=1)


class RovibrationalPrecursors(_Immutable):
    """Deeply immutable tensors in a fixed, retained equilibrium axis gauge."""

    schema_version: Literal["cochem.torq.rovibrational-precursors/1"] = (
        "cochem.torq.rovibrational-precursors/1"
    )
    identity: PrecursorIdentity
    evidence_class: Literal["derived_geometric_mathematics"] = (
        "derived_geometric_mathematics"
    )
    independent_scientific_qualification: Literal[False] = False
    identification_ready: Literal[False] = False
    normal_coordinate_unit: Literal["bohr*sqrt(electron_mass)"] = (
        "bohr*sqrt(electron_mass)"
    )
    inertia_unit: Literal["electron_mass*bohr^2"] = "electron_mass*bohr^2"
    first_derivative_unit: Literal["bohr*sqrt(electron_mass)"] = (
        "bohr*sqrt(electron_mass)"
    )
    second_derivative_unit: Literal["dimensionless"] = "dimensionless"
    coriolis_unit: Literal["dimensionless"] = "dimensionless"
    coriolis_convention: Literal[
        "zeta[axis,k,l]=sum_atom cross(L_atom_k,L_atom_l)[axis]"
    ] = "zeta[axis,k,l]=sum_atom cross(L_atom_k,L_atom_l)[axis]"
    derivative_convention: Literal[
        "full geometric inertia tensor; Cartesian-linear Q; fixed equilibrium axes"
    ] = "full geometric inertia tensor; Cartesian-linear Q; fixed equilibrium axes"
    isotope_mass_u_to_electron_mass: float = Field(gt=0)
    input_geometry_bohr: Matrix
    input_mass_weighted_modes: Matrix
    center_of_mass_bohr: tuple[float, float, float]
    principal_axes_columns: Matrix
    centered_geometry_principal_bohr: Matrix
    mass_weighted_modes_principal: Tensor3
    harmonic_frequencies_cm1: tuple[float, ...]
    inertia_electron_mass_bohr2: Matrix
    principal_moments_electron_mass_bohr2: tuple[float, float, float]
    coriolis_zeta: Tensor3
    inertia_first_derivative: Tensor3
    inertia_second_derivative: Tensor4
    mode_orthonormality_residual: float = Field(ge=0)
    translation_residual: float = Field(ge=0)
    rotational_eckart_residual: float = Field(ge=0)
    external_subspace_residual: float = Field(ge=0)
    harmonic_external_residual: float = Field(ge=0)
    orthogonality_tolerance: float = Field(gt=0)
    maximum_harmonic_external_residual: float = Field(gt=0)
    inertia_degeneracy_relative_tolerance: float = Field(ge=0)
    mode_degeneracy_relative_tolerance: float = Field(ge=0)
    degenerate_inertia_axis_pairs: tuple[tuple[int, int], ...]
    degenerate_mode_pairs: tuple[tuple[int, int], ...]
    zero_inertia_axes: tuple[int, ...]
    quality_flags: tuple[str, ...]
    unavailable_products: tuple[UnavailableRovibrationalProduct, ...]

    @model_validator(mode="after")
    def geometric_readback(self) -> Self:
        """A typed readback cannot substitute unrelated tensors or qualifications."""
        atoms, count = len(self.identity.symbols), len(self.identity.mode_order)
        positions = finite_array(self.centered_geometry_principal_bohr, (atoms, 3))
        masses = finite_array(self.identity.isotope_masses_u, (atoms,))
        modes = finite_array(self.mass_weighted_modes_principal, (atoms, 3, count))
        frequencies = finite_array(self.harmonic_frequencies_cm1, (count,))
        axes = finite_array(self.principal_axes_columns, (3, 3))
        input_positions = finite_array(self.input_geometry_bohr, (atoms, 3))
        input_modes = finite_array(self.input_mass_weighted_modes, (3 * atoms, count))
        moments = finite_array(self.principal_moments_electron_mass_bohr2, (3,))
        if (
            not atoms
            or not count
            or np.any(masses <= 0)
            or np.any(frequencies <= 0)
            or self.isotope_mass_u_to_electron_mass != ATOMIC_MASS_ELECTRON
            or not np.allclose(axes.T @ axes, np.eye(3), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(axes), 1, rtol=0, atol=1e-10)
        ):
            raise ValueError(
                "Precursor shapes, units or right-handed axes are inconsistent."
            )
        identity = self.identity
        copied_context = ScientificContext.model_validate(
            {
                "evidence_class": identity.input_evidence_class,
                "symbols": list(identity.symbols),
                "atom_ids": list(identity.atom_ids),
                "atom_id_policy": identity.atom_id_policy,
                "isotope_numbers": list(identity.isotope_numbers),
                "isotope_masses_u": list(identity.isotope_masses_u),
                "isotope_selection_policies": list(identity.isotope_selection_policies),
                "isotope_reference_sha256": identity.isotope_reference_sha256,
                "geometry_bohr": input_positions.tolist(),
                "geometry_sha256": identity.geometry_sha256,
                "charge": identity.charge,
                "multiplicity": identity.multiplicity,
                "frame_type": "principal_inertia",
                "frame_axes_columns": axes.tolist(),
                "frame_sha256": identity.frame_sha256,
                "mode_count": count,
                "mode_order": list(identity.mode_order),
                "mode_basis_sha256": identity.mode_basis_sha256,
                "recipe_sha256": identity.recipe_sha256,
                "protocol_sha256": identity.protocol_sha256,
                "parent_artifact_sha256": list(identity.parent_artifact_sha256),
            }
        )
        if (
            identity.input_context_sha256
            != artifact_digest(copied_context.model_dump(mode="json"))
            or identity.mode_basis_sha256
            != artifact_digest(
                {
                    "mass_weighted_modes": input_modes,
                    "isotope_masses_u": masses,
                    "geometry_bohr": input_positions,
                    "convention": _CONVENTION,
                }
            )
            or identity.equilibrium_rotor_geometry_sha256
            != artifact_digest({"geometry_bohr": input_positions, "masses_u": masses})
            or "0" * 64
            in (
                identity.harmonic_source_sha256,
                identity.recipe_sha256,
                identity.protocol_sha256,
                identity.isotope_reference_sha256,
                identity.algorithm_source_sha256,
                *identity.parent_artifact_sha256,
            )
            or identity.constants_sha256 != artifact_digest(constants_provenance())
            or identity.harmonic_declaration_sha256
            != _harmonic_declaration_digest(
                identity.harmonic_source_sha256,
                identity.mode_basis_sha256,
                frequencies,
                self.harmonic_external_residual,
            )
        ):
            raise ValueError(
                "Retained input arrays or identities disagree with their digests."
            )
        electron_masses = masses * self.isotope_mass_u_to_electron_mass
        center = np.average(input_positions, weights=electron_masses, axis=0)
        centered = input_positions - center
        if (
            not np.array_equal(center, self.center_of_mass_bohr)
            or not np.array_equal(centered @ axes, positions)
            or not np.array_equal(
                np.einsum("aik,ij->ajk", input_modes.reshape(atoms, 3, count), axes),
                modes,
            )
        ):
            raise ValueError(
                "Retained COM/frame transform differs from the original input."
            )
        flattened = modes.reshape(3 * atoms, count)
        translation = np.einsum("a,aik->ik", np.sqrt(electron_masses), modes)
        rotation = np.zeros((3, count))
        for atom in range(atoms):
            rotation += (
                np.sqrt(electron_masses[atom])
                * np.cross(positions[atom], modes[atom].T).T
            )
        radius = float(np.sqrt(np.sum(electron_masses * np.sum(positions**2, axis=1))))
        if (
            radius == 0
            or np.linalg.norm(flattened.T @ flattened - np.eye(count))
            > self.orthogonality_tolerance
            or np.linalg.norm(translation) / np.sqrt(electron_masses.sum())
            > self.orthogonality_tolerance
            or np.linalg.norm(rotation) / radius > self.orthogonality_tolerance
            or np.linalg.norm(np.average(positions, weights=masses, axis=0))
            > self.orthogonality_tolerance
        ):
            raise ValueError(
                "Retained mode normalization or COM/Eckart constraints fail."
            )
        expected = _geometric_tensors(positions, electron_masses, modes)
        for observed, tensor in zip(
            (
                self.inertia_electron_mass_bohr2,
                self.coriolis_zeta,
                self.inertia_first_derivative,
                self.inertia_second_derivative,
            ),
            expected,
        ):
            values = finite_array(observed, tensor.shape)
            if not np.allclose(values, tensor, rtol=1e-12, atol=1e-12):
                raise ValueError(
                    "Retained geometric tensors disagree with their actual modes."
                )
        scale = float(np.max(np.abs(expected[0])))
        if (
            np.any(moments < 0)
            or np.any(np.diff(moments) < 0)
            or not np.allclose(
                expected[0], np.diag(moments), rtol=1e-10, atol=scale * 1e-12
            )
        ):
            raise ValueError(
                "Retained source principal moments disagree with actual inertia."
            )
        inertia_pairs = _pairs(moments, self.inertia_degeneracy_relative_tolerance)
        mode_pairs = _pairs(frequencies, self.mode_degeneracy_relative_tolerance)
        zero_axes = tuple(index for index, moment in enumerate(moments) if moment == 0)
        canonical_moments = equilibrium_rotor(
            input_positions, masses
        ).principal_moments_u_bohr2
        canonical_zero_axes = tuple(
            index for index, moment in enumerate(canonical_moments) if moment == 0
        )
        if (
            self.degenerate_mode_pairs != mode_pairs
            or self.degenerate_inertia_axis_pairs != inertia_pairs
            or self.zero_inertia_axes != zero_axes
            or zero_axes != canonical_zero_axes
            or self.quality_flags
            != _quality_flags(inertia_pairs, mode_pairs, zero_axes)
        ):
            raise ValueError(
                "Degeneracy, zero-axis or quality flags disagree with retained arrays."
            )
        external = []
        for axis in np.eye(3):
            external.append(
                (np.tile(axis, (atoms, 1)) * np.sqrt(electron_masses)[:, None]).ravel()
            )
            external.append(
                (np.cross(axis, centered) * np.sqrt(electron_masses)[:, None]).ravel()
            )
        basis, singular, _ = np.linalg.svd(np.asarray(external).T, full_matrices=True)
        rank = int(np.sum(singular > max(float(singular.max()) * 1e-10, 1e-13)))
        original_modes = input_modes.reshape(atoms, 3, count)
        original_rotation = np.zeros((3, count))
        for atom in range(atoms):
            original_rotation += (
                np.sqrt(electron_masses[atom])
                * np.cross(centered[atom], original_modes[atom].T).T
            )
        original_radius = float(
            np.sqrt(np.sum(electron_masses * np.sum(centered**2, axis=1)))
        )
        residuals = (
            float(np.linalg.norm(input_modes.T @ input_modes - np.eye(count))),
            float(
                np.linalg.norm(
                    np.einsum("a,aik->ik", np.sqrt(electron_masses), original_modes)
                )
                / np.sqrt(electron_masses.sum())
            ),
            float(np.linalg.norm(original_rotation) / original_radius),
            float(np.linalg.norm(basis[:, :rank].T @ input_modes)),
        )
        if (
            count != 3 * atoms - rank
            or max(residuals) > self.orthogonality_tolerance
            or self.harmonic_external_residual > self.maximum_harmonic_external_residual
            or not np.allclose(
                residuals,
                (
                    self.mode_orthonormality_residual,
                    self.translation_residual,
                    self.rotational_eckart_residual,
                    self.external_subspace_residual,
                ),
                rtol=1e-12,
                atol=1e-15,
            )
        ):
            raise ValueError(
                "Retained residuals or complete-subspace gates are inconsistent."
            )
        names = tuple(product.name for product in self.unavailable_products)
        if names != (
            "rovibrational_kinetic_operator",
            "rotation_vibration_alpha",
            "vibration_corrected_constants",
            "centrifugal_distortion",
            "full_rovibrational_vpt2",
        ):
            raise ValueError(
                "Unavailable higher products cannot be omitted or renamed."
            )
        return self


def _matrix(value: np.ndarray) -> Matrix:
    return tuple(tuple(float(x) for x in row) for row in value)


def _tensor3(value: np.ndarray) -> Tensor3:
    return tuple(_matrix(plane) for plane in value)


def _tensor4(value: np.ndarray) -> Tensor4:
    return tuple(_tensor3(block) for block in value)


def _pairs(values: np.ndarray, tolerance: float) -> tuple[tuple[int, int], ...]:
    return tuple(
        (first, second)
        for first in range(len(values))
        for second in range(first + 1, len(values))
        if abs(float(values[first] - values[second]))
        <= tolerance * max(abs(float(values[first])), abs(float(values[second])))
    )


def _quality_flags(
    inertia_pairs: tuple[tuple[int, int], ...],
    mode_pairs: tuple[tuple[int, int], ...],
    zero_axes: tuple[int, ...],
) -> tuple[str, ...]:
    flags = ["geometric_precursors_only", "input_mode_phases_and_frame_gauge_retained"]
    if inertia_pairs:
        flags.append("degenerate_inertia_axes_have_basis_dependent_components")
    if mode_pairs:
        flags.append("degenerate_normal_modes_have_basis_dependent_components")
    if zero_axes:
        flags.append("zero_inertia_axis_no_inversion_or_finite_rotational_constant")
    return tuple(flags)


def _harmonic_declaration_digest(
    source_sha256: str,
    mode_basis_sha256: str,
    frequencies_cm1: np.ndarray,
    external_residual_relative: float,
) -> str:
    """Internal binding of supplied harmonic fields, not native authentication."""
    return artifact_digest(
        {
            "source_sha256": source_sha256,
            "mode_basis_sha256": mode_basis_sha256,
            "frequencies_cm1": frequencies_cm1,
            "external_residual_relative": external_residual_relative,
        }
    )


def _geometric_tensors(
    positions: np.ndarray, masses: np.ndarray, modes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = modes.shape[2]
    displacements = modes / np.sqrt(masses)[:, None, None]
    inertia = np.eye(3) * np.sum(masses * np.sum(positions**2, axis=1))
    inertia -= np.einsum("a,ai,aj->ij", masses, positions, positions)
    zeta = np.zeros((3, count, count))
    for atom in range(len(masses)):
        vectors = modes[atom].T
        zeta += np.cross(vectors[:, None, :], vectors[None, :, :]).transpose(2, 0, 1)
    dot_first = np.einsum("a,ai,aik->k", masses, positions, displacements)
    first = 2 * np.eye(3)[:, :, None] * dot_first
    first -= np.einsum("a,ai,ajk->ijk", masses, positions, displacements)
    first -= np.einsum("a,aj,aik->ijk", masses, positions, displacements)
    dot_second = np.einsum("a,aik,ail->kl", masses, displacements, displacements)
    second = 2 * np.eye(3)[:, :, None, None] * dot_second
    second -= np.einsum("a,aik,ajl->ijkl", masses, displacements, displacements)
    second -= np.einsum("a,ail,ajk->ijkl", masses, displacements, displacements)
    return inertia, zeta, first, second


def build_rovibrational_precursors(
    harmonic: HarmonicResult,
    rotor: EquilibriumRotor,
    context: ScientificContext,
    *,
    orthogonality_tolerance: float = 1e-10,
    maximum_harmonic_external_residual: float = 1e-5,
    inertia_degeneracy_relative_tolerance: float = 1e-8,
    mode_degeneracy_relative_tolerance: float = 1e-8,
) -> RovibrationalPrecursors:
    """Derive zeta and analytic dI/dQ, d²I/dQ² from retained real mode arrays.

    The supplied right-handed principal axes are retained, including arbitrary
    gauges in degenerate subspaces. The axes are not re-diagonalized at displaced
    geometries. Therefore these are tensor derivatives, not principal-moment
    derivatives. Positive projected curvature and an external-Hessian residual
    are necessary input checks, not proof of optimized/native method accuracy.
    """
    if not isinstance(harmonic, HarmonicResult) or not isinstance(
        rotor, EquilibriumRotor
    ):
        raise ValueError("Actual HarmonicResult and EquilibriumRotor inputs required.")
    if not isinstance(context, ScientificContext):
        raise ValueError("A validated ScientificContext is required.")
    # Revalidate a copy: existing context lists are mutable despite frozen fields.
    context = ScientificContext.model_validate(context.model_dump(mode="python"))
    declared_digests = (
        harmonic.source_digest,
        context.recipe_sha256,
        context.protocol_sha256,
        context.isotope_reference_sha256,
        *context.parent_artifact_sha256,
    )
    if "0" * 64 in declared_digests:
        raise ValueError(
            "Missing source/recipe/isotope/parent identity is not evidence."
        )
    parameters = (
        orthogonality_tolerance,
        maximum_harmonic_external_residual,
        inertia_degeneracy_relative_tolerance,
        mode_degeneracy_relative_tolerance,
    )
    if (
        any(isinstance(x, bool) or not np.isfinite(x) or x < 0 for x in parameters)
        or min(parameters[:2]) <= 0
    ):
        raise ValueError("Explicit finite nonnegative tolerances are required.")
    if harmonic.convention != _CONVENTION or harmonic.stationary_character != (
        "positive_definite_vibrational_hessian"
    ):
        raise ValueError(
            "The supported complete positive mass-weighted basis is required."
        )
    coordinates = finite_array(harmonic.coordinates_bohr)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Ordered Cartesian geometry requires [atoms,3] shape.")
    atoms = len(coordinates)
    masses_u = finite_array(harmonic.isotope_masses_u, (atoms,))
    if not atoms or np.any(masses_u <= 0):
        raise ValueError("Every atom requires a finite positive isotope mass.")
    masses = masses_u * ATOMIC_MASS_ELECTRON
    frequencies = finite_array(harmonic.frequencies_cm1)
    if frequencies.ndim != 1 or not len(frequencies) or np.any(frequencies <= 0):
        raise ValueError("At least one complete positive harmonic mode is required.")
    count = len(frequencies)
    modes = finite_array(harmonic.mass_weighted_modes, (3 * atoms, count))
    cartesian = finite_array(harmonic.cartesian_modes, (3 * atoms, count))
    angular = finite_array(harmonic.angular_frequencies_au, (count,))
    eigenvalues = finite_array(harmonic.eigenvalues_au, (count,))
    transform = finite_array(harmonic.dimensionless_to_cartesian, (3 * atoms, count))
    if (
        np.any(angular <= 0)
        or np.any(eigenvalues <= 0)
        or not np.allclose(frequencies, angular * HARTREE_CM1, rtol=1e-12, atol=1e-10)
        or not np.allclose(eigenvalues, angular**2, rtol=1e-12, atol=1e-15)
        or not np.allclose(
            cartesian * np.sqrt(np.repeat(masses, 3))[:, None],
            modes,
            rtol=0,
            atol=orthogonality_tolerance,
        )
        or not np.allclose(
            transform * np.sqrt(angular)[None, :],
            cartesian,
            rtol=0,
            atol=orthogonality_tolerance,
        )
    ):
        raise ValueError("Normal-Q/dimensionless transforms and frequencies disagree.")
    mode_residual = float(np.linalg.norm(modes.T @ modes - np.eye(count)))
    center = np.average(coordinates, axis=0, weights=masses)
    centered = coordinates - center
    external = []
    for axis in np.eye(3):
        external.append((np.tile(axis, (atoms, 1)) * np.sqrt(masses)[:, None]).ravel())
        external.append((np.cross(axis, centered) * np.sqrt(masses)[:, None]).ravel())
    external_basis, singular, _ = np.linalg.svd(
        np.asarray(external).T, full_matrices=True
    )
    rank = int(np.sum(singular > max(float(singular.max()) * 1e-10, 1e-13)))
    if harmonic.external_rank != rank or count != 3 * atoms - rank:
        raise ValueError("The mode count/external rank does not match this geometry.")
    modes_by_atom = modes.reshape(atoms, 3, count)
    translation_residual = float(
        np.linalg.norm(np.einsum("a,aik->ik", np.sqrt(masses), modes_by_atom))
        / np.sqrt(masses.sum())
    )
    rotational_sum = np.zeros((3, count))
    for atom in range(atoms):
        rotational_sum += (
            np.sqrt(masses[atom]) * np.cross(centered[atom], modes_by_atom[atom].T).T
        )
    radius = float(np.sqrt(np.sum(masses * np.sum(centered**2, axis=1))))
    if radius == 0:
        raise ValueError("A nonzero reference geometry is required.")
    rotational_residual = float(np.linalg.norm(rotational_sum) / radius)
    external_residual = float(np.linalg.norm(external_basis[:, :rank].T @ modes))
    hessian_residual = harmonic.external_residual_relative
    if (
        max(mode_residual, translation_residual, rotational_residual, external_residual)
        > orthogonality_tolerance
        or isinstance(hessian_residual, bool)
        or not np.isfinite(hessian_residual)
        or not 0 <= hessian_residual <= maximum_harmonic_external_residual
    ):
        raise ValueError(
            "Mass orthonormality, COM/Eckart or Hessian residual gate failed."
        )
    mode_digest = artifact_digest(
        {
            "mass_weighted_modes": modes,
            "isotope_masses_u": masses_u,
            "geometry_bohr": coordinates,
            "convention": harmonic.convention,
        }
    )
    geometry_digest = artifact_digest(
        {"geometry_bohr": coordinates, "masses_u": masses_u}
    )
    axes = finite_array(rotor.principal_axes_columns, (3, 3))
    if (
        not np.array_equal(masses_u, rotor.isotope_masses_u)
        or rotor.geometry_digest != geometry_digest
        or rotor.observable != "Be"
        or not np.allclose(axes.T @ axes, np.eye(3), rtol=0, atol=1e-10)
        or not np.isclose(np.linalg.det(axes), 1, rtol=0, atol=1e-10)
        or not np.array_equal(coordinates, context.geometry_bohr)
        or not np.array_equal(masses_u, context.isotope_masses_u)
        or context.mode_count != count
        or context.mode_basis_sha256 != mode_digest
        or context.frame_type != "principal_inertia"
        or not np.array_equal(axes, context.frame_axes_columns)
    ):
        raise ValueError(
            "Rotor, isotope, geometry, mode or right-handed frame identity differs."
        )
    positions = centered @ axes
    local_modes = np.einsum("aik,ij->ajk", modes_by_atom, axes)
    inertia, zeta, first, second = _geometric_tensors(positions, masses, local_modes)
    expected_moments = (
        finite_array(rotor.principal_moments_u_bohr2, (3,)) * ATOMIC_MASS_ELECTRON
    )
    scale = float(np.max(np.abs(inertia)))
    if (
        np.any(expected_moments < 0)
        or np.any(np.diff(expected_moments) < 0)
        or not np.allclose(
            inertia, np.diag(expected_moments), rtol=1e-10, atol=scale * 1e-12
        )
    ):
        raise ValueError("The retained axes/moments do not diagonalize actual inertia.")
    inertia_pairs = _pairs(expected_moments, inertia_degeneracy_relative_tolerance)
    mode_pairs = _pairs(frequencies, mode_degeneracy_relative_tolerance)
    zero_axes = tuple(
        index for index, moment in enumerate(expected_moments) if moment == 0
    )
    flags = _quality_flags(inertia_pairs, mode_pairs, zero_axes)
    identity = PrecursorIdentity(
        input_context_sha256=artifact_digest(context.model_dump(mode="json")),
        input_evidence_class=context.evidence_class,
        symbols=tuple(context.symbols),
        atom_ids=tuple(context.atom_ids),
        atom_id_policy=context.atom_id_policy,
        isotope_numbers=tuple(context.isotope_numbers),
        isotope_masses_u=tuple(context.isotope_masses_u),
        isotope_selection_policies=tuple(context.isotope_selection_policies),
        isotope_reference_sha256=context.isotope_reference_sha256,
        charge=context.charge,
        multiplicity=context.multiplicity,
        geometry_sha256=context.geometry_sha256,
        harmonic_source_sha256=harmonic.source_digest,
        harmonic_declaration_sha256=_harmonic_declaration_digest(
            harmonic.source_digest, mode_digest, frequencies, float(hessian_residual)
        ),
        mode_basis_sha256=mode_digest,
        mode_order=tuple(context.mode_order),
        frame_sha256=context.frame_sha256,
        equilibrium_rotor_geometry_sha256=geometry_digest,
        recipe_sha256=context.recipe_sha256,
        protocol_sha256=context.protocol_sha256,
        parent_artifact_sha256=tuple(context.parent_artifact_sha256),
        algorithm_source_sha256=_SOURCE_SHA256,
        constants_sha256=artifact_digest(constants_provenance()),
    )
    unavailable = tuple(
        UnavailableRovibrationalProduct(
            name=name,
            reason=(
                "Geometric mode/inertia precursors do not implement "
                "this physical model."
            ),
        )
        for name in (
            "rovibrational_kinetic_operator",
            "rotation_vibration_alpha",
            "vibration_corrected_constants",
            "centrifugal_distortion",
            "full_rovibrational_vpt2",
        )
    )
    return RovibrationalPrecursors(
        identity=identity,
        isotope_mass_u_to_electron_mass=ATOMIC_MASS_ELECTRON,
        input_geometry_bohr=_matrix(coordinates),
        input_mass_weighted_modes=_matrix(modes),
        center_of_mass_bohr=(float(center[0]), float(center[1]), float(center[2])),
        principal_axes_columns=_matrix(axes),
        centered_geometry_principal_bohr=_matrix(positions),
        mass_weighted_modes_principal=_tensor3(local_modes),
        harmonic_frequencies_cm1=tuple(float(x) for x in frequencies),
        inertia_electron_mass_bohr2=_matrix(inertia),
        principal_moments_electron_mass_bohr2=(
            float(expected_moments[0]),
            float(expected_moments[1]),
            float(expected_moments[2]),
        ),
        coriolis_zeta=_tensor3(zeta),
        inertia_first_derivative=_tensor3(first),
        inertia_second_derivative=_tensor4(second),
        mode_orthonormality_residual=mode_residual,
        translation_residual=translation_residual,
        rotational_eckart_residual=rotational_residual,
        external_subspace_residual=external_residual,
        harmonic_external_residual=float(hessian_residual),
        orthogonality_tolerance=orthogonality_tolerance,
        maximum_harmonic_external_residual=maximum_harmonic_external_residual,
        inertia_degeneracy_relative_tolerance=inertia_degeneracy_relative_tolerance,
        mode_degeneracy_relative_tolerance=mode_degeneracy_relative_tolerance,
        degenerate_inertia_axis_pairs=inertia_pairs,
        degenerate_mode_pairs=mode_pairs,
        zero_inertia_axes=zero_axes,
        quality_flags=tuple(flags),
        unavailable_products=unavailable,
    )
