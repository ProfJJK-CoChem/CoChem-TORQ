"""Bounded variational nuclear motion in a retained rectilinear Eckart chart.

This is a Dirichlet finite-domain Laplace--Beltrami model, not a Watson
effective-Hamiltonian reduction or a semirigid VPT2 implementation. Atomic units
are used throughout. See docs/development/rovibrational_solver.md for the
coordinate metric, body-fixed generator convention and domain restrictions.
"""

from __future__ import annotations

import json
from hashlib import sha256
from itertools import product
from math import prod, sqrt
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.linalg import eigh
from typing_extensions import Self

from ..units import HARTREE_CM1, HARTREE_JOULE, h
from .harmonic import artifact_digest, finite_array
from .results import ForceFieldData
from .rovibrational import RovibrationalPrecursors

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
_SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


class _Immutable(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )


class StationaryReference(_Immutable):
    """Actual gradient declaration bound to a retained reference artifact.

    The workflow authenticates the native bytes. This record never establishes
    stationarity from a Hessian or silently replaces an absent gradient by zero.
    A finite accepted residual remains part of the polynomial-model error.
    """

    source_artifact_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    geometry_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    evidence_class: Literal["engine_calculation", "mathematical_model"]
    gradient_hartree_bohr: tuple[tuple[float, float, float], ...]
    maximum_allowed_gradient_hartree_bohr: float = Field(gt=0)
    reference_gradient_policy: Literal[
        "actual gradient within declared tolerance; linear PES term omitted"
    ] = "actual gradient within declared tolerance; linear PES term omitted"

    @model_validator(mode="after")
    def gradient_gate(self) -> Self:
        values = np.asarray(self.gradient_hartree_bohr)
        if (
            not len(values)
            or values.ndim != 2
            or values.shape[1] != 3
            or np.max(np.abs(values)) > self.maximum_allowed_gradient_hartree_bohr
            or self.source_artifact_sha256 == "0" * 64
        ):
            raise ValueError("Actual reference gradient fails its stationarity gate.")
        return self


class VariationalProtocol(_Immutable):
    """Explicit box, self-adjoint boundary condition and computational budget."""

    half_widths_dimensionless: tuple[float, ...]
    sine_basis_counts: tuple[int, ...]
    quadrature_orders: tuple[int, ...]
    angular_momenta: tuple[int, ...] = (0, 1)
    retained_states_per_j: int = Field(ge=1, le=100)
    maximum_matrix_dimension: int = Field(ge=1, le=2500)
    maximum_quadrature_nodes: int = Field(ge=1, le=250000)
    maximum_workspace_bytes: int = Field(default=268435456, ge=1, le=68719476736)
    boundary_condition: Literal["Dirichlet on every face of the Q box"] = (
        "Dirichlet on every face of the Q box"
    )
    quadrature_rule: Literal["tensor product Gauss-Legendre"] = (
        "tensor product Gauss-Legendre"
    )

    @model_validator(mode="after")
    def bounded_domain(self) -> Self:
        count = len(self.half_widths_dimensionless)
        if (
            not count
            or count != len(self.sine_basis_counts)
            or count != len(self.quadrature_orders)
            or min(self.half_widths_dimensionless) <= 0
            or min(self.sine_basis_counts) <= 0
            or any(
                q < 2 * b + 5
                for b, q in zip(self.sine_basis_counts, self.quadrature_orders)
            )
            or not self.angular_momenta
            or self.angular_momenta != tuple(sorted(set(self.angular_momenta)))
            or self.angular_momenta[0] != 0
            or self.angular_momenta[-1] > 5
        ):
            raise ValueError(
                "Positive complete box/basis counts, quadrature >= 2*basis+5, "
                "and distinct J=0,... requested sectors (maximum J=5) required."
            )
        if (
            prod(self.quadrature_orders) > self.maximum_quadrature_nodes
            or prod(self.sine_basis_counts) * (2 * self.angular_momenta[-1] + 1)
            > self.maximum_matrix_dimension
        ):
            raise ValueError("The requested finite model exceeds its explicit budget.")
        basis = prod(self.sine_basis_counts)
        nodes = prod(self.quadrature_orders)
        dimension = basis * (2 * self.angular_momenta[-1] + 1)
        # Conservative dense-array estimate, including derivative/grid arrays,
        # matrix components and LAPACK work. This is a preflight upper budget,
        # not a promise about external BLAS/library overhead or host capacity.
        estimated_workspace = 2 * (
            (count + 5) * nodes * basis * 8
            + 20 * basis**2 * 8
            + 8 * dimension**2 * 16
            + 20 * nodes * 8
        )
        if estimated_workspace > self.maximum_workspace_bytes:
            raise ValueError("The finite model exceeds its dense workspace budget.")
        return self


class KineticMetric(_Immutable):
    """Metric values, not a kinetic-operator or effective-constant result."""

    q_dimensionless: tuple[float, ...]
    inertia_electron_mass_bohr2: tuple[tuple[float, ...], ...]
    rotational_vibrational_cross_metric: tuple[tuple[float, ...], ...]
    vibrational_metric_diagonal: tuple[float, ...]
    rotational_schur_complement: tuple[tuple[float, ...], ...]
    relative_volume_density: float = Field(gt=0)


class VariationalSector(_Immutable):
    J: int = Field(ge=0, le=5)
    matrix_dimension: int = Field(ge=1)
    energies_hartree_above_electronic_reference: tuple[float, ...]
    primitive_vibration_expectations_hartree: tuple[float, ...]
    metric_induced_vibration_expectations_hartree: tuple[float, ...]
    rotation_expectations_hartree: tuple[float, ...]
    coriolis_expectations_hartree: tuple[float, ...]
    potential_expectations_hartree: tuple[float, ...]
    hamiltonian_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    overlap_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    coefficient_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    maximum_eigenpair_relative_residual: float = Field(ge=0)
    overlap_orthonormality_residual: float = Field(ge=0)
    state_labels: Literal["J and sorted eigenstate index; unassigned vibrations"] = (
        "J and sorted eigenstate index; unassigned vibrations"
    )
    lab_m_degeneracy: int = Field(ge=1)
    matrix_artifact_name: str | None = None
    matrix_artifact_sha256: str | None = Field(default=None, pattern="^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def expectation_identity(self) -> Self:
        energy = np.asarray(self.energies_hartree_above_electronic_reference)
        contributions = (
            self.primitive_vibration_expectations_hartree,
            self.metric_induced_vibration_expectations_hartree,
            self.rotation_expectations_hartree,
            self.coriolis_expectations_hartree,
            self.potential_expectations_hartree,
        )
        if (
            not len(energy)
            or len(energy) > self.matrix_dimension
            or np.any(np.diff(energy) < 0)
            or any(len(values) != len(energy) for values in contributions)
            or self.lab_m_degeneracy != 2 * self.J + 1
            or (self.matrix_artifact_name is None)
            != (self.matrix_artifact_sha256 is None)
            or not np.allclose(
                np.sum(contributions, axis=0), energy, rtol=1e-10, atol=1e-12
            )
        ):
            raise ValueError("Sector energies/contributions/degeneracy disagree.")
        return self


class RovibrationalVariationalResult(_Immutable):
    schema_version: Literal["cochem.torq.local-rovibrational-variational/1"] = (
        "cochem.torq.local-rovibrational-variational/1"
    )
    identity: Literal[
        "bounded rectilinear Eckart Laplace-Beltrami variational model"
    ] = "bounded rectilinear Eckart Laplace-Beltrami variational model"
    precursor_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    precursor_context_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    force_field_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    force_field_context_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    native_parent_artifact_sha256: tuple[str, ...]
    algorithm_source_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    protocol: VariationalProtocol
    stationary_reference: StationaryReference
    electronic_reference_energy_hartree: float
    normal_coordinate_unit: Literal["q=sqrt(omega)*Q; dimensionless"] = (
        "q=sqrt(omega)*Q; dimensionless"
    )
    body_fixed_commutator: Literal["[Ja,Jb]=-i epsilon_abc Jc; hbar=1"] = (
        "[Ja,Jb]=-i epsilon_abc Jc; hbar=1"
    )
    volume_convention: Literal[
        "sqrt(det(I-C*D^-1*C.T)/det(Ie)); constant det(D) cancels"
    ] = "sqrt(det(I-C*D^-1*C.T)/det(Ie)); constant det(D) cancels"
    uniform_schur_eigenvalue_lower_bound: float = Field(gt=0)
    minimum_sampled_schur_eigenvalue: float = Field(gt=0)
    sectors: tuple[VariationalSector, ...]
    basis_convergence: Literal["requires separate comparison"] = (
        "requires separate comparison"
    )
    quadrature_convergence: Literal["requires separate comparison"] = (
        "requires separate comparison"
    )
    domain_convergence: Literal["requires separate comparison"] = (
        "requires separate comparison"
    )
    resonance_treatment: Literal[
        "finite-basis couplings diagonalized; no perturbative denominators"
    ] = "finite-basis couplings diagonalized; no perturbative denominators"
    effective_watson_constants: None = None
    rotation_vibration_alpha: None = None
    full_semirigid_vpt2: Literal[False] = False
    matrix_artifact_policy: Literal[
        "matrix archives retained", "matrix archives unavailable"
    ]
    nuclear_spin_restrictions_applied: Literal[False] = False
    independent_scientific_qualification: Literal[False] = False
    identification_ready: Literal[False] = False
    limitations: tuple[str, ...]
    content_sha256: str = Field(pattern="^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def result_binding(self) -> Self:
        if (
            tuple(sector.J for sector in self.sectors) != self.protocol.angular_momenta
            or self.minimum_sampled_schur_eigenvalue
            < self.uniform_schur_eigenvalue_lower_bound * (1 - 1e-10)
            or any(
                (sector.matrix_artifact_name is not None)
                != (self.matrix_artifact_policy == "matrix archives retained")
                for sector in self.sectors
            )
            or self.content_sha256
            != artifact_digest(self.model_dump(mode="json", exclude={"content_sha256"}))
        ):
            raise ValueError(
                "Result sectors, metric certificate or content binding fail."
            )
        return self


class VariationalComparison(_Immutable):
    comparison: Literal["basis", "quadrature", "domain"]
    first_result_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    second_result_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    maximum_compared_energy_difference_cm1: float = Field(ge=0)
    threshold_cm1: float = Field(gt=0)
    passed_for_compared_sorted_levels: bool
    common_state_counts_by_j: tuple[tuple[int, int], ...]
    independent_scientific_qualification: Literal[False] = False
    comparison_scope: Literal[
        "same-J sorted low eigenenergies; no vibrational assignment or error bound"
    ] = "same-J sorted low eigenenergies; no vibrational assignment or error bound"


class RovibrationalEnergyDifference(_Immutable):
    result_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    upper_state: tuple[int, int]
    lower_state: tuple[int, int]
    upper_sector_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    lower_sector_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    energy_difference_hartree: float
    frequency_cm1: float
    frequency_mhz: float
    identity: Literal["unassigned finite-model eigenenergy difference"] = (
        "unassigned finite-model eigenenergy difference"
    )
    transition_allowed: None = None
    dipole_strength: None = None
    identification_ready: Literal[False] = False


def body_fixed_angular_momentum(J: int) -> tuple[ComplexArray, ...]:  # noqa: N803
    """Hermitian integer-J generators in K=-J,...,+J, anomalous commutator."""
    if type(J) is not int or not 0 <= J <= 5:
        raise ValueError("An integer rotational angular momentum 0 <= J <= 5 required.")
    size = 2 * J + 1
    raising = np.zeros((size, size), dtype=np.complex128)
    for column, projection in enumerate(range(-J, J)):
        raising[column + 1, column] = sqrt(J * (J + 1) - projection * (projection + 1))
    x = (raising + raising.T) / 2
    # Complex conjugation of ordinary space-fixed generators gives the body
    # commutator. The representation and cross term use this same gauge.
    y = np.conjugate((raising - raising.T) / (2j))
    z = np.diag(np.arange(-J, J + 1)).astype(np.complex128)
    return x, y, z


def _validated_inputs(
    precursors: RovibrationalPrecursors, force_field: ForceFieldData
) -> tuple[RovibrationalPrecursors, ForceFieldData]:
    if not isinstance(precursors, RovibrationalPrecursors) or not isinstance(
        force_field, ForceFieldData
    ):
        raise ValueError("Actual validated precursors and ForceFieldData required.")
    precursors = RovibrationalPrecursors.model_validate_json(
        precursors.model_dump_json()
    )
    force_field = ForceFieldData.model_validate_json(force_field.model_dump_json())
    context = force_field.scientific_context
    identity = precursors.identity
    required = (
        (tuple(context.symbols), identity.symbols),
        (tuple(context.atom_ids), identity.atom_ids),
        (context.atom_id_policy, identity.atom_id_policy),
        (tuple(context.isotope_numbers), identity.isotope_numbers),
        (tuple(context.isotope_masses_u), identity.isotope_masses_u),
        (
            tuple(context.isotope_selection_policies),
            identity.isotope_selection_policies,
        ),
        (context.isotope_reference_sha256, identity.isotope_reference_sha256),
        (context.geometry_sha256, identity.geometry_sha256),
        (context.charge, identity.charge),
        (context.multiplicity, identity.multiplicity),
        (context.mode_basis_sha256, identity.mode_basis_sha256),
        (tuple(context.mode_order), identity.mode_order),
        (context.frame_sha256, identity.frame_sha256),
        (context.recipe_sha256, identity.recipe_sha256),
        (context.evidence_class, identity.input_evidence_class),
        (force_field.harmonic_source_digest, identity.harmonic_source_sha256),
    )
    if (
        any(first != second for first, second in required)
        or not force_field.derivative_converged
        or not np.allclose(
            force_field.frequencies_hartree,
            np.asarray(precursors.harmonic_frequencies_cm1) / HARTREE_CM1,
            rtol=1e-12,
            atol=1e-15,
        )
        or precursors.zero_inertia_axes
    ):
        raise ValueError(
            "Force field/precursor geometry, isotope, modes, state, method or "
            "convergence differs; this chart requires a nonlinear reference."
        )
    return precursors, force_field


def _metric_arrays(
    precursors: RovibrationalPrecursors, frequencies: RealArray, q: RealArray
) -> tuple[RealArray, RealArray, RealArray]:
    Q = q / np.sqrt(frequencies)  # noqa: N806
    inertia = np.asarray(precursors.inertia_electron_mass_bohr2).copy()
    inertia += np.einsum("abk,k->ab", precursors.inertia_first_derivative, Q)
    inertia += 0.5 * np.einsum(
        "abkl,k,l->ab", precursors.inertia_second_derivative, Q, Q
    )
    cross = np.einsum("aji,j->ai", precursors.coriolis_zeta, Q)
    cross /= np.sqrt(frequencies)[None, :]
    schur = inertia - (cross * frequencies[None, :]) @ cross.T
    return inertia, cross, schur


def local_kinetic_metric(
    precursors: RovibrationalPrecursors, q_dimensionless: tuple[float, ...]
) -> KineticMetric:
    """Evaluate exact geometric metric in the declared local chart, reject poles."""
    if not isinstance(precursors, RovibrationalPrecursors):
        raise ValueError("Validated geometric precursors required.")
    precursors = RovibrationalPrecursors.model_validate_json(
        precursors.model_dump_json()
    )
    frequencies = np.asarray(precursors.harmonic_frequencies_cm1) / HARTREE_CM1
    q = finite_array(q_dimensionless, frequencies.shape)
    inertia, cross, schur = _metric_arrays(precursors, frequencies, q)
    if precursors.zero_inertia_axes or np.min(np.linalg.eigvalsh(schur)) <= 0:
        raise ValueError("A nonlinear nonsingular local Eckart chart is required.")
    return KineticMetric(
        q_dimensionless=tuple(float(x) for x in q),
        inertia_electron_mass_bohr2=_rows(inertia),
        rotational_vibrational_cross_metric=_rows(cross),
        vibrational_metric_diagonal=tuple(float(x) for x in 1 / frequencies),
        rotational_schur_complement=_rows(schur),
        relative_volume_density=float(
            sqrt(
                np.linalg.det(schur)
                / np.linalg.det(np.asarray(precursors.inertia_electron_mass_bohr2))
            )
        ),
    )


def _rows(array: RealArray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in array)


def _uniform_metric_certificate(
    precursors: RovibrationalPrecursors,
    frequencies: RealArray,
    widths: RealArray,
) -> float:
    """Weyl bound on projected rotation-map singular values over the WHOLE box."""
    modes = np.asarray(precursors.mass_weighted_modes_principal)
    count = modes.shape[-1]
    flattened = modes.reshape(-1, count)
    projector = np.eye(flattened.shape[0]) - flattened @ flattened.T
    inertia = np.asarray(precursors.inertia_electron_mass_bohr2)
    original_smallest = sqrt(float(np.min(np.linalg.eigvalsh(inertia))))
    variation_bound = 0.0
    for mode in range(count):
        derivative_rotation = np.stack(
            [np.cross(axis, modes[:, :, mode]).ravel() for axis in np.eye(3)],
            axis=1,
        )
        variation_bound += (
            widths[mode]
            / sqrt(frequencies[mode])
            * float(np.linalg.norm(projector @ derivative_rotation, ord=2))
        )
    remainder = original_smallest - variation_bound
    if remainder <= original_smallest * 1e-8:
        raise ValueError(
            "The chosen box has no positive uniform local-chart certificate; "
            "reduce its widths or use a different coordinate chart. "
            "Nonsingular quadrature samples alone cannot authorize the domain."
        )
    return remainder**2


def _sine_grid(
    protocol: VariationalProtocol,
) -> tuple[RealArray, RealArray, RealArray, tuple[RealArray, ...]]:
    nodes, weights, values, derivatives = [], [], [], []
    for width, count, order in zip(
        protocol.half_widths_dimensionless,
        protocol.sine_basis_counts,
        protocol.quadrature_orders,
    ):
        x, w = np.polynomial.legendre.leggauss(order)
        q = width * x
        indices = np.arange(1, count + 1)
        argument = (q[:, None] + width) * indices[None, :] * np.pi / (2 * width)
        nodes.append(q)
        weights.append(width * w)
        values.append(np.sin(argument) / sqrt(width))
        derivatives.append(
            np.cos(argument) * indices[None, :] * np.pi / (2 * width**1.5)
        )
    node_indices = np.asarray(
        tuple(product(*(range(order) for order in protocol.quadrature_orders)))
    )
    basis_indices = np.asarray(
        tuple(product(*(range(count) for count in protocol.sine_basis_counts)))
    )
    q = np.column_stack([x[node_indices[:, i]] for i, x in enumerate(nodes)])
    weight = np.prod(
        np.column_stack([w[node_indices[:, i]] for i, w in enumerate(weights)]),
        axis=1,
    )
    basis = np.ones((len(q), len(basis_indices)))
    factors = []
    for mode, value in enumerate(values):
        factors.append(value[node_indices[:, mode, None], basis_indices[None, :, mode]])
        basis *= factors[-1]
    gradient = []
    for mode in range(len(nodes)):
        term = derivatives[mode][
            node_indices[:, mode, None], basis_indices[None, :, mode]
        ].copy()
        for other, factor in enumerate(factors):
            if other != mode:
                term *= factor
        gradient.append(term)
    return q, weight, basis, tuple(gradient)


def _polynomial_potential(force_field: ForceFieldData, q: RealArray) -> RealArray:
    frequency = np.asarray(force_field.frequencies_hartree)
    values = 0.5 * np.einsum("i,ni,ni->n", frequency, q, q)
    values += np.einsum("ijk,ni,nj,nk->n", force_field.cubic_hartree, q, q, q) / 6
    values += (
        np.einsum("ijkl,ni,nj,nk,nl->n", force_field.quartic_hartree, q, q, q, q) / 24
    )
    return np.asarray(values, dtype=float)


def _matrix_digest(matrix: ComplexArray | RealArray) -> str:
    return artifact_digest({"real": matrix.real, "imaginary": matrix.imag})


def solve_local_rovibrational_model(
    precursors: RovibrationalPrecursors,
    force_field: ForceFieldData,
    protocol: VariationalProtocol,
    *,
    stationary_reference: StationaryReference,
    artifact_directory: Path | None = None,
) -> RovibrationalVariationalResult:
    """Diagonalize a genuine local nuclear-motion model for each requested J.

    The matrix weak form is positive for kinetic energy. No pseudopotential is
    added: the coordinate-volume measure remains in both H and its overlap S.
    Resonant states within the finite basis are coupled directly; absent states
    remain a basis/domain limitation, never an artificially shifted denominator.
    """
    precursors, force_field = _validated_inputs(precursors, force_field)
    if not isinstance(stationary_reference, StationaryReference):
        raise ValueError("Actual source-bound stationary-reference gradient required.")
    stationary_reference = StationaryReference.model_validate_json(
        stationary_reference.model_dump_json()
    )
    if (
        stationary_reference.source_artifact_sha256
        not in precursors.identity.parent_artifact_sha256
        or stationary_reference.geometry_sha256 != precursors.identity.geometry_sha256
        or stationary_reference.evidence_class
        != precursors.identity.input_evidence_class
        or len(stationary_reference.gradient_hartree_bohr)
        != len(precursors.identity.symbols)
    ):
        raise ValueError(
            "Reference gradient/source/geometry/evidence identity differs."
        )
    if not isinstance(protocol, VariationalProtocol):
        raise ValueError("An explicit validated finite-model protocol is required.")
    protocol = VariationalProtocol.model_validate_json(protocol.model_dump_json())
    frequencies = np.asarray(force_field.frequencies_hartree)
    if len(protocol.half_widths_dimensionless) != len(frequencies):
        raise ValueError("The box/basis must cover every retained normal mode.")
    widths = np.asarray(protocol.half_widths_dimensionless)
    certificate = _uniform_metric_certificate(precursors, frequencies, widths)
    if artifact_directory is not None:
        if not isinstance(artifact_directory, Path):
            raise ValueError("A fresh explicit Path is required for matrix artifacts.")
        artifact_directory.mkdir(parents=True, exist_ok=False)
    q, integration_weights, basis, gradient = _sine_grid(protocol)
    schurs, crosses = [], []
    for displacement in q:
        _, cross, schur = _metric_arrays(precursors, frequencies, displacement)
        schurs.append(schur)
        crosses.append(cross)
    schur_array = np.asarray(schurs)
    cross_array = np.asarray(crosses)
    sampled_minimum = float(np.min(np.linalg.eigvalsh(schur_array)))
    if sampled_minimum < certificate * (1 - 1e-10):
        raise ValueError("Actual metric violates the uniform chart certificate.")
    inverse = np.linalg.inv(schur_array)
    original_det = np.linalg.det(np.asarray(precursors.inertia_electron_mass_bohr2))
    weight = integration_weights * np.sqrt(np.linalg.det(schur_array) / original_det)
    overlap = basis.T @ (weight[:, None] * basis)
    primitive = np.zeros_like(overlap)
    cross_gradient = np.zeros((len(q), 3, basis.shape[1]))
    for mode, derivative in enumerate(gradient):
        primitive += (
            0.5 * frequencies[mode] * (derivative.T @ (weight[:, None] * derivative))
        )
        cross_gradient += (
            cross_array[:, :, mode, None] * frequencies[mode] * derivative[:, None, :]
        )
    induced = np.zeros_like(overlap)
    angular_coefficients = []
    mixed_coefficients = []
    for first in range(3):
        angular_row = []
        mixed = np.zeros_like(overlap)
        for second in range(3):
            multiplier = weight * inverse[:, first, second]
            angular_row.append(basis.T @ (multiplier[:, None] * basis))
            mixed += basis.T @ (multiplier[:, None] * cross_gradient[:, second, :])
            induced += (
                0.5
                * cross_gradient[:, first, :].T
                @ (multiplier[:, None] * cross_gradient[:, second, :])
            )
        angular_coefficients.append(angular_row)
        mixed_coefficients.append(mixed)
    potential_values = _polynomial_potential(force_field, q)
    potential = basis.T @ (weight[:, None] * potential_values[:, None] * basis)
    sectors = []
    for angular_momentum in protocol.angular_momenta:
        spin_size = 2 * angular_momentum + 1
        spin_identity = np.eye(spin_size)
        operators = body_fixed_angular_momentum(angular_momentum)
        rotation = np.zeros((overlap.shape[0] * spin_size,) * 2, dtype=np.complex128)
        coriolis = np.zeros_like(rotation)
        for first in range(3):
            coriolis += 0.5j * np.kron(
                mixed_coefficients[first] - mixed_coefficients[first].T,
                operators[first],
            )
            for second in range(3):
                rotation += 0.5 * np.kron(
                    angular_coefficients[first][second],
                    operators[first] @ operators[second],
                )
        components = (
            np.kron(primitive, spin_identity),
            np.kron(induced, spin_identity),
            rotation,
            coriolis,
            np.kron(potential, spin_identity),
        )
        hamiltonian = np.zeros_like(rotation)
        for component in components:
            hamiltonian += component
        full_overlap = np.kron(overlap, spin_identity)
        for component in (*components, hamiltonian):
            if not np.allclose(component, component.conj().T, rtol=1e-11, atol=1e-13):
                raise ValueError("The actual weak-form Hamiltonian is not Hermitian.")
        retained = min(protocol.retained_states_per_j, len(hamiltonian))
        energies, coefficients = eigh(
            hamiltonian,
            full_overlap,
            subset_by_index=(0, retained - 1),
            driver="gvx",
            check_finite=True,
        )
        residual = (
            hamiltonian @ coefficients
            - (full_overlap @ coefficients) * energies[None, :]
        )
        denominator = (
            np.linalg.norm(hamiltonian)
            + np.abs(energies) * np.linalg.norm(full_overlap)
        ) * np.linalg.norm(coefficients, axis=0)
        relative_residual = float(
            np.max(np.linalg.norm(residual, axis=0) / np.maximum(denominator, 1e-15))
        )
        normalization = float(
            np.linalg.norm(
                coefficients.conj().T @ full_overlap @ coefficients - np.eye(retained)
            )
        )
        expectations = tuple(
            tuple(
                float(value)
                for value in np.real(
                    np.einsum("ni,ni->i", coefficients.conj(), component @ coefficients)
                )
            )
            for component in components
        )
        if relative_residual > 1e-9 or normalization > 1e-9:
            raise ValueError("Generalized eigenpair/normalization checks failed.")
        artifact_name, artifact_sha = None, None
        if artifact_directory is not None:
            artifact_name = f"J-{angular_momentum:03d}.npz"
            artifact_path = artifact_directory / artifact_name
            with artifact_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    hamiltonian_hartree=hamiltonian,
                    overlap=full_overlap,
                    eigenenergies_hartree=energies,
                    eigenvector_columns=coefficients,
                    primitive_vibration_hartree=components[0],
                    metric_induced_vibration_hartree=components[1],
                    rotation_hartree=components[2],
                    coriolis_hartree=components[3],
                    potential_hartree=components[4],
                    metadata_json=np.asarray(
                        json.dumps(
                            {
                                "precursor_sha256": artifact_digest(
                                    precursors.model_dump(mode="json")
                                ),
                                "force_field_sha256": force_field.source_digest,
                                "algorithm_source_sha256": _SOURCE_SHA256,
                                "protocol": protocol.model_dump(mode="json"),
                                "stationary_reference": stationary_reference.model_dump(
                                    mode="json"
                                ),
                                "J": angular_momentum,
                            },
                            sort_keys=True,
                        )
                    ),
                )
            artifact_sha = sha256(artifact_path.read_bytes()).hexdigest()
        sectors.append(
            VariationalSector(
                J=angular_momentum,
                matrix_dimension=len(hamiltonian),
                energies_hartree_above_electronic_reference=tuple(
                    float(x) for x in energies
                ),
                primitive_vibration_expectations_hartree=expectations[0],
                metric_induced_vibration_expectations_hartree=expectations[1],
                rotation_expectations_hartree=expectations[2],
                coriolis_expectations_hartree=expectations[3],
                potential_expectations_hartree=expectations[4],
                hamiltonian_sha256=_matrix_digest(hamiltonian),
                overlap_sha256=_matrix_digest(full_overlap),
                coefficient_sha256=_matrix_digest(coefficients),
                maximum_eigenpair_relative_residual=relative_residual,
                overlap_orthonormality_residual=normalization,
                lab_m_degeneracy=spin_size,
                matrix_artifact_name=artifact_name,
                matrix_artifact_sha256=artifact_sha,
            )
        )
    context = force_field.scientific_context
    data = {
        "precursor_sha256": artifact_digest(precursors.model_dump(mode="json")),
        "precursor_context_sha256": precursors.identity.input_context_sha256,
        "force_field_sha256": force_field.source_digest,
        "force_field_context_sha256": artifact_digest(context.model_dump(mode="json")),
        "native_parent_artifact_sha256": tuple(
            sorted(
                set(precursors.identity.parent_artifact_sha256)
                | set(context.parent_artifact_sha256)
                | {
                    record.source_artifact_sha256
                    for record in force_field.displacement_records
                    if record.source_artifact_sha256 is not None
                }
            )
        ),
        "algorithm_source_sha256": _SOURCE_SHA256,
        "protocol": protocol,
        "stationary_reference": stationary_reference,
        "matrix_artifact_policy": (
            "matrix archives retained"
            if artifact_directory is not None
            else "matrix archives unavailable"
        ),
        "electronic_reference_energy_hartree": force_field.reference_energy_hartree,
        "uniform_schur_eigenvalue_lower_bound": certificate,
        "minimum_sampled_schur_eigenvalue": sampled_minimum,
        "sectors": tuple(sectors),
        "limitations": (
            "finite local Dirichlet coordinate box; domain convergence unestablished",
            "finite sine basis/quadrature; independent convergence required",
            "quartic rectilinear electronic PES; omitted fifth and higher derivatives",
            "no full semirigid VPT2, Watson constants, alpha or calibrated B0",
            "no large-amplitude chart transitions, tunneling or spin restrictions",
            "sorted eigenlevels unassigned; no dipole surface/transition intensities",
            "input declarations bound; caller authenticates retained native bytes",
            "linear PES term omitted with retained actual reference-gradient tolerance",
        ),
    }
    # Build defaults without bypassing final scientific validation. The final
    # digest covers defaults too, so changing a qualification cannot go unnoticed.
    provisional = RovibrationalVariationalResult.model_construct(
        _fields_set=set(data), **data, content_sha256="0" * 64
    )
    payload = provisional.model_dump(mode="json", exclude={"content_sha256"})
    return RovibrationalVariationalResult.model_validate_json(
        json.dumps({**payload, "content_sha256": artifact_digest(payload)})
    )


def compare_local_variational_results(
    first: RovibrationalVariationalResult,
    second: RovibrationalVariationalResult,
    *,
    comparison: Literal["basis", "quadrature", "domain"],
    threshold_cm1: float,
) -> VariationalComparison:
    """Compare exactly one changed finite-model parameter family, retain failures."""
    if comparison not in ("basis", "quadrature", "domain"):
        raise ValueError("Choose an explicit basis, quadrature or domain comparison.")
    if (
        isinstance(threshold_cm1, bool)
        or not np.isfinite(threshold_cm1)
        or threshold_cm1 <= 0
    ):
        raise ValueError(
            "An explicit finite positive comparison threshold is required."
        )
    first = RovibrationalVariationalResult.model_validate_json(first.model_dump_json())
    second = RovibrationalVariationalResult.model_validate_json(
        second.model_dump_json()
    )
    if any(
        getattr(first, name) != getattr(second, name)
        for name in (
            "precursor_sha256",
            "force_field_sha256",
            "algorithm_source_sha256",
            "precursor_context_sha256",
            "force_field_context_sha256",
            "stationary_reference",
        )
    ):
        raise ValueError(
            "Both comparisons must bind the same scientific inputs/algorithm."
        )
    parameter = {
        "basis": "sine_basis_counts",
        "quadrature": "quadrature_orders",
        "domain": "half_widths_dimensionless",
    }[comparison]
    for name in (
        "sine_basis_counts",
        "quadrature_orders",
        "half_widths_dimensionless",
        "angular_momenta",
        "boundary_condition",
        "quadrature_rule",
    ):
        old, new = getattr(first.protocol, name), getattr(second.protocol, name)
        if (name == parameter and old == new) or (name != parameter and old != new):
            raise ValueError(
                "Change exactly the declared parameter family per comparison."
            )
    differences = []
    counts = []
    for old_sector, new_sector in zip(first.sectors, second.sectors):
        old = np.asarray(old_sector.energies_hartree_above_electronic_reference)
        new = np.asarray(new_sector.energies_hartree_above_electronic_reference)
        count = min(len(old), len(new))
        counts.append((old_sector.J, count))
        differences.extend(np.abs(old[:count] - new[:count]) * HARTREE_CM1)
    maximum = float(max(differences))
    return VariationalComparison(
        comparison=comparison,
        first_result_sha256=first.content_sha256,
        second_result_sha256=second.content_sha256,
        maximum_compared_energy_difference_cm1=maximum,
        threshold_cm1=float(threshold_cm1),
        passed_for_compared_sorted_levels=maximum <= threshold_cm1,
        common_state_counts_by_j=tuple(counts),
    )


def rovibrational_energy_difference(
    result: RovibrationalVariationalResult,
    *,
    upper_state: tuple[int, int],
    lower_state: tuple[int, int],
) -> RovibrationalEnergyDifference:
    """Return a source-bound level difference without inventing an allowed line."""
    result = RovibrationalVariationalResult.model_validate_json(
        result.model_dump_json()
    )
    selected = []
    for state in (upper_state, lower_state):
        if len(state) != 2 or any(
            type(value) is not int or value < 0 for value in state
        ):
            raise ValueError(
                "State identity is an integer (J, sorted eigenstate index)."
            )
        sector = next((value for value in result.sectors if value.J == state[0]), None)
        if sector is None or state[1] >= len(
            sector.energies_hartree_above_electronic_reference
        ):
            raise ValueError(
                "The actual retained calculation does not contain this state."
            )
        selected.append(sector)
    difference = (
        selected[0].energies_hartree_above_electronic_reference[upper_state[1]]
        - selected[1].energies_hartree_above_electronic_reference[lower_state[1]]
    )
    return RovibrationalEnergyDifference(
        result_sha256=result.content_sha256,
        upper_state=upper_state,
        lower_state=lower_state,
        upper_sector_sha256=selected[0].hamiltonian_sha256,
        lower_sector_sha256=selected[1].hamiltonian_sha256,
        energy_difference_hartree=difference,
        frequency_cm1=difference * HARTREE_CM1,
        frequency_mhz=difference * HARTREE_JOULE / h / 1e6,
    )
