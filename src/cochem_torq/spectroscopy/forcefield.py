"""Actual energy-displacement cubic/quartic derivatives in dimensionless modes."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations, product

import numpy as np

from .harmonic import (
    ATOMIC_MASS_ELECTRON,
    HARTREE_CM1,
    HarmonicResult,
    artifact_digest,
    finite_array,
)


@dataclass(frozen=True)
class EnergyEvaluation:
    """Optional raw-result manifest digest from a real displaced calculation."""

    energy_hartree: float
    source_artifact_sha256: str


@dataclass(frozen=True)
class DisplacementEnergy:
    q_dimensionless: tuple[float, ...]
    energy_hartree: float
    geometry_sha256: str
    source_artifact_sha256: str | None


@dataclass(frozen=True)
class ForceField:
    frequencies_hartree: np.ndarray
    cubic_hartree: np.ndarray
    quartic_hartree: np.ndarray
    cubic_coarse_hartree: np.ndarray
    quartic_coarse_hartree: np.ndarray
    steps_dimensionless: tuple[float, float]
    reference_energy_hartree: float
    derivative_converged: bool
    absolute_tolerance_hartree: float
    relative_tolerance: float
    evaluation_count: int
    evaluator_identity: str
    harmonic_source_digest: str
    displacement_manifest_digest: str
    source_digest: str
    displacement_records: tuple[DisplacementEnergy, ...]
    coordinate_convention: str = (
        "q=sqrt(omega)*Q; dimensionless; Vn=sum(phi_i... q_i...)/n!"
    )
    quartic_scope: str = "full"


_STENCILS = {
    1: ((-1, 1), (-0.5, 0.5)),
    2: ((-1, 0, 1), (1, -2, 1)),
    3: ((-2, -1, 1, 2), (-0.5, 1, -1, 0.5)),
    4: ((-2, -1, 0, 1, 2), (1, -4, 6, -4, 1)),
}


def build_force_field(
    harmonic: HarmonicResult,
    energy_evaluator: Callable[[np.ndarray], float | EnergyEvaluation],
    *,
    evaluator_identity: str,
    steps: tuple[float, float] = (0.08, 0.04),
    absolute_tolerance_hartree: float = 1e-6,
    relative_tolerance: float = 0.02,
    max_evaluations: int = 20000,
) -> ForceField:
    """Differentiate real evaluator energies, caching exactly matching displacements.

    The evaluator must independently converge the same electronic recipe/state at
    every displaced geometry and raise on failure. This routine never reuses the
    reference geometry's energy for a displacement. Two central-difference scales
    produce a recorded convergence comparison; failed convergence is retained and
    blocks VPT2. No numerical derivative is described as analytic.
    """
    if not isinstance(harmonic, HarmonicResult):
        raise ValueError("An actual harmonic result with its mode basis is required.")
    if harmonic.stationary_character != "positive_definite_vibrational_hessian":
        raise ValueError("A complete positive vibrational Hessian is required.")
    if not evaluator_identity.strip():
        raise ValueError("An explicit evaluator recipe/build identity is required.")
    if (
        len(steps) != 2
        or not all(math.isfinite(x) and x > 0 for x in steps)
        or steps[0] <= steps[1]
    ):
        raise ValueError(
            "Provide two positive decreasing dimensionless displacement scales."
        )
    if (
        not (
            math.isfinite(absolute_tolerance_hartree)
            and absolute_tolerance_hartree >= 0
            and math.isfinite(relative_tolerance)
            and relative_tolerance >= 0
        )
        or type(max_evaluations) is not int
        or max_evaluations < 1
    ):
        raise ValueError(
            "Nonnegative convergence tolerances and an evaluation budget are required."
        )
    modes = len(harmonic.frequencies_cm1)
    if not modes:
        raise ValueError("An atom has no anharmonic vibrational force field.")
    geometry = finite_array(harmonic.coordinates_bohr)
    atoms = len(geometry)
    if geometry.shape != (atoms, 3) or modes != 3 * atoms - harmonic.external_rank:
        raise ValueError("Harmonic geometry and complete mode count must agree.")
    masses = finite_array(harmonic.isotope_masses_u, (atoms,))
    frequencies = finite_array(harmonic.frequencies_cm1, (modes,))
    angular = finite_array(harmonic.angular_frequencies_au, (modes,))
    eigenvalues = finite_array(harmonic.eigenvalues_au, (modes,))
    weighted_modes = finite_array(harmonic.mass_weighted_modes, (3 * atoms, modes))
    cartesian_modes = finite_array(harmonic.cartesian_modes, (3 * atoms, modes))
    transform = finite_array(harmonic.dimensionless_to_cartesian, (3 * atoms, modes))
    if (
        np.any(masses <= 0)
        or np.any(frequencies <= 0)
        or np.any(angular <= 0)
        or not np.allclose(frequencies, angular * HARTREE_CM1, rtol=1e-12, atol=1e-10)
        or not np.allclose(eigenvalues, angular**2, rtol=1e-12, atol=1e-15)
        or not np.allclose(
            weighted_modes.T @ weighted_modes, np.eye(modes), rtol=0, atol=1e-10
        )
        or not np.allclose(
            cartesian_modes,
            weighted_modes
            / np.sqrt(np.repeat(masses * ATOMIC_MASS_ELECTRON, 3))[:, None],
            rtol=1e-12,
            atol=1e-14,
        )
        or not np.allclose(
            transform,
            cartesian_modes / np.sqrt(angular)[None, :],
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "Harmonic frequencies, masses and coordinate normalization disagree."
        )
    cache: dict[tuple[float, ...], float] = {}
    records: dict[tuple[float, ...], DisplacementEnergy] = {}

    def evaluate(q: tuple[float, ...]) -> float:
        if q not in cache:
            if len(cache) >= max_evaluations:
                raise RuntimeError(
                    "Displacement budget exhausted; no partial force field is returned."
                )
            coordinates = harmonic.coordinates_bohr + (
                transform @ np.asarray(q)
            ).reshape(-1, 3)
            returned = energy_evaluator(coordinates.copy())
            source_sha256 = None
            if isinstance(returned, EnergyEvaluation):
                import re

                source_sha256 = returned.source_artifact_sha256
                if re.fullmatch(r"[a-f0-9]{64}", source_sha256) is None:
                    raise ValueError(
                        "Raw calculation manifest must have a SHA-256 digest."
                    )
                energy = returned.energy_hartree
            else:
                energy = returned
            if (
                isinstance(energy, (bool, np.bool_))
                or np.iscomplexobj(energy)
                or not np.isscalar(energy)
                or not np.isfinite(energy)
            ):
                raise ValueError(
                    "Displaced evaluator must return finite converged hartree energy."
                )
            cache[q] = float(energy)
            records[q] = DisplacementEnergy(
                q, float(energy), artifact_digest(coordinates), source_sha256
            )
        return cache[q]

    reference_energy = evaluate(tuple([0.0] * modes))

    def derivative(indices: tuple[int, ...], step: float) -> float:
        multiplicities = sorted(Counter(indices).items())
        terms = []
        for choice in product(
            *[range(len(_STENCILS[order][0])) for _, order in multiplicities]
        ):
            q = np.zeros(modes)
            coefficient = 1.0
            for stencil_index, (mode, order) in zip(choice, multiplicities):
                nodes, weights = _STENCILS[order]
                q[mode] = step * nodes[stencil_index]
                coefficient *= weights[stencil_index]
            # Reference subtraction reduces cancellation from a large absolute
            # electronic energy without changing a derivative of order >= 1.
            terms.append(coefficient * (evaluate(tuple(q)) - reference_energy))
        return math.fsum(terms) / step ** len(indices)

    def tensor(order: int, step: float) -> np.ndarray:
        result = np.empty((modes,) * order)
        for indices in combinations_with_replacement(range(modes), order):
            value = derivative(indices, step)
            for permutation in set(permutations(indices)):
                result[permutation] = value
        return result

    cubic_coarse = tensor(3, steps[0])
    quartic_coarse = tensor(4, steps[0])
    cubic = tensor(3, steps[1])
    quartic = tensor(4, steps[1])
    converged = all(
        np.all(
            np.abs(fine - coarse)
            <= absolute_tolerance_hartree + relative_tolerance * np.abs(fine)
        )
        for fine, coarse in ((cubic, cubic_coarse), (quartic, quartic_coarse))
    )
    manifest = [record.__dict__ for _, record in sorted(records.items())]
    manifest_digest = artifact_digest(
        {
            "recipe": evaluator_identity,
            "harmonic": harmonic.source_digest,
            "displacements": manifest,
        }
    )
    digest = artifact_digest(
        {
            "harmonic": harmonic.source_digest,
            "cubic": cubic,
            "quartic": quartic,
            "steps": steps,
            "manifest": manifest_digest,
        }
    )
    return ForceField(
        finite_array(harmonic.angular_frequencies_au),
        finite_array(cubic),
        finite_array(quartic),
        finite_array(cubic_coarse),
        finite_array(quartic_coarse),
        tuple(steps),
        reference_energy,
        bool(converged),
        absolute_tolerance_hartree,
        relative_tolerance,
        len(cache),
        evaluator_identity,
        harmonic.source_digest,
        manifest_digest,
        digest,
        tuple(record for _, record in sorted(records.items())),
    )
