"""Exact finite-J rigid-rotor Hamiltonian and explicitly labeled screening lines.

This model excludes centrifugal distortion, hyperfine structure and internal
rotation. It is never identified as SPCAT or an identification-ready catalog.
"""
# Angular-momentum symbols and Einstein A retain the published notation.
# ruff: noqa: N803, N806, N815

from __future__ import annotations

import re
from dataclasses import dataclass
from math import exp, factorial, isfinite, sqrt
from typing import Any

import numpy as np

from ..units import DEBYE_COULOMB_METRE, c, epsilon_0, h, k, pi
from .harmonic import EquilibriumRotor, finite_array


@dataclass(frozen=True)
class GroundStateRotor:
    constants_mhz: tuple[float | None, float | None, float | None]
    equilibrium_geometry_digest: str
    correction_artifact_sha256: str
    correction_method: str
    independent_validation_reference: str
    observable: str = "B0"


def apply_rotation_vibration_correction(
    rotor: EquilibriumRotor,
    alpha_mhz_by_axis: dict[str, Any],
    *,
    correction_method: str,
    correction_artifact_sha256: str,
    independent_validation_reference: str,
    equilibrium_geometry_digest: str,
    isotope_masses_u: Any,
    degeneracies: Any | None = None,
) -> GroundStateRotor:
    """B0(axis)=Be(axis)-sum_i degeneracy_i*alpha_i(axis)/2.

    Inputs must already be independently qualified rotation–vibration interactions
    in this rotor's axis frame and isotopologue. The correction record is externally
    supplied; a harmonic Hessian or vibrational-only VPT2 result cannot fill it.
    Grouped degenerate modes require explicit degeneracies; ungrouped mode
    components use one. No fitted empirical scaling is added.
    """
    if (
        not correction_method.strip()
        or not independent_validation_reference.strip()
        or re.fullmatch(r"[a-f0-9]{64}", correction_artifact_sha256) is None
    ):
        raise ValueError(
            "Correction method, artifact digest and independent validation "
            "reference are required."
        )
    correction_masses = finite_array(isotope_masses_u, rotor.isotope_masses_u.shape)
    if equilibrium_geometry_digest != rotor.geometry_digest or not np.array_equal(
        correction_masses, rotor.isotope_masses_u
    ):
        raise ValueError(
            "Correction must identify this exact geometry, axis convention "
            "and isotopologue."
        )
    axes = tuple(
        axis
        for axis, constant in zip(("A", "B", "C"), rotor.constants_mhz)
        if constant is not None
    )
    if set(alpha_mhz_by_axis) != set(axes):
        raise ValueError(
            "Supply alpha for the defined principal axes; undefined linear A is absent."
        )
    arrays = {axis: finite_array(alpha_mhz_by_axis[axis]) for axis in axes}
    lengths = {len(array) for array in arrays.values() if array.ndim == 1}
    if (
        len(lengths) != 1
        or any(array.ndim != 1 for array in arrays.values())
        or next(iter(lengths), 0) == 0
    ):
        raise ValueError(
            "Alpha requires a same-length nonempty mode list for every defined axis."
        )
    size = next(iter(lengths))
    weight = (
        np.ones(size) if degeneracies is None else finite_array(degeneracies, (size,))
    )
    if np.any(weight < 1) or np.any(weight != np.floor(weight)):
        raise ValueError("Mode degeneracies must be positive integers.")
    values = tuple(
        None if constant is None else float(constant - np.dot(weight, arrays[axis]) / 2)
        for axis, constant in zip(("A", "B", "C"), rotor.constants_mhz)
    )
    constants = (values[0], values[1], values[2])
    if any(value is not None and value <= 0 for value in constants):
        raise ValueError("Vibration-corrected constants must be positive.")
    return GroundStateRotor(
        constants,
        rotor.geometry_digest,
        correction_artifact_sha256,
        correction_method,
        independent_validation_reference,
    )


def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Integer angular-momentum Racah formula, sufficient for rigid rotors.

    No half-integer/hyperfine quantum numbers are implied. The catalog bounds J to
    30 to keep direct factorial evaluation well conditioned.
    """
    values = (j1, j2, j3, m1, m2, m3)
    if any(type(value) is not int for value in values) or min(j1, j2, j3) < 0:
        raise ValueError("Integer nonnegative angular momenta are required.")
    if max(j1, j2, j3) > 30:
        raise ValueError("Direct Racah evaluation is bounded to angular momentum 30.")
    if (
        m1 + m2 + m3
        or any(abs(m) > j for j, m in ((j1, m1), (j2, m2), (j3, m3)))
        or j3 < abs(j1 - j2)
        or j3 > j1 + j2
    ):
        return 0.0
    triangle = (
        factorial(j1 + j2 - j3)
        * factorial(j1 - j2 + j3)
        * factorial(-j1 + j2 + j3)
        / factorial(j1 + j2 + j3 + 1)
    )
    norm = np.prod(
        [
            factorial(j + m) * factorial(j - m)
            for j, m in ((j1, m1), (j2, m2), (j3, m3))
        ],
        dtype=object,
    )
    start = max(0, j2 - j3 - m1, j1 - j3 + m2)
    stop = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    total = 0.0
    for z in range(start, stop + 1):
        denominator = (
            factorial(z)
            * factorial(j1 + j2 - j3 - z)
            * factorial(j1 - m1 - z)
            * factorial(j2 + m2 - z)
            * factorial(j3 - j2 + m1 + z)
            * factorial(j3 - j1 - m2 + z)
        )
        total += (-1) ** z / denominator
    return float((-1) ** (j1 - j2 - m3) * sqrt(float(triangle * norm)) * total)


@dataclass(frozen=True)
class RotationalLevel:
    J: int
    eigenstate_index: int
    energy_mhz: float
    K_basis: tuple[int, ...]
    coefficients: np.ndarray


@dataclass(frozen=True)
class RigidRotorLine:
    frequency_mhz: float
    upper_J: int
    upper_eigenstate_index: int
    lower_J: int
    lower_eigenstate_index: int
    lower_energy_mhz: float
    summed_dipole_strength_debye2: float
    relative_absorption_weight_debye2: float
    einstein_A_s1: float


@dataclass(frozen=True)
class RigidRotorCatalog:
    lines: tuple[RigidRotorLine, ...]
    temperature_kelvin: float
    partition_function: float
    J_max: int
    partition_relative_tail_indicator: float
    partition_converged_at_requested_tolerance: bool
    constant_observable: str
    model_identity: str = (
        "TORQ exact finite-J electric-dipole rigid-rotor screening model v1"
    )
    quantum_number_convention: str = (
        "J and sorted Hamiltonian eigenstate index; no inferred Ka/Kc assignment"
    )
    intensity_convention: str = (
        "sum over M and three lab polarizations; Boltzmann population times "
        "stimulated-emission correction; no instrument model"
    )
    nuclear_spin_convention: str = (
        "nuclear-spin weights excluded; every rotational eigenstate weight one; "
        "no permutation-symmetry restrictions applied"
    )
    identification_qualified: bool = False


def rigid_rotor_levels(
    constants_mhz: tuple[float | None, float | None, float | None], J_max: int
) -> tuple[RotationalLevel, ...]:
    if type(J_max) is not int or not 1 <= J_max <= 30:
        raise ValueError("J_max must be an integer from 1 to 30.")
    if len(constants_mhz) != 3 or any(
        value is not None and (not isfinite(value) or value <= 0)
        for value in constants_mhz
    ):
        raise ValueError(
            "Three defined/absent positive rotational constants are required."
        )
    a, b, cc = constants_mhz
    if b is None or cc is None:
        raise ValueError("An atom has no rotational spectrum.")
    linear = a is None
    if linear and not np.isclose(b, cc, rtol=1e-10, atol=0):
        raise ValueError("A linear rotor requires equal B and C.")
    result = []
    for J in range(J_max + 1):
        if a is None:
            result.append(
                RotationalLevel(J, 0, float(b * J * (J + 1)), (0,), finite_array([1.0]))
            )
            continue
        ks = np.arange(-J, J + 1)
        diagonal = a * ks**2 + 0.5 * (b + cc) * (J * (J + 1) - ks**2)
        matrix = np.diag(diagonal.astype(float))
        for index, K in enumerate(ks[:-2]):
            value = (
                (b - cc)
                / 4
                * sqrt((J * (J + 1) - K * (K + 1)) * (J * (J + 1) - (K + 1) * (K + 2)))
            )
            matrix[index, index + 2] = matrix[index + 2, index] = value
        energies, vectors = np.linalg.eigh(matrix)
        result.extend(
            RotationalLevel(
                J,
                index,
                float(energy),
                tuple(int(x) for x in ks),
                finite_array(vectors[:, index]),
            )
            for index, energy in enumerate(energies)
        )
    return tuple(result)


def _line_strength(
    upper: RotationalLevel, lower: RotationalLevel, dipole: np.ndarray
) -> float:
    spherical = {
        0: complex(dipole[0]),
        1: -(dipole[1] + 1j * dipole[2]) / sqrt(2),
        -1: (dipole[1] - 1j * dipole[2]) / sqrt(2),
    }
    amplitude = 0j
    for i, ku in enumerate(upper.K_basis):
        for j, kl in enumerate(lower.K_basis):
            q = ku - kl
            if abs(q) > 1:
                continue
            amplitude += (
                upper.coefficients[i].conjugate()
                * lower.coefficients[j]
                * (-1) ** ku
                * wigner_3j(upper.J, 1, lower.J, -ku, q, kl)
                * spherical[q]
            )
    return float((2 * upper.J + 1) * (2 * lower.J + 1) * abs(amplitude) ** 2)


def rigid_rotor_catalog(
    constants_mhz: tuple[float | None, float | None, float | None],
    dipole_principal_axes_debye: Any,
    *,
    temperature_kelvin: float,
    J_max: int,
    constant_observable: str,
    partition_tail_tolerance: float = 1e-5,
    line_strength_tolerance_debye2: float = 1e-14,
) -> RigidRotorCatalog:
    """Compute actual rotor eigenvalues/matrix elements and thermal line weights.

    The partition-tail indicator is the fraction carried by the last two J shells;
    it is a convergence indicator rather than a rigorous infinite-sum error bound.
    Failure is retained in the catalog and prevents a converged-spectrum claim.
    """
    if (
        not isfinite(temperature_kelvin)
        or temperature_kelvin <= 0
        or constant_observable not in ("Be", "B0", "experiment")
    ):
        raise ValueError(
            "Positive temperature and explicit constant identity are required."
        )
    if (
        not isfinite(partition_tail_tolerance)
        or partition_tail_tolerance <= 0
        or not isfinite(line_strength_tolerance_debye2)
        or line_strength_tolerance_debye2 < 0
    ):
        raise ValueError(
            "Positive partition tolerance and nonnegative line-strength "
            "tolerance are required."
        )
    dipole = finite_array(dipole_principal_axes_debye, (3,))
    if constants_mhz[0] is None and np.any(np.abs(dipole[1:]) > 1e-12):
        raise ValueError(
            "Linear rigid-rotor model supports only the molecular-axis dipole."
        )
    levels = rigid_rotor_levels(constants_mhz, J_max)
    beta = h * 1e6 / (k * temperature_kelvin)
    populations = [
        (2 * level.J + 1) * exp(-beta * level.energy_mhz) for level in levels
    ]
    partition = float(sum(populations))
    tail = float(
        sum(
            weight for level, weight in zip(levels, populations) if level.J >= J_max - 1
        )
        / partition
    )
    by_j = {
        J: tuple(level for level in levels if level.J == J) for J in range(J_max + 1)
    }
    lines = []
    for lower_J in range(J_max + 1):
        for upper_J in (lower_J, lower_J + 1):
            if upper_J > J_max or upper_J + lower_J == 0:
                continue
            for first in by_j[lower_J]:
                for second in by_j[upper_J]:
                    upper, lower = (
                        (second, first)
                        if second.energy_mhz > first.energy_mhz
                        else (first, second)
                    )
                    if upper.energy_mhz <= lower.energy_mhz or (
                        upper_J == lower_J
                        and first.eigenstate_index >= second.eigenstate_index
                    ):
                        continue
                    strength = _line_strength(upper, lower, dipole)
                    if strength <= line_strength_tolerance_debye2:
                        continue
                    frequency = upper.energy_mhz - lower.energy_mhz
                    absorption = (
                        strength
                        * exp(-beta * lower.energy_mhz)
                        * (-np.expm1(-beta * frequency))
                        / partition
                    )
                    # Sum over all upper/lower M and lab polarizations divided by
                    # the upper magnetic degeneracy gives the spontaneous rate.
                    rate = (
                        16
                        * pi**3
                        * (frequency * 1e6) ** 3
                        / (3 * epsilon_0 * h * c**3)
                        * strength
                        * DEBYE_COULOMB_METRE**2
                        / (2 * upper.J + 1)
                    )
                    lines.append(
                        RigidRotorLine(
                            frequency,
                            upper.J,
                            upper.eigenstate_index,
                            lower.J,
                            lower.eigenstate_index,
                            lower.energy_mhz,
                            strength,
                            float(absorption),
                            float(rate),
                        )
                    )
    return RigidRotorCatalog(
        tuple(sorted(lines, key=lambda line: line.frequency_mhz)),
        temperature_kelvin,
        partition,
        J_max,
        tail,
        tail <= partition_tail_tolerance,
        constant_observable,
    )
