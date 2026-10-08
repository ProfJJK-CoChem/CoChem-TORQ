"""Vibrational-only second-order perturbation and explicit resonance polyads.

This is the Cartesian rectilinear normal-coordinate vibrational Hamiltonian:
H0=sum_i omega_i (n_i+1/2), V3=phi_ijk q_i q_j q_k/3!,
V4=phi_ijkl q_i q_j q_k q_l/4!. It excludes Coriolis, rotational,
curvilinear kinetic and vibration–rotation terms. It cannot produce alpha or B0.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import factorial, isfinite, sqrt

import numpy as np

from ..units import HARTREE_CM1
from .forcefield import ForceField
from .harmonic import finite_array

State = tuple[int, ...]


@dataclass(frozen=True)
class Resonance:
    state_a: State
    state_b: State
    operator_order: int
    harmonic_detuning_cm1: float
    coupling_cm1: float
    kind: str


@dataclass(frozen=True)
class VibrationalVPT2Result:
    ground_energy_hartree: float
    fundamental_frequencies_cm1: np.ndarray
    harmonic_frequencies_cm1: np.ndarray
    state_corrections_hartree: np.ndarray
    force_field_digest: str
    variant: str = "nonresonant rectilinear vibrational Rayleigh–Schrodinger VPT2"
    rotation_vibration_available: bool = False
    independent_scientific_qualification: bool = False


@dataclass(frozen=True)
class PolyadResult:
    basis_states: tuple[State, ...]
    energies_hartree: np.ndarray
    mixing_columns: np.ndarray
    effective_hamiltonian_hartree: np.ndarray
    force_field_digest: str
    variant: str = "Hermitian second-order Van Vleck vibrational polyad"
    independent_scientific_qualification: bool = False


def _check(force_field: ForceField) -> int:
    count = len(force_field.frequencies_hartree)
    if not force_field.derivative_converged:
        raise ValueError("Two-scale anharmonic derivative convergence has not passed.")
    if count == 0 or np.any(force_field.frequencies_hartree <= 0):
        raise ValueError("All normal-mode frequencies must be positive.")
    finite_array(force_field.cubic_hartree, (count,) * 3)
    finite_array(force_field.quartic_hartree, (count,) * 4)
    for tensor in (force_field.cubic_hartree, force_field.quartic_hartree):
        for axis in range(1, tensor.ndim):
            if not np.allclose(
                tensor, tensor.swapaxes(0, axis), rtol=1e-12, atol=1e-15
            ):
                raise ValueError(
                    "Force constants must have complete permutation symmetry."
                )
    return count


def _energy(state: State, frequencies: np.ndarray) -> float:
    return float(np.dot(np.asarray(state) + 0.5, frequencies))


def _operator_action(state: State, tensor: np.ndarray) -> dict[State, float]:
    """Exact finite support of a polynomial in harmonic oscillator coordinates."""
    result: dict[State, float] = defaultdict(float)
    for indices in combinations_with_replacement(range(len(state)), tensor.ndim):
        counts = Counter(indices)
        coefficient = float(tensor[indices]) / np.prod(
            [factorial(power) for power in counts.values()]
        )
        if coefficient == 0:
            continue
        terms = {state: coefficient}
        for mode in indices:
            next_terms: dict[State, float] = defaultdict(float)
            for occupation, value in terms.items():
                if occupation[mode] > 0:
                    lower = list(occupation)
                    lower[mode] -= 1
                    next_terms[tuple(lower)] += value * sqrt(occupation[mode] / 2)
                upper = list(occupation)
                upper[mode] += 1
                next_terms[tuple(upper)] += value * sqrt((occupation[mode] + 1) / 2)
            terms = next_terms
        for occupation, value in terms.items():
            result[occupation] += value
    return dict(result)


def _targets(count: int) -> tuple[State, ...]:
    return (tuple([0] * count),) + tuple(
        tuple(int(i == j) for i in range(count)) for j in range(count)
    )


def analyze_resonances(
    force_field: ForceField,
    *,
    states: Iterable[State] | None = None,
    detuning_threshold_cm1: float = 10.0,
    coupling_to_detuning_threshold: float = 0.1,
    coupling_floor_cm1: float = 1e-6,
) -> tuple[Resonance, ...]:
    """Return near-degeneracy candidates and strong-coupling diagnostics.

    A large single-mode cubic matrix element is an applicability diagnostic,
    not evidence of a Fermi or Darling–Dennison resonance. Thresholds are
    explicit conservative protocol choices, not a calibrated accuracy bound.
    """
    count = _check(force_field)
    if not all(
        isfinite(x) and x > 0
        for x in (
            detuning_threshold_cm1,
            coupling_to_detuning_threshold,
            coupling_floor_cm1,
        )
    ):
        raise ValueError("Explicit finite positive resonance thresholds are required.")
    selected = tuple(states) if states is not None else _targets(count)
    seen = set()
    resonances = []
    for state in selected:
        if len(state) != count or any(type(n) is not int or n < 0 for n in state):
            raise ValueError(
                "Resonance states require integer occupations for every mode."
            )
        for tensor in (force_field.cubic_hartree, force_field.quartic_hartree):
            for other, value in _operator_action(state, tensor).items():
                if other == state:
                    continue
                first, second = sorted((state, other))
                key = (first, second, tensor.ndim)
                if key in seen:
                    continue
                seen.add(key)
                detuning = (
                    abs(
                        _energy(state, force_field.frequencies_hartree)
                        - _energy(other, force_field.frequencies_hartree)
                    )
                    * HARTREE_CM1
                )
                coupling = abs(value) * HARTREE_CM1
                if coupling > coupling_floor_cm1 and (
                    detuning <= detuning_threshold_cm1
                    or coupling >= coupling_to_detuning_threshold * detuning
                ):
                    changes = np.asarray(state) - np.asarray(other)
                    positive_quanta = int(np.sum(changes[changes > 0]))
                    negative_quanta = int(-np.sum(changes[changes < 0]))
                    # State totals alone do not establish a resonance class:
                    # one mode's cubic n→n+1 coupling is not a Fermi exchange.
                    kind = "vibrational"
                    if tensor.ndim == 3 and sorted(
                        (positive_quanta, negative_quanta)
                    ) == [1, 2]:
                        kind = "Fermi"
                    elif tensor.ndim == 4 and positive_quanta == negative_quanta == 2:
                        kind = "Darling-Dennison"
                    elif detuning > detuning_threshold_cm1:
                        kind = "strong_anharmonic_coupling"
                    resonances.append(
                        Resonance(first, second, tensor.ndim, detuning, coupling, kind)
                    )
    return tuple(resonances)


def vibrational_vpt2(
    force_field: ForceField,
    *,
    detuning_threshold_cm1: float = 10.0,
    coupling_to_detuning_threshold: float = 0.1,
) -> VibrationalVPT2Result:
    """Nonresonant VPT2 using finite, exact sums over V3-connected states.

    Quartic first order and cubic second order are retained; quartic second order
    and cubic higher orders would exceed this perturbative Hamiltonian order.
    Near-degeneracy candidates stop this routine and may be handled by an
    explicitly supplied polyad. Strong nondegenerate couplings separately stop
    it at the configured perturbative applicability gate; a perturbative polyad
    alone is not proof that such a model is valid. Returned frequencies remain
    exploratory until independent molecular reference qualification has passed.
    """
    count = _check(force_field)
    states = _targets(count)
    resonances = analyze_resonances(
        force_field,
        states=states,
        detuning_threshold_cm1=detuning_threshold_cm1,
        coupling_to_detuning_threshold=coupling_to_detuning_threshold,
    )
    if resonances:
        strong = sum(item.kind == "strong_anharmonic_coupling" for item in resonances)
        if strong:
            raise ValueError(
                f"Detected {strong} strong anharmonic coupling diagnostic(s) "
                "that exceed the configured perturbative applicability gate; "
                "independent nonperturbative validation or an explicitly "
                "validated treatment is required. "
                f"Additional resonance candidate(s): {len(resonances) - strong}."
            )
        raise ValueError(
            f"Detected {len(resonances)} resonance(s); "
            "explicit polyad treatment required."
        )
    corrections = []
    energies = []
    for state in states:
        e0 = _energy(state, force_field.frequencies_hartree)
        correction = _operator_action(state, force_field.quartic_hartree).get(
            state, 0.0
        )
        for other, coupling in _operator_action(
            state, force_field.cubic_hartree
        ).items():
            if other == state or coupling == 0:
                continue
            denominator = e0 - _energy(other, force_field.frequencies_hartree)
            if abs(denominator) <= np.finfo(float).eps * max(abs(e0), 1):
                raise ValueError(
                    "Nonzero cubic coupling has a singular perturbative denominator."
                )
            correction += coupling**2 / denominator
        corrections.append(correction)
        energies.append(e0 + correction)
    fundamentals = (np.asarray(energies[1:]) - energies[0]) * HARTREE_CM1
    if np.any(fundamentals <= 0):
        raise ValueError(
            "Nonpositive corrected fundamental: inapplicable perturbation theory."
        )
    return VibrationalVPT2Result(
        energies[0],
        finite_array(fundamentals),
        finite_array(force_field.frequencies_hartree * HARTREE_CM1),
        finite_array(corrections),
        force_field.source_digest,
    )


def solve_vibrational_polyad(
    force_field: ForceField,
    states: Iterable[State],
    *,
    external_denominator_threshold_cm1: float = 10.0,
) -> PolyadResult:
    """Diagonalize a supplied polyad including cubic/quartic matrix elements.

    External second-order cubic coupling is symmetrized with the Van Vleck
    denominator. A resonant omitted external state must be added to the polyad;
    it is never discarded or repaired by an arbitrary denominator shift.
    """
    count = _check(force_field)
    basis = tuple(states)
    if (
        not basis
        or len(set(basis)) != len(basis)
        or any(
            len(state) != count or any(type(n) is not int or n < 0 for n in state)
            for state in basis
        )
    ):
        raise ValueError("Polyad requires unique complete integer oscillator states.")
    if (
        not isfinite(external_denominator_threshold_cm1)
        or external_denominator_threshold_cm1 <= 0
    ):
        raise ValueError("External denominator threshold must be positive and finite.")
    harmonic_energies = np.asarray(
        [_energy(state, force_field.frequencies_hartree) for state in basis]
    )
    cubic = [_operator_action(state, force_field.cubic_hartree) for state in basis]
    quartic = [_operator_action(state, force_field.quartic_hartree) for state in basis]
    matrix = np.diag(harmonic_energies)
    for i, state_i in enumerate(basis):
        for j, state_j in enumerate(basis):
            matrix[i, j] += cubic[j].get(state_i, 0.0) + quartic[j].get(state_i, 0.0)
            for external in cubic[i].keys() & cubic[j].keys() - set(basis):
                coupling_product = cubic[i][external] * cubic[j][external]
                if coupling_product == 0:
                    continue
                ek = _energy(external, force_field.frequencies_hartree)
                di, dj = harmonic_energies[i] - ek, harmonic_energies[j] - ek
                if (
                    min(abs(di), abs(dj)) * HARTREE_CM1
                    <= external_denominator_threshold_cm1
                ):
                    raise ValueError(
                        f"Include resonant external state {external} in the polyad."
                    )
                matrix[i, j] += 0.5 * coupling_product * (1 / di + 1 / dj)
    if not np.allclose(matrix, matrix.T, rtol=1e-11, atol=1e-14):
        raise ValueError("Numerical effective Hamiltonian is not Hermitian.")
    energies, mixing = np.linalg.eigh((matrix + matrix.T) / 2)
    return PolyadResult(
        basis,
        finite_array(energies),
        finite_array(mixing),
        finite_array(matrix),
        force_field.source_digest,
    )
