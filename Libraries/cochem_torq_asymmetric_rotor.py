"""Numerical Watson A rotor levels with explicit unassigned eigenstate identity.

This legacy kernel does not qualify a molecular Hamiltonian or generate a
spectroscopic identification catalog. Its former dipole defaults, guessed Ka/Kc
assignments and arbitrary intensity normalization have been removed. The named
rigid-rotor screening catalog is available in ``cochem_torq.spectroscopy.rotational``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np


def _finite_real(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be an explicitly supplied finite real number.")


@dataclass(frozen=True)
class RotationalConstants:
    """Declared principal-axis constants and Watson A parameters in MHz.

    All zero distortion parameters select the rigid approximation; they are not
    observations establishing that molecular distortion is physically zero.
    Dipoles in Debye are absent unless actually supplied by the caller.
    """

    A: float
    B: float
    C: float
    D_J: float = 0.0
    D_JK: float = 0.0
    D_K: float = 0.0
    d_1: float = 0.0  # Watson A delta_J, not a Watson S parameter.
    d_2: float = 0.0  # Watson A delta_K, not a Watson S parameter.
    mu_a: float | None = None
    mu_b: float | None = None
    mu_c: float | None = None

    def __post_init__(self) -> None:
        for name in ("A", "B", "C", "D_J", "D_JK", "D_K", "d_1", "d_2"):
            _finite_real(getattr(self, name), name)
        if not self.A >= self.B >= self.C > 0:
            raise ValueError("Principal-axis constants must satisfy A >= B >= C > 0.")
        for name in ("mu_a", "mu_b", "mu_c"):
            value = getattr(self, name)
            if value is not None:
                _finite_real(value, name)

    @property
    def hamiltonian_model(self) -> str:
        if any((self.D_J, self.D_JK, self.D_K, self.d_1, self.d_2)):
            return (
                "declared Watson A quartic numerical model; independently unqualified"
            )
        return "rigid-rotor approximation; centrifugal distortion omitted"


@dataclass(frozen=True)
class TransitionRecord:
    """Legacy import compatibility; no unqualified records are produced."""

    freq_mhz: float
    intensity: float
    j_upper: int
    ka_upper: int
    kc_upper: int
    j_lower: int
    ka_lower: int
    kc_lower: int
    e_lower_cm1: float
    dipole_type: str


class AsymmetricTopDiagonalizer:
    """Diagonalize declared Watson A matrices without inferred Ka/Kc labels."""

    quantum_number_convention = "J and sorted eigenstate index; Ka/Kc unassigned"
    identification_qualified = False

    def __init__(
        self,
        constants: RotationalConstants,
        reduction: Literal["A", "S"] = "A",
        j_max: int = 5,
    ) -> None:
        if not isinstance(constants, RotationalConstants):
            raise TypeError("Supply explicitly validated RotationalConstants.")
        if reduction == "S":
            raise NotImplementedError(
                "Watson S reduction is not implemented or qualified."
            )
        if reduction != "A":
            raise ValueError("The only implemented reduction is Watson A.")
        if type(j_max) is not int or not 0 <= j_max <= 30:
            raise ValueError("j_max must be an integer from 0 to 30.")
        self.constants = constants
        self.reduction = reduction
        self.j_max = j_max
        self._energy_levels: dict[tuple[int, int], float] = {}
        self._level_vectors: dict[tuple[int, int], np.ndarray] = {}

    def _build_watson_a_matrix(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Declared A-reduced quartic matrix in integer |J,K> angular-momentum basis."""
        if type(j) is not int or not 0 <= j <= 30:
            raise ValueError("J must be an integer from 0 to 30.")
        k_values = np.arange(-j, j + 1)
        matrix = np.zeros((2 * j + 1, 2 * j + 1), dtype=np.float64)
        constants = self.constants
        angular_momentum = j * (j + 1)
        for index, projection in enumerate(k_values):
            matrix[index, index] = (
                0.5 * (constants.B + constants.C) * (angular_momentum - projection**2)
                + constants.A * projection**2
                - constants.D_J * angular_momentum**2
                - constants.D_JK * angular_momentum * projection**2
                - constants.D_K * projection**4
            )
            if projection + 2 <= j:
                coefficient = (
                    0.25 * (constants.B - constants.C)
                    - constants.d_1 * angular_momentum
                    - 0.5 * constants.d_2 * ((projection + 2) ** 2 + projection**2)
                )
                ladder = np.sqrt(
                    angular_momentum - projection * (projection + 1)
                ) * np.sqrt(angular_momentum - (projection + 1) * (projection + 2))
                matrix[index, index + 2] = coefficient * ladder
                matrix[index + 2, index] = coefficient * ladder
        if not np.isfinite(matrix).all():
            raise ValueError("The declared Hamiltonian overflows its numerical range.")
        return matrix, k_values

    def solve_energy_levels(self) -> dict[tuple[int, int], float]:
        """Return (J, ascending eigenstate index) → numerical energy in MHz.

        An eigenstate index is not an assigned Ka/Kc quantum number. Quartic
        truncation accuracy and its applicable J range require separate evidence.
        """
        self._energy_levels.clear()
        self._level_vectors.clear()
        for angular_momentum in range(self.j_max + 1):
            matrix, _ = self._build_watson_a_matrix(angular_momentum)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            for index, (value, vector) in enumerate(zip(eigenvalues, eigenvectors.T)):
                identity = (angular_momentum, index)
                self._energy_levels[identity] = float(value)
                self._level_vectors[identity] = vector
        return dict(self._energy_levels)

    def compute_transitions(
        self,
        freq_min_mhz: float = 0.0,
        freq_max_mhz: float = 200000.0,
        temperature_k: float = 298.15,
    ) -> list[TransitionRecord]:
        """Reject unqualified legacy transition/intensity generation explicitly."""
        for name, value in (
            ("freq_min_mhz", freq_min_mhz),
            ("freq_max_mhz", freq_max_mhz),
            ("temperature_k", temperature_k),
        ):
            _finite_real(value, name)
        if freq_min_mhz < 0 or freq_max_mhz <= freq_min_mhz or temperature_k <= 0:
            raise ValueError("Use increasing nonnegative frequencies and positive T.")
        if any(
            value is None
            for value in (self.constants.mu_a, self.constants.mu_b, self.constants.mu_c)
        ):
            raise ValueError(
                "Catalog intensity requires all three actual dipole components."
            )
        raise NotImplementedError(
            "Legacy transition matrix elements, quantum assignments and intensity "
            "conventions are unqualified. Use the explicitly named rigid-rotor "
            "screening catalog in cochem_torq.spectroscopy.rotational when applicable."
        )

    def export_line_catalog_parquet(
        self,
        output_path: Path | str,
        freq_min_mhz: float = 0.0,
        freq_max_mhz: float = 200000.0,
        temperature_k: float = 298.15,
    ) -> Path:
        """Validate support before creating any purported scientific catalog."""
        self.compute_transitions(freq_min_mhz, freq_max_mhz, temperature_k)
        raise NotImplementedError("No qualified legacy line catalog can be exported.")
