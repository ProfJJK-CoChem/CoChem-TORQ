"""Spectroscopy integrity checks and explicit statistical-mechanics helpers.

Rigid-rotor partitions use the classical high-temperature approximation;
harmonic partitions require identified positive vibrational modes. These are
model calculations, not validated VPT2 or low-temperature line-list spectra.
The historical native SPCAT writers and mixed-engine parser are blocked until
their conventions have independent reference validation. Verified input decks
can be executed with ``cochem_torq_spcat.PickettSPCATRunner``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import physical_constants

try:
    import molsym  # type: ignore[import-untyped]

    _MOLSYM_AVAILABLE = True
except ImportError:
    _MOLSYM_AVAILABLE = False

try:
    from cochem_base.config_loader import (
        get_base_root,
        get_repo_root,
    )
    from cochem_base.exceptions import (
        AirGapViolationError,
        FortranOverflowError,
        LAMTriggerError,
        ProvenanceErrorCode,
        SPCATBridgeError,
    )
except ImportError:

    class ProvenanceErrorCode:  # type: ignore[no-redef]
        AIRGAP_VIOLATION = "AIRGAP_VIOLATION"
        FORTRAN_OVERFLOW = "FORTRAN_OVERFLOW"
        LAM_TRIGGER = "LAM_TRIGGER"
        SPCAT_BRIDGE_ERROR = "SPCAT_BRIDGE_ERROR"

    class SPCATBridgeError(Exception):  # type: ignore[no-redef]
        def __init__(
            self,
            message: str,
            error_code: Any = ProvenanceErrorCode.SPCAT_BRIDGE_ERROR,
            details: dict[str, Any] | None = None,
        ) -> None:
            super().__init__(message)
            self.message = message
            self.error_code = error_code
            self.details = details or {}

    class AirGapViolationError(SPCATBridgeError):  # type: ignore[no-redef]
        pass

    class FortranOverflowError(SPCATBridgeError):  # type: ignore[no-redef]
        pass

    class LAMTriggerError(SPCATBridgeError):  # type: ignore[no-redef]
        pass

    def get_repo_root() -> Path:  # type: ignore[no-redef]
        return Path(__file__).resolve().parent.parent

    def get_base_root() -> Path:  # type: ignore[no-redef]
        cand = Path(__file__).resolve().parent.parent.parent / "CoChem-BASE"
        return cand if cand.exists() else get_repo_root()


logger = logging.getLogger(__name__)


# =============================================================================
# 1. Fundamental Physical Constants and Measured Conversion Factors
# =============================================================================


@dataclass(frozen=True)
class CODATA2022:
    """SI defining constants and measured CODATA conversion factors.

    h, k_B, c and N_A are exact SI definitions; the atomic mass constant and
    resulting rotational conversion factor have measurement uncertainty.
    """

    # Planck constant (exact, SI definition 2019) [J * s]
    H: float = 6.62607015e-34
    # Boltzmann constant (exact, SI definition 2019) [J * K^-1]
    K_B: float = 1.380649e-23
    # Speed of light in vacuum (exact) [m * s^-1]
    C_M_S: float = 299792458.0
    # Speed of light in vacuum (exact) [cm * s^-1]
    C_CM_S: float = 29979245800.0
    # Rotational constant factor C_rot = h / (8 * pi^2) in [MHz * u * Angstrom^2]
    # h / (8 * pi^2 * u * 1e-20) * 1e-6 MHz = 505379.008435
    C_ROT: float = H / (8 * math.pi**2 * physical_constants["atomic mass constant"][0] * 1e-20) * 1e-6
    # Avogadro constant (exact) [mol^-1]
    N_A: float = 6.02214076e23
    # Atomic mass constant [kg] (CODATA 2022 recommended value)
    AMU_KG: float = 1.66053906892e-27
    # h * c / k_B conversion factor [K * cm]
    # (6.62607015e-34 * 29979245800.0) / 1.380649e-23 = 1.4387768775039336
    HC_OVER_KB: float = 1.4387768775039336
    # k_B / h factor for rotational partition function [Hz / K] = [s^-1 * K^-1]
    # 1.380649e-23 / 6.62607015e-34 = 20836619123.33 Hz/K
    KB_OVER_H: float = 1.380649e-23 / 6.62607015e-34


CONSTANTS = CODATA2022()

# Module-level aliases for immutable CODATA 2022 constants
CODATA_YEAR: int = 2022
PLANCK_CONSTANT_JS: float = CONSTANTS.H
BOLTZMANN_CONSTANT_JK: float = CONSTANTS.K_B
SPEED_OF_LIGHT_CMS: float = CONSTANTS.C_CM_S
SPEED_OF_LIGHT_MS: float = CONSTANTS.C_M_S
ROTATIONAL_FACTOR_C_ROT: float = CONSTANTS.C_ROT
C_ROT: float = CONSTANTS.C_ROT
HC_OVER_KB: float = CONSTANTS.HC_OVER_KB
KB_OVER_H: float = CONSTANTS.KB_OVER_H


# =============================================================================
# 2. Data Structures and Transfer Objects
# =============================================================================


@dataclass
class SymmetryDivisorResult:
    """Structured result of point-group symmetry resolution and spin weight assignment."""

    point_group: str
    sigma: int
    spin_statistical_weights: list[int] | None
    spin_weight_ratio_str: str | None
    effective_divisor: float
    guardrail_status: str
    equivalent_atom_groups: dict[str, list[int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to serializable dictionary."""
        return asdict(self)


@dataclass
class PartitionFunctionResult:
    """Structured internal partition function evaluation across a temperature grid."""

    temperatures: list[float]
    q_rot: dict[float, float]
    q_vib: dict[float, float]
    q_total: dict[float, float]
    dropped_lam_frequencies: list[float] = field(default_factory=list)
    stiff_frequencies: list[float] = field(default_factory=list)
    is_dvr_coupled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to serializable dictionary."""
        return asdict(self)


@dataclass
class SPCATParameter:
    """Rigidly formatted parameter record for Pickett's SPCAT .var/.par file."""

    param_id: int
    value: float
    uncertainty: float
    label: str
    formatted_line: str


@dataclass
class SPCATPayload:
    """Complete package of SPCAT input files, cryptographic hashes, and provenance manifest."""

    molecule_name: str
    var_content: str
    int_contents: dict[float, str]
    provenance_manifest: dict[str, Any]
    sha256_var: str
    sha256_int: dict[float, str]
    var_filepath: str | None = None
    int_filepaths: dict[float, str] = field(default_factory=dict)
    provenance_filepath: str | None = None


# Point group to rotational symmetry number sigma mapping
_POINT_GROUP_SIGMAS: dict[str, int] = {
    "C1": 1,
    "Cs": 1,
    "Ci": 1,
    "C2": 2,
    "C2v": 2,
    "C2h": 2,
    "C3": 3,
    "C3v": 3,
    "C3h": 3,
    "C4": 4,
    "C4v": 4,
    "C4h": 4,
    "C5": 5,
    "C5v": 5,
    "C5h": 5,
    "C6": 6,
    "C6v": 6,
    "C6h": 6,
    "D2": 4,
    "D2h": 4,
    "D2d": 4,
    "D3": 6,
    "D3h": 6,
    "D3d": 6,
    "D4": 8,
    "D4h": 8,
    "D4d": 8,
    "D5": 10,
    "D5h": 10,
    "D5d": 10,
    "D6": 12,
    "D6h": 12,
    "D6d": 12,
    "Td": 12,
    "Th": 12,
    "Oh": 24,
    "O": 24,
    "Ih": 60,
    "I": 60,
    "Cinfv": 1,
    "Dinfh": 2,
    "Kh": 1,
}


def _pg_to_sigma(pg: str) -> int:
    """Resolve rotational symmetry number sigma from Schoenflies point group string."""
    clean = pg.strip()
    if clean not in _POINT_GROUP_SIGMAS:
        raise SPCATBridgeError(f"Unsupported point group {pg!r}; symmetry number unavailable.")
    return _POINT_GROUP_SIGMAS[clean]


# =============================================================================
# 3. Low-Frequency LAM Trap (Physical Guardrail against RRHO Failure)
# =============================================================================


def low_frequency_lam_trap(
    harmonic_frequencies: Sequence[float],
    threshold_cm1: float = 50.0,
    zero_mode_cutoff: float = 1e-4,
) -> list[float]:
    """Trap vibrational normal mode frequencies below threshold (< 50 cm^-1).

    Under the Rigid-Rotor Harmonic-Oscillator (RRHO) approximation, low-frequency
    vibrational modes (< 50 cm^-1) correspond to Large Amplitude Motions (LAM)
    such as methyl internal rotation, ring puckering, or low-barrier torsion.
    Simple harmonic partition functions diverge and fail catastrophically for LAM.
    This guardrail intercepts these modes, raises a LAMTriggerError with
    LAM_TRIGGER error code, and demands execution of Phase 7 DVR solvers.

    Args:
        harmonic_frequencies: Sequence of vibrational normal mode frequencies (cm^-1).
        threshold_cm1: Critical LAM frequency threshold in cm^-1 (default: 50.0).
        zero_mode_cutoff: Tolerance below which modes are treated as zero/translational (default: 1e-4).

    Returns:
        Validated list of stiff vibrational frequencies (all >= threshold_cm1).

    Raises:
        LAMTriggerError: If any genuine vibrational mode is below threshold_cm1.
    """
    flagged_lam_modes: list[float] = []
    stiff_modes: list[float] = []

    for raw_freq in harmonic_frequencies:
        freq = float(raw_freq)
        if not math.isfinite(freq):
            raise ValueError("Vibrational frequencies must be finite.")
        if freq <= zero_mode_cutoff:
            raise ValueError("Vibrational modes must be positive and identified independently of external zero modes.")
        if freq < threshold_cm1:
            flagged_lam_modes.append(freq)
        else:
            stiff_modes.append(freq)

    if flagged_lam_modes:
        min_lam = min(flagged_lam_modes)
        error_msg = (
            f"LAM detected: vibrational frequency {min_lam:.2f} cm^-1 is below "
            f"threshold {threshold_cm1:.1f} cm^-1. Rigid-Rotor Harmonic-Oscillator (RRHO) "
            f"approximation is invalid. Phase 7 DVR solvers are physically required."
        )
        logger.warning(
            "[LAM_TRIGGER] %s (Flagged modes: %s)", error_msg, flagged_lam_modes
        )
        raise LAMTriggerError(
            message=error_msg,
            error_code=ProvenanceErrorCode.LAM_TRIGGER,
            details={
                "flagged_frequencies": [float(f) for f in flagged_lam_modes],
                "threshold_cm1": float(threshold_cm1),
                "stiff_frequencies_count": len(stiff_modes),
                "total_frequencies_evaluated": len(harmonic_frequencies),
                "min_lam_frequency": float(min_lam),
            },
        )

    return stiff_modes


# Backward-compatible alias
low_frequency_trap = low_frequency_lam_trap


# =============================================================================
# 4. MolSym Symmetry Solver & Nuclear Spin Statistical Weights
# =============================================================================


def _resolve_nuclear_spin_ratio(point_group, symbols, equivalent_groups):
    """State-resolved nuclear-spin weights need an isotope/permutation contract."""
    raise SPCATBridgeError(
        "Nuclear-spin weights cannot be inferred from proton counts and point group alone. "
        "Supply a validated isotope- and symmetry-resolved statistical-weight model."
    )


def apply_symmetry_divisors(
    geometry_array: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    symbols: Sequence[str] | None = None,
    use_nuclear_spin: bool = False,
    enforce_guardrail: bool = True,
) -> SymmetryDivisorResult:
    """Resolve a classical symmetry divisor with MolSym or report unavailable.

    Coordinates are Angstroms. Nuclear-spin state weights require an independent
    isotope/permutation treatment and remain unavailable in this adapter.
    """
    flat_coords: list[float] = []
    if isinstance(geometry_array, np.ndarray):
        flat_coords = [float(x) for x in geometry_array.flatten()]
    else:
        for item in geometry_array:
            if isinstance(item, list | tuple | np.ndarray | Sequence):
                for x in item:
                    flat_coords.append(float(x))
            else:
                flat_coords.append(float(item))

    if len(flat_coords) % 3 != 0:
        raise SPCATBridgeError(
            message=f"Invalid flattened coordinate size {len(flat_coords)}, must be multiple of 3",
            error_code=ProvenanceErrorCode.SPCAT_BRIDGE_ERROR,
        )

    coords_np = np.array(flat_coords).reshape(-1, 3)
    num_atoms = coords_np.shape[0]

    if num_atoms == 0 or not np.isfinite(coords_np).all():
        raise SPCATBridgeError("Symmetry requires a nonempty finite geometry.")
    if symbols is None:
        raise SPCATBridgeError("Symmetry requires explicit atom and isotope identities.")
    elif len(symbols) != num_atoms:
        raise SPCATBridgeError(
            message=f"Symbols length ({len(symbols)}) does not match atom count ({num_atoms})",
            error_code=ProvenanceErrorCode.SPCAT_BRIDGE_ERROR,
        )

    if any(not re.fullmatch(r"[A-Z][a-z]?", s) or s in {"D", "T"} for s in symbols):
        raise SPCATBridgeError("Automatic isotope-resolved symmetry is not qualified in this adapter.")
    if use_nuclear_spin:
        raise SPCATBridgeError("Exact nuclear-spin weights require a validated state-resolved isotope/permutation model.")
    point_group = None
    sigma = None
    equivalent_groups: dict[str, list[int]] = {}

    if _MOLSYM_AVAILABLE:
        try:
            schema = {
                "symbols": [str(s).strip() for s in symbols],
                "geometry": (coords_np.reshape(-1) / (physical_constants["Bohr radius"][0] / 1e-10)).tolist(),
                "masses": [get_atomic_mass(s) for s in symbols],
            }
            mol = molsym.Molecule.from_schema(schema)
            try:
                sym = molsym.Symtext.from_molecule(mol)
                point_group = str(sym.pg).strip()
                sigma = int(sym.rotational_symmetry_number)
            except Exception:
                pg_info = molsym.find_point_group(mol)
                point_group = str(pg_info[0]).strip()
                sigma = _pg_to_sigma(point_group)

            # Extract symmetry equivalent atom sets
            try:
                seas = mol.find_SEAs()
                for idx, sea in enumerate(seas):
                    subset = [int(i) for i in getattr(sea, "subset", [])]
                    equivalent_groups[f"SEA_{idx}"] = subset
            except Exception as sea_err:
                logger.debug("MolSym find_SEAs non-fatal error: %s", sea_err)

        except Exception as err:
            raise SPCATBridgeError(f"MolSym symmetry analysis failed: {err}") from err
    else:
        point_group, sigma = _fallback_point_group_solver(coords_np, symbols)

    spin_weights, ratio_str = None, None

    effective_divisor = float(sigma)
    guardrail_status = "CLASSICAL_SIGMA_APPLIED; NUCLEAR_SPIN_WEIGHTS_UNAVAILABLE"

    return SymmetryDivisorResult(
        point_group=point_group,
        sigma=sigma,
        spin_statistical_weights=spin_weights,
        spin_weight_ratio_str=ratio_str,
        effective_divisor=effective_divisor,
        guardrail_status=guardrail_status,
        equivalent_atom_groups=equivalent_groups,
        metadata={
            "num_atoms": num_atoms,
            "symbols": list(symbols),
            "use_nuclear_spin": bool(use_nuclear_spin),
            "enforce_guardrail": bool(enforce_guardrail),
            "spin_weights_status": "unavailable: state-resolved model required",
            "symmetry_source": "MolSym",
        },
    )


def _fallback_point_group_solver(coords, symbols):
    """Unqualified geometric point-group guesses are not scientific results."""
    raise SPCATBridgeError("Symmetry determination unavailable: a validated point-group solver is required.")


def get_atomic_mass(symbol: str) -> float:
    """Resolve an isotope mass; bare elements select the most abundant isotope.

    Standard atomic weights are mixtures and cannot define one isotopologue.
    Elements without a natural-abundance default require an explicit isotope.
    """
    from Libraries.cochem_isotopes import isotope_mass
    return isotope_mass(symbol)


def calculate_rotational_constants_from_geometry(
    geometry: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    symbols: Sequence[str],
) -> dict[str, float | None]:
    """Calculate rotational constants A, B, C in MHz from Cartesian geometry and atomic symbols.

    Calculates center of mass, shifts coordinates to COM, constructs inertia tensor
    in u * Angstrom^2 using Mendeleev dynamic atomic masses, diagonalizes to find
    principal moments Ia <= Ib <= Ic, and computes:
        A = C_rot / Ia
        B = C_rot / Ib
        C = C_rot / Ic
    where C_rot = h / (8 * pi^2 * u * 1e-20) * 1e-6 MHz = CONSTANTS.C_ROT.
    """
    flat_coords: list[float] = []
    if isinstance(geometry, np.ndarray):
        flat_coords = [float(x) for x in geometry.flatten()]
    else:
        for item in geometry:
            if isinstance(item, list | tuple | np.ndarray | Sequence):
                for x in item:
                    flat_coords.append(float(x))
            else:
                flat_coords.append(float(item))

    coords = np.array(flat_coords, dtype=np.float64).reshape(-1, 3)
    num_atoms = coords.shape[0]
    if num_atoms < 2 or not np.isfinite(coords).all():
        raise ValueError("Rotational constants require at least two atoms with finite coordinates.")
    if len(symbols) != num_atoms:
        raise ValueError(
            f"Number of symbols ({len(symbols)}) does not match number of atoms ({num_atoms})"
        )

    masses = np.array([get_atomic_mass(s) for s in symbols], dtype=np.float64)
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Total molecular mass must be greater than zero.")

    # Center of mass
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    shifted = coords - com

    # Inertia tensor (u * Angstrom^2)
    x = shifted[:, 0]
    y = shifted[:, 1]
    z = shifted[:, 2]

    ixx = float(np.sum(masses * (y**2 + z**2)))
    iyy = float(np.sum(masses * (x**2 + z**2)))
    izz = float(np.sum(masses * (x**2 + y**2)))
    ixy = float(-np.sum(masses * x * y))
    ixz = float(-np.sum(masses * x * z))
    iyz = float(-np.sum(masses * y * z))

    i_tensor = np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=np.float64,
    )

    eigenvalues, _ = np.linalg.eigh(i_tensor)
    eigenvalues = np.sort(eigenvalues)

    ia, ib, ic = float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2])

    c_rot = CONSTANTS.C_ROT
    tolerance = max(ic, 1.0) * 1e-12
    if ia < -tolerance or ib <= tolerance or ic <= tolerance:
        raise ValueError("Geometry has singular or invalid rotational moments.")
    a_mhz = float(c_rot / ia) if ia > tolerance else None
    b_mhz = float(c_rot / ib)
    c_mhz = float(c_rot / ic)

    return {
        "A": a_mhz,
        "B": b_mhz,
        "C": c_mhz,
        "Ia": ia,
        "Ib": ib,
        "Ic": ic,
    }


# =============================================================================
# 5. Statistical Mechanics Partition Functions & Vibrational Coupling
# =============================================================================


def calculate_rotational_partition_function(
    a_mhz: float,
    b_mhz: float,
    c_mhz: float,
    temp_k: float,
    sigma: float = 1.0,
    is_linear: bool = False,
) -> float:
    """Calculate the classical high-temperature rigid-rotor approximation Q_rot(T).

    Formulations:
    - Asymmetric Top: Q_rot(T) = (sqrt(pi) / sigma) * (k_B * T / (h * 1e6))^(3/2) / sqrt(A * B * C)
    - Linear Rotor:   Q_rot(T) = (k_B * T) / (sigma * (h * 1e6) * B)

    Args:
        a_mhz: Rotational constant A in MHz.
        b_mhz: Rotational constant B in MHz.
        c_mhz: Rotational constant C in MHz.
        temp_k: Thermodynamic temperature in Kelvin.
        sigma: Rotational symmetry number (default: 1.0).
        is_linear: True if molecule is a linear rotor.

    Returns:
        Rotational partition function Q_rot(T) (dimensionless).
    """
    if not math.isfinite(temp_k) or temp_k <= 0:
        raise ValueError("Temperature must be finite and positive.")
    if not math.isfinite(sigma) or sigma < 1:
        raise ValueError("An explicit valid rotational symmetry divisor is required.")

    sigma_eff = float(sigma)
    kb_over_h_mhz = CONSTANTS.K_B / (CONSTANTS.H * 1e6)

    if is_linear:
        b_eff = float(b_mhz)
        if not math.isfinite(b_eff) or b_eff <= 0:
            raise ValueError("Linear-rotor B must be finite and positive.")
        return (kb_over_h_mhz * temp_k) / (sigma_eff * b_eff)

    if any(v is None or not math.isfinite(v) or v <= 0 for v in (a_mhz, b_mhz, c_mhz)):
        raise ValueError("Asymmetric-rotor A/B/C must be finite and positive.")
    a_eff, b_eff, c_eff = float(a_mhz), float(b_mhz), float(c_mhz)

    factor = (kb_over_h_mhz * temp_k) ** 1.5
    abc_sqrt = math.sqrt(a_eff * b_eff * c_eff)
    q_rot = (math.sqrt(math.pi) / sigma_eff) * (factor / abc_sqrt)
    return float(q_rot)


def calculate_vibrational_partition_function(
    frequencies_cm1: Sequence[float],
    temp_k: float,
    exclude_frequencies: Sequence[float] | None = None,
) -> float:
    """Calculate vibrational partition function Q_vib(T) referenced to ZPVE.

    Q_vib(T) = prod_{i, nu_i not in exclude} [ 1 / (1 - exp(- h * c * nu_i / (k_B * T))) ]

    Args:
        frequencies_cm1: Sequence of normal mode harmonic frequencies in cm^-1.
        temp_k: Thermodynamic temperature in Kelvin.
        exclude_frequencies: Frequencies to drop (e.g. LAM modes handled by DVR).

    Returns:
        Vibrational partition function Q_vib(T) (dimensionless).
    """
    if not math.isfinite(temp_k) or temp_k <= 0:
        raise ValueError("Temperature must be finite and positive.")
    if frequencies_cm1 is None:
        raise ValueError("Vibrational frequencies are unavailable.")

    excluded_set: list[float] = (
        [float(x) for x in exclude_frequencies] if exclude_frequencies else []
    )
    q_vib = 1.0
    hc_over_kb = CONSTANTS.HC_OVER_KB  # ~ 1.4387768775 K*cm

    for raw_f in frequencies_cm1:
        f = float(raw_f)
        if not math.isfinite(f) or f <= 0:
            raise ValueError("Partition functions require finite positive vibrational modes, with external modes already identified and removed.")
        matches = [i for i, excluded in enumerate(excluded_set) if abs(f - excluded) < 1e-8]
        if matches:
            excluded_set.pop(matches[0])
            continue

        x = (hc_over_kb * f) / temp_k
        if x > 500.0:
            factor = 1.0
        else:
            factor = 1.0 / -math.expm1(-x)

        q_vib *= factor

    if excluded_set:
        raise ValueError("A requested excluded mode was not present in the frequency list.")
    if not math.isfinite(q_vib):
        raise ValueError("Vibrational partition function overflowed.")
    return float(q_vib)


def vibrational_partition_coupling(
    q_rot_dvr, q_vib_orca, temp_array, lam_frequency=None, all_frequencies=None,
) -> dict[float, float]:
    """Multiply explicitly supplied partitions at each temperature.

    Sequence arguments contain partition values; frequencies must be named in
    all_frequencies. This prevents guessing a quantity from its magnitude.
    The caller is responsible for specifying nonoverlapping mode subspaces.
    """
    temps = [float(t) for t in temp_array]
    if not temps or any(not math.isfinite(t) or t <= 0 for t in temps):
        raise ValueError("Partition temperatures must be finite and positive.")
    if lam_frequency is not None and all_frequencies is None:
        raise ValueError("Dropping LAM modes requires explicit all_frequencies.")
    excluded = [] if lam_frequency is None else (
        [float(lam_frequency)] if np.isscalar(lam_frequency) else list(lam_frequency)
    )
    def value_at(values, idx, temperature, quantity):
        if callable(values):
            value = values(temperature)
        elif isinstance(values, dict):
            if temperature not in values:
                raise ValueError(f"Missing {quantity} at {temperature} K.")
            value = values[temperature]
        elif isinstance(values, (list, tuple, np.ndarray)):
            if len(values) != len(temps):
                raise ValueError(f"{quantity} must have one value per temperature.")
            value = values[idx]
        elif isinstance(values, (int, float)):
            value = values
        else:
            raise ValueError(f"Missing or unsupported {quantity} values.")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{quantity} must be finite and positive.")
        return value
    result = {}
    for idx, temperature in enumerate(temps):
        qr = value_at(q_rot_dvr, idx, temperature, "rotational/DVR partition")
        qv = (
            calculate_vibrational_partition_function(all_frequencies, temperature, excluded)
            if all_frequencies is not None else
            value_at(q_vib_orca, idx, temperature, "vibrational partition")
        )
        total = qr * qv
        if not math.isfinite(total):
            raise ValueError("Combined partition function overflowed.")
        result[temperature] = total
    return result


def compute_coupled_partition_functions(
    a_mhz: float,
    b_mhz: float,
    c_mhz: float,
    frequencies_cm1: Sequence[float],
    temp_array: Sequence[float],
    sigma: float = 1.0,
    lam_frequency: float | Sequence[float] | None = None,
    is_dvr: bool = False,
) -> PartitionFunctionResult:
    """Compute complete coupled partition functions with metadata tracking."""
    if is_dvr or lam_frequency is not None:
        raise ValueError("This helper has no DVR spectrum input. Use explicit nonoverlapping partition values with vibrational_partition_coupling.")
    temps = [float(t) for t in temp_array]
    q_rot_dict: dict[float, float] = {}
    q_vib_dict: dict[float, float] = {}
    q_total_dict: dict[float, float] = {}

    excluded: list[float] = []
    if lam_frequency is not None:
        if isinstance(lam_frequency, int | float):
            excluded.append(float(lam_frequency))
        else:
            excluded.extend([float(x) for x in lam_frequency])

    stiff = [
        f for f in frequencies_cm1 if not any(abs(f - ex) < 0.1 for ex in excluded)
    ]

    for t in temps:
        q_r = calculate_rotational_partition_function(
            a_mhz, b_mhz, c_mhz, t, sigma=sigma
        )
        q_v = calculate_vibrational_partition_function(
            frequencies_cm1, t, exclude_frequencies=excluded
        )
        q_rot_dict[t] = q_r
        q_vib_dict[t] = q_v
        q_total_dict[t] = q_r * q_v

    return PartitionFunctionResult(
        temperatures=temps,
        q_rot=q_rot_dict,
        q_vib=q_vib_dict,
        q_total=q_total_dict,
        dropped_lam_frequencies=excluded,
        stiff_frequencies=stiff,
        is_dvr_coupled=bool(is_dvr),
        metadata={"rotational_model": "classical_high_temperature_rigid_rotor", "vibrational_model": "harmonic_ZPVE_referenced"},
    )


# =============================================================================
# 6. Fortran Overflow Guard
# =============================================================================


def fortran_overflow_guard(
    tensor_dictionary: dict[str, Any] | Sequence[Any] | float | int | np.ndarray,
    max_limit: float = 1e308,
    clamp_on_overflow: bool = False,
) -> Any:
    """Trap values exceeding Double Precision mathematical ceilings (|val| > 1e308).

    Un-deperturbed resonances from VPT2 or divergent perturbation calculations can
    yield wildly oscillating constants that exceed Fortran REAL*8 limits (~10^308),
    causing SPCAT to crash or emit 'NON-POSITIVE DEFINITE' matrix errors.
    This guard actively scans incoming parameter tensors, logs a CRITICAL warning,
    and raises FortranOverflowError to block corrupted parameters.

    Args:
        tensor_dictionary: Dictionary, nested list, array, or scalar of parameters.
        max_limit: Hard double precision magnitude limit (default: 1e308).
        clamp_on_overflow: Legacy argument; True is rejected because it changes results.

    Returns:
        Validated data structure with original numerical values preserved.

    Raises:
        FortranOverflowError: If any value is nonfinite or exceeds max_limit.
    """

    if clamp_on_overflow:
        raise ValueError("Clamping invalid scientific parameters is prohibited; repair or recompute the source data.")
    if not math.isfinite(max_limit) or max_limit <= 0:
        raise ValueError("max_limit must be finite and positive.")
    def _inspect_and_guard(val: Any, path: str) -> Any:
        if isinstance(val, dict):
            return {
                k: _inspect_and_guard(v, f"{path}.{k}" if path else str(k))
                for k, v in val.items()
            }
        elif isinstance(val, list | tuple):
            return [
                _inspect_and_guard(item, f"{path}[{i}]") for i, item in enumerate(val)
            ]
        elif isinstance(val, np.ndarray):
            try:
                max_val = float(np.max(np.abs(val))) if val.size > 0 else 0.0
            except (TypeError, ValueError):
                raise ValueError(f"Parameter array {path!r} must be numeric.") from None

            if max_val > max_limit or math.isinf(max_val) or math.isnan(max_val):
                msg = (
                    f"CRITICAL: Fortran Double Precision overflow detected in array '{path}': "
                    f"max magnitude {max_val} exceeds limit {max_limit:.1e}"
                )
                logger.critical("[FORTRAN_OVERFLOW] %s", msg)
                raise FortranOverflowError(
                    message=msg,
                    error_code=ProvenanceErrorCode.FORTRAN_OVERFLOW,
                    details={
                        "path": path,
                        "max_magnitude": float(max_val),
                        "limit": float(max_limit),
                    },
                )
            return val
        elif isinstance(val, int | float):
            fval = float(val)
            if math.isinf(fval) or math.isnan(fval) or abs(fval) > max_limit:
                msg = (
                    f"CRITICAL: Fortran Double Precision overflow detected for parameter '{path}': "
                    f"value {fval} exceeds hard limit {max_limit:.1e}"
                )
                logger.critical("[FORTRAN_OVERFLOW] %s", msg)
                raise FortranOverflowError(
                    message=msg,
                    error_code=ProvenanceErrorCode.FORTRAN_OVERFLOW,
                    details={
                        "parameter": path,
                        "value": str(val),
                        "limit": float(max_limit),
                    },
                )
            return val
        return val

    return _inspect_and_guard(tensor_dictionary, "")


# =============================================================================
# 7. Fortran Double Precision Formatter & Alignment Engine
# =============================================================================


def format_fortran_double(
    val: float,
    width: int = 22,
    precision: int = 15,
    compact: bool = False,
) -> str:
    """Convert a Python float into strict Fortran Double Precision scientific notation ('D').

    Format Specification:
        1.567e-05 -> '1.567D-05' (compact) or ' 1.567000000000000D-05' (fixed width).

    Args:
        val: Numerical float value.
        width: Field width for right alignment (ignored if compact=True).
        precision: Decimal precision in mantissa.
        compact: If True, returns minimal scientific representation without trailing zeros.

    Returns:
        Formatted Fortran Double Precision string.
    """
    fval = float(val)
    fortran_overflow_guard(fval)
    if type(width) is not int or width < 1 or type(precision) is not int or precision < 0:
        raise ValueError("Fortran field width must be positive and precision nonnegative")
    if fval == 0.0:
        base = "0.000D+00" if compact else "0." + ("0" * precision) + "D+00"
        if not compact and len(base) > width:
            raise FortranOverflowError("Zero cannot fit the requested Fortran field width")
        return base if compact else f"{base:>{width}}"

    sci_str = f"{fval:.{precision}e}"
    if "e" in sci_str or "E" in sci_str:
        mantissa, exponent = sci_str.replace("E", "e").split("e")
        exp_int = int(exponent)
        exp_sign = "+" if exp_int >= 0 else "-"
        exp_formatted = f"{exp_sign}{abs(exp_int):02d}"
        if compact:
            parts = mantissa.split(".")
            if len(parts) == 2:
                dec = parts[1].rstrip("0")
                if len(dec) < 3:
                    dec = dec.ljust(3, "0")
                mantissa = f"{parts[0]}.{dec}"
            return f"{mantissa}D{exp_formatted}"
        else:
            raw_res = f"{mantissa}D{exp_formatted}"
            if len(raw_res) > width:
                overflow = len(raw_res) - width
                adj_prec = max(1, precision - overflow)
                sci_str = f"{fval:.{adj_prec}e}"
                mantissa, _ = sci_str.replace("E", "e").split("e")
                raw_res = f"{mantissa}D{exp_formatted}"
            if len(raw_res) > width:
                raise FortranOverflowError("Number cannot fit the requested Fortran field width")
            return f"{raw_res:>{width}}"

    formatted = f"{sci_str}".replace("e", "D").replace("E", "D")
    return formatted if compact else f"{formatted:>{width}}"


def fortran_double_precision_formatter(
    val_or_id: Any,
    val: float | None = None,
    uncertainty: float | None = None,
    label: str = "",
    width: int = 22,
    precision: int = 15,
    compact: bool = False,
) -> str | list[str]:
    """Format single floats, parameter lines, or parameter dictionaries into Pickett Fortran strings.

    Signatures supported:
    1. Single float value:
       `fortran_double_precision_formatter(0.00001567)` -> `'1.567D-05'`
    2. Parameter line:
       `fortran_double_precision_formatter(20000, 0.00001567, uncertainty=1e-7, label="DJ")`
       -> `'     20000   1.567000000000000D-05   1.000000000000000D-07  / DJ'`
    3. Dictionary of parameters:
       `fortran_double_precision_formatter({'20000': 1.567e-5, '10000': 435360.0})` -> list of lines

    Args:
        val_or_id: Numerical float value, integer parameter ID (e.g. 20000), or parameter dict.
        val: Parameter value when val_or_id is a parameter ID.
        uncertainty: Estimated uncertainty in MHz (default: 0.0).
        label: Descriptive comment label (e.g. 'DJ', 'A').
        width: Column width for numbers (default: 22).
        precision: Mantissa precision (default: 15).
        compact: If True, uses compact scientific notation (e.g. '1.567D-05').

    Returns:
        Formatted Fortran string or list of formatted lines.
    """
    if isinstance(val_or_id, dict):
        lines: list[str] = []
        for p_id, p_val in val_or_id.items():
            if isinstance(p_val, tuple | list):
                p_v = float(p_val[0])
                p_u = float(p_val[1]) if len(p_val) > 1 else None
                p_lbl = str(p_val[2]) if len(p_val) > 2 else ""
            else:
                p_v = float(p_val)
                p_u = None
                p_lbl = ""
            line = fortran_double_precision_formatter(
                val_or_id=p_id,
                val=p_v,
                uncertainty=p_u,
                label=p_lbl,
                width=width,
                precision=precision,
                compact=compact,
            )
            lines.append(str(line))
        return lines

    if val is not None:
        if uncertainty is None or not math.isfinite(float(uncertainty)) or uncertainty < 0:
            raise ValueError("Parameter uncertainty must be explicitly supplied, finite and nonnegative.")
        param_id_int = int(val_or_id)
        val_str = format_fortran_double(
            val, width=width, precision=precision, compact=compact
        )
        unc_str = format_fortran_double(
            uncertainty, width=width, precision=precision, compact=compact
        )
        lbl_part = f"  / {label}" if label else ""
        return f"{param_id_int:>10}  {val_str}  {unc_str}{lbl_part}"

    if isinstance(val_or_id, int | float):
        return format_fortran_double(
            float(val_or_id), width=width, precision=precision, compact=compact
        )

    raise TypeError("Fortran parameters must be numeric or explicitly typed parameter records.")


# =============================================================================
# 8. Pickett SPCAT .var and .int ASCII Generation
# =============================================================================

# Native parameter-code/reduction conventions remain unqualified. Absence must
# not be represented by an invented or partially incorrect mapping.
PICKETT_PARAMETER_CODES: dict[str, int] | None = None

def generate_spcat_var(
    molecule_name, parameters, title=None, nopt=0, nwarn=0, erpar=1.0,
    wtfac=1.0, scale=1.0, maxit=50, filepath=None,
) -> str:
    """Reject the legacy unqualified Pickett parameter writer.

    Its parameter-ID/reduction mapping and fitted uncertainty conventions have
    not passed a reference comparison. Generating native input is blocked until
    those contracts are implemented and validated against an installed SPCAT.
    """
    raise NotImplementedError(
        "Native SPCAT .var generation is unqualified. Use independently verified "
        "input decks with Libraries.cochem_torq_spcat.PickettSPCATRunner."
    )


def generate_spcat_int(
    molecule_name, dipoles, temperatures=298.15, tag=1, ver=1, ibx=0,
    nq=0, rrot=0.0, tem=0.0, sthk=0.0, wtk=0.0, title=None,
    filepath_template=None,
) -> dict[float, str]:
    """Reject native intensity input until its partition/unit schema is qualified."""
    raise NotImplementedError(
        "Native SPCAT .int generation is unqualified: explicit partition values, "
        "principal-axis dipoles and validated control-field conventions are required. "
        "Use independently verified decks with PickettSPCATRunner."
    )


def validate_airgap_boundary(target_path: str | Path) -> Path:
    """Validate that target output path adheres to the Tripartite Air-Gap isolation boundary.

    Ring 1: Static Repository Root (Domain A) is read-only for runtime scratch/log files.
    Directly writing volatile simulation scratch files into Ring 1 static repository
    (outside authorized test/scratch directories) raises an AirGapViolationError.

    Args:
        target_path: Target filesystem path to validate.

    Returns:
        Resolved absolute Path.

    Raises:
        AirGapViolationError: If target attempts to write directly into protected Ring 1 static root.
    """
    resolved = Path(target_path).resolve()
    base_root = get_base_root().resolve()
    repo_root = get_repo_root().resolve()

    # Check if target is located within static execution boundaries
    for root_dir in (base_root, repo_root):
        try:
            rel = resolved.relative_to(root_dir)
            rel_parts = rel.parts
            # If target is within CoChem-BASE root
            if rel_parts and rel_parts[0] == "CoChem-BASE":
                sub_parts = rel_parts[1:]
            else:
                sub_parts = rel_parts

            if sub_parts and sub_parts[0] in (
                "test_suite",
                "tests",
                ".pytest_cache",
                "scratch",
            ):
                return resolved

            raise AirGapViolationError(
                message=f"Air-Gap violation: forbidden write into Ring 1 static execution tier: {resolved}",
                error_code=ProvenanceErrorCode.AIRGAP_VIOLATION,
                details={
                    "path": str(resolved),
                    "ring": "Ring 1 (Domain A)",
                    "base_root": str(base_root),
                    "repo_root": str(repo_root),
                },
            )
        except ValueError:
            pass

    return resolved


def compute_sha256(content: str | bytes) -> str:
    """Compute deterministic SHA-256 hexadecimal hash string."""
    raw = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(raw).hexdigest()


def generate_spcat_provenance_manifest(
    molecule_name: str,
    var_content: str,
    int_contents: dict[float, str],
    symmetry_result: SymmetryDivisorResult,
    partition_results: dict[float, float],
    output_path: str | Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate SHA-256 cryptographic provenance manifest for SPCAT execution package."""
    sha256_var = compute_sha256(var_content)
    sha256_int = {str(t): compute_sha256(c) for t, c in int_contents.items()}

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "Stage 5.1 (Statistical Mechanics & SPCAT Bridge)",
        "molecule_name": molecule_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "codata_constants": {
            "h_j_s": CONSTANTS.H,
            "k_b_j_k": CONSTANTS.K_B,
            "c_cm_s": CONSTANTS.C_CM_S,
            "c_rot_mhz_u_ang2": CONSTANTS.C_ROT,
            "hc_over_kb_k_cm": CONSTANTS.HC_OVER_KB,
        },
        "symmetry": symmetry_result.to_dict(),
        "partition_functions": {str(k): v for k, v in partition_results.items()},
        "cryptographic_hashes": {
            "sha256_var": sha256_var,
            "sha256_int": sha256_int,
        },
        "airgap_rings": {
            "ring_1_domain_a": "Static Execution Tier (Read-Only Repo)",
            "ring_2_domain_c": "Ephemeral Scratch Tier (RAM-Disk /dev/shm)",
            "ring_3_domain_b": "Dynamic Artifact Vault ($COCHEM_ARTIFACTS_DIR)",
        },
        "metadata": extra_metadata if extra_metadata is not None else {},
    }

    if output_path is not None:
        target = Path(output_path).resolve()
        validate_airgap_boundary(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target.with_suffix(
            f".tmp_{os.getpid()}_{int(datetime.now().timestamp())}"
        )
        temp_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temp_file.replace(target)

    return manifest


def build_complete_spcat_payload(
    molecule_name, geometry, symbols, rotational_constants_mhz, dipoles_debye,
    harmonic_frequencies_cm1, temperatures=(2.0, 10.0, 50.0, 298.15),
    quartic_distortion=None, lam_frequency=None, use_nuclear_spin=False, output_dir=None,
) -> SPCATPayload:
    """Block unqualified native input generation before producing any files."""
    raise NotImplementedError(
        "Native SPCAT deck generation requires validated parameter, intensity, "
        "partition and state conventions. Use independently verified .var/.int "
        "decks with PickettSPCATRunner; no synthetic defaults are supplied."
    )


# =============================================================================
# 10. 3-Tier Routing Protocol (MPQC Primary, ORCA Secondary, CFOUR Legacy)
# =============================================================================


@dataclass
class ThreeTierRoutingResult:
    """Structured resolution of the 3-Tier Ab Initio Routing Protocol."""

    selected_tier: int
    primary_engine: str
    electronic_energy_hartree: float | None
    harmonic_frequencies: list[float] | None
    vpt2_x_matrix: np.ndarray | None
    dipole_moments_debye: dict[str, float] | None
    is_mpqc_primary: bool
    is_analytic_vpt2_active: bool
    routing_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize routing result to dictionary."""
        return {
            "selected_tier": self.selected_tier,
            "primary_engine": self.primary_engine,
            "electronic_energy_hartree": self.electronic_energy_hartree,
            "harmonic_frequencies": None if self.harmonic_frequencies is None else [float(f) for f in self.harmonic_frequencies],
            "vpt2_x_matrix": self.vpt2_x_matrix.tolist()
            if self.vpt2_x_matrix is not None
            else None,
            "dipole_moments_debye": self.dipole_moments_debye,
            "is_mpqc_primary": self.is_mpqc_primary,
            "is_analytic_vpt2_active": self.is_analytic_vpt2_active,
            "routing_metadata": self.routing_metadata,
        }


def route_3tier_abinitio_payload(
    mpqc_data=None, orca_data=None, cfour_data=None, require_analytic_vpt2=False,
) -> ThreeTierRoutingResult:
    """Select supplied records while preserving absence and actual method metadata.

    A matrix alone does not establish analytic VPT2. Analytic provenance must be
    explicit in the supplied record. This function does not run an engine.
    """
    candidates = [(1, "MPQC", mpqc_data), (2, "ORCA", orca_data), (3, "CFOUR", cfour_data)]
    for tier, engine, data in candidates:
        if data is None:
            continue
        if not isinstance(data, dict) or not data:
            raise ValueError(f"{engine} result is empty or malformed.")
        energy = data.get("energy_hartree", data.get("electronic_energy", data.get("eccsd_t", data.get("ccsd_t_f12_energy"))))
        frequencies = data.get("frequencies", data.get("harmonic_frequencies"))
        dipoles = data.get("dipoles", data.get("dipole_moments"))
        matrix = data.get("x_matrix", data.get("anharmonic_x_matrix"))
        if energy is not None and not math.isfinite(float(energy)):
            raise ValueError("Electronic energy must be finite when provided.")
        if frequencies is not None:
            frequencies = [float(f) for f in frequencies]
            if not np.isfinite(frequencies).all():
                raise ValueError("Frequencies must be finite when provided.")
        if dipoles is not None:
            if data.get("dipole_coordinate_frame") != "principal_axes" or data.get("dipole_units") != "debye":
                raise ValueError("Spectroscopic dipoles require explicit principal-axis/debye provenance.")
            if not isinstance(dipoles, dict) or not any(set(keys) <= set(dipoles) for keys in [("a", "b", "c"), ("mu_a", "mu_b", "mu_c")]):
                raise ValueError("Dipole vector must contain all three explicitly provided components.")
            if not all(math.isfinite(float(value)) for value in dipoles.values()):
                raise ValueError("Dipole components must be finite.")
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0 or not np.isfinite(matrix).all():
                raise ValueError("VPT2 matrix must be a nonempty finite square matrix.")
            if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
                raise ValueError("VPT2 matrix must be symmetric in its stated mode convention.")
        analytic = matrix is not None and data.get("vpt2_derivative_kind") == "analytic"
        if require_analytic_vpt2 and not analytic:
            continue
        if all(value is None for value in (energy, frequencies, dipoles, matrix)):
            raise ValueError(f"{engine} record contains no scientific observables.")
        return ThreeTierRoutingResult(
            selected_tier=tier, primary_engine=engine,
            electronic_energy_hartree=float(energy) if energy is not None else None,
            harmonic_frequencies=frequencies, vpt2_x_matrix=matrix,
            dipole_moments_debye=dipoles, is_mpqc_primary=engine == "MPQC",
            is_analytic_vpt2_active=analytic,
            routing_metadata={"selection": "supplied record", "method": data.get("method"), "raw": data},
        )
    raise ValueError("No supplied record meets the requested observable/provenance requirements.")


# =============================================================================
# 11. TorqSpcatBridge Compatibility Adapter
# =============================================================================


class TorqSpcatBridge:
    """Read explicitly typed spectroscopic data; native file generation is gated."""

    def __init__(self, tensor_json_path, mpqc_out_path, temperature_k=298.15):
        self.tensor_file = Path(tensor_json_path)
        self.mpqc_file = Path(mpqc_out_path)
        self.orca_file = self.mpqc_file
        self.temperature = self.temperature_k = float(temperature_k)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive.")
        self.tensor_data = self._load_json(self.tensor_file)
        self.point_id = str(self.tensor_data.get("point_id", self.tensor_file.stem))
        if type(self.tensor_data.get("is_linear")) is not bool:
            raise ValueError("Explicit rotor classification is required.")
        self.is_linear = self.tensor_data["is_linear"]
        constants = self.tensor_data.get("tensors", {}).get("rotational_constants_MHz")
        if not isinstance(constants, dict):
            raise ValueError("Rotational constants are unavailable.")
        for axis in (("B", "C") if self.is_linear else ("A", "B", "C")):
            if axis not in constants or constants[axis] is None or not math.isfinite(float(constants[axis])) or float(constants[axis]) <= 0:
                raise ValueError(f"Missing or invalid rotational constant {axis}.")
        self.rot_A_MHz = None if self.is_linear else float(constants["A"])
        self.rot_B_MHz, self.rot_C_MHz = float(constants["B"]), float(constants["C"])
        self.sigma = self._determine_symmetry_divisor()
        self.frequencies_cm1 = None
        self.dipole_moments = None

    def _load_json(self, filepath):
        def reject_constant(token):
            raise ValueError(f"Nonfinite JSON number {token} is prohibited.")
        with open(filepath, encoding="utf-8") as source:
            data = json.load(source, parse_constant=reject_constant)
        if not isinstance(data, dict) or not data:
            raise ValueError("Tensor JSON must be a nonempty object.")
        return data

    def _determine_symmetry_divisor(self):
        symmetry = self.tensor_data.get("symmetry")
        if isinstance(symmetry, dict) and symmetry.get("source"):
            sigma = symmetry.get("sigma")
            if type(sigma) is int and sigma >= 1:
                return sigma
            raise ValueError("Explicit symmetry divisor must be a positive integer.")
        coordinates, symbols = self.tensor_data.get("coordinates"), self.tensor_data.get("symbols")
        if coordinates is None or symbols is None:
            raise ValueError("Symmetry provenance or a resolvable geometry is required.")
        return apply_symmetry_divisors(coordinates, symbols).sigma

    def parse_mpqc_observables(self):
        raise NotImplementedError(
            "The mixed MPQC/ORCA text parser is unqualified. It did not establish "
            "dipole units, the principal-axis transformation or vibrational-mode identity. "
            "Use a validated engine property adapter."
        )

    def calculate_partition_functions(self):
        if self.frequencies_cm1 is None:
            raise ValueError("Vibrational characterization is unavailable; Q_vib was not replaced by 1.")
        qr = calculate_rotational_partition_function(
            self.rot_A_MHz, self.rot_B_MHz, self.rot_C_MHz,
            self.temperature_k, float(self.sigma), self.is_linear,
        )
        qv = calculate_vibrational_partition_function(self.frequencies_cm1, self.temperature_k)
        return qr, qv, qr * qv

    def generate_spcat_files(self):
        raise NotImplementedError("Native SPCAT file generation is unqualified; use verified decks with PickettSPCATRunner.")

    def export_spcat_catalog(self):
        self.generate_spcat_files()


__all__ = [
    "AirGapViolationError",
    "FortranOverflowError",
    "LAMTriggerError",
    "ProvenanceErrorCode",
    "SPCATBridgeError",
    "ThreeTierRoutingResult",
    "route_3tier_abinitio_payload",
    "TorqSpcatBridge",
    "CODATA2022",
    "CONSTANTS",
    "CODATA_YEAR",
    "PLANCK_CONSTANT_JS",
    "BOLTZMANN_CONSTANT_JK",
    "SPEED_OF_LIGHT_CMS",
    "SPEED_OF_LIGHT_MS",
    "ROTATIONAL_FACTOR_C_ROT",
    "C_ROT",
    "HC_OVER_KB",
    "KB_OVER_H",
    "SymmetryDivisorResult",
    "PartitionFunctionResult",
    "SPCATParameter",
    "SPCATPayload",
    "PICKETT_PARAMETER_CODES",
    "low_frequency_lam_trap",
    "low_frequency_trap",
    "apply_symmetry_divisors",
    "calculate_rotational_partition_function",
    "calculate_vibrational_partition_function",
    "vibrational_partition_coupling",
    "compute_coupled_partition_functions",
    "fortran_overflow_guard",
    "format_fortran_double",
    "fortran_double_precision_formatter",
    "generate_spcat_var",
    "generate_spcat_int",
    "validate_airgap_boundary",
    "compute_sha256",
    "generate_spcat_provenance_manifest",
    "build_complete_spcat_payload",
    "get_atomic_mass",
    "calculate_rotational_constants_from_geometry",
]
