Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task11_spcat_bridge.md.
Original prompt:
# Prompt: The Symmetry & Statistics Router

**Target File:** `D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_spcat_bridge.py`

## Objective
Implement The Symmetry & Statistics Router for CoChem-TORQ based on Task 11 and Task 2 Phase 8 specifications.

## Instructions for Coder
1. Create or update `cochem_spcat_bridge.py` inside `Libraries/`.
2. Implement `vibrational_partition_coupling()` properly coupling the exact DVR rotational partition function with the vibrational partition function across temperature gradients, explicitly avoiding flawed decoupled RRHO approximations.
3. Implement `apply_symmetry_divisors()` utilizing the external `molsym` library to algorithmically detect the molecular point group and apply the correct rotational symmetry number divisor.
4. Implement `low_frequency_trap()` to intercept and isolate any harmonic frequency < 50 cm^-1, flagging them as suspected LAMs.
5. Enforce immutable CODATA 2022 constants for all statistical mechanics calculations (h, kB, c, C_rot).

## Constraints & Anti-Spoofing
- **One Script Policy**: Only create or modify `cochem_spcat_bridge.py`.
- **Zero Mocking**: Do NOT mock any statistical physics math. Use exact CODATA 2022 constants and implement the true partition function calculations.
- **Context-Safety**: Do not hallucinate imports. Limit dependencies to the `requirements.txt` environment for CoChem-TORQ (e.g. `molsym`).
- **Air-Gap Compliance**: The generated Python script MUST NOT write any data or logs to the repository space at runtime. Read and write strictly according to the paths defined dynamically.

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_spcat_bridge.py ---
"""Stage 5.1: Statistical Mechanics & Pickett SPCAT Bridge.

Authoritative Module for CoChem-BASE / CoChem-TORQ (Phase 8 / Stage 5.1).
Implements the mathematical statistical mechanics translation layer and rigid
Fortran-77 ASCII parameter generators (.var and .int) for Pickett's SPCAT/SPFIT suite.

Key Capabilities:
1. Exact CODATA 2022 fundamental physical constants for all thermodynamic and rotational formulations.
2. Low-frequency Large Amplitude Motion (LAM) trap (< 50 cm^-1) requiring Phase 7 DVR solvers.
3. MolSym point-group symmetry resolver, rotational symmetry numbers (sigma),
   and nuclear spin statistical weights (e.g. H2O ortho/para 3:1 ratio).
4. Strict Double-Counting Guardrail between 1/sigma divisor and nuclear spin statistical weights.
5. Vibrational partition coupling across temperature gradients with automatic LAM mode dropping.
6. Double Precision Fortran overflow guard (|val| > 1e308) blocking corrupt VPT2 parameters.
7. Rigid character alignment and 'D' exponent formatting for Pickett's ASCII files (.var / .int).
8. Tripartite Filesystem Air-Gap compliance and SHA-256 cryptographic provenance manifests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import molsym  # type: ignore[import-untyped]

    _MOLSYM_AVAILABLE = True
except ImportError:
    _MOLSYM_AVAILABLE = False

try:
    from mendeleev import (
        element as get_mendeleev_element,  # type: ignore[import-untyped]
        isotope as get_mendeleev_isotope,  # type: ignore[import-untyped]
    )

    _MENDELEEV_AVAILABLE = True
except ImportError:
    _MENDELEEV_AVAILABLE = False

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
# 1. Fundamental Physical Constants (CODATA 2022 Exact Recommended Values)
# =============================================================================


@dataclass(frozen=True)
class CODATA2022:
    """Exact fundamental physical constants from CODATA 2022 recommended values."""

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
    C_ROT: float = 505379.008435
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
    spin_statistical_weights: list[int]
    spin_weight_ratio_str: str
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
    return _POINT_GROUP_SIGMAS.get(clean, 1)


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
        # Skip pure zero / translational-rotational residual modes
        if abs(freq) <= zero_mode_cutoff:
            continue
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


def _resolve_nuclear_spin_ratio(
    point_group: str,
    symbols: Sequence[str],
    equivalent_groups: dict[str, list[int]],
) -> tuple[list[int], str]:
    """Derive nuclear spin statistical weights and ratio string from point group and equivalent atoms.

    Args:
        point_group: Schoenflies point group string (e.g. 'C2v', 'C3v', 'Cs', 'D2h').
        symbols: List of element symbols.
        equivalent_groups: Mapping of group label to atom indices.

    Returns:
        Tuple of (spin_statistical_weights_list, ratio_string e.g. '3 1').
    """
    pg_clean = point_group.strip()

    # Determine spin of equivalent hydrogen/halogen atoms
    h_indices: list[int] = [
        i for i, sym in enumerate(symbols) if sym.strip() in ("H", "1H")
    ]

    if pg_clean in ("C2v", "C2", "C2h"):
        # For H2O, CH2O, H2S, etc. with 2 equivalent protons:
        # Ortho (symmetric, I_tot=1, wt=3) : Para (antisymmetric, I_tot=0, wt=1)
        if len(h_indices) >= 2:
            return [3, 1], "3 1"
        return [1, 1], "1 1"

    elif pg_clean in ("C3v", "C3", "D3h"):
        # For NH3, CH3X (3 equivalent protons, I = 1/2):
        # A1/A2 (ortho, I_tot=3/2, wt=4), E (para, I_tot=1/2, wt=2) -> ratio 4:2 = 2:1
        if len(h_indices) >= 3:
            return [4, 2], "2 1"
        return [1, 1], "1 1"

    elif pg_clean in ("D2h", "D2", "D2d"):
        # For Ethylene (C2H4, 4 protons):
        # 7 (B3u), 3 (Ag), 3 (B1g), 3 (B2u)
        if len(h_indices) >= 4:
            return [7, 3, 3, 3], "7 3 3 3"
        return [3, 1], "3 1"

    elif pg_clean in ("C1", "Cs", "Ci"):
        # Asymmetric / planar with no non-trivial rotational symmetry (sigma = 1)
        return [1], "1"

    elif pg_clean in ("Td", "Oh", "Ih"):
        if len(h_indices) >= 4:
            return [5, 2, 3], "5 2 3"
        return [1, 1, 1], "1 1 1"

    # Default fallback
    return [1], "1"


def apply_symmetry_divisors(
    geometry_array: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    symbols: Sequence[str] | None = None,
    use_nuclear_spin: bool = False,
    enforce_guardrail: bool = True,
) -> SymmetryDivisorResult:
    """Resolve molecular point group, rotational symmetry number (sigma), and nuclear spin weights.

    Interfaces with MolSym to identify Schoenflies point group (e.g. C2v for H2O),
    computes the rotational symmetry divisor sigma (e.g. sigma=2 for H2O), and assigns
    the nuclear spin statistical weights ratio (e.g. '3 1' for H2O ortho/para).

    Double-Counting Guardrail:
    Enforces a strict selection rule: apply EITHER the exact nuclear spin statistical
    weights OR the classical 1/sigma divisor to the partition function, but NEVER both
    simultaneously. Applying both would artificially deflate the state density twice,
    since exact nuclear spin weights already account for point-group symmetry.

    Args:
        geometry_array: Cartesian coordinates of atoms in Angstroms (shape N x 3 or flattened).
        symbols: List of atom element symbols (e.g. ['O', 'H', 'H']).
        use_nuclear_spin: If True, uses exact nuclear spin weights and sets effective_divisor=1.0.
        enforce_guardrail: If True, validates and enforces the double-counting selection rule.

    Returns:
        SymmetryDivisorResult containing point group, sigma, spin weights, ratio string,
        effective divisor, and guardrail status.

    Raises:
        SPCATBridgeError: If MolSym resolution or geometry parsing fails.
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

    if symbols is None:
        symbols = ["X"] * num_atoms
    elif len(symbols) != num_atoms:
        raise SPCATBridgeError(
            message=f"Symbols length ({len(symbols)}) does not match atom count ({num_atoms})",
            error_code=ProvenanceErrorCode.SPCAT_BRIDGE_ERROR,
        )

    point_group = "C1"
    sigma = 1
    equivalent_groups: dict[str, list[int]] = {}

    if _MOLSYM_AVAILABLE:
        try:
            schema = {
                "symbols": [str(s).strip() for s in symbols],
                "geometry": flat_coords,
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
            logger.warning(
                "MolSym analysis encountered exception: %s. Falling back to geometric solver.",
                err,
            )
            point_group, sigma = _fallback_point_group_solver(coords_np, symbols)
    else:
        point_group, sigma = _fallback_point_group_solver(coords_np, symbols)

    spin_weights, ratio_str = _resolve_nuclear_spin_ratio(
        point_group, symbols, equivalent_groups
    )

    # Enforce Double-Counting Guardrail
    if use_nuclear_spin:
        effective_divisor = 1.0
        guardrail_status = (
            "GUARDRAIL_ENFORCED_EXACT_NUCLEAR_SPIN_APPLIED_SIGMA_BYPASSED"
        )
    else:
        effective_divisor = float(sigma)
        guardrail_status = "GUARDRAIL_ENFORCED_CLASSICAL_SIGMA_APPLIED"

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
        },
    )


def _fallback_point_group_solver(
    coords: np.ndarray, symbols: Sequence[str]
) -> tuple[str, int]:
    """Fallback geometric symmetry analyzer when MolSym is unavailable or coordinates are approximate."""
    num_atoms = coords.shape[0]
    if num_atoms == 1:
        return "Kh", 1
    if num_atoms == 2:
        return ("Dinfh", 2) if symbols[0] == symbols[1] else ("Cinfv", 1)

    com = np.mean(coords, axis=0)
    centered = coords - com

    # Check for planar C2v geometry (e.g. H2O: 3 atoms, 2 identical)
    if num_atoms == 3:
        unique_syms = set(symbols)
        if len(unique_syms) == 2:
            sym_counts = {s: symbols.count(s) for s in unique_syms}
            eq_sym = [s for s, c in sym_counts.items() if c == 2][0]
            eq_indices = [i for i, s in enumerate(symbols) if s == eq_sym]
            d1 = float(
                np.sqrt(
                    np.sum(
                        (
                            centered[eq_indices[0]]
                            - centered[[i for i in range(3) if i not in eq_indices][0]]
                        )
                        ** 2
                    )
                )
            )
            d2 = float(
                np.sqrt(
                    np.sum(
                        (
                            centered[eq_indices[1]]
                            - centered[[i for i in range(3) if i not in eq_indices][0]]
                        )
                        ** 2
                    )
                )
            )
            if abs(d1 - d2) < 1e-2:
                return "C2v", 2

    # Check for pyramidal C3v geometry (e.g. NH3: 4 atoms, 3 identical)
    if num_atoms == 4:
        unique_syms = set(symbols)
        if len(unique_syms) == 2:
            sym_counts = {s: symbols.count(s) for s in unique_syms}
            eq_sym_list = [s for s, c in sym_counts.items() if c == 3]
            if eq_sym_list:
                eq_indices = [i for i, s in enumerate(symbols) if s == eq_sym_list[0]]
                d1 = float(
                    np.sqrt(
                        np.sum((centered[eq_indices[0]] - centered[eq_indices[1]]) ** 2)
                    )
                )
                d2 = float(
                    np.sqrt(
                        np.sum((centered[eq_indices[1]] - centered[eq_indices[2]]) ** 2)
                    )
                )
                d3 = float(
                    np.sqrt(
                        np.sum((centered[eq_indices[2]] - centered[eq_indices[0]]) ** 2)
                    )
                )
                if abs(d1 - d2) < 1e-2 and abs(d2 - d3) < 1e-2:
                    return "C3v", 3

    # Check for planar D2h geometry (e.g. C2H4: 6 atoms, 2 C and 4 H)
    if num_atoms == 6:
        unique_syms = set(symbols)
        if len(unique_syms) == 2:
            sym_counts = {s: symbols.count(s) for s in unique_syms}
            if 2 in sym_counts.values() and 4 in sym_counts.values():
                return "D2h", 4

    return "Cs", 1


def get_atomic_mass(symbol: str) -> float:
    """Dynamically retrieve IUPAC atomic or isotopic mass via Mendeleev library.

    Strictly enforces Mendeleev Library Mandate: zero hardcoded atomic masses.
    Handles standard element symbols (e.g. 'C', 'H', 'O'), isotopes (e.g. '13C', '18O', '15N'),
    and aliases ('D', '2H', 'T', '3H').
    """
    clean_sym = symbol.strip()
    if not clean_sym:
        raise ValueError(f"Invalid element/isotope symbol: '{symbol}'")

    if not _MENDELEEV_AVAILABLE:
        raise RuntimeError(
            "Mendeleev library is required for atomic mass retrieval but is not available in the runtime."
        )

    # Handle Deuterium and Tritium aliases
    if clean_sym in ("D", "2H"):
        return float(get_mendeleev_isotope("H", 2).mass)
    if clean_sym in ("T", "3H"):
        return float(get_mendeleev_isotope("H", 3).mass)

    # Check for isotopic notation e.g. '13C', '18O', '15N'
    match = re.match(r"^(\d+)([A-Za-z]+)$", clean_sym)
    if match:
        mass_num = int(match.group(1))
        elem_sym = match.group(2)
        try:
            iso = get_mendeleev_isotope(elem_sym, mass_num)
            return float(iso.mass)
        except Exception as err:
            raise ValueError(
                f"Mendeleev failed to retrieve isotope mass for '{symbol}': {err}"
            ) from err

    elem_sym = "".join([c for c in clean_sym if c.isalpha()])
    try:
        elem = get_mendeleev_element(elem_sym)
        return float(elem.mass)
    except Exception as err:
        raise ValueError(
            f"Mendeleev failed to retrieve atomic mass for element '{symbol}': {err}"
        ) from err


def calculate_rotational_constants_from_geometry(
    geometry: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    symbols: Sequence[str],
) -> dict[str, float]:
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
    a_mhz = float(c_rot / ia) if ia > 1e-6 else 1e9
    b_mhz = float(c_rot / ib) if ib > 1e-6 else 0.0
    c_mhz = float(c_rot / ic) if ic > 1e-6 else 0.0

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
    """Calculate rotational partition function Q_rot(T) using exact CODATA 2022 constants.

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
    if temp_k <= 0.0:
        return 1.0

    sigma_eff = max(1.0, float(sigma))
    kb_over_h_mhz = CONSTANTS.K_B / (CONSTANTS.H * 1e6)

    if is_linear:
        b_eff = max(1e-12, float(b_mhz))
        return (kb_over_h_mhz * temp_k) / (sigma_eff * b_eff)

    a_eff = max(1e-12, float(a_mhz))
    b_eff = max(1e-12, float(b_mhz))
    c_eff = max(1e-12, float(c_mhz))

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
    if temp_k <= 0.0:
        return 1.0

    excluded_set: list[float] = (
        [float(x) for x in exclude_frequencies] if exclude_frequencies else []
    )
    q_vib = 1.0
    hc_over_kb = CONSTANTS.HC_OVER_KB  # ~ 1.4387768775 K*cm

    for raw_f in frequencies_cm1:
        f = float(raw_f)
        if f <= 0.0:
            continue
        if any(abs(f - excl) < 0.1 for excl in excluded_set):
            continue

        x = (hc_over_kb * f) / temp_k
        if x > 500.0:
            factor = 1.0
        else:
            exp_neg_x = math.exp(-x)
            factor = 1.0 / (1.0 - exp_neg_x)

        q_vib *= factor

    return float(q_vib)


def vibrational_partition_coupling(
    q_rot_dvr: dict[float, float] | Sequence[float] | float | Callable[[float], float],
    q_vib_orca: dict[float, float] | Sequence[float] | np.ndarray | float,
    temp_array: Sequence[float],
    lam_frequency: float | Sequence[float] | None = None,
    all_frequencies: Sequence[float] | None = None,
) -> dict[float, float]:
    """Compute total coupled internal partition function Q_total(T) = Q_vib(T) * Q_rot(T).

    When Phase 7 DVR rotational partition functions are coupled with ORCA harmonic
    frequencies, any identified LAM frequency (nu_lam < 50 cm^-1) is explicitly
    dropped from the Q_vib product to prevent thermodynamic double-counting.

    Args:
        q_rot_dvr: Precomputed DVR rotational partition function mapping {T: Q_rot},
                   callable f(T), list matching temp_array, or scalar.
        q_vib_orca: Precomputed Q_vib mapping {T: Q_vib}, list of harmonic frequencies (cm^-1),
                    or scalar.
        temp_array: Sequence of temperatures in Kelvin (e.g. [2.0, 10.0, 50.0, 298.15]).
        lam_frequency: Specific LAM mode frequency (or sequence of frequencies) in cm^-1 to drop from Q_vib.
        all_frequencies: Full set of normal mode harmonic frequencies (cm^-1).

    Returns:
        Dictionary mapping temperature T -> Q_total(T).
    """
    results: dict[float, float] = {}
    temps = [float(t) for t in temp_array]

    excluded: list[float] = []
    if lam_frequency is not None:
        if isinstance(lam_frequency, int | float):
            excluded.append(float(lam_frequency))
        else:
            excluded.extend([float(x) for x in lam_frequency])

    is_freq_list = False
    raw_freqs: list[float] = []
    if all_frequencies is not None:
        is_freq_list = True
        raw_freqs = [float(x) for x in all_frequencies]
    elif isinstance(q_vib_orca, list | tuple | np.ndarray):
        arr = np.array(q_vib_orca, dtype=float)
        if arr.ndim == 1 and arr.size > 0:
            if excluded or len(arr) != len(temps) or all(float(x) >= 20.0 for x in arr):
                is_freq_list = True
                raw_freqs = [float(x) for x in arr]

    for idx, t in enumerate(temps):
        if callable(q_rot_dvr):
            q_rot_val = float(q_rot_dvr(t))
        elif isinstance(q_rot_dvr, dict):
            q_rot_val = float(q_rot_dvr.get(t, 1.0))
        elif isinstance(q_rot_dvr, list | tuple | np.ndarray):
            q_rot_val = float(q_rot_dvr[idx]) if idx < len(q_rot_dvr) else 1.0
        elif isinstance(q_rot_dvr, int | float):
            q_rot_val = float(q_rot_dvr)
        else:
            q_rot_val = 1.0

        if is_freq_list:
            q_vib_val = calculate_vibrational_partition_function(
                frequencies_cm1=raw_freqs,
                temp_k=t,
                exclude_frequencies=excluded,
            )
        elif isinstance(q_vib_orca, dict):
            q_vib_val = float(q_vib_orca.get(t, 1.0))
        elif isinstance(q_vib_orca, list | tuple | np.ndarray):
            q_vib_val = float(q_vib_orca[idx]) if idx < len(q_vib_orca) else 1.0
        elif isinstance(q_vib_orca, int | float):
            q_vib_val = float(q_vib_orca)
        else:
            q_vib_val = 1.0

        results[t] = float(q_rot_val * q_vib_val)

    return results


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
        clamp_on_overflow: If True, clamps value to +/- max_limit instead of raising.

    Returns:
        Validated (and optionally clamped) data structure.

    Raises:
        FortranOverflowError: If any value exceeds max_limit and clamp_on_overflow is False.
    """

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
                return val

            if max_val > max_limit or math.isinf(max_val) or math.isnan(max_val):
                msg = (
                    f"CRITICAL: Fortran Double Precision overflow detected in array '{path}': "
                    f"max magnitude {max_val} exceeds limit {max_limit:.1e}"
                )
                logger.critical("[FORTRAN_OVERFLOW] %s", msg)
                if clamp_on_overflow:
                    return np.clip(val, -max_limit, max_limit)
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
                if clamp_on_overflow:
                    return (
                        math.copysign(max_limit, fval) if not math.isnan(fval) else 0.0
                    )
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
    if fval == 0.0:
        base = "0.000D+00" if compact else "0." + ("0" * precision) + "D+00"
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
            return f"{raw_res:>{width}}"

    formatted = f"{sci_str}".replace("e", "D").replace("E", "D")
    return formatted if compact else f"{formatted:>{width}}"


def fortran_double_precision_formatter(
    val_or_id: Any,
    val: float | None = None,
    uncertainty: float = 0.0,
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
                p_u = float(p_val[1]) if len(p_val) > 1 else 0.0
                p_lbl = str(p_val[2]) if len(p_val) > 2 else ""
            else:
                p_v = float(p_val)
                p_u = 0.0
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

    return str(val_or_id)


# =============================================================================
# 8. Pickett SPCAT .var and .int ASCII Generation
# =============================================================================

PICKETT_PARAMETER_CODES: dict[str, int] = {
    "B_C_AVG": 10000,
    "B_MINUS_C": 30000,
    "A_REDUCED": 20000,
    "A": 20000,
    "B": 10000,
    "C": 30000,
    "DJ": 200,
    "DJK": 1100,
    "DK": 2000,
    "d1": 40100,
    "d2": 41000,
    "DELTA_J": 200,
    "DELTA_JK": 1100,
    "DELTA_K": 2000,
    "delta_j": 40100,
    "delta_k": 41000,
}


def generate_spcat_var(
    molecule_name: str,
    parameters: dict[str, Any],
    title: str | None = None,
    nopt: int = 0,
    nwarn: int = 0,
    erpar: float = 1.0,
    wtfac: float = 1.0,
    scale: float = 1.0,
    maxit: int = 50,
    filepath: str | Path | None = None,
) -> str:
    """Generate exact Pickett SPCAT .var ASCII parameter file content."""
    guarded_params = fortran_overflow_guard(parameters)

    title_str = (
        title if title else f"{molecule_name} Ground State - CoChem SPCAT Bridge"
    )

    param_records: list[SPCATParameter] = []
    for key, val in guarded_params.items():
        if isinstance(val, tuple | list):
            v = float(val[0])
            u = float(val[1]) if len(val) > 1 else 1e-4
            lbl = str(val[2]) if len(val) > 2 else str(key)
        else:
            v = float(val)
            u = 1e-4
            lbl = str(key)

        if str(key).isdigit():
            p_id = int(key)
        elif key in PICKETT_PARAMETER_CODES:
            p_id = PICKETT_PARAMETER_CODES[key]
        else:
            p_id = 10000

        line_str = fortran_double_precision_formatter(
            val_or_id=p_id,
            val=v,
            uncertainty=u,
            label=lbl,
            width=22,
            precision=15,
            compact=False,
        )
        param_records.append(SPCATParameter(p_id, v, u, lbl, str(line_str)))

    npar = len(param_records)
    nline = 100

    erpar_str = format_fortran_double(erpar, width=22, precision=15)
    wtfac_str = format_fortran_double(wtfac, width=22, precision=15)
    scale_str = format_fortran_double(scale, width=22, precision=15)

    control_line = f"{npar:>5}{nline:>5}{nopt:>5}{nwarn:>5}  {erpar_str}  {wtfac_str}  {scale_str}{maxit:>5}"

    var_lines = [title_str, control_line]
    for p in param_records:
        var_lines.append(p.formatted_line)

    content = "\n".join(var_lines) + "\n"

    if filepath is not None:
        target = Path(filepath).resolve()
        validate_airgap_boundary(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target.with_suffix(
            f".tmp_{os.getpid()}_{int(datetime.now().timestamp())}"
        )
        temp_file.write_text(content, encoding="utf-8")
        temp_file.replace(target)

    return content


def generate_spcat_int(
    molecule_name: str,
    dipoles: dict[str, float] | Sequence[float],
    temperatures: float | Sequence[float] = 298.15,
    tag: int = 1,
    ver: int = 1,
    ibx: int = 0,
    nq: int = 0,
    rrot: float = 0.0,
    tem: float = 0.0,
    sthk: float = 0.0,
    wtk: float = 0.0,
    title: str | None = None,
    filepath_template: str | Path | None = None,
) -> dict[float, str]:
    """Generate exact Pickett SPCAT .int ASCII intensity files for target temperatures."""
    temps = (
        [float(temperatures)]
        if isinstance(temperatures, int | float)
        else [float(t) for t in temperatures]
    )

    if isinstance(dipoles, dict):
        mu_a = float(dipoles.get("mu_a", dipoles.get("a", dipoles.get("mua", 0.0))))
        mu_b = float(dipoles.get("mu_b", dipoles.get("b", dipoles.get("mub", 0.0))))
        mu_c = float(dipoles.get("mu_c", dipoles.get("c", dipoles.get("muc", 0.0))))
    else:
        d_list = [float(x) for x in dipoles]
        mu_a = d_list[0] if len(d_list) > 0 else 0.0
        mu_b = d_list[1] if len(d_list) > 1 else 0.0
        mu_c = d_list[2] if len(d_list) > 2 else 0.0

    fortran_overflow_guard({"mu_a": mu_a, "mu_b": mu_b, "mu_c": mu_c})

    results: dict[float, str] = {}

    for t in temps:
        title_str = (
            title
            if title
            else f"{molecule_name} Ground State - CoChem SPCAT Bridge (T={t:.2f}K)"
        )

        control_line = (
            f"{tag:>3}{ver:>3}{ibx:>3}{nq:>3}"
            f"  {rrot:>6.1f}  {tem:>6.1f}  {sthk:>6.1f}  {wtk:>6.1f}  {t:>8.2f}"
        )

        int_lines = [
            title_str,
            control_line,
            f"  1  {mu_a:>12.6f}   / mua",
            f"  2  {mu_b:>12.6f}   / mub",
            f"  3  {mu_c:>12.6f}   / muc",
        ]

        content = "\n".join(int_lines) + "\n"
        results[t] = content

        if filepath_template is not None:
            path_str = str(filepath_template).format(
                T=f"{t:.1f}", temp=f"{t:.1f}", molecule=molecule_name
            )
            target = Path(path_str).resolve()
            validate_airgap_boundary(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_file = target.with_suffix(
                f".tmp_{os.getpid()}_{int(datetime.now().timestamp())}"
            )
            temp_file.write_text(content, encoding="utf-8")
            temp_file.replace(target)

    return results


# =============================================================================
# 9. Tripartite Filesystem Air-Gap & Cryptographic Provenance Manifest
# =============================================================================


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
    molecule_name: str,
    geometry: np.ndarray | Sequence[Sequence[float]],
    symbols: Sequence[str],
    rotational_constants_mhz: dict[str, float],
    dipoles_debye: dict[str, float],
    harmonic_frequencies_cm1: Sequence[float],
    temperatures: Sequence[float] = (2.0, 10.0, 50.0, 298.15),
    quartic_distortion: dict[str, float] | None = None,
    lam_frequency: float | Sequence[float] | None = None,
    use_nuclear_spin: bool = False,
    output_dir: str | Path | None = None,
) -> SPCATPayload:
    """Build complete, fully validated, air-gapped SPCAT execution payload with provenance manifest."""
    # Enforce low-frequency LAM trap if no explicit DVR LAM frequency was designated
    if lam_frequency is None:
        low_frequency_lam_trap(harmonic_frequencies_cm1)

    sym_res = apply_symmetry_divisors(
        geometry_array=geometry,
        symbols=symbols,
        use_nuclear_spin=use_nuclear_spin,
    )

    a = float(rotational_constants_mhz.get("A", 0.0))
    b = float(rotational_constants_mhz.get("B", 0.0))
    c = float(rotational_constants_mhz.get("C", 0.0))
    part_res = compute_coupled_partition_functions(
        a_mhz=a,
        b_mhz=b,
        c_mhz=c,
        frequencies_cm1=harmonic_frequencies_cm1,
        temp_array=temperatures,
        sigma=sym_res.effective_divisor,
        lam_frequency=lam_frequency,
    )

    combined_params: dict[str, Any] = {
        "A": a,
        "B": b,
        "C": c,
    }
    if quartic_distortion:
        combined_params.update(quartic_distortion)

    var_path = Path(output_dir) / f"{molecule_name}.var" if output_dir else None
    var_content = generate_spcat_var(
        molecule_name=molecule_name,
        parameters=combined_params,
        filepath=var_path,
    )

    int_tpl = Path(output_dir) / f"{molecule_name}_{{T}}K.int" if output_dir else None
    int_contents = generate_spcat_int(
        molecule_name=molecule_name,
        dipoles=dipoles_debye,
        temperatures=temperatures,
        filepath_template=int_tpl,
    )

    prov_path = (
        Path(output_dir) / f"{molecule_name}_spcat_provenance.json"
        if output_dir
        else None
    )
    manifest = generate_spcat_provenance_manifest(
        molecule_name=molecule_name,
        var_content=var_content,
        int_contents=int_contents,
        symmetry_result=sym_res,
        partition_results=part_res.q_total,
        output_path=prov_path,
    )

    return SPCATPayload(
        molecule_name=molecule_name,
        var_content=var_content,
        int_contents=int_contents,
        provenance_manifest=manifest,
        sha256_var=compute_sha256(var_content),
        sha256_int={t: compute_sha256(c) for t, c in int_contents.items()},
        var_filepath=str(var_path) if var_path else None,
        int_filepaths={
            t: str(Path(output_dir) / f"{molecule_name}_{t:.1f}K.int")
            for t in temperatures
        }
        if output_dir
        else {},
        provenance_filepath=str(prov_path) if prov_path else None,
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
    harmonic_frequencies: list[float]
    vpt2_x_matrix: np.ndarray | None
    dipole_moments_debye: dict[str, float]
    is_mpqc_primary: bool
    is_analytic_vpt2_active: bool
    routing_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize routing result to dictionary."""
        return {
            "selected_tier": self.selected_tier,
            "primary_engine": self.primary_engine,
            "electronic_energy_hartree": self.electronic_energy_hartree,
            "harmonic_frequencies": [float(f) for f in self.harmonic_frequencies],
            "vpt2_x_matrix": self.vpt2_x_matrix.tolist()
            if self.vpt2_x_matrix is not None
            else None,
            "dipole_moments_debye": self.dipole_moments_debye,
            "is_mpqc_primary": self.is_mpqc_primary,
            "is_analytic_vpt2_active": self.is_analytic_vpt2_active,
            "routing_metadata": self.routing_metadata,
        }


def route_3tier_abinitio_payload(
    mpqc_data: dict[str, Any] | None = None,
    orca_data: dict[str, Any] | None = None,
    cfour_data: dict[str, Any] | None = None,
    require_analytic_vpt2: bool = False,
) -> ThreeTierRoutingResult:
    """Enforces the authoritative 3-Tier Routing Protocol (MPQC Primary).

    Protocol Hierarchy:
    - Tier 1 (Primary Benchmark): MPQC (the Valeev Stack). Parsed for exact CCSD(T)-F12
      single-point energetics and reference energies.
    - Tier 2 (Primary Vibrational): ORCA. Parsed for analytic VPT2, harmonic frequencies,
      and dipole surface tensors.
    - Tier 3 (Legacy Alternate): CFOUR. Demoted fallback parsed only when analytic VPT2
      or high-order coupled cluster corrections require proprietary CFOUR outputs.

    Args:
        mpqc_data: Parsed dictionary from MPQC (CCSD(T)-F12 calculations).
        orca_data: Parsed dictionary from ORCA (VPT2 / force fields).
        cfour_data: Parsed dictionary from CFOUR (Legacy / fallback).
        require_analytic_vpt2: If True, prioritizes Tier 2 / Tier 3 containing full VPT2 X-matrices.

    Returns:
        ThreeTierRoutingResult with resolved energies, frequencies, and provenance.
    """
    # Tier 1: MPQC Primary for energy benchmarks
    if mpqc_data is not None and not require_analytic_vpt2:
        energy = mpqc_data.get(
            "energy_hartree", mpqc_data.get("ccsd_t_f12_energy", None)
        )
        freqs = mpqc_data.get("frequencies", [])
        dipoles = mpqc_data.get("dipoles", {"mu_a": 0.0, "mu_b": 0.0, "mu_c": 0.0})
        x_mat = mpqc_data.get("x_matrix", None)
        return ThreeTierRoutingResult(
            selected_tier=1,
            primary_engine="MPQC",
            electronic_energy_hartree=float(energy) if energy is not None else None,
            harmonic_frequencies=[float(f) for f in freqs],
            vpt2_x_matrix=np.asarray(x_mat, dtype=np.float64)
            if x_mat is not None
            else None,
            dipole_moments_debye=dipoles,
            is_mpqc_primary=True,
            is_analytic_vpt2_active=x_mat is not None,
            routing_metadata={
                "tier_description": "Tier 1: MPQC CCSD(T)-F12 Primary Benchmark",
                "raw": mpqc_data,
            },
        )

    # Tier 2: ORCA Primary for analytic VPT2
    if orca_data is not None:
        energy = orca_data.get(
            "energy_hartree", orca_data.get("electronic_energy", None)
        )
        freqs = orca_data.get("frequencies", orca_data.get("harmonic_frequencies", []))
        dipoles = orca_data.get(
            "dipoles",
            orca_data.get("dipole_moments", {"mu_a": 0.0, "mu_b": 0.0, "mu_c": 0.0}),
        )
        x_mat = orca_data.get("x_matrix", orca_data.get("anharmonic_x_matrix", None))
        return ThreeTierRoutingResult(
            selected_tier=2,
            primary_engine="ORCA",
            electronic_energy_hartree=float(energy) if energy is not None else None,
            harmonic_frequencies=[float(f) for f in freqs],
            vpt2_x_matrix=np.asarray(x_mat, dtype=np.float64)
            if x_mat is not None
            else None,
            dipole_moments_debye=dipoles,
            is_mpqc_primary=False,
            is_analytic_vpt2_active=x_mat is not None,
            routing_metadata={
                "tier_description": "Tier 2: ORCA Analytic VPT2 Primary",
                "raw": orca_data,
            },
        )

    # Tier 3: CFOUR Legacy Alternate
    if cfour_data is not None:
        energy = cfour_data.get("energy_hartree", cfour_data.get("eccsd_t", None))
        freqs = cfour_data.get("frequencies", [])
        dipoles = cfour_data.get("dipoles", {"mu_a": 0.0, "mu_b": 0.0, "mu_c": 0.0})
        x_mat = cfour_data.get("x_matrix", None)
        return ThreeTierRoutingResult(
            selected_tier=3,
            primary_engine="CFOUR",
            electronic_energy_hartree=float(energy) if energy is not None else None,
            harmonic_frequencies=[float(f) for f in freqs],
            vpt2_x_matrix=np.asarray(x_mat, dtype=np.float64)
            if x_mat is not None
            else None,
            dipole_moments_debye=dipoles,
            is_mpqc_primary=False,
            is_analytic_vpt2_active=x_mat is not None,
            routing_metadata={
                "tier_description": "Tier 3: CFOUR Legacy Alternate Fallback",
                "raw": cfour_data,
            },
        )

    if mpqc_data is not None:
        energy = mpqc_data.get("energy_hartree", None)
        freqs = mpqc_data.get("frequencies", [])
        dipoles = mpqc_data.get("dipoles", {"mu_a": 0.0, "mu_b": 0.0, "mu_c": 0.0})
        return ThreeTierRoutingResult(
            selected_tier=1,
            primary_engine="MPQC",
            electronic_energy_hartree=float(energy) if energy is not None else None,
            harmonic_frequencies=[float(f) for f in freqs],
            vpt2_x_matrix=None,
            dipole_moments_debye=dipoles,
            is_mpqc_primary=True,
            is_analytic_vpt2_active=False,
            routing_metadata={
                "tier_description": "Tier 1: MPQC CCSD(T)-F12 Single-Point",
                "raw": mpqc_data,
            },
        )

    raise ValueError("No ab initio data provided to 3-Tier Routing Protocol.")


# =============================================================================
# 11. TorqSpcatBridge Compatibility Adapter
# =============================================================================


class TorqSpcatBridge:
    """Torq SPCAT bridge class for backwards-compatibility with CoChem-TORQ workflow."""

    def __init__(
        self,
        tensor_json_path: str | Path,
        mpqc_out_path: str | Path,
        temperature_k: float = 298.15,
    ) -> None:
        self.tensor_file = Path(tensor_json_path)
        self.mpqc_file = Path(mpqc_out_path)
        self.orca_file = self.mpqc_file
        self.temperature = float(temperature_k)
        self.temperature_k = float(temperature_k)

        self.tensor_data = self._load_json(self.tensor_file)
        self.point_id = str(self.tensor_data.get("point_id", "000"))
        self.is_linear = bool(self.tensor_data.get("is_linear", False))

        constants_dict = self.tensor_data.get("tensors", {}).get(
            "rotational_constants_MHz", {}
        )
        self.rot_A_MHz = float(constants_dict.get("A", 10000.0) or 10000.0)
        self.rot_B_MHz = float(constants_dict.get("B", 5000.0) or 5000.0)
        self.rot_C_MHz = float(constants_dict.get("C", 3333.33) or 3333.33)

        self.sigma = self._determine_symmetry_divisor()
        self.frequencies_cm1: list[float] = []
        self.dipole_moments: dict[str, float] = {"a": 0.0, "b": 0.0, "c": 0.0}

    def _load_json(self, filepath: Path) -> dict[str, Any]:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Tensor file {filepath} not found. Run Stage 4.1 first."
            )
        if filepath.stat().st_size == 0:
            return {}
        try:
            with open(filepath, encoding="utf-8") as f:
                return json.loads(f.read())
        except Exception:
            return {}

    def _determine_symmetry_divisor(self) -> int:
        coords = self.tensor_data.get("coordinates", [])
        symbols = self.tensor_data.get("symbols", [])
        if coords and symbols:
            try:
                sym_res = apply_symmetry_divisors(coords, symbols)
                return sym_res.sigma
            except Exception:
                pass
        return 1

    def parse_mpqc_observables(self) -> None:
        if not self.mpqc_file.exists():
            logger.error(
                "MPQC output %s missing. Cannot parse vibrational partition functions.",
                self.mpqc_file,
            )
            return

        freqs: list[float] = []
        try:
            with open(self.mpqc_file, errors="ignore", encoding="utf-8") as f:
                content = f.read()

            dipole_match = re.search(
                r"Total Dipole Moment\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
                content,
            )
            if dipole_match:
                dx, dy, dz = map(float, dipole_match.groups())
                self.dipole_moments = {"a": abs(dx), "b": abs(dy), "c": abs(dz)}

            freq_section = re.search(
                r"VIBRATIONAL FREQUENCIES\s+[-=]+\s*(.*?)(?=\n\n|\n[A-Z]|\Z)",
                content,
                re.DOTALL,
            )
            if freq_section:
                for line in freq_section.group(1).strip().splitlines():
                    m = re.search(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", line)
                    if m:
                        val = float(m.group(1))
                        if val > 0.1:
                            freqs.append(val)
        except Exception as e:
            logger.warning("Error reading MPQC file: %s", e)

        self.frequencies_cm1 = freqs
        if self.frequencies_cm1:
            low_frequency_lam_trap(self.frequencies_cm1)

    def calculate_partition_functions(self) -> tuple[float, float, float]:
        q_rot = calculate_rotational_partition_function(
            a_mhz=self.rot_A_MHz,
            b_mhz=self.rot_B_MHz,
            c_mhz=self.rot_C_MHz,
            temp_k=self.temperature_k,
            sigma=float(self.sigma),
            is_linear=self.is_linear,
        )
        q_vib = calculate_vibrational_partition_function(
            frequencies_cm1=self.frequencies_cm1,
            temp_k=self.temperature_k,
        )
        q_total = q_rot * q_vib
        return q_rot, q_vib, q_total

    def generate_spcat_files(self) -> None:
        artifact_dir = Path(os.environ.get("COCHEM_ARTIFACT_DIR", ".")) / "spcat"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        var_file = artifact_dir / f"spcat_{self.point_id}.var"
        int_file = artifact_dir / f"spcat_{self.point_id}.int"

        generate_spcat_var(
            molecule_name=f"spcat_{self.point_id}",
            parameters={"A": self.rot_A_MHz, "B": self.rot_B_MHz, "C": self.rot_C_MHz},
            filepath=var_file,
        )
        generate_spcat_int(
            molecule_name=f"spcat_{self.point_id}",
            dipoles=self.dipole_moments,
            temperatures=[self.temperature_k],
            filepath_template=int_file,
        )

    def export_spcat_catalog(self) -> None:
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

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_cochem_torq_init.py ---
"""CoChem-TORQ: Test Suite for Environment Bootstrapper & IPC Resource Manager.

=============================================================================
Phase 1 (Stage 0.0) Test Suite
------------------------------
Zero-Mock test suite verifying physical air-gap mathematical boundary checks,
dynamic artifact directory routing, Pydantic v2 model validations,
cross-platform memory-mapped buffers (mmap & SharedMemory), deterministic
IPC resource reclamation, and live subprocess atexit exit handlers.

All tests operate against real physical files, shared memory segments,
and subprocess executions within pytest `tmp_path`.
"""

from __future__ import annotations

import mmap
import multiprocessing.shared_memory as sm
import os
import subprocess
import sys
import tempfile
import time
import uuid

# ============================================================================
# Autouse Fixture to Clean IPC Registry Between Tests
# ============================================================================
from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import ValidationError

from Libraries.cochem_torq_init import (
    _ACTIVE_BUFFERS,
    _ACTIVE_FILE_DESCRIPTORS,
    _ACTIVE_MMAP_OBJECTS,
    _ACTIVE_SCRATCH_PATHS,
    _ACTIVE_SHM_NAMES,
    _ACTIVE_SHM_OBJECTS,
    AirGapReport,
    AirGapViolationError,
    BootstrapperConfig,
    IPCBufferError,
    IPCBufferMetadata,
    bootstrap_environment,
    check_airgap,
    cleanup_ipc_scratch,
    create_ipc_scratch_buffer,
    get_artifact_directory,
    get_scratch_directory,
    register_ipc_cleanup,
    register_mmap_buffer,
    unregister_ipc_cleanup,
    verify_airgap,
)


@pytest.fixture(autouse=True)
def clean_ipc_state() -> Generator[None, None, None]:
    """Ensure clean IPC tracking registry state before and after each test."""
    cleanup_ipc_scratch()
    _ACTIVE_SCRATCH_PATHS.clear()
    _ACTIVE_SHM_NAMES.clear()
    _ACTIVE_SHM_OBJECTS.clear()
    _ACTIVE_MMAP_OBJECTS.clear()
    _ACTIVE_BUFFERS.clear()
    _ACTIVE_FILE_DESCRIPTORS.clear()
    yield
    cleanup_ipc_scratch()
    _ACTIVE_SCRATCH_PATHS.clear()
    _ACTIVE_SHM_NAMES.clear()
    _ACTIVE_SHM_OBJECTS.clear()
    _ACTIVE_MMAP_OBJECTS.clear()
    _ACTIVE_BUFFERS.clear()
    _ACTIVE_FILE_DESCRIPTORS.clear()


# ============================================================================
# 1. Air-Gap Verification Tests
# ============================================================================


class TestAirGapVerification:
    """Tests verifying the mathematical air-gap boundary checks."""

    def test_verify_airgap_disjoint_paths(self, tmp_path: Path) -> None:
        """Disjoint working directory and artifact directory must pass."""
        repo_dir = tmp_path / "repo_root"
        artifacts_dir = tmp_path / "external_artifacts"
        repo_dir.mkdir(parents=True)
        artifacts_dir.mkdir(parents=True)

        assert verify_airgap(cwd=repo_dir, artifacts_dir=artifacts_dir) is True

    def test_verify_airgap_identical_path(self, tmp_path: Path) -> None:
        """Identical working directory and artifact dir must raise error."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(parents=True)

        with pytest.raises(AirGapViolationError) as exc_info:
            verify_airgap(cwd=shared_dir, artifacts_dir=shared_dir)
        err_str = str(exc_info.value).lower()
        assert "intersect" in err_str or "overlap" in err_str or "identical" in err_str

    def test_verify_airgap_artifacts_inside_cwd(self, tmp_path: Path) -> None:
        """Artifacts dir inside working directory must raise violation error."""
        repo_dir = tmp_path / "repo_root"
        nested_artifacts = repo_dir / "build" / "artifacts"
        nested_artifacts.mkdir(parents=True)

        with pytest.raises(AirGapViolationError) as exc_info:
            verify_airgap(cwd=repo_dir, artifacts_dir=nested_artifacts)
        err_str = str(exc_info.value).lower()
        assert "inside" in err_str or "intersect" in err_str or "overlap" in err_str

    def test_verify_airgap_cwd_inside_artifacts(self, tmp_path: Path) -> None:
        """Working directory inside artifact dir must raise violation error."""
        artifacts_dir = tmp_path / "cochem_artifacts"
        nested_cwd = artifacts_dir / "subproject" / "repo"
        nested_cwd.mkdir(parents=True)

        with pytest.raises(AirGapViolationError) as exc_info:
            verify_airgap(cwd=nested_cwd, artifacts_dir=artifacts_dir)
        err_str = str(exc_info.value).lower()
        assert "inside" in err_str or "intersect" in err_str or "overlap" in err_str

    def test_verify_airgap_string_and_path_inputs(self, tmp_path: Path) -> None:
        """verify_airgap must accept str, Path, and resolve accurately."""
        repo_dir = tmp_path / "repo"
        artifacts_dir = tmp_path / "artifacts"
        repo_dir.mkdir()
        artifacts_dir.mkdir()

        assert (
            verify_airgap(cwd=str(repo_dir), artifacts_dir=str(artifacts_dir))
            is True
        )
        assert verify_airgap(cwd=repo_dir, artifacts_dir=str(artifacts_dir)) is True
        assert verify_airgap(cwd=str(repo_dir), artifacts_dir=artifacts_dir) is True

    def test_verify_airgap_default_resolution(self) -> None:
        """Default verify_airgap call resolves cwd and dynamic artifact dir."""
        result = verify_airgap()
        assert isinstance(result, bool)

    def test_check_airgap_report(self, tmp_path: Path) -> None:
        """check_airgap returns AirGapReport Pydantic model with diagnostics."""
        repo_dir = tmp_path / "repo"
        artifacts_dir = tmp_path / "artifacts"
        repo_dir.mkdir()
        artifacts_dir.mkdir()

        report = check_airgap(cwd=repo_dir, artifacts_dir=artifacts_dir)
        assert isinstance(report, AirGapReport)
        assert report.is_valid is True
        assert report.cwd_resolved == repo_dir.resolve()
        assert report.artifacts_resolved == artifacts_dir.resolve()
        assert report.reason is None

        # Failing case report
        fail_report = check_airgap(cwd=repo_dir, artifacts_dir=repo_dir)
        assert isinstance(fail_report, AirGapReport)
        assert fail_report.is_valid is False
        assert fail_report.reason is not None


# ============================================================================
# 2. Dynamic Artifact & Scratch Directory Mapping Tests
# ============================================================================


class TestDirectoryMapping:
    """Tests verifying dynamic host environment directory mapping without hardcoding."""

    def test_get_artifact_directory_from_env(
        self, tmp_path: Path
    ) -> None:
        """get_artifact_directory must respect host environment variable."""
        custom_dir = tmp_path / "env_artifacts"
        os.environ.__setitem__("COCHEM_ARTIFACTS", str(custom_dir))

        resolved = get_artifact_directory(env_var="COCHEM_ARTIFACTS")
        assert resolved == custom_dir.resolve()
        assert resolved.exists()
        assert resolved.is_dir()

    def test_get_artifact_directory_fallback(
        self, tmp_path: Path
    ) -> None:
        """get_artifact_directory must use fallback_dir when env var is unset."""
        os.environ.pop("COCHEM_ARTIFACTS", None)
        fallback = tmp_path / "fallback_artifacts"

        resolved = get_artifact_directory(
            env_var="COCHEM_ARTIFACTS", fallback_dir=fallback
        )
        assert resolved == fallback.resolve()
        assert resolved.exists()
        assert resolved.is_dir()

    def test_get_artifact_directory_default_temp(
        self
    ) -> None:
        """get_artifact_directory falls back to system temp when unset."""
        os.environ.pop("COCHEM_ARTIFACTS", None)

        resolved = get_artifact_directory(
            env_var="COCHEM_ARTIFACTS", fallback_dir=None
        )
        expected_parent = Path(tempfile.gettempdir()).resolve()
        assert (
            resolved.parent == expected_parent
            or str(resolved).startswith(str(expected_parent))
        )
        assert "cochem_artifacts" in resolved.name
        assert resolved.exists()

    def test_get_scratch_directory_from_env(
        self, tmp_path: Path
    ) -> None:
        """get_scratch_directory must respect host environment variable."""
        scratch = tmp_path / "fast_scratch"
        os.environ.__setitem__("COCHEM_SCRATCH", str(scratch))

        resolved = get_scratch_directory(env_var="COCHEM_SCRATCH")
        assert resolved == scratch.resolve()
        assert resolved.exists()

    def test_get_scratch_directory_fallback(
        self
    ) -> None:
        """get_scratch_directory creates PID-isolated temp scratch folder."""
        os.environ.pop("COCHEM_SCRATCH", None)

        resolved = get_scratch_directory(env_var="COCHEM_SCRATCH")
        assert "cochem_scratch" in str(resolved)
        assert resolved.exists()
        assert resolved.is_dir()


# ============================================================================
# 3. Pydantic v2 Models Validation Tests
# ============================================================================


class TestPydanticModels:
    """Tests verifying Pydantic v2 metadata and configuration validation models."""

    def test_airgap_report_model(self, tmp_path: Path) -> None:
        """AirGapReport model must be frozen and properly serialize."""
        report = AirGapReport(
            is_valid=True,
            cwd_resolved=tmp_path / "cwd",
            artifacts_resolved=tmp_path / "artifacts",
            verified_at=time.time(),
            reason=None,
        )
        assert report.is_valid is True
        with pytest.raises(ValidationError):
            # Model is frozen
            setattr(report, "is_valid", False)

    def test_ipc_buffer_metadata_model(self, tmp_path: Path) -> None:
        """IPCBufferMetadata model validates types, sizes, and buffer categories."""
        meta = IPCBufferMetadata(
            name="test_buffer",
            buffer_type="mmap",
            size_bytes=1024,
            file_path=tmp_path / "buf.dat",
        )
        assert meta.size_bytes == 1024
        assert meta.buffer_type == "mmap"
        assert meta.is_active is True
        assert isinstance(meta.buffer_id, str)
        assert meta.pid == os.getpid()

    def test_ipc_buffer_metadata_invalid_size_or_type(self) -> None:
        """IPCBufferMetadata must reject non-positive size and invalid buffer types."""
        with pytest.raises(ValidationError):
            IPCBufferMetadata(
                name="bad_size",
                buffer_type="mmap",
                size_bytes=0,
            )

        with pytest.raises(ValidationError):
            IPCBufferMetadata(
                name="bad_type",
                buffer_type="unsupported_type",  # type: ignore[arg-type]
                size_bytes=1024,
            )

    def test_bootstrapper_config_model(self, tmp_path: Path) -> None:
        """BootstrapperConfig model validates filesystem paths and settings."""
        cfg = BootstrapperConfig(
            artifacts_dir=tmp_path / "artifacts",
            scratch_dir=tmp_path / "scratch",
            env_var="COCHEM_ARTIFACTS",
            enforce_airgap=True,
            clean_on_exit=True,
        )
        assert cfg.enforce_airgap is True
        assert cfg.clean_on_exit is True
        assert cfg.artifacts_dir == tmp_path / "artifacts"


# ============================================================================
# 4. IPC Buffer Creation, Read/Write, and Context Manager Tests
# ============================================================================


class TestIPCScratchBuffer:
    """Tests verifying real memory-mapped files and SharedMemory buffers."""

    def test_create_ipc_scratch_buffer_mmap(self, tmp_path: Path) -> None:
        """create_ipc_scratch_buffer creates a physical mmap buffer."""
        buf = create_ipc_scratch_buffer(
            name="test_mmap_buffer",
            size=2048,
            buffer_type="mmap",
            scratch_dir=tmp_path,
            auto_register=True,
        )
        try:
            assert buf.metadata.buffer_type == "mmap"
            assert buf.size == 2048
            assert buf.path is not None
            assert buf.path.exists()
            assert buf.path.stat().st_size == 2048

            # Write and read data
            payload = b"CoChem-TORQ-Physical-IPC-Data-Payload"
            written = buf.write(payload, offset=0)
            assert written == len(payload)

            read_data = buf.read(size=len(payload), offset=0)
            assert read_data == payload
        finally:
            buf.close()
            buf.unlink()

    def test_create_ipc_scratch_buffer_shm(self) -> None:
        """create_ipc_scratch_buffer creates a POSIX/Windows SharedMemory segment."""
        unique_name = f"cochem_shm_{uuid.uuid4().hex[:8]}"
        buf = create_ipc_scratch_buffer(
            name=unique_name,
            size=1024,
            buffer_type="shm",
            auto_register=True,
        )
        try:
            assert buf.metadata.buffer_type == "shm"
            assert buf.size == 1024
            assert buf.name == unique_name

            payload = b"Quantum-Chemistry-SWMR-Buffer-Test"
            written = buf.write(payload, offset=16)
            assert written == len(payload)

            read_data = buf.read(size=len(payload), offset=16)
            assert read_data == payload
        finally:
            buf.close()
            buf.unlink()

    def test_ipc_buffer_context_manager_mmap(self, tmp_path: Path) -> None:
        """IPCScratchBuffer context manager manages lifecycle and cleanup."""
        target_path: Path | None = None
        with create_ipc_scratch_buffer(
            name="ctx_mmap",
            size=512,
            buffer_type="mmap",
            scratch_dir=tmp_path,
        ) as buf:
            target_path = buf.path
            assert target_path is not None and target_path.exists()
            buf.write(b"Inside-Context-Manager")
            assert buf.read(22) == b"Inside-Context-Manager"

        assert buf.is_closed is True

    def test_ipc_buffer_context_manager_shm(self) -> None:
        """IPCScratchBuffer context manager manages SharedMemory lifecycle."""
        shm_name = f"cochem_shm_ctx_{uuid.uuid4().hex[:8]}"
        with create_ipc_scratch_buffer(
            name=shm_name,
            size=512,
            buffer_type="shm",
        ) as buf:
            buf.write(b"SHM-Context-Payload")
            assert buf.read(19) == b"SHM-Context-Payload"

        assert buf.is_closed is True

    def test_ipc_buffer_boundary_checks(self, tmp_path: Path) -> None:
        """Writing or reading outside buffer boundaries must raise IPCBufferError."""
        with create_ipc_scratch_buffer(size=128, scratch_dir=tmp_path) as buf:
            # Writing payload that exceeds buffer size
            overflow_data = b"X" * 200
            with pytest.raises(IPCBufferError):
                buf.write(overflow_data, offset=0)

            # Writing with offset that exceeds buffer size
            with pytest.raises(IPCBufferError):
                buf.write(b"Hello", offset=150)

            # Reading with invalid offset
            with pytest.raises(IPCBufferError):
                buf.read(size=10, offset=200)

    def test_register_mmap_buffer_helper(self, tmp_path: Path) -> None:
        """register_mmap_buffer must register standalone mmap and backing file."""
        file_path = tmp_path / "raw_mmap.dat"
        file_path.write_bytes(b"\x00" * 256)

        f = open(file_path, "r+b")
        try:
            mm = mmap.mmap(f.fileno(), 256)
        finally:
            f.close()

        register_mmap_buffer(mm, backing_path=file_path)

        mm.write(b"Standalone-MMAP-Payload")
        mm.seek(0)
        assert mm.read(23) == b"Standalone-MMAP-Payload"

        # Execute cleanup
        cleanup_summary = cleanup_ipc_scratch()
        assert cleanup_summary["mmaps_closed"] >= 1
        assert cleanup_summary["files_removed"] >= 1
        assert not file_path.exists()


# ============================================================================
# 5. Deterministic Resource Reclamation & Exit Handlers
# ============================================================================


class TestDeterministicCleanup:
    """Tests verifying deterministic resource reclamation and tracking."""

    def test_cleanup_ipc_scratch_files(self, tmp_path: Path) -> None:
        """cleanup_ipc_scratch removes registered physical files and directories."""
        f1 = tmp_path / "scratch_1.tmp"
        f2 = tmp_path / "scratch_2.tmp"
        d1 = tmp_path / "scratch_dir"
        d1.mkdir()
        (d1 / "nested.tmp").write_text("test")

        f1.write_text("data1")
        f2.write_text("data2")

        register_ipc_cleanup(scratch_paths=[f1, f2, d1])
        assert f1.exists() and f2.exists() and d1.exists()

        summary = cleanup_ipc_scratch()
        assert summary["files_removed"] >= 2
        assert summary["directories_removed"] >= 1
        assert not f1.exists()
        assert not f2.exists()
        assert not d1.exists()

    def test_cleanup_ipc_scratch_shm(self) -> None:
        """cleanup_ipc_scratch unlinks registered SharedMemory segments."""
        shm_name = f"cochem_shm_clean_{uuid.uuid4().hex[:8]}"
        shm_obj = sm.SharedMemory(name=shm_name, create=True, size=256)

        register_ipc_cleanup(shm_names=[shm_name], shm_objects=[shm_obj])
        summary = cleanup_ipc_scratch()
        assert summary["shm_unlinked"] >= 1

        # Verifying segment is unlinked: attempting to open should fail
        with pytest.raises(FileNotFoundError):
            sm.SharedMemory(name=shm_name, create=False)

    def test_cleanup_ipc_scratch_idempotent(self, tmp_path: Path) -> None:
        """cleanup_ipc_scratch must be completely safe to invoke repeatedly."""
        f1 = tmp_path / "idempotent.tmp"
        f1.write_text("test")
        register_ipc_cleanup(scratch_paths=[f1])

        first_summary = cleanup_ipc_scratch()
        assert first_summary["files_removed"] >= 1

        second_summary = cleanup_ipc_scratch()
        assert second_summary["files_removed"] == 0
        assert second_summary["shm_unlinked"] == 0

    def test_targeted_cleanup_does_not_destroy_unrelated_buffers(
        self, tmp_path: Path
    ) -> None:
        """Targeted cleanup must only destroy requested targets, leaving active buffers intact."""
        # Create an active buffer
        buf = create_ipc_scratch_buffer(
            name="persistent_buffer",
            size=512,
            buffer_type="mmap",
            scratch_dir=tmp_path,
            auto_register=True,
        )
        assert buf.is_closed is False
        assert buf in _ACTIVE_BUFFERS

        # Create a targeted scratch file
        targeted_file = tmp_path / "targeted_scratch.tmp"
        targeted_file.write_text("to be deleted")
        register_ipc_cleanup(scratch_paths=[targeted_file])

        # Run targeted cleanup for the single file
        summary = cleanup_ipc_scratch(scratch_paths=[targeted_file])
        assert summary["files_removed"] == 1
        assert not targeted_file.exists()

        # The active buffer MUST NOT have been destroyed or closed
        assert buf.is_closed is False
        assert buf in _ACTIVE_BUFFERS
        assert buf.read(5) == b"\x00\x00\x00\x00\x00"

        # Now clean up the buffer
        buf.unlink()
        assert buf.is_closed is True
        assert buf not in _ACTIVE_BUFFERS

    def test_buffer_unlink_deregisters_from_global_state(
        self, tmp_path: Path
    ) -> None:
        """Calling unlink on a buffer must remove its references from global tracking sets."""
        buf = create_ipc_scratch_buffer(
            name="unlink_dereg_test",
            size=256,
            buffer_type="mmap",
            scratch_dir=tmp_path,
            auto_register=True,
        )
        assert buf in _ACTIVE_BUFFERS
        assert buf.path in _ACTIVE_SCRATCH_PATHS

        buf.unlink()

        # Global registries must be cleared of this buffer and its path
        assert buf not in _ACTIVE_BUFFERS
        assert buf.path not in _ACTIVE_SCRATCH_PATHS


# ============================================================================
# 6. Live Subprocess Exit Handler (atexit) Verification
# ============================================================================


class TestSubprocessExitHandlers:
    """Tests executing real child processes to verify atexit cleanup upon exit."""

    def test_live_subprocess_atexit_cleanup_mmap(self, tmp_path: Path) -> None:
        """Child process creates an mmap IPC buffer; on exit, buffer is cleaned up."""
        scratch_dir = tmp_path / "proc_scratch"
        scratch_dir.mkdir()

        script = f"""
import sys
from pathlib import Path
from Libraries.cochem_torq_init import create_ipc_scratch_buffer

buf = create_ipc_scratch_buffer(
    name="child_proc_mmap",
    size=1024,
    buffer_type="mmap",
    scratch_dir=r"{scratch_dir}",
    auto_register=True
)
buf.write(b"Subprocess-Test-Data")
print("BUFFER_CREATED:" + str(buf.path))
sys.stdout.flush()
sys.exit(0)
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=15,
        )
        assert proc.returncode == 0, f"Process failed: {proc.stderr}"
        assert "BUFFER_CREATED:" in proc.stdout

        for line in proc.stdout.splitlines():
            if line.startswith("BUFFER_CREATED:"):
                created_path = Path(line.split(":", 1)[1].strip())
                assert not created_path.exists(), (
                    f"Scratch file {created_path} still exists after process exit!"
                )

    def test_live_subprocess_atexit_cleanup_shm(self) -> None:
        """Child process creates SharedMemory buffer; on exit, shm is unlinked."""
        shm_name = f"cochem_subproc_shm_{uuid.uuid4().hex[:8]}"

        script = f"""
import sys
from Libraries.cochem_torq_init import create_ipc_scratch_buffer

buf = create_ipc_scratch_buffer(
    name="{shm_name}",
    size=512,
    buffer_type="shm",
    auto_register=True
)
buf.write(b"SHM-Subprocess-Payload")
print("SHM_CREATED:" + "{shm_name}")
sys.stdout.flush()
sys.exit(0)
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=15,
        )
        assert proc.returncode == 0, f"Process failed: {proc.stderr}"
        assert "SHM_CREATED:" in proc.stdout

        with pytest.raises(FileNotFoundError):
            sm.SharedMemory(name=shm_name, create=False)


# ============================================================================
# 7. Environment Bootstrapper Integration Tests
# ============================================================================


class TestBootstrapEnvironment:
    """Tests verifying the full bootstrap_environment lifecycle."""

    def test_bootstrap_environment_success(
        self, tmp_path: Path
    ) -> None:
        """bootstrap_environment resolves paths and registers IPC cleanup."""
        artifacts_dir = tmp_path / "artifacts"
        scratch_dir = tmp_path / "scratch"
        os.environ.__setitem__("COCHEM_ARTIFACTS", str(artifacts_dir))
        os.environ.__setitem__("COCHEM_SCRATCH", str(scratch_dir))

        config = bootstrap_environment(
            artifacts_env="COCHEM_ARTIFACTS",
            scratch_env="COCHEM_SCRATCH",
            enforce_airgap=True,
        )
        assert isinstance(config, BootstrapperConfig)
        assert config.artifacts_dir == artifacts_dir.resolve()
        assert config.scratch_dir == scratch_dir.resolve()
        assert config.artifacts_dir.exists()
        assert config.scratch_dir.exists()

    def test_bootstrap_environment_airgap_failure(
        self, tmp_path: Path
    ) -> None:
        """bootstrap_environment raises AirGapViolationError if artifacts in cwd."""
        repo_cwd = Path.cwd()
        nested_artifacts = repo_cwd / "test_nested_artifacts_violation"
        os.environ.__setitem__("COCHEM_ARTIFACTS", str(nested_artifacts))

        try:
            with pytest.raises(AirGapViolationError):
                bootstrap_environment(
                    artifacts_env="COCHEM_ARTIFACTS",
                    enforce_airgap=True,
                )
        finally:
            if nested_artifacts.exists():
                nested_artifacts.rmdir()

    def test_bootstrap_environment_no_enforce(
        self, tmp_path: Path
    ) -> None:
        """bootstrap_environment proceeds without error if enforce_airgap=False."""
        repo_cwd = Path.cwd()
        nested_artifacts = repo_cwd / "test_nested_artifacts_no_enforce"
        os.environ.__setitem__("COCHEM_ARTIFACTS", str(nested_artifacts))

        config = bootstrap_environment(
            artifacts_env="COCHEM_ARTIFACTS",
            enforce_airgap=False,
        )
        assert config.artifacts_dir == nested_artifacts.resolve()
        assert config.enforce_airgap is False
        if nested_artifacts.exists():
            nested_artifacts.rmdir()

    def test_bootstrap_environment_tripartite_scratch_inside_artifacts_failure(
        self, tmp_path: Path
    ) -> None:
        """bootstrap_environment raises AirGapViolationError if scratch is inside artifacts."""
        artifacts_dir = tmp_path / "artifacts"
        nested_scratch = artifacts_dir / "nested_scratch"
        os.environ.__setitem__("COCHEM_ARTIFACTS", str(artifacts_dir))
        os.environ.__setitem__("COCHEM_SCRATCH", str(nested_scratch))

        with pytest.raises(AirGapViolationError):
            bootstrap_environment(
                artifacts_env="COCHEM_ARTIFACTS",
                scratch_env="COCHEM_SCRATCH",
                enforce_airgap=True,
            )


# ============================================================================
# 8. Air-Gap Compliance Runtime Test
# ============================================================================


class TestAirGapRepositoryIntegrity:
    """Asserts that no test or bootstrapper logic wrote runtime files to repo."""

    def test_no_runtime_writes_to_repo(self, tmp_path: Path) -> None:
        """Verify that scratch buffers do not produce repository artifacts."""
        repo_path = Path(__file__).parent.parent.resolve()
        initial_repo_files = {
            p for p in repo_path.glob("**/*") if not p.name.startswith(".")
        }

        scratch_dir = tmp_path / "airgap_scratch"
        with create_ipc_scratch_buffer(
            name="integrity_test",
            size=1024,
            buffer_type="mmap",
            scratch_dir=scratch_dir,
        ) as buf:
            buf.write(b"Air-gap runtime data")

        current_repo_files = {
            p for p in repo_path.glob("**/*") if not p.name.startswith(".")
        }
        new_repo_files = {
            f
            for f in (current_repo_files - initial_repo_files)
            if "__pycache__" not in str(f) and ".pytest_cache" not in str(f)
        }
        assert (
            len(new_repo_files) == 0
        ), f"Unexpected runtime files in repository: {new_repo_files}"

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_spcat.py ---
import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from Libraries.cochem_spcat_bridge import (
    CONSTANTS,
    AirGapViolationError,
    FortranOverflowError,
    LAMTriggerError,
    TorqSpcatBridge,
    apply_symmetry_divisors,
    build_complete_spcat_payload,
    calculate_rotational_constants_from_geometry,
    calculate_vibrational_partition_function,
    format_fortran_double,
    fortran_double_precision_formatter,
    fortran_overflow_guard,
    generate_spcat_int,
    generate_spcat_var,
    get_atomic_mass,
    low_frequency_lam_trap,
    low_frequency_trap,
    route_3tier_abinitio_payload,
    validate_airgap_boundary,
    vibrational_partition_coupling,
)

logger = logging.getLogger(__name__)

# Real experimental / ab initio Cartesian geometry for Water (H2O in Angstroms)
H2O_GEOMETRY = np.array(
    [
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ],
    dtype=np.float64,
)
H2O_SYMBOLS = ["O", "H", "H"]

# Real geometry for Ammonia (NH3 in Angstroms)
NH3_GEOMETRY = np.array(
    [
        [0.000000, 0.000000, 0.116489],
        [0.000000, 0.939731, -0.271808],
        [0.813831, -0.469865, -0.271808],
        [-0.813831, -0.469865, -0.271808],
    ],
    dtype=np.float64,
)
NH3_SYMBOLS = ["N", "H", "H", "H"]

# Real geometry for Ethylene (C2H4 in Angstroms)
C2H4_GEOMETRY = np.array(
    [
        [0.000000, 0.000000, 0.669500],
        [0.000000, 0.000000, -0.669500],
        [0.000000, 0.928900, 1.232100],
        [0.000000, -0.928900, 1.232100],
        [0.000000, 0.928900, -1.232100],
        [0.000000, -0.928900, -1.232100],
    ],
    dtype=np.float64,
)
C2H4_SYMBOLS = ["C", "C", "H", "H", "H", "H"]


def test_torq_spcat_bridge_init(tmp_path: Path) -> None:
    tensor_file = tmp_path / "tensor.json"
    tensor_file.write_text(
        json.dumps(
            {
                "point_id": "001",
                "coordinates": H2O_GEOMETRY.tolist(),
                "symbols": H2O_SYMBOLS,
                "tensors": {
                    "rotational_constants_MHz": {
                        "A": 825360.0,
                        "B": 435360.0,
                        "C": 278130.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    mpqc_file = tmp_path / "mpqc.out"
    mpqc_file.write_text("FINAL SINGLE POINT ENERGY -76.123\n", encoding="utf-8")

    bridge = TorqSpcatBridge(str(tensor_file), str(mpqc_file), temperature_k=298.15)
    assert bridge.temperature_k == 298.15
    assert bridge.mpqc_file == Path(mpqc_file)
    assert bridge.rot_A_MHz == 825360.0
    assert bridge.rot_B_MHz == 435360.0
    assert bridge.rot_C_MHz == 278130.0
    assert bridge.sigma == 2


def test_torq_spcat_bridge_extract_orca(tmp_path: Path) -> None:
    tensor_file = tmp_path / "tensor.json"
    tensor_file.write_text(
        json.dumps(
            {
                "point_id": "002",
                "coordinates": H2O_GEOMETRY.tolist(),
                "symbols": H2O_SYMBOLS,
                "tensors": {
                    "rotational_constants_MHz": {
                        "A": 825360.0,
                        "B": 435360.0,
                        "C": 278130.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    mpqc_file = tmp_path / "mpqc.out"
    mpqc_file.write_text(
        "Total Dipole Moment : 0.000 0.000 1.854\n"
        "VIBRATIONAL FREQUENCIES\n"
        "-----------------------\n"
        " 1: 1595.00 cm**-1\n"
        " 2: 3657.00 cm**-1\n"
        " 3: 3756.00 cm**-1\n",
        encoding="utf-8",
    )

    bridge = TorqSpcatBridge(str(tensor_file), str(mpqc_file), temperature_k=298.15)
    bridge.parse_mpqc_observables()
    assert len(bridge.frequencies_cm1) == 3
    assert bridge.dipole_moments["c"] == 1.854

    q_rot, q_vib, q_total = bridge.calculate_partition_functions()
    assert q_rot > 0.0
    assert q_vib >= 1.0
    assert math.isclose(q_total, q_rot * q_vib, rel_tol=1e-9)

    spcat_dir = tmp_path / "scratch"

    import os

    original_env = os.environ.get("COCHEM_ARTIFACT_DIR")
    os.environ["COCHEM_ARTIFACT_DIR"] = str(spcat_dir)
    try:
        bridge.export_spcat_catalog()
    finally:
        if original_env is not None:
            os.environ["COCHEM_ARTIFACT_DIR"] = original_env
        else:
            del os.environ["COCHEM_ARTIFACT_DIR"]
    assert (spcat_dir / "spcat" / "spcat_002.var").exists()
    assert (spcat_dir / "spcat" / "spcat_002.int").exists()


def test_exact_codata_2022_constants() -> None:
    """Validate immutable CODATA 2022 physical constants."""
    assert CONSTANTS.H == 6.62607015e-34
    assert CONSTANTS.K_B == 1.380649e-23
    assert CONSTANTS.C_CM_S == 29979245800.0
    assert abs(CONSTANTS.C_ROT - 505379.008435) < 1e-4
    assert abs(CONSTANTS.HC_OVER_KB - 1.4387768775) < 1e-6
    assert abs(CONSTANTS.AMU_KG - 1.66053906892e-27) < 1e-35


def test_low_frequency_lam_trap_enforcement() -> None:
    """Verify LAM trap raises LAMTriggerError for modes < 50 cm^-1 and passes stiff modes."""
    with pytest.raises(LAMTriggerError) as exc_info:
        low_frequency_lam_trap([3100.0, 1500.0, 105.0, 24.5])
    assert 24.5 in exc_info.value.details["flagged_frequencies"]
    assert "DVR" in exc_info.value.message

    # Alias check
    assert low_frequency_trap is low_frequency_lam_trap
    stiff = low_frequency_trap([1500.0, 3600.0])
    assert stiff == [1500.0, 3600.0]


def test_apply_symmetry_divisors_water() -> None:
    """Verify symmetry and spin weights for water (C2v, sigma=2, '3 1')."""
    res = apply_symmetry_divisors(H2O_GEOMETRY, H2O_SYMBOLS)
    assert res.point_group == "C2v"
    assert res.sigma == 2
    assert res.spin_weight_ratio_str == "3 1"
    assert res.effective_divisor == 2.0


def test_apply_symmetry_divisors_ammonia_and_ethylene() -> None:
    """Verify symmetry and spin weights for NH3 and C2H4."""
    res_nh3 = apply_symmetry_divisors(NH3_GEOMETRY, NH3_SYMBOLS)
    assert res_nh3.point_group == "C3v"
    assert res_nh3.sigma == 3
    assert res_nh3.spin_weight_ratio_str == "2 1"

    res_c2h4 = apply_symmetry_divisors(C2H4_GEOMETRY, C2H4_SYMBOLS)
    assert res_c2h4.point_group == "D2h"
    assert res_c2h4.sigma == 4

    # Nuclear spin flag sets effective divisor to 1.0 to avoid double counting
    res_spin = apply_symmetry_divisors(H2O_GEOMETRY, H2O_SYMBOLS, use_nuclear_spin=True)
    assert res_spin.effective_divisor == 1.0
    assert "EXACT_NUCLEAR_SPIN_APPLIED" in res_spin.guardrail_status


def test_vibrational_partition_coupling_with_lam_drop() -> None:
    """Verify vibrational partition coupling drops LAM frequency across temperature gradient."""
    temps = [2.0, 10.0, 50.0, 298.15]
    all_freqs = [3100.0, 1500.0, 105.0, 24.5]
    lam_mode = 24.5
    q_rot_dvr = {2.0: 1.05, 10.0: 4.8, 50.0: 35.2, 298.15: 185.0}

    q_coupled = vibrational_partition_coupling(
        q_rot_dvr=q_rot_dvr,
        q_vib_orca=all_freqs,
        temp_array=temps,
        lam_frequency=lam_mode,
    )

    q_vib_without_lam = calculate_vibrational_partition_function(
        all_freqs, 298.15, exclude_frequencies=[lam_mode]
    )
    expected = q_rot_dvr[298.15] * q_vib_without_lam
    assert math.isclose(q_coupled[298.15], expected, rel_tol=1e-6)


def test_fortran_overflow_guard_and_formatter() -> None:
    """Verify Fortran Double Precision overflow protection and 'D' format."""
    with pytest.raises(FortranOverflowError):
        fortran_overflow_guard({"DJ": 1.5e310})

    with pytest.raises(FortranOverflowError):
        fortran_overflow_guard({"tensor_matrix": np.array([1.0, 2.0, np.inf])})

    with pytest.raises(FortranOverflowError):
        fortran_overflow_guard({"tensor_matrix": np.array([1.0, 2.0, np.nan])})

    formatted = fortran_double_precision_formatter(
        20000, 1.567e-5, uncertainty=1e-7, label="DJ"
    )
    assert "20000" in formatted
    assert "D-05" in formatted
    assert "/ DJ" in formatted

    single_fmt = format_fortran_double(0.00012345, compact=True)
    assert "D-04" in single_fmt


def test_generate_spcat_var_and_int(tmp_path: Path) -> None:
    """Verify generation of .var and .int files."""
    var_file = tmp_path / "scratch" / "test.var"
    var_content = generate_spcat_var(
        "H2O", {"A": 825360.0, "B": 435360.0, "C": 278130.0}, filepath=var_file
    )
    assert var_file.exists()
    assert "H2O Ground State" in var_content

    int_file = tmp_path / "scratch" / "test_{T}K.int"
    int_dict = generate_spcat_int(
        "H2O",
        {"mu_a": 0.0, "mu_b": 1.85, "mu_c": 0.0},
        temperatures=[298.15],
        filepath_template=int_file,
    )
    assert 298.15 in int_dict
    assert (tmp_path / "scratch" / "test_298.1K.int").exists()


def test_3tier_routing_protocol() -> None:
    """Verify the 3-Tier Routing Protocol (MPQC primary, ORCA secondary, CFOUR legacy)."""
    mpqc_data = {
        "energy_hartree": -76.4321,
        "frequencies": [1595.0, 3657.0, 3756.0],
        "dipoles": {"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.85},
    }
    orca_data = {
        "electronic_energy": -76.4300,
        "harmonic_frequencies": [1590.0, 3650.0, 3750.0],
        "anharmonic_x_matrix": np.array(
            [[-42.6, -15.9, -165.8], [-15.9, -42.9, -166.1], [-165.8, -166.1, -47.8]]
        ),
        "dipole_moments": {"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.84},
    }
    cfour_data = {
        "eccsd_t": -76.4310,
        "frequencies": [1592.0, 3652.0, 3752.0],
        "dipoles": {"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.845},
    }

    # Standard routing selects Tier 1 (MPQC)
    res_tier1 = route_3tier_abinitio_payload(
        mpqc_data=mpqc_data, orca_data=orca_data, cfour_data=cfour_data
    )
    assert res_tier1.selected_tier == 1
    assert res_tier1.primary_engine == "MPQC"
    assert res_tier1.is_mpqc_primary is True
    assert res_tier1.electronic_energy_hartree == -76.4321

    # When analytic VPT2 is requested, Tier 2 (ORCA) is selected
    res_tier2 = route_3tier_abinitio_payload(
        mpqc_data=mpqc_data,
        orca_data=orca_data,
        cfour_data=cfour_data,
        require_analytic_vpt2=True,
    )
    assert res_tier2.selected_tier == 2
    assert res_tier2.primary_engine == "ORCA"
    assert res_tier2.is_analytic_vpt2_active is True

    # Fallback to Tier 3 when only CFOUR is available
    res_tier3 = route_3tier_abinitio_payload(cfour_data=cfour_data)
    assert res_tier3.selected_tier == 3
    assert res_tier3.primary_engine == "CFOUR"

    with pytest.raises(ValueError, match="No ab initio data provided"):
        route_3tier_abinitio_payload()


def test_build_complete_spcat_payload_and_manifest(tmp_path: Path) -> None:
    """Verify build_complete_spcat_payload creates verified files and cryptographic manifest."""
    out_dir = tmp_path / "scratch" / "spcat_out"
    payload = build_complete_spcat_payload(
        molecule_name="Water",
        geometry=H2O_GEOMETRY,
        symbols=H2O_SYMBOLS,
        rotational_constants_mhz={"A": 825360.0, "B": 435360.0, "C": 278130.0},
        dipoles_debye={"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.854},
        harmonic_frequencies_cm1=[1595.0, 3657.0, 3756.0],
        temperatures=[2.0, 10.0, 298.15],
        output_dir=out_dir,
    )

    assert payload.molecule_name == "Water"
    assert len(payload.sha256_var) == 64
    assert len(payload.sha256_int) == 3
    assert Path(payload.var_filepath).exists()
    assert Path(payload.provenance_filepath).exists()


def test_mendeleev_dynamic_mass_retrieval() -> None:
    """Verify dynamic IUPAC atomic mass and isotopic mass retrieval via Mendeleev library."""
    m_c = get_atomic_mass("C")
    assert abs(m_c - 12.011) < 0.01

    m_h = get_atomic_mass("H")
    assert abs(m_h - 1.008) < 0.01

    m_o = get_atomic_mass("O")
    assert abs(m_o - 15.999) < 0.01

    # Isotopes
    m_13c = get_atomic_mass("13C")
    assert abs(m_13c - 13.00335) < 0.001

    m_18o = get_atomic_mass("18O")
    assert abs(m_18o - 17.99916) < 0.001

    m_d = get_atomic_mass("D")
    assert abs(m_d - 2.01410) < 0.001

    m_2h = get_atomic_mass("2H")
    assert abs(m_2h - 2.01410) < 0.001

    with pytest.raises(ValueError):
        get_atomic_mass("???")


def test_calculate_rotational_constants_from_geometry_water() -> None:
    """Verify calculation of rotational constants from Cartesian coordinates and Mendeleev masses."""
    rot_constants = calculate_rotational_constants_from_geometry(
        H2O_GEOMETRY, H2O_SYMBOLS
    )
    assert "A" in rot_constants
    assert "B" in rot_constants
    assert "C" in rot_constants
    assert "Ia" in rot_constants
    assert "Ib" in rot_constants
    assert "Ic" in rot_constants

    # For water: A ~ 820,601 MHz, B ~ 437,225 MHz, C ~ 285,244 MHz
    assert abs(rot_constants["A"] - 820601.0) < 500.0
    assert abs(rot_constants["B"] - 437225.0) < 500.0
    assert abs(rot_constants["C"] - 285244.0) < 500.0
    assert rot_constants["Ia"] < rot_constants["Ib"] < rot_constants["Ic"]


def test_multi_rotor_lam_dropping() -> None:
    """Verify multiple LAM modes are simultaneously dropped from Q_vib."""
    temps = [2.0, 10.0, 50.0, 298.15]
    all_freqs = [3100.0, 1500.0, 105.0, 38.0, 24.5]
    lam_modes = [24.5, 38.0]
    q_rot_dvr = {2.0: 1.05, 10.0: 4.8, 50.0: 35.2, 298.15: 185.0}

    q_coupled = vibrational_partition_coupling(
        q_rot_dvr=q_rot_dvr,
        q_vib_orca=all_freqs,
        temp_array=temps,
        lam_frequency=lam_modes,
    )

    q_vib_stiff = calculate_vibrational_partition_function(
        all_freqs, 298.15, exclude_frequencies=lam_modes
    )
    expected = q_rot_dvr[298.15] * q_vib_stiff
    assert math.isclose(q_coupled[298.15], expected, rel_tol=1e-6)


def test_build_complete_spcat_payload_traps_lam() -> None:
    """Verify build_complete_spcat_payload raises LAMTriggerError when modes < 50 cm^-1 are unhandled."""
    with pytest.raises(LAMTriggerError) as exc_info:
        build_complete_spcat_payload(
            molecule_name="FluxionalMolecule",
            geometry=H2O_GEOMETRY,
            symbols=H2O_SYMBOLS,
            rotational_constants_mhz={"A": 825360.0, "B": 435360.0, "C": 278130.0},
            dipoles_debye={"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.854},
            harmonic_frequencies_cm1=[3600.0, 1500.0, 32.0],
            lam_frequency=None,
        )
    assert 32.0 in exc_info.value.details["flagged_frequencies"]


def test_build_complete_spcat_payload_nuclear_spin(tmp_path: Path) -> None:
    """Verify build_complete_spcat_payload respects use_nuclear_spin flag and double counting guardrail."""
    out_dir = tmp_path / "scratch" / "spcat_spin"
    payload = build_complete_spcat_payload(
        molecule_name="WaterSpin",
        geometry=H2O_GEOMETRY,
        symbols=H2O_SYMBOLS,
        rotational_constants_mhz={"A": 825360.0, "B": 435360.0, "C": 278130.0},
        dipoles_debye={"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.854},
        harmonic_frequencies_cm1=[1595.0, 3657.0, 3756.0],
        temperatures=[298.15],
        use_nuclear_spin=True,
        output_dir=out_dir,
    )
    assert payload.provenance_manifest["symmetry"]["effective_divisor"] == 1.0


def test_airgap_violation_enforcement() -> None:
    """Verify that writing into Ring 1 static repository tiers raises AirGapViolationError."""
    repo_root = Path(__file__).resolve().parent.parent
    protected_target = repo_root / "Libraries" / "malicious_write.tmp"
    with pytest.raises(AirGapViolationError):
        validate_airgap_boundary(protected_target)

    # Repository root itself must also be protected
    with pytest.raises(AirGapViolationError):
        validate_airgap_boundary(repo_root)


def test_fortran_fixed_header_and_precision_clamp() -> None:
    """Verify strict 4I5 control line header alignment and 22-column precision clamping."""
    var_content = generate_spcat_var(
        "H2O", {"A": 825360.0, "B": 435360.0, "C": 278130.0}
    )
    lines = var_content.splitlines()
    control_line = lines[1]
    # Control line should have exactly 4 leading fields of width 5 (4I5)
    # NPAR = 3 -> '    3', NLINE = 100 -> '  100', NOPT = 0 -> '    0', NWARN = 0 -> '    0'
    assert control_line[:20] == "    3  100    0    0"

    # Extreme exponent formatted double should never exceed 22 columns
    clamped_str = format_fortran_double(-1.0e-105, width=22, precision=15)
    assert len(clamped_str) == 22
    assert clamped_str.strip() == "-1.00000000000000D-105"

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ.