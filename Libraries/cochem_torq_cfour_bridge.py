"""CFOUR input, partial-output and immutable-archive helpers.

Missing quantities remain unavailable. The parser has not been certified on
release-pinned genuine CFOUR fixtures, so it does not infer scientific
convergence or effective method/basis from a successful process exit. The
unsupported generic gradient/Hessian adapter fails before launching a job.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CFOURUnavailableError(RuntimeError):
    """The requested CFOUR executable or validated capability is unavailable."""


try:
    from mendeleev import element as get_mendeleev_element

    _MENDELEEV_AVAILABLE = True
except ImportError:
    _MENDELEEV_AVAILABLE = False


# ============================================================================
# 0. Subprocess Lifecycle Management & Zombie Cleanup
# ============================================================================


def cleanup_zombies() -> None:
    """Reap only CFOUR processes started by this module, never other users' jobs."""
    for process in tuple(_ACTIVE_CFOUR_PROCESSES):
        if process.poll() is None:
            process.kill()
        process.wait()
        _ACTIVE_CFOUR_PROCESSES.discard(process)


_ACTIVE_CFOUR_PROCESSES: set[subprocess.Popen] = set()
atexit.register(cleanup_zombies)

# Module-level logger
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-CFOUR] %(message)s"
)
logger = logging.getLogger("TorqCfourBridge")

CFOUR_PATH = os.environ.get("CFOUR_PATH", "xcfour")
ARTIFACTS_DIR = os.environ.get(
    "COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts")
)


# ============================================================================
# 1. Recorded Physical Conversion Constants & Dynamic Mendeleev Retrievals
# ============================================================================


@dataclass(frozen=True)
class CFOURPhysicalConstants:
    """Recorded conversion values; only SI-defined h and c are exact."""

    # Speed of light in vacuum [cm / s]
    C_CM_S: float = 29979245800.0
    # Planck constant [J * s]
    PLANCK_H: float = 6.62607015e-34
    # Atomic mass unit to kg [kg / amu]
    AMU_TO_KG: float = 1.66053906660e-27
    # Bohr radius to Angstroms [Angstrom / bohr]
    BOHR_TO_ANGSTROM: float = 0.529177210903
    ANGSTROM_TO_BOHR: float = 1.0 / 0.529177210903
    # Hartree to eV
    HARTREE_TO_EV: float = 27.211386245988
    # Hartree to kcal/mol
    HARTREE_TO_KCAL_MOL: float = 627.509474063
    # Wave number to Frequency conversion factor [MHz / cm^-1]
    CM1_TO_MHZ: float = 29979.2458
    MHZ_TO_CM1: float = 1.0 / 29979.2458
    # Atomic unit of electric dipole moment to Debye
    AU_TO_DEBYE: float = 2.5417464519
    # EFG atomic units to nuclear quadrupole coupling conversion factor [kHz / (a.u. * mbarn)]
    # chi [kHz] = EFG [a.u.] * Q [mbarn] * 234.96474 (Method Matrix Section 9.3 & 14.1)
    EFG_TO_KHZ_FACTOR: float = 234.96474
    # Rotational constant conversion factor: h / (8 * pi^2) in amu * Angstrom^2 * MHz
    # A [MHz] = INERTIA_FACTOR / I_A [amu * Angstrom^2]
    INERTIA_TO_MHZ_FACTOR: float = 505379.00878


CONSTANTS = CFOURPhysicalConstants()


def get_atomic_mass(symbol: str) -> float:
    """
    Dynamically retrieves standard atomic weight (amu) using mendeleev.
    Strictly follows the Mendeleev Mandate.
    """
    clean_sym = symbol.strip().rstrip(":").capitalize()
    if _MENDELEEV_AVAILABLE:
        el = get_mendeleev_element(clean_sym)
        if el.atomic_weight is not None:
            return float(el.atomic_weight)
        if el.mass is not None:
            return float(el.mass)
    raise ValueError(f"Could not retrieve atomic mass for element symbol '{symbol}'.")


def get_isotopic_mass(symbol: str, mass_number: Optional[int] = None) -> float:
    """
    Dynamically retrieves isotopic mass using mendeleev.
    """
    clean_sym = symbol.strip().rstrip(":").capitalize()
    if _MENDELEEV_AVAILABLE:
        el = get_mendeleev_element(clean_sym)
        if mass_number is None:
            return get_atomic_mass(symbol)
        for iso in el.isotopes:
            if iso.mass_number == mass_number:
                return float(iso.mass)
    raise ValueError(f"No isotopic mass is available for {symbol}-{mass_number}.")


def get_atomic_number(symbol: str) -> int:
    """
    Dynamically retrieves atomic number (Z) using mendeleev.
    """
    clean_sym = symbol.strip().rstrip(":").capitalize()
    if _MENDELEEV_AVAILABLE:
        el = get_mendeleev_element(clean_sym)
        return int(el.atomic_number)
    raise ValueError(f"Could not retrieve atomic number for element symbol '{symbol}'.")


# ============================================================================
# 2. Mathematical Rigid Rotor & Geometry Utilities
# ============================================================================


def compute_center_of_mass(
    symbols: Sequence[str], coordinates: np.ndarray
) -> np.ndarray:
    """
    Computes the center of mass vector using dynamic Mendeleev masses.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    masses = np.array([get_atomic_mass(s) for s in symbols], dtype=np.float64)
    total_mass = np.sum(masses)
    if total_mass <= 0.0:
        raise ValueError("Total molecular mass must be greater than zero.")
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    return com


def compute_inertia_tensor(
    symbols: Sequence[str], coordinates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the principal moments of inertia and rotational constants (MHz).
    Returns (moments_amu_ang2, principal_axes, rotational_constants_mhz).
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    com = compute_center_of_mass(symbols, coords)
    shifted_coords = coords - com
    masses = np.array([get_atomic_mass(s) for s in symbols], dtype=np.float64)

    # Construct 3x3 inertia tensor
    I_tensor = np.full((3, 3), 0.0, dtype=np.float64)
    for i, m in enumerate(masses):
        x, y, z = shifted_coords[i]
        I_tensor[0, 0] += m * (y * y + z * z)
        I_tensor[1, 1] += m * (x * x + z * z)
        I_tensor[2, 2] += m * (x * x + y * y)
        I_tensor[0, 1] -= m * x * y
        I_tensor[0, 2] -= m * x * z
        I_tensor[1, 2] -= m * y * z

    I_tensor[1, 0] = I_tensor[0, 1]
    I_tensor[2, 0] = I_tensor[0, 2]
    I_tensor[2, 1] = I_tensor[1, 2]

    eigvals, eigvecs = np.linalg.eigh(I_tensor)
    # Sort in ascending order: I_A <= I_B <= I_C
    sort_idx = np.argsort(eigvals)
    sorted_moments = eigvals[sort_idx]
    sorted_axes = eigvecs[:, sort_idx]

    # A singular principal moment requires a separate linear/atomic model.
    if not np.all(np.isfinite(sorted_moments)) or np.any(sorted_moments <= 1e-6):
        raise ValueError(
            "Three finite rotational constants require a nonlinear rotor; use a qualified linear/atomic model."
        )
    rot_constants_mhz = CONSTANTS.INERTIA_TO_MHZ_FACTOR / sorted_moments

    return sorted_moments, sorted_axes, rot_constants_mhz


# ============================================================================
# 3. Pydantic Structured Data Models
# ============================================================================


class CFOURRotationalConstants(BaseModel):
    """Equilibrium and vibrationally corrected rotational constants."""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_assignment=True
    )

    Ae_MHz: Optional[float] = None
    Be_MHz: Optional[float] = None
    Ce_MHz: Optional[float] = None
    Ae_cm1: Optional[float] = None
    Be_cm1: Optional[float] = None
    Ce_cm1: Optional[float] = None

    A0_MHz: Optional[float] = None
    B0_MHz: Optional[float] = None
    C0_MHz: Optional[float] = None
    A0_cm1: Optional[float] = None
    B0_cm1: Optional[float] = None
    C0_cm1: Optional[float] = None

    alpha_A_MHz: Optional[List[float]] = None
    alpha_B_MHz: Optional[List[float]] = None
    alpha_C_MHz: Optional[List[float]] = None

    alpha_A_cm1: Optional[List[float]] = None
    alpha_B_cm1: Optional[List[float]] = None
    alpha_C_cm1: Optional[List[float]] = None

    inertial_defect_amu_angstrom2: Optional[float] = None
    asymmetry_kappa: Optional[float] = None
    asymmetry_source: Optional[str] = None


class CFOURQuarticDistortion(BaseModel):
    """Watson A-reduction and S-reduction quartic centrifugal distortion constants."""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_assignment=True
    )

    # Watson A-reduction parameters (kHz)
    Delta_J_kHz: Optional[float] = None
    Delta_JK_kHz: Optional[float] = None
    Delta_K_kHz: Optional[float] = None
    delta_J_kHz: Optional[float] = None
    delta_K_kHz: Optional[float] = None

    # Watson S-reduction parameters (kHz)
    D_J_kHz: Optional[float] = None
    D_JK_kHz: Optional[float] = None
    D_K_kHz: Optional[float] = None
    d_1_kHz: Optional[float] = None
    d_2_kHz: Optional[float] = None

    reduction_type: Optional[str] = None


class CFOURSexticDistortion(BaseModel):
    """
    Watson A-reduction and S-reduction sextic centrifugal distortion constants.
    Values and reduction must be parsed with explicit units.
    """

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_assignment=True
    )

    # Watson A-reduction parameters (Hz)
    Phi_J_Hz: Optional[float] = None
    Phi_JK_Hz: Optional[float] = None
    Phi_KJ_Hz: Optional[float] = None
    Phi_K_Hz: Optional[float] = None
    phi_J_Hz: Optional[float] = None
    phi_JK_Hz: Optional[float] = None
    phi_K_Hz: Optional[float] = None

    # Watson S-reduction parameters (Hz)
    H_J_Hz: Optional[float] = None
    H_JK_Hz: Optional[float] = None
    H_KJ_Hz: Optional[float] = None
    H_K_Hz: Optional[float] = None
    h_1_Hz: Optional[float] = None
    h_2_Hz: Optional[float] = None
    h_3_Hz: Optional[float] = None

    reduction_type: Optional[str] = None


class CFOURVibrationalData(BaseModel):
    """Harmonic and VPT2 anharmonic vibrational frequencies and force fields."""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_assignment=True
    )

    harmonic_frequencies_cm1: Optional[List[float]] = None
    anharmonic_frequencies_cm1: Optional[List[float]] = None
    ir_intensities_km_mol: Optional[List[float]] = None
    symmetry_irreps: Optional[List[str]] = None
    harmonic_zpe_kcal_mol: Optional[float] = None
    anharmonic_zpe_kcal_mol: Optional[float] = None
    x_matrix_cm1: Optional[List[List[float]]] = None
    cubic_force_constants_cm1: Optional[Dict[str, float]] = None
    semidiagonal_quartic_cm1: Optional[Dict[str, float]] = None


class CFOURDipoleMoment(BaseModel):
    """Electric dipole moment in principal axis frame and Cartesian components."""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_assignment=True
    )

    mu_a_debye: Optional[float] = None
    mu_b_debye: Optional[float] = None
    mu_c_debye: Optional[float] = None
    mu_total_debye: Optional[float] = None

    mu_a_au: Optional[float] = None
    mu_b_au: Optional[float] = None
    mu_c_au: Optional[float] = None
    mu_total_au: Optional[float] = None


class CFOURNuclearQuadrupole(BaseModel):
    """Electric Field Gradient (EFG) and Nuclear Quadrupole Coupling Tensor."""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_assignment=True
    )

    atom_index: int = Field(ge=1)
    element: str = Field(min_length=1)
    isotope_mass_number: Optional[int] = Field(default=None, ge=1)
    nuclear_q_source: Optional[str] = None
    tensor_frame: str = "engine_output_unverified"
    chi_principal_values_kHz: Optional[List[float]] = None
    Q_mbarn: Optional[float] = None
    efg_tensor_au: Optional[List[List[float]]] = None
    chi_tensor_kHz: Optional[List[List[float]]] = None
    chi_aa_kHz: Optional[float] = None
    chi_bb_kHz: Optional[float] = None
    chi_cc_kHz: Optional[float] = None
    asymmetry_eta: Optional[float] = None


class CFOURSpinRotation(BaseModel):
    """Nuclear spin-rotation coupling tensor (SPINROT=ON)."""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_assignment=True
    )

    atom_index: int = Field(ge=1)
    element: str = Field(min_length=1)
    tensor_frame: str = "engine_output_unverified"
    C_tensor_kHz: Optional[List[List[float]]] = None
    C_aa_kHz: Optional[float] = None
    C_bb_kHz: Optional[float] = None
    C_cc_kHz: Optional[float] = None
    C_iso_kHz: Optional[float] = None


class CFOUREnergies(BaseModel):
    """Electronic energies and corrections in Hartree."""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_assignment=True
    )

    scf_energy_hartree: Optional[float] = None
    mp2_energy_hartree: Optional[float] = None
    ccsd_energy_hartree: Optional[float] = None
    ccsd_t_energy_hartree: Optional[float] = None
    final_energy_hartree: Optional[float] = None
    correlation_energy_hartree: Optional[float] = None
    dboc_correction_hartree: Optional[float] = None
    relativistic_correction_hartree: Optional[float] = None


class CFOUROutputPayload(BaseModel):
    """Master structured container for complete CFOUR calculation results."""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_assignment=True
    )

    molecule_name: str = ""
    calc_method: Optional[str] = None
    requested_method: Optional[str] = None
    basis_set: Optional[str] = None
    requested_basis: Optional[str] = None
    energies: CFOUREnergies = Field(default_factory=CFOUREnergies)
    rotational_constants: CFOURRotationalConstants = Field(
        default_factory=CFOURRotationalConstants
    )
    quartic_distortion: CFOURQuarticDistortion = Field(
        default_factory=CFOURQuarticDistortion
    )
    sextic_distortion: CFOURSexticDistortion = Field(
        default_factory=CFOURSexticDistortion
    )
    vibrational_data: CFOURVibrationalData = Field(default_factory=CFOURVibrationalData)
    dipole_moment: CFOURDipoleMoment = Field(default_factory=CFOURDipoleMoment)
    quadrupole_coupling: List[CFOURNuclearQuadrupole] = Field(default_factory=list)
    spin_rotation: List[CFOURSpinRotation] = Field(default_factory=list)
    optimized_coordinates: Optional[List[List[Union[str, float]]]] = None
    raw_output_sha256: str = ""
    calculation_converged: Optional[bool] = None
    process_returncode: Optional[int] = None
    validation_issues: List[str] = Field(default_factory=list)
    wall_clock_seconds: Optional[float] = None


# ============================================================================
# 4. Z-Matrix Builder with Dummy-Atom Singularity Protection
# ============================================================================


class CFOURZmatBuilder:
    """
    Generates CFOUR-compliant Z-matrix internal coordinate files (ZMAT).
    Enforces Method Matrix §9.5:
    - 3-character variable names (e.g. R1, A1, D1)
    - Dummy atom 'X' insertion to avoid 0° and 180° angle singularities
    - Single-space delimited variable blocks
    - Correct *CFOUR(...) keywords, memory specifications, and %isotopes sections.
    """

    @staticmethod
    def _compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def _compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Computes angle p1-p2-p3 in degrees."""
        v1 = p1 - p2
        v2 = p3 - p2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if not np.isfinite([norm1, norm2]).all() or norm1 < 1e-8 or norm2 < 1e-8:
            raise ValueError(
                "Undefined Z-matrix angle: finite nonzero bond vectors are required."
            )
        cosine = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _compute_dihedral(
        p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
    ) -> float:
        """Computes dihedral angle p1-p2-p3-p4 in degrees [-180, 180]."""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)
        norm_b2 = np.linalg.norm(b2)

        if (
            not np.isfinite([norm_n1, norm_n2, norm_b2]).all()
            or norm_n1 < 1e-8
            or norm_n2 < 1e-8
            or norm_b2 < 1e-8
        ):
            raise ValueError(
                "Undefined Z-matrix dihedral: finite nonzero reference planes and axis are required."
            )

        n1 /= norm_n1
        n2 /= norm_n2
        m1 = np.cross(n1, b2 / norm_b2)

        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return float(np.degrees(np.arctan2(y, x)))

    @classmethod
    def cartesian_to_zmat(
        cls,
        symbols: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        title: str = "CFOUR Calculation",
        optimize_all: bool = True,
        linear_angle_threshold_deg: float = 5.0,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Converts Cartesian coordinates to a valid CFOUR Z-matrix string with variable block.
        Inserts dummy atoms 'X' when angles approach 0° or 180° to avoid singularities.
        """
        n_atoms = len(symbols)
        if n_atoms == 0:
            raise ValueError("Cannot construct ZMAT from empty atom list.")

        geometry = np.asarray(coordinates, dtype=np.float64)
        if geometry.shape != (n_atoms, 3) or not np.isfinite(geometry).all():
            raise ValueError(
                "Z-matrix input requires aligned finite Cartesian coordinates."
            )
        for index in range(n_atoms):
            if np.any(
                np.linalg.norm(geometry[index + 1 :] - geometry[index], axis=1) < 1e-8
            ):
                raise ValueError("Coincident nuclei cannot define a Z-matrix geometry.")
        coords = [coordinate for coordinate in geometry]
        zmat_lines: List[str] = []
        variables: Dict[str, float] = {}

        # Tracking placed points (symbol, 3D position, 1-based index)
        placed_symbols: List[str] = []
        placed_positions: List[np.ndarray] = []

        var_r_count = 0
        var_a_count = 0
        var_d_count = 0

        for i in range(n_atoms):
            sym = symbols[i].strip()
            curr_pos = coords[i]

            if i == 0:
                zmat_lines.append(f"{sym}")
                placed_symbols.append(sym)
                placed_positions.append(curr_pos)

            elif i == 1:
                var_r_count += 1
                r_name = f"R{var_r_count}"
                dist = cls._compute_distance(curr_pos, placed_positions[0])
                zmat_lines.append(f"{sym} 1 {r_name}")
                variables[r_name] = dist
                placed_symbols.append(sym)
                placed_positions.append(curr_pos)

            elif i == 2:
                # Check for angle singularity between atom 0, atom 1, and atom 2
                ang = cls._compute_angle(
                    curr_pos, placed_positions[1], placed_positions[0]
                )
                if ang < linear_angle_threshold_deg or ang > (
                    180.0 - linear_angle_threshold_deg
                ):
                    # Introduce dummy atom X
                    var_r_count += 1
                    rx_name = f"R{var_r_count}"
                    v_axis = placed_positions[1] - placed_positions[0]
                    v_axis /= np.linalg.norm(v_axis)
                    perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    if abs(np.dot(perp, v_axis)) > 0.9:
                        perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    perp = np.cross(v_axis, perp)
                    perp /= np.linalg.norm(perp)

                    x_pos = placed_positions[1] + perp * 1.0
                    x_idx = len(placed_positions) + 1

                    zmat_lines.append(f"X 2 {rx_name} 1 90.0")
                    variables[rx_name] = 1.0
                    placed_symbols.append("X")
                    placed_positions.append(x_pos)

                    # Now place atom 2 relative to X
                    var_r_count += 1
                    r_name = f"R{var_r_count}"
                    var_a_count += 1
                    a_name = f"A{var_a_count}"

                    dist = cls._compute_distance(curr_pos, placed_positions[1])
                    ang_x = cls._compute_angle(curr_pos, placed_positions[1], x_pos)

                    zmat_lines.append(f"{sym} 2 {r_name} {x_idx} {a_name}")
                    variables[r_name] = dist
                    variables[a_name] = ang_x
                    placed_symbols.append(sym)
                    placed_positions.append(curr_pos)
                else:
                    var_r_count += 1
                    r_name = f"R{var_r_count}"
                    var_a_count += 1
                    a_name = f"A{var_a_count}"
                    dist = cls._compute_distance(curr_pos, placed_positions[1])
                    ang = cls._compute_angle(
                        curr_pos, placed_positions[1], placed_positions[0]
                    )

                    zmat_lines.append(f"{sym} 2 {r_name} 1 {a_name}")
                    variables[r_name] = dist
                    variables[a_name] = ang
                    placed_symbols.append(sym)
                    placed_positions.append(curr_pos)

            else:
                # 4th atom onwards: Distance to atom (i), Angle to atom (i-1), Dihedral to atom (i-2)
                ref1 = len(placed_positions)
                ref2 = ref1 - 1
                ref3 = ref2 - 1

                dist = cls._compute_distance(curr_pos, placed_positions[ref1 - 1])
                ang = cls._compute_angle(
                    curr_pos, placed_positions[ref1 - 1], placed_positions[ref2 - 1]
                )
                dih = cls._compute_dihedral(
                    curr_pos,
                    placed_positions[ref1 - 1],
                    placed_positions[ref2 - 1],
                    placed_positions[ref3 - 1],
                )

                var_r_count += 1
                r_name = f"R{var_r_count}"
                var_a_count += 1
                a_name = f"A{var_a_count}"
                var_d_count += 1
                d_name = f"D{var_d_count}"

                zmat_lines.append(
                    f"{sym} {ref1} {r_name} {ref2} {a_name} {ref3} {d_name}"
                )
                variables[r_name] = dist
                variables[a_name] = ang
                variables[d_name] = dih
                placed_symbols.append(sym)
                placed_positions.append(curr_pos)

        # Assemble the ZMAT body
        body = [title]
        body.extend(zmat_lines)
        body.append("")  # Mandatory blank line before variables

        opt_flag = "*" if optimize_all else ""
        for vname, val in variables.items():
            if vname.startswith("R"):
                body.append(f"{vname}{opt_flag} = {val:12.6f}")
            else:
                body.append(f"{vname}{opt_flag} = {val:12.4f}")

        return "\n".join(body), variables

    @classmethod
    def generate_full_zmat_input(
        cls,
        symbols: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        method: str,
        basis: str,
        reference: str = "RHF",
        frozen_core: bool = True,
        anharm: str = "VPT2",
        vib: str = "EXACT",
        props: str = "FIRST_ORDER",
        memory_gb: int = 32,
        optimize: bool = False,
        extra_keywords: Optional[Dict[str, Any]] = None,
        isotopes: Optional[Sequence[int]] = None,
        title: str = "CFOUR Coupled-Cluster Calculation",
    ) -> str:
        """
        Constructs a complete CFOUR input deck (ZMAT) with parameter block and %isotopes.
        """
        zmat_body, _ = cls.cartesian_to_zmat(
            symbols=symbols, coordinates=coordinates, title=title, optimize_all=optimize
        )

        # Method Matrix standard keyword dictionary
        cfour_kw: Dict[str, Any] = {
            "CALC": method.upper(),
            "BASIS": basis.upper(),
            "REFERENCE": reference.upper(),
            "FROZEN_CORE": "ON" if frozen_core else "OFF",
            "ABCDTYPE": "AOBASIS",
            "CC_PROG": "ECC",
            "SPHERICAL": "ON",
            "UNITS": "ANGSTROM",
            "MEMORY_SIZE": memory_gb,
            "MEM_UNIT": "GB",
            "SCF_CONV": 10,
            "CC_CONV": 10,
            "LINEQ_CONV": 10,
            "GEO_CONV": 5,
        }

        if vib:
            cfour_kw["VIB"] = vib.upper()
        if anharm:
            cfour_kw["ANHARM"] = anharm.upper()
            cfour_kw["ANH_STEPSIZ"] = 50000
            cfour_kw["FD_PROJECT"] = "ON"
        if props:
            cfour_kw["PROPS"] = props.upper()

        if extra_keywords:
            cfour_kw.update(extra_keywords)

        # Construct *CFOUR(...) block
        kw_lines = []
        for k, v in cfour_kw.items():
            kw_lines.append(f"{k}={v}")
        cfour_block = "*CFOUR(\n" + "\n".join(kw_lines) + "\n)"

        # Construct %isotopes block if provided
        isotope_block = ""
        if isotopes:
            iso_lines = [str(iso) for iso in isotopes]
            isotope_block = "\n\n%isotopes\n" + "\n".join(iso_lines)

        return f"{zmat_body}\n\n{cfour_block}{isotope_block}\n"


# ============================================================================
# 5. CFOUR Output Parser Engine
# ============================================================================


class CFOUROutputParser:
    """
    High-fidelity regular expression parser for CFOUR text outputs.
    Extracts energies, harmonic/anharmonic frequencies, rotational constants (Ae, Be, Ce, A0, B0, C0),
    alpha parameters, Watson quartic/sextic centrifugal distortion, dipole moments, EFGs, and spin-rotation.
    """

    @classmethod
    def parse_energies(cls, text: str) -> CFOUREnergies:
        """Keep parsed components without promoting missing higher-level energy."""
        energies = CFOUREnergies()
        number = r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])"
        patterns = {
            "scf_energy_hartree": r"(?:E\(SCF\)\s*=|Reference energy\s+is)\s*",
            "mp2_energy_hartree": r"(?:Total MP2 energy\s*:|E\(MP2\)\s*=)\s*",
            "ccsd_energy_hartree": r"(?:Total CCSD energy\s*:|E\(CCSD\)\s*=)\s*",
            "ccsd_t_energy_hartree": r"(?:Total CCSD\(T\) energy\s*:|E\(CCSD\(T\)\)\s*=)\s*",
            "dboc_correction_hartree": r"(?:Total DBOC\s*=|DBOC energy correction\s*:)\s*",
            "relativistic_correction_hartree": r"(?:Relativistic correction\s*=|Total Relativistic Energy\s*:)\s*",
        }
        for field, pattern in patterns.items():
            matches = list(re.finditer(pattern + number, text))
            if matches:
                value = matches[-1].group(1).replace("D", "E").replace("d", "e")
                setattr(energies, field, float(value))
        # A specifically reported total is retained as reported, but parsing
        # alone never establishes the requested method's scientific convergence.
        totals = list(
            re.finditer(
                r"Total energy for calculation type (?:SCF|HF|MP2|CCSD|CCSD\(T\))\s*:\s*"
                + number,
                text,
            )
        )
        if totals:
            energies.final_energy_hartree = float(
                totals[-1].group(1).replace("D", "E").replace("d", "e")
            )
        return energies

    @classmethod
    def parse_rotational_constants(cls, text: str) -> CFOURRotationalConstants:
        """Extracts equilibrium Ae, Be, Ce and effective ground state A0, B0, C0."""
        rc = CFOURRotationalConstants()

        # Equilibrium constants (cm^-1 or MHz)
        m_rc_cm = re.search(
            r"Rotational constants\s*\(in cm-1\):\s*A\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])\s*B\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])\s*C\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])",
            text,
            re.IGNORECASE,
        )
        m_rc_mhz = re.search(
            r"Rotational constants\s*\(in MHz\):\s*A\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])\s*B\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])\s*C\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])",
            text,
            re.IGNORECASE,
        )

        if m_rc_mhz:
            rc.Ae_MHz = float(m_rc_mhz.group(1).replace("D", "E").replace("d", "e"))
            rc.Be_MHz = float(m_rc_mhz.group(2).replace("D", "E").replace("d", "e"))
            rc.Ce_MHz = float(m_rc_mhz.group(3).replace("D", "E").replace("d", "e"))
            rc.Ae_cm1 = rc.Ae_MHz * CONSTANTS.MHZ_TO_CM1
            rc.Be_cm1 = rc.Be_MHz * CONSTANTS.MHZ_TO_CM1
            rc.Ce_cm1 = rc.Ce_MHz * CONSTANTS.MHZ_TO_CM1
        elif m_rc_cm:
            rc.Ae_cm1 = float(m_rc_cm.group(1).replace("D", "E").replace("d", "e"))
            rc.Be_cm1 = float(m_rc_cm.group(2).replace("D", "E").replace("d", "e"))
            rc.Ce_cm1 = float(m_rc_cm.group(3).replace("D", "E").replace("d", "e"))
            rc.Ae_MHz = rc.Ae_cm1 * CONSTANTS.CM1_TO_MHZ
            rc.Be_MHz = rc.Be_cm1 * CONSTANTS.CM1_TO_MHZ
            rc.Ce_MHz = rc.Ce_cm1 * CONSTANTS.CM1_TO_MHZ

        # Ground state rotational constants (A0, B0, C0) and vibration-rotation shifts
        m_shift_sec = re.search(
            r"Be,\s*B0\s*AND\s*B-B0\s*SHIFTS\s*FOR\s*SINGLY\s*EXCITED\s*VIBRATIONAL\s*STATES.*?\n(.*?)(?=\n\n|\n[A-Z]|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m_shift_sec:
            lines = m_shift_sec.group(1).strip().splitlines()
            for line in lines:
                if "Ground State" in line or "State 0" in line:
                    values = re.sub(r"^\s*(?:Ground State|State 0)\s+", "", line)
                    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", values)]
                    if len(nums) == 3:
                        rc.A0_cm1 = nums[0]
                        rc.B0_cm1 = nums[1]
                        rc.C0_cm1 = nums[2]
                        rc.A0_MHz = rc.A0_cm1 * CONSTANTS.CM1_TO_MHZ
                        rc.B0_MHz = rc.B0_cm1 * CONSTANTS.CM1_TO_MHZ
                        rc.C0_MHz = rc.C0_cm1 * CONSTANTS.CM1_TO_MHZ

        # Vibration-rotation alpha constants (alpha_A, alpha_B, alpha_C)
        m_alpha_sec = re.search(
            r"Vibration-rotation\s*interaction\s*constants\s*\(in MHz\):\s*\n(.*?)(?=\n\n|\n[A-Z]|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m_alpha_sec:
            for line in m_alpha_sec.group(1).strip().splitlines():
                parts = line.split()
                if len(parts) == 4:
                    try:
                        a_A = float(parts[1])
                        a_B = float(parts[2])
                        a_C = float(parts[3])
                        if rc.alpha_A_MHz is None:
                            for name in (
                                "alpha_A_MHz",
                                "alpha_B_MHz",
                                "alpha_C_MHz",
                                "alpha_A_cm1",
                                "alpha_B_cm1",
                                "alpha_C_cm1",
                            ):
                                setattr(rc, name, [])
                        rc.alpha_A_MHz.append(a_A)
                        rc.alpha_B_MHz.append(a_B)
                        rc.alpha_C_MHz.append(a_C)
                        rc.alpha_A_cm1.append(a_A * CONSTANTS.MHZ_TO_CM1)
                        rc.alpha_B_cm1.append(a_B * CONSTANTS.MHZ_TO_CM1)
                        rc.alpha_C_cm1.append(a_C * CONSTANTS.MHZ_TO_CM1)
                    except ValueError:
                        continue

        # Do not derive B0 from a potentially incomplete or ambiguously
        # degenerate alpha table. The typed spectroscopy stage validates that.
        ground = (rc.A0_MHz, rc.B0_MHz, rc.C0_MHz)
        equilibrium = (rc.Ae_MHz, rc.Be_MHz, rc.Ce_MHz)
        selected = None
        if all(value is not None for value in ground):
            selected, rc.asymmetry_source = ground, "vibrational_ground_state"
        elif all(value is not None for value in equilibrium):
            selected, rc.asymmetry_source = equilibrium, "equilibrium"
        if selected is not None:
            A, B, C = selected
            if abs(A - C) > 1e-6:
                rc.asymmetry_kappa = float((2.0 * B - A - C) / (A - C))

        return rc

    @classmethod
    def parse_vibrational_data(cls, text: str) -> CFOURVibrationalData:
        """Extracts harmonic frequencies, VPT2 anharmonic fundamentals, and ZPE."""
        vib = CFOURVibrationalData()

        # Only a bare list with explicit decimal/scientific numbers is
        # understood here. Indexed tables require a versioned format parser;
        # scanning every number could mistake mode indices for frequencies.
        number = r"[-+]?(?:\d+\.\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
        headers = {
            "harmonic_frequencies_cm1": r"Harmonic\s+vibrational\s+frequencies",
            "anharmonic_frequencies_cm1": r"(?:Anharmonic\s+vibrational\s+frequencies|VPT2\s+Fundamentals)",
        }
        for field, header in headers.items():
            match = re.search(
                header + r"\s*\(cm-1\):\s*\n(.*?)(?=\n\s*\n|\Z)",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                block = match.group(1).strip()
                if re.fullmatch(number + r"(?:\s+" + number + r")*", block):
                    # Negative/zero frequencies are evidence, never discarded.
                    setattr(
                        vib,
                        field,
                        [
                            float(item.replace("D", "E").replace("d", "e"))
                            for item in block.split()
                        ],
                    )

        # Zero-point energy (ZPE)
        m_zpe = re.search(
            r"Zero\s*point\s*vibrational\s*energy\s*:\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])\s*kcal/mol",
            text,
            re.IGNORECASE,
        )
        if m_zpe:
            vib.harmonic_zpe_kcal_mol = float(
                m_zpe.group(1).replace("D", "E").replace("d", "e")
            )

        m_azpe = re.search(
            r"Anharmonic\s*zero\s*point\s*energy\s*:\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])\s*kcal/mol",
            text,
            re.IGNORECASE,
        )
        if m_azpe:
            vib.anharmonic_zpe_kcal_mol = float(
                m_azpe.group(1).replace("D", "E").replace("d", "e")
            )

        return vib

    @classmethod
    def parse_centrifugal_distortion(
        cls, text: str
    ) -> Tuple[CFOURQuarticDistortion, CFOURSexticDistortion]:
        """Parse explicit named values with explicit units and case semantics."""
        quartic, sextic = CFOURQuarticDistortion(), CFOURSexticDistortion()
        number = r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)"
        groups = (
            (quartic, "kHz", {"Delta_J", "Delta_JK", "Delta_K", "delta_J", "delta_K"}),
            (
                sextic,
                "Hz",
                {"Phi_J", "Phi_JK", "Phi_KJ", "Phi_K", "phi_J", "phi_JK", "phi_K"},
            ),
        )
        for model, unit, a_names in groups:
            reductions = set()
            for field in type(model).model_fields:
                if not field.endswith("_" + unit):
                    continue
                parameter = field[: -(len(unit) + 1)]
                spelling = re.escape(parameter)
                if parameter.startswith("phi_"):
                    spelling = re.escape("phi_") + "".join(
                        "[" + item + item.lower() + "]" for item in parameter[4:]
                    )
                matches = list(
                    re.finditer(
                        r"\b"
                        + spelling
                        + r"\b\s*[=:]\s*"
                        + number
                        + r"\s*"
                        + re.escape(unit)
                        + r"\b",
                        text,
                    )
                )
                if matches:
                    value = matches[-1].group(1).replace("D", "E").replace("d", "e")
                    setattr(model, field, float(value))
                    reductions.add("A" if parameter in a_names else "S")
            model.reduction_type = (
                next(iter(reductions))
                if len(reductions) == 1
                else ("mixed" if reductions else None)
            )
        return quartic, sextic

    @classmethod
    def parse_dipole_moment(cls, text: str) -> CFOURDipoleMoment:
        """Extracts electric dipole moments in Debye and a.u."""
        dip = CFOURDipoleMoment()

        # Match principal axis dipole components
        m_dip = re.search(
            r"Dipole\s+moment\s*\(Debye\)\s*:\s*mu_a\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])\s*mu_b\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])\s*mu_c\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.])\s*(?:Total\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)(?![\w.]))?",
            text,
            re.IGNORECASE,
        )
        if m_dip:
            dip.mu_a_debye = float(m_dip.group(1).replace("D", "E").replace("d", "e"))
            dip.mu_b_debye = float(m_dip.group(2).replace("D", "E").replace("d", "e"))
            dip.mu_c_debye = float(m_dip.group(3).replace("D", "E").replace("d", "e"))
            dip.mu_total_debye = (
                float(m_dip.group(4).replace("D", "E").replace("d", "e"))
                if m_dip.group(4)
                else math.sqrt(
                    dip.mu_a_debye**2 + dip.mu_b_debye**2 + dip.mu_c_debye**2
                )
            )
            dip.mu_a_au = dip.mu_a_debye / CONSTANTS.AU_TO_DEBYE
            dip.mu_b_au = dip.mu_b_debye / CONSTANTS.AU_TO_DEBYE
            dip.mu_c_au = dip.mu_c_debye / CONSTANTS.AU_TO_DEBYE
            dip.mu_total_au = dip.mu_total_debye / CONSTANTS.AU_TO_DEBYE

        return dip

    @classmethod
    def parse_quadrupole_coupling(
        cls,
        text: str,
        nuclear_q_mbarn: Optional[Dict[str, float]] = None,
        isotope_by_atom: Optional[Dict[int, int]] = None,
        nuclear_q_source: Optional[str] = None,
    ) -> List[CFOURNuclearQuadrupole]:
        """Retain EFGs; convert only with explicit isotope-specific Q and source.

        ``isotope_by_atom`` uses one-based output atom numbers. Q keys are
        isotope labels such as ``14N`` or ``35Cl``, not bare element names.
        Tensor axes are not assumed to equal rotational principal axes.
        """
        if nuclear_q_mbarn and (not isotope_by_atom or not nuclear_q_source):
            raise ValueError(
                "Quadrupole conversion requires isotope_by_atom and nuclear_q_source."
            )
        q_constants = {
            key.upper(): value for key, value in (nuclear_q_mbarn or {}).items()
        }
        if any(
            re.fullmatch(r"[1-9]\d*[A-Z][A-Z]?", key) is None for key in q_constants
        ):
            raise ValueError(
                "Nuclear quadrupole moments must use isotope keys, e.g. 14N or 35Cl."
            )
        if any(not math.isfinite(value) for value in q_constants.values()):
            raise ValueError("Nuclear quadrupole moments must be finite.")
        results = []
        pattern = r"Electric\s+field\s+gradient\s+tensor\s+for\s+atom\s+(\d+)\s*\(([A-Za-z]+)\)\s*\(in a\.u\.\):\s*\n(.*?)(?=\n\s*\n|\Z)"
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            tensor = cls._parse_tensor_block(match.group(3))
            if tensor is None:
                continue
            atom_idx, element = int(match.group(1)), match.group(2).upper()
            isotope = (isotope_by_atom or {}).get(atom_idx)
            q_value = (
                q_constants.get(f"{isotope}{element}") if isotope is not None else None
            )
            result = CFOURNuclearQuadrupole(
                atom_index=atom_idx,
                element=element,
                isotope_mass_number=isotope,
                Q_mbarn=q_value,
                nuclear_q_source=nuclear_q_source if q_value is not None else None,
                efg_tensor_au=tensor.tolist(),
            )
            if q_value is not None:
                chi = tensor * q_value * CONSTANTS.EFG_TO_KHZ_FACTOR
                result.chi_tensor_kHz = chi.tolist()
                if np.allclose(chi, chi.T, rtol=0.0, atol=1e-10):
                    values = np.linalg.eigvalsh(chi)
                    values = values[np.argsort(np.abs(values))]
                    result.chi_principal_values_kHz = values.tolist()
                    if values[2] != 0:
                        result.asymmetry_eta = float(
                            abs((values[0] - values[1]) / values[2])
                        )
                # No chi_aa/bb/cc: EFG principal axes are not the rotor axes.
            results.append(result)
        return results

    @staticmethod
    def _parse_tensor_block(block: str) -> Optional[np.ndarray]:
        lines = [line.split() for line in block.strip().splitlines()]
        if len(lines) != 3 or any(len(row) != 3 for row in lines):
            return None
        try:
            tensor = np.array(
                [
                    [float(item.replace("D", "E").replace("d", "e")) for item in row]
                    for row in lines
                ]
            )
        except ValueError:
            return None
        return tensor if np.all(np.isfinite(tensor)) else None

    @classmethod
    def parse_spin_rotation(cls, text: str) -> List[CFOURSpinRotation]:
        """Preserve the output-frame tensor without inventing a rotor-frame map."""
        results = []
        pattern = r"Spin-rotation\s+tensor\s+for\s+atom\s+(\d+)\s*\(([A-Za-z]+)\)\s*\(in kHz\):\s*\n(.*?)(?=\n\s*\n|\Z)"
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            tensor = cls._parse_tensor_block(match.group(3))
            if tensor is not None:
                results.append(
                    CFOURSpinRotation(
                        atom_index=int(match.group(1)),
                        element=match.group(2).upper(),
                        C_tensor_kHz=tensor.tolist(),
                        C_iso_kHz=float(np.trace(tensor) / 3.0),
                    )
                )
        return results

    @classmethod
    def parse_full_output(
        cls, output_content: str, molecule_name: str = ""
    ) -> CFOUROutputPayload:
        """
        Parses a complete CFOUR output string into a structured CFOUROutputPayload object.
        """
        sha256_hash = hashlib.sha256(output_content.encode("utf-8")).hexdigest()
        energies = cls.parse_energies(output_content)
        rot_constants = cls.parse_rotational_constants(output_content)
        vibrational_data = cls.parse_vibrational_data(output_content)
        quartic_dist, sextic_dist = cls.parse_centrifugal_distortion(output_content)
        dipole = cls.parse_dipole_moment(output_content)
        quadrupole = cls.parse_quadrupole_coupling(output_content)
        spin_rot = cls.parse_spin_rotation(output_content)

        failed = bool(
            re.search(
                r"not\s+converged|\bERROR\b|\bFATAL\b", output_content, re.IGNORECASE
            )
        )
        # Absence of an error is not positive, calculation-specific convergence
        # evidence. No genuine release fixture has qualified that parser yet.
        converged = False if failed else None

        payload = CFOUROutputPayload(
            molecule_name=molecule_name,
            energies=energies,
            rotational_constants=rot_constants,
            quartic_distortion=quartic_dist,
            sextic_distortion=sextic_dist,
            vibrational_data=vibrational_data,
            dipole_moment=dipole,
            quadrupole_coupling=quadrupole,
            spin_rotation=spin_rot,
            raw_output_sha256=sha256_hash,
            calculation_converged=converged,
            validation_issues=[
                "Calculation-specific convergence and effective method/basis parsing are not yet qualified."
            ],
        )
        return payload


# ============================================================================
# 6. CFOUR Execution Engine & Chained Restart Manager
# ============================================================================


class TorqCfourExecutor:
    """
    Orchestrates CFOUR job execution, scratch workspace isolation,
    binary execution, restart file propagation, and output parsing.
    Direct run interface. Generic gradient execution is explicitly unsupported
    until a genuine CFOUR gradient parser and adapter have been qualified.
    """

    def __init__(self, cfour_path: Optional[str] = None) -> None:
        self.cfour_path = cfour_path

    def resolve_binary(self) -> Path:
        """Resolve an explicit/site CFOUR binary without requiring CoChem-BASE."""
        candidate = (
            self.cfour_path or os.environ.get("CFOUR_PATH") or shutil.which("xcfour")
        )
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if path.is_file() and os.access(path, os.X_OK):
                return path
        raise CFOURUnavailableError(
            "CFOUR executable (xcfour) is not configured or executable."
        )

    def execute(self, job_spec: Any) -> Any:
        """Fail before launch: this bridge has no qualified gradient adapter."""
        raise CFOURUnavailableError(
            "Generic CFOUR gradient/Hessian execution is unavailable: a frequency "
            "vector is not a Hessian, and an empty gradient is not a result. "
            "Use a separately qualified derivative adapter."
        )

    def run_cfour_job(
        self,
        job_name: str,
        symbols: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        method: str,
        basis: str,
        reference: str = "RHF",
        frozen_core: bool = True,
        anharm: str = "VPT2",
        vib: str = "EXACT",
        props: str = "FIRST_ORDER",
        memory_gb: int = 32,
        optimize: bool = False,
        extra_keywords: Optional[Dict[str, Any]] = None,
        isotopes: Optional[Sequence[int]] = None,
        scratch_dir: Optional[Union[str, Path]] = None,
        genbas_path: Optional[Union[str, Path]] = None,
        restart_files_dir: Optional[Union[str, Path]] = None,
        timeout_seconds: int = 86400,
    ) -> Tuple[CFOUROutputPayload, Path]:
        """
        Executes a CFOUR job in an isolated scratch directory and returns parsed results.
        """
        bin_path = self.resolve_binary()

        if scratch_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="torq_cfour_"))
        else:
            work_dir = Path(scratch_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate ZMAT input file
        zmat_content = CFOURZmatBuilder.generate_full_zmat_input(
            symbols=symbols,
            coordinates=coordinates,
            method=method,
            basis=basis,
            reference=reference,
            frozen_core=frozen_core,
            anharm=anharm,
            vib=vib,
            props=props,
            memory_gb=memory_gb,
            optimize=optimize,
            extra_keywords=extra_keywords,
            isotopes=isotopes,
            title=f"CFOUR {job_name} Execution",
        )

        zmat_file = work_dir / "ZMAT"
        zmat_file.write_text(zmat_content, encoding="utf-8")

        # 2. Provision GENBAS file if available
        if genbas_path and Path(genbas_path).exists():
            shutil.copy2(str(genbas_path), str(work_dir / "GENBAS"))
        elif (Path(ARTIFACTS_DIR) / "GENBAS").exists():
            shutil.copy2(str(Path(ARTIFACTS_DIR) / "GENBAS"), str(work_dir / "GENBAS"))

        # 3. Provision restart files (JOBARC, JAINDX, OPTARC, MOINTS, MOABCD, FCMINT)
        if restart_files_dir and Path(restart_files_dir).exists():
            for rfile in [
                "JOBARC",
                "JAINDX",
                "OPTARC",
                "MOINTS",
                "MOABCD",
                "FCMINT",
                "FCMFINAL",
            ]:
                src = Path(restart_files_dir) / rfile
                if src.exists():
                    shutil.copy2(str(src), str(work_dir / rfile))

        # 4. Execute xcfour
        output_file = work_dir / f"{job_name}.out"
        start_time = time.perf_counter()

        try:
            logger.info(f"Launching CFOUR execution '{job_name}' in {work_dir}...")
            with open(output_file, "w", encoding="utf-8") as out_f:
                res = subprocess.Popen(
                    [str(bin_path)],
                    cwd=str(work_dir),
                    stdout=out_f,
                    stderr=subprocess.STDOUT,
                )
                _ACTIVE_CFOUR_PROCESSES.add(res)
                try:
                    res.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    res.kill()
                    res.wait()
                    raise
                finally:
                    _ACTIVE_CFOUR_PROCESSES.discard(res)
            wall_time = time.perf_counter() - start_time

            if res.returncode != 0:
                logger.warning(f"CFOUR exited with return code {res.returncode}.")

        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"CFOUR job '{job_name}' exceeded execution timeout of {timeout_seconds}s."
            )
        except Exception as e:
            logger.error(f"Execution error running CFOUR: {e}")
            raise

        # 5. Parse output
        output_content = output_file.read_text(encoding="utf-8", errors="ignore")
        payload = CFOUROutputParser.parse_full_output(
            output_content, molecule_name=job_name
        )
        payload.wall_clock_seconds = wall_time
        payload.process_returncode = res.returncode
        payload.requested_method = method
        payload.requested_basis = basis
        if res.returncode != 0:
            payload.calculation_converged = False
            payload.validation_issues.append(
                f"CFOUR process exited with code {res.returncode}."
            )

        return payload, work_dir


# ============================================================================
# 7. Bridge Integration: SPCAT Payload & HDF5 Database Export
# ============================================================================


def export_cfour_to_spcat_dict(payload: CFOUROutputPayload) -> Dict[str, Any]:
    """
    Formats parsed CFOUR spectroscopic constants into a structured dictionary
    directly consumable by cochem_spcat_bridge.route_3tier_abinitio_payload.
    """
    if payload.calculation_converged is not True:
        raise ValueError(
            "SPCAT export requires positively verified scientific convergence."
        )
    if not payload.calc_method or not payload.basis_set:
        raise ValueError(
            "SPCAT export requires the verified effective method and basis."
        )
    rc = payload.rotational_constants
    constants = (rc.A0_MHz, rc.B0_MHz, rc.C0_MHz)
    dipoles = (
        payload.dipole_moment.mu_a_debye,
        payload.dipole_moment.mu_b_debye,
        payload.dipole_moment.mu_c_debye,
    )
    if any(value is None for value in constants):
        raise ValueError(
            "Ground-state SPCAT export requires A0, B0 and C0; Be is not a substitute."
        )
    if any(value is None for value in dipoles):
        raise ValueError(
            "Intensity export requires measured/computed principal-axis dipole components."
        )
    if any(value <= 0 for value in constants):
        raise ValueError(
            "This SPCAT profile requires three positive nonlinear-rotor constants."
        )
    return {
        "energy_hartree": payload.energies.final_energy_hartree,
        "eccsd_t": payload.energies.ccsd_t_energy_hartree,
        "rotational_constants_kind": "vibrational_ground_state",
        "rotational_constants_mhz": dict(zip(("A", "B", "C"), constants)),
        "frequencies": payload.vibrational_data.harmonic_frequencies_cm1,
        "anharmonic_frequencies": payload.vibrational_data.anharmonic_frequencies_cm1,
        "dipoles": dict(zip(("mu_a", "mu_b", "mu_c"), dipoles)),
        "quartic_distortion_khz": {
            key.removesuffix("_kHz"): value
            for key, value in payload.quartic_distortion.model_dump().items()
            if key.endswith("_kHz")
        },
        "quartic_reduction": payload.quartic_distortion.reduction_type,
        "sextic_distortion_hz": {
            key.removesuffix("_Hz"): value
            for key, value in payload.sextic_distortion.model_dump().items()
            if key.endswith("_Hz")
        },
        "sextic_reduction": payload.sextic_distortion.reduction_type,
        "nuclear_quadrupole": [
            item.model_dump() for item in payload.quadrupole_coupling
        ],
        "spin_rotation": [item.model_dump() for item in payload.spin_rotation],
        "sha256": payload.raw_output_sha256,
        "converged": payload.calculation_converged,
        "validation_issues": payload.validation_issues,
    }


def save_cfour_to_hdf5(
    payload: CFOUROutputPayload,
    h5_file_path: Union[str, Path],
    dataset_group: str = "ab_initio/cfour",
) -> None:
    """Archive a complete typed payload, retaining JSON nulls for unknown data.

    This ordinary HDF5 writer is not a SWMR/transaction coordinator. An archive
    group is immutable to prevent mixing a partial attempt with older results.
    """
    h5_path = Path(h5_file_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "a") as handle:
        if dataset_group in handle:
            raise ValueError(
                "CFOUR archive groups are immutable; select a new attempt group."
            )
        group = handle.create_group(dataset_group)
        group.attrs["schema_version"] = "2"
        group.attrs["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        group.attrs["sha256_provenance"] = payload.raw_output_sha256
        # Persist nulls/status/provenance even where HDF5 cannot represent a
        # missing scalar attribute. No unknown quantity becomes numeric zero.
        group.create_dataset(
            "payload_json",
            data=payload.model_dump_json(),
            dtype=h5py.string_dtype("utf-8"),
        )
        for model in (
            payload.energies,
            payload.rotational_constants,
            payload.quartic_distortion,
            payload.sextic_distortion,
            payload.dipole_moment,
        ):
            for key, value in model.model_dump().items():
                if value is not None and isinstance(value, (int, float)):
                    group.attrs[key] = value
        for field in ("harmonic_frequencies_cm1", "anharmonic_frequencies_cm1"):
            value = getattr(payload.vibrational_data, field)
            if value is not None:
                group.create_dataset(field, data=np.asarray(value, dtype=np.float64))
        handle.flush()


# ============================================================================
# 8. Parallel Queue-Split Decomposition Manager
# ============================================================================


class CFOURDecompositionManager:
    """
    Manages queue-split parallel decomposition of high-level VPT2 jobs across
    irreducible representations (FD_IRREP) and assembly via CFOUR post-processors.
    (Method Matrix Section 8B.6 & 9.4).
    """

    @classmethod
    def generate_irrep_subjob_input(
        cls,
        base_zmat: str,
        irrep_index: int,
        freq_parallel: bool = True,
    ) -> str:
        """
        Injects FD_IRREP and parallel VPT2 flags for multi-job queue execution.
        """
        extra_lines = [
            f"FD_IRREP={irrep_index}",
            "FD_PROJECT=OFF",
        ]
        if freq_parallel:
            extra_lines.extend(["FREQ_ALGORITHM=PARALLEL", "ANH_ALGORITHM=PARALLEL"])

        insertion = "\n".join(extra_lines)
        if re.search(
            r"(\*CFOUR\(.*?)(\n\s*\))", base_zmat, flags=re.DOTALL | re.IGNORECASE
        ):
            modified_zmat = re.sub(
                r"(\*CFOUR\(.*?)(\n\s*\))",
                rf"\1\n{insertion}\2",
                base_zmat,
                count=1,
                flags=re.DOTALL | re.IGNORECASE,
            )
        else:
            modified_zmat = re.sub(r"(?m)^\s*\)", f"{insertion}\n)", base_zmat, count=1)
        return modified_zmat

    @classmethod
    def get_assembly_command_chain(cls) -> List[str]:
        """
        Returns the canonical pipeline of commands required to assemble decomposed
        force constants into the final force field: xjoda -> xsymcor -> xja2fja -> xcubic.
        """
        return ["xjoda", "xsymcor", "xja2fja", "xcubic"]


# ============================================================================
# Exported Public Interface
# ============================================================================

__all__ = [
    "CFOURUnavailableError",
    "CFOURPhysicalConstants",
    "CONSTANTS",
    "get_atomic_mass",
    "get_isotopic_mass",
    "get_atomic_number",
    "compute_center_of_mass",
    "compute_inertia_tensor",
    "CFOURRotationalConstants",
    "CFOURQuarticDistortion",
    "CFOURSexticDistortion",
    "CFOURVibrationalData",
    "CFOURDipoleMoment",
    "CFOURNuclearQuadrupole",
    "CFOURSpinRotation",
    "CFOUREnergies",
    "CFOUROutputPayload",
    "CFOURZmatBuilder",
    "CFOUROutputParser",
    "TorqCfourExecutor",
    "export_cfour_to_spcat_dict",
    "save_cfour_to_hdf5",
    "CFOURDecompositionManager",
    "cleanup_zombies",
]


if __name__ == "__main__":
    logger.info("Initializing CoChem-TORQ CFOUR Bridge Diagnostic Verification...")

    # 1. Test Dynamic Mendeleev Mass Retrieval
    c_m = get_atomic_mass("C")
    n_m = get_atomic_mass("N")
    o_m = get_atomic_mass("O")
    logger.info(f"Dynamic Mendeleev Atomic Weights: C={c_m}, N={n_m}, O={o_m}")

    # 2. Test Z-Matrix Generation for Water and Linear Ar-HCN
    w_syms = ["O", "H", "H"]
    w_coords = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
    zmat_w, _ = CFOURZmatBuilder.cartesian_to_zmat(w_syms, w_coords)
    logger.info("Generated Water ZMAT:\n" + zmat_w)

    lin_syms = ["Ar", "H", "C", "N"]
    lin_coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 4.065],
        [0.0, 0.0, 5.221],
    ]
    zmat_lin, _ = CFOURZmatBuilder.cartesian_to_zmat(lin_syms, lin_coords)
    logger.info("Generated Linear Dummy-Atom ZMAT:\n" + zmat_lin)

    # 3. Test Full Deck Generation & Decomposition
    deck = CFOURZmatBuilder.generate_full_zmat_input(
        symbols=w_syms,
        coordinates=w_coords,
        method="CCSD(T)",
        basis="ANO1",
        anharm="VPT2",
        props="FIRST_ORDER",
        isotopes=[16, 1, 1],
    )
    subjob = CFOURDecompositionManager.generate_irrep_subjob_input(deck, irrep_index=1)
    assert "CALC=CCSD(T)" in subjob and "FD_IRREP=1" in subjob
    logger.info("CFOUR Full Deck and Irrep Decomposition verified successfully.")

    logger.info("CoChem-TORQ CFOUR Bridge self-check complete.")
