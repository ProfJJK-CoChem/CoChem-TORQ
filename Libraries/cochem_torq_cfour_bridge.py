"""
CoChem-TORQ: CFOUR Coupled-Cluster Analytic Hessian & Sextic Distortion Bridge
=============================================================================
Phase 5 (Stage 4.5) Authoritative Quantum Engine Bridge
-------------------------------------------------------
Governs CFOUR (Coupled-Cluster techniques for Computational Chemistry) execution,
Z-matrix internal coordinate generation with dummy-atom (X) singularity protection
for linear/quasi-linear angles, coupled-cluster analytic second derivatives,
VPT2 anharmonic force fields, Watson A/S-reduction quartic and sextic centrifugal
distortion calculations, nuclear quadrupole coupling (EFG conversion), nuclear
spin-rotation tensors, DBOC, relativistic corrections, and Pickett SPCAT / HDF5
provenance integration.

Authoritative Standards & Method Matrix v4/v5 Compliance:
- Method Matrix (§9, §13, §14; Tables 3-C, 4-C, 6-C, 8-C)
- Mendeleev Mandate: Dynamic atomic and isotopic mass lookups via mendeleev library.
- CODATA 2022 Exact Physical Constants
- Tripartite Filesystem Air-Gap Compliance & SHA-256 Cryptographic Provenance
- Anti-Spoofing & Physical Execution Protocol Mandate: Physical execution and analytical validation.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import psutil
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cochem_base.interfaces import ElectronicStructureExecutor
from cochem_base.schemas import GradientPayload, QuantumJobSpec
from cochem_base.environment import BinaryRegistry, PathRegistry
from cochem_base.exceptions import BinaryNotFoundError

try:
    from mendeleev import element as get_mendeleev_element
    _MENDELEEV_AVAILABLE = True
except ImportError:
    _MENDELEEV_AVAILABLE = False


# ============================================================================
# 0. Subprocess Lifecycle Management & Zombie Cleanup
# ============================================================================

def cleanup_zombies() -> None:
    """
    Terminates orphaned CFOUR subprocesses to prevent host resource exhaustion.
    Scans for xcfour, cfour, xncc, xjoda, xvpt2, xsymcor, xja2fja, xcubic, xbcktrn.
    """
    cfour_binaries = {
        "xcfour", "cfour", "xncc", "xjoda", "xvpt2", "xsymcor",
        "xja2fja", "xcubic", "xbcktrn", "xdvdol", "xvprops"
    }
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pname = str(proc.info["name"]).lower()
            if any(b in pname for b in cfour_binaries):
                for child in proc.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

atexit.register(cleanup_zombies)

# Module-level logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-CFOUR] %(message)s")
logger = logging.getLogger("TorqCfourBridge")

CFOUR_PATH = os.environ.get("CFOUR_PATH", "xcfour")
ARTIFACTS_DIR = os.environ.get("COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts"))


# ============================================================================
# 1. Fundamental Constants (CODATA 2022) & Dynamic Mendeleev Retrievals
# ============================================================================

@dataclass(frozen=True)
class CFOURPhysicalConstants:
    """CODATA 2022 Fundamental Constants for Quantum Chemistry and Spectroscopy."""
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
    return get_atomic_mass(symbol)


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
    symbols: Sequence[str],
    coordinates: np.ndarray
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
    symbols: Sequence[str],
    coordinates: np.ndarray
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

    # Calculate rotational constants A >= B >= C in MHz
    rot_constants_mhz = np.full(3, 0.0, dtype=np.float64)
    for k in range(3):
        if sorted_moments[k] > 1e-6:
            rot_constants_mhz[k] = CONSTANTS.INERTIA_TO_MHZ_FACTOR / sorted_moments[k]
        else:
            rot_constants_mhz[k] = 0.0

    return sorted_moments, sorted_axes, rot_constants_mhz


# ============================================================================
# 3. Pydantic Structured Data Models
# ============================================================================

class CFOURRotationalConstants(BaseModel):
    """Equilibrium and vibrationally corrected rotational constants."""
    model_config = ConfigDict(extra="ignore")

    Ae_MHz: float = 0.0
    Be_MHz: float = 0.0
    Ce_MHz: float = 0.0
    Ae_cm1: float = 0.0
    Be_cm1: float = 0.0
    Ce_cm1: float = 0.0

    A0_MHz: float = 0.0
    B0_MHz: float = 0.0
    C0_MHz: float = 0.0
    A0_cm1: float = 0.0
    B0_cm1: float = 0.0
    C0_cm1: float = 0.0

    alpha_A_MHz: List[float] = Field(default_factory=list)
    alpha_B_MHz: List[float] = Field(default_factory=list)
    alpha_C_MHz: List[float] = Field(default_factory=list)

    alpha_A_cm1: List[float] = Field(default_factory=list)
    alpha_B_cm1: List[float] = Field(default_factory=list)
    alpha_C_cm1: List[float] = Field(default_factory=list)

    inertial_defect_amu_angstrom2: float = 0.0
    asymmetry_kappa: float = 0.0


class CFOURQuarticDistortion(BaseModel):
    """Watson A-reduction and S-reduction quartic centrifugal distortion constants."""
    model_config = ConfigDict(extra="ignore")

    # Watson A-reduction parameters (kHz)
    Delta_J_kHz: float = 0.0
    Delta_JK_kHz: float = 0.0
    Delta_K_kHz: float = 0.0
    delta_J_kHz: float = 0.0
    delta_K_kHz: float = 0.0

    # Watson S-reduction parameters (kHz)
    D_J_kHz: float = 0.0
    D_JK_kHz: float = 0.0
    D_K_kHz: float = 0.0
    d_1_kHz: float = 0.0
    d_2_kHz: float = 0.0

    reduction_type: str = "A"


class CFOURSexticDistortion(BaseModel):
    """
    Watson A-reduction and S-reduction sextic centrifugal distortion constants.
    CFOUR unique capability (Method Matrix Section 9.3 & 14.1, Table 6-C).
    """
    model_config = ConfigDict(extra="ignore")

    # Watson A-reduction parameters (Hz)
    Phi_J_Hz: float = 0.0
    Phi_JK_Hz: float = 0.0
    Phi_KJ_Hz: float = 0.0
    Phi_K_Hz: float = 0.0
    phi_J_Hz: float = 0.0
    phi_JK_Hz: float = 0.0
    phi_K_Hz: float = 0.0

    # Watson S-reduction parameters (Hz)
    H_J_Hz: float = 0.0
    H_JK_Hz: float = 0.0
    H_KJ_Hz: float = 0.0
    H_K_Hz: float = 0.0
    h_1_Hz: float = 0.0
    h_2_Hz: float = 0.0
    h_3_Hz: float = 0.0

    reduction_type: str = "A"


class CFOURVibrationalData(BaseModel):
    """Harmonic and VPT2 anharmonic vibrational frequencies and force fields."""
    model_config = ConfigDict(extra="ignore")

    harmonic_frequencies_cm1: List[float] = Field(default_factory=list)
    anharmonic_frequencies_cm1: List[float] = Field(default_factory=list)
    ir_intensities_km_mol: List[float] = Field(default_factory=list)
    symmetry_irreps: List[str] = Field(default_factory=list)
    harmonic_zpe_kcal_mol: float = 0.0
    anharmonic_zpe_kcal_mol: float = 0.0
    x_matrix_cm1: List[List[float]] = Field(default_factory=list)
    cubic_force_constants_cm1: Dict[str, float] = Field(default_factory=dict)
    semidiagonal_quartic_cm1: Dict[str, float] = Field(default_factory=dict)


class CFOURDipoleMoment(BaseModel):
    """Electric dipole moment in principal axis frame and Cartesian components."""
    model_config = ConfigDict(extra="ignore")

    mu_a_debye: float = 0.0
    mu_b_debye: float = 0.0
    mu_c_debye: float = 0.0
    mu_total_debye: float = 0.0

    mu_a_au: float = 0.0
    mu_b_au: float = 0.0
    mu_c_au: float = 0.0
    mu_total_au: float = 0.0


class CFOURNuclearQuadrupole(BaseModel):
    """Electric Field Gradient (EFG) and Nuclear Quadrupole Coupling Tensor."""
    model_config = ConfigDict(extra="ignore")

    atom_index: int = 0
    element: str = ""
    Q_mbarn: float = 0.0
    efg_tensor_au: List[List[float]] = Field(default_factory=lambda: [[0.0]*3 for _ in range(3)])
    chi_tensor_kHz: List[List[float]] = Field(default_factory=lambda: [[0.0]*3 for _ in range(3)])
    chi_aa_kHz: float = 0.0
    chi_bb_kHz: float = 0.0
    chi_cc_kHz: float = 0.0
    asymmetry_eta: float = 0.0


class CFOURSpinRotation(BaseModel):
    """Nuclear spin-rotation coupling tensor (SPINROT=ON)."""
    model_config = ConfigDict(extra="ignore")

    atom_index: int = 0
    element: str = ""
    C_tensor_kHz: List[List[float]] = Field(default_factory=lambda: [[0.0]*3 for _ in range(3)])
    C_aa_kHz: float = 0.0
    C_bb_kHz: float = 0.0
    C_cc_kHz: float = 0.0
    C_iso_kHz: float = 0.0


class CFOUREnergies(BaseModel):
    """Electronic energies and corrections in Hartree."""
    model_config = ConfigDict(extra="ignore")

    scf_energy_hartree: float = 0.0
    mp2_energy_hartree: Optional[float] = None
    ccsd_energy_hartree: Optional[float] = None
    ccsd_t_energy_hartree: Optional[float] = None
    final_energy_hartree: float = 0.0
    correlation_energy_hartree: Optional[float] = None
    dboc_correction_hartree: Optional[float] = None
    relativistic_correction_hartree: Optional[float] = None


class CFOUROutputPayload(BaseModel):
    """Master structured container for complete CFOUR calculation results."""
    model_config = ConfigDict(extra="ignore")

    molecule_name: str = ""
    calc_method: str = "CCSD(T)"
    basis_set: str = "ANO1"
    energies: CFOUREnergies = Field(default_factory=CFOUREnergies)
    rotational_constants: CFOURRotationalConstants = Field(default_factory=CFOURRotationalConstants)
    quartic_distortion: CFOURQuarticDistortion = Field(default_factory=CFOURQuarticDistortion)
    sextic_distortion: CFOURSexticDistortion = Field(default_factory=CFOURSexticDistortion)
    vibrational_data: CFOURVibrationalData = Field(default_factory=CFOURVibrationalData)
    dipole_moment: CFOURDipoleMoment = Field(default_factory=CFOURDipoleMoment)
    quadrupole_coupling: List[CFOURNuclearQuadrupole] = Field(default_factory=list)
    spin_rotation: List[CFOURSpinRotation] = Field(default_factory=list)
    optimized_coordinates: List[List[Union[str, float]]] = Field(default_factory=list)
    raw_output_sha256: str = ""
    calculation_converged: bool = True
    wall_clock_seconds: float = 0.0


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
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        cosine = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _compute_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
        """Computes dihedral angle p1-p2-p3-p4 in degrees [-180, 180]."""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)
        norm_b2 = np.linalg.norm(b2)

        if norm_n1 < 1e-8 or norm_n2 < 1e-8 or norm_b2 < 1e-8:
            return 0.0

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

        coords = [np.asarray(c, dtype=np.float64) for c in coordinates]
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
                ang = cls._compute_angle(curr_pos, placed_positions[1], placed_positions[0])
                if ang < linear_angle_threshold_deg or ang > (180.0 - linear_angle_threshold_deg):
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
                    ang = cls._compute_angle(curr_pos, placed_positions[1], placed_positions[0])

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
                ang = cls._compute_angle(curr_pos, placed_positions[ref1 - 1], placed_positions[ref2 - 1])
                dih = cls._compute_dihedral(curr_pos, placed_positions[ref1 - 1], placed_positions[ref2 - 1], placed_positions[ref3 - 1])

                var_r_count += 1
                r_name = f"R{var_r_count}"
                var_a_count += 1
                a_name = f"A{var_a_count}"
                var_d_count += 1
                d_name = f"D{var_d_count}"

                zmat_lines.append(f"{sym} {ref1} {r_name} {ref2} {a_name} {ref3} {d_name}")
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
        method: str = "CCSD(T)",
        basis: str = "ANO1",
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
            symbols=symbols,
            coordinates=coordinates,
            title=title,
            optimize_all=optimize
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
        """Extracts electronic energies and corrections."""
        energies = CFOUREnergies()

        # SCF Energy
        m_scf = re.search(r"E\(SCF\)\s*=\s*([-\d\.]+)\s*a\.u\.", text) or \
                re.search(r"Total Energy\s*=\s*([-\d\.]+)\s*a\.u\.", text) or \
                re.search(r"Reference energy\s+is\s+([-\d\.]+)", text)
        if m_scf:
            energies.scf_energy_hartree = float(m_scf.group(1))

        # MP2 Energy
        m_mp2 = re.search(r"Total MP2 energy\s*:\s*([-\d\.]+)", text) or \
                re.search(r"E\(MP2\)\s*=\s*([-\d\.]+)", text)
        if m_mp2:
            energies.mp2_energy_hartree = float(m_mp2.group(1))

        # CCSD Energy
        m_ccsd = re.search(r"Total CCSD energy\s*:\s*([-\d\.]+)", text) or \
                 re.search(r"E\(CCSD\)\s*=\s*([-\d\.]+)", text)
        if m_ccsd:
            energies.ccsd_energy_hartree = float(m_ccsd.group(1))

        # CCSD(T) Energy
        m_ccsdt = re.search(r"Total CCSD\(T\) energy\s*:\s*([-\d\.]+)", text) or \
                  re.search(r"E\(CCSD\(T\)\)\s*=\s*([-\d\.]+)", text) or \
                  re.search(r"Total energy for calculation type CCSD\(T\)\s*:\s*([-\d\.]+)", text)
        if m_ccsdt:
            energies.ccsd_t_energy_hartree = float(m_ccsdt.group(1))
            energies.final_energy_hartree = energies.ccsd_t_energy_hartree
        elif energies.ccsd_energy_hartree is not None:
            energies.final_energy_hartree = energies.ccsd_energy_hartree
        elif energies.mp2_energy_hartree is not None:
            energies.final_energy_hartree = energies.mp2_energy_hartree
        else:
            energies.final_energy_hartree = energies.scf_energy_hartree

        # DBOC Correction
        m_dboc = re.search(r"Total DBOC\s*=\s*([-\d\.]+)\s*a\.u\.", text) or \
                 re.search(r"DBOC energy correction\s*:\s*([-\d\.]+)", text)
        if m_dboc:
            energies.dboc_correction_hartree = float(m_dboc.group(1))

        # Relativistic Correction
        m_rel = re.search(r"Relativistic correction\s*=\s*([-\d\.]+)\s*a\.u\.", text) or \
                re.search(r"Total Relativistic Energy\s*:\s*([-\d\.]+)", text)
        if m_rel:
            energies.relativistic_correction_hartree = float(m_rel.group(1))

        return energies

    @classmethod
    def parse_rotational_constants(cls, text: str) -> CFOURRotationalConstants:
        """Extracts equilibrium Ae, Be, Ce and effective ground state A0, B0, C0."""
        rc = CFOURRotationalConstants()

        # Equilibrium constants (cm^-1 or MHz)
        m_rc_cm = re.search(r"Rotational constants\s*\(in cm-1\):\s*A\s*=\s*([\d\.]+)\s*B\s*=\s*([\d\.]+)\s*C\s*=\s*([\d\.]+)", text, re.IGNORECASE)
        m_rc_mhz = re.search(r"Rotational constants\s*\(in MHz\):\s*A\s*=\s*([\d\.]+)\s*B\s*=\s*([\d\.]+)\s*C\s*=\s*([\d\.]+)", text, re.IGNORECASE)

        if m_rc_mhz:
            rc.Ae_MHz = float(m_rc_mhz.group(1))
            rc.Be_MHz = float(m_rc_mhz.group(2))
            rc.Ce_MHz = float(m_rc_mhz.group(3))
            rc.Ae_cm1 = rc.Ae_MHz * CONSTANTS.MHZ_TO_CM1
            rc.Be_cm1 = rc.Be_MHz * CONSTANTS.MHZ_TO_CM1
            rc.Ce_cm1 = rc.Ce_MHz * CONSTANTS.MHZ_TO_CM1
        elif m_rc_cm:
            rc.Ae_cm1 = float(m_rc_cm.group(1))
            rc.Be_cm1 = float(m_rc_cm.group(2))
            rc.Ce_cm1 = float(m_rc_cm.group(3))
            rc.Ae_MHz = rc.Ae_cm1 * CONSTANTS.CM1_TO_MHZ
            rc.Be_MHz = rc.Be_cm1 * CONSTANTS.CM1_TO_MHZ
            rc.Ce_MHz = rc.Ce_cm1 * CONSTANTS.CM1_TO_MHZ

        # Ground state rotational constants (A0, B0, C0) and vibration-rotation shifts
        m_shift_sec = re.search(r"Be,\s*B0\s*AND\s*B-B0\s*SHIFTS\s*FOR\s*SINGLY\s*EXCITED\s*VIBRATIONAL\s*STATES.*?\n(.*?)(?=\n\n|\n[A-Z]|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m_shift_sec:
            lines = m_shift_sec.group(1).strip().splitlines()
            for line in lines:
                if "Ground State" in line or "State 0" in line:
                    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
                    if len(nums) >= 3:
                        rc.A0_cm1 = nums[0]
                        rc.B0_cm1 = nums[1]
                        rc.C0_cm1 = nums[2]
                        rc.A0_MHz = rc.A0_cm1 * CONSTANTS.CM1_TO_MHZ
                        rc.B0_MHz = rc.B0_cm1 * CONSTANTS.CM1_TO_MHZ
                        rc.C0_MHz = rc.C0_cm1 * CONSTANTS.CM1_TO_MHZ

        # Vibration-rotation alpha constants (alpha_A, alpha_B, alpha_C)
        m_alpha_sec = re.search(r"Vibration-rotation\s*interaction\s*constants\s*\(in MHz\):\s*\n(.*?)(?=\n\n|\n[A-Z]|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m_alpha_sec:
            for line in m_alpha_sec.group(1).strip().splitlines():
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        a_A = float(parts[1])
                        a_B = float(parts[2])
                        a_C = float(parts[3])
                        rc.alpha_A_MHz.append(a_A)
                        rc.alpha_B_MHz.append(a_B)
                        rc.alpha_C_MHz.append(a_C)
                        rc.alpha_A_cm1.append(a_A * CONSTANTS.MHZ_TO_CM1)
                        rc.alpha_B_cm1.append(a_B * CONSTANTS.MHZ_TO_CM1)
                        rc.alpha_C_cm1.append(a_C * CONSTANTS.MHZ_TO_CM1)
                    except ValueError:
                        continue

        # If A0 was not directly parsed from section, compute B0 = Be - 0.5 * sum(alpha)
        if rc.A0_MHz == 0.0 and len(rc.alpha_A_MHz) > 0 and rc.Ae_MHz > 0.0:
            rc.A0_MHz = rc.Ae_MHz - 0.5 * sum(rc.alpha_A_MHz)
            rc.B0_MHz = rc.Be_MHz - 0.5 * sum(rc.alpha_B_MHz)
            rc.C0_MHz = rc.Ce_MHz - 0.5 * sum(rc.alpha_C_MHz)
            rc.A0_cm1 = rc.A0_MHz * CONSTANTS.MHZ_TO_CM1
            rc.B0_cm1 = rc.B0_MHz * CONSTANTS.MHZ_TO_CM1
            rc.C0_cm1 = rc.C0_MHz * CONSTANTS.MHZ_TO_CM1

        # Ray's asymmetry parameter kappa = (2B - A - C) / (A - C)
        A = rc.A0_MHz if rc.A0_MHz > 0.0 else rc.Ae_MHz
        B = rc.B0_MHz if rc.B0_MHz > 0.0 else rc.Be_MHz
        C = rc.C0_MHz if rc.C0_MHz > 0.0 else rc.Ce_MHz
        if abs(A - C) > 1e-6:
            rc.asymmetry_kappa = float((2.0 * B - A - C) / (A - C))

        return rc

    @classmethod
    def parse_vibrational_data(cls, text: str) -> CFOURVibrationalData:
        """Extracts harmonic frequencies, VPT2 anharmonic fundamentals, and ZPE."""
        vib = CFOURVibrationalData()

        # Harmonic frequencies
        m_harm = re.search(r"Harmonic\s+vibrational\s+frequencies\s*\(cm-1\):\s*\n(.*?)(?=\n\n|\n[A-Z]|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m_harm:
            for line in m_harm.group(1).strip().splitlines():
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                for num in matches:
                    try:
                        val = float(num)
                        if val > 0.1:
                            vib.harmonic_frequencies_cm1.append(val)
                    except ValueError:
                        pass

        # Anharmonic fundamental frequencies
        m_anh = re.search(r"Anharmonic\s+vibrational\s+frequencies\s*\(cm-1\):\s*\n(.*?)(?=\n\n|\n[A-Z]|\Z)", text, re.DOTALL | re.IGNORECASE) or \
                re.search(r"VPT2\s+Fundamentals\s*\(cm-1\):\s*\n(.*?)(?=\n\n|\n[A-Z]|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m_anh:
            for line in m_anh.group(1).strip().splitlines():
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                for num in matches:
                    try:
                        val = float(num)
                        if val > 0.1:
                            vib.anharmonic_frequencies_cm1.append(val)
                    except ValueError:
                        pass

        # Zero-point energy (ZPE)
        m_zpe = re.search(r"Zero\s*point\s*vibrational\s*energy\s*:\s*([\d\.]+)\s*kcal/mol", text, re.IGNORECASE)
        if m_zpe:
            vib.harmonic_zpe_kcal_mol = float(m_zpe.group(1))

        m_azpe = re.search(r"Anharmonic\s*zero\s*point\s*energy\s*:\s*([\d\.]+)\s*kcal/mol", text, re.IGNORECASE)
        if m_azpe:
            vib.anharmonic_zpe_kcal_mol = float(m_azpe.group(1))

        return vib

    @classmethod
    def parse_centrifugal_distortion(
        cls,
        text: str
    ) -> Tuple[CFOURQuarticDistortion, CFOURSexticDistortion]:
        """
        Parses Watson A- and S-reduction quartic and sextic centrifugal distortion constants.
        Enforces case-sensitive parsing to distinguish uppercase (Delta, Phi, D, H)
        from lowercase (delta, phi, d, h) reduction parameters.
        """
        quartic = CFOURQuarticDistortion()
        sextic = CFOURSexticDistortion()

        # Watson A-reduction Quartic Distortion (kHz)
        m_dj = re.search(r"\bDelta_J\b\s*[=:]\s*([-\d\.]+)", text)
        if m_dj:
            quartic.Delta_J_kHz = float(m_dj.group(1))

        m_djk = re.search(r"\bDelta_JK\b\s*[=:]\s*([-\d\.]+)", text)
        if m_djk:
            quartic.Delta_JK_kHz = float(m_djk.group(1))

        m_dk = re.search(r"\bDelta_K\b\s*[=:]\s*([-\d\.]+)", text)
        if m_dk:
            quartic.Delta_K_kHz = float(m_dk.group(1))

        m_delj = re.search(r"\bdelta_[Jj]\b\s*[=:]\s*([-\d\.]+)", text)
        if m_delj:
            quartic.delta_J_kHz = float(m_delj.group(1))

        m_delk = re.search(r"\bdelta_[Kk]\b\s*[=:]\s*([-\d\.]+)", text)
        if m_delk:
            quartic.delta_K_kHz = float(m_delk.group(1))

        # Watson S-reduction Quartic Distortion (kHz)
        m_sdj = re.search(r"\bD_J\b\s*[=:]\s*([-\d\.]+)", text)
        if m_sdj:
            quartic.D_J_kHz = float(m_sdj.group(1))
            quartic.reduction_type = "S"

        m_sdjk = re.search(r"\bD_JK\b\s*[=:]\s*([-\d\.]+)", text)
        if m_sdjk:
            quartic.D_JK_kHz = float(m_sdjk.group(1))

        m_sdk = re.search(r"\bD_K\b\s*[=:]\s*([-\d\.]+)", text)
        if m_sdk:
            quartic.D_K_kHz = float(m_sdk.group(1))

        m_d1 = re.search(r"\bd_?1\b\s*[=:]\s*([-\d\.]+)", text)
        if m_d1:
            quartic.d_1_kHz = float(m_d1.group(1))

        m_d2 = re.search(r"\bd_?2\b\s*[=:]\s*([-\d\.]+)", text)
        if m_d2:
            quartic.d_2_kHz = float(m_d2.group(1))

        # Watson A-reduction Sextic Centrifugal Distortion (Hz)
        m_phij = re.search(r"\bPhi_J\b\s*[=:]\s*([-\d\.]+)", text)
        if m_phij:
            sextic.Phi_J_Hz = float(m_phij.group(1))

        m_phijk = re.search(r"\bPhi_JK\b\s*[=:]\s*([-\d\.]+)", text)
        if m_phijk:
            sextic.Phi_JK_Hz = float(m_phijk.group(1))

        m_phikj = re.search(r"\bPhi_KJ\b\s*[=:]\s*([-\d\.]+)", text)
        if m_phikj:
            sextic.Phi_KJ_Hz = float(m_phikj.group(1))

        m_phik = re.search(r"\bPhi_K\b\s*[=:]\s*([-\d\.]+)", text)
        if m_phik:
            sextic.Phi_K_Hz = float(m_phik.group(1))

        m_sphij = re.search(r"\bphi_[Jj]\b\s*[=:]\s*([-\d\.]+)", text)
        if m_sphij:
            sextic.phi_J_Hz = float(m_sphij.group(1))

        m_sphijk = re.search(r"\bphi_[Jj][Kk]\b\s*[=:]\s*([-\d\.]+)", text)
        if m_sphijk:
            sextic.phi_JK_Hz = float(m_sphijk.group(1))

        m_sphik = re.search(r"\bphi_[Kk]\b\s*[=:]\s*([-\d\.]+)", text)
        if m_sphik:
            sextic.phi_K_Hz = float(m_sphik.group(1))

        # Watson S-reduction Sextic Centrifugal Distortion (Hz)
        m_hj = re.search(r"\bH_J\b\s*[=:]\s*([-\d\.]+)", text)
        if m_hj:
            sextic.H_J_Hz = float(m_hj.group(1))
            sextic.reduction_type = "S"

        m_hjk = re.search(r"\bH_JK\b\s*[=:]\s*([-\d\.]+)", text)
        if m_hjk:
            sextic.H_JK_Hz = float(m_hjk.group(1))

        m_hkj = re.search(r"\bH_KJ\b\s*[=:]\s*([-\d\.]+)", text)
        if m_hkj:
            sextic.H_KJ_Hz = float(m_hkj.group(1))

        m_hk = re.search(r"\bH_K\b\s*[=:]\s*([-\d\.]+)", text)
        if m_hk:
            sextic.H_K_Hz = float(m_hk.group(1))

        m_h1 = re.search(r"\bh_?1\b\s*[=:]\s*([-\d\.]+)", text)
        if m_h1:
            sextic.h_1_Hz = float(m_h1.group(1))

        m_h2 = re.search(r"\bh_?2\b\s*[=:]\s*([-\d\.]+)", text)
        if m_h2:
            sextic.h_2_Hz = float(m_h2.group(1))

        m_h3 = re.search(r"\bh_?3\b\s*[=:]\s*([-\d\.]+)", text)
        if m_h3:
            sextic.h_3_Hz = float(m_h3.group(1))

        return quartic, sextic

    @classmethod
    def parse_dipole_moment(cls, text: str) -> CFOURDipoleMoment:
        """Extracts electric dipole moments in Debye and a.u."""
        dip = CFOURDipoleMoment()

        # Match principal axis dipole components
        m_dip = re.search(r"Dipole\s+moment\s*\(Debye\)\s*:\s*mu_a\s*=\s*([-\d\.]+)\s*mu_b\s*=\s*([-\d\.]+)\s*mu_c\s*=\s*([-\d\.]+)\s*(?:Total\s*=\s*([\d\.]+))?", text, re.IGNORECASE)
        if m_dip:
            dip.mu_a_debye = float(m_dip.group(1))
            dip.mu_b_debye = float(m_dip.group(2))
            dip.mu_c_debye = float(m_dip.group(3))
            dip.mu_total_debye = float(m_dip.group(4)) if m_dip.group(4) else math.sqrt(
                dip.mu_a_debye**2 + dip.mu_b_debye**2 + dip.mu_c_debye**2
            )
            dip.mu_a_au = dip.mu_a_debye / CONSTANTS.AU_TO_DEBYE
            dip.mu_b_au = dip.mu_b_debye / CONSTANTS.AU_TO_DEBYE
            dip.mu_c_au = dip.mu_c_debye / CONSTANTS.AU_TO_DEBYE
            dip.mu_total_au = dip.mu_total_debye / CONSTANTS.AU_TO_DEBYE
        else:
            # Fallback for Cartesian dipole block
            m_cart = re.search(r"Dipole\s+moment\s*:\s*X\s*=\s*([-\d\.]+)\s*Y\s*=\s*([-\d\.]+)\s*Z\s*=\s*([-\d\.]+)\s*Total\s*=\s*([\d\.]+)", text, re.IGNORECASE)
            if m_cart:
                dip.mu_a_debye = float(m_cart.group(1))
                dip.mu_b_debye = float(m_cart.group(2))
                dip.mu_c_debye = float(m_cart.group(3))
                dip.mu_total_debye = float(m_cart.group(4))
                dip.mu_a_au = dip.mu_a_debye / CONSTANTS.AU_TO_DEBYE
                dip.mu_b_au = dip.mu_b_debye / CONSTANTS.AU_TO_DEBYE
                dip.mu_c_au = dip.mu_c_debye / CONSTANTS.AU_TO_DEBYE
                dip.mu_total_au = dip.mu_total_debye / CONSTANTS.AU_TO_DEBYE

        return dip

    @classmethod
    def parse_quadrupole_coupling(
        cls,
        text: str,
        nuclear_q_mbarn: Optional[Dict[str, float]] = None
    ) -> List[CFOURNuclearQuadrupole]:
        """
        Parses Electric Field Gradient (EFG) tensors and converts to nuclear quadrupole coupling constants.
        Conversion formula: chi [kHz] = EFG [a.u.] * Q [mbarn] * 234.96474
        """
        q_constants: Dict[str, float] = {
            "N": 20.44,   # 14N nuclear quadrupole moment in mbarn
            "D": 2.860,   # 2H (D) nuclear quadrupole moment in mbarn
            "CL": -81.65, # 35Cl in mbarn
            "BR": 313.0,  # 79Br in mbarn
            "I": -696.0,  # 127I in mbarn
        }
        if nuclear_q_mbarn:
            q_constants.update(nuclear_q_mbarn)

        results: List[CFOURNuclearQuadrupole] = []

        # Find EFG sections
        efg_matches = re.finditer(r"Electric\s+field\s+gradient\s+tensor\s+for\s+atom\s+(\d+)\s*\(([A-Za-z]+)\)\s*\(in a\.u\.\):\s*\n(.*?)(?=\n\n|\n[A-Z]|\Z)", text, re.DOTALL | re.IGNORECASE)
        for m in efg_matches:
            atom_idx = int(m.group(1))
            sym = m.group(2).upper()
            efg_lines = m.group(3).strip().splitlines()
            efg_mat = []
            for line in efg_lines:
                row = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
                if len(row) >= 3:
                    efg_mat.append(row[:3])

            if len(efg_mat) == 3:
                efg_tensor = np.array(efg_mat, dtype=np.float64)
                q_val = q_constants.get(sym, 0.0)
                chi_tensor = efg_tensor * q_val * CONSTANTS.EFG_TO_KHZ_FACTOR

                # Diagonalize to get principal components
                eigvals = np.linalg.eigvalsh(chi_tensor)
                sorted_idx = np.argsort(np.abs(eigvals))
                chi_xx = float(eigvals[sorted_idx[0]])
                chi_yy = float(eigvals[sorted_idx[1]])
                chi_zz = float(eigvals[sorted_idx[2]])
                eta = float(abs((chi_xx - chi_yy) / chi_zz)) if abs(chi_zz) > 1e-6 else 0.0

                nq = CFOURNuclearQuadrupole(
                    atom_index=atom_idx,
                    element=sym,
                    Q_mbarn=q_val,
                    efg_tensor_au=efg_tensor.tolist(),
                    chi_tensor_kHz=chi_tensor.tolist(),
                    chi_aa_kHz=chi_xx,
                    chi_bb_kHz=chi_yy,
                    chi_cc_kHz=chi_zz,
                    asymmetry_eta=eta
                )
                results.append(nq)

        return results

    @classmethod
    def parse_spin_rotation(cls, text: str) -> List[CFOURSpinRotation]:
        """Extracts nuclear spin-rotation coupling tensors (SPINROT=ON)."""
        results: List[CFOURSpinRotation] = []

        sr_matches = re.finditer(r"Spin-rotation\s+tensor\s+for\s+atom\s+(\d+)\s*\(([A-Za-z]+)\)\s*\(in kHz\):\s*\n(.*?)(?=\n\n|\n[A-Z]|\Z)", text, re.DOTALL | re.IGNORECASE)
        for m in sr_matches:
            atom_idx = int(m.group(1))
            sym = m.group(2).upper()
            lines = m.group(3).strip().splitlines()
            mat = []
            for line in lines:
                row = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
                if len(row) >= 3:
                    mat.append(row[:3])

            if len(mat) == 3:
                tensor = np.array(mat, dtype=np.float64)
                caa = float(tensor[0, 0])
                cbb = float(tensor[1, 1])
                ccc = float(tensor[2, 2])
                ciso = float(np.trace(tensor) / 3.0)

                sr = CFOURSpinRotation(
                    atom_index=atom_idx,
                    element=sym,
                    C_tensor_kHz=tensor.tolist(),
                    C_aa_kHz=caa,
                    C_bb_kHz=cbb,
                    C_cc_kHz=ccc,
                    C_iso_kHz=ciso
                )
                results.append(sr)

        return results

    @classmethod
    def parse_full_output(cls, output_content: str, molecule_name: str = "") -> CFOUROutputPayload:
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

        converged = "The calculation has not converged" not in output_content and \
                    "ERROR" not in output_content

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
            calculation_converged=converged
        )
        return payload


# ============================================================================
# 6. CFOUR Execution Engine & Chained Restart Manager
# ============================================================================

class TorqCfourExecutor(ElectronicStructureExecutor):
    """
    Orchestrates CFOUR job execution, scratch workspace isolation,
    binary execution, restart file propagation, and output parsing.
    Complies with ElectronicStructureExecutor interface.
    """

    def __init__(self, cfour_path: Optional[str] = None) -> None:
        self.cfour_path = cfour_path

    def resolve_binary(self) -> Path:
        """Resolve the CFOUR executable (xcfour) via environment or BinaryRegistry."""
        from cochem_base.environment import BinaryRegistry
        from cochem_base.exceptions import BinaryNotFoundError
        if self.cfour_path and self.cfour_path != "xcfour":
            p = Path(self.cfour_path).resolve()
            if p.is_file():
                return p
        try:
            return BinaryRegistry.resolve("xcfour")
        except BinaryNotFoundError:
            raise BinaryNotFoundError(
                "[MISSING DATA] CFOUR executable (xcfour) not found. "
                "Cannot execute coupled-cluster analytic force fields."
            )

    def execute(self, job_spec: QuantumJobSpec) -> GradientPayload:
        """Standard execution interface for electronic structure calculations."""
        payload, work_dir = self.run_cfour_job(
            job_name=job_spec.job_id,
            symbols=job_spec.symbols,
            coordinates=job_spec.coordinates,
            method=job_spec.method,
            basis=job_spec.basis_set,
        )
        energy = payload.energies.final_energy if payload.energies and payload.energies.final_energy is not None else 0.0
        return GradientPayload(
            energy=energy,
            gradient=[],
            hessian=payload.vibrational_data.harmonic_frequencies if payload.vibrational_data else None,
            status="SUCCESS",
        )

    def run_cfour_job(
        self,
        job_name: str,
        symbols: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        method: str = "CCSD(T)",
        basis: str = "ANO1",
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
            work_dir = PathRegistry.create_scratch_dir(f"cfour_{job_name}")
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
            for rfile in ["JOBARC", "JAINDX", "OPTARC", "MOINTS", "MOABCD", "FCMINT", "FCMFINAL"]:
                src = Path(restart_files_dir) / rfile
                if src.exists():
                    shutil.copy2(str(src), str(work_dir / rfile))

        # 4. Execute xcfour
        output_file = work_dir / f"{job_name}.out"
        start_time = time.perf_counter()

        try:
            logger.info(f"Launching CFOUR execution '{job_name}' in {work_dir}...")
            with open(output_file, "w", encoding="utf-8") as out_f:
                res = subprocess.run(
                    [str(bin_path)],
                    cwd=str(work_dir),
                    stdout=out_f,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False
                )
            wall_time = time.perf_counter() - start_time

            if res.returncode != 0:
                logger.warning(f"CFOUR exited with return code {res.returncode}.")

        except subprocess.TimeoutExpired:
            cleanup_zombies()
            raise TimeoutError(f"CFOUR job '{job_name}' exceeded execution timeout of {timeout_seconds}s.")
        except Exception as e:
            logger.error(f"Execution error running CFOUR: {e}")
            raise

        # 5. Parse output
        output_content = output_file.read_text(encoding="utf-8", errors="ignore")
        payload = CFOUROutputParser.parse_full_output(output_content, molecule_name=job_name)
        payload.wall_clock_seconds = wall_time

        return payload, work_dir


# ============================================================================
# 7. Bridge Integration: SPCAT Payload & HDF5 Database Export
# ============================================================================

def export_cfour_to_spcat_dict(payload: CFOUROutputPayload) -> Dict[str, Any]:
    """
    Formats parsed CFOUR spectroscopic constants into a structured dictionary
    directly consumable by cochem_spcat_bridge.route_3tier_abinitio_payload.
    """
    rc = payload.rotational_constants
    A_mhz = rc.A0_MHz if rc.A0_MHz > 0.0 else rc.Ae_MHz
    B_mhz = rc.B0_MHz if rc.B0_MHz > 0.0 else rc.Be_MHz
    C_mhz = rc.C0_MHz if rc.C0_MHz > 0.0 else rc.Ce_MHz

    return {
        "eccsd_t": payload.energies.final_energy_hartree,
        "energy_hartree": payload.energies.final_energy_hartree,
        "rotational_constants_mhz": {"A": A_mhz, "B": B_mhz, "C": C_mhz},
        "frequencies": payload.vibrational_data.harmonic_frequencies_cm1,
        "anharmonic_frequencies": payload.vibrational_data.anharmonic_frequencies_cm1,
        "dipoles": {
            "mu_a": payload.dipole_moment.mu_a_debye,
            "mu_b": payload.dipole_moment.mu_b_debye,
            "mu_c": payload.dipole_moment.mu_c_debye,
        },
        "quartic_distortion_khz": {
            "Delta_J": payload.quartic_distortion.Delta_J_kHz,
            "Delta_JK": payload.quartic_distortion.Delta_JK_kHz,
            "Delta_K": payload.quartic_distortion.Delta_K_kHz,
            "delta_J": payload.quartic_distortion.delta_J_kHz,
            "delta_K": payload.quartic_distortion.delta_K_kHz,
        },
        "sextic_distortion_hz": {
            "Phi_J": payload.sextic_distortion.Phi_J_Hz,
            "Phi_JK": payload.sextic_distortion.Phi_JK_Hz,
            "Phi_KJ": payload.sextic_distortion.Phi_KJ_Hz,
            "Phi_K": payload.sextic_distortion.Phi_K_Hz,
            "phi_J": payload.sextic_distortion.phi_J_Hz,
            "phi_JK": payload.sextic_distortion.phi_JK_Hz,
            "phi_K": payload.sextic_distortion.phi_K_Hz,
        },
        "nuclear_quadrupole": [q.model_dump() for q in payload.quadrupole_coupling],
        "spin_rotation": [sr.model_dump() for sr in payload.spin_rotation],
        "sha256": payload.raw_output_sha256,
        "converged": payload.calculation_converged
    }


def save_cfour_to_hdf5(
    payload: CFOUROutputPayload,
    h5_file_path: Union[str, Path],
    dataset_group: str = "ab_initio/cfour"
) -> None:
    """
    Serializes complete CFOUR calculation results into the Master SWMR HDF5 store (landscape.h5).
    """
    h5_path = Path(h5_file_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(h5_path), "a") as f:
        grp = f.require_group(dataset_group)

        # Store energies
        grp.attrs["final_energy_hartree"] = payload.energies.final_energy_hartree
        grp.attrs["scf_energy_hartree"] = payload.energies.scf_energy_hartree
        if payload.energies.ccsd_t_energy_hartree is not None:
            grp.attrs["ccsd_t_energy_hartree"] = payload.energies.ccsd_t_energy_hartree

        # Store rotational constants
        grp.attrs["Ae_MHz"] = payload.rotational_constants.Ae_MHz
        grp.attrs["Be_MHz"] = payload.rotational_constants.Be_MHz
        grp.attrs["Ce_MHz"] = payload.rotational_constants.Ce_MHz
        grp.attrs["A0_MHz"] = payload.rotational_constants.A0_MHz
        grp.attrs["B0_MHz"] = payload.rotational_constants.B0_MHz
        grp.attrs["C0_MHz"] = payload.rotational_constants.C0_MHz
        grp.attrs["asymmetry_kappa"] = payload.rotational_constants.asymmetry_kappa

        # Store quartic and sextic parameters
        grp.attrs["Delta_J_kHz"] = payload.quartic_distortion.Delta_J_kHz
        grp.attrs["Delta_JK_kHz"] = payload.quartic_distortion.Delta_JK_kHz
        grp.attrs["Delta_K_kHz"] = payload.quartic_distortion.Delta_K_kHz
        grp.attrs["Phi_J_Hz"] = payload.sextic_distortion.Phi_J_Hz
        grp.attrs["Phi_JK_Hz"] = payload.sextic_distortion.Phi_JK_Hz
        grp.attrs["Phi_K_Hz"] = payload.sextic_distortion.Phi_K_Hz

        # Store dipole moments
        grp.attrs["mu_a_debye"] = payload.dipole_moment.mu_a_debye
        grp.attrs["mu_b_debye"] = payload.dipole_moment.mu_b_debye
        grp.attrs["mu_c_debye"] = payload.dipole_moment.mu_c_debye
        grp.attrs["mu_total_debye"] = payload.dipole_moment.mu_total_debye

        # Store frequency arrays
        if payload.vibrational_data.harmonic_frequencies_cm1:
            if "harmonic_frequencies_cm1" in grp:
                del grp["harmonic_frequencies_cm1"]
            grp.create_dataset(
                "harmonic_frequencies_cm1",
                data=np.array(payload.vibrational_data.harmonic_frequencies_cm1, dtype=np.float64)
            )

        if payload.vibrational_data.anharmonic_frequencies_cm1:
            if "anharmonic_frequencies_cm1" in grp:
                del grp["anharmonic_frequencies_cm1"]
            grp.create_dataset(
                "anharmonic_frequencies_cm1",
                data=np.array(payload.vibrational_data.anharmonic_frequencies_cm1, dtype=np.float64)
            )

        grp.attrs["sha256_provenance"] = payload.raw_output_sha256
        grp.attrs["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    logger.info(f"Successfully archived CFOUR payload to HDF5 at {h5_path} (Group: {dataset_group}).")


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
        if re.search(r"(\*CFOUR\(.*?)(\n\s*\))", base_zmat, flags=re.DOTALL | re.IGNORECASE):
            modified_zmat = re.sub(
                r"(\*CFOUR\(.*?)(\n\s*\))",
                rf"\1\n{insertion}\2",
                base_zmat,
                count=1,
                flags=re.DOTALL | re.IGNORECASE
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
    lin_coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.065], [0.0, 0.0, 5.221]]
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

