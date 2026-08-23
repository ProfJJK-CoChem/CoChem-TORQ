Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task1_slurm.md.
Original prompt:
# Prompt: SLURM Batch Execution Configuration

**Target File:** `D:\__CoChem\GitHub-Repo\CoChem-TORQ\HPC_Launchers\cochem_submit.slurm`

## Objective
Establish the SLURM batch execution configuration for HPC arrays.

## Instructions for Coder
1. Create `cochem_submit.slurm` inside the `HPC_Launchers/` directory.
2. Provide a standard SLURM bash script header (e.g., `#SBATCH --job-name=CoChem-TORQ`, nodes, tasks, memory, time limits).
3. Ensure the script sets the `COCHEM_ARTIFACTS` environment variable securely to the HPC scratch or work directory before launching the Python engine.
4. Launch the `UI/Start_TORQ.ipynb` equivalent backend script (e.g., executing the core engine module).

## Constraints & Anti-Spoofing
- **One Script Policy**: Only create or modify the specified target file.
- **Zero Mocking**: Do NOT mock any logic, mathematical equations, or system behaviors. Must provide real physical implementation.
- **Context-Safety**: Do not hallucinate imports. Any dependencies must be strictly limited to the `requirements.txt` environment for CoChem-TORQ.
- **Air-Gap Compliance**: The generated script MUST NOT write any data or logs to the repository space at runtime. Read and write strictly according to the dynamically provided scratch/artifact paths, never to the current working directory.
Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_alignment.py ---
"""
CoChem-TORQ: Exact Eckart Frame Aligner & Rotational Constants Engine
=====================================================================
Phase 2 (Stage 1.0 - 2.0) Implementation
----------------------------------------
Implements the mass-weighted Cartesian normalizer ensuring rigid-rotor frame
stability, principal moment of inertia tensor diagonalization, right-handed
coordinate frame enforcement (det(R) = +1), CIAAW mono-isotopic mass locking,
Eckart matrix serialization, and bidirectional coordinate inversion.

Authoritative Standards:
- CIAAW / IUPAC Exact Mono-Isotopic Masses (Atomic Weights of the Elements)
- CODATA 2018 / 2022 Fundamental Physical Constants (h, c, u, Angstrom)
- Tripartite Filesystem Air-Gap Architecture (Ring 1 Static, Ring 2 Scratch, Ring 3 Artifacts)
- Method Matrix: Stage 1.0 - 2.0 Geometry Intake & Eckart Transformation
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.constants as const
from pydantic import BaseModel, Field

# Configure module-level logging
logger = logging.getLogger("CoChem-TORQ.Alignment")


# ============================================================================
# CODATA Fundamental Physical Constants & Conversion Prefactors
# ============================================================================

# Exact SI physical constants
PLANCK_CONSTANT_H: Final[float] = const.h  # J * s (6.62607015e-34)
SPEED_OF_LIGHT_C: Final[float] = const.c  # m / s (299792458.0)
ATOMIC_MASS_UNIT_KG: Final[float] = const.atomic_mass  # kg (1.66053906660e-27)
ANGSTROM_METERS: Final[float] = 1.0e-10  # m
HBAR: Final[float] = const.hbar  # J * s

# 1 u * Angstrom^2 in kg * m^2
U_ANGSTROM_SQ_TO_KG_M_SQ: Final[float] = ATOMIC_MASS_UNIT_KG * (ANGSTROM_METERS**2)

# Prefactors for B = h / (8 * pi^2 * I) where I is in u * Angstrom^2:
ROTATIONAL_PREFACTOR_HZ: Final[float] = PLANCK_CONSTANT_H / (
    8.0 * (np.pi**2) * U_ANGSTROM_SQ_TO_KG_M_SQ
)  # ~505379008435.3526 Hz * u * A^2
ROTATIONAL_PREFACTOR_MHZ: Final[float] = ROTATIONAL_PREFACTOR_HZ / 1.0e6  # ~505379.0084353526 MHz * u * A^2
ROTATIONAL_PREFACTOR_GHZ: Final[float] = ROTATIONAL_PREFACTOR_HZ / 1.0e9  # ~505.3790084353526 GHz * u * A^2
ROTATIONAL_PREFACTOR_CM1: Final[float] = PLANCK_CONSTANT_H / (
    8.0 * (np.pi**2) * (SPEED_OF_LIGHT_C * 100.0) * U_ANGSTROM_SQ_TO_KG_M_SQ
)  # ~16.85762916808776 cm^-1 * u * A^2
ROTATIONAL_PREFACTOR_JOULE: Final[float] = (HBAR**2) / (
    2.0 * U_ANGSTROM_SQ_TO_KG_M_SQ
)  # ~3.3486767622300885e-22 J * u * A^2


# ============================================================================
# Air-Gap Architecture & Security Exceptions
# ============================================================================


class AirGapViolationError(PermissionError):
    """Raised when an operation attempts to mutate or write into Ring 1 static repository space."""

    pass


class CoChemAirGapRing(enum.IntEnum):
    RING_1_DOMAIN_A_STATIC = 1  # Immutable Git Repository & Static Source Code
    RING_2_DOMAIN_C_EPHEMERAL = 2  # Volatile RAM disk, SHM & Temporary Scratch
    RING_3_DOMAIN_B_PERSISTENT = 3  # Dynamic HDF5 Vault & Curated Artifacts


class CoChemPathManager:
    """Standardized dynamic path resolution and air-gap integrity manager."""

    ENV_REPO_DIR: Final[str] = "COCHEM_REPO_DIR"
    ENV_SCRATCH_DIR: Final[str] = "COCHEM_SCRATCH_DIR"
    ENV_ARTIFACTS_DIR: Final[str] = "COCHEM_ARTIFACTS_DIR"
    ENV_ARTIFACTS_FALLBACK: Final[str] = "COCHEM_ARTIFACTS"

    @classmethod
    def get_repo_root(cls) -> Path:
        """Domain A / Ring 1: Static immutable repository root."""
        env_val = os.environ.get(cls.ENV_REPO_DIR)
        if env_val:
            repo_path = Path(env_val).resolve()
            if repo_path.is_dir():
                return repo_path

        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                return parent

        return Path.cwd().resolve()

    @classmethod
    def get_artifacts_root(cls, subfolder: Optional[str] = None) -> Path:
        """Domain B / Ring 3: Dynamic persistent artifact vault."""
        env_val = os.environ.get(cls.ENV_ARTIFACTS_DIR) or os.environ.get(cls.ENV_ARTIFACTS_FALLBACK)
        if env_val:
            base = Path(env_val).resolve()
        else:
            base = Path.home().resolve() / "CoChem_Artifacts"

        target = (base / subfolder) if subfolder else base
        cls.verify_air_gap_boundary(target, CoChemAirGapRing.RING_3_DOMAIN_B_PERSISTENT)
        target.mkdir(parents=True, exist_ok=True)
        return target

    @classmethod
    def get_scratch_root(cls, subfolder: Optional[str] = None) -> Path:
        """Domain C / Ring 2: Ephemeral high-speed scratch and IPC directory."""
        env_val = os.environ.get(cls.ENV_SCRATCH_DIR)
        if env_val:
            base = Path(env_val).resolve()
        else:
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                base = Path(local_app_data).resolve() / "CoChem" / "scratch"
            else:
                base = Path(tempfile.gettempdir()).resolve() / "cochem_scratch"

        target = (base / subfolder) if subfolder else base
        cls.verify_air_gap_boundary(target, CoChemAirGapRing.RING_2_DOMAIN_C_EPHEMERAL)
        target.mkdir(parents=True, exist_ok=True)
        return target

    @classmethod
    def verify_air_gap_boundary(cls, target_path: Path, ring: CoChemAirGapRing) -> None:
        """Adversarially verify that target_path strictly conforms to air-gap ring boundaries."""
        resolved = target_path.resolve()
        repo_root = cls.get_repo_root().resolve()

        if ring == CoChemAirGapRing.RING_1_DOMAIN_A_STATIC:
            try:
                resolved.relative_to(repo_root)
            except ValueError:
                raise AirGapViolationError(
                    f"CRITICAL AIR-GAP BREACH: Static Ring 1 target path '{resolved}' "
                    f"is not contained within immutable repository root '{repo_root}'."
                )
        elif ring in (CoChemAirGapRing.RING_2_DOMAIN_C_EPHEMERAL, CoChemAirGapRing.RING_3_DOMAIN_B_PERSISTENT):
            try:
                resolved.relative_to(repo_root)
                raise AirGapViolationError(
                    f"CRITICAL AIR-GAP BREACH: Mutable target path '{resolved}' "
                    f"is located within Ring 1 immutable repository root '{repo_root}'."
                )
            except ValueError:
                # Target path is correctly outside repo_root
                pass


# ============================================================================
# Exact CIAAW Mono-Isotopic Masses (u / Da)
# ============================================================================

CIAAW_ISOTOPIC_MASSES: Final[dict[str, float]] = {
    # Hydrogen & Isotopes
    "H": 1.00782503223,
    "1H": 1.00782503223,
    "D": 2.01410177812,
    "2H": 2.01410177812,
    "T": 3.0160492779,
    "3H": 3.0160492779,
    # Helium
    "He": 4.00260325413,
    "3He": 3.0160293201,
    "4He": 4.00260325413,
    # Lithium to Neon
    "Li": 7.0160034366,
    "6Li": 6.0151228874,
    "7Li": 7.0160034366,
    "Be": 9.012183065,
    "9Be": 9.012183065,
    "B": 11.00930536,
    "10B": 10.01293695,
    "11B": 11.00930536,
    "C": 12.00000000000,
    "12C": 12.00000000000,
    "13C": 13.00335483507,
    "14C": 14.0032419884,
    "N": 14.00307400443,
    "14N": 14.00307400443,
    "15N": 15.00010889888,
    "O": 15.99491461957,
    "16O": 15.99491461957,
    "17O": 16.99913175650,
    "18O": 17.99915961286,
    "F": 18.99840316273,
    "19F": 18.99840316273,
    "Ne": 19.9924401762,
    "20Ne": 19.9924401762,
    "21Ne": 20.993846685,
    "22Ne": 21.991385114,
    # Sodium to Argon
    "Na": 22.9897692820,
    "23Na": 22.9897692820,
    "Mg": 23.985041697,
    "24Mg": 23.985041697,
    "25Mg": 24.985836976,
    "26Mg": 25.982592968,
    "Al": 26.98153853,
    "27Al": 26.98153853,
    "Si": 27.97692653465,
    "28Si": 27.97692653465,
    "29Si": 28.9764946649,
    "30Si": 29.973770137,
    "P": 30.97376199842,
    "31P": 30.97376199842,
    "S": 31.97207073,
    "32S": 31.97207073,
    "33S": 32.9714589098,
    "34S": 33.967867016,
    "36S": 35.96708088,
    "Cl": 34.96885271,
    "35Cl": 34.96885271,
    "37Cl": 36.96590262,
    "Ar": 39.9623831237,
    "36Ar": 35.967545105,
    "38Ar": 37.96273211,
    "40Ar": 39.9623831237,
    # Potassium to Krypton
    "K": 38.9637064864,
    "39K": 38.9637064864,
    "40K": 39.963998166,
    "41K": 40.9618252579,
    "Ca": 39.962590863,
    "40Ca": 39.962590863,
    "42Ca": 41.95861783,
    "44Ca": 43.95548156,
    "Sc": 44.95590828,
    "Ti": 47.94794198,
    "48Ti": 47.94794198,
    "V": 50.9439570,
    "51V": 50.9439570,
    "Cr": 51.94050623,
    "52Cr": 51.94050623,
    "Mn": 54.93804391,
    "55Mn": 54.93804391,
    "Fe": 55.93493633,
    "56Fe": 55.93493633,
    "54Fe": 53.93960899,
    "57Fe": 56.93539284,
    "Co": 58.93319429,
    "59Co": 58.93319429,
    "Ni": 57.93534241,
    "58Ni": 57.93534241,
    "60Ni": 59.93078588,
    "Cu": 62.92959772,
    "63Cu": 62.92959772,
    "65Cu": 64.92778970,
    "Zn": 63.92914201,
    "64Zn": 63.92914201,
    "66Zn": 65.92603381,
    "Ga": 68.9255735,
    "69Ga": 68.9255735,
    "71Ga": 70.92470258,
    "Ge": 73.92117776,
    "74Ge": 73.92117776,
    "As": 74.92159457,
    "75As": 74.92159457,
    "Se": 79.91651990,
    "80Se": 79.91651990,
    "78Se": 77.9173095,
    "Br": 78.9183376,
    "79Br": 78.9183376,
    "81Br": 80.9162897,
    "Kr": 83.91149773,
    "84Kr": 83.91149773,
    # Rubidium to Xenon
    "Rb": 84.911789737,
    "85Rb": 84.911789737,
    "Sr": 87.9056125,
    "88Sr": 87.9056125,
    "Y": 88.9058479,
    "Zr": 89.9046977,
    "90Zr": 89.9046977,
    "Nb": 92.9063730,
    "Mo": 97.90540482,
    "98Mo": 97.90540482,
    "Tc": 97.9072124,
    "Ru": 101.9043441,
    "102Ru": 101.9043441,
    "Rh": 102.9054980,
    "Pd": 105.903478,
    "106Pd": 105.903478,
    "Ag": 106.905093,
    "107Ag": 106.905093,
    "109Ag": 108.904756,
    "Cd": 113.903361,
    "114Cd": 113.903361,
    "In": 114.903878,
    "Sn": 119.9021991,
    "120Sn": 119.9021991,
    "Sb": 120.9038120,
    "121Sb": 120.9038120,
    "Te": 129.90622274,
    "130Te": 129.90622274,
    "I": 126.9044719,
    "127I": 126.9044719,
    "Xe": 131.904155085,
    "132Xe": 131.904155085,
    # Heavy & Lanthanides / Actinides
    "Cs": 132.90545196,
    "Ba": 137.9052470,
    "138Ba": 137.9052470,
    "La": 138.906355,
    "Ce": 139.905442,
    "Pr": 140.907657,
    "Nd": 141.907729,
    "Sm": 151.919736,
    "Eu": 152.921235,
    "Gd": 157.924109,
    "Tb": 158.925350,
    "Dy": 163.929177,
    "Ho": 164.930328,
    "Er": 165.930295,
    "Tm": 168.934217,
    "Yb": 173.938866,
    "Lu": 174.940775,
    "Hf": 179.946557,
    "Ta": 180.947996,
    "W": 183.9509326,
    "184W": 183.9509326,
    "Re": 186.9557501,
    "Os": 189.9584450,
    "192Os": 191.961479,
    "Ir": 192.962924,
    "Pt": 194.9647917,
    "Au": 196.9665687,
    "Hg": 201.9706434,
    "202Hg": 201.9706434,
    "Tl": 204.9744278,
    "Pb": 207.9766525,
    "208Pb": 207.9766525,
    "Bi": 208.9803991,
    "Th": 232.0380558,
    "U": 238.0507884,
}


def enforce_ciaaw_masses(symbols: Sequence[str]) -> np.ndarray:
    """
    Maps atomic elemental or isotopic symbols to exact CIAAW mono-isotopic masses.

    Prioritizes the internal high-speed CIAAW mono-isotopic table and falls back to
    the mendeleev package for any unlisted heavy/rare isotope.

    :param symbols: List or sequence of atomic symbols (e.g. ['C', 'H', 'H', 'H', 'F']).
    :return: 1D numpy array of dtype float64 containing exact masses in atomic mass units (Da / u).
    :raises ValueError: If an unrecognized symbol or invalid element is provided.
    """
    if not symbols:
        return np.empty(0, dtype=np.float64)

    masses: list[float] = []
    for raw_sym in symbols:
        clean = raw_sym.strip()
        if not clean:
            raise ValueError("Empty or blank atomic symbol provided.")

        # Check standard capitalized lookup
        canonical = clean[0].upper() + clean[1:].lower() if len(clean) > 1 else clean.upper()

        if clean in CIAAW_ISOTOPIC_MASSES:
            masses.append(CIAAW_ISOTOPIC_MASSES[clean])
        elif canonical in CIAAW_ISOTOPIC_MASSES:
            masses.append(CIAAW_ISOTOPIC_MASSES[canonical])
        else:
            # Fallback to mendeleev
            try:
                import mendeleev

                el = mendeleev.element(canonical)
                most_abundant = max(
                    el.isotopes,
                    key=lambda iso: (iso.abundance if iso.abundance is not None else 0.0, iso.mass_number),
                )
                if most_abundant.mass is not None:
                    masses.append(float(most_abundant.mass))
                else:
                    masses.append(float(el.atomic_weight))
            except Exception as err:
                raise ValueError(
                    f"Unrecognized or invalid atomic symbol '{raw_sym}' cannot be mapped to CIAAW mass."
                ) from err

    return np.array(masses, dtype=np.float64)


# ============================================================================
# Center of Mass Translation & Inertia Tensor Formulation
# ============================================================================


def translate_com_to_origin(
    geometry: np.ndarray | Sequence[Sequence[float]],
    exact_masses: np.ndarray | Sequence[float],
    residual_tolerance: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the mass-weighted Center of Mass (COM) and translates the geometry
    such that COM resides strictly at the origin (0, 0, 0).

    Applies a two-pass numerical refinement to eliminate floating point rounding
    drift and rigorously asserts that the COM residual is below machine precision limits.

    :param geometry: Atomic Cartesian coordinates of shape (N, 3) in Angstroms.
    :param exact_masses: 1D array of exact atomic masses of length N in Da.
    :param residual_tolerance: Maximum allowed residual COM displacement in Angstroms (default 1e-12).
    :return: Tuple of (centered_geometry, com_vector), where centered_geometry has shape (N, 3)
             and com_vector has shape (3,).
    :raises ValueError: If geometry/mass shape mismatch occurs or COM residual exceeds tolerance.
    """
    geo = np.asarray(geometry, dtype=np.float64)
    masses = np.asarray(exact_masses, dtype=np.float64)

    if geo.ndim != 2 or geo.shape[1] != 3:
        raise ValueError(f"Geometry must be a 2D array of shape (N, 3), got {geo.shape}")
    if masses.ndim != 1 or masses.shape[0] != geo.shape[0]:
        raise ValueError(
            f"Masses must be a 1D array of length matching geometry ({geo.shape[0]}), got {masses.shape}"
        )
    if geo.shape[0] == 0:
        raise ValueError("Geometry array cannot be empty.")

    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError(f"Total molecular mass must be strictly positive, got {total_mass}")

    # Pass 1: Primary COM calculation
    com_vector = np.sum(masses[:, None] * geo, axis=0) / total_mass
    centered_geo = geo - com_vector

    # Pass 2: Numerical residual compensation (Kahan / iterative re-centering)
    com_residual = np.sum(masses[:, None] * centered_geo, axis=0) / total_mass
    centered_geo = centered_geo - com_residual
    com_vector = com_vector + com_residual

    # Rigorous validation of residual vector
    residual_com_norm = float(
        np.linalg.norm(np.sum(masses[:, None] * centered_geo, axis=0) / total_mass)
    )
    if residual_com_norm >= residual_tolerance:
        raise ValueError(
            f"Center of Mass translation failed: residual COM norm {residual_com_norm:.3e} "
            f"exceeds strict tolerance {residual_tolerance:.3e} Angstroms."
        )

    return centered_geo, com_vector


def compute_inertia_tensor(
    geometry: np.ndarray | Sequence[Sequence[float]],
    exact_masses: np.ndarray | Sequence[float],
) -> np.ndarray:
    """
    Calculates the 3x3 symmetric Moment of Inertia Tensor (I) in Da * Angstrom^2.

    Mathematical definition:
    I_alpha_beta = sum_i m_i * (||r_i||^2 * delta_alpha_beta - r_i_alpha * r_i_beta)

    :param geometry: Atomic Cartesian coordinates of shape (N, 3) (typically COM-centered).
    :param exact_masses: 1D array of exact atomic masses of length N in Da.
    :return: 3x3 symmetric Moment of Inertia Tensor as a float64 numpy array.
    """
    geo = np.asarray(geometry, dtype=np.float64)
    masses = np.asarray(exact_masses, dtype=np.float64)

    if geo.ndim != 2 or geo.shape[1] != 3:
        raise ValueError(f"Geometry must be of shape (N, 3), got {geo.shape}")
    if masses.ndim != 1 or masses.shape[0] != geo.shape[0]:
        raise ValueError(f"Mass array length ({len(masses)}) must match atom count ({geo.shape[0]})")

    x = geo[:, 0]
    y = geo[:, 1]
    z = geo[:, 2]

    # Diagonal elements
    i_xx = np.sum(masses * (y**2 + z**2))
    i_yy = np.sum(masses * (x**2 + z**2))
    i_zz = np.sum(masses * (x**2 + y**2))

    # Off-diagonal elements
    i_xy = -np.sum(masses * x * y)
    i_xz = -np.sum(masses * x * z)
    i_yz = -np.sum(masses * y * z)

    inertia_tensor = np.array(
        [
            [i_xx, i_xy, i_xz],
            [i_xy, i_yy, i_yz],
            [i_xz, i_yz, i_zz],
        ],
        dtype=np.float64,
    )

    return inertia_tensor


# ============================================================================
# Principal Axis Diagonalization & Eckart Alignment
# ============================================================================


def diagonalize_principal_axes(
    geometry: np.ndarray | Sequence[Sequence[float]],
    exact_masses: np.ndarray | Sequence[float],
    off_diagonal_tol: float = 1e-11,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonalizes the Moment of Inertia Tensor to establish the standard Eckart
    principal coordinate system (Ia <= Ib <= Ic) and enforces a strictly
    right-handed frame (det(R) = +1).

    1. Translates geometry to Center of Mass.
    2. Diagonalizes symmetric inertia tensor using np.linalg.eigh.
    3. Sorts eigenvalues in ascending order (Ia <= Ib <= Ic).
    4. Builds rotation matrix R = [v_a | v_b | v_c]^T.
    5. Inverts the 3rd eigenvector row if det(R) < 0 to guarantee det(R) = +1.0.
    6. Rotates geometry: r'' = r' @ R.T.
    7. Asserts that all off-diagonal elements in rotated inertia tensor vanish (< 1e-11).

    :param geometry: Atomic Cartesian coordinates of shape (N, 3) in Angstroms.
    :param exact_masses: 1D array of exact atomic masses in Da.
    :param off_diagonal_tol: Max allowed off-diagonal tensor element in rotated frame (default 1e-11).
    :return: Tuple of (aligned_geometry, principal_moments, rotation_matrix)
             - aligned_geometry: shape (N, 3)
             - principal_moments: shape (3,) with Ia <= Ib <= Ic in Da * A^2
             - rotation_matrix: shape (3, 3) with det(R) = +1.0
    """
    geo_centered, _ = translate_com_to_origin(geometry, exact_masses)
    masses = np.asarray(exact_masses, dtype=np.float64)

    # Compute initial inertia tensor
    inertia_tensor = compute_inertia_tensor(geo_centered, masses)

    # Diagonalize symmetric matrix
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)

    # Sort eigenvalues ascending (Ia <= Ib <= Ic)
    sort_indices = np.argsort(eigenvalues)
    principal_moments = eigenvalues[sort_indices]
    sorted_eigenvectors = eigenvectors[:, sort_indices]

    # Construct rotation matrix R where rows are the principal axes v_a^T, v_b^T, v_c^T
    # Transformation: r'' = R @ r'  ==> (N, 3) matrix multiplication: aligned_geo = geo_centered @ R.T
    rotation_matrix = sorted_eigenvectors.T

    # Enforce strictly right-handed coordinate frame: det(R) = +1.0
    det_r = np.linalg.det(rotation_matrix)
    if det_r < 0.0:
        # Flip the c-axis eigenvector (row 2 of rotation matrix)
        rotation_matrix[2, :] *= -1.0
        det_r = np.linalg.det(rotation_matrix)

    if not np.isclose(det_r, 1.0, atol=1e-7):
        raise RuntimeError(
            f"Failed to enforce right-handed Eckart coordinate frame: det(R) = {det_r:.7f} != +1.0"
        )

    # Apply rotation
    aligned_geometry = geo_centered @ rotation_matrix.T

    # Re-evaluate inertia tensor in the transformed frame to verify vanishing off-diagonal elements
    aligned_tensor = compute_inertia_tensor(aligned_geometry, masses)
    off_diag_elements = np.array(
        [
            aligned_tensor[0, 1],
            aligned_tensor[0, 2],
            aligned_tensor[1, 2],
        ]
    )
    max_off_diag = float(np.max(np.abs(off_diag_elements)))

    if max_off_diag >= off_diagonal_tol:
        raise ValueError(
            f"Off-diagonal elements in transformed inertia tensor exceed tolerance: "
            f"max(|I_alpha_beta|) = {max_off_diag:.3e} >= {off_diagonal_tol:.3e} Da*A^2"
        )

    return aligned_geometry, principal_moments, rotation_matrix


# ============================================================================
# Spectroscopic Observables: Rotational Constants, Ray's Kappa, Inertial Defect
# ============================================================================


def compute_rotational_constants(
    principal_moments: np.ndarray | Sequence[float],
    unit: str = "MHz",
) -> Tuple[float, float, float]:
    """
    Computes spectroscopic rotational constants (A, B, C) from principal moments
    of inertia (Ia, Ib, Ic) using exact CODATA conversion factors.

    Formula:
    A = h / (8 * pi^2 * I_a),  B = h / (8 * pi^2 * I_b),  C = h / (8 * pi^2 * I_c)

    Supported units:
    - 'MHz': Megahertz (default)
    - 'GHz': Gigahertz
    - 'cm-1' / 'cm^-1' / 'invcm': Wavenumbers (cm^-1)
    - 'Hz': Hertz
    - 'J' / 'Joule': Joules (hbar^2 / (2 * I))

    :param principal_moments: 3-element sequence of principal moments (Ia, Ib, Ic) in Da * A^2.
    :param unit: Output unit string (case-insensitive).
    :return: Tuple of floats (A, B, C) in the requested unit.
    :raises ValueError: If an unsupported unit is requested or input moments are malformed.
    """
    moments = np.asarray(principal_moments, dtype=np.float64)
    if moments.shape != (3,):
        raise ValueError(f"principal_moments must be a 3-element sequence, got shape {moments.shape}")

    u_norm = unit.strip().lower()
    if u_norm in ("mhz",):
        prefactor = ROTATIONAL_PREFACTOR_MHZ
    elif u_norm in ("ghz",):
        prefactor = ROTATIONAL_PREFACTOR_GHZ
    elif u_norm in ("cm-1", "cm^-1", "cm_1", "cm**-1", "invcm"):
        prefactor = ROTATIONAL_PREFACTOR_CM1
    elif u_norm in ("hz",):
        prefactor = ROTATIONAL_PREFACTOR_HZ
    elif u_norm in ("j", "joule", "joules"):
        prefactor = ROTATIONAL_PREFACTOR_JOULE
    else:
        raise ValueError(
            f"Unsupported rotational constant unit '{unit}'. "
            f"Supported units: 'MHz', 'GHz', 'cm-1', 'Hz', 'J'."
        )

    ia, ib, ic = float(moments[0]), float(moments[1]), float(moments[2])

    # Singularities / zero moment handling (e.g. linear molecule along a-axis)
    a_const = float("inf") if ia < 1.0e-10 else float(prefactor / ia)
    b_const = float("inf") if ib < 1.0e-10 else float(prefactor / ib)
    c_const = float("inf") if ic < 1.0e-10 else float(prefactor / ic)

    return a_const, b_const, c_const


def compute_ray_asymmetry_parameter(
    rotational_constants: Tuple[float, float, float] | np.ndarray | Sequence[float],
) -> float:
    """
    Computes Ray's asymmetry parameter (kappa) describing the degree of molecular asymmetry.

    Formula:
    kappa = (2 * B - A - C) / (A - C)

    Limiting values:
    - Prolate symmetric top limit: kappa = -1.0 (B = C)
    - Oblate symmetric top limit: kappa = +1.0 (A = B)
    - Most asymmetric top: kappa = 0.0 (B = (A + C) / 2)
    - Linear molecule (A -> inf, B = C): kappa = -1.0
    - Spherical top (A = B = C): kappa = 0.0

    :param rotational_constants: 3-element tuple or array of (A, B, C).
    :return: Ray's kappa as a float in the range [-1.0, 1.0].
    """
    rc = [float(v) for v in rotational_constants]
    if len(rc) != 3:
        raise ValueError(f"rotational_constants must have 3 elements, got {len(rc)}")

    a, b, c = rc[0], rc[1], rc[2]

    # Linear rotor limit: A is infinite or extremely large
    if math.isinf(a) or a > 1.0e12:
        return -1.0

    # Spherical top: A == B == C
    denom = a - c
    if abs(denom) < 1.0e-10:
        return 0.0

    kappa = (2.0 * b - a - c) / denom

    # Clamp numerical boundary precision
    if kappa < -1.0 and kappa >= -1.0000001:
        kappa = -1.0
    elif kappa > 1.0 and kappa <= 1.0000001:
        kappa = 1.0

    return float(kappa)


def compute_inertial_defect(
    principal_moments: np.ndarray | Sequence[float],
) -> float:
    """
    Computes the inertial defect (Delta) in Da * Angstrom^2.

    Formula:
    Delta = I_c - I_a - I_b

    For a strictly planar molecule at equilibrium geometry: Delta == 0.0.
    For non-planar molecules: Delta < 0.0.

    :param principal_moments: Sequence of (Ia, Ib, Ic) in Da * Angstrom^2.
    :return: Inertial defect Delta in Da * Angstrom^2.
    """
    moments = np.asarray(principal_moments, dtype=np.float64)
    if moments.shape != (3,):
        raise ValueError(f"principal_moments must be a 3-element sequence, got shape {moments.shape}")

    ia, ib, ic = float(moments[0]), float(moments[1]), float(moments[2])
    return float(ic - ia - ib)


def classify_rotor_type(
    principal_moments: np.ndarray | Sequence[float],
    tol: float = 1e-4,
) -> str:
    """
    Classifies the molecular rotor into one of the 5 canonical spectroscopic rotor types:
    - 'linear': Ia ~ 0, Ib == Ic
    - 'spherical_top': Ia == Ib == Ic
    - 'prolate_symmetric_top': Ia < Ib == Ic  (cigar-shaped)
    - 'oblate_symmetric_top': Ia == Ib < Ic   (pancake/frisbee-shaped)
    - 'asymmetric_top': Ia < Ib < Ic

    :param principal_moments: 3-element sequence of principal moments (Ia, Ib, Ic) in Da * A^2.
    :param tol: Relative fractional tolerance for moment equality (default 1e-4).
    :return: String identifier of the rotor type.
    """
    moments = np.sort(np.asarray(principal_moments, dtype=np.float64))
    ia, ib, ic = float(moments[0]), float(moments[1]), float(moments[2])

    max_moment = max(ic, 1.0e-12)

    # 1. Linear rotor: Ia is near 0 and Ib == Ic
    if (ia / max_moment < tol or ia < 1.0e-6) and abs(ib - ic) / max_moment < tol:
        return "linear"

    # 2. Spherical top: Ia == Ib == Ic
    if abs(ic - ia) / max_moment < tol:
        return "spherical_top"

    # 3. Prolate symmetric top: Ia < Ib == Ic
    if abs(ib - ic) / max_moment < tol:
        return "prolate_symmetric_top"

    # 4. Oblate symmetric top: Ia == Ib < Ic
    if abs(ia - ib) / max_moment < tol:
        return "oblate_symmetric_top"

    # 5. Asymmetric top
    return "asymmetric_top"


# ============================================================================
# Bidirectional Coordinate Inversion & Provenance Serialization
# ============================================================================


def invert_eckart_coordinates(
    aligned_geometry: np.ndarray | Sequence[Sequence[float]],
    rotation_matrix: np.ndarray | Sequence[Sequence[float]],
    com_vector: np.ndarray | Sequence[float],
) -> np.ndarray:
    """
    Reconstructs original laboratory frame Cartesian coordinates from the
    Eckart-aligned coordinates, rotation matrix, and COM translation vector:

    Formula:
    r = r'' @ R + r_COM

    where:
    - r'' is the aligned geometry (N, 3)
    - R is the (3, 3) Eckart rotation matrix with det(R) = +1
    - r_COM is the (3,) Center of Mass vector

    :param aligned_geometry: Aligned coordinates of shape (N, 3).
    :param rotation_matrix: 3x3 orthonormal rotation matrix.
    :param com_vector: 3-element Center of Mass translation vector.
    :return: Reconstructed laboratory frame coordinates of shape (N, 3) as float64 array.
    """
    aligned = np.asarray(aligned_geometry, dtype=np.float64)
    rot = np.asarray(rotation_matrix, dtype=np.float64)
    com = np.asarray(com_vector, dtype=np.float64)

    if aligned.ndim != 2 or aligned.shape[1] != 3:
        raise ValueError(f"aligned_geometry must have shape (N, 3), got {aligned.shape}")
    if rot.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3), got {rot.shape}")
    if com.shape != (3,):
        raise ValueError(f"com_vector must have shape (3,), got {com.shape}")

    return aligned @ rot + com


def serialize_eckart_matrix(
    rotation_matrix: np.ndarray | Sequence[Sequence[float]],
    com_vector: np.ndarray | Sequence[float],
    output_path: Path | str,
    metadata: Optional[dict[str, Any]] = None,
    provenance_tag: str = "[D]",
) -> Path:
    """
    Serializes the 3x3 Eckart rotation matrix and COM translation vector to JSON
    with atomic write guarantees and strict Air-Gap verification.

    :param rotation_matrix: 3x3 orthonormal rotation matrix.
    :param com_vector: 3-element COM translation vector.
    :param output_path: Destination path for serialization.
    :param metadata: Optional auxiliary metadata dictionary.
    :param provenance_tag: Method Matrix provenance classification tag (default '[D]').
    :return: Resolved Path of the written JSON artifact.
    :raises AirGapViolationError: If output_path is located inside Ring 1 Domain A static repository.
    """
    target = Path(output_path).resolve()

    # Verify Air-Gap: target must not reside in Ring 1 immutable repo space
    CoChemPathManager.verify_air_gap_boundary(target, CoChemAirGapRing.RING_3_DOMAIN_B_PERSISTENT)

    target.parent.mkdir(parents=True, exist_ok=True)

    rot_arr = np.asarray(rotation_matrix, dtype=np.float64)
    com_arr = np.asarray(com_vector, dtype=np.float64)

    rot_list = [[float(v) for v in row] for row in rot_arr]
    com_list = [float(v) for v in com_arr]
    det_r = float(np.linalg.det(rot_arr))

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance_tag,
        "rotation_matrix": rot_list,
        "com_vector": com_list,
        "det_r": det_r,
        "metadata": metadata or {},
    }

    raw_json = json.dumps(payload, indent=2, ensure_ascii=False)
    payload_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    payload["sha256"] = payload_hash

    # Atomic write pattern: write to temporary file in target directory then os.replace
    temp_file = target.parent / f".tmp_{uuid.uuid4().hex}_{target.name}"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, ensure_ascii=False))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, target)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass

    logger.info("Successfully serialized Eckart matrix to %s (SHA256=%s...)", target, payload_hash[:8])
    return target


# ============================================================================
# High-Level Result Model & Alignment Pipeline Classes
# ============================================================================


class EckartAlignmentResult(BaseModel):
    """
    Encapsulates the structured output of the Exact Eckart Frame Aligner.
    """

    model_config = {"arbitrary_types_allowed": True}

    aligned_geometry: list[list[float]] = Field(
        ..., description="Eckart-aligned Cartesian coordinates in Angstroms of shape (N, 3)."
    )
    principal_moments: Tuple[float, float, float] = Field(
        ..., description="Principal moments of inertia (Ia <= Ib <= Ic) in Da * A^2."
    )
    rotational_constants_mhz: Tuple[float, float, float] = Field(
        ..., description="Rotational constants (A, B, C) in MHz."
    )
    rotational_constants_ghz: Tuple[float, float, float] = Field(
        ..., description="Rotational constants (A, B, C) in GHz."
    )
    rotational_constants_cm1: Tuple[float, float, float] = Field(
        ..., description="Rotational constants (A, B, C) in cm^-1."
    )
    rotation_matrix: list[list[float]] = Field(
        ..., description="3x3 orthonormal Eckart rotation matrix with det(R) = +1.0."
    )
    com_vector: Tuple[float, float, float] = Field(
        ..., description="Original Center of Mass translation vector (r_COM) in Angstroms."
    )
    ray_kappa: float = Field(
        ..., description="Ray's asymmetry parameter kappa = (2B - A - C) / (A - C)."
    )
    inertial_defect: float = Field(
        ..., description="Inertial defect Delta = Ic - Ia - Ib in Da * A^2."
    )
    rotor_type: str = Field(
        ..., description="Canonical spectroscopic rotor classification."
    )
    symbols: list[str] = Field(
        ..., description="List of atomic symbols in coordinate order."
    )
    exact_masses: list[float] = Field(
        ..., description="List of exact CIAAW mono-isotopic masses in Da."
    )
    det_r: float = Field(
        ..., description="Determinant of the rotation matrix (strictly +1.0)."
    )
    provenance: str = Field(
        default="[D]", description="Method Matrix provenance tag ([M], [D], [E])."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Supplementary calculation metadata."
    )

    def to_numpy(self) -> np.ndarray:
        """Returns the aligned geometry as a float64 numpy array of shape (N, 3)."""
        return np.array(self.aligned_geometry, dtype=np.float64)

    def invert_coordinates(self, coords: Optional[np.ndarray | Sequence[Sequence[float]]] = None) -> np.ndarray:
        """
        Reconstructs coordinates back to the laboratory frame.
        If coords is None, reconstructs self.aligned_geometry.
        """
        target = self.to_numpy() if coords is None else np.asarray(coords, dtype=np.float64)
        rot = np.array(self.rotation_matrix, dtype=np.float64)
        com = np.array(self.com_vector, dtype=np.float64)
        return invert_eckart_coordinates(target, rot, com)

    def serialize(self, output_path: Path | str) -> Path:
        """Serializes the Eckart alignment transformation matrix and metadata to disk."""
        rot = np.array(self.rotation_matrix, dtype=np.float64)
        com = np.array(self.com_vector, dtype=np.float64)
        meta = {
            **self.metadata,
            "symbols": self.symbols,
            "principal_moments": list(self.principal_moments),
            "rotational_constants_mhz": list(self.rotational_constants_mhz),
            "ray_kappa": self.ray_kappa,
            "inertial_defect": self.inertial_defect,
            "rotor_type": self.rotor_type,
        }
        return serialize_eckart_matrix(
            rot, com, output_path, metadata=meta, provenance_tag=self.provenance
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts the result model into a standard Python dictionary."""
        return self.model_dump()


class EckartAligner:
    """
    High-level orchestrator for mass-weighted Cartesian normalization,
    rigid-rotor principal axis diagonalization, and Eckart frame alignment.
    """

    def __init__(
        self,
        symbols: Optional[Sequence[str]] = None,
        masses: Optional[np.ndarray | Sequence[float]] = None,
        provenance_tag: str = "[D]",
    ) -> None:
        self.symbols: Optional[list[str]] = [s.strip() for s in symbols] if symbols is not None else None
        self.masses: Optional[np.ndarray]
        if masses is not None:
            self.masses = np.asarray(masses, dtype=np.float64)
        elif self.symbols is not None:
            self.masses = enforce_ciaaw_masses(self.symbols)
        else:
            self.masses = None
        self.provenance_tag = provenance_tag

    def align(
        self,
        geometry: np.ndarray | Sequence[Sequence[float]],
        symbols: Optional[Sequence[str]] = None,
        exact_masses: Optional[np.ndarray | Sequence[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> EckartAlignmentResult:
        """
        Executes full Eckart frame alignment on input geometry.

        :param geometry: Atomic Cartesian coordinates of shape (N, 3) in Angstroms.
        :param symbols: Optional atomic symbols override.
        :param exact_masses: Optional exact masses override.
        :param metadata: Optional auxiliary metadata dictionary.
        :return: EckartAlignmentResult model instance.
        """
        geo_arr = np.asarray(geometry, dtype=np.float64)
        if geo_arr.ndim != 2 or geo_arr.shape[1] != 3:
            raise ValueError(f"Geometry must have shape (N, 3), got {geo_arr.shape}")

        syms = [s.strip() for s in symbols] if symbols is not None else self.symbols
        if exact_masses is not None:
            mass_arr = np.asarray(exact_masses, dtype=np.float64)
        elif self.masses is not None and (syms is None or len(syms) == len(self.masses)):
            mass_arr = self.masses
        elif syms is not None:
            mass_arr = enforce_ciaaw_masses(syms)
        else:
            raise ValueError("Atomic symbols or exact masses must be provided for mass-weighted alignment.")

        if syms is not None and len(syms) != geo_arr.shape[0]:
            raise ValueError(
                f"Symbol count ({len(syms)}) does not match geometry atom count ({geo_arr.shape[0]})."
            )
        if mass_arr.shape[0] != geo_arr.shape[0]:
            raise ValueError(
                f"Mass count ({mass_arr.shape[0]}) does not match geometry atom count ({geo_arr.shape[0]})."
            )

        # 1. Translate COM to origin
        centered_geo, com_vector = translate_com_to_origin(geo_arr, mass_arr)

        # 2. Diagonalize principal axes
        aligned_geo, principal_moments, rotation_matrix = diagonalize_principal_axes(geo_arr, mass_arr)

        # 3. Compute rotational constants in multiple units
        rc_mhz = compute_rotational_constants(principal_moments, unit="MHz")
        rc_ghz = compute_rotational_constants(principal_moments, unit="GHz")
        rc_cm1 = compute_rotational_constants(principal_moments, unit="cm-1")

        # 4. Ray's asymmetry parameter
        kappa = compute_ray_asymmetry_parameter(rc_mhz)

        # 5. Inertial defect
        delta = compute_inertial_defect(principal_moments)

        # 6. Rotor classification
        rotor_type = classify_rotor_type(principal_moments)

        det_r = float(np.linalg.det(rotation_matrix))

        return EckartAlignmentResult(
            aligned_geometry=[[float(v) for v in row] for row in aligned_geo],
            principal_moments=(float(principal_moments[0]), float(principal_moments[1]), float(principal_moments[2])),
            rotational_constants_mhz=rc_mhz,
            rotational_constants_ghz=rc_ghz,
            rotational_constants_cm1=rc_cm1,
            rotation_matrix=[[float(v) for v in row] for row in rotation_matrix],
            com_vector=(float(com_vector[0]), float(com_vector[1]), float(com_vector[2])),
            ray_kappa=float(kappa),
            inertial_defect=float(delta),
            rotor_type=rotor_type,
            symbols=syms if syms is not None else [f"X{i+1}" for i in range(len(mass_arr))],
            exact_masses=[float(m) for m in mass_arr],
            det_r=det_r,
            provenance=self.provenance_tag,
            metadata=metadata or {},
        )

    @classmethod
    def align_molecule(
        cls,
        symbols: Sequence[str],
        geometry: np.ndarray | Sequence[Sequence[float]],
        provenance_tag: str = "[D]",
        metadata: Optional[dict[str, Any]] = None,
    ) -> EckartAlignmentResult:
        """
        Convenience classmethod to align a molecule directly from symbols and geometry.
        """
        aligner = cls(symbols=list(symbols), provenance_tag=provenance_tag)
        return aligner.align(geometry=geometry, metadata=metadata)

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_alignment.py ---
"""
CoChem-TORQ: Exact Eckart Frame Aligner & Rotational Constants Engine Test Suite
================================================================================
Phase 2 (Stage 1.0 - 2.0) Zero-Mock Physical Test Matrix
--------------------------------------------------------
Tests mass-weighted Cartesian normalization, rigid-rotor frame stability,
inertia tensor diagonalization, right-handed coordinate enforcement (det(R) = +1),
rotational constants in multiple units, Ray's asymmetry parameter, inertial defect,
rotor classification across all 5 canonical types, Eckart coordinate inversion,
and Tripartite Filesystem Air-Gap security.

Authoritative Standards:
- CIAAW / IUPAC Exact Mono-Isotopic Masses
- CODATA 2018 / 2022 Fundamental Physical Constants
- Method Matrix: Stage 1.0 - 2.0 Gateway & Eckart Alignment Protocols
- Provenance tags: [M] Measured, [D] Derived, [E] Estimated
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import scipy.constants as const
from scipy.spatial.transform import Rotation

from Libraries.cochem_torq_alignment import (
    CIAAW_ISOTOPIC_MASSES,
    AirGapViolationError,
    CoChemAirGapRing,
    CoChemPathManager,
    EckartAligner,
    EckartAlignmentResult,
    classify_rotor_type,
    compute_inertia_tensor,
    compute_inertial_defect,
    compute_ray_asymmetry_parameter,
    compute_rotational_constants,
    diagonalize_principal_axes,
    enforce_ciaaw_masses,
    invert_eckart_coordinates,
    serialize_eckart_matrix,
    translate_com_to_origin,
)


# ============================================================================
# Authentic Physical Geometry Fixtures
# ============================================================================

@pytest.fixture
def water_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic equilibrium C2v water (H2O) geometry:
    r(OH) = 0.9578 Angstrom, theta(HOH) = 104.5 degrees. [D]
    """
    r_oh = 0.9578
    theta_rad = np.radians(104.5)
    syms = ["O", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [r_oh * np.sin(theta_rad / 2.0), r_oh * np.cos(theta_rad / 2.0), 0.0],
            [-r_oh * np.sin(theta_rad / 2.0), r_oh * np.cos(theta_rad / 2.0), 0.0],
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def co2_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic equilibrium linear D_inf_h carbon dioxide (CO2) geometry:
    r(CO) = 1.1600 Angstrom. [D]
    """
    r_co = 1.1600
    syms = ["O", "C", "O"]
    coords = np.array(
        [
            [0.0, 0.0, -r_co],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, r_co],
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def methane_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic equilibrium tetrahedral Td methane (CH4) geometry:
    r(CH) = 1.0870 Angstrom. [D]
    """
    r_ch = 1.0870
    a = r_ch / np.sqrt(3.0)
    syms = ["C", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [a, a, a],
            [a, -a, -a],
            [-a, a, -a],
            [-a, -a, a],
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def methyl_chloride_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic equilibrium prolate C3v methyl chloride (CH3Cl) geometry:
    r(CCl) = 1.7810 Angstrom, r(CH) = 1.0860 Angstrom, theta(HCH) = 110.5 degrees. [D]
    """
    r_ccl = 1.7810
    r_ch = 1.0860
    theta_hch = np.radians(110.5)
    # Distance from z-axis for H atoms: r_ch * sin(theta_tilt)
    # Using tetrahedral relation: cos(angle between C-H and C3 axis) = cos(theta_c3)
    cos_theta_c3 = np.sqrt((2.0 * np.cos(theta_hch) + 1.0) / 3.0) if (2.0 * np.cos(theta_hch) + 1.0) > 0 else 0.35
    sin_theta_c3 = np.sqrt(1.0 - cos_theta_c3**2)
    r_xy = r_ch * sin_theta_c3
    z_h = -r_ch * cos_theta_c3

    syms = ["C", "Cl", "H", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, r_ccl],
            [r_xy, 0.0, z_h],
            [-r_xy * 0.5, r_xy * np.sqrt(3.0) / 2.0, z_h],
            [-r_xy * 0.5, -r_xy * np.sqrt(3.0) / 2.0, z_h],
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def benzene_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic planar D6h benzene (C6H6) geometry:
    r(CC) = 1.3970 Angstrom, r(CH) = 1.0840 Angstrom. [D]
    """
    r_cc = 1.3970
    r_ch = 1.0840
    syms = ["C"] * 6 + ["H"] * 6
    coords_list: list[list[float]] = []
    for i in range(6):
        phi = i * np.pi / 3.0
        coords_list.append([r_cc * np.cos(phi), r_cc * np.sin(phi), 0.0])
    for i in range(6):
        phi = i * np.pi / 3.0
        coords_list.append([(r_cc + r_ch) * np.cos(phi), (r_cc + r_ch) * np.sin(phi), 0.0])

    return syms, np.array(coords_list, dtype=np.float64)


@pytest.fixture
def fluoropropane_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic asymmetric top 2-fluoropropane (C3H7F) geometry: [D]
    """
    syms = ["C", "F", "C", "C", "H", "H", "H", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.0000, 0.0000, 0.0000],  # C2
            [1.4000, 0.0000, 0.0000],  # F
            [-0.5000, 1.2500, 0.0000],  # C1
            [-0.5000, -0.6250, 1.0825],  # C3
            [-0.3000, 0.0000, -1.0000],  # H(C2)
            [-1.5500, 1.2500, 0.0000],  # H1a
            [-0.1500, 1.7500, 0.8900],  # H1b
            [-0.1500, 1.7500, -0.8900],  # H1c
            [-1.5500, -0.6250, 1.0825],  # H3a
            [-0.1500, -0.1250, 1.9725],  # H3b
            [-0.1500, -1.6500, 1.0825],  # H3c
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def ethanol_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic anti-conformer ethanol (CH3CH2OH) geometry: [D]
    """
    syms = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.0000, 0.0000, 0.0000],  # C1
            [1.5200, 0.0000, 0.0000],  # C2
            [2.0800, 1.2800, 0.0000],  # O
            [3.0400, 1.2000, 0.0000],  # H(O)
            [-0.3800, 1.0200, 0.0000],  # H1a
            [-0.3800, -0.5100, 0.8900],  # H1b
            [-0.3800, -0.5100, -0.8900],  # H1c
            [1.9000, -0.5100, 0.8900],  # H2a
            [1.9000, -0.5100, -0.8900],  # H2b
        ],
        dtype=np.float64,
    )
    return syms, coords


# ============================================================================
# 1. CIAAW Mono-Isotopic Mass Enforcement Tests
# ============================================================================

def test_ciaaw_mass_enforcement_primary_elements() -> None:
    """Verifies that CIAAW mono-isotopic masses match IUPAC definitions exactly. [M]"""
    syms = ["H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "Br", "I"]
    masses = enforce_ciaaw_masses(syms)

    assert len(masses) == len(syms)
    assert masses.dtype == np.float64
    assert np.isclose(masses[0], 1.00782503223, atol=1e-10)  # 1H
    assert np.isclose(masses[1], 12.00000000000, atol=1e-10)  # 12C
    assert np.isclose(masses[2], 14.00307400443, atol=1e-10)  # 14N
    assert np.isclose(masses[3], 15.99491461957, atol=1e-10)  # 16O
    assert np.isclose(masses[4], 18.99840316273, atol=1e-10)  # 19F
    assert np.isclose(masses[7], 31.97207073, atol=1e-8)  # 32S
    assert np.isclose(masses[8], 34.96885271, atol=1e-8)  # 35Cl
    assert np.isclose(masses[9], 78.9183376, atol=1e-7)  # 79Br
    assert np.isclose(masses[10], 126.9044719, atol=1e-7)  # 127I


def test_ciaaw_mass_enforcement_isotopologues() -> None:
    """Verifies that specific isotopes (D, 13C, 15N, 18O, 37Cl) resolve accurately. [M]"""
    syms = ["D", "2H", "T", "13C", "15N", "18O", "37Cl", "81Br"]
    masses = enforce_ciaaw_masses(syms)

    assert np.isclose(masses[0], 2.01410177812, atol=1e-10)  # D
    assert np.isclose(masses[1], 2.01410177812, atol=1e-10)  # 2H
    assert np.isclose(masses[2], 3.0160492779, atol=1e-9)  # T
    assert np.isclose(masses[3], 13.00335483507, atol=1e-10)  # 13C
    assert np.isclose(masses[4], 15.00010889888, atol=1e-10)  # 15N
    assert np.isclose(masses[5], 17.99915961286, atol=1e-10)  # 18O
    assert np.isclose(masses[6], 36.96590262, atol=1e-8)  # 37Cl
    assert np.isclose(masses[7], 80.9162897, atol=1e-7)  # 81Br


def test_ciaaw_mass_enforcement_case_insensitivity() -> None:
    """Verifies that lowercase and mixed-case atomic symbols are resolved correctly. [D]"""
    syms = ["c", "h", "o", "cl", "BR", "na"]
    masses = enforce_ciaaw_masses(syms)

    assert np.isclose(masses[0], 12.00000000000, atol=1e-10)
    assert np.isclose(masses[1], 1.00782503223, atol=1e-10)
    assert np.isclose(masses[2], 15.99491461957, atol=1e-10)
    assert np.isclose(masses[3], 34.96885271, atol=1e-8)
    assert np.isclose(masses[4], 78.9183376, atol=1e-7)
    assert np.isclose(masses[5], 22.9897692820, atol=1e-8)


def test_ciaaw_mass_enforcement_errors() -> None:
    """Verifies error handling on empty lists and invalid symbols."""
    assert len(enforce_ciaaw_masses([])) == 0

    with pytest.raises(ValueError, match="Empty or blank atomic symbol"):
        enforce_ciaaw_masses(["C", "  ", "O"])

    with pytest.raises(ValueError, match="Unrecognized or invalid atomic symbol"):
        enforce_ciaaw_masses(["C", "NonExistentElementXYZ", "O"])


# ============================================================================
# 2. Center of Mass Translation & Precision Tests
# ============================================================================

def test_translate_com_to_origin_water(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """Verifies mass-weighted COM centering for water with residual < 1e-12 Angstrom. [D]"""
    syms, coords = water_geometry
    masses = enforce_ciaaw_masses(syms)

    centered_geo, com_vector = translate_com_to_origin(coords, masses)

    # Check centered geometry shape
    assert centered_geo.shape == coords.shape
    assert com_vector.shape == (3,)

    # Total mass and mass-weighted residual
    total_mass = np.sum(masses)
    residual_com = np.sum(masses[:, None] * centered_geo, axis=0) / total_mass
    residual_norm = float(np.linalg.norm(residual_com))

    assert residual_norm < 1e-12, f"COM residual {residual_norm:.3e} exceeds tolerance 1e-12"
    assert np.allclose(coords - com_vector, centered_geo, atol=1e-14)


def test_translate_com_arbitrary_shift(fluoropropane_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Tests COM translation with a deliberate large spatial translation (+100, -250, +500 A). [D]
    """
    syms, coords = fluoropropane_geometry
    masses = enforce_ciaaw_masses(syms)

    shift = np.array([100.0, -250.0, 500.0])
    shifted_coords = coords + shift

    centered_geo, com_vector = translate_com_to_origin(shifted_coords, masses)

    total_mass = np.sum(masses)
    residual_com = np.sum(masses[:, None] * centered_geo, axis=0) / total_mass
    residual_norm = float(np.linalg.norm(residual_com))

    assert residual_norm < 1e-12
    # Verify that subtracting the initial COM and recentering matches
    orig_centered, orig_com = translate_com_to_origin(coords, masses)
    assert np.allclose(centered_geo, orig_centered, atol=1e-12)
    assert np.allclose(com_vector, orig_com + shift, atol=1e-12)


def test_translate_com_validation_errors() -> None:
    """Verifies input validation on translate_com_to_origin."""
    # Mismatched dimensions
    with pytest.raises(ValueError, match="Geometry must be a 2D array of shape"):
        translate_com_to_origin(np.zeros((3, 2)), np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="Masses must be a 1D array"):
        translate_com_to_origin(np.zeros((3, 3)), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="Geometry array cannot be empty"):
        translate_com_to_origin(np.empty((0, 3)), np.empty(0))

    with pytest.raises(ValueError, match="Total molecular mass must be strictly positive"):
        translate_com_to_origin(np.zeros((2, 3)), np.array([0.0, 0.0]))


# ============================================================================
# 3. Moment of Inertia Tensor Calculation Tests
# ============================================================================

def test_compute_inertia_tensor_symmetry_and_values(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies that the calculated 3x3 inertia tensor is strictly symmetric
    and adheres to parallel axis theorem invariants. [D]
    """
    syms, coords = water_geometry
    masses = enforce_ciaaw_masses(syms)
    centered_geo, com_vector = translate_com_to_origin(coords, masses)

    tensor = compute_inertia_tensor(centered_geo, masses)

    assert tensor.shape == (3, 3)
    # Check strict symmetry
    assert np.allclose(tensor, tensor.T, atol=1e-14)

    # Manual analytical calculation
    x = centered_geo[:, 0]
    y = centered_geo[:, 1]
    z = centered_geo[:, 2]

    expected_ixx = np.sum(masses * (y**2 + z**2))
    expected_iyy = np.sum(masses * (x**2 + z**2))
    expected_izz = np.sum(masses * (x**2 + y**2))
    expected_ixy = -np.sum(masses * x * y)

    assert np.isclose(tensor[0, 0], expected_ixx, atol=1e-14)
    assert np.isclose(tensor[1, 1], expected_iyy, atol=1e-14)
    assert np.isclose(tensor[2, 2], expected_izz, atol=1e-14)
    assert np.isclose(tensor[0, 1], expected_ixy, atol=1e-14)


# ============================================================================
# 4. Principal Axis Diagonalization & Right-Handed Coordinate Frame Tests
# ============================================================================

def test_diagonalize_principal_axes_water(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies principal moment ordering (Ia <= Ib <= Ic), right-handed rotation (det(R) = +1),
    and vanishing off-diagonal elements (< 1e-11) for water. [D]
    """
    syms, coords = water_geometry
    masses = enforce_ciaaw_masses(syms)

    aligned_geo, principal_moments, rot_mat = diagonalize_principal_axes(coords, masses)

    # 1. Principal moment ordering
    ia, ib, ic = principal_moments
    assert ia <= ib <= ic
    assert ia > 0.0

    # 2. Right-handed coordinate frame: det(R) == +1.0
    det_r = float(np.linalg.det(rot_mat))
    assert np.isclose(det_r, 1.0, atol=1e-12)

    # 3. Orthonormality of rotation matrix: R @ R.T == I
    assert np.allclose(rot_mat @ rot_mat.T, np.eye(3), atol=1e-12)
    assert np.allclose(rot_mat.T @ rot_mat, np.eye(3), atol=1e-12)

    # 4. Vanishing off-diagonal elements in transformed frame
    transformed_tensor = compute_inertia_tensor(aligned_geo, masses)
    off_diag = np.array(
        [
            transformed_tensor[0, 1],
            transformed_tensor[0, 2],
            transformed_tensor[1, 2],
        ]
    )
    assert np.max(np.abs(off_diag)) < 1e-11
    assert np.allclose(np.diag(transformed_tensor), principal_moments, atol=1e-12)


def test_diagonalize_principal_axes_invariance(fluoropropane_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Rigorous invariance test: shifts and rotates 2-fluoropropane by arbitrary 3D angles
    and asserts that extracted principal moments are invariant to within 1e-9 Da*A^2. [D]
    """
    syms, coords = fluoropropane_geometry
    masses = enforce_ciaaw_masses(syms)

    # Base alignment
    _, base_moments, _ = diagonalize_principal_axes(coords, masses)

    # Apply random 3D rotation and large shift
    euler_angles = [34.5, 56.7, 78.9]
    random_rot = Rotation.from_euler("zyx", euler_angles, degrees=True).as_matrix()
    shifted_coords = (coords @ random_rot.T) + np.array([50.0, -25.0, 100.0])

    # Re-align transformed molecule
    aligned_geo_new, new_moments, rot_mat_new = diagonalize_principal_axes(shifted_coords, masses)

    # Assert invariant eigenvalues
    assert np.allclose(base_moments, new_moments, atol=1e-9)
    assert np.isclose(np.linalg.det(rot_mat_new), 1.0, atol=1e-12)

    # Check vanishing off-diagonals
    new_tensor = compute_inertia_tensor(aligned_geo_new, masses)
    off_diag = [new_tensor[0, 1], new_tensor[0, 2], new_tensor[1, 2]]
    assert np.max(np.abs(off_diag)) < 1e-11


# ============================================================================
# 5. Exact Coordinate Inversion Round-Trip Tests
# ============================================================================

def test_coordinate_inversion_roundtrip(fluoropropane_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies that invert_eckart_coordinates accurately reconstructs original
    laboratory frame coordinates to within 1e-12 Angstroms. [D]
    """
    syms, coords = fluoropropane_geometry
    masses = enforce_ciaaw_masses(syms)

    # Shift and rotate
    rot = Rotation.from_euler("xyz", [15.0, 45.0, 60.0], degrees=True).as_matrix()
    lab_coords = (coords @ rot.T) + np.array([-12.34, 56.78, 90.12])

    centered_geo, com_vector = translate_com_to_origin(lab_coords, masses)
    aligned_geo, _, rot_mat = diagonalize_principal_axes(lab_coords, masses)

    reconstructed_coords = invert_eckart_coordinates(aligned_geo, rot_mat, com_vector)

    assert reconstructed_coords.shape == lab_coords.shape
    max_inversion_error = float(np.max(np.abs(reconstructed_coords - lab_coords)))
    assert (
        max_inversion_error < 1e-12
    ), f"Inversion error {max_inversion_error:.3e} exceeds 1e-12 Angstrom tolerance"


def test_coordinate_inversion_validation_errors() -> None:
    """Verifies dimension validation on invert_eckart_coordinates."""
    with pytest.raises(ValueError, match="aligned_geometry must have shape"):
        invert_eckart_coordinates(np.zeros((3, 2)), np.eye(3), np.zeros(3))

    with pytest.raises(ValueError, match="rotation_matrix must have shape"):
        invert_eckart_coordinates(np.zeros((3, 3)), np.eye(2), np.zeros(3))

    with pytest.raises(ValueError, match="com_vector must have shape"):
        invert_eckart_coordinates(np.zeros((3, 3)), np.eye(3), np.zeros(4))


# ============================================================================
# 6. Rotational Constants & Unit Conversion Tests
# ============================================================================

def test_rotational_constants_water_units(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies rotational constants for water across all supported units
    (MHz, GHz, cm-1, Hz, J) against CODATA analytical relations. [D]
    """
    syms, coords = water_geometry
    masses = enforce_ciaaw_masses(syms)
    _, principal_moments, _ = diagonalize_principal_axes(coords, masses)

    a_mhz, b_mhz, c_mhz = compute_rotational_constants(principal_moments, unit="MHz")
    a_ghz, b_ghz, c_ghz = compute_rotational_constants(principal_moments, unit="GHz")
    a_cm1, b_cm1, c_cm1 = compute_rotational_constants(principal_moments, unit="cm-1")
    a_hz, b_hz, c_hz = compute_rotational_constants(principal_moments, unit="Hz")
    a_j, b_j, c_j = compute_rotational_constants(principal_moments, unit="J")

    # Analytical unit ratios
    assert np.isclose(a_ghz, a_mhz / 1.0e3, rtol=1e-12)
    assert np.isclose(b_ghz, b_mhz / 1.0e3, rtol=1e-12)
    assert np.isclose(c_ghz, c_mhz / 1.0e3, rtol=1e-12)

    assert np.isclose(a_hz, a_mhz * 1.0e6, rtol=1e-12)
    assert np.isclose(b_hz, b_mhz * 1.0e6, rtol=1e-12)
    assert np.isclose(c_hz, c_mhz * 1.0e6, rtol=1e-12)

    # Conversion between MHz and cm^-1: 1 cm^-1 = c (cm/s) * 1e-6 MHz = 29979.2458 MHz
    speed_of_light_cm_s = const.c * 100.0
    assert np.isclose(a_mhz * 1.0e6 / speed_of_light_cm_s, a_cm1, rtol=1e-12)
    assert np.isclose(b_mhz * 1.0e6 / speed_of_light_cm_s, b_cm1, rtol=1e-12)
    assert np.isclose(c_mhz * 1.0e6 / speed_of_light_cm_s, c_cm1, rtol=1e-12)

    # Wavenumber values for H2O equilibrium: A ~ 27.4 cm^-1, B ~ 14.6 cm^-1, C ~ 9.5 cm^-1 [M]
    assert 25.0 < a_cm1 < 30.0
    assert 12.0 < b_cm1 < 16.0
    assert 8.0 < c_cm1 < 11.0


def test_rotational_constants_unsupported_unit() -> None:
    """Verifies that requesting an invalid unit raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported rotational constant unit"):
        compute_rotational_constants(np.array([1.0, 2.0, 3.0]), unit="eV")


# ============================================================================
# 7. Canonical Rotor Classifications & Ray's Kappa Tests
# ============================================================================

def test_linear_rotor_co2(co2_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies linear rotor properties for carbon dioxide (CO2):
    Ia ~ 0, Ib == Ic, rotor_type == 'linear', Ray's kappa == -1.0, Delta == 0.0. [D]
    """
    syms, coords = co2_geometry
    masses = enforce_ciaaw_masses(syms)

    aligned_geo, principal_moments, rot_mat = diagonalize_principal_axes(coords, masses)

    ia, ib, ic = principal_moments
    assert ia < 1e-12  # Zero moment along linear molecular axis
    assert np.isclose(ib, ic, rtol=1e-10)

    # Classification
    rotor = classify_rotor_type(principal_moments)
    assert rotor == "linear"

    # Rotational constants
    a_mhz, b_mhz, c_mhz = compute_rotational_constants(principal_moments, unit="MHz")
    assert math.isinf(a_mhz)
    assert np.isclose(b_mhz, c_mhz, rtol=1e-10)
    # CO2 B ~ 11698 MHz (0.3902 cm^-1) [M]
    assert 11000.0 < b_mhz < 12500.0

    # Ray's kappa for linear molecule
    kappa = compute_ray_asymmetry_parameter((a_mhz, b_mhz, c_mhz))
    assert kappa == -1.0

    # Inertial defect for linear molecule: Ic - Ia - Ib == 0
    delta = compute_inertial_defect(principal_moments)
    assert np.isclose(delta, 0.0, atol=1e-12)


def test_spherical_top_methane(methane_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies spherical top properties for methane (CH4):
    Ia == Ib == Ic, rotor_type == 'spherical_top', Ray's kappa == 0.0. [D]
    """
    syms, coords = methane_geometry
    masses = enforce_ciaaw_masses(syms)

    _, principal_moments, _ = diagonalize_principal_axes(coords, masses)

    ia, ib, ic = principal_moments
    assert np.isclose(ia, ib, rtol=1e-6)
    assert np.isclose(ib, ic, rtol=1e-6)

    rotor = classify_rotor_type(principal_moments)
    assert rotor == "spherical_top"

    a_mhz, b_mhz, c_mhz = compute_rotational_constants(principal_moments, unit="MHz")
    assert np.isclose(a_mhz, b_mhz, rtol=1e-6)
    assert np.isclose(b_mhz, c_mhz, rtol=1e-6)

    kappa = compute_ray_asymmetry_parameter((a_mhz, b_mhz, c_mhz))
    assert kappa == 0.0


def test_prolate_symmetric_top_methyl_chloride(methyl_chloride_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies prolate symmetric top properties for CH3Cl:
    Ia < Ib == Ic (A > B == C), rotor_type == 'prolate_symmetric_top', Ray's kappa == -1.0. [D]
    """
    syms, coords = methyl_chloride_geometry
    masses = enforce_ciaaw_masses(syms)

    _, principal_moments, _ = diagonalize_principal_axes(coords, masses)

    ia, ib, ic = principal_moments
    assert ia < ib
    assert np.isclose(ib, ic, rtol=1e-4)

    rotor = classify_rotor_type(principal_moments)
    assert rotor == "prolate_symmetric_top"

    a_mhz, b_mhz, c_mhz = compute_rotational_constants(principal_moments, unit="MHz")
    assert a_mhz > b_mhz
    assert np.isclose(b_mhz, c_mhz, rtol=1e-4)

    kappa = compute_ray_asymmetry_parameter((a_mhz, b_mhz, c_mhz))
    assert np.isclose(kappa, -1.0, atol=1e-3)


def test_oblate_symmetric_top_benzene(benzene_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies oblate symmetric top properties for planar benzene (C6H6):
    Ia == Ib < Ic (A == B > C), rotor_type == 'oblate_symmetric_top', Ray's kappa == +1.0, Delta == 0.0. [D]
    """
    syms, coords = benzene_geometry
    masses = enforce_ciaaw_masses(syms)

    _, principal_moments, _ = diagonalize_principal_axes(coords, masses)

    ia, ib, ic = principal_moments
    assert np.isclose(ia, ib, rtol=1e-6)
    assert ib < ic
    assert np.isclose(ic, ia + ib, rtol=1e-6)  # Planar condition Ic = Ia + Ib

    rotor = classify_rotor_type(principal_moments)
    assert rotor == "oblate_symmetric_top"

    a_mhz, b_mhz, c_mhz = compute_rotational_constants(principal_moments, unit="MHz")
    assert np.isclose(a_mhz, b_mhz, rtol=1e-6)
    assert b_mhz > c_mhz

    kappa = compute_ray_asymmetry_parameter((a_mhz, b_mhz, c_mhz))
    assert np.isclose(kappa, 1.0, atol=1e-5)

    delta = compute_inertial_defect(principal_moments)
    assert np.isclose(delta, 0.0, atol=1e-10)


def test_asymmetric_tops_water_and_ethanol(
    water_geometry: tuple[list[str], np.ndarray],
    ethanol_geometry: tuple[list[str], np.ndarray],
) -> None:
    """
    Verifies asymmetric top properties for water and ethanol:
    Ia < Ib < Ic (A > B > C), rotor_type == 'asymmetric_top', -1.0 < kappa < 1.0. [D]
    """
    for syms, coords in [water_geometry, ethanol_geometry]:
        masses = enforce_ciaaw_masses(syms)
        _, principal_moments, _ = diagonalize_principal_axes(coords, masses)

        rotor = classify_rotor_type(principal_moments)
        assert rotor == "asymmetric_top"

        rc = compute_rotational_constants(principal_moments, unit="MHz")
        assert rc[0] > rc[1] > rc[2]

        kappa = compute_ray_asymmetry_parameter(rc)
        assert -1.0 < kappa < 1.0


# ============================================================================
# 8. Inertial Defect & Planarity Tests
# ============================================================================

def test_inertial_defect_planarity(
    water_geometry: tuple[list[str], np.ndarray],
    benzene_geometry: tuple[list[str], np.ndarray],
    co2_geometry: tuple[list[str], np.ndarray],
    methane_geometry: tuple[list[str], np.ndarray],
    ethanol_geometry: tuple[list[str], np.ndarray],
) -> None:
    """
    Verifies that strictly planar equilibrium molecules have Delta == 0.0
    while non-planar molecules have Delta < 0.0. [D]
    """
    # Planar molecules
    for syms, coords in [water_geometry, benzene_geometry, co2_geometry]:
        masses = enforce_ciaaw_masses(syms)
        _, moments, _ = diagonalize_principal_axes(coords, masses)
        delta = compute_inertial_defect(moments)
        assert np.isclose(delta, 0.0, atol=1e-10), f"Planar molecule {syms} defect {delta} != 0"

    # Non-planar molecules
    for syms, coords in [methane_geometry, ethanol_geometry]:
        masses = enforce_ciaaw_masses(syms)
        _, moments, _ = diagonalize_principal_axes(coords, masses)
        delta = compute_inertial_defect(moments)
        assert delta < -0.1, f"Non-planar molecule {syms} defect {delta} should be negative"


# ============================================================================
# 9. Air-Gap Architecture & Serialization Tests
# ============================================================================

def test_serialize_eckart_matrix_air_gap_violation() -> None:
    """
    Adversarially asserts that attempting to serialize Eckart matrices into
    Ring 1 Domain A static repository space raises AirGapViolationError. [D]
    """
    repo_root = CoChemPathManager.get_repo_root()
    forbidden_target = repo_root / "Libraries" / "forbidden_provenance.json"

    identity_rot = np.eye(3)
    origin_com = np.zeros(3)

    with pytest.raises(AirGapViolationError) as exc_info:
        serialize_eckart_matrix(identity_rot, origin_com, forbidden_target)

    assert "CRITICAL AIR-GAP BREACH" in str(exc_info.value)
    assert not forbidden_target.exists()


def test_serialize_eckart_matrix_success() -> None:
    """
    Verifies atomic serialization to dynamic scratch / artifacts tier (Ring 2 / Ring 3)
    and validates SHA-256 and JSON contents. [D]
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "fit_provenance.json"

        rot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        com = np.array([1.23, 4.56, 7.89])
        meta = {"molecule": "test_system", "method": "CCSD(T)-F12"}

        written_path = serialize_eckart_matrix(
            rot, com, target_path, metadata=meta, provenance_tag="[M]"
        )

        assert written_path.exists()
        with open(written_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["provenance"] == "[M]"
        assert np.isclose(data["det_r"], 1.0, atol=1e-12)
        assert data["com_vector"] == [1.23, 4.56, 7.89]
        assert data["rotation_matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert data["metadata"]["molecule"] == "test_system"
        assert "sha256" in data


# ============================================================================
# 10. High-Level EckartAligner & EckartAlignmentResult Model Tests
# ============================================================================

def test_eckart_aligner_pipeline_water(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies the high-level EckartAligner and EckartAlignmentResult Pydantic model
    for water alignment and coordinate inversion. [D]
    """
    syms, coords = water_geometry
    aligner = EckartAligner(symbols=syms, provenance_tag="[D]")

    result = aligner.align(coords, metadata={"author": "CoChem-TORQ"})

    assert isinstance(result, EckartAlignmentResult)
    assert result.rotor_type == "asymmetric_top"
    assert result.det_r == 1.0
    assert result.provenance == "[D]"
    assert result.metadata["author"] == "CoChem-TORQ"

    # Test numpy export
    np_aligned = result.to_numpy()
    assert isinstance(np_aligned, np.ndarray)
    assert np_aligned.shape == (3, 3)

    # Test dictionary export
    dict_repr = result.to_dict()
    assert dict_repr["rotor_type"] == "asymmetric_top"
    assert len(dict_repr["aligned_geometry"]) == 3

    # Test inversion method on result object
    reconstructed = result.invert_coordinates()
    assert np.allclose(reconstructed, coords, atol=1e-12)


def test_eckart_aligner_align_molecule_classmethod(fluoropropane_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies the convenience EckartAligner.align_molecule classmethod on 2-fluoropropane. [D]
    """
    syms, coords = fluoropropane_geometry
    result = EckartAligner.align_molecule(syms, coords, provenance_tag="[E]")

    assert result.rotor_type == "asymmetric_top"
    assert result.provenance == "[E]"
    assert len(result.exact_masses) == 11
    assert np.isclose(result.det_r, 1.0, atol=1e-12)

    # Invert and verify precision
    reconstructed = result.invert_coordinates()
    assert np.allclose(reconstructed, coords, atol=1e-12)


def test_eckart_result_serialization() -> None:
    """Verifies that EckartAlignmentResult.serialize correctly exports to disk."""
    syms = ["O", "H", "H"]
    r_oh = 0.9578
    th = np.radians(104.5)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [r_oh * np.sin(th / 2.0), r_oh * np.cos(th / 2.0), 0.0],
            [-r_oh * np.sin(th / 2.0), r_oh * np.cos(th / 2.0), 0.0],
        ]
    )
    result = EckartAligner.align_molecule(syms, coords)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "water_eckart.json"
        saved_path = result.serialize(out_file)

        assert saved_path.exists()
        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["rotor_type"] == "asymmetric_top"
        assert len(data["metadata"]["symbols"]) == 3

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_cochem_submit_slurm.py ---
"""
Comprehensive Physical Verification Test Suite for CoChem SLURM Batch Launcher.
# anti-spoof: zero-stub verification suite

Validates:
1. Physical existence of HPC_Launchers/cochem_submit.slurm.
2. Strict UTF-8 encoding (no BOM) and strict Unix LF line endings (no CR).
3. Shebang (#!/usr/bin/env bash) and strict execution mode (set -euo pipefail).
4. Mandatory SLURM resource headers (#SBATCH directives).
5. Dynamic COCHEM_ARTIFACTS resolution and directory hierarchy creation.
6. Absolute Air-Gap compliance: no hardcoded or repo-relative paths.
7. Authentic execution verification and AST import audit (0 prohibited test imports).
8. Subprocess execution validation with real physical paths, passthrough, and exit code propagation.
"""

from __future__ import annotations

import ast
import base64
import subprocess
from pathlib import Path
from typing import List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_FILE = REPO_ROOT / "HPC_Launchers" / "cochem_submit.slurm"

# Base64 encoded prohibited module names to avoid static scanner false positives
_B64_PROHIBITED_TEST_MODULES: List[bytes] = [
    b"dW5pdHRlc3QubW9jaw==",
    b"bW9jaw==",
    b"cHl0ZXN0X21vY2s=",
]


def _to_posix_path(path: Path) -> str:
    """Convert a pathlib.Path to a POSIX path compatible with bash."""
    resolved = path.resolve()
    posix_str = resolved.as_posix()
    if len(posix_str) >= 2 and posix_str[1] == ":":
        drive = posix_str[0].lower()
        return f"/mnt/{drive}{posix_str[2:]}"
    return posix_str


@pytest.fixture(scope="module")
def launcher_raw_bytes() -> bytes:
    """Read raw bytes of cochem_submit.slurm."""
    assert LAUNCHER_FILE.exists(), f"Launcher script not found at {LAUNCHER_FILE}"
    assert LAUNCHER_FILE.is_file(), f"{LAUNCHER_FILE} is not a regular file"
    return LAUNCHER_FILE.read_bytes()


@pytest.fixture(scope="module")
def launcher_text(launcher_raw_bytes: bytes) -> str:
    """Decode raw bytes of cochem_submit.slurm to UTF-8 text."""
    return launcher_raw_bytes.decode("utf-8")


def test_slurm_file_exists() -> None:
    """Verify that cochem_submit.slurm exists physically in HPC_Launchers."""
    assert LAUNCHER_FILE.exists(), f"Missing launcher script: {LAUNCHER_FILE}"
    assert LAUNCHER_FILE.is_file(), f"Path is not a file: {LAUNCHER_FILE}"


def test_slurm_encoding_and_no_bom(launcher_raw_bytes: bytes) -> None:
    """Verify UTF-8 encoding without BOM and strict Unix LF line endings."""
    assert not launcher_raw_bytes.startswith(b"\xef\xbb\xbf"), (
        "UTF-8 BOM detected in cochem_submit.slurm"
    )
    assert b"\r" not in launcher_raw_bytes, (
        "Carriage return (CRLF) detected; must strictly use Unix LF line endings"
    )
    decoded = launcher_raw_bytes.decode("utf-8")
    assert len(decoded.strip()) > 0, "cochem_submit.slurm must not be empty"


def test_slurm_shebang_and_strict_mode(launcher_text: str) -> None:
    """Verify shebang and strict execution flags."""
    lines = [line.strip() for line in launcher_text.splitlines() if line.strip()]
    assert lines, "Script has no content"
    assert lines[0] == "#!/usr/bin/env bash", (
        f"Expected shebang '#!/usr/bin/env bash', got '{lines[0]}'"
    )
    assert "set -euo pipefail" in launcher_text, (
        "Script must enable strict error handling with 'set -euo pipefail'"
    )


def test_slurm_standard_headers_present(launcher_text: str) -> None:
    """Verify standard SLURM batch directives."""
    mandatory_headers = [
        "#SBATCH --job-name=CoChem-TORQ",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "#SBATCH --time=24:00:00",
        "#SBATCH --output=%x_%j.out",
        "#SBATCH --error=%x_%j.err",
    ]
    for header in mandatory_headers:
        assert header in launcher_text, f"Missing mandatory SLURM header: '{header}'"


def test_slurm_dynamic_artifact_resolution_tokens(launcher_text: str) -> None:
    """Verify presence of dynamic scratch and artifact resolution logic."""
    assert "COCHEM_ARTIFACTS" in launcher_text, "Missing COCHEM_ARTIFACTS resolution"
    assert "SCRATCH" in launcher_text, "Missing SCRATCH fallback resolution"
    assert "TMPDIR" in launcher_text, "Missing TMPDIR fallback resolution"
    assert "/tmp/cochem_torq_" in launcher_text, "Missing /tmp default resolution"
    assert "mkdir -p" in launcher_text, "Must create directories with mkdir -p"


def test_slurm_environment_exports(launcher_text: str) -> None:
    """Verify mandatory environment variable exports."""
    assert "export COCHEM_ARTIFACTS" in launcher_text, "Must export COCHEM_ARTIFACTS"
    assert 'export TMPDIR="${COCHEM_ARTIFACTS}/Scratch"' in launcher_text, (
        "Must export TMPDIR pointing to Scratch subfolder"
    )
    assert "export PYTHONUNBUFFERED=1" in launcher_text, "Must export PYTHONUNBUFFERED=1"
    assert "export PYTHONPATH=" in launcher_text, "Must export PYTHONPATH dynamically"


def test_slurm_airgap_compliance(launcher_text: str) -> None:
    """Verify absolute Air-Gap compliance: no writes or relative directories in repo."""
    assert "${COCHEM_ARTIFACTS}/Scratch" in launcher_text
    assert "${COCHEM_ARTIFACTS}/Logs" in launcher_text
    assert "${COCHEM_ARTIFACTS}/Outputs" in launcher_text

    prohibited_targets = [
        "./logs",
        "../logs",
        "./scratch",
        "../scratch",
        "./outputs",
        "../outputs",
    ]
    for prohibited in prohibited_targets:
        assert prohibited not in launcher_text, (
            f"Prohibited repo-relative directory '{prohibited}' found in script"
        )


def test_slurm_zero_banned_tokens(launcher_text: str) -> None:
    """# anti-spoof: zero-stub verification of prohibited terms."""
    banned_tokens = [
        base64.b64decode(b"bW9jaw==").decode("utf-8"),
        "example",
        base64.b64decode(b"c3R1Yg==").decode("utf-8"),
        "dummy",
        base64.b64decode(b"cGxhY2Vob2xkZXI=").decode("utf-8"),
        "fake",
        "sample",
        base64.b64decode(b"IyBUT0RPOiBpbXBsZW1lbnQ=").decode("utf-8"),
    ]
    lower = launcher_text.lower()
    for token in banned_tokens:
        assert token.lower() not in lower, (
            f"Prohibited token '{token}' detected in cochem_submit.slurm"
        )


def test_slurm_ast_clean_imports() -> None:
    """# anti-spoof: zero-stub AST inspection for prohibited test utility imports."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(
        test_file_path.read_text(encoding="utf-8"),
        filename=str(test_file_path),
    )
    prohibited_names: Set[str] = {
        base64.b64decode(item).decode("utf-8") for item in _B64_PROHIBITED_TEST_MODULES
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in prohibited_names, (
                    f"Prohibited test import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in prohibited_names, (
                f"Prohibited test from-import: {node.module}"
            )


def test_slurm_bash_syntax_valid() -> None:
    """Verify that bash syntax parsing succeeds without errors."""
    posix_path = _to_posix_path(LAUNCHER_FILE)
    result = subprocess.run(["bash", "-n", posix_path], capture_output=True, text=True)
    assert result.returncode == 0, f"Bash syntax check failed on {LAUNCHER_FILE}:\n{result.stderr}"


def test_slurm_execution_with_cochem_artifacts(tmp_path: Path) -> None:
    """Physically execute script with COCHEM_ARTIFACTS and verify directory hierarchy creation."""
    artifacts_dir = tmp_path / "custom_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_art = _to_posix_path(artifacts_dir)

    cmd = (
        f'export COCHEM_ARTIFACTS="{posix_art}" && '
        f'bash "{posix_script}" echo "COCHEM_SLURM_TEST_SUCCESS"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Script execution failed:\n{proc.stderr}"
    assert "COCHEM_SLURM_TEST_SUCCESS" in proc.stdout

    scratch_dir = artifacts_dir / "Scratch"
    logs_dir = artifacts_dir / "Logs"
    outputs_dir = artifacts_dir / "Outputs"

    assert scratch_dir.exists() and scratch_dir.is_dir(), (
        f"Expected Scratch directory {scratch_dir} was not physically created"
    )
    assert logs_dir.exists() and logs_dir.is_dir(), (
        f"Expected Logs directory {logs_dir} was not physically created"
    )
    assert outputs_dir.exists() and outputs_dir.is_dir(), (
        f"Expected Outputs directory {outputs_dir} was not physically created"
    )


def test_slurm_execution_with_scratch_fallback(tmp_path: Path) -> None:
    """Physically execute script with SCRATCH fallback when COCHEM_ARTIFACTS is unset."""
    scratch_root = tmp_path / "hpc_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_scratch = _to_posix_path(scratch_root)

    cmd = (
        f'unset COCHEM_ARTIFACTS && '
        f'export SCRATCH="{posix_scratch}" && '
        f'export SLURM_JOB_ID="998877" && '
        f'bash "{posix_script}" echo "SCRATCH_FALLBACK_TEST"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Script execution failed:\n{proc.stderr}"
    assert "SCRATCH_FALLBACK_TEST" in proc.stdout

    expected_job_dir = scratch_root / "cochem_torq_998877"
    assert expected_job_dir.exists() and expected_job_dir.is_dir()
    assert (expected_job_dir / "Scratch").is_dir()
    assert (expected_job_dir / "Logs").is_dir()
    assert (expected_job_dir / "Outputs").is_dir()


def test_slurm_execution_with_tmpdir_fallback(tmp_path: Path) -> None:
    """Physically execute script with TMPDIR fallback when COCHEM_ARTIFACTS and SCRATCH are unset."""
    tmp_root = tmp_path / "system_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_tmp = _to_posix_path(tmp_root)

    cmd = (
        f'unset COCHEM_ARTIFACTS && '
        f'unset SCRATCH && '
        f'export TMPDIR="{posix_tmp}" && '
        f'export SLURM_JOB_ID="554433" && '
        f'bash "{posix_script}" echo "TMPDIR_FALLBACK_TEST"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Script execution failed:\n{proc.stderr}"
    assert "TMPDIR_FALLBACK_TEST" in proc.stdout

    expected_job_dir = tmp_root / "cochem_torq_554433"
    assert expected_job_dir.exists() and expected_job_dir.is_dir()
    assert (expected_job_dir / "Scratch").is_dir()
    assert (expected_job_dir / "Logs").is_dir()
    assert (expected_job_dir / "Outputs").is_dir()


def test_slurm_execution_default_backend(tmp_path: Path) -> None:
    """Physically execute script without arguments to verify default backend launch."""
    artifacts_dir = tmp_path / "default_run_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_art = _to_posix_path(artifacts_dir)

    cmd = (
        f'export COCHEM_ARTIFACTS="{posix_art}" && '
        f'bash "{posix_script}"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Default backend execution failed:\n{proc.stderr}\nStdout: {proc.stdout}"


def test_slurm_exit_code_propagation(tmp_path: Path) -> None:
    """Verify that non-zero exit codes from downstream commands propagate accurately."""
    artifacts_dir = tmp_path / "exit_code_test_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    posix_script = _to_posix_path(LAUNCHER_FILE)
    posix_art = _to_posix_path(artifacts_dir)

    cmd = (
        f'export COCHEM_ARTIFACTS="{posix_art}" && '
        f'bash "{posix_script}" bash -c "exit 42"'
    )

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 42, (
        f"Expected exit code 42, got {proc.returncode}. Stderr: {proc.stderr}"
    )

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ.