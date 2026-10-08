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

def _real_finite_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    objects = np.asarray(value, dtype=object)
    if np.iscomplexobj(raw) or any(
        isinstance(item, (bool, np.bool_)) for item in objects.flat
    ):
        raise ValueError(
            f"{name} must contain finite real numbers, not complex or boolean values."
        )
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite real numbers.")
    if name == "Masses" and np.any(result <= 0):
        raise ValueError("Individual masses must be strictly positive.")
    return result


def enforce_ciaaw_masses(symbols: Sequence[str]) -> np.ndarray:
    """Resolve tabulated isotope masses using the shared explicit isotope policy.

    The historical function name is retained for compatibility. The installed
    Mendeleev table is the actual provider; these measured masses have uncertainty
    and are not asserted to be exact constants or independently verified CIAAW
    reference values. Missing natural abundance requires an explicit isotope.
    """
    from Libraries.cochem_isotopes import isotope_mass

    masses = []
    for symbol in symbols:
        clean = symbol.strip()
        if not clean:
            raise ValueError("Empty or blank atomic symbol provided.")
        if clean.upper() in ("D", "T"):
            clean = clean.upper()
        try:
            mass = isotope_mass(clean)
        except ValueError as error:
            raise ValueError(
                f"Unrecognized or invalid atomic symbol {symbol!r}: {error}"
            ) from error
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"Tabulated isotope mass is not finite and positive: {symbol!r}")
        masses.append(mass)
    return np.asarray(masses, dtype=np.float64)


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
    geo = _real_finite_array(geometry, "Geometry")
    masses = _real_finite_array(exact_masses, "Masses")

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
    geo = _real_finite_array(geometry, "Geometry")
    masses = _real_finite_array(exact_masses, "Masses")

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
    masses = _real_finite_array(exact_masses, "Masses")

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
) -> float | None:
    """
    Computes Ray's asymmetry parameter (kappa) describing the degree of molecular asymmetry.

    Formula:
    kappa = (2 * B - A - C) / (A - C)

    Limiting values:
    - Prolate symmetric top limit: kappa = -1.0 (B = C)
    - Oblate symmetric top limit: kappa = +1.0 (A = B)
    - Most asymmetric top: kappa = 0.0 (B = (A + C) / 2)
    - Linear molecule (A -> inf, B = C): kappa = -1.0
    - Spherical top (A = B = C): undefined (None).

    :param rotational_constants: 3-element tuple or array of (A, B, C).
    :return: Ray's kappa in [-1.0, 1.0], or None for a spherical top.
    """
    rc = [float(v) for v in rotational_constants]
    if len(rc) != 3:
        raise ValueError(f"rotational_constants must have 3 elements, got {len(rc)}")

    a, b, c = rc[0], rc[1], rc[2]

    if any(math.isnan(value) or value <= 0 for value in rc):
        raise ValueError("Rotational constants must be positive and cannot contain NaN.")
    if math.isinf(a):
        if a > 0 and math.isfinite(b) and math.isfinite(c) and b == c:
            return -1.0  # Explicit analytic linear-rotor limit, not a finite A.
        raise ValueError("The linear-rotor limit requires positive finite B = C.")
    if not all(math.isfinite(value) for value in rc) or not a >= b >= c:
        raise ValueError("Finite rotational constants must be ordered A >= B >= C.")
    if a == c:
        return None
    # This algebraic form avoids overflowing 2*B and preserves actual finite
    # near-spherical and very-large-A values instead of substituting a limit.
    denominator = a - c
    return float((b - c) / denominator - (a - b) / denominator)


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
    ray_kappa: float | None = Field(
        ..., description="Ray's asymmetry parameter kappa = (2B - A - C) / (A - C)."
    )
    inertial_defect: float = Field(
        ..., description="Inertial defect Delta = Ic - Ia - Ib in Da * A^2."
    )
    rotor_type: str = Field(
        ..., description="Canonical spectroscopic rotor classification."
    )
    symbols: list[str] | None = Field(
        ..., description="Supplied atomic symbols, or None when only masses were supplied."
    )
    exact_masses: list[float] = Field(
        ..., description="Masses actually used in Da; their source is retained in metadata."
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
        self._caller_supplied_masses = masses is not None
        if masses is not None:
            self.masses = _real_finite_array(masses, "Masses")
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
        geo_arr = _real_finite_array(geometry, "Geometry")
        if geo_arr.ndim != 2 or geo_arr.shape[1] != 3:
            raise ValueError(f"Geometry must have shape (N, 3), got {geo_arr.shape}")

        syms = [s.strip() for s in symbols] if symbols is not None else self.symbols
        if exact_masses is not None:
            mass_arr = _real_finite_array(exact_masses, "Masses")
            mass_source = "caller_supplied"
        elif symbols is not None:
            mass_arr = enforce_ciaaw_masses(syms)
            mass_source = "Mendeleev_isotope_database"
        elif self.masses is not None:
            mass_arr = self.masses
            mass_source = (
                "caller_supplied" if self._caller_supplied_masses
                else "Mendeleev_isotope_database"
            )
        elif syms is not None:
            mass_arr = enforce_ciaaw_masses(syms)
            mass_source = "Mendeleev_isotope_database"
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
        from Libraries.cochem_isotopes import isotope_record

        isotope_records = (
            [
                isotope_record(symbol.upper() if symbol.upper() in ("D", "T") else symbol)
                for symbol in syms
            ] if mass_source == "Mendeleev_isotope_database" else None
        )
        if isotope_records is not None and not np.array_equal(
            mass_arr, [record["mass_u"] for record in isotope_records]
        ):
            raise ValueError("Used masses differ from declared isotope database records.")
        factual_metadata = {
            **(metadata or {}),
            "mass_source": mass_source,
            "isotope_records": isotope_records,
            "symbols_available": syms is not None,
            "ray_kappa_status": (
                "undefined_for_spherical_top" if kappa is None
                else "linear_rotor_limit" if math.isinf(rc_mhz[0])
                else "derived_from_finite_rotational_constants"
            ),
        }

        return EckartAlignmentResult(
            aligned_geometry=[[float(v) for v in row] for row in aligned_geo],
            principal_moments=(float(principal_moments[0]), float(principal_moments[1]), float(principal_moments[2])),
            rotational_constants_mhz=rc_mhz,
            rotational_constants_ghz=rc_ghz,
            rotational_constants_cm1=rc_cm1,
            rotation_matrix=[[float(v) for v in row] for row in rotation_matrix],
            com_vector=(float(com_vector[0]), float(com_vector[1]), float(com_vector[2])),
            ray_kappa=kappa,
            inertial_defect=float(delta),
            rotor_type=rotor_type,
            symbols=syms,
            exact_masses=[float(m) for m in mass_arr],
            det_r=det_r,
            provenance=self.provenance_tag,
            metadata=factual_metadata,
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
