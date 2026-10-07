"""CoChem-TORQ: Quantum Tensor Harvester & Inertial Provenance Engine.

Phase 6 (Stage 4.1) Core Module
-------------------------------------------------------------------------------
Mathematically derives the Inertia Tensor, Principal Moments of Inertia
(Ia <= Ib <= Ic), and Rotational Constants (A, B, C in MHz and GHz) from
optimized Cartesian coordinates.
Implements:
1. Exact CODATA 2022 physical constants and CIAAW / AME2020 mass tables.
2. Inertia tensor diagonalization, planar moments (P_aa, P_bb, P_cc),
   and inertial defect (Delta).
3. Cartesian Protections (Linearity Trap near 180-degree singularities)
   projecting linear and quasi-linear configurations into 2D cylindrical
   coordinates (z, rho, phi) with rotational degree-of-freedom regularization
   (DOF=2) and singularity damping.
4. Ray's Asymmetry Parameter (kappa) classification and Dynamic Switch
   between standard spectroscopic representations (Ir, Il, IIr, IIl, IIIr, IIIl)
   with Wang Hamiltonian sub-block mapping.
5. ORCA VPT2 parser for Darling-Dennison resonances, Coriolis couplings,
   Watson centrifugal distortion constants, and Raman polarizabilities.
6. HDF5 / JSON structured export gateways with Air-Gap directory compliance.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, cast

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.ipc as pa_ipc  # type: ignore[import-untyped]
import scipy.linalg as sla  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

# Optional blake3 check
try:
    import blake3  # type: ignore[import-untyped]

    _BLAKE3_AVAILABLE = True
except ImportError:
    _BLAKE3_AVAILABLE = False

# Configure module logger
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-Tensor] %(message)s"
)
logger = logging.getLogger("TorqTensorExt")

# Environment artifact directory default
ARTIFACTS_DIR = os.environ.get(
    "COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts")
)

# =============================================================================
# 1. Exact CODATA 2022 Physical Constants & Rotational Conversion Factors
# =============================================================================

CODATA_YEAR: Final[int] = 2022
PLANCK_CONSTANT_JS: Final[float] = 6.62607015e-34  # Exact J * s (SI definition)
SPEED_OF_LIGHT_C: Final[float] = 299792458.0  # Exact m / s (SI definition)
C_M_S: Final[float] = SPEED_OF_LIGHT_C  # Legacy alias
ATOMIC_MASS_CONSTANT_U: Final[float] = 1.66053906892e-27  # Measured kg/u (CODATA 2022)
AMU_TO_KG: Final[float] = ATOMIC_MASS_CONSTANT_U  # Legacy alias
ANGSTROM_TO_M: Final[float] = 1.0e-10  # Exact m

# Rotational conversion factor:
# C_rot = h / (8 * pi^2 * u * 1e-20) * 1e-6 (MHz * u * Angstrom^2)
# C_rot = 505379.0084354078 MHz * u * Angstrom^2
C_ROT_MHZ: Final[float] = (
    PLANCK_CONSTANT_JS
    / (8.0 * (math.pi**2) * ATOMIC_MASS_CONSTANT_U * (ANGSTROM_TO_M**2))
) * 1e-6
AMU_A2_TO_MHZ: Final[float] = C_ROT_MHZ  # Exact conversion factor in MHz * u * A^2
C_ROT_GHZ: Final[float] = C_ROT_MHZ * 1e-3  # GHz * u * A^2
C_ROT_CM1: Final[float] = (C_ROT_MHZ * 1e6) / (
    SPEED_OF_LIGHT_C * 100.0
)  # cm^-1 * u * A^2

# =============================================================================
# 2. CIAAW / AME2020 Exact Mono-Isotopic Mass Tables
# =============================================================================

def get_atomic_mass(symbol: str) -> float:
    """Resolve the requested tabulated isotope without substituting another mass."""
    from Libraries.cochem_isotopes import isotope_mass
    return isotope_mass(symbol)


def is_ghost_atom(symbol: str) -> bool:
    """Checks whether an atomic symbol represents a ghost atom (Z_i = 0, Gh, Ghost, X, Bq, 0).

    :param symbol: Element or particle symbol string.
    :return: True if the symbol is a ghost / dummy atom without mass, False otherwise.
    """
    s = symbol.strip().lower()
    if s in ("gh", "ghost", "x", "bq", "0", "gh0"):
        return True
    if s.startswith("gh:") or s.startswith("gh-") or s.startswith("gh_"):
        return True
    return False


def filter_ghost_atoms(
    coordinates: npt.ArrayLike,
    symbols: list[str],
    masses: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], list[str], npt.NDArray[np.float64], list[int]]:
    """Filter explicitly labelled ghost atoms; invalid physical masses are errors.

    :param coordinates: (N, 3) Cartesian coordinates in Angstroms.
    :param symbols: (N,) atomic symbols.
    :param masses: Optional (N,) atomic masses in u.
    :return: Tuple of (filtered_coordinates, filtered_symbols, filtered_masses, valid_indices).
    """
    coords_arr = np.array(cast(Any, coordinates), dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError(f"Coordinates must have shape (N, 3), got {coords_arr.shape}.")
    n_atoms = coords_arr.shape[0]
    if len(symbols) != n_atoms:
        raise ValueError(f"Symbols length ({len(symbols)}) != coordinates count ({n_atoms}).")

    if masses is not None:
        mass_arr = np.array(cast(Any, masses), dtype=np.float64)
        if mass_arr.shape != (n_atoms,):
            raise ValueError(f"Masses must have shape ({n_atoms},), got {mass_arr.shape}")
    else:
        mass_arr = None

    valid_indices: list[int] = []
    filtered_symbols: list[str] = []
    filtered_masses_list: list[float] = []

    for i in range(n_atoms):
        sym = symbols[i].strip()
        if is_ghost_atom(sym):
            continue
        if mass_arr is not None:
            m = float(mass_arr[i])
            if not math.isfinite(m) or m <= 0.0:
                raise ValueError(f"Physical atom {i} ({sym}) requires a finite positive mass")
        else:
            m = get_atomic_mass(sym)
            if not math.isfinite(m) or m <= 0.0:
                raise ValueError(f"Physical atom {i} ({sym}) requires a finite positive mass")

        valid_indices.append(i)
        filtered_symbols.append(sym)
        filtered_masses_list.append(m)

    if not valid_indices:
        raise ValueError("Cannot process system: all atoms were filtered out as ghost atoms (Z_i = 0 / Gh).")

    filtered_coords = coords_arr[valid_indices]
    filtered_masses = np.array(filtered_masses_list, dtype=np.float64)
    return filtered_coords, filtered_symbols, filtered_masses, valid_indices


# =============================================================================
# 3. Pydantic Structured Data Models
# =============================================================================


class RotationalConstants(BaseModel):
    """Pydantic model for molecular rotational constants across unit systems."""

    model_config = ConfigDict(frozen=True)

    A_MHz: float | None = Field(  # noqa: N815
        default=None, description="A constant in MHz (None for linear)"
    )
    B_MHz: float = Field(..., description="B constant in MHz")  # noqa: N815
    C_MHz: float = Field(..., description="C constant in MHz")  # noqa: N815
    A_GHz: float | None = Field(  # noqa: N815
        default=None, description="A constant in GHz"
    )
    B_GHz: float = Field(..., description="B constant in GHz")  # noqa: N815
    C_GHz: float = Field(..., description="C constant in GHz")  # noqa: N815
    A_cm1: float | None = Field(  # noqa: N815
        default=None, description="A constant in cm^-1"
    )
    B_cm1: float = Field(..., description="B constant in cm^-1")  # noqa: N815
    C_cm1: float = Field(..., description="C constant in cm^-1")  # noqa: N815


class PlanarMoments(BaseModel):
    """Planar moments P_aa = sum m*a^2, P_bb = sum m*b^2, P_cc = sum m*c^2."""

    model_config = ConfigDict(frozen=True)

    P_aa: float = Field(..., description="Planar moment P_aa in u*A^2")  # noqa: N815
    P_bb: float = Field(..., description="Planar moment P_bb in u*A^2")  # noqa: N815
    P_cc: float = Field(..., description="Planar moment P_cc in u*A^2")  # noqa: N815
    units: str = Field(default="u*Angstrom^2")


class InertiaTensorResult(BaseModel):
    """Container for rigorous inertia tensor diagonalization outputs."""

    model_config = ConfigDict(frozen=True)

    inertia_tensor_u_A2: list[list[float]] = Field(  # noqa: N815
        ..., description="3x3 moment of inertia tensor in u*Angstrom^2"
    )
    inertia_tensor_kg_m2: list[list[float]] = Field(
        ..., description="3x3 moment of inertia tensor in kg*m^2"
    )
    principal_moments_u_A2: list[float] = Field(  # noqa: N815
        ..., description="Sorted principal moments [Ia, Ib, Ic] in u*Angstrom^2"
    )
    principal_moments_kg_m2: list[float] = Field(
        ..., description="Sorted principal moments [Ia, Ib, Ic] in kg*m^2"
    )
    principal_axes_matrix: list[list[float]] = Field(
        ..., description="3x3 right-handed eigenvector rotation matrix"
    )
    center_of_mass_A: list[float] = Field(  # noqa: N815
        ..., description="Center of mass in Angstroms"
    )
    total_mass_u: float = Field(..., description="Total molecular mass in u")
    rotational_constants: RotationalConstants
    planar_moments: PlanarMoments
    inertial_defect_u_A2: float = Field(  # noqa: N815
        ..., description="Inertial defect Delta = Ic - Ia - Ib in u*Angstrom^2"
    )
    is_planar: bool = Field(
        ..., description="Planar geometry flag (Delta == 0 within tolerance)"
    )


class CartesianProtectionResult(BaseModel):
    """Container for Cartesian protection and Linearity Trap regularization."""

    model_config = ConfigDict(frozen=True)

    is_linear: bool = Field(..., description="Strict linearity flag")
    is_quasi_linear: bool = Field(..., description="Quasi-linear flag")
    LINEAR_SINGULARITY: bool = Field(
        default=False, description="Flag for linear coordinate singularity collapse (Ia < 1e-6)"
    )
    linear_singularity: bool = Field(
        default=False, description="Alias for LINEAR_SINGULARITY"
    )
    rotational_dof: int = Field(
        ..., description="Rotational DOF (2 for linear, 3 for non-linear)"
    )
    collinear_axis: list[float] = Field(
        ..., description="Unit vector along collinear backbone"
    )
    cylindrical_coordinates: list[dict[str, float]] = Field(
        ..., description="Projected cylindrical coordinates (z, rho, phi)"
    )
    singularity_damping_applied: bool = Field(
        ..., description="Flag indicating if partition singularity was damped"
    )
    damping_factor: float = Field(..., description="Regularization factor applied")
    protected_rotational_constants: RotationalConstants


class AsymmetryResult(BaseModel):
    """Container for Ray's asymmetry parameter and representation choice."""

    model_config = ConfigDict(frozen=True)

    kappa: float | None = Field(..., description="Ray's asymmetry parameter; undefined for spherical tops")
    rotor_type: str = Field(..., description="Top classification")
    recommended_representation: str = Field(
        ..., description="Optimal representation (e.g. Ir, IIIr)"
    )
    axis_mapping: dict[str, str] = Field(
        ..., description="Space/body axis mapping e.g. {'x':'b','y':'c','z':'a'}"
    )
    transformation_matrix: list[list[float]] = Field(
        ..., description="3x3 coordinate permutation matrix"
    )
    wang_subblocks: list[str] = Field(
        ..., description="Wang Hamiltonian sub-blocks [E+, E-, O+, O-]"
    )
    description: str = Field(..., description="Detailed classification notes")


class VPT2Data(BaseModel):
    """Container for ORCA VPT2, Coriolis, and Centrifugal Distortion tensors."""

    model_config = ConfigDict(frozen=True)

    darling_dennison: list[dict[str, Any]] = Field(default_factory=list)
    coriolis_couplings: dict[str, list[float]] = Field(default_factory=dict)
    centrifugal_distortion: dict[str, list[float]] = Field(default_factory=dict)
    raman_polarizability: list[float] = Field(default_factory=list)
    is_divergent: bool = Field(default=False)
    divergence_details: list[str] = Field(default_factory=list)


class TorqTensorOutput(BaseModel):
    """Comprehensive high-level payload for CoChem-TORQ tensor harvesting."""

    model_config = ConfigDict(frozen=True)

    point_id: str
    symbols: list[str]
    coordinates: list[list[float]]
    inertia: InertiaTensorResult
    cartesian_protection: CartesianProtectionResult
    asymmetry: AsymmetryResult
    vpt2: VPT2Data | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# 4. Core Mathematical Engine: Diagonalization & Inertial Invariants
# =============================================================================


def diagonalize_inertia_tensor(
    coordinates: npt.ArrayLike,
    masses: npt.ArrayLike | None = None,
    symbols: list[str] | None = None,
    filter_ghosts: bool = True,
) -> InertiaTensorResult:
    """Computes COM, builds moment of inertia tensor, diagonalizes to principal axes,
    and derives rotational constants, planar moments, and inertial defect.

    Enforces CIAAW mono-isotopic masses, ghost-atom stripping (Z_i = 0 / Gh),
    barycentric Center-of-Mass translation, LAPACK eigh spectral diagonalization
    (Ia <= Ib <= Ic), and SO(3) Right-Handedness Parity Lock (det(R_PA) = +1.0).

    :param coordinates: (N, 3) Cartesian coordinates in Angstroms.
    :param masses: (N,) atomic masses in u (optional if symbols provided).
    :param symbols: (N,) atomic symbols (optional if masses provided).
    :param filter_ghosts: If True, automatically filters out ghost atoms (Gh, Ghost, Z_i=0).
    :return: Rigorous InertiaTensorResult data model.
    """
    coords: npt.NDArray[np.float64] = np.array(
        cast(Any, coordinates), dtype=np.float64
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"Coordinates must have shape (N, 3), got shape {coords.shape}."
        )

    n_atoms = coords.shape[0]
    if symbols is None and masses is None:
        raise ValueError("Either masses or symbols must be supplied.")

    if filter_ghosts and symbols is not None:
        active_coords, active_syms, mass_arr, _ = filter_ghost_atoms(
            coordinates=coords, symbols=symbols, masses=masses
        )
    else:
        active_coords = coords
        if masses is not None:
            mass_arr = np.array(cast(Any, masses), dtype=np.float64)
            if mass_arr.shape != (n_atoms,):
                raise ValueError(f"Masses must have shape ({n_atoms},), got {mass_arr.shape}")
        elif symbols is not None:
            if len(symbols) != n_atoms:
                raise ValueError(
                    f"Symbols length ({len(symbols)}) != atom count ({n_atoms})."
                )
            mass_arr = np.array(
                [get_atomic_mass(sym) for sym in symbols], dtype=np.float64
            )

    if not np.isfinite(active_coords).all() or not np.isfinite(mass_arr).all() or np.any(mass_arr <= 0):
        raise ValueError("Finite coordinates and strictly positive physical masses are required")
    total_mass = float(np.sum(mass_arr))
    if total_mass <= 0.0:
        raise ValueError("Total molecular mass must be strictly positive.")

    # 1. Barycentric Shift to Center of Mass (COM)
    com = np.sum(active_coords * mass_arr[:, None], axis=0) / total_mass
    rel_coords = active_coords - com

    x = rel_coords[:, 0]
    y = rel_coords[:, 1]
    z = rel_coords[:, 2]

    # 2. Construct 3x3 Moment of Inertia Tensor I
    i_xx = float(np.sum(mass_arr * (y**2 + z**2)))
    i_yy = float(np.sum(mass_arr * (x**2 + z**2)))
    i_zz = float(np.sum(mass_arr * (x**2 + y**2)))
    i_xy = -float(np.sum(mass_arr * x * y))
    i_xz = -float(np.sum(mass_arr * x * z))
    i_yz = -float(np.sum(mass_arr * y * z))

    inertia_tensor_u_a2 = np.array(
        [[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]], dtype=np.float64
    )

    # 3. LAPACK eigh Spectral Diagonalization (Ia <= Ib <= Ic)
    evals, evecs = sla.eigh(inertia_tensor_u_a2)

    # Sort eigenvalues ascending: Ia <= Ib <= Ic
    idx = np.argsort(evals)
    evals_sorted = evals[idx]
    evecs_sorted = evecs[:, idx].copy()

    # SO(3) Right-Handedness Parity Lock: det(R_PA) == +1.0
    if float(sla.det(evecs_sorted)) < 0.0:
        # Enforce right-handed coordinate frame: v_c = v_a x v_b
        evecs_sorted[:, 2] = np.cross(evecs_sorted[:, 0], evecs_sorted[:, 1])
        norm_c = float(sla.norm(evecs_sorted[:, 2]))
        if norm_c > 1e-12:
            evecs_sorted[:, 2] /= norm_c

    # Convert to kg * m^2
    evals_kg_m2 = evals_sorted * ATOMIC_MASS_CONSTANT_U * (ANGSTROM_TO_M**2)
    inertia_tensor_kg_m2 = (
        inertia_tensor_u_a2 * ATOMIC_MASS_CONSTANT_U * (ANGSTROM_TO_M**2)
    )

    i_a = float(evals_sorted[0])
    i_b = float(evals_sorted[1])
    i_c = float(evals_sorted[2])

    # 4. Planar Moments of Inertia: P_aa, P_bb, P_cc
    # P_aa = 0.5 * (Ib + Ic - Ia) = sum m * a^2
    # P_bb = 0.5 * (Ia + Ic - Ib) = sum m * b^2
    # P_cc = 0.5 * (Ia + Ib - Ic) = sum m * c^2
    p_aa = float(0.5 * (i_b + i_c - i_a))
    p_bb = float(0.5 * (i_a + i_c - i_b))
    p_cc = float(0.5 * (i_a + i_b - i_c))

    # 5. Inertial Defect: Delta = Ic - Ia - Ib
    # In planar systems: Delta == 0.0 exactly at equilibrium.
    delta = float(i_c - i_a - i_b)
    is_planar = bool(abs(delta) < 1e-4 or abs(p_cc) < 1e-4)

    # 6. Rotational Constants A, B, C (MHz, GHz, cm^-1) via NIST CODATA 2022
    # Only numerical zero is singular; a finite near-linear moment retains A.
    moment_roundoff = 100 * np.finfo(np.float64).eps * max(abs(i_b), abs(i_c))
    if i_a < -moment_roundoff:
        raise ValueError("Inertia tensor has a materially negative principal moment")
    is_linear_singularity = abs(i_a) <= moment_roundoff
    if is_linear_singularity:
        a_mhz = None
        a_ghz = None
        a_cm1 = None
    else:
        a_mhz = float(C_ROT_MHZ / i_a)
        a_ghz = float(C_ROT_GHZ / i_a)
        a_cm1 = float(C_ROT_CM1 / i_a)

    if i_b <= 1e-12 or i_c <= 1e-12:
        raise ValueError("Rotational constants are undefined for a point/monatomic geometry")
    b_mhz = float(C_ROT_MHZ / i_b)
    b_ghz = float(C_ROT_GHZ / i_b)
    b_cm1 = float(C_ROT_CM1 / i_b)

    c_mhz = float(C_ROT_MHZ / i_c)
    c_ghz = float(C_ROT_GHZ / i_c)
    c_cm1 = float(C_ROT_CM1 / i_c)

    rot_consts = RotationalConstants(
        A_MHz=a_mhz,
        B_MHz=b_mhz,
        C_MHz=c_mhz,
        A_GHz=a_ghz,
        B_GHz=b_ghz,
        C_GHz=c_ghz,
        A_cm1=a_cm1,
        B_cm1=b_cm1,
        C_cm1=c_cm1,
    )

    planar_moments = PlanarMoments(P_aa=p_aa, P_bb=p_bb, P_cc=p_cc)

    return InertiaTensorResult(
        inertia_tensor_u_A2=cast(list[list[float]], inertia_tensor_u_a2.tolist()),
        inertia_tensor_kg_m2=cast(list[list[float]], inertia_tensor_kg_m2.tolist()),
        principal_moments_u_A2=[i_a, i_b, i_c],
        principal_moments_kg_m2=cast(list[float], evals_kg_m2.tolist()),
        principal_axes_matrix=cast(list[list[float]], evecs_sorted.tolist()),
        center_of_mass_A=cast(list[float], com.tolist()),
        total_mass_u=total_mass,
        rotational_constants=rot_consts,
        planar_moments=planar_moments,
        inertial_defect_u_A2=delta,
        is_planar=is_planar,
    )


# =============================================================================
# 5. Cartesian Protections (Linearity Trap & Singularity Regularization)
# =============================================================================


def apply_cartesian_protections(
    coordinates: npt.ArrayLike,
    symbols: list[str] | None = None,
    masses: npt.ArrayLike | None = None,
    threshold_linear: float = 1e-3,
    angle_tolerance_deg: float = 1.0,
) -> CartesianProtectionResult:
    """Detects linear and quasi-linear topologies near 180-degree singularities.

    Projects atomic coordinates into cylindrical frame (z, rho, phi) and regularizes
    rotational degrees of freedom (DOF=2) and constants to prevent partition
    function overflow.

    :param coordinates: (N, 3) Cartesian coordinates in Angstroms.
    :param symbols: (N,) atomic symbols.
    :param masses: (N,) atomic masses in u.
    :param threshold_linear: Moment of inertia threshold (u*A^2) below which Ia
        triggers linearity.
    :param angle_tolerance_deg: Angular tolerance in degrees for collinearity.
    :return: CartesianProtectionResult model.
    """
    coords: npt.NDArray[np.float64] = np.array(
        cast(Any, coordinates), dtype=np.float64
    )
    n_atoms = coords.shape[0]

    inertia_res = diagonalize_inertia_tensor(coords, masses=masses, symbols=symbols)
    i_a = inertia_res.principal_moments_u_A2[0]
    i_b = inertia_res.principal_moments_u_A2[1]
    i_c = inertia_res.principal_moments_u_A2[2]

    com = np.array(inertia_res.center_of_mass_A, dtype=np.float64)
    rel_coords = coords - com

    # 1. Collinear Principal Axis Determination
    evecs = np.array(inertia_res.principal_axes_matrix, dtype=np.float64)
    # The axis of linear backbone is eigenvector of smallest moment of inertia
    collinear_axis = evecs[:, 0]
    axis_norm = float(sla.norm(collinear_axis))
    if axis_norm < 1e-12:
        collinear_axis = np.array([0.0, 0.0, 1.0])
    else:
        collinear_axis = collinear_axis / axis_norm

    # 2. Geometric Linearity Check
    projections = np.dot(rel_coords, collinear_axis)
    perp_vectors = rel_coords - np.outer(projections, collinear_axis)
    perp_distances = np.sqrt(np.sum(perp_vectors**2, axis=1))
    max_perp_dist = float(np.max(perp_distances)) if n_atoms > 0 else 0.0

    is_strict_linear = inertia_res.rotational_constants.A_MHz is None

    # Quasi-linear check (e.g. floppy complexes with angle close to 180 deg)
    is_quasi_linear = False
    if not is_strict_linear and n_atoms >= 3:
        angles = []
        for i in range(1, n_atoms - 1):
            v1 = coords[i - 1] - coords[i]
            v2 = coords[i + 1] - coords[i]
            n1 = float(sla.norm(v1))
            n2 = float(sla.norm(v2))
            if n1 > 1e-6 and n2 > 1e-6:
                cos_theta = np.dot(v1, v2) / (n1 * n2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle_deg = math.degrees(math.acos(cos_theta))
                angles.append(angle_deg)
        if angles and all(abs(180.0 - ang) <= angle_tolerance_deg for ang in angles):
            is_quasi_linear = True
        elif i_a < threshold_linear or max_perp_dist < 0.05:
            is_quasi_linear = True

    is_any_linear = is_strict_linear  # Quasi-linear warnings must not change the physical rotor.

    # 3. Project to 2D Cylindrical Coordinates (z, rho, phi)
    u_z = collinear_axis
    if abs(u_z[0]) < 0.9:
        arb = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        arb = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    cross_x = np.array(
        [
            arb[1] * u_z[2] - arb[2] * u_z[1],
            arb[2] * u_z[0] - arb[0] * u_z[2],
            arb[0] * u_z[1] - arb[1] * u_z[0],
        ],
        dtype=np.float64,
    )
    norm_x = float(sla.norm(cross_x))
    u_x = cross_x / norm_x if norm_x > 1e-12 else np.array([1.0, 0.0, 0.0])

    cross_y = np.array(
        [
            u_z[1] * u_x[2] - u_z[2] * u_x[1],
            u_z[2] * u_x[0] - u_z[0] * u_x[2],
            u_z[0] * u_x[1] - u_z[1] * u_x[0],
        ],
        dtype=np.float64,
    )
    norm_y = float(sla.norm(cross_y))
    u_y = cross_y / norm_y if norm_y > 1e-12 else np.array([0.0, 1.0, 0.0])

    cylindrical_coords = []
    for i in range(n_atoms):
        r_i = rel_coords[i]
        z_val = float(np.dot(r_i, u_z))
        x_val = float(np.dot(r_i, u_x))
        y_val = float(np.dot(r_i, u_y))
        rho_val = float(math.sqrt(x_val**2 + y_val**2))
        phi_val = float(math.atan2(y_val, x_val))
        cylindrical_coords.append(
            {"z_axial": z_val, "rho_radial": rho_val, "phi_azimuthal": phi_val}
        )

    # 4. Regularize Rotational Constants & Degrees of Freedom
    if is_any_linear:
        rot_dof = 2
        singularity_damping_applied = True
        damping_factor = 1.0
        # For a linear rotor, B = C = C_rot / (0.5 * (Ib + Ic))
        mean_i_perp = 0.5 * (i_b + i_c)
        if mean_i_perp <= 0:
            raise ValueError("Linear rotor requires positive perpendicular inertia")
        b_eff_mhz = float(C_ROT_MHZ / mean_i_perp)
        b_eff_ghz = float(C_ROT_GHZ / mean_i_perp)
        b_eff_cm1 = float(C_ROT_CM1 / mean_i_perp)

        protected_rot = RotationalConstants(
            A_MHz=None,
            B_MHz=b_eff_mhz,
            C_MHz=b_eff_mhz,
            A_GHz=None,
            B_GHz=b_eff_ghz,
            C_GHz=b_eff_ghz,
            A_cm1=None,
            B_cm1=b_eff_cm1,
            C_cm1=b_eff_cm1,
        )
    else:
        rot_dof = 3
        singularity_damping_applied = False
        damping_factor = 0.0
        protected_rot = inertia_res.rotational_constants

    is_linear_singularity = is_strict_linear

    return CartesianProtectionResult(
        is_linear=is_strict_linear,
        is_quasi_linear=is_quasi_linear,
        LINEAR_SINGULARITY=is_linear_singularity,
        linear_singularity=is_linear_singularity,
        rotational_dof=rot_dof,
        collinear_axis=cast(list[float], collinear_axis.tolist()),
        cylindrical_coordinates=cylindrical_coords,
        singularity_damping_applied=singularity_damping_applied,
        damping_factor=damping_factor,
        protected_rotational_constants=protected_rot,
    )


# =============================================================================
# 6. Ray's Asymmetry Parameter & Dynamic Representation Switch
# =============================================================================


def calculate_rays_asymmetry(
    a_const: float | None = None,
    b_const: float | None = None,
    c_const: float | None = None,
    **kwargs: Any,
) -> AsymmetryResult:
    """Calculates Ray's asymmetry parameter kappa = (2*B - A - C) / (A - C)

    and classifies the molecular top.

    :param a_const: Rotational constant A in MHz (or None for linear rotors).
    :param b_const: Rotational constant B in MHz.
    :param c_const: Rotational constant C in MHz.
    :return: AsymmetryResult data model.
    """
    a_val = kwargs.get("A", a_const)
    b_val = kwargs.get("B", b_const)
    c_val = kwargs.get("C", c_const)

    if b_val is None or c_val is None or not np.isfinite([b_val, c_val]).all() or min(b_val, c_val) <= 0:
        raise ValueError("Finite positive B and C are required")
    if a_val is not None and (not math.isfinite(a_val) or a_val <= 0 or a_val < b_val or b_val < c_val):
        raise ValueError("Rotational constants must satisfy A >= B >= C > 0")
    if a_val is None:
        if not math.isclose(b_val, c_val, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("A is undefined only for the declared linear limit B = C")
        # Linear rotor: limiting prolate with A -> infinity
        return AsymmetryResult(
            kappa=-1.0,
            rotor_type="Linear Rotor",
            recommended_representation="Ir",
            axis_mapping={"x": "b", "y": "c", "z": "a"},
            transformation_matrix=[
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            wang_subblocks=["E+", "E-", "O+", "O-"],
            description="Linear rotor (rotational DOF = 2, limiting prolate).",
        )

    # Spherical Top: A == B == C
    diff_ac = a_val - c_val
    if abs(diff_ac) < 1e-9 or (abs(a_val - b_val) < 1e-9 and abs(b_val - c_val) < 1e-9):
        return AsymmetryResult(
            kappa=None,
            rotor_type="Spherical Top",
            recommended_representation="Ir",
            axis_mapping={"x": "b", "y": "c", "z": "a"},
            transformation_matrix=[
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            wang_subblocks=["E+", "E-", "O+", "O-"],
            description="Spherical top (Ia = Ib = Ic, isotropic constants).",
        )

    # General asymmetric rotor
    raw_kappa = (2.0 * b_val - a_val - c_val) / diff_ac
    kappa = float(np.clip(raw_kappa, -1.0, 1.0))

    if kappa <= -0.999999 or abs(b_val - c_val) < 1e-6 * b_val:
        rotor_type = "Prolate Symmetric"
        desc = "Prolate symmetric top (A > B = C, kappa = -1)."
        rec_rep = "Ir"
        axis_map = {"x": "b", "y": "c", "z": "a"}
        t_mat = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    elif kappa >= 0.999999 or abs(a_val - b_val) < 1e-6 * a_val:
        rotor_type = "Oblate Symmetric"
        desc = "Oblate symmetric top (A = B > C, kappa = +1)."
        rec_rep = "IIIr"
        axis_map = {"x": "a", "y": "b", "z": "c"}
        t_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    elif -1.0 < kappa < -0.05:
        rotor_type = "Asymmetric Prolate"
        desc = f"Asymmetric prolate top (-1 < kappa < 0, kappa = {kappa:.5f})."
        rec_rep = "Ir"
        axis_map = {"x": "b", "y": "c", "z": "a"}
        t_mat = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    elif -0.05 <= kappa <= 0.05:
        rotor_type = "Most Asymmetric"
        desc = f"Most asymmetric top (kappa ~ 0, kappa = {kappa:.5f})."
        rec_rep = "Ir" if kappa <= 0 else "IIIr"
        axis_map = (
            {"x": "b", "y": "c", "z": "a"}
            if kappa <= 0
            else {"x": "a", "y": "b", "z": "c"}
        )
        t_mat = (
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
            if kappa <= 0
            else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    else:  # 0.05 < kappa < 1.0
        rotor_type = "Asymmetric Oblate"
        desc = f"Asymmetric oblate top (0 < kappa < 1, kappa = {kappa:.5f})."
        rec_rep = "IIIr"
        axis_map = {"x": "a", "y": "b", "z": "c"}
        t_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    return AsymmetryResult(
        kappa=kappa,
        rotor_type=rotor_type,
        recommended_representation=rec_rep,
        axis_mapping=axis_map,
        transformation_matrix=t_mat,
        wang_subblocks=["E+", "E-", "O+", "O-"],
        description=desc,
    )


def dynamic_representation_switch(
    kappa: float | None = None,
    a_const: float | None = None,
    b_const: float | None = None,
    c_const: float | None = None,
    preferred_type: Literal["auto", "Ir", "Il", "IIr", "IIl", "IIIr", "IIIl"]
    | str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    """Dynamically maps molecular principal axes (a, b, c) to spectroscopic body frame.

    Supports all 6 standard King-Hainer-Cross representations:
    I^r:   x->b, y->c, z->a (right-handed, prolate-optimal)
    I^l:   x->c, y->b, z->a (left-handed)
    II^r:  x->c, y->a, z->b (right-handed)
    II^l:  x->a, y->c, z->b (left-handed)
    III^r: x->a, y->b, z->c (right-handed, oblate-optimal)
    III^l: x->b, y->a, z->c (left-handed)

    :param kappa: Ray's asymmetry parameter.
    :param a_const: Rotational constant A in MHz.
    :param b_const: Rotational constant B in MHz.
    :param c_const: Rotational constant C in MHz.
    :param preferred_type: Representation choice or 'auto'.
    :return: Dictionary containing representation details and matrices.
    """
    a_val = kwargs.get("A", a_const)
    b_val = kwargs.get("B", b_const)
    c_val = kwargs.get("C", c_const)

    if kappa is None:
        if b_val is not None and c_val is not None:
            asym = calculate_rays_asymmetry(a_val, b_val, c_val)
            kappa_val = asym.kappa
        else:
            raise ValueError("Provide rotational constants or an explicit asymmetry parameter")
    else:
        kappa_val = float(kappa)
        if not math.isfinite(kappa_val) or not -1 <= kappa_val <= 1:
            raise ValueError("Ray asymmetry parameter must be finite and in [-1, 1]")

    rep_table: dict[str, dict[str, Any]] = {
        "Ir": {
            "axis_mapping": {"x": "b", "y": "c", "z": "a"},
            "transformation_matrix": [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            "right_handed": True,
            "quantization_axis": "a",
            "optimal_for": "Prolate-like rotors (kappa <= 0)",
        },
        "Il": {
            "axis_mapping": {"x": "c", "y": "b", "z": "a"},
            "transformation_matrix": [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "right_handed": False,
            "quantization_axis": "a",
            "optimal_for": "Prolate left-handed frame",
        },
        "IIr": {
            "axis_mapping": {"x": "c", "y": "a", "z": "b"},
            "transformation_matrix": [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "right_handed": True,
            "quantization_axis": "b",
            "optimal_for": "Intermediate asymmetry representation",
        },
        "IIl": {
            "axis_mapping": {"x": "a", "y": "c", "z": "b"},
            "transformation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            "right_handed": False,
            "quantization_axis": "b",
            "optimal_for": "Intermediate left-handed frame",
        },
        "IIIr": {
            "axis_mapping": {"x": "a", "y": "b", "z": "c"},
            "transformation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "right_handed": True,
            "quantization_axis": "c",
            "optimal_for": "Oblate-like rotors (kappa > 0)",
        },
        "IIIl": {
            "axis_mapping": {"x": "b", "y": "a", "z": "c"},
            "transformation_matrix": [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "right_handed": False,
            "quantization_axis": "c",
            "optimal_for": "Oblate left-handed frame",
        },
    }

    chosen_rep = (
        "Ir"
        if preferred_type == "auto" and (kappa_val is None or kappa_val <= 0.0)
        else ("IIIr" if preferred_type == "auto" else preferred_type)
    )

    rep_info = rep_table[chosen_rep]

    return {
        "representation": chosen_rep,
        "kappa": kappa_val,
        "axis_mapping": rep_info["axis_mapping"],
        "transformation_matrix": rep_info["transformation_matrix"],
        "is_right_handed": rep_info["right_handed"],
        "quantization_axis": rep_info["quantization_axis"],
        "recommendation_reason": rep_info["optimal_for"],
        "wang_subblocks": ["E+", "E-", "O+", "O-"],
    }


# =============================================================================
# 7. Eckart Dipole Phase-Lock Guard & Parity Preservation
# =============================================================================


def dipole_phase_guard(
    raw_dipole_vector: npt.ArrayLike,
    eckart_matrix: npt.ArrayLike,
    principal_axes_matrix: npt.ArrayLike | None = None,
) -> dict[str, Any]:
    """Projects Cartesian dipole moment onto principal axes while locking phase relative
    to reference Eckart frame to prevent unphysical sign flips during torsional sweeps.

    Tracks alignment phases against the reference Eckart frame, ensuring parity
    preservation (det(R_locked) = +1.0) and eliminating artificial sign inversions
    in projected dipole components (mu_a, mu_b, mu_c).

    :param raw_dipole_vector: (3,) Cartesian dipole moment vector (in Debye or a.u.).
    :param eckart_matrix: (3, 3) Reference Eckart orientation matrix or previous step frame.
    :param principal_axes_matrix: (3, 3) Optional current principal axes matrix (R_PA).
        If None, eckart_matrix is used as the reference projector.
    :return: Dictionary containing:
        - 'mu_Cart': List of Cartesian dipole components [mu_x, mu_y, mu_z].
        - 'mu_PA': List of projected principal axis dipole components [mu_a, mu_b, mu_c].
        - 'mu_norm': Magnitude of the dipole vector.
        - 'det_R_locked': Determinant of the phase-locked rotation matrix (+1.0).
        - 'phase_flips': List of signed multipliers [-1 or +1] applied to each axis.
        - 'R_locked': (3, 3) Phase-locked right-handed rotation matrix.
    """
    mu_cart = np.array(cast(Any, raw_dipole_vector), dtype=np.float64)
    if mu_cart.shape != (3,):
        raise ValueError(f"raw_dipole_vector must have shape (3,), got {mu_cart.shape}")

    r_ref = np.array(cast(Any, eckart_matrix), dtype=np.float64)
    if r_ref.shape != (3, 3):
        raise ValueError(f"eckart_matrix must have shape (3, 3), got {r_ref.shape}")

    if principal_axes_matrix is not None:
        r_pa = np.array(cast(Any, principal_axes_matrix), dtype=np.float64)
        if r_pa.shape != (3, 3):
            raise ValueError(f"principal_axes_matrix must have shape (3, 3), got {r_pa.shape}")
    else:
        r_pa = np.copy(r_ref)

    # Phase-lock check for each axis vector against reference Eckart axis
    r_locked = np.copy(r_pa)
    flips = [1, 1, 1]

    for k in range(3):
        overlap = float(np.dot(r_pa[:, k], r_ref[:, k]))
        if overlap < 0.0:
            r_locked[:, k] = -r_locked[:, k]
            flips[k] = -1

    # Parity check: det(R_locked) must be strictly +1.0 in SO(3)
    det_val = float(sla.det(r_locked))
    if det_val < 0.0:
        # Enforce right-handed SO(3) parity lock: v_c = v_a x v_b
        r_locked[:, 2] = np.cross(r_locked[:, 0], r_locked[:, 1])
        norm_c = float(sla.norm(r_locked[:, 2]))
        if norm_c > 1e-12:
            r_locked[:, 2] /= norm_c
        flips[2] = -flips[2]
        det_val = float(sla.det(r_locked))

    # Project raw Cartesian dipole onto locked principal axes: mu_PA = R_locked^T * mu_Cart
    mu_pa = np.dot(r_locked.T, mu_cart)
    mu_norm = float(sla.norm(mu_cart))

    return {
        "mu_Cart": cast(list[float], mu_cart.tolist()),
        "mu_PA": cast(list[float], mu_pa.tolist()),
        "mu_norm": mu_norm,
        "det_R_locked": det_val,
        "phase_flips": flips,
        "R_locked": cast(list[list[float]], r_locked.tolist()),
    }


# =============================================================================
# 8. BLAKE3 Cryptographic Sealing & Zero-Copy PyArrow IPC Buffer Allocation
# =============================================================================


def compute_blake3_seal(
    data: bytes | bytearray | memoryview | str | dict[str, Any] | BaseModel,
) -> str:
    """Computes a 256-bit BLAKE3 cryptographic hash digest.

    Requires BLAKE3; another algorithm must not be mislabeled as BLAKE3.

    :param data: Input data (bytes, string, dict, or Pydantic model).
    :return: Hexadecimal hash string.
    """
    if isinstance(data, BaseModel):
        raw_bytes = data.model_dump_json().encode("utf-8")
    elif isinstance(data, dict):
        raw_bytes = json.dumps(data, sort_keys=True, allow_nan=False).encode("utf-8")
    elif isinstance(data, str):
        raw_bytes = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(data)
    else:
        raw_bytes = str(data).encode("utf-8")

    if not _BLAKE3_AVAILABLE:
        raise RuntimeError("BLAKE3 digest requested but blake3 is not installed")
    return blake3.blake3(raw_bytes).hexdigest()


def allocate_pyarrow_ipc_buffer(
    tensor_output: TorqTensorOutput | dict[str, Any],
) -> tuple[pa.Buffer, str]:
    """Allocates a zero-copy PyArrow RecordBatch / Table IPC buffer containing
    the harvested quantum tensors, sealed with BLAKE3 cryptographic digest.

    :param tensor_output: Validated TorqTensorOutput model or tensor extraction dictionary.
    :return: (pa_buffer, blake3_seal_hex).
    """
    if isinstance(tensor_output, TorqTensorOutput):
        d = tensor_output.model_dump()
    else:
        d = dict(tensor_output)

    point_id = str(d.get("point_id", "000"))

    # Inertia data
    in_data = d.get("inertia", {})
    if isinstance(in_data, InertiaTensorResult):
        in_data = in_data.model_dump()
    rc_data = in_data.get("rotational_constants", d.get("rotational_constants", {}))
    if isinstance(rc_data, RotationalConstants):
        rc_data = rc_data.model_dump()

    def finite_or_none(value):
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Nonfinite tensor observable")
        return value

    a_mhz = finite_or_none(rc_data.get("A_MHz", rc_data.get("A")))
    b_mhz = finite_or_none(rc_data.get("B_MHz", rc_data.get("B")))
    c_mhz = finite_or_none(rc_data.get("C_MHz", rc_data.get("C")))
    pm = in_data.get("principal_moments_u_A2", d.get("principal_moments_u_A2"))
    if pm is not None and len(pm) != 3:
        raise ValueError("Principal moments require exactly three components")
    i_a, i_b, i_c = [finite_or_none(v) for v in pm] if pm is not None else [None, None, None]
    asym_data = d.get("asymmetry", {})
    if isinstance(asym_data, AsymmetryResult):
        asym_data = asym_data.model_dump()
    kappa_val = finite_or_none(asym_data.get("kappa"))
    rotor_type_str = asym_data.get("rotor_type")
    rec_rep_str = asym_data.get("recommended_representation")
    prot_data = d.get("cartesian_protection", {})
    if isinstance(prot_data, CartesianProtectionResult):
        prot_data = prot_data.model_dump()
    is_linear = prot_data.get("is_linear")
    linear_singularity = prot_data.get("LINEAR_SINGULARITY", prot_data.get("linear_singularity"))
    is_planar = in_data.get("is_planar", d.get("is_planar"))
    inertial_defect = finite_or_none(in_data.get("inertial_defect_u_A2", d.get("inertial_defect_u_A2")))

    # Construct PyArrow RecordBatch schema
    schema = pa.schema(
        [
            ("point_id", pa.string()),
            ("total_mass_u", pa.float64()),
            ("I_a_u_A2", pa.float64()),
            ("I_b_u_A2", pa.float64()),
            ("I_c_u_A2", pa.float64()),
            ("A_MHz", pa.float64()),
            ("B_MHz", pa.float64()),
            ("C_MHz", pa.float64()),
            ("kappa", pa.float64()),
            ("rotor_type", pa.string()),
            ("representation", pa.string()),
            ("is_linear", pa.bool_()),
            ("LINEAR_SINGULARITY", pa.bool_()),
            ("is_planar", pa.bool_()),
            ("inertial_defect_u_A2", pa.float64()),
        ]
    )

    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([point_id], type=pa.string()),
            pa.array([finite_or_none(in_data.get("total_mass_u"))], type=pa.float64()),
            pa.array([i_a], type=pa.float64()),
            pa.array([i_b], type=pa.float64()),
            pa.array([i_c], type=pa.float64()),
            pa.array([a_mhz], type=pa.float64()),
            pa.array([b_mhz], type=pa.float64()),
            pa.array([c_mhz], type=pa.float64()),
            pa.array([kappa_val], type=pa.float64()),
            pa.array([rotor_type_str], type=pa.string()),
            pa.array([rec_rep_str], type=pa.string()),
            pa.array([is_linear], type=pa.bool_()),
            pa.array([linear_singularity], type=pa.bool_()),
            pa.array([is_planar], type=pa.bool_()),
            pa.array([inertial_defect], type=pa.float64()),
        ],
        schema=schema,
    )

    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)

    buf = sink.getvalue()
    seal = compute_blake3_seal(buf.to_pybytes())
    return buf, seal


# =============================================================================
# 9. High-Level TorqTensorExtractor Class (Integration & Provenance)
# =============================================================================


class TorqTensorExtractor:
    """CoChem-TORQ Stage 4.1 Tensor Extraction and Quantum Provenance Harvester.

    Processes optimized geometries into rigorous quantum rotational tensors,
    applies Cartesian protections against linearity singularities, dynamically
    switches representations, parses ORCA VPT2 and Coriolis coupling matrices,
    and exports payloads into HDF5 and JSON formats with Air-Gap compliance.
    """

    def __init__(
        self,
        symbols: list[str],
        coordinates: npt.ArrayLike,
        point_id: str = "000",
        masses: npt.ArrayLike | None = None,
        orca_file: str | Path | None = None,
    ) -> None:
        """Initializes the tensor extractor.

        :param symbols: List of chemical element or isotope symbols.
        :param coordinates: Nx3 Cartesian coordinates in Angstroms.
        :param point_id: Topographic identifier for provenance tracking.
        :param masses: Optional custom atomic masses in u.
        :param orca_file: Optional path to ORCA output file.
        """
        self.symbols = list(symbols)
        self.coordinates: npt.NDArray[np.float64] = np.array(
            cast(Any, coordinates), dtype=np.float64
        )
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError(
                f"Coordinates must have shape (N, 3), got {self.coordinates.shape}."
            )
        if len(self.symbols) != self.coordinates.shape[0]:
            n_s = len(self.symbols)
            n_c = self.coordinates.shape[0]
            raise ValueError(f"Symbols count ({n_s}) != coords count ({n_c}).")

        self.point_id = str(point_id)
        self.orca_file = Path(orca_file) if orca_file else None

        if masses is not None:
            self.masses: npt.NDArray[np.float64] = np.array(
                cast(Any, masses), dtype=np.float64
            )
        else:
            self.masses = np.array(
                [get_atomic_mass(sym) for sym in self.symbols], dtype=np.float64
            )

        self.total_mass = float(np.sum(self.masses))

        # Lazy cache for structured outputs
        self._inertia_result: InertiaTensorResult | None = None
        self._protection_result: CartesianProtectionResult | None = None
        self._asymmetry_result: AsymmetryResult | None = None
        self._full_output: TorqTensorOutput | None = None

        # Legacy backward-compatible attributes
        self.inertia_tensor: npt.NDArray[np.float64] | None = None
        self.rotational_constants: dict[str, float] | None = None
        self.vpt2_resonances: dict[str, Any] = {}
        self.coriolis_couplings: dict[str, Any] = {}
        self.centrifugal_distortion: dict[str, Any] = {}

    def _compute_inertia_tensor(self) -> npt.NDArray[np.float64]:
        """Computes the 3x3 inertia tensor from atomic coordinates in u*Angstrom^2."""
        res = self.get_inertia_result()
        self.inertia_tensor = np.array(res.inertia_tensor_u_A2, dtype=np.float64)
        return self.inertia_tensor

    def _compute_rotational_constants(self) -> dict[str, float]:
        """Computes rotational constants (A, B, C in MHz) with physical conversion."""
        res = self.get_inertia_result()
        rc = res.rotational_constants
        self.rotational_constants = {
            "A": rc.A_MHz,
            "B": rc.B_MHz,
            "C": rc.C_MHz,
        }
        return self.rotational_constants

    def get_inertia_result(self) -> InertiaTensorResult:
        """Derives full InertiaTensorResult model."""
        if self._inertia_result is None:
            self._inertia_result = diagonalize_inertia_tensor(
                coordinates=self.coordinates, masses=self.masses, symbols=self.symbols
            )
            self.inertia_tensor = np.array(
                self._inertia_result.inertia_tensor_u_A2, dtype=np.float64
            )
            rc = self._inertia_result.rotational_constants
            self.rotational_constants = {
                "A": rc.A_MHz,
                "B": rc.B_MHz,
                "C": rc.C_MHz,
            }
        return self._inertia_result

    def get_cartesian_protection(
        self, threshold_linear: float = 1e-3, angle_tolerance_deg: float = 1.0
    ) -> CartesianProtectionResult:
        """Derives Cartesian protection result."""
        if self._protection_result is None:
            self._protection_result = apply_cartesian_protections(
                coordinates=self.coordinates,
                symbols=self.symbols,
                masses=self.masses,
                threshold_linear=threshold_linear,
                angle_tolerance_deg=angle_tolerance_deg,
            )
        return self._protection_result

    def get_asymmetry_result(self) -> AsymmetryResult:
        """Derives Ray's asymmetry parameter and top classification."""
        if self._asymmetry_result is None:
            in_res = self.get_inertia_result()
            rc = in_res.rotational_constants
            self._asymmetry_result = calculate_rays_asymmetry(
                rc.A_MHz, rc.B_MHz, rc.C_MHz
            )
        return self._asymmetry_result

    def extract_tensors(self) -> dict[str, Any]:
        """Extracts comprehensive rotational, inertial, and symmetry tensors."""
        in_res = self.get_inertia_result()
        prot_res = self.get_cartesian_protection()
        asym_res = self.get_asymmetry_result()

        return {
            "point_id": self.point_id,
            "symbols": self.symbols,
            "coordinates": self.coordinates.tolist(),
            "rotational_constants": {
                "A": in_res.rotational_constants.A_MHz,
                "B": in_res.rotational_constants.B_MHz,
                "C": in_res.rotational_constants.C_MHz,
            },
            "rotational_constants_detailed": in_res.rotational_constants.model_dump(),
            "inertia_tensor": in_res.inertia_tensor_u_A2,
            "principal_moments_u_A2": in_res.principal_moments_u_A2,
            "principal_moments_kg_m2": in_res.principal_moments_kg_m2,
            "principal_axes_matrix": in_res.principal_axes_matrix,
            "planar_moments": in_res.planar_moments.model_dump(),
            "inertial_defect_u_A2": in_res.inertial_defect_u_A2,
            "is_planar": in_res.is_planar,
            "cartesian_protection": prot_res.model_dump(),
            "asymmetry": asym_res.model_dump(),
        }

    def get_full_output(self, orca_file: str | Path | None = None) -> TorqTensorOutput:
        """Produces a validated Pydantic TorqTensorOutput payload."""
        in_res = self.get_inertia_result()
        prot_res = self.get_cartesian_protection()
        asym_res = self.get_asymmetry_result()

        vpt2_model: VPT2Data | None = None
        target_orca = orca_file or self.orca_file
        if target_orca and Path(target_orca).exists():
            vpt2_dict = self.extract_vpt2_data(target_orca)
            vpt2_model = VPT2Data(
                darling_dennison=vpt2_dict.get("darling_dennison", []),
                coriolis_couplings=vpt2_dict.get("coriolis_couplings", {}),
                centrifugal_distortion=vpt2_dict.get("centrifugal_distortion", {}),
                raman_polarizability=vpt2_dict.get("raman_polarizability", []),
                is_divergent=vpt2_dict.get("is_divergent", False),
                divergence_details=vpt2_dict.get("divergence_details", []),
            )

        if self._full_output is None or orca_file is not None:
            self._full_output = TorqTensorOutput(
                point_id=self.point_id,
                symbols=self.symbols,
                coordinates=cast(list[list[float]], self.coordinates.tolist()),
                inertia=in_res,
                cartesian_protection=prot_res,
                asymmetry=asym_res,
                vpt2=vpt2_model,
                metadata={
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "codata_year": CODATA_YEAR,
                    "c_rot_mhz": C_ROT_MHZ,
                },
            )
        return self._full_output

    def guard_dipole(
        self,
        raw_dipole_vector: npt.ArrayLike,
        eckart_matrix: npt.ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Applies Eckart Dipole Phase-Lock Guard to raw Cartesian dipole moment.

        :param raw_dipole_vector: (3,) Cartesian dipole vector.
        :param eckart_matrix: (3, 3) Reference Eckart rotation matrix (defaults to current principal axes).
        :return: Phase-locked dipole dictionary with det(R_locked)=+1.0 and projected mu_PA.
        """
        in_res = self.get_inertia_result()
        pa_mat = in_res.principal_axes_matrix
        ref_mat = eckart_matrix if eckart_matrix is not None else pa_mat
        return dipole_phase_guard(
            raw_dipole_vector=raw_dipole_vector,
            eckart_matrix=ref_mat,
            principal_axes_matrix=pa_mat,
        )

    def to_pyarrow_ipc_buffer(self) -> tuple[pa.Buffer, str]:
        """Serializes harvested quantum tensors into a zero-copy PyArrow IPC buffer with BLAKE3 seal.

        :return: Tuple of (PyArrow Buffer, BLAKE3 seal hex string).
        """
        full_out = self.get_full_output()
        return allocate_pyarrow_ipc_buffer(full_out)

    def get_blake3_seal(self) -> str:
        """Returns 256-bit BLAKE3 cryptographic seal of the harvested quantum tensor payload.

        :return: Hexadecimal hash digest string.
        """
        full_out = self.get_full_output()
        return compute_blake3_seal(full_out)

    # =========================================================================
    # ORCA VPT2, Coriolis & Centrifugal Distortion Parsing
    # =========================================================================

    def _parse_orca_vib_block(self, orca_file: str | Path) -> dict[str, Any]:
        """Parses ORCA %vib block for advanced VPT2 data using regex parsing.

        Extracts:
        1. Darling-Dennison Resonances
        2. Coriolis Coupling Matrices (x, y, z axes)
        3. Centrifugal Distortion Constants (D_J, D_JK, D_K, d_1, d_2)
        4. Raman Polarizability Derivatives
        """
        orca_path = Path(orca_file)
        logger.info(f"Parsing ORCA %vib block from {orca_path}")

        vpt2_data: dict[str, Any] = {
            "darling_dennison": [],
            "coriolis_couplings": {"x": [], "y": [], "z": []},
            "centrifugal_distortion": {
                "D_J": [],
                "D_JK": [],
                "D_K": [],
                "d_1": [],
                "d_2": [],
            },
            "raman_polarizability": [],
            "is_divergent": False,
            "divergence_details": [],
        }

        if not orca_path.exists():
            return vpt2_data

        try:
            content = orca_path.read_text(encoding="utf-8", errors="ignore")

            # 1. Darling-Dennison resonances
            dd_pat = (
                r"Darling-Dennison\s+Mode\s+(\d+)\s+Mode\s+(\d+)\s+K\s*=\s*"
                r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)"
            )
            dd_matches = re.findall(dd_pat, content, re.IGNORECASE)
            for m in dd_matches:
                vpt2_data["darling_dennison"].append(
                    {"mode1": int(m[0]), "mode2": int(m[1]), "resonance": float(m[2])}
                )

            # 2. Coriolis couplings per Cartesian axis (X, Y, Z)
            for axis in ["x", "y", "z"]:
                cor_pat = (
                    rf"Coriolis Coupling Matrix \({axis.upper()}\)\s+[-=]+\s*"
                    r"(.*?)(?=\n\n|\n[A-Z]|\Z)"
                )
                cor_section = re.search(cor_pat, content, re.DOTALL)
                if cor_section:
                    vals = [
                        float(v)
                        for v in re.findall(
                            r"-?\d+\.\d+(?:[eE][-+]?\d+)?", cor_section.group(1)
                        )
                    ]
                    vpt2_data["coriolis_couplings"][axis] = vals

            # 3. Centrifugal distortion constants
            for key in ["D_J", "D_JK", "D_K", "d_1", "d_2"]:
                cd_match = re.search(
                    rf"{key}\s*=\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)", content
                )
                if cd_match:
                    vpt2_data["centrifugal_distortion"][key] = [
                        float(cd_match.group(1))
                    ]

            # 4. Raman polarizability derivatives
            deriv_match = re.findall(
                r"Polarizability\s+derivative\s*:\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)",
                content,
                re.IGNORECASE,
            )
            if deriv_match:
                vpt2_data["raman_polarizability"] = [float(x) for x in deriv_match]

        except Exception as e:
            logger.error(f"Error parsing ORCA VPT2 file {orca_path}: {e}")
            raise

        return vpt2_data

    def extract_vpt2_data(
        self, orca_file: str | Path, is_lam_complex: bool = False
    ) -> dict[str, Any]:
        """Extracts VPT2 data from ORCA output with divergence check."""
        logger.info(f"Extracting VPT2 data from ORCA output: {orca_file}")
        vpt2_data = self._parse_orca_vib_block(orca_file)
        if is_lam_complex:
            logger.info("LAM complex detected - extracting advanced VPT2 additions.")
            vpt2_data.update(self._extract_lam_vpt2_additions(orca_file))

        is_div, div_reasons = self._check_divergence(
            vpt2_data.get("centrifugal_distortion", {})
        )
        vpt2_data["is_divergent"] = is_div
        vpt2_data["divergence_details"] = div_reasons
        return vpt2_data

    def _extract_lam_vpt2_additions(self, orca_file: str | Path) -> dict[str, Any]:
        """Extracts additional VPT2 data required for Large-Amplitude Motion."""
        lam_data: dict[str, Any] = {
            "darling_dennison_resonances": [],
            "coriolis_coupling_matrices": {"x": [], "y": [], "z": []},
            "centrifugal_distortion_constants": {
                "D_J": [],
                "D_JK": [],
                "D_K": [],
                "d_1": [],
                "d_2": [],
            },
        }
        orca_path = Path(orca_file)
        if not orca_path.exists():
            return lam_data

        try:
            content = orca_path.read_text(encoding="utf-8", errors="ignore")
            dd_matches = re.findall(
                r"Resonance\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)", content
            )
            for m in dd_matches:
                lam_data["darling_dennison_resonances"].append(
                    {
                        "mode1": int(m[0]),
                        "mode2": int(m[1]),
                        "resonance_strength": float(m[2]),
                    }
                )
        except Exception as e:
            logger.error(f"Error extracting LAM VPT2 additions: {e}")
            raise
        return lam_data

    def _check_divergence(
        self, distortion_constants: dict[str, list[float]]
    ) -> tuple[bool, list[str]]:
        """Validates distortion constants against unphysical divergence (> 1e6 MHz)."""
        divergent = False
        reasons: list[str] = []
        for key, values in distortion_constants.items():
            if len(values) > 0:
                max_val = float(np.max(np.abs(np.array(values, dtype=np.float64))))
                if max_val > 1e6 or math.isnan(max_val) or math.isinf(max_val):
                    msg = f"Unphysical centrifugal distortion constant {key}: {max_val}"
                    logger.warning(msg)
                    divergent = True
                    reasons.append(msg)
        if divergent:
            logger.warning("Divergence detected - recommending switch to DVR protocol.")
        return divergent, reasons

    def extract_thermal_nmr(self, trajectory_file: str | Path | None = None) -> dict[str, Any]:
        """Require per-nucleus electronic shielding and thermal weights.

        Coordinates alone do not determine a calculated NMR shielding. The legacy
        geometry-to-shielding surrogate and cross-nucleus 'thermal average' were removed.
        """
        raise RuntimeError("Thermal NMR unavailable: validated per-frame/per-nucleus shielding and weighting adapter required")

    def extract_raman_polarizability(self, orca_file: str | Path | None = None) -> dict[str, Any]:
        """Require a qualified polarizability-derivative parser with units/mode mapping."""
        raise RuntimeError("Raman derivatives unavailable: a qualified electronic-response parser is required; inertia is not polarizability")

    def extract_spin_hamiltonian(
        self, orca_file: str | Path | None = None
    ) -> dict[str, Any]:
        """Extracts Spin Hamiltonian parameters."""
        raise RuntimeError(
            "Anti-spoofing mandate: Unverified Spin Hamiltonian code removed. "
            "Use full quantum engine output for electronic EPR/NMR g-tensor."
        )

    # =========================================================================
    # JSON and HDF5 Export Gateways
    # =========================================================================

    def export_tensor(self, output_file: str | Path = "torq_tensors.json") -> None:
        """Exports all extracted tensors to a JSON file respecting target directory."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        result = self.extract_tensors()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Tensor data exported to {out_path}")

    def export_vpt2_tensor(
        self,
        output_file: str | Path = "torq_vpt2.json",
        orca_file: str | Path | None = None,
    ) -> None:
        """Exports VPT2 resonance data to JSON."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        target_file = orca_file or self.orca_file
        if not target_file or not Path(target_file).exists():
            raise FileNotFoundError(f"Target ORCA file not found: {target_file}")
            
        vpt2_data = self.extract_vpt2_data(target_file)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(vpt2_data, f, indent=2)

        logger.info(f"VPT2 data exported to {out_path}")

    def export_lam_vpt2_tensor(
        self,
        output_file: str | Path = "torq_lam_vpt2.json",
        orca_file: str | Path | None = None,
    ) -> None:
        """Exports LAM-specific VPT2 tensor data."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        target_file = orca_file or self.orca_file
        if not target_file or not Path(target_file).exists():
            raise FileNotFoundError(f"Target ORCA file not found: {target_file}")
            
        vpt2_data = self.extract_vpt2_data(target_file, is_lam_complex=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(vpt2_data, f, indent=2)

        logger.info(f"LAM VPT2 data exported to {out_path}")

    def export_to_hdf5(
        self, h5_file_path: str | Path, data_dict: dict[str, Any]
    ) -> None:
        """Exports data dictionary to an HDF5 group for CoChem-SCRIBE integration."""
        out_path = Path(h5_file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with h5py.File(out_path, "a") as f:
                group_name = f"point_{self.point_id}"
                if group_name in f:
                    point_group = f[group_name]
                else:
                    point_group = f.create_group(group_name)

                for key, value in data_dict.items():
                    if isinstance(value, list | np.ndarray):
                        arr = np.array(value)
                        if key in point_group:
                            del point_group[key]
                        if arr.dtype.kind in ("U", "S", "O"):
                            dt = h5py.string_dtype(encoding="utf-8")
                            point_group.create_dataset(
                                key, data=np.array(value, dtype=object), dtype=dt
                            )
                        else:
                            point_group.create_dataset(key, data=arr)
                    elif isinstance(value, int | float | str | bool):
                        point_group.attrs[key] = value
                    elif isinstance(value, dict):
                        sub_json = json.dumps(value)
                        point_group.attrs[f"{key}_json"] = sub_json

            logger.info(
                f"Data exported to HDF5 tensor at {out_path} under {group_name}"
            )
        except Exception as e:
            logger.error(f"Failed to export to HDF5 at {out_path}: {e}")
            raise

    def export_to_hdf5_with_sinc_dvr(
        self, h5_file_path: str | Path, dvr_data: dict[str, Any]
    ) -> None:
        """Exports Sinc-DVR tunneling and vibrational wavefunctions to HDF5."""
        out_path = Path(h5_file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with h5py.File(out_path, "a") as f:
                group_name = f"point_{self.point_id}"
                if group_name in f:
                    point_group = f[group_name]
                else:
                    point_group = f.create_group(group_name)

                if "wavefunction" in dvr_data:
                    if "wavefunction" in point_group:
                        del point_group["wavefunction"]
                    point_group.create_dataset(
                        "wavefunction", data=np.array(dvr_data["wavefunction"])
                    )

                if "energy_levels" in dvr_data:
                    if "energy_levels" in point_group:
                        del point_group["energy_levels"]
                    point_group.create_dataset(
                        "energy_levels", data=np.array(dvr_data["energy_levels"])
                    )

                if "tunneling_splitting" in dvr_data:
                    point_group.attrs["tunneling_splitting"] = float(
                        dvr_data["tunneling_splitting"]
                    )

                if "kraitchman_coords" in dvr_data:
                    if "kraitchman_coords" in point_group:
                        del point_group["kraitchman_coords"]
                    point_group.create_dataset(
                        "kraitchman_coords",
                        data=np.array(dvr_data["kraitchman_coords"]),
                    )

            logger.info(f"Sinc-DVR data exported to HDF5 at {out_path}")
        except Exception as e:
            logger.error(f"Failed to export Sinc-DVR data to HDF5 at {out_path}: {e}")
            raise


# =============================================================================
# CLI Self-Test Runner
# =============================================================================

if __name__ == "__main__":
    test_symbols = ["O", "C", "O"]
    test_coords = [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]
    extractor = TorqTensorExtractor(test_symbols, test_coords, point_id="co2_test")
    res = extractor.get_full_output()
    print(f"Point: {res.point_id}")
    print(f"Linear: {res.cartesian_protection.is_linear}")
    print(f"Rotational DOF: {res.cartesian_protection.rotational_dof}")
    print(f"B: {res.inertia.rotational_constants.B_MHz:.4f} MHz")
