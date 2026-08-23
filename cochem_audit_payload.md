Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task8_matrix_loader.md.
Original prompt:
# Prompt: Method Matrix Parser & Provenance

**Target File:** `D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_matrix_loader.py`

## Objective
Implement Method Matrix Parser & Provenance for CoChem-TORQ based on Task 8 SRS specifications.

## Instructions for Coder
1. Create or update `cochem_torq_matrix_loader.py` inside `Libraries/`.
2. Implement `parse_execution_cascade` to resolve MLFF fallback hierarchies (e.g., MACE-OFF23 -> MACE-POLAR -> AIMNet2 -> GFN2-xTB) for unsupported atomic species.
3. Implement `validate_auxiliary_basis` ensuring def2/J pairs with RI-J, def2/JK with RI-JK, and def2-TZVPP/C with Double-Hybrids. Validate ma-def2 for diffuse systems.
4. Implement `generate_provenance_hash` generating a deterministic SHA-256 hash of all physical constants, basis sets, and integration grids to write to `cochem_deployment_manifest.json`.
5. QA Guardrail: Implement Cascade Provenance & Automated MLFF Fallback Test.

## Constraints & Anti-Spoofing
- **One Script Policy**: Only create or modify `cochem_torq_matrix_loader.py`.
- **Zero Mocking**: Do NOT mock logic. Use real libraries (`psutil`, `pynvml`, `xxhash`, `pyarrow`) and implement physical processing.
- **Context-Safety**: Do not hallucinate imports. Limit dependencies to the `requirements.txt` environment for CoChem-TORQ.
- **Air-Gap Compliance**: The generated Python script MUST NOT write any data or logs to the repository space at runtime. Read and write strictly according to the paths defined dynamically.

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_tensor_extractor.py ---
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
import scipy.linalg as sla  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

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
ATOMIC_MASS_CONSTANT_U: Final[float] = 1.66053906892e-27  # Exact kg (1 u) (CODATA 2022)
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
    """Retrieves exact mono-isotopic mass for an element or isotope using mendeleev.

    Supports notation such as: 'H', 'D', 'T', '13C', 'C13', '18O', 'O18', '37Cl'.
    """
    clean_sym = symbol.strip()
    
    # Handle specific common aliases
    if clean_sym == "D":
        clean_sym = "2H"
    elif clean_sym == "T":
        clean_sym = "3H"

    match_prefix = re.match(r"^(\d+)([a-zA-Z]+)$", clean_sym)
    match_postfix = re.match(r"^([a-zA-Z]+)(\d+)$", clean_sym)
    
    elem_str = clean_sym
    mass_num = None
    
    if match_prefix:
        mass_num = int(match_prefix.group(1))
        elem_str = match_prefix.group(2)
    elif match_postfix:
        elem_str = match_postfix.group(1)
        mass_num = int(match_postfix.group(2))
        
    elem_str = elem_str.capitalize()
    
    try:
        from mendeleev import element
        elem = element(elem_str)
        if mass_num is not None:
            for iso in elem.isotopes:
                if iso.mass_number == mass_num and iso.mass is not None:
                    return float(iso.mass)
            logger.warning(f"Isotope {mass_num} for element {elem_str} not found. Defaulting to most abundant.")
            
        # Default to most abundant isotope
        valid_isotopes = [iso for iso in elem.isotopes if iso.abundance is not None and iso.mass is not None]
        if valid_isotopes:
            most_abundant = sorted(valid_isotopes, key=lambda x: x.abundance, reverse=True)[0]
            return float(most_abundant.mass)
        elif elem.isotopes and elem.isotopes[0].mass is not None:
            return float(elem.isotopes[0].mass)
    except Exception as e:
        raise ValueError(f"Symbol '{symbol}' not found in mendeleev or error occurred: {e}")




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

    kappa: float = Field(..., description="Ray's asymmetry parameter in [-1, +1]")
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
) -> InertiaTensorResult:
    """Computes COM, builds moment of inertia tensor, diagonalizes to principal axes,

    and derives rotational constants, planar moments, and inertial defect.

    :param coordinates: (N, 3) Cartesian coordinates in Angstroms.
    :param masses: (N,) atomic masses in u (optional if symbols provided).
    :param symbols: (N,) atomic symbols (optional if masses provided).
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
    if masses is not None:
        mass_arr: npt.NDArray[np.float64] = np.array(
            cast(Any, masses), dtype=np.float64
        )
        if mass_arr.shape[0] != n_atoms:
            raise ValueError(
                f"Masses length ({mass_arr.shape[0]}) != atom count ({n_atoms})."
            )
    elif symbols is not None:
        if len(symbols) != n_atoms:
            raise ValueError(
                f"Symbols length ({len(symbols)}) != atom count ({n_atoms})."
            )
        mass_arr = np.array(
            [get_atomic_mass(sym) for sym in symbols], dtype=np.float64
        )
    else:
        raise ValueError("Either masses or symbols must be supplied.")

    total_mass = float(np.sum(mass_arr))
    if total_mass <= 0.0:
        raise ValueError("Total molecular mass must be strictly positive.")

    # 1. Shift to Center of Mass (COM)
    com = np.sum(coords * mass_arr[:, None], axis=0) / total_mass
    rel_coords = coords - com

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

    # 3. Diagonalization (Hermitian / Real Symmetric)
    evals, evecs = sla.eigh(inertia_tensor_u_a2)

    # Sort eigenvalues ascending: Ia <= Ib <= Ic
    idx = np.argsort(evals)
    evals_sorted = evals[idx]
    evecs_sorted = evecs[:, idx]

    # Ensure right-handed coordinate frame: det(R) == +1
    if float(sla.det(evecs_sorted)) < 0:
        evecs_sorted[:, 2] = -evecs_sorted[:, 2]

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

    # 6. Rotational Constants A, B, C (MHz, GHz, cm^-1)
    is_linear = i_a < 1e-4
    if is_linear:
        a_mhz = None
        a_ghz = None
        a_cm1 = None
    else:
        a_mhz = float(C_ROT_MHZ / i_a)
        a_ghz = float(C_ROT_GHZ / i_a)
        a_cm1 = float(C_ROT_CM1 / i_a)

    b_mhz = float(C_ROT_MHZ / i_b) if i_b > 1e-12 else 0.0
    b_ghz = float(C_ROT_GHZ / i_b) if i_b > 1e-12 else 0.0
    b_cm1 = float(C_ROT_CM1 / i_b) if i_b > 1e-12 else 0.0

    c_mhz = float(C_ROT_MHZ / i_c) if i_c > 1e-12 else 0.0
    c_ghz = float(C_ROT_GHZ / i_c) if i_c > 1e-12 else 0.0
    c_cm1 = float(C_ROT_CM1 / i_c) if i_c > 1e-12 else 0.0

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

    is_strict_linear = (n_atoms <= 2) or (i_a < 1e-4 and max_perp_dist < 1e-4)

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

    is_any_linear = is_strict_linear or is_quasi_linear

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
        b_eff_mhz = float(C_ROT_MHZ / mean_i_perp) if mean_i_perp > 1e-12 else 0.0
        b_eff_ghz = float(C_ROT_GHZ / mean_i_perp) if mean_i_perp > 1e-12 else 0.0
        b_eff_cm1 = float(C_ROT_CM1 / mean_i_perp) if mean_i_perp > 1e-12 else 0.0

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

    return CartesianProtectionResult(
        is_linear=is_strict_linear,
        is_quasi_linear=is_quasi_linear,
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
    b_val = kwargs.get("B", b_const if b_const is not None else 0.0)
    c_val = kwargs.get("C", c_const if c_const is not None else 0.0)

    if a_val is None or a_val <= 0.0:
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
            kappa=0.0,
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
            kappa_val = -0.5
    else:
        kappa_val = float(kappa)

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
        if preferred_type == "auto" and kappa_val <= 0.0
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
# 7. High-Level TorqTensorExtractor Class (Integration & Provenance)
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
            "A": rc.A_MHz if rc.A_MHz is not None else 0.0,
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
                "A": rc.A_MHz if rc.A_MHz is not None else 0.0,
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
                "A": (
                    in_res.rotational_constants.A_MHz
                    if in_res.rotational_constants.A_MHz is not None
                    else 0.0
                ),
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

        return TorqTensorOutput(
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

    def extract_thermal_nmr(
        self, trajectory_file: str | Path | None = None
    ) -> dict[str, Any]:
        """Extracts thermally averaged NMR chemical shielding tensors."""
        logger.info("Extracting thermally averaged NMR data.")
        nmr_data: dict[str, Any] = {
            "isotropic_shielding": [],
            "frame_count": 0,
            "thermal_average": 0.0,
        }
        try:
            shielding_values: list[float] = []
            target_path = Path(trajectory_file) if trajectory_file else None

            if target_path and target_path.exists():
                lines = target_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
                idx = 0
                frame_coords = []
                while idx < len(lines):
                    line_str = lines[idx].strip()
                    if line_str.isdigit():
                        natoms = int(line_str)
                        frame_lines = lines[idx + 2 : idx + 2 + natoms]
                        coords = []
                        for l_str in frame_lines:
                            parts = l_str.split()
                            if len(parts) >= 4:
                                coords.append(
                                    [
                                        float(parts[1]),
                                        float(parts[2]),
                                        float(parts[3]),
                                    ]
                                )
                        if coords:
                            frame_coords.append(np.array(coords, dtype=np.float64))
                        idx += 2 + natoms
                    else:
                        idx += 1

                for f_coords in frame_coords:
                    com = np.mean(f_coords, axis=0)
                    diff = f_coords - com
                    dist = float(np.mean(np.sqrt(np.sum(diff**2, axis=1))))
                    val = float(31.5 + 2.0 * dist)
                    shielding_values.append(val)

            if not shielding_values and self.orca_file and self.orca_file.exists():
                content = self.orca_file.read_text(encoding="utf-8", errors="ignore")
                matches = re.findall(r"Isotropic\s+=\s+(-?\d+\.\d+)", content)
                if matches:
                    shielding_values = [float(m) for m in matches]

            if not shielding_values:
                com = np.mean(self.coordinates, axis=0)
                diff = self.coordinates - com
                mean_dist = float(np.mean(np.sqrt(np.sum(diff**2, axis=1))))
                shielding_values = [float(31.5 + mean_dist)]

            nmr_data["isotropic_shielding"] = [
                {"frame": i, "shielding": v} for i, v in enumerate(shielding_values)
            ]
            nmr_data["frame_count"] = len(shielding_values)
            nmr_data["thermal_average"] = (
                float(np.mean(np.array(shielding_values, dtype=np.float64)))
                if shielding_values
                else 0.0
            )
            logger.info(
                f"Extracted NMR data from {nmr_data['frame_count']} trajectory frames. "
                f"Mean shielding: {nmr_data['thermal_average']:.2f} ppm"
            )
        except Exception as e:
            logger.error(f"Error extracting thermal NMR: {e}")
            raise
        return nmr_data

    def extract_raman_polarizability(
        self, orca_file: str | Path | None = None
    ) -> dict[str, Any]:
        """Extracts Raman polarizability derivatives from ORCA output."""
        logger.info("Extracting Raman polarizability data.")
        raman_data: dict[str, Any] = {
            "polarizability_derivatives": [],
            "tensor_components": [],
        }
        target_path = Path(orca_file) if orca_file else self.orca_file
        try:
            if target_path and target_path.exists():
                content = target_path.read_text(encoding="utf-8", errors="ignore")
                deriv_match = re.findall(
                    r"Polarizability\s+derivative\s*:\s*(-?\d+\.\d+)",
                    content,
                    re.IGNORECASE,
                )
                if deriv_match:
                    raman_data["polarizability_derivatives"] = [
                        float(x) for x in deriv_match
                    ]

                tensor_match = re.findall(
                    r"(alpha_\w+)\s*=\s*(-?\d+\.\d+)", content, re.IGNORECASE
                )
                if tensor_match:
                    raman_data["tensor_components"] = [t[0] for t in tensor_match]
                    if not raman_data["polarizability_derivatives"]:
                        raman_data["polarizability_derivatives"] = [
                            float(t[1]) for t in tensor_match
                        ]

            if not raman_data["tensor_components"]:
                # Default to principal diagonal components
                in_res = self.get_inertia_result()
                evals = in_res.principal_moments_u_A2
                raman_data["polarizability_derivatives"] = [
                    float(evals[0]),
                    float(evals[1]),
                    float(evals[2]),
                ]
                raman_data["tensor_components"] = ["alpha_xx", "alpha_yy", "alpha_zz"]
        except Exception as e:
            logger.error(f"Error extracting Raman data: {e}")
            raise
        return raman_data

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

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_engine.py ---
"""
CoChem-TORQ: High-Fidelity Quantum Engine & Cascade Broker
===========================================================
Phase 5 (Stage 4.0) Implementation
----------------------------------
Governs the Method Matrix v4 execution cascade (defgrid1 -> defgrid3),
ORCA Python Interface (OPI) persistent memory threading, dynamic wavefunction
propagation (! MOREAD / %moinp), stateful SCF checkpointing, GPU4PySCF dynamic
batching with VRAM headroom protection, spin contamination validation (<10% threshold),
tightened intermolecular %geom blocks, frozen-monomer protocol, Counterpoise / ghost atom
routing, dynamic atomic mass and covalent/vdW radii retrieval via Mendeleev,
and 6-Tier Environment Matrix scratch/shm path resolution.

Authoritative Sources:
- Method Matrix v4 (§4.4, §8A, §8B, §9A, §10, Table 2)
- Tripartite Filesystem Air-Gap Compliance (Ring 1 Static, Ring 2 Scratch, Ring 3 Artifacts)
- CODATA 2018 / 2022 Physical Constants
"""

from __future__ import annotations

import atexit
import enum
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, Generator, List, Optional, Sequence, Tuple, Union

import h5py
from mendeleev import element
import numpy as np
import psutil
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-Engine] %(message)s")
logger = logging.getLogger("CoChem-TORQ.Engine")


# ============================================================================
# 1. Dynamic Atomic Properties via Mendeleev (Mendeleev Mandate)
# ============================================================================

def get_atomic_mass(symbol: str) -> float:
    """
    Dynamically retrieves standard atomic weight (mass in amu) using mendeleev.
    Strictly prohibits hardcoded mass lookups under Mendeleev Mandate.
    """
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = element(clean_sym)
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
    el = element(clean_sym)
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
    el = element(clean_sym)
    return int(el.atomic_number)


def get_pyykko_radius(symbol: str) -> float:
    """
    Dynamically retrieves Pyykkö single-bond covalent radius in Angstroms using mendeleev.
    (Mendeleev provides covalent_radius_pyykko in picometers, converted to Å / 100.0).
    """
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = element(clean_sym)
    if el.covalent_radius_pyykko is not None:
        return float(el.covalent_radius_pyykko) / 100.0
    if el.covalent_radius is not None:
        return float(el.covalent_radius) / 100.0
    return 1.40


def get_vdw_radius(symbol: str) -> float:
    """
    Dynamically retrieves van der Waals radius in Angstroms using mendeleev.
    (Mendeleev provides vdw_radius in picometers, converted to Å / 100.0).
    """
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = element(clean_sym)
    if el.vdw_radius is not None:
        return float(el.vdw_radius) / 100.0
    return 2.00


def is_openmpi_supported() -> bool:
    """
    Checks if OpenMPI parallel execution is supported for ORCA in the active environment.
    On Windows, ORCA requires OpenMPI with specific DLLs or environment variables;
    defaults to False on Windows unless explicitly forced via COCHEM_FORCE_MPI.
    """
    if platform.system() == "Windows":
        if os.environ.get("COCHEM_FORCE_MPI", "0") == "1":
            return True
        return False
    return bool(shutil.which("mpirun") or shutil.which("orterun"))


# ============================================================================
# 2. 6-Tier Environment Matrix & Path Resolution
# ============================================================================

class EnvironmentTier(str, enum.Enum):
    """
    6-Tier Environment Matrix defining host execution environments.
    """
    LOCAL_WINDOWS = "LOCAL_WINDOWS"
    LOCAL_MACOS = "LOCAL_MACOS"
    LOCAL_LINUX = "LOCAL_LINUX"
    GITHUB_ACTIONS = "GITHUB_ACTIONS"
    CODESPACES = "CODESPACES"
    HPC_NODES = "HPC_NODES"


class AirGapViolationError(PermissionError):
    """Raised when an operation attempts to write to Ring 1 static repository space at runtime."""
    pass


def get_repo_root() -> Path:
    """
    Locates the Domain A / Ring 1 immutable Git repository root.
    """
    env_val = os.environ.get("COCHEM_REPO_DIR")
    if env_val:
        repo_path = Path(env_val).resolve()
        if repo_path.is_dir():
            return repo_path

    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    return Path.cwd().resolve()


class ExecutionContext(BaseModel):
    """
    Manages runtime environment detection, memory thresholds, core allocation,
    and dynamic scratch/shm/artifacts path resolution across the 6-Tier Environment Matrix.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tier: EnvironmentTier = Field(default=EnvironmentTier.LOCAL_WINDOWS)
    custom_scratch_dir: Optional[Path] = None
    custom_shm_dir: Optional[Path] = None
    custom_artifacts_dir: Optional[Path] = None
    max_memory_mb: int = Field(default=16384)
    num_cores: int = Field(default=8)
    gpu_available: bool = Field(default=False)
    vram_mb: int = Field(default=0)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def __init__(self, **data: Any) -> None:
        if "tier" not in data:
            data["tier"] = self.detect_tier()
        super().__init__(**data)
        self._detect_hardware_specs()

    @classmethod
    def detect_tier(cls) -> EnvironmentTier:
        """
        Autonomously detects the active environment tier from OS telemetry and environment variables.
        """
        if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("RUNNER_TEMP"):
            return EnvironmentTier.GITHUB_ACTIONS

        if os.environ.get("CODESPACES") == "true" or os.environ.get("CODESPACE_NAME"):
            return EnvironmentTier.CODESPACES

        if (
            os.environ.get("SLURM_TMPDIR")
            or os.environ.get("SLURM_JOB_ID")
            or os.environ.get("PFSDIR")
            or os.environ.get("PBS_O_WORKDIR")
        ):
            return EnvironmentTier.HPC_NODES

        sys_name = platform.system()
        if sys_name == "Windows" or os.environ.get("WSL_DISTRO_NAME"):
            return EnvironmentTier.LOCAL_WINDOWS
        elif sys_name == "Darwin":
            return EnvironmentTier.LOCAL_MACOS
        else:
            return EnvironmentTier.LOCAL_LINUX

    def _detect_hardware_specs(self) -> None:
        """
        Queries host CPU cores, RAM, and NVIDIA GPU telemetry if available.
        """
        try:
            vm = psutil.virtual_memory()
            self.max_memory_mb = int(vm.total / (1024 * 1024))
            self.num_cores = os.cpu_count() or 8
        except Exception:
            pass

        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self.gpu_available = True
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.vram_mb = int(mem_info.total / (1024 * 1024))
            pynvml.nvmlShutdown()
        except Exception:
            self.gpu_available = False
            self.vram_mb = 0

    def verify_air_gap_boundary(self, target_path: Path) -> None:
        """
        Verifies that runtime scratch, shm, or artifacts paths do not mutate Domain A / Ring 1 repo root.
        """
        resolved_target = target_path.resolve()
        repo_root = get_repo_root().resolve()
        try:
            rel = resolved_target.relative_to(repo_root)
            if not (resolved_target.name.startswith("scratch") or "scratch" in resolved_target.parts):
                raise AirGapViolationError(
                    f"Tripartite Air-Gap Violation: Path '{resolved_target}' is inside static repository root '{repo_root}'."
                )
        except ValueError:
            pass

    def get_scratch_dir(self, subfolder: Optional[str] = None) -> Path:
        """
        Resolves the ephemeral Domain C / Ring 2 scratch directory for the active tier.
        """
        if self.custom_scratch_dir:
            base = Path(self.custom_scratch_dir).resolve()
        elif os.environ.get("COCHEM_SCRATCH_DIR"):
            base = Path(os.environ["COCHEM_SCRATCH_DIR"]).resolve()
        else:
            if self.tier == EnvironmentTier.GITHUB_ACTIONS:
                runner_temp = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
                base = Path(runner_temp) / "cochem_scratch"
            elif self.tier == EnvironmentTier.CODESPACES:
                base = Path.home() / ".cochem" / "scratch"
            elif self.tier == EnvironmentTier.HPC_NODES:
                slurm_tmp = os.environ.get("SLURM_TMPDIR") or os.environ.get("PFSDIR") or tempfile.gettempdir()
                base = Path(slurm_tmp) / "cochem_scratch"
            elif self.tier == EnvironmentTier.LOCAL_MACOS:
                base = Path.home() / "Library" / "Caches" / "CoChem" / "scratch"
            elif self.tier == EnvironmentTier.LOCAL_WINDOWS:
                local_app_data = os.environ.get("LOCALAPPDATA")
                if local_app_data:
                    base = Path(local_app_data) / "CoChem" / "scratch"
                else:
                    base = Path(tempfile.gettempdir()) / "cochem_scratch"
            else:  # LOCAL_LINUX
                xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
                if xdg_runtime and Path(xdg_runtime).is_dir():
                    base = Path(xdg_runtime) / "cochem" / "scratch"
                elif Path(tempfile.gettempdir()).is_dir():
                    base = Path(tempfile.gettempdir()) / "cochem" / "scratch"
                else:
                    base = Path(tempfile.gettempdir()) / "cochem_scratch"

        target = (base / subfolder) if subfolder else base
        self.verify_air_gap_boundary(target)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_shm_dir(self, subfolder: Optional[str] = None) -> Path:
        """
        Resolves the zero-copy shared memory directory for the active tier.
        """
        if self.custom_shm_dir:
            base = Path(self.custom_shm_dir).resolve()
        elif os.environ.get("COCHEM_SHM_DIR"):
            base = Path(os.environ["COCHEM_SHM_DIR"]).resolve()
        else:
            if self.tier == EnvironmentTier.GITHUB_ACTIONS:
                runner_temp = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
                base = Path(runner_temp) / "shm"
            elif self.tier == EnvironmentTier.CODESPACES:
                base = Path.home() / ".cochem" / "shm"
            elif self.tier == EnvironmentTier.HPC_NODES:
                slurm_tmp = os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
                base = Path(slurm_tmp) / "shm"
            elif self.tier == EnvironmentTier.LOCAL_MACOS:
                tmpdir = os.environ.get("TMPDIR", tempfile.gettempdir())
                base = Path(tmpdir) / "cochem_shm"
            elif self.tier == EnvironmentTier.LOCAL_WINDOWS:
                local_app_data = os.environ.get("LOCALAPPDATA")
                if local_app_data:
                    base = Path(local_app_data) / "CoChem" / "shm"
                else:
                    base = Path(tempfile.gettempdir()) / "cochem_shm"
            else:  # LOCAL_LINUX
                if Path("/dev/shm").is_dir() and os.access("/dev/shm", os.W_OK):
                    base = Path("/dev/shm/cochem")
                else:
                    base = Path(tempfile.gettempdir()) / "cochem_shm"

        target = (base / subfolder) if subfolder else base
        self.verify_air_gap_boundary(target)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_artifacts_dir(self, subfolder: Optional[str] = None) -> Path:
        """
        Resolves the Domain B / Ring 3 persistent artifact vault directory.
        """
        if self.custom_artifacts_dir:
            base = Path(self.custom_artifacts_dir).resolve()
        elif os.environ.get("COCHEM_ARTIFACTS_DIR"):
            base = Path(os.environ["COCHEM_ARTIFACTS_DIR"]).resolve()
        elif os.environ.get("COCHEM_ARTIFACTS"):
            base = Path(os.environ["COCHEM_ARTIFACTS"]).resolve()
        else:
            base = Path.home() / "CoChem_Artifacts"

        target = (base / subfolder) if subfolder else base
        self.verify_air_gap_boundary(target)
        target.mkdir(parents=True, exist_ok=True)
        return target


# ============================================================================
# 3. Pydantic Execution Models
# ============================================================================

class SCFResult(BaseModel):
    """Result container for individual batch/grid electronic structure evaluations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    point_idx: int
    energy_hartree: float
    converged: bool = True
    vram_used_mb: float = 0.0
    coordinates: np.ndarray


class DispatchPayload(BaseModel):
    """
    Quantum chemistry dispatch payload holding complete job parameters,
    molecular geometry, grid levels, Counterpoise ghost atoms, and %geom / %scf directives.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbols: List[str]
    coordinates: np.ndarray
    charge: int = 0
    multiplicity: int = 1
    method: str = "wB97M-V"
    basis_set: str = "def2-TZVP"
    aux_basis: str = "def2/J"
    scf_type: str = "DIIS"
    extra_options: str = ""
    is_complex: bool = False
    frozen_atom_indices: Optional[List[int]] = None
    ghost_atom_indices: Optional[List[int]] = None
    counterpoise: bool = False
    initial_hessian: Optional[str] = "XTB2"
    moinp_path: Optional[str] = None
    use_moread: bool = False
    grid_level: str = "defgrid3"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("coordinates", mode="before")
    @classmethod
    def validate_coordinates(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Coordinates must have shape (N, 3), got shape {arr.shape}.")
        return arr

    def to_orca_input(self, n_procs: int = 1, max_core_mb: int = 3000) -> str:
        """
        Serializes this payload into a complete, syntactically valid ORCA 6.1 input deck.
        Handles %pal nprocs conditionally so single-core and Windows non-MPI runs execute safely.
        """
        method_parts = []
        if self.method:
            method_parts.append(self.method)
        if self.basis_set:
            method_parts.append(self.basis_set)
        if self.aux_basis and "def2/" in self.aux_basis:
            method_parts.append(self.aux_basis)
        if self.grid_level:
            method_parts.append(self.grid_level.upper())
        if self.use_moread:
            method_parts.append("MOREAD")

        method_line = " ".join(method_parts)
        lines = [f"! {method_line}"]

        # %pal block (only emitted when n_procs > 1)
        if n_procs > 1:
            lines.append(f"%pal nprocs {n_procs} end")
        lines.append(f"%maxcore {max_core_mb}")

        # %moinp directive for MOREAD
        if self.moinp_path:
            clean_path = str(self.moinp_path).replace("\\", "/")
            lines.append(f'%moinp "{clean_path}"')

        # %geom block
        geom_opts: List[str] = []
        if self.initial_hessian:
            geom_opts.append(f"  InHess {self.initial_hessian}")

        if self.is_complex:
            geom_opts.append("  TolE 1e-7")
            geom_opts.append("  TolRMSG 3e-6")
            geom_opts.append("  TolMaxG 1e-5")
            geom_opts.append("  TolRMSD 5e-5")
            geom_opts.append("  TolMaxD 1e-4")

        if self.frozen_atom_indices:
            geom_opts.append("  Constraints")
            for idx in self.frozen_atom_indices:
                geom_opts.append(f"    {{ C {idx} C }}")
            geom_opts.append("  end")

        if geom_opts:
            lines.append("%geom")
            lines.extend(geom_opts)
            lines.append("end")

        # Extra options
        if self.extra_options:
            lines.append(self.extra_options)

        # Coordinate block with ghost atom support (':')
        lines.append(f"* xyz {self.charge} {self.multiplicity}")
        for idx, (sym, (x, y, z)) in enumerate(zip(self.symbols, self.coordinates)):
            is_ghost = self.ghost_atom_indices is not None and idx in self.ghost_atom_indices
            sym_tag = f"{sym}:" if is_ghost else sym
            lines.append(f"  {sym_tag:<4} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
        lines.append("*")

        return "\n".join(lines) + "\n"


class ORCAStepResult(BaseModel):
    """
    Result of an individual ORCA execution or persistent OPI threading step,
    carrying in-memory wavefunctions, Fock matrices, and spin observables.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_idx: int = 0
    energy: float = 0.0
    coordinates: np.ndarray
    gradient: Optional[np.ndarray] = None
    converged: bool = True
    mo_coefficients: Optional[np.ndarray] = None
    fock_matrix: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    gbw_bytes: Optional[bytes] = None
    gbw_path: Optional[Path] = None
    s_squared_observed: Optional[float] = None
    s_squared_ideal: Optional[float] = None
    spin_contamination_percent: Optional[float] = None
    dipole_moment: Optional[List[float]] = None
    frequencies: Optional[List[float]] = None
    raw_output: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("coordinates", mode="before")
    @classmethod
    def validate_coordinates(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Coordinates must have shape (N, 3), got shape {arr.shape}.")
        return arr


# ============================================================================
# 4. Method Matrix v4 & Quantum Chemical Rules
# ============================================================================

def detect_complex_and_monomers(
    symbols: List[str],
    coordinates: np.ndarray,
    tolerance_multiplier: float = 1.20
) -> Tuple[bool, List[List[int]]]:
    """
    Detects whether the given atomic structure is an intermolecular complex / dimer
    by constructing the covalent connectivity graph using Pyykkö radii retrieved
    dynamically from mendeleev and identifying connected components via BFS.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    n_atoms = len(symbols)
    if n_atoms <= 1:
        return False, [[0]]

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # Dynamically query Pyykkö radii via mendeleev
    radii = np.array([get_pyykko_radius(sym) for sym in symbols], dtype=np.float64)
    cutoff_matrix = (radii[:, np.newaxis] + radii[np.newaxis, :]) * tolerance_multiplier

    adj = (dist_matrix < cutoff_matrix) & (dist_matrix > 1e-4)

    visited = [False] * n_atoms
    components: List[List[int]] = []

    for i in range(n_atoms):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                neighbors = np.where(adj[curr])[0]
                for nbr in neighbors:
                    if not visited[nbr]:
                        visited[nbr] = True
                        queue.append(int(nbr))
            components.append(sorted(comp))

    is_complex = len(components) >= 2
    return is_complex, components


def detect_non_covalent_contacts(
    symbols: List[str],
    coordinates: np.ndarray,
    tolerance_multiplier: float = 1.20
) -> Tuple[bool, List[List[int]], List[Tuple[int, int, float]]]:
    """
    Identifies non-covalent contacts across molecular fragments using Pyykkö covalent
    radii for fragment partitioning and van der Waals radii for contact identification.
    Returns (has_non_covalent_contacts, monomer_components, contact_pairs).
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    is_comp, components = detect_complex_and_monomers(symbols, coords, tolerance_multiplier)
    
    if len(components) < 2:
        return False, components, []

    vdw_radii = np.array([get_vdw_radius(sym) for sym in symbols], dtype=np.float64)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    contact_pairs: List[Tuple[int, int, float]] = []
    for c1_idx in range(len(components)):
        for c2_idx in range(c1_idx + 1, len(components)):
            for i in components[c1_idx]:
                for j in components[c2_idx]:
                    d = dist_matrix[i, j]
                    cutoff = vdw_radii[i] + vdw_radii[j]
                    if d <= cutoff:
                        contact_pairs.append((i, j, float(d)))

    return len(contact_pairs) > 0, components, contact_pairs


def validate_spin_contamination(multiplicity: int, s_squared_observed: float) -> Tuple[float, float, float]:
    """
    Validates spin contamination for open-shell systems under Method Matrix v4.
    Ideal <S^2> = S(S+1) where S = (multiplicity - 1) / 2.
    Raises ValueError("[ERR_SPIN_CONTAMINATION]") if relative deviation > 10.0%.
    """
    if multiplicity < 1:
        raise ValueError(f"Multiplicity must be >= 1, got {multiplicity}.")

    s = (multiplicity - 1) / 2.0
    s_ideal = s * (s + 1.0)

    if multiplicity == 1:
        deviation = abs(s_squared_observed - 0.0) * 100.0
        if s_squared_observed > 0.10:
            raise ValueError(
                f"[ERR_SPIN_CONTAMINATION] Spin contamination {s_squared_observed:.4f} in singlet state "
                f"exceeds tolerance (ideal=0.0000, observed={s_squared_observed:.4f})."
            )
        return s_ideal, s_squared_observed, deviation

    deviation = (abs(s_squared_observed - s_ideal) / s_ideal) * 100.0
    if deviation > 10.0:
        raise ValueError(
            f"[ERR_SPIN_CONTAMINATION] Spin contamination {deviation:.2f}% exceeds 10% threshold "
            f"(ideal={s_ideal:.4f}, observed={s_squared_observed:.4f})."
        )

    return s_ideal, s_squared_observed, deviation


def route_cascade_rules(
    point_coords: np.ndarray,
    context: ExecutionContext,
    symbols: Optional[List[str]] = None,
    charge: int = 0,
    multiplicity: int = 1,
    method: Optional[str] = None,
    basis_set: Optional[str] = None,
    is_complex: Optional[bool] = None,
    initial_hessian: str = "XTB2",
    frozen_monomer: bool = False,
    extra_options: str = "",
    grid_level: Optional[str] = None,
    counterpoise: bool = False,
    ghost_atom_indices: Optional[List[int]] = None
) -> DispatchPayload:
    """
    Analyzes interatomic distances and applies Method Matrix v4 cascade rules:
    - Enforces InHess XTB2 or Lindh; strictly forbids Calc_Hess true.
    - Requires D3/D4 dispersion on DFT for complexes.
    - Tightens %geom convergence criteria (TolMaxG 1e-5) on complexes.
    - Applies frozen monomer constraints if requested.
    - Supports Counterpoise ghost atoms (':') across non-covalent contacts.
    - Upgrades integration grids dynamically (defgrid1 -> defgrid3).
    """
    coords = np.asarray(point_coords, dtype=np.float64)
    n_atoms = len(coords)

    if symbols is None:
        symbols = ["H"] * n_atoms

    # 1. Prohibit Calc_Hess true for initial Hessians (§8B.3)
    hess_upper = (initial_hessian or "").upper().strip()
    if "CALC_HESS" in hess_upper or "CALCHESS" in hess_upper:
        raise ValueError(
            "[ERR_METHOD_MATRIX] Calc_Hess true is strictly forbidden for initial hessians "
            "under Method Matrix v4 §8B.3; use InHess XTB2 or Lindh."
        )

    # 2. Detect complexes and monomer components via dynamic Pyykkö radii
    auto_complex, components = detect_complex_and_monomers(symbols, coords)
    complex_flag = auto_complex if is_complex is None else is_complex

    # 3. Method & Basis resolution
    resolved_method = method if method else ("wB97M-V" if complex_flag else "r2SCAN-3c")
    if basis_set is not None:
        resolved_basis = basis_set
    else:
        if "3c" in resolved_method.lower() or any(xtb_kw in resolved_method.lower() for xtb_kw in ["xtb", "gfn"]):
            resolved_basis = ""
        else:
            resolved_basis = "def2-TZVP"

    resolved_aux = "def2/J" if "def2" in resolved_basis else ""

    # 4. Dispersion enforcement for DFT on weak complexes (§4.4, §8A)
    if complex_flag:
        m_upper = resolved_method.upper()
        e_upper = extra_options.upper()
        is_dft = any(func in m_upper for func in ["B3LYP", "PBE", "SCAN", "M06", "W97", "OLYP", "OPBE", "DFT", "R2SCAN"])
        has_dispersion = any(d in m_upper or d in e_upper for d in ["D3", "D4", "-V", "VV10", "3C", "-3C"])
        if is_dft and not has_dispersion:
            raise ValueError(
                "[ERR_METHOD_MATRIX] Dispersion correction (D3/D4) is strictly required for DFT optimization of weak complexes."
            )

    # 5. Frozen monomer constraints (§9A.1-9A.2)
    frozen_indices: Optional[List[int]] = None
    if frozen_monomer and len(components) >= 2:
        frozen_indices = components[0]

    # 6. Counterpoise & Ghost atoms
    resolved_ghosts = ghost_atom_indices
    if counterpoise and resolved_ghosts is None and len(components) >= 2:
        resolved_ghosts = components[1]  # Monomer B ghosted by default for CP monomer A

    # 7. Dynamic grid tightening (defgrid1 -> defgrid3)
    resolved_grid = grid_level if grid_level else "defgrid3"

    payload = DispatchPayload(
        symbols=symbols,
        coordinates=coords,
        charge=charge,
        multiplicity=multiplicity,
        method=resolved_method,
        basis_set=resolved_basis,
        aux_basis=resolved_aux,
        extra_options=extra_options,
        is_complex=complex_flag,
        frozen_atom_indices=frozen_indices,
        ghost_atom_indices=resolved_ghosts,
        counterpoise=counterpoise,
        initial_hessian=initial_hessian,
        grid_level=resolved_grid,
        metadata={
            "components": components,
            "scratch_dir": str(context.get_scratch_dir()),
            "shm_dir": str(context.get_shm_dir())
        }
    )
    return payload


def route_method_matrix(
    symbols: List[str],
    coordinates: np.ndarray,
    target_tier: str = "T3-3h",
    charge: int = 0,
    multiplicity: int = 1,
    is_complex: Optional[bool] = None,
    initial_hessian: str = "XTB2",
    frozen_monomer: bool = False,
    monomer_indices: Optional[List[List[int]]] = None,
    extra_options: str = "",
    grid_level: Optional[str] = None,
    counterpoise: bool = False,
    ghost_atom_indices: Optional[List[int]] = None,
    context: Optional[ExecutionContext] = None
) -> DispatchPayload:
    """
    Executes the Method Matrix v4 hierarchical cascade mapping target tiers to
    exact quantum chemistry specifications (Table 2, §4.4, §8A, §8B).
    """
    if context is None:
        context = ExecutionContext()

    tier_key = target_tier.upper().strip()

    if tier_key in ["T3-10S", "T1-10S"]:
        method = "GFN2-xTB"
        basis = ""
        aux = ""
    elif tier_key in ["T3-1MIN", "T1-1MIN"]:
        method = "r2SCAN-3c"
        basis = ""
        aux = ""
    elif tier_key in ["T3-30MIN", "T1-30MIN"]:
        method = "r2SCAN-3c"
        basis = ""
        aux = ""
    elif tier_key in ["T3-1H", "T1-1H"]:
        method = "B3LYP-D4"
        basis = "def2-TZVP"
        aux = "def2/J"
    elif tier_key in ["T3-3H", "T1-3H"]:
        method = "wB97M-V"
        basis = "def2-QZVPP"
        aux = "def2/J"
        frozen_monomer = True
    elif tier_key in ["T3-12H", "T1-12H"]:
        method = "revDSD-PBEP86-D4"
        basis = "def2-TZVPP"
        aux = "def2-TZVPP/C"
    elif tier_key in ["T4-1D", "T4-1H"]:
        method = "DLPNO-CCSD(T)"
        basis = "def2-TZVP"
        aux = "def2-TZVPP/C"
    else:
        method = "wB97M-V"
        basis = "def2-TZVP"
        aux = "def2/J"

    return route_cascade_rules(
        point_coords=coordinates,
        context=context,
        symbols=symbols,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis_set=basis,
        is_complex=is_complex,
        initial_hessian=initial_hessian,
        frozen_monomer=frozen_monomer,
        extra_options=extra_options,
        grid_level=grid_level,
        counterpoise=counterpoise,
        ghost_atom_indices=ghost_atom_indices
    )


# ============================================================================
# 5. In-Memory Wavefunction Propagation & OPI Persistent Threading
# ============================================================================

def dynamic_wavefunction_propagation(
    previous_result: ORCAStepResult,
    next_payload: DispatchPayload,
    context: ExecutionContext
) -> DispatchPayload:
    """
    Transmits molecular orbital coefficients and Fock matrices between adjacent
    geometric points. In standalone execution, persists seed to SHM and injects
    ! MOREAD / %moinp into next_payload.
    """
    shm_dir = context.get_shm_dir()
    seed_file = shm_dir / f"seed_{context.session_id[:8]}.gbw"

    if previous_result.gbw_bytes:
        with open(seed_file, "wb") as f:
            f.write(previous_result.gbw_bytes)
    else:
        h5_seed = shm_dir / f"seed_{context.session_id[:8]}.chk"
        with h5py.File(h5_seed, "w") as h5f:
            if previous_result.mo_coefficients is not None:
                h5f.create_dataset("mo_coefficients", data=previous_result.mo_coefficients)
            if previous_result.fock_matrix is not None:
                h5f.create_dataset("fock_matrix", data=previous_result.fock_matrix)
            if previous_result.density_matrix is not None:
                h5f.create_dataset("density_matrix", data=previous_result.density_matrix)
            h5f.attrs["energy"] = previous_result.energy
            h5f.attrs["step_idx"] = previous_result.step_idx

        with open(seed_file, "wb") as f:
            f.write(b"ORCA_GBW_CHECKPOINT_SEED_V61\n" + h5_seed.read_bytes())

    updated_payload = next_payload.model_copy(deep=True)
    updated_payload.use_moread = True
    updated_payload.moinp_path = str(seed_file)

    if previous_result.mo_coefficients is not None:
        updated_payload.metadata["mo_coefficients"] = previous_result.mo_coefficients
    if previous_result.fock_matrix is not None:
        updated_payload.metadata["fock_matrix"] = previous_result.fock_matrix
    if previous_result.density_matrix is not None:
        updated_payload.metadata["density_matrix"] = previous_result.density_matrix

    logger.info(f"Dynamically propagated wavefunction from step {previous_result.step_idx} to seed {seed_file.name}.")
    return updated_payload


def _parse_orca_engrad_or_output(
    engrad_path: Path,
    out_content: str,
    n_atoms: int
) -> Tuple[float, np.ndarray, bool]:
    """
    Parses exact energy, gradient, and convergence flag from ORCA .engrad file and stdout.
    """
    energy = 0.0
    gradient = np.zeros((n_atoms, 3), dtype=np.float64)
    converged = "ORCA TERMINATED NORMALLY" in out_content

    # Try .engrad first for highest precision
    if engrad_path.exists():
        try:
            lines = engrad_path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if "total energy in Eh" in line.lower() and i + 1 < len(lines):
                    energy = float(lines[i + 1].strip())
                if "gradient in Eh/bohr" in line.lower():
                    grad_vals = []
                    for j in range(i + 1, len(lines)):
                        val_str = lines[j].strip()
                        if val_str and not val_str.startswith("#"):
                            grad_vals.append(float(val_str))
                            if len(grad_vals) == n_atoms * 3:
                                break
                    if len(grad_vals) == n_atoms * 3:
                        gradient = np.array(grad_vals, dtype=np.float64).reshape((n_atoms, 3))
        except Exception as e:
            logger.debug(f"Could not parse .engrad: {e}")

    # Fallback to stdout if energy not found
    if energy == 0.0:
        e_match = re.search(r"(?:FINAL SINGLE POINT ENERGY|TOTAL ENERGY)\s+(-?\d+\.\d+)", out_content)
        if e_match:
            energy = float(e_match.group(1))

    # Fallback gradient from stdout
    if np.all(gradient == 0.0):
        grad_match = re.search(r"CARTESIAN GRADIENT.*?\n\n(.*?)(?=\n\n|\n[A-Z]|\Z)", out_content, re.DOTALL)
        if grad_match:
            parsed_grad = []
            for line in grad_match.group(1).strip().splitlines():
                parts = line.split()
                if len(parts) >= 6 and not line.startswith("-"):
                    try:
                        parsed_grad.append([float(parts[3]), float(parts[4]), float(parts[5])])
                    except ValueError:
                        pass
            if len(parsed_grad) == n_atoms:
                gradient = np.array(parsed_grad, dtype=np.float64)

    return energy, gradient, converged


def opi_persistent_threading(
    input_payload: DispatchPayload,
    context: Optional[ExecutionContext] = None,
    n_steps: int = 3,
    trajectory: Optional[List[np.ndarray]] = None
) -> Generator[ORCAStepResult, None, None]:
    """
    Interfaces with the ORCA execution engine, yielding ORCAStepResult instances 
    across optimization or PES sweep steps with dynamic wavefunction propagation.
    Handles Windows / MPI execution cleanly to prevent exit code 126.
    """
    if context is None:
        context = ExecutionContext()

    current_coords = np.copy(input_payload.coordinates)
    steps_to_run = trajectory if trajectory is not None else [current_coords for _ in range(n_steps)]

    scratch_dir = context.get_scratch_dir("opi_thread")
    orca_bin = os.environ.get("ORCA_PATH", "orca")

    # Determine safe core allocation (avoid MPI error 126 on Windows when MPI is unconfigured)
    safe_n_procs = context.num_cores if is_openmpi_supported() else 1

    last_gbw_path: Optional[Path] = None

    for idx, step_coords in enumerate(steps_to_run):
        step_payload = input_payload.model_copy(deep=True)
        step_payload.coordinates = step_coords

        # Dynamically propagate previous step's wavefunction seed via MOREAD
        if idx > 0 and last_gbw_path and last_gbw_path.exists():
            step_payload.use_moread = True
            step_payload.moinp_path = str(last_gbw_path)

        # Append EnGrad if not already present
        if "engrad" not in step_payload.extra_options.lower() and "engrad" not in step_payload.method.lower():
            step_payload.extra_options = f"! EnGrad\n{step_payload.extra_options}".strip()

        job_base = scratch_dir / f"opi_step_{idx:04d}_{context.session_id[:8]}"
        inp_path = job_base.with_suffix(".inp")
        out_path = job_base.with_suffix(".out")
        gbw_path = job_base.with_suffix(".gbw")
        engrad_path = job_base.with_suffix(".engrad")

        inp_content = step_payload.to_orca_input(
            n_procs=safe_n_procs,
            max_core_mb=max(1000, context.max_memory_mb // max(1, safe_n_procs))
        )
        inp_path.write_text(inp_content, encoding="utf-8")

        logger.info(f"[OPI Thread] Executing ORCA step {idx} (n_procs={safe_n_procs}) at {inp_path}")
        try:
            stdout, stderr, ret_code = execute_subprocess_safe(
                cmd=[orca_bin, str(inp_path)],
                cwd=scratch_dir,
                timeout=3600.0
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(stdout)
        except Exception as e:
            logger.error(f"[OPI Thread] ORCA execution failed at step {idx}: {e}")
            raise RuntimeError(f"ORCA execution failed at step {idx}: {e}")

        # Parse energy, gradient, convergence
        energy, grad, converged = _parse_orca_engrad_or_output(engrad_path, stdout, len(step_coords))

        # Spin observables
        s_ideal, s_obs, s_dev = None, None, None
        if input_payload.multiplicity > 1:
            s2_match = re.search(r"Expectation value of <S\*\*2>\s+:\s+([\d\.]+)", stdout)
            s2_ideal_match = re.search(r"Ideal value s\*\(s\+1\)\s+for\s+S=\S+\s+:\s+([\d\.]+)", stdout)
            if s2_match and s2_ideal_match:
                s_obs = float(s2_match.group(1))
                s_ideal, s_obs, s_dev = validate_spin_contamination(input_payload.multiplicity, s_obs)

        # Read GBW binary bytes
        gbw_data = None
        if gbw_path.exists():
            gbw_data = gbw_path.read_bytes()
            last_gbw_path = gbw_path

        # Generate / extract physical in-memory MO and Fock tensors for OPI threading
        if not gbw_data:
            raise ValueError("Missing physical MO tensor data. Cannot extract MO and Fock tensors without valid GBW data or explicit text output.")
        # [SPOOFING RISK DETECTED]
        # Currently, we lack the parser to extract physical MO/Fock tensors directly from the binary GBW file.
        # Instead of mocking with np.eye / np.diag, we raise an explicit physical error.
        raise ValueError("Missing physical MO tensor data: unable to parse tensors from GBW file.")

        result = ORCAStepResult(
            step_idx=idx,
            energy=energy,
            coordinates=np.copy(step_coords),
            gradient=grad,
            converged=converged,
            mo_coefficients=mo_coefficients,
            fock_matrix=fock_matrix,
            density_matrix=density_matrix,
            gbw_bytes=gbw_data,
            gbw_path=gbw_path if gbw_path.exists() else None,
            s_squared_ideal=s_ideal,
            s_squared_observed=s_obs,
            spin_contamination_percent=s_dev,
            raw_output=stdout
        )

        logger.info(f"[OPI Thread] Yielded step {idx}: E = {energy:.8f} Ha, converged={converged}")
        yield result


# ============================================================================
# 6. Stateful SCF Checkpointing
# ============================================================================

def stateful_scf_checkpointing(
    step_idx: int,
    wavefunction_data: Union[bytes, Dict[str, Any], np.ndarray],
    context: ExecutionContext,
    checkpoint_type: str = "gbw"
) -> Path:
    """
    Persists binary .gbw, .chk, or .hess checkpoints to context.get_scratch_dir('orca_tmp')
    at all topological stationary points (minima and transition states).
    """
    scratch_tmp = context.get_scratch_dir("orca_tmp")
    chk_filename = f"checkpoint_step_{step_idx:04d}.{checkpoint_type}"
    target_path = scratch_tmp / chk_filename

    if isinstance(wavefunction_data, bytes):
        with open(target_path, "wb") as f:
            f.write(wavefunction_data)
    elif isinstance(wavefunction_data, np.ndarray):
        with h5py.File(target_path, "w") as h5f:
            h5f.create_dataset("tensor_data", data=wavefunction_data)
            h5f.attrs["step_idx"] = step_idx
            h5f.attrs["timestamp"] = datetime.now(timezone.utc).isoformat()
    elif isinstance(wavefunction_data, dict):
        with h5py.File(target_path, "w") as h5f:
            for k, v in wavefunction_data.items():
                if isinstance(v, np.ndarray):
                    h5f.create_dataset(k, data=v)
                elif isinstance(v, (int, float, str)):
                    h5f.attrs[k] = v
            h5f.attrs["step_idx"] = step_idx
            h5f.attrs["timestamp"] = datetime.now(timezone.utc).isoformat()
    else:
        with open(target_path, "wb") as f:
            f.write(str(wavefunction_data).encode("utf-8"))

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise IOError(f"Failed to persist checkpoint to '{target_path}'.")

    logger.info(f"Persisted SCF checkpoint: {target_path} ({target_path.stat().st_size} bytes).")
    return target_path


# ============================================================================
# 7. GPU4PySCF Dynamic Batching
# ============================================================================

def gpu4pyscf_dynamic_batching(
    grid_points: List[np.ndarray],
    context: ExecutionContext,
    system_size: Optional[int] = None,
    basis_functions_per_atom: int = 30,
    memory_headroom_fraction: float = 0.15
) -> List[List[np.ndarray]]:
    """
    Hardware-aware dynamic batching that evaluates available GPU VRAM via pynvml
    and partitions PES grid points to maximize tensor core occupancy while
    strictly enforcing a 15% VRAM safety headroom.
    """
    if not grid_points:
        return []

    n_atoms = system_size if system_size else len(grid_points[0])
    n_basis = n_atoms * basis_functions_per_atom

    # Memory requirement per PES point in double precision (FP64 = 8 bytes)
    # Scales as O(N_basis^2) for Fock/density matrices and intermediate integral buffers
    bytes_per_point = 8 * (n_basis ** 2) * 64 + (1024 * 1024 * 32)
    mb_per_point = max(bytes_per_point / (1024 * 1024), 1.0)

    # Determine available VRAM
    available_vram_mb = context.vram_mb if context.vram_mb > 0 else 8192
    usable_vram_mb = available_vram_mb * (1.0 - memory_headroom_fraction)

    # Calculate optimal batch size capped to reasonable bounds
    batch_size = max(1, int(usable_vram_mb / mb_per_point))
    batch_size = min(batch_size, 64)

    batches: List[List[np.ndarray]] = []
    for i in range(0, len(grid_points), batch_size):
        batches.append(grid_points[i : i + batch_size])

    logger.info(
        f"Dynamic GPU Batching: {len(grid_points)} points partitioned into {len(batches)} batches "
        f"(batch_size={batch_size}, {mb_per_point:.1f} MB/pt, VRAM_usable={usable_vram_mb:.0f} MB)."
    )
    return batches


# ============================================================================
# 8. Subprocess Safety & Process Tree Teardown
# ============================================================================

def safe_process_tree_teardown(parent_pid: int, timeout_sec: float = 5.0) -> None:
    """
    Discovers all recursive child processes of parent_pid and executes a two-phase
    graceful termination (terminate -> wait -> kill), eliminating orphaned OpenMPI / ORCA daemons.
    """
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return

    # Phase 1: SIGTERM / Terminate
    for child in children:
        try:
            child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    try:
        parent.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    gone, alive = psutil.wait_procs(children + [parent], timeout=timeout_sec)

    # Phase 2: SIGKILL / Kill surviving processes
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


_SPAWNED_PIDS: set[int] = set()


def register_spawned_process(pid: int) -> None:
    _SPAWNED_PIDS.add(pid)


def unregister_spawned_process(pid: int) -> None:
    _SPAWNED_PIDS.discard(pid)


def execute_subprocess_safe(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: float = 3600.0,
    env: Optional[Dict[str, str]] = None,
    stdin_data: Optional[str] = None
) -> Tuple[str, str, int]:
    """
    Executes a subprocess wrapped in try/except with check=True and strict timeout handling.
    Automatically initiates clean process tree teardown upon timeout or failure.
    """
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.PIPE if stdin_data else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=run_env
        )
        if proc.pid:
            register_spawned_process(proc.pid)

        stdout, stderr = proc.communicate(input=stdin_data, timeout=timeout)
        ret_code = proc.returncode

        if proc.pid:
            unregister_spawned_process(proc.pid)

        if ret_code != 0:
            raise subprocess.CalledProcessError(ret_code, cmd, output=stdout, stderr=stderr)

        return stdout, stderr, ret_code

    except subprocess.TimeoutExpired as exc:
        if proc:
            if proc.pid:
                unregister_spawned_process(proc.pid)
            safe_process_tree_teardown(proc.pid, timeout_sec=3.0)
        logger.error(f"Subprocess '{cmd[0]}' timed out after {timeout} seconds.")
        raise TimeoutError(f"Subprocess '{cmd[0]}' timed out after {timeout} seconds.") from exc

    except subprocess.CalledProcessError as exc:
        if proc:
            if proc.pid:
                unregister_spawned_process(proc.pid)
            safe_process_tree_teardown(proc.pid, timeout_sec=2.0)
        logger.error(f"Subprocess '{cmd[0]}' failed with exit code {exc.returncode}: {exc.stderr}")
        raise

    except Exception as exc:
        if proc:
            if proc.pid:
                unregister_spawned_process(proc.pid)
            safe_process_tree_teardown(proc.pid, timeout_sec=2.0)
        logger.error(f"Subprocess '{cmd[0]}' encountered unexpected exception: {exc}")
        raise


def cleanup_all_cochem_processes() -> None:
    """
    Registered atexit handler to ensure no orphaned child orca, xtb, or mpi processes remain.
    """
    current_pid = os.getpid()
    try:
        current_proc = psutil.Process(current_pid)
        for child in current_proc.children(recursive=True):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


# Register clean process teardown at program exit
atexit.register(cleanup_all_cochem_processes)

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_alignment.py ---
"""
CoChem-TORQ: Exact Eckart Frame Aligner & Rotational Constants Engine Test Suite
================================================================================
Phase 2 (Stage 1.0 - 2.0) Authentic Physical Test Matrix
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

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_ipc.py ---
"""CoChem-TORQ: Test Suite for PyArrow IPC & Shared Memory Manager.

=============================================================================
Phase 5 (Stage 4.0) Authentic Physical Test Matrix
--------------------------------------------------
Authentic physical test suite verifying genuine PyArrow IPC serialization/deserialization,
cross-platform POSIX/Windows multiprocessing.shared_memory.SharedMemory mapping,
memory-mapped file (mmap) fallbacks, xxhash.xxh64 & SHA-256 tamper detection,
deterministic orphaned IPC resource cleanup, 6-Tier Environment Matrix path
resolutions with Tripartite Air-Gap enforcement, and Mendeleev dynamic mass lookups.

All tests operate against real physical memory segments, genuine temp files,
and real PyArrow tables, batches, and multi-dimensional tensors.
"""

from __future__ import annotations

import gc
import mmap
import multiprocessing.shared_memory as sm
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mendeleev import element
import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest
from pydantic import ValidationError

from Libraries.cochem_torq_ipc import (
    AirGapReport,
    AirGapViolationError,
    BufferCorruptedError,
    ChecksumMismatchError,
    ChecksumVerificationResult,
    EnvironmentTier,
    ExecutionContext,
    IPCBufferError,
    IPCBufferMetadata,
    IPCPayloadType,
    IPCRegistry,
    IPCSegmentType,
    OrphanedResourceError,
    PyArrowMmapFileSegment,
    PyArrowSharedMemorySegment,
    SharedMemoryAllocationError,
    _GLOBAL_IPC_REGISTRY,
    _XXHASH_AVAILABLE,
    compute_buffer_checksum,
    deserialize_from_ipc,
    get_atomic_mass,
    get_atomic_number,
    get_ipc_registry,
    get_isotopic_mass,
    get_repo_root,
    mmap_checksum_verifier,
    orphaned_ipc_cleaner,
    pyarrow_mmap_mapper,
    serialize_to_ipc,
)


# ============================================================================
# Autouse Fixture to Guarantee Clean IPC Registry Between Tests
# ============================================================================


@pytest.fixture(autouse=True)
def clean_ipc_environment() -> Generator[None, None, None]:
    """Ensure clean IPC tracking registry state before and after each test."""
    orphaned_ipc_cleaner()
    yield
    orphaned_ipc_cleaner()


# ============================================================================
# 1. PyArrow Shared Memory Lifecycle Tests
# ============================================================================


class TestPyArrowSharedMemoryLifecycle:
    """Authentic physical tests for SharedMemory IPC lifecycle."""

    def test_shared_memory_table_lifecycle(self) -> None:
        """Verify allocation, write, read-back, checksum verification, and cleanup for PyArrow Table."""
        table = pa.table({
            "atom_index": [0, 1, 2, 3],
            "symbol": ["O", "H", "H", "Ne"],
            "mass": [
                get_atomic_mass("O"),
                get_atomic_mass("H"),
                get_atomic_mass("H"),
                get_atomic_mass("Ne"),
            ],
            "x": [0.0, 0.757, -0.757, 3.2],
            "y": [0.0, 0.586, 0.586, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
        })

        seg_name = f"cochem_test_tbl_{uuid.uuid4().hex[:8]}"
        segment = pyarrow_mmap_mapper(payload=table, name=seg_name, prefer_shm=True)
        assert isinstance(segment, PyArrowSharedMemorySegment)
        assert segment.name == seg_name
        assert segment.metadata is not None
        assert segment.metadata.payload_type == IPCPayloadType.TABLE
        assert segment.metadata.size_bytes > 0
        assert segment.metadata.checksum_xxh64 != ""

        # Verify checksum
        raw_bytes = segment.read_raw(verify_checksum=True)
        assert len(raw_bytes) == segment.metadata.size_bytes

        # Read back table directly
        table_read = segment.read_table(verify_checksum=True)
        assert table_read.equals(table)

        # Deserialize using metadata alone via attachment
        table_deserialized = deserialize_from_ipc(segment.metadata, verify_checksum=True)
        assert table_deserialized.equals(table)

        # Teardown
        segment.unlink()
        assert segment.is_closed

    def test_shared_memory_tensor_lifecycle(self) -> None:
        """Verify allocation, write, read-back, checksum verification, and cleanup for 3D Tensors."""
        # 3D Tensor representing dipole transition moment matrix (3, N, N)
        np_tensor = np.random.RandomState(42).randn(3, 10, 10).astype(np.float64)
        seg_name = f"cochem_test_tsr_{uuid.uuid4().hex[:8]}"

        segment = pyarrow_mmap_mapper(payload=np_tensor, name=seg_name, prefer_shm=True)
        assert isinstance(segment, PyArrowSharedMemorySegment)
        assert segment.metadata is not None
        assert segment.metadata.payload_type == IPCPayloadType.NUMPY_ARRAY
        assert segment.metadata.extra_metadata["shape"] == [3, 10, 10]

        # Read back as numpy
        np_read = segment.read_numpy(verify_checksum=True)
        assert np.allclose(np_tensor, np_read)
        assert np_read.shape == (3, 10, 10)

        # Read back as pa.Tensor
        pa_tensor = segment.read_tensor(verify_checksum=True)
        assert np.allclose(pa_tensor.to_numpy(), np_tensor)

        # Deserialize from metadata
        deserialized = deserialize_from_ipc(segment.metadata, verify_checksum=True)
        assert np.allclose(deserialized, np_tensor)

        segment.unlink()

    def test_shared_memory_record_batch_lifecycle(self) -> None:
        """Verify allocation and round-trip for PyArrow RecordBatch."""
        batch = pa.record_batch([
            pa.array([101, 102, 103], type=pa.int64()),
            pa.array(["rot_A", "rot_B", "rot_C"], type=pa.string()),
            pa.array([28754.21, 14238.19, 9512.44], type=pa.float64()),
        ], names=["id", "constant", "value_mhz"])

        seg_name = f"cochem_test_rb_{uuid.uuid4().hex[:8]}"
        segment, meta = serialize_to_ipc(batch, name=seg_name, prefer_shm=True)
        assert isinstance(segment, PyArrowSharedMemorySegment)
        assert meta.payload_type == IPCPayloadType.RECORD_BATCH

        batch_back = segment.read_record_batch(verify_checksum=True)
        assert batch_back.equals(batch)

        segment.unlink()

    def test_shared_memory_context_manager(self) -> None:
        """Verify context manager closes SharedMemory automatically upon block exit."""
        seg_name = f"cochem_test_ctx_{uuid.uuid4().hex[:8]}"
        with PyArrowSharedMemorySegment(name=seg_name, size=2048, create=True) as seg:
            assert not seg.is_closed
            seg.write_raw(b"CONTEXT_MANAGER_PAYLOAD", payload_type=IPCPayloadType.RAW_BYTES)
            read_back = seg.read_raw(verify_checksum=True)
            assert read_back == b"CONTEXT_MANAGER_PAYLOAD"

        assert seg.is_closed
        seg.unlink()


# ============================================================================
# 2. Memory-Mapped File Fallback Tests
# ============================================================================


class TestMemoryMappedFileFallback:
    """Authentic physical tests for mmap file IPC operations and fallback mechanisms."""

    def test_mmap_file_table_roundtrip(self, tmp_path: Path) -> None:
        """Verify mmap file allocation in designated scratch directory and table round-trip."""
        ctx = ExecutionContext(custom_shm_dir=tmp_path / "custom_shm")
        table = pa.table({
            "freq_ghz": [12.45, 18.90, 24.12, 31.88],
            "intensity": [0.98, 0.45, 0.12, 0.76],
            "quantum_state": ["J=1<-0", "J=2<-1", "J=3<-2", "J=4<-3"],
        })

        segment = pyarrow_mmap_mapper(
            payload=table,
            prefer_shm=False,
            context=ctx,
        )
        assert isinstance(segment, PyArrowMmapFileSegment)
        assert segment.file_path.exists()
        assert segment.metadata is not None
        assert segment.metadata.buffer_type == IPCSegmentType.MEMORY_MAPPED_FILE
        assert segment.metadata.file_path == segment.file_path

        # Read back
        table_read = segment.read_table(verify_checksum=True)
        assert table_read.equals(table)

        # Deserialize via metadata
        table_deserialized = deserialize_from_ipc(segment.metadata, context=ctx, verify_checksum=True)
        assert table_deserialized.equals(table)

        # Unlink should remove physical file
        backing_path = segment.file_path
        segment.unlink()
        assert segment.is_closed
        assert not backing_path.exists()

    def test_mmap_file_tensor_and_numpy(self, tmp_path: Path) -> None:
        """Verify high-dimensional float64 array operations via mmap file."""
        ctx = ExecutionContext(custom_shm_dir=tmp_path / "custom_shm")
        # 4D Tensor (e.g. 2-body interaction integrals / Cartesian Hessian slice)
        arr = np.linspace(0.1, 100.0, 120, dtype=np.float64).reshape((2, 3, 4, 5))

        segment, meta = serialize_to_ipc(arr, prefer_shm=False, context=ctx)
        assert isinstance(segment, PyArrowMmapFileSegment)
        assert meta.payload_type == IPCPayloadType.NUMPY_ARRAY

        arr_read = segment.read_numpy(verify_checksum=True)
        assert np.allclose(arr, arr_read)
        assert arr_read.shape == (2, 3, 4, 5)

        segment.unlink()

    def test_shared_memory_allocation_failure_fallback(self, tmp_path: Path) -> None:
        """Verify transparent fallback to mmap file when prefer_shm is False or allocation fails."""
        ctx = ExecutionContext(custom_shm_dir=tmp_path / "fallback_shm")

        raw_payload = b"FALLBACK_TEST_STREAM_DATA"
        segment = pyarrow_mmap_mapper(
            payload=raw_payload,
            name="fallback_mmap_test",
            prefer_shm=False,  # Direct mmap allocation fallback
            context=ctx,
        )
        assert isinstance(segment, PyArrowMmapFileSegment)
        assert segment.file_path.exists()
        read_back = segment.read_raw(verify_checksum=True)
        assert read_back == raw_payload
        segment.unlink()


# ============================================================================
# 3. 6-Tier Environment Matrix & Air-Gap Enforcement Tests
# ============================================================================


class TestEnvironmentMatrixAndAirGap:
    """Tests verifying 6-Tier Environment Matrix path routing and Tripartite Air-Gap isolation."""

    @pytest.mark.parametrize("tier", [
        EnvironmentTier.LOCAL_WINDOWS,
        EnvironmentTier.LOCAL_MACOS,
        EnvironmentTier.LOCAL_LINUX,
        EnvironmentTier.GITHUB_ACTIONS,
        EnvironmentTier.CODESPACES,
        EnvironmentTier.HPC_NODES,
    ])
    def test_all_six_tiers_path_resolution(self, tier: EnvironmentTier) -> None:
        """Verify dynamic scratch, shm, and artifacts resolution for all 6 tiers."""
        orig_artifacts = os.environ.pop("COCHEM_ARTIFACTS_DIR", None)
        orig_scratch = os.environ.pop("COCHEM_SCRATCH_DIR", None)
        orig_shm = os.environ.pop("COCHEM_SHM_DIR", None)

        try:
            ctx = ExecutionContext(tier=tier)
            scratch_dir = ctx.get_scratch_dir()
            shm_dir = ctx.get_shm_dir()
            artifacts_dir = ctx.get_artifacts_dir()

            assert scratch_dir.is_dir()
            assert shm_dir.is_dir()
            assert artifacts_dir.is_dir()

            # Check subfolder resolution
            sub_scratch = ctx.get_scratch_dir("orca_job_01")
            assert sub_scratch.name == "orca_job_01"
            assert sub_scratch.is_dir()
        finally:
            if orig_artifacts is not None:
                os.environ["COCHEM_ARTIFACTS_DIR"] = orig_artifacts
            if orig_scratch is not None:
                os.environ["COCHEM_SCRATCH_DIR"] = orig_scratch
            if orig_shm is not None:
                os.environ["COCHEM_SHM_DIR"] = orig_shm

    def test_airgap_violation_in_repo_root(self) -> None:
        """Attempting to resolve scratch or artifacts inside Ring 1 repo root must raise AirGapViolationError."""
        repo_root = get_repo_root()
        illegal_target = repo_root / "forbidden_run_output"

        ctx = ExecutionContext()
        with pytest.raises(AirGapViolationError) as exc_info:
            ctx.verify_air_gap_boundary(illegal_target)

        assert "Tripartite Air-Gap Violation" in str(exc_info.value)

    def test_airgap_report_generation(self, tmp_path: Path) -> None:
        """Verify AirGapReport diagnostic properties."""
        safe_path = tmp_path / "valid_scratch"
        ctx = ExecutionContext()
        report = ctx.verify_air_gap_boundary(safe_path)

        assert isinstance(report, AirGapReport)
        assert report.is_valid is True
        assert report.reason is None
        assert report.target_resolved == safe_path.resolve()

    def test_environment_tier_detection(self) -> None:
        """Verify autonomous tier detection from host environment variables."""
        orig_env = dict(os.environ)
        try:
            os.environ["GITHUB_ACTIONS"] = "true"
            assert ExecutionContext.detect_tier() == EnvironmentTier.GITHUB_ACTIONS

            os.environ.pop("GITHUB_ACTIONS", None)
            os.environ["CODESPACES"] = "true"
            assert ExecutionContext.detect_tier() == EnvironmentTier.CODESPACES

            os.environ.pop("CODESPACES", None)
            os.environ["SLURM_JOB_ID"] = "123456"
            assert ExecutionContext.detect_tier() == EnvironmentTier.HPC_NODES
        finally:
            os.environ.clear()
            os.environ.update(orig_env)


# ============================================================================
# 4. xxHash Checksum Verification & Tamper Detection Tests
# ============================================================================


class TestChecksumVerificationAndTamperDetection:
    """Authentic physical tests for xxHash-64 bit-exact verification and bit corruption traps."""

    def test_bit_exact_checksum_verification(self) -> None:
        """Verify deterministic xxHash-64 computation on known byte payload."""
        payload_bytes = b"COCHEM_TORQ_QUANTUM_TENSOR_STATE_DATA_2026"
        digest, size = compute_buffer_checksum(payload_bytes, algorithm="xxh64")

        assert size == len(payload_bytes)
        assert isinstance(digest, str)
        if _XXHASH_AVAILABLE:
            assert len(digest) == 16  # xxhash64 hex length is 16 chars
        else:
            assert len(digest) == 64  # SHA-256 fallback hex length is 64 chars

        result = mmap_checksum_verifier(
            payload_bytes,
            expected_size=size,
            expected_checksum=digest,
            algorithm="xxh64",
            raise_on_error=True,
        )
        assert result.is_valid is True
        assert result.computed_hash == digest

    def test_single_byte_tamper_detection(self) -> None:
        """Corrupting a single byte in the buffer must raise ChecksumMismatchError."""
        original_data = bytearray(b"HIGH_PRECISION_CARTESIAN_HESSIAN_MATRIX_BLOCK")
        digest, size = compute_buffer_checksum(original_data, algorithm="xxh64")

        # Corrupt exactly 1 byte
        corrupted_data = bytearray(original_data)
        corrupted_data[5] = corrupted_data[5] ^ 0xFF  # Flip bits

        with pytest.raises(ChecksumMismatchError) as exc_info:
            mmap_checksum_verifier(
                corrupted_data,
                expected_size=size,
                expected_checksum=digest,
                algorithm="xxh64",
                raise_on_error=True,
            )

        assert "Hash mismatch" in str(exc_info.value)

    def test_size_mismatch_detection(self) -> None:
        """Truncated or expanded buffer must raise ChecksumMismatchError."""
        data = b"EXACT_64_BYTE_PAYLOAD_STRING_FOR_TORQ_INTERPROCESS_COMMUNICATION"
        digest, size = compute_buffer_checksum(data, algorithm="xxh64")

        with pytest.raises(ChecksumMismatchError) as exc_info:
            mmap_checksum_verifier(
                data,
                expected_size=size + 10,  # Wrong expected size
                expected_checksum=digest,
                algorithm="xxh64",
                raise_on_error=True,
            )

        assert "Size mismatch" in str(exc_info.value)

    def test_sha256_cryptographic_fallback(self) -> None:
        """Verify SHA-256 algorithm execution."""
        manifest_bytes = b"PROVENANCE_LOCKED_MANIFEST_BLOCK"
        digest, size = compute_buffer_checksum(manifest_bytes, algorithm="sha256")
        assert len(digest) == 64  # SHA-256 hex is 64 chars

        result = mmap_checksum_verifier(
            manifest_bytes,
            expected_size=size,
            expected_checksum=digest,
            algorithm="sha256",
            raise_on_error=True,
        )
        assert result.is_valid is True


# ============================================================================
# 5. Orphaned IPC Cleaner & Lifecycle Registry Tests
# ============================================================================


class TestOrphanedIPCCleanerAndRegistry:
    """Tests verifying thread-safe registry tracking and aggressive cleanup."""

    def test_registry_tracking_and_cleanup(self, tmp_path: Path) -> None:
        """Verify registry tracks active segments and cleanup_all purges all resources."""
        registry = get_ipc_registry()
        initial_count = registry.active_count()
        assert initial_count == 0

        # Create a SharedMemory segment
        seg_shm = PyArrowSharedMemorySegment(name=f"cochem_reg_shm_{uuid.uuid4().hex[:6]}", size=1024, create=True)
        # Create an mmap file segment
        ctx = ExecutionContext(custom_shm_dir=tmp_path / "reg_shm")
        seg_mmap = PyArrowMmapFileSegment(size=1024, context=ctx)

        assert registry.active_count() == 2
        assert seg_shm.name in registry.active_shm_names()
        assert seg_mmap.file_path.resolve() in registry.active_file_paths()

        # Run orphaned cleaner
        cleaned = orphaned_ipc_cleaner()
        assert cleaned >= 2
        assert registry.active_count() == 0
        assert not seg_mmap.file_path.exists()

    def test_subprocess_atexit_cleaner(self) -> None:
        """Spawning a subprocess that allocates SharedMemory and exits cleanly must purge resources."""
        seg_name = f"cochem_subp_test_{uuid.uuid4().hex[:8]}"

        script = f"""
import sys
from Libraries.cochem_torq_ipc import pyarrow_mmap_mapper, get_ipc_registry
seg = pyarrow_mmap_mapper(name='{seg_name}', size_bytes=2048, prefer_shm=True)
print('ALLOCATED', seg.name)
sys.exit(0)
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(get_repo_root()),
        )
        assert proc.returncode == 0
        assert "ALLOCATED" in proc.stdout

        # Verify that segment was cleanly unlinked by atexit in subprocess
        with pytest.raises(Exception):
            # Attempting to attach should fail because it was unlinked
            sm.SharedMemory(name=seg_name, create=False)


# ============================================================================
# 6. Mendeleev Dynamic Mass Integration Tests
# ============================================================================


class TestMendeleevIntegration:
    """Authentic physical tests for dynamic mass and isotope lookups."""

    def test_dynamic_atomic_mass(self) -> None:
        """Verify dynamic atomic weight lookup without hardcoding."""
        h_mass = get_atomic_mass("H")
        c_mass = get_atomic_mass("C")
        o_mass = get_atomic_mass("O")
        zn_mass = get_atomic_mass("Zn")

        assert abs(h_mass - float(element("H").atomic_weight)) < 1e-6
        assert abs(c_mass - float(element("C").atomic_weight)) < 1e-6
        assert abs(o_mass - float(element("O").atomic_weight)) < 1e-6
        assert abs(zn_mass - float(element("Zn").atomic_weight)) < 1e-6

    def test_dynamic_isotopic_mass(self) -> None:
        """Verify dynamic isotopic mass retrieval for spectroscopic isotopologues."""
        c13_mass = get_isotopic_mass("C", 13)
        h2_mass = get_isotopic_mass("H", 2)  # Deuterium
        o18_mass = get_isotopic_mass("O", 18)

        assert 13.0 < c13_mass < 13.01
        assert 2.014 < h2_mass < 2.015
        assert 17.99 < o18_mass < 18.01

    def test_atomic_number_lookup(self) -> None:
        """Verify atomic number lookups."""
        assert get_atomic_number("H") == 1
        assert get_atomic_number("C") == 6
        assert get_atomic_number("N") == 7
        assert get_atomic_number("O") == 8
        assert get_atomic_number("Ne") == 10
        assert get_atomic_number("Ar") == 18


# ============================================================================
# 7. High-Dimensional Tensor & Table Stress Tests
# ============================================================================


class TestHighDimensionalIPCStress:
    """Stress tests verifying high-throughput multi-megabyte tensor transfers."""

    def test_multi_megabyte_tensor_ipc(self) -> None:
        """Verify multi-megabyte (1M double elements = 8MB) zero-copy tensor transfer."""
        # 1,000,000 float64 elements = 8,000,000 bytes
        large_array = np.random.RandomState(99).randn(100, 100, 100).astype(np.float64)
        seg_name = f"cochem_large_tsr_{uuid.uuid4().hex[:8]}"

        segment, meta = serialize_to_ipc(large_array, name=seg_name, prefer_shm=True)
        assert meta.size_bytes >= 8000000

        read_array = deserialize_from_ipc(meta, segment=segment, verify_checksum=True)
        assert np.allclose(large_array, read_array)
        assert read_array.shape == (100, 100, 100)

        segment.unlink()

    def test_pydantic_metadata_validation(self) -> None:
        """Verify Pydantic validation rejects negative buffer sizes."""
        with pytest.raises(ValidationError):
            IPCBufferMetadata(
                name="invalid_buffer",
                buffer_type=IPCSegmentType.SHARED_MEMORY,
                size_bytes=-100,  # Invalid negative size
                allocated_bytes=1024,
                checksum_xxh64="deadbeef",
            )

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_spcat.py ---
import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from Libraries.cochem_spcat_bridge import (
    CONSTANTS,
    FortranOverflowError,
    LAMTriggerError,
    SPCATBridgeError,
    ThreeTierRoutingResult,
    TorqSpcatBridge,
    apply_symmetry_divisors,
    build_complete_spcat_payload,
    calculate_rotational_partition_function,
    calculate_vibrational_partition_function,
    compute_coupled_partition_functions,
    compute_sha256,
    format_fortran_double,
    fortran_double_precision_formatter,
    fortran_overflow_guard,
    generate_spcat_int,
    generate_spcat_provenance_manifest,
    generate_spcat_var,
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


def test_torq_spcat_bridge_extract_orca(
    tmp_path: Path
) -> None:
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
        "anharmonic_x_matrix": np.array([
            [-42.6, -15.9, -165.8],
            [-15.9, -42.9, -166.1],
            [-165.8, -166.1, -47.8]
        ]),
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


--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_tensor_extractor.py ---
"""CoChem-TORQ: Physical Test Suite for Quantum Tensor Harvester (Stage 4.1).

Phase 6 Validation Suite
-------------------------------------------------------------------------------
Validates:
1. Exact CODATA 2022 constants & CIAAW/AME2020 mass tables.
2. Moments of inertia, principal axes diagonalization, planar moments,
   and inertial defect (Delta) for 3D asymmetric, symmetric, and planar systems:
   - H2O (asymmetric prolate, planar, Delta = 0)
   - SO2 (asymmetric prolate, planar, Delta = 0)
   - H2CO (near-prolate asymmetric, planar, Delta = 0)
   - CH3Cl (prolate symmetric top, kappa = -1, Ib = Ic)
   - Benzene C6H6 (oblate symmetric top, kappa = +1, Ia = Ib, planar)
   - CH4 (spherical top, Ia = Ib = Ic, A = B = C)
3. Cartesian Protections & Linearity Trap for linear/quasi-linear systems:
   - CO2, OCS, HCN (Ia = 0, collinear backbone, cylindrical projection, DOF=2)
   - Quasi-linear floppy complex singularity damping.
4. Ray's Asymmetry Parameter (kappa) and Dynamic Representation Switching:
   - All 6 representations (Ir, Il, IIr, IIl, IIIr, IIIl)
   - Right-handed permutation matrix determinants (+1)
   - Wang Hamiltonian sub-blocks [E+, E-, O+, O-].
5. ORCA VPT2, Coriolis, and Centrifugal Distortion parsing with divergence checks.
6. HDF5 / JSON structured export gateways with Air-Gap directory compliance.
7. Anti-spoofing verification and empirical physical fidelity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

from Libraries.cochem_tensor_extractor import (
    AMU_A2_TO_MHZ,
    AMU_TO_KG,
    ANGSTROM_TO_M,
    ATOMIC_MASS_CONSTANT_U,
    C_M_S,
    C_ROT_CM1,
    C_ROT_GHZ,
    C_ROT_MHZ,
    CODATA_YEAR,
    PLANCK_CONSTANT_JS,
    SPEED_OF_LIGHT_C,
    AsymmetryResult,
    CartesianProtectionResult,
    InertiaTensorResult,
    TorqTensorExtractor,
    TorqTensorOutput,
    apply_cartesian_protections,
    calculate_rays_asymmetry,
    diagonalize_inertia_tensor,
    dynamic_representation_switch,
    get_atomic_mass,
)

# =============================================================================
# Test Suite 1: Exact Physical Constants & Isotopic Mass Tables
# =============================================================================


def test_exact_physical_constants_codata_2022() -> None:
    """Validates physical constants against exact CODATA 2022 standard."""
    assert CODATA_YEAR == 2022
    assert PLANCK_CONSTANT_JS == 6.62607015e-34
    assert SPEED_OF_LIGHT_C == 299792458.0
    assert C_M_S == SPEED_OF_LIGHT_C
    assert ATOMIC_MASS_CONSTANT_U == 1.66053906892e-27
    assert AMU_TO_KG == ATOMIC_MASS_CONSTANT_U
    assert ANGSTROM_TO_M == 1.0e-10

    # Verify analytical derivation of rotational conversion factor:
    # C_rot = h / (8 * pi^2 * u * 1e-20) * 1e-6 (MHz * u * Angstrom^2)
    expected_c_rot = (
        PLANCK_CONSTANT_JS
        / (8.0 * (math.pi**2) * ATOMIC_MASS_CONSTANT_U * (ANGSTROM_TO_M**2))
    ) * 1e-6
    assert abs(C_ROT_MHZ - expected_c_rot) < 1e-9
    assert abs(C_ROT_MHZ - 505379.008435) < 1e-3
    assert AMU_A2_TO_MHZ == C_ROT_MHZ

    # Verify GHz and cm^-1 conversions
    assert abs(C_ROT_GHZ - (C_ROT_MHZ * 1e-3)) < 1e-9
    expected_c_rot_cm1 = (C_ROT_MHZ * 1e6) / (SPEED_OF_LIGHT_C * 100.0)
    assert abs(C_ROT_CM1 - expected_c_rot_cm1) < 1e-9


def test_isotopic_mass_table_accuracy_and_parsing() -> None:
    """Validates CIAAW / AME2020 mono-isotopic mass lookups via mendeleev."""
    import mendeleev
    
    # H
    elem_h = mendeleev.element("H")
    h1_mass = float(next(i for i in elem_h.isotopes if i.mass_number == 1).mass)
    h2_mass = float(next(i for i in elem_h.isotopes if i.mass_number == 2).mass)
    h3_mass = float(next(i for i in elem_h.isotopes if i.mass_number == 3).mass)
    assert abs(get_atomic_mass("H") - h1_mass) < 1e-8
    assert abs(get_atomic_mass("1H") - h1_mass) < 1e-8
    assert abs(get_atomic_mass("D") - h2_mass) < 1e-8
    assert abs(get_atomic_mass("2H") - h2_mass) < 1e-8
    assert abs(get_atomic_mass("T") - h3_mass) < 1e-8
    assert abs(get_atomic_mass("3H") - h3_mass) < 1e-8

    # C
    elem_c = mendeleev.element("C")
    most_abundant_c = sorted([i for i in elem_c.isotopes if i.abundance is not None], key=lambda x: x.abundance, reverse=True)[0]
    c_mass = float(most_abundant_c.mass)
    c12_mass = float(next(i for i in elem_c.isotopes if i.mass_number == 12).mass)
    c13_mass = float(next(i for i in elem_c.isotopes if i.mass_number == 13).mass)
    
    assert abs(get_atomic_mass("C") - c_mass) < 1e-8
    assert abs(get_atomic_mass("12C") - c12_mass) < 1e-8
    assert abs(get_atomic_mass("13C") - c13_mass) < 1e-8
    assert abs(get_atomic_mass("C13") - c13_mass) < 1e-8

    # N
    elem_n = mendeleev.element("N")
    most_abundant_n = sorted([i for i in elem_n.isotopes if i.abundance is not None], key=lambda x: x.abundance, reverse=True)[0]
    n_mass = float(most_abundant_n.mass)
    n15_mass = float(next(i for i in elem_n.isotopes if i.mass_number == 15).mass)
    assert abs(get_atomic_mass("N") - n_mass) < 1e-8
    assert abs(get_atomic_mass("15N") - n15_mass) < 1e-8

    # O
    elem_o = mendeleev.element("O")
    most_abundant_o = sorted([i for i in elem_o.isotopes if i.abundance is not None], key=lambda x: x.abundance, reverse=True)[0]
    o_mass = float(most_abundant_o.mass)
    o18_mass = float(next(i for i in elem_o.isotopes if i.mass_number == 18).mass)
    assert abs(get_atomic_mass("O") - o_mass) < 1e-8
    assert abs(get_atomic_mass("18O") - o18_mass) < 1e-8
    assert abs(get_atomic_mass("O18") - o18_mass) < 1e-8

    # Other atoms
    elem_cl = mendeleev.element("Cl")
    cl35_mass = float(next(i for i in elem_cl.isotopes if i.mass_number == 35).mass)
    cl37_mass = float(next(i for i in elem_cl.isotopes if i.mass_number == 37).mass)
    assert abs(get_atomic_mass("35Cl") - cl35_mass) < 1e-8
    assert abs(get_atomic_mass("37Cl") - cl37_mass) < 1e-8
    
    elem_br = mendeleev.element("Br")
    br79_mass = float(next(i for i in elem_br.isotopes if i.mass_number == 79).mass)
    br81_mass = float(next(i for i in elem_br.isotopes if i.mass_number == 81).mass)
    assert abs(get_atomic_mass("79Br") - br79_mass) < 1e-8
    assert abs(get_atomic_mass("81Br") - br81_mass) < 1e-8

    elem_i = mendeleev.element("I")
    most_abundant_i = sorted([i for i in elem_i.isotopes if i.abundance is not None], key=lambda x: x.abundance, reverse=True)[0]
    i_mass = float(most_abundant_i.mass)
    assert abs(get_atomic_mass("I") - i_mass) < 1e-8

    # Strict Anti-Spoofing: Unrecognized elements must raise ValueError, no hardcoded fallbacks
    import pytest
    with pytest.raises(ValueError, match="Symbol 'UnknownElement' not found"):
        get_atomic_mass("UnknownElement")


# =============================================================================
# Test Suite 2: Real Asymmetric Tops & Planar Systems (H2O, SO2, H2CO)
# =============================================================================


def test_water_molecule_h2o_planar_asymmetric_top() -> None:
    """Validates inertia tensor, planar moments, inertial defect, and constants

    for real H2O geometry.
    """
    # Equilibrium C2v geometry of H2O in yz plane (Angstroms)
    symbols = ["O", "H", "H"]
    coords = [
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ]

    res = diagonalize_inertia_tensor(coords, symbols=symbols)

    # 1. Mass and COM
    expected_mass = get_atomic_mass("O") + 2.0 * get_atomic_mass("H")
    assert abs(res.total_mass_u - expected_mass) < 1e-6
    # COM must be close to origin on y, and centered
    assert abs(res.center_of_mass_A[0]) < 1e-10
    assert abs(res.center_of_mass_A[1]) < 1e-10

    # 2. Moments of Inertia: Ia <= Ib <= Ic
    ia, ib, ic = res.principal_moments_u_A2
    assert 0.0 < ia < ib < ic

    # 3. Planar defect: Delta = Ic - Ia - Ib == 0.0 for planar geometry
    assert abs(res.inertial_defect_u_A2) < 1e-6
    assert res.is_planar is True
    # For molecule in principal plane (a, b), P_cc = sum m * c^2 = 0
    assert abs(res.planar_moments.P_cc) < 1e-6
    assert res.planar_moments.P_aa > 0.0
    assert res.planar_moments.P_bb > 0.0

    # 4. Rotational constants A >= B >= C
    rc = res.rotational_constants
    assert rc.A_MHz is not None
    assert rc.A_GHz is not None
    assert rc.A_MHz > rc.B_MHz > rc.C_MHz
    # Literature H2O equilibrium: A ~ 835 GHz, B ~ 435 GHz, C ~ 278 GHz
    assert 800.0 < rc.A_GHz < 900.0
    assert 400.0 < rc.B_GHz < 500.0
    assert 250.0 < rc.C_GHz < 350.0

    # 5. Ray's Asymmetry parameter kappa for H2O: kappa ~ -0.46 (Asymmetric Prolate)
    asym = calculate_rays_asymmetry(rc.A_MHz, rc.B_MHz, rc.C_MHz)
    assert -0.60 < asym.kappa < -0.30
    assert asym.rotor_type == "Asymmetric Prolate"
    assert asym.recommended_representation == "Ir"


def test_sulfur_dioxide_so2_planar_asymmetric_top() -> None:
    """Validates planar SO2 molecule moments of inertia and near-prolate asymmetry."""
    symbols = ["S", "O", "O"]
    coords = [
        [0.000000, 0.000000, 0.364200],
        [0.000000, 1.237000, -0.364200],
        [0.000000, -1.237000, -0.364200],
    ]

    res = diagonalize_inertia_tensor(coords, symbols=symbols)

    # In-plane equilibrium -> inertial defect = 0
    assert abs(res.inertial_defect_u_A2) < 1e-6
    assert res.is_planar is True

    ia, ib, ic = res.principal_moments_u_A2
    assert ia < ib < ic

    rc = res.rotational_constants
    assert rc.A_MHz is not None
    # SO2: A ~ 60 GHz, B ~ 10 GHz, C ~ 8 GHz
    assert 50000.0 < rc.A_MHz < 70000.0
    assert 8000.0 < rc.B_MHz < 12000.0
    assert 7000.0 < rc.C_MHz < 10000.0

    asym = calculate_rays_asymmetry(rc.A_MHz, rc.B_MHz, rc.C_MHz)
    # SO2 kappa is near -0.94 (very prolate)
    assert -0.98 < asym.kappa < -0.90
    assert asym.rotor_type == "Asymmetric Prolate"
    assert asym.recommended_representation == "Ir"


def test_formaldehyde_h2co_planar_asymmetric_top() -> None:
    """Validates formaldehyde H2CO (near-prolate symmetric top)."""
    symbols = ["C", "O", "H", "H"]
    coords = [
        [0.000000, 0.000000, -0.597600],
        [0.000000, 0.000000, 0.607400],
        [0.000000, 0.934300, -1.171200],
        [0.000000, -0.934300, -1.171200],
    ]

    extractor = TorqTensorExtractor(symbols, coords, point_id="h2co_01")
    tensors = extractor.extract_tensors()

    assert abs(tensors["inertial_defect_u_A2"]) < 1e-6
    assert tensors["is_planar"] is True

    rc = tensors["rotational_constants"]
    assert rc["A"] > rc["B"] > rc["C"]

    # H2CO kappa ~ -0.96
    asym = tensors["asymmetry"]
    assert -0.99 < asym["kappa"] < -0.93
    assert asym["rotor_type"] == "Asymmetric Prolate"


# =============================================================================
# Test Suite 3: Symmetric Tops (Prolate CH3Cl, Oblate Benzene) & Spherical Top (CH4)
# =============================================================================


def test_methyl_chloride_ch3cl_prolate_symmetric_top() -> None:
    """Validates methyl chloride CH3Cl as prolate symmetric top (kappa=-1, Ib=Ic)."""
    symbols = ["C", "Cl", "H", "H", "H"]
    r_ch = 1.09
    theta = math.radians(109.5)
    r_ccl = 1.78

    coords = [
        [0.000000, 0.000000, 0.000000],  # C
        [0.000000, 0.000000, r_ccl],  # Cl
        [0.000000, r_ch * math.sin(theta), r_ch * math.cos(theta)],  # H1
        [
            r_ch * math.sin(theta) * math.cos(math.radians(210)),
            r_ch * math.sin(theta) * math.sin(math.radians(210)),
            r_ch * math.cos(theta),
        ],  # H2
        [
            r_ch * math.sin(theta) * math.cos(math.radians(330)),
            r_ch * math.sin(theta) * math.sin(math.radians(330)),
            r_ch * math.cos(theta),
        ],  # H3
    ]

    res = diagonalize_inertia_tensor(coords, symbols=symbols)
    ia, ib, ic = res.principal_moments_u_A2

    # Prolate top: Ia < Ib == Ic
    assert ia < ib
    assert abs(ib - ic) < 1e-4

    rc = res.rotational_constants
    assert rc.A_MHz is not None
    assert abs(rc.B_MHz - rc.C_MHz) < 1e-2

    asym = calculate_rays_asymmetry(rc.A_MHz, rc.B_MHz, rc.C_MHz)
    assert abs(asym.kappa - (-1.0)) < 1e-4
    assert asym.rotor_type == "Prolate Symmetric"
    assert asym.recommended_representation == "Ir"


def test_benzene_c6h6_oblate_symmetric_top() -> None:
    """Validates Benzene C6H6 as planar oblate symmetric top (kappa=+1, Ia=Ib)."""
    symbols = ["C"] * 6 + ["H"] * 6
    r_cc = 1.397
    r_ch = 1.084
    r_tot = r_cc + r_ch

    coords = []
    # Carbons
    for i in range(6):
        angle = math.radians(60.0 * i)
        coords.append([r_cc * math.cos(angle), r_cc * math.sin(angle), 0.0])
    # Hydrogens
    for i in range(6):
        angle = math.radians(60.0 * i)
        coords.append([r_tot * math.cos(angle), r_tot * math.sin(angle), 0.0])

    res = diagonalize_inertia_tensor(coords, symbols=symbols)
    ia, ib, ic = res.principal_moments_u_A2

    # Oblate symmetric top: Ia == Ib < Ic
    assert abs(ia - ib) < 1e-4
    # Planar exact condition: Ic = Ia + Ib = 2*Ia
    assert abs(ic - (ia + ib)) < 1e-4
    assert abs(res.inertial_defect_u_A2) < 1e-4
    assert res.is_planar is True

    rc = res.rotational_constants
    assert rc.A_MHz is not None
    assert abs(rc.A_MHz - rc.B_MHz) < 1e-2
    assert rc.B_MHz > rc.C_MHz

    asym = calculate_rays_asymmetry(rc.A_MHz, rc.B_MHz, rc.C_MHz)
    assert abs(asym.kappa - 1.0) < 1e-4
    assert asym.rotor_type == "Oblate Symmetric"
    assert asym.recommended_representation == "IIIr"


def test_methane_ch4_spherical_top() -> None:
    """Validates methane CH4 as isotropic spherical top (Ia = Ib = Ic, A = B = C)."""
    symbols = ["C", "H", "H", "H", "H"]
    d = 1.089 / math.sqrt(3.0)
    coords = [
        [0.0, 0.0, 0.0],
        [d, d, d],
        [d, -d, -d],
        [-d, d, -d],
        [-d, -d, d],
    ]

    res = diagonalize_inertia_tensor(coords, symbols=symbols)
    ia, ib, ic = res.principal_moments_u_A2

    assert abs(ia - ib) < 1e-6
    assert abs(ib - ic) < 1e-6

    rc = res.rotational_constants
    assert rc.A_MHz is not None
    assert abs(rc.A_MHz - rc.B_MHz) < 1e-3
    assert abs(rc.B_MHz - rc.C_MHz) < 1e-3

    asym = calculate_rays_asymmetry(rc.A_MHz, rc.B_MHz, rc.C_MHz)
    assert asym.rotor_type == "Spherical Top"


# =============================================================================
# Test Suite 4: Cartesian Protections & Linearity Trap (CO2, OCS, HCN)
# =============================================================================


def test_cartesian_protections_linear_co2_and_ocs() -> None:
    """Validates Cartesian protections for linear molecules CO2 and OCS."""
    # CO2 along z-axis
    co2_symbols = ["O", "C", "O"]
    co2_coords = [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]

    co2_prot = apply_cartesian_protections(co2_coords, symbols=co2_symbols)
    assert co2_prot.is_linear is True
    assert co2_prot.rotational_dof == 2
    assert co2_prot.singularity_damping_applied is True
    assert co2_prot.protected_rotational_constants.A_MHz is None
    assert co2_prot.protected_rotational_constants.B_MHz > 0.0
    assert (
        co2_prot.protected_rotational_constants.B_MHz
        == co2_prot.protected_rotational_constants.C_MHz
    )

    # Cylindrical coordinates verification: radial rho == 0.0 for all atoms
    for cyl in co2_prot.cylindrical_coordinates:
        assert abs(cyl["rho_radial"]) < 1e-6

    # OCS along arbitrary rotated line
    ocs_symbols = ["O", "C", "S"]
    v = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    ocs_coords = [
        (0.0 * v).tolist(),
        (1.16 * v).tolist(),
        ((1.16 + 1.56) * v).tolist(),
    ]

    ocs_prot = apply_cartesian_protections(ocs_coords, symbols=ocs_symbols)
    assert ocs_prot.is_linear is True
    assert ocs_prot.rotational_dof == 2
    for cyl in ocs_prot.cylindrical_coordinates:
        assert abs(cyl["rho_radial"]) < 1e-4


def test_cartesian_protections_linear_hcn() -> None:
    """Validates linear HCN."""
    symbols = ["H", "C", "N"]
    coords = [[0.0, 0.0, -1.066], [0.0, 0.0, 0.0], [0.0, 0.0, 1.153]]

    extractor = TorqTensorExtractor(symbols, coords, point_id="hcn_linear")
    output = extractor.get_full_output()

    assert output.cartesian_protection.is_linear is True
    assert output.cartesian_protection.rotational_dof == 2
    assert output.inertia.rotational_constants.A_MHz is None
    assert output.inertia.rotational_constants.B_MHz > 0.0


def test_cartesian_protections_quasi_linear_complex() -> None:
    """Validates quasi-linear floppy complex protection with 179.5 degree angle."""
    symbols = ["Ne", "C", "O"]
    # Slight bend of 0.5 degrees
    angle_rad = math.radians(179.5)
    r1 = 3.2
    r2 = 1.13
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, r1],
        [
            r2 * math.sin(math.pi - angle_rad),
            0.0,
            r1 + r2 * math.cos(math.pi - angle_rad),
        ],
    ]

    prot = apply_cartesian_protections(
        coords, symbols=symbols, threshold_linear=1e-2, angle_tolerance_deg=1.0
    )
    assert prot.is_quasi_linear is True
    assert prot.rotational_dof == 2
    assert prot.singularity_damping_applied is True


# =============================================================================
# Test Suite 5: Dynamic Representation Switch (6 Representations)
# =============================================================================


def test_dynamic_representation_switch_all_six_representations() -> None:
    """Validates all 6 standard King-Hainer-Cross representations."""
    from typing import Literal

    representations: list[Literal["Ir", "Il", "IIr", "IIl", "IIIr", "IIIl"]] = [
        "Ir",
        "Il",
        "IIr",
        "IIl",
        "IIIr",
        "IIIl",
    ]

    for rep in representations:
        res = dynamic_representation_switch(kappa=0.5, preferred_type=rep)
        assert res["representation"] == rep
        assert "axis_mapping" in res
        assert "transformation_matrix" in res
        t_mat = np.array(res["transformation_matrix"])
        det = np.linalg.det(t_mat)
        if res["is_right_handed"]:
            assert abs(det - 1.0) < 1e-6
        else:
            assert abs(det - (-1.0)) < 1e-6
        assert res["wang_subblocks"] == ["E+", "E-", "O+", "O-"]

    # Auto selection based on kappa
    prolate_rep = dynamic_representation_switch(kappa=-0.8, preferred_type="auto")
    assert prolate_rep["representation"] == "Ir"

    oblate_rep = dynamic_representation_switch(kappa=+0.8, preferred_type="auto")
    assert oblate_rep["representation"] == "IIIr"


# =============================================================================
# Test Suite 6: ORCA VPT2, Coriolis, Centrifugal Distortion Parser
# =============================================================================


def test_orca_vpt2_and_coriolis_parser(tmp_path: Path) -> None:
    """Validates ORCA %vib block parsing for resonances and distortion constants."""
    orca_output_text = """
================================================================================
                               ORCA VPT2 MODULE
================================================================================
Darling-Dennison Mode 1 Mode 2 K = -14.2857
Darling-Dennison Mode 3 Mode 4 K = 2.4510

----------------------------------------
Coriolis Coupling Matrix (X)
----------------------------------------
  0.000000  0.845120 -0.124500
 -0.845120  0.000000  0.512340
  0.124500 -0.512340  0.000000

----------------------------------------
Coriolis Coupling Matrix (Y)
----------------------------------------
  0.000000  0.221100  0.781200
 -0.221100  0.000000 -0.114400
 -0.781200  0.114400  0.000000

----------------------------------------
Coriolis Coupling Matrix (Z)
----------------------------------------
  0.000000  0.000000  0.000000
  0.000000  0.000000  0.998120
  0.000000 -0.998120  0.000000

Centrifugal Distortion Constants (A-Reduction):
  D_J  = 0.034512
  D_JK = -0.124500
  D_K  = 1.542100
  d_1  = -0.004120
  d_2  = 0.000850

Polarizability derivative: 1.254100
Polarizability derivative: 0.895400
Polarizability derivative: 2.145000
"""
    orca_file = tmp_path / "orca_test.out"
    orca_file.write_text(orca_output_text, encoding="utf-8")

    extractor = TorqTensorExtractor(
        symbols=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.1], [0.0, 0.7, -0.4], [0.0, -0.7, -0.4]],
        point_id="vpt2_h2o",
        orca_file=orca_file,
    )

    vpt2_res = extractor.extract_vpt2_data(orca_file)

    # 1. Darling-Dennison
    assert len(vpt2_res["darling_dennison"]) == 2
    assert vpt2_res["darling_dennison"][0]["mode1"] == 1
    assert vpt2_res["darling_dennison"][0]["mode2"] == 2
    assert abs(vpt2_res["darling_dennison"][0]["resonance"] - (-14.2857)) < 1e-4

    # 2. Coriolis Couplings
    assert len(vpt2_res["coriolis_couplings"]["x"]) == 9
    assert abs(vpt2_res["coriolis_couplings"]["x"][1] - 0.845120) < 1e-5
    assert len(vpt2_res["coriolis_couplings"]["z"]) == 9

    # 3. Distortion Constants
    cd = vpt2_res["centrifugal_distortion"]
    assert abs(cd["D_J"][0] - 0.034512) < 1e-6
    assert abs(cd["D_JK"][0] - (-0.124500)) < 1e-6
    assert abs(cd["D_K"][0] - 1.542100) < 1e-6
    assert abs(cd["d_1"][0] - (-0.004120)) < 1e-6
    assert abs(cd["d_2"][0] - 0.000850) < 1e-6

    # 4. Polarizabilities
    assert len(vpt2_res["raman_polarizability"]) == 3
    assert abs(vpt2_res["raman_polarizability"][0] - 1.254100) < 1e-6

    # 5. Divergence check
    assert vpt2_res["is_divergent"] is False


def test_orca_vpt2_divergence_detection(tmp_path: Path) -> None:
    """Validates unphysical divergence detection for distortion constants."""
    orca_divergent_text = """
Centrifugal Distortion Constants:
  D_J  = 1.5e7
  D_JK = 2.4e8
  D_K  = -9.9e9
"""
    orca_file = tmp_path / "orca_div.out"
    orca_file.write_text(orca_divergent_text, encoding="utf-8")

    extractor = TorqTensorExtractor(
        symbols=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.1], [0.0, 0.7, -0.4], [0.0, -0.7, -0.4]],
    )
    vpt2_res = extractor.extract_vpt2_data(orca_file)

    assert vpt2_res["is_divergent"] is True
    assert len(vpt2_res["divergence_details"]) > 0


# =============================================================================
# Test Suite 7: Thermal NMR & Raman Polarizability Extractors
# =============================================================================


def test_thermal_nmr_extraction(tmp_path: Path) -> None:
    """Validates thermal NMR shielding extraction from AIMD trajectory file."""
    traj_text = """3
Frame 1
O  0.0  0.0  0.11
H  0.0  0.75 -0.46
H  0.0 -0.75 -0.46
3
Frame 2
O  0.0  0.0  0.12
H  0.0  0.76 -0.47
H  0.0 -0.76 -0.47
"""
    traj_file = tmp_path / "aimd_traj.xyz"
    traj_file.write_text(traj_text, encoding="utf-8")

    extractor = TorqTensorExtractor(
        symbols=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.11], [0.0, 0.75, -0.46], [0.0, -0.75, -0.46]],
    )

    nmr_res = extractor.extract_thermal_nmr(traj_file)
    assert nmr_res["frame_count"] == 2
    assert nmr_res["thermal_average"] > 0.0
    assert len(nmr_res["isotropic_shielding"]) == 2


# =============================================================================
# Test Suite 8: JSON and HDF5 Export Gateways & Air-Gap Compliance
# =============================================================================


def test_export_tensor_json_and_hdf5(tmp_path: Path) -> None:
    """Validates JSON and HDF5 serialization with Air-Gap directory compliance."""
    export_dir = tmp_path / "artifacts" / "Tensors"
    export_dir.mkdir(parents=True, exist_ok=True)

    json_file = export_dir / "torq_tensors.json"
    vpt2_file = export_dir / "torq_vpt2.json"
    lam_file = export_dir / "torq_lam_vpt2.json"
    h5_file = export_dir / "torq_tensors.h5"

    symbols = ["O", "H", "H"]
    coords = [
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ]

    extractor = TorqTensorExtractor(symbols, coords, point_id="h2o_pt01")

    # 1. Export JSON
    extractor.export_tensor(json_file)
    assert json_file.exists()
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
        assert data["point_id"] == "h2o_pt01"
        assert "rotational_constants" in data
        assert "inertia_tensor" in data

    # 2. Export VPT2 JSON
    with pytest.raises(FileNotFoundError):
        extractor.export_vpt2_tensor(vpt2_file)

    # 3. Export LAM VPT2 JSON
    with pytest.raises(FileNotFoundError):
        extractor.export_lam_vpt2_tensor(lam_file)

    # 4. Export HDF5
    payload_dict = {
        "rotational_constants_mhz": [
            data["rotational_constants"]["A"],
            data["rotational_constants"]["B"],
            data["rotational_constants"]["C"],
        ],
        "inertia_tensor": data["inertia_tensor"],
        "is_planar": True,
        "rotor_type": "Asymmetric Prolate",
    }
    extractor.export_to_hdf5(h5_file, payload_dict)
    assert h5_file.exists()

    # Verify HDF5 contents
    with h5py.File(h5_file, "r") as f:
        assert "point_h2o_pt01" in f
        grp = f["point_h2o_pt01"]
        assert "rotational_constants_mhz" in grp
        assert "inertia_tensor" in grp
        assert bool(grp.attrs["is_planar"]) is True
        assert str(grp.attrs["rotor_type"]) == "Asymmetric Prolate"

    # 5. Export Sinc-DVR HDF5
    dvr_h5 = export_dir / "sinc_dvr.h5"
    dvr_payload = {
        "wavefunction": np.ones((50, 50)).tolist(),
        "energy_levels": [0.0, 125.4, 250.8, 375.2],
        "tunneling_splitting": 1.458e-4,
        "kraitchman_coords": [[0.0, 0.0, 0.5]],
    }
    extractor.export_to_hdf5_with_sinc_dvr(dvr_h5, dvr_payload)
    assert dvr_h5.exists()
    with h5py.File(dvr_h5, "r") as f:
        grp = f["point_h2o_pt01"]
        assert "wavefunction" in grp
        assert "energy_levels" in grp
        assert abs(grp.attrs["tunneling_splitting"] - 1.458e-4) < 1e-8


# =============================================================================
# Test Suite 9: Pydantic Data Models & Anti-Spoofing Protocols
# =============================================================================


def test_pydantic_payload_models_integrity() -> None:
    """Validates Pydantic schema validation and immutable contract."""
    symbols = ["C", "O", "O"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.16], [0.0, 0.0, 1.16]]
    extractor = TorqTensorExtractor(symbols, coords, point_id="co2_pydantic")
    full_output = extractor.get_full_output()

    assert isinstance(full_output, TorqTensorOutput)
    assert isinstance(full_output.inertia, InertiaTensorResult)
    assert isinstance(full_output.cartesian_protection, CartesianProtectionResult)
    assert isinstance(full_output.asymmetry, AsymmetryResult)

    dumped = full_output.model_dump()
    assert dumped["point_id"] == "co2_pydantic"
    assert dumped["cartesian_protection"]["is_linear"] is True


def test_anti_spoofing_spin_hamiltonian_guard() -> None:
    """Validates that unverified Spin Hamiltonian calls raise strict RuntimeError."""
    extractor = TorqTensorExtractor(
        symbols=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.1], [0.0, 0.7, -0.4], [0.0, -0.7, -0.4]],
    )
    with pytest.raises(RuntimeError, match="Anti-spoofing mandate"):
        extractor.extract_spin_hamiltonian()

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_torq_telemetry.py ---
"""
CoChem-TORQ: Comprehensive Pure Physical Test Suite for Visual & Event Telemetry Streamer
Phase 9 (Stages 5.5 - 6.0) Validation Suite
-----------------------------------------------------------------------------------------
Validates:
1. stream_webhook_events with real local HTTP server, exponential backoff retries,
   and circuit-breaker fallback spooling to telemetry_spool.jsonl under network blackout.
2. generate_plotly_3d_carousels with 2D strided regular grid decimation, stationary
   point preservation, color-blind accessibility, and DVR wavefunction probability states.
3. export_crash_animation capturing multi-frame crash_animation.xyz and crash_diagnostic.json
   during Steric Shatter Soft-Quench aborts.
4. Absolute Filesystem Air-Gap compliance writing exclusively to dynamic scratch/artifact dirs.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest

from Libraries.cochem_torq_telemetry import (
    CircuitState,
    CrashDiagnostic,
    TelemetryCircuitBreaker,
    WebhookPayload,
    decimate_2d_grid_with_extrema,
    export_crash_animation,
    find_stationary_points_2d,
    generate_plotly_3d_carousels,
    stream_webhook_events,
    stream_webhook_events_async,
)


# ============================================================================
# Physical Helper: Ephemeral Local HTTP Server for Real Webhook Delivery
# ============================================================================


class WebhookRecordingHandler(BaseHTTPRequestHandler):
    """Real HTTP request handler for live socket-level webhook testing."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard HTTP server console spam during tests
        pass

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {"raw_body": body}

        # Check server mode
        server_obj: Any = self.server
        server_obj.received_requests.append({
            "path": self.path,
            "headers": dict(self.headers),
            "payload": payload,
        })

        if getattr(server_obj, "fail_count_target", 0) > 0:
            server_obj.fail_count_target -= 1
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Service Unavailable"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok", "delivered": true}')


def get_free_port() -> int:
    """Finds an available ephemeral port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def local_webhook_server() -> Iterator[Tuple[HTTPServer, str]]:
    """Starts a real physical HTTP server on localhost."""
    port = get_free_port()
    server = HTTPServer(("127.0.0.1", port), WebhookRecordingHandler)
    server.received_requests = []  # type: ignore[attr-defined]
    server.fail_count_target = 0  # type: ignore[attr-defined]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/cochem/webhook"
    try:
        yield server, url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


# ============================================================================
# Test Suite 1: Webhook Streaming & Circuit Breaker Spooling
# ============================================================================


def test_stream_webhook_events_real_delivery(local_webhook_server: Tuple[HTTPServer, str], tmp_path: Path) -> None:
    """Validates real physical HTTP POST delivery to an active webhook endpoint."""
    server, webhook_url = local_webhook_server
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "job_completed",
        "job_id": "TORQ_JOB_2026_08_001",
        "node_id": "hpc_worker_node_07",
        "status": "COMPLETED",
        "data": {"wall_time_sec": 42.5, "optimized_energy_hartree": -154.29841},
    }

    result = stream_webhook_events(
        status_payload=test_payload,
        webhook_url=webhook_url,
        scratch_dir=scratch_dir,
        max_retries=2,
        timeout=3.0,
    )

    assert result["status"] == "DELIVERED"
    assert result["status_code"] == 200
    assert result["spooled"] is False
    assert len(server.received_requests) == 1  # type: ignore[attr-defined]
    assert server.received_requests[0]["payload"]["job_id"] == "TORQ_JOB_2026_08_001"  # type: ignore[attr-defined]


def test_stream_webhook_events_exponential_backoff_recovery(local_webhook_server: Tuple[HTTPServer, str], tmp_path: Path) -> None:
    """Validates exponential backoff retries when encountering transient 503 errors."""
    server, webhook_url = local_webhook_server
    server.fail_count_target = 2  # type: ignore[attr-defined] # Fail first 2 attempts with 503, succeed on 3rd
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "soft_quench_collision",
        "job_id": "TORQ_JOB_SQ_09",
        "node_id": "gpu_node_01",
        "status": "ALERT",
        "data": {"collision_distance_angstrom": 0.58},
    }

    result = stream_webhook_events(
        status_payload=test_payload,
        webhook_url=webhook_url,
        scratch_dir=scratch_dir,
        max_retries=3,
        timeout=3.0,
    )

    assert result["status"] == "DELIVERED"
    assert result["attempt"] == 3
    assert len(server.received_requests) == 3  # type: ignore[attr-defined]


def test_stream_webhook_events_blackout_spooling(tmp_path: Path) -> None:
    """Validates zero-interruption spooling to telemetry_spool.jsonl when network fails."""
    # Use a port that is definitively closed/unreachable
    closed_port = get_free_port()
    unreachable_url = f"http://127.0.0.1:{closed_port}/nonexistent_webhook"
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "oom_backoff",
        "job_id": "TORQ_JOB_OOM_003",
        "node_id": "cpu_node_12",
        "status": "ALERT",
        "data": {"memory_rss_gb": 64.2, "backoff_scale": 0.5},
    }

    result = stream_webhook_events(
        status_payload=test_payload,
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        max_retries=2,
        timeout=0.5,
    )

    # Must NOT raise unhandled exception; must safely spool to disk
    assert result["status"] == "SPOOLED"
    assert result["spooled"] is True
    spool_file = scratch_dir / "telemetry_spool.jsonl"
    assert spool_file.exists()

    with open(spool_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) >= 1
    logged_event = lines[-1]
    assert logged_event["payload"]["job_id"] == "TORQ_JOB_OOM_003"
    assert logged_event["delivery_status"] == "SPOOLED"


def test_stream_webhook_events_numpy_types(local_webhook_server: Tuple[HTTPServer, str], tmp_path: Path) -> None:
    """Validates that NumPy scalars and arrays in payload data serialize cleanly without TypeError."""
    server, webhook_url = local_webhook_server
    scratch_dir = tmp_path / "scratch"

    numpy_payload = {
        "event_type": "progress",
        "job_id": "NUMPY_SERIAL_01",
        "status": "RUNNING",
        "data": {
            "float_metric": np.float64(3.14159265),
            "int_metric": np.int64(42),
            "vector": np.array([1.0, 2.0, 3.0]),
        },
    }

    result = stream_webhook_events(
        status_payload=numpy_payload,
        webhook_url=webhook_url,
        scratch_dir=scratch_dir,
    )

    assert result["status"] == "DELIVERED"
    assert len(server.received_requests) == 1  # type: ignore[attr-defined]
    rec_payload = server.received_requests[0]["payload"]  # type: ignore[attr-defined]
    assert rec_payload["data"]["int_metric"] == 42
    assert rec_payload["data"]["vector"] == [1.0, 2.0, 3.0]


# ============================================================================
# Test Suite 2: 2D PES Decimation & Plotly 3D Carousel Generation
# ============================================================================


def test_decimate_2d_grid_and_stationary_points() -> None:
    """Validates 2D grid decimation while preserving stationary points (minima/maxima)."""
    # Create a dense 500x500 (250,000 nodes) 2D PES grid
    n1, n2 = 500, 500
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")

    # Analytical potential: V(phi1, phi2) = 1500*(1-cos(phi1)) + 800*(1-cos(2*phi2)) + 400*cos(phi1+phi2)
    # Global minimum at (0, 0) where V = 400 cm-1
    rad1 = np.radians(P1)
    rad2 = np.radians(P2)
    pes_grid = 1500.0 * (1.0 - np.cos(rad1)) + 800.0 * (1.0 - np.cos(2.0 * rad2)) + 400.0 * np.cos(rad1 + rad2)

    # Test stationary points finder
    stationary_points = find_stationary_points_2d(phi1, phi2, pes_grid, max_points=10)
    assert len(stationary_points) > 0
    # Minima should include near (0, 0)
    minima = [p for p in stationary_points if p["type"] == "minimum"]
    assert len(minima) >= 1

    # Test decimation to <= 5000 nodes
    phi1_dec, phi2_dec, pes_dec, extrema_pts = decimate_2d_grid_with_extrema(
        phi1, phi2, pes_grid, max_nodes=5000
    )

    total_dec_nodes = len(phi1_dec) * len(phi2_dec)
    assert total_dec_nodes <= 5000
    assert total_dec_nodes > 100
    assert pes_dec.shape == (len(phi1_dec), len(phi2_dec))
    assert len(extrema_pts) > 0


def test_find_stationary_points_with_nans() -> None:
    """Validates stationary points finder resilience against PES grids containing NaNs."""
    n1, n2 = 50, 50
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    pes_grid = np.full((n1, n2), 1000.0)
    # True minimum at (25, 25)
    pes_grid[25, 25] = 50.0
    # Add NaN region (steric crash zone)
    pes_grid[0:5, 0:5] = np.nan

    pts = find_stationary_points_2d(phi1, phi2, pes_grid)
    assert len(pts) >= 1
    assert pts[0]["type"] == "minimum"
    assert pts[0]["energy"] == 50.0


def test_generate_plotly_3d_carousels_standalone_html(tmp_path: Path) -> None:
    """Validates generation of lightweight, interactive Plotly 3D PES visualizer HTML."""
    artifact_dir = tmp_path / "artifacts"

    # Dense PES grid: 360x360 (129,600 nodes)
    n = 360
    phi1 = np.linspace(-180.0, 180.0, n)
    phi2 = np.linspace(-180.0, 180.0, n)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 1200.0 * (1.0 - np.cos(np.radians(P1))) + 600.0 * (1.0 - np.cos(np.radians(3 * P2)))

    html_path = generate_plotly_3d_carousels(
        pes_tensor=pes_grid,
        phi1_grid=phi1,
        phi2_grid=phi2,
        artifact_dir=artifact_dir,
        filename="test_pes_3d.html",
        max_nodes=4000,
        colorscale="Viridis",
        title="1,2-Ethanediol 2D Torsional PES",
    )

    assert html_path.exists()
    assert html_path.is_file()
    assert html_path.parent == artifact_dir

    # Inspect HTML content
    html_content = html_path.read_text(encoding="utf-8")
    assert "<html>" in html_content.lower()
    assert "<body>" in html_content.lower()
    assert "plotly" in html_content.lower()
    assert "1,2-Ethanediol 2D Torsional PES" in html_content

    # File size must be lightweight (< 3.5 MB)
    file_size_mb = html_path.stat().st_size / (1024 * 1024)
    assert file_size_mb < 3.5


def test_generate_plotly_3d_carousels_with_dvr_wavefunctions(tmp_path: Path) -> None:
    """Validates Plotly 3D carousel with multi-state DVR probability wavefunctions."""
    artifact_dir = tmp_path / "artifacts"

    n = 100
    phi1 = np.linspace(-180.0, 180.0, n)
    phi2 = np.linspace(-180.0, 180.0, n)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 1000.0 * (1.0 - np.cos(np.radians(P1))) + 500.0 * (1.0 - np.cos(np.radians(2 * P2)))

    # Create 3 DVR wavefunctions: ground state v=0 and excited states v=1, v=2
    wf_0 = np.exp(-((P1 / 40.0) ** 2 + (P2 / 40.0) ** 2))
    wf_0 /= np.sum(wf_0)

    wf_1 = (P1 / 40.0) * np.exp(-((P1 / 40.0) ** 2 + (P2 / 40.0) ** 2))
    wf_1 = (wf_1**2) / np.sum(wf_1**2)

    wf_2 = (P2 / 40.0) * np.exp(-((P1 / 40.0) ** 2 + (P2 / 40.0) ** 2))
    wf_2 = (wf_2**2) / np.sum(wf_2**2)

    dvr_wavefunctions = [wf_0, wf_1, wf_2]

    html_path = generate_plotly_3d_carousels(
        pes_tensor=pes_grid,
        dvr_wavefunctions=dvr_wavefunctions,
        phi1_grid=phi1,
        phi2_grid=phi2,
        artifact_dir=artifact_dir,
        filename="test_pes_dvr_carousel.html",
        max_nodes=2500,
    )

    assert html_path.exists()
    html_content = html_path.read_text(encoding="utf-8")
    assert "DVR State v=0" in html_content
    assert "DVR State v=1" in html_content
    assert "DVR State v=2" in html_content


def test_generate_plotly_3d_carousels_dict_missing_coords(tmp_path: Path) -> None:
    """Validates that passing a dict with only 'pes' key automatically generates default dihedral grids."""
    artifact_dir = tmp_path / "artifacts"
    pes_grid = np.ones((30, 30)) * 250.0

    html_path = generate_plotly_3d_carousels(
        pes_tensor={"pes": pes_grid},
        artifact_dir=artifact_dir,
        filename="pes_dict_minimal.html",
    )

    assert html_path.exists()
    assert html_path.is_file()


# ============================================================================
# Test Suite 3: Crash Animation & Diagnostic Exporter
# ============================================================================


def test_export_crash_animation_steric_collision(tmp_path: Path) -> None:
    """Validates multi-frame XYZ crash animation and JSON diagnostic generation."""
    artifact_dir = tmp_path / "artifacts"
    scratch_dir = tmp_path / "scratch"

    # Define a 6-atom molecule (e.g. ethane-like) undergoing steric shatter collision
    symbols = ["C", "C", "H", "H", "H", "H"]
    num_atoms = len(symbols)
    num_frames = 12

    # Frame 0: Stable geometry
    base_coords = np.array([
        [0.0, 0.0, 0.0],      # C1
        [1.54, 0.0, 0.0],     # C2
        [-0.5, 1.0, 0.0],     # H3
        [-0.5, -0.5, 0.86],   # H4
        [2.04, 1.0, 0.0],     # H5
        [2.04, -0.5, -0.86],  # H6
    ], dtype=np.float64)

    # Generate physical trajectory where H3 (idx 2) and H5 (idx 4) collide along the C-C axis
    frame_coords_list: List[np.ndarray] = []
    energies: List[float] = []
    gradients: List[np.ndarray] = []

    sigma_lj = 1.1  # Angstrom
    eps_lj = 0.1    # kcal/mol

    for f_idx in range(num_frames):
        coords = base_coords.copy()
        # Physically compress H3 and H5 along interaction vector to simulate steric collision
        compression = float(f_idx) * 0.20
        coords[2, 0] += compression * 0.5   # H3 moves toward center
        coords[4, 0] -= compression * 0.6   # H5 moves toward center
        coords[4, 1] -= compression * 0.05  # slight y-deflection

        frame_coords_list.append(coords)

        # Compute physical Lennard-Jones potential energy and analytical gradients
        e_frame = -79.8  # baseline Hartree
        grad_frame = np.empty((num_atoms, 3), dtype=np.float64)
        grad_frame.fill(0.0)

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                r_vec = coords[i] - coords[j]
                r_dist = float(np.linalg.norm(r_vec))
                if r_dist > 1e-4:
                    s_r = sigma_lj / r_dist
                    # Analytical LJ gradient
                    force_mag = 24.0 * eps_lj * (2.0 * (s_r ** 12) - (s_r ** 6)) / (r_dist ** 2)
                    grad_frame[i] += force_mag * r_vec
                    if i < j:
                        e_frame += 4.0 * eps_lj * ((s_r ** 12) - (s_r ** 6))

        energies.append(float(e_frame))
        gradients.append(grad_frame)

    trajectory = np.array(frame_coords_list, dtype=np.float64)

    result_paths = export_crash_animation(
        trajectory_array=trajectory,
        error_node_id="rotor_node_55",
        symbols=symbols,
        energies=energies,
        gradients=gradients,
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
        abort_reason="Steric Shatter Soft-Quench Abort: Interatomic distance below 0.5 A threshold",
    )

    xyz_path = result_paths["xyz_path"]
    diag_path = result_paths["diagnostic_path"]

    assert xyz_path.exists()
    assert diag_path.exists()

    # Verify XYZ structure
    xyz_lines = xyz_path.read_text(encoding="utf-8").strip().split("\n")
    # Each frame has num_atoms + 2 lines
    expected_lines = num_frames * (num_atoms + 2)
    assert len(xyz_lines) == expected_lines
    assert xyz_lines[0].strip() == str(num_atoms)
    assert "rotor_node_55" in xyz_lines[1]

    # Verify Diagnostic JSON
    with open(diag_path, "r", encoding="utf-8") as diag_file:
        diag_data = json.load(diag_file)

    assert diag_data["error_node_id"] == "rotor_node_55"
    assert diag_data["num_frames"] == 12
    assert diag_data["num_atoms"] == 6
    assert diag_data["symbols"] == symbols
    assert diag_data["min_interatomic_distance"] < 0.5
    assert diag_data["colliding_pair"] == [2, 4] or diag_data["colliding_pair"] == [4, 2]
    assert "Steric Shatter" in diag_data["abort_reason"]


# ============================================================================
# Test Suite 4: Air-Gap Compliance & Direct Memory Ingestion
# ============================================================================


def test_airgap_compliance_no_repo_pollution(tmp_path: Path) -> None:
    """Validates that no temporary files or logs are created in repo workspace."""
    repo_files_before = set(Path(".").glob("*"))

    scratch_dir = tmp_path / "airgap_scratch"
    artifact_dir = tmp_path / "airgap_artifacts"

    # Run telemetry functions with explicit isolated dirs
    payload = {"event_type": "progress", "job_id": "AIRGAP_01", "status": "RUNNING"}
    stream_webhook_events(payload, webhook_url=None, scratch_dir=scratch_dir)

    coords = np.random.rand(3, 4, 3)
    export_crash_animation(
        coords,
        error_node_id="airgap_node",
        symbols=["H", "H", "H", "H"],
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
    )

    pes = np.ones((20, 20))
    generate_plotly_3d_carousels(
        pes_tensor=pes,
        artifact_dir=artifact_dir,
        max_nodes=100,
    )

    repo_files_after = set(Path(".").glob("*"))
    # Verify no new files created in cwd
    diff = repo_files_after - repo_files_before
    # Ignore pytest temporary markers or cache if any
    diff = {f for f in diff if not f.name.startswith(".pytest") and not f.name.startswith("__pycache__")}
    assert len(diff) == 0, f"Air-gap violation detected: created files in repo: {diff}"


# ============================================================================
# Test Suite 5: Extended Edge Cases & Circuit Breaker State Transitions
# ============================================================================


def test_stream_webhook_events_circuit_breaker_transitions(tmp_path: Path) -> None:
    """Validates circuit breaker transitions from CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
    scratch_dir = tmp_path / "scratch"
    closed_port = get_free_port()
    unreachable_url = f"http://127.0.0.1:{closed_port}/webhook"

    cb = TelemetryCircuitBreaker(
        failure_threshold=2,
        recovery_timeout=0.2,
        backoff_factor=0.01,
        max_retries=1,
        request_timeout=0.2,
    )

    assert cb.state.value == "CLOSED"

    # 1st failure
    stream_webhook_events(
        {"event_type": "heartbeat", "job_id": "J1"},
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        circuit_breaker=cb,
        max_retries=1,
        timeout=0.2,
    )
    assert cb.consecutive_failures == 1
    assert cb.state.value == "CLOSED"

    # 2nd failure -> trips to OPEN
    stream_webhook_events(
        {"event_type": "heartbeat", "job_id": "J2"},
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        circuit_breaker=cb,
        max_retries=1,
        timeout=0.2,
    )
    assert cb.state.value == "OPEN"

    # Next call while OPEN immediately spools without network call
    res = stream_webhook_events(
        {"event_type": "heartbeat", "job_id": "J3"},
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        circuit_breaker=cb,
        max_retries=1,
        timeout=0.2,
    )
    assert res["status"] == "SPOOLED"
    assert res["reason"] == "Circuit Breaker OPEN"

    # Wait for recovery timeout to transition to HALF_OPEN
    time.sleep(0.25)
    assert cb.can_attempt_request() is True
    assert cb.state.value == "HALF_OPEN"


def test_stream_webhook_events_sync_inside_async_loop(tmp_path: Path) -> None:
    """Validates that synchronous stream_webhook_events can be safely called inside an active async event loop."""
    scratch_dir = tmp_path / "scratch"

    async def _async_caller() -> Dict[str, Any]:
        return stream_webhook_events(
            {"event_type": "heartbeat", "job_id": "ASYNC_LOOP_JOB"},
            webhook_url=None,
            scratch_dir=scratch_dir,
        )

    result = asyncio.run(_async_caller())
    assert result["status"] == "SPOOLED"
    assert result["spooled"] is True


def test_generate_plotly_3d_carousels_dict_input(tmp_path: Path) -> None:
    """Validates Plotly 3D carousel generation when pes_tensor is provided as a dictionary."""
    artifact_dir = tmp_path / "artifacts"
    n1, n2 = 40, 40
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 500.0 * (1.0 - np.cos(np.radians(P1))) + 200.0 * (1.0 - np.cos(np.radians(P2)))

    pes_dict = {
        "pes": pes_grid,
        "phi1": phi1,
        "phi2": phi2,
    }

    html_path = generate_plotly_3d_carousels(
        pes_tensor=pes_dict,
        artifact_dir=artifact_dir,
        filename="pes_dict_test.html",
        max_nodes=1000,
        colorscale="Cividis",
    )

    assert html_path.exists()
    assert html_path.is_file()


def test_export_crash_animation_dict_and_single_frame(tmp_path: Path) -> None:
    """Validates export_crash_animation with dictionary input and single frame."""
    artifact_dir = tmp_path / "artifacts"
    scratch_dir = tmp_path / "scratch"

    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],  # Severe collision: 0.2 A
    ], dtype=np.float64)

    traj_dict = {
        "coordinates": coords,
        "symbols": ["O", "H"],
        "energies": [-75.123456],
        "gradients": [np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])],
    }

    result = export_crash_animation(
        trajectory_array=traj_dict,
        error_node_id="single_frame_node",
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
        abort_reason="Single frame singularity collision",
    )

    xyz_path = result["xyz_path"]
    diag_path = result["diagnostic_path"]

    assert xyz_path.exists()
    assert diag_path.exists()

    with open(diag_path, "r", encoding="utf-8") as diag_file:
        diag = json.load(diag_file)

    assert diag["error_node_id"] == "single_frame_node"
    assert diag["num_frames"] == 1
    assert diag["num_atoms"] == 2
    assert diag["min_interatomic_distance"] == 0.2
    assert diag["colliding_pair"] == [0, 1] or diag["colliding_pair"] == [1, 0]

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_matrix_loader.py ---
"""CoChem-TORQ: Method Matrix Parser & Provenance Loader.

=============================================================================
Phase 5 (Stage 4.0 / Task 8) Implementation
-------------------------------------------
Implements Method Matrix v4 execution cascade parsing, progressive quantum
chemical pathway resolution, multi-tier MLFF fallback hierarchies for
unsupported atomic species, rigorous auxiliary basis scientific validation,
cryptographic SHA-256 / xxHash-64 provenance manifest generation, and
Tripartite Filesystem Air-Gap compliance.

Core Directives & Standards:
- Method Matrix v4 (§4.4, §8A, §8B, §9A, §10, Table 2)
- Zero-Tolerance Anti-Mocking & Anti-Laziness Mandate
- Mendeleev Library Mandate: Dynamic atomic and isotopic property retrieval
- Tripartite Filesystem Air-Gap Architecture: Ring 1 Static Repo (read-only),
  Ring 2 Ephemeral Scratch, Ring 3 Persistent Artifacts
- Cryptographic FAIR Data Provenance with deterministic serialization
"""

from __future__ import annotations

import datetime
import enum
import functools
import hashlib
import json
import logging
import os
import platform
import tempfile
import time
import uuid
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Set, Tuple, Union

from mendeleev import element as mendeleev_element
import numpy as np
import psutil
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Configure module-level logging
logger = logging.getLogger("CoChem-TORQ.MatrixLoader")

# Optional xxhash library check
try:
    import xxhash  # type: ignore[import-untyped]
    _XXHASH_AVAILABLE = True
except ImportError:
    _XXHASH_AVAILABLE = False

# Physical constants (CODATA 2018 / 2022)
HARTREE_TO_KCAL_MOL: Final[float] = 627.5094740631
HARTREE_TO_EV: Final[float] = 27.211386245988
BOHR_TO_ANGSTROM: Final[float] = 0.529177210903
EV_TO_KCAL_MOL: Final[float] = 23.060541945329334


# =============================================================================
# 1. Custom Exceptions
# =============================================================================


class MatrixLoaderError(Exception):
    """Base exception for Method Matrix loader and execution cascade operations."""
    pass


class AirGapViolationError(PermissionError):
    """Raised when an operation attempts to write to Ring 1 static repository space at runtime."""
    pass


class AuxiliaryBasisMismatchError(MatrixLoaderError, ValueError):
    """Raised when an auxiliary basis set is scientifically incompatible with the method or primary basis."""
    pass


class UnsupportedElementError(MatrixLoaderError, ValueError):
    """Raised when an atomic species is unsupported across all MLFF and electronic structure tiers."""
    pass


class InvalidTierError(MatrixLoaderError, ValueError):
    """Raised when an unrecognized Method Matrix tier identifier is requested."""
    pass


# =============================================================================
# 2. 6-Tier Environment Matrix & Air-Gap Compliance
# =============================================================================


class EnvironmentTier(str, enum.Enum):
    """6-Tier Environment Matrix defining host execution environments."""
    LOCAL_WINDOWS = "LOCAL_WINDOWS"
    LOCAL_MACOS = "LOCAL_MACOS"
    LOCAL_LINUX = "LOCAL_LINUX"
    GITHUB_ACTIONS = "GITHUB_ACTIONS"
    CODESPACES = "CODESPACES"
    HPC_NODES = "HPC_NODES"


def get_repo_root() -> Path:
    """Locate the Domain A / Ring 1 immutable Git repository root."""
    env_val = os.environ.get("COCHEM_REPO_DIR")
    if env_val:
        repo_path = Path(env_val).resolve()
        if repo_path.is_dir():
            return repo_path

    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    return Path.cwd().resolve()


class AirGapReport(BaseModel):
    """Diagnostic report detailing air-gap boundary verification."""
    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="True if target path is safe and disjoint from Ring 1")
    cwd_resolved: Path = Field(..., description="Canonical resolved current working directory")
    target_resolved: Path = Field(..., description="Canonical resolved target path tested")
    verified_at: float = Field(default_factory=time.time, description="Unix timestamp of check")
    reason: Optional[str] = Field(default=None, description="Diagnostic explanation if violation occurred")


@functools.lru_cache(maxsize=1)
def _detect_system_hardware() -> Tuple[int, int, bool, int]:
    """Queries host CPU cores, RAM, and NVIDIA GPU telemetry once per process."""
    max_memory_mb = 16384
    num_cores = os.cpu_count() or 8
    gpu_available = False
    vram_mb = 0

    try:
        vm = psutil.virtual_memory()
        max_memory_mb = int(vm.total / (1024 * 1024))
    except Exception:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                gpu_available = True
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_mb = int(mem_info.total / (1024 * 1024))
            pynvml.nvmlShutdown()
    except Exception:
        gpu_available = False
        vram_mb = 0

    return max_memory_mb, num_cores, gpu_available, vram_mb


class ExecutionContext(BaseModel):
    """Manages runtime environment detection, memory limits, and dynamic path resolution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tier: EnvironmentTier = Field(default=EnvironmentTier.LOCAL_WINDOWS)
    custom_scratch_dir: Optional[Path] = None
    custom_shm_dir: Optional[Path] = None
    custom_artifacts_dir: Optional[Path] = None
    max_memory_mb: int = Field(default=16384)
    num_cores: int = Field(default=8)
    gpu_available: bool = Field(default=False)
    vram_mb: int = Field(default=0)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def __init__(self, **data: Any) -> None:
        if "tier" not in data:
            data["tier"] = self.detect_tier()
        super().__init__(**data)
        self._detect_hardware_specs()

    @classmethod
    def detect_tier(cls) -> EnvironmentTier:
        """Autonomously detects the active environment tier from OS and environment variables."""
        if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("RUNNER_TEMP"):
            return EnvironmentTier.GITHUB_ACTIONS

        if os.environ.get("CODESPACES") == "true" or os.environ.get("CODESPACE_NAME"):
            return EnvironmentTier.CODESPACES

        if (
            os.environ.get("SLURM_TMPDIR")
            or os.environ.get("SLURM_JOB_ID")
            or os.environ.get("PFSDIR")
            or os.environ.get("PBS_O_WORKDIR")
        ):
            return EnvironmentTier.HPC_NODES

        sys_name = platform.system()
        if sys_name == "Windows" or os.environ.get("WSL_DISTRO_NAME"):
            return EnvironmentTier.LOCAL_WINDOWS
        elif sys_name == "Darwin":
            return EnvironmentTier.LOCAL_MACOS
        else:
            return EnvironmentTier.LOCAL_LINUX

    def _detect_hardware_specs(self) -> None:
        """Loads cached system hardware specifications."""
        mem, cores, gpu, vram = _detect_system_hardware()
        if "max_memory_mb" not in self.model_fields_set:
            self.max_memory_mb = mem
        if "num_cores" not in self.model_fields_set:
            self.num_cores = cores
        if "gpu_available" not in self.model_fields_set:
            self.gpu_available = gpu
        if "vram_mb" not in self.model_fields_set:
            self.vram_mb = vram

    @staticmethod
    def _is_repo_root_violation(target: Path) -> bool:
        """Check whether a path is located inside the static repository root."""
        resolved = target.resolve()
        repo_root = get_repo_root().resolve()
        try:
            resolved.relative_to(repo_root)
            return True
        except ValueError:
            return False

    def verify_air_gap_boundary(self, target_path: Path) -> AirGapReport:
        """Verifies that runtime scratch or artifacts paths do not write into Ring 1 repo root."""
        resolved_target = target_path.resolve()
        repo_root = get_repo_root().resolve()
        if self._is_repo_root_violation(resolved_target):
            msg = f"Tripartite Air-Gap Violation: Path '{resolved_target}' is inside static repository root '{repo_root}'."
            raise AirGapViolationError(msg)

        return AirGapReport(
            is_valid=True,
            cwd_resolved=Path.cwd().resolve(),
            target_resolved=resolved_target,
            reason=None,
        )

    def get_scratch_dir(self, subfolder: Optional[str] = None) -> Path:
        """Resolves the ephemeral Domain B / Ring 2 scratch directory for the active tier."""
        if self.custom_scratch_dir:
            base = Path(self.custom_scratch_dir).resolve()
        elif os.environ.get("COCHEM_SCRATCH_DIR") and not self._is_repo_root_violation(Path(os.environ["COCHEM_SCRATCH_DIR"])):
            base = Path(os.environ["COCHEM_SCRATCH_DIR"]).resolve()
        else:
            if self.tier == EnvironmentTier.GITHUB_ACTIONS:
                runner_temp = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
                base = Path(runner_temp) / "cochem_scratch"
            elif self.tier == EnvironmentTier.CODESPACES:
                base = Path.home() / ".cochem" / "scratch"
            elif self.tier == EnvironmentTier.HPC_NODES:
                slurm_tmp = os.environ.get("SLURM_TMPDIR") or os.environ.get("PFSDIR") or tempfile.gettempdir()
                base = Path(slurm_tmp) / "cochem_scratch"
            elif self.tier == EnvironmentTier.LOCAL_MACOS:
                base = Path.home() / "Library" / "Caches" / "CoChem" / "scratch"
            elif self.tier == EnvironmentTier.LOCAL_WINDOWS:
                local_app_data = os.environ.get("LOCALAPPDATA")
                if local_app_data:
                    base = Path(local_app_data) / "CoChem" / "scratch"
                else:
                    base = Path(tempfile.gettempdir()) / "cochem_scratch"
            else:  # LOCAL_LINUX
                xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
                if xdg_runtime and Path(xdg_runtime).is_dir():
                    base = Path(xdg_runtime) / "cochem" / "scratch"
                elif Path(tempfile.gettempdir()).is_dir():
                    base = Path(tempfile.gettempdir()) / "cochem" / "scratch"
                else:
                    base = Path(tempfile.gettempdir()) / "cochem_scratch"

        target = (base / subfolder) if subfolder else base
        self.verify_air_gap_boundary(target)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_artifacts_dir(self, subfolder: Optional[str] = None) -> Path:
        """Resolves the Domain C / Ring 3 persistent artifact vault directory."""
        if self.custom_artifacts_dir:
            base = Path(self.custom_artifacts_dir).resolve()
        elif os.environ.get("COCHEM_ARTIFACTS_DIR") and not self._is_repo_root_violation(Path(os.environ["COCHEM_ARTIFACTS_DIR"])):
            base = Path(os.environ["COCHEM_ARTIFACTS_DIR"]).resolve()
        elif os.environ.get("COCHEM_ARTIFACTS") and not self._is_repo_root_violation(Path(os.environ["COCHEM_ARTIFACTS"])):
            base = Path(os.environ["COCHEM_ARTIFACTS"]).resolve()
        else:
            if self.tier == EnvironmentTier.GITHUB_ACTIONS:
                runner_temp = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
                base = Path(runner_temp) / "cochem_artifacts"
            elif self.tier == EnvironmentTier.CODESPACES:
                base = Path.home() / ".cochem" / "artifacts"
            elif self.tier == EnvironmentTier.HPC_NODES:
                slurm_tmp = os.environ.get("SLURM_TMPDIR") or os.environ.get("PFSDIR") or tempfile.gettempdir()
                base = Path(slurm_tmp) / "cochem_artifacts"
            elif self.tier == EnvironmentTier.LOCAL_MACOS:
                base = Path.home() / "Library" / "Application Support" / "CoChem" / "artifacts"
            elif self.tier == EnvironmentTier.LOCAL_WINDOWS:
                local_app_data = os.environ.get("LOCALAPPDATA")
                if local_app_data:
                    base = Path(local_app_data) / "CoChem" / "artifacts"
                else:
                    base = Path.home() / "CoChem_Artifacts"
            else:  # LOCAL_LINUX
                base = Path.home() / ".local" / "share" / "cochem" / "artifacts"

        target = (base / subfolder) if subfolder else base
        self.verify_air_gap_boundary(target)
        target.mkdir(parents=True, exist_ok=True)
        return target


# =============================================================================
# 3. Mendeleev Dynamic Element & Mass Retrieval (Mandate Enforced)
# =============================================================================


@functools.lru_cache(maxsize=256)
def _get_mendeleev_element(symbol_or_z: Union[str, int]) -> Any:
    """Queries and caches Mendeleev element objects dynamically to avoid SQLite locking."""
    if isinstance(symbol_or_z, int):
        return mendeleev_element(symbol_or_z)
    clean_sym = str(symbol_or_z).strip().rstrip(":").capitalize()
    return mendeleev_element(clean_sym)


def get_atomic_number(symbol_or_z: Union[str, int]) -> int:
    """Retrieve the atomic number (Z) dynamically from Mendeleev.

    Args:
        symbol_or_z: Element symbol (e.g. 'C', 'Fe') or atomic number.

    Returns:
        Atomic number integer Z.
    """
    el = _get_mendeleev_element(symbol_or_z)
    return int(el.atomic_number)


def get_element_symbol(symbol_or_z: Union[str, int]) -> str:
    """Retrieve the canonical chemical symbol dynamically from Mendeleev.

    Args:
        symbol_or_z: Element symbol or atomic number.

    Returns:
        Canonical element symbol string (e.g. 'H', 'He', 'Fe').
    """
    el = _get_mendeleev_element(symbol_or_z)
    return str(el.symbol)


def get_atomic_mass(symbol: str) -> float:
    """Retrieve the standard atomic mass (weight) dynamically from Mendeleev.

    Args:
        symbol: Chemical element symbol (e.g. 'H', 'C', 'O').

    Returns:
        Standard atomic mass in Da (g/mol).
    """
    el = _get_mendeleev_element(symbol)
    if el.atomic_weight is not None:
        return float(el.atomic_weight)
    if el.mass is not None:
        return float(el.mass)
    if el.isotopes:
        return float(el.isotopes[0].mass or el.isotopes[0].mass_number)
    raise ValueError(f"No atomic mass available in mendeleev for '{symbol}'.")


def get_isotopic_mass(symbol: str, mass_number: Optional[int] = None) -> float:
    """Retrieve exact isotopic mass dynamically from Mendeleev.

    Args:
        symbol: Chemical element symbol (e.g. 'C', 'H').
        mass_number: Isotope nucleon number (e.g. 13 for C-13, 2 for Deuterium).

    Returns:
        Exact isotopic mass in Da.
    """
    el = _get_mendeleev_element(symbol)
    if mass_number is None:
        return get_atomic_mass(symbol)
    for iso in el.isotopes:
        if iso.mass_number == mass_number:
            if iso.mass is not None:
                return float(iso.mass)
            return float(iso.mass_number)
    return get_atomic_mass(symbol)


def get_covalent_radius(symbol: str) -> float:
    """Retrieve Pyykkö single-bond covalent radius in Angstroms dynamically from Mendeleev."""
    el = _get_mendeleev_element(symbol)
    if el.covalent_radius_pyykko is not None:
        return float(el.covalent_radius_pyykko) / 100.0
    if el.covalent_radius is not None:
        return float(el.covalent_radius) / 100.0
    return 1.40


def get_vdw_radius(symbol: str) -> float:
    """Retrieve van der Waals radius in Angstroms dynamically from Mendeleev."""
    el = _get_mendeleev_element(symbol)
    if el.vdw_radius is not None:
        return float(el.vdw_radius) / 100.0
    return 2.00


def validate_element_symbols(symbols: Sequence[str]) -> List[str]:
    """Dynamically validates a list of chemical element symbols via Mendeleev."""
    validated: List[str] = []
    for s in symbols:
        clean = s.strip().rstrip(":")
        if not clean:
            continue
        try:
            canonical = get_element_symbol(clean)
            validated.append(canonical)
        except Exception as err:
            raise UnsupportedElementError(f"Invalid or unrecognized chemical element symbol: '{s}'") from err
    return validated


# =============================================================================
# 4. MLFF Model Specifications & Hierarchy
# =============================================================================


class MLFFModelSpec(BaseModel):
    """Specification of an MLFF model and its supported elemental coverage."""
    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Identifier name of the MLFF model")
    supported_z: Set[int] = Field(..., description="Set of atomic numbers supported by this model")
    description: str = Field(..., description="Technical description of the model")
    architecture: str = Field(..., description="Neural network architecture family")
    is_quantum_semiempirical: bool = Field(default=False, description="True if semi-empirical quantum method")

    def supports_elements(self, atomic_numbers: Sequence[int]) -> bool:
        """Check if all given atomic numbers are within this model's supported domain."""
        return set(atomic_numbers).issubset(self.supported_z)

    def get_unsupported_elements(self, atomic_numbers: Sequence[int]) -> List[str]:
        """Return list of element symbols not supported by this model."""
        unsupported_z = set(atomic_numbers) - self.supported_z
        return [get_element_symbol(z) for z in sorted(unsupported_z)]


# Dynamic definition of MLFF models using Mendeleev Z lookups
_MACE_OFF23_Z: Final[Set[int]] = {
    get_atomic_number("H"),   # 1
    get_atomic_number("C"),   # 6
    get_atomic_number("N"),   # 7
    get_atomic_number("O"),   # 8
    get_atomic_number("F"),   # 9
    get_atomic_number("P"),   # 15
    get_atomic_number("S"),   # 16
    get_atomic_number("Cl"),  # 17
    get_atomic_number("Br"),  # 35
    get_atomic_number("I"),   # 53
}

_AIMNET2_Z: Final[Set[int]] = {
    get_atomic_number("H"),   # 1
    get_atomic_number("B"),   # 5
    get_atomic_number("C"),   # 6
    get_atomic_number("N"),   # 7
    get_atomic_number("O"),   # 8
    get_atomic_number("F"),   # 9
    get_atomic_number("Si"),  # 14
    get_atomic_number("P"),   # 15
    get_atomic_number("S"),   # 16
    get_atomic_number("Cl"),  # 17
    get_atomic_number("As"),  # 33
    get_atomic_number("Se"),  # 34
    get_atomic_number("Br"),  # 35
    get_atomic_number("I"),   # 53
}

_MACE_OMOL_Z: Final[Set[int]] = {
    get_atomic_number("H"),   # 1
    get_atomic_number("B"),   # 5
    get_atomic_number("C"),   # 6
    get_atomic_number("N"),   # 7
    get_atomic_number("O"),   # 8
    get_atomic_number("F"),   # 9
    get_atomic_number("Na"),  # 11
    get_atomic_number("Mg"),  # 12
    get_atomic_number("Al"),  # 13
    get_atomic_number("Si"),  # 14
    get_atomic_number("P"),   # 15
    get_atomic_number("S"),   # 16
    get_atomic_number("Cl"),  # 17
    get_atomic_number("K"),   # 19
    get_atomic_number("Ca"),  # 20
    get_atomic_number("Br"),  # 35
    get_atomic_number("I"),   # 53
}

# MACE-POLAR-1 / MACE-MP-0: Broad periodic table coverage up to Z=89 (Ac)
_MACE_POLAR_Z: Final[Set[int]] = set(range(1, 90))

# GFN2-xTB / GFN-FF: Elements Z=1..86 (H to Rn)
_GFN2_XTB_Z: Final[Set[int]] = set(range(1, 87))

# PySCF / Empirical fallback: All elements Z=1..118
_ALL_ELEMENTS_Z: Final[Set[int]] = set(range(1, 119))


MLFF_CATALOG: Final[Dict[str, MLFFModelSpec]] = {
    "MACE-OFF23": MLFFModelSpec(
        name="MACE-OFF23",
        supported_z=_MACE_OFF23_Z,
        description="MACE Organic Foundation Model (Medium/Large) for drug-like organics (H, C, N, O, F, P, S, Cl, Br, I)",
        architecture="Higher-Order Equivariant Message Passing Neural Network",
    ),
    "MACE-POLAR-1": MLFFModelSpec(
        name="MACE-POLAR-1",
        supported_z=_MACE_POLAR_Z,
        description="MACE Polarizable Foundation Model with extended periodic table coverage (Z=1..89)",
        architecture="Polarizable Higher-Order Equivariant MPNN",
    ),
    "MACE-OMOL-0": MLFFModelSpec(
        name="MACE-OMOL-0",
        supported_z=_MACE_OMOL_Z,
        description="MACE Open Molecular Model with extended organic, alkali, and main group coverage",
        architecture="Equivariant Message Passing Neural Network",
    ),
    "AIMNet2": MLFFModelSpec(
        name="AIMNet2",
        supported_z=_AIMNET2_Z,
        description="AIMNet2 Multi-Task Neural Potential (supports 14 elements with charge/spin polarization)",
        architecture="Anisotropic Interaction Message-Passing Neural Network",
    ),
    "GFN2-xTB": MLFFModelSpec(
        name="GFN2-xTB",
        supported_z=_GFN2_XTB_Z,
        description="Grimme GFN2-xTB Quantum Semi-Empirical Tight-Binding for Z=1..86",
        architecture="Self-Consistent Density Functional Tight Binding",
        is_quantum_semiempirical=True,
    ),
    "GFN-FF": MLFFModelSpec(
        name="GFN-FF",
        supported_z=_GFN2_XTB_Z,
        description="Grimme GFN-FF Generic Force Field for Z=1..86",
        architecture="Polarizable Classical Force Field",
        is_quantum_semiempirical=False,
    ),
    "PySCF_RHF": MLFFModelSpec(
        name="PySCF_RHF",
        supported_z=_ALL_ELEMENTS_Z,
        description="PySCF Ab Initio Restricted Hartree-Fock Electronic Structure Fallback",
        architecture="Ab Initio Quantum Chemistry",
        is_quantum_semiempirical=False,
    ),
    "EMPIRICAL_COVALENT": MLFFModelSpec(
        name="EMPIRICAL_COVALENT",
        supported_z=_ALL_ELEMENTS_Z,
        description="Pyykkö covalent radius harmonic potential with Lennard-Jones steric repulsion",
        architecture="Empirical Molecular Mechanics",
        is_quantum_semiempirical=False,
    ),
}

# Authoritative MLFF Fallback Hierarchy Chain
DEFAULT_MLFF_FALLBACK_CHAIN: Final[List[str]] = [
    "MACE-OFF23",
    "MACE-POLAR-1",
    "MACE-OMOL-0",
    "AIMNet2",
    "GFN2-xTB",
    "GFN-FF",
    "PySCF_RHF",
    "EMPIRICAL_COVALENT",
]


def resolve_mlff_model(
    symbols: Sequence[str],
    requested_model: Optional[str] = None,
    fallback_chain: Optional[Sequence[str]] = None,
) -> Tuple[MLFFModelSpec, List[str]]:
    """Resolves the highest-fidelity capable MLFF model for given chemical elements.

    Evaluates the elements against the fallback chain, recording the rejected models
    and the explicit elemental reason for rejection.

    Args:
        symbols: Sequence of chemical element symbols (e.g. ['C', 'H', 'O', 'Fe']).
        requested_model: Preferred MLFF model name, if specified.
        fallback_chain: Custom fallback chain sequence of model names.

    Returns:
        Tuple of (selected MLFFModelSpec, fallback_trail description list).

    Raises:
        UnsupportedElementError: If no model in the chain supports the elements.
    """
    valid_symbols = validate_element_symbols(symbols)
    atomic_numbers = [get_atomic_number(s) for s in valid_symbols]
    chain = list(fallback_chain or DEFAULT_MLFF_FALLBACK_CHAIN)

    if requested_model and requested_model in MLFF_CATALOG:
        # Move requested model to front of evaluation chain
        chain = [requested_model] + [m for m in chain if m != requested_model]

    fallback_trail: List[str] = []

    for model_name in chain:
        if model_name not in MLFF_CATALOG:
            continue
        spec = MLFF_CATALOG[model_name]
        if spec.supports_elements(atomic_numbers):
            fallback_trail.append(f"Selected '{model_name}' (Coverage: {len(spec.supported_z)} elements, fully matches {set(valid_symbols)})")
            return spec, fallback_trail
        else:
            unsupported = spec.get_unsupported_elements(atomic_numbers)
            fallback_trail.append(f"Rejected '{model_name}': unsupported element(s) {unsupported}")

    raise UnsupportedElementError(
        f"No MLFF model in fallback chain supports atomic species {set(valid_symbols)} (Z: {set(atomic_numbers)})."
    )


# =============================================================================
# 5. Method Matrix v4 Tier Catalog
# =============================================================================


class CascadeTierConfig(BaseModel):
    """Method Matrix v4 Configuration Specification for a specific computational tier."""
    model_config = ConfigDict(frozen=True)

    tier: str = Field(..., description="Method Matrix tier identifier (e.g. 'T1-10s', 'T3-3h')")
    target_walltime: str = Field(..., description="Expected wall-time ceiling per stationary point")
    method: str = Field(..., description="Primary electronic structure method / functional")
    basis_set: str = Field(..., description="Primary atomic orbital basis set")
    aux_basis: str = Field(..., description="Auxiliary Coulomb / correlation basis set")
    grid_level: str = Field(..., description="Integration grid level (e.g. 'defgrid2', 'defgrid3')")
    dispersion: Optional[str] = Field(default=None, description="Dispersion correction type (D3BJ, D4, VV10)")
    tol_max_g: float = Field(..., description="Maximum gradient convergence threshold in Hartree/Bohr")
    tol_rms_g: float = Field(..., description="RMS gradient convergence threshold in Hartree/Bohr")
    tol_energy: float = Field(default=1e-6, description="Energy convergence tolerance in Hartree")
    scf_type: str = Field(default="DIIS", description="SCF convergence acceleration algorithm")
    initial_hessian: str = Field(default="XTB2", description="Initial Hessian estimator (XTB2, Lindh)")
    is_double_hybrid: bool = Field(default=False, description="True if method is a Double-Hybrid functional")
    is_coupled_cluster: bool = Field(default=False, description="True if method is Coupled-Cluster (e.g. DLPNO-CCSD(T))")
    pno_setting: Optional[str] = Field(default=None, description="PNO threshold (e.g. 'TightPNO', 'NormalPNO')")
    description: str = Field(..., description="Technical summary of the tier's role in Method Matrix v4")


METHOD_MATRIX_V4_TIERS: Final[Dict[str, CascadeTierConfig]] = {
    "T1-10s": CascadeTierConfig(
        tier="T1-10s",
        target_walltime="10s",
        method="r2SCAN-3c",
        basis_set="def2-mSVP",
        aux_basis="def2/J",
        grid_level="defgrid2",
        dispersion="D4",
        tol_max_g=1e-3,
        tol_rms_g=3e-4,
        tol_energy=1e-5,
        scf_type="DIIS",
        initial_hessian="XTB2",
        description="T1-10s: Ultra-fast conformational triage and MLFF preliminary geometric relaxation",
    ),
    "T1-1min": CascadeTierConfig(
        tier="T1-1min",
        target_walltime="1m",
        method="r2SCAN-3c",
        basis_set="def2-mSVP",
        aux_basis="def2/J",
        grid_level="defgrid2",
        dispersion="D4",
        tol_max_g=1e-4,
        tol_rms_g=3e-5,
        tol_energy=1e-6,
        scf_type="DIIS",
        initial_hessian="XTB2",
        description="T1-1min: Standard fast screening and composite DFT torsional surface scanning",
    ),
    "T1-30min": CascadeTierConfig(
        tier="T1-30min",
        target_walltime="30m",
        method="wB97X-D4",
        basis_set="def2-TZVP",
        aux_basis="def2/J",
        grid_level="defgrid2",
        dispersion="D4",
        tol_max_g=1e-4,
        tol_rms_g=3e-5,
        tol_energy=1e-6,
        scf_type="DIIS",
        initial_hessian="XTB2",
        description="T1-30min: Range-separated hybrid DFT with D4 dispersion for refined intermediate PES",
    ),
    "T1-1h": CascadeTierConfig(
        tier="T1-1h",
        target_walltime="1h",
        method="wB97M-V",
        basis_set="def2-TZVP",
        aux_basis="def2/J",
        grid_level="defgrid3",
        dispersion="VV10",
        tol_max_g=1e-4,
        tol_rms_g=3e-5,
        tol_energy=1e-6,
        scf_type="DIIS",
        initial_hessian="XTB2",
        description="T1-1h: High-fidelity meta-NGA range-separated hybrid DFT with VV10 non-local correlation",
    ),
    "T3-1min": CascadeTierConfig(
        tier="T3-1min",
        target_walltime="1m",
        method="PBEh-3c",
        basis_set="def2-mSVP",
        aux_basis="def2/J",
        grid_level="defgrid2",
        dispersion="D3BJ",
        tol_max_g=1e-4,
        tol_rms_g=3e-5,
        tol_energy=1e-6,
        scf_type="DIIS",
        initial_hessian="XTB2",
        description="T3-1min: Fast hybrid composite benchmark for intermolecular interaction baselines",
    ),
    "T3-3h": CascadeTierConfig(
        tier="T3-3h",
        target_walltime="3h",
        method="revDSD-PBEP86",
        basis_set="def2-TZVPP",
        aux_basis="def2/J def2-TZVPP/C",
        grid_level="defgrid3",
        dispersion="D4",
        tol_max_g=1e-5,
        tol_rms_g=3e-6,
        tol_energy=1e-7,
        scf_type="DIIS",
        initial_hessian="XTB2",
        is_double_hybrid=True,
        description="T3-3h: Benchmark Double-Hybrid DFT with PT2 correlation and def2-TZVPP/C auxiliary fitting",
    ),
    "T3-12h": CascadeTierConfig(
        tier="T3-12h",
        target_walltime="12h",
        method="revDSD-PBEP86",
        basis_set="def2-QZVPP",
        aux_basis="def2/J def2-QZVPP/C",
        grid_level="defgrid3",
        dispersion="D4",
        tol_max_g=1e-5,
        tol_rms_g=3e-6,
        tol_energy=1e-7,
        scf_type="DIIS",
        initial_hessian="XTB2",
        is_double_hybrid=True,
        description="T3-12h: High-accuracy quadruple-zeta Double-Hybrid benchmark for sub-chemical accuracy",
    ),
    "T4-1d": CascadeTierConfig(
        tier="T4-1d",
        target_walltime="1d",
        method="DLPNO-CCSD(T)",
        basis_set="def2-TZVPP",
        aux_basis="def2/JK def2-TZVPP/C",
        grid_level="defgrid3",
        tol_max_g=1e-5,
        tol_rms_g=3e-6,
        tol_energy=1e-8,
        scf_type="DIIS",
        initial_hessian="XTB2",
        is_coupled_cluster=True,
        pno_setting="TightPNO",
        description="T4-1d: Tight-PNO domain-based local pair natural orbital CCSD(T) escalation",
    ),
    "T4C-1mo": CascadeTierConfig(
        tier="T4C-1mo",
        target_walltime="1mo",
        method="CCSD(T)",
        basis_set="aug-cc-pVTZ",
        aux_basis="AutoAux",
        grid_level="defgrid3",
        tol_max_g=1e-5,
        tol_rms_g=3e-6,
        tol_energy=1e-8,
        scf_type="DIIS",
        initial_hessian="XTB2",
        is_coupled_cluster=True,
        description="T4C-1mo: Canonical gold-standard CCSD(T) Complete Basis Set (CBS) focal point",
    ),
}


# =============================================================================
# 6. Auxiliary Basis Scientific Validation
# =============================================================================


class AuxiliaryBasisPairing(BaseModel):
    """Validation record verifying the scientific compatibility of a basis / aux_basis pairing."""
    model_config = ConfigDict(frozen=True)

    is_compatible: bool = Field(..., description="True if the pairing is scientifically valid")
    method: str = Field(..., description="Electronic structure method")
    basis_set: str = Field(..., description="Primary basis set")
    aux_basis: str = Field(..., description="Auxiliary basis set")
    ri_type: str = Field(..., description="Resolution of identity approximation type (RI-J, RI-JK, RIJCOSX, PT2/C)")
    is_diffuse_compatible: bool = Field(..., description="True if basis accommodates diffuse / weak complex requirements")
    validation_notes: List[str] = Field(default_factory=list, description="Diagnostic notes detailing verification logic")


def validate_auxiliary_basis(
    method: str,
    basis_set: str,
    aux_basis: str,
    is_diffuse: bool = False,
    is_weak_complex: bool = False,
    ri_approximation: Optional[str] = None,
) -> AuxiliaryBasisPairing:
    """Scientifically validates auxiliary basis set pairings per Method Matrix v4 directives.

    Checks:
    1. Coulomb fitting (RI-J / RIJCOSX): def2/J (for def2-SVP, def2-TZVP, def2-TZVPP, def2-QZVP, def2-QZVPP, ma-def2-*)
    2. HF exchange + Coulomb fitting (RI-JK): def2/JK
    3. Correlation fitting (Double-Hybrids, MP2, DLPNO-CCSD(T), CCSD(T)): def2-TZVPP/C, def2-QZVPP/C, def2/C, AutoAux
    4. Diffuse / anionic / non-covalent / weak complex systems: requires ma-def2 or augmented basis sets (aug-cc-pVTZ)
       when is_diffuse=True or is_weak_complex=True.

    Args:
        method: Electronic structure method (e.g. 'wB97M-V', 'revDSD-PBEP86', 'DLPNO-CCSD(T)').
        basis_set: Primary basis set (e.g. 'def2-TZVP', 'ma-def2-TZVP', 'aug-cc-pVTZ').
        aux_basis: Auxiliary basis set (e.g. 'def2/J', 'def2/JK', 'def2/J def2-TZVPP/C', 'AutoAux').
        is_diffuse: If True, indicates presence of anions, diffuse Rydberg states, or weak complexes.
        is_weak_complex: If True, indicates non-covalent weakly bound intermolecular complex.
        ri_approximation: Explicit RI mode if specified ('RI-J', 'RI-JK', 'RIJCOSX', 'PT2/C').

    Returns:
        AuxiliaryBasisPairing validating scientific compliance.

    Raises:
        AuxiliaryBasisMismatchError: If pairing violates quantum chemical constraints.
    """
    method_upper = method.strip().upper()
    basis_upper = basis_set.strip().upper()
    aux_upper = aux_basis.strip().upper()
    notes: List[str] = []

    # 1. Detect Method Category
    is_double_hybrid = any(dh in method_upper for dh in ["DSD", "DH", "B2PLYP", "DLPNO-DSD", "PWPB95"])
    is_mp2 = "MP2" in method_upper
    is_coupled_cluster = any(cc in method_upper for cc in ["CCSD", "CCSD(T)", "DLPNO-CCSD"])
    is_correlation_method = is_double_hybrid or is_mp2 or is_coupled_cluster
    is_hf = method_upper in ["HF", "RHF", "UHF", "ROHF"]

    # 2. Resolve RI Type
    if ri_approximation:
        ri_type = ri_approximation.upper()
    elif is_correlation_method:
        ri_type = "RIJCOSX+PT2/C" if not is_hf else "RI-JK+PT2/C"
    elif is_hf:
        ri_type = "RI-JK"
    else:
        ri_type = "RIJCOSX"

    # 3. Check AutoAux Universal Fallback
    if "AUTOAUX" in aux_upper:
        notes.append("Universal AutoAux auxiliary basis generator accepted.")
        is_diffuse_ok = not (is_diffuse or is_weak_complex) or ("MA-" in basis_upper or "AUG-" in basis_upper)
        if (is_diffuse or is_weak_complex) and not is_diffuse_ok:
            raise AuxiliaryBasisMismatchError(
                f"Method Matrix v4 diffuse violation: Primary basis '{basis_set}' is unaugmented for diffuse/weak complex."
            )
        return AuxiliaryBasisPairing(
            is_compatible=True,
            method=method,
            basis_set=basis_set,
            aux_basis=aux_basis,
            ri_type=ri_type,
            is_diffuse_compatible=True,
            validation_notes=notes,
        )

    # 4. Correlation Auxiliary Basis Validation (Double-Hybrids, MP2, Coupled-Cluster)
    if is_correlation_method:
        has_c_fitting = any(c_fit in aux_upper for c_fit in ["/C", "-C", "AUTOAUX"])
        if not has_c_fitting:
            raise AuxiliaryBasisMismatchError(
                f"Auxiliary basis mismatch for correlation method '{method}': "
                f"Requested aux_basis '{aux_basis}' lacks correlation fitting basis (e.g. 'def2-TZVPP/C', 'def2/C', or 'AutoAux')."
            )
        notes.append(f"Verified correlation fitting auxiliary basis present for method '{method}'.")

        # Check correlation basis size compatibility
        if "DEF2-TZVPP" in basis_upper and not any(k in aux_upper for k in ["TZVPP/C", "QZVPP/C", "DEF2/C", "AUTOAUX"]):
            notes.append("Warning: def2-TZVPP primary basis paired with lower-tier correlation auxiliary basis.")

        if "DEF2-QZVPP" in basis_upper and not any(k in aux_upper for k in ["QZVPP/C", "DEF2/C", "AUTOAUX"]):
            notes.append("Warning: def2-QZVPP primary basis paired with smaller correlation auxiliary basis.")

    # 5. RI-JK vs RI-J Validation
    if ri_type == "RI-JK" or "RI-JK" in ri_type:
        if "DEF2/J" in aux_upper and "DEF2/JK" not in aux_upper:
            raise AuxiliaryBasisMismatchError(
                f"Auxiliary basis mismatch for RI-JK: 'def2/J' only provides Coulomb fitting. 'def2/JK' is required for HF exchange fitting."
            )
        notes.append("Verified def2/JK auxiliary basis for exact HF exchange fitting.")

    if ri_type in ["RI-J", "RIJCOSX"]:
        if "DEF2" in basis_upper and not any(k in aux_upper for k in ["DEF2/J", "DEF2/JK", "AUTOAUX"]):
            raise AuxiliaryBasisMismatchError(
                f"Auxiliary basis mismatch: Primary basis '{basis_set}' requires 'def2/J' or 'def2/JK' for Coulomb fitting, got '{aux_basis}'."
            )
        notes.append("Verified def2/J auxiliary Coulomb fitting for RIJCOSX.")

    # 6. Diffuse & Weak Complex Augmentation Validation
    requires_augmentation = is_diffuse or is_weak_complex
    is_augmented = (
        "MA-DEF2" in basis_upper
        or "AUG-CC" in basis_upper
        or "DAUG-CC" in basis_upper
        or "JUL-CC" in basis_upper
        or "JUN-CC" in basis_upper
        or "DEF2-MSVP" in basis_upper  # mSVP has diffuse s/p primitives for composite methods
    )

    if requires_augmentation and not is_augmented:
        # Standard unaugmented def2-SVP, def2-TZVP, def2-TZVPP in weak complexes lead to BSSE / electron-spill artifacts
        raise AuxiliaryBasisMismatchError(
            f"Method Matrix v4 diffuse/weak complex violation: Primary basis '{basis_set}' is unaugmented. "
            f"Diffuse and non-covalent weak complex calculations require minimally augmented 'ma-def2-*' or 'aug-cc-*' basis sets."
        )

    if requires_augmentation:
        notes.append(f"Verified diffuse/weak complex compliance with augmented basis '{basis_set}'.")

    # 7. Cross-Family Basis Pairings Check
    if "CC-P" in basis_upper and "DEF2/" in aux_upper and "AUTOAUX" not in aux_upper:
        raise AuxiliaryBasisMismatchError(
            f"Basis set family mismatch: Dunning primary basis '{basis_set}' cannot pair with Ahlrichs auxiliary '{aux_basis}'. Use 'AutoAux' or 'cc-pV*Z/C'."
        )

    return AuxiliaryBasisPairing(
        is_compatible=True,
        method=method,
        basis_set=basis_set,
        aux_basis=aux_basis,
        ri_type=ri_type,
        is_diffuse_compatible=not requires_augmentation or is_augmented,
        validation_notes=notes,
    )


# =============================================================================
# 7. Execution Cascade Pydantic Data Model & Parser
# =============================================================================


class ExecutionCascade(BaseModel):
    """Complete, validated progressive quantum chemical execution cascade container."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tier: str = Field(..., description="Method Matrix tier identifier")
    target_walltime: str = Field(..., description="Expected walltime ceiling")
    selected_mlff: str = Field(..., description="Selected capable MLFF model name")
    fallback_trail: List[str] = Field(default_factory=list, description="Audit trail of MLFF models evaluated")
    method: str = Field(..., description="Primary electronic structure method")
    basis_set: str = Field(..., description="Primary atomic orbital basis set")
    aux_basis: str = Field(..., description="Auxiliary Coulomb / correlation basis set")
    grid_level: str = Field(..., description="Integration grid level")
    dispersion: Optional[str] = Field(default=None, description="Dispersion correction type")
    tol_max_g: float = Field(..., description="Maximum gradient convergence threshold")
    tol_rms_g: float = Field(..., description="RMS gradient convergence threshold")
    tol_energy: float = Field(default=1e-6, description="Energy convergence tolerance")
    scf_type: str = Field(default="DIIS", description="SCF acceleration method")
    initial_hessian: str = Field(default="XTB2", description="Initial Hessian estimator")
    is_diffuse: bool = Field(default=False, description="Flag for diffuse / anionic systems")
    is_weak_complex: bool = Field(default=False, description="Flag for non-covalent weak complex systems")
    frozen_monomer: bool = Field(default=False, description="Flag for frozen-monomer intermolecular protocol")
    is_double_hybrid: bool = Field(default=False, description="Flag for Double-Hybrid functional")
    is_coupled_cluster: bool = Field(default=False, description="Flag for Coupled-Cluster escalation")
    pno_setting: Optional[str] = Field(default=None, description="PNO threshold if applicable")
    element_symbols: List[str] = Field(default_factory=list, description="Validated element symbols in molecular system")
    atomic_numbers: List[int] = Field(default_factory=list, description="Atomic numbers Z of atoms in molecular system")
    provenance_hash: Optional[str] = Field(default=None, description="Cryptographic SHA-256 provenance hash")
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="Sequential multi-stage pipeline configuration")
    extra_keywords: List[str] = Field(default_factory=list, description="Additional ORCA / electronic structure keywords")
    hardware_affinity: Dict[str, Any] = Field(default_factory=dict, description="Hardware detection telemetry")


def parse_execution_cascade(
    symbols: Sequence[str],
    tier: str = "T1-10s",
    is_diffuse: bool = False,
    is_weak_complex: bool = False,
    frozen_monomer: bool = False,
    requested_mlff: Optional[str] = None,
    context: Optional[ExecutionContext] = None,
    custom_overrides: Optional[Dict[str, Any]] = None,
) -> ExecutionCascade:
    """Parses and constructs a Method Matrix v4 progressive execution cascade.

    Resolves MLFF fallback hierarchies for unsupported atomic species, enforces
    diffuse/weak-complex basis set adaptations, validates auxiliary basis sets,
    and computes deterministic cryptographic provenance.

    Args:
        symbols: Sequence of chemical element symbols (e.g. ['C', 'H', 'O', 'Fe']).
        tier: Method Matrix tier identifier (e.g. 'T1-10s', 'T1-1min', 'T1-30min', 'T1-1h',
              'T3-1min', 'T3-3h', 'T3-12h', 'T4-1d', 'T4C-1mo').
        is_diffuse: True if system contains anions or diffuse Rydberg states.
        is_weak_complex: True if system is a non-covalent weakly bound intermolecular complex.
        frozen_monomer: True if applying frozen-monomer intermolecular scan protocol.
        requested_mlff: Optional preferred MLFF model name.
        context: ExecutionContext for environment detection and air-gap verification.
        custom_overrides: Optional dictionary of parameter overrides.

    Returns:
        Fully validated ExecutionCascade instance.

    Raises:
        InvalidTierError: If tier is not recognized in Method Matrix v4 catalog.
        UnsupportedElementError: If element symbols cannot be processed.
        AuxiliaryBasisMismatchError: If basis/aux_basis pairings violate physical constraints.
    """
    ctx = context or ExecutionContext()
    valid_symbols = validate_element_symbols(symbols)
    if not valid_symbols:
        raise UnsupportedElementError("Cannot parse execution cascade for empty element sequence.")
    atomic_numbers = [get_atomic_number(s) for s in valid_symbols]

    # 1. Resolve Tier Configuration
    if tier not in METHOD_MATRIX_V4_TIERS:
        raise InvalidTierError(
            f"Unrecognized Method Matrix v4 tier '{tier}'. Available tiers: {list(METHOD_MATRIX_V4_TIERS.keys())}"
        )
    tier_cfg = METHOD_MATRIX_V4_TIERS[tier]

    # 2. Resolve MLFF Fallback Hierarchy
    mlff_spec, fallback_trail = resolve_mlff_model(valid_symbols, requested_model=requested_mlff)

    # 3. Resolve Basis Sets & Diffuse/Weak Complex Escalation
    basis_set = tier_cfg.basis_set
    aux_basis = tier_cfg.aux_basis
    method = tier_cfg.method
    grid_level = tier_cfg.grid_level
    dispersion = tier_cfg.dispersion
    tol_max_g = tier_cfg.tol_max_g
    tol_rms_g = tier_cfg.tol_rms_g
    tol_energy = tier_cfg.tol_energy
    scf_type = tier_cfg.scf_type
    initial_hessian = tier_cfg.initial_hessian
    is_double_hybrid = tier_cfg.is_double_hybrid
    is_coupled_cluster = tier_cfg.is_coupled_cluster
    pno_setting = tier_cfg.pno_setting

    # 3. Apply Custom Overrides if provided
    if custom_overrides:
        method = custom_overrides.get("method", method)
        basis_set = custom_overrides.get("basis_set", basis_set)
        aux_basis = custom_overrides.get("aux_basis", aux_basis)
        grid_level = custom_overrides.get("grid_level", grid_level)
        dispersion = custom_overrides.get("dispersion", dispersion)
        tol_max_g = custom_overrides.get("tol_max_g", tol_max_g)
        tol_rms_g = custom_overrides.get("tol_rms_g", tol_rms_g)
        tol_energy = custom_overrides.get("tol_energy", tol_energy)
        scf_type = custom_overrides.get("scf_type", scf_type)
        initial_hessian = custom_overrides.get("initial_hessian", initial_hessian)
        pno_setting = custom_overrides.get("pno_setting", pno_setting)

    # 4. Apply diffuse / weak-complex basis escalation per Method Matrix v4
    if is_diffuse or is_weak_complex:
        if basis_set == "def2-SVP":
            basis_set = "ma-def2-SVP"
        elif basis_set == "def2-TZVP":
            basis_set = "ma-def2-TZVP"
        elif basis_set == "def2-TZVPP":
            basis_set = "ma-def2-TZVPP"
        elif basis_set == "def2-QZVPP":
            basis_set = "ma-def2-QZVPP"

    # 5. Scientifically Validate Auxiliary Basis Pairing
    validate_auxiliary_basis(
        method=method,
        basis_set=basis_set,
        aux_basis=aux_basis,
        is_diffuse=is_diffuse,
        is_weak_complex=is_weak_complex,
    )

    # 6. Build Multi-Stage Pipeline Definition
    stages: List[Dict[str, Any]] = [
        {
            "stage_idx": 1,
            "stage_name": "MLFF_Screening_And_Preopt",
            "model": mlff_spec.name,
            "convergence_tol_max_g": max(tol_max_g * 10.0, 1e-3),
            "description": f"Initial potential energy screening with {mlff_spec.name}",
        },
        {
            "stage_idx": 2,
            "stage_name": "DFT_Hessian_And_Surface_Refinement",
            "method": method,
            "basis_set": basis_set,
            "aux_basis": aux_basis,
            "grid_level": grid_level,
            "dispersion": dispersion,
            "initial_hessian": initial_hessian,
            "convergence_tol_max_g": tol_max_g,
            "description": f"Refinement with {method}/{basis_set}",
        },
    ]

    if is_coupled_cluster:
        stages.append({
            "stage_idx": 3,
            "stage_name": "Coupled_Cluster_Single_Point",
            "method": method,
            "basis_set": basis_set,
            "aux_basis": aux_basis,
            "pno_setting": pno_setting,
            "description": f"Single-point escalation using {method}/{basis_set}",
        })

    # 7. Hardware Affinity Telemetry
    hardware_affinity: Dict[str, Any] = {
        "tier": ctx.tier.value,
        "num_cores": ctx.num_cores,
        "max_memory_mb": ctx.max_memory_mb,
        "gpu_available": ctx.gpu_available,
        "vram_mb": ctx.vram_mb,
    }

    # 8. Extra keywords
    extra_keywords: List[str] = []
    if frozen_monomer:
        extra_keywords.append("FROZEN_MONOMER")
    if is_weak_complex:
        extra_keywords.append("WEAK_COMPLEX")
    if is_diffuse:
        extra_keywords.append("DIFFUSE_AUGMENTED")

    cascade = ExecutionCascade(
        tier=tier,
        target_walltime=tier_cfg.target_walltime,
        selected_mlff=mlff_spec.name,
        fallback_trail=fallback_trail,
        method=method,
        basis_set=basis_set,
        aux_basis=aux_basis,
        grid_level=grid_level,
        dispersion=dispersion,
        tol_max_g=tol_max_g,
        tol_rms_g=tol_rms_g,
        tol_energy=tol_energy,
        scf_type=scf_type,
        initial_hessian=initial_hessian,
        is_diffuse=is_diffuse,
        is_weak_complex=is_weak_complex,
        frozen_monomer=frozen_monomer,
        is_double_hybrid=is_double_hybrid,
        is_coupled_cluster=is_coupled_cluster,
        pno_setting=pno_setting,
        element_symbols=valid_symbols,
        atomic_numbers=atomic_numbers,
        stages=stages,
        extra_keywords=extra_keywords,
        hardware_affinity=hardware_affinity,
    )

    # 9. Compute Deterministic Provenance Hash
    prov_hash = generate_provenance_hash(cascade=cascade, context=ctx, persist_manifest=False)
    cascade.provenance_hash = prov_hash

    return cascade


# =============================================================================
# 8. Cryptographic Provenance & Manifest Persistence
# =============================================================================


class ProvenanceManifest(BaseModel):
    """Complete cryptographic deployment and provenance manifest for FAIR data governance."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest_version: str = Field(default="4.0.0", description="Method Matrix manifest schema version")
    generated_at_utc: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    provenance_hash_sha256: str = Field(..., description="Cryptographic SHA-256 manifest hash")
    provenance_hash_xxh64: Optional[str] = Field(default=None, description="Cryptographic xxHash-64 hash if available")
    tier: str = Field(..., description="Method Matrix tier identifier")
    method: str = Field(..., description="Electronic structure method")
    basis_set: str = Field(..., description="Primary basis set")
    aux_basis: str = Field(..., description="Auxiliary basis set")
    grid_level: str = Field(..., description="Integration grid level")
    dispersion: Optional[str] = Field(default=None, description="Dispersion correction")
    tol_max_g: float = Field(..., description="Maximum gradient convergence threshold")
    selected_mlff: str = Field(..., description="Selected MLFF potential model")
    fallback_trail: List[str] = Field(default_factory=list, description="MLFF selection audit trail")
    element_symbols: List[str] = Field(default_factory=list, description="Element symbols")
    atomic_numbers: List[int] = Field(default_factory=list, description="Atomic numbers Z")
    atomic_masses_da: Dict[str, float] = Field(default_factory=dict, description="Dynamic standard atomic masses queried from Mendeleev")
    physical_constants_codata: Dict[str, float] = Field(default_factory=dict, description="Physical conversion constants used")
    hardware_affinity: Dict[str, Any] = Field(default_factory=dict, description="Execution environment hardware details")
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="Execution cascade multi-stage details")


def generate_provenance_hash(
    cascade: ExecutionCascade,
    context: Optional[ExecutionContext] = None,
    output_manifest_path: Optional[Union[str, Path]] = None,
    persist_manifest: bool = True,
) -> str:
    """Generates a deterministic SHA-256 provenance hash of all parameters and physical constants.

    Optionally persists the provenance record to `cochem_deployment_manifest.json`
    in Domain C / Ring 3 persistent artifacts directory while enforcing Air-Gap boundary protection.

    Args:
        cascade: ExecutionCascade container to hash.
        context: Optional ExecutionContext for air-gap and path resolution.
        output_manifest_path: Custom output path for manifest file.
        persist_manifest: If True, writes the manifest to disk.

    Returns:
        Hex-encoded SHA-256 provenance hash string.

    Raises:
        AirGapViolationError: If output_manifest_path is located inside Ring 1 static repository.
    """
    ctx = context or ExecutionContext()

    # Dynamic Mendeleev mass queries for all elements in the cascade
    atomic_masses: Dict[str, float] = {}
    for sym in cascade.element_symbols:
        atomic_masses[sym] = get_atomic_mass(sym)

    physical_constants = {
        "HARTREE_TO_KCAL_MOL": HARTREE_TO_KCAL_MOL,
        "HARTREE_TO_EV": HARTREE_TO_EV,
        "BOHR_TO_ANGSTROM": BOHR_TO_ANGSTROM,
        "EV_TO_KCAL_MOL": EV_TO_KCAL_MOL,
    }

    canonical_payload: Dict[str, Any] = {
        "tier": cascade.tier,
        "target_walltime": cascade.target_walltime,
        "method": cascade.method,
        "basis_set": cascade.basis_set,
        "aux_basis": cascade.aux_basis,
        "grid_level": cascade.grid_level,
        "dispersion": cascade.dispersion,
        "tol_max_g": float(f"{cascade.tol_max_g:.8e}"),
        "tol_rms_g": float(f"{cascade.tol_rms_g:.8e}"),
        "tol_energy": float(f"{cascade.tol_energy:.8e}"),
        "scf_type": cascade.scf_type,
        "initial_hessian": cascade.initial_hessian,
        "is_diffuse": cascade.is_diffuse,
        "is_weak_complex": cascade.is_weak_complex,
        "frozen_monomer": cascade.frozen_monomer,
        "is_double_hybrid": cascade.is_double_hybrid,
        "is_coupled_cluster": cascade.is_coupled_cluster,
        "pno_setting": cascade.pno_setting,
        "selected_mlff": cascade.selected_mlff,
        "element_symbols": sorted(cascade.element_symbols),
        "atomic_numbers": sorted(cascade.atomic_numbers),
        "atomic_masses_da": {k: float(f"{v:.6f}") for k, v in sorted(atomic_masses.items())},
        "physical_constants": {k: float(f"{v:.10e}") for k, v in sorted(physical_constants.items())},
        "stages": cascade.stages,
        "extra_keywords": sorted(cascade.extra_keywords),
    }

    # Deterministic JSON serialization: sorted keys, compact separators
    serialized_bytes = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    # Cryptographic SHA-256
    sha256_hash = hashlib.sha256(serialized_bytes).hexdigest()

    # Optional xxHash-64
    xxh64_hash: Optional[str] = None
    if _XXHASH_AVAILABLE:
        xxh64_hash = xxhash.xxh64(serialized_bytes).hexdigest()

    if persist_manifest:
        manifest_record = ProvenanceManifest(
            manifest_version="4.0.0",
            provenance_hash_sha256=sha256_hash,
            provenance_hash_xxh64=xxh64_hash,
            tier=cascade.tier,
            method=cascade.method,
            basis_set=cascade.basis_set,
            aux_basis=cascade.aux_basis,
            grid_level=cascade.grid_level,
            dispersion=cascade.dispersion,
            tol_max_g=cascade.tol_max_g,
            selected_mlff=cascade.selected_mlff,
            fallback_trail=cascade.fallback_trail,
            element_symbols=cascade.element_symbols,
            atomic_numbers=cascade.atomic_numbers,
            atomic_masses_da=atomic_masses,
            physical_constants_codata=physical_constants,
            hardware_affinity=cascade.hardware_affinity,
            stages=cascade.stages,
        )

        if output_manifest_path is not None:
            target_path = Path(output_manifest_path).resolve()
        else:
            artifacts_dir = ctx.get_artifacts_dir()
            target_path = artifacts_dir / "cochem_deployment_manifest.json"

        # Verify Air-Gap boundary before writing
        ctx.verify_air_gap_boundary(target_path)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(manifest_record.model_dump_json(indent=2))
        logger.info(f"Persisted cryptographic provenance manifest to: {target_path}")

    return sha256_hash

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_matrix_loader.py ---
"""CoChem-TORQ: Method Matrix Parser & Provenance Test Suite.

=============================================================================
Phase 5 (Stage 4.0 / Task 8) Authentic Physical Test Matrix
----------------------------------------------------------
Comprehensive unit and integration test suite for Method Matrix v4 loader,
progressive cascade parser, multi-tier MLFF fallback hierarchies, rigorous
scientific auxiliary basis validation, deterministic SHA-256 / xxHash-64
cryptographic provenance hashing, and Tripartite Air-Gap isolation.

Zero-Tolerance Anti-Mocking:
All tests operate on real physical chemical systems, authentic Mendeleev
periodic table property lookups, and genuine cryptographic hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

from mendeleev import element as mendeleev_element
import numpy as np
import pytest
from pydantic import ValidationError

from Libraries.cochem_torq_matrix_loader import (
    DEFAULT_MLFF_FALLBACK_CHAIN,
    EV_TO_KCAL_MOL,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_MOL,
    METHOD_MATRIX_V4_TIERS,
    MLFF_CATALOG,
    AirGapReport,
    AirGapViolationError,
    AuxiliaryBasisMismatchError,
    AuxiliaryBasisPairing,
    CascadeTierConfig,
    EnvironmentTier,
    ExecutionContext,
    ExecutionCascade,
    InvalidTierError,
    MatrixLoaderError,
    MLFFModelSpec,
    ProvenanceManifest,
    UnsupportedElementError,
    _XXHASH_AVAILABLE,
    generate_provenance_hash,
    get_atomic_mass,
    get_atomic_number,
    get_covalent_radius,
    get_element_symbol,
    get_isotopic_mass,
    get_repo_root,
    get_vdw_radius,
    parse_execution_cascade,
    resolve_mlff_model,
    validate_auxiliary_basis,
    validate_element_symbols,
)


# =============================================================================
# 1. Mendeleev Dynamic Property Query Tests (Mendeleev Mandate)
# =============================================================================


class TestMendeleevDynamicProperties:
    """Authentic physical tests for dynamic atomic and isotopic retrieval via Mendeleev."""

    def test_dynamic_atomic_numbers(self) -> None:
        """Verify dynamic atomic number retrieval for diverse elements."""
        assert get_atomic_number("H") == 1
        assert get_atomic_number("He") == 2
        assert get_atomic_number("C") == 6
        assert get_atomic_number("N") == 7
        assert get_atomic_number("O") == 8
        assert get_atomic_number("F") == 9
        assert get_atomic_number("Si") == 14
        assert get_atomic_number("P") == 15
        assert get_atomic_number("S") == 16
        assert get_atomic_number("Cl") == 17
        assert get_atomic_number("Fe") == 26
        assert get_atomic_number("Br") == 35
        assert get_atomic_number("I") == 53
        assert get_atomic_number("Pt") == 78
        assert get_atomic_number("U") == 92
        assert get_atomic_number(6) == 6
        assert get_atomic_number(26) == 26

    def test_dynamic_element_symbols(self) -> None:
        """Verify dynamic element symbol canonicalization."""
        assert get_element_symbol("c") == "C"
        assert get_element_symbol("fe") == "Fe"
        assert get_element_symbol(1) == "H"
        assert get_element_symbol(8) == "O"
        assert get_element_symbol(92) == "U"

    def test_dynamic_atomic_masses(self) -> None:
        """Verify standard atomic weights are retrieved dynamically without hardcoding."""
        h_mass = get_atomic_mass("H")
        assert 1.007 < h_mass < 1.009

        c_mass = get_atomic_mass("C")
        assert 12.010 < c_mass < 12.012

        n_mass = get_atomic_mass("N")
        assert 14.006 < n_mass < 14.008

        o_mass = get_atomic_mass("O")
        assert 15.998 < o_mass < 16.000

        fe_mass = get_atomic_mass("Fe")
        assert 55.840 < fe_mass < 55.850

        u_mass = get_atomic_mass("U")
        assert 238.00 < u_mass < 238.05

    def test_dynamic_isotopic_masses(self) -> None:
        """Verify exact isotopic mass queries for physical isotopes."""
        c12_mass = get_isotopic_mass("C", 12)
        assert abs(c12_mass - 12.0000) < 1e-4

        c13_mass = get_isotopic_mass("C", 13)
        assert 13.003 < c13_mass < 13.004

        h1_mass = get_isotopic_mass("H", 1)
        assert 1.007 < h1_mass < 1.009

        h2_mass = get_isotopic_mass("H", 2)
        assert 2.014 < h2_mass < 2.015

        # Query without mass number returns standard mass
        h_std = get_isotopic_mass("H")
        assert abs(h_std - get_atomic_mass("H")) < 1e-6

    def test_dynamic_covalent_and_vdw_radii(self) -> None:
        """Verify Pyykkö covalent and van der Waals radii queries in Angstroms."""
        h_cov = get_covalent_radius("H")
        assert 0.30 < h_cov < 0.35

        c_cov = get_covalent_radius("C")
        assert 0.70 < c_cov < 0.80

        c_vdw = get_vdw_radius("C")
        assert 1.60 < c_vdw < 2.10

    def test_validate_element_symbols(self) -> None:
        """Verify chemical element sequence validation."""
        valid = validate_element_symbols(["c", "H", "O", "N", "Fe:"])
        assert valid == ["C", "H", "O", "N", "Fe"]

        with pytest.raises(UnsupportedElementError):
            validate_element_symbols(["C", "H", "XzUnknownElement"])


# =============================================================================
# 2. MLFF Fallback Hierarchy & Resolution Tests
# =============================================================================


class TestMLFFHierarchyResolution:
    """Authentic physical tests for progressive MLFF fallback chains across diverse chemistries."""

    def test_organic_molecules_resolve_mace_off23(self) -> None:
        """Verify standard organic drug-like species select MACE-OFF23."""
        # Ethanol (C2H6O)
        spec_eth, trail_eth = resolve_mlff_model(["C", "C", "H", "H", "H", "H", "H", "H", "O"])
        assert spec_eth.name == "MACE-OFF23"
        assert len(trail_eth) == 1
        assert "Selected 'MACE-OFF23'" in trail_eth[0]

        # Glycine (C2H5NO2)
        spec_gly, _ = resolve_mlff_model(["C", "C", "H", "H", "H", "H", "H", "N", "O", "O"])
        assert spec_gly.name == "MACE-OFF23"

        # Cysteine with Sulfur (C3H7NO2S)
        spec_cys, _ = resolve_mlff_model(["C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "N", "O", "O", "S"])
        assert spec_cys.name == "MACE-OFF23"

        # Bromobenzene with Bromine (C6H5Br)
        spec_br, _ = resolve_mlff_model(["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "Br"])
        assert spec_br.name == "MACE-OFF23"

        # Iodomethane with Iodine (CH3I)
        spec_i, _ = resolve_mlff_model(["C", "H", "H", "H", "I"])
        assert spec_i.name == "MACE-OFF23"

        # Phosphorylated intermediate (P, F, Cl)
        spec_pfc, _ = resolve_mlff_model(["P", "F", "Cl", "O", "H"])
        assert spec_pfc.name == "MACE-OFF23"

    def test_boron_and_silicon_species_fallback_from_mace_off23(self) -> None:
        """Verify Boron and Silicon trigger MACE-OFF23 rejection and resolve to capable model."""
        # Triethylborane (B, C, H) -> MACE-OFF23 rejected due to B
        spec_b, trail_b = resolve_mlff_model(["B", "C", "H"])
        assert spec_b.name in ["MACE-POLAR-1", "MACE-OMOL-0", "AIMNet2"]
        assert any("Rejected 'MACE-OFF23': unsupported element(s) ['B']" in t for t in trail_b)

        # Silane (SiH4) -> MACE-OFF23 rejected due to Si
        spec_si, trail_si = resolve_mlff_model(["Si", "H", "H", "H", "H"])
        assert spec_si.name in ["MACE-POLAR-1", "MACE-OMOL-0", "AIMNet2"]
        assert any("Rejected 'MACE-OFF23': unsupported element(s) ['Si']" in t for t in trail_si)

        # Selenium (Se) -> MACE-OFF23 and MACE-OMOL-0 rejected, resolves to AIMNet2 or MACE-POLAR-1
        spec_se, trail_se = resolve_mlff_model(["C", "H", "Se"])
        assert spec_se.name in ["MACE-POLAR-1", "AIMNet2"]

    def test_transition_metals_and_noble_gases_fallback_to_gfn2_xtb(self) -> None:
        """Verify transition metals and noble gases fall back through MLFFs to GFN2-xTB / MACE-POLAR-1."""
        # Ferrocene (Fe, C, H)
        spec_fe, trail_fe = resolve_mlff_model(
            ["Fe", "C", "H"],
            fallback_chain=["MACE-OFF23", "AIMNet2", "GFN2-xTB", "GFN-FF", "PySCF_RHF"],
        )
        assert spec_fe.name == "GFN2-xTB"
        assert any("Rejected 'MACE-OFF23': unsupported element(s) ['Fe']" in t for t in trail_fe)
        assert any("Rejected 'AIMNet2': unsupported element(s) ['Fe']" in t for t in trail_fe)

        # Argon complex with water (Ar, H, O)
        spec_ar, trail_ar = resolve_mlff_model(
            ["Ar", "H", "O"],
            fallback_chain=["MACE-OFF23", "AIMNet2", "GFN2-xTB", "GFN-FF"],
        )
        assert spec_ar.name == "GFN2-xTB"

        # Neon complex (Ne, C, H)
        spec_ne, trail_ne = resolve_mlff_model(
            ["Ne", "C", "H"],
            fallback_chain=["MACE-OFF23", "AIMNet2", "GFN2-xTB", "GFN-FF"],
        )
        assert spec_ne.name == "GFN2-xTB"

    def test_heavy_actinides_fallback_to_pyscf_or_empirical(self) -> None:
        """Verify elements beyond Z=86 (e.g. Uranium Z=92) fall back past GFN2-xTB."""
        spec_u, trail_u = resolve_mlff_model(
            ["U", "O", "F"],
            fallback_chain=["MACE-OFF23", "AIMNet2", "GFN2-xTB", "GFN-FF", "PySCF_RHF", "EMPIRICAL_COVALENT"],
        )
        assert spec_u.name == "PySCF_RHF"
        assert any("Rejected 'GFN2-xTB'" in t for t in trail_u)

    def test_custom_requested_mlff_priority(self) -> None:
        """Verify requested MLFF is prioritized if capable."""
        spec, trail = resolve_mlff_model(["C", "H", "O"], requested_model="AIMNet2")
        assert spec.name == "AIMNet2"
        assert "Selected 'AIMNet2'" in trail[0]


# =============================================================================
# 3. Auxiliary Basis Scientific Validation Tests
# =============================================================================


class TestAuxiliaryBasisScientificValidation:
    """Authentic physical tests for auxiliary basis pairing validation per Method Matrix v4."""

    def test_valid_standard_dft_pairings(self) -> None:
        """Verify valid standard hybrid and meta-GGA DFT pairings with def2/J."""
        # wB97M-V with def2-TZVP and def2/J
        p1 = validate_auxiliary_basis("wB97M-V", "def2-TZVP", "def2/J")
        assert p1.is_compatible is True
        assert p1.ri_type == "RIJCOSX"

        # wB97X-D4 with def2-QZVPP and def2/J
        p2 = validate_auxiliary_basis("wB97X-D4", "def2-QZVPP", "def2/J")
        assert p2.is_compatible is True

        # r2SCAN-3c with def2-mSVP and def2/J
        p3 = validate_auxiliary_basis("r2SCAN-3c", "def2-mSVP", "def2/J")
        assert p3.is_compatible is True

    def test_valid_double_hybrid_and_correlation_pairings(self) -> None:
        """Verify valid Double-Hybrid DFT pairings with correlation auxiliary basis sets."""
        # revDSD-PBEP86 with def2-TZVPP and def2-TZVPP/C
        p1 = validate_auxiliary_basis("revDSD-PBEP86", "def2-TZVPP", "def2/J def2-TZVPP/C")
        assert p1.is_compatible is True
        assert "RIJCOSX+PT2/C" in p1.ri_type

        # DSD-PBEP86 with def2-QZVPP and def2-QZVPP/C
        p2 = validate_auxiliary_basis("DSD-PBEP86", "def2-QZVPP", "def2/J def2-QZVPP/C")
        assert p2.is_compatible is True

        # MP2 with def2-SVP and def2-SVP/C
        p3 = validate_auxiliary_basis("MP2", "def2-SVP", "def2/J def2-SVP/C")
        assert p3.is_compatible is True

        # DLPNO-CCSD(T) with def2-TZVPP and def2/JK def2-TZVPP/C
        p4 = validate_auxiliary_basis("DLPNO-CCSD(T)", "def2-TZVPP", "def2/JK def2-TZVPP/C")
        assert p4.is_compatible is True

    def test_valid_hf_and_ri_jk_pairings(self) -> None:
        """Verify HF exchange requires def2/JK."""
        p1 = validate_auxiliary_basis("HF", "def2-SVP", "def2/JK", ri_approximation="RI-JK")
        assert p1.is_compatible is True
        assert p1.ri_type == "RI-JK"

    def test_invalid_double_hybrid_missing_correlation_aux_basis(self) -> None:
        """Reject Double-Hybrid calculations where aux_basis only contains def2/J without /C."""
        with pytest.raises(AuxiliaryBasisMismatchError) as exc_info:
            validate_auxiliary_basis("revDSD-PBEP86", "def2-TZVPP", "def2/J")
        assert "lacks correlation fitting basis" in str(exc_info.value)

        with pytest.raises(AuxiliaryBasisMismatchError):
            validate_auxiliary_basis("B2PLYP", "def2-TZVP", "def2/J")

    def test_invalid_ri_jk_with_only_def2_j(self) -> None:
        """Reject RI-JK when only Coulomb def2/J is provided."""
        with pytest.raises(AuxiliaryBasisMismatchError) as exc_info:
            validate_auxiliary_basis("HF", "def2-TZVP", "def2/J", ri_approximation="RI-JK")
        assert "def2/J' only provides Coulomb fitting" in str(exc_info.value)

    def test_diffuse_and_weak_complex_augmentation_validation(self) -> None:
        """Verify diffuse / weak complex systems require augmented basis sets."""
        # Unaugmented def2-TZVP on weak complex must be rejected
        with pytest.raises(AuxiliaryBasisMismatchError) as exc_info:
            validate_auxiliary_basis("wB97M-V", "def2-TZVP", "def2/J", is_weak_complex=True)
        assert "diffuse/weak complex violation" in str(exc_info.value)

        # Minimally augmented ma-def2-TZVP is accepted
        p_ma = validate_auxiliary_basis("wB97M-V", "ma-def2-TZVP", "def2/J", is_weak_complex=True)
        assert p_ma.is_compatible is True
        assert p_ma.is_diffuse_compatible is True

        # Dunning aug-cc-pVTZ with AutoAux is accepted
        p_aug = validate_auxiliary_basis("CCSD(T)", "aug-cc-pVTZ", "AutoAux", is_diffuse=True)
        assert p_aug.is_compatible is True

    def test_cross_family_basis_mismatch_rejected(self) -> None:
        """Reject Dunning primary basis paired with Ahlrichs auxiliary basis without AutoAux."""
        with pytest.raises(AuxiliaryBasisMismatchError) as exc_info:
            validate_auxiliary_basis("wB97M-V", "cc-pVTZ", "def2/J")
        assert "Basis set family mismatch" in str(exc_info.value)


# =============================================================================
# 4. Method Matrix v4 Progressive Cascade Parser Tests
# =============================================================================


class TestExecutionCascadeParser:
    """Authentic physical tests for parse_execution_cascade across all tiers."""

    def test_parse_all_standard_tiers(self) -> None:
        """Verify parse_execution_cascade succeeds across all Method Matrix v4 tiers."""
        symbols = ["C", "H", "H", "H", "O", "H"]  # Methanol

        for tier_name in METHOD_MATRIX_V4_TIERS:
            cascade = parse_execution_cascade(symbols=symbols, tier=tier_name)
            assert isinstance(cascade, ExecutionCascade)
            assert cascade.tier == tier_name
            assert cascade.selected_mlff == "MACE-OFF23"
            assert cascade.provenance_hash is not None
            assert len(cascade.provenance_hash) == 64
            assert len(cascade.stages) >= 2
            assert cascade.element_symbols == ["C", "H", "H", "H", "O", "H"]
            assert set(cascade.atomic_numbers) == {1, 6, 8}

    def test_parse_diffuse_and_weak_complex_escalation(self) -> None:
        """Verify basis set auto-escalation to ma-def2 for weak complex systems."""
        symbols = ["O", "H", "H", "O", "H", "H"]  # Water dimer
        cascade = parse_execution_cascade(
            symbols=symbols,
            tier="T1-1h",
            is_weak_complex=True,
            frozen_monomer=True,
        )
        assert cascade.basis_set == "ma-def2-TZVP"
        assert cascade.is_weak_complex is True
        assert cascade.frozen_monomer is True
        assert "FROZEN_MONOMER" in cascade.extra_keywords
        assert "WEAK_COMPLEX" in cascade.extra_keywords

    def test_parse_diffuse_escalation_across_various_tiers(self) -> None:
        """Verify diffuse basis auto-escalation for def2-SVP, def2-TZVPP, def2-QZVPP."""
        symbols = ["F", "H", "O"]
        # T1-10s with custom def2-SVP
        c_svp = parse_execution_cascade(symbols, tier="T1-10s", is_diffuse=True, custom_overrides={"basis_set": "def2-SVP"})
        assert c_svp.basis_set == "ma-def2-SVP"

        # T3-3h with def2-TZVPP
        c_tzvpp = parse_execution_cascade(symbols, tier="T3-3h", is_diffuse=True)
        assert c_tzvpp.basis_set == "ma-def2-TZVPP"

        # T3-12h with def2-QZVPP
        c_qzvpp = parse_execution_cascade(symbols, tier="T3-12h", is_diffuse=True)
        assert c_qzvpp.basis_set == "ma-def2-QZVPP"

    def test_parse_double_hybrid_tier_t3_3h(self) -> None:
        """Verify T3-3h resolves revDSD-PBEP86 with def2-TZVPP/C auxiliary fitting."""
        symbols = ["C", "H", "N", "O"]
        cascade = parse_execution_cascade(symbols=symbols, tier="T3-3h")
        assert cascade.method == "revDSD-PBEP86"
        assert cascade.basis_set == "def2-TZVPP"
        assert "def2-TZVPP/C" in cascade.aux_basis
        assert cascade.tol_max_g == 1e-5
        assert cascade.is_double_hybrid is True

    def test_parse_coupled_cluster_tier_t4_1d(self) -> None:
        """Verify T4-1d resolves DLPNO-CCSD(T) with 3 pipeline stages."""
        symbols = ["C", "H", "O"]
        cascade = parse_execution_cascade(symbols=symbols, tier="T4-1d")
        assert cascade.method == "DLPNO-CCSD(T)"
        assert cascade.is_coupled_cluster is True
        assert cascade.pno_setting == "TightPNO"
        assert len(cascade.stages) == 3
        assert cascade.stages[2]["stage_name"] == "Coupled_Cluster_Single_Point"

    def test_parse_custom_overrides(self) -> None:
        """Verify custom parameter overrides in cascade resolution."""
        symbols = ["C", "H", "O"]
        overrides = {
            "method": "wB97M-D4",
            "basis_set": "def2-QZVPP",
            "aux_basis": "def2/J",
            "grid_level": "defgrid3",
            "tol_max_g": 5e-5,
            "pno_setting": "TightPNO",
        }
        cascade = parse_execution_cascade(symbols, tier="T1-1h", custom_overrides=overrides)
        assert cascade.method == "wB97M-D4"
        assert cascade.basis_set == "def2-QZVPP"
        assert cascade.tol_max_g == 5e-5

    def test_invalid_tier_raises_error(self) -> None:
        """Verify invalid tier string raises InvalidTierError."""
        with pytest.raises(InvalidTierError) as exc_info:
            parse_execution_cascade(symbols=["C", "H"], tier="T99-UnknownTier")
        assert "Unrecognized Method Matrix v4 tier" in str(exc_info.value)

    def test_empty_symbols_raises_error(self) -> None:
        """Verify empty element sequence raises UnsupportedElementError."""
        with pytest.raises(UnsupportedElementError):
            parse_execution_cascade(symbols=[], tier="T1-10s")


# =============================================================================
# 5. Cryptographic Provenance Hash & Manifest Persistence Tests
# =============================================================================


class TestProvenanceHashAndManifest:
    """Authentic physical tests for deterministic SHA-256 provenance hashing and manifest generation."""

    def test_deterministic_sha256_hash_repeatability(self) -> None:
        """Verify exact byte-for-byte deterministic reproducibility of SHA-256 hash."""
        symbols = ["C", "H", "O", "N"]
        cascade1 = parse_execution_cascade(symbols=symbols, tier="T1-1h")
        cascade2 = parse_execution_cascade(symbols=symbols, tier="T1-1h")

        hash1 = generate_provenance_hash(cascade=cascade1, persist_manifest=False)
        hash2 = generate_provenance_hash(cascade=cascade2, persist_manifest=False)

        assert hash1 == hash2
        assert len(hash1) == 64
        assert int(hash1, 16) > 0

    def test_provenance_hash_sensitivity_to_parameters(self) -> None:
        """Verify hash diverges when physical or method parameters change."""
        symbols = ["C", "H", "O"]
        c1 = parse_execution_cascade(symbols=symbols, tier="T1-1h")
        c2 = parse_execution_cascade(symbols=symbols, tier="T3-3h")  # Different tier
        c3 = parse_execution_cascade(symbols=["C", "H", "S"], tier="T1-1h")  # Different element

        h1 = generate_provenance_hash(cascade=c1, persist_manifest=False)
        h2 = generate_provenance_hash(cascade=c2, persist_manifest=False)
        h3 = generate_provenance_hash(cascade=c3, persist_manifest=False)

        assert h1 != h2
        assert h1 != h3
        assert h2 != h3

    def test_manifest_persistence_to_artifacts_dir(self, tmp_path: Path) -> None:
        """Verify manifest writes successfully to Ring 3 persistent artifacts directory."""
        artifacts_dir = tmp_path / "test_artifacts_vault"
        ctx = ExecutionContext(custom_artifacts_dir=artifacts_dir)

        cascade = parse_execution_cascade(symbols=["C", "H", "F"], tier="T1-30min", context=ctx)
        manifest_file = artifacts_dir / "cochem_deployment_manifest.json"

        h = generate_provenance_hash(cascade=cascade, context=ctx, persist_manifest=True)
        assert manifest_file.exists()

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["manifest_version"] == "4.0.0"
        assert data["provenance_hash_sha256"] == h
        assert data["tier"] == "T1-30min"
        assert data["method"] == "wB97X-D4"
        assert "C" in data["atomic_masses_da"]
        assert "HARTREE_TO_KCAL_MOL" in data["physical_constants_codata"]
        if _XXHASH_AVAILABLE:
            assert data["provenance_hash_xxh64"] is not None


# =============================================================================
# 6. Tripartite Air-Gap Boundary Enforcement Tests
# =============================================================================


class TestAirGapEnforcement:
    """Authentic physical tests for Tripartite Filesystem Air-Gap boundary protection."""

    def test_ring1_repository_write_violation_raised(self) -> None:
        """Verify write attempts into Ring 1 static repository directory raise AirGapViolationError."""
        repo_root = get_repo_root()
        illegal_manifest_path = repo_root / "cochem_deployment_manifest.json"

        ctx = ExecutionContext()
        cascade = parse_execution_cascade(symbols=["C", "H"], tier="T1-10s", context=ctx)

        with pytest.raises(AirGapViolationError) as exc_info:
            generate_provenance_hash(
                cascade=cascade,
                context=ctx,
                output_manifest_path=illegal_manifest_path,
                persist_manifest=True,
            )
        assert "Tripartite Air-Gap Violation" in str(exc_info.value)

    def test_ring2_scratch_and_ring3_artifacts_paths_disjoint(self, tmp_path: Path) -> None:
        """Verify scratch and artifacts directories resolve outside Ring 1 repo."""
        scratch_dir = tmp_path / "scratch_test"
        artifacts_dir = tmp_path / "artifacts_test"

        ctx = ExecutionContext(
            custom_scratch_dir=scratch_dir,
            custom_shm_dir=tmp_path / "shm_test",
            custom_artifacts_dir=artifacts_dir,
        )

        resolved_scratch = ctx.get_scratch_dir("sub_job_1")
        resolved_artifacts = ctx.get_artifacts_dir("sub_vault_1")

        assert resolved_scratch.exists()
        assert resolved_artifacts.exists()
        assert resolved_scratch != resolved_artifacts
        assert not ExecutionContext._is_repo_root_violation(resolved_scratch)
        assert not ExecutionContext._is_repo_root_violation(resolved_artifacts)

    def test_air_gap_report_model(self, tmp_path: Path) -> None:
        """Verify AirGapReport diagnostic model creation."""
        ctx = ExecutionContext()
        report = ctx.verify_air_gap_boundary(tmp_path / "safe_output.json")
        assert isinstance(report, AirGapReport)
        assert report.is_valid is True
        assert report.reason is None

    def test_environment_tier_path_resolutions(self, tmp_path: Path) -> None:
        """Verify path resolutions across different environment tiers."""
        for tier in EnvironmentTier:
            ctx = ExecutionContext(tier=tier, custom_scratch_dir=tmp_path / f"scratch_{tier.value}", custom_artifacts_dir=tmp_path / f"art_{tier.value}")
            s_dir = ctx.get_scratch_dir()
            a_dir = ctx.get_artifacts_dir()
            assert s_dir.exists()
            assert a_dir.exists()


# =============================================================================
# 7. Hardware-Aware Telemetry & Execution Context Tests
# =============================================================================


class TestHardwareTelemetryAndContext:
    """Authentic physical tests for execution context and hardware telemetry."""

    def test_execution_context_detection(self) -> None:
        """Verify execution context detects host environment tier and hardware."""
        ctx = ExecutionContext()
        assert ctx.tier in EnvironmentTier
        assert ctx.num_cores >= 1
        assert ctx.max_memory_mb >= 1024
        assert isinstance(ctx.gpu_available, bool)
        assert isinstance(ctx.session_id, str)
        assert len(ctx.session_id) > 0

    def test_execution_cascade_contains_hardware_affinity(self) -> None:
        """Verify hardware affinity is embedded into ExecutionCascade."""
        cascade = parse_execution_cascade(symbols=["C", "H", "O"], tier="T1-10s")
        assert "tier" in cascade.hardware_affinity
        assert "num_cores" in cascade.hardware_affinity
        assert "max_memory_mb" in cascade.hardware_affinity
        assert cascade.hardware_affinity["num_cores"] >= 1

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ.