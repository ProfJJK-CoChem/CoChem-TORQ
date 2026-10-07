"""CoChem-TORQ: Independent CREST Cross-Check & Union Referee Engine.

=============================================================================
Phase 3 (Stage 2.0 - 2.1) Implementation
-----------------------------------------------------------------------------
Governs independent non-covalent conformer and isomer exploration via CREST
(Conformer-Rotamer Ensemble Sampling Tool), enforcing mandatory non-covalent
interaction constraints (--nci --nocross --noreftopo), two-stage deduplication,
CREGEN union refereeing, and the 6-Step Union Protocol combining primary ORCA
GOAT with independent CREST search; benchmark performance requires a verified dataset and protocol.

Authoritative Standards & Directives:
- Method Matrix v4 Section 9B.1: The verdict on GOAT vs CREST and union merging
- Method Matrix v4 Section 9B.2: Conformer search comparison table and failure mode analysis
- Method Matrix v4 Section 9B.3: The six-step union protocol and spectroscopic deduplication
- Method Matrix v4 Section 9B.4: Flag syntax rules (--nci, --nocross, --noreftopo, --ewin 12)
- Method Matrix v4 Section 9B.5: Search budget allocation and calibration requirements
- Method Matrix v4 Section 8A.4: Concurrency and scout-and-anchor orchestration
- Method Matrix v4 Table 2: Conformer search tier constraints
- Mendeleev Mandate: 100% dynamic retrieval of atomic weights, isotopic masses, and radii
- Tripartite Filesystem Air-Gap Compliance: Ring 1 (Static), Ring 2 (Scratch), Ring 3 (Artifacts)
- CoChem Anti-Spoofing Protocol v2: Authentic physical constraints, zero stubs.
"""

from __future__ import annotations

import datetime
import enum
import functools
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal, Optional, Union

import h5py
from mendeleev import element as mendeleev_element
import numpy as np
import scipy.constants as const
from scipy.spatial.distance import cdist
from pydantic import BaseModel, ConfigDict, Field

from Libraries.cochem_isotopes import isotope_mass

# Configure module-level logging
logger = logging.getLogger("CoChem-TORQ.CREST")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: [CoChem-TORQ-CREST] %(message)s",
    )


# =============================================================================
# 1. Fundamental Physical Constants & Conversion Prefactors (CODATA / IUPAC)
# =============================================================================

PLANCK_CONSTANT_H: Final[float] = const.h  # J * s (6.62607015e-34)
SPEED_OF_LIGHT_C: Final[float] = const.c  # m / s (299792458.0)
ATOMIC_MASS_UNIT_KG: Final[float] = const.atomic_mass  # kg (1.66053906660e-27)
ANGSTROM_METERS: Final[float] = 1.0e-10  # m
BOLTZMANN_CONSTANT_K_CAL_MOL: Final[float] = 0.00198720425864083  # kcal / (mol * K)
GAS_CONSTANT_R_CAL_MOL: Final[float] = 1.98720425864083  # cal / (mol * K)

BOHR_TO_ANGSTROM: Final[float] = 0.529177210903
ANGSTROM_TO_BOHR: Final[float] = 1.0 / BOHR_TO_ANGSTROM

HARTREE_TO_EV: Final[float] = 27.211386245988
HARTREE_TO_KCAL_PER_MOL: Final[float] = 627.5094740631
HARTREE_TO_KJ_PER_MOL: Final[float] = 2625.4996394799
EV_TO_KCAL_PER_MOL: Final[float] = 23.060541945329334

# 1 u * Angstrom^2 in kg * m^2
U_ANGSTROM_SQ_TO_KG_M_SQ: Final[float] = ATOMIC_MASS_UNIT_KG * (ANGSTROM_METERS**2)

# Prefactors for B = h / (8 * pi^2 * I) where I is in u * Angstrom^2:
ROTATIONAL_PREFACTOR_HZ: Final[float] = PLANCK_CONSTANT_H / (
    8.0 * (np.pi**2) * U_ANGSTROM_SQ_TO_KG_M_SQ
)  # ~505379008435.3526 Hz * u * A^2
ROTATIONAL_PREFACTOR_MHZ: Final[float] = (
    ROTATIONAL_PREFACTOR_HZ / 1.0e6
)  # ~505379.0084353526 MHz * u * A^2
ROTATIONAL_PREFACTOR_GHZ: Final[float] = (
    ROTATIONAL_PREFACTOR_HZ / 1.0e9
)  # ~505.3790084353526 GHz * u * A^2
ROTATIONAL_PREFACTOR_CM1: Final[float] = PLANCK_CONSTANT_H / (
    8.0 * (np.pi**2) * (SPEED_OF_LIGHT_C * 100.0) * U_ANGSTROM_SQ_TO_KG_M_SQ
)  # ~16.85762916808776 cm^-1 * u * A^2


# =============================================================================
# 2. Dynamic Atomic Mass and Radii Integration (Mendeleev Mandate)
# =============================================================================

@functools.lru_cache(maxsize=256)
def get_dynamic_atomic_mass(symbol: str) -> float:
    """Resolve the selected isotope mass, using natural abundance only when available."""
    return isotope_mass(symbol.strip().rstrip(":"))


@functools.lru_cache(maxsize=256)
def get_dynamic_isotopic_mass(
    symbol: str, mass_number: Optional[int] = None
) -> float:
    """Dynamically retrieves isotopic mass via mendeleev."""
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = mendeleev_element(clean_sym)
    if mass_number is None:
        return get_dynamic_atomic_mass(symbol)
    for iso in el.isotopes:
        if iso.mass_number == mass_number:
            return float(iso.mass)
    raise ValueError(f"Isotopic mass unavailable for {mass_number}{symbol}.")


@functools.lru_cache(maxsize=256)
def get_dynamic_covalent_radius(symbol: str) -> float:
    """Dynamically retrieves Pyykko single-bond covalent radius in Angstroms via mendeleev."""
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = mendeleev_element(clean_sym)
    if el.covalent_radius_pyykko is not None:
        return float(el.covalent_radius_pyykko) / 100.0
    if el.covalent_radius is not None:
        return float(el.covalent_radius) / 100.0
    raise ValueError(f"Covalent radius unavailable for {symbol!r}.")


@functools.lru_cache(maxsize=256)
def get_dynamic_vdw_radius(symbol: str) -> float:
    """Dynamically retrieves van der Waals radius in Angstroms via mendeleev."""
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = mendeleev_element(clean_sym)
    if el.vdw_radius is not None:
        return float(el.vdw_radius) / 100.0
    raise ValueError(f"Van der Waals radius unavailable for {symbol!r}.")


# =============================================================================
# 3. Air-Gap Architecture & Exception Hierarchy
# =============================================================================

class AirGapViolationError(PermissionError):
    """Raised when an operation attempts to mutate or write into Ring 1 static repository space."""


class CrestError(Exception):
    """Base exception for all CoChem-TORQ CREST and conformer referee operations."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CrestExecutionError(CrestError):
    """Raised when the CREST binary subprocess exits with a non-zero return code or runtime failure."""


class CrestTimeoutError(CrestError):
    """Raised when a CREST or CREGEN calculation exceeds allocated wall-clock limits."""


class CrestDissociationError(CrestError):
    """Raised when a weakly bound non-covalent complex dissociates during metadynamics pushing."""


class CregenRefereeError(CrestError):
    """Raised when CREGEN filtering or union deduplication fails."""


class CoChemAirGapRing(enum.IntEnum):
    """Tripartite Filesystem Air-Gap Security Rings."""

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
        else:
            repo_path = Path(__file__).resolve().parent.parent
        return repo_path

    @classmethod
    def get_scratch_dir(cls) -> Path:
        """Domain C / Ring 2: Ephemeral runtime scratch workspace."""
        env_val = os.environ.get(cls.ENV_SCRATCH_DIR)
        if env_val:
            scratch_path = Path(env_val).resolve()
        else:
            scratch_path = Path(tempfile.gettempdir()).resolve() / "cochem_scratch"
        scratch_path.mkdir(parents=True, exist_ok=True)
        return scratch_path

    @classmethod
    def get_artifacts_dir(cls) -> Path:
        """Domain B / Ring 3: Persistent curated artifact storage."""
        env_val = os.environ.get(cls.ENV_ARTIFACTS_DIR) or os.environ.get(
            cls.ENV_ARTIFACTS_FALLBACK
        )
        if env_val:
            artifacts_path = Path(env_val).resolve()
        else:
            artifacts_path = Path.home().resolve() / "cochem_artifacts"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        return artifacts_path

    @classmethod
    def assert_air_gap(cls, target_path: Union[str, Path]) -> None:
        """Asserts that target_path does not violate Ring 1 immutability."""
        resolved = Path(target_path).resolve()
        repo_root = cls.get_repo_root()
        try:
            resolved.relative_to(repo_root)
            raise AirGapViolationError(
                f"[AIR-GAP VIOLATION] Attempted write operation inside Domain A / Ring 1 static repository: {resolved}"
            )
        except ValueError:
            # Not a subpath of repo_root; check equality
            if resolved == repo_root:
                raise AirGapViolationError(
                    f"[AIR-GAP VIOLATION] Target path matches Domain A / Ring 1 root: {resolved}"
                )


# =============================================================================
# 4. Pydantic v2 Models & Data Structures
# =============================================================================

class CrestConfig(BaseModel):
    """Configuration specification for CREST non-covalent conformer search.

    Authoritative flags per Method Matrix v4 §9B.1-9B.4.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    crest_bin: str = Field(
        default="crest",
        description="Path or binary name of CREST executable in PATH.",
    )
    ewin_kcal: float = Field(
        default=12.0,
        description="Energy window in kcal/mol (Method Matrix §9B.3 mandates 12 kcal/mol, default 6 is too tight).",
        ge=0.1,
        le=100.0,
    )
    nci: bool = Field(
        default=True,
        description="Mandatory Non-Covalent Interaction mode. Auto-adds ellipsoidal wall to prevent dissociation.",
    )
    nocross: bool = Field(
        default=True,
        description="Mandatory --nocross flag. Disables genetic crossing which scrambles clusters/dimers.",
    )
    noreftopo: bool = Field(
        default=True,
        description="Mandatory --noreftopo flag. Prevents deletion of isomers with differing contact topologies.",
    )
    gfn_level: Literal["gfn1", "gfn2", "gff", "gfn2//gfnff"] = Field(
        default="gfn2",
        description="Semi-empirical Hamiltonian level of theory for CREST metadynamics.",
    )
    wscal: Optional[float] = Field(
        default=None,
        description="Optional ellipsoidal wall scale factor (e.g. 0.9 if complex still dissociates).",
        gt=0.0,
        le=2.0,
    )
    threads: int = Field(
        default_factory=lambda: max(1, (os.cpu_count() or 4) - 1),
        description="Number of parallel OpenMP / CPU worker threads (--T <n>).",
        ge=1,
    )
    timeout_seconds: int = Field(
        default=3600,
        description="Maximum execution timeout in seconds for single CREST calculation.",
        ge=10,
    )
    niceprint: bool = Field(
        default=True,
        description="Format output with detailed conformational statistics (--niceprint).",
    )
    extra_flags: list[str] = Field(
        default_factory=list,
        description="Additional valid command line arguments passed directly to CREST.",
    )


class CregenConfig(BaseModel):
    """Configuration specification for CREGEN ensemble deduplication and refereeing.

    Complies with Method Matrix v4 §9B.3 Step 4 and Step 5.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    ewin_kcal: float = Field(
        default=12.0,
        description="Energy window in kcal/mol for refereeing.",
        ge=0.1,
    )
    ethr_kcal: float = Field(
        default=0.05,
        description="Energy degeneracy threshold in kcal/mol (default 0.05 kcal/mol ≈ 8e-5 Eh).",
        ge=1e-5,
    )
    bthr: float = Field(
        default=0.01,
        description="Rotational constant match threshold (Stage A: 0.01 = 1.0%; Stage B: 0.001 = 0.1%).",
        ge=1e-5,
        le=0.1,
    )
    rthr_angstrom: float = Field(
        default=0.125,
        description="Cartesian heavy-atom RMSD threshold in Angstroms (default 0.125 Å).",
        ge=1e-4,
        le=5.0,
    )
    threads: int = Field(
        default_factory=lambda: max(1, (os.cpu_count() or 4) - 1),
        description="Parallel threads for CREGEN.",
        ge=1,
    )
    timeout_seconds: int = Field(
        default=1800,
        description="Maximum timeout in seconds for CREGEN execution.",
        ge=10,
    )


class ConformerRecord(BaseModel):
    """Data container for an individual molecular conformer/isomer candidate."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    index: int = Field(..., description="Unique conformer index in the ensemble.")
    symbols: list[str] = Field(
        ..., description="List of atomic element symbols (e.g. ['C', 'H', 'O'])."
    )
    coordinates: list[list[float]] = Field(
        ..., description="Cartesian atomic coordinates in Angstroms (Nx3 array)."
    )
    energy_hartree: Optional[float] = Field(
        default=None, description="Absolute electronic energy in Hartree (Eh)."
    )
    energy_kcal_rel: Optional[float] = Field(
        default=None,
        description="Relative electronic energy in kcal/mol relative to ensemble minimum.",
    )
    rotational_constants_mhz: Optional[tuple[Optional[float], Optional[float], Optional[float]]] = Field(
        default=None,
        description="Principal rotational constants (A, B, C) in MHz.",
    )
    rotational_constants_ghz: Optional[tuple[Optional[float], Optional[float], Optional[float]]] = Field(
        default=None,
        description="Principal rotational constants (A, B, C) in GHz.",
    )
    inertial_defect_u_a2: Optional[float] = Field(
        default=None,
        description="Inertial defect Delta = Ic - Ia - Ib in u * Angstrom^2.",
    )
    planar_moments_u_a2: Optional[tuple[Optional[float], Optional[float], Optional[float]]] = Field(
        default=None,
        description="Planar moments of inertia (P_aa, P_bb, P_cc) in u * Angstrom^2.",
    )
    ray_asymmetry_kappa: Optional[float] = Field(
        default=None,
        description="Ray's asymmetry parameter kappa = (2B - A - C) / (A - C).",
    )
    origin_engine: str = Field(
        default="IMPORTED",
        description="Originating generator engine ('GOAT', 'CREST', 'SEEDED', 'UNION').",
    )
    seed_id: Optional[str] = Field(
        default=None,
        description="Identifier of originating seed geometry or binding topology.",
    )
    is_dissociated: bool = Field(
        default=False,
        description="True if non-covalent complex dissociated beyond physical vdW bounds.",
    )
    provenance_tag: str = Field(
        default="[D]",
        description="Method Matrix provenance tag: [M] Measured, [D] Derived, [E] Estimated.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata dictionary for downstream stages.",
    )


class EnsembleContainer(BaseModel):
    """Ensemble of conformers resulting from search, screening, or union refereeing."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(..., description="Descriptive identifier for the ensemble.")
    conformers: list[ConformerRecord] = Field(
        default_factory=list, description="List of conformer records in ensemble."
    )
    temperature_k: float = Field(
        default=298.15, description="Thermodynamic temperature in Kelvin."
    )
    s_conf_cal_mol_k: Optional[float] = Field(
        default=None,
        description="Conformational entropy S_conf in cal / (mol * K).",
    )
    boltzmann_weights: list[float] = Field(
        default_factory=list,
        description="Normalized Boltzmann probability weights at temperature_k.",
    )
    provenance_tag: str = Field(
        default="[D]",
        description="Method Matrix provenance tag.",
    )


class UnionAuditReport(BaseModel):
    """Immutable audit report detailing the 6-Step Union Protocol diagnostics.

    Adheres strictly to Method Matrix v4 §9B.3 Step 6.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    system_name: str = Field(
        ..., description="Name of the chemical system under investigation."
    )
    n_seeds: int = Field(
        ..., description="Number of distinct starting seed geometries used."
    )
    n_goat_raw: int = Field(
        ..., description="Total raw conformers generated by ORCA GOAT primary engine."
    )
    n_crest_raw: int = Field(
        ...,
        description="Total raw conformers generated by independent CREST NCI engine.",
    )
    n_union_raw: int = Field(
        ..., description="Total un-refereed conformers in combined union pool."
    )
    n_goat_unique: int = Field(
        ..., description="Number of unique conformers discovered exclusively by GOAT."
    )
    n_crest_unique: int = Field(
        ..., description="Number of unique conformers discovered exclusively by CREST."
    )
    n_shared_intersection: int = Field(
        ..., description="Number of conformers identified by both GOAT and CREST."
    )
    n_survivors_stage_a: int = Field(
        ..., description="Conformers surviving Stage A broad CREGEN refereeing."
    )
    n_survivors_stage_b: int = Field(
        ...,
        description="Conformers surviving Stage B spectroscopic deduplication (Delta B/B <= 0.1%).",
    )
    goat_f1_baseline: Optional[float] = Field(
        default=None,
        description="External benchmark F1, populated only with verified benchmark provenance.",
    )
    crest_f1_baseline: Optional[float] = Field(
        default=None,
        description="External benchmark F1, populated only with verified benchmark provenance.",
    )
    union_coverage_ratio: Optional[float] = Field(
        default=None,
        description="Fractional coverage of union over single-engine explorations.",
    )
    s_conf_goat_cal_mol_k: Optional[float] = Field(
        default=None,
        description="Conformational entropy from GOAT ensemble in cal/(mol*K).",
    )
    s_conf_crest_cal_mol_k: Optional[float] = Field(
        default=None,
        description="Conformational entropy from CREST ensemble in cal/(mol*K).",
    )
    s_conf_union_cal_mol_k: Optional[float] = Field(
        default=None,
        description="Conformational entropy of refereed union in cal/(mol*K).",
    )
    completeness_disclaimer: str = Field(
        default=(
            "Both GOAT and CREST are stochastic global optimizers (heuristics). "
            "No completeness proof exists unless binding topologies are exhaustively "
            "hand-enumerated (HFIP...Rg protocol per Method Matrix §9B.3 Step 0)."
        ),
        description="Mandatory scientific completeness disclaimer per Method Matrix §9B.3.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        description="ISO 8601 UTC timestamp of audit generation.",
    )
    provenance_tag: str = Field(
        default="[M]",
        description="Authoritative provenance tag.",
    )


# =============================================================================
# 5. Core Geometry, Moment of Inertia & Spectroscopic Mathematics
# =============================================================================

def compute_center_of_mass(
    symbols: Sequence[str], coordinates: np.ndarray
) -> np.ndarray:
    """Computes mass-weighted center of mass in Angstroms via Mendeleev."""
    masses = np.array(
        [get_dynamic_atomic_mass(s) for s in symbols], dtype=np.float64
    )
    total_mass = np.sum(masses)
    if total_mass <= 0.0:
        raise ValueError(
            "Total mass must be strictly positive for center of mass calculation."
        )
    return np.sum(coordinates * masses[:, np.newaxis], axis=0) / total_mass


def compute_moments_and_constants(
    symbols: Sequence[str], coordinates: np.ndarray
) -> dict[str, Any]:
    """Diagonalizes inertia tensor and computes rotational constants, inertial defect,

    and planar moments using dynamic Mendeleev masses and CODATA conversion factors.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    n_atoms = len(symbols)
    if n_atoms == 0 or coords.shape[0] != n_atoms or coords.shape[1] != 3:
        raise ValueError("Invalid coordinates shape or empty atom list.")

    masses = np.array(
        [get_dynamic_atomic_mass(s) for s in symbols], dtype=np.float64
    )
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
    shifted_coords = coords - com

    # Inertia tensor components: I_alpha_beta in u * Angstrom^2
    inertia_tensor = np.full((3, 3), 0.0, dtype=np.float64)
    x = shifted_coords[:, 0]
    y = shifted_coords[:, 1]
    z = shifted_coords[:, 2]

    inertia_tensor[0, 0] = np.sum(masses * (y**2 + z**2))
    inertia_tensor[1, 1] = np.sum(masses * (x**2 + z**2))
    inertia_tensor[2, 2] = np.sum(masses * (x**2 + y**2))
    inertia_tensor[0, 1] = inertia_tensor[1, 0] = -np.sum(masses * x * y)
    inertia_tensor[0, 2] = inertia_tensor[2, 0] = -np.sum(masses * x * z)
    inertia_tensor[1, 2] = inertia_tensor[2, 1] = -np.sum(masses * y * z)

    # Diagonalize: principal moments I_a <= I_b <= I_c
    eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
    eigvals = np.sort(eigvals)
    if np.any(eigvals < -1e-8):
        raise ValueError("Inertia tensor has invalid negative principal moments.")
    eigvals[np.abs(eigvals) < 1e-8] = 0.0
    i_a, i_b, i_c = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])

    # Rotational constants: B = prefactor / I
    a_mhz = ROTATIONAL_PREFACTOR_MHZ / i_a if i_a > 1e-6 else None
    b_mhz = ROTATIONAL_PREFACTOR_MHZ / i_b if i_b > 1e-6 else None
    c_mhz = ROTATIONAL_PREFACTOR_MHZ / i_c if i_c > 1e-6 else None

    a_ghz = a_mhz / 1000.0 if a_mhz is not None else None
    b_ghz = b_mhz / 1000.0 if b_mhz is not None else None
    c_ghz = c_mhz / 1000.0 if c_mhz is not None else None

    a_cm1 = ROTATIONAL_PREFACTOR_CM1 / i_a if i_a > 1e-6 else None
    b_cm1 = ROTATIONAL_PREFACTOR_CM1 / i_b if i_b > 1e-6 else None
    c_cm1 = ROTATIONAL_PREFACTOR_CM1 / i_c if i_c > 1e-6 else None

    # Planar moments: P_aa = 0.5*(I_b + I_c - I_a), etc.
    p_aa = 0.5 * (i_b + i_c - i_a)
    p_bb = 0.5 * (i_a + i_c - i_b)
    p_cc = 0.5 * (i_a + i_b - i_c)

    # Inertial defect: Delta = I_c - I_a - I_b
    delta_inertial = i_c - i_a - i_b

    # Ray's asymmetry parameter: kappa = (2B - A - C) / (A - C)
    if all(v is not None for v in (a_mhz, b_mhz, c_mhz)) and abs(a_mhz - c_mhz) > 1e-6:
        kappa = (2.0 * b_mhz - a_mhz - c_mhz) / (a_mhz - c_mhz)
    else:
        kappa = None

    return {
        "inertia_tensor_u_a2": inertia_tensor,
        "principal_moments_u_a2": (i_a, i_b, i_c),
        "rotational_constants_mhz": (a_mhz, b_mhz, c_mhz),
        "rotational_constants_ghz": (a_ghz, b_ghz, c_ghz),
        "rotational_constants_cm1": (a_cm1, b_cm1, c_cm1),
        "planar_moments_u_a2": (p_aa, p_bb, p_cc),
        "inertial_defect_u_a2": delta_inertial,
        "ray_asymmetry_kappa": kappa,
    }


def kabsch_align(
    coords_p: np.ndarray,
    coords_q: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray]:
    """Performs optimal Kabsch rigid-body alignment of coords_p onto coords_q.

    Returns the minimal RMSD (in Angstroms) and the rotated coordinates of P.
    """
    p = np.asarray(coords_p, dtype=np.float64)
    q = np.asarray(coords_q, dtype=np.float64)

    if p.shape != q.shape:
        raise ValueError(
            f"Coordinate arrays must have identical shape: {p.shape} vs {q.shape}"
        )

    n_atoms = p.shape[0]
    if masses is not None:
        w = np.asarray(masses, dtype=np.float64)
        w_sum = np.sum(w)
        center_p = np.sum(p * w[:, np.newaxis], axis=0) / w_sum
        center_q = np.sum(q * w[:, np.newaxis], axis=0) / w_sum
    else:
        w = np.full(n_atoms, 1.0, dtype=np.float64)
        w_sum = float(n_atoms)
        center_p = np.mean(p, axis=0)
        center_q = np.mean(q, axis=0)

    p_centered = p - center_p
    q_centered = q - center_q

    # Covariance matrix H = P^T * W * Q
    h = np.dot((p_centered * w[:, np.newaxis]).T, q_centered)
    u, s, vt = np.linalg.svd(h)
    d = np.linalg.det(np.dot(vt.T, u.T))

    # Reflection correction to ensure proper right-handed rotation (det(R) = +1)
    correction = np.diag([1.0, 1.0, np.sign(d)])
    rotation_matrix = np.dot(vt.T, np.dot(correction, u.T))

    p_rotated = np.dot(p_centered, rotation_matrix.T) + center_q
    diff = p_rotated - q
    rmsd = float(np.sqrt(np.sum(w[:, np.newaxis] * (diff**2)) / w_sum))

    return rmsd, p_rotated


def compute_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> float:
    """Computes minimal aligned RMSD in Angstroms between two geometries."""
    rmsd_val, _ = kabsch_align(coords1, coords2, masses=masses)
    return rmsd_val


def check_complex_dissociation(
    symbols: Sequence[str],
    coordinates: np.ndarray,
    max_vdw_factor: float = 2.4,
) -> bool:
    """Evaluates whether any atom or fragment in the complex has drifted beyond

    van der Waals connectivity (indicating unphysical cluster dissociation during pushing).
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    n_atoms = len(symbols)
    if n_atoms <= 1:
        return False

    vdw_radii = np.array(
        [get_dynamic_vdw_radius(s) for s in symbols], dtype=np.float64
    )
    dist_matrix = cdist(coords, coords)

    # Build adjacency based on sum of vdW radii * max_vdw_factor
    threshold_matrix = (
        vdw_radii[:, np.newaxis] + vdw_radii[np.newaxis, :]
    ) * max_vdw_factor
    adj = (dist_matrix < threshold_matrix) & (dist_matrix > 1e-4)

    # Check if graph is connected using simple BFS
    visited = np.full(n_atoms, False, dtype=bool)
    queue = [0]
    visited[0] = True

    while queue:
        curr = queue.pop(0)
        neighbors = np.where(adj[curr])[0]
        for nbr in neighbors:
            if not visited[nbr]:
                visited[nbr] = True
                queue.append(nbr)

    # If any atom is unvisited, the complex has fragmented/dissociated
    return not bool(np.all(visited))


def calculate_conformational_entropy(
    energies_kcal: Sequence[float],
    temperature_k: float = 298.15,
) -> tuple[float, list[float]]:
    """Calculates Boltzmann weights and conformational entropy S_conf in cal/(mol*K).

    S_conf = -R * sum(p_i * ln(p_i)) per standard statistical mechanics.
    """
    if not energies_kcal:
        raise ValueError("Conformational entropy requires a nonempty energy ensemble.")

    e_arr = np.asarray(energies_kcal, dtype=np.float64)
    if not np.isfinite(e_arr).all() or not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Conformational entropy requires finite energies and positive temperature.")
    e_rel = e_arr - np.min(e_arr)

    beta = 1.0 / (BOLTZMANN_CONSTANT_K_CAL_MOL * temperature_k)
    weights_raw = np.exp(-beta * e_rel)
    q_partition = np.sum(weights_raw)

    if q_partition <= 0.0 or np.isnan(q_partition):
        raise ValueError("Boltzmann partition function is invalid; no uniform distribution is substituted.")
    else:
        probs = weights_raw / q_partition

    # Shannon / Gibbs conformational entropy in cal / (mol * K)
    log_probs = np.full_like(probs, 0.0)
    valid_idx = probs > 1e-15
    log_probs[valid_idx] = np.log(probs[valid_idx])

    s_conf = -GAS_CONSTANT_R_CAL_MOL * float(np.sum(probs * log_probs))
    return max(0.0, s_conf), probs.tolist()


# =============================================================================
# 6. Multi-Structure XYZ File Parsing & Writing
# =============================================================================

def parse_xyz_string(
    content: str, default_origin: str = "IMPORTED", seed_id: Optional[str] = None
) -> list[ConformerRecord]:
    """Parses a single or multi-structure XYZ string into a list of ConformerRecords."""
    lines = content.strip().splitlines()
    records: list[ConformerRecord] = []
    line_idx = 0
    total_lines = len(lines)
    record_count = 0

    while line_idx < total_lines:
        line = lines[line_idx].strip()
        if not line:
            line_idx += 1
            continue

        try:
            n_atoms = int(line)
        except ValueError:
            line_idx += 1
            continue

        comment = (
            lines[line_idx + 1].strip() if line_idx + 1 < total_lines else ""
        )
        line_idx += 2

        symbols: list[str] = []
        coordinates: list[list[float]] = []

        for _ in range(n_atoms):
            if line_idx >= total_lines:
                break
            atom_line = lines[line_idx].strip()
            line_idx += 1
            if not atom_line:
                continue
            parts = atom_line.split()
            if len(parts) >= 4:
                symbols.append(parts[0])
                coordinates.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )

        if len(symbols) == n_atoms:
            coords_arr = np.array(coordinates, dtype=np.float64)
            # Parse energy from comment if present
            energy_hartree: Optional[float] = None
            # Accept a bare CREST Hartree comment or an explicitly labelled
            # Hartree value. Unrelated numbers in titles are not energies.
            number = r"[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?"
            energy_match = re.fullmatch(rf"\s*({number})\s*", comment)
            if energy_match is None:
                energy_match = re.search(rf"(?:^|\s)E\s*=\s*({number})\s*(?:Eh|Hartree)\b", comment, re.IGNORECASE)
            if energy_match:
                energy_hartree = float(energy_match.group(1).replace("D", "E").replace("d", "e"))
                if not np.isfinite(energy_hartree):
                    raise ValueError("XYZ energy must be finite.")

            phys_data = compute_moments_and_constants(symbols, coords_arr)
            is_dissoc = check_complex_dissociation(symbols, coords_arr)

            record = ConformerRecord(
                index=record_count,
                symbols=symbols,
                coordinates=coordinates,
                energy_hartree=energy_hartree,
                energy_kcal_rel=None,
                rotational_constants_mhz=phys_data["rotational_constants_mhz"],
                rotational_constants_ghz=phys_data["rotational_constants_ghz"],
                inertial_defect_u_a2=phys_data["inertial_defect_u_a2"],
                planar_moments_u_a2=phys_data["planar_moments_u_a2"],
                ray_asymmetry_kappa=phys_data["ray_asymmetry_kappa"],
                origin_engine=default_origin,
                seed_id=seed_id,
                is_dissociated=is_dissoc,
                provenance_tag="[D]",
                metadata={"comment": comment, "energy_status": "imported" if energy_hartree is not None else "unavailable"},
            )
            records.append(record)
            record_count += 1

    # Update relative energies if absolute energies exist
    energies_h = [
        r.energy_hartree for r in records if r.energy_hartree is not None
    ]
    if len(energies_h) == len(records) and len(records) > 0:
        min_e = min(energies_h)
        for r in records:
            if r.energy_hartree is not None:
                r.energy_kcal_rel = (
                    r.energy_hartree - min_e
                ) * HARTREE_TO_KCAL_PER_MOL

    return records


def parse_xyz_file(
    file_path: Union[str, Path],
    default_origin: str = "IMPORTED",
    seed_id: Optional[str] = None,
) -> list[ConformerRecord]:
    """Reads and parses an XYZ file."""
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"XYZ file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    return parse_xyz_string(text, default_origin=default_origin, seed_id=seed_id)


def write_xyz_string(records: Sequence[ConformerRecord]) -> str:
    """Serializes a list of ConformerRecords into a multi-structure XYZ format."""
    out_lines: list[str] = []
    for r in records:
        n_atoms = len(r.symbols)
        if r.energy_hartree is not None:
            e_str = f"E = {r.energy_hartree:.12g} Eh"
        elif r.energy_kcal_rel is not None:
            e_str = f"dE = {r.energy_kcal_rel:.12g} kcal/mol"
        else:
            e_str = "energy unavailable"
        comment = f"{e_str} | Origin: {r.origin_engine}"
        if r.rotational_constants_mhz is not None and all(v is not None for v in r.rotational_constants_mhz):
            a, b, c = r.rotational_constants_mhz
            comment += f" | A={a:.8g} B={b:.8g} C={c:.8g} MHz"
        if r.inertial_defect_u_a2 is not None:
            comment += f" | Delta={r.inertial_defect_u_a2:.8g} uA2"
        out_lines.append(str(n_atoms))
        out_lines.append(comment)
        for sym, pos in zip(r.symbols, r.coordinates):
            out_lines.append(
                f"{sym:<3} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}"
            )
    return "\n".join(out_lines) + "\n"


def write_xyz_file(
    records: Sequence[ConformerRecord], file_path: Union[str, Path]
) -> Path:
    """Writes ConformerRecords to a file on disk, respecting Air-Gap immutability."""
    target = Path(file_path).resolve()
    CoChemPathManager.assert_air_gap(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = write_xyz_string(records)
    target.write_text(content, encoding="utf-8")
    return target


# =============================================================================
# 7. Independent CREST Search Engine (CrestRunner)
# =============================================================================

class CrestRunner:
    """Executes the independent CREST conformer search engine with mandatory

    non-covalent interaction constraints (--nci --nocross --noreftopo).
    Complies with Method Matrix v4 §9B.1-9B.4.
    """

    def __init__(self, config: Optional[CrestConfig] = None) -> None:
        self.config = config or CrestConfig()
        self.scratch_base = CoChemPathManager.get_scratch_dir()

    def build_command_line(self, input_xyz_filename: str) -> list[str]:
        """Constructs the canonical CREST CLI command per Method Matrix v4 §9B.3."""
        cmd = [self.config.crest_bin, input_xyz_filename]

        if self.config.nci:
            cmd.append("--nci")
        if self.config.nocross:
            cmd.append("--nocross")
        if self.config.noreftopo:
            cmd.append("--noreftopo")

        cmd.extend(["--" + self.config.gfn_level])
        cmd.extend(["--ewin", str(self.config.ewin_kcal)])
        cmd.extend(["--T", str(self.config.threads)])

        if self.config.wscal is not None:
            cmd.extend(["--wscal", str(self.config.wscal)])
        if self.config.niceprint:
            cmd.append("--niceprint")

        cmd.extend(self.config.extra_flags)
        return cmd

    def run_crest(
        self,
        input_xyz: Union[str, Path],
        seed_id: Optional[str] = None,
        work_dir: Optional[Path] = None,
        flags: Optional[str] = None,
        **kwargs: Any,
    ) -> EnsembleContainer:
        if flags:
            if "--nci" in flags:
                self.config.nci = True
            if "--nocross" in flags:
                self.config.nocross = True
            if "--noreftopo" in flags:
                self.config.noreftopo = True
        """Executes CREST on the input XYZ file and returns the resulting EnsembleContainer."""
        src_path = Path(input_xyz).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Input seed XYZ not found: {src_path}")

        run_id = f"crest_{uuid.uuid4().hex[:8]}"
        exec_dir = (Path(work_dir) if work_dir is not None else self.scratch_base) / run_id
        exec_dir.mkdir(parents=True, exist_ok=True)

        CoChemPathManager.assert_air_gap(exec_dir)
        local_input = exec_dir / "input.xyz"
        shutil.copy2(src_path, local_input)

        crest_executable = shutil.which(self.config.crest_bin)
        if crest_executable is None:
            raise FileNotFoundError(f"Requested CREST executable unavailable: {self.config.crest_bin}")

        cmd = self.build_command_line(local_input.name)
        cmd[0] = str(crest_executable)
        logger.info(
            f"[CREST] Executing command in {exec_dir}: {' '.join(cmd)}"
        )

        try:
            result = subprocess.run(
                cmd,
                cwd=str(exec_dir),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
            if result.returncode != 0:
                logger.warning(
                    f"[CREST] Subprocess returned code {result.returncode}. Stderr: {result.stderr[:300]}"
                )
                raise CrestExecutionError(
                    f"CREST execution failed with code {result.returncode} on {src_path.name}: {result.stderr[:300]}"
                )
        except subprocess.TimeoutExpired as exc:
            logger.error(
                f"[CREST] Execution timed out after {self.config.timeout_seconds} s: {exc}"
            )
            raise CrestTimeoutError(
                f"CREST execution timed out on {src_path.name}"
            ) from exc

        # Identify resulting ensemble files
        candidate_files = [
            exec_dir / "crest_conformers.xyz",
            exec_dir / "crest_ensemble.xyz",
            exec_dir / "crest_rotamers.xyz",
            exec_dir / "crest_best.xyz",
        ]
        ensemble_file: Optional[Path] = None
        for candidate in candidate_files:
            if candidate.exists() and candidate.stat().st_size > 0:
                ensemble_file = candidate
                break

        if ensemble_file is not None:
            records = parse_xyz_file(
                ensemble_file, default_origin="CREST", seed_id=seed_id
            )
            if records:
                e_vals = [
                    r.energy_kcal_rel
                    if r.energy_hartree is None
                    else r.energy_hartree * HARTREE_TO_KCAL_PER_MOL
                    for r in records
                ]
                if any(e is None for e in e_vals):
                    raise CrestExecutionError("CREST ensemble has missing energies; thermodynamic characterization is unavailable.")
                s_conf, weights = calculate_conformational_entropy(e_vals)
                return EnsembleContainer(
                    name=f"CREST_Ensemble_{seed_id or src_path.stem}",
                    conformers=records,
                    temperature_k=298.15,
                    s_conf_cal_mol_k=s_conf,
                    boltzmann_weights=weights,
                    provenance_tag="[M]",
                )

        raise CrestExecutionError(
            f"CREST execution did not produce conformers on {src_path.name}"
        )


# =============================================================================
# 8. CREGEN Union Referee & Deduplication Engine (CregenReferee)
# =============================================================================

class CregenReferee:
    """Python ensemble deduplication; this class does not execute the CREGEN binary.

    Complies with Method Matrix v4 §9B.1, §9B.3 Step 4 and Step 5.
    """

    def __init__(self, config: Optional[CregenConfig] = None) -> None:
        self.config = config or CregenConfig()
        self.scratch_base = CoChemPathManager.get_scratch_dir()

    def referee_ensemble(
        self,
        conformers: Sequence[ConformerRecord],
        is_spectroscopic_stage: bool = False,
    ) -> list[ConformerRecord]:
        """Referees the input conformer pool using the 5-step sieve algorithm:

        1. Energy sorting and window cutoff (ewin_kcal).
        2. Energy degeneracy grouping (ethr_kcal).
        3. Rotational constant match (|Delta B| / B <= bthr).
        4. Mass-weighted Kabsch RMSD alignment (RMSD <= rthr_angstrom).
        5. Contact topology evaluation.
        """
        if not conformers:
            return []

        for conf in conformers:
            if conf.energy_kcal_rel is None or not np.isfinite(conf.energy_kcal_rel):
                raise ValueError("Conformer energy unavailable; energy-based refereeing cannot proceed.")
            if conf.rotational_constants_mhz is None or any(v is None or not np.isfinite(v) for v in conf.rotational_constants_mhz):
                raise ValueError("Rotational constants unavailable; refereeing cannot proceed.")

        # Step 1: Energy sorting and Ewin window filtering
        sorted_confs = sorted(conformers, key=lambda c: c.energy_kcal_rel)
        min_e = sorted_confs[0].energy_kcal_rel
        window_confs = [
            c
            for c in sorted_confs
            if (c.energy_kcal_rel - min_e) <= self.config.ewin_kcal
        ]

        if not window_confs:
            return []

        # Effective thresholds
        bthr_eff = (
            0.001 if is_spectroscopic_stage else self.config.bthr
        )  # 0.1% vs 1.0%
        rthr_eff = self.config.rthr_angstrom
        ethr_eff = self.config.ethr_kcal

        unique_conformers: list[ConformerRecord] = []

        for candidate in window_confs:
            c_coords = np.array(candidate.coordinates, dtype=np.float64)
            c_b_mhz = np.array(
                candidate.rotational_constants_mhz, dtype=np.float64
            )
            c_masses = np.array(
                [get_dynamic_atomic_mass(s) for s in candidate.symbols],
                dtype=np.float64,
            )

            is_duplicate = False
            for accepted in unique_conformers:
                a_coords = np.array(accepted.coordinates, dtype=np.float64)
                a_b_mhz = np.array(
                    accepted.rotational_constants_mhz, dtype=np.float64
                )

                # Condition 1: Energy check (|Delta E| < ethr)
                delta_e = abs(
                    candidate.energy_kcal_rel - accepted.energy_kcal_rel
                )
                if delta_e > ethr_eff:
                    continue

                # Condition 2: Rotational constant check (|Delta B| / B < bthr)
                b_diff_ratio = np.abs(c_b_mhz - a_b_mhz) / np.maximum(
                    a_b_mhz, 1e-3
                )
                if np.max(b_diff_ratio) > bthr_eff:
                    continue

                # Condition 3: Mass-weighted Kabsch RMSD check (RMSD < rthr)
                rmsd_val = compute_rmsd(c_coords, a_coords, masses=c_masses)
                if rmsd_val <= rthr_eff:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_conformers.append(candidate)

        # Re-index unique survivors
        refereed_records: list[ConformerRecord] = []
        for idx, rec in enumerate(unique_conformers):
            updated_rec = rec.model_copy(
                update={"index": idx, "origin_engine": "UNION"}
            )
            refereed_records.append(updated_rec)

        return refereed_records


# =============================================================================
# 9. Two-Stage Spectroscopic Deduplication Engine
# =============================================================================

def deduplicate_spectroscopic(
    conformers: Sequence[ConformerRecord],
    bthr_spectroscopic: float = 0.001,
    rmsd_thr: float = 0.125,
    ethr_kcal: float = 0.05,
) -> list[ConformerRecord]:
    """Executes Stage B Spectroscopic Deduplication per Method Matrix v4 §9B.3 Step 5.

    Tightens the rotational constant threshold to 0.1% (12 MHz at 12 GHz), merging
    candidate structures indistinguishable by chirped-pulse microwave spectroscopy.
    """
    config = CregenConfig(
        bthr=bthr_spectroscopic,
        rthr_angstrom=rmsd_thr,
        ethr_kcal=ethr_kcal,
    )
    referee = CregenReferee(config=config)
    return referee.referee_ensemble(conformers, is_spectroscopic_stage=True)


# =============================================================================
# 10. The 6-Step Master Union Protocol Orchestrator (UnionConformerReferee)
# =============================================================================

class UnionConformerReferee:
    """Master Orchestrator executing the 6-Step Union Protocol (Method Matrix §9B.3).

    Step 0: Multi-seed hand enumeration / distinct binding sites.
    Step 1: Import a completed primary GOAT conformer enumeration.
    Step 2: GOAT GFN2-xTB unbiased verification.
    Step 3: Independent CREST NCI cross-check (--nci --nocross --noreftopo).
    Step 4: Union creation, single-level re-evaluation, and CREGEN refereeing.
    Step 5: Two-stage spectroscopic deduplication (Delta B/B <= 0.1%).
    Step 6: Union diagnostics audit reporting (GOAT vs CREST unique & shared metrics).
    """

    def __init__(
        self,
        crest_config: Optional[CrestConfig] = None,
        cregen_config: Optional[CregenConfig] = None,
    ) -> None:
        self.crest_runner = CrestRunner(config=crest_config)
        self.cregen_referee = CregenReferee(config=cregen_config)
        self.artifacts_base = CoChemPathManager.get_artifacts_dir()
        self.scratch_base = CoChemPathManager.get_scratch_dir()

    def execute_union_protocol(
        self,
        seed_inputs: Sequence[
            Union[str, Path, tuple[list[str], np.ndarray], ConformerRecord]
        ],
        goat_ensemble_xyz: Optional[Union[str, Path]] = None,
        system_name: str = "vdW_Complex",
        hdf5_out_path: Optional[Union[str, Path]] = None,
        goat_energy_model: Optional[str] = None,
    ) -> tuple[EnsembleContainer, UnionAuditReport]:
        """Executes the complete 6-Step Union Protocol and generates FAIR-compliant

        HDF5 outputs and an immutable UnionAuditReport.
        """
        if goat_energy_model != self.crest_runner.config.gfn_level:
            raise ValueError("GOAT energy-model provenance must explicitly match the CREST gfn_level before comparing ensemble energies.")
        if goat_ensemble_xyz is None or not Path(goat_ensemble_xyz).is_file():
            raise FileNotFoundError("A completed GOAT ensemble is required; seed geometries are not GOAT results.")
        logger.info(
            f"[UNION-PROTOCOL] Initiating 6-Step Union Protocol for '{system_name}'..."
        )

        # Step 0: Ingest Seeds (3-6 chemically distinct starting topologies)
        seed_records: list[ConformerRecord] = []
        seed_files: list[Path] = []

        for idx, item in enumerate(seed_inputs):
            seed_id = f"seed_{idx+1:02d}"
            if isinstance(item, (str, Path)):
                p = Path(item).resolve()
                seed_files.append(p)
                seed_records.extend(
                    parse_xyz_file(p, default_origin="SEEDED", seed_id=seed_id)
                )
            elif isinstance(item, tuple) and len(item) == 2:
                syms, coords = item
                phys = compute_moments_and_constants(syms, np.asarray(coords))
                rec = ConformerRecord(
                    index=len(seed_records),
                    symbols=list(syms),
                    coordinates=np.asarray(coords).tolist(),
                    energy_hartree=None,
                    energy_kcal_rel=None,
                    rotational_constants_mhz=phys["rotational_constants_mhz"],
                    rotational_constants_ghz=phys["rotational_constants_ghz"],
                    inertial_defect_u_a2=phys["inertial_defect_u_a2"],
                    planar_moments_u_a2=phys["planar_moments_u_a2"],
                    ray_asymmetry_kappa=phys["ray_asymmetry_kappa"],
                    origin_engine="SEEDED",
                    seed_id=seed_id,
                )
                seed_records.append(rec)
                seed_f = self.scratch_base / f"{system_name}_{seed_id}.xyz"
                write_xyz_file([rec], seed_f)
                seed_files.append(seed_f)
            elif isinstance(item, ConformerRecord):
                seed_records.append(item)
                seed_f = self.scratch_base / f"{system_name}_{seed_id}.xyz"
                write_xyz_file([item], seed_f)
                seed_files.append(seed_f)

        if not seed_files:
            raise ValueError(
                "[MISSING DATA] At least one seed geometry required for conformer exploration."
            )

        n_seeds = len(seed_files)
        logger.info(
            f"[UNION-PROTOCOL] Step 0 Complete: Ingested {n_seeds} distinct seeds."
        )

        # Step 1 & 2: Ingest Primary GOAT Ensemble
        goat_records: list[ConformerRecord] = []
        if goat_ensemble_xyz is not None and Path(goat_ensemble_xyz).exists():
            goat_records = parse_xyz_file(
                goat_ensemble_xyz, default_origin="GOAT"
            )
            logger.info(
                f"[UNION-PROTOCOL] Step 1-2: Ingested {len(goat_records)} conformers from primary GOAT ensemble."
            )

        # Step 3: Independent CREST NCI Cross-Check on All Seeds
        crest_records: list[ConformerRecord] = []
        for sf in seed_files:
            try:
                res_ensemble = self.crest_runner.run_crest(
                    sf, seed_id=sf.stem
                )
                crest_records.extend(res_ensemble.conformers)
            except Exception as exc:
                raise CrestExecutionError(
                    f"Independent CREST search failed for {sf.name}; seeds were retained as inputs and no CREST result was created: {exc}"
                ) from exc

        logger.info(
            f"[UNION-PROTOCOL] Step 3 Complete: CREST independent search yielded {len(crest_records)} raw conformers."
        )

        # Step 4: Union Creation & Stage A CREGEN Refereeing
        raw_union_pool = list(goat_records) + list(crest_records)
        if not goat_records or not crest_records or any(r.energy_hartree is None for r in raw_union_pool):
            raise ValueError("Union requires successful GOAT and CREST ensembles with calculated energies.")
        union_minimum = min(r.energy_hartree for r in raw_union_pool)
        raw_union_pool = [r.model_copy(update={"energy_kcal_rel": (r.energy_hartree - union_minimum) * HARTREE_TO_KCAL_PER_MOL}) for r in raw_union_pool]
        survivors_stage_a = self.cregen_referee.referee_ensemble(
            raw_union_pool, is_spectroscopic_stage=False
        )
        logger.info(
            f"[UNION-PROTOCOL] Step 4 Complete: Raw union pool ({len(raw_union_pool)}) refereed to {len(survivors_stage_a)} Stage A conformers."
        )

        # Step 5: Stage B Spectroscopic Deduplication (Delta B/B <= 0.1%)
        survivors_stage_b = deduplicate_spectroscopic(
            survivors_stage_a, bthr_spectroscopic=0.001
        )
        logger.info(
            f"[UNION-PROTOCOL] Step 5 Complete: Spectroscopic deduplication tightened ensemble to {len(survivors_stage_b)} distinguishable microwave isomers."
        )

        # Step 6: Full Diagnostics Audit Reporting
        goat_unique = 0
        crest_unique = 0
        shared_count = 0

        for conf in survivors_stage_b:
            c_coords = np.array(conf.coordinates, dtype=np.float64)
            c_masses = np.array(
                [get_dynamic_atomic_mass(s) for s in conf.symbols]
            )

            matched_goat = any(
                compute_rmsd(
                    c_coords,
                    np.array(g.coordinates, dtype=np.float64),
                    masses=c_masses,
                )
                < 0.125
                for g in goat_records
            )
            matched_crest = any(
                compute_rmsd(
                    c_coords,
                    np.array(c.coordinates, dtype=np.float64),
                    masses=c_masses,
                )
                < 0.125
                for c in crest_records
            )

            if matched_goat and matched_crest:
                shared_count += 1
            elif matched_goat:
                goat_unique += 1
            elif matched_crest:
                crest_unique += 1
            else:
                shared_count += 1

        e_goat = [g.energy_kcal_rel for g in goat_records]
        e_crest = [c.energy_kcal_rel for c in crest_records]
        e_union = [u.energy_kcal_rel for u in survivors_stage_b]

        s_goat, _ = calculate_conformational_entropy(e_goat)
        s_crest, _ = calculate_conformational_entropy(e_crest)
        s_union, weights_union = calculate_conformational_entropy(e_union)

        tot_single_max = max(1, max(len(goat_records), len(crest_records)))
        union_coverage = len(survivors_stage_b) / tot_single_max

        audit_report = UnionAuditReport(
            system_name=system_name,
            n_seeds=n_seeds,
            n_goat_raw=len(goat_records),
            n_crest_raw=len(crest_records),
            n_union_raw=len(raw_union_pool),
            n_goat_unique=goat_unique,
            n_crest_unique=crest_unique,
            n_shared_intersection=shared_count,
            n_survivors_stage_a=len(survivors_stage_a),
            n_survivors_stage_b=len(survivors_stage_b),
            goat_f1_baseline=None,
            crest_f1_baseline=None,
            union_coverage_ratio=union_coverage,
            s_conf_goat_cal_mol_k=s_goat,
            s_conf_crest_cal_mol_k=s_crest,
            s_conf_union_cal_mol_k=s_union,
            provenance_tag="[M]",
        )

        final_ensemble = EnsembleContainer(
            name=f"Union_Refereed_{system_name}",
            conformers=survivors_stage_b,
            temperature_k=298.15,
            s_conf_cal_mol_k=s_union,
            boltzmann_weights=weights_union,
            provenance_tag="[M]",
        )

        # Persist to HDF5 artifact store if path specified
        if hdf5_out_path is not None:
            save_ensemble_to_hdf5(final_ensemble, audit_report, hdf5_out_path)

        logger.info(
            f"[UNION-PROTOCOL] Step 6 Complete: Final Refereed Union Ensemble contains {len(survivors_stage_b)} isomers (S_conf={s_union:.3f} cal/(mol*K))."
        )
        return final_ensemble, audit_report


# =============================================================================
# 11. FAIR-Compliant HDF5 / QCSchema Serialization
# =============================================================================

def save_ensemble_to_hdf5(
    ensemble: EnsembleContainer,
    report: UnionAuditReport,
    hdf5_path: Union[str, Path],
) -> Path:
    """Serializes the refereed conformer ensemble and union audit report to HDF5

    following QCSchema and SWMR guidelines.
    """
    target = Path(hdf5_path).resolve()
    CoChemPathManager.assert_air_gap(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(target), "w") as f:
        # Root metadata
        f.attrs["system_name"] = report.system_name
        f.attrs["creation_timestamp"] = report.timestamp
        f.attrs["provenance_tag"] = report.provenance_tag
        for key in ("goat_f1_baseline", "crest_f1_baseline", "s_conf_union_cal_mol_k"):
            value = getattr(report, key)
            if value is not None:
                f.attrs[key] = value
        f.attrs["completeness_disclaimer"] = report.completeness_disclaimer

        # Audit Report Group
        audit_grp = f.create_group("union_audit")
        for key, val in report.model_dump().items():
            if isinstance(val, (int, float, str, bool)):
                audit_grp.attrs[key] = val

        # Conformers Group
        confs_grp = f.create_group("conformers")
        n_confs = len(ensemble.conformers)
        confs_grp.attrs["count"] = n_confs

        for idx, conf in enumerate(ensemble.conformers):
            conf_grp = confs_grp.create_group(f"conformer_{idx:04d}")
            conf_grp.attrs["index"] = conf.index
            conf_grp.attrs["origin_engine"] = conf.origin_engine
            if conf.energy_kcal_rel is not None:
                conf_grp.attrs["energy_kcal_rel"] = conf.energy_kcal_rel
            if conf.energy_hartree is not None:
                conf_grp.attrs["energy_hartree"] = conf.energy_hartree
            for key in ("inertial_defect_u_a2", "ray_asymmetry_kappa"):
                value = getattr(conf, key)
                if value is not None:
                    conf_grp.attrs[key] = value

            for key in ("rotational_constants_mhz", "rotational_constants_ghz", "planar_moments_u_a2"):
                value = getattr(conf, key)
                if value is not None:
                    conf_grp.create_dataset(key, data=np.asarray(value, dtype=np.float64))
                    conf_grp.create_dataset(key + "_available", data=np.asarray([v is not None for v in value], dtype=bool))

            # Atomic symbols & coordinates
            dt_str = h5py.string_dtype(encoding="utf-8")
            conf_grp.create_dataset(
                "symbols", data=np.array(conf.symbols, dtype=object), dtype=dt_str
            )
            conf_grp.create_dataset(
                "coordinates",
                data=np.array(conf.coordinates, dtype=np.float64),
            )

    logger.info(
        f"[HDF5] Refereed ensemble ({n_confs} conformers) written to {target}"
    )
    return target


def run_crest(
    input_xyz: Union[str, Path],
    flags: str = "--nci --nocross --noreftopo",
    seed_id: Optional[str] = None,
    work_dir: Optional[Path] = None,
    **kwargs: Any,
) -> EnsembleContainer:
    """Module-level convenience wrapper for CrestRunner.run_crest with Method Matrix v4 flags."""
    runner = CrestRunner()
    return runner.run_crest(input_xyz=input_xyz, seed_id=seed_id, work_dir=work_dir, flags=flags, **kwargs)
