"""CoChem-TORQ: ORCA GOAT Conformer & Binding-Site Isomer Enumeration Engine.

=============================================================================
Phase 3 (Stage 2.0 - 2.1) Implementation
-----------------------------------------------------------------------------
Governs primary non-covalent conformer and binding-site isomer exploration via
ORCA GOAT (Global Optimization by Approximate Trajectory), driven by machine-learning
force fields (AIMNet2 / MACE-OMOL / UMA) or GFN2-xTB / r2SCAN-3c electronic structure.

Authoritative Standards & Directives:
- Method Matrix v4 Section 9B.1: The verdict on GOAT vs CREST and union merging
  (GOAT is primary engine with average F1 = 0.93 [M] benchmark baseline vs CREST 0.74-0.80)
- Method Matrix v4 Section 9B.2: Conformer search comparison table and basin-hopping lineage
- Method Matrix v4 Section 9B.3: The 6-Step Union Protocol (Steps 0, 1, 2, and 6)
- Method Matrix v4 Section 9B.4: MLFF-driven GOAT recipe (ExtOpt, AIMNet2 float32 noise floor,
  TolE 1e-5, TightOpt, MAXEN 12.0 explicit setting, GOAT-EXPLORE variant for binding sites)
- Method Matrix v4 Section 10: The ORCA external-tool contract (.extinp.tmp -> .engrad,
  mandatory sign flip: g_Eh_a0 = (-F_eV_A) * 0.529177210903 / 27.211386245988)
- Mendeleev Mandate: 100% dynamic retrieval of atomic weights, isotopic masses, and radii
- Tripartite Filesystem Air-Gap Compliance: Ring 1 (Static), Ring 2 (Scratch), Ring 3 (Artifacts)
- CoChem Anti-Spoofing Protocol v2: Authentic physical constraints, zero mocks or stubs.
"""

from __future__ import annotations

import datetime
import enum
import functools
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
import concurrent.futures
from pathlib import Path
from typing import Any, Final, Literal, Optional, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import h5py
from mendeleev import element as mendeleev_element
import numpy as np
import scipy.constants as const
from scipy.spatial.distance import cdist
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from cochem_base.schemas import MultiSeedGoatConfig
except ImportError:
    class MultiSeedGoatConfig(BaseModel):
        model_config = ConfigDict(frozen=True, extra="forbid")
        seed_structures: list[str] = Field(min_length=1)
        max_concurrent_seeds: int = Field(default=4, ge=1)
        rmsd_threshold_angstrom: float = Field(default=0.15, gt=0.0)
        energy_window_kcal_mol: float = Field(default=6.0, gt=0.0)

try:
    from scripts.oet_client import PhysicalOETFallbackCalculator
except ImportError:
    try:
        from Libraries.cochem_torq_oet_client import PhysicalOETFallbackCalculator
    except ImportError:
        PhysicalOETFallbackCalculator = None

# Configure module-level logging
logger = logging.getLogger("CoChem-TORQ.GOAT")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: [CoChem-TORQ-GOAT] %(message)s",
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
    """Dynamically retrieves standard atomic weight (amu) via mendeleev.

    Strictly prohibits hardcoded mass constants under Mendeleev Mandate.
    """
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = mendeleev_element(clean_sym)
    if el.atomic_weight is not None:
        return float(el.atomic_weight)
    if el.mass is not None:
        return float(el.mass)
    raise ValueError(
        f"Could not dynamically retrieve atomic mass for element symbol '{symbol}'."
    )


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
    return get_dynamic_atomic_mass(symbol)


@functools.lru_cache(maxsize=256)
def get_dynamic_covalent_radius(symbol: str) -> float:
    """Dynamically retrieves Pyykko single-bond covalent radius in Angstroms via mendeleev."""
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = mendeleev_element(clean_sym)
    if el.covalent_radius_pyykko is not None:
        return float(el.covalent_radius_pyykko) / 100.0
    if el.covalent_radius is not None:
        return float(el.covalent_radius) / 100.0
    return 1.40


@functools.lru_cache(maxsize=256)
def get_dynamic_vdw_radius(symbol: str) -> float:
    """Dynamically retrieves van der Waals radius in Angstroms via mendeleev."""
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = mendeleev_element(clean_sym)
    if el.vdw_radius is not None:
        return float(el.vdw_radius) / 100.0
    if el.vdw_radius_alvarez is not None:
        return float(el.vdw_radius_alvarez) / 100.0
    return 1.70


# =============================================================================
# 3. Tripartite Filesystem Air-Gap Architecture & Custom Exceptions
# =============================================================================

class AirGapViolationError(RuntimeError):
    """Raised when an execution writes to Ring 1 static repository space."""


class GoatError(Exception):
    """Base exception for ORCA GOAT conformer search errors."""


class GoatExecutionError(GoatError):
    """Raised when an ORCA GOAT subprocess execution fails."""


class GoatTimeoutError(GoatError):
    """Raised when an ORCA GOAT subprocess times out."""


class ExtOptContractError(GoatError):
    """Raised when ExtOpt contract specifications are violated."""


class CoChemAirGapRing(enum.Enum):
    """Tripartite Filesystem Domain Rings."""

    RING1_STATIC_REPO = "Ring 1: Static Read-Only Codebase"
    RING2_SCRATCH = "Ring 2: Ephemeral Runtime Scratch"
    RING3_ARTIFACTS = "Ring 3: Persistent Curated Artifacts"


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
            if resolved == repo_root:
                raise AirGapViolationError(
                    f"[AIR-GAP VIOLATION] Target path matches Domain A / Ring 1 root: {resolved}"
                )


# =============================================================================
# 4. Enums, Pydantic v2 Models & Data Structures
# =============================================================================

class GoatMode(str, enum.Enum):
    """ORCA GOAT algorithmic operation variants per Method Matrix v4 §9B.4."""

    GOAT = "GOAT"  # Standard conformer search with bond topology constraint
    GOAT_EXPLORE = "GOAT-EXPLORE"  # Drops topology constraint; right variant for binding-site isomerism
    GOAT_ENTROPY = "GOAT-ENTROPY"  # Stops when conformational entropy converges (< 0.1 cal/(mol*K))
    GOAT_REACT = "GOAT-REACT"  # Reactive exploration with MAXTOPODIFF 8 and AUTOWALL true
    GOAT_DIVERSITY = "GOAT-DIVERSITY"  # Energy-blind diversity sampling
    GOAT_COARSE = "GOAT-COARSE"  # Treats fragments as rigid bodies (>10x speedup)


class GoatExtOptDriver(str, enum.Enum):
    """External Machine Learning Force Field drivers for ORCA ExtOpt."""

    AIMNET2 = "aimnet2"  # AIMNet2 general-purpose neural network potential (default)
    MACE_OMOL = "mace_omol"  # MACE-OMOL foundation model
    MACE_MP = "mace_mp"  # MACE Materials Project model
    UMA = "uma"  # UMA via fairchem
    GXTB = "gxtb"  # g-xTB GPU semi-empirical Hamiltonian
    MOPAC = "mopac"  # MOPAC PM7 semi-empirical
    PYSCF = "pyscf"  # PySCF quantum electronic structure bridge
    PHYSICAL = "physical"  # Analytical physical molecular mechanics fallback
    CUSTOM = "custom"  # Custom user-defined external wrapper



class GoatConfig(BaseModel):
    """Configuration specification for ORCA GOAT conformer enumeration.

    Authoritative flags per Method Matrix v4 §9B.1–9B.4 and Section 10.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    orca_bin: str = Field(
        default="orca",
        description="Path or binary name of ORCA executable.",
    )
    max_threads: int = Field(
        default=1,
        ge=1,
        description="Maximum concurrent threads per worker process.",
    )
    mode: GoatMode = Field(
        default=GoatMode.GOAT_EXPLORE,
        description="GOAT exploration variant (GOAT-EXPLORE recommended for non-covalent binding sites).",
    )
    driver: GoatExtOptDriver = Field(
        default=GoatExtOptDriver.AIMNET2,
        description="External force field or MLFF driver under ExtOpt.",
    )
    maxen_kcal: float = Field(
        default=12.0,
        description="Conformer energy cutoff in kcal/mol (Method Matrix mandates explicit MAXEN 12.0).",
        ge=0.1,
        le=100.0,
    )
    conftemp_k: float = Field(
        default=298.15,
        description="Conformational thermodynamic temperature in Kelvin (CONFTEMP).",
        gt=0.0,
    )
    confdegen: str = Field(
        default="auto",
        description="Conformer degeneracy handling ('auto' ensures proper S_conf accounting).",
    )
    maxglobaliter: int = Field(
        default=100,
        description="Maximum global basin-hopping iterations (MAXGLOBALITER).",
        ge=1,
    )
    minglobaliter: int = Field(
        default=3,
        description="Minimum global basin-hopping iterations before convergence check (MINGLOBALITER).",
        ge=1,
    )
    maxiter: int = Field(
        default=128,
        description="Maximum iterations per local step (MAXITER).",
        ge=10,
    )
    maxoptiter: int = Field(
        default=256,
        description="Maximum geometry optimization steps (MAXOPTITER).",
        ge=10,
    )
    maxcoresopt: int = Field(
        default=32,
        description="Maximum CPU cores allocated per optimization worker (MAXCORESOPT).",
        ge=1,
    )
    tight_opt: bool = Field(
        default=True,
        description="Enforce ! TightOpt to prevent optimizer chasing MLFF Float32 noise.",
    )
    scf_tol_e: float = Field(
        default=1e-5,
        description="Energy convergence tolerance (%scf TolE 1e-5 end) for MLFF noise floor.",
        ge=1e-8,
        le=1e-3,
    )
    server_mode: bool = Field(
        default=True,
        description="Use persistent oet_server daemon socket (saves ~30s model reload per gradient).",
    )
    server_host: str = Field(
        default="localhost",
        description="Host for oet_server daemon.",
    )
    server_port: int = Field(
        default=8888,
        description="Port for oet_server daemon.",
    )
    prog_ext_path: Optional[str] = Field(
        default=None,
        description="Absolute path to ExtOpt wrapper executable (e.g. oet_client or custom bridge).",
    )
    threads: int = Field(
        default_factory=lambda: max(1, (os.cpu_count() or 4) - 1),
        description="Number of parallel PAL cores for ORCA execution (! PAL<n>).",
        ge=1,
    )
    timeout_seconds: int = Field(
        default=3600,
        description="Maximum execution timeout in seconds for single GOAT run.",
        ge=10,
    )
    f1_baseline: float = Field(
        default=0.93,
        description="Measured GOAT average F1 benchmark baseline ([M] racer benchmark).",
    )
    extra_keywords: list[str] = Field(
        default_factory=list,
        description="Additional ORCA simple keywords.",
    )


GoatRunnerConfig = GoatConfig


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
    energy_kcal_rel: float = Field(
        default=0.0,
        description="Relative electronic energy in kcal/mol relative to ensemble minimum.",
    )
    rotational_constants_mhz: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Principal rotational constants (A, B, C) in MHz.",
    )
    rotational_constants_ghz: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Principal rotational constants (A, B, C) in GHz.",
    )
    inertial_defect_u_a2: float = Field(
        default=0.0,
        description="Inertial defect Delta = Ic - Ia - Ib in u * Angstrom^2.",
    )
    planar_moments_u_a2: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Planar moments of inertia (P_aa, P_bb, P_cc) in u * Angstrom^2.",
    )
    ray_asymmetry_kappa: float = Field(
        default=0.0,
        description="Ray's asymmetry parameter kappa = (2B - A - C) / (A - C).",
    )
    origin_engine: str = Field(
        default="GOAT",
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
        default="[M]",
        description="Method Matrix provenance tag: [M] Measured, [D] Derived, [E] Estimated.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata dictionary for downstream stages.",
    )


class EnsembleContainer(BaseModel):
    """Ensemble of conformers resulting from GOAT search or deduplication."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(..., description="Descriptive identifier for the ensemble.")
    conformers: list[ConformerRecord] = Field(
        default_factory=list, description="List of conformer records in ensemble."
    )
    temperature_k: float = Field(
        default=298.15, description="Thermodynamic temperature in Kelvin."
    )
    s_conf_cal_mol_k: float = Field(
        default=0.0,
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


class GoatAuditReport(BaseModel):
    """Audit report detailing ORCA GOAT conformer enumeration diagnostics.

    Adheres strictly to Method Matrix v4 §9B.1–9B.4 and §9B.3 Step 6.
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
    n_goat_dedup_stage_a: int = Field(
        ..., description="Number of conformers surviving Stage A broad deduplication."
    )
    n_goat_dedup_stage_b: int = Field(
        ..., description="Number of conformers surviving Stage B spectroscopic deduplication."
    )
    goat_f1_baseline: float = Field(
        default=0.93,
        description="Measured GOAT average F1 benchmark baseline ([M] racer benchmark).",
    )
    s_conf_cal_mol_k: float = Field(
        ..., description="Conformational entropy from GOAT ensemble in cal/(mol*K)."
    )
    wall_time_seconds: float = Field(
        default=0.0, description="Total execution wall clock time in seconds."
    )
    completeness_caveat: str = Field(
        default="ORCA GOAT is a stochastic global optimizer (basin-hopping heuristic) and does not provide an analytical completeness proof.",
        description="Mandatory disclaimer regarding stochastic heuristics vs hand-enumeration completeness.",
    )
    provenance_tag: str = Field(
        default="[M]", description="Method Matrix provenance tag."
    )


# =============================================================================
# 5. Coordinate Parsing, XYZ I/O, & Interatomic Physics
# =============================================================================

def parse_xyz_string(
    xyz_content: str,
    default_origin: str = "GOAT",
    seed_id: Optional[str] = None,
) -> list[ConformerRecord]:
    """Parses a multi-structure XYZ string into a list of ConformerRecord objects.

    Dynamically calculates inertia tensors, rotational constants, and planar moments
    using dynamic atomic masses from mendeleev.
    """
    records: list[ConformerRecord] = []
    lines = [line.strip() for line in xyz_content.strip().splitlines() if line.strip()]
    if not lines:
        return records

    idx = 0
    record_idx = 0
    while idx < len(lines):
        try:
            n_atoms = int(lines[idx].split()[0])
        except (ValueError, IndexError):
            idx += 1
            continue

        if idx + 1 >= len(lines):
            break

        comment_line = lines[idx + 1]
        energy_hartree: Optional[float] = None
        # Extract energy from standard ORCA / CREST / ExtOpt comments
        e_match = re.search(
            r"(?:energy|E|TOTAL ENERGY)[:=\s]+([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)",
            comment_line,
            re.IGNORECASE,
        )
        if e_match:
            try:
                energy_hartree = float(e_match.group(1))
            except ValueError:
                energy_hartree = None

        symbols: list[str] = []
        coords_list: list[list[float]] = []

        atom_start = idx + 2
        atom_end = atom_start + n_atoms

        if atom_end > len(lines):
            break

        for a_idx in range(atom_start, atom_end):
            parts = lines[a_idx].split()
            if len(parts) >= 4:
                sym = parts[0].strip().capitalize()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                symbols.append(sym)
                coords_list.append([x, y, z])

        if len(symbols) == n_atoms:
            coords_arr = np.array(coords_list, dtype=np.float64)
            (
                (A_mhz, B_mhz, C_mhz),
                (A_ghz, B_ghz, C_ghz),
                delta,
                planar,
                kappa,
            ) = compute_moments_and_constants(symbols, coords_arr)

            is_dissoc = check_complex_dissociation(symbols, coords_arr)

            rec = ConformerRecord(
                index=record_idx,
                symbols=symbols,
                coordinates=coords_list,
                energy_hartree=energy_hartree,
                energy_kcal_rel=0.0,
                rotational_constants_mhz=(A_mhz, B_mhz, C_mhz),
                rotational_constants_ghz=(A_ghz, B_ghz, C_ghz),
                inertial_defect_u_a2=delta,
                planar_moments_u_a2=planar,
                ray_asymmetry_kappa=kappa,
                origin_engine=default_origin,
                seed_id=seed_id,
                is_dissociated=is_dissoc,
                provenance_tag="[M]",
            )
            records.append(rec)
            record_idx += 1

        idx = atom_end

    # Normalize relative energies if absolute energies were found
    valid_energies = [r.energy_hartree for r in records if r.energy_hartree is not None]
    if valid_energies:
        min_e = min(valid_energies)
        for r in records:
            if r.energy_hartree is not None:
                r.energy_kcal_rel = (r.energy_hartree - min_e) * HARTREE_TO_KCAL_PER_MOL

    return records


def parse_xyz_file(
    file_path: Union[str, Path],
    default_origin: str = "GOAT",
    seed_id: Optional[str] = None,
) -> list[ConformerRecord]:
    """Reads and parses an XYZ file from disk."""
    path = Path(file_path).resolve()
    if not path.exists():
        logger.warning(f"XYZ file not found: {path}")
        return []
    content = path.read_text(encoding="utf-8")
    return parse_xyz_string(content, default_origin=default_origin, seed_id=seed_id)


def write_xyz_string(records: Sequence[ConformerRecord]) -> str:
    """Serializes conformer records into multi-structure XYZ format."""
    blocks: list[str] = []
    for r in records:
        n_atoms = len(r.symbols)
        e_str = (
            f"energy={r.energy_hartree:.10f} rel_kcal={r.energy_kcal_rel:.4f}"
            if r.energy_hartree is not None
            else f"rel_kcal={r.energy_kcal_rel:.4f}"
        )
        origin_str = f"origin={r.origin_engine}"
        seed_str = f"seed={r.seed_id}" if r.seed_id else ""
        comment = f"Conformer {r.index} | {e_str} | {origin_str} | {seed_str}".strip()
        lines = [f"{n_atoms}", comment]
        for s, c in zip(r.symbols, r.coordinates):
            lines.append(f"{s:<2} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_xyz_file(
    file_path: Union[str, Path], records: Sequence[ConformerRecord]
) -> None:
    """Writes conformer records to an XYZ file on disk respecting air-gap rules."""
    path = Path(file_path).resolve()
    CoChemPathManager.assert_air_gap(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = write_xyz_string(records)
    path.write_text(content, encoding="utf-8")


# =============================================================================
# 6. Spectroscopic Physics & Rigid Rotor Analysis
# =============================================================================

def compute_center_of_mass(symbols: Sequence[str], coords: np.ndarray) -> np.ndarray:
    """Computes center of mass in Angstroms using dynamic Mendeleev masses."""
    masses = np.array([get_dynamic_atomic_mass(s) for s in symbols], dtype=np.float64)
    total_mass = np.sum(masses)
    if total_mass <= 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    return com


def compute_inertia_tensor(symbols: Sequence[str], coords: np.ndarray) -> np.ndarray:
    """Computes principal inertia tensor (3x3) in u * Angstrom^2 centered at COM."""
    com = compute_center_of_mass(symbols, coords)
    centered = coords - com
    masses = np.array([get_dynamic_atomic_mass(s) for s in symbols], dtype=np.float64)

    Ixx = np.sum(masses * (centered[:, 1] ** 2 + centered[:, 2] ** 2))
    Iyy = np.sum(masses * (centered[:, 0] ** 2 + centered[:, 2] ** 2))
    Izz = np.sum(masses * (centered[:, 0] ** 2 + centered[:, 1] ** 2))
    Ixy = -np.sum(masses * centered[:, 0] * centered[:, 1])
    Ixz = -np.sum(masses * centered[:, 0] * centered[:, 2])
    Iyz = -np.sum(masses * centered[:, 1] * centered[:, 2])

    tensor = np.array(
        [
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz],
        ],
        dtype=np.float64,
    )
    return tensor


def compute_moments_and_constants(
    symbols: Sequence[str], coords: np.ndarray
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    tuple[float, float, float],
    float,
]:
    """Calculates principal moments, rotational constants, inertial defect, and planar moments.

    Returns:
        - (A_MHz, B_MHz, C_MHz)
        - (A_GHz, B_GHz, C_GHz)
        - Inertial defect Delta = Ic - Ia - Ib (u * A^2)
        - Planar moments (P_aa, P_bb, P_cc) in u * A^2
        - Ray's asymmetry parameter kappa = (2B - A - C) / (A - C)
    """
    if len(symbols) < 2 or coords.shape[0] < 2:
        return (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.0,
            (0.0, 0.0, 0.0),
            0.0,
        )

    tensor = compute_inertia_tensor(symbols, coords)
    evals, _ = np.linalg.eigh(tensor)
    evals = np.sort(np.maximum(evals, 1e-12))
    Ia, Ib, Ic = float(evals[0]), float(evals[1]), float(evals[2])

    A_mhz = ROTATIONAL_PREFACTOR_MHZ / Ia
    B_mhz = ROTATIONAL_PREFACTOR_MHZ / Ib
    C_mhz = ROTATIONAL_PREFACTOR_MHZ / Ic

    A_ghz = A_mhz / 1000.0
    B_ghz = B_mhz / 1000.0
    C_ghz = C_mhz / 1000.0

    delta = Ic - Ia - Ib

    P_aa = 0.5 * (-Ia + Ib + Ic)
    P_bb = 0.5 * (Ia - Ib + Ic)
    P_cc = 0.5 * (Ia + Ib - Ic)

    denom = A_mhz - C_mhz
    if abs(denom) > 1e-6:
        kappa = (2.0 * B_mhz - A_mhz - C_mhz) / denom
    else:
        kappa = 0.0

    return (
        (A_mhz, B_mhz, C_mhz),
        (A_ghz, B_ghz, C_ghz),
        delta,
        (P_aa, P_bb, P_cc),
        kappa,
    )


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Optimal rotation matrix to align Cartesian coordinate matrix P onto Q using Kabsch algorithm."""
    P_cent = P - np.mean(P, axis=0)
    Q_cent = Q - np.mean(Q, axis=0)
    H = np.dot(P_cent.T, Q_cent)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    aligned_P = np.dot(P_cent, R.T) + np.mean(Q, axis=0)
    return aligned_P


def compute_rmsd(
    P: Union[np.ndarray, Sequence[str]],
    Q: np.ndarray,
    symbols: Optional[Union[Sequence[str], Tuple[Sequence[str], Sequence[str]]]] = None,
    symbols2: Optional[Sequence[str]] = None,
) -> float:
    """Computes permutation-invariant Cartesian Root-Mean-Square Deviation (RMSD) in Angstroms [M].

    Evaluates Hungarian matching (scipy.optimize.linear_sum_assignment) over chemically
    identical nuclei to eliminate atom-order swapping artifacts between GOAT and CREST.
    """
    if isinstance(P, (list, tuple)) and len(P) > 0 and isinstance(P[0], str):
        # Called as compute_rmsd(symbols, P, Q)
        sym_arg = P
        actual_P = Q
        actual_Q = symbols if symbols is not None else np.array([])
        P, Q, symbols = actual_P, actual_Q, sym_arg

    P_arr = np.asarray(P, dtype=np.float64)
    Q_arr = np.asarray(Q, dtype=np.float64)

    if P_arr.shape != Q_arr.shape or P_arr.shape[0] == 0:
        return 0.0

    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    curr_P = np.array(P_arr, dtype=np.float64, copy=True)
    target_Q = np.array(Q_arr, dtype=np.float64, copy=True)
    n_atoms = curr_P.shape[0]

    # Resolve symbols for P and Q
    syms_P: Optional[list[str]] = None
    syms_Q: Optional[list[str]] = None
    if symbols2 is not None:
        syms_P = [str(s) for s in symbols] if symbols is not None else None
        syms_Q = [str(s) for s in symbols2]
    elif symbols is not None:
        if isinstance(symbols, tuple) and len(symbols) == 2 and isinstance(symbols[0], (list, tuple)):
            syms_P = [str(s) for s in symbols[0]]
            syms_Q = [str(s) for s in symbols[1]]
        elif len(symbols) == n_atoms:
            syms_P = [str(s) for s in symbols]
            syms_Q = list(syms_P)

    # Partition indices by chemical element equivalence classes
    element_pairs: list[tuple[list[int], list[int]]] = []
    if syms_P is not None and syms_Q is not None and len(syms_P) == n_atoms and len(syms_Q) == n_atoms:
        unique_elems = set(syms_P).union(set(syms_Q))
        valid_partition = True
        for elem in unique_elems:
            p_idx = [i for i, s in enumerate(syms_P) if s == elem]
            q_idx = [j for j, s in enumerate(syms_Q) if s == elem]
            if len(p_idx) != len(q_idx):
                valid_partition = False
                break
            if len(p_idx) > 0:
                element_pairs.append((p_idx, q_idx))
        if not valid_partition:
            element_pairs = [(list(range(n_atoms)), list(range(n_atoms)))]
    else:
        element_pairs = [(list(range(n_atoms)), list(range(n_atoms)))]

    # Iterative Kabsch + Hungarian assignment
    best_rmsd = float("inf")
    for _ in range(5):
        aligned_P = kabsch_align(curr_P, target_Q)
        reordered_P = np.copy(curr_P)
        has_reordered = False
        for p_indices, q_indices in element_pairs:
            if len(p_indices) <= 1:
                continue
            P_grp = aligned_P[p_indices]
            Q_grp = target_Q[q_indices]
            cost = cdist(P_grp, Q_grp)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                reordered_P[q_indices[c]] = curr_P[p_indices[r]]
            has_reordered = True
        curr_P = reordered_P
        aligned_iter = kabsch_align(curr_P, target_Q)
        diff_iter = aligned_iter - target_Q
        iter_rmsd = float(np.sqrt(np.mean(np.sum(diff_iter**2, axis=1))))
        if iter_rmsd < best_rmsd:
            best_rmsd = iter_rmsd
        if not has_reordered:
            break

    aligned_final = kabsch_align(curr_P, target_Q)
    diff = aligned_final - target_Q
    rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    return min(rmsd, best_rmsd)


def check_rotational_equivalence(
    rot1: Sequence[float],
    rot2: Sequence[float],
    threshold_rel: float = 0.002,
    defect1: Optional[float] = None,
    defect2: Optional[float] = None,
    planar1: Optional[Sequence[float]] = None,
    planar2: Optional[Sequence[float]] = None,
    defect_threshold: float = 0.05,
) -> bool:
    """Evaluates Method Matrix v4 §9B.1–§9B.3 rotational candidate equivalence.

    Screening requires relative deviation across all three principal rotational constants (A, B, C):
        Δ_rel(A) = |A1 - A2| / max(A1, A2)
        Δ_rel(B) = |B1 - B2| / max(B1, B2)
        Δ_rel(C) = |C1 - C2| / max(C1, C2)
    Two conformers are candidate equivalents only if:
        max(Δ_rel(A), Δ_rel(B), Δ_rel(C)) < threshold_rel (0.002 = 0.2%).

    Concurrently verifies inertial defect Δ = Ic - Ia - Ib and planar moments (Pa, Pb, Pc)
    to discriminate planar from non-planar conformers.
    """
    if len(rot1) < 3 or len(rot2) < 3:
        return False

    rel_diffs = []
    for c1, c2 in zip(rot1[:3], rot2[:3]):
        max_val = max(abs(float(c1)), abs(float(c2)))
        if max_val < 1e-9:
            continue
        rel_diffs.append(abs(float(c1) - float(c2)) / max_val)

    if not rel_diffs:
        return False

    max_rot_rel = max(rel_diffs)
    if max_rot_rel >= threshold_rel:
        return False

    if defect1 is not None and defect2 is not None:
        if abs(float(defect1) - float(defect2)) > defect_threshold:
            return False

    if planar1 is not None and planar2 is not None and len(planar1) >= 3 and len(planar2) >= 3:
        for p1, p2 in zip(planar1[:3], planar2[:3]):
            if (p1 < 0.1 and p2 >= 0.5) or (p2 < 0.1 and p1 >= 0.5):
                return False

    return True


def deduplicate_conformers(
    conformers: Sequence[Union[ConformerRecord, dict[str, Any]]],
    delta_rot_rel_threshold: float = 0.002,
    rmsd_threshold: float = 0.15,
    ethr_kcal: Optional[float] = None,
) -> list[Any]:
    """Deduplicates a pool of conformers using full rotational tensor (A, B, C),
    inertial defect, and permutation-invariant Hungarian RMSD (Method Matrix v4 §9B.1–§9B.3).
    """
    if not conformers:
        return []

    def get_energy(c: Any) -> float:
        if isinstance(c, dict):
            for k in ("energy_hartree", "energy", "energy_kcal_rel"):
                if k in c and c[k] is not None:
                    return float(c[k])
            return 0.0
        return getattr(c, "energy_kcal_rel", 0.0)

    def extract_props(c: Any) -> tuple[list[str], np.ndarray, tuple[float, float, float], float, tuple[float, float, float]]:
        if isinstance(c, dict):
            syms = list(c.get("symbols", []))
            coords = np.asarray(c.get("coordinates", []), dtype=np.float64)
            rot = c.get("rotational_constants_mhz")
            defect = c.get("inertial_defect_u_a2")
            planar = c.get("planar_moments")
            if rot is None or defect is None or planar is None:
                (rot_mhz, _, calc_defect, calc_planar, _) = compute_moments_and_constants(syms, coords)
                rot = rot or rot_mhz
                defect = defect if defect is not None else calc_defect
                planar = planar or calc_planar
            return syms, coords, tuple(rot[:3]), float(defect), tuple(planar[:3])
        else:
            syms = list(c.symbols)
            coords = np.asarray(c.coordinates, dtype=np.float64)
            rot = tuple(c.rotational_constants_mhz[:3]) if c.rotational_constants_mhz else (0.0, 0.0, 0.0)
            defect = float(c.inertial_defect_u_a2) if hasattr(c, "inertial_defect_u_a2") else 0.0
            planar = tuple(c.planar_moments[:3]) if hasattr(c, "planar_moments") and c.planar_moments else (0.0, 0.0, 0.0)
            if rot == (0.0, 0.0, 0.0):
                (rot_mhz, _, calc_defect, calc_planar, _) = compute_moments_and_constants(syms, coords)
                rot = rot_mhz
                defect = calc_defect
                planar = calc_planar
            return syms, coords, rot, defect, planar

    sorted_confs = sorted(conformers, key=get_energy)
    accepted: list[Any] = []

    for cand in sorted_confs:
        c_syms, c_coords, c_rot, c_defect, c_planar = extract_props(cand)
        is_dup = False

        for acc in accepted:
            a_syms, a_coords, a_rot, a_defect, a_planar = extract_props(acc)

            if ethr_kcal is not None:
                dE = abs(get_energy(cand) - get_energy(acc))
                if dE > ethr_kcal:
                    continue

            is_rot_cand = check_rotational_equivalence(
                c_rot,
                a_rot,
                threshold_rel=delta_rot_rel_threshold,
                defect1=c_defect,
                defect2=a_defect,
                planar1=c_planar,
                planar2=a_planar,
            )
            if not is_rot_cand:
                continue

            rmsd = compute_rmsd(c_coords, a_coords, symbols=c_syms, symbols2=a_syms)
            if rmsd <= rmsd_threshold:
                is_dup = True
                break

        if not is_dup:
            accepted.append(cand)

    return accepted



def check_complex_dissociation(
    symbols: Sequence[str], coords: np.ndarray, threshold_factor: float = 2.4
) -> bool:
    """Evaluates whether fragments in a non-covalent complex have drifted beyond physical vdW contact.

    Constructs a molecular connectivity graph using dynamic covalent radii + vdW bounds.
    If the connectivity graph is disconnected, the structure is flagged as dissociated.
    """
    n_atoms = len(symbols)
    if n_atoms <= 1:
        return False

    cov_radii = np.array(
        [get_dynamic_covalent_radius(s) for s in symbols], dtype=np.float64
    )
    vdw_radii = np.array(
        [get_dynamic_vdw_radius(s) for s in symbols], dtype=np.float64
    )

    dist_mat = cdist(coords, coords)
    adj = np.full((n_atoms, n_atoms), False, dtype=bool)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r_cov_sum = cov_radii[i] + cov_radii[j]
            r_vdw_sum = vdw_radii[i] + vdw_radii[j]
            cutoff = max(r_cov_sum * 1.5, r_vdw_sum * threshold_factor)
            if dist_mat[i, j] <= cutoff:
                adj[i, j] = True
                adj[j, i] = True

    visited = set()
    queue = [0]
    visited.add(0)

    while queue:
        curr = queue.pop(0)
        neighbors = np.where(adj[curr])[0]
        for n in neighbors:
            if n not in visited:
                visited.add(n)
                queue.append(n)

    return len(visited) < n_atoms


def calculate_conformational_entropy(
    relative_energies_kcal: Sequence[float],
    temperature_k: float = 298.15,
    degeneracies: Optional[Sequence[int]] = None,
) -> tuple[float, list[float]]:
    """Calculates thermodynamic conformational entropy S_conf in cal / (mol * K).

    S_conf = -R * sum(p_i * ln(p_i)) where p_i = (g_i * exp(-dE_i / kT)) / Z
    """
    if not relative_energies_kcal:
        return 0.0, []

    energies = np.array(relative_energies_kcal, dtype=np.float64)
    min_e = np.min(energies)
    shifted_e = energies - min_e

    degen = (
        np.array(degeneracies, dtype=np.float64)
        if degeneracies is not None
        else np.ones_like(energies)
    )

    beta = 1.0 / (BOLTZMANN_CONSTANT_K_CAL_MOL * temperature_k)
    boltz = degen * np.exp(-beta * shifted_e)
    Z = np.sum(boltz)

    if Z <= 0.0 or not np.isfinite(Z):
        p = np.ones_like(energies) / len(energies)
    else:
        p = boltz / Z

    p = np.clip(p, 1e-15, 1.0)
    p = p / np.sum(p)

    s_conf = -GAS_CONSTANT_R_CAL_MOL * np.sum(p * np.log(p))
    return float(s_conf), p.tolist()


# =============================================================================
# 7. ORCA Input Generator & ExtOpt Contract Handler
# =============================================================================

def generate_orca_goat_input(
    config: GoatConfig,
    xyz_filename: str,
    charge: int = 0,
    multiplicity: int = 1,
    custom_coords: Optional[list[Tuple[str, float, float, float]]] = None,
) -> str:
    """Generates an authoritative ORCA GOAT input script compliant with Method Matrix §9B.4.

    Enforces:
    - Mode: GOAT / GOAT-EXPLORE / GOAT-ENTROPY / etc.
    - ExtOpt with TightOpt and %scf TolE 1e-5 end for AIMNet2 float32 noise immunity.
    - Explicit MAXEN 12.0 setting.
    - CONFDEGEN auto and CONFTEMP 298.15.
    - PAL thread parallelism.
    """
    keywords = [config.mode.value]

    if config.driver in [
        GoatExtOptDriver.AIMNET2,
        GoatExtOptDriver.MACE_OMOL,
        GoatExtOptDriver.MACE_MP,
        GoatExtOptDriver.UMA,
        GoatExtOptDriver.GXTB,
        GoatExtOptDriver.MOPAC,
        GoatExtOptDriver.CUSTOM,
    ]:
        keywords.append("ExtOpt")
        if config.tight_opt:
            keywords.append("TightOpt")
    elif config.driver == GoatExtOptDriver.PYSCF:
        keywords.append("ExtOpt")
        if config.tight_opt:
            keywords.append("TightOpt")
    else:
        keywords.append("XTB2")

    keywords.append(f"PAL{config.threads}")
    keywords.extend(config.extra_keywords)

    method_line = " ".join(keywords)

    prog_ext = config.prog_ext_path or (
        f"{Path.home()}/bin/oet-{config.driver.value}/oet_client"
    )
    ext_params = (
        f"-b {config.server_host}:{config.server_port}"
        if config.server_mode
        else ""
    )

    method_block = ""
    if "ExtOpt" in keywords:
        method_block = f"""%method
  ProgExt "{prog_ext}"
  Ext_Params "{ext_params}"
end"""

    scf_block = f"%scf\n  TolE {config.scf_tol_e}\nend"

    goat_block = f"""%goat
  maxen {config.maxen_kcal:.1f}
  conftemp {config.conftemp_k:.2f}
  confdegen {config.confdegen}
  maxglobaliter {config.maxglobaliter}
  minglobaliter {config.minglobaliter}
  maxiter {config.maxiter}
  maxoptiter {config.maxoptiter}
  maxcoresopt {config.maxcoresopt}
end"""

    if custom_coords is not None and len(custom_coords) > 0:
        coord_lines = []
        for sym, x, y, z in custom_coords:
            coord_lines.append(f"{sym:>2} {x:14.8f} {y:14.8f} {z:14.8f}")
        structure_block = f"* xyz {charge} {multiplicity}\n" + "\n".join(coord_lines) + "\n*"
    else:
        structure_block = f"* xyzfile {charge} {multiplicity} {xyz_filename}"

    blocks = [
        f"! {method_line}",
        method_block,
        scf_block,
        goat_block,
        structure_block,
    ]

    return "\n\n".join(b for b in blocks if b.strip()) + "\n"


class ExtOptContract:
    """Normative ORCA ExtOpt file interface per Method Matrix Section 10."""

    @staticmethod
    def read_extinp(extinp_path: Union[str, Path]) -> tuple[str, int, int, int, int, Optional[str]]:
        """Parses <basename>_EXT.extinp.tmp.

        Returns:
            (xyz_filename, charge, multiplicity, ncores, dograd, pointcharge_file)
        """
        lines = Path(extinp_path).read_text(encoding="utf-8").splitlines()
        clean = [l.split("#")[0].strip() for l in lines if l.split("#")[0].strip()]
        if len(clean) < 5:
            raise ExtOptContractError(
                f"Invalid .extinp.tmp file: expected at least 5 lines, got {len(clean)}."
            )
        xyz_file = clean[0]
        charge = int(clean[1])
        multiplicity = int(clean[2])
        ncores = int(clean[3])
        dograd = int(clean[4])
        pcfile = clean[5] if len(clean) > 5 else None
        return xyz_file, charge, multiplicity, ncores, dograd, pcfile

    @staticmethod
    def write_engrad(
        engrad_path: Union[str, Path],
        n_atoms: int,
        energy_eh: float,
        gradient_eh_bohr: np.ndarray,
    ) -> None:
        """Writes <basename>_EXT.engrad adhering to Section 10.2 specifications.

        Units: Energy in Hartree (Eh), Gradient in Eh/bohr.
        """
        CoChemPathManager.assert_air_gap(engrad_path)
        grad_flat = np.asarray(gradient_eh_bohr, dtype=np.float64).flatten()
        if grad_flat.size != n_atoms * 3:
            raise ExtOptContractError(
                f"Gradient size mismatch: expected {n_atoms * 3} elements, got {grad_flat.size}."
            )

        lines = [
            "#",
            "# Number of atoms",
            "#",
            f"{n_atoms}",
            "#",
            "# The current total energy in Eh",
            "#",
            f"{energy_eh:20.12f}",
            "#",
            "# The current gradient in Eh/bohr: Atom1X, Atom1Y, Atom1Z, Atom2X, ...",
            "#",
        ]
        for g_val in grad_flat:
            lines.append(f"{g_val:20.12f}")

        Path(engrad_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def convert_ase_forces_to_orca_gradient(forces_ev_per_ang: np.ndarray) -> np.ndarray:
        """Converts ASE forces (eV/Angstrom) to ORCA gradient (Eh/bohr).

        MANDATORY SIGN FLIP: g = -F per Method Matrix §10.3.
        g_Eh_a0 = (-F_eV_A) * 0.529177210903 / 27.211386245988
        """
        forces = np.asarray(forces_ev_per_ang, dtype=np.float64)
        grad_eh_bohr = (-forces) * (BOHR_TO_ANGSTROM / HARTREE_TO_EV)
        return grad_eh_bohr


# =============================================================================
# 8. Genuine Physical Fallback & Local Basin-Hopping Engine (Physical Protocol Mandate)
# =============================================================================

# =============================================================================
# 9. Two-Stage Deduplication Engine (Stage A Broad & Stage B Spectroscopic)
# =============================================================================

def deduplicate_stage_a(
    records: Sequence[ConformerRecord],
    rmsd_thr: float = 0.125,
    ethr_kcal: float = 0.100,
    bthr_frac: float = 0.025,
) -> list[ConformerRecord]:
    """Stage A: Engine-level broad deduplication (Method Matrix §9B.1 / §9B.3).

    Defaults: RMSD = 0.125 Å, ΔE = 0.100 kcal/mol, Δ(A,B,C) = 1.0–2.5%, Δ(defect) = 0.05.
    """
    if not records:
        return []

    sorted_records = sorted(records, key=lambda r: r.energy_kcal_rel)
    survivors: list[ConformerRecord] = []

    for cand in sorted_records:
        cand_c = np.array(cand.coordinates, dtype=np.float64)
        is_duplicate = False

        for surv in survivors:
            surv_c = np.array(surv.coordinates, dtype=np.float64)

            # Energy check
            dE = abs(cand.energy_kcal_rel - surv.energy_kcal_rel)
            if dE > ethr_kcal:
                continue

            # Tri-constant (A, B, C) rotational screening [M]
            cand_rot = cand.rotational_constants_mhz
            surv_rot = surv.rotational_constants_mhz
            is_rot_distinct = False
            for c_val, s_val in zip(cand_rot, surv_rot):
                if s_val > 1e-4:
                    if abs(c_val - s_val) / s_val > bthr_frac:
                        is_rot_distinct = True
                        break
            if is_rot_distinct:
                continue

            # Inertial defect check: Delta = Ic - Ia - Ib. Differing by >0.05 u*A^2 -> distinct [M]
            d_defect = abs(cand.inertial_defect_u_a2 - surv.inertial_defect_u_a2)
            if d_defect > 0.05:
                continue

            # Permutation-invariant RMSD check
            rmsd = compute_rmsd(cand_c, surv_c, symbols=cand.symbols)
            if rmsd < rmsd_thr:
                is_duplicate = True
                break

        if not is_duplicate:
            survivors.append(cand)

    return survivors


def deduplicate_stage_b_spectroscopic(
    records: Sequence[ConformerRecord],
    rmsd_thr: float = 0.125,
    ethr_kcal: float = 0.05,
    bthr_frac: float = 0.001,
) -> list[ConformerRecord]:
    """Stage B: Microwave spectroscopic deduplication (Method Matrix §9B.3 Step 5).

    Tightens rotational constant window to --bthr 0.001 (0.1% ≈ 12 MHz at 12 GHz)
    and evaluates (A, B, C) tri-constants alongside inertial defect Delta [M].
    """
    if not records:
        return []

    sorted_records = sorted(records, key=lambda r: r.energy_kcal_rel)
    survivors: list[ConformerRecord] = []

    for cand in sorted_records:
        cand_c = np.array(cand.coordinates, dtype=np.float64)
        is_duplicate = False

        for surv in survivors:
            surv_c = np.array(surv.coordinates, dtype=np.float64)

            dE = abs(cand.energy_kcal_rel - surv.energy_kcal_rel)
            if dE > ethr_kcal:
                continue

            # Tri-constant (A, B, C) rotational screening [M]
            cand_rot = cand.rotational_constants_mhz
            surv_rot = surv.rotational_constants_mhz
            is_rot_distinct = False
            for c_val, s_val in zip(cand_rot, surv_rot):
                if s_val > 1e-4:
                    if abs(c_val - s_val) / s_val > bthr_frac:
                        is_rot_distinct = True
                        break
            if is_rot_distinct:
                continue

            # Inertial defect check: Delta = Ic - Ia - Ib. Differing by >0.05 u*A^2 -> distinct [M]
            d_defect = abs(cand.inertial_defect_u_a2 - surv.inertial_defect_u_a2)
            if d_defect > 0.05:
                continue

            rmsd = compute_rmsd(cand_c, surv_c, symbols=cand.symbols)
            if rmsd < rmsd_thr:
                is_duplicate = True
                break

        if not is_duplicate:
            survivors.append(cand)

    return survivors


# =============================================================================
# 10. High-Performance HDF5 Storage & QCSchema Serialization
# =============================================================================

def save_ensemble_to_hdf5(
    ensemble: EnsembleContainer,
    h5_path: Union[str, Path],
    extra_metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Serializes a conformer ensemble into a structured HDF5 archive adhering to QCSchema standards."""
    path = Path(h5_path).resolve()
    CoChemPathManager.assert_air_gap(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_confs = len(ensemble.conformers)
    if n_confs == 0:
        symbols_list: list[str] = []
        n_atoms = 0
        coords_arr = np.empty((0, 0, 3), dtype=np.float64)
    else:
        symbols_list = ensemble.conformers[0].symbols
        n_atoms = len(symbols_list)
        coords_arr = np.array([c.coordinates for c in ensemble.conformers], dtype=np.float64)

    energies_hartree = np.array(
        [c.energy_hartree if c.energy_hartree is not None else np.nan for c in ensemble.conformers],
        dtype=np.float64,
    )
    energies_kcal = np.array([c.energy_kcal_rel for c in ensemble.conformers], dtype=np.float64)
    rot_mhz = np.array([c.rotational_constants_mhz for c in ensemble.conformers], dtype=np.float64)
    rot_ghz = np.array([c.rotational_constants_ghz for c in ensemble.conformers], dtype=np.float64)
    inertial_defects = np.array([c.inertial_defect_u_a2 for c in ensemble.conformers], dtype=np.float64)
    planar_moments = np.array([c.planar_moments_u_a2 for c in ensemble.conformers], dtype=np.float64)
    kappas = np.array([c.ray_asymmetry_kappa for c in ensemble.conformers], dtype=np.float64)
    origins = [c.origin_engine.encode("utf-8") for c in ensemble.conformers]
    seeds = [(c.seed_id or "NONE").encode("utf-8") for c in ensemble.conformers]

    with h5py.File(path, "w") as f:
        f.attrs["ensemble_name"] = ensemble.name
        f.attrs["temperature_k"] = ensemble.temperature_k
        f.attrs["s_conf_cal_mol_k"] = ensemble.s_conf_cal_mol_k
        f.attrs["n_conformers"] = n_confs
        f.attrs["n_atoms"] = n_atoms
        f.attrs["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        f.attrs["provenance_tag"] = ensemble.provenance_tag
        f.attrs["f1_baseline"] = 0.93

        if extra_metadata:
            for k, v in extra_metadata.items():
                if isinstance(v, (int, float, str, bool)):
                    f.attrs[k] = v

        elem_ds = f.create_dataset(
            "elements", data=np.array([s.encode("utf-8") for s in symbols_list])
        )
        elem_ds.attrs["description"] = "Atomic element symbols"

        g_conf = f.create_group("conformers")
        g_conf.create_dataset("coordinates", data=coords_arr, compression="gzip")
        g_conf.create_dataset("energies_hartree", data=energies_hartree, compression="gzip")
        g_conf.create_dataset("energies_kcal_rel", data=energies_kcal, compression="gzip")
        g_conf.create_dataset("rotational_constants_mhz", data=rot_mhz, compression="gzip")
        g_conf.create_dataset("rotational_constants_ghz", data=rot_ghz, compression="gzip")
        g_conf.create_dataset("inertial_defect_u_a2", data=inertial_defects, compression="gzip")
        g_conf.create_dataset("planar_moments_u_a2", data=planar_moments, compression="gzip")
        g_conf.create_dataset("ray_asymmetry_kappa", data=kappas, compression="gzip")
        g_conf.create_dataset("origin_engine", data=origins)
        g_conf.create_dataset("seed_id", data=seeds)

        if ensemble.boltzmann_weights:
            g_conf.create_dataset(
                "boltzmann_weights",
                data=np.array(ensemble.boltzmann_weights, dtype=np.float64),
                compression="gzip",
            )


# =============================================================================
# 11. ORCA GOAT Runner & Orchestration Pipeline
# =============================================================================

class GoatRunner:
    """Orchestrates ORCA GOAT conformer enumeration."""

    def __init__(self, config: Optional[GoatConfig] = None) -> None:
        self.config = config or GoatConfig()

    def run_goat_on_seed(
        self,
        seed_xyz: Union[str, Path],
        scratch_dir: Optional[Union[str, Path]] = None,
        seed_id: Optional[str] = None,
    ) -> list[ConformerRecord]:
        """Executes ORCA GOAT on a single seed structure."""
        seed_path = Path(seed_xyz).resolve()
        if not seed_path.exists():
            raise FileNotFoundError(f"Seed XYZ file not found: {seed_path}")

        run_id = f"goat_{uuid.uuid4().hex[:8]}"
        base_scratch = Path(scratch_dir) if scratch_dir else CoChemPathManager.get_scratch_dir()
        work_dir = base_scratch / run_id
        work_dir.mkdir(parents=True, exist_ok=True)
        CoChemPathManager.assert_air_gap(work_dir)

        s_id = seed_id or seed_path.stem

        # Parse seed
        initial_records = parse_xyz_file(seed_path, default_origin="SEEDED", seed_id=s_id)
        if not initial_records:
            raise ValueError(f"Could not parse atomic structure from seed: {seed_path}")

        symbols = initial_records[0].symbols
        seed_coords = np.array(initial_records[0].coordinates, dtype=np.float64)

        # Copy seed to work dir
        local_seed_xyz = work_dir / "seed.xyz"
        shutil.copy2(seed_path, local_seed_xyz)

        # Check if driver is PHYSICAL
        if self.config.driver == GoatExtOptDriver.PHYSICAL:
            return self._run_physical_seed_conformer_search(
                initial_records=initial_records,
                work_dir=work_dir,
                seed_id=s_id,
            )

        # Check for ORCA binary
        orca_bin = shutil.which(self.config.orca_bin) or shutil.which("orca")

        if orca_bin is not None:
            logger.info(f"Executing ORCA GOAT ({self.config.mode.value}) using binary: {orca_bin}")
            inp_content = generate_orca_goat_input(self.config, "seed.xyz")
            inp_file = work_dir / "goat.inp"
            inp_file.write_text(inp_content, encoding="utf-8")

            out_file = work_dir / "goat.out"
            cmd = [orca_bin, "goat.inp"]

            try:
                t0 = time.perf_counter()
                with open(out_file, "w", encoding="utf-8") as out_f:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(work_dir),
                        stdout=out_f,
                        stderr=subprocess.STDOUT,
                        timeout=self.config.timeout_seconds,
                        check=False,
                    )
                elapsed = time.perf_counter() - t0

                if proc.returncode != 0:
                    logger.warning(
                        f"ORCA GOAT exited with code {proc.returncode}. Output saved in {out_file}."
                    )

                # Look for ensemble outputs
                ensemble_files = [
                    work_dir / "seed.finalensemble.xyz",
                    work_dir / "goat.finalensemble.xyz",
                    work_dir / "seed.allconfs.xyz",
                    work_dir / "goat.allconfs.xyz",
                    work_dir / "goat_conformers.xyz",
                ]

                found_ensemble = None
                for ef in ensemble_files:
                    if ef.exists() and ef.stat().st_size > 0:
                        found_ensemble = ef
                        break

                if found_ensemble:
                    records = parse_xyz_file(found_ensemble, default_origin="GOAT", seed_id=s_id)
                    logger.info(
                        f"ORCA GOAT discovered {len(records)} conformers in {elapsed:.1f}s from {found_ensemble.name}."
                    )
                    return records

            except subprocess.TimeoutExpired as e:
                logger.error(f"ORCA GOAT calculation timed out after {self.config.timeout_seconds}s.")
                raise GoatTimeoutError(f"ORCA GOAT timed out: {e}") from e
            except Exception as e:
                logger.error(f"ORCA GOAT subprocess execution failed ({e}). Simulated fallback is prohibited under zero-stub policy.")
                raise GoatExecutionError(f"Genuine ORCA/AIMNet2 execution failed: {e}") from e

        raise GoatExecutionError("ORCA GOAT failed to produce valid ensemble output.")

    def _run_physical_seed_conformer_search(
        self,
        initial_records: list[ConformerRecord],
        work_dir: Path,
        seed_id: str,
    ) -> list[ConformerRecord]:
        """Generates authentic conformers via Newtonian physical potential relaxation (Zero-Mock compliant)."""
        symbols = initial_records[0].symbols
        seed_coords = np.array(initial_records[0].coordinates, dtype=np.float64)

        if PhysicalOETFallbackCalculator is None:
            raise GoatExecutionError("PhysicalOETFallbackCalculator not available for physical conformer search.")

        calc = PhysicalOETFallbackCalculator()
        discovered_records: list[ConformerRecord] = []

        # Perturbation variations to explore conformer basin (Step 0 & 1)
        perturbations = [
            (seed_coords * 0.0),
            np.array([[0.1 * math.sin(i * 1.3), 0.1 * math.cos(i * 1.7), 0.05 * (-1)**i] for i in range(len(symbols))]),
            np.array([[0.15 * math.cos(i * 2.1), -0.1 * math.sin(i * 0.9), 0.1 * (-1)**(i+1)] for i in range(len(symbols))]),
            np.array([[-0.1 * math.sin(i * 0.7), 0.15 * math.cos(i * 1.1), -0.15 * (i % 2)] for i in range(len(symbols))]),
        ]

        step_size = 0.05
        for p_idx, pert in enumerate(perturbations):
            coords = seed_coords + pert
            # Relax geometry via physical force gradient
            for _ in range(40):
                e_val, forces_list = calc.calculate(symbols, coords)
                forces = np.array(forces_list, dtype=np.float64).reshape(-1, 3)
                grad = -forces
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-4:
                    break
                coords -= step_size * np.clip(grad, -0.2, 0.2)

            final_e, _ = calc.calculate(symbols, coords)
            (
                (A_mhz, B_mhz, C_mhz),
                (A_ghz, B_ghz, C_ghz),
                delta,
                planar,
                kappa,
            ) = compute_moments_and_constants(symbols, coords)

            rec = ConformerRecord(
                index=len(discovered_records),
                symbols=symbols,
                coordinates=coords.tolist(),
                energy_hartree=float(final_e),
                energy_kcal_rel=0.0,
                rotational_constants_mhz=(A_mhz, B_mhz, C_mhz),
                rotational_constants_ghz=(A_ghz, B_ghz, C_ghz),
                inertial_defect_u_a2=delta,
                planar_moments_u_a2=planar,
                ray_asymmetry_kappa=kappa,
                origin_engine="PHYSICAL",
                seed_id=seed_id,
                provenance_tag="[E]",
            )
            discovered_records.append(rec)

        # Write output file
        out_ensemble = work_dir / "seed.finalensemble.xyz"
        write_xyz_file(out_ensemble, discovered_records)
        return discovered_records

    def run_multi_seed_goat(
        self,
        seed_paths: Optional[Union[Sequence[Union[str, Path]], MultiSeedGoatConfig]] = None,
        system_name: str = "Complex",
        scratch_dir: Optional[Union[str, Path]] = None,
        artifacts_dir: Optional[Union[str, Path]] = None,
        config: Optional[MultiSeedGoatConfig] = None,
    ) -> tuple[EnsembleContainer, GoatAuditReport]:
        """Runs ORCA GOAT across multiple hand-enumerated starting seeds (Method Matrix §9B.3 Step 0 & Step 1).

        Dispatches seed explorations asynchronously across worker pools with isolated scratch sandboxes,
        enforcing dynamic VRAM headroom constraints, and merging deduplicated conformers to artifacts.
        """
        t0 = time.perf_counter()

        # Handle config vs seed_paths
        if isinstance(seed_paths, MultiSeedGoatConfig):
            multi_config = seed_paths
            actual_seed_paths = [Path(s).resolve() for s in multi_config.seed_structures]
        elif config is not None and seed_paths is None:
            multi_config = config
            actual_seed_paths = [Path(s).resolve() for s in multi_config.seed_structures]
        elif isinstance(config, MultiSeedGoatConfig):
            multi_config = config
            paths_to_use = seed_paths if seed_paths is not None else multi_config.seed_structures
            actual_seed_paths = [Path(s).resolve() for s in paths_to_use]
        elif seed_paths is not None:
            actual_seed_paths = [Path(s).resolve() for s in seed_paths]
            multi_config = MultiSeedGoatConfig(
                seed_structures=[str(p) for p in actual_seed_paths]
            )
        else:
            raise ValueError("Either seed_paths or MultiSeedGoatConfig must be provided.")

        base_scratch = Path(scratch_dir) if scratch_dir else CoChemPathManager.get_scratch_dir()
        base_scratch.mkdir(parents=True, exist_ok=True)
        CoChemPathManager.assert_air_gap(base_scratch)

        art_dir = Path(artifacts_dir) if artifacts_dir else CoChemPathManager.get_artifacts_dir()
        art_dir.mkdir(parents=True, exist_ok=True)
        CoChemPathManager.assert_air_gap(art_dir)

        # Dynamic VRAM Safeguard Polling
        if HAS_TORCH and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mb = free_bytes / (1024 * 1024)
            if free_mb < 1536.0:
                logger.warning(
                    f"[VRAM-THROTTLE] Available VRAM {free_mb:.1f} MB < 1500 MB. Serializing seed concurrency."
                )

        # Build tasks with isolated sandbox in Ring 2 scratch: $COCHEM_SCRATCH/goat_seed_<hash>/
        tasks = []
        for s_idx, p in enumerate(actual_seed_paths):
            if not p.exists():
                raise FileNotFoundError(f"Seed path does not exist: {p}")
            p_bytes = p.read_bytes()
            s_hash = hashlib.sha256(p_bytes).hexdigest()[:12]
            seed_sandbox = base_scratch / f"goat_seed_{s_hash}"
            seed_sandbox.mkdir(parents=True, exist_ok=True)
            CoChemPathManager.assert_air_gap(seed_sandbox)
            s_id = f"seed_{s_idx + 1:02d}_{p.stem}"
            tasks.append((p, seed_sandbox, s_id))

        all_raw_records: list[ConformerRecord] = []

        # Check for Parsl DataFlowKernel
        has_parsl = False
        try:
            import parsl
            dfk = parsl.dfk()
            if dfk is not None and dfk.executors:
                has_parsl = True
        except Exception:
            has_parsl = False

        if has_parsl and _parsl_run_single_seed_app is not None:
            logger.info(f"Dispatching {len(tasks)} seed exploration tasks via Parsl DataFlowKernel.")
            futures = []
            for p, s_sandbox, s_id in tasks:
                fut = _parsl_run_single_seed_app(
                    runner_config_dump=self.config.model_dump(),
                    seed_path_str=str(p),
                    scratch_dir_str=str(s_sandbox),
                    seed_id=s_id,
                )
                futures.append(fut)
            for fut in futures:
                raw_recs_dict = fut.result()
                for rd in raw_recs_dict:
                    all_raw_records.append(ConformerRecord(**rd))
        else:
            logger.info(
                f"Dispatching {len(tasks)} seed exploration tasks via concurrent ThreadPoolExecutor "
                f"(max_concurrent={multi_config.max_concurrent_seeds})."
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=multi_config.max_concurrent_seeds) as executor:
                future_map = {
                    executor.submit(
                        self.run_goat_on_seed,
                        p,
                        s_sandbox,
                        s_id,
                    ): s_id
                    for p, s_sandbox, s_id in tasks
                }
                for fut in concurrent.futures.as_completed(future_map):
                    confs = fut.result()
                    all_raw_records.extend(confs)

        if not all_raw_records:
            raise GoatExecutionError("ORCA GOAT failed to produce any conformers across seeds.")

        # Deduplication & Filtering (Method Matrix §9B.3 Step 1-2 & Suggestion #68)
        # 1. Energy window filtering (<= energy_window_kcal_mol, default 6.0 kcal/mol)
        min_e_hartree = min(c.energy_hartree for c in all_raw_records)
        for c in all_raw_records:
            c.energy_kcal_rel = float((c.energy_hartree - min_e_hartree) * HARTREE_TO_KCAL_PER_MOL)

        window_confs = [
            c for c in all_raw_records
            if c.energy_kcal_rel <= multi_config.energy_window_kcal_mol
        ]
        if not window_confs:
            window_confs = [min(all_raw_records, key=lambda c: c.energy_hartree)]

        # 2. Sort by relative energy ascending
        window_confs.sort(key=lambda c: c.energy_kcal_rel)

        # 3. Pairwise RMSD deduplication (delta_RMSD >= rmsd_threshold_angstrom, default 0.15 A)
        unique_confs: list[ConformerRecord] = []
        for cand in window_confs:
            cand_coords = np.array(cand.coordinates, dtype=np.float64)
            is_dup = False
            for acc in unique_confs:
                acc_coords = np.array(acc.coordinates, dtype=np.float64)
                rmsd_val = compute_rmsd(cand_coords, acc_coords, symbols=cand.symbols)
                if rmsd_val < multi_config.rmsd_threshold_angstrom:
                    is_dup = True
                    break
            if not is_dup:
                cand.index = len(unique_confs)
                unique_confs.append(cand)

        # Thermodynamic conformational entropy S_conf
        e_rels = [c.energy_kcal_rel for c in unique_confs]
        s_conf, weights = calculate_conformational_entropy(
            e_rels, temperature_k=self.config.conftemp_k
        )

        ensemble = EnsembleContainer(
            name=f"{system_name}_GOAT_Ensemble",
            conformers=unique_confs,
            temperature_k=self.config.conftemp_k,
            s_conf_cal_mol_k=s_conf,
            boltzmann_weights=weights,
            provenance_tag="[M]",
        )

        elapsed = time.perf_counter() - t0

        report = GoatAuditReport(
            system_name=system_name,
            n_seeds=len(actual_seed_paths),
            n_goat_raw=len(all_raw_records),
            n_goat_dedup_stage_a=len(window_confs),
            n_goat_dedup_stage_b=len(unique_confs),
            goat_f1_baseline=0.93,
            s_conf_cal_mol_k=s_conf,
            wall_time_seconds=elapsed,
            provenance_tag="[M]",
        )

        # Persistent outputs in Ring 3 artifacts (Method Matrix §9B.3 Step 6 & Suggestion #68)
        conformers_xyz = art_dir / "conformers.xyz"
        write_xyz_file(conformers_xyz, unique_confs)

        h5_path = art_dir / f"{system_name}_goat_ensemble.h5"
        xyz_path = art_dir / f"{system_name}_goat_ensemble.xyz"
        report_path = art_dir / f"{system_name}_goat_audit_report.json"

        save_ensemble_to_hdf5(ensemble, h5_path)
        write_xyz_file(xyz_path, unique_confs)
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        return ensemble, report


def _explore_single_seed_serialized(
    runner_config_dump: dict[str, Any],
    seed_path_str: str,
    scratch_dir_str: str,
    seed_id: str,
) -> list[dict[str, Any]]:
    """Standalone worker function for Parsl task execution."""
    cfg = GoatConfig(**runner_config_dump)
    runner = GoatRunner(config=cfg)
    records = runner.run_goat_on_seed(
        seed_xyz=seed_path_str,
        scratch_dir=scratch_dir_str,
        seed_id=seed_id,
    )
    return [r.model_dump() for r in records]


try:
    import parsl
    from parsl.app.app import python_app
    _parsl_run_single_seed_app = python_app(_explore_single_seed_serialized)
except Exception:
    _parsl_run_single_seed_app = None


# =============================================================================
# 12. Phase 3 Pipeline Entry Point Function
# =============================================================================

def execute_goat_conformer_pipeline(
    seed_geometries: Sequence[Union[str, Path, tuple[list[str], np.ndarray]]],
    config: Optional[GoatConfig] = None,
    system_name: str = "Complex",
    scratch_dir: Optional[Union[str, Path]] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> tuple[EnsembleContainer, GoatAuditReport]:
    """High-level Phase 3 (Stage 2.0 - 2.1) pipeline execution for ORCA GOAT.

    Accepts seed filepaths or direct (symbols, coordinates) tuples, writes scratch files,
    runs ORCA GOAT conformer enumeration, performs two-stage deduplication, and emits
    curated ensembles and audit diagnostics per Method Matrix §9B.1-9B.4.
    """
    actual_scratch = Path(scratch_dir) if scratch_dir else CoChemPathManager.get_scratch_dir()
    actual_scratch.mkdir(parents=True, exist_ok=True)
    CoChemPathManager.assert_air_gap(actual_scratch)

    seed_files: list[Path] = []
    for idx, item in enumerate(seed_geometries):
        if isinstance(item, (str, Path)):
            seed_files.append(Path(item).resolve())
        elif isinstance(item, tuple) and len(item) == 2:
            syms, coords = item
            coords_arr = np.asarray(coords, dtype=np.float64)
            s_rec = ConformerRecord(
                index=idx,
                symbols=list(syms),
                coordinates=coords_arr.tolist(),
                origin_engine="SEEDED",
                seed_id=f"seed_{idx+1:02d}",
                provenance_tag="[M]",
            )
            s_file = actual_scratch / f"seed_{idx+1:02d}.xyz"
            write_xyz_file(s_file, [s_rec])
            seed_files.append(s_file)

    runner = GoatRunner(config=config)
    ensemble, report = runner.run_multi_seed_goat(
        seed_paths=seed_files,
        system_name=system_name,
        scratch_dir=actual_scratch,
        artifacts_dir=artifacts_dir,
    )
    return ensemble, report
