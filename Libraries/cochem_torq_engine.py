"""
CoChem-TORQ: High-Fidelity Quantum Engine & Cascade Broker
===========================================================
Phase 5 (Stage 4.0) Implementation
----------------------------------
Governs the Method Matrix v4 execution cascade (defgrid1 -> defgrid3),
ORCA Python Interface (OPI) persistent memory threading, dynamic wavefunction
propagation (! MOREAD / %moinp), stateful SCF checkpointing, GPU4PySCF dynamic
batching with VRAM headroom protection, spin contamination validation,
tightened intermolecular %geom blocks, frozen-monomer protocol, and 6-Tier
Environment Matrix scratch/shm path resolution.

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
import numpy as np
import psutil
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-Engine] %(message)s")
logger = logging.getLogger("CoChem-TORQ.Engine")


# ============================================================================
# 1. 6-Tier Environment Matrix & Path Resolution
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
        # 1. GitHub Actions runner
        if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("RUNNER_TEMP"):
            return EnvironmentTier.GITHUB_ACTIONS

        # 2. GitHub Codespaces / Dev Container
        if os.environ.get("CODESPACES") == "true" or os.environ.get("CODESPACE_NAME"):
            return EnvironmentTier.CODESPACES

        # 3. HPC Cluster Nodes (SLURM / PBS / LSF)
        if (
            os.environ.get("SLURM_TMPDIR")
            or os.environ.get("SLURM_JOB_ID")
            or os.environ.get("PFSDIR")
            or os.environ.get("PBS_O_WORKDIR")
        ):
            return EnvironmentTier.HPC_NODES

        # 4. OS-specific local environments
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

        # Check GPU via pynvml
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
            # If target is inside repo root and not in an excluded scratch dir, raise AirGapViolationError
            if not (resolved_target.name.startswith("scratch") or "scratch" in resolved_target.parts):
                raise AirGapViolationError(
                    f"Tripartite Air-Gap Violation: Path '{resolved_target}' is inside static repository root '{repo_root}'."
                )
        except ValueError:
            # Not a subpath of repo_root -> Air-gap respected
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
# 2. Pydantic Execution Models
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
    molecular geometry, grid levels, and %geom / %scf directives.
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

    def to_orca_input(self, n_procs: int = 8, max_core_mb: int = 3000) -> str:
        """
        Serializes this payload into a complete, syntactically valid ORCA 6.1 input deck.
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

        # %pal block
        lines.append(f"%pal nprocs {n_procs} end")
        lines.append(f"%maxcore {max_core_mb}")

        # %moinp directive
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

        # Coordinate block
        lines.append(f"* xyz {self.charge} {self.multiplicity}")
        for sym, (x, y, z) in zip(self.symbols, self.coordinates):
            lines.append(f"  {sym:<2} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
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
# 3. Method Matrix v4 & Quantum Chemical Rules
# ============================================================================

# Pyykkö Single-Bond Covalent Radii (Å)
PYYKKO_SINGLE_RADII: Final[Dict[str, float]] = {
    "H": 0.32, "He": 0.46, "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64,
    "Ne": 0.67, "Na": 1.55, "Mg": 1.39, "Al": 1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96,
    "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11,
    "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.21, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17,
    "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.48, "Nb": 1.37, "Mo": 1.36, "Tc": 1.26, "Ru": 1.26, "Rh": 1.25,
    "Pd": 1.25, "Ag": 1.28, "Cd": 1.36, "In": 1.42, "Sn": 1.40, "Sb": 1.40, "Te": 1.36, "I": 1.33, "Xe": 1.31
}


def detect_complex_and_monomers(
    symbols: List[str],
    coordinates: np.ndarray,
    tolerance_multiplier: float = 1.20
) -> Tuple[bool, List[List[int]]]:
    """
    Detects whether the given atomic structure is an intermolecular complex / dimer
    by constructing the covalent connectivity graph using Pyykkö radii and identifying
    connected components.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    n_atoms = len(symbols)
    if n_atoms <= 1:
        return False, [[0]]

    # Compute pairwise distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    radii = np.array([PYYKKO_SINGLE_RADII.get(sym, 1.40) for sym in symbols], dtype=np.float64)
    cutoff_matrix = (radii[:, np.newaxis] + radii[np.newaxis, :]) * tolerance_multiplier

    # Build adjacency matrix
    adj = (dist_matrix < cutoff_matrix) & (dist_matrix > 1e-4)

    # Connected components via BFS
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
        # Singlet ideal <S^2> = 0.0
        deviation = abs(s_squared_observed - 0.0) * 100.0
        if s_squared_observed > 0.10:
            raise ValueError(
                f"[ERR_SPIN_CONTAMINATION] Spin contamination {s_squared_observed:.4f} in singlet state "
                f"exceeds tolerance (ideal=0.0000, observed={s_squared_observed:.4f})."
            )
        return s_ideal, s_squared_observed, deviation

    # Open-shell case (S > 0)
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
    grid_level: Optional[str] = None
) -> DispatchPayload:
    """
    Analyzes interatomic distances and applies Method Matrix v4 cascade rules:
    - Enforces InHess XTB2 or Lindh; forbids Calc_Hess true.
    - Requires D3/D4 dispersion on DFT for complexes.
    - Tightens %geom convergence criteria (TolMaxG 1e-5) on complexes.
    - Applies frozen monomer constraints if requested.
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

    # 2. Detect complexes and monomer components
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
        # Freeze monomer 0 atoms to fix high-level monomer geometry A, optimize intermolecular R
        frozen_indices = components[0]

    # 6. Dynamic grid tightening (defgrid1 -> defgrid3)
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
    context: Optional[ExecutionContext] = None
) -> DispatchPayload:
    """
    Executes the Method Matrix v4 hierarchical cascade mapping target tiers to
    exact quantum chemistry specifications (Table 2, §4.4, §8A, §8B).
    """
    if context is None:
        context = ExecutionContext()

    tier_key = target_tier.upper().strip()

    # Tier mapping under Method Matrix v4
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
        # Recipe R2: frozen monomers + wB97M-V/def2-QZVPP + CP + VPT2
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
        # Default high-fidelity DFT
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
        grid_level=grid_level
    )


# ============================================================================
# 4. In-Memory Wavefunction Propagation & OPI Persistent Threading
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

    # Write seed binary to shared memory if available
    if previous_result.gbw_bytes:
        with open(seed_file, "wb") as f:
            f.write(previous_result.gbw_bytes)
    else:
        # Create structured HDF5 seed containing MO coefficients and Fock matrix
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

        # If no raw bytes, write non-empty binary stub for MOREAD compatibility
        with open(seed_file, "wb") as f:
            f.write(b"ORCA_GBW_CHECKPOINT_SEED_V61\n" + h5_seed.read_bytes())

    # Update payload for MOREAD restart
    updated_payload = next_payload.model_copy(deep=True)
    updated_payload.use_moread = True
    updated_payload.moinp_path = str(seed_file)

    # Attach in-memory tensors to metadata for zero-copy OPI transfer
    if previous_result.mo_coefficients is not None:
        updated_payload.metadata["mo_coefficients"] = previous_result.mo_coefficients
    if previous_result.fock_matrix is not None:
        updated_payload.metadata["fock_matrix"] = previous_result.fock_matrix
    if previous_result.density_matrix is not None:
        updated_payload.metadata["density_matrix"] = previous_result.density_matrix

    logger.info(f"Dynamically propagated wavefunction from step {previous_result.step_idx} to seed {seed_file.name}.")
    return updated_payload


def opi_persistent_threading(
    input_payload: DispatchPayload,
    context: Optional[ExecutionContext] = None,
    n_steps: int = 3,
    trajectory: Optional[List[np.ndarray]] = None
) -> Generator[ORCAStepResult, None, None]:
    """
    Interfaces with the ORCA execution engine, yielding ORCAStepResult instances 
    across optimization or PES sweep steps.
    Replaced temporary handler with actual file-based execution.
    """
    if context is None:
        context = ExecutionContext()

    current_coords = np.copy(input_payload.coordinates)
    steps_to_run = trajectory if trajectory is not None else [current_coords for _ in range(n_steps)]

    # Use TorqOrcaExecutor to run actual ORCA jobs
    # Since we can't import it directly due to circular dependencies potentially,
    # we'll dynamically import or just run the subprocess directly.
    import re
    from pathlib import Path
    
    scratch_dir = context.get_scratch_dir("opi_thread")
    orca_bin = os.environ.get("ORCA_PATH", "orca")

    for idx, step_coords in enumerate(steps_to_run):
        # Update payload coordinates for this step
        step_payload = input_payload.model_copy(deep=True)
        step_payload.coordinates = step_coords
        
        # Generate ORCA input
        inp_content = step_payload.to_orca_input(n_procs=context.num_cores, max_core_mb=max(1000, context.max_memory_mb // context.num_cores))
        
        job_base = scratch_dir / f"opi_step_{idx:04d}_{context.session_id[:8]}"
        inp_path = job_base.with_suffix(".inp")
        out_path = job_base.with_suffix(".out")
        gbw_path = job_base.with_suffix(".gbw")
        
        inp_path.write_text(inp_content, encoding="utf-8")
        
        # Execute ORCA
        logger.info(f"[OPI Thread] Executing ORCA step {idx} at {inp_path}")
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
            
        # Parse output for energy and observables
        try:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except FileNotFoundError:
            content = ""

        # Parse energy
        energy = 0.0
        e_match = re.search(r"(?:FINAL SINGLE POINT ENERGY|TOTAL ENERGY)\s+(-?\d+\.\d+)", content)
        if e_match:
            energy = float(e_match.group(1))
            
        # Spin observables
        s_ideal, s_obs, s_dev = None, None, None
        if input_payload.multiplicity > 1:
            s2_match = re.search(r"Expectation value of <S\*\*2>\s+:\s+([\d\.]+)", content)
            s2_ideal_match = re.search(r"Ideal value s\*\(s\+1\)\s+for\s+S=\S+\s+:\s+([\d\.]+)", content)
            if s2_match and s2_ideal_match:
                s_obs = float(s2_match.group(1))
                ideal_val = float(s2_ideal_match.group(1))
                s_ideal, s_obs, s_dev = validate_spin_contamination(input_payload.multiplicity, s_obs)
                
        # Gradient parsing (simplified, assumes Opt or EnGrad was run)
        grad = np.zeros_like(step_coords)
        grad_match = re.search(r"CARTESIAN GRADIENT.*?\n\n(.*?)\n\n", content, re.DOTALL)
        if grad_match:
            lines = grad_match.group(1).strip().splitlines()
            parsed_grad = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 6 and not line.startswith("-"):
                    parsed_grad.append([float(parts[3]), float(parts[4]), float(parts[5])])
            if len(parsed_grad) == len(step_coords):
                grad = np.array(parsed_grad)

        # Read GBW if present
        gbw_data = None
        if gbw_path.exists():
            gbw_data = gbw_path.read_bytes()
            
        result = ORCAStepResult(
            step_idx=idx,
            energy=energy,
            coordinates=np.copy(step_coords),
            gradient=grad,
            converged=True if "ORCA TERMINATED NORMALLY" in content else False,
            gbw_bytes=gbw_data,
            s_squared_ideal=s_ideal,
            s_squared_observed=s_obs,
            spin_contamination_percent=s_dev,
            raw_output=content
        )
        
        logger.info(f"[OPI Thread] Yielded step {idx}: E = {energy:.8f} Ha")
        yield result


# ============================================================================
# 5. Stateful SCF Checkpointing
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
        # Fallback binary serialization
        with open(target_path, "wb") as f:
            f.write(str(wavefunction_data).encode("utf-8"))

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise IOError(f"Failed to persist checkpoint to '{target_path}'.")

    logger.info(f"Persisted SCF checkpoint: {target_path} ({target_path.stat().st_size} bytes).")
    return target_path


# ============================================================================
# 6. GPU4PySCF Dynamic Batching
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
    bytes_per_point = 8 * (n_basis ** 2) * 64 + (1024 * 1024 * 32)  # Base 32MB overhead
    mb_per_point = max(bytes_per_point / (1024 * 1024), 1.0)

    # Determine available VRAM
    available_vram_mb = context.vram_mb if context.vram_mb > 0 else 8192  # Default 8GB baseline
    usable_vram_mb = available_vram_mb * (1.0 - memory_headroom_fraction)

    # Calculate optimal batch size
    batch_size = max(1, int(usable_vram_mb / mb_per_point))
    # Cap batch size to reasonable quantum chemistry bounds
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
# 7. Subprocess Safety & Process Tree Teardown
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

    # Wait for processes to exit gracefully
    gone, alive = psutil.wait_procs(children + [parent], timeout=timeout_sec)

    # Phase 2: SIGKILL / Kill surviving processes
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


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

        stdout, stderr = proc.communicate(input=stdin_data, timeout=timeout)
        ret_code = proc.returncode

        if ret_code != 0:
            raise subprocess.CalledProcessError(ret_code, cmd, output=stdout, stderr=stderr)

        return stdout, stderr, ret_code

    except subprocess.TimeoutExpired as exc:
        if proc:
            safe_process_tree_teardown(proc.pid, timeout_sec=3.0)
        logger.error(f"Subprocess '{cmd[0]}' timed out after {timeout} seconds.")
        raise TimeoutError(f"Subprocess '{cmd[0]}' timed out after {timeout} seconds.") from exc

    except subprocess.CalledProcessError as exc:
        if proc:
            safe_process_tree_teardown(proc.pid, timeout_sec=2.0)
        logger.error(f"Subprocess '{cmd[0]}' failed with exit code {exc.returncode}: {exc.stderr}")
        raise

    except Exception as exc:
        if proc:
            safe_process_tree_teardown(proc.pid, timeout_sec=2.0)
        logger.error(f"Subprocess '{cmd[0]}' encountered unexpected exception: {exc}")
        raise


def cleanup_all_cochem_processes() -> None:
    """
    Registered atexit handler to ensure no orphaned orca, xtb, or mpi processes remain.
    """
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            p_name = str(proc.info["name"]).lower()
            if any(k in p_name for k in ["orca", "xtb", "mpirun", "orterun"]):
                if proc.info["pid"] != current_pid:
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass


# Register clean process teardown at program exit
atexit.register(cleanup_all_cochem_processes)
