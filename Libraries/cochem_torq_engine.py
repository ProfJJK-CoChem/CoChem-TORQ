"""ORCA subprocess execution, strict artifact parsing, and explicit checkpoints.

The historical ``opi_persistent_threading`` name is retained for compatibility;
this adapter launches independent ORCA processes and reuses native GBW files.
It does not provide an in-memory OPI session or synthesize missing properties.
Directory separation is storage hygiene, not a security isolation boundary.
"""

from __future__ import annotations

import atexit
import enum
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, Union

import h5py
import numpy as np
import psutil
from mendeleev import element
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.constants import physical_constants

if TYPE_CHECKING:
    from cochem_base.schemas import HardwareTelemetryReport


class EngineExecutionError(RuntimeError):
    """TORQ engine execution lacks required physical evidence."""


class SpinContaminationError(EngineExecutionError):
    """An observed spin diagnostic exceeds the configured acceptance rule."""


# Configure module-level logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-Engine] %(message)s"
)
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
        if iso.mass_number == mass_number and iso.mass is not None:
            return float(iso.mass)
    raise ValueError(
        f"No measured/database isotopic mass available for {symbol}-{mass_number}."
    )


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
    raise ValueError(f"Pyykkö covalent radius unavailable for {symbol}.")


def get_vdw_radius(symbol: str) -> float:
    """
    Dynamically retrieves van der Waals radius in Angstroms using mendeleev.
    (Mendeleev provides vdw_radius in picometers, converted to Å / 100.0).
    """
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = element(clean_sym)
    if el.vdw_radius is not None:
        return float(el.vdw_radius) / 100.0
    raise ValueError(f"van der Waals radius unavailable for {symbol}.")


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
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent.resolve()

    env_val = os.environ.get("COCHEM_REPO_DIR")
    if env_val:
        repo_path = Path(env_val).resolve()
        if repo_path.is_dir():
            return repo_path

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

        if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
            self.gpu_available = False
            self.vram_mb = 0
            return

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

    def get_telemetry(self, precision: str = "float64") -> HardwareTelemetryReport:
        """
        Queries honest OS/driver telemetry and returns an authentic HardwareTelemetryReport (Suggestion #52).
        """
        try:
            from cochem_base.schemas import HardwareTelemetryReport
        except ImportError as exc:
            raise RuntimeError(
                "BASE hardware telemetry integration requires the genuine cochem_base.schemas package."
            ) from exc
        gpu_avail = False
        dev_count = 0
        dev_name = "None"
        vram_total = 0.0
        vram_free = 0.0

        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            gpu_avail = False
            dev_count = 0
            dev_name = "None"
            vram_total = 0.0
            vram_free = 0.0
        else:
            try:
                import pynvml

                pynvml.nvmlInit()
                dev_count = pynvml.nvmlDeviceGetCount()
                if dev_count > 0:
                    gpu_avail = True
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    name = pynvml.nvmlDeviceGetName(handle)
                    dev_name = (
                        name.decode("utf-8") if isinstance(name, bytes) else str(name)
                    )
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_total = float(mem_info.total) / (1024.0 * 1024.0)
                    vram_free = float(mem_info.free) / (1024.0 * 1024.0)
                pynvml.nvmlShutdown()
            except Exception:
                gpu_avail = False
                dev_count = 0
                dev_name = "None"
                vram_total = 0.0
                vram_free = 0.0

        is_mps = False
        try:
            import platform

            import torch

            if (
                platform.system() == "Darwin"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                is_mps = True
        except Exception:
            is_mps = False

        if gpu_avail and vram_free >= 2048.0:
            runtime = "cuda"
        elif is_mps:
            if precision.lower() in ("float64", "fp64", "double"):
                runtime = "cpu"
            else:
                runtime = "mps"
        else:
            runtime = "cpu"

        return HardwareTelemetryReport(
            device_count=dev_count,
            gpu_available=gpu_avail,
            device_name=dev_name,
            vram_total_mb=vram_total,
            vram_free_mb=vram_free,
            selected_runtime=runtime,
            provenance="[M]",
        )

    @classmethod
    def probe_hardware(cls) -> HardwareTelemetryReport:
        """Non-initializing hardware probe via NVML without locking CUDA driver contexts [M]."""
        ctx = cls()
        return ctx.get_telemetry()

    def get_dispatch_contract(self) -> Dict[str, Any]:
        """Passes immutable execution contract down to Computational Tier workers [M]."""
        telemetry = self.get_telemetry()
        device = (
            "cuda"
            if telemetry.gpu_available and telemetry.vram_free_mb >= 2048.0
            else "cpu"
        )
        return {
            "device": device,
            "num_threads": self.num_cores,
            "session_id": self.session_id,
            "provenance": "[M]",
        }

    def verify_air_gap_boundary(self, target_path: Path) -> None:
        """
        Verifies that runtime scratch, shm, or artifacts paths do not mutate Domain A / Ring 1 repo root.
        """
        resolved_target = target_path.resolve()
        repo_root = get_repo_root().resolve()
        try:
            _ = resolved_target.relative_to(repo_root)
            if not (
                resolved_target.name.startswith("scratch")
                or "scratch" in resolved_target.parts
            ):
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
                slurm_tmp = (
                    os.environ.get("SLURM_TMPDIR")
                    or os.environ.get("PFSDIR")
                    or tempfile.gettempdir()
                )
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

    model_config = ConfigDict(arbitrary_types_allowed=True, allow_inf_nan=False)

    point_idx: int
    energy_hartree: float
    converged: bool = False
    vram_used_mb: Optional[float] = None
    coordinates: np.ndarray

    @field_validator("coordinates", mode="before")
    @classmethod
    def validate_coordinates(cls, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if (
            array.ndim != 2
            or array.shape[1] != 3
            or len(array) == 0
            or not np.isfinite(array).all()
        ):
            raise ValueError("Coordinates must be nonempty, finite and shaped (N, 3).")
        return array


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
    executor: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("coordinates", mode="before")
    @classmethod
    def validate_coordinates(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"Coordinates must have shape (N, 3), got shape {arr.shape}."
            )
        if not np.isfinite(arr).all() or len(arr) == 0:
            raise ValueError("Coordinates must be nonempty and finite.")
        return arr

    @model_validator(mode="after")
    def validate_atom_mapping(self) -> "DispatchPayload":
        if len(self.symbols) != len(self.coordinates):
            raise ValueError(
                "One explicit atom symbol is required for each coordinate row."
            )
        for symbol in self.symbols:
            if not re.fullmatch(r"[A-Z][a-z]?", symbol) or symbol in {"D", "T"}:
                raise ValueError(
                    "ORCA adapter requires element symbols; explicit isotope labels need a supported isotope-mass contract."
                )
            get_atomic_number(symbol)
        if self.multiplicity < 1:
            raise ValueError("Multiplicity must be positive.")
        for indices in (self.frozen_atom_indices, self.ghost_atom_indices):
            if indices is not None and any(
                i < 0 or i >= len(self.symbols) for i in indices
            ):
                raise ValueError(
                    "Atom indices must refer to the submitted atom mapping."
                )
        return self

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
        # Purge automatic ghosting of atoms from standard geometry optimization (! Opt) decks
        is_opt = (
            "opt" in self.extra_options.lower()
            or "opt" in self.method.lower()
            or any("opt" in line_text.lower() for line_text in lines)
        )
        lines.append(f"* xyz {self.charge} {self.multiplicity}")
        for idx, (sym, (x, y, z)) in enumerate(
            zip(self.symbols, self.coordinates, strict=False)
        ):
            is_ghost = (not is_opt) and (
                self.ghost_atom_indices is not None and idx in self.ghost_atom_indices
            )
            sym_tag = f"{sym}:" if is_ghost else sym
            lines.append(f"  {sym_tag:<4} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
        lines.append("*")

        return "\n".join(lines) + "\n"

    def generate_orca_deck(self, n_procs: int = 1, max_core_mb: int = 3000) -> str:
        """Alias for to_orca_input to generate complete ORCA input deck."""
        return self.to_orca_input(n_procs=n_procs, max_core_mb=max_core_mb)


class ORCAStepResult(BaseModel):
    """
    Result of an individual ORCA execution or persistent OPI threading step,
    carrying in-memory wavefunctions, Fock matrices, and spin observables.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, allow_inf_nan=False)

    step_idx: int = 0
    energy: float
    coordinates: np.ndarray
    input_coordinates: Optional[np.ndarray] = None
    coordinates_source: Optional[str] = None
    gradient: Optional[np.ndarray] = None
    converged: bool = False
    normally_terminated: bool = False
    scf_converged: bool = False
    optimization_converged: Optional[bool] = None
    mo_coefficients: Optional[np.ndarray] = None
    fock_matrix: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    gbw_bytes: Optional[bytes] = None
    gbw_path: Optional[Path] = None
    s_squared_observed: Optional[float] = None
    s_squared_ideal: Optional[float] = None
    spin_contamination_percent: Optional[float] = None
    dipole_moment: Optional[List[float]] = None
    dipole_coordinate_frame: Optional[str] = None
    dipole_units: Optional[str] = None
    frequencies: Optional[List[float]] = None
    raw_output: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("coordinates", mode="before")
    @classmethod
    def validate_coordinates(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"Coordinates must have shape (N, 3), got shape {arr.shape}."
            )
        if not np.isfinite(arr).all() or len(arr) == 0:
            raise ValueError("Coordinates must be nonempty and finite.")
        return arr

    @model_validator(mode="after")
    def validate_observables(self) -> "ORCAStepResult":
        for name in ("gradient", "input_coordinates"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value, dtype=np.float64)
                if arr.shape != self.coordinates.shape or not np.isfinite(arr).all():
                    raise ValueError(
                        f"{name} must be finite and match coordinate shape."
                    )
                setattr(self, name, arr)
        if self.dipole_moment is not None and len(self.dipole_moment) != 3:
            raise ValueError("A dipole vector requires exactly three components.")
        if self.converged and not (self.normally_terminated and self.scf_converged):
            raise ValueError(
                "Convergence requires normal termination and explicit electronic convergence."
            )
        if self.converged and self.optimization_converged is False:
            raise ValueError(
                "An unconverged optimization cannot be a converged result."
            )
        return self


# ============================================================================
# 4. Method Matrix v4 & Quantum Chemical Rules
# ============================================================================


def detect_complex_and_monomers(
    symbols: List[str], coordinates: np.ndarray, tolerance_multiplier: float = 1.20
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
    symbols: List[str], coordinates: np.ndarray, tolerance_multiplier: float = 1.20
) -> Tuple[bool, List[List[int]], List[Tuple[int, int, float]]]:
    """
    Identifies non-covalent contacts across molecular fragments using Pyykkö covalent
    radii for fragment partitioning and van der Waals radii for contact identification.
    Returns (has_non_covalent_contacts, monomer_components, contact_pairs).
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    is_comp, components = detect_complex_and_monomers(
        symbols, coords, tolerance_multiplier
    )

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


def calculate_discrete_counterpoise_energy(
    e_ab: float, e_a_ghost: float, e_b_ghost: float
) -> float:
    """
    Evaluates discrete 3-point counterpoise interaction energy on frozen-monomer relaxed structure:
    Delta E_CP = E_AB^{AB} - E_A^{AB} - E_B^{AB} [M].
    """
    return float(e_ab - e_a_ghost - e_b_ghost)


def extract_s_squared_from_orca_output(content: str) -> Optional[float]:
    """
    Extracts <S^2> expectation value across all ORCA versions via robust regex patterns.
    """
    matches = re.findall(
        r"<\s*S\s*(?:\*\*|\^)\s*2\s*>\s*:\s*(\S+)", content, re.IGNORECASE
    )
    return _finite_number(matches[-1], "spin expectation value") if matches else None


def validate_spin_contamination(
    arg1: Union[int, str, None] = None,
    arg2: Union[float, int, str, None] = None,
    is_unrestricted: bool = True,
    **kwargs: Any,
) -> Tuple[float, float, float]:
    if "multiplicity" in kwargs:
        if arg1 is not None and not isinstance(arg1, int):
            content = str(arg1)
            arg2 = kwargs["multiplicity"]
        else:
            arg1 = kwargs["multiplicity"]
    if "s_squared" in kwargs or "s2" in kwargs or "s_squared_observed" in kwargs:
        arg2 = kwargs.get(
            "s_squared", kwargs.get("s2", kwargs.get("s_squared_observed"))
        )
    """
    Validates spin contamination for open-shell systems under Method Matrix v4 §8B.3.
    Accepts either:
      (multiplicity: int, s_squared_observed: float | str, is_unrestricted: bool = True)
    or
      (content: str, multiplicity: int, is_unrestricted: bool = True)

    Ideal <S^2> = S_ideal * (S_ideal + 1) where S_ideal = (multiplicity - 1) / 2.
    Raises:
      MissingTelemetryError: If spin observable <S^2> cannot be extracted from unrestricted calculation output.
      SpinContaminationError: If relative spin deviation exceeds 10.0%.
    """
    if isinstance(arg1, str):
        content = arg1
        multiplicity = int(arg2)
        s2_val = extract_s_squared_from_orca_output(content)
        if s2_val is None:
            raise EngineExecutionError(
                "[MISSING DATA] No observed <S^2> expectation value was supplied."
            )
    elif isinstance(arg2, str):
        multiplicity = int(arg1)
        try:
            s2_val = float(arg2)
        except ValueError:
            s2_val = extract_s_squared_from_orca_output(arg2)
            if s2_val is None:
                raise EngineExecutionError(
                    "[MISSING DATA] No observed <S^2> expectation value was supplied."
                ) from None
    else:
        multiplicity = int(arg1)
        if arg2 is None:
            raise EngineExecutionError(
                "[MISSING DATA] No observed <S^2> expectation value was supplied."
            )
        s2_val = float(arg2)

    if multiplicity < 1:
        raise ValueError(f"Multiplicity must be >= 1, got {multiplicity}.")
    if not np.isfinite(s2_val) or s2_val < 0:
        raise ValueError("Observed <S^2> must be finite and nonnegative.")

    s_ideal = (multiplicity - 1) / 2.0
    s_ideal_prod = s_ideal * (s_ideal + 1.0)
    allow_spin = kwargs.get("allow_spin_contamination", False)

    if multiplicity == 1:
        spin_dev = abs(s2_val - 0.0)
        if s2_val > 0.10 and not allow_spin:
            raise SpinContaminationError(
                f"[ERR_SPIN_CONTAMINATION] Electronic state spin contamination exceeds 10% limit: "
                f"<S^2> = {s2_val:.4f}, Ideal = 0.0000 (deviation {s2_val:.1%})."
            )
        return s_ideal_prod, s2_val, spin_dev * 100.0

    spin_dev = abs(s2_val - s_ideal_prod) / s_ideal_prod
    if spin_dev > 0.10 and not allow_spin:
        raise SpinContaminationError(
            f"[ERR_SPIN_CONTAMINATION] Electronic state spin contamination exceeds 10% limit: "
            f"<S^2> = {s2_val:.4f}, Ideal = {s_ideal_prod:.4f} (deviation {spin_dev:.1%})."
        )

    return s_ideal_prod, s2_val, spin_dev * 100.0


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
    ghost_atom_indices: Optional[List[int]] = None,
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
    if symbols is None:
        raise ValueError(
            "Atom symbols are required; element identities cannot be inferred from coordinates."
        )

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
        if "3c" in resolved_method.lower() or any(
            xtb_kw in resolved_method.lower() for xtb_kw in ["xtb", "gfn"]
        ):
            resolved_basis = ""
        else:
            resolved_basis = "def2-TZVP"

    resolved_aux = "def2/J" if "def2" in resolved_basis else ""

    # 4. Dispersion enforcement for DFT on weak complexes (§4.4, §8A)
    if complex_flag:
        m_upper = resolved_method.upper()
        e_upper = extra_options.upper()
        is_dft = any(
            func in m_upper
            for func in [
                "B3LYP",
                "PBE",
                "SCAN",
                "M06",
                "W97",
                "OLYP",
                "OPBE",
                "DFT",
                "R2SCAN",
            ]
        )
        has_dispersion = any(
            d in m_upper or d in e_upper
            for d in ["D3", "D4", "-V", "VV10", "3C", "-3C"]
        )
        if is_dft and not has_dispersion:
            raise ValueError(
                "[ERR_METHOD_MATRIX] Dispersion correction (D3/D4) is strictly required for DFT optimization of weak complexes."
            )

    # 5. Frozen monomer constraints (§9A.1-9A.2)
    frozen_indices: Optional[List[int]] = None
    if frozen_monomer and len(components) >= 2:
        frozen_indices = components[0]

    # 6. Counterpoise & Ghost atoms
    # Eradicate ghost-atom injection during active geometry relaxations (! Opt).
    # Counterpoise calculations are decoupled and coordinated via discrete single-point jobs post-optimization.
    is_opt_deck = "opt" in extra_options.lower() or "opt" in resolved_method.lower()
    resolved_ghosts = None if is_opt_deck else ghost_atom_indices
    if (
        not is_opt_deck
        and counterpoise
        and resolved_ghosts is None
        and len(components) >= 2
    ):
        resolved_ghosts = components[1]

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
            "shm_dir": str(context.get_shm_dir()),
        },
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
    context: Optional[ExecutionContext] = None,
    tier_key: Optional[str] = None,
) -> DispatchPayload:
    """
    Executes the Method Matrix v4 hierarchical cascade mapping target tiers to
    exact quantum chemistry specifications (Table 2, §4.4, §8A, §8B).
    """
    if context is None:
        context = ExecutionContext()

    resolved_tier = tier_key if tier_key is not None else target_tier
    tier_key = resolved_tier.upper().strip()

    if tier_key in ["T3-10S", "T1-10S"]:
        method = "GFN2-xTB"
        basis = ""
    elif tier_key in ["T3-1MIN", "T1-1MIN"]:
        method = "r2SCAN-3c"
        basis = ""
    elif tier_key in ["T3-30MIN", "T1-30MIN"]:
        method = "r2SCAN-3c"
        basis = ""
    elif tier_key in ["T3-1H", "T1-1H"]:
        method = "B3LYP-D4"
        basis = "def2-TZVP"
    elif tier_key in ["T3-3H", "T1-3H"]:
        method = "wB97M-V"
        basis = "def2-QZVPP"
        frozen_monomer = True
    elif tier_key in ["T3-12H", "T1-12H"]:
        method = "revDSD-PBEP86-D4"
        basis = "def2-TZVPP"
    elif tier_key in ["T4-1D", "T4-1H"]:
        method = "DLPNO-CCSD(T)"
        basis = "def2-TZVP"
    elif (
        tier_key
        in ["T3C", "T4C", "T3-C", "T4-C", "T3C-3D", "T4C-1MO", "CFOUR_VPT2", "CFOUR"]
        or tier_key.startswith("T3C")
        or tier_key.startswith("T4C")
        or "CFOUR" in tier_key
    ):
        raise NotImplementedError(
            "CFOUR method-matrix execution is unavailable: TORQ has no qualified "
            "CFOUR gradient/anharmonic derivative adapter. Provision the actual "
            "engine and genuine basis library in a separate calculation environment, "
            "then independently qualify native derivative parsing and the exact "
            "method before enabling this route. No BASE installation or lower-level "
            "method substitutes for that scientific qualification."
        )
    else:
        raise ValueError(
            f"Unsupported method tier {resolved_tier!r}; no method was substituted."
        )

    payload = route_cascade_rules(
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
        ghost_atom_indices=ghost_atom_indices,
    )
    return payload


# ============================================================================
# 5. In-Memory Wavefunction Propagation & OPI Persistent Threading
# ============================================================================


def dynamic_wavefunction_propagation(
    previous_result: ORCAStepResult,
    next_payload: DispatchPayload,
    context: ExecutionContext,
) -> DispatchPayload:
    """
    Transmits molecular orbital coefficients and Fock matrices between adjacent
    geometric points. In standalone execution, persists seed to SHM and injects
    ! MOREAD / %moinp into next_payload.
    """
    if not previous_result.converged:
        raise ValueError(
            "Wavefunction propagation requires a converged source calculation."
        )
    if not previous_result.gbw_bytes:
        raise ValueError(
            "An authentic ORCA GBW artifact is required; arrays cannot be converted to GBW here."
        )
    shm_dir = context.get_shm_dir()
    seed_file = shm_dir / f"seed_{uuid.uuid4().hex}.gbw"
    seed_file.write_bytes(previous_result.gbw_bytes)

    updated_payload = next_payload.model_copy(deep=True)
    updated_payload.use_moread = True
    updated_payload.moinp_path = str(seed_file)

    if previous_result.mo_coefficients is not None:
        updated_payload.metadata["mo_coefficients"] = previous_result.mo_coefficients
    if previous_result.fock_matrix is not None:
        updated_payload.metadata["fock_matrix"] = previous_result.fock_matrix
    if previous_result.density_matrix is not None:
        updated_payload.metadata["density_matrix"] = previous_result.density_matrix

    logger.info(
        f"Dynamically propagated wavefunction from step {previous_result.step_idx} to seed {seed_file.name}."
    )
    return updated_payload


class ORCAParseError(ValueError):
    """Required physical output is absent, malformed, inconsistent or nonfinite."""


def _source_artifact_metadata(artifact_paths: List[Path]) -> Dict[str, Any]:
    """Digest the actual parser/input artifacts and bind their path-to-digest map.

    The manifest is sorted compact UTF-8 JSON, not a claim of RFC 8785
    canonicalization. Content digests establish file identity, not correctness
    of the calculation. Missing paths cannot receive an invented digest.
    """
    if not artifact_paths:
        raise ORCAParseError(
            "Physical result provenance requires actual source artifacts."
        )
    digests: Dict[str, str] = {}
    for artifact_path in artifact_paths:
        path = artifact_path.resolve(strict=True)
        if not path.is_file():
            raise ORCAParseError(f"Source artifact is not a regular file: {path}.")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[str(path)] = digest.hexdigest()
    manifest = json.dumps(
        digests,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return {
        "artifact_sha256": digests,
        "source_manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
        "source_manifest_format": "sorted_compact_json_path_to_sha256_v1",
    }


def _finite_number(token: str, quantity: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ORCAParseError(f"Malformed {quantity}: {token!r}.") from exc
    if not np.isfinite(value):
        raise ORCAParseError(f"Nonfinite {quantity} is not a physical result.")
    return value


def _engrad_section(content: str, title: str) -> List[str]:
    """Read a single named section without consuming data from the next one."""
    lines = content.splitlines()
    starts = [i for i, line in enumerate(lines) if title in line.lower()]
    if len(starts) != 1:
        raise ORCAParseError(f"Expected exactly one .engrad section: {title}.")
    rows: List[str] = []
    for line in lines[starts[0] + 1 :]:
        stripped = line.strip()
        if stripped.startswith("#"):
            if stripped.lstrip("#").strip():
                break
            continue
        if stripped:
            rows.append(stripped)
    return rows


def _read_orca_engrad(
    path: Path, n_atoms: int
) -> Tuple[float, np.ndarray, List[int], np.ndarray]:
    """Return Eh, Eh/bohr, atom identities, and Angstrom coordinates from ORCA."""
    if n_atoms < 1:
        raise ORCAParseError("Expected atom count must be positive.")
    content = path.read_text(encoding="utf-8")
    count = _engrad_section(content, "number of atoms")
    if len(count) != 1 or count[0] != str(n_atoms):
        raise ORCAParseError(
            "ORCA .engrad atom count does not match the requested structure."
        )
    energies = _engrad_section(content, "total energy in eh")
    if len(energies) != 1:
        raise ORCAParseError("ORCA .engrad requires exactly one total energy.")
    energy = _finite_number(energies[0], "energy")
    gradient_rows = _engrad_section(content, "gradient in eh/bohr")
    if len(gradient_rows) != 3 * n_atoms:
        raise ORCAParseError("Incomplete or oversized ORCA .engrad gradient.")
    gradient = np.array([_finite_number(v, "gradient") for v in gradient_rows]).reshape(
        n_atoms, 3
    )
    coordinate_rows = _engrad_section(
        content, "atomic numbers and current coordinates in bohr"
    )
    if len(coordinate_rows) != n_atoms:
        raise ORCAParseError("Incomplete or oversized ORCA .engrad geometry.")
    atomic_numbers: List[int] = []
    coordinates = []
    for row in coordinate_rows:
        fields = row.split()
        if (
            len(fields) != 4
            or not fields[0].isdigit()
            or not 1 <= int(fields[0]) <= 118
        ):
            raise ORCAParseError(
                "Malformed atomic identity/coordinate row in ORCA .engrad."
            )
        atomic_numbers.append(int(fields[0]))
        coordinates.append([_finite_number(v, "coordinate") for v in fields[1:]])
    bohr_in_angstrom = physical_constants["Bohr radius"][0] / 1e-10
    return energy, gradient, atomic_numbers, np.asarray(coordinates) * bohr_in_angstrom


def _orca_scf_converged(out_content: str, property_path: Optional[Path] = None) -> bool:
    """Normal termination alone is not evidence of electronic convergence."""
    if re.search(
        r"SCF\s+(?:NOT|DID NOT|HAS NOT)\s+CONVERG|SCF FAILED",
        out_content,
        re.IGNORECASE,
    ):
        return False
    if property_path is not None and property_path.is_file():
        content = property_path.read_text(encoding="utf-8")
        blocks = re.findall(r"(?ms)^\$Single_Point_Data\s*\n(.*?)^\$End\s*$", content)
        if blocks:
            matches = re.findall(
                r'&Converged\s+\[&Type\s+"Boolean"\]\s+(true|false)\b',
                blocks[-1],
                re.IGNORECASE,
            )
            if matches:
                return matches[-1].lower() == "true"
    return bool(re.search(r"SCF CONVERGED AFTER\s+\d+", out_content, re.IGNORECASE))


def _parse_orca_engrad_or_output(
    engrad_path: Path, out_content: str, n_atoms: int
) -> Tuple[float, np.ndarray, bool]:
    """Require finite observed Eh/Eh-bohr values and explicit SCF evidence.

    Existing malformed artifacts are rejected. A missing artifact may be read
    from the final stdout block, but missing values never become physical zeros.
    The returned flag represents electronic convergence, not optimization.
    """
    if engrad_path.is_file():
        energy, gradient, _, _ = _read_orca_engrad(engrad_path, n_atoms)
    else:
        matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(\S+)", out_content)
        if not matches:
            raise ORCAParseError("Missing final ORCA energy.")
        energy = _finite_number(matches[-1], "energy")
        sections = out_content.rsplit("CARTESIAN GRADIENT", 1)
        if len(sections) != 2:
            raise ORCAParseError("Missing final ORCA gradient.")
        rows = []
        for line in sections[-1].splitlines():
            match = re.match(
                r"\s*(\d+)\s+[A-Za-z]+\s*:\s*(\S+)\s+(\S+)\s+(\S+)\s*$", line
            )
            if match:
                if int(match.group(1)) != len(rows):
                    raise ORCAParseError("Noncontiguous ORCA gradient atom mapping.")
                rows.append([_finite_number(v, "gradient") for v in match.groups()[1:]])
            elif rows:
                break
        if len(rows) != n_atoms:
            raise ORCAParseError("Incomplete final ORCA Cartesian gradient.")
        gradient = np.asarray(rows)
    converged = _orca_scf_converged(
        out_content, engrad_path.with_suffix(".property.txt")
    )
    return energy, gradient, converged


def _read_orca_final_coordinates(
    job_base: Path, out_content: str, symbols: List[str]
) -> Tuple[np.ndarray, str]:
    """Read final engine geometry; never label the submitted geometry optimized."""
    xyz_path = job_base.with_suffix(".xyz")
    if xyz_path.is_file():
        lines = xyz_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or lines[0].strip() != str(len(symbols)):
            raise ORCAParseError("Final ORCA XYZ has an invalid atom count.")
        rows = [line.split() for line in lines[2:] if line.strip()]
        if len(rows) != len(symbols) or any(len(row) != 4 for row in rows):
            raise ORCAParseError("Final ORCA XYZ is truncated or malformed.")
        if [row[0].capitalize() for row in rows] != [
            s.rstrip(":").capitalize() for s in symbols
        ]:
            raise ORCAParseError("Final ORCA XYZ atom mapping differs from input.")
        return np.asarray(
            [[_finite_number(v, "coordinate") for v in row[1:]] for row in rows]
        ), str(xyz_path)
    engrad_path = job_base.with_suffix(".engrad")
    if engrad_path.is_file():
        _, _, numbers, coordinates = _read_orca_engrad(engrad_path, len(symbols))
        if numbers != [get_atomic_number(s) for s in symbols]:
            raise ORCAParseError("Final ORCA .engrad atom mapping differs from input.")
        return coordinates, str(engrad_path)
    raise ORCAParseError(
        "Missing final engine geometry (.xyz or .engrad); input geometry was not substituted."
    )


def opi_persistent_threading(
    input_payload: DispatchPayload,
    context: Optional[ExecutionContext] = None,
    n_steps: int = 3,
    trajectory: Optional[List[np.ndarray]] = None,
) -> Generator[ORCAStepResult, None, None]:
    """
    Interfaces with the ORCA execution engine, yielding ORCAStepResult instances
    across optimization or PES sweep steps with dynamic wavefunction propagation.
    Handles Windows / MPI execution cleanly to prevent exit code 126.
    """
    if input_payload.executor not in (None, "ORCA", "orca", "TorqOrcaExecutor"):
        raise ValueError(
            f"ORCA adapter cannot execute requested executor {input_payload.executor!r}."
        )
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    if context is None:
        context = ExecutionContext()

    current_coords = np.copy(input_payload.coordinates)
    steps_to_run = (
        trajectory
        if trajectory is not None
        else [current_coords for _ in range(n_steps)]
    )

    # Each invocation owns a fresh directory; files from earlier attempts must
    # never satisfy the current attempt's parse or convergence requirements.
    scratch_dir = context.get_scratch_dir(f"opi_thread/{uuid.uuid4().hex}")
    orca_bin = os.environ.get("ORCA_PATH", "orca")
    if not shutil.which(orca_bin):
        raise FileNotFoundError(
            f"Requested ORCA executable is unavailable: {orca_bin!r}."
        )

    # Determine safe core allocation (avoid MPI error 126 on Windows when MPI is unconfigured)
    safe_n_procs = context.num_cores if is_openmpi_supported() else 1

    last_gbw_path: Optional[Path] = None

    for idx, step_coords in enumerate(steps_to_run):
        if trajectory is None:
            step_coords = current_coords
        step_payload = DispatchPayload.model_validate(
            {**input_payload.model_dump(), "coordinates": step_coords}
        )
        keywords = re.findall(
            r"[A-Za-z][A-Za-z0-9_-]*",
            step_payload.method + "\n" + step_payload.extra_options,
        )
        optimization_requested = any(
            k.lower()
            in {"opt", "tightopt", "verytightopt", "looseopt", "optts", "copt"}
            for k in keywords
        )

        # Dynamically propagate previous step's wavefunction seed via MOREAD
        if idx > 0 and last_gbw_path and last_gbw_path.exists():
            step_payload.use_moread = True
            step_payload.moinp_path = str(last_gbw_path)

        # Append EnGrad if not already present
        if (
            "engrad" not in step_payload.extra_options.lower()
            and "engrad" not in step_payload.method.lower()
        ):
            step_payload.extra_options = (
                f"! EnGrad\n{step_payload.extra_options}".strip()
            )

        job_base = scratch_dir / f"opi_step_{idx:04d}_{context.session_id[:8]}"
        inp_path = job_base.with_suffix(".inp")
        out_path = job_base.with_suffix(".out")
        err_path = job_base.with_suffix(".err")
        gbw_path = job_base.with_suffix(".gbw")
        engrad_path = job_base.with_suffix(".engrad")

        inp_content = step_payload.to_orca_input(
            n_procs=safe_n_procs,
            max_core_mb=max(1000, context.max_memory_mb // max(1, safe_n_procs)),
        )
        inp_path.write_text(inp_content, encoding="utf-8")

        logger.info(
            f"[OPI Thread] Executing ORCA step {idx} (n_procs={safe_n_procs}) at {inp_path}"
        )
        try:
            stdout, stderr, ret_code = execute_subprocess_safe(
                cmd=[orca_bin, str(inp_path)], cwd=scratch_dir, timeout=3600.0
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(stdout)
            err_path.write_text(stderr, encoding="utf-8")
        except subprocess.CalledProcessError as e:
            out_path.write_text(e.stdout or "", encoding="utf-8")
            err_path.write_text(e.stderr or "", encoding="utf-8")
            raise RuntimeError(
                f"ORCA execution failed at step {idx}; see {out_path} and {err_path}."
            ) from e
        except Exception as e:
            logger.error(f"[OPI Thread] ORCA execution failed at step {idx}: {e}")
            raise RuntimeError(f"ORCA execution failed at step {idx}: {e}") from e

        # Parse energy, gradient, convergence
        energy, grad, scf_converged = _parse_orca_engrad_or_output(
            engrad_path, stdout, len(step_coords)
        )
        normally_terminated = "ORCA TERMINATED NORMALLY" in stdout
        optimization_converged = (
            (
                "THE OPTIMIZATION HAS CONVERGED" in stdout
                and "THE OPTIMIZATION DID NOT CONVERGE" not in stdout
            )
            if optimization_requested
            else None
        )
        converged = (
            normally_terminated
            and scf_converged
            and optimization_converged is not False
        )
        if not converged:
            raise ORCAParseError(
                f"ORCA did not provide all requested convergence evidence: normal termination={normally_terminated}, "
                f"electronic convergence={scf_converged}, optimization convergence={optimization_converged}. "
                f"Artifacts retained at {scratch_dir}."
            )
        if optimization_requested:
            final_coords, coordinates_source = _read_orca_final_coordinates(
                job_base, stdout, input_payload.symbols
            )
        else:
            # Fixed-geometry EnGrad evaluates the requested coordinates. Its
            # coordinates remain explicitly labelled as input, never optimized.
            final_coords, coordinates_source = (
                np.copy(step_coords),
                "input:fixed_geometry",
            )
        if engrad_path.is_file():
            _, _, numbers, gradient_coords = _read_orca_engrad(
                engrad_path, len(step_coords)
            )
            if numbers != [get_atomic_number(s) for s in input_payload.symbols]:
                raise ORCAParseError(
                    "Gradient atom identities differ from requested atom mapping."
                )
            if not np.allclose(final_coords, gradient_coords, atol=1e-5, rtol=0):
                raise ORCAParseError(
                    "Final geometry and gradient artifact do not describe the same coordinates."
                )

        # Spin observables
        s_ideal, s_obs, s_dev = None, None, None
        if input_payload.multiplicity > 1:
            s_ideal, s_obs, s_dev = validate_spin_contamination(
                stdout, input_payload.multiplicity
            )

        # Read GBW binary bytes
        gbw_data = None
        if gbw_path.exists():
            gbw_data = gbw_path.read_bytes()
            last_gbw_path = gbw_path

        # Note: In-memory MO/Fock arrays require an external MOLDEN parser or orca_2mkl.
        # The raw physical binary checkpoint is fully preserved in gbw_bytes for MOREAD propagation.
        mo_coefficients = None
        fock_matrix = None
        density_matrix = None

        source_artifacts = [inp_path, out_path, err_path]
        if step_payload.moinp_path:
            seed_path = Path(step_payload.moinp_path)
            source_artifacts.append(
                seed_path if seed_path.is_absolute() else scratch_dir / seed_path
            )
        if engrad_path.is_file():
            source_artifacts.append(engrad_path)
        property_path = job_base.with_suffix(".property.txt")
        if property_path.is_file():
            source_artifacts.append(property_path)
        if optimization_requested and coordinates_source != str(engrad_path):
            source_artifacts.append(Path(coordinates_source))
        artifact_metadata = _source_artifact_metadata(source_artifacts)

        result = ORCAStepResult(
            step_idx=idx,
            energy=energy,
            coordinates=final_coords,
            input_coordinates=np.copy(step_coords),
            coordinates_source=coordinates_source,
            gradient=grad,
            converged=converged,
            normally_terminated=normally_terminated,
            scf_converged=scf_converged,
            optimization_converged=optimization_converged,
            mo_coefficients=mo_coefficients,
            fock_matrix=fock_matrix,
            density_matrix=density_matrix,
            gbw_bytes=gbw_data,
            gbw_path=gbw_path if gbw_path.exists() else None,
            s_squared_ideal=s_ideal,
            s_squared_observed=s_obs,
            spin_contamination_percent=s_dev,
            raw_output=stdout,
            metadata={
                **artifact_metadata,
                "engine": "ORCA",
                "executable": orca_bin,
                "energy_units": "hartree",
                "gradient_units": "hartree/bohr",
                "coordinate_units": "angstrom",
                "gradient_is_force": False,
                "geometry_role": "optimized" if optimization_requested else "input",
                "input_path": str(inp_path),
                "stdout_path": str(out_path),
                "stderr_path": str(err_path),
                "unavailable_properties": [
                    "frequencies",
                    "dipole_moment",
                    "mo_coefficients",
                    "fock_matrix",
                    "density_matrix",
                ],
            },
        )

        current_coords = final_coords.copy()
        logger.info(
            f"[OPI Thread] Yielded step {idx}: E = {energy:.8f} Ha, converged={converged}"
        )
        yield result


# ============================================================================
# 6. Stateful SCF Checkpointing
# ============================================================================


def stateful_scf_checkpointing(
    step_idx: int,
    wavefunction_data: Union[bytes, Dict[str, Any], np.ndarray],
    context: ExecutionContext,
    checkpoint_type: str = "gbw",
) -> Path:
    """
    Persists binary .gbw, .chk, or .hess checkpoints to context.get_scratch_dir('orca_tmp')
    at all topological stationary points (minima and transition states).
    """
    if not re.fullmatch(r"[a-zA-Z0-9_]+", checkpoint_type):
        raise ValueError("checkpoint_type must be a plain format name.")
    if not isinstance(wavefunction_data, (bytes, dict, np.ndarray)):
        raise TypeError(
            "Checkpoint data must be authentic binary data or explicitly typed arrays."
        )
    if isinstance(wavefunction_data, bytes) and not wavefunction_data:
        raise ValueError(
            "Empty checkpoint data is unavailable, not a valid checkpoint."
        )
    if isinstance(wavefunction_data, np.ndarray):
        if not wavefunction_data.size or not np.isfinite(wavefunction_data).all():
            raise ValueError("Checkpoint arrays must be nonempty and finite.")
    if isinstance(wavefunction_data, dict):
        if not wavefunction_data:
            raise ValueError("Empty checkpoint data is unavailable.")
        for key, value in wavefunction_data.items():
            if not isinstance(value, (np.ndarray, int, float, str)):
                raise TypeError(f"Unsupported checkpoint value for {key!r}.")
            if isinstance(value, np.ndarray) and (
                not value.size or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"Checkpoint array {key!r} must be nonempty and finite."
                )
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"Checkpoint scalar {key!r} must be finite.")
    # TORQ arrays are HDF5 archives, not ORCA native GBW/HESS checkpoint files.
    extension = checkpoint_type if isinstance(wavefunction_data, bytes) else "h5"
    scratch_tmp = context.get_scratch_dir("orca_tmp")
    chk_filename = f"checkpoint_step_{step_idx:04d}.{extension}"
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

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise IOError(f"Failed to persist checkpoint to '{target_path}'.")

    logger.info(
        f"Persisted SCF checkpoint: {target_path} ({target_path.stat().st_size} bytes)."
    )
    return target_path


# ============================================================================
# 7. GPU4PySCF Dynamic Batching
# ============================================================================


def gpu4pyscf_dynamic_batching(
    grid_points: List[np.ndarray],
    context: ExecutionContext,
    system_size: Optional[int] = None,
    basis_functions_per_atom: int = 30,
    memory_headroom_fraction: float = 0.15,
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
    bytes_per_point = 8 * (n_basis**2) * 64 + (1024 * 1024 * 32)
    mb_per_point = max(bytes_per_point / (1024 * 1024), 1.0)

    # Determine available VRAM
    if not context.gpu_available or context.vram_mb <= 0:
        raise ValueError(
            "GPU batching requires observed GPU memory; no device capacity was assumed."
        )
    if not 0 <= memory_headroom_fraction < 1:
        raise ValueError("memory_headroom_fraction must be in [0, 1).")
    available_vram_mb = context.vram_mb
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
    stdin_data: Optional[str] = None,
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
            env=run_env,
        )
        if proc.pid:
            register_spawned_process(proc.pid)

        stdout, stderr = proc.communicate(input=stdin_data, timeout=timeout)
        ret_code = proc.returncode

        if proc.pid:
            unregister_spawned_process(proc.pid)

        if ret_code != 0:
            raise subprocess.CalledProcessError(
                ret_code, cmd, output=stdout, stderr=stderr
            )

        return stdout, stderr, ret_code

    except subprocess.TimeoutExpired as exc:
        if proc:
            if proc.pid:
                unregister_spawned_process(proc.pid)
            safe_process_tree_teardown(proc.pid, timeout_sec=3.0)
        logger.error(f"Subprocess '{cmd[0]}' timed out after {timeout} seconds.")
        raise TimeoutError(
            f"Subprocess '{cmd[0]}' timed out after {timeout} seconds."
        ) from exc

    except subprocess.CalledProcessError as exc:
        if proc:
            if proc.pid:
                unregister_spawned_process(proc.pid)
            safe_process_tree_teardown(proc.pid, timeout_sec=2.0)
        logger.error(
            f"Subprocess '{cmd[0]}' failed with exit code {exc.returncode}: {exc.stderr}"
        )
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
