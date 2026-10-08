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
import math
import os
import platform
import tempfile
import time
import uuid
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Union

from mendeleev import element as mendeleev_element
import psutil
from pydantic import BaseModel, ConfigDict, Field

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
    return _positive_tabulated_property(el.atomic_weight, el.symbol, "atomic_weight")


def _positive_tabulated_property(value: Any, symbol: str, name: str) -> float:
    """Require the named database property; never substitute another definition."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"Mendeleev {name} is unavailable for {symbol}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Mendeleev {name} is not finite and positive for {symbol}.")
    return result


def get_isotopic_mass(symbol: str, mass_number: Optional[int] = None) -> float:
    """Retrieve a tabulated isotope mass without substituting atomic weights.

    Args:
        symbol: Chemical element symbol (e.g. 'C', 'H').
        mass_number: Isotope nucleon number (e.g. 13 for C-13, 2 for Deuterium).

    Returns:
        Tabulated isotope mass in Da; unspecified isotopes use natural abundance.
    """
    from Libraries.cochem_isotopes import isotope_mass

    if mass_number is not None and (type(mass_number) is not int or mass_number <= 0):
        raise ValueError("An isotope mass number must be a positive integer.")
    el = _get_mendeleev_element(symbol)
    label = el.symbol if mass_number is None else f"{mass_number}{el.symbol}"
    return _positive_tabulated_property(isotope_mass(label), el.symbol, "isotope_mass")


def get_covalent_radius(symbol: str) -> float:
    """Retrieve Pyykkö single-bond covalent radius in Angstroms dynamically from Mendeleev."""
    el = _get_mendeleev_element(symbol)
    return _positive_tabulated_property(
        el.covalent_radius_pyykko, el.symbol, "covalent_radius_pyykko"
    ) / 100.0


def get_vdw_radius(symbol: str) -> float:
    """Retrieve van der Waals radius in Angstroms dynamically from Mendeleev."""
    el = _get_mendeleev_element(symbol)
    return _positive_tabulated_property(el.vdw_radius, el.symbol, "vdw_radius") / 100.0


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

# No checkpoint-specific domain manifest is currently qualified for this name.
_MACE_POLAR_Z: Final[Set[int]] = set()

# GFN2-xTB / GFN-FF: Elements Z=1..86 (H to Rn)
_GFN2_XTB_Z: Final[Set[int]] = set(range(1, 87))



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
        description="Unqualified model name; checkpoint-specific elemental coverage is unavailable",
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

}

# Retained for source compatibility only; model substitutions require a new plan.
DEFAULT_MLFF_FALLBACK_CHAIN: Final[List[str]] = []


def resolve_mlff_model(
    symbols: Sequence[str],
    requested_model: Optional[str] = None,
    fallback_chain: Optional[Sequence[str]] = None,
) -> Tuple[MLFFModelSpec, List[str]]:
    """Check one explicitly selected advisory model, without method substitution.

    Historical element lists are candidate metadata, not evidence of a supplied
    checkpoint's chemical/state domain or validated executable capability.
    """
    valid_symbols = validate_element_symbols(symbols)
    if not valid_symbols:
        raise UnsupportedElementError("An advisory model requires a nonempty molecular system.")
    if fallback_chain:
        if len(fallback_chain) != 1 or (requested_model and fallback_chain[0] != requested_model):
            raise ValueError("Automatic model substitution is disabled; select one model in an explicit plan.")
        requested_model = fallback_chain[0]
    if not requested_model:
        raise ValueError("Specify the requested advisory model; no ML model is selected implicitly.")
    if requested_model not in MLFF_CATALOG:
        raise ValueError(f"Unknown or unsupported advisory model: {requested_model}")
    if requested_model == "MACE-POLAR-1":
        raise UnsupportedElementError("MACE-POLAR-1 has no checkpoint-specific validated domain manifest.")
    spec = MLFF_CATALOG[requested_model]
    atomic_numbers = [get_atomic_number(symbol) for symbol in valid_symbols]
    if not spec.supports_elements(atomic_numbers):
        raise UnsupportedElementError(
            f"Requested model {requested_model} does not list elements {spec.get_unsupported_elements(atomic_numbers)}; "
            "no alternative method has been selected."
        )
    return spec, [f"Requested {requested_model}: candidate element coverage only; advisory, unvalidated until checkpoint/domain qualification."]


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
                "Auxiliary basis mismatch for RI-JK: 'def2/J' only provides Coulomb fitting. 'def2/JK' is required for HF exchange fitting."
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
    selected_mlff: Optional[str] = Field(default=None, description="Explicit advisory model, if requested")
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

    # ML is optional and advisory under the accepted D03 policy.
    mlff_spec, fallback_trail = (resolve_mlff_model(valid_symbols, requested_model=requested_mlff)
                                if requested_mlff else (None, ["ML advisory stage not requested."]))

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

    if mlff_spec is not None:
        stages.insert(0, {
            "stage_idx": 0, "stage_name": "ML_Advisory", "model": mlff_spec.name,
            "advisory_only": True, "capability_status": "unvalidated",
            "description": "Optional candidate prioritization; no final observables, pruning or convergence certification.",
        })

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
        selected_mlff=mlff_spec.name if mlff_spec else None,
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
    selected_mlff: Optional[str] = Field(default=None, description="Explicit advisory model, if requested")
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
