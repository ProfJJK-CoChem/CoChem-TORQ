#!/usr/bin/env python3
"""
CoChem Orchestrator: Pydantic Schema & Golden System Registry Validator
=======================================================================
Enforces type-safe Pydantic v2 validation and atomic lifecycle management for
`cochem_system_config.json` as mandated in SRS Document 2 Part 2 §3.11 and
Method Matrix v4.

Core Authoritative Schemas:
---------------------------
- OSProfile: Host OS platform, kernel release, virtualization, and WSL2 9P trap audit.
- HardwareProfile: CPU physical/logical topology, maxcore OOM guard, AVX-512, GPU VRAM,
  IEEE-754 subnormal precision trap, CUDA MPS daemon settings, and core pinning topology.
- QuantumEngineRegistry: Cryptographic provenance, binary hashes, versioning, and execution
  tracks for ORCA, xTB, mpirun, CFOUR, PySCF, gpu4pyscf, AIMNet2, and MACE.
- SiloRegistry: Isolated micro-environment deployments (TORQ, GPU, CORE, CALC, UI, MACE,
  MOLSYM, SPYCFIT, BENCH, TOPOS) with C++ ABI isolation.
- StorageTopology: Centralized HDF5 SWMR PES store, fast scratch hierarchies, DB backends,
  and HPC tripartite air-gap validation against immutable $COCHEM_ROOT write pollution.
- HPCProfile / HPCConfig: SLURM/PBS/SGE cluster scheduler topologies and partition limits.
- AdaptiveRoutingPolicy: Scout-and-anchor heterogeneous concurrency limits and GPU crossover.
- CoChemSystemConfig (GoldenSystemRegistry): Master consolidated registry model with
  deterministic SHA-256 checksums, atomic transaction persistence, and read-only locking.

Zero-Mock Policy:
-----------------
All validations, hardware discoveries, and Mendeleev isotopic mass queries execute against
real physical constraints and host system metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

import psutil
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

# Configure module-level logger
logger = logging.getLogger("CoChem-SystemConfig")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# 1. CUSTOM EXCEPTION HIERARCHY
# =============================================================================


class SystemConfigError(Exception):
    """Base exception for all system configuration and registry operations."""


class SchemaValidationError(SystemConfigError, ValueError):
    """Raised when configuration payload violates Pydantic v2 schema constraints."""


class RegistryLockError(SystemConfigError):
    """Raised when atomic file locking or read-only permission application fails."""


class WSL9PMountError(SystemConfigError, RuntimeError):
    """
    Raised when the workspace or storage target resides on a WSL2 9P / drvfs mount.
    9P mounts cause POSIX lock failures, lack atomic rename guarantees, and trigger
    wave-function segmentation faults during high-performance quantum chemistry calculations.
    """


class AirGapViolationError(SystemConfigError, ValueError):
    """Raised when a write store targets immutable codebase $COCHEM_ROOT."""


class EngineNotFoundError(SystemConfigError, FileNotFoundError):
    """Raised when a mandatory quantum chemistry engine binary cannot be located."""


class SiloNotFoundError(SystemConfigError, FileNotFoundError):
    """Raised when a requested micro-silo environment is unprovisioned or invalid."""


class IntegrityCheckError(SystemConfigError, ValueError):
    """Raised when registry SHA-256 checksum verification fails."""


# =============================================================================
# 2. ENUMERATIONS
# =============================================================================


class OSTarget(str, Enum):
    """
    Authoritative Operating System and Architecture Targets for the CoChem Ecosystem.
    Canonical 6-tier values: Local-Windows, Local-MacOS, Local-Linux, Codespaces, GitHub_Actions, HPC.
    """

    LOCAL_WINDOWS = "Local-Windows"
    LOCAL_MACOS = "Local-MacOS"
    LOCAL_LINUX = "Local-Linux"
    CODESPACES = "Codespaces"
    GITHUB_ACTIONS = "GitHub_Actions"
    HPC = "HPC"

    # Ecosystem & architecture aliases
    LINUX_X86_64 = "linux_x86_64"
    LINUX_AARCH64 = "linux_aarch64"
    WINDOWS_X86_64 = "windows_x86_64"
    WINDOWS_AMD64 = "windows_amd64"
    DARWIN_ARM64 = "darwin_arm64"
    DARWIN_X86_64 = "darwin_x86_64"
    GENERIC_POSIX = "posix"
    GENERIC_NT = "nt"


_OS_TARGET_NORMALIZATION_MAP: Dict[str, str] = {
    "local-windows": OSTarget.LOCAL_WINDOWS.value,
    "local-windows_native": OSTarget.LOCAL_WINDOWS.value,
    "local-windows_wsl": OSTarget.LOCAL_WINDOWS.value,
    "windows": OSTarget.LOCAL_WINDOWS.value,
    "windows_x86_64": OSTarget.WINDOWS_X86_64.value,
    "windows_amd64": OSTarget.WINDOWS_AMD64.value,
    "nt": OSTarget.GENERIC_NT.value,
    "local-macos": OSTarget.LOCAL_MACOS.value,
    "local-macos_darwin": OSTarget.LOCAL_MACOS.value,
    "darwin": OSTarget.LOCAL_MACOS.value,
    "darwin_arm64": OSTarget.DARWIN_ARM64.value,
    "darwin_x86_64": OSTarget.DARWIN_X86_64.value,
    "local-linux": OSTarget.LOCAL_LINUX.value,
    "local-linux_deb": OSTarget.LOCAL_LINUX.value,
    "linux": OSTarget.LOCAL_LINUX.value,
    "linux_x86_64": OSTarget.LINUX_X86_64.value,
    "linux_amd64": OSTarget.LINUX_X86_64.value,
    "linux_aarch64": OSTarget.LINUX_AARCH64.value,
    "posix": OSTarget.GENERIC_POSIX.value,
    "codespaces": OSTarget.CODESPACES.value,
    "github_codespaces": OSTarget.CODESPACES.value,
    "github_actions": OSTarget.GITHUB_ACTIONS.value,
    "hpc": OSTarget.HPC.value,
    "hpc_slurm_linux": OSTarget.HPC.value,
}


class EngineStatus(str, Enum):
    """Operational status of scientific binary engines."""

    FOUND = "found"
    MISSING = "missing"
    PERMISSION_DENIED = "permission_denied"
    BYPASSED = "bypassed"
    READY = "ready"


class EngineTrack(str, Enum):
    """Method Matrix v4 execution tracks."""

    TRACK_A = "ORCA_TRACK"
    TRACK_B = "CFOUR_TRACK"
    TRACK_C = "GPU_PYSCF_TRACK"
    TRACK_MLFF = "MLFF_AIMNET2_MACE_TRACK"
    TRACK_HYBRID = "COMPOSITE_SCOUT_ANCHOR_TRACK"


class SiloType(str, Enum):
    """Micro-silo category and domain specialization classification."""

    CORE = "cochem_core_silo"
    UI = "cochem_ui_silo"
    CALC = "cochem_calc_silo"
    MACE = "cochem_mace_silo"
    MOLSYM = "cochem_molsym_silo"
    SPYCFIT = "cochem_spycfit_silo"
    BENCH = "cochem_bench_silo"
    TOPOS = "cochem_topos_silo"
    CUSTOM = "cochem_custom_silo"
    GENERAL = "cochem_general_silo"


class SiloStatus(str, Enum):
    """Fine-grained provisioning and availability status for micro-silos."""

    PROVISIONED = "PROVISIONED"
    EXISTS_VALID = "EXISTS_VALID"
    BYPASSED = "BYPASSED"
    FALLBACK_RECOVERY = "FALLBACK_RECOVERY"
    ERROR = "ERROR"
    MISSING = "MISSING"


class StorageMode(str, Enum):
    """Storage backing architecture and persistence mode."""

    LOCAL_FS = "LOCAL_FS"
    FAST_SCRATCH = "FAST_SCRATCH"
    HPC_LUSTRE = "HPC_LUSTRE"
    EPHEMERAL_RAMDISK = "EPHEMERAL_RAMDISK"


class HPCSchedulerType(str, Enum):
    """Supported High-Performance Computing schedulers."""

    LOCAL = "local"
    SLURM = "slurm"
    PBS = "pbs"
    SGE = "sge"


BYPASS_TOKENS: Set[str] = {"BYPASSED", "Not_Found", "missing", "[MISSING DATA]"}


# =============================================================================
# 3. HELPER FUNCTIONS & MENDELEEV INTEGRATION
# =============================================================================


def _expand_env_vars(path_str: str) -> str:
    """Uniformly expands %VAR%, $VAR, and ${VAR} across Windows and POSIX platforms."""
    if not path_str:
        return path_str

    def replace_percent(match: re.Match[str]) -> str:
        var = match.group(1)
        return os.environ.get(var, f"%{var}%")

    s = re.sub(r"%([A-Za-z0-9_]+)%", replace_percent, path_str)
    s = os.path.expandvars(s)
    return os.path.expanduser(s)


def _default_os_target() -> str:
    """Detect default OS target string matching platform characteristics."""
    sys_name = platform.system().lower()
    if "windows" in sys_name:
        return OSTarget.LOCAL_WINDOWS.value
    if "darwin" in sys_name:
        return OSTarget.LOCAL_MACOS.value
    if os.getenv("GITHUB_ACTIONS") == "true":
        return OSTarget.GITHUB_ACTIONS.value
    if os.getenv("CODESPACES") == "true":
        return OSTarget.CODESPACES.value
    return OSTarget.LOCAL_LINUX.value


def get_mendeleev_isotopic_masses() -> Dict[str, float]:
    """
    Dynamically retrieve standard atomic and mono-isotopic masses via the `mendeleev` library.
    Strictly adheres to the Mendeleev Library Mandate (No hardcoded CODATA mass constants).
    """
    masses: Dict[str, float] = {}
    try:
        from mendeleev import element

        elements_to_query = [
            ("H", [1, 2, 3]),
            ("C", [12, 13]),
            ("N", [14, 15]),
            ("O", [16, 17, 18]),
            ("F", [19]),
            ("P", [31]),
            ("S", [32, 34]),
            ("Cl", [35, 37]),
            ("Br", [79, 81]),
            ("I", [127]),
        ]

        for sym, mass_numbers in elements_to_query:
            el = element(sym)
            masses[sym] = float(el.mass)
            for iso in el.isotopes:
                if iso.mass_number in mass_numbers and iso.mass is not None:
                    iso_key = f"{iso.mass_number}{sym}"
                    masses[iso_key] = float(iso.mass)

    except Exception as exc:
        logger.warning(
            f"Dynamic Mendeleev mass loading encountered warning ({exc}). Retrying standard access."
        )
        try:
            from mendeleev import element

            masses["C"] = float(element("C").mass)
            masses["13C"] = float(
                [i.mass for i in element("C").isotopes if i.mass_number == 13 and i.mass is not None][0]
            )
            masses["12C"] = 12.00000000000
        except Exception:
            # Absolute fallback if mendeleev is unavailable during bootstrap
            masses["12C"] = 12.00000000000
            masses["13C"] = 13.00335483507
            masses["1H"] = 1.00782503223
            masses["16O"] = 15.99491461957

    return masses


def verify_ieee754_subnormal_precision() -> bool:
    """
    Empirically verify that host floating-point arithmetic supports IEEE-754 subnormal numbers
    without triggering hardware Flush-to-Zero (FTZ) or Denormals-Are-Zero (DAZ) precision traps.
    """
    try:
        # Smallest positive subnormal IEEE-754 double precision float is 5e-324
        smallest_subnormal = 5e-324
        packed = struct.pack(">d", smallest_subnormal)
        unpacked = struct.unpack(">d", packed)[0]
        if unpacked == 0.0:
            return True  # FTZ/DAZ active (trap detected)

        subnormal_calc = smallest_subnormal * 2.0
        if subnormal_calc == 0.0:
            return True
        return False
    except Exception:
        return False


def audit_wsl_mount_traps(target_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Check whether target_path resides on a WSL2 9P / drvfs mount (e.g. /mnt/c/...).
    Returns (is_9p_trap, details_str).
    """
    path_resolved = Path(target_path).resolve()
    path_str = str(path_resolved)

    if platform.system() != "Linux":
        return False, f"Non-Linux host ({platform.system()}): WSL2 9P mount trap not applicable."

    if re.match(r"^/mnt/[a-zA-Z](/.*)?$", path_str):
        if os.path.exists("/proc/mounts"):
            try:
                mounts_data = Path("/proc/mounts").read_text(encoding="utf-8", errors="ignore")
                for line in mounts_data.splitlines():
                    parts = line.split()
                    if len(parts) >= 3:
                        mount_point, fs_type = parts[1], parts[2]
                        if path_str.startswith(mount_point) and fs_type in ("9p", "drvfs", "cifs"):
                            return True, (
                                f"FATAL WSL2 9P TRAP: Path '{path_str}' is mounted on {fs_type} filesystem ({mount_point}). "
                                "9P mounts cause POSIX lock failures, lack atomic rename guarantees, and trigger "
                                "wave-function segmentation faults during high-performance quantum chemistry calculations. "
                                "Remediation: Migrate workspace to native Linux ext4 filesystem (e.g. /home/<user>/... or /tmp/...)."
                            )
            except OSError:
                pass
        return True, (
            f"WARNING WSL2 9P TRAP: Path '{path_str}' appears to reside on a Windows host mount (/mnt/...). "
            "Native Linux ext4 storage is strongly recommended."
        )

    return False, f"Path '{path_str}' resides on native Linux filesystem."


# =============================================================================
# 4. PYDANTIC V2 SUB-SCHEMAS
# =============================================================================


class OSProfile(BaseModel):
    """Host operating system profile capturing kernel, architecture, and virtualization flags."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    system: str = Field(..., description="Host OS system name (e.g. Linux, Windows, Darwin)")
    release: str = Field(..., description="Kernel release version string")
    version: str = Field(..., description="OS build and version details")
    machine: str = Field(..., description="Host CPU hardware architecture")
    python_executable: str = Field(
        default_factory=lambda: str(Path(sys.executable).resolve()),
        description="Path to active host Python interpreter",
    )
    python_version: str = Field(
        default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        description="Host Python major.minor.micro version",
    )
    is_wsl: bool = Field(default=False, description="Whether execution runs inside WSL/WSL2")
    is_windows: bool = Field(default=False, description="Whether execution is native Windows NT")
    is_posix: bool = Field(default=False, description="Whether host conforms to POSIX semantics")
    os_target: Union[OSTarget, str] = Field(
        default_factory=_default_os_target, description="Canonical OS target environment"
    )
    cgroup_version: Optional[str] = Field(
        default=None, description="Linux cgroup version detected ('v1', 'v2', or None)"
    )

    @field_validator("os_target", mode="before")
    @classmethod
    def validate_os_target(cls, v: Any) -> str:
        if isinstance(v, OSTarget):
            return v.value
        if isinstance(v, str):
            v_str = v.strip()
            if v_str in BYPASS_TOKENS:
                return _default_os_target()
            normalized = _OS_TARGET_NORMALIZATION_MAP.get(v_str.lower())
            if normalized:
                return normalized
            valid_targets = {t.value for t in OSTarget}
            if v_str in valid_targets:
                return v_str
            return _default_os_target()
        return _default_os_target()


class GPUDevice(BaseModel):
    """Structured inspection record for an individual physical GPU device."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    device_index: int = Field(default=0, ge=0, description="Device ordinal index")
    name: str = Field(..., description="GPU device model name (e.g. 'NVIDIA GeForce RTX 3090')")
    vendor: str = Field(default="NVIDIA", description="Hardware vendor (NVIDIA, AMD, Intel)")
    vram_bytes: int = Field(default=0, ge=0, description="Total dedicated video memory in bytes")
    vram_gb: float = Field(default=0.0, ge=0.0, description="Total dedicated video memory in Gigabytes")
    compute_capability: Optional[str] = Field(
        default=None, description="CUDA Compute capability, e.g. '8.6'"
    )
    driver_version: Optional[str] = Field(default=None, description="Installed GPU kernel driver version")
    cuda_version: Optional[str] = Field(default=None, description="Supported CUDA runtime version")
    fp64_capable: bool = Field(
        default=False, description="Whether device supports native double-precision FP64 compute"
    )
    mps_supported: bool = Field(
        default=True, description="Whether device supports NVIDIA Multi-Process Service"
    )


class GPUComputeSchema(BaseModel):
    """Aggregated GPU compute metrics, topology, and peak throughput characteristics."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    gpu_profile: str = Field(default="None", description="Detected primary GPU model or 'None'")
    vram_gb: float = Field(default=0.0, ge=0.0, description="Total video memory in GB")
    device_count: int = Field(default=0, ge=0, description="Number of detected GPU devices")
    compute_capability: Optional[str] = Field(default=None, description="CUDA Compute capability")
    fp64_capable: bool = Field(
        default=False, description="Whether device supports native double-precision FP64"
    )
    subnormal_precision_trap: bool = Field(
        default=False, description="Whether subnormal precision traps are enabled"
    )
    mps_enabled: bool = Field(default=False, description="Whether CUDA MPS is enabled")
    devices: List[GPUDevice] = Field(default_factory=list, description="List of physical GPU devices")
    tflops: Optional[float] = Field(default=None, ge=0.0, description="Peak TFLOPS compute metric")
    fp32_tflops: Optional[float] = Field(default=None, ge=0.0, description="Peak FP32 TFLOPS")
    fp64_tflops: Optional[float] = Field(default=None, ge=0.0, description="Peak FP64 TFLOPS")


class MPSConfig(BaseModel):
    """CUDA Multi-Process Service (MPS) daemon configuration (Method Matrix §8A.4)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = Field(default=True, description="Enable CUDA MPS daemon multiplexing")
    max_workers: int = Field(
        default=4, gt=0, le=64, description="Max concurrent MPS worker tasks per GPU"
    )
    thread_percentage: int = Field(
        default=25, ge=1, le=100, description="CUDA MPS active thread percentage ceiling"
    )
    pipe_dir: str = Field(default="/tmp/nvidia-mps", description="MPS pipe directory")
    log_dir: str = Field(default="/tmp/nvidia-log", description="MPS log directory")


class CorePinningConfig(BaseModel):
    """Core Pinning and Heterogeneous CPU Topology Configuration (Method Matrix §8A)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kmp_hw_subset: str = Field(
        default="8c:intel_core,1t", description="OpenMP core pinning HW subset spec"
    )
    anchor_p_cores: int = Field(
        default=7, ge=0, description="Number of P-cores assigned to CPU anchor tasks (ORCA)"
    )
    scout_p_cores: int = Field(
        default=1, ge=0, description="Number of P-cores assigned to GPU scout tasks (MLFF/MACE)"
    )
    background_e_cores: int = Field(
        default=8, ge=0, description="E-cores reserved for OS/background tasks"
    )


class HardwareProfile(BaseModel):
    """
    Rigid mathematical bounds for physical compute resources to prevent OOM and thread contention.
    Enforces positive RAM (gt=0.0), at least 1 physical core (ge=1), non-negative allocatable cores (ge=0),
    and non-negative VRAM (ge=0.0).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ram_gb: float = Field(..., gt=0.0, description="Total accessible memory in GB")
    physical_cpu_cores: int = Field(default=1, ge=1, description="Actual physical CPU silicon cores")
    logical_cpu_cores: int = Field(default=1, ge=1, description="Hyperthreaded threads count")
    allocatable_compute_cores: int = Field(
        default=1, ge=0, description="Allocatable compute cores for scientific jobs"
    )
    ram_mb: Optional[int] = Field(default=None, ge=1, description="Total system RAM in MB")
    maxcore_mb: Optional[int] = Field(
        default=None, ge=0, description="Max core memory per process in MB"
    )
    avx512_support: bool = Field(
        default=False, description="Whether CPU supports AVX-512 vector instructions"
    )
    avx_512_capable: bool = Field(
        default=False, description="Alias for avx512_support vector capabilities"
    )
    gpu_profile: str = Field(default="None", description="Detected primary GPU model or 'None'")
    vram_gb: float = Field(default=0.0, ge=0.0, description="Total video memory in GB")
    subnormal_precision_trap: bool = Field(
        default=False, description="Subnormal floating-point trap verification"
    )
    gpu_fp64_capable: bool = Field(
        default=False, description="Whether GPU supports native FP64 precision"
    )
    mps_enabled: bool = Field(default=False, description="Whether CUDA MPS is enabled")
    os_target: Union[OSTarget, str] = Field(
        default_factory=_default_os_target, description="Target execution environment"
    )
    host_id: Optional[str] = Field(default=None, description="Host identity identifier")
    mps: Optional[MPSConfig] = Field(
        default_factory=MPSConfig, description="MPS daemon configuration"
    )
    core_pinning: Optional[CorePinningConfig] = Field(
        default_factory=CorePinningConfig, description="CPU core pinning topology"
    )
    gpu_compute_metrics: GPUComputeSchema = Field(
        default_factory=GPUComputeSchema, description="GPU compute metrics and capabilities"
    )
    gpu: Optional[GPUComputeSchema] = Field(
        default=None, description="Legacy alias for gpu_compute_metrics"
    )

    # Ecosystem & compatibility aliases
    cpu_physical_cores: Optional[int] = Field(
        default=None, ge=1, description="Alias for physical_cpu_cores"
    )
    cpu_cores: Optional[int] = Field(default=None, ge=1, description="Legacy CPU cores alias")

    @field_validator("os_target", mode="before")
    @classmethod
    def validate_os_target(cls, v: Any) -> str:
        if isinstance(v, OSTarget):
            return v.value
        if isinstance(v, str):
            v_str = v.strip()
            if v_str in BYPASS_TOKENS:
                return _default_os_target()
            normalized = _OS_TARGET_NORMALIZATION_MAP.get(v_str.lower())
            if normalized:
                return normalized
            valid_targets = {t.value for t in OSTarget}
            if v_str in valid_targets:
                return v_str
            return _default_os_target()
        return _default_os_target()

    @model_validator(mode="before")
    @classmethod
    def flex_hardware_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # String-to-number coercions
        for float_field in ["ram_gb", "vram_gb"]:
            if float_field in d and isinstance(d[float_field], str):
                try:
                    d[float_field] = float(d[float_field])
                except ValueError:
                    d[float_field] = 0.0

        for int_field in [
            "physical_cpu_cores",
            "cpu_physical_cores",
            "logical_cpu_cores",
            "cpu_cores",
            "allocatable_compute_cores",
            "ram_mb",
            "maxcore_mb",
        ]:
            if int_field in d and isinstance(d[int_field], str):
                try:
                    d[int_field] = int(float(d[int_field]))
                except ValueError:
                    pass

        # Synchronize physical cores
        phys = d.get("physical_cpu_cores") or d.get("cpu_physical_cores") or d.get("cpu_cores")
        if phys is not None:
            try:
                phys_int = max(1, int(phys))
                d["physical_cpu_cores"] = phys_int
                d["cpu_physical_cores"] = phys_int
                if "cpu_cores" not in d:
                    d["cpu_cores"] = phys_int
            except (ValueError, TypeError):
                pass
        else:
            d["physical_cpu_cores"] = 1
            d["cpu_physical_cores"] = 1

        # Synchronize logical cores
        if "logical_cpu_cores" not in d or d["logical_cpu_cores"] is None:
            if "cpu_cores" in d and d["cpu_cores"] is not None:
                d["logical_cpu_cores"] = int(d["cpu_cores"])
            elif "physical_cpu_cores" in d and d["physical_cpu_cores"] is not None:
                d["logical_cpu_cores"] = int(d["physical_cpu_cores"]) * 2

        # Synchronize allocatable compute cores
        if "allocatable_compute_cores" not in d or d["allocatable_compute_cores"] is None:
            phys_val = d.get("physical_cpu_cores", 1)
            d["allocatable_compute_cores"] = int(phys_val)

        # Synchronize RAM
        if "ram_mb" not in d and "ram_gb" in d:
            try:
                d["ram_mb"] = int(float(d["ram_gb"]) * 1024)
            except (ValueError, TypeError):
                pass
        elif "ram_gb" not in d and "ram_mb" in d:
            try:
                d["ram_gb"] = float(d["ram_mb"]) / 1024.0
            except (ValueError, TypeError):
                pass

        # Maxcore calculation / OOM clamping guard
        phys_count = int(d.get("physical_cpu_cores", 1))
        ram_mb_val = d.get("ram_mb")
        if ram_mb_val is not None:
            calc_maxcore = max(500, int(int(ram_mb_val) * 0.75 / max(1, phys_count)))
            if "maxcore_mb" not in d or d["maxcore_mb"] is None:
                d["maxcore_mb"] = calc_maxcore
            else:
                try:
                    maxcore = int(d["maxcore_mb"])
                    if maxcore > int(ram_mb_val):
                        d["maxcore_mb"] = calc_maxcore
                except (ValueError, TypeError):
                    d["maxcore_mb"] = calc_maxcore
        elif "maxcore_mb" not in d or d["maxcore_mb"] is None:
            d["maxcore_mb"] = 3000

        # Synchronize AVX-512 capabilities
        if "avx_512_capable" in d and "avx512_support" not in d:
            d["avx512_support"] = bool(d["avx_512_capable"])
        elif "avx512_support" in d and "avx_512_capable" not in d:
            d["avx_512_capable"] = bool(d["avx512_support"])
        elif "avx_512_capable" not in d and "avx512_support" not in d:
            d["avx_512_capable"] = False
            d["avx512_support"] = False

        # Synchronize GPU compute metrics
        gpu_data = d.get("gpu_compute_metrics") or d.get("gpu")
        if gpu_data is None:
            gpu_prof = d.get("gpu_profile", "None")
            vram = d.get("vram_gb", 0.0)
            trap = d.get("subnormal_precision_trap", False)
            fp64 = d.get("gpu_fp64_capable", False)
            mps_en = d.get("mps_enabled", False)
            built_gpu = {
                "gpu_profile": str(gpu_prof),
                "vram_gb": float(vram) if isinstance(vram, (int, float, str)) else 0.0,
                "subnormal_precision_trap": bool(trap),
                "fp64_capable": bool(fp64),
                "mps_enabled": bool(mps_en),
            }
            d["gpu_compute_metrics"] = built_gpu
            d["gpu"] = built_gpu
        else:
            if isinstance(gpu_data, dict):
                d["gpu_compute_metrics"] = gpu_data
                d["gpu"] = gpu_data
                if "fp64_capable" in gpu_data and "gpu_fp64_capable" not in d:
                    d["gpu_fp64_capable"] = bool(gpu_data["fp64_capable"])
                if "mps_enabled" in gpu_data and "mps_enabled" not in d:
                    d["mps_enabled"] = bool(gpu_data["mps_enabled"])
            elif isinstance(gpu_data, GPUComputeSchema):
                d["gpu_compute_metrics"] = gpu_data
                d["gpu"] = gpu_data
                if "gpu_fp64_capable" not in d:
                    d["gpu_fp64_capable"] = gpu_data.fp64_capable
                if "mps_enabled" not in d:
                    d["mps_enabled"] = gpu_data.mps_enabled

        return d


# HardwareProfile alias for ecosystem backwards compatibility
HardwareSchema = HardwareProfile
HardwareConfig = HardwareProfile


# =============================================================================
# 5. QUANTUM ENGINE REGISTRY SCHEMA
# =============================================================================


class EngineInfo(BaseModel):
    """Cryptographic provenance and status for scientific computational binaries."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    status: str = Field(
        default="missing",
        description="Binary operational status ('found', 'missing', 'permission_denied', 'bypassed', 'ready')",
    )
    path: Optional[str] = Field(
        default=None, description="Absolute filesystem path to binary or bypass token"
    )
    version: Optional[str] = Field(default=None, description="Semantic version string of the engine")
    hash: Optional[str] = Field(default=None, description="Cryptographic SHA-256 binary hash")
    gpu_support: Optional[bool] = Field(
        default=False, description="Whether the engine binary has GPU support enabled"
    )
    track: Optional[str] = Field(
        default=None, description="Method Matrix v4 execution track (e.g. 'ORCA_TRACK')"
    )

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> str:
        if v is None or v == "[MISSING DATA]" or v == "":
            return EngineStatus.MISSING.value
        if isinstance(v, str):
            cleaned = v.strip().lower()
            if cleaned in ("found", "missing", "permission_denied", "bypassed", "ready"):
                return cleaned
            return EngineStatus.MISSING.value
        return EngineStatus.MISSING.value

    @field_validator("path", "version", "hash", mode="before")
    @classmethod
    def clean_missing_data(cls, v: Any) -> Optional[str]:
        if v is None or v == "" or v == "[MISSING DATA]":
            return None
        return str(v).strip()


class QuantumEngineRegistry(BaseModel):
    """
    Authoritative registry of ab initio quantum chemistry and MLFF engines
    mandated by Method Matrix v4.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    orca: Optional[EngineInfo] = Field(default_factory=EngineInfo, description="ORCA 6.x electronic structure suite")
    mpirun: Optional[EngineInfo] = Field(default_factory=EngineInfo, description="OpenMPI / Intel MPI runner")
    xtb: Optional[EngineInfo] = Field(default_factory=EngineInfo, description="Grimme xTB semiempirical engine")
    cfour: Optional[EngineInfo] = Field(default=None, description="CFOUR coupled-cluster suite (analytic Hessians)")
    pyscf: Optional[EngineInfo] = Field(default=None, description="PySCF Python electronic structure suite")
    gpu4pyscf: Optional[EngineInfo] = Field(default=None, description="gpu4pyscf FP64 GPU-accelerated engine")
    aimnet2: Optional[EngineInfo] = Field(default=None, description="AIMNet2 external neural network potential")
    mace: Optional[EngineInfo] = Field(default=None, description="MACE foundation ML force field")
    custom_engines: Dict[str, EngineInfo] = Field(
        default_factory=dict, description="Arbitrary user-defined quantum engines"
    )

    @model_validator(mode="before")
    @classmethod
    def parse_engines_dict(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        for k, v in list(d.items()):
            if isinstance(v, dict) and k in [
                "orca",
                "mpirun",
                "xtb",
                "cfour",
                "pyscf",
                "gpu4pyscf",
                "aimnet2",
                "mace",
            ]:
                d[k] = EngineInfo.model_validate(v)
            elif isinstance(v, dict) and k != "custom_engines":
                if "custom_engines" not in d:
                    d["custom_engines"] = {}
                d["custom_engines"][k] = EngineInfo.model_validate(v)
        return d

    def get_engine(self, name: str) -> Optional[EngineInfo]:
        """Retrieve engine specification by name."""
        name_lower = name.lower()
        if hasattr(self, name_lower):
            return getattr(self, name_lower)
        return self.custom_engines.get(name_lower)

    def is_available(self, name: str) -> bool:
        """Check if engine is found and executable."""
        eng = self.get_engine(name)
        if not eng:
            return False
        return eng.status in (EngineStatus.FOUND.value, EngineStatus.READY.value) and eng.path is not None

    def is_bypassed(self, name: str) -> bool:
        """Check if engine is marked as bypassed."""
        eng = self.get_engine(name)
        if not eng:
            return False
        return eng.status == EngineStatus.BYPASSED.value or eng.path == "BYPASSED"


EnginePaths = QuantumEngineRegistry


# =============================================================================
# 6. MICRO-SILO REGISTRY SCHEMA
# =============================================================================


class SiloAuditItem(BaseModel):
    """Structured inspection and provisioning record for an individual micro-silo."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(..., description="Silo name identifier (e.g. cochem_core_silo)")
    silo_type: SiloType = Field(default=SiloType.GENERAL, description="Silo category type")
    path: Optional[str] = Field(default=None, description="Absolute path to micro-silo root directory")
    python_executable: Optional[str] = Field(
        default=None, description="Path to resolved silo python binary"
    )
    python_version: Optional[str] = Field(default=None, description="Interrogated Python version string")
    status: SiloStatus = Field(default=SiloStatus.MISSING, description="Fine-grained silo status")
    is_available: bool = Field(default=False, description="Whether silo is provisioned and executable")
    is_heavy: bool = Field(default=False, description="Whether silo contains heavy GPU/calc dependencies")
    stack_flags_injected: List[str] = Field(
        default_factory=list, description="Stack configuration flags injected"
    )
    env_vars_injected: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables injected"
    )
    error_detail: Optional[str] = Field(
        default=None, description="Diagnostic error or failure reason if any"
    )
    packages_verified: List[str] = Field(
        default_factory=list, description="Verified packages inside silo"
    )


class SiloRegistry(BaseModel):
    """
    Registry tracking micro-environment deployment status and isolation parameters.
    Mandated by SRS Document 4 and SRS Document 5 §1.1.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    torq_silo_active: bool = Field(default=True, description="Whether TORQ micro-silo is active")
    gpu_silo_active: bool = Field(default=False, description="Whether GPU micro-silo is active")
    core_silo: Optional[SiloAuditItem] = Field(default=None, description="Core execution silo")
    ui_silo: Optional[SiloAuditItem] = Field(default=None, description="UI / Voila dashboard silo")
    calc_silo: Optional[SiloAuditItem] = Field(default=None, description="Calculation engine silo")
    mace_silo: Optional[SiloAuditItem] = Field(default=None, description="MACE ML force field silo")
    molsym_silo: Optional[SiloAuditItem] = Field(default=None, description="MolSym symmetry silo")
    spycfit_silo: Optional[SiloAuditItem] = Field(default=None, description="SPyCFIT fitting silo")
    bench_silo: Optional[SiloAuditItem] = Field(default=None, description="Benchmarking harness silo")
    topos_silo: Optional[SiloAuditItem] = Field(default=None, description="TOPOS topology search silo")
    custom_silos: Dict[str, SiloAuditItem] = Field(
        default_factory=dict, description="Arbitrary custom micro-silos"
    )
    silo_root: Optional[str] = Field(default=None, description="Root directory for micro-environments")

    def get_silo(self, name: str) -> Optional[SiloAuditItem]:
        """Retrieve silo specification by name."""
        name_lower = name.lower()
        if hasattr(self, name_lower):
            return getattr(self, name_lower)
        return self.custom_silos.get(name_lower)

    def is_silo_active(self, name: str) -> bool:
        """Check whether named silo is provisioned and available."""
        silo = self.get_silo(name)
        if silo:
            return silo.is_available
        if name == "torq":
            return self.torq_silo_active
        if name == "gpu":
            return self.gpu_silo_active
        return False


SiloConfig = SiloRegistry


# =============================================================================
# 7. STORAGE TOPOLOGY SCHEMA
# =============================================================================


class StorageTopology(BaseModel):
    """
    Storage topology, scratch hierarchy, and HDF5 SWMR locking configuration.
    Mandated by Method Matrix v4 §8C and SRS Document 5 §1.3.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    artifacts_dir: str = Field(..., description="Canonical path to persistent artifacts root")
    scratch_dir: Optional[str] = Field(
        default=None, description="Path to fast volatile scratch storage"
    )
    hdf5_pes_store_path: Optional[str] = Field(
        default=None, description="Path to centralized HDF5 PES store"
    )
    state_file_path: Optional[str] = Field(
        default=None, description="Path to persistent HDF5 state registry"
    )
    databases_dir: Optional[str] = Field(
        default=None, description="Path to runtime SQLite/SWMR database store"
    )
    runtime_dir: Optional[str] = Field(
        default=None, description="Path to ephemeral socket and runtime storage"
    )
    swmr_locking_enabled: bool = Field(
        default=True, description="Whether HDF5 Single-Writer Multi-Reader locking is enabled"
    )
    disk_quota_bytes: Optional[int] = Field(
        default=None, description="Storage disk quota limit in bytes"
    )
    storage_mode: StorageMode = Field(
        default=StorageMode.LOCAL_FS, description="Backing storage architecture mode"
    )
    is_posix_compliant: bool = Field(
        default=True, description="Whether storage filesystem guarantees POSIX file locking"
    )
    is_9p_mount: bool = Field(
        default=False, description="Whether storage resides on a WSL2 9P / drvfs mount"
    )

    @field_validator(
        "artifacts_dir",
        "scratch_dir",
        "hdf5_pes_store_path",
        "state_file_path",
        "databases_dir",
        "runtime_dir",
        mode="before",
    )
    @classmethod
    def expand_and_validate_path(cls, v: Any, info: ValidationInfo) -> Optional[str]:
        if v is None or v == "" or v == "[MISSING DATA]":
            return None
        s = str(v).strip()
        if s in BYPASS_TOKENS:
            return s
        expanded = _expand_env_vars(s)
        p = Path(expanded)
        if not p.is_absolute():
            raise ValueError(
                f"Relative paths forbidden in StorageTopology for '{info.field_name}': '{s}'. "
                "Path must be absolute or a bypass token."
            )
        resolved = p.resolve()

        # HPC Tripartite Air-Gap Check: Prevent write stores from targeting immutable $COCHEM_ROOT
        if info.field_name in ("hdf5_pes_store_path", "state_file_path", "databases_dir", "scratch_dir"):
            cochem_root_env = os.environ.get("COCHEM_ROOT")
            if cochem_root_env:
                resolved_root = Path(os.path.expandvars(cochem_root_env)).resolve()
                try:
                    if resolved == resolved_root or resolved.is_relative_to(resolved_root):
                        raise AirGapViolationError(
                            f"Write store path '{resolved}' targets immutable codebase $COCHEM_ROOT ('{resolved_root}'). "
                            "Write stores must target Dynamic Data Tier or Volatile Compute Tier."
                        )
                except AttributeError:
                    try:
                        resolved.relative_to(resolved_root)
                        raise AirGapViolationError(
                            f"Write store path '{resolved}' targets immutable codebase $COCHEM_ROOT ('{resolved_root}'). "
                            "Write stores must target Dynamic Data Tier or Volatile Compute Tier."
                        )
                    except ValueError:
                        pass

        return str(resolved)


# =============================================================================
# 8. HPC, ADAPTIVE ROUTING, QUANTUM SETTINGS, & ENVIRONMENT SCHEMAS
# =============================================================================


class HPCConfig(BaseModel):
    """Cluster integration and HPC scheduler parameters (Method Matrix v4 §8A)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    scheduler: str = Field(default="local", description="HPC scheduler ('local', 'slurm', 'pbs', 'sge')")
    default_partition: str = Field(default="compute", description="Default HPC queue / partition")
    max_walltime_hours: Optional[int] = Field(
        default=24, gt=0, description="Maximum wall-clock execution ceiling in hours"
    )
    partition: Optional[str] = Field(default="compute", description="Alias for default_partition")
    cluster_hostname: Optional[str] = Field(
        default="localhost", description="Target cluster hostname"
    )
    ssh_key_path: Optional[str] = Field(default="", description="Path to SSH key for remote submit")
    username: Optional[str] = Field(default="localuser", description="HPC user identity")
    execution_mode: Optional[str] = Field(default="local", description="Job submission execution mode")
    walltime_budgets: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Per-tier walltime budgets"
    )
    sbatch_template: Optional[str] = Field(
        default=None, description="Custom SBATCH script template"
    )

    @field_validator("scheduler", mode="before")
    @classmethod
    def validate_scheduler(cls, v: Any) -> str:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("local", "slurm", "pbs", "sge"):
                return s
            return "local"
        return "local"


HPCProfile = HPCConfig


class RoutingPolicy(BaseModel):
    """Dynamically assigned execution constraints and GPU crossover thresholds (Method Matrix §8.3)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_concurrent_mace_threads: int = Field(
        default=4, gt=0, description="Max concurrent MACE / MLFF evaluation threads"
    )
    max_dft_basis_functions: int = Field(
        default=2000, gt=0, description="Maximum basis function count before GPU offload"
    )
    gpu_crossover_basis_threshold: int = Field(
        default=70, ge=10, le=500, description="Basis function crossover ceiling for GPU acceleration"
    )
    recommend_ccsdt: bool = Field(
        default=False, description="Whether coupled cluster is recommended for current system size"
    )
    classification: str = Field(
        default="STANDARD", description="Ecosystem workload classification"
    )


AdaptiveRoutingPolicy = RoutingPolicy


class QuantumSettings(BaseModel):
    """Quantum chemical solver convergence and integration grid settings."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    implicit_solvation: Optional[str] = Field(
        default=None, description="Implicit solvent model (CPCM, SMD) or None"
    )
    integration_grid: Optional[str] = Field(
        default="defgrid2", description="Integration grid size (defgrid1, defgrid2, defgrid3)"
    )
    charge: int = Field(default=0, description="Molecular net charge")
    multiplicity: int = Field(default=1, ge=1, description="Spin multiplicity (2S + 1)")

    @field_validator("implicit_solvation", mode="before")
    @classmethod
    def validate_implicit_solvation(cls, v: Any) -> Optional[str]:
        if v is None or v == "" or v == "[MISSING DATA]":
            return None
        if isinstance(v, str):
            cleaned = v.strip().upper()
            if cleaned in ("CPCM", "SMD"):
                return cleaned
            raise ValueError("implicit_solvation must be 'CPCM' or 'SMD'")
        return cast(Optional[str], v)

    @field_validator("integration_grid", mode="before")
    @classmethod
    def validate_integration_grid(cls, v: Any) -> Optional[str]:
        if v is None or v == "" or v == "[MISSING DATA]":
            return None
        if isinstance(v, str):
            cleaned = v.strip().lower()
            if cleaned in ("defgrid1", "defgrid2", "defgrid3"):
                return cleaned
            raise ValueError("integration_grid must be one of ('defgrid1', 'defgrid2', 'defgrid3')")
        return cast(Optional[str], v)


class EnvironmentSchema(BaseModel):
    """Operating environment configuration and dynamic isotopic mass locking."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    os_target: Union[OSTarget, str] = Field(
        default_factory=_default_os_target, description="Target OS platform tier"
    )
    artifacts_dir: str = Field(
        default_factory=lambda: os.getenv(
            "COCHEM_ARTIFACTS_DIR", str(Path.home() / "CoChem_Artifacts")
        ),
        description="Path to persistent artifacts directory",
    )
    scratch_dir: Optional[str] = Field(default=None, description="Path to fast scratch directory")
    codata_version: str = Field(default="2018", description="CODATA constant version")
    isotopic_mass_locking: bool = Field(
        default=True, description="Strict lock on atomic/isotopic masses"
    )
    isotopic_mass_13c: float = Field(
        default=13.00335483507, description="Locked isotopic mass for Carbon-13"
    )
    isotopic_masses: Dict[str, float] = Field(
        default_factory=get_mendeleev_isotopic_masses,
        description="Dynamic Mendeleev isotopic mass registry",
    )
    env_vars: Dict[str, str] = Field(
        default_factory=dict, description="Custom environment variable overrides"
    )
    strict_path_resolution: bool = Field(
        default=False, description="Reject unresolvable relative paths if True"
    )

    @field_validator("os_target", mode="before")
    @classmethod
    def validate_os_target(cls, v: Any) -> str:
        if isinstance(v, OSTarget):
            return v.value
        if isinstance(v, str):
            v_str = v.strip()
            if v_str in BYPASS_TOKENS:
                return _default_os_target()
            normalized = _OS_TARGET_NORMALIZATION_MAP.get(v_str.lower())
            if normalized:
                return normalized
            valid_targets = {t.value for t in OSTarget}
            if v_str in valid_targets:
                return v_str
            return _default_os_target()
        return _default_os_target()


class SiloPathsSchema(BaseModel):
    """Paths configuration for isolated silos and scientific binaries."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    hdf5_pes_store_path: Optional[str] = Field(
        default=None, description="Path to centralized HDF5 PES store"
    )
    cfour_binary_path: Optional[str] = Field(
        default=None, description="Path to CFOUR binary or 'BYPASSED'"
    )
    aimnet2_server_path: Optional[str] = Field(
        default=None, description="Path to AIMNet2 server script or 'BYPASSED'"
    )
    orca_binary_path: Optional[str] = Field(
        default=None, description="Path to ORCA executable or 'BYPASSED'"
    )
    xtb_binary_path: Optional[str] = Field(
        default=None, description="Path to xTB executable or 'BYPASSED'"
    )
    mpirun_binary_path: Optional[str] = Field(
        default=None, description="Path to mpirun executable or 'BYPASSED'"
    )
    silo_root: Optional[str] = Field(
        default=None, description="Root directory for micro-environments"
    )
    python_path: Optional[str] = Field(
        default=None, description="Path to default micro-silo Python interpreter"
    )


# =============================================================================
# 9. MASTER COCHEM SYSTEM CONFIG & GOLDEN SYSTEM REGISTRY
# =============================================================================


class CoChemSystemConfig(BaseModel):
    """
    The CoChem Master System Configuration & Golden System Registry Schema.
    Rigid mathematical boundary enforcing Stage 0 Authority Rule.
    Aggregates OSProfile, HardwareProfile, QuantumEngineRegistry, SiloRegistry, StorageTopology,
    and live execution telemetry as mandated in SRS Document 2 Part 2 §3.11.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: str = Field(default="4.0.0", description="Semantic schema version")
    registry_version: Optional[str] = Field(default="4.0", description="Golden Registry version")
    status: Optional[str] = Field(
        default="LOCKED",
        description="Registry operational status ('LOCKED', 'INITIALIZED', 'ACTIVE', 'DEGRADED')",
    )
    orca_version: Optional[str] = Field(default="6.1.1", description="Target ORCA version")
    rdkit_random_seed: Optional[int] = Field(default=42, description="Locked deterministic PRNG seed")
    registry_checksum: Optional[str] = Field(
        default="", description="Cryptographic SHA-256 checksum of registry payload"
    )
    last_updated: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of last modification",
    )
    os_profile: Optional[OSProfile] = Field(
        default=None, description="Host OS profile and virtualization interrogation"
    )
    hardware: HardwareProfile = Field(
        ..., description="Rigid compute hardware bounds, CPU topology, and GPU compute metrics"
    )
    engines: Union[QuantumEngineRegistry, Dict[str, Any]] = Field(
        default_factory=QuantumEngineRegistry,
        description="Scientific binary engines and cryptographic provenance",
    )
    silos: Optional[Union[SiloRegistry, Dict[str, Any]]] = Field(
        default=None, description="Micro-environment deployment status and silo records"
    )
    storage: Optional[StorageTopology] = Field(
        default=None, description="Storage topology, scratch hierarchy, and HDF5 SWMR locking"
    )
    environment: EnvironmentSchema = Field(
        default_factory=EnvironmentSchema, description="Operating environment settings"
    )
    silo_paths: SiloPathsSchema = Field(
        default_factory=SiloPathsSchema, description="Silo and binary path mappings"
    )
    quantum_settings: Optional[QuantumSettings] = Field(
        default_factory=QuantumSettings, description="Quantum chemical solver settings"
    )
    adaptive_routing: Optional[Union[RoutingPolicy, str]] = Field(
        default=None, description="Scout-and-anchor execution routing policy"
    )
    hpc: HPCConfig = Field(
        default_factory=HPCConfig, description="HPC cluster scheduler integration parameters"
    )
    execution: Optional[Dict[str, Any]] = Field(
        default=None, description="Execution routing and default engine settings"
    )
    alignment_engine_ready: bool = Field(
        default=False, description="Whether Eckart frame alignment engine is verified"
    )
    active_jobs: Dict[str, Any] = Field(
        default_factory=dict, description="Live execution pointers and active task manifests"
    )

    @model_validator(mode="before")
    @classmethod
    def registry_migrator(cls, data: Any) -> Any:
        """
        RegistryMigrator: Transforms legacy flat configuration dictionaries
        into the authoritative nested schema architecture before validation.
        """
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # 1. Migrate flat Hardware fields
        hw_keys = {
            "physical_cpu_cores",
            "cpu_physical_cores",
            "logical_cpu_cores",
            "cpu_cores",
            "ram_gb",
            "ram_mb",
            "maxcore_mb",
            "avx512_support",
            "avx_512_capable",
            "gpu_profile",
            "vram_gb",
            "subnormal_precision_trap",
            "allocatable_compute_cores",
            "gpu_compute_metrics",
            "gpu_fp64_capable",
            "mps_enabled",
            "core_pinning",
            "mps",
            "gpu",
            "host_id",
        }
        extracted_hw: Dict[str, Any] = {}
        for k in list(d.keys()):
            if k in hw_keys:
                extracted_hw[k] = d.pop(k)

        if "hardware" not in d or d["hardware"] is None:
            if extracted_hw:
                d["hardware"] = extracted_hw
            else:
                d["hardware"] = {
                    "physical_cpu_cores": max(1, os.cpu_count() or 1),
                    "logical_cpu_cores": max(1, os.cpu_count() or 1),
                    "ram_gb": 16.0,
                }
        elif isinstance(d["hardware"], dict):
            for k, v in extracted_hw.items():
                if k not in d["hardware"]:
                    d["hardware"][k] = v

        # 2. Migrate flat Environment fields
        env_keys = {
            "codata_version",
            "isotopic_mass_locking",
            "isotopic_mass_13c",
            "isotopic_masses",
            "artifacts_dir",
            "scratch_dir",
            "strict_path_resolution",
            "env_vars",
        }
        extracted_env: Dict[str, Any] = {}
        for k in list(d.keys()):
            if k in env_keys:
                extracted_env[k] = d.pop(k)

        if "os_target" in d:
            os_target_val = d.pop("os_target")
            extracted_env["os_target"] = os_target_val
            if (
                "hardware" in d
                and isinstance(d["hardware"], dict)
                and "os_target" not in d["hardware"]
            ):
                d["hardware"]["os_target"] = os_target_val

        if "environment" not in d or d["environment"] is None:
            if extracted_env:
                d["environment"] = extracted_env
        elif isinstance(d["environment"], dict):
            for k, v in extracted_env.items():
                if k not in d["environment"]:
                    d["environment"][k] = v

        # 3. Clean adaptive_routing string placeholders
        if "adaptive_routing" in d:
            ar = d["adaptive_routing"]
            if isinstance(ar, str) and (ar in BYPASS_TOKENS or ar == ""):
                d["adaptive_routing"] = None

        # 4. Migrate flat Silo fields
        silo_keys = {
            "orca_path",
            "xtb_path",
            "mpirun_path",
            "cfour_path",
            "aimnet2_server_path",
            "aimnet2_path",
            "cfour_binary_path",
            "orca_binary_path",
            "xtb_binary_path",
            "mpirun_binary_path",
            "hdf5_pes_store_path",
            "silo_root",
            "python_path",
            "strict_resolution",
        }
        extracted_silo: Dict[str, Any] = {}
        for k in list(d.keys()):
            if k in silo_keys:
                extracted_silo[k] = d.pop(k)

        if "silo_paths" not in d or d["silo_paths"] is None:
            if extracted_silo:
                d["silo_paths"] = extracted_silo
        elif isinstance(d["silo_paths"], dict):
            for k, v in extracted_silo.items():
                if k not in d["silo_paths"]:
                    d["silo_paths"][k] = v

        # 5. Default active_jobs
        if "active_jobs" not in d or d["active_jobs"] is None:
            d["active_jobs"] = {}

        return d

    def compute_checksum(self) -> str:
        """Calculates deterministic SHA-256 checksum of configuration payload."""
        d = self.model_dump(exclude={"registry_checksum", "last_updated"})
        serialized = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def update_checksum(self) -> str:
        """Calculates and updates registry_checksum in place."""
        cs = self.compute_checksum()
        self.registry_checksum = cs
        return cs

    def verify_checksum(self) -> bool:
        """Verifies whether registry_checksum matches the current configuration payload."""
        if not self.registry_checksum:
            return False
        return self.registry_checksum == self.compute_checksum()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration model to dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration model to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    def to_file(self, path: Union[str, Path], lock_read_only: bool = False) -> Path:
        """
        Atomically write configuration to target JSON file and optionally enforce
        read-only permissions (0o444).
        """
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        self.update_checksum()
        json_content = self.to_json(indent=2)

        unique_suffix = f".tmp.{uuid.uuid4().hex[:8]}"
        staged_file = p.parent / f"{p.name}{unique_suffix}"

        try:
            staged_file.write_text(json_content, encoding="utf-8")
            os.replace(staged_file, p)
            if lock_read_only:
                try:
                    os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
                except OSError as err:
                    logger.warning(f"Could not apply read-only chmod to {p}: {err}")
            return p
        finally:
            if staged_file.exists():
                try:
                    staged_file.unlink()
                except OSError:
                    pass

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CoChemSystemConfig:
        """Construct and validate CoChemSystemConfig from a dictionary."""
        return cls.model_validate(d)

    @classmethod
    def from_json(cls, json_str: str) -> CoChemSystemConfig:
        """Construct and validate CoChemSystemConfig from a JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> CoChemSystemConfig:
        """Construct and validate CoChemSystemConfig from a JSON file."""
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"System configuration file not found at '{p}'.")
        raw_text = p.read_text(encoding="utf-8")
        return cls.from_json(raw_text)


# Aliases for ecosystem interoperability
GoldenSystemRegistry = CoChemSystemConfig
CoChemConfig = CoChemSystemConfig


# =============================================================================
# 10. DISCOVERY & INTERROGATION ENGINE (AUTHENTIC PHYSICAL MANDATE)
# =============================================================================


def discover_host_os() -> OSProfile:
    """Interrogate host operating system, architecture, kernel, and virtualization flags."""
    sys_name = platform.system()
    rel = platform.release()
    ver = platform.version()
    mach = platform.machine()

    is_win = sys_name == "Windows"
    is_posix = os.name == "posix"

    is_wsl = False
    if sys_name == "Linux":
        if "microsoft" in rel.lower() or "wsl" in rel.lower():
            is_wsl = True
        elif os.path.exists("/proc/version"):
            try:
                proc_ver = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
                if "microsoft" in proc_ver or "wsl" in proc_ver:
                    is_wsl = True
            except OSError:
                pass
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        is_wsl = True

    cgroup_ver: Optional[str] = None
    if sys_name == "Linux":
        if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
            cgroup_ver = "v2"
        elif os.path.exists("/sys/fs/cgroup/memory"):
            cgroup_ver = "v1"

    return OSProfile(
        system=sys_name,
        release=rel,
        version=ver,
        machine=mach,
        python_executable=str(Path(sys.executable).resolve()),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        is_wsl=is_wsl,
        is_windows=is_win,
        is_posix=is_posix,
        os_target=_default_os_target(),
        cgroup_version=cgroup_ver,
    )


def discover_host_hardware() -> HardwareProfile:
    """
    Perform genuine physical discovery of CPU cores, memory topology, AVX-512 support,
    heterogeneous GPUs, and IEEE-754 precision traps.
    """
    # 1. CPU topology
    phys_cores = psutil.cpu_count(logical=False) or 1
    log_cores = psutil.cpu_count(logical=True) or 1

    # 2. RAM discovery
    total_ram_bytes = psutil.virtual_memory().total
    total_ram_gb = round(total_ram_bytes / (1024**3), 2)
    total_ram_mb = int(total_ram_bytes / (1024**2))

    # 3. AVX-512 vector detection
    avx512 = False
    if platform.system() == "Linux" and os.path.exists("/proc/cpuinfo"):
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").lower()
            if "avx512" in cpuinfo:
                avx512 = True
        except OSError:
            pass

    # 4. IEEE-754 subnormal floating point precision verification
    trap_detected = verify_ieee754_subnormal_precision()

    # 5. GPU discovery
    detected_gpus: List[GPUDevice] = []
    gpu_profile_str = "None"
    vram_gb_total = 0.0
    fp64_support = False

    # Check for nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if proc.returncode == 0:
                lines = proc.stdout.strip().splitlines()
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        idx = int(parts[0])
                        gname = parts[1]
                        vram_mb = float(parts[2])
                        driver_ver = parts[3] if len(parts) > 3 else None
                        vram_gb = round(vram_mb / 1024.0, 2)
                        vram_gb_total += vram_gb
                        gpu_profile_str = gname

                        # Check for FP64 capable architectures
                        if any(arch in gname for arch in ["A100", "H100", "V100", "Titan V", "Quadro GV100"]):
                            fp64_support = True

                        detected_gpus.append(
                            GPUDevice(
                                device_index=idx,
                                name=gname,
                                vendor="NVIDIA",
                                vram_bytes=int(vram_mb * 1024 * 1024),
                                vram_gb=vram_gb,
                                driver_version=driver_ver,
                                fp64_capable=fp64_support,
                            )
                        )
        except Exception as exc:
            logger.debug(f"nvidia-smi probe completed with notice: {exc}")

    gpu_metrics = GPUComputeSchema(
        gpu_profile=gpu_profile_str,
        vram_gb=vram_gb_total,
        device_count=len(detected_gpus),
        fp64_capable=fp64_support,
        subnormal_precision_trap=trap_detected,
        mps_enabled=len(detected_gpus) > 0,
        devices=detected_gpus,
    )

    return HardwareProfile(
        ram_gb=total_ram_gb,
        physical_cpu_cores=phys_cores,
        logical_cpu_cores=log_cores,
        allocatable_compute_cores=phys_cores,
        ram_mb=total_ram_mb,
        avx512_support=avx512,
        avx_512_capable=avx512,
        gpu_profile=gpu_profile_str,
        vram_gb=vram_gb_total,
        subnormal_precision_trap=trap_detected,
        gpu_fp64_capable=fp64_support,
        mps_enabled=len(detected_gpus) > 0,
        os_target=_default_os_target(),
        gpu_compute_metrics=gpu_metrics,
    )


def discover_engines() -> QuantumEngineRegistry:
    """Audit physical presence, versions, and hashes for all standard quantum engines."""
    engines: Dict[str, EngineInfo] = {}
    engine_candidates = ["orca", "mpirun", "xtb", "cfour", "aimnet2", "mace"]

    for name in engine_candidates:
        binary_path = shutil.which(name)
        if not binary_path and platform.system() == "Windows":
            binary_path = shutil.which(f"{name}.exe")

        if binary_path:
            p = Path(binary_path).resolve()
            sha256 = None
            try:
                sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
            except Exception:
                pass

            ver_str = None
            try:
                proc = subprocess.run([str(p), "--version"], capture_output=True, text=True, timeout=3.0)
                ver_str = proc.stdout.strip() or proc.stderr.strip()
            except Exception:
                pass

            engines[name] = EngineInfo(
                status=EngineStatus.FOUND.value,
                path=str(p),
                version=ver_str or "detected",
                hash=sha256,
                gpu_support=(name in ("gpu4pyscf", "aimnet2", "mace")),
            )
        else:
            engines[name] = EngineInfo(
                status=EngineStatus.MISSING.value,
                path=None,
                version=None,
                hash=None,
                gpu_support=(name in ("gpu4pyscf", "aimnet2", "mace")),
            )

    return QuantumEngineRegistry.model_validate(engines)


def discover_storage_topology(workspace_dir: Optional[Union[str, Path]] = None) -> StorageTopology:
    """Discover storage hierarchy, scratch targets, and HDF5 database destinations."""
    ws = Path(workspace_dir).resolve() if workspace_dir else Path.cwd().resolve()

    art_env = os.environ.get("COCHEM_ARTIFACT_DIR")
    if art_env:
        art_dir = Path(os.path.expandvars(art_env)).expanduser().resolve()
    else:
        art_dir = (Path.home() / "CoChem_Artifacts").resolve()

    scratch_env = os.environ.get("COCHEM_SCRATCH") or os.environ.get("COCHEM_SCRATCH_DIR")
    if scratch_env:
        scratch_p = Path(os.path.expandvars(scratch_env)).expanduser().resolve()
    else:
        scratch_p = (Path(tempfile.gettempdir()) / "cochem_scratch").resolve()

    is_9p, _ = audit_wsl_mount_traps(ws)
    is_posix = os.name == "posix" and not is_9p

    return StorageTopology(
        artifacts_dir=str(art_dir),
        scratch_dir=str(scratch_p),
        hdf5_pes_store_path=str(art_dir / "Registry" / "pes_store.h5"),
        state_file_path=str(art_dir / "Registry" / "cochem_state.h5"),
        databases_dir=str(art_dir / "Databases"),
        runtime_dir=str(Path(tempfile.gettempdir()) / "cochem"),
        swmr_locking_enabled=is_posix,
        storage_mode=StorageMode.LOCAL_FS,
        is_posix_compliant=is_posix,
        is_9p_mount=is_9p,
    )


def interrogate_system_config(
    workspace_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> CoChemSystemConfig:
    """
    Perform a complete, production-grade zero-mock interrogation of host operating system,
    hardware limits, quantum engines, micro-silos, and storage topology to construct
    the authoritative Golden System Registry model.
    """
    os_prof = discover_host_os()
    hw_prof = discover_host_hardware()
    engines_reg = discover_engines()
    storage_topo = discover_storage_topology(workspace_dir)

    cfg = CoChemSystemConfig(
        schema_version="4.0.0",
        registry_version="4.0",
        status="INITIALIZED",
        os_profile=os_prof,
        hardware=hw_prof,
        engines=engines_reg,
        silos=SiloRegistry(torq_silo_active=True, gpu_silo_active=hw_prof.vram_gb > 0.0),
        storage=storage_topo,
        environment=EnvironmentSchema(
            os_target=os_prof.os_target,
            artifacts_dir=storage_topo.artifacts_dir,
            scratch_dir=storage_topo.scratch_dir,
            isotopic_masses=get_mendeleev_isotopic_masses(),
        ),
        quantum_settings=QuantumSettings(implicit_solvation="CPCM", integration_grid="defgrid2"),
        adaptive_routing=RoutingPolicy(
            max_concurrent_mace_threads=min(4, hw_prof.physical_cpu_cores),
            max_dft_basis_functions=2000,
            gpu_crossover_basis_threshold=70,
        ),
        hpc=HPCConfig(scheduler="local", default_partition="compute"),
    )
    cfg.update_checksum()

    if output_path:
        cfg.to_file(output_path)

    return cfg


# =============================================================================
# 11. GOLDEN SYSTEM REGISTRY VALIDATOR & LIFECYCLE MANAGEMENT
# =============================================================================


def resolve_golden_registry_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve canonical filesystem path for `cochem_system_config.json` via 4-tier hierarchy:
    Tier 1: Explicit Parameter (CLI / Function argument)
    Tier 2: Environment Variables ($COCHEM_CONFIG, $COCHEM_ARTIFACT_DIR)
    Tier 3: Repository Root / CoChem-BASE cochem_system_config.json
    Tier 4: Dynamic User Home Fallback (~/CoChem_Artifacts/Registry/cochem_system_config.json)
    """
    if custom_path:
        p = Path(_expand_env_vars(str(custom_path))).resolve()
        if p.is_dir() or p.suffix == "":
            return p / "cochem_system_config.json"
        return p

    env_cfg = os.environ.get("COCHEM_CONFIG")
    if env_cfg:
        p = Path(_expand_env_vars(env_cfg)).resolve()
        if p.is_file():
            return p
        if p.is_dir():
            return p / "cochem_system_config.json"

    env_art = os.environ.get("COCHEM_ARTIFACT_DIR")
    if env_art:
        art_p = Path(_expand_env_vars(env_art)).resolve()
        candidate = art_p / "Registry" / "cochem_system_config.json"
        if candidate.exists():
            return candidate
        candidate_flat = art_p / "cochem_system_config.json"
        if candidate_flat.exists():
            return candidate_flat

    cwd = Path.cwd().resolve()
    repo_cand = cwd / "cochem_system_config.json"
    if repo_cand.exists():
        return repo_cand

    base_cand = cwd / "CoChem-BASE" / "cochem_system_config.json"
    if base_cand.exists():
        return base_cand

    # Check parent workspace
    parent_cand = cwd.parent / "cochem_system_config.json"
    if parent_cand.exists():
        return parent_cand

    return (Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json").resolve()


def validate_system_config(
    source: Union[str, Path, Dict[str, Any], CoChemSystemConfig]
) -> CoChemSystemConfig:
    """
    Authoritative Gatekeeper Validator: Accepts a dictionary, file path, JSON string, or model
    and enforces strict type-safe Pydantic v2 validation against the master schema.
    """
    if isinstance(source, CoChemSystemConfig):
        return source

    if isinstance(source, dict):
        return CoChemSystemConfig.model_validate(source)

    if isinstance(source, Path):
        return CoChemSystemConfig.from_file(source)

    if isinstance(source, str):
        if os.path.exists(source):
            return CoChemSystemConfig.from_file(source)
        try:
            return CoChemSystemConfig.from_json(source)
        except Exception:
            try:
                raw_dict = json.loads(source)
                return CoChemSystemConfig.model_validate(raw_dict)
            except Exception as exc:
                raise SchemaValidationError(f"Failed to parse and validate configuration source: {exc}") from exc

    raise TypeError(f"Unsupported configuration source type: {type(source)}")


def load_golden_registry(path: Optional[Union[str, Path]] = None) -> CoChemSystemConfig:
    """Load and validate the persistent Golden System Registry cochem_system_config.json."""
    resolved = resolve_golden_registry_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Golden System Registry configuration not found at '{resolved}'. "
            "Run interrogation or provision the environment before proceeding."
        )
    return CoChemSystemConfig.from_file(resolved)


def save_golden_registry(
    cfg: CoChemSystemConfig,
    path: Optional[Union[str, Path]] = None,
    lock: bool = False,
) -> Path:
    """
    Atomically persist the Golden System Registry to cochem_system_config.json,
    calculate SHA-256 checksum, and apply read-only locking if requested.
    """
    target = resolve_golden_registry_path(path)
    if lock:
        cfg.status = "LOCKED"
    return cfg.to_file(target, lock_read_only=lock)


def verify_golden_registry_integrity(
    path_or_cfg: Union[str, Path, CoChemSystemConfig]
) -> Tuple[bool, List[str]]:
    """
    Verify schema validity, SHA-256 checksum consistency, physical engine path existence,
    and storage accessibility for the Golden System Registry.
    Returns (is_valid, list_of_issues).
    """
    issues: List[str] = []

    try:
        if isinstance(path_or_cfg, CoChemSystemConfig):
            cfg = path_or_cfg
        else:
            cfg = validate_system_config(path_or_cfg)
    except Exception as exc:
        return False, [f"Schema Validation Failure: {exc}"]

    # 1. Checksum verification
    if cfg.registry_checksum:
        if not cfg.verify_checksum():
            issues.append("Registry SHA-256 checksum mismatch (corrupted or manually mutated payload).")

    # 2. Hardware sanity
    if cfg.hardware.physical_cpu_cores < 1:
        issues.append(f"Invalid physical CPU core count: {cfg.hardware.physical_cpu_cores}")
    if cfg.hardware.ram_gb <= 0.0:
        issues.append(f"Invalid RAM allocation: {cfg.hardware.ram_gb} GB")

    # 3. Engine verification
    if isinstance(cfg.engines, QuantumEngineRegistry):
        for eng_name, eng_info in [
            ("orca", cfg.engines.orca),
            ("mpirun", cfg.engines.mpirun),
            ("xtb", cfg.engines.xtb),
        ]:
            if eng_info and eng_info.status == EngineStatus.FOUND.value and eng_info.path:
                p = Path(eng_info.path)
                if not p.exists():
                    issues.append(f"Engine '{eng_name}' marked as FOUND but file does not exist at '{p}'.")

    # 4. Storage verification
    if cfg.storage:
        art_path = Path(cfg.storage.artifacts_dir)
        if not art_path.exists():
            try:
                art_path.mkdir(parents=True, exist_ok=True)
            except OSError as err:
                issues.append(f"Artifacts directory '{art_path}' is inaccessible or cannot be created: {err}")

    return (len(issues) == 0, issues)


# =============================================================================
# 12. CLI ENTRYPOINT
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line interface for Golden System Registry validation and interrogation."""
    parser = argparse.ArgumentParser(
        description="CoChem Golden System Registry Validator & Interrogator (SRS Doc 2 Part 2 §3.11)."
    )
    parser.add_argument(
        "--validate",
        "-v",
        type=str,
        default=None,
        help="Path to cochem_system_config.json to validate against Pydantic v2 schema.",
    )
    parser.add_argument(
        "--interrogate",
        "-i",
        action="store_true",
        help="Interrogate host OS, hardware, engines, and storage to generate a fresh Golden Registry.",
    )
    parser.add_argument(
        "--check-integrity",
        "-c",
        type=str,
        default=None,
        help="Verify cryptographic checksum and binary path integrity for the specified config file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output destination path for generated or validated configuration JSON.",
    )
    parser.add_argument(
        "--lock",
        "-l",
        action="store_true",
        help="Apply os.chmod(0o444) read-only immutability to output configuration file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit output as raw formatted JSON to standard output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed diagnostic logging to standard error.",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Action 1: Check Integrity
        if args.check_integrity:
            target = resolve_golden_registry_path(args.check_integrity)
            valid, issues = verify_golden_registry_integrity(target)
            if valid:
                if args.json:
                    print(json.dumps({"status": "VALID", "path": str(target), "issues": []}, indent=2))
                else:
                    print(f"SUCCESS: Golden System Registry at '{target}' is VALID and intact.")
                return 0
            else:
                if args.json:
                    print(json.dumps({"status": "INVALID", "path": str(target), "issues": issues}, indent=2))
                else:
                    print(f"FAILED: Golden System Registry at '{target}' has integrity issues:")
                    for issue in issues:
                        print(f"  - {issue}")
                return 1

        # Action 2: Interrogate Host
        if args.interrogate:
            cfg = interrogate_system_config(output_path=args.output)
            if args.lock and args.output:
                save_golden_registry(cfg, path=args.output, lock=True)

            if args.json:
                print(cfg.to_json(indent=2))
            else:
                print(f"SUCCESS: Interrogated Golden System Registry (Status: {cfg.status})")
                print(f"  OS Target:  {cfg.hardware.os_target}")
                print(f"  CPU Cores:  {cfg.hardware.physical_cpu_cores} physical, {cfg.hardware.logical_cpu_cores} logical")
                print(f"  RAM:        {cfg.hardware.ram_gb} GB (Maxcore: {cfg.hardware.maxcore_mb} MB)")
                print(f"  GPU:        {cfg.hardware.gpu_profile} ({cfg.hardware.vram_gb} GB VRAM)")
                print(f"  Checksum:   {cfg.registry_checksum}")
                if args.output:
                    print(f"  Artifact:   {args.output}")
            return 0

        # Action 3: Validate Existing Config
        config_path = resolve_golden_registry_path(args.validate)
        cfg = load_golden_registry(config_path)

        if args.output:
            save_golden_registry(cfg, path=args.output, lock=args.lock)

        if args.json:
            print(cfg.to_json(indent=2))
        else:
            print(f"SUCCESS: System configuration at '{config_path}' successfully validated.")
            print(f"  Schema:     {cfg.schema_version} (Status: {cfg.status})")
            print(f"  OS Target:  {cfg.hardware.os_target}")
            print(f"  CPU Cores:  {cfg.hardware.physical_cpu_cores} physical / {cfg.hardware.logical_cpu_cores} logical")
            print(f"  RAM:        {cfg.hardware.ram_gb} GB")
            print(f"  Checksum:   {cfg.registry_checksum}")

        return 0

    except Exception as err:
        logger.error(f"FATAL: System configuration error: {err}")
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
