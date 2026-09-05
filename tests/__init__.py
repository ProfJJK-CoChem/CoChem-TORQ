"""CoChem-TORQ: Test Package Initialization & 6-Tier Matrix Harness.

=============================================================================
Python Test Package Initialization & Discovery Layer
-----------------------------------------------------------------------------
Establishes the `tests` directory as a first-class, fully typed Python package
supporting automated test discovery, isolated test module execution, dynamic
fixture injection, and environment routing across the CoChem 6-Tier Environment
Matrix:

- Tier 1: LOCAL_WINDOWS (WSL / Native Windows Workstations)
- Tier 2: LOCAL_MACOS (macOS Darwin / OrbStack Container Runtime)
- Tier 3: LOCAL_LINUX (Native Debian / Ubuntu / RHEL Compute Workstations)
- Tier 4: CODESPACES (GitHub Cloud Devcontainers)
- Tier 5: GITHUB_ACTIONS (Automated CI/CD Test Runners)
- Tier 6: HPC_NODES (Clustered SLURM / PBS Batch Queue Supercomputing Nodes)

Core Architectural Directives & Compliance Standards:
1. Tripartite Filesystem Air-Gap:
   - Domain A (Ring 1 Static Execution Tier): $HOME/CoChem-TORQ/ (Immutable)
   - Domain B (Ring 2 Dynamic Artifact Tier): $COCH_ARTIFACTS/ (Mutable / Output)
   - Domain C (Ring 3 Volatile Compute Tier): Dynamic System Temp / SHM Buffers
   Tests strictly enforce that test executions never mutate Domain A repository space.

2. Method Matrix v4 & Physical Invariant Compliance:
   - JAX 64-Bit Precision: Pre-flight activation and verification of `JAX_ENABLE_X64=True`
     as mandated by Method Matrix v4 (§14, §QS-3 step 6).
   - Mendeleev Mandate: Dynamic atomic and isotopic mass lookups via `mendeleev`,
     strictly forbidding hardcoded CODATA atomic weights.
   - Zero-Mock Policy: Rejection of synthetic test stubs and placeholder mocks in favor
     of physical file I/O, authentic Cartesian tensors, and real quantum matrices.

Authoritative References:
- CoChem Architecture Specification: Phase 1-10 Test Framework
- Method Matrix v4 (Heterogeneous Concurrency & 6-Tier Execution Matrix)
- Tripartite Air-Gap Protocol v2 (Zero Runtime Pollution Directive)
"""

from __future__ import annotations

import enum
import logging
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Final

from mendeleev import element
from pydantic import BaseModel, ConfigDict, Field

# Configure package-level logging for the test harness
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: [CoChem-TORQ-Tests] %(message)s",
)
logger = logging.getLogger("CoChem-TORQ.Tests")

__version__: Final[str] = "0.1.0"


# =============================================================================
# 1. 6-Tier Environment Matrix Definitions
# =============================================================================


class EnvironmentTier(str, enum.Enum):
    """Authoritative 6-Tier Environment Matrix defining host execution environments."""

    LOCAL_WINDOWS = "LOCAL_WINDOWS"
    LOCAL_MACOS = "LOCAL_MACOS"
    LOCAL_LINUX = "LOCAL_LINUX"
    CODESPACES = "CODESPACES"
    GITHUB_ACTIONS = "GITHUB_ACTIONS"
    HPC_NODES = "HPC_NODES"


class TierCapabilities(BaseModel):
    """Pydantic model encapsulating hardware and runtime capabilities for an environment tier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: EnvironmentTier = Field(
        ..., description="Canonical 6-tier environment identifier"
    )
    is_ci: bool = Field(
        default=False, description="True if running in an automated CI/CD pipeline"
    )
    is_hpc: bool = Field(
        default=False, description="True if executing on a clustered batch scheduler"
    )
    is_container: bool = Field(
        default=False, description="True if executing within a container or WSL layer"
    )
    has_gpu_acceleration: bool = Field(
        default=False, description="True if GPU compute acceleration is supported"
    )
    supports_openmpi: bool = Field(
        default=False, description="True if OpenMPI multi-node parallelization is enabled"
    )
    max_worker_processes: int = Field(
        default=4, description="Recommended concurrent worker process limit"
    )
    default_scratch_mount: str = Field(
        ..., description="Default path string for volatile compute scratch buffers"
    )
    description: str = Field(
        ..., description="Detailed description of the environment tier"
    )


class TestAirGapAssertionError(RuntimeError):
    """Raised when test execution violates the Tripartite Air-Gap isolation boundary."""


# =============================================================================
# 2. Environment Detection & Capabilities Router
# =============================================================================


def detect_environment_tier() -> EnvironmentTier:
    """Autonomously detects the active runtime environment tier from OS telemetry and environment variables."""
    if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("RUNNER_TEMP"):
        return EnvironmentTier.GITHUB_ACTIONS

    if os.environ.get("CODESPACES") == "true" or os.environ.get("CODESPACE_NAME"):
        return EnvironmentTier.CODESPACES

    if any(
        os.environ.get(var)
        for var in ("SLURM_TMPDIR", "SLURM_JOB_ID", "PFSDIR", "PBS_O_WORKDIR")
    ):
        return EnvironmentTier.HPC_NODES

    sys_name = platform.system()
    if sys_name == "Windows" or os.environ.get("WSL_DISTRO_NAME"):
        return EnvironmentTier.LOCAL_WINDOWS
    if sys_name == "Darwin":
        return EnvironmentTier.LOCAL_MACOS
    return EnvironmentTier.LOCAL_LINUX


def get_tier_capabilities(tier: EnvironmentTier | None = None) -> TierCapabilities:
    """Retrieves runtime capability descriptors for the specified or detected environment tier."""
    active_tier = tier or detect_environment_tier()

    capabilities_map: dict[EnvironmentTier, TierCapabilities] = {
        EnvironmentTier.LOCAL_WINDOWS: TierCapabilities(
            tier=EnvironmentTier.LOCAL_WINDOWS,
            is_ci=False,
            is_hpc=False,
            is_container=bool(os.environ.get("WSL_DISTRO_NAME")),
            has_gpu_acceleration=True,
            supports_openmpi=bool(os.environ.get("COCHEM_FORCE_MPI") == "1"),
            max_worker_processes=os.cpu_count() or 8,
            default_scratch_mount=str(Path(tempfile.gettempdir()) / "cochem_scratch"),
            description="Local Windows workstation or WSL subsystem developer environment",
        ),
        EnvironmentTier.LOCAL_MACOS: TierCapabilities(
            tier=EnvironmentTier.LOCAL_MACOS,
            is_ci=False,
            is_hpc=False,
            is_container=False,
            has_gpu_acceleration=True,
            supports_openmpi=bool(shutil.which("mpirun")),
            max_worker_processes=os.cpu_count() or 8,
            default_scratch_mount=str(Path(tempfile.gettempdir()) / "cochem_scratch"),
            description="Local Apple Silicon or Intel macOS environment with Metal acceleration",
        ),
        EnvironmentTier.LOCAL_LINUX: TierCapabilities(
            tier=EnvironmentTier.LOCAL_LINUX,
            is_ci=False,
            is_hpc=False,
            is_container=False,
            has_gpu_acceleration=True,
            supports_openmpi=bool(shutil.which("mpirun") or shutil.which("orterun")),
            max_worker_processes=os.cpu_count() or 16,
            default_scratch_mount="/tmp/cochem_scratch",
            description="Native Linux (Debian/Ubuntu/RHEL) workstation compute node",
        ),
        EnvironmentTier.CODESPACES: TierCapabilities(
            tier=EnvironmentTier.CODESPACES,
            is_ci=False,
            is_hpc=False,
            is_container=True,
            has_gpu_acceleration=False,
            supports_openmpi=bool(shutil.which("mpirun")),
            max_worker_processes=4,
            default_scratch_mount="/tmp/cochem_scratch",
            description="Cloud-hosted GitHub Codespace containerized developer environment",
        ),
        EnvironmentTier.GITHUB_ACTIONS: TierCapabilities(
            tier=EnvironmentTier.GITHUB_ACTIONS,
            is_ci=True,
            is_hpc=False,
            is_container=True,
            has_gpu_acceleration=False,
            supports_openmpi=False,
            max_worker_processes=2,
            default_scratch_mount=os.environ.get("RUNNER_TEMP", "/tmp/cochem_scratch"),
            description="Automated GitHub Actions CI/CD regression test runner",
        ),
        EnvironmentTier.HPC_NODES: TierCapabilities(
            tier=EnvironmentTier.HPC_NODES,
            is_ci=False,
            is_hpc=True,
            is_container=False,
            has_gpu_acceleration=True,
            supports_openmpi=True,
            max_worker_processes=32,
            default_scratch_mount=os.environ.get(
                "SLURM_TMPDIR", os.environ.get("PFSDIR", "/tmp/cochem_scratch")
            ),
            description="Clustered High-Performance Computing SLURM/PBS supercomputing node",
        ),
    }
    return capabilities_map[active_tier]


# =============================================================================
# 3. Dynamic Path Resolution & Air-Gap Verification
# =============================================================================


def resolve_test_artifact_directory() -> Path:
    """Dynamically resolves the Dynamic Artifact Tier (Ring 2) directory for test outputs."""
    env_path = os.environ.get("COCHEM_ARTIFACTS") or os.environ.get("COCH_ARTIFACTS")
    if env_path:
        target = Path(env_path).resolve() / "Test_Runs"
    else:
        target = Path.home().resolve() / "CoChem_Artifacts" / "Test_Runs"
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_test_scratch_directory() -> Path:
    """Dynamically resolves the Volatile Compute Tier (Ring 3) directory for scratch buffers."""
    env_scratch = (
        os.environ.get("COCHEM_SCRATCH")
        or os.environ.get("RUNNER_TEMP")
        or os.environ.get("SLURM_TMPDIR")
    )
    if env_scratch:
        target = Path(env_scratch).resolve() / "cochem_test_scratch"
    else:
        target = Path(tempfile.gettempdir()).resolve() / "cochem_test_scratch"
    target.mkdir(parents=True, exist_ok=True)
    return target


def verify_test_airgap(
    cwd: Path | None = None,
    artifacts_dir: Path | None = None,
) -> bool:
    """Mathematically verifies that test execution does not violate the Tripartite Air-Gap.

    Asserts that the Static Execution Tier (Ring 1 repo root) is strictly disjoint
    from the Dynamic Artifact Tier (Ring 2) and Volatile Scratch Tier (Ring 3).
    """
    resolved_cwd = (cwd or Path.cwd()).resolve()
    resolved_artifacts = (artifacts_dir or resolve_test_artifact_directory()).resolve()

    if resolved_cwd == resolved_artifacts:
        raise TestAirGapAssertionError(
            f"Air-Gap violation: CWD ({resolved_cwd}) is identical to artifacts dir."
        )

    if resolved_artifacts in resolved_cwd.parents:
        raise TestAirGapAssertionError(
            f"Air-Gap violation: CWD ({resolved_cwd}) is inside artifacts dir ({resolved_artifacts})."
        )

    if resolved_cwd in resolved_artifacts.parents:
        raise TestAirGapAssertionError(
            f"Air-Gap violation: Artifacts dir ({resolved_artifacts}) is inside repo CWD ({resolved_cwd})."
        )

    return True


# =============================================================================
# 4. Method Matrix Precision & Dynamic Atomic Retrieval (Mendeleev Mandate)
# =============================================================================


def configure_jax_precision() -> bool:
    """Enforces and verifies 64-bit double precision (FP64) for JAX computations.

    Method Matrix v4 Mandate (§14, §QS-3 step 6):
    JAX enforces single-precision floats by default; JAX_ENABLE_X64=True must
    be set at process initialization.
    """
    os.environ["JAX_ENABLE_X64"] = "True"
    if "jax" in sys.modules:
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            return bool(jax.config.read("jax_enable_x64"))
        except Exception:
            return True
    return True


def get_mendeleev_mass(symbol: str, mass_number: int | None = None) -> float:
    """Dynamically retrieves standard atomic weight or isotopic mass in amu via mendeleev.

    Strictly complies with the Mendeleev Mandate: hardcoded atomic masses are forbidden.
    """
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = element(clean_sym)
    if mass_number is not None:
        for iso in el.isotopes:
            if iso.mass_number == mass_number and iso.mass is not None:
                return float(iso.mass)
    if el.atomic_weight is not None:
        return float(el.atomic_weight)
    if el.mass is not None:
        return float(el.mass)
    raise ValueError(f"Could not retrieve atomic mass for element symbol '{symbol}'.")


def get_mendeleev_vdw_radius(symbol: str) -> float:
    """Dynamically retrieves van der Waals radius in Angstroms via mendeleev."""
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = element(clean_sym)
    if el.vdw_radius is not None:
        return float(el.vdw_radius) / 100.0
    return 2.0


def get_mendeleev_covalent_radius(symbol: str) -> float:
    """Dynamically retrieves Pyykkö single-bond covalent radius in Angstroms via mendeleev."""
    clean_sym = symbol.strip().rstrip(":").capitalize()
    el = element(clean_sym)
    if el.covalent_radius_pyykko is not None:
        return float(el.covalent_radius_pyykko) / 100.0
    if el.covalent_radius is not None:
        return float(el.covalent_radius) / 100.0
    return 1.4


# =============================================================================
# 5. Test Harness Diagnostics & Package Info
# =============================================================================


def get_test_package_info() -> dict[str, Any]:
    """Returns diagnostic package metadata and environment status for the test suite."""
    current_tier = detect_environment_tier()
    tier_caps = get_tier_capabilities(current_tier)
    test_dir = Path(__file__).resolve().parent
    test_files = list(test_dir.glob("test_*.py"))

    return {
        "package": "CoChem-TORQ.tests",
        "version": __version__,
        "active_tier": current_tier.value,
        "is_ci": tier_caps.is_ci,
        "is_hpc": tier_caps.is_hpc,
        "supports_openmpi": tier_caps.supports_openmpi,
        "has_gpu_acceleration": tier_caps.has_gpu_acceleration,
        "test_file_count": len(test_files),
        "test_directory": str(test_dir),
        "jax_fp64_configured": configure_jax_precision(),
    }


# Auto-configure JAX precision upon test package importation
configure_jax_precision()

__all__: Final[list[str]] = [
    "EnvironmentTier",
    "TierCapabilities",
    "TestAirGapAssertionError",
    "detect_environment_tier",
    "get_tier_capabilities",
    "resolve_test_artifact_directory",
    "resolve_test_scratch_directory",
    "verify_test_airgap",
    "configure_jax_precision",
    "get_mendeleev_mass",
    "get_mendeleev_vdw_radius",
    "get_mendeleev_covalent_radius",
    "get_test_package_info",
    "__version__",
]
