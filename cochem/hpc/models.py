"""CoChem HPC and scaling domain models and domain exception hierarchy.

Provides immutable Pydantic v2 data models for CUDA budgets, Slurm job directives,
dry-run execution manifests, and MPI execution configurations.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CoChemHpcScalingError(Exception):
    """Base domain exception for CoChem HPC, scaling, and runtime execution errors."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class CudaMemoryExhaustionError(CoChemHpcScalingError):
    """Raised when GPU VRAM requirements exceed available headroom after backpressure timeout."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class SlurmResourceValidationError(CoChemHpcScalingError):
    """Raised when Slurm directives violate partition bounds or invalid walltime/memory specifications."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class MpiProcessSupervisorError(CoChemHpcScalingError):
    """Raised when MPI launch fails, pipes deadlock, or child processes terminate unexpectedly."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class CudaResourceBudget(BaseModel):
    """Immutable data model representing physical CUDA device VRAM budget."""

    model_config = ConfigDict(frozen=True)

    device_id: int = Field(..., ge=0, description="CUDA device ordinal index")
    total_vram_mb: int = Field(..., gt=0, description="Total physical VRAM in megabytes")
    free_vram_mb: int = Field(..., ge=0, description="Currently unallocated physical VRAM in megabytes")
    reserved_headroom_mb: int = Field(
        default=1024,
        ge=256,
        description="Mandatory memory headroom reserved for OS/driver kernels in megabytes",
    )
    fractional_limit: float = Field(
        default=0.85,
        gt=0.0,
        le=1.0,
        description="Safety allocation multiplier enforcing a 15% protective buffer",
    )

    @property
    def available_vram_mb(self) -> int:
        """Calculate effective allocatable VRAM enforcing reserved headroom and fractional safety buffer."""
        net_free = self.free_vram_mb - self.reserved_headroom_mb
        if net_free <= 0:
            return 0
        return int(net_free * self.fractional_limit)


class SlurmJobDirectiveSpec(BaseModel):
    """Immutable data model specifying Slurm cluster submission parameters."""

    model_config = ConfigDict(frozen=True)

    job_name: str = Field(..., min_length=1, max_length=64, description="Cluster job identifier string")
    partition: str = Field(..., description="Target cluster partition queue")
    nodes: int = Field(default=1, ge=1, description="Number of allocated compute nodes")
    ntasks_per_node: int = Field(default=1, ge=1, description="Number of MPI tasks or ranks per node")
    cpus_per_task: int = Field(default=1, ge=1, description="Number of CPU cores per MPI task for OpenMP threading")
    gpus_per_node: Optional[int] = Field(default=None, ge=0, description="Number of GPUs per node if required")
    walltime_str: str = Field(
        ...,
        pattern=r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$",
        description="Slurm execution walltime string formatted as D-HH:MM:SS or HH:MM:SS",
    )
    memory_per_node_mb: int = Field(..., gt=0, description="Total memory allocated per node in megabytes")
    account: Optional[str] = Field(default=None, description="HPC charge account string")
    qos: Optional[str] = Field(default=None, description="Quality of Service designation")
    scratch_dir: str = Field(..., description="Per-job ephemeral scratch directory in POSIX path format")
    artifact_dir: str = Field(..., description="Append-only persistent artifacts destination in POSIX path format")


class SlurmDryRunResult(BaseModel):
    """Immutable data model representing preflight Slurm dry-run compilation outcome."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="True if directives pass all preflight schema and partition limits")
    generated_script_content: str = Field(..., description="Rendered SBATCH shell script content")
    estimated_memory_per_rank_mb: int = Field(
        ...,
        description="Calculated memory ceiling per MPI rank in megabytes",
    )
    orca_maxcore_mb: Optional[int] = Field(
        default=None,
        description="Calculated ORCA %maxcore parameter per MPI core in megabytes",
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Collection of validation failure descriptions",
    )
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="Collection of non-fatal execution advisory notices",
    )


class MpiClusterExecutionConfig(BaseModel):
    """Immutable data model configuring multi-node MPI cluster execution."""

    model_config = ConfigDict(frozen=True)

    launcher: str = Field(
        default="srun",
        pattern=r"^(srun|mpirun|mpiexec)$",
        description="Designated MPI binary launcher (srun, mpirun, or mpiexec)",
    )
    n_nodes: int = Field(default=1, ge=1, description="Total node count allocated for launch")
    n_tasks_per_node: int = Field(default=1, ge=1, description="Task rank count per host")
    cpus_per_task: int = Field(default=1, ge=1, description="OpenMP thread concurrency per task rank")
    environment_vars: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variable overrides exported across cluster nodes",
    )
    timeout_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        description="Maximum permissible execution duration before process tree termination",
    )
