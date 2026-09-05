"""Pydantic v2 data models and contracts for TORQ Molecular Dynamics (Chunk 21).

Method Matrix v4 Provenance Tags:
- [M] Mandated: Strict frozen schemas, extra="forbid", double-precision validation.
- [D] Derived: Rigorous state transitions, replica exchange bounds, geometric temperature validation.
- [E] Empirical: Sensible defaults calibrated from empirical water potential benchmarks.

Strict Zero-Mock Mandate v3: Completely authentic physics and mathematical rigor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch


class VelocityVerletConfig(BaseModel):
    """Configuration contract for Symplectic Velocity Verlet Integrator [M]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestep_fs: float = Field(default=0.5, gt=0.0, le=1.0)
    n_steps: int = Field(gt=0, default=20000)
    save_interval: int = Field(gt=0, default=10)
    device: str = Field(default="cpu")
    dtype: Literal["float64"] = Field(default="float64")
    energy_drift_tolerance: float = Field(default=1e-4, gt=0.0)
    remove_com_momentum: bool = Field(default=True)


class REMDConfig(BaseModel):
    """Configuration contract for Replica Exchange Molecular Dynamics Engine [M]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_replicas: int = Field(ge=2, default=4)
    t_min_k: float = Field(gt=0.0, default=300.0)
    t_max_k: float = Field(gt=0.0, default=600.0)
    swap_interval_steps: int = Field(ge=10, default=100)
    total_steps_per_replica: int = Field(gt=0, default=100000)
    friction_ps: float = Field(default=1.0, gt=0.0)
    output_dir: Path

    @model_validator(mode="after")
    def validate_remd_parameters(self) -> "REMDConfig":
        """Validate physical thermodynamic bounds and schedule consistency [D]."""
        if self.t_max_k <= self.t_min_k:
            raise ValueError(
                f"t_max_k ({self.t_max_k}) must be strictly greater than t_min_k ({self.t_min_k})"
            )
        if self.total_steps_per_replica < self.swap_interval_steps:
            raise ValueError(
                f"total_steps_per_replica ({self.total_steps_per_replica}) must be >= "
                f"swap_interval_steps ({self.swap_interval_steps})"
            )
        return self


class MDState(BaseModel):
    """Thermodynamic and kinematic state snapshot during MD simulation [M]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    time_fs: float = Field(ge=0.0)
    potential_energy_ev: float
    kinetic_energy_ev: float = Field(ge=0.0)
    total_energy_ev: float
    temperature_k: float = Field(ge=0.0)


class TrajectoryFrame(NamedTuple):
    """In-memory trajectory frame containing tensors for coordinates, velocities, and forces [M]."""

    step: int
    time_fs: float
    atomic_numbers: torch.Tensor  # Shape: [N_atoms], dtype: torch.int64
    coordinates: torch.Tensor  # Shape: [N_atoms, 3], dtype: torch.float64
    velocities: torch.Tensor  # Shape: [N_atoms, 3], dtype: torch.float64
    forces: torch.Tensor  # Shape: [N_atoms, 3], dtype: torch.float64
    potential_energy_ev: float
    kinetic_energy_ev: float


class ExchangeLog(BaseModel):
    """Audit log entry for replica exchange swap trials [M]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_step: int
    replica_i: int
    replica_j: int
    temp_i_k: float
    temp_j_k: float
    energy_i_ev: float
    energy_j_ev: float
    p_swap: float = Field(ge=0.0, le=1.0)
    accepted: bool
