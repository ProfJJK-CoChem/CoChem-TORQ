"""Pydantic v2 schemas and validation contracts for TORQ Training Dynamics.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Enforces frozen=True, extra="forbid", and strict validation invariants.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TrainingDynamicsConfig(BaseModel):
    """Configuration contract for MLFF training dynamics and autograd checkpointing. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(default=32, ge=1, description="Per-device batch size [M]")
    gradient_checkpointing: bool = Field(default=True, description="Enable activation recomputation [D]")
    mixed_precision_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16", description="AMP precision [M]"
    )
    max_gradient_norm: float = Field(default=1.0, gt=0.0, description="Gradient clipping norm ceiling [M]")
    energy_loss_weight: float = Field(default=1.0, ge=0.0, description="Loss weight for potential energy [M]")
    force_loss_weight: float = Field(default=100.0, ge=0.0, description="Loss weight for atomic forces [M]")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Initial optimizer learning rate [M]")
    weight_decay: float = Field(default=1e-5, ge=0.0, description="AdamW weight decay regularization [M]")


class TransferLearningConfig(BaseModel):
    """Configuration contract for ANI-2x domain adaptation and parameter transfer. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model_path: Path = Field(..., description="Local path to pre-trained ANI-2x weights [M]")
    expected_sha256: str = Field(
        ...,
        pattern=r"^[0-9a-fA-F]{64}$",
        description="Cryptographic SHA-256 hash [M]",
    )
    freeze_symmetry_functions: bool = Field(default=True, description="Freeze AEV feature extractor [M]")
    layer_decay_rate: float = Field(default=0.8, gt=0.0, le=1.0, description="LLRD decay factor [M]")
    new_species_atomic_numbers: List[int] = Field(
        default_factory=list, description="New atomic numbers Z to initialize [M]"
    )

    @field_validator("new_species_atomic_numbers")
    @classmethod
    def validate_atomic_numbers(cls, v: List[int]) -> List[int]:
        """Ensure all atomic numbers satisfy 1 <= Z <= 118 and return sorted unique values."""
        for z in v:
            if z < 1 or z > 118:
                raise ValueError(f"Atomic number Z={z} must be between 1 and 118 inclusive.")
        return sorted(list(set(v)))


class TorchScriptExportConfig(BaseModel):
    """Configuration contract for dedicated C++ TorchScript compilation and parity verification. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    export_path: Path = Field(..., description="Destination path for serialized TorchScript .pt model [M]")
    energy_relative_tolerance: float = Field(
        default=1e-6, gt=0.0, description="Max relative energy error [M]"
    )
    force_parity_tolerance_hartree_angstrom: float = Field(
        default=1e-6,
        gt=0.0,
        description="Max absolute force error in Hartree/Å evaluated in float64 [M]",
    )
    validate_against_fixtures: bool = Field(
        default=True, description="Execute validation before finalizing export [M]"
    )


class DistributedEarlyStoppingConfig(BaseModel):
    """Configuration contract for multi-GPU deadlock-free distributed early stopping. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    patience_epochs: int = Field(default=15, ge=1, description="Epochs to wait before early termination [M]")
    min_delta_hartree: float = Field(
        default=1e-5, ge=0.0, description="Minimum validation loss improvement in Hartree [M]"
    )
    synchronize_ranks: Literal[True] = Field(
        default=True, description="Mandatory broadcast early exit flag across all DDP ranks [D]"
    )


class LossLandscapeConfig(BaseModel):
    """Configuration contract for scale-invariant filter-normalized loss surface profiling. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grid_resolution: int = Field(default=25, ge=5, le=100, description="Grid points per axis [M]")
    range_min: float = Field(default=-1.0, description="Normalized coordinate min [M]")
    range_max: float = Field(default=1.0, description="Normalized coordinate max [M]")
    filter_normalization: bool = Field(
        default=True, description="Normalize random directions by filter Frobenius norm [D]"
    )
    output_plot_path: Path = Field(..., description="Destination path for rendered contour plot [M]")

    @model_validator(mode="after")
    def validate_coordinate_range(self) -> LossLandscapeConfig:
        """Verify range_min is strictly less than range_max."""
        if self.range_min >= self.range_max:
            raise ValueError(
                f"range_min ({self.range_min}) must be strictly less than range_max ({self.range_max})."
            )
        return self


class GNNWarmRestartSchedulerConfig(BaseModel):
    """Configuration contract for GNN warm restart learning rate scheduler. [D]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_lr: float = Field(default=1e-4, gt=0.0, description="Peak initial learning rate eta_max [D]")
    min_lr: float = Field(default=1e-7, ge=0.0, description="Minimum learning rate floor eta_min [D]")
    warmup_steps: int = Field(default=1000, ge=100, description="Linear warmup steps [D]")
    first_cycle_steps: int = Field(default=10000, ge=500, description="Duration of initial restart cycle T_0 [D]")
    cycle_multiplier: float = Field(default=1.5, ge=1.0, description="Cycle expansion factor T_mult [D]")
    restart_decay: float = Field(default=0.75, gt=0.0, le=1.0, description="Peak LR attenuation factor gamma_restart [D]")
    max_grad_norm: float = Field(default=1.0, gt=0.0, description="Ceiling for gradient norm clipping [D]")

    @model_validator(mode="after")
    def validate_lr_bounds(self) -> GNNWarmRestartSchedulerConfig:
        if self.min_lr >= self.initial_lr:
            raise ValueError(
                f"min_lr ({self.min_lr}) must be strictly less than initial_lr ({self.initial_lr})"
            )
        return self


class ForceMatchingLossConfig(BaseModel):
    """Configuration contract for second-order autograd force-matching loss engine. [D]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    energy_weight: float = Field(default=1.0, ge=0.0, description="Loss weight for potential energy [D]")
    force_weight: float = Field(default=50.0, ge=0.0, description="Loss weight for atomic force vectors [D]")
    virial_weight: float = Field(default=0.01, ge=0.0, description="Loss weight for periodic virial stress [D]")
    huber_delta_energy: float = Field(default=0.01, gt=0.0, description="Huber transition delta for energy in eV [D]")
    huber_delta_force: float = Field(default=0.05, gt=0.0, description="Huber transition delta in eV/Angstrom [D]")
    normalization_mode: Literal["atom_norm", "coordinate_component"] = Field(
        default="atom_norm", description="Normalization mode: atom_norm (N_atoms) or coordinate_component (3 * N_atoms) [D]"
    )
    create_graph: bool = Field(default=True, description="Retain second-order graph in autograd for forces [D]")
    track_angular_similarity: bool = Field(default=True, description="Compute cosine similarity metric [D]")


class DynamicBatchScalerConfig(BaseModel):
    """Configuration contract for dual-budget graph packer and OOM recovery. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_node_budget: int = Field(default=4096, ge=64, description="Maximum atoms per micro-batch [M]")
    max_edge_budget: int = Field(default=32768, ge=256, description="Maximum sparse edges per micro-batch [M]")
    backoff_factor: float = Field(default=0.75, gt=0.1, lt=1.0, description="Budget step-down on OOM [M]")
    max_recovery_retries: int = Field(default=3, ge=1, description="Max retry attempts per failed micro-batch [M]")
    target_vram_fraction: float = Field(default=0.85, gt=0.1, lt=0.98, description="Target VRAM ceiling [M]")


class C2GraphPrunerConfig(BaseModel):
    """Configuration contract for C^2-smooth reciprocal sparse graph pruning. [D]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cutoff_radius_angstrom: float = Field(default=5.0, gt=1.0, description="Spatial interaction cutoff r_c [D]")
    covalent_core_radius_angstrom: float = Field(default=1.5, gt=0.5, description="Core radius r_cov guaranteed degree >= 1 [D]")
    enforce_reciprocal_edges: bool = Field(default=True, description="Enforce undirected edge reciprocity [D]")
    switching_polynomial_degree: Literal[5] = Field(default=5, description="Quintic polynomial C^2 cutoff envelope [D]")
    stochastic_edge_dropout: float = Field(default=0.0, ge=0.0, le=0.5, description="Banned on coordinate graphs; invariant heads only [D]")

    @model_validator(mode="after")
    def validate_radii(self) -> C2GraphPrunerConfig:
        if self.covalent_core_radius_angstrom >= self.cutoff_radius_angstrom:
            raise ValueError(
                f"covalent_core_radius_angstrom ({self.covalent_core_radius_angstrom}) must be strictly less than cutoff_radius_angstrom ({self.cutoff_radius_angstrom})"
            )
        return self

